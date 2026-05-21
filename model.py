"""
model.py
WilsonCowanEIB 다이나믹스, EIBLinearCoupling 정의 및 네트워크 빌드.

Stage 3
-------
- build_network()는 data["graph"]가 있으면 그대로 사용한다.
- data_loader.py가 DenseDelayGraph를 제공하면 tract delay가 자동으로 네트워크에 반영된다.
"""
import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.tvb_monitors.bold import Bold

from config import Config


class WilsonCowanEIB(AbstractDynamics):
    """
    두 집단(E/I) Wilson-Cowan 모델.
    c_ei는 FIC가 per-node로 조정해 rE_hz → target을 유지한다.
    """

    STATE_NAMES = ("E", "I")
    INITIAL_STATE = (0.2, 0.1)
    AUXILIARY_NAMES = ("S_e", "S_i", "rE_hz", "rI_hz")

    DEFAULT_PARAMS = Bunch(
        tau_e=10.0, tau_i=10.0,
        c_ee=12.0, c_ei=10.0, c_ie=12.0, c_ii=3.0,
        a_e=1.0, a_i=1.0,
        b_e=0.0, b_i=0.0,
        c_e=1.0, c_i=1.0,
        alpha_e=1.2, alpha_i=2.0,
        theta_e=2.0, theta_i=3.5,
        k_e=1.0, k_i=1.0,
        r_e=0.0, r_i=0.0,
        P=0.5, Q=0.0, I_ext=0.0,
        lamda=1.0,
        rE_max_hz=20.0, rI_max_hz=20.0,
    )

    COUPLING_INPUTS = {"coupling": 2}

    def dynamics(self, time_ms, state, params, coupling, external):
        excitatory_activity = state[0]
        inhibitory_activity = state[1]
        long_range_excitation = coupling.coupling[0]
        feedforward_inhibition = coupling.coupling[1]

        excitatory_input = params.alpha_e * (
            params.c_ee * excitatory_activity
            - params.c_ei * inhibitory_activity
            + params.P
            + params.I_ext
            - params.theta_e
            + long_range_excitation
        )
        inhibitory_input = params.alpha_i * (
            params.c_ie * excitatory_activity
            - params.c_ii * inhibitory_activity
            + params.Q
            - params.theta_i
            + params.lamda * feedforward_inhibition
        )

        sigmoid_excitatory = params.c_e / (
            1.0 + jnp.exp(
                -jnp.clip(params.a_e * (excitatory_input - params.b_e), -500.0, 500.0)
            )
        )
        sigmoid_inhibitory = params.c_i / (
            1.0 + jnp.exp(
                -jnp.clip(params.a_i * (inhibitory_input - params.b_i), -500.0, 500.0)
            )
        )

        excitatory_firing_rate_hz = params.rE_max_hz * sigmoid_excitatory
        inhibitory_firing_rate_hz = params.rI_max_hz * sigmoid_inhibitory

        excitatory_derivative = (
            -excitatory_activity
            + (params.k_e - params.r_e * excitatory_activity) * sigmoid_excitatory
        ) / params.tau_e
        inhibitory_derivative = (
            -inhibitory_activity
            + (params.k_i - params.r_i * inhibitory_activity) * sigmoid_inhibitory
        ) / params.tau_i

        derivatives = jnp.array([excitatory_derivative, inhibitory_derivative])
        auxiliary_outputs = jnp.array([
            sigmoid_excitatory,
            sigmoid_inhibitory,
            excitatory_firing_rate_hz,
            inhibitory_firing_rate_hz,
        ])
        return derivatives, auxiliary_outputs


class EIBLinearCoupling(InstantaneousCoupling):
    """
    장거리 흥분(wLRE)과 전향 억제(wFFI)를 독립적으로 조정한다.
    """

    N_OUTPUT_STATES = 2
    DEFAULT_PARAMS = Bunch(wLRE=1.0, wFFI=1.0)

    def pre(self, incoming_states, local_states, params):
        source_excitation = incoming_states[0]
        return jnp.stack(
            [source_excitation * params.wLRE, source_excitation * params.wFFI],
            axis=0,
        )

    def post(self, summed_inputs, local_states, params):
        return summed_inputs


def build_network(cfg: Config, data: dict) -> tuple:
    """
    네트워크를 생성하고 초기 워밍업 시뮬레이션을 실행한다.

    Returns
    -------
    network, initial_state, bold_monitor, warmup_result
    """
    n_nodes = data["n_nodes"]

    graph = data.get("graph", None)
    if graph is None:
        graph = DenseGraph(data["weights"], region_labels=data["region_labels"])

    dynamics = WilsonCowanEIB(
        c_ei=6.0 * jnp.ones((n_nodes,), dtype=jnp.float32)
    )

    coupling = EIBLinearCoupling(incoming_states=["E"])
    coupling.params.wLRE = jnp.ones((n_nodes, n_nodes), dtype=jnp.float32)
    coupling.params.wFFI = jnp.ones((n_nodes, n_nodes), dtype=jnp.float32)

    noise = AdditiveNoise(sigma=cfg.additive_noise_sigma, apply_to="E")

    network = Network(
        dynamics=dynamics,
        coupling={"coupling": coupling},
        graph=graph,
        noise=noise,
    )

    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    compiled_model, initial_state = prepare(
        network, solver,
        t1=cfg.warmup_duration_ms,
        dt=cfg.integration_dt_ms,
    )

    print("[MODEL] Running warmup simulation...")
    warmup_result = jax.block_until_ready(compiled_model(initial_state))
    if hasattr(network, "update_history"):
        try:
            network.update_history(warmup_result)
        except Exception:
            pass

    final_e_mean = float(warmup_result.data[-1, 0, :].mean())
    final_i_mean = float(warmup_result.data[-1, 1, :].mean())
    print(f"[MODEL] Warmup done — E mean={final_e_mean:.4f}  I mean={final_i_mean:.4f}")
    print(f"[MODEL] Graph type: {type(graph).__name__}")

    bold_monitor = Bold(
        period=cfg.bold_repetition_time_ms,
        downsample_period=4.0,
        voi=0,
        history=warmup_result,
    )

    return network, initial_state, bold_monitor, warmup_result
