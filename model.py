"""
model.py
ReducedWongWangEIB 다이나믹스, EIBLinearCoupling 정의 및 네트워크 빌드.

Stage 3
-------
- build_network()는 data["graph"]가 있으면 그대로 사용한다.
- NOTE: tract delay는 cfg.use_delay 로 켜고 끈다(기본 True). True면 EIBDelayedCoupling
  (DelayedCoupling)이 data["delays"](=lengths/conduction_speed)를 히스토리 버퍼 조회에
  반영한다(DDE/SDDE). False면 coupling이 InstantaneousCoupling이라 DenseDelayGraph를
  넣어도 delays가 무시되고 delay-free ODE/SDE가 된다.
"""
import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import (
    DelayedCoupling,
    InstantaneousCoupling,
)
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from config import Config
from pipeline_contracts import make_bold_monitor


def _rww_H(x, d):
    """
    Reduced Wong-Wang 전달함수 H(x) = x / (1 - exp(-d·x)).

    x=0 에서 0/0 제거가능 특이점(참값 limit = 1/d)을 가진다. float32 에서 어떤
    노드의 입력이 x≈0 에 닿으면 nan 이 발생해 시뮬 전체를 오염시키므로,
    |d·x|<1e-4 구간은 해석적 극한 1/d 로 대체한다(double-where: nan 비전파).
    """
    dx = d * x
    near = jnp.abs(dx) < 1e-4
    safe_x = jnp.where(near, 1.0, x)               # bad-branch 입력을 0에서 떨어뜨림
    # float32 exp 오버플로 방지: x 가 강한 음수(포화 노드의 강한 억제)면 -d·safe_x 가 큰
    # 양수 → exp 가 inf → forward 는 유한(H→0)이지만 backward 가 inf 연산 → NaN → gradient
    # 전체가 NaN(→ zero_nans 로 0 → 최적화 정지). 큰 |x| 에서 H 는 이미 포화라 clip 이
    # forward 를 거의 안 바꾼다(x≫0→x, x≪0→0). exp(±80) 은 float32 안전범위.
    exp_arg = jnp.clip(-d * safe_x, -80.0, 80.0)
    H_full = safe_x / (1.0 - jnp.exp(exp_arg))
    return jnp.where(near, 1.0 / d, H_full)


class ReducedWongWangEIB(AbstractDynamics):
    """
    두 집단(E/I) Reduced Wong-Wang 모델 (EI_Tuning.ipynb local dynamics).

    상태 E/I 는 synaptic gating S_e/S_i 이다 (WC activity 와 인덱스 호환을 위해
    이름만 E/I 로 유지). 파이프라인 계약 보존:
      - state[0]=S_e, state[1]=S_i
      - aux[2]=rE_hz(=H_e), aux[3]=rI_hz(=H_i)  → FIC firing-rate 타깃과 호환
      - params.c_ei 는 RWW 의 J_i (S_i→E 억제 가중) 로 쓰인다. FIC 가 per-node 로 조정.
    """

    STATE_NAMES = ("E", "I")
    INITIAL_STATE = (0.001, 0.001)
    AUXILIARY_NAMES = ("S_e", "S_i", "rE_hz", "rI_hz")
    DEFAULT_PARAMS = Bunch(
        # Excitatory population
        a_e=310.0,            # input gain
        b_e=125.0,            # input shift [Hz]
        d_e=0.160,            # input scaling [s]
        gamma_e=0.641 / 1000, # kinetic
        tau_e=100.0,          # NMDA decay [ms]
        w_p=1.4,              # recurrent excitation
        W_e=1.0,              # external input scale
        # Inhibitory population
        a_i=615.0,
        b_i=177.0,
        d_i=0.087,
        gamma_i=1.0 / 1000,
        tau_i=10.0,
        W_i=0.7,
        # Synaptic weights
        J_N=0.15,             # NMDA current [nA]
        c_ei=1.0,             # == J_i (FIC per-node 조정 대상)
        # External inputs
        I_o=0.382,
        I_ext=0.0,
        # Coupling
        lamda=1.0,
        # WC 호환용 (파이프라인 fallback/part3 에서만 참조; aux 가 우선)
        rE_max_hz=20.0,
        rI_max_hz=20.0,
    )

    COUPLING_INPUTS = {"coupling": 2}

    def dynamics(self, time_ms, state, params, coupling, external):
        S_e = state[0]
        S_i = state[1]

        # coupling 출력에 J_N 스케일 (노트북과 동일)
        c_lre = params.J_N * coupling.coupling[0]  # long-range excitation
        c_ffi = params.J_N * coupling.coupling[1]  # feedforward inhibition

        J_N_S_e = params.J_N * S_e

        # Excitatory input  (c_ei == J_i)
        x_e_pre = (
            params.w_p * J_N_S_e
            - params.c_ei * S_i
            + params.W_e * params.I_o
            + c_lre
            + params.I_ext
        )
        x_e = params.a_e * x_e_pre - params.b_e
        H_e = _rww_H(x_e, params.d_e)

        dS_e_dt = -(S_e / params.tau_e) + (1.0 - S_e) * H_e * params.gamma_e

        # Inhibitory input
        x_i_pre = (
            J_N_S_e
            - S_i
            + params.W_i * params.I_o
            + params.lamda * c_ffi
        )
        x_i = params.a_i * x_i_pre - params.b_i
        H_i = _rww_H(x_i, params.d_i)

        dS_i_dt = -(S_i / params.tau_i) + H_i * params.gamma_i

        derivatives = jnp.array([dS_e_dt, dS_i_dt])
        # aux[2]=H_e, aux[3]=H_i → FIC 가 firing-rate(Hz) 타깃으로 사용
        auxiliary_outputs = jnp.array([S_e, S_i, H_e, H_i])
        return derivatives, auxiliary_outputs


# 하위호환: 기존 import (`from model import WilsonCowanEIB`) 유지.
# DEFAULT_PARAMS.rE_max_hz/rI_max_hz 참조도 그대로 동작.
WilsonCowanEIB = ReducedWongWangEIB


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


class EIBDelayedCoupling(DelayedCoupling):
    """EIBLinearCoupling 의 delay(DDE) 버전.

    pre() 가 받는 delayed_states[i] 는 [n_target, n_source] 2D 라서 wLRE/wFFI (n,n) 와
    elementwise 로 맞는다(instantaneous 는 [n_source] 브로드캐스트). 식은 동일.
    """

    N_OUTPUT_STATES = 2
    DEFAULT_PARAMS = Bunch(wLRE=1.0, wFFI=1.0)

    pre = EIBLinearCoupling.pre
    post = EIBLinearCoupling.post


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

    # RWW local dynamics (EI_Tuning.ipynb). 파라미터는 cfg.rww_* 에서 읽는다
    # (getattr 기본값 = DEFAULT_PARAMS 동일값 → cfg 에 없어도 동작 불변).
    # c_ei(=J_i) 만 per-node 로 두고 cfg.wc_c_ei_init 으로 초기화 (FIC 가 조정).
    _c_ei_init = float(getattr(cfg, "wc_c_ei_init", 1.0))
    _dp = ReducedWongWangEIB.DEFAULT_PARAMS
    _rww = {k: float(getattr(cfg, f"rww_{k}", _dp[k]))
            for k in ("a_e", "b_e", "d_e", "gamma_e", "tau_e", "w_p", "W_e",
                      "a_i", "b_i", "d_i", "gamma_i", "tau_i", "W_i",
                      "J_N", "I_o", "I_ext", "lamda", "rE_max_hz", "rI_max_hz")}
    dynamics = ReducedWongWangEIB(
        c_ei=_c_ei_init * jnp.ones((n_nodes,), dtype=jnp.float32),
        **_rww,
    )

    _use_delay = bool(getattr(cfg, "use_delay", False))
    _coupling_cls = EIBDelayedCoupling if _use_delay else EIBLinearCoupling
    coupling = _coupling_cls(incoming_states=["E"])
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

    # BOLD 모니터는 각 Part의 build_bold_monitor와 동일 커널을 써야 한다(make_bold_monitor
    # 단일 소스). 커널 duration이 어긋나면 warmup history 길이가 kernel_samples와 안 맞아
    # mode='valid' 합성곱이 조용히 깨진다(과거 기본 20s ↔ cfg 32s 불일치 버그).
    bold_monitor = make_bold_monitor(cfg, history=warmup_result)

    return network, initial_state, bold_monitor, warmup_result
