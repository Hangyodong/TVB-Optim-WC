"""
part4_dbs.py
Biphasic DBS 자극을 시뮬레이션하고 pre vs during PSD를 비교한다.

Stage 3
-------
- Phase 1 최종 결과를 StateBundle로 직접 소비한다.
- legacy plain TVB state도 계속 허용한다.
- 자극 주입 위치는 true_p_t 방식으로 고정한다.
- LFP observable은 E+I로 고정한다.
- Phase 1 richer bundle state를 최대한 이어받아 DBS를 시작한다.
"""
import os
import types

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch

from tvboptim.experimental.network_dynamics import prepare
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun

from config import Config
from pipeline_contracts import StateBundle, build_bundle_from_legacy_state


DEFAULT_STIMULUS_MODES = ("true_p_t",)


def run_dbs_stimulation(
    network,
    bundle_in: StateBundle = None,
    optimized_state=None,
    cfg: Config = None,
    data: dict = None,
) -> None:
    """
    각 자극 타겟에 대해 biphasic pulse train을 주입하고
    pre vs during LFP 및 PSD를 비교한다.
    """
    bundle_base = _coerce_dbs_bundle(bundle_in, optimized_state, cfg, data)
    bundle_base.apply_to_network(network)

    os.makedirs(cfg.dbs_output_base_dir, exist_ok=True)

    derived = _compute_derived_parameters(cfg)
    _print_stimulation_summary(cfg, derived)

    n_nodes = data["n_nodes"]
    solver_dbs = BoundedSolver(Heun(), low=0.0, high=1.0)
    original_dynamics = network.dynamics.dynamics

    stimulus_modes = ("true_p_t",)

    if getattr(cfg, 'dbs_parallel_targets', False):
        print(
            "[DBS] cfg.dbs_parallel_targets=True 감지: "
            "현재 patch는 Option α(JIT cache 재사용)만 지원하므로 sequential 경로로 fallback합니다. "
            "(진정한 batched DBS는 별도 patch에서 model 재설계 필요)"
        )
    # === Patch 4 Fix 2: stim 배열 미리 통합, JIT recompile 방지 ===
    _target_items = list(cfg.dbs_target_regions.items())
    _pre_steps = int(round(cfg.dbs_pre_stimulation_duration_ms / cfg.integration_dt_ms))
    _stim_steps = derived["n_pulses"] * derived["period_steps"]
    _total_steps = _pre_steps + _stim_steps

    # 모든 target의 stim 배열을 미리 생성 (Python 루프, 빠름)
    _all_stim_arrays = {}
    for _tl, _ti in _target_items:
        _arr, _ = _build_biphasic_pulse_train(
            target_node_index=_ti,
            n_nodes=n_nodes,
            onset_step=_pre_steps,
            amplitude=cfg.dbs_pulse_amplitude,
            phase_duration_steps=derived["phase_duration_steps"],
            n_pulses=derived["n_pulses"],
            total_steps=_total_steps,
            gap_steps=derived["gap_steps"],
        )
        _all_stim_arrays[_tl] = _arr

    # target별로 실행 (stim 배열은 재사용, _run_single_target에 직접 전달)
    for target_label, target_node_index in _target_items:
        _run_single_target(
            network=network,
            solver=solver_dbs,
            original_dynamics=original_dynamics,
            bundle_base=bundle_base,
            target_label=target_label,
            target_node_index=target_node_index,
            n_nodes=n_nodes,
            derived=derived,
            cfg=cfg,
            stimulus_modes=stimulus_modes,
            prebuilt_stim_array=_all_stim_arrays[target_label],
        )

    print(f"\n[DBS] All outputs saved to: {os.path.abspath(cfg.dbs_output_base_dir)}")


run_dbs = run_dbs_stimulation


# ── 입력 bundle 구성 ──────────────────────────────────────────

def _coerce_dbs_bundle(bundle_in, optimized_state, cfg: Config, data: dict) -> StateBundle:
    if isinstance(bundle_in, StateBundle):
        return bundle_in
    if isinstance(optimized_state, StateBundle):
        return optimized_state
    if optimized_state is None:
        raise ValueError("run_dbs_stimulation requires either bundle_in or optimized_state.")
    return build_bundle_from_legacy_state(
        state=optimized_state,
        cfg=cfg,
        data=data,
        stage="grad",
        bold_history=None,
        bold_window=None,
        internal_state=None,
        delay_history=None,
        c_ei_frozen=False,
        metadata={"rng_seed": int(getattr(cfg, "bundle_rng_seed", 42))},
    )


# ── 단일 타겟 시뮬레이션 ──────────────────────────────────────

def _run_single_target(
    network,
    solver,
    original_dynamics,
    bundle_base: StateBundle,
    target_label,
    target_node_index,
    n_nodes,
    derived,
    cfg,
    stimulus_modes,
    prebuilt_stim_array=None,  # Patch 4: 미리 빌드된 stim 배열 (None이면 내부에서 생성)
) -> None:
    print(f"\n{'='*60}\nTarget: {target_label} (node {target_node_index})\n{'='*60}")

    target_save_dir = os.path.join(cfg.dbs_output_base_dir, target_label)
    os.makedirs(target_save_dir, exist_ok=True)

    pre_steps = int(round(cfg.dbs_pre_stimulation_duration_ms / cfg.integration_dt_ms))
    stim_steps = derived["n_pulses"] * derived["period_steps"]  # gap 포함한 실제 총 길이
    total_steps = pre_steps + stim_steps
    total_duration_ms = total_steps * cfg.integration_dt_ms
    onset_step = pre_steps

    # Patch 4: 미리 빌드된 stim 배열이 있으면 재사용, 없으면 기존 방식으로 생성
    if prebuilt_stim_array is not None:
        stimulation_array = prebuilt_stim_array
        actual_pulse_count = derived["n_pulses"]
    else:
        stimulation_array, actual_pulse_count = _build_biphasic_pulse_train(
            target_node_index=target_node_index,
            n_nodes=n_nodes,
            onset_step=onset_step,
            amplitude=cfg.dbs_pulse_amplitude,
            phase_duration_steps=derived["phase_duration_steps"],
            n_pulses=derived["n_pulses"],
            total_steps=total_steps,
            gap_steps=derived["gap_steps"],
        )
    stimulation_jax = jnp.asarray(stimulation_array, dtype=jnp.float32)

    print(
        f"[INFO] actual_pulses={actual_pulse_count}  "
        f"total={total_duration_ms / 1000:.1f}s  gapless_biphasic=True"
    )

    for stimulus_mode in stimulus_modes:
        print(f"[DBS] stimulus_mode = {stimulus_mode}")
        mode_save_dir = os.path.join(target_save_dir, stimulus_mode)
        os.makedirs(mode_save_dir, exist_ok=True)

        stimulated_fn = _make_stimulated_dynamics(
            original_dynamics=original_dynamics,
            stimulation_jax=stimulation_jax,
            cfg=cfg,
            stimulus_mode=stimulus_mode,
        )
        network.dynamics.dynamics = types.MethodType(stimulated_fn, network.dynamics)

        try:
            bundle_base.apply_to_network(network)
            compiled_model, simulation_state = prepare(
                network, solver, t1=int(total_duration_ms), dt=cfg.integration_dt_ms
            )
            simulation_state = bundle_base.apply_to_state(simulation_state)
            simulation_result = jax.block_until_ready(compiled_model(simulation_state))
        finally:
            network.dynamics.dynamics = original_dynamics

        observable_name = "E_plus_I"
        observable_save_dir = os.path.join(mode_save_dir, observable_name)
        os.makedirs(observable_save_dir, exist_ok=True)
        _analyze_and_plot(
            simulation_result=simulation_result,
            stimulation_array=stimulation_array,
            target_label=target_label,
            target_node_index=target_node_index,
            onset_step=onset_step,
            stim_steps=stim_steps,
            derived=derived,
            cfg=cfg,
            save_dir=observable_save_dir,
            stimulus_mode=stimulus_mode,
            observable_name=observable_name,
        )


# ── 자극 dynamics 생성 ───────────────────────────────────────

def _make_stimulated_dynamics(original_dynamics, stimulation_jax, cfg, stimulus_mode: str):
    def stimulated_dynamics(self, time_ms, state, params, coupling, external):
        dtype = state.dtype
        excitatory_activity = state[0]
        inhibitory_activity = state[1]
        long_range_excitation = coupling.coupling[0]
        feedforward_inhibition = coupling.coupling[1]

        time_step_index = jnp.clip(
            jnp.round(time_ms / cfg.integration_dt_ms).astype(jnp.int32),
            0, stimulation_jax.shape[0] - 1,
        )
        stimulus_vector = jnp.asarray(stimulation_jax[time_step_index], dtype=dtype)

        alpha_e = jnp.asarray(params.alpha_e, dtype=dtype)
        alpha_i = jnp.asarray(params.alpha_i, dtype=dtype)
        c_ee = jnp.asarray(params.c_ee, dtype=dtype)
        c_ei = jnp.asarray(params.c_ei, dtype=dtype)
        c_ie = jnp.asarray(params.c_ie, dtype=dtype)
        c_ii = jnp.asarray(params.c_ii, dtype=dtype)
        P = jnp.asarray(params.P, dtype=dtype)
        Q = jnp.asarray(params.Q, dtype=dtype)
        I_ext = jnp.asarray(params.I_ext, dtype=dtype)
        theta_e = jnp.asarray(params.theta_e, dtype=dtype)
        theta_i = jnp.asarray(params.theta_i, dtype=dtype)
        lamda = jnp.asarray(params.lamda, dtype=dtype)
        a_e = jnp.asarray(params.a_e, dtype=dtype)
        a_i = jnp.asarray(params.a_i, dtype=dtype)
        b_e = jnp.asarray(params.b_e, dtype=dtype)
        b_i = jnp.asarray(params.b_i, dtype=dtype)
        c_e = jnp.asarray(params.c_e, dtype=dtype)
        c_i = jnp.asarray(params.c_i, dtype=dtype)
        k_e = jnp.asarray(params.k_e, dtype=dtype)
        k_i = jnp.asarray(params.k_i, dtype=dtype)
        r_e = jnp.asarray(params.r_e, dtype=dtype)
        r_i = jnp.asarray(params.r_i, dtype=dtype)
        tau_e = jnp.asarray(params.tau_e, dtype=dtype)
        tau_i = jnp.asarray(params.tau_i, dtype=dtype)
        rE_max_hz = jnp.asarray(params.rE_max_hz, dtype=dtype)
        rI_max_hz = jnp.asarray(params.rI_max_hz, dtype=dtype)
        clip_lo = jnp.asarray(-500.0, dtype=dtype)
        clip_hi = jnp.asarray(500.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)

        if stimulus_mode == "true_p_t":
            inside_stim = stimulus_vector
            outside_stim = jnp.zeros_like(stimulus_vector)
        elif stimulus_mode == "tvb_default":
            inside_stim = jnp.zeros_like(stimulus_vector)
            outside_stim = stimulus_vector
        elif stimulus_mode == "hybrid":
            inside_stim = half * stimulus_vector
            outside_stim = half * stimulus_vector
        else:
            raise ValueError(f"Unsupported stimulus_mode: {stimulus_mode}")

        excitatory_input = alpha_e * (
            c_ee * excitatory_activity
            - c_ei * inhibitory_activity
            + P
            + I_ext
            + inside_stim
            - theta_e
            + long_range_excitation
        )
        inhibitory_input = alpha_i * (
            c_ie * excitatory_activity
            - c_ii * inhibitory_activity
            + Q
            - theta_i
            + lamda * feedforward_inhibition
        )

        sigmoid_excitatory = c_e / (
            jnp.asarray(1.0, dtype=dtype) + jnp.exp(-jnp.clip(a_e * (excitatory_input - b_e), clip_lo, clip_hi))
        )
        sigmoid_inhibitory = c_i / (
            jnp.asarray(1.0, dtype=dtype) + jnp.exp(-jnp.clip(a_i * (inhibitory_input - b_i), clip_lo, clip_hi))
        )

        excitatory_derivative = (
            -excitatory_activity
            + (k_e - r_e * excitatory_activity) * sigmoid_excitatory
        ) / tau_e
        excitatory_derivative = excitatory_derivative + outside_stim

        inhibitory_derivative = (
            -inhibitory_activity
            + (k_i - r_i * inhibitory_activity) * sigmoid_inhibitory
        ) / tau_i

        return (
            jnp.stack([excitatory_derivative, inhibitory_derivative], axis=0),
            jnp.stack([
                sigmoid_excitatory,
                sigmoid_inhibitory,
                rE_max_hz * sigmoid_excitatory,
                rI_max_hz * sigmoid_inhibitory,
            ], axis=0),
        )

    return stimulated_dynamics


# ── 분석 및 시각화 ────────────────────────────────────────────

def _analyze_and_plot(
    simulation_result,
    stimulation_array,
    target_label,
    target_node_index,
    onset_step,
    stim_steps,
    derived,
    cfg,
    save_dir,
    stimulus_mode,
    observable_name,
) -> None:
    neural_data = np.asarray(simulation_result.data, dtype=np.float32)
    time_axis_ms = np.arange(neural_data.shape[0], dtype=np.float32) * cfg.integration_dt_ms
    stim_end_ms = (onset_step + stim_steps) * cfg.integration_dt_ms

    lfp_signal = _resolve_observable_signal(neural_data, target_node_index, observable_name)

    pre_mask = time_axis_ms < onset_step * cfg.integration_dt_ms
    during_mask = (
        (time_axis_ms >= onset_step * cfg.integration_dt_ms)
        & (time_axis_ms < stim_end_ms)
    )

    lfp_pre = lfp_signal[pre_mask]
    lfp_during = lfp_signal[during_mask]

    fs_hz = max(1000.0 / cfg.integration_dt_ms, 1000.0)
    freq_pre, psd_pre = _compute_psd(lfp_pre, fs_hz, cfg.dbs_psd_max_frequency_hz)
    freq_during, psd_during = _compute_psd(lfp_during, fs_hz, cfg.dbs_psd_max_frequency_hz)

    beta_ratio_pre = _compute_beta_power_ratio(
        freq_pre, psd_pre, cfg.dbs_beta_band_low_hz, cfg.dbs_beta_band_high_hz
    )
    beta_ratio_during = _compute_beta_power_ratio(
        freq_during, psd_during, cfg.dbs_beta_band_low_hz, cfg.dbs_beta_band_high_hz
    )

    print(
        f"[INFO] {stimulus_mode} / {observable_name}  "
        f"mean pre={np.mean(lfp_pre):.6f}  during={np.mean(lfp_during):.6f}  "
        f"beta pre={beta_ratio_pre:.4f}  during={beta_ratio_during:.4f}"
    )

    _plot_lfp_timeseries(
        time_axis_ms, lfp_signal, onset_step, stim_end_ms,
        target_label, derived, cfg, save_dir,
        stimulus_mode, observable_name,
    )
    _plot_stim_waveform_full(
        time_axis_ms, stimulation_array, target_node_index, target_label,
        derived, save_dir, stimulus_mode,
    )
    _plot_stim_waveform_zoomed(
        time_axis_ms, stimulation_array, target_node_index, target_label,
        onset_step, derived, cfg, save_dir, stimulus_mode,
    )
    _plot_psd_comparison(
        freq_pre, psd_pre, freq_during, psd_during,
        beta_ratio_pre, beta_ratio_during,
        target_label, cfg, save_dir,
        stimulus_mode, observable_name,
    )
    _plot_lfp_segment_comparison(
        lfp_pre, lfp_during, target_label, cfg, save_dir,
        stimulus_mode, observable_name,
    )

    pd.DataFrame({
        "time_ms": time_axis_ms,
        "lfp_e_plus_i": lfp_signal,
        "stimulus": stimulation_array[:, target_node_index],
    }).to_csv(os.path.join(save_dir, "lfp_timeseries.csv"), index=False)

    pd.DataFrame({
        "frequency_hz": freq_pre,
        "psd_pre_v2_per_hz": psd_pre,
        "psd_during_v2_per_hz": psd_during,
    }).to_csv(os.path.join(save_dir, "psd_pre_vs_during.csv"), index=False)


# ── observable ───────────────────────────────────────────────

def _resolve_observable_signal(neural_data: np.ndarray, target_node_index: int, observable_name: str) -> np.ndarray:
    E = neural_data[:, 0, target_node_index]
    I = neural_data[:, 1, target_node_index]
    if observable_name == "E":
        return E
    if observable_name == "I":
        return I
    if observable_name == "E_plus_I":
        return E + I
    if observable_name == "E_minus_I":
        return E - I
    raise ValueError(f"Unsupported observable_name: {observable_name}")


# ── 플롯 함수 ─────────────────────────────────────────────────

def _plot_lfp_timeseries(time_axis_ms, lfp_signal, onset_step, stim_end_ms, target_label, derived, cfg, save_dir, stimulus_mode, observable_name):
    fig, axis = plt.subplots(figsize=(14, 3.2))
    axis.plot(time_axis_ms / 1000.0, lfp_signal, color="black", linewidth=0.4)
    axis.axvspan(
        onset_step * cfg.integration_dt_ms / 1000.0,
        stim_end_ms / 1000.0,
        color="tab:green", alpha=0.20,
        label=f"During stim ({cfg.dbs_stimulation_duration_ms / 1000:.0f}s)",
    )
    axis.set_title(
        f"{target_label} | {stimulus_mode} | LFP(E+I)  "
        f"[{derived['n_pulses']} pulses @ {derived['freq_actual_hz']:.0f} Hz]"
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("LFP (E+I)")
    axis.legend(loc="upper right")
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot1_lfp_timeseries.png"), dpi=150)
    plt.show()


def _plot_stim_waveform_full(time_axis_ms, stimulation_array, target_node_index, target_label, derived, save_dir, stimulus_mode):
    fig, axis = plt.subplots(figsize=(14, 2.5))
    axis.plot(time_axis_ms / 1000.0, stimulation_array[:, target_node_index], color="tab:green", linewidth=0.5)
    axis.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axis.set_title(
        f"{target_label} | {stimulus_mode} | Stimulus waveform (full)  "
        f"[{derived['n_pulses']} gapless biphasic pulses]"
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot2_stim_waveform_full.png"), dpi=150)
    plt.show()


def _plot_stim_waveform_zoomed(
    time_axis_ms,
    stimulation_array,
    target_node_index,
    target_label,
    onset_step,
    derived,
    cfg,
    save_dir,
    stimulus_mode,
    zoom_pulse_count: int = 10,
):
    zoom_steps = zoom_pulse_count * derived["biphasic_steps"]
    zoom_start_ms = onset_step * cfg.integration_dt_ms - 2.0
    zoom_end_ms = (onset_step + zoom_steps) * cfg.integration_dt_ms + 2.0
    zoom_mask = (time_axis_ms >= zoom_start_ms) & (time_axis_ms <= zoom_end_ms)

    zoomed_time = time_axis_ms[zoom_mask]
    zoomed_stim = stimulation_array[:, target_node_index][zoom_mask]

    fig, axis = plt.subplots(figsize=(12, 2.8))
    axis.step(zoomed_time / 1000.0, zoomed_stim, where="post", color="tab:green", linewidth=1.5)
    axis.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axis.fill_between(
        zoomed_time / 1000.0, zoomed_stim, 0,
        where=(zoomed_stim > 0), color="tomato", alpha=0.4,
        label=f"Anodic (+{cfg.dbs_pulse_amplitude})",
    )
    axis.fill_between(
        zoomed_time / 1000.0, zoomed_stim, 0,
        where=(zoomed_stim < 0), color="steelblue", alpha=0.4,
        label=f"Cathodic (−{cfg.dbs_pulse_amplitude})",
    )
    axis.set_title(
        f"{target_label} | {stimulus_mode} | Biphasic detail  "
        f"(first {zoom_pulse_count} pulses, gapless @ {derived['freq_actual_hz']:.0f} Hz)"
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.legend(loc="upper right")
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot3_stim_waveform_zoom.png"), dpi=150)
    plt.show()


def _plot_psd_comparison(
    freq_pre,
    psd_pre,
    freq_during,
    psd_during,
    beta_ratio_pre,
    beta_ratio_during,
    target_label,
    cfg,
    save_dir,
    stimulus_mode,
    observable_name,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, xmax in zip(axes, [100.0, 50.0]):
        axis.semilogy(freq_pre,    psd_pre,    color="steelblue", linewidth=1.2,
                      label=f"Pre (β={beta_ratio_pre:.3f})")
        axis.semilogy(freq_during, psd_during, color="tomato",    linewidth=1.2,
                      label=f"During (β={beta_ratio_during:.3f})")
        axis.axvspan(
            cfg.dbs_beta_band_low_hz, cfg.dbs_beta_band_high_hz,
            alpha=0.12, color="orange",
            label=f"Beta ({cfg.dbs_beta_band_low_hz:.0f}–{cfg.dbs_beta_band_high_hz:.0f} Hz)",
        )
        axis.set_xlim(0, xmax)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power [V²]  (log scale)")
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(fontsize=8)
    axes[0].set_title(f"{target_label} | {stimulus_mode} | LFP(E+I) | PSD 0–100 Hz")
    axes[1].set_title(f"{target_label} | {stimulus_mode} | LFP(E+I) | PSD 0–50 Hz")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot4_psd_pre_vs_during.png"), dpi=150)
    plt.show()


def _plot_lfp_segment_comparison(lfp_pre, lfp_during, target_label, cfg, save_dir, stimulus_mode, observable_name):
    fs_hz = 1000.0 / cfg.integration_dt_ms
    segment_samples = int(cfg.dbs_waveform_segment_seconds * fs_hz)

    pre_segment = lfp_pre[-min(segment_samples, len(lfp_pre)):]
    during_segment = lfp_during[:min(segment_samples, len(lfp_during))]
    time_pre = np.arange(len(pre_segment)) / fs_hz
    time_during = np.arange(len(during_segment)) / fs_hz

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
    axes[0].plot(time_pre, pre_segment, color="steelblue", linewidth=0.7)
    axes[0].set_title(f"{target_label} | {stimulus_mode} | {observable_name} | Pre-stim (last {cfg.dbs_waveform_segment_seconds:.0f}s)")
    axes[0].set_ylabel(observable_name)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_during, during_segment, color="tomato", linewidth=0.7)
    axes[1].set_title(f"{target_label} | {stimulus_mode} | {observable_name} | During-stim (first {cfg.dbs_waveform_segment_seconds:.0f}s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(observable_name)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot5_lfp_pre_vs_during.png"), dpi=150)
    plt.show()


# ── 헬퍼 ──────────────────────────────────────────────────────

def _compute_derived_parameters(cfg: Config) -> dict:
    dt_ms = cfg.integration_dt_ms

    # phase width = 1ms 고정
    phase_duration_steps = 1
    phase_ms             = phase_duration_steps * dt_ms
    biphasic_steps       = phase_duration_steps * 2   # 2 steps = 2ms
    biphasic_ms          = biphasic_steps * dt_ms

    # period: cfg.dbs_stimulation_frequency_hz 로부터 역산
    period_steps   = max(biphasic_steps + 1,
                         int(round(1000.0 / (cfg.dbs_stimulation_frequency_hz * dt_ms))))
    period_ms      = period_steps * dt_ms
    freq_actual_hz = 1000.0 / period_ms

    # gap = period - biphasic (항상 >= 1)
    gap_steps = period_steps - biphasic_steps
    gap_ms    = gap_steps * dt_ms

    n_pulses = int(cfg.dbs_stimulation_duration_ms / period_ms)

    return {
        "phase_duration_steps": phase_duration_steps,
        "period_steps":         period_steps,
        "period_ms":            period_ms,
        "gap_steps":            gap_steps,
        "gap_ms":               gap_ms,
        "biphasic_steps":       biphasic_steps,
        "biphasic_ms":          biphasic_ms,
        "n_pulses":             n_pulses,
        "phase_ms":             phase_ms,
        "freq_actual_hz":       freq_actual_hz,
    }


def _print_stimulation_summary(cfg: Config, derived: dict) -> None:
    print("=" * 60)
    print("DBS stimulation setup")
    print("=" * 60)
    print(f"  stimulus_mode  = true_p_t")
    print(f"  actual_freq    = {derived['freq_actual_hz']:.2f} Hz  (gapless biphasic, phase_step=1)")
    print(
        f"  n_pulses       = {derived['n_pulses']}"
        f"  (auto from {cfg.dbs_stimulation_duration_ms / 1000:.0f}s, no inter-pulse gap)"
    )
    print(
        f"  biphasic       = +{cfg.dbs_pulse_amplitude} ({derived['phase_ms']:.0f}ms)"
        f" → -{cfg.dbs_pulse_amplitude} ({derived['phase_ms']:.0f}ms)"
    )
    print(f"  pre_dur        = {cfg.dbs_pre_stimulation_duration_ms / 1000:.0f}s")
    print(f"  stim_dur       = {cfg.dbs_stimulation_duration_ms / 1000:.0f}s")
    print("=" * 60)


def _build_biphasic_pulse_train(
    target_node_index: int,
    n_nodes: int,
    onset_step: int,
    amplitude: float,
    phase_duration_steps: int,
    n_pulses: int,
    total_steps: int,
    gap_steps: int = 0,
) -> tuple:
    """
    Gap 포함 biphasic pulse train.

    각 펄스 구조 (phase=1ms, 125Hz 예시):
        |+|-|  gap 6ms  |+|-|  gap 6ms  |
         1ms 1ms          1ms 1ms
        |←── period 8ms ──→|
    """
    stimulation_array  = np.zeros((total_steps, n_nodes), dtype=np.float32)
    actual_pulse_count = 0
    period_steps       = phase_duration_steps * 2 + gap_steps  # 펄스 1개 전체 길이

    for pulse_index in range(n_pulses):
        anodic_start   = onset_step + pulse_index * period_steps
        cathodic_start = anodic_start + phase_duration_steps
        anodic_end     = min(anodic_start   + phase_duration_steps, total_steps)
        cathodic_end   = min(cathodic_start + phase_duration_steps, total_steps)

        if anodic_start >= total_steps or cathodic_start >= total_steps:
            break

        stimulation_array[anodic_start:anodic_end,     target_node_index] = +amplitude
        stimulation_array[cathodic_start:cathodic_end, target_node_index] = -amplitude
        actual_pulse_count += 1

    return stimulation_array, actual_pulse_count


def _compute_psd(signal: np.ndarray, fs_hz: float, f_max_hz: float) -> tuple:
    """
    Welch PSD 계산.
    - window  : hann
    - nperseg : fs (1초 분량, fs >= 1000 보장)
    - noverlap: nperseg // 2  (50%)
    - nfft    : nperseg
    - scaling : 'spectrum'
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 4:
        return np.asarray([0.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)

    # fs >= 1000Hz 보장
    fs_hz  = max(fs_hz, 1000.0)
    nperseg = int(fs_hz)                         # 1초 분량 = fs samples
    nperseg = min(nperseg, signal.size)          # 신호보다 길 수 없음
    nperseg = max(nperseg, 4)
    noverlap = nperseg // 2                      # 50%
    nfft     = nperseg

    frequencies, power_spectrum = welch(
        signal,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="spectrum",                      # V^2 (density 아님)
    )
    frequency_mask = frequencies <= f_max_hz
    return (
        frequencies[frequency_mask].astype(np.float32),
        power_spectrum[frequency_mask].astype(np.float32),
    )


def _compute_beta_power_ratio(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    beta_low_hz: float,
    beta_high_hz: float,
    total_band_low_hz: float = 1.0,
) -> float:
    beta_mask = (frequencies >= beta_low_hz) & (frequencies <= beta_high_hz)
    total_mask = frequencies >= total_band_low_hz

    if not np.any(beta_mask) or not np.any(total_mask):
        return np.nan

    beta_power = trapezoid(power_spectrum[beta_mask], frequencies[beta_mask])
    total_power = trapezoid(power_spectrum[total_mask], frequencies[total_mask])
    return beta_power / max(total_power, 1e-12)