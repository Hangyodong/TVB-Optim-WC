"""
part1_fic.py

Stage 3
-------
- StateBundle 계약으로 FIC 입력/출력을 통일한다.
- FIC 종료 후 c_ei를 freeze_c_ei()로 동결한다.
- final bundle에는 init_dynamics / bold_history / bold_window / internal_state /
  delay_history를 함께 저장한다.
- legacy notebook의 old-style 호출도 계속 허용한다.
"""
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr
from tvboptim.utils import cache

from config import Config
from model import WilsonCowanEIB
from pipeline_contracts import (
    ParamSet,
    StateBundle,
    advance_internal_state,
    build_bundle_from_legacy_state,
    capture_internal_state,
    capture_network_delay_history,
    sync_network_delay_history,
    update_bold_history,
)


def run_fic(
    network,
    bundle_in: StateBundle = None,
    initial_state=None,
    bold_monitor=None,
    warmup_result=None,
    cfg: Config = None,
    data: dict = None,
) -> StateBundle:
    """
    FIC 루프를 실행하고 c_ei가 아직 frozen 되지 않은 StateBundle을 반환한다.

    허용 호출 방식
    -------------
    1) run_fic(network, bundle_in=StateBundle, cfg=cfg, data=data)
    2) run_fic(network, initial_state=..., bold_monitor=..., warmup_result=..., cfg=cfg, data=data)
    """
    bundle_init = _coerce_fic_bundle(
        network=network,
        bundle_in=bundle_in,
        initial_state=initial_state,
        bold_monitor=bold_monitor,
        warmup_result=warmup_result,
        cfg=cfg,
        data=data,
    )

    cache_name = (
        f"fic_{data['cache_tag']}"
        f"_rE{cfg.fic_target_firing_rate_hz:g}"
        f"_eta{str(cfg.fic_learning_rate).replace('.', 'p')}"
        f"_steps{cfg.fic_max_iterations}"
        f"_dur{cfg.fic_step_duration_ms}"
        f"_skip{cfg.fic_step_skip_tr}"
        f"_fp{bundle_init.fingerprint()}"
    )

    @cache(cache_name, redo=False)
    def _cached_run():
        return _run_fic_loop_pure(network, bundle_init.to_dict(), cfg, data)

    result = _cached_run()

    bundle_fic = StateBundle.from_dict(result["bundle"])
    bundle_fic.apply_to_network(network)

    print(
        f"[FIC] c_ei updated.  "
        f"mean c_ei = {bundle_fic.params.c_ei.mean():.4f}  "
        f"final rE_hz = {result['final_rE_hz']:.3f} Hz"
    )

    _plot_fic_results(result, cfg, data)
    return bundle_fic


# ── FIC 실행 ──────────────────────────────────────────────────

def _run_fic_loop_pure(
    network,
    init_dict: dict,
    cfg: Config,
    data: dict,
) -> dict:
    bundle_in = StateBundle.from_dict(init_dict)
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)

    step_model, tuned_state = bundle_in.to_tvb_state(
        network,
        solver,
        t1=cfg.fic_step_duration_ms,
        dt=cfg.integration_dt_ms,
    )
    tuned_bold_monitor = bundle_in.build_bold_monitor(cfg)

    metadata = bundle_in.metadata
    metadata.setdefault("rng_seed", int(cfg.bundle_rng_seed))

    rE_max = float(WilsonCowanEIB.DEFAULT_PARAMS.rE_max_hz)
    rI_max = float(WilsonCowanEIB.DEFAULT_PARAMS.rI_max_hz)
    skip_tr = cfg.fic_step_skip_tr
    bold_tr_ms = cfg.bold_repetition_time_ms

    pre_fic_result = step_model(tuned_state)

    bold_signal_buffer = []
    mean_E_history = []
    mean_firing_rate_history = []
    consecutive_convergence_count = 0
    start_time = time.time()

    total_tr = int(cfg.fic_step_duration_ms / bold_tr_ms)
    use_tr = total_tr - skip_tr
    print_every = 25

    print(
        f"[FIC] target={cfg.fic_target_firing_rate_hz} Hz  "
        f"eta={cfg.fic_learning_rate}  max_steps={cfg.fic_max_iterations}"
    )
    print(
        f"  step={cfg.fic_step_duration_ms/1000:.0f}s  "
        f"skip={skip_tr} TR  use={use_tr} TR"
    )

    for step_index in range(cfg.fic_max_iterations):
        step_result = step_model(tuned_state)
        bold_output = tuned_bold_monitor(step_result)

        bold_ys = bold_output.ys
        bold_used = bold_ys[skip_tr:]
        bold_signal_buffer.append(np.asarray(bold_used[:, 0, :], dtype=np.float32))

        mean_excitatory_rate, mean_inhibitory_rate = _extract_firing_rates_from_bold(
            step_result, skip_tr, rE_max, rI_max, bold_tr_ms, cfg.integration_dt_ms
        )
        current_mean_rate = float(jnp.mean(mean_excitatory_rate))
        current_mean_E = float(jnp.mean(step_result.data[:, 0, :]))

        mean_firing_rate_history.append(current_mean_rate)
        mean_E_history.append(current_mean_E)

        tuned_state.initial_state.dynamics = step_result.data[-1]
        tuned_bold_monitor = update_bold_history(tuned_bold_monitor, step_result)
        internal_state, metadata = advance_internal_state(tuned_state, metadata)

        rate_error = mean_excitatory_rate - cfg.fic_target_firing_rate_hz
        update_delta = cfg.fic_learning_rate * mean_inhibitory_rate * rate_error
        tuned_state.dynamics.c_ei = jnp.clip(
            tuned_state.dynamics.c_ei + update_delta, 0.0, 20.0
        )

        absolute_error_hz = abs(current_mean_rate - cfg.fic_target_firing_rate_hz)
        consecutive_convergence_count = (
            consecutive_convergence_count + 1
            if absolute_error_hz < cfg.fic_early_stop_tolerance_hz
            else 0
        )

        if (step_index + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"  step {step_index+1:>4}/{cfg.fic_max_iterations}"
                f"  mean_E={current_mean_E:.4f}"
                f"  rE={current_mean_rate:.3f} Hz"
                f"  err={absolute_error_hz:.3f}"
                f"  ({elapsed:.1f}s)"
            )

        if consecutive_convergence_count >= cfg.fic_early_stop_patience:
            print(f"[FIC] Early stop at step {step_index + 1}")
            break

    post_fic_result = step_model(tuned_state)
    post_bold_output = tuned_bold_monitor(post_fic_result)

    tuned_state.initial_state.dynamics = post_fic_result.data[-1]
    tuned_bold_monitor = update_bold_history(tuned_bold_monitor, post_fic_result)
    internal_state, metadata = advance_internal_state(tuned_state, metadata)
    delay_history = sync_network_delay_history(network, post_fic_result)

    bold_signal_arr = (
        np.concatenate(bold_signal_buffer, axis=0).astype(np.float32)
        if bold_signal_buffer else
        np.zeros((1, data["n_nodes"]), dtype=np.float32)
    )
    mean_E_arr = np.asarray(mean_E_history, dtype=np.float32)
    mean_rE_hz_arr = np.asarray(mean_firing_rate_history, dtype=np.float32)

    final_rate = mean_firing_rate_history[-1] if mean_firing_rate_history else float("nan")
    print(
        f"[FIC] Done — final mean_E={mean_E_history[-1]:.4f}  "
        f"final rE_hz={final_rate:.3f} Hz  "
        f"err={abs(final_rate - cfg.fic_target_firing_rate_hz):.3f} Hz"
    )

    fic_params = ParamSet(
        c_ei=np.asarray(tuned_state.dynamics.c_ei, dtype=np.float32),
        wLRE=np.asarray(bundle_in.params.wLRE, dtype=np.float32),
        wFFI=np.asarray(bundle_in.params.wFFI, dtype=np.float32),
        c_ei_frozen=False,
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)

    bundle_fic = bundle_in.advance(
        new_params=fic_params,
        new_init_dynamics=np.asarray(post_fic_result.data[-1], dtype=np.float32),
        new_bold_history=np.asarray(tuned_bold_monitor.history, dtype=np.float32),
        new_bold_window=np.asarray(bold_signal_arr, dtype=np.float32),
        new_internal_state=internal_state,
        new_delay_history=delay_history,
        next_stage="fic",
        metadata_update=metadata,
    )

    pre_fic_fc, pre_fic_corr, pre_fic_rmse = _compute_bundle_fc_summary(
        network, bundle_in, cfg, data
    )
    post_fic_fc, post_fic_corr, post_fic_rmse = _compute_bundle_fc_summary(
        network, bundle_fic, cfg, data
    )
    bundle_fic = bundle_fic.with_metadata({
        "post_fic_fc_matrix": np.asarray(post_fic_fc, dtype=np.float32),
        "post_fic_fc_corr": float(post_fic_corr),
        "post_fic_fc_rmse": float(post_fic_rmse),
    })

    return {
        "bundle": bundle_fic.to_dict(),
        "bold_signal": bold_signal_arr,
        "final_rE_hz": float(final_rate),
        "mean_rE_hz_history": mean_rE_hz_arr,
        "mean_E_history": mean_E_arr,
        "pre_fic_neural": np.asarray(pre_fic_result.data, dtype=np.float32),
        "post_fic_neural": np.asarray(post_fic_result.data, dtype=np.float32),
        "pre_fic_fc": np.asarray(pre_fic_fc, dtype=np.float32),
        "pre_fic_fc_corr": float(pre_fic_corr),
        "pre_fic_fc_rmse": float(pre_fic_rmse),
        "post_fic_fc": np.asarray(post_fic_fc, dtype=np.float32),
        "post_fic_fc_corr": float(post_fic_corr),
        "post_fic_fc_rmse": float(post_fic_rmse),
    }


# ── 입력 bundle 구성 ──────────────────────────────────────────

def _coerce_fic_bundle(
    network,
    bundle_in: StateBundle,
    initial_state,
    bold_monitor,
    warmup_result,
    cfg: Config,
    data: dict,
) -> StateBundle:
    if isinstance(bundle_in, StateBundle):
        return bundle_in

    if initial_state is None or bold_monitor is None or warmup_result is None:
        raise ValueError("run_fic requires either bundle_in or (initial_state, bold_monitor, warmup_result).")

    initial_params = ParamSet.default(data["n_nodes"], c_ei_init=10.0).sanitize(
        data["sc_mask"], cfg.connectivity_weight_max
    )

    internal_state = capture_internal_state(initial_state)
    delay_history = capture_network_delay_history(network)
    metadata = {
        "rng_seed": int(cfg.bundle_rng_seed),
    }
    return StateBundle.from_warmup(
        warmup_result=warmup_result,
        bold_monitor_template=bold_monitor,
        initial_params=initial_params,
        internal_state=internal_state,
        delay_history=delay_history,
        stage="warmup",
        metadata=metadata,
    )


# ── firing rate helpers ───────────────────────────────────────

def _extract_firing_rates_from_bold(
    step_result,
    skip_tr: int,
    rE_max: float,
    rI_max: float,
    bold_tr_ms: float,
    dt_ms: float,
) -> tuple:
    skip_steps = int(skip_tr * bold_tr_ms / dt_ms)
    total_steps = step_result.data.shape[0]
    start = min(skip_steps, total_steps - 1)
    data_used = step_result.data[start:]

    has_auxiliary = (
        hasattr(step_result, "auxiliary")
        and step_result.auxiliary is not None
    )
    if has_auxiliary:
        aux_used = step_result.auxiliary[start:]
        return (
            jnp.mean(aux_used[:, 2, :], axis=0),
            jnp.mean(aux_used[:, 3, :], axis=0),
        )
    return (
        rE_max * jnp.mean(data_used[:, 0, :], axis=0),
        rI_max * jnp.mean(data_used[:, 1, :], axis=0),
    )


def _extract_firing_rates(step_result, rE_max: float, rI_max: float) -> tuple:
    has_auxiliary = (
        hasattr(step_result, "auxiliary")
        and step_result.auxiliary is not None
    )
    if has_auxiliary:
        return (
            jnp.mean(step_result.auxiliary[:, 2, :], axis=0),
            jnp.mean(step_result.auxiliary[:, 3, :], axis=0),
        )
    return (
        rE_max * jnp.mean(step_result.data[:, 0, :], axis=0),
        rI_max * jnp.mean(step_result.data[:, 1, :], axis=0),
    )



def _compute_fc_from_bold_output(bold_output, skip_tr: int, eps: float = 1e-6) -> np.ndarray:
    ts = np.asarray(
        bold_output.ys[:, 0, :] if bold_output.ys.ndim == 3 else bold_output.ys,
        dtype=np.float32,
    )
    ts = ts[int(skip_tr):]
    ts = np.nan_to_num(ts)
    ts = ts - ts.mean(axis=0, keepdims=True)
    std = np.maximum(ts.std(axis=0, keepdims=True), eps)
    ts_norm = ts / std
    fc = (ts_norm.T @ ts_norm) / max(ts_norm.shape[0] - 1, 1)
    fc = np.clip(fc, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(fc, 0.0)
    return fc


def _compute_bundle_fc_summary(
    network,
    bundle: StateBundle,
    cfg: Config,
    data: dict,
    sim_duration_ms: int = None,
    skip_tr: int = None,
) -> tuple:
    if sim_duration_ms is None:
        sim_duration_ms = int(cfg.eib_posthoc_duration_ms)
    if skip_tr is None:
        skip_tr = int(cfg.eib_posthoc_skip_tr)

    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    sim_model, sim_state = bundle.to_tvb_state(
        network, solver, t1=sim_duration_ms, dt=cfg.integration_dt_ms
    )
    bold_monitor = bundle.build_bold_monitor(cfg)
    sim_result = sim_model(sim_state)
    bold_output = bold_monitor(sim_result)
    fc_matrix = _compute_fc_from_bold_output(bold_output, skip_tr)
    fc_target = np.asarray(data["fc_target"], dtype=np.float32)
    corr_value = float(fc_corr(jnp.asarray(fc_matrix), jnp.asarray(fc_target)))
    rmse_value = float(np.sqrt(np.mean((fc_matrix - fc_target) ** 2)))
    return fc_matrix, corr_value, rmse_value


# ── 시각화 ────────────────────────────────────────────────────

def _plot_fic_results(result: dict, cfg: Config, data: dict) -> None:
    mean_rE_hz_history = np.asarray(result["mean_rE_hz_history"])
    bold_signal = np.asarray(result["bold_signal"])
    pre_data = np.asarray(result["pre_fic_neural"])
    post_data = np.asarray(result["post_fic_neural"])

    rE_max = float(WilsonCowanEIB.DEFAULT_PARAMS.rE_max_hz)
    target_excitatory_level = cfg.fic_target_firing_rate_hz / rE_max

    fig, axes = plt.subplots(2, 2, figsize=(8.1, 7))
    fig.suptitle("Part 1 — FIC Results", fontsize=13)

    for axis, neural_data, title in zip(
        axes[0],
        [pre_data[:, 0, :], post_data[:, 0, :]],
        ["Before FIC", "After FIC"],
    ):
        axis.plot(neural_data, alpha=0.6, linewidth=0.8)
        axis.axhline(
            target_excitatory_level,
            color="orange", linestyle="--", linewidth=2,
            label=f"E equiv. ({target_excitatory_level:.2f}, rE={cfg.fic_target_firing_rate_hz:.1f}Hz)",
        )
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Time step")
        axis.set_ylabel("E (Excitatory activity)")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)

    axes[1, 0].plot(mean_rE_hz_history, linewidth=2, label="Mean rE_hz")
    axes[1, 0].axhline(
        cfg.fic_target_firing_rate_hz,
        color="orange", linestyle="--", linewidth=2,
        label=f"Target {cfg.fic_target_firing_rate_hz:.1f} Hz",
    )
    axes[1, 0].set_title(
        f"FIC Convergence — {cfg.fic_target_firing_rate_hz:.0f} Hz target\n"
        f"(skip {cfg.fic_step_skip_tr} TR, use {int(cfg.fic_step_duration_ms/cfg.bold_repetition_time_ms)-cfg.fic_step_skip_tr} TR)"
    )
    axes[1, 0].set_xlabel("FIC iteration")
    axes[1, 0].set_ylabel("rE_hz (Hz)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if bold_signal.shape[0] > 1:
        mean_bold = bold_signal.mean(axis=1)
        for region_bold in bold_signal.T:
            axes[1, 1].plot(region_bold, alpha=0.3, linewidth=0.5)
        axes[1, 1].plot(mean_bold, color="steelblue", linewidth=2.0, label="Mean")
        axes[1, 1].set_title(
            f"BOLD Signal (skip {cfg.fic_step_skip_tr} TR removed)"
        )
        axes[1, 1].set_xlabel("BOLD sample")
        axes[1, 1].set_ylabel("BOLD signal")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fc_target = np.asarray(data["fc_target"], dtype=np.float32)
    pre_fic_fc = np.asarray(result["pre_fic_fc"], dtype=np.float32)
    post_fic_fc = np.asarray(result["post_fic_fc"], dtype=np.float32)

    print(
        f"[FIC] FC comparison  "
        f"Pre-opt corr={result['pre_fic_fc_corr']:.4f}, Pre-opt rmse={result['pre_fic_fc_rmse']:.4f}  "
        f"Post-FIC corr={result['post_fic_fc_corr']:.4f}, Post-FIC rmse={result['post_fic_fc_rmse']:.4f}"
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Part 1 — FC matrices", fontsize=13)
    
    titles = [
        "Target FC",
        f"Pre-opt FC\n(corr={result['pre_fic_fc_corr']:.4f}, rmse={result['pre_fic_fc_rmse']:.4f})",
        f"Post-FIC FC\n(corr={result['post_fic_fc_corr']:.4f}, rmse={result['post_fic_fc_rmse']:.4f})",
    ]
    for axis, fc_matrix, title in zip(axes, [fc_target, pre_fic_fc, post_fic_fc], titles):
        image = axis.imshow(
            np.nan_to_num(fc_matrix),
            vmin=cfg.fc_plot_vmin, vmax=cfg.fc_plot_vmax,
            cmap="RdBu_r",
        )
        axis.set_title(title)
        axis.set_xlabel("Region")
        axis.set_ylabel("Region")
        plt.colorbar(image, ax=axis, fraction=0.046)

    plt.tight_layout()
    plt.show()


# ── Beta oscillation check ────────────────────────────────────

def _analyze_beta_oscillation_fic(neural_data: np.ndarray, cfg: Config) -> None:
    summary = _compute_beta_summary_from_neural(neural_data, cfg)
    _plot_beta_summary(summary, cfg, title="Part 1 — Beta Oscillation Check")


def _compute_beta_summary_from_neural(neural_data: np.ndarray, cfg: Config) -> dict:
    lfp_signal = neural_data[:, 0, :] + neural_data[:, 1, :]
    mean_lfp = np.mean(lfp_signal, axis=1)

    fs_hz = 1000.0 / cfg.integration_dt_ms
    nperseg = int(fs_hz * 2)
    noverlap = nperseg // 2

    frequencies, normalized_psd = _compute_normalized_psd(
        mean_lfp, fs_hz, nperseg, noverlap, cfg.dbs_psd_max_frequency_hz
    )
    beta_ratio = _compute_beta_power_ratio(
        frequencies, normalized_psd,
        cfg.dbs_beta_band_low_hz, cfg.dbs_beta_band_high_hz,
    )

    return {
        "lfp_signal": mean_lfp,
        "frequencies": frequencies,
        "psd": normalized_psd,
        "beta_ratio": beta_ratio,
    }


def _plot_beta_summary(summary: dict, cfg: Config, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.suptitle(f"{title}  |  beta ratio={summary['beta_ratio']:.4f}", fontsize=13)

    axes[0].plot(summary["lfp_signal"], linewidth=0.7, color="black")
    axes[0].set_title("Mean LFP (E + I)")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("LFP")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(summary["frequencies"], summary["psd"], linewidth=1.4, color="steelblue")
    axes[1].axvspan(
        cfg.dbs_beta_band_low_hz, cfg.dbs_beta_band_high_hz,
        alpha=0.15, color="orange",
        label=f"Beta {cfg.dbs_beta_band_low_hz:.0f}-{cfg.dbs_beta_band_high_hz:.0f} Hz",
    )
    axes[1].set_xlim(0, cfg.dbs_psd_max_frequency_hz)
    axes[1].set_title("Normalized PSD")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD [a.u.]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _compute_normalized_psd(
    signal: np.ndarray,
    fs_hz: float,
    nperseg: int,
    noverlap: int,
    f_max_hz: float,
) -> tuple:
    frequencies, power_spectrum = welch(
        signal, fs=fs_hz, nperseg=nperseg, noverlap=noverlap
    )
    total_power = trapezoid(power_spectrum, frequencies)
    if total_power > 0:
        power_spectrum = power_spectrum / total_power
    frequency_mask = frequencies <= f_max_hz
    return frequencies[frequency_mask], power_spectrum[frequency_mask]


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
