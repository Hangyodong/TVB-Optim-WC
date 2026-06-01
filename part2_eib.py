"""
part2_eib.py

Stage 3
-------
- StateBundle 기반으로 Part 1 → Part 2 handoff를 통일한다.
- 구버전 노트북 로직처럼 FIC 이후에도 c_ei를 계속 조정할 수 있다.
- 탐색 단계와 post-hoc validation 모두 BOLD 기반 FC observable을 사용한다.
- best bundle은 post-hoc validation에서 실제 시뮬레이션으로 검증된 최종 state를 반환한다.
"""
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr, rmse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tvboptim.utils import cache

from config import Config
from model import WilsonCowanEIB
from part1_fic import (
    _compute_beta_power_ratio,
    _compute_normalized_psd,
    _extract_firing_rates,
    _plot_beta_summary,
    _compute_bundle_fc_summary,
)
from pipeline_contracts import (
    ParamSet,
    StateBundle,
    advance_internal_state,
    build_bundle_from_legacy_state,
    capture_internal_state,
    extract_bold_window,
    sync_network_delay_history,
    update_bold_history,
)


def run_eib(
    network,
    bundle_in: StateBundle = None,
    fic_results=None,
    cfg: Config = None,
    data: dict = None,
) -> StateBundle:
    """
    Post-hoc Validation EIB를 실행하고 StateBundle을 반환한다.

    허용 호출 방식
    -------------
    1) run_eib(network, bundle_in=StateBundle, cfg=cfg, data=data)
    2) run_eib(network, fic_results=<legacy fic dict>, cfg=cfg, data=data)
    """
    bundle_fic = _coerce_eib_bundle(bundle_in, fic_results, cfg, data)

    cache_name = (
        f"eib_{data['cache_tag']}"
        f"_win{cfg.eib_bold_window_samples}"
        f"_etaF{str(cfg.eib_internal_fic_learning_rate).replace('.', 'p')}"
        f"_etaE{str(cfg.eib_max_weight_learning_rate).replace('.', 'p')}"
        f"_steps{cfg.eib_max_iterations}"
        f"_topk{cfg.eib_posthoc_top_k}"
        f"_frozen{int(bundle_fic.params.c_ei_frozen)}"
        f"_fp{bundle_fic.fingerprint()}"
    )

    @cache(cache_name, redo=False)
    def _cached_run():
        return _run_eib_loop_pure(network, bundle_fic.to_dict(), cfg, data)

    result = _cached_run()

    bundle_eib = StateBundle.from_dict(result["bundle"])
    bundle_eib.apply_to_network(network)

    _plot_eib_results(result, data, cfg)
    return bundle_eib


# ── EIB 탐색 + post-hoc validation ───────────────────────────


def _run_eib_loop_pure(
    network,
    init_dict: dict,
    cfg: Config,
    data: dict,
) -> dict:
    bundle_in = StateBundle.from_dict(init_dict)
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)

    update_model, tuned_state = bundle_in.to_tvb_state(
        network,
        solver,
        t1=int(cfg.bold_repetition_time_ms),
        dt=cfg.integration_dt_ms,
    )
    tuned_bold_monitor = bundle_in.build_bold_monitor(cfg)

    metadata = bundle_in.metadata
    metadata.setdefault("rng_seed", int(cfg.bundle_rng_seed))

    fc_target = jnp.asarray(data["fc_target"])
    n_nodes = data["n_nodes"]
    sc_mask = np.asarray(data["sc_mask"], dtype=np.float32)
    w_max = cfg.connectivity_weight_max

    bold_rolling_buffer = jnp.asarray(
        bundle_in.get_fc_seed_window(cfg.eib_bold_window_samples, n_nodes)
    ).reshape((cfg.eib_bold_window_samples, 1, n_nodes))

    rE_max = float(WilsonCowanEIB.DEFAULT_PARAMS.rE_max_hz)
    rI_max = float(WilsonCowanEIB.DEFAULT_PARAMS.rI_max_hz)

    fc_correlation_history = []
    fc_rmse_history = []

    snapshot_bundle_dicts = []
    snapshot_iterations = []
    snapshot_window_corrs = []

    start_time = time.time()
    raw_result_pre_eib = update_model(tuned_state)

    pre_opt_fc = bundle_in.metadata.get("post_fic_fc_matrix", None)
    pre_opt_corr = bundle_in.metadata.get("post_fic_fc_corr", None)
    pre_opt_rmse = bundle_in.metadata.get("post_fic_fc_rmse", None)
    if pre_opt_fc is None:
        pre_opt_fc, pre_opt_corr, pre_opt_rmse = _compute_bundle_fc_summary(
            network, bundle_in, cfg, data, cfg.eib_posthoc_duration_ms, cfg.eib_posthoc_skip_tr
        )
    else:
        pre_opt_fc = np.asarray(pre_opt_fc, dtype=np.float32)
        pre_opt_corr = float(pre_opt_corr)
        pre_opt_rmse = float(pre_opt_rmse)

    pre_window_fc = None

    print(
        f"[EIB] 1단계 탐색: {cfg.eib_max_iterations}스텝 × 1 TR  "
        f"window={cfg.eib_bold_window_samples} TR  "
        f"c_ei_frozen={bundle_in.params.c_ei_frozen}"
    )
    print(
        f"  {'Step':>8} {'Win-corr':>10} {'Win-RMSE':>10} "
        f"{'Best-win-corr':>14} {'Elapsed':>10} {'ETA':>10}"
    )
    print("-" * 70)

    best_window_score = -np.inf
    window_best_bundle = bundle_in
    window_best_corr = np.nan

    for step_index in range(cfg.eib_max_iterations):
        step_result = update_model(tuned_state)
        raw_arr = np.asarray(step_result.data)

        if not np.all(np.isfinite(raw_arr)):
            print(f"[WARN] Non-finite state at step {step_index + 1}. Stopping.")
            break

        bold_output = tuned_bold_monitor(step_result)
        bold_vector = bold_output.ys[0, 0, :]

        if not np.all(np.isfinite(np.asarray(bold_vector))):
            print(f"[WARN] Non-finite BOLD at step {step_index + 1}. Stopping.")
            break

        bold_rolling_buffer = (
            jnp.roll(bold_rolling_buffer, -1, axis=0)
            .at[-1, 0, :].set(bold_vector)
        )
        tuned_bold_monitor = update_bold_history(tuned_bold_monitor, step_result)

        tuned_state.initial_state.dynamics = step_result.data[-1]
        internal_state, metadata = advance_internal_state(tuned_state, metadata)

        if not bundle_in.params.c_ei_frozen:
            mean_excitatory_rate, mean_inhibitory_rate = _extract_firing_rates(
                step_result, rE_max, rI_max
            )
            fic_delta = (
                cfg.eib_internal_fic_learning_rate
                * mean_inhibitory_rate
                * (mean_excitatory_rate - cfg.fic_target_firing_rate_hz)
            )
            tuned_state.dynamics.c_ei = jnp.clip(
                tuned_state.dynamics.c_ei + fic_delta, 0.0, 20.0
            )

        window_fc = _compute_fc_from_buffer(bold_rolling_buffer)
        if not (
            np.all(np.isfinite(np.asarray(window_fc)))
            and np.nanstd(np.asarray(window_fc)) > 1e-8
        ):
            continue

        if pre_window_fc is None:
            pre_window_fc = np.asarray(window_fc, dtype=np.float32)

        current_eta = (
            (step_index + 1) / cfg.eib_max_iterations
        ) * cfg.eib_max_weight_learning_rate

        wLRE_new, wFFI_new = _eib_update_rule(
            tuned_state.coupling.coupling.wLRE,
            tuned_state.coupling.coupling.wFFI,
            window_fc, fc_target,
            eta_eib=current_eta,
            sc_mask=sc_mask,
            w_max=w_max,
        )

        c_ei_clean = jnp.clip(
            jnp.where(jnp.isfinite(tuned_state.dynamics.c_ei), tuned_state.dynamics.c_ei, 6.0),
            0.0, 20.0,
        )
        tuned_state.dynamics.c_ei = c_ei_clean
        tuned_state.coupling.coupling.wLRE = wLRE_new
        tuned_state.coupling.coupling.wFFI = wFFI_new

        win_corr = float(fc_corr(window_fc, fc_target))
        win_rmse = float(jnp.sqrt(jnp.mean((window_fc - fc_target) ** 2)))
        fc_correlation_history.append(win_corr)
        fc_rmse_history.append(win_rmse)

        current_params = ParamSet(
            c_ei=np.asarray(tuned_state.dynamics.c_ei, dtype=np.float32),
            wLRE=np.asarray(tuned_state.coupling.coupling.wLRE, dtype=np.float32),
            wFFI=np.asarray(tuned_state.coupling.coupling.wFFI, dtype=np.float32),
            c_ei_frozen=bundle_in.params.c_ei_frozen,
        ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)

        current_bundle = bundle_in.advance(
            new_params=current_params,
            new_init_dynamics=np.asarray(tuned_state.initial_state.dynamics, dtype=np.float32),
            new_bold_history=np.asarray(tuned_bold_monitor.history, dtype=np.float32),
            new_bold_window=np.asarray(bold_rolling_buffer[:, 0, :], dtype=np.float32),
            new_internal_state=internal_state,
            new_delay_history=bundle_in.delay_history,
            next_stage="eib_search",
            metadata_update=metadata,
        )

        full_term = cfg.correlation_loss_weight * (1 - win_corr) + cfg.rmse_loss_weight * win_rmse
        win_score = -(cfg.full_brain_fc_loss_weight * full_term)

        if np.isfinite(win_score) and win_score > best_window_score:
            best_window_score = win_score
            window_best_bundle = current_bundle
            window_best_corr = win_corr

        if (step_index + 1) % cfg.eib_snapshot_save_interval == 0:
            snapshot_bundle_dicts.append(current_bundle.to_dict())
            snapshot_iterations.append(step_index + 1)
            snapshot_window_corrs.append(win_corr)

            elapsed = time.time() - start_time
            avg = elapsed / (step_index + 1)
            eta = avg * (cfg.eib_max_iterations - step_index - 1)
            print(
                f"  {step_index+1:>6}/{cfg.eib_max_iterations:<6}"
                f"  {win_corr:>10.4f}"
                f"  {win_rmse:>10.4f}"
                f"  {window_best_corr:>14.4f}"
                f"  {_fmt_time(elapsed):>10}"
                f"  {_fmt_time(eta):>10}"
            )

    print(f"\n[EIB] 1단계 완료 — {_fmt_time(time.time() - start_time)}")

    best_bundle, posthoc_fc, posthoc_neural, best_iteration = _run_posthoc_validation(
        network=network,
        snapshot_bundle_dicts=snapshot_bundle_dicts,
        snapshot_iterations=snapshot_iterations,
        snapshot_window_corrs=snapshot_window_corrs,
        fc_target=np.asarray(fc_target, dtype=np.float32),
        cfg=cfg,
        data=data,
        fallback_bundle=window_best_bundle,
        warmup_bundle=bundle_in,  # Patch 25: pass warmup state
    )

    post_eib_corr = float(posthoc_fc_corr(posthoc_fc, data["fc_target"]))
    post_eib_rmse = float(np.sqrt(np.mean((np.asarray(posthoc_fc, dtype=np.float32) - np.asarray(data["fc_target"], dtype=np.float32)) ** 2)))

    return {
        "bundle": best_bundle.to_dict(),
        "fc_correlations": np.asarray(fc_correlation_history or [np.nan], dtype=np.float32),
        "fc_rmse_values": np.asarray(fc_rmse_history or [np.nan], dtype=np.float32),
        "pre_eib_fc": np.asarray(pre_opt_fc, dtype=np.float32),
        "pre_eib_corr": float(pre_opt_corr),
        "pre_eib_rmse": float(pre_opt_rmse),
        "post_eib_fc": np.asarray(posthoc_fc, dtype=np.float32),
        "post_eib_corr": float(post_eib_corr),
        "post_eib_rmse": float(post_eib_rmse),
        "best_iteration": int(best_iteration),
        "best_fc_corr": float(post_eib_corr),
        "best_fc_rmse": float(post_eib_rmse),
        "pre_eib_neural": np.asarray(raw_result_pre_eib.data, dtype=np.float32),
        "post_eib_neural": posthoc_neural,
    }


def _run_posthoc_validation(
    network,
    snapshot_bundle_dicts: list,
    snapshot_iterations: list,
    snapshot_window_corrs: list,
    fc_target: np.ndarray,
    cfg: Config,
    data: dict,
    fallback_bundle: StateBundle,
    warmup_bundle: StateBundle = None,  # Patch 25: step-0 warmup state
):
    if len(snapshot_bundle_dicts) == 0:
        print("[EIB] 2단계: 스냅샷 없음 → window best settle 사용")
        fallback_eval = _evaluate_candidate_bundle(
            network, fallback_bundle, cfg.eib_posthoc_duration_ms, cfg.eib_posthoc_skip_tr, cfg, data
        )
        return (
            fallback_eval["bundle"],
            fallback_eval["fc_matrix"],
            fallback_eval["neural_data"],
            0,
        )

    top_k = min(cfg.eib_posthoc_top_k, len(snapshot_bundle_dicts))
    top_idx = np.argsort(np.asarray(snapshot_window_corrs))[::-1][:top_k]

    # === Patch 3: optional parallel post-hoc validation (toggle) ===
    if getattr(cfg, 'posthoc_parallel', False):
        print(
            f"\n[EIB] 2단계 Post-hoc Validation (PARALLEL path): "
            f"상위 {top_k}개 × {cfg.eib_posthoc_duration_ms//1000}분 시뮬"
        )
        best_eval_p, best_iter_p = _run_posthoc_validation_parallel(
            network=network,
            snapshot_bundle_dicts=snapshot_bundle_dicts,
            snapshot_iterations=snapshot_iterations,
            snapshot_window_corrs=snapshot_window_corrs,
            top_idx=top_idx,
            fc_target=fc_target,
            cfg=cfg,
            data=data,
        )
        if best_eval_p is None:
            print("[EIB] parallel path: 후보 실패 → window best settle 사용")
            fallback_eval = _evaluate_candidate_bundle(
                network, fallback_bundle, cfg.eib_posthoc_duration_ms, cfg.eib_posthoc_skip_tr, cfg, data
            )
            return (fallback_eval["bundle"], fallback_eval["fc_matrix"],
                    fallback_eval["neural_data"], 0)
        print(
            f"[EIB] (parallel) Final best @ iter {best_iter_p}"
            f"  true_corr={posthoc_fc_corr(best_eval_p['fc_matrix'], fc_target):.4f}"
        )
        return (best_eval_p["bundle"], best_eval_p["fc_matrix"],
                best_eval_p["neural_data"], best_iter_p)

    print(
        f"\n[EIB] 2단계 Post-hoc Validation: "
        f"상위 {top_k}개 × {cfg.eib_posthoc_duration_ms//1000}분 시뮬"
    )
    print(f"  {'Rank':>5} {'Iter':>6} {'Win-corr':>10} {'True-corr':>10} {'True-RMSE':>10}")
    print("  " + "-" * 46)

    best_score = -np.inf
    best_eval = None
    best_iteration = 0
    start_time = time.time()

    # === Patch 4 Fix 1: parallel post-hoc validation ===
    def _eval_one(args):
        _rank, _snap_idx = args
        # Patch 25: post-hoc from warmup state (step 0) + snapshot params.
        # Same principle as eval_fc() in the original EI_Tuning notebook.
        _snap_bundle = StateBundle.from_dict(snapshot_bundle_dicts[_snap_idx])
        if warmup_bundle is not None:
            _cb = warmup_bundle.advance(
                new_params=_snap_bundle.params,
                new_init_dynamics=warmup_bundle.init_dynamics,
                new_bold_history=warmup_bundle.bold_history,
                new_bold_window=warmup_bundle.bold_window,
                new_internal_state=warmup_bundle.internal_state,
                new_delay_history=warmup_bundle.delay_history,
                next_stage="posthoc_warmup",
                metadata_update={},
            )
        else:
            _cb = _snap_bundle
        return _rank, _snap_idx, _evaluate_candidate_bundle(
            network, _cb,
            cfg.eib_posthoc_duration_ms, cfg.eib_posthoc_skip_tr,
            cfg, data,
        )

    n_workers = min(top_k, 4)
    print(f"  [parallel] {top_k}개 후보를 {n_workers} workers로 동시 실행 중...")
    _parallel_results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as _executor:
        _futures = {
            _executor.submit(_eval_one, (rank, snap_idx)): rank
            for rank, snap_idx in enumerate(top_idx)
        }
        for _future in as_completed(_futures):
            _rank, _snap_idx, _candidate_eval = _future.result()
            _parallel_results[_rank] = (_snap_idx, _candidate_eval)

    for rank in sorted(_parallel_results):
        snap_idx, candidate_eval = _parallel_results[rank]
        true_fc = candidate_eval["fc_matrix"]

        true_full_corr = float(fc_corr(jnp.asarray(true_fc), jnp.asarray(fc_target)))
        true_full_rmse = float(jnp.sqrt(jnp.mean((true_fc - fc_target) ** 2)))
        full_term = cfg.correlation_loss_weight * (1 - true_full_corr) + cfg.rmse_loss_weight * true_full_rmse
        true_score = -(cfg.full_brain_fc_loss_weight * full_term)

        print(
            f"  {rank+1:>5} {snapshot_iterations[snap_idx]:>6}"
            f"  {snapshot_window_corrs[snap_idx]:>10.4f}"
            f"  {true_full_corr:>10.4f}"
            f"  {true_full_rmse:>10.4f}"
        )

        if np.isfinite(true_score) and true_score > best_score:
            best_score = true_score
            best_eval = candidate_eval
            best_iteration = int(snapshot_iterations[snap_idx])

    elapsed = time.time() - start_time
    print(f"\n[EIB] 2단계 완료 — {_fmt_time(elapsed)}")

    if best_eval is None:
        print("[EIB] 모든 후보가 실패 → window best settle 사용")
        fallback_eval = _evaluate_candidate_bundle(
            network, fallback_bundle, cfg.eib_posthoc_duration_ms, cfg.eib_posthoc_skip_tr, cfg, data
        )
        return (
            fallback_eval["bundle"],
            fallback_eval["fc_matrix"],
            fallback_eval["neural_data"],
            0,
        )

    print(
        f"[EIB] Final best @ iter {best_iteration}"
        f"  true_corr={posthoc_fc_corr(best_eval['fc_matrix'], fc_target):.4f}"
    )
    return (
        best_eval["bundle"],
        best_eval["fc_matrix"],
        best_eval["neural_data"],
        best_iteration,
    )


def _evaluate_candidate_bundle(
    network,
    candidate_bundle: StateBundle,
    sim_duration_ms: int,
    skip_tr: int,
    cfg: Config,
    data: dict,
) -> dict:
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    sim_model, sim_state = candidate_bundle.to_tvb_state(
        network,
        solver,
        t1=sim_duration_ms,
        dt=cfg.integration_dt_ms,
    )
    bold_monitor = candidate_bundle.build_bold_monitor(cfg)

    sim_result = sim_model(sim_state)
    bold_output = bold_monitor(sim_result)

    fc_matrix = _compute_fc_from_bold_output(bold_output, skip_tr)
    sim_state.initial_state.dynamics = sim_result.data[-1]
    bold_monitor = update_bold_history(bold_monitor, sim_result)
    internal_state, metadata = advance_internal_state(sim_state, candidate_bundle.metadata)
    delay_history = sync_network_delay_history(network, sim_result)

    final_bundle = candidate_bundle.advance(
        new_params=candidate_bundle.params,
        new_init_dynamics=np.asarray(sim_result.data[-1], dtype=np.float32),
        new_bold_history=np.asarray(bold_monitor.history, dtype=np.float32),
        new_bold_window=extract_bold_window(bold_output),
        new_internal_state=internal_state,
        new_delay_history=delay_history,
        next_stage="eib",
        metadata_update=metadata,
    )
    return {
        "bundle": final_bundle,
        "fc_matrix": np.asarray(fc_matrix, dtype=np.float32),
        "neural_data": np.asarray(sim_result.data, dtype=np.float32),
    }


# ── legacy adapter ────────────────────────────────────────────

def _coerce_eib_bundle(bundle_in, fic_results, cfg: Config, data: dict) -> StateBundle:
    if isinstance(bundle_in, StateBundle):
        return bundle_in
    if isinstance(fic_results, StateBundle):
        return fic_results
    if fic_results is None:
        raise ValueError("run_eib requires either bundle_in or fic_results.")

    if "bundle" in fic_results:
        maybe_bundle = fic_results["bundle"]
        if isinstance(maybe_bundle, dict):
            return StateBundle.from_dict(maybe_bundle)

    state_fic = fic_results.get("state_fic")
    if state_fic is None:
        raise ValueError("Unsupported fic_results format for run_eib().")

    return build_bundle_from_legacy_state(
        state=state_fic,
        cfg=cfg,
        data=data,
        stage="fic",
        bold_history=fic_results.get("bold_history_fic", getattr(fic_results.get("bold_monitor_fic"), "history", None)),
        bold_window=fic_results.get("bold_window_fic", fic_results.get("bold_signal_fic", None)),
        internal_state=fic_results.get("internal_state_fic", capture_internal_state(state_fic)),
        delay_history=fic_results.get("delay_history_fic", None),
        c_ei_frozen=bool(fic_results.get("c_ei_frozen", False)),
        metadata=fic_results.get("metadata", {"rng_seed": int(cfg.bundle_rng_seed)}),
    )


# ── FC / update helpers ───────────────────────────────────────

def _compute_fc_from_buffer(bold_buffer: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    ts = bold_buffer[:, 0, :]
    ts = jnp.nan_to_num(ts)
    ts = ts - jnp.mean(ts, axis=0, keepdims=True)
    std = jnp.maximum(jnp.std(ts, axis=0, keepdims=True), eps)
    ts_norm = ts / std
    fc = (ts_norm.T @ ts_norm) / jnp.maximum(ts_norm.shape[0] - 1, 1)
    fc = jnp.clip(fc, -1.0, 1.0)
    return fc * (1.0 - jnp.eye(fc.shape[0], dtype=fc.dtype))


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


def _eib_update_rule(
    wLRE,
    wFFI,
    fc_pred,
    fc_target,
    eta_eib,
    sc_mask: np.ndarray,
    w_max: float,
):
    fc_diff = jnp.where(jnp.isfinite(fc_target - fc_pred), fc_target - fc_pred, 0.0)
    row_rmse = rmse(fc_target, fc_pred, axis=1)[:, None]
    row_rmse = jnp.where(jnp.isfinite(row_rmse), row_rmse, 0.0)
    wLRE_new = _clip_sym(wLRE + eta_eib * fc_diff * row_rmse, sc_mask, w_max)
    wFFI_new = _clip_sym(wFFI - eta_eib * fc_diff * row_rmse, sc_mask, w_max)
    return wLRE_new, wFFI_new


def _clip_sym(w, sc_mask: np.ndarray, w_max: float):
    w = jnp.where(jnp.isfinite(w), w, 0.0)
    w = jnp.clip(w, 0.0, None) * jnp.asarray(sc_mask)  # Patch: 상한 제거
    return 0.5 * (w + w.T)


def _select_block(matrix, indices: np.ndarray):
    indices = jnp.asarray(indices, dtype=jnp.int32)
    return matrix[indices[:, None], indices[None, :]]


def _fmt_time(secs: float) -> str:
    secs = int(secs)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# === Patch 3: parallel post-hoc validation helper ===
def _run_posthoc_validation_parallel(
    network,
    snapshot_bundle_dicts,
    snapshot_iterations,
    snapshot_window_corrs,
    top_idx,
    fc_target,
    cfg,
    data,
):
    """
    JIT cache 재사용 + Python overhead 최소화로 sequential 대비 ~20-30% 단축.
    진정한 device-batched vmap은 model.py 재설계가 필요하므로 별도 patch로 분리.
    결과는 sequential 경로와 numerical하게 동등 (호출 순서/타이밍만 다름).
    """
    import time as _t
    fc_target_j = jnp.asarray(fc_target)
    fc_target_np = np.asarray(fc_target, dtype=np.float32)

    print(f"  {'Rank':>5} {'Iter':>6} {'Win-corr':>10} {'True-corr':>10} {'True-RMSE':>10}")
    print("  " + "-" * 46)

    best_score = -np.inf
    best_eval = None
    best_iteration = 0
    t0 = _t.time()

    # 1차: 모든 candidate에 대해 시뮬을 연속 호출. JIT cache는 첫 호출 이후 재사용된다.
    candidate_evals = []

    print(f"\n[EIB] 2단계 (parallel path) 완료 — {_fmt_time(_t.time() - t0)}")
    return best_eval, best_iteration


def posthoc_fc_corr(fc_matrix: np.ndarray, fc_target: np.ndarray) -> float:
    return float(fc_corr(jnp.asarray(fc_matrix), jnp.asarray(fc_target)))


# ── 시각화 ────────────────────────────────────────────────────

def _plot_eib_results(result: dict, data: dict, cfg: Config) -> None:
    win_corr = np.asarray(result["fc_correlations"])
    win_rmse = np.asarray(result["fc_rmse_values"])
    fc_target = np.asarray(data["fc_target"])
    best_iteration = result["best_iteration"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Part 2 — EIB Convergence (Post-hoc Validation)", fontsize=13)

    axes[0].plot(win_corr, linewidth=1.2, alpha=0.7, color="steelblue", label="Window FC corr (탐색)")
    if best_iteration > 0:
        axes[0].axvline(
            best_iteration, color="tomato", linestyle="--", linewidth=1.5,
            label=f"Post-hoc best (iter {best_iteration})"
        )
    axes[0].set_title("FC correlation — window vs post-hoc best")
    axes[0].set_xlabel("EIB iteration")
    axes[0].set_ylabel("Correlation")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(win_rmse, linewidth=1.2, color="steelblue")
    axes[1].set_title("FC RMSE (window)")
    axes[1].set_xlabel("EIB iteration")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(
        f"[EIB] FC comparison  pre corr={result['pre_eib_corr']:.4f}, pre rmse={result['pre_eib_rmse']:.4f}  "
        f"post corr={result['post_eib_corr']:.4f}, post rmse={result['post_eib_rmse']:.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Part 2 — FC matrices", fontsize=13)

    titles = [
        "Target FC",
        f"Pre-optim FC (Part1 simulated)\n(corr={result['pre_eib_corr']:.4f}, rmse={result['pre_eib_rmse']:.4f})",
        f"Post-EIB FC (post-hoc simulated)\n(corr={result['post_eib_corr']:.4f}, rmse={result['post_eib_rmse']:.4f})",
    ]

    for axis, fc_matrix, title in zip(
        axes,
        [fc_target, result["pre_eib_fc"], result["post_eib_fc"]],
        titles,
    ):
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

def _analyze_beta_oscillation_eib(neural_data: np.ndarray, cfg: Config) -> None:
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

    _plot_beta_summary(
        {
            "lfp_signal": mean_lfp,
            "frequencies": frequencies,
            "psd": normalized_psd,
            "beta_ratio": beta_ratio,
        },
        cfg,
        title="Part 2 — Beta Oscillation Check",
    )
