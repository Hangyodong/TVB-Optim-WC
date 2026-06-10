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
import heapq
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
    extract_bold_window,
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
        f"_se{cfg.fic_target_se:g}"
        f"_eta{str(cfg.fic_learning_rate).replace('.', 'p')}"
        f"_steps{cfg.fic_max_iterations}"
        f"_dur{cfg.fic_step_duration_ms}"
        f"_skip{cfg.fic_step_skip_tr}"
        f"_bestbundlev4"  # best를 settle step 없이 그대로 핸드오프 — 이전 캐시 무효화
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

    # best 후보 선택용: rE 오차가 작은 top_k step만 스냅샷을 materialize한다.
    # 매 step 전체 state를 to_dict()로 직렬화하던 것(특히 매 step 커지는 bold_history의
    # host 전송)을 후보 자격을 얻은 step에서만 수행하도록 줄인다.
    # max-heap을 (-rE_error, step_index, bundle_dict)로 유지 → rE 오차 최소 top_k 보존.
    top_k = 10
    fic_topk_heap = []
    n_steps_completed = 0

    # wLRE / wFFI는 루프 동안 불변 → host 변환을 루프 밖에서 1회만 수행한다.
    wLRE_np = np.asarray(bundle_in.params.wLRE, dtype=np.float32)
    wFFI_np = np.asarray(bundle_in.params.wFFI, dtype=np.float32)

    total_tr = int(cfg.fic_step_duration_ms / bold_tr_ms)
    use_tr = total_tr - skip_tr
    print_every = 25

    print(
        f"[FIC] target S_e={cfg.fic_target_se}  "
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
        # 두 스칼라 mean을 device에서 stack해 1회 전송으로 host 동기화 (기존 float() ×2 → ×1)
        means_host = np.asarray(
            jnp.stack([jnp.mean(mean_excitatory_rate), jnp.mean(step_result.data[:, 0, :])])
        )
        current_mean_rate = float(means_host[0])
        current_mean_E = float(means_host[1])

        mean_firing_rate_history.append(current_mean_rate)
        mean_E_history.append(current_mean_E)

        tuned_state.initial_state.dynamics = step_result.data[-1]
        tuned_bold_monitor = update_bold_history(tuned_bold_monitor, step_result)
        internal_state, metadata = advance_internal_state(tuned_state, metadata)

        # ── best 후보 선택용 스냅샷: rE 오차 top_k 자격을 얻은 step만 materialize ──────
        # (Part 2 _run_eib_loop_pure의 current_bundle 저장 방식과 동일하게 to_dict() 직렬화)
        # 이 시점: c_ei는 update_delta 적용 전(현재 step), init_dynamics·bold_history·
        # internal_state는 이미 이 step으로 갱신됨 → 전 필드 일관.
        # 자격이 없는 step은 전체 state host 전송(특히 매 step 커지는 bold_history)을 건너뛴다.
        # 선택 지표: S_e gating 전역 오차 (EI_Tuning FIC 타깃)
        se_error = abs(current_mean_E - cfg.fic_target_se)
        qualifies_topk = (
            len(fic_topk_heap) < top_k or se_error < -fic_topk_heap[0][0]
        )
        if qualifies_topk:
            snap_bold_window = (
                np.concatenate(bold_signal_buffer, axis=0).astype(np.float32)
                if bold_signal_buffer else
                np.zeros((1, data["n_nodes"]), dtype=np.float32)
            )
            current_params = ParamSet(
                c_ei=np.asarray(tuned_state.dynamics.c_ei, dtype=np.float32),
                wLRE=wLRE_np,  # 루프 불변 — 루프 밖에서 1회 변환
                wFFI=wFFI_np,
                c_ei_frozen=bundle_in.params.c_ei_frozen,
            ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)
            current_bundle = bundle_in.advance(
                new_params=current_params,
                new_init_dynamics=np.asarray(tuned_state.initial_state.dynamics, dtype=np.float32),
                new_bold_history=np.asarray(tuned_bold_monitor.history, dtype=np.float32),
                new_bold_window=snap_bold_window,
                new_internal_state=internal_state,
                new_delay_history=bundle_in.delay_history,  # delay_history는 step별 저장 안 함(loop 종료 시 동기화)
                next_stage="fic_search",
                metadata_update=metadata,
            )
            # step_index를 두 번째 키로 둬 -rE_error 동률 시에도 dict 비교를 피한다.
            heapq.heappush(
                fic_topk_heap,
                (-se_error, step_index, current_bundle.to_dict()),
            )
            if len(fic_topk_heap) > top_k:
                heapq.heappop(fic_topk_heap)
        n_steps_completed += 1

        # EI_Tuning FIC 규칙 (S_e gating per-node):
        #   d_c_ei = eta * mean_S_i * (mean_S_e - target_se)
        mean_se_node, mean_si_node = _extract_gating_from_step(
            step_result, skip_tr, bold_tr_ms, cfg.integration_dt_ms
        )
        update_delta = (
            cfg.fic_learning_rate * mean_si_node * (mean_se_node - cfg.fic_target_se)
        )
        tuned_state.dynamics.c_ei = jnp.clip(
            tuned_state.dynamics.c_ei + update_delta, 0.0, 20.0
        )

        consecutive_convergence_count = (
            consecutive_convergence_count + 1
            if se_error < cfg.fic_early_stop_tolerance_se
            else 0
        )

        if (step_index + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"  step {step_index+1:>4}/{cfg.fic_max_iterations}"
                f"  mean_S_e={current_mean_E:.4f}"
                f"  rE={current_mean_rate:.3f} Hz"
                f"  se_err={se_error:.4f}"
                f"  ({elapsed:.1f}s)"
            )

        if consecutive_convergence_count >= cfg.fic_early_stop_patience:
            print(f"[FIC] Early stop at step {step_index + 1}")
            break

    # ── best 후보 선택 (Part 2 post-hoc validation 방식) ─────────
    # 1) rE 오차 오름차순 top_k → 2) 후보별 bundle을 재시뮬해 true FC 평가
    # → 3) Part 2와 동일한 true_score로 best 선택 → 4) 평가된 bundle 전체 복원
    fic_best_step = -1
    fic_best_rE_err = float("nan")
    fic_best_corr = float("nan")
    fic_candidates_rE_errors = np.zeros((0,), dtype=np.float32)
    best_bundle = None  # best 후보 선택 시 핸드오프 대상; None이면 settle fallback

    if fic_topk_heap:
        # heap에 보존된 top_k 후보를 rE 오차 오름차순(동률 시 step_index 오름차순)으로 정렬한다.
        # → 기존 sorted(fic_snapshots)[:top_k]와 동일한 후보 집합·순서.
        candidates = sorted(
            ((step, -neg_err, dct) for (neg_err, step, dct) in fic_topk_heap),
            key=lambda item: (item[1], item[0]),
        )
        fic_candidates_rE_errors = np.asarray(
            [item[1] for item in candidates], dtype=np.float32
        )
        print(
            f"[FIC] 후보 {len(candidates)}개 선택 — "
            f"전체 {n_steps_completed} step 중 rE 오차 최소 top_k={top_k}"
        )
        print(f"  {'Step':>6} {'rE-err':>8} {'True-corr':>10} {'True-RMSE':>10} {'Score':>10}")
        print("  " + "-" * 48)

        fc_target_np = np.asarray(data["fc_target"], dtype=np.float32)
        best_score = -np.inf
        best_eval = None

        for cand_step, cand_rE_err, cand_dict in candidates:
            # 후보 bundle을 Part 2 _evaluate_candidate_bundle과 동일 방식으로 재시뮬한다.
            cand_eval = _evaluate_fic_candidate_bundle(
                network,
                StateBundle.from_dict(cand_dict),
                cfg.fic_posthoc_duration_ms,
                cfg.fic_posthoc_skip_tr,
                cfg, data,
            )
            cand_corr = float(fc_corr(jnp.asarray(cand_eval["fc_matrix"]), jnp.asarray(fc_target_np)))
            cand_rmse = float(np.sqrt(np.mean((cand_eval["fc_matrix"] - fc_target_np) ** 2)))
            # Part 2와 동일한 true_score 공식.
            full_term = cfg.correlation_loss_weight * (1 - cand_corr) + cfg.rmse_loss_weight * cand_rmse
            cand_score = -(cfg.full_brain_fc_loss_weight * full_term)
            print(
                f"  {cand_step:>6} {cand_rE_err:>8.3f} "
                f"{cand_corr:>10.4f} {cand_rmse:>10.4f} {cand_score:>10.4f}"
            )
            if np.isfinite(cand_score) and cand_score > best_score:
                best_score = cand_score
                best_eval = cand_eval
                fic_best_step = int(cand_step)
                fic_best_rE_err = float(cand_rE_err)
                fic_best_corr = float(cand_corr)

        if best_eval is not None:
            best_bundle = best_eval["bundle"]
            print(
                f"[FIC] best 후보 — step={fic_best_step} "
                f"rE_err={fic_best_rE_err:.3f} corr={fic_best_corr:.4f}"
            )
        else:
            print("[FIC] 모든 후보 score가 비유한 → best 선택 건너뜀 (마지막 state 유지).")
    else:
        # 스냅샷이 없으면 patch 이전 동작 유지 (마지막 c_ei / bold_signal_buffer 사용).
        print("[FIC] 기록된 스냅샷이 없어 best 선택을 건너뛴다.")

    mean_E_arr = np.asarray(mean_E_history, dtype=np.float32)
    mean_rE_hz_arr = np.asarray(mean_firing_rate_history, dtype=np.float32)

    final_rate = mean_firing_rate_history[-1] if mean_firing_rate_history else float("nan")
    print(
        f"[FIC] Done — final mean_E={mean_E_history[-1]:.4f}  "
        f"final rE_hz={final_rate:.3f} Hz  "
        f"err={abs(final_rate - cfg.fic_target_firing_rate_hz):.3f} Hz"
    )

    if best_bundle is not None:
        # ── best 후보를 추가 settle step 없이 '그대로' 핸드오프 ───────────────
        # best_bundle은 post-hoc 재시뮬로 이미 settle된 상태. init_dynamics·
        # bold_history·bold_window·internal_state·delay_history를 그대로 보존하고
        # (advance에 None으로 두면 best 값 유지), params만 c_ei_frozen 플래그를
        # cfg에 맞춰 다시 stamp한다(c_ei/wLRE/wFFI 값은 best 그대로).
        best_params = ParamSet(
            c_ei=np.asarray(best_bundle.params.c_ei, dtype=np.float32),
            wLRE=np.asarray(best_bundle.params.wLRE, dtype=np.float32),
            wFFI=np.asarray(best_bundle.params.wFFI, dtype=np.float32),
            c_ei_frozen=bool(getattr(cfg, "freeze_c_ei_after_fic", False)),
        ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)
        bundle_fic = best_bundle.advance(new_params=best_params, next_stage="fic")
        bold_signal_arr = np.asarray(best_bundle.bold_window, dtype=np.float32)
        post_fic_neural = np.asarray(best_eval["neural_data"], dtype=np.float32)
    else:
        # ── best 미선택(nan/스냅샷 없음): 마지막 state + 1 step settle (기존 fallback) ──
        post_fic_result = step_model(tuned_state)

        tuned_state.initial_state.dynamics = post_fic_result.data[-1]
        tuned_bold_monitor = update_bold_history(tuned_bold_monitor, post_fic_result)
        internal_state, metadata = advance_internal_state(tuned_state, metadata)
        delay_history = sync_network_delay_history(network, post_fic_result)

        bold_signal_arr = (
            np.concatenate(bold_signal_buffer, axis=0).astype(np.float32)
            if bold_signal_buffer else
            np.zeros((1, data["n_nodes"]), dtype=np.float32)
        )
        fic_params = ParamSet(
            c_ei=np.asarray(tuned_state.dynamics.c_ei, dtype=np.float32),
            wLRE=np.asarray(bundle_in.params.wLRE, dtype=np.float32),
            wFFI=np.asarray(bundle_in.params.wFFI, dtype=np.float32),
            c_ei_frozen=bool(getattr(cfg, "freeze_c_ei_after_fic", False)),
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
        post_fic_neural = np.asarray(post_fic_result.data, dtype=np.float32)

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
        "post_fic_neural": post_fic_neural,
        "pre_fic_fc": np.asarray(pre_fic_fc, dtype=np.float32),
        "pre_fic_fc_corr": float(pre_fic_corr),
        "pre_fic_fc_rmse": float(pre_fic_rmse),
        "post_fic_fc": np.asarray(post_fic_fc, dtype=np.float32),
        "post_fic_fc_corr": float(post_fic_corr),
        "post_fic_fc_rmse": float(post_fic_rmse),
        # best c_ei 선택 결과 (전체 state 스냅샷 patch)
        "fic_best_step": int(fic_best_step),
        "fic_best_rE_err": float(fic_best_rE_err),
        "fic_best_corr": float(fic_best_corr),
        "fic_candidates_rE_errors": np.asarray(fic_candidates_rE_errors, dtype=np.float32),
    }


# ── 후보 bundle post-hoc 평가 (Part 2 _evaluate_candidate_bundle 방식) ──

def _evaluate_fic_candidate_bundle(
    network,
    candidate_bundle: StateBundle,
    sim_duration_ms: int,
    skip_tr: int,
    cfg: Config,
    data: dict,
) -> dict:
    """
    Part 2의 _evaluate_candidate_bundle과 동일하게 후보 bundle을 재시뮬레이션해
    true FC와 settle된 final bundle을 반환한다.
    """
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
        next_stage="fic",
        metadata_update=metadata,
    )
    return {
        "bundle": final_bundle,
        "fc_matrix": np.asarray(fc_matrix, dtype=np.float32),
        "neural_data": np.asarray(sim_result.data, dtype=np.float32),
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

    # Patch 15: c_ei 초기값을 cfg.wc_c_ei_init에서 읽는다 (dataset 전환).
    _c_ei_init = float(getattr(cfg, "wc_c_ei_init", 10.0))
    initial_params = ParamSet.default(data["n_nodes"], c_ei_init=_c_ei_init).sanitize(
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


def _extract_gating_from_step(
    step_result,
    skip_tr: int,
    bold_tr_ms: float,
    dt_ms: float,
) -> tuple:
    """EI_Tuning FIC 타깃용 per-node 평균 S_e/S_i gating (skip_tr 만큼 transient 제거)."""
    skip_steps = int(skip_tr * bold_tr_ms / dt_ms)
    total_steps = step_result.data.shape[0]
    start = min(skip_steps, total_steps - 1)
    data_used = step_result.data[start:]
    return (
        jnp.mean(data_used[:, 0, :], axis=0),
        jnp.mean(data_used[:, 1, :], axis=0),
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

    target_excitatory_level = cfg.fic_target_se  # FIC 타깃 (S_e gating)

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
            label=f"target S_e = {target_excitatory_level:.2f}",
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
