#!/usr/bin/env python3
"""
main_ppmi_pre.py — 기존 EIB 캐시를 직접 로드해 Part 3 만 실행하는 배치 러너 (ppmi 0~8)

목적
----
코드를 실행하면 subject idx 0..8 중 **Part 2(EIB) 캐시가 존재하는 idx**에 대해
자동으로 Part 3(Full-matrix Gradient) 튜닝을 진행한다.
EIB 캐시가 없는 idx는 skip 한다.

왜 run_fic/run_eib 를 호출하지 않는가 (중요)
--------------------------------------------
run_fic/run_eib 의 캐시 키는 `bundle_init.fingerprint()` 에 의존하고, 이 fingerprint 는
warmup(720k-step 적분) 최종 상태를 포함한다. warmup 출력은 프로세스/디바이스 간 bit 단위로
재현되지 않으므로(특히 GPU 비결정성), 새 프로세스에서 run_fic 를 호출하면 캐시가 미스되어
FIC/EIB 를 **처음부터 재튜닝**한다.

따라서 이 러너는 EIB 캐시 .pkl 을 **직접 언피클**해 bundle_eib 를 복원하고, 곧장 Part 3 에
넣는다. Part 3 는 bundle_eib(전체 state) + network 만 필요하며, warmup 결과는 Part 3 에서
사용하지 않는다(state 는 bundle_eib 로 덮어씀). → warmup fp 와 무관하게 캐시 재사용.

EIB 캐시 선택 규칙
-----------------
- 각 idx 의 캐시 폴더(cache/ppmi216/v_ppmi216_s<idx>_*/) 안의 eib_*.pkl 을 모두 스캔.
- 여러 개면 post_eib_corr(없으면 best_fc_corr) 가 가장 높은 것을 선택.
- 하나도 없으면 skip.

prepare/make_config/Part3 로직은 main_ppmi.py 의 것을 그대로 재사용한다.
"""
import glob
import os
import pickle

# 풀배치는 window 720(EIB 측정선) → full backprop OOM → gradient checkpointing 필수.
# 기본 ON (사용자가 PART3_REMAT_SCAN=0 으로 끌 수 있음). part3 optimizer.run 에서 런타임에 읽음.
os.environ.setdefault("PART3_REMAT_SCAN", "1")

# main_ppmi 의 모듈 레벨 설정(JAX env, matplotlib Agg, figure 저장 패치)과
# 데이터 준비/Config 생성 함수를 그대로 재사용한다. (main_ppmi.main() 은 실행되지 않음)
import main_ppmi as M

from data_loader        import load_data
from model              import build_network
from part1_fic          import run_fic
from part2_eib          import run_eib
from part3_gradient     import run_gradient_optimization, run_sigma_optimization
from pipeline_contracts import (
    ParamSet,
    StateBundle,
    capture_internal_state,
    capture_network_delay_history,
)

# ── 배치 설정 ────────────────────────────────────────────────────────────
IDX_RANGE   = [1]                # 실험: idx 1
NOISE_LEVEL = 0.02               # main_ppmi 기본값과 동일
CACHE_ROOT  = "cache/ppmi216"    # cache_run_label = "ppmi216"

# ── FULL 재튜닝 모드 (part2/part3 rmse 가중치 변경 반영) ─────────────────
# part2(EIB)는 eib 캐시 로드 시 실행 안 됨 → rmse 가중치 바꾸려면 part1→2→3 전체 재실행 필요.
# eib 캐시 키엔 rmse 가중치가 없으므로 RETUNE_TAG 로 cache_version 을 바꿔 강제 fresh 재튜닝.
FULL_RETUNE       = True          # True: part1→2→3 full. False: 기존 eib 캐시 로드 후 part3 만.
PART2_RMSE_WEIGHT = 0.4           # EIB selection: cfg.rmse_loss_weight (기본 0.2 → 0.4)
PART2_CORR_WEIGHT = 0.6           # EIB selection: cfg.correlation_loss_weight (합=1)
PART3_RMSE_WEIGHT = 0.4           # Gradient loss: cfg.optimizer_rmse_weight (기본 0.2 → 0.4)
PART3_CORR_WEIGHT = 0.6           # Gradient loss: cfg.optimizer_global_corr_weight (합=1)
RETUNE_TAG        = "_rmse0p4"    # cache_version 접미사 → fic/eib/grad 캐시 강제 fresh

# ── Part 3.5 — per-node noise σ 최적화 ─────────────────────────────────────
# Part3(grad) 뒤에서 c_ei/wLRE/wFFI freeze 한 채 per-node additive noise σ 만 autodiff.
# σ↔c_ei 축퇴 차단(Part3 가 weight 먼저 맞춤 → Part3.5 가 잔차만 σ 로).
SIGMA_OPT        = True            # False: Part3.5 skip
SIGMA_PER_NODE   = True            # False: 공유 스칼라 σ 1개(파일럿: grad 생존 확인)
SIGMA_MAX        = 0.10            # BoundedParameter 상한(큰 σ→RWW 폭발 방지)
SIGMA_L2_WEIGHT  = 0.01            # L2(σ - mean σ): 노드별 spread 과적합 억제
SIGMA_LR         = None            # None → cfg.optimizer_learning_rate 재사용
SIGMA_MAX_STEPS  = None            # None → cfg.optimizer_max_steps 재사용

# ── GRAD 캐시 모드: part3(grad) 캐시 있는 idx만 로드 → Part3.5만 실행 ─────
# FULL_RETUNE 보다 우선. corr 상위 TOP_K idx 선택(part3 까지 이미 돈 것).
GRAD_CACHE_MODE  = True
GRAD_TOP_K       = 5               # post_grad_fc_corr 상위 K idx
GRAD_IDX_SCAN    = range(0, 9)     # 스캔 대상 idx


# ── Part 3.5 — σ 최적화 (공용) ──────────────────────────────────────────
def run_sigma_for_idx(idx: int, network, bundle_grad, cfg, data):
    """Part3 결과 bundle 을 받아 per-node σ 최적화. SIGMA_OPT=False 면 그대로 반환."""
    if not SIGMA_OPT:
        return bundle_grad
    print(f"\n  [idx {idx}] Part 3.5 — per-node σ 최적화  "
          f"per_node={SIGMA_PER_NODE}  σ_max={SIGMA_MAX}  L2={SIGMA_L2_WEIGHT}")
    bundle_sigma = run_sigma_optimization(
        network         = network,
        bundle_in       = bundle_grad,
        cfg             = cfg,
        data            = data,
        sigma_per_node  = SIGMA_PER_NODE,
        sigma_max       = SIGMA_MAX,
        sigma_l2_weight = SIGMA_L2_WEIGHT,
        sigma_lr        = SIGMA_LR,
        max_steps       = SIGMA_MAX_STEPS,
    )
    meta = bundle_sigma.metadata
    print(f"  [idx {idx}] Part3.5 done  "
          f"post_sigma_fc_corr={meta.get('post_sigma_fc_corr')}  "
          f"post_sigma_fc_rmse={meta.get('post_sigma_fc_rmse')}")
    return bundle_sigma


# ── EIB 캐시 탐지 + 선택 ────────────────────────────────────────────────
def find_best_eib_cache(idx: int):
    """idx 의 eib_*.pkl 중 post_eib_corr(없으면 best_fc_corr) 최고를 선택.
    반환: (corr, path, obj) 또는 None."""
    paths = glob.glob(os.path.join(CACHE_ROOT, f"v_ppmi216_s{idx}_*", "eib_*.pkl"))
    best = None
    for path in paths:
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:
            print(f"  [warn] {os.path.basename(path)} 로드 실패: {exc}")
            continue
        if not (isinstance(obj, dict) and "bundle" in obj):
            continue
        corr = obj.get("post_eib_corr")
        if corr is None:
            corr = obj.get("best_fc_corr", float("-inf"))
        if best is None or corr > best[0]:
            best = (float(corr), path, obj)
    return best


# ── GRAD 캐시 탐지 + 선택 (post_grad_fc_corr 최고) ─────────────────────
def find_best_grad_cache(idx: int):
    """idx 의 grad_*.pkl 중 post_grad_fc_corr 최고를 선택.
    반환: (corr, path, obj) 또는 None."""
    paths = glob.glob(os.path.join(CACHE_ROOT, f"v_ppmi216_s{idx}_*", "grad_*.pkl"))
    best = None
    for path in paths:
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:
            print(f"  [warn] {os.path.basename(path)} 로드 실패: {exc}")
            continue
        if not (isinstance(obj, dict) and "bundle" in obj):
            continue
        corr = obj["bundle"].get("metadata", {}).get("post_grad_fc_corr", float("-inf"))
        if best is None or corr > best[0]:
            best = (float(corr), path, obj)
    return best


# ── 단일 idx Part 3.5 (grad 캐시 직접 로드) ─────────────────────────────
def run_part35_from_grad_cache(idx: int, grad_corr: float, grad_path: str, grad_obj: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  [idx {idx}] Part 3.5 (grad 캐시 직접 로드)")
    print(f"     grad cache: {os.path.basename(grad_path)}")
    print(f"     post_grad_corr = {grad_corr:.4f}")
    print("=" * 70)

    # base cfg (grad 캐시가 a0p8_g0p2 등 base make_config 가중치로 생성됨 → 일치)
    p = M.prepare_ppmi_data(idx, NOISE_LEVEL)
    cfg = M.make_config(p, idx)

    # stale 가드: grad 캐시 TR/window 가 현재 cfg 와 다르면(옛 TR 캐시) σ 튜닝 부정합 → skip.
    tr_tag = f"_TR{cfg.optimizer_bold_window_tr}_"
    if tr_tag not in os.path.basename(grad_path):
        print(f"  [idx {idx}] SKIP — grad 캐시 TR 불일치(현재 {tr_tag.strip('_')}). "
              f"새 TR 로 part1→2→3 재튜닝 후 Part3.5 필요(run_sigma_idx23.py).")
        return

    cfg.print_summary()
    data = load_data(cfg)

    # network build (warmup 1회 실행되나 출력은 미사용 — bundle_grad 로 덮어씀)
    network, _initial_state, _bold_monitor, _warmup_result = build_network(cfg, data)

    # grad 캐시 → bundle_grad 복원
    bundle_grad = StateBundle.from_dict(grad_obj["bundle"])
    bundle_grad.apply_to_network(network)   # delay history 복원
    print(f"  [grad] bundle 복원 stage={bundle_grad.stage}  "
          f"mean c_ei={float(bundle_grad.params.c_ei.mean()):.4f}")

    # Part 3.5: per-node σ
    bundle_sigma = run_sigma_for_idx(idx, network, bundle_grad, cfg, data)
    print(f"  [idx {idx}] DONE  stage={bundle_sigma.stage}")
    print(bundle_sigma)


# ── 단일 idx Part 3 실행 ────────────────────────────────────────────────
def run_part3_for_idx(idx: int, eib_corr: float, eib_path: str, eib_obj: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  [idx {idx}] Part 3 tuning")
    print(f"     EIB cache: {os.path.basename(eib_path)}")
    print(f"     EIB post_corr = {eib_corr:.4f}")
    print("=" * 70)

    # 데이터 준비 + Config (main_ppmi 재사용). cfg 오버라이드 없음.
    p = M.prepare_ppmi_data(idx, NOISE_LEVEL)
    cfg = M.make_config(p, idx)
    cfg.print_summary()

    # Data loading
    data = load_data(cfg)

    # Network build (warmup 1회 실행되나 그 출력은 Part3 에서 사용하지 않음 — bundle_eib 로 덮어씀)
    network, _initial_state, _bold_monitor, _warmup_result = build_network(cfg, data)

    # EIB 캐시 .pkl 에서 bundle_eib 직접 복원 (run_fic/run_eib 우회 → 재튜닝 없음)
    bundle_eib = StateBundle.from_dict(eib_obj["bundle"])
    print(f"  [eib] bundle 복원 stage={bundle_eib.stage}  "
          f"c_ei_frozen={bundle_eib.params.c_ei_frozen}  "
          f"mean c_ei={float(bundle_eib.params.c_ei.mean()):.4f}")

    # Part 3: Full-matrix Gradient (실제 실행)
    bundle_grad = run_gradient_optimization(
        network   = network,
        bundle_in = bundle_eib,
        cfg       = cfg,
        data      = data,
    )
    meta = bundle_grad.metadata
    print(f"  [idx {idx}] Part3 done  stage={bundle_grad.stage}  "
          f"post_grad_fc_corr={meta.get('post_grad_fc_corr')}  "
          f"post_grad_fc_rmse={meta.get('post_grad_fc_rmse')}")
    print(bundle_grad)

    # Part 3.5: per-node σ 최적화
    bundle_grad = run_sigma_for_idx(idx, network, bundle_grad, cfg, data)
    print(bundle_grad)


# ── FULL 재튜닝 (part1→2→3, rmse 가중치 변경 반영) ──────────────────────
def run_full_retune_for_idx(idx: int) -> None:
    print("\n" + "=" * 70)
    print(f"  [idx {idx}] FULL 재튜닝 (part1→2→3)  "
          f"part2_rmse={PART2_RMSE_WEIGHT}  part3_rmse={PART3_RMSE_WEIGHT}")
    print("=" * 70)

    p = M.prepare_ppmi_data(idx, NOISE_LEVEL)
    cfg = M.make_config(p, idx)
    # 가중치 오버라이드 (corr+rmse 합=1)
    cfg.correlation_loss_weight   = PART2_CORR_WEIGHT   # part2 EIB selection corr
    cfg.rmse_loss_weight          = PART2_RMSE_WEIGHT   # part2 EIB selection rmse
    cfg.optimizer_global_corr_weight = PART3_CORR_WEIGHT  # part3 gradient corr
    cfg.optimizer_rmse_weight     = PART3_RMSE_WEIGHT   # part3 gradient rmse
    # eib 캐시 키에 rmse 가중치 없음 → cache_version 변경으로 강제 fresh (옛 rmse 0.2 캐시 hit 방지)
    cfg.cache_version = f"{cfg.cache_version}{RETUNE_TAG}"
    print(f"[cfg] cache_version={cfg.cache_version}  "
          f"rmse_loss_weight={cfg.rmse_loss_weight}  optimizer_rmse_weight={cfg.optimizer_rmse_weight}")
    cfg.print_summary()

    data = load_data(cfg)

    # warmup → bundle_init
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    initial_params = ParamSet.default(
        data["n_nodes"], c_ei_init=cfg.wc_c_ei_init
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)
    bundle_init = StateBundle.from_warmup(
        warmup_result          = warmup_result,
        bold_monitor_template  = bold_monitor,
        initial_params         = initial_params,
        internal_state         = capture_internal_state(initial_state),
        delay_history          = capture_network_delay_history(network),
        stage                  = "warmup",
    )

    # Part 1: FIC
    bundle_fic = run_fic(network=network, bundle_in=bundle_init, cfg=cfg, data=data)
    print(f"  [FIC] mean c_ei={bundle_fic.params.c_ei.mean():.4f}")

    # Part 2: EIB (rmse_loss_weight=0.4 반영)
    bundle_eib = run_eib(network=network, bundle_in=bundle_fic, cfg=cfg, data=data)
    print(f"  [EIB] post_eib_fc_corr={bundle_eib.metadata.get('post_eib_fc_corr')}")

    # Part 3: Gradient (optimizer_rmse_weight=0.4 반영, remat)
    bundle_grad = run_gradient_optimization(
        network       = network,
        bundle_in     = bundle_eib,
        warmup_bundle = bundle_init,
        cfg           = cfg,
        data          = data,
    )
    meta = bundle_grad.metadata
    print(f"  [idx {idx}] Part3 done  post_grad_fc_corr={meta.get('post_grad_fc_corr')}  "
          f"post_grad_fc_rmse={meta.get('post_grad_fc_rmse')}")
    print(bundle_grad)

    # Part 3.5: per-node σ 최적화
    bundle_grad = run_sigma_for_idx(idx, network, bundle_grad, cfg, data)
    print(f"  [idx {idx}] DONE")
    print(bundle_grad)


# ── 메인: idx 0..8 스캔 후 EIB 캐시 있는 것만 Part 3 ────────────────────
def main():
    if GRAD_CACHE_MODE:
        # grad 캐시 스캔 → corr 상위 GRAD_TOP_K idx 선택 → Part3.5 만 실행
        found = {}     # idx -> (corr, path, obj)
        for idx in GRAD_IDX_SCAN:
            best = find_best_grad_cache(idx)
            if best is not None:
                found[idx] = best
        ranked = sorted(found.items(), key=lambda kv: kv[1][0], reverse=True)
        top = ranked[:GRAD_TOP_K]

        print("=" * 70)
        print(f"  main_ppmi_pre — Part 3.5 (grad 캐시 직접 로드)  top_K={GRAD_TOP_K}")
        print("=" * 70)
        print("  grad 캐시 corr 랭킹:")
        for rank, (idx, (corr, path, _)) in enumerate(ranked):
            mark = "✓" if rank < GRAD_TOP_K else " "
            print(f"    [{mark}] s{idx}: corr={corr:.4f}  {os.path.basename(os.path.dirname(path))}")
        print("=" * 70)

        for idx, (corr, path, obj) in top:
            try:
                run_part35_from_grad_cache(idx, corr, path, obj)
            except Exception as exc:
                import traceback
                print(f"\n[idx {idx}] FAILED: {exc}")
                traceback.print_exc()

        print("\n" + "=" * 70)
        print(f"  Part3.5 complete: {[idx for idx, _ in top]}")
        print("=" * 70)
        return

    if FULL_RETUNE:
        print("=" * 70)
        print(f"  main_ppmi_pre — FULL 재튜닝 모드  idxs={list(IDX_RANGE)}")
        print(f"    part2 rmse_loss_weight={PART2_RMSE_WEIGHT}  "
              f"part3 optimizer_rmse_weight={PART3_RMSE_WEIGHT}  tag={RETUNE_TAG}")
        print("=" * 70)
        for idx in IDX_RANGE:
            try:
                run_full_retune_for_idx(idx)
            except Exception as exc:
                import traceback
                print(f"\n[idx {idx}] FAILED: {exc}")
                traceback.print_exc()
        print("\n" + "=" * 70)
        print(f"  Full retune complete: {list(IDX_RANGE)}")
        print("=" * 70)
        return

    plan = {}      # idx -> (corr, path, obj)
    skipped = []
    for idx in IDX_RANGE:
        best = find_best_eib_cache(idx)
        if best is not None:
            plan[idx] = best
        else:
            skipped.append(idx)

    print("=" * 70)
    print("  main_ppmi_pre — Part 3 batch over ppmi subjects 0..8 (EIB cache 직접 로드)")
    print("=" * 70)
    print("  EIB 캐시 있음 → Part3 진행:")
    for idx in plan:
        corr, path, _ = plan[idx]
        print(f"     s{idx}: corr={corr:.4f}  {os.path.basename(path)}")
    print(f"  EIB 캐시 없음 → skip       : "
          + (", ".join(f"s{i}" for i in skipped) if skipped else "(없음)"))
    print("=" * 70)

    for idx in plan:
        corr, path, obj = plan[idx]
        try:
            run_part3_for_idx(idx, corr, path, obj)
        except Exception as exc:
            import traceback
            print(f"\n[idx {idx}] FAILED: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"  Batch complete.  Part3 attempted: {list(plan.keys())}  skipped: {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
