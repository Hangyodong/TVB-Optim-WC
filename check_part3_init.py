#!/usr/bin/env python3
"""
check_part3_init.py — 전 idx(0..8) part3 초기 loss/corr 진단

각 idx에 대해 EIB 캐시를 로드하고, part3가 보는 **초기 상태**(최적화 0스텝)의
FC corr / loss 를 1회 forward 시뮬로 계산한다. (gradient 없음 → remat 불필요, 가벼움)

출력:
  - EIB 저장 corr (캐시 파일의 post_eib_corr, plain)
  - part3 재시뮬 corr: plain / edge-weighted(part3가 표시하는 "Full Corr")
  - part3 초기 loss (0.8·block_corr + 0.2·rmse, activity=0)
  - block corr (ctx/cross/sub)
  - gap = EIB저장 - 재시뮬(edge-weighted): 클수록 비재현(non-ergodic) → part1부터 재실행 대상

낮은 idx(재시뮬 edge-weighted corr < THRESH)는 part1부터 다시 하라고 목록으로 뽑아준다.

실행:
  python3 check_part3_init.py
  python3 check_part3_init.py --thresh 0.70      # 재실행 판정 임계 변경
  python3 check_part3_init.py --window 720        # 측정 window (기본 cfg값=720)
"""
import argparse

import numpy as np
import jax.numpy as jnp

import main_ppmi_pre as P          # find_best_eib_cache, M(=main_ppmi), load_data, build_network, StateBundle
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    weighted_corr_loss,
    weighted_rmse_loss,
    prepare_block_corr_terms,
    block_corr_loss,
    compute_block_corrs,
)
from tvboptim.observations.observation import fc_corr


def diagnose_idx(idx, window=None):
    best = P.find_best_eib_cache(idx)
    if best is None:
        return None
    eib_corr, path, obj = best

    p = P.M.prepare_ppmi_data(idx, P.NOISE_LEVEL)
    cfg = P.M.make_config(p, idx)
    if window is not None:
        cfg.optimizer_bold_window_tr = int(window)
        cfg.optimizer_bold_skip_tr = int(window) // 5
    data = P.load_data(cfg)

    network, *_ = P.build_network(cfg, data)
    bundle_eib = P.StateBundle.from_dict(obj["bundle"])

    # part3 와 동일 조건으로 초기 FC 시뮬 (settled state + 720window/skip144, forward only)
    t1 = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    fc = compute_simulated_fc(
        network, bundle_eib, cfg,
        sim_duration_ms=t1, skip_tr=cfg.optimizer_bold_skip_tr,
    )
    fc_j = jnp.asarray(fc)
    tgt = np.asarray(data["fc_target"], dtype=np.float32)
    tgt_safe = jnp.asarray(np.nan_to_num(tgt))
    fc_W = jnp.asarray(data["fc_edge_weight"])

    # corr (part3 표시 = edge-weighted), plain
    edgew = float(1.0 - weighted_corr_loss(fc_j, tgt_safe, fc_W))
    plain = float(fc_corr(fc_j, tgt_safe))

    # part3 초기 loss (0.8·block_corr + 0.2·rmse + 0·act)
    block_terms = prepare_block_corr_terms(
        data["fc_block_masks"],
        {"cc": cfg.corr_block_weight_cc,
         "cross": cfg.corr_block_weight_cross,
         "subsub": cfg.corr_block_weight_subsub},
    )
    rmse_mask = 1.0 - jnp.eye(tgt.shape[0], dtype=fc_W.dtype)
    L_global = float(block_corr_loss(fc_j, tgt_safe, block_terms))
    L_rmse = float(weighted_rmse_loss(fc_j, tgt_safe, rmse_mask))
    loss = cfg.optimizer_global_corr_weight * L_global + cfg.optimizer_rmse_weight * L_rmse

    bc = compute_block_corrs(fc, tgt, data["cortex_indices"], data["subcortex_indices"])

    return {
        "idx": idx,
        "eib_stored_corr": float(eib_corr),
        "resim_plain": plain,
        "resim_edgew": edgew,
        "init_loss": loss,
        "ctx": bc["ctx"], "cross": bc["cross"], "sub": bc["sub"],
        "gap": float(eib_corr) - edgew,
        "cache": path.split("/")[-1],
    }


def main():
    ap = argparse.ArgumentParser(description="전 idx part3 초기 loss/corr 진단")
    ap.add_argument("--thresh", type=float, default=0.70,
                    help="재시뮬 edge-weighted corr < thresh → part1 재실행 대상 (기본 0.70)")
    ap.add_argument("--window", type=int, default=None,
                    help="측정 window TR (미지정 시 cfg 기본=720)")
    ap.add_argument("--idxs", type=str, default="0,1,2,3,4,5,6,7,8",
                    help="검사할 idx 콤마구분 (기본 0..8)")
    args = ap.parse_args()
    idxs = [int(x) for x in args.idxs.split(",") if x.strip() != ""]

    rows = []
    for idx in idxs:
        print(f"\n===== idx {idx} 진단 중 =====")
        try:
            r = diagnose_idx(idx, window=args.window)
        except Exception as exc:
            import traceback
            print(f"[idx {idx}] FAILED: {exc}")
            traceback.print_exc()
            r = None
        if r is None:
            print(f"[idx {idx}] EIB 캐시 없음 → skip")
            continue
        rows.append(r)
        print(f"[idx {idx}] EIB저장={r['eib_stored_corr']:.4f}  "
              f"재시뮬 plain={r['resim_plain']:.4f} edge-w={r['resim_edgew']:.4f}  "
              f"init_loss={r['init_loss']:.4f}  gap={r['gap']:+.4f}  "
              f"block(ctx/cross/sub)={r['ctx']:.3f}/{r['cross']:.3f}/{r['sub']:.3f}")

    # ── 요약 표 ──
    print("\n" + "=" * 100)
    print("  part3 초기 진단 요약 (재시뮬 = part3가 실제로 보는 시작점)")
    print("=" * 100)
    print(f"  {'idx':>3} {'EIB저장':>8} {'재시뮬plain':>11} {'재시뮬edgeW':>11} "
          f"{'init_loss':>9} {'gap':>7} {'ctx':>6} {'cross':>6} {'sub':>6}")
    print("  " + "-" * 96)
    for r in sorted(rows, key=lambda x: x["resim_edgew"]):
        print(f"  {r['idx']:>3} {r['eib_stored_corr']:>8.4f} {r['resim_plain']:>11.4f} "
              f"{r['resim_edgew']:>11.4f} {r['init_loss']:>9.4f} {r['gap']:>+7.4f} "
              f"{r['ctx']:>6.3f} {r['cross']:>6.3f} {r['sub']:>6.3f}")

    # ── 재실행 대상 ──
    redo = [r["idx"] for r in rows if r["resim_edgew"] < args.thresh]
    good = [r["idx"] for r in rows if r["resim_edgew"] >= args.thresh]
    print("\n" + "=" * 100)
    print(f"  재시뮬 edge-w corr >= {args.thresh}  (정상, part3 진행): {good}")
    print(f"  재시뮬 edge-w corr <  {args.thresh}  (part1부터 재실행 권장): {redo}")
    print("=" * 100)
    if redo:
        print("  ※ 재실행 대상은 non-ergodic 의심 → noise↓(예 0.005) 후 part1/2/3 재튜닝 권장.")


if __name__ == "__main__":
    main()
