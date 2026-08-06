#!/usr/bin/env python3
"""se_at_optimum.py — activity_loss 가 정말 mean_S_e 를 0.25 에 묶고 있나.

part3 loss = 0.80*block_corr + 0.20*rmse + 1.0*mean((mean_S_e - 0.25)^2)
가중치는 fc 항과 같은 자릿수(1.0 vs 0.80)지만 **항의 스케일**이 다르다.
optimized 종점에서 실측 S_e 를 재고 세 항의 실제 기여를 나란히 찍는다.

실행: python3 se_at_optimum.py --idxs 4,5,6
"""
import argparse

import numpy as np
import jax

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part3_gradient import _settle_bundle
from pipeline_contracts import StateBundle
from formula_2x2 import SUBDIR, load_ref, measure_se, _get

import glob
import os
import pickle


def load_opt_bundle(sub, fc_hashes_fn, FC):
    """load_ref 와 같은 후보 선택 규칙, 단 params 가 아니라 bundle 전체를 돌려준다."""
    hs = fc_hashes_fn(FC)
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        if "formulafic" in f or "fwarm" in f or "oomtest" in f:
            continue
        if not any(h in os.path.basename(os.path.dirname(f)) for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle", d)
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            best, bc = (b if isinstance(b, dict) else b.to_dict()), c
    return (StateBundle.from_dict(best) if best is not None else None), bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", type=str, default="4,5,6")
    ap.add_argument("--noise-level", type=float, default=0.02)
    a = ap.parse_args()

    from formula_2x2 import fc_hashes

    rows = []
    for idx in [int(x) for x in a.idxs.split(",") if x.strip()]:
        sub = SUBDIR[idx]
        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)
        cfg.cache_version = f"{cfg.cache_version}_seopt"
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        network, *_ = build_network(cfg, data)
        t1 = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

        FC = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
        b, corr = load_opt_bundle(sub, fc_hashes, FC)
        if b is None:
            print(f"idx {idx} ({sub}): optimized grad 캐시 없음 — skip", flush=True)
            continue

        b, _, _ = _settle_bundle(network, b, cfg, sim_duration_ms=t1,
                                 skip_tr=cfg.optimizer_bold_skip_tr,
                                 next_stage="eib")
        se_mean = measure_se(network, b, cfg)

        # part3 loss 세 항의 실제 크기 (corr 은 캐시 metadata 의 post_grad_fc_corr)
        act = (se_mean - cfg.fic_target_se) ** 2
        corr_term = cfg.optimizer_global_corr_weight * (1.0 - corr)
        act_term = cfg.optimizer_activity_weight * act
        rows.append(dict(idx=idx, sub=sub, se=se_mean, corr=corr,
                         corr_term=corr_term, act_term=act_term))
        print(f"idx {idx} ({sub})  mean_S_e={se_mean:.4f}  corr={corr:.4f}  "
              f"corr항={corr_term:.4f}  activity항={act_term:.6f}  "
              f"비 {corr_term/max(act_term,1e-12):.0f}배", flush=True)
        jax.clear_caches()

    if rows:
        print("\n" + "=" * 70)
        print(f"{'idx':>4} {'mean_S_e':>9} {'corr':>8} {'corr항':>9} {'act항':>10} {'비':>8}")
        for r in rows:
            print(f"{r['idx']:>4} {r['se']:>9.4f} {r['corr']:>8.4f} "
                  f"{r['corr_term']:>9.4f} {r['act_term']:>10.6f} "
                  f"{r['corr_term']/max(r['act_term'],1e-12):>7.0f}x")
        se = np.array([r["se"] for r in rows])
        print(f"\nmean_S_e 평균 {se.mean():.4f}  범위 {se.min():.4f}~{se.max():.4f}  "
              f"(타깃 0.25)")


if __name__ == "__main__":
    main()
