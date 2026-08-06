#!/usr/bin/env python3
"""run_perfect_fic.py — 변형 가중치(FC 와 corr ±1) 위에서 FIC 만 돌려 FC corr 확인.

조건 2 개를 같은 절차로 돌린다 (Part2 EIB / Part3 grad 없음):
  A 원본 optimized wLRE/wFFI + FIC   <- 기준선. Part2/3 생략의 비용을 분리한다.
  B 변형 wLRE/wFFI (r=±1) + FIC      <- 관심 조건

FIC 는 c_ei=1.0 에서 시작(파이프라인 Part1 과 동일), eta=0.5, 2000 step.
최종 FC 는 optimizer_bold_window_tr × TR (=600s) 시뮬 후 계산.

참고: 원본 가중치 + optimized c_ei (full pipeline) = 0.7524

실행: python3 run_perfect_fic.py
"""
import numpy as np
import jax

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (ParamSet, StateBundle, capture_internal_state,
                                capture_network_delay_history)

IDX = 4
NPZ = "output_ppmi_pd/_rel_data_idx4_perfect.npz"
FULL_PIPELINE_REF = 0.7524


def main():
    d = np.load(NPZ)
    p = M.prepare_pd_data(IDX, 0.02)
    cfg = M.make_config(p, IDX, use_delay=True)
    cfg.fic_max_iterations = 2000
    cfg.fic_posthoc_top_k = 10
    base_cv = cfg.cache_version

    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    iu = np.triu_indices(n, 1)
    scb = mask.astype(bool)
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

    conds = [
        ("A 원본 optimized w + FIC", "orig", d[f"wlre_{IDX}"], d[f"wffi_{IDX}"]),
        ("B 변형 w (r=±1) + FIC", "perf", d[f"wlre_{IDX}_perf"], d[f"wffi_{IDX}_perf"]),
    ]
    rows = []
    for label, tag, wl, wf in conds:
        cfg.cache_version = f"{base_cv}_pfic{tag}"
        ps = ParamSet(c_ei=np.ones(n, np.float32),
                      wLRE=np.asarray(wl, np.float32),
                      wFFI=np.asarray(wf, np.float32),
                      c_ei_frozen=False).sanitize(mask, cap)
        v_l, v_f = np.asarray(ps.wLRE)[scb], np.asarray(ps.wFFI)[scb]
        print(f"\n{'='*74}\n{label}")
        print(f"  wLRE mean {v_l.mean():.4f} max {v_l.max():.4f} | "
              f"wFFI mean {v_f.mean():.4f} max {v_f.max():.4f} | "
              f"합 mean {(v_l+v_f).mean():.4f}", flush=True)

        bundle = run_fic(network=network, cfg=cfg, data=data,
                         bundle_in=StateBundle.from_warmup(
                             warmup_result=warmup_result,
                             bold_monitor_template=bold_monitor, initial_params=ps,
                             internal_state=capture_internal_state(initial_state),
                             delay_history=capture_network_delay_history(network),
                             stage="warmup"))
        fc = np.asarray(compute_simulated_fc(network, bundle, cfg,
                                             sim_duration_ms=dur,
                                             skip_tr=cfg.optimizer_bold_skip_tr),
                        np.float32)
        x, y = FC_emp[iu], fc[iu]
        corr = float(np.corrcoef(x, y)[0, 1])
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        mt = scb[iu]
        corr_sc = float(np.corrcoef(x[mt], y[mt])[0, 1])
        c_ei = np.asarray(bundle.params.c_ei)
        rows.append((label, corr, rmse, corr_sc, c_ei.mean()))
        print(f"### {label}: corr {corr:+.4f}  rmse {rmse:.4f}  "
              f"SC>0 corr {corr_sc:+.4f}  c_ei mean {c_ei.mean():.4f} "
              f"[{c_ei.min():.3f},{c_ei.max():.3f}]", flush=True)
        jax.clear_caches()

    print(f"\n{'='*74}")
    print(f"{'조건':<26}{'corr':>9}{'rmse':>9}{'SC>0 corr':>11}{'c_ei mean':>11}")
    for label, corr, rmse, corr_sc, ce in rows:
        print(f"{label:<26}{corr:>+9.4f}{rmse:>9.4f}{corr_sc:>+11.4f}{ce:>11.4f}")
    print(f"{'참고: full pipeline':<26}{FULL_PIPELINE_REF:>+9.4f}")


if __name__ == "__main__":
    main()
