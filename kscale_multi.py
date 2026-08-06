#!/usr/bin/env python3
"""k=1.03 이 subject 공통인가 — idx5/6 에서 같은 스캔.

idx4 결과: k 1.00/1.03/1.06 -> corr 0.4700/0.6247/0.2947, S_e 0.478/0.405/0.189.
봉우리가 가지 전환 직전(높은 가지 끝)에 있다. idx5/6 에서 같은 k 인지 본다.

사용: python3 kscale_multi.py 5 6
"""
import sys
import numpy as np

sys.path.insert(0, "/scratch/home/wog3597/optim")

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (ParamSet, StateBundle, capture_internal_state,
                                capture_network_delay_history)
from formula_reproduce import IU, SUBDIR, loo_coefs, pair
from analytic_fic import analytic_c_ei

BASE = {4: 0.6497, 5: 0.6210, 6: 0.5152, 7: np.nan, 8: np.nan}
import os
KS = tuple(float(x) for x in os.environ.get("KS", "1.00,1.02,1.04,1.06,1.08").split(","))

for idx in [int(a) for a in sys.argv[1:]] or [5, 6]:
    sub = SUBDIR[idx]
    cw, _ = loo_coefs(sub)
    p = M.prepare_pd_data(idx, 0.02)
    cfg = M.make_config(p, idx, use_delay=True)
    base_cv = cfg.cache_version
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    mtri = mask.astype(bool)[IU]

    wl = np.asarray(cw["wLRE"][0] + cw["wLRE"][1] * FC_emp, np.float32)
    wf = np.asarray(cw["wFFI"][0] + cw["wFFI"][1] * FC_emp, np.float32)
    ps1 = ParamSet(c_ei=np.ones(n, np.float32), wLRE=wl, wFFI=wf,
                   c_ei_frozen=False).sanitize(mask, cap)
    c_an = np.asarray(np.clip(analytic_c_ei(np.asarray(data["weights"], np.float64),
                                            np.asarray(ps1.wLRE, np.float64),
                                            np.asarray(ps1.wFFI, np.float64))[0],
                              0, 20), np.float32)
    print(f"\n{'='*76}\nidx{idx} ({sub})  해석해 c_ei mean {c_an.mean():.4f}  "
          f"기준 {BASE[idx]:.4f}", flush=True)

    for k in KS:
        cfg.cache_version = f"{base_cv}_mk{int(k*100)}"
        cfg.fic_max_iterations = 60
        cfg.fic_learning_rate = 0.0
        cfg.fic_posthoc_top_k = 1
        ps = ParamSet(c_ei=(c_an * k).astype(np.float32), wLRE=ps1.wLRE,
                      wFFI=ps1.wFFI, c_ei_frozen=False).sanitize(mask, cap)
        b = run_fic(network=network, cfg=cfg, data=data,
                    bundle_in=StateBundle.from_warmup(
                        warmup_result=warmup_result, bold_monitor_template=bold_monitor,
                        initial_params=ps,
                        internal_state=capture_internal_state(initial_state),
                        delay_history=capture_network_delay_history(network),
                        stage="warmup"))
        fc = np.asarray(compute_simulated_fc(network, b, cfg, sim_duration_ms=dur,
                                             skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
        c, r = pair(fc, FC_emp)
        print(f"### idx{idx} k={k:.2f}  c_ei {(c_an*k).mean():.4f}  corr {c:+.4f}  "
              f"rmse {r:.4f}  SC>0 {np.corrcoef(fc[IU][mtri], FC_emp[IU][mtri])[0,1]:+.4f}  "
              f"(기준 {BASE[idx]:+.4f})", flush=True)
    import jax
    jax.clear_caches()
