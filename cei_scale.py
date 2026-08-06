#!/usr/bin/env python3
"""해석적 c_ei × k 를 고정(eta=0)하고 어느 k 에서 낮은 가지로 넘어가나.

관측: 해석해 1.9350 -> mean_S_e 0.475 (높은 가지, 초기값 0.15/0.25/0.47 무관)
      FIC2000  1.9755 -> mean_S_e 0.222 (낮은 가지)
      = 2.1% 차이가 attractor 를 가른다 (안장-마디 분기 근처).

k 를 훑어 임계점을 찾고, 넘긴 뒤 FC corr 을 잰다.
되면 FIC 완전 생략 가능 (해석해 + 상수 보정 1개).
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

idx = 4
BASE = 0.6497
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
print(f"idx{idx} 해석해 c_ei mean {c_an.mean():.4f}  (FIC2000 = 1.9755, 비 {1.9755/c_an.mean():.4f})",
      flush=True)

for k in (1.00, 1.03, 1.06, 1.12, 1.25):
    cfg.cache_version = f"{base_cv}_k{int(k*100)}"
    cfg.fic_max_iterations = 60
    cfg.fic_learning_rate = 0.0
    cfg.fic_posthoc_top_k = 1
    ps = ParamSet(c_ei=(c_an * k).astype(np.float32), wLRE=ps1.wLRE, wFFI=ps1.wFFI,
                  c_ei_frozen=False).sanitize(mask, cap)
    print(f"\n>>> k={k:.2f}  c_ei mean {(c_an*k).mean():.4f}", flush=True)
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
    print(f"### k={k:.2f}  c_ei {(c_an*k).mean():.4f}  corr {c:+.4f}  rmse {r:.4f}  "
          f"SC>0 {np.corrcoef(fc[IU][mtri], FC_emp[IU][mtri])[0,1]:+.4f}  (기준 {BASE:+.4f})",
          flush=True)
