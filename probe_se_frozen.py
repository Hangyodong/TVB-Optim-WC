#!/usr/bin/env python3
"""해석적 c_ei 를 고정(eta=0)한 채 100 step 돌려 mean_S_e 가 어디 앉는지 본다.

파이프라인 코드(run_fic) 그대로 사용 — 사이드 스크립트 상태전진 이슈 회피.
eta=0 이면 update_delta=0 이라 c_ei 는 해석해 그대로 유지된다.
비교: FIC2000(c_ei 자유) 최종 mean_E = 0.2220 (idx4).
"""
import sys
import numpy as np

sys.path.insert(0, "/scratch/home/wog3597/optim")

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from pipeline_contracts import (ParamSet, StateBundle, capture_internal_state,
                                capture_network_delay_history)
from formula_reproduce import SUBDIR, loo_coefs
from analytic_fic import analytic_c_ei

for idx in (4,):
    sub = SUBDIR[idx]
    cw, _ = loo_coefs(sub)
    p = M.prepare_pd_data(idx, 0.02)
    cfg = M.make_config(p, idx, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_frozen100"
    cfg.fic_max_iterations = 100
    cfg.fic_learning_rate = 0.0          # c_ei 고정
    cfg.fic_posthoc_top_k = 1
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]

    wl = np.asarray(cw["wLRE"][0] + cw["wLRE"][1] * FC_emp, np.float32)
    wf = np.asarray(cw["wFFI"][0] + cw["wFFI"][1] * FC_emp, np.float32)
    ps = ParamSet(c_ei=np.ones(n, np.float32), wLRE=wl, wFFI=wf,
                  c_ei_frozen=False).sanitize(mask, cap)
    c_an = np.asarray(np.clip(analytic_c_ei(np.asarray(data["weights"], np.float64),
                                            np.asarray(ps.wLRE, np.float64),
                                            np.asarray(ps.wFFI, np.float64))[0],
                              0, 20), np.float32)
    print(f"[probe] idx{idx} 해석적 c_ei mean {c_an.mean():.4f} (eta=0 으로 고정)", flush=True)
    b = run_fic(network=network, cfg=cfg, data=data,
                bundle_in=StateBundle.from_warmup(
                    warmup_result=warmup_result, bold_monitor_template=bold_monitor,
                    initial_params=ParamSet(c_ei=c_an, wLRE=ps.wLRE, wFFI=ps.wFFI,
                                            c_ei_frozen=False).sanitize(mask, cap),
                    internal_state=capture_internal_state(initial_state),
                    delay_history=capture_network_delay_history(network),
                    stage="warmup"))
    print(f"[probe] 종료 c_ei mean {np.asarray(b.params.c_ei).mean():.4f} "
          f"(해석해 {c_an.mean():.4f} 에서 이동 "
          f"{abs(np.asarray(b.params.c_ei).mean()-c_an.mean()):.5f})", flush=True)
