#!/usr/bin/env python3
"""3차 공식 가중치 + FIC 로 optim 이 재현되나 — formula_reproduce.py 의 3차 버전.

formula_reproduce.py 는 LOO **선형** 계수로 wLRE/wFFI 를 배정하고 FIC 를 돌린다
(idx4/5/6 = 0.6497 / 0.6210 / 0.5152, optimized 대비 75%).
여기서는 같은 절차에 계수만 3차로 바꾼다. 두 변형:

  cubic  : np.polyfit(deg=3) 계수        — 고 FC 에서 되돌아옴(wFFI>0) 결함 있음
  hinge3 : clip 을 손실에 넣고 LM 적합    — 되돌아옴 없음

가중치는 ParamSet.sanitize 가 clip[0, w_max] → *sc_mask → 대칭화 하므로
SC=0 엣지는 자동으로 0 이 된다.

사용: python3 formula_cubic.py --idxs 4,5,6 --model hinge3
"""
import argparse
import glob
import hashlib
import os
import pickle

import numpy as np
from scipy.optimize import least_squares

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (ParamSet, StateBundle, capture_internal_state,
                                capture_network_delay_history)
from formula_reproduce import IU, SUBDIR, pair

BASE_LIN = {4: 0.6497, 5: 0.6210, 6: 0.5152}          # formula_reproduce (선형)
OPT = {4: 0.7515, 5: 0.7857, 6: 0.8322, 7: 0.7564, 8: 0.7315}
WMAX_FIT = 10.0


def load_params(sub):
    """FC 해시 일치 grad 캐시 중 최고 corr 의 optimized wLRE/wFFI."""
    hs = {hashlib.sha1(np.asarray(np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv",
                                             delimiter=",")[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest")):
            continue
        if not any(h in os.path.basename(os.path.dirname(f)) for h in hs):
            continue
        d = pickle.load(open(f, "rb"))
        b = d.get("bundle", d) if isinstance(d, dict) else d
        md = (b.get("metadata", {}) if isinstance(b, dict) else getattr(b, "metadata", {})) or {}
        c = float(md.get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = b.get("params") if isinstance(b, dict) else getattr(b, "params")
            g = (lambda o, k: o.get(k) if isinstance(o, dict) else getattr(o, k))
            best, bc = {k: np.asarray(g(p, k), np.float64)
                        for k in ("wLRE", "wFFI")}, c
    return best


def hinge3_fit(x, y):
    def res(p):
        return np.clip(np.polyval(p, x), 0.0, WMAX_FIT) - y
    return least_squares(res, np.polyfit(x, y, 3), method="lm", max_nfev=40000).x


def loo_cubic(target_sub, model):
    """본인 제외 subject 들의 SC>0 엣지를 통합해 3차 계수 적합."""
    xs, wl, wf = [], [], []
    used = []
    for s in SUBDIR.values():
        if s == target_sub or not os.path.exists(f"output_ppmi_pd/{s}/inputs/weight.csv"):
            continue
        ref = load_params(s)
        if ref is None:
            continue
        used.append(s)
        W = np.loadtxt(f"output_ppmi_pd/{s}/inputs/weight.csv", delimiter=",")
        FC = np.loadtxt(f"output_ppmi_pd/{s}/inputs/FC.csv", delimiter=",")
        np.fill_diagonal(FC, 0.0)
        m = (W > 0)[IU]
        xs.append(FC[IU][m]); wl.append(ref["wLRE"][IU][m]); wf.append(ref["wFFI"][IU][m])
    x = np.concatenate(xs)
    fit = (lambda a, b: np.polyfit(a, b, 3)) if model == "cubic" else hinge3_fit
    return fit(x, np.concatenate(wl)), fit(x, np.concatenate(wf)), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", default="4,5,6")
    ap.add_argument("--model", default="hinge3", choices=("cubic", "hinge3"))
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--fic-steps", type=int, default=2000)
    ap.add_argument("--fic-topk", type=int, default=10)
    a = ap.parse_args()

    rows = []
    for idx in [int(v) for v in a.idxs.split(",")]:
        sub = SUBDIR[idx]
        cl, cf, used = loo_cubic(sub, a.model)
        print(f"\n{'='*74}\nidx{idx} ({sub})  LOO({a.model}) from {used}")
        print(f"  wLRE = {cl[3]:+.4f} {cl[2]:+.4f}FC {cl[1]:+.4f}FC² {cl[0]:+.4f}FC³")
        print(f"  wFFI = {cf[3]:+.4f} {cf[2]:+.4f}FC {cf[1]:+.4f}FC² {cf[0]:+.4f}FC³", flush=True)

        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)
        cfg.cache_version = f"{cfg.cache_version}_f{a.model}"
        cfg.fic_max_iterations = a.fic_steps
        cfg.fic_posthoc_top_k = a.fic_topk
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
        network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
        mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]

        # 엣지별 3차 배정. sanitize 가 clip[0,cap] → *sc_mask → 대칭화 (SC=0 은 0).
        wl = np.asarray(np.polyval(cl, FC_emp), np.float32)
        wf = np.asarray(np.polyval(cf, FC_emp), np.float32)
        ps = ParamSet(c_ei=np.ones(n, np.float32), wLRE=wl, wFFI=wf,
                      c_ei_frozen=False).sanitize(mask, cap)
        scb = mask.astype(bool)
        print(f"  가중치: wLRE mean {np.asarray(ps.wLRE)[scb].mean():.3f} "
              f"(0인 SC>0 엣지 {(np.asarray(ps.wLRE)[scb] <= 1e-6).mean()*100:.1f}%) | "
              f"wFFI mean {np.asarray(ps.wFFI)[scb].mean():.3f} "
              f"({(np.asarray(ps.wFFI)[scb] <= 1e-6).mean()*100:.1f}%)", flush=True)

        bundle = run_fic(network=network, cfg=cfg, data=data,
                         bundle_in=StateBundle.from_warmup(
                             warmup_result=warmup_result,
                             bold_monitor_template=bold_monitor, initial_params=ps,
                             internal_state=capture_internal_state(initial_state),
                             delay_history=capture_network_delay_history(network),
                             stage="warmup"))
        dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
        fc = np.asarray(compute_simulated_fc(network, bundle, cfg, sim_duration_ms=dur,
                                             skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
        corr, rmse = pair(fc, FC_emp)
        mt = scb[IU]
        rows.append((idx, corr, rmse, np.asarray(bundle.params.c_ei).mean()))
        print(f"### idx{idx} {a.model}: corr {corr:+.4f}  rmse {rmse:.4f}  "
              f"SC>0 {np.corrcoef(fc[IU][mt], FC_emp[IU][mt])[0,1]:+.4f}  "
              f"c_ei {np.asarray(bundle.params.c_ei).mean():.4f}  | "
              f"선형 {BASE_LIN.get(idx, float('nan')):.4f}  optimized {OPT[idx]:.4f}", flush=True)
        import jax
        jax.clear_caches()

    print(f"\n{'='*74}\n{'idx':>4} {'3차 corr':>9} {'선형 corr':>10} {'차이':>8} "
          f"{'optimized':>10} {'재현율':>8}")
    for idx, c, _, _ in rows:
        b = BASE_LIN.get(idx, np.nan)
        print(f"{idx:>4} {c:>9.4f} {b:>10.4f} {c-b:>+8.4f} {OPT[idx]:>10.4f} "
              f"{c/OPT[idx]*100:>7.1f}%")


if __name__ == "__main__":
    main()
