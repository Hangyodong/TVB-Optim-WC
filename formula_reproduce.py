#!/usr/bin/env python3
"""formula_reproduce.py — "Emp FC 공식 가중치 + S_e=0.25 FIC c_ei" 가 TVB OPTIM 을 재현하나.

FC 행렬 3개를 만들어 서로 다 비교한다 (upper-tri, SC>0 무관 전체 엣지):

  FC_formula   wLRE/wFFI = a + b*FC_emp (LOO 계수) + 그 위에서 FIC 로 얻은 c_ei
  FC_optim     TVB 7시간 최적화 파라미터로 같은 harness 재시뮬 (저장값 재현 확인용)
  FC_emp       경험 FC (타깃)

⚠ noise_level 은 파이프라인 기본 0.02 여야 한다. additive_noise_sigma 가 cache_tag 에
없어 0.01 로 돌리면 캐시는 매칭되고 시뮬만 절반 노이즈로 돌아 corr 이 조용히 무너진다.

실행: python3 formula_reproduce.py --idxs 4,5,6
"""
import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from part1_fic import run_fic

IU = np.triu_indices(163, 1)
SUBDIR = {0: "100001", 1: "100005", 2: "100012", 3: "100268", 4: "100878",
          5: "100889", 6: "100905", 7: "100952", 8: "101025", 9: "101038"}


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def fc_hashes(FC):
    return {hashlib.sha1(np.asarray(FC[:8, :8], t).tobytes()).hexdigest()[:10]
            for t in (np.float32, np.float64)}


def load_ref(sub):
    """FC 해시 일치 grad 캐시 중 최고 corr → (params, 저장 corr, 저장 FC)."""
    fp = f"output_ppmi_pd/{sub}/inputs/FC.csv"
    if not os.path.exists(fp):
        return None, np.nan, None
    hs = fc_hashes(np.loadtxt(fp, delimiter=","))
    best, bc, bfc = None, -9.0, None
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest")):
            continue
        if not any(h in os.path.basename(os.path.dirname(f)) for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle", d)
        md = _get(b, "metadata", {}) or {}
        c = float(md.get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = _get(b, "params")
            best = {k: np.asarray(_get(p, k), np.float64)
                    for k in ("c_ei", "wLRE", "wFFI")}
            bc, bfc = c, np.asarray(md.get("post_grad_fc_matrix"), np.float32)
    return best, bc, bfc


def loo_coefs(target_sub):
    acc = {"wLRE": [], "wFFI": []}
    used = []
    for s in SUBDIR.values():
        if s == target_sub or not os.path.exists(f"output_ppmi_pd/{s}/inputs/weight.csv"):
            continue
        ref, _, _ = load_ref(s)
        if ref is None:
            continue
        used.append(s)
        W = np.loadtxt(f"output_ppmi_pd/{s}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"output_ppmi_pd/{s}/inputs/FC.csv", delimiter=",")
        m = (W > 0)[IU]
        A = np.stack([FCe[IU][m], np.ones(int(m.sum()))], 1)
        for nm in ("wLRE", "wFFI"):
            (b, a), *_ = np.linalg.lstsq(A, ref[nm][IU][m], rcond=None)
            acc[nm].append((float(a), float(b)))
    return {nm: tuple(np.mean(acc[nm], axis=0)) for nm in acc}, used


def pair(a, b):
    """upper-tri Pearson r 과 RMSE."""
    x, y = np.asarray(a)[IU], np.asarray(b)[IU]
    return float(np.corrcoef(x, y)[0, 1]), float(np.sqrt(np.mean((x - y) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", type=str, default="4,5,6")
    ap.add_argument("--noise-level", type=float, default=0.02)
    a = ap.parse_args()

    rows = []
    for idx in [int(x) for x in a.idxs.split(",") if x.strip()]:
        sub = SUBDIR[idx]
        cw, used = loo_coefs(sub)
        print(f"\n{'='*72}\nidx {idx} ({sub})  LOO 계수 from {used}")
        print(f"  wLRE = {cw['wLRE'][0]:+.4f} {cw['wLRE'][1]:+.4f}*FC_emp")
        print(f"  wFFI = {cw['wFFI'][0]:+.4f} {cw['wFFI'][1]:+.4f}*FC_emp", flush=True)

        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)
        # FIC 캐시 분리용 태그. fic_posthoc_duration_ms 는 cache_name 에 없으므로
        # 값이 바뀌면 반드시 태그도 바꿔야 구 캐시를 안 물어온다.
        cfg.cache_version = f"{cfg.cache_version}_formulafic600"
        # fic_posthoc_duration_ms 는 make_config 의 파이프라인 값(600_000) 그대로 둔다.
        # 이 값이 FIC top_k 후보 재시뮬 길이 → 넘겨줄 c_ei 선택을 바꾼다(part1_fic.py:299).
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
        network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
        mask, cap = data["sc_mask"], cfg.connectivity_weight_max
        n = data["n_nodes"]

        wl = np.asarray(cw["wLRE"][0] + cw["wLRE"][1] * FC_emp, np.float32)
        wf = np.asarray(cw["wFFI"][0] + cw["wFFI"][1] * FC_emp, np.float32)
        ps_c1 = ParamSet(c_ei=np.ones(n, np.float32), wLRE=wl, wFFI=wf,
                         c_ei_frozen=False).sanitize(mask, cap)
        bundle_init = StateBundle.from_warmup(
            warmup_result=warmup_result, bold_monitor_template=bold_monitor,
            initial_params=ps_c1,
            internal_state=capture_internal_state(initial_state),
            delay_history=capture_network_delay_history(network), stage="warmup")

        print("[FIC] 공식 가중치 위에서 S_e -> 0.25 (캐시 있으면 즉시)...", flush=True)
        bundle_fic = run_fic(network=network, bundle_in=bundle_init, cfg=cfg, data=data)
        c_fic = np.asarray(bundle_fic.params.c_ei, np.float32)

        ref, stored_corr, stored_fc = load_ref(sub)
        dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

        def sim(ps):
            b = bundle_fic.advance(new_params=ps)
            return np.asarray(compute_simulated_fc(
                network, b, cfg, sim_duration_ms=dur,
                skip_tr=cfg.optimizer_bold_skip_tr), np.float32)

        FC_formula = sim(bundle_fic.params)
        FC_optim = sim(ParamSet(c_ei=np.asarray(ref["c_ei"], np.float32),
                                wLRE=np.asarray(ref["wLRE"], np.float32),
                                wFFI=np.asarray(ref["wFFI"], np.float32),
                                c_ei_frozen=False).sanitize(mask, cap))

        r = dict(idx=idx, sub=sub)
        r["formula_vs_emp"], r["formula_vs_emp_rmse"] = pair(FC_formula, FC_emp)
        r["optim_vs_emp"], r["optim_vs_emp_rmse"] = pair(FC_optim, FC_emp)
        r["formula_vs_optim"], r["formula_vs_optim_rmse"] = pair(FC_formula, FC_optim)
        r["stored_corr"] = float(stored_corr)
        r["c_ei_fic_mean"] = float(c_fic.mean())
        r["c_ei_opt_mean"] = float(np.asarray(ref["c_ei"]).mean())
        rows.append(r)
        print(f"  FC_formula vs FC_emp    r={r['formula_vs_emp']:+.4f}  rmse={r['formula_vs_emp_rmse']:.4f}")
        print(f"  FC_optim   vs FC_emp    r={r['optim_vs_emp']:+.4f}  rmse={r['optim_vs_emp_rmse']:.4f}"
              f"   (저장값 {stored_corr:.4f})")
        print(f"  FC_formula vs FC_optim  r={r['formula_vs_optim']:+.4f}  rmse={r['formula_vs_optim_rmse']:.4f}",
              flush=True)

        np.savez_compressed(f"output_ppmi_pd/_reproduce_idx{idx}.npz",
                            FC_formula=FC_formula, FC_optim=FC_optim, FC_emp=FC_emp)
        import jax
        jax.clear_caches()

    print(f"\n{'='*96}")
    print(f"{'idx':>4} {'formula~emp':>12} {'optim~emp':>11} {'formula~optim':>14} "
          f"{'재현율':>8} {'rmse f~e':>9} {'rmse o~e':>9} {'rmse f~o':>9}")
    print("-" * 96)
    for r in rows:
        print(f"{r['idx']:>4} {r['formula_vs_emp']:>+12.4f} {r['optim_vs_emp']:>+11.4f} "
              f"{r['formula_vs_optim']:>+14.4f} "
              f"{r['formula_vs_emp']/r['optim_vs_emp']*100:>7.1f}% "
              f"{r['formula_vs_emp_rmse']:>9.4f} {r['optim_vs_emp_rmse']:>9.4f} "
              f"{r['formula_vs_optim_rmse']:>9.4f}")
    if rows:
        print("-" * 96)
        print(f"{'평균':>4} {np.mean([r['formula_vs_emp'] for r in rows]):>+12.4f} "
              f"{np.mean([r['optim_vs_emp'] for r in rows]):>+11.4f} "
              f"{np.mean([r['formula_vs_optim'] for r in rows]):>+14.4f}")
    with open("output_ppmi_pd/_reproduce.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
