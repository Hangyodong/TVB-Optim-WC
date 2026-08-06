#!/usr/bin/env python3
"""formula_2x2.py — 재현 실패의 책임이 가중치인가 c_ei 인가.

가중치 {공식, optimized} x c_ei {FIC(S_e=0.25), optimized} 2x2.
모든 조건을 같은 방식으로 600s 정착시킨 뒤 noise 실현 5개 평균으로 채점한다.

⚠ noise_level 은 파이프라인과 같은 0.02 여야 한다. additive_noise_sigma 가 cache_tag 에
없어서 0.01 로 돌리면 캐시는 정상 매칭되고 시뮬만 절반 노이즈로 돌아 corr 이 조용히 무너진다
(idx4 optimized 재현: 0.02 -> 0.7524 = 저장값 일치 / 0.01 -> 0.4253).

  A 공식가중치 + FIC c_ei        <- 0회 최적화 경로
  B 공식가중치 + optimized c_ei  <- c_ei 만 정답이면 얼마나 오르나
  C optimized가중치 + FIC c_ei   <- 가중치만 정답이면 얼마나 오르나
  D optimized + optimized        <- 표준 7시간 기준선

실행: python3 formula_2x2.py --idxs 4,5,6
"""
import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part3_gradient import _compute_fc_from_bold_output, _settle_bundle
from pipeline_contracts import ParamSet, StateBundle
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr

IU = np.triu_indices(163, 1)
SUBDIR = {0: "100001", 1: "100005", 2: "100012", 3: "100268", 4: "100878",
          5: "100889", 6: "100905", 7: "100952", 8: "101025", 9: "101038"}
EVAL_SEEDS = [42, 1, 7, 123, 2024]


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def fc_hashes(FC):
    return {hashlib.sha1(np.asarray(FC[:8, :8], t).tobytes()).hexdigest()[:10]
            for t in (np.float32, np.float64)}


def load_ref(sub):
    fp = f"output_ppmi_pd/{sub}/inputs/FC.csv"
    if not os.path.exists(fp):
        return None
    hs = fc_hashes(np.loadtxt(fp, delimiter=","))
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
            p = _get(b, "params")
            best, bc = {k: np.asarray(_get(p, k), np.float64)
                        for k in ("c_ei", "wLRE", "wFFI")}, c
    return best


def loo_coefs(target_sub):
    acc = {"wLRE": [], "wFFI": []}
    used = []
    for s in SUBDIR.values():
        if s == target_sub or not os.path.exists(f"output_ppmi_pd/{s}/inputs/weight.csv"):
            continue
        ref = load_ref(s)
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


def load_fic_cei(sub, fc_tag):
    """formula_fic.py 가 공식 가중치 위에서 구한 FIC c_ei."""
    hits = glob.glob(f"output_ppmi_pd/{sub}/cache/*formulafic*/fic_*{fc_tag}*.pkl")
    if not hits:
        return None, None
    with open(hits[0], "rb") as fh:
        d = pickle.load(fh)
    b = _get(d, "bundle", d)
    return (np.asarray(_get(_get(b, "params"), "c_ei"), np.float32),
            StateBundle.from_dict(b if isinstance(b, dict) else b.to_dict()))


def measure_se(network, bundle, cfg, dur_ms=60_000):
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    model, state = bundle.to_tvb_state(network, solver, t1=int(dur_ms),
                                       dt=cfg.integration_dt_ms)
    return float(np.mean(np.asarray(model(state).data[:, 0, :])))


def holdout(network, bundle, cfg, tgt):
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    cs = []
    for sd in EVAL_SEEDS:
        model, state = bundle.to_tvb_state(network, solver, t1=dur,
                                           dt=cfg.integration_dt_ms)
        monitor = bundle.build_bold_monitor(cfg)
        ns = jnp.asarray(state._internal.noise_samples)
        _, subkey = jax.random.split(jax.random.PRNGKey(int(sd)))
        state._internal.noise_samples = jax.random.normal(subkey, ns.shape, ns.dtype)
        fc = np.asarray(_compute_fc_from_bold_output(
            monitor(model(state)), cfg.optimizer_bold_skip_tr), np.float32)
        cs.append(float(fc_corr(fc, tgt)))
    return float(np.mean(cs)), float(np.std(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", type=str, default="4,5,6")
    ap.add_argument("--noise-level", type=float, default=0.02)
    a = ap.parse_args()

    rows = []
    for idx in [int(x) for x in a.idxs.split(",") if x.strip()]:
        sub = SUBDIR[idx]
        cw, used = loo_coefs(sub)
        print(f"\n{'='*78}\nidx {idx} ({sub})  LOO 계수 (from {used}):")
        print(f"  wLRE = {cw['wLRE'][0]:+.4f} {cw['wLRE'][1]:+.4f}*FC")
        print(f"  wFFI = {cw['wFFI'][0]:+.4f} {cw['wFFI'][1]:+.4f}*FC", flush=True)

        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)
        cfg.cache_version = f"{cfg.cache_version}_2x2"
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        tgt = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
        network, *_ = build_network(cfg, data)
        mask, cap = data["sc_mask"], cfg.connectivity_weight_max
        t1 = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

        fc_tag = data["cache_tag"].split("_")[-1]
        c_fic, base_bundle = load_fic_cei(sub, fc_tag)
        ref = load_ref(sub)
        if c_fic is None or ref is None:
            print(f"  [skip] FIC 캐시 또는 grad 캐시 없음")
            continue
        c_opt = np.asarray(ref["c_ei"], np.float32)
        wl_f = np.asarray(cw["wLRE"][0] + cw["wLRE"][1] * tgt, np.float32)
        wf_f = np.asarray(cw["wFFI"][0] + cw["wFFI"][1] * tgt, np.float32)
        wl_o = np.asarray(ref["wLRE"], np.float32)
        wf_o = np.asarray(ref["wFFI"], np.float32)

        conds = {
            "A 공식 w + FIC c_ei":   (wl_f, wf_f, c_fic),
            "B 공식 w + opt c_ei":   (wl_f, wf_f, c_opt),
            "C opt w + FIC c_ei":    (wl_o, wf_o, c_fic),
            "D opt w + opt c_ei":    (wl_o, wf_o, c_opt),
        }
        print(f"  c_ei: FIC mean={c_fic.mean():.3f}  opt mean={c_opt.mean():.3f}")
        print(f"  {'조건':<22} {'hold-out corr':>16} {'S_e':>8}")
        for nm, (wl, wf, ce) in conds.items():
            ps = ParamSet(c_ei=ce, wLRE=wl, wFFI=wf,
                          c_ei_frozen=False).sanitize(mask, cap)
            b = base_bundle.advance(new_params=ps)
            b, _, _ = _settle_bundle(network, b, cfg, sim_duration_ms=t1,
                                     skip_tr=cfg.optimizer_bold_skip_tr,
                                     next_stage="eib")
            m, s = holdout(network, b, cfg, tgt)
            se = measure_se(network, b, cfg)
            rows.append(dict(idx=idx, sub=sub, cond=nm, corr=m, sd=s, se=se))
            print(f"  {nm:<22} {m:+.4f} +- {s:.4f}  {se:>8.4f}", flush=True)

        import jax as _j
        _j.clear_caches()

    print(f"\n{'='*78}\n요약 (hold-out, noise 실현 5개 평균)\n{'='*78}")
    print(f"{'idx':>4} {'조건':<22} {'corr':>9} {'sd':>7} {'S_e':>8}")
    for r in rows:
        print(f"{r['idx']:>4} {r['cond']:<22} {r['corr']:>+9.4f} {r['sd']:>7.4f} {r['se']:>8.4f}")
    for c in ["A 공식 w + FIC c_ei", "B 공식 w + opt c_ei",
              "C opt w + FIC c_ei", "D opt w + opt c_ei"]:
        v = [r["corr"] for r in rows if r["cond"] == c]
        if v:
            print(f"  평균 {c:<22} {np.mean(v):+.4f}")

    os.makedirs("output_ppmi_pd/_2x2", exist_ok=True)
    with open("output_ppmi_pd/_2x2/results.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
