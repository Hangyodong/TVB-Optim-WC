#!/usr/bin/env python3
"""formula_fic.py — "수식 2개 + S_e=0.25" 만으로 어디까지 맞는가.

control_leakage.py 는 c_ei 를 7시간 Part3 결과에서 가져다 썼다 → 반칙이었다.
여기서는 진짜 0-최적화 경로를 검증한다:

  1. wLRE/wFFI 를 FC_emp 선형식으로 채운다 (최적화 0회)
       wLRE = a_L + b_L * FC_emp,  wFFI = a_F + b_F * FC_emp    (LOO 계수)
  2. 그 가중치 위에서 FIC 를 돌려 S_e -> 0.25 를 맞추는 c_ei 를 찾는다 (~25분)
  3. 정방향 시뮬 1회로 FC corr / rmse 측정

c_ei 가 wLRE 에 의존하므로 순서가 중요하다. 기존 캐시의 c_ei(=wLRE 1.0 기준으로
튜닝됨)를 그대로 쓰면 안 된다.

비교 조건 (모두 동일 network):
  formula+FIC       공식 가중치 + FIC 로 얻은 c_ei          <- 핵심 답
  formula+reg       공식 가중치 + FC-strength 회귀 c_ei     (FIC 도 생략, 즉시)
  formula+cei1      공식 가중치 + c_ei=1.0                  (S_e 제어 없음)
  identity+FIC      wLRE=wFFI=1 + 같은 FIC c_ei             (가중치 기여 분리)

캐시는 cache_version 에 _formulafic 접미사를 붙여 기존 결과와 격리한다.

실행: python3 formula_fic.py --idx 4
"""
import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np
import jax.numpy as jnp

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr

IU = np.triu_indices(163, 1)
SUBDIR = {0: "100001", 1: "100005", 2: "100012", 3: "100268", 4: "100878",
          5: "100889", 6: "100905", 7: "100952", 8: "101025", 9: "101038"}


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def fc_hashes(FC):
    """data_loader._build_cache_tag 의 fc 지문 (float32/64 양쪽)."""
    return {hashlib.sha1(np.asarray(FC[:8, :8], t).tobytes()).hexdigest()[:10]
            for t in (np.float32, np.float64)}


def load_ref(sub):
    """현재 FC.csv 와 해시가 일치하는 grad 캐시 중 최고 corr → (params, corr).

    .mat 이 2026-07-28 교체돼 그 이전 런은 다른 FC 를 타깃으로 최적화됐다.
    캐시 태그의 fc 해시가 현재 FC.csv 와 맞는 런만 유효하다 (= use_delay 기본 ON 이후).
    """
    fp = f"output_ppmi_pd/{sub}/inputs/FC.csv"
    if not os.path.exists(fp):
        return None, float("nan")
    hs = fc_hashes(np.loadtxt(fp, delimiter=","))
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        tag = os.path.basename(os.path.dirname(f))
        if not any(h in tag for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle") if (isinstance(d, dict) and "bundle" in d) else d
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = _get(b, "params")
            best, bc = dict(c_ei=np.asarray(_get(p, "c_ei"), np.float64),
                            wLRE=np.asarray(_get(p, "wLRE"), np.float64),
                            wFFI=np.asarray(_get(p, "wFFI"), np.float64)), c
        del d, b
    return best, bc


def loo_coefs(target_sub):
    """target_sub 를 제외한 FC-매칭 subject 로 계수 적합.

    매칭 subject 가 현재 3명뿐이라 실질 n=2 적합이다. 계수 산포가 매우 작아
    (a sd=0.02, b sd=0.14) 실용상 문제없지만 held-out 강도는 약하다.
    """
    accW, accC, used = {"wLRE": [], "wFFI": []}, [], []
    for idx, s in SUBDIR.items():
        if s == target_sub or not os.path.exists(f"output_ppmi_pd/{s}/inputs/weight.csv"):
            continue
        ref, _ = load_ref(s)
        if ref is None:
            continue
        used.append(s)
        W = np.loadtxt(f"output_ppmi_pd/{s}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"output_ppmi_pd/{s}/inputs/FC.csv", delimiter=",")
        m = (W > 0)[IU]
        x = FCe[IU][m]
        A = np.stack([x, np.ones_like(x)], 1)
        for nm in ("wLRE", "wFFI"):
            (b, a), *_ = np.linalg.lstsq(A, ref[nm][IU][m], rcond=None)
            accW[nm].append((float(a), float(b)))
        fs = np.abs(FCe).sum(1)
        z = (fs - fs.mean()) / fs.std()
        (b, a), *_ = np.linalg.lstsq(np.stack([z, np.ones_like(z)], 1), ref["c_ei"], rcond=None)
        accC.append((float(a), float(b)))
    if not used:
        raise SystemExit(f"[{target_sub}] FC-매칭된 참조 subject 없음 → 계수 적합 불가")
    print(f"[계수 적합에 쓴 subject] {used}")
    return ({nm: tuple(np.mean(accW[nm], axis=0)) for nm in accW},
            tuple(np.mean(accC, axis=0)))


def measure_se(network, bundle, cfg, dur_ms):
    """평균 S_e (상태변수 0) — FIC 루프가 쓰는 정의와 동일."""
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    step_model, state = bundle.to_tvb_state(network, solver, t1=int(dur_ms),
                                            dt=cfg.integration_dt_ms)
    res = step_model(state)
    return float(np.mean(np.asarray(res.data[:, 0, :])))


def score(network, bundle, cfg, target):
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    fc = np.asarray(compute_simulated_fc(network, bundle, cfg, sim_duration_ms=dur,
                                         skip_tr=cfg.optimizer_bold_skip_tr))
    tgt = np.nan_to_num(np.asarray(target, np.float32))
    return (float(fc_corr(fc.astype(np.float32), tgt)),
            float(np.sqrt(np.mean((fc[IU] - tgt[IU]) ** 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--posthoc-ms", type=int, default=240_000,
                    help="FIC 후보 재시뮬 길이 축소(기본 600s→240s). 최종 채점은 별도 600s.")
    a = ap.parse_args()
    idx = a.idx
    sub = SUBDIR[idx]

    cw, cc = loo_coefs(sub)
    print(f"[LOO 계수] wLRE = {cw['wLRE'][0]:+.4f} {cw['wLRE'][1]:+.4f}*FC")
    print(f"           wFFI = {cw['wFFI'][0]:+.4f} {cw['wFFI'][1]:+.4f}*FC")
    print(f"           c_ei = {cc[0]:+.4f} {cc[1]:+.4f}*z(FC_strength)")

    p = M.prepare_pd_data(idx, a.noise_level)
    cfg = M.make_config(p, idx, use_delay=True)   # 2026-07-28 이후 파이프라인 기본값
    cfg.cache_version = f"{cfg.cache_version}_formulafic"   # 기존 캐시와 격리
    cfg.fic_posthoc_duration_ms = a.posthoc_ms
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)

    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    tgt, mask, cap = data["fc_target"], data["sc_mask"], cfg.connectivity_weight_max
    n = data["n_nodes"]

    # ── 공식 가중치 (최적화 0회) ──
    wl = cw["wLRE"][0] + cw["wLRE"][1] * tgt
    wf = cw["wFFI"][0] + cw["wFFI"][1] * tgt
    fs = np.abs(tgt).sum(1)
    z = (fs - fs.mean()) / fs.std()
    c_reg = np.clip(cc[0] + cc[1] * z, 0.0, 20.0)

    ps_formula_c1 = ParamSet(c_ei=np.ones(n, np.float32),
                             wLRE=np.asarray(wl, np.float32),
                             wFFI=np.asarray(wf, np.float32),
                             c_ei_frozen=False).sanitize(mask, cap)

    bundle_init = StateBundle.from_warmup(
        warmup_result=warmup_result, bold_monitor_template=bold_monitor,
        initial_params=ps_formula_c1,
        internal_state=capture_internal_state(initial_state),
        delay_history=capture_network_delay_history(network), stage="warmup")

    # ── FIC: 공식 가중치 위에서 S_e -> 0.25 인 c_ei 탐색 ──
    print(f"\n[FIC] 공식 가중치 위에서 실행 (target S_e={cfg.fic_target_se})...")
    bundle_fic = run_fic(network=network, bundle_in=bundle_init, cfg=cfg, data=data)
    c_fic = np.asarray(bundle_fic.params.c_ei, np.float64)
    print(f"[FIC] c_ei: mean={c_fic.mean():.4f} sd={c_fic.std():.4f} "
          f"range=[{c_fic.min():.3f},{c_fic.max():.3f}]")

    ref, ref_corr = load_ref(sub)
    ones = np.ones((n, n), np.float32)
    conds = {
        "formula + FIC c_ei": bundle_fic.params,
        "formula + reg c_ei": ParamSet(c_ei=np.asarray(c_reg, np.float32),
                                       wLRE=np.asarray(wl, np.float32),
                                       wFFI=np.asarray(wf, np.float32),
                                       c_ei_frozen=False).sanitize(mask, cap),
        "formula + c_ei=1": ps_formula_c1,
        "identity + FIC c_ei": ParamSet(c_ei=np.asarray(c_fic, np.float32),
                                        wLRE=ones, wFFI=ones,
                                        c_ei_frozen=False).sanitize(mask, cap),
    }
    if ref is not None:
        conds["optimized (7h ref)"] = ParamSet(
            c_ei=np.asarray(ref["c_ei"], np.float32),
            wLRE=np.asarray(ref["wLRE"], np.float32),
            wFFI=np.asarray(ref["wFFI"], np.float32),
            c_ei_frozen=False).sanitize(mask, cap)

    print(f"\n{'='*78}\nidx {idx} (subject {sub})   저장된 Part3 corr = {ref_corr:.4f}\n{'='*78}")
    print(f"{'condition':>22} {'corr':>8} {'rmse':>8} {'mean S_e':>9} {'c_ei mean':>10}")
    out = []
    for nm, ps in conds.items():
        b = bundle_fic.advance(new_params=ps)
        corr, rmse = score(network, b, cfg, tgt)
        se = measure_se(network, b, cfg, 60_000)
        cm = float(np.asarray(ps.c_ei).mean())
        out.append(dict(idx=idx, sub=sub, cond=nm, corr=corr, rmse=rmse, se=se, c_ei=cm))
        print(f"{nm:>22} {corr:>+8.4f} {rmse:>8.4f} {se:>9.4f} {cm:>10.4f}")

    os.makedirs("output_ppmi_pd/_formula_fic", exist_ok=True)
    with open(f"output_ppmi_pd/_formula_fic/idx{idx}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n저장: output_ppmi_pd/_formula_fic/idx{idx}.json")


if __name__ == "__main__":
    main()
