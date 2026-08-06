#!/usr/bin/env python3
"""formula_warmstart.py — 공식을 최적화 '대체'가 아니라 '출발점'으로 쓴다.

control_leakage.py / formula_fic.py 는 공식 가중치를 그대로 채점했다 (0회 최적화).
여기서는 공식으로 wLRE/wFFI 를 edge 별로 채운 뒤 그 지점에서 Part3 gradient 를 돌린다.

  wLRE = a_L + b_L * FC_emp,  wFFI = a_F + b_F * FC_emp   (LOO 계수)
    -> FIC (S_e -> 0.25) 로 c_ei          [formula_fic.py 가 만든 캐시 재사용]
    -> Part3 gradient N steps             [EIB 10000 step 을 통째로 건너뜀]

근거: EIB 갱신식 wLRE += eta*(FC_emp - FC_sim)*rmse 를 wLRE=1 에서 적분하면 FC_emp 의
아핀함수다. 즉 공식 초기값 = EIB 종점 근사 -> EIB 단계가 잉여가 된다.

⚠ noise_level 은 반드시 파이프라인과 같은 0.02 여야 한다 (main_ppmi_pd.py 의 CLI 기본값).
additive_noise_sigma 가 cache_tag 에 안 들어가서, 0.01 로 돌리면 캐시는 정상 매칭되고
시뮬만 절반 노이즈로 돌아 corr 이 0.75 -> 0.43 으로 조용히 무너진다.

채점은 noise 실현 5개 평균이다 (단일 draw 의 분산 제거용).

실행: python3 formula_warmstart.py --idx 4 --steps 100
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
from part3_gradient import (
    _compute_fc_from_bold_output, _settle_bundle, run_gradient_optimization,
)
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
    """현재 FC.csv 해시와 일치하는 grad 캐시 중 최고 corr → params dict."""
    fp = f"output_ppmi_pd/{sub}/inputs/FC.csv"
    if not os.path.exists(fp):
        return None
    hs = fc_hashes(np.loadtxt(fp, delimiter=","))
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        if "formulafic" in f or "fwarm" in f:
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
    """target 제외 FC-매칭 subject 로 a + b*FC_emp 계수 적합 (평균)."""
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
    if not used:
        raise SystemExit(f"[{target_sub}] FC-매칭 참조 subject 없음")
    print(f"[계수 적합 subject] {used}")
    return {nm: tuple(np.mean(acc[nm], axis=0)) for nm in acc}


def load_formulafic_bundle(sub, data):
    """formula_fic.py 가 남긴 FIC 캐시(공식 가중치 위의 c_ei) 재사용."""
    fc_tag = data["cache_tag"].split("_")[-1]      # fc<해시>
    hits = glob.glob(f"output_ppmi_pd/{sub}/cache/*formulafic*/fic_*{fc_tag}*.pkl")
    for f in hits:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle", d)
        return StateBundle.from_dict(b if isinstance(b, dict) else b.to_dict()), f
    return None, None


def holdout_corr(network, bundle, cfg, tgt):
    """noise 실현 5개 평균 corr — 저장 metadata 의 train corr 대신 쓸 값."""
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
    return float(np.mean(cs)), float(np.std(cs)), cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--noise-level", type=float, default=0.02)
    a = ap.parse_args()
    sub = SUBDIR[a.idx]

    cw = loo_coefs(sub)
    print(f"[LOO] wLRE = {cw['wLRE'][0]:+.4f} {cw['wLRE'][1]:+.4f}*FC")
    print(f"[LOO] wFFI = {cw['wFFI'][0]:+.4f} {cw['wFFI'][1]:+.4f}*FC")

    p = M.prepare_pd_data(a.idx, a.noise_level)
    cfg = M.make_config(p, a.idx, use_delay=True)
    cfg.optimizer_max_steps = a.steps
    base_version = cfg.cache_version
    cfg.cache_version = f"{base_version}_fwarm"
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    tgt = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, _, _, _ = build_network(cfg, data)
    mask, cap = data["sc_mask"], cfg.connectivity_weight_max

    # ── 공식 가중치 + formula_fic 이 구한 FIC c_ei ──
    fic_bundle, fic_path = load_formulafic_bundle(sub, data)
    if fic_bundle is None:
        raise SystemExit("formulafic FIC 캐시 없음 — formula_fic.py 를 먼저 실행")
    print(f"[FIC 캐시] {os.path.basename(fic_path)}")
    c_ei = np.asarray(fic_bundle.params.c_ei, np.float32)
    print(f"[FIC c_ei] mean={c_ei.mean():.4f} range=[{c_ei.min():.3f},{c_ei.max():.3f}]")

    ps_start = ParamSet(
        c_ei=c_ei,
        wLRE=np.asarray(cw["wLRE"][0] + cw["wLRE"][1] * tgt, np.float32),
        wFFI=np.asarray(cw["wFFI"][0] + cw["wFFI"][1] * tgt, np.float32),
        c_ei_frozen=False).sanitize(mask, cap)
    # formula_fic 의 FIC 캐시는 posthoc 240s 로 돌아 bold_window 가 (96,163) 이다.
    # Part3 는 t1_opt=600s ↔ 240 TR window 를 기대하므로 그대로 넣으면 BOLD convolve
    # 가 부풀어 AD tape 이 30GB 를 요구하며 OOM 난다. 600s 정착으로 window 를 맞춘다.
    bundle_start = fic_bundle.advance(new_params=ps_start)
    t1_opt = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    print(f"[settle] bold_window {np.asarray(bundle_start.bold_window).shape} "
          f"→ {t1_opt/1000:.0f}s 정착...", flush=True)
    bundle_start, _, _ = _settle_bundle(
        network, bundle_start, cfg, sim_duration_ms=t1_opt,
        skip_tr=cfg.optimizer_bold_skip_tr, next_stage="eib")
    print(f"[settle] → {np.asarray(bundle_start.bold_window).shape}")

    m0, s0, _ = holdout_corr(network, bundle_start, cfg, tgt)
    print(f"\n[출발점] 공식+FIC hold-out corr = {m0:+.4f} +- {s0:.4f}")

    # ── Part3 gradient (EIB 건너뜀) ──
    # steps=0 이면 생략. A10(23GB) 에서는 Part3 가 delay 버퍼 스택(600k step x 163 x 84
    # x 4B = 30.6GB)으로 OOM 난다 — 표준 EIB bundle 로도 동일. 더 큰 GPU 필요.
    bundle_grad = bundle_start
    train_corr, m1, s1, cs1 = float("nan"), m0, s0, []
    if a.steps > 0:
        print(f"\n[Part3] {a.steps} step 시작 (EIB 생략)...", flush=True)
        bundle_grad = run_gradient_optimization(
            network=network, bundle_in=bundle_start, cfg=cfg, data=data)
        train_corr = float((_get(bundle_grad, "metadata", {}) or {}).get(
            "post_grad_fc_corr", np.nan))
        m1, s1, cs1 = holdout_corr(network, bundle_grad, cfg, tgt)
        print(f"\n[결과] warm-start Part3  train corr={train_corr:+.4f}  "
              f"hold-out={m1:+.4f} +- {s1:.4f}")

    # ── 기준선: 표준 경로(FIC->EIB->Part3) 결과의 hold-out ──
    ref = load_ref(sub)
    m2 = s2 = float("nan")
    if ref is not None:
        b_ref = bundle_grad.advance(new_params=ParamSet(
            c_ei=np.asarray(ref["c_ei"], np.float32),
            wLRE=np.asarray(ref["wLRE"], np.float32),
            wFFI=np.asarray(ref["wFFI"], np.float32),
            c_ei_frozen=False).sanitize(mask, cap))
        m2, s2, _ = holdout_corr(network, b_ref, cfg, tgt)

    print(f"\n{'='*70}")
    print(f"idx {a.idx} ({sub})  steps={a.steps}")
    print(f"  공식+FIC (0 step)            hold-out {m0:+.4f} +- {s0:.4f}")
    print(f"  + Part3 {a.steps} step (EIB 생략)  hold-out {m1:+.4f} +- {s1:.4f}")
    print(f"  표준 FIC->EIB->Part3         hold-out {m2:+.4f} +- {s2:.4f}")
    print("=" * 70)

    os.makedirs("output_ppmi_pd/_fwarm", exist_ok=True)
    with open(f"output_ppmi_pd/_fwarm/idx{a.idx}.json", "w") as fh:
        json.dump(dict(idx=a.idx, sub=sub, steps=a.steps,
                       start=m0, start_sd=s0,
                       warm=m1, warm_sd=s1, warm_train=train_corr,
                       standard=m2, standard_sd=s2, warm_seeds=cs1), fh, indent=1)


if __name__ == "__main__":
    main()
