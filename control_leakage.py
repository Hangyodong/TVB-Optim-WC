#!/usr/bin/env python3
"""control_leakage.py — 최적화된 wLRE/wFFI 가 경험 FC 의 폐형식 함수인지 검증.

측정 결과: r(wLRE_edge, FC_emp_edge) = 0.94~0.99 (subject 내), R2 = 0.89~0.97.
EIB 규칙이 wLRE += eta*(FC_emp - FC_sim)*rmse 이므로 이는 규칙의 적분 결과일 수
있다. 그렇다면 7시간 최적화는 계수 2개짜리 공식을 재발견하는 것이고, 동시에
경험 FC 가 커플링에 직접 기입되므로 보고된 corr 의 해석이 달라진다.

4개 조건을 동일 network / 동일 c_ei / 동일 init_dynamics 로 각 1회 정방향 시뮬:

  optimized     Part3 최종 wLRE/wFFI                        (기준선, 재현 확인)
  formula_self  wLRE = a + b*FC_emp(본인)   계수는 나머지 subject 로 LOO 적합
                (실제 풀 = idx 4/5/6 3명뿐. 나머지 7명의 grad 캐시는 전부 2026-07-28
                 이전 .mat 이라 FC 해시가 안 맞아 load_best_grad 에서 걸러진다
                 → LOO = 본인 제외 "2명" 평균이고, 계수 풀과 평가셋이 동일 3명이다)
  formula_other 동일 공식에 *다른 subject* 의 FC_emp        (귀무 대조)
  identity      wLRE = wFFI = 1 (파이프라인 초기값)         (하한)

formula_self 가 optimized 에 근접하면 최적화 대부분이 공식으로 대체 가능하다.
formula_other 가 formula_self 에 근접하면 subject 특이성이 없다는 뜻이다.

실행: python3 control_leakage.py --idxs 4,5,6
"""
import argparse
import gc
import glob
import os
import pickle

import numpy as np

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from pipeline_contracts import ParamSet, StateBundle
from part3_gradient import compute_simulated_fc
from tvboptim.observations.observation import fc_corr

IU = np.triu_indices(163, 1)
# subject_idx -> 출력 폴더명 (cache_version 이 v_pdaal163_tr25_s<idx>)
SUBDIR = {0: "100001", 1: "100005", 2: "100012", 3: "100268", 4: "100878",
          5: "100889", 6: "100905", 7: "100952", 8: "101025", 9: "101038"}


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def fc_hashes(FC):
    """data_loader._build_cache_tag 의 fc 지문 (float32/64 양쪽)."""
    import hashlib
    return {hashlib.sha1(np.asarray(FC[:8, :8], t).tobytes()).hexdigest()[:10]
            for t in (np.float32, np.float64)}


def load_best_grad(sub_dir):
    """현재 FC.csv 와 해시가 일치하는 캐시 중 post_grad_fc_corr 최대.

    .mat 이 2026-07-28 교체돼 그 이전 런은 다른 FC 를 타깃으로 최적화됐다.
    config/STEPS/delay 플래그로 거르면 구 FC 런이 섞여 R2 가 0.94->0.66 으로 희석된다.
    """
    fp = f"output_ppmi_pd/{sub_dir}/inputs/FC.csv"
    if not os.path.exists(fp):
        return None, -9.0
    hs = fc_hashes(np.loadtxt(fp, delimiter=","))
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub_dir}/cache/*/grad_*.pkl"):
        if not any(h in os.path.basename(os.path.dirname(f)) for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle") if (isinstance(d, dict) and "bundle" in d) else d
        md = _get(b, "metadata", {}) or {}
        c = float(md.get("post_grad_fc_corr", float("nan")))
        if np.isfinite(c) and c > bc:
            best, bc = (b, f), c
        del d
    return best, bc


def fit_coefs(sub_dirs):
    """subject 별 wLRE = a + b*FC_emp, wFFI = a2 + b2*FC_emp 적합 → 계수 dict."""
    out = {}
    for s in sub_dirs:
        got, _ = load_best_grad(s)
        if got is None:
            continue
        b, _f = got
        p = _get(b, "params")
        W = np.loadtxt(f"output_ppmi_pd/{s}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"output_ppmi_pd/{s}/inputs/FC.csv", delimiter=",")
        m = (W > 0)[IU]
        x = FCe[IU][m]
        A = np.stack([x, np.ones_like(x)], 1)
        cf = {}
        for nm in ("wLRE", "wFFI"):
            y = np.asarray(_get(p, nm), np.float64)[IU][m]
            (bb, aa), *_ = np.linalg.lstsq(A, y, rcond=None)
            cf[nm] = (float(aa), float(bb))
        out[s] = cf
    return out


def formula_params(fc_src, coefs, c_ei, sc_mask, cap):
    """wLRE/wFFI 를 a + b*FC 로 채운 ParamSet (sanitize 가 마스킹/클립/대칭화)."""
    aL, bL = coefs["wLRE"]
    aF, bF = coefs["wFFI"]
    return ParamSet(
        c_ei=np.asarray(c_ei, np.float32),
        wLRE=np.asarray(aL + bL * fc_src, np.float32),
        wFFI=np.asarray(aF + bF * fc_src, np.float32),
        c_ei_frozen=False,
    ).sanitize(sc_mask, cap)


# 합/차 분해 비선형식 (7명 전체 적합):
#   S  = wLRE + wFFI = 2.062 (상수 취급)
#   Dl = wLRE - wFFI = -0.6287 + 5.6917*FC - 0.7346*FC^2   (LOO R2 = 0.880)
NL_S, NL_D = 2.0618, (-0.6287, 5.6917, -0.7346)


def formula_params_nl(fc_src, c_ei, sc_mask, cap):
    """비선형(합/차 분해) 버전. 계수 4개로 wLRE·wFFI 둘 다 생성."""
    a, b, c = NL_D
    dl = a + b * fc_src + c * fc_src ** 2
    return ParamSet(
        c_ei=np.asarray(c_ei, np.float32),
        wLRE=np.asarray((NL_S + dl) / 2.0, np.float32),
        wFFI=np.asarray((NL_S - dl) / 2.0, np.float32),
        c_ei_frozen=False,
    ).sanitize(sc_mask, cap)


def score(network, bundle, cfg, target):
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    fc = np.asarray(compute_simulated_fc(network, bundle, cfg,
                                         sim_duration_ms=dur,
                                         skip_tr=cfg.optimizer_bold_skip_tr))
    tgt = np.nan_to_num(np.asarray(target, np.float32))
    corr = float(fc_corr(fc.astype(np.float32), tgt))
    rmse = float(np.sqrt(np.mean((fc[IU] - tgt[IU]) ** 2)))
    return corr, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", type=str, default="4,5,6")
    ap.add_argument("--noise-level", type=float, default=0.02)
    a = ap.parse_args()
    idxs = [int(x) for x in a.idxs.split(",") if x.strip()]

    all_dirs = [d for d in SUBDIR.values()
                if os.path.exists(f"output_ppmi_pd/{d}/inputs/weight.csv")]
    print(f"[fit] subject 별 a + b*FC_emp 계수 적합 ({len(all_dirs)}명)...")
    COEF = fit_coefs(all_dirs)
    for s, c in COEF.items():
        print(f"  {s}: wLRE a={c['wLRE'][0]:+.3f} b={c['wLRE'][1]:+.3f} | "
              f"wFFI a={c['wFFI'][0]:+.3f} b={c['wFFI'][1]:+.3f}")

    rows = []
    for idx in idxs:
        sub = SUBDIR[idx]
        if sub not in COEF:
            print(f"[idx {idx}] 계수 없음 → skip")
            continue
        others = [s for s in COEF if s != sub]
        # leave-one-subject-out 계수 (본인 제외 평균)
        loo = {nm: (float(np.mean([COEF[o][nm][0] for o in others])),
                    float(np.mean([COEF[o][nm][1] for o in others])))
               for nm in ("wLRE", "wFFI")}

        print(f"\n{'='*84}\nidx {idx} (subject {sub})   LOO 계수: "
              f"wLRE={loo['wLRE'][0]:+.3f}{loo['wLRE'][1]:+.3f}*FC  "
              f"wFFI={loo['wFFI'][0]:+.3f}{loo['wFFI'][1]:+.3f}*FC\n{'='*84}")

        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)   # 2026-07-28 이후 파이프라인 기본값
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

        got, stored = load_best_grad(sub)
        bdict, path = got
        bundle = StateBundle.from_dict(bdict)
        op = bundle.params
        c_ei = np.asarray(op.c_ei, np.float32)
        print(f"  캐시: {os.path.basename(path)}  저장된 post_grad_corr={stored:.4f}")

        tgt = data["fc_target"]
        mask, cap = data["sc_mask"], cfg.connectivity_weight_max
        # 귀무 대조용 다른 subject FC (본인 다음 인덱스)
        other = others[(others.index(sub) + 1) % len(others)] if sub in others else others[0]
        fc_other = np.loadtxt(f"output_ppmi_pd/{other}/inputs/FC.csv", delimiter=",")
        np.fill_diagonal(fc_other, 0.0)

        # 본인 계수로 재구성 = 최적 가중치를 span{1, FC_emp} 에 정사영한 것.
        # optimized 와의 차이 = 잔차(R2 의 나머지 6%)가 FC 적합에 기여하는 몫.
        fp_own = formula_params(tgt, COEF[sub], c_ei, mask, cap)
        half = ParamSet(c_ei=c_ei,
                        wLRE=0.5 * (np.asarray(op.wLRE) + fp_own.wLRE),
                        wFFI=0.5 * (np.asarray(op.wFFI) + fp_own.wFFI),
                        c_ei_frozen=False).sanitize(mask, cap)

        conds = {
            "optimized": ParamSet(c_ei=c_ei, wLRE=op.wLRE, wFFI=op.wFFI,
                                  c_ei_frozen=False).sanitize(mask, cap),
            "optimized (rerun, noise floor)": ParamSet(
                c_ei=c_ei, wLRE=op.wLRE, wFFI=op.wFFI,
                c_ei_frozen=False).sanitize(mask, cap),
            "formula_own (own coef)": fp_own,
            "formula_own + 50% residual": half,
            "formula_self (linear)": formula_params(tgt, loo, c_ei, mask, cap),
            "formula_self (nonlinear)": formula_params_nl(tgt, c_ei, mask, cap),
            f"formula_other({other})": formula_params(fc_other, loo, c_ei, mask, cap),
            "identity": ParamSet(c_ei=c_ei,
                                 wLRE=np.ones((163, 163), np.float32),
                                 wFFI=np.ones((163, 163), np.float32),
                                 c_ei_frozen=False).sanitize(mask, cap),
        }
        for nm, ps in conds.items():
            corr, rmse = score(network, bundle.advance(new_params=ps), cfg, tgt)
            rows.append((idx, sub, nm, corr, rmse))
            print(f"    {nm:<24} corr={corr:+.4f}  rmse={rmse:.4f}")

        import jax
        jax.clear_caches()
        del network, initial_state, bold_monitor, warmup_result, data
        gc.collect()

    print("\n" + "=" * 84)
    print("요약  (동일 c_ei / 동일 init_dynamics, wLRE·wFFI 만 교체)")
    print("=" * 84)
    print(f"{'idx':>4} {'subject':>9} {'condition':>24} {'corr':>8} {'rmse':>8}")
    for r in rows:
        print(f"{r[0]:>4} {r[1]:>9} {r[2]:>24} {r[3]:>+8.4f} {r[4]:>8.4f}")


if __name__ == "__main__":
    main()
