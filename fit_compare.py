#!/usr/bin/env python3
"""모형 비교를 공정하게: 계수 수 보정(adj R², AIC) + subject-LOO 외삽 R².

in-sample R² 는 계수가 많을수록 무조건 오른다. 목표가 '새 subject 예측' 이므로
4명으로 적합 → 남은 1명 예측이 진짜 기준.
"""
import glob
import hashlib
import os
import pickle

import numpy as np
from scipy.optimize import least_squares

IU = np.triu_indices(163, 1)
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025"}
WMAX = 10.0


def load_ref(sub):
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
            best = {k: np.asarray(g(p, k), np.float64) for k in ("wLRE", "wFFI")}
            bc = c
    return best, bc


def r2(y, yh):
    return 1.0 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


# 모형: (이름, 계수개수, fit(x,y)->theta, pred(theta,x))
def mk_poly(deg):
    return (deg + 1,
            lambda x, y: np.polyfit(x, y, deg),
            lambda t, x: np.polyval(t, x))


def hinge_fit(x, y):
    def res(p):
        return np.clip(p[0] + p[1] * x, 0.0, WMAX) - y
    p0 = np.polyfit(x, y, 1)[::-1]
    return least_squares(res, p0, method="lm", max_nfev=20000).x


def hinge3_fit(x, y):
    """clip(3차, 0, WMAX) — 곡률과 제약을 둘 다 넣은 모형."""
    def res(p):
        return np.clip(np.polyval(p, x), 0.0, WMAX) - y
    return least_squares(res, np.polyfit(x, y, 3), method="lm", max_nfev=40000).x


MODELS = {
    "선형(2)": mk_poly(1),
    "2차(3)": mk_poly(2),
    "3차(4)": mk_poly(3),
    "hinge(2)": (2, hinge_fit, lambda t, x: np.clip(t[0] + t[1] * x, 0.0, WMAX)),
    "hinge3(4)": (4, hinge3_fit, lambda t, x: np.clip(np.polyval(t, x), 0.0, WMAX)),
}

ROWS = {}
for idx, sub in SUBDIR.items():
    ref, _ = load_ref(sub)
    if ref is None:
        continue
    W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
    FC = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
    np.fill_diagonal(FC, 0.0)
    m = (W > 0)[IU]
    ROWS[idx] = dict(x=FC[IU][m], wLRE=ref["wLRE"][IU][m], wFFI=ref["wFFI"][IU][m])

for pname in ("wLRE", "wFFI"):
    print(f"\n{'='*96}\n### {pname}\n{'='*96}")
    print(f"{'모형':>10} {'k':>2} | {'in-sample R²':>12} {'adj R²':>9} {'ΔAIC':>9} | "
          f"{'LOO 외삽 R² (subject별)':>34} {'평균':>8}")
    print("-" * 96)
    aics = {}
    res_tbl = {}
    for name, (k, fit, pred) in MODELS.items():
        ins, adjs, aic_l, loo = [], [], [], []
        for idx, d in ROWS.items():
            x, y = d["x"], d[pname]
            n = x.size
            th = fit(x, y)
            rr = r2(y, pred(th, x))
            ins.append(rr)
            adjs.append(1 - (1 - rr) * (n - 1) / (n - k - 1))
            rss = np.sum((y - pred(th, x)) ** 2)
            aic_l.append(n * np.log(rss / n) + 2 * k)
            # LOO: 다른 4명 통합으로 적합 → 이 subject 예측
            ox = np.concatenate([ROWS[j]["x"] for j in ROWS if j != idx])
            oy = np.concatenate([ROWS[j][pname] for j in ROWS if j != idx])
            loo.append(r2(y, pred(fit(ox, oy), x)))
        aics[name] = np.mean(aic_l)
        res_tbl[name] = (np.mean(ins), np.mean(adjs), loo, np.mean(loo))
    base = min(aics.values())
    for name, (k, _, _) in [(n_, MODELS[n_]) for n_ in MODELS]:
        mi, ma, loo, ml = res_tbl[name]
        print(f"{name:>10} {k:>2} | {mi:>12.4f} {ma:>9.4f} {aics[name]-base:>9.1f} | "
              f"{'  '.join(f'{v:.4f}' for v in loo):>34} {ml:>8.4f}")
