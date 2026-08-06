#!/usr/bin/env python3
"""idx4~8 optimized wLRE/wFFI vs 경험 FC — 선형/2차/3차/hinge 적합과 R².

SC>0 엣지(upper-tri)만. hinge 는 오늘 확인된 박스제약 [0, w_max] 을 모형에 넣은 것:
    w = clip(a + b*FC, 0, w_max)
자유 엣지(clip 안 걸린 것)만으로 적합했을 때의 R² 도 같이 낸다 — clip 이 선형 R² 를
기계적으로 깎는 부분을 분리하기 위함.
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


def fit_poly(x, y, deg):
    c = np.polyfit(x, y, deg)
    return c, r2(y, np.polyval(c, x))


def fit_hinge(x, y):
    """y = clip(a + b*x, 0, WMAX). 계수 2개, 비선형(꺾임)."""
    def res(p):
        return np.clip(p[0] + p[1] * x, 0.0, WMAX) - y
    p0 = np.polyfit(x, y, 1)[::-1]
    s = least_squares(res, p0, method="lm", max_nfev=20000)
    return s.x, r2(y, np.clip(s.x[0] + s.x[1] * x, 0.0, WMAX))


def fit_exp(x, y):
    """y = a + b*exp(c*x). 계수 3개."""
    def res(p):
        return p[0] + p[1] * np.exp(np.clip(p[2] * x, -50, 50)) - y
    best, br = None, -9
    for c0 in (0.5, 1.0, 2.0, -1.0):
        try:
            s = least_squares(res, [y.mean(), 0.5, c0], method="lm", max_nfev=20000)
            rr = r2(y, s.x[0] + s.x[1] * np.exp(np.clip(s.x[2] * x, -50, 50)))
            if rr > br:
                best, br = s.x, rr
        except Exception:
            pass
    return best, br


ROWS = {}
for idx, sub in SUBDIR.items():
    ref, corr = load_ref(sub)
    if ref is None:
        continue
    W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
    FC = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
    np.fill_diagonal(FC, 0.0)
    m = (W > 0)[IU]
    x = FC[IU][m]
    ROWS[idx] = dict(x=x, wLRE=ref["wLRE"][IU][m], wFFI=ref["wFFI"][IU][m], corr=corr)

for pname in ("wLRE", "wFFI"):
    print(f"\n{'='*104}\n### {pname}  vs  경험 FC   (SC>0 엣지)\n{'='*104}")
    print(f"{'idx':>3} {'n':>6} {'clip%':>6} | {'선형 R²':>8} {'2차 R²':>8} {'3차 R²':>8} "
          f"{'지수 R²':>8} {'hinge R²':>9} | {'자유엣지만 선형 R²':>17} | 선형식")
    acc = {k: [] for k in ("lin", "q", "c", "e", "h", "free")}
    for idx, d in ROWS.items():
        x, y = d["x"], d[pname]
        free = y > 1e-6
        (b1, a1), rl = fit_poly(x, y, 1)
        _, rq = fit_poly(x, y, 2)
        _, rc = fit_poly(x, y, 3)
        _, re_ = fit_exp(x, y)
        ph, rh = fit_hinge(x, y)
        (bf, af), rf = fit_poly(x[free], y[free], 1)
        for k, v in zip(("lin", "q", "c", "e", "h", "free"), (rl, rq, rc, re_, rh, rf)):
            acc[k].append(v)
        print(f"{idx:>3} {x.size:>6} {(~free).mean()*100:>5.1f}% | {rl:>8.4f} {rq:>8.4f} "
              f"{rc:>8.4f} {re_:>8.4f} {rh:>9.4f} | {rf:>17.4f} | "
              f"{a1:+.4f} {b1:+.4f}*FC")
    print("-" * 104)
    print(f"{'평균':>3} {'':>6} {'':>6} | {np.mean(acc['lin']):>8.4f} {np.mean(acc['q']):>8.4f} "
          f"{np.mean(acc['c']):>8.4f} {np.mean(acc['e']):>8.4f} {np.mean(acc['h']):>9.4f} | "
          f"{np.mean(acc['free']):>17.4f}")

# ── 두 파라미터 사이 관계 ────────────────────────────────────────────
print(f"\n{'='*104}\n### wFFI vs wLRE  (파라미터끼리)\n{'='*104}")
print(f"{'idx':>3} | {'wLRE+wFFI 평균':>13} {'sd':>7} | {'선형 R²':>8} | "
      f"{'hinge max(0,S-wLRE) 최적 S':>26} {'R²':>8}")
for idx, d in ROWS.items():
    wl, wf = d["wLRE"], d["wFFI"]
    s = wl + wf
    _, rl = fit_poly(wl, wf, 1)
    Ss = np.arange(1.6, 3.01, 0.005)
    rr = [r2(wf, np.clip(S - wl, 0.0, WMAX)) for S in Ss]
    k = int(np.argmax(rr))
    print(f"{idx:>3} | {s.mean():>13.4f} {s.std():>7.4f} | {rl:>8.4f} | "
          f"{Ss[k]:>26.3f} {rr[k]:>8.4f}")

# ── 5명 통합(pooled) ────────────────────────────────────────────────
print(f"\n{'='*104}\n### 5명 통합 (pooled)\n{'='*104}")
X = np.concatenate([d["x"] for d in ROWS.values()])
for pname in ("wLRE", "wFFI"):
    Y = np.concatenate([d[pname] for d in ROWS.values()])
    (b1, a1), rl = fit_poly(X, Y, 1)
    q, rq = fit_poly(X, Y, 2)
    ph, rh = fit_hinge(X, Y)
    print(f"{pname}: 선형  {a1:+.4f} {b1:+.4f}*FC                       R²={rl:.4f}")
    print(f"      2차   {q[2]:+.4f} {q[1]:+.4f}*FC {q[0]:+.4f}*FC²        R²={rq:.4f}")
    print(f"      hinge clip({ph[0]:+.4f} {ph[1]:+.4f}*FC, 0, {WMAX:g})   R²={rh:.4f}"
          f"   (영점 FC={-ph[0]/ph[1]:+.4f})")
