#!/usr/bin/env python3
"""3차 적합의 내부: 기저 분해, 잔차 구조, subject별 계수, 그리고 hinge3 LM 반복 경로."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.optimize import least_squares

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from fit_compare import ROWS, r2, WMAX                       # noqa: E402

OUT = "output_ppmi_pd/_figs/cubic_detail.png"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#ebe9e3"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"   # 검증된 categorical 1,2,3,8
DENS = LinearSegmentedColormap.from_list("b", [
    "#e8eef5", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])

X = np.concatenate([d["x"] for d in ROWS.values()])
Y = np.concatenate([d["wFFI"] for d in ROWS.values()])
C3P = np.polyfit(X, Y, 3)                     # pooled 3차 (선형 LSQ, 닫힌형)
G = np.linspace(X.min(), X.max(), 500)

# ── hinge3 를 LM 으로 적합하며 반복 경로 기록 ──────────────────────
PATH = []


def res(p):
    r = np.clip(np.polyval(p, X), 0.0, WMAX) - Y
    PATH.append((p.copy(), float(0.5 * np.sum(r ** 2))))
    return r


sol = least_squares(res, C3P.copy(), method="lm", max_nfev=40000)
H3 = sol.x
# 반복별 최소 cost 누적(LM 은 시험 step 도 평가하므로 running-min 이 실제 경로)
costs = np.minimum.accumulate([c for _, c in PATH])
thetas = np.array([p for p, _ in PATH])

fig = plt.figure(figsize=(16.2, 9.4), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.26, left=0.055, right=0.985,
                      top=0.868, bottom=0.075)


def frame(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color("#d8d6d0"); s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


# ── A  subject별 3차 곡선 ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 0]); frame(ax)
ax.hexbin(X, Y, gridsize=66, cmap=DENS, norm=LogNorm(1, 900), linewidths=0, zorder=1)
for idx, d in ROWS.items():
    ax.plot(G, np.polyval(np.polyfit(d["x"], d["wFFI"], 3), G), color=MUTED, lw=1.0,
            zorder=3, alpha=0.9)
ax.plot(G, np.polyval(C3P, G), color=C2, lw=2.6, zorder=4)
ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=2)
ax.set_xlim(X.min() - 0.02, X.max() + 0.02); ax.set_ylim(-0.9, 3.4)
ax.text(0.04, 0.10, "가는 회색 = subject별 3차 (5개)\n굵은 주황 = pooled 3차",
        transform=ax.transAxes, fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec="#d8d6d0", lw=0.8))
ax.set_title("A  wFFI 3차 적합 — subject별 vs 통합", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("경험 FC", color=INK2, fontsize=9.5)
ax.set_ylabel("wFFI (optimized)", color=INK2, fontsize=9.5)

# ── B  기저 분해 ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1]); frame(ax)
d3, c2_, b1, a0 = C3P[0], C3P[1], C3P[2], C3P[3]
for lab, v, col, ls in [(f"상수 {a0:+.3f}", np.full_like(G, a0), C1, "-"),
                        (f"1차 {b1:+.3f}·FC", b1 * G, C2, "-"),
                        (f"2차 {c2_:+.3f}·FC²", c2_ * G ** 2, C3, "-"),
                        (f"3차 {d3:+.3f}·FC³", d3 * G ** 3, C4, "-")]:
    ax.plot(G, v, color=col, lw=2.0, ls=ls, zorder=3, label=lab)
ax.plot(G, np.polyval(C3P, G), color=INK, lw=2.4, ls=(0, (4, 2)), zorder=4, label="합")
ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=2)
ax.set_xlim(X.min() - 0.02, X.max() + 0.02)
ax.set_title("B  항별 기여 — 3차항이 고 FC 에서 되돌린다", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("경험 FC", color=INK2, fontsize=9.5)
ax.set_ylabel("기여값", color=INK2, fontsize=9.5)
lg = ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=1)
for t in lg.get_texts():
    t.set_color(INK2)

# ── C  잔차 구조 ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2]); frame(ax)
r3 = Y - np.polyval(C3P, X)
ax.hexbin(X, r3, gridsize=66, cmap=DENS, norm=LogNorm(1, 900), linewidths=0, zorder=1)
ax.axhline(0, color=INK, lw=1.2, zorder=3)
bins = np.linspace(X.min(), X.max(), 26)
bc = 0.5 * (bins[1:] + bins[:-1])
bm = [r3[(X >= lo) & (X < hi)].mean() if ((X >= lo) & (X < hi)).sum() > 20 else np.nan
      for lo, hi in zip(bins[:-1], bins[1:])]
ax.plot(bc, bm, color=C2, lw=2.2, marker="o", ms=4, zorder=4)
ax.set_xlim(X.min() - 0.02, X.max() + 0.02); ax.set_ylim(-0.75, 0.75)
ax.set_title("C  3차 잔차 — 고 FC 에서 계통 편향", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("경험 FC", color=INK2, fontsize=9.5)
ax.set_ylabel("잔차 (실제 − 3차예측)", color=INK2, fontsize=9.5)
ax.text(0.04, 0.08, "주황 = 구간 평균 잔차", transform=ax.transAxes, fontsize=9.5,
        color=C2, bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec="#d8d6d0", lw=0.8))

# ── D  LM 반복: cost ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0]); frame(ax)
ax.plot(np.arange(len(costs)), costs, color=C4, lw=2.2, zorder=3)
ax.set_yscale("log")
ax.set_title("D  hinge3 LM 반복 — 비용 수렴", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("잔차함수 평가 횟수", color=INK2, fontsize=9.5)
ax.set_ylabel("0.5·SSR (log)", color=INK2, fontsize=9.5)
ax.text(0.97, 0.93,
        f"시작 {costs[0]:.1f} → 종료 {costs[-1]:.1f}\n"
        f"평가 {len(costs)}회 · njev {sol.njev}\n{sol.message[:38]}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
        bbox=dict(boxstyle="round,pad=0.42", fc=SURFACE, ec="#d8d6d0", lw=0.8))

# ── E  LM 반복: 계수 궤적 ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 1]); frame(ax)
for k, (lab, col) in enumerate([("FC³", C4), ("FC²", C3), ("FC", C2), ("상수", C1)]):
    ax.plot(np.arange(thetas.shape[0]), thetas[:, k], color=col, lw=2.0, zorder=3)
    ax.annotate(f"{lab} {thetas[-1, k]:+.3f}", (thetas.shape[0] - 1, thetas[-1, k]),
                xytext=(-6, {0: 10, 1: -12, 2: 8, 3: 8}[k]), textcoords="offset points",
                color=col, fontsize=9.5, fontweight="bold", ha="right")
ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=2)
for k, col in enumerate([C4, C3, C2, C1]):
    ax.scatter([0], [C3P[k]], s=26, facecolor=SURFACE, edgecolor=col, lw=1.6, zorder=4)
ax.set_title("E  hinge3 LM 계수 궤적 (출발 = 3차 해)", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("잔차함수 평가 횟수", color=INK2, fontsize=9.5)
ax.set_ylabel("계수값", color=INK2, fontsize=9.5)

# ── F  subject별 계수 ──────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2]); frame(ax)
terms = ["상수", "FC", "FC²", "FC³"]
xs = np.arange(4); w = 0.15
for si, idx in enumerate(ROWS):
    c = np.polyfit(ROWS[idx]["x"], ROWS[idx]["wFFI"], 3)[::-1]
    ax.bar(xs + (si - 2) * w, c, w * 0.86, color=MUTED, alpha=0.55, zorder=2)
ax.bar(xs, C3P[::-1], w * 5.0, facecolor="none", edgecolor=C2, lw=2.2, zorder=3)
for k in range(4):
    ax.text(xs[k], C3P[::-1][k] + (0.16 if C3P[::-1][k] >= 0 else -0.34),
            f"{C3P[::-1][k]:+.3f}", ha="center", fontsize=9.5, color=C2,
            fontweight="bold", zorder=4)
ax.axhline(0, color=INK, lw=1.0, zorder=2)
ax.set_xticks(xs); ax.set_xticklabels(terms, color=INK2, fontsize=10)
ax.set_ylim(-3.6, 2.6)
ax.set_title("F  3차 계수 — 회색 = subject별, 테두리 = pooled", color=INK,
             fontsize=11.5, fontweight="bold", loc="left", pad=8)
ax.set_ylabel("계수값", color=INK2, fontsize=9.5)

fig.suptitle("wFFI 3차 적합의 내부 — 그리고 hinge3 비선형 회귀의 반복 경로",
             color=INK, fontsize=14, fontweight="bold", x=0.055, ha="left", y=0.953)
fig.text(0.055, 0.912,
         "3차(polyfit)는 기저 [1, FC, FC², FC³] 위의 선형 최소제곱 — 반복 없는 닫힌형 해. "
         "clip 이 들어간 hinge3 만 파라미터에 비선형이라 Levenberg–Marquardt 반복이 필요하다.",
         color=INK2, fontsize=10.5, ha="left")
fig.savefig(OUT, dpi=132, facecolor=SURFACE)
print("saved", OUT)
print(f"pooled 3차   = {C3P[3]:+.4f} {C3P[2]:+.4f}*FC {C3P[1]:+.4f}*FC² {C3P[0]:+.4f}*FC³"
      f"   R²={r2(Y, np.polyval(C3P, X)):.4f}")
print(f"pooled hinge3= clip({H3[3]:+.4f} {H3[2]:+.4f}*FC {H3[1]:+.4f}*FC² {H3[0]:+.4f}*FC³, 0, 10)"
      f"   R²={r2(Y, np.clip(np.polyval(H3, X), 0, WMAX)):.4f}")
print(f"LM: nfev={sol.nfev} njev={sol.njev} status={sol.status} · {sol.message}")
print(f"    cost {costs[0]:.2f} -> {costs[-1]:.2f}  ({(1-costs[-1]/costs[0])*100:.1f}% 감소)")
