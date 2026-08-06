#!/usr/bin/env python3
"""idx4~8 optimized wLRE/wFFI vs 경험 FC — 모형 적합 비교 figure."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from fit_compare import ROWS, hinge_fit, hinge3_fit, r2, WMAX   # noqa: E402

OUT = "output_ppmi_pd/_figs/param_fc_fits.png"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#ebe9e3"
C_LIN, C_CUB, C_HIN, C_H3 = "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
DENS = LinearSegmentedColormap.from_list("blues", [
    "#e8eef5", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])

X = np.concatenate([d["x"] for d in ROWS.values()])
P = {p: np.concatenate([d[p] for d in ROWS.values()]) for p in ("wLRE", "wFFI")}

FIT = {}
for p in ("wLRE", "wFFI"):
    y = P[p]
    FIT[p] = dict(lin=np.polyfit(X, y, 1), cub=np.polyfit(X, y, 3),
                  h1=hinge_fit(X, y), h3=hinge3_fit(X, y))

MODELS = [("선형", "lin", C_LIN, lambda t, g: np.polyval(t, g)),
          ("3차", "cub", C_CUB, lambda t, g: np.polyval(t, g)),
          ("hinge", "h1", C_HIN, lambda t, g: np.clip(t[0] + t[1] * g, 0, WMAX)),
          ("hinge3", "h3", C_H3, lambda t, g: np.clip(np.polyval(t, g), 0, WMAX))]

# LOO 외삽 R² (subject 단위)
LOO = {p: {} for p in P}
for p in P:
    for nm, key, _, pred in MODELS:
        v = []
        for idx, d in ROWS.items():
            ox = np.concatenate([ROWS[j]["x"] for j in ROWS if j != idx])
            oy = np.concatenate([ROWS[j][p] for j in ROWS if j != idx])
            th = (np.polyfit(ox, oy, 1) if key == "lin" else
                  np.polyfit(ox, oy, 3) if key == "cub" else
                  hinge_fit(ox, oy) if key == "h1" else hinge3_fit(ox, oy))
            v.append(r2(d[p], pred(th, d["x"])))
        LOO[p][nm] = (np.mean(v), v)

fig = plt.figure(figsize=(16.2, 9.4), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.24, left=0.052, right=0.985,
                      top=0.875, bottom=0.075)
G = np.linspace(X.min(), X.max(), 500)


def frame(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color("#d8d6d0"); s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


def dens(ax, xx, yy, ylim):
    ax.hexbin(xx, yy, gridsize=68, cmap=DENS, norm=LogNorm(1, 900),
              linewidths=0, zorder=1)
    ax.set_xlim(X.min() - 0.02, X.max() + 0.02); ax.set_ylim(*ylim)


# ── A  wLRE ~ FC ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0]); frame(ax)
dens(ax, X, P["wLRE"], (-0.3, 5.2))
for nm, key, col, pred in MODELS:
    if nm in ("hinge",):
        continue
    ax.plot(G, pred(FIT["wLRE"][key], G), color=col, lw=2.0, zorder=3,
            ls="-" if nm != "3차" else (0, (5, 2)))
ax.text(0.60, 0.30, "선형 / 3차 / hinge3\n세 곡선이 겹친다", transform=ax.transAxes,
        fontsize=9.5, color=INK, ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec="#d8d6d0", lw=0.8))
ax.set_title("A  wLRE vs 경험 FC — 모형 선택 무의미", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("경험 FC", color=INK2, fontsize=9.5)
ax.set_ylabel("wLRE (optimized)", color=INK2, fontsize=9.5)

# ── B  wFFI ~ FC ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1]); frame(ax)
dens(ax, X, P["wFFI"], (-0.9, 3.4))
ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=2)
LBL_B = {"선형": 0.68, "3차": 0.90, "hinge": 0.30, "hinge3": 0.52}   # 축 안에 보이는 x
for nm, key, col, pred in MODELS:
    ax.plot(G, pred(FIT["wFFI"][key], G), color=col, lw=2.1, zorder=3)
    xl = LBL_B[nm]
    yl = pred(FIT["wFFI"][key], np.array([xl]))[0]
    ax.annotate(nm, (xl, yl), xytext=(0, {"선형": -13, "3차": 11, "hinge": 12,
                                          "hinge3": -13}[nm]),
                textcoords="offset points", color=col, fontsize=10,
                fontweight="bold", ha="center", va="center", zorder=4)
ax.set_title("B  wFFI vs 경험 FC — 여기서 갈린다", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("경험 FC", color=INK2, fontsize=9.5)
ax.set_ylabel("wFFI (optimized)", color=INK2, fontsize=9.5)

# ── C  고 FC 확대 ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2]); frame(ax)
s = X >= 0.40
ax.hexbin(X[s], P["wFFI"][s], gridsize=42, cmap=DENS, norm=LogNorm(1, 200),
          linewidths=0, zorder=1)
g2 = np.linspace(0.40, X.max(), 300)
for nm, key, col, pred in MODELS:
    ax.plot(g2, pred(FIT["wFFI"][key], g2), color=col, lw=2.2, zorder=3)
    ax.annotate(nm, (X.max(), pred(FIT["wFFI"][key], np.array([X.max()]))[0]),
                xytext=(-4, {"선형": 0, "3차": 9, "hinge": -11, "hinge3": 3}[nm]),
                textcoords="offset points", color=col, fontsize=10,
                fontweight="bold", va="center", ha="right")
ax.axhline(0, color=MUTED, lw=1.0, ls=":", zorder=2)
ax.set_xlim(0.40, X.max() + 0.01); ax.set_ylim(-0.85, 0.62)
ax.set_title("C  FC ≥ 0.40 확대 — 3차만 되돌아온다", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("경험 FC", color=INK2, fontsize=9.5)
ax.set_ylabel("wFFI", color=INK2, fontsize=9.5)
ax.text(0.03, 0.06, "실제 wFFI = 0 (박스제약 하한)", transform=ax.transAxes,
        fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec="#d8d6d0", lw=0.8))

# ── D  LOO 외삽 R² ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0]); frame(ax)
names = [m[0] for m in MODELS]
xs = np.arange(len(names)); w = 0.36
for si, (p, col) in enumerate([("wLRE", "#2a78d6"), ("wFFI", "#eb6834")]):
    v = [LOO[p][n][0] for n in names]
    ax.bar(xs + (si - 0.5) * w, v, w * 0.9, color=col, label=p, zorder=2)
    for gi, n in enumerate(names):     # 점은 오른쪽, 라벨은 왼쪽 → 충돌 없음
        ax.scatter(np.full(5, xs[gi] + (si - 0.5) * w + 0.085), LOO[p][n][1], s=13,
                   facecolor=SURFACE, edgecolor=INK2, linewidths=0.8, zorder=3)
    for gi, vv in enumerate(v):
        ax.text(xs[gi] + (si - 0.5) * w - 0.085, max(vv, max(LOO[p][names[gi]][1])) + 0.004,
                f"{vv:.4f}", ha="center", fontsize=8.8, color=INK, zorder=4)
ax.set_xticks(xs); ax.set_xticklabels(names, color=INK2, fontsize=9.5)
ax.set_ylim(0.83, 0.965)
ax.set_ylabel("LOO 외삽 R²", color=INK2, fontsize=9.5)
ax.set_title("D  subject-LOO 외삽 R² (점 = 개별 subject)", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
lg = ax.legend(frameon=False, fontsize=9.5, loc="lower right", ncol=2)
for t in lg.get_texts():
    t.set_color(INK2)

# ── E  wFFI 구간별 RMSE ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1]); frame(ax)
bands = [(-1, 0.45, "FC < 0.45\n(엣지 92%)"), (0.45, 2, "FC ≥ 0.45\n(엣지 8%)")]
xs = np.arange(len(bands)); w = 0.2
for mi, (nm, key, col, pred) in enumerate(MODELS):
    v = []
    for lo, hi, _ in bands:
        s = (X >= lo) & (X < hi)
        v.append(np.sqrt(np.mean((pred(FIT["wFFI"][key], X[s]) - P["wFFI"][s]) ** 2)))
    ax.bar(xs + (mi - 1.5) * w, v, w * 0.88, color=col, label=nm, zorder=2)
    for gi, vv in enumerate(v):
        ax.text(xs[gi] + (mi - 1.5) * w, vv + 0.008, f"{vv:.3f}", ha="center",
                fontsize=8.8, color=INK, va="bottom", zorder=4)
ax.set_xticks(xs); ax.set_xticklabels([b[2] for b in bands], color=INK2, fontsize=9.5)
ax.set_ylim(0, 0.40)                       # 선형의 0.340 까지 담아야 라벨이 안 샌다
ax.set_ylabel("wFFI RMSE", color=INK2, fontsize=9.5)
ax.set_title("E  wFFI 구간별 RMSE — 고 FC 에서 3차가 진다", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
lg = ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper left")
for t in lg.get_texts():
    t.set_color(INK2)

# ── F  wFFI ~ wLRE ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2]); frame(ax)
ax.hexbin(P["wLRE"], P["wFFI"], gridsize=62, cmap=DENS, norm=LogNorm(1, 900),
          linewidths=0, zorder=1)
gl = np.linspace(0, 5.0, 300)
ax.plot(gl, np.clip(2.0 - gl, 0, WMAX), color=C_H3, lw=2.3, zorder=3)
ax.annotate("wFFI = max(0, 2 − wLRE)", (2.9, 0.0), xytext=(0, 16),
            textcoords="offset points", color=C_H3, fontsize=10.5, fontweight="bold")
rr = r2(P["wFFI"], np.clip(2.0 - P["wLRE"], 0, WMAX))
ax.text(0.97, 0.94, f"R² = {rr:.4f}", transform=ax.transAxes, ha="right", va="top",
        fontsize=12, color=INK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec="#d8d6d0", lw=0.8))
ax.set_xlim(-0.15, 5.0); ax.set_ylim(-0.15, 3.4)
ax.set_title("F  두 파라미터끼리 — FC 없이도 설명된다", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("wLRE (optimized)", color=INK2, fontsize=9.5)
ax.set_ylabel("wFFI (optimized)", color=INK2, fontsize=9.5)

fig.suptitle("optimized wLRE / wFFI 와 경험 FC 의 관계 — idx4~8 통합 (SC>0 엣지 33,190개)",
             color=INK, fontsize=14, fontweight="bold", x=0.052, ha="left", y=0.958)
fig.text(0.052, 0.918,
         "wLRE 는 선형으로 끝난다. wFFI 만 비선형이 필요하고, 그 비선형의 정체는 곡률이 아니라 0 에서의 박스제약(clip)이다.",
         color=INK2, fontsize=10.5, ha="left")
fig.savefig(OUT, dpi=132, facecolor=SURFACE)
print("saved", OUT)
