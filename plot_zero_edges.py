#!/usr/bin/env python3
"""SC>0 인데 wLRE 또는 wFFI 가 0 인 엣지: 시뮬 FC 행렬 + 평균."""
import glob
import hashlib
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 한글 글리프: DejaVu Sans 에 없다. Noto Sans CJK(.ttc)는 한글 포함.
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

IU = np.triu_indices(163, 1)
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025"}
OUT = "output_ppmi_pd/_figs/zero_edges_fc.png"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # 검증된 categorical 1,2
DIV = LinearSegmentedColormap.from_list("bgr", [
    "#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb",
    "#f0efec",                                                   # 중성 회색 중점
    "#fbd7d7", "#f4a3a3", "#e97070", "#e34948", "#c22f2f", "#8f1f1f"])


def load_ref(sub):
    hs = {hashlib.sha1(np.asarray(np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv",
                                             delimiter=",")[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc, bfc = None, -9.0, None
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
            bc, bfc = c, np.asarray(md.get("post_grad_fc_matrix"), np.float64)
    return best, bc, bfc


D = {}
for idx, sub in SUBDIR.items():
    ref, corr, S = load_ref(sub)
    if ref is None or S is None:
        continue
    W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
    E = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
    np.fill_diagonal(E, 0.0)
    np.fill_diagonal(S, 0.0)
    m = (W > 0)[IU]
    wl, wf = ref["wLRE"][IU][m], ref["wFFI"][IU][m]
    D[idx] = dict(E=E, S=S, m=m, e=E[IU][m], s=S[IU][m],
                  zl=wl <= 1e-6, zf=wf <= 1e-6, corr=corr,
                  ii=IU[0][m], jj=IU[1][m])

K = 4                                                    # 행렬 패널은 idx4
d = D[K]
NORM = Normalize(-0.9, 0.9)

fig = plt.figure(figsize=(15.5, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.26,
                      left=0.055, right=0.965, top=0.885, bottom=0.085)


def frame(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color("#d8d6d0")
        s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


# ── (A) 전체 시뮬 FC ────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0]); frame(ax)
ax.imshow(d["S"], cmap=DIV, norm=NORM, interpolation="nearest")
ax.set_title("A  시뮬 FC 전체 (idx4, optimized)", color=INK, fontsize=11,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("node", color=INK2, fontsize=9)
ax.set_ylabel("node", color=INK2, fontsize=9)

# ── (B)(C) 0-파라미터 엣지만 ────────────────────────────────────────
for col, (key, lab) in enumerate([("s", "B  그 엣지의 시뮬 FC"),
                                  ("e", "C  그 엣지의 경험 FC")], start=1):
    ax = fig.add_subplot(gs[0, col]); frame(ax)
    ax.set_xlim(-2, 165); ax.set_ylim(165, -2); ax.set_aspect("equal")
    v = d[key]
    for sel, mk in [(d["zf"], "o"), (d["zl"], "s")]:
        ii, jj = d["ii"][sel], d["jj"][sel]
        ax.scatter(np.r_[jj, ii], np.r_[ii, jj], c=np.r_[v[sel], v[sel]],
                   cmap=DIV, norm=NORM, s=14, marker=mk,
                   linewidths=0.4, edgecolors=SURFACE)
    mf, ml = v[d["zf"]].mean(), v[d["zl"]].mean()
    ax.set_title(lab, color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("node", color=INK2, fontsize=9)
    ax.text(0.03, 0.055,
            f"● wFFI=0 (n={d['zf'].sum()})  평균 {mf:+.4f}\n"
            f"■ wLRE=0 (n={d['zl'].sum()})  평균 {ml:+.4f}",
            transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom",
            bbox=dict(boxstyle="round,pad=0.42", fc=SURFACE, ec="#d8d6d0", lw=0.8))

cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=DIV),
                  ax=fig.axes[:3], fraction=0.016, pad=0.012)
cb.set_label("FC (Pearson r)", color=INK2, fontsize=9)
cb.ax.tick_params(colors=MUTED, labelsize=8)
cb.outline.set_edgecolor("#d8d6d0")

# ── (D) 5 subject 평균: 경험 vs 시뮬 ────────────────────────────────
ax = fig.add_subplot(gs[1, :2]); frame(ax)
groups = [("wLRE=0\n(SC>0)", "zl"), ("wFFI=0\n(SC>0)", "zf"), ("전체 SC>0", None)]
xs = np.arange(len(groups)); w = 0.34
for si, (nm, col) in enumerate([("경험 FC", BLUE), ("시뮬 FC", ORANGE)]):
    vals, pts = [], []
    for gi, (_, k) in enumerate(groups):
        per = [(D[i]["e"] if si == 0 else D[i]["s"])[D[i][k] if k else slice(None)].mean()
               for i in D]
        vals.append(np.mean(per)); pts.append(per)
    ax.bar(xs + (si - 0.5) * w, vals, w * 0.92, color=col, label=nm, zorder=2)
    for gi, per in enumerate(pts):
        # 점은 막대 오른쪽 절반에 몰아두고, 값 라벨은 왼쪽 절반 위에 → 충돌 없음
        ax.scatter(np.full(len(per), xs[gi] + (si - 0.5) * w + 0.075), per, s=16,
                   facecolor=SURFACE, edgecolor=INK2, linewidths=0.9, zorder=3)
    for gi, (vv, per) in enumerate(zip(vals, pts)):
        top = max(vv, max(per)) if vv >= 0 else min(vv, min(per))
        ax.text(xs[gi] + (si - 0.5) * w - 0.085,
                top + (0.035 if vv >= 0 else -0.055),
                f"{vv:+.3f}", ha="center", fontsize=9.5, color=INK, zorder=4)
ax.axhline(0, color="#c9c7c0", lw=1.0, zorder=1)
ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in groups], color=INK2, fontsize=9.5)
ax.set_ylabel("FC 평균", color=INK2, fontsize=9.5)
ax.set_title("D  0-파라미터 엣지의 경험 vs 시뮬 FC (5 subject 평균, 점=개별 subject)",
             color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
lg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
for t in lg.get_texts():
    t.set_color(INK2)
ax.grid(axis="y", color="#ebe9e3", lw=0.8, zorder=0)
ax.set_axisbelow(True)

# ── (E) 노드별 0-엣지 개수 ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2]); frame(ax)
cnt = np.bincount(np.r_[d["ii"][d["zf"]], d["jj"][d["zf"]]], minlength=163)
ax.bar(np.arange(163), cnt, width=1.0, color=BLUE, zorder=2)
ax.set_title("E  노드별 wFFI=0 엣지 수 (idx4)", color=INK, fontsize=11,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("node", color=INK2, fontsize=9)
ax.set_ylabel("엣지 수", color=INK2, fontsize=9)
ax.axhline(cnt.mean(), color=ORANGE, lw=1.6, ls="--", zorder=3)
ax.set_ylim(0, cnt.max() * 1.22)
ax.text(0.985, 0.955, f"균등 분포 시 {cnt.mean():.1f}개/노드   최대 {cnt.max()}개",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=ORANGE, zorder=4,
        bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec="#d8d6d0", lw=0.8))
ax.grid(axis="y", color="#ebe9e3", lw=0.8, zorder=0)
ax.set_axisbelow(True)

fig.suptitle("SC>0 인데 wLRE 또는 wFFI 가 0 인 엣지 — 박스제약 활성면",
             color=INK, fontsize=13.5, fontweight="bold", x=0.055, ha="left", y=0.962)
fig.text(0.055, 0.925,
         "두 집합은 서로소(둘 다 0인 엣지 5 subject 전부 0개). wFFI=0 은 FC 최상위, wLRE=0 은 FC 최하위 엣지.",
         color=INK2, fontsize=10, ha="left")
fig.savefig(OUT, dpi=135, facecolor=SURFACE)
print("saved", OUT)
for i in D:
    print(f"idx{i}  wLRE=0 n={D[i]['zl'].sum():3d} emp {D[i]['e'][D[i]['zl']].mean():+.4f} "
          f"sim {D[i]['s'][D[i]['zl']].mean():+.4f} | wFFI=0 n={D[i]['zf'].sum():3d} "
          f"emp {D[i]['e'][D[i]['zf']].mean():+.4f} sim {D[i]['s'][D[i]['zf']].mean():+.4f}")
