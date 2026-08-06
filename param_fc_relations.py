#!/usr/bin/env python3
"""optimized (wLRE, wFFI, c_ei) 와 경험 FC / 시뮬 FC 의 관계 정리 + figure.

엣지 수준: wLRE, wFFI  vs  FC_emp, FC_sim   (SC>0 upper-tri)
노드 수준: c_ei         vs  해석적 고정점해, SC 총입력, 노드평균 FC
핵심 질문: 가중치가 '자기가 만든 시뮬 FC' 보다 '경험 FC' 에 더 붙어 있나 (= 타깃 누출 징후)
"""
import glob
import hashlib
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from analytic_fic import analytic_c_ei, sc_norm_log1pm

IU = np.triu_indices(163, 1)
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025"}
OUT = "output_ppmi_pd/_figs/param_fc_relations.png"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#ebe9e3"
C_EMP, C_SIM, C_A, C_B = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
DENS = LinearSegmentedColormap.from_list("b", [
    "#e8eef5", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
DIV = LinearSegmentedColormap.from_list("bgr", [
    "#cde2fb", "#e8eef5", "#f0efec", "#fbd7d7", "#f4a3a3", "#e34948", "#8f1f1f"])


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
            best = {k: np.asarray(g(p, k), np.float64) for k in ("c_ei", "wLRE", "wFFI")}
            bc, bfc = c, np.asarray(md.get("post_grad_fc_matrix"), np.float64)
    return best, bc, bfc


D = {}
for idx, sub in SUBDIR.items():
    ref, corr, S = load_ref(sub)
    if ref is None or S is None:
        continue
    W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
    E = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
    np.fill_diagonal(E, 0.0); np.fill_diagonal(S, 0.0)
    Wn = sc_norm_log1pm(W)
    m = (W > 0)[IU]
    c_an = np.clip(analytic_c_ei(Wn, ref["wLRE"], ref["wFFI"])[0], 0, 20)
    mask = (Wn > 0)
    D[idx] = dict(e=E[IU][m], s=S[IU][m], wl=ref["wLRE"][IU][m], wf=ref["wFFI"][IU][m],
                  cei=ref["c_ei"], c_an=c_an, corr=corr,
                  s_i=Wn.sum(1), t_i=(Wn * E).sum(1),
                  Erow=(E * mask).sum(1) / np.maximum(mask.sum(1), 1),
                  Srow=(S * mask).sum(1) / np.maximum(mask.sum(1), 1),
                  L_i=(Wn * ref["wLRE"]).sum(1), F_i=(Wn * ref["wFFI"]).sum(1))

E_ = np.concatenate([d["e"] for d in D.values()])
S_ = np.concatenate([d["s"] for d in D.values()])
WL = np.concatenate([d["wl"] for d in D.values()])
WF = np.concatenate([d["wf"] for d in D.values()])
NODE = {k: np.concatenate([d[k] for d in D.values()])
        for k in ("cei", "c_an", "s_i", "t_i", "Erow", "Srow", "L_i", "F_i")}


def rr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


print(f"{'='*88}\n엣지 수준 (SC>0, n={E_.size})  — 상관계수\n{'='*88}")
print(f"{'':>10} {'FC_emp':>9} {'FC_sim':>9} | subject별 FC_emp / FC_sim")
for nm, v in (("wLRE", WL), ("wFFI", WF)):
    per = "  ".join(f"{rr(d[nm[1].lower() == 'l' and 'wl' or 'wf'], d['e']):.3f}/"
                    f"{rr(d[nm[1].lower() == 'l' and 'wl' or 'wf'], d['s']):.3f}" for d in D.values())
    print(f"{nm:>10} {rr(v, E_):>9.4f} {rr(v, S_):>9.4f} | {per}")
print(f"{'FC_sim':>10} {rr(S_, E_):>9.4f} {'—':>9}")
print(f"{'wFFI~wLRE':>10} {rr(WF, WL):>9.4f}   max(0,2−wLRE) R² "
      f"{1 - np.sum((WF - np.clip(2 - WL, 0, 10))**2)/np.sum((WF - WF.mean())**2):.4f}")

print(f"\n{'='*88}\n노드 수준 (n={NODE['cei'].size}) — c_ei 상관계수\n{'='*88}")
for nm, lab in (("c_an", "해석적 고정점해"), ("L_i", "Σ W·wLRE (총흥분입력)"),
                ("F_i", "Σ W·wFFI (총FFI입력)"), ("s_i", "SC 총입력"),
                ("t_i", "Σ W·FC_emp"), ("Erow", "노드평균 FC_emp"),
                ("Srow", "노드평균 FC_sim")):
    print(f"  c_ei ~ {lab:<24} r = {rr(NODE['cei'], NODE[nm]):+.4f}   R² = {rr(NODE['cei'], NODE[nm])**2:.4f}")

# ── figure ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16.2, 9.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.27, left=0.052, right=0.985,
                      top=0.862, bottom=0.075)


def frame(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color("#d8d6d0"); s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


def panel_edge(ax, w, wlab, title):
    for v, col, lab in ((E_, C_EMP, "경험 FC"), (S_, C_SIM, "시뮬 FC")):
        b = np.polyfit(v, w, 1)
        g = np.linspace(v.min(), v.max(), 200)
        ax.scatter(v[::7], w[::7], s=2.2, color=col, alpha=0.12, linewidths=0, zorder=1)
        ax.plot(g, np.polyval(b, g), color=col, lw=2.4, zorder=3)
        ax.annotate(f"{lab}  r={rr(w, v):.3f}", (g[-1], np.polyval(b, g[-1])),
                    xytext=(-8, 10 if lab.startswith("경험") else -16),
                    textcoords="offset points", color=col, fontsize=10,
                    fontweight="bold", ha="right")
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("FC", color=INK2, fontsize=9.5)
    ax.set_ylabel(wlab, color=INK2, fontsize=9.5)


ax = fig.add_subplot(gs[0, 0]); frame(ax)
panel_edge(ax, WL, "wLRE (optimized)", "A  wLRE — 경험 FC 가 시뮬 FC 보다 더 붙는다")
ax.set_ylim(-0.2, 5.0)

ax = fig.add_subplot(gs[0, 1]); frame(ax)
panel_edge(ax, WF, "wFFI (optimized)", "B  wFFI — 같은 방향, 부호만 반대")
ax.set_ylim(-0.2, 3.4)

ax = fig.add_subplot(gs[0, 2]); frame(ax)
ax.hexbin(WL, WF, gridsize=60, cmap=DENS, norm=LogNorm(1, 900), linewidths=0, zorder=1)
gl = np.linspace(0, 5, 300)
ax.plot(gl, np.clip(2 - gl, 0, 10), color=C_B, lw=2.4, zorder=3)
r2h = 1 - np.sum((WF - np.clip(2 - WL, 0, 10)) ** 2) / np.sum((WF - WF.mean()) ** 2)
ax.text(0.96, 0.93, f"wFFI = max(0, 2 − wLRE)\nR² = {r2h:.4f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=11, color=C_B, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec="#d8d6d0", lw=0.8))
ax.set_xlim(-0.15, 5); ax.set_ylim(-0.15, 3.4)
ax.set_title("C  두 가중치는 사실상 자유도 1개", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("wLRE (optimized)", color=INK2, fontsize=9.5)
ax.set_ylabel("wFFI (optimized)", color=INK2, fontsize=9.5)

# D  c_ei vs 해석해
ax = fig.add_subplot(gs[1, 0]); frame(ax)
for si, (idx, d) in enumerate(D.items()):
    ax.scatter(d["c_an"], d["cei"], s=13, color=C_EMP, alpha=0.55, linewidths=0, zorder=2)
lim = [1.0, 3.5]
ax.plot(lim, lim, color=INK, lw=1.3, ls="--", zorder=3)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.text(0.04, 0.93, f"r = {rr(NODE['cei'], NODE['c_an']):.4f}\n점선 = y=x",
        transform=ax.transAxes, va="top", fontsize=11, color=INK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec="#d8d6d0", lw=0.8))
ax.set_title("D  c_ei — 해석적 고정점해가 그대로 맞춘다", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
ax.set_xlabel("해석적 c_ei (시뮬 0회)", color=INK2, fontsize=9.5)
ax.set_ylabel("c_ei (optimized)", color=INK2, fontsize=9.5)

# E  c_ei vs 노드 특징
ax = fig.add_subplot(gs[1, 1]); frame(ax)
feats = [("L_i", "Σ W·wLRE\n(총흥분입력)", C_A), ("t_i", "Σ W·FC_emp", C_EMP),
         ("Erow", "노드평균\nFC_emp", C_SIM), ("s_i", "SC 총입력", C_B)]
xs = np.arange(len(feats))
v = [rr(NODE["cei"], NODE[k]) ** 2 for k, _, _ in feats]
ax.bar(xs, v, 0.62, color=[c for _, _, c in feats], zorder=2)
for i, vv in enumerate(v):
    ax.text(i, vv + 0.015, f"{vv:.3f}", ha="center", fontsize=10, color=INK, zorder=4)
ax.axhline(rr(NODE["cei"], NODE["c_an"]) ** 2, color=INK, lw=1.6, ls="--", zorder=3)
ax.text(len(feats) - 0.5, rr(NODE["cei"], NODE["c_an"]) ** 2 + 0.018,
        f"해석해 {rr(NODE['cei'], NODE['c_an'])**2:.3f}", ha="right", fontsize=10,
        color=INK, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([f[1] for f in feats], color=INK2, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("c_ei 설명력 R²", color=INK2, fontsize=9.5)
ax.set_title("E  c_ei 는 단일 노드특징으로 안 잡힌다", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)

# F  요약 R² 히트맵
ax = fig.add_subplot(gs[1, 2]); frame(ax)
rows_l = ["wLRE", "wFFI", "c_ei(노드)"]
cols_l = ["FC_emp", "FC_sim", "상대 파라미터"]
Mx = np.array([
    [rr(WL, E_) ** 2, rr(WL, S_) ** 2, rr(WL, WF) ** 2],
    [rr(WF, E_) ** 2, rr(WF, S_) ** 2, rr(WF, WL) ** 2],
    [rr(NODE["cei"], NODE["Erow"]) ** 2, rr(NODE["cei"], NODE["Srow"]) ** 2,
     rr(NODE["cei"], NODE["c_an"]) ** 2]])
im = ax.imshow(Mx, cmap=DIV, vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{Mx[i, j]:.3f}", ha="center", va="center", fontsize=13,
                color=INK if Mx[i, j] < 0.62 else SURFACE, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(cols_l, color=INK2, fontsize=10)
ax.set_yticks(range(3)); ax.set_yticklabels(rows_l, color=INK2, fontsize=10)
ax.grid(False)
ax.set_title("F  R² 요약 (c_ei 행은 노드 수준)", color=INK, fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
cb.set_label("R²", color=INK2, fontsize=9.5)
cb.ax.tick_params(colors=MUTED, labelsize=8)
cb.outline.set_edgecolor("#d8d6d0")

fig.suptitle("optimized 파라미터 ↔ 경험 FC / 시뮬 FC — idx4~8 통합",
             color=INK, fontsize=14, fontweight="bold", x=0.052, ha="left", y=0.952)
fig.text(0.052, 0.912,
         f"엣지 {E_.size:,}개(SC>0) · 노드 {NODE['cei'].size}개. "
         f"가중치는 자기가 만든 시뮬 FC 보다 경험 FC 에 더 붙어 있다 — 타깃이 커플링에 기입돼 있다는 뜻.",
         color=INK2, fontsize=10.5, ha="left")
fig.savefig(OUT, dpi=132, facecolor=SURFACE)
print(f"\nsaved {OUT}")
