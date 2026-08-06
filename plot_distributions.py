#!/usr/bin/env python3
"""plot_distributions.py — wLRE/wFFI 와 FC_emp/FC_sim 의 분포 비교.

행1: 가중치 (SC>0 상삼각 엣지만 — 그 밖에는 파라미터가 존재하지 않는다)
행2: FC (전체 상삼각)
열: subject. density 정규화라 엣지 수가 달라도 모양이 비교된다.

출력: output_ppmi_pd/_figs/distributions.png
실행: python3 plot_distributions.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wlre_fc import (GRID, INK, INK2, IU, MUTED, OUT, SERIES, SUBDIR,
                          SURFACE, load_ref, _style)

FC_COL = {"FC_emp": "#2a78d6", "FC_sim": "#eb6834"}   # 카테고리 slot 1/2


def _hist(ax, series, lo, hi, bins=70):
    """series = {label: (values, color)}. 계단 히스토그램 + 중앙값 눈금."""
    for lab, (v, c) in series.items():
        ax.hist(v, bins=bins, range=(lo, hi), density=True, histtype="step",
                lw=1.8, color=c, label=f"{lab}  (median {np.median(v):.3f})")
        ax.axvline(np.median(v), color=c, lw=1, ls=":", alpha=0.7)
    ax.set_xlim(lo, hi)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _style(ax)
    leg = ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor(SURFACE)
    for t in leg.get_texts():
        t.set_color(INK)


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = list(SUBDIR.items())
    fig, axes = plt.subplots(2, len(subs), figsize=(3.6 * len(subs), 7.2),
                             facecolor=SURFACE)
    for k, (idx, sub) in enumerate(subs):
        ref, corr, _ = load_ref(sub)
        W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
        np.fill_diagonal(FCe, 0.0)
        m = (W > 0)[IU]
        wl = np.asarray(ref["wLRE"], float)[IU][m]
        wf = np.asarray(ref["wFFI"], float)[IU][m]
        e = FCe[IU]
        s = np.asarray(ref["FC_sim"], float)
        np.fill_diagonal(s, 0.0)
        s = s[IU]

        _hist(axes[0, k], {"wLRE": (wl, SERIES["wLRE"]), "wFFI": (wf, SERIES["wFFI"])},
              0.0, 5.0)
        axes[0, k].axvline(1.0, color=MUTED, lw=1, ls="--")
        axes[0, k].set_title(f"idx {idx} (sub {sub})   n={m.sum():,} 엣지",
                             color=INK, fontsize=11, pad=6)
        z0 = (wl <= 1e-6).sum(), (wf <= 1e-6).sum()
        axes[0, k].text(0.97, 0.55, f"0 인 엣지\nwLRE {z0[0]}  wFFI {z0[1]}",
                        transform=axes[0, k].transAxes, ha="right", va="top",
                        color=INK2, fontsize=7.5)

        _hist(axes[1, k], {"FC_emp": (e, FC_COL["FC_emp"]), "FC_sim": (s, FC_COL["FC_sim"])},
              -1.0, 1.0)
        axes[1, k].axvline(0.0, color=MUTED, lw=1, ls="--")
        axes[1, k].set_title(f"corr {corr:.4f}   n={len(e):,} 엣지",
                             color=INK, fontsize=11, pad=6)
        axes[1, k].text(0.03, 0.97,
                        f"mean {e.mean():+.3f} → {s.mean():+.3f}\n"
                        f"sd   {e.std():.3f} → {s.std():.3f}",
                        transform=axes[1, k].transAxes, ha="left", va="top",
                        color=INK2, fontsize=7.5)
        axes[1, k].set_xlabel("FC (상관계수)", color=INK2, fontsize=9)
        axes[0, k].set_xlabel("가중치", color=INK2, fontsize=9)
    axes[0, 0].set_ylabel("밀도", color=INK2, fontsize=10)
    axes[1, 0].set_ylabel("밀도", color=INK2, fontsize=10)

    fig.suptitle("가중치(wLRE·wFFI)와 FC(경험·시뮬) 분포 — subject 5명", color=INK,
                 fontsize=14, y=0.985)
    fig.text(0.5, 0.012,
             "행1: SC>0 상삼각 엣지의 가중치. 점선=중앙값, 파선=초기값 1.0. "
             "0 에 몰린 질량은 clip 하한에 닿은 엣지(주로 wFFI).   "
             "행2: 전체 상삼각 FC. 파선=0.  density 정규화 — 엣지 수가 달라도 모양 비교 가능.",
             ha="center", color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    p = f"{OUT}/distributions.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
