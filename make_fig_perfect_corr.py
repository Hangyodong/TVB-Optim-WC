#!/usr/bin/env python3
"""make_fig_perfect_corr.py — idx4 wLRE/wFFI 를 FC 와 corr ±1 로 변형한 결과 figure.

행1 행렬(SC==0 회색): emp FC | wLRE 원본 | wLRE 변형 | wFFI 원본 | wFFI 변형
행2 산점도          : 위 네 가중치의 vs FC 산점도 + 합(wLRE+wFFI) 분포 전/후

색: 가중치 4개 = 공통 sequential(Purples, 공통 축) / FC = 발산형(RdBu_r, ±대칭)

실행: python3 make_fig_perfect_corr.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDX = 4
NPZ = "output_ppmi_pd/_rel_data_idx4_perfect.npz"
OUT = "figures/idx4_perfect_corr.png"
IU = np.triu_indices(163, 1)
ORIG, PERF = "#DD8452", "#4C72B0"


def main():
    d = np.load(NPZ)
    FC = d[f"fc_{IDX}"].astype(np.float64)
    M = d[f"norm_{IDX}"] > 0
    np.fill_diagonal(M, False)
    W = {"wLRE orig": d[f"wlre_{IDX}"].astype(np.float64),
         "wLRE r=+1": d[f"wlre_{IDX}_perf"].astype(np.float64),
         "wFFI orig": d[f"wffi_{IDX}"].astype(np.float64),
         "wFFI r=−1": d[f"wffi_{IDX}_perf"].astype(np.float64)}
    m = M[IU]
    f = FC[IU][m]
    w_hi = max(np.percentile(A[M], 99.5) for A in W.values())
    fc_lim = float(np.percentile(np.abs(FC[IU]), 99.5))

    fig, axes = plt.subplots(2, 5, figsize=(23, 9.4))

    # ── 행 1: 행렬 ───────────────────────────────────────────────
    def show(ax, A, ttl, cmap, lim, mask_sc0=True):
        cm = plt.get_cmap(cmap).copy()
        Ashow = np.ma.masked_where(~M, A) if mask_sc0 else A
        if mask_sc0:
            cm.set_bad("0.80")
        im = ax.imshow(Ashow, cmap=cm, vmin=lim[0], vmax=lim[1],
                       aspect="equal", interpolation="nearest")
        ax.axhline(111.5, color="0.35", lw=.6, alpha=.55)
        ax.axvline(111.5, color="0.35", lw=.6, alpha=.55)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    show(axes[0, 0], FC, "empirical FC", "RdBu_r", (-fc_lim, fc_lim), mask_sc0=False)
    for ax, (nm, A) in zip(axes[0, 1:], W.items()):
        r = np.corrcoef(f, A[IU][m])[0, 1]
        show(ax, A, f"{nm}   (r vs FC = {r:+.3f})", "Purples", (0.0, w_hi))

    # ── 행 2: 산점도 4개 + 합 분포 ───────────────────────────────
    for ax, (nm, A) in zip(axes[1, :4], W.items()):
        v = A[IU][m]
        r = np.corrcoef(f, v)[0, 1]
        col = PERF if "r=" in nm else ORIG
        ax.scatter(f, v, s=1.6, alpha=.07, color=col, linewidths=0, rasterized=True)
        ax.set_xlabel("empirical FC", fontsize=9.5)
        ax.set_ylabel(nm, fontsize=9.5)
        ax.set_title(f"{nm} vs FC   r = {r:+.6f}", fontsize=10)
        ax.set_ylim(-0.15, w_hi * 1.05)
        ax.grid(alpha=.22)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    ax = axes[1, 4]
    s0 = (W["wLRE orig"] + W["wFFI orig"])[IU][m]
    s1 = (W["wLRE r=+1"] + W["wFFI r=−1"])[IU][m]
    bins = np.linspace(1.0, 4.8, 80)
    ax.hist(s0, bins=bins, color=ORIG, alpha=.72,
            label=f"original  ({(np.abs(s0-2)<0.05).mean()*100:.0f}% within 2±0.05)")
    ax.hist(s1, bins=bins, color=PERF, alpha=.72,
            label=f"transformed  ({(np.abs(s1-2)<0.05).mean()*100:.0f}% within 2±0.05)")
    ax.axvline(2.0, color="0.25", lw=1.2, ls="--")
    ax.set_xlabel("wLRE + wFFI", fontsize=9.5)
    ax.set_ylabel("edge count", fontsize=9.5)
    ax.set_title("Side effect: the emergent sum≈2 structure is lost", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.22)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"idx {IDX} — weights forced to perfect correlation with empirical FC "
        "(wLRE r=+1, wFFI r=−1);  grey = SC == 0", fontsize=13.5, y=0.985)
    fig.text(0.5, 0.005,
             "Transform = affine map of FC that pins each parameter's original range [min=0, max], "
             "applied on SC>0 edges only, then clip[0,10] · SC-mask · symmetrise (same projection as "
             "ParamSet.sanitize). An affine map has only 2 degrees of freedom, so fixing the range "
             "means the mean is not preserved; moment matching was unusable because it yields "
             "negative weights whose clipping would break r = ±1.",
             ha="center", fontsize=8.4, color="0.35")
    fig.tight_layout(rect=(0, 0.028, 1, 0.965))
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=165, bbox_inches="tight", facecolor="white")
    print(f"[fig] {OUT}")
    for nm, A in W.items():
        v = A[IU][m]
        print(f"  {nm:<11} r={np.corrcoef(f,v)[0,1]:+.6f}  mean {v.mean():.4f}  "
              f"sd {v.std():.4f}  max {v.max():.4f}  =0 {(v<=1e-12).sum()}")


if __name__ == "__main__":
    main()
