#!/usr/bin/env python3
"""make_fig_corr_compare.py — 회색 마스킹 뷰(SC>0 엣지)에서 쌍별 상관 비교.

패널
  A 쌍별 상관 × 3 스케일(엣지/8x8 블록/노드) + subject 개별 점
  B 마스킹 전/후 대비 — 공유 0-마스크가 만들던 허위 유사성의 크기
  C 산점도 corrected SC vs wLRE   (약한 관계)
  D 산점도 empirical FC vs wLRE   (사실상 동치)

색 배정
  3 스케일 = 순서형 범주 -> 단일 색상 sequential 3단계 (fine->coarse)
  전/후    = 범주 2개    -> #DD8452 / #4C72B0 (validate_palette.js 전 항목 PASS,
             contrast WARN 은 막대 직접 라벨로 해소)

실행: python3 make_fig_corr_compare.py --out figures/corr_compare_masked.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

IDX = (4, 5, 6, 7, 8)
IU = np.triu_indices(163, 1)
PAIRS = [("corrSC–wLRE", "norm", "wlre"), ("corrSC–wFFI", "norm", "wffi"),
         ("corrSC–FC", "norm", "fc"), ("wLRE–FC", "wlre", "fc"),
         ("wFFI–FC", "wffi", "fc"), ("wLRE–wFFI", "wlre", "wffi")]
SCALE_C = ["#9ecae1", "#4292c6", "#08519c"]          # fine -> coarse
BEFORE, AFTER = "#DD8452", "#4C72B0"


def blk_nz(M, mask, k=8):
    n = (163 // k) * k
    A = M[:n, :n].reshape(n // k, k, n // k, k)
    Mk = mask[:n, :n].reshape(n // k, k, n // k, k)
    return A.sum((1, 3)) / np.maximum(Mk.sum((1, 3)), 1)


def metrics(d, i, ka, kb):
    A, B = d[f"{ka}_{i}"].copy(), d[f"{kb}_{i}"].copy()
    np.fill_diagonal(A, 0); np.fill_diagonal(B, 0)
    M = d[f"norm_{i}"] > 0; np.fill_diagonal(M, False)
    m = M[IU]
    x, y = A[IU][m], B[IU][m]
    Ab, Bb = blk_nz(A * M, M.astype(float)), blk_nz(B * M, M.astype(float))
    ib = np.triu_indices(Ab.shape[0], 1)
    cnt = np.maximum(M.sum(1), 1)
    an, bn = (A * M).sum(1) / cnt, (B * M).sum(1) / cnt
    return (np.corrcoef(x, y)[0, 1], np.corrcoef(Ab[ib], Bb[ib])[0, 1],
            np.corrcoef(an, bn)[0, 1])


def unmasked(d, i, ka, kb):
    """0-마스크를 안 걷어낸 값 — 전체 셀 / 블록평균(0 포함)."""
    A, B = d[f"{ka}_{i}"].copy(), d[f"{kb}_{i}"].copy()
    np.fill_diagonal(A, 0); np.fill_diagonal(B, 0)
    k = 8; n = (163 // k) * k
    bm = lambda M: M[:n, :n].reshape(n // k, k, n // k, k).mean((1, 3))
    Ab, Bb = bm(A), bm(B)
    ib = np.triu_indices(Ab.shape[0], 1)
    return np.corrcoef(A[IU], B[IU])[0, 1], np.corrcoef(Ab[ib], Bb[ib])[0, 1]


def binned(x, y, nb=18):
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    q[-1] += 1e-9
    c, m = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() > 5:
            c.append(x[s].mean()); m.append(y[s].mean())
    return np.array(c), np.array(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="output_ppmi_pd/_rel_data.npz")
    ap.add_argument("--out", default="figures/corr_compare_masked.png")
    a = ap.parse_args()
    d = np.load(a.npz)

    per = {lab: np.array([metrics(d, i, ka, kb) for i in IDX])
           for lab, ka, kb in PAIRS}
    mask_eff = {lab: (np.array([unmasked(d, i, ka, kb) for i in IDX]),
                      np.array([metrics(d, i, ka, kb)[:2] for i in IDX]))
                for lab, ka, kb in PAIRS[:2]}

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.4))

    # ── A: 쌍별 × 스케일 ─────────────────────────────────────────
    ax = axes[0, 0]
    labs = [p[0] for p in PAIRS]
    xb = np.arange(len(labs)); w = 0.26
    for s, (nm, col) in enumerate(zip(["edge", "8×8 block", "node"], SCALE_C)):
        vals = np.array([per[l][:, s].mean() for l in labs])
        pos = xb + (s - 1) * w
        ax.bar(pos, vals, w * 0.88, color=col, label=nm, zorder=2)
        for k, l in enumerate(labs):                       # subject 개별 점
            v = per[l][:, s]
            ax.scatter(np.full(5, pos[k]) + np.linspace(-.05, .05, 5), v,
                       s=7, color="0.15", alpha=.75, zorder=4, linewidths=0)
        for p, v in zip(pos, vals):
            ax.text(p, v + (0.05 if v >= 0 else -0.09), f"{v:+.2f}",
                    ha="center", fontsize=7.3, color="0.2", zorder=5)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xb); ax.set_xticklabels(labs, fontsize=8.8)
    ax.set_ylim(-1.18, 1.18); ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_title("A  Pairwise correlation on SC>0 edges, three scales\n"
                 "(dots = individual subjects, n=5)", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower left")
    ax.grid(axis="y", alpha=.25, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ── B: 마스킹 전/후 ──────────────────────────────────────────
    ax = axes[0, 1]
    groups, before, after = [], [], []
    for lab in [p[0] for p in PAIRS[:2]]:
        un, ma = mask_eff[lab]
        for s, nm in enumerate(["edge", "8×8 block"]):
            groups.append(f"{lab}\n{nm}")
            before.append(un[:, s].mean()); after.append(ma[:, s].mean())
    xb = np.arange(len(groups)); w = 0.36
    ax.bar(xb - w / 2, before, w, color=BEFORE, label="incl. SC==0 cells (unmasked)", zorder=2)
    ax.bar(xb + w / 2, after, w, color=AFTER, label="SC==0 greyed out (SC>0 only)", zorder=2)
    for p, v in zip(xb - w / 2, before):
        ax.text(p, v + (.03 if v >= 0 else -.07), f"{v:+.2f}", ha="center", fontsize=8.2, color="0.2")
    for p, v in zip(xb + w / 2, after):
        ax.text(p, v + (.03 if v >= 0 else -.07), f"{v:+.2f}", ha="center", fontsize=8.2, color="0.2")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xb); ax.set_xticklabels(groups, fontsize=8.2)
    ax.set_ylim(-1.02, 1.10); ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_title("B  The similarity the eye sees is the shared zero-mask\n"
                 "(imposed by ParamSet.sanitize, not learned)", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=.25, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ── C, D: 산점도 (풀링, SC>0) ────────────────────────────────
    for ax, (xk, xlab, col) in zip(
            axes[1], [("norm", "corrected SC", SCALE_C[1]),
                      ("fc", "empirical FC", SCALE_C[2])]):
        X = np.concatenate([d[f"{xk}_{i}"][IU][(d[f"norm_{i}"] > 0)[IU]] for i in IDX])
        Y = np.concatenate([d[f"wlre_{i}"][IU][(d[f"norm_{i}"] > 0)[IU]] for i in IDX])
        r = np.corrcoef(X, Y)[0, 1]; rho = spearmanr(X, Y).statistic
        ax.scatter(X, Y, s=1.6, alpha=.05, color=col, linewidths=0, rasterized=True)
        bx, by = binned(X, Y)
        ax.plot(bx, by, "-o", color="#B03030", lw=1.8, ms=4.2, zorder=5,
                label="binned mean (18 quantile bins)")
        ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel("optimized wLRE", fontsize=10)
        ax.set_title(f"{'C' if xk=='norm' else 'D'}  wLRE vs {xlab} — pooled SC>0 edges "
                     f"(n={len(X):,})\nPearson r = {r:+.3f}   Spearman ρ = {rho:+.3f}",
                     fontsize=10.5, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        ax.grid(alpha=.22)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Optimized coupling tracks empirical FC, not corrected SC — "
                 "PPMI PD idx 4–8, AAL163", fontsize=13, y=0.985)
    fig.text(0.5, 0.005,
             "corrSC = corrected SC (cfg.sc_norm='log1pm'), the same array the simulation uses "
             "and the 'corrected SC' column of sc_fc_matrices_masked.png — raw streamline counts "
             "are never used here.   A/B report the mean of per-subject r;  C/D pool all subjects' "
             "edges, which runs lower because per-subject SC scale offsets are mixed in.",
             ha="center", fontsize=8.2, color="0.35")
    fig.tight_layout(rect=(0, 0.028, 1, 0.965))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"[fig] {a.out}")
    for lab in labs:
        v = per[lab]
        print(f"  {lab:<11} edge {v[:,0].mean():+.3f}  block {v[:,1].mean():+.3f}  "
              f"node {v[:,2].mean():+.3f}")


if __name__ == "__main__":
    main()
