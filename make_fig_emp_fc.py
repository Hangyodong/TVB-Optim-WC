#!/usr/bin/env python3
"""subject 별 경험 FC 행렬을 한 줄에 나란히 그린다.

FC 는 .mat 에서 직접 읽는다 — output_ppmi_pd/<sub>/inputs/FC.csv 는 그 subject 를 마지막으로
실행한 시점의 추출본이라 subject 마다 데이터 버전이 다를 수 있다.

사용:
    python3 make_fig_emp_fc.py                       # idx 4,5,6
    python3 make_fig_emp_fc.py --idxs 0,1,2
    python3 make_fig_emp_fc.py --idxs 4,5,6 --out output_ppmi_pd/figures/Fig_2.tiff
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

MAT_PATH = "data/AALv3/FC_AAL_ComBat_all_163.mat"
MAT_KEY = "FC_ComBat"
GROUP = "PD"


def load_pd_entries():
    m = sio.loadmat(MAT_PATH)
    return [r for r in m[MAT_KEY].ravel()
            if str(np.asarray(r["group"]).ravel()[0]) == GROUP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", default="4,5,6", help="PD subject index (0-based), 콤마 구분")
    ap.add_argument("--out", default="output_ppmi_pd/figures/Fig_empirical_FC.tiff")
    args = ap.parse_args()

    idxs = [int(x) for x in args.idxs.split(",") if x.strip()]
    entries = load_pd_entries()
    print(f"[fig] {MAT_PATH}  PD {len(entries)}명 중 idx {idxs}")

    mats, subs = [], []
    for i in idxs:
        s = entries[i]
        sub = str(np.asarray(s["subject"]).ravel()[0]).strip()
        fc = np.nan_to_num(np.asarray(s["FC"], dtype=np.float64))
        np.fill_diagonal(fc, 0.0)
        mats.append(fc)
        subs.append(sub)
        od = ~np.eye(fc.shape[0], dtype=bool)
        print(f"   idx{i} sub{sub}: mean={fc[od].mean():+.4f} "
              f"min={fc[od].min():+.3f} max={fc[od].max():+.3f}")

    n = mats[0].shape[0]
    PT = 8.0
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
        "xtick.labelsize": PT, "ytick.labelsize": PT,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # 1단 조판 3.5 in 폭, 패널 3개 + 공용 colorbar (Fig_3 와 같은 규격).
    fig = plt.figure(figsize=(3.5, 2.0))
    gs = fig.add_gridspec(1, len(mats), wspace=0.30,
                          left=0.095, right=0.995, top=0.75, bottom=0.30)
    tick = [0, n - 1]
    im = None
    for k, (mat, sub, i) in enumerate(zip(mats, subs, idxs)):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="equal", interpolation="nearest")
        ax.set_title(f"Subject {sub}", pad=3)
        ax.set_xlabel("Brain region", labelpad=1.5)
        ax.set_xticks(tick)
        ax.set_yticks(tick)
        if k == 0:
            ax.set_ylabel("Brain region", labelpad=1.5)
        else:
            ax.set_yticklabels([])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for lab, ax in zip("abcdef", fig.axes):
        top = inv.transform(ax.title.get_window_extent(rend))[1][1]
        fig.text(ax.get_position().x0 - 0.045, top + 0.004, lab,
                 fontweight="bold", ha="left", va="bottom")

    cax = fig.add_axes([0.30, 0.10, 0.42, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Correlation coefficient", labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, size=2.0, pad=1.5)
    cb.set_ticks([-1, 0, 1])

    base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    fig.savefig(base + ".tiff", dpi=300, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    with Image.open(base + ".tiff") as t:   # 알파 채널 없는 RGB 로 (저널 요건)
        t.convert("RGB").save(base + ".tiff", dpi=(300, 300), compression="tiff_lzw")
    fig.savefig(base + ".eps", dpi=300)
    fig.savefig(base + ".png", dpi=300)
    for ext in (".tiff", ".eps", ".png"):
        print(f"[fig] saved -> {base}{ext}")


if __name__ == "__main__":
    main()
