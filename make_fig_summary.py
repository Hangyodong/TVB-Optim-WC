#!/usr/bin/env python3
"""FC 행렬 figure — 실측 / 최적화 전 / 최적화 후 를 한 줄에.

사용:
    python3 make_fig_summary.py                 # sub 100001(idx0), 최신 캐시
    python3 make_fig_summary.py --sub 100878    # idx4
    python3 make_fig_summary.py --sub 100001 --cache-tag v_pdaal163_tr25_s0_N163_sc2d64c65cfb_fc240edad76c
"""
import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 강조 표시할 edge (region_labels.txt 의 이름 그대로). subject 별로 다르므로 sub_num 으로 건다.
HIGHLIGHT_EDGES = {
    "100878": [   # idx4 — CEI–UPDRS 유의 edge 13/20
        ("Precentral_L",         "Thal_VPL_R"),
        ("Precentral_L",         "Thal_PuA_R"),
        ("Frontal_Sup_2_L",      "Thal_LP_R"),
        ("Frontal_Sup_2_L",      "Thal_PuA_L"),
        ("Frontal_Sup_2_L",      "Thal_PuL_L"),
        ("Frontal_Sup_2_R",      "Vermis_6"),
        ("Frontal_Sup_2_R",      "Thal_LP_R"),
        ("Frontal_Inf_Oper_R",   "Paracentral_Lobule_L"),
        ("Frontal_Sup_Medial_R", "Thal_LP_R"),
        ("Frontal_Med_Orb_L",    "Cerebellum_4_5_L"),
        ("Paracentral_Lobule_L", "Thal_VA_L"),
        ("Paracentral_Lobule_L", "Thal_VL_L"),
        ("Paracentral_Lobule_L", "Thal_VL_R"),
    ],
}


def fc_corr(a, b):
    """off-diag Pearson corr."""
    od = ~np.eye(a.shape[0], dtype=bool)
    return float(np.corrcoef(a[od], b[od])[0, 1])


def load_pkl(cache_dir, prefix):
    hits = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_*.pkl")), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"{prefix}_*.pkl 없음 → {cache_dir}")
    return pickle.load(open(hits[-1], "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="100001", help="subject 폴더명(sub_num)")
    ap.add_argument("--root", default="output_ppmi_pd")
    ap.add_argument("--cache-tag", default=None, help="캐시 폴더명. 미지정=fic/eib/grad 다 있는 최신 것")
    ap.add_argument("--out", default=None, help="저장 경로(기본 <sub>/figures/Fig_3.tiff)")
    ap.add_argument("--mark-edges", action="store_true",
                    help="HIGHLIGHT_EDGES 의 edge 를 검은 테두리 상자로 표시(기본 off)")
    args = ap.parse_args()

    sub_dir = os.path.join(args.root, args.sub)
    in_dir = os.path.join(sub_dir, "inputs")

    if args.cache_tag:
        cache_dir = os.path.join(sub_dir, "cache", args.cache_tag)
    else:
        cands = [d for d in glob.glob(os.path.join(sub_dir, "cache", "*"))
                 if glob.glob(os.path.join(d, "fic_*.pkl")) and glob.glob(os.path.join(d, "grad_*.pkl"))]
        if not cands:
            raise FileNotFoundError(f"fic+grad 캐시 있는 폴더 없음 → {sub_dir}/cache")
        cache_dir = max(cands, key=os.path.getmtime)
    print(f"[fig] cache = {os.path.basename(cache_dir)}")

    fc_emp = np.loadtxt(os.path.join(in_dir, "FC.csv"), delimiter=",")
    np.fill_diagonal(fc_emp, 0.0)

    labels = [l.strip() for l in open(os.path.join(in_dir, "region_labels.txt")) if l.strip()]
    edges = []
    for a, b in (HIGHLIGHT_EDGES.get(args.sub, []) if args.mark_edges else []):
        # 이름이 안 맞으면 조용히 넘어가지 않고 바로 실패 — 그림에 엉뚱한 칸이 찍히면 안 된다.
        edges.append((labels.index(a), labels.index(b)))

    fic = load_pkl(cache_dir, "fic")
    grad = load_pkl(cache_dir, "grad")
    fc_pre = np.asarray(fic["pre_fic_fc"], dtype=np.float64)
    fc_post = np.asarray(grad["post_opt_fc"], dtype=np.float64)

    n = fc_emp.shape[0]

    # 저널 서식: 모든 텍스트 8 pt Times New Roman. TNR 은 이 시스템에 없어서 metric 호환
    # 대체(Liberation Serif / Nimbus Roman)로 렌더된다 — 폴백 체인 맨 앞에 TNR 을 둬서
    # TNR 있는 환경에서 다시 뽑으면 진짜 TNR 로 나온다.
    PT = 8.0
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
        "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        # 글꼴 임베드(TrueType) — EPS/PDF 제출본 요건.
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # 폭은 1단 조판 기준 3.5 in. FC 행렬 3개를 한 줄에 + 공용 colorbar 하나.
    # (패널당 폭이 0.9 in 뿐이라 패널마다 colorbar 를 달면 행렬이 뭉개진다.)
    # 패널 박스를 정사각(0.9 x 0.9 in)으로 딱 맞춰 잡는다. 박스 비율이 어긋나면
    # aspect="equal" 이 남는 공간을 위아래 여백으로 흘려서 제목과 행렬이 벌어진다.
    fig = plt.figure(figsize=(3.5, 2.0))
    gs = fig.add_gridspec(1, 3, wspace=0.30,
                          left=0.095, right=0.995, top=0.75, bottom=0.30)
    ax_e = fig.add_subplot(gs[0, 0])
    ax_pre = fig.add_subplot(gs[0, 1])
    ax_post = fig.add_subplot(gs[0, 2])

    # colormap 은 파이프라인 plot 규약을 따른다(FC = RdBu_r).
    # FC 범위는 Config.fc_plot_vmin/vmax(-1..1) 고정.
    tick = [0, n - 1]
    # 강조 edge 는 한 칸(163분의 1 ≈ 0.006 in)이라 그대로 그리면 안 보인다 → 9칸짜리
    # 테두리 상자로 키운다. 행렬이 대칭이라 (i,j)/(j,i) 둘 다 표시.
    BOX = 4.5

    def panel(ax, mat, title, cmap, vmin=None, vmax=None, ylabel=False):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        ax.set_title(title, pad=3)
        # 축 라벨은 기호·약어 없이 완전한 단어로 적는다(저널 규정).
        ax.set_xlabel("Brain region", labelpad=1.5)
        if ylabel:
            ax.set_ylabel("Brain region", labelpad=1.5)
        ax.set_xticks(tick)
        ax.set_yticks(tick)
        if not ylabel:
            ax.set_yticklabels([])   # 눈금값은 맨 왼쪽 패널만 — 옆 패널과 붙는다
        for i, j in edges:
            for r, c in ((i, j), (j, i)):
                ax.add_patch(mpatches.Rectangle(
                    (c - BOX, r - BOX), 2 * BOX, 2 * BOX,
                    fill=False, edgecolor="black", linewidth=0.35))
        return im

    # 실측 / 최적화 전 / 최적화 후 기능적 연결성. 패널 폭이 0.9 in 라 제목은 줄당 12자
    # 이내로 끊고, 세 패널 모두 3줄로 맞춰 행렬 상단을 정렬한다.
    # 제목은 2줄로 통일(빈 줄 1개로 높이를 맞춤) → 세 패널 제목 상단이 정렬되고
    # 제목-행렬 간격이 최소가 된다. "simulated" 라는 설명은 caption 몫.
    im = panel(ax_e, fc_emp, "Empirical\nFC matrix", "RdBu_r", -1, 1, ylabel=True)

    for ax, mat, when in ((ax_pre, fc_pre, "Before"), (ax_post, fc_post, "After")):
        m = np.asarray(mat, dtype=np.float64).copy()
        np.fill_diagonal(m, 0.0)
        # 컬러스케일은 실측과 같은 -1..1 고정(패널 간 비교용) → 최적화 전은 값이 작아
        # 균일해 보인다(실제 범위는 caption 에).
        panel(ax, m, f"{when} optimization\nr = {fc_corr(m, fc_emp):+.3f}",
              "RdBu_r", -1, 1)

    # 공용 colorbar (세 패널 스케일이 같으므로 하나면 충분).
    cax = fig.add_axes([0.30, 0.10, 0.42, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Correlation coefficient", labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, size=2.0, pad=1.5)
    cb.set_ticks([-1, 0, 1])

    # 전체 제목은 넣지 않는다 — 저널 서식상 그림 설명은 본문 caption 이 맡는다.

    # 패널 문자는 각 패널 좌상단 바깥(제목 블록 위쪽 왼쪽). 제목 줄 수가 1~4줄로 달라서
    # 고정 오프셋으로는 어긋난다 → 렌더 후 실제 제목 bbox 를 재서 그 위에 맞춘다.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for lab, ax in zip("abc", (ax_e, ax_pre, ax_post)):
        top = inv.transform(ax.title.get_window_extent(rend))[1][1]
        fig.text(ax.get_position().x0 - 0.045, top + 0.004, lab,
                 fontweight="bold", ha="left", va="bottom")

    out = args.out or os.path.join(sub_dir, "figures", "Fig_3.tiff")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    base = os.path.splitext(out)[0]
    saved = []
    # 제출본: 300 dpi TIFF(LZW 무손실) + 글꼴 임베드 EPS. 미리보기용 PNG 도 같이.
    fig.savefig(base + ".tiff", dpi=300, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    # matplotlib 은 RGBA 로 쓴다 — alpha 채널 있는 TIFF 를 거부하는 저널이 있어 RGB 로 평탄화.
    from PIL import Image
    with Image.open(base + ".tiff") as _tif:
        _tif.convert("RGB").save(base + ".tiff", dpi=(300, 300), compression="tiff_lzw")
    saved.append(base + ".tiff")
    fig.savefig(base + ".eps", dpi=300)
    saved.append(base + ".eps")
    fig.savefig(base + ".png", dpi=300)
    saved.append(base + ".png")
    for f in saved:
        print(f"[fig] saved -> {f}")


if __name__ == "__main__":
    main()
