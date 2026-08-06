#!/usr/bin/env python3
"""make_fig_sc_fc.py — 보정 전/후 SC 와 경험 FC 행렬 figure.

열: raw SC(선형 색축) | 보정 SC(cfg.sc_norm=log1pm) | optimized wLRE | wFFI | empirical FC
행: subject idx

색 배정 규칙
  SC       = 크기(magnitude) -> 단일 색상 sequential (Blues), 열 공통 축
  wLRE/wFFI= 크기            -> 단일 색상 sequential (Purples), **두 열 공통 축**(상보 관계라
                                같은 축이어야 비교됨). SC 와 다른 색상 = 다른 물리량 표시
  FC       = 극성(polarity)  -> 발산형 (RdBu_r), 0 이 중립 중앙, ±대칭

wLRE/wFFI 는 Part3 grad 캐시(본선)의 최종 파라미터 — [[stage_trace]] 와 같은 캐시 선택.

실행: python3 make_fig_sc_fc.py --idxs 4,5,6,7,8 --out figures/sc_fc_matrices.png
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import main_ppmi_pd as M
from data_loader import load_data
from formula_2x2 import SUBDIR
from stage_trace import plain_cache_dir, newest, load_bundle


def collect(idx, noise):
    p = M.prepare_pd_data(idx, noise)
    cfg = M.make_config(p, idx, use_delay=True)
    data = load_data(cfg)
    fc = np.asarray(data["fc_target"], np.float32).copy()
    np.fill_diagonal(fc, 0.0)
    raw = np.loadtxt(p["sc_csv"], delimiter=",")
    np.fill_diagonal(raw, 0.0)

    sub = SUBDIR[idx]
    cdir = plain_cache_dir(sub)
    grad = newest(glob.glob(f"{cdir}/grad_*.pkl")) if cdir else None
    if grad is None:
        raise SystemExit(f"idx{idx} ({sub}): 본선 grad 캐시 없음 — optimized 가중치 불가")
    b, md = load_bundle(grad)
    return dict(idx=idx, sub=sub, raw=raw,
                norm=np.asarray(data["weights"], np.float32),
                wlre=np.asarray(b.params.wLRE, np.float32),
                wffi=np.asarray(b.params.wFFI, np.float32),
                corr=float(md.get("post_grad_fc_corr", np.nan)),
                fc=np.nan_to_num(fc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", default="4,5,6,7,8")
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--out", default="figures/sc_fc_matrices.png")
    ap.add_argument("--mask-sc0", action="store_true",
                    help="SC==0 셀을 회색 처리 (구조 유래 4개 열만). 공유 0-마스크가 만드는 "
                         "허위 유사성을 제거해 '존재하는 엣지끼리'만 비교하게 한다.")
    a = ap.parse_args()

    rows = [collect(int(v), a.noise_level) for v in a.idxs.split(",") if v.strip()]
    n = len(rows)

    # 열별 공통 색축 — subject 간 비교가 가능해야 한다.
    raw_pos = np.concatenate([r["raw"][r["raw"] > 0].ravel() for r in rows])
    raw_hi = np.percentile(raw_pos, 99.5)
    norm_hi = np.percentile(np.concatenate(
        [r["norm"][r["norm"] > 0].ravel() for r in rows]), 99.5)
    # wLRE/wFFI 는 상보 관계라 **하나의 공통 축**이어야 서로 비교된다.
    w_hi = float(np.percentile(np.concatenate(
        [r[k][r[k] > 0].ravel() for r in rows for k in ("wlre", "wffi")]), 99.5))
    fc_lim = float(np.percentile(np.abs(np.concatenate(
        [r["fc"].ravel() for r in rows])), 99.5))

    # raw 는 선형 색축으로 둔다 — 로그로 그리면 보정과 거의 같아 보여서
    # heavy-tail 이라는 보정 이유가 그림에서 사라진다.
    cols = [
        ("raw SC\n(streamline count, linear)", "raw", "Blues", (0.0, raw_hi)),
        ("corrected SC\n(log1pm, node-input rescaled)", "norm", "Blues", (0.0, norm_hi)),
        ("optimized wLRE\n(Part3, long-range excitation)", "wlre", "Purples", (0.0, w_hi)),
        ("optimized wFFI\n(Part3, feedforward inhibition)", "wffi", "Purples", (0.0, w_hi)),
        ("empirical FC", "fc", "RdBu_r", (-fc_lim, fc_lim)),
    ]

    fig, axes = plt.subplots(n, len(cols), figsize=(3.7 * len(cols), 3.6 * n))
    axes = np.atleast_2d(axes)
    ims = [None] * len(cols)

    for i, r in enumerate(rows):
        sc0 = r["norm"] <= 0                    # 구조 연결이 없는 셀
        for j, (ttl, key, cmap, lim) in enumerate(cols):
            ax = axes[i, j]
            M = r[key]
            cm = plt.get_cmap(cmap).copy()
            # FC(마지막 열)는 SC==0 에서도 실측값이 있으므로 가리지 않는다.
            if a.mask_sc0 and key != "fc":
                M = np.ma.masked_where(sc0, M)
                cm.set_bad("0.80")
            ims[j] = ax.imshow(M, cmap=cm, aspect="equal",
                               interpolation="nearest", vmin=lim[0], vmax=lim[1])
            # cortex(0..111) / subcortex(112..) 경계 — 옅은 안내선
            ax.axhline(111.5, color="0.35", lw=0.6, alpha=0.55)
            ax.axvline(111.5, color="0.35", lw=0.6, alpha=0.55)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color("0.7")
            if i == 0:
                ax.set_title(ttl, fontsize=10, pad=8)
            if j == 0:
                ax.set_ylabel(f"idx {r['idx']}\n{r['sub']}\nFC r={r['corr']:.3f}",
                              fontsize=9.5, rotation=0, ha="right", va="center",
                              labelpad=30)

    sub_note = ("  |  grey = SC == 0 (no structural connection; masked in the four "
                "structure-derived columns, kept in FC where values are real)"
                if a.mask_sc0 else "")
    fig.suptitle("Structure → optimized coupling → function — PPMI PD, AAL163 "
                 "(line at cortex / subcortex boundary)" + sub_note,
                 fontsize=11.5 if a.mask_sc0 else 12.5, y=0.995)
    fig.tight_layout(rect=(0.0, 0.042, 1.0, 0.985))

    # colorbar — 하단 가로 배치. wLRE/wFFI(열 2-3)는 축이 같아 하나로 합친다.
    bars = [(0, 0, "streamlines"), (1, 1, "corrected weight"),
            (2, 3, "optimized weight (wLRE / wFFI, shared scale)"),
            (4, 4, "Pearson r")]
    for j0, j1, label in bars:
        a0 = axes[-1, j0].get_position()
        a1 = axes[-1, j1].get_position()
        cax = fig.add_axes([a0.x0, 0.020, a1.x1 - a0.x0, 0.0085])
        cb = fig.colorbar(ims[j0], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=8, length=2, width=0.6)
        cb.outline.set_linewidth(0.6)
        cb.set_label(label, fontsize=8.5, labelpad=2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\n[fig] {a.out}")
    print(f"  raw SC       color range 0 ~ {raw_hi:.0f}")
    print(f"  corrected SC color range 0 ~ {norm_hi:.4f}")
    print(f"  wLRE/wFFI    color range 0 ~ {w_hi:.3f} (공통)")
    print(f"  emp FC       color range ±{fc_lim:.3f}")
    for r in rows:
        m = r["norm"] > 0
        wl, wf = r["wlre"][m], r["wffi"][m]
        print(f"  idx{r['idx']}: SC>0 {m.mean()*100:.1f}%  "
              f"node input {r['norm'].sum(1).mean():.2f}  |  "
              f"wLRE {wl.mean():.3f}±{wl.std():.3f} (0인 엣지 {(wl<=1e-6).mean()*100:.1f}%)  "
              f"wFFI {wf.mean():.3f}±{wf.std():.3f} ({(wf<=1e-6).mean()*100:.1f}%)  |  "
              f"FC r={r['corr']:.4f}")


if __name__ == "__main__":
    main()
