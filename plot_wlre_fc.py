#!/usr/bin/env python3
"""plot_wlre_fc.py — idx 4/5/6 의 wLRE / wFFI / FC_emp 행렬과 회귀 산점도.

계수 출처는 formula_reproduce.load_ref 와 동일 규칙: 현재 inputs/FC.csv 해시와
일치하는 grad 캐시 중 post_grad_fc_corr 최대 (= delay3 런). 회귀는 SC>0 상삼각.

출력: output_ppmi_pd/_figs/idx<N>_matrices.png, idx<N>_regression.png
실행: python3 plot_wlre_fc.py
"""
import glob
import hashlib
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# DejaVu Sans 에 한글 글리프가 없어 라벨이 두부(□)로 나온다. Noto Sans CJK 의 .ttc 는
# index 0 인 "JP" 만 matplotlib 에 등록되는데, 한글 글리프는 CJK 전 지역판이 공유하므로
# JP 로도 정상 렌더된다.
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

N = 163
CORTEX = 112
IU = np.triu_indices(N, 1)
# 최종 데이터셋 FC_AAL_ComBat_all_163_v4_nomed.mat (262 entry, PD 206) 에 존재하고
# 현재 FC 해시와 일치하는 grad 캐시를 가진 subject 만. 101038 은 이 .mat 에 없어 제외.
# (v4_nomed 의 FC 는 현 inputs/FC.csv 와 완전 일치 확인 — 기존 최적화 유효)
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025"}
OUT = "output_ppmi_pd/_figs"

# dataviz 팔레트: 발산=blue<->red, 중립 회색 #f0efec (흰색이 아님 — surface 로 안 녹게).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"wLRE": "#2a78d6", "wFFI": "#eb6834"}   # 카테고리 slot 1/2, all-pairs 검증 통과
DIV = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#0d366b", "#184f95", "#3987e5", "#9ec5f4", "#f0efec",
     "#f5a9a8", "#e34948", "#b32c2b", "#7a1b1a"])
DIV.set_bad("#e8e7e2")   # SC==0 (가중치 없음) — 발산 스케일의 극값과 안 헷갈리게 무채색

# 가중치는 0 이 자연 원점인 크기량 → 발산이 아니라 순차(단일 hue, 밝음→어둠)가 맞다.
# dataviz 팔레트 blue 램프 100→700, 최저단을 surface 로 붙여 0 이 흰색이 되게 한다.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
     "#256abf", "#184f95", "#0d366b"])
SEQ.set_bad("#e6e4dc")   # SC==0 — 데이터(흰→파랑)보다 뒤로 물러나되 흰색과는 구분되는 무채색


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def load_ref(sub):
    """현재 FC 해시와 일치하는 grad 캐시 중 corr 최대 → (params+FC_sim, corr, tag)."""
    hs = {hashlib.sha1(np.asarray(np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv",
                                             delimiter=",")[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc, btag = None, -9.0, ""
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        dirn = os.path.basename(os.path.dirname(f))
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest")):
            continue
        if not any(h in dirn for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle", d)
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = _get(b, "params")
            best = {k: np.asarray(_get(p, k), np.float64) for k in ("wLRE", "wFFI")}
            # Part3 종료 시점 저장 FC — 재시뮬 없이 모든 subject 에서 쓸 수 있다.
            best["FC_sim"] = np.asarray(
                (_get(b, "metadata", {}) or {}).get("post_grad_fc_matrix"), np.float64)
            bc, btag = c, dirn
    return best, bc, btag


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _heat(ax, M, title, center, mask=None, v=None, vmin=None, vmax=None):
    """center=숫자면 그 값 기준 발산(DIV), center=None 이면 순차(SEQ).

    가중치처럼 0 이 자연 원점인 크기량은 center=None 으로 순차 램프를 쓴다 — 발산으로
    그리면 0 이 램프의 반대쪽 극단(진한 파랑)이 돼서 "억제"로 오독된다.
    vmin/vmax 를 주면 그대로, 없으면 v(기본 상하위 1%)로 center 대칭.
    """
    D = np.array(M, float)
    if mask is not None:
        D[~mask] = np.nan
    if center is None:
        lo = float(np.nanmin(D)) if vmin is None else vmin
        hi = float(np.nanmax(D)) if vmax is None else vmax
        im = ax.imshow(D, cmap=SEQ, vmin=lo, vmax=hi, interpolation="nearest")
        _finish_heat(ax, im, title)
        return im
    if v is None:
        v = np.nanpercentile(np.abs(D - center), 99)
    lo = center - v if vmin is None else vmin
    hi = center + v if vmax is None else vmax
    im = ax.imshow(D, cmap=DIV, norm=TwoSlopeNorm(center, lo, hi),
                   interpolation="nearest")
    _finish_heat(ax, im, title)
    return im


def _finish_heat(ax, im, title):
    for pos in (CORTEX - 0.5,):        # cortex(112) | subcortex(51) 경계
        ax.axhline(pos, color=INK, lw=0.6, alpha=0.35)
        ax.axvline(pos, color=INK, lw=0.6, alpha=0.35)
    ax.set_title(title, color=INK, fontsize=11, pad=8)
    _style(ax)
    ax.set_xticks([0, CORTEX, N - 1])
    ax.set_yticks([0, CORTEX, N - 1])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(colors=MUTED, labelsize=8)


def fit(x, y):
    (b, a), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
    r2 = 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return a, b, r2


def main():
    os.makedirs(OUT, exist_ok=True)
    for idx, sub in SUBDIR.items():
        ref, corr, tag = load_ref(sub)
        if ref is None:
            print(f"idx {idx}: 해시 일치 캐시 없음 → skip")
            continue
        W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
        np.fill_diagonal(FCe, 0.0)
        scm = W > 0
        wl, wf = ref["wLRE"], ref["wFFI"]

        # ---- 행렬 4장 -------------------------------------------------------
        # FC_sim = 위 wLRE/wFFI(+optimized c_ei) 로 σ=0.02 재시뮬한 것.
        # formula_reproduce.py 가 저장해 둔 것을 그대로 쓴다(재시뮬 불필요).
        FCs = ref.get("FC_sim")
        if FCs is not None and not np.isfinite(np.asarray(FCs)).any():
            FCs = None
        ncol = 4 if FCs is not None else 3
        fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.3), facecolor=SURFACE)
        # 가중치: 하한 0(물리적 바닥) 고정, 1.0 기준 발산, 상한은 둘 공유.
        vwmax = max(float(np.nanmax(w[scm])) for w in (wl, wf))
        _heat(axes[0], wl, "wLRE  (Part3 최종)", None, scm, vmin=0.0, vmax=vwmax)
        _heat(axes[1], wf, "wFFI  (Part3 최종)", None, scm, vmin=0.0, vmax=vwmax)
        # FC: ±1 고정 — subject·조건 간 색이 그대로 비교된다.
        _heat(axes[2], FCe, "FC_emp  (경험 FC)", 0.0, vmin=-1.0, vmax=1.0)
        rsim = np.nan
        if FCs is not None:
            rsim = float(np.corrcoef(np.asarray(FCs)[IU], FCe[IU])[0, 1])
            _heat(axes[3], FCs, f"FC_sim  (재시뮬, σ=0.02)   r={rsim:+.4f}", 0.0,
                  vmin=-1.0, vmax=1.0)
        fig.suptitle(
            f"idx {idx} (sub {sub})  —  wLRE / wFFI / FC_emp / FC_sim     "
            f"post_grad corr = {corr:.4f}",
            color=INK, fontsize=13, y=0.99)
        fig.text(0.5, 0.015,
                 f"wLRE·wFFI 는 순차 램프(흰색=0, 진할수록 강한 결합), 스케일 [0, {vwmax:.2f}] 공유, "
                 f"회색=SC 엣지 없음 ({(~scm).sum()/N**2*100:.0f}% 셀). "
                 f"FC_emp·FC_sim 은 0 기준·스케일 [-1, +1] 고정, 전체 엣지. "
                 f"검은 선 = cortex(0–111) | subcortex(112–162) 경계.   cache: {tag}",
                 ha="center", color=INK2, fontsize=8)
        fig.tight_layout(rect=[0, 0.045, 1, 0.96])
        p1 = f"{OUT}/idx{idx}_matrices.png"
        fig.savefig(p1, dpi=150, facecolor=SURFACE)
        plt.close(fig)

        # ---- FC 단위로 되돌린 가중치 ------------------------------------------
        # w = a + b*FC_emp 의 역변환 FC_hat = (w - a)/b. 최대·최소 고정 + 비례
        # 선형변환이라 패턴이 보존되고, 계수는 이 subject 의 회귀에서 나온다.
        # wFFI 는 b<0 이라 자동으로 부호가 뒤집혀 FC 와 같은 방향이 된다.
        m_ = scm[IU]
        aL, bL, _ = fit(FCe[IU][m_], wl[IU][m_])
        aF, bF, _ = fit(FCe[IU][m_], wf[IU][m_])
        hatL, hatF = (wl - aL) / bL, (wf - aF) / bF
        rL = float(np.corrcoef(hatL[IU][m_], FCe[IU][m_])[0, 1])
        rF = float(np.corrcoef(hatF[IU][m_], FCe[IU][m_])[0, 1])

        fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.3), facecolor=SURFACE)
        _heat(ax[0], hatL, f"wLRE → FC 단위   r={rL:+.4f}", 0.0, scm, vmin=-1.0, vmax=1.0)
        _heat(ax[1], hatF, f"wFFI → FC 단위   r={rF:+.4f}", 0.0, scm, vmin=-1.0, vmax=1.0)
        _heat(ax[2], FCe, "FC_emp  (경험 FC)", 0.0, vmin=-1.0, vmax=1.0)
        if FCs is not None:
            _heat(ax[3], FCs, f"FC_sim  (재시뮬)   r={rsim:+.4f}", 0.0, vmin=-1.0, vmax=1.0)
        fig.suptitle(f"idx {idx} (sub {sub})  —  가중치를 FC 단위로 역변환, 전부 [-1,+1] 스케일",
                     color=INK, fontsize=13, y=0.99)
        fig.text(0.5, 0.015,
                 f"wLRE→FC: (w {-aL:+.3f}) / {bL:+.3f}     "
                 f"wFFI→FC: (w {-aF:+.3f}) / {bF:+.3f}     "
                 f"(b<0 이라 wFFI 는 부호가 되돌려짐).  회색=SC 엣지 없음 — 가중치가 존재하지 "
                 f"않는 칸이라 FC_emp/FC_sim 패널에만 값이 있다.",
                 ha="center", color=INK2, fontsize=8)
        fig.tight_layout(rect=[0, 0.045, 1, 0.96])
        p3 = f"{OUT}/idx{idx}_matrices_fcunits.png"
        fig.savefig(p3, dpi=150, facecolor=SURFACE)
        plt.close(fig)

        # ---- 회귀 산점도 ----------------------------------------------------
        m = scm[IU]
        x = FCe[IU][m]
        ys = {"wLRE": wl[IU][m], "wFFI": wf[IU][m]}
        lo = min(v.min() for v in ys.values())
        hi = max(v.max() for v in ys.values())
        pad = 0.05 * (hi - lo)

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4), facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [2.4, 1]})
        QS = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        for c, nm in enumerate(("wLRE", "wFFI")):
            ax, axr = axes[0, c], axes[1, c]
            y = ys[nm]
            a, b, r2 = fit(x, y)
            q = np.polyfit(x, y, 2)
            yq = np.polyval(q, x)
            r2q = 1 - np.sum((y - yq) ** 2) / np.sum((y - y.mean()) ** 2)
            ax.scatter(x, y, s=2.0, alpha=0.10, color=SERIES[nm], linewidths=0,
                       rasterized=True)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, a + b * xs, color=INK, lw=2, zorder=3, label="선형")
            ax.plot(xs, np.polyval(q, xs), color=INK, lw=1.6, ls="--", zorder=3,
                    label="2차")
            ax.axhline(1.0, color=MUTED, lw=1, ls=":", zorder=2)
            # 선형 적합이 상단 극단을 얼마나 놓치는지 — 실측 max 와 예측을 잇는다
            xm = x[np.argmax(y)]
            pm = a + b * xm
            ax.plot([xm, xm], [pm, y.max()], color="#b32c2b", lw=1.6, zorder=4)
            ax.plot([xm], [y.max()], "o", ms=6, mfc="none", mec="#b32c2b", mew=1.6,
                    zorder=4)
            ax.text(xm, y.max(), f"  max {y.max():.2f}\n  (선형예측 {pm:.2f})",
                    color="#b32c2b", fontsize=8, va="center", ha="left")
            nclip = int((y <= 1e-6).sum())
            if nclip:
                ax.axhline(0.0, color="#b32c2b", lw=1, ls="--", zorder=2)
                ax.text(0.97, 0.06, f"y=0 클리핑 {nclip:,} ({nclip/len(y)*100:.1f}%)",
                        transform=ax.transAxes, va="bottom", ha="right",
                        color="#b32c2b", fontsize=8.5)
            ax.set_title(f"{nm}  vs  FC_emp", color=INK, fontsize=12, pad=8)
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(True, color=GRID, lw=0.7)
            ax.set_axisbelow(True)
            _style(ax)
            ax.text(0.03, 0.96,
                    f"선형  {a:+.3f} {b:+.3f}·FC      R² {r2:.4f}\n"
                    f"2차   {q[2]:+.3f} {q[1]:+.3f}·FC {q[0]:+.3f}·FC²   R² {r2q:.4f}\n"
                    f"n = {len(x):,}",
                    transform=ax.transAxes, va="top", ha="left", color=INK, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, lw=0.8))
            leg = ax.legend(fontsize=8, loc="lower right", frameon=True)
            leg.get_frame().set_edgecolor(GRID)
            leg.get_frame().set_facecolor(SURFACE)
            for t in leg.get_texts():
                t.set_color(INK)

            # 잔차 진단: FC 분위 구간별 평균. 선형이면 전 구간 0 — U 자면 곡률이다.
            r = y - (a + b * x)
            ed = np.percentile(x, QS)
            cen, mu = [], []
            for i in range(len(QS) - 1):
                sel = ((x >= ed[i]) & (x <= ed[i + 1])) if i == len(QS) - 2 else \
                      ((x >= ed[i]) & (x < ed[i + 1]))
                if sel.any():
                    cen.append(0.5 * (QS[i] + QS[i + 1]))
                    mu.append(r[sel].mean())
            axr.axhline(0.0, color=MUTED, lw=1)
            axr.plot(cen, mu, "-o", color=SERIES[nm], lw=1.8, ms=5)
            axr.set_xlabel("FC_emp 분위 (%)", color=INK2, fontsize=10)
            axr.set_title("선형 적합 잔차 (구간 평균)", color=INK, fontsize=10, pad=6)
            axr.grid(True, color=GRID, lw=0.7)
            axr.set_axisbelow(True)
            _style(axr)
        axes[0, 0].set_ylabel("가중치", color=INK2, fontsize=10)
        axes[1, 0].set_ylabel("잔차", color=INK2, fontsize=10)
        fig.suptitle(f"idx {idx} (sub {sub})  —  엣지별 회귀 (SC>0 상삼각, n={len(x):,})",
                     color=INK, fontsize=13, y=0.985)
        fig.text(0.5, 0.012,
                 "상단: 점선=초기값 1.0, 실선=선형 적합, 파선=2차 적합. 빨간 표시=실측 max 와 "
                 "선형예측의 갭.   하단: 잔차가 U 자면 위로 볼록한 곡률 — 양끝(특히 상위 1%)에서 "
                 "선형이 과소평가한다.",
                 ha="center", color=INK2, fontsize=8)
        fig.tight_layout(rect=[0, 0.045, 1, 0.95])
        p2 = f"{OUT}/idx{idx}_regression.png"
        fig.savefig(p2, dpi=150, facecolor=SURFACE)
        plt.close(fig)

        aL, bL, r2L = fit(x, ys["wLRE"])
        aF, bF, r2F = fit(x, ys["wFFI"])
        print(f"idx {idx} ({sub})  corr={corr:.4f}  n={len(x):,}\n"
              f"  wLRE = {aL:+.4f} {bL:+.4f}*FC  R2={r2L:.4f}\n"
              f"  wFFI = {aF:+.4f} {bF:+.4f}*FC  R2={r2F:.4f}\n"
              f"  -> {p1}\n  -> {p3}\n  -> {p2}")


if __name__ == "__main__":
    main()
