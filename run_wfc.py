#!/usr/bin/env python3
"""run_wfc.py — wLRE = FC_emp, wFFI = -FC_emp 로 두고 시뮬 FC 추출.

절편 0 / 계수 1 의 엄밀한 비례형. sanitize 가 [0, cap] 으로 클리핑하므로
FC_emp>0 인 엣지에서 wFFI 는 전부 0, FC_emp<0 인 엣지에서 wLRE 가 0 이 된다.
초기값 1.0 대비 전 엣지가 약해지는 조건이라 결합 자체가 크게 줄어든다.

c_ei 는 이 가중치 *위에서* FIC 로 얻는다 (c_ei=1 로 두면 corr 이 0 으로 붕괴).

출력: output_ppmi_pd/_wfc/idx<N>.npz, _wfc/summary.json,
      output_ppmi_pd/_figs/idx<N>_wfc_matrices.png
실행: python3 run_wfc.py --idxs 4,5,6
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from formula_reproduce import IU, SUBDIR, pair
from plot_wlre_fc import CORTEX, INK, INK2, N, SURFACE, _heat, fit, load_ref

OUT, FIGS = "output_ppmi_pd/_wfc", "output_ppmi_pd/_figs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", type=str, default="4,5,6")
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--frozen-part1", action="store_true",
                    help="c_ei 를 표준 파이프라인 Part1(identity 가중치 위 FIC) 값으로 "
                         "쓰고 freeze. 캐시 히트라 FIC 재실행 없음. 미지정이면 이 "
                         "가중치 위에서 FIC 를 새로 돌린다.")
    ap.add_argument("--no-clip", action="store_true",
                    help="sanitize 의 clip(0, w_max) 를 건너뛰고 SC 마스크+대칭화만 "
                         "적용. wFFI=-FC_emp 의 음수가 그대로 모델에 들어간다 "
                         "(파이프라인이 한 번도 밟은 적 없는 영역).")
    ap.add_argument("--no-mask", action="store_true",
                    help="sanitize 의 * sc_mask 도 건너뛴다. 그래프가 이미 SC 를 곱하므로"
                         "(coupling/linear.py:123) 수치적으로 중복 — FC_sim 은 동일해야"
                         " 한다. wLRE 를 FC_emp 와 문자 그대로 같게 만들기 위한 것.")
    ap.add_argument("--weights", type=str, default="fc", choices=["fc", "hat", "corr1"],
                    help="fc = scale*FC_emp / -scale*FC_emp. "
                         "hat = Part3 최적 가중치를 회귀 역변환해 FC 단위로 되돌린 것 "
                         "((w-a)/b). 잔차를 품고 있어 순수 FC_emp 와 미세하게 다르다. "
                         "b<0 이라 wFFI 는 부호가 뒤집혀 FC 와 같은 방향이 된다. "
                         "corr1 = FC_emp 의 선형사상으로 대체하되 min/max 를 Part3 최적 "
                         "가중치의 것에 고정 → corr 이 정확히 +1/-1. 잔차 0, E/I 대립 유지.")
    ap.add_argument("--wmax-mult", type=str, default="1,1",
                    help="corr1 전용. 'L,F' — Part3 실측 max 에 곱할 배수. "
                         "1,1 이면 실측 max 그대로. 이 두 스칼라가 사실상 유일한 자유도.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="wLRE = scale*FC_emp, wFFI = -scale*FC_emp. 절편 0 을 유지한 채 "
                         "가중치 크기만 키워 '스케일 부족' 가설을 검정한다.")
    ap.add_argument("--sc-norm", type=str, default=None,
                    choices=["log1p", "max", "log1pm"],
                    help="SC 정규화. 기본(미지정)=cfg 값 log1pm. 'max'=w/max 로 log 보정 "
                         "없음. 정규화가 바뀌면 SC 해시가 달라져 캐시도 자동 분리된다.")
    a = ap.parse_args()
    sfx = ((f"_{a.weights}" if a.weights != "fc" else "")
           + ("_frozen" if a.frozen_part1 else "")
           + (f"_s{a.scale:g}".replace(".", "p") if a.scale != 1.0 else "")
           + ("" if a.wmax_mult == "1,1" else "_wm" + a.wmax_mult.replace(",", "-").replace(".", "p"))
           + (f"_sc{a.sc_norm}" if a.sc_norm else "")
           + ("_noclip" if a.no_clip else "") + ("_nomask" if a.no_mask else ""))

    def prep(v, mask, w_max):
        """플래그에 따라 sanitize 의 clip / 마스킹을 선택적으로 건너뛴다."""
        v = np.nan_to_num(np.asarray(v, np.float32), nan=0.0, posinf=w_max, neginf=0.0)
        if not a.no_clip:
            v = np.clip(v, 0.0, w_max)
        if not a.no_mask:
            v = v * np.asarray(mask, np.float32)
        return (0.5 * (v + v.T)).astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    rows = []
    for idx in [int(x) for x in a.idxs.split(",") if x.strip()]:
        sub = SUBDIR[idx]
        print(f"\n{'='*72}\nidx {idx} ({sub})  wLRE=FC_emp, wFFI=-FC_emp", flush=True)

        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)
        if a.frozen_part1:
            # cache_version 을 건드리지 않아야 표준 파이프라인 Part1 캐시를 물어온다.
            # freeze 플래그는 FIC cache_name 에 없어서(part1_fic.py:86-93) 캐시 무효화
            # 없이 재적용된다 — 즉 FIC 재실행 0 초.
            cfg.freeze_c_ei_after_fic = True
        else:
            cfg.cache_version = f"{cfg.cache_version}_wfc"
        if a.sc_norm:
            cfg.sc_norm = a.sc_norm
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
        network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
        mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]

        if a.weights == "hat":
            ref, _c, _t = load_ref(sub)
            m_ = (np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv",
                             delimiter=",") > 0)[IU]
            x = FC_emp[IU][m_].astype(float)
            aL, bL, _ = fit(x, ref["wLRE"][IU][m_])
            aF, bF, _ = fit(x, ref["wFFI"][IU][m_])
            SRC_L = (ref["wLRE"] - aL) / bL
            SRC_F = (ref["wFFI"] - aF) / bF
            print(f"  hat 역변환: wLRE=(w{-aL:+.4f})/{bL:+.4f}  wFFI=(w{-aF:+.4f})/{bF:+.4f}",
                  flush=True)
        elif a.weights == "corr1":
            ref, _c, _t = load_ref(sub)
            m_ = (np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv",
                             delimiter=",") > 0)[IU]
            x = FC_emp[IU][m_].astype(float)
            lo, hi = float(x.min()), float(x.max())
            u = (FC_emp - lo) / (hi - lo)          # [0,1] 정규화
            Lmin, Lmax = float(ref["wLRE"][IU][m_].min()), float(ref["wLRE"][IU][m_].max())
            Fmin, Fmax = float(ref["wFFI"][IU][m_].min()), float(ref["wFFI"][IU][m_].max())
            mL, mF = (float(t) for t in a.wmax_mult.split(","))
            Lmax, Fmax = Lmax * mL, Fmax * mF
            SRC_L = Lmin + u * (Lmax - Lmin)       # FC 증가 → wLRE 증가 (corr=+1)
            SRC_F = Fmax - u * (Fmax - Fmin)       # FC 증가 → wFFI 감소 (corr=-1), E/I 대립 유지
            print(f"  corr1: wLRE [{Lmin:.3f},{Lmax:.3f}]  wFFI [{Fmin:.3f},{Fmax:.3f}] "
                  f"(FC [{lo:.3f},{hi:.3f}] 선형사상)", flush=True)
        else:
            SRC_L, SRC_F = FC_emp, -FC_emp

        # FIC 입력과 시뮬 입력이 같은 가공을 받아야 한다 (prep 일원화).
        ps = ParamSet(c_ei=np.ones(n, np.float32),
                      wLRE=prep(a.scale * SRC_L, mask, cap),
                      wFFI=prep(a.scale * SRC_F, mask, cap),
                      c_ei_frozen=False)
        scm = mask.astype(bool)
        wl, wf = np.asarray(ps.wLRE, float), np.asarray(ps.wFFI, float)
        print(f"  가공 후 SC>0 엣지: wLRE mean={wl[scm].mean():+.4f} "
              f"음수={np.mean(wl[scm] < 0)*100:.1f}% 0={np.mean(wl[scm] == 0)*100:.1f}%   "
              f"wFFI mean={wf[scm].mean():+.4f} "
              f"음수={np.mean(wf[scm] < 0)*100:.1f}% 0={np.mean(wf[scm] == 0)*100:.1f}%",
              flush=True)

        # frozen 모드: FIC 는 파이프라인과 똑같이 identity 가중치 위에서 (캐시 히트).
        # 기본 모드: 이 wfc 가중치 위에서 FIC 를 새로 돌린다.
        fic_init = (ParamSet.default(n, c_ei_init=cfg.wc_c_ei_init).sanitize(mask, cap)
                    if a.frozen_part1 else ps)
        print(f"[FIC] S_e -> {cfg.fic_target_se} "
              f"({'표준 Part1 캐시 재사용 + freeze' if a.frozen_part1 else '신규'})...",
              flush=True)
        t0 = time.time()
        bundle_fic = run_fic(
            network=network, cfg=cfg, data=data,
            bundle_in=StateBundle.from_warmup(
                warmup_result=warmup_result, bold_monitor_template=bold_monitor,
                initial_params=fic_init,
                internal_state=capture_internal_state(initial_state),
                delay_history=capture_network_delay_history(network), stage="warmup"))
        c_ei = np.asarray(bundle_fic.params.c_ei, np.float32)
        print(f"[FIC] done {time.time()-t0:.0f}s  c_ei mean={c_ei.mean():.3f} "
              f"max={c_ei.max():.3f}  frozen={bundle_fic.params.c_ei_frozen}", flush=True)

        # Part1 이 넘긴 full state 위에 가중치만 갈아끼운다 (파이프라인 handoff 방식).
        # sanitize() 를 안 부르고 prep() 로 직접 만든다 — advance()/apply_to_state() 는
        # 재sanitize 하지 않으므로(pipeline_contracts.py:276,307) 값이 그대로 모델에 간다.
        ps = ParamSet(c_ei=c_ei,
                      wLRE=prep(a.scale * SRC_L, mask, cap),
                      wFFI=prep(a.scale * SRC_F, mask, cap),
                      c_ei_frozen=bool(a.frozen_part1))
        wl, wf = np.asarray(ps.wLRE, float), np.asarray(ps.wFFI, float)
        print(f"  투입 가중치 SC>0: wLRE [{wl[scm].min():+.4f},{wl[scm].max():+.4f}] "
              f"mean={wl[scm].mean():+.4f} 음수={np.mean(wl[scm] < 0)*100:.1f}%  |  "
              f"wFFI [{wf[scm].min():+.4f},{wf[scm].max():+.4f}] "
              f"mean={wf[scm].mean():+.4f} 음수={np.mean(wf[scm] < 0)*100:.1f}%", flush=True)
        dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
        FC_sim = np.asarray(compute_simulated_fc(
            network, bundle_fic.advance(new_params=ps), cfg, sim_duration_ms=dur,
            skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
        corr, rmse = pair(FC_sim, FC_emp)
        r = dict(idx=idx, sub=sub, corr=corr, rmse=rmse,
                 c_ei_mean=float(c_ei.mean()),
                 wLRE_mean=float(wl[scm].mean()), wFFI_mean=float(wf[scm].mean()),
                 wFFI_zero_pct=float(np.mean(wf[scm] <= 1e-6) * 100),
                 wLRE_zero_pct=float(np.mean(wl[scm] <= 1e-6) * 100),
                 sim_mean=float(FC_sim[IU].mean()), emp_mean=float(FC_emp[IU].mean()))
        rows.append(r)
        print(f"  FC_sim vs FC_emp   corr={corr:+.4f}  rmse={rmse:.4f}", flush=True)
        np.savez_compressed(f"{OUT}/idx{idx}{sfx}.npz", FC_sim=FC_sim, FC_emp=FC_emp,
                            wLRE=wl, wFFI=wf, c_ei=c_ei)

        # ---- 4 패널 ---------------------------------------------------------
        fig, ax = plt.subplots(1, 4, figsize=(20.8, 5.3), facecolor=SURFACE)
        vw = max(np.nanpercentile(np.abs(w if a.no_mask else w[scm]), 99) for w in (wl, wf))
        hm = None if a.no_mask else scm
        wc = 1.0 if a.weights == "corr1" else 0.0
        _heat(ax[0], wl, f"wLRE ({a.weights})", wc, hm,
              vmin=0.0, vmax=float(np.nanmax([wl[scm].max(), wf[scm].max()])))
        _heat(ax[1], wf, f"wFFI ({a.weights})", wc, hm,
              vmin=0.0, vmax=float(np.nanmax([wl[scm].max(), wf[scm].max()])))
        vf = np.nanpercentile(np.abs(FC_emp), 99)
        _heat(ax[2], FC_emp, "FC_emp  (경험 FC)", 0.0, vmin=-1.0, vmax=1.0)
        _heat(ax[3], FC_sim, f"FC_sim   r={corr:+.4f}  rmse={rmse:.4f}", 0.0, vmin=-1.0, vmax=1.0)
        fig.suptitle(f"idx {idx} (sub {sub})  —  wLRE=FC_emp / wFFI=−FC_emp "
                     f"(절편 0{', clip 우회' if a.no_clip else ''})   c_ei = {'표준 Part1 FIC + freeze' if a.frozen_part1 else '이 가중치 위 FIC'}",
                     color=INK, fontsize=13, y=0.99)
        fig.text(0.5, 0.015,
                 f"가중치는 0 기준(초기값 1.0 은 스케일 밖 — 전 엣지가 초기값보다 약하다). "
                 + (f"clip 우회: 음수 그대로 투입 (wFFI 음수 {np.mean(wf[scm] < 0)*100:.0f}%, "
                    f"wLRE 음수 {np.mean(wl[scm] < 0)*100:.0f}%). "
                    if a.no_clip else
                    f"sanitize 클리핑으로 wFFI 는 {r['wFFI_zero_pct']:.0f}%, "
                    f"wLRE 는 {r['wLRE_zero_pct']:.0f}% 가 0. ") + 
                 f"회색=SC 엣지 없음. FC_emp·FC_sim 은 0 기준·공통 스케일. "
                 f"검은 선 = cortex | subcortex 경계.",
                 ha="center", color=INK2, fontsize=8)
        fig.tight_layout(rect=[0, 0.045, 1, 0.96])
        fp = f"{FIGS}/idx{idx}_wfc{sfx}_matrices.png"
        fig.savefig(fp, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(f"  -> {fp}", flush=True)

        with open(f"{OUT}/summary{sfx}_idx{idx}.json", "w") as fh:
            json.dump(rows, fh, indent=1)
        import jax
        jax.clear_caches()

    print(f"\n{'='*72}")
    print(f"{'idx':>4} {'sub':>9} {'corr':>9} {'rmse':>8} {'c_ei':>7} "
          f"{'wLRE0%':>7} {'wFFI0%':>7} {'sim mean':>9} {'emp mean':>9}")
    for r in rows:
        print(f"{r['idx']:>4} {r['sub']:>9} {r['corr']:>+9.4f} {r['rmse']:>8.4f} "
              f"{r['c_ei_mean']:>7.3f} {r['wLRE_zero_pct']:>7.1f} "
              f"{r['wFFI_zero_pct']:>7.1f} {r['sim_mean']:>9.4f} {r['emp_mean']:>9.4f}")


if __name__ == "__main__":
    main()
