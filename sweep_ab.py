#!/usr/bin/env python3
"""sweep_ab.py — 가중치를 스칼라 2개로 줄이고 (A,B) 격자를 훑는다.

    wLRE = c + A * FC_emp
    wFFI = c - B * FC_emp        (c = 절편, 기본 1.0 = 파이프라인 초기값)

엣지별 자유 파라미터 2*(SC edge) 를 **스칼라 2개**로 대체한 뒤 각 (A,B) 에서
정방향 시뮬 → FC_sim 을 FC_emp 와 corr / RMSE 비교.

c_ei 는 격자마다 FIC 를 돌릴 수 없으므로(1회 ~15분) **참조점 (A0,B0) 에서 한 번만
FIC 를 돌려 고정**한다. 따라서 응답면은 c_ei 가 A0,B0 에 맞춰진 조건부 단면이다.
격자 최적점이 A0,B0 에서 멀면 그 점에서 FIC 를 다시 돌려야 한다(--refit-best).

실행:
  python3 sweep_ab.py --idxs 4 --grid coarse
  python3 sweep_ab.py --idxs 4,5,6 --grid full --refit-best
"""
import argparse
import json
import os
import time

import numpy as np

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from formula_reproduce import IU, SUBDIR, load_ref, pair

OUT = "output_ppmi_pd/_sweep"

GRIDS = {
    # 비용 측정용
    "probe":  (np.array([0.0, 2.6]), np.array([2.1])),
    "mid":    (np.array([0.0, 1.0, 2.0, 2.6, 3.2, 4.0]),
               np.array([0.0, 1.0, 2.1, 3.0])),
    "coarse": (np.array([0.0, 1.3, 2.6, 3.9]), np.array([0.0, 1.0, 2.1, 3.1])),
    "full":   (np.array([0.0, 0.8, 1.6, 2.4, 3.2, 4.0]),
               np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])),
}


def loo_slopes(target_sub):
    """절편을 c 로 고정했을 때의 원점통과 기울기, 본인 제외 평균.

    (w - c) = A * FC_emp 를 SC>0 상삼각에서 최소제곱.
    """
    accL, accF, used = [], [], []
    for s in SUBDIR.values():
        if s == target_sub:
            continue
        ref, _, _ = load_ref(s)
        if ref is None:
            continue
        used.append(s)
        W = np.loadtxt(f"output_ppmi_pd/{s}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"output_ppmi_pd/{s}/inputs/FC.csv", delimiter=",")
        m = (W > 0)[IU]
        x = FCe[IU][m]
        denom = float(x @ x)
        accL.append(float(x @ (ref["wLRE"][IU][m] - 1.0)) / denom)
        accF.append(-float(x @ (ref["wFFI"][IU][m] - 1.0)) / denom)
    return float(np.mean(accL)), float(np.mean(accF)), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", type=str, default="4")
    ap.add_argument("--grid", type=str, default="coarse", choices=sorted(GRIDS))
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--intercept", type=float, default=1.0,
                    help="c. 0.0 이면 엄밀한 Y=AX (FC=0 엣지의 커플링이 사라진다)")
    ap.add_argument("--refit-points", type=str, default="",
                    help='"A,B;A,B" — 이 점들은 각자 FIC 를 다시 돌린다. 격자의 '
                         'c_ei 고정이 응답면 모양을 만든 것인지 검증용 (점당 ~21분).')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    As, Bs = GRIDS[a.grid]

    for idx in [int(x) for x in a.idxs.split(",") if x.strip()]:
        sub = SUBDIR[idx]
        A0, B0, used = loo_slopes(sub)
        print(f"\n{'='*78}\nidx {idx} ({sub})  참조 기울기 LOO from {used}: "
              f"A0={A0:.4f}  B0={B0:.4f}   절편 c={a.intercept}", flush=True)

        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)
        cfg.cache_version = f"{cfg.cache_version}_sweepab_c{a.intercept:g}"
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
        network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
        mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
        dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

        def mk(A, B, c_ei):
            return ParamSet(
                c_ei=np.asarray(c_ei, np.float32),
                wLRE=np.asarray(a.intercept + A * FC_emp, np.float32),
                wFFI=np.asarray(a.intercept - B * FC_emp, np.float32),
                c_ei_frozen=False).sanitize(mask, cap)

        # c_ei: 참조점 (A0,B0) 에서 FIC 1회 → 격자 전체에 고정
        print(f"[FIC] 참조 가중치(A0,B0) 위에서 S_e -> {cfg.fic_target_se} ...", flush=True)
        t0 = time.time()
        bundle_fic = run_fic(
            network=network, cfg=cfg, data=data,
            bundle_in=StateBundle.from_warmup(
                warmup_result=warmup_result, bold_monitor_template=bold_monitor,
                initial_params=mk(A0, B0, np.ones(n, np.float32)),
                internal_state=capture_internal_state(initial_state),
                delay_history=capture_network_delay_history(network), stage="warmup"))
        c_ei = np.asarray(bundle_fic.params.c_ei, np.float32)
        print(f"[FIC] done {time.time()-t0:.0f}s  c_ei mean={c_ei.mean():.3f}", flush=True)

        rows, path = [], f"{OUT}/idx{idx}_c{a.intercept:g}_{a.grid}.json"
        pts = [(float(A), float(B)) for A in As for B in Bs]
        pts.append((A0, B0))                      # 참조점
        print(f"  {'A':>6} {'B':>6} {'corr':>8} {'rmse':>8} {'clip0%':>7} {'s':>6}")
        for i, (A, B) in enumerate(pts):
            t = time.time()
            ps = mk(A, B, c_ei)
            fc = np.asarray(compute_simulated_fc(
                network, bundle_fic.advance(new_params=ps), cfg,
                sim_duration_ms=dur, skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
            corr, rmse = pair(fc, FC_emp)
            m = mask.astype(bool)[IU] if mask.ndim == 2 else None
            wf = np.asarray(ps.wFFI)[IU][m]
            clip = float((wf <= 1e-6).mean() * 100)
            dt = time.time() - t
            rows.append(dict(A=A, B=B, corr=corr, rmse=rmse, wffi_clip_pct=clip,
                             ref=(i == len(pts) - 1)))
            print(f"  {A:>6.3f} {B:>6.3f} {corr:>+8.4f} {rmse:>8.4f} {clip:>7.1f} {dt:>6.1f}",
                  flush=True)
            with open(path, "w") as fh:      # 매 점마다 저장 — 중단돼도 남는다
                json.dump(dict(idx=idx, sub=sub, intercept=a.intercept,
                               A0=A0, B0=B0, rows=rows), fh, indent=1)

        # 격자는 c_ei 를 참조점에 고정한 조건부 단면이다. 지정 점에서 FIC 를 다시
        # 돌려 봉우리가 진짜인지, c_ei 불일치가 만든 것인지 가른다.
        for spec in [s for s in a.refit_points.split(";") if s.strip()]:
            A, B = (float(v) for v in spec.split(","))
            print(f"[FIC-refit] A={A} B={B} ...", flush=True)
            t = time.time()
            bf = run_fic(
                network=network, cfg=cfg, data=data,
                bundle_in=StateBundle.from_warmup(
                    warmup_result=warmup_result, bold_monitor_template=bold_monitor,
                    initial_params=mk(A, B, np.ones(n, np.float32)),
                    internal_state=capture_internal_state(initial_state),
                    delay_history=capture_network_delay_history(network),
                    stage="warmup"))
            fc = np.asarray(compute_simulated_fc(
                network, bf, cfg, sim_duration_ms=dur,
                skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
            corr, rmse = pair(fc, FC_emp)
            fixed = next((r["corr"] for r in rows
                          if abs(r["A"] - A) < 1e-6 and abs(r["B"] - B) < 1e-6), np.nan)
            rows.append(dict(A=A, B=B, corr=corr, rmse=rmse, refit=True,
                             corr_fixed_cei=float(fixed),
                             c_ei_mean=float(np.asarray(bf.params.c_ei).mean())))
            print(f"[FIC-refit] A={A} B={B}  corr={corr:+.4f} (c_ei 고정판 {fixed:+.4f})"
                  f"  rmse={rmse:.4f}  {time.time()-t:.0f}s", flush=True)
            with open(path, "w") as fh:
                json.dump(dict(idx=idx, sub=sub, intercept=a.intercept,
                               A0=A0, B0=B0, rows=rows), fh, indent=1)

        best = max(rows, key=lambda r: r["corr"])
        print(f"  -> best corr {best['corr']:+.4f} @ A={best['A']:.3f} B={best['B']:.3f}"
              f"   (rmse {best['rmse']:.4f})\n  -> {path}", flush=True)

        import jax
        jax.clear_caches()


if __name__ == "__main__":
    main()
