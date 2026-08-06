#!/usr/bin/env python3
"""sweep_wmax.py — wLRE_max 를 훑는다. 나머지는 전부 종속변수.

    wLRE = 0 + u * M           u = (FC_emp - FC_min) / (FC_max - FC_min),  M = wLRE_max
    wFFI = max(0, 2 - wLRE)    (실측 R² 0.986~0.999, 재현율 100%)
    c_ei = 그 가중치 위에서 FIC (S_e -> 0.25)

자유 파라미터가 **M 하나**뿐인 모델이다. M 을 격자로 훑어 corr/rmse 를 재고 최적 M 을 찾는다.

비용: 점당 FIC 가 지배적(2000 step ≈ 19분). 청크 안에서는 직전 M 의 c_ei 로 warm start
하고 step 을 줄여(--fic-steps-warm) 크게 단축한다. 청크 첫 점만 full FIC.

병렬: 같은 subject 를 여러 프로세스가 동시에 열면 inputs/*.csv 를 동시에 써서
"Region label count mismatch" 로 죽는다 → --stagger 초 만큼 시차를 두고 띄운다.
GPU OOM 도 실측됐다(DBS 3-job). --jobs 는 보수적으로.

사용:
  python3 sweep_wmax.py --idxs 4 --jobs 6                 # 4.00~5.00, 0.01 간격 101점
  python3 sweep_wmax.py --idxs 4 --mmin 3.5 --mmax 5.5 --step 0.05
출력: output_ppmi_pd/_wmax/idx<N>_chunk<k>.json  (+ 합쳐서 idx<N>.csv)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "output_ppmi_pd/_wmax"


def build_weights(FC_emp, sc_mask, M):
    """wLRE = [0, M] 선형사상, wFFI = max(0, 2 - wLRE). 둘 다 SC 마스킹 + 대칭화."""
    iu = np.triu_indices(FC_emp.shape[0], 1)
    m = sc_mask.astype(bool)[iu]
    x = FC_emp[iu][m]
    u = (FC_emp - float(x.min())) / (float(x.max()) - float(x.min()))
    wl = np.clip(u * M, 0.0, None) * sc_mask
    wf = np.maximum(0.0, 2.0 - wl) * sc_mask
    sym = lambda v: (0.5 * (v + v.T)).astype(np.float32)
    return sym(wl), sym(wf)


def worker(a):
    """--worker 로 넘어온 M 리스트를 순차 처리. 청크 내 warm start."""
    import main_ppmi_pd as M_
    from data_loader import load_data
    from model import build_network
    from part1_fic import run_fic
    from part3_gradient import compute_simulated_fc
    from pipeline_contracts import (ParamSet, StateBundle, capture_internal_state,
                                    capture_network_delay_history)
    from formula_reproduce import IU, pair

    Ms = [float(t) for t in a.worker.split(",") if t.strip()]
    idx = a.idxs
    p = M_.prepare_pd_data(idx, a.noise_level)
    cfg = M_.make_config(p, idx, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_wmax"
    # 스윕은 best-후보 선택이 필요 없다 — 후보 재시뮬(각 600s)이 FIC 시간의 ~25%.
    cfg.fic_posthoc_top_k = a.fic_topk
    cfg.fic_early_stop_tolerance_se = a.es_tol
    cfg.fic_early_stop_patience = a.es_patience
    cfg.fic_early_stop_window = a.es_window
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    mask, n = data["sc_mask"], data["n_nodes"]
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

    rows, c_prev = [], None
    path = f"{OUT}/idx{idx}_chunk{a.chunk}.json"
    os.makedirs(OUT, exist_ok=True)
    for k, M in enumerate(Ms):
        t0 = time.time()
        wl, wf = build_weights(FC_emp, mask, M)
        # warm start: 직전 M 의 c_ei 에서 출발 → FIC step 을 크게 줄일 수 있다
        c0 = np.full(n, cfg.wc_c_ei_init, np.float32) if c_prev is None else c_prev
        cfg.fic_max_iterations = a.fic_steps if c_prev is None else a.fic_steps_warm
        ps0 = ParamSet(c_ei=c0, wLRE=wl, wFFI=wf, c_ei_frozen=False)
        bundle = run_fic(
            network=network, cfg=cfg, data=data,
            bundle_in=StateBundle.from_warmup(
                warmup_result=warmup_result, bold_monitor_template=bold_monitor,
                initial_params=ps0,
                internal_state=capture_internal_state(initial_state),
                delay_history=capture_network_delay_history(network), stage="warmup"))
        c_ei = np.asarray(bundle.params.c_ei, np.float32)
        c_prev = c_ei
        fc = np.asarray(compute_simulated_fc(
            network, bundle, cfg, sim_duration_ms=dur,
            skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
        corr, rmse = pair(fc, FC_emp)
        scm = mask.astype(bool)
        rows.append(dict(M=M, corr=corr, rmse=rmse, c_ei_mean=float(c_ei.mean()),
                         c_ei_max=float(c_ei.max()),
                         wLRE_mean=float(wl[scm].mean()), wFFI_mean=float(wf[scm].mean()),
                         wFFI_zero_pct=float((wf[scm] <= 1e-6).mean() * 100),
                         sec=round(time.time() - t0, 1)))
        print(f"[{k+1}/{len(Ms)}] M={M:.2f}  corr={corr:+.4f}  rmse={rmse:.4f}  "
              f"c_ei={c_ei.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(path, "w") as fh:      # 매 점 저장 — 중단돼도 남는다
            json.dump(rows, fh, indent=1)
    print(f"[worker] 완료 {len(rows)}점 → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", type=int, default=4, help="subject idx (worker 는 1개만)")
    ap.add_argument("--mmin", type=float, default=4.0)
    ap.add_argument("--mmax", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=6, help="동시 워커 수. GPU OOM 주의")
    ap.add_argument("--stagger", type=float, default=45.0,
                    help="워커 기동 간격(초). inputs/*.csv 동시 쓰기 경합 방지")
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--fic-steps", type=int, default=2000, help="청크 첫 점 FIC step")
    ap.add_argument("--fic-steps-warm", type=int, default=400,
                    help="warm start 이후 FIC step. 직전 c_ei 에서 출발하므로 짧아도 된다")
    ap.add_argument("--fic-topk", type=int, default=1,
                    help="FIC best 후보 재시뮬 개수. 기본 10 → 1 로 줄이면 25%% 단축")
    ap.add_argument("--es-tol", type=float, default=0.010,
                    help="early stop: se_error 이동평균 임계. 실측 수렴 수준은 커플링 세기에 "
                         "따라 0.004~0.018 → 0.010 이 절충")
    ap.add_argument("--es-patience", type=int, default=100,
                    help="이동평균이 임계 미만으로 유지돼야 하는 step 수")
    ap.add_argument("--es-window", type=int, default=50, help="이동평균 창(step)")
    ap.add_argument("--worker", type=str, default="", help="(내부) 이 M 리스트를 처리")
    ap.add_argument("--chunk", type=int, default=0, help="(내부) 청크 번호")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.worker:
        worker(a)
        return

    Ms = np.round(np.arange(a.mmin, a.mmax + a.step / 2, a.step), 4)
    chunks = [Ms[i::a.jobs] for i in range(a.jobs)]     # 라운드로빈 → 청크별 부하 균등
    chunks = [c for c in chunks if len(c)]
    os.makedirs(OUT, exist_ok=True)
    print(f"# idx{a.idxs}  M {a.mmin}~{a.mmax} step {a.step} = {len(Ms)}점 "
          f"→ {len(chunks)} 청크 (청크당 {len(chunks[0])}점)")
    print(f"# FIC: 첫 점 {a.fic_steps} step, 이후 warm start {a.fic_steps_warm} step")
    est = (a.fic_steps + (len(chunks[0]) - 1) * a.fic_steps_warm) * 0.57 / 60
    print(f"# 예상: 청크당 ~{est:.0f}분 (FIC 0.57초/step 실측 기준)\n")
    if a.dry_run:
        for k, c in enumerate(chunks):
            print(f"  chunk{k}: {len(c)}점  {c[0]:.2f}~{c[-1]:.2f}")
        return

    procs = []
    for k, c in enumerate(chunks):
        cmd = [sys.executable, "-u", os.path.join(HERE, "sweep_wmax.py"),
               "--idxs", str(a.idxs), "--noise-level", str(a.noise_level),
               "--fic-steps", str(a.fic_steps), "--fic-steps-warm", str(a.fic_steps_warm),
               "--fic-topk", str(a.fic_topk), "--es-tol", str(a.es_tol),
               "--es-patience", str(a.es_patience), "--es-window", str(a.es_window),
               "--chunk", str(k), "--worker", ",".join(f"{v:.4f}" for v in c)]
        log = os.path.join(OUT, f"idx{a.idxs}_chunk{k}.log")
        fh = open(log, "w")
        procs.append((k, subprocess.Popen(cmd, cwd=HERE, stdout=fh,
                                          stderr=subprocess.STDOUT), fh))
        print(f"# chunk{k} 기동 ({len(c)}점, {c[0]:.2f}~{c[-1]:.2f})  log={log}", flush=True)
        if k < len(chunks) - 1:
            time.sleep(a.stagger)

    rc = {}
    for k, p, fh in procs:
        rc[k] = p.wait()
        fh.close()
        print(f"# chunk{k} 종료 rc={rc[k]}", flush=True)

    rows = []
    for k in rc:
        f = f"{OUT}/idx{a.idxs}_chunk{k}.json"
        if os.path.exists(f):
            rows += json.load(open(f))
    rows.sort(key=lambda r: r["M"])
    if rows:
        import csv
        with open(f"{OUT}/idx{a.idxs}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        best = max(rows, key=lambda r: r["corr"])
        print(f"\n# {len(rows)}점 수집 → {OUT}/idx{a.idxs}.csv")
        print(f"# best corr {best['corr']:+.4f} @ M={best['M']:.2f} (rmse {best['rmse']:.4f})")
    fails = [k for k, v in rc.items() if v != 0]
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
