#!/usr/bin/env python3
"""dbs_from_cache.py — Part3(grad) 캐시를 직접 로드해 Part4 DBS 실행 (진폭×주파수 sweep).

FIC/EIB/Part3 를 재계산하지 않는다. grad pkl 에 최적화 params(c_ei/wLRE/wFFI)가 다 있으므로
DBS 는 그것만 있으면 된다. 네트워크(+warmup)는 1회만 짓고 (amp,freq) 조합마다 자극만 다시.

사용:
    # 단일
    python3 dbs_from_cache.py --subject-idx 0 --dbs-amplitudes 1.0 --dbs-freqs 143
    # sweep (진폭 6 × 주파수 5 = 30 조합)
    python3 dbs_from_cache.py --subject-idx 0 \
        --dbs-amplitudes 0.5,1.0,1.5,2.0,2.5,3.0 \
        --dbs-freqs 20,125,143,167,200 \
        --dbs-outroot output_ppmi_pd/dbs_sweep/idx0

출력: <outroot>/amp<a>_f<freq>/{STN_L,STN_R,GP_L,GP_R}/fc_pre_during_diff.png (+csv)
"""
import argparse
import gc
import glob
import os
import pickle
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("PART3_REMAT_SCAN", "1")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pathlib as _pl

import main_ppmi_pd as M
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: None    # part4 는 savefig 로 직접 저장 → show no-op

from data_loader import load_data
from model import build_network
from pipeline_contracts import StateBundle
from part4_dbs import run_dbs_stimulation


def _parse_list(s):
    return [float(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser(description="Part3 grad 캐시에서 DBS sweep(파이프라인 재계산 없음)")
    ap.add_argument("--subject-idx", dest="subject_idx", type=int, default=0)
    ap.add_argument("--noise-level", dest="noise_level", type=float, default=0.02)
    ap.add_argument("--dbs-amplitudes", dest="dbs_amplitudes", type=str, default="1.0",
                    help="진폭 리스트(콤마). 예: 0.5,1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--dbs-freqs", dest="dbs_freqs", type=str, default="130",
                    help="주파수 리스트 Hz(콤마). dt=1ms 양자화: 20,125,143,167,200")
    ap.add_argument("--dbs-outroot", dest="dbs_outroot", type=str, default=None,
                    help="sweep 출력 루트(미지정 시 <sub>/dbs_sweep). 조합별 amp<a>_f<f>/ 하위폴더")
    ap.add_argument("--cei-scale", dest="cei_scale", type=float, default=1.0,
                    help="grad c_ei 에 곱할 배수(인과 테스트용). 예: 3.0 → 억제↑/흥분성↓")
    ap.add_argument("--neural-stride", dest="neural_stride", type=int, default=4,
                    help="타깃 노드 neural CSV 다운샘플 stride (dt1ms 기준 4=250Hz). 0=저장 안 함")
    ap.add_argument("--seed", type=int, default=None,
                    help="노이즈 시드(cfg.bundle_rng_seed). 진폭 경향이 단일 실현의 산물이 "
                         "아닌지 보려면 시드 여러 개로 돌려 비교할 것")
    a = ap.parse_args()

    amps = _parse_list(a.dbs_amplitudes)
    freqs = _parse_list(a.dbs_freqs)

    p = M.prepare_pd_data(a.subject_idx, a.noise_level)
    cfg = M.make_config(p, a.subject_idx)
    cfg.dbs_save_neural_csv = a.neural_stride > 0
    cfg.dbs_neural_csv_stride = max(1, a.neural_stride)
    if a.seed is not None:
        cfg.bundle_rng_seed = a.seed
        print(f"[dbs_from_cache] bundle_rng_seed = {a.seed}")
    outroot = a.dbs_outroot or os.path.join(p["out_dir"], "dbs_sweep")

    M._FIG["dir"] = _pl.Path(p["out_dir"]) / "figures"
    M._FIG["dir"].mkdir(parents=True, exist_ok=True)

    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)

    # 네트워크(+warmup) 1회
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

    # grad 캐시 로드 = 최적화 상태
    gfiles = sorted(glob.glob(os.path.join(data["cache_dir"], "grad_*.pkl")), key=os.path.getmtime)
    if not gfiles:
        sys.exit(f"[dbs_from_cache] grad 캐시 없음: {data['cache_dir']}  → Part3 먼저 완료 필요")
    with open(gfiles[-1], "rb") as fh:
        grad = pickle.load(fh)
    bundle_grad = StateBundle.from_dict(grad["bundle"])
    corr = grad["bundle"].get("metadata", {}).get("post_grad_fc_corr")
    print(f"[dbs_from_cache] grad 로드: {os.path.basename(gfiles[-1])}  post_grad_corr={corr}")

    # 인과 테스트: c_ei 배수 조정(억제 강도 변경 → DBS 반응이 c_ei에 좌우되는지 검증)
    if a.cei_scale != 1.0:
        from pipeline_contracts import ParamSet
        op = bundle_grad.params
        newp = ParamSet(c_ei=np.asarray(op.c_ei, dtype=np.float32) * a.cei_scale,
                        wLRE=op.wLRE, wFFI=op.wFFI, c_ei_frozen=op.c_ei_frozen)
        bundle_grad = bundle_grad.advance(new_params=newp)
        print(f"[cei-scale] c_ei ×{a.cei_scale} → mean {float(newp.c_ei.mean()):.3f}")
    print(f"[dbs_from_cache] sweep: 진폭 {amps} × 주파수 {freqs} = {len(amps)*len(freqs)} 조합 → {outroot}")

    import jax
    tot = len(amps) * len(freqs)
    n = 0
    for amp in amps:
        for freq in freqs:
            n += 1
            amp_s = f"{amp:g}"; freq_s = f"{freq:g}"
            combo_dir = os.path.join(outroot, f"amp{amp_s}_f{freq_s}")
            # resume: 4타깃 결과가 다 있어야 skip. neural CSV 를 켠 경우 그것도 있어야 한다
            # — FC png 만 보면 구 스윕 결과 때문에 CSV 를 한 개도 안 만들고 전부 건너뛴다.
            def _n(pat):
                return len(glob.glob(os.path.join(combo_dir, "*", "*", pat)))
            if _n("fc_pre_during_diff.png") >= 4 and (
                    not cfg.dbs_save_neural_csv or _n("neural_timeseries_target.csv") >= 4):
                print(f"[{n}/{tot}] amp{amp_s}_f{freq_s} 이미 완료 → skip")
                continue
            cfg.dbs_pulse_amplitude = amp
            cfg.dbs_stimulation_frequency_hz = freq
            cfg.dbs_output_base_dir = combo_dir
            print(f"\n{'='*60}\n[{n}/{tot}] amp=±{amp_s}  freq={freq_s}Hz  → {combo_dir}\n{'='*60}")
            run_dbs_stimulation(network=network, bundle_in=bundle_grad, cfg=cfg, data=data)
            # 조합 간 JIT 캐시(+stim array 상수) 해제 → GPU 메모리 누적/OOM 방지
            jax.clear_caches()
            gc.collect()

    print(f"\n[dbs_from_cache] sweep 완료 ({n} 조합) → {outroot}")


if __name__ == "__main__":
    main()
