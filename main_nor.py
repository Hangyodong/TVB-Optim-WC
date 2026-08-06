#!/usr/bin/env python3
"""
main.py — EI Tuning 파이프라인 스크립트 실행

Usage:
    python3 main.py                         # mouse (default)
    python3 main.py --dataset human         # human
    python3 main.py --dataset mouse --skip-dbs
    python3 main.py --fic-only              # FIC만 실행
    python3 main.py --resume                # 캐시 이어서

노트북(main.ipynb)과 동일한 실행 흐름.
"""
import argparse
import os
import sys

# ── 0. JAX 환경변수 (노트북 Cell 1과 동일) ───────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR",   "platform")

import jax
jax.config.update("jax_enable_x64", False)
print(f"backend : {jax.default_backend()}")
print(f"devices : {jax.devices()}")
print(f"jax_enable_x64 : {jax.config.jax_enable_x64}")

# ── 1. 임포트 ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # 화면 없이 저장만

import matplotlib.pyplot as _plt
import datetime as _dt, pathlib as _pl, re as _re

# 실행 시각 기반 출력 폴더 생성
_FIG_DIR = _pl.Path(f"figures_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
_FIG_DIR.mkdir(exist_ok=True)
_fig_counter = {"n": 0}

def _slugify(text):
    return _re.sub(r"[^\w가-힣\-]", "_", text.strip())[:60] or "figure"

def _get_label(fig, idx):
    try:
        t = fig._suptitle.get_text()
        if t: return _slugify(t)
    except Exception: pass
    for ax in fig.axes:
        try:
            t = ax.get_title()
            if t: return _slugify(t)
        except Exception: pass
    return f"figure_{idx:03d}"

if not hasattr(_plt, "_main_original_show"):
    _plt._main_original_show = _plt.show

def _patched_show(*args, **kwargs):
    for fn in _plt.get_fignums():
        fig = _plt.figure(fn)
        _fig_counter["n"] += 1
        n = _fig_counter["n"]
        label = _get_label(fig, n)
        out = _FIG_DIR / f"{n:03d}_{label}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  [fig] {out}")
    _plt._main_original_show(*args, **kwargs)

_plt.show = _patched_show
print(f"[FIG] 출력 폴더: {_FIG_DIR.resolve()}")


from config              import Config
from data_loader         import load_data
from model               import build_network
from part1_fic           import run_fic
from part2_eib           import run_eib
from part3_gradient      import run_gradient_optimization
from part4_dbs           import run_dbs_stimulation
from pipeline_contracts  import (
    ParamSet,
    StateBundle,
    capture_internal_state,
    capture_network_delay_history,
)

# ── 2. 인자 파싱 ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="EI Tuning Pipeline (nor: normal mouse CHA-22)")
    parser.add_argument(
        "--dataset", choices=["nor"], default="nor",
        help="Dataset to use (fixed: nor)"
    )
    parser.add_argument("--skip-fic",      action="store_true", help="FIC 건너뜀")
    parser.add_argument("--skip-eib",      action="store_true", help="EIB 건너뜀")
    parser.add_argument("--skip-gradient", action="store_true", help="Gradient 건너뜀")
    parser.add_argument("--skip-dbs",      action="store_true", help="DBS 건너뜀")
    parser.add_argument("--fic-only",      action="store_true", help="FIC만 실행")
    parser.add_argument("--resume",        action="store_true", help="캐시 이어서 실행")
    parser.add_argument(
        "--freeze-c-ei", dest="freeze_c_ei",
        default=None, action="store_true",
        help="FIC 후 c_ei 동결(고정). 미지정 시 Config 기본값 사용",
    )
    parser.add_argument(
        "--no-freeze-c-ei", dest="freeze_c_ei",
        action="store_false",
        help="FIC 후 c_ei 학습 허용(동결 해제)",
    )
    return parser.parse_args()

# ── 3. Dataset별 파라미터 (노트북 Cell 4와 동일) ─────────────────────────
_DATASET_PARAMS = {
    "nor": dict(
        # RWW 모델 사용 — WC 파라미터는 Config 요구사항으로 유지(모델 미참조). mouse 값 차용.
        wc_c_ei_init=10.0,
        fic_target_firing_rate_hz=2.0,   # 진단용 Hz 표시 (FIC 제어는 fic_target_se=S_e gating)
        # DBS: CHA-22 익명 라벨 → STN/GPe/GPi 매핑 불가 → 타깃 없음(DBS는 main에서 skip)
        dbs_target_regions={},
        # Bold HRF: mouse-specific
        bold_hrf_k1=5.6,
        bold_hrf_V0=0.02,
        bold_hrf_tau_s=0.8,
        bold_hrf_tau_f=0.4,
        bold_hrf_scaling=1.0 / 3.0,
        bold_hrf_duration_ms=32_000.0,
        sc_csv="nor/sub-419087_mouse_CHA_weight.csv",
        length_csv="nor/sub-419087_mouse_CHA_tract_length.csv",
        fc_csv="nor/sub-419087_ses-1_task-rest_bold_RAS_resampled_cleaned_FC_matrix.csv",
        region_txt="nor/mouse_CHA.txt",
        tract_conduction_speed=3.0,
        additive_noise_sigma=0.02,
    ),
}

# ── 4. Config 생성 (노트북 Cell 5와 동일) ────────────────────────────────
def make_config(dataset: str) -> Config:
    p = _DATASET_PARAMS[dataset]
    return Config(
        # ── 데이터 경로 ──────────────────────────────────────────
        region_txt                          = p["region_txt"],
        sc_csv                              = p["sc_csv"],
        length_csv                          = p["length_csv"],
        fc_csv                              = p["fc_csv"],

        # ── 캐시 (노트북 Cell 5와 동일: 기존 캐시 재사용) ────────
        cache_version                       = f"v_cha22_p33segating_{dataset}",
        cache_run_label                     = f"{dataset}_cha22",

        # ── 시뮬레이션 공통 ──────────────────────────────────────
        integration_dt_ms                   = 1.0,
        warmup_duration_ms                  = 720_000,
        bold_repetition_time_ms             = 1000.0,
        tract_conduction_speed              = p["tract_conduction_speed"],
        additive_noise_sigma                = p["additive_noise_sigma"],

        # ── Part 1 — FIC (구버전 notebook 로직) ─────────────────
        fic_target_firing_rate_hz           = p["fic_target_firing_rate_hz"],
        fic_learning_rate                   = 1e-3,
        fic_max_iterations                  = 2000,
        fic_early_stop_patience             = 500,
        fic_early_stop_tolerance_hz         = 0.10,
        fic_step_duration_ms                = 1_000,
        fic_step_skip_tr                    = 0,
        freeze_c_ei_after_fic               = False,   # True: Part2(EIB)에서만 c_ei 동결. Part3는 항상 c_ei 최적화

        # ── Part 2 — EIB (구버전 notebook 로직) ─────────────────
        eib_max_iterations                  = 8000,
        eib_internal_fic_learning_rate      = 0.05,
        eib_max_weight_learning_rate        = 0.002,
        eib_bold_window_samples             = 720,
        eib_snapshot_save_interval          = 50,
        connectivity_weight_max             = 1.5,
        eib_posthoc_duration_ms             = 720_000,
        eib_posthoc_skip_tr                 = 60,

        # ── Part 3 — Full Gradient (구버전 notebook 로직) ───────
        optimizer_learning_rate             = 0.0005,
        optimizer_max_steps                 = 1000,
        optimizer_chunk_steps               = 10,
        optimizer_bold_window_tr            = 720,
        optimizer_bold_skip_tr              = 60,

        # ── Phase 1 final baseline settle ────────────────────────
        baseline_settle_duration_ms         = 0,

        # ── EIB score 계산용 ──────────────────────────────────────
        pd_fit_region_count                 = 14,
        full_brain_fc_loss_weight           = 1.00,
        pd_fit_block_loss_weight            = 0.00,
        correlation_loss_weight             = 0.80,
        rmse_loss_weight                    = 0.20,

        # ── Patch 9: Gradient 3-term loss weights ────────────────
        optimizer_global_corr_weight        = 0.40,
        optimizer_nodewise_corr_weight      = 0.40,
        optimizer_rmse_weight               = 0.20,

        # ── Subcortex FC fitting emphasis: 블록 loss 점유율(합=1) ─
        # mouse CHA 22노드 라벨(FRO_/BG_/THL_)은 cortex prefix(7Networks_/Cortex_)
        # 매칭 안 돼 전부 1블록 → W 균일 → 자동 off. 켜려면 data_loader.
        # _CORTEX_LABEL_PREFIXES에 CHA cortex 규칙 추가. 값은 그대로 둬도 무해.
        fc_block_share_cortex               = 0.50,
        fc_block_share_cross                = 0.40,
        fc_block_share_subsub               = 0.10,

        # ── Part 4 — DBS ─────────────────────────────────────────
        dbs_target_regions           = p["dbs_target_regions"],
        dbs_pulse_amplitude                 = 10.0,
        dbs_stimulation_frequency_hz        = 130.0,
        dbs_phase_duration_steps            = 1,
        dbs_pre_stimulation_duration_ms     = 720_000.0,
        dbs_stimulation_duration_ms         = 60_000.0,
        dbs_fc_pre_transient_skip_ms        = 60_000.0,   # pre-stim FC 앞 transient 제거

        wc_c_ei_init = p["wc_c_ei_init"],

        # ── Bold HRF (dataset별 자동 설정, Patch 21) ───────────
        bold_hrf_k1          = p["bold_hrf_k1"],
        bold_hrf_V0          = p["bold_hrf_V0"],
        bold_hrf_tau_s       = p["bold_hrf_tau_s"],
        bold_hrf_tau_f       = p["bold_hrf_tau_f"],
        bold_hrf_scaling     = p["bold_hrf_scaling"],
        bold_hrf_duration_ms = p["bold_hrf_duration_ms"],
    )

# ── 5. 메인 실행 ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.dataset = "nor"
    args.skip_dbs = True  # CHA-22 익명 라벨 → DBS 타깃 매핑 불가 → DBS 비활성

    print("=" * 60)
    print(f"  EI Tuning Pipeline — dataset={args.dataset}")
    print("=" * 60)

    # fic-only 플래그
    if args.fic_only:
        args.skip_eib      = True
        args.skip_gradient = True
        args.skip_dbs      = True

    # Config (Cell 5)
    cfg = make_config(args.dataset)
    if args.freeze_c_ei is not None:               # CLI override (--freeze-c-ei / --no-freeze-c-ei)
        cfg.freeze_c_ei_after_fic = args.freeze_c_ei
    print(f"[cfg] freeze_c_ei_after_fic={cfg.freeze_c_ei_after_fic}")
    cfg.print_summary()

    # Data loading (Cell 7)
    print("\n[0] Loading data...")
    data = load_data(cfg)

    # fc_target 대각 행렬 0으로 변환
    import numpy as _np_diag
    _np_diag.fill_diagonal(data["fc_target"], 0.0)
    print("[DATA] fc_target diagonal → 0")

    # Network build + warmup (Cell 9)
    print("\n[0] Building network + warmup...")
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

    initial_params = ParamSet.default(
        data["n_nodes"], c_ei_init=cfg.wc_c_ei_init
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)

    bundle_init = StateBundle.from_warmup(
        warmup_result          = warmup_result,
        bold_monitor_template  = bold_monitor,
        initial_params         = initial_params,
        internal_state         = capture_internal_state(initial_state),
        delay_history          = capture_network_delay_history(network),
        stage                  = "warmup",
    )
    print(bundle_init)

    # Part 1: FIC (Cell 13)
    if not args.skip_fic:
        print("\n[1] Running FIC...")
        bundle_fic = run_fic(
            network   = network,
            bundle_in = bundle_init,
            cfg       = cfg,
            data      = data,
        )
        print(f"[FIC] c_ei_frozen={bundle_fic.params.c_ei_frozen}  "
              f"mean c_ei={bundle_fic.params.c_ei.mean():.4f}")
        print(bundle_fic)
    else:
        print("\n[1] FIC skipped (--skip-fic)")
        bundle_fic = bundle_init

    # Part 2: EIB (Cell 15)
    if not args.skip_eib:
        print("\n[2] Running EIB...")
        bundle_eib = run_eib(
            network   = network,
            bundle_in = bundle_fic,
            cfg       = cfg,
            data      = data,
        )
        print(f"[EIB] stage={bundle_eib.stage}  c_ei_frozen={bundle_eib.params.c_ei_frozen}")
        print(bundle_eib)
    else:
        print("\n[2] EIB skipped (--skip-eib)")
        bundle_eib = bundle_fic

    # Part 3: Full Gradient (Cell 17)
    if not args.skip_gradient:
        print("\n[3] Running Gradient Optimization...")
        bundle_grad = run_gradient_optimization(
            network       = network,
            bundle_in     = bundle_eib,
            warmup_bundle = bundle_init,
            cfg           = cfg,
            data          = data,
        )
        print(f"[Part3] stage={bundle_grad.stage}  c_ei_frozen={bundle_grad.params.c_ei_frozen}")
        print(bundle_grad)
    else:
        print("\n[3] Gradient skipped (--skip-gradient)")
        bundle_grad = bundle_eib

    # Part 4: DBS — 항상 Part 3 (Full Gradient) 결과를 입력으로 사용 (Cell 21)
    if not args.skip_dbs:
        print("\n[4] Running DBS Stimulation...")
        run_dbs_stimulation(
            network   = network,
            bundle_in = bundle_grad,
            cfg       = cfg,
            data      = data,
        )
    else:
        print("\n[4] DBS skipped (--skip-dbs)")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
