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
from part3_gradient      import run_gradient_optimization, run_lowrank_optimization
from part4_dbs           import run_dbs_stimulation
from pipeline_contracts  import (
    ParamSet,
    StateBundle,
    capture_internal_state,
    capture_network_delay_history,
)

# ── 2. 인자 파싱 ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="EI Tuning Pipeline")
    parser.add_argument(
        "--dataset", choices=["mouse", "human"], default="mouse",
        help="Dataset to use (default: mouse)"
    )
    parser.add_argument("--skip-fic",      action="store_true", help="FIC 건너뜀")
    parser.add_argument("--skip-eib",      action="store_true", help="EIB 건너뜀")
    parser.add_argument("--skip-gradient", action="store_true", help="Gradient 건너뜀")
    parser.add_argument("--skip-lowrank",  action="store_true", help="LowRank 건너뜀")
    parser.add_argument("--skip-dbs",      action="store_true", help="DBS 건너뜀")
    parser.add_argument("--fic-only",      action="store_true", help="FIC만 실행")
    parser.add_argument("--resume",        action="store_true", help="캐시 이어서 실행")
    return parser.parse_args()

# ── 3. Dataset별 파라미터 (노트북 Cell 4와 동일) ─────────────────────────
_DATASET_PARAMS = {
    "human": dict(
        # SanzLeonet et al. 2014 — frequency peak at 20 Hz
        wc_c_ee=10.0, wc_c_ei=6.0, wc_c_ie=10.0, wc_c_ii=1.0,
        wc_r_e=0.0,  wc_r_i=0.0,
        wc_tau_e=10.0, wc_tau_i=10.0,
        wc_alpha_e=1.2, wc_alpha_i=2.0,
        wc_theta_e=2.0, wc_theta_i=3.5,
        wc_k_e=1.0, wc_k_i=1.0,
        wc_a_e=1.0, wc_a_i=1.0,
        wc_b_e=0.0, wc_b_i=0.0,
        wc_c_e=1.0, wc_c_i=1.0,
        wc_P=0.5, wc_Q=0.0, wc_lamda=1.0,
        wc_rE_max_hz=20.0, wc_rI_max_hz=20.0,
        wc_c_ei_init=6.0,
        fic_target_firing_rate_hz=4.0,
        # DBS targets (PD25subcortex, 0-based node index)
        dbs_target_regions={
            "STN_L": 404,
            "GPe_L": 410,
            "GPe_R": 411,
            "GPi_L": 412,
        },
        # Bold HRF: library defaults for human
        bold_hrf_k1=5.6,
        bold_hrf_V0=0.02,
        bold_hrf_tau_s=0.8,
        bold_hrf_tau_f=0.4,
        bold_hrf_scaling=1.0 / 3.0,
        bold_hrf_duration_ms=20_000.0,  # 20s (library default)
        sc_csv="human/weight.csv",
        length_csv="human/tract_length.csv",
        fc_csv="human/fc_matrix.csv",
        region_txt="human/Custom_Schaefer400_PD25subcortex_1mm.txt",
        tract_conduction_speed=1.0,
        additive_noise_sigma=0.01,
    ),
    "mouse": dict(
        # Current mouse parameters
        wc_c_ee=11.0, wc_c_ei=10.0, wc_c_ie=10.0, wc_c_ii=1.0,
        wc_r_e=1.0,  wc_r_i=1.0,
        wc_tau_e=10.0, wc_tau_i=10.0,
        wc_alpha_e=1.2, wc_alpha_i=2.0,
        wc_theta_e=2.0, wc_theta_i=3.5,
        wc_k_e=1.0, wc_k_i=1.0,
        wc_a_e=1.0, wc_a_i=1.0,
        wc_b_e=0.0, wc_b_i=0.0,
        wc_c_e=1.0, wc_c_i=1.0,
        wc_P=0.5, wc_Q=0.0, wc_lamda=1.0,
        wc_rE_max_hz=20.0, wc_rI_max_hz=20.0,
        wc_c_ei_init=10.0,
        fic_target_firing_rate_hz=2.0,
        # DBS targets (Atlas_43, 0-based node index)
        dbs_target_regions={
            "STN_L": 11,
            "GPe_L": 5,
            "GPe_R": 6,
            "GPi_L": 7,
        },
        # Bold HRF: mouse-specific
        bold_hrf_k1=5.6,
        bold_hrf_V0=0.02,
        bold_hrf_tau_s=0.8,
        bold_hrf_tau_f=0.4,
        bold_hrf_scaling=1.0 / 3.0,
        bold_hrf_duration_ms=32_000.0,  # 32s (mouse-specific)
        sc_csv="weight_nor.csv",
        length_csv="tract_length_nor.csv",
        fc_csv="FC_nor.csv",
        region_txt="Custom_Schaefer400_PD25subcortex_1mm.txt",
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

        # ── Part 3B — Low-rank (Full Gradient 결과 기반) ────────
        lowrank_rank                        = 6,
        lowrank_max_steps                   = 300,
        lowrank_learning_rate               = 0.0002,
        lowrank_bold_window_tr              = 720,
        lowrank_bold_skip_tr                = 60,
        lowrank_delta_scale                 = 0.15,
        lowrank_factor_init                 = 0.01,
        lowrank_activity_weight             = 0.01,
        lowrank_factor_penalty              = 1e-4,
        lowrank_seed                        = 17,

        # ── Phase 1 final baseline settle ────────────────────────
        baseline_settle_duration_ms         = 0,

        # ── EIB score 계산용 ──────────────────────────────────────
        pd_fit_region_count                 = 14,
        full_brain_fc_loss_weight           = 1.00,
        pd_fit_block_loss_weight            = 0.00,
        correlation_loss_weight             = 0.80,
        rmse_loss_weight                    = 0.20,

        # ── Patch 9: Gradient / LowRank 3-term loss weights ──────
        optimizer_global_corr_weight        = 0.40,
        optimizer_nodewise_corr_weight      = 0.40,
        optimizer_rmse_weight               = 0.20,
        lowrank_global_corr_weight          = 0.40,
        lowrank_nodewise_corr_weight        = 0.40,
        lowrank_rmse_weight                 = 0.20,

        # ── Part 4 — DBS ─────────────────────────────────────────
        dbs_target_regions           = p["dbs_target_regions"],
        dbs_pulse_amplitude                 = 10.0,
        dbs_stimulation_frequency_hz        = 130.0,
        dbs_phase_duration_steps            = 1,
        dbs_pre_stimulation_duration_ms     = 60_000.0,
        dbs_stimulation_duration_ms         = 60_000.0,

        # ── WC model params (dataset별 자동 설정, Patch 15) ─────
        wc_c_ee      = p["wc_c_ee"],
        wc_c_ei      = p["wc_c_ei"],
        wc_c_ie      = p["wc_c_ie"],
        wc_c_ii      = p["wc_c_ii"],
        wc_r_e       = p["wc_r_e"],
        wc_r_i       = p["wc_r_i"],
        wc_tau_e     = p["wc_tau_e"],
        wc_tau_i     = p["wc_tau_i"],
        wc_alpha_e   = p["wc_alpha_e"],
        wc_alpha_i   = p["wc_alpha_i"],
        wc_theta_e   = p["wc_theta_e"],
        wc_theta_i   = p["wc_theta_i"],
        wc_k_e       = p["wc_k_e"],
        wc_k_i       = p["wc_k_i"],
        wc_a_e       = p["wc_a_e"],
        wc_a_i       = p["wc_a_i"],
        wc_b_e       = p["wc_b_e"],
        wc_b_i       = p["wc_b_i"],
        wc_c_e       = p["wc_c_e"],
        wc_c_i       = p["wc_c_i"],
        wc_P         = p["wc_P"],
        wc_Q         = p["wc_Q"],
        wc_lamda     = p["wc_lamda"],
        wc_rE_max_hz = p["wc_rE_max_hz"],
        wc_rI_max_hz = p["wc_rI_max_hz"],
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

    print("=" * 60)
    print(f"  EI Tuning Pipeline — dataset={args.dataset}")
    print("=" * 60)

    # fic-only 플래그
    if args.fic_only:
        args.skip_eib      = True
        args.skip_gradient = True
        args.skip_lowrank  = True
        args.skip_dbs      = True

    # Config (Cell 5)
    cfg = make_config(args.dataset)
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

    # Part 3B: LowRank (Cell 19)
    if not args.skip_lowrank:
        print("\n[3B] Running LowRank Optimization...")
        bundle_lowrank = run_lowrank_optimization(
            network       = network,
            bundle_in     = bundle_grad,
            warmup_bundle = bundle_init,
            cfg           = cfg,
            data          = data,
        )
        print(f"[Part3B] stage={bundle_lowrank.stage}  c_ei_frozen={bundle_lowrank.params.c_ei_frozen}")
        print(bundle_lowrank)
    else:
        print("\n[3B] LowRank skipped (--skip-lowrank)")
        bundle_lowrank = bundle_grad

    # Part 4: DBS — grad vs lowrank corr 비교 후 best bundle 선택 (Cell 21)
    if not args.skip_dbs:
        print("\n[4] Running DBS Stimulation...")
        from part3_gradient import compute_simulated_fc
        from tvboptim.observations.observation import fc_corr
        import jax.numpy as _jnp

        _fc_target = _jnp.asarray(data["fc_target"])
        _sim_ms  = 180_000
        _skip_tr = 30

        print("\n" + "=" * 52)
        print("  DBS input bundle 선택 (grad vs lowrank)")
        print("=" * 52)

        _fc_grad = compute_simulated_fc(network, bundle_grad, cfg, _sim_ms, _skip_tr)
        _corr_grad = float(fc_corr(_jnp.asarray(_fc_grad), _fc_target))
        print(f"  grad    corr={_corr_grad:.4f}")

        _fc_lr = compute_simulated_fc(network, bundle_lowrank, cfg, _sim_ms, _skip_tr)
        _corr_lr = float(fc_corr(_jnp.asarray(_fc_lr), _fc_target))
        print(f"  lowrank corr={_corr_lr:.4f}")

        print("-" * 52)
        if _corr_lr >= _corr_grad:
            bundle_for_dbs = bundle_lowrank
            print(f"  → DBS input: lowrank  (corr={_corr_lr:.4f})")
        else:
            bundle_for_dbs = bundle_grad
            print(f"  → DBS input: grad  (corr={_corr_grad:.4f})")
        print("=" * 52 + "\n")

        run_dbs_stimulation(
            network   = network,
            bundle_in = bundle_for_dbs,
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
