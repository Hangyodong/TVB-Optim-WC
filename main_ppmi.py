#!/usr/bin/env python3
"""
main_ppmi.py — EI Tuning 파이프라인 스크립트 실행 (ppmi: 216-node SC/FC from .mat)

Usage:
    python3 main_ppmi.py                          # subject idx=1, noise=0.02
    python3 main_ppmi.py --subject-idx 0          # N번째 subject 선택 (0-based)
    python3 main_ppmi.py --noise-level 0.0        # 무잡음
    python3 main_ppmi.py --fic-only               # FIC만 실행
    python3 main_ppmi.py --resume                 # 캐시 이어서

노트북(main_ppmi.ipynb)과 동일한 실행 흐름.
PPMI_SC_FC_cell.mat 의 SUBJECT_IDX 번째 subject 를 CSV 로 추출해 사용한다.
DBS 는 STN 노드 매핑 불명 → skip.
"""
import argparse
import os
import sys

# ── 0. JAX 환경변수 (노트북 Cell 1과 동일) ───────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR",   "platform")
# Part3/3.5 full backprop(210TR×2.4s=504k step)은 remat 없으면 OOM(187GiB).
# scan-body checkpointing 기본 ON (PART3_REMAT_SCAN=0 으로 끌 수 있음). main_ppmi_pre 와 동일.
os.environ.setdefault("PART3_REMAT_SCAN", "1")

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
from part3_gradient      import run_gradient_optimization, run_sigma_optimization
from part4_dbs           import run_dbs_stimulation
from pipeline_contracts  import (
    ParamSet,
    StateBundle,
    capture_internal_state,
    capture_network_delay_history,
)

# ── PPMI .mat 경로 (노트북 Cell 2와 동일) ────────────────────────────────
MAT_PATH = "/scratch/home/wog3597/vbi/PPMI_SC_FC_cell.mat"
PPMI_DIR = "ppmi"        # 변환된 CSV 출력 디렉토리


# ── 2. 인자 파싱 ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="EI Tuning Pipeline (ppmi: 216-node SC/FC from .mat)")
    parser.add_argument(
        "--dataset", choices=["ppmi"], default="ppmi",
        help="Dataset to use (fixed: ppmi)"
    )
    parser.add_argument(
        "--subject-idx", dest="subject_idx", type=int, default=1,
        help="사용할 subject 0-based index (0 ~ N-1). 노트북 SUBJECT_IDX",
    )
    parser.add_argument(
        "--noise-level", dest="noise_level", type=float, default=0.02,
        help="가산 잡음 표준편차 σ (additive_noise_sigma). 0 이면 무잡음",
    )
    parser.add_argument("--skip-fic",      action="store_true", help="FIC 건너뜀")
    parser.add_argument("--skip-eib",      action="store_true", help="EIB 건너뜀")
    parser.add_argument("--skip-gradient", action="store_true", help="Gradient 건너뜀")
    parser.add_argument("--skip-sigma",    action="store_true", help="Part3.5 σ 튜닝 건너뜀")
    parser.add_argument("--skip-dbs",      action="store_true", help="DBS 건너뜀")
    parser.add_argument("--fic-only",      action="store_true", help="FIC만 실행")
    # ── Part 3.5 — per-node noise σ 튜닝 ──────────────────────────────────
    parser.add_argument("--sigma-shared",  dest="sigma_per_node", action="store_false",
                        default=True, help="공유 스칼라 σ 1개(파일럿). 기본은 per-node")
    parser.add_argument("--sigma-max",     dest="sigma_max", type=float, default=0.10,
                        help="σ BoundedParameter 상한 (기본 0.10)")
    parser.add_argument("--sigma-l2",      dest="sigma_l2", type=float, default=0.01,
                        help="σ spread L2 가중치 (기본 0.01)")
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


# ── 3. PPMI .mat → CSV 추출 (노트북 Cell 2와 동일) ───────────────────────
def prepare_ppmi_data(subject_idx: int, noise_level: float) -> dict:
    """PPMI_SC_FC_cell.mat 의 subject_idx 번째 데이터를 CSV 로 추출하고
    _DATASET_PARAMS['ppmi'] 에 해당하는 dict 를 반환한다."""
    import numpy as _np_ppmi
    import scipy.io as _sio_ppmi

    os.makedirs(PPMI_DIR, exist_ok=True)

    _m = _sio_ppmi.loadmat(MAT_PATH)
    #   1열 subject_num, 2열 SC weight, 3열 SC tract_length, 4열 FC matrix
    #   stack 형태(N,216,216)가 있으면 우선 사용, 없으면 cell 에서 추출
    _subj = _np_ppmi.asarray(_m["subject_num"]).ravel()
    _N_SUBJ = _subj.shape[0]
    assert 0 <= subject_idx < _N_SUBJ, f"subject_idx must be 0..{_N_SUBJ-1}, got {subject_idx}"

    if "weight_stack" in _m:
        _w  = _np_ppmi.asarray(_m["weight_stack"])[subject_idx]
        _tl = _np_ppmi.asarray(_m["tract_length_stack"])[subject_idx]
        _fc = _np_ppmi.asarray(_m["fc_stack"])[subject_idx]
    else:
        _w  = _np_ppmi.asarray(_m["weight_cell"][subject_idx, 0])
        _tl = _np_ppmi.asarray(_m["tract_length_cell"][subject_idx, 0])
        _fc = _np_ppmi.asarray(_m["fc_cell"][subject_idx, 0])

    _w  = _np_ppmi.nan_to_num(_w,  nan=0.0).astype(_np_ppmi.float64)
    _tl = _np_ppmi.nan_to_num(_tl, nan=0.0).astype(_np_ppmi.float64)
    _fc = _np_ppmi.nan_to_num(_fc, nan=0.0).astype(_np_ppmi.float64)
    _n_nodes = _w.shape[0]
    assert _w.shape == _tl.shape == _fc.shape == (_n_nodes, _n_nodes), "matrix shape mismatch"

    _subj_id = int(_subj[subject_idx])

    # CSV 저장 (data_loader 는 header 없는 CSV 로 읽는다)
    _sc_csv  = os.path.join(PPMI_DIR, "weight.csv")
    _len_csv = os.path.join(PPMI_DIR, "tract_length.csv")
    _fc_csv  = os.path.join(PPMI_DIR, "FC.csv")
    _reg_txt = os.path.join(PPMI_DIR, "region_labels.txt")

    _np_ppmi.savetxt(_sc_csv,  _w,  delimiter=",")
    _np_ppmi.savetxt(_len_csv, _tl, delimiter=",")
    _np_ppmi.savetxt(_fc_csv,  _fc, delimiter=",")

    # region label 매핑: PPMI 216 = Schaefer200 cortical(1-200, "7Networks_*")
    #   + 16 subcortical(201-216, PD25: red nucleus/SN/STN/caudate/putamen/GPe/GPi/thalamus L·R).
    # Custom_Schaefer200_7net_PD25subcortex.txt(216줄, "<idx> <label>")의 label 컬럼만 추출해 저장
    # → derive_cortex_subcortex_indices가 cortex/subcortex 분리 → cc/cross/ss block corr 활성.
    _LABEL_SRC = os.path.join(os.path.dirname(os.path.abspath(MAT_PATH)).replace("/vbi", "/optim"),
                              "Custom_Schaefer200_7net_PD25subcortex.txt")
    if not os.path.exists(_LABEL_SRC):
        _LABEL_SRC = "Custom_Schaefer200_7net_PD25subcortex.txt"   # optim/ 작업 디렉토리 fallback
    with open(_LABEL_SRC, "r", encoding="utf-8-sig") as _fh:
        _raw_labels = [ln.strip() for ln in _fh if ln.strip()]
    assert len(_raw_labels) == _n_nodes, (
        f"label count {len(_raw_labels)} != n_nodes {_n_nodes} ({_LABEL_SRC})")
    _labels = []
    for _ln in _raw_labels:
        _parts = _ln.split(None, 1)                      # "<idx> <label>" → label
        _labels.append(_parts[1].strip() if len(_parts) == 2 and _parts[0].isdigit() else _ln)
    with open(_reg_txt, "w") as _fh:
        for _lab in _labels:
            _fh.write(_lab + "\n")
    _n_ctx = sum(1 for _l in _labels if _l.startswith(("7Networks_", "Cortex_")))

    print(f"Dataset: ppmi  (subject idx={subject_idx} / {_N_SUBJ},  subject_num={_subj_id})")
    print(f"  n_nodes = {_n_nodes}")
    print(f"  weight     -> {_sc_csv}")
    print(f"  tract_len  -> {_len_csv}")
    print(f"  FC         -> {_fc_csv}")
    print(f"  region_txt -> {_reg_txt}  (cortex={_n_ctx}, subcortex={_n_nodes - _n_ctx})  src={_LABEL_SRC}")

    return dict(
        # RWW 모델 사용 — WC 파라미터는 Config 요구사항으로 유지(모델 미참조). mouse 값 차용.
        wc_c_ei_init=1.0,   # RWW 표준 J_i (10.0이면 S_e가 0 근처로 짓눌려 FIC 불가)
        fic_target_firing_rate_hz=2.0,   # 진단용 Hz 표시 (FIC 제어는 fic_target_se=S_e gating)
        # PPMI 216-node atlas — STN 노드 매핑 불명 → DBS skip
        dbs_target_regions={},
        bold_hrf_k1=5.6,
        bold_hrf_V0=0.02,
        bold_hrf_tau_s=0.8,
        bold_hrf_tau_f=0.4,
        bold_hrf_scaling=1.0 / 3.0,
        bold_hrf_duration_ms=32_000.0,
        sc_csv=_sc_csv,
        length_csv=_len_csv,
        fc_csv=_fc_csv,
        region_txt=_reg_txt,
        tract_conduction_speed=3.0,
        additive_noise_sigma=noise_level,
    )


# ── 4. Config 생성 (노트북 cfg=Config(...) 셀과 동일) ────────────────────
def make_config(p: dict, subject_idx: int) -> Config:
    return Config(
        # ── 데이터 경로 ──────────────────────────────────────────
        region_txt                          = p["region_txt"],
        sc_csv                              = p["sc_csv"],
        length_csv                          = p["length_csv"],
        fc_csv                              = p["fc_csv"],
        cache_version                       = f"v_ppmi216_s{subject_idx}",
        cache_run_label                     = "ppmi216",

        # ── 시뮬레이션 공통 ──────────────────────────────────────
        integration_dt_ms                   = 1.0,
        warmup_duration_ms                  = 720_000,
        # -- Patch 13: Bold HRF parameters -------------------
        bold_hrf_k1          = p["bold_hrf_k1"],
        bold_hrf_V0          = p["bold_hrf_V0"],
        bold_hrf_tau_s       = p["bold_hrf_tau_s"],
        bold_hrf_tau_f       = p["bold_hrf_tau_f"],
        bold_hrf_scaling     = p["bold_hrf_scaling"],
        bold_hrf_duration_ms = p["bold_hrf_duration_ms"],

        bold_repetition_time_ms             = 2400.0,   # PPMI 실측 TR=2.4s (was 1000)
        tract_conduction_speed              = p["tract_conduction_speed"],
        additive_noise_sigma                = p["additive_noise_sigma"],

        # ── Part 1 — FIC (구버전 notebook 로직) ─────────────────
        fic_target_firing_rate_hz           = p["fic_target_firing_rate_hz"],
        fic_learning_rate                   = 0.5,    # EI_Tuning 참조값 (init c_ei=1.0 기준)
        fic_max_iterations                  = 2000,
        fic_early_stop_patience             = 500,
        fic_step_duration_ms                = 2_400,   # =1 TR @2.4s (total_tr=dur/TR≥1 필수)
        fic_step_skip_tr                    = 0,
        fic_posthoc_duration_ms             = 504_000,  # 210 TR × 2.4s (config 기본 720k override)
        fic_posthoc_skip_tr                 = 42,       # 20% of 210
        freeze_c_ei_after_fic               = False,   # True: Part2(EIB)에서만 c_ei 동결. Part3는 항상 c_ei 최적화

        # ── Part 2 — EIB (구버전 notebook 로직) ─────────────────
        eib_max_iterations                  = 10000,
        eib_internal_fic_learning_rate      = 0.05,
        eib_max_weight_learning_rate        = 0.002,
        eib_bold_window_samples             = 210,   # PPMI 210 TR (was 720) — 경험 데이터와 동일 샘플수
        eib_snapshot_save_interval          = 50,
        connectivity_weight_max             = 2.0,   # B: was 1.5. wLRE 14.7% 천장 포화 해소 → FC 더 세게 빌드 → corr↑→RMSE↓
        eib_posthoc_duration_ms             = 504_000,  # 210 TR × 2.4s (was 720k)
        eib_posthoc_skip_tr                 = 42,        # 20% of 210 (was 60)

        # ── Part 3 — Full Gradient (구버전 notebook 로직) ───────
        optimizer_learning_rate             = 0.0001,   # (b) 5e-4→1e-4: adamaxw overshoot 방지(step1 sweet spot 유지)
        optimizer_max_steps                 = 250,   # A: was 100. step100서 loss 미수렴(마지막 chunk 5~8% 강하중) → 수렴분 회수
        optimizer_chunk_steps               = 10,
        optimizer_bold_window_tr            = 210,  # EIB(210)와 동일 측정선 = PPMI 경험 TR수. 504k ms < 720k → backprop step↓. PART3_REMAT_SCAN=1 권장.
        optimizer_bold_skip_tr              = 42,   # 20% of 210, 유효 168 TR

        # ── EIB score 계산용 ──────────────────────────────────────
        pd_fit_region_count                 = 14,
        full_brain_fc_loss_weight           = 1.00,
        correlation_loss_weight             = 0.80,   # EIB scoring (keep)
        rmse_loss_weight                    = 0.20,   # EIB scoring (keep)

        # ── Patch 9: Gradient 3-term loss weights ────────────────
        optimizer_global_corr_weight        = 0.80,   # alpha: global FC corr
        optimizer_nodewise_corr_weight      = 0.40,   # beta:  node-wise FC corr (part3 미사용)
        optimizer_rmse_weight               = 0.20,   # gamma: global FC RMSE
        optimizer_rmse_block                = True,   # rmse도 cc/cross/ss 블록분할(corr_block 가중 재사용) — subcortex 균형
        optimizer_activity_weight           = 0.0,    # (b) 0.01→0: EIB엔 없던 S_e 게이팅 항 제거(EIB 최적점 안 흔들게)

        # ── Subcortex FC fitting emphasis: 블록 loss 점유율(합=1) ─
        fc_block_share_cortex               = 0.50,
        fc_block_share_cross                = 0.40,
        fc_block_share_subsub               = 0.10,

        # ── Block-split corr loss 가중치(part2 선택 + part3 gradient) ──
        corr_block_weight_cc                = 0.40,   # cortex-cortex corr
        corr_block_weight_cross             = 0.40,   # cross (ctx↔sub) corr
        corr_block_weight_subsub            = 0.20,   # sub-sub corr (edge 적어 noisy→낮게)

        # ── Part 4 — DBS ─────────────────────────────────────────
        dbs_target_regions               = p["dbs_target_regions"],
        dbs_pulse_amplitude                 = 10.0,
        dbs_stimulation_frequency_hz        = 130.0,
        dbs_pre_stimulation_duration_ms     = 60_000.0,
        dbs_stimulation_duration_ms         = 60_000.0,

        wc_c_ei_init = p["wc_c_ei_init"],
    )


# ── 5. 메인 실행 ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.dataset = "ppmi"
    args.skip_dbs = True  # PPMI 216-node atlas → STN 표적 매핑 불가 → DBS 비활성

    print("=" * 60)
    print(f"  EI Tuning Pipeline — dataset={args.dataset} (subject_idx={args.subject_idx})")
    print("=" * 60)

    # fic-only 플래그
    if args.fic_only:
        args.skip_eib      = True
        args.skip_gradient = True
        args.skip_sigma    = True
        args.skip_dbs      = True
    # gradient skip 시 σ 튜닝(part3 출력 의존) 자동 skip
    if args.skip_gradient:
        args.skip_sigma = True

    # PPMI .mat → CSV 추출 (Cell 2)
    p = prepare_ppmi_data(args.subject_idx, args.noise_level)

    # Config (cfg=Config 셀)
    cfg = make_config(p, args.subject_idx)
    if args.freeze_c_ei is not None:               # CLI override (--freeze-c-ei / --no-freeze-c-ei)
        cfg.freeze_c_ei_after_fic = args.freeze_c_ei
    print(f"[cfg] freeze_c_ei_after_fic={cfg.freeze_c_ei_after_fic}")
    cfg.print_summary()

    # Data loading (Cell 3)
    print("\n[0] Loading data...")
    data = load_data(cfg)

    # Network build + warmup (Cell 4)
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

    # Part 1: FIC (Cell 5)
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

    # Part 2: EIB (Cell 6)
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

    # Part 3: Full Gradient (Cell 7)
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

    # Part 3.5: per-node noise σ 튜닝 (c_ei/wLRE/wFFI freeze, σ만 autodiff)
    if not args.skip_sigma:
        print("\n[3.5] Running per-node noise σ optimization...")
        bundle_sigma = run_sigma_optimization(
            network         = network,
            bundle_in       = bundle_grad,
            cfg             = cfg,
            data            = data,
            sigma_per_node  = args.sigma_per_node,
            sigma_max       = args.sigma_max,
            sigma_l2_weight = args.sigma_l2,
        )
        meta = bundle_sigma.metadata
        print(f"[Part3.5] stage={bundle_sigma.stage}  "
              f"post_sigma_fc_corr={meta.get('post_sigma_fc_corr')}  "
              f"post_sigma_fc_rmse={meta.get('post_sigma_fc_rmse')}")
        print(bundle_sigma)
        bundle_grad = bundle_sigma   # downstream(Part4)도 σ 최적 bundle 사용
    else:
        print("\n[3.5] σ optimization skipped (--skip-sigma)")

    # Part 4: DBS — PPMI 216-node atlas는 STN 표적 매핑 불가 → DBS skip (Cell 8)
    if not args.skip_dbs:
        print("\n[4] Running DBS Stimulation...")
        run_dbs_stimulation(
            network   = network,
            bundle_in = bundle_grad,
            cfg       = cfg,
            data      = data,
        )
    else:
        print("\n[4] DBS skipped — PPMI 216-node atlas has no STN target mapping")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
