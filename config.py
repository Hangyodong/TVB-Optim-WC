"""
config.py
모든 하이퍼파라미터를 한 곳에서 관리한다.
main.ipynb Cell 2에서만 값을 수정한다.

Stage 3 추가
------------
- pd_fit_region_indices / pd_fit_region_labels:
  atlas ordering에 덜 의존하는 PD fit block 지정용.
- bundle_rng_seed:
  stage 경계에서 noise state를 가능한 한 안정적으로 이어가기 위한 기본 seed.
- baseline_settle_duration_ms:
  Part 3 / Part 3B 종료 후 final params로 baseline state를 다시 정의하는 no-stim settling 길이.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class Config:
    # ── 데이터 경로 ──────────────────────────────────────────
    region_txt:     str = "Atlas_43.txt"
    sc_csv:         str = "weight.csv"
    length_csv:     str = "tract_length.csv"
    fc_csv:         str = "FC_compact.csv"
    cache_root_dir: str = "./cache"      # (deprecated; 경로 빌드에 미사용)
    cache_run_label: str = "nor_42"      # 캐시 저장/로드 폴더 이름 → optim/cache/<label>/<cache_tag>
    param_save_dir: str = "./optimized_params"
    cache_version:  str = "v_eituning_oldlogic_match_p3_p7_p8_p9_pm_p12_p13_p14_ce10_p15_p19_p21_p22_p25_p26_p27_p31_p32capfix_p33segating"
    # SC 정규화: "log1pm"=log1p(w)(0 edge 유지) 를 "max" 노드입력에 재스케일. 기본.
    #   약한 edge dynamic range 복원(median NZ 0.2%→~37%/max) + 노드입력 보존(≈max-norm)
    #   → FIC 정상 수렴(S_e=0.25, c_ei~1.16, 포화 0). max-norm이 뭉갠 약한/cross edge 구동력 부여.
    # "max"=w/max (heavy-tail 보존, 안정하나 약한 edge 뭉갬). "log1p"=log1p(w+0.5)/max (입력 16~18×→과흥분, deprecated).
    sc_norm:        str = "log1pm"

    # ── Wilson-Cowan parameters (mouse 전용 · RWW/PD 미사용 死코드) ──
    # main_mouse.py 만 사용. RWW/PD 파이프라인은 아래 RWW 섹션 + model.py 사용.
    # (wc_c_ei_init 만 예외적으로 RWW c_ei(J_i) per-node 초기값으로 공유됨)
    wc_c_ee      : float = 11.0    # mouse default
    wc_c_ei      : float = 10.0
    wc_c_ie      : float = 10.0
    wc_c_ii      : float = 1.0
    wc_r_e       : float = 1.0
    wc_r_i       : float = 1.0
    wc_tau_e     : float = 10.0
    wc_tau_i     : float = 10.0
    wc_alpha_e   : float = 1.2
    wc_alpha_i   : float = 2.0
    wc_theta_e   : float = 2.0
    wc_theta_i   : float = 3.5
    wc_k_e       : float = 1.0
    wc_k_i       : float = 1.0
    wc_a_e       : float = 1.0
    wc_a_i       : float = 1.0
    wc_b_e       : float = 0.0
    wc_b_i       : float = 0.0
    wc_c_e       : float = 1.0
    wc_c_i       : float = 1.0
    wc_P         : float = 0.5
    wc_Q         : float = 0.0
    wc_lamda     : float = 1.0
    wc_rE_max_hz : float = 20.0
    wc_rI_max_hz : float = 20.0
    wc_c_ei_init : float = 10.0   # FIC 초기값 (RWW c_ei=J_i per-node 초기값으로도 공유; make_config가 1.0 지정)

    # ── RWW (Reduced Wong-Wang) model parameters ──────────────
    # 실제 PD/RWW 모델(model.py ReducedWongWangEIB)이 쓰는 파라미터. build_network가
    # cfg.rww_* 를 읽어 dynamics 생성자에 넘긴다 (값=model.py DEFAULT_PARAMS 동일 → 동작 불변).
    # c_ei(=J_i)는 per-node 최적화 대상이라 여기 없음(초기값=wc_c_ei_init, FIC/part3가 조정).
    rww_a_e       : float = 310.0        # E input gain
    rww_b_e       : float = 125.0        # E input shift [Hz]
    rww_d_e       : float = 0.160        # E input scaling [s]
    rww_gamma_e   : float = 0.641 / 1000 # E kinetic
    rww_tau_e     : float = 100.0        # E NMDA decay [ms]
    rww_w_p       : float = 1.4          # recurrent excitation
    rww_W_e       : float = 1.0          # E external input scale
    rww_a_i       : float = 615.0        # I input gain
    rww_b_i       : float = 177.0        # I input shift [Hz]
    rww_d_i       : float = 0.087        # I input scaling [s]
    rww_gamma_i   : float = 1.0 / 1000   # I kinetic
    rww_tau_i     : float = 10.0         # I GABA decay [ms]
    rww_W_i       : float = 0.7          # I external input scale
    rww_J_N       : float = 0.15         # NMDA current [nA]
    rww_I_o       : float = 0.382        # external input current
    rww_I_ext     : float = 0.0          # extra external (DBS 등)
    rww_lamda     : float = 1.0          # coupling scale
    rww_rE_max_hz : float = 20.0         # E firing-rate 진단 상한
    rww_rI_max_hz : float = 20.0         # I firing-rate 진단 상한

    # ── 시뮬레이션 공통 ──────────────────────────────────────
    integration_dt_ms:       float = 1.0
    warmup_duration_ms:      int   = 300_000
    bold_repetition_time_ms: float = 1000.0

    # ── Bold monitor HRF parameters (Patch 13) ──────────────
    # Calibrate to match mouse HRF target
    bold_hrf_k1          : float = 5.6
    bold_hrf_V0          : float = 0.02
    bold_hrf_tau_s       : float = 0.8        # seconds
    bold_hrf_tau_f       : float = 0.4        # seconds
    bold_hrf_scaling     : float = 1.0 / 3.0
    bold_hrf_duration_ms : float = 20_000.0   # ms (20s default; mouse uses 32s)
    tract_conduction_speed:  float = 3.0
    # True(기본) 면 model.build_network 가 EIBDelayedCoupling(DDE) 을 써서 tract delay
    # (=length/tract_conduction_speed) 를 실제 시뮬에 반영한다. False=delay-free ODE/SDE.
    # 비용은 163노드에서 forward·backprop 모두 +1% 수준(측정치).
    use_delay:               bool  = True
    additive_noise_sigma:    float = 0.01
    bundle_rng_seed:         int   = 42
    fc_eval_n_seeds:         int   = 1   # >1: FC eval을 N개 noise seed로 평균 → corr/rmse 분산↓ (기본 1=현행)

    # ── FC 시각화 공통 ────────────────────────────────────────
    fc_plot_vmin: float = -1.0
    fc_plot_vmax: float =  1.0

    # ── Part 1 — FIC ─────────────────────────────────────────
    # EI_Tuning FIC: S_e gating 을 타깃으로 c_ei(=J_i) per-node 조정.
    #   d_c_ei = eta * mean_S_i * (mean_S_e - fic_target_se)
    fic_target_se:               float = 0.25  # FIC 제어 타깃 (S_e gating, EI_Tuning)
    fic_early_stop_tolerance_se: float = 0.005 # |mean S_e − target| 수렴 허용치
    fic_target_firing_rate_hz:   float = 4.0   # 진단용 발화율(Hz) 표시 (FIC 제어엔 미사용)
    fic_learning_rate:           float = 1e-3
    fic_max_iterations:          int   = 2000
    fic_posthoc_top_k:           int   = 10    # best 후보 재시뮬 개수. 스윕에선 1~2 로
    fic_early_stop_patience:     int   = 500
    fic_early_stop_window:       int   = 50    # se_error 이동평균 창. 순간값은
    #   노이즈로 0.002~0.035 진동해 "연속 N step 미만" 조건이 사실상 안 걸린다.
    fic_early_stop_tolerance_hz: float = 0.10  # (deprecated; tolerance_se 사용)
    fic_step_duration_ms:        int   = 1_000
    fic_step_skip_tr:            int   = 0
    # best 후보 post-hoc 평가용 (EIB posthoc 기본값과 동일)
    fic_posthoc_duration_ms:     int   = 720_000
    fic_posthoc_skip_tr:         int   = 20
    # True: lock per-node c_ei after FIC (EIB / Part3 / Part3B keep it frozen).
    # False (default): old-logic — c_ei keeps updating through EIB and gradient.
    freeze_c_ei_after_fic:       bool  = False

    # ── Part 2 — EIB ─────────────────────────────────────────
    eib_max_iterations:             int   = 8000
    eib_internal_fic_learning_rate: float = 0.05
    eib_max_weight_learning_rate:   float = 0.002
    eib_bold_window_samples:        int   = 150
    eib_snapshot_save_interval:     int   = 50
    connectivity_weight_max:        float = 1.5

    eib_posthoc_duration_ms:        int   = 720_000
    eib_posthoc_skip_tr:            int   = 20
    # posthoc 후보 스냅샷 수: 수렴영역에서 균등 K개 재시뮬 → true corr 최고 선택.
    # (patch30 단일 argmax-window 는 window 과적합 스냅샷을 골라 true corr 낮을 위험 → 다수후보로 완화)
    eib_posthoc_top_k:              int   = 8

    # ── Part 3 — Full-matrix gradient ────────────────────────
    optimizer_learning_rate:  float = 0.002
    optimizer_max_steps:      int   = 200
    optimizer_chunk_steps:    int   = 5
    optimizer_bold_window_tr: int   = 720
    optimizer_bold_skip_tr:   int   = 8
    optimizer_activity_weight: float = 0.01   # full-matrix activity reg weight (was hardcoded)

    # 캐시 pkl 경량화: post-hoc neural trace 를 이 stride 로 다운샘플해 저장(plot 전용, metric 무관).
    # 1=원본(기본, 기존 러너 불변). 예: 50 → 600k step→12k 샘플, 557MB→~11MB.
    neural_cache_stride: int = 1

    # ── Phase 1 final baseline settle ────────────────────────
    baseline_settle_duration_ms: Optional[int] = 0

    # ── EIB score 계산용 ──────────────────────────────────────
    pd_fit_region_count:       int   = 14
    full_brain_fc_loss_weight: float = 0.35
    pd_fit_block_loss_weight:  float = 0.65
    correlation_loss_weight:   float = 0.70
    rmse_loss_weight:          float = 0.30

    # === Patch 9: 3-term loss weights ===
    optimizer_global_corr_weight:   float = 0.4   # alpha: global FC corr
    optimizer_nodewise_corr_weight: float = 0.4   # beta:  node-wise FC corr
    optimizer_rmse_weight:          float = 0.2   # gamma: global FC RMSE
    # rmse 항을 block_corr처럼 cc/cross/ss 블록분할(corr_block_weight_* 가중 재사용).
    # False=plain off-diag RMSE(cortex 지배). True=블록 동등가중(subcortex 균형).
    # subcortex 없는 atlas는 cc만 활성 → True/False 동치.
    optimizer_rmse_block:           bool  = False

    # ── Subcortex FC fitting emphasis (블록 loss 점유율) ──────
    # whole-brain corr는 cortex-cortex edge(216노드 기준 85.7%)가 지배해
    # subcortex가 덜 fit된다. 아래는 FC loss(EIB update·선택, Part3 gradient)에서
    # 각 블록이 차지할 "점유율"(합=1로 정규화; 상대비만 의미). 실제 per-edge 가중은
    # data_loader가 atlas의 블록별 edge수로 나눠 계산 → atlas 크기 무관(216/416 동일).
    # subcortex 라벨 구분 없는 atlas(mouse CHA 등)는 자동 uniform=off.
    # balanced 기본: cortex 0.50 / cross(ctx↔sub) 0.40 / sub-sub 0.10.
    #   - 점유율 모두 같게(=edge수 비례 복원) 두면 whole-brain과 동일.
    #   - cross가 subcortex FC 주력(edge 많아 안정). sub-sub은 적어 noisy → 낮게.
    fc_block_share_cortex: float = 0.50
    fc_block_share_cross:  float = 0.40
    fc_block_share_subsub: float = 0.10

    # ── Block-split corr loss (cc / cross / sub-sub) ─────────────
    # whole-matrix corr는 edge수가 많은 cortex-cortex(Schaefer200+PD25 기준
    # cc 19900 / cross 5000 / ss 300)에 지배됨. corr 항을 블록별로 쪼개
    # 각 블록을 edge수와 무관하게 동등 가중 → subcortex fit 강제 균형.
    # 아래 비율은 합=1로 정규화(상대비만 의미). subcortex 없는 atlas(mouse CHA)는
    # cross/ss 블록 비어 자동으로 whole corr(cc=full)로 fallback.
    #   - ss는 edge 적어 noisy → cc/cross보다 낮게 두는 게 안전.
    corr_block_weight_cc:     float = 0.40   # cortex-cortex corr
    corr_block_weight_cross:  float = 0.40   # cross (ctx↔sub) corr
    corr_block_weight_subsub: float = 0.20   # sub-sub corr

    pd_fit_region_indices: Optional[Sequence[int]] = None
    pd_fit_region_labels:  Optional[Sequence[str]] = None

    # ── Part 4 — DBS ─────────────────────────────────────────
    # 출력물 선택: LFP 원시 시계열은 조합당 27MB(sweep 240조합 = 6.6GB) 라 기본 off,
    # BOLD 시계열(TR 해상도, 조합당 ~0.8MB)을 대신 저장한다. PSD/beta 분석은 LFP 저장과
    # 무관하게 계속 나온다(메모리 상에서 계산).
    dbs_save_lfp_timeseries:         bool  = False
    dbs_save_neural_csv:             bool  = True   # 자극 타깃 노드 S_e/S_i 전구간 CSV
    dbs_neural_csv_stride:           int   = 4      # dt1ms 기준 4 → 250 Hz
    dbs_save_bold_timeseries:        bool  = True
    dbs_pulse_amplitude:             float = 1.0
    dbs_stimulation_frequency_hz:    float = 130.0
    dbs_phase_duration_steps:        int   = 1
    dbs_pre_stimulation_duration_ms: float = 720_000.0
    dbs_stimulation_duration_ms:     float = 60_000.0
    dbs_fc_pre_transient_skip_ms:    float = 60_000.0   # pre-stim FC에서 버릴 앞부분 transient
    dbs_target_regions: dict = field(default_factory=lambda: {
        "STN_L": 11,
        "GPe_L": 5,
        "GPe_R": 6,
        "GPi_L": 7,
    })
    dbs_waveform_segment_seconds: float = 5.0
    dbs_psd_max_frequency_hz:     float = 100.0
    dbs_beta_band_low_hz:         float = 13.0
    dbs_beta_band_high_hz:        float = 30.0
    dbs_output_base_dir:          str   = "./dbs_analysis"
    dbs_stimulus_modes: tuple = field(default_factory=lambda: ("true_p_t",))

    # === Patch 3: GPU/batching toggles (default OFF preserves prior behavior) ===
    gpu_batch_size: int = 1
    posthoc_parallel: bool = False
    dbs_parallel_targets: bool = False

    def get_baseline_settle_duration_ms(self) -> int:
        if self.baseline_settle_duration_ms is not None:
            return int(self.baseline_settle_duration_ms)
        return int(self.optimizer_bold_window_tr * self.bold_repetition_time_ms)

    def print_summary(self) -> None:
        print("=" * 56)
        print("  Config")
        print("=" * 56)
        for key, value in self.__dict__.items():
            print(f"  {key:<44} = {value}")
        print("=" * 56)
