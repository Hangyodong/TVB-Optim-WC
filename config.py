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
    cache_root_dir: str = "./cache"
    param_save_dir: str = "./optimized_params"
    cache_version:  str = "v_eituning_oldlogic_match_p3_p7_p8_p9_pm_p12_p13_p14"

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
    bold_hrf_duration_ms : float = 32_000.0   # ms (32s for mouse)
    tract_conduction_speed:  float = 3.0
    additive_noise_sigma:    float = 0.01
    bundle_rng_seed:         int   = 42

    # ── FC 시각화 공통 ────────────────────────────────────────
    fc_plot_vmin: float = -1.0
    fc_plot_vmax: float =  1.0

    # ── Part 1 — FIC ─────────────────────────────────────────
    fic_target_firing_rate_hz:   float = 4.0
    fic_learning_rate:           float = 1e-3
    fic_max_iterations:          int   = 2000
    fic_early_stop_patience:     int   = 500
    fic_early_stop_tolerance_hz: float = 0.10
    fic_step_duration_ms:        int   = 1_000
    fic_step_skip_tr:            int   = 0

    # ── Part 2 — EIB ─────────────────────────────────────────
    eib_max_iterations:             int   = 8000
    eib_internal_fic_learning_rate: float = 0.05
    eib_max_weight_learning_rate:   float = 0.002
    eib_bold_window_samples:        int   = 150
    eib_snapshot_save_interval:     int   = 50
    connectivity_weight_max:        float = 1.5
    eib_posthoc_top_k:              int   = 10
    eib_posthoc_duration_ms:        int   = 300_000
    eib_posthoc_skip_tr:            int   = 20

    # ── Part 3 — Full-matrix gradient ────────────────────────
    optimizer_learning_rate:  float = 0.002
    optimizer_max_steps:      int   = 200
    optimizer_chunk_steps:    int   = 5
    optimizer_bold_window_tr: int   = 96
    optimizer_bold_skip_tr:   int   = 8

    # ── Part 3B — Low-rank gradient ──────────────────────────
    lowrank_rank:            int   = 6
    lowrank_max_steps:       int   = 120
    lowrank_learning_rate:   float = 0.002
    lowrank_bold_window_tr:  int   = 96
    lowrank_bold_skip_tr:    int   = 8
    lowrank_delta_scale:     float = 0.15
    lowrank_factor_init:     float = 0.01
    lowrank_activity_weight: float = 0.01
    lowrank_factor_penalty:  float = 1e-4
    lowrank_seed:            int   = 17

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
    lowrank_global_corr_weight:     float = 0.4
    lowrank_nodewise_corr_weight:   float = 0.4
    lowrank_rmse_weight:            float = 0.2

    pd_fit_region_indices: Optional[Sequence[int]] = None
    pd_fit_region_labels:  Optional[Sequence[str]] = None

    # ── Part 4 — DBS ─────────────────────────────────────────
    dbs_pulse_amplitude:             float = 1.0
    dbs_stimulation_frequency_hz:    float = 130.0
    dbs_phase_duration_steps:        int   = 1
    dbs_pre_stimulation_duration_ms: float = 60_000.0
    dbs_stimulation_duration_ms:     float = 60_000.0
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
