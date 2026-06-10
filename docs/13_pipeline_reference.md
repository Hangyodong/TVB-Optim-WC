# 13 — Pipeline Reference

## 1. Overview

This project tunes a two-population (E/I) Wilson-Cowan whole-brain network so its
simulated BOLD functional connectivity (FC) matches an empirical target FC, then
uses the tuned model to study deep-brain-stimulation (DBS) effects on beta-band
power. The pipeline runs four stages — FIC → EIB → gradient (full + low-rank) →
DBS — passing state between them through a single `StateBundle`/`ParamSet`
contract. The default mouse atlas is 42 regions (`Atlas_43.txt`); the notebook is
currently set to the `human` dataset (Schaefer400 + 25 subcortex = 425 nodes).

## 2. Repository file map

| filename | size hint | role |
|---|---|---|
| `config.py` | ~7.8 KB | All hyperparameters in one `Config` dataclass. |
| `data_loader.py` | ~8.1 KB | Load + preprocess SC/length/FC; build `DenseDelayGraph`; cache tag. |
| `model.py` | ~6.8 KB | `WilsonCowanEIB` dynamics, `EIBLinearCoupling`, `build_network` + warmup. |
| `pipeline_contracts.py` | ~26 KB | `ParamSet`, `StateBundle`, internal/delay state capture, FC helpers. |
| `part1_fic.py` | ~20 KB | Part 1 FIC: per-node `c_ei` tuning to target firing rate. |
| `part2_eib.py` | ~24 KB | ★ Part 2 EIB: rolling-window FC tuning of `wLRE`/`wFFI` (+`c_ei`) + post-hoc validation. |
| `part3_gradient.py` | ~38 KB | Part 3A full-matrix gradient + Part 3B low-rank gradient. |
| `part4_dbs.py` | ~28 KB | Part 4 biphasic DBS injection, pre-vs-during PSD/beta analysis. |
| `main.ipynb` | ~35 KB | 24-cell driver notebook (dataset switch, run all parts, figure export). |
| `main.py` / `main_mouse.py` / `main_human.py` | ~18 KB each | Script equivalents of the notebook (not in scope of this read). |
| `timing_utils.py` | ~2.7 KB | Read-only runtime estimator used by Cell 11. |
| `human/weight.csv` | — (runtime only) | Human structural connectivity matrix. |
| `human/tract_length.csv` | — (runtime only) | Human tract length matrix (→ delays). |
| `human/fc_matrix.csv` | — (runtime only) | Human target FC matrix. |
| `human/Custom_Schaefer400_PD25subcortex_1mm.txt` | — (runtime only) | Human region labels. |
| `mouse/weight_nor.csv` | — (runtime only) | Mouse SC matrix. |
| `mouse/tract_length_nor.csv` | — (runtime only) | Mouse tract length matrix. |
| `mouse/FC_nor.csv` | — (runtime only) | Mouse target FC matrix. |
| `mouse/Atlas_43.txt` | — (runtime only) | Mouse region labels (42 non-empty lines). |
| `cache/<cache_tag>/` | generated | Per-stage cached run dicts (`fic_`, `eib_`, `grad_`, `grad_lowrank_`). |
| `dbs_analysis/<target>/<mode>/E_plus_I/` | generated | DBS PNG plots + `lfp_timeseries.csv`, `psd_pre_vs_during.csv`. |
| `optimized_params/` | generated | `param_save_dir` (declared in config; not written by the read files). |
| `all_figures_export_*/`, `paper_figures_*/` | generated | Notebook Cell 22/23 figure exports. |

## 3. Dependency stack

| package | use |
|---|---|
| `jax` / `jax.numpy` | All numerics, JIT, autodiff, PRNG. `jax_enable_x64=False`. |
| `equinox` (`eqx`) | `eqx.Module`, `eqx.tree_at`, `eqx.filter_jit`, `eqx.filter_value_and_grad`. |
| `optax` | `adamaxw`, `clip_by_global_norm`, `zero_nans`, `chain`. |
| `numpy` | Host-side arrays, serialization, fingerprints. |
| `pandas` | CSV read (`_load_matrices`); DBS CSV output. |
| `scipy` | `scipy.signal.welch`, `scipy.integrate.trapezoid` (PSD / beta ratio). |
| `matplotlib` | All plotting. |
| `tvboptim` | In-house TVB-optimization library (not on PyPI). |

`tvboptim` submodule import paths actually used:

```python
tvboptim.experimental.network_dynamics            # Network, prepare
tvboptim.experimental.network_dynamics.graph      # DenseDelayGraph, DenseGraph
tvboptim.experimental.network_dynamics.core.bunch # Bunch
tvboptim.experimental.network_dynamics.coupling.base   # InstantaneousCoupling
tvboptim.experimental.network_dynamics.dynamics.base   # AbstractDynamics
tvboptim.experimental.network_dynamics.noise      # AdditiveNoise
tvboptim.experimental.network_dynamics.solvers    # BoundedSolver, Heun
tvboptim.observations.tvb_monitors.bold           # Bold, LotkaVolterraHRFKernel
tvboptim.observations.observation                 # fc_corr, rmse
tvboptim.optim.optax                              # OptaxOptimizer
tvboptim.types                                    # BoundedParameter, Parameter
tvboptim.utils                                    # cache, set_cache_path
```

## 4. Config reference (config.py)

Notebook override column reflects the active `dataset="human"` values from Cell 4
(`_DATASET_PARAMS["human"]`) and Cell 5 (`Config(...)`). "—" means unchanged.

### Paths

| field | default | nb override | description |
|---|---|---|---|
| `region_txt` | `Atlas_43.txt` | `human/Custom_Schaefer400_PD25subcortex_1mm.txt` | Region label file. |
| `sc_csv` | `weight.csv` | `human/weight.csv` | Structural connectivity CSV. |
| `length_csv` | `tract_length.csv` | `human/tract_length.csv` | Tract length CSV. |
| `fc_csv` | `FC_compact.csv` | `human/fc_matrix.csv` | Target FC CSV. |
| `cache_root_dir` | `./cache` | — | Cache root (subfolder per cache_tag). |
| `param_save_dir` | `./optimized_params` | — | Param save dir. |
| `cache_version` | `v_eituning_oldlogic_match_..._p31` | — | Cache-busting version string. |

### Simulation (shared)

| field | default | nb override | description |
|---|---|---|---|
| `integration_dt_ms` | `1.0` | — | Heun step size (ms). |
| `warmup_duration_ms` | `300_000` | `720_000` | Warmup sim length (ms). |
| `bold_repetition_time_ms` | `1000.0` | — | BOLD TR / sample period (ms). |
| `tract_conduction_speed` | `3.0` | `1.0` (human) | mm/ms; `delays = lengths / speed`. |
| `additive_noise_sigma` | `0.01` | `0.01` (human) / `0.02` (mouse) | E-population additive noise σ. |
| `bundle_rng_seed` | `42` | — | Default noise-resume seed. |
| `bold_hrf_k1` | `5.6` | — | HRF k₁. |
| `bold_hrf_V0` | `0.02` | — | HRF resting blood volume. |
| `bold_hrf_tau_s` | `0.8` | — | HRF signal-decay τ (s). |
| `bold_hrf_tau_f` | `0.4` | — | HRF flow τ (s). |
| `bold_hrf_scaling` | `1/3` | — | HRF kernel scaling. |
| `bold_hrf_duration_ms` | `20_000.0` | `20_000` (human) / `32_000` (mouse) | HRF kernel length (ms). |
| `wc_*` (24 WC params) | mouse values | dataset-specific (see Cell 4) | Wilson-Cowan dynamics params. |
| `wc_c_ei_init` | `10.0` | `6.0` (human) / `10.0` (mouse) | Initial per-node `c_ei`. |
| `fc_plot_vmin` / `fc_plot_vmax` | `-1.0` / `1.0` | — | FC colormap limits. |

### Part 1 — FIC

| field | default | nb override | description |
|---|---|---|---|
| `fic_target_firing_rate_hz` | `4.0` | `4.0` | Target excitatory rate (Hz). |
| `fic_learning_rate` | `1e-3` | `1e-3` | FIC `c_ei` learning rate. |
| `fic_max_iterations` | `2000` | `2000` | Max FIC steps. |
| `fic_early_stop_patience` | `500` | `500` | Consecutive in-tol steps to stop. |
| `fic_early_stop_tolerance_hz` | `0.10` | — | Convergence tolerance (Hz). |
| `fic_step_duration_ms` | `1_000` | `1_000` | Sim length per FIC step (ms). |
| `fic_step_skip_tr` | `0` | `0` | TR dropped before averaging. |

### Part 2 — EIB

| field | default | nb override | description |
|---|---|---|---|
| `eib_max_iterations` | `8000` | `8000` | Search steps (1 TR each). |
| `eib_internal_fic_learning_rate` | `0.05` | `0.05` | In-loop `c_ei` (FIC) LR. |
| `eib_max_weight_learning_rate` | `0.002` | `0.002` | Peak weight LR (ramped). |
| `eib_bold_window_samples` | `150` | `720` | Rolling FC window length (TR). |
| `eib_update_interval` | `1` | `1` | Update wLRE/wFFI every N TR. |
| `eib_snapshot_save_interval` | `50` | `50` | Snapshot/print cadence. |
| `connectivity_weight_max` | `1.5` | `1.5` | Weight clip ceiling `w_max`. |
| `eib_posthoc_duration_ms` | `720_000` | `720_000` | Post-hoc validation sim length. |
| `eib_posthoc_skip_tr` | `20` | `60` | Post-hoc TR skip. |

### Part 3 — Full Gradient

| field | default | nb override | description |
|---|---|---|---|
| `optimizer_learning_rate` | `0.002` | `0.0005` | adamaxw LR. |
| `optimizer_max_steps` | `200` | `1000` | Total grad steps. |
| `optimizer_chunk_steps` | `5` | `10` | Steps per optimizer.run chunk. |
| `optimizer_bold_window_tr` | `720` | `720` | Sim window per loss eval (TR). |
| `optimizer_bold_skip_tr` | `8` | `60` | TR skipped before FC. |

### Part 3B — Low-rank

| field | default | nb override | description |
|---|---|---|---|
| `lowrank_rank` | `6` | `6` | Rank of weight correction. |
| `lowrank_max_steps` | `120` | `300` | Low-rank grad steps. |
| `lowrank_learning_rate` | `0.002` | `0.0002` | adamaxw LR. |
| `lowrank_bold_window_tr` | `96` | `720` | Sim window per loss eval (TR). |
| `lowrank_bold_skip_tr` | `8` | `60` | TR skipped before FC. |
| `lowrank_delta_scale` | `0.15` | `0.15` | Scale of `U Vᵀ` weight delta. |
| `lowrank_factor_init` | `0.01` | `0.01` | Factor init std multiplier. |
| `lowrank_activity_weight` | `0.01` | `0.01` | Activity-reg weight. |
| `lowrank_factor_penalty` | `1e-4` | `1e-4` | L2 penalty on factors. |
| `lowrank_seed` | `17` | `17` | RandomState seed for factors. |
| `baseline_settle_duration_ms` | `0` | `0` | Optional final no-stim settle. |

### EIB scoring weights

| field | default | nb override | description |
|---|---|---|---|
| `pd_fit_region_count` | `14` | `14` | PD-fit block size (fallback). |
| `full_brain_fc_loss_weight` | `0.35` | `1.00` | Whole-brain FC term weight (EIB score). |
| `pd_fit_block_loss_weight` | `0.65` | `0.00` | PD-block term weight (unused → dead). |
| `correlation_loss_weight` | `0.70` | `0.80` | (1−corr) weight in EIB window score. |
| `rmse_loss_weight` | `0.30` | `0.20` | RMSE weight in EIB window score. |
| `optimizer_global_corr_weight` | `0.4` | `0.40` | Grad α: global FC corr. |
| `optimizer_nodewise_corr_weight` | `0.4` | `0.40` | Grad β: node-wise FC corr. |
| `optimizer_rmse_weight` | `0.2` | `0.20` | Grad γ: global FC RMSE. |
| `lowrank_global_corr_weight` | `0.4` | `0.40` | Low-rank α. |
| `lowrank_nodewise_corr_weight` | `0.4` | `0.40` | Low-rank β. |
| `lowrank_rmse_weight` | `0.2` | `0.20` | Low-rank γ. |
| `pd_fit_region_indices` | `None` | — | Explicit PD-fit indices. |
| `pd_fit_region_labels` | `None` | — | Explicit PD-fit labels. |

### Part 4 — DBS

| field | default | nb override | description |
|---|---|---|---|
| `dbs_pulse_amplitude` | `1.0` | `10.0` | Biphasic pulse amplitude. |
| `dbs_stimulation_frequency_hz` | `130.0` | `130.0` | Requested stim frequency. |
| `dbs_phase_duration_steps` | `1` | `1` | Phase width (steps); pulse code fixes 1ms. |
| `dbs_pre_stimulation_duration_ms` | `60_000.0` | `60_000.0` | Pre-stim baseline (ms). |
| `dbs_stimulation_duration_ms` | `60_000.0` | `60_000.0` | Stim duration (ms). |
| `dbs_target_regions` | `{STN_L:11, GPe_L:5, GPe_R:6, GPi_L:7}` | human: `{404,410,411,412}` | Stim target node indices. |
| `dbs_waveform_segment_seconds` | `5.0` | — | Segment-compare window (s). |
| `dbs_psd_max_frequency_hz` | `100.0` | — | PSD x-limit (Hz). |
| `dbs_beta_band_low_hz` / `_high_hz` | `13.0` / `30.0` | — | Beta band. |
| `dbs_output_base_dir` | `./dbs_analysis` | — | DBS output root. |
| `dbs_stimulus_modes` | `("true_p_t",)` | — | Active stim modes. |
| `gpu_batch_size` | `1` | — | Patch 3 GPU toggle (off). |
| `posthoc_parallel` | `False` | — | Patch 3 toggle (off). |
| `dbs_parallel_targets` | `False` | — | Patch 3 toggle (falls back to sequential). |

## 5. Data flow — input files and transforms

### 5.1 Input files

| file | shape | dtype | transform |
|---|---|---|---|
| `weight.csv` (SC) | `(N,N)` | f64→f32 | diag→0, `sc_mask=(w>0)`, `log1p(w+0.5)`, /max, ×mask. |
| `tract_length.csv` | `(N,N)` | f64→f32 | `delays = lengths / tract_conduction_speed`. |
| `fc_matrix.csv` (FC target) | `(N,N)` | f64→f32 | Cast only; used raw as target. |
| `region_txt` | `N` labels | str | Parsed (plain / TSV-`name` / `"<int> label"`). |

### 5.2 `load_data()` pseudocode

```python
weights, lengths, fc_target = read_csv(header=None)         # f64
assert all are (N,N); len(labels)==N
fill_diagonal(weights, 0)
sc_mask = (weights > 0).float32                              # sparse mask
assert sc_mask.sum() > 0
weights = log1p(weights + 0.5)                              # compress dynamic range
weights = weights / max(weights)                            # normalise to [0,1]
weights = weights * sc_mask                                 # re-impose sparsity
delays  = lengths / cfg.tract_conduction_speed              # ms
cast weights, lengths, delays, fc_target -> float32
cache_tag = f"{cache_version}_N{N}_sc{sha1(w[:8,:8])[:10]}_fc{sha1(fc[:8,:8])[:10]}"
cache_dir = cache_root/cache_tag ; set_cache_path(cache_dir)
graph = DenseDelayGraph(weights.f64, delays.f64, region_labels)   # delay-aware
return {weights, lengths, delays, fc_target, sc_mask,
        region_labels, n_nodes, cache_tag, cache_dir, graph}
```

### 5.3 `build_network()` outputs

```python
# WilsonCowanEIB init:
dynamics = WilsonCowanEIB(c_ei = wc_c_ei_init * ones((N,)))   # c_ei per-node
for each wc_* in cfg: dynamics.params[name] = float(cfg.wc_<name>)   # scalar override
# EIBLinearCoupling init:
coupling.params.wLRE = ones((N,N))     # long-range excitation gain
coupling.params.wFFI = ones((N,N))     # feed-forward inhibition gain
noise   = AdditiveNoise(sigma=cfg.additive_noise_sigma, apply_to="E")
solver  = BoundedSolver(Heun(), low=0.0, high=1.0)
compiled, initial_state = prepare(network, solver, t1=warmup_duration_ms, dt=integration_dt_ms)
warmup_result = block_until_ready(compiled(initial_state))   # eager warmup run
bold_monitor  = Bold(period=TR, downsample_period=4.0, voi=0, history=warmup_result)
return network, initial_state, bold_monitor, warmup_result
```

`warmup_result.data` shape = `(warmup_steps, 2, N)` (E,I per node per step);
`warmup_result.data[-1]` (shape `(2,N)`) seeds `StateBundle.init_dynamics`.

## 6. StateBundle / ParamSet contract

### 6.1 ParamSet fields

| field | shape | clip / range |
|---|---|---|
| `c_ei` | `(N,)` | `clip(0, 20)`; NaN→6.0 (`_clean_c_ei`). |
| `wLRE` | `(N,N)` | `clip(0, w_max)` × `sc_mask`, symmetrised `0.5(W+Wᵀ)`. |
| `wFFI` | `(N,N)` | same as wLRE. |
| `c_ei_frozen` | bool | If True, `update_c_ei` is a no-op. |

Methods: `freeze_c_ei`, `update_c_ei(delta)`, `update_weights`, `sanitize`,
`to_jax`, `to_numpy_dict`/`from_numpy_dict`, `default(N, c_ei_init)`.

### 6.2 StateBundle fields

| field | shape | meaning |
|---|---|---|
| `params` | `ParamSet` | c_ei / wLRE / wFFI. |
| `init_dynamics` | `(2,N)` | Neural endpoint seeding next stage. |
| `bold_history` | `(H,1,N)` | BOLD monitor history for HRF resume. |
| `bold_window` | `(W,N)` | Recent emitted BOLD samples for FC seed. |
| `internal_state` | dict of arrays | `_internal` arrays incl. `noise_samples`. |
| `delay_history` | array | Network delay-line snapshot. |
| `stage` | str | Stage tag string. |
| `metadata` | dict | `rng_key`/`rng_seed`, stamped FC matrices, etc. |

### 6.3 Key methods

| method | role |
|---|---|
| `advance(...)` | Immutable copy with selectively replaced fields + merged metadata. |
| `apply_to_network(net)` | Restore delay history into network (`restore_network_delay_history`). |
| `apply_to_state(tvb_state)` | Inject c_ei/wLRE/wFFI/init_dynamics + restore `_internal`. |
| `to_tvb_state(net, solver, t1, dt)` | `prepare(...)` then `apply_to_state`; returns `(model, state)`. |
| `build_bold_monitor(cfg)` | New `Bold` with custom `LotkaVolterraHRFKernel`; grafts `bold_history`. |
| `get_fc_seed_window(n, N)` | Last-`n` BOLD window (pad with zeros if short) to seed rolling buffer. |
| `fingerprint()` | sha1 (12 hex) over params + init_dynamics + window/history/internal/delay + stage. |
| `to_dict`/`from_dict` | Numpy-only (de)serialization for caching. |
| `from_warmup(...)` | Build initial bundle from `warmup_result.data[-1]` + monitor history. |

### 6.4 Cache key structure

```python
# Part 1 FIC (part1_fic.run_fic)
f"fic_{cache_tag}_rE{fic_target_firing_rate_hz:g}"
f"_eta{fic_learning_rate}_steps{fic_max_iterations}"
f"_dur{fic_step_duration_ms}_skip{fic_step_skip_tr}_fp{bundle_init.fingerprint()}"

# Part 2 EIB (part2_eib.run_eib)
f"eib_{cache_tag}_win{eib_bold_window_samples}"
f"_etaF{eib_internal_fic_learning_rate}_etaE{eib_max_weight_learning_rate}"
f"_steps{eib_max_iterations}_frozen{int(c_ei_frozen)}_fp{bundle_fic.fingerprint()}"

# Part 3A full gradient (part3_gradient.run_gradient_optimization)
f"grad_{cache_tag}_TR{optimizer_bold_window_tr}_SKIP{optimizer_bold_skip_tr}"
f"_STEPS{optimizer_max_steps}_LR{optimizer_learning_rate}"
f"_frozen{int(c_ei_frozen)}_fp{bundle_start.fingerprint()}"

# Part 3B low-rank (part3_gradient.run_lowrank_optimization)
f"grad_lowrank_{cache_tag}_rank{lowrank_rank}_TR{lowrank_bold_window_tr}"
f"_STEPS{lowrank_max_steps}_LR{lowrank_learning_rate}"
f"_ds{lowrank_delta_scale}_fp{bundle_start.fingerprint()}"
```
(`.` in float LRs is replaced by `p`.) Part 4 DBS is not cached.

## 7. Pipeline stage-by-stage reference

### 7.1 Warmup

- **Entry:** `build_network(cfg, data)` → `StateBundle.from_warmup(warmup_result, bold_monitor, initial_params, internal_state, delay_history, stage="warmup")`
- **Bundle tag:** (none) → `"warmup"`
- **Hyperparams:** `warmup_duration_ms`, `integration_dt_ms`, `wc_*`, `wc_c_ei_init`, `additive_noise_sigma`, `connectivity_weight_max`.
- **Algorithm:**
  1. Build `DenseDelayGraph` from data (already in `data["graph"]`).
  2. Init `WilsonCowanEIB` with per-node `c_ei`, scalar `wc_*` overrides.
  3. Init `EIBLinearCoupling` wLRE=wFFI=ones, `AdditiveNoise` on E.
  4. `prepare()` and eagerly run warmup with `block_until_ready`.
  5. Read `warmup_result.data[-1]` as `init_dynamics`; capture monitor history.
  6. `initial_params = ParamSet.default(N, c_ei_init=wc_c_ei_init).sanitize(...)`.
- **Cache key:** none (warmup is recomputed each session, fed into downstream fingerprints).

### 7.2 Part 1 — FIC

- **Entry:** `run_fic(network, bundle_in=StateBundle, cfg, data) -> StateBundle`
- **Tag:** `"warmup"` → `"fic"` (`c_ei_frozen=False`)
- **Hyperparams:** `fic_target_firing_rate_hz`, `fic_learning_rate`, `fic_max_iterations`, `fic_early_stop_patience`, `fic_early_stop_tolerance_hz`, `fic_step_duration_ms`, `fic_step_skip_tr`.
- **Algorithm (`_run_fic_loop_pure`):**
  1. Build step model (`t1=fic_step_duration_ms`) + BOLD monitor from bundle.
  2. Each step: simulate 1s, run monitor, average excitatory/inhibitory rates (skip `skip_tr`).
  3. Carry `init_dynamics = step_result.data[-1]`; roll BOLD history; advance noise.
  4. Update rule `c_ei += η·rI·(rE − target)`, clip `[0,20]` (see §8.1).
  5. Track mean E / mean rE history; early-stop after `patience` in-tol steps.
  6. Final post-FIC step; build `fic_params` (still unfrozen) and `bundle_fic`.
  7. Compute pre/post FIC FC summaries; stamp `post_fic_fc_*` into metadata.
- **Output dict keys:** `bundle`, `bold_signal`, `final_rE_hz`, `mean_rE_hz_history`, `mean_E_history`, `pre_fic_neural`, `post_fic_neural`, `pre_fic_fc`, `pre_fic_fc_corr`, `pre_fic_fc_rmse`, `post_fic_fc`, `post_fic_fc_corr`, `post_fic_fc_rmse`.
- **Cache key:** `fic_...` (§6.4).

### 7.3 Part 2 — EIB ★ (core stage)

- **Entry:** `run_eib(network, bundle_in=StateBundle, cfg, data) -> StateBundle`
- **Tag:** `"fic"` → `"eib"` (post-hoc) / `"eib_search"` during loop
- **Hyperparams:** `eib_max_iterations`, `eib_bold_window_samples`, `eib_internal_fic_learning_rate`, `eib_max_weight_learning_rate`, `eib_update_interval`, `eib_snapshot_save_interval`, `connectivity_weight_max`, `eib_posthoc_duration_ms`, `eib_posthoc_skip_tr`, `full_brain_fc_loss_weight`, `correlation_loss_weight`, `rmse_loss_weight`, `fic_target_firing_rate_hz`.
- **Algorithm (`_run_eib_loop_pure`):**
  1. Build 1-TR step model + monitor; seed rolling buffer `(win,1,N)` from bundle window.
  2. Each step: simulate 1 TR, push BOLD vector into rolling buffer (`jnp.roll`).
  3. If `c_ei` unfrozen, apply in-loop FIC update with `eib_internal_fic_learning_rate`.
  4. Compute window FC (`_compute_fc_from_buffer`); skip step if non-finite/zero-var.
  5. Ramp `eta = (step+1)/max_iter · eib_max_weight_learning_rate`; every `eib_update_interval` apply `_eib_update_rule` to wLRE/wFFI (see §8.2).
  6. Score window (`win_score`); track best bundle; snapshot every `snapshot_save_interval`.
  7. Post-hoc validation: pick snapshot with max window-corr, re-simulate from warmup state + that snapshot's params for `eib_posthoc_duration_ms`, compute true FC.
  8. Stamp `post_eib_fc_*` into metadata; return best validated bundle.
- **Output dict keys:** `bundle`, `fc_correlations`, `fc_rmse_values`, `pre_eib_fc`, `pre_eib_corr`, `pre_eib_rmse`, `post_eib_fc`, `post_eib_corr`, `post_eib_rmse`, `best_iteration`, `best_fc_corr`, `best_fc_rmse`, `pre_eib_neural`, `post_eib_neural`.
- **Cache key:** `eib_...` (§6.4).

### 7.4 Part 3A — Full-matrix gradient

- **Entry:** `run_gradient_optimization(network, bundle_in, warmup_bundle=bundle_init, cfg, data) -> StateBundle`
- **Tag:** `"eib"` → `"grad"` (re-started from `grad_warmup_start`, `c_ei_frozen=False`)
- **Hyperparams:** `optimizer_learning_rate`, `optimizer_max_steps`, `optimizer_chunk_steps`, `optimizer_bold_window_tr`, `optimizer_bold_skip_tr`, `optimizer_global_corr_weight`, `optimizer_nodewise_corr_weight`, `optimizer_rmse_weight`, `fic_target_firing_rate_hz`.
- **Algorithm (`_run_full_gradient_pure`):**
  1. Rebuild bundle from warmup state + EIB params; sanitize.
  2. Build compiled model (`t1=optimizer_bold_window_tr·TR`) + monitor.
  3. Wrap `c_ei` as `BoundedParameter(0,20)`, `wLRE`/`wFFI` as `Parameter`.
  4. `compute_loss = α·L_global + β·L_nodewise + γ·L_rmse + 0.01·activity_reg` (see §8.3).
  5. `OptaxOptimizer` (zero_nans → clip_by_global_norm(0.1) → adamaxw) in chunks of `chunk_steps`.
  6. Track best-loss param snapshot; build `best_params` (sanitized).
  7. Evaluate post-opt FC without settle; stamp `post_grad_fc_*` into metadata.
- **Output dict keys:** `bundle`, `loss_history`, `pre_opt_fc`, `post_opt_fc`, `post_opt_neural`.
- **Cache key:** `grad_...` (§6.4).

### 7.5 Part 3B — Low-rank gradient

- **Entry:** `run_lowrank_optimization(network, bundle_in=bundle_grad, warmup_bundle=bundle_init, cfg, data) -> StateBundle`
- **Tag:** `"grad"` → `"lowrank"` (re-started from `lowrank_warmup_start`)
- **Hyperparams:** `lowrank_rank`, `lowrank_max_steps`, `lowrank_learning_rate`, `lowrank_bold_window_tr`, `lowrank_bold_skip_tr`, `lowrank_delta_scale`, `lowrank_factor_init`, `lowrank_activity_weight`, `lowrank_factor_penalty`, `lowrank_seed`, `lowrank_global_corr_weight`, `lowrank_nodewise_corr_weight`, `lowrank_rmse_weight`.
- **Algorithm (`_run_lowrank_pure`):**
  1. Take gradient-stage params as base (`wLRE_base`, `wFFI_base`, `c_ei_base`).
  2. Init `LowRankTrainable` factors `U,V` of shape `(N,rank)` (×`factor_init`), seeded RandomState.
  3. `_reconstruct_weights`: `W_eff = clip(W_base + delta_scale·UVᵀ, 0, w_max)·mask`, symmetrise.
  4. Loss = α·L_global + β·L_nodewise + γ·L_rmse + `activity_weight`·act + `factor_penalty`·‖factors‖².
  5. `eqx.filter_jit` step with adamaxw chain; track best loss.
  6. Reconstruct best weights + clipped `c_ei`; build `best_params` (sanitized).
  7. Evaluate post-opt FC without settle.
- **Output dict keys:** `bundle`, `loss_history`, `pre_opt_fc`, `post_opt_fc`, `post_opt_neural`.
- **Cache key:** `grad_lowrank_...` (§6.4).

### 7.6 Part 4 — DBS stimulation

- **Entry:** `run_dbs_stimulation(network, bundle_in=bundle_for_dbs, cfg, data) -> None`
- **Tag:** consumes `"grad"`/`"lowrank"` bundle; no output bundle (writes files).
- **Hyperparams:** `dbs_target_regions`, `dbs_pulse_amplitude`, `dbs_stimulation_frequency_hz`, `dbs_phase_duration_steps`, `dbs_pre_stimulation_duration_ms`, `dbs_stimulation_duration_ms`, `dbs_psd_max_frequency_hz`, `dbs_beta_band_low_hz`/`_high_hz`, `dbs_waveform_segment_seconds`, `dbs_output_base_dir`.
- **Algorithm:**
  1. `_compute_derived_parameters`: phase=1ms, biphasic=2 steps, `period_steps=round(1000/(freq·dt))`, `gap=period−biphasic`, `n_pulses=stim_dur/period_ms`.
  2. Pre-build biphasic pulse-train arrays `(total_steps, N)` per target (anodic +A then cathodic −A).
  3. Per target: monkey-patch `network.dynamics.dynamics` with stimulated closure (injects stim into E input for `true_p_t` mode); `prepare` + apply bundle; run.
  4. `finally` restores `original_dynamics`.
  5. Split LFP (E+I at target) into pre/during masks; Welch PSD (`scaling="spectrum"`).
  6. Compute beta-band ratio pre vs during; emit 5 PNG plots + 2 CSVs per target/mode.
- **Output dict keys:** none (side-effect: files under `dbs_analysis/<target>/true_p_t/E_plus_I/`).
- **Cache key:** none (recompiles per (target, mode)).

## 8. Loss / scoring definitions

### 8.1 FIC update rule

```python
rate_error   = mean_rE_hz - fic_target_firing_rate_hz      # per node
update_delta = fic_learning_rate * mean_rI_hz * rate_error
c_ei = clip(c_ei + update_delta, 0.0, 20.0)
```
(EIB uses the same shape with `eib_internal_fic_learning_rate`.)

### 8.2 EIB update rule

`_eib_update_rule(wLRE, wFFI, fc_pred, fc_target, eta, sc_mask, w_max)`:

```python
fc_diff   = where(finite, fc_target - fc_pred, 0)
row_rmse  = rmse(fc_target, fc_pred, axis=1)[:, None]       # per-row RMSE
wLRE_new  = clip_sym(wLRE + eta * fc_diff * row_rmse)       # boost excitation where pred < target
wFFI_new  = clip_sym(wFFI - eta * fc_diff * row_rmse)       # inverse for inhibition
# clip_sym: where(finite)→clip(0,None)·sc_mask, then 0.5(W+Wᵀ)   (upper clip removed)
```

Window score used for best-bundle selection:

```python
full_term = correlation_loss_weight * (1 - win_corr) + rmse_loss_weight * win_rmse
win_score = -(full_brain_fc_loss_weight * full_term)        # higher is better
```
(`pd_fit_block_loss_weight` is NOT used here — see §13.)

### 8.3 Gradient loss

```python
total = α·L_global + β·L_nodewise + γ·L_rmse + w_act·activity_reg [+ factor_penalty·‖factors‖²]
```
where (all on off-diagonal, masked):
- `_compute_correlation_loss` = `1 - corr(predFC, targetFC)` over off-diagonal.
- `_compute_nodewise_corr_loss` = `1 - mean_i corr(predFC[i], targetFC[i])`.
- `_compute_rmse_loss` = masked RMSE of `predFC - targetFC`.
- `_compute_activity_regularization` = `mean((rE_max·mean_E[-500:] - fic_target)²)`.
- Full-matrix uses a fixed `+0.01·activity_reg`; low-rank uses `lowrank_activity_weight·act + lowrank_factor_penalty·(mean‖lre_u‖²+‖lre_v‖²+‖ffi_u‖²+‖ffi_v‖²)`.

## 9. Rolling window FC

The EIB search keeps a fixed-length ring buffer of the most recent BOLD samples
and recomputes window FC every step to drive the weight update.

- **Buffer:** `bold_rolling_buffer`, shape `(eib_bold_window_samples, 1, N)`, dtype f32 (jnp). Seeded by `bundle.get_fc_seed_window(eib_bold_window_samples, N)` reshaped to `(win,1,N)`.
- **Slide:** each step `bold_rolling_buffer = jnp.roll(buffer, -1, axis=0).at[-1,0,:].set(bold_vector)` — drops oldest, appends newest emitted BOLD TR.
- **`_compute_fc_from_buffer`:**
  ```python
  ts = buffer[:, 0, :]                     # (win, N)
  ts = nan_to_num(ts)
  ts = ts - mean(ts, axis=0)               # mean-center per node
  ts = ts / maximum(std(ts, axis=0), eps)  # std-normalise
  fc = (tsᵀ @ ts) / max(win-1, 1)          # gram / (T-1)
  fc = clip(fc, -1, 1)
  fc = fc * (1 - eye(N))                    # zero diagonal
  ```
- **Governing config:** `eib_bold_window_samples` (window length / buffer rows), `bold_repetition_time_ms` (TR = sample period; one new BOLD sample per step), `eib_max_iterations` (number of slide steps).

## 10. Tensor shapes cheat sheet

| object | shape | notes |
|---|---|---|
| `sim_result.data` | `(steps, 2, N)` | E,I activity per step (`steps = t1/dt`). |
| `sim_result.auxiliary` | `(steps, 4, N)` | S_e, S_i, rE_hz, rI_hz (if present). |
| `bold_output.ys` | `(n_TR, 1, N)` | Emitted BOLD samples (voi=0). |
| `bold_monitor.history` | `(H, 1, N)` | HRF state history. |
| `bold_rolling_buffer` (EIB) | `(eib_bold_window_samples, 1, N)` | Ring buffer. |
| `ParamSet.c_ei` | `(N,)` | per-node E→I gain. |
| `ParamSet.wLRE` / `wFFI` | `(N, N)` | coupling gains. |
| `StateBundle.init_dynamics` | `(2, N)` | warmup/stage endpoint. |
| `StateBundle.bold_window` | `(W, N)` | FC seed window. |
| `LowRankTrainable.{lre,ffi}_{u,v}` | `(N, rank)` | low-rank factors. |
| `stimulation_array` (DBS) | `(total_steps, N)` | biphasic pulse train. |

(N = 42 mouse / 425 human; values dependent on dataset are — runtime only.)

## 11. Notebook cell map

| cell | type | purpose |
|---|---|---|
| 0 | md | Title / "edit only Cell 2". |
| 1 | code | JAX env vars, `jax_enable_x64=False`, imports. |
| 2 | md | "Cell 2 — Config" header. |
| 3 | md | Dataset-switch explanation. |
| 4 | code | `dataset="human"`; `_DATASET_PARAMS` (human & mouse) → `_p`. |
| 5 | code | Build `cfg = Config(...)` from `_p` + overrides; `cfg.print_summary()`. |
| 6 | md | "Cell 3 — Data Loading". |
| 7 | code | `data = load_data(cfg)`. |
| 8 | md | "Cell 4 — Build Network & Warmup". |
| 9 | code | `build_network`; `ParamSet.default`; `bundle_init = StateBundle.from_warmup`. |
| 10 | md | Timing-diagnostics header. |
| 11 | code | Read-only timing estimate (`timing_utils`, dispatch micro-benchmark). |
| 12 | md | "Cell 5 — Part 1: FIC". |
| 13 | code | `bundle_fic = run_fic(...)`; assert `c_ei` unfrozen. |
| 14 | md | "Cell 6 — Part 2: EIB". |
| 15 | code | `bundle_eib = run_eib(...)`; assert unfrozen. |
| 16 | md | "Cell 7 — Part 3: Full Gradient". |
| 17 | code | `bundle_grad = run_gradient_optimization(..., warmup_bundle=bundle_init)`. |
| 18 | md | "Part 3B: Low-rank". |
| 19 | code | `bundle_lowrank = run_lowrank_optimization(..., warmup_bundle=bundle_init)`. |
| 20 | md | "Cell 8 — Part 4: DBS". |
| 21 | code | Pick grad-vs-lowrank by sim FC corr → `bundle_for_dbs`; `run_dbs_stimulation`. |
| 22 | code | Export all open matplotlib figures → `all_figures_export_<stamp>/` (+zip). |
| 23 | code | Build paper PSD figure from `dbs_analysis/*/psd_pre_vs_during.csv`. |

> **Note (Cell 22 caveat from the spec):** the reference text warns of a cell that
> "re-runs Parts 1→4 end-to-end." In the **current** notebook, Cell 21 already
> runs Part 4 and Cells 22–23 only export figures; there is no end-to-end re-run
> cell in this version. Treat any earlier "Cell 22 re-runs everything" warning as
> applying to a prior notebook layout — running top-to-bottom here executes each
> part once, but a cold cache still makes Cells 13–19 multi-hour.

## 12. Known issues and sharp edges

### 12.1 Colab risks
1. `tvboptim` is not on PyPI — Colab must `pip install` it from private source before Cell 1; the notebook has no install cell.
2. JAX env vars (`XLA_PYTHON_CLIENT_PREALLOCATE`, `..._ALLOCATOR`) must be set before any `import jax`; pre-running `import jax` elsewhere silently negates them.
3. `jax_enable_x64=False` globally — `weights.astype(np.float64)` to `DenseDelayGraph` is downcast to f32 inside JIT; expect no 64-bit precision downstream.
4. Working directory: Colab cwd is `/content/`; data files must be uploaded there (or `os.chdir`) or `pd.read_csv` fails.
5. Disk persistence: `cache/`, `dbs_analysis/`, `paper_*`, `all_figures_export_*` vanish at session end unless Drive is mounted.
6. `fc_csv` mismatch: a missing preferred file silently falls back via `_resolve_path`; a renamed-only FC changes the cache fingerprint (`[:8,:8]` bytes feed `_build_cache_tag`).
7. Long single-cell runtimes vs Colab idle timeout — Part 3/DBS cells can exceed free-tier limits; use Pro+ or a batch script.

### 12.2 Long-runtime traps
- Notebook overrides default config toward heavier compute (`optimizer_max_steps=1000` vs 200, `optimizer_bold_window_tr=720` vs default, `lowrank_max_steps=300`, `eib_bold_window_samples=720`, `warmup_duration_ms=720_000`). Reviewers reading `config.py` defaults underestimate runtime by ~5–10×.
- `run_dbs_stimulation` recompiles per (target, mode): the injected `stimulated_dynamics` closure captures a new `stimulation_jax`, so JAX re-traces each outer-loop iteration — 4 targets × ~120 s simulated time × JIT compile dominates cost.
- `tract_conduction_speed` is `1.0` in the human notebook (vs `3.0` default) → 3× larger delays, 3× delay-line memory at `dt=1.0 ms`.

### 12.3 Bugs / sharp edges (from source reading)
- Atlas filename vs content mismatch: `Atlas_43.txt` has 42 non-empty lines; matrices are 42×42 — consistent, but "43" is misleading.
- `run_fic`/`run_eib` keep `c_ei_frozen=False` despite the `part1_fic.py` docstring claiming `freeze_c_ei()`. Intentional (old-logic continues tuning `c_ei`), but the docstring lies — trust `_run_fic_loop_pure`.
- `pd_fit_block_loss_weight` is wired to scoring config but unused in `_eib_update_rule`; Cell 5 sets it `0.0` → dead weight. A future nonzero value would NOT take effect in the active EIB update.
- `extract_bold_window` (cast-only) and `_compute_fc_from_bold_output` (`nan_to_num`) differ on NaN handling; a future caller of `extract_bold_window` could propagate NaNs into a bundle.
- `_make_stimulated_dynamics` monkey-patches `network.dynamics.dynamics`; a process kill / Ctrl-C between the patch and the `finally` leaves the network stimulated until re-`build_network`.
- Cache fingerprint hashes only the first 8×8 of SC and FC — node permutations that leave the corner intact collide on `cache_tag`.
- `build_network` runs the warmup eagerly (`jax.block_until_ready`) — Cell 9 is the first big compile-and-run.
- `BoundedSolver(Heun(), low=0, high=1)` clamps E,I to `[0,1]`; pushing `fic_target_firing_rate_hz` near `rE_max_hz=20` saturates E at 1.0 and FIC silently fails to converge.
- `run_fic` cache name includes `_fp{bundle_init.fingerprint()}`, which depends on `delay_history`; a change in `capture_network_delay_history` silently staleifies all caches.

### NEW issues found in this read
- **[NEW]** `eib_update_interval` gating uses `step_index % cfg.eib_update_interval == 0` (`part2_eib.py:207`). With the default/override `1` this updates every TR, but for `N>1` the FIRST step (`step_index==0`) updates and the cadence is offset by one; there is no guard for `eib_update_interval==0` (would raise `ZeroDivisionError`).
- **[NEW]** Cell 5 source has an indentation irregularity: `eib_update_interval = 1,` is indented deeper than its sibling kwargs. Harmless to Python (still a kwarg) but easy to misread as nested.
- **[NEW]** `dbs_pulse_amplitude` is `10.0` in the notebook (vs `1.0` default) while E,I are clamped to `[0,1]` by `BoundedSolver`. The stim adds into `excitatory_input` (pre-sigmoid) so it is not directly clipped, but the 10× amplitude vs the default is a large, easily-overlooked override.
- **[NEW]** Cell 11 references `cfg.dbs_baseline_duration_ms` via `getattr(..., None)` — that field does not exist on `Config`; the code correctly falls back to `dbs_pre_stimulation_duration_ms`, but the name suggests a config field that was never added.
- **[NEW]** `timing_utils` is imported only inside Cell 11; it is a required file for a clean top-to-bottom notebook run but is absent from `06_known_errors.md`'s dependency notes.
- **[NEW]** `_extract_firing_rates_from_bold` falls back to `rE_max * mean(E)` when `auxiliary` is missing, but the true sigmoid-based `rE_hz` (auxiliary index 2) is `rE_max·sigmoid` — the fallback over/under-estimates rate whenever `k_e - r_e·E ≠ sigmoid`, so FIC convergence target meaning shifts silently between the aux and no-aux paths.

## 13. Design decisions and open questions

| decision | current behaviour | alternative | comment accuracy |
|---|---|---|---|
| `c_ei` not frozen after FIC | `_run_fic_loop_pure` returns `ParamSet(..., c_ei_frozen=False)`; EIB keeps tuning `c_ei` with `eib_internal_fic_learning_rate`. | Call `freeze_c_ei()` after FIC so only wLRE/wFFI move in EIB. | Module docstring in `part1_fic.py` ("FIC 종료 후 c_ei를 freeze_c_ei()로 동결한다") is **wrong**; the in-loop comment in `part2_eib.py` ("구버전 notebook 로직처럼 FIC 이후에도 c_ei를 계속 조정") is **correct**. |
| `pd_fit_block_loss_weight = 0.0` | PD-fit block term contributes nothing; EIB score = whole-brain term only (`_eib_update_rule` never reads it). | Nonzero weight to bias tuning toward PD subcortex block. | Cell 5 comment frames it as a scoring weight; accurate that it is currently inert, but no comment warns that the update rule ignores it entirely. |
| `full_brain_fc_loss_weight = 1.0` (nb) vs `0.35` default | EIB window score scaled entirely by whole-brain term. | Split weight with PD block (as default 0.35/0.65). | Matches "oldlogic_match" intent; comment-free. |
| EIB upper weight clip removed | `_clip_sym` clips only `clip(0, None)` then ×mask (no `w_max` ceiling), despite `w_max` being passed in. | Re-impose `clip(0, w_max)` as `ParamSet.sanitize`/low-rank do. | Inline comment `# Patch: 상한 제거` (ceiling removed) **accurately** describes the code; note `sanitize()` re-applies the ceiling afterward, so the effective bundle params ARE bounded by `w_max`. |
| Post-hoc validation = single best snapshot | Patch 30: always `argmax(window_corr)` snapshot, exactly 1 re-simulation from warmup state + that snapshot's params. | Re-validate top-K snapshots and pick best true FC (older behaviour). | Comments (`Patch 30`, `Patch 25`) accurately describe the single-resim-from-warmup design. |
| Parts 3/3B restart from warmup state | `warmup_bundle.advance(new_params=<stage params>)` (Patch 26) — neural state reset to step-0 warmup, only params carried. | Continue from predecessor's settled `init_dynamics`. | Comments ("optimize from warmup state (step 0) + EIB/grad-tuned params") accurately describe behaviour. |
| `dbs_parallel_targets` toggle | If True, prints a notice and falls back to sequential (no true batched DBS). | Implement batched multi-target DBS. | Inline message accurately states only Option α (JIT cache reuse) is supported. |
