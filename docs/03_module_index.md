# 03 — Module Index

Quick reference for every public-ish symbol you'd actually call or reach for.
All signatures are read directly from source; no execution.

---

## `config.py`

### `class Config` (`@dataclass`)

One dataclass, ~50 fields, grouped by stage. Highlights:

**Paths**
- `region_txt = "Atlas_43.txt"`, `sc_csv = "weight.csv"`, `length_csv = "tract_length.csv"`, `fc_csv = "FC_compact.csv"`
- `cache_root_dir = "./cache"`, `param_save_dir = "./optimized_params"`
- `cache_version = "v_eituning_oldlogic_match"` ← bump this to invalidate caches.

**Simulation**
- `integration_dt_ms = 1.0`, `warmup_duration_ms = 300_000`
- `bold_repetition_time_ms = 1000.0`, `tract_conduction_speed = 3.0`
  (notebook overrides to `1.0` → larger delays)
- `additive_noise_sigma = 0.01`, `bundle_rng_seed = 42`

**Part 1 (FIC)** — `fic_target_firing_rate_hz=4.0`, `fic_max_iterations=2000`, `fic_step_duration_ms=1_000` (1 s step), early-stop after 500 consecutive `<0.10 Hz` errors.

**Part 2 (EIB)** — `eib_max_iterations=8000`, rolling `eib_bold_window_samples=150`, snapshot every 50 steps, post-hoc top-k=10 × `eib_posthoc_duration_ms=300_000` ms (5 min) each.

**Part 3 (full grad)** — `optimizer_learning_rate=0.002`, `optimizer_max_steps=200` default (notebook bumps to **1000**), `optimizer_chunk_steps=5` (notebook: 1), `optimizer_bold_window_tr=96` (notebook: 180), skip 8 TR (notebook: 30).

**Part 3B (low-rank)** — `lowrank_rank=6`, `lowrank_delta_scale=0.15`, `lowrank_factor_init=0.01`, `lowrank_seed=17`, `lowrank_max_steps=120` (notebook: 150).

**EIB scoring weights** — `full_brain_fc_loss_weight=0.35`, `pd_fit_block_loss_weight=0.65`, `correlation_loss_weight=0.70`, `rmse_loss_weight=0.30`. Notebook overrides to (1.00, 0.00, 0.80, 0.20).

**Part 4 (DBS)** — `dbs_pulse_amplitude=1.0` (notebook: **10.0**), `dbs_stimulation_frequency_hz=130.0`, `dbs_phase_duration_steps=1` (1 ms phase), 60 s pre + 60 s during. Targets: `{STN_L:11, GPe_L:5, GPe_R:6, GPi_L:7}`. Modes: `("true_p_t",)` only.

### Methods
- `get_baseline_settle_duration_ms() → int` — returns the override if set, else `optimizer_bold_window_tr * bold_repetition_time_ms`.
- `print_summary()` — prints all fields.

---

## `data_loader.py`

### `load_data(cfg) → dict`

Returns:
```python
{
  "weights":       (42, 42) float32, log1p+max-normalised, masked by SC,
  "lengths":       (42, 42) float32 mm,
  "delays":        (42, 42) float32 ms  = lengths / cfg.tract_conduction_speed,
  "fc_target":     (42, 42) float32,
  "sc_mask":       (42, 42) float32 ∈ {0,1},
  "region_labels": list[str] length 42,
  "n_nodes":       42,
  "cache_tag":     "v_..._N42_sc<10hex>_fc<10hex>",
  "cache_dir":     "./cache/<cache_tag>",
  "graph":         DenseDelayGraph,
}
```

Side effects: creates cache dir, calls `tvboptim.utils.set_cache_path(...)`,
opens a matplotlib figure (3 panels) and `plt.show()`s.

### Internal helpers (underscored)

`_resolve_path`, `_load_region_labels`, `_load_matrices`, `_validate_shapes`,
`_cast_to_float32`, `_build_cache_tag` (SHA-1 over top-left 8×8 block of SC and FC),
`_create_cache_dir`, `_plot_data_matrices`.

---

## `model.py`

### `class WilsonCowanEIB(AbstractDynamics)`

- `STATE_NAMES = ("E", "I")`, `INITIAL_STATE = (0.2, 0.1)`
- `AUXILIARY_NAMES = ("S_e", "S_i", "rE_hz", "rI_hz")`
- `DEFAULT_PARAMS` (Bunch): includes `c_ei=6.0`, `rE_max_hz=20.0`, `rI_max_hz=20.0`, sigmoid clip ±500.
- `dynamics(time_ms, state, params, coupling, external)` →
  `(dE/dt, dI/dt)` and auxiliary `(S_e, S_i, rE_hz, rI_hz)`.

### `class EIBLinearCoupling(InstantaneousCoupling)`

- `N_OUTPUT_STATES = 2`, `DEFAULT_PARAMS = Bunch(wLRE=1.0, wFFI=1.0)`.
- `pre()` stacks two channels = `source_E * wLRE` and `source_E * wFFI`.
- `post()` is identity on summed inputs.

### `build_network(cfg, data) → (network, initial_state, bold_monitor, warmup_result)`

- Uses `data["graph"]` if present, otherwise constructs `DenseGraph(weights, region_labels=...)`.
- Wires `AdditiveNoise(sigma=cfg.additive_noise_sigma, apply_to="E")` and `BoundedSolver(Heun(), 0, 1)`.
- Compiles via `prepare(network, solver, t1=cfg.warmup_duration_ms, dt=cfg.integration_dt_ms)`.
- Runs the warmup, calls `network.update_history(warmup_result)` if available, then builds `Bold(period=cfg.bold_repetition_time_ms, downsample_period=4.0, voi=0, history=warmup_result)`.

---

## `pipeline_contracts.py`

### `@dataclass ParamSet`

Fields: `c_ei: (n_nodes,) float32`, `wLRE: (n_nodes,n_nodes) float32`, `wFFI: (n_nodes,n_nodes) float32`, `c_ei_frozen: bool`.

Methods:
- `freeze_c_ei() → ParamSet`
- `update_c_ei(delta) → ParamSet` (no-op if frozen; clips to [0, 20])
- `update_weights(new_wLRE, new_wFFI, sc_mask, w_max) → ParamSet`
- `sanitize(sc_mask, w_max)` — NaN/inf scrub, clip, mask, symmetrize.
- `to_jax() → (jnp, jnp, jnp)`, `to_numpy_dict()`, `from_numpy_dict()`
- `ParamSet.default(n_nodes, c_ei_init=6.0)`

### `class StateBundle`

Constructor takes `params`, `init_dynamics`, `bold_history`, `bold_window`, `internal_state`, `delay_history`, `stage`, `metadata`.

Properties: `params`, `init_dynamics`, `bold_history`, `bold_window`, `internal_state`, `noise_state` (= `internal_state["noise_samples"]` if present), `delay_history`, `stage`, `metadata`.

**Legacy views** (so old TVB-state code keeps working):
- `bundle.dynamics.c_ei`
- `bundle.coupling.coupling.{wLRE,wFFI}`
- `bundle.initial_state.dynamics`

**Mutation (immutable)**: `advance(...)`, `with_params(...)`, `with_metadata(...)`.

**Apply to running sim**:
- `apply_to_network(network)` — restores delay history.
- `apply_to_state(tvb_state)` — writes params, init dynamics, internal state into a TVB state object.
- `to_tvb_state(network, solver, t1, dt) → (compiled_model, tvb_state)`
- `build_bold_monitor(cfg) → Bold` — restores BOLD history via `eqx.tree_at`.

**Window/FC seed**: `get_fc_seed_window(n_samples, n_nodes) → (n_samples, n_nodes)` float32.

**Serialization**: `to_dict()` / `from_dict()` (used by the disk cache). `from_warmup(...)` builds a stage="warmup" bundle from a freshly-warmed network.

**Fingerprint**: SHA-1 of `(c_ei, wLRE, wFFI, init_dynamics, bold_window, bold_history[:8], internal_state heads, delay_history.ravel()[:128], stage)`. Used as part of every cache key.

### Module helpers

- `capture_internal_state(tvb_state) → dict | None`
- `restore_internal_state(tvb_state, internal_state)`
- `advance_internal_state(tvb_state, metadata) → (internal_state, metadata)` — splits `metadata["rng_key"]` forward and rewrites `internal.noise_samples`.
- `update_bold_history(monitor, sim_result)` — rolls history and writes new BOLD samples.
- `extract_bold_window(bold_output) → (T, N) float32`
- `capture_network_delay_history(network) → np.ndarray | None`
- `restore_network_delay_history(network, delay_history)`
- `sync_network_delay_history(network, sim_result) → np.ndarray | None`
- `resolve_pd_fit_indices(cfg, data) → np.ndarray[int32]` — picks the PD-fit
  block by label/index/count (first `cfg.pd_fit_region_count` regions by default).
- `build_bundle_from_legacy_state(state, cfg, data, stage, ...)` — adapter for callers that still pass raw TVB state.

---

## `part1_fic.py`

### `run_fic(network, bundle_in=None, initial_state=None, bold_monitor=None, warmup_result=None, cfg=None, data=None) → StateBundle`

Two call shapes:
1. `run_fic(network, bundle_in=StateBundle, cfg=cfg, data=data)` ← notebook uses this.
2. Legacy: `run_fic(network, initial_state=..., bold_monitor=..., warmup_result=..., cfg=cfg, data=data)`.

Cache key: `fic_<cache_tag>_rE<target>_eta<lr>_steps<max>_dur<step_ms>_skip<tr>_fp<bundle.fingerprint>`.

Adds to `bundle.metadata`: `post_fic_fc_matrix`, `post_fic_fc_corr`, `post_fic_fc_rmse`.
Output stage tag: `"fic"`. `c_ei_frozen` is **False** in the returned bundle (matches the "old logic" comment).

### Internal
- `_run_fic_loop_pure(network, init_dict, cfg, data) → dict`
- `_extract_firing_rates_from_bold`, `_extract_firing_rates`
- `_compute_fc_from_bold_output(bold_output, skip_tr)` — NaN-scrub, mean-center, std-normalise, gram-divided-by-(T−1), clip, zero-diag.
- `_compute_bundle_fc_summary(network, bundle, cfg, data, sim_duration_ms?, skip_tr?)` — re-simulates and returns `(fc, corr, rmse)`.
- `_plot_fic_results`, `_analyze_beta_oscillation_fic`, `_compute_beta_summary_from_neural`, `_plot_beta_summary`, `_compute_normalized_psd`, `_compute_beta_power_ratio`.

(`_compute_beta_*`, `_compute_normalized_psd`, `_plot_beta_summary` are re-imported by `part2_eib` and `part3_gradient`.)

---

## `part2_eib.py`

### `run_eib(network, bundle_in=None, fic_results=None, cfg=None, data=None) → StateBundle`

Cache key: `eib_<cache_tag>_win<W>_etaF<lr>_etaE<lr>_steps<max>_topk<k>_frozen<int>_fp<...>`.

Pipeline inside `_run_eib_loop_pure`:
1. Initialise rolling BOLD buffer of shape `(W, 1, N)` from `bundle_in.get_fc_seed_window`.
2. For each step (1 TR of simulation):
   - Advance one TR, roll BOLD buffer.
   - If `not c_ei_frozen`: do an internal FIC nudge on `c_ei`.
   - Compute `window_fc` from buffer, compute `wLRE/wFFI` update by `_eib_update_rule`.
   - Ramped learning rate: `current_eta = (step/max) * eib_max_weight_learning_rate`.
   - Snapshot every `eib_snapshot_save_interval` steps.
3. Post-hoc validation (`_run_posthoc_validation`): pick top-k snapshots by window-FC corr, re-simulate each for `eib_posthoc_duration_ms`, keep the best by `(correlation_loss_weight, rmse_loss_weight)`-weighted true score.

Returned bundle: stage `"eib"`.

### Internal
- `_eib_update_rule(wLRE, wFFI, fc_pred, fc_target, eta_eib, sc_mask, w_max)` — row-RMSE-weighted ascent on (target − pred) for wLRE, descent for wFFI; `_clip_sym` clips, masks, symmetrizes.
- `_compute_fc_from_buffer` (JAX), `_compute_fc_from_bold_output` (NumPy).
- `_evaluate_candidate_bundle` — runs the post-hoc sim and returns `{bundle, fc_matrix, neural_data}`.
- `posthoc_fc_corr(fc_matrix, fc_target) → float`.
- `_plot_eib_results`, `_analyze_beta_oscillation_eib`.
- `_coerce_eib_bundle` — accepts `StateBundle`, legacy dict (with `"bundle"` key or `"state_fic"`).

---

## `part3_gradient.py`

### `run_gradient_optimization(network, bundle_in=None, eib_results=None, cfg=None, data=None, warmup_result=None) → StateBundle`

Cache key: `grad_<cache_tag>_TR<W>_SKIP<S>_STEPS<N>_LR<lr>_frozen<int>_fp<...>`.

Wraps `c_ei`, `wLRE`, `wFFI` as `BoundedParameter`/`Parameter`. Loss = `(1 − corr(fc, fc_target))` + `0.01 * activity_reg`, where `activity_reg` is the squared deviation of mean E over the last 500 steps from `cfg.fic_target_firing_rate_hz`. Optimizer = `optax.chain(zero_nans, clip_by_global_norm(0.1), adamaxw(lr))`, run in chunks of `optimizer_chunk_steps`. Tracks best-loss params.

Returned bundle: stage `"grad"`.

### `class LowRankTrainable(eqx.Module)` and `run_lowrank_optimization(...)`

Cache key: `grad_lowrank_<cache_tag>_rank<r>_TR<W>_STEPS<N>_LR<lr>_ds<delta_scale>_fp<...>`.

Trainable: `c_ei (N,)`, `lre_u (N,r)`, `lre_v (N,r)`, `ffi_u (N,r)`, `ffi_v (N,r)` (init `N(0, σ²)` with `σ = lowrank_factor_init`, RNG `lowrank_seed`).

Effective weights: `w_eff = clip(w_base + δ · U·Vᵀ, 0, w_max) · sc_mask`, then symmetrise. Loss = `(1−corr) + lowrank_activity_weight·activity + lowrank_factor_penalty·||factors||²`. Same `optax.chain` (`zero_nans → clip_by_global_norm(0.1) → adamaxw`).

Returned bundle: stage `"lowrank"`.

### Shared helpers
- `compute_simulated_fc(network, bundle, cfg, sim_duration_ms=300_000, skip_tr=60) → (N,N) np.ndarray`
- `_simulate_bundle`, `_evaluate_bundle_without_settle`, `_settle_bundle`
- `_coerce_gradient_bundle`, `_coerce_lowrank_bundle` (legacy adapters)
- `_run_full_optimization_loop(...)` — chunked Optax loop with progress prints.
- `_compute_fc_differentiable(bold_output, skip_tr)` (JAX) and `_compute_fc_from_bold_output(...)` (NumPy).
- `_compute_correlation_loss(predicted, target) → 1 − Pearson(predicted, target)` over off-diagonal.
- `_compute_activity_regularization(sim_result, cfg)`.
- `_compute_rmse_metric(predicted, target) → float` over off-diagonal.
- `_plot_gradient_results`, `_analyze_beta_oscillation_gradient`.

---

## `part4_dbs.py`

### `run_dbs_stimulation(network, bundle_in=None, optimized_state=None, cfg=None, data=None) → None`

Also exposed as `run_dbs` (alias). For each `(label, node_index)` in `cfg.dbs_target_regions`:

1. `_compute_derived_parameters(cfg)` derives biphasic period (forced phase_step=1 → 2-step pulse), gapless period from `dbs_stimulation_frequency_hz`, total pulse count from `dbs_stimulation_duration_ms`.
2. `_build_biphasic_pulse_train(...)` builds `(total_steps, N)` float32 of ±amplitude only at the target node.
3. `_make_stimulated_dynamics(...)` returns a replacement `dynamics` method; `network.dynamics.dynamics = types.MethodType(stimulated_fn, network.dynamics)` (always restored in `finally`).
4. Compiles and simulates `pre_steps + stim_steps`.
5. `_analyze_and_plot` writes 5 PNGs + `lfp_timeseries.csv` + `psd_pre_vs_during.csv` under
   `<cfg.dbs_output_base_dir>/<target_label>/<stimulus_mode>/<observable_name>/`.

Only `stimulus_mode = "true_p_t"` is exercised. The function is hard-coded to use
`E + I` as the LFP observable.

### Internal
- `_resolve_observable_signal` — supports `"E"`, `"I"`, `"E_plus_I"`, `"E_minus_I"`.
- `_compute_psd` — Welch, hann, nperseg = fs (clamped ≥1000), 50 % overlap, scaling `"spectrum"` (so units are V², not V²/Hz).
- `_compute_beta_power_ratio(frequencies, psd, beta_low, beta_high, total_band_low=1.0)`.
- Plots: `_plot_lfp_timeseries`, `_plot_stim_waveform_full`, `_plot_stim_waveform_zoomed`, `_plot_psd_comparison`, `_plot_lfp_segment_comparison`.
