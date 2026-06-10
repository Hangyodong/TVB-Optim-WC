# 04 — Data Flow

Constants: `N = 42` regions (CSV column/row count is authoritative;
`Atlas_43.txt` contains 42 lines despite the "_43" filename). Everything
downstream is `float32` unless noted.

## Input Files

| File | Shape | dtype on disk | After load |
|---|---|---|---|
| `weight.csv` | 42 × 42, comma-separated, **no header** | numeric | `np.float64` → `np.float32` |
| `tract_length.csv` | 42 × 42, **no header** | numeric (mm) | `np.float64` → `np.float32` |
| `FC_compact.csv` | 42 × 42, **no header**, correlations in `[-1, 1]` | numeric | `np.float64` → `np.float32` |
| `Atlas_43.txt` | 42 lines (`"<int> <label>"`) | utf-8 text | `list[str]` length 42 |

Note: the notebook's Cell 3 sets `fc_csv = "FC.csv"`. `data_loader._resolve_path`
falls back to `FC_compact.csv` if `FC.csv` is missing, so the load still works.

## `load_data(cfg)` Transforms

```
weights_raw   = read_csv(weight.csv)            # (42,42) float64
fill_diagonal(weights_raw, 0.0)
sc_mask       = (weights_raw > 0).astype(f32)   # (42,42) ∈ {0,1}
assert sc_mask.sum() > 0

weights       = log1p(weights_raw + 0.5)        # squashes heavy tail
weights      /= weights.max()                   # ∈ [0, 1]
weights      *= sc_mask                         # re-impose zeros

lengths       = read_csv(tract_length.csv)      # (42,42) mm
delays        = lengths / cfg.tract_conduction_speed   # (42,42) ms
fc_target     = read_csv(FC_compact.csv)        # (42,42) ∈ [-1,1]

# all four cast to float32
graph         = DenseDelayGraph(weights.astype(f64),
                                delays.astype(f64),
                                region_labels=region_labels)
cache_tag     = f"{cfg.cache_version}_N42_sc{sha1(W[:8,:8])[:10]}_fc{sha1(FC[:8,:8])[:10]}"
```

Result `data` keys (shapes):

| Key | Shape | Notes |
|---|---|---|
| `weights` | `(42,42) f32` | symmetric, masked, normalized to `[0,1]` |
| `lengths` | `(42,42) f32` | raw mm |
| `delays` | `(42,42) f32` | `lengths / tract_conduction_speed` (ms) |
| `fc_target` | `(42,42) f32` | empirical FC |
| `sc_mask` | `(42,42) f32` | binary `{0,1}` |
| `region_labels` | `list[str]` len 42 | from Atlas |
| `n_nodes` | int | 42 |
| `cache_tag` | str | feeds every cache key |
| `cache_dir` | str | created on disk; also set as global tvboptim cache path |
| `graph` | `DenseDelayGraph` | wraps `(weights f64, delays f64, labels)` |

## `build_network` Outputs

`network` (`tvboptim.experimental.network_dynamics.Network`) wraps:
- `dynamics = WilsonCowanEIB(c_ei=6.0·jnp.ones((N,), f32))` — `c_ei` is **per-node**.
- `coupling = {"coupling": EIBLinearCoupling(incoming_states=["E"])}` with
  `wLRE = jnp.ones((N,N), f32)`, `wFFI = jnp.ones((N,N), f32)`.
- `graph = data["graph"]`
- `noise = AdditiveNoise(sigma=cfg.additive_noise_sigma, apply_to="E")`

`prepare(network, BoundedSolver(Heun(), 0, 1), t1=warmup_duration_ms, dt=1.0)` →
`(compiled_model, initial_state)`. Warmup is 300 s by default, so
`warmup_result.data` has shape **`(300_000, 2, 42)`** = `(steps, E/I, nodes)`.

`Bold(period=1000, downsample_period=4.0, voi=0, history=warmup_result)` —
emits one BOLD sample per simulated second.

## Per-stage Tensor Shapes

### State during simulation

Inside any stage's compiled model on a `t1=T_ms` window:

| Quantity | Shape | dtype | Notes |
|---|---|---|---|
| `sim_result.data` | `(T_ms/dt, 2, N)` | jnp f32 | (steps, E/I, nodes) |
| `sim_result.auxiliary` | `(T_ms/dt, 4, N)` | jnp f32 | `(S_e, S_i, rE_hz, rI_hz)` if monitored |
| `bold_output.ys` | `(n_tr, 1, N)` | jnp f32 | one row per emitted TR |
| `bold_monitor.history` | `(H, 1, N)` | jnp f32 | rolling BOLD hemodynamic history |
| `network._internal.noise_samples` | implementation-defined | jnp f32 | re-randomised by `advance_internal_state` |

### `ParamSet`

| Field | Shape |
|---|---|
| `c_ei` | `(42,) f32`, clipped to `[0, 20]` |
| `wLRE` | `(42, 42) f32`, clipped to `[0, w_max=1.5]`, masked by `sc_mask`, symmetric |
| `wFFI` | same |
| `c_ei_frozen` | `bool` |

### `StateBundle`

| Field | Shape | Source |
|---|---|---|
| `params` | `ParamSet` | per-stage best |
| `init_dynamics` | `(2, 42) f32` | `sim_result.data[-1]` of previous stage |
| `bold_history` | `(H, 1, 42) f32` or `None` | `bold_monitor.history` snapshot |
| `bold_window` | `(W, 42) f32` or `None` | most recent BOLD samples actually used for FC |
| `internal_state` | `dict[str, np.ndarray]` or `None` | mostly `{"noise_samples": ...}` |
| `delay_history` | `np.ndarray` or `None` | from `network.history.data` |
| `stage` | str | `"warmup"`, `"fic"`, `"eib"`, `"eib_search"`, `"grad"`, `"lowrank"` |
| `metadata` | dict | `rng_seed`, `rng_key (uint32)`, `post_fic_fc_matrix`, `post_fic_fc_corr`, `post_fic_fc_rmse` |

## Stage-to-Stage Flow

```
load_data(cfg)
   └─ data: dict
      │
build_network(cfg, data)
   └─ network, initial_state, bold_monitor, warmup_result
      │
StateBundle.from_warmup(
    warmup_result, bold_monitor,
    ParamSet.default(N).sanitize(sc_mask, w_max),
    capture_internal_state(initial_state),
    capture_network_delay_history(network),
    stage="warmup")
   └─ bundle_init
      │
run_fic(network, bundle_in=bundle_init, cfg, data)
   stage="fic"  params.c_ei tuned per-node, c_ei_frozen=False
   metadata += {post_fic_fc_matrix, post_fic_fc_corr, post_fic_fc_rmse}
   └─ bundle_fic
      │
run_eib(network, bundle_in=bundle_fic, cfg, data)
   internal: rolling 150-TR window → 8000 × 1-TR sim steps
   then post-hoc: top-10 snapshots × 300 s sim → best by weighted score
   stage="eib"  params.wLRE,wFFI updated, c_ei still updated if not frozen
   └─ bundle_eib
      │
run_gradient_optimization(network, bundle_in=bundle_eib, cfg, data)
   loss = (1 − corr_FC) + 0.01·activity_reg
   optax.chain(zero_nans, clip_by_global_norm(0.1), adamaxw(0.002))
   chunked Adamax-W over c_ei + wLRE + wFFI
   stage="grad"
   └─ bundle_grad
      │
run_lowrank_optimization(network, bundle_in=bundle_grad, cfg, data)
   trainable = c_ei + (lre_u, lre_v, ffi_u, ffi_v) of shape (N, r=6)
   w_eff = clip(w_base + δ·UVᵀ, 0, w_max)·sc_mask, sym
   stage="lowrank"
   └─ bundle_lowrank
      │
run_dbs_stimulation(network, bundle_in=bundle_lowrank or bundle_grad, cfg, data)
   for each (label, node) in cfg.dbs_target_regions:
       stim = biphasic pulse train, shape (pre+stim_steps, 42), only target column nonzero
       network.dynamics.dynamics ← stimulated_fn (restored in finally)
       compile+run pre+during sim; LFP = E + I at target node
       writes dbs_analysis/<label>/true_p_t/E_plus_I/{*.png, lfp_timeseries.csv, psd_pre_vs_during.csv}
   returns None
```

## Loss Definitions (single source of truth)

- **`_compute_correlation_loss(p, t)`** (`part3_gradient.py`): Pearson over the off-diagonal mask (subtracts mask-weighted mean before dot product), returns `1 - corr`. So a perfect fit is 0.
- **Full / low-rank gradient total loss**: `corr_loss + activity_weight · activity_reg + (low-rank only) factor_penalty · factor_norm`.
  - `activity_reg = mean((rE_max · mean_E_over_last_500_steps − target_rE)²)`.
- **EIB scoring** (used inside `_eib_update_rule` and post-hoc selection):
  - `full_term = correlation_loss_weight · (1 − corr) + rmse_loss_weight · rmse`
  - `score = −(full_brain_fc_loss_weight · full_term)` — higher is better.
  - `pd_fit_block_loss_weight` is defined but not multiplied into the score in the inspected `_eib_update_rule` (the PD-fit block term is read by other helpers via `resolve_pd_fit_indices`; not exercised in the notebook's current Cell 3 weights of `(1.0, 0.0, 0.8, 0.2)`).
- **`_eib_update_rule`** weight update (per row `i`):
  - `Δ = fc_diff · row_rmse(target, pred, axis=1)`  → `wLRE ← clip(wLRE + η·Δ, 0, w_max) · sc_mask, sym`
  - `wFFI ← clip(wFFI − η·Δ, 0, w_max) · sc_mask, sym`
  - `η` ramps linearly from 0 to `eib_max_weight_learning_rate` over `eib_max_iterations`.
