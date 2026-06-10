# 07 — Refactor Rules

Hard constraints any refactor must respect, plus the soft conventions the
existing code follows.

## Invariants you cannot break

1. **`StateBundle` is the only stage handoff.** Every `run_*` function must
   accept a `StateBundle` and return a `StateBundle`. Legacy adapters
   (`build_bundle_from_legacy_state`, the `_coerce_*_bundle` functions) exist
   for back-compat — do not extend the legacy paths; extend the bundle paths.

2. **`ParamSet.sanitize(sc_mask, w_max)` is the param normaliser.** Any new
   place that mutates `c_ei`, `wLRE`, or `wFFI` and produces a bundle **must**
   call `sanitize(...)` before storing. Otherwise NaNs, asymmetry, or
   off-mask edges leak forward.
   - `c_ei ∈ [0, 20]`, NaN/inf scrubbed to a sane default.
   - `wLRE`, `wFFI ∈ [0, cfg.connectivity_weight_max]`, masked by `sc_mask`,
     symmetrised as `0.5 · (W + Wᵀ)`.

3. **All on-disk cache keys are `(stage-tag, relevant cfg fields, upstream
   bundle.fingerprint())`.** When you add a new cfg field that affects the
   stage's *computation*, add it to the cache name. Otherwise stale caches
   will silently shadow new behaviour. When you add a new bundle field that
   affects the stage's *output*, add it to `StateBundle.fingerprint()`.

4. **`build_network` runs the warmup.** Downstream code expects
   `warmup_result` to have already been simulated for `cfg.warmup_duration_ms`.
   Do not lazily defer the warmup; the `Bold` monitor and the initial bundle
   are seeded from it.

5. **JAX env vars must be set before `import jax`.** The notebook's Cell 1
   does this. If you extract initialization into a module, those `os.environ`
   writes must happen at the top of the module before any transitive
   `import jax`.

6. **`run_dbs_stimulation` patches `network.dynamics.dynamics`.** Always wrap
   the patch in `try/finally` and restore in `finally`. Never patch and
   `return` without restoring — subsequent stages run a stimulated network.

7. **BOLD time-series go through `_compute_fc_*` helpers.** Don't compute FC
   ad-hoc. The helpers handle NaN scrubbing, mean-centring, std-flooring,
   diagonal-zeroing. If you need a new FC variant, add a sibling helper, not
   an inline computation.

## Soft conventions to keep

- **Korean inline comments** are pervasive. Don't translate or strip them
  unilaterally — they document author intent ("구버전 notebook 로직", "old logic
  match") that flag deliberate non-obvious choices.
- **No `__main__` blocks.** The notebook is the entry point; modules are libraries.
- **No tests, no CI.** Adding either is fine; not having them is the current state.
- **Per-stage `_run_*_pure(network, init_dict, cfg, data) → dict`** is the
  cacheable inner kernel. It accepts a `bundle.to_dict()` (so it can be JSON-
  ish for the cache) and returns a dict with at least `{"bundle": ...}`.
  Outer `run_*` wraps it with `@cache(...)` and `StateBundle.from_dict(...)`.
- **Underscored helpers are module-private** but four (`_compute_beta_*`,
  `_compute_normalized_psd`, `_plot_beta_summary`, `_extract_firing_rates`,
  `_compute_bundle_fc_summary`) are imported across files. Treat *those five
  specifically* as public-ish; don't rename or change their signatures
  without grepping for cross-module imports.
- **Print logging, not logging-module.** Every stage prints a banner, progress
  rows, and a `[STAGE] Done — ...` line. Match the format if you add stages.
- **Plotting always uses `matplotlib` + `plt.show()`** (no Agg backend
  switches). The Cell 22 figure-export trick monkey-patches `plt.show` —
  do not change `plt.show()` calls to e.g. `plt.savefig + close` directly.

## When you add a new stage

Minimum checklist:

- [ ] Define a `run_<stage>(network, bundle_in: StateBundle, cfg, data) → StateBundle`.
- [ ] Build the inner `_run_<stage>_pure(network, init_dict, cfg, data) → dict`.
- [ ] Use `bundle.to_tvb_state(network, solver, t1, dt)` to get `(model, state)` so internal/delay/init state restore correctly.
- [ ] Use `bundle.build_bold_monitor(cfg)` instead of constructing `Bold(...)` from scratch.
- [ ] At loop end: call `update_bold_history`, `advance_internal_state`, `sync_network_delay_history`, and `bundle.advance(...)` with **all** of `new_params`, `new_init_dynamics`, `new_bold_history`, `new_bold_window`, `new_internal_state`, `new_delay_history`, `next_stage`, `metadata_update`.
- [ ] Sanitize params with `cfg.connectivity_weight_max` and `data["sc_mask"]`.
- [ ] Wrap with `@cache(name)` where `name` includes the upstream fingerprint and all cfg fields you read.
- [ ] If the loss uses different correlation/RMSE weights than EIB, add new `cfg` fields rather than reusing EIB's.

## When you change a hyperparameter default in `config.py`

The notebook's Cell 3 sets most fields **explicitly** (e.g. `optimizer_max_steps=1000`),
so editing the dataclass default won't change current runs. But the cache
keys are built from the *runtime* `cfg` field values, so any per-stage cache
that doesn't include a particular field in its cache name will silently
re-use stale results when that field changes. Cross-reference your new field
against the cache name strings in `part1_fic.py:65`, `part2_eib.py:59`,
`part3_gradient.py:95`, `part3_gradient.py:266`.

## When you change `Atlas_43.txt` or the CSVs

- Row/column counts must remain identical across all four files. The
  `_validate_shapes` assertion will catch mismatch.
- `_build_cache_tag` hashes the top-left 8×8 of SC and FC. Any change
  visible in that corner will rotate the cache tag. Any change *outside*
  that corner will NOT rotate it — bump `cfg.cache_version` manually.

## When you touch caching

`tvboptim.utils.cache` and `set_cache_path` are external. Treat them as opaque.
The only knob this repo turns is `cfg.cache_version` and the per-stage cache
name strings. Do not invent ad-hoc on-disk file naming for stage outputs —
that's what the cache is for.

## Style of edits

- Don't add planning/decision comments. Don't write "// added for X" comments.
- Don't reformat unrelated lines while making a fix.
- Keep `from __future__ import annotations` in `pipeline_contracts.py` —
  it lets the class self-reference `"StateBundle"` in type hints.
