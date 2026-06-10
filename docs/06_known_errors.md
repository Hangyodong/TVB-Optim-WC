# 06 — Known Errors, Risks, and Sharp Edges

Issues observed by reading source. Nothing here was reproduced by running.

## Colab compatibility risks

1. **`tvboptim` is not on PyPI.** Colab cells must `pip install` (or `pip install -e`) it from your private source before Cell 1 can import. The notebook does not include this install cell.
2. **JAX env vars must be set before any JAX import.** Cell 1 does this correctly, but if a user reorders or pre-runs `import jax` from a different cell (e.g. for a check), `XLA_PYTHON_CLIENT_PREALLOCATE` and `XLA_PYTHON_CLIENT_ALLOCATOR` will silently have no effect — XLA will preallocate the GPU.
3. **`jax_enable_x64=False`** is set globally. Some `tvboptim` operations that look 64-bit (e.g. the `weights.astype(np.float64)` passed to `DenseDelayGraph`) will be downcast back to f32 inside JIT. Don't expect 64-bit precision anywhere downstream.
4. **Working directory.** Colab's default cwd is `/content/`. Cell 5 calls `load_data` with bare filenames; you must upload the four data files to `/content/` (or `os.chdir(...)` first) or the run dies on `pd.read_csv`.
5. **Disk persistence.** `cache/`, `dbs_analysis/`, `paper_*_*/`, `all_figures_export_*` are written under cwd. On Colab they vanish at session end unless you mount Drive.
6. **`fc_csv` mismatch (notebook vs repo).** Cell 3 sets `fc_csv = "FC.csv"`, but the file shipped is `FC_compact.csv`. `data_loader._resolve_path` silently substitutes the fallback `"FC_compact.csv"`, so it works **only if the fallback file is present**. If you upload only `FC.csv` (renamed) without `FC_compact.csv`, the resolver still returns `"FC.csv"` and `pd.read_csv` proceeds — but **the cache fingerprint will differ** from any pre-built cache because the FC matrix top-left 8×8 bytes feed `_build_cache_tag`. Plan: either rename to `FC_compact.csv`, or set `cfg.fc_csv` to match the actual file you upload.
7. **Long single-cell runtimes vs Colab idle timeout.** Cells 17 and 22 can each easily exceed Colab's free-tier 12-hour limit on CPU and 90-min idle on GPU. Use Colab Pro+ or batch via a script, not the notebook.

## Long-runtime traps

- **Cell 22 re-runs Parts 1–4 end to end.** Anyone who runs the notebook top to bottom pays the runtime twice. If the cache is warm (no config change after Cell 21), Cells 13–19 become disk reads and only DBS re-runs; if any cache key shifted, expect a multi-hour repeat.
- **Notebook overrides defaults toward heavier compute.** Cell 3 sets `optimizer_max_steps=1000` (default 200), `optimizer_chunk_steps=1` (default 5 → 5× more compile-then-step cycles), `optimizer_bold_window_tr=180` (default 96, ≈2× sim time per step), `lowrank_bold_window_tr=180`, `lowrank_max_steps=150`. Reviewers reading `config.py` defaults will underestimate runtime by ~5–10×.
- **`run_dbs_stimulation` recompiles per (target, mode).** The injected `stimulated_dynamics` closure captures a new `stimulation_jax`, so JAX re-traces each iteration of the outer loop. 4 targets × ~120 s of simulated time × JIT compile is the dominant cost.
- **`tract_conduction_speed` is 1.0 in the notebook vs 3.0 default.** Delays are 3× larger. With `dt = 1.0 ms` and 42×42 delay matrix, this triples the delay-line memory the solver needs to track.

## Known bugs / sharp edges (from source reading)

- **Atlas filename vs content mismatch.** `Atlas_43.txt` has **42** non-empty lines. `_load_region_labels` filters blank lines, so `len(region_labels)==42`. The matrices are 42×42, so consistent — but the "43" in the filename is misleading.
- **`run_eib` returns `c_ei_frozen=False`** even after FIC is done. The Stage-3 comment in `part1_fic.py` says "FIC 종료 후 c_ei를 freeze_c_ei()로 동결한다" but the code actually constructs `ParamSet(..., c_ei_frozen=False)`. EIB then continues to tune `c_ei`. This is intentional ("구버전 notebook 로직처럼 FIC 이후에도 c_ei를 계속 조정") but the docstring/comment lies. Don't trust the docstring; trust `_run_fic_loop_pure`.
- **`pd_fit_block_loss_weight` is wired to scoring but unused in the active EIB update.** Cell 3 sets it to `0.0`, so this is currently dead weight. If a future config sets it nonzero, double-check `_eib_update_rule` — that function ignores the PD-fit block and only uses `full_brain_fc_loss_weight`, `correlation_loss_weight`, `rmse_loss_weight`.
- **`extract_bold_window` and `_compute_fc_from_bold_output` differ on NaN handling.** The former just casts; the latter calls `np.nan_to_num`. If a stage hands a NaN-laced bold_output to a downstream FC computation that uses `extract_bold_window`, NaNs propagate into the bundle. Stages currently always run FC through `nan_to_num`, but a future caller of `extract_bold_window` should be aware.
- **`_make_stimulated_dynamics` monkey-patches `network.dynamics.dynamics`.** If an exception inside the JAX trace leaks out *before* the `try`/`finally` enters, the patch is not applied yet (safe). If it leaks out *after* `network.dynamics.dynamics = MethodType(...)` and *before* `finally` runs (e.g. process kill, Colab tab close, Ctrl-C during JIT compile), the network is left in a stimulated state until you re-`build_network`.
- **Cache fingerprint includes only the first 8×8 of SC and FC.** Two FC matrices that differ outside the top-left 8×8 will hash to the same `cache_tag`. Unlikely in practice, but a real collision risk if someone permutes node order without touching the corner.
- **`build_network` runs the warmup eagerly with `jax.block_until_ready`** even though Cell 7 only stores `warmup_result` into a `StateBundle`. This is by design (the bundle needs the final state), but it means Cell 7 itself is the first big compile-and-run.
- **`BoundedSolver(Heun(), low=0.0, high=1.0)`** clamps `E` and `I` to `[0, 1]` at every Heun step. Targets like `rE = 4 Hz` translate to `E ≈ 0.2` so this is comfortable, but pushing `fic_target_firing_rate_hz` near `rE_max_hz = 20.0` saturates `E` at 1.0 and FIC will silently fail to converge.
- **`Cell 8` (PSD figure) precedes Cell 21 in the notebook.** Running cells in order, Cell 8 will report "(directory not found)" instead of plotting, because Part 4 hasn't generated `dbs_analysis/*` yet. Move Cell 8 below Cell 21, or just skip it on the first pass.
- **`run_fic` cache name includes `_fp{bundle_init.fingerprint()}`** which depends on `bundle_init.delay_history`. If a future patch changes how `capture_network_delay_history` reads from the network (different attribute or dtype), the fingerprint shifts and all prior caches go stale silently.

## Defensive `# TODO` candidates

- Make `cache_version` include `jax.lib.__version__` or `jaxlib` build, since float32 ops aren't bit-stable across JAX versions and a stale cache could mislead.
- Tighten `_build_cache_tag` to hash the whole matrix rather than `[:8, :8]`.
- Add a smoke test that round-trips a `StateBundle` through `to_dict`/`from_dict` and re-applies it to a fresh `prepare(...)` to catch silent shape mismatches early.
- Add `requirements.txt` (or `environment.yml`) — at minimum pin `jax`, `equinox`, `optax` versions that the in-house `tvboptim` was tested against.
