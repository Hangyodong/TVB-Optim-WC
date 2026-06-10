# 05 — Runbook

## Required Environment

Hard requirements (read from `import` statements only):

- **Python ≥ 3.9** (uses `from __future__ import annotations`, `dataclass(field(default_factory=...))`).
- **`jax`** with a working backend. The notebook explicitly sets these env vars **before** importing JAX:
  ```python
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]   = "platform"
  jax.config.update("jax_enable_x64", False)
  ```
  All array math runs in `float32`.
- **`equinox`** — used for PyTree manipulation (`eqx.tree_at`, `eqx.Module`, `eqx.filter_*`, `eqx.apply_updates`, `eqx.filter_jit`).
- **`optax`** — `optax.chain(zero_nans, clip_by_global_norm(0.1), adamaxw(lr))`.
- **`numpy`**, **`scipy`** (`scipy.signal.welch`, `scipy.integrate.trapezoid`), **`pandas`** (CSV IO + DBS export), **`matplotlib`**, **`IPython`** (`display`, `FileLink` in notebook export cell).
- **`tvboptim`** (in-house). Specifically:
  - `tvboptim.experimental.network_dynamics.{Network, prepare}`
  - `tvboptim.experimental.network_dynamics.core.bunch.Bunch`
  - `tvboptim.experimental.network_dynamics.coupling.base.InstantaneousCoupling`
  - `tvboptim.experimental.network_dynamics.dynamics.base.AbstractDynamics`
  - `tvboptim.experimental.network_dynamics.graph.{DenseGraph, DenseDelayGraph}`
  - `tvboptim.experimental.network_dynamics.noise.AdditiveNoise`
  - `tvboptim.experimental.network_dynamics.solvers.{BoundedSolver, Heun}`
  - `tvboptim.observations.tvb_monitors.bold.Bold`
  - `tvboptim.observations.observation.{fc_corr, rmse}`
  - `tvboptim.optim.optax.OptaxOptimizer`
  - `tvboptim.types.{Parameter, BoundedParameter}`
  - `tvboptim.utils.{cache, set_cache_path}`

There is no `requirements.txt`, no `pyproject.toml`, no setup script in the inspected repo. Install `tvboptim` from wherever your lab keeps it; the rest is plain `pip install jax equinox optax scipy pandas matplotlib`.

## Working Directory

All paths in `Config` are relative. Run the notebook from `/scratch/home/wog3597/optim/` (or wherever the four data files live) so that `weight.csv`, `tract_length.csv`, `FC_compact.csv`, and `Atlas_43.txt` are visible as bare filenames.

## Recommended Order

Run cells **top to bottom**:

| Cell | Time order of magnitude |
|---:|---|
| 1 imports / JAX env | seconds |
| 3 `Config(...)` | instant |
| 5 `load_data` | seconds (plots) |
| 7 `build_network` + warmup | **minutes** (300 s of simulated time = compile + warmup) |
| 13 `run_fic` | **minutes** (up to 2000 × 1 s sim steps, early-stop possible) |
| 15 `run_eib` | **tens of minutes** (8000 × 1 TR sim steps + top-10 × 5 min sims) |
| 17 `run_gradient_optimization` | **tens of minutes to hours** (notebook overrides `optimizer_max_steps=1000` with `chunk_steps=1` and 180 TR windows) |
| 19 `run_lowrank_optimization` | **tens of minutes** (150 steps × 180 TR window) |
| 21 `run_dbs_stimulation` | **tens of minutes** (4 targets × 120 s of simulated time each, JAX recompile per target) |
| 22 figure-export re-run | **doubles the wall-clock above** if cache misses |
| 23 stacked PSD plot | seconds (reads Cell 21 CSVs) |

Skip Cell 8 on first run — it expects PSD CSVs that don't exist until Cell 21 finishes.
Skip Cells 9–11 (empty placeholders).

## Re-running After a Config Change

The disk cache is keyed on:
1. `cfg.cache_version` ← bumping this invalidates **everything**.
2. `data["cache_tag"]` ← changes only if SC or FC matrix bytes change.
3. The relevant `cfg.*` hyperparameters embedded in each stage's cache name (`fic_lr`, `eib_win`, `optimizer_lr`, `lowrank_rank`, etc.).
4. The upstream bundle's `fingerprint()` (includes its `params`, `init_dynamics`, slices of `bold_history`/`bold_window`/`internal_state`/`delay_history`, and stage string).

If a downstream stage's output looks stale, your upstream bundle changed.
Either bump `cache_version` or `rm -rf cache/<cache_tag>/`.

## Outputs You'll See on Disk

- `cache/<cache_tag>/...` — every stage's pickled result.
- `dbs_analysis/<TARGET>/true_p_t/E_plus_I/` per Cell 21:
  - `plot1_lfp_timeseries.png`
  - `plot2_stim_waveform_full.png`
  - `plot3_stim_waveform_zoom.png`
  - `plot4_psd_pre_vs_during.png`
  - `plot5_lfp_pre_vs_during.png`
  - `lfp_timeseries.csv` (`time_ms, lfp_e_plus_i, stimulus`)
  - `psd_pre_vs_during.csv` (`frequency_hz, psd_pre_v2_per_hz, psd_during_v2_per_hz`)
- `paper_dbs_psd_<ts>/DBS_PSD_STN_GPe_GPi.{png,pdf}` (Cell 8)
- `paper_figures_<ts>/DBS_PSD_4regions_stacked_log.{png,pdf}` (Cell 23)
- `all_figures_export_<ts>/` + `all_figures_export_<ts>.zip` (Cell 22)

## GPU vs CPU

The pipeline is JIT-compiled JAX. On CPU it is **usable but slow** — expect the
full Cell 13 → 21 chain to take an hour or more. With a GPU backend (CUDA),
runtime drops to single-digit minutes per stage at the notebook's settings.
The env vars in Cell 1 are tuned to keep XLA from preallocating the whole GPU.

## What NOT to Run

- Do **not** import these modules with the heavy `tvboptim` backend if you only
  need to read source. Use `Read`/`grep`.
- Do **not** run Cell 22 unless you actually want both the original pipeline
  result and a second full pipeline pass for figure export.
- Do **not** run `run_dbs_stimulation` partially and Ctrl-C — it monkey-patches
  `network.dynamics.dynamics`. The `finally` block restores it, but only if the
  function returns or raises cleanly. After a KeyboardInterrupt, re-run
  Cell 7 to rebuild the network.

## Timing Diagnostics & GPU Tuning (Patch 3)

`main.ipynb`에 Cell 7 직후 **timing diagnostics 셀**이 삽입되었다.
시뮬레이션을 실행하지 않고 다음을 출력한다:
- JAX backend / device 확인 (CPU vs GPU)
- JAX per-dispatch latency (200회 noop)
- 단계별 예상 소요 시간 (분석적 추정, ±50% 오차)

### GPU 메모리 prealloc

기본은 `XLA_PYTHON_CLIENT_PREALLOCATE=false`. 외부에서 override 가능 (setdefault 패턴):
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
```
단독 GPU 점유 시 prealloc=true가 더 빠르고 OOM도 줄어든다.

### Patch 3 새 cfg toggle (default OFF)

```python
cfg = Config(
    # ... 기존 설정 ...
    posthoc_parallel=True,        # EIB post-hoc top-k JIT cache 재사용 path
    gpu_batch_size=1,             # 예약 필드
    dbs_parallel_targets=False,   # 현재 sequential fallback (경고 출력)
)
```

`posthoc_parallel=True` → EIB post-hoc (30-50분)이 약 20-30% 단축.
결과는 sequential과 numerical 동등 (호출 타이밍만 다름).

### cache_version bump 경고

`cache_version`이 `v_eituning_oldlogic_match` → `v_eituning_oldlogic_match_p3`로 변경되어
**모든 기존 캐시는 무효화**된다. 디스크 절약:
```bash
rm -rf cache/v_eituning_oldlogic_match_N42_*
```
새 캐시는 `cache/v_eituning_oldlogic_match_p3_N42_*/`에 쓰인다.

### 진정한 device-batched parallelism

Wilson-Cowan dynamics를 batch-aware하게 재작성하면 vmap으로 ~80% 단축 가능하나,
이는 `model.py` 변경(scientific code)이므로 별도 patch(Patch 4 spec)로 분리한다.

