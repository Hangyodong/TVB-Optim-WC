# 01 — Repo Overview

## Purpose

This repo runs a **whole-brain Wilson–Cowan E/I-tuning pipeline** on a 42-region
mouse/rodent atlas, then injects **biphasic Deep Brain Stimulation (DBS)** into
target nuclei and compares pre vs during power spectra. It is built on top of
the `tvboptim` library (TVB-style network dynamics + JAX backend).

The pipeline produces, in order:

1. **FIC** — per-node feedback inhibitory control (`c_ei`) so that each region's
   mean excitatory firing rate `rE` lands on a target (default 4 Hz).
2. **EIB** — joint tuning of `c_ei` and the two coupling weight matrices
   `wLRE` / `wFFI` so simulated BOLD-FC approaches the empirical target FC.
3. **Full-matrix gradient** — Adamax-W optimization of `c_ei`, `wLRE`, `wFFI`
   with a (1 − corr) + activity-regularization loss.
4. **Low-rank gradient (Part 3B)** — adds a rank-`r` correction `Δw = δ·U·Vᵀ`
   on top of the full-matrix solution.
5. **DBS** — for each target region (STN_L, GPe_L, GPe_R, GPi_L) injects a
   gapless biphasic pulse train (default 130 Hz, 60 s pre + 60 s during) and
   computes pre-vs-during LFP/PSD/β-band ratio.

## Entry Point

The single user-facing entry point is **`main.ipynb`**. The eight `.py` files
exist purely so cells stay short — there is no CLI, no `__main__`, no test
suite. The notebook hard-codes its data filenames relative to the working
directory.

## Architectural Spine

All stage-to-stage handoff goes through one contract:

- **`ParamSet`** — `(c_ei, wLRE, wFFI, c_ei_frozen)` (`pipeline_contracts.py`).
- **`StateBundle`** — `(params, init_dynamics, bold_history, bold_window,
  internal_state, delay_history, stage, metadata)`.

Each `run_*` function takes a `StateBundle` and returns a `StateBundle`. The
bundle also exposes `dynamics`/`coupling`/`initial_state` views so legacy
TVB-style code that touched `state.dynamics.c_ei` etc. still works.

## Caching

`tvboptim.utils.cache` is used as a decorator inside every stage. The cache key
is built from a stage tag + the bundle's `fingerprint()` (SHA-1 over params,
init dynamics, BOLD window/history head, internal-state head, delay-history
head, and stage string). Re-running with identical config and identical
upstream bundle is a disk read.

Cache root: `./cache/<cache_version>_N42_sc<10hex>_fc<10hex>/`. Cache version
is `config.Config.cache_version` (default `"v_eituning_oldlogic_match"`).

## Dependencies (import surface)

`jax`, `equinox`, `optax`, `numpy`, `pandas`, `scipy`, `matplotlib`, and the
in-house `tvboptim` package (`tvboptim.experimental.network_dynamics.*`,
`tvboptim.observations.*`, `tvboptim.optim.optax.OptaxOptimizer`,
`tvboptim.types.{Parameter, BoundedParameter}`, `tvboptim.utils.{cache,
set_cache_path}`).

## Data

42-region atlas. SC / tract-length / FC are 42×42 CSVs (no header). Labels
sit in `Atlas_43.txt`. See `04_data_flow.md` for shapes and transforms.

## Out of Scope for Docs

- `tvboptim` internals (treated as a stable library).
- Heavy execution: don't run `main.ipynb` or any `run_*` function while
  authoring these notes — they take tens of minutes on a GPU and longer on
  CPU. Static inspection only.
