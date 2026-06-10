# 02 — Repo Tree

The repo is flat. Everything lives in one directory; there is no `src/`,
no package, no submodules.

```
optim/
├── main.ipynb               6.4 MB   Single user-facing notebook (24 cells, ~1.1k LoC + many embedded outputs)
├── config.py                5.6 KB   One @dataclass with every hyperparameter
├── data_loader.py           6.7 KB   CSV → numpy → DenseDelayGraph; writes plot of SC / delays / FC
├── model.py                 6.1 KB   WilsonCowanEIB dynamics, EIBLinearCoupling, build_network()
├── pipeline_contracts.py   25.3 KB   ParamSet, StateBundle, internal-state capture/restore, fingerprinting
├── part1_fic.py            20.3 KB   FIC loop (per-node c_ei tuning to target rE)
├── part2_eib.py            23.0 KB   EIB exploration loop + post-hoc top-k validation
├── part3_gradient.py       32.9 KB   Full-matrix and low-rank (Part 3B) gradient optimizers
├── part4_dbs.py            26.4 KB   Biphasic pulse train + PSD/β-band analysis per target
├── Atlas_43.txt             1.3 KB   42 region labels (note: file is named "_43" but contains 42 lines)
├── weight.csv              13.5 KB   42×42 structural connectivity (SC) weights, no header
├── tract_length.csv        12.2 KB   42×42 tract-length matrix (mm), no header
└── FC_compact.csv          28.4 KB   42×42 empirical functional connectivity target, no header
```

Generated at runtime (not committed):

```
cache/                       tvboptim disk cache (created by data_loader → set_cache_path)
optimized_params/            Reserved by Config.param_save_dir; not actively written by the inspected code
dbs_analysis/                Created by part4_dbs.run_dbs_stimulation (PNGs + CSVs per target)
paper_dbs_psd_<timestamp>/   Created by notebook Cell 8 (PSD figure helper)
paper_figures_<timestamp>/   Created by notebook Cell 23 (stacked 4-region PSD)
all_figures_export_<ts>/     Created by notebook Cell 22 (mass figure re-run + ZIP)
```

## Per-file Role (one-line each)

| File | Role |
|---|---|
| `config.py` | Single `@dataclass Config` — every knob the pipeline reads. `print_summary()` and `get_baseline_settle_duration_ms()` are the only methods. |
| `data_loader.py` | `load_data(cfg) → dict`. Reads three CSVs, builds SC mask, log1p-normalises weights, divides lengths by `tract_conduction_speed` to get delays, instantiates a `DenseDelayGraph`, sets the on-disk cache path, and plots W/D/FC. |
| `model.py` | `WilsonCowanEIB` (E/I two-population dynamics, JAX), `EIBLinearCoupling` (separate wLRE/wFFI), `build_network(cfg, data)` (assembles Network + Bold monitor + runs `warmup_duration_ms` warmup). |
| `pipeline_contracts.py` | `ParamSet` / `StateBundle` and helpers: `capture_internal_state`, `restore_internal_state`, `advance_internal_state` (splits the noise RNG forward), `capture/restore/sync_network_delay_history`, `update_bold_history`, `extract_bold_window`, `build_bundle_from_legacy_state`, `resolve_pd_fit_indices`. |
| `part1_fic.py` | `run_fic(network, bundle_in, cfg, data) → StateBundle`. FIC loop tunes `c_ei` per node; produces post-FIC FC summary in `bundle.metadata`. |
| `part2_eib.py` | `run_eib(network, bundle_in, cfg, data) → StateBundle`. Two-stage: rolling-window EIB exploration with snapshots + post-hoc top-k longer-sim validation. |
| `part3_gradient.py` | `run_gradient_optimization(...)` (Adamax-W over full c_ei + wLRE + wFFI) and `run_lowrank_optimization(...)` (rank-r `Δw` correction on top of grad result). Shares loss helpers `_compute_correlation_loss`, `_compute_activity_regularization`. |
| `part4_dbs.py` | `run_dbs_stimulation(network, bundle_in, cfg, data) → None`. For each `cfg.dbs_target_regions` entry, builds a gapless biphasic pulse train, monkey-patches `network.dynamics.dynamics` with `_make_stimulated_dynamics(...)`, runs pre+during simulation, writes plots + `lfp_timeseries.csv` + `psd_pre_vs_during.csv`. |

## Notebook Cell Map (what runs where)

| Cell | Type | Purpose |
|---:|---|---|
| 0 | md | Title |
| 1 | code | **JAX env** (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `_ALLOCATOR=platform`), `jax_enable_x64=False`, imports |
| 2 | md | "edit only Cell 3" warning |
| 3 | code | `cfg = Config(...)` — only place hyperparameters are set |
| 4 | md | — |
| 5 | code | `data = load_data(cfg)` |
| 6 | md | — |
| 7 | code | `build_network(cfg, data)` + `bundle_init = StateBundle.from_warmup(...)` |
| 8 | code | Pre-built PSD figure helper (reads `dbs_analysis/<target>/...psd_pre_vs_during.csv`). Will fail until Part 4 has run. |
| 9–11 | code | Empty placeholders |
| 12 | md | — |
| 13 | code | `bundle_fic = run_fic(...)` |
| 14 | md | — |
| 15 | code | `bundle_eib = run_eib(...)` |
| 16 | md | — |
| 17 | code | `bundle_grad = run_gradient_optimization(...)` |
| 18 | md | — |
| 19 | code | `bundle_lowrank = run_lowrank_optimization(bundle_in=bundle_grad, ...)` |
| 20 | md | — |
| 21 | code | `run_dbs_stimulation(bundle_in=bundle_lowrank or bundle_grad, ...)` |
| 22 | code | **Re-runs Parts 1→4 end-to-end** with a `plt.show()` monkey-patch that saves every figure into `all_figures_export_<ts>/` + zips it |
| 23 | code | 4-row stacked PSD plot across `cfg.dbs_target_regions` keys |

> **Important:** Cell 22 silently re-executes the entire pipeline. If the cache
> tag hasn't changed, those calls become disk reads; if it has, every stage
> runs again. See `06_known_errors.md`.
