# 09 — Baseline Smoke Report

Lightweight, read-only baseline checks. No `run_*` function or `main.ipynb`
cell was executed. Confirms what the 01–08 docs describe is actually what's
sitting on disk.

- **Date:** 2026-05-19
- **Host:** Linux 4.18 (RHEL-class, kernel `4.18.0-513.5.1.el8_9.x86_64`)
- **Working dir:** `/scratch/home/wog3597/optim`

---

## Environment

```
pwd            /scratch/home/wog3597/optim
python         /scratch/home/wog3597/anaconda3/bin/python
python --ver   Python 3.13.9
```

Result: **OK.** ≥ 3.9 required (see `05_runbook.md`); 3.13.9 satisfies that.

## Directory listing (`ls -la`)

22 entries (4 KB dir block, parent dir, 8 source `.py`, 4 data files, 1 notebook,
8 generated `0*_*.md` docs from the prior turn). No `cache/`, `dbs_analysis/`,
`paper_*`, `all_figures_export_*`, or `__pycache__` yet — the workspace is
clean of generated artefacts. File sizes match `02_repo_tree.md` exactly.

## Byte-compile (`python -m compileall -q .`)

```
exit=0
```

Result: **OK.** All `.py` files compile cleanly. AST parse cross-check on the
eight source files also passes (no SyntaxError):

```
AST-OK  config.py
AST-OK  pipeline_contracts.py
AST-OK  data_loader.py
AST-OK  model.py
AST-OK  part1_fic.py
AST-OK  part2_eib.py
AST-OK  part3_gradient.py
AST-OK  part4_dbs.py
```

## Dependency probe

Every import surface documented in `05_runbook.md` is present:

| Package | Status | Version |
|---|---|---|
| `numpy` | OK | 2.2.6 |
| `pandas` | OK | 2.3.3 |
| `scipy` | OK | 1.17.1 |
| `matplotlib` | OK | 3.10.6 |
| `jax` | OK | 0.7.2 |
| `jaxlib` | OK | 0.7.2 |
| `equinox` | OK | 0.13.6 |
| `optax` | OK | 0.2.8 |
| `tvboptim` | **OK** | 0.2.6 |
| `IPython` | OK | 9.7.0 |

Result: **OK.** No dependency-risk fallbacks triggered. In a Colab session
without the in-house `tvboptim` available, the next step (local-module
import) would have failed; here it doesn't.

## Local module imports

Importing each module imports its `tvboptim` symbols and instantiates module
constants, but does **not** invoke any pipeline function. All clean:

```
IMPORT-OK  config
IMPORT-OK  pipeline_contracts
IMPORT-OK  data_loader
IMPORT-OK  model
IMPORT-OK  part1_fic
IMPORT-OK  part2_eib
IMPORT-OK  part3_gradient
IMPORT-OK  part4_dbs
```

Result: **OK.** Confirms the `tvboptim` API surface this repo expects
(`Network`, `prepare`, `DenseDelayGraph`, `DenseGraph`, `AdditiveNoise`,
`BoundedSolver`, `Heun`, `Bold`, `fc_corr`, `rmse`, `OptaxOptimizer`,
`Parameter`, `BoundedParameter`, `cache`, `set_cache_path`,
`InstantaneousCoupling`, `AbstractDynamics`, `Bunch`) is all available in
`tvboptim==0.2.6`.

## `Config()` instantiation

Defaults from `config.py` come up cleanly (no notebook overrides applied):

```
fc_csv          = FC_compact.csv          ← matches the file shipped
region_txt      = Atlas_43.txt
cache_version   = v_eituning_oldlogic_match
baseline_settle = 0 ms                    ← Optional[int]=0 path (not the optimizer-window fallback)
warmup_duration = 300000 ms
fic_max_iters   = 2000
eib_max_iters   = 8000
opt_max_steps   = 200                     ← notebook Cell 3 overrides to 1000 (see 06_known_errors.md §"Long-runtime traps")
lowrank_steps   = 120                     ← notebook Cell 3 overrides to 150
dbs_targets     = {'STN_L': 11, 'GPe_L': 5, 'GPe_R': 6, 'GPi_L': 7}
```

Result: **OK.** Note `Config.fc_csv` default is `"FC_compact.csv"`; the
notebook's Cell 3 sets `"FC.csv"` and relies on `data_loader._resolve_path`
fallback (`06_known_errors.md` flag #6).

## Notebook JSON validity

```
size_bytes        6_376_493              (~6.1 MiB)
nbformat          4.5
JSON              valid (parsed without exception)
cell_count        24
by_type           markdown=9, code=15
code_loc          568 lines (non-blank code cells only)
empty_code_cells  3                      (matches cells 9–11 documented in 02_repo_tree.md)
embedded outputs  116
```

Result: **OK.** Cell count and type breakdown match `02_repo_tree.md` ("Notebook
Cell Map"). The bulk of the 6 MB size is the 116 embedded outputs (PNGs base64-
encoded), not source.

## CSV shapes

```
weight.csv          shape=(42, 42)  dtype=float64  min=0.0000e+00  max=1.0000e+00   nan=0  square=True
tract_length.csv    shape=(42, 42)  dtype=float64  min=0.0000e+00  max=8.2888e+00   nan=0  square=True
FC_compact.csv      shape=(42, 42)  dtype=float64  min=-5.5286e-01 max=7.7008e-01   nan=0  square=True
Atlas_43.txt        non-empty lines=42
                    first='1 MOp5_R_Primary_motor_area_Layer_5'
                    last ='42 Cortex_17_R_CIN_Posterior'
```

Result: **OK.** All three matrices are square 42×42, no NaNs, label count
matches. Confirms the "42, not 43" finding documented in `06_known_errors.md`
and `02_repo_tree.md`.

### Range observations (new, not in 01–08)

These are properties of the **raw on-disk** data, before any transform:

- `weight.csv` already sits in `[0, 1]` with max exactly `1.0`. The
  `_load_matrices` reader doesn't normalise; `load_data` then applies
  `log1p(W + 0.5)` and divides by the new max, so the on-disk max=1.0
  becomes `log1p(1.5)/log1p(...) ≈ 1.0` only if 1.0 was already the row
  maximum — which it is. **Takeaway:** the SC matrix in this repo is
  pre-normalised; the in-pipeline `log1p` is a soft squash, not a hard
  normalisation. Don't expect the SC to look like raw streamline counts.
- `tract_length.csv` max is **8.29** — small for raw human-brain mm, plausible
  for mouse (mouse brain ≈ 10 mm across), or this matrix is in some scaled
  unit. The docs (`config.py:tract_conduction_speed`) treat the unit as
  "mm" and report delays in "ms" via `lengths / speed`. With notebook
  Cell 3 override `tract_conduction_speed = 1.0`, max delay ≈ 8.3 ms.
- `FC_compact.csv` range `[-0.55, 0.77]` is typical for a correlation matrix
  with diagonal zeroed; the diagonal is **not** all 1.0 here, so the loader
  does **not** need to zero it explicitly for the FC target.

## Overall

**No blockers.** All eight invariants the docs assume about the workspace hold:

- Source compiles, parses, and imports.
- `tvboptim` 0.2.6 satisfies the import surface.
- Data files exist, are well-formed, and have the documented 42×42 / 42-row shape.
- `main.ipynb` is valid JSON with the cell map described in `02_repo_tree.md`.
- `Config()` round-trips with defaults, and the `fc_csv` mismatch flagged in
  `06_known_errors.md` is real (default `FC_compact.csv`, notebook Cell 3
  override `FC.csv`).

The only environment-side caveat: **3 empty code cells** (Cells 9–11) in the
notebook would noisily appear in `jupyter nbconvert --to script` output. Not a
functional risk, just a tidy-up note for the upload guide in `08`.

---

### Reproduce

```bash
cd /scratch/home/wog3597/optim
python --version
python -m compileall -q .
python -c "import config; from config import Config; Config().print_summary()" | head -40
python -c "import pandas as pd; print(pd.read_csv('weight.csv', header=None).shape)"
python -c "import pandas as pd; print(pd.read_csv('tract_length.csv', header=None).shape)"
python -c "import pandas as pd; print(pd.read_csv('FC_compact.csv', header=None).shape)"
python -c "print(sum(1 for ln in open('Atlas_43.txt') if ln.strip()))"
python -c "import json; nb=json.load(open('main.ipynb')); print(len(nb['cells']))"
```

Expect: 42 / 42 / 42 / 42 / 24, no exceptions.
