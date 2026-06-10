# 11 — Patch 1 Report

## Status
APPLIED

## Spec
- Source: `10_patch1_spec.md`
- Patch: replace `"FC.csv"` with `"FC_compact.csv"` in `main.ipynb` Cell 3.

## Files changed
- `main.ipynb` — Cell 3 (cell_id `9709587f`), one line. The cell's `source`
  was replaced with byte-identical text except for the single
  `"FC.csv"` → `"FC_compact.csv"` substitution on the `fc_csv` field.

## Files NOT changed
- All `.py` source files.
- All CSV / TXT data files (`weight.csv`, `tract_length.csv`, `FC_compact.csv`, `Atlas_43.txt`).
- All `0*_*.md` docs (including `06_known_errors.md` — its follow-up
  edit is deferred per spec).
- All notebook `outputs` arrays.

## Pre-edit state
- cells_before: 24
- Cell 3 had `"FC.csv"`: 1
- Cell 3 had `"FC_compact.csv"`: 0

## Post-edit state
- cells_after: 24
- Cell 3 has `"FC.csv"`: 0   (expect 0)
- Cell 3 has `"FC_compact.csv"`: 1   (expect 1)

## Acceptance checks
| # | Check | Result |
|---|---|---|
| 1 | JSON valid + cell_count==24 | PASS |
| 2 | Cell 3 string assertions | PASS |
| 3 | `compileall -q .` clean | PASS |
| 4 | `import data_loader` clean | PASS |
| 5 | `FC_compact.csv` present, `FC.csv` absent | PASS |

## What was NOT done (intentional)
- No pipeline execution (`run_fic`, `run_eib`, `run_gradient_optimization`,
  `run_lowrank_optimization`, `run_dbs_stimulation` were not called).
- No notebook cell re-run; embedded outputs unchanged.
- No `06_known_errors.md` wording update (deferred per spec § "Optional
  doc follow-up").
- No edits to `data_loader._resolve_path`; the fallback stays in place.

## Behavior delta (vs. spec table)
- File `load_data` reads: `FC_compact.csv` (unchanged; was via fallback, now direct).
- `cache_tag`: unchanged (FC matrix bytes identical).
- On-disk caches under `cache/<cache_tag>/`: preserved.
- Equations, losses, constants, long-run behavior: unchanged.

## Rollback
Single-line revert in `main.ipynb` Cell 3:
`"FC_compact.csv"` → `"FC.csv"`. No cache invalidation required.
