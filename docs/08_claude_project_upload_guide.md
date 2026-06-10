# 08 — Claude Project Upload Guide

What to upload to a Claude **Project Knowledge** so future chats have full
context without re-reading the repo.

## Recommended bundle

Upload these 16 files. Total ≈ 250 KB, well under any Project limit.

### Tier 1 — must include (knowledge docs)

| File | Why |
|---|---|
| `01_repo_overview.md` | Frame the project in one read. |
| `02_repo_tree.md` | Tells the model what files exist and what they do. |
| `03_module_index.md` | Function/class signatures so it can cite by name. |
| `04_data_flow.md` | Shapes + transforms — answers most "what shape?" questions without code. |
| `05_runbook.md` | Dependencies, env vars, runtime budget. |
| `06_known_errors.md` | Steers around Colab/runtime traps. |
| `07_refactor_rules.md` | Invariants the model must respect when editing. |
| `08_claude_project_upload_guide.md` | This file — meta. |

### Tier 2 — source files (read-only context)

| File | Why upload |
|---|---|
| `config.py` | Hyperparameter surface — the model will reach for this constantly. |
| `pipeline_contracts.py` | The architectural spine. Every other module imports from here. |
| `model.py` | Dynamics + coupling math — needed to answer "what does `c_ei` do?" type questions. |
| `data_loader.py` | Small; documents the SC/FC transforms. |
| `part1_fic.py`, `part2_eib.py`, `part3_gradient.py`, `part4_dbs.py` | One per stage; cross-referenced by the docs. |

### Tier 3 — data fixtures

| File | Why |
|---|---|
| `Atlas_43.txt` | 42 region labels. Lets the model resolve `STN_L`, `GPe_R`, etc. without guessing. |

### Do **not** upload

- `main.ipynb` — 6.4 MB, mostly base64 PNG outputs. Too noisy. Upload the
  cell map in `02_repo_tree.md` instead. If you need cell-level discussion,
  extract the source-only view with `jupyter nbconvert --to script main.ipynb`
  and upload the resulting `main.py`.
- `weight.csv`, `tract_length.csv`, `FC_compact.csv` — large numeric blobs.
  The model cannot do anything with the raw numbers; the *shape* (42×42) and
  *normalization* are already documented in `04_data_flow.md`. Skip unless
  you specifically want to discuss a numerical value at a known (i, j).
- `cache/`, `dbs_analysis/`, `paper_*` — generated; not part of the source
  of truth.

## Project instructions you should set

Paste this into the Project's custom instructions field (or
`CLAUDE.md` if checked into a future repo):

```
This is the EI-tuning Wilson-Cowan pipeline. Treat the eight 0*_*.md files
as authoritative for repo structure, data shapes, and conventions; trust
them over your own re-derivation from source.

Hard rules:
- All stage handoff goes through StateBundle (see pipeline_contracts.py).
- Never bypass ParamSet.sanitize() when producing a new bundle.
- Cache keys live in the run_* functions; if you change a cfg field that
  affects a stage's computation, update that stage's cache name string too.
- Do not execute the notebook or any run_* function — they take 10s of
  minutes to hours. Reason statically.
- Korean inline comments document deliberate "old-logic-match" choices;
  preserve them on edit.

When the user asks "how does X work":
1. Cite the file:line.
2. Quote the smallest relevant snippet.
3. Defer to the docs for shapes/runtime/invariants.
```

## Refreshing the Project

When you change source:

1. Re-upload only the changed `.py` files.
2. If the change touched stage boundaries, shapes, or hyperparameters,
   regenerate the affected doc(s) (`03_module_index.md`, `04_data_flow.md`,
   `06_known_errors.md`) and re-upload.
3. Bump `cfg.cache_version` in `config.py` if the change should invalidate
   on-disk caches, and note the bump in `06_known_errors.md`.

## What this Project is good for

- Quick Q&A on shapes, cache behaviour, stage order.
- Reviewing a proposed `part5_*` or new loss term against the invariants.
- Debugging "why is my cache stale?" — `06_known_errors.md` + `07_refactor_rules.md` cover most failure modes.

## What it is **not** good for

- Predicting numerical FC/PSD results without running the code.
- Tuning hyperparameters by inspection — the loss surface is empirical;
  the model can only relay defaults and the notebook's overrides.
- Debugging `tvboptim` internals. That library is not in the upload set.
