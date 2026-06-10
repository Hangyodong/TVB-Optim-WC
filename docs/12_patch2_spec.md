# 12 — Patch 2 Spec

**Issue addressed: 3 empty code cells in `main.ipynb` (cells 9, 10, 11).**
This is the only outstanding item flagged in `09_baseline_smoke_report.md`'s
"Overall" section after Patch 1 closed the FC path mismatch.

(Note: the user referenced `11_patch1_result.md`; the actual file written
by Patch 1 is `11_patch1_report.md`. Same content, just a filename
difference.)

---

## Goal

Delete the three empty code cells (current 0-indexed positions 9, 10, 11)
from `main.ipynb`. After the patch, the notebook contains 21 cells instead
of 24. No source-bearing cell is added, modified, reordered, or has its
outputs touched.

## Why (evidence)

- `09_baseline_smoke_report.md` § "Notebook JSON validity":
  `empty_code_cells: 3` — matches the cells documented in `02_repo_tree.md`'s
  cell map as positions 9–11.
- `09_baseline_smoke_report.md` § "Overall" (final paragraph):
  > The only environment-side caveat: **3 empty code cells** (Cells 9–11) in
  > the notebook would noisily appear in `jupyter nbconvert --to script`
  > output. Not a functional risk, just a tidy-up note for the upload guide
  > in `08`.
- The earlier `Read main.ipynb` output during Patch 1 application confirmed
  the three cells' JSON form: `<cell id="…"></cell id="…">` with no source.
  Their cell IDs are stable across the file:
  - `b0858d6a-e0ce-423b-899e-dc4557bc93b9` (current index 9)
  - `50ed008a-cb44-487b-be62-da3500b61cf1` (current index 10)
  - `f2dda7d0-f698-497c-a7d7-096261db8360` (current index 11)
- `07_refactor_rules.md` § "Soft conventions to keep" does not protect empty
  cells; the only notebook-specific rule is "Plotting always uses `matplotlib`
  + `plt.show()`" — irrelevant here.
- `07_refactor_rules.md` § "Style of edits": "Don't reformat unrelated lines
  while making a fix." Honored — we delete three empty cells and touch nothing
  else.

## Scope

**Single file edited:** `main.ipynb`. Three cell deletions, by cell ID, so the
operations are robust against intermediate index shifts.

**Files NOT edited:**
- Any `.py` source.
- Any `.csv` or `.txt` data file.
- Any `0*_*.md` doc (including `02_repo_tree.md`'s cell map — its
  post-deletion staleness is the optional doc follow-up below; explicitly
  deferred from Patch 2).
- The `06_known_errors.md` Patch 1 follow-up (still deferred).

## Exact change

Three `NotebookEdit` calls, in this order, each with `edit_mode=delete`:

```text
NotebookEdit(
  notebook_path = "/scratch/home/wog3597/optim/main.ipynb",
  cell_id       = "b0858d6a-e0ce-423b-899e-dc4557bc93b9",
  edit_mode     = "delete",
)
NotebookEdit(
  notebook_path = "/scratch/home/wog3597/optim/main.ipynb",
  cell_id       = "50ed008a-cb44-487b-be62-da3500b61cf1",
  edit_mode     = "delete",
)
NotebookEdit(
  notebook_path = "/scratch/home/wog3597/optim/main.ipynb",
  cell_id       = "f2dda7d0-f698-497c-a7d7-096261db8360",
  edit_mode     = "delete",
)
```

Each call must succeed independently. If any call cannot locate its cell ID,
STOP — do not retry with `cell_number` indices (they shift after each
deletion and ID-based addressing is the explicit guard against that).

## Pre-edit checks

```bash
cd /scratch/home/wog3597/optim

# A. Repo still intact (sanity).
ls main.ipynb 12_patch2_spec.md

# B. Snapshot cell count and the three target cells' state.
python -c "
import json
nb = json.load(open('main.ipynb'))
print('cells_before:', len(nb['cells']))
target_ids = {
    'b0858d6a-e0ce-423b-899e-dc4557bc93b9',
    '50ed008a-cb44-487b-be62-da3500b61cf1',
    'f2dda7d0-f698-497c-a7d7-096261db8360',
}
empties = 0
for c in nb['cells']:
    if c.get('id') in target_ids:
        src = ''.join(c.get('source', []))
        assert c['cell_type'] == 'code', f'{c[\"id\"]} is not a code cell'
        assert src.strip() == '', f'{c[\"id\"]} source is not empty: {src!r}'
        empties += 1
assert empties == 3, f'expected to find 3 target cells, found {empties}'
print('pre-edit OK: 3 empty code cells confirmed by ID')
"
```

If A or B fails (any target ID missing, not a code cell, or non-empty source),
STOP and report — do not attempt the deletions.

## Behavior delta

| Aspect | Before | After |
|---|---|---|
| Notebook cell count | 24 | 21 |
| Empty code cells | 3 | 0 |
| Non-empty cell IDs preserved | — | all preserved (only deletions touch the 3 target IDs) |
| `nbconvert --to script` output | includes 3 blank stanzas | no blank stanzas |
| Cell 3 (`Config`) source | unchanged from Patch 1 | unchanged |
| Source of any non-target cell | unchanged | unchanged |
| Outputs arrays of non-target cells | unchanged | unchanged |
| Equations, losses, constants, data | unchanged | unchanged |
| `cache_tag` | unchanged (no SC/FC bytes touched) | unchanged |
| On-disk caches under `cache/<cache_tag>/` | preserved | preserved |
| Long-run behavior of any pipeline stage | unchanged | unchanged |

The deletions cannot affect runtime because the cells contain no source. They
exist purely as JSON placeholders.

## Rollback

Three `NotebookEdit` calls in `edit_mode=insert` against `main.ipynb`,
inserting empty code cells at positions 9, 10, 11 (after the current cell 8).
The restored cells will receive fresh cell IDs — the original IDs are not
preserved by the notebook format on re-insert, but that's cosmetic (no other
file references those IDs).

If exact ID preservation matters, an alternative rollback is a git revert of
the single `main.ipynb` change (recommended if the user is using version
control on the notebook).

## Acceptance checks (lightweight only — no pipeline execution)

```bash
cd /scratch/home/wog3597/optim

# 1. Notebook still valid JSON; cell count dropped by exactly 3.
python -c "import json; nb=json.load(open('main.ipynb')); print('cells_after:', len(nb['cells']))"
# expect: 21

# 2. The three target cell IDs are gone; no empty code cell remains.
python -c "
import json
nb = json.load(open('main.ipynb'))
target_ids = {
    'b0858d6a-e0ce-423b-899e-dc4557bc93b9',
    '50ed008a-cb44-487b-be62-da3500b61cf1',
    'f2dda7d0-f698-497c-a7d7-096261db8360',
}
ids = {c.get('id') for c in nb['cells']}
assert target_ids.isdisjoint(ids), 'target cell IDs still present'
empties = sum(
    1 for c in nb['cells']
    if c['cell_type']=='code' and ''.join(c.get('source', [])).strip() == ''
)
assert empties == 0, f'still {empties} empty code cells'
print('empty-cell deletions OK')
"

# 3. Cell 3 (Config) is byte-identical to post-Patch-1 state.
python -c "
import json
nb = json.load(open('main.ipynb'))
src = ''.join(nb['cells'][3]['source'])
assert 'fc_csv' in src
assert '\"FC_compact.csv\"' in src
assert '\"FC.csv\"' not in src
print('cell-3 Config unchanged (Patch 1 still intact)')
"

# 4. Compileall still clean across all .py files.
python -m compileall -q .

# 5. data_loader still imports.
python -c "import data_loader; print('data_loader import OK')"

# 6. Non-empty cell count is preserved (24 - 3 empty = 21; should match cells_after).
python -c "
import json
nb = json.load(open('main.ipynb'))
n_md = sum(1 for c in nb['cells'] if c['cell_type']=='markdown')
n_code_nonempty = sum(1 for c in nb['cells'] if c['cell_type']=='code' and ''.join(c.get('source', [])).strip())
print(f'md={n_md}  code_nonempty={n_code_nonempty}  total={n_md + n_code_nonempty}')
assert n_md == 9, f'markdown cell count changed: {n_md}'
assert n_code_nonempty == 12, f'non-empty code cell count changed: {n_code_nonempty}'
print('cell-class breakdown unchanged')
"
```

Pass criteria:
- Step 1 prints `cells_after: 21`.
- Step 2 prints `empty-cell deletions OK`.
- Step 3 prints `cell-3 Config unchanged (Patch 1 still intact)`.
- Step 4 emits no errors (silent on success).
- Step 5 prints `data_loader import OK`.
- Step 6 prints `md=9  code_nonempty=12  total=21` then `cell-class breakdown unchanged`.

If any check fails, STOP and report which one, with raw stderr. Do NOT
escalate to running notebook cells, importing JAX/tvboptim, or executing
any `run_*` function. The lightweight checks are definitive for this patch.

## Out of scope (explicitly)

- No edits to any source-bearing cell. The Cell 8 ordering quirk
  (PSD figure helper sits before Part 4 produces data) is NOT addressed
  here — that cell already handles the missing-directory case gracefully
  and a fix would either move the cell (index churn) or add logic (scope
  creep). Defer.
- No deletion of the `# 06_known_errors.md` Patch 1 follow-up paragraph
  (still deferred).
- No update to `02_repo_tree.md`'s "Notebook Cell Map" table to reflect
  the post-deletion index shift — see "Optional doc follow-up" below.
- No new files. No new dependencies. No new cells. No cell reordering.
- No changes to any `.py`, `.csv`, or `.txt` file.
- No execution of `run_fic`, `run_eib`, `run_gradient_optimization`,
  `run_lowrank_optimization`, or `run_dbs_stimulation`.

## Optional doc follow-up (NOT part of Patch 2)

After Patch 2 lands, `02_repo_tree.md` § "Notebook Cell Map (what runs
where)" will show stale indices for the cells that previously sat at
positions 12 through 23 (post-deletion they shift to 9 through 20). The
user-visible markdown headers inside the notebook ("Cell 5 — Part 1: FIC",
etc.) are unaffected, so the human-facing numbering still reads cleanly.

Suggested follow-up, deferred to a later patch (call it "patch 2.1,
doc-only"):

```diff
- | 9–11 | code | Empty placeholders |
- | 12 | md | — |
- | 13 | code | `bundle_fic = run_fic(...)` |
- | 14 | md | — |
- | 15 | code | `bundle_eib = run_eib(...)` |
- | 16 | md | — |
- | 17 | code | `bundle_grad = run_gradient_optimization(...)` |
- | 18 | md | — |
- | 19 | code | `bundle_lowrank = run_lowrank_optimization(bundle_in=bundle_grad, ...)` |
- | 20 | md | — |
- | 21 | code | `run_dbs_stimulation(bundle_in=bundle_lowrank or bundle_grad, ...)` |
- | 22 | code | **Re-runs Parts 1→4 end-to-end** ... |
- | 23 | code | 4-row stacked PSD plot ... |
+ | 9 | md | — |
+ | 10 | code | `bundle_fic = run_fic(...)` |
+ | 11 | md | — |
+ | 12 | code | `bundle_eib = run_eib(...)` |
+ | 13 | md | — |
+ | 14 | code | `bundle_grad = run_gradient_optimization(...)` |
+ | 15 | md | — |
+ | 16 | code | `bundle_lowrank = run_lowrank_optimization(bundle_in=bundle_grad, ...)` |
+ | 17 | md | — |
+ | 18 | code | `run_dbs_stimulation(bundle_in=bundle_lowrank or bundle_grad, ...)` |
+ | 19 | code | **Re-runs Parts 1→4 end-to-end** ... |
+ | 20 | code | 4-row stacked PSD plot ... |
```

Treat this as a separate, doc-only follow-up. Do not bundle it with
Patch 2.

---

**Do not apply this patch until the user explicitly says: Proceed with Patch 2.**
