# 10 — Patch 1 Spec

**Option chosen: #3 — fix the `FC.csv` vs `FC_compact.csv` path mismatch
in the notebook.**

This is the smallest patch the evidence in 06 and 09 supports, and the only
one that touches a confirmed (not hypothetical) bug.

---

## Goal

Align `main.ipynb` Cell 3 with the FC matrix actually shipped in the repo, so
`data_loader.load_data` finds the file by its real name instead of relying on
the silent fallback inside `_resolve_path`.

One-character class of change: replace the string `"FC.csv"` with
`"FC_compact.csv"` in exactly one location.

## Why (evidence)

- `09_baseline_smoke_report.md` → "Dependency probe" / "Config instantiation":
  default `Config.fc_csv = "FC_compact.csv"` matches the file shipped; the
  notebook Cell 3 sets `fc_csv = "FC.csv"`, which does not exist on disk.
- `06_known_errors.md` § "Colab compatibility risks" #6: the mismatch is
  papered over by `data_loader._resolve_path`, which prints
  `[DATA] FC.csv not found → using FC_compact.csv`. The fallback is fragile —
  if a future user uploads only their own `FC.csv` (a different matrix), the
  cache_tag rotates silently and downstream stages may pick wrong cached
  results.
- `04_data_flow.md` § "Input Files": `FC_compact.csv` is the canonical 42×42
  empirical FC target.
- `07_refactor_rules.md` § "When you change `Atlas_43.txt` or the CSVs"
  permits filename edits as long as shape and corner bytes don't move. This
  patch does not touch the files themselves, so corner bytes are unchanged
  and the cache_tag does not rotate.

## Scope

**Single file edited:** `main.ipynb`, one cell, one string literal.

**Files NOT edited:** any `.py` source, any `.csv`, `Atlas_43.txt`, all `0*_*.md`
docs except a one-line cross-reference in `06_known_errors.md` (optional —
see "Optional doc follow-up" below; treat that as out-of-scope for Patch 1
unless the user opts in).

## Exact change

In `main.ipynb`, Cell 3 (the only `Config(...)` constructor in the notebook):

```diff
 cfg = Config(
     # ── 데이터 경로 ──────────────────────────────────────────
     region_txt                          = "Atlas_43.txt",
     sc_csv                              = "weight.csv",
     length_csv                          = "tract_length.csv",
-    fc_csv                              = "FC.csv",
+    fc_csv                              = "FC_compact.csv",
```

Everything else in Cell 3 stays byte-identical (including all whitespace,
comments, and the `cfg.print_summary()` call at the end).

How the edit will be made when Patch 1 is approved:

- Use the `Edit` tool against `main.ipynb`.
- `old_string`: the JSON-quoted source-cell line as it appears in the
  notebook's `source` array, including the surrounding context needed for
  uniqueness:
  ```
  "    length_csv                          = \"tract_length.csv\",\n",
  "    fc_csv                              = \"FC.csv\",\n",
  ```
- `new_string`: identical except `"FC.csv"` → `"FC_compact.csv"`.
- `replace_all`: false (single occurrence expected; will fail loudly if not).

No re-execution of cells. No notebook output cleared. The patch is purely a
source-cell text edit; the embedded `outputs` arrays stay as-is.

## Behavior delta

| Aspect | Before | After |
|---|---|---|
| Which file `load_data` reads | `FC_compact.csv` (via fallback) | `FC_compact.csv` (directly) |
| Stdout from data load | prints `"[DATA] FC.csv not found → using FC_compact.csv"` | no fallback print |
| `cache_tag` value | `v_..._N42_sc<10hex>_fc<10hex>` (computed from FC bytes) | **identical** (same bytes) |
| All on-disk caches under `cache/<cache_tag>/` | preserved | **preserved** |
| Numerics, losses, equations, constants | unchanged | unchanged |
| Long-run behavior, runtime budget | unchanged | unchanged |
| What happens if a user drops a different `FC.csv` next to `FC_compact.csv` | silently used instead → cache_tag rotates | ignored; canonical `FC_compact.csv` used |

Nothing the docs warn about ("equations, objectives, numerical constants,
data files, or long-run behavior") is touched.

## Rollback

```diff
-    fc_csv                              = "FC_compact.csv",
+    fc_csv                              = "FC.csv",
```

Single-line revert. No state to clean up, no cache to invalidate, no derived
artefacts to regenerate.

## Acceptance check (no heavy run)

Same lightweight checks as `09_baseline_smoke_report.md`, plus one
notebook-level grep. None of these execute the pipeline.

```bash
cd /scratch/home/wog3597/optim

# 1. Notebook still valid JSON, same cell count.
python -c "import json; nb=json.load(open('main.ipynb')); print(len(nb['cells']))"
# expect: 24

# 2. The new string appears in Cell 3 exactly once, the old one is gone.
python -c "
import json
nb = json.load(open('main.ipynb'))
src = ''.join(nb['cells'][3]['source'])
assert 'fc_csv' in src
assert '\"FC_compact.csv\"' in src
assert '\"FC.csv\"' not in src
print('cell-3 fc_csv string updated OK')
"

# 3. Compileall still clean.
python -m compileall -q .

# 4. data_loader still imports.
python -c "import data_loader; print('OK')"

# 5. Spot-confirm the shipped file is the one being pointed at.
ls -la FC_compact.csv FC.csv 2>&1 | head -5
# expect: FC_compact.csv exists; FC.csv does not (ls returns "No such file" for FC.csv)
```

Pass = first four print as expected; fifth confirms only `FC_compact.csv` is
present. No pipeline cell is run.

## Out of scope (explicitly)

- No changes to equations or hyperparameters in Cell 3. Specifically: every
  field other than `fc_csv` keeps its existing notebook value, including the
  overrides that diverge from `config.py` defaults (`optimizer_max_steps=1000`,
  `optimizer_chunk_steps=1`, `optimizer_bold_window_tr=180`, etc.). These are
  the long-run-behavior knobs the user said not to touch.
- No edits to `data_loader._resolve_path`. The fallback stays in place as a
  defensive net; we're just no longer leaning on it.
- No CSV renames or content edits.
- No new dependencies, no new imports, no new cells.
- No addition of a smoke/debug cell (that was Option 1; we deliberately chose
  the narrower Option 3 to keep the diff minimal).

## Optional doc follow-up (NOT part of Patch 1)

After Patch 1 lands, the wording in `06_known_errors.md` § "Colab
compatibility risks" item #6 will become slightly stale (the mismatch is no
longer present in the notebook). Suggested edit, deferred to a later patch:

```diff
- 6. **`fc_csv` mismatch (notebook vs repo).** Cell 3 sets `fc_csv = "FC.csv"`,
-    but the file shipped is `FC_compact.csv`. ...
+ 6. **`fc_csv` resilience.** `data_loader._resolve_path` still falls back from
+    `fc_csv` to `"FC_compact.csv"` if the configured path is missing. Notebook
+    Cell 3 now points directly at `FC_compact.csv` (Patch 1), but the fallback
+    remains for users who edit Cell 3 to point elsewhere — be aware that the
+    fallback silently swaps files without rotating the cache_version, only the
+    cache_tag (via FC corner bytes).
```

Track this as "patch 1.1, doc-only" — do not bundle it with Patch 1, because
the user asked for one minimal change.

---

**Do not apply this patch until the user explicitly says: Proceed with Patch 1.**
