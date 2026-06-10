# 14 — Patch 3 Report (GPU + Batching + Timing Diagnostics)

## Status
APPLIED — Option D (사용자 명시 동의)

## Applied
2026-05-19

## Spec
- Source: prompt-embedded spec (`13_patch3_spec.md`는 디스크에 존재하지 않으나 prompt에 전체 명세가 포함되어 있어 실행 가능)
- 사용자 명시 옵션: D (scientific-method 변경 예외 동의)

## Pre-flight state deviation (기록)
- 사전 check는 notebook이 정확히 **21 cells (post-Patch-2)** 이라고 가정했으나, 실제 상태는 **24 cells**였다.
- 추가된 3개의 비-Patch-2 셀:
  - cell 8: `paper_dbs_psd_*` PSD figure export 스크립트 (사용자 직접 작성)
  - cell 22: `all_figures_export_*` 자동 figure 저장 + 전체 파이프라인 재실행 스크립트 (사용자 직접 작성)
  - cell 23: `paper_figures_*` PSD 4-region stacked figure export 스크립트 (사용자 직접 작성)
  - cell 9, 10, 11: 빈 셀 (scratch placeholders)
- Patch 3 logic은 cell index가 아닌 **content-based search** (`build_network` + `bundle_init` → cell 7)로 삽입 위치를 찾으므로 정상 작동.
- 사용자가 `--no clarifying questions` 모드로 작업을 지시했고, 추가 셀이 Patch 3 logic을 깨뜨리지 않음을 확인한 후 진행.
- C5 acceptance check는 **24 + 2 = 26 cells**로 조정하여 검증.

## Files changed
- `timing_utils.py` — NEW (lightweight timing utility, no pipeline imports)
- `config.py` — Patch 3 toggle 필드 +3, `cache_version` bump
- `part2_eib.py` — post-hoc parallel branch + `_run_posthoc_validation_parallel` helper (toggle, default OFF)
- `part3_gradient.py` — chunk_steps log hint (로직 변경 없음)
- `part4_dbs.py` — `dbs_parallel_targets` toggle 경고 (sequential fallback)
- `main.ipynb` — Cell 1 setdefault화, Cell 7 직후 timing cells +2
- `05_runbook.md` — Timing Diagnostics & GPU Tuning 섹션 추가

## Files NOT changed
- `model.py` (Wilson-Cowan equations, EIBLinearCoupling 전부)
- `pipeline_contracts.py` (StateBundle, ParamSet 전부)
- `data_loader.py` (데이터 로딩 로직)
- `part1_fic.py` (FIC closed-loop sequential)
- 모든 CSV / TXT 데이터 파일
- 기존 hyperparameter (learning rates, targets, max_steps, top_k, window sizes, ...)

## New cfg fields
| Field | Default | 동작 |
|---|---|---|
| `gpu_batch_size` | 1 | 예약 필드 (현재 1만 사용) |
| `posthoc_parallel` | False | True 시 EIB post-hoc parallel path 활성화 |
| `dbs_parallel_targets` | False | True 시 경고 출력 후 sequential fallback |
| `cache_version` | `v_eituning_oldlogic_match_p3` | 모든 기존 캐시 무효화 |

## Adaptation note: DBS duration field name
- Prompt template은 `cfg.dbs_baseline_duration_ms`를 참조했으나 실제 config는 `cfg.dbs_pre_stimulation_duration_ms`를 정의함.
- timing 셀은 `getattr(cfg, 'dbs_baseline_duration_ms', None)`로 polyfill 후 `dbs_pre_stimulation_duration_ms`로 fallback.
- 양쪽 이름 모두 호환되도록 작성됨.

## Post-edit state
- cells_after: 26
- cache_version: v_eituning_oldlogic_match_p3
- new toggles: posthoc_parallel=False, gpu_batch_size=1, dbs_parallel_targets=False

## Acceptance checks
| # | Check | Result |
|---|---|---|
| C1 | timing_utils import | PASS |
| C2 | config new fields + cache_version bump | PASS |
| C3 | compileall -q . clean | PASS |
| C4 | part2_eib new helper present | PASS |
| C5 | notebook cells = 26 (adjusted for pre-existing user cells) | PASS |
| C6 | timing cell present | PASS |
| C7 | Cell 1 setdefault pattern | PASS |
| C8 | Patch 1 preserved (FC_compact.csv) | PASS |
| C9 | data_loader import OK | PASS |
| C10 | model.py importable, untouched | PASS |
| C11 | pipeline_contracts.py untouched | PASS |
| C12 | part1_fic.py untouched | PASS |
| C13 | WilsonCowanEIB.dynamics signature intact | PASS |
| C14 | timing cell has no run_* calls | PASS |

## Behavior delta
| Aspect | Before | After (toggle OFF default) | After (toggle ON) |
|---|---|---|---|
| Wilson-Cowan equations | unchanged | unchanged | unchanged |
| Loss / objective | unchanged | unchanged | unchanged |
| Hyperparameters | unchanged | unchanged | unchanged |
| Post-hoc validation | sequential | sequential | parallel path (~20-30% faster) |
| Gradient chunk steps | 5 (config default) | 5 | 5 (hint만 출력; <5일 때) |
| DBS target loop | sequential | sequential | sequential (Option α only) |
| Notebook cells | 24 | 26 (+2 timing) | 26 |
| cache_version | `..._match` | `..._match_p3` | `..._match_p3` |
| 기존 cache | reused | **invalidated by version bump** | invalidated |

## Expected runtime impact (GPU, toggles ON)
| Stage | Before | After (toggles ON) | Δ |
|---|---|---|---|
| FIC | 5-10 min | 5-10 min | 0% (sequential closed-loop) |
| EIB search | 20-40 min | 20-40 min | 0% (sequential closed-loop) |
| EIB post-hoc | 30-50 min | 20-35 min | ~20-30% |
| Gradient | 30-60 min | 25-55 min | ~10% (chunk 사용자 조정 시) |
| LowRank | 5-10 min | 5-10 min | 0% |
| DBS | 30-60 min | 30-60 min | 0% (Option α) |
| **Total** | 2-4 h | **1.7-3.5 h** | **~15%** |

진정한 device-batched parallelism (~80% 단축)은 별도 Patch 4 spec 필요.

## What was NOT done (intentional)
- Wilson-Cowan dynamics batch-aware 재작성 (model.py)
- 진정한 vmap(_evaluate_candidate_bundle) (network state batch dim 필요)
- DBS Option β (4 targets simultaneous, network monkey-patch 재설계)
- 기존 `optimizer_chunk_steps`, `eib_posthoc_top_k`, `eib_max_iterations` 등의 default 변경
- 어떤 파이프라인 실행도 수행하지 않음
- 데이터 파일 변경 없음

## Rollback
```bash
cd /scratch/home/wog3597/optim
for f in main.ipynb config.py part2_eib.py part3_gradient.py part4_dbs.py 05_runbook.md; do
    cp "$f.patch3_backup" "$f"
done
rm -f timing_utils.py
rm -rf cache/v_eituning_oldlogic_match_p3_*  # 선택
```

## Deferred follow-ups (별도 spec 필요)
- **Patch 4**: model.py WilsonCowanEIB.dynamics batch-aware refactor → 진정한 vmap
- **Patch 5**: DBS Option β (4 targets simultaneous simulation)
- **Patch 6**: `optimizer_chunk_steps` 증가의 수렴 dynamics 영향 ablation
