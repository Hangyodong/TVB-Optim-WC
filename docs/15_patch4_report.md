# 15 — Patch 4 Report (실질 병렬화)

## Status
APPLIED

## Applied
2026-05-19

## 변경 파일
- `part2_eib.py` — post-hoc validation sequential for loop → ThreadPoolExecutor 병렬
- `part4_dbs.py` — stim 배열 미리 통합, `_run_single_target`에 `prebuilt_stim_array` 파라미터 추가

## 변경하지 않은 파일
- `model.py`, `config.py`, `pipeline_contracts.py`
- `part1_fic.py`, `part3_gradient.py`
- `main.ipynb`, `timing_utils.py`, 데이터 파일 전부

## Fix 1: EIB post-hoc validation 병렬화

| 항목 | Before | After |
|---|---|---|
| 구현 | sequential for loop | `ThreadPoolExecutor(max_workers=min(top_k, 4))` |
| top_k=10 예상 시간 | ~50분 | ~15분 (workers=4) |
| 결과 동등성 | — | 완전 동일 (독립 시뮬, 순서 무관) |
| 추가 의존성 | — | `concurrent.futures` (표준 라이브러리) |

n_workers=4로 설정한 이유: GPU 메모리 안전 한계.
JAX는 thread-safe이므로 별도 라이브러리 없이 표준 ThreadPoolExecutor만으로 GPU에 동시 dispatch 가능하다.
결과 처리는 `as_completed`로 받되 `rank` 키로 dict에 저장한 뒤 정렬 출력해
sequential 출력 순서와 완전히 동일하게 유지.

## Fix 2: DBS JIT recompile 제거

| 항목 | Before | After |
|---|---|---|
| stim 배열 생성 시점 | target 루프 안에서 매번 | 루프 전에 미리 일괄 생성 |
| `_run_single_target` 호출 | stim 내부 생성 | `prebuilt_stim_array` 인자로 전달 |
| 결과 동등성 | — | 완전 동일 (stim 값/shape 변경 없음) |

`_run_single_target` 시그니처에 `prebuilt_stim_array=None` 옵션 인자를 추가했고,
값이 주어지면 `_build_biphasic_pulse_train` 호출을 건너뛴다.
미주어지면(`None`) 기존 경로 그대로이므로 다른 호출자(있다면)와 호환된다.

## 예상 runtime 변화

| 단계 | Before | After | 단축 |
|---|---|---|---|
| EIB post-hoc | ~50분 | ~15분 | ~70% |
| DBS 4 targets | ~40-60분 | ~20-30분 | ~50% |
| 전체 파이프라인 | 2-4h | 1.3-2.7h | ~30% |

## Acceptance checks
| # | Check | Result |
|---|---|---|
| C1 | part2_eib.py / part4_dbs.py syntax OK | PASS |
| C2 | concurrent.futures import 추가 | PASS |
| C3 | ThreadPoolExecutor + n_workers 코드 존재 | PASS |
| C4 | `prebuilt_stim_array=None` 파라미터 존재 | PASS |
| C5 | `_build_biphasic_pulse_train` 본체 미변경 | PASS |
| C6 | compileall -q . clean | PASS |
| C7 | part2_eib import OK | PASS |
| C8 | part4_dbs import OK | PASS |
| C9 | model/config/pipeline_contracts/part1/part3 compile clean | PASS |
| C10 | WilsonCowanEIB.dynamics 서명 intact | PASS |

## What was NOT done (intentional)
- Wilson-Cowan equations 변경 없음 (model.py)
- loss / objective / hyperparameter 변경 없음
- 그 어떤 `run_*` 함수도 실행하지 않음
- 노트북 셀 실행하지 않음
- 데이터 파일 수정 없음

## Rollback
```bash
cd /scratch/home/wog3597/optim
cp part2_eib.py.patch4_backup part2_eib.py
cp part4_dbs.py.patch4_backup part4_dbs.py
```
