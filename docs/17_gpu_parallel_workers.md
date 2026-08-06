# 17 — GPU N-Worker 병렬 실행

**결론: GPU 1개에 optim 프로세스를 여러 개 동시에 돌릴 수 있다.** GPU당 1개 제한은 없다.
`run_all_pd.sh` 가 이미 범위 인자를 받으므로 **새 코드 없이** N워커 병렬이 가능하다.

측정일: 2026-07-15 / 장비: NVIDIA A10 23GB / 대상: `main_ppmi_pd.py --fic-only` (Part1 FIC 2000 step 완주)

---

## 1. 실측 결과

| 동시 워커 | 개당 소요 | 처리량 | GPU 메모리 | 결과 |
|---|---|---|---|---|
| 2개 (idx 2,3) | 21분 06초 | 0.095개/분 | 1.14GB × 2 = 2.3GB | 정상, 에러 0 |
| 4개 (idx 4~7) | 28분 14초 | 0.142개/분 | 1.10GB × 4 = 4.7GB | 정상, 에러 0 |

- **4워커가 2워커 대비 처리량 1.5배.** 개당 시간은 21→28분(+34%)이나 총 처리량은 증가.
- **GPU util: 4워커에서도 평균 39% (peak 90%).** → 6~8워커까지 처리량이 더 오를 여지 있음.
- 4워커 전부 `Pipeline complete`, FIC 정상 수렴 (mean c_ei 1.21~1.25).

## 2. 왜 가능한가

1. **`compute_mode=Default`** — `Exclusive_Process` 가 아니므로 프로세스 수 제한이 없다.
   (`nvidia-smi --query-gpu=compute_mode --format=csv` 로 확인)
2. **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** (`main_ppmi_pd.py` 상단) — JAX 가 GPU 메모리를
   선점하지 않고 **프로세스당 실사용분 ~1.1GB 만** 잡는다. **이 설정이 병렬의 핵심이다.**
   같은 GPU 에서 실측 비교:

   | 설정 | 프로세스 1개가 잡는 메모리 | 23GB 에 몇 개 |
   |---|---|---|
   | `PREALLOCATE=true` (JAX 기본) | **17218 MiB (=75% 선점)** | 1개 |
   | `PREALLOCATE=false` (이 repo) | **~1100 MiB (실사용분만)** | ~20개 |

   기본값대로 두면 첫 프로세스가 75% 를 선점해 2번째가 메모리를 못 잡는다.
   `main_ppmi_pd.py` 는 `os.environ.setdefault` 로 이미 false 를 넣으므로 그대로 쓰면 된다.
3. **116노드 모델이 작다** — 1개만 돌리면 GPU 가 논다(util 39% @ 4워커). 병목은 메모리가 아니라
   컴퓨트 공유이며, 메모리는 23GB / 1.1GB ≈ 20개분으로 사실상 제약이 아니다.

## 3. 실행 방법

`run_all_pd.sh <GPU> <START> <END>` 를 범위만 나눠 여러 개 띄운다.

```bash
cd /scratch/home/wog3597/optim

# 4워커 병렬 (idx 2~241 분할)
bash run_all_pd.sh 0   2  60  > output_ppmi_pd/_w1.log 2>&1 &
bash run_all_pd.sh 0  61 120  > output_ppmi_pd/_w2.log 2>&1 &
bash run_all_pd.sh 0 121 180  > output_ppmi_pd/_w3.log 2>&1 &
bash run_all_pd.sh 0 181 241  > output_ppmi_pd/_w4.log 2>&1 &
```

PBS 로 제출할 때는 job 하나 안에서 위처럼 여러 워커를 띄운다
(`qsub -v START=,END= qsub_pd.sh` 는 워커 1개짜리 범위 지정용).

### 워커 안전성 — 범위는 반드시 겹치지 않게

- 완료 marker 가 `output_ppmi_pd/_logs/s{idx}.done` 로 **idx 별**이고,
  로그도 `s{idx}.log` 라 **범위가 안 겹치면 워커끼리 충돌하지 않는다.**
- 캐시도 `cache_version = v_pd{parc}_tr25_s{idx}` 로 idx 가 들어가 섞이지 않는다.
- **범위를 겹치면 같은 idx 를 두 워커가 동시에 잡는다.** marker 는 완료 *후* 에 생성되므로
  실행 중인 idx 를 다른 워커가 skip 하지 않는다(race). 분할은 필수.
- 워커별 stdout 은 다른 파일로 리다이렉트할 것 (`_w1.log`, `_w2.log`, …). `_batch.log` 하나로
  몰면 섞인다.

## 4. 주의사항

- **위 수치는 FIC 기준이다.** EIB / Part3(full backprop) 는 메모리를 더 쓴다.
  `PART3_REMAT_SCAN=1` (기본 ON) 이 scan-body checkpointing 으로 완화하지만 **미측정**.
  → 전체 파이프라인 병렬은 **4워커로 시작해 `nvidia-smi` 로 확인 후 늘릴 것.**
- **노드마다 GPU 가 다르다.** 대화형 세션 = A10 23GB, PBS 배치가 잡은 노드 = H100 NVL 94GB.
  H100 을 받으면 메모리 여유는 약 4배.
- 모니터링:
  ```bash
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
  ```

## 5. 관련 문서

- `05_runbook.md` — 환경/의존성
- `run_all_pd.sh` — 순차 배치 + resume marker
- `qsub_pd.sh` — PBS 제출 (`-v GPU=,START=,END=`)
