# 18 — main_ppmi_pd.py 코드 분석

**결론: 파이프라인 알고리즘(FIC/EIB/Part3)은 참조 노트북 `EI_Tuning (11).ipynb`와 일치한다. 문제는 알고리즘이 아니라 그 아래 라이브러리 사용 계층에 있다.** 가장 큰 것 셋: ① **tract delay가 시뮬레이션에 전혀 반영되지 않는다**, ② **BOLD HRF 커널과 history 길이가 어긋나 Part1/2의 FC 측정이 무증상으로 깨진다**, ③ **Part3의 Adam 모멘트가 chunk마다 리셋된다**. 셋 다 에러를 내지 않고 조용히 틀린 값을 낸다.

분석일: 2026-07-16 / 대상: `main_ppmi_pd.py` + 구성 모듈 전체 / 방법: 정적 분석(로그인 노드, 실행 없음) + `tvboptim`/`equinox` 라이브러리 소스 대조 + 실제 산출물 확인

> **[업데이트 2026-07-20]** 발견 ③(S1-3, Part3 Adam 모멘트 chunk 리셋)은 이후 패치에서 **해결됨** — `part3_gradient.py:799 _optax_run_persist`가 opt_state를 청크 간 유지한다. 아래 §4 표 [3]·§S1-3·§7-3은 그 시점 기준 기록이며 **현재 코드엔 해당 없음**. S1-2(HRF)도 `make_bold_monitor` 단일화로 **코드 수정 적용**(2026-07-20, 재실행 검증 필요). **S1-1(delay)은 미사용 확정**(사용자 결정 2026-07-20) — 모델이 이미 delay-free라 동작 변경 없음, `model.py` docstring 정정 반영, 죽은 배선 정리는 선택. 구조/프레임워크 관점 분석은 `19_pipeline_framework_analysis.md` 참조.

---

## 1. 파이프라인 구조

`main_ppmi_pd.py`는 얇은 오케스트레이터다. 로직은 전부 모듈에 있다.

```
main_ppmi_pd.py          .mat → CSV 추출, Config 조립, Part 1→4 호출
├ config.py              @dataclass Config — 하이퍼파라미터 단일 소스
├ data_loader.py         CSV → weights/delays/fc_target/블록마스크, 그래프, 캐시경로
├ model.py               ReducedWongWangEIB + EIBLinearCoupling + warmup
├ pipeline_contracts.py  ★ ParamSet/StateBundle 계약 + 공용 loss
├ part1_fic.py           FIC (c_ei 튜닝)
├ part2_eib.py           EIB (wLRE/wFFI 튜닝)
├ part3_gradient.py      full backprop (+ Part3.5 σ)
└ part4_dbs.py           DBS 자극 (기본 skip)
```

**데이터 흐름**: `data/AALv3/TR_2.5_PD.mat`(242명) → subject 1명 → 168노드 중 `Thal_Re_L`(242명 전원 FC 결측) 제거 → **167노드**(cortex 116 / subcortex 51) → `output_ppmi_pd/<sub_num>/inputs/*.csv` 로 쓰고 `load_data`가 다시 읽는다. CSV 왕복은 캐시 fingerprint를 로드된 행렬 기준으로 잡기 위한 것으로 보인다.

**단계 간 전달**은 `StateBundle` 하나로 통일 — `params`(c_ei/wLRE/wFFI) + `init_dynamics` + `bold_history` + `bold_window` + `internal_state` + `delay_history`. 불변 갱신(`advance`)만 한다. **설계는 좋다. 문제는 이 필드 중 일부가 실제로는 아무 효과가 없다는 것**(§4 참조).

## 2. 모델

이름과 달리 **Wilson-Cowan이 아니라 Reduced Wong-Wang**이다. `WilsonCowanEIB = ReducedWongWangEIB` 별칭만 하위호환용으로 남았다.

- 상태 `E`/`I`는 실제로는 synaptic gating `S_e`/`S_i`.
- `params.c_ei` == RWW의 `J_i` (S_i→E 억제 가중). FIC의 per-node 조정 대상.
- 전달함수 `H(x)=x/(1−exp(−d·x))`는 x=0에서 제거가능 특이점 → `_rww_H`가 double-where로 극한 `1/d` 대체 (float32 NaN 전파 방지).
- `aux[2]=H_e`, `aux[3]=H_i` → Hz 진단용.

## 3. Part별 동작

| Part | 튜닝 대상 | 방법 | 규모 |
|---|---|---|---|
| 1 FIC | `c_ei` per-node | `Δc_ei = η·mean_S_i·(mean_S_e − 0.25)` | 2000 iter × 1 TR |
| 2 EIB | `wLRE`/`wFFI` | 240 TR rolling window FC → `Δw ∝ (FC_tgt−FC)·row_rmse` | 10000 iter × 1 TR |
| 3 Grad | c_ei + wLRE + wFFI | optax adamaxw full backprop | 250 step × 60만 step 적분 |
| 3.5 σ | per-node noise σ | backprop (c_ei/w 동결 → 축퇴 차단) | 기본 skip |
| 4 DBS | — | STN/GP biphasic 자극 | 기본 skip |

- **FIC 타깃은 Hz가 아니라 S_e gating 0.25**다. `fic_target_firing_rate_hz=2.0`은 진단 출력 전용.
- **Part3가 비용을 지배**한다: `240 TR × 2500ms = 600,000ms`, dt=1ms → 60만 step 적분을 250회 backprop. `PART3_REMAT_SCAN=1`(기본 ON)이 `jax.lax.scan` body를 `jax.checkpoint`로 감싼다. 없으면 OOM.
- **Part3 loss**: `0.8·block_corr + 0.2·block_rmse + 0.0·activity`. block_corr는 cc/cross/sub-sub을 edge 수와 무관하게 동등가중(0.4/0.4/0.2) — cortex-cortex edge가 압도적이라 subcortex가 안 맞는 문제를 강제 균형.

---

## 4. 발견 사항

| # | 심각도 | 문제 | 위치 |
|---|---|---|---|
| 1 | **S1** | tract delay 시뮬에 미반영 (ODE/SDE로 동작) | `model.py:131` |
| 2 | **S1** | ~~BOLD HRF 커널 20s vs history 5000 불일치 → Part1/2 FC 측정 붕괴~~ → **🔧 수정 적용(2026-07-20)** `make_bold_monitor` 단일화, 재실행 검증 필요 | `model.py:207`·`pipeline_contracts.py:575` |
| ~~3~~ | ~~S1~~ | ~~Adam 모멘트가 chunk마다 리셋~~ → **✅ 해결(2026-07-20)** `_optax_run_persist` | `part3_gradient.py:799` |
| 4 | S2 | adamaxw decay가 clip 전 raw value에 적용 | `part3_gradient.py:810` |
| 5 | S2 | `BoundedParameter`=hard clip → σ(low=0) gradient death | `part3_gradient.py:520` |
| 6 | S2 | `capture_internal_state()` 항상 None | `pipeline_contracts.py:527` |
| 7 | S2 | noise key 고정 → `prepare()`마다 동일 realization | `noise/base.py:43` |
| 8 | S3 | figure 중복 저장 O(n²) — **실물 확인** | `main_ppmi_pd.py:75` |
| 9 | S3 | `main_ppmi_pd_part3.py` 즉시 TypeError | `main_ppmi_pd_part3.py:79` |
| 10 | S3 | `--freeze-c-ei`가 Part3에 무효 | `part3_gradient.py:176` |
| 11 | S4 | Part4 DBS 4건 (기본 skip) | `part4_dbs.py` |

### S1-1. tract delay가 시뮬레이션에 전혀 반영되지 않는다 (확인)

파이프라인은 delay를 **계산하고, 그래프에 싣고, 행렬을 플롯하고, 모든 bundle과 pkl에 저장한다. 그리고 적분에는 쓰지 않는다.**

증거 체인:
- `model.py:131` — `class EIBLinearCoupling(InstantaneousCoupling)`.
- `coupling/base.py:551-599` — `InstantaneousCoupling.prepare()`는 `del t0, t1 # Unused`, docstring이 `dt: not used for instantaneous coupling`, 반환 `coupling_state: Empty Bunch (no internal state)`. `graph.delays`도 `network.get_history()`도 참조하지 않는다.
- `coupling/base.py:601-660` — `compute()`는 `summed = pre_states @ graph.weights`만. delay 없음.
- **`network.get_history(` 호출처는 라이브러리 전체에서 `coupling/base.py:813` 단 하나** — `DelayedCoupling.prepare()` 내부. 이 경로는 실행되지 않는다.

→ 모델은 DDE가 아니라 **delay-free ODE/SDE**다. `model.py:8`의 주석 *"DenseDelayGraph를 제공하면 tract delay가 자동으로 네트워크에 반영된다"*는 **거짓**이다.

死코드가 되는 것: `data_loader.py:55`(`delays = lengths/speed`), `:87`(`DenseDelayGraph`), `:247`(delay 행렬 플롯), `main_ppmi_pd.py:276`(`tract_conduction_speed=3.0`), `capture/restore/sync_network_delay_history`, `_delay_tail_len`, 모든 pkl의 `delay_history`.

**부수 효과**: "캐시 hit 시 delay history가 어긋나 결과가 달라진다"는 우려는 **실효 없음**으로 정정된다 — 애초에 delay를 안 쓰므로.

고치려면 `EIBLinearCoupling`을 `DelayedCoupling` 상속으로 바꿔야 한다(`pre`/`post` 시그니처 확인 필요). 그대로 두려면 delay 관련 코드·주석을 걷어내고 "지연 없음"을 명시해야 한다. **어느 쪽이든 과학적 주장에 직결되므로 결정이 필요하다.**

### S1-2. BOLD HRF 커널과 history 길이 불일치 (~~확인~~ → 🔧 수정 적용 2026-07-20)

> **수정 적용됨**: `build_network`와 각 Part가 `pipeline_contracts.make_bold_monitor(cfg, history)` 단일 소스로 Bold를 만들도록 통합(`model.py:207`, `pipeline_contracts.py:575`·`:324`). 이제 warmup 모니터도 cfg 32s 커널을 써서 `bold_history`가 8000샘플(=kernel_samples)로 잡혀 파이프라인 모니터와 정렬된다. **코드만 반영 — FC 절대값 교정은 subject 재실행으로 검증 필요(로그인 노드 시뮬 미실행).** 아래 원문은 수정 전 기록이다.

`model.py:205-210`의 Bold 생성이 **`cfg.bold_hrf_*` 6개를 전부 무시**한다:

```python
bold_monitor = Bold(period=cfg.bold_repetition_time_ms,
                    downsample_period=4.0, voi=0, history=warmup_result)   # kernel= 없음
```

→ 기본 `LotkaVolterraHRFKernel(duration=20_000)` → `_process_history`가 `ceil(20000/4)=5000` 샘플로 trim → `bundle_init.bold_history` = **(5000, 1, 167)**.

반면 `pipeline_contracts.py:328` `build_bold_monitor`는 `duration=cfg.bold_hrf_duration_ms=32_000` → **`kernel_samples=8000`**.

- `Bold.__call__`은 `convolution_mode="valid"`이고 `history is None`일 때 정확히 `zeros(kernel_samples)`를 prepend한다 → **`len(history)==kernel_samples`가 설계 전제**.
- `pipeline_contracts.py:597` `update_bold_history`가 `history_len=history.shape[0]`로 길이를 고정해 roll → **5000이 파이프라인 끝까지 자가교정 없이 유지**.
- `eqx.tree_at`(`:342`)은 **성공한다** — Bold의 동적 리프 중 `None`은 `history` 하나뿐이라 `except: pass`에 안 삼켜진다. 즉 5000이 8000-커널 모니터에 **실제로 주입된다**.

귀결:
- **Part1 FIC(`fic_step_duration_ms=2_500`)와 Part2 EIB(`t1=bold_repetition_time_ms`) 둘 다 1 TR step** → `5000+625=5625 < 8000` → `jax.scipy.signal.fftconvolve`가 mode='valid'에서 **예외 없이 조용히 입력을 swap**한다(`signal.py:125-131`: 1-D는 `no_swap or swap`이 항상 참). 출력 2376샘플 → `min_len=1` → **`part2_eib.py:172`의 `bold_output.ys[0,0,:]`가 의미 없는 값**. `:174`의 유한성 체크는 유한하므로 통과 → **무증상**.
- 긴 시뮬(600s posthoc/gradient): `(8000−5000)×4ms = 12s` 시프트 + 출력 3000샘플 부족. `bold_time`은 `ts`에서 독립 계산(`bold.py:283`)이라 **라벨과 내용이 12s 어긋난다**. FC는 전 노드 공통 시프트라 상대적으로 덜 민감.

`config.py:76` 기본값 20_000이면 5000==5000으로 무해하다. 32_000을 세팅하는 러너(`main_ppmi_pd`/`main_ppmi`/`main_pd`/`main_nor`/mouse)만 해당.

**고치는 법**: `build_network`가 `build_bold_monitor`와 동일한 kernel/params를 쓰게 한다.

### S1-3. Adam 모멘트가 chunk마다 리셋된다 (~~확인~~ → ✅ 해결됨 2026-07-20)

> **해결됨**: 현재 `part3_gradient.py:799 _optax_run_persist`가 optax chain을 직접 구동해 `opt_state`를 25청크 내내 스레드한다(`_run_full_optimization_loop`가 `opt_state=None`으로 1회 init 후 청크마다 전달·회수, `:847`·`:875`). adamax 모멘트가 더는 초기화되지 않는다. **아래 원문은 수정 전 기록이다.**

`OptaxOptimizer.run()`이 호출마다 `opt_state = self.optimizer.init(diff_state)`(`optim/optax.py:209`)를 실행하고 **opt_state를 반환하지 않는다**(`:232`). 즉 `run()`은 재개 불가 설계다.

`part3_gradient.py:840-846`은 `while` 루프로 `optimizer.run(state, max_steps=chunk)`를 반복 호출한다. `optimizer_chunk_steps=10`이므로 **250 step이 "10 step 최적화 × 25회"**가 된다 — adamax의 1차/무한차 모멘트가 25번 초기화된다. Part3.5 σ 루프(`:600`)도 동일.

`_run_full_optimization_loop`가 chunk를 나누는 목적은 진행률 출력과 best 추적인데, 그 대가로 optimizer 상태를 잃는다. `chunk_steps`를 키우면 리셋은 줄지만 로그가 성겨진다 — 근본 해결은 `run()`이 opt_state를 받고 돌려주게 하거나, 한 번에 250 step 돌리는 것.

### S2. 최적화 품질에 영향 (확인)

- **[4] `adamaxw`의 weight decay가 clip 전 raw `.value`에 걸린다** → c_ei/wLRE/wFFI가 매 step 0쪽으로 끌린다. 의도한 정규화가 아니면 `adamax`를 쓰거나 decay=0.
- **[5] `BoundedParameter`는 sigmoid 재매개화가 아니라 hard clip**이다(`types/parameter.py:724-726`: `__jax_array__`가 `jnp.clip`). raw 값이 경계를 벗어나면 **gradient가 정확히 0이 되어 영구 고착**. Part3.5의 σ는 `low=0.0`(`part3_gradient.py:520`)이라 위험. 라이브러리에 `SigmoidBoundedParameter`가 따로 있다.
- **[6] `capture_internal_state()`가 항상 `None`을 반환한다**: `pipeline_contracts.py:527`이 `dir(internal)`을 도는데 `internal`은 `Bunch`(dict 서브클래스)이고 `Bunch.__setattr__`이 값을 dict **키**에 넣는다. `__dir__`가 없으니 `dir()`은 dict 메서드만 → 전부 callable → skip → `return captured or None` = **None**. `internal_state`/`noise_state`는 영구 None, `restore_internal_state`는 상시 no-op, part1/2/3의 `new_internal_state=` 경로 전체가 死코드.
- **[7] noise realization이 고정**: `AdditiveNoise`가 `key=` 없이 생성되어(`model.py:176`) `self.key = jax.random.key(0)`이고 갱신되지 않는다(`noise/base.py:43`). `prepare()`는 노이즈를 **전량 사전 생성**하므로 shape만 같으면 **매번 동일한 realization**이다. [6] 때문에 stage 경계에서 재추첨도 복원되지 않는다.
  → Part3의 pathwise gradient에는 오히려 **고정 noise가 필요**하므로 수학적으로는 정확하다. 다만 **250 step 전부 같은 dW draw**를 쓰므로 그 draw에 과적합할 여지가 있다. FC 자체가 noise-driven인 모델이라 무시할 수 없다.

### S3. 운영/도구

- **[8] figure 중복 저장 — 실물 확인.** `_patched_show`(`main_ppmi_pd.py:75-87`)가 `get_fignums()` 전체를 매번 저장하고 **close를 안 한다**. Agg 백엔드라 `plt.show()`는 figure를 닫지 않는다 → n번째 show에서 figure 1..n 전부 재저장. O(n²).
  ```
  output_ppmi_pd/42429/figures:
    001_Structural_Weights.png     ← show#1
    002_Structural_Weights.png     ← show#2 중복
    003_Part_1___FIC_Results.png
    004_Structural_Weights.png     ← show#3 중복
    005_Part_1___FIC_Results.png   ← 중복
    006_Part_1___FC_matrices.png
  ```
  `savefig` 뒤 `_plt.close(fig)` 한 줄.
- **[9] `main_ppmi_pd_part3.py`가 즉시 죽는다.** `:79`가 `M.prepare_pd_data(subject_idx, parcellation, noise_level)` 3인자인데 `main_ppmi_pd.py:190` 정의는 2인자. schaefer 시절 잔재(`--parcellation 100/200`)가 AALv3 전환 때 안 따라왔다. import는 통과(`if __name__` 가드)하고 `main()` 첫 실질 문장에서 `TypeError`.
- **[10] `--freeze-c-ei`가 Part3에 무효**: `part3_gradient.py:176`에 `c_ei_frozen = False` 하드코딩("Part 3: c_ei는 항상 최적화 대상"). 플래그 이름은 전역처럼 읽히지만 Part2에만 적용된다.
- **캐시**: `@cache`의 키는 **파일명 하나뿐**이고 함수 인자는 해싱되지 않는다(`utils/caching.py:52-60`) → 프로젝트가 `cache_name`에 cfg/fingerprint를 수동으로 이어붙이는 건 **올바른 대응**이다. `fingerprint()`가 warmup 최종상태를 포함하는 건 사실이나(`pipeline_contracts.py:434`), warmup 노이즈는 고정 시드라 **"GPU 비결정성 때문에 프로세스 간 항상 miss"는 미입증**이다(`main_ppmi_pd_part3.py:11-15` 독스트링의 저자 주장). 60만 step 카오스 증폭 가능성은 있으나 실행 없이 확정 불가.

### S4. Part 4 DBS (기본 skip이라 현재 영향 없음)

- **자극 진폭 포화**: `dbs_pulse_amplitude=1.0` nA vs 배경 구동 `W_e·I_o=0.382` nA → 2.6배. `x_e=310·x_e_pre−125`이므로 anodic에서 `H_e≈303 Hz`. 전달함수가 강한 정류기라 **biphasic이 전하평형이어도 순 흥분 DC로 정류**되고 `S_e`가 1.0에 포화(추론, 수치 확인). mouse/WC 시절 무차원 값의 잔재로 보인다.
- **주파수 오차**: `period_steps = max(3, round(1000/130)) = 8` → **실제 125 Hz**(3.8% 오차). dt=1ms에서 위상당 1샘플이 한계.
- **during-stim FC가 16 TR**: TR=2500ms, stim=60s → `during[296:312]=16 TR`로 **167노드 FC를 추정 → rank ≤ 15, 사실상 잡음**. pre는 264 TR. 가드는 `<2`만 본다(`:462`). 자극 길이를 늘리거나 창 길이를 맞춰야 한다.
- **메모리**: stim 배열 `(780000,167) f32 = 521 MB/target` × 4 = **2.1 GB**를 미리 생성. "JIT recompile 방지" 주석은 거짓 — `prepare()`가 target마다 호출되어 어차피 재컴파일된다.
- `dbs_stimulus_modes`/`dbs_phase_duration_steps`는 하드코딩에 덮여 死코드. `config.py:180-185`의 mouse 인덱스 기본값은 167노드에서 **in-range라 조용히 엉뚱한 노드를 자극**하지만 `main_ppmi_pd.py:372`가 라벨 매칭으로 덮어써 안전하다.

---

## 5. 정상으로 확인된 것 (재조사 불필요)

- **알고리즘이 참조 노트북 `EI_Tuning (11).ipynb`와 일치한다** — FIC 규칙, EIB 규칙, EIB LR ramp 전부 부호·계수·정규화까지 동일.
- **`main_ppmi_pd`가 형제 러너 중 가장 최신·충실하다.** `main_pd`/`main_nor`/`main`은 아직 deprecated된 `fic_early_stop_tolerance_hz`를 세팅하고(`part1_fic.py:239`는 `_se`만 읽음 → 死값), `fic_learning_rate=1e-3`+`wc_c_ei_init=10.0`으로 노트북(0.5/1.0)과 어긋난다.
- `StateBundle.apply_to_state()`의 setattr은 **위험한 해킹이 아니라 라이브러리가 문서화한 정식 용법**이다(`solve.py:216-225`가 `config.dynamics.G = 2.5`를 권장).
- per-node σ broadcast 정상, 적분식 `σ·sqrt(dt)·dW` 정상(stochastic Heun 표준).
- `Parameter`로 감싼 필드만 최적화 대상인 것 맞음. `wLRE/wFFI`를 무제약 `Parameter`로 두는 건 `_project_weights`가 매 step 투영하므로 버그 아님.
- `DROP_LABELS` 처리가 형제보다 엄격 — `_np.ix_`로 SC/FC/LEN/labels 동시 슬라이싱, `nan_to_num`을 DROP **뒤**에 수행해 결측 진단 보존, `derive_cortex_subcortex_indices`를 직접 호출해 `data_loader`와 기준 일치.
- Part3의 loss 재설계(activity weight 0)는 의도된 것 — 근거가 `main_ppmi.py:297` 주석에 있다(단 `main_ppmi_pd.py:355`엔 근거 주석이 복사 안 됨).

## 6. 死코드 (main_ppmi_pd 경로 기준)

- `config.py`의 `wc_*` 약 25줄 — `model.py`가 override 루프를 제거해 **`wc_c_ei_init` 하나만 살아있다**.
- `optimizer_nodewise_corr_weight=0.40` — part3가 안 읽는다(config 주석도 인정).
- `pd_fit_region_count`, `resolve_pd_fit_indices`, `pd_fit_block_loss_weight`, `baseline_settle_duration_ms` — 구 `main.py` 전용.
- `gpu_batch_size`, `posthoc_parallel`, `dbs_parallel_targets` — 코드베이스 어디서도 안 읽는다.
- delay 관련 전체 (§4 S1-1).
- `internal_state` 전달 경로 전체 (§4 S2-6).
- part2의 `_select_block`/`pre_window_fc`, 각 part의 `_analyze_beta_oscillation_*` — 정의만 있고 호출 없음.
- `_compute_fc_from_bold_output`가 part1/2/3/4에 **4벌 중복**.

---

## 7. 권장 조치 순서

1. ~~**S1-1 결정**~~ **✅ 결정됨(2026-07-20): delay 미사용.** 모델이 이미 delay-free라 동작 변경 없음(`model.py` docstring 정정 반영). 죽은 delay 배선(`data_loader` delays·`DenseDelayGraph`·플롯·`delay_history`) 제거는 선택적 정리.
2. ~~**S1-2 수정**~~ **🔧 적용됨(2026-07-20)** — `build_network`가 `make_bold_monitor`로 cfg HRF를 쓴다(`model.py:207`). **전 subject 재실행 필요.**
3. ~~**S1-3 수정** — chunk 리셋.~~ **✅ 해결됨** — `_optax_run_persist`가 opt_state 왕복 구현(`part3_gradient.py:799`).
4. S3-8/9 — 각각 한 줄 수정.
5. S2-4/5 — optimizer 설정 재검토.

**§8 산출물은 S1-1·S1-2 수정 전 생성분 — 재현·해석에 쓰지 말 것.** (S1-2 코드 수정 적용·재실행 필요, S1-3 해결됨)

---

## 8. 실제 산출물 분석 — `output_ppmi_pd/AALv3/`

정적 분석의 결론을 실제 완주 산출물에 대조했다. **§4의 버그들이 그대로 찍혀 있다.** (분석 방법: figure 제목에 렌더된 수치 판독. pkl 언피클은 로그인 노드 py 실행 금지라 하지 않음.)

### 8.1 무엇이 있나

subject **2명 완주** — `100001`(subject_idx 0) / `100005`(idx 1). 각 79~80MB. 167노드 AALv3 PD, **Part 1(FIC) → 2(EIB) → 3(Gradient) 전 단계 완료**. Part 3.5/4는 기본 skip이라 없음.

```
<subj>/
├ inputs/    weight.csv, tract_length.csv, FC.csv, region_labels.txt (167줄)
├ cache/     fic_*.pkl, eib_*.pkl, grad_*.pkl  (각 ~20MB)
└ figures/   36개 png
```

캐시 태그 `v_pdaal_tr25_s{0,1}_N167` → main_ppmi_pd 최신 경로 산출물 확정.

### 8.2 결과 수치 — 파이프라인은 작동한다

| stage | 100001 (idx0) | 100005 (idx1) |
|---|---|---|
| Pre (초기) corr | −0.0022 | ~0 |
| Post-FIC corr / rmse | 0.0241 / 0.1713 | 0.0190 / 0.1731 |
| Post-EIB corr / rmse | 0.5163 / 0.1368 | 0.4545 / 0.1478 |
| **Post-Part3 corr / rmse** | **0.7041 / 0.1237** | **0.6236 / 0.1380** |

- **단계별 기여가 명확하다.** FIC는 corr을 거의 안 올린다(0.02) — 정상이다. FIC는 노드 gating을 0.25에 맞추는 단계지 FC 적합이 목적이 아니다. Post-FIC FC matrix가 거의 백지(구조 없음)인 것이 그 증거. **corr을 만드는 건 EIB(→0.5)와 Part3(→0.7)**다.
- **최종 corr 0.62~0.70은 이 종류 모델(RWW + SC 기반 개인 FC 적합)에서 정상 범위**다(문헌 통상 0.5~0.7).
- **Block-wise corr (100001, Part3 post)**: full 0.70 / ctx-ctx 0.70 / **cross 0.63** / **sub-sub 0.85**. block 가중(0.4/0.4/0.2)이 subcortex를 강제 균형시킨 효과가 실제로 나타남 — subcortex가 오히려 가장 잘 맞고, 물리적으로 가장 어려운 cross(피질↔피질하)가 가장 약하다. 설계 의도대로 동작.

### 8.3 §4 버그의 실물 증거

1. **figure 36개 = 삼각수 T(8) — S3-8(O(n²) 중복 저장)의 완벽한 확증.** 고유 figure는 8종인데 36번 저장됐다. `001_Structural_Weights`가 subject당 8회 반복(001·002·004·007·011·016·022·029번). 80MB 중 실데이터는 ~15MB, 나머지는 중복.
2. **이 corr 값들은 S1-2(BOLD HRF 불일치)의 영향을 받은 값이다.** Part3 posthoc(600s)은 12s 시프트 상태로, EIB 창(1 TR)은 fftconvolve swap 영역에서 측정됐다. → **절대값 0.70을 신뢰하면 안 된다.** 상대 추세(FIC < EIB < Part3)는 유효하나 눈금이 틀어져 있다.
3. **S1-1(delay 미반영):** `tract_length.csv`가 inputs에 저장돼 있으나 이 결과 생성에 쓰이지 않았다. delay-free 시뮬 결과다.

### 8.4 경로 이상 — `AALv3/` 한 겹이 더 있다

정상 경로는 `output_ppmi_pd/<sub_num>/`인데 이 결과는 `output_ppmi_pd/AALv3/<sub_num>/`에 있다. `<sub_num>` 앞에 `AALv3`가 한 겹 더 붙었다.

- 현재 `main_ppmi_pd.py`는 `out_dir = OUTPUT_DIR/<sub_num>`(`:239`)라 이 경로를 만들지 않는다.
- git status에서 `main_ppmi_pd.py`가 modified(M) → **이 산출물은 현재 워킹트리보다 이전 버전 코드로 생성**됐다(당시 `out_dir`에 `AALv3`가 끼어 있었던 것으로 보인다).
- 지금 코드로 재실행하면 `output_ppmi_pd/100001/`에 새로 떨어진다(루트에 `100012`, `40533` 등 다른 세대 폴더가 별도로 존재).
- → **`AALv3/`는 구버전 경로의 고아 결과.** 정리 대상 후보다. 단 20MB pkl × 3은 재계산에 subject당 수 시간 걸리므로 값이 필요하면 백업 후 삭제.

### 8.5 종합

- **과학적으로는 성공** — 단계별 개선이 명확하고 최종 corr·block corr이 정상 범위.
- **단 절대 수치는 S1-1/S1-2 버그로 눈금이 틀어져 있어 논문 값으로 쓰면 안 된다.** §7의 S1 셋을 고친 뒤 재실행 필요.
- 2명뿐이라 통계 불가 — 242명 배치의 파일럿 2건으로 보인다.
