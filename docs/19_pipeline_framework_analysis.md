# 19 — main_ppmi_pd 파이프라인 & 프레임워크 구조

**결론: `main_ppmi_pd.py`는 개인 구조적 연결성(SC)에서 그 사람의 기능적 연결성(FC)을 재현하는 Reduced Wong-Wang 뇌 네트워크 최적화 파이프라인이다.** 얇은 오케스트레이터일 뿐, 로직은 전부 `tvboptim`(미분가능 TVB 엔진) 위에 얹힌 모듈에 있다. 단계는 heuristic(FIC → EIB) 다음 gradient(Part3)로 난이도를 올린다. 설계(단계 간 `StateBundle` full-state 전달)는 견고하나, **문제는 전부 "라이브러리 사용 계층"에 있고(§6) 조용히 틀린 값을 낸다.**

작성일: 2026-07-20 / 대상: `main_ppmi_pd.py` + 구성 모듈 전체 + `tvboptim` 라이브러리 / 방법: 정적 분석(실행 없음). 버그 목록·산출물 대조는 `18_main_ppmi_pd_analysis.md` 참조 — 이 문서는 **구조/프레임워크** 관점이다.

---

## 1. 프레임워크 스택 (3계층 + 모델)

```
┌─ JAX ─────────── lax.scan 적분, checkpoint(remat), value_and_grad, float32
├─ equinox ─────── PyTree Module, eqx.tree_at 불변갱신, Parameter 리프 = 최적화 대상
├─ optax ───────── adamaxw + zero_nans + clip_by_global_norm
└─ tvboptim ────── 미분가능 TVB 시뮬레이션 엔진 (핵심)
```

### 1.1 tvboptim — 미분가능 TVB 엔진

조립 방식 (`model.py:build_network`):
```
Network(dynamics, coupling, graph, noise)
  → prepare(network, solver, t1, dt)   # config PyTree 빌드 + JIT 컴파일
  → compiled_fn(config)                # 단일 jax.lax.scan → NativeSolution(.ys/.data, .ts)
```
`config`는 순수 PyTree(`Bunch`, 정렬키). 파라미터가 그 안에 살아서 **`jax.grad`/`vmap`이 config를 미분/배치**하는 게 최적화·병렬의 원리다.

| 하위 모듈 | 역할 | 이 프로젝트 사용 |
|---|---|---|
| `dynamics/base.AbstractDynamics` | 노드 국소 방정식 계약 `(deriv, aux)` | `ReducedWongWangEIB` (`model.py:40`) |
| `coupling/base` | `InstantaneousCoupling`(ODE) vs `DelayedCoupling`(DDE) | `EIBLinearCoupling(Instantaneous)` (`model.py:131`) |
| `graph` | `DenseGraph` / `DenseDelayGraph`(+delays) | `DenseDelayGraph` (`data_loader.py:87`) |
| `noise` | `AdditiveNoise` — SDE 확산항 | `AdditiveNoise(sigma, apply_to="E")` |
| `solvers/native` | `Euler`/`Heun`/`RK4`, `BoundedSolver`(클램프) | `BoundedSolver(Heun(), 0, 1)` |
| `observations/…/bold.Bold` | HRF 합성곱 BOLD 모니터 | Part별 `build_bold_monitor` |
| `optim/optax.OptaxOptimizer` | optax 래퍼 | Part3 (단, §6·§7 주의) |
| `types/parameter` | `Parameter`/`BoundedParameter`/`SigmoidBoundedParameter` | c_ei/wLRE/wFFI/σ 래핑 |

**적분기 내부**(`solve.py`): coupling precompute는 pass당 1회, `op(state, inputs)` 단일 스텝이 전체 궤적을 도는 **하나의 `jax.lax.scan`**. 노이즈는 `prepare()`에서 **전량 사전생성**(`noise_samples[n_steps, n_noise, n_nodes]`, 고정 key=0), 스텝마다 `σ·√dt·dW` 스케일해 주입. 이 **고정 realization**이 SDE gradient를 결정적으로 만들어 Part3 backprop을 성립시킨다.

### 1.2 모델 — Reduced Wong-Wang (Wilson-Cowan 아님)

이름 `WilsonCowanEIB`는 하위호환 별칭. 실제는 2집단 RWW. 상태 `E`/`I`는 사실 **synaptic gating `S_e`/`S_i`** (`model.py:85–123`):

```
흥분:  x_e = a_e·(w_p·J_N·S_e − c_ei·S_i + W_e·I_o + J_N·wLRE·cpl + I_ext) − b_e
       H_e = _rww_H(x_e, d_e)
       dS_e/dt = −S_e/τ_e + (1−S_e)·H_e·γ_e      (+ noise on E)
억제:  x_i = a_i·(J_N·S_e − S_i + W_i·I_o + λ·J_N·wFFI·cpl) − b_i
       H_i = _rww_H(x_i, d_i)
       dS_i/dt = −S_i/τ_i + H_i·γ_i
```
- **`c_ei` == RWW의 `J_i`** (S_i→E 억제, per-node) → **FIC 튜닝 대상**.
- **`wLRE`/`wFFI`** = N×N coupling 행렬. `pre()`가 source 흥분을 두 채널로 나눠 유효 엣지가 `graph.weights ⊙ wLRE`(와 ⊙wFFI) → **EIB 튜닝 대상**.
- 전달함수 `H(x)=x/(1−exp(−d·x))`는 x=0에서 0/0 특이점(참값 1/d). `_rww_H`가 double-where로 `|d·x|<1e-4`를 `1/d`로 대체 → float32 NaN 전파 차단. (라이브러리 원본엔 이 가드 없음 — 프로젝트 패치.)
- solver `BoundedSolver(Heun, 0, 1)`가 S를 매 스텝 [0,1] 하드클램프.

### 1.3 최적화 대상 = `Parameter` 래핑

리프가 `Parameter` 인스턴스일 때만 미분됨. `__jax_array__`가 재매개화 훅:
- `Parameter`: raw 값 그대로.
- `BoundedParameter`: `jnp.clip(value, low, high)` — **하드클립, 경계 밖 gradient=0 고착**.
- `SigmoidBoundedParameter`: `low+(high−low)·sigmoid(value)` — 부드러운 재매개화(고착 없음, 존재하나 미사용).

---

## 2. 데이터 흐름

```
data/AALv3/FC_AAL_ComBat_all_163_PD.mat (FC_ComBat, 242명 PD)
  → group=='PD' 필터 후 subject 1명 (FC/SC_weight/SC_length, 163노드)
     · FC = ComBat 배치보정 harmonized (7 스캐너 사이트 batch effect 제거, off-diag NaN 0)
     · SC_weight = subject별 streamline count,  SC_length = tract length(mm)
     · DROP 없음 (163 아틀라스가 결측노드 Thal_Re_L 등 이미 제외 → clean)
  → output_ppmi_pd/<sub_num>/inputs/*.csv write → load_data 가 다시 read (캐시 fingerprint용 왕복)
      · weights = normalize(log1p(SC+0.5)) ⊙ sc_mask
      · delays  = SC_length / tract_conduction_speed(3.0)   ← 계산은 하나 미사용(§6-1)
      · DenseDelayGraph(weights, delays)
      · cortex 112 / subcortex 51 (라벨 이름 집합 판정, data_loader.py:138)
      · fc_edge_weight(블록점유 0.50/0.40/0.10) + fc_block_masks(cc/cross/ss)
      · cache_tag = v_pdaal163_tr25_s{idx}_N163_sc{hash}_fc{hash}
  → build_network + warmup 600s → bundle_init(StateBundle)
```

163노드 = cortex 112 + subcortex 51 (basal ganglia + thalamus + 뇌간핵 + STN). 라벨: `AAL163_labels.txt`(TSV, region 열), 노드순서 = txt aal_code = .mat labels(검증됨).

---

## 3. 파이프라인 계약 — `StateBundle` / `ParamSet`

단계 간 전달을 **불변 객체 하나**로 통일 (`pipeline_contracts.py`). 이 프로젝트 설계의 핵심.

**`ParamSet`**: `c_ei`(벡터) + `wLRE`/`wFFI`(행렬) + `c_ei_frozen`.

**`StateBundle`** 담는 것:

| 필드 | 내용 |
|---|---|
| `params` | ParamSet |
| `init_dynamics` | 다음 stage 시작 neural endpoint (`warmup_result.data[-1]`) |
| `bold_history` | BOLD HRF 합성곱 재개용 history (§6-2 관련) |
| `bold_window` | 직전 window 실제 emit된 BOLD 샘플 (EIB rolling buffer seed) |
| `internal_state` | `noise_samples` 등 |
| `delay_history` | delay graph 복원용 (미사용, §6-1) |
| `metadata` | rng_key + `post_<stage>_fc_matrix/corr/rmse` |

`advance(...)`는 지정 안 한 필드를 **그대로 계승**(불변 갱신) → **full-state passthrough**. FIC는 `c_ei`만, EIB는 `wLRE/wFFI`(+동결 안 하면 c_ei)만 쓰고 나머지 전부 계승.

**공용 loss**(part1/2/3 공유): `weighted_corr_loss`/`weighted_rmse_loss`, `block_corr_loss`/`block_rmse_loss`(cc/cross/ss 분할·edge수 무관 동등가중 → subcortex 균형), `nodewise_corr_loss`(Part3 미사용).

---

## 4. 단계별 동작 (실제 `main_ppmi_pd` 실행값 기준)

| Part | 튜닝 | 규칙 | step 단위 | 규모 | best 선택 |
|---|---|---|---|---|---|
| **1 FIC** | `c_ei` per-node | `Δc_ei = 0.5·mean_S_i·(mean_S_e − 0.25)` | 2500ms (=1 TR) | 2000 iter | FC 기반 **top_k=10 posthoc 재시뮬** |
| **2 EIB** | `wLRE`/`wFFI` | `wLRE += η·fc_diff·row_rmse`, `wFFI −= …`(반대부호) | 1 TR (online) | 10000 iter | **단일 argmax 스냅샷 + 1회 재시뮬** (patch30) |
| **3 Grad** | c_ei + wLRE + wFFI | optax adamaxw full backprop | 240 TR×2500 = **600s 적분** | 250 step (chunk10 → 25청크) | 최저 loss 스냅샷 |
| **3.5 σ** | per-node noise σ | backprop (c_ei/w 동결) | 600s | 200 step | 기본 skip |
| **4 DBS** | — | STN/GP biphasic 자극 | 720s pre + 60s stim | 4타깃 순차 | 기본 skip |

- **FIC**: per-node 시간평균 gating으로 `c_ei` 조정. 타깃은 **Hz 아니라 S_e gating 0.25**. best는 se_error 작은 top_k=10을 각각 재시뮬해 FC score로 선택. (branch `patch30`은 EIB만 단일선택 이관, **part1엔 top-k 잔존**.)
- **EIB**: 1 TR씩 전진 + 240 TR rolling window FC. `fc_target > fc_pred`인 곳에서 장거리 흥분↑/전향 억제↓. LR 선형 램프. 스냅샷은 params만 저장, posthoc 때 `warmup_bundle` 무거운 state와 결합.
- **Part3**: 600,000 step 적분을 250회 backprop (비용 지배). `PART3_REMAT_SCAN=1`이 scan body checkpointing으로 OOM 방지. loss `0.80·block_corr + 0.20·block_rmse + 0.0·activity`. `c_ei→BoundedParameter[0,20]`, `wLRE/wFFI→Parameter`(청크 후 `_project_weights` 투영). **`_optax_run_persist`가 adamax 모멘트를 25청크 내내 유지**(과거 리셋 버그 수정, §7).
- **σ**: c_ei/σ 축퇴쌍 회피 위해 c_ei/w 동결 후 σ만 튜닝.
- **DBS**: `dynamics` monkeypatch로 biphasic 펄스(±1nA)를 `x_e_pre`에 주입. freq 130 지정이나 정수 반올림 → 실제 125Hz. 자극 전/중 BOLD FC 차이 저장.

---

## 5. config 기본값 ≠ 실제 실행값 (오버라이드 주의)

`config.py` 기본값만 읽으면 오해함. `main_ppmi_pd.make_config`가 덮어씀:

| 항목 | config 기본 | 실제 |
|---|---|---|
| `bold_repetition_time_ms` | 1000 | **2500** (TR=2.5s) |
| `optimizer_bold_window_tr` | 720 | **240** (=600s) |
| `optimizer_learning_rate` | 0.002 | **0.0001** |
| `optimizer_max_steps` / `chunk` | 200 / 5 | **250 / 10** |
| `optimizer_global_corr_weight` | 0.4 | **0.80** |
| `optimizer_activity_weight` | 0.01 | **0.0** |
| `optimizer_rmse_block` | False | **True** |
| `fic_learning_rate` | 1e-3 | **0.5** |
| `wc_c_ei_init` | 10.0 | **1.0** |
| `connectivity_weight_max` | 1.5 | **2.0** |
| `bold_hrf_duration_ms` | 20000 | **32000** (← §6-2 유발) |
| `neural_cache_stride` | 1 | **50** |

---

## 6. 라이브러리 사용 계층 문제 2개 — 쉬운 설명

두 문제의 공통점: **에러를 안 내고 조용히 틀린 값을 낸다.** 알고리즘(FIC/EIB 규칙)은 맞다. 그 아래 라이브러리를 잘못 연결한 것이다.

> **현황(2026-07-20)**: 6-2(HRF)는 **수정 적용**(재실행 검증 필요). 6-1(delay)은 **미사용 확정**(사용자 결정) — 모델이 이미 delay-free라 동작 변경 없음, 죽은 배선 정리는 선택.

### 6-1. Tract delay가 시뮬레이션에 안 들어간다

> **결정(2026-07-20)**: delay **미사용 확정**. 모델은 이미 delay-free(아래 참조)라 "안 쓰기"에 코드 변경은 불필요 — 현재 동작 그대로다. 오해 소지 있던 `model.py` docstring "delay 자동 반영"은 정정 반영됨. 죽은 delay 배선(`data_loader` delays 계산·`DenseDelayGraph`·delay 플롯·`delay_history`)은 **선택적 정리** 대상(동작엔 무영향).

**비유**: 모든 직원의 통근시간을 계산하고, 예쁜 지도로 그리고, 파일로 저장한다. 그런데 정작 출근은 다들 **순간이동**으로 한다. 통근시간 계산은 아무 데도 안 쓰인다.

**실제로 벌어지는 일**:
- 파이프라인은 tract delay(뇌 영역 A→B 신호 전달시간 = 거리/전도속도)를 **계산하고**(`data_loader.py:55`), `DenseDelayGraph`에 **싣고**, 행렬로 **플롯하고**, 모든 bundle·pkl에 **저장한다**.
- **그런데 적분에는 안 쓴다.** coupling이 `InstantaneousCoupling`(`model.py:131`)이라서다. 이 클래스의 `compute()`는 **다른 영역의 "현재" 상태만** 읽는다 — `graph.delays`도, 과거 상태를 담은 history 버퍼(`network.get_history()`)도 건드리지 않는다.
- 지연을 실제로 쓰는 건 라이브러리의 `DelayedCoupling`뿐인데, 이 모델은 그걸 안 쓴다.

**왜 조용한가**: NativeSolver는 `max_delay>0`이어도 에러를 안 낸다(Diffrax 경로만 막음). 그래서 delay 있는 그래프를 넣어도 그냥 무시하고 돈다.

**결과**: 모델은 DDE(지연미분방정식)가 아니라 **지연 없는 ODE/SDE**다. 모든 신호가 즉시 전파된다고 가정한 FC가 나온다. 지연은 뇌 진동의 위상관계 → FC 구조에 실제로 영향을 주므로(여기 지연 ~수십 ms), **"지연 기반 개인 FC"라는 과학적 주장이 성립 안 한다.** `model.py:8` 주석 "tract delay가 자동으로 네트워크에 반영된다"는 **거짓**.

**고치는 법**: `EIBLinearCoupling`을 `DelayedCoupling` 상속으로 바꾼다(pre/post 시그니처 확인 필요). 안 쓸 거면 delay 계산·플롯·저장 코드와 주석을 걷어내고 "지연 없음"을 명시. **어느 쪽이든 과학적 주장에 직결 → 결정이 필요.**

### 6-2. BOLD HRF 커널과 warmup history 길이가 어긋난다  🔧 수정 적용(2026-07-20)

> **수정 적용됨**: `build_network`와 각 Part가 `pipeline_contracts.make_bold_monitor(cfg, history)` 단일 소스로 Bold를 만들게 통합했다. warmup 모니터도 cfg 32s 커널을 써서 `bold_history`가 8000샘플(=kernel_samples)로 잡혀 정렬된다. **코드만 반영 — FC 절대값 교정은 subject 재실행으로 검증 필요.** 아래는 원리 설명.

**비유**: 녹음 스튜디오에서 리버브를 걸려면 앞에 **8초짜리 pre-roll 침묵**을 깔아야 정렬이 맞는다. 그런데 워밍업이 **5초짜리 pre-roll**만 만들어 건네준다. 장비는 8초를 기대하는데 5초가 들어오니, 소리는 나오지만 **3초 밀린 채** 나온다. 에러는 안 뜬다.

**실제로 벌어지는 일** (BOLD = 신경활동을 HRF 커널로 합성곱해 fMRI 신호로 바꾸는 모니터):
- BOLD 합성곱은 시작 부분에 **커널 길이만큼의 warm-up 버퍼**(history)가 필요하다. `mode="valid"` fftconvolve의 설계 전제가 `len(history) == kernel_samples`.
- **모니터가 두 군데서 서로 다른 HRF 설정으로 만들어진다**:
  - `build_network`의 Bold(`model.py:205`)는 `kernel=` 없이 **기본 20초 커널** → warm-up 5000샘플. 이게 warmup history로 캡처돼 `bundle_init.bold_history`가 됨.
  - Part별 `build_bold_monitor`(`pipeline_contracts.py:322`)는 cfg의 **32초 커널** → warm-up **8000샘플** 기대. 여기에 위 **5000샘플 history를 주입**한다.
- 8000을 기대하는 모니터에 5000이 들어감 → 합성곱 정렬이 깨진다.

**왜 조용한가**: `mode='valid'` fftconvolve는 1-D에서 입력이 커널보다 짧으면 **예외 없이 입출력을 swap**한다. 값은 유한하게 나오므로 유한성 체크를 통과한다.

**결과**:
- **Part1 FIC·Part2 EIB**(둘 다 1 TR step): `5000+625 < 8000` → swap 영역에서 측정 → **FC가 무의미한 값**인데 무증상.
- **긴 시뮬(Part3 posthoc 600s)**: `(8000−5000)×4ms = 12초` 시프트. FC는 전 노드 공통 시프트라 상대적으로 덜 민감하나 눈금이 틀어진다.
- `config.py` 기본 `bold_hrf_duration_ms=20000`이면 5000==5000이라 무해. **32000을 세팅하는 러너**(`main_ppmi_pd` 등)만 해당.

**고친 방식(적용됨)**: `build_network`가 raw `Bold(...)` 대신 `make_bold_monitor(cfg, history=warmup_result)`를 호출 → 각 Part의 `build_bold_monitor`와 커널이 자동 일치. 단일 소스라 재드리프트 불가. **전 subject 재실행 필요.**

**주의**: 이 버그는 **절대 FC 수치의 눈금**을 틀어놨었다. 수정 전 산출물의 **절대 corr(예: 0.70)은 논문 값으로 쓰면 안 된다** — 재실행 필요. 단계별 상대 추세(FIC < EIB < Part3)는 유효.

---

## 7. `18_..._analysis.md`와의 관계 (S1-3 해결됨)

doc 18의 세 번째 대형 지적 **S1-3(Part3 Adam 모멘트가 chunk마다 리셋)은 이후 패치에서 해결**됐다. `part3_gradient.py:799 _optax_run_persist`가 optax chain을 직접 구동해 `opt_state`를 25청크 내내 스레드한다(과거엔 `OptaxOptimizer.run()`이 청크마다 `init()` 재실행 → 25회 모멘트 초기화). doc 18 §4[3]·§7-3은 그 시점 기록이며 현재 코드엔 해당 없음.

**§6-1(delay)은 미사용 확정**(사용자 결정) — 모델이 이미 delay-free라 동작 변경 없음. §6-2(HRF)는 수정 적용(재실행 검증 필요). 나머지 S2~S4는 doc 18 참조.

---

## 8. 死코드 (main_ppmi_pd 경로)

- delay 관련 전체(§6-1), `internal_state` 전달 경로(`capture_internal_state`가 `Bunch.__dir__` 부재로 항상 None — doc 18 S2-6).
- `config.py` `wc_*` ~25줄(`wc_c_ei_init`만 생존).
- `optimizer_nodewise_corr_weight`·`nodewise_corr_loss` — Part3 미사용.
- 각 part `_analyze_beta_oscillation_*`, part1 `_extract_firing_rates`/`_settle_bundle`, part2 `_select_block`, `_compute_fc_from_bold_output` 4벌 중복.
- DBS `stimulus_modes`/`phase_duration_steps`, `gpu_batch_size`/`posthoc_parallel`/`dbs_parallel_targets`.

---

## 9. 종합

- **설계는 좋다**: `StateBundle` full-state passthrough, 공유 loss, 미분가능 엔진 위 heuristic→gradient 상승 구조. 알고리즘은 참조 노트북과 일치.
- **문제는 라이브러리 사용 계층**: (1) delay 미연결 — **미사용 확정**(모델 이미 delay-free), (2) BOLD HRF 커널 불일치 — **수정 적용(재실행 검증 필요)**. 둘 다 무증상이었다.
- doc 18 S1-3(Adam)은 그새 수정됨.
- **§6-2 수정본 재실행 전 산출물의 절대 FC 수치는 재현·해석·논문에 쓰지 말 것.** 상대 추세는 유효. (delay는 미사용 확정.)
