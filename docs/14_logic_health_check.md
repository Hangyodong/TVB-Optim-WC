# 14 — 파이프라인 로직 상태 체크

> 정적 코드 리딩만으로 작성. 실행·임포트·수정 없음. 주석이 아니라 **코드가 실제로 하는 일**만 기록한다.
> 검토 대상: `config.py`, `model.py`, `part1_fic.py`, `part2_eib.py`, `part3_gradient.py`, `part4_dbs.py`, `pipeline_contracts.py`

## 요약 테이블

| 단계 | 상태 | 한줄 설명 |
|---|---|---|
| Warmup / build_network | ✅ 정상 | DEFAULT_PARAMS↔cfg.wc_* 일치, noise는 E만, warmup 결과 downstream 사용됨. BoundedSolver가 I도 [0,1] clip(saturation 위험은 낮음) |
| Part 1 — FIC 업데이트 규칙 | ⚠️ 주의 | rI 게이팅은 수학적으로 타당하나 rI→0 시 controller 정지. c_ei 상한 20은 임의값 |
| Part 1 — freeze 플래그 | ⚠️ 주의 | FIC 루프는 freeze와 무관하게 항상 c_ei 갱신(설계 의도). 플래그는 EIB에만 적용, Part3/3B는 강제 False |
| Part 1 — best 선택 / 캐시 | ⚠️ 주의 | 스냅샷은 update 직전 기록(정확). 캐시 키에 best-선택 로직 토큰 없음 → cache_version에만 의존 |
| Part 2 — eta 스케줄 | ⚠️ 주의 | 선형 ramp가 **마지막 step에서 peak** → 후반 overshoot 가능. best-snapshot이 완화 |
| Part 2 — wLRE/wFFI 업데이트 | ✅ 정상 | `_clip_sym`가 `w_max` 상한 적용(복원 완료). 반대부호 대칭 업데이트, bounded |
| Part 2 — c_ei_frozen / 시드 / post-hoc | ✅ 정상 | frozen 플래그 정확히 읽힘, seed window shape 일치, post-hoc는 warmup state+snapshot params 재시뮬 |
| Part 3A — gradient | ⚠️ 주의 | wLRE/wFFI 최적화 중 unbounded(추출 시점에만 clip). activity weight 0.01 하드코딩, plot 라벨 불일치 |
| Part 3B — low-rank | ⚠️ 주의 | 대칭화가 masking 뒤(SC 비대칭 시 mask 깨짐). delta는 w_max로 bound, c_ei 학습됨 |
| Part 4 — DBS | ⚠️ 주의 | charge-balanced 정상, finally 복원 보장. 실제 주파수 125Hz(요청 130Hz, 정수 step 양자화) |
| Pipeline contracts | ✅ 정상 | fingerprint에 bold_window 포함, advance 전 필드 전파, round-trip은 float32 강제(정밀도 손실 가능) |

---

## 단계별 상세

### Warmup / build_network (`model.py`)

**Q1. DEFAULT_PARAMS ↔ cfg.wc_* 일치?**
- 코드 근거: `model.py:34-61` DEFAULT_PARAMS, `config.py:32-57` wc_*, override 루프 `model.py:160-168`.
- 판정: ✅
- 설명: 모든 공통 항목이 수치 일치한다(c_ee 11, c_ei 10, c_ie 10, c_ii 1, alpha_e 1.2, alpha_i 2.0, theta_e 2.0, theta_i 3.5, P 0.5, Q 0.0, lamda 1.0, rE_max/rI_max 20, tau 10 등). c_ei는 스칼라 DEFAULT(10) 대신 `wc_c_ei_init`(10)로 per-node 초기화된다(`model.py:156-159`). `I_ext`는 DEFAULT에만 있고(0.0) 대응 cfg 필드가 없어 override 루프에서 건드리지 않는다 — 0.0 유지로 안전.
- 권고: 생략.

**Q2. AdditiveNoise sigma는 E에만?**
- 코드 근거: `model.py:174` `AdditiveNoise(sigma=cfg.additive_noise_sigma, apply_to="E")`.
- 판정: ✅
- 설명: noise는 `apply_to="E"`로 흥분 집단에만 주입된다. I에는 직접 noise 없음.

**Q3. BoundedSolver [0,1]은 E와 I 모두 clip?**
- 코드 근거: `model.py:183` `BoundedSolver(Heun(), low=0.0, high=1.0)`. 동일 패턴이 모든 stage 재사용.
- 판정: ✅(주의)
- 설명: BoundedSolver는 state 전체([E, I])를 [0,1]로 clip한다. WC activity가 본래 [0,1] 정규화 변수이므로 의도된 동작. I가 1에 도달하면 hard clip되어 미분정보가 손실될 수 있으나 정상 범위에서 saturation은 드물다.
- 권고: I가 상한 1에 자주 닿는 파라미터셋에서만 dynamics 왜곡 모니터.

**Q4. warmup_duration_ms 결과가 downstream에서 실제로 사용되는가?**
- 코드 근거: `model.py:191` warmup 실행 → `:198-200` 최종 E/I, `:207` bold_monitor의 `history=warmup_result`, `pipeline_contracts.py:401-427 from_warmup`가 `warmup_result.data[-1]`을 init_dynamics로, monitor.history를 bold_history로 저장.
- 판정: ✅
- 설명: warmup의 마지막 neural state와 BOLD history가 StateBundle에 보존되어 FIC 시작점으로 사용된다. 폐기되지 않음.

---

### Part 1 — FIC (`part1_fic.py`)

**Q1. 업데이트 규칙 `c_ei += η·rI·(rE−target)`, rI 게이팅 타당? rI→0이면?**
- 코드 근거: `part1_fic.py:180-184`. `rate_error = mean_rE − target`, `update_delta = lr · mean_rI · rate_error`, clip[0,20].
- 판정: ⚠️
- 설명: excitatory_input의 c_ei 항이 `−c_ei·I`이므로 c_ei가 E에 미치는 민감도는 I(여기선 rI 대용)에 비례한다 — rI 게이팅은 수학적으로 타당. 방향도 옳다(rE>target → delta>0 → c_ei↑ → 억제↑ → rE↓). 다만 rI→0이면 update_delta→0이라 **controller가 정지**해 rE가 target에서 벗어나도 교정 불가.
- 권고: rI가 0 근처로 붕괴하는 파라미터셋에서 FIC 수렴 정체 가능성 모니터.

**Q2. c_ei clip [0,20] 상한이 과학적 근거인가 임의값인가?**
- 코드 근거: `part1_fic.py:182-184`, 동일 상한이 EIB(`part2_eib.py:190,224`), Part3(`part3_gradient.py:233,424`), sanitize(`pipeline_contracts.py:470`)에서 반복.
- 판정: ⚠️
- 설명: 상한 20은 초기값 10의 2배로, 코드·주석 어디에도 생리학적 근거가 없다. 안전 가드 성격의 임의값.
- 권고: 상한 도달 노드가 있으면 임의 cap에 묶인 것이므로 결과 해석 시 유의.

**Q3. freeze_c_ei_after_fic가 `_run_fic_loop_pure`와 반환 ParamSet에서 실제로 존중되는가?**
- 코드 근거: 루프 c_ei 갱신 `part1_fic.py:182-184`(freeze 검사 **없음**), 반환 시 `:286` `c_ei_frozen=bool(getattr(cfg,"freeze_c_ei_after_fic",False))`. EIB 소비 `part2_eib.py:180`. Part3 강제 False `part3_gradient.py:166`, Part3B `:375`.
- 판정: ⚠️(버그 아님, 설계상 주의)
- 설명: FIC 루프 자체는 freeze 플래그와 무관하게 항상 c_ei를 갱신한다(의도: freeze는 FIC **이후** 단계용). 반환 ParamSet의 `c_ei_frozen`만 플래그를 반영한다. 이 플래그는 **EIB에서만** c_ei 갱신을 차단하고, Part3/3B는 주석대로 항상 `c_ei_frozen=False`로 덮어써 무시한다. 즉 이름이 시사하는 "Part3까지 동결"은 코드상 적용되지 않는다.
- 권고: 네이밍과 실제 적용 범위(EIB 한정)의 불일치를 문서화.

**Q4. best c_ei 선택: 스냅샷이 update_delta 이전에 기록되는가? fallback이 옛 동작을 보존하는가?**
- 코드 근거: 스냅샷 `part1_fic.py:172-178` → update `:180-184`. fallback `:255-257`.
- 판정: ✅
- 설명: step 151에서 시뮬레이션은 **갱신 전 c_ei**로 실행돼 current_mean_rate를 만들고, 그 (c_ei, rE) 쌍을 :172-178에서 기록한 뒤 :180-184에서 갱신한다. 따라서 스냅샷의 c_ei와 rE는 일관된 쌍이다 — 정확. 스냅샷이 비면(max_iterations=0 같은 엣지) best 선택을 건너뛰고 마지막 c_ei를 그대로 쓴다 → 옛 동작 보존.

**Q5. `_evaluate_candidates_fc_corr`의 sim_duration_ms = eib_posthoc_duration_ms(720s) 적절한가?**
- 코드 근거: `part1_fic.py:354-355` `sim_duration_ms = int(cfg.eib_posthoc_duration_ms)`.
- 판정: ⚠️
- 설명: 후보 선택 기준이 FC corr이므로 안정적 FC 추정을 위해 긴 시뮬(720s)이 필요하다 — 1s(fic_step_duration_ms)로는 FC가 무의미. 트레이드오프: (1) 비용 — 후보 10개×720s는 무겁다(vmap으로 완화), (2) **목적 혼합** — FIC의 1차 목표는 firing-rate인데 선택 기준은 FC corr이라 두 목표가 섞인다. 결과적으로 "rE 오차 최소 top_k 중 FC corr 최대"를 고른다.
- 권고: 비용 부담 시 후보 수/duration 조정 고려. 목적 혼합은 의도된 설계로 보임.

**Q6. vmap 경로: eqx.tree_at이 c_ei만 교체. wLRE/wFFI/init_dynamics/bold_history 모두 base에서 상속?**
- 코드 근거: `part1_fic.py:362-365` `bundle_in.to_tvb_state`로 공유 sim_state·monitor 1회 구성, `:390-391` c_ei leaf만 교체.
- 판정: ✅
- 설명: sim_state는 bundle_in 파라미터(wLRE/wFFI/init_dynamics)와 bold_monitor로 만들어지고, 후보별로 c_ei만 바꾼다 → 나머지는 모두 bundle_in에서 공유 상속. FIC 단계라 wLRE/wFFI는 아직 ones(또는 warmup) 그대로이므로 일관적.

**Q7. 캐시 키가 best-선택 로직 변경을 포함하는가?**
- 코드 근거: `part1_fic.py:68-77` cache_name = cache_tag + rE + eta + steps + dur + skip + cef + `fingerprint(bundle_init)`. best 선택은 `_cached_run`(`:79-82`) **내부**.
- 판정: ⚠️
- 설명: best-선택 알고리즘 변경은 캐시되는 함수 본문 안에서 일어나고 키에는 직접 토큰이 없다. 따라서 무효화는 `cache_tag`가 내포한 `cfg.cache_version`(`config.py:28`, 현재 `..._p31_p32capfix`)에 전적으로 의존한다. 로직만 바꾸고 cache_version을 안 올리면 **patch 이전 캐시가 재사용**될 수 있다.
- 권고: best-선택 로직 변경 시 반드시 cache_version 또는 cache_name에 토큰 추가.

---

### Part 2 — EIB (`part2_eib.py`)

**Q1. eta 스케줄 `(step+1)/max_iter × max_lr` — peak 시점과 overshoot?**
- 코드 근거: `part2_eib.py:203-205`.
- 판정: ⚠️
- 설명: step_index=k에서 eta=(k+1)/N·max_lr. 최대는 k=N−1일 때 eta = N/N·max_lr = **max_lr**(=0.002). 즉 가장 큰 가중치 업데이트가 **마지막 step**에 일어난다. corr이 중간에 수렴했더라도 후반 큰 step이 overshoot를 유발할 수 있다.
- 완화: 스냅샷을 매 50 step 저장하고(`:261-264`) post-hoc가 window_corr 최댓값 스냅샷을 고르므로(`:346`) 후반 overshoot 결과는 자동 배제된다.
- 권고: 후반 overshoot 폭이 큰 데이터셋에선 cosine decay 등 대체 스케줄 검토.

**Q2. `_eib_update_rule`: wLRE += η·fc_diff·row_rmse, wFFI -= 동일. 동시 업데이트 안정성?**
- 코드 근거: `part2_eib.py:533-538`. `fc_diff = target − pred`.
- 판정: ✅(경미한 주의)
- 설명: pred<target(저연결)이면 fc_diff>0 → wLRE↑(흥분↑) & wFFI↓(전향억제↓). 둘 다 FC를 **같은 방향**으로 밀어 실효 gain이 2배가 되는 셈이나, eta가 작고(≤0.002) `_clip_sym`이 [0,w_max]·mask·대칭화로 매 step 제약하므로 발산하지 않는다.
- 권고: 생략(eta 상향 시에만 주의).

**Q2-b. `_clip_sym`의 w_max 상한 적용 여부**
- 코드 근거: `part2_eib.py:541-544` `jnp.clip(w, 0.0, w_max) * sc_mask` 후 `0.5·(w+w.T)`. 주석 `# restore w_max cap (revert patch19)`.
- 판정: ✅
- 설명: 이전 버전에서 상한이 None이던 것이 복원되어, 탐색 중 wLRE/wFFI가 `[0, w_max=1.5]`로 매 step bound된다. 0 floor + w_max ceil + mask + 대칭화 모두 적용. (대칭화가 mask 뒤라는 순서 이슈는 sc_mask가 대칭이면 무해 — Q 참고.)

**Q3. c_ei_frozen이 bundle_in.params에서 정확히 읽히는가? freeze=True가 EIB에 미치는 영향?**
- 코드 근거: `part2_eib.py:180` `if not bundle_in.params.c_ei_frozen:` 가드로 내부 FIC 갱신(`:181-191`) 스킵.
- 판정: ✅
- 설명: frozen 플래그가 정확히 읽힌다. freeze_c_ei_after_fic=True면 FIC 출력 ParamSet.c_ei_frozen=True가 되고, EIB는 c_ei 내부 갱신을 건너뛴다(가중치만 학습). 정상.

**Q4. bold_rolling_buffer 시드: get_fc_seed_window shape이 (win,1,n)과 맞는가?**
- 코드 근거: `pipeline_contracts.py:351-372`가 `(n_samples, n_nodes)` 2D 반환(부족 시 zero-pad), `part2_eib.py:110-112`가 `.reshape((win,1,n))`.
- 판정: ✅
- 설명: 헬퍼는 2D를 반환하고 reshape로 3D 채널 축을 추가한다 — 일치. 시드가 짧으면 앞을 0으로 패딩.

**Q5. 스냅샷이 하나도 저장되지 않을 위험?**
- 코드 근거: 저장 조건 `part2_eib.py:261` `(step+1)%eib_snapshot_save_interval==0`. 빈 경우 처리 `:332-342`.
- 판정: ✅
- 설명: 50 step 전에 non-finite로 break하거나 max_iter<50이면 스냅샷이 빌 수 있으나, 그 경우 post-hoc가 `window_best_bundle`(매 step 추적, `:256-259`) settle로 폴백한다 — 안전.

**Q6. post-hoc: warmup state에서 재시뮬인가 snapshot의 init_dynamics에서인가?**
- 코드 근거: `part2_eib.py:289` `warmup_bundle=bundle_in`, `:360-371` snapshot params를 warmup_bundle의 init_dynamics/bold_history/window/internal/delay 위에 graft.
- 판정: ✅
- 설명: 스냅샷의 **파라미터만** 취하고 시작 state는 warmup_bundle(=EIB 입력 = FIC 종료 state)에서 가져온다. 즉 탐색 trajectory의 transient init_dynamics는 버리고 깨끗한 일관 시작점에서 재시뮬 — 원본 eval_fc() 원리. 함의: 스냅샷이 누적해온 내부 state 의존성이 제거돼 파라미터 자체의 FC 성능을 공정 평가.

---

### Part 3A — Full-matrix gradient (`part3_gradient.py`)

**Q1. optimizer chain `zero_nans → clip_by_global_norm(0.1) → adamaxw(lr)` 적절?**
- 코드 근거: `part3_gradient.py:677-684`.
- 판정: ⚠️
- 설명: clip이 adamaxw **앞**이므로 raw gradient의 global L2 norm을 0.1로 제한한 뒤 적응 스케일링한다. human(425 노드)에서 wLRE/wFFI는 각 ~18만 entry라 norm 0.1은 상당히 보수적(작은 step). 정적으로 "적절"을 단정할 수 없는 판단 영역이나, 발산 방지 가드로는 합리적.
- 권고: 수렴이 너무 느리면 clip norm 상향 실험.

**Q2. Loss의 0.01 activity weight는 하드코딩인가 configurable인가?**
- 코드 근거: `part3_gradient.py:190-195`, `:205-210`에서 리터럴 `0.01 * act_l`. 실제 loss는 3-term(global/nodewise corr + rmse, 가중치 cfg `:191-193`) + 0.01·activity. plot 라벨 `:916` "1 − corr + 0.01×activity".
- 판정: ⚠️
- 설명: activity 가중치 0.01은 **하드코딩**(cfg 필드 아님). low-rank는 `cfg.lowrank_activity_weight`(0.01)를 쓰지만 full-matrix는 리터럴. 또한 plot 라벨이 실제 3-term loss와 불일치(라벨은 단일 corr+activity만 표기).
- 권고: cfg 필드(예: optimizer_activity_weight)로 승격하고 plot 라벨을 3-term으로 수정.

**Q3. c_ei는 BoundedParameter[0,20], wLRE/wFFI는 unbounded Parameter — 왜? 하류 보호?**
- 코드 근거: `part3_gradient.py:231-241`(c_ei BoundedParameter, wLRE/wFFI 평범 Parameter), 추출 `:252-257` sanitize → `_clean_weight_matrix` [0,w_max].
- 판정: ⚠️
- 설명: 최적화 중 wLRE/wFFI는 음수·과대값을 탐색할 수 있고, 그 raw 값으로 `compiled_model`이 시뮬한다. 상한 보호는 **best_params 추출 시점의 sanitize에서만** 적용된다. 즉 저장 파라미터는 [0,w_max]·mask·대칭이지만, 최적화 도중 시뮬에는 무제한 값이 들어가 비현실 동역학을 거칠 수 있다.
- 권고: 학습 안정성 위해 wLRE/wFFI도 BoundedParameter화 검토(동역학 변경 감수 필요).

**Q4. chunk_steps(노트북 override=1)의 JIT 재컴파일 빈도?**
- 코드 근거: 루프 `part3_gradient.py:709-713` `optimizer.run(current_state, max_steps=chunk)`. 경고 `:702-707`.
- 판정: ⚠️
- 설명: chunk 값이 매번 동일(=1)하므로 `optimizer.run`은 같은 max_steps로 1회만 컴파일된다 — **step마다 재컴파일은 아니다**. 다만 chunk=1은 outer Python 루프가 step당 1회 돌아 Python/디스패치 overhead가 크다(코드가 chunk<5에 경고).
- 권고: 수렴 dynamics 변경을 감수할 수 있으면 chunk 5~10 권장.

**Q5. best_params 추적: 매 chunk마다인가 loss 개선 시에만인가?**
- 코드 근거: `part3_gradient.py:722-725` `if step_loss < best_loss: best_params_dict = _snapshot_params(...)`.
- 판정: ✅
- 설명: chunk 경계마다 loss를 재고, **개선될 때만** 스냅샷 갱신.

---

### Part 3B — Low-rank gradient (`part3_gradient.py`)

**Q1. w_eff = clip(w_base+δUVᵀ,0,w_max)×sc_mask, 대칭화. 대칭화가 masking 전/후? 순서 중요?**
- 코드 근거: `part3_gradient.py:416-419` — clip+mask(`:416-417`) 후 `0.5·(w+w.T)`(`:418-419`).
- 판정: ⚠️
- 설명: 순서는 clip→mask→대칭화. sc_mask가 대칭이면 무해. 비대칭이면 0으로 마스킹된 (i,j)에 대칭화가 (j,i)의 비마스킹 값을 들여와 **마스크가 깨질 수 있다**. clip[0,w_max]은 두 ≤w_max 값의 평균도 ≤w_max라 상한은 유지. EIB `_clip_sym`도 동일 순서.
- 권고: sc_mask 대칭성 확인(보통 무방향 SC라 대칭). 비대칭이면 대칭화 후 재-mask 권장.

**Q2. factor_penalty가 네 factor 모두에 동등 적용?**
- 코드 근거: `part3_gradient.py:438-441` `factor_l = mean(lre_u²)+mean(lre_v²)+mean(ffi_u²)+mean(ffi_v²)`, `:447` `cfg.lowrank_factor_penalty * factor_l`.
- 판정: ✅
- 설명: 네 factor의 mean-square 합에 단일 penalty 계수 → 동등 적용.

**Q3. delta_scale=0.15 — 단일 weight 최대 perturbation? w_max로 bound?**
- 코드 근거: δ `part3_gradient.py:414-415` `delta_scale·(U@Vᵀ)`, 적용·clip `:416-417`.
- 판정: ✅
- 설명: U,V가 학습돼 δ의 단일 엔트리 크기는 선험적 무제한이지만, `w_eff = clip(w_base+δ, 0, w_max)`로 **결과 가중치가 [0, w_max=1.5]로 bound**. 따라서 실효 perturbation은 w_max에 의해 제한된다. factor_init=0.01과 factor_penalty가 δ 크기를 간접 억제.

**Q4. low-rank의 c_ei는 학습되는가 동결인가?**
- 코드 근거: `part3_gradient.py:375` `c_ei_frozen=False`, loss `:424` `jnp.clip(trainable.c_ei,0,20)`, 추출 `:524`.
- 판정: ✅
- 설명: c_ei는 LowRankTrainable의 leaf로 항상 학습된다(clip[0,20]). frozen 분기(`:521-522`)는 현재 c_ei_frozen=False라 사용되지 않음.

---

### Part 4 — DBS (`part4_dbs.py`)

**Q1. biphasic pulse가 charge-balanced인가? phase_duration_steps=1, dt=1ms일 때 실제 주파수?**
- 코드 근거: pulse 생성 `part4_dbs.py:652-663`(+amp 1 step, −amp 1 step), 주파수 `:586-589`.
- 판정: ⚠️(charge balance는 ✅)
- 설명: +A×1step + (−A)×1step = 0 → **charge-balanced 정상**. 주파수: biphasic_steps=2, period_steps = max(3, round(1000/(130·1))) = max(3, 8) = 8, freq_actual = 1000/8 = **125 Hz**. 요청 130Hz가 정수 step 양자화로 125Hz가 된다.
- 권고: 정확한 130Hz가 필요하면 dt 축소(예 0.5ms) 또는 비정수 period 처리. 코드는 freq_actual_hz를 출력해 투명.

**Q2. LFP = E + I 정의가 표준인가? 대안은?**
- 코드 근거: `part4_dbs.py:419-420`, 옵션 `:415-423`(E, I, E+I, E−I).
- 판정: ⚠️
- 설명: E+I는 합산 활동 proxy로 방어 가능하나 유일 표준은 아니다. 대안: E만, E−I(쌍극자), 시냅스 전류 가중합. 코드는 4종 observable을 지원하나 실행은 E_plus_I 고정(`:202`).
- 권고: 분석 목적에 따라 observable 선택을 노출 고려.

**Q3. BoundedSolver [0,1]이 E,I clip. amplitude=10이면 타겟 노드 즉시 saturation? 생물학적 해석?**
- 코드 근거: stim이 `inside_stim`으로 excitatory_input(시그모이드 전)에 가산 `part4_dbs.py:283`. 기본 amplitude=1.0(`config.py:145`).
- 판정: ⚠️(정보성)
- 설명: true_p_t에서 stim은 시그모이드 입력에 더해진다 → amp=10이면 sigmoid≈1로 포화, E는 (k_e−r_e·E)·1로 끌려가 solver [0,1] 상한 근처까지 상승. 즉 **노드가 최대 발화로 포화**. 생물학적으로는 supra-threshold 강제 구동에 해당. 기본값 1.0은 중간 수준.
- 권고: amp는 동역학 saturation을 고려해 설정(기본 1.0은 안전).

**Q4. pre vs during이 같은 초기 state에서 실행되는가? settle 구간?**
- 코드 근거: 단일 연속 시뮬 `part4_dbs.py:194-198`(t1=total_duration = pre+stim), 분할 `:347-351`.
- 판정: ✅
- 설명: pre(앞 60s)와 during(이어지는 60s)은 **하나의 연속 trajectory**를 시간 마스크로 나눈 것이다. during은 pre 끝 state에서 그대로 이어지며 별도 settle 없음. 함의: during 구간 앞쪽에 자극 onset transient가 포함된다.
- 권고: onset transient 영향을 빼려면 during 초반을 PSD에서 제외 검토.

**Q5. network.dynamics.dynamics monkey-patch가 JAX trace 에러에도 finally로 복원되는가?**
- 코드 근거: `part4_dbs.py:190` patch, `:192-200` try/finally, `:200` `network.dynamics.dynamics = original_dynamics`.
- 판정: ✅
- 설명: patch 후 prepare·compiled_model 호출이 try 안, 복원이 finally에 있어 trace 에러를 포함한 어떤 예외에도 원본 dynamics가 복원된다. 보장됨.

---

### Pipeline contracts (`pipeline_contracts.py`)

**Q1. fingerprint() 포함 필드? bold_window 포함? FIC best-선택이 bold_window를 바꾸면 fingerprint가 변하는가?**
- 코드 근거: `pipeline_contracts.py:429-448`.
- 판정: ✅
- 설명: 항상 c_ei·wLRE·wFFI·init_dynamics를 해시하고, 존재하면 bold_window(전체), bold_history(앞 8행), internal_state(키별 앞 64원소, 정렬), delay_history(앞 128원소), stage 문자열을 추가한다. **bold_window 포함됨**. FIC 출력 bundle의 bold_window가 바뀌면 그 fingerprint가 바뀌고 → downstream(EIB) 캐시 키가 바뀐다. 단 FIC **자신**의 캐시 키는 **입력** bundle fingerprint를 쓰므로 Part1-Q7의 staleness와는 별개.

**Q2. to_dict / from_dict가 np/jnp 양쪽을 round-trip에서 올바로 처리? dtype 손실?**
- 코드 근거: `to_numpy_dict`(`:94-100`), `_to_numpy_or_none`(`:462-465`), `from_dict`(`:387-398`), `_normalize_internal_state`(`:480-491`), `_to_numpy_metadata`(`:494-506`).
- 판정: ✅(주의)
- 설명: 파라미터·state 배열은 모두 `np.float32`로 강제 캐스팅된다(jnp→np 변환은 np.asarray로 안전). 따라서 원본이 float64였다면 **정밀도 손실** 발생. c_ei_frozen(bool) 보존. internal_state는 원 dtype 유지(noise float32, rng_key는 metadata에서 uint32 유지). round-trip 자체는 일관.
- 권고: float64 정밀이 필요한 값이 있으면 float32 강제를 재검토(현 파이프라인은 float32 일관이라 문제 낮음).

**Q3. advance()가 모든 필드를 전파하는가? 조용히 누락되는 필드?**
- 코드 근거: `pipeline_contracts.py:261-295`.
- 판정: ✅
- 설명: 6개 state 필드(params, init_dynamics, bold_history, bold_window, internal_state, delay_history) + stage + metadata 모두를 다룬다. 인자가 None이면 **기존 값 유지**(누락이 아니라 상속). Part3에서 `advance(new_params=..., next_stage="grad")`처럼 일부만 바꾸면 나머지는 warmup-start state에서 상속 — 의도된 동작. internal_state=None 상속은 직전 noise를 이어가므로 stale가 아니라 연속성 목적.

---

## 신규 발견 이슈

- **[NEW] (⚠️) Part3A activity weight 하드코딩 + plot 라벨 불일치** — `part3_gradient.py:194,209`의 `0.01`은 cfg 미연동(low-rank는 `cfg.lowrank_activity_weight` 사용). plot 라벨 `:916` "1 − corr + 0.01×activity"는 실제 3-term loss(global+nodewise corr+rmse)와 불일치. 본문 Part3A-Q2 참조.
- **[NEW] (⚠️) FIC top_k 잔존 vs EIB top_k 제거 비대칭** — `part1_fic.py:209` `top_k=10`은 다중 후보→fc_corr→best 선택을 유지. 반면 EIB는 `part2_eib.py:344-346`에서 window_corr 단일 best만 사용. 버그는 아니나 두 단계 best-선택 정책이 다름.
- **[NEW] (⚠️) `cfg.cache_version`이 cache_name에 직접 미참조** — 모든 단계 cache_name(`part1_fic.py:69-77`, `part2_eib.py:59-67`, `part3_gradient.py:113-121`,`:330-338`)이 `data['cache_tag']`만 포함. 무효화가 cache_tag↔cache_version 연결(data_loader 소관, 본 검토 범위 밖)에 의존하는 단일 실패점. 현재 `cache_version`에 `_p32capfix`는 있으나 `_p30` 토큰은 누락(이전 patch가 `_p31`로 우연히 무효화 처리).
- **[NEW] (정보) DBS during 구간에 onset transient 포함** — pre/during이 단일 연속 trajectory의 시간 분할(`part4_dbs.py:347-351`)이라 during 앞부분에 자극 onset 과도응답이 섞인다. 본문 Part4-Q4 참조.
- **[NEW] (정보) `_compute_activity_regularization`가 마지막 500 step 가정** — `part3_gradient.py:855` `data[-500:]`. 시뮬 길이가 500 step 미만이면 전체를 쓰지만(슬라이스 안전), window가 짧은 설정에선 평균 구간이 의도와 다를 수 있음.
- **[NEW] (정보) rE_max 출처 일관성 확인** — FIC·activity reg가 `WilsonCowanEIB.DEFAULT_PARAMS.rE_max_hz`(=20.0, `model.py:59`)를 읽고 이는 `cfg.wc_rE_max_hz`(20.0)와 일치. cfg에서 wc_rE_max_hz를 바꿔도 이 헬퍼들은 DEFAULT_PARAMS를 보므로 불일치 소지 — 현재 값 같아 영향 없음.

---

## 변경 금지 확인

- 코드를 실행·임포트·trace하지 않았다. 정적 리딩만 수행.
- 어떤 소스 파일도 수정하지 않았다.
- 과학적 상수·방정식·하이퍼파라미터를 변경하지 않았다(c_ei 상한 20, eta 스케줄, delta_scale 0.15, clip norm 0.1, w_max 1.5, DBS 주파수 등 모두 원본 그대로 보고).
- 주석과 코드가 불일치하는 지점(Part3 plot 라벨, FIC freeze 적용 범위 등)은 **코드 동작 기준**으로 명시 플래그했다.
