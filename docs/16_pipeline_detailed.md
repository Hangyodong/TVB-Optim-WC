# 16 — Detailed Pipeline (mechanics, formulas, file:line)

Companion to `13_pipeline_reference.md` (overview + file map). This doc goes
deep: exact equations, loop algorithms, cache keys, bounds, and where each lives
in code. Reflects HEAD on `patch30-remove-posthoc-topk` (incl. patch31
`eib_update_interval` + the `_clip_sym` w_max cap restore).

---

## 0. Pipeline at a glance

```
data_loader.load_data ──> build_network (+warmup 300s) ──> StateBundle(warmup)
        │                                                        │
        ▼                                                        ▼
   Part 1 FIC  ── tune c_ei → target firing rate ───────────> bundle(fic)
        │                                                        │
        ▼                                                        ▼
   Part 2 EIB  ── rolling-window FC tune wLRE/wFFI (+c_ei) ──> bundle(eib)
        │                                                        │
        ▼                                                        ▼
   Part 3A full-matrix gradient ── 3-term FC loss ───────────> bundle(part3)
   Part 3B low-rank gradient    ── δ = scale·U Vᵀ ───────────> bundle(part3b)
        │                                                        │
        ▼                                                        ▼
   Part 4 DBS  ── biphasic stim, pre-vs-during PSD/beta ─────> figures/CSV
```

State flows through one `StateBundle`/`ParamSet` contract
(`pipeline_contracts.py`). Each stage is cached by a content-keyed
`cache_name` (dataset `cache_tag` + stage hyperparams + input bundle
`fingerprint()`).

Default mouse atlas = 42 regions (`Atlas_43.txt`); notebook currently set to
human (Schaefer400 + 25 subcortex = 425 nodes).

---

## 1. Data layer — `data_loader.py`

`load_data(cfg)` → dict keys:
`weights, lengths, delays, fc_target, sc_mask, region_labels, n_nodes,
cache_tag, cache_dir, graph`.

**SC preprocessing** (`data_loader.py:43-53`), in order:
1. zero diagonal
2. `sc_mask = (weights > 0).astype(float32)` — binary structural mask
3. log: `weights = log1p(weights + 0.5)`
4. max-norm: `weights /= weights.max()` (if max > 0)
5. re-mask: `weights *= sc_mask`

**Delays** (`:55`): `delays = lengths / cfg.tract_conduction_speed`
(speed = 3.0 mm/ms → delays in ms).

**FC target**: loaded from CSV, no preprocessing (`:38`).

**Graph** (`:64-68`): `DenseDelayGraph(weights, delays, region_labels)` — carries
tract delays into integration.

**cache_tag** (`:61,218-232`):
`{cache_version}_N{n_nodes}_sc{fp(weights)}_fc{fp(fc_target)}`, where fp = SHA1
of top-left 8×8 sample, 10 hex chars. `cache_version` lives in `config.py:28`
(currently `..._p27_p31_p32capfix`); bumping it invalidates every stage cache.

---

## 2. Model — `model.py`

State vars `(E, I)` ∈ [0,1] (BoundedSolver clips both). Initial `(0.2, 0.1)`.
Aux outputs: `S_e, S_i, rE_hz, rI_hz`.

**Dynamics** (`model.py:65-117`):
```
I_e = α_e·(c_ee·E − c_ei·I + P + I_ext − θ_e + LRE)
I_i = α_i·(c_ie·E − c_ii·I + Q       − θ_i + λ·FFI)
S_e = c_e / (1 + exp(−a_e·clip(I_e − b_e, −500, 500)))
S_i = c_i / (1 + exp(−a_i·clip(I_i − b_i, −500, 500)))
rE_hz = rE_max_hz·S_e ;  rI_hz = rI_max_hz·S_i
dE/dt = (−E + (k_e − r_e·E)·S_e) / τ_e
dI/dt = (−I + (k_i − r_i·I)·S_i) / τ_i
```
`c_ei` is **per-node** (shape `(n_nodes,)`); it is the FIC/EIB knob.

**EIBLinearCoupling** (`:120-136`): decouples long-range excitation and
feedforward inhibition. `pre()` scales source E by `wLRE` and `wFFI`
independently → stacked output of size 2 = `[wLRE·E_src, wFFI·E_src]`.

**build_network** (`:139-210`):
1. graph = `data["graph"]` (DenseDelayGraph) else `DenseGraph(weights)`
2. dynamics: `c_ei = c_ei_init·ones(n)` (`wc_c_ei_init=10`), all `wc_*` set
3. coupling: `wLRE = wFFI = ones((n,n))`
4. noise: `AdditiveNoise(sigma=0.01, apply_to="E")` — **E only**
5. **warmup**: simulate `warmup_duration_ms` (300 s) → `warmup_result.data[-1]`
   = settled `(2, n)` state
6. BOLD monitor seeded with warmup history; `period=TR`, downsample 4.0, voi=0.

---

## 3. State contract — `pipeline_contracts.py`

**ParamSet** (`:34-119`): `c_ei (n,)`, `wLRE (n,n)`, `wFFI (n,n)`, `c_ei_frozen: bool`.

**StateBundle** (`:166-206`): `_params`, `_init_dynamics (2,n)`,
`_bold_history` (HRF continuation), `_bold_window` (FC seed),
`_internal_state` (noise samples…), `_delay_history` (delay-graph snapshot),
`_stage`, `_metadata` (rng_key/seed…).

**fingerprint()** (`:429-448`): SHA1 (12 hex) over c_ei, wLRE, wFFI,
init_dynamics, bold_window, bold_history[:8], internal_state[:64/key],
delay_history[:128], stage string. Drives downstream cache keys.

**sanitize()** (`:79-85`) → `_clean_c_ei` + `_clean_weight_matrix`, keeps frozen flag:
- `_clean_c_ei` (`:468-470`): NaN→6.0, +inf→20, −inf→0, clip **[0, 20]**.
- `_clean_weight_matrix` (`:473-477`): NaN→0, +inf→w_max, −inf→0, clip
  **[0, w_max]**, ×sc_mask, then symmetrize `0.5·(W + Wᵀ)`.

`advance(...)` creates next-stage bundle (propagates state caches before stage
flips). `capture/restore/advance_internal_state` manage noise RNG continuity.

---

## 4. Part 1 — FIC (`part1_fic.py`)

**Goal**: per-node `c_ei` so mean excitatory rate → `fic_target_firing_rate_hz`
(4.0 Hz) while keeping FC structure.

**Entry `run_fic`** (`:41-95`). cache_name (`:69-77`):
`fic_{cache_tag}_rE{target}_eta{lr}_steps{iters}_dur{step_ms}_skip{skip_tr}[_cef1]_fp{fp}`.
(`_cef1` only if `freeze_c_ei_after_fic`.)

**Loop `_run_fic_loop_pure`** (`:100-205`), per iteration:
1. `step_result = step_model(state)` — simulate `fic_step_duration_ms` (1000 ms)
2. BOLD → firing rates, drop `fic_step_skip_tr` TRs (`:158-160`)
3. `current_mean_rate = mean(rE_hz)`
4. carry state: `state.dynamics = step_result.data[-1]`

**c_ei update** (`:180-184`) — rI-gated:
```
rate_error  = mean_rE − fic_target_firing_rate_hz
update_delta = fic_learning_rate · mean_rI · rate_error      # gated by rI
c_ei_new    = clip(c_ei + update_delta, 0.0, 20.0)
```
⚠ If `mean_rI → 0` the controller stalls (delta → 0). Clip cap 20 is an
arbitrary guard (2× init), no physiological basis.

**Snapshot** (`:172-178`): record `(step, c_ei, mean_rE_hz)` **before** applying
the update.

**Early stop** (`:187-205`): `|err| < fic_early_stop_tolerance_hz` (0.10) for
`fic_early_stop_patience` (500) consecutive steps.

**top_k selection** (`:207-254`) — *FIC still uses top_k* (unlike EIB):
1. rank snapshots by `|rE − target|`, keep `top_k=10` (`:209`)
2. parallel `fc_corr` of the 10 candidates → pick `argmax(corr)` (`:243-246`)
3. restore best `c_ei` into state (`:254`)

**Return** (`:289-310`): `bundle.advance(next_stage="fic", c_ei_frozen=freeze flag)`
carrying final dynamics + bold history/window + internal/delay state.
Reports `final_rE_hz`.

**Plots**: FIC results 2×2 (pre/post E dynamics, rE convergence, BOLD) +
FC matrices 1×3 (target / pre-opt / post-FIC with corr·rmse).

---

## 5. Part 2 — EIB (`part2_eib.py`)

**Goal**: rolling-window FC tuning of `wLRE`/`wFFI` (and `c_ei` if not frozen),
then post-hoc validation by full re-simulation from the warmup state.

**Entry `run_eib`** (`:42-79`). cache_name (`:59-67`):
`eib_{cache_tag}_win{window}_etaF{fic_lr}_etaE{weight_lr}_steps{iters}_frozen{0|1}_fp{fp}`.

**Loop `_run_eib_loop_pure`** (`:85-318`):

*Rolling BOLD window* (`:110-112,171-174`): buffer `(eib_bold_window_samples=150, 1, n)`;
each step `roll(-1)` then set `[-1,0,:] = bold_vector`.

*Predicted FC* (`:497-505`): z-score columns, `fc = zᵀz/(n−1)`, clip[−1,1],
zero diagonal.

*c_ei internal FIC* (`:180-191`, only if **not** frozen):
```
fic_delta = eib_internal_fic_learning_rate · mean_rI · (mean_rE − fic_target)
c_ei = clip(c_ei + fic_delta, 0, 20)
```

*Weight update `_eib_update_rule`* (`:524-538`):
```
fc_diff  = target_fc − pred_fc            # finite-guarded
row_rmse = rmse(target_fc, pred_fc, axis=1)[:, None]
wLRE_new = _clip_sym(wLRE + eta·fc_diff·row_rmse, sc_mask, w_max)
wFFI_new = _clip_sym(wFFI − eta·fc_diff·row_rmse, sc_mask, w_max)   # opposite sign
```

*Eta schedule* (`:203-205`): `eta = (step+1)/eib_max_iterations · eib_max_weight_learning_rate`
— linear ramp, **peaks at the last step** (⚠ late overshoot risk; best-snapshot
selection mitigates).

*Update interval* (patch31, `:207`): update fires only when
`step % eib_update_interval == 0`; weights held constant between.

*`_clip_sym`* (`:541-544`, **w_max cap restored**):
```
w = nan_to_zero(w)
w = clip(w, 0.0, w_max) · sc_mask      # cap restored (revert patch19)
return 0.5·(w + wᵀ)                    # mask BEFORE symmetrize
```
⚠ order is mask→symmetrize; if sc_mask were asymmetric, symmetrize could leak
across the mask. (Mask is symmetric in practice.)

*Snapshots* (`:261-264`): every `eib_snapshot_save_interval` (50) steps store
`(bundle_dict, iteration, window_corr)`.

**Post-hoc `_run_posthoc_validation`** (`:321-416`) — patch30:
1. `best_snap_idx = argmax(snapshot_window_corrs)` (`:344-347`) — single best,
   no more top_k.
2. graft snapshot params onto **warmup** bundle (reset state cache, keep warmup
   init_dynamics/bold_history/delay_history) (`:358-373`).
3. re-simulate `eib_posthoc_duration_ms` (720 s, skip 20 TR) → full-length FC →
   `true_corr = fc_corr(fc, fc_target)`, true RMSE.

**Return** (`:303-318`): bundle(`next_stage="eib"`) + histories
`fc_correlations`, `fc_rmse_values`, pre/post FC + corr/rmse, `best_iteration`,
neural traces.

**Plots**: convergence (window corr + RMSE, best marked) + FC 1×3
(target / pre-EIB / post-EIB validated).

---

## 6. Part 3 — Gradient (`part3_gradient.py`)

Shared loss (3-term FC + activity reg):
```
L = α·L_global_corr + β·L_nodewise_corr + γ·L_rmse + 0.01·L_activity
```
- α, β, γ from cfg (defaults 0.4 / 0.4 / 0.2).
- **activity weight 0.01 is hardcoded** (`:194`), not a config field — plot
  labels may not show this 4th term.
- `L_activity` = L2 of mean excitatory firing vs target (`:853-856`).

### 6A Full-matrix (`:182-...`)
- optimizer (`:677-684`): optax `chain(zero_nans, clip_by_global_norm(0.1),
  adamaxw(optimizer_learning_rate=0.002))`.
- steps: `optimizer_max_steps=200`, chunked by `optimizer_chunk_steps=5`.
- FC window: `optimizer_bold_window_tr=720`, skip 8.
- params: `c_ei` BoundedParameter[0,20] (always trained, `c_ei_frozen=False`),
  `wLRE`/`wFFI` plain Parameters (**unbounded during optimization**).
- effective weight: `clip(w_base+δ, 0, w_max)·sc_mask`, symmetrize.
- ⚠ in-optimization sim sees **raw unbounded** wLRE/wFFI; bounds enforced only at
  best-params `sanitize()` (`:257`).
- cache: `grad_{cache_tag}_TR{win}_SKIP{skip}_STEPS{steps}_LR{lr}_frozen{0|1}_fp{fp}`.

### 6B Low-rank (`:330-...`)
- correction: `δ = lowrank_delta_scale·(U @ Vᵀ)`, `delta_scale=0.15`.
- rank `lowrank_rank=6`; U,V `(n, rank)` ~ `N(0, factor_init=0.01)`; seed 17.
- learned: U,V (LRE+FFI pairs) and `c_ei` (bounded[0,20]); base wLRE/wFFI fixed,
  modified only via rank-6 δ.
- factor penalty `lowrank_factor_penalty=1e-4` on `mean(U²+V²)` (`:438-447`).
- effective: `clip(w_base+δ, 0, w_max)·sc_mask`, symmetrize → final weight bound
  by w_max=1.5 regardless of U,V magnitude.
- steps 120, window 96 TR, lr 0.002. optimizer same adamaxw chain.
- cache: `grad_lowrank_{cache_tag}_rank{r}_TR{win}_STEPS{steps}_LR{lr}_ds{scale}_frozen{0|1}_fp{fp}`.

---

## 7. Part 4 — DBS (`part4_dbs.py`)

**Injection** (`:576-665`): biphasic, charge-balanced.
- requested `dbs_stimulation_frequency_hz = 130`; **actual** =
  `1000/period_ms` where `period_steps = round(1000/(freq·dt))` (`:587`) →
  integer quantization → ~125 Hz at dt=1 ms. (Logged.)
- pulse: `+A` for `phase_duration_steps` (1 step) then `−A` (`:661-662`); gap of
  zeros fills rest of period; net charge ∫ = 0.
- amplitude `dbs_pulse_amplitude=1.0`; targets `dbs_target_regions` dict.
- mode `true_p_t` (default): added to excitatory input directly; alts
  `tvb_default`, `hybrid`.
- **restoration** (`:199-200`): `finally: network.dynamics.dynamics =
  original_dynamics` — guaranteed revert even on failure.

**Analysis** (`:668-719`): Welch PSD, Hann window, nperseg=1 s, 50% overlap,
`scaling="spectrum"`, f_max=`dbs_psd_max_frequency_hz=100`.
- beta band: `[dbs_beta_band_low_hz=13, dbs_beta_band_high_hz=30]` (`:712`).
- beta ratio = `trapz(psd[beta]) / trapz(psd[total])` pre vs during (`:718-719`).
- pre = 60 s before onset, during = 60 s stim; observable = E+I LFP.

**Outputs**: CSV (`lfp_timeseries`, `psd_pre_vs_during`) + 5 plots per target
(LFP timeseries, full stim train, 10-pulse zoom, PSD pre/during w/ beta shaded,
LFP pre/during segment).

---

## 8. Cache & reproducibility notes

- Every stage cache keyed on `data['cache_tag']` (⊃ `cache_version`) +
  stage hyperparams + input `bundle.fingerprint()`. Logic-only changes that
  don't alter any of these are **not** auto-invalidated → bump `cache_version`
  (`config.py:28`) when changing stage *algorithms*.
- `cache_version` history embeds patch tokens; `_p30` is absent (covered by
  `_p31`); `_p32capfix` added for the `_clip_sym` cap restore.
- RNG: `bundle_rng_seed=42`, noise on E only; `advance_internal_state` splits the
  key per stage to keep noise streams reproducible.

---

## 9. Known caveats (see `14_logic_health_check.md`)

| area | note |
|---|---|
| FIC | controller stalls if mean_rI→0; c_ei cap 20 arbitrary; **still uses top_k** |
| EIB | eta peaks at last step (late overshoot); mask→symmetrize order assumes symmetric mask |
| Part 3 | activity weight 0.01 hardcoded (not in plot labels); in-opt weights unbounded, bounded only at sanitize |
| Part 3B | symmetrize after mask; δ bounded via w_max clip |
| DBS | actual freq ~125 Hz (integer step quantization of 130 Hz request) |
