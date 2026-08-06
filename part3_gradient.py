"""
part3_gradient.py

Stage 3
-------
- Part 2 → Part 3 입력을 StateBundle로 통일한다.
- full-matrix optimizer는 구버전 notebook 로직처럼 best params를 바로 반환한다.
- legacy notebook의 old-style 호출도 계속 허용한다.
"""
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.types import BoundedParameter, Parameter
from tvboptim.types.stateutils import combine_state, partition_state
from tvboptim.utils import cache

from config import Config
from model import WilsonCowanEIB
from remat_scan_patch import remat_scan, remat_scan_enabled
from part1_fic import (
    _compute_beta_power_ratio,
    _compute_normalized_psd,
    _plot_beta_summary,
)
from pipeline_contracts import (
    ParamSet,
    StateBundle,
    advance_internal_state,
    build_bundle_from_legacy_state,
    capture_internal_state,
    eval_fc_multiseed,
    extract_bold_window,
    sync_network_delay_history,
    update_bold_history,
    weighted_corr_loss,
    weighted_rmse_loss,
    prepare_block_corr_terms,
    block_corr_loss,
    block_rmse_loss,
    compute_block_corrs,
    plot_block_corr_bars,
)


# ══════════════════════════════════════════════════════════════
# 공용: StateBundle 시뮬레이션
# ══════════════════════════════════════════════════════════════

def compute_simulated_fc(
    network,
    bundle: StateBundle,
    cfg: Config,
    sim_duration_ms: int = 300_000,
    skip_tr: int = 60,
) -> np.ndarray:
    fc_matrix, _ = _simulate_bundle(network, bundle, cfg, sim_duration_ms, skip_tr)
    return fc_matrix


def _simulate_bundle(
    network,
    bundle: StateBundle,
    cfg: Config,
    sim_duration_ms: int,
    skip_tr: int,
):
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    sim_model, sim_state = bundle.to_tvb_state(
        network, solver, t1=sim_duration_ms, dt=cfg.integration_dt_ms
    )
    bold_monitor = bundle.build_bold_monitor(cfg)

    fc_matrix, sim_result = eval_fc_multiseed(
        sim_model, sim_state, bold_monitor, _compute_fc_from_bold_output, skip_tr,
        int(cfg.bundle_rng_seed), int(getattr(cfg, "fc_eval_n_seeds", 1)))
    return np.asarray(fc_matrix, dtype=np.float32), np.asarray(sim_result.data, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# Part 3 — Full-matrix gradient optimization
# ══════════════════════════════════════════════════════════════

def run_gradient_optimization(
    network,
    bundle_in: StateBundle = None,
    eib_results=None,
    cfg: Config = None,
    data: dict = None,
    warmup_result=None,
    warmup_bundle: StateBundle = None,  # Patch 26: step-0 warmup start
) -> StateBundle:
    """
    Full-matrix gradient 최적화를 실행하고 구버전 notebook 로직처럼 raw optimized StateBundle을 반환한다.
    """
    bundle_eib = _coerce_gradient_bundle(bundle_in, eib_results, cfg, data, stage="eib")

    # part2(EIB) 최종 상태를 그대로 시작점으로 사용한다 (full-state passthrough).
    # 이전 Patch 26은 warmup(step0) 상태 + EIB params로 재시작했으나, part1→part2와
    # 동일하게 직전 단계의 settled 상태를 그대로 이어받도록 되돌린다.
    # warmup_bundle 인자는 하위호환을 위해 받지만 더 이상 상태 치환에 쓰지 않는다.
    bundle_start = bundle_eib

    # #1: loss 가중치를 캐시 키에 포함 → α(global corr)/γ(rmse)/activity/block-corr
    #     변경 시 캐시 무효화. (이전엔 누락되어 가중치만 바꾸면 stale grad 캐시 재사용)
    def _wtag(v):
        return str(v).replace('.', 'p').replace('-', 'm')
    loss_tag = (
        f"a{_wtag(cfg.optimizer_global_corr_weight)}"
        f"_g{_wtag(cfg.optimizer_rmse_weight)}"
        f"_rb{int(cfg.optimizer_rmse_block)}"
        f"_act{_wtag(cfg.optimizer_activity_weight)}"
        f"_bc{_wtag(cfg.corr_block_weight_cc)}"
        f"-{_wtag(cfg.corr_block_weight_cross)}"
        f"-{_wtag(cfg.corr_block_weight_subsub)}"
    )
    cache_name = (
        f"grad_{data['cache_tag']}"
        f"_TR{cfg.optimizer_bold_window_tr}"
        f"_SKIP{cfg.optimizer_bold_skip_tr}"
        f"_STEPS{cfg.optimizer_max_steps}"
        f"_LR{_wtag(cfg.optimizer_learning_rate)}"
        f"_{loss_tag}"
        f"_frozen{int(bundle_start.params.c_ei_frozen)}"
        f"_optpersist"   # Adam momentum now persists across chunks → invalidate old cache
        f"_fp{bundle_start.fingerprint()}"
        + (f"_seed{int(getattr(cfg, 'fc_eval_n_seeds', 1))}" if int(getattr(cfg, 'fc_eval_n_seeds', 1)) > 1 else "")
    )

    @cache(cache_name, redo=False)
    def _cached_run():
        return _run_full_gradient_pure(network, bundle_start.to_dict(), cfg, data)

    result = _cached_run()
    bundle_grad = StateBundle.from_dict(result["bundle"])
    bundle_grad.apply_to_network(network)

    _plot_gradient_results(
        bundle_grad,
        np.asarray(result["loss_history"]),
        data,
        cfg,
        pre_opt_fc=np.asarray(result["pre_opt_fc"]),
        post_opt_fc=np.asarray(result["post_opt_fc"]),
        title="Part 3 — Full-matrix Gradient Optimization",
    )
    return bundle_grad


def _run_full_gradient_pure(
    network,
    init_dict: dict,
    cfg: Config,
    data: dict,
) -> dict:
    bundle_in = StateBundle.from_dict(init_dict)
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    t1_opt = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

    compiled_model, initial_opt_state = bundle_in.to_tvb_state(
        network,
        solver,
        t1=t1_opt,
        dt=cfg.integration_dt_ms,
    )

    clean_params = bundle_in.params.sanitize(data["sc_mask"], cfg.connectivity_weight_max)
    initial_opt_state.dynamics.c_ei = jnp.asarray(clean_params.c_ei)
    initial_opt_state.coupling.coupling.wLRE = jnp.asarray(clean_params.wLRE)
    initial_opt_state.coupling.coupling.wFFI = jnp.asarray(clean_params.wFFI)

    # freeze_c_ei_after_fic=True 이면 FIC 가 calibrate 한 c_ei(생리적, S_e gating 0.25 달성 ~2.0)를
    # Part3 도 고정 → wLRE/wFFI 만 튜닝. degeneracy(Part3 가 c_ei 를 과흥분 방향 0.2대로 drift →
    # DBS 등 downstream 예측이 그 아티팩트에 좌우) 차단. bundle 의 c_ei_frozen 플래그를 존중한다.
    c_ei_frozen = bool(getattr(bundle_in.params, "c_ei_frozen", False))
    bold_monitor_opt = bundle_in.build_bold_monitor(cfg)
    fc_target_safe = jnp.asarray(np.nan_to_num(data["fc_target"], nan=0.0))
    # subcortex-가중 edge 행렬(diag 0) — 표시용 corr metric에만 사용.
    fc_W = jnp.asarray(data["fc_edge_weight"])
    # RMSE는 블록/subcortex 가중 없이 표준 off-diag RMSE → (1-eye) 마스크.
    plain_rmse_mask = 1.0 - jnp.eye(fc_target_safe.shape[0], dtype=fc_W.dtype)
    # block-split corr: cc/cross/ss를 edge수 무관 동등 가중 → subcortex fit 균형.
    # subcortex 없는 atlas는 cc(=full off-diag)만 활성 → 기존 whole corr와 동치.
    block_corr_terms = prepare_block_corr_terms(
        data["fc_block_masks"],
        {"cc": cfg.corr_block_weight_cc,
         "cross": cfg.corr_block_weight_cross,
         "subsub": cfg.corr_block_weight_subsub},
    )
    # rmse 블록분할(옵션): corr와 동일 block 가중 재사용. False면 plain (1-eye) RMSE.
    rmse_block_terms = (
        prepare_block_corr_terms(
            data["fc_block_masks"],
            {"cc": cfg.corr_block_weight_cc,
             "cross": cfg.corr_block_weight_cross,
             "subsub": cfg.corr_block_weight_subsub},
        )
        if cfg.optimizer_rmse_block else None
    )

    def _rmse_term(fc):
        if rmse_block_terms is not None:
            return block_rmse_loss(fc, fc_target_safe, rmse_block_terms)
        return weighted_rmse_loss(fc, fc_target_safe, plain_rmse_mask)

    # Stage handoff: pre-opt FC = predecessor (EIB) stored post-FC for exact
    # plot continuity; fall back to a fresh sim if the bundle lacks it.
    pre_opt_fc = bundle_in.metadata.get("post_eib_fc_matrix", None)
    if pre_opt_fc is None:
        pre_opt_fc = compute_simulated_fc(
            network,
            bundle_in,
            cfg,
            sim_duration_ms=t1_opt,
            skip_tr=cfg.optimizer_bold_skip_tr,
        )

    def compute_loss(state):
        sim = compiled_model(state)
        bold = bold_monitor_opt(sim)
        fc = _compute_fc_differentiable(bold, cfg.optimizer_bold_skip_tr)
        L_global   = block_corr_loss(fc, fc_target_safe, block_corr_terms)
        L_rmse     = _rmse_term(fc)
        act_l      = _compute_activity_regularization(sim, cfg)
        return (
            cfg.optimizer_global_corr_weight     * L_global
            + cfg.optimizer_rmse_weight          * L_rmse
            + cfg.optimizer_activity_weight * act_l
        )

    def compute_loss_and_metrics(state):
        sim = compiled_model(state)
        bold = bold_monitor_opt(sim)
        fc = _compute_fc_differentiable(bold, cfg.optimizer_bold_skip_tr)
        L_global   = block_corr_loss(fc, fc_target_safe, block_corr_terms)
        L_rmse     = _rmse_term(fc)
        act_l      = _compute_activity_regularization(sim, cfg)
        total = (
            cfg.optimizer_global_corr_weight     * L_global
            + cfg.optimizer_rmse_weight          * L_rmse
            + cfg.optimizer_activity_weight * act_l
        )
        # 표시용 metric은 기존 whole edge-weighted corr 유지(블록 loss와 별개).
        full_corr = 1.0 - weighted_corr_loss(fc, fc_target_safe, fc_W)
        return total, L_global, act_l, full_corr

    try:
        initial_loss, _, _, init_corr = compute_loss_and_metrics(initial_opt_state)
        initial_loss = float(jnp.nan_to_num(initial_loss, nan=1e4))
        init_corr = float(jnp.nan_to_num(init_corr, nan=-1.0))
    except Exception as error:
        print(f"[WARN] initial loss evaluation failed: {error}")
        initial_loss, init_corr = 1e4, float("nan")

    print(
        f"[Part3] Initial loss: {initial_loss:.6f}  |  "
        f"Full Corr: {init_corr:.4f}"
    )
    print(
        f"  t1_opt={t1_opt}ms ({t1_opt/1000:.1f}s)  "
        f"max_steps={cfg.optimizer_max_steps}  chunk={cfg.optimizer_chunk_steps}  "
        f"lr={cfg.optimizer_learning_rate}  fc_skip_tr={cfg.optimizer_bold_skip_tr}"
    )

    if not c_ei_frozen:
        initial_opt_state.dynamics.c_ei = BoundedParameter(
            initial_opt_state.dynamics.c_ei, low=0.0, high=20.0
        )
    # frozen: leave c_ei as a plain array so the optimizer never updates it.
    initial_opt_state.coupling.coupling.wLRE = Parameter(
        initial_opt_state.coupling.coupling.wLRE
    )
    initial_opt_state.coupling.coupling.wFFI = Parameter(
        initial_opt_state.coupling.coupling.wFFI
    )

    best_params_dict, loss_history = _run_full_optimization_loop(
        compute_loss=compute_loss,
        compute_loss_and_metrics=compute_loss_and_metrics,
        initial_state=initial_opt_state,
        initial_loss=initial_loss,
        cfg=cfg,
        c_ei_frozen=c_ei_frozen,
        sc_mask_j=jnp.asarray(data["sc_mask"], dtype=jnp.float32),  # #2
        w_max=float(cfg.connectivity_weight_max),                   # #2
    )

    best_params = ParamSet(
        c_ei=np.asarray(best_params_dict["c_ei"], dtype=np.float32),
        wLRE=np.asarray(best_params_dict["wLRE"], dtype=np.float32),
        wFFI=np.asarray(best_params_dict["wFFI"], dtype=np.float32),
        c_ei_frozen=c_ei_frozen,
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)

    candidate_bundle = bundle_in.advance(
        new_params=best_params,
        next_stage="grad",
    )
    post_opt_fc, post_opt_neural = _evaluate_bundle_without_settle(
        network=network,
        bundle_in=candidate_bundle,
        cfg=cfg,
        sim_duration_ms=t1_opt,
        skip_tr=cfg.optimizer_bold_skip_tr,
    )

    # Stamp post-grad FC into metadata for downstream plot continuity.
    candidate_bundle = candidate_bundle.with_metadata({
        "post_grad_fc_matrix": np.asarray(post_opt_fc, dtype=np.float32),
        "post_grad_fc_corr": float(fc_corr(jnp.asarray(post_opt_fc), fc_target_safe)),
        "post_grad_fc_rmse": _compute_rmse_metric(post_opt_fc, data["fc_target"]),
    })

    _ns = max(1, int(getattr(cfg, "neural_cache_stride", 1)))   # cache 경량화(plot 전용 trace)
    return {
        "bundle": candidate_bundle.to_dict(),
        "loss_history": np.asarray(loss_history, dtype=np.float32),
        "pre_opt_fc": np.asarray(pre_opt_fc, dtype=np.float32),
        "post_opt_fc": np.asarray(post_opt_fc, dtype=np.float32),
        "post_opt_neural": np.asarray(post_opt_neural, dtype=np.float32)[::_ns],
    }


# ══════════════════════════════════════════════════════════════
# Part 3.5 — Per-node noise sigma (σ) optimization
# ══════════════════════════════════════════════════════════════
#
# Part 3(grad) 뒤에서 per-node additive noise σ 만 autodiff 로 최적화한다.
# 설계 근거
#   - σ 와 c_ei 는 둘 다 node 분산→FC 크기를 조절하는 축퇴쌍이다. 같은 stage 에서
#     joint 최적화하면 optimizer 가 축퇴 manifold 위를 방황한다. 그래서 Part3 에서
#     c_ei/wLRE/wFFI 를 먼저 맞추고, Part3.5 에서 그것들을 freeze(plain array)한 채
#     σ 만 잔차 FC error 에 맞춘다 → 축퇴 차단.
#   - σ 는 BoundedParameter[0, σ_max] 로 제약(큰 σ 가 RWW 를 폭발시킴, sqrt-diffusion).
#   - FC 는 분산 정규화된 상관이라 per-node σ 를 약하게만 구속한다 → 노드별 spread 를
#     L2(σ - mean σ) 로 정규화해 과적합을 억제(글로벌 레벨은 자유).
# 미분 가능성
#   - 적분에서 noise = σ·sqrt(dt)·dW, dW(=_internal.noise_samples)는 stage 경계에
#     고정 캡처된다 → dLoss/dσ 가 exact pathwise gradient. seed 고정 → 결정적 grad.
# 영속성
#   - 최적 σ 는 network.noise.params.sigma 에 in-place 주입한다. 이후 어떤 prepare()
#     /to_tvb_state() 도 이 σ 를 읽으므로 Part4 등 downstream 이 그대로 사용한다.
#     bundle.metadata["sigma_opt"] 에도 기록(캐시 hit 시 재주입용).

def run_sigma_optimization(
    network,
    bundle_in: StateBundle,
    cfg: Config,
    data: dict,
    sigma_per_node: bool = True,
    sigma_max: float = None,
    sigma_l2_weight: float = 0.01,
    sigma_lr: float = None,
    max_steps: int = None,
) -> StateBundle:
    """Part 3.5 — Part3(grad) 출력 bundle 을 받아 per-node noise σ 만 최적화한다.

    sigma_per_node=False 면 공유 스칼라 σ 1개만 free(파일럿: grad 생존 확인용).
    반환: σ 가 network 에 주입되고 metadata 에 기록된 StateBundle.
    """
    if not isinstance(bundle_in, StateBundle):
        raise TypeError("run_sigma_optimization requires a StateBundle bundle_in (Part3 output).")

    sigma0 = float(np.mean(np.asarray(network.noise.params.sigma, dtype=np.float32)))
    if sigma_max is None:
        sigma_max = max(0.1, sigma0 * 5.0)
    if sigma_lr is None:
        sigma_lr = float(cfg.optimizer_learning_rate)
    if max_steps is None:
        max_steps = int(cfg.optimizer_max_steps)

    def _wtag(v):
        return str(v).replace('.', 'p').replace('-', 'm')
    cache_name = (
        f"sigma_{data['cache_tag']}"
        f"_pn{int(sigma_per_node)}"
        f"_TR{cfg.optimizer_bold_window_tr}_SKIP{cfg.optimizer_bold_skip_tr}"
        f"_STEPS{max_steps}_LR{_wtag(sigma_lr)}"
        f"_smax{_wtag(round(float(sigma_max), 5))}_l2{_wtag(sigma_l2_weight)}"
        f"_rb{int(cfg.optimizer_rmse_block)}"
        f"_s0{_wtag(round(sigma0, 5))}"
        f"_optpersist"   # Adam momentum now persists across chunks → invalidate old cache
        f"_fp{bundle_in.fingerprint()}"
    )

    @cache(cache_name, redo=False)
    def _cached_run():
        return _run_sigma_opt_pure(
            network, bundle_in.to_dict(), cfg, data,
            sigma_per_node=bool(sigma_per_node), sigma_max=float(sigma_max),
            sigma_l2_weight=float(sigma_l2_weight), sigma_lr=float(sigma_lr),
            max_steps=int(max_steps),
        )

    result = _cached_run()
    sigma_opt = np.asarray(result["sigma_opt"], dtype=np.float32)

    # in-place 주입 → downstream prepare() 가 최적 σ 를 읽음. cache hit 시에도 동일 적용.
    network.noise.params.sigma = jnp.asarray(sigma_opt)

    bundle_sigma = StateBundle.from_dict(result["bundle"])
    bundle_sigma.apply_to_network(network)

    _plot_sigma_results(
        sigma_opt=sigma_opt,
        sigma0=sigma0,
        loss_history=np.asarray(result["loss_history"]),
        pre_fc=np.asarray(result["pre_fc"]),
        post_fc=np.asarray(result["post_fc"]),
        data=data,
        cfg=cfg,
        title="Part 3.5 — Per-node noise σ optimization",
    )
    return bundle_sigma


def _run_sigma_opt_pure(
    network,
    init_dict: dict,
    cfg: Config,
    data: dict,
    sigma_per_node: bool,
    sigma_max: float,
    sigma_l2_weight: float,
    sigma_lr: float,
    max_steps: int,
) -> dict:
    bundle_in = StateBundle.from_dict(init_dict)
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    t1_opt = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    n_nodes = int(data["n_nodes"])

    sigma0_scalar = float(np.mean(np.asarray(network.noise.params.sigma, dtype=np.float32)))
    sigma_init_vec = jnp.full((n_nodes,), sigma0_scalar, dtype=jnp.float32)

    # network σ 는 스칼라로 둔 채 prepare → noise_samples shape 는 σ 값과 무관(_state_indices만).
    compiled_model, opt_state = bundle_in.to_tvb_state(
        network, solver, t1=t1_opt, dt=cfg.integration_dt_ms,
    )

    # c_ei/wLRE/wFFI: plain array(freeze) — optimizer 가 건드리지 않음(축퇴 차단).
    clean = bundle_in.params.sanitize(data["sc_mask"], cfg.connectivity_weight_max)
    opt_state.dynamics.c_ei = jnp.asarray(clean.c_ei)
    opt_state.coupling.coupling.wLRE = jnp.asarray(clean.wLRE)
    opt_state.coupling.coupling.wFFI = jnp.asarray(clean.wFFI)

    bold_monitor = bundle_in.build_bold_monitor(cfg)
    fc_target_safe = jnp.asarray(np.nan_to_num(data["fc_target"], nan=0.0))
    fc_W = jnp.asarray(data["fc_edge_weight"])
    plain_rmse_mask = 1.0 - jnp.eye(fc_target_safe.shape[0], dtype=fc_W.dtype)
    block_corr_terms = prepare_block_corr_terms(
        data["fc_block_masks"],
        {"cc": cfg.corr_block_weight_cc,
         "cross": cfg.corr_block_weight_cross,
         "subsub": cfg.corr_block_weight_subsub},
    )
    # rmse 블록분할(옵션): Part3 와 동일 (corr block 가중 재사용). False면 plain RMSE.
    rmse_block_terms = (
        prepare_block_corr_terms(
            data["fc_block_masks"],
            {"cc": cfg.corr_block_weight_cc,
             "cross": cfg.corr_block_weight_cross,
             "subsub": cfg.corr_block_weight_subsub},
        )
        if cfg.optimizer_rmse_block else None
    )

    def _rmse_term(fc):
        if rmse_block_terms is not None:
            return block_rmse_loss(fc, fc_target_safe, rmse_block_terms)
        return weighted_rmse_loss(fc, fc_target_safe, plain_rmse_mask)

    # pre-FC = Part3 가 stamp 한 post_grad FC(plot 연속성). 없으면 fresh sim.
    pre_fc = bundle_in.metadata.get("post_grad_fc_matrix", None)
    if pre_fc is None:
        pre_fc = compute_simulated_fc(
            network, bundle_in, cfg,
            sim_duration_ms=t1_opt, skip_tr=cfg.optimizer_bold_skip_tr,
        )

    def _sigma_l2(state):
        s = _unwrap_param(state.noise.sigma)
        return jnp.mean((s - jnp.mean(s)) ** 2)

    def compute_loss(state):
        sim = compiled_model(state)
        bold = bold_monitor(sim)
        fc = _compute_fc_differentiable(bold, cfg.optimizer_bold_skip_tr)
        L_global = block_corr_loss(fc, fc_target_safe, block_corr_terms)
        L_rmse = _rmse_term(fc)
        act_l = _compute_activity_regularization(sim, cfg)
        return (
            cfg.optimizer_global_corr_weight * L_global
            + cfg.optimizer_rmse_weight * L_rmse
            + cfg.optimizer_activity_weight * act_l
            + sigma_l2_weight * _sigma_l2(state)
        )

    def compute_full_corr(state):
        sim = compiled_model(state)
        bold = bold_monitor(sim)
        fc = _compute_fc_differentiable(bold, cfg.optimizer_bold_skip_tr)
        return 1.0 - weighted_corr_loss(fc, fc_target_safe, fc_W)

    try:
        init_loss = float(jnp.nan_to_num(compute_loss(opt_state), nan=1e4))
        init_corr = float(jnp.nan_to_num(compute_full_corr(opt_state), nan=-1.0))
    except Exception as error:
        print(f"[WARN] Part3.5 initial eval failed: {error}")
        init_loss, init_corr = 1e4, float("nan")

    print(
        f"[Part3.5] Initial loss={init_loss:.6f}  FullCorr={init_corr:.4f}  "
        f"σ0={sigma0_scalar:.4f}  per_node={sigma_per_node}  σ_max={sigma_max:.4f}  "
        f"L2={sigma_l2_weight}  lr={sigma_lr}  steps={max_steps}"
    )

    # σ 만 free param. per_node=False 면 공유 스칼라 1개(파일럿).
    if sigma_per_node:
        opt_state.noise.sigma = BoundedParameter(sigma_init_vec, low=0.0, high=float(sigma_max))
    else:
        opt_state.noise.sigma = BoundedParameter(
            jnp.asarray(sigma0_scalar, dtype=jnp.float32), low=0.0, high=float(sigma_max)
        )

    best_sigma, loss_history = _run_sigma_optimization_loop(
        compute_loss=compute_loss,
        compute_full_corr=compute_full_corr,
        initial_state=opt_state,
        initial_loss=init_loss,
        sigma_lr=float(sigma_lr),
        max_steps=int(max_steps),
        cfg=cfg,
    )

    best_sigma = np.asarray(best_sigma, dtype=np.float32)
    if best_sigma.ndim == 0:
        best_sigma = np.full((n_nodes,), float(best_sigma), dtype=np.float32)

    # 최적 σ 주입 후 post-FC 평가(downstream prepare 가 이 σ 를 읽음).
    network.noise.params.sigma = jnp.asarray(best_sigma)
    candidate_bundle = bundle_in.advance(new_params=bundle_in.params, next_stage="sigma")
    post_fc, _ = _evaluate_bundle_without_settle(
        network=network, bundle_in=candidate_bundle, cfg=cfg,
        sim_duration_ms=t1_opt, skip_tr=cfg.optimizer_bold_skip_tr,
    )
    candidate_bundle = candidate_bundle.with_metadata({
        "sigma_opt": best_sigma,
        "post_sigma_fc_matrix": np.asarray(post_fc, dtype=np.float32),
        "post_sigma_fc_corr": float(fc_corr(jnp.asarray(post_fc), fc_target_safe)),
        "post_sigma_fc_rmse": _compute_rmse_metric(post_fc, data["fc_target"]),
    })

    return {
        "bundle": candidate_bundle.to_dict(),
        "loss_history": loss_history,
        "sigma_opt": best_sigma,
        "pre_fc": np.asarray(pre_fc, dtype=np.float32),
        "post_fc": np.asarray(post_fc, dtype=np.float32),
    }


def _run_sigma_optimization_loop(
    compute_loss,
    compute_full_corr,
    initial_state,
    initial_loss: float,
    sigma_lr: float,
    max_steps: int,
    cfg: Config,
):
    # adamaxw + zero_nans + global-norm clip — Part3 와 동일 chain. σ clip 은
    # BoundedParameter 가 처리하므로 별도 투영 불필요.
    optimizer = OptaxOptimizer(
        compute_loss,
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(0.1),
            optax.adamaxw(learning_rate=sigma_lr),
        ),
    )

    losses = []
    state = initial_state
    completed = 0
    best_loss = float("inf")
    best_sigma = _to_numpy_param_array(state.noise.sigma)
    opt_state = None   # persist Adam momentum across chunks (see _optax_run_persist)
    start_time = time.time()

    print(f"[SIGMA] Starting σ optimization  (total {max_steps} steps, chunk={cfg.optimizer_chunk_steps})")
    print(
        f"  {'Step':>8} {'Loss':>12} {'FullCorr':>10} {'BestLoss':>12} "
        f"{'σmean':>9} {'σstd':>9} {'Elapsed':>10}"
    )
    print("-" * 76)

    while completed < max_steps:
        chunk = min(cfg.optimizer_chunk_steps, max_steps - completed)
        with remat_scan():
            state, opt_state = _optax_run_persist(optimizer, state, chunk, opt_state)
        completed += chunk

        try:
            step_loss = float(jnp.nan_to_num(compute_loss(state), nan=initial_loss))
            full_corr = float(jnp.nan_to_num(compute_full_corr(state), nan=-1.0))
        except Exception:
            step_loss, full_corr = initial_loss, float("nan")

        losses.append(step_loss)
        s = _unwrap_param(state.noise.sigma)
        if np.isfinite(step_loss) and step_loss < best_loss:
            best_loss = step_loss
            best_sigma = np.asarray(s, dtype=np.float32)

        elapsed = time.time() - start_time
        print(
            f"  {completed:>6}/{max_steps:<6}  {step_loss:>12.6f}  {full_corr:>10.4f}"
            f"  {best_loss:>12.6f}  {float(jnp.mean(s)):>9.4f}  {float(jnp.std(s)):>9.4f}"
            f"  {_fmt_time(elapsed):>10}"
        )

    print("-" * 76)
    print(f"[SIGMA] Complete!  Best loss: {best_loss:.6f}")
    return best_sigma, np.asarray(losses, dtype=np.float32)


def _plot_sigma_results(
    sigma_opt: np.ndarray,
    sigma0: float,
    loss_history: np.ndarray,
    pre_fc: np.ndarray,
    post_fc: np.ndarray,
    data: dict,
    cfg: Config,
    title: str,
) -> None:
    fc_target = np.asarray(data["fc_target"], dtype=np.float32)
    pre_fc = np.asarray(pre_fc, dtype=np.float32)
    post_fc = np.asarray(post_fc, dtype=np.float32)
    sigma_opt = np.asarray(sigma_opt, dtype=np.float32)

    pre_corr = float(fc_corr(jnp.asarray(pre_fc), jnp.asarray(fc_target)))
    post_corr = float(fc_corr(jnp.asarray(post_fc), jnp.asarray(fc_target)))
    pre_rmse = _compute_rmse_metric(pre_fc, fc_target)
    post_rmse = _compute_rmse_metric(post_fc, fc_target)

    print(
        f"[Part3.5 Plot] Pre-σ   corr={pre_corr:.4f}  rmse={pre_rmse:.4f}\n"
        f"[Part3.5 Plot] Post-σ  corr={post_corr:.4f}  rmse={post_rmse:.4f}\n"
        f"[Part3.5 Plot] σ  mean={sigma_opt.mean():.4f}  std={sigma_opt.std():.4f}  "
        f"min={sigma_opt.min():.4f}  max={sigma_opt.max():.4f}  (σ0={sigma0:.4f})"
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(title, fontsize=13)

    axes[0, 0].plot(loss_history, linewidth=1.5, color="black")
    if len(loss_history) > 0:
        axes[0, 0].scatter(0, loss_history[0], s=60, color="steelblue", zorder=5, label="start")
        axes[0, 0].scatter(len(loss_history) - 1, loss_history[-1], s=60,
                           color="tomato", zorder=5, label="end")
    axes[0, 0].set_title("Loss convergence")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(np.arange(sigma_opt.shape[0]), sigma_opt, color="darkorange")
    axes[0, 1].axhline(sigma0, color="black", linestyle="--", linewidth=1, label=f"σ0={sigma0:.3f}")
    axes[0, 1].set_title("Optimized per-node σ")
    axes[0, 1].set_xlabel("Node")
    axes[0, 1].set_ylabel("σ")
    axes[0, 1].legend()

    axes[0, 2].hist(sigma_opt, bins=min(20, max(5, sigma_opt.shape[0] // 2)), color="darkorange")
    axes[0, 2].axvline(sigma0, color="black", linestyle="--", linewidth=1)
    axes[0, 2].set_title("σ distribution")
    axes[0, 2].set_xlabel("σ")
    axes[0, 2].set_ylabel("count")

    fc_titles = [
        "Target FC",
        f"Pre-σ FC\n(corr={pre_corr:.4f}, rmse={pre_rmse:.4f})",
        f"Post-σ FC\n(corr={post_corr:.4f}, rmse={post_rmse:.4f})",
    ]
    for ax, fc_mat, ttl in zip(axes[1], [fc_target, pre_fc, post_fc], fc_titles):
        image = ax.imshow(
            np.nan_to_num(fc_mat),
            vmin=cfg.fc_plot_vmin, vmax=cfg.fc_plot_vmax,
            cmap="RdBu_r",
        )
        ax.set_title(ttl)
        ax.set_xlabel("Region")
        ax.set_ylabel("Region")
        plt.colorbar(image, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# Shared settle / adapters
# ══════════════════════════════════════════════════════════════

def _evaluate_bundle_without_settle(
    network,
    bundle_in: StateBundle,
    cfg: Config,
    sim_duration_ms: int,
    skip_tr: int,
):
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    eval_model, eval_state = bundle_in.to_tvb_state(
        network,
        solver,
        t1=sim_duration_ms,
        dt=cfg.integration_dt_ms,
    )
    eval_monitor = bundle_in.build_bold_monitor(cfg)

    eval_fc, eval_result = eval_fc_multiseed(
        eval_model, eval_state, eval_monitor, _compute_fc_from_bold_output, skip_tr,
        int(cfg.bundle_rng_seed), int(getattr(cfg, "fc_eval_n_seeds", 1)))
    return np.asarray(eval_fc, dtype=np.float32), np.asarray(eval_result.data, dtype=np.float32)


def _settle_bundle(
    network,
    bundle_in: StateBundle,
    cfg: Config,
    sim_duration_ms: int,
    skip_tr: int,
    next_stage: str,
):
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    settle_model, settle_state = bundle_in.to_tvb_state(
        network,
        solver,
        t1=sim_duration_ms,
        dt=cfg.integration_dt_ms,
    )
    settle_monitor = bundle_in.build_bold_monitor(cfg)

    settle_result = settle_model(settle_state)
    settle_bold_output = settle_monitor(settle_result)
    settle_fc = _compute_fc_from_bold_output(settle_bold_output, skip_tr)

    settle_state.initial_state.dynamics = settle_result.data[-1]
    settle_monitor = update_bold_history(settle_monitor, settle_result)
    internal_state, metadata = advance_internal_state(settle_state, bundle_in.metadata)
    delay_history = sync_network_delay_history(network, settle_result)

    settled_bundle = bundle_in.advance(
        new_params=bundle_in.params,
        new_init_dynamics=np.asarray(settle_result.data[-1], dtype=np.float32),
        new_bold_history=np.asarray(settle_monitor.history, dtype=np.float32),
        new_bold_window=extract_bold_window(settle_bold_output),
        new_internal_state=internal_state,
        new_delay_history=delay_history,
        next_stage=next_stage,
        metadata_update=metadata,
    )
    return settled_bundle, np.asarray(settle_fc, dtype=np.float32), np.asarray(settle_result.data, dtype=np.float32)


def _coerce_gradient_bundle(bundle_in, eib_results, cfg: Config, data: dict, stage: str) -> StateBundle:
    if isinstance(bundle_in, StateBundle):
        return bundle_in
    if isinstance(eib_results, StateBundle):
        return eib_results
    if eib_results is None:
        raise ValueError("run_gradient_optimization requires either bundle_in or eib_results.")
    if isinstance(eib_results, dict) and "bundle" in eib_results:
        return StateBundle.from_dict(eib_results["bundle"])
    if isinstance(eib_results, dict) and "state_ei" in eib_results:
        return build_bundle_from_legacy_state(
            state=eib_results["state_ei"],
            cfg=cfg,
            data=data,
            stage=stage,
            bold_history=eib_results.get("best_bundle", {}).get("bold_history", None),
            bold_window=None,
            internal_state=eib_results.get("internal_state_eib", capture_internal_state(eib_results["state_ei"])),
            delay_history=eib_results.get("delay_history_eib", None),
            c_ei_frozen=bool(eib_results.get("c_ei_frozen", False)),
            metadata=eib_results.get("metadata", {"rng_seed": int(cfg.bundle_rng_seed)}),
        )
    raise ValueError("Unsupported eib_results format for run_gradient_optimization().")


# ══════════════════════════════════════════════════════════════
# Full-matrix optimization loop
# ══════════════════════════════════════════════════════════════

def _optax_run_persist(opt, state, n_steps, opt_state):
    """`n_steps` optax updates that KEEP `opt_state` across calls.

    tvboptim `OptaxOptimizer.run()` calls `optimizer.init()` on every invocation
    (optim/optax.py:209) and never returns opt_state, so the chunked `run()` loops
    below reset Adam(max) momentum each chunk — 250 steps ran as 25 momentum
    restarts. Driving the optax chain directly threads opt_state through instead.
    Mirrors run()'s python step loop verbatim (no lax.scan → remat_scan still wraps
    only the inner sim scan). The chain here (zero_nans/clip/adamaxw) uses no
    value_fn, so run()'s extra update() kwargs are omitted. Parameter wrappers are
    preserved because optax's own tree_map descends into the eqx.Module leaves.
    """
    diff_state, static_state = partition_state(state)
    loss_fn = jax.jit(lambda d, s: opt.loss(combine_state(d, s)))
    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    if opt_state is None:
        opt_state = opt.optimizer.init(diff_state)
    for _ in range(int(n_steps)):
        _, grads = grad_fn(diff_state, static_state)
        updates, opt_state = opt.optimizer.update(grads, opt_state, diff_state)
        diff_state = optax.apply_updates(diff_state, updates)
    return combine_state(diff_state, static_state), opt_state


def _run_full_optimization_loop(
    compute_loss,
    compute_loss_and_metrics,
    initial_state,
    initial_loss: float,
    cfg: Config,
    c_ei_frozen: bool,
    sc_mask_j=None,   # #2: 제약 투영용 SC mask (jnp). None이면 투영 생략(하위호환)
    w_max: float = 1.5,
):
    optimizer = OptaxOptimizer(
        compute_loss,
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(0.1),
            optax.adamaxw(learning_rate=cfg.optimizer_learning_rate),
        ),
    )

    all_loss_values = []
    current_state = initial_state
    completed_steps = 0
    best_loss = float("inf")
    best_params_dict = _snapshot_params(current_state, c_ei_frozen)
    opt_state = None   # persist Adam momentum across chunks (see _optax_run_persist)
    start_time = time.time()

    print(
        f"[GRAD] Starting full-matrix optimization  "
        f"(total {cfg.optimizer_max_steps} steps, chunk={cfg.optimizer_chunk_steps})"
    )
    print(
        f"  {'Step':>8} {'Loss':>12} {'FullCorr':>10} "
        f"{'BestLoss':>12} {'Step/s':>8} {'Elapsed':>10} {'ETA':>10}"
    )
    print("-" * 82)
    if cfg.optimizer_chunk_steps < 5:
        print(
            "  [hint] optimizer_chunk_steps={} 는 Python overhead가 큽니다. "
            "수렴 dynamics 변경 감수 시 cfg.optimizer_chunk_steps=5~10 권장."
            .format(cfg.optimizer_chunk_steps)
        )
    if remat_scan_enabled():
        print("  [REMAT] PART3_REMAT_SCAN=1 → scan checkpointing ON "
              "(AD tape 메모리↓, ~1.5-2x 느림, 결과 동일)")

    while completed_steps < cfg.optimizer_max_steps:
        chunk = min(cfg.optimizer_chunk_steps, cfg.optimizer_max_steps - completed_steps)
        t_chunk = time.time()
        # (B) gradient checkpointing: PART3_REMAT_SCAN=1 이면 scan body remat → 메모리↓.
        # 첫 chunk 트레이스가 패치 하에 컴파일되고 이후 chunk 는 재사용. 비활성 시 no-op.
        with remat_scan():
            current_state, opt_state = _optax_run_persist(optimizer, current_state, chunk, opt_state)
        completed_steps += chunk

        # #2: 제약 투영 (ParamSet.sanitize와 동일 순서: clip[0,w_max] → *sc_mask → 대칭화).
        #     평가/스냅샷/반환 가중치를 최종 sanitize 결과와 일치 → best_loss ≠ post_opt_fc 제거.
        #     wLRE/wFFI만 투영(c_ei는 BoundedParameter로 이미 [0,20] 제약).
        if sc_mask_j is not None:
            current_state = _project_weights(current_state, sc_mask_j, w_max)

        try:
            total_l, _, _, full_corr = compute_loss_and_metrics(current_state)
            step_loss = float(jnp.nan_to_num(total_l, nan=initial_loss))
            full_corr = float(jnp.nan_to_num(full_corr, nan=-1.0))
        except Exception:
            step_loss, full_corr = initial_loss, float("nan")

        all_loss_values.append(step_loss)
        if np.isfinite(step_loss) and step_loss < best_loss:
            best_loss = step_loss
            best_params_dict = _snapshot_params(current_state, c_ei_frozen)

        elapsed = time.time() - start_time
        remaining = (elapsed / max(completed_steps, 1)) * (cfg.optimizer_max_steps - completed_steps)
        step_per_sec = chunk / max(time.time() - t_chunk, 1e-12)
        print(
            f"  {completed_steps:>6}/{cfg.optimizer_max_steps:<6}"
            f"  {step_loss:>12.6f}  {full_corr:>10.4f}"
            f"  {best_loss:>12.6f}  {step_per_sec:>8.2f}"
            f"  {_fmt_time(elapsed):>10}  {_fmt_time(remaining):>10}"
        )

    print("-" * 82)
    print(f"[GRAD] Complete!  Best loss: {best_loss:.6f}")
    return best_params_dict, np.asarray(all_loss_values, dtype=np.float32)


def _to_numpy_param_array(value) -> np.ndarray:
    current = value
    for _ in range(4):
        if isinstance(current, (np.ndarray, jnp.ndarray)):
            break
        if hasattr(current, "value") and not callable(getattr(current, "value")):
            current = getattr(current, "value")
            continue
        if hasattr(current, "parameter") and not callable(getattr(current, "parameter")):
            current = getattr(current, "parameter")
            continue
        break
    try:
        return np.asarray(current, dtype=np.float32)
    except Exception:
        return np.asarray(jnp.asarray(current), dtype=np.float32)


def _snapshot_params(state, c_ei_frozen: bool) -> dict:
    c_ei = _to_numpy_param_array(state.dynamics.c_ei)
    wLRE = _to_numpy_param_array(state.coupling.coupling.wLRE)
    wFFI = _to_numpy_param_array(state.coupling.coupling.wFFI)
    return {
        "c_ei": c_ei,
        "wLRE": wLRE,
        "wFFI": wFFI,
        "c_ei_frozen": bool(c_ei_frozen),
    }


def _unwrap_param(value) -> jnp.ndarray:
    """Parameter/BoundedParameter 래퍼를 벗겨 on-device jnp 배열 반환 (host 왕복 없음)."""
    current = value
    for _ in range(4):
        if isinstance(current, (np.ndarray, jnp.ndarray)):
            break
        if hasattr(current, "value") and not callable(getattr(current, "value")):
            current = getattr(current, "value")
            continue
        if hasattr(current, "parameter") and not callable(getattr(current, "parameter")):
            current = getattr(current, "parameter")
            continue
        break
    return jnp.asarray(current)


def _project_weights(state, sc_mask_j, w_max):
    """#2: wLRE/wFFI를 ParamSet.sanitize와 동일하게 제약면에 투영.
    순서 일치 필수 — clip[0,w_max] → *sc_mask → 0.5(W+Wᵀ). idempotent라 최종 sanitize와 동치.
    optimizer가 보는 가중치 = 반환 bundle 가중치 → train/eval 불일치 제거."""
    w_max_j = jnp.asarray(w_max, dtype=jnp.float32)
    for attr in ("wLRE", "wFFI"):
        w = _unwrap_param(getattr(state.coupling.coupling, attr))
        w = jnp.clip(w, 0.0, w_max_j)
        w = w * sc_mask_j
        w = 0.5 * (w + w.T)
        setattr(state.coupling.coupling, attr, Parameter(w))
    return state


# ══════════════════════════════════════════════════════════════
# 공용 loss / FC helpers
# ══════════════════════════════════════════════════════════════

def _compute_fc_differentiable(bold_output, skip_tr: int, eps: float = 1e-6) -> jnp.ndarray:
    ts = bold_output.ys[:, 0, :] if bold_output.ys.ndim == 3 else bold_output.ys
    if ts.shape[0] <= max(int(skip_tr), 1):
        return jnp.zeros((ts.shape[-1], ts.shape[-1]), dtype=ts.dtype)
    ts = ts[int(skip_tr):]
    ts = jnp.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    ts = ts - jnp.mean(ts, axis=0, keepdims=True)
    # eps 를 sqrt 안에 넣어 backward-safe. 원래 maximum(jnp.std(ts), eps) 는 분산 0 노드
    # (constant BOLD)에서 jnp.std 의 sqrt(0) backward 가 NaN → maximum 이 forward 만 막고
    # backward 는 0·NaN=NaN → gradient 전체가 NaN → optax.zero_nans 가 0 으로 덮어
    # Part3 최적화가 완전히 멈춘다(loss frozen). ts 는 위에서 이미 demean 됨.
    var = jnp.mean(ts * ts, axis=0, keepdims=True)
    std = jnp.sqrt(var + eps ** 2)
    ts_norm = ts / std
    fc = (ts_norm.T @ ts_norm) / jnp.maximum(ts_norm.shape[0] - 1, 1)
    fc = jnp.clip(fc, -1.0, 1.0)
    return fc * (1.0 - jnp.eye(fc.shape[0], dtype=fc.dtype))


def _compute_fc_from_bold_output(bold_output, skip_tr: int, eps: float = 1e-6) -> np.ndarray:
    ts = np.asarray(
        bold_output.ys[:, 0, :] if bold_output.ys.ndim == 3 else bold_output.ys,
        dtype=np.float32,
    )
    ts = ts[int(skip_tr):]
    ts = np.nan_to_num(ts)
    ts = ts - ts.mean(axis=0, keepdims=True)
    std = np.maximum(ts.std(axis=0, keepdims=True), eps)
    ts_norm = ts / std
    fc = (ts_norm.T @ ts_norm) / max(ts_norm.shape[0] - 1, 1)
    fc = np.clip(fc, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(fc, 0.0)
    return fc


def _compute_activity_regularization(simulation_result, cfg: Config) -> jnp.ndarray:
    # EI_Tuning loss: target S_e gating (mean over final window)
    mean_e = jnp.mean(simulation_result.data[-500:, 0, :], axis=0)
    return jnp.mean((mean_e - jnp.float32(cfg.fic_target_se)) ** 2)


def _compute_rmse_metric(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = 1.0 - np.eye(predicted.shape[0], dtype=np.float32)
    diff = (predicted - target) * mask
    n = max(float(mask.sum()), 1.0)
    return float(np.sqrt(np.sum(diff ** 2) / n))


def _fmt_time(secs: float) -> str:
    secs = int(secs)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════

def _plot_gradient_results(
    optimized_bundle: StateBundle,
    loss_history: np.ndarray,
    data: dict,
    cfg: Config,
    pre_opt_fc: np.ndarray,
    post_opt_fc: np.ndarray,
    title: str,
) -> None:
    fc_target = np.asarray(data["fc_target"], dtype=np.float32)
    pre_opt_fc = np.asarray(pre_opt_fc, dtype=np.float32)
    post_opt_fc = np.asarray(post_opt_fc, dtype=np.float32)
    wLRE_opt = np.nan_to_num(np.asarray(optimized_bundle.params.wLRE))
    wFFI_opt = np.nan_to_num(np.asarray(optimized_bundle.params.wFFI))

    pre_corr = float(fc_corr(jnp.asarray(pre_opt_fc), jnp.asarray(fc_target)))
    post_corr = float(fc_corr(jnp.asarray(post_opt_fc), jnp.asarray(fc_target)))
    pre_rmse = _compute_rmse_metric(pre_opt_fc, fc_target)
    post_rmse = _compute_rmse_metric(post_opt_fc, fc_target)

    print(
        f"[Part3 Plot] Pre-opt  corr={pre_corr:.4f}  rmse={pre_rmse:.4f}\n"
        f"[Part3 Plot] Post-opt corr={post_corr:.4f}  rmse={post_rmse:.4f}"
    )
    _bc = compute_block_corrs(
        post_opt_fc, fc_target, data["cortex_indices"], data["subcortex_indices"]
    )
    print(
        f"[Part3 Plot] Block corr post  ctx-ctx={_bc['ctx']:.4f}  "
        f"cross={_bc['cross']:.4f}  sub-sub={_bc['sub']:.4f}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(title, fontsize=13)

    axes[0, 0].plot(loss_history, linewidth=1.5, color="black")
    if len(loss_history) > 0:
        axes[0, 0].scatter(0, loss_history[0], s=60, color="steelblue", zorder=5, label="start")
        axes[0, 0].scatter(len(loss_history)-1, loss_history[-1], s=60,
                           color="tomato", zorder=5, label="end")
    axes[0, 0].set_title(
        "Loss convergence\n"
        f"({cfg.optimizer_global_corr_weight:g}·block_corr "
        f"+ {cfg.optimizer_rmse_weight:g}·rmse "
        f"+ {cfg.optimizer_activity_weight:g}·activity)"
    )
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for ax, mat, ttl in zip(
        axes[0, 1:],
        [wFFI_opt, wLRE_opt],
        ["Optimized wFFI", "Optimized wLRE"],
    ):
        image = ax.imshow(mat, vmin=0, vmax=cfg.connectivity_weight_max, cmap="viridis")
        ax.set_title(ttl)
        ax.set_xlabel("Source")
        ax.set_ylabel("Target")
        plt.colorbar(image, ax=ax, fraction=0.046)

    fc_titles = [
        "Target FC",
        f"Pre-opt FC\n(corr={pre_corr:.4f}, rmse={pre_rmse:.4f})",
        f"Post-opt FC\n(corr={post_corr:.4f}, rmse={post_rmse:.4f})",
    ]

    for ax, fc_mat, ttl in zip(
        axes[1],
        [fc_target, pre_opt_fc, post_opt_fc],
        fc_titles,
    ):
        image = ax.imshow(
            np.nan_to_num(fc_mat),
            vmin=cfg.fc_plot_vmin, vmax=cfg.fc_plot_vmax,
            cmap="RdBu_r",
        )
        ax.set_title(ttl)
        ax.set_xlabel("Region")
        ax.set_ylabel("Region")
        plt.colorbar(image, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()

    # 블록별 FC corr (full / ctx-ctx / cross / sub-sub) — subcortex fitting 진단
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_block_corr_bars(
        ax, pre_opt_fc, post_opt_fc, fc_target,
        data["cortex_indices"], data["subcortex_indices"],
        title=f"{title} — Block-wise FC corr",
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# Beta oscillation check
# ══════════════════════════════════════════════════════════════

def _analyze_beta_oscillation_gradient(neural_data: np.ndarray, cfg: Config, title: str) -> None:
    lfp_signal = neural_data[:, 0, :] + neural_data[:, 1, :]
    mean_lfp = np.mean(lfp_signal, axis=1)

    fs_hz = 1000.0 / cfg.integration_dt_ms
    nperseg = int(fs_hz * 2)
    noverlap = nperseg // 2

    frequencies, normalized_psd = _compute_normalized_psd(
        mean_lfp, fs_hz, nperseg, noverlap, cfg.dbs_psd_max_frequency_hz
    )
    beta_ratio = _compute_beta_power_ratio(
        frequencies, normalized_psd,
        cfg.dbs_beta_band_low_hz, cfg.dbs_beta_band_high_hz,
    )
    _plot_beta_summary(
        {
            "lfp_signal": mean_lfp,
            "frequencies": frequencies,
            "psd": normalized_psd,
            "beta_ratio": beta_ratio,
        },
        cfg,
        title=title,
    )
