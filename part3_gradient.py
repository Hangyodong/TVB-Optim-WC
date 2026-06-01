"""
part3_gradient.py

Stage 3
-------
- Part 2 → Part 3 입력을 StateBundle로 통일한다.
- full-matrix optimizer는 구버전 notebook 로직처럼 best params를 바로 반환한다.
- Part 3B low-rank branch도 동일한 StateBundle 계약을 사용한다.
- legacy notebook의 old-style 호출도 계속 허용한다.
"""
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.types import BoundedParameter, Parameter
from tvboptim.utils import cache

from config import Config
from model import WilsonCowanEIB
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
    extract_bold_window,
    sync_network_delay_history,
    update_bold_history,
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

    sim_result = sim_model(sim_state)
    bold_output = bold_monitor(sim_result)
    fc_matrix = _compute_fc_from_bold_output(bold_output, skip_tr)
    return fc_matrix, np.asarray(sim_result.data, dtype=np.float32)


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

    # Patch 26: optimize from warmup state (step 0) + EIB-tuned params,
    # consistent with EIB post-hoc validation.
    if warmup_bundle is not None:
        bundle_start = warmup_bundle.advance(
            new_params=bundle_eib.params,
            new_init_dynamics=warmup_bundle.init_dynamics,
            new_bold_history=warmup_bundle.bold_history,
            new_bold_window=warmup_bundle.bold_window,
            new_internal_state=warmup_bundle.internal_state,
            new_delay_history=warmup_bundle.delay_history,
            next_stage="grad_warmup_start",
            metadata_update={},
        )
    else:
        bundle_start = bundle_eib

    cache_name = (
        f"grad_{data['cache_tag']}"
        f"_TR{cfg.optimizer_bold_window_tr}"
        f"_SKIP{cfg.optimizer_bold_skip_tr}"
        f"_STEPS{cfg.optimizer_max_steps}"
        f"_LR{str(cfg.optimizer_learning_rate).replace('.', 'p')}"
        f"_frozen{int(bundle_start.params.c_ei_frozen)}"
        f"_fp{bundle_start.fingerprint()}"
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

    c_ei_frozen = False
    bold_monitor_opt = bundle_in.build_bold_monitor(cfg)
    fc_target_safe = jnp.asarray(np.nan_to_num(data["fc_target"], nan=0.0))

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
        L_global   = _compute_correlation_loss(fc, fc_target_safe)
        L_nodewise = _compute_nodewise_corr_loss(fc, fc_target_safe)
        L_rmse     = _compute_rmse_loss(fc, fc_target_safe)
        act_l      = _compute_activity_regularization(sim, cfg)
        return (
            cfg.optimizer_global_corr_weight     * L_global
            + cfg.optimizer_nodewise_corr_weight * L_nodewise
            + cfg.optimizer_rmse_weight          * L_rmse
            + 0.01 * act_l
        )

    def compute_loss_and_metrics(state):
        sim = compiled_model(state)
        bold = bold_monitor_opt(sim)
        fc = _compute_fc_differentiable(bold, cfg.optimizer_bold_skip_tr)
        L_global   = _compute_correlation_loss(fc, fc_target_safe)
        L_nodewise = _compute_nodewise_corr_loss(fc, fc_target_safe)
        L_rmse     = _compute_rmse_loss(fc, fc_target_safe)
        act_l      = _compute_activity_regularization(sim, cfg)
        total = (
            cfg.optimizer_global_corr_weight     * L_global
            + cfg.optimizer_nodewise_corr_weight * L_nodewise
            + cfg.optimizer_rmse_weight          * L_rmse
            + 0.01 * act_l
        )
        return total, L_global, act_l, float(1.0 - L_global)

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

    initial_opt_state.dynamics.c_ei = BoundedParameter(
        initial_opt_state.dynamics.c_ei, low=0.0, high=20.0
    )
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

    # Stage handoff: stamp post-grad FC so low-rank reads it as pre-opt FC.
    candidate_bundle = candidate_bundle.with_metadata({
        "post_grad_fc_matrix": np.asarray(post_opt_fc, dtype=np.float32),
        "post_grad_fc_corr": float(fc_corr(jnp.asarray(post_opt_fc), fc_target_safe)),
        "post_grad_fc_rmse": _compute_rmse_metric(post_opt_fc, data["fc_target"]),
    })

    return {
        "bundle": candidate_bundle.to_dict(),
        "loss_history": np.asarray(loss_history, dtype=np.float32),
        "pre_opt_fc": np.asarray(pre_opt_fc, dtype=np.float32),
        "post_opt_fc": np.asarray(post_opt_fc, dtype=np.float32),
        "post_opt_neural": np.asarray(post_opt_neural, dtype=np.float32),
    }


# ══════════════════════════════════════════════════════════════
# Part 3B — Low-rank gradient optimization
# ══════════════════════════════════════════════════════════════

class LowRankTrainable(eqx.Module):
    c_ei:  jnp.ndarray
    lre_u: jnp.ndarray
    lre_v: jnp.ndarray
    ffi_u: jnp.ndarray
    ffi_v: jnp.ndarray


def run_lowrank_optimization(
    network,
    bundle_in: StateBundle = None,
    optimized_state=None,
    eib_results=None,
    cfg: Config = None,
    data: dict = None,
    warmup_result=None,
    warmup_bundle: StateBundle = None,  # Patch 26: step-0 warmup start
) -> StateBundle:
    """
    Low-rank correction 최적화를 실행하고 Full Gradient 결과 위에서 바로 StateBundle을 반환한다.
    """
    bundle_grad = _coerce_lowrank_bundle(bundle_in, optimized_state, eib_results, cfg, data)

    # Patch 26: optimize from warmup state (step 0) + gradient-tuned params.
    if warmup_bundle is not None:
        bundle_start = warmup_bundle.advance(
            new_params=bundle_grad.params,
            new_init_dynamics=warmup_bundle.init_dynamics,
            new_bold_history=warmup_bundle.bold_history,
            new_bold_window=warmup_bundle.bold_window,
            new_internal_state=warmup_bundle.internal_state,
            new_delay_history=warmup_bundle.delay_history,
            next_stage="lowrank_warmup_start",
            metadata_update={},
        )
    else:
        bundle_start = bundle_grad

    cache_name = (
        f"grad_lowrank_{data['cache_tag']}"
        f"_rank{cfg.lowrank_rank}"
        f"_TR{cfg.lowrank_bold_window_tr}"
        f"_STEPS{cfg.lowrank_max_steps}"
        f"_LR{str(cfg.lowrank_learning_rate).replace('.', 'p')}"
        f"_ds{str(cfg.lowrank_delta_scale).replace('.', 'p')}"
        f"_fp{bundle_start.fingerprint()}"
    )

    @cache(cache_name, redo=False)
    def _cached_run():
        return _run_lowrank_pure(network, bundle_start.to_dict(), cfg, data)

    result = _cached_run()
    bundle_lowrank = StateBundle.from_dict(result["bundle"])
    bundle_lowrank.apply_to_network(network)

    _plot_gradient_results(
        bundle_lowrank,
        np.asarray(result["loss_history"]),
        data,
        cfg,
        pre_opt_fc=np.asarray(result["pre_opt_fc"]),
        post_opt_fc=np.asarray(result["post_opt_fc"]),
        title="Part 3B — Low-rank Gradient Optimization",
    )
    return bundle_lowrank


def _run_lowrank_pure(
    network,
    init_dict: dict,
    cfg: Config,
    data: dict,
) -> dict:
    bundle_in = StateBundle.from_dict(init_dict)
    rank = cfg.lowrank_rank
    w_max = cfg.connectivity_weight_max
    delta_scale = cfg.lowrank_delta_scale
    n_nodes = data["n_nodes"]
    sc_mask_np = np.asarray(data["sc_mask"], dtype=np.float32)
    sc_mask_jnp = jnp.asarray(sc_mask_np)
    fc_target_safe = jnp.asarray(np.nan_to_num(data["fc_target"], nan=0.0))
    c_ei_frozen = False

    t1_lr = int(cfg.lowrank_bold_window_tr * cfg.bold_repetition_time_ms)
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    compiled_model_lr, state_lr_template = bundle_in.to_tvb_state(
        network,
        solver,
        t1=t1_lr,
        dt=cfg.integration_dt_ms,
    )
    bold_monitor_lr = bundle_in.build_bold_monitor(cfg)

    c_ei_base = np.asarray(bundle_in.params.c_ei, dtype=np.float32)
    wLRE_base = np.asarray(bundle_in.params.wLRE, dtype=np.float32)
    wFFI_base = np.asarray(bundle_in.params.wFFI, dtype=np.float32)

    key = np.random.RandomState(cfg.lowrank_seed)
    fi = cfg.lowrank_factor_init
    init_trainable = LowRankTrainable(
        c_ei=jnp.asarray(c_ei_base),
        lre_u=jnp.asarray(key.normal(size=(n_nodes, rank)).astype(np.float32) * fi),
        lre_v=jnp.asarray(key.normal(size=(n_nodes, rank)).astype(np.float32) * fi),
        ffi_u=jnp.asarray(key.normal(size=(n_nodes, rank)).astype(np.float32) * fi),
        ffi_v=jnp.asarray(key.normal(size=(n_nodes, rank)).astype(np.float32) * fi),
    )

    # Stage handoff: pre-opt FC = predecessor (gradient) stored post-FC for
    # exact plot continuity; fall back to a fresh sim if the bundle lacks it.
    pre_opt_fc = bundle_in.metadata.get("post_grad_fc_matrix", None)
    if pre_opt_fc is None:
        pre_opt_fc = compute_simulated_fc(
            network,
            bundle_in,
            cfg,
            sim_duration_ms=t1_lr,
            skip_tr=cfg.lowrank_bold_skip_tr,
        )

    def _reconstruct_weights(trainable: LowRankTrainable):
        delta_lre = delta_scale * (trainable.lre_u @ trainable.lre_v.T)
        delta_ffi = delta_scale * (trainable.ffi_u @ trainable.ffi_v.T)
        wLRE_eff = jnp.clip(jnp.asarray(wLRE_base) + delta_lre, 0.0, w_max) * sc_mask_jnp
        wFFI_eff = jnp.clip(jnp.asarray(wFFI_base) + delta_ffi, 0.0, w_max) * sc_mask_jnp
        wLRE_eff = 0.5 * (wLRE_eff + wLRE_eff.T)
        wFFI_eff = 0.5 * (wFFI_eff + wFFI_eff.T)
        return wLRE_eff, wFFI_eff

    def lowrank_loss_fn(trainable: LowRankTrainable) -> jnp.ndarray:
        wLRE_eff, wFFI_eff = _reconstruct_weights(trainable)
        c_ei_eff = jnp.asarray(c_ei_base) if c_ei_frozen else jnp.clip(trainable.c_ei, 0.0, 20.0)

        state = eqx.tree_at(lambda s: s.dynamics.c_ei,          state_lr_template, c_ei_eff)
        state = eqx.tree_at(lambda s: s.coupling.coupling.wLRE, state,             wLRE_eff)
        state = eqx.tree_at(lambda s: s.coupling.coupling.wFFI, state,             wFFI_eff)

        sim = compiled_model_lr(state)
        bold = bold_monitor_lr(sim)
        fc = _compute_fc_differentiable(bold, cfg.lowrank_bold_skip_tr)

        L_global   = _compute_correlation_loss(fc, fc_target_safe)
        L_nodewise = _compute_nodewise_corr_loss(fc, fc_target_safe)
        L_rmse     = _compute_rmse_loss(fc, fc_target_safe)
        act_l = _compute_activity_regularization(sim, cfg)
        factor_l = (
            jnp.mean(trainable.lre_u ** 2) + jnp.mean(trainable.lre_v ** 2)
            + jnp.mean(trainable.ffi_u ** 2) + jnp.mean(trainable.ffi_v ** 2)
        )
        return (
            cfg.lowrank_global_corr_weight     * L_global
            + cfg.lowrank_nodewise_corr_weight * L_nodewise
            + cfg.lowrank_rmse_weight          * L_rmse
            + cfg.lowrank_activity_weight * act_l
            + cfg.lowrank_factor_penalty * factor_l
        )

    lr_optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(0.1),
        optax.adamaxw(learning_rate=cfg.lowrank_learning_rate),
    )

    @eqx.filter_jit
    def lr_step(trainable, opt_state):
        loss_value, grads = eqx.filter_value_and_grad(lowrank_loss_fn)(trainable)
        params = eqx.filter(trainable, eqx.is_array)
        updates, new_opt_state = lr_optimizer.update(grads, opt_state, params=params)
        new_trainable = eqx.apply_updates(trainable, updates)
        return new_trainable, new_opt_state, loss_value

    try:
        init_loss_lr = float(jnp.nan_to_num(lowrank_loss_fn(init_trainable), nan=1e4))
    except Exception as error:
        print(f"[WARN] lowrank initial loss failed: {error}")
        init_loss_lr = 1e4

    print("\n[LOWRANK] Starting low-rank optimization...")
    print(
        f"  rank={rank}  delta_scale={delta_scale}  "
        f"t1={t1_lr}ms ({t1_lr/1000:.0f}s)  "
        f"max_steps={cfg.lowrank_max_steps}  lr={cfg.lowrank_learning_rate}"
    )
    if c_ei_frozen:
        print("  c_ei FROZEN: low-rank branch updates only wLRE / wFFI")

    opt_state = lr_optimizer.init(eqx.filter(init_trainable, eqx.is_array))
    trainable = init_trainable
    best_loss = float("inf")
    best_trainable = trainable
    loss_history = []
    start_time = time.time()

    print(
        f"  {'Step':>8} {'Loss':>12} {'BestLoss':>12} "
        f"{'Step/s':>8} {'Elapsed':>10} {'ETA':>10}"
    )
    print("-" * 72)

    for step in range(cfg.lowrank_max_steps):
        t_step = time.time()
        trainable, opt_state, loss_value = lr_step(trainable, opt_state)
        step_loss = float(jnp.nan_to_num(loss_value, nan=init_loss_lr))
        loss_history.append(step_loss)

        if np.isfinite(step_loss) and step_loss < best_loss:
            best_loss = step_loss
            best_trainable = trainable

        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            step_per_sec = 1.0 / max(time.time() - t_step, 1e-9)
            remaining = (elapsed / (step + 1)) * (cfg.lowrank_max_steps - step - 1)
            print(
                f"  {step+1:>6}/{cfg.lowrank_max_steps:<6}"
                f"  {step_loss:>12.6f}"
                f"  {best_loss:>12.6f}"
                f"  {step_per_sec:>8.2f}"
                f"  {_fmt_time(elapsed):>10}"
                f"  {_fmt_time(remaining):>10}"
            )

    print("-" * 72)
    print(f"[LOWRANK] Done!  Best loss: {best_loss:.6f}")

    best_wLRE, best_wFFI = _reconstruct_weights(best_trainable)
    best_c_ei = np.asarray(jnp.clip(best_trainable.c_ei, 0.0, 20.0), dtype=np.float32)

    best_params = ParamSet(
        c_ei=best_c_ei,
        wLRE=np.asarray(best_wLRE, dtype=np.float32),
        wFFI=np.asarray(best_wFFI, dtype=np.float32),
        c_ei_frozen=c_ei_frozen,
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)

    candidate_bundle = bundle_in.advance(
        new_params=best_params,
        next_stage="lowrank",
    )
    post_opt_fc, post_opt_neural = _evaluate_bundle_without_settle(
        network=network,
        bundle_in=candidate_bundle,
        cfg=cfg,
        sim_duration_ms=t1_lr,
        skip_tr=cfg.lowrank_bold_skip_tr,
    )

    return {
        "bundle": candidate_bundle.to_dict(),
        "loss_history": np.asarray(loss_history, dtype=np.float32),
        "pre_opt_fc": np.asarray(pre_opt_fc, dtype=np.float32),
        "post_opt_fc": np.asarray(post_opt_fc, dtype=np.float32),
        "post_opt_neural": np.asarray(post_opt_neural, dtype=np.float32),
    }


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

    eval_result = eval_model(eval_state)
    eval_bold_output = eval_monitor(eval_result)
    eval_fc = _compute_fc_from_bold_output(eval_bold_output, skip_tr)
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


def _coerce_lowrank_bundle(bundle_in, optimized_state, eib_results, cfg: Config, data: dict) -> StateBundle:
    if isinstance(bundle_in, StateBundle):
        return bundle_in
    if isinstance(optimized_state, StateBundle):
        return optimized_state
    if optimized_state is None:
        raise ValueError("run_lowrank_optimization requires either bundle_in or optimized_state.")
    return build_bundle_from_legacy_state(
        state=optimized_state,
        cfg=cfg,
        data=data,
        stage="grad",
        bold_history=None,
        bold_window=None,
        internal_state=capture_internal_state(optimized_state),
        delay_history=None,
        c_ei_frozen=False,
        metadata={"rng_seed": int(cfg.bundle_rng_seed)},
    )


# ══════════════════════════════════════════════════════════════
# Full-matrix optimization loop
# ══════════════════════════════════════════════════════════════

def _run_full_optimization_loop(
    compute_loss,
    compute_loss_and_metrics,
    initial_state,
    initial_loss: float,
    cfg: Config,
    c_ei_frozen: bool,
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

    while completed_steps < cfg.optimizer_max_steps:
        chunk = min(cfg.optimizer_chunk_steps, cfg.optimizer_max_steps - completed_steps)
        t_chunk = time.time()
        current_state, _ = optimizer.run(current_state, max_steps=chunk)
        completed_steps += chunk

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
    std = jnp.maximum(jnp.std(ts, axis=0, keepdims=True), eps)
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


def _compute_correlation_loss(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    mask = 1.0 - jnp.eye(predicted.shape[0], dtype=predicted.dtype)
    p = predicted * mask
    t = target * mask
    n = jnp.maximum(jnp.sum(mask), 1.0)
    p_c = p - jnp.sum(p) / n
    t_c = t - jnp.sum(t) / n
    num = jnp.sum(p_c * t_c)
    den = jnp.sqrt(
        jnp.maximum(jnp.sum(p_c ** 2), 1e-10)
        * jnp.maximum(jnp.sum(t_c ** 2), 1e-10)
    )
    return 1.0 - num / den


def _compute_nodewise_corr_loss(
    predicted: jnp.ndarray, target: jnp.ndarray
) -> jnp.ndarray:
    n = predicted.shape[0]
    mask = 1.0 - jnp.eye(n, dtype=predicted.dtype)

    def _row_corr(i):
        x = predicted[i] * mask[i]
        y = target[i]    * mask[i]
        n_eff = jnp.maximum(mask[i].sum(), 1.0)
        xm = x - jnp.sum(x) / n_eff
        ym = y - jnp.sum(y) / n_eff
        num = jnp.sum(xm * ym)
        den = jnp.sqrt(
            jnp.maximum(jnp.sum(xm ** 2), 1e-10)
            * jnp.maximum(jnp.sum(ym ** 2), 1e-10)
        )
        return num / den

    row_corrs = jax.vmap(_row_corr)(jnp.arange(n))
    return 1.0 - jnp.mean(row_corrs)


def _compute_rmse_loss(
    predicted: jnp.ndarray, target: jnp.ndarray
) -> jnp.ndarray:
    mask = 1.0 - jnp.eye(predicted.shape[0], dtype=predicted.dtype)
    diff = (predicted - target) * mask
    n = jnp.maximum(mask.sum(), 1.0)
    return jnp.sqrt(jnp.sum(diff ** 2) / n)


def _compute_activity_regularization(simulation_result, cfg: Config) -> jnp.ndarray:
    rE_max = jnp.float32(WilsonCowanEIB.DEFAULT_PARAMS.rE_max_hz)
    mean_e = jnp.mean(simulation_result.data[-500:, 0, :], axis=0)
    return jnp.mean((rE_max * mean_e - jnp.float32(cfg.fic_target_firing_rate_hz)) ** 2)


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

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(title, fontsize=13)

    axes[0, 0].plot(loss_history, linewidth=1.5, color="black")
    if len(loss_history) > 0:
        axes[0, 0].scatter(0, loss_history[0], s=60, color="steelblue", zorder=5, label="start")
        axes[0, 0].scatter(len(loss_history)-1, loss_history[-1], s=60,
                           color="tomato", zorder=5, label="end")
    axes[0, 0].set_title("Loss convergence\n(1 − corr + 0.01×activity)")
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
