#!/usr/bin/env python3
"""
run_sigma_idx23.py — idx 2, 3 대상 part1→2→3→3.5 풀 파이프라인 (PPMI TR=2.4s/210TR)

왜 grad 캐시 직접 로드가 아니라 풀 재튜닝인가
----------------------------------------------
make_config 의 window 파라미터를 PPMI 실측(TR=2400ms, 210 TR)으로 바꿨다.
기존 grad/eib/fic 캐시는 옛 TR(1000ms/720TR)에서 생성됐으므로 **stale** 이다.
옛 weight(1000ms 기준 최적) 위에 새 TR 로 σ 만 튜닝하면 정합성이 깨진다.
→ idx 2,3 을 새 TR 로 part1→2→3 재튜닝한 뒤 Part 3.5(σ) 를 올린다.

cache_name 에 TR/window 가 포함되므로(_TR210, _win210, _dur2400) 옛 _TR720 캐시와
충돌하지 않고 새로 생성된다(옛 pkl 은 디스크에 남지만 hit 안 됨).

base 가중치(rmse 0.2 / corr 0.8) 사용 — main_ppmi_pre 의 FULL_RETUNE rmse0p4
오버라이드는 적용하지 않는다.

실행: python run_sigma_idx23.py        (GPU 노드)
"""
import main_ppmi_pre as M

from data_loader        import load_data
from model              import build_network
from part1_fic          import run_fic
from part2_eib          import run_eib
from part3_gradient     import run_gradient_optimization
from pipeline_contracts import (
    ParamSet,
    StateBundle,
    capture_internal_state,
    capture_network_delay_history,
)

TARGET_IDX  = [2, 3]
NOISE_LEVEL = M.NOISE_LEVEL
PILOT       = False   # True → SIGMA_PER_NODE=False (공유 스칼라 σ, grad 생존 확인)


def run_full_pipeline_for_idx(idx: int) -> None:
    print("\n" + "=" * 70)
    print(f"  [idx {idx}] part1→2→3→3.5 풀 파이프라인 (TR=2.4s/210TR)")
    print("=" * 70)

    p = M.prepare_ppmi_data(idx, NOISE_LEVEL)
    cfg = M.make_config(p, idx)   # 새 TR/window 반영
    cfg.print_summary()

    data = load_data(cfg)

    # warmup → bundle_init
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    initial_params = ParamSet.default(
        data["n_nodes"], c_ei_init=cfg.wc_c_ei_init
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)
    bundle_init = StateBundle.from_warmup(
        warmup_result          = warmup_result,
        bold_monitor_template  = bold_monitor,
        initial_params         = initial_params,
        internal_state         = capture_internal_state(initial_state),
        delay_history          = capture_network_delay_history(network),
        stage                  = "warmup",
    )

    # Part 1: FIC
    bundle_fic = run_fic(network=network, bundle_in=bundle_init, cfg=cfg, data=data)
    print(f"  [FIC] mean c_ei={bundle_fic.params.c_ei.mean():.4f}")

    # Part 2: EIB
    bundle_eib = run_eib(network=network, bundle_in=bundle_fic, cfg=cfg, data=data)
    print(f"  [EIB] post_eib_fc_corr={bundle_eib.metadata.get('post_eib_fc_corr')}")

    # Part 3: Gradient
    bundle_grad = run_gradient_optimization(
        network       = network,
        bundle_in     = bundle_eib,
        warmup_bundle = bundle_init,
        cfg           = cfg,
        data          = data,
    )
    meta = bundle_grad.metadata
    print(f"  [Part3] post_grad_fc_corr={meta.get('post_grad_fc_corr')}  "
          f"post_grad_fc_rmse={meta.get('post_grad_fc_rmse')}")

    # Part 3.5: per-node σ
    bundle_sigma = M.run_sigma_for_idx(idx, network, bundle_grad, cfg, data)
    print(f"  [idx {idx}] DONE  stage={bundle_sigma.stage}")
    print(bundle_sigma)


def main():
    if PILOT:
        M.SIGMA_PER_NODE = False

    print("=" * 70)
    print(f"  idx {TARGET_IDX}  full retune + Part3.5  per_node={M.SIGMA_PER_NODE}  "
          f"σ_max={M.SIGMA_MAX}  L2={M.SIGMA_L2_WEIGHT}")
    print("=" * 70)

    done, failed = [], []
    for idx in TARGET_IDX:
        try:
            run_full_pipeline_for_idx(idx)
            done.append(idx)
        except Exception as exc:
            import traceback
            print(f"\n[idx {idx}] FAILED: {exc}")
            traceback.print_exc()
            failed.append(idx)

    print("\n" + "=" * 70)
    print(f"  done: {done}   failed: {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
