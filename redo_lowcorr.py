#!/usr/bin/env python3
"""
redo_lowcorr.py — corr 낮은 idx를 part1부터 full 재튜닝 (FIC→EIB→Gradient)

main_ppmi_pre.py(EIB 캐시 로드)와 달리, 지정 idx에 대해 **part1/2/3 전체를 새로** 돌린다.
같은 config면 FIC/EIB가 결정적이라 같은 결과 → 반드시 뭔가 바꿔야 개선됨.
이 스크립트는 noise 와 cache_version tag 를 바꿔 fresh 재튜닝을 강제한다.

기본: noise 0.02→0.005 (ergodic 안정화), cache_version 에 _n{noise} 접미사.

실행:
  python3 redo_lowcorr.py                          # idx 0,4,7,8, noise 0.005
  python3 redo_lowcorr.py --idxs 0                 # idx 0 만 (먼저 1개 테스트 권장)
  python3 redo_lowcorr.py --idxs 0,4,7,8 --noise 0.005
  python3 redo_lowcorr.py --idxs 0 --noise 0.01    # noise 더 약하게

주의:
  - FIC(2000) + EIB(10000) + part3(100) full → idx당 수 시간. 먼저 --idxs 0 으로 1개 검증.
  - part3 720window → gradient checkpointing 자동 ON (PART3_REMAT_SCAN).
  - noise 바꾸면 cache_version 도 바뀌어(_n{noise}) 기존 0.02 캐시와 충돌 안 함.
"""
import argparse
import os

# part3 720window full backprop OOM 방지 → checkpointing 기본 ON
os.environ.setdefault("PART3_REMAT_SCAN", "1")

import main_ppmi as M
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


def redo_idx(idx: int, noise: float) -> dict:
    print("\n" + "=" * 70)
    print(f"  [idx {idx}] FULL 재튜닝 (part1→2→3)  noise={noise}")
    print("=" * 70)

    # noise 반영 + cache_version 에 noise 접미사 → fresh (cache_tag 에 noise 없으므로 필수)
    p = M.prepare_ppmi_data(idx, noise)
    cfg = M.make_config(p, idx)
    cfg.cache_version = f"{cfg.cache_version}_n{str(noise).replace('.', 'p')}"
    print(f"[cfg] cache_version={cfg.cache_version}  additive_noise_sigma={cfg.additive_noise_sigma}")
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

    # Part 1: FIC (fresh)
    bundle_fic = run_fic(network=network, bundle_in=bundle_init, cfg=cfg, data=data)
    print(f"[FIC] mean c_ei={bundle_fic.params.c_ei.mean():.4f}")

    # Part 2: EIB (fresh)
    bundle_eib = run_eib(network=network, bundle_in=bundle_fic, cfg=cfg, data=data)
    eib_corr = bundle_eib.metadata.get("post_eib_fc_corr")
    print(f"[EIB] post_eib_fc_corr={eib_corr}")

    # Part 3: Gradient (fresh, remat)
    bundle_grad = run_gradient_optimization(
        network       = network,
        bundle_in     = bundle_eib,
        warmup_bundle = bundle_init,
        cfg           = cfg,
        data          = data,
    )
    meta = bundle_grad.metadata
    print(f"[idx {idx}] DONE  EIB_corr={eib_corr}  "
          f"post_grad_fc_corr={meta.get('post_grad_fc_corr')}  "
          f"post_grad_fc_rmse={meta.get('post_grad_fc_rmse')}")
    return {
        "idx": idx,
        "eib_corr": eib_corr,
        "post_grad_corr": meta.get("post_grad_fc_corr"),
        "post_grad_rmse": meta.get("post_grad_fc_rmse"),
    }


def main():
    ap = argparse.ArgumentParser(description="corr 낮은 idx part1부터 full 재튜닝")
    ap.add_argument("--idxs", type=str, default="0,4,7,8",
                    help="재튜닝할 idx 콤마구분 (기본 0,4,7,8)")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="additive_noise_sigma (기본 0.005, 원래 0.02). 작을수록 FC 안정/재현↑")
    args = ap.parse_args()
    idxs = [int(x) for x in args.idxs.split(",") if x.strip() != ""]

    print("=" * 70)
    print(f"  redo_lowcorr — full 재튜닝  idxs={idxs}  noise={args.noise}")
    print("=" * 70)

    results = []
    for idx in idxs:
        try:
            results.append(redo_idx(idx, args.noise))
        except Exception as exc:
            import traceback
            print(f"\n[idx {idx}] FAILED: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  재튜닝 결과 요약")
    print("=" * 70)
    print(f"  {'idx':>3} {'EIB_corr':>10} {'post_grad':>10} {'rmse':>8}")
    for r in results:
        ec = r["eib_corr"]; pg = r["post_grad_corr"]; rm = r["post_grad_rmse"]
        print(f"  {r['idx']:>3} {ec if ec is None else f'{ec:.4f}':>10} "
              f"{pg if pg is None else f'{pg:.4f}':>10} {rm if rm is None else f'{rm:.4f}':>8}")


if __name__ == "__main__":
    main()
