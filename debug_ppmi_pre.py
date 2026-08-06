#!/usr/bin/env python3
"""
debug_ppmi_pre.py — main_ppmi_pre.py 단계별 디버그 러너

main_ppmi_pre 의 Part3 흐름을 단계별(gate→load→build→eib→grad)로 끊어 실행하며
각 단계 wall-time 을 출력한다. EIB 는 캐시 .pkl 을 직접 로드한다(재튜닝 없음).

사용 예
-------
    python3 debug_ppmi_pre.py --idx 1 --stage gate    # eib 캐시 스캔만, 즉시
    python3 debug_ppmi_pre.py --idx 1 --stage load    # prepare+config+load_data (~1s)
    python3 debug_ppmi_pre.py --idx 1 --stage eib     # build(warmup) + eib pkl 로드
    python3 debug_ppmi_pre.py --idx 1 --stage grad    # Part3 풀 실행
    python3 debug_ppmi_pre.py --idx 1 --stage grad --quick        # part3 2스텝만
    python3 debug_ppmi_pre.py --idx 1 --stage grad --max-steps 3  # part3 3스텝만

옵션
----
    --idx N        대상 subject idx (기본 1)
    --stage S      gate|load|build|eib|grad (기본 grad; 누적 실행)
    --max-steps N  part3 optimizer_max_steps 축소 (eib pkl 로드는 그대로). 예: 2
    --quick        --max-steps 2 와 동일
"""
import argparse
import os
import time
import traceback


STAGES = ["gate", "load", "build", "eib", "grad"]


def _hr(title: str) -> None:
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70)


class _Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.label}] start ...")
        return self
    def __exit__(self, *exc):
        self.dt = time.time() - self.t0
        print(f"[{self.label}] done in {self.dt:.2f}s")
        return False


def main():
    ap = argparse.ArgumentParser(description="단계별 디버그 러너 for main_ppmi_pre")
    ap.add_argument("--idx", type=int, default=1)
    ap.add_argument("--stage", choices=STAGES, default="grad")
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=None,
                    help="part3 optimizer_max_steps 만 축소. 예: --max-steps 2")
    ap.add_argument("--quick", action="store_true", help="--max-steps 2 와 동일")
    ap.add_argument("--window", type=int, default=None,
                    help="part3 optimizer_bold_window_tr 오버라이드 (메모리 천장 probe용). 예: --window 720")
    ap.add_argument("--skip", type=int, default=None,
                    help="part3 optimizer_bold_skip_tr 오버라이드. 미지정 시 window//5")
    ap.add_argument("--remat", action="store_true",
                    help="(B) gradient checkpointing 켜기 (PART3_REMAT_SCAN=1). 메모리↓ ~2x 느림. 720 재도전용")
    args = ap.parse_args()

    if args.remat:
        os.environ["PART3_REMAT_SCAN"] = "1"   # remat_scan_enabled() 가 런타임에 읽음

    target = STAGES.index(args.stage)
    timings = {}

    _hr("import")
    import main_ppmi_pre as P
    print("import OK")

    # ── gate: eib 캐시 탐지/선택 ─────────────────────────────────────
    _hr(f"[gate] idx={args.idx}")
    best = P.find_best_eib_cache(args.idx)
    if best is None:
        print(f"[gate] idx {args.idx}: EIB 캐시 없음 → skip 대상.")
        return
    eib_corr, eib_path, eib_obj = best
    print(f"선택된 EIB 캐시: {os.path.basename(eib_path)}")
    print(f"  post_eib_corr = {eib_corr:.4f}")
    if target == STAGES.index("gate"):
        return

    # ── load: prepare + config + load_data ───────────────────────────
    _hr(f"[load] prepare + config + load_data  idx={args.idx}")
    with _Timer("load") as t:
        p = P.M.prepare_ppmi_data(args.idx, P.NOISE_LEVEL)
        cfg = P.M.make_config(p, args.idx)
        ms = 2 if args.quick else args.max_steps
        if ms is not None:
            cfg.optimizer_max_steps = int(ms)
            cfg.optimizer_chunk_steps = min(cfg.optimizer_chunk_steps, int(ms))
            print(f"[debug] optimizer_max_steps={cfg.optimizer_max_steps} "
                  f"chunk={cfg.optimizer_chunk_steps}")
        if args.window is not None:
            cfg.optimizer_bold_window_tr = int(args.window)
            cfg.optimizer_bold_skip_tr = int(args.skip) if args.skip is not None else int(args.window) // 5
            print(f"[debug] optimizer_bold_window_tr={cfg.optimizer_bold_window_tr} "
                  f"skip={cfg.optimizer_bold_skip_tr}  (메모리 probe)")
        data = P.load_data(cfg)
    timings["load"] = t.dt
    import numpy as np
    print(f"  n_nodes={data['n_nodes']}  cortex/sub="
          f"{len(data['cortex_indices'])}/{len(data['subcortex_indices'])}"
          f"  SC nonzero={int(np.asarray(data['sc_mask']).sum())}")
    print(f"  cache_tag={data['cache_tag']}")
    if target == STAGES.index("load"):
        return

    # ── build: network (warmup 1회, Part3 미사용) ───────────────────
    _hr("[build] build_network (warmup — Part3 state 와 무관)")
    with _Timer("build") as t:
        network, _is, _bm, _wr = P.build_network(cfg, data)
    timings["build"] = t.dt
    if target == STAGES.index("build"):
        return

    # ── eib: 캐시 .pkl 직접 로드 (재튜닝 없음) ──────────────────────
    _hr("[eib] EIB 캐시 .pkl 직접 로드 → bundle_eib")
    with _Timer("eib-load") as t:
        bundle_eib = P.StateBundle.from_dict(eib_obj["bundle"])
    timings["eib"] = t.dt
    print(f"  stage={bundle_eib.stage}  c_ei_frozen={bundle_eib.params.c_ei_frozen}  "
          f"mean c_ei={float(bundle_eib.params.c_ei.mean()):.4f}  fp={bundle_eib.fingerprint()}")
    if target == STAGES.index("eib"):
        return

    # ── grad: Part 3 ─────────────────────────────────────────────────
    _hr("[grad] run_gradient_optimization (Part 3 — 실제 실행)")
    with _Timer("grad") as t:
        bundle_grad = P.run_gradient_optimization(
            network   = network,
            bundle_in = bundle_eib,
            cfg       = cfg,
            data      = data,
        )
    timings["grad"] = t.dt
    meta = bundle_grad.metadata
    print(f"  grad stage={bundle_grad.stage}  c_ei_frozen={bundle_grad.params.c_ei_frozen}")
    print(f"  post_grad_fc_corr={meta.get('post_grad_fc_corr')}  "
          f"post_grad_fc_rmse={meta.get('post_grad_fc_rmse')}")

    _hr("timings (s)")
    for k in ["load", "build", "eib", "grad"]:
        if k in timings:
            print(f"  {k:>6}: {timings[k]:8.2f}")
    print("\nDEBUG RUN OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[DEBUG FAILED] {exc}")
        traceback.print_exc()
