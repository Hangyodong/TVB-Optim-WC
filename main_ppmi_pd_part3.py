#!/usr/bin/env python3
"""
main_ppmi_pd_part3.py — EIB 캐시를 직접 로드해 Part 3 만 실행하는 resume 러너.

목적
----
main_ppmi_pd.py 가 Part3 에서 죽어도(OOM 등) FIC/EIB 를 다시 안 돌리고 Part3 만 이어서.

왜 run_fic/run_eib 를 다시 안 부르나 (중요)
-------------------------------------------
run_fic/run_eib 캐시 키는 bundle_init.fingerprint() 에 의존하고, 이 fingerprint 는
warmup(수십만 step 적분) 최종상태를 포함한다. warmup 출력은 GPU 비결정성으로 프로세스 간
bit 재현이 안 되므로, 새 프로세스에서 run_fic 를 부르면 캐시 미스 → FIC/EIB 를 처음부터
재튜닝한다. 따라서 이 러너는 EIB 캐시 .pkl 을 직접 언피클해 bundle_eib 를 복원하고 곧장
Part3 에 넣는다. Part3 는 bundle_eib(전체 state)만 필요하며 warmup 결과는 쓰지 않는다.

사용:
    python3 main_ppmi_pd_part3.py --subject-idx 0
    python3 main_ppmi_pd_part3.py --subject-idx 0 --parcellation 200
    for i in $(seq 0 241); do python3 main_ppmi_pd_part3.py --subject-idx $i; done
"""
import argparse
import glob
import os
import pickle
import pathlib as _pl

# main_ppmi_pd import 시 JAX 초기화 + plt.show 패치 + prepare/make_config 재사용.
import main_ppmi_pd as M
from data_loader        import load_data
from model              import build_network
from part3_gradient     import run_gradient_optimization
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)


def parse_args():
    ap = argparse.ArgumentParser(description="tr_2.5_PD Part3-only resume 러너")
    ap.add_argument("--subject-idx",  dest="subject_idx", type=int, default=0)
    ap.add_argument("--parcellation", dest="parcellation", type=int, choices=[100, 200], default=100)
    ap.add_argument("--noise-level",  dest="noise_level", type=float, default=0.02)
    return ap.parse_args()


def _load_best_eib(cache_dir: str) -> dict:
    """cache_dir 의 eib_*.pkl 을 모두 스캔 → post_eib_corr(없으면 best_fc_corr) 최대 선택."""
    pkls = glob.glob(os.path.join(cache_dir, "eib_*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"EIB 캐시 없음: {cache_dir}/eib_*.pkl\n"
            f"  → 먼저 main_ppmi_pd.py 로 FIC+EIB 를 돌려 EIB 캐시를 만들어야 한다.")
    best, best_corr, best_path = None, float("-inf"), None
    for f in sorted(pkls):
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)} 언피클 실패: {e}")
            continue
        corr = r.get("post_eib_corr", r.get("best_fc_corr", float("nan")))
        corr = float(corr) if corr is not None else float("nan")
        print(f"  eib pkl {os.path.basename(f)[:60]}...  post_eib_corr={corr:.4f}")
        if corr == corr and corr > best_corr:   # finite & max
            best, best_corr, best_path = r, corr, f
    if best is None:
        raise RuntimeError("유효한 EIB 캐시(finite corr) 없음")
    print(f"  → 선택: {os.path.basename(best_path)[:60]}...  corr={best_corr:.4f}")
    return best


def main():
    args = parse_args()
    print("=" * 60)
    print(f"  Part3-only resume — tr_2.5_PD schaefer{args.parcellation} (idx={args.subject_idx})")
    print("=" * 60)

    # 1) 입력 추출 + config (main_ppmi_pd 와 동일 → 같은 cache_dir/output 경로)
    p = M.prepare_pd_data(args.subject_idx, args.parcellation, args.noise_level)
    M._FIG["dir"] = _pl.Path(p["out_dir"]) / "figures"
    M._FIG["dir"].mkdir(parents=True, exist_ok=True)
    cfg = M.make_config(p, args.subject_idx)

    # warmup 은 network graph 빌드용으로만 필요(결과 미사용) → 짧게(비용 절감).
    cfg.warmup_duration_ms = 2000

    # 2) 데이터 + network
    print("\n[0] Loading data + building network (short warmup)...")
    data = load_data(cfg)
    import numpy as _np
    _np.fill_diagonal(data["fc_target"], 0.0)
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

    # warmup_bundle (Part3 는 참조만, 미사용 — 서명 호환용)
    initial_params = ParamSet.default(
        data["n_nodes"], c_ei_init=cfg.wc_c_ei_init
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)
    bundle_init = StateBundle.from_warmup(
        warmup_result=warmup_result, bold_monitor_template=bold_monitor,
        initial_params=initial_params, internal_state=capture_internal_state(initial_state),
        delay_history=capture_network_delay_history(network), stage="warmup",
    )

    # 3) EIB 캐시 직접 로드 → bundle_eib
    print(f"\n[EIB] 캐시 로드: {data['cache_dir']}")
    eib_result = _load_best_eib(data["cache_dir"])
    bundle_eib = StateBundle.from_dict(eib_result["bundle"])
    bundle_eib.apply_to_network(network)
    print(f"[EIB] bundle 복원 완료  stage={bundle_eib.stage}  "
          f"c_ei_frozen={bundle_eib.params.c_ei_frozen}")

    # 4) Part 3 (Full Gradient)만 실행
    print("\n[3] Running Gradient Optimization (resume)...")
    bundle_grad = run_gradient_optimization(
        network=network, bundle_in=bundle_eib, warmup_bundle=bundle_init, cfg=cfg, data=data,
    )
    print(f"[Part3] stage={bundle_grad.stage}  "
          f"post_grad_fc_corr={bundle_grad.metadata.get('post_grad_fc_corr')}")

    print("\n" + "=" * 60)
    print(f"  Part3 resume complete. → {p['out_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
