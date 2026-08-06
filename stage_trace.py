#!/usr/bin/env python3
"""stage_trace.py — Part1(FIC) → Part2(EIB) → Part3(grad) 로 c_ei / mean_S_e 가 어떻게 변하나.

각 단계 캐시 bundle 을 같은 방식으로 600s settle 후 60s 시간·노드 평균 S_e 를 잰다
(se_at_optimum.py 와 동일 척도). 파이프라인 본선 캐시만 쓴다 —
cache_version 에 추가 태그(analytic25/formulafic/seopt/...) 붙은 곁가지 실험은 제외.

실행: python3 stage_trace.py --idxs 4,5,6,7,8
"""
import argparse
import glob
import hashlib
import os
import pickle
import re

import numpy as np
import jax

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part3_gradient import _settle_bundle
from pipeline_contracts import StateBundle
from formula_2x2 import SUBDIR, measure_se, _get

# 본선 캐시 디렉터리: 추가 태그 없는 v_pdaal163_tr25_s<N>_delay3_N163_sc..._fc...
PLAIN_DIR = re.compile(r"v_pdaal163_tr25_s\d+_delay3_N163_sc\w+_fc\w+$")
# Part1 본선 FIC: eta=0.5, 2000 step (eta0p0 은 c_ei 고정 probe 라 제외)
FIC_PIPE = "_eta0p5_steps2000_"


def plain_cache_dir(sub):
    FC = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
    hs = {hashlib.sha1(np.asarray(FC[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    for d in glob.glob(f"output_ppmi_pd/{sub}/cache/*"):
        b = os.path.basename(d)
        if PLAIN_DIR.fullmatch(b) and any(h in b for h in hs):
            return d
    return None


def newest(paths):
    return max(paths, key=os.path.getmtime) if paths else None


def _wlre_uniform(path):
    """Part1 출발점(wLRE=wFFI=1 균일)인지. 재진입 FIC 캐시를 걸러낸다."""
    try:
        w = np.asarray(load_bundle(path)[0].params.wLRE)
        nz = w != 0.0          # SC==0 엣지는 항상 0 → 전체 std 로는 판별 못 한다
        return bool(nz.any()) and float(w[nz].std()) == 0.0
    except Exception:
        return False


def stage_files(cdir):
    # Part1 은 균일 가중치(wLRE=wFFI=1)에서 출발한 실행만 본선이다. 같은 디렉터리에
    # 이미 튜닝된 가중치 위에서 FIC 를 다시 돌린 재진입 캐시가 섞여 있어(태그 없이 저장됨)
    # mtime 으로 고르면 그쪽이 잡힌다 — wLRE 균일 여부로 판별한다.
    cands = [f for f in glob.glob(f"{cdir}/fic_*.pkl") if FIC_PIPE in f]
    fresh = [f for f in cands if _wlre_uniform(f)]
    if cands and not fresh:
        print(f"  [warn] {os.path.basename(cdir)}: 균일 가중치 Part1 없음 → 재진입 캐시 사용")
    fic = newest(fresh or cands)
    return {"part1_fic": fic,
            "part2_eib": newest(glob.glob(f"{cdir}/eib_*.pkl")),
            "part3_grad": newest(glob.glob(f"{cdir}/grad_*.pkl"))}


def load_bundle(path):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    b = _get(d, "bundle", d)
    md = (_get(b, "metadata", {}) or {})
    return StateBundle.from_dict(b if isinstance(b, dict) else b.to_dict()), md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", default="4,5,6,7,8")
    ap.add_argument("--noise-level", type=float, default=0.02)
    a = ap.parse_args()

    rows = []
    for idx in [int(v) for v in a.idxs.split(",") if v.strip()]:
        sub = SUBDIR[idx]
        cdir = plain_cache_dir(sub)
        if cdir is None:
            print(f"idx {idx} ({sub}): 본선 캐시 디렉터리 없음 — skip", flush=True)
            continue
        files = stage_files(cdir)
        if any(v is None for v in files.values()):
            miss = [k for k, v in files.items() if v is None]
            print(f"idx {idx} ({sub}): 단계 캐시 누락 {miss} — skip", flush=True)
            continue

        p = M.prepare_pd_data(idx, a.noise_level)
        cfg = M.make_config(p, idx, use_delay=True)
        cfg.cache_version = f"{cfg.cache_version}_trace"
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)
        network, *_ = build_network(cfg, data)
        t1 = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

        for stage, path in files.items():
            b, md = load_bundle(path)
            c0 = np.asarray(b.params.c_ei, np.float32)
            b, _, _ = _settle_bundle(network, b, cfg, sim_duration_ms=t1,
                                     skip_tr=cfg.optimizer_bold_skip_tr,
                                     next_stage="eib")
            se = measure_se(network, b, cfg)
            corr = float(md.get("post_grad_fc_corr",
                                md.get("post_eib_fc_corr",
                                       md.get("post_fic_fc_corr", np.nan))))
            rows.append(dict(idx=idx, stage=stage, c_mean=float(c0.mean()),
                             c_min=float(c0.min()), c_max=float(c0.max()),
                             c_sd=float(c0.std()), se=se, corr=corr))
            print(f"idx {idx} {stage:11} c_ei {c0.mean():.4f} "
                  f"[{c0.min():.3f},{c0.max():.3f}] sd {c0.std():.4f}  "
                  f"mean_S_e {se:.4f}  corr {corr:.4f}", flush=True)
        jax.clear_caches()

    if not rows:
        return
    print("\n" + "=" * 88)
    print(f"{'idx':>4} {'stage':<12} {'c_ei mean':>10} {'c_ei sd':>9} "
          f"{'c_ei 범위':>16} {'mean_S_e':>9} {'corr':>8}")
    print("-" * 88)
    for r in rows:
        print(f"{r['idx']:>4} {r['stage']:<12} {r['c_mean']:>10.4f} {r['c_sd']:>9.4f} "
              f"{'[%.3f, %.3f]' % (r['c_min'], r['c_max']):>16} "
              f"{r['se']:>9.4f} {r['corr']:>8.4f}")

    print("\n단계간 변화 (Part1 기준)")
    print(f"{'idx':>4} {'d c_ei (1→2)':>13} {'d c_ei (2→3)':>13} {'d c_ei (1→3)':>13} "
          f"{'d S_e (1→3)':>12}")
    for idx in sorted({r["idx"] for r in rows}):
        g = {r["stage"]: r for r in rows if r["idx"] == idx}
        if len(g) < 3:
            continue
        c1, c2, c3 = (g[k]["c_mean"] for k in ("part1_fic", "part2_eib", "part3_grad"))
        s1, s3 = g["part1_fic"]["se"], g["part3_grad"]["se"]
        print(f"{idx:>4} {c2-c1:>+12.4f}% {c3-c2:>+12.4f}  {c3-c1:>+12.4f}  {s3-s1:>+12.4f}"
              .replace("%", " "))
    with open("output_ppmi_pd/_stage_trace.json", "w") as fh:
        import json
        json.dump(rows, fh, indent=1)
    print("\n저장: output_ppmi_pd/_stage_trace.json")


if __name__ == "__main__":
    main()
