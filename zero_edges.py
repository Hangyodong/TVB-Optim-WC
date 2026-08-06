#!/usr/bin/env python3
"""SC>0 인데 wLRE/wFFI 가 0 인 엣지의 정체 — 박스제약 활성면인가, 적합 부족인가, 노드 문제인가."""
import glob
import hashlib
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/scratch/home/wog3597/optim")
IU = np.triu_indices(163, 1)
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025"}


def load_ref(sub):
    hs = {hashlib.sha1(np.asarray(np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv",
                                             delimiter=",")[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        # 주의: 'frozen' 은 정상 파일명의 frozen0 과 충돌한다 — 쓰지 말 것.
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest")):
            continue
        if not any(h in os.path.basename(os.path.dirname(f)) for h in hs):
            continue
        d = pickle.load(open(f, "rb"))
        b = d.get("bundle", d) if isinstance(d, dict) else d
        md = (b.get("metadata", {}) if isinstance(b, dict) else getattr(b, "metadata", {})) or {}
        c = float(md.get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = b.get("params") if isinstance(b, dict) else getattr(b, "params")
            g = (lambda o, k: o.get(k) if isinstance(o, dict) else getattr(o, k))
            best = {k: np.asarray(g(p, k), np.float64) for k in ("c_ei", "wLRE", "wFFI")}
            bc = c
    return best, bc


print("가설 A: 아핀관계 clip → wFFI=0 은 FC 높은 엣지, wLRE=0 은 FC 매우 낮은 엣지")
print("가설 B: 적합 부족(무작위) → 0 엣지 FC 분포가 전체와 같음")
print("가설 C: 노드 문제 → 0 이 특정 노드에 몰림\n")
print(f"{'idx':>3} {'nSC>0':>6} | {'wFFI=0':>7} {'그 FC 평균':>10} {'전체 FC 평균':>11} {'FC>임계 비율':>11} {'일치도':>7} "
      f"| {'wLRE=0':>7} {'그 FC 평균':>10} | {'노드집중':>8}")
for idx, sub in SUBDIR.items():
    ref, sc = load_ref(sub)
    if ref is None:
        print(f"{idx:>3}  캐시 없음")
        continue
    W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
    FC = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
    np.fill_diagonal(FC, 0.0)
    m = (W > 0)[IU]
    x = FC[IU][m]
    wl, wf = ref["wLRE"][IU][m], ref["wFFI"][IU][m]
    zf, zl = wf <= 1e-6, wl <= 1e-6

    # 자기 subject 아핀 계수로 영점 임계 계산
    A = np.stack([x, np.ones(x.size)], 1)
    (bf, af), *_ = np.linalg.lstsq(A, wf, rcond=None)
    (bl, al), *_ = np.linalg.lstsq(A, wl, rcond=None)
    thr_f = -af / bf if bf else np.nan          # wFFI=0 되는 FC (bf<0 이므로 상한)
    thr_l = -al / bl if bl else np.nan
    pred_f = x > thr_f
    agree = (pred_f == zf).mean() * 100

    # 노드 집중도: 0 엣지가 특정 노드에 몰리나 (엣지당 노드 2개 카운트)
    ii, jj = IU[0][m], IU[1][m]
    cnt = np.bincount(np.concatenate([ii[zf], jj[zf]]), minlength=163)
    conc = cnt.max() / max(1, cnt.sum()) * 163   # 1.0=균등, 클수록 집중

    print(f"{idx:>3} {m.sum():>6} | {zf.sum():>7} {x[zf].mean():>10.4f} {x.mean():>11.4f} "
          f"{pred_f.mean()*100:>10.1f}% {agree:>6.1f}% | {zl.sum():>7} "
          f"{(x[zl].mean() if zl.sum() else np.nan):>10.4f} | {conc:>8.2f}")
    if idx == 4:
        print(f"      아핀: wFFI = {af:+.4f} {bf:+.4f}*FC -> 영점 FC={thr_f:.4f}")
        print(f"            wLRE = {al:+.4f} {bl:+.4f}*FC -> 영점 FC={thr_l:.4f}")
        print(f"      wFFI=0 엣지 FC 범위 {x[zf].min():.3f}~{x[zf].max():.3f}  "
              f"(전체 {x.min():.3f}~{x.max():.3f})")
        if zl.sum():
            print(f"      wLRE=0 엣지 FC 범위 {x[zl].min():.3f}~{x[zl].max():.3f}")
