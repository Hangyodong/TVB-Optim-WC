#!/usr/bin/env python3
"""make_perfect_corr_weights.py — idx4 의 wLRE/wFFI 를 FC 와 corr ±1 이 되도록 변형.

r = +1  <=>  wLRE 가 FC 의 **증가 아핀함수**   (잔차 0)
r = −1  <=>  wFFI 가 FC 의 **감소 아핀함수**

제약 (파이프라인 sanitize 와 동일):
  0 이상, SC==0 엣지는 0, 대칭, [0, w_max] clip.

아핀 사상은 자유도가 2 개(기울기·절편)다. 여기서는 **원본의 범위 [min, max] 를 고정**한다:
  원본 min 은 wLRE/wFFI 모두 정확히 0 이므로, 결과적으로 min=0 · max=원본 max 로 양 끝을 못박는다.
    wLRE_new = max_wLRE · (FC − FC_min) / (FC_max − FC_min)      (증가)
    wFFI_new = max_wFFI · (FC_max − FC) / (FC_max − FC_min)      (감소)
  전 구간이 [0, max] 안이라 clip 이 no-op → r 이 정확히 ±1.

⚠ 모멘트 매칭(평균·sd 보존)은 쓸 수 없다 — 음수가 나오고(wLRE 46 개, wFFI 1094 개)
  clip 이 걸리는 순간 r 이 ±1 에서 깨진다.
⚠ max 를 고정하면 평균은 보존되지 않는다 (자유도가 2 개뿐이라 둘 다는 불가).

실행: python3 make_perfect_corr_weights.py
"""
import numpy as np

IDX = 4
NPZ = "output_ppmi_pd/_rel_data.npz"
OUT = "output_ppmi_pd/_rel_data_idx4_perfect.npz"


def affine_to_fc(FC, mask, target_max, increasing):
    """FC 의 아핀함수. mask 안에서 min=0, max=target_max, 단조 방향 지정."""
    f = FC[mask]
    span = f.max() - f.min()
    out = np.zeros_like(FC)
    out[mask] = target_max * ((f - f.min()) if increasing else (f.max() - f)) / span
    return out


def main():
    d = np.load(NPZ)
    FC = d[f"fc_{IDX}"].astype(np.float64)
    N = d[f"norm_{IDX}"].astype(np.float64)
    WL = d[f"wlre_{IDX}"].astype(np.float64)
    WF = d[f"wffi_{IDX}"].astype(np.float64)
    M = N > 0
    np.fill_diagonal(M, False)

    WL2 = affine_to_fc(FC, M, WL[M].max(), True)
    WF2 = affine_to_fc(FC, M, WF[M].max(), False)

    # 파이프라인과 동일한 투영: clip → 마스크 → 대칭화
    for A in (WL2, WF2):
        np.clip(A, 0.0, 10.0, out=A)
        A *= M
        A[:] = 0.5 * (A + A.T)

    print(f"idx{IDX}  SC>0 엣지 {int(M.sum())}")
    print(f"{'':12}{'r(FC,·)':>10}{'평균':>9}{'sd':>8}{'min':>8}{'max':>8}{'=0 개수':>9}")
    for nm, A in (("wLRE 원본", WL), ("wLRE 변형", WL2),
                  ("wFFI 원본", WF), ("wFFI 변형", WF2)):
        v = A[M]
        r = np.corrcoef(FC[M], v)[0, 1]
        print(f"{nm:12}{r:>+10.6f}{v.mean():>9.4f}{v.std():>8.4f}"
              f"{v.min():>8.4f}{v.max():>8.4f}{int((v <= 1e-12).sum()):>9}")

    s0, s1 = (WL + WF)[M], (WL2 + WF2)[M]
    print(f"\n합 wLRE+wFFI : 원본 {s0.mean():.4f} ± {s0.std():.4f}   "
          f"변형 {s1.mean():.4f} ± {s1.std():.4f}")
    print(f"  |합−2| < 0.05 비율: 원본 {(np.abs(s0-2)<0.05).mean()*100:.1f}%  "
          f"변형 {(np.abs(s1-2)<0.05).mean()*100:.1f}%")
    print(f"대칭 확인: wLRE {np.allclose(WL2, WL2.T)}  wFFI {np.allclose(WF2, WF2.T)}")
    print(f"SC==0 에서 0 확인: wLRE {np.all(WL2[~M] == 0)}  wFFI {np.all(WF2[~M] == 0)}")

    out = {k: d[k] for k in d.files}
    out[f"wlre_{IDX}_perf"] = WL2.astype(np.float32)
    out[f"wffi_{IDX}_perf"] = WF2.astype(np.float32)
    np.savez_compressed(OUT, **out)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
