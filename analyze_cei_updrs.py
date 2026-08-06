#!/usr/bin/env python3
"""c_ei(part3 fitted) vs MDS-UPDRS-III 중증도 상관 — PD 바이오마커 가설 검증.

가설: PD 병태 = hypersync/약한 억제 → 낮은 c_ei. 낮을수록 중증(UPDRS3↑) → 음의 상관.
배치가 act1p0 grad 캐시를 더 만들수록 N 자동 증가. 그냥 다시 실행하면 갱신됨.

  python3 analyze_cei_updrs.py
"""
import glob, os, pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

MAT = "data/AALv3/FC_AAL_ComBat_all_163_PD.mat"
UPDRS = "data/MDS-UPDRS_Part_III_13Jul2026.csv"


def _cei_mean(sub):
    fs = sorted(glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*act1p0*.pkl"),
                key=os.path.getmtime)
    if not fs:
        return None
    d = pickle.load(open(fs[-1], "rb"))

    def find(o, k):
        if isinstance(o, dict):
            if k in o:
                return o[k]
            for v in o.values():
                r = find(v, k)
                if r is not None:
                    return r
        return None
    c = find(d.get("bundle", d), "c_ei")
    return float(np.asarray(c, dtype=float).mean()) if c is not None else None


def _scan_month(session):
    s = str(session)
    return int(s[:4]) * 12 + int(s[4:6]) if len(s) >= 6 else None


def _infodt_month(v):
    # "MM/YYYY"
    try:
        mm, yy = str(v).split("/")
        return int(yy) * 12 + int(mm)
    except Exception:
        return None


def _pick_updrs(df_sub, scan_m):
    """scan 시점에 가장 가까운 visit, OFF/drug-naive 우선, NP3TOT 반환."""
    d = df_sub.dropna(subset=["NP3TOT"]).copy()
    if d.empty:
        return None, None, None
    d["m"] = d["INFODT"].map(_infodt_month)
    d["dist"] = (d["m"] - scan_m).abs() if scan_m else 0
    # OFF 또는 drug-naive(PDMEDYN==0) 우선순위(0), ON은 후순위(1)
    d["offpri"] = np.where((d["PDSTATE"] == "OFF") | (d["PDMEDYN"] == 0), 0, 1)
    d = d.sort_values(["dist", "offpri"])
    top = d.iloc[0]
    return float(top["NP3TOT"]), top["EVENT_ID"], ("OFF/naive" if top["offpri"] == 0 else "ON")


def main():
    m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)["FC_ComBat"]
    sess = {int(e.subject): _scan_month(e.session) for e in m}
    df = pd.read_csv(UPDRS, low_memory=False)

    rows = []
    for sub in sorted(sess):
        c = _cei_mean(sub)
        if c is None:
            continue  # 아직 fitting 안 된 subject
        u, visit, med = _pick_updrs(df[df["PATNO"] == sub], sess[sub])
        if u is None:
            continue
        rows.append((sub, c, u, visit, med))

    if len(rows) < 2:
        print(f"fitting 완료 subject {len(rows)}명 — 상관엔 최소 2명 필요.")
        return

    out = pd.DataFrame(rows, columns=["subject", "c_ei_mean", "UPDRS3", "visit", "med"])
    out = out.sort_values("c_ei_mean")
    print(out.to_string(index=False))
    out.to_csv("output_ppmi_pd/cei_updrs_pairs.csv", index=False)

    cs, us = out["c_ei_mean"].values, out["UPDRS3"].values
    sr, sp = stats.spearmanr(cs, us)
    pr, pp = stats.pearsonr(cs, us)
    print(f"\nN={len(rows)}  가설: c_ei↓ → UPDRS3↑ (음의 상관 기대)")
    print(f"  Spearman r={sr:+.3f}  p={sp:.4f}")
    print(f"  Pearson  r={pr:+.3f}  p={pp:.4f}")
    if len(rows) < 10:
        print("  ⚠ N<10 — 통계적 의미 없음(방향만 참고). 배치 더 돌려 N↑ 후 재판정.")


if __name__ == "__main__":
    main()
