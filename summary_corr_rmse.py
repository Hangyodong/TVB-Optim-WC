#!/usr/bin/env python3
"""
summary_corr_rmse.py — idx 별 Part2(EIB) / Part3(grad) 최종 corr·rmse 요약

각 idx 의 캐시(.pkl)를 직접 언피클해 stage 별 최종 FC corr/rmse 를 표로 출력한다.
- Part2 = eib_*.pkl 의 post_eib_corr / post_eib_rmse (EIB 최종)
- Part3 = grad_*.pkl 의 bundle.metadata.post_grad_fc_corr / post_grad_fc_rmse
- 같은 idx 에 캐시가 여러 개면 corr 최고를 선택.
- Part3 캐시 없으면 Part2 까지만 출력(grad 칸은 '-').

network/데이터 로드·warmup 없이 pickle read 만 → 빠르고 CPU 전용.
실행: python summary_corr_rmse.py
"""
import glob
import os
import pickle

CACHE_ROOT = "cache/ppmi216"
IDX_SCAN   = range(0, 9)


def _best_cache(idx: int, prefix: str, corr_getter):
    """idx 의 {prefix}_*.pkl 중 corr 최고를 선택. 반환 (corr, rmse, dirname) 또는 None."""
    paths = glob.glob(os.path.join(CACHE_ROOT, f"v_ppmi216_s{idx}_*", f"{prefix}_*.pkl"))
    best = None
    for path in paths:
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:
            print(f"  [warn] {os.path.basename(path)} 로드 실패: {exc}")
            continue
        if not (isinstance(obj, dict) and "bundle" in obj):
            continue
        corr, rmse = corr_getter(obj)
        if corr is None:
            continue
        if best is None or corr > best[0]:
            best = (float(corr), (float(rmse) if rmse is not None else float("nan")),
                    os.path.basename(os.path.dirname(path)))
    return best


def _eib_corr_rmse(obj: dict):
    corr = obj.get("post_eib_corr")
    rmse = obj.get("post_eib_rmse")
    if corr is None:  # 구캐시 fallback
        corr = obj.get("best_fc_corr")
        rmse = obj.get("best_fc_rmse")
    return corr, rmse


def _grad_corr_rmse(obj: dict):
    m = obj.get("bundle", {}).get("metadata", {})
    return m.get("post_grad_fc_corr"), m.get("post_grad_fc_rmse")


def main():
    print("=" * 78)
    print("  idx 별 Part2(EIB) / Part3(grad) 최종 corr·rmse  (corr 최고 캐시 선택)")
    print("=" * 78)
    header = f"  {'idx':>3} | {'P2 corr':>8} {'P2 rmse':>8} | {'P3 corr':>8} {'P3 rmse':>8} | last"
    print(header)
    print("  " + "-" * 74)

    rows = []
    for idx in IDX_SCAN:
        eib  = _best_cache(idx, "eib",  _eib_corr_rmse)
        grad = _best_cache(idx, "grad", _grad_corr_rmse)
        if eib is None and grad is None:
            continue
        if eib is not None:
            p2c, p2r, _ = eib
            p2c_s, p2r_s = f"{p2c:.4f}", f"{p2r:.4f}"
        else:
            p2c = float("-inf"); p2c_s = p2r_s = "   -   "
        if grad is not None:
            p3c, p3r, _ = grad
            p3c_s, p3r_s, last = f"{p3c:.4f}", f"{p3r:.4f}", "part3"
        else:
            p3c_s = p3r_s = "   -   "; last = "part2"
        print(f"  {idx:>3} | {p2c_s:>8} {p2r_s:>8} | {p3c_s:>8} {p3r_s:>8} | {last}")
        rows.append((idx, eib, grad))

    print("  " + "-" * 74)

    # corr 랭킹 (Part3 우선, 없으면 Part2)
    def _rank_corr(row):
        idx, eib, grad = row
        if grad is not None:
            return grad[0]
        return eib[0] if eib is not None else float("-inf")
    ranked = sorted(rows, key=_rank_corr, reverse=True)
    print("  최종 corr 랭킹 (Part3 있으면 Part3, 없으면 Part2):")
    for rank, row in enumerate(ranked):
        idx, eib, grad = row
        stage = "P3" if grad is not None else "P2"
        print(f"    {rank+1}. s{idx}  corr={_rank_corr(row):.4f}  ({stage})")
    print("=" * 78)


if __name__ == "__main__":
    main()
