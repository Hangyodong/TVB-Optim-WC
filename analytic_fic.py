#!/usr/bin/env python3
"""해석적 FIC — 시뮬 없이 c_ei 를 닫힌형으로 푼다 (Deco 2014 방식, RWW 고정점).

목표 S_e* = 0.25 를 전 노드에 강제하면:
  H_e* = S_e*/(tau_e*(1-S_e*)*gamma_e)          <- 전 노드 공통 스칼라
  x_e* = H_e^{-1}(H_e*)                          <- 스칼라 1개 근 찾기
  c_lre_i = J_N*S_e* * sum_j W_ij*wLRE_ij        <- 닫힌형
  c_ffi_i = J_N*S_e* * sum_j W_ij*wFFI_ij        <- 닫힌형
  S_i_i : x_i = a_i*(J_N*S_e* - S_i + W_i*I_o + lamda*c_ffi_i) - b_i,
          S_i = tau_i*gamma_i*H(x_i,d_i)          <- 노드별 스칼라 고정점
  c_ei_i = [w_p*J_N*S_e* + W_e*I_o + c_lre_i - (x_e*+b_e)/a_e] / S_i_i

시뮬 0회. 밀리초.
"""
import numpy as np

P = dict(a_e=310.0, b_e=125.0, d_e=0.160, gamma_e=0.641 / 1000, tau_e=100.0,
         w_p=1.4, W_e=1.0, a_i=615.0, b_i=177.0, d_i=0.087, gamma_i=1.0 / 1000,
         tau_i=10.0, W_i=0.7, J_N=0.15, I_o=0.382, I_ext=0.0, lamda=1.0)
IU = np.triu_indices(163, 1)
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025"}


def H(x, d):
    x = np.asarray(x, np.float64)
    out = np.where(np.abs(d * x) < 1e-9, 1.0 / d, 0.0)
    safe = np.where(np.abs(d * x) < 1e-9, 1.0, x)
    return np.where(np.abs(d * x) < 1e-9, 1.0 / d, safe / (1.0 - np.exp(-d * safe)))


def inv_H(target, d):
    """H 는 단조증가. 이분법."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if H(mid, d) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sc_norm_log1pm(W):
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    mask = (W > 0).astype(np.float64)
    w_max_mean = (W / W.max()).sum(1).mean()
    Wl = np.log1p(W) * mask
    return Wl * (w_max_mean / Wl.sum(1).mean())


def analytic_c_ei(Wn, wLRE, wFFI, S_e_star=0.25, p=P):
    H_e_star = S_e_star / (p["tau_e"] * (1 - S_e_star) * p["gamma_e"])
    x_e_star = inv_H(H_e_star, p["d_e"])
    L = (Wn * wLRE).sum(1)
    F = (Wn * wFFI).sum(1)
    c_lre = p["J_N"] * S_e_star * L
    c_ffi = p["J_N"] * S_e_star * F

    # S_i 노드별 고정점 (damped iteration)
    S_i = np.full(Wn.shape[0], 0.1)
    for _ in range(500):
        x_i = p["a_i"] * (p["J_N"] * S_e_star - S_i + p["W_i"] * p["I_o"]
                          + p["lamda"] * c_ffi) - p["b_i"]
        S_i_new = p["tau_i"] * p["gamma_i"] * H(x_i, p["d_i"])
        S_i = 0.5 * S_i + 0.5 * S_i_new
    x_i = p["a_i"] * (p["J_N"] * S_e_star - S_i + p["W_i"] * p["I_o"]
                      + p["lamda"] * c_ffi) - p["b_i"]

    num = (p["w_p"] * p["J_N"] * S_e_star + p["W_e"] * p["I_o"] + c_lre
           + p["I_ext"] - (x_e_star + p["b_e"]) / p["a_e"])
    return num / S_i, dict(H_e_star=H_e_star, x_e_star=x_e_star, S_i=S_i,
                           c_lre=c_lre, c_ffi=c_ffi, L=L, F=F)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/scratch/home/wog3597/optim")
    from formula_reproduce import load_ref

    print(f"{'idx':>3} {'c_ei 해석해':>22} {'c_ei FIC/optim':>22} {'노드별 corr':>10}  S_i 범위")
    for idx, sub in SUBDIR.items():
        ref, sc, _ = load_ref(sub)
        if ref is None:
            continue
        W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        Wn = sc_norm_log1pm(W)
        c_a, aux = analytic_c_ei(Wn, ref["wLRE"], ref["wFFI"])
        c_o = np.asarray(ref["c_ei"], np.float64)
        r = np.corrcoef(c_a, c_o)[0, 1]
        print(f"{idx:>3}  mean {c_a.mean():7.3f} sd {c_a.std():6.3f}   "
              f"mean {c_o.mean():7.3f} sd {c_o.std():6.3f}   r={r:+.4f}  "
              f"S_i {aux['S_i'].min():.4f}~{aux['S_i'].max():.4f}")
    print(f"\nH_e* = {analytic_c_ei(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))[1]['H_e_star']:.4f} Hz"
          f"   x_e* = {analytic_c_ei(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))[1]['x_e_star']:.6f}")
