#!/usr/bin/env python3
"""dbs_amp_calib.py — DBS amp(nA) ↔ 임상 mA 캘리브레이션.

임상에서 mA 를 올리면 뉴런 하나가 받는 전류가 커지는 게 아니라(축삭은 역치 넘으면
펄스당 1스파이크로 포화) VTA 가 넓어져 recruit 되는 뉴런 수가 늘어난다.
mean-field 노드는 STN 집단 평균이라 '바깥쪽 뉴런'이 없으므로, 그 recruit 비율을
집단 평균 발화율로 받아야 한다. 그래서 대응은 발화율 축에서 만난다:

    mA  → VTA 반경 → STN 내 recruit 비율 f → 목표 H_e = (1-f)*H_base + f*freq   (해석식)
    amp → 시뮬레이션 → 자극 중 평균 H_e                                          (이 스크립트)

사용:
    python3 dbs_amp_calib.py --subject-idx 4 --target STN_L
    python3 dbs_amp_calib.py --subject-idx 4 --target STN_L --amps 0.05,0.1,0.2,0.4

출력: <out_dir>/dbs_amp_calib/<target>/{calib.csv, calib.png}

GOTCHA: --noise-level 기본 0.02. 0.01 로 돌리면 cache_tag 가 σ 를 포함하지 않아
캐시가 조용히 매칭되고 결과만 달라진다. 바꾸지 말 것.
"""
import argparse
import glob
import os
import pickle
import sys
import types

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib as _pl

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from pipeline_contracts import StateBundle
from model import ReducedWongWangEIB
from part4_dbs import (
    _build_biphasic_pulse_train,
    _compute_derived_parameters,
    _make_stimulated_dynamics,
)

import jax
import jax.numpy as jnp
from tvboptim.experimental.network_dynamics import prepare
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun


def _get(row, key, default):
    """옛 CSV 에는 frac_at_bound 열이 없다 → --replot 하위호환."""
    v = row.get(key, default)
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def _h_e_from_s_e(s_e, tau_e, gamma_e):
    """기록된 S_e 에서 발화율 H_e[Hz] 복원.

    모니터는 state(S_e,S_i) 2채널만 남기고 aux(rE_hz)는 버린다. 하지만
        dS_e/dt = -S_e/tau_e + (1-S_e)*H_e*gamma_e
    를 뒤집으면 H_e 가 나온다. 노이즈가 E(=S_e)에 직접 들어가 수치미분은
    증폭되므로, 창 평균에서 <dS_e/dt> ~ 0 인 준정상 상태 형태를 쓴다.
    tau_e=100ms 라 125Hz 펄스열은 S_e 가 저역통과 → 창 평균에서 유효.
    자극 개시 직후 transient 는 호출부에서 잘라낸다.
    """
    return (s_e / tau_e) / ((1.0 - s_e) * gamma_e)


# ── 임상 mA → 목표 집단 발화율 (해석식) ─────────────────────────

def clinical_targets(h_base_hz, stim_freq_hz, sigma=0.2, e_th=150.0,
                     stn_semi_axes=(6.0, 2.0, 3.0), n_mc=2_000_000, seed=0):
    """단극 점전원 + 균질 등방 조직 가정.

    E(r) = I/(4*pi*sigma*r^2) → r_VTA = sqrt(I / (4*pi*sigma*E_th))
    STN 을 타원체로 두고 전극이 중심에 있다고 가정, recruit 비율은 몬테카를로.

    sigma  : 회백질 전도도 [S/m], 0.1~0.3 범위
    e_th   : 큰 유수축삭 활성화 역치 [V/m] @90us, 100~200 범위
    r_VTA ∝ 1/sqrt(sigma*e_th) 이고 작은 r 에서 f ~ r^3 이라 이 둘이 절대값을 지배한다.
    """
    a, b, c = stn_semi_axes
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, (n_mc, 3)) * np.array([a, b, c])
    pts = u[(u ** 2 / np.array([a, b, c]) ** 2).sum(1) <= 1.0]
    radius = np.linalg.norm(pts, axis=1)

    rows = []
    for i_ma in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        r_mm = np.sqrt((i_ma * 1e-3) / (4 * np.pi * sigma * e_th)) * 1e3
        f = float((radius <= r_mm).mean())
        rows.append({
            "I_mA": i_ma,
            "r_VTA_mm": r_mm,
            "f_recruit": f,
            "H_e_target_hz": (1.0 - f) * h_base_hz + f * stim_freq_hz,
        })
    return rows


# ── amp → 자극 중 평균 H_e (시뮬레이션) ─────────────────────────

def measure_h_e(network, bundle, cfg, data, node_index, amp, derived,
                tau_e, gamma_e, settle_ms=500.0):
    """단일 amp 에 대해 pre/during 평균 H_e 를 잰다."""
    dt = cfg.integration_dt_ms
    pre_steps = int(round(cfg.dbs_pre_stimulation_duration_ms / dt))
    stim_steps = derived["n_pulses"] * derived["period_steps"]
    total_steps = pre_steps + stim_steps

    stim_array, _ = _build_biphasic_pulse_train(
        target_node_index=node_index,
        n_nodes=data["n_nodes"],
        onset_step=pre_steps,
        amplitude=amp,
        phase_duration_steps=derived["phase_duration_steps"],
        n_pulses=derived["n_pulses"],
        total_steps=total_steps,
        gap_steps=derived["gap_steps"],
    )

    original_dynamics = network.dynamics.dynamics
    stimulated_fn = _make_stimulated_dynamics(
        original_dynamics=original_dynamics,
        stimulation_jax=jnp.asarray(stim_array, dtype=jnp.float32),
        cfg=cfg,
        stimulus_mode="true_p_t",
    )
    network.dynamics.dynamics = types.MethodType(stimulated_fn, network.dynamics)
    try:
        bundle.apply_to_network(network)
        compiled, sim_state = prepare(
            network, BoundedSolver(Heun(), low=0.0, high=1.0),
            t1=int(total_steps * dt), dt=dt,
        )
        sim_state = bundle.apply_to_state(sim_state)
        result = jax.block_until_ready(compiled(sim_state))
    finally:
        network.dynamics.dynamics = original_dynamics

    rec = np.asarray(result.data, dtype=np.float32)
    if rec.ndim != 3 or rec.shape[1] < 2:
        raise RuntimeError(f"state 레이아웃이 예상과 다르다: shape={rec.shape}")

    settle = int(round(settle_ms / dt))
    raw = rec[:, 0, node_index]
    raw_pre = raw[pre_steps // 2:pre_steps]                       # pre 앞 절반 = transient 버림
    raw_during = raw[pre_steps + settle:pre_steps + stim_steps]   # 자극 개시 transient 버림

    # 진단 통계는 원본에서 낸다. H_e 복원만 (1-S_e) 0 나눗셈 방지로 clip.
    # (clip 한 값으로 S_e_during_max 를 보고하면 솔버 경계 도달이 제 clip 에 가려진다)
    s_pre = np.clip(raw_pre, 0.0, 0.999)
    s_during = np.clip(raw_during, 0.0, 0.999)
    at_bound = float((raw_during >= 1.0).mean())

    return {
        "H_e_pre_hz": float(_h_e_from_s_e(s_pre, tau_e, gamma_e).mean()),
        "H_e_during_hz": float(_h_e_from_s_e(s_during, tau_e, gamma_e).mean()),
        "S_e_pre": float(raw_pre.mean()),
        "S_e_during": float(raw_during.mean()),
        "S_e_during_max": float(raw_during.max()),
        # >0 이면 적분기가 경계를 넘어 잘렸다는 뜻 → 그 amp 의 H_e 는 신뢰 불가
        "frac_at_bound": at_bound,
    }


def main():
    ap = argparse.ArgumentParser(description="DBS amp(nA) ↔ 임상 mA 캘리브레이션")
    ap.add_argument("--subject-idx", dest="subject_idx", type=int, default=4)
    ap.add_argument("--noise-level", dest="noise_level", type=float, default=0.02)
    ap.add_argument("--target", default="STN_L")
    ap.add_argument("--amps", default="0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.6,1.0")
    ap.add_argument("--freq", type=float, default=130.0)
    ap.add_argument("--pre-ms", dest="pre_ms", type=float, default=4000.0)
    ap.add_argument("--stim-ms", dest="stim_ms", type=float, default=4000.0)
    ap.add_argument("--sigma", type=float, default=0.2, help="조직 전도도 [S/m]")
    ap.add_argument("--e-th", dest="e_th", type=float, default=150.0, help="축삭 역치 [V/m]")
    ap.add_argument("--replot", action="store_true",
                    help="시뮬 없이 저장된 CSV 로 그림만 다시 그린다(GPU 불필요)")
    ap.add_argument("--tag", default="",
                    help="출력 폴더 접미사. 안 주면 같은 타깃 재실행이 이전 CSV 를 덮어쓴다")
    a = ap.parse_args()

    amps = [float(x) for x in a.amps.replace(" ", "").split(",") if x]

    p = M.prepare_pd_data(a.subject_idx, a.noise_level)
    cfg = M.make_config(p, a.subject_idx)
    M._FIG["dir"] = _pl.Path(p["out_dir"]) / "figures"
    M._FIG["dir"].mkdir(parents=True, exist_ok=True)

    if a.target not in cfg.dbs_target_regions:
        sys.exit(f"[calib] 타깃 없음: {a.target}. 가능: {list(cfg.dbs_target_regions)}")
    node_index = cfg.dbs_target_regions[a.target]

    out_dir = os.path.join(p["out_dir"], "dbs_amp_calib", a.target + a.tag)
    if a.replot:
        import pandas as pd
        rows = pd.read_csv(os.path.join(out_dir, "amp_response.csv")).to_dict("records")
        clin = pd.read_csv(os.path.join(out_dir, "clinical_map.csv")).to_dict("records")
        h_base = float(np.mean([r["H_e_pre_hz"] for r in rows]))
        _report_and_plot(rows, clin, h_base, _compute_derived_parameters(cfg),
                         out_dir, a, p["sub_num"])
        return

    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    network, _, _, _ = build_network(cfg, data)

    gfiles = sorted(glob.glob(os.path.join(data["cache_dir"], "grad_*.pkl")),
                    key=os.path.getmtime)
    if not gfiles:
        sys.exit(f"[calib] grad 캐시 없음: {data['cache_dir']} → Part3 먼저 완료 필요")
    with open(gfiles[-1], "rb") as fh:
        grad = pickle.load(fh)
    bundle = StateBundle.from_dict(grad["bundle"])
    corr = grad["bundle"].get("metadata", {}).get("post_grad_fc_corr")
    print(f"[calib] grad 로드: {os.path.basename(gfiles[-1])}  post_grad_corr={corr}")

    cfg.dbs_pre_stimulation_duration_ms = a.pre_ms
    cfg.dbs_stimulation_duration_ms = a.stim_ms
    cfg.dbs_stimulation_frequency_hz = a.freq
    derived = _compute_derived_parameters(cfg)

    # build_network 과 동일한 fallback 경로로 실제 사용된 값을 읽는다
    _dp = ReducedWongWangEIB.DEFAULT_PARAMS
    tau_e = float(getattr(cfg, "rww_tau_e", _dp["tau_e"]))
    gamma_e = float(getattr(cfg, "rww_gamma_e", _dp["gamma_e"]))
    print(f"[calib] sub={p['sub_num']} target={a.target}(node {node_index})  "
          f"freq_actual={derived['freq_actual_hz']:.1f}Hz  pulses={derived['n_pulses']}  "
          f"tau_e={tau_e} gamma_e={gamma_e:.6f}")

    rows = []
    for i, amp in enumerate(amps, 1):
        m = measure_h_e(network, bundle, cfg, data, node_index, amp, derived,
                        tau_e=tau_e, gamma_e=gamma_e)
        m["amp_nA"] = amp
        rows.append(m)
        print(f"[{i}/{len(amps)}] amp={amp:<5g}  H_e pre={m['H_e_pre_hz']:7.2f}  "
              f"during={m['H_e_during_hz']:7.2f} Hz   "
              f"S_e {m['S_e_pre']:.3f}→{m['S_e_during']:.3f} (max {m['S_e_during_max']:.3f})")
        jax.clear_caches()

    h_base = float(np.mean([r["H_e_pre_hz"] for r in rows]))
    clin = clinical_targets(h_base, derived["freq_actual_hz"], sigma=a.sigma, e_th=a.e_th)

    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "amp_response.csv"), index=False)

    _report_and_plot(rows, clin, h_base, derived, out_dir, a, p["sub_num"])


def _report_and_plot(rows, clin, h_base, derived, out_dir, a, sub_num):
    """amp→H_e 곡선을 뒤집어 임상 mA 대응 amp 를 역산하고, 표·그림을 저장한다."""
    import pandas as pd

    amp_arr = np.array([r["amp_nA"] for r in rows])
    he_arr = np.array([r["H_e_during_hz"] for r in rows])
    order = np.argsort(he_arr)   # 단조 구간에서만 유효
    for c in clin:
        t = c["H_e_target_hz"]
        c["amp_nA_equiv"] = (float(np.interp(t, he_arr[order], amp_arr[order]))
                             if he_arr.min() <= t <= he_arr.max() else np.nan)
    pd.DataFrame(clin).to_csv(os.path.join(out_dir, "clinical_map.csv"), index=False)

    print(f"\n[calib] baseline H_e = {h_base:.2f} Hz   entrain = {derived['freq_actual_hz']:.1f} Hz")
    print(f"{'I(mA)':>6} {'r_VTA(mm)':>10} {'f_recruit':>10} {'H_e목표(Hz)':>12} {'amp(nA)':>9}")
    for c in clin:
        eq = "  범위밖" if np.isnan(c["amp_nA_equiv"]) else f"{c['amp_nA_equiv']:9.3f}"
        print(f"{c['I_mA']:6.1f} {c['r_VTA_mm']:10.2f} {c['f_recruit']:10.3f} "
              f"{c['H_e_target_hz']:12.1f} {eq}")

    # ponytail: 라벨 전부 영문 — 시스템에 한글 폰트가 없어 matplotlib 이 두부로 렌더한다
    fig, ax = plt.subplots(figsize=(7, 5))

    # 높은 발화율 자체는 정상이다(125Hz entrain 시 자연 포화 S_e≈0.889). 문제는 S_e 가
    # 정확히 1.0 에 닿는 것 — 유한 발화율로는 도달 불가(H*g*t/(1+H*g*t)<1)이므로 적분기
    # 오버슈트를 BoundedSolver 가 자른 것이다. 그 구간의 H_e 는 (1-S_e)→0 라 무의미.
    at_bound = np.array([_get(r, "frac_at_bound",
                              1.0 if _get(r, "S_e_during_max", 0.0) >= 1.0 else 0.0)
                         for r in rows])
    clipped = amp_arr[at_bound > 0]
    if clipped.size:
        ax.axvspan(clipped.min(), amp_arr.max(), color="tab:orange", alpha=0.12, zorder=0)
        # y 는 axes fraction — 데이터 그리기 전이라 get_ylim() 은 아직 기본값이다
        ax.annotate("$S_e$ hits solver bound (1.0)\nintegrator overshoot, not physiology",
                    (clipped.min(), 0.55), xycoords=("data", "axes fraction"),
                    fontsize=8, color="tab:orange", va="top", ha="left")

    ax.plot(amp_arr, he_arr, "o-", color="black", label="simulated: mean $H_e$ during stim")
    ax.axhline(h_base, ls=":", color="gray", label=f"baseline {h_base:.1f} Hz")
    for c in clin:
        if not np.isnan(c["amp_nA_equiv"]):
            ok = not clipped.size or c["amp_nA_equiv"] < clipped.min()
            ax.axhline(c["H_e_target_hz"], ls="--", lw=0.6,
                       color="tab:red" if ok else "tab:gray", alpha=0.5)
            ax.annotate(f"{c['I_mA']:g} mA" + ("" if ok else " (clipped)"),
                        (amp_arr.min(), c["H_e_target_hz"]), fontsize=8,
                        color="tab:red" if ok else "tab:gray", va="bottom")
    ax.set_xlabel("dbs_pulse_amplitude [nA]")
    ax.set_ylabel("mean $H_e$ during stim [Hz]")
    ax.set_title(f"sub {sub_num} / {a.target} — amp vs clinical mA "
                 f"($\\sigma$={a.sigma}, $E_{{th}}$={a.e_th} V/m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calib.png"), dpi=150)
    print(f"\n[calib] 저장 → {out_dir}")


def _self_check():
    """H_e 복원식 왕복 검산: 정상상태 S_e = H*g*t/(1+H*g*t) 를 넣으면 H 가 돌아와야 한다."""
    tau_e, gamma_e = 100.0, 0.641 / 1000
    for h in (1.0, 5.0, 20.0, 130.0):
        s = h * gamma_e * tau_e / (1.0 + h * gamma_e * tau_e)
        back = _h_e_from_s_e(np.array([s]), tau_e, gamma_e)[0]
        assert abs(back - h) < 1e-3 * max(h, 1.0), f"복원 실패: {h} → {back}"


if __name__ == "__main__":
    _self_check()
    main()
