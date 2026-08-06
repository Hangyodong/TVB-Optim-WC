"""
pipeline_contracts.py

Stage 3 — 정식 파이프라인 계약

핵심 목표
---------
1) ParamSet / StateBundle로 Part 1 → Part 2 → Part 3 → Part 4 상태 전달을 통일한다.
2) BOLD monitor history와 FC 계산용 BOLD sample window를 분리 저장한다.
3) noise / delay 관련 내부 상태를 최대한 bundle에 보존해 stage 경계 reset을 줄인다.
4) legacy notebook 코드도 계속 동작하도록 TVB-state 호환 뷰를 제공한다.
"""
from __future__ import annotations

import hashlib
import types
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tvboptim.experimental.network_dynamics import prepare
from tvboptim.observations.tvb_monitors.bold import Bold, LotkaVolterraHRFKernel


# ────────────────────────────────────────────────────────────────────────────
# ParamSet
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamSet:
    """
    c_ei / wLRE / wFFI를 하나의 단위로 관리한다.
    """
    c_ei: np.ndarray
    wLRE: np.ndarray
    wFFI: np.ndarray
    c_ei_frozen: bool = False

    def freeze_c_ei(self) -> "ParamSet":
        return ParamSet(
            c_ei=self.c_ei.copy(),
            wLRE=self.wLRE.copy(),
            wFFI=self.wFFI.copy(),
            c_ei_frozen=True,
        )

    def update_c_ei(self, delta: np.ndarray) -> "ParamSet":
        if self.c_ei_frozen:
            return self
        new_c_ei = np.clip(
            np.asarray(self.c_ei, dtype=np.float32) + np.asarray(delta, dtype=np.float32),
            0.0, 20.0,
        ).astype(np.float32)
        return ParamSet(
            c_ei=new_c_ei,
            wLRE=np.asarray(self.wLRE, dtype=np.float32),
            wFFI=np.asarray(self.wFFI, dtype=np.float32),
            c_ei_frozen=False,
        )

    def update_weights(
        self,
        new_wLRE: np.ndarray,
        new_wFFI: np.ndarray,
        sc_mask: np.ndarray,
        w_max: float,
    ) -> "ParamSet":
        return ParamSet(
            c_ei=np.asarray(self.c_ei, dtype=np.float32),
            wLRE=_clean_weight_matrix(new_wLRE, sc_mask, w_max),
            wFFI=_clean_weight_matrix(new_wFFI, sc_mask, w_max),
            c_ei_frozen=self.c_ei_frozen,
        )

    def sanitize(self, sc_mask: np.ndarray, w_max: float) -> "ParamSet":
        return ParamSet(
            c_ei=_clean_c_ei(self.c_ei),
            wLRE=_clean_weight_matrix(self.wLRE, sc_mask, w_max),
            wFFI=_clean_weight_matrix(self.wFFI, sc_mask, w_max),
            c_ei_frozen=self.c_ei_frozen,
        )

    def to_jax(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            jnp.asarray(self.c_ei),
            jnp.asarray(self.wLRE),
            jnp.asarray(self.wFFI),
        )

    def to_numpy_dict(self) -> dict:
        return {
            "c_ei": np.asarray(self.c_ei, dtype=np.float32),
            "wLRE": np.asarray(self.wLRE, dtype=np.float32),
            "wFFI": np.asarray(self.wFFI, dtype=np.float32),
            "c_ei_frozen": bool(self.c_ei_frozen),
        }

    @classmethod
    def from_numpy_dict(cls, d: dict) -> "ParamSet":
        return cls(
            c_ei=np.asarray(d["c_ei"], dtype=np.float32),
            wLRE=np.asarray(d["wLRE"], dtype=np.float32),
            wFFI=np.asarray(d["wFFI"], dtype=np.float32),
            c_ei_frozen=bool(d.get("c_ei_frozen", False)),
        )

    @classmethod
    def default(cls, n_nodes: int, c_ei_init: float = 10.0) -> "ParamSet":
        ones = np.ones((n_nodes, n_nodes), dtype=np.float32)
        return cls(
            c_ei=np.full(n_nodes, c_ei_init, dtype=np.float32),
            wLRE=ones.copy(),
            wFFI=ones.copy(),
            c_ei_frozen=False,
        )


# ────────────────────────────────────────────────────────────────────────────
# StateBundle compatibility views
# ────────────────────────────────────────────────────────────────────────────

class _DynamicsView:
    def __init__(self, bundle: "StateBundle"):
        self._bundle = bundle

    @property
    def c_ei(self):
        return jnp.asarray(self._bundle.params.c_ei)


class _CouplingParamView:
    def __init__(self, bundle: "StateBundle"):
        self._bundle = bundle

    @property
    def wLRE(self):
        return jnp.asarray(self._bundle.params.wLRE)

    @property
    def wFFI(self):
        return jnp.asarray(self._bundle.params.wFFI)


class _CouplingView:
    def __init__(self, bundle: "StateBundle"):
        self.coupling = _CouplingParamView(bundle)


class _InitialStateView:
    def __init__(self, bundle: "StateBundle"):
        self._bundle = bundle

    @property
    def dynamics(self):
        return jnp.asarray(self._bundle.init_dynamics)


# ────────────────────────────────────────────────────────────────────────────
# StateBundle
# ────────────────────────────────────────────────────────────────────────────

class StateBundle:
    """
    Stage 경계를 넘는 핵심 상태를 하나의 계약으로 묶는다.

    Stored state
    ------------
    params
        c_ei / wLRE / wFFI
    init_dynamics
        다음 stage 시작점의 neural endpoint
    bold_history
        BOLD hemodynamic state를 재개하기 위한 monitor history
    bold_window
        FC 계산을 위해 직전 window에서 실제로 emit된 BOLD sample
    internal_state
        noise_samples를 포함한 tvb_state._internal 배열들
    delay_history
        delay-aware graph 복원을 위한 network history snapshot
    metadata
        rng_key 등 stage 간 보존해야 하는 부가 상태
    """

    def __init__(
        self,
        params: ParamSet,
        init_dynamics: np.ndarray,
        bold_history: Optional[np.ndarray],
        bold_window: Optional[np.ndarray] = None,
        internal_state: Optional[Dict[str, np.ndarray]] = None,
        delay_history: Optional[np.ndarray] = None,
        stage: str = "",
        metadata: Optional[dict] = None,
    ):
        self._params = params
        self._init_dynamics = _to_numpy_or_none(init_dynamics)
        self._bold_history = _to_numpy_or_none(bold_history)
        self._bold_window = _to_numpy_or_none(bold_window)
        self._internal_state = _normalize_internal_state(internal_state)
        self._delay_history = _to_numpy_or_none(delay_history)
        self._stage = stage
        self._metadata = dict(metadata or {})

    # ── 읽기 전용 접근자 ──────────────────────────────────────
    @property
    def params(self) -> ParamSet:
        return self._params

    @property
    def init_dynamics(self) -> np.ndarray:
        return self._init_dynamics

    @property
    def bold_history(self) -> Optional[np.ndarray]:
        return self._bold_history

    @property
    def bold_window(self) -> Optional[np.ndarray]:
        return self._bold_window

    @property
    def internal_state(self) -> Optional[Dict[str, np.ndarray]]:
        return self._internal_state

    @property
    def noise_state(self) -> Optional[np.ndarray]:
        if self._internal_state is None:
            return None
        return self._internal_state.get("noise_samples")

    @property
    def delay_history(self) -> Optional[np.ndarray]:
        return self._delay_history

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    # ── legacy notebook compatibility ─────────────────────────
    @property
    def dynamics(self):
        return _DynamicsView(self)

    @property
    def coupling(self):
        return _CouplingView(self)

    @property
    def initial_state(self):
        return _InitialStateView(self)

    # ── 불변 갱신 ─────────────────────────────────────────────
    def advance(
        self,
        new_params: Optional[ParamSet] = None,
        new_init_dynamics: Optional[np.ndarray] = None,
        new_bold_history: Optional[np.ndarray] = None,
        new_bold_window: Optional[np.ndarray] = None,
        new_internal_state: Optional[Dict[str, np.ndarray]] = None,
        new_delay_history: Optional[np.ndarray] = None,
        next_stage: Optional[str] = None,
        metadata_update: Optional[dict] = None,
    ) -> "StateBundle":
        new_metadata = dict(self._metadata)
        if metadata_update:
            new_metadata.update(metadata_update)

        return StateBundle(
            params=new_params if new_params is not None else self._params,
            init_dynamics=(
                self._init_dynamics if new_init_dynamics is None else new_init_dynamics
            ),
            bold_history=(
                self._bold_history if new_bold_history is None else new_bold_history
            ),
            bold_window=(
                self._bold_window if new_bold_window is None else new_bold_window
            ),
            internal_state=(
                self._internal_state if new_internal_state is None else new_internal_state
            ),
            delay_history=(
                self._delay_history if new_delay_history is None else new_delay_history
            ),
            stage=self._stage if next_stage is None else next_stage,
            metadata=new_metadata,
        )

    def with_params(self, new_params: ParamSet, next_stage: Optional[str] = None) -> "StateBundle":
        return self.advance(new_params=new_params, next_stage=next_stage)

    def with_metadata(self, metadata_update: dict) -> "StateBundle":
        return self.advance(metadata_update=metadata_update)

    # ── prepare + state 주입 ─────────────────────────────────
    def apply_to_network(self, network) -> None:
        restore_network_delay_history(network, self._delay_history)

    def apply_to_state(self, tvb_state):
        c_ei_j, wLRE_j, wFFI_j = self._params.to_jax()
        tvb_state.dynamics.c_ei = c_ei_j
        tvb_state.coupling.coupling.wLRE = wLRE_j
        tvb_state.coupling.coupling.wFFI = wFFI_j
        tvb_state.initial_state.dynamics = jnp.asarray(self._init_dynamics)
        restore_internal_state(tvb_state, self._internal_state)
        return tvb_state

    def to_tvb_state(self, network, solver, t1: int, dt: float):
        self.apply_to_network(network)
        compiled_model, tvb_state = prepare(network, solver, t1=t1, dt=dt)
        tvb_state = self.apply_to_state(tvb_state)
        return compiled_model, tvb_state

    def build_bold_monitor(self, cfg) -> Bold:
        # 커널/params는 make_bold_monitor 단일 소스에서 (build_network warmup과 동일 커널).
        monitor = make_bold_monitor(cfg, history=None)
        if self._bold_history is None:
            return monitor
        try:
            monitor = eqx.tree_at(
                lambda m: m.history,
                monitor,
                jnp.asarray(self._bold_history),
            )
        except Exception:
            pass
        return monitor

    def get_fc_seed_window(self, n_samples: int, n_nodes: int) -> np.ndarray:
        if self._bold_window is not None:
            window = np.asarray(self._bold_window, dtype=np.float32)
        elif self._bold_history is not None:
            hist = np.asarray(self._bold_history, dtype=np.float32)
            if hist.ndim == 3:
                window = hist[:, 0, :]
            else:
                window = hist
        else:
            window = np.zeros((0, n_nodes), dtype=np.float32)

        if window.ndim == 3:
            window = window[:, 0, :]

        if window.shape[0] >= n_samples:
            return window[-n_samples:].astype(np.float32)

        padded = np.zeros((n_samples, n_nodes), dtype=np.float32)
        if window.size > 0:
            padded[-window.shape[0]:] = window.astype(np.float32)
        return padded

    # ── 직렬화 / fingerprint ──────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "params": self._params.to_numpy_dict(),
            "init_dynamics": None if self._init_dynamics is None else self._init_dynamics.copy(),
            "bold_history": None if self._bold_history is None else self._bold_history.copy(),
            "bold_window": None if self._bold_window is None else self._bold_window.copy(),
            "internal_state": _normalize_internal_state(self._internal_state),
            "delay_history": None if self._delay_history is None else self._delay_history.copy(),
            "stage": self._stage,
            "metadata": _to_numpy_metadata(self._metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateBundle":
        return cls(
            params=ParamSet.from_numpy_dict(d["params"]),
            init_dynamics=d.get("init_dynamics"),
            bold_history=d.get("bold_history"),
            bold_window=d.get("bold_window"),
            internal_state=d.get("internal_state"),
            delay_history=d.get("delay_history"),
            stage=d.get("stage", ""),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_warmup(
        cls,
        warmup_result,
        bold_monitor_template,
        initial_params: ParamSet,
        internal_state: Optional[Dict[str, np.ndarray]] = None,
        delay_history: Optional[np.ndarray] = None,
        stage: str = "warmup",
        metadata: Optional[dict] = None,
    ) -> "StateBundle":
        init_dyn = np.asarray(warmup_result.data[-1], dtype=np.float32)
        bold_history = None
        if hasattr(bold_monitor_template, "history"):
            try:
                bold_history = np.asarray(bold_monitor_template.history, dtype=np.float32)
            except Exception:
                bold_history = None
        return cls(
            params=initial_params,
            init_dynamics=init_dyn,
            bold_history=bold_history,
            bold_window=None,
            internal_state=internal_state,
            delay_history=delay_history,
            stage=stage,
            metadata=metadata,
        )

    def fingerprint(self) -> str:
        pieces = [
            np.asarray(self._params.c_ei, dtype=np.float32),
            np.asarray(self._params.wLRE, dtype=np.float32),
            np.asarray(self._params.wFFI, dtype=np.float32),
            np.asarray(self._init_dynamics, dtype=np.float32),
        ]
        if self._bold_window is not None:
            pieces.append(np.asarray(self._bold_window, dtype=np.float32))
        if self._bold_history is not None:
            hist = np.asarray(self._bold_history, dtype=np.float32)
            pieces.append(hist[: min(8, hist.shape[0])])
        if self._internal_state:
            for key in sorted(self._internal_state):
                pieces.append(np.asarray(self._internal_state[key], dtype=np.float32).ravel()[:64])
        if self._delay_history is not None:
            delay = np.asarray(self._delay_history, dtype=np.float32)
            pieces.append(delay.ravel()[:128])
        stage_bytes = self._stage.encode("utf-8")
        return _fingerprint_arrays(*pieces, extra_bytes=stage_bytes)

    def __repr__(self) -> str:
        return (
            f"StateBundle(stage={self._stage!r}, "
            f"c_ei_frozen={self._params.c_ei_frozen}, "
            f"init_dyn_shape={None if self._init_dynamics is None else self._init_dynamics.shape})"
        )


# ────────────────────────────────────────────────────────────────────────────
# 공용 헬퍼
# ────────────────────────────────────────────────────────────────────────────

def _to_numpy_or_none(array_like):
    if array_like is None:
        return None
    return np.asarray(array_like, dtype=np.float32)


def _clean_c_ei(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=6.0, posinf=20.0, neginf=0.0)
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 20.0).astype(np.float32)


def _clean_weight_matrix(values: np.ndarray, sc_mask: np.ndarray, w_max: float) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=w_max, neginf=0.0)
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, w_max)
    values = values * np.asarray(sc_mask, dtype=np.float32)
    return (0.5 * (values + values.T)).astype(np.float32)


def _normalize_internal_state(internal_state: Optional[dict]) -> Optional[Dict[str, np.ndarray]]:
    if internal_state is None:
        return None
    normalized = {}
    for key, value in dict(internal_state).items():
        if value is None:
            continue
        try:
            normalized[key] = np.asarray(value)
        except Exception:
            continue
    return normalized or None


def _to_numpy_metadata(metadata: Optional[dict]) -> dict:
    if metadata is None:
        return {}
    out = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
            continue
        try:
            out[key] = np.asarray(value) if key.endswith("_key") else value
        except Exception:
            out[key] = value
    return out


def _fingerprint_arrays(*arrays, extra_bytes: bytes = b"") -> str:
    sha = hashlib.sha1()
    for arr in arrays:
        try:
            arr_np = np.asarray(arr, dtype=np.float32)
            sha.update(arr_np.tobytes())
        except Exception:
            continue
    sha.update(extra_bytes)
    return sha.hexdigest()[:12]


def capture_internal_state(tvb_state) -> Optional[Dict[str, np.ndarray]]:
    internal = getattr(tvb_state, "_internal", None)
    if internal is None:
        return None

    captured = {}
    for attr_name in dir(internal):
        if attr_name.startswith("_"):
            continue
        try:
            attr_value = getattr(internal, attr_name)
        except Exception:
            continue
        if callable(attr_value):
            continue
        try:
            attr_array = np.asarray(attr_value)
        except Exception:
            continue
        if attr_array.dtype == object:
            continue
        captured[attr_name] = attr_array.copy()
    return captured or None


def restore_internal_state(tvb_state, internal_state: Optional[Dict[str, np.ndarray]]) -> None:
    if internal_state is None:
        return
    internal = getattr(tvb_state, "_internal", None)
    if internal is None:
        return
    for attr_name, attr_value in internal_state.items():
        if not hasattr(internal, attr_name):
            continue
        try:
            setattr(internal, attr_name, jnp.asarray(attr_value))
        except Exception:
            try:
                setattr(internal, attr_name, np.asarray(attr_value))
            except Exception:
                continue


def advance_internal_state(tvb_state, metadata: Optional[dict] = None) -> Tuple[Optional[Dict[str, np.ndarray]], dict]:
    metadata_out = dict(metadata or {})
    internal = getattr(tvb_state, "_internal", None)
    if internal is None or not hasattr(internal, "noise_samples"):
        return capture_internal_state(tvb_state), metadata_out

    try:
        key = metadata_out.get("rng_key", None)
        if key is None:
            rng_key = jax.random.PRNGKey(int(metadata_out.get("rng_seed", 42)))
        else:
            rng_key = jnp.asarray(key, dtype=jnp.uint32)
        rng_key, subkey = jax.random.split(rng_key)
        new_noise = jax.random.normal(
            subkey,
            jnp.asarray(internal.noise_samples).shape,
            dtype=jnp.asarray(internal.noise_samples).dtype,
        )
        internal.noise_samples = new_noise
        metadata_out["rng_key"] = np.asarray(rng_key, dtype=np.uint32)
    except Exception:
        pass
    return capture_internal_state(tvb_state), metadata_out


def eval_fc_multiseed(eval_model, eval_state, eval_monitor, compute_fc_fn, skip_tr, base_seed, n_seeds):
    """FC를 n_seeds개 noise draw로 평균. n_seeds<=1 이면 단일 draw(현행과 동일). return (fc_mean, last_sim_result)."""
    import numpy as _np
    if n_seeds is None or int(n_seeds) <= 1:
        res = eval_model(eval_state)
        return compute_fc_fn(eval_monitor(res), skip_tr), res
    fcs = []
    last = None
    for i in range(int(n_seeds)):
        internal = eval_state._internal
        _, subkey = jax.random.split(jax.random.PRNGKey(int(base_seed) + i))
        internal.noise_samples = jax.random.normal(
            subkey, jnp.asarray(internal.noise_samples).shape, jnp.asarray(internal.noise_samples).dtype)
        last = eval_model(eval_state)
        fcs.append(_np.asarray(compute_fc_fn(eval_monitor(last), skip_tr), dtype=_np.float32))
    return _np.mean(_np.stack(fcs, axis=0), axis=0), last


def make_bold_monitor(cfg, history=None) -> Bold:
    """cfg HRF 파라미터로 Bold 모니터 생성 — warmup(build_network)과 각 Part가 동일
    커널을 쓰도록 강제하는 단일 소스.

    두 곳이 서로 다른 커널(예: 기본 20s vs cfg 32s)로 만들어지면 warmup history 길이
    (kernel_samples = ceil(duration/downsample_period))가 파이프라인 모니터 기대치와
    어긋나, mode='valid' fftconvolve가 예외 없이 입출력을 swap해 FC 측정이 조용히 깨진다.

    history: NativeSolution이면 Bold.__init__이 downsample→last kernel_samples로 처리.
             None이면 빈 warm-up(호출부가 tree_at으로 사전처리된 history 주입).
    """
    kernel = LotkaVolterraHRFKernel(
        tau_s    = getattr(cfg, "bold_hrf_tau_s",    0.8),
        tau_f    = getattr(cfg, "bold_hrf_tau_f",    0.4),
        scaling  = getattr(cfg, "bold_hrf_scaling",  1.0 / 3.0),
        duration = getattr(cfg, "bold_hrf_duration_ms", 20_000.0),
    )
    return Bold(
        period=cfg.bold_repetition_time_ms,
        downsample_period=4.0,
        voi=0,
        history=history,
        k_1    = getattr(cfg, "bold_hrf_k1", 5.6),
        V_0    = getattr(cfg, "bold_hrf_V0", 0.02),
        kernel = kernel,
    )


def update_bold_history(monitor, simulation_result):
    if monitor is None or not hasattr(monitor, "history"):
        return monitor
    history_accessor = lambda tree: tree.history
    history = monitor.history
    if history is None:
        return monitor

    history_len = int(history.shape[0])
    new_data = simulation_result.data[:, 0:1, :]
    n_new_steps = int(new_data.shape[0])
    n_insert = min(history_len, n_new_steps)

    if n_insert <= 0:
        return monitor

    new_history = jnp.roll(history, -n_insert, axis=0)
    new_history = new_history.at[-n_insert:, :, :].set(new_data[-n_insert:])
    return eqx.tree_at(history_accessor, monitor, new_history)


def extract_bold_window(bold_output) -> np.ndarray:
    ys = np.asarray(bold_output.ys[:, 0, :] if bold_output.ys.ndim == 3 else bold_output.ys, dtype=np.float32)
    return ys


# ── Edge-weighted FC loss (subcortex emphasis) ───────────────────
# W는 off-diagonal edge 가중 행렬(diag 0). block weight가 모두 1.0이면
# W == (1 - eye)가 되어 아래 함수들은 기존 비가중 corr/rmse와 정확히 동일.
def weighted_corr_loss(predicted: jnp.ndarray, target: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    """1 - (W-가중 Pearson corr). W=(1-eye)일 때 비가중 corr loss와 동치."""
    n = jnp.maximum(jnp.sum(W), 1.0)
    p_mean = jnp.sum(W * predicted) / n
    t_mean = jnp.sum(W * target) / n
    pc = predicted - p_mean
    tc = target - t_mean
    num = jnp.sum(W * pc * tc)
    den = jnp.sqrt(
        jnp.maximum(jnp.sum(W * pc ** 2), 1e-10)
        * jnp.maximum(jnp.sum(W * tc ** 2), 1e-10)
    )
    return 1.0 - num / den


def weighted_rmse_loss(predicted: jnp.ndarray, target: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    """sqrt(W-가중 평균제곱오차). W=(1-eye)일 때 비가중 rmse loss와 동치."""
    n = jnp.maximum(jnp.sum(W), 1.0)
    diff = predicted - target
    return jnp.sqrt(jnp.sum(W * diff ** 2) / n)


# ── Block-split corr loss (cc / cross / sub-sub) ─────────────────
# whole-matrix corr는 edge수 많은 cortex-cortex가 지배 → subcortex fit 안됨.
# 블록별 corr을 따로 계산해 edge수 무관 동등 가중 → 균형. RMSE는 분할 안 함.
def prepare_block_corr_terms(block_masks: dict, weights: dict) -> list:
    """비어있지 않은 블록만 골라 (jnp mask, 정규화 weight) 리스트 반환.
    subcortex 없는 atlas는 cross/ss 비어 cc(=full off-diag)만 남아 whole corr와 동치.
    한 번만 호출(static) → loss closure는 jit/eager 양쪽에서 안전."""
    active = []
    for key in ("cc", "cross", "subsub"):
        m = block_masks.get(key)
        w = float(weights.get(key, 0.0))
        if m is None or w <= 0.0:
            continue
        if float(np.sum(np.asarray(m))) <= 0.0:
            continue
        active.append((jnp.asarray(m, dtype=jnp.float32), w))
    total_w = sum(w for _, w in active)
    if total_w <= 0.0:
        raise ValueError("prepare_block_corr_terms: 활성 블록 없음 (모든 mask/weight=0)")
    return [(m, w / total_w) for m, w in active]


def block_corr_loss(predicted: jnp.ndarray, target: jnp.ndarray, terms: list) -> jnp.ndarray:
    """blockwise (1 - W-가중 Pearson corr)의 가중합. terms=prepare_block_corr_terms(...)."""
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for m, w in terms:
        total = total + jnp.float32(w) * weighted_corr_loss(predicted, target, m)
    return total


def block_rmse_loss(predicted: jnp.ndarray, target: jnp.ndarray, terms: list) -> jnp.ndarray:
    """blockwise RMSE의 가중합. terms=prepare_block_corr_terms(...) 재사용(cc/cross/ss).
    plain RMSE는 edge수 많은 cortex-cortex가 지배 → block_corr와 동일 동기로 블록별
    RMSE를 edge수 무관 동등 가중. subcortex 없는 atlas는 cc만 활성 → plain off-diag RMSE와 동치.
    각 블록 RMSE=sqrt(블록 edge 평균제곱오차), terms의 정규화 weight로 합산."""
    diff2 = (predicted - target) ** 2
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for m, w in terms:
        n = jnp.maximum(jnp.sum(m), 1.0)
        total = total + jnp.float32(w) * jnp.sqrt(jnp.sum(m * diff2) / n)
    return total


def nodewise_corr_loss(
    predicted: jnp.ndarray, target: jnp.ndarray, W: jnp.ndarray, node_w: jnp.ndarray
) -> jnp.ndarray:
    """행별 W-가중 corr → node_w 가중 평균한 (1 - corr) loss.
    W=(1-eye), node_w 균일이면 기존(대각 제외, 노드 균일 평균)과 동치.
    part2 선택 score와 part3 gradient loss가 동일 nodewise 항을 쓰도록 공유."""
    n = predicted.shape[0]

    def _row_corr(i):
        w = W[i]
        x = predicted[i]
        y = target[i]
        n_eff = jnp.maximum(jnp.sum(w), 1.0)
        xm = x - jnp.sum(w * x) / n_eff
        ym = y - jnp.sum(w * y) / n_eff
        num = jnp.sum(w * xm * ym)
        den = jnp.sqrt(
            jnp.maximum(jnp.sum(w * xm ** 2), 1e-10)
            * jnp.maximum(jnp.sum(w * ym ** 2), 1e-10)
        )
        return num / den

    row_corrs = jax.vmap(_row_corr)(jnp.arange(n))
    return 1.0 - jnp.sum(node_w * row_corrs) / jnp.maximum(jnp.sum(node_w), 1.0)


# ── Block-wise FC corr (진단/플롯용, 순수 numpy) ──────────────────
def _flat_pearson(p_block: np.ndarray, t_block: np.ndarray) -> float:
    pf = np.asarray(p_block, dtype=np.float64).ravel()
    tf = np.asarray(t_block, dtype=np.float64).ravel()
    m = np.isfinite(pf) & np.isfinite(tf)
    if int(m.sum()) < 2:
        return float("nan")
    pf, tf = pf[m], tf[m]
    if pf.std() < 1e-8 or tf.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(pf, tf)[0, 1])


def compute_block_corrs(fc_pred, fc_target, cortex_idx, subcortex_idx) -> Dict[str, float]:
    """full / cortex-cortex / cross(ctx↔sub) / sub-sub 블록별 off-diag Pearson corr."""
    p = np.asarray(fc_pred, dtype=np.float64)
    t = np.asarray(fc_target, dtype=np.float64)
    n = p.shape[0]
    ci = np.asarray(cortex_idx, dtype=np.int64)
    si = np.asarray(subcortex_idx, dtype=np.int64)

    def _block(ri, cj, drop_diag):
        pb = p[np.ix_(ri, cj)].copy()
        tb = t[np.ix_(ri, cj)].copy()
        if drop_diag:
            d = np.eye(pb.shape[0], pb.shape[1], dtype=bool)
            pb[d] = np.nan
            tb[d] = np.nan
        return _flat_pearson(pb, tb)

    full = _block(np.arange(n), np.arange(n), drop_diag=True)
    return {
        "full":  full,
        "ctx":   _block(ci, ci, True) if ci.size else float("nan"),
        "cross": _block(ci, si, False) if (ci.size and si.size) else float("nan"),
        "sub":   _block(si, si, True) if si.size else float("nan"),
    }


def plot_block_corr_bars(ax, pre_fc, post_fc, fc_target, cortex_idx, subcortex_idx,
                         title="Block-wise FC corr") -> Tuple[dict, dict]:
    """ax에 pre/post 블록별 corr 그룹 막대. matplotlib는 호출부가 만든 ax로 주입."""
    pre = compute_block_corrs(pre_fc, fc_target, cortex_idx, subcortex_idx)
    post = compute_block_corrs(post_fc, fc_target, cortex_idx, subcortex_idx)
    keys = ["full", "ctx", "cross", "sub"]
    labels = ["full", "ctx-ctx", "cross", "sub-sub"]
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, [pre[k] for k in keys], w, label="pre", color="silver")
    ax.bar(x + w / 2, [post[k] for k in keys], w, label="post", color="steelblue")
    for xi, k in enumerate(keys):
        ax.text(xi + w / 2, post[k], f"{post[k]:.2f}", ha="center",
                va="bottom" if post[k] >= 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FC correlation")
    ax.set_title(title)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return pre, post


def _delay_tail_len(network) -> Optional[int]:
    """DDE 지연 버퍼는 마지막 max_delay 구간만 쓰인다(network.get_history). 게다가 bundle 의
    delay_history 는 cross-process restore 에 사용되지 않는다(update_history 는 solution.ys 를
    요구하는데 restore_network_delay_history 는 .data payload 를 넘겨 항상 no-op). 따라서
    warmup 전체 trajectory(수백 MB)를 저장할 필요 없이 max_delay 를 덮는 tail 만 보존한다.
    dt=1ms 가정(모든 러너 integration_dt_ms=1.0). max_delay 없거나 0(지연없음)이면 None=전체 유지."""
    md = getattr(network, "max_delay", None)
    if md is None:
        graph = getattr(network, "graph", None)
        md = getattr(graph, "max_delay", None)
    if md is None or md <= 0:
        return None
    return int(md) + 256   # generous margin


def capture_network_delay_history(network) -> Optional[np.ndarray]:
    tail = _delay_tail_len(network)
    for attr_name in ("history", "_history"):
        if not hasattr(network, attr_name):
            continue
        try:
            attr_value = getattr(network, attr_name)
        except Exception:
            continue
        if attr_value is None:
            continue
        try:
            raw = attr_value.data if hasattr(attr_value, "data") else attr_value
            if tail is not None:
                try:
                    raw = raw[-tail:]      # device-side slice → host 전송량·pkl 크기 최소화
                except Exception:
                    pass
            return np.asarray(raw, dtype=np.float32)
        except Exception:
            continue
    return None


def restore_network_delay_history(network, delay_history: Optional[np.ndarray]) -> None:
    if delay_history is None or not hasattr(network, "update_history"):
        return

    payloads = [
        types.SimpleNamespace(data=jnp.asarray(delay_history)),
        types.SimpleNamespace(data=np.asarray(delay_history)),
        np.asarray(delay_history),
    ]
    for payload in payloads:
        try:
            network.update_history(payload)
            return
        except Exception:
            continue


def sync_network_delay_history(network, simulation_result) -> Optional[np.ndarray]:
    if hasattr(network, "update_history"):
        try:
            network.update_history(simulation_result)
        except Exception:
            pass
    return capture_network_delay_history(network)


def resolve_pd_fit_indices(cfg, data: dict) -> np.ndarray:
    if getattr(cfg, "pd_fit_region_indices", None):
        return np.asarray(cfg.pd_fit_region_indices, dtype=np.int32)

    if getattr(cfg, "pd_fit_region_labels", None):
        label_to_index = {label: idx for idx, label in enumerate(data["region_labels"])}
        indices = [
            label_to_index[label]
            for label in cfg.pd_fit_region_labels
            if label in label_to_index
        ]
        if indices:
            return np.asarray(indices, dtype=np.int32)

    return np.arange(int(cfg.pd_fit_region_count), dtype=np.int32)


def build_bundle_from_legacy_state(
    state,
    cfg,
    data: dict,
    stage: str,
    bold_history: Optional[np.ndarray] = None,
    bold_window: Optional[np.ndarray] = None,
    internal_state: Optional[Dict[str, np.ndarray]] = None,
    delay_history: Optional[np.ndarray] = None,
    c_ei_frozen: bool = False,
    metadata: Optional[dict] = None,
) -> StateBundle:
    params = ParamSet(
        c_ei=np.asarray(state.dynamics.c_ei, dtype=np.float32),
        wLRE=np.asarray(state.coupling.coupling.wLRE, dtype=np.float32),
        wFFI=np.asarray(state.coupling.coupling.wFFI, dtype=np.float32),
        c_ei_frozen=c_ei_frozen,
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)

    init_dynamics = np.asarray(state.initial_state.dynamics, dtype=np.float32)
    if internal_state is None:
        internal_state = capture_internal_state(state)

    return StateBundle(
        params=params,
        init_dynamics=init_dynamics,
        bold_history=bold_history,
        bold_window=bold_window,
        internal_state=internal_state,
        delay_history=delay_history,
        stage=stage,
        metadata=metadata,
    )
