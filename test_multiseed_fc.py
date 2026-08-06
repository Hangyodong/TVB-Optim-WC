"""eval_fc_multiseed 경량 self-check (실제 파이프라인 미실행). PYTHONPATH=. python3 test_multiseed_fc.py"""
import numpy as np
import jax.numpy as jnp
from pipeline_contracts import eval_fc_multiseed


class _Internal:
    def __init__(self, noise):
        self.noise_samples = noise


class _State:
    def __init__(self, noise):
        self._internal = _Internal(noise)


calls = {"n": 0}
recorded = []


def model(s):
    # 결과 = 현재 noise_samples 평균 (draw마다 달라짐)
    calls["n"] += 1
    return {"val": float(jnp.asarray(s._internal.noise_samples).mean())}


def monitor(res):
    return res  # identity


def compute_fc(res, skip_tr):
    v = res["val"]
    fc = np.array([[v, 2 * v], [2 * v, v]], dtype=np.float32)
    recorded.append(fc.copy())
    return fc


# ── n_seeds=1: 모델 1회, 단일 draw FC 그대로 ──
state = _State(jnp.array([0.5, -0.5, 1.0], dtype=jnp.float32))
calls["n"] = 0
recorded.clear()
fc1, res1 = eval_fc_multiseed(model, state, monitor, compute_fc, 0, 42, 1)
assert calls["n"] == 1, calls["n"]
assert len(recorded) == 1
v0 = float(jnp.asarray(state._internal.noise_samples).mean())  # n_seeds=1은 noise 불변
assert np.allclose(fc1, np.array([[v0, 2 * v0], [2 * v0, v0]], dtype=np.float32))

# ── n_seeds=5: 서로 다른 5 draw 평균 ──
state = _State(jnp.array([0.5, -0.5, 1.0], dtype=jnp.float32))
calls["n"] = 0
recorded.clear()
fc5, res5 = eval_fc_multiseed(model, state, monitor, compute_fc, 0, 42, 5)
assert calls["n"] == 5, calls["n"]
assert len(recorded) == 5
mats = np.stack(recorded, axis=0)
assert not np.allclose(mats, mats[0]), "draw들이 서로 달라야 함"
assert np.allclose(fc5, mats.mean(axis=0), atol=1e-6)

print("OK")
