"""use_delay 스모크 체크: EIBDelayedCoupling 이 delay 를 실제로 쓰는지.

python3 test_use_delay.py  →  assert 통과하면 OK.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax.numpy as jnp
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph

from config import Config
from model import build_network, EIBDelayedCoupling, EIBLinearCoupling

N = 8
rng = np.random.default_rng(0)
w = rng.random((N, N)).astype(np.float64) * 0.1
np.fill_diagonal(w, 0.0)
lengths = rng.uniform(10.0, 90.0, (N, N))          # mm
labels = [f"r{i}" for i in range(N)]


def _data(speed):
    return {
        "n_nodes": N,
        "weights": w.astype(np.float32),
        "region_labels": labels,
        "graph": DenseDelayGraph(w, lengths / speed, region_labels=labels),
    }


def _cfg(use_delay):
    return Config(warmup_duration_ms=2_000, integration_dt_ms=1.0,
                  tract_conduction_speed=3.0, use_delay=use_delay,
                  bold_hrf_duration_ms=1_000.0, bold_repetition_time_ms=500.0)


for use_delay, cls in ((False, EIBLinearCoupling), (True, EIBDelayedCoupling)):
    net, _, _, res = build_network(_cfg(use_delay), _data(3.0))
    coup = net.coupling["coupling"]
    assert isinstance(coup, cls), f"use_delay={use_delay} → {type(coup).__name__}"
    assert np.isfinite(np.asarray(res.data)).all(), f"use_delay={use_delay}: NaN/Inf"
    md = getattr(net, "max_delay", None) or getattr(net.graph, "max_delay", None)
    print(f"use_delay={use_delay}  coupling={type(coup).__name__}  max_delay={md}")

# delay ON 은 속도가 바뀌면 궤적이 달라져야 한다(무시되면 완전히 동일해짐).
r_fast = build_network(_cfg(True), _data(30.0))[3].data
r_slow = build_network(_cfg(True), _data(1.0))[3].data
d = float(jnp.abs(jnp.asarray(r_fast) - jnp.asarray(r_slow)).max())
assert d > 1e-6, f"전도속도 30 vs 1 mm/ms 궤적 동일 → delay 미반영 (max|diff|={d:.3e})"
print(f"OK — speed 30 vs 1 mm/ms 궤적 차 max={d:.4e}")
