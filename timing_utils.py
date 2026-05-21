"""
timing_utils.py — Lightweight timing prediction & micro-benchmark utilities.
파이프라인 파일(part*.py)을 import하지 않는다. 과학 연산을 수행하지 않는다.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class StageTimingEstimate:
    stage: str
    n_steps: int
    measured_step_sec: float
    estimated_total_sec: float

    def fmt(self) -> str:
        total = self.estimated_total_sec
        if total < 60:
            t = f"{total:.1f}s"
        elif total < 3600:
            t = f"{total/60:.1f}min"
        else:
            t = f"{total/3600:.2f}h"
        return (f"[{self.stage:>16}] steps={self.n_steps:>6d}  "
                f"per-step={self.measured_step_sec*1000:>8.2f}ms  total≈{t}")


def measure_callable(fn: Callable, warmup: int = 1, repeat: int = 3) -> float:
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    return float(sum(durations) / len(durations))


def estimate_stage(stage_name, n_steps, per_step_fn=None,
                    fallback_step_sec=0.5, warmup=1, repeat=3):
    if per_step_fn is not None:
        per_step = measure_callable(per_step_fn, warmup, repeat)
    else:
        per_step = float(fallback_step_sec)
    return StageTimingEstimate(stage_name, int(n_steps), per_step, per_step * n_steps)


def print_estimates(estimates) -> None:
    total = sum(e.estimated_total_sec for e in estimates)
    print("=" * 76)
    print(f"{'Pipeline timing estimates (analytical + JAX latency)':^76}")
    print("=" * 76)
    for e in estimates:
        print(e.fmt())
    print("-" * 76)
    if total < 60:
        t = f"{total:.1f}s"
    elif total < 3600:
        t = f"{total/60:.1f}min"
    else:
        t = f"{total/3600:.2f}h"
    print(f"  {'TOTAL (excl. caching, compile)':>54}  ≈ {t}")
    print("=" * 76)


def gpu_summary() -> dict:
    try:
        import jax
        devices = jax.devices()
        return {
            "backend": jax.default_backend(),
            "device_count": len(devices),
            "devices": [str(d) for d in devices],
            "default_device": str(devices[0]) if devices else None,
            "x64_enabled": jax.config.read("jax_enable_x64"),
        }
    except Exception as exc:
        return {"error": str(exc)}


def print_gpu_summary() -> None:
    info = gpu_summary()
    print("=" * 76)
    print(f"{'JAX backend / device summary':^76}")
    print("=" * 76)
    for k, v in info.items():
        print(f"  {k:>18}: {v}")
    print("=" * 76)
