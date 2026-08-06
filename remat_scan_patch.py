"""remat_scan_patch.py — opt-in gradient checkpointing for tvboptim lax.scan

환경변수 PART3_REMAT_SCAN=1 일 때, jax.lax.scan 의 body(op)를 jax.checkpoint 로 감싼다.
- forward 결과·gradient 는 비트수준 동일 (checkpoint 은 forward 투명).
- backward 시 scan-body 내부를 저장하지 않고 재계산 → 장기 적분 AD tape 메모리 大절감.

PART3_REMAT_POLICY (기본 "dots") 로 재계산량 조절:
  nothing    : scan body 전부 재계산. 최소 메모리, 최대 재계산 = 가장 느림(구 동작).
  dots       : matmul(dot, no-batch) 출력만 저장 → 그만큼 재계산↓, 메모리 소폭↑. 새 기본.
  dots_all   : 모든 dot 출력 저장.
  everything : 재계산 없음(= remat off). 메모리 최대 → 장기적분 OOM 위험(A10 불가).
               ※ H100 94GB 는 full 240-TR 도 ~60GB 로 fit 가능성 → PART3_REMAT_SCAN=0 과 동등.

사용 (part3 optimizer.run 주변):
    from remat_scan_patch import remat_scan
    with remat_scan():            # PART3_REMAT_SCAN=1 이면 활성, 아니면 no-op
        optimizer.run(...)

전역 jax.lax.scan 을 임시 교체하고 finally 에서 무조건 복원하므로 누수 없음.
"""
import contextlib
import os

import jax


def remat_scan_enabled() -> bool:
    return os.environ.get("PART3_REMAT_SCAN", "0") == "1"


def _resolve_policy():
    """PART3_REMAT_POLICY 이름 → jax checkpoint policy (또는 None=save-nothing)."""
    # 기본 nothing: 벤치 결과 dots 는 무속도이득(모델이 latency-bound라 matmul 저장이
    # 순차 재계산 latency 를 못 줄임) + 비트-비동일 → 안전한 구 동작을 기본으로.
    # 진짜 remat 속도이득은 PART3_REMAT_SCAN=0 (remat off, H100 94GB fit 시 ~1.5-2x).
    name = os.environ.get("PART3_REMAT_POLICY", "nothing").strip().lower()
    if name in ("nothing", "none", ""):
        return None  # jax.checkpoint 기본(save nothing) = 구 동작
    cp = jax.checkpoint_policies
    attr = {
        "dots":       "dots_with_no_batch_dims_saveable",
        "dots_all":   "dots_saveable",
        "everything": "everything_saveable",
    }.get(name, "dots_with_no_batch_dims_saveable")
    return getattr(cp, attr, None)


@contextlib.contextmanager
def remat_scan(enabled: bool = None):
    """jax.lax.scan body 를 jax.checkpoint(policy) 로 감싸는 컨텍스트. enabled=None 이면 env 로 결정."""
    if enabled is None:
        enabled = remat_scan_enabled()
    if not enabled:
        yield
        return

    orig_scan = jax.lax.scan
    policy = _resolve_policy()

    def _remat_scan(f, *args, **kwargs):
        # f = scan body (carry, x) -> (carry, y). checkpoint 로 감싸면 backward 재계산.
        ckpt = jax.checkpoint(f) if policy is None else jax.checkpoint(f, policy=policy)
        return orig_scan(ckpt, *args, **kwargs)

    jax.lax.scan = _remat_scan
    try:
        yield
    finally:
        jax.lax.scan = orig_scan
