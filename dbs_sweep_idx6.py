#!/usr/bin/env python3
"""dbs_sweep_idx6.py — 실행중 DBS sweep 전부 중단 후 idx6 sweep(resume) 단독 실행.

idx6 hang 복구용. 먼저 dbs_sweep_456.py(드라이버) + dbs_from_cache.py(모든 worker)를
SIGTERM(→안 죽으면 SIGKILL)로 정리하고, idx6 만 GPU 독점으로 재개(완료 combo skip).

사용:
  python3 dbs_sweep_idx6.py                    # 전부 kill 후 idx6 sweep(resume)
  python3 dbs_sweep_idx6.py --dry-run          # kill 대상 + 실행 명령만 출력(미실행)
  python3 dbs_sweep_idx6.py --no-kill          # kill 생략, idx6 만 실행
  python3 dbs_sweep_idx6.py --amps 1.0 --freqs 130   # 커스텀 sweep
"""
import argparse, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
KILL_PATTERNS = ["dbs_sweep_456.py", "dbs_from_cache.py"]   # 드라이버 + 모든 worker


def _pids(pattern):
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    me = os.getpid()
    return [int(p) for p in r.stdout.split() if p.strip() and int(p) != me]


def kill_sweeps(dry=False):
    targets = sorted({pid for pat in KILL_PATTERNS for pid in _pids(pat)})
    if not targets:
        print("[kill] 실행중 sweep 프로세스 없음"); return
    if dry:
        print(f"[kill] (dry-run) 종료 대상 PID: {targets}"); return
    print(f"[kill] SIGTERM → {targets}")
    for pid in targets:
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    time.sleep(5)                                  # graceful 종료 대기
    for pid in targets:                            # 아직 살아있으면 SIGKILL
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            print(f"[kill] SIGKILL {pid} (SIGTERM 무시)")
        except (ProcessLookupError, OSError):
            pass
    time.sleep(2)
    left = sorted({pid for pat in KILL_PATTERNS for pid in _pids(pat)})
    print(f"[kill] 정리 완료. 남은 프로세스: {left or '없음'}")


def main():
    ap = argparse.ArgumentParser(description="실행중 sweep kill 후 idx6 재개")
    ap.add_argument("--idx", type=int, default=6)
    ap.add_argument("--amps",  default="0.5,1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--freqs", default="20,125,143,167,200")
    ap.add_argument("--noise-level", default="0.02")
    ap.add_argument("--outroot", default="output_ppmi_pd/dbs_sweep_log1pm")
    ap.add_argument("--mem-frac", default="0.5", help="GPU 독점이라 여유. H100이면 무관(preallocate off)")
    ap.add_argument("--no-kill", action="store_true", help="kill 생략")
    ap.add_argument("--dry-run", action="store_true", help="kill 대상+명령만 출력, 미실행")
    a = ap.parse_args()

    if not a.no_kill:
        kill_sweeps(dry=a.dry_run)

    out = os.path.join(a.outroot, f"idx{a.idx}")
    cmd = [sys.executable, "-u", os.path.join(HERE, "dbs_from_cache.py"),
           "--subject-idx", str(a.idx), "--noise-level", a.noise_level,
           "--dbs-amplitudes", a.amps, "--dbs-freqs", a.freqs, "--dbs-outroot", out]
    print(f"[run] idx{a.idx} sweep (resume, 완료 combo skip) → {out}\n  {' '.join(cmd)}", flush=True)
    if a.dry_run:
        print("[run] (dry-run) 미실행"); return

    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=a.mem_frac, PYTHONUNBUFFERED="1")
    t = time.time()
    rc = subprocess.run(cmd, cwd=HERE, env=env).returncode
    print(f"[done] idx{a.idx} rc={rc}  ({(time.time()-t)/60:.1f} min)", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
