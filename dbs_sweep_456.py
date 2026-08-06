#!/usr/bin/env python3
"""dbs_sweep_456.py — idx 4/5/6 DBS sweep 순차 실행 (log1pm grad 캐시 사용).

각 idx 를 dbs_from_cache.py 서브프로세스로 순차 호출(크래시 격리 + idx간 GPU 상태 초기화).
dbs_from_cache 는 config 기본 sc_norm=log1pm → 최신 log1pm grad 캐시를 자동 로드
(로그의 'grad 로드 ... post_grad_corr=0.756/0.784/0.819' 로 확인 가능).
resume: 조합별 결과(fc png 4개) 있으면 dbs_from_cache 가 알아서 skip.

사용:
  python3 dbs_sweep_456.py                          # 순차 (기본 sweep amp6×freq5=30조합/idx)
  python3 dbs_sweep_456.py --jobs 3                 # idx 3개 동시 (H100: proc당 ~1.5GB, 여유)
  python3 dbs_sweep_456.py --dry-run                # 실행 명령만 출력(검증)
  python3 dbs_sweep_456.py --idxs 4,6               # idx 부분집합
  python3 dbs_sweep_456.py --amps 1.0,2.0 --freqs 130   # 커스텀
출력: output_ppmi_pd/dbs_sweep_log1pm/idx<N>/amp<a>_f<f>/<target>/fc_pre_during_diff.png
--jobs>1 이면 idx별 로그 <outroot>/idx<N>.log 에 분리 저장(stdout 안 섞이게).
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def make_monitor(outroot, idxs, ncomb, interval=10):
    """mon_dbs_sweep.sh 를 주기적으로 그려주는 렌더러. watch(1) 을 따로 띄울 필요를 없앤다.
    자식 stdout 이 파일로 빠지는 max_jobs>1 에서만 쓴다 — jobs=1 은 sweep 로그가
    터미널로 직접 흘러서 화면을 덮어쓰면 그 로그를 못 본다."""
    mon = os.path.join(HERE, "mon_dbs_sweep.sh")
    if not os.path.exists(mon):
        return None
    last = [0.0]

    def render(final=False):
        now = time.time()
        if not final and now - last[0] < interval:
            return
        last[0] = now
        r = subprocess.run(["bash", mon, outroot, " ".join(str(i) for i in idxs), str(ncomb)],
                           cwd=HERE, capture_output=True, text=True)
        # 커서 홈 + 화면 지움 → watch 처럼 제자리 갱신
        sys.stdout.write("\033[H\033[J" + r.stdout)
        sys.stdout.flush()

    return render


def run_pool(jobs_list, max_jobs, env, monitor=None):
    """jobs_list=[(idx,cmd,logpath)]. 최대 max_jobs 개 동시 실행. rc dict 반환.
    max_jobs==1 이면 순차(로그 stdout), >1 이면 idx별 파일 로그.
    monitor 가 주어지면 대기 루프에서 진행 화면을 갱신한다."""
    pending = list(jobs_list)
    running = {}   # Popen -> (idx, t0, fh)
    results = {}
    ended = []     # monitor 모드에서 지연 출력할 종료 메시지
    while pending or running:
        while pending and len(running) < max_jobs:
            idx, cmd, logp = pending.pop(0)
            if max_jobs == 1:
                print(f"{'#'*72}\n# idx{idx} 시작 → {os.path.dirname(logp)}\n{'#'*72}", flush=True)
                fh = None; out = None
            else:
                os.makedirs(os.path.dirname(logp), exist_ok=True)
                fh = open(logp, "w"); out = fh
                print(f"# launch idx{idx}  (log: {logp})", flush=True)
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=out,
                                 stderr=(subprocess.STDOUT if out else None))
            running[p] = (idx, time.time(), fh)
        done = [p for p in running if p.poll() is not None]
        for p in done:
            idx, t0, fh = running.pop(p)
            if fh: fh.close()
            results[idx] = p.returncode
            msg = (f"# idx{idx} 종료 rc={p.returncode} ({(time.time()-t0)/60:.1f} min)"
                   + ("" if p.returncode == 0 else "  ⚠️실패"))
            # monitor 가 화면을 지우며 갱신하므로 즉시 출력하면 곧 사라진다 → 끝에 모아 찍는다.
            if monitor:
                ended.append(msg)
            else:
                print(msg, flush=True)
        if running:
            if monitor:
                monitor()
            time.sleep(3)
    if monitor:
        monitor(final=True)
        for m in ended:
            print(m, flush=True)
    return results

def main():
    ap = argparse.ArgumentParser(description="idx 4/5/6 DBS sweep (log1pm grad)")
    ap.add_argument("--idxs",  default="4,5,6",                   help="콤마 구분 subject idx")
    ap.add_argument("--amps",  default="0.5,1.0,1.5,2.0,2.5,3.0", help="DBS 진폭 리스트")
    ap.add_argument("--freqs", default="20,125,143,167,200",      help="DBS 주파수 Hz 리스트")
    ap.add_argument("--noise-level", default="0.02")
    ap.add_argument("--outroot", default="output_ppmi_pd/dbs_sweep_log1pm",
                    help="출력 루트(idx별 하위폴더). 옛 max-norm(dbs_sweep)과 분리")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="동시 실행 idx 수. H100 등 큰 GPU면 3 (proc당 ~1.5GB, preallocate off).")
    ap.add_argument("--mem-frac", default=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4"),
                    help="XLA GPU mem fraction 상한(preallocate off라 실제는 on-demand)")
    ap.add_argument("--dry-run", action="store_true", help="명령만 출력, 미실행")
    ap.add_argument("--no-monitor", action="store_true",
                    help="jobs>1 에서 자동으로 뜨는 진행 화면을 끄고 기존 로그만 출력")
    a = ap.parse_args()

    idxs  = [int(x) for x in a.idxs.split(",") if x.strip()]
    ncomb = len([x for x in a.amps.split(",") if x.strip()]) * len([x for x in a.freqs.split(",") if x.strip()])
    env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=a.mem_frac, PYTHONUNBUFFERED="1")

    print(f"# idx={idxs}  amps=[{a.amps}] × freqs=[{a.freqs}] = {ncomb}조합/idx × 4타깃(STN_L/R,GP_L/R)")
    print(f"# 총 {len(idxs)}×{ncomb} = {len(idxs)*ncomb} 조합 (각 pre+during 600s sim). jobs={a.jobs} 동시. resume 지원.\n")

    jobs_list = []
    for idx in idxs:
        out = os.path.join(a.outroot, f"idx{idx}")
        cmd = [sys.executable, "-u", os.path.join(HERE, "dbs_from_cache.py"),
               "--subject-idx", str(idx), "--noise-level", a.noise_level,
               "--dbs-amplitudes", a.amps, "--dbs-freqs", a.freqs, "--dbs-outroot", out]
        jobs_list.append((idx, cmd, os.path.join(out, f"idx{idx}.log")))
        print(f"# idx{idx} → {out}\n#   {' '.join(cmd)}", flush=True)

    if a.dry_run:
        print("\n# (dry-run) 명령 출력만", flush=True); return

    # jobs>1 이면 자식 로그가 파일로 빠져 stdout 이 비니 진행 화면을 그 자리에 그린다.
    # jobs=1 은 sweep 로그가 터미널로 흘러서 화면을 덮으면 안 된다 → 모니터 없음.
    jobs = max(1, a.jobs)
    monitor = None
    if jobs > 1 and not a.no_monitor and sys.stdout.isatty():
        monitor = make_monitor(a.outroot, idxs, ncomb)

    results = run_pool(jobs_list, jobs, env, monitor)
    fails = [i for i, rc in results.items() if rc != 0]
    print(f"\n# 전체 DBS sweep 완료. 성공 {len(results)-len(fails)}/{len(results)}"
          + (f"  실패 idx={fails}" if fails else ""), flush=True)
    # 하나라도 실패하면 non-zero. rc=0 이면 호출자(run_after_optim.sh)가 성공으로 보고
    # 빈 결과를 zip 해버린다 — 2026-07-28/29 두 번 그렇게 헛돌았다.
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
