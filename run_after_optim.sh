#!/bin/bash
# optim 완료 대기 → idx 4,5,6 DBS sweep → zip 압축까지 자동 실행.
#
#   nohup bash run_after_optim.sh > output_ppmi_pd/after_optim.log 2>&1 &
#   nohup bash run_after_optim.sh 4 5 6 > output_ppmi_pd/after_optim.log 2>&1 &   # idx 지정
#
# 완료 판정 = Part3 grad 캐시(grad_*.pkl) 존재. 로그 파일/`.done` 마커는 못 쓴다:
# 터미널에서 직접 띄운 실행은 로그 파일이 없고, 옛 실행이 남긴 "Pipeline complete"/`.done`
# 은 FC 가 바뀐 뒤에도 남아 완료로 오판한다(2026-07-28 sweep 이 이 이유로 헛돌았다).
# cache_dir 은 cfg(delay 태그 + SC/FC 해시)에서만 정확히 나오므로 python 으로 1회 계산.
set -u
cd /scratch/home/wog3597/optim

IDXS="${*:-4 5 6}"
STAMP=$(date '+%Y%m%d_%H%M')
SWEEP_ROOT=output_ppmi_pd/dbs_sweep_log1pm
ZIP="output_ppmi_pd/dbs_sweep_${STAMP}.zip"
TIMEOUT=${TIMEOUT:-21600}          # idx 하나당 최대 대기(초). 기본 6h

log() { echo "[$(date '+%F %T')] $*"; }

log "===== cache_dir 계산 (idx $IDXS) ====="
CACHE_MAP=$(JAX_PLATFORMS=cpu python3 -c "
import main_ppmi_pd as M
from data_loader import load_data
for i in [$(echo "$IDXS" | tr ' ' ',')]:
    p = M.prepare_pd_data(i, 0.02)
    print('CACHEDIR', i, load_data(M.make_config(p, i))['cache_dir'])
" 2>&1 | grep '^CACHEDIR')
[ -n "$CACHE_MAP" ] || { log "cache_dir 계산 실패 → 중단"; exit 1; }
echo "$CACHE_MAP"

cache_of() { echo "$CACHE_MAP" | awk -v i="$1" '$2==i {print $3}'; }

alive() {   # idx -> 0=optim 프로세스 살아있음. 다른 노드(H100)에서 돌 수 있어 PBS 노드까지 본다.
    pgrep -f "main_ppmi_pd.py --subject-idx $1" >/dev/null && return 0
    local n j
    # `qstat -fu` 는 exec_host 를 안 준다 → 잡 ID 별로 -f 를 돌려야 노드 이름이 나온다.
    for n in $(for j in $(qstat -u "$USER" 2>/dev/null | awk '/^[0-9]/{print $1}'); do
                   qstat -f "$j" 2>/dev/null | grep -oP 'exec_host = \K[^/]+'; done | sort -u); do
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$n" \
            "pgrep -f 'main_ppmi_pd.py --subject-idx $1'" >/dev/null 2>&1 && return 0
    done
    return 1
}

wait_one() {   # idx -> 0=grad 캐시 준비됨 1=실패
    local i="$1" d f sz prev=0 dead=0 t0
    d=$(cache_of "$i")
    [ -n "$d" ] || { log "idx$i cache_dir 없음"; return 1; }
    t0=$(date +%s)
    while true; do
        f=$(ls -t "$d"/grad_*.pkl 2>/dev/null | head -1)
        if [ -n "$f" ]; then
            sz=$(stat -c %s "$f")
            # 22MB 짜리를 쓰는 도중 읽으면 깨진다 → 크기가 한 주기 동안 안 변해야 완료로 본다.
            if [ "$sz" -gt 20000000 ] && [ "$sz" = "$prev" ]; then
                log "idx$i 완료 — $(basename "$f")"; return 0
            fi
            prev=$sz
        else
            # grad 도 없고 프로세스도 없으면 죽은 것. ssh 일시 실패 대비로 2회 연속일 때만.
            if alive "$i"; then dead=0; else
                dead=$((dead + 1))
                [ "$dead" -ge 2 ] && { log "idx$i 실패 — grad 캐시 없고 프로세스도 없음"; return 1; }
            fi
        fi
        [ $(( $(date +%s) - t0 )) -gt "$TIMEOUT" ] && { log "idx$i 실패 — ${TIMEOUT}s 초과"; return 1; }
        sleep 60
    done
}

log "===== 대기 시작: idx [$IDXS] ====="
OK=""
for i in $IDXS; do
    if wait_one "$i"; then OK="$OK $i"; else log "idx$i 는 sweep 대상에서 제외"; fi
done
OK="${OK# }"
[ -n "$OK" ] || { log "완료된 idx 없음 → 중단"; exit 1; }
log "optim 완료 idx: [$OK]"

# 이전 FC 로 만든 sweep 결과가 남아 있으면 dbs_from_cache 가 'skip' 해버린다 → 옆으로 치운다.
for i in $OK; do
    if [ -d "$SWEEP_ROOT/idx${i}" ]; then
        mv "$SWEEP_ROOT/idx${i}" "$SWEEP_ROOT/idx${i}_oldfc_${STAMP}"
        log "기존 sweep 결과 보관: $SWEEP_ROOT/idx${i}_oldfc_${STAMP}"
    fi
done

CSV="$(echo "$OK" | tr ' ' ',')"
# jobs=1 고정. A10(23GB)에서 jobs=3 은 OOM 이다 — jupyter 커널이 5GB 상주하고 조합마다
# 자극 배열 746MB + cuFFT work 998MB 를 잡는다. docstring 의 "H100 이면 3" 은 94GB 기준.
log "===== DBS sweep 시작 (idx $OK, jobs=1) ====="
python3 -u dbs_sweep_456.py --idxs "$CSV" --jobs 1
rc=$?
if [ $rc -ne 0 ]; then
    log "sweep rc=$rc → 1회 재시도(완료 조합은 resume 으로 skip)"
    python3 -u dbs_sweep_456.py --idxs "$CSV" --jobs 1
    rc=$?
fi
log "DBS sweep 종료 rc=$rc"
[ $rc -eq 0 ] || { log "sweep 실패 → zip 생략"; exit $rc; }

log "===== zip 압축 ====="
sub_of() { ls -d output_ppmi_pd/*/cache/*_s"${1}"_*N* 2>/dev/null | head -1 | cut -d/ -f2; }
PATHS=""
for i in $OK; do
    s=$(sub_of "$i")
    PATHS="$PATHS $SWEEP_ROOT/idx${i}"
    [ -n "$s" ] && PATHS="$PATHS output_ppmi_pd/${s}/figures output_ppmi_pd/${s}/inputs"
done
# 제외: 캐시 pkl(개당 22MB). LFP 시계열은 cfg.dbs_save_lfp_timeseries=False 라 애초에
# 생성되지 않지만, 옛 결과를 재압축하는 경우를 위해 패턴은 남겨둔다.
# BOLD 시계열(bold_timeseries.csv, 조합당 ~0.8MB)은 포함한다.
zip -r -q "$ZIP" $PATHS -x "*/cache/*" "*lfp_timeseries.csv"
log "zip 완료: $ZIP  ($(du -h "$ZIP" | cut -f1))"
log "===== 전체 완료 ====="
