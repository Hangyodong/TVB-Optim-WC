#!/bin/bash
# idx0 DBS sweep 모니터.  사용: bash mon_dbs.sh   또는   watch -n 30 bash mon_dbs.sh
# 인자로 idx 바꿀 수 있음: bash mon_dbs.sh 1
cd /scratch/home/wog3597/optim

IDX=${1:-0}
LOG=output_ppmi_pd/dbs_sweep_idx${IDX}.log
OUT=output_ppmi_pd/dbs_sweep/idx${IDX}
TOTAL=30

echo "=== DBS sweep idx${IDX} 모니터  ($(date '+%F %H:%M:%S')) ==="

# 1) 프로세스 살아있나
if pgrep -f "dbs_from_cache.py --subject-idx ${IDX}\b" >/dev/null 2>&1; then
    echo "상태  : 실행중 (PID $(pgrep -f "dbs_from_cache.py --subject-idx ${IDX}\b" | tr '\n' ' '))"
else
    echo "상태  : 종료 또는 중단됨"
fi

# 2) 현재 조합 (로그의 마지막 [n/30])
cur=$(grep -oE "\[[0-9]+/${TOTAL}\]" "$LOG" 2>/dev/null | tail -1)
echo "현재  : ${cur:-warmup/grad로드 중}"

# 3) 완료 조합 = 4타깃 fc_pre_during_diff.png 다 있는 amp*_f* 폴더 수
done=0
for d in "$OUT"/amp*/; do
    [ -d "$d" ] || continue
    n=$(find "$d" -name fc_pre_during_diff.png 2>/dev/null | wc -l)
    [ "$n" -ge 4 ] && done=$((done+1))
done
echo "완료  : $done / $TOTAL 조합 (4타깃 FC png 기준)"

# 4) 최근 로그 (XLA 경고 제외)
echo "--- 로그 끝 4줄:"
grep -vE "No SoL config|sol_gpu_cost_model" "$LOG" 2>/dev/null | tail -4

# 5) GPU
echo "--- GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"

# 6) 에러 있으면 표시
err=$(grep -E "Traceback|RESOURCE_EXHAUSTED|Error:" "$LOG" 2>/dev/null | tail -2)
[ -n "$err" ] && echo "!!! 에러: $err"
exit 0
