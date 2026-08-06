#!/bin/bash
# idx 별 optim 진행 모니터 (mon_idx0.sh 의 범용판).
#   bash mon_idx.sh 4              # 1회 출력
#   watch -n 30 bash mon_idx.sh 4  # 30초마다 갱신
#   bash mon_idx.sh 4 경로.log     # 로그 경로 직접 지정
# 로그는 인자로 안 주면 아래 후보에서 가장 최근 것을 자동으로 고른다.
cd /scratch/home/wog3597/optim
I="${1:?사용: bash mon_idx.sh <idx> [logfile]}"
L="${2:-}"
if [ -z "$L" ]; then
  L=$(ls -t output_ppmi_pd/s${I}_newfc.log output_ppmi_pd/_logs/s${I}.log \
         output_ppmi_pd/_par_logs/s${I}.log 2>/dev/null | head -1)
fi
[ -n "$L" ] || { echo "idx$I: 로그 없음"; exit 1; }
G() { grep -vE "No SoL"; }   # 무해한 커널 경고 제거

echo "════ idx$I  $(date '+%F %H:%M:%S')  log=$L ════"
PID=$(pgrep -f "main_ppmi_pd.py --subject-idx $I\b" | head -1)
if [ -n "$PID" ]; then
  echo "상태: 실행중 (PID $PID, 경과 $(ps -o etime= -p "$PID" | tr -d ' '))"
elif grep -q "Pipeline complete" "$L" 2>/dev/null; then
  echo "상태: 완료"
else
  echo "상태: 미실행 (로컬 프로세스 없음 — PBS job 이면 qstat 확인)"
fi
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"

echo ""
echo "-- 데이터/캐시 --"
grep -E "sub_num=|cache_tag" "$L" | G | tail -2
echo ""
echo "-- 단계 마커 --"
grep -iE "\[1\] Running FIC|\[2\] Running EIB|\[3\] Running Grad|c_ei updated|Post-hoc|GRAD\] Complete|Pipeline complete|Traceback|OOM|RESOURCE_EXHAUSTED" "$L" | G | tail -6
echo ""
echo "-- 최근 진행 --"
grep -iE "step |corr|loss|mean_S_e" "$L" | G | tail -5
echo ""
echo "-- log tail --"
tail -3 "$L" | G
