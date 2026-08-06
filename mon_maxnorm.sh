#!/bin/bash
# idx0 max-norm optim 모니터.  사용: watch -n 30 bash mon_maxnorm.sh
cd /scratch/home/wog3597/optim
L=output_ppmi_pd/optim_idx0_maxnorm.log
echo "=== idx0 max-norm optim ($(date '+%F %H:%M:%S')) ==="
if pgrep -f "main_ppmi_pd.py --subject-idx 0" | grep -qv skip; then
  echo "상태: 실행중 (PID $(pgrep -f 'main_ppmi_pd.py --subject-idx 0'|head -1),  $(ps -o etime= -p $(pgrep -f 'main_ppmi_pd.py --subject-idx 0'|head -1) 2>/dev/null|tr -d ' ') 경과)"
else
  echo "상태: 종료/완료"
fi
echo ""
echo "-- SC norm / warmup E (0.39=안정, 0.9=포화) --"
grep -iE "SC norm|노드당 입력|Warmup done" "$L" 2>/dev/null | grep -vE "No SoL" | tail -2
echo ""
echo "-- 단계 마커 --"
grep -iE "\[1\] Running FIC|c_ei updated|mean c_ei|\[2\] Running EIB|Final best|Post-hoc|\[3\] Running Grad|Part3.*Initial|Post-opt corr|Pipeline complete|Traceback|OOM|RESOURCE" "$L" 2>/dev/null | grep -vE "No SoL" | tail -6
echo ""
echo "-- FIC 수렴 (mean_S_e→0.25, se_err→0) --"
grep -E "step [0-9]+/2000" "$L" 2>/dev/null | tail -3
echo ""
echo "-- 최근 (corr/loss) --"
grep -iE "corr=|rmse=|loss|True-corr" "$L" 2>/dev/null | grep -vE "No SoL" | tail -3
echo ""
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"
echo "log tail: $(tail -1 "$L" 2>/dev/null | grep -vE 'No SoL' | cut -c1-80)"
exit 0
