#!/bin/bash
# idx0 frozen optim 모니터.  사용: bash mon_idx0.sh   또는   watch -n 30 bash mon_idx0.sh
cd /scratch/home/wog3597/optim
L=output_ppmi_pd/optim_idx0_seed5.log
echo "=== idx0 multiseed(N=5) + activity reg ($(date '+%F %H:%M:%S')) ==="
if pgrep -f "main_ppmi_pd.py --subject-idx 0" | grep -qv skip-gradient; then
  echo "상태: 실행중 (PID $(pgrep -f 'main_ppmi_pd.py --subject-idx 0' | head -1))"
else
  echo "상태: 종료/대기"
fi
echo ""
echo "-- 단계 마커 --"
grep -iE "\[FIC\] c_ei updated|c_ei_frozen=True|\[EIB\]|Post-hoc|\[Part3\]|Pipeline complete|Traceback|OOM|RESOURCE_EXHAUSTED" "$L" 2>/dev/null | grep -vE "No SoL" | tail -6
echo ""
echo "-- 최근 진행 (Step/corr/loss) --"
grep -iE "Step |corr|loss|iter" "$L" 2>/dev/null | grep -vE "No SoL" | tail -4
echo ""
echo "-- c_ei drift 감시 (원본식: 생리적 유지 기대, 0.2대면 drift) --"
grep -iE "mean c_ei|c_ei mean|c_ei=|activity" "$L" 2>/dev/null | grep -vE "No SoL" | tail -4
echo ""
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"
echo "log tail:"; tail -2 "$L" 2>/dev/null | grep -vE "No SoL"
exit 0
