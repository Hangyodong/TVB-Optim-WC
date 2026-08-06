#!/bin/bash
# 병렬 optim(idx 0~3) 디테일 모니터.  사용: watch -n 15 bash mon_par.sh
cd /scratch/home/wog3597/optim
LOGD=output_ppmi_pd/_par_logs
IDXS="${*:-0 1 2 3}"   # 인자 전부 사용 (mon_par.sh 2 3 → "2 3", 따옴표 불필요)
declare -A SUB=([0]=100001 [1]=100005 [2]=100012 [3]=100268)
GVIS() { grep -vE "No SoL" ; }

echo "════════ 병렬 optim 모니터  $(date '+%F %H:%M:%S') ════════"
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null | head -1)"
run=$(pgrep -fc "python3 main_ppmi_pd.py --subject-idx" 2>/dev/null)
done=$(grep -l "Pipeline complete" $LOGD/s*.log 2>/dev/null | wc -l)
echo "실행중 worker: $run개    완료: $done"
for p in $(pgrep -f "python3 main_ppmi_pd.py --subject-idx" 2>/dev/null); do
  ix=$(ps -o cmd= -p "$p" 2>/dev/null | grep -oE "subject-idx [0-9]+" | grep -oE "[0-9]+$")
  read cpu et < <(ps -o %cpu=,etime= -p "$p" 2>/dev/null)
  echo "   ├ PID $p  idx$ix  CPU ${cpu}%  경과 $et"
done
echo ""

for i in $IDXS; do
  L="$LOGD/s${i}.log"; sub=${SUB[$i]:-?}
  if [ ! -f "$L" ]; then echo "─── idx$i ($sub): 로그 없음(미시작)"; echo ""; continue; fi
  # 단계 판정
  if   grep -q "Pipeline complete" "$L"; then st="✅완료"
  elif tail -30 "$L" | grep -qE "Traceback|RESOURCE_EXHAUSTED|OOM|Error:"; then st="❌에러"
  elif grep -q "GRAD\] Complete\|Part3 Plot\] Post-opt" "$L"; then st="Part3✔"
  elif grep -q "\[3\] Running Grad\|Part3\] Initial" "$L"; then st="Part3"
  elif grep -q "\[2\] Running EIB" "$L"; then st="EIB"
  elif grep -q "\[1\] Running FIC" "$L"; then st="FIC"
  elif grep -q "Warmup done\|Building network" "$L"; then st="warmup"
  else st="시작중"; fi
  # 경과 시간
  pid=$(pgrep -f "main_ppmi_pd.py --subject-idx $i\$" 2>/dev/null | head -1)
  et=$([ -n "$pid" ] && ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')

  echo "─── idx$i ($sub) ─── [$st] ${et:+경과 $et}"
  # ▶ 현재 step (FIC 2000 / EIB 10000 / Part3 250 중 최신)
  prog=$(grep -E "step [0-9]+/2000|^ +[0-9]+/10000 |^ +[0-9]+/250 " "$L" 2>/dev/null | tail -1 | sed 's/^ *//' | cut -c1-72)
  [ -n "$prog" ] && echo "   ▶ 현재 step: $prog"
  # 환경
  grep -oE "SC norm='[a-z0-9]+'.*mean=[0-9.]+" "$L" | tail -1 | sed 's/^/   /'
  grep -oE "Warmup done — E mean=[0-9.]+  I mean=[0-9.]+" "$L" | tail -1 | sed 's/^/   [warmup] /'
  # FIC
  fcei=$(grep -oE "mean c_ei ?= ?[0-9.]+" "$L" | tail -1)
  fstep=$(grep -E "step [0-9]+/2000" "$L" | tail -1 | grep -oE "step [0-9]+/2000  mean_S_e=[0-9.]+.*se_err=[0-9.]+")
  [ -n "$fstep" ] && echo "   [FIC] $fstep"
  [ -n "$fcei" ] && echo "   [FIC] c_ei $fcei"
  # EIB step 행 (N/10000)  +  best
  estep=$(grep -E "^ +[0-9]+/10000 " "$L" | tail -1)
  ebest=$(grep -oE "Final best @ iter [0-9]+.*true_corr=[0-9.]+.*" "$L" | tail -1)
  [ -n "$estep" ] && echo "   [EIB] step/win-corr/win-rmse/best/elapsed/eta:$estep"
  [ -n "$ebest" ] && echo "   [EIB] $ebest"
  # Part3 step 행 (N/250, EIB의 /10000 제외)
  pstep=$(grep -E "^ +[0-9]+/[0-9]+ " "$L" | grep -v "/10000" | tail -1)
  [ -n "$pstep" ] && echo "   [Part3] step/max loss corr best sps:$pstep"
  pfin=$(grep -oE "Post-opt corr=[0-9.]+  rmse=[0-9.]+" "$L" | tail -1)
  [ -n "$pfin" ] && echo "   ✅ [최종] $pfin"
  # 마지막 raw 1줄(진행 감)
  echo "   ┄ $(tail -1 "$L" 2>/dev/null | GVIS | cut -c1-78)"
  echo ""
done
exit 0
