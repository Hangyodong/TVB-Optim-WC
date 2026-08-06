#!/bin/bash
# log1pm FIC(part1) 상세 모니터 — 진행 step/ETA/수렴/과흥분 판정.
#   사용: watch -n 10 bash mon_fic.sh   [로그경로]
#   (unbuffered 실행이어야 live: PYTHONUNBUFFERED=1 python3 -u ...)
cd /scratch/home/wog3597/optim
L="${1:-/var/tmp/pbs.71190.KITSM02/claude-2335/-scratch-home-wog3597-optim/38e98c22-dd0f-42c2-9adf-8b3eca585ff1/scratchpad/fic_log1pm_idx6.log}"
PAT="main_ppmi_pd.py --subject-idx 6 --fic-only"
MAXSTEP=2000
GVIS() { grep -vaE "No SoL|FigureCanvasAgg|_main_original_show"; }

echo "════════ log1pm FIC 상세 모니터  $(date '+%F %H:%M:%S') ════════"

# ── 프로세스/GPU ──
PID=$(pgrep -f "$PAT" | head -1)
if [ -n "$PID" ]; then
  read cpu et < <(ps -o %cpu=,etime= -p "$PID" 2>/dev/null)
  echo "프로세스 : 실행중  PID $PID  CPU ${cpu}%  경과 $et"
else
  echo "프로세스 : 종료됨"
fi
echo "GPU      : $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"

# ── SC norm 적용 확인 ──
scline=$(grep -aiE "SC norm|노드당 입력" "$L" 2>/dev/null | GVIS | tail -1)
[ -n "$scline" ] && echo "SC norm  : $(echo "$scline" | sed 's/^\[DATA\] *//' | cut -c1-70)"

# ── 단계(phase) 판정 ──
if   grep -qa "c_ei updated" "$L" 2>/dev/null;                         then phase="✅ 완료 (FIC 종료)"
elif tail -40 "$L" 2>/dev/null | grep -qaE "Traceback|RESOURCE_EXHAUSTED|OOM|Error:"; then phase="❌ 에러 (로그tail 확인)"
elif grep -qa "후보.*선택\|best 후보" "$L" 2>/dev/null;                then phase="후보선택→posthoc(600s sim)"
elif grep -qaE "step +[0-9]+/$MAXSTEP" "$L" 2>/dev/null;               then phase="FIC 루프 진행중"
elif grep -qa "\[FIC\] target S_e" "$L" 2>/dev/null;                   then phase="FIC 시작(첫 step 대기)"
elif grep -qaiE "Warmup" "$L" 2>/dev/null;                            then phase="warmup"
else phase="초기화(load/build)"; fi
echo "단계     : $phase"
echo ""

# ── FIC 진행 step + ETA ──
last=$(grep -aE "step +[0-9]+/$MAXSTEP" "$L" 2>/dev/null | GVIS | tail -1)
if [ -n "$last" ]; then
  step=$(echo "$last"   | grep -oE "step +[0-9]+" | grep -oE "[0-9]+")
  se=$(echo "$last"     | grep -oE "mean_S_e=[0-9.]+" | grep -oE "[0-9.]+")
  rE=$(echo "$last"     | grep -oE "rE=[0-9.]+"       | grep -oE "[0-9.]+")
  err=$(echo "$last"    | grep -oE "se_err=[0-9.]+"   | grep -oE "[0-9.]+")
  elap=$(echo "$last"   | grep -oE "\([0-9.]+s\)"     | grep -oE "[0-9.]+")
  pct=$(awk -v s="$step" -v m="$MAXSTEP" 'BEGIN{printf "%.1f", 100*s/m}')
  # progress bar
  bar=$(awk -v s="$step" -v m="$MAXSTEP" 'BEGIN{n=int(30*s/m); for(i=0;i<30;i++) printf (i<n?"█":"░")}')
  eta="?"
  if [ -n "$elap" ] && [ "$step" -gt 0 ]; then
    eta=$(awk -v e="$elap" -v s="$step" -v m="$MAXSTEP" 'BEGIN{r=(m-s)*e/s; printf "%dm%02ds", int(r/60), int(r)%60}')
  fi
  echo "진행 step: $step/$MAXSTEP  ($pct%)  [$bar]"
  echo "           mean_S_e=$se (→0.25)  rE=$rE Hz  se_err=$err  경과 ${elap}s  ETA≈$eta"
else
  echo "진행 step: (아직 FIC step 출력 없음 — warmup/초기화 중)"
fi
echo ""

# ── 수렴 추이 (최근 5개 step) ──
echo "-- 수렴 추이 (mean_S_e 0.25 수렴, se_err→0) --"
grep -aE "step +[0-9]+/$MAXSTEP" "$L" 2>/dev/null | GVIS | tail -5 | sed 's/^ *//'
echo ""

# ── 최종/후보 결과 ──
fin=$(grep -aiE "Early stop|후보.*선택|best 후보|c_ei updated|mean c_ei|final rE_hz" "$L" 2>/dev/null | GVIS | tail -4)
[ -n "$fin" ] && { echo "-- FIC 결과 --"; echo "$fin" | sed 's/^ *//'; echo ""; }

# ── 과흥분 자동판정 (현재 mean_S_e) ──
cur_se="${se:-$(grep -oaE "mean_S_e=[0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")}"
if [ -n "$cur_se" ]; then
  verdict=$(awk -v s="$cur_se" 'BEGIN{
    if(s>0.6) print "⚠️⚠️ 포화위험 (>0.6, 상한0.9)";
    else if(s>0.4) print "⚠️ 높음 (>0.4)";
    else if(s>=0.2 && s<=0.3) print "✓ target 근접 (0.25±)";
    else if(s<0.2) print "△ 낮음 (under-excited)";
    else print "· 진행중"}')
  echo "과흥분판정: mean_S_e=$cur_se  →  $verdict"
fi
echo "log tail : $(tail -1 "$L" 2>/dev/null | GVIS | cut -c1-95)"
exit 0
