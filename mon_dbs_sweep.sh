#!/bin/bash
# DBS sweep(log1pm, idx4/5/6 병렬) 모니터 — worker별 progress bar.
#   사용: watch -n 10 bash mon_dbs_sweep.sh
#   인자: bash mon_dbs_sweep.sh <outroot> <"idx목록"> <조합수>
cd /scratch/home/wog3597/optim
ROOT="${1:-output_ppmi_pd/dbs_sweep_log1pm}"
IDXS="${2:-4 5 6}"
NCOMB="${3:-30}"
NTGT=4                                   # STN_L/R, GP_L/R
PER=$((NCOMB*NTGT))                       # idx당 목표 png (120)
NIDX=$(echo $IDXS | wc -w)
TOTAL=$((PER*NIDX))
NOW=$(date +%s)

bar() {  # cur tot width → █░ 막대
  local c=$1 t=${2:-1} w=$3 i n s=""
  [ "$t" -le 0 ] && t=1
  n=$(( c*w/t )); [ $n -gt $w ] && n=$w
  for ((i=0;i<w;i++)); do [ $i -lt $n ] && s+="█" || s+="░"; done
  printf "%s" "$s"
}
pct() { awk -v c=$1 -v t=${2:-1} 'BEGIN{printf "%.0f", (t>0?100*c/t:0)}'; }
hms() { local s=${1%.*}; printf "%dh%02dm" $((s/3600)) $(((s%3600)/60)); }

# 집계는 대상 idx 폴더만 본다. $ROOT 전체를 훑으면 idx*_oldfc_* 보관본까지 세서
# TOTAL 이 100% 를 넘고 경과시간도 옛 png mtime 으로 엉킨다.
DIRS=""; for i in $IDXS; do [ -d "$ROOT/idx$i" ] && DIRS="$DIRS $ROOT/idx$i"; done
[ -n "$DIRS" ] || DIRS="$ROOT"

# ── 전체 png 수 + 시작시각(가장 오래된 png) ──
mapfile -t MT < <(find $DIRS -name fc_pre_during_diff.png -printf '%T@\n' 2>/dev/null | sort -n)
DONE=${#MT[@]}
first=${MT[0]%.*}; elapsed=$(( NOW - ${first:-$NOW} ))
rate=$(awk -v d=$DONE -v e=$elapsed 'BEGIN{print (e>0&&d>0)? d/e : 0}')   # png/sec
eta="?"
[ "$DONE" -gt 0 ] && [ "$DONE" -lt "$TOTAL" ] && eta=$(awk -v r=$rate -v rem=$((TOTAL-DONE)) 'BEGIN{print (r>0)? rem/r : 0}')

# ── worker 활성도(파일 갱신 기준 = 노드 무관) ──
active=0; finished=0
for i in $IDXS; do
  O="$ROOT/idx$i"; L="$ROOT/idx$i/idx$i.log"
  n=$(find "$O" -name fc_pre_during_diff.png 2>/dev/null | wc -l)
  if grep -qa "sweep 완료" "$L" 2>/dev/null || [ "$n" -ge "$PER" ]; then finished=$((finished+1))
  elif [ -n "$(find "$O" -name fc_pre_during_diff.png -newermt '-8 min' 2>/dev/null | head -1)" ]; then active=$((active+1)); fi
done
GPU=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
[ -z "$GPU" ] && GPU="(이 노드 GPU 없음 — job 노드서 실행해야 표시)"

echo "╔══════════ DBS sweep monitor (log1pm)   $(date '+%F %H:%M:%S') ══════════"
echo "║ worker : 실행중 $active · 완료 $finished / $NIDX  (파일갱신 기준, 노드무관)"
echo "║ GPU    : $GPU   [$(hostname)]"
printf "║ TOTAL  : %4d/%d png  [%s] %s%%   경과 %s  ETA≈%s\n" \
  "$DONE" "$TOTAL" "$(bar $DONE $TOTAL 24)" "$(pct $DONE $TOTAL)" "$(hms $elapsed)" \
  "$([ "$eta" = "?" ] && echo "?" || hms $eta)"
echo "╟─────────────────────────────────────────────────────────────────────"

for i in $IDXS; do
  L="$ROOT/idx$i/idx$i.log"; O="$ROOT/idx$i"
  # 현재 조합 [k/30] + amp/freq
  line=$(grep -aoE "\[[0-9]+/$NCOMB\] amp=±[0-9.]+  freq=[0-9]+Hz" "$L" 2>/dev/null | tail -1)
  k=$(echo "$line" | grep -oE "^\[[0-9]+" | tr -d '[')
  af=$(echo "$line" | grep -oE "amp=±[0-9.]+  freq=[0-9]+Hz" | tr -s ' ')
  # 현재 타깃
  tgt=$(grep -aoE "STN_L|STN_R|GP_L|GP_R" "$L" 2>/dev/null | tail -1)
  # png 수 + 상태
  n=$(find "$O" -name fc_pre_during_diff.png 2>/dev/null | wc -l)
  if grep -qa "sweep 완료" "$L" 2>/dev/null || [ "$n" -ge "$PER" ]; then st="✅완료"
  elif grep -qaE "Traceback|Error:|RESOURCE_EXHAUSTED|OOM" "$L" 2>/dev/null && ! pgrep -f "subject-idx $i\b">/dev/null; then st="❌에러"
  else
    fresh=$(find "$O" -name fc_pre_during_diff.png -newermt '-8 min' 2>/dev/null | head -1)
    [ -n "$fresh" ] && st="⏱실행중" || st="…대기/warmup"
  fi
  printf "║ idx%s combo %2s/%d [%s]%s%%  %-22s tgt:%-5s  png %3d/%d  %s\n" \
    "$i" "${k:-0}" "$NCOMB" "$(bar ${k:-0} $NCOMB 12)" "$(pct ${k:-0} $NCOMB)" \
    "${af:-warmup}" "${tgt:--}" "$n" "$PER" "$st"
done

echo "╟─────────────────────────────────────────────────────────────────────"
# 최근 완료 png 3개
find $DIRS -name fc_pre_during_diff.png -printf '%TH:%TM  %p\n' 2>/dev/null | sort | tail -3 | \
  sed -E "s#$ROOT/##; s#/true_p_t/fc_pre_during_diff.png##" | sed 's/^/║ 최근 /'
echo "╚══════════════════════════════════════════════════════════════════════"
