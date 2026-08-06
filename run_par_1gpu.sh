#!/bin/bash
# 단일 GPU 병렬 optim (H100 94GB). N 워커가 idx 범위를 나눠 동시 실행.
#   EIB=CPU-bound(GPU util~0) → 코어 수 제한(16코어면 ~6-8). Part3=GPU 메모리 제한(~12).
#   resume: 끝난 idx는 marker(_par_logs/s<i>.done)로 skip → 중단 후 재실행하면 이어서.
#   max-norm 기본(config sc_norm="max") 자동 적용.
#
# 사용:
#   bash run_par_1gpu.sh                 # 6워커, idx 0~241
#   bash run_par_1gpu.sh 8               # 8워커
#   bash run_par_1gpu.sh 6 0 241         # 6워커, 범위 명시
#   nohup bash run_par_1gpu.sh 6 > output_ppmi_pd/_par.log 2>&1 &   # 백그라운드
#   처음부터: rm output_ppmi_pd/_par_logs/s*.done
set -u
cd /scratch/home/wog3597/optim
N=${1:-6}; START=${2:-0}; END=${3:-241}
LOGDIR=output_ppmi_pd/_par_logs; mkdir -p "$LOGDIR"
TOTAL=$((END - START + 1))
CHUNK=$(( (TOTAL + N - 1) / N ))
echo "=== 병렬 $N 워커  idx $START..$END  (워커당 ~$CHUNK)  단일 GPU  $(date '+%F %T') ==="
pids=()
for ((w=0; w<N; w++)); do
    s=$((START + w*CHUNK)); e=$((s + CHUNK - 1)); [ $e -gt $END ] && e=$END
    [ $s -gt $END ] && break
    (
        for i in $(seq $s $e); do
            m="$LOGDIR/s${i}.done"
            [ -f "$m" ] && { echo "[w$w] idx $i skip(done)"; continue; }
            echo "[w$w] idx $i start $(date '+%T')"
            python3 main_ppmi_pd.py --subject-idx "$i" > "$LOGDIR/s${i}.log" 2>&1 \
                && touch "$m" && echo "[w$w] idx $i DONE" \
                || echo "[w$w] idx $i FAIL"
        done
    ) &
    pids+=($!)
    echo "  워커 $w: idx $s..$e  (PID $!)"
done
wait "${pids[@]}"
echo "=== 전체 완료 $(date '+%F %T') ==="
