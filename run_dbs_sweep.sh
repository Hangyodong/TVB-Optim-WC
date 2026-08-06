#!/bin/bash
# DBS 진폭×주파수 sweep (grad 캐시서 직접 → FIC/EIB/Part3 재계산 없음).
#   진폭:   0.5, 1.0, 1.5, 2.0, 2.5, 3.0  (모델단위)
#   주파수: 20, 125, 143, 167, 200 Hz  (dt=1ms 실제: 20/125/142.86/166.67/200)
#   → 6 × 5 = 30 조합, 각 4타깃(STN_L/R + Pallidum_L/R), pre/during 각 600s.
#   네트워크는 subject당 1회 빌드(warmup), 조합마다 자극만 다시.
#
# 사용:
#   bash run_dbs_sweep.sh 0            # subject idx 0
#   bash run_dbs_sweep.sh 0 1          # 여러 subject
#   nohup bash run_dbs_sweep.sh 0 > output_ppmi_pd/dbs_sweep.log 2>&1 &
#
# 출력: output_ppmi_pd/dbs_sweep/idx<i>/amp<a>_f<freq>/{STN_L,STN_R,GP_L,GP_R}/fc_pre_during_diff.png
#
# 주의: grad(Part3) 캐시 필요. 없으면 먼저 `python3 main_ppmi_pd.py --subject-idx <i>` 로 Part3 완료.
#       시간: subject당 warmup 3min + 30조합 × 4타깃 (forward) → 대략 수 시간(nohup 권장).

cd /scratch/home/wog3597/optim

AMPS="0.5,1.0,1.5,2.0,2.5,3.0"
FREQS="20,125,143,167,200"

if [ $# -eq 0 ]; then IDXS="0"; else IDXS="$*"; fi

for i in $IDXS; do
    echo "=========================================================="
    echo "=== [$(date '+%F %T')] idx $i  DBS sweep 시작 (진폭 6 × 주파수 5 = 30) ==="
    echo "=========================================================="
    python3 dbs_from_cache.py --subject-idx "$i" \
        --dbs-amplitudes "$AMPS" --dbs-freqs "$FREQS" \
        --dbs-outroot "output_ppmi_pd/dbs_sweep/idx${i}"
    echo "=== [$(date '+%F %T')] idx $i sweep 종료 (rc=$?) ==="
done

echo ""
echo "출력: output_ppmi_pd/dbs_sweep/idx<i>/amp<a>_f<freq>/{STN_L,STN_R,GP_L,GP_R}/"
