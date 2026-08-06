#!/bin/bash
# w/max SC 정규화(log1p 미사용)로 optim — 원본 EI_Tuning regime(안정, 노드당 입력 ~1.3).
#   기존 log1p 캐시와 sc-hash 분리 → 완료된 결과 안 건드림. output_ppmi_pd/<sub>/ 공유(figure만 덮어씀).
#
# 사용:
#   bash run_maxnorm.sh              # idx0만 (검증용)
#   bash run_maxnorm.sh 0 4          # idx0~4
#   bash run_maxnorm.sh 0 241        # 전체
#   nohup bash run_maxnorm.sh 0 241 > output_ppmi_pd/_batch_maxnorm.log 2>&1 &   # 백그라운드
set -u
cd /scratch/home/wog3597/optim
START=${1:-0}
END=${2:-$START}
echo "=== max-norm optim  idx $START..$END  ($(date '+%F %T')) ==="
for i in $(seq "$START" "$END"); do
    echo "--- idx $i  start $(date '+%F %T') ---"
    python3 main_ppmi_pd.py --subject-idx "$i" --sc-norm max
    echo "--- idx $i  end $(date '+%F %T')  rc=$? ---"
done
echo "=== 완료 ($(date '+%F %T')) ==="
