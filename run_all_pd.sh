#!/usr/bin/env bash
# tr_2.5_PD subject 순차 배치 실행.
#   한 subject 끝나면 다음 subject 자동 실행. 실패해도 다음으로 진행(크래시 격리).
#   이미 끝난 idx 는 marker 로 skip → 중단 후 재실행하면 이어서 진행(resume).
#
# 사용:
#   bash run_all_pd.sh                # GPU0, idx 0~241 전체
#   bash run_all_pd.sh 0 0 241        # GPU0, idx 0~241 (명시)
#   bash run_all_pd.sh 1 0 120        # GPU1, idx 0~120  (다GPU 분할용)
#   nohup bash run_all_pd.sh 0 0 241 > output_ppmi_pd/_batch.log 2>&1 &   # 로그아웃 후에도 유지
#
# 처음부터 다시: rm output_ppmi_pd/_logs/s*.done
set -u

cd /scratch/home/wog3597/optim

GPU=${1:-0}       # 인자1: GPU id (기본 0)
START=${2:-0}     # 인자2: 시작 idx (기본 0)
END=${3:-241}     # 인자3: 끝 idx 포함 (기본 241 = 242명 마지막)

LOGDIR=output_ppmi_pd/_logs
mkdir -p "$LOGDIR"

echo "=== batch start  GPU=$GPU  idx $START..$END  ($(date '+%F %T')) ==="
for i in $(seq "$START" "$END"); do
    marker="$LOGDIR/s${i}.done"
    if [[ -f "$marker" ]]; then
        echo "[skip] idx $i (already done)"
        continue
    fi
    echo "[run ] idx $i  start $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES="$GPU" python3 main_ppmi_pd.py --subject-idx "$i" \
        > "$LOGDIR/s${i}.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        touch "$marker"
        echo "[done] idx $i  end $(date '+%F %T')"
    else
        echo "[FAIL] idx $i  rc=$rc  → $LOGDIR/s${i}.log"
    fi
done
echo "=== batch complete  ($(date '+%F %T')) ==="
