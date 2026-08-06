#!/bin/bash
# tr_2.5_PD 여러 subject 를 1개 H100 에 packing 해 병렬 실행 (PBS job 1개).
#   H100 94GB 는 subject 당 Part3 메모리(수 GB)를 여러 개 얹어도 여유.
#   시뮬이 scan latency-bound 라 1워커는 GPU util ~39%(doc 17) → packing 이 효율적.
#   재개(marker) 동작: 중단/timeout 돼도 다시 qsub 하면 남은 idx 만 이어서.
#
# 제출:
#   qsub -v IDXS="0 1 2 3 4" qsub_par_pd.sh     # idx 0~4 (실제 subject 인덱스로 교체)
#   qsub qsub_par_pd.sh                          # 기본 IDXS="0 1 2 3 4"
# 상태:   qstat -u wog3597
# 로그:   tail -f output_ppmi_pd/_par.log        # 배치 로그
#         tail -f output_ppmi_pd/_w<idx>.log     # 워커별 로그
# 취소:   qdel <jobid>
# 처음부터: rm output_ppmi_pd/_logs/s*.done      # HRF 수정 재실행 시 marker 없으면 생략 가능
#
# ── PBS 자원 (H100 1개, 5워커 packing 기준) ─────────────────────────
#PBS -N pd_par
#PBS -q remote
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /scratch/home/wog3597/optim/output_ppmi_pd/_par.log

set -u

source /scratch/home/wog3597/anaconda3/etc/profile.d/conda.sh
conda activate base

cd /scratch/home/wog3597/optim

# idx 목록은 -v IDXS="..." 로 넘긴다. 각 워커가 subject 1개씩, 같은 H100 에 병렬로 얹힌다.
IDXS="${IDXS:-0 1 2 3 4}"

echo "=== parallel batch start  idx=[$IDXS]  ($(date '+%F %T')) ==="
for i in $IDXS; do
    echo "[launch] worker idx $i  → output_ppmi_pd/_w${i}.log"
    bash run_all_pd.sh 0 "$i" "$i" > "output_ppmi_pd/_w${i}.log" 2>&1 &
done
wait   # 모든 워커가 끝날 때까지 job 유지 (없으면 job 즉시 종료 → 워커 kill)
echo "=== parallel batch complete  ($(date '+%F %T')) ==="
