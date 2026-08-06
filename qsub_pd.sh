#!/bin/bash
# tr_2.5_PD 전체 배치를 독립 PBS job으로 제출.
#   ParaView 대화형 세션과 무관하게 실행 → 세션 닫아도 유지, walltime 길게.
#   재개(marker) 그대로 동작: 중단돼도 다시 qsub 하면 이어서.
#
# 제출:   qsub qsub_pd.sh                      # idx 0~241 전체
#         qsub -v START=1,END=1 qsub_pd.sh     # idx 1 (2번째 subject) 만
#         qsub -v START=1,END=241 qsub_pd.sh   # idx 1부터 끝까지
#         qsub -v GPU=1,START=1,END=120 qsub_pd.sh
# 상태:   qstat -u wog3597
# 로그:   tail -f /scratch/home/wog3597/optim/output_ppmi_pd/_batch.log
# 취소:   qdel <jobid>
#
# ── PBS 자원 요청 (클러스터 정책에 맞게 수정) ──────────────────────
#PBS -N pd_batch
#PBS -q remote
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o /scratch/home/wog3597/optim/output_ppmi_pd/_batch.log

set -u

# conda base 활성 (batch job은 PATH 초기화되므로 명시)
source /scratch/home/wog3597/anaconda3/etc/profile.d/conda.sh
conda activate base

cd /scratch/home/wog3597/optim

# 범위는 qsub -v GPU=,START=,END= 로 넘긴다. 미지정 시 GPU0 / idx 0~241 전체.
bash run_all_pd.sh "${GPU:-0}" "${START:-0}" "${END:-241}"
