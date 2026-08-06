#!/bin/bash
# DBS sweep 을 독립 PBS job 으로 제출 (idx 2개를 GPU 1장에 packing).
#
# 제출:
#   qsub qsub_dbs_sweep.sh                    # 아래 IDXS 기본값(5,6) 으로 병렬 실행
#   qsub -q base_g qsub_dbs_sweep.sh          # H100(kitsg001) 로 보내고 싶을 때
#
# ⚠ 이 클러스터의 remote 큐는 `qsub -v IDXS=...` 를 거부한다("cannot send environment
#   with the job"). idx 를 바꾸려면 아래 IDXS 기본값 줄을 직접 편집할 것.
# 상태:   qstat -u wog3597
# 로그:   tail -f output_ppmi_pd/_dbs_sweep_pbs.log        # job 로그
#         tail -f output_ppmi_pd/dbs_sweep_log1pm/idx<N>/idx<N>.log   # idx별 로그
# 취소:   qdel <jobid>
# 재개:   resume 내장 — 완료된 조합(fc png 4개)은 skip 하므로 그냥 다시 qsub 하면 된다.
#
# ── 자원 산정 근거 ──────────────────────────────────────────────────
# sweep 프로세스 1개 = 정상 4.4GB, 피크 7.5GB(cuFFT work area + 자극 배열 할당 시).
# JOBS=2 → 피크 15GB. A10(23GB) 전용이면 여유, H100 NVL(94GB)이면 더 여유.
# JOBS=3 은 A10 에서 터진다 — 2026-07-29 로컬에서 jupyter 커널 5GB 상주분까지 겹쳐
# RESOURCE_EXHAUSTED 로 3개 전멸했다. 전용 GPU 라도 A10 에서 3 이상은 올리지 말 것.
#
#PBS -N dbs_sweep
#PBS -q remote
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /scratch/home/wog3597/optim/output_ppmi_pd/_dbs_sweep_pbs.log

set -u

# batch job 은 PATH 가 초기화되므로 conda 를 명시적으로 활성화.
source /scratch/home/wog3597/anaconda3/etc/profile.d/conda.sh
conda activate base

cd /scratch/home/wog3597/optim

IDXS="${IDXS:-5,6}"
JOBS="${JOBS:-2}"

echo "=== DBS sweep PBS start  idx=[$IDXS]  jobs=$JOBS  ($(date '+%F %T')) ==="
echo "    node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u dbs_sweep_456.py --idxs "$IDXS" --jobs "$JOBS"
rc=$?

echo "=== DBS sweep PBS end  rc=$rc  ($(date '+%F %T')) ==="
exit $rc
