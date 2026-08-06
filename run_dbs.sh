#!/bin/bash
# Part4 DBS 실행: STN_L / STN_R / Pallidum_L(GP_L) / Pallidum_R(GP_R) 각각 자극 (총 4케이스, 순차).
#   - 자극 조건: config 기본 유지 (진폭 1.0nA / 130Hz / biphasic).
#   - pre(transient/baseline)·during 지속시간 = 시뮬레이션 시간(600s, make_config 에서 설정).
#   - FIC/EIB/Part3 는 이미 완료된 subject 면 캐시 hit → DBS 만 새로 실행.
#
# 사용:
#   bash run_dbs.sh              # idx 0,1 (sub 100001, 100005)
#   bash run_dbs.sh 0 1 2        # 원하는 subject idx
#   nohup bash run_dbs.sh 0 1 > output_ppmi_pd/dbs_batch.log 2>&1 &   # 백그라운드
#
# 출력 (subject·타깃별):
#   output_ppmi_pd/<sub_num>/dbs_analysis/{STN_L,STN_R,GP_L,GP_R}/
#     fc_pre_during_diff.png   ← 자극 전 / 중 / 차 FC matrix figure
#     fc_pre_stim.csv , fc_during_stim.csv , fc_diff_during_minus_pre.csv , fc_summary.csv

cd /scratch/home/wog3597/optim

if [ $# -eq 0 ]; then IDXS="0 1"; else IDXS="$*"; fi

for i in $IDXS; do
    echo "=========================================================="
    echo "=== [$(date '+%F %T')] DBS  subject idx $i  시작 ==="
    echo "=========================================================="
    python3 main_ppmi_pd.py --subject-idx "$i" --enable-dbs
    rc=$?
    echo "=== [$(date '+%F %T')] idx $i  DBS 종료 (rc=$rc) ==="
    if [ $rc -ne 0 ]; then echo "  [FAIL] idx $i 실패 → 로그 확인"; fi
done

echo ""
echo "완료. 결과: output_ppmi_pd/<sub_num>/dbs_analysis/<TARGET>/fc_pre_during_diff.png"
