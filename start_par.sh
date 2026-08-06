#!/bin/bash
# idx 0~3 병렬 실행 + 디테일 모니터 즉시 출력.
#   H100 PBS 쉘(KITSG003)에서: bash start_par.sh
#   watch Ctrl-C 해도 optim은 nohup으로 계속 돎.
cd /scratch/home/wog3597/optim
echo "== 옛 값 idx0 정리 =="
pkill -f "main_ppmi_pd.py --subject-idx 0" 2>/dev/null && echo "  killed" || echo "  없음"
mkdir -p output_ppmi_pd/_par_logs
echo "== idx 0~3 병렬 실행 (max-norm, EIB LR 0.1/0.005, 캡10) =="
for i in 0 1 2 3; do
    nohup python3 main_ppmi_pd.py --subject-idx "$i" \
        > output_ppmi_pd/_par_logs/s${i}.log 2>&1 &
    echo "  idx $i → PID $!"
done
echo ""
echo "== 3초 후 디테일 모니터 (15초 갱신, Ctrl-C로 모니터만 종료) =="
sleep 3
watch -n 15 bash mon_par.sh
