#!/bin/bash
# 도는 워커 전부 멈추고 idx 2,3만 재실행 (PYTHONUNBUFFERED=1 → step 실시간 로그).
#   ★ H100 PBS 쉘(KITSG003)서 실행: bash restart_23.sh
#   (A10서 실행하면 A10서 돎 — H100 원하면 반드시 base_g 쉘에서)
cd /scratch/home/wog3597/optim

echo "== 1) 도는 optim 워커 전부 종료 =="
pkill -9 -f "python3 main_ppmi_pd.py --subject-idx" 2>/dev/null && echo "  killed" || echo "  없음"
sleep 3

echo "== 2) idx 2,3 재실행 (unbuffered, FIC/EIB 캐시-hit → Part3부터) =="
mkdir -p output_ppmi_pd/_par_logs
for i in 2 3; do
    nohup env PYTHONUNBUFFERED=1 PART3_REMAT_SCAN=1 \
        python3 main_ppmi_pd.py --subject-idx "$i" \
        > output_ppmi_pd/_par_logs/s${i}.log 2>&1 &
    echo "  idx $i → PID $!"
done

echo ""
echo "== 3) 모니터 (10초 갱신, Ctrl-C로 모니터만 종료 — optim은 계속) =="
sleep 4
watch -n 10 bash mon_par.sh "2 3"
