#!/bin/bash
# c_ei 인과 테스트 모니터.  사용: bash mon_ceitest.sh   또는   watch -n 20 bash mon_ceitest.sh
# 100001 DBS(STN_R,GP_L @ amp1/125Hz)를 c_ei ×1.5/2/3 로 → Δ가 떨어지면 "높은 c_ei→무반응" 인과 확정.
cd /scratch/home/wog3597/optim
LOG=output_ppmi_pd/dbs_ceitest.log

echo "=== c_ei 인과테스트 모니터 ($(date '+%F %H:%M:%S')) ==="
if pgrep -f "dbs_from_cache.py --subject-idx 0 --cei-scale" >/dev/null 2>&1; then
    echo "상태: 실행중"
else
    echo "상태: 종료/대기"
fi
cur=$(grep -oE "cei-scale [0-9.]+x" "$LOG" 2>/dev/null | tail -1)
echo "현재: ${cur:-시작 전(warmup)}"
echo ""
echo "  [기준] c_ei ×1 (mean 0.23): STN_R Δ=+0.54  GP_L Δ=+0.51   ← 반응"
python3 - <<'PY'
import glob, pandas as pd, numpy as np
for s in ["1.5","2","3"]:
    parts=[]
    done=False
    for t in ["STN_R","GP_L"]:
        f=glob.glob(f"output_ppmi_pd/dbs_ceitest/cei{s}x/{t}/*/fc_summary.csv")
        if f:
            done=True
            d=pd.read_csv(f[0]).set_index('metric')['value']
            parts.append(f"{t} Δ={float(d.get('mean_diff_during_minus_pre',np.nan)):+.3f}")
        else:
            parts.append(f"{t} Δ=--")
    tag = "" if done else " (대기)"
    print(f"  c_ei ×{s} (mean {0.233*float(s):.2f}): " + "  ".join(parts) + tag)
PY
echo ""
grep -vE "No SoL" "$LOG" 2>/dev/null | tail -3
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"
exit 0
