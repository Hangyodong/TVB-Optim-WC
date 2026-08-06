#!/bin/bash
# optim 실행 가능 견적 (RTX 5060 Ti / WSL 등). optim 폴더서: bash check_env.sh
echo "════════════ optim 환경 견적 ($(date '+%F %H:%M')) ════════════"

echo ""; echo "## 1. GPU ##"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null \
  || echo "  ✗ nvidia-smi 없음 (GPU 인식 안 됨)"

echo ""; echo "## 2. JAX GPU 지원 (★ 핵심 — Blackwell sm_120) ##"
python3 - <<'PY'
try:
    import jax, jax.numpy as jnp
    print(f"  jax {jax.__version__}")
    try:
        import jaxlib; print(f"  jaxlib {jaxlib.__version__}")
    except Exception: pass
    be = jax.default_backend()
    print(f"  backend = {be}   " + ("✓ GPU" if be=="gpu" else "✗ CPU → GPU 못 씀!"))
    print(f"  devices = {jax.devices()}")
    try:
        import jax.numpy as jnp
        x = jnp.ones((2048, 2048))
        y = float((x @ x).sum())   # 실제 GPU 커널 실행
        print(f"  ✓ GPU matmul OK (sum={y:.3e}) → sm_120 커널 정상")
    except Exception as e:
        print(f"  ✗ matmul 실패: {str(e)[:90]}")
        print(f"    → Blackwell 미지원 의심. jaxlib 업그레이드 필요.")
except Exception as e:
    print(f"  ✗ jax import 실패: {str(e)[:90]}")
PY

echo ""; echo "## 3. 필요 패키지 ##"
python3 - <<'PY'
for p in ["numpy","scipy","matplotlib","pandas","equinox","optax","jax","tvboptim"]:
    try:
        m = __import__(p); print(f"  ✓ {p:12s} {getattr(m,'__version__','?')}")
    except Exception:
        print(f"  ✗ {p:12s} 없음")
PY

echo ""; echo "## 4. 코드 파일 ##"
miss=0
for f in main_ppmi_pd.py config.py data_loader.py model.py part1_fic.py part2_eib.py part3_gradient.py pipeline_contracts.py remat_scan_patch.py; do
  [ -f "$f" ] && echo "  ✓ $f" || { echo "  ✗ $f 없음"; miss=1; }
done

echo ""; echo "## 5. 데이터 ##"
[ -f data/AALv3/FC_AAL_ComBat_all_163.mat ] && echo "  ✓ FC_AAL_ComBat_all_163.mat" || { echo "  ✗ FC_AAL_ComBat_all_163.mat 없음"; miss=1; }
[ -f data/AALv3/AAL163_labels.txt ] && echo "  ✓ AAL163_labels.txt" || echo "  ✗ AAL163_labels.txt 없음"
ls data/MDS-UPDRS*.csv >/dev/null 2>&1 && echo "  ✓ UPDRS csv" || echo "  (UPDRS csv 없음 — optim엔 불필요, 분석용)"

echo ""; echo "════════════ 판정 ════════════"
echo "  ✓ backend=gpu + matmul OK + 패키지 O + 코드/데이터 O  → 돌아감 (1개, PART3_REMAT_SCAN=1)"
echo "  ✗ backend=cpu / matmul 실패  → pip install -U 'jax[cuda12]'  (Blackwell 지원 버전)"
[ "$miss" = 1 ] && echo "  ✗ 코드/데이터 누락  → 클러스터(/scratch/home/wog3597/optim)서 복사"
echo ""
echo "  실행 테스트(GPU OK 확인 후): PART3_REMAT_SCAN=1 python3 main_ppmi_pd.py --subject-idx 0"
