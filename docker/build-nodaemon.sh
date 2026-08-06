#!/usr/bin/env bash
# 도커 데몬 없이 이 파이프라인의 이미지를 만든다.
#
#   ./docker/build-nodaemon.sh                  # 기본값으로 빌드
#   OUT=/path/img.tar ./docker/build-nodaemon.sh
#
# 공유 HPC 노드에서는 /var/run/docker.sock 이 root:docker 0660 이고 계정이 docker
# 그룹에 없어 `docker build` 가 불가능하다. rootless docker 도 답이 아니다 —
# rootlesskit 은 subuid/subgid 없이는 안 뜨고, /etc/subuid 를 쓰려면 root 가 필요하다.
#
# 그래서 컨테이너를 한 번도 실행하지 않는다. 호스트 파이썬이 베이스 이미지와 같은
# cp313/manylinux x86_64 라, pip 이 받은 site-packages 를 그대로 레이어로 얹으면 된다.
# 이미지 조립은 crane(go-containerregistry 정적 바이너리, 데몬·root 불필요)이 한다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-python:3.13-slim}"
TAG="${TAG:-local/optim:latest}"
OUT="${OUT:-$REPO/optim-image.tar}"

# 스테이징은 반드시 노드 로컬 디스크에. NFS 에 잡으면 pip 이 수만 개 작은 파일을
# 푸느라 한 시간을 D 상태로 보낸다(실측 67분 중 CPU 시간 1분). 로컬은 같은 일이 7분.
WORK="${WORK:-/var/tmp/optim-img-build}"
SITE="$WORK/rootfs/usr/local/lib/python3.13/site-packages"

command -v crane >/dev/null || {
    echo "crane 이 PATH 에 없다. 받아서 쓸 것:" >&2
    echo "  curl -sSL https://github.com/google/go-containerregistry/releases/download/v0.20.2/go-containerregistry_Linux_x86_64.tar.gz | tar xz crane" >&2
    exit 1
}

rm -rf "$WORK"; mkdir -p "$SITE" "$WORK/rootfs/opt/optim/data"

echo "[1/4] 의존성 설치 → $SITE"
# 베이스가 python 3.13.14 라 휠 태그(cp313)가 호스트와 일치한다. 이게 이 방식의 전제다.
pip install --no-cache-dir --only-binary=:all: --target "$SITE" \
    "tvboptim==0.2.6" "jax[cuda12]==0.7.2" "scipy==1.17.1" pillow pandas matplotlib

echo "[2/4] 코드·데이터 복사"
cp "$REPO"/*.py "$REPO"/*.ipynb "$WORK/rootfs/opt/optim/"
cp "$REPO"/requirements_ppmi_pd.txt "$WORK/rootfs/opt/optim/" 2>/dev/null || true
cp -r "$REPO/data/AALv3" "$WORK/rootfs/opt/optim/data/"   # Schaeffer 는 이 파이프라인이 안 쓴다

echo "[3/4] 레이어 tar"
find "$WORK/rootfs" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
tar -C "$WORK/rootfs" -cf "$WORK/layer.tar" .

echo "[4/4] 이미지 조립 → $OUT"
# --repo 와 -t 를 동시에 주면 crane 이 거부한다. -t 만 쓴다.
crane mutate "$BASE_IMAGE" --append "$WORK/layer.tar" --workdir /opt/optim -t "$TAG" -o "$OUT"

rm -rf "$WORK"
echo "완료: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "  docker load -i $OUT"
echo "  docker run --rm --gpus all $TAG python3 main.py --subject-idx 4"
