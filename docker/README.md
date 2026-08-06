# 컨테이너 이미지

`./docker/build-nodaemon.sh` 하나면 `docker load` 가능한 tar 가 나온다.
도커 데몬이 필요 없어서 공유 HPC 노드에서도 돌아간다.

```bash
curl -sSL https://github.com/google/go-containerregistry/releases/download/v0.20.2/go-containerregistry_Linux_x86_64.tar.gz | tar xz crane
PATH="$PWD:$PATH" ./docker/build-nodaemon.sh

docker load -i optim-image.tar
docker run --rm --gpus all -v "$PWD/out:/opt/optim/output_ppmi_pd" \
  local/optim:latest python3 main.py --subject-idx 4
```

## 왜 Dockerfile 이 아닌가

이 코드가 도는 노드는 `/var/run/docker.sock` 이 `root:docker 0660` 이고 계정이 docker
그룹에 없다. rootless docker 도 막힌다 — rootlesskit 은 subuid/subgid 범위 없이는
시작하지 않고, `/etc/subuid` 를 쓰려면 root 가 필요한데 그게 없는 상황이다.
podman/buildah/apptainer 도 설치돼 있지 않다.

그래서 이 스크립트는 컨테이너를 한 번도 실행하지 않는다. 호스트 파이썬이 베이스
`python:3.13-slim`(python 3.13.14)과 같은 cp313/manylinux x86_64 라 휠 태그가 일치하고,
따라서 `pip install --target` 으로 받은 site-packages 를 그대로 레이어로 얹을 수 있다.
이미지 조립은 `crane`(정적 바이너리)이 한다.

Dockerfile 을 쓸 수 있는 환경이라면 그쪽이 낫다. 이건 어디까지나 우회책이다.

## 내용물과 크기

베이스 `python:3.13-slim` + 의존성 + `/opt/optim`(코드) + `/opt/optim/data/AALv3`(478MB).
결과 ~3.9GB, 압축 해제 ~6.0GB. CUDA 는 `nvidia-*-cu12` pip 휠로 이미지 안에 들어가므로
호스트엔 **NVIDIA 드라이버 + nvidia-container-toolkit** 만 있으면 된다.

`WORKDIR /opt/optim`, `Cmd ["python3"]`, Entrypoint 없음.

## 함정

- **스테이징을 NFS 에 두지 말 것.** `pip install --target` 이 수만 개 작은 파일을 푸는데,
  NFS 에서는 67분이 걸렸고 그중 CPU 시간은 1분이었다(나머지는 D 상태 I/O 대기).
  노드 로컬 디스크에서는 같은 일이 7분. 스크립트가 `/var/tmp` 를 쓰는 이유다.
- **`crane mutate` 에 `--repo` 와 `-t` 를 같이 주면 거부한다**
  (`repository can't be set when a tag is specified`). `-t` 만 쓸 것.
- `--only-binary=:all:` 를 빼면 소스 빌드로 빠질 수 있는데, 베이스 이미지에는 컴파일러가
  없으므로 그렇게 만든 이미지는 못 쓴다.

## 데몬 없이 검증하기

레이어 sha256 이 config 의 `rootfs.diff_ids` 와 맞는지 보면 `docker load` 가능 여부를
알 수 있다. 실제 동작은 레이어를 풀어 user namespace + chroot 로 확인한다.

```bash
mkdir rootfs && for l in $(...); do tar -xzf $l -C rootfs; done
mkdir -p nvlib && cp -L /lib64/libcuda.so.1 nvlib/     # /lib64 통째로 넣으면 glibc 를 덮어써서 아무것도 안 뜬다
unshare -U -r -m bash -c '
  mount --make-rprivate /
  for d in dev proc sys; do mount --rbind /$d rootfs/$d; done
  mount --rbind $PWD/nvlib rootfs/nvlib
  chroot rootfs env LD_LIBRARY_PATH=/nvlib /usr/local/bin/python3 -c "import jax; print(jax.devices())"'
```

2026-08-06 검증: 레이어 5개 diff_id 전부 일치, chroot 임포트 전부 통과,
`jax.devices()` → `[CudaDevice(id=0)]`.
