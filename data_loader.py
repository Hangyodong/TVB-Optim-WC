"""
data_loader.py

Stage 3
-------
- DenseDelayGraph를 기본 graph로 반환한다.
- lengths / tract_conduction_speed 로 계산한 tract delay를 data["graph"]에 통합한다.
- build_network()가 data["graph"]를 그대로 사용할 수 있게 dict에 graph 키를 포함한다.
"""
import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.utils import set_cache_path

from config import Config


def load_data(cfg: Config) -> dict:
    """
    SC / tract_length / FC 데이터를 로드하고 전처리한다.

    Returns
    -------
    dict
        weights, lengths, delays, fc_target, sc_mask,
        region_labels, n_nodes, cache_tag, cache_dir, graph
    """
    sc_path     = _resolve_path(cfg.sc_csv,     "weight_compact.csv")
    length_path = _resolve_path(cfg.length_csv, "tract_length_compact.csv")
    fc_path     = _resolve_path(cfg.fc_csv,     "FC_compact.csv")

    region_labels = _load_region_labels(cfg.region_txt)
    weights, lengths, fc_target = _load_matrices(sc_path, length_path, fc_path)

    n_nodes = weights.shape[0]
    _validate_shapes(weights, lengths, fc_target, region_labels, n_nodes)

    np.fill_diagonal(weights, 0.0)
    sc_mask = (weights > 0).astype(np.float32)

    if sc_mask.sum() == 0:
        raise RuntimeError("SC has no positive edges to use as a sparse mask.")

    # SC 정규화 (cfg.sc_norm). "max"=원본 EI_Tuning w/max (heavy-tail 보존, 안정).
    # "log1p"=log1p(w+0.5)/max (약한 edge 부풀림 → 노드당 입력 21× → 과흥분/포화).
    # "log1pm"=log1p(w) (offset 없음, 0은 0 유지) 를 "max" 의 노드입력에 재스케일.
    #   약한 edge dynamic range 복원(median NZ 0.2%→~37%/max)하되 "log1p" 의 16~18×
    #   입력 팽창(FIC c_ei 상한 20 초과 → 과흥분)을 제거. hub 압축 → 노드입력 분포 좁아짐(더 안전).
    _norm = getattr(cfg, "sc_norm", "log1pm")
    if _norm == "max":
        weights_max = float(np.max(weights))
        if weights_max > 0:
            weights = weights / weights_max
    elif _norm == "log1pm":
        w_max_mean = (weights / float(np.max(weights))).sum(1).mean()  # "max" 노드입력 목표
        weights = np.log1p(weights) * sc_mask
        w_log_mean = weights.sum(1).mean()
        if w_log_mean > 0:
            weights = weights * (w_max_mean / w_log_mean)
    else:
        weights = np.log1p(weights + 0.5)
        weights_max = float(np.max(weights))
        if weights_max > 0:
            weights = weights / weights_max
    weights = weights * sc_mask
    print(f"[DATA] SC norm='{getattr(cfg, 'sc_norm', 'log1pm')}'  노드당 입력 mean={weights.sum(1).mean():.2f}")

    delays = lengths / cfg.tract_conduction_speed

    weights, lengths, delays, fc_target = _cast_to_float32(
        weights, lengths, delays, fc_target
    )

    cortex_idx, subcortex_idx = derive_cortex_subcortex_indices(region_labels)
    fc_edge_weight = _build_fc_edge_weight(
        n_nodes, cortex_idx, subcortex_idx,
        cfg.fc_block_share_cortex, cfg.fc_block_share_cross, cfg.fc_block_share_subsub,
    )
    fc_block_masks = _build_fc_block_masks(n_nodes, cortex_idx, subcortex_idx)
    print(
        f"[DATA] cortex={len(cortex_idx)}  subcortex={len(subcortex_idx)}  "
        f"FC block share ctx/cross/sub="
        f"{cfg.fc_block_share_cortex}/{cfg.fc_block_share_cross}/{cfg.fc_block_share_subsub}"
    )

    cache_tag = _build_cache_tag(cfg.cache_version, n_nodes, weights, fc_target)
    cache_dir = _create_cache_dir(cfg.cache_run_label, cache_tag)

    graph = DenseDelayGraph(
        weights.astype(np.float64),
        delays.astype(np.float64),
        region_labels=region_labels,
    )

    print(f"[DATA] n_nodes={n_nodes}  SC nonzero={int(sc_mask.sum())}")
    print(f"[DATA] weights:   min={weights.min():.4e}  max={weights.max():.4e}")
    print(f"[DATA] fc_target: min={fc_target.min():.4e}  max={fc_target.max():.4e}")
    print(
        f"[DATA] delays:    min={delays.min():.2f}ms  "
        f"max={delays.max():.2f}ms  "
        f"(speed={cfg.tract_conduction_speed} mm/ms)"
    )
    print(
        f"[DATA] SC mask:   {int(sc_mask.sum())} / {n_nodes * n_nodes}"
        f"  ({sc_mask.sum()/(n_nodes*n_nodes)*100:.1f}%)"
    )
    print(f"[DATA] cache_tag = {cache_tag}")
    print(f"[DATA] Graph type: {type(graph).__name__}")

    _plot_data_matrices(weights, delays, fc_target, cfg)

    return {
        "weights":       weights,
        "lengths":       lengths,
        "delays":        delays,
        "fc_target":     fc_target,
        "sc_mask":       sc_mask,
        "region_labels": region_labels,
        "n_nodes":       n_nodes,
        "cortex_indices":    cortex_idx,
        "subcortex_indices": subcortex_idx,
        "fc_edge_weight":    fc_edge_weight,
        "fc_block_masks":    fc_block_masks,
        "cache_tag":     cache_tag,
        "cache_dir":     cache_dir,
        "graph":         graph,
    }


# ── Cortex / Subcortex split + FC edge weight ────────────────────
# 라벨 prefix로 cortex 판정: human Schaefer = "7Networks_", mouse CHA = "Cortex_".
# 매칭 안 되면 subcortex로 분류(human 16개 PD25 구조명, mouse 피질하 등).
_CORTEX_LABEL_PREFIXES = ("7Networks_", "Cortex_")

# AAL3 (AAL3v1_1mm_PD25stn) 는 cortex prefix 규칙이 없다(Precentral_L, Frontal_Sup_2_L …).
# prefix 방식 그대로 두면 168개 전부 subcortex 로 떨어져 FC block 가중치가 무의미해지므로,
# subcortex 를 명시 집합으로 판정하고 나머지를 cortex 로 본다 (subcortex 52 / cortex 116).
# 포함: basal ganglia + thalamus 전체 + 뇌간 핵(VTA/SN/Red_N/LC/Raphe) + STN.
# 제외(=cortex 로 분류): Cerebellum·Vermis, Hippocampus, Amygdala, ACC_* — 사용자 결정.
_AAL3_SUBCORTEX_LABELS = frozenset({
    "Caudate_L", "Caudate_R", "Putamen_L", "Putamen_R",
    "Pallidum_L", "Pallidum_R", "N_Acc_L", "N_Acc_R",
    "Thal_AV_L", "Thal_AV_R", "Thal_LP_L", "Thal_LP_R",
    "Thal_VA_L", "Thal_VA_R", "Thal_VL_L", "Thal_VL_R",
    "Thal_VPL_L", "Thal_VPL_R", "Thal_IL_L", "Thal_IL_R",
    "Thal_Re_L", "Thal_Re_R", "Thal_MDm_L", "Thal_MDm_R",
    "Thal_MDl_L", "Thal_MDl_R", "Thal_LGN_L", "Thal_LGN_R",
    "Thal_MGN_L", "Thal_MGN_R", "Thal_PuA_L", "Thal_PuA_R",
    "Thal_PuM_L", "Thal_PuM_R", "Thal_PuL_L", "Thal_PuL_R",
    "Thal_PuI_L", "Thal_PuI_R",
    "VTA_L", "VTA_R", "SN_pc_L", "SN_pc_R", "SN_pr_L", "SN_pr_R",
    "Red_N_L", "Red_N_R", "LC_L", "LC_R", "Raphe_D", "Raphe_M",
    "STN_L", "STN_R",
})


def derive_cortex_subcortex_indices(region_labels):
    labels = [str(l).strip() for l in region_labels]
    # AAL3 판정: cortex prefix 가 하나도 없고 AAL3 subcortex 이름이 보이면 집합 방식으로 전환.
    # (Schaefer/mouse 는 prefix 가 잡히므로 기존 경로 그대로)
    if (not any(l.startswith(_CORTEX_LABEL_PREFIXES) for l in labels)
            and any(l in _AAL3_SUBCORTEX_LABELS for l in labels)):
        cortex    = [i for i, l in enumerate(labels) if l not in _AAL3_SUBCORTEX_LABELS]
        subcortex = [i for i, l in enumerate(labels) if l in _AAL3_SUBCORTEX_LABELS]
    else:
        cortex    = [i for i, l in enumerate(labels) if l.startswith(_CORTEX_LABEL_PREFIXES)]
        subcortex = [i for i, l in enumerate(labels) if not l.startswith(_CORTEX_LABEL_PREFIXES)]
    return (
        np.asarray(cortex, dtype=np.int32),
        np.asarray(subcortex, dtype=np.int32),
    )


def _build_fc_edge_weight(n_nodes, cortex_idx, subcortex_idx, s_ctx, s_cross, s_subsub):
    """블록 점유율(share)을 atlas의 블록별 edge수로 나눠 per-edge 가중 W(diag 0) 생성.
    per-edge weight ∝ share / n_edges(block). cortex-cortex per-edge를 1.0 기준으로
    정규화(가독성; weighted corr/rmse는 ΣW 정규화라 전체 스케일 무관).
    - 점유율이 edge수 비례면 W가 균일 → (1-eye)와 동치(off).
    - subcortex 구분 없는 atlas(블록 1개)는 균일 → 자동 off."""
    ci = np.asarray(cortex_idx, dtype=np.int64)
    si = np.asarray(subcortex_idx, dtype=np.int64)
    nc, ns = ci.size, si.size
    n_cc = nc * (nc - 1) // 2          # off-diagonal edge 수
    n_cross = nc * ns
    n_ss = ns * (ns - 1) // 2

    def _per_edge(share, n_edges):
        return (float(share) / n_edges) if n_edges > 0 else 0.0

    w_cc = _per_edge(s_ctx, n_cc)
    w_cross = _per_edge(s_cross, n_cross)
    w_ss = _per_edge(s_subsub, n_ss)

    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if nc:
        W[np.ix_(ci, ci)] = w_cc
    if nc and ns:
        W[np.ix_(ci, si)] = w_cross
        W[np.ix_(si, ci)] = w_cross
    if ns:
        W[np.ix_(si, si)] = w_ss
    np.fill_diagonal(W, 0.0)

    # cortex-cortex per-edge=1.0 기준 정규화(없으면 최대값 기준).
    norm = w_cc if w_cc > 0 else (W.max() if W.max() > 0 else 1.0)
    return (W / norm).astype(np.float32)


def _build_fc_block_masks(n_nodes, cortex_idx, subcortex_idx):
    """블록 corr loss용 off-diag indicator mask 3개(cc / cross / sub-sub).
    block-split corr는 각 블록을 edge수 무관 동등 가중 → subcortex fit 균형.
    subcortex 없는 atlas는 cross/ss가 전부 0 → cc==full off-diag로 whole corr fallback."""
    ci = np.asarray(cortex_idx, dtype=np.int64)
    si = np.asarray(subcortex_idx, dtype=np.int64)
    eye = np.eye(n_nodes, dtype=np.float32)

    cc = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    cross = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    ss = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if ci.size:
        cc[np.ix_(ci, ci)] = 1.0
    if ci.size and si.size:
        cross[np.ix_(ci, si)] = 1.0
        cross[np.ix_(si, ci)] = 1.0
    if si.size:
        ss[np.ix_(si, si)] = 1.0
    # diag 제거(weighted corr은 off-diag edge 기준).
    cc *= (1.0 - eye)
    ss *= (1.0 - eye)
    return {"cc": cc, "cross": cross, "subsub": ss}


# ── 시각화 ────────────────────────────────────────────────────

def _plot_data_matrices(
    weights:   np.ndarray,
    delays:    np.ndarray,
    fc_target: np.ndarray,
    cfg:       Config,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 4))

    image_w = axes[0].imshow(weights, cmap="RdBu_r", vmin=0, vmax=1)
    axes[0].set_title("Structural Weights")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Region")
    plt.colorbar(image_w, ax=axes[0], fraction=0.046)

    image_d = axes[1].imshow(delays, cmap="viridis")
    axes[1].set_title("Tract Delays (ms)")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Region")
    plt.colorbar(image_d, ax=axes[1], fraction=0.046, label="ms")

    image_fc = axes[2].imshow(
        fc_target,
        vmin=cfg.fc_plot_vmin, vmax=cfg.fc_plot_vmax,
        cmap="RdBu_r",
    )
    axes[2].set_title("Target Functional Connectivity")
    axes[2].set_xlabel("Region")
    axes[2].set_ylabel("Region")
    plt.colorbar(image_fc, ax=axes[2], label="Correlation", fraction=0.046)

    plt.tight_layout()
    plt.show()


# ── 내부 헬퍼 ─────────────────────────────────────────────────

def _resolve_path(preferred: str, fallback: str) -> str:
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        print(f"[DATA] {preferred} not found → using {fallback}")
        return fallback
    return preferred


def _load_region_labels(region_txt_path: str) -> list:
    """Load region labels supporting three formats:
    1. Plain:       one label per line
    2. TSV:         tab-separated with header containing "name" column
    3. Index+space: "<int> <label_name>" (e.g. Schaefer/Atlas style)
    encoding="utf-8-sig" strips any leading BOM.
    """
    labels = []
    with open(region_txt_path, "r", encoding="utf-8-sig") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    if not lines:
        return labels
    first = lines[0]
    # Format 2: TSV with header containing "name" column
    if "\t" in first and "name" in first.lower():
        cols = first.split("\t")
        try:
            name_idx = cols.index("name")
        except ValueError:
            name_idx = -1
        for line in lines[1:]:
            parts = line.split("\t")
            if name_idx >= 0 and name_idx < len(parts):
                labels.append(parts[name_idx].strip())
            else:
                labels.append(line)
    # Format 3: "<int> <label>" space-separated
    elif " " in first:
        parts0 = first.split(None, 1)
        if len(parts0) == 2 and parts0[0].isdigit():
            for line in lines:
                p2 = line.split(None, 1)
                if len(p2) == 2 and p2[0].isdigit():
                    labels.append(p2[1].strip())
                elif p2:
                    labels.append(p2[-1].strip())
        else:
            labels = list(lines)
    # Format 1: plain labels
    else:
        labels = list(lines)
    return labels


def _load_matrices(sc_path: str, length_path: str, fc_path: str) -> tuple:
    weights   = pd.read_csv(sc_path,     header=None).to_numpy(dtype=np.float64)
    lengths   = pd.read_csv(length_path, header=None).to_numpy(dtype=np.float64)
    fc_target = pd.read_csv(fc_path,     header=None).to_numpy(dtype=np.float64)
    return weights, lengths, fc_target


def _validate_shapes(
    weights: np.ndarray,
    lengths: np.ndarray,
    fc_target: np.ndarray,
    region_labels: list,
    n_nodes: int,
) -> None:
    assert weights.shape   == (n_nodes, n_nodes), f"SC shape mismatch: {weights.shape}"
    assert lengths.shape   == (n_nodes, n_nodes), f"Length shape mismatch: {lengths.shape}"
    assert fc_target.shape == (n_nodes, n_nodes), f"FC shape mismatch: {fc_target.shape}"
    assert len(region_labels) == n_nodes, (
        f"Region label count mismatch: {len(region_labels)} vs {n_nodes}"
    )


def _cast_to_float32(*arrays: np.ndarray) -> tuple:
    return tuple(np.asarray(arr, dtype=np.float32) for arr in arrays)


def _build_cache_tag(
    cache_version: str,
    n_nodes: int,
    weights: np.ndarray,
    fc_target: np.ndarray,
) -> str:
    def _fingerprint(matrix: np.ndarray) -> str:
        sample = matrix[:min(8, matrix.shape[0]), :min(8, matrix.shape[1])]
        return hashlib.sha1(sample.tobytes()).hexdigest()[:10]

    return (
        f"{cache_version}_N{n_nodes}_"
        f"sc{_fingerprint(weights)}_"
        f"fc{_fingerprint(fc_target)}"
    )


def _create_cache_dir(cache_run_label: str, cache_tag: str) -> str:
    # 라이브러리(set_cache_path)가 optim/cache 를 root로 잡고 experiment 하위에 저장한다.
    # experiment = "<cache_run_label>/<cache_tag>" → optim/cache/<label>/<tag>
    experiment = os.path.join(cache_run_label, cache_tag)
    cache_path = set_cache_path(experiment)   # 실제 절대 저장 경로 반환
    os.makedirs(cache_path, exist_ok=True)
    return cache_path