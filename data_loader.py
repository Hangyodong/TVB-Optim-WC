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

    weights = np.log1p(weights + 0.5)
    weights_max = float(np.max(weights))
    if weights_max > 0:
        weights = weights / weights_max
    weights = weights * sc_mask

    delays = lengths / cfg.tract_conduction_speed

    weights, lengths, delays, fc_target = _cast_to_float32(
        weights, lengths, delays, fc_target
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
        "cache_tag":     cache_tag,
        "cache_dir":     cache_dir,
        "graph":         graph,
    }


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