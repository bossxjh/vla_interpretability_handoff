from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import regression_metrics
from .utils import ensure_dir


DEFAULT_DYNAMIC_TARGETS = (
    "pickup_offset",
    "place_offset",
    "action",
    "policy_pred_action",
    "progress",
)


@dataclass
class FullTokenRolloutDataset:
    rollout_dir: Path
    samples: pd.DataFrame
    activations: pd.DataFrame
    layer_names: list[str]
    summary: dict[str, Any]


def analyze_pi0_dynamic_circuit(
    rollout_dir: Path,
    output_dir: Path,
    targets: tuple[str, ...] = DEFAULT_DYNAMIC_TARGETS,
    pooling: str = "mean",
    alpha: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
    top_k_layers: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    dataset = load_full_token_rollout_dataset(rollout_dir)
    ensure_dir(output_dir)
    dataset.samples.to_csv(output_dir / "pi0_dynamic_samples.csv", index=False)
    dataset.activations.to_csv(output_dir / "pi0_dynamic_activation_manifest.csv", index=False)
    write_json(output_dir / "pi0_dynamic_metadata.json", dataset.summary)
    token_layout_path = rollout_dir / "token_layout.json"
    if token_layout_path.exists():
        write_json(output_dir / "token_layout.json", read_json(token_layout_path))

    norm_summary = summarize_activation_norms(dataset)
    norm_summary.to_csv(output_dir / "pi0_dynamic_activation_norms.csv", index=False)
    plot_activation_norm_heatmap(
        norm_summary,
        output_dir / "pi0_dynamic_activation_norm_heatmap_raw.png",
        normalize="none",
        title="PI0 full-token activation norm over rollout time",
    )
    plot_activation_norm_heatmap(
        norm_summary,
        output_dir / "pi0_dynamic_activation_norm_heatmap_layer_zscore.png",
        normalize="layer_zscore",
        title="PI0 activation norm over rollout time, normalized within each layer",
    )
    plot_episode_activation_norm_heatmaps(norm_summary, output_dir / "episode_heatmaps")

    curves: dict[str, pd.DataFrame] = {}
    sample_predictions: dict[str, pd.DataFrame] = {}
    for target in targets:
        y = target_matrix(dataset.samples, target)
        results, predictions = train_dynamic_layerwise_probe(
            dataset=dataset,
            y=y,
            target=target,
            pooling=pooling,
            alpha=alpha,
            test_size=test_size,
            seed=seed,
        )
        results.to_csv(output_dir / f"pi0_dynamic_probe_{target}.csv", index=False)
        predictions.to_csv(output_dir / f"pi0_dynamic_probe_{target}_sample_predictions.csv", index=False)
        curves[target] = results
        sample_predictions[target] = predictions

    probe_summary = summarize_probe_curves(curves)
    probe_summary.to_csv(output_dir / "pi0_dynamic_probe_summary.csv", index=False)
    plot_probe_curves(curves, output_dir / "pi0_dynamic_probe_r2_curves.png")

    nodes, edges = build_candidate_circuit_graph(curves, dataset.layer_names, top_k_layers=top_k_layers)
    nodes.to_csv(output_dir / "pi0_dynamic_circuit_nodes.csv", index=False)
    edges.to_csv(output_dir / "pi0_dynamic_circuit_edges.csv", index=False)
    write_circuit_graph_json(nodes, edges, output_dir / "pi0_dynamic_circuit_graph.json")
    plot_candidate_circuit(nodes, edges, output_dir / "pi0_dynamic_circuit_graph.png")
    return nodes, edges, curves


def load_full_token_rollout_dataset(rollout_dir: Path) -> FullTokenRolloutDataset:
    summary_path = rollout_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"rollout_dir": str(rollout_dir)}
    episode_dirs = sorted(path for path in rollout_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No `episode_*` directories found under {rollout_dir}.")

    sample_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        steps_path = episode_dir / "steps.jsonl"
        index_path = episode_dir / "activation_index.jsonl"
        if not steps_path.exists():
            continue
        steps = read_jsonl(steps_path)
        activation_index = read_jsonl(index_path) if index_path.exists() else []
        activation_steps = {int(row["step"]) for row in activation_index}
        for row in steps:
            sample_rows.append(flatten_step_row(row, episode_dir, has_activation=int(row["step"]) in activation_steps))
        for row in activation_index:
            activation_rows.append(flatten_activation_row(row, episode_dir))

    if not sample_rows or not activation_rows:
        raise ValueError(f"No usable steps/activation index found under {rollout_dir}.")
    samples = pd.DataFrame(sample_rows).sort_values(["episode_index", "step"]).reset_index(drop=True)
    activations = pd.DataFrame(activation_rows).sort_values(["episode_index", "step", "layer_index"]).reset_index(drop=True)
    layer_names = (
        activations[["layer_index", "layer_name"]]
        .drop_duplicates()
        .sort_values("layer_index")["layer_name"]
        .astype(str)
        .tolist()
    )
    summary = {
        **summary,
        "rollout_dir": str(rollout_dir),
        "num_step_rows": int(len(samples)),
        "num_activation_files": int(len(activations)),
        "num_layers_in_manifest": int(len(layer_names)),
        "layer_names_from_manifest": layer_names,
    }
    return FullTokenRolloutDataset(rollout_dir=rollout_dir, samples=samples, activations=activations, layer_names=layer_names, summary=summary)


def summarize_activation_norms(dataset: FullTokenRolloutDataset) -> pd.DataFrame:
    rows = []
    episode_success = dataset.samples.groupby("episode_index")["success"].max().to_dict()
    episode_num_steps = dataset.samples.groupby("episode_index")["step"].max().add(1).to_dict()
    for row in tqdm(dataset.activations.to_dict("records"), desc="summarize activation norms"):
        array = np.load(row["path"], mmap_mode="r")
        token_norm = np.linalg.norm(np.asarray(array, dtype=np.float32), axis=-1)
        episode_index = int(row["episode_index"])
        step = int(row["step"])
        rows.append(
            {
                "episode_index": episode_index,
                "step": step,
                "episode_success": bool(episode_success.get(episode_index, False)),
                "episode_progress": float(step) / float(max(int(episode_num_steps.get(episode_index, 1)) - 1, 1)),
                "layer_index": int(row["layer_index"]),
                "layer_name": str(row["layer_name"]),
                "num_tokens": int(array.shape[0]) if array.ndim >= 2 else 1,
                "hidden_dim": int(array.shape[-1]),
                "norm_mean": float(np.mean(token_norm)),
                "norm_std": float(np.std(token_norm)),
                "norm_max": float(np.max(token_norm)),
                "norm_argmax_token": int(np.argmax(token_norm)),
            }
        )
    return pd.DataFrame(rows)


def train_dynamic_layerwise_probe(
    dataset: FullTokenRolloutDataset,
    y: np.ndarray,
    target: str,
    pooling: str,
    alpha: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pooling not in ("mean", "last", "flatten", "token_norms"):
        raise ValueError(f"Unsupported pooling `{pooling}`.")
    samples = dataset.samples[dataset.samples["has_activation"].astype(bool)].copy()
    if len(samples) != len(y):
        y = y[samples.index.to_numpy()]
    if len(samples) < 4:
        raise ValueError(f"Need at least 4 replan samples with activations, got {len(samples)}.")
    train_idx, test_idx = train_test_split(np.arange(len(samples)), test_size=test_size, random_state=seed)
    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError(f"Need at least 2 train/test samples, got train={len(train_idx)}, test={len(test_idx)}.")

    result_rows = []
    prediction_rows = []
    for layer_index, layer_name in tqdm(
        list(enumerate(dataset.layer_names)),
        desc=f"train dynamic probes ({target}, {pooling})",
    ):
        x = load_layer_feature_matrix(dataset, samples, layer_index, pooling)
        model = Ridge(alpha=alpha)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        metrics = regression_metrics(y[test_idx], pred)
        result_rows.append(
            {
                "target": target,
                "pooling": pooling,
                "layer": layer_index,
                "layer_name": layer_name,
                **{key: metrics.get(key, np.nan) for key in ["r2_x", "r2_y", "r2_z", "r2_mean", "mse", "cos_sim"]},
            }
        )
        all_pred = model.predict(x)
        sample_meta = samples[["episode_index", "step"]].reset_index(drop=True)
        for sample_i, meta in sample_meta.iterrows():
            prediction_rows.append(
                {
                    "target": target,
                    "pooling": pooling,
                    "layer": layer_index,
                    "layer_name": layer_name,
                    "episode_index": int(meta["episode_index"]),
                    "step": int(meta["step"]),
                    "split": "test" if sample_i in set(test_idx.tolist()) else "train",
                    "prediction": all_pred[sample_i].astype(float).tolist(),
                    "ground_truth": y[sample_i].astype(float).tolist(),
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(prediction_rows)


def load_layer_feature_matrix(
    dataset: FullTokenRolloutDataset,
    samples: pd.DataFrame,
    layer_index: int,
    pooling: str,
) -> np.ndarray:
    features = []
    indexed = dataset.activations[dataset.activations["layer_index"].astype(int) == layer_index].copy()
    path_by_key = {
        (int(row["episode_index"]), int(row["step"])): row["path"]
        for row in indexed.to_dict("records")
    }
    for row in samples.to_dict("records"):
        key = (int(row["episode_index"]), int(row["step"]))
        if key not in path_by_key:
            raise KeyError(f"Missing activation for episode={key[0]}, step={key[1]}, layer={layer_index}.")
        array = np.asarray(np.load(path_by_key[key], mmap_mode="r"), dtype=np.float32)
        features.append(pool_activation(array, pooling))
    max_width = max(len(feature) for feature in features)
    padded = [
        np.pad(feature, (0, max_width - len(feature)), mode="constant") if len(feature) < max_width else feature
        for feature in features
    ]
    return np.stack(padded, axis=0).astype(np.float32)


def pool_activation(array: np.ndarray, pooling: str) -> np.ndarray:
    if array.ndim == 1:
        return array.astype(np.float32)
    if pooling == "mean":
        return array.mean(axis=0).astype(np.float32)
    if pooling == "last":
        return array[-1].astype(np.float32)
    if pooling == "flatten":
        return array.reshape(-1).astype(np.float32)
    if pooling == "token_norms":
        return np.linalg.norm(array.astype(np.float32), axis=-1)
    raise ValueError(f"Unsupported pooling `{pooling}`.")


def target_matrix(samples: pd.DataFrame, target: str) -> np.ndarray:
    rows = samples[samples["has_activation"].astype(bool)]
    if target == "pickup_offset":
        cols = ["pickup_offset_x", "pickup_offset_y", "pickup_offset_z"]
    elif target == "place_offset":
        cols = ["place_offset_x", "place_offset_y", "place_offset_z"]
    elif target == "action":
        cols = [col for col in rows.columns if col.startswith("action_")]
    elif target == "policy_pred_action":
        cols = [col for col in rows.columns if col.startswith("policy_pred_action_")]
    elif target == "progress":
        values = []
        for _, row in rows.iterrows():
            episode_steps = samples[samples["episode_index"] == row["episode_index"]]["step"].max()
            values.append([float(row["step"]) / float(max(int(episode_steps), 1))])
        return np.asarray(values, dtype=np.float32)
    else:
        raise KeyError(f"Unknown dynamic target `{target}`. Available targets: {DEFAULT_DYNAMIC_TARGETS}.")
    if not cols:
        raise KeyError(f"No columns found for target `{target}`.")
    return rows[cols].to_numpy(dtype=np.float32)


def summarize_probe_curves(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for target, frame in curves.items():
        best = frame.sort_values("r2_mean", ascending=False).iloc[0]
        rows.append(
            {
                "target": target,
                "pooling": best["pooling"],
                "best_layer": int(best["layer"]),
                "best_layer_name": best["layer_name"],
                "best_r2_mean": float(best["r2_mean"]),
                "best_mse": float(best["mse"]),
                "best_cos_sim": float(best["cos_sim"]),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_circuit_graph(
    curves: dict[str, pd.DataFrame],
    layer_names: list[str],
    top_k_layers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    for target, frame in curves.items():
        ranked = frame.sort_values("r2_mean", ascending=False).head(top_k_layers)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            node_rows.append(
                {
                    "node_id": f"{target}:L{int(row.layer):02d}",
                    "target": target,
                    "rank": rank,
                    "layer": int(row.layer),
                    "layer_name": str(row.layer_name),
                    "score": float(row.r2_mean),
                    "mse": float(row.mse),
                    "cos_sim": float(row.cos_sim),
                }
            )
    nodes = pd.DataFrame(node_rows)
    edge_rows = []
    for target, group in nodes.groupby("target"):
        ordered = group.sort_values("layer")
        records = ordered.to_dict("records")
        for source, dest in zip(records[:-1], records[1:]):
            edge_rows.append(
                {
                    "source": source["node_id"],
                    "target": dest["node_id"],
                    "target_variable": target,
                    "source_layer": int(source["layer"]),
                    "target_layer": int(dest["layer"]),
                    "edge_type": "same_target_layer_order",
                    "score": float(min(source["score"], dest["score"])),
                }
            )
    edges = pd.DataFrame(edge_rows)
    return nodes, edges


def plot_activation_norm_heatmap(
    norms: pd.DataFrame,
    path: Path,
    normalize: str = "none",
    title: str = "PI0 full-token activation norm over rollout time",
    group_cols: tuple[str, ...] = ("layer_index", "step"),
) -> None:
    frame = norms.copy()
    value_col = "norm_mean"
    color_label = "Mean token norm"
    if normalize == "layer_zscore":
        means = frame.groupby("layer_index")["norm_mean"].transform("mean")
        stds = frame.groupby("layer_index")["norm_mean"].transform("std").replace(0, np.nan)
        frame["norm_mean_layer_zscore"] = (frame["norm_mean"] - means) / (stds + 1e-8)
        frame["norm_mean_layer_zscore"] = frame["norm_mean_layer_zscore"].fillna(0.0)
        value_col = "norm_mean_layer_zscore"
        color_label = "Within-layer z-score"
    elif normalize != "none":
        raise ValueError(f"Unknown heatmap normalization `{normalize}`.")
    pivot = frame.groupby(list(group_cols), as_index=False)[value_col].mean()
    heat = pivot.pivot(index="layer_index", columns="step", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = "coolwarm" if normalize == "layer_zscore" else "viridis"
    vmax = None
    vmin = None
    if normalize == "layer_zscore":
        limit = float(np.nanpercentile(np.abs(heat.to_numpy()), 98))
        vmax = max(limit, 1e-6)
        vmin = -vmax
    image = ax.imshow(heat.to_numpy(), aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Layer index")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=color_label)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_episode_activation_norm_heatmaps(norms: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    for episode_index, frame in norms.groupby("episode_index"):
        success = bool(frame["episode_success"].max()) if "episode_success" in frame else False
        suffix = "success" if success else "failure"
        plot_activation_norm_heatmap(
            frame,
            output_dir / f"episode_{int(episode_index):03d}_{suffix}_raw.png",
            normalize="none",
            title=f"Episode {int(episode_index):03d} activation norm ({suffix})",
        )
        plot_activation_norm_heatmap(
            frame,
            output_dir / f"episode_{int(episode_index):03d}_{suffix}_layer_zscore.png",
            normalize="layer_zscore",
            title=f"Episode {int(episode_index):03d} activation norm, layer-normalized ({suffix})",
        )
    for success, frame in norms.groupby("episode_success"):
        suffix = "success" if bool(success) else "failure"
        plot_activation_norm_heatmap(
            frame,
            output_dir / f"group_{suffix}_layer_zscore.png",
            normalize="layer_zscore",
            title=f"{suffix.title()} episodes: layer-normalized activation norm",
        )


def plot_probe_curves(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for target, frame in curves.items():
        ax.plot(frame["layer"], frame["r2_mean"], marker="o", linewidth=2, label=target.replace("_", " "))
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean R2")
    ax.set_title("PI0 dynamic full-token rollout probes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_candidate_circuit(nodes: pd.DataFrame, edges: pd.DataFrame, path: Path) -> None:
    if nodes.empty:
        return
    targets = list(nodes["target"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(10, max(4, 1.0 + len(targets) * 0.8)))
    y_by_target = {target: idx for idx, target in enumerate(targets)}
    for _, row in nodes.iterrows():
        ax.scatter(row["layer"], y_by_target[row["target"]], s=max(30.0, 180.0 * max(float(row["score"]), 0.05)), alpha=0.75)
        ax.text(row["layer"], y_by_target[row["target"]] + 0.08, f"L{int(row['layer'])}", ha="center", fontsize=8)
    for _, edge in edges.iterrows():
        source = nodes[nodes["node_id"] == edge["source"]].iloc[0]
        dest = nodes[nodes["node_id"] == edge["target"]].iloc[0]
        y = y_by_target[source["target"]]
        ax.plot([source["layer"], dest["layer"]], [y, y], color="gray", alpha=0.35, linewidth=1 + 2 * max(float(edge["score"]), 0.0))
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([target.replace("_", " ") for target in targets])
    ax.set_xlabel("Layer index")
    ax.set_title("Candidate dynamic circuit nodes from layerwise decodability")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def flatten_step_row(row: dict[str, Any], episode_dir: Path, has_activation: bool) -> dict[str, Any]:
    flat = {
        "episode_dir": str(episode_dir),
        "episode_index": int(row["episode_index"]),
        "step": int(row["step"]),
        "instruction": row.get("instruction", ""),
        "image_path": row.get("image_path", ""),
        "wrist_image_path": row.get("wrist_image_path", ""),
        "activation_step_dir": row.get("activation_step_dir", ""),
        "has_activation": bool(has_activation),
        "is_replan": bool(row.get("is_replan", has_activation)),
        "reward": float(row.get("reward", 0.0)),
        "terminated": bool(row.get("terminated", False)),
        "truncated": bool(row.get("truncated", False)),
        "success": bool(row.get("success", False)),
        "queued_actions_remaining": int(row.get("queued_actions_remaining", 0)),
    }
    add_vector(flat, "gripper_position", row.get("gripper_position", []), axes=("x", "y", "z"))
    add_vector(flat, "pickup_offset", row.get("pickup_target_offset", []), axes=("x", "y", "z"))
    add_vector(flat, "place_offset", row.get("place_target_offset", []), axes=("x", "y", "z"))
    add_vector(flat, "action", row.get("action", []))
    add_vector(flat, "policy_pred_action", row.get("policy_pred_action", []))
    return flat


def flatten_activation_row(row: dict[str, Any], episode_dir: Path) -> dict[str, Any]:
    path = Path(row["path"])
    if not path.is_absolute():
        path = episode_dir / path
    return {
        "episode_index": int(row["episode_index"]),
        "step": int(row["step"]),
        "layer_index": int(row["layer_index"]),
        "layer_name": str(row["layer_name"]),
        "path": str(path),
        "shape": row.get("shape", []),
        "dtype": row.get("dtype", ""),
    }


def add_vector(flat: dict[str, Any], prefix: str, values: Any, axes: tuple[str, ...] | None = None) -> None:
    if values is None:
        return
    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    for index, value in enumerate(values_array):
        suffix = axes[index] if axes is not None and index < len(axes) else str(index)
        flat[f"{prefix}_{suffix}"] = float(value)


def write_circuit_graph_json(nodes: pd.DataFrame, edges: pd.DataFrame, path: Path) -> None:
    write_json(path, {"nodes": nodes.to_dict("records"), "edges": edges.to_dict("records")})


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
