from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .probes import train_layerwise_ridge
from .utils import ensure_dir


DEFAULT_ROLLOUT_TARGETS = ("pickup_offset", "place_offset", "action", "executed_action_chunk", "progress")


def analyze_rollout_interpretability(
    rollout_dir: Path,
    output_dir: Path,
    targets: tuple[str, ...] = DEFAULT_ROLLOUT_TARGETS,
    action_chunk_horizon: int = 4,
    alpha: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dataset = load_rollout_replan_dataset(rollout_dir, action_chunk_horizon=action_chunk_horizon)
    ensure_dir(output_dir)
    curves: dict[str, pd.DataFrame] = {}
    for target in targets:
        if target not in dataset.targets:
            available = sorted(dataset.targets)
            raise KeyError(f"Unknown rollout probe target `{target}`. Available targets: {available}.")
        y = dataset.targets[target]
        results = train_layerwise_ridge(
            dataset.x_layers,
            y,
            dataset.layer_names,
            alpha=alpha,
            test_size=test_size,
            seed=seed,
            shuffle_labels=False,
            save_dir=None,
            prefix=f"rollout_{target}_probe",
        )
        results.to_csv(output_dir / f"layerwise_rollout_probe_{target}.csv", index=False)
        curves[target] = results

        shuffled = train_layerwise_ridge(
            dataset.x_layers,
            y,
            dataset.layer_names,
            alpha=alpha,
            test_size=test_size,
            seed=seed,
            shuffle_labels=True,
            save_dir=None,
            prefix=f"rollout_{target}_probe",
        )
        shuffled.to_csv(output_dir / f"layerwise_rollout_probe_{target}_shuffled.csv", index=False)

    summary = summarize_rollout_probe_curves(curves)
    summary.to_csv(output_dir / "rollout_probe_target_summary.csv", index=False)
    save_rollout_metadata(output_dir, dataset.metadata, targets, action_chunk_horizon)
    plot_rollout_r2_comparison(curves, output_dir / "rollout_probe_targets_r2_comparison.png")
    return summary, curves


class RolloutReplanDataset:
    def __init__(
        self,
        x_layers: np.ndarray,
        layer_names: list[str],
        targets: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> None:
        self.x_layers = x_layers
        self.layer_names = layer_names
        self.targets = targets
        self.metadata = metadata


def load_rollout_replan_dataset(rollout_dir: Path, action_chunk_horizon: int = 4) -> RolloutReplanDataset:
    if action_chunk_horizon <= 0:
        raise ValueError(f"`action_chunk_horizon` must be positive, got {action_chunk_horizon}.")
    episode_dirs = sorted(path for path in rollout_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No `episode_*` directories found under {rollout_dir}.")

    x_layers_by_episode = []
    layer_names: list[str] | None = None
    target_rows: list[dict[str, Any]] = []
    num_skipped_chunks = 0
    for episode_dir in episode_dirs:
        steps_path = episode_dir / "steps.jsonl"
        activations_path = episode_dir / "replan_activations.npz"
        if not steps_path.exists() or not activations_path.exists():
            continue
        steps = _read_jsonl(steps_path)
        steps_by_index = {int(row["step"]): row for row in steps}
        data = np.load(activations_path, allow_pickle=True)
        x_layers = data["X_layers"].astype(np.float32)
        episode_layer_names = [str(item) for item in data["layer_names"].tolist()]
        if layer_names is None:
            layer_names = episode_layer_names
        elif layer_names != episode_layer_names:
            raise ValueError(f"Layer names differ in {activations_path}.")
        step_indices = data["step_indices"].astype(int).tolist()
        if x_layers.shape[1] != len(step_indices):
            raise ValueError(
                f"Activation sample count and step_indices differ in {activations_path}: "
                f"{x_layers.shape[1]} vs {len(step_indices)}."
            )

        keep_indices = []
        for replan_index, step in enumerate(step_indices):
            row = steps_by_index.get(int(step))
            if row is None:
                raise ValueError(f"Step {step} from {activations_path} is missing in {steps_path}.")
            chunk = _future_actions(steps_by_index, int(step), action_chunk_horizon)
            if chunk is None:
                num_skipped_chunks += 1
                continue
            keep_indices.append(replan_index)
            num_steps = max(len(steps), 1)
            target_rows.append(
                {
                    "episode_dir": str(episode_dir),
                    "episode_index": int(row["episode_index"]),
                    "step": int(row["step"]),
                    "progress": [float(row["step"]) / float(max(num_steps - 1, 1))],
                    "pickup_offset": row["pickup_target_offset"],
                    "place_offset": row["place_target_offset"],
                    "action": row["action"],
                    "executed_action_chunk": chunk.reshape(-1).astype(float).tolist(),
                    "success": bool(row.get("success", False)),
                }
            )
        if keep_indices:
            x_layers_by_episode.append(x_layers[:, keep_indices, :])

    if not x_layers_by_episode or layer_names is None or not target_rows:
        raise ValueError(f"No usable replan activations with complete action chunks found in {rollout_dir}.")

    x_layers_all = np.concatenate(x_layers_by_episode, axis=1)
    targets = {
        "pickup_offset": np.asarray([row["pickup_offset"] for row in target_rows], dtype=np.float32),
        "place_offset": np.asarray([row["place_offset"] for row in target_rows], dtype=np.float32),
        "action": np.asarray([row["action"] for row in target_rows], dtype=np.float32),
        "executed_action_chunk": np.asarray([row["executed_action_chunk"] for row in target_rows], dtype=np.float32),
        "progress": np.asarray([row["progress"] for row in target_rows], dtype=np.float32),
    }
    metadata = {
        "rollout_dir": str(rollout_dir),
        "num_episodes": len(episode_dirs),
        "num_replan_samples": int(x_layers_all.shape[1]),
        "num_layers": int(x_layers_all.shape[0]),
        "hidden_dim": int(x_layers_all.shape[2]),
        "action_chunk_horizon": int(action_chunk_horizon),
        "num_skipped_terminal_replans": int(num_skipped_chunks),
        "samples": target_rows,
    }
    return RolloutReplanDataset(x_layers_all, layer_names, targets, metadata)


def summarize_rollout_probe_curves(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for target, frame in curves.items():
        best = frame.sort_values("r2_mean", ascending=False).iloc[0]
        rows.append(
            {
                "target": target,
                "best_layer": int(best["layer"]),
                "best_layer_name": best["layer_name"],
                "best_r2_mean": float(best["r2_mean"]),
                "best_mse": float(best["mse"]),
                "best_cos_sim": float(best["cos_sim"]),
            }
        )
    return pd.DataFrame(rows)


def plot_rollout_r2_comparison(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for target, frame in curves.items():
        ax.plot(frame["layer"], frame["r2_mean"], marker="o", linewidth=2, label=target.replace("_", " "))
    ax.axvline(_first_expert_layer(curves), color="black", linestyle="--", linewidth=1, alpha=0.35)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean R2")
    ax.set_title("PI0.5 rollout replan hidden-state probes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rollout_metadata(
    output_dir: Path,
    metadata: dict[str, Any],
    targets: tuple[str, ...],
    action_chunk_horizon: int,
) -> None:
    with open(output_dir / "rollout_probe_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "probe_targets": list(targets),
                "action_chunk_horizon": int(action_chunk_horizon),
                "target_definitions": {
                    "pickup_offset": "pickup_target_position - gripper_position at the replan step",
                    "place_offset": "place_target_position - gripper_position at the replan step",
                    "action": "executed low-level action at the replan step",
                    "executed_action_chunk": "flattened sequence of executed actions from the replan step over the requested horizon",
                    "progress": "step index normalized by episode length",
                },
            },
            f,
            indent=2,
        )


def _future_actions(steps_by_index: dict[int, dict[str, Any]], start_step: int, horizon: int) -> np.ndarray | None:
    actions = []
    for step in range(start_step, start_step + horizon):
        row = steps_by_index.get(step)
        if row is None or "action" not in row:
            return None
        actions.append(row["action"])
    return np.asarray(actions, dtype=np.float32)


def _first_expert_layer(curves: dict[str, pd.DataFrame]) -> int:
    if not curves:
        return 0
    frame = next(iter(curves.values()))
    expert = frame[frame["layer_name"].astype(str).str.startswith("expert_layer")]
    if expert.empty:
        return 0
    return int(expert["layer"].min())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
