from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_jsonl, write_jsonl


@dataclass
class StateSample:
    sample_id: int
    image_path: str
    instruction: str
    target_position: list[float]
    gripper_position: list[float]
    target_offset: list[float]
    target_pixel: list[int]
    gripper_pixel: list[int]
    background_color: list[int]
    wrist_image_path: str | None = None
    observation_state: list[float] | None = None
    source_demo_id: str | None = None
    source_frame_index: int | None = None
    source_task_name: str | None = None
    source_hdf5_path: str | None = None
    target_role: str | None = None
    target_joint_name: str | None = None
    gt_action: list[float] | None = None
    gt_action_chunk: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "instruction": self.instruction,
            "target_position": self.target_position,
            "gripper_position": self.gripper_position,
            "target_offset": self.target_offset,
            "target_pixel": self.target_pixel,
            "gripper_pixel": self.gripper_pixel,
            "background_color": self.background_color,
            "wrist_image_path": self.wrist_image_path,
            "observation_state": self.observation_state,
            "source_demo_id": self.source_demo_id,
            "source_frame_index": self.source_frame_index,
            "source_task_name": self.source_task_name,
            "source_hdf5_path": self.source_hdf5_path,
            "target_role": self.target_role,
            "target_joint_name": self.target_joint_name,
            "gt_action": self.gt_action,
            "gt_action_chunk": self.gt_action_chunk,
        }


def save_states(path: Path, samples: list[StateSample]) -> None:
    write_jsonl(path, [sample.to_dict() for sample in samples])


def load_states(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _drop_constant_zero_z(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2 and values.shape[1] == 3 and np.allclose(values[:, 2], 0.0):
        return values[:, :2]
    return values


def states_to_targets(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    targets = {
        "offset": np.asarray([s["target_offset"] for s in states], dtype=np.float32),
        "target_position": np.asarray([s["target_position"] for s in states], dtype=np.float32),
        "gripper_position": np.asarray([s["gripper_position"] for s in states], dtype=np.float32),
    }
    if all(s.get("gt_action") is not None for s in states):
        targets["gt_action"] = np.asarray([s["gt_action"] for s in states], dtype=np.float32)
    if all(s.get("gt_action_chunk") is not None for s in states):
        targets["gt_action_chunk"] = _flatten_action_chunks([s["gt_action_chunk"] for s in states])
    return {name: _drop_constant_zero_z(values) for name, values in targets.items()}


def _flatten_action_chunks(chunks: list[list[list[float]]]) -> np.ndarray:
    arrays = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
    if not arrays:
        raise ValueError("Cannot flatten an empty action chunk list.")
    if any(array.ndim != 2 for array in arrays):
        shapes = [array.shape for array in arrays[:5]]
        raise ValueError(f"Expected action chunks shaped [horizon, action_dim], got examples: {shapes}.")
    min_horizon = min(array.shape[0] for array in arrays)
    if min_horizon <= 0:
        raise ValueError("At least one action chunk is empty.")
    return np.stack([array[:min_horizon].reshape(-1) for array in arrays], axis=0).astype(np.float32)
