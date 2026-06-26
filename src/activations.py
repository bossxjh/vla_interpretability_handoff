from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir


def stack_layer_activations(hidden_by_sample: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    if not hidden_by_sample:
        raise ValueError("Cannot stack an empty activation list.")
    layer_names = list(hidden_by_sample[0].keys())
    layers = []
    for name in layer_names:
        if any(name not in sample for sample in hidden_by_sample):
            raise ValueError(f"Layer `{name}` is missing from at least one sample.")
        layers.append(np.stack([sample[name] for sample in hidden_by_sample], axis=0).astype(np.float32))
    max_width = max(layer.shape[1] for layer in layers)
    padded_layers = [
        np.pad(layer, ((0, 0), (0, max_width - layer.shape[1]))) if layer.shape[1] < max_width else layer
        for layer in layers
    ]
    return np.stack(padded_layers, axis=0), layer_names


def save_activations(
    path: Path,
    x_layers: np.ndarray,
    layer_names: list[str],
    targets: dict[str, np.ndarray],
    y_action: np.ndarray,
    metadata: dict[str, Any],
    y_action_chunk: np.ndarray | None = None,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "X_layers": x_layers,
        "layer_names": np.asarray(layer_names),
        **{f"y_{name}": values.astype(np.float32) for name, values in targets.items()},
        "y_action": y_action.astype(np.float32),
        "metadata": json.dumps(metadata),
    }
    if y_action_chunk is not None:
        payload["y_action_chunk"] = y_action_chunk.astype(np.float32)
    np.savez_compressed(
        path,
        **payload,
    )


def load_activations(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    loaded = {
        "X_layers": data["X_layers"],
        "layer_names": [str(x) for x in data["layer_names"].tolist()],
        "metadata": json.loads(str(data["metadata"])),
    }
    for key in data.files:
        if key.startswith("y_"):
            loaded[key] = data[key]
    return loaded
