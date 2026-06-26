from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .utils import write_json


def find_opposite_direction_pair(y_offset: np.ndarray) -> tuple[int, int]:
    norms = np.linalg.norm(y_offset, axis=1) + 1e-8
    dirs = y_offset / norms[:, None]
    cosine = dirs @ dirs.T
    np.fill_diagonal(cosine, 1.0)
    flat_idx = int(np.argmin(cosine))
    return tuple(int(v) for v in np.unravel_index(flat_idx, cosine.shape))


def surrogate_patch_demo(
    x_layers: np.ndarray,
    y_offset: np.ndarray,
    layer_names: list[str],
    results_csv: Path,
    probes_dir: Path,
    requested_layer: str | int,
    output_path: Path,
) -> dict[str, object]:
    results = pd.read_csv(results_csv)
    if requested_layer == "auto":
        layer_idx = int(results.sort_values("r2_mean", ascending=False).iloc[0]["layer"])
    else:
        layer_idx = int(requested_layer)
    layer_name = layer_names[layer_idx]
    probe = joblib.load(probes_dir / f"offset_probe_{layer_name}.joblib")

    idx_a, idx_b = find_opposite_direction_pair(y_offset)
    original_a = y_offset[idx_a]
    original_b = y_offset[idx_b]

    pred_b_before = probe.predict(x_layers[layer_idx, idx_b : idx_b + 1])[0]
    patched_b_pred = probe.predict(x_layers[layer_idx, idx_a : idx_a + 1])[0]

    before_dist = float(np.linalg.norm(pred_b_before - original_a))
    after_dist = float(np.linalg.norm(patched_b_pred - original_a))
    shift_toward_a_score = (before_dist - after_dist) / (before_dist + 1e-8)

    payload: dict[str, object] = {
        "mode": "surrogate_probe_readout",
        "layer": layer_idx,
        "layer_name": layer_name,
        "sample_A": idx_a,
        "sample_B": idx_b,
        "original_A_offset": original_a.tolist(),
        "original_B_offset": original_b.tolist(),
        "original_B_pred_offset": pred_b_before.tolist(),
        "patched_B_pred_offset": patched_b_pred.tolist(),
        "shift_toward_A_score": float(shift_toward_a_score),
        "note": "This minimal demo patches the saved representation and reads it out with the trained offset probe.",
    }
    write_json(output_path, payload)
    return payload
