from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .metrics import regression_metrics
from .utils import ensure_dir


def train_layerwise_ridge(
    x_layers: np.ndarray,
    y: np.ndarray,
    layer_names: list[str],
    alpha: float,
    test_size: float,
    seed: int,
    shuffle_labels: bool = False,
    save_dir: Path | None = None,
    prefix: str = "offset_probe",
) -> pd.DataFrame:
    if len(y) < 4:
        raise ValueError(
            f"Layerwise probing needs at least 4 samples, got {len(y)}. "
            "If activation extraction was run with `--max-samples`, rerun it without that smoke-test limit first."
        )
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y = y[rng.permutation(len(y))]

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed)
    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError(
            f"Layerwise probing needs at least 2 train and 2 test samples, got train={len(train_idx)}, "
            f"test={len(test_idx)}. Increase the number of extracted states or adjust `probe.test_size`."
        )
    rows: list[dict[str, float | int | str]] = []
    if save_dir is not None:
        ensure_dir(save_dir)

    for layer_idx, layer_name in tqdm(list(enumerate(layer_names)), desc="train layerwise probes"):
        x = x_layers[layer_idx]
        model = Ridge(alpha=alpha)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        metrics = regression_metrics(y[test_idx], pred)
        row: dict[str, float | int | str] = {"layer": layer_idx, "layer_name": layer_name}
        for key in ["r2_x", "r2_y", "r2_z", "r2_mean", "mse", "cos_sim"]:
            row[key] = metrics.get(key, np.nan)
        rows.append(row)

        if save_dir is not None and not shuffle_labels:
            joblib.dump(model, save_dir / f"{prefix}_{layer_name}.joblib")

    return pd.DataFrame(rows)


def best_layer_from_results(results: pd.DataFrame) -> int:
    return int(results.sort_values("r2_mean", ascending=False).iloc[0]["layer"])
