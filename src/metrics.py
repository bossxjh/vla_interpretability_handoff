from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def cosine_similarity_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    denom = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1) + 1e-8
    return float(np.mean(np.sum(y_true * y_pred, axis=1) / denom))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    per_dim_r2 = np.atleast_1d(r2_score(y_true, y_pred, multioutput="raw_values"))
    return {
        "r2_mean": float(np.mean(per_dim_r2)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "cos_sim": cosine_similarity_mean(y_true, y_pred),
        **{f"r2_{axis}": float(value) for axis, value in zip(["x", "y", "z"], per_dim_r2)},
    }
