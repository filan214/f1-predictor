"""Shared evaluation metrics for the qualifying-prediction model.

Mirrors ``src.models.metrics`` but the outcome being ranked is the qualifying
grid itself (``grid_clean``), not a separate ``target_*`` column - every
qualifying participant in this dataset gets a grid slot, so there is no
DNF/no-show concept to handle here the way the race model handles DNFs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RACE_KEYS = ["season", "round"]


def evaluate_quali_predictions(meta: pd.DataFrame, pred: np.ndarray, name: str) -> dict:
    """Compute MAE, Spearman, pole accuracy, and top-10 (Q3) accuracy.

    Parameters
    ----------
    meta : DataFrame with RACE_KEYS and ``grid_clean``.
    pred : predicted grid position per row (lower = better), row-aligned.
    """
    work = meta[RACE_KEYS + ["grid_clean"]].reset_index(drop=True).copy()
    work["pred"] = np.asarray(pred)
    work["pred_rank"] = work.groupby(RACE_KEYS)["pred"].rank(method="first")

    mae = float((work["pred"] - work["grid_clean"]).abs().mean())
    spearman = float(work["pred"].corr(work["grid_clean"], method="spearman"))

    p1 = work[work["pred_rank"] == 1]
    pole_acc = float((p1["grid_clean"] == 1).mean())
    top10_acc = float(((work["pred_rank"] <= 10) == (work["grid_clean"] <= 10)).mean())

    return {
        "model": name,
        "mae": mae,
        "spearman": spearman,
        "pole_acc": pole_acc,
        "top10_acc": top10_acc,
        "n_rows": int(len(work)),
    }


def quali_metrics_table(rows: list[dict]) -> pd.DataFrame:
    cols = ["model", "mae", "spearman", "pole_acc", "top10_acc", "n_rows"]
    return pd.DataFrame(rows)[cols].round(4)
