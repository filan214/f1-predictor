"""Shared evaluation metrics.

A model produces one continuous *expected finishing position* per driver per
race. From that single output we derive every PRD metric:

* **MAE** and **Spearman** on classified finishers (the primary signal).
* **winner / podium / points** outcomes by *ranking predictions within a race*
  (the lowest predicted position is the predicted winner, etc.).
* **winner log-loss** from a softmax over negative predicted positions, so a
  regression model still yields comparable win probabilities.

Every metric is computed the same way for the baseline and every model, so the
comparison table is apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RACE_KEYS = ["season", "round"]
WIN_PROB_TEMPERATURE = 3.0  # softmax sharpness for win-probability derivation


def _race_win_logloss(work: pd.DataFrame, pred_col: str, temperature: float) -> float:
    """Mean negative log-prob assigned to the actual winner, over test races."""
    losses: list[float] = []
    for _, g in work.groupby(RACE_KEYS):
        if g["target_is_winner"].sum() == 0:  # no classified winner -> skip race
            continue
        logits = -g[pred_col].to_numpy() / temperature
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        winner_idx = int(np.argmax(g["target_is_winner"].to_numpy()))
        losses.append(-np.log(max(p[winner_idx], 1e-15)))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_predictions(
    meta: pd.DataFrame,
    pred: np.ndarray,
    name: str,
    temperature: float = WIN_PROB_TEMPERATURE,
) -> dict:
    """Compute the full metric suite for one set of predictions.

    Parameters
    ----------
    meta : DataFrame with RACE_KEYS and the ``target_*`` columns.
    pred : predicted finishing position per row (lower = better), row-aligned.
    """
    work = meta[RACE_KEYS + [
        "target_finish_position", "target_is_winner",
        "target_is_podium", "target_is_points",
    ]].reset_index(drop=True).copy()
    work["pred"] = np.asarray(pred)

    # Rank predictions within each race -> derived classifications.
    work["pred_rank"] = work.groupby(RACE_KEYS)["pred"].rank(method="first")
    work["pred_is_winner"] = (work["pred_rank"] == 1).astype(int)
    work["pred_is_podium"] = (work["pred_rank"] <= 3).astype(int)
    work["pred_is_points"] = (work["pred_rank"] <= 10).astype(int)

    # Regression quality on classified finishers only.
    cl = work[work["target_finish_position"].notna()]
    mae = float((cl["pred"] - cl["target_finish_position"]).abs().mean())
    spearman = float(
        cl["pred"].corr(cl["target_finish_position"], method="spearman")
    )

    # Winner accuracy: did the predicted P1 actually win their race?
    p1 = work[work["pred_rank"] == 1]
    winner_acc = float(p1["target_is_winner"].mean())

    return {
        "model": name,
        "mae": mae,
        "spearman": spearman,
        "winner_logloss": _race_win_logloss(work, "pred", temperature),
        "winner_acc": winner_acc,
        "podium_f1": float(f1_score(work["target_is_podium"], work["pred_is_podium"])),
        "points_acc": float(
            accuracy_score(work["target_is_points"], work["pred_is_points"])
        ),
        "n_rows": int(len(work)),
        "n_classified": int(len(cl)),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble per-model metric dicts into a tidy comparison table."""
    cols = [
        "model", "mae", "spearman", "winner_logloss", "winner_acc",
        "podium_f1", "points_acc", "n_rows", "n_classified",
    ]
    return pd.DataFrame(rows)[cols].round(4)
