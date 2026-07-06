"""Unit tests for qualifying-model evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.quali_metrics import evaluate_quali_predictions, quali_metrics_table


def test_evaluate_quali_predictions_perfect():
    meta = pd.DataFrame({
        "season": [2024, 2024, 2024], "round": [1, 1, 1],
        "grid_clean": [1.0, 2.0, 3.0],
    })
    result = evaluate_quali_predictions(meta, np.array([1.0, 2.0, 3.0]), "perfect")
    assert result["mae"] == 0.0
    assert result["spearman"] == 1.0
    assert result["pole_acc"] == 1.0
    assert result["top10_acc"] == 1.0


def test_evaluate_quali_predictions_wrong_pole():
    meta = pd.DataFrame({
        "season": [2024, 2024, 2024], "round": [1, 1, 1],
        "grid_clean": [1.0, 2.0, 3.0],
    })
    result = evaluate_quali_predictions(meta, np.array([2.0, 1.0, 3.0]), "swapped")
    assert result["pole_acc"] == 0.0
    assert result["mae"] > 0.0


def test_quali_metrics_table_has_expected_columns():
    rows = [
        {"model": "a", "mae": 1.0, "spearman": 0.9, "pole_acc": 0.5,
         "top10_acc": 0.8, "n_rows": 20},
    ]
    table = quali_metrics_table(rows)
    assert list(table.columns) == ["model", "mae", "spearman", "pole_acc", "top10_acc", "n_rows"]


def test_top10_acc_detects_top10_boundary_disagreement():
    # 12 drivers, actual grid 1..12. Swap the predicted values for the
    # actual-P10 and actual-P11 drivers so their top-10 membership flips:
    # predicted-top-10 then disagrees with actual-top-10 for exactly 2 rows.
    grid = [float(g) for g in range(1, 13)]
    pred = grid.copy()
    pred[9], pred[10] = 11.0, 10.0
    meta = pd.DataFrame({
        "season": [2024] * 12, "round": [1] * 12, "grid_clean": grid,
    })
    result = evaluate_quali_predictions(meta, np.array(pred, dtype=float), "boundary")
    assert result["top10_acc"] == pytest.approx(10 / 12)
