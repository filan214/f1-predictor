"""Tests for qualifying-grid prediction.

Requires the trained qualifying models (Task 4/5) to exist — skipped if not.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.build_quali_features import build_pre_quali_features, resolve_entry_list
from src.inference.predict_quali import predict_quali, predicted_grid_dict

MODELS_DIR = Path("models")

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "quali_lgbm.joblib").exists(),
    reason="requires trained qualifying models (python -m src.models.train_quali)",
)


def test_predict_quali_returns_valid_permutation():
    entries = resolve_entry_list(2024)
    features_df = build_pre_quali_features(2024, 1, "bahrain", "2024-03-02", entries)
    pred = predict_quali(features_df)
    assert sorted(pred["predicted_grid"].tolist()) == list(range(1, len(entries) + 1))


def test_predicted_grid_dict_shape():
    entries = resolve_entry_list(2024)[:5]
    features_df = build_pre_quali_features(2024, 1, "bahrain", "2024-03-02", entries)
    pred = predict_quali(features_df)
    grid = predicted_grid_dict(pred)
    assert set(grid.keys()) == set(entries)
    assert sorted(grid.values()) == list(range(1, 6))
