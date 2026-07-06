"""Unit tests for the qualifying dataset module.

The critical property under test: rolling-qualifying features must never let
a race's own grid_clean leak into that same row's features (shift(1) before
every rolling aggregate).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models.quali_dataset import (
    FEATURE_COLS, FEATURES_PATH, QualiPreprocessor, add_rolling_quali_features,
    get_quali_splits, quali_design_matrix,
)


def _toy_frame() -> pd.DataFrame:
    # Two drivers on the same constructor across 3 chronological races.
    # round1: a=1, b=2 | round2: a=3, b=1 | round3: a=2, b=4
    return pd.DataFrame({
        "season": [2024, 2024, 2024, 2024, 2024, 2024],
        "round": [1, 1, 2, 2, 3, 3],
        "race_order": [0, 0, 1, 1, 2, 2],
        "driver_id": ["a", "b", "a", "b", "a", "b"],
        "constructor_id": ["team1", "team1", "team1", "team1", "team1", "team1"],
        "grid_clean": [1.0, 2.0, 3.0, 1.0, 2.0, 4.0],
    })


def test_first_race_has_no_rolling_history():
    out = add_rolling_quali_features(_toy_frame())
    first_race = out[out["race_order"] == 0]
    assert first_race["driver_avg_grid_5"].isna().all()
    assert first_race["driver_best_grid_5"].isna().all()
    assert first_race["constructor_avg_grid_5"].isna().all()


def test_rolling_average_excludes_current_race():
    out = add_rolling_quali_features(_toy_frame())
    # Driver "a" at race_order=2: prior races are grid 1 (r0), 3 (r1) -> mean
    # 2.0, min 1.0. Its OWN current grid_clean (2.0 at r2) must not count.
    row = out[(out["driver_id"] == "a") & (out["race_order"] == 2)].iloc[0]
    assert row["driver_avg_grid_5"] == 2.0
    assert row["driver_best_grid_5"] == 1.0


def test_constructor_avg_pools_both_cars_excluding_current_race():
    out = add_rolling_quali_features(_toy_frame())
    # At race_order=2, prior team races: r0 (grids 1,2) and r1 (grids 3,1) ->
    # pooled mean = (1+2+3+1)/4 = 1.75, same value for both drivers' rows.
    row_a = out[(out["driver_id"] == "a") & (out["race_order"] == 2)].iloc[0]
    row_b = out[(out["driver_id"] == "b") & (out["race_order"] == 2)].iloc[0]
    assert row_a["constructor_avg_grid_5"] == 1.75
    assert row_b["constructor_avg_grid_5"] == 1.75


def test_vs_teammate_compares_to_teammate_only_not_pooled_average():
    out = add_rolling_quali_features(_toy_frame())
    # At race_order=2: a's own rolling avg (prior [1,3]) = 2.0. Teammate b's
    # prior races [2,1] -> teammate rolling avg = 1.5. vs_teammate = 0.5.
    row_a = out[(out["driver_id"] == "a") & (out["race_order"] == 2)].iloc[0]
    assert row_a["driver_grid_vs_teammate_5"] == 0.5


@pytest.mark.skipif(not Path(FEATURES_PATH).exists(), reason="requires features.parquet")
def test_get_quali_splits_partition_by_season():
    train, val, test = get_quali_splits()
    assert train["season"].max() <= 2022
    assert set(val["season"].unique()) == {2023}
    assert set(test["season"].unique()) == {2024}
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


@pytest.mark.skipif(not Path(FEATURES_PATH).exists(), reason="requires features.parquet")
def test_preprocessor_produces_no_nans():
    train, val, _ = get_quali_splits()
    pre = QualiPreprocessor().fit(train)
    X, y, meta = quali_design_matrix(val, pre)
    assert not X.isna().any().any()
    assert len(X) == len(y) == len(meta)
    assert list(X.columns) == pre.columns_


@pytest.mark.skipif(not Path(FEATURES_PATH).exists(), reason="requires features.parquet")
def test_feature_cols_exclude_qualifying_derived_columns():
    forbidden = {
        "grid_clean", "grid_position", "quali_best_seconds",
        "gap_to_pole_seconds", "quali_session_reached", "reached_q3",
        "quali_gap_to_teammate", "grid_penalty", "grid_vs_championship",
    }
    assert forbidden.isdisjoint(FEATURE_COLS)
