"""Tests for pre-qualifying feature assembly.

Integration-style: uses the real data/processed/features.parquet, matching
how this project has no synthetic fixtures for its other inference tests
either (there are none pre-existing). Skipped if the parquet isn't present.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.inference.build_quali_features import (
    FEATURES_PATH, build_pre_quali_features, resolve_entry_list,
)
from src.models.quali_dataset import FEATURE_COLS

pytestmark = pytest.mark.skipif(
    not Path(FEATURES_PATH).exists(), reason="requires data/processed/features.parquet",
)


def test_resolve_entry_list_returns_drivers_for_2024():
    entries = resolve_entry_list(2024)
    assert len(entries) >= 15
    assert all(isinstance(c, str) and c.isupper() for c in entries)


def test_build_pre_quali_features_excludes_grid_and_quali_columns():
    entries = resolve_entry_list(2024)[:3]
    out = build_pre_quali_features(2024, 1, "bahrain", "2024-03-02", entries)
    assert len(out) == 3
    assert set(FEATURE_COLS).issubset(out.columns)
    forbidden = {
        "grid_clean", "grid_position", "quali_best_seconds",
        "gap_to_pole_seconds", "quali_session_reached", "reached_q3",
        "quali_gap_to_teammate", "grid_penalty", "grid_vs_championship",
    }
    assert forbidden.isdisjoint(out.columns)


def test_build_pre_quali_features_handles_debutant():
    out = build_pre_quali_features(2024, 1, "bahrain", "2024-03-02", ["ZZZ_NEW_DRIVER"])
    assert len(out) == 1
    assert out.iloc[0]["driver_elo_pre"] == 1500.0
    assert out.iloc[0]["driver_form_races"] == 0.0
    assert out.iloc[0]["constructor_elo_pre"] == 1500.0
    assert out.iloc[0]["perf_rating_pre"] == 1500.0
    assert out.iloc[0]["driver_elo_experience"] == 0.0


def test_build_pre_quali_features_populates_rolling_quali_for_active_driver():
    # An established driver from 2024 has real (non-NaN) rolling-qualifying
    # history carried forward — not a placeholder. Guards against the rolling
    # columns silently becoming NaN for active drivers.
    entries = resolve_entry_list(2024)
    out = build_pre_quali_features(2024, 10, "silverstone", "2024-07-07", entries)
    active = out[out["driver_code"] == entries[0]].iloc[0]
    assert pd.notna(active["driver_avg_grid_5"])
    assert pd.notna(active["constructor_avg_grid_5"])
    assert active["driver_avg_grid_5"] > 0


def test_resolve_entry_list_falls_back_to_prior_season():
    # A season just beyond the ingested data falls back to the last season
    # that has data, rather than failing.
    feat = pd.read_parquet(FEATURES_PATH)
    max_season = int(feat["season"].max())
    assert resolve_entry_list(max_season + 1) == resolve_entry_list(max_season)
