"""Unit tests for the championship-order qualifying baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.quali_baseline import baseline_predictions


def test_baseline_ranks_by_championship_position():
    df = pd.DataFrame({
        "season": [2024, 2024, 2024], "round": [5, 5, 5],
        "championship_position": [3.0, 1.0, 2.0],
        "driver_elo_pre": [1500.0, 1500.0, 1500.0],
    })
    pred = baseline_predictions(df)
    assert list(pred) == [3.0, 1.0, 2.0]


def test_baseline_falls_back_to_elo_when_standings_missing():
    df = pd.DataFrame({
        "season": [2024, 2024], "round": [1, 1],
        "championship_position": [np.nan, np.nan],
        "driver_elo_pre": [1600.0, 1500.0],
    })
    pred = baseline_predictions(df)
    # Higher ELO -> predicted a better (numerically lower) grid rank.
    assert pred[0] == 1.0
    assert pred[1] == 2.0


def test_baseline_ranks_missing_standings_after_classified_drivers():
    # One race: two drivers with valid championship positions, one without.
    # The standings-less driver must NOT take pole despite the best ELO.
    df = pd.DataFrame({
        "season": [2024, 2024, 2024], "round": [10, 10, 10],
        "championship_position": [2.0, 1.0, np.nan],
        "driver_elo_pre": [1500.0, 1500.0, 1700.0],
    })
    pred = baseline_predictions(df)
    assert pred[1] == 1.0   # championship leader -> pole
    assert pred[0] == 2.0   # championship P2 -> P2
    assert pred[2] == 3.0   # no standings -> last, even with the best ELO
