"""Data loading, temporal splitting, and design matrix for the qualifying
model.

Mirrors ``src.models.dataset`` but the target is ``grid_clean`` (the actual
starting slot) instead of finishing position, and every qualifying/grid-
derived column is excluded from the inputs - none of it is knowable before
qualifying happens. In exchange, this module adds new rolling-qualifying
features (a driver's/constructor's recent grid history) computed here from
``grid_clean``, with the same leakage discipline as
``src.features.rolling``: every aggregate uses
``groupby(...).shift(1).rolling(window)`` so a race's own grid_clean is never
part of its own feature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .dataset import FEATURES_PATH, TEST_SEASON, TRAIN_END, VAL_SEASON

logger = logging.getLogger(__name__)

TARGET = "grid_clean"
ROLL_WINDOW = 5

# Quali-safe subset of the race model's features: ratings, rolling FINISH
# form, constructor form, standings, circuit, weather. Deliberately excludes
# grid_clean/grid_position and everything derived from qualifying itself
# (quali_best_seconds, gap_to_pole_seconds, quali_session_reached, reached_q3,
# quali_gap_to_teammate, grid_penalty, grid_vs_championship) - none of that
# is knowable before qualifying happens.
BASE_FEATURE_COLS = [
    "championship_position", "championship_points", "championship_wins",
    "constructor_position", "constructor_points", "constructor_wins",
    "driver_elo_pre", "constructor_elo_pre", "perf_rating_pre",
    "driver_elo_experience",
    "driver_avg_finish_5", "driver_avg_points_5", "driver_dnf_rate_5",
    "driver_form_races", "constructor_reliability_5",
    "constructor_points_per_race_5", "constructor_avg_pit_seconds_5",
    "circuit_is_street", "circuit_overtaking_index", "circuit_pole_win_pct",
    "circuit_dnf_rate_hist", "circuit_avg_pitstops_hist",
    "circuit_avg_stint_laps_hist", "circuit_history_races",
    "air_temp_avg", "track_temp_avg", "humidity_avg", "wind_speed_avg",
    "rain_flag", "weather_missing",
]

# New rolling-qualifying features, computed in this module from grid_clean.
QUALI_ROLLING_COLS = [
    "driver_avg_grid_5", "driver_best_grid_5",
    "constructor_avg_grid_5", "driver_grid_vs_teammate_5",
]

FEATURE_COLS = BASE_FEATURE_COLS + QUALI_ROLLING_COLS

META_COLS = [
    "season", "round", "driver_id", "driver_code", "constructor_id",
    "circuit_id", TARGET,
]


def _shift_roll(group: pd.Series, *, window: int, how: str) -> pd.Series:
    """Trailing rolling aggregate that excludes the current row (shift(1))."""
    shifted = group.shift(1).rolling(window, min_periods=1)
    return getattr(shifted, how)()


def add_rolling_quali_features(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Attach leakage-free rolling-qualifying features derived from grid_clean.

    * ``driver_avg_grid_5`` / ``driver_best_grid_5`` - the driver's mean/best
      starting grid over their last ``window`` races.
    * ``constructor_avg_grid_5`` - both cars pooled, mean starting grid over
      the constructor's last ``window`` races.
    * ``driver_grid_vs_teammate_5`` - the driver's rolling average grid minus
      their TEAMMATE's rolling average grid (teammate only, computed by
      subtracting the driver's own rolling sum/count from the pooled
      constructor rolling sum/count) over the same window.
    """
    out = df.sort_values(["driver_id", "race_order"]).copy()

    g = out.groupby("driver_id", group_keys=False)["grid_clean"]
    out["driver_avg_grid_5"] = g.apply(lambda s: _shift_roll(s, window=window, how="mean"))
    out["driver_best_grid_5"] = g.apply(lambda s: _shift_roll(s, window=window, how="min"))
    out["_driver_roll_sum"] = g.apply(lambda s: _shift_roll(s, window=window, how="sum"))
    out["_driver_roll_count"] = g.apply(lambda s: _shift_roll(s, window=window, how="count"))

    # Team pooled grid (both cars), rolled over the constructor's prior races.
    team_race = (
        out.groupby(["constructor_id", "season", "round"], as_index=False)
        .agg(team_grid_sum=("grid_clean", "sum"), team_grid_count=("grid_clean", "count"))
        .merge(
            out[["season", "round", "race_order"]].drop_duplicates(),
            on=["season", "round"],
        )
        .sort_values(["constructor_id", "race_order"])
    )
    gc = team_race.groupby("constructor_id", group_keys=False)
    team_race["_team_roll_sum"] = gc["team_grid_sum"].apply(
        lambda s: _shift_roll(s, window=window, how="sum")
    )
    team_race["_team_roll_count"] = gc["team_grid_count"].apply(
        lambda s: _shift_roll(s, window=window, how="sum")
    )
    team_race["constructor_avg_grid_5"] = (
        team_race["_team_roll_sum"] / team_race["_team_roll_count"]
    )

    out = out.merge(
        team_race[["constructor_id", "season", "round",
                    "_team_roll_sum", "_team_roll_count", "constructor_avg_grid_5"]],
        on=["constructor_id", "season", "round"], how="left",
    )

    driver_sum_filled = out["_driver_roll_sum"].fillna(0.0)
    teammate_sum = out["_team_roll_sum"] - driver_sum_filled
    teammate_count = out["_team_roll_count"] - out["_driver_roll_count"]
    teammate_avg = teammate_sum / teammate_count.where(teammate_count > 0)
    out["driver_grid_vs_teammate_5"] = out["driver_avg_grid_5"] - teammate_avg

    out = out.drop(columns=[
        "_driver_roll_sum", "_driver_roll_count", "_team_roll_sum", "_team_roll_count",
    ])
    return out


def load_quali_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.features.build_features` first."
        )
    return add_rolling_quali_features(pd.read_parquet(path))


def get_quali_splits(
    path: Path = FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train<=2022, val==2023, test==2024) frames in time order."""
    df = load_quali_features(path).sort_values(["race_order", "grid_clean"])
    train = df[df["season"] <= TRAIN_END].copy()
    val = df[df["season"] == VAL_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()
    logger.info(
        "Quali splits — train %d (<=%d) | val %d (%d) | test %d (%d)",
        len(train), TRAIN_END, len(val), VAL_SEASON, len(test), TEST_SEASON,
    )
    return train, val, test


class QualiPreprocessor:
    """Median imputation (fit on training rows only) + missingness indicators.

    Identical mechanism to ``src.models.dataset.Preprocessor``, fit over the
    qualifying model's ``FEATURE_COLS`` instead.
    """

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.missing_cols_: list[str] = []
        self.columns_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "QualiPreprocessor":
        X = df[FEATURE_COLS]
        self.missing_cols_ = [c for c in FEATURE_COLS if X[c].isna().any()]
        self.medians_ = X.median()
        self.columns_ = FEATURE_COLS + [f"{c}__missing" for c in self.missing_cols_]
        logger.info(
            "Quali preprocessor fit: %d features, %d with missing values -> indicators",
            len(FEATURE_COLS), len(self.missing_cols_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("QualiPreprocessor must be fit before transform().")
        out = df[FEATURE_COLS].copy()
        for c in self.missing_cols_:
            out[f"{c}__missing"] = df[c].isna().astype("int8")
        out = out.fillna(self.medians_).fillna(0.0)
        return out[self.columns_].reset_index(drop=True)


def quali_design_matrix(
    df: pd.DataFrame, processor: QualiPreprocessor
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) for a split. Every ``grid_clean`` value is present
    (no DNF-style gap for qualifying), so unlike the race model there is no
    ``classified()`` filter to apply."""
    X = processor.transform(df)
    y = df[TARGET].reset_index(drop=True)
    meta = df[META_COLS].reset_index(drop=True)
    return X, y, meta
