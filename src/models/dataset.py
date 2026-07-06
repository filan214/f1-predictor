"""Data loading, temporal splitting, and preprocessing for modelling.

Single source of truth for *which* columns are features, *how* the data splits
in time, and *how* missing values are handled — so baseline, training,
evaluation, and explanation all see an identical, leakage-free design matrix.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")

# Temporal split (the whole project hinges on this never being shuffled).
TRAIN_END = 2022      # train: seasons <= 2022
VAL_SEASON = 2023     # validation: Optuna tuning + stacking meta-learner
TEST_SEASON = 2024    # test: touched only for final reporting

TARGET = "target_finish_position"
RACE_KEYS = ["season", "round"]
RANDOM_STATE = 42

# Pre-race-knowable features only. Raw outcome columns (finish_position, points,
# status) and identifiers are deliberately excluded.
FEATURE_COLS = [
    # grid & qualifying
    "grid_clean", "grid_position", "quali_best_seconds", "gap_to_pole_seconds",
    "quali_session_reached", "reached_q3", "quali_gap_to_teammate", "grid_penalty",
    # standings (pre-race)
    "championship_position", "championship_points", "championship_wins",
    "constructor_position", "constructor_points", "constructor_wins",
    "grid_vs_championship",
    # ratings
    "driver_elo_pre", "constructor_elo_pre", "perf_rating_pre",
    "driver_elo_experience",
    # rolling form
    "driver_avg_finish_5", "driver_avg_points_5", "driver_dnf_rate_5",
    "driver_form_races", "constructor_reliability_5",
    "constructor_points_per_race_5", "constructor_avg_pit_seconds_5",
    # circuit
    "circuit_is_street", "circuit_overtaking_index", "circuit_pole_win_pct",
    "circuit_dnf_rate_hist", "circuit_avg_pitstops_hist",
    "circuit_avg_stint_laps_hist", "circuit_history_races",
    # weather
    "air_temp_avg", "track_temp_avg", "humidity_avg", "wind_speed_avg",
    "rain_flag", "weather_missing",
]

# Carried alongside X for ranking / metric derivation (not model inputs).
META_COLS = [
    "season", "round", "driver_id", "driver_code", "constructor_id",
    "circuit_id", "grid_clean", "target_finish_position", "target_is_winner",
    "target_is_podium", "target_is_points", "target_is_dnf", "rain_flag",
]


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.features.build_features` first."
        )
    return pd.read_parquet(path)


def get_splits(
    path: Path = FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train<=2022, val==2023, test==2024) frames in time order."""
    df = load_features(path).sort_values(["race_order", "grid_clean"])
    train = df[df["season"] <= TRAIN_END].copy()
    val = df[df["season"] == VAL_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()
    logger.info(
        "Splits — train %d (<=%d) | val %d (%d) | test %d (%d)",
        len(train), TRAIN_END, len(val), VAL_SEASON, len(test), TEST_SEASON,
    )
    return train, val, test


class Preprocessor:
    """Median imputation (fit on training rows only) + missingness indicators.

    Produces a fully numeric, NaN-free, stably-ordered design matrix so that
    Random Forest and the LinearRegression meta-learner work and SHAP feature
    names stay aligned. Imputation statistics come only from the fit (training)
    data — never from validation or test — so there is no leakage.
    """

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.missing_cols_: list[str] = []
        self.columns_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = df[FEATURE_COLS]
        self.missing_cols_ = [c for c in FEATURE_COLS if X[c].isna().any()]
        self.medians_ = X.median()
        self.columns_ = FEATURE_COLS + [f"{c}__missing" for c in self.missing_cols_]
        logger.info(
            "Preprocessor fit: %d features, %d with missing values -> indicators",
            len(FEATURE_COLS), len(self.missing_cols_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("Preprocessor must be fit before transform().")
        out = df[FEATURE_COLS].copy()
        for c in self.missing_cols_:
            out[f"{c}__missing"] = df[c].isna().astype("int8")
        out = out.fillna(self.medians_).fillna(0.0)
        return out[self.columns_].reset_index(drop=True)


def design_matrix(
    df: pd.DataFrame, processor: Preprocessor
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) for a split. ``y`` is NaN for DNFs (kept for ranking)."""
    X = processor.transform(df)
    y = df[TARGET].reset_index(drop=True)
    meta = df[META_COLS].reset_index(drop=True)
    return X, y, meta


def classified(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Restrict to classified finishers (a real finishing position to regress)."""
    mask = y.notna()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
