# Qualifying-Prediction Model + Two-Stage Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second model that predicts the F1 qualifying grid itself (not just consuming it), and wire it into `run_predict.py` so a race can be forecast before real qualifying has happened.

**Architecture:** A new, fully isolated `src/models/quali_*` + `src/inference/*quali*` module family mirrors the existing race-model stack (temporal split, baseline, Optuna-tuned RF/XGB/LGBM stacking ensemble), trained on a quali-safe feature subset plus new leakage-free rolling-qualifying features derived from `features.parquet`'s existing `grid_clean` column. `run_predict.py` gains a `--predict-grid` flag that chains quali-model output straight into the existing, unmodified `build_pre_race_features` → `predict_race` path.

**Tech Stack:** Python, pandas, scikit-learn, XGBoost, LightGBM, Optuna, joblib, pytest.

## Global Constraints

- Temporal split is fixed project-wide: train seasons ≤2022, validate 2023, test 2024 (`src/models/dataset.py: TRAIN_END/VAL_SEASON/TEST_SEASON`). The qualifying model reuses these exact boundaries.
- `RANDOM_STATE = 42` (from `src/models/dataset.py`) for every stochastic step (RF, Optuna sampler, XGB/LGBM seeds).
- **No edits to `src/features/`, `src/models/dataset.py`, `src/models/train.py`'s public behavior, or `data/processed/features.parquet`.** The existing race model and its inference path must remain byte-identical in behavior.
- All new qualifying-model artefacts are saved under `models/quali_*.joblib` and `models/quali_manifest.json` — never overwrite `models/lgbm.joblib`, `models/rf.joblib`, `models/xgb.joblib`, `models/stack_meta.joblib`, `models/preprocessor.joblib`, or `models/manifest.json` (these back the current README numbers and the live dashboard).
- Every rolling feature must use `groupby(...).shift(1).rolling(window, min_periods=1)` — the current race's own row must never contribute to its own feature value (leakage discipline used throughout this codebase).
- Run tests with `python -m pytest tests/ -v` from the repo root (no `pytest.ini`/`conftest.py` exists yet; `-m pytest` puts the repo root on `sys.path` so `from src...` imports resolve, matching how `python -m src.features.build_features` already runs).
- **Never run `python -m src.models.train` for real during this plan.** It overwrites the production race-model artefacts backing the README's published numbers. Only the new `train_quali` entry point should ever be executed live.

---

### Task 1: Extract shared tuning helpers into `src/models/tuning.py`

**Files:**
- Create: `src/models/tuning.py`
- Modify: `src/models/train.py:1-129` (remove the extracted functions, import from `tuning` instead)
- Test: `tests/models/test_tuning.py`

**Interfaces:**
- Produces: `train_random_forest(X, y, random_state: int) -> RandomForestRegressor`, `tune_xgboost(Xtr, ytr, Xval, yval, n_trials: int, random_state: int) -> dict`, `tune_lightgbm(Xtr, ytr, Xval, yval, n_trials: int, random_state: int) -> dict`, `fit_xgb(params: dict, X, y, random_state: int) -> XGBRegressor`, `fit_lgbm(params: dict, X, y, random_state: int) -> LGBMRegressor`. These are consumed by both `src/models/train.py` (Task 1) and `src/models/train_quali.py` (Task 4).

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_tuning.py`:

```python
"""Smoke tests for the shared Optuna-tuned base-learner helpers.

Uses tiny synthetic data and n_trials=1 so this runs in under a second - it
checks wiring (return types, that fit_* consumes tune_*'s output), not tuning
quality. Tuning quality is judged later from the real training runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.tuning import (
    fit_lgbm, fit_xgb, train_random_forest, tune_lightgbm, tune_xgboost,
)


def _toy_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * 2 + rng.normal(scale=0.1, size=n))
    return X, y


def test_train_random_forest_returns_fitted_model():
    X, y = _toy_data()
    rf = train_random_forest(X, y, random_state=42)
    preds = rf.predict(X)
    assert len(preds) == len(y)


def test_tune_and_fit_xgboost_roundtrip():
    Xtr, ytr = _toy_data(seed=1)
    Xval, yval = _toy_data(seed=2)
    params = tune_xgboost(Xtr, ytr, Xval, yval, n_trials=1, random_state=42)
    model = fit_xgb(params, Xtr, ytr, random_state=42)
    assert len(model.predict(Xval)) == len(yval)


def test_tune_and_fit_lightgbm_roundtrip():
    Xtr, ytr = _toy_data(seed=1)
    Xval, yval = _toy_data(seed=2)
    params = tune_lightgbm(Xtr, ytr, Xval, yval, n_trials=1, random_state=42)
    model = fit_lgbm(params, Xtr, ytr, random_state=42)
    assert len(model.predict(Xval)) == len(yval)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_tuning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.tuning'`

- [ ] **Step 3: Create `src/models/tuning.py`**

```python
"""Shared Optuna-tuned base-learner helpers, reused by both the race model
(``src.models.train``) and the qualifying model (``src.models.train_quali``).

Pure functions over ``(X, y)`` with an explicit ``random_state`` - no coupling
to either pipeline's dataset module, so the same tuning/fitting logic isn't
duplicated between the two model families.
"""

from __future__ import annotations

import logging
import warnings

import optuna
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def train_random_forest(X, y, random_state: int) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, random_state=random_state,
    )
    rf.fit(X, y)
    return rf


def _tune(objective, n_trials: int, random_state: int) -> dict:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(Xtr, ytr, Xval_c, yval_c, n_trials: int, random_state: int) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        model = XGBRegressor(
            objective="reg:squarederror", random_state=random_state,
            n_jobs=-1, **params,
        )
        model.fit(Xtr, ytr)
        return mean_absolute_error(yval_c, model.predict(Xval_c))

    best = _tune(objective, n_trials, random_state)
    logger.info("XGBoost best params: %s", best)
    return best


def tune_lightgbm(Xtr, ytr, Xval_c, yval_c, n_trials: int, random_state: int) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = LGBMRegressor(
            random_state=random_state, n_jobs=-1, verbosity=-1, **params,
        )
        model.fit(Xtr, ytr)
        return mean_absolute_error(yval_c, model.predict(Xval_c))

    best = _tune(objective, n_trials, random_state)
    logger.info("LightGBM best params: %s", best)
    return best


def fit_xgb(params: dict, X, y, random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror", random_state=random_state, n_jobs=-1, **params,
    ).fit(X, y)


def fit_lgbm(params: dict, X, y, random_state: int) -> LGBMRegressor:
    return LGBMRegressor(
        random_state=random_state, n_jobs=-1, verbosity=-1, **params,
    ).fit(X, y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/models/test_tuning.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Refactor `src/models/train.py` to use the shared helpers**

Replace lines 20-129 of `src/models/train.py` (the imports block through the end of `fit_lgbm`) with:

```python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from .dataset import (
    RANDOM_STATE, Preprocessor, classified, design_matrix, get_splits,
)
from .metrics import evaluate_predictions, metrics_table
from .tuning import fit_lgbm, fit_xgb, train_random_forest, tune_lightgbm, tune_xgboost

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
```

Then update every call site in `train_all()` to pass `RANDOM_STATE` explicitly (it was previously closured from the module-level constant inside the now-removed local functions):

```python
    # --- 1. Random Forest ---
    logger.info("Training Random Forest...")
    rf = train_random_forest(Xtr_c, ytr_c, RANDOM_STATE)
    importances = (
        sorted(zip(Xtr_c.columns, rf.feature_importances_), key=lambda t: -t[1])
    )
    logger.info("Top RF feature importances:\n%s", "\n".join(
        f"  {f:32s} {imp:.4f}" for f, imp in importances[:15]))

    # --- 2. Optuna-tuned XGBoost & LightGBM ---
    logger.info("Tuning XGBoost (%d trials)...", n_trials)
    xgb_params = tune_xgboost(Xtr_c, ytr_c, Xval_c, yval_c, n_trials, RANDOM_STATE)
    xgb = fit_xgb(xgb_params, Xtr_c, ytr_c, RANDOM_STATE)

    logger.info("Tuning LightGBM (%d trials)...", n_trials)
    lgbm_params = tune_lightgbm(Xtr_c, ytr_c, Xval_c, yval_c, n_trials, RANDOM_STATE)
    lgbm = fit_lgbm(lgbm_params, Xtr_c, ytr_c, RANDOM_STATE)
```

Everything else in `src/models/train.py` (the stacking meta-learner fit, persistence, manifest, validation-progression table, `main()`) is unchanged.

**Do not run `python -m src.models.train`** to verify this — that would overwrite the production `models/*.joblib` artefacts. Instead verify by reading the diff: confirm `train_all()`'s logic is identical except for the `RANDOM_STATE` argument now being passed explicitly instead of closured, and that `optuna`/`warnings`/`LGBMRegressor`/`XGBRegressor`/`RandomForestRegressor`/`mean_absolute_error` imports have been removed from `train.py` (they now live only in `tuning.py`).

- [ ] **Step 6: Run the full test suite to confirm nothing broke**

Run: `python -m pytest tests/ -v`
Expected: PASS (still just the 3 tuning tests; train.py has no direct tests, verified by inspection per Step 5)

- [ ] **Step 7: Commit**

```bash
git add src/models/tuning.py src/models/train.py tests/models/test_tuning.py
git commit -m "Extract shared Optuna tuning helpers into src/models/tuning.py"
```

---

### Task 2: Qualifying dataset — rolling-quali features, splits, preprocessor

**Files:**
- Create: `src/models/quali_dataset.py`
- Test: `tests/models/test_quali_dataset.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TARGET = "grid_clean"`, `FEATURE_COLS: list[str]`, `META_COLS: list[str]`, `add_rolling_quali_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame`, `load_quali_features(path=FEATURES_PATH) -> pd.DataFrame`, `get_quali_splits(path=FEATURES_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`, `class QualiPreprocessor` (`.fit(df)`, `.transform(df)`), `quali_design_matrix(df, processor) -> tuple[X, y, meta]`. Consumed by Tasks 3, 4, 6.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_quali_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/test_quali_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.quali_dataset'`

- [ ] **Step 3: Create `src/models/quali_dataset.py`**

```python
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

from .dataset import FEATURES_PATH, RANDOM_STATE, TEST_SEASON, TRAIN_END, VAL_SEASON

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_quali_dataset.py -v`
Expected: PASS (7 tests; the 3 marked `skipif` run since `data/processed/features.parquet` exists in this repo)

- [ ] **Step 5: Commit**

```bash
git add src/models/quali_dataset.py tests/models/test_quali_dataset.py
git commit -m "Add qualifying dataset module with leakage-safe rolling-quali features"
```

---

### Task 3: Qualifying metrics + championship-order baseline

**Files:**
- Create: `src/models/quali_metrics.py`
- Create: `src/models/quali_baseline.py`
- Test: `tests/models/test_quali_metrics.py`
- Test: `tests/models/test_quali_baseline.py`

**Interfaces:**
- Consumes: `get_quali_splits` from Task 2 (`src.models.quali_dataset`).
- Produces: `evaluate_quali_predictions(meta: pd.DataFrame, pred: np.ndarray, name: str) -> dict`, `quali_metrics_table(rows: list[dict]) -> pd.DataFrame`, `baseline_predictions(df: pd.DataFrame) -> np.ndarray`, `run_baseline() -> pd.DataFrame`. Consumed by Tasks 4 and 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_quali_metrics.py`:

```python
"""Unit tests for qualifying-model evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
```

Create `tests/models/test_quali_baseline.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/test_quali_metrics.py tests/models/test_quali_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError` for both `src.models.quali_metrics` and `src.models.quali_baseline`

- [ ] **Step 3: Create `src/models/quali_metrics.py`**

```python
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
```

- [ ] **Step 4: Create `src/models/quali_baseline.py`**

```python
"""Stage 0 — the championship-order qualifying baseline.

"Predict grid order = current championship-standing order." This is the floor
every trained qualifying model must clear. Round 1 of each season has no
prior in-season standings (NaN), so those rows fall back to ranking by driver
ELO (which carries across seasons) instead of being left undefined.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from .quali_dataset import get_quali_splits
from .quali_metrics import evaluate_quali_predictions, quali_metrics_table

logger = logging.getLogger(__name__)


def baseline_predictions(df: pd.DataFrame) -> np.ndarray:
    """Rank each race by championship position, falling back to -ELO."""
    fallback = -df["driver_elo_pre"].fillna(1500.0)
    key = df["championship_position"].where(df["championship_position"].notna(), fallback)
    tmp = df[["season", "round"]].copy()
    tmp["key"] = key.to_numpy()
    return tmp.groupby(["season", "round"])["key"].rank(method="first").to_numpy()


def run_baseline() -> pd.DataFrame:
    _, val, test = get_quali_splits()
    rows = [
        evaluate_quali_predictions(val, baseline_predictions(val), "baseline_championship (val 2023)"),
        evaluate_quali_predictions(test, baseline_predictions(test), "baseline_championship (test 2024)"),
    ]
    return quali_metrics_table(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    table = run_baseline()
    print("\n=== Championship-order qualifying baseline ===")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_quali_metrics.py tests/models/test_quali_baseline.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add src/models/quali_metrics.py src/models/quali_baseline.py \
  tests/models/test_quali_metrics.py tests/models/test_quali_baseline.py
git commit -m "Add qualifying metrics and championship-order baseline"
```

---

### Task 4: Train the qualifying model (RF → Optuna XGB/LGBM → stacking)

**Files:**
- Create: `src/models/train_quali.py`

**Interfaces:**
- Consumes: `get_quali_splits`, `QualiPreprocessor`, `quali_design_matrix` (Task 2); `evaluate_quali_predictions`, `quali_metrics_table`, `baseline_predictions` (Task 3); `train_random_forest`, `tune_xgboost`, `tune_lightgbm`, `fit_xgb`, `fit_lgbm` (Task 1).
- Produces: `train_all(n_trials: int = 30) -> dict` (manifest dict); persists `models/quali_preprocessor.joblib`, `models/quali_rf.joblib`, `models/quali_xgb.joblib`, `models/quali_lgbm.joblib`, `models/quali_stack_meta.joblib`, `models/quali_manifest.json`. Consumed by Task 5 (evaluation) and Task 7 (inference).

There is no unit test for this task — it is an orchestration script over real Optuna tuning, exactly like `src/models/train.py` (which also has no test). Correctness is verified by actually running it in Step 2 below and inspecting the printed validation table.

- [ ] **Step 1: Create `src/models/train_quali.py`**

```python
"""Stage 1-3 — train Random Forest, Optuna-tuned XGBoost/LightGBM, and a
stack for the qualifying-prediction model.

Mirrors ``src.models.train``'s progression using the shared helpers in
``src.models.tuning``, but predicts ``grid_clean`` instead of finishing
position, over the quali-safe feature set (no qualifying/grid-derived
inputs — see ``src.models.quali_dataset``).

Run::

    python -m src.models.train_quali               # default 30 Optuna trials/model
    python -m src.models.train_quali --trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from .dataset import RANDOM_STATE
from .quali_baseline import baseline_predictions
from .quali_dataset import QualiPreprocessor, get_quali_splits, quali_design_matrix
from .quali_metrics import evaluate_quali_predictions, quali_metrics_table
from .tuning import fit_lgbm, fit_xgb, train_random_forest, tune_lightgbm, tune_xgboost

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def train_all(n_trials: int = 30) -> dict:
    train, val, test = get_quali_splits()
    pre = QualiPreprocessor().fit(train)

    Xtr, ytr, _ = quali_design_matrix(train, pre)
    Xval, yval, mval = quali_design_matrix(val, pre)
    logger.info("Quali training rows: %d | val rows: %d", len(Xtr), len(Xval))

    # --- 1. Random Forest ---
    logger.info("Training Random Forest...")
    rf = train_random_forest(Xtr, ytr, RANDOM_STATE)
    importances = sorted(zip(Xtr.columns, rf.feature_importances_), key=lambda t: -t[1])
    logger.info("Top RF feature importances:\n%s", "\n".join(
        f"  {f:32s} {imp:.4f}" for f, imp in importances[:15]))

    # --- 2. Optuna-tuned XGBoost & LightGBM ---
    logger.info("Tuning XGBoost (%d trials)...", n_trials)
    xgb_params = tune_xgboost(Xtr, ytr, Xval, yval, n_trials, RANDOM_STATE)
    xgb = fit_xgb(xgb_params, Xtr, ytr, RANDOM_STATE)

    logger.info("Tuning LightGBM (%d trials)...", n_trials)
    lgbm_params = tune_lightgbm(Xtr, ytr, Xval, yval, n_trials, RANDOM_STATE)
    lgbm = fit_lgbm(lgbm_params, Xtr, ytr, RANDOM_STATE)

    # --- 3. Stacking: meta-learner on held-out 2023 base predictions ---
    logger.info("Fitting stacking meta-learner on 2023 base predictions...")
    base_val = np.column_stack([rf.predict(Xval), xgb.predict(Xval), lgbm.predict(Xval)])
    meta = LinearRegression(positive=True).fit(base_val, yval)
    logger.info("Meta coefficients [rf, xgb, lgbm] = %s (intercept %.3f)",
                np.round(meta.coef_, 3), meta.intercept_)

    # --- persist ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, MODELS_DIR / "quali_preprocessor.joblib")
    joblib.dump(rf, MODELS_DIR / "quali_rf.joblib")
    joblib.dump(xgb, MODELS_DIR / "quali_xgb.joblib")
    joblib.dump(lgbm, MODELS_DIR / "quali_lgbm.joblib")
    joblib.dump(meta, MODELS_DIR / "quali_stack_meta.joblib")
    manifest = {
        "xgb_params": xgb_params,
        "lgbm_params": lgbm_params,
        "meta_coef": meta.coef_.tolist(),
        "meta_intercept": float(meta.intercept_),
        "n_features": Xtr.shape[1],
        "n_train_rows": int(len(Xtr)),
        "n_trials": n_trials,
        "rf_top_features": [f for f, _ in importances[:15]],
    }
    (MODELS_DIR / "quali_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Saved qualifying models + manifest to %s/", MODELS_DIR)

    # --- validation-season progression (test stays untouched until evaluate) ---
    def full_pred(model):
        return model.predict(pre.transform(val))

    base_val_all = np.column_stack([
        rf.predict(pre.transform(val)), xgb.predict(pre.transform(val)),
        lgbm.predict(pre.transform(val)),
    ])
    rows = [
        evaluate_quali_predictions(mval, baseline_predictions(val), "baseline_championship"),
        evaluate_quali_predictions(mval, full_pred(rf), "random_forest"),
        evaluate_quali_predictions(mval, full_pred(xgb), "xgboost"),
        evaluate_quali_predictions(mval, full_pred(lgbm), "lightgbm"),
        evaluate_quali_predictions(mval, meta.predict(base_val_all), "stack (in-sample val*)"),
    ]
    table = quali_metrics_table(rows)
    print("\n=== Qualifying model — validation 2023 progression (model selection) ===")
    print(table.to_string(index=False))
    print("* stack val metric is in-sample (meta trained on 2023); see test table.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train qualifying-prediction models")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per tuned model (default 30).")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    train_all(args.trials)
    logger.info("Training complete. Run `python -m src.models.evaluate_quali` for test metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it for real**

Run: `python -m src.models.train_quali`
Expected: Completes (this tunes 2 models × 30 Optuna trials each over ~4-5k rows — comparable runtime to the original `python -m src.models.train`, likely several minutes). Prints a validation-2023 progression table. Confirm at least one of `random_forest`/`xgboost`/`lightgbm`/`stack` has a lower `mae` than `baseline_championship` in that table — if not, inspect `add_rolling_quali_features` for a bug before proceeding (a qualifying model should comfortably beat ranking by raw championship position).

Confirm the new artefacts exist and the old ones are untouched:

Run: `python -c "from pathlib import Path; import time; print([p.name for p in Path('models').glob('quali_*')])"`
Expected: `['quali_lgbm.joblib', 'quali_manifest.json', 'quali_preprocessor.joblib', 'quali_rf.joblib', 'quali_stack_meta.joblib', 'quali_xgb.joblib']`

Run: `git status --short models/`
Expected: only `quali_*` files listed as untracked/new (models/ is gitignored, so this is just a visual sanity check that no `lgbm.joblib`/`rf.joblib`/etc. timestamp changed — compare with `ls -la models/*.joblib` before/after if in doubt).

- [ ] **Step 3: Commit**

```bash
git add src/models/train_quali.py
git commit -m "Add qualifying model training pipeline (RF -> Optuna XGB/LGBM -> stack)"
```

---

### Task 5: Evaluate the qualifying model on the 2024 test season

**Files:**
- Create: `src/models/evaluate_quali.py`

**Interfaces:**
- Consumes: `get_quali_splits` (Task 2); `baseline_predictions`, `evaluate_quali_predictions`, `quali_metrics_table` (Task 3); the joblib artefacts saved by Task 4.
- Produces: `run_evaluation() -> dict` (updated manifest); `reports/quali_metrics_test.csv`, `reports/quali_metrics_val.csv`, `reports/quali_predictions_2024.json`; updates `models/quali_manifest.json` with `best_model`, `test_mae`, `baseline_test_mae`, `improvement_pct`. This `test_mae`/`improvement_pct` is read back by `run_predict.py` in Task 8 to annotate predicted-grid output.

No unit test — this is an orchestration/reporting script over the real trained artefacts, exactly like `src/models/evaluate.py` (which also has no test). Verified by running it and reading the printed table.

- [ ] **Step 1: Create `src/models/evaluate_quali.py`**

```python
"""Stage — evaluate every qualifying model on the untouched 2024 test season.

Run::

    python -m src.models.evaluate_quali
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .quali_baseline import baseline_predictions
from .quali_dataset import get_quali_splits
from .quali_metrics import evaluate_quali_predictions, quali_metrics_table

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
RAW_RACES = Path("data/raw/races.csv")


def _load_models() -> dict:
    # joblib/pickle load is safe here: these artefacts are produced by our own
    # `train_quali.py` into a local, project-owned directory — not untrusted
    # input (same justification as src.models.evaluate._load_models).
    needed = ["quali_preprocessor", "quali_rf", "quali_xgb", "quali_lgbm", "quali_stack_meta"]
    missing = [n for n in needed if not (MODELS_DIR / f"{n}.joblib").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing qualifying model artefacts {missing}. Run "
            "`python -m src.models.train_quali`."
        )
    return {n: joblib.load(MODELS_DIR / f"{n}.joblib") for n in needed}


def _predict_all(models: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = models["quali_preprocessor"].transform(df)
    rf = models["quali_rf"].predict(X)
    xgb = models["quali_xgb"].predict(X)
    lgbm = models["quali_lgbm"].predict(X)
    stack = models["quali_stack_meta"].predict(np.column_stack([rf, xgb, lgbm]))
    return {
        "baseline_championship": baseline_predictions(df),
        "random_forest": rf,
        "xgboost": xgb,
        "lightgbm": lgbm,
        "stack": stack,
    }


def _table_for(meta: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    return quali_metrics_table(
        [evaluate_quali_predictions(meta, p, name) for name, p in preds.items()]
    )


def export_predictions(
    test: pd.DataFrame, pred: np.ndarray, best_model: str, out_path: Path
) -> None:
    df = test.reset_index(drop=True).copy()
    df["pred_grid"] = pred
    df["pred_rank"] = df.groupby(["season", "round"])["pred_grid"].rank(method="first")

    race_name = {}
    if RAW_RACES.exists():
        races = pd.read_csv(RAW_RACES)
        race_name = {(int(r.season), int(r.round)): r.race_name for r in races.itertuples()}

    records = []
    for r in df.itertuples():
        records.append({
            "season": int(r.season),
            "round": int(r.round),
            "race_name": race_name.get((int(r.season), int(r.round))),
            "driver_id": r.driver_id,
            "driver_code": r.driver_code,
            "constructor_id": r.constructor_id,
            "predicted_grid_raw": round(float(r.pred_grid), 3),
            "predicted_rank": int(r.pred_rank),
            "actual_grid": int(r.grid_clean),
        })

    payload = {
        "meta": {
            "test_season": int(df["season"].iloc[0]),
            "best_model": best_model,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_predictions": len(records),
        },
        "predictions": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Exported %d qualifying predictions -> %s", len(records), out_path)


def run_evaluation() -> dict:
    models = _load_models()
    _, val, test = get_quali_splits()

    val_preds = _predict_all(models, val)
    test_preds = _predict_all(models, test)
    val_table = _table_for(val, val_preds)
    test_table = _table_for(test, test_preds)

    ml = val_table[val_table["model"] != "baseline_championship"]
    best_model = ml.loc[ml["mae"].idxmin(), "model"]

    print("\n=== Qualifying model — validation 2023 (model selection) ===")
    print(val_table.to_string(index=False))
    print("\n=== Qualifying model — TEST 2024 (final, untouched) ===")
    print(test_table.to_string(index=False))

    base_mae = float(test_table.loc[test_table.model == "baseline_championship", "mae"].iloc[0])
    best_mae = float(test_table.loc[test_table.model == best_model, "mae"].iloc[0])
    improvement = 100.0 * (base_mae - best_mae) / base_mae
    print(f"\nBest qualifying model (selected on val): {best_model}")
    print(f"Test MAE: baseline {base_mae:.3f} -> {best_model} {best_mae:.3f} "
          f"({improvement:.1f}% better)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(REPORTS_DIR / "quali_metrics_test.csv", index=False)
    val_table.to_csv(REPORTS_DIR / "quali_metrics_val.csv", index=False)
    export_predictions(
        test, test_preds[best_model], best_model,
        REPORTS_DIR / "quali_predictions_2024.json",
    )

    tree_models = ["random_forest", "xgboost", "lightgbm"]
    best_tree = ml[ml["model"].isin(tree_models)].sort_values("mae").iloc[0]["model"]
    manifest_path = MODELS_DIR / "quali_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "best_model": best_model,
        "best_tree_model": best_tree,
        "test_mae": best_mae,
        "baseline_test_mae": base_mae,
        "improvement_pct": round(improvement, 2),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it for real**

Run: `python -m src.models.evaluate_quali`
Expected: Prints validation and test tables; confirm the printed "Best qualifying model" line shows a positive `improvement_pct` over `baseline_championship`. Confirm `reports/quali_metrics_test.csv`, `reports/quali_metrics_val.csv`, `reports/quali_predictions_2024.json` were created, and `models/quali_manifest.json` now has `test_mae`/`baseline_test_mae`/`improvement_pct` populated.

- [ ] **Step 3: Commit**

```bash
git add src/models/evaluate_quali.py
git commit -m "Add qualifying model evaluation on the 2024 test season"
```

(`reports/quali_*` and `models/quali_*` are gitignored like their race-model counterparts — nothing to add there.)

---

### Task 6: Pre-qualifying feature assembly for live inference

**Files:**
- Create: `src/inference/build_quali_features.py`
- Test: `tests/inference/test_build_quali_features.py`

**Interfaces:**
- Consumes: `FEATURE_COLS`, `load_quali_features` (Task 2, `src.models.quali_dataset`).
- Produces: `resolve_entry_list(season: int, features_path=FEATURES_PATH) -> list[str]`, `build_pre_quali_features(season, round_num, circuit_id, race_date, entries: list[str], features_path=FEATURES_PATH, rainfall=0.0) -> pd.DataFrame`. Consumed by Task 8 (`run_predict.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_build_quali_features.py`:

```python
"""Tests for pre-qualifying feature assembly.

Integration-style: uses the real data/processed/features.parquet, matching
how this project has no synthetic fixtures for its other inference tests
either (there are none pre-existing). Skipped if the parquet isn't present.
"""
from __future__ import annotations

from pathlib import Path

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/inference/test_build_quali_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.inference.build_quali_features'`

- [ ] **Step 3: Create `src/inference/build_quali_features.py`**

```python
"""Assemble a pre-QUALIFYING feature row per driver for an UPCOMING race.

Mirrors ``src.inference.build_race_features`` but for the qualifying model:
there is no grid/qualifying input here at all, since grid position is exactly
what this model predicts. Every column in
``src.models.quali_dataset.FEATURE_COLS`` (ratings, rolling finish + rolling
qualifying form, standings, circuit, weather) is either carried forward from
a driver's most recent historical row or synthesised from venue
history/climatology — never from the upcoming race's own qualifying result.

This module is read-only with respect to the trained pipeline — it imports
``FEATURE_COLS`` from ``quali_dataset`` so the row layout can never drift from
what the qualifying model expects.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.quali_dataset import FEATURE_COLS, load_quali_features

logger = logging.getLogger(__name__)

FEATURES_PATH = "data/processed/features.parquet"
ELO_BASE = 1500.0  # mirrors src.features.util.ELO_BASE — debutant prior

_CIRCUIT_COLS = [
    "circuit_is_street", "circuit_overtaking_index", "circuit_pole_win_pct",
    "circuit_dnf_rate_hist", "circuit_avg_pitstops_hist",
    "circuit_avg_stint_laps_hist", "circuit_history_races",
]
_WEATHER_NUM = ["air_temp_avg", "track_temp_avg", "humidity_avg", "wind_speed_avg"]

ID_COLS = ["season", "round", "circuit_id", "race_date",
           "driver_code", "driver_id", "constructor_id"]


def resolve_entry_list(season: int, features_path: str = FEATURES_PATH) -> list[str]:
    """Driver codes from the most recent COMPLETED race of ``season``.

    Falls back to the previous season's final race if ``season`` has no rows
    yet (e.g. querying before that season's round 1 has been ingested).
    """
    feat = pd.read_parquet(features_path)
    pool = feat[feat["season"] == season]
    if pool.empty:
        pool = feat[feat["season"] == season - 1]
    if pool.empty:
        raise ValueError(
            f"No historical data for season {season} or {season - 1} in "
            f"{features_path} to infer an entry list. Pass --entries explicitly."
        )
    latest_round = pool["race_order"].max()
    entries = sorted(
        pool.loc[pool["race_order"] == latest_round, "driver_code"].unique().tolist()
    )
    logger.info("Resolved entry list for season %s from race_order %s: %d drivers.",
                season, latest_round, len(entries))
    return entries


def _latest_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per driver_code, indexed by driver_code."""
    return (
        feat.sort_values("race_order")
        .groupby("driver_code", as_index=False)
        .tail(1)
        .set_index("driver_code")
    )


def _circuit_profile(feat: pd.DataFrame, circuit_id: str, rainfall: float) -> dict:
    """Circuit characteristics + weather climatology for the venue."""
    hist = feat[feat["circuit_id"] == circuit_id]
    prof: dict[str, float] = {}

    if len(hist):
        last = hist.sort_values("race_order").iloc[-1]
        for c in _CIRCUIT_COLS:
            prof[c] = float(last[c]) if pd.notna(last[c]) else np.nan
        measured = hist[hist["weather_missing"] == 0]
        src = measured if len(measured) else hist
        for c in _WEATHER_NUM:
            prof[c] = float(src[c].mean()) if src[c].notna().any() else np.nan
        prof["weather_missing"] = 0 if len(measured) else 1
    else:
        logger.warning(
            "No history for circuit_id=%s — using median imputation for its "
            "circuit/weather features.", circuit_id,
        )
        for c in _CIRCUIT_COLS + _WEATHER_NUM:
            prof[c] = np.nan
        prof["weather_missing"] = 1

    prof["rain_flag"] = 1 if (rainfall and rainfall > 0) else 0
    return prof


def build_pre_quali_features(
    season: int,
    round_num: int,
    circuit_id: str,
    race_date: str,
    entries: list[str],
    features_path: str = FEATURES_PATH,
    rainfall: float = 0.0,
) -> pd.DataFrame:
    """Build one qualifying-model-ready feature row per driver in ``entries``.

    Unlike ``build_pre_race_features``, there is no grid/qualifying input —
    grid position is exactly what this model predicts.
    """
    feat = load_quali_features(features_path)
    latest = _latest_rows(feat)
    circuit = _circuit_profile(feat, circuit_id, rainfall)

    rows: list[dict] = []
    debutants: list[str] = []

    for raw_code in entries:
        code = raw_code.strip().upper()
        if code in latest.index:
            src = latest.loc[code]
            row = src[FEATURE_COLS].astype(float).to_dict()
            driver_id = src["driver_id"]
            constructor_id = src["constructor_id"]
        else:
            debutants.append(code)
            row = {c: np.nan for c in FEATURE_COLS}
            row.update({
                "driver_elo_pre": ELO_BASE,
                "constructor_elo_pre": ELO_BASE,
                "perf_rating_pre": ELO_BASE,
                "driver_elo_experience": 0.0,
                "driver_form_races": 0.0,
            })
            driver_id = code.lower()
            constructor_id = "unknown"

        for c in _CIRCUIT_COLS + _WEATHER_NUM + ["rain_flag", "weather_missing"]:
            row[c] = circuit[c]

        row.update({
            "season": int(season), "round": int(round_num),
            "circuit_id": circuit_id, "race_date": race_date,
            "driver_code": code, "driver_id": driver_id,
            "constructor_id": constructor_id,
        })
        rows.append(row)

    out = pd.DataFrame(rows, columns=FEATURE_COLS + ID_COLS)
    if debutants:
        logger.info(
            "%d debutant(s) with no history -> neutral prior: %s",
            len(debutants), ", ".join(debutants),
        )
    logger.info("Built pre-qualifying features: %d drivers x %d feature cols",
                len(out), len(FEATURE_COLS))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/inference/test_build_quali_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/inference/build_quali_features.py tests/inference/test_build_quali_features.py
git commit -m "Add pre-qualifying feature assembly for live inference"
```

---

### Task 7: Predict the qualifying grid

**Files:**
- Create: `src/inference/predict_quali.py`
- Test: `tests/inference/test_predict_quali.py`

**Interfaces:**
- Consumes: `models/quali_*.joblib` (Task 4); output of `build_pre_quali_features` (Task 6).
- Produces: `predict_quali(features_df: pd.DataFrame) -> pd.DataFrame` (columns `OUTPUT_COLS = ["predicted_grid", "driver_code", "constructor_id", "predicted_position_raw"]`), `predicted_grid_dict(pred: pd.DataFrame) -> dict[str, int]`. Consumed by Task 8 (`run_predict.py`), where `predicted_grid_dict`'s output is passed as the `grid` argument to the existing `build_pre_race_features`.

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_predict_quali.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/inference/test_predict_quali.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.inference.predict_quali'`

- [ ] **Step 3: Create `src/inference/predict_quali.py`**

```python
"""Run the trained qualifying model on pre-qualifying feature rows and rank
the field into a predicted starting grid.

Loads the qualifying stacking ensemble (Random Forest + XGBoost + LightGBM +
LinearRegression meta-learner, the same architecture as the race model) and
the fitted qualifying preprocessor, predicts an expected grid slot per
driver, then ranks the field into a valid 1..N grid permutation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
RF_FILE = "quali_rf.joblib"
XGB_FILE = "quali_xgb.joblib"
LGBM_FILE = "quali_lgbm.joblib"
STACK_FILE = "quali_stack_meta.joblib"
PREPROCESSOR_FILE = "quali_preprocessor.joblib"

OUTPUT_COLS = ["predicted_grid", "driver_code", "constructor_id", "predicted_position_raw"]


def _load() -> dict:
    # joblib/pickle load is safe here: these artefacts are produced by our own
    # `src.models.train_quali` into a local, project-owned directory — not
    # untrusted input (same justification as src.inference.predict_race._load).
    files = [PREPROCESSOR_FILE, RF_FILE, XGB_FILE, LGBM_FILE, STACK_FILE]
    for f in files:
        if not (MODELS_DIR / f).exists():
            raise FileNotFoundError(
                f"Missing {MODELS_DIR / f}. Train the qualifying model first "
                "(`python -m src.models.train_quali`)."
            )
    return {
        "preprocessor": joblib.load(MODELS_DIR / PREPROCESSOR_FILE),
        "rf": joblib.load(MODELS_DIR / RF_FILE),
        "xgb": joblib.load(MODELS_DIR / XGB_FILE),
        "lgbm": joblib.load(MODELS_DIR / LGBM_FILE),
        "stack": joblib.load(MODELS_DIR / STACK_FILE),
    }


def predict_quali(features_df: pd.DataFrame) -> pd.DataFrame:
    """Predict and rank the qualifying grid for an upcoming session.

    Parameters
    ----------
    features_df
        Output of ``build_pre_quali_features`` (one row per driver).

    Returns
    -------
    DataFrame sorted by predicted grid slot with columns ``OUTPUT_COLS``.
    ``predicted_grid`` is always a valid 1..N permutation for the field size.
    """
    models = _load()
    X = models["preprocessor"].transform(features_df)
    rf = models["rf"].predict(X)
    xgb = models["xgb"].predict(X)
    lgbm = models["lgbm"].predict(X)
    stack = models["stack"].predict(np.column_stack([rf, xgb, lgbm]))

    out = features_df[["driver_code", "constructor_id"]].copy().reset_index(drop=True)
    out["predicted_position_raw"] = np.round(stack, 3)
    out["predicted_grid"] = out["predicted_position_raw"].rank(method="first").astype(int)
    out = out.sort_values("predicted_grid").reset_index(drop=True)
    logger.info("Predicted qualifying grid for %d drivers; predicted pole: %s",
                len(out), out.iloc[0]["driver_code"])
    return out[OUTPUT_COLS]


def predicted_grid_dict(pred: pd.DataFrame) -> dict[str, int]:
    """``{driver_code: grid_slot}`` — the same shape ``build_pre_race_features``
    expects for its ``grid`` argument."""
    return dict(zip(pred["driver_code"], pred["predicted_grid"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/inference/test_predict_quali.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite so far**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (tuning, quali_dataset, quali_metrics, quali_baseline, build_quali_features, predict_quali)

- [ ] **Step 6: Commit**

```bash
git add src/inference/predict_quali.py tests/inference/test_predict_quali.py
git commit -m "Add qualifying-grid prediction and ranking"
```

---

### Task 8: Wire `--predict-grid` into `run_predict.py`

**Files:**
- Modify: `run_predict.py` (full file, currently 191 lines — see below for exact replacements)
- Test: `tests/test_run_predict_cli.py`

**Interfaces:**
- Consumes: `resolve_entry_list`, `build_pre_quali_features` (Task 6); `predict_quali`, `predicted_grid_dict` (Task 7); existing `fetch_qualifying_grid`, `find_next_race`, `build_pre_race_features`, `predict_race` (unchanged).
- Produces: `--predict-grid` / `--entries` CLI flags; `payload["meta"]["grid_source"]` (`"real" | "predicted" | "manual"`) and `payload["quali_prediction"]` in the saved JSON when the grid was predicted.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_predict_cli.py`:

```python
"""Unit tests for run_predict.py's argument wiring and grid-source branching.

These monkeypatch the heavy pipeline functions (fetch/predict/build) so the
test exercises only the CLI logic: mutual-exclusivity validation, and which
branch sets ``grid_source`` in the saved JSON.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import run_predict


def _stub_race_result() -> pd.DataFrame:
    return pd.DataFrame({
        "predicted_rank": [1], "driver_code": ["VER"], "constructor_id": ["red_bull"],
        "grid_position": [1], "predicted_position_raw": [1.2],
        "win_probability": [0.5], "podium_probability": [0.8], "points_probability": [0.9],
    })


def _stub_quali_pred() -> pd.DataFrame:
    return pd.DataFrame({
        "predicted_grid": [1], "driver_code": ["VER"],
        "constructor_id": ["red_bull"], "predicted_position_raw": [1.1],
    })


@pytest.fixture(autouse=True)
def stub_heavy_functions(monkeypatch, tmp_path):
    monkeypatch.setattr(run_predict, "build_pre_race_features",
                         lambda **kw: pd.DataFrame({"driver_code": ["VER"]}))
    monkeypatch.setattr(run_predict, "predict_race", lambda df: _stub_race_result())
    monkeypatch.setattr(run_predict, "REPORTS_DIR", tmp_path)
    return tmp_path


def test_manual_mode_requires_a_grid_source():
    with pytest.raises(SystemExit):
        run_predict.main(["--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02"])


def test_manual_mode_rejects_two_grid_sources():
    with pytest.raises(SystemExit):
        run_predict.main([
            "--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02",
            "--grid", "VER:1", "--auto-grid",
        ])


def test_predict_grid_sets_grid_source_predicted(monkeypatch, stub_heavy_functions):
    monkeypatch.setattr(run_predict, "resolve_entry_list", lambda season: ["VER"])
    monkeypatch.setattr(run_predict, "build_pre_quali_features",
                         lambda *a, **kw: pd.DataFrame({"driver_code": ["VER"]}))
    monkeypatch.setattr(run_predict, "predict_quali", lambda df: _stub_quali_pred())
    monkeypatch.setattr(run_predict, "predicted_grid_dict", lambda pred: {"VER": 1})

    run_predict.main([
        "--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02",
        "--predict-grid",
    ])

    payload = json.loads((stub_heavy_functions / "prediction_2024_bahrain.json").read_text())
    assert payload["meta"]["grid_source"] == "predicted"
    assert "quali_prediction" in payload


def test_manual_grid_sets_grid_source_manual(stub_heavy_functions):
    run_predict.main([
        "--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02",
        "--grid", "VER:1",
    ])

    payload = json.loads((stub_heavy_functions / "prediction_2024_bahrain.json").read_text())
    assert payload["meta"]["grid_source"] == "manual"
    assert "quali_prediction" not in payload


def test_next_race_falls_back_to_predicted_grid_on_value_error(monkeypatch, stub_heavy_functions):
    from src.inference.schedule import NextRace

    nxt = NextRace(season=2026, round=10, circuit_id="spa", race_date="2026-07-19",
                   country="Belgium", tag="belgium", days_until=13)
    monkeypatch.setattr(run_predict, "find_next_race", lambda: nxt)

    def _raise(*a, **kw):
        raise ValueError("no quali yet")

    monkeypatch.setattr(run_predict, "fetch_qualifying_grid", _raise)
    monkeypatch.setattr(run_predict, "resolve_entry_list", lambda season: ["VER"])
    monkeypatch.setattr(run_predict, "build_pre_quali_features",
                         lambda *a, **kw: pd.DataFrame({"driver_code": ["VER"]}))
    monkeypatch.setattr(run_predict, "predict_quali", lambda df: _stub_quali_pred())
    monkeypatch.setattr(run_predict, "predicted_grid_dict", lambda pred: {"VER": 1})

    run_predict.main(["--next-race"])

    payload = json.loads((stub_heavy_functions / "prediction_2026_belgium.json").read_text())
    assert payload["meta"]["grid_source"] == "predicted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_predict_cli.py -v`
Expected: FAIL — `AttributeError` (e.g. `<module 'run_predict'> does not have the attribute 'resolve_entry_list'`) since `--predict-grid` and its imports don't exist in `run_predict.py` yet, and `test_manual_mode_requires_a_grid_source` currently passes (only 2 sources exist today) rather than failing — confirming the test suite exercises the not-yet-built 3-way validation.

- [ ] **Step 3: Update `run_predict.py`**

Replace the imports block (lines 38-46):

```python
from src.inference.build_quali_features import build_pre_quali_features, resolve_entry_list
from src.inference.build_race_features import build_pre_race_features
from src.inference.predict_quali import predict_quali, predicted_grid_dict
from src.inference.predict_race import predict_race
from src.inference.qualifying import fetch_qualifying_grid
from src.inference.schedule import find_next_race

RACES_CSV = Path("data/raw/races.csv")
REPORTS_DIR = Path("reports")
logger = logging.getLogger("predict")
```

Replace the module docstring's usage examples (lines 8-27) to document the new flag:

```python
Usage::

    # Manual grid:
    python run_predict.py --season 2026 --circuit red_bull_ring \
      --date 2026-06-28 --rainfall 0 \
      --grid "VER:1,NOR:2,LEC:3,HAM:4,PIA:5,RUS:6,SAI:7,ALO:8,ANT:9,TSU:10,\
LAW:11,STR:12,HUL:13,BEA:14,OCO:15,GAS:16,DOO:17,HAD:18,BOR:19,MAG:20"

    # Real qualifying grid, fetched automatically (ingests the season if needed):
    python run_predict.py --season 2026 --round 9 --circuit silverstone \
      --date 2026-07-06 --auto-grid

    # Fully automatic — detect the next race, fetch/predict its grid, predict:
    python run_predict.py --next-race

    # Predict the grid too (before real qualifying exists):
    python run_predict.py --season 2026 --round 10 --circuit spa \
      --date 2026-07-19 --predict-grid

The round number and a friendly output filename are looked up from
``data/raw/races.csv`` when the season is present there; otherwise pass
``--round`` and/or ``--out`` explicitly. ``--auto-grid``/``--predict-grid``
need a round number to look up qualifying/build features, so pass ``--round``
when the race isn't in ``races.csv`` yet. ``--predict-grid`` uses a second,
independent model that forecasts the qualifying grid itself — see
"Honest limitations" in the README; it is strictly less reliable than a real
qualifying grid (``--auto-grid``).
"""
```

Replace the full `main()` function (lines 95-186) with:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict an upcoming F1 race.")
    parser.add_argument("--next-race", action="store_true",
                        help="auto-detect the next upcoming race from races.csv, "
                             "fetch its qualifying grid (or predict it if "
                             "qualifying hasn't happened yet), and predict. Needs "
                             "no other flags.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--circuit", default=None, help="circuit_id, e.g. red_bull_ring")
    parser.add_argument("--date", default=None, help="race date YYYY-MM-DD")
    parser.add_argument("--rainfall", type=float, default=0.0,
                        help="forecast rainfall (>0 sets the wet-race flag).")
    parser.add_argument("--grid", default=None,
                        help='starting grid as "CODE:POS,CODE:POS,..." '
                             "(omit if using --auto-grid/--predict-grid).")
    parser.add_argument("--auto-grid", action="store_true",
                        help="fetch the real qualifying grid from "
                             "data/raw/qualifying.csv for --season/--round, "
                             "ingesting the season first if it's missing.")
    parser.add_argument("--predict-grid", action="store_true",
                        help="predict the qualifying grid with the qualifying "
                             "model instead of using a real one (for races "
                             "that haven't qualified yet). See --entries.")
    parser.add_argument("--entries", default=None,
                        help='comma-separated driver codes for --predict-grid, '
                             'e.g. "VER,NOR,LEC,...". Defaults to the entry '
                             "list from the season's most recent completed "
                             "race.")
    parser.add_argument("--round", type=int, default=None,
                        help="round number (else looked up from races.csv). "
                             "Required for --auto-grid/--predict-grid if the "
                             "race isn't in races.csv yet.")
    parser.add_argument("--features-path", default="data/processed/features.parquet")
    parser.add_argument("--out", default=None, help="output JSON path.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )

    quali_pred = None  # set to a DataFrame when the grid came from the quali model

    if args.next_race:
        # Fully automatic: detect the next race, then fetch/predict its grid.
        nxt = find_next_race()
        season, round_num = nxt.season, nxt.round
        circuit_id, date_str, tag = nxt.circuit_id, nxt.race_date, nxt.tag
        if nxt.days_until <= 3 and args.rainfall == 0.0:
            print(f"\n[!] {circuit_id} is {nxt.days_until} day(s) away and no "
                  "--rainfall was set. If rain is forecast, re-run with "
                  "--rainfall <mm> to enable the wet-race flag.\n")
        try:
            grid = fetch_qualifying_grid(season, round_num)
            grid_source = "real"
        except ValueError:
            print(f"\n[i] No real qualifying available yet for {circuit_id} — "
                  "predicting the grid with the qualifying model instead.\n")
            entries = resolve_entry_list(season)
            quali_features = build_pre_quali_features(
                season, round_num, circuit_id, date_str, entries,
                rainfall=args.rainfall,
            )
            quali_pred = predict_quali(quali_features)
            grid = predicted_grid_dict(quali_pred)
            grid_source = "predicted"
    else:
        missing = [name for name, val in (("--season", args.season),
                   ("--circuit", args.circuit), ("--date", args.date)) if not val]
        if missing:
            parser.error(f"{', '.join(missing)} required unless --next-race is used.")
        modes_set = sum([bool(args.grid), args.auto_grid, args.predict_grid])
        if modes_set != 1:
            parser.error("provide exactly one of --grid, --auto-grid, or --predict-grid.")
        season, circuit_id, date_str = args.season, args.circuit, args.date
        looked_round, tag = lookup_race(season, circuit_id)
        round_num = args.round if args.round is not None else (looked_round or 0)
        if looked_round is None and args.round is None:
            logger.warning("Race not found in races.csv for season %s circuit %s; "
                           "using round=%d (pass --round for the real round).",
                           season, circuit_id, round_num)
        if args.auto_grid:
            if round_num == 0:
                parser.error("--auto-grid needs a round number; pass --round N "
                             "(the race isn't in races.csv yet).")
            grid = fetch_qualifying_grid(season, round_num)
            grid_source = "real"
        elif args.predict_grid:
            entries = ([c.strip().upper() for c in args.entries.split(",")]
                       if args.entries else resolve_entry_list(season))
            quali_features = build_pre_quali_features(
                season, round_num, circuit_id, date_str, entries,
                rainfall=args.rainfall,
            )
            quali_pred = predict_quali(quali_features)
            grid = predicted_grid_dict(quali_pred)
            grid_source = "predicted"
        else:
            grid = parse_grid(args.grid)
            grid_source = "manual"

    features_df = build_pre_race_features(
        season=season, round_num=round_num, circuit_id=circuit_id,
        race_date=date_str, grid=grid, features_path=args.features_path,
        rainfall=args.rainfall,
    )
    result = predict_race(features_df)

    print(f"\n=== Predicted {season} round {round_num} — {circuit_id} "
          f"({date_str}, rainfall={args.rainfall}, grid={grid_source}) ===")
    print(format_table(result))

    out_path = Path(args.out) if args.out else REPORTS_DIR / f"prediction_{season}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "season": season,
            "round": round_num,
            "circuit_id": circuit_id,
            "race_date": date_str,
            "rainfall": args.rainfall,
            "model": "lightgbm",
            "grid_source": grid_source,
            "n_drivers": len(result),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "predictions": result.to_dict(orient="records"),
    }
    if quali_pred is not None:
        quali_manifest_path = Path("models/quali_manifest.json")
        quali_test_mae = None
        if quali_manifest_path.exists():
            quali_test_mae = json.loads(quali_manifest_path.read_text()).get("test_mae")
        payload["quali_prediction"] = {
            "model_test_mae": quali_test_mae,
            "predicted_grid": quali_pred.to_dict(orient="records"),
        }
        print(f"\n[i] Grid was PREDICTED (qualifying model test MAE: "
              f"{quali_test_mae}), not real — race forecast is rougher than "
              "usual on top of the model's own error.")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_predict_cli.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS across every module built in Tasks 1–8.

- [ ] **Step 6: Manual end-to-end smoke test**

Run: `python run_predict.py --season 2024 --round 1 --circuit bahrain --date 2024-03-02 --predict-grid`
Expected: Prints a predicted grid note, a ranked race table, and `Saved -> reports/prediction_2024_bahrain.json`. Open that JSON and confirm `meta.grid_source == "predicted"` and a `quali_prediction` block is present.

- [ ] **Step 7: Commit**

```bash
git add run_predict.py tests/test_run_predict_cli.py
git commit -m "Wire --predict-grid two-stage pipeline into run_predict.py"
```

---

### Task 9: Document the qualifying model in the README

**Files:**
- Modify: `README.md`

**Interfaces:** None (documentation only).

- [ ] **Step 1: Add a new section after "Quick start" (before "## Project structure")**

Insert into `README.md` immediately before the `## Project structure` heading:

```markdown
## Qualifying-prediction model (two-stage forecasting)

The race model above needs a real starting grid as input. A second,
independent model (`src/models/quali_*`, `src/inference/*quali*`) predicts
**the grid itself** — trained the same way (temporal split, Optuna-tuned
RF/XGB/LGBM stack) on a qualifying-safe feature set (ratings, rolling finish
form, standings, circuit, weather, plus new rolling-qualifying features) that
excludes every column derived from qualifying/grid data.

Chain it into a race forecast **before qualifying has happened** with:

```bash
# Train + evaluate the qualifying model once (writes models/quali_*.joblib)
python -m src.models.train_quali
python -m src.models.evaluate_quali

# Predict the grid, then the race, in one command
python run_predict.py --season 2026 --round 10 --circuit spa \
  --date 2026-07-19 --predict-grid
```

`--next-race` uses this automatically as a fallback whenever real qualifying
isn't available yet. The output JSON's `meta.grid_source` is `"real"`,
`"predicted"`, or `"manual"` so consumers can tell which kind of forecast
they're looking at, and a `quali_prediction` block is included whenever the
grid was predicted.

**This is strictly less reliable than a real grid.** Qualifying pace is
dominated by current-season car development, which the carried-forward
rating/form features only approximate. `--predict-grid` forecasts inherit the
qualifying model's own error on top of the race model's — swap to
`--auto-grid` once real qualifying results exist for a genuine forecast.
```

- [ ] **Step 2: Update the "Project structure" code block**

In the `models/` line of the `src/` tree (inside the fenced block under `## Project structure`), replace:

```
├── models/      dataset.py, train.py, evaluate.py, explain.py (SHAP), baseline.py
└── inference/   build_race_features.py, predict_race.py   — live forecasting
```

with:

```
├── models/      dataset.py, train.py, evaluate.py, explain.py (SHAP), baseline.py
│                quali_dataset.py, train_quali.py, evaluate_quali.py, quali_baseline.py — qualifying model
└── inference/   build_race_features.py, predict_race.py,
                 build_quali_features.py, predict_quali.py  — live forecasting (race + qualifying)
```

- [ ] **Step 3: Add a bullet to "Honest limitations"**

In the `## Honest limitations` section, add a fourth bullet after the existing three:

```markdown
- **`--predict-grid` forecasts stack two models' error.** The qualifying model beats a championship-order baseline by a real but modest margin — qualifying pace is a noisier target than race finish. A race forecast built from a *predicted* grid is strictly rougher than one built from a real grid; use `--auto-grid` once qualifying has actually happened.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document the qualifying-prediction model and --predict-grid pipeline"
```
