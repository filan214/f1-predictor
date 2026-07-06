"""Shared Optuna-tuned base-learner helpers, reused by both the race model
(``src.models.train``) and the qualifying model (``src.models.train_quali``).

Pure functions over ``(X, y)`` with an explicit ``random_state`` (defaulting to
the project-wide ``RANDOM_STATE`` seed from ``src.models.dataset``), so the same
tuning/fitting logic isn't duplicated between the two model families and both
seed their estimators identically.
"""

from __future__ import annotations

import logging
import warnings

import optuna
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from .dataset import RANDOM_STATE

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def train_random_forest(X, y, random_state: int = RANDOM_STATE) -> RandomForestRegressor:
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


def tune_xgboost(Xtr, ytr, Xval_c, yval_c, n_trials: int, random_state: int = RANDOM_STATE) -> dict:
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


def tune_lightgbm(Xtr, ytr, Xval_c, yval_c, n_trials: int, random_state: int = RANDOM_STATE) -> dict:
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


def fit_xgb(params: dict, X, y, random_state: int = RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror", random_state=random_state, n_jobs=-1, **params,
    ).fit(X, y)


def fit_lgbm(params: dict, X, y, random_state: int = RANDOM_STATE) -> LGBMRegressor:
    return LGBMRegressor(
        random_state=random_state, n_jobs=-1, verbosity=-1, **params,
    ).fit(X, y)
