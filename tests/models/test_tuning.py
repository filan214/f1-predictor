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
