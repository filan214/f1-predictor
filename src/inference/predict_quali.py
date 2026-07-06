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
