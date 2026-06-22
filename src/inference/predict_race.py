"""Run the trained model on pre-race feature rows and rank the field.

Loads the canonical LightGBM model (the one selected on the 2023 validation
season) and the fitted preprocessor, predicts an expected finishing position per
driver, then derives win / podium / points probabilities.

Probabilities come from a small Monte-Carlo simulation: each driver's finishing
position is drawn many times from ``Normal(predicted_position, sigma)``, the
field is ranked within every draw, and we count how often each driver lands P1 /
top-3 / top-10. This respects the structural constraint that exactly one driver
wins, three reach the podium, and ten score — so the three probabilities are
mutually consistent (unlike independent per-driver sigmoids). ``sigma`` is set
to the model's approximate test residual spread.

NOTE on artefact names: the task brief referred to ``models/lightgbm.joblib`` and
``models/preprocessor.json``; the actual trained artefacts produced by
``src.models.train`` are ``models/lgbm.joblib`` and ``models/preprocessor.joblib``
(a joblib pickle, not JSON). We load the real files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODEL_FILE = "lgbm.joblib"             # canonical LightGBM (selected on 2023 val)
PREPROCESSOR_FILE = "preprocessor.joblib"

# ~ residual std implied by the model's ~2.2 MAE on the 2024 test season
# (for a roughly-Gaussian error, std ~= MAE * 1.25).
POSITION_SIGMA = 2.8
N_SIM = 20_000
RANDOM_STATE = 42

OUTPUT_COLS = [
    "predicted_rank", "driver_code", "constructor_id", "grid_position",
    "predicted_position_raw", "win_probability", "podium_probability",
    "points_probability",
]


def _load():
    """Load the fitted preprocessor and LightGBM model.

    joblib/pickle load is safe here: these artefacts are produced by our own
    ``src.models.train`` into a local, project-owned directory — not untrusted
    input. (Unpickling the preprocessor imports ``src.models.dataset`` for its
    class definition; it does not modify it.)
    """
    for f in (PREPROCESSOR_FILE, MODEL_FILE):
        if not (MODELS_DIR / f).exists():
            raise FileNotFoundError(
                f"Missing {MODELS_DIR / f}. Train the model first "
                "(`python -m src.models.train`)."
            )
    pre = joblib.load(MODELS_DIR / PREPROCESSOR_FILE)
    model = joblib.load(MODELS_DIR / MODEL_FILE)
    return pre, model


def _simulate_probabilities(
    raw: np.ndarray, sigma: float = POSITION_SIGMA,
    n: int = N_SIM, seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo win/podium/points probabilities from predicted positions."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=raw[None, :], scale=sigma, size=(n, raw.shape[0]))
    # Rank within each simulated race (1 = best). argsort-of-argsort -> ranks.
    ranks = draws.argsort(axis=1).argsort(axis=1) + 1
    win = (ranks == 1).mean(axis=0)
    podium = (ranks <= 3).mean(axis=0)
    points = (ranks <= 10).mean(axis=0)
    return win, podium, points


def predict_race(features_df: pd.DataFrame) -> pd.DataFrame:
    """Predict and rank an upcoming race.

    Parameters
    ----------
    features_df
        Output of ``build_pre_race_features`` (one row per driver, containing
        every ``FEATURE_COLS`` column plus ``driver_code``/``constructor_id``/
        ``grid_position``).

    Returns
    -------
    DataFrame sorted by predicted finishing order with columns ``OUTPUT_COLS``.
    """
    pre, model = _load()
    X = pre.transform(features_df)
    raw = np.asarray(model.predict(X), dtype=float)

    out = features_df[["driver_code", "constructor_id", "grid_position"]].copy()
    out = out.reset_index(drop=True)
    out["grid_position"] = out["grid_position"].astype(int)
    out["predicted_position_raw"] = np.round(raw, 3)

    win, podium, points = _simulate_probabilities(raw)
    out["win_probability"] = np.round(win, 4)
    out["podium_probability"] = np.round(podium, 4)
    out["points_probability"] = np.round(points, 4)

    out["predicted_rank"] = out["predicted_position_raw"].rank(method="first").astype(int)
    out = out.sort_values("predicted_rank").reset_index(drop=True)
    logger.info("Predicted %d drivers; pole-to-flag favourite: %s",
                len(out), out.iloc[0]["driver_code"])
    return out[OUTPUT_COLS]
