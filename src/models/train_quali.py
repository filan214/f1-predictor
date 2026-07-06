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
