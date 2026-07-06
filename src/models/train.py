"""Stage 1-3 — train Random Forest, Optuna-tuned XGBoost/LightGBM, and a stack.

Progressive complexity, every stage benchmarked against the last:

1. **Random Forest** (sensible defaults) — mainly for feature-importance intuition.
2. **XGBoost** and **LightGBM** — each tuned by Optuna, minimising MAE on the
   2023 validation season (trained on 2010-2022). No shuffle; the holdout is the
   next season in time.
3. **Stacking** — base models predict the held-out 2023 season; those
   out-of-sample predictions train a LinearRegression meta-learner, which never
   sees base predictions on data the bases were trained on.

All artefacts are persisted to ``models/`` for evaluation and explanation.
Run::

    python -m src.models.train               # default 30 Optuna trials/model
    python -m src.models.train --trials 50
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

from .dataset import (
    RANDOM_STATE, Preprocessor, classified, design_matrix, get_splits,
)
from .metrics import evaluate_predictions, metrics_table
from .tuning import fit_lgbm, fit_xgb, train_random_forest, tune_lightgbm, tune_xgboost

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_all(n_trials: int = 30) -> dict:
    train, val, test = get_splits()
    pre = Preprocessor().fit(train)

    Xtr, ytr, _ = design_matrix(train, pre)
    Xval, yval, mval = design_matrix(val, pre)
    Xtr_c, ytr_c = classified(Xtr, ytr)
    Xval_c, yval_c = classified(Xval, yval)
    logger.info("Training rows (classified): %d | val classified: %d",
                len(Xtr_c), len(Xval_c))

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

    # --- 3. Stacking: meta-learner on held-out 2023 base predictions ---
    logger.info("Fitting stacking meta-learner on 2023 base predictions...")
    base_val = np.column_stack([
        rf.predict(Xval_c), xgb.predict(Xval_c), lgbm.predict(Xval_c),
    ])
    # Non-negative weights: the base predictions are highly correlated, so an
    # unconstrained fit produces unstable +/- coefficients that overfit the small
    # validation set. A non-negative blend is stable and interpretable.
    meta = LinearRegression(positive=True).fit(base_val, yval_c)
    logger.info("Meta coefficients [rf, xgb, lgbm] = %s (intercept %.3f)",
                np.round(meta.coef_, 3), meta.intercept_)

    # --- persist ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(rf, MODELS_DIR / "rf.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgb.joblib")
    joblib.dump(lgbm, MODELS_DIR / "lgbm.joblib")
    joblib.dump(meta, MODELS_DIR / "stack_meta.joblib")
    manifest = {
        "xgb_params": xgb_params,
        "lgbm_params": lgbm_params,
        "meta_coef": meta.coef_.tolist(),
        "meta_intercept": float(meta.intercept_),
        "n_features": Xtr_c.shape[1],
        "n_train_classified": int(len(Xtr_c)),
        "n_trials": n_trials,
        "rf_top_features": [f for f, _ in importances[:15]],
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Saved models + manifest to %s/", MODELS_DIR)

    # --- validation-season progression (test stays untouched until evaluate) ---
    def full_pred(model):
        return model.predict(pre.transform(val))

    base_val_all = np.column_stack([
        rf.predict(pre.transform(val)),
        xgb.predict(pre.transform(val)),
        lgbm.predict(pre.transform(val)),
    ])
    rows = [
        evaluate_predictions(mval, val["grid_clean"].to_numpy(), "baseline_grid"),
        evaluate_predictions(mval, full_pred(rf), "random_forest"),
        evaluate_predictions(mval, full_pred(xgb), "xgboost"),
        evaluate_predictions(mval, full_pred(lgbm), "lightgbm"),
        evaluate_predictions(mval, meta.predict(base_val_all), "stack (in-sample val*)"),
    ]
    table = metrics_table(rows)
    print("\n=== Validation 2023 progression (model selection) ===")
    print(table.to_string(index=False))
    print("* stack val metric is in-sample (meta trained on 2023); see test table.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train F1 models")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per tuned model (default 30).")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    train_all(args.trials)
    logger.info("Training complete. Run `python -m src.models.evaluate` for test metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
