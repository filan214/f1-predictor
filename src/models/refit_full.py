"""Stage 5 — fold the 2023 validation season into final training.

Model *selection* is finished (see ``models/manifest.json``: LightGBM was the
chosen model, with Optuna-tuned XGBoost/LightGBM hyperparameters). With the
choice made, the principled next step is to stop holding 2023 out and train the
final base learners on the **full 2010-2023** history. The 2024 season stays
completely untouched as the test set, so this remains leakage-free — 2024 never
informs any fit, hyperparameter, or weight.

Two things change versus ``train.py``:

1. **Bases** (Random Forest, XGBoost, LightGBM) are refit on 2010-2023 using the
   *already-selected* hyperparameters (re-tuning is skipped: the choice is done,
   and we no longer keep a clean held-out season to tune against).
2. **Stacking meta** can no longer use a single held-out 2023 season for its
   out-of-sample base predictions. Instead we build **TimeSeriesSplit OOF**
   predictions across the whole 2010-2023 span: in each expanding-window fold the
   bases are fit on the past and predict the next block, so every meta training
   row is an out-of-fold base prediction. The final bases (fit on all of
   2010-2023) are then used for 2024 inference.

This module is deliberately non-destructive: it reads the tuned params from the
existing manifest but does **not** overwrite the canonical ``models/`` artefacts
or ``reports/model_metrics_test.csv`` (those document the literal-guide split
behind notebook 03 and ``predictions_2024.json``). It writes new report files.

Run::

    python -m src.models.refit_full              # 5 TimeSeriesSplit OOF folds
    python -m src.models.refit_full --splits 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from .dataset import Preprocessor, classified, design_matrix, get_splits
from .metrics import evaluate_predictions, metrics_table
from .train import fit_lgbm, fit_xgb, train_random_forest

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
TREE_NAMES = ("random_forest", "xgboost", "lightgbm")


# --------------------------------------------------------------------------- #
# Tuned hyperparameters (already selected — reused, not re-tuned)
# --------------------------------------------------------------------------- #
def _load_tuned_params() -> tuple[dict, dict]:
    """Read the Optuna-selected XGBoost/LightGBM params from the manifest."""
    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run `python -m src.models.train` first "
            "so the selected hyperparameters exist to reuse."
        )
    manifest = json.loads(manifest_path.read_text())
    try:
        return manifest["xgb_params"], manifest["lgbm_params"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            "manifest.json is missing tuned params; re-run `src.models.train`."
        ) from exc


# --------------------------------------------------------------------------- #
# Base learners
# --------------------------------------------------------------------------- #
def _fit_bases(X, y, xgb_params: dict, lgbm_params: dict) -> dict:
    """Fit all three base learners on (X, y) with the fixed selected params."""
    return {
        "random_forest": train_random_forest(X, y),
        "xgboost": fit_xgb(xgb_params, X, y),
        "lightgbm": fit_lgbm(lgbm_params, X, y),
    }


def _predict_bases(bases: dict, X) -> np.ndarray:
    """Column-stack base predictions in a stable [rf, xgb, lgbm] order."""
    return np.column_stack([bases[name].predict(X) for name in TREE_NAMES])


# --------------------------------------------------------------------------- #
# Stacking meta via TimeSeriesSplit OOF over the full 2010-2023 span
# --------------------------------------------------------------------------- #
def _oof_meta(
    X: pd.DataFrame, y: pd.Series, xgb_params: dict, lgbm_params: dict, n_splits: int
) -> tuple[LinearRegression, int]:
    """Train the meta-learner on out-of-fold base predictions.

    ``X``/``y`` must be in chronological order (they are: ``get_splits`` sorts by
    ``race_order`` and we concatenate train-then-val). Each fold fits the bases on
    the earlier block and predicts the next, so no meta row sees a base trained on
    its own data. Rows in the first fold get no OOF prediction and are dropped.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full((len(X), len(TREE_NAMES)), np.nan)
    for k, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        fold_bases = _fit_bases(
            X.iloc[tr_idx], y.iloc[tr_idx], xgb_params, lgbm_params
        )
        oof[te_idx] = _predict_bases(fold_bases, X.iloc[te_idx])
        logger.info(
            "OOF fold %d/%d: train %d -> predict %d",
            k, n_splits, len(tr_idx), len(te_idx),
        )

    mask = ~np.isnan(oof).any(axis=1)
    n_used = int(mask.sum())
    # Non-negative blend, identical rationale to train.py: the base predictions
    # are highly correlated, so an unconstrained fit gives unstable +/- weights.
    meta = LinearRegression(positive=True).fit(oof[mask], y.to_numpy()[mask])
    logger.info(
        "Meta trained on %d OOF rows; coef [rf, xgb, lgbm] = %s (intercept %.3f)",
        n_used, np.round(meta.coef_, 3), meta.intercept_,
    )
    return meta, n_used


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def refit_full(n_splits: int = 5) -> pd.DataFrame:
    xgb_params, lgbm_params = _load_tuned_params()
    train, val, test = get_splits()

    # Fold 2023 in: full training span = 2010-2023, still chronological.
    full = pd.concat([train, val], ignore_index=True)
    logger.info(
        "Final training span: %d rows (2010-2023) = train %d + val %d; "
        "test held out: %d (2024)", len(full), len(train), len(val), len(test),
    )

    # Preprocessor fit on the full training span only (2024 never seen).
    pre = Preprocessor().fit(full)
    Xf, yf, _ = design_matrix(full, pre)
    Xf_c, yf_c = classified(Xf, yf)
    logger.info("Classified training rows (folded): %d", len(Xf_c))

    # Meta-learner from TimeSeriesSplit OOF predictions over 2010-2023.
    logger.info("Building TimeSeriesSplit OOF meta (%d folds)...", n_splits)
    meta, _ = _oof_meta(Xf_c, yf_c, xgb_params, lgbm_params, n_splits)

    # Final bases: refit on ALL classified 2010-2023 for 2024 inference.
    logger.info("Refitting final base learners on full 2010-2023...")
    bases = _fit_bases(Xf_c, yf_c, xgb_params, lgbm_params)

    # --- Evaluate on the untouched 2024 test season ---
    X_test = pre.transform(test)
    base_test = _predict_bases(bases, X_test)
    preds = {
        "baseline_grid": test["grid_clean"].to_numpy(),
        "random_forest": base_test[:, 0],
        "xgboost": base_test[:, 1],
        "lightgbm": base_test[:, 2],
        "stack": meta.predict(base_test),
    }
    table = metrics_table(
        [evaluate_predictions(test, p, name) for name, p in preds.items()]
    )
    return table


def _comparison(folded: pd.DataFrame) -> pd.DataFrame | None:
    """Side-by-side original-split vs folded MAE, if the original CSV exists."""
    orig_path = REPORTS_DIR / "model_metrics_test.csv"
    if not orig_path.exists():
        return None
    orig = pd.read_csv(orig_path)[["model", "mae", "spearman"]]
    merged = orig.merge(folded[["model", "mae", "spearman"]], on="model",
                        suffixes=("_2010_2022", "_2010_2023"))
    merged["mae_delta"] = (
        merged["mae_2010_2023"] - merged["mae_2010_2022"]
    ).round(4)
    merged["mae_pct_change"] = (
        100.0 * merged["mae_delta"] / merged["mae_2010_2022"]
    ).round(2)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fold 2023 into training and re-evaluate on 2024."
    )
    parser.add_argument("--splits", type=int, default=5,
                        help="TimeSeriesSplit folds for the OOF meta (default 5).")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    warnings.filterwarnings("ignore", category=UserWarning)
    table = refit_full(args.splits)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "model_metrics_test_folded.csv"
    table.to_csv(out_path, index=False)
    print("\n=== TEST 2024 - bases folded to 2010-2023 (untouched test) ===")
    print(table.to_string(index=False))

    base_mae = float(table.loc[table.model == "baseline_grid", "mae"].iloc[0])
    for headline in ("lightgbm", "stack"):
        m = float(table.loc[table.model == headline, "mae"].iloc[0])
        print(f"{headline:>10}: MAE {m:.3f}  "
              f"({100 * (base_mae - m) / base_mae:.1f}% better than grid {base_mae:.3f})")
    print(f"\nWrote {out_path}")

    cmp = _comparison(table)
    if cmp is not None:
        cmp_path = REPORTS_DIR / "model_metrics_test_comparison.csv"
        cmp.to_csv(cmp_path, index=False)
        print("\n=== Original (2010-2022) vs folded (2010-2023) on 2024 test ===")
        print(cmp.to_string(index=False))
        print(f"Wrote {cmp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
