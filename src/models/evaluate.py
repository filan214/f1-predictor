"""Stage — evaluate every model on the untouched 2024 test season.

Loads the persisted models, reproduces the temporal design matrices, and reports
the full PRD metric suite for the baseline and every model in one table. Model
*selection* uses the 2023 validation season (so the 2024 test set never informs
which model "wins"); the winner's predictions are exported as structured JSON.

Run::

    python -m src.models.evaluate
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

from .dataset import get_splits
from .metrics import WIN_PROB_TEMPERATURE, evaluate_predictions, metrics_table

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
RAW_RACES = Path("data/raw/races.csv")


# --------------------------------------------------------------------------- #
# Prediction assembly
# --------------------------------------------------------------------------- #
def _load_models() -> dict:
    # joblib/pickle load is safe here: these artefacts are produced by our own
    # `train.py` into a local, project-owned directory — not untrusted input.
    needed = ["preprocessor", "rf", "xgb", "lgbm", "stack_meta"]
    missing = [n for n in needed if not (MODELS_DIR / f"{n}.joblib").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artefacts {missing}. Run `python -m src.models.train`."
        )
    return {n: joblib.load(MODELS_DIR / f"{n}.joblib") for n in needed}


def _predict_all(models: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predicted finishing position per row for every model on ``df``."""
    X = models["preprocessor"].transform(df)
    rf = models["rf"].predict(X)
    xgb = models["xgb"].predict(X)
    lgbm = models["lgbm"].predict(X)
    stack = models["stack_meta"].predict(np.column_stack([rf, xgb, lgbm]))
    return {
        "baseline_grid": df["grid_clean"].to_numpy(),
        "random_forest": rf,
        "xgboost": xgb,
        "lightgbm": lgbm,
        "stack": stack,
    }


def _table_for(meta: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    return metrics_table(
        [evaluate_predictions(meta, p, name) for name, p in preds.items()]
    )


# --------------------------------------------------------------------------- #
# Structured JSON export
# --------------------------------------------------------------------------- #
def _win_probabilities(df: pd.DataFrame, pred: np.ndarray, temp: float) -> np.ndarray:
    """Softmax over negative predicted positions, normalised within each race."""
    out = np.zeros(len(df), dtype=float)
    tmp = df[["season", "round"]].reset_index(drop=True).copy()
    tmp["pred"] = pred
    for _, idx in tmp.groupby(["season", "round"]).groups.items():
        pos = tmp.loc[idx, "pred"].to_numpy()
        logits = -pos / temp
        logits -= logits.max()
        p = np.exp(logits)
        out[list(idx)] = p / p.sum()
    return out


def export_predictions(
    test: pd.DataFrame, pred: np.ndarray, best_model: str, out_path: Path
) -> None:
    df = test.reset_index(drop=True).copy()
    df["pred_position"] = pred
    df["pred_rank"] = df.groupby(["season", "round"])["pred_position"].rank(method="first")
    df["win_probability"] = _win_probabilities(df, pred, WIN_PROB_TEMPERATURE)

    # Friendly race name from the raw schedule, if available.
    race_name = {}
    if RAW_RACES.exists():
        races = pd.read_csv(RAW_RACES)
        race_name = {
            (int(r.season), int(r.round)): r.race_name
            for r in races.itertuples()
        }

    records = []
    for r in df.itertuples():
        records.append({
            "season": int(r.season),
            "round": int(r.round),
            "race_name": race_name.get((int(r.season), int(r.round))),
            "driver_id": r.driver_id,
            "driver_code": r.driver_code,
            "constructor_id": r.constructor_id,
            "grid": None if pd.isna(r.grid_clean) else int(r.grid_clean),
            "predicted_position": round(float(r.pred_position), 3),
            "predicted_rank": int(r.pred_rank),
            "win_probability": round(float(r.win_probability), 4),
            "predicted_podium": bool(r.pred_rank <= 3),
            "predicted_points": bool(r.pred_rank <= 10),
            "actual_finish": None if pd.isna(r.target_finish_position)
            else int(r.target_finish_position),
            "actual_dnf": bool(r.target_is_dnf),
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
    logger.info("Exported %d predictions -> %s", len(records), out_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_evaluation() -> dict:
    models = _load_models()
    _, val, test = get_splits()

    val_preds = _predict_all(models, val)
    test_preds = _predict_all(models, test)
    val_table = _table_for(val, val_preds)
    test_table = _table_for(test, test_preds)

    # Select the winner on validation (never on test).
    ml = val_table[val_table["model"] != "baseline_grid"]
    best_model = ml.loc[ml["mae"].idxmin(), "model"]

    print("\n=== Validation 2023 (model selection) ===")
    print(val_table.to_string(index=False))
    print("\n=== TEST 2024 (final, untouched) ===")
    print(test_table.to_string(index=False))

    base_mae = float(test_table.loc[test_table.model == "baseline_grid", "mae"].iloc[0])
    best_mae = float(test_table.loc[test_table.model == best_model, "mae"].iloc[0])
    improvement = 100.0 * (base_mae - best_mae) / base_mae
    print(f"\nBest model (selected on val): {best_model}")
    print(f"Test MAE: baseline {base_mae:.3f} -> {best_model} {best_mae:.3f} "
          f"({improvement:.1f}% better) | PRD goal <=1.5")

    # Persist artefacts.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(REPORTS_DIR / "model_metrics_test.csv", index=False)
    val_table.to_csv(REPORTS_DIR / "model_metrics_val.csv", index=False)
    export_predictions(
        test, test_preds[best_model], best_model,
        REPORTS_DIR / "predictions_2024.json",
    )

    # Record selection (and the best tree model for SHAP) in the manifest.
    tree_models = ["random_forest", "xgboost", "lightgbm"]
    best_tree = ml[ml["model"].isin(tree_models)].sort_values("mae").iloc[0]["model"]
    manifest_path = MODELS_DIR / "manifest.json"
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
