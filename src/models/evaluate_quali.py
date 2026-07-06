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
