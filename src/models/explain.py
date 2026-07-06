"""Stage 4 — SHAP explainability for the best tree model.

SHAP's TreeExplainer needs a tree model, so we explain the strongest *base*
learner (the stack's meta-learner is a thin linear blend over these). Produces:

* a global importance bar plot and beeswarm over the 2024 test season,
* per-prediction waterfall plots for a few narrative case studies
  (a surprise podium, a wet-race drive, a notable under-performer),
* a JSON of the case-study breakdowns for the write-up.

Run::

    python -m src.models.explain
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

from .dataset import get_splits  # noqa: E402

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
PLOTS_DIR = Path("reports/shap_plots")
_MODEL_FILE = {"random_forest": "rf", "xgboost": "xgb", "lightgbm": "lgbm"}


def _load_best_tree() -> tuple[str, object, object]:
    # Trusted local artefacts produced by our own train.py (see evaluate.py note).
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    best_tree = manifest.get("best_tree_model", "xgboost")
    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / f"{_MODEL_FILE[best_tree]}.joblib")
    logger.info("Explaining best tree model: %s", best_tree)
    return best_tree, model, pre


def _case_studies(test: pd.DataFrame, pred: np.ndarray) -> list[dict]:
    """Pick a few interesting, correctly-handled 2024 predictions."""
    df = test.reset_index(drop=True).copy()
    df["pred"] = pred
    df["pred_rank"] = df.groupby(["season", "round"])["pred"].rank(method="first")
    df["row"] = np.arange(len(df))
    cases = []

    # 1. Surprise podium: started >= P6 but actually finished top-3.
    surprise = df[(df["grid_clean"] >= 6) & (df["target_is_podium"] == 1)]
    if len(surprise):
        s = surprise.sort_values("grid_clean", ascending=False).iloc[0]
        cases.append(("surprise_podium", s))

    # 2. Wet race: a strong drive in the rain.
    wet = df[(df["rain_flag"] == 1) & (df["target_is_points"] == 1)]
    if len(wet):
        w = wet.sort_values("pred").iloc[0]
        cases.append(("wet_race_points", w))

    # 3. Pole-to-win (cleanly predicted front-runner).
    front = df[(df["grid_clean"] == 1) & (df["target_is_winner"] == 1)]
    if len(front):
        cases.append(("pole_to_win", front.iloc[0]))

    # Fallback to ensure >= 3 cases: best-predicted finishers.
    while len(cases) < 3:
        extra = df.sort_values("pred").iloc[len(cases)]
        cases.append((f"top_runner_{len(cases)}", extra))

    return [(label, int(s["row"]), s) for label, s in cases[:3]]


def run_explain(sample: int = 400) -> None:
    best_tree, model, pre = _load_best_tree()
    _, _, test = get_splits()
    X_test = pre.transform(test)

    explainer = shap.TreeExplainer(model)
    expl = explainer(X_test)  # Explanation over the full test season
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- global importance ---
    plt.figure()
    shap.plots.bar(expl, max_display=15, show=False)
    plt.title(f"SHAP global importance — {best_tree} (2024 test)")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "global_importance.png", dpi=120)
    plt.close()

    plt.figure()
    shap.plots.beeswarm(expl, max_display=15, show=False)
    plt.title(f"SHAP beeswarm — {best_tree} (2024 test)")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "beeswarm.png", dpi=120)
    plt.close()
    logger.info("Saved global SHAP plots to %s/", PLOTS_DIR)

    # --- case studies ---
    narratives = []
    for label, row, info in _case_studies(test, model.predict(X_test)):
        plt.figure()
        shap.plots.waterfall(expl[row], max_display=12, show=False)
        title = (f"{info['driver_code']} — {info['constructor_id']} "
                 f"(grid {int(info['grid_clean'])}, finished "
                 f"{'DNF' if pd.isna(info['target_finish_position']) else int(info['target_finish_position'])})")
        plt.title(f"{label}: {title}", fontsize=9)
        plt.tight_layout(); plt.savefig(PLOTS_DIR / f"case_{label}.png", dpi=120)
        plt.close()

        contrib = sorted(
            zip(X_test.columns, expl[row].values), key=lambda t: -abs(t[1])
        )[:6]
        narratives.append({
            "case": label,
            "driver_code": info["driver_code"],
            "constructor_id": info["constructor_id"],
            "grid": int(info["grid_clean"]),
            "actual_finish": None if pd.isna(info["target_finish_position"])
            else int(info["target_finish_position"]),
            "predicted_position": round(float(model.predict(X_test)[row]), 2),
            "base_value": round(float(expl[row].base_values), 3),
            "top_contributions": [
                {"feature": f, "shap": round(float(v), 3)} for f, v in contrib
            ],
        })
        logger.info("Case %s: %s", label, title)

    (PLOTS_DIR / "case_studies.json").write_text(json.dumps(narratives, indent=2))
    logger.info("Saved %d case studies + narratives to %s/", len(narratives), PLOTS_DIR)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_explain()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
