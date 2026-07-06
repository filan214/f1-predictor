"""Stage 0 — the grid-position baseline.

"Predict finishing position = starting grid position." This is the floor every
later model must clear. Measured here on the validation (2023) and test (2024)
seasons so the rest of the pipeline has a concrete number to beat.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from .dataset import get_splits
from .metrics import evaluate_predictions, metrics_table

logger = logging.getLogger(__name__)


def baseline_predictions(df: pd.DataFrame) -> np.ndarray:
    """The baseline simply predicts each driver finishes where they start."""
    return df["grid_clean"].to_numpy()


def run_baseline() -> pd.DataFrame:
    _, val, test = get_splits()
    rows = [
        evaluate_predictions(val, baseline_predictions(val), "baseline_grid (val 2023)"),
        evaluate_predictions(test, baseline_predictions(test), "baseline_grid (test 2024)"),
    ]
    return metrics_table(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    table = run_baseline()
    print("\n=== Grid-position baseline ===")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
