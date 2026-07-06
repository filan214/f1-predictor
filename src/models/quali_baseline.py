"""Stage 0 — the championship-order qualifying baseline.

"Predict grid order = current championship-standing order." This is the floor
every trained qualifying model must clear. Round 1 of each season has no
prior in-season standings (NaN), so those rows fall back to ranking by driver
ELO (which carries across seasons) instead of being left undefined.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from .quali_dataset import get_quali_splits
from .quali_metrics import evaluate_quali_predictions, quali_metrics_table

logger = logging.getLogger(__name__)


def baseline_predictions(df: pd.DataFrame) -> np.ndarray:
    """Rank each race by championship position; drivers with no standings
    (NaN — e.g. round 1, or a mid-season debutant) sort AFTER classified
    drivers, ordered among themselves by driver ELO (carried across seasons).

    Ranks on two keys via a stable sort — championship position (missing ->
    +inf, i.e. last) then -ELO as a tie-break — rather than a single
    mixed-scale key, so a partial-NaN race can't put a standings-less driver
    spuriously on pole.
    """
    work = df[["season", "round"]].copy()
    work["_cp"] = df["championship_position"].fillna(np.inf)
    work["_elo"] = -df["driver_elo_pre"].fillna(1500.0)
    work["_orig"] = np.arange(len(work))
    ranked = work.sort_values(["season", "round", "_cp", "_elo", "_orig"])
    ranked["_rank"] = ranked.groupby(["season", "round"]).cumcount() + 1
    return ranked.sort_values("_orig")["_rank"].to_numpy(dtype=float)


def run_baseline() -> pd.DataFrame:
    _, val, test = get_quali_splits()
    rows = [
        evaluate_quali_predictions(val, baseline_predictions(val), "baseline_championship (val 2023)"),
        evaluate_quali_predictions(test, baseline_predictions(test), "baseline_championship (test 2024)"),
    ]
    return quali_metrics_table(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )
    table = run_baseline()
    print("\n=== Championship-order qualifying baseline ===")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
