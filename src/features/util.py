"""Shared helpers and constants for feature engineering.

Centralises the bits every feature module needs: raw-table loading, a single
canonical *chronological* race ordering (so trailing windows mean the same
thing everywhere), lap-time parsing, and small domain constants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
ROLL_WINDOW = 5          # trailing races for rolling "form" features
ELO_BASE = 1500.0        # starting rating for every new driver / constructor

# A driver who DNFs has no finishing position. For *form* features we still
# want a retirement to count as a poor result, so we impute it to the back of
# a nominal 20-car grid. The separate ``*_dnf_rate`` features let a model
# disentangle "slow" from "unreliable".
DNF_FILL_POSITION = 20

# Circuits run on temporary public-road layouts. Used for the binary
# ``circuit_is_street`` feature. Keys are Ergast/Jolpica ``circuit_id`` values
# as they actually appear in ``races.csv`` (verified against the ingested data).
# Albert Park (parkland) and Montreal (semi-permanent) are deliberately left
# out as ambiguous; the clear-cut street circuits are included.
STREET_CIRCUITS = frozenset(
    {"monaco", "marina_bay", "baku", "jeddah", "valencia", "miami", "vegas"}
)


# --------------------------------------------------------------------------- #
# Raw I/O
# --------------------------------------------------------------------------- #
def load_raw(name: str) -> pd.DataFrame:
    """Load a raw table by stem (e.g. ``"results"``) from ``data/raw/``."""
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw table '{name}' not found at {path}. Run ingestion first."
        )
    df = pd.read_csv(path)
    logger.debug("Loaded %s: %d rows", name, len(df))
    return df


# --------------------------------------------------------------------------- #
# Chronological ordering
# --------------------------------------------------------------------------- #
def race_chronology(races: pd.DataFrame) -> pd.DataFrame:
    """Return ``[season, round, race_date, race_order]`` sorted by real date.

    ``race_order`` is a dense 0-based index over every race in the dataset,
    ordered by actual calendar date (falling back to season+round when a date
    is missing). Every feature module joins on this so that "the previous five
    races" means the same chronological thing across modules.
    """
    cols = ["season", "round", "race_date"]
    chrono = races[cols].copy()
    chrono["race_date"] = pd.to_datetime(chrono["race_date"], errors="coerce")
    # Sort by date primarily; season/round break ties and cover missing dates.
    chrono = chrono.sort_values(
        ["race_date", "season", "round"], na_position="last"
    ).reset_index(drop=True)
    chrono["race_order"] = range(len(chrono))
    return chrono


def attach_race_order(df: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Left-join the canonical ``race_order`` (and ``race_date``) onto ``df``."""
    chrono = race_chronology(races)
    return df.merge(chrono, on=["season", "round"], how="left")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_lap_time(value: object) -> float:
    """Parse a lap time string into seconds.

    Handles ``"1:30.031"`` (m:ss.mmm), plain ``"30.5"`` seconds, and
    missing/blank values (-> NaN).
    """
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    try:
        if ":" in s:
            mins, secs = s.split(":", 1)
            return int(mins) * 60 + float(secs)
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def is_dnf_series(finish_position: pd.Series) -> pd.Series:
    """1 where a driver did not record a classified finishing position."""
    return finish_position.isna().astype("int8")
