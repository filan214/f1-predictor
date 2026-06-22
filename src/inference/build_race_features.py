"""Assemble a pre-race feature row per driver for an UPCOMING race.

The trained model expects exactly the columns in
``src.models.dataset.FEATURE_COLS``. For a race that hasn't happened we cannot
read those features from ``features.parquet`` directly, so we synthesise them
from information that *is* knowable before lights-out:

* **Carried forward** from each driver's most recent historical row — driver &
  constructor ELO, rolling form, championship standings, constructor reliability.
  (These are exactly the "as of before the race" quantities the features were
  built to be, so carrying the latest one forward is leakage-free and current.)
* **Set from the grid** supplied by qualifying — grid position, grid penalty,
  grid-vs-championship delta, and a Q-session proxy from the grid slot.
* **Set from circuit history** — the venue's overtaking index, pole-win %, DNF
  rate, pit/stint characteristics, taken from its most recent running.
* **Set from circuit weather climatology** — historical average temps/humidity/
  wind for the venue, with ``rain_flag`` driven by the caller's forecast.

Debutants with no history (e.g. 2025/2026 rookies) start from a neutral prior:
baseline 1500 ELO, zero experience, and median-imputed form (flagged by the
preprocessor's missingness indicators, exactly as in training).

This module is read-only with respect to the trained pipeline — it imports
``FEATURE_COLS`` so the row layout can never drift from what the model expects.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Read-only import: the single source of truth for the model's input columns.
from src.models.dataset import FEATURE_COLS

logger = logging.getLogger(__name__)

FEATURES_PATH = "data/processed/features.parquet"
ELO_BASE = 1500.0  # mirrors src.features.util.ELO_BASE — debutant prior

# Columns we synthesise fresh for the upcoming race. Everything else in
# FEATURE_COLS is carried forward from the driver's latest historical row.
_CIRCUIT_COLS = [
    "circuit_is_street", "circuit_overtaking_index", "circuit_pole_win_pct",
    "circuit_dnf_rate_hist", "circuit_avg_pitstops_hist",
    "circuit_avg_stint_laps_hist", "circuit_history_races",
]
_WEATHER_NUM = ["air_temp_avg", "track_temp_avg", "humidity_avg", "wind_speed_avg"]

# Identifier columns carried alongside the features for downstream output.
ID_COLS = ["season", "round", "circuit_id", "race_date",
           "driver_code", "driver_id", "constructor_id"]


def _latest_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per driver_code, indexed by driver_code."""
    return (
        feat.sort_values("race_order")
        .groupby("driver_code", as_index=False)
        .tail(1)
        .set_index("driver_code")
    )


def _circuit_profile(feat: pd.DataFrame, circuit_id: str, rainfall: float) -> dict:
    """Circuit characteristics + weather climatology for the venue."""
    hist = feat[feat["circuit_id"] == circuit_id]
    prof: dict[str, float] = {}

    if len(hist):
        last = hist.sort_values("race_order").iloc[-1]
        for c in _CIRCUIT_COLS:
            prof[c] = float(last[c]) if pd.notna(last[c]) else np.nan
        # Weather climatology: average the venue's *measured* weather rows.
        measured = hist[hist["weather_missing"] == 0]
        src = measured if len(measured) else hist
        for c in _WEATHER_NUM:
            prof[c] = float(src[c].mean()) if src[c].notna().any() else np.nan
        prof["weather_missing"] = 0 if len(measured) else 1
    else:
        logger.warning(
            "No history for circuit_id=%s — using median imputation for its "
            "circuit/weather features.", circuit_id,
        )
        for c in _CIRCUIT_COLS + _WEATHER_NUM:
            prof[c] = np.nan
        prof["weather_missing"] = 1

    prof["rain_flag"] = 1 if (rainfall and rainfall > 0) else 0
    return prof


def build_pre_race_features(
    season: int,
    round_num: int,
    circuit_id: str,
    race_date: str,
    grid: dict,
    features_path: str = FEATURES_PATH,
    rainfall: float = 0.0,
    quali_times: Optional[dict] = None,
) -> pd.DataFrame:
    """Build one model-ready feature row per driver in ``grid``.

    Parameters
    ----------
    season, round_num, circuit_id, race_date
        Identify the upcoming race (e.g. 2026, 8, "red_bull_ring", "2026-06-28").
    grid
        ``{driver_code: grid_position}`` from qualifying, e.g. ``{"VER": 1, ...}``.
    rainfall
        Forecast rainfall; any value > 0 sets ``rain_flag`` for every driver.
    quali_times
        Optional ``{driver_code: best_lap_seconds}``; when given, ``gap_to_pole``
        is computed from it, otherwise it falls back to the training median.

    Returns
    -------
    DataFrame with every column in ``FEATURE_COLS`` plus identifier columns,
    one row per driver, ready for ``predict_race``.
    """
    feat = pd.read_parquet(features_path)
    latest = _latest_rows(feat)
    circuit = _circuit_profile(feat, circuit_id, rainfall)
    medians = feat[FEATURE_COLS].median(numeric_only=True)
    quali_times = quali_times or {}
    pole = min(quali_times.values()) if quali_times else None

    rows: list[dict] = []
    debutants: list[str] = []

    for code, gpos in grid.items():
        gpos = float(gpos)
        if code in latest.index:
            src = latest.loc[code]
            row = src[FEATURE_COLS].astype(float).to_dict()
            driver_id = src["driver_id"]
            constructor_id = src["constructor_id"]
        else:
            debutants.append(code)
            row = {c: np.nan for c in FEATURE_COLS}
            row.update({
                "driver_elo_pre": ELO_BASE,
                "constructor_elo_pre": ELO_BASE,
                "perf_rating_pre": ELO_BASE,
                "driver_elo_experience": 0.0,
                "driver_form_races": 0.0,
            })
            driver_id = code.lower()
            constructor_id = "unknown"

        # --- grid (from qualifying) ---
        row["grid_clean"] = gpos
        row["grid_position"] = gpos
        row["grid_penalty"] = 0.0
        champ = row.get("championship_position")
        row["grid_vs_championship"] = (gpos - champ) if pd.notna(champ) else np.nan

        # --- circuit + weather (from venue history/climatology) ---
        for c in _CIRCUIT_COLS + _WEATHER_NUM + ["rain_flag", "weather_missing"]:
            row[c] = circuit[c]

        # --- qualifying pace ---
        if code in quali_times:
            row["quali_best_seconds"] = float(quali_times[code])
            row["gap_to_pole_seconds"] = float(quali_times[code]) - pole
        else:
            row["quali_best_seconds"] = float(medians.get("quali_best_seconds", np.nan))
            row["gap_to_pole_seconds"] = float(medians.get("gap_to_pole_seconds", np.nan))
        # Q-session reached, inferred from the grid slot (top-10 reach Q3, etc.).
        row["reached_q3"] = 1.0 if gpos <= 10 else 0.0
        row["quali_session_reached"] = 3.0 if gpos <= 10 else (2.0 if gpos <= 15 else 1.0)
        row["quali_gap_to_teammate"] = float(medians.get("quali_gap_to_teammate", 0.0))

        row.update({
            "season": int(season), "round": int(round_num),
            "circuit_id": circuit_id, "race_date": race_date,
            "driver_code": code, "driver_id": driver_id,
            "constructor_id": constructor_id,
        })
        rows.append(row)

    out = pd.DataFrame(rows, columns=FEATURE_COLS + ID_COLS)
    if debutants:
        logger.info(
            "%d debutant(s) with no history -> neutral prior (1500 ELO, "
            "median-imputed form): %s", len(debutants), ", ".join(debutants),
        )
    logger.info("Built pre-race features: %d drivers x %d feature cols",
                len(out), len(FEATURE_COLS))
    return out
