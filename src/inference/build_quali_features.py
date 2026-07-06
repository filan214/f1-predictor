"""Assemble a pre-QUALIFYING feature row per driver for an UPCOMING race.

Mirrors ``src.inference.build_race_features`` but for the qualifying model:
there is no grid/qualifying input here at all, since grid position is exactly
what this model predicts. Every column in
``src.models.quali_dataset.FEATURE_COLS`` (ratings, rolling finish + rolling
qualifying form, standings, circuit, weather) is either carried forward from
a driver's most recent historical row or synthesised from venue
history/climatology — never from the upcoming race's own qualifying result.

This module is read-only with respect to the trained pipeline — it imports
``FEATURE_COLS`` from ``quali_dataset`` so the row layout can never drift from
what the qualifying model expects.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.quali_dataset import FEATURE_COLS, load_quali_features

logger = logging.getLogger(__name__)

FEATURES_PATH = "data/processed/features.parquet"
ELO_BASE = 1500.0  # mirrors src.features.util.ELO_BASE — debutant prior

_CIRCUIT_COLS = [
    "circuit_is_street", "circuit_overtaking_index", "circuit_pole_win_pct",
    "circuit_dnf_rate_hist", "circuit_avg_pitstops_hist",
    "circuit_avg_stint_laps_hist", "circuit_history_races",
]
_WEATHER_NUM = ["air_temp_avg", "track_temp_avg", "humidity_avg", "wind_speed_avg"]

ID_COLS = ["season", "round", "circuit_id", "race_date",
           "driver_code", "driver_id", "constructor_id"]


def resolve_entry_list(season: int, features_path: str = FEATURES_PATH) -> list[str]:
    """Driver codes from the most recent COMPLETED race of ``season``.

    Falls back to the previous season's final race if ``season`` has no rows
    yet (e.g. querying before that season's round 1 has been ingested).
    """
    feat = pd.read_parquet(features_path)
    pool = feat[feat["season"] == season]
    if pool.empty:
        pool = feat[feat["season"] == season - 1]
    if pool.empty:
        raise ValueError(
            f"No historical data for season {season} or {season - 1} in "
            f"{features_path} to infer an entry list. Pass --entries explicitly."
        )
    latest_round = pool["race_order"].max()
    entries = sorted(
        pool.loc[pool["race_order"] == latest_round, "driver_code"].unique().tolist()
    )
    logger.info("Resolved entry list for season %s from race_order %s: %d drivers.",
                season, latest_round, len(entries))
    return entries


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


def build_pre_quali_features(
    season: int,
    round_num: int,
    circuit_id: str,
    race_date: str,
    entries: list[str],
    features_path: str = FEATURES_PATH,
    rainfall: float = 0.0,
) -> pd.DataFrame:
    """Build one qualifying-model-ready feature row per driver in ``entries``.

    Unlike ``build_pre_race_features``, there is no grid/qualifying input —
    grid position is exactly what this model predicts.
    """
    feat = load_quali_features(features_path)
    latest = _latest_rows(feat)
    circuit = _circuit_profile(feat, circuit_id, rainfall)

    rows: list[dict] = []
    debutants: list[str] = []

    for raw_code in entries:
        code = raw_code.strip().upper()
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

        for c in _CIRCUIT_COLS + _WEATHER_NUM + ["rain_flag", "weather_missing"]:
            row[c] = circuit[c]

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
            "%d debutant(s) with no history -> neutral prior: %s",
            len(debutants), ", ".join(debutants),
        )
    logger.info("Built pre-qualifying features: %d drivers x %d feature cols",
                len(out), len(FEATURE_COLS))
    return out
