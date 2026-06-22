"""Per-circuit historical characteristics.

Circuits recur roughly once a season, so these features use an **expanding**
window over every *prior* race at the same circuit (via ``shift(1).expanding()``)
rather than a fixed trailing count. The first time a circuit appears its history
is undefined (NaN) and the ``circuit_history_races`` count is 0.

Features:
    * ``circuit_is_street``           — static layout flag (street vs. permanent).
    * ``circuit_overtaking_index``    — mean |grid - finish| at the circuit;
                                        high = lots of position change.
    * ``circuit_pole_win_pct``        — share of past races won from pole.
    * ``circuit_dnf_rate_hist``       — mean DNF rate (incident/SC proxy; we have
                                        no direct safety-car data).
    * ``circuit_avg_pitstops_hist``   — mean stops per driver (strategy demand,
                                        from pit_stops, 2011+).
    * ``circuit_avg_stint_laps_hist`` — mean stint length (tyre-deg proxy, from
                                        tire_stints, 2018+; shorter = higher deg).

Strategy note: actual stints/stops *of the current race* are post-race facts
and would leak. Only *historical* circuit averages are used, which are knowable
before lights-out.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .util import STREET_CIRCUITS, attach_race_order, is_dnf_series

logger = logging.getLogger(__name__)


def _expand_mean(group: pd.Series) -> pd.Series:
    """Expanding mean over strictly prior rows (current race excluded)."""
    return group.shift(1).expanding(min_periods=1).mean()


def compute_circuit_features(
    results: pd.DataFrame,
    races: pd.DataFrame,
    pit_stops: pd.DataFrame,
    tire_stints: pd.DataFrame,
) -> pd.DataFrame:
    """Circuit history features, one row per (season, round)."""
    circuit_map = races[["season", "round", "circuit_id"]].copy()

    res = results.merge(circuit_map, on=["season", "round"], how="left").copy()
    res["_is_dnf"] = is_dnf_series(res["finish_position"])
    # Pit-lane starts (grid 0) count as back-of-grid for position-change math.
    res["_grid_clean"] = res["grid"].replace(0, np.nan)
    field_max = res.groupby(["season", "round"])["_grid_clean"].transform("max")
    res["_grid_clean"] = res["_grid_clean"].fillna(field_max + 1)
    res["_pos_change"] = (res["_grid_clean"] - res["finish_position"]).abs()

    # --- per-race circuit summary ---
    per_race = (
        res.groupby(["circuit_id", "season", "round"])
        .agg(
            overtaking=("_pos_change", "mean"),
            dnf_rate=("_is_dnf", "mean"),
        )
        .reset_index()
    )
    # Pole-winner flag: did the race winner start P1?
    winners = res[res["finish_position"] == 1]
    pole_win = (
        winners.assign(pole_won=(winners["grid"] == 1).astype("int8"))
        .groupby(["circuit_id", "season", "round"])["pole_won"]
        .max()
        .reset_index()
    )
    per_race = per_race.merge(
        pole_win, on=["circuit_id", "season", "round"], how="left"
    )

    # --- strategy demand: stops per driver (pit_stops, 2011+) ---
    if not pit_stops.empty:
        stops_per_driver = (
            pit_stops.groupby(["season", "round", "driver_id"])["stop_number"]
            .max()
            .reset_index()
        )
        race_stops = (
            stops_per_driver.merge(circuit_map, on=["season", "round"], how="left")
            .groupby(["circuit_id", "season", "round"])["stop_number"]
            .mean()
            .reset_index()
            .rename(columns={"stop_number": "avg_pitstops"})
        )
        per_race = per_race.merge(
            race_stops, on=["circuit_id", "season", "round"], how="left"
        )
    else:
        per_race["avg_pitstops"] = pd.NA

    # --- tyre-deg proxy: mean stint length (tire_stints, 2018+) ---
    if not tire_stints.empty:
        race_stint = (
            tire_stints.merge(circuit_map, on=["season", "round"], how="left")
            .groupby(["circuit_id", "season", "round"])["lap_count"]
            .mean()
            .reset_index()
            .rename(columns={"lap_count": "avg_stint_laps"})
        )
        per_race = per_race.merge(
            race_stint, on=["circuit_id", "season", "round"], how="left"
        )
    else:
        per_race["avg_stint_laps"] = pd.NA

    # --- expand over prior races at each circuit ---
    per_race = attach_race_order(per_race, races).sort_values(
        ["circuit_id", "race_order"]
    )
    g = per_race.groupby("circuit_id", group_keys=False)
    per_race["circuit_overtaking_index"] = g["overtaking"].apply(_expand_mean)
    per_race["circuit_pole_win_pct"] = g["pole_won"].apply(_expand_mean)
    per_race["circuit_dnf_rate_hist"] = g["dnf_rate"].apply(_expand_mean)
    per_race["circuit_avg_pitstops_hist"] = g["avg_pitstops"].apply(_expand_mean)
    per_race["circuit_avg_stint_laps_hist"] = g["avg_stint_laps"].apply(_expand_mean)
    per_race["circuit_history_races"] = g.cumcount()  # prior races at circuit
    per_race["circuit_is_street"] = (
        per_race["circuit_id"].isin(STREET_CIRCUITS).astype("int8")
    )

    out = per_race[
        [
            "season", "round", "circuit_id", "circuit_is_street",
            "circuit_overtaking_index", "circuit_pole_win_pct",
            "circuit_dnf_rate_hist", "circuit_avg_pitstops_hist",
            "circuit_avg_stint_laps_hist", "circuit_history_races",
        ]
    ]
    logger.info(
        "Circuit features: %d races | %d street-circuit races",
        len(out), int(out["circuit_is_street"].sum()),
    )
    return out
