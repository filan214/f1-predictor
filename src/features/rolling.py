"""Trailing-window "form" features for drivers and constructors.

Every aggregate here uses ``groupby(...).shift(1).rolling(window)`` so the
*current* race is never part of its own feature — the cornerstone of leakage
control. Windows are measured in races (chronological), not calendar time.

Driver form (last ``ROLL_WINDOW`` races):
    * ``driver_avg_finish_5``  — mean finishing position (DNF imputed to back).
    * ``driver_avg_points_5``  — mean championship points scored.
    * ``driver_dnf_rate_5``    — fraction of races retired.
    * ``driver_form_races``    — races actually in the window (confidence).

Constructor form (last ``ROLL_WINDOW`` team races, both cars pooled):
    * ``constructor_reliability_5``  — 1 - DNF rate across both cars.
    * ``constructor_points_per_race_5``
    * ``constructor_avg_pit_seconds_5`` — pit-crew execution speed.
"""

from __future__ import annotations

import logging

import pandas as pd

from .util import DNF_FILL_POSITION, ROLL_WINDOW, attach_race_order, is_dnf_series

logger = logging.getLogger(__name__)


def _shift_roll(group: pd.Series, *, window: int, how: str) -> pd.Series:
    """Trailing rolling aggregate that excludes the current row (shift(1))."""
    shifted = group.shift(1).rolling(window, min_periods=1)
    return getattr(shifted, how)()


def _driver_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.sort_values(["driver_id", "race_order"]).copy()
    df["_finish_form"] = df["finish_position"].fillna(DNF_FILL_POSITION)
    df["_is_dnf"] = is_dnf_series(df["finish_position"])

    g = df.groupby("driver_id", group_keys=False)
    df["driver_avg_finish_5"] = g["_finish_form"].apply(
        lambda s: _shift_roll(s, window=window, how="mean")
    )
    df["driver_avg_points_5"] = g["points"].apply(
        lambda s: _shift_roll(s, window=window, how="mean")
    )
    df["driver_dnf_rate_5"] = g["_is_dnf"].apply(
        lambda s: _shift_roll(s, window=window, how="mean")
    )
    df["driver_form_races"] = g["_is_dnf"].apply(
        lambda s: _shift_roll(s, window=window, how="count")
    )
    return df[
        [
            "season", "round", "driver_id",
            "driver_avg_finish_5", "driver_avg_points_5",
            "driver_dnf_rate_5", "driver_form_races",
        ]
    ]


def _constructor_form(
    results: pd.DataFrame, pit_stops: pd.DataFrame, chrono_df: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Per-(constructor, race) summaries rolled over the team's prior races."""
    res = results.copy()
    res["_is_dnf"] = is_dnf_series(res["finish_position"])

    # One row per team per race, pooling both cars.
    team_race = (
        res.groupby(["constructor_id", "season", "round"])
        .agg(
            team_dnf=("_is_dnf", "sum"),
            team_entries=("_is_dnf", "count"),
            team_points=("points", "sum"),
        )
        .reset_index()
    )

    # Pit-stop execution time: attach constructor to each stop, then average
    # per team per race. (pit_stops exists 2011+, so 2010 stays null.)
    if not pit_stops.empty:
        ps = pit_stops.merge(
            res[["season", "round", "driver_id", "constructor_id"]],
            on=["season", "round", "driver_id"],
            how="left",
        )
        team_pit = (
            ps.groupby(["constructor_id", "season", "round"])["duration_seconds"]
            .mean()
            .reset_index()
            .rename(columns={"duration_seconds": "team_pit_seconds"})
        )
        team_race = team_race.merge(
            team_pit, on=["constructor_id", "season", "round"], how="left"
        )
    else:
        team_race["team_pit_seconds"] = pd.NA

    team_race = team_race.merge(
        chrono_df, on=["season", "round"], how="left"
    ).sort_values(["constructor_id", "race_order"])

    g = team_race.groupby("constructor_id", group_keys=False)
    # Reliability over a window of races: roll numerator and denominator
    # separately so the rate is correct even with variable entry counts.
    roll_dnf = g["team_dnf"].apply(lambda s: _shift_roll(s, window=window, how="sum"))
    roll_entries = g["team_entries"].apply(
        lambda s: _shift_roll(s, window=window, how="sum")
    )
    team_race["constructor_reliability_5"] = 1.0 - (roll_dnf / roll_entries)
    team_race["constructor_points_per_race_5"] = g["team_points"].apply(
        lambda s: _shift_roll(s, window=window, how="mean")
    )
    team_race["constructor_avg_pit_seconds_5"] = g["team_pit_seconds"].apply(
        lambda s: _shift_roll(s, window=window, how="mean")
    )

    return team_race[
        [
            "season", "round", "constructor_id",
            "constructor_reliability_5", "constructor_points_per_race_5",
            "constructor_avg_pit_seconds_5",
        ]
    ]


def compute_rolling_features(
    results: pd.DataFrame,
    pit_stops: pd.DataFrame,
    races: pd.DataFrame,
    *,
    window: int = ROLL_WINDOW,
) -> pd.DataFrame:
    """Driver + constructor trailing-form features, one row per driver-race."""
    base = attach_race_order(
        results[
            ["season", "round", "driver_id", "constructor_id",
             "finish_position", "points"]
        ].copy(),
        races,
    )
    chrono_df = base[["season", "round", "race_order"]].drop_duplicates()

    driver = _driver_form(base, window)
    constructor = _constructor_form(results, pit_stops, chrono_df, window)

    out = (
        base[["season", "round", "driver_id", "constructor_id"]]
        .merge(driver, on=["season", "round", "driver_id"], how="left")
        .merge(constructor, on=["season", "round", "constructor_id"], how="left")
        .drop(columns=["constructor_id"])
    )
    logger.info("Rolling features: %d driver-race rows", len(out))
    return out
