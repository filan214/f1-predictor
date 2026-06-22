"""Feature-matrix orchestrator.

Joins every feature module onto a results-grain base (one row per driver per
race), derives qualifying / grid / standings features inline, attaches the
prediction targets, runs a leakage audit, and writes
``data/processed/features.parquet``.

Run::

    python -m src.features.build_features
    python -m src.features.build_features --out data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from .circuit import compute_circuit_features
from .driver_ratings import compute_driver_ratings
from .rolling import compute_rolling_features
from .util import PROCESSED_DIR, attach_race_order, load_raw, parse_lap_time
from .weather import compute_weather_features

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Inline feature blocks (qualifying / grid / standings)
# --------------------------------------------------------------------------- #
def _qualifying_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Gap-to-pole, session reached, and teammate pace from qualifying times."""
    q = qualifying.copy()
    for col in ("q1_time", "q2_time", "q3_time"):
        q[f"_{col}_s"] = q[col].map(parse_lap_time)

    q["quali_best_seconds"] = q[["_q1_time_s", "_q2_time_s", "_q3_time_s"]].min(
        axis=1
    )
    # Which session a driver reached, from which times they actually set.
    q["quali_session_reached"] = (
        q["_q1_time_s"].notna().astype(int)
        + q["_q2_time_s"].notna().astype(int)
        + q["_q3_time_s"].notna().astype(int)
    ).clip(upper=3)
    q["reached_q3"] = q["_q3_time_s"].notna().astype("int8")

    # Gap to the session's pole time (fastest car in the field).
    pole = q.groupby(["season", "round"])["quali_best_seconds"].transform("min")
    q["gap_to_pole_seconds"] = q["quali_best_seconds"] - pole

    # Gap to the *teammate's* best lap (pure driver-vs-car-control signal).
    g = q.groupby(["season", "round", "constructor_id"])["quali_best_seconds"]
    team_min = g.transform("min")
    team_min2 = g.transform(
        lambda s: s.nsmallest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan
    )
    teammate_best = np.where(
        q["quali_best_seconds"] <= team_min, team_min2, team_min
    )
    q["quali_gap_to_teammate"] = q["quali_best_seconds"] - teammate_best

    return q[
        [
            "season", "round", "driver_id", "grid_position",
            "quali_best_seconds", "gap_to_pole_seconds",
            "quali_session_reached", "reached_q3", "quali_gap_to_teammate",
        ]
    ]


def _standings_features(
    driver_standings: pd.DataFrame, constructor_standings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-race championship standings, keyed back to the race they precede.

    Ingestion tagged each standings snapshot with ``season_round`` (= season)
    and ``before_round`` (= the round it precedes). Round 1 of every season has
    no within-season prior round, so those races get NaN standings — the
    carried-forward ELO is the cross-season signal there.
    """
    # season_round == season and before_round == round-of-the-race-it-precedes.
    # Drop the raw season/after_round first to avoid a duplicate 'season' label.
    ds = (
        driver_standings.drop(columns=["season", "after_round"])
        .rename(columns={"season_round": "season", "before_round": "round"})[
            ["season", "round", "driver_id", "championship_position",
             "championship_points", "wins"]
        ]
        .rename(columns={"wins": "championship_wins"})
    )

    cs = (
        constructor_standings.drop(columns=["season", "after_round"])
        .rename(columns={"season_round": "season", "before_round": "round"})[
            ["season", "round", "constructor_id", "constructor_position",
             "constructor_points", "constructor_wins"]
        ]
    )
    return ds, cs


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_feature_matrix() -> pd.DataFrame:
    logger.info("Loading raw tables...")
    races = load_raw("races")
    results = load_raw("results")
    qualifying = load_raw("qualifying")
    pit_stops = load_raw("pit_stops")
    driver_standings = load_raw("driver_standings")
    constructor_standings = load_raw("constructor_standings")
    race_weather = load_raw("race_weather")
    tire_stints = load_raw("tire_stints")

    # --- base grain: one row per driver per race ---
    base = results[
        [
            "season", "round", "driver_id", "driver_code", "constructor_id",
            "grid", "finish_position", "points", "status",
        ]
    ].copy()
    base = attach_race_order(base, races)  # adds race_order + race_date
    base = base.merge(
        races[["season", "round", "circuit_id", "country"]],
        on=["season", "round"], how="left",
    )

    # Pit-lane starts (grid 0) -> back of grid for any grid-based maths.
    field_max = base.groupby(["season", "round"])["grid"].transform(
        lambda s: s.replace(0, np.nan).max()
    )
    base["grid_clean"] = base["grid"].replace(0, np.nan)
    base["grid_clean"] = base["grid_clean"].fillna(field_max + 1)

    # --- qualifying ---
    quali = _qualifying_features(qualifying)
    base = base.merge(quali, on=["season", "round", "driver_id"], how="left")
    # Grid penalty = positions lost between qualifying classification and start.
    base["grid_penalty"] = (base["grid_clean"] - base["grid_position"]).clip(lower=0)

    # --- standings (pre-race) ---
    ds, cs = _standings_features(driver_standings, constructor_standings)
    base = base.merge(ds, on=["season", "round", "driver_id"], how="left")
    base = base.merge(cs, on=["season", "round", "constructor_id"], how="left")
    base["grid_vs_championship"] = base["grid_clean"] - base["championship_position"]

    # --- driver/constructor ELO ---
    ratings = compute_driver_ratings(results, races)
    base = base.merge(ratings, on=["season", "round", "driver_id"], how="left")

    # --- rolling form ---
    rolling = compute_rolling_features(results, pit_stops, races)
    base = base.merge(rolling, on=["season", "round", "driver_id"], how="left")

    # --- circuit history ---
    circuit = compute_circuit_features(results, races, pit_stops, tire_stints)
    base = base.merge(
        circuit.drop(columns=["circuit_id"]), on=["season", "round"], how="left"
    )

    # --- weather ---
    weather = compute_weather_features(races, race_weather)
    base = base.merge(weather, on=["season", "round"], how="left")

    # --- targets (derived from the current race; for modelling, not features) ---
    base["target_finish_position"] = base["finish_position"]
    base["target_is_dnf"] = base["finish_position"].isna().astype("int8")
    base["target_is_winner"] = (base["finish_position"] == 1).astype("int8")
    base["target_is_podium"] = (base["finish_position"] <= 3).astype("int8")
    base["target_is_points"] = (base["finish_position"] <= 10).astype("int8")

    base = base.sort_values(["race_order", "grid_clean"]).reset_index(drop=True)
    logger.info("Feature matrix: %d rows x %d cols", *base.shape)
    return base


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
# Columns that legitimately describe the current race outcome (targets) or are
# identifiers — excluded from the "suspicious correlation" check.
_NON_FEATURE = {
    "season", "round", "race_order", "driver_id", "driver_code",
    "constructor_id", "circuit_id", "country", "race_date", "status",
    "finish_position", "points",
    "target_finish_position", "target_is_dnf", "target_is_winner",
    "target_is_podium", "target_is_points", "weather_source",
}


def leakage_audit(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Correlate every feature with finishing position; flag suspicious ones.

    Grid and qualifying features are *expected* to be strongly (negatively)
    correlated — that is the baseline the model must beat, not leakage. The
    audit flags anything with ``|corr| >= threshold``, which would indicate a
    feature accidentally carrying the post-race result.
    """
    classified = df[df["target_finish_position"].notna()]
    feature_cols = [
        c for c in df.columns
        if c not in _NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])
    ]
    corrs = {
        c: classified[c].corr(classified["target_finish_position"])
        for c in feature_cols
    }
    audit = (
        pd.DataFrame({"feature": list(corrs), "corr_with_finish": list(corrs.values())})
        .assign(abs_corr=lambda d: d["corr_with_finish"].abs())
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    flagged = audit[audit["abs_corr"] >= threshold]
    if len(flagged):
        logger.warning(
            "LEAKAGE AUDIT: %d feature(s) with |corr| >= %.2f:\n%s",
            len(flagged), threshold, flagged.to_string(index=False),
        )
    else:
        logger.info(
            "Leakage audit clean: no feature exceeds |corr| %.2f with finish. "
            "Top correlates:\n%s",
            threshold, audit.head(8).to_string(index=False),
        )
    return audit


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the F1 feature matrix")
    parser.add_argument(
        "--out", default=str(PROCESSED_DIR / "features.parquet"),
        help="Output parquet path.",
    )
    args = parser.parse_args(argv)
    setup_logging()

    matrix = build_feature_matrix()
    leakage_audit(matrix)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(args.out, index=False)
    logger.info("Wrote %s (%d rows, %d cols)", args.out, *matrix.shape)

    # Coverage / null summary for the feature columns.
    null_share = matrix.isna().mean().sort_values(ascending=False)
    logger.info(
        "Feature null shares (top 12):\n%s",
        (null_share[null_share > 0].head(12) * 100).round(1).to_string(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
