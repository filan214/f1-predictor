"""Driver (and constructor) skill ratings via a carried-forward dual ELO.

The single most distinctive feature in the pipeline. A plain ELO rating
conflates *driver skill* with *car performance* — a midfield driver in a
dominant car looks unbeatable. We separate the two with a **dual ELO**:

* Each race result updates both a per-driver rating and a per-constructor
  rating from the same surprise signal (actual vs. expected finishing order).
* A sharing coefficient ``alpha`` routes most of a strong/weak result to the
  *constructor* rating, so the *driver* rating drifts only when a driver beats
  what their machinery alone would predict. Over a career this decontaminates
  skill from car.

Ratings are **carried forward across seasons** (a driver changing teams keeps
their skill rating — the key methodological choice, mirroring a Bayesian
carry-forward baseline). The feature value stored for each race is the rating
*as of before* that race, so there is no leakage.

Design notes
------------
* Only *classified finishers* take part in a race's rating update. A mechanical
  DNF is not a skill signal; reliability is modelled separately in the rolling
  constructor features.
* Expectation uses a combined "performance rating"
  ``driver + constructor_weight * (constructor - base)`` so a driver in a fast
  car is correctly expected to finish ahead — only deviations move the rating.
* The update is a multiplayer ELO expressed as the mean of pairwise duels
  against every other classified finisher, scaled by ``K``.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from .util import ELO_BASE, attach_race_order

logger = logging.getLogger(__name__)

# Tuning constants (sensible defaults; not fitted — these are priors).
DEFAULT_K = 24.0              # ELO step size
DEFAULT_ALPHA = 0.35         # share of each result routed to the constructor
DEFAULT_CONSTRUCTOR_WEIGHT = 0.75  # car contribution to expected performance


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard ELO expectation that A finishes ahead of B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_driver_ratings(
    results: pd.DataFrame,
    races: pd.DataFrame,
    *,
    k: float = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    constructor_weight: float = DEFAULT_CONSTRUCTOR_WEIGHT,
    base: float = ELO_BASE,
) -> pd.DataFrame:
    """Compute pre-race driver/constructor ELO features.

    Parameters
    ----------
    results, races
        Raw ``results.csv`` and ``races.csv`` frames.

    Returns
    -------
    DataFrame with one row per (season, round, driver_id) and columns:
        ``driver_elo_pre``           rating before the race (carried forward),
        ``constructor_elo_pre``      car rating before the race,
        ``perf_rating_pre``          combined driver+car expected performance,
        ``driver_elo_experience``    number of prior races for the driver
                                     (rating confidence; new drivers are noisy).
    """
    df = attach_race_order(
        results[
            ["season", "round", "driver_id", "constructor_id", "finish_position"]
        ].copy(),
        races,
    )
    df = df.sort_values(["race_order", "season", "round"]).reset_index(drop=True)

    driver_elo: dict[str, float] = defaultdict(lambda: base)
    constructor_elo: dict[str, float] = defaultdict(lambda: base)
    driver_races: dict[str, int] = defaultdict(int)

    feature_rows: list[dict] = []

    for _, race_df in df.groupby("race_order", sort=True):
        season = int(race_df["season"].iloc[0])
        round_num = int(race_df["round"].iloc[0])

        # --- 1. Snapshot pre-race ratings as the feature values. ---
        for _, r in race_df.iterrows():
            d = r["driver_id"]
            c = r["constructor_id"]
            d_elo = driver_elo[d]
            c_elo = constructor_elo[c]
            feature_rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "driver_id": d,
                    "driver_elo_pre": d_elo,
                    "constructor_elo_pre": c_elo,
                    "perf_rating_pre": d_elo
                    + constructor_weight * (c_elo - base),
                    "driver_elo_experience": driver_races[d],
                }
            )

        # --- 2. Update ratings from this race's classified finishers. ---
        classified = race_df[race_df["finish_position"].notna()]
        entrants = [
            (
                r["driver_id"],
                r["constructor_id"],
                float(r["finish_position"]),
                driver_elo[r["driver_id"]]
                + constructor_weight * (constructor_elo[r["constructor_id"]] - base),
            )
            for _, r in classified.iterrows()
        ]
        n = len(entrants)

        # Experience counts every classified start (used for confidence only).
        for d, _c, _pos, _perf in entrants:
            driver_races[d] += 1

        if n < 2:
            continue  # nothing to compare against

        # Multiplayer ELO = mean pairwise duel outcome vs. every rival.
        raw_delta: dict[int, float] = {}
        for i in range(n):
            d_i, _c_i, pos_i, perf_i = entrants[i]
            expected = 0.0
            actual = 0.0
            for j in range(n):
                if i == j:
                    continue
                _d_j, _c_j, pos_j, perf_j = entrants[j]
                expected += _expected_score(perf_i, perf_j)
                actual += 1.0 if pos_i < pos_j else 0.0  # lower pos = better
            raw_delta[i] = k * (actual - expected) / (n - 1)

        # Driver keeps (1 - alpha) of the surprise; the rest feeds the car.
        constructor_deltas: dict[str, list[float]] = defaultdict(list)
        for i in range(n):
            d_i, c_i, _pos_i, _perf_i = entrants[i]
            driver_elo[d_i] += (1.0 - alpha) * raw_delta[i]
            constructor_deltas[c_i].append(raw_delta[i])

        # A constructor's update is alpha * the mean of its cars' surprises,
        # keeping its rating on the same scale as a single-car update.
        for c, deltas in constructor_deltas.items():
            constructor_elo[c] += alpha * (sum(deltas) / len(deltas))

    out = pd.DataFrame(feature_rows)
    logger.info(
        "Driver ratings: %d driver-race rows | final driver ELO range %.0f-%.0f",
        len(out),
        min(driver_elo.values()),
        max(driver_elo.values()),
    )
    return out
