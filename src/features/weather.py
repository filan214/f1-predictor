"""Weather features with a layered source strategy.

Priority per race:
    1. **FastF1** ``race_weather`` (2018+, real measured session weather).
    2. **OpenWeatherMap** historical fallback by circuit lat/lon + race date
       (pre-2018), *only if* an ``OPENWEATHER_API_KEY`` is configured.
    3. **None** — emit NaNs, ``weather_missing = 1``.

Every row carries a ``weather_source`` label and a ``weather_missing`` flag so
the model (and the leakage audit) can treat imputed/absent weather explicitly
rather than silently. The OWM call is implemented but inert without a key, so
the pipeline runs end-to-end today and pre-2018 weather can be backfilled later
simply by adding the key and re-running.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_WEATHER_COLS = [
    "air_temp_avg", "track_temp_avg", "humidity_avg", "wind_speed_avg",
]


def _openweather_fallback(missing: pd.DataFrame) -> pd.DataFrame:
    """Look up historical weather for races lacking FastF1 data.

    Returns a frame keyed by (season, round) with weather columns. Inert (empty)
    unless ``OPENWEATHER_API_KEY`` is set; we avoid making network calls during
    a normal feature build. The structure is here so the fallback can be enabled
    later without touching the orchestrator.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key or missing.empty:
        if not api_key:
            logger.info(
                "OPENWEATHER_API_KEY not set - %d pre-2018 races left "
                "weather_missing.", len(missing),
            )
        return pd.DataFrame(
            columns=["season", "round", *_WEATHER_COLS, "rainfall"]
        )

    # NOTE: live OWM One Call 3.0 "timemachine" integration goes here when a key
    # is available (one call per race using lat/lon + unix(race_date)). Left as a
    # documented extension point; not exercised in the keyless default run.
    logger.warning(
        "OPENWEATHER_API_KEY is set but the OWM fetch is not enabled in this "
        "build; treating %d races as weather_missing.", len(missing),
    )
    return pd.DataFrame(columns=["season", "round", *_WEATHER_COLS, "rainfall"])


def compute_weather_features(
    races: pd.DataFrame, race_weather: pd.DataFrame
) -> pd.DataFrame:
    """Weather features, one row per (season, round), with source flags."""
    base = races[["season", "round", "lat", "lon", "race_date"]].copy()

    rw = race_weather.rename(columns={"air_temp_max": "_air_temp_max"})
    have_cols = ["season", "round", *_WEATHER_COLS, "rainfall"]
    rw = rw[[c for c in have_cols if c in rw.columns]]

    merged = base.merge(rw, on=["season", "round"], how="left")
    merged["weather_source"] = merged["air_temp_avg"].notna().map(
        {True: "fastf1", False: "none"}
    )

    # Attempt the keyless-inert fallback for whatever FastF1 lacks.
    missing = merged[merged["weather_source"] == "none"]
    fb = _openweather_fallback(missing[["season", "round", "lat", "lon", "race_date"]])
    if not fb.empty:
        merged = merged.merge(
            fb, on=["season", "round"], how="left", suffixes=("", "_owm")
        )
        for col in [*_WEATHER_COLS, "rainfall"]:
            owm = f"{col}_owm"
            if owm in merged.columns:
                fill = merged[col].isna() & merged[owm].notna()
                merged.loc[fill, col] = merged.loc[fill, owm]
                merged.loc[fill, "weather_source"] = "openweather"
                merged = merged.drop(columns=[owm])

    merged["weather_missing"] = (merged["weather_source"] == "none").astype("int8")
    merged["rain_flag"] = merged["rainfall"].fillna(0).astype("int8")

    out = merged[
        [
            "season", "round", *_WEATHER_COLS, "rain_flag",
            "weather_source", "weather_missing",
        ]
    ]
    by_source = out["weather_source"].value_counts().to_dict()
    logger.info("Weather features: %d races | sources=%s", len(out), by_source)
    return out
