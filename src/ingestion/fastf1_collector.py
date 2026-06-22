"""FastF1 data collector for tire and weather features (2018+ only).

FastF1 exposes session data that the Ergast/Jolpica API does not: per-stint
tire compounds, lap-by-lap weather, and qualifying pace. Coverage starts in
2018. Every collector returns an empty DataFrame on failure (and logs a
warning) so a single bad session never aborts a long ingestion run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import fastf1

logger = logging.getLogger(__name__)

_CACHE_READY = False


def setup_cache(cache_dir: str = "data/fastf1_cache") -> None:
    """Enable the FastF1 disk cache. Call once before loading any session."""
    global _CACHE_READY
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(path))
    _CACHE_READY = True
    logger.info("FastF1 cache enabled at %s", path.resolve())


def get_tire_stints(season: int, round_num: int) -> pd.DataFrame:
    """Per-driver tire stints for a race (compound, tyre life, lap span)."""
    columns = [
        "season", "round", "driver_code", "stint", "compound", "fresh_tyre",
        "tyre_life_start", "start_lap", "end_lap", "lap_count",
    ]
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps is None or laps.empty:
            return pd.DataFrame(columns=columns)

        grouped = laps.groupby(["Driver", "Stint"], dropna=True)
        agg = grouped.agg(
            compound=("Compound", "first"),
            fresh_tyre=("FreshTyre", "first"),
            tyre_life_start=("TyreLife", "min"),
            start_lap=("LapNumber", "min"),
            end_lap=("LapNumber", "max"),
            lap_count=("LapNumber", "count"),
        ).reset_index()

        agg = agg.rename(columns={"Driver": "driver_code", "Stint": "stint"})
        agg.insert(0, "round", round_num)
        agg.insert(0, "season", season)
        agg["stint"] = agg["stint"].astype("Int64")
        return agg[columns]
    except Exception as exc:  # noqa: BLE001 — never let one race kill the run
        logger.warning(
            "tire_stints failed for %s round %s: %s", season, round_num, exc
        )
        return pd.DataFrame(columns=columns)


def get_race_weather(season: int, round_num: int) -> pd.DataFrame:
    """Single-row weather summary for a race session."""
    columns = [
        "season", "round", "air_temp_avg", "air_temp_max", "track_temp_avg",
        "humidity_avg", "wind_speed_avg", "rainfall",
    ]
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(telemetry=False, weather=True, messages=False)
        weather = session.weather_data
        if weather is None or weather.empty:
            return pd.DataFrame(columns=columns)

        rainfall = int(bool(weather["Rainfall"].any())) if "Rainfall" in weather else 0
        row = {
            "season": season,
            "round": round_num,
            "air_temp_avg": float(weather["AirTemp"].mean()),
            "air_temp_max": float(weather["AirTemp"].max()),
            "track_temp_avg": float(weather["TrackTemp"].mean()),
            "humidity_avg": float(weather["Humidity"].mean()),
            "wind_speed_avg": float(weather["WindSpeed"].mean()),
            "rainfall": rainfall,
        }
        return pd.DataFrame([row], columns=columns)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "race_weather failed for %s round %s: %s", season, round_num, exc
        )
        return pd.DataFrame(columns=columns)


def get_qualifying_pace(season: int, round_num: int) -> pd.DataFrame:
    """Best lap time per driver in each qualifying session (Q1/Q2/Q3)."""
    columns = ["season", "round", "driver_code", "session", "best_lap_seconds"]
    try:
        session = fastf1.get_session(season, round_num, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps is None or laps.empty:
            return pd.DataFrame(columns=columns)

        rows = []
        # FastF1 does not split qualifying laps by Q1/Q2/Q3 directly, so use
        # the official segment boundaries to bucket each lap.
        try:
            segments = session.laps.split_qualifying_sessions()
        except Exception:  # noqa: BLE001 — fall back to a flat best-lap below
            segments = None

        if segments:
            for label, seg in zip(("Q1", "Q2", "Q3"), segments):
                if seg is None or seg.empty:
                    continue
                best = seg.groupby("Driver")["LapTime"].min().dropna()
                for driver_code, lap_time in best.items():
                    rows.append(
                        {
                            "season": season,
                            "round": round_num,
                            "driver_code": driver_code,
                            "session": label,
                            "best_lap_seconds": lap_time.total_seconds(),
                        }
                    )
        else:
            best = laps.groupby("Driver")["LapTime"].min().dropna()
            for driver_code, lap_time in best.items():
                rows.append(
                    {
                        "season": season,
                        "round": round_num,
                        "driver_code": driver_code,
                        "session": "Q",
                        "best_lap_seconds": lap_time.total_seconds(),
                    }
                )

        return pd.DataFrame(rows, columns=columns)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "qualifying_pace failed for %s round %s: %s", season, round_num, exc
        )
        return pd.DataFrame(columns=columns)
