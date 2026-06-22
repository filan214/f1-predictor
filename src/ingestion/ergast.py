"""Jolpica F1 API client.

A drop-in replacement for the deprecated Ergast API (retired January 2025).
Jolpica (``api.jolpi.ca/ergast/f1``) serves the identical JSON schema, so the
parsing logic here mirrors what the classic Ergast endpoints returned.

All public ``get_*`` functions return a :class:`pandas.DataFrame`. Network
access is funnelled through :func:`_get`, which applies throttling, retries with
exponential backoff, and special handling for HTTP 429 rate limiting.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
REQUEST_DELAY = 0.6  # seconds between successful requests
MAX_RETRIES = 6

# Reuse a single session for connection pooling across the many small calls.
_session = requests.Session()
_session.headers.update({"User-Agent": "f1-predictor/1.0 (ingestion)"})


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
def _parse_retry_after(value: Optional[str]) -> Optional[int]:
    """Parse a ``Retry-After`` header (delta-seconds form) into an int."""
    if not value:
        return None
    try:
        return max(0, int(float(value.strip())))
    except (TypeError, ValueError):
        return None


def _get(endpoint: str, params: Optional[dict] = None) -> dict:
    """GET ``{BASE_URL}/{endpoint}`` and return the parsed JSON body.

    Retry strategy:
      * HTTP 429   -> wait ``15 * attempt`` seconds, then retry.
      * other 4xx  -> raise immediately (client error, retrying won't help).
      * 5xx / net  -> retry with ``2 ** attempt`` second backoff.

    Sleeps :data:`REQUEST_DELAY` after every successful response to stay polite
    with the shared community-hosted API.
    """
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            # Network-level failure (DNS, connection reset, timeout).
            last_exc = exc
            backoff = 2 ** attempt
            logger.warning(
                "Network error on %s (attempt %d/%d): %s — retrying in %ds",
                url, attempt, MAX_RETRIES, exc, backoff,
            )
            time.sleep(backoff)
            continue

        status = resp.status_code

        if status == 429:
            # Prefer the server's Retry-After hint; otherwise grow the wait.
            # Jolpica enforces both a per-second burst and an hourly cap for
            # anonymous clients, so a hard 429 can need a long cooldown.
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            wait = max(retry_after, 15 * attempt) if retry_after else 15 * attempt
            logger.warning(
                "Rate limited (429) on %s (attempt %d/%d) — waiting %ds",
                url, attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue

        if 400 <= status < 500:
            # Any other client error is not transient.
            resp.raise_for_status()

        if status >= 500:
            last_exc = requests.HTTPError(f"{status} server error for {url}")
            backoff = 2 ** attempt
            logger.warning(
                "Server error %d on %s (attempt %d/%d) — retrying in %ds",
                status, url, attempt, MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            continue

        # 2xx success.
        try:
            data = resp.json()
        except ValueError as exc:
            last_exc = exc
            backoff = 2 ** attempt
            logger.warning(
                "Bad JSON from %s (attempt %d/%d): %s — retrying in %ds",
                url, attempt, MAX_RETRIES, exc, backoff,
            )
            time.sleep(backoff)
            continue

        time.sleep(REQUEST_DELAY)
        return data

    raise RuntimeError(
        f"Failed to fetch {url} after {MAX_RETRIES} attempts"
    ) from last_exc


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _safe_int(value: Any) -> Optional[int]:
    """Convert to int, returning None for missing / non-numeric values."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_to_seconds(value: Optional[str]) -> Optional[float]:
    """Parse a pit-stop duration: ``"23.4"`` or ``"1:23.4"`` -> seconds."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        if ":" in value:
            mins, secs = value.split(":", 1)
            return int(mins) * 60 + float(secs)
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Public endpoints
# --------------------------------------------------------------------------- #
def get_race_schedule(season: int) -> pd.DataFrame:
    """Race calendar for a season."""
    data = _get(f"{season}/races.json", params={"limit": 25})
    races = (
        data.get("MRData", {})
        .get("RaceTable", {})
        .get("Races", [])
    )

    rows = []
    for race in races:
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})
        rows.append(
            {
                "season": _safe_int(race.get("season")),
                "round": _safe_int(race.get("round")),
                "race_name": race.get("raceName"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "locality": location.get("locality"),
                "country": location.get("country"),
                "lat": _safe_float(location.get("lat")),
                "lon": _safe_float(location.get("long")),
                "race_date": race.get("date"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "season", "round", "race_name", "circuit_id", "circuit_name",
            "locality", "country", "lat", "lon", "race_date",
        ],
    )


def get_race_results(season: int, round_num: int) -> pd.DataFrame:
    """Finishing results for a single race."""
    data = _get(f"{season}/{round_num}/results.json", params={"limit": 30})
    races = (
        data.get("MRData", {})
        .get("RaceTable", {})
        .get("Races", [])
    )
    if not races:
        return _empty_results()

    results = races[0].get("Results", [])
    rows = []
    for res in results:
        driver = res.get("Driver", {})
        constructor = res.get("Constructor", {})
        fastest = res.get("FastestLap", {}) or {}
        position_text = res.get("positionText")

        # finish_position is an int only when the driver actually classified
        # to a numeric position; "R"/"D"/"W"/"E"/"F" denote non-finishers.
        finish_position = (
            _safe_int(position_text)
            if position_text is not None and str(position_text).isdigit()
            else None
        )

        time_obj = res.get("Time", {}) or {}
        finish_time_ms = _safe_int(time_obj.get("millis"))

        rows.append(
            {
                "season": season,
                "round": round_num,
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code"),
                "constructor_id": constructor.get("constructorId"),
                "number": _safe_int(res.get("number")),
                "grid": _safe_int(res.get("grid")),
                "finish_position": finish_position,
                "position_text": position_text,
                "points": _safe_float(res.get("points")),
                "laps_completed": _safe_int(res.get("laps")),
                "status": res.get("status"),
                "finish_time_ms": finish_time_ms,
                "fastest_lap_rank": _safe_int(fastest.get("rank")),
                "fastest_lap_time": (fastest.get("Time", {}) or {}).get("time"),
            }
        )

    return pd.DataFrame(rows, columns=_empty_results().columns)


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "season", "round", "driver_id", "driver_code", "constructor_id",
            "number", "grid", "finish_position", "position_text", "points",
            "laps_completed", "status", "finish_time_ms",
            "fastest_lap_rank", "fastest_lap_time",
        ]
    )


def get_qualifying_results(season: int, round_num: int) -> pd.DataFrame:
    """Qualifying results (Q1/Q2/Q3 times) for a single race."""
    data = _get(f"{season}/{round_num}/qualifying.json")
    races = (
        data.get("MRData", {})
        .get("RaceTable", {})
        .get("Races", [])
    )
    columns = [
        "season", "round", "driver_id", "driver_code", "constructor_id",
        "grid_position", "q1_time", "q2_time", "q3_time",
    ]
    if not races:
        return pd.DataFrame(columns=columns)

    quali = races[0].get("QualifyingResults", [])
    rows = []
    for q in quali:
        driver = q.get("Driver", {})
        constructor = q.get("Constructor", {})
        rows.append(
            {
                "season": season,
                "round": round_num,
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code"),
                "constructor_id": constructor.get("constructorId"),
                "grid_position": _safe_int(q.get("position")),
                "q1_time": q.get("Q1") or None,
                "q2_time": q.get("Q2") or None,
                "q3_time": q.get("Q3") or None,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def get_driver_standings(season: int, round_num: int) -> pd.DataFrame:
    """Driver championship standings as of after ``round_num``."""
    data = _get(f"{season}/{round_num}/driverStandings.json")
    standings_lists = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    columns = [
        "season", "after_round", "driver_id", "championship_position",
        "championship_points", "wins",
    ]
    if not standings_lists:
        return pd.DataFrame(columns=columns)

    standings_list = standings_lists[0]
    after_round = _safe_int(standings_list.get("round"))
    rows = []
    for standing in standings_list.get("DriverStandings", []):
        driver = standing.get("Driver", {})
        rows.append(
            {
                "season": season,
                "after_round": after_round,
                "driver_id": driver.get("driverId"),
                "championship_position": _safe_int(standing.get("position")),
                "championship_points": _safe_float(standing.get("points")),
                "wins": _safe_int(standing.get("wins")),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def get_constructor_standings(season: int, round_num: int) -> pd.DataFrame:
    """Constructor championship standings as of after ``round_num``."""
    data = _get(f"{season}/{round_num}/constructorStandings.json")
    standings_lists = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    columns = [
        "season", "after_round", "constructor_id", "constructor_position",
        "constructor_points", "constructor_wins",
    ]
    if not standings_lists:
        return pd.DataFrame(columns=columns)

    standings_list = standings_lists[0]
    after_round = _safe_int(standings_list.get("round"))
    rows = []
    for standing in standings_list.get("ConstructorStandings", []):
        constructor = standing.get("Constructor", {})
        rows.append(
            {
                "season": season,
                "after_round": after_round,
                "constructor_id": constructor.get("constructorId"),
                "constructor_position": _safe_int(standing.get("position")),
                "constructor_points": _safe_float(standing.get("points")),
                "constructor_wins": _safe_int(standing.get("wins")),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def get_pit_stops(season: int, round_num: int) -> pd.DataFrame:
    """Pit stop log for a single race."""
    data = _get(f"{season}/{round_num}/pitstops.json", params={"limit": 100})
    races = (
        data.get("MRData", {})
        .get("RaceTable", {})
        .get("Races", [])
    )
    columns = [
        "season", "round", "driver_id", "stop_number", "lap",
        "time_of_day", "duration_seconds",
    ]
    if not races:
        return pd.DataFrame(columns=columns)

    pit_stops = races[0].get("PitStops", [])
    rows = []
    for stop in pit_stops:
        rows.append(
            {
                "season": season,
                "round": round_num,
                "driver_id": stop.get("driverId"),
                "stop_number": _safe_int(stop.get("stop")),
                "lap": _safe_int(stop.get("lap")),
                "time_of_day": stop.get("time"),
                "duration_seconds": _duration_to_seconds(stop.get("duration")),
            }
        )

    return pd.DataFrame(rows, columns=columns)
