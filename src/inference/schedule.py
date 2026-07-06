"""Find the next upcoming race from the ingested schedule.

Powers ``run_predict.py --next-race``: read ``data/raw/races.csv``, pick the
earliest race whose date is still in the future, and hand back everything the
predictor needs (season, round, circuit, date). Read-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RACES_CSV = "data/raw/races.csv"


@dataclass
class NextRace:
    season: int
    round: int
    circuit_id: str
    race_date: str          # ISO YYYY-MM-DD
    country: str
    tag: str                # filename-safe country slug
    days_until: int


def find_next_race(races_path: str = RACES_CSV, today: date | None = None) -> NextRace:
    """Return the next race whose date is today or later.

    We use ``race_date >= today`` (not strictly ``>``) so a race happening *today*
    — whose qualifying is already done — is predicted rather than skipped in
    favour of a later race that hasn't qualified yet.

    Parameters
    ----------
    races_path
        Path to the ingested ``races.csv``.
    today
        Reference date (defaults to the real system date). Useful for testing.

    Raises
    ------
    FileNotFoundError
        If the schedule CSV is missing.
    ValueError
        If no upcoming race exists in the schedule (ingest newer seasons).
    """
    path = Path(races_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{races_path} not found — run `python run_ingestion.py` first."
        )
    today = today or date.today()

    races = pd.read_csv(path)
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    future = races[races["race_date"].dt.date >= today].sort_values("race_date")
    if future.empty:
        last = pd.to_datetime(races["race_date"]).max()
        raise ValueError(
            f"No upcoming race after {today} in {races_path} "
            f"(last scheduled race is {last.date() if pd.notna(last) else 'unknown'}). "
            "Ingest a newer season with `python run_ingestion.py --seasons <year>`."
        )

    row = future.iloc[0]
    race_dt = row["race_date"].date()
    country = str(row.get("country", "") or "").strip()
    tag = country.lower().replace(" ", "_") or str(row["circuit_id"])
    nxt = NextRace(
        season=int(row["season"]),
        round=int(row["round"]),
        circuit_id=str(row["circuit_id"]),
        race_date=race_dt.isoformat(),
        country=country,
        tag=tag,
        days_until=(race_dt - today).days,
    )
    logger.info("Next race: %s round %s — %s (%s), in %d day(s).",
                nxt.season, nxt.round, nxt.circuit_id, nxt.race_date, nxt.days_until)
    return nxt
