"""Fetch a real qualifying grid for an upcoming/completed race.

Powers ``run_predict.py --auto-grid``: instead of typing the 20-driver grid by
hand, pull it straight from the ingested ``data/raw/qualifying.csv``. If the
requested season/round isn't there yet, run an incremental Jolpica ingestion for
that season (append-only, checkpointed) and read it back.

This module only *reads* raw data and shells out to the existing
``run_ingestion.py`` entry point; it does not modify the ingestion, feature, or
model code.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

QUALIFYING_CSV = "data/raw/qualifying.csv"


def _read_grid(season: int, round_num: int, qualifying_path: str) -> dict[str, int] | None:
    """Return ``{driver_code: grid_position}`` for the race, or None if absent."""
    path = Path(qualifying_path)
    if not path.exists():
        return None
    quali = pd.read_csv(path)
    race = quali[(quali["season"] == season) & (quali["round"] == round_num)]
    race = race.dropna(subset=["driver_code", "grid_position"])
    if race.empty:
        return None
    grid = {
        str(row.driver_code).strip().upper(): int(row.grid_position)
        for row in race.itertuples(index=False)
    }
    return grid or None


def _run_ingestion(season: int, repo_root: Path) -> None:
    """Shell out to the project's own ingestion entry point for one season.

    Runs ``python run_ingestion.py --seasons {season} --source ergast`` — a
    self-owned script, not untrusted input. Append-only via the checkpoint store,
    so it never overwrites existing seasons. Output streams to the console so the
    user sees progress.
    """
    script = repo_root / "run_ingestion.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Cannot auto-ingest: {script} not found. Ingest {season} manually "
            f"with `python run_ingestion.py --seasons {season} --source ergast`."
        )
    cmd = [sys.executable, str(script), "--seasons", str(season), "--source", "ergast"]
    logger.info("Qualifying for %s not found locally — running: %s", season, " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        raise RuntimeError(
            f"Ingestion for season {season} failed (exit {result.returncode}). "
            "Run it manually and re-try, or pass --grid explicitly."
        )


def fetch_qualifying_grid(
    season: int,
    round_num: int,
    qualifying_path: str = QUALIFYING_CSV,
    auto_ingest: bool = True,
) -> dict[str, int]:
    """Return the real starting grid ``{driver_code: grid_position}`` for a race.

    Looks in ``qualifying_path`` first; if the season/round is missing and
    ``auto_ingest`` is set, runs an incremental ingestion for ``season`` and
    reads again.

    Raises
    ------
    ValueError
        If no qualifying data can be found even after ingestion.
    """
    grid = _read_grid(season, round_num, qualifying_path)
    if grid is not None:
        logger.info("Loaded real qualifying grid for %s round %s: %d drivers.",
                    season, round_num, len(grid))
        return grid

    if auto_ingest:
        repo_root = Path(qualifying_path).resolve().parents[2]
        _run_ingestion(season, repo_root)
        grid = _read_grid(season, round_num, qualifying_path)
        if grid is not None:
            logger.info("Loaded qualifying grid for %s round %s after ingestion: "
                        "%d drivers.", season, round_num, len(grid))
            return grid

    raise ValueError(
        f"No qualifying data for season {season} round {round_num} in "
        f"{qualifying_path} (even after ingestion). The round may not have "
        "qualified yet, or the round number is wrong. Pass --grid manually."
    )
