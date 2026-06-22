"""F1 data ingestion entry point.

Collects 2010-2024 F1 data from the Jolpica (Ergast) API and FastF1, writing
raw CSV tables to ``data/raw/``. A checkpoint system persists progress after
every race so an interrupted run resumes exactly where it stopped.

Usage::

    python run_ingestion.py                       # full 2010-2024, both sources
    python run_ingestion.py --source ergast       # Jolpica only
    python run_ingestion.py --source fastf1        # FastF1 only
    python run_ingestion.py --season 2024          # a single season
    python run_ingestion.py --seasons 2025 2026    # extend with newer seasons

The collection is additive: new (season, round) rows are appended to the
existing ``data/raw/`` CSVs via the checkpoint store, and rounds already present
are skipped — so extending the dataset to 2025/2026 never overwrites 2010-2024.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from src.ingestion import ergast
from src.ingestion import fastf1_collector as ff1

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ERGAST_SEASONS = list(range(2010, 2025))
FASTF1_SEASONS = list(range(2018, 2025))
RAW_DIR = Path("data/raw")
CHECKPOINT_DIR = RAW_DIR / "checkpoints"
FASTF1_CACHE_DIR = "data/fastf1_cache"

logger = logging.getLogger("ingestion")
 

"""sasasas"""""

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ingestion.log", encoding="utf-8"),
        ],
    )


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.csv"


def _load_checkpoint(name: str) -> pd.DataFrame:
    """Load an existing checkpoint CSV, or an empty frame if none exists."""
    path = _checkpoint_path(name)
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_checkpoint(name: str, df: pd.DataFrame) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_checkpoint_path(name), index=False)


def _already_collected(df: pd.DataFrame, season: int, round_num: int) -> bool:
    """Return True if a (season, round) row already exists in ``df``."""
    if df.empty or "season" not in df.columns or "round" not in df.columns:
        return False
    return bool(
        ((df["season"] == season) & (df["round"] == round_num)).any()
    )


def _append_checkpoint(
    store: dict[str, pd.DataFrame], name: str, new_rows: pd.DataFrame
) -> None:
    """Append new rows to an in-memory checkpoint frame and persist it."""
    if new_rows is None or new_rows.empty:
        return
    existing = store.get(name)
    if existing is None or existing.empty:
        combined = new_rows.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    store[name] = combined
    _save_checkpoint(name, combined)


# --------------------------------------------------------------------------- #
# Ergast (Jolpica) collection
# --------------------------------------------------------------------------- #
def collect_ergast(seasons: list[int]) -> None:
    logger.info("=== Ergast/Jolpica collection: seasons %s ===", seasons)

    names = [
        "races", "results", "qualifying", "pit_stops",
        "driver_standings", "constructor_standings",
    ]
    store = {name: _load_checkpoint(name) for name in names}

    for season in tqdm(seasons, desc="Ergast seasons", unit="season"):
        try:
            schedule = ergast.get_race_schedule(season)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not fetch schedule for %s: %s", season, exc)
            continue

        if schedule.empty:
            logger.warning("No races found for season %s", season)
            continue

        # Track which seasons' schedule rows we've already stored.
        if not _season_in(store["races"], season):
            _append_checkpoint(store, "races", schedule)

        for _, race in schedule.iterrows():
            round_num = int(race["round"])
            logger.info(
                "Season %s round %02d — %s",
                season, round_num, race.get("race_name", "?"),
            )

            # A failed fetch (e.g. an exhausted 429 retry budget) must not kill
            # the whole run. Log it and move on — checkpoints persist after each
            # endpoint, so re-running fills any gap left behind here.
            try:
                # --- results ---
                if not _already_collected(store["results"], season, round_num):
                    results = ergast.get_race_results(season, round_num)
                    _append_checkpoint(store, "results", results)

                # --- qualifying ---
                if not _already_collected(store["qualifying"], season, round_num):
                    quali = ergast.get_qualifying_results(season, round_num)
                    _append_checkpoint(store, "qualifying", quali)

                # --- pit stops ---
                if not _already_collected(store["pit_stops"], season, round_num):
                    pits = ergast.get_pit_stops(season, round_num)
                    _append_checkpoint(store, "pit_stops", pits)

                # --- standings: as of *before* this race (after round-1) ---
                # Round 1 has no previous round, so skip standings there.
                if round_num > 1:
                    if not _standings_collected(
                        store["driver_standings"], season, round_num
                    ):
                        ds = ergast.get_driver_standings(season, round_num - 1)
                        ds = _tag_standings(ds, season, round_num)
                        _append_checkpoint(store, "driver_standings", ds)

                    if not _standings_collected(
                        store["constructor_standings"], season, round_num
                    ):
                        cs = ergast.get_constructor_standings(season, round_num - 1)
                        cs = _tag_standings(cs, season, round_num)
                        _append_checkpoint(store, "constructor_standings", cs)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Skipping season %s round %s after fetch failure: %s",
                    season, round_num, exc,
                )
                continue

    # Merge checkpoints into final CSVs.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        df = store.get(name, pd.DataFrame())
        out = RAW_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(df))


def _season_in(df: pd.DataFrame, season: int) -> bool:
    if df.empty or "season" not in df.columns:
        return False
    return bool((df["season"] == season).any())


def _tag_standings(df: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    """Tag standings rows with the race they precede."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["before_round"] = round_num
    df["season_round"] = season
    return df


def _standings_collected(df: pd.DataFrame, season: int, round_num: int) -> bool:
    """Standings checkpoints are keyed by (season_round, before_round)."""
    if df.empty or "season_round" not in df.columns or "before_round" not in df.columns:
        return False
    return bool(
        ((df["season_round"] == season) & (df["before_round"] == round_num)).any()
    )


# --------------------------------------------------------------------------- #
# FastF1 collection
# --------------------------------------------------------------------------- #
def collect_fastf1(seasons: list[int]) -> None:
    logger.info("=== FastF1 collection: seasons %s ===", seasons)

    races_path = RAW_DIR / "races.csv"
    if not races_path.exists():
        logger.error(
            "races.csv not found — run the ergast source first so FastF1 "
            "knows which rounds exist."
        )
        return

    races = pd.read_csv(races_path)
    ff1.setup_cache(FASTF1_CACHE_DIR)

    names = ["tire_stints", "race_weather", "qualifying_pace_fastf1"]
    store = {name: _load_checkpoint(name) for name in names}

    collectors: dict[str, Callable[[int, int], pd.DataFrame]] = {
        "tire_stints": ff1.get_tire_stints,
        "race_weather": ff1.get_race_weather,
        "qualifying_pace_fastf1": ff1.get_qualifying_pace,
    }

    for season in tqdm(seasons, desc="FastF1 seasons", unit="season"):
        season_races = races[races["season"] == season]
        if season_races.empty:
            logger.warning("No races in races.csv for season %s", season)
            continue

        for _, race in season_races.iterrows():
            round_num = int(race["round"])
            logger.info("FastF1 — season %s round %02d", season, round_num)

            for name, fn in collectors.items():
                if _already_collected(store[name], season, round_num):
                    continue
                df = fn(season, round_num)
                # Persist a sentinel-free record only when we got rows; an
                # empty result is retried next run (sessions can be flaky).
                _append_checkpoint(store, name, df)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        df = store.get(name, pd.DataFrame())
        out = RAW_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(df))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 data ingestion")
    parser.add_argument(
        "--source",
        choices=["ergast", "fastf1", "all"],
        default="all",
        help="Which data source to collect (default: all).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Collect a single season instead of the full range.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Collect specific seasons, e.g. --seasons 2025 2026. Overrides "
        "--season; appends to existing data via the checkpoint store.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()

    if args.source in ("ergast", "all"):
        if args.seasons:
            seasons = args.seasons
        elif args.season:
            seasons = [args.season]
        else:
            seasons = ERGAST_SEASONS
        collect_ergast(seasons)

    if args.source in ("fastf1", "all"):
        if args.seasons:
            seasons = [s for s in args.seasons if s >= 2018]
            skipped = [s for s in args.seasons if s < 2018]
            if skipped:
                logger.warning(
                    "Seasons %s are outside FastF1 coverage (2018+) - skipping.",
                    skipped,
                )
        elif args.season:
            seasons = [args.season] if args.season in FASTF1_SEASONS else []
            if not seasons:
                logger.warning(
                    "Season %s is outside FastF1 coverage (2018+) - skipping.",
                    args.season,
                )
        else:
            seasons = FASTF1_SEASONS
        if seasons:
            collect_fastf1(seasons)

    logger.info("Ingestion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
