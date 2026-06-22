"""Live race-prediction entry point.

Given an upcoming race's circuit, date, forecast rainfall and starting grid,
this assembles pre-race features (carrying each driver's latest form forward),
runs the trained LightGBM model, prints a ranked forecast, and saves a
structured JSON for the dashboard.

Usage::

    python run_predict.py --season 2026 --circuit red_bull_ring \
      --date 2026-06-28 --rainfall 0 \
      --grid "VER:1,NOR:2,LEC:3,HAM:4,PIA:5,RUS:6,SAI:7,ALO:8,ANT:9,TSU:10,\
LAW:11,STR:12,HUL:13,BEA:14,OCO:15,GAS:16,DOO:17,HAD:18,BOR:19,MAG:20"

The round number and a friendly output filename are looked up from
``data/raw/races.csv`` when the season is present there; otherwise pass
``--round`` and/or ``--out`` explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.inference.build_race_features import build_pre_race_features
from src.inference.predict_race import predict_race

RACES_CSV = Path("data/raw/races.csv")
REPORTS_DIR = Path("reports")
logger = logging.getLogger("predict")


def parse_grid(spec: str) -> dict:
    """Parse ``"VER:1,NOR:2,..."`` into ``{"VER": 1, "NOR": 2, ...}``."""
    grid: dict[str, int] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Bad grid entry {token!r}; expected CODE:POS.")
        code, pos = token.split(":", 1)
        grid[code.strip().upper()] = int(pos)
    if not grid:
        raise ValueError("Empty grid.")
    return grid


def lookup_race(season: int, circuit_id: str) -> tuple[int | None, str]:
    """Return (round_num, tag) for the race from races.csv, if available.

    ``tag`` is the country slug (e.g. "austria") used for the output filename,
    falling back to the circuit id.
    """
    if RACES_CSV.exists():
        races = pd.read_csv(RACES_CSV)
        match = races[(races["season"] == season) & (races["circuit_id"] == circuit_id)]
        if len(match):
            row = match.iloc[0]
            country = str(row.get("country", "") or "").strip().lower().replace(" ", "_")
            return int(row["round"]), (country or circuit_id)
    return None, circuit_id


def format_table(result: pd.DataFrame) -> str:
    disp = result.copy()
    for c in ("win_probability", "podium_probability", "points_probability"):
        disp[c] = (disp[c] * 100).round(1).astype(str) + "%"
    disp = disp.rename(columns={
        "predicted_rank": "P", "driver_code": "driver", "constructor_id": "team",
        "grid_position": "grid", "predicted_position_raw": "pred_pos",
        "win_probability": "win%", "podium_probability": "podium%",
        "points_probability": "points%",
    })
    return disp.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict an upcoming F1 race.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--circuit", required=True, help="circuit_id, e.g. red_bull_ring")
    parser.add_argument("--date", required=True, help="race date YYYY-MM-DD")
    parser.add_argument("--rainfall", type=float, default=0.0,
                        help="forecast rainfall (>0 sets the wet-race flag).")
    parser.add_argument("--grid", required=True,
                        help='starting grid as "CODE:POS,CODE:POS,..."')
    parser.add_argument("--round", type=int, default=None,
                        help="round number (else looked up from races.csv).")
    parser.add_argument("--features-path", default="data/processed/features.parquet")
    parser.add_argument("--out", default=None, help="output JSON path.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )

    grid = parse_grid(args.grid)
    looked_round, tag = lookup_race(args.season, args.circuit)
    round_num = args.round if args.round is not None else (looked_round or 0)
    if looked_round is None:
        logger.warning("Race not found in races.csv for season %s circuit %s; "
                       "using round=%d (ingest the season for the real round).",
                       args.season, args.circuit, round_num)

    features_df = build_pre_race_features(
        season=args.season, round_num=round_num, circuit_id=args.circuit,
        race_date=args.date, grid=grid, features_path=args.features_path,
        rainfall=args.rainfall,
    )
    result = predict_race(features_df)

    print(f"\n=== Predicted {args.season} round {round_num} — {args.circuit} "
          f"({args.date}, rainfall={args.rainfall}) ===")
    print(format_table(result))

    out_path = Path(args.out) if args.out else REPORTS_DIR / f"prediction_{args.season}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "season": args.season,
            "round": round_num,
            "circuit_id": args.circuit,
            "race_date": args.date,
            "rainfall": args.rainfall,
            "model": "lightgbm",
            "n_drivers": len(result),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "predictions": result.to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
