"""Live race-prediction entry point.

Given an upcoming race's circuit, date, forecast rainfall and starting grid,
this assembles pre-race features (carrying each driver's latest form forward),
runs the trained LightGBM model, prints a ranked forecast, and saves a
structured JSON for the dashboard.

Usage::

    # Manual grid:
    python run_predict.py --season 2026 --circuit red_bull_ring \
      --date 2026-06-28 --rainfall 0 \
      --grid "VER:1,NOR:2,LEC:3,HAM:4,PIA:5,RUS:6,SAI:7,ALO:8,ANT:9,TSU:10,\
LAW:11,STR:12,HUL:13,BEA:14,OCO:15,GAS:16,DOO:17,HAD:18,BOR:19,MAG:20"

    # Real qualifying grid, fetched automatically (ingests the season if needed):
    python run_predict.py --season 2026 --round 9 --circuit silverstone \
      --date 2026-07-06 --auto-grid

    # Fully automatic — detect the next race, fetch/predict its grid, predict:
    python run_predict.py --next-race

    # Predict the grid too (before real qualifying exists):
    python run_predict.py --season 2026 --round 10 --circuit spa \
      --date 2026-07-19 --predict-grid

The round number and a friendly output filename are looked up from
``data/raw/races.csv`` when the season is present there; otherwise pass
``--round`` and/or ``--out`` explicitly. ``--auto-grid``/``--predict-grid``
need a round number to look up qualifying/build features, so pass ``--round``
when the race isn't in ``races.csv`` yet. ``--predict-grid`` uses a second,
independent model that forecasts the qualifying grid itself — see
"Honest limitations" in the README; it is strictly less reliable than a real
qualifying grid (``--auto-grid``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.inference.build_quali_features import build_pre_quali_features, resolve_entry_list
from src.inference.build_race_features import build_pre_race_features
from src.inference.predict_quali import predict_quali, predicted_grid_dict
from src.inference.predict_race import predict_race
from src.inference.qualifying import fetch_qualifying_grid
from src.inference.schedule import find_next_race

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
    parser.add_argument("--next-race", action="store_true",
                        help="auto-detect the next upcoming race from races.csv, "
                             "fetch its qualifying grid (or predict it if "
                             "qualifying hasn't happened yet), and predict. Needs "
                             "no other flags.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--circuit", default=None, help="circuit_id, e.g. red_bull_ring")
    parser.add_argument("--date", default=None, help="race date YYYY-MM-DD")
    parser.add_argument("--rainfall", type=float, default=0.0,
                        help="forecast rainfall (>0 sets the wet-race flag).")
    parser.add_argument("--grid", default=None,
                        help='starting grid as "CODE:POS,CODE:POS,..." '
                             "(omit if using --auto-grid/--predict-grid).")
    parser.add_argument("--auto-grid", action="store_true",
                        help="fetch the real qualifying grid from "
                             "data/raw/qualifying.csv for --season/--round, "
                             "ingesting the season first if it's missing.")
    parser.add_argument("--predict-grid", action="store_true",
                        help="predict the qualifying grid with the qualifying "
                             "model instead of using a real one (for races "
                             "that haven't qualified yet). See --entries.")
    parser.add_argument("--entries", default=None,
                        help='comma-separated driver codes for --predict-grid, '
                             'e.g. "VER,NOR,LEC,...". Defaults to the entry '
                             "list from the season's most recent completed "
                             "race.")
    parser.add_argument("--round", type=int, default=None,
                        help="round number (else looked up from races.csv). "
                             "Required for --auto-grid/--predict-grid if the "
                             "race isn't in races.csv yet.")
    parser.add_argument("--features-path", default="data/processed/features.parquet")
    parser.add_argument("--out", default=None, help="output JSON path.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
    )

    quali_pred = None  # set to a DataFrame when the grid came from the quali model

    if args.next_race:
        # Fully automatic: detect the next race, then fetch/predict its grid.
        nxt = find_next_race()
        season, round_num = nxt.season, nxt.round
        circuit_id, date_str, tag = nxt.circuit_id, nxt.race_date, nxt.tag
        if nxt.days_until <= 3 and args.rainfall == 0.0:
            print(f"\n[!] {circuit_id} is {nxt.days_until} day(s) away and no "
                  "--rainfall was set. If rain is forecast, re-run with "
                  "--rainfall <mm> to enable the wet-race flag.\n")
        try:
            grid = fetch_qualifying_grid(season, round_num)
            grid_source = "real"
        except ValueError:
            print(f"\n[i] No real qualifying available yet for {circuit_id} — "
                  "predicting the grid with the qualifying model instead.\n")
            entries = resolve_entry_list(season)
            quali_features = build_pre_quali_features(
                season, round_num, circuit_id, date_str, entries,
                features_path=args.features_path, rainfall=args.rainfall,
            )
            quali_pred = predict_quali(quali_features)
            grid = predicted_grid_dict(quali_pred)
            grid_source = "predicted"
    else:
        missing = [name for name, val in (("--season", args.season),
                   ("--circuit", args.circuit), ("--date", args.date)) if not val]
        if missing:
            parser.error(f"{', '.join(missing)} required unless --next-race is used.")
        modes_set = sum([bool(args.grid), args.auto_grid, args.predict_grid])
        if modes_set != 1:
            parser.error("provide exactly one of --grid, --auto-grid, or --predict-grid.")
        season, circuit_id, date_str = args.season, args.circuit, args.date
        looked_round, tag = lookup_race(season, circuit_id)
        round_num = args.round if args.round is not None else (looked_round or 0)
        if looked_round is None and args.round is None:
            logger.warning("Race not found in races.csv for season %s circuit %s; "
                           "using round=%d (pass --round for the real round).",
                           season, circuit_id, round_num)
        if args.auto_grid:
            if round_num == 0:
                parser.error("--auto-grid needs a round number; pass --round N "
                             "(the race isn't in races.csv yet).")
            grid = fetch_qualifying_grid(season, round_num)
            grid_source = "real"
        elif args.predict_grid:
            if round_num == 0:
                parser.error("--predict-grid needs a round number; pass --round N "
                             "(the race isn't in races.csv yet).")
            entries = ([c.strip().upper() for c in args.entries.split(",")]
                       if args.entries else resolve_entry_list(season))
            quali_features = build_pre_quali_features(
                season, round_num, circuit_id, date_str, entries,
                features_path=args.features_path, rainfall=args.rainfall,
            )
            quali_pred = predict_quali(quali_features)
            grid = predicted_grid_dict(quali_pred)
            grid_source = "predicted"
        else:
            grid = parse_grid(args.grid)
            grid_source = "manual"

    features_df = build_pre_race_features(
        season=season, round_num=round_num, circuit_id=circuit_id,
        race_date=date_str, grid=grid, features_path=args.features_path,
        rainfall=args.rainfall,
    )
    result = predict_race(features_df)

    print(f"\n=== Predicted {season} round {round_num} — {circuit_id} "
          f"({date_str}, rainfall={args.rainfall}, grid={grid_source}) ===")
    print(format_table(result))

    out_path = Path(args.out) if args.out else REPORTS_DIR / f"prediction_{season}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "season": season,
            "round": round_num,
            "circuit_id": circuit_id,
            "race_date": date_str,
            "rainfall": args.rainfall,
            "model": "lightgbm",
            "grid_source": grid_source,
            "n_drivers": len(result),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "predictions": result.to_dict(orient="records"),
    }
    if quali_pred is not None:
        quali_manifest_path = Path("models/quali_manifest.json")
        quali_test_mae = None
        if quali_manifest_path.exists():
            quali_test_mae = json.loads(quali_manifest_path.read_text()).get("test_mae")
        payload["quali_prediction"] = {
            "model_test_mae": quali_test_mae,
            "predicted_grid": quali_pred.to_dict(orient="records"),
        }
        print(f"\n[i] Grid was PREDICTED (qualifying model test MAE: "
              f"{quali_test_mae}), not real — race forecast is rougher than "
              "usual on top of the model's own error.")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
