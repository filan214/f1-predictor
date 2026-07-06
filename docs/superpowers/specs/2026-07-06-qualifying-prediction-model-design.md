# Qualifying-Prediction Model + Two-Stage Pipeline — Design

**Status:** Approved
**Date:** 2026-07-06

## 1. Problem

The existing race model (`src/models/`, LightGBM, test MAE 2.23) takes the starting
grid as an **input** feature — it cannot forecast a race until real qualifying has
happened. This blocks predicting races further in advance (e.g. Spa before its
qualifying session runs).

This project adds a second, independent model that predicts the qualifying grid
itself, plus a pipeline that chains it into the existing race model for a full
pre-qualifying race forecast.

## 2. Goals / Non-goals

- Goal: a qualifying-position regression model (mirroring the race model's
  rigor: temporal split, baseline, Optuna-tuned RF/XGB/LGBM stack, SHAP).
- Goal: a two-stage inference path — predict grid → feed race model — wired into
  `run_predict.py` via a new `--predict-grid` flag.
- Non-goal: changing `src/features/`, `features.parquet`, or the existing race
  model/pipeline in any way.
- Non-goal: live/intra-qualifying prediction (Q1→Q2→Q3 progression modeling).

## 3. Architecture

```
src/models/
  quali_dataset.py     # quali design matrix, rolling-quali features, temporal split, target=grid_clean
  train_quali.py       # baseline -> RF/XGB/LGBM -> Optuna -> stacking; saves models + manifest
  evaluate_quali.py    # MAE / Spearman / pole & top-3(Q3) accuracy vs baseline
src/inference/
  build_quali_features.py  # per-driver pre-QUALIFYING feature row (carry-forward, NO grid inputs)
  predict_quali.py         # load quali model -> predict grid order -> {driver_code: grid_slot}
models/
  quali_lgbm.joblib, quali_xgb.joblib, quali_rf.joblib, quali_stack_meta.joblib,
  quali_preprocessor.joblib, quali_manifest.json
run_predict.py         # + --predict-grid flag: use quali model instead of real qualifying
```

No edits to `src/features/`, `src/models/dataset.py`, `src/models/train.py`, or
`data/processed/features.parquet`. The existing race model and its inference path
are untouched; all new code is additive.

## 4. Target & features

**Target:** `grid_clean` from `features.parquet` (the actual starting slot; 0 NaN
across 7065 rows). Trained as regression, then ranked within each race to produce
a valid 1..N grid permutation.

**Reused features** (quali-safe subset of the race model's existing columns):
driver/constructor ELO (`driver_elo_pre`, `constructor_elo_pre`, `perf_rating_pre`,
`driver_elo_experience`), rolling driver form (`driver_avg_finish_5`,
`driver_avg_points_5`, `driver_dnf_rate_5`, `driver_form_races`), constructor form
(`constructor_reliability_5`, `constructor_points_per_race_5`,
`constructor_avg_pit_seconds_5`), standings (`championship_position`,
`championship_points`, `championship_wins`, `constructor_position`,
`constructor_points`, `constructor_wins`), all circuit features
(`circuit_is_street`, `circuit_overtaking_index`, `circuit_pole_win_pct`,
`circuit_dnf_rate_hist`, `circuit_avg_pitstops_hist`, `circuit_avg_stint_laps_hist`,
`circuit_history_races`), and weather (`air_temp_avg`, `track_temp_avg`,
`humidity_avg`, `wind_speed_avg`, `rain_flag`, `weather_missing`).

**New rolling-qualifying features**, computed in `quali_dataset.py` from
`grid_clean` via `groupby("driver_id").shift(1).rolling(5)` (strictly trailing,
same leakage discipline as the race model's rolling features):

- `driver_avg_grid_5` — driver's average starting grid slot, last 5 races.
- `driver_best_grid_5` — driver's best (minimum) grid slot, last 5 races.
- `constructor_avg_grid_5` — constructor's average grid slot (both cars), last 5
  races.
- `driver_grid_vs_teammate_5` — driver's rolling average grid slot minus
  teammate's, last 5 races (relative quali form within the same car).

**Excluded** — these either *are* the target or aren't knowable before qualifying
happens, so they must not appear as inputs: `grid_clean`, `grid_position`,
`quali_best_seconds`, `gap_to_pole_seconds`, `quali_session_reached`,
`reached_q3`, `quali_gap_to_teammate`, `grid_penalty`, `grid_vs_championship`.

## 5. Modeling approach

- Same temporal split as the race model: train ≤2022, validate 2023, test 2024
  (reuses the boundaries defined in `src/models/dataset.py`, not the module
  itself, to avoid coupling).
- **Baseline floor:** predict grid order = current championship-position order
  (i.e. rank by `championship_position` pre-race). Report its MAE/Spearman; every
  trained model must beat it.
- Progression: Random Forest (feature-importance intuition) → XGBoost + LightGBM
  (Optuna-tuned, ~30 trials) → stacking ensemble (LinearRegression meta-learner
  over base model predictions, proper stacking CV).
- Shared tuning/stacking logic factored into small helpers reused by both
  `train.py` and `train_quali.py` rather than copy-pasted, where it doesn't
  couple the two pipelines' data.
- Persist all stage models + `models/quali_manifest.json` (hyperparameters, test
  MAE, baseline MAE, improvement %, top feature importances) mirroring
  `models/manifest.json`.
- Metrics: MAE, Spearman rank correlation, pole accuracy (P(predicted P1 == real
  P1)), Q3/top-10 accuracy.

## 6. Two-stage inference pipeline

For a race that hasn't had real qualifying yet:

```
entry list (drivers)
  -> build_quali_features()   # carry-forward, no grid/quali inputs
  -> predict_quali()          # -> {driver_code: predicted grid_slot}
  -> build_pre_race_features(grid=...)   # existing, unmodified
  -> predict_race()                       # existing, unmodified
  -> race forecast JSON
```

- **Entry list**: defaults to the driver lineup from the most recent completed
  race (carried forward via the same `_latest_rows` pattern used in
  `build_race_features.py`), with an optional manual override for lineup changes.
- **`run_predict.py --predict-grid`**: when set, skip fetching real qualifying
  and instead run the quali model to synthesize a grid, then continue through
  the existing `build_pre_race_features` / `predict_race` path unchanged. Usable
  standalone or combined with `--next-race` (auto-predict the grid for the next
  race when real qualifying isn't available yet). It becomes a third option in
  the existing mutual-exclusivity check alongside `--grid` / `--auto-grid` —
  exactly one of the three must be set in manual (`--season`/`--circuit`/`--date`)
  mode, and `--next-race` mode picks `--predict-grid` automatically whenever
  `fetch_qualifying_grid` finds no real data instead of raising.
- Output JSON gains a `"grid_source": "predicted" | "real" | "manual"` field in
  `meta`, and when predicted, an embedded `quali_prediction` block (the predicted
  grid + model MAE context) so the dashboard can show both stages.

## 7. Known limitations (to document in README)

Qualifying pace is dominated by current-season car development, which the
carried-forward rating/form features approximate but lag. Expect the qualifying
model to beat the championship-order baseline by a real but modest margin — this
is a harder, noisier target than race finish. Two-stage race predictions
(`--predict-grid`) additionally inherit the qualifying model's error on top of
the race model's own, so they are strictly less reliable than predictions made
from a real qualifying grid (`--auto-grid`). This will be stated explicitly in
CLI output and the README, not just implied.

## 8. Testing

- Unit: rolling-quali features never see the current race's own grid (leakage
  check on a hand-picked driver/race); `predict_quali` output is always a valid
  1..N permutation for the field size; entry-list carry-forward selects the
  correct/most-recent lineup.
- Regression: quali model beats the championship-order baseline on the 2024 test
  season (MAE and Spearman); end-to-end `run_predict.py --predict-grid` on an
  upcoming race produces a valid race JSON with `grid_source: "predicted"`.
