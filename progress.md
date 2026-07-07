# Project Progress — F1 Race & Qualifying Predictor

_Last updated: 2026-07-08_

A two-stage machine-learning pipeline that forecasts Formula 1 results: a
**qualifying model** predicts the starting grid, which feeds a **race model** that
predicts finishing position. Trained on 2010–2024, with a live-inference path
extended to 2025–2026.

---

## ✅ What's been completed

### Data & ingestion
- Raw data ingested to `data/raw/*.csv` (9 tables: results, qualifying, driver/
  constructor standings, races, pit stops, tyre stints, weather).
- Processed dataset `data/processed/features.parquet` — **6,432 rows × 58 cols**,
  one leakage-free row per driver per race, 2010–2024 (305 races, 84 drivers).
- Data extended to **2025–2026** for live inference (next race: **Spa, 2026 round
  10, 2026-07-19**). Note: `races.csv`/`results.csv` carry partial 2025/26 rows;
  EDA/modeling are scoped to 2010–2024.

### Feature engineering (`src/features/`)
- Dual-ELO ratings (driver vs constructor), rolling form (finish/points/DNF),
  constructor reliability, circuit history, weather, standings.
- Leakage discipline: every rolling aggregate uses `groupby(...).shift(1).rolling()`;
  a `leakage_audit` fires on each build (max |corr| with finish < 0.95).

### Race model — Phase 3 (`src/models/`)
- Temporal split: **train ≤2022 / validate 2023 / test 2024**; selection on
  validation only.
- Optuna-tuned RF / XGBoost / LightGBM + positive-weight `LinearRegression`
  stacking meta.
- **Result: test MAE 2.23 vs 2.61 grid baseline (14% better)**, Spearman
  0.785→0.823. Folding-2023 experiment (`refit_full.py`) confirmed ~2.2 is the
  honest leakage-free ceiling. SHAP explainability + case studies generated.

### Qualifying model + two-stage pipeline (`src/models/quali_*`, `src/inference/`)
- Separate model predicting `grid_clean` from a **quali-safe** feature subset
  (all grid/qualifying-derived columns excluded) plus engineered rolling-grid
  features. Same stack architecture as the race model.
- **Result: deployed stack MAE 3.31 vs 3.34 championship-order baseline; pole
  accuracy 33% → 46%.** Honest, modest margin — championship order is a hard
  baseline. LightGBM has best raw MAE (3.21).
- Wired into `run_predict.py`: `--predict-grid` and `--next-race` (falls back to
  predicting the grid when real qualifying doesn't exist yet; `grid_source`
  ∈ {real, predicted, manual}).

### Notebooks — portfolio presentation (this session's focus)
- **`notebooks/01_EDA.ipynb`** (NEW) — exploratory analysis of the raw data:
  source inventory, calendar growth, grid turnover, championship dominance,
  qualifying→race conversion (pole→win 52.7%, Spearman 0.762), reliability/DNF
  trends, pit & tyre data, 2018 weather-coverage cliff.
- **`notebooks/02_feature_analysis.ipynb`** — polished: dataset, targets, grid
  baseline, leakage audit, rolling-feature hand check, dual-ELO, correlations,
  missingness.
- **`notebooks/03_model_results.ipynb`** — polished: model progression, full
  metric tables, predicted-vs-actual, race card, SHAP, case studies, folding test.
- **`notebooks/04_qualifying_model.ipynb`** — end-to-end quali model: features,
  training, evaluation, SHAP, pole hit/miss strip, slopegraph, grid-region
  segments, two-stage pipeline.
- All four share one visual theme, TL;DR hero cards, question-driven headers, and
  Styler tables. Executed end-to-end with **zero errors**; committed as `f93f590`.

---

## 🧭 Key decisions made

- **Two-stage design (Approach A):** derive quali features inside new
  `src/models/quali_*` modules from `features.parquet` — zero changes to
  `src/features/`. Qualifying model mirrors the full race stack.
- **Strict temporal split + validation-only selection** to avoid test-set leakage;
  2024 kept untouched until final reporting.
- **Deployed model = the stack, not the val-MAE-best learner.** Val MAE selects
  XGBoost, but `predict_quali` runs the stack (better pole caller). Notebook 04's
  "where it helps" visuals recompute from the stack for consistency with the
  33%→46% headline.
- **Honesty over hype:** ≤1.5 MAE PRD goal reported as unrealistic (real baseline
  2.6, large irreducible variance); quali margin reported as modest but real.
- **Notebook polish rebuilds via nbformat scripts + `nbconvert --execute`**
  (kernel `python3` → `D:\PHyton\python.exe`); analysis code reproduced verbatim
  so numbers are identical, only restyled. Training is NOT re-run in notebooks —
  they load saved artefacts for speed/determinism.
- **EDA scoped to 2010–2024** (not the raw 2025/26 rows) to match the modeling
  scope and avoid misleading partial seasons.
- **Git ownership:** the user makes all commits/pushes; implementation is left in
  the working tree with commit messages provided.

---

## ⏳ Pending / unfinished

- **Phase 4 dashboard** — not started. `reports/predictions_2024.json` and
  `reports/quali_predictions_2024.json` are ready to drive it.
- **Qualifying model margin is thin** — beats the championship baseline mainly on
  pole accuracy, not raw MAE. Room to improve (see next steps).
- **Weather forecast integration for live prediction** — `--rainfall` is manual;
  no automated forecast fetch for upcoming races.
- **2025/26 live predictions** — Belgium/Spa forecast exists
  (`reports/prediction_2026_belgium.json`); no automated per-round refresh.

---

## ▶️ Next steps

1. **Build the Phase 4 dashboard** from the prediction JSONs (race card + quali
   card, grid_source badge, confidence/limitations copy).
2. **Improve the qualifying model** — try quali-specific features (recent
   qualifying pace from `qualifying_pace_fastf1.csv`, track-type interactions),
   or a rank/learning-to-rank objective instead of regression-then-rank.
3. **Automate live inference** — schedule `run_predict.py --next-race` per race
   weekend; optionally auto-fetch a rainfall forecast.
4. **Portfolio README pass** — ensure the four-notebook arc (01→04) is linked and
   the headline numbers are current.

---

## 📄 Files / sections last touched (this session)

| File | Change |
|---|---|
| `notebooks/01_EDA.ipynb` | **Created** — raw-data EDA (8 sections), scoped 2010–2024 |
| `notebooks/02_feature_analysis.ipynb` | Rebuilt with theme, hero card, Styler tables |
| `notebooks/03_model_results.ipynb` | Rebuilt: Styler metric tables, color-coded race card |
| `notebooks/04_qualifying_model.ipynb` | Full polish: hero card, SHAP, pole strip, slopegraph, segments |
| `reports/prediction_2026_belgium.json` | Updated (committed alongside notebooks) |
| `progress.md` | **This file** |

All notebook work is committed (`f93f590`); working tree is otherwise clean.
Build scripts used to (re)generate notebooks live in the session scratchpad, not
the repo.
