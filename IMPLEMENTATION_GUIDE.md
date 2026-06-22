# Implementation Guide — F1 Race Outcome Predictor

**Companion to:** PRD v1.0
**Audience:** You (the builder), and any Claude Code conversation implementing a phase
**Last updated:** June 2026

This guide takes the project from empty folder to working model (and optional dashboard), phase by phase. Each phase lists its goal, the files to create, key implementation notes, and a "done when" checklist before moving on.

---

## Project structure (target end state)

```
f1-predictor/
├── run_ingestion.py              # Phase 1 — entry point
├── requirements.txt
├── .env.example  →  .env
├── README.md
├── data/
│   ├── raw/                      # CSVs from ingestion
│   │   └── checkpoints/          # Resume-from-failure state
│   ├── fastf1_cache/             # FastF1 disk cache
│   └── processed/                # Feature matrix output
├── notebooks/
│   ├── 01_eda.ipynb              # Phase 1
│   ├── 02_feature_analysis.ipynb # Phase 2
│   └── 03_model_results.ipynb    # Phase 3
├── src/
│   ├── __init__.py
│   ├── ingestion/                # Phase 1 — DONE
│   │   ├── __init__.py
│   │   ├── ergast.py
│   │   └── fastf1_collector.py
│   ├── features/                 # Phase 2
│   │   ├── __init__.py
│   │   ├── build_features.py     # Orchestrator
│   │   ├── driver_ratings.py     # ELO / Bayesian
│   │   ├── rolling.py            # Rolling averages
│   │   ├── circuit.py            # Circuit characteristics
│   │   └── weather.py            # Weather joins + fallback
│   ├── models/                   # Phase 3
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py            # SHAP
│   └── export/                   # Phase 4 (optional)
│       └── to_supabase.py
└── dashboard/                    # Phase 4 (optional) — Next.js app
```

> Keep each Claude Code conversation scoped to one `src/` subpackage. Tell it explicitly not to modify files outside the target folder — the same discipline you use on FormWatch and Smart Finn Track.

---

## Phase 1 — Data ingestion + EDA  *(2–3 weeks)*

**Status: ingestion code complete.** Remaining work is running it and exploring the output.

### Step 1.1 — Run ingestion
```bash
pip install -r requirements.txt
cp .env.example .env          # fill in keys later; not needed for Jolpica
python run_ingestion.py --season 2024     # smoke test first
python run_ingestion.py                   # full 2010–2024 run (~2–3 hrs)
```
The checkpoint system means an interrupted run resumes from the last completed race. FastF1's first run per session downloads ~150MB, cached thereafter.

### Step 1.2 — EDA notebook (`notebooks/01_eda.ipynb`)
Explore before engineering features. Answer:
- How strong is grid position alone? (Compute Spearman r vs finish — expect ≈0.85.)
- What's the DNF rate by constructor and era?
- How are finishing positions distributed? (Expect mid-pack clustering — relevant to loss choice.)
- How many races have rain? (Wet races are rare but high-signal.)
- Are there data gaps — missing qualifying, sprint-weekend oddities, grid penalties?

### Done when
- [ ] All 9 raw CSVs populated in `data/raw/`.
- [ ] EDA notebook documents the grid-position baseline strength.
- [ ] Known data quirks (sprint races, grid penalties, DNS/DNF coding) are catalogued.

---

## Phase 2 — Feature engineering  *(2 weeks)*

**Goal:** turn raw CSVs into one clean feature matrix (`data/processed/features.parquet`), one row per driver per race, with zero temporal leakage.

### Step 2.1 — Driver ratings (`src/features/driver_ratings.py`)
Implement a Bayesian/ELO rating updated after each race. **Carry the rating forward across seasons** — this is the key methodological choice (same principle as FormWatch's Bayesian baseline carry-forward). A driver changing teams keeps their skill rating rather than resetting.

Implementation notes:
- Start every driver at a baseline rating (e.g. 1500 ELO).
- Update after each race based on finishing position relative to expected.
- Weight the update by constructor strength so team performance doesn't fully contaminate driver skill.
- Store the rating *as of before* each race as the feature value.

### Step 2.2 — Rolling features (`src/features/rolling.py`)
Trailing-window aggregates with no peek at the current race:
- Driver: avg finish, avg points, DNF rate (last 5 races).
- Constructor: reliability index, points-per-race, avg pit stop time.
Use `groupby(...).shift(1).rolling(window)` patterns so the current race is never included.

### Step 2.3 — Circuit features (`src/features/circuit.py`)
Per-circuit historical stats: overtaking index (positions changed start→finish), pole-wins %, safety-car probability, street vs permanent flag.

### Step 2.4 — Weather (`src/features/weather.py`)
Join FastF1 `race_weather` where available (2018+). For earlier races, fall back to OpenWeatherMap historical by circuit lat/lon + race date. Always emit a `weather_source` flag and a `weather_missing` indicator.

### Step 2.5 — Orchestrate (`src/features/build_features.py`)
Join everything into the final matrix. Add a `feature_analysis.ipynb` to inspect correlations and run the leakage audit (no feature should near-perfectly predict the target).

### Done when
- [ ] `features.parquet` exists, one row per driver per race.
- [ ] Driver ratings carry across seasons and update correctly.
- [ ] Every rolling feature verified leakage-free (spot-check a few rows by hand).
- [ ] Weather coverage documented with source flags.

---

## Phase 3 — Modeling + evaluation  *(2–3 weeks)*

**Goal:** beat the baseline by ≥25% MAE with an explainable model.

### Step 3.1 — Baseline (`src/models/baseline.py`)
Predict finish = grid position. Compute MAE and Spearman r on the test season. **Every later model must clearly beat this** — if it doesn't, audit for leakage or feature bugs before tuning.

### Step 3.2 — Train (`src/models/train.py`)
1. Random Forest first — read `feature_importances_` for intuition.
2. XGBoost + LightGBM, tuned with Optuna (`n_estimators`, `max_depth`, `learning_rate`, `subsample`).
3. Stacking ensemble — base models → LinearRegression meta-learner, with proper stacking CV so the meta-learner never sees base predictions on data they trained on.

**Critical:** temporal split throughout — train 2010–2022, validate 2023, test 2024. Use `TimeSeriesSplit` for CV. Never `train_test_split` with shuffle.

### Step 3.3 — Evaluate (`src/models/evaluate.py`)
Report all primary + secondary metrics from the PRD. Compare every model against the baseline in one table. Derive winner/podium/points outcomes from the position predictions by ranking within each race.

### Step 3.4 — Explain (`src/models/explain.py`)
SHAP on the best model: global importance + per-prediction breakdowns. Pull at least 3 narrative case studies (e.g. a correctly predicted surprise podium, a wet-race driver effect). These are your interview talking points.

### Done when
- [ ] Best model reaches ≤1.5 MAE on 2024 test.
- [ ] Results table compares all stages vs baseline.
- [ ] SHAP surfaces ≥3 defensible insights.
- [ ] Predictions exportable as structured JSON.

---

## Phase 4 — Dashboard  *(optional, 2–3 weeks)*

**Goal:** a pre-race forecast page reusing your existing stack. Skip if the portfolio goal is met by the notebook + report.

### Step 4.1 — Export (`src/export/to_supabase.py`)
Write predictions to a Supabase `race_predictions` table as structured JSON (typed fields, not raw markdown — same pattern as Smart Finn Track). Cache so the dashboard doesn't recompute on each load.

> Remember the Supabase URL distinction: the `https://<id>.supabase.co` Project URL is for the JS client; the PostgreSQL pooler connection string is for any direct DB / Drizzle access.

### Step 4.2 — Dashboard (`dashboard/`)
Next.js 15 App Router + TypeScript. Read predictions from Supabase, render with Recharts: win-probability bars, predicted grid-to-finish movement, per-driver SHAP contribution. shadcn/ui for components, deploy to Vercel.

### Done when
- [ ] Predictions live in Supabase.
- [ ] Dashboard renders a pre-race forecast for a selected race.
- [ ] Deployed to Vercel on free tier.

---

## Cross-cutting principles

- **No leakage, ever.** Every feature answers "was this knowable before lights-out?" If no, drop it.
- **Baseline first, always.** Beating grid position is the whole game; measure against it constantly.
- **Temporal splits only.** Random shuffling silently invalidates the entire project.
- **Structured JSON outputs** for any AI/UI handoff, with caching — carried over from your other projects.
- **Scope each Claude Code session** to one subpackage and forbid edits outside it.
- **Free tier throughout** — Jolpica, FastF1, OpenWeatherMap, Supabase, Vercel, Tableau Public.

---

## Suggested next action

Phase 1 ingestion is built. The natural next step is **Phase 2, Step 2.1 — the driver ratings module** — it's the most distinctive, interview-worthy piece and reuses your FormWatch Bayesian carry-forward logic directly.
