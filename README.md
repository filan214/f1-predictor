# 🏎️ F1 Race Outcome Predictor

Predicts Formula 1 race finishing order from information knowable **before lights-out** — the starting grid, each driver's and team's current form, circuit characteristics, and weather. Trained on **15 seasons (2010–2024)** of race data, it outputs a full ranked field plus **win / podium / points probabilities** for every driver, and can forecast a race that **hasn't happened yet**.

> **Example — 2026 Austrian Grand Prix** (generated pre-race from 2025–26 form):
> | P | Driver | Team | Win % | Podium % |
> |---|--------|------|------:|---------:|
> | 1 | LEC | Ferrari | 23.3% | 55.7% |
> | 2 | HAM | Ferrari | 16.3% | 45.1% |
> | 3 | VER | Red Bull | 15.3% | 44.3% |
> | 4 | RUS | Mercedes | 12.2% | 37.9% |
> | 5 | NOR | McLaren | 12.0% | 37.4% |

---

## Why this project

Betting-market and pundit predictions are opaque. This is a fully reproducible, **leakage-audited** ML pipeline that goes end-to-end: raw API ingestion → feature engineering → temporal model selection → live pre-race inference. Every design decision (temporal splits, dual-ELO ratings, Monte-Carlo probabilities) is chosen to answer one question honestly: *how much of a race result is actually predictable before the start?*

**The honest answer: quite a lot of the order, very little of the winner.** See [Results](#results).

---

## Results (2024 test season — never seen during training)

| Model | MAE ↓ | Spearman ↑ | Winner acc. | Podium F1 | Points acc. |
|-------|------:|-----------:|------------:|----------:|------------:|
| Baseline (grid = result) | 2.61 | 0.785 | 0.458 | 0.667 | 0.837 |
| Random Forest | 2.27 | 0.821 | 0.500 | 0.639 | 0.854 |
| XGBoost | 2.33 | 0.812 | 0.417 | 0.653 | 0.845 |
| **LightGBM** *(production)* | **2.23** | **0.823** | **0.583** | **0.694** | **0.858** |
| Stacking ensemble | 2.22 | 0.823 | 0.583 | 0.694 | 0.858 |

**MAE = mean absolute error in finishing positions.** The model lands each driver **~2.2 places** from their true finish, beating the naïve "you finish where you qualified" baseline by ~14%. It calls the winner correctly ~58% of the time and gets points-scorers right ~86% of the time.

> **Reproduced, not cherry-picked:** I also tested folding the 2023 validation season into training (a common "use all your data" move). It **slightly hurt** 2024 MAE (2.22 → 2.27), so the leaner split is kept. The negative result is reported in `reports/model_metrics_test_comparison.csv` and `notebooks/03_model_results.ipynb` rather than hidden.

---

## How it works

```
Jolpica (Ergast) API ──┐
                       ├──► data/raw ──► feature engineering ──► features.parquet ──► models ──► prediction
FastF1 (2018+ telem.) ─┘
```

**1. Ingestion** (`src/ingestion/`) — checkpointed, append-only, rate-limit-aware collection of results, qualifying, pit stops and standings from the Jolpica API, plus tyre/weather telemetry from FastF1.

**2. Features** (`src/features/`) — ~50 engineered signals, all built strictly "as of before the race" (leakage-audited, max feature↔target corr 0.77):
- **Dual ELO ratings** for drivers *and* constructors, carried across seasons
- **Rolling form** (recent finishes, points, DNFs)
- **Circuit history** (overtaking difficulty, pole-to-win %, DNF rate) via expanding windows
- **Weather climatology** per venue + a wet-race flag
- **Qualifying pace** (gap to pole, session reached, teammate delta)

**3. Modeling** (`src/models/`) — temporal split (train 2010–2022 / validate 2023 / test 2024), Optuna-tuned RF, XGBoost and LightGBM base learners, combined by a non-negative linear meta-learner over **TimeSeriesSplit out-of-fold** predictions (no leakage into the stack).

**4. Live inference** (`src/inference/`) — carries each driver's most-recent feature row forward as their pre-race state, overrides it with the actual grid + venue + forecast, and runs the model. **Win/podium/points probabilities come from a 20k-draw Monte-Carlo simulation** that ranks the field in each draw — so exactly one driver wins, three reach the podium, and ten score in every simulated race (the probabilities are mutually consistent, unlike independent per-driver sigmoids).

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Ingest the data (checkpointed; safe to resume). ~1–2 h on first run.
python run_ingestion.py                     # 2010–2024
python run_ingestion.py --seasons 2025 2026 # extend to current seasons

# 3. Build features
python -m src.features.build_features

# 4. Train + evaluate (writes models/ and reports/)
python -m src.models.train

# 5. Forecast an upcoming race
python run_predict.py \
  --season 2026 --circuit red_bull_ring --date 2026-06-28 --rainfall 0 \
  --grid "VER:1,NOR:2,LEC:3,HAM:4,PIA:5,RUS:6,SAI:7,ALO:8,ANT:9,TSU:10,LAW:11,STR:12,HUL:13,BEA:14,OCO:15,GAS:16,DOO:17,HAD:18,BOR:19,MAG:20"
```

Step 5 prints a ranked table and writes `reports/prediction_2026_austria.json`.

> **Note:** `data/`, `models/` and FastF1 caches are git-ignored — they're regenerated by steps 2–4. Only source code, notebooks and reports are tracked.

---

## Project structure

```
src/
├── ingestion/   ergast.py, fastf1_collector.py      — API + telemetry collection
├── features/    driver_ratings.py (ELO), rolling.py, circuit.py, weather.py
├── models/      dataset.py, train.py, evaluate.py, explain.py (SHAP), baseline.py
└── inference/   build_race_features.py, predict_race.py   — live forecasting
notebooks/       02_feature_analysis.ipynb, 03_model_results.ipynb
reports/         metrics CSVs, SHAP plots, prediction JSONs
run_ingestion.py, run_predict.py                     — CLI entry points
```

---

## Honest limitations

- **The winner is barely predictable.** Even the best model is right ~58% of the time; F1 is genuinely high-variance (safety cars, strategy, DNFs). The model is far better at the *overall order* than at the *podium lottery*.
- **The 2026 forecast is "form carried forward."** The model has never seen a 2026 car, and **2026 is a major regulation reset** — so treat that prediction as *current driver/team form projected onto a new season*, not a calibrated 2026 probability. Swap in the **real qualifying grid** after Saturday for a genuine forecast.
- **No mid-race dynamics.** This predicts from the pre-race state only; it doesn't model in-race strategy, tyre degradation live, or weather changes during the race.

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · FastF1 · [Jolpica/Ergast API](https://api.jolpi.ca/ergast/f1)

---

*Built as a portfolio project. Data © Jolpica/Ergast and FastF1 under their respective terms.*
