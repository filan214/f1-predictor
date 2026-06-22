# Product Requirements Document — F1 Race Outcome Predictor

**Version:** 1.0
**Status:** Planning → Phase 1 (Data Ingestion) in progress
**Owner:** Filan
**Last updated:** June 2026

---

## 1. Overview

### 1.1 Problem statement

Formula 1 race outcomes are shaped by a tangle of interacting factors — grid position, driver and constructor form, circuit characteristics, weather, and tire strategy. No single factor explains results, and the dominant one (grid position) leaves a large amount of variance unexplained, precisely in the high-value cases that interest fans and analysts: when does a driver outperform their starting slot, and why.

This project trains a machine learning model on historical F1 data (2010–2024) to forecast race results before lights-out, using only information available pre-race. The deliverable is both a defensible ML pipeline and an explainable analysis of *what drives* F1 outcomes.

### 1.2 Goals

- Predict race finishing positions and derived outcomes (winner, podium, points) for every driver in a race.
- Beat a grid-position baseline by a clear, measurable margin.
- Produce explainable predictions (SHAP) that surface genuine domain insight, not just accuracy numbers.
- Stay entirely within free-tier services — zero monthly cost.
- Serve as a portfolio centerpiece demonstrating end-to-end ML: data engineering, feature design, modeling, evaluation, and (optionally) deployment.

### 1.3 Non-goals

- **Live in-race prediction.** This is a pre-race forecast, not a lap-by-lap live model.
- **Betting or odds generation.** Probabilities are for analysis and explainability, not wagering.
- **Pre-2010 historical analysis.** Regulation eras before 2010 introduce noise that hurts relevance to modern racing.
- **Telemetry-level modeling.** We use aggregated stint/weather data, not raw car telemetry.

---

## 2. Success metrics

### 2.1 Primary metric

**Mean Absolute Error (MAE) on finishing position.** Target: beat the grid-position baseline (≈2.1 MAE) by at least 25%, i.e. reach **≤1.5 MAE** on a held-out test season.

### 2.2 Secondary metrics

| Metric | Target | What it measures |
|---|---|---|
| Spearman rank correlation | ≥ 0.88 | Quality of the predicted finishing *order* |
| Winner log-loss | Lower than baseline | Calibration of win probabilities |
| Podium F1-score | ≥ 0.65 | Precision + recall on top-3 detection |
| Points-finish accuracy | ≥ 0.80 | Top-10 binary classification accuracy |

### 2.3 Qualitative success

- SHAP analysis yields at least 3 non-obvious, defensible insights about feature interactions (e.g. wet-weather driver effects, circuit-specific overtaking).
- The model demonstrably has no temporal leakage — verified by the baseline comparison and a leakage audit.

---

## 3. Prediction targets

The pipeline produces four outputs per driver per race, trained as related but distinct tasks:

| Target | Type | Notes |
|---|---|---|
| Final position (P1–P20) | Ordinal regression | Primary target; MAE-optimized |
| Race winner | Binary classification | Highest-value, hardest target |
| Podium finish (top 3) | Binary classification | |
| Points finish (top 10) | Binary classification | |

A single regression model predicting position can derive all four by ranking; alternatively, train task-specific classifiers. The plan starts with position regression and adds classifiers if the derived outputs underperform.

---

## 4. Data requirements

### 4.1 Sources

| Source | Access | Coverage | Provides |
|---|---|---|---|
| **Jolpica F1 API** | Free, no auth (Ergast replacement) | 1950–present | Results, qualifying, pit stops, standings |
| **FastF1** | Python library, free | 2018–present | Tire stints, race weather, qualifying pace |
| **OpenWeatherMap** | Free tier, 1,000 calls/day | Historical by lat/lon | Weather fallback for pre-2018 races |
| **Kaggle F1 dataset** | Free download | 1950–2020 | Optional bootstrap for early EDA |

> **Note:** The original Ergast API was retired in January 2025. Jolpica (`api.jolpi.ca/ergast/f1/`) is the drop-in community replacement with an identical JSON format.

### 4.2 Scope

- **Training window:** 2010–2024 (~290 races).
- **Era weighting consideration:** the 2014+ turbo-hybrid era is most predictive of current conditions; document this when interpreting results.
- **FastF1 features** are only available 2018+. Plan handles this with explicit null-flagging and an OpenWeatherMap fallback rather than dropping pre-2018 races.

### 4.3 Raw output tables

`races`, `results`, `qualifying`, `pit_stops`, `driver_standings`, `constructor_standings`, `tire_stints`, `race_weather`, `qualifying_pace_fastf1` — written as CSVs to `data/raw/`.

---

## 5. Feature engineering

Features are computed **per driver per race** and must only use information available before the race start. Approximate importance tiers (to be confirmed empirically via SHAP):

### 5.1 Grid & qualifying (highest impact)
Grid position, gap to pole (seconds), Q1/Q2/Q3 progression, qualifying pace vs teammate, grid vs championship-position delta.

### 5.2 Driver performance (highest impact)
Rolling average finish (last 5 races), Bayesian/ELO driver rating carried forward across seasons, circuit-specific historical finish, career DNF rate, wet-race performance index.

### 5.3 Constructor / team (high impact)
Constructor championship position, reliability index (DNF rate last 5 races), points-per-race rolling average, average pit stop execution time.

### 5.4 Circuit characteristics (medium impact)
Overtaking opportunity index, street vs permanent, historical safety-car probability, "pole wins" percentage at the circuit.

### 5.5 Weather (medium impact)
Rain probability, air temperature, wind speed, humidity — most powerful when crossed with driver-specific wet performance.

### 5.6 Strategy (moderate impact)
Expected pit stop count, starting tire compound, circuit tire degradation rate, undercut/overcut suitability index.

### 5.7 Leakage controls
- Standings are taken as of *before* the race (after round N-1).
- Rolling features use a strict trailing window with no peek at the target race.
- A dedicated audit checks no feature correlates suspiciously with the outcome in ways implying future knowledge.

---

## 6. Modeling approach

Progressive complexity, each stage benchmarked against the previous:

| Stage | Model | Purpose | Target MAE |
|---|---|---|---|
| 0 | Grid-position heuristic | Floor to beat | ~2.1 |
| 1 | Random Forest | Feature-importance intuition | ~1.7 |
| 2 | XGBoost + LightGBM (Optuna-tuned) | Best single model | ~1.4 |
| 3 | Stacking ensemble | Meta-learner over base models | ~1.2 |

### 6.1 Validation — temporal, never random
Train on 2010–2022, validate on 2023, test on 2024. Use `TimeSeriesSplit` for cross-validation. Random splits leak future races into training and invalidate results.

### 6.2 Explainability
SHAP values on the best model. Global importance plus per-prediction breakdowns for narrative case studies (e.g. a surprise podium).

---

## 7. Technical stack

**ML / data:** Python 3.11+, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, Optuna, FastF1, requests, tqdm.

**Analysis / viz:** matplotlib, seaborn, plotly, Tableau Public, Jupyter.

**Dashboard (optional, Phase 4):** Next.js 15, TypeScript, Supabase (PostgreSQL), Recharts, shadcn/ui, Vercel.

**Architecture reuse:** Python ML pipeline exports predictions as structured JSON → Supabase `race_predictions` table → Next.js dashboard renders pre-race forecasts. This mirrors the FormWatch ingest architecture and the Smart Finn Track structured-JSON + caching pattern.

---

## 8. Deliverables

1. Data ingestion module (Jolpica + FastF1 collectors) — **complete**.
2. Feature engineering pipeline.
3. EDA notebook with distribution and correlation analysis.
4. Model training pipeline with temporal CV and Optuna tuning.
5. Evaluation report (metrics + SHAP insights).
6. (Optional) Next.js prediction dashboard on Vercel.
7. Technical PRD (this document) + recruiter-facing narrative version.

---

## 9. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Temporal leakage | Invalid results | Strict trailing windows; baseline audit; pre-race standings only |
| Grid position dominates, model adds little | Weak portfolio story | Focus features and SHAP on grid-vs-finish deltas |
| FastF1 only covers 2018+ | Missing features pre-2018 | Null-flag + OpenWeatherMap fallback; era-aware analysis |
| Jolpica rate limits / downtime | Slow ingestion | Checkpoint system; request throttling; Kaggle bootstrap |
| Class imbalance (winners are rare) | Poor winner detection | Class weights; calibrated probabilities; F1 over accuracy |
| RFM/NTILE-style skew analog (most drivers cluster mid-pack) | Misleading splits | Inspect distributions; consider ordinal-aware loss |

---

## 10. Timeline

| Phase | Scope | Duration |
|---|---|---|
| 1 | Data ingestion + EDA | 2–3 weeks |
| 2 | Feature engineering | 2 weeks |
| 3 | Modeling + evaluation | 2–3 weeks |
| 4 | Dashboard (optional) | 2–3 weeks |

**Total:** ~2–3 months for the full build, or ~6 weeks for the ML core without the dashboard.
