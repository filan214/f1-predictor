"""Modeling package — Phase 3.

Trains a progression of models (grid baseline -> Random Forest -> Optuna-tuned
XGBoost/LightGBM -> stacking ensemble) on the leakage-free feature matrix, under
a strict temporal split (train 2010-2022, validate 2023, test 2024), and
explains the best model with SHAP.
"""
