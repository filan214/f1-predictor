"""Feature engineering package.

Turns the raw ingestion CSVs in ``data/raw/`` into a single, leakage-free
feature matrix (``data/processed/features.parquet``) with one row per driver
per race. See :mod:`src.features.build_features` for the orchestrator.
"""
