"""Unit tests for run_predict.py's argument wiring and grid-source branching.

These monkeypatch the heavy pipeline functions (fetch/predict/build) so the
test exercises only the CLI logic: mutual-exclusivity validation, and which
branch sets ``grid_source`` in the saved JSON.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import run_predict


def _stub_race_result() -> pd.DataFrame:
    return pd.DataFrame({
        "predicted_rank": [1], "driver_code": ["VER"], "constructor_id": ["red_bull"],
        "grid_position": [1], "predicted_position_raw": [1.2],
        "win_probability": [0.5], "podium_probability": [0.8], "points_probability": [0.9],
    })


def _stub_quali_pred() -> pd.DataFrame:
    return pd.DataFrame({
        "predicted_grid": [1], "driver_code": ["VER"],
        "constructor_id": ["red_bull"], "predicted_position_raw": [1.1],
    })


@pytest.fixture(autouse=True)
def stub_heavy_functions(monkeypatch, tmp_path):
    monkeypatch.setattr(run_predict, "build_pre_race_features",
                         lambda **kw: pd.DataFrame({"driver_code": ["VER"]}))
    monkeypatch.setattr(run_predict, "predict_race", lambda df: _stub_race_result())
    monkeypatch.setattr(run_predict, "REPORTS_DIR", tmp_path)
    return tmp_path


def test_manual_mode_requires_a_grid_source():
    with pytest.raises(SystemExit):
        run_predict.main(["--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02"])


def test_manual_mode_rejects_two_grid_sources():
    with pytest.raises(SystemExit):
        run_predict.main([
            "--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02",
            "--grid", "VER:1", "--auto-grid",
        ])


def test_predict_grid_sets_grid_source_predicted(monkeypatch, stub_heavy_functions):
    monkeypatch.setattr(run_predict, "resolve_entry_list", lambda season: ["VER"])
    monkeypatch.setattr(run_predict, "build_pre_quali_features",
                         lambda *a, **kw: pd.DataFrame({"driver_code": ["VER"]}))
    monkeypatch.setattr(run_predict, "predict_quali", lambda df: _stub_quali_pred())
    monkeypatch.setattr(run_predict, "predicted_grid_dict", lambda pred: {"VER": 1})

    run_predict.main([
        "--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02",
        "--predict-grid",
    ])

    payload = json.loads((stub_heavy_functions / "prediction_2024_bahrain.json").read_text())
    assert payload["meta"]["grid_source"] == "predicted"
    assert "quali_prediction" in payload


def test_manual_grid_sets_grid_source_manual(stub_heavy_functions):
    run_predict.main([
        "--season", "2024", "--circuit", "bahrain", "--date", "2024-03-02",
        "--grid", "VER:1",
    ])

    payload = json.loads((stub_heavy_functions / "prediction_2024_bahrain.json").read_text())
    assert payload["meta"]["grid_source"] == "manual"
    assert "quali_prediction" not in payload


def test_next_race_falls_back_to_predicted_grid_on_value_error(monkeypatch, stub_heavy_functions):
    from src.inference.schedule import NextRace

    nxt = NextRace(season=2026, round=10, circuit_id="spa", race_date="2026-07-19",
                   country="Belgium", tag="belgium", days_until=13)
    monkeypatch.setattr(run_predict, "find_next_race", lambda: nxt)

    def _raise(*a, **kw):
        raise ValueError("no quali yet")

    monkeypatch.setattr(run_predict, "fetch_qualifying_grid", _raise)
    monkeypatch.setattr(run_predict, "resolve_entry_list", lambda season: ["VER"])
    monkeypatch.setattr(run_predict, "build_pre_quali_features",
                         lambda *a, **kw: pd.DataFrame({"driver_code": ["VER"]}))
    monkeypatch.setattr(run_predict, "predict_quali", lambda df: _stub_quali_pred())
    monkeypatch.setattr(run_predict, "predicted_grid_dict", lambda pred: {"VER": 1})

    run_predict.main(["--next-race"])

    payload = json.loads((stub_heavy_functions / "prediction_2026_belgium.json").read_text())
    assert payload["meta"]["grid_source"] == "predicted"


def test_next_race_uses_real_grid_when_qualifying_exists(monkeypatch, stub_heavy_functions):
    from src.inference.schedule import NextRace

    nxt = NextRace(season=2024, round=1, circuit_id="bahrain", race_date="2024-03-02",
                   country="Bahrain", tag="bahrain", days_until=30)
    monkeypatch.setattr(run_predict, "find_next_race", lambda: nxt)
    monkeypatch.setattr(run_predict, "fetch_qualifying_grid", lambda season, rnd: {"VER": 1})

    run_predict.main(["--next-race"])

    payload = json.loads((stub_heavy_functions / "prediction_2024_bahrain.json").read_text())
    assert payload["meta"]["grid_source"] == "real"
    assert "quali_prediction" not in payload
