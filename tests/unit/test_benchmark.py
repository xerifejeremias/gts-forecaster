"""Timing benchmark: synthetic data, extrapolation, per-model records."""

import pandas as pd
import pytest

from gts_forecaster.benchmark import _extrapolate, main, run_benchmark, summary_lines, synthetic_panel
from gts_forecaster.config import load_config
from gts_forecaster.models.naive import NaiveForecaster
from helpers import use_core_catalog

@pytest.fixture(autouse=True)
def _core_catalog(monkeypatch):
    use_core_catalog(monkeypatch)  # pipeline tests: no tuning variants of the tree models



def _cfg(**extra):
    return load_config({"frequency": "monthly", "horizon": 6, "n_folds": 3, **extra})


def test_synthetic_panel_looks_like_prepared_input():
    cfg = _cfg()
    data, static = synthetic_panel(cfg, 5, 30)
    assert data["unique_id"].nunique() == 5 and len(data) == 150
    assert {"unique_id", "ds", "y", "is_filled", *cfg.explanatory_columns} <= set(data.columns)
    assert (data["y"] >= 0).all()
    assert list(static.columns) == ["unique_id", *cfg.static_columns] and len(static) == 5
    assert pd.infer_freq(data[data.unique_id == "bench-000000"]["ds"]) == "MS"


def test_extrapolation_is_a_fixed_cost_plus_a_cost_per_series():
    assert _extrapolate([10, 30], [2.0, 4.0], 1000) == pytest.approx(1.0 + 0.1 * 1000)
    assert _extrapolate([10], [2.0], 1000) == pytest.approx(200.0)
    assert _extrapolate([10, 30], [3.0, 3.0], 300) == pytest.approx(30.0)  # no slope: per-series rate


def test_run_benchmark_estimates_every_model(monkeypatch):
    import gts_forecaster.benchmark as bm

    monkeypatch.setattr(bm, "build_models_from_config", lambda *a, **k: {"naive": NaiveForecaster()})
    table = run_benchmark(_cfg(), target_series=1000, history=36, sizes=[5, 10])
    row = table.iloc[0]
    assert row["status"] == "ok" and row["sizes"] == "5 10" and row["supports_update"]
    assert row["est_backtest_s"] == pytest.approx(3 * row["est_fit_predict_s"], abs=0.11)
    assert row["est_forecast_frozen_s"] == pytest.approx(row["est_predict_s"])


def test_a_failing_model_is_recorded_and_the_others_continue(monkeypatch):
    import gts_forecaster.benchmark as bm

    class Broken(NaiveForecaster):
        def _fit(self, series, train):
            raise RuntimeError("library missing")

    monkeypatch.setattr(bm, "build_models_from_config",
                        lambda *a, **k: {"broken": Broken(), "naive": NaiveForecaster()})
    table = run_benchmark(_cfg(), target_series=100, history=24, sizes=[3])
    status = dict(zip(table["model"], table["status"]))
    assert status == {"broken": "failed", "naive": "ok"}
    assert "library missing" in table.set_index("model").loc["broken", "note"]
    assert any("FAILED broken" in line for line in summary_lines(table, _cfg(), 100, 24))


def test_slow_model_is_stopped_after_the_small_sample(monkeypatch):
    import gts_forecaster.benchmark as bm

    monkeypatch.setattr(bm, "build_models_from_config", lambda *a, **k: {"naive": NaiveForecaster()})
    table = run_benchmark(_cfg(), target_series=100, history=24, sizes=[3, 30], max_seconds=0.0)
    assert table.iloc[0]["sizes"] == "3" and "stopped after 3 series" in table.iloc[0]["note"]


def test_too_short_history_is_reported_not_crashing(monkeypatch):
    import gts_forecaster.benchmark as bm
    from gts_forecaster.models.naive import SeasonalNaiveForecaster

    monkeypatch.setattr(bm, "build_models_from_config",
                        lambda *a, **k: {"seasonal_naive": SeasonalNaiveForecaster(season_length=12)})
    table = run_benchmark(_cfg(), target_series=100, history=8, sizes=[3])
    assert table.iloc[0]["status"] == "failed" and "too short" in table.iloc[0]["note"]


def test_command_line_writes_the_csv(tmp_path):
    import contextlib
    import io

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = main(["--frequency", "monthly", "--series", "500", "--sizes", "3", "6",
                     "--history", "30", "--out-dir", str(tmp_path)])
    assert code == 0
    files = list(tmp_path.glob("benchmark_monthly_*.csv"))
    assert len(files) == 1 and "naive" in set(pd.read_csv(files[0])["model"])
    assert "TOTAL fast models" in printed.getvalue()
