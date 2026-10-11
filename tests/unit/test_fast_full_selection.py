"""One backtest, two selections: selection_fast.json (fast models) and selection.json (all models).

fast_only on (default) -> forecast/train/report use selection_fast.json and fitted_models/<f>/fast/;
fast_only off -> selection.json and fitted_models/<f>/full/. Decided with the user 2026-10-06.
"""

import pytest

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gts_forecaster.config import load_config
from gts_forecaster.models.naive import DriftForecaster, NaiveForecaster, SeasonalNaiveForecaster
from helpers import static_table
from helpers import use_core_catalog

@pytest.fixture(autouse=True)
def _core_catalog(monkeypatch):
    use_core_catalog(monkeypatch)  # pipeline tests: no tuning variants of the tree models


N = 24


class SlowDrift(DriftForecaster):
    """Stands in for Prophet / Orbit: a slow model (perfect on a straight line)."""

    cost = "slow"


def _models(cfg, *args, **kwargs):
    models = {"naive": NaiveForecaster(), "seasonal_naive": SeasonalNaiveForecaster(season_length=12)}
    if not cfg.fast_only:
        models["slow_drift"] = SlowDrift()
    return models


def _patch(monkeypatch):
    import gts_forecaster.pipelines.backtest as bt
    import gts_forecaster.pipelines.forecast as fc
    import gts_forecaster.pipelines.train as tr

    for module in (bt, fc, tr):
        monkeypatch.setattr(module, "build_models_from_config", _models)


def _prepared(tmp_path, fast_only=True, **extra):
    cfg = load_config({
        "frequency": "monthly", "horizon": 3, "n_folds": 2, "top_n": 1,
        "registry_dir": tmp_path / "reg", "backtesting_dir": tmp_path / "bt",
        "fitted_models_dir": tmp_path / "fm", "fast_only": fast_only, **extra,
    })
    dates = pd.date_range("2022-01-01", periods=N, freq="MS")
    df = pd.concat([
        pd.DataFrame({"unique_id": "linear", "ds": dates, "y": np.arange(1.0, N + 1)}),
        pd.DataFrame({"unique_id": "const", "ds": dates, "y": 5.0}),
    ], ignore_index=True)
    df["is_filled"] = False
    return SimpleNamespace(
        data=df, config=cfg, cutoff=df["ds"].max(),
        fallback=pd.DataFrame(columns=["unique_id"]), static=static_table(df["unique_id"]),
    )


def _names(doc, uid):
    return [m["name"] for m in doc["series"][uid]["models"]]


def test_one_backtest_writes_both_selections(tmp_path, monkeypatch):
    _patch(monkeypatch)
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    out = run_backtest_pipeline(_prepared(tmp_path, fast_only=False))
    folder = tmp_path / "reg" / "monthly"
    full = json.loads((folder / "selection.json").read_text())
    fast = json.loads((folder / "selection_fast.json").read_text())
    assert _names(full, "linear") == ["slow_drift"]  # the slow model wins where it is best
    assert "slow_drift" not in _names(fast, "linear")  # the fast selection never names it
    assert full["meta"]["variant"] == "full" and fast["meta"]["variant"] == "fast"
    assert full["meta"]["includes_slow_models"] is True
    assert "slow_drift" in full["meta"]["models"] and "slow_drift" not in fast["meta"]["models"]
    assert (folder / "backtest_scores.csv").is_file() and (folder / "backtest_scores_fast.csv").is_file()
    assert set(out.selections) == {"fast", "full"}
    assert out.selection_path == folder / "selection.json"  # this run had fast_only off


def test_fast_backtest_gives_identical_selections(tmp_path, monkeypatch):
    _patch(monkeypatch)
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    run_backtest_pipeline(_prepared(tmp_path, fast_only=True))
    folder = tmp_path / "reg" / "monthly"
    full = json.loads((folder / "selection.json").read_text())
    fast = json.loads((folder / "selection_fast.json").read_text())
    assert full["series"] == fast["series"]
    assert full["meta"]["includes_slow_models"] is False


def test_forecast_uses_the_selection_matching_fast_only(tmp_path, monkeypatch):
    _patch(monkeypatch)
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline

    run_backtest_pipeline(_prepared(tmp_path, fast_only=False))  # score everything once
    # default (fast_only on): the slow model is not built and not needed
    fast = run_forecast_pipeline(_prepared(tmp_path, fast_only=True), output_dir=tmp_path / "out_fast")
    assert "slow_drift" not in fast.by_model.columns
    # fast_only off: the full selection, which uses the slow model for "linear"
    full = run_forecast_pipeline(_prepared(tmp_path, fast_only=False), output_dir=tmp_path / "out_full")
    linear = full.forecast[full.forecast["unique_id"] == "linear"]["forecast"].tolist()
    assert linear == [25.0, 26.0, 27.0]  # drift continues the line exactly


def test_train_and_frozen_forecast_keep_fast_and_full_apart(tmp_path, monkeypatch):
    _patch(monkeypatch)
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline
    from gts_forecaster.pipelines.train import run_train_pipeline

    run_backtest_pipeline(_prepared(tmp_path, fast_only=False))
    fast_out = run_train_pipeline(_prepared(tmp_path, fast_only=True))
    full_out = run_train_pipeline(_prepared(tmp_path, fast_only=False))
    assert fast_out.folder == tmp_path / "fm" / "monthly" / "fast"
    assert full_out.folder == tmp_path / "fm" / "monthly" / "full"
    assert "slow_drift" not in fast_out.trained and "slow_drift" in full_out.trained
    frozen = run_forecast_pipeline(
        _prepared(tmp_path, fast_only=True, forecast_mode="frozen_models"), output_dir=tmp_path / "o"
    )
    assert set(frozen.timings["status"]) == {"updated"}  # loaded from fast/, nothing refitted


def test_locked_entries_are_kept_per_file(tmp_path, monkeypatch):
    _patch(monkeypatch)
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    run_backtest_pipeline(_prepared(tmp_path, fast_only=False))
    path = tmp_path / "reg" / "monthly" / "selection_fast.json"
    doc = json.loads(path.read_text())
    doc["series"]["const"] = {"locked": True, "models": [{"name": "seasonal_naive", "weight": 1.0}]}
    path.write_text(json.dumps(doc))
    run_backtest_pipeline(_prepared(tmp_path, fast_only=False))
    fast = json.loads(path.read_text())
    full = json.loads((tmp_path / "reg" / "monthly" / "selection.json").read_text())
    assert _names(fast, "const") == ["seasonal_naive"] and fast["series"]["const"]["locked"]
    assert not full["series"]["const"]["locked"]  # the lock lives only in the file it was set in


def test_fast_backtest_leaves_an_earlier_full_selection_untouched(tmp_path, monkeypatch):
    """Decided 2026-10-06: a fast-only backtest must not overwrite a selection where Prophet /
    Orbit competed (it did not score them)."""
    _patch(monkeypatch)
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline

    folder = tmp_path / "reg" / "monthly"
    run_backtest_pipeline(_prepared(tmp_path, fast_only=False))
    full_before = (folder / "selection.json").read_bytes()
    scores_before = (folder / "backtest_scores.csv").read_bytes()
    fast_before = json.loads((folder / "selection_fast.json").read_text())

    out = run_backtest_pipeline(_prepared(tmp_path, fast_only=True))
    assert out.full_selection_kept
    assert (folder / "selection.json").read_bytes() == full_before
    assert (folder / "backtest_scores.csv").read_bytes() == scores_before
    fast_after = json.loads((folder / "selection_fast.json").read_text())
    assert fast_after["meta"]["created"] >= fast_before["meta"]["created"]
    assert fast_after["meta"]["includes_slow_models"] is False
    # the kept full selection still works for a forecast with the slow models switched on
    full = run_forecast_pipeline(_prepared(tmp_path, fast_only=False), output_dir=tmp_path / "o")
    assert "slow_drift" in full.by_model.columns


def test_fast_backtest_writes_selection_json_when_there_is_no_full_one(tmp_path, monkeypatch):
    _patch(monkeypatch)
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    folder = tmp_path / "reg" / "monthly"
    first = run_backtest_pipeline(_prepared(tmp_path, fast_only=True))
    assert not first.full_selection_kept and (folder / "selection.json").is_file()
    # an earlier fast-only selection.json holds no slow-model results, so it is refreshed
    (folder / "selection.json").write_text(
        (folder / "selection.json").read_text().replace('"top_n": 1', '"top_n": 1, "marker": 1')
    )
    second = run_backtest_pipeline(_prepared(tmp_path, fast_only=True))
    assert not second.full_selection_kept
    assert "marker" not in (folder / "selection.json").read_text()
