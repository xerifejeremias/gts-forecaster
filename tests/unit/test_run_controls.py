"""Run controls: model filter, error isolation, timings, series sampling, partial runs."""

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.data_preparation import PreparedInput, limit_series
from gts_forecaster.evaluation.backtest import run_backtest
from gts_forecaster.forecasting import ForecastError, combine_forecasts, forecast_by_model
from gts_forecaster.models.catalog import build_models, filter_specs, load_model_specs
from gts_forecaster.models.naive import NaiveForecaster
from gts_forecaster.selection import select_models
from gts_forecaster.splitting import forecast_split
from helpers import BASELINES

N = 24


class Boom(NaiveForecaster):
    """A model that always crashes when fitted."""

    def _fit(self, series, train):
        raise RuntimeError("kaput")


def _panel(lengths=(N, N, N)):
    frames = []
    for k, n in enumerate(lengths):
        dates = pd.date_range("2022-01-01", periods=N, freq="MS")[-n:]
        frames.append(pd.DataFrame({"unique_id": f"s{k}", "ds": dates, "y": np.arange(1.0, n + 1) + k}))
    df = pd.concat(frames, ignore_index=True)
    df["is_filled"] = False
    return df


def _models(with_boom=True):
    models = build_models(load_model_specs(types=BASELINES), 12)
    if with_boom:
        models["boom"] = Boom()
    return models


def _backtest(models, **kw):
    return run_backtest(
        _panel(), models, horizon=3, pandas_freq="MS", n_folds=2, metric="mae",
        season_length=12, **kw,
    )


# ---- model filter ----------------------------------------------------------------------------

def _specs():
    return {n: {"type": n} for n in ("naive", "seasonal_naive", "drift", "lightgbm", "xgboost")}


def test_include_keeps_listed_models_and_protected_ones():
    kept = filter_specs(_specs(), ["lightgbm"], [], {"naive", "seasonal_naive"})
    assert set(kept) == {"lightgbm", "naive", "seasonal_naive"}


def test_exclude_removes_models():
    kept = filter_specs(_specs(), None, ["xgboost"], {"naive"})
    assert "xgboost" not in kept and "lightgbm" in kept


def test_filter_rejects_unknown_names_and_excluding_protected():
    with pytest.raises(ConfigError, match="nope"):
        filter_specs(_specs(), ["nope"], [], {"naive"})
    with pytest.raises(ConfigError, match="naive"):
        filter_specs(_specs(), None, ["naive"], {"naive"})


def test_no_filter_returns_everything():
    assert set(filter_specs(_specs(), None, [], {"naive"})) == set(_specs())


# ---- config ----------------------------------------------------------------------------------

def test_run_control_defaults():
    cfg = load_config()
    assert cfg.models is None and cfg.exclude_models == ()
    assert cfg.max_series is None and cfg.on_model_error == "skip"
    assert not cfg.is_partial_run


def test_partial_run_flags():
    assert load_config({"max_series": 5}).is_partial_run
    assert load_config({"models": ["lightgbm"]}).is_partial_run
    assert load_config({"exclude_models": ["xgboost"]}).is_partial_run


@pytest.mark.parametrize(
    "override",
    [{"max_series": 0}, {"max_series": 1.5}, {"on_model_error": "ignore"}, {"models": "naive"}],
)
def test_invalid_run_controls_are_rejected(override):
    with pytest.raises(ConfigError):
        load_config(override)


# ---- backtest error isolation and timings ----------------------------------------------------

def test_failing_model_is_skipped_and_reported():
    r = _backtest(_models())
    assert "boom" not in set(r.scores["model"])
    assert {"naive", "drift"} <= set(r.scores["model"])
    assert set(r.failures["model"]) == {"boom"}
    assert len(r.failures) == 2  # one per fold
    assert "RuntimeError: kaput" in r.failures["error"].iloc[0]


def test_failing_model_raises_when_asked():
    with pytest.raises(RuntimeError, match="kaput"):
        _backtest(_models(), on_error="raise")


def test_invalid_on_error_is_rejected():
    with pytest.raises(ValueError, match="on_error"):
        _backtest(_models(), on_error="ignore")


def test_timings_have_one_row_per_fold_and_model():
    r = _backtest(_models())
    assert len(r.timings) == 2 * 4
    assert set(r.timings["status"]) == {"ok", "failed"}
    ok = r.timings[r.timings["status"] == "ok"]
    assert (ok["n_series"] == 3).all() and (ok["fit_seconds"] >= 0).all()


def test_selection_still_works_when_a_model_failed():
    r = _backtest(_models())
    sel = select_models(r, _models(), top_n=2, ensemble_weights="equal")
    used = {n for e in sel.series.values() for n, _ in e.models}
    assert "boom" not in used


def test_no_failures_gives_empty_failure_table():
    r = _backtest(_models(with_boom=False))
    assert r.failures.empty and set(r.timings["status"]) == {"ok"}


# ---- forecast side ---------------------------------------------------------------------------

def _split():
    return forecast_split(_panel(), 3, "MS")


def test_forecast_by_model_isolates_failure():
    failures, timings = [], []
    wide = forecast_by_model(_split(), _models(), on_error="skip", failures=failures, timings=timings)
    assert wide["boom"].isna().all() and wide["naive"].notna().all()
    assert failures[0]["model"] == "boom"
    assert {t["model"]: t["status"] for t in timings}["boom"] == "failed"


def test_forecast_by_model_raises_by_default():
    with pytest.raises(RuntimeError):
        forecast_by_model(_split(), _models())


def _sel(models):
    return {"series": {f"s{k}": {"locked": False, "models": models} for k in range(3)}}


def test_combine_without_drop_missing_errors_on_failed_model():
    wide = forecast_by_model(_split(), _models(), on_error="skip")
    sel = _sel([{"name": "naive", "weight": 0.5}, {"name": "boom", "weight": 0.5}])
    with pytest.raises(ForecastError, match="boom"):
        combine_forecasts(wide, sel)


def test_combine_drop_missing_rescales_weights(caplog):
    wide = forecast_by_model(_split(), _models(), on_error="skip")
    sel = _sel([{"name": "naive", "weight": 0.5}, {"name": "boom", "weight": 0.5}])
    out = combine_forecasts(wide, sel, drop_missing=True)
    expected = wide.groupby("unique_id")["naive"].first()
    got = out.groupby("unique_id")["forecast"].first()
    assert np.allclose(got.to_numpy(), expected.to_numpy())


def test_combine_drop_missing_errors_when_nothing_left():
    wide = forecast_by_model(_split(), _models(), on_error="skip")
    with pytest.raises(ForecastError, match="none of the selected"):
        combine_forecasts(wide, _sel([{"name": "boom", "weight": 1.0}]), drop_missing=True)


# ---- limit_series ----------------------------------------------------------------------------

def _prepared(lengths):
    data = _panel(lengths)
    static = pd.DataFrame({"unique_id": [f"s{k}" for k in range(len(lengths))], "market": "DE04"})
    fallback = pd.DataFrame({"unique_id": ["s0", "x"], "fallback_value": [1.0, 2.0]})
    return PreparedInput(data, fallback, pd.DataFrame(), pd.Timestamp("2023-12-01"),
                         load_config(), static)


def test_limit_series_none_or_large_returns_same_object():
    p = _prepared((N, N, N))
    assert limit_series(p, None) is p
    assert limit_series(p, 10) is p


def test_limit_series_keeps_shortest_and_longest_and_filters_tables():
    p = _prepared((10, 24, 16, 20, 12))
    q = limit_series(p, 3)
    kept = set(q.data["unique_id"])
    assert len(kept) == 3
    assert "s0" in kept and "s1" in kept  # shortest (10) and longest (24)
    assert set(q.static["unique_id"]) == kept
    assert set(q.fallback["unique_id"]) <= kept
    assert len(p.data["unique_id"].unique()) == 5  # original untouched


def test_limit_series_is_deterministic_and_handles_one():
    p = _prepared((10, 24, 16, 20, 12))
    assert set(limit_series(p, 3).data["unique_id"]) == set(limit_series(p, 3).data["unique_id"])
    assert limit_series(p, 1).data["unique_id"].nunique() == 1


def test_limit_series_rejects_zero():
    with pytest.raises(ValueError):
        limit_series(_prepared((N, N)), 0)
