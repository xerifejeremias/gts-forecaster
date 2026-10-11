"""StatsForecast adapters, tested against a stand-in for the library.

The stand-in mimics the parts of the StatsForecast API the adapters use: the models take
`season_length` and `alias`, `StatsForecast(models, freq, n_jobs, fallback_model)` has `fit(df)` and
`predict(h)`, and the forecast has `unique_id`, `ds` and one column per model. The forecast of a
fake model is the mean of the last `season_length` values plus a constant offset.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.models.catalog import MODEL_TYPES, build_models, load_model_specs
from gts_forecaster.models.statsforecast_adapter import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    AutoThetaForecaster,
    MSTLForecaster,
)
from helpers import BASELINES, make_panel, make_uuid

STATS_TYPES = ("auto_ets", "auto_arima", "auto_theta", "mstl")


def _baseline_and_stats_specs():
    # only the models these tests are about, whichever other model libraries are installed: the
    # default entry of each type (named like the type), not the tuning versions (ets_*, theta_*)
    specs = load_model_specs(types=[*BASELINES, *STATS_TYPES])
    return {name: spec for name, spec in specs.items() if name == spec["type"]}


def _fake_library(unique_id_as_index=False, shift_dates=False, drop_series=None):
    class FakeModel:
        offset = 0.0

        def __init__(self, season_length=1, alias=None, **kwargs):
            self.season_length = season_length
            self.alias = alias or type(self).__name__
            self.kwargs = kwargs

        def forecast_values(self, y, h):
            m = self.season_length if isinstance(self.season_length, int) else self.season_length[0]
            return np.full(h, y[-m:].mean() + self.offset + self.kwargs.get("bonus", 0.0))

    class AutoETS(FakeModel):
        offset = 0.0

    class AutoARIMA(FakeModel):
        offset = 100.0

    class AutoTheta(FakeModel):
        offset = 200.0

    class MSTL(FakeModel):
        offset = 300.0

    class Naive(FakeModel):
        def __init__(self, alias=None):
            super().__init__(1, alias)

    class StatsForecast:
        created = []

        def __init__(self, models, freq=None, n_jobs=1, fallback_model=None):
            self.models, self.freq, self.n_jobs, self.fallback_model = models, freq, n_jobs, fallback_model
            StatsForecast.created.append(self)

        def fit(self, df):
            assert list(df.columns) == ["unique_id", "ds", "y"], df.columns
            self.df = df
            return self

        def predict(self, h):
            rows = []
            for uid, g in self.df.groupby("unique_id", sort=False):
                if uid == drop_series:
                    continue
                g = g.sort_values("ds")
                start = g["ds"].iloc[-1]
                dates = pd.date_range(start, periods=h + 1, freq=self.freq)[1:]
                if shift_dates:
                    dates = dates + pd.Timedelta(days=1)
                for model in self.models:
                    rows.append(pd.DataFrame({
                        "unique_id": uid, "ds": dates,
                        model.alias: model.forecast_values(g["y"].to_numpy(), h),
                    }))
            out = pd.concat(rows, ignore_index=True)
            return out.set_index("unique_id") if unique_id_as_index else out

    models_module = types.ModuleType("statsforecast.models")
    for cls in (AutoETS, AutoARIMA, AutoTheta, MSTL, Naive):
        setattr(models_module, cls.__name__, cls)
    package = types.ModuleType("statsforecast")
    package.StatsForecast = StatsForecast
    package.models = models_module
    return package, models_module, StatsForecast


def _install(monkeypatch, **kwargs):
    package, models_module, stats_forecast = _fake_library(**kwargs)
    monkeypatch.setitem(sys.modules, "statsforecast", package)
    monkeypatch.setitem(sys.modules, "statsforecast.models", models_module)
    return stats_forecast


def _train(n_series=2, n=30):
    df = make_panel(n_series, n, internal=True)
    df["is_filled"] = False
    return df


def _future(uids, last, h, freq="MS"):
    frames = []
    for uid in uids:
        ds = pd.date_range(last, periods=h + 1, freq=freq)[1:]
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "horizon_step": range(1, h + 1)}))
    return pd.concat(frames, ignore_index=True)


def test_adapter_forecasts_each_series_and_keeps_row_order(monkeypatch):
    _install(monkeypatch)
    train = _train()
    uids = [make_uuid(0), make_uuid(1)]
    model = AutoETSForecaster(season_length=3, freq="MS").fit(train)
    future = _future(uids, train["ds"].max(), 4).sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = model.predict(future)
    assert out["unique_id"].tolist() == future["unique_id"].tolist()
    assert out["ds"].tolist() == future["ds"].tolist()
    # y runs 1..30 in the first series and 2..60 in the second; the fake forecast is the mean of
    # the last 3 values
    expected = out["unique_id"].map({uids[0]: 29.0, uids[1]: 58.0})
    assert np.allclose(out["yhat"], expected)


def test_each_adapter_builds_its_own_library_model(monkeypatch):
    _install(monkeypatch)
    train = _train(1)
    future = _future([make_uuid(0)], train["ds"].max(), 2)
    expected = {AutoETSForecaster: 0.0, AutoARIMAForecaster: 100.0,
                AutoThetaForecaster: 200.0, MSTLForecaster: 300.0}
    for cls, offset in expected.items():
        out = cls(season_length=2, freq="MS").fit(train).predict(future)
        assert np.allclose(out["yhat"], 29.5 + offset), cls.__name__


def test_options_are_passed_on_and_override_defaults(monkeypatch):
    stats_forecast = _install(monkeypatch)
    train = _train(1)
    AutoETSForecaster(season_length=2, freq="MS", options={"bonus": 5.0}).fit(train)
    fitted = stats_forecast.created[-1]
    assert fitted.models[0].kwargs == {"bonus": 5.0}
    assert fitted.freq == "MS" and fitted.n_jobs == 1  # no process start-up cost by default
    assert fitted.fallback_model is not None
    AutoETSForecaster(season_length=2, freq="MS", fallback=False, n_jobs=4).fit(train)
    assert stats_forecast.created[-1].fallback_model is None
    assert stats_forecast.created[-1].n_jobs == 4
    MSTLForecaster(season_length=12, freq="MS").fit(train)
    assert stats_forecast.created[-1].models[0].season_length == [12]
    MSTLForecaster(season_length=7, freq="D", options={"season_length": [7, 365]}, min_length=10).fit(train)
    assert stats_forecast.created[-1].models[0].season_length == [7, 365]


def test_unique_id_as_index_is_handled(monkeypatch):
    _install(monkeypatch, unique_id_as_index=True)
    train = _train()
    out = AutoETSForecaster(season_length=1, freq="MS").fit(train).predict(
        _future([make_uuid(0), make_uuid(1)], train["ds"].max(), 3)
    )
    assert len(out) == 6
    assert np.allclose(out["yhat"], out["unique_id"].map({make_uuid(0): 30.0, make_uuid(1): 60.0}))


def test_shorter_horizon_than_the_library_returns_is_fine(monkeypatch):
    _install(monkeypatch)
    train = _train()
    model = AutoETSForecaster(season_length=1, freq="MS").fit(train)
    future = pd.concat([
        _future([make_uuid(0)], train["ds"].max(), 5),
        _future([make_uuid(1)], train["ds"].max(), 2),
    ], ignore_index=True)
    assert len(model.predict(future)) == 7


def test_dates_that_do_not_line_up_are_an_error(monkeypatch):
    _install(monkeypatch, shift_dates=True)
    train = _train(1)
    model = AutoETSForecaster(season_length=1, freq="MS").fit(train)
    with pytest.raises(ValueError, match="differ from the requested"):
        model.predict(_future([make_uuid(0)], train["ds"].max(), 3))


def test_a_series_without_forecast_is_an_error(monkeypatch):
    _install(monkeypatch, drop_series=make_uuid(1))
    train = _train()
    model = AutoETSForecaster(season_length=1, freq="MS").fit(train)
    with pytest.raises(ValueError, match="no forecast"):
        model.predict(_future([make_uuid(0), make_uuid(1)], train["ds"].max(), 3))


def test_missing_library_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "statsforecast", None)
    with pytest.raises(ImportError, match="extras stats"):
        AutoETSForecaster(season_length=1, freq="MS").fit(_train(1))


def test_min_train_length_and_seasonality_flags():
    assert AutoETSForecaster(season_length=1).min_train_length == 7  # the library's ETS search needs 7
    assert AutoETSForecaster(season_length=3).min_train_length == 7
    assert AutoThetaForecaster(season_length=1).min_train_length == 4
    assert AutoETSForecaster(season_length=12).min_train_length == 24
    assert AutoARIMAForecaster(season_length=1).min_train_length == 6
    assert AutoThetaForecaster(season_length=7).min_train_length == 14
    assert AutoETSForecaster(season_length=12, min_length=10).min_train_length == 10
    assert MSTLForecaster.requires_seasonality and not AutoETSForecaster.requires_seasonality


def test_too_short_series_is_refused_before_the_library_is_called(monkeypatch):
    _install(monkeypatch)
    with pytest.raises(ValueError, match="at least 24"):
        AutoETSForecaster(season_length=12, freq="MS").fit(_train(1, 10))


def test_clone_and_params_roundtrip():
    model = AutoARIMAForecaster(season_length=12, freq="MS", options={"d": 1}, n_jobs=2)
    clone = model.clone()
    assert clone is not model and clone.get_params() == model.get_params()
    assert set(model.get_params()) == {
        "season_length", "freq", "options", "n_jobs", "fallback", "min_length",
        "max_history_years",
    }


def test_catalog_registers_the_types():
    for name in STATS_TYPES:
        assert name in MODEL_TYPES
        specs = load_model_specs(types=[name])
        assert name in specs


def test_catalog_skips_models_whose_library_is_missing(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "statsforecast", None)
    built = build_models(_baseline_and_stats_specs(), 12, pandas_freq="MS")
    assert set(built) == set(BASELINES)
    assert "statsforecast" in caplog.text and "--extras stats" in caplog.text


def test_catalog_builds_stats_models_with_season_and_freq(monkeypatch):
    _install(monkeypatch)
    built = build_models(_baseline_and_stats_specs(), 12, pandas_freq="MS")
    assert set(built) == set(BASELINES) | set(STATS_TYPES)
    assert built["auto_ets"].season_length == 12 and built["auto_ets"].freq == "MS"
    yearly = build_models(_baseline_and_stats_specs(), 1, pandas_freq="YS")
    assert "mstl" not in yearly and "auto_ets" in yearly


def test_catalog_needs_the_pandas_frequency(monkeypatch):
    _install(monkeypatch)
    with pytest.raises(ConfigError, match="pandas frequency"):
        build_models(load_model_specs(types=["auto_ets"]), 12)


def test_catalog_leaves_out_slow_models_when_fast_only(monkeypatch):
    class SlowModel(MODEL_TYPES["naive"]):
        cost = "slow"

    monkeypatch.setitem(MODEL_TYPES, "slow_naive", SlowModel)
    specs = {"naive": {"type": "naive", "params": {}, "features": None},
             "slow": {"type": "slow_naive", "params": {}, "features": None}}
    assert set(build_models(specs, 12)) == {"naive"}
    assert set(build_models(specs, 12, fast_only=True)) == {"naive"}
    assert set(build_models(specs, 12, fast_only=False)) == {"naive", "slow"}


def test_fast_only_config_default_and_validation():
    assert load_config({"frequency": "monthly"}).fast_only is True
    assert load_config({"frequency": "monthly", "fast_only": False}).fast_only is False
    with pytest.raises(ConfigError, match="fast_only"):
        load_config({"frequency": "monthly", "fast_only": "no"})


def test_mstl_minimum_follows_the_longest_period():
    assert MSTLForecaster(season_length=7, options={"season_length": [7, 365]}).min_train_length == 730
    assert MSTLForecaster(season_length=12).min_train_length == 24
    with pytest.raises(ValueError, match="periods"):
        MSTLForecaster(season_length=7, options={"season_length": [7, 1]}).min_train_length


def test_arima_drops_seasonality_for_long_seasons(monkeypatch):
    stats_forecast = _install(monkeypatch)
    train = _train(1, 60)
    # weekly (52) is over the limit: non-seasonal ARIMA, so it also needs far less history
    AutoARIMAForecaster(season_length=52, freq="MS").fit(train)
    assert stats_forecast.created[-1].models[0].season_length == 1
    assert AutoARIMAForecaster(season_length=52).min_train_length == 6
    # monthly (12) and daily (7) keep their season
    AutoARIMAForecaster(season_length=12, freq="MS").fit(train)
    assert stats_forecast.created[-1].models[0].season_length == 12
    assert AutoARIMAForecaster(season_length=12).min_train_length == 24
    # options can still force a seasonal fit
    forced = AutoARIMAForecaster(season_length=52, freq="MS", options={"season_length": 52})
    assert forced.min_train_length == 104
    AutoARIMAForecaster(
        season_length=52, freq="MS", options={"season_length": 52}, min_length=10
    ).fit(train)
    assert stats_forecast.created[-1].models[0].season_length == 52


def test_max_history_years_fits_on_recent_data_only(monkeypatch):
    stats_forecast = _install(monkeypatch)
    train = make_panel(2, 3 * 365 + 50, pandas_freq="D", internal=True)
    train["is_filled"] = False
    future = _future([make_uuid(0), make_uuid(1)], train["ds"].max(), 3, freq="D")
    model = AutoARIMAForecaster(season_length=7, freq="D", max_history_years={"daily": 2})
    model.fit(train).predict(future)
    fitted = stats_forecast.created[-1].df
    assert fitted.groupby("unique_id").size().tolist() == [730, 730]
    # the most recent observations are kept, so the forecast starts from the same last date
    assert (fitted.groupby("unique_id")["ds"].max() == train.groupby("unique_id")["ds"].max()).all()
    # a frequency that is not listed uses the whole series
    AutoThetaForecaster(season_length=7, freq="D", max_history_years={"monthly": 1}).fit(train)
    assert len(stats_forecast.created[-1].df) == len(train)
    AutoThetaForecaster(season_length=7, freq="D").fit(train)
    assert len(stats_forecast.created[-1].df) == len(train)


@pytest.mark.parametrize("bad", [3, {"hourly": 3}, {"daily": 0}, {"daily": "3"}, {"monthly": 1}])
def test_bad_max_history_years_is_refused_at_start_up(bad):
    # {"monthly": 1} is 12 months, fewer than the 24 a seasonal monthly ARIMA needs
    with pytest.raises(ValueError, match="max_history_years"):
        AutoARIMAForecaster(season_length=12, freq="MS", max_history_years=bad)


def test_fallback_series_are_reported(monkeypatch, caplog):
    """The library swaps in the naive model silently; the adapter must say which series."""
    import logging

    _install(monkeypatch)
    train = _train()
    model = AutoETSForecaster(season_length=1, freq="MS")

    class Naive:  # what the library stores for a series that fell back
        pass

    class Ets:
        pass

    def fake_fit(self_sf, df):
        self_sf.uids = sorted(df["unique_id"].unique())
        self_sf.fitted_ = np.array([[Naive()], [Ets()]], dtype=object)
        return self_sf

    with caplog.at_level(logging.WARNING):
        model.fit(train)  # first fit with the normal fake to get the class
        cls = type(model.sf_)
        monkeypatch.setattr(cls, "fit", fake_fit)
        model.fit(train)
    assert model.fallback_ids_ == [make_uuid(0)]
    assert "naive forecast instead" in caplog.text


def test_arima_minimum_history_follows_the_season_length_in_options():
    # weekly profile (52) with a 13-week cycle in options: two cycles of 13, not of 52
    assert AutoARIMAForecaster(season_length=52, options={"season_length": 13}).min_train_length == 26
    assert AutoARIMAForecaster(season_length=12, options={"season_length": 1}).min_train_length == 6
    assert AutoARIMAForecaster(season_length=52, min_length=40).min_train_length == 40
