"""Prophet and Orbit adapters, tested against stand-ins for the libraries.

The stand-in Prophet forecasts the mean of its training values plus 10 times the value of each added
regressor. The stand-in Orbit models return the last training value plus the step number, as a
DataFrame with the date column and `prediction`. Both can be told to fail for a given series.
"""

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.models.bayesian_adapters import (
    OrbitDLTForecaster,
    OrbitLGTForecaster,
    ProphetForecaster,
)
from gts_forecaster.models.catalog import MODEL_TYPES, build_models, build_models_from_config, load_model_specs
from helpers import BASELINES, make_panel, make_uuid

SLOW = ("prophet", "orbit_dlt", "orbit_lgt")


class _Calls:
    prophet_args = []
    orbit_args = []
    regressors = []


def _fake_prophet(fail_when_mean_above=None):
    class Prophet:
        def __init__(self, **kwargs):
            _Calls.prophet_args.append(kwargs)
            self.regressors = []

        def add_regressor(self, name):
            self.regressors.append(name)

        def fit(self, df):
            assert list(df.columns)[:2] == ["ds", "y"]
            assert list(df.columns)[2:] == self.regressors
            if fail_when_mean_above is not None and df["y"].mean() > fail_when_mean_above:
                raise RuntimeError("stan failed")
            self.mean = df["y"].mean()
            return self

        def predict(self, df):
            values = np.full(len(df), self.mean)
            for name in self.regressors:
                values = values + 10.0 * df[name].to_numpy()
            return pd.DataFrame({"ds": df["ds"], "yhat": values, "yhat_lower": values - 1})

    package = types.ModuleType("prophet")
    package.Prophet = Prophet
    return package


def _fake_orbit():
    class _Orbit:
        def __init__(self, **kwargs):
            _Calls.orbit_args.append((type(self).__name__, kwargs))
            self.kwargs = kwargs

        def fit(self, df):
            assert self.kwargs["response_col"] == "y" and self.kwargs["date_col"] == "ds"
            self.last = df["y"].iloc[-1]
            return self

        def predict(self, df):
            regressors = self.kwargs["regressor_col"] or []
            assert all(c in df.columns for c in regressors)
            steps = df["ds"].rank(method="first").to_numpy()  # the step follows from the date
            # returned in reverse order on purpose: the adapter must match by date
            out = pd.DataFrame({"ds": df["ds"], "prediction": self.last + steps})
            return out.iloc[::-1].reset_index(drop=True)

    class DLT(_Orbit):
        pass

    class LGT(_Orbit):
        pass

    models = types.ModuleType("orbit.models")
    models.DLT, models.LGT = DLT, LGT
    package = types.ModuleType("orbit")
    package.models = models
    return package, models


def _install(monkeypatch, **kwargs):
    _Calls.prophet_args.clear(); _Calls.orbit_args.clear()
    monkeypatch.setitem(sys.modules, "prophet", _fake_prophet(**kwargs))
    package, models = _fake_orbit()
    monkeypatch.setitem(sys.modules, "orbit", package)
    monkeypatch.setitem(sys.modules, "orbit.models", models)


def _train(n_series=2, n=30, **columns):
    df = make_panel(n_series, n, internal=True)
    df["is_filled"] = False
    return df.assign(**columns) if columns else df


def _future(train, h, **columns):
    frames = []
    for uid, g in train.groupby("unique_id", sort=False):
        ds = pd.date_range(g["ds"].max(), periods=h + 1, freq="MS")[1:]
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "horizon_step": range(1, h + 1)}))
    out = pd.concat(frames, ignore_index=True)
    return out.assign(**columns) if columns else out


def test_these_models_are_slow_per_series_models():
    for cls in (ProphetForecaster, OrbitDLTForecaster, OrbitLGTForecaster):
        assert cls.cost == "slow" and cls.scope == "local"
    assert ProphetForecaster.install_extra == "prophet" and OrbitDLTForecaster.install_extra == "orbit"


def test_prophet_forecasts_each_series_and_keeps_the_row_order(monkeypatch):
    _install(monkeypatch)
    train = _train()
    model = ProphetForecaster(season_length=12, freq="MS").fit(train)
    future = _future(train, 3).sample(frac=1.0, random_state=5).reset_index(drop=True)
    out = model.predict(future)
    assert out["unique_id"].tolist() == future["unique_id"].tolist()
    assert out["ds"].tolist() == future["ds"].tolist()
    expected = out["unique_id"].map({make_uuid(0): 15.5, make_uuid(1): 31.0})  # mean of 1..30, 2..60
    assert np.allclose(out["yhat"], expected)


def test_prophet_seasonality_follows_the_frequency_and_options_override(monkeypatch):
    _install(monkeypatch)
    train = _train(1)
    ProphetForecaster(freq="MS").fit(train)
    assert _Calls.prophet_args[-1] == dict(yearly_seasonality="auto", weekly_seasonality=False, daily_seasonality=False)
    ProphetForecaster(freq="YS").fit(train)
    assert _Calls.prophet_args[-1]["yearly_seasonality"] is False
    ProphetForecaster(freq="D").fit(train)
    assert _Calls.prophet_args[-1]["weekly_seasonality"] == "auto"
    ProphetForecaster(freq="MS", options={"yearly_seasonality": False, "seasonality_mode": "multiplicative"}).fit(train)
    assert _Calls.prophet_args[-1]["yearly_seasonality"] is False
    assert _Calls.prophet_args[-1]["seasonality_mode"] == "multiplicative"


def test_prophet_known_columns_become_regressors(monkeypatch):
    _install(monkeypatch)
    train = _train(1, x_promo=0.0)
    model = ProphetForecaster(freq="MS", known_columns=("x_promo",)).fit(train)
    out = model.predict(_future(train, 2, x_promo=[0.0, 1.0]))
    assert np.allclose(out["yhat"], [15.5, 25.5])
    with pytest.raises(ValueError, match="missing known column"):
        model.predict(_future(train, 2))
    with pytest.raises(ValueError, match="not in the training data"):
        ProphetForecaster(freq="MS", known_columns=("nope",)).fit(train)


@pytest.mark.parametrize("cls,name", [(OrbitDLTForecaster, "DLT"), (OrbitLGTForecaster, "LGT")])
def test_orbit_models_forecast_by_date(monkeypatch, cls, name):
    _install(monkeypatch)
    train = _train()
    model = cls(season_length=12, freq="MS").fit(train)
    future = _future(train, 3).sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = model.predict(future)
    assert out["unique_id"].tolist() == future["unique_id"].tolist()
    last = out["unique_id"].map({make_uuid(0): 30.0, make_uuid(1): 60.0})
    assert np.allclose(out["yhat"], last + future["horizon_step"])
    kind, kwargs = _Calls.orbit_args[0]
    assert kind == name
    assert kwargs["seasonality"] == 12 and kwargs["estimator"] == "stan-map"
    assert kwargs["n_bootstrap_draws"] == -1 and kwargs["regressor_col"] is None and kwargs["seed"] == 8888


def test_orbit_without_a_season_and_with_options_and_regressors(monkeypatch):
    _install(monkeypatch)
    train = _train(1, x_promo=0.0)
    OrbitDLTForecaster(season_length=1, freq="YS", known_columns=("x_promo",),
                       options={"seed": 1, "estimator": "stan-mcmc"}).fit(train)
    kwargs = _Calls.orbit_args[-1][1]
    assert kwargs["seasonality"] == -1 and kwargs["regressor_col"] == ["x_promo"]
    assert kwargs["seed"] == 1 and kwargs["estimator"] == "stan-mcmc"


def test_a_failing_series_falls_back_to_its_last_value_with_a_warning(monkeypatch, caplog):
    _install(monkeypatch, fail_when_mean_above=20.0)  # series 2 has mean 31
    train = _train()
    model = ProphetForecaster(freq="MS").fit(train)
    out = model.predict(_future(train, 2))
    by_id = out.groupby("unique_id")["yhat"].apply(list).to_dict()
    assert np.allclose(by_id[make_uuid(0)], [15.5, 15.5])
    assert by_id[make_uuid(1)] == [60.0, 60.0]
    assert make_uuid(1) in caplog.text and "last value" in caplog.text


def test_without_fallback_the_error_is_raised(monkeypatch):
    _install(monkeypatch, fail_when_mean_above=20.0)
    with pytest.raises(RuntimeError, match="stan failed"):
        ProphetForecaster(freq="MS", fallback=False).fit(_train())


def test_a_failing_prediction_falls_back_too(monkeypatch, caplog):
    _install(monkeypatch)
    train = _train(1)
    model = OrbitDLTForecaster(freq="MS").fit(train)

    def broken(df):
        raise RuntimeError("no draws")

    model.models_[make_uuid(0)].predict = broken
    out = model.predict(_future(train, 2))
    assert out["yhat"].tolist() == [30.0, 30.0] and "failed to predict" in caplog.text


def test_missing_library_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "prophet", None)
    monkeypatch.setitem(sys.modules, "orbit.models", None)
    monkeypatch.setitem(sys.modules, "orbit", None)
    with pytest.raises(ImportError, match="extras prophet"):
        ProphetForecaster(freq="MS").fit(_train(1))
    with pytest.raises(ImportError, match="extras orbit"):
        OrbitDLTForecaster(freq="MS").fit(_train(1))


def test_min_train_length():
    assert ProphetForecaster(season_length=1).min_train_length == 10
    assert OrbitDLTForecaster(season_length=12).min_train_length == 24
    assert ProphetForecaster(season_length=12, min_length=6).min_train_length == 6
    with pytest.raises(ValueError, match="season_length"):
        ProphetForecaster(season_length=0)


def test_clone_keeps_parameters():
    model = OrbitLGTForecaster(season_length=12, freq="MS", known_columns=("x",), options={"seed": 3},
                               fallback=False, min_length=20)
    clone = model.clone()
    assert clone is not model and clone.get_params() == model.get_params()


# ---- catalog and the fast_only switch ----------------------------------------------------------


def test_catalog_entries_exist_and_are_registered():
    specs = load_model_specs()
    for name in SLOW:
        assert name in specs and name in MODEL_TYPES


def test_slow_models_are_left_out_by_default(monkeypatch):
    _install(monkeypatch)
    specs = load_model_specs(types=SLOW)
    assert build_models(specs, 12, pandas_freq="MS") == {}
    assert build_models(specs, 12, pandas_freq="MS", fast_only=True) == {}
    built = build_models(specs, 12, pandas_freq="MS", fast_only=False, known_columns=["x_promo"])
    assert set(built) == set(SLOW)
    assert built["prophet"].known_columns == ("x_promo",) and built["orbit_dlt"].season_length == 12


def test_slow_models_follow_the_fast_only_config(monkeypatch, tmp_path):
    _install(monkeypatch)
    for name in ("default.yaml", "frequency_profiles.yaml", "features.yaml"):
        (tmp_path / name).write_text(open(f"config/{name}").read())
    (tmp_path / "models.yaml").write_text(
        "naive:\n  type: naive\nprophet:\n  type: prophet\norbit_dlt:\n  type: orbit_dlt\n"
    )
    fast = load_config({"frequency": "monthly", "horizon": 3}, config_dir=tmp_path)
    assert set(build_models_from_config(fast, tmp_path)) == {"naive"}
    full = load_config({"frequency": "monthly", "horizon": 3, "fast_only": False}, config_dir=tmp_path)
    assert set(build_models_from_config(full, tmp_path)) == {"naive", "prophet", "orbit_dlt"}


def test_slow_models_missing_a_library_are_skipped_with_a_warning_when_enabled(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "prophet", None)
    monkeypatch.setitem(sys.modules, "orbit", None)
    built = build_models(load_model_specs(types=SLOW), 12, pandas_freq="MS", fast_only=False)
    assert built == {}
    assert "--extras prophet" in caplog.text and "--extras orbit" in caplog.text


def test_the_command_line_switch_reaches_the_config():
    from gts_forecaster.pipelines.backtest import main  # noqa: F401  (import check only)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-only", action=argparse.BooleanOptionalAction, default=None)
    assert parser.parse_args(["--no-fast-only"]).fast_only is False
    assert parser.parse_args([]).fast_only is None
    assert load_config({"frequency": "monthly", "fast_only": parser.parse_args(["--no-fast-only"]).fast_only}).fast_only is False


def test_failing_on_every_series_stops_instead_of_faking_a_forecast(monkeypatch):
    """E.g. CmdStan missing: every series fails, which is a setup problem, not a data problem."""
    _install(monkeypatch, fail_when_mean_above=-1.0)  # every series fails
    with pytest.raises(RuntimeError, match="failed on all 2 series.*stan failed"):
        ProphetForecaster(season_length=12, freq="MS").fit(_train())
