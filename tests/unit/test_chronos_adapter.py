"""Chronos adapters, tested against a stand-in for the chronos library and PyTorch.

The stand-in Chronos-2 forecasts the mean of the last three values of a series, plus 10 times the
future value of a known covariate called `x_promo` when one is given. The stand-in Chronos-Bolt
forecasts the last value plus the step number. Both record how they were called.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.models import chronos_adapter
from gts_forecaster.models.catalog import MODEL_TYPES, build_models, build_models_from_config, load_model_specs
from gts_forecaster.models.chronos_adapter import (
    ChronosBoltForecaster,
    Chronos2Forecaster,
    clear_pipeline_cache,
)
from helpers import BASELINES, make_panel, make_uuid

FOUNDATION = ("chronos2", "chronos2_small", "chronos_bolt_small", "chronos_bolt_tiny")


class _Calls:
    loaded = []
    predict_df = []
    quantiles = []


def _fake_chronos(shift_dates=False):
    class Chronos2Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, device_map=None):
            _Calls.loaded.append(("chronos2", model_id, device_map))
            return cls()

        def predict_df(self, df, future_df=None, prediction_length=1, quantile_levels=(0.5,),
                       id_column="item_id", timestamp_column="timestamp", target="target", freq=None):
            _Calls.predict_df.append(
                dict(df=df.copy(), future_df=None if future_df is None else future_df.copy(),
                     prediction_length=prediction_length, quantile_levels=list(quantile_levels),
                     id_column=id_column, timestamp_column=timestamp_column, target=target, freq=freq))
            frames = []
            for uid, g in df.groupby(id_column, sort=False):
                g = g.sort_values(timestamp_column)
                start = g[timestamp_column].iloc[-1]
                step = freq or pd.infer_freq(g[timestamp_column])
                dates = pd.date_range(start, periods=prediction_length + 1, freq=step)[1:]
                if shift_dates:
                    dates = dates + pd.Timedelta(days=1)
                base = g[target].to_numpy()[-3:].mean()
                values = np.full(prediction_length, base)
                if future_df is not None and "x_promo" in future_df.columns:
                    promo = future_df[future_df[id_column] == uid].sort_values(timestamp_column)["x_promo"].to_numpy()
                    values = values + 10.0 * promo
                frames.append(pd.DataFrame({id_column: uid, timestamp_column: dates,
                                            "target_name": target, "predictions": values,
                                            "0.5": values}))
            # unsorted on purpose: the adapter must not depend on the order
            return pd.concat(frames, ignore_index=True).iloc[::-1].reset_index(drop=True)

    class BaseChronosPipeline:
        @classmethod
        def from_pretrained(cls, model_id, device_map=None):
            _Calls.loaded.append(("bolt", model_id, device_map))
            return cls()

        # Same signature as chronos-forecasting 2.x: the first argument is `inputs` (not `context`).
        def predict_quantiles(self, inputs, prediction_length=None, quantile_levels=(0.1, 0.5, 0.9)):
            context = inputs
            _Calls.quantiles.append(dict(context=list(context), prediction_length=prediction_length,
                                         quantile_levels=list(quantile_levels)))
            out = np.zeros((len(context), prediction_length, len(quantile_levels)))
            for i, series in enumerate(context):
                out[i, :, 0] = np.asarray(series)[-1] + np.arange(1, prediction_length + 1)
            return out, out[:, :, 0]

    package = types.ModuleType("chronos")
    package.Chronos2Pipeline = Chronos2Pipeline
    package.BaseChronosPipeline = BaseChronosPipeline
    torch = types.ModuleType("torch")
    torch.float32 = np.float32
    torch.tensor = lambda values, dtype=None: np.asarray(values, dtype=dtype)
    return package, torch


def _install(monkeypatch, **kwargs):
    clear_pipeline_cache()
    _Calls.loaded.clear(); _Calls.predict_df.clear(); _Calls.quantiles.clear()
    package, torch = _fake_chronos(**kwargs)
    monkeypatch.setitem(sys.modules, "chronos", package)
    monkeypatch.setitem(sys.modules, "torch", torch)


def _train(n_series=2, n=30, extra=None):
    df = make_panel(n_series, n, internal=True)
    df["is_filled"] = False
    if extra:
        df = df.assign(**extra)
    return df


def _future(train, h, **columns):
    frames = []
    for uid, g in train.groupby("unique_id", sort=False):
        ds = pd.date_range(g["ds"].max(), periods=h + 1, freq="MS")[1:]
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "horizon_step": range(1, h + 1)}))
    out = pd.concat(frames, ignore_index=True)
    return out.assign(**columns) if columns else out


def test_chronos2_forecasts_each_series_and_keeps_the_row_order(monkeypatch):
    _install(monkeypatch)
    train = _train()
    model = Chronos2Forecaster(freq="MS").fit(train)
    future = _future(train, 3).sample(frac=1.0, random_state=2).reset_index(drop=True)
    out = model.predict(future)
    assert out["unique_id"].tolist() == future["unique_id"].tolist()
    assert out["ds"].tolist() == future["ds"].tolist()
    # series 1: y = 1..30, series 2: y = 2..60; the stand-in forecasts the mean of the last 3 values
    expected = out["unique_id"].map({make_uuid(0): 29.0, make_uuid(1): 58.0})
    assert np.allclose(out["yhat"], expected)


def test_chronos2_is_called_with_the_model_format_columns(monkeypatch):
    _install(monkeypatch)
    train = _train()
    Chronos2Forecaster(model_id="autogluon/chronos-2-small", freq="MS", device_map="cpu").fit(train).predict(
        _future(train, 4))
    assert _Calls.loaded == [("chronos2", "autogluon/chronos-2-small", "cpu")]
    call = _Calls.predict_df[0]
    assert (call["id_column"], call["timestamp_column"], call["target"]) == ("unique_id", "ds", "y")
    assert call["prediction_length"] == 4 and call["quantile_levels"] == [0.5]
    assert list(call["df"].columns) == ["unique_id", "ds", "y"] and call["future_df"] is None


def test_the_network_is_loaded_once_and_shared_between_clones(monkeypatch):
    _install(monkeypatch)
    train = _train()
    model = Chronos2Forecaster(freq="MS")
    model.fit(train)
    model.clone().fit(train)
    Chronos2Forecaster(freq="MS").fit(train)
    assert len(_Calls.loaded) == 1
    Chronos2Forecaster(model_id="autogluon/chronos-2-small", freq="MS").fit(train)
    assert len(_Calls.loaded) == 2


def test_max_context_keeps_the_most_recent_observations(monkeypatch):
    _install(monkeypatch)
    train = _train(1, 30)
    model = Chronos2Forecaster(freq="MS", max_context=10).fit(train)
    model.predict(_future(train, 2))
    context = _Calls.predict_df[0]["df"]
    assert len(context) == 10 and context["y"].tolist() == list(range(21, 31))


def test_known_columns_are_sent_as_future_covariates(monkeypatch):
    _install(monkeypatch)
    train = _train(2, 30, {"x_promo": 0.0})
    model = Chronos2Forecaster(freq="MS", known_columns=("x_promo",)).fit(train)
    future = _future(train, 2, x_promo=[0.0, 1.0, 0.0, 1.0])
    out = model.predict(future)
    call = _Calls.predict_df[0]
    assert "x_promo" in call["df"].columns and list(call["future_df"].columns) == ["unique_id", "ds", "x_promo"]
    base = out["unique_id"].map({make_uuid(0): 29.0, make_uuid(1): 58.0}).to_numpy()
    assert np.allclose(out["yhat"] - base, [0.0, 10.0, 0.0, 10.0])
    with pytest.raises(ValueError, match="missing known column"):
        model.predict(_future(train, 2))
    with pytest.raises(ValueError, match="same number of future steps"):
        model.predict(future.iloc[:3])


def test_past_covariates_are_the_explanatory_columns_that_are_not_known(monkeypatch):
    _install(monkeypatch)
    train = _train(2, 30, {"x_promo": 0.0, "x_temp": 1.0})
    kwargs = dict(freq="MS", known_columns=("x_promo",), explanatory_columns=("x_promo", "x_temp"))
    Chronos2Forecaster(use_past_covariates=True, **kwargs).fit(train).predict(_future(train, 2, x_promo=0.0))
    call = _Calls.predict_df[-1]
    assert list(call["df"].columns) == ["unique_id", "ds", "y", "x_promo", "x_temp"]
    assert list(call["future_df"].columns) == ["unique_id", "ds", "x_promo"]  # x_temp: past only
    Chronos2Forecaster(use_past_covariates=False, **kwargs).fit(train).predict(_future(train, 2, x_promo=0.0))
    assert list(_Calls.predict_df[-1]["df"].columns) == ["unique_id", "ds", "y", "x_promo"]


def test_only_requested_series_are_sent(monkeypatch):
    _install(monkeypatch)
    train = _train()
    model = Chronos2Forecaster(freq="MS").fit(train)
    future = _future(train, 2)
    out = model.predict(future[future["unique_id"] == make_uuid(1)])
    assert set(out["unique_id"]) == {make_uuid(1)}
    assert set(_Calls.predict_df[0]["df"]["unique_id"]) == {make_uuid(1)}


def test_misaligned_dates_are_an_error(monkeypatch):
    _install(monkeypatch, shift_dates=True)
    train = _train(1)
    model = Chronos2Forecaster(freq="MS").fit(train)
    with pytest.raises(ValueError, match="differ from the requested"):
        model.predict(_future(train, 2))


def test_bolt_forecasts_the_median_and_ignores_covariates(monkeypatch):
    _install(monkeypatch)
    train = _train(2, 30, {"x_promo": 1.0})
    model = ChronosBoltForecaster(freq="MS").fit(train)
    out = model.predict(_future(train, 3))
    assert _Calls.loaded == [("bolt", "amazon/chronos-bolt-small", "cpu")]
    call = _Calls.quantiles[0]
    assert call["prediction_length"] == 3 and call["quantile_levels"] == [0.5]
    assert [len(c) for c in call["context"]] == [30, 30]
    expected = [31.0, 32.0, 33.0, 61.0, 62.0, 63.0]  # last value + step
    assert np.allclose(out["yhat"], expected)
    assert out["ds"].tolist() == _future(train, 3)["ds"].tolist()


def test_bolt_has_no_covariate_parameters():
    assert "known_columns" not in ChronosBoltForecaster().get_params()
    assert "known_columns" in Chronos2Forecaster().get_params()


def test_missing_library_gives_install_hint(monkeypatch):
    clear_pipeline_cache()
    monkeypatch.setitem(sys.modules, "chronos", None)
    with pytest.raises(ImportError, match="extras foundation"):
        Chronos2Forecaster(freq="MS").fit(_train(1))


def test_settings_are_validated_and_min_length_applies():
    with pytest.raises(ValueError, match="max_context"):
        Chronos2Forecaster(max_context=0)
    with pytest.raises(ValueError, match="min_length"):
        ChronosBoltForecaster(min_length=True)
    assert Chronos2Forecaster().min_train_length == 5
    assert Chronos2Forecaster(min_length=12).min_train_length == 12
    assert Chronos2Forecaster().model_id == "amazon/chronos-2"
    assert ChronosBoltForecaster().model_id == "amazon/chronos-bolt-small"


def test_clone_keeps_parameters():
    model = Chronos2Forecaster(model_id="autogluon/chronos-2-small", freq="D", max_context=64,
                               known_columns=("x_promo",), use_past_covariates=True)
    clone = model.clone()
    assert clone is not model and clone.get_params() == model.get_params()
    assert clone.model_id == "autogluon/chronos-2-small"


# ---- catalog -------------------------------------------------------------------------------


def test_catalog_has_the_foundation_entries():
    specs = load_model_specs()
    for name in FOUNDATION:
        assert name in specs
    assert specs["chronos2_small"]["params"] == {"model_id": "autogluon/chronos-2-small"}
    assert specs["chronos_bolt_tiny"]["params"] == {"model_id": "amazon/chronos-bolt-tiny"}
    assert MODEL_TYPES["chronos2"] is Chronos2Forecaster


def test_catalog_skips_foundation_models_when_the_library_is_missing(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "chronos", None)
    built = build_models(load_model_specs(types=["chronos2", "chronos_bolt"]), 12, pandas_freq="MS")
    assert built == {}
    assert "--extras foundation" in caplog.text


def test_catalog_passes_known_and_explanatory_columns(monkeypatch):
    _install(monkeypatch)
    built = build_models(load_model_specs(types=["chronos2", "chronos_bolt"]), 12, pandas_freq="MS",
                         known_columns=["x_promo"], explanatory_columns=["x_promo", "x_temp"])
    assert set(built) == set(FOUNDATION)
    assert built["chronos2"].known_columns == ("x_promo",)
    assert built["chronos2"].explanatory_columns == ("x_promo", "x_temp")
    assert built["chronos2_small"].model_id == "autogluon/chronos-2-small"
    assert built["chronos_bolt_tiny"].model_id == "amazon/chronos-bolt-tiny"
    assert built["chronos2"].freq == "MS"


def test_catalog_builds_from_a_config_with_known_columns(monkeypatch, tmp_path):
    _install(monkeypatch)
    for name in ("default.yaml", "frequency_profiles.yaml", "features.yaml"):
        (tmp_path / name).write_text(open(f"config/{name}").read())
    (tmp_path / "models.yaml").write_text("c:\n  type: chronos2\n")
    cfg = load_config({"frequency": "monthly", "horizon": 3, "known_future_columns": ["x_promo"]},
                      config_dir=tmp_path)
    model = build_models_from_config(cfg, tmp_path)["c"]
    assert model.known_columns == ("x_promo",)
    assert model.explanatory_columns == cfg.explanatory_columns


def test_chronos2_passes_the_configured_frequency(monkeypatch):
    """Without freq, Chronos-2 infers it per series and fails on any series with a gap."""
    _install(monkeypatch)
    train = _train()
    Chronos2Forecaster(freq="MS").fit(train).predict(_future(train, 2))
    assert _Calls.predict_df[-1]["freq"] == "MS"
