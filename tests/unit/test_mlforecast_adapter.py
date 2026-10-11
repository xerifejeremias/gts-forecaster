"""MLForecast adapters, tested against a stand-in for mlforecast and the boosting libraries.

The stand-in MLForecast builds lag features, joins the dynamic (calendar / known) and static
columns, trains the regressor it was given (real scikit-learn estimators) and forecasts recursively.
It ignores rolling windows, date features and target transforms but records them, so the tests check
that the adapter passes the right things on and uses the results correctly.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.features.builder import FeatureBuilder, FeatureSet, load_feature_config
from gts_forecaster.features.builder import build_feature_builders
from gts_forecaster.features.static import StaticSpec
from gts_forecaster.models.catalog import MODEL_TYPES, build_models, build_models_from_config, load_model_specs
from gts_forecaster.models.mlforecast_adapter import (
    LassoForecaster,
    LightGBMForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)
from helpers import BASELINES, make_panel, make_uuid

ML_TYPES = ("lightgbm", "xgboost", "random_forest", "ridge", "lasso")


class _Recorder:
    last = None


def _fake_mlforecast(missing_exog_check=True):
    class RollingMean:
        def __init__(self, window_size):
            self.window_size = window_size

    class RollingStd(RollingMean):
        pass

    class RollingMin(RollingMean):
        pass

    class RollingMax(RollingMean):
        pass

    class _Scaler:
        def __init__(self, *args):
            self.args = args

    class LocalStandardScaler(_Scaler):
        pass

    class LocalMinMaxScaler(_Scaler):
        pass

    class Differences(_Scaler):
        pass

    class MLForecast:
        def __init__(self, models, freq, lags, lag_transforms=None, date_features=None,
                     target_transforms=None, num_threads=1):
            self.models, self.freq, self.lags = models, freq, lags
            self.lag_transforms, self.date_features = lag_transforms, date_features
            self.target_transforms, self.num_threads = target_transforms, num_threads
            _Recorder.last = self

        def fit(self, df, id_col, time_col, target_col, static_features=None, dropna=True):
            assert (id_col, time_col, target_col) == ("unique_id", "ds", "y")
            self.static = list(static_features) if static_features is not None else None
            self.fit_columns = list(df.columns)
            self.dynamic = [c for c in df.columns if c not in (id_col, time_col, target_col, *self.static)]
            self.history = {}
            rows, ys = [], []
            for uid, g in df.sort_values([id_col, time_col]).groupby(id_col, sort=False):
                y = g[target_col].to_numpy(float)
                self.history[uid] = (g[time_col].iloc[-1], y, g[self.static].iloc[-1] if self.static else None)
                for t in range(max(self.lags), len(y)):
                    row = [y[t - lag] for lag in self.lags]
                    row += list(g[self.dynamic].iloc[t]) + (list(g[self.static].iloc[t]) if self.static else [])
                    rows.append(row)
                    ys.append(y[t])
            assert rows, "no training rows"
            self.estimator = next(iter(self.models.values()))
            self.estimator.fit(np.array(rows, float), np.array(ys))
            return self

        def predict(self, h, X_df=None, ids=None, new_df=None):
            if self.dynamic and X_df is None:
                raise ValueError("X_df is required")
            history = self.history
            if new_df is not None:  # newer history, same fitted estimator
                history = {}
                for uid, g in new_df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
                    history[uid] = (
                        g["ds"].iloc[-1], g["y"].to_numpy(float),
                        g[self.static].iloc[-1] if self.static else None,
                    )
            out = []
            for uid in ids if ids is not None else list(history):
                last, y, static = history[uid]
                dates = pd.date_range(last, periods=h + 1, freq=self.freq)[1:]
                values = list(y)
                for step, ds in enumerate(dates):
                    row = [values[-lag] for lag in self.lags]
                    if self.dynamic:
                        x = X_df[(X_df["unique_id"] == uid) & (X_df["ds"] == ds)]
                        if len(x) != 1:
                            raise ValueError(f"X_df has no row for {uid} {ds}")
                        row += list(x[self.dynamic].iloc[0])
                    if self.static:
                        row += list(static)
                    values.append(float(self.estimator.predict(np.array([row], float))[0]))
                out.append(pd.DataFrame({"unique_id": uid, "ds": dates, "gts_model": values[len(y):]}))
            return pd.concat(out, ignore_index=True)

    lag_module = types.ModuleType("mlforecast.lag_transforms")
    for cls in (RollingMean, RollingStd, RollingMin, RollingMax):
        setattr(lag_module, cls.__name__, cls)
    target_module = types.ModuleType("mlforecast.target_transforms")
    for cls in (LocalStandardScaler, LocalMinMaxScaler, Differences):
        setattr(target_module, cls.__name__, cls)
    package = types.ModuleType("mlforecast")
    package.MLForecast = MLForecast
    package.lag_transforms = lag_module
    package.target_transforms = target_module
    return package, lag_module, target_module


def _fake_boosting():
    """Stand-ins for lightgbm and xgboost: a small scikit-learn forest (accepts NaN and constant or
    empty columns, like the real ones) that records the parameters it was given."""
    from sklearn.ensemble import RandomForestRegressor

    class _Boost(RandomForestRegressor):
        def __init__(self, **params):
            super().__init__(n_estimators=20, random_state=0)
            self.params = params

        def get_params(self, deep=True):
            return {}

    lightgbm = types.ModuleType("lightgbm")
    lightgbm.LGBMRegressor = type("LGBMRegressor", (_Boost,), {})
    xgboost = types.ModuleType("xgboost")
    xgboost.XGBRegressor = type("XGBRegressor", (_Boost,), {})
    return lightgbm, xgboost


def _install(monkeypatch, boosting=True):
    package, lag_module, target_module = _fake_mlforecast()
    monkeypatch.setitem(sys.modules, "mlforecast", package)
    monkeypatch.setitem(sys.modules, "mlforecast.lag_transforms", lag_module)
    monkeypatch.setitem(sys.modules, "mlforecast.target_transforms", target_module)
    if boosting:
        lightgbm, xgboost = _fake_boosting()
        monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)
        monkeypatch.setitem(sys.modules, "xgboost", xgboost)


def _trend_train(n_series=3, n=40, extra=None):
    """Series y = a + b*t with different levels: a linear lag model can extrapolate them."""
    frames = []
    for i in range(n_series):
        ds = pd.date_range("2021-01-01", periods=n, freq="MS")
        y = 100.0 * (i + 1) + (i + 1) * np.arange(n)
        frame = pd.DataFrame({"unique_id": make_uuid(i), "ds": ds, "y": y, "is_filled": False})
        frames.append(frame if extra is None else frame.assign(**extra(i, n)))
    return pd.concat(frames, ignore_index=True)


def _future(train, h):
    frames = []
    for uid, g in train.groupby("unique_id", sort=False):
        ds = pd.date_range(g["ds"].max(), periods=h + 1, freq="MS")[1:]
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "horizon_step": range(1, h + 1)}))
    return pd.concat(frames, ignore_index=True)


def _builder(set_kwargs=None, static=None, freq="MS", season=12):
    feature_set = FeatureSet(name="s", calendar="all", fourier_order=0, **(set_kwargs or {}))
    return FeatureBuilder(feature_set, pandas_freq=freq, season_length=season, static=static)


def test_ridge_extrapolates_a_trend_through_the_recursive_forecast(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1, 2], regressor_params={"alpha": 1e-8})
    out = model.fit(train).predict(_future(train, 4))
    expected = [100.0 * (i + 1) + (i + 1) * t for i in range(3) for t in range(40, 44)]
    assert np.allclose(out["yhat"], expected, rtol=1e-3)


def test_predict_keeps_the_row_order_of_future(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1, 2], regressor_params={"alpha": 1e-8}).fit(train)
    future = _future(train, 3).sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = model.predict(future)
    assert out["unique_id"].tolist() == future["unique_id"].tolist()
    assert out["ds"].tolist() == future["ds"].tolist()
    level = out["unique_id"].map({make_uuid(i): 100.0 * (i + 1) for i in range(3)})
    assert (out["yhat"] > level).all()


def test_only_requested_series_are_predicted(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1, 2]).fit(train)
    future = _future(train, 2)
    future = future[future["unique_id"] == make_uuid(1)]
    assert set(model.predict(future)["unique_id"]) == {make_uuid(1)}


def test_library_settings_are_passed_to_mlforecast(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    LassoForecaster(season_length=12, freq="MS", lags=[1, "season"], rolling_windows=[3, 12],
                    rolling_stats=("mean", "max"), date_features=["month"],
                    target_transforms=("local_minmax", "seasonal_difference"), num_threads=2).fit(train)
    mlf = _Recorder.last
    assert mlf.freq == "MS" and mlf.lags == [1, 12] and mlf.num_threads == 2
    assert mlf.date_features == ["month"]
    windows = [(type(t).__name__, t.window_size) for t in mlf.lag_transforms[1]]
    assert windows == [("RollingMean", 3), ("RollingMax", 3), ("RollingMean", 12), ("RollingMax", 12)]
    assert [type(t).__name__ for t in mlf.target_transforms] == ["LocalMinMaxScaler", "Differences"]
    assert mlf.target_transforms[1].args == ([12],)


def test_defaults_per_frequency(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    RidgeForecaster(season_length=12, freq="MS").fit(train)
    mlf = _Recorder.last
    assert mlf.lags == [1, 2, 3, 12]
    assert mlf.date_features == ["month", "quarter"]
    assert [t.window_size for t in mlf.lag_transforms[1]] == [3, 3, 12, 12]
    assert [type(t).__name__ for t in mlf.target_transforms] == ["LocalStandardScaler"]
    # yearly data: season 1, so the seasonal lag and window collapse, no date features
    RidgeForecaster(season_length=1, freq="YS").fit(_yearly(train))
    mlf = _Recorder.last
    assert mlf.lags == [1, 2, 3] and mlf.date_features == []
    assert [t.window_size for t in mlf.lag_transforms[1]] == [3, 3]


def _yearly(train):
    out = []
    for uid, g in train.groupby("unique_id", sort=False):
        out.append(g.assign(ds=pd.date_range("1980-01-01", periods=len(g), freq="YS")))
    return pd.concat(out, ignore_index=True)


def test_min_train_length():
    assert RidgeForecaster(season_length=12).min_train_length == 16      # window 12 ends at lag 1, so 13 values, + 3 rows to train on
    assert RidgeForecaster(season_length=1, freq="YS").min_train_length == 7  # windows 3 -> 4 vs lag 3
    assert RidgeForecaster(season_length=12, lags=[1], rolling_windows=[]).min_train_length == 4
    assert RidgeForecaster(season_length=12, min_length=30).min_train_length == 30


def test_lags_and_windows_per_frequency():
    lags = {"monthly": [1, 2, 3, 6, 12, 24], "daily": [1, 7, 28]}
    windows = {"monthly": [3, 6, 12]}
    monthly = RidgeForecaster(season_length=12, freq="MS", lags=lags, rolling_windows=windows)
    assert monthly._resolved_lags() == (1, 2, 3, 6, 12, 24) and monthly._resolved_windows() == (3, 6, 12)
    assert monthly.min_train_length == 27  # lag 24 + 3
    daily = RidgeForecaster(season_length=7, freq="D", lags=lags, rolling_windows=windows)
    assert daily._resolved_lags() == (1, 7, 28) and daily._resolved_windows() == (3, 7)  # windows: default
    weekly = RidgeForecaster(season_length=52, freq="W-MON", lags=lags)  # not listed: the default
    assert weekly._resolved_lags() == (1, 2, 3, 52)
    with pytest.raises(ValueError, match="unknown frequency"):
        RidgeForecaster(season_length=12, freq="MS", lags={"hourly": [1]})


def test_known_columns_reach_the_model_and_need_future_values(monkeypatch):
    _install(monkeypatch)

    def extra(i, n):
        rng = np.random.default_rng(i)
        return {"x_promo": rng.integers(0, 2, n).astype(float)}

    train = _trend_train(2, 60, extra)
    # y depends on the promo: add 50 when the promo is on
    train["y"] = train["y"] + 50.0 * train["x_promo"]
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1], rolling_windows=[],
                            known_columns=["x_promo"], regressor_params={"alpha": 1e-8}).fit(train)
    assert _Recorder.last.dynamic == ["x_promo"]
    future = _future(train, 2).assign(x_promo=[0.0, 0.0, 0.0, 0.0])
    off = model.predict(future)["yhat"].to_numpy()
    on = model.predict(future.assign(x_promo=[1.0, 1.0, 1.0, 1.0]))["yhat"].to_numpy()
    assert (on - off > 25).all()
    with pytest.raises(ValueError, match="missing known column"):
        model.predict(_future(train, 2))
    with pytest.raises(ValueError, match="not in the training data"):
        RidgeForecaster(season_length=12, freq="MS", known_columns=["nope"]).fit(train)


def test_unknown_explanatory_columns_are_ignored(monkeypatch):
    _install(monkeypatch)
    train = _trend_train(2, 40, lambda i, n: {"x_temp": np.arange(n, dtype=float)})
    RidgeForecaster(season_length=12, freq="MS", lags=[1]).fit(train)
    assert "x_temp" not in _Recorder.last.fit_columns


def test_calendar_features_from_a_feature_set_are_dynamic_and_known_in_the_future(monkeypatch):
    _install(monkeypatch)
    builder = _builder()
    calendar = builder.calendar_columns()
    assert calendar
    train = _trend_train()
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1, 2], feature_builder=builder).fit(train)
    mlf = _Recorder.last
    assert mlf.dynamic == calendar and mlf.static == []
    assert mlf.date_features == []  # the feature set supplies the calendar
    out = model.predict(_future(train, 3))
    assert len(out) == 9 and np.isfinite(out["yhat"]).all()


def test_static_features_are_static_columns_from_training_data_only(monkeypatch):
    _install(monkeypatch)
    train = _trend_train(4)
    static = pd.DataFrame({"unique_id": [make_uuid(i) for i in range(4)],
                           "market": ["DE04", "PT04", "DE04", "PT04"]})
    spec = StaticSpec(identifiers=("market",), group_by=(("market",),), group_stats=("mean",))
    builder = _builder({"static": spec}, static=static)
    assert builder.static_columns
    RidgeForecaster(season_length=12, freq="MS", lags=[1], feature_builder=builder).fit(train)
    mlf = _Recorder.last
    assert mlf.static == builder.static_columns
    assert set(builder.static_columns) <= set(mlf.fit_columns)
    assert not set(builder.static_columns) & set(mlf.dynamic)


def test_a_feature_set_with_lags_is_refused():
    builder = _builder({"lags": (1, 2)})
    with pytest.raises(ValueError, match="must not define lags"):
        RidgeForecaster(season_length=12, feature_builder=builder)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="rolling_stats"):
        RidgeForecaster(rolling_stats=["median"])
    with pytest.raises(ValueError, match="target_transforms"):
        RidgeForecaster(target_transforms=["log"])
    with pytest.raises(ValueError, match="season_length"):
        RidgeForecaster(season_length=0)


def test_missing_library_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlforecast", None)
    with pytest.raises(ImportError, match="extras ml"):
        RidgeForecaster(season_length=12, freq="MS").fit(_trend_train())


def test_boosting_defaults_can_be_overridden(monkeypatch):
    _install(monkeypatch)
    model = LightGBMForecaster(season_length=12, regressor_params={"n_estimators": 7, "num_leaves": 4})
    params = model._make_regressor().params
    assert params["n_estimators"] == 7 and params["num_leaves"] == 4
    assert params["learning_rate"] == 0.05 and params["n_jobs"] == -1 and params["verbosity"] == -1
    assert XGBoostForecaster(season_length=12)._make_regressor().params["tree_method"] == "hist"


def test_real_sklearn_regressors_build():
    forest = RandomForestForecaster(season_length=12, regressor_params={"n_estimators": 5})._make_regressor()
    assert forest.n_estimators == 5 and forest.n_jobs == -1 and forest.random_state == 0
    ridge = RidgeForecaster(season_length=12, regressor_params={"alpha": 3.0})._make_regressor()
    assert [name for name, _ in ridge.steps] == ["simpleimputer", "standardscaler", "ridge"]
    assert ridge.steps[-1][1].alpha == 3.0
    assert LassoForecaster(season_length=12)._make_regressor().steps[-1][1].alpha == 0.001


def test_every_regressor_trains_through_the_fake_library(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    future = _future(train, 2)
    for cls in (LightGBMForecaster, XGBoostForecaster, RandomForestForecaster, RidgeForecaster, LassoForecaster):
        kwargs = {"regressor_params": {"n_estimators": 10}} if cls is RandomForestForecaster else {}
        out = cls(season_length=12, freq="MS", lags=[1, 2], **kwargs).fit(train).predict(future)
        assert len(out) == 6 and np.isfinite(out["yhat"]).all(), cls.__name__


def test_clone_keeps_parameters_and_shares_the_builder():
    builder = _builder()
    model = LassoForecaster(season_length=12, freq="MS", lags=[1, 2], feature_builder=builder,
                            known_columns=("x",), regressor_params={"alpha": 0.5})
    clone = model.clone()
    assert clone is not model and clone.get_params() == model.get_params()
    assert clone.feature_builder is builder
    assert model.scope == "global"


def test_future_with_unequal_horizons_is_refused_when_exogenous_features_are_used(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1], feature_builder=_builder()).fit(train)
    future = pd.concat([_future(train, 3)[lambda d: d["unique_id"] == make_uuid(0)],
                        _future(train, 2)[lambda d: d["unique_id"] == make_uuid(1)]])
    with pytest.raises(ValueError, match="future steps 1..3"):
        model.predict(future)


def test_future_with_a_missing_step_is_refused(monkeypatch):
    """X_df rows are matched to steps by position, so a gap would shift the inputs silently."""
    _install(monkeypatch)
    train = _trend_train()
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1], feature_builder=_builder()).fit(train)
    future = _future(train, 4)
    future = future[future["horizon_step"] != 2]  # 3 rows per series, but steps 1, 3, 4
    with pytest.raises(ValueError, match="future steps 1..4"):
        model.predict(future)


def test_date_feature_clashing_with_a_column_is_refused(monkeypatch):
    _install(monkeypatch)
    train = _trend_train(extra=lambda i, n: {"month": np.arange(n) % 12})
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1], date_features=["month"],
                            known_columns=["month"])
    with pytest.raises(ValueError, match="clash"):
        model.fit(train)


# ---- catalog -------------------------------------------------------------------------------


def test_catalog_registers_the_types_and_default_entries():
    specs = load_model_specs()
    for name in ML_TYPES:
        assert name in MODEL_TYPES
        assert name in specs
    assert specs["lightgbm_static"]["skip_without_static"] is True
    assert specs["lightgbm"]["skip_without_static"] is False


def test_catalog_skips_ml_models_when_libraries_are_missing(monkeypatch, caplog):
    for name in ("mlforecast", "lightgbm", "xgboost", "statsforecast"):
        monkeypatch.setitem(sys.modules, name, None)
    built = build_models(load_model_specs(types=ML_TYPES), 12, pandas_freq="MS")
    assert built == {}
    assert "--extras ml" in caplog.text


def test_catalog_skips_only_the_model_whose_regressor_package_is_missing(monkeypatch):
    _install(monkeypatch, boosting=False)
    monkeypatch.setitem(sys.modules, "xgboost", None)
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    built = build_models(load_model_specs(types=ML_TYPES), 12,
                         feature_builders={"calendar_only": _builder(), "global_static": _builder()},
                         pandas_freq="MS")
    # without lightgbm and xgboost only the scikit-learn models are left (every random forest setting)
    rf = {n for n, spec in load_model_specs(types=ML_TYPES).items() if spec["type"] == "random_forest"}
    assert set(built) == rf | {"ridge", "lasso"} and "random_forest" in rf


def _config_dir(tmp_path, models_yaml):
    for name in ("default.yaml", "frequency_profiles.yaml", "features.yaml"):
        (tmp_path / name).write_text(open(f"config/{name}").read())
    (tmp_path / "models.yaml").write_text(models_yaml)
    return tmp_path


def _static_table():
    ids = [make_uuid(i) for i in range(4)]
    return pd.DataFrame({"unique_id": ids, "market": ["DE04", "PT04", "DE04", "PT04"],
                         "product": ["F0101", "F0101", "M0101", "M0101"],
                         "company": ["company A", "company A", "company B", "company B"]})


def test_catalog_builds_ml_models_with_known_columns_and_builders(monkeypatch, tmp_path):
    _install(monkeypatch)
    directory = _config_dir(tmp_path, open("config/models.yaml").read())
    cfg = load_config({"frequency": "monthly", "horizon": 3, "known_future_columns": ["x_promo"]},
                      config_dir=directory)
    models = build_models_from_config(cfg, directory, _static_table())
    assert {"lightgbm", "lightgbm_static", "xgboost", "xgboost_static", "random_forest", "ridge",
            "lasso"} <= set(models)
    ridge = models["ridge"]
    assert ridge.known_columns == ("x_promo",) and ridge.freq == "MS" and ridge.season_length == 12
    assert ridge.feature_builder.calendar_columns() and not ridge.feature_builder.static_columns
    assert models["lightgbm_static"].feature_builder.static_columns


def test_catalog_skips_static_models_without_static_columns(monkeypatch, tmp_path, caplog):
    _install(monkeypatch)
    directory = _config_dir(tmp_path, open("config/models.yaml").read())
    cfg = load_config({"frequency": "monthly", "horizon": 3, "static_columns": []}, config_dir=directory)
    models = build_models_from_config(cfg, directory, pd.DataFrame({"unique_id": []}))
    assert "lightgbm" in models and "lightgbm_static" not in models and "xgboost_static" not in models
    assert "lightgbm_static" in caplog.text and "static_columns" in caplog.text


def test_catalog_still_refuses_static_sets_when_not_marked_optional(monkeypatch, tmp_path):
    _install(monkeypatch)
    directory = _config_dir(tmp_path, "lightgbm:\n  type: lightgbm\n  features: global_static\n")
    cfg = load_config({"frequency": "monthly", "horizon": 3}, config_dir=directory)
    with pytest.raises(ConfigError, match="static_columns"):
        build_models_from_config(cfg, directory, pd.DataFrame({"unique_id": []}))


def test_catalog_reports_invalid_model_settings(monkeypatch, tmp_path):
    _install(monkeypatch)
    directory = _config_dir(tmp_path, "ridge:\n  type: ridge\n  params:\n    rolling_stats: [median]\n")
    cfg = load_config({"frequency": "monthly", "horizon": 3}, config_dir=directory)
    with pytest.raises(ConfigError, match="rolling_stats"):
        build_models_from_config(cfg, directory, _static_table())


def test_catalog_rejects_a_feature_set_with_lags_for_ml_models(monkeypatch, tmp_path):
    _install(monkeypatch)
    directory = _config_dir(tmp_path, "ridge:\n  type: ridge\n  features: calendar_and_history\n")
    cfg = load_config({"frequency": "monthly", "horizon": 3}, config_dir=directory)
    with pytest.raises(ConfigError, match="must not define lags"):
        build_models_from_config(cfg, directory, _static_table())


def test_skip_without_static_must_be_boolean(tmp_path):
    (tmp_path / "models.yaml").write_text("a:\n  type: ridge\n  skip_without_static: maybe\n")
    with pytest.raises(ConfigError, match="skip_without_static"):
        load_model_specs(tmp_path)


def test_update_forecasts_from_newer_history_without_refitting(monkeypatch):
    _install(monkeypatch)
    old, new = _trend_train(n=40), _trend_train(n=43)
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1, 2], regressor_params={"alpha": 1e-8})
    model.fit(old)
    estimator = _Recorder.last.estimator
    assert model.supports_update
    model.update(new)
    assert _Recorder.last.estimator is estimator  # same fitted regressor, nothing refitted
    out = model.predict(_future(new, 3))
    expected = [100.0 * (i + 1) + (i + 1) * t for i in range(3) for t in range(43, 46)]
    assert np.allclose(out["yhat"], expected, rtol=1e-3)


def test_predict_without_update_does_not_pass_new_df(monkeypatch):
    _install(monkeypatch)
    train = _trend_train()
    model = RidgeForecaster(season_length=12, freq="MS", lags=[1, 2]).fit(train)
    assert model.new_df_ is None
    model.predict(_future(train, 2))  # the fake would fail on an unexpected argument otherwise


def test_profile_ml_features_are_the_default_lags_per_group():
    specs = load_model_specs(types=["ridge", "random_forest"])
    specs = {"ridge": specs["ridge"], "random_forest": specs["random_forest"],
             "own": {"type": "random_forest", "params": {"lags": [1, 2]}, "features": None}}
    builders = {"calendar_only": _builder()}
    defaults = {"trees": {"lags": [1, 4], "rolling_windows": [4]}}
    built = build_models(specs, 12, builders, pandas_freq="MS", ml_features=defaults)
    assert built["random_forest"]._resolved_lags() == (1, 4) and built["random_forest"]._resolved_windows() == (4,)
    assert built["ridge"]._resolved_lags() == (1, 2, 3, 12)  # linear group not given: adapter default
    assert built["own"]._resolved_lags() == (1, 2)  # models.yaml lags win
