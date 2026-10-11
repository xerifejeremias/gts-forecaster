from types import SimpleNamespace

import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.features.known_future import (
    attach_known_future,
    known_columns_from_test,
    load_known_future,
)
from gts_forecaster.models.base import BaseForecaster
from helpers import make_panel
from helpers import use_core_catalog

@pytest.fixture(autouse=True)
def _core_catalog(monkeypatch):
    use_core_catalog(monkeypatch)  # pipeline tests: no tuning variants of the tree models



def _future(ids=("a", "b"), n=3):
    ds = pd.date_range("2025-01-01", periods=n, freq="MS")
    return pd.concat(
        [pd.DataFrame({"unique_id": u, "ds": ds, "horizon_step": range(1, n + 1)}) for u in ids],
        ignore_index=True,
    )


def _table(with_uuid=False, ids=("a", "b"), n=3, extra=False):
    ds = pd.date_range("2025-01-01", periods=n, freq="MS")
    if with_uuid:
        df = pd.concat(
            [pd.DataFrame({"uuid": u, "date": ds, "x_promo": [float(i % 2) for i in range(n)]}) for u in ids]
        )
    else:
        df = pd.DataFrame({"date": ds, "x_promo": [1.0, 0.0, 1.0][:n]})
    if extra:
        df["x_temp"] = 20.0
    return df.reset_index(drop=True)


def test_default_config_has_no_known_columns():
    cfg = load_config({"frequency": "monthly", "horizon": 3})
    assert cfg.known_future_columns == () and cfg.future_features_path is None


def test_config_accepts_known_columns_and_path():
    cfg = load_config(
        {"frequency": "monthly", "horizon": 3, "known_future_columns": ["x_promo"],
         "future_features_path": "data/{frequency}/promo.csv"}
    )
    assert cfg.known_future_columns == ("x_promo",)
    assert cfg.future_features_path.name == "promo.csv" and "monthly" in str(cfg.future_features_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"known_future_columns": ["nope"]},
        {"known_future_columns": ["x_promo", "x_promo"]},
        {"known_future_columns": "x_promo"},
        {"future_features_path": "f.csv"},  # path without known columns
    ],
)
def test_config_rejects_bad_known_settings(overrides):
    with pytest.raises(ConfigError):
        load_config({"frequency": "monthly", "horizon": 3, **overrides})


def test_same_values_for_all_series_when_no_uuid():
    known = load_known_future(_table(), ["x_promo"])
    out = attach_known_future(_future(), known, ["x_promo"])
    assert out[out.unique_id == "a"]["x_promo"].tolist() == [1.0, 0.0, 1.0]
    assert out[out.unique_id == "b"]["x_promo"].tolist() == [1.0, 0.0, 1.0]
    assert len(out) == 6


def test_values_per_series_with_uuid():
    t = _table(with_uuid=True)
    t.loc[(t.uuid == "b") & (t.date == "2025-01-01"), "x_promo"] = 9.0
    out = attach_known_future(_future(), load_known_future(t, ["x_promo"]), ["x_promo"])
    assert out[(out.unique_id == "b") & (out.ds == "2025-01-01")]["x_promo"].iloc[0] == 9.0
    assert out[(out.unique_id == "a") & (out.ds == "2025-01-01")]["x_promo"].iloc[0] == 0.0


def test_extra_columns_and_rows_outside_window_are_ignored():
    known = load_known_future(_table(n=3, extra=True), ["x_promo"])
    assert list(known.columns) == ["ds", "x_promo"]
    out = attach_known_future(_future(n=2), known, ["x_promo"])
    assert len(out) == 4


def test_loads_from_csv(tmp_path):
    path = tmp_path / "f.csv"
    _table().to_csv(path, index=False)
    assert len(load_known_future(path, ["x_promo"])) == 3
    with pytest.raises(DataContractError):
        load_known_future(tmp_path / "missing.csv", ["x_promo"])


def test_gaps_stop_the_run():
    known = load_known_future(_table(n=2), ["x_promo"])
    with pytest.raises(DataContractError, match="do not cover"):
        attach_known_future(_future(n=3), known, ["x_promo"])
    per_series = load_known_future(_table(with_uuid=True, ids=("a",)), ["x_promo"])
    with pytest.raises(DataContractError, match="b @"):
        attach_known_future(_future(), per_series, ["x_promo"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns="date"),
        lambda d: d.drop(columns="x_promo"),
        lambda d: d.assign(x_promo=[1.0, None, 1.0]),
        lambda d: d.assign(x_promo=[1.0, float("inf"), 1.0]),
        lambda d: d.assign(x_promo=["a", "b", "c"]),
        lambda d: pd.concat([d, d.iloc[:1]]),
        lambda d: d.assign(date=["x", "y", "z"]),
    ],
)
def test_invalid_tables_are_rejected(mutate):
    with pytest.raises(DataContractError):
        load_known_future(mutate(_table()), ["x_promo"])


def test_nothing_attached_when_no_column_is_known():
    f = _future()
    assert attach_known_future(f, pd.DataFrame(), []) is f
    assert known_columns_from_test(f, pd.DataFrame(), []) is f


def test_backtest_future_gets_actual_values_of_known_columns():
    panel = make_panel(2, 6, internal=True)
    test = panel[panel.ds >= "2024-05-01"]
    future = test[["unique_id", "ds"]].assign(horizon_step=[1, 2] * 2)
    out = known_columns_from_test(future, test, ["x1"])
    assert out["x1"].tolist() == test["x1"].tolist() and len(out) == len(future)


class _Probe(BaseForecaster):
    seen = None

    def _fit(self, series, train):
        self.last_ = {u: y[-1] for u, y in series.items()}

    def _predict(self, future):
        type(self).seen = list(future.columns)
        return future["unique_id"].map(self.last_).to_numpy(dtype=float)


def test_backtest_passes_known_columns_to_models_and_unknown_ones_never():
    from gts_forecaster.evaluation.backtest import run_backtest

    df = make_panel(2, 12, internal=True)
    df["is_filled"] = False
    run_backtest(df, {"naive": _Probe()}, horizon=2, pandas_freq="MS", n_folds=1,
                 known_columns=["x1"])
    assert _Probe.seen == ["unique_id", "ds", "horizon_step", "x1"]
    run_backtest(df, {"naive": _Probe()}, horizon=2, pandas_freq="MS", n_folds=1)
    assert _Probe.seen == ["unique_id", "ds", "horizon_step"]


def _prepared(tmp_path, known):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline  # noqa: F401

    cfg = load_config(
        {"frequency": "monthly", "horizon": 3, "n_folds": 1, "top_n": 1,
         "registry_dir": tmp_path, "backtesting_dir": tmp_path / "bt",
         "known_future_columns": known}
    )
    df = make_panel(2, 24, internal=True, explanatory=("x_promo",))
    df["is_filled"] = False
    from helpers import static_table

    return SimpleNamespace(data=df, config=cfg, cutoff=df["ds"].max(),
                           fallback=pd.DataFrame(columns=["unique_id"]), static=static_table(df["unique_id"]))


def test_forecast_pipeline_requires_values_for_known_columns(tmp_path):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline

    prepared = _prepared(tmp_path, ["x_promo"])
    run_backtest_pipeline(prepared)
    with pytest.raises(DataContractError, match="no future"):
        run_forecast_pipeline(prepared, output_dir=tmp_path / "out")
    future_dates = pd.date_range("2026-01-01", periods=3, freq="MS")
    good = pd.DataFrame({"date": future_dates, "x_promo": [0.0, 1.0, 0.0]})
    out = run_forecast_pipeline(prepared, output_dir=tmp_path / "out", known_future=good)
    assert len(out.forecast) == 6
    with pytest.raises(DataContractError, match="do not cover"):
        run_forecast_pipeline(prepared, output_dir=tmp_path / "out", known_future=good.iloc[:2])


def test_forecast_pipeline_needs_nothing_by_default(tmp_path):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline

    prepared = _prepared(tmp_path, [])
    run_backtest_pipeline(prepared)
    assert len(run_forecast_pipeline(prepared, output_dir=tmp_path / "out").forecast) == 6
