from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.forecasting import ForecastError, combine_forecasts, forecast_by_model
from gts_forecaster.models.catalog import build_models, load_model_specs
from helpers import BASELINES
from gts_forecaster.splitting import forecast_split

N = 24


def _panel():
    dates = pd.date_range("2022-01-01", periods=N, freq="MS")
    frames = [
        pd.DataFrame({"unique_id": "linear", "ds": dates, "y": np.arange(1.0, N + 1)}),
        pd.DataFrame({"unique_id": "const", "ds": dates, "y": 5.0}),
        pd.DataFrame({"unique_id": "short", "ds": dates[-1:], "y": [3.0]}),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["is_filled"] = False
    return df


def _models():
    return build_models(load_model_specs(types=BASELINES), 12)


def _split(h=3):
    return forecast_split(_panel(), h, "MS")


def _sel(**series):
    return {"series": {u: {"locked": False, "models": m} for u, m in series.items()}}


def test_wide_has_one_column_per_model_and_all_dates():
    wide = forecast_by_model(_split(), _models())
    assert list(wide.columns) == ["unique_id", "ds", "naive", "seasonal_naive", "drift"]
    assert len(wide) == 3 * 3
    row = wide[(wide.unique_id == "linear")].iloc[0]
    assert row["naive"] == 24.0 and row["drift"] == 25.0


def test_model_that_cannot_forecast_a_series_leaves_nan():
    wide = forecast_by_model(_split(), _models())
    short = wide[wide.unique_id == "short"]
    assert short["naive"].notna().all() and short["drift"].isna().all()


def test_combine_is_weighted_average_of_selected_models():
    wide = forecast_by_model(_split(), _models())
    sel = _sel(
        linear=[{"name": "naive", "weight": 0.25}, {"name": "drift", "weight": 0.75}],
        const=[{"name": "naive", "weight": 1.0}],
        short=[{"name": "naive", "weight": 1.0}],
    )
    out = combine_forecasts(wide, sel)
    assert list(out.columns) == ["ds", "unique_id", "forecast"]
    first = out[out.unique_id == "linear"].iloc[0]
    assert first["forecast"] == pytest.approx(0.25 * 24 + 0.75 * 25)
    assert out.forecast.notna().all() and len(out) == 9


def test_fallback_series_get_constant_forecast():
    split = _split()
    wide = forecast_by_model(split, _models())
    sel = _sel(
        linear=[{"name": "naive", "weight": 1.0}],
        const=[{"name": "naive", "weight": 1.0}],
        short=[{"name": "naive", "weight": 1.0}],
    )
    fallback = pd.DataFrame({"unique_id": ["gap"], "fallback_value": [7.0]})
    out = combine_forecasts(wide, sel, fallback, split.horizon_dates)
    gap = out[out.unique_id == "gap"]
    assert len(gap) == 3 and (gap.forecast == 7.0).all()


def test_missing_selection_entry_fails():
    wide = forecast_by_model(_split(), _models())
    with pytest.raises(ForecastError):
        combine_forecasts(wide, _sel(linear=[{"name": "naive", "weight": 1.0}]))


def test_selected_model_without_forecast_fails():
    wide = forecast_by_model(_split(), _models())
    sel = _sel(
        linear=[{"name": "naive", "weight": 1.0}],
        const=[{"name": "naive", "weight": 1.0}],
        short=[{"name": "drift", "weight": 1.0}],
    )
    with pytest.raises(ForecastError):
        combine_forecasts(wide, sel)


def test_unknown_model_column_fails():
    wide = forecast_by_model(_split(), _models())
    sel = _sel(
        linear=[{"name": "nope", "weight": 1.0}],
        const=[{"name": "naive", "weight": 1.0}],
        short=[{"name": "naive", "weight": 1.0}],
    )
    with pytest.raises(ForecastError):
        combine_forecasts(wide, sel)


def test_fallback_without_dates_fails():
    wide = forecast_by_model(_split(), _models())
    sel = _sel(**{u: [{"name": "naive", "weight": 1.0}] for u in ("linear", "const", "short")})
    with pytest.raises(ForecastError):
        combine_forecasts(wide, sel, pd.DataFrame({"unique_id": ["g"], "fallback_value": [1.0]}))


def test_pipeline_end_to_end(tmp_path):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline

    cfg = load_config(
        {"frequency": "monthly", "horizon": 3, "n_folds": 2, "top_n": 2, "registry_dir": tmp_path, "backtesting_dir": tmp_path / "bt"}
    )
    df = _panel()
    fb = pd.DataFrame({"unique_id": ["gap"], "fallback_value": [7.0]})
    from helpers import static_table

    static = static_table([*df["unique_id"], "gap"])
    prepared = SimpleNamespace(data=df, config=cfg, cutoff=df["ds"].max(), fallback=fb, static=static)
    run_backtest_pipeline(prepared)
    out = run_forecast_pipeline(prepared, output_dir=tmp_path / "out")
    final = pd.read_csv(out.forecast_path)
    assert list(final.columns) == ["ds", "uuid", "forecast", "market", "product", "company"]
    platform = pd.read_csv(out.output_path)
    assert out.output_path.name == "forecast_output.csv"
    assert list(platform.columns) == ["uuid", "date", "market", "product", "company", "forecast"]
    assert len(platform) == 12 and set(platform.uuid) == {"linear", "const", "short", "gap"}
    assert platform.date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()  # dates written as YYYY-MM-DD
    assert set(final.uuid) == {"linear", "const", "short", "gap"}
    assert final.forecast.notna().all() and len(final) == 12
    assert out.by_model_path.is_file()
    # the sanity checks ran (the file is written even when nothing is flagged)
    assert (out.forecast_path.parent / "forecast_checks.csv").is_file() and out.checks is not None


def test_pipeline_with_failing_checks_and_raise_writes_nothing(tmp_path, monkeypatch):
    from gts_forecaster.forecast_checks import ForecastCheckError
    from gts_forecaster.pipelines import forecast as forecast_module
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    cfg = load_config({"frequency": "monthly", "horizon": 3, "n_folds": 2, "top_n": 2, "registry_dir": tmp_path,
                       "backtesting_dir": tmp_path / "bt", "forecast_check_action": "raise"})
    df = _panel()
    prepared = SimpleNamespace(data=df, config=cfg, cutoff=df["ds"].max(),
                               fallback=pd.DataFrame({"unique_id": [], "fallback_value": []}), static=None)
    run_backtest_pipeline(prepared)
    real = forecast_module.combine_forecasts
    monkeypatch.setattr(forecast_module, "combine_forecasts",
                        lambda *a, **k: real(*a, **k).assign(forecast=lambda f: f["forecast"] * 50 + 1000))
    with pytest.raises(ForecastCheckError):
        forecast_module.run_forecast_pipeline(prepared, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_pipeline_stops_without_selection_file(tmp_path):
    from gts_forecaster.contracts.errors import DataContractError
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline

    cfg = load_config({"frequency": "monthly", "horizon": 3, "registry_dir": tmp_path, "backtesting_dir": tmp_path / "bt"})
    df = _panel()
    prepared = SimpleNamespace(
        data=df, config=cfg, cutoff=df["ds"].max(), fallback=pd.DataFrame(columns=["unique_id"])
    )
    with pytest.raises(DataContractError):
        run_forecast_pipeline(prepared, output_dir=tmp_path / "out")
