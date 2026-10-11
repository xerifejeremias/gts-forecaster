"""forecast_floor: negative forecasts become 0 (volume data)."""

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.evaluation.backtest import run_backtest
from gts_forecaster.forecasting import combine_forecasts, forecast_by_model
from gts_forecaster.models.naive import DriftForecaster
from gts_forecaster.output import build_output
from gts_forecaster.splitting import forecast_split

N = 12


def _falling():
    """A series that falls by 10 per month and ends at 5: drift goes negative right away."""
    dates = pd.date_range("2024-01-01", periods=N, freq="MS")
    df = pd.DataFrame({"unique_id": "s", "ds": dates, "y": 5.0 + 10.0 * np.arange(N)[::-1]})
    df["is_filled"] = False
    return df


def test_config_default_is_zero_and_can_be_switched_off(tmp_path):
    import shutil

    from gts_forecaster.config import project_root

    assert load_config().forecast_floor == 0.0
    config_dir = tmp_path / "config"
    shutil.copytree(project_root() / "config", config_dir)
    default = config_dir / "default.yaml"
    default.write_text(default.read_text().replace("forecast_floor: 0", "forecast_floor: null"))
    assert load_config(config_dir=config_dir).forecast_floor is None
    assert load_config({"forecast_floor": 1}).forecast_floor == 1.0
    with pytest.raises(ConfigError):
        load_config({"forecast_floor": "zero"})


def test_forecast_by_model_floors_each_model():
    split = forecast_split(_falling(), 3, "MS")
    raw = forecast_by_model(split, {"drift": DriftForecaster()})
    floored = forecast_by_model(split, {"drift": DriftForecaster()}, forecast_floor=0.0)
    assert (raw["drift"] < 0).all()
    assert (floored["drift"] == 0).all()


def test_backtest_scores_use_floored_predictions():
    kw = dict(horizon=2, pandas_freq="MS", n_folds=1, metric="mae", season_length=12)
    raw = run_backtest(_falling(), {"drift": DriftForecaster()}, **kw)
    floored = run_backtest(_falling(), {"drift": DriftForecaster()}, forecast_floor=0.0, **kw)
    assert (floored.predictions["yhat"] >= 0).all()
    assert (raw.predictions["yhat"] >= 0).all()  # the test window is still positive here
    # Train ends at 25, actuals 15 and 5; drift predicts 15 and 5 exactly, so nothing changes.
    assert floored.scores["score"].iloc[0] == pytest.approx(raw.scores["score"].iloc[0])


def test_backtest_floor_changes_negative_predictions():
    df = _falling()
    df = pd.concat([df, df.iloc[[-1]].assign(ds=pd.Timestamp("2025-01-01"), y=0.0),
                    df.iloc[[-1]].assign(ds=pd.Timestamp("2025-02-01"), y=0.0)], ignore_index=True)
    kw = dict(horizon=2, pandas_freq="MS", n_folds=1, metric="mae", season_length=12)
    raw = run_backtest(df, {"drift": DriftForecaster()}, **kw)
    floored = run_backtest(df, {"drift": DriftForecaster()}, forecast_floor=0.0, **kw)
    assert (raw.predictions["yhat"] < 0).any()
    assert (floored.predictions["yhat"] >= 0).all()
    assert floored.scores["score"].iloc[0] < raw.scores["score"].iloc[0]  # actuals are 0


def test_combine_floors_fallback_values():
    dates = pd.date_range("2025-01-01", periods=2, freq="MS")
    wide = pd.DataFrame({"unique_id": "a", "ds": dates, "naive": [1.0, 2.0]})
    sel = {"series": {"a": {"models": [{"name": "naive", "weight": 1.0}]}}}
    fb = pd.DataFrame({"unique_id": ["b"], "fallback_value": [-3.0]})
    out = combine_forecasts(wide, sel, fb, dates, forecast_floor=0.0)
    assert out.loc[out.unique_id == "b", "forecast"].tolist() == [0.0, 0.0]


def test_output_contract_rejects_values_below_floor():
    dates = pd.date_range("2025-01-01", periods=2, freq="MS")
    final = pd.DataFrame({"ds": dates, "unique_id": "a", "forecast": [-1.0, 2.0]})
    with pytest.raises(DataContractError, match="below"):
        build_output(final, None, expected_ids={"a"}, horizon_dates=dates, forecast_floor=0.0)
    build_output(final, None, expected_ids={"a"}, horizon_dates=dates)  # no floor: allowed
