import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError
from gts_forecaster.evaluation.metrics import METRICS, get_metric
from gts_forecaster.models.base import HORIZON_STEP
from gts_forecaster.models.catalog import build_models, load_model_specs
from gts_forecaster.models.naive import DriftForecaster, NaiveForecaster, SeasonalNaiveForecaster
from helpers import BASELINES, make_panel, make_uuid


def _future(uid, start, n, freq="MS"):
    ds = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"unique_id": uid, "ds": ds, "horizon_step": range(1, n + 1)})


def _train(n=6):
    df = make_panel(1, n, internal=True)
    df["is_filled"] = False
    return df


def test_naive_repeats_last_value():
    m = NaiveForecaster().fit(_train())
    out = m.predict(_future(make_uuid(0), "2024-07-01", 3))
    assert out["yhat"].tolist() == [6.0, 6.0, 6.0]


def test_drift_continues_line():
    m = DriftForecaster().fit(_train())  # y = 1..6, slope 1
    out = m.predict(_future(make_uuid(0), "2024-07-01", 3))
    assert out["yhat"].tolist() == [7.0, 8.0, 9.0]


def test_seasonal_naive_repeats_last_season():
    m = SeasonalNaiveForecaster(season_length=3).fit(_train())  # last season 4,5,6
    out = m.predict(_future(make_uuid(0), "2024-07-01", 5))
    assert out["yhat"].tolist() == [4.0, 5.0, 6.0, 4.0, 5.0]


def test_seasonal_naive_rejects_bad_season():
    with pytest.raises(ValueError):
        SeasonalNaiveForecaster(season_length=0).fit(_train())


def test_predict_unknown_series_fails():
    m = NaiveForecaster().fit(_train())
    with pytest.raises(ValueError):
        m.predict(_future("other", "2024-07-01", 2))


def test_params_and_clone():
    m = SeasonalNaiveForecaster(season_length=4)
    assert m.get_params() == {"season_length": 4}
    assert NaiveForecaster().get_params() == {}
    c = m.clone()
    assert c is not m and c.season_length == 4
    c.set_params(season_length=7)
    assert m.season_length == 4
    with pytest.raises(ValueError):
        m.set_params(nope=1)


def test_metrics_values():
    y, p = np.array([2.0, 4.0]), np.array([1.0, 6.0])
    assert get_metric("mae")(y, p) == pytest.approx(1.5)
    assert get_metric("rmse")(y, p) == pytest.approx(np.sqrt(2.5))
    assert get_metric("mape")(y, p) == pytest.approx(50.0)
    assert get_metric("wape")(y, p) == pytest.approx(50.0)
    assert get_metric("smape")(y, p) == pytest.approx(100 * (2 / 3 + 0.4) / 2)
    # mase: mae 1.5 divided by mean |diff| of the training data (lag 1)
    assert get_metric("mase")(y, p, np.array([1.0, 2.0, 4.0]), 1) == pytest.approx(1.5 / 1.5)


def test_metrics_undefined_is_nan():
    zeros = np.array([0.0, 0.0])
    assert np.isnan(get_metric("mape")(zeros, np.array([1.0, 1.0])))


def test_unknown_metric_fails():
    with pytest.raises((ValueError, KeyError)):
        get_metric("nope")


def test_all_registered_metrics_zero_on_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    train = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    for name in METRICS:
        assert get_metric(name)(y, y.copy(), train, 1) == pytest.approx(0.0), name


def test_catalog_builds_and_skips_seasonal_for_yearly():
    specs = load_model_specs(types=BASELINES)
    assert set(build_models(specs, 12)) == {"naive", "seasonal_naive", "drift"}
    assert set(build_models(specs, 1)) == {"naive", "drift"}
    assert build_models(specs, 12)["seasonal_naive"].season_length == 12


def test_catalog_rejects_unknown_type(tmp_path):
    (tmp_path / "models.yaml").write_text("a:\n  type: nope\n")
    with pytest.raises(ConfigError):
        load_model_specs(tmp_path)


def test_catalog_rejects_bad_params():
    with pytest.raises(ConfigError):
        build_models({"n": {"type": "naive", "params": {"x": 1}}}, 1)
