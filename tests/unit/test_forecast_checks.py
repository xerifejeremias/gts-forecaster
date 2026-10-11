"""Sanity checks of the final forecasts (forecast_checks.py)."""

import logging

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.cleaning.outliers import extension_scores
from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.forecast_checks import (
    ForecastCheckError, check_forecasts, check_series, report_flags,
)

M = 12


def _seasonal(n, level=100.0, amplitude=30.0, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return level + amplitude * np.sin(2 * np.pi * t / M) + rng.normal(0, noise, n)


def _dates(n):
    return pd.date_range("2025-01-01", periods=n, freq="MS")


def _check(history, forecast, **kwargs):
    options = {"season_length": M, "threshold": 5.0, "max_ratio": 3.0, **kwargs}
    return check_series("s", np.asarray(history, float), _dates(len(forecast)), np.asarray(forecast, float), **options)


def _checks(flags):
    return sorted({f["check"] for f in flags})


def test_plausible_seasonal_forecast_is_not_flagged():
    y = _seasonal(60 + M)
    assert _check(y[:60], y[60:]) == []


def test_flat_forecast_of_a_seasonal_series_is_not_an_outlier():
    y = _seasonal(60)
    assert _check(y, np.full(M, y[-M:].mean())) == []


def test_spike_in_the_forecast_is_an_outlier_at_its_date():
    y = _seasonal(60 + M)
    forecast = y[60:].copy()
    forecast[4] *= 4
    flags = _check(y[:60], forecast)
    assert _checks(flags) == ["outlier"]
    assert flags[0]["ds"] == _dates(M)[4] and flags[0]["score"] > 5


def test_forecast_at_the_wrong_level_is_flagged():
    y = _seasonal(60 + M)
    # same seasonal swings, level 100 -> 1000; and everything a tenth
    assert _checks(_check(y[:60], y[60:] + 900)) == ["start_jump", "total_change"]
    assert _checks(_check(y[:60], y[60:] / 10)) == ["start_jump", "total_change"]
    # swings ten times larger than ever seen are flagged as outliers as well
    assert "outlier" in _checks(_check(y[:60], y[60:] * 10))


def test_all_zero_forecast_after_positive_history_is_flagged():
    y = _seasonal(60)
    assert _checks(_check(y, np.zeros(M))) == ["zero"]
    assert _check(np.zeros(60), np.zeros(M)) == []  # a series that really is zero


def test_one_noisy_reference_alone_does_not_flag_start_jump():
    # a season ago the series was near zero, the last values are normal: not a wrong start
    y = _seasonal(60)
    y[60 - M : 60 - M + 3] = [5.0, -3.0, 4.0]
    assert "start_jump" not in _checks(_check(y, np.full(M, y[-3:].mean())))


def test_short_history_skips_the_outlier_check_but_not_the_ratios():
    history = np.array([100.0, 110, 90, 105])
    assert _check(history, np.full(3, 100.0), season_length=1) == []
    assert "start_jump" in _checks(_check(history, np.full(3, 1000.0), season_length=1))


def test_extension_scores_use_the_history_spread():
    history = _seasonal(40, amplitude=0)
    smooth = np.full(6, 100.0)
    assert np.nanmax(np.abs(extension_scores(history, smooth))) < 5
    assert np.isnan(extension_scores(history[:5], smooth)).all()  # too little history


def test_check_forecasts_runs_per_series_and_skips_fallback_series():
    y = _seasonal(60)
    history = pd.concat([
        pd.DataFrame({"unique_id": uid, "ds": pd.date_range("2020-01-01", periods=60, freq="MS"), "y": y})
        for uid in ("a", "b", "fb")
    ])
    future = pd.date_range("2025-01-01", periods=M, freq="MS")
    forecast = pd.concat([
        pd.DataFrame({"ds": future, "unique_id": "a", "forecast": y[-M:]}),
        pd.DataFrame({"ds": future, "unique_id": "b", "forecast": 0.0}),
        pd.DataFrame({"ds": future, "unique_id": "fb", "forecast": 0.0}),
    ])
    flags = check_forecasts(history, forecast, season_length=M, threshold=5.0, max_ratio=3.0, skip_ids={"fb"})
    assert set(flags["unique_id"]) == {"b"} and list(flags["check"]) == ["zero"]


def test_report_flags_warns_or_raises(caplog):
    flags = pd.DataFrame([{"unique_id": "b", "check": "zero", "ds": None, "value": 0.0, "reference": 9.0,
                           "score": None, "threshold": None, "message": "the whole forecast is 0"}])
    with caplog.at_level(logging.WARNING):
        report_flags(flags, "warn")
    assert "flagged 1 series" in caplog.text and "the whole forecast is 0" in caplog.text
    with pytest.raises(ForecastCheckError, match="Nothing was written"):
        report_flags(flags, "raise")
    report_flags(flags.iloc[:0], "raise")  # nothing flagged: no error


def test_check_settings_defaults_and_validation():
    cfg = load_config({"frequency": "monthly"})
    assert cfg.forecast_check_action == "warn" and cfg.forecast_check_max_ratio == 3.0
    assert cfg.forecast_check_threshold == cfg.outlier_threshold  # null in default.yaml
    assert load_config({"frequency": "monthly", "forecast_check_threshold": 7}).forecast_check_threshold == 7
    for bad in ({"forecast_check_action": "maybe"}, {"forecast_check_threshold": 0},
                {"forecast_check_max_ratio": 1}, {"forecast_check_max_ratio": True}):
        with pytest.raises(ConfigError, match="forecast_check"):
            load_config({"frequency": "monthly", **bad})
