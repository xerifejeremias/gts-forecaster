"""Performance metrics. Pick one by name in the config; every metric is "lower is better".

A metric compares the actual and forecast values of ONE series over ONE backtest window and
returns a single number (or NaN when it is not defined, for example MAPE with an actual of 0).

    metric(y_true, y_pred, y_train, season_length) -> float

`y_train` and `season_length` are only used by scaled metrics such as MASE.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]


def mae(y_true, y_pred, y_train=None, season_length=1) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred, y_train=None, season_length=1) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, y_train=None, season_length=1) -> float:
    """Mean absolute percentage error, in percent. NaN if any actual value is 0."""
    if np.any(y_true == 0):
        return float("nan")
    return float(100 * np.mean(np.abs((y_true - y_pred) / y_true)))


def smape(y_true, y_pred, y_train=None, season_length=1) -> float:
    """Symmetric MAPE, in percent. A point where actual and forecast are both 0 counts as 0 error."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(
        2 * np.abs(y_true - y_pred), denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0
    )
    return float(100 * np.mean(ratio))


def wape(y_true, y_pred, y_train=None, season_length=1) -> float:
    """Weighted absolute percentage error, in percent: total absolute error / total actual."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float("nan")
    return float(100 * np.sum(np.abs(y_true - y_pred)) / total)


def mase(y_true, y_pred, y_train=None, season_length=1) -> float:
    """Mean absolute scaled error: MAE divided by the in-sample MAE of the seasonal naive model.

    Falls back to a lag of 1 when the training data is not longer than one season. NaN if the
    scale is 0 or cannot be computed.
    """
    if y_train is None:
        return float("nan")
    lag = season_length if len(y_train) > season_length else 1
    if len(y_train) <= lag:
        return float("nan")
    scale = np.mean(np.abs(y_train[lag:] - y_train[:-lag]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


METRICS: dict[str, MetricFn] = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
    "wape": wape,
    "mase": mase,
}


def get_metric(name: str) -> MetricFn:
    """Look up a metric by name (case does not matter)."""
    try:
        return METRICS[str(name).lower()]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}. Choose one of {sorted(METRICS)}") from None
