"""Features built from the target's own past: lags and rolling statistics.

For a row at time t, `lag_k` is y[t-k], and `roll_<stat>_<w>` is the statistic of the w values that
end `rolling_lag` periods before t: y[t-L-w+1 .. t-L]. A row therefore never sees its own value
or anything later, so there is no leakage.

Forecasting several steps ahead: for the step h (h = 1 is the first forecast period) the value
y[t-k] is only known when k >= h. A feature that would need a value from inside the forecast
window is NaN in `future_target_features`, and `usable_at_step` tells which features are known
at each step. Models that predict directly should therefore use lags and a `rolling_lag` that are
at least the horizon. Models that predict step by step (recursively) build their own lags and do
not need these columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.features.errors import FeatureError
from gts_forecaster.models.base import HORIZON_STEP

ROLLING_STATS = ("mean", "std", "min", "max", "median")


def lag_name(k: int) -> str:
    return f"lag_{k}"


def roll_name(stat: str, window: int) -> str:
    return f"roll_{stat}_{window}"


def check_target_spec(lags, windows, stats, rolling_lag) -> None:
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in lags):
        raise FeatureError(f"lags must be integers >= 1, got {list(lags)}")
    if any(not isinstance(w, int) or isinstance(w, bool) or w < 2 for w in windows):
        raise FeatureError(f"rolling windows must be integers >= 2, got {list(windows)}")
    bad = [s for s in stats if s not in ROLLING_STATS]
    if bad:
        raise FeatureError(f"unknown rolling statistic(s) {bad}. Known: {list(ROLLING_STATS)}")
    if not isinstance(rolling_lag, int) or isinstance(rolling_lag, bool) or rolling_lag < 1:
        raise FeatureError(f"rolling_lag must be an integer >= 1, got {rolling_lag!r}")


def target_feature_names(lags, windows, stats) -> list[str]:
    return [lag_name(k) for k in lags] + [roll_name(s, w) for w in windows for s in stats]


def history_target_features(
    df: pd.DataFrame, lags, windows, stats, rolling_lag: int
) -> pd.DataFrame:
    """Lag and rolling columns for every row of `df` (model format), in the same row order.

    NaN where the series does not have enough history yet.
    """
    check_target_spec(lags, windows, stats, rolling_lag)
    ordered = df.sort_values([ID, DATE], kind="stable")
    by_series = ordered.groupby(ID, sort=False)[TARGET]
    columns: dict[str, pd.Series] = {}
    for k in lags:
        columns[lag_name(k)] = by_series.shift(k)
    shifted = by_series.shift(rolling_lag)
    grouped = shifted.groupby(ordered[ID], sort=False)
    for w in windows:
        rolling = grouped.rolling(w, min_periods=w)
        for stat in stats:
            columns[roll_name(stat, w)] = (
                getattr(rolling, stat)().reset_index(level=0, drop=True)
            )
    table = pd.DataFrame(columns, index=ordered.index)
    return table.reindex(df.index)


def future_target_features(
    train: pd.DataFrame, future: pd.DataFrame, lags, windows, stats, rolling_lag: int
) -> pd.DataFrame:
    """The same columns for the rows of `future` (unique_id, ds, horizon_step), from `train` only.

    Values that would lie inside the forecast window are NaN.
    """
    check_target_spec(lags, windows, stats, rolling_lag)
    history = {
        uid: g.sort_values(DATE, kind="stable")[TARGET].to_numpy(dtype=float)
        for uid, g in train.groupby(ID, sort=False)
    }
    names = target_feature_names(lags, windows, stats)
    out = np.full((len(future), len(names)), np.nan)
    ids = future[ID].to_numpy()
    steps = future[HORIZON_STEP].to_numpy()
    for row, (uid, h) in enumerate(zip(ids, steps, strict=True)):
        y = history.get(uid)
        if y is None:
            continue
        n = len(y)
        col = 0
        for k in lags:
            idx = n - 1 + h - k  # position of y[t-k] in the history
            if k >= h and idx >= 0:
                out[row, col] = y[idx]
            col += 1
        end = n - 1 + h - rolling_lag  # last value of the rolling window
        for w in windows:
            window = y[end - w + 1 : end + 1] if rolling_lag >= h and end - w + 1 >= 0 else None
            for stat in stats:
                if window is not None:
                    out[row, col] = _stat(window, stat)
                col += 1
    return pd.DataFrame(out, columns=names, index=future.index)


def _stat(window: np.ndarray, stat: str) -> float:
    if stat == "std":
        return float(np.std(window, ddof=1))  # same as pandas
    return float(getattr(np, stat)(window))


def usable_at_step(lags, windows, stats, rolling_lag: int, step: int) -> list[str]:
    """Names of the target features whose values are known at forecast step `step`."""
    names = [lag_name(k) for k in lags if k >= step]
    if rolling_lag >= step:
        names += [roll_name(s, w) for w in windows for s in stats]
    return names
