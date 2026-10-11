"""Cutoff-based train/test split, shared by the backtest and the forecast pipeline.

A split is defined by a cutoff date and a forecast horizon:

  train   every row with ds <= cutoff
  test    the next `horizon` periods after the cutoff (empty for a real forecast, where
          there is nothing after the cutoff yet)

The horizon is always an argument given by the caller (normally `config.horizon`). Nothing in
this module has a default horizon, and it is not tied to a frequency.

  - Forecast pipeline:  `forecast_split(...)`: the cutoff is the last date in the data.
  - Backtest pipeline:  `backtest_splits(...)`: several cutoffs, rolling back from the end.

The data is in the model format (unique_id, ds, y, ...) and is complete: it has passed the
input contract and the missing-data step.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from gts_forecaster.columns import DATE, ID

if TYPE_CHECKING:  # only for type hints; avoids importing the pandera based input contract
    from gts_forecaster.data_preparation import PreparedInput

HORIZON_STEP = "horizon_step"


class SplitError(ValueError):
    """The requested split cannot be made from this data."""


@dataclass(frozen=True)
class Split:
    """One train/test split."""

    cutoff: pd.Timestamp  # last date of the training data
    horizon: int  # number of periods after the cutoff
    train: pd.DataFrame  # rows with ds <= cutoff, all columns of the input
    test: pd.DataFrame  # rows in the horizon window, plus `horizon_step`; empty for a forecast
    future: pd.DataFrame  # unique_id, ds, horizon_step for every trained series and horizon date
    horizon_dates: pd.DatetimeIndex  # the `horizon` dates after the cutoff
    skipped_ids: tuple[str, ...]  # series with no data at or before the cutoff
    fold: int | None = None  # 1 = oldest backtest fold; None for a forecast split

    @property
    def is_forecast(self) -> bool:
        """True when there is no test window (a real forecast, not a backtest fold)."""
        return self.test.empty

    @property
    def train_lengths(self) -> pd.Series:
        """Number of training observations per series."""
        return self.train.groupby(ID, sort=False).size()


def check_horizon(horizon: object) -> int:
    """The horizon must be an integer >= 1."""
    return _check_positive_int("horizon", horizon)


def _check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise SplitError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _check_panel(df: pd.DataFrame) -> None:
    missing = [c for c in (ID, DATE) if c not in df.columns]
    if missing:
        raise SplitError(f"data is missing column(s) {missing}; expected the model format")
    if df.empty:
        raise SplitError("data is empty")


def _horizon_dates(cutoff: pd.Timestamp, horizon: int, pandas_freq: str) -> pd.DatetimeIndex:
    """The `horizon` dates that follow the cutoff on the frequency grid."""
    return pd.date_range(cutoff, periods=horizon + 1, freq=pandas_freq)[1:]


def _future_frame(ids: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    steps = pd.Series(range(1, len(dates) + 1), name=HORIZON_STEP)
    grid = pd.DataFrame({DATE: dates, HORIZON_STEP: steps.to_numpy()})
    frames = [grid.assign(**{ID: uid}) for uid in ids]
    if not frames:
        return pd.DataFrame(columns=[ID, DATE, HORIZON_STEP])
    return pd.concat(frames, ignore_index=True)[[ID, DATE, HORIZON_STEP]]


def _build_split(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    pandas_freq: str,
    with_test: bool,
    fold: int | None,
) -> Split:
    train = df.loc[df[DATE] <= cutoff]
    if train.empty:
        raise SplitError(f"no training data at or before the cutoff {cutoff:%Y-%m-%d}")
    trained_ids = train[ID].drop_duplicates().tolist()
    trained = set(trained_ids)
    skipped = tuple(i for i in df[ID].drop_duplicates() if i not in trained)

    dates = _horizon_dates(cutoff, horizon, pandas_freq)
    future = _future_frame(trained_ids, dates)
    future[DATE] = future[DATE].astype(df[DATE].dtype)

    if with_test:
        # series keep their order of appearance in the data, then date
        test = df.merge(future, on=[ID, DATE], how="inner")
        order = {uid: k for k, uid in enumerate(trained_ids)}
        test = test.assign(_o=test[ID].map(order)).sort_values(["_o", DATE])
        test = test.drop(columns="_o").reset_index(drop=True)
    else:
        test = df.iloc[0:0].assign(**{HORIZON_STEP: pd.Series(dtype="int64")})

    return Split(
        cutoff=cutoff,
        horizon=horizon,
        train=train.reset_index(drop=True),
        test=test,
        future=future,
        horizon_dates=dates,
        skipped_ids=skipped,
        fold=fold,
    )


def split_at_cutoff(
    df: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    horizon: int,
    pandas_freq: str,
    fold: int | None = None,
) -> Split:
    """Train on everything up to `cutoff`, test on the next `horizon` periods.

    The cutoff must be a date on the frequency grid and the whole test window must lie inside
    the data. Series that have no data at or before the cutoff are listed in `skipped_ids`.
    """
    _check_panel(df)
    horizon = check_horizon(horizon)
    cutoff = pd.Timestamp(cutoff)
    grid = pd.date_range(df[DATE].min(), df[DATE].max(), freq=pandas_freq)
    if cutoff not in grid:
        raise SplitError(
            f"cutoff {cutoff:%Y-%m-%d} is not a date of the '{pandas_freq}' grid within the data "
            f"({grid[0]:%Y-%m-%d} to {grid[-1]:%Y-%m-%d})"
        )
    last_needed = _horizon_dates(cutoff, horizon, pandas_freq)[-1]
    if last_needed > df[DATE].max():
        raise SplitError(
            f"the test window for cutoff {cutoff:%Y-%m-%d} and horizon {horizon} ends on "
            f"{last_needed:%Y-%m-%d}, after the last date in the data ({df[DATE].max():%Y-%m-%d})"
        )
    return _build_split(df, cutoff, horizon, pandas_freq, with_test=True, fold=fold)


def forecast_split(df: pd.DataFrame, horizon: int, pandas_freq: str) -> Split:
    """Split for a real forecast: train on all data, with the cutoff at the last date.

    `test` is empty; `future` lists the dates to forecast for every series.
    """
    _check_panel(df)
    horizon = check_horizon(horizon)
    return _build_split(df, df[DATE].max(), horizon, pandas_freq, with_test=False, fold=None)


def backtest_cutoffs(
    df: pd.DataFrame,
    horizon: int,
    pandas_freq: str,
    n_folds: int,
    step: int | None = None,
    min_train_periods: int = 1,
) -> list[pd.Timestamp]:
    """Cutoffs for a rolling-origin backtest, oldest first.

    The newest cutoff leaves exactly `horizon` periods of test data at the end. Earlier cutoffs
    are `step` periods apart (default: `horizon`, so test windows do not overlap).
    `min_train_periods` is the least history the oldest fold must have, counted on the
    overall date range.
    """
    _check_panel(df)
    horizon = check_horizon(horizon)
    n_folds = _check_positive_int("n_folds", n_folds)
    step = horizon if step is None else _check_positive_int("step", step)
    min_train_periods = _check_positive_int("min_train_periods", min_train_periods)

    grid = pd.date_range(df[DATE].min(), df[DATE].max(), freq=pandas_freq)
    newest = len(grid) - 1 - horizon
    oldest = newest - step * (n_folds - 1)
    if oldest < min_train_periods - 1:
        raise SplitError(
            f"not enough history for {n_folds} fold(s) with horizon {horizon} and step {step}: "
            f"the data has {len(grid)} periods, but the oldest fold would leave only "
            f"{max(oldest + 1, 0)} period(s) of training data (minimum {min_train_periods})"
        )
    return [grid[oldest + step * k] for k in range(n_folds)]


def backtest_splits(
    df: pd.DataFrame,
    horizon: int,
    pandas_freq: str,
    n_folds: int,
    step: int | None = None,
    min_train_periods: int = 1,
) -> list[Split]:
    """One split per backtest cutoff, oldest first (`fold` = 1 is the oldest)."""
    cutoffs = backtest_cutoffs(df, horizon, pandas_freq, n_folds, step, min_train_periods)
    return [
        split_at_cutoff(df, cutoff, horizon, pandas_freq, fold=k)
        for k, cutoff in enumerate(cutoffs, start=1)
    ]


# --- helpers that take the output of data_preparation, with the horizon from the config -------


def forecast_split_from_prepared(
    prepared: PreparedInput, horizon: int | None = None
) -> Split:
    """`forecast_split` using the frequency and, unless overridden, the horizon of the config."""
    return forecast_split(
        prepared.data,
        prepared.config.horizon if horizon is None else horizon,
        prepared.config.profile.pandas_freq,
    )


def backtest_splits_from_prepared(
    prepared: PreparedInput,
    n_folds: int,
    step: int | None = None,
    horizon: int | None = None,
    min_train_periods: int = 1,
) -> list[Split]:
    """`backtest_splits` using the frequency and, unless overridden, the horizon of the config."""
    return backtest_splits(
        prepared.data,
        prepared.config.horizon if horizon is None else horizon,
        prepared.config.profile.pandas_freq,
        n_folds,
        step,
        min_train_periods,
    )
