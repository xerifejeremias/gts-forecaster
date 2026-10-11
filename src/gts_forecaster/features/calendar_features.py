"""Calendar features: known for any date, so they are available in the future too.

They are computed from the period each date stands for. By default the date marks the START of its
period (a monthly date 2024-03-01 stands for March 2024, a weekly date for the 7 days from that
date); set `period_anchor: end` in config/features.yaml if your dates mark the END of the period.

Which features exist depends on the frequency (for example `day_of_week` only means something for
daily data). A feature set can ask for `all` of them or for named ones; a named feature that does
not exist for the current frequency is skipped with a log message, so one feature set can serve
all four frequencies.

  n_days               days in the period
  n_business_days      Monday to Friday days in the period that are not holidays
  share_business_days  n_business_days / n_days
  n_holidays           holidays in the period (only when holidays are configured)
  n_mondays .. n_sundays   how many of each weekday the period has (monthly only)
  is_business_day      1 on a business day (daily only)
  is_holiday           1 on a holiday (daily only, only when holidays are configured)
  month, quarter       calendar month and quarter of the period start (not yearly)
  week_of_year         ISO week (weekly and daily)
  day_of_week          0 = Monday .. 6 = Sunday (daily)
  day_of_month, day_of_year, is_month_start, is_month_end   (daily)
  year_sin_k, year_cos_k   Fourier terms of the yearly cycle, k = 1..fourier_order (not yearly)
  week_sin_k, week_cos_k   Fourier terms of the weekly cycle (daily)
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from gts_forecaster.columns import DATE
from gts_forecaster.features.errors import FeatureError

logger = logging.getLogger(__name__)

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
ALL_FREQS = frozenset({"yearly", "monthly", "weekly", "daily"})

# name -> frequencies where the feature exists
CALENDAR_FEATURES: dict[str, frozenset[str]] = {
    "n_days": ALL_FREQS,
    "n_business_days": ALL_FREQS,
    "share_business_days": ALL_FREQS,
    "n_holidays": ALL_FREQS,
    **{f"n_{d}s": frozenset({"monthly"}) for d in _WEEKDAYS},
    "is_business_day": frozenset({"daily"}),
    "is_holiday": frozenset({"daily"}),
    "month": frozenset({"monthly", "weekly", "daily"}),
    "quarter": frozenset({"monthly", "weekly", "daily"}),
    "week_of_year": frozenset({"weekly", "daily"}),
    "day_of_week": frozenset({"daily"}),
    "day_of_month": frozenset({"daily"}),
    "day_of_year": frozenset({"daily"}),
    "is_month_start": frozenset({"daily"}),
    "is_month_end": frozenset({"daily"}),
}
_NEEDS_HOLIDAYS = {"n_holidays", "is_holiday"}


def frequency_kind(pandas_freq: str) -> str:
    """yearly | monthly | weekly | daily for a pandas offset alias."""
    name = type(to_offset(pandas_freq)).__name__
    for prefix, kind in (("Year", "yearly"), ("Month", "monthly"), ("Week", "weekly"), ("Day", "daily")):
        if name.startswith(prefix):
            return kind
    raise FeatureError(f"calendar features do not support the frequency {pandas_freq!r}")


def available_calendar_features(
    pandas_freq: str, has_holidays: bool = False, fourier_order: int = 0
) -> list[str]:
    """Names of all calendar features that exist for this frequency."""
    kind = frequency_kind(pandas_freq)
    names = [
        n for n, kinds in CALENDAR_FEATURES.items()
        if kind in kinds and (has_holidays or n not in _NEEDS_HOLIDAYS)
    ]
    return names + fourier_names(kind, fourier_order)


def fourier_names(kind: str, order: int) -> list[str]:
    names = []
    for k in range(1, order + 1):
        if kind in ("monthly", "weekly", "daily"):
            names += [f"year_sin_{k}", f"year_cos_{k}"]
        if kind == "daily":
            names += [f"week_sin_{k}", f"week_cos_{k}"]
    return names


def _period_bounds(
    dates: pd.DatetimeIndex, pandas_freq: str, anchor: str
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """First day of each period and the day after its last day."""
    offset = to_offset(pandas_freq)
    if anchor == "start":
        return dates, dates + offset
    if anchor == "end":
        return (dates - offset) + pd.Timedelta(days=1), dates + pd.Timedelta(days=1)
    raise FeatureError(f"period_anchor must be 'start' or 'end', got {anchor!r}")


def _days(index: pd.DatetimeIndex) -> np.ndarray:
    return index.to_numpy().astype("datetime64[D]")


def _count_in_period(days: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """How many of the (sorted, unique) `days` fall in each [start, end)."""
    if len(days) == 0:
        return np.zeros(len(start), dtype=int)
    return np.searchsorted(days, end, side="left") - np.searchsorted(days, start, side="left")


def _compute(
    name: str, kind: str, start: pd.DatetimeIndex, end: pd.DatetimeIndex, holidays: np.ndarray
) -> np.ndarray:
    s, e = _days(start), _days(end)
    n_days = (e - s).astype(int)
    if name == "n_days":
        return n_days
    if name == "n_business_days":
        return np.busday_count(s, e, holidays=holidays)
    if name == "share_business_days":
        return np.busday_count(s, e, holidays=holidays) / n_days
    if name == "n_holidays":
        return _count_in_period(holidays, s, e)
    if name.startswith("n_") and name.endswith("s") and name[2:-1] in _WEEKDAYS:
        mask = "".join("1" if d == name[2:-1] else "0" for d in _WEEKDAYS)
        return np.busday_count(s, e, weekmask=mask)
    if name == "is_business_day":
        return np.is_busday(s, holidays=holidays).astype(int)
    if name == "is_holiday":
        return _count_in_period(holidays, s, e)
    if name == "month":
        return start.month.to_numpy()
    if name == "quarter":
        return start.quarter.to_numpy()
    if name == "week_of_year":
        return start.isocalendar().week.to_numpy().astype(int)
    if name == "day_of_week":
        return start.dayofweek.to_numpy()
    if name == "day_of_month":
        return start.day.to_numpy()
    if name == "day_of_year":
        return start.dayofyear.to_numpy()
    if name == "is_month_start":
        return start.is_month_start.astype(int)
    if name == "is_month_end":
        return start.is_month_end.astype(int)
    raise FeatureError(f"unknown calendar feature {name!r}")


def _fourier(name: str, start: pd.DatetimeIndex) -> np.ndarray:
    cycle, kind, k = name.split("_")
    if cycle == "year":
        days_in_year = np.where(start.is_leap_year, 366.0, 365.0)
        phase = (start.dayofyear.to_numpy() - 1) / days_in_year
    else:
        phase = start.dayofweek.to_numpy() / 7.0
    angle = 2 * np.pi * int(k) * phase
    return np.sin(angle) if kind == "sin" else np.cos(angle)


def calendar_features(
    df: pd.DataFrame,
    *,
    pandas_freq: str,
    names: str | Sequence[str] = "all",
    fourier_order: int = 0,
    holidays: Collection[np.datetime64 | pd.Timestamp | str] = (),
    anchor: str = "start",
) -> pd.DataFrame:
    """The calendar feature columns for the dates in `df[ds]`, in the same row order.

    `names` is "all" or a list of feature names. The result has one column per feature, so
    it can be concatenated to `df`.
    """
    kind = frequency_kind(pandas_freq)
    holiday_days = np.unique(pd.to_datetime(list(holidays)).to_numpy().astype("datetime64[D]"))
    unknown = [n for n in ([] if names == "all" else names) if not _is_known(n)]
    if unknown:
        raise FeatureError(f"unknown calendar feature(s) {unknown}. Known: {sorted(CALENDAR_FEATURES)}")
    available = available_calendar_features(pandas_freq, bool(len(holiday_days)), fourier_order)
    wanted = (
        available if names == "all"
        else list(dict.fromkeys([*names, *fourier_names(kind, fourier_order)]))
    )
    skipped = [n for n in wanted if n not in available]
    if skipped:
        logger.info("calendar features not available for %s data, skipped: %s", kind, skipped)
    wanted = [n for n in wanted if n in available]

    unique_dates = pd.DatetimeIndex(pd.unique(df[DATE]))
    start, end = _period_bounds(unique_dates, pandas_freq, anchor)
    columns = {}
    for name in wanted:
        columns[name] = (
            _fourier(name, start) if _is_fourier(name) else _compute(name, kind, start, end, holiday_days)
        )
    table = pd.DataFrame(columns, index=unique_dates)
    return table.reindex(pd.DatetimeIndex(df[DATE])).reset_index(drop=True).set_axis(df.index)


def _is_fourier(name: str) -> bool:
    parts = name.split("_")
    return len(parts) == 3 and parts[0] in ("year", "week") and parts[1] in ("sin", "cos") and parts[2].isdigit()


def _is_known(name: str) -> bool:
    return name in CALENDAR_FEATURES or _is_fourier(name)
