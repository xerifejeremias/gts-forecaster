"""Missing data handling (see OVERVIEW.md, "Missing data handling").

A period is missing if its row is absent or its target / an explanatory value is null.
For every run of consecutive missing periods in a series:

  - at the end of the series, run <= max_fill_periods: forward fill
  - in the middle of the series, run <= max_fill_periods: linear interpolation
  - longer run (anywhere): the series gets a fallback forecast and no model is applied

All series are brought to one common cutoff, the latest date in the data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, IS_FILLED, TARGET
from gts_forecaster.contracts.errors import DataContractError

logger = logging.getLogger(__name__)

FALLBACKS = ("last_value", "zero")

REPORT_COLUMNS = [ID, "gap_type", "first_missing", "last_missing", "n_periods", "action"]
FALLBACK_COLUMNS = [ID, "last_observed_date", "last_observed_value", "fallback_value", "reason"]


@dataclass
class MissingDataResult:
    """Output of handle_missing_data."""

    data: pd.DataFrame  # complete series for the model pipeline, with an `is_filled` column
    fallback: pd.DataFrame  # series that get a fallback forecast instead of models
    report: pd.DataFrame  # one row per missing run, with the action taken
    cutoff: pd.Timestamp  # common last date of all series


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end positions (inclusive) of each run of True in a boolean array."""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def split_short_series(
    df: pd.DataFrame, min_series_length: int, target_column: str = TARGET
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the series with fewer than `min_series_length` rows (decided by the user
    2026-10-11: a brand-new series must not stop the run). They get the last observed target as
    their forecast for the whole horizon, with a warning, and never reach a model.

    `df` is in the model format (unique_id, ds, y, ...). Returns (the other series, a fallback
    table with FALLBACK_COLUMNS)."""
    counts = df.groupby(ID, sort=False)[DATE].transform("size")
    short = df[counts < min_series_length]
    rows: list[dict] = []
    for uuid, group in short.groupby(ID, sort=False):
        observed = group.sort_values(DATE).set_index(DATE)[target_column].dropna()
        if observed.empty:
            raise DataContractError(f"Series {uuid} has no observed target value.")
        value = float(observed.iloc[-1])
        reason = f"only {len(group)} row(s), fewer than min_series_length={min_series_length}"
        rows.append({
            ID: uuid, "last_observed_date": observed.index[-1], "last_observed_value": value,
            "fallback_value": value, "reason": reason,
        })
        logger.warning(
            "uuid %s: %s, so no model is applied. The forecast will be the last observed value "
            "(%s, %s) for every horizon step.", uuid, reason, value, f"{observed.index[-1]:%Y-%m-%d}",
        )
    rest = df[counts >= min_series_length].reset_index(drop=True)
    return rest, pd.DataFrame(rows, columns=FALLBACK_COLUMNS)


def handle_missing_data(
    df: pd.DataFrame,
    *,
    pandas_freq: str,
    value_columns: Sequence[str],
    target_column: str = TARGET,
    max_fill_periods: int = 3,
    fallback: str = "last_value",
) -> MissingDataResult:
    """Repair missing periods and split off the series that need a fallback forecast.

    `df` must have columns `unique_id`, `ds` and `value_columns` (model names), and have passed the
    structural input contract (unique, aligned dates).
    """
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {list(FALLBACKS)}, got {fallback!r}")
    if max_fill_periods < 0:
        raise ValueError("max_fill_periods must be >= 0")

    value_columns = list(value_columns)
    cutoff = df[DATE].max()
    cleaned: list[pd.DataFrame] = []
    fallback_rows: list[dict] = []
    report_rows: list[dict] = []

    for uuid, group in df.groupby(ID, sort=False):
        series = group.sort_values(DATE).set_index(DATE)[value_columns]
        observed_target = series[target_column].dropna()
        complete = series.notna().all(axis=1)
        if observed_target.empty or not complete.any():
            raise DataContractError(
                f"Series {uuid} has no period where the target and all explanatory values are present."
            )

        # Rows before the first complete period carry no usable information: the series starts later.
        series = series.loc[complete[complete].index[0] :]
        index = pd.date_range(series.index[0], cutoff, freq=pandas_freq)
        series = series.reindex(index)
        missing = series.isna().any(axis=1).to_numpy()

        needs_fallback = False
        for first, last in _runs(missing):
            n = last - first + 1
            gap_type = "trailing" if last == len(index) - 1 else "interior"
            if n > max_fill_periods:
                action = "fallback"
                needs_fallback = True
            else:
                action = "forward_fill" if gap_type == "trailing" else "interpolate"
            report_rows.append(
                {
                    ID: uuid,
                    "gap_type": gap_type,
                    "first_missing": index[first],
                    "last_missing": index[last],
                    "n_periods": n,
                    "action": action,
                }
            )
            logger.warning(
                "uuid %s: %d missing period(s) %s from %s to %s -> %s",
                uuid,
                n,
                "at the end" if gap_type == "trailing" else "in the middle",
                f"{index[first]:%Y-%m-%d}",
                f"{index[last]:%Y-%m-%d}",
                {
                    "forward_fill": "will be forward filled",
                    "interpolate": "will be linearly interpolated",
                    "fallback": f"longer than max_fill_periods={max_fill_periods}",
                }[action],
            )

        if needs_fallback:
            last_value = float(observed_target.iloc[-1])
            value = last_value if fallback == "last_value" else 0.0
            reason = f"missing run longer than max_fill_periods={max_fill_periods}"
            fallback_rows.append(
                {
                    ID: uuid,
                    "last_observed_date": observed_target.index[-1],
                    "last_observed_value": last_value,
                    "fallback_value": value,
                    "reason": reason,
                }
            )
            logger.warning(
                "uuid %s: data is missing (%s), so no model is applied. "
                "The forecast will be %s.",
                uuid,
                reason,
                f"the last observed value ({value})" if fallback == "last_value" else "0",
            )
            continue

        filled = series
        if missing.any():
            filled = series.interpolate(method="linear", limit_area="inside").ffill()
        out = filled.astype(float)
        out[IS_FILLED] = missing
        out.index.name = DATE
        out = out.reset_index()
        out.insert(0, ID, uuid)
        cleaned.append(out)

    if cleaned:
        data = pd.concat(cleaned, ignore_index=True)
        data[DATE] = data[DATE].astype("datetime64[ns]")  # one resolution on every pandas version
    else:
        data = pd.DataFrame(columns=[ID, DATE, *value_columns, IS_FILLED])
    return MissingDataResult(
        data=data,
        fallback=pd.DataFrame(fallback_rows, columns=FALLBACK_COLUMNS),
        report=pd.DataFrame(report_rows, columns=REPORT_COLUMNS),
        cutoff=cutoff,
    )
