"""Input contract rules that depend on the data frequency. Plain pandas, no pandera."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.contracts.errors import DataContractError

_MAX_LISTED = 10


def _listed(items: Sequence[object]) -> str:
    shown = ", ".join(str(i) for i in items[:_MAX_LISTED])
    extra = len(items) - _MAX_LISTED
    return shown + (f" ... and {extra} more" if extra > 0 else "")


def check_not_empty(df: pd.DataFrame) -> None:
    """The input must contain at least one row."""
    if df.empty:
        raise DataContractError("Input data is empty: it has no rows.")


def check_dates_aligned(df: pd.DataFrame, pandas_freq: str) -> None:
    """Every date must fall on the frequency grid (e.g. month starts for 'MS')."""
    dates = pd.DatetimeIndex(df["date"].unique())
    grid = pd.date_range(dates.min(), dates.max(), freq=pandas_freq)
    bad = dates[~dates.isin(grid)]
    if len(bad):
        offenders = df.loc[df["date"].isin(bad), "uuid"].drop_duplicates().tolist()
        raise DataContractError(
            f"{len(bad)} date(s) are not aligned to frequency '{pandas_freq}' "
            f"(first: {bad.min():%Y-%m-%d}). Affected uuid(s): {_listed(offenders)}"
        )


def check_min_length(df: pd.DataFrame, min_series_length: int) -> None:
    """Every series needs at least `min_series_length` rows. Short series are listed, not dropped."""
    counts = df.groupby("uuid", sort=False).size()
    short = counts[counts < min_series_length]
    if len(short):
        details = [f"{uuid} ({n} rows)" for uuid, n in short.items()]
        raise DataContractError(
            f"{len(short)} series have fewer than {min_series_length} rows: {_listed(details)}"
        )


def check_complete_panel(
    df: pd.DataFrame, pandas_freq: str, value_columns: Sequence[str], cutoff: pd.Timestamp
) -> None:
    """After missing data is repaired: no nulls, no gaps, and every series ends at the cutoff.

    Works on the model format (`unique_id`, `ds`, ...).
    """
    problems: list[str] = []

    null_counts = df[list(value_columns)].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        problems.append(
            "null values remain in: " + ", ".join(f"{c} ({n})" for c, n in null_counts.items())
        )

    bad_uuids = []
    for uuid, group in df.groupby(ID, sort=False):
        expected = pd.date_range(group[DATE].min(), cutoff, freq=pandas_freq)
        actual = pd.DatetimeIndex(group[DATE])
        if len(actual) != len(expected) or not actual.equals(expected):
            bad_uuids.append(uuid)
    if bad_uuids:
        problems.append(
            f"{len(bad_uuids)} series have gaps or do not end at the cutoff "
            f"{cutoff:%Y-%m-%d}: {_listed(bad_uuids)}"
        )

    if problems:
        raise DataContractError("Data is not complete after missing-data handling: " + "; ".join(problems))


def check_model_frame(df: pd.DataFrame, explanatory_columns: Sequence[str]) -> None:
    """Final shape of the data handed to the models.

    Columns are exactly unique_id, ds, y, the explanatory columns and is_filled; unique_id is text,
    ds is a timezone-free datetime, y and the explanatory columns are floats; rows are sorted by
    unique_id (in order of first appearance) and ds; there are no nulls.
    """
    expected = [ID, DATE, TARGET, *explanatory_columns, "is_filled"]
    problems: list[str] = []
    if list(df.columns) != expected:
        problems.append(f"columns are {list(df.columns)}, expected {expected}")
    else:
        if not (pd.api.types.is_string_dtype(df[ID]) or df[ID].dtype == object):
            problems.append(f"{ID} must be text, got {df[ID].dtype}")
        if str(df[DATE].dtype) != "datetime64[ns]":
            problems.append(f"{DATE} must be datetime64[ns] without time zone, got {df[DATE].dtype}")
        for column in [TARGET, *explanatory_columns]:
            if df[column].dtype != "float64":
                problems.append(f"{column} must be float64, got {df[column].dtype}")
        if df["is_filled"].dtype != bool:
            problems.append(f"is_filled must be bool, got {df['is_filled'].dtype}")
        if df.isna().any().any():
            problems.append("contains null values")
        if df.duplicated(subset=[ID, DATE]).any():
            problems.append(f"duplicate ({ID}, {DATE}) rows")
        order = df[ID].drop_duplicates().tolist()
        ordered = df.assign(_rank=df[ID].map({u: i for i, u in enumerate(order)}))
        if not ordered.sort_values(["_rank", DATE]).index.equals(df.index):
            problems.append(f"rows are not sorted by {ID} and {DATE}")
    if problems:
        raise DataContractError("Model data frame is not in the expected shape: " + "; ".join(problems))
