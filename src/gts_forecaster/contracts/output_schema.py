"""Output contract: the final forecast file handed to the receiving platform.

Long format, one row per series and forecast date:

    uuid, date, <static columns, e.g. market, product, company>, forecast

(names before the optional `output_rename` from the config). Plain pandas checks, so the contract
does not depend on the validation library. Any breach is a DataContractError and nothing is
written.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

from gts_forecaster.contracts.errors import DataContractError

OUT_ID, OUT_DATE, OUT_FORECAST = "uuid", "date", "forecast"


def output_columns(static_columns: Sequence[str] = ()) -> list[str]:
    return [OUT_ID, OUT_DATE, *static_columns, OUT_FORECAST]


def validate_output(
    df: pd.DataFrame,
    *,
    static_columns: Sequence[str] = (),
    expected_ids: Collection[str],
    horizon_dates: pd.DatetimeIndex,
    forecast_floor: float | None = None,
) -> pd.DataFrame:
    """Check the output table (before renaming). Returns it unchanged if it passes."""
    problems: list[str] = []
    expected_columns = output_columns(static_columns)
    if list(df.columns) != expected_columns:
        raise DataContractError(f"output columns are {list(df.columns)}, expected {expected_columns}")

    nulls = [c for c in expected_columns if df[c].isna().any()]
    if nulls:
        problems.append(f"empty values in column(s) {nulls}")
    if not pd.api.types.is_numeric_dtype(df[OUT_FORECAST]):
        problems.append("forecast is not numeric")
    elif not np.isfinite(df[OUT_FORECAST].to_numpy(dtype=float, na_value=np.nan)).all():
        problems.append("forecast has infinite or missing values")
    elif forecast_floor is not None and (df[OUT_FORECAST] < forecast_floor).any():
        problems.append(f"{int((df[OUT_FORECAST] < forecast_floor).sum())} forecast(s) below {forecast_floor}")
    if not pd.api.types.is_datetime64_any_dtype(df[OUT_DATE]):
        problems.append("date is not a date column")
    if df.duplicated([OUT_ID, OUT_DATE]).any():
        problems.append(f"{int(df.duplicated([OUT_ID, OUT_DATE]).sum())} duplicated (uuid, date) row(s)")

    ids = set(df[OUT_ID])
    missing = sorted(set(expected_ids) - ids)
    extra = sorted(ids - set(expected_ids))
    if missing:
        problems.append(f"{len(missing)} series without a forecast, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected series, e.g. {extra[:3]}")

    expected_dates = set(pd.DatetimeIndex(horizon_dates))
    if pd.api.types.is_datetime64_any_dtype(df[OUT_DATE]):
        wrong = [u for u, g in df.groupby(OUT_ID, sort=False) if set(g[OUT_DATE]) != expected_dates or len(g) != len(expected_dates)]
        if wrong:
            problems.append(
                f"{len(wrong)} series do not have exactly the {len(expected_dates)} horizon dates, e.g. {wrong[:3]}"
            )
    for column in static_columns:
        varying = df.groupby(OUT_ID)[column].nunique()
        if (varying > 1).any():
            problems.append(f"column {column!r} changes within a series, e.g. {list(varying[varying > 1].index[:3])}")

    if problems:
        raise DataContractError("output contract failed: " + "; ".join(problems))
    return df
