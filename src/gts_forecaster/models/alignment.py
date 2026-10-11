"""Matching the forecast table of a library to the rows of `future`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID
from gts_forecaster.models.base import HORIZON_STEP

_STEP = "_gts_step"
_MODEL_DATE = "_gts_model_ds"
_MODEL_VALUE = "_gts_yhat"


def align_forecast(
    future: pd.DataFrame, forecast: pd.DataFrame, *, value_column: str, owner: str, freq: str
) -> np.ndarray:
    """The forecast value for each row of `future`, in the order of `future`.

    `forecast` has `unique_id`, `ds` and `value_column`, with the steps 1, 2, ... of every series (rows
    are sorted by date within a series). Step h of a series is matched with the row of `future` that has `horizon_step` h, and the
    dates must agree: a frequency or anchor mismatch is an error, never a shifted forecast.
    """
    if ID not in forecast.columns:  # older library versions return unique_id as the index
        forecast = forecast.reset_index()
    forecast = forecast[[ID, DATE, value_column]].sort_values([ID, DATE], kind="stable").rename(
        columns={DATE: _MODEL_DATE, value_column: _MODEL_VALUE}
    )
    forecast[_STEP] = forecast.groupby(ID, sort=False).cumcount() + 1

    wanted = future[[ID, DATE, HORIZON_STEP]].reset_index(drop=True)
    merged = wanted.merge(
        forecast, how="left", left_on=[ID, HORIZON_STEP], right_on=[ID, _STEP], validate="m:1"
    )
    missing = merged[_MODEL_VALUE].isna() & merged[_MODEL_DATE].isna()
    if missing.any():
        raise ValueError(f"{owner} returned no forecast for {int(missing.sum())} requested rows")
    wrong_date = pd.to_datetime(merged[DATE]) != pd.to_datetime(merged[_MODEL_DATE])
    if wrong_date.any():
        first = merged.loc[wrong_date].iloc[0]
        raise ValueError(
            f"{owner} forecasts dates that differ from the requested ones "
            f"(series {first[ID]}: expected {first[DATE]}, got {first[_MODEL_DATE]}). "
            f"Check that freq={freq!r} matches the data."
        )
    return merged[_MODEL_VALUE].to_numpy(dtype=float)
