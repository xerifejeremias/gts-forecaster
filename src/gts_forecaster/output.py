"""Post-processor: turn the internal forecast into the file for the receiving platform.

Internal result (forecasting.combine_forecasts): ds, unique_id, forecast.
Output (long format, one row per series and date):

    uuid, date, market, product, company, ..., forecast

The identifier (static) columns come from the input file, in the configured order. Column names
can be changed with `output_rename` in config/default.yaml, for example
{uuid: unique_id, date: ds, market: market_code}. The table is checked against the output
contract (contracts/output_schema.py) before it is written. Only the final (ensemble) forecast
goes into this file; the per-model forecasts stay in forecasts_by_model.csv.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import pandas as pd

from gts_forecaster.columns import DATE, ID
from gts_forecaster.contracts.output_schema import (
    OUT_DATE, OUT_FORECAST, OUT_ID, output_columns, validate_output,
)
from gts_forecaster.forecasting import FORECAST


def build_output(
    final: pd.DataFrame,
    static: pd.DataFrame | None,
    *,
    static_columns: Sequence[str] = (),
    expected_ids: Collection[str],
    horizon_dates: pd.DatetimeIndex,
    rename: Mapping[str, str] | None = None,
    forecast_floor: float | None = None,
) -> pd.DataFrame:
    """The validated output table, with the final column names."""
    out = final.rename(columns={ID: OUT_ID, DATE: OUT_DATE, FORECAST: OUT_FORECAST})
    if static_columns:
        if static is None or not len(static):
            raise ValueError("static_columns are configured but no static table was given")
        table = static.rename(columns={ID: OUT_ID})[[OUT_ID, *static_columns]]
        out = out.merge(table, on=OUT_ID, how="left")
    out = out[output_columns(static_columns)]
    out = out.sort_values([OUT_ID, OUT_DATE], kind="stable").reset_index(drop=True)
    validate_output(
        out, static_columns=static_columns, expected_ids=expected_ids,
        horizon_dates=horizon_dates, forecast_floor=forecast_floor,
    )
    return out.rename(columns=dict(rename or {}))


def write_output(df: pd.DataFrame, path: Path) -> None:
    """Write atomically; dates as YYYY-MM-DD."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    os.replace(tmp, path)
