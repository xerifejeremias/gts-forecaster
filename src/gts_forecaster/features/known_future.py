"""Explanatory variables whose future values the user knows.

Rule: an explanatory column is *unknown* in the future unless the user lists it in
`known_future_columns` and supplies its future values. By default none are known, and models
must then work without explanatory variables in the future window.

Supplying the values: a table (DataFrame or CSV) with
    date, <known columns>                  the same values for every series, or
    uuid, date, <known columns>            values per series.
It must cover every series and every forecast date, with no missing or infinite values. Extra
columns and rows outside the forecast window are ignored.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, RAW_DATE, RAW_ID
from gts_forecaster.contracts.errors import DataContractError

logger = logging.getLogger(__name__)
_MAX_LISTED = 5


def _read(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.is_file():
        raise DataContractError(f"Future features file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # unreadable or malformed file
        raise DataContractError(f"Could not read future features file {path}: {exc}") from exc


def load_known_future(
    source: pd.DataFrame | str | Path, known_columns: Sequence[str]
) -> pd.DataFrame:
    """Read and check the table. Returns unique_id (optional), ds and the known columns."""
    known_columns = list(known_columns)
    df = _read(source)
    problems: list[str] = []
    if RAW_DATE not in df.columns:
        problems.append(f"missing column '{RAW_DATE}'")
    missing = [c for c in known_columns if c not in df.columns]
    if missing:
        problems.append(f"missing known column(s) {missing}")
    if problems:
        raise DataContractError("Future features: " + "; ".join(problems))

    keep = ([RAW_ID] if RAW_ID in df.columns else []) + [RAW_DATE, *known_columns]
    ignored = [c for c in df.columns if c not in keep]
    if ignored:
        logger.warning("future features: ignoring column(s) %s (not in known_future_columns)", ignored)
    df = df[keep].rename(columns={RAW_ID: ID, RAW_DATE: DATE})

    try:
        df[DATE] = pd.to_datetime(df[DATE]).astype("datetime64[ns]")
    except (ValueError, TypeError) as exc:
        raise DataContractError(f"Future features: 'date' has unparseable values: {exc}") from exc
    for column in known_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    values = df[known_columns].to_numpy(dtype=float)
    if np.isnan(values).any() or np.isinf(values).any():
        raise DataContractError(
            "Future features: known columns contain missing, non-numeric or infinite values"
        )
    keys = [ID, DATE] if ID in df.columns else [DATE]
    if df.duplicated(keys).any():
        raise DataContractError(f"Future features: duplicate rows for {keys}")
    return df.reset_index(drop=True)


def attach_known_future(
    future: pd.DataFrame, known: pd.DataFrame, known_columns: Collection[str]
) -> pd.DataFrame:
    """Add the known columns to `future` (unique_id, ds, horizon_step). Stops on any gap."""
    known_columns = list(known_columns)
    if not known_columns:
        return future
    keys = [ID, DATE] if ID in known.columns else [DATE]
    out = future.merge(known[[*keys, *known_columns]], on=keys, how="left", validate="m:1")
    gaps = out[out[known_columns].isna().any(axis=1)]
    if len(gaps):
        sample = gaps[[ID, DATE]].head(_MAX_LISTED).assign(
            **{DATE: lambda d: d[DATE].dt.strftime("%Y-%m-%d")}
        )
        listed = ", ".join(f"{u} @ {d}" for u, d in zip(sample[ID], sample[DATE], strict=True))
        raise DataContractError(
            f"Future features do not cover {len(gaps)} forecast row(s) for {known_columns} "
            f"(first: {listed}). Provide values for every series and forecast date."
        )
    return out


def known_columns_from_test(
    future: pd.DataFrame, test: pd.DataFrame, known_columns: Collection[str]
) -> pd.DataFrame:
    """Backtest: the known columns' actual values in the test window stand in for the supplied
    future values (they are known in advance by definition)."""
    known_columns = list(known_columns)
    if not known_columns:
        return future
    return future.merge(test[[ID, DATE, *known_columns]], on=[ID, DATE], how="left", validate="1:1")
