"""Small builders shared by the tests."""

from __future__ import annotations

import uuid

import pandas as pd

EXPLANATORY = ("x1", "x2")
# Model types that need no optional library; tests that build models from models.yaml use these.
BASELINES = ("naive", "seasonal_naive", "drift")


def make_uuid(i: int) -> str:
    return str(uuid.UUID(int=i + 1, version=4))


def make_panel(
    n_series: int = 2,
    n_periods: int = 8,
    pandas_freq: str = "MS",
    start: str = "2024-01-01",
    explanatory: tuple[str, ...] = EXPLANATORY,
    internal: bool = False,
) -> pd.DataFrame:
    """A valid, complete panel: target is 1, 2, 3, ... and each explanatory column is a multiple of it.

    Columns are uuid/date/target as in an input file, or unique_id/ds/y with `internal=True`.
    """
    frames = []
    for i in range(n_series):
        dates = pd.date_range(start, periods=n_periods, freq=pandas_freq)
        target = pd.Series(range(1, n_periods + 1), dtype=float).to_numpy() * (i + 1)
        frame = pd.DataFrame({"uuid": make_uuid(i), "date": dates, "target": target})
        for k, name in enumerate(explanatory, start=1):
            frame[name] = target * k
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    return panel.rename(columns={"uuid": "unique_id", "date": "ds", "target": "y"}) if internal else panel


def drop_rows(df: pd.DataFrame, uuid_value: str, dates: list[str]) -> pd.DataFrame:
    """Remove rows of one series, simulating missing periods."""
    id_col, date_col = ("unique_id", "ds") if "unique_id" in df.columns else ("uuid", "date")
    mask = (df[id_col] == uuid_value) & df[date_col].isin(pd.to_datetime(dates))
    return df.loc[~mask].reset_index(drop=True)


def static_table(ids, columns=("market", "product", "company")) -> pd.DataFrame:
    """One row per series with made-up identifier values (for pipeline tests)."""
    ids = list(dict.fromkeys(ids))
    return pd.DataFrame({"unique_id": ids, **{c: [f"{c}_{k % 2}" for k in range(len(ids))] for c in columns}})
