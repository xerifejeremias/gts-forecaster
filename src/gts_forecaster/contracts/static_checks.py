"""Checks for the static (identifier) columns: market, product category, company and so on.

These columns describe a whole series, so each series (uuid) must have exactly one value in each
of them, and that value must be non-empty text.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from gts_forecaster.contracts.errors import DataContractError

_MAX_LISTED = 5


def check_static_columns(df: pd.DataFrame, static_columns: Sequence[str], id_column: str = "uuid") -> None:
    """Raise DataContractError if a static column is blank, not text, or changes within a series."""
    problems: list[str] = []
    for column in static_columns:
        values = df[column]
        blank = values.isna() | values.astype(str).str.strip().eq("")
        if blank.any():
            ids = df.loc[blank, id_column].drop_duplicates().head(_MAX_LISTED).tolist()
            problems.append(f"column {column!r} has empty values (uuid {', '.join(map(str, ids))})")
            continue
        changing = df.groupby(id_column, sort=False)[column].nunique(dropna=False)
        changing = changing[changing > 1]
        if len(changing):
            ids = changing.index[:_MAX_LISTED].tolist()
            problems.append(
                f"column {column!r} must have one value per uuid, but it changes for "
                f"{len(changing)} series (uuid {', '.join(map(str, ids))})"
            )
    if problems:
        raise DataContractError("Static columns do not match the contract: " + "; ".join(problems))


def extract_static_table(
    df: pd.DataFrame, static_columns: Sequence[str], id_column: str = "uuid"
) -> pd.DataFrame:
    """One row per series with its static values, in order of first appearance."""
    table = df[[id_column, *static_columns]].drop_duplicates(id_column).reset_index(drop=True)
    for column in static_columns:
        table[column] = table[column].astype(str).str.strip()
    return table
