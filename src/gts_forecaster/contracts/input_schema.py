"""Strict input data contract, step 1: structure. Uses pandera.

Missing periods (absent rows, or null target/explanatory values) are NOT rejected here.
They are repaired later by cleaning.missing, and the result is checked again with
frequency_checks.check_complete_panel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pandera.pandas as pa

from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.static_checks import check_static_columns
from gts_forecaster.contracts.frequency_checks import (
    check_dates_aligned,
    check_min_length,
    check_not_empty,
)

UUID_REGEX = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
_MAX_FAILURES_SHOWN = 5


ID_FORMATS = ("any", "uuid")


def build_input_schema(
    explanatory_columns: Sequence[str], static_columns: Sequence[str] = (), id_format: str = "any"
) -> pa.DataFrameSchema:
    """Schema for the loaded input: exactly these columns, with these types."""
    no_inf = pa.Check(
        lambda s: ~np.isinf(s),
        name="no_infinite_values",
        error="contains infinite values",
    )

    def numeric() -> pa.Column:
        # Numeric columns may contain nulls here (repaired later); ints are accepted as floats.
        return pa.Column(float, checks=no_inf, nullable=True, coerce=True)

    columns: dict[str, pa.Column] = {
        "uuid": pa.Column(
            str, nullable=False,
            # any non-empty text by default; id_format "uuid" also requires the UUID format
            checks=pa.Check.str_matches(UUID_REGEX) if id_format == "uuid" else pa.Check.str_length(min_value=1),
        ),
        "date": pa.Column("datetime64[ns]", nullable=False),
        "target": numeric(),
    }
    for name in explanatory_columns:
        columns[name] = numeric()
    for name in static_columns:  # identifiers such as market, product category, company
        columns[name] = pa.Column(str, nullable=False)

    return pa.DataFrameSchema(columns, strict=True, coerce=False)


def _format_failures(exc: Exception) -> str:
    cases = getattr(exc, "failure_cases", None)
    if not isinstance(cases, pd.DataFrame) or not {"column", "check", "failure_case"} <= set(cases.columns):
        return f"  - {exc}"
    lines = []
    for (column, check), group in cases.groupby(["column", "check"], dropna=False, sort=False):
        examples = ", ".join(repr(v) for v in group["failure_case"].head(_MAX_FAILURES_SHOWN))
        more = len(group) - _MAX_FAILURES_SHOWN
        suffix = f" ... and {more} more" if more > 0 else ""
        lines.append(f"  - column {column!r}: {check} (e.g. {examples}{suffix})")
    return "\n".join(lines)


def _check_no_duplicates(df: pd.DataFrame) -> None:
    """Each (uuid, date) pair may appear only once."""
    duplicated = df.duplicated(subset=["uuid", "date"], keep=False)
    if duplicated.any():
        first = df.loc[duplicated, ["uuid", "date"]].iloc[0]
        n = int(df.duplicated(subset=["uuid", "date"]).sum())
        raise DataContractError(
            f"{n} duplicate (uuid, date) row(s), for example uuid {first['uuid']} "
            f"on {first['date']:%Y-%m-%d}"
        )


def validate_input_structure(
    df: pd.DataFrame,
    explanatory_columns: Sequence[str],
    pandas_freq: str,
    min_series_length: int,
    static_columns: Sequence[str] = (),
    id_format: str = "any",
) -> pd.DataFrame:
    """Check the loaded input against the contract. Returns the validated frame.

    Raises DataContractError on the first group of violations.
    """
    check_not_empty(df)
    schema = build_input_schema(explanatory_columns, static_columns, id_format)
    try:
        validated = schema.validate(df, lazy=True)
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as exc:
        raise DataContractError(
            "Input data does not match the contract:\n" + _format_failures(exc)
        ) from exc
    _check_no_duplicates(validated)
    check_static_columns(validated, static_columns)
    check_dates_aligned(validated, pandas_freq)
    check_min_length(validated, min_series_length)
    return validated
