"""Read the input file. Any failure to read it is a DataContractError."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from gts_forecaster.contracts.errors import DataContractError


def load_input_csv(path: Path, static_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Read an input CSV. Does not validate the content (that is the contract's job)."""
    path = Path(path)
    if not path.is_file():
        raise DataContractError(f"Input file not found: {path}")
    try:
        df = pd.read_csv(path, dtype={"uuid": str, **{c: str for c in static_columns}}, parse_dates=["date"])
    except Exception as exc:  # unreadable file, bad encoding, missing 'date' column, ...
        raise DataContractError(f"Could not read input file {path}: {exc}") from exc

    # Normalise date resolution so the contract sees one datetime type.
    # Time-zone aware dates are left untouched so the contract can reject them.
    if pd.api.types.is_datetime64_dtype(df["date"]):
        df["date"] = df["date"].astype("datetime64[ns]")
    return df
