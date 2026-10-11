"""Column names.

Input files use the user-facing names `uuid`, `date`, `target`. After the input contract has
passed, the data is renamed to the Nixtla names `unique_id`, `ds`, `y` that MLForecast and
StatsForecast expect. All code after that point uses the model names.
"""

from __future__ import annotations

import pandas as pd

# user-facing names, as they appear in the input file
RAW_ID, RAW_DATE, RAW_TARGET = "uuid", "date", "target"

# model names (Nixtla convention)
ID, DATE, TARGET = "unique_id", "ds", "y"

IS_FILLED = "is_filled"

RAW_TO_MODEL = {RAW_ID: ID, RAW_DATE: DATE, RAW_TARGET: TARGET}


def to_model_format(df: pd.DataFrame) -> pd.DataFrame:
    """Rename uuid/date/target to unique_id/ds/y. Explanatory columns keep their names."""
    return df.rename(columns=RAW_TO_MODEL)
