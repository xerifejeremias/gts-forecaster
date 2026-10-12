"""Read the input data: CSV, Parquet or an SQL query. Any failure to read it is a DataContractError.

`load_input(config)` is the single way in. It reads the source chosen by `data_source` (auto: from the
file suffix, or SQL when a query is configured), renames the configured input columns to the pipeline's
input names (`input_columns`: id -> uuid, date -> date, target -> target), drops columns the pipeline does
not use (or stops, `extra_columns: error`) and normalises the types. It does not validate the content:
that is the contract's job (contracts/input_schema.py).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from gts_forecaster.columns import RAW_DATE, RAW_ID, RAW_TARGET
from gts_forecaster.contracts.errors import DataContractError

logger = logging.getLogger(__name__)

DATA_SOURCES = ("auto", "csv", "parquet", "sql")
INPUT_ROLES = {"id": RAW_ID, "date": RAW_DATE, "target": RAW_TARGET}
_PARQUET_SUFFIXES = (".parquet", ".pq")


def resolve_source(config) -> str:
    """csv | parquet | sql for this run."""
    if config.data_source != "auto":
        return config.data_source
    if config.sql_query or config.sql_query_file:
        return "sql"
    return "parquet" if Path(config.data_path).suffix.lower() in _PARQUET_SUFFIXES else "csv"


def sql_query_text(config) -> str:
    """The configured query ({frequency} replaced)."""
    if config.sql_query:
        text = config.sql_query
    elif config.sql_query_file:
        path = Path(config.sql_query_file)
        if not path.is_file():
            raise DataContractError(f"SQL query file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        raise DataContractError("data_source is sql, but neither sql_query nor sql_query_file is set")
    return text.replace("{frequency}", config.frequency)


def _sql_url(config) -> str:
    url = config.sql_connection or os.environ.get(config.sql_connection_env)
    if not url:
        raise DataContractError(
            f"no SQL connection: set sql_connection in the config or the environment variable "
            f"{config.sql_connection_env} (a SQLAlchemy URL, e.g. postgresql://user:password@host/db)"
        )
    return url


def masked_url(url: str) -> str:
    """The connection URL without its password, for logs and run records."""
    return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", url)


def _read(config, source: str) -> pd.DataFrame:
    names = dict(config.input_columns)
    as_text = {names["id"]: str, **{c: str for c in config.static_columns}}
    if source == "csv":
        path = Path(config.data_path)
        if not path.is_file():
            raise DataContractError(f"Input file not found: {path}")
        return pd.read_csv(path, dtype=as_text)
    if source == "parquet":
        path = Path(config.data_path)
        if not path.is_file():
            raise DataContractError(f"Input file not found: {path}")
        return pd.read_parquet(path)
    try:
        import sqlalchemy
    except ImportError as exc:  # pragma: no cover - installed with the project
        raise DataContractError("SQL input needs SQLAlchemy: `poetry install --extras sql`") from exc
    query, url = sql_query_text(config), _sql_url(config)
    logger.info("reading the input with an SQL query from %s", masked_url(url))
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.connect() as connection:
            return pd.read_sql(sqlalchemy.text(query), connection)
    finally:
        engine.dispose()


def _as_text(series: pd.Series) -> pd.Series:
    """Text values, keeping missing values missing (so the contract can report them)."""
    return series.where(series.isna(), series.astype(str))


def load_input(config) -> pd.DataFrame:
    """Read the configured input and return it with the pipeline's input column names."""
    source = resolve_source(config)
    try:
        df = _read(config, source)
    except DataContractError:
        raise
    except Exception as exc:  # unreadable file, bad encoding, SQL error, ...
        raise DataContractError(f"Could not read the input ({source}): {exc}") from exc

    names = dict(config.input_columns)
    missing = [f"{names[role]} ({role})" for role in INPUT_ROLES if names[role] not in df.columns]
    if missing:
        raise DataContractError(
            f"the input has no column(s) {missing}; set input_columns in the config to the names the "
            f"data uses. Columns found: {list(df.columns)[:20]}"
        )
    df = df.rename(columns={names[role]: raw for role, raw in INPUT_ROLES.items()})
    used = [RAW_ID, RAW_DATE, RAW_TARGET, *config.explanatory_columns, *config.static_columns]
    extra = [c for c in df.columns if c not in used]
    if extra:
        if config.extra_columns == "error":
            raise DataContractError(f"the input has columns the pipeline does not use: {extra}")
        logger.info("input: ignoring %d column(s) the pipeline does not use: %s", len(extra), extra[:10])
        df = df.drop(columns=extra)

    df[RAW_ID] = _as_text(df[RAW_ID])
    for column in config.static_columns:
        if column in df.columns:
            df[column] = _as_text(df[column])
    try:
        dates = pd.to_datetime(df[RAW_DATE])
    except Exception as exc:
        raise DataContractError(f"the date column cannot be read as dates: {exc}") from exc
    # One datetime resolution for the contract; time-zone aware dates are left for it to reject.
    df[RAW_DATE] = dates.astype("datetime64[ns]") if pd.api.types.is_datetime64_dtype(dates) else dates
    return df


def input_fingerprint(config) -> dict[str, Any]:
    """What a run read, for its manifest (files: see pipelines/run.py; SQL: the query and server)."""
    source = resolve_source(config)
    if source != "sql":
        return {"source": source}
    try:
        query = sql_query_text(config)
    except DataContractError as exc:
        return {"source": "sql", "error": str(exc)}
    url = config.sql_connection or os.environ.get(config.sql_connection_env) or ""
    return {"source": "sql", "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "connection": masked_url(url) if url else None}


def load_input_csv(path: Path, static_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Read a CSV that already uses the pipeline's input names (uuid, date, target)."""
    path = Path(path)
    if not path.is_file():
        raise DataContractError(f"Input file not found: {path}")
    try:
        df = pd.read_csv(path, dtype={"uuid": str, **{c: str for c in static_columns}}, parse_dates=["date"])
    except Exception as exc:  # unreadable file, bad encoding, missing 'date' column, ...
        raise DataContractError(f"Could not read input file {path}: {exc}") from exc
    if pd.api.types.is_datetime64_dtype(df["date"]):
        df["date"] = df["date"].astype("datetime64[ns]")
    return df
