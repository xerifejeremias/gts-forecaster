"""Output contract and post-processor: the long-format file for the receiving platform."""

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.output_schema import validate_output
from gts_forecaster.output import build_output, write_output

DATES = pd.date_range("2025-10-01", periods=3, freq="MS")
STATIC = ("market", "product", "company")


def _final(ids=("a", "b")):
    return pd.concat(
        [pd.DataFrame({"ds": DATES, "unique_id": u, "forecast": [1.0, 2.0, 3.0]}) for u in ids],
        ignore_index=True,
    )


def _static(ids=("a", "b")):
    return pd.DataFrame({"unique_id": list(ids), "market": "DE04", "product": "F0101", "company": "company A"})


def _build(final=None, static=None, ids=("a", "b"), **kw):
    return build_output(
        _final() if final is None else final, _static() if static is None else static,
        static_columns=STATIC, expected_ids=set(ids), horizon_dates=DATES, **kw,
    )


def test_long_format_with_static_columns_in_order():
    out = _build()
    assert list(out.columns) == ["uuid", "date", "market", "product", "company", "forecast"]
    assert len(out) == 6 and out["market"].eq("DE04").all()
    assert out["uuid"].tolist() == ["a"] * 3 + ["b"] * 3


def test_rename_changes_only_the_names():
    out = _build(rename={"uuid": "unique_id", "date": "ds", "market": "market_code"})
    assert list(out.columns) == ["unique_id", "ds", "market_code", "product", "company", "forecast"]


def test_without_static_columns():
    out = build_output(_final(), None, expected_ids={"a", "b"}, horizon_dates=DATES)
    assert list(out.columns) == ["uuid", "date", "forecast"]


def test_missing_series_is_rejected():
    with pytest.raises(DataContractError, match="without a forecast"):
        _build(ids=("a", "b", "c"))


def test_missing_horizon_date_is_rejected():
    final = _final().drop(index=2)
    with pytest.raises(DataContractError, match="horizon dates"):
        _build(final=final)


def test_nan_forecast_is_rejected():
    final = _final()
    final.loc[0, "forecast"] = np.nan
    with pytest.raises(DataContractError, match="empty values"):
        _build(final=final)


def test_missing_static_value_is_rejected():
    static = _static()
    static.loc[1, "market"] = None
    with pytest.raises(DataContractError, match="market"):
        _build(static=static)


def test_duplicate_rows_are_rejected():
    df = _build()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicated"):
        validate_output(df, static_columns=STATIC, expected_ids={"a", "b"}, horizon_dates=DATES)


def test_wrong_columns_are_rejected():
    df = _build().drop(columns="company")
    with pytest.raises(DataContractError, match="columns"):
        validate_output(df, static_columns=STATIC, expected_ids={"a", "b"}, horizon_dates=DATES)


def test_written_file_uses_iso_dates(tmp_path):
    path = tmp_path / "x" / "forecast_output.csv"
    write_output(_build(), path)
    text = path.read_text().splitlines()
    assert text[0] == "uuid,date,market,product,company,forecast"
    assert text[1].startswith("a,2025-10-01,DE04,F0101,company A,1.0")


def test_config_defaults_and_validation():
    cfg = load_config()
    assert cfg.output_file == "forecast_output.csv" and cfg.output_rename == ()
    cfg = load_config({"output_rename": {"uuid": "unique_id", "market": "market_code"}})
    assert dict(cfg.output_rename) == {"uuid": "unique_id", "market": "market_code"}
    for bad in (
        {"output_rename": {"nope": "x"}},  # not an output column
        {"output_rename": {"uuid": "forecast"}},  # duplicate name
        {"output_file": "../out.csv"},
        {"output_file": "out.txt"},
    ):
        with pytest.raises(ConfigError):
            load_config(bad)
