"""Step 1 of the input contract (pandera): structure. Every rule has a test that makes it fail."""

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.input_schema import validate_input_structure
from helpers import EXPLANATORY, make_panel


def validate(df, **kwargs):
    kwargs.setdefault("explanatory_columns", EXPLANATORY)
    kwargs.setdefault("pandas_freq", "MS")
    kwargs.setdefault("min_series_length", 3)
    return validate_input_structure(df, **kwargs)


def test_valid_input_passes_and_is_returned():
    df = make_panel()
    out = validate(df)
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns)


def test_integer_explanatory_columns_are_accepted_as_numbers():
    df = make_panel()
    df["x1"] = df["x1"].astype(int)
    out = validate(df)
    assert out["x1"].dtype == float


def test_nulls_in_numeric_columns_pass_here_and_are_repaired_later():
    df = make_panel()
    df.loc[3, "target"] = np.nan
    df.loc[5, "x2"] = np.nan
    validate(df)


def test_missing_rows_pass_here_and_are_repaired_later():
    df = make_panel().drop(index=[3, 4]).reset_index(drop=True)
    validate(df)


def test_missing_column_fails():
    with pytest.raises(DataContractError, match="x2"):
        validate(make_panel().drop(columns="x2"))


def test_missing_target_column_fails():
    with pytest.raises(DataContractError, match="target"):
        validate(make_panel().drop(columns="target"))


def test_unexpected_extra_column_fails():
    df = make_panel()
    df["surprise"] = 1.0
    with pytest.raises(DataContractError, match="surprise"):
        validate(df)


def test_text_in_a_numeric_column_fails():
    df = make_panel()
    df["target"] = df["target"].astype(object)
    df.loc[2, "target"] = "n/a"
    with pytest.raises(DataContractError, match="target"):
        validate(df)


def test_dates_stored_as_text_fail():
    df = make_panel()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    with pytest.raises(DataContractError, match="date"):
        validate(df)


def test_timezone_aware_dates_fail():
    df = make_panel()
    df["date"] = df["date"].dt.tz_localize("UTC")
    with pytest.raises(DataContractError, match="date"):
        validate(df)


def test_null_date_fails():
    df = make_panel()
    df.loc[1, "date"] = pd.NaT
    with pytest.raises(DataContractError, match="date"):
        validate(df)


def test_null_uuid_fails():
    df = make_panel()
    df.loc[1, "uuid"] = None
    with pytest.raises(DataContractError, match="uuid"):
        validate(df)


@pytest.mark.parametrize("bad", ["not-a-uuid", "1234", "", "g0000000-0000-4000-8000-000000000000"])
def test_malformed_uuid_fails(bad):
    df = make_panel()
    df.loc[0, "uuid"] = bad
    with pytest.raises(DataContractError, match="uuid"):
        validate(df, id_format="uuid")  # the UUID format is checked only when asked for
    validate(df.assign(uuid=df["uuid"].where(df.index != 0, df.loc[1, "uuid"])))  # control: valid ids pass


def test_any_text_id_is_accepted_by_default_but_not_empty():
    df = make_panel()
    df["uuid"] = df["uuid"].map({u: f"DE04_F0101_{k}" for k, u in enumerate(df["uuid"].unique())})
    validate(df)
    df.loc[0, "uuid"] = ""
    with pytest.raises(DataContractError, match="uuid"):
        validate(df)


@pytest.mark.parametrize("column", ["target", "x1"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_fail(column, value):
    df = make_panel()
    df.loc[2, column] = value
    with pytest.raises(DataContractError, match="infinite"):
        validate(df)


def test_duplicate_uuid_date_rows_fail():
    df = make_panel()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate"):
        validate(df)


def test_empty_input_fails():
    with pytest.raises(DataContractError, match="empty"):
        validate(make_panel().iloc[0:0])


def test_dates_off_the_frequency_grid_fail():
    df = make_panel()
    df.loc[2, "date"] = pd.Timestamp("2024-03-15")
    with pytest.raises(DataContractError, match="not aligned"):
        validate(df)


def test_series_below_the_minimum_length_fail():
    with pytest.raises(DataContractError, match="fewer than 3 rows"):
        validate(make_panel(n_periods=2))


def test_all_problems_are_reported_together():
    df = make_panel()
    df["surprise"] = 1.0
    df.loc[0, "uuid"] = "bad"
    with pytest.raises(DataContractError) as info:
        validate(df, id_format="uuid")
    message = str(info.value)
    assert "surprise" in message and "uuid" in message


def test_static_columns_are_part_of_the_contract():
    df = make_panel()
    df["market"] = "DE04"
    assert list(validate(df, static_columns=["market"]).columns)[-1] == "market"
    with pytest.raises(DataContractError):
        validate(df)  # an undeclared extra column is rejected (strict schema)
    with pytest.raises(DataContractError):
        validate(make_panel(), static_columns=["market"])  # a declared column must exist


def test_blank_or_changing_static_values_are_rejected():
    df = make_panel()
    df["market"] = "DE04"
    blank = df.copy()
    blank.loc[0, "market"] = None
    with pytest.raises(DataContractError):
        validate(blank, static_columns=["market"])
    changing = df.copy()
    changing.loc[1, "market"] = "PT04"
    with pytest.raises(DataContractError, match="one value per uuid"):
        validate(changing, static_columns=["market"])
