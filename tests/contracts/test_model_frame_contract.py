"""The final shape of the data handed to the models."""

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.cleaning.missing import handle_missing_data
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.frequency_checks import check_model_frame
from helpers import EXPLANATORY, make_panel

VALUE_COLUMNS = ["y", *EXPLANATORY]


def good_frame():
    return handle_missing_data(
        make_panel(n_series=3, internal=True), pandas_freq="MS", value_columns=VALUE_COLUMNS
    ).data


def test_output_of_the_missing_data_step_is_a_valid_model_frame():
    check_model_frame(good_frame(), EXPLANATORY)


def test_wrong_column_names_fail():
    df = good_frame().rename(columns={"y": "target"})
    with pytest.raises(DataContractError, match="columns are"):
        check_model_frame(df, EXPLANATORY)


def test_extra_column_fails():
    df = good_frame()
    df["extra"] = 1.0
    with pytest.raises(DataContractError, match="columns are"):
        check_model_frame(df, EXPLANATORY)


def test_wrong_column_order_fails():
    df = good_frame()[["ds", "unique_id", "y", *EXPLANATORY, "is_filled"]]
    with pytest.raises(DataContractError, match="columns are"):
        check_model_frame(df, EXPLANATORY)


def test_dates_as_text_fail():
    df = good_frame()
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    with pytest.raises(DataContractError, match="ds must be"):
        check_model_frame(df, EXPLANATORY)


def test_timezone_aware_dates_fail():
    df = good_frame()
    df["ds"] = df["ds"].dt.tz_localize("UTC")
    with pytest.raises(DataContractError, match="ds must be"):
        check_model_frame(df, EXPLANATORY)


def test_integer_target_fails():
    df = good_frame()
    df["y"] = df["y"].astype(int)
    with pytest.raises(DataContractError, match="y must be float64"):
        check_model_frame(df, EXPLANATORY)


def test_nulls_fail():
    df = good_frame()
    df.loc[2, "y"] = np.nan
    with pytest.raises(DataContractError, match="null"):
        check_model_frame(df, EXPLANATORY)


def test_duplicate_rows_fail():
    df = good_frame()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate"):
        check_model_frame(df, EXPLANATORY)


def test_unsorted_rows_fail():
    df = good_frame().sample(frac=1, random_state=3).reset_index(drop=True)
    with pytest.raises(DataContractError, match="not sorted"):
        check_model_frame(df, EXPLANATORY)
