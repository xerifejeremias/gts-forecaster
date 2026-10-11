import logging

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.cleaning.missing import handle_missing_data
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.frequency_checks import check_complete_panel
from helpers import EXPLANATORY, drop_rows, make_panel, make_uuid

VALUE_COLUMNS = ["y", *EXPLANATORY]


def panel(**kwargs):
    return make_panel(internal=True, **kwargs)
U0, U1 = make_uuid(0), make_uuid(1)


def run(df, **kwargs):
    kwargs.setdefault("pandas_freq", "MS")
    kwargs.setdefault("value_columns", VALUE_COLUMNS)
    return handle_missing_data(df, **kwargs)


def series(result, uuid):
    return result.data[result.data["unique_id"] == uuid].reset_index(drop=True)


def test_complete_data_is_unchanged_apart_from_the_flag():
    df = panel()
    result = run(df)
    assert result.fallback.empty and result.report.empty
    assert result.cutoff == df["ds"].max()
    assert list(result.data.columns) == ["unique_id", "ds", *VALUE_COLUMNS, "is_filled"]
    assert not result.data["is_filled"].any()
    pd.testing.assert_frame_equal(result.data.drop(columns="is_filled"), df, check_dtype=False)


def test_trailing_gap_up_to_the_limit_is_forward_filled():
    df = drop_rows(panel(n_periods=8), U1, ["2024-06-01", "2024-07-01", "2024-08-01"])
    result = run(df)
    s = series(result, U1)
    assert len(s) == 8 and s["ds"].iloc[-1] == pd.Timestamp("2024-08-01")
    assert (s["y"].iloc[-3:] == s["y"].iloc[4]).all()  # value of 2024-05-01
    assert (s["x1"].iloc[-3:] == s["x1"].iloc[4]).all()
    assert s["is_filled"].tolist() == [False] * 5 + [True] * 3
    assert result.fallback.empty
    row = result.report.iloc[0]
    assert (row["gap_type"], row["n_periods"], row["action"]) == ("trailing", 3, "forward_fill")


def test_trailing_gap_longer_than_the_limit_uses_the_fallback():
    df = drop_rows(panel(n_periods=8), U1, ["2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"])
    result = run(df)
    assert result.data["unique_id"].unique().tolist() == [U0]  # the series is not given to the models
    fb = result.fallback.iloc[0]
    assert fb["unique_id"] == U1
    assert fb["last_observed_date"] == pd.Timestamp("2024-04-01")
    assert fb["last_observed_value"] == fb["fallback_value"] == 8.0  # 4th value of series 1 is 4 * 2
    assert result.report.iloc[0]["action"] == "fallback"


def test_interior_gap_up_to_the_limit_is_linearly_interpolated():
    df = drop_rows(panel(n_series=1, n_periods=8), U0, ["2024-03-01", "2024-04-01"])
    s = series(run(df), U0)
    assert len(s) == 8
    assert s["y"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]  # linear data stays linear
    assert s["x2"].tolist() == [2, 4, 6, 8, 10, 12, 14, 16]
    assert s["is_filled"].tolist() == [False, False, True, True] + [False] * 4


def test_interpolation_uses_the_surrounding_values():
    df = panel(n_series=1, n_periods=5)
    df.loc[1:3, "y"] = [np.nan, np.nan, np.nan]
    df.loc[0, "y"], df.loc[4, "y"] = 10.0, 18.0
    s = series(run(df), U0)
    assert s["y"].tolist() == [10.0, 12.0, 14.0, 16.0, 18.0]


def test_interior_gap_longer_than_the_limit_uses_the_fallback():
    df = drop_rows(panel(n_periods=8), U0, ["2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"])
    result = run(df)
    assert result.data["unique_id"].unique().tolist() == [U1]
    fb = result.fallback.iloc[0]
    assert fb["unique_id"] == U0
    assert fb["last_observed_date"] == pd.Timestamp("2024-08-01")  # last real value, after the gap
    assert fb["last_observed_value"] == 8.0
    assert result.report.iloc[0]["gap_type"] == "interior"


def test_fallback_zero_forecasts_zero_but_still_records_the_last_value():
    df = drop_rows(panel(n_periods=8), U1, ["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"])
    fb = run(df, fallback="zero").fallback.iloc[0]
    assert fb["fallback_value"] == 0.0
    assert fb["last_observed_value"] == 6.0  # 3rd value of series 1 is 3 * 2


def test_the_limit_is_per_run_not_per_series():
    # two separate gaps of 2 and a trailing gap of 2: each is within the limit of 3
    df = drop_rows(
        panel(n_series=2, n_periods=12),
        U0,
        ["2024-03-01", "2024-04-01", "2024-07-01", "2024-08-01", "2024-11-01", "2024-12-01"],
    )
    result = run(df)
    assert result.fallback.empty
    assert len(series(result, U0)) == 12
    assert result.report["action"].tolist() == ["interpolate", "interpolate", "forward_fill"]


def test_max_fill_periods_zero_sends_any_gap_to_the_fallback():
    df = drop_rows(panel(), U0, ["2024-08-01"])
    result = run(df, max_fill_periods=0)
    assert result.fallback["unique_id"].tolist() == [U0]


def test_null_values_count_as_missing():
    df = panel(n_series=1, n_periods=6)
    df.loc[2, "x1"] = np.nan  # only an explanatory value is missing
    result = run(df)
    s = series(result, U0)
    assert s["is_filled"].tolist() == [False, False, True, False, False, False]
    assert s["x1"].tolist() == [1, 2, 3, 4, 5, 6]


def test_null_explanatory_at_the_end_is_forward_filled():
    df = panel(n_series=1, n_periods=6)
    df.loc[5, "x2"] = np.nan
    s = series(run(df), U0)
    assert s["x2"].iloc[-1] == s["x2"].iloc[-2] and s["is_filled"].iloc[-1]


def test_leading_incomplete_rows_are_trimmed():
    df = panel(n_series=1, n_periods=6)
    df.loc[0:1, "y"] = np.nan
    s = series(run(df), U0)
    assert len(s) == 4 and s["ds"].iloc[0] == pd.Timestamp("2024-03-01")


def test_series_with_no_complete_period_is_a_contract_error():
    df = panel(n_series=1, n_periods=4)
    df["y"] = np.nan
    with pytest.raises(DataContractError, match="no period where"):
        run(df)


def test_all_series_end_on_the_common_cutoff_after_repair():
    df = drop_rows(panel(n_series=3, n_periods=10), make_uuid(2), ["2024-09-01", "2024-10-01"])
    df = drop_rows(df, make_uuid(1), ["2024-05-01"])
    result = run(df)
    assert result.cutoff == pd.Timestamp("2024-10-01")
    check_complete_panel(result.data, "MS", VALUE_COLUMNS, result.cutoff)


def test_weekly_and_daily_grids_are_respected():
    for freq, start, gap in [("W-MON", "2024-01-01", "2024-01-15"), ("D", "2024-01-01", "2024-01-03")]:
        df = drop_rows(panel(n_series=1, n_periods=6, pandas_freq=freq, start=start), U0, [gap])
        result = run(df, pandas_freq=freq)
        check_complete_panel(result.data, freq, VALUE_COLUMNS, result.cutoff)
        assert result.data["is_filled"].sum() == 1


def test_row_order_of_the_input_does_not_matter():
    df = drop_rows(panel(n_periods=8), U0, ["2024-04-01"]).sample(frac=1, random_state=1)
    result = run(df)
    assert series(result, U0)["y"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_warnings_name_the_uuid_the_dates_and_the_action(caplog):
    df = drop_rows(panel(n_periods=8), U0, ["2024-03-01"])
    df = drop_rows(df, U1, ["2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"])
    with caplog.at_level(logging.WARNING):
        run(df)
    text = caplog.text
    assert U0 in text and "2024-03-01" in text and "linearly interpolated" in text
    assert U1 in text and "2024-05-01" in text and "2024-08-01" in text
    assert "no model is applied" in text and "last observed value" in text


def test_trailing_fill_warning_says_forward_fill(caplog):
    df = drop_rows(panel(n_periods=8), U1, ["2024-08-01"])
    with caplog.at_level(logging.WARNING):
        run(df)
    assert "forward filled" in caplog.text


def test_zero_fallback_warning_says_zero(caplog):
    df = drop_rows(panel(n_periods=8), U1, ["2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"])
    with caplog.at_level(logging.WARNING):
        run(df, fallback="zero")
    assert "will be 0" in caplog.text


@pytest.mark.parametrize("kwargs", [{"fallback": "mean"}, {"max_fill_periods": -1}])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        run(panel(), **kwargs)


def test_every_series_on_fallback_returns_empty_model_data():
    df = drop_rows(panel(n_series=1, n_periods=8), U0, ["2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"])
    result = run(df)
    assert result.data.empty and result.fallback["unique_id"].tolist() == [U0]
