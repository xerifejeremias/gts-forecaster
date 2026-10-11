"""Frequency dependent input rules and the post-repair completeness check."""

import pandas as pd
import pytest

from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.frequency_checks import (
    check_complete_panel,
    check_dates_aligned,
    check_min_length,
    check_not_empty,
)
from helpers import EXPLANATORY, drop_rows, make_panel, make_uuid

VALUE_COLUMNS = ["y", *EXPLANATORY]


def test_empty_input_fails():
    with pytest.raises(DataContractError, match="empty"):
        check_not_empty(make_panel().iloc[0:0])


@pytest.mark.parametrize(
    "pandas_freq,start",
    [("YS", "2000-01-01"), ("MS", "2024-01-01"), ("W-MON", "2024-01-01"), ("D", "2024-01-03")],
)
def test_aligned_dates_pass(pandas_freq, start):
    check_dates_aligned(make_panel(pandas_freq=pandas_freq, start=start), pandas_freq)


def test_mid_month_dates_fail_for_monthly():
    df = make_panel(pandas_freq="MS")
    df.loc[3, "date"] = pd.Timestamp("2024-04-15")
    with pytest.raises(DataContractError, match="not aligned"):
        check_dates_aligned(df, "MS")


def test_tuesday_fails_for_weekly_monday():
    df = make_panel(pandas_freq="W-TUE", start="2024-01-02")  # Tuesdays
    with pytest.raises(DataContractError, match="not aligned"):
        check_dates_aligned(df, "W-MON")


def test_year_end_fails_for_year_start():
    df = make_panel(pandas_freq="365D", start="2020-12-31")
    with pytest.raises(DataContractError, match="not aligned"):
        check_dates_aligned(df, "YS")


def test_alignment_error_names_the_affected_uuid():
    df = make_panel(n_series=2)
    df.loc[df["uuid"] == make_uuid(1), "date"] += pd.Timedelta(days=1)
    with pytest.raises(DataContractError, match=make_uuid(1)):
        check_dates_aligned(df, "MS")


def test_min_length_passes_at_the_floor():
    check_min_length(make_panel(n_periods=3), min_series_length=3)


def test_short_series_is_listed_not_dropped():
    df = pd.concat([make_panel(n_series=1, n_periods=8), make_panel(n_series=2, n_periods=2).iloc[2:]])
    with pytest.raises(DataContractError, match=r"1 series have fewer than 3 rows.*\(2 rows\)"):
        check_min_length(df, min_series_length=3)


def test_complete_panel_passes():
    df = make_panel(internal=True)
    check_complete_panel(df, "MS", VALUE_COLUMNS, df["ds"].max())


def test_nulls_remaining_fail():
    df = make_panel(internal=True)
    df.loc[2, "x1"] = None
    with pytest.raises(DataContractError, match=r"null values remain.*x1 \(1\)"):
        check_complete_panel(df, "MS", VALUE_COLUMNS, df["ds"].max())


def test_gap_in_the_middle_fails():
    df = drop_rows(make_panel(internal=True), make_uuid(0), ["2024-03-01"])
    with pytest.raises(DataContractError, match="gaps or do not end at the cutoff"):
        check_complete_panel(df, "MS", VALUE_COLUMNS, df["ds"].max())


def test_series_ending_early_fails():
    df = drop_rows(make_panel(internal=True), make_uuid(1), ["2024-08-01"])
    with pytest.raises(DataContractError, match=make_uuid(1)):
        check_complete_panel(df, "MS", VALUE_COLUMNS, pd.Timestamp("2024-08-01"))
