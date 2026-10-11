import pandas as pd

from gts_forecaster.columns import ID, DATE, TARGET, to_model_format
from helpers import make_panel


def test_names_follow_the_nixtla_convention():
    assert (ID, DATE, TARGET) == ("unique_id", "ds", "y")


def test_to_model_format_renames_only_the_three_columns():
    out = to_model_format(make_panel())
    assert list(out.columns) == ["unique_id", "ds", "y", "x1", "x2"]


def test_to_model_format_keeps_the_values_and_does_not_change_the_input():
    df = make_panel()
    out = to_model_format(df)
    assert list(df.columns)[:3] == ["uuid", "date", "target"]
    pd.testing.assert_series_equal(out["y"], df["target"], check_names=False)
    pd.testing.assert_series_equal(out["ds"], df["date"], check_names=False)
