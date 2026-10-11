from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.splitting import (
    SplitError,
    backtest_cutoffs,
    backtest_splits,
    backtest_splits_from_prepared,
    check_horizon,
    forecast_split,
    forecast_split_from_prepared,
    split_at_cutoff,
)
from helpers import make_panel, make_uuid

FREQS = {"YS": "2000-01-01", "MS": "2024-01-01", "W-MON": "2024-01-01", "D": "2024-01-01"}


def panel(n_series=2, n_periods=12, freq="MS", start=None, **kwargs):
    return make_panel(
        n_series=n_series, n_periods=n_periods, pandas_freq=freq, start=start or FREQS[freq],
        internal=True, **kwargs,
    )


def late_start_panel():
    """Series 0 has 12 periods, series 1 starts at period 7 (so has only 6), all end together."""
    df = panel(n_series=2, n_periods=12)
    late = df["unique_id"] == make_uuid(1)
    return df.loc[~late | (df["ds"] >= pd.Timestamp("2024-07-01"))].reset_index(drop=True)


def dates(freq, start, n):
    return list(pd.date_range(start, periods=n, freq=freq))


# ---- horizon is an input, never fixed ------------------------------------------------------


@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_test_window_is_exactly_the_next_horizon_periods(freq, horizon):
    df = panel(freq=freq, n_periods=20)
    cutoff = pd.date_range(FREQS[freq], periods=20, freq=freq)[9]
    split = split_at_cutoff(df, cutoff, horizon, freq)
    expected = dates(freq, cutoff, horizon + 1)[1:]
    assert list(split.horizon_dates) == expected
    one_series = split.test[split.test["unique_id"] == make_uuid(0)]
    assert one_series["ds"].tolist() == expected
    assert one_series["horizon_step"].tolist() == list(range(1, horizon + 1))
    assert len(split.test) == 2 * horizon


def test_changing_the_horizon_changes_the_split():
    df = panel()
    cutoff = pd.Timestamp("2024-06-01")
    short = split_at_cutoff(df, cutoff, 2, "MS")
    long = split_at_cutoff(df, cutoff, 5, "MS")
    assert len(short.test) == 4 and len(long.test) == 10
    assert len(short.train) == len(long.train)


def test_there_is_no_default_horizon():
    with pytest.raises(TypeError):
        forecast_split(panel(), pandas_freq="MS")
    with pytest.raises(TypeError):
        split_at_cutoff(panel(), "2024-06-01", pandas_freq="MS")


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3", None])
def test_invalid_horizon_is_rejected(bad):
    with pytest.raises(SplitError, match="horizon"):
        check_horizon(bad)
    with pytest.raises(SplitError, match="horizon"):
        forecast_split(panel(), bad, "MS")


def test_numpy_integer_horizon_is_accepted():
    assert check_horizon(np.int64(3)) == 3


# ---- split at a cutoff ---------------------------------------------------------------------


def test_train_ends_at_the_cutoff_and_test_starts_after_it():
    df = panel()
    split = split_at_cutoff(df, "2024-08-01", 3, "MS")
    assert split.cutoff == pd.Timestamp("2024-08-01")
    assert split.train["ds"].max() == split.cutoff
    assert split.test["ds"].min() > split.cutoff
    assert len(split.train) == 2 * 8 and len(split.test) == 2 * 3
    assert not split.is_forecast and split.fold is None


def test_train_and_test_do_not_overlap_and_cover_the_expected_rows():
    df = panel()
    split = split_at_cutoff(df, "2024-05-01", 4, "MS")
    keys_train = set(zip(split.train["unique_id"], split.train["ds"]))
    keys_test = set(zip(split.test["unique_id"], split.test["ds"]))
    assert not keys_train & keys_test
    assert len(keys_train) + len(keys_test) == 2 * (5 + 4)


def test_test_values_come_from_the_data_with_all_columns():
    df = panel()
    split = split_at_cutoff(df, "2024-08-01", 2, "MS")
    row = split.test[(split.test["unique_id"] == make_uuid(1)) & (split.test["ds"] == "2024-09-01")]
    original = df[(df["unique_id"] == make_uuid(1)) & (df["ds"] == "2024-09-01")]
    assert row["y"].iloc[0] == original["y"].iloc[0] == 18.0
    assert {"x1", "x2", "horizon_step"} <= set(split.test.columns)


def test_last_possible_cutoff_is_end_minus_horizon():
    df = panel()
    split_at_cutoff(df, "2024-09-01", 3, "MS")  # test window Oct, Nov, Dec = end of data
    with pytest.raises(SplitError, match="after the last date"):
        split_at_cutoff(df, "2024-10-01", 3, "MS")


def test_cutoff_off_the_grid_is_rejected():
    with pytest.raises(SplitError, match="not a date"):
        split_at_cutoff(panel(), "2024-05-15", 2, "MS")


def test_cutoff_outside_the_data_is_rejected():
    with pytest.raises(SplitError, match="not a date"):
        split_at_cutoff(panel(), "2023-01-01", 2, "MS")


def test_series_without_data_before_the_cutoff_are_skipped_and_reported():
    df = late_start_panel()
    split = split_at_cutoff(df, "2024-05-01", 2, "MS")
    assert split.skipped_ids == (make_uuid(1),)
    assert split.train["unique_id"].unique().tolist() == [make_uuid(0)]
    assert split.test["unique_id"].unique().tolist() == [make_uuid(0)]
    assert split.future["unique_id"].unique().tolist() == [make_uuid(0)]


def test_train_lengths_per_series():
    split = split_at_cutoff(late_start_panel(), "2024-09-01", 2, "MS")
    assert split.train_lengths.to_dict() == {make_uuid(0): 9, make_uuid(1): 3}
    assert split.skipped_ids == ()


def test_input_is_not_modified():
    df = panel()
    before = df.copy()
    split_at_cutoff(df, "2024-05-01", 2, "MS")
    pd.testing.assert_frame_equal(df, before)


def test_split_keeps_is_filled_for_scoring():
    df = panel().assign(is_filled=False)
    df.loc[(df["unique_id"] == make_uuid(0)) & (df["ds"] == "2024-07-01"), "is_filled"] = True
    split = split_at_cutoff(df, "2024-06-01", 2, "MS")
    flagged = split.test[split.test["is_filled"]]
    assert flagged["ds"].tolist() == [pd.Timestamp("2024-07-01")]


def test_missing_columns_are_rejected():
    with pytest.raises(SplitError, match="model format"):
        split_at_cutoff(make_panel(), "2024-05-01", 2, "MS")


def test_empty_data_is_rejected():
    with pytest.raises(SplitError, match="empty"):
        forecast_split(panel().iloc[0:0], 2, "MS")


# ---- forecast split ------------------------------------------------------------------------


@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("horizon", [1, 3, 7])
def test_forecast_split_uses_all_data_and_has_no_test_window(freq, horizon):
    df = panel(freq=freq, n_periods=10)
    split = forecast_split(df, horizon, freq)
    last = df["ds"].max()
    assert split.cutoff == last and split.is_forecast
    assert len(split.train) == len(df) and split.test.empty
    assert list(split.horizon_dates) == dates(freq, last, horizon + 1)[1:]
    assert len(split.future) == 2 * horizon


def test_forecast_split_future_has_every_series_and_step():
    split = forecast_split(panel(), 3, "MS")
    assert split.future.columns.tolist() == ["unique_id", "ds", "horizon_step"]
    one = split.future[split.future["unique_id"] == make_uuid(1)]
    assert one["ds"].tolist() == dates("MS", "2024-12-01", 4)[1:]
    assert one["horizon_step"].tolist() == [1, 2, 3]


def test_forecast_test_frame_has_the_same_columns_as_a_backtest_test_frame():
    df = panel()
    assert list(forecast_split(df, 2, "MS").test.columns) == list(
        split_at_cutoff(df, "2024-06-01", 2, "MS").test.columns
    )


# ---- backtest cutoffs and splits -----------------------------------------------------------


def test_default_step_is_the_horizon_and_newest_cutoff_leaves_one_window():
    cutoffs = backtest_cutoffs(panel(), horizon=2, pandas_freq="MS", n_folds=3)
    # data ends 2024-12-01; newest cutoff = 2024-10-01; step 2 back: 2024-08-01, 2024-06-01
    assert cutoffs == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-08-01"), pd.Timestamp("2024-10-01")]


def test_custom_step_gives_overlapping_windows():
    cutoffs = backtest_cutoffs(panel(), 3, "MS", n_folds=3, step=1)
    assert cutoffs == [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-01")]


def test_single_fold_is_the_most_recent_window():
    assert backtest_cutoffs(panel(), 4, "MS", n_folds=1) == [pd.Timestamp("2024-08-01")]


def test_backtest_splits_are_numbered_oldest_first_and_never_leak():
    splits = backtest_splits(panel(), 2, "MS", n_folds=3)
    assert [s.fold for s in splits] == [1, 2, 3]
    assert [s.cutoff for s in splits] == sorted(s.cutoff for s in splits)
    for s in splits:
        assert s.train["ds"].max() <= s.cutoff < s.test["ds"].min()
        assert s.test["horizon_step"].max() == 2
    assert [len(s.train) for s in splits] == [2 * 6, 2 * 8, 2 * 10]


def test_default_step_gives_non_overlapping_test_windows():
    splits = backtest_splits(panel(), 2, "MS", n_folds=3)
    seen = [set(s.test["ds"]) for s in splits]
    assert not (seen[0] & seen[1]) and not (seen[1] & seen[2])


def test_late_starting_series_join_the_later_folds():
    splits = backtest_splits(late_start_panel(), 2, "MS", n_folds=3)
    assert [s.skipped_ids for s in splits] == [(make_uuid(1),), (), ()]


def test_not_enough_history_for_the_folds_is_rejected():
    with pytest.raises(SplitError, match="not enough history"):
        backtest_cutoffs(panel(n_periods=12), 4, "MS", n_folds=3)  # needs 4 + 2*4 + 1 = 13


def test_min_train_periods_is_respected():
    backtest_cutoffs(panel(), 2, "MS", n_folds=3, min_train_periods=6)
    with pytest.raises(SplitError, match="not enough history"):
        backtest_cutoffs(panel(), 2, "MS", n_folds=3, min_train_periods=7)


@pytest.mark.parametrize("kwargs", [{"n_folds": 0}, {"n_folds": 1.5}, {"n_folds": 2, "step": 0}, {"n_folds": 2, "min_train_periods": 0}])
def test_invalid_backtest_arguments_are_rejected(kwargs):
    with pytest.raises(SplitError):
        backtest_cutoffs(panel(), 2, "MS", **kwargs)


@pytest.mark.parametrize("freq", list(FREQS))
def test_backtest_works_on_every_frequency(freq):
    splits = backtest_splits(panel(freq=freq, n_periods=30), 4, freq, n_folds=3)
    assert len(splits) == 3
    for s in splits:
        assert list(s.horizon_dates) == dates(freq, s.cutoff, 5)[1:]


# ---- same module serves the forecast and the backtest --------------------------------------


def test_forecast_and_backtest_share_the_same_split_rules():
    df = panel()
    fold = split_at_cutoff(df, "2024-09-01", 3, "MS")
    forecast = forecast_split(df.loc[df["ds"] <= "2024-09-01"], 3, "MS")  # data as it was at the cutoff
    pd.testing.assert_frame_equal(fold.train, forecast.train)
    pd.testing.assert_frame_equal(fold.future, forecast.future)
    assert fold.horizon_dates.equals(forecast.horizon_dates)


# ---- helpers using the prepared input and the config ---------------------------------------


def prepared(horizon, freq="monthly", n_periods=24):
    config = load_config({"frequency": freq, "horizon": horizon})
    return SimpleNamespace(data=panel(n_periods=n_periods), config=config)


def test_horizon_comes_from_the_config():
    split = forecast_split_from_prepared(prepared(horizon=4))
    assert split.horizon == 4 and len(split.horizon_dates) == 4


def test_horizon_can_be_overridden_per_call():
    split = forecast_split_from_prepared(prepared(horizon=4), horizon=6)
    assert split.horizon == 6


def test_backtest_helper_uses_the_config_horizon_and_frequency():
    splits = backtest_splits_from_prepared(prepared(horizon=3), n_folds=2)
    assert [s.horizon for s in splits] == [3, 3]
    assert splits[-1].test["ds"].max() == pd.Timestamp("2025-12-01")
    splits = backtest_splits_from_prepared(prepared(horizon=3), n_folds=2, horizon=1)
    assert [s.horizon for s in splits] == [1, 1]
