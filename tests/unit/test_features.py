import sys
import types

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.features.builder import (
    CalendarSettings,
    FeatureBuilder,
    FeatureSet,
    build_feature_builders,
    holiday_dates,
    load_feature_config,
)
from gts_forecaster.features.calendar_features import (
    available_calendar_features,
    calendar_features,
    frequency_kind,
)
from gts_forecaster.features.errors import FeatureError
from gts_forecaster.features.target import (
    future_target_features,
    history_target_features,
    usable_at_step,
)
from gts_forecaster.models.base import BaseForecaster
from gts_forecaster.models.catalog import MODEL_TYPES, build_models, build_models_from_config, load_model_specs


def _dates(freq, n, start="2024-01-01"):
    return pd.DataFrame({"unique_id": "a", "ds": pd.date_range(start, periods=n, freq=freq)})


# ---- calendar ---------------------------------------------------------------------------------


def test_monthly_business_days_and_weekday_counts():
    cal = calendar_features(_dates("MS", 4), pandas_freq="MS")
    assert cal["n_business_days"].tolist() == [23, 21, 21, 22]  # Jan..Apr 2024
    assert cal["n_days"].tolist() == [31, 29, 31, 30]
    assert cal["share_business_days"].iloc[0] == pytest.approx(23 / 31)
    assert cal["n_mondays"].tolist() == [5, 4, 4, 5]
    assert cal[[f"n_{d}s" for d in "monday tuesday wednesday thursday friday saturday sunday".split()]].sum(axis=1).tolist() == [31, 29, 31, 30]
    assert cal["month"].tolist() == [1, 2, 3, 4] and cal["quarter"].tolist() == [1, 1, 1, 2]


def test_holidays_reduce_business_days():
    cal = calendar_features(_dates("MS", 1, "2024-12-01"), pandas_freq="MS",
                            holidays=["2024-12-25", "2024-12-26", "2024-12-28"])  # last is a Saturday
    assert cal["n_business_days"].iloc[0] == 22 - 2  # December 2024 has 22 weekdays; the 25th and 26th are two of them
    assert cal["n_holidays"].iloc[0] == 3


def test_daily_features():
    cal = calendar_features(_dates("D", 7, "2024-12-23"), pandas_freq="D", holidays=["2024-12-25"])
    assert cal["is_business_day"].tolist() == [1, 1, 0, 1, 1, 0, 0]
    assert cal["is_holiday"].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert cal["day_of_week"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert cal["day_of_month"].iloc[0] == 23 and cal["day_of_year"].iloc[0] == 358
    assert cal["week_of_year"].iloc[0] == 52
    d = calendar_features(_dates("D", 3, "2024-01-30"), pandas_freq="D")
    assert d["is_month_end"].tolist() == [0, 1, 0] and d["is_month_start"].tolist() == [0, 0, 1]


def test_weekly_period_covers_seven_days_from_the_date():
    cal = calendar_features(_dates("W-MON", 2, "2024-01-01"), pandas_freq="W-MON", holidays=["2024-01-02"])
    assert cal["n_days"].tolist() == [7, 7]
    assert cal["n_business_days"].tolist() == [4, 5]
    assert cal["week_of_year"].tolist() == [1, 2]


def test_end_anchored_dates_describe_the_period_they_end():
    # a date of 2024-03-31 with anchor=end stands for March
    df = _dates("ME", 1, "2024-03-31")
    cal = calendar_features(df, pandas_freq="ME", anchor="end", names=["n_days", "n_business_days"])
    assert cal["n_days"].iloc[0] == 31 and cal["n_business_days"].iloc[0] == 21


def test_features_missing_for_a_frequency_are_skipped_and_unknown_names_fail():
    monthly = calendar_features(_dates("MS", 2), pandas_freq="MS", names=["month", "day_of_week"])
    assert list(monthly.columns) == ["month"]
    yearly = calendar_features(_dates("YS", 2), pandas_freq="YS", names="all")
    assert list(yearly.columns) == ["n_days", "n_business_days", "share_business_days"]
    assert yearly["n_days"].tolist() == [366, 365]
    with pytest.raises(FeatureError, match="unknown calendar feature"):
        calendar_features(_dates("MS", 2), pandas_freq="MS", names=["motnh"])
    with pytest.raises(FeatureError):
        frequency_kind("h")


def test_fourier_terms_are_periodic_and_bounded():
    cal = calendar_features(_dates("MS", 25), pandas_freq="MS", names=[], fourier_order=2)
    assert list(cal.columns) == ["year_sin_1", "year_cos_1", "year_sin_2", "year_cos_2"]
    assert cal.abs().max().max() <= 1.0
    # same month one year apart (not through a leap day): identical phase for monthly starts
    assert cal["year_sin_1"].iloc[1] == pytest.approx(cal["year_sin_1"].iloc[13], abs=0.02)


def test_calendar_rows_keep_the_order_of_the_input_across_series():
    ds = pd.date_range("2024-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"unique_id": ["a", "b", "a", "b", "a", "b"],
                       "ds": [ds[0], ds[0], ds[1], ds[1], ds[2], ds[2]]}, index=[10, 11, 12, 13, 14, 15])
    cal = calendar_features(df, pandas_freq="MS", names=["month"])
    assert cal.index.tolist() == [10, 11, 12, 13, 14, 15]
    assert cal["month"].tolist() == [1, 1, 2, 2, 3, 3]


def test_available_features_depend_on_frequency_and_holidays():
    assert "is_business_day" in available_calendar_features("D")
    assert "is_holiday" not in available_calendar_features("D")
    assert "is_holiday" in available_calendar_features("D", has_holidays=True)
    assert "day_of_week" not in available_calendar_features("MS")


# ---- target features ----------------------------------------------------------------------------


def _train(n=30, uid="a", offset=0.0):
    ds = pd.date_range("2022-01-01", periods=n, freq="MS")
    return pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.arange(n, dtype=float) + offset})


def test_lags_and_rolling_values():
    t = _train()
    f = history_target_features(t, [1, 12], [3], ["mean", "std", "min", "max", "median"], 1)
    assert f["lag_1"].iloc[5] == 4.0 and np.isnan(f["lag_1"].iloc[0])
    assert f["lag_12"].iloc[20] == 8.0 and f["lag_12"].iloc[:12].isna().all()
    # row 10 sees y[7], y[8], y[9] only
    assert f["roll_mean_3"].iloc[10] == pytest.approx(8.0)
    assert f["roll_std_3"].iloc[10] == pytest.approx(1.0)
    assert (f["roll_min_3"].iloc[10], f["roll_max_3"].iloc[10], f["roll_median_3"].iloc[10]) == (7, 9, 8)
    assert f["roll_mean_3"].iloc[:3].isna().all() and not np.isnan(f["roll_mean_3"].iloc[3])


def test_rolling_lag_shifts_the_window():
    f = history_target_features(_train(), [], [3], ["mean"], 12)
    assert f["roll_mean_3"].iloc[20] == pytest.approx(np.mean([6, 7, 8]))  # y[t-12-2 .. t-12]


def test_features_never_use_the_current_or_later_values():
    t = _train()
    base = history_target_features(t, [1, 2], [3], ["mean"], 1)
    changed = t.copy()
    changed.loc[15:, "y"] += 1000
    after = history_target_features(changed, [1, 2], [3], ["mean"], 1)
    pd.testing.assert_frame_equal(base.iloc[:15], after.iloc[:15])  # rows up to the change are untouched
    assert after.iloc[15].equals(base.iloc[15])  # row 15 only sees values before row 15


def test_series_do_not_leak_into_each_other():
    both = pd.concat([_train(20, "a"), _train(20, "b", offset=1000.0)], ignore_index=True)
    f = history_target_features(both, [1], [3], ["mean"], 1)
    assert np.isnan(f["lag_1"].iloc[20])  # first row of series b has no lag
    assert f["lag_1"].iloc[21] == 1000.0


def test_history_features_keep_the_row_order_of_unsorted_input():
    t = _train(10).sample(frac=1, random_state=0)
    f = history_target_features(t, [1], [], [], 1)
    assert f.index.tolist() == t.index.tolist()
    ordered = history_target_features(t.sort_values("ds"), [1], [], [], 1)
    pd.testing.assert_series_equal(f["lag_1"].sort_index(), ordered["lag_1"].sort_index())


def test_future_features_match_history_features_where_known():
    full = _train(40)
    lags, windows, stats, rl = [1, 3, 6, 12], [3, 6], ["mean", "std"], 2
    reference = history_target_features(full, lags, windows, stats, rl)
    for cutoff in (20, 30, 36):
        train = full.iloc[:cutoff]
        horizon = 4
        future = pd.DataFrame({"unique_id": "a", "ds": full["ds"].iloc[cutoff:cutoff + horizon].to_numpy(),
                               "horizon_step": range(1, horizon + 1)})
        got = future_target_features(train, future, lags, windows, stats, rl)
        for step in range(1, horizon + 1):
            known = usable_at_step(lags, windows, stats, rl, step)
            want = reference.iloc[cutoff + step - 1]
            for col in got.columns:
                value = got.iloc[step - 1][col]
                if col in known and not np.isnan(want[col]):
                    assert value == pytest.approx(want[col]), (cutoff, step, col)
                elif col not in known:
                    assert np.isnan(value), (cutoff, step, col)


def test_future_features_are_empty_when_they_need_forecast_window_values():
    train = _train(30)
    future = pd.DataFrame({"unique_id": "a", "ds": pd.date_range("2024-07-01", periods=3, freq="MS"),
                           "horizon_step": [1, 2, 3]})
    f = future_target_features(train, future, [1, 2, 12], [3], ["mean"], 1)
    assert f["lag_1"].isna().tolist() == [False, True, True]
    assert f["lag_2"].isna().tolist() == [False, False, True]
    assert f["lag_12"].notna().all()
    assert f["roll_mean_3"].isna().tolist() == [False, True, True]
    assert f["lag_12"].tolist() == [18.0, 19.0, 20.0]


def test_future_features_for_unknown_series_and_short_history_are_nan():
    train = _train(5)
    future = pd.DataFrame({"unique_id": ["a", "zzz"], "ds": pd.to_datetime(["2022-06-01"] * 2), "horizon_step": [1, 1]})
    f = future_target_features(train, future, [1, 12], [3], ["mean"], 1)
    assert f["lag_1"].iloc[0] == 4.0 and np.isnan(f["lag_12"].iloc[0])
    assert f.iloc[1].isna().all()


@pytest.mark.parametrize(
    "lags,windows,stats,rl",
    [([0], [], [], 1), ([1.5], [], [], 1), ([], [1], ["mean"], 1), ([], [3], ["var"], 1), ([1], [], [], 0)],
)
def test_invalid_target_specs_are_rejected(lags, windows, stats, rl):
    with pytest.raises(FeatureError):
        history_target_features(_train(), lags, windows, stats, rl)


# ---- builder ------------------------------------------------------------------------------------


def _set(**kw):
    return FeatureSet(name=kw.pop("name", "s"), **kw)


def test_season_tokens_are_resolved_per_frequency():
    s = _set(calendar=(), lags=("season", "2*season", 3), rolling_windows=("season", 3))
    monthly = FeatureBuilder(s, pandas_freq="MS", season_length=12)
    assert monthly.lags == (12, 24, 3) and monthly.windows == (12, 3)
    daily = FeatureBuilder(s, pandas_freq="D", season_length=7)
    assert daily.lags == (7, 14, 3) and daily.windows == (7, 3)
    yearly = FeatureBuilder(s, pandas_freq="YS", season_length=1)
    assert yearly.lags == (1, 2, 3) and yearly.windows == (3,)  # window of 1 is dropped
    with pytest.raises(FeatureError):
        FeatureBuilder(_set(lags=("monthly",)), pandas_freq="MS", season_length=12)


def test_builder_adds_calendar_and_history_columns_for_train_and_future():
    b = FeatureBuilder(_set(calendar="all", lags=(1,), rolling_windows=(3,), rolling_stats=("mean",)),
                       pandas_freq="MS", season_length=12)
    train = _train(24)
    train["x_promo"] = 1.0
    out = b.transform_history(train)
    assert list(out.columns[:4]) == ["unique_id", "ds", "y", "x_promo"]
    assert set(b.feature_columns) <= set(out.columns) and len(out) == len(train)
    assert (out["x_promo"] == 1.0).all()  # explanatory columns pass through
    future = pd.DataFrame({"unique_id": "a", "ds": pd.date_range("2024-01-01", periods=2, freq="MS"),
                           "horizon_step": [1, 2], "x_promo": [0.0, 1.0]})
    fx = b.transform_future(train, future)
    assert fx["x_promo"].tolist() == [0.0, 1.0]
    assert fx["n_business_days"].tolist() == [23, 21]
    assert fx["lag_1"].isna().tolist() == [False, True]


def test_usable_at_step_lists_calendar_always_and_history_when_known():
    b = FeatureBuilder(_set(calendar=("month",), lags=(1, 12), rolling_windows=(3,), rolling_stats=("mean",)),
                       pandas_freq="MS", season_length=12)
    assert b.usable_at_step(1) == ["month", "lag_1", "lag_12", "roll_mean_3"]
    assert b.usable_at_step(2) == ["month", "lag_12"]
    assert b.usable_at_step(13) == ["month"]


def test_builder_refuses_to_overwrite_existing_columns():
    b = FeatureBuilder(_set(calendar=("month",)), pandas_freq="MS", season_length=12)
    train = _train(5).assign(month=7)
    with pytest.raises(FeatureError, match="overwrite"):
        b.transform_history(train)


def test_empty_feature_set_adds_nothing():
    b = FeatureBuilder(_set(), pandas_freq="MS", season_length=12)
    assert b.feature_columns == []
    train = _train(5)
    pd.testing.assert_frame_equal(b.transform_history(train), train)


def test_extra_holidays_change_business_day_features():
    plain = FeatureBuilder(_set(calendar=("n_business_days",)), pandas_freq="MS", season_length=12)
    closed = FeatureBuilder(_set(calendar=("n_business_days",)), pandas_freq="MS", season_length=12,
                            calendar=CalendarSettings(extra_holidays=("2024-01-02",)))
    df = _dates("MS", 1)
    assert plain.transform_future(_train(3), df.assign(horizon_step=1))["n_business_days"].iloc[0] == 23
    assert closed.transform_future(_train(3), df.assign(horizon_step=1))["n_business_days"].iloc[0] == 22


def test_country_holidays_use_the_holidays_package_when_installed(monkeypatch):
    fake = types.ModuleType("holidays")
    fake.country_holidays = lambda country, years: {pd.Timestamp(f"{y}-12-25").date(): "Christmas" for y in years}
    monkeypatch.setitem(sys.modules, "holidays", fake)
    dates = holiday_dates(CalendarSettings(country="PT", extra_holidays=("2024-05-01",)), [2024])
    assert dates == [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-12-25")]


def test_country_holidays_without_the_package_give_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "holidays", None)  # makes the import fail
    with pytest.raises(FeatureError, match="not installed"):
        holiday_dates(CalendarSettings(country="PT"), [2024])


# ---- features.yaml ---------------------------------------------------------------------------------


def test_default_feature_file_loads_and_every_set_builds_for_every_frequency():
    cfg = load_feature_config()
    assert {"calendar_only", "calendar_and_history"} <= set(cfg.sets)
    for freq, season in (("YS", 1), ("MS", 12), ("W-MON", 52), ("D", 7)):
        static = pd.DataFrame({"unique_id": ["a"], "market": ["DE04"], "product": ["F0101"],
                               "company": ["company A"]})
        builders = build_feature_builders(cfg, list(cfg.sets), pandas_freq=freq,
                                          season_length=season, static=static)
        for builder in builders.values():
            frame = _dates(freq, 80)
            frame["y"] = np.arange(80.0)
            out = builder.transform_history(frame)
            assert len(out) == 80 and not out[builder.calendar_columns()].isna().any().any()


def test_feature_file_errors(tmp_path):
    (tmp_path / "features.yaml").write_text("sets:\n  a:\n    lagz: [1]\n")
    with pytest.raises(FeatureError, match="unknown key"):
        load_feature_config(tmp_path)
    (tmp_path / "features.yaml").write_text("calendar:\n  period_anchor: middle\n")
    with pytest.raises(FeatureError):
        load_feature_config(tmp_path)
    (tmp_path / "features.yaml").write_text("calendar:\n  extra_holidays: [not-a-date]\n")
    with pytest.raises(FeatureError):
        load_feature_config(tmp_path)
    (tmp_path / "features.yaml").write_text("sets:\n  a:\n    calendar: month\n")
    with pytest.raises(FeatureError):
        load_feature_config(tmp_path)
    assert load_feature_config(tmp_path / "missing").sets == {}


def test_unknown_feature_set_name_is_reported():
    with pytest.raises(FeatureError, match="not defined"):
        build_feature_builders(load_feature_config(), ["nope"], pandas_freq="MS", season_length=12)


# ---- models using feature sets --------------------------------------------------------------------


class _FeatureUser(BaseForecaster):
    def __init__(self, feature_builder=None):
        self.feature_builder = feature_builder

    def _fit(self, series, train):
        self.train_ = train
        self.last_ = {u: y[-1] for u, y in series.items()}

    def _predict(self, future):
        self.future_features_ = self.feature_builder.transform_future(self.train_, future)
        return future["unique_id"].map(self.last_).to_numpy(dtype=float)


def test_models_can_ask_for_a_feature_set_and_get_a_builder(tmp_path, monkeypatch):
    monkeypatch.setitem(MODEL_TYPES, "feature_user", _FeatureUser)
    (tmp_path / "default.yaml").write_text(open("config/default.yaml").read())
    (tmp_path / "frequency_profiles.yaml").write_text(open("config/frequency_profiles.yaml").read())
    (tmp_path / "features.yaml").write_text(open("config/features.yaml").read())
    (tmp_path / "models.yaml").write_text(
        "naive:\n  type: naive\nwith_features:\n  type: feature_user\n  features: calendar_only\n"
    )
    cfg = load_config({"frequency": "monthly", "horizon": 3}, config_dir=tmp_path)
    models = build_models_from_config(cfg, tmp_path)
    assert isinstance(models["with_features"].feature_builder, FeatureBuilder)
    clone = models["with_features"].clone()
    assert clone.feature_builder is models["with_features"].feature_builder


def test_model_catalog_rejects_bad_feature_requests(tmp_path):
    specs = {"n": {"type": "naive", "params": {}, "features": "calendar_only"}}
    with pytest.raises(ConfigError, match="does not use feature sets"):
        build_models(specs, 12, {})
    (tmp_path / "models.yaml").write_text("n:\n  type: naive\n  features: 3\n")
    with pytest.raises(ConfigError):
        load_model_specs(tmp_path)


def test_default_models_use_only_feature_sets_without_lags():
    """Baseline and StatsForecast models need no feature set; the MLForecast models use sets that
    only add calendar and identifier features, because MLForecast builds the lags itself."""
    config = load_feature_config()
    specs = load_model_specs()
    for name, spec in specs.items():
        if spec["features"] is None:
            continue
        assert spec["features"] in config.sets, name
        assert not config.sets[spec["features"]].lags, name
        assert not config.sets[spec["features"]].rolling_windows, name
