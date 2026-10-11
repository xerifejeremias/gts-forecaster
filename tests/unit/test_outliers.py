import numpy as np
import pandas as pd
import pytest

from gts_forecaster.cleaning.outliers import MIN_OBSERVATIONS, handle_outliers, robust_scores
from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.evaluation.backtest import run_backtest
from gts_forecaster.models.catalog import build_models, load_model_specs
from helpers import BASELINES
from helpers import use_core_catalog

@pytest.fixture(autouse=True)
def _core_catalog(monkeypatch):
    use_core_catalog(monkeypatch)  # pipeline tests: no tuning variants of the tree models



def _frame(y, uid="a", start="2020-01-01", freq="MS", filled=None):
    ds = pd.date_range(start, periods=len(y), freq=freq)
    df = pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.asarray(y, dtype=float)})
    df["is_filled"] = False if filled is None else filled
    return df


def _noisy_line(n=48, seed=0):
    rng = np.random.default_rng(seed)
    return 100 + 2.0 * np.arange(n) + rng.normal(0, 1.0, n)


def test_spike_is_replaced_by_interpolation():
    y = _noisy_line()
    y[20] += 500
    out = handle_outliers(_frame(y), action="interpolate")
    assert out.report["ds"].dt.strftime("%Y-%m").tolist() == ["2021-09"]
    assert out.data["is_outlier"].sum() == 1
    cleaned = out.data["y"].to_numpy()
    assert cleaned[20] == pytest.approx((y[19] + y[21]) / 2)
    np.testing.assert_allclose(np.delete(cleaned, 20), np.delete(y, 20))
    assert out.report["y_raw"].iloc[0] == pytest.approx(y[20])


def test_negative_spike_and_two_separate_spikes():
    y = _noisy_line()
    y[10] -= 400
    y[30] += 400
    assert handle_outliers(_frame(y)).data["is_outlier"].sum() == 2


def test_clean_noisy_series_is_untouched():
    for seed in range(5):
        out = handle_outliers(_frame(_noisy_line(seed=seed)))
        assert out.data["is_outlier"].sum() == 0
        assert out.report.empty


def test_trend_and_smooth_cycle_are_not_flagged():
    t = np.arange(60)
    y = 50 + 3 * t + 40 * np.sin(2 * np.pi * t / 9)
    assert handle_outliers(_frame(y)).data["is_outlier"].sum() == 0


def test_seasonal_peaks_are_not_flagged_but_a_spike_in_a_seasonal_series_is():
    t = np.arange(72)
    rng = np.random.default_rng(1)
    y = 200 + 80 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, len(t))
    assert handle_outliers(_frame(y), season_length=12).data["is_outlier"].sum() == 0
    y[40] += 300
    out = handle_outliers(_frame(y), season_length=12)
    assert out.data["is_outlier"].to_numpy().nonzero()[0].tolist() == [40]


def test_spike_in_a_constant_series_is_flagged():
    y = np.full(30, 7.0)
    y[15] = 70.0
    out = handle_outliers(_frame(y))
    assert out.data["is_outlier"].to_numpy().nonzero()[0].tolist() == [15]
    np.testing.assert_allclose(out.data["y"].to_numpy(), 7.0, rtol=1e-5)  # clipped to the bound


def test_first_and_last_two_points_are_never_flagged():
    y = _noisy_line()
    y[[0, 1, -2, -1]] += 1000
    assert handle_outliers(_frame(y)).data["is_outlier"].sum() == 0


def test_filled_periods_are_never_flagged():
    y = _noisy_line()
    y[20] += 500
    filled = np.zeros(len(y), bool)
    filled[20] = True
    out = handle_outliers(_frame(y, filled=filled), max_share=1.0)
    assert not out.data["is_outlier"].iloc[20]


def test_short_series_is_left_alone():
    y = np.array([1.0, 2.0, 3.0, 400.0, 5.0, 6.0, 7.0])
    assert len(y) < MIN_OBSERVATIONS
    assert handle_outliers(_frame(y)).data["y"].tolist() == y.tolist()


def test_series_with_too_many_flags_is_left_unchanged():
    y = _noisy_line(40)
    y[[8, 12, 16, 20, 24, 28]] += 800  # 15% of the points
    out = handle_outliers(_frame(y), max_share=0.1)
    assert out.data["y"].tolist() == pytest.approx(y.tolist())
    assert out.report.empty
    assert handle_outliers(_frame(y), max_share=0.3).data["is_outlier"].sum() == 6


def test_threshold_controls_sensitivity():
    y = _noisy_line()
    y[20] += 12
    assert handle_outliers(_frame(y), threshold=20).data["is_outlier"].sum() == 0
    assert handle_outliers(_frame(y), threshold=5).data["is_outlier"].sum() == 1


def test_method_none_changes_nothing():
    y = _noisy_line()
    y[20] += 500
    out = handle_outliers(_frame(y), method="none")
    assert out.data["y"].tolist() == pytest.approx(y.tolist()) and out.report.empty


def test_series_are_handled_independently_and_other_columns_kept():
    a, b = _noisy_line(seed=1), _noisy_line(seed=2) * 100
    a[20] += 500
    df = pd.concat([_frame(a, "a"), _frame(b, "b")], ignore_index=True)
    df["x1"] = 3.0
    out = handle_outliers(df)
    assert out.report["unique_id"].tolist() == ["a"]
    assert (out.data["x1"] == 3.0).all() and len(out.data) == len(df)
    assert out.data[out.data.unique_id == "b"]["y"].tolist() == pytest.approx(b.tolist())


def test_does_not_modify_its_input():
    y = _noisy_line()
    y[20] += 500
    df = _frame(y)
    before = df.copy()
    handle_outliers(df)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "nope"}, {"threshold": 0}, {"threshold": -1}, {"max_share": 0}, {"max_share": 1.5}],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        handle_outliers(_frame(_noisy_line()), **kwargs)


def test_robust_scores_are_nan_at_the_ends():
    s = robust_scores(_noisy_line())
    assert np.isnan(s[[0, 1, -2, -1]]).all() and not np.isnan(s[2:-2]).any()


# ---- config -------------------------------------------------------------------------------


def test_config_defaults_and_validation():
    cfg = load_config({"frequency": "monthly"})
    assert (cfg.outlier_method, cfg.outlier_threshold, cfg.outlier_max_share) == ("mad", 5.0, 0.1)
    for bad in (
        {"outlier_method": "x"}, {"outlier_threshold": 0}, {"outlier_threshold": "5"},
        {"outlier_max_share": 0}, {"outlier_max_share": 2},
    ):
        with pytest.raises(ConfigError):
            load_config({"frequency": "monthly", **bad})


# ---- backtest -----------------------------------------------------------------------------


def _panel_with_spike(spike_at):
    y = 100 + 2.0 * np.arange(40.0)
    y[spike_at] += 1000
    df = _frame(y)
    return df


def _naive_only():
    return {"naive": build_models(load_model_specs(types=BASELINES), 12)["naive"]}


def test_backtest_cleans_training_data_but_scores_against_raw_actuals():
    df = _panel_with_spike(34)  # in the training data of the newest fold (cutoff 37, horizon 3)
    raw = run_backtest(df, _naive_only(), horizon=3, pandas_freq="MS", n_folds=1)
    clean = run_backtest(df, _naive_only(), horizon=3, pandas_freq="MS", n_folds=1,
                         outlier_method="mad")
    assert len(clean.outliers) == 1 and clean.outliers["fold"].iloc[0] == 1
    assert clean.outliers["y_raw"].iloc[0] == pytest.approx(df["y"].iloc[34])
    # the spike is far from the last training value, so cleaning does not change naive here;
    # the actuals in the predictions are always the raw values
    assert clean.predictions["y"].tolist() == raw.predictions["y"].tolist()


def test_backtest_does_not_clean_the_test_window():
    df = _panel_with_spike(38)  # inside the test window of the newest fold
    out = run_backtest(df, _naive_only(), horizon=3, pandas_freq="MS", n_folds=1,
                       outlier_method="mad")
    assert out.outliers.empty
    assert out.predictions["y"].max() > 1000


def test_backtest_cleaning_changes_forecast_when_last_points_are_affected():
    df = _panel_with_spike(35)  # the last training value before cutoff index 36 is index 36
    y = df["y"].to_numpy().copy()
    df2 = _frame(np.append(y[:36], [y[35] - 1000 + 1000]))  # no-op sanity frame
    assert len(df2) == 37
    spiky = _frame(100 + 2.0 * np.arange(40.0))
    spiky.loc[36, "y"] += 1000  # last training row of the newest fold
    # edge rows are never flagged, so a spike on the last training point is kept
    out = run_backtest(spiky, _naive_only(), horizon=3, pandas_freq="MS", n_folds=1,
                       outlier_method="mad")
    assert out.outliers.empty
    spiky2 = _frame(100 + 2.0 * np.arange(40.0))
    spiky2.loc[30, "y"] += 1000
    out2 = run_backtest(spiky2, _naive_only(), horizon=3, pandas_freq="MS", n_folds=2, step=3,
                        outlier_method="mad")
    assert set(out2.outliers["fold"]) == {1, 2}  # each fold finds it in its own training data


def test_pipelines_write_outlier_reports(tmp_path):
    from datetime import datetime
    from types import SimpleNamespace

    from gts_forecaster.pipelines.backtest import run_backtest_pipeline
    from gts_forecaster.pipelines.forecast import run_forecast_pipeline

    cfg = load_config({"frequency": "monthly", "horizon": 3, "n_folds": 1, "top_n": 1,
                       "registry_dir": tmp_path, "backtesting_dir": tmp_path / "bt"})
    df = _panel_with_spike(20)
    from helpers import static_table

    prepared = SimpleNamespace(data=df, config=cfg, cutoff=df["ds"].max(),
                               fallback=pd.DataFrame(columns=["unique_id"]), static=static_table(df["unique_id"]))
    bt = run_backtest_pipeline(prepared, run_time=datetime(2026, 10, 5, 13, 0))
    assert (bt.forecasts_path.parent / "backtest_outliers_20261005_1300.csv").is_file()
    fc = run_forecast_pipeline(prepared, output_dir=tmp_path / "out")
    reported = pd.read_csv(tmp_path / "out" / "monthly" / "outliers.csv")
    assert list(reported.columns) == ["uuid", "ds", "y_raw", "y_clean", "score"]
    assert len(reported) == len(fc.outliers) == 1

    off = load_config({"frequency": "monthly", "horizon": 3, "n_folds": 1, "top_n": 1,
                       "registry_dir": tmp_path / "r2", "backtesting_dir": tmp_path / "bt2",
                       "outlier_method": "none"})
    prepared.config = off
    run_backtest_pipeline(prepared)
    run_forecast_pipeline(prepared, output_dir=tmp_path / "out2")
    assert not (tmp_path / "out2" / "monthly" / "outliers.csv").exists()


# ---- what flagged points are replaced with ---------------------------------------------------


def _spiked(delta, at=20):
    y = _noisy_line()
    y[at] += delta
    return y


def test_clip_keeps_a_high_value_high_but_not_extreme():
    y = _spiked(500)
    clipped = handle_outliers(_frame(y), action="clip")
    value = clipped.data["y"].iloc[20]
    interpolated = handle_outliers(_frame(y), action="interpolate").data["y"].iloc[20]
    assert interpolated < value < y[20]  # above the neighbours' level, far below the raw spike
    expected = 100 + 2.0 * 20
    assert 0 < value - expected < 15  # about threshold x spread above the expected value
    assert clipped.report["y_clean"].iloc[0] == pytest.approx(value)


def test_clip_keeps_a_low_value_low():
    y = _spiked(-500)
    value = handle_outliers(_frame(y), action="clip").data["y"].iloc[20]
    interpolated = handle_outliers(_frame(y), action="interpolate").data["y"].iloc[20]
    assert y[20] < value < interpolated


def test_clip_bound_follows_the_threshold():
    y = _spiked(500)
    tight = handle_outliers(_frame(y), action="clip", threshold=4).data["y"].iloc[20]
    loose = handle_outliers(_frame(y), action="clip", threshold=8).data["y"].iloc[20]
    assert tight < loose


def test_median_and_mean_use_the_unflagged_values_of_the_series():
    y = _spiked(500)
    others = np.delete(y, 20)
    med = handle_outliers(_frame(y), action="median").data["y"].iloc[20]
    mean = handle_outliers(_frame(y), action="mean").data["y"].iloc[20]
    assert med == pytest.approx(np.median(others))
    assert mean == pytest.approx(np.mean(others))


@pytest.mark.parametrize("action", ["interpolate", "clip", "median", "mean"])
def test_every_action_only_changes_flagged_points(action):
    y = _spiked(500)
    out = handle_outliers(_frame(y), action=action)
    np.testing.assert_allclose(np.delete(out.data["y"].to_numpy(), 20), np.delete(y, 20))
    assert out.data["is_outlier"].sum() == 1 and abs(out.data["y"].iloc[20]) < 1000


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError):
        handle_outliers(_frame(_noisy_line()), action="drop")


def test_action_config_default_and_validation():
    assert load_config({"frequency": "monthly"}).outlier_action == "clip"
    assert load_config({"frequency": "monthly", "outlier_action": "median"}).outlier_action == "median"
    with pytest.raises(ConfigError):
        load_config({"frequency": "monthly", "outlier_action": "drop"})


def test_backtest_uses_the_chosen_action():
    df = _panel_with_spike(20)
    base = dict(horizon=3, pandas_freq="MS", n_folds=1, outlier_method="mad")
    clip = run_backtest(df, _naive_only(), outlier_action="clip", **base)
    interp = run_backtest(df, _naive_only(), outlier_action="interpolate", **base)
    assert clip.outliers["y_clean"].iloc[0] != interp.outliers["y_clean"].iloc[0]


def test_default_action_is_clip():
    import inspect

    y = _spiked(500)
    assert inspect.signature(handle_outliers).parameters["action"].default == "clip"
    default = handle_outliers(_frame(y)).data["y"].iloc[20]
    assert default == handle_outliers(_frame(y), action="clip").data["y"].iloc[20]
    assert default != handle_outliers(_frame(y), action="interpolate").data["y"].iloc[20]
    assert inspect.signature(run_backtest).parameters["outlier_action"].default == "clip"
