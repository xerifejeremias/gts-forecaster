"""Models tab data: run time per model, own top-N count and use in the selection."""

import pandas as pd

from gts_forecaster.report.models import model_summary, timings_file_for

SELECTION = {
    "meta": {"top_n": 2, "models": ["naive", "drift", "ets", "slow", "broken"]},
    "series": {
        "a": {"models": [{"name": "ets", "weight": 0.5}, {"name": "drift", "weight": 0.5}]},
        "b": {"models": [{"name": "ets", "weight": 0.5}, {"name": "naive", "weight": 0.5}]},
    },
}
SCORES = pd.DataFrame({
    "uuid": ["a"] * 4 + ["b"] * 4 + ["b"],
    "model": ["ets", "slow", "drift", "naive", "ets", "drift", "slow", "naive", "ensemble"],
    "rank": [1, 2, 3, 4, 1, 2, 3, 4, None],
})
TIMINGS = pd.DataFrame({
    "fold": [1, 2, 1, 2, 1, 2, 1, 2, 1],
    "model": ["naive", "naive", "drift", "drift", "ets", "ets", "slow", "slow", "broken"],
    "n_series": [2, 2, 2, 2, 2, 2, 2, 2, 0],
    "fit_seconds": [0.1, 0.3, 0.1, 0.1, 1.0, 3.0, 10.0, 20.0, 0.0],
    "predict_seconds": [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1.0, 1.0, 0.0],
    "status": ["ok"] * 8 + ["failed"],
})


def _by_name(summary):
    return {m["name"]: m for m in summary["models"]}


def test_run_time_is_the_mean_fit_plus_predict_per_fold():
    models = _by_name(model_summary(SELECTION, SCORES, TIMINGS))
    assert models["ets"]["seconds"] == 2.0
    assert models["slow"]["seconds"] == 16.0
    assert models["slow"]["n_folds"] == 2
    assert models["broken"]["seconds"] is None
    assert models["broken"]["status"] == "failed"
    assert models["broken"]["n_folds"] == 0


def test_top_n_and_use_in_final_forecasts_are_counted_per_series():
    summary = model_summary(SELECTION, SCORES, TIMINGS)
    models = _by_name(summary)
    assert summary["top_n"] == 2 and summary["has_ranks"]
    assert models["ets"]["top_n_series"] == 2 and models["ets"]["used_series"] == 2
    assert models["slow"]["top_n_series"] == 1 and models["slow"]["used_series"] == 0
    assert models["naive"]["top_n_series"] == 0 and models["naive"]["used_series"] == 1
    assert "ensemble" not in models  # the scores file's ensemble row is not a model


def test_missing_inputs_give_partial_or_no_summary():
    assert model_summary(None, None, None) is None
    summary = model_summary(SELECTION, None, None)
    assert not summary["has_ranks"]
    assert all(m["seconds"] is None and m["status"] == "not run" for m in summary["models"])


def test_timings_file_shares_the_backtest_stamp(tmp_path):
    forecasts = tmp_path / "backtest_forecasts_20261012_0927.csv"
    assert timings_file_for(forecasts) is None
    timings = tmp_path / "backtest_timings_20261012_0927.csv"
    timings.write_text("fold,model\n")
    assert timings_file_for(forecasts) == timings
    assert timings_file_for(None) is None
