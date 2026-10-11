"""Accuracy tab data: backtest and live forecasts next to their actuals."""

import json

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.report.accuracy import (
    FINAL_KEY, backtest_rows, encode, live_rows, newest_backtest_file,
)


def _backtest_file():
    rows = []
    for uid in ("a", "b"):
        for model, bump in (("naive", 0.0), ("drift", 1.0)):
            for fold, cutoff in ((1, "2024-03-01"), (2, "2024-06-01")):
                for h in (1, 2):
                    ds = pd.Timestamp(cutoff) + pd.DateOffset(months=h)
                    rows.append({"uuid": uid, "model": model, "fold": fold, "cutoff": cutoff,
                                 "ds": ds.strftime("%Y-%m-%d"), "horizon_step": h, "actual": 10.0 * h,
                                 "forecast": 10.0 + bump, "is_filled": False})
    table = pd.DataFrame(rows)
    table.loc[(table.uuid == "b") & (table.horizon_step == 2), "is_filled"] = True  # not a real actual
    return table


SELECTION = {"series": {
    "a": {"models": [{"name": "naive", "weight": 0.5}, {"name": "drift", "weight": 0.5}]},
    "b": {"models": [{"name": "drift", "weight": 1.0}]},
}}


def test_backtest_rows_have_one_column_per_model_and_the_rebuilt_final():
    rows = backtest_rows(_backtest_file(), SELECTION)
    assert {"naive", "drift", FINAL_KEY, "actual", "horizon_step"} <= set(rows.columns)
    assert len(rows) == 2 * 2 + 2 * 1  # filled periods of b are left out
    a = rows[rows.uuid == "a"]
    assert np.allclose(a[FINAL_KEY], 10.5)  # 0.5 * 10 + 0.5 * 11
    assert np.allclose(rows.loc[rows.uuid == "b", FINAL_KEY], 11.0)


def test_backtest_final_is_empty_without_a_selection_entry():
    rows = backtest_rows(_backtest_file(), {"series": {"a": SELECTION["series"]["a"]}})
    assert rows.loc[rows.uuid == "b", FINAL_KEY].isna().all()
    assert backtest_rows(_backtest_file(), None)[FINAL_KEY].isna().all()


def test_backtest_file_with_missing_columns_is_refused():
    with pytest.raises(ValueError, match="missing"):
        backtest_rows(_backtest_file().drop(columns="actual"), SELECTION)


def _run(folder, run_id, cutoff, final, by_model=None):
    out = folder / run_id / "outputs" / "forecast"
    out.mkdir(parents=True)
    (folder / run_id / "manifest.json").write_text(json.dumps({"input": {"data_cutoff": cutoff}}))
    final.to_csv(out / "forecasts.csv", index=False)
    if by_model is not None:
        by_model.to_csv(out / "forecasts_by_model.csv", index=False)


def test_live_rows_match_earlier_forecasts_with_the_actuals_that_arrived(tmp_path):
    runs = tmp_path / "runs"
    ds = pd.date_range("2025-01-01", periods=3, freq="MS")
    final = pd.DataFrame({"ds": ds, "uuid": "a", "forecast": [1.0, 2.0, 3.0], "market": "DE04"})
    by_model = pd.DataFrame({"uuid": "a", "ds": ds, "naive": [5.0, 5.0, 5.0], "market": "DE04"})
    _run(runs, "20250101_000000", "2024-12-01", final, by_model)
    _run(runs, "20250101_000001", "2024-12-01", final.assign(forecast=9.0), by_model)  # same data: newest wins
    _run(runs, "20250301_000000", "2025-02-01", final.iloc[2:], by_model.iloc[2:])  # no actuals yet
    history = pd.DataFrame({"uuid": "a", "date": pd.date_range("2024-01-01", "2025-02-01", freq="MS"), "target": 4.0})
    rows = live_rows(runs, history)
    assert list(rows["cutoff"].unique()) == [pd.Timestamp("2024-12-01")]
    assert len(rows) == 2  # Jan and Feb have actuals, Mar not yet
    assert rows[FINAL_KEY].tolist() == [9.0, 9.0] and rows["naive"].tolist() == [5.0, 5.0]
    assert rows["horizon_step"].tolist() == [1, 2] and "market" not in rows.columns


def test_live_rows_without_runs_are_empty(tmp_path):
    history = pd.DataFrame({"uuid": ["a"], "date": ["2024-01-01"], "target": [1.0]})
    assert live_rows(tmp_path / "nothing", history).empty


def test_encode_is_columnar_and_indexes_the_report_series():
    rows = backtest_rows(_backtest_file(), SELECTION)
    data = encode({"backtest": rows, "live": pd.DataFrame()}, ["b", "a", "zzz"])
    assert set(data["sources"]) == {"backtest"}
    src = data["sources"]["backtest"]
    assert src["origins"] == ["2024-03-01", "2024-06-01"]
    assert set(src["s"]) == {0, 1}  # "b" is 0, "a" is 1 in the report's series order
    assert len(src["a"]) == len(src["d"]) == len(src["f"]["naive"]) == len(rows)
    assert set(data["models"]) == {"naive", "drift"} and FINAL_KEY in src["f"]
    json.dumps(data, allow_nan=False)


def test_encode_drops_series_not_in_the_report_and_handles_nothing():
    rows = backtest_rows(_backtest_file(), SELECTION)
    assert encode({"backtest": rows}, ["zzz"]) is None
    assert encode({}, ["a"]) is None


def test_large_data_keeps_only_the_selection_models():
    rows = backtest_rows(_backtest_file(), SELECTION)
    data = encode({"backtest": rows}, ["a", "b"], max_values=5, keep_models=["drift"])
    assert data["models"] == ["drift"] and "naive" not in data["sources"]["backtest"]["f"]
    assert "only the models used in the selection" in data["note"]


def test_newest_backtest_file_prefers_the_one_named_by_the_selection(tmp_path):
    for name in ("backtest_forecasts_20260101_0900.csv", "backtest_forecasts_20260102_0900.csv"):
        (tmp_path / name).write_text("x")
    assert newest_backtest_file(tmp_path, None).name == "backtest_forecasts_20260102_0900.csv"
    named = {"meta": {"backtest_forecasts_file": "backtest_forecasts_20260101_0900.csv"}}
    assert newest_backtest_file(tmp_path, named).name == "backtest_forecasts_20260101_0900.csv"
    assert newest_backtest_file(tmp_path / "none", None) is None


def test_report_pipeline_adds_the_accuracy_data_unless_switched_off(tmp_path):
    from gts_forecaster.config import load_config
    from gts_forecaster.pipelines.report import run_report_pipeline

    history = pd.DataFrame({
        "uuid": np.repeat(["a", "b"], 9),
        "date": list(pd.date_range("2024-01-01", periods=9, freq="MS")) * 2,
        "target": 10.0, "market": "DE04", "product": "F0101", "company": "A",
    })
    data_path = tmp_path / "in.csv"
    history.to_csv(data_path, index=False)
    out = tmp_path / "outputs" / "monthly"
    out.mkdir(parents=True)
    pd.DataFrame({"ds": pd.date_range("2024-10-01", periods=2, freq="MS").repeat(2), "uuid": ["a", "b"] * 2,
                  "forecast": 1.0}).to_csv(out / "forecasts.csv", index=False)
    bt = tmp_path / "bt" / "monthly"
    bt.mkdir(parents=True)
    _backtest_file().to_csv(bt / "backtest_forecasts_20260101_0900.csv", index=False)
    cfg = load_config({"data_path": str(data_path), "registry_dir": tmp_path / "reg",
                       "backtesting_dir": tmp_path / "bt", "runs_dir": tmp_path / "runs"})
    result = run_report_pipeline(cfg, output_dir=tmp_path / "outputs")
    acc = json.loads(result.json_path.read_text())["accuracy"]
    assert acc["backtest_file"] == "backtest_forecasts_20260101_0900.csv"
    assert set(acc["sources"]) == {"backtest"} and acc["sources"]["backtest"]["origins"]
    result = run_report_pipeline(cfg, output_dir=tmp_path / "outputs", accuracy=False)
    assert "accuracy" not in json.loads(result.json_path.read_text())
