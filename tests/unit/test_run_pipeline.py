"""run-pipeline: one command for all steps, with a folder per run (log, manifest, copies)."""

import hashlib
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.pipelines.run import default_steps, run_pipeline
from helpers import static_table

N = 24


def _setup(tmp_path, **extra):
    dates = pd.date_range("2022-01-01", periods=N, freq="MS")
    df = pd.concat([
        pd.DataFrame({"unique_id": "linear", "ds": dates, "y": np.arange(1.0, N + 1)}),
        pd.DataFrame({"unique_id": "const", "ds": dates, "y": 5.0}),
    ], ignore_index=True)
    df["is_filled"] = False
    static = static_table(df["unique_id"])
    raw = df.rename(columns={"unique_id": "uuid", "ds": "date", "y": "target"}).merge(
        static.rename(columns={"unique_id": "uuid"}), on="uuid"
    )
    data_path = tmp_path / "input.csv"
    raw.to_csv(data_path, index=False)
    cfg = load_config({
        "frequency": "monthly", "horizon": 3, "n_folds": 2, "top_n": 2,
        "data_path": str(data_path), "registry_dir": tmp_path / "reg",
        "backtesting_dir": tmp_path / "bt", "fitted_models_dir": tmp_path / "fm",
        "runs_dir": tmp_path / "runs", **extra,
    })
    prepared = SimpleNamespace(
        data=df, config=cfg, cutoff=df["ds"].max(),
        fallback=pd.DataFrame(columns=["unique_id"]), static=static,
    )
    calls = []

    def prepare(c):
        calls.append(c)
        return prepared

    return cfg, prepare, calls, data_path


def test_default_steps_depend_on_forecast_mode(tmp_path):
    cfg, *_ = _setup(tmp_path)
    assert default_steps(cfg) == ["backtest", "forecast", "report"]
    cfg, *_ = _setup(tmp_path, forecast_mode="frozen_models")
    assert default_steps(cfg) == ["backtest", "train", "forecast", "report"]


def test_full_run_keeps_a_complete_record(tmp_path):
    cfg, prepare, calls, data_path = _setup(tmp_path)
    result = run_pipeline(cfg, output_dir=tmp_path / "out", prepare=prepare,
                          run_time=datetime(2026, 10, 6, 10, 15, 0))
    assert result.status == "ok" and result.run_id == "20261006_101500"
    assert len(calls) == 1  # the input is read and checked once for all steps
    run_dir = tmp_path / "runs" / "monthly" / "20261006_101500"
    assert result.run_dir == run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "ok" and manifest["finished"]
    assert [s["step"] for s in manifest["steps"]] == ["backtest", "forecast", "report"]
    assert all(s["status"] == "ok" for s in manifest["steps"])
    assert manifest["input"]["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert manifest["input"]["series_for_models"] == 2
    assert manifest["config"]["horizon"] == 3 and manifest["fast_only"] is True
    assert "python" in manifest["environment"]
    # copies of what the steps produced
    assert (run_dir / "outputs" / "backtest" / "selection_fast.json").is_file()
    assert (run_dir / "outputs" / "backtest" / "selection.json").is_file()
    assert (run_dir / "outputs" / "forecast" / "forecast_output.csv").is_file()
    assert (run_dir / "outputs" / "report" / "forecast_report.html").is_file()
    assert (run_dir / "config" / "default.yaml").is_file()
    log = (run_dir / "run.log").read_text()
    assert "step backtest: ok" in log and "step report: ok" in log
    assert (tmp_path / "runs" / "monthly" / "latest.txt").read_text().strip() == "20261006_101500"


def test_a_failing_step_stops_the_run_and_is_recorded(tmp_path, monkeypatch):
    import gts_forecaster.pipelines.forecast as fc

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fc, "run_forecast_pipeline", boom)
    cfg, prepare, _, _ = _setup(tmp_path)
    result = run_pipeline(cfg, output_dir=tmp_path / "out", prepare=prepare)
    assert result.status == "failed"
    status = {s["step"]: s["status"] for s in result.steps}
    assert status == {"backtest": "ok", "forecast": "failed", "report": "skipped"}
    failed = next(s for s in result.steps if s["step"] == "forecast")
    assert failed["error"] == "RuntimeError: disk full"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert "disk full" in (result.run_dir / "run.log").read_text()


def test_a_failing_input_check_fails_the_first_step(tmp_path):
    cfg, _, _, _ = _setup(tmp_path)

    def bad_input(c):
        raise ValueError("input breaks the contract")

    result = run_pipeline(cfg, output_dir=tmp_path / "out", prepare=bad_input)
    assert result.status == "failed"
    assert result.steps[0]["error"] == "ValueError: input breaks the contract"
    assert [s["status"] for s in result.steps[1:]] == ["skipped", "skipped"]


def test_selected_steps_run_in_pipeline_order(tmp_path):
    cfg, prepare, _, _ = _setup(tmp_path)
    run_pipeline(cfg, steps=["backtest"], output_dir=tmp_path / "out", prepare=prepare)
    result = run_pipeline(cfg, steps=["report", "forecast"], output_dir=tmp_path / "out", prepare=prepare)
    assert [s["step"] for s in result.steps] == ["forecast", "report"]
    assert result.status == "ok"


def test_frozen_models_run_trains_then_forecasts_from_the_saved_models(tmp_path):
    cfg, prepare, _, _ = _setup(tmp_path, forecast_mode="frozen_models")
    result = run_pipeline(cfg, output_dir=tmp_path / "out", prepare=prepare)
    assert result.status == "ok"
    assert (result.run_dir / "outputs" / "train" / "manifest.json").is_file()
    timings = pd.read_csv(result.run_dir / "outputs" / "forecast" / "model_timings.csv")
    assert set(timings["status"]) == {"updated"}


def test_runs_in_the_same_second_get_separate_folders(tmp_path):
    cfg, prepare, _, _ = _setup(tmp_path)
    when = datetime(2026, 10, 6, 10, 15, 0)
    first = run_pipeline(cfg, steps=["backtest"], prepare=prepare, run_time=when, output_dir=tmp_path / "o")
    second = run_pipeline(cfg, steps=["backtest"], prepare=prepare, run_time=when, output_dir=tmp_path / "o")
    assert first.run_dir != second.run_dir and second.run_id == "20261006_101500_1"


def test_unknown_step_is_rejected(tmp_path):
    cfg, prepare, _, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="unknown step"):
        run_pipeline(cfg, steps=["deploy"], prepare=prepare)
