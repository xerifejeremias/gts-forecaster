"""Automatic runs: schedule (Task Scheduler commands), jobs, clean-up, alerts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, project_root
from gts_forecaster.ops.alerts import send_alert
from gts_forecaster.ops.housekeeping import clean_backtests, clean_runs
from gts_forecaster.ops.jobs import run_job
from gts_forecaster.ops.schedule import AlertSettings, Schedule, Trigger, create_command, load_schedule
from gts_forecaster.pipelines.run import RunResult


def _schedule_dir(tmp_path, text):
    (tmp_path / "schedule.yaml").write_text(text, encoding="utf-8")
    return tmp_path


# ---- schedule ------------------------------------------------------------------------------------


def test_shipped_schedule_runs_monthly_forecast_and_quarterly_backtest():
    schedule = load_schedule()
    monthly = schedule.jobs["monthly"]
    assert monthly["forecast"] == Trigger("month", "06:00", day=3)
    assert monthly["backtest"] == Trigger("months", "02:00", day=1, months=3)
    assert schedule.horizons["monthly"] == 18 and schedule.keep_runs == 36


def test_every_rhythm_becomes_the_right_schtasks_trigger(tmp_path):
    text = """jobs:
  daily:
    forecast: {every: day, time: "05:30"}
    backtest: {every_months: 6, day: 2, time: "01:00"}
  weekly:
    forecast: {every: week, weekday: fri, time: "07:00"}
  monthly:
    forecast: {every: month, day: 28, time: "23:59"}
"""
    schedule = load_schedule(_schedule_dir(tmp_path, text))
    assert schedule.jobs["daily"]["forecast"].schtasks_args() == ["/SC", "DAILY", "/ST", "05:30"]
    assert schedule.jobs["daily"]["backtest"].schtasks_args() == ["/SC", "MONTHLY", "/MO", "6", "/D", "2", "/ST", "01:00"]
    assert schedule.jobs["weekly"]["forecast"].schtasks_args() == ["/SC", "WEEKLY", "/D", "FRI", "/ST", "07:00"]
    assert schedule.jobs["monthly"]["forecast"].schtasks_args() == ["/SC", "MONTHLY", "/D", "28", "/ST", "23:59"]
    assert "backtest" not in schedule.jobs["weekly"]  # not listed: does not run
    command = create_command("weekly", "forecast", schedule.jobs["weekly"]["forecast"], Path("C:/p"))
    assert command[:6] == ["schtasks", "/Create", "/F", "/TN", "GTS Forecaster\\weekly forecast", "/TR"]
    assert command[6].endswith('scheduled_run.bat" --frequency weekly --job forecast')


@pytest.mark.parametrize("bad", [
    "jobs: {hourly: {forecast: {every: day}}}",
    "jobs: {daily: {forecast: {every: hour}}}",
    "jobs: {daily: {forecast: {every: day, time: '25:00'}}}",
    "jobs: {monthly: {forecast: {every: month, day: 31}}}",
    "jobs: {weekly: {forecast: {every: week, weekday: FUNDAY}}}",
    "jobs: {monthly: {backtest: {every_months: 0}}}",
    "jobs: {monthly: {report: {every: day}}}",
    "alerts: {pager: true}",
])
def test_bad_schedules_are_refused(tmp_path, bad):
    with pytest.raises(ConfigError, match="schedule"):
        load_schedule(_schedule_dir(tmp_path, bad))


# ---- clean-up ------------------------------------------------------------------------------------


def test_clean_runs_keeps_the_newest_and_the_latest(tmp_path):
    folder = tmp_path / "monthly"
    for run_id in ("20260101_000000", "20260201_000000", "20260301_000000", "20260401_000000"):
        (folder / run_id).mkdir(parents=True)
    (folder / "latest.txt").write_text("20260101_000000")  # protected even though it is old
    (folder / "notes").mkdir()  # not a run folder: left alone
    deleted = clean_runs(tmp_path, "monthly", keep=2)
    assert deleted == ["20260201_000000"]
    assert sorted(p.name for p in folder.iterdir()) == ["20260101_000000", "20260301_000000", "20260401_000000",
                                                         "latest.txt", "notes"]


def test_clean_backtests_keeps_the_selected_one(tmp_path):
    bt, registry = tmp_path / "bt" / "daily", tmp_path / "registry" / "daily"
    bt.mkdir(parents=True)
    registry.mkdir(parents=True)
    stamps = ["20260101_0100", "20260201_0100", "20260301_0100", "20260401_0100"]
    for stamp in stamps:
        for kind in ("forecasts", "timings"):
            (bt / f"backtest_{kind}_{stamp}.csv").write_text("x")
    (registry / "selection_fast.json").write_text(json.dumps({"meta": {"backtest_forecasts_file": f"backtest_forecasts_{stamps[0]}.csv"}}))
    deleted = clean_backtests(tmp_path / "bt", tmp_path / "registry", "daily", keep=1)
    assert deleted == [stamps[1], stamps[2]]
    assert sorted(p.name for p in bt.iterdir()) == sorted(
        f"backtest_{k}_{s}.csv" for k in ("forecasts", "timings") for s in (stamps[0], stamps[3]))


# ---- jobs ----------------------------------------------------------------------------------------


def _fake_run(status="ok", flagged=0, calls=None):
    def run(cfg, steps=None, config_dir=None, output_dir=None):
        if calls is not None:
            calls.append((cfg.frequency, cfg.horizon, steps))
        folder = Path(output_dir) / cfg.frequency
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "forecast_report.html").write_text("<html></html>")
        pd.DataFrame({"uuid": [f"s{i}" for i in range(flagged)], "check": "zero"}).to_csv(
            folder / "forecast_checks.csv", index=False)
        failed = status != "ok"
        return RunResult("20260101_000000", Path(cfg.runs_dir), status,
                         [{"step": "forecast", "status": "failed" if failed else "ok",
                           "error": "ValueError: boom" if failed else None}], Path("m.json"))
    return run


def _job_dirs(tmp_path):
    return {"runs_dir": tmp_path / "runs", "registry_dir": tmp_path / "registry", "backtesting_dir": tmp_path / "bt"}


def _patch_config(monkeypatch, tmp_path):
    from gts_forecaster.ops import jobs as jobs_module

    real = jobs_module.load_config
    monkeypatch.setattr(jobs_module, "load_config", lambda overrides, config_dir=None: real({**overrides, **_job_dirs(tmp_path)}))


def test_forecast_job_runs_forecast_and_report_and_writes_a_status(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)
    calls = []
    schedule = Schedule(horizons={"monthly": 18}, report_copy_dir=tmp_path / "share")
    code = run_job("monthly", "forecast", schedule=schedule, output_dir=tmp_path / "out", run=_fake_run(calls=calls))
    assert code == 0 and calls == [("monthly", 18, ["forecast", "report"])]
    status = json.loads((tmp_path / "runs" / "monthly" / "last_job.json").read_text())
    assert status["status"] == "ok" and status["flagged_series"] == 0
    assert (tmp_path / "share" / "monthly_forecast_report.html").is_file()
    assert not (tmp_path / "runs" / "alerts.log").exists()  # nothing to report


def test_backtest_job_runs_every_step(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)
    calls = []
    run_job("daily", "backtest", schedule=Schedule(), output_dir=tmp_path / "out", run=_fake_run(calls=calls))
    assert calls == [("daily", 14, None)]  # None = run-pipeline's default steps; horizon from the profile


def test_failed_job_alerts_and_exits_1(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)
    code = run_job("monthly", "forecast", schedule=Schedule(), output_dir=tmp_path / "out", run=_fake_run("failed"))
    assert code == 1
    log = (tmp_path / "runs" / "alerts.log").read_text(encoding="utf-8")
    assert "monthly forecast FAILED" in log and "ValueError: boom" in log


def test_flagged_series_alert_only_when_asked(tmp_path, monkeypatch):
    _patch_config(monkeypatch, tmp_path)
    run_job("weekly", "forecast", schedule=Schedule(), output_dir=tmp_path / "out", run=_fake_run(flagged=2))
    assert "flagged 2 series" in (tmp_path / "runs" / "alerts.log").read_text(encoding="utf-8")
    quiet = Schedule(alerts=AlertSettings(on_flags=False))
    (tmp_path / "runs" / "alerts.log").unlink()
    run_job("weekly", "forecast", schedule=quiet, output_dir=tmp_path / "out", run=_fake_run(flagged=2))
    assert not (tmp_path / "runs" / "alerts.log").exists()


# ---- alerts --------------------------------------------------------------------------------------


def test_alert_by_email_and_webhook(tmp_path, monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["server"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, message):
            sent["to"], sent["subject"] = message["To"], message["Subject"]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: sent.setdefault("hook", request.full_url) and FakeResponse())
    monkeypatch.setenv("GTS_SMTP_PASSWORD", "pw")
    monkeypatch.setenv("TEAMS_HOOK", "https://example.invalid/hook")
    settings = AlertSettings(email_to=("me@example.com",), smtp_host="smtp.example.com", smtp_user="me",
                             webhook_env="TEAMS_HOOK")
    reached = send_alert("subject", "body", settings, tmp_path / "alerts.log")
    assert reached == ["log", "email", "webhook"]
    assert sent["server"] == ("smtp.example.com", 587) and sent["tls"] and sent["login"] == ("me", "pw")
    assert sent["to"] == "me@example.com" and sent["hook"] == "https://example.invalid/hook"


def test_a_failing_email_never_breaks_the_job(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr("smtplib.SMTP", broken)
    settings = AlertSettings(email_to=("me@example.com",), smtp_host="smtp.example.com")
    assert send_alert("s", "b", settings, tmp_path / "alerts.log") == ["log"]
    assert "s" in (tmp_path / "alerts.log").read_text(encoding="utf-8")


def test_wrapper_script_exists():
    text = (project_root() / "scripts" / "scheduled_run.bat").read_text(encoding="utf-8")
    assert "gts_forecaster.ops.jobs" in text and "scheduler.log" in text
