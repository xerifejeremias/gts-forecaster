"""The automatic-run schedule (config/schedule.yaml) and its Windows Task Scheduler tasks.

    poetry run schedule show       the jobs and the exact schtasks commands
    poetry run schedule install    create (or update) the tasks
    poetry run schedule remove     delete them

Every task runs scripts/scheduled_run.bat, which calls `gts_forecaster.ops.jobs` with the frequency
and the job (forecast or backtest) and appends its output to runs/scheduler.log.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gts_forecaster.config import ConfigError, project_root

FREQUENCIES = ("yearly", "monthly", "weekly", "daily")
JOBS = ("forecast", "backtest")
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
TASK_FOLDER = "GTS Forecaster"
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class Trigger:
    """When a job runs: every day / week / month, or every `months` months (backtest)."""

    every: str  # day | week | month | months
    time: str = "06:00"
    weekday: str = "MON"
    day: int = 1
    months: int = 1

    def describe(self) -> str:
        if self.every == "day":
            return f"every day at {self.time}"
        if self.every == "week":
            return f"every {self.weekday} at {self.time}"
        if self.every == "month":
            return f"day {self.day} of every month at {self.time}"
        return f"day {self.day} of every {self.months} months at {self.time}"

    def schtasks_args(self) -> list[str]:
        """The /SC ... part of a schtasks /Create command."""
        if self.every == "day":
            return ["/SC", "DAILY", "/ST", self.time]
        if self.every == "week":
            return ["/SC", "WEEKLY", "/D", self.weekday, "/ST", self.time]
        if self.every == "month":
            return ["/SC", "MONTHLY", "/D", str(self.day), "/ST", self.time]
        return ["/SC", "MONTHLY", "/MO", str(self.months), "/D", str(self.day), "/ST", self.time]


@dataclass(frozen=True)
class AlertSettings:
    on_failure: bool = True
    on_flags: bool = True
    email_to: tuple[str, ...] = ()
    email_from: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password_env: str = "GTS_SMTP_PASSWORD"
    webhook_env: str | None = None


@dataclass(frozen=True)
class Schedule:
    jobs: dict[str, dict[str, Trigger]] = field(default_factory=dict)  # frequency -> job -> trigger
    horizons: dict[str, int] = field(default_factory=dict)
    keep_runs: int = 36
    keep_backtests: int = 4
    report_copy_dir: Path | None = None
    alerts: AlertSettings = AlertSettings()


def _int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ConfigError(f"schedule: {name} must be a whole number from {low} to {high}, got {value!r}")
    return value


def _trigger(where: str, job: str, raw: Any) -> Trigger:
    if not isinstance(raw, dict):
        raise ConfigError(f"schedule: {where}.{job} must be a mapping")
    time = str(raw.get("time", "06:00"))
    if not _TIME.match(time):
        raise ConfigError(f"schedule: {where}.{job}.time must be HH:MM (24 h), got {time!r}")
    if job == "backtest":
        unknown = set(raw) - {"every_months", "day", "time"}
        if unknown:
            raise ConfigError(f"schedule: {where}.backtest takes every_months, day, time; not {sorted(unknown)}")
        return Trigger("months", time, day=_int(raw.get("day", 1), f"{where}.backtest.day", 1, 28),
                       months=_int(raw.get("every_months", 3), f"{where}.backtest.every_months", 1, 12))
    unknown = set(raw) - {"every", "weekday", "day", "time"}
    if unknown:
        raise ConfigError(f"schedule: {where}.forecast takes every, weekday, day, time; not {sorted(unknown)}")
    every = raw.get("every")
    if every not in ("day", "week", "month"):
        raise ConfigError(f"schedule: {where}.forecast.every must be day, week or month, got {every!r}")
    weekday = str(raw.get("weekday", "MON")).upper()
    if weekday not in WEEKDAYS:
        raise ConfigError(f"schedule: {where}.forecast.weekday must be one of {list(WEEKDAYS)}, got {weekday!r}")
    return Trigger(every, time, weekday=weekday, day=_int(raw.get("day", 1), f"{where}.forecast.day", 1, 28))


def load_schedule(config_dir: Path | None = None) -> Schedule:
    """Read and validate config/schedule.yaml (no file: nothing scheduled)."""
    path = (Path(config_dir) if config_dir else project_root() / "config") / "schedule.yaml"
    if not path.is_file():
        return Schedule()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping")
    jobs: dict[str, dict[str, Trigger]] = {}
    horizons: dict[str, int] = {}
    for frequency, entry in (raw.get("jobs") or {}).items():
        if frequency not in FREQUENCIES:
            raise ConfigError(f"schedule: unknown frequency {frequency!r}; use {list(FREQUENCIES)}")
        if not isinstance(entry, dict) or set(entry) - {"forecast", "backtest", "horizon"}:
            raise ConfigError(f"schedule: jobs.{frequency} takes forecast, backtest and horizon")
        jobs[frequency] = {job: _trigger(f"jobs.{frequency}", job, entry[job]) for job in JOBS if entry.get(job)}
        if entry.get("horizon") is not None:
            horizons[frequency] = _int(entry["horizon"], f"jobs.{frequency}.horizon", 1, 10_000)
    alerts_raw = raw.get("alerts") or {}
    known = set(AlertSettings.__dataclass_fields__)
    if set(alerts_raw) - known:
        raise ConfigError(f"schedule: unknown alert setting(s) {sorted(set(alerts_raw) - known)}")
    email_to = alerts_raw.get("email_to") or ()
    alerts = AlertSettings(**{**alerts_raw, "email_to": tuple([email_to] if isinstance(email_to, str) else email_to)})
    copy_dir = raw.get("report_copy_dir")
    return Schedule(
        jobs=jobs, horizons=horizons,
        keep_runs=_int(raw.get("keep_runs", 36), "keep_runs", 1, 100_000),
        keep_backtests=_int(raw.get("keep_backtests", 4), "keep_backtests", 1, 1000),
        report_copy_dir=Path(copy_dir) if copy_dir else None,
        alerts=alerts,
    )


def task_name(frequency: str, job: str) -> str:
    return f"{TASK_FOLDER}\\{frequency} {job}"


def create_command(frequency: str, job: str, trigger: Trigger, root: Path | None = None) -> list[str]:
    """The schtasks /Create command (as an argument list) for one job."""
    script = (root or project_root()) / "scripts" / "scheduled_run.bat"
    action = f'"{script}" --frequency {frequency} --job {job}'
    return ["schtasks", "/Create", "/F", "/TN", task_name(frequency, job), "/TR", action,
            *trigger.schtasks_args()]


def delete_command(frequency: str, job: str) -> list[str]:
    return ["schtasks", "/Delete", "/F", "/TN", task_name(frequency, job)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show, create or delete the automatic runs (Windows Task Scheduler).")
    parser.add_argument("action", choices=["show", "install", "remove"], nargs="?", default="show")
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        schedule = load_schedule(args.config_dir)
    except ConfigError as exc:
        print(f"Schedule error: {exc}", file=sys.stderr)
        return 2
    planned = [(f, job, t) for f, jobs in schedule.jobs.items() for job, t in jobs.items()]
    if not planned:
        print("No jobs in config/schedule.yaml.")
        return 0
    failed = 0
    for frequency, job, trigger in planned:
        print(f"{task_name(frequency, job):32s} {trigger.describe()}")
        command = create_command(frequency, job, trigger) if args.action != "remove" else delete_command(frequency, job)
        if args.action == "show":
            print("    " + subprocess.list2cmdline(command))
            continue
        done = subprocess.run(command, capture_output=True, text=True)
        print("    " + (done.stdout or done.stderr).strip())
        failed += done.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
