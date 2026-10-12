"""One automatic job: what Windows Task Scheduler runs (through scripts/scheduled_run.bat).

    poetry run scheduled-run --frequency monthly --job forecast    forecast + report
    poetry run scheduled-run --frequency monthly --job backtest    the full pipeline

After the pipeline (recorded as usual in runs/<frequency>/<run id>/): copy the report to
`report_copy_dir` (if set), clean up old runs and backtests, write runs/<frequency>/last_job.json and send
an alert when a step failed or the forecast checks flagged series (config/schedule.yaml). Exit code 0 when
every step succeeded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.forecast_checks import CHECKS_FILE
from gts_forecaster.ops.alerts import send_alert
from gts_forecaster.ops.housekeeping import clean_backtests, clean_runs
from gts_forecaster.ops.schedule import JOBS, Schedule, load_schedule
from gts_forecaster.pipelines.report import REPORT_FILE
from gts_forecaster.pipelines.run import RunResult, run_pipeline

logger = logging.getLogger(__name__)

JOB_STEPS = {"forecast": ["forecast", "report"], "backtest": None}  # None: every step (run-pipeline default)


def _flagged_series(output_dir: Path, frequency: str) -> int:
    path = Path(output_dir) / frequency / CHECKS_FILE
    if not path.is_file():
        return 0
    try:
        return int(pd.read_csv(path, dtype={"uuid": str})["uuid"].nunique())
    except Exception:  # noqa: BLE001 - an unreadable file means no count, not a failed job
        return 0


def run_job(
    frequency: str,
    job: str,
    *,
    schedule: Schedule | None = None,
    config_dir: Path | None = None,
    output_dir: Path = Path("outputs"),
    run: Callable[..., RunResult] = run_pipeline,
) -> int:
    if job not in JOBS:
        raise ValueError(f"job must be one of {list(JOBS)}, got {job!r}")
    schedule = schedule if schedule is not None else load_schedule(config_dir)
    cfg = load_config({"frequency": frequency, "horizon": schedule.horizons.get(frequency)}, config_dir)
    started = datetime.now()
    result = run(cfg, steps=JOB_STEPS[job], config_dir=config_dir, output_dir=Path(output_dir))
    ok = result.status == "ok"
    flagged = _flagged_series(output_dir, frequency) if ok else 0

    copied = None
    report = Path(output_dir) / frequency / REPORT_FILE
    if ok and schedule.report_copy_dir and report.is_file():
        try:
            schedule.report_copy_dir.mkdir(parents=True, exist_ok=True)
            copied = schedule.report_copy_dir / f"{frequency}_forecast_report.html"
            shutil.copy2(report, copied)
        except OSError as exc:
            logger.warning("the report could not be copied to %s: %s", schedule.report_copy_dir, exc)
            copied = None

    deleted_runs = clean_runs(cfg.runs_dir, frequency, schedule.keep_runs)
    deleted_backtests = clean_backtests(cfg.backtesting_dir, cfg.registry_dir, frequency, schedule.keep_backtests)

    summary = {
        "frequency": frequency, "job": job, "run_id": result.run_id, "status": result.status,
        "started": started.strftime("%Y-%m-%d %H:%M:%S"), "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "flagged_series": flagged, "report_copy": str(copied) if copied else None,
        "deleted_runs": len(deleted_runs), "deleted_backtests": len(deleted_backtests),
        "steps": [{k: s.get(k) for k in ("step", "status", "error")} for s in result.steps],
    }
    status_path = Path(cfg.runs_dir) / frequency / "last_job.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log = Path(cfg.runs_dir) / "alerts.log"
    if not ok and schedule.alerts.on_failure:
        errors = "\n".join(f"{s.get('step')}: {s.get('error')}" for s in result.steps if s.get("status") == "failed")
        send_alert(f"GTS Forecaster: {frequency} {job} FAILED (run {result.run_id})",
                   f"{errors}\nRun record: {result.run_dir}", schedule.alerts, log)
    elif flagged and schedule.alerts.on_flags:
        send_alert(f"GTS Forecaster: {frequency} forecast flagged {flagged} series (run {result.run_id})",
                   f"The forecasts were delivered; check the flagged series in "
                   f"{Path(output_dir) / frequency / CHECKS_FILE} or the report's Checks filter.",
                   schedule.alerts, log)
    logger.info("job %s %s finished: %s (run %s, %d flagged series)", frequency, job, result.status,
                result.run_id, flagged)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one automatic job (forecast + report, or backtest).")
    parser.add_argument("--frequency", required=True, choices=["yearly", "monthly", "weekly", "daily"])
    parser.add_argument("--job", required=True, choices=list(JOBS))
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return run_job(args.frequency, args.job, config_dir=args.config_dir, output_dir=args.output_dir)
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
