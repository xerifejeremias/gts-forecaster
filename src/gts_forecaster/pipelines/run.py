"""One command for a whole run, with a record of every run.

Run:  poetry run run-pipeline --frequency monthly
      poetry run run-pipeline --frequency monthly --steps forecast report        (quick daily run)
      poetry run run-pipeline --frequency monthly --no-fast-only --steps backtest (score Prophet/Orbit)

Steps, in this order (default: all of them; `train` only when forecast_mode is frozen_models):
  backtest  score the models, write selection_fast.json / selection.json
  train     fit the selected models and save them (frozen_models mode)
  forecast  forecast, write the platform file forecast_output.csv
  report    build the HTML report

The input file is read and checked once and shared by the steps. If a step fails the run stops,
the steps after it are marked "skipped", and the exit code is 1.

Every run gets its own folder <runs_dir>/<frequency>/<run id>/ (run id = start time, e.g.
20261006_101500) with:
  run.log         everything the steps logged
  manifest.json   run id, status, timings and errors per step, the resolved settings, library
                  versions, and a fingerprint (size, sha256, rows) of the input file
  config/         a copy of the config files used
  outputs/<step>/ a copy of what each step produced (selections, scores, forecasts, report ...)
<runs_dir>/<frequency>/latest.txt holds the id of the newest run. The normal output folders
(registry/, outputs/, ...) are still written as before; the run folder is the record of the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gts_forecaster.cleaning.outliers import OUTLIER_ACTIONS, OUTLIER_METHODS
from gts_forecaster.columns import ID
from gts_forecaster.config import FORECAST_MODES, ON_MODEL_ERROR, RunConfig, load_config, project_root
from gts_forecaster.data_preparation import prepare_input_data
from gts_forecaster.evaluation.metrics import METRICS
from gts_forecaster.registry import scores_file

logger = logging.getLogger(__name__)

STEPS = ("backtest", "train", "forecast", "report")
RUN_ID_FORMAT = "%Y%m%d_%H%M%S"  # no colon: not allowed in Windows file names
MANIFEST = "manifest.json"
LOG_FILE = "run.log"
LIBRARIES = (
    "gts-forecaster", "pandas", "numpy", "pandera", "pyyaml", "statsforecast", "mlforecast",
    "lightgbm", "xgboost", "scikit-learn", "chronos-forecasting", "torch", "prophet", "orbit-ml",
    "holidays",
)


@dataclasses.dataclass
class RunResult:
    run_id: str
    run_dir: Path
    status: str  # "ok" or "failed"
    steps: list[dict]
    manifest_path: Path


def default_steps(cfg: RunConfig) -> list[str]:
    """All steps; `train` only matters when forecasts use the saved models."""
    return [s for s in STEPS if s != "train" or cfg.forecast_mode == "frozen_models"]


def run_pipeline(
    cfg: RunConfig,
    *,
    steps: Sequence[str] | None = None,
    config_dir: Path | None = None,
    output_dir: Path = Path("outputs"),
    run_time: datetime | None = None,
    prepare: Callable[[RunConfig], Any] = prepare_input_data,
) -> RunResult:
    """Run the steps in order and keep a record of the run. Never raises for a failing step:
    the failure is logged and recorded, and the result's status is "failed"."""
    steps = list(default_steps(cfg) if not steps else steps)
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    steps = [s for s in STEPS if s in steps]  # always in pipeline order

    started = run_time or datetime.now()
    run_id = started.strftime(RUN_ID_FORMAT)
    base = Path(cfg.runs_dir) / cfg.frequency
    run_dir = base / run_id
    suffix = 1
    while run_dir.exists():  # two runs in the same second
        run_dir = base / f"{run_id}_{suffix}"
        suffix += 1
    run_id = run_dir.name
    run_dir.mkdir(parents=True)

    handler = logging.FileHandler(run_dir / LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "frequency": cfg.frequency,
        "started": _utc_now(),
        "finished": None,
        "status": "running",
        "steps_requested": steps,
        "fast_only": cfg.fast_only,
        "forecast_mode": cfg.forecast_mode,
        "partial_run": cfg.is_partial_run,
        "input": _fingerprint(Path(cfg.data_path)),
        "config": _jsonable(dataclasses.asdict(cfg)),
        "environment": _environment(),
        "steps": [],
    }
    _copy_config(config_dir, run_dir / "config")
    _write_manifest(run_dir, manifest)
    logger.info("run %s: %s, steps %s", run_id, cfg.frequency, steps)

    records: list[dict] = []
    prepared = None
    failed = False
    try:
        for step in steps:
            record = {"step": step, "status": "skipped", "seconds": 0.0, "error": None, "outputs": []}
            records.append(record)
            if failed:
                continue
            t0 = time.perf_counter()
            try:
                if prepared is None and step != "report":
                    prepared = prepare(cfg)
                    manifest["input"]["series_for_models"] = int(prepared.data[ID].nunique())
                    manifest["input"]["data_cutoff"] = str(getattr(prepared, "cutoff", ""))[:10]
                files = _run_step(step, cfg, prepared, config_dir, Path(output_dir))
                record["outputs"] = _copy_outputs(files, run_dir / "outputs" / step)
                record["status"] = "ok"
            except Exception as exc:  # record it and stop; the next steps depend on this one
                failed = True
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                logger.error("step %s FAILED: %s\n%s", step, record["error"], traceback.format_exc())
            record["seconds"] = round(time.perf_counter() - t0, 2)
            logger.info("step %s: %s in %.1fs", step, record["status"], record["seconds"])
            manifest["steps"] = records
            _write_manifest(run_dir, manifest)
    finally:
        manifest["steps"] = records
        manifest["finished"] = _utc_now()
        manifest["status"] = "failed" if failed else "ok"
        _write_manifest(run_dir, manifest)
        (base / "latest.txt").write_text(run_id + "\n", encoding="utf-8")
        logger.info("run %s finished: %s", run_id, manifest["status"])
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous_level)
    return RunResult(run_id, run_dir, manifest["status"], records, run_dir / MANIFEST)


def _run_step(step: str, cfg: RunConfig, prepared, config_dir: Path | None, output_dir: Path) -> list[Path]:
    """Run one step; return the files it produced (to copy into the run folder)."""
    if step == "backtest":
        from gts_forecaster.pipelines.backtest import run_backtest_pipeline

        out = run_backtest_pipeline(prepared, config_dir=config_dir)
        files = [path for path, _ in (out.selections or {}).values()]
        files += [out.selection_path.parent / scores_file(True), out.selection_path.parent / scores_file(False)]
        files += [out.forecasts_path, out.timings_path, out.failures_path]
        outliers = out.forecasts_path.with_name(out.forecasts_path.name.replace("backtest_forecasts", "backtest_outliers"))
        return files + [outliers]
    if step == "train":
        from gts_forecaster.pipelines.train import run_train_pipeline

        out = run_train_pipeline(prepared, config_dir=config_dir)
        if out.failed:
            logger.warning("train: model(s) failed and will be refitted at forecast time: %s", list(out.failed))
        return [out.folder / MANIFEST, out.folder / "train_timings.csv"]
    if step == "forecast":
        from gts_forecaster.pipelines.forecast import CHECKS_FILE, run_forecast_pipeline

        out = run_forecast_pipeline(prepared, output_dir=output_dir, config_dir=config_dir)
        folder = out.forecast_path.parent
        return [
            out.output_path, out.forecast_path, out.by_model_path, folder / "model_timings.csv",
            folder / "model_failures.csv", folder / "outliers.csv", folder / CHECKS_FILE,
        ]
    if step == "report":
        from gts_forecaster.pipelines.report import run_report_pipeline

        out = run_report_pipeline(cfg, output_dir=output_dir)
        return [out.html_path, out.json_path]
    raise ValueError(f"unknown step {step!r}")


def _copy_outputs(files: Sequence[Path | None], target: Path) -> list[str]:
    copied = []
    for path in files:
        if path is None or not Path(path).is_file():
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / Path(path).name)
        copied.append(str(Path(path)))
    return copied


def _copy_config(config_dir: Path | None, target: Path) -> None:
    source = Path(config_dir) if config_dir is not None else project_root() / "config"
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)


def _fingerprint(path: Path) -> dict[str, Any]:
    """Size, sha256 and row count of the input file: shows later exactly which data a run used."""
    info: dict[str, Any] = {"path": str(path)}
    if not path.is_file():
        info["error"] = "file not found"
        return info
    digest, rows = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            rows += block.count(b"\n")
    stat = path.stat()
    info.update(
        size_bytes=stat.st_size,
        modified=datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sha256=digest.hexdigest(),
        lines=rows,
    )
    return info


def _environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.node(),
        "libraries": versions,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    tmp = run_dir / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, run_dir / MANIFEST)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline steps and keep a record of the run.")
    parser.add_argument("--frequency", help="yearly | monthly | weekly | daily")
    parser.add_argument("--steps", nargs="+", choices=STEPS,
                        help="steps to run (default: all; train only with forecast_mode frozen_models)")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--metric", choices=sorted(METRICS))
    parser.add_argument("--n-folds", type=int)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--mode", dest="forecast_mode", choices=FORECAST_MODES)
    parser.add_argument(
        "--fast-only", action=argparse.BooleanOptionalAction, default=None,
        help="leave out slow models (Prophet, Orbit); --no-fast-only includes them",
    )
    parser.add_argument("--outlier-method", choices=OUTLIER_METHODS)
    parser.add_argument("--outlier-action", choices=OUTLIER_ACTIONS)
    parser.add_argument("--on-model-error", choices=ON_MODEL_ERROR)
    parser.add_argument("--max-series", type=int, help="trial run on an even sample of N series")
    parser.add_argument("--data-path")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    overrides = {k: v for k, v in vars(args).items() if k not in ("config_dir", "output_dir", "steps")}
    try:
        config = load_config(overrides, config_dir=args.config_dir)
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    result = run_pipeline(config, steps=args.steps, config_dir=args.config_dir, output_dir=args.output_dir)

    print(f"{'OK' if result.status == 'ok' else 'FAILED'}: run {result.run_id} ({config.frequency})")
    for record in result.steps:
        line = f"  {record['step']:9} {record['status']:8} {record['seconds']:8.1f}s"
        if record["error"]:
            line += f"  {record['error']}"
        print(line)
    print(f"  record:    {result.run_dir}  (run.log, manifest.json)")
    if config.forecast_mode == "frozen_models" and "train" not in [r["step"] for r in result.steps]:
        print("  note:      frozen_models uses the models saved by an earlier train step")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
