"""Report pipeline: build a self-contained HTML report from saved results.

Run:  poetry run report --frequency monthly

Reads the input file (history), outputs/<frequency>/forecasts.csv and, if present, the selection
in registry/<frequency>/selection.json. It never runs a model. Writes, to outputs/<frequency>/:
  forecast_report.html        open it in any browser (works offline, no server needed)
  forecast_report_data.json   the same data as plain JSON, for a future web app / dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.io.loaders import load_input_csv
from gts_forecaster.registry import registry_path, selection_file
from gts_forecaster.report.accuracy import backtest_rows, encode, live_rows, newest_backtest_file
from gts_forecaster.report.data import build_report_data
from gts_forecaster.report.render import write_report

logger = logging.getLogger(__name__)

REPORT_FILE = "forecast_report.html"
REPORT_DATA_FILE = "forecast_report_data.json"


@dataclass
class ReportOutput:
    html_path: Path
    json_path: Path
    n_series: int


def run_report_pipeline(
    config, *, output_dir: Path, model_lines: bool = True, accuracy: bool = True
) -> ReportOutput:
    folder = Path(output_dir) / config.frequency
    forecasts_path = folder / "forecasts.csv"
    if not forecasts_path.is_file():
        raise DataContractError(
            f"No forecasts found at {forecasts_path}. Run `poetry run forecast` first."
        )
    forecasts = pd.read_csv(forecasts_path, dtype={"uuid": str}, parse_dates=["ds"])
    by_model = None
    by_model_path = folder / "forecasts_by_model.csv"
    if model_lines:
        if by_model_path.is_file():
            by_model = pd.read_csv(by_model_path, dtype={"uuid": str}, parse_dates=["ds"])
        else:
            logger.warning("no %s: the report will have no per-model lines", by_model_path)
    history = load_input_csv(config.data_path, config.static_columns)

    selection = None
    selection_path = registry_path(config.registry_dir, config.frequency) / selection_file(config.fast_only)
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        logger.warning("no selection.json at %s: the report will not name the models", selection_path)
    # Series that have a forecast but no entry in the selection got the fallback forecast.
    fallback_ids = (
        sorted(set(forecasts["uuid"]) - set(selection["series"])) if selection is not None else []
    )
    data = build_report_data(
        history,
        forecasts,
        pandas_freq=config.profile.pandas_freq,
        frequency=config.frequency,
        static_columns=config.static_columns,
        selection=selection,
        fallback_ids=fallback_ids,
        by_model=by_model,
        meta={
            "metric": (selection or {}).get("meta", {}).get("metric"),
            "forecast_mode": config.forecast_mode,
        },
    )
    if accuracy:
        data["accuracy"] = _accuracy_data(config, selection, history, [s["id"] for s in data["series"]])
    html_path, json_path = folder / REPORT_FILE, folder / REPORT_DATA_FILE
    write_report(data, html_path, json_path)
    return ReportOutput(html_path, json_path, len(data["series"]))


def _accuracy_data(config, selection, history, series_ids):
    """Backtest and live forecasts next to their actuals, for the Accuracy tab (None if none)."""
    sources = {}
    backtest_file = newest_backtest_file(
        registry_path(config.backtesting_dir, config.frequency), selection
    )
    if backtest_file is not None:
        table = pd.read_csv(backtest_file, dtype={"uuid": str})
        try:
            sources["backtest"] = backtest_rows(table, selection)
        except ValueError as exc:
            logger.warning("accuracy view: cannot use %s (%s)", backtest_file.name, exc)
    else:
        logger.warning("accuracy view: no backtest forecasts found; run the backtest first")
    sources["live"] = live_rows(Path(config.runs_dir) / config.frequency, history)
    keep = {m["name"] for e in (selection or {}).get("series", {}).values() for m in e["models"]} | {"naive"}
    encoded = encode(sources, series_ids, keep_models=sorted(keep))
    if encoded is not None:
        encoded["backtest_file"] = backtest_file.name if backtest_file is not None else None
    return encoded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the HTML forecast report.")
    parser.add_argument("--frequency", help="yearly | monthly | weekly | daily")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--no-accuracy", action="store_true",
        help="leave the Accuracy tab out (backtest and live forecasts vs actuals)",
    )
    parser.add_argument(
        "--no-model-lines", action="store_true",
        help="leave the per-model forecasts out of the report (smaller file)",
    )
    parser.add_argument(
        "--fast-only", action=argparse.BooleanOptionalAction, default=None,
        help="which selection names the models: selection_fast.json (default) or, with --no-fast-only, selection.json",
    )
    parser.add_argument("--data-path")
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    overrides = {k: v for k, v in vars(args).items() if k not in ("config_dir", "output_dir", "no_model_lines", "no_accuracy")}
    try:
        config = load_config(overrides, config_dir=args.config_dir)
        out = run_report_pipeline(
            config, output_dir=args.output_dir, model_lines=not args.no_model_lines,
            accuracy=not args.no_accuracy,
        )
    except (ConfigError, DataContractError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {config.frequency}, {out.n_series} series")
    print(f"  report:           {out.html_path}")
    print(f"  data (JSON):      {out.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
