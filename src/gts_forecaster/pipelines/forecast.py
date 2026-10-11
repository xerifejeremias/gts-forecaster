"""Forecast pipeline: use the saved selection (selection_fast.json by default, selection.json
with --no-fast-only) and either refit its models (frozen_selection) or load them (frozen_models).

Run:  poetry run forecast --frequency monthly --horizon 12
      poetry run forecast --frequency monthly --mode frozen_models   (models saved by `train`)

Every model is refitted on all of the data, and each series is forecast with the models chosen in
registry/<frequency>/selection.json. Writes, to outputs/<frequency>/:
  forecasts_by_model.csv   unique_id, ds, one column per model
  forecasts.csv            ds, uuid, forecast
  outliers.csv             training points that were replaced as outliers
  forecast_checks.csv      series whose forecast failed a sanity check (forecast_checks.py)
The horizon may differ from the one used in the backtest.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from gts_forecaster.cleaning.outliers import OUTLIER_ACTIONS, OUTLIER_METHODS, handle_outliers
from gts_forecaster.columns import ID
from gts_forecaster.config import FORECAST_MODES, ON_MODEL_ERROR, ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.selection_schema import validate_selection
from gts_forecaster.data_preparation import PreparedInput, limit_series, prepare_input_data
from gts_forecaster.features.known_future import attach_known_future, load_known_future
from gts_forecaster.forecast_checks import CHECKS_FILE, check_forecasts, report_flags
from gts_forecaster.forecasting import ForecastError, combine_forecasts, forecast_by_model
from gts_forecaster.models.catalog import build_models_from_config
from gts_forecaster.models.store import (
    fitted_models_path, load_fitted_model, load_manifest, params_signature,
)
from gts_forecaster.output import build_output, write_output
from gts_forecaster.registry import read_selection, registry_path, selection_file
from gts_forecaster.selection import add_missing_series
from gts_forecaster.splitting import SplitError, forecast_split

logger = logging.getLogger(__name__)



@dataclass
class ForecastOutput:
    by_model: pd.DataFrame
    forecast: pd.DataFrame
    outliers: pd.DataFrame
    by_model_path: Path
    forecast_path: Path
    failures: pd.DataFrame | None = None
    timings: pd.DataFrame | None = None
    output: pd.DataFrame | None = None  # the validated file for the receiving platform
    output_path: Path | None = None
    checks: pd.DataFrame | None = None  # sanity-check flags, one row per flag (None: checks off)


def training_split(prepared: PreparedInput):
    """All data up to the cutoff, with outliers handled as configured: what the models are fitted on."""
    cfg = prepared.config
    split = forecast_split(prepared.data, cfg.horizon, cfg.profile.pandas_freq)
    cleaned = handle_outliers(
        split.train, method=cfg.outlier_method, threshold=cfg.outlier_threshold,
        max_share=cfg.outlier_max_share, season_length=cfg.profile.season_length,
        action=cfg.outlier_action,
    )
    return replace(split, train=cleaned.data), cleaned


def selected_model_names(selection: dict) -> list[str]:
    """The models named anywhere in selection.json, in first-use order."""
    names: dict[str, None] = {}
    for entry in selection["series"].values():
        for member in entry["models"]:
            names[member["name"]] = None
    return list(names)


def load_frozen_models(prepared: PreparedInput, models: dict) -> dict:
    """The saved models that are still valid for this run, by name (frozen-models mode)."""
    cfg = prepared.config
    folder = fitted_models_path(cfg.fitted_models_dir, cfg.frequency, cfg.fast_only)
    manifest = load_manifest(folder)
    if manifest.get("frequency") not in (None, cfg.frequency):
        raise ConfigError(
            f"the saved models are for frequency {manifest['frequency']!r}, not {cfg.frequency!r}"
        )
    trained_until = manifest.get("data_cutoff")
    if trained_until and prepared.cutoff < pd.Timestamp(trained_until):
        raise DataContractError(
            f"the data ends {prepared.cutoff:%Y-%m-%d}, before the saved models' training data "
            f"({trained_until}). Use data that is at least as new, or run `train` again."
        )
    fitted = {}
    for name, template in models.items():
        entry = manifest["models"].get(name)
        if entry is None:
            logger.warning("model '%s' was not saved by `train`; it will be refitted", name)
            continue
        if entry["signature"] != params_signature(template):
            logger.warning(
                "model '%s' settings changed since `train` (models.yaml); it will be refitted. "
                "Run `train` again to freeze the new settings.", name,
            )
            continue
        model = load_fitted_model(folder, manifest, name)
        if model is not None:
            fitted[name] = model
    return fitted


def run_forecast_pipeline(
    prepared: PreparedInput,
    *,
    output_dir: Path,
    config_dir: Path | None = None,
    known_future: pd.DataFrame | Path | str | None = None,
) -> ForecastOutput:
    """`known_future`: a table or CSV path with the future values of `known_future_columns`
    (default: `future_features_path` from the config). Not needed when no column is known."""
    cfg = prepared.config
    prepared = limit_series(prepared, cfg.max_series)
    # The saved selection may name any model, so the models/exclude_models trial filters are not
    # applied here; use selection.json (or fast_only) to control what is forecast.
    models = build_models_from_config(
        cfg, config_dir, getattr(prepared, "static", None), apply_model_filter=False
    )
    # fast_only on (default) -> selection_fast.json, off -> selection.json: same model set as this run
    selection_path = registry_path(cfg.registry_dir, cfg.frequency) / selection_file(cfg.fast_only)
    selection = read_selection(selection_path, models)
    if selection["meta"].get("horizon") not in (None, cfg.horizon):
        logger.warning(
            "forecast horizon %s differs from the backtest horizon %s",
            cfg.horizon,
            selection["meta"].get("horizon"),
        )

    split, cleaned = training_split(prepared)
    # series without an entry (new since the backtest) get the short-series ensemble, not a stop
    selection, _ = add_missing_series(
        selection, split.train_lengths, models, cfg.short_series_models, cfg.short_series_n
    )
    validate_selection(selection, models, expected_ids=prepared.data[ID].unique().tolist())
    fitted = None
    if cfg.forecast_mode == "frozen_models" or cfg.forecast_models == "selected":
        used = selected_model_names(selection)
        models = {name: models[name] for name in used}  # only what the selection needs
    if cfg.forecast_mode == "frozen_models":
        fitted = load_frozen_models(prepared, models)
    if cfg.known_future_columns:
        source = known_future if known_future is not None else cfg.future_features_path
        if source is None:
            raise DataContractError(
                f"known_future_columns {list(cfg.known_future_columns)} are set, but no future "
                "values were given (future_features_path or --future-features)"
            )
        table = load_known_future(source, cfg.known_future_columns)
        split = replace(split, future=attach_known_future(split.future, table, cfg.known_future_columns))
    failure_rows: list[dict] = []
    timing_rows: list[dict] = []
    wide = forecast_by_model(
        split, models, on_error=cfg.on_model_error, failures=failure_rows,
        timings=timing_rows, fitted=fitted, forecast_floor=cfg.forecast_floor,
    )
    final = combine_forecasts(
        wide, selection, prepared.fallback, split.horizon_dates,
        drop_missing=cfg.on_model_error == "skip", forecast_floor=cfg.forecast_floor,
    )
    # Sanity checks before anything is written, so `raise` leaves no output behind.
    checks = None
    if cfg.forecast_check_action != "off":
        checks = check_forecasts(
            prepared.data, final, season_length=cfg.profile.season_length,
            threshold=cfg.forecast_check_threshold, max_ratio=cfg.forecast_check_max_ratio,
            skip_ids=set(prepared.fallback[ID]) if len(prepared.fallback) else (),
        )
        report_flags(checks, cfg.forecast_check_action)
    failures = pd.DataFrame(failure_rows, columns=["model", "error"])
    timings = pd.DataFrame(
        timing_rows, columns=["model", "n_series", "fit_seconds", "predict_seconds", "status"]
    )
    folder = Path(output_dir) / cfg.frequency
    folder.mkdir(parents=True, exist_ok=True)
    by_model_path, forecast_path = folder / "forecasts_by_model.csv", folder / "forecasts.csv"
    # The identifier columns (market, product, company ...) are repeated on every row, so the files
    # can be grouped and totalled by them without going back to the input data.
    static = getattr(prepared, "static", None)
    static = static if static is not None and len(static) else None
    for table, path in ((wide, by_model_path), (final, forecast_path)):
        out = table if static is None else table.merge(static, on=ID, how="left")
        out.rename(columns={ID: "uuid"}).to_csv(path, index=False)
    if cfg.outlier_method != "none":
        cleaned.report.rename(columns={ID: "uuid"}).to_csv(folder / "outliers.csv", index=False)
    # Final file for the receiving platform: long format, checked against the output contract.
    expected_ids = set(prepared.data[ID]) | set(prepared.fallback[ID]) if len(prepared.fallback) else set(prepared.data[ID])
    output = build_output(
        final, static if static is not None else getattr(prepared, "static", None),
        static_columns=cfg.static_columns, expected_ids=expected_ids,
        horizon_dates=split.horizon_dates, rename=dict(cfg.output_rename),
        forecast_floor=cfg.forecast_floor,
    )
    output_path = folder / cfg.output_file
    write_output(output, output_path)
    timings.to_csv(folder / "model_timings.csv", index=False)
    checks_file = folder / CHECKS_FILE
    if checks is not None:  # written even when empty, so it is clear the checks ran
        out = checks if static is None else checks.merge(static, on=ID, how="left")
        out.rename(columns={ID: "uuid"}).to_csv(checks_file, index=False)
    elif checks_file.exists():
        checks_file.unlink()
    failures_file = folder / "model_failures.csv"
    if len(failures):
        failures.to_csv(failures_file, index=False)
    elif failures_file.exists():  # do not leave a stale file from an earlier run
        failures_file.unlink()
    return ForecastOutput(
        wide, final, cleaned.report, by_model_path, forecast_path, failures, timings, output, output_path,
        checks,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecast with the saved selection.")
    parser.add_argument("--frequency", help="yearly | monthly | weekly | daily")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--outlier-method", choices=OUTLIER_METHODS)
    parser.add_argument("--outlier-threshold", type=float)
    parser.add_argument("--outlier-action", choices=OUTLIER_ACTIONS)
    parser.add_argument("--known-columns", nargs="+", dest="known_future_columns")
    parser.add_argument("--future-features", dest="future_features_path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--fast-only", action=argparse.BooleanOptionalAction, default=None,
        help="leave out slow models (Prophet, Orbit); --no-fast-only includes them",
    )
    parser.add_argument(
        "--mode", dest="forecast_mode", choices=FORECAST_MODES,
        help="frozen_selection: refit the selected models; frozen_models: use the models saved by train",
    )
    parser.add_argument("--fitted-models-dir", type=Path, help="where `train` saved the models")
    parser.add_argument("--max-series", type=int, help="trial run on an even sample of N series")
    parser.add_argument(
        "--on-model-error", choices=ON_MODEL_ERROR,
        help="skip: a crashing model is left out and its weight is shared by the others (default)",
    )
    parser.add_argument("--data-path")
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    overrides = {
        k: v for k, v in vars(args).items() if k not in ("config_dir", "output_dir")
    }
    try:
        config = load_config(overrides, config_dir=args.config_dir)
        out = run_forecast_pipeline(
            prepare_input_data(config), output_dir=args.output_dir, config_dir=args.config_dir
        )
    except (ConfigError, DataContractError, SplitError, ForecastError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {config.frequency}, horizon {config.horizon}, mode {config.forecast_mode}")
    print(f"  series forecast:  {out.forecast[ID].nunique()}")
    print(f"  per model:        {out.by_model_path}")
    print(f"  final forecast:   {out.forecast_path}")
    print(f"  OUTPUT (platform):{out.output_path}  ({len(out.output)} rows, long format)")
    if out.failures is not None and len(out.failures):
        print(f"  FAILED models:    {', '.join(out.failures['model'])} (see model_failures.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
