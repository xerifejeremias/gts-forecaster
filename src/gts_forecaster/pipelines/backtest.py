"""Backtest pipeline: score the candidate models per series and write the selection.

Run:  poetry run backtest --frequency monthly --horizon 12 --metric mae --top-n 3

Every model is scored once; from those scores two human-editable selections are written to
<registry_dir>/<frequency>/: selection_fast.json (best among the fast models, used when fast_only
is on, the default) and selection.json (best among all models that ran, incl. Prophet / Orbit when
the backtest ran with --no-fast-only), each with its backtest_scores[_fast].csv. Also
and <backtesting_dir>/<frequency>/backtest_forecasts_YYYYMMDD_HHMM.csv (forecast vs actual of
every model, fold and date). That file is new on every run; older ones are never overwritten.
Entries marked "locked": true in an existing selection file are kept as they are.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gts_forecaster.cleaning.outliers import OUTLIER_ACTIONS, OUTLIER_METHODS
from gts_forecaster.config import ON_MODEL_ERROR, ConfigError, RunConfig, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.selection_schema import validate_selection
from gts_forecaster.data_preparation import PreparedInput, limit_series, prepare_input_data
from gts_forecaster.evaluation.backtest import BacktestResult, run_backtest
from gts_forecaster.evaluation.metrics import METRICS
from gts_forecaster.models.catalog import build_models_from_config
from gts_forecaster.registry import (
    backtest_forecasts_table,
    build_selection_document,
    merge_locked,
    new_backtest_forecasts_path,
    read_existing_selection,
    registry_path,
    write_backtest_forecasts_atomic,
    write_json_atomic,
    write_scores_atomic,
    scores_file,
    selection_file,
)
from gts_forecaster.selection import SelectionError, select_models
from gts_forecaster.splitting import SplitError

logger = logging.getLogger(__name__)


@dataclass
class BacktestOutput:
    selection_path: Path
    scores_path: Path
    forecasts_path: Path
    document: dict
    result: BacktestResult
    kept_locked: list[str]
    timings_path: Path | None = None
    failures_path: Path | None = None
    partial: bool = False
    # both selections: {"fast": (path, document), "full": (path, document)}
    selections: dict | None = None
    # True when selection.json was left as it was (fast-only run after a full backtest)
    full_selection_kept: bool = False


def run_backtest_pipeline(
    prepared: PreparedInput, *, config_dir: Path | None = None, run_time: datetime | None = None
) -> BacktestOutput:
    """Backtest on prepared input, choose models, write the registry files."""
    cfg: RunConfig = prepared.config
    prepared = limit_series(prepared, cfg.max_series)
    models = build_models_from_config(cfg, config_dir, getattr(prepared, "static", None))
    unknown = [m for m in cfg.short_series_models if m not in models]
    if unknown:  # e.g. seasonal_naive is left out for yearly data
        logger.info("short-series models not available for %s: %s", cfg.frequency, unknown)

    result = run_backtest(
        prepared.data,
        models,
        horizon=cfg.horizon,
        pandas_freq=cfg.profile.pandas_freq,
        n_folds=cfg.n_folds,
        step=cfg.backtest_step,
        metric=cfg.metric,
        season_length=cfg.profile.season_length,
        known_columns=cfg.known_future_columns,
        outlier_method=cfg.outlier_method,
        outlier_threshold=cfg.outlier_threshold,
        outlier_max_share=cfg.outlier_max_share,
        outlier_action=cfg.outlier_action,
        on_error=cfg.on_model_error,
        forecast_floor=cfg.forecast_floor,
    )
    if len(result.failures):
        logger.warning(
            "%d model run(s) failed and were left out: %s",
            len(result.failures), sorted(result.failures["model"].unique()),
        )
    # One scoring run, two selections: "full" chooses among every model that was backtested,
    # "fast" among the fast ones only (Prophet / Orbit left out). A model's score does not depend
    # on which other models compete, so the fast selection needs no extra backtest.
    fast_models = {name: m for name, m in models.items() if getattr(m, "cost", "fast") != "slow"}
    includes_slow = len(fast_models) < len(models)
    if not includes_slow:
        logger.info(
            "no slow models were backtested (fast_only is on). Run the backtest with --no-fast-only "
            "to let Prophet / Orbit compete."
        )
    variants = {"fast": fast_models, "full": models}
    selections = {
        variant: select_models(
            result,
            candidates,
            top_n=cfg.top_n,
            ensemble_weights=cfg.ensemble_weights,
            short_series_models=cfg.short_series_models,
            short_series_n=cfg.short_series_n,
        )
        for variant, candidates in variants.items()
    }

    partial = cfg.is_partial_run
    # A trial run (model subset or sample of series) must not replace the real selection.
    folder = registry_path(cfg.registry_dir, f"{cfg.frequency}_partial" if partial else cfg.frequency)
    if partial:
        logger.warning(
            "partial run (models/exclude_models/max_series set): the selection is written to %s, "
            "not to the normal registry folder", folder,
        )
    backtest_dir = registry_path(cfg.backtesting_dir, cfg.frequency)
    meta = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frequency": cfg.frequency,
        "horizon": cfg.horizon,
        "metric": cfg.metric,
        "n_folds": cfg.n_folds,
        "top_n": cfg.top_n,
        "ensemble_weights": cfg.ensemble_weights,
        "data_cutoff": prepared.cutoff.strftime("%Y-%m-%d"),
        "includes_slow_models": includes_slow,
    }
    # The forecasts vs actuals of this backtest; the report's accuracy view reads this file.
    forecasts_path = new_backtest_forecasts_path(backtest_dir, run_time or datetime.now())
    meta["backtest_forecasts_file"] = forecasts_path.name
    # Written before the selections, which name this file: an interruption in between must not
    # leave a selection pointing to a file that does not exist.
    write_backtest_forecasts_atomic(forecasts_path, backtest_forecasts_table(result))
    if cfg.outlier_method != "none":  # same timestamp as the forecasts file
        outliers_path = forecasts_path.with_name(
            forecasts_path.name.replace("backtest_forecasts", "backtest_outliers")
        )
        result.outliers.rename(columns={"unique_id": "uuid"}).to_csv(outliers_path, index=False)
    stamp = forecasts_path.name.replace("backtest_forecasts", "")
    timings_path = forecasts_path.with_name(f"backtest_timings{stamp}")
    result.timings.to_csv(timings_path, index=False)
    failures_path = None
    if len(result.failures):
        failures_path = forecasts_path.with_name(f"backtest_failures{stamp}")
        result.failures.to_csv(failures_path, index=False)
    written: dict[str, tuple[Path, dict]] = {}
    all_kept: list[str] = []
    full_left_untouched = False
    for variant, candidates in variants.items():
        fast = variant == "fast"
        selection = selections[variant]
        path = folder / selection_file(fast)
        if not fast and not includes_slow:
            previous = _previous_full_selection(path)
            if previous is not None:
                # A fast-only backtest did not score Prophet / Orbit, so it must not replace a full
                # selection in which they competed (decided with the user 2026-10-06).
                logger.warning(
                    "%s was left untouched: it comes from a backtest that included the slow models "
                    "(created %s), and this run did not score them. Run the backtest with "
                    "--no-fast-only to refresh it.", path.name, previous["meta"].get("created"),
                )
                written[variant] = (path, previous)
                full_left_untouched = True
                continue
        document = build_selection_document(
            selection, {**meta, "variant": variant, "models": list(candidates)}
        )
        existing = read_existing_selection(path, candidates)
        document, kept = merge_locked(document, existing)
        if kept:
            logger.info("%s: kept %d locked selection(s): %s", path.name, len(kept), kept)
            all_kept += [k for k in kept if k not in all_kept]
        validate_selection(document, candidates, expected_ids=list(selection.series))
        write_json_atomic(path, document)
        write_scores_atomic(folder / scores_file(fast), selection.table)
        written[variant] = (path, document)
    main_variant = "fast" if cfg.fast_only else "full"
    if full_left_untouched:
        logger.info("selection_fast.json was updated; selection.json was kept")
    selection_path, document = written[main_variant]
    scores_path = folder / scores_file(cfg.fast_only)
    kept = all_kept
    return BacktestOutput(
        selection_path, scores_path, forecasts_path, document, result, kept,
        timings_path, failures_path, partial, written, full_left_untouched,
    )


def _previous_full_selection(path: Path) -> dict | None:
    """The existing selection.json if it was made by a backtest that scored the slow models."""
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # unreadable: let the normal path rewrite it
    meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
    return doc if meta.get("includes_slow_models") is True else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the models and write selection.json.")
    parser.add_argument("--frequency", help="yearly | monthly | weekly | daily")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--metric", choices=sorted(METRICS))
    parser.add_argument("--n-folds", type=int)
    parser.add_argument("--step", type=int, dest="backtest_step")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--ensemble-weights", choices=["equal", "inverse_error"])
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--outlier-method", choices=OUTLIER_METHODS)
    parser.add_argument("--outlier-threshold", type=float)
    parser.add_argument("--outlier-action", choices=OUTLIER_ACTIONS)
    parser.add_argument("--backtesting-dir", type=Path)
    parser.add_argument("--known-columns", nargs="+", dest="known_future_columns")
    parser.add_argument(
        "--fast-only", action=argparse.BooleanOptionalAction, default=None,
        help="leave out slow models (Prophet, Orbit); --no-fast-only includes them",
    )
    parser.add_argument("--models", nargs="+", help="run only these models (plus the baselines)")
    parser.add_argument("--exclude-models", nargs="+", help="leave these models out")
    parser.add_argument("--max-series", type=int, help="trial run on an even sample of N series")
    parser.add_argument(
        "--on-model-error", choices=ON_MODEL_ERROR,
        help="skip: a crashing model is left out and the rest finish (default); raise: stop",
    )
    parser.add_argument("--data-path")
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    overrides = {k: v for k, v in vars(args).items() if k != "config_dir"}
    try:
        config = load_config(overrides, config_dir=args.config_dir)
        out = run_backtest_pipeline(prepare_input_data(config), config_dir=args.config_dir)
    except (ConfigError, DataContractError, SplitError, SelectionError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    picks = pd.Series([len(e["models"]) for e in out.document["series"].values()])
    print(f"OK: {config.frequency}, horizon {config.horizon}, metric {config.metric}")
    print(f"  series selected:  {len(picks)} ({int((picks > 1).sum())} ensembles)")
    for variant, (path, doc) in (out.selections or {}).items():
        n_ens = sum(1 for e in doc["series"].values() if len(e["models"]) > 1)
        print(f"  selection {variant + ':':5}  {path}  ({len(doc['series'])} series, {n_ens} ensembles)")
    if out.selections:
        full_meta = out.selections["full"][1]["meta"]
        if out.full_selection_kept:
            print(f"  note:             selection.json kept from the earlier full backtest ({full_meta.get('created')}); "
                  "this fast-only run did not score Prophet / Orbit")
        elif not full_meta.get("includes_slow_models"):
            print("  note:             slow models (Prophet, Orbit) were not backtested; the full selection "
                  "equals the fast one (use --no-fast-only to include them)")
    print(f"  scores:           {out.scores_path}")
    print(f"  forecasts/actuals:{out.forecasts_path}")
    print(f"  timings:          {out.timings_path}")
    slowest = (
        out.result.timings.groupby("model")[["fit_seconds", "predict_seconds"]].sum().sum(axis=1)
        .sort_values(ascending=False).head(3)
    )
    if len(slowest):
        print("  slowest models:   " + ", ".join(f"{m} {t:.0f}s" for m, t in slowest.items()))
    if out.failures_path is not None:
        failed = sorted(out.result.failures["model"].unique())
        print(f"  FAILED models:    {', '.join(failed)} (details: {out.failures_path})")
    if out.partial:
        print("  PARTIAL RUN:      selection written to the *_partial registry folder, "
              "the real selection.json was not touched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
