"""Train pipeline: fit the models named in the selection on all the data and save them.

The selection is selection_fast.json when fast_only is on (default) and selection.json when it is
off; the models go to <fitted_models_dir>/<frequency>/fast/ or .../full/ to match.

Run:  poetry run train --frequency monthly

Needs a selection.json from `backtest`. Only the models the selection uses are fitted (each on
the series that have enough history for it). They are saved to
<fitted_models_dir>/<frequency>/ together with a manifest. Then
    poetry run forecast --frequency monthly --mode frozen_models
applies them to the newest data without refitting. Run `train` again whenever the selection,
models.yaml / features.yaml or the series set changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gts_forecaster.cleaning.outliers import OUTLIER_ACTIONS, OUTLIER_METHODS
from gts_forecaster.columns import ID
from gts_forecaster.config import ON_MODEL_ERROR, ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.data_preparation import PreparedInput, prepare_input_data
from gts_forecaster.models.catalog import build_models_from_config
from gts_forecaster.models.store import fitted_models_path, save_fitted_models
from gts_forecaster.pipelines.forecast import selected_model_names, training_split
from gts_forecaster.registry import read_selection, registry_path, selection_file
from gts_forecaster.splitting import SplitError

logger = logging.getLogger(__name__)


@dataclass
class TrainOutput:
    folder: Path
    trained: list[str]
    failed: dict[str, str]  # model -> error
    not_saved: list[str]  # fitted but could not be pickled
    timings: pd.DataFrame


def run_train_pipeline(prepared: PreparedInput, *, config_dir: Path | None = None) -> TrainOutput:
    cfg = prepared.config
    if cfg.is_partial_run:
        raise ConfigError(
            "train needs the full series set: do not use models / exclude_models / max_series"
        )
    models = build_models_from_config(
        cfg, config_dir, getattr(prepared, "static", None), apply_model_filter=False
    )
    selection = read_selection(
        registry_path(cfg.registry_dir, cfg.frequency) / selection_file(cfg.fast_only),
        models,
        expected_ids=prepared.data[ID].unique().tolist(),
    )
    names = selected_model_names(selection)
    split, _ = training_split(prepared)

    fitted, failed, rows = {}, {}, []
    for number, name in enumerate(names, start=1):
        template = models[name]
        label = f"model {name} ({number}/{len(names)})"
        eligible = split.train_lengths.index[split.train_lengths >= template.min_train_length]
        if len(eligible) == 0:
            logger.info("%s: no series long enough, not trained", label)
            rows.append({"model": name, "n_series": 0, "fit_seconds": 0.0, "status": "no series"})
            continue
        started = time.perf_counter()
        try:
            fitted[name] = template.clone().fit(split.train[split.train[ID].isin(eligible)])
        except Exception as exc:
            if cfg.on_model_error == "raise":
                raise
            seconds = time.perf_counter() - started
            failed[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("%s FAILED after %.1fs: %s", label, seconds, failed[name])
            rows.append({"model": name, "n_series": len(eligible), "fit_seconds": round(seconds, 3), "status": "failed"})
            continue
        seconds = time.perf_counter() - started
        logger.info("%s: %d series, %.1fs", label, len(eligible), seconds)
        rows.append({"model": name, "n_series": len(eligible), "fit_seconds": round(seconds, 3), "status": "ok"})

    folder = fitted_models_path(cfg.fitted_models_dir, cfg.frequency, cfg.fast_only)
    meta = {
        "frequency": cfg.frequency,
        "fast_only": cfg.fast_only,
        "data_cutoff": prepared.cutoff.strftime("%Y-%m-%d"),
        "season_length": cfg.profile.season_length,
        "n_series": int(prepared.data[ID].nunique()),
        "selection_created": selection.get("meta", {}).get("created"),
        "failed": failed,
    }
    not_saved = save_fitted_models(folder, fitted, meta)
    timings = pd.DataFrame(rows, columns=["model", "n_series", "fit_seconds", "status"])
    timings.to_csv(folder / "train_timings.csv", index=False)
    return TrainOutput(folder, list(fitted), failed, not_saved, timings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the selected models and save them.")
    parser.add_argument("--frequency", help="yearly | monthly | weekly | daily")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--fitted-models-dir", type=Path)
    parser.add_argument("--outlier-method", choices=OUTLIER_METHODS)
    parser.add_argument("--outlier-threshold", type=float)
    parser.add_argument("--outlier-action", choices=OUTLIER_ACTIONS)
    parser.add_argument("--known-columns", nargs="+", dest="known_future_columns")
    parser.add_argument(
        "--fast-only", action=argparse.BooleanOptionalAction, default=None,
        help="leave out slow models (Prophet, Orbit); --no-fast-only includes them",
    )
    parser.add_argument("--on-model-error", choices=ON_MODEL_ERROR)
    parser.add_argument("--data-path")
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    overrides = {k: v for k, v in vars(args).items() if k != "config_dir"}
    try:
        config = load_config(overrides, config_dir=args.config_dir)
        out = run_train_pipeline(prepare_input_data(config), config_dir=args.config_dir)
    except (ConfigError, DataContractError, SplitError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {config.frequency}, trained {len(out.trained)} model(s)")
    print(f"  saved to:         {out.folder}")
    if out.failed:
        print(f"  FAILED models:    {', '.join(out.failed)} (they are refitted at forecast time)")
    if out.not_saved:
        print(f"  not savable:      {', '.join(out.not_saved)} (refitted at forecast time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
