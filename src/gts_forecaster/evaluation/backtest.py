"""Rolling-origin backtest: fit every candidate model on each fold and score it per series.

For each fold (a cutoff from `splitting.backtest_splits`) and each model:
  1. keep only the series with enough training data for that model,
  2. fit a fresh copy of the model on their training rows,
  3. predict the horizon window and compare with the actual values.

Periods that were filled in during missing-data handling (`is_filled`) are not scored.
The metric is chosen by name (see evaluation/metrics.py).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gts_forecaster.cleaning.outliers import REPORT_COLUMNS as OUTLIER_COLUMNS
from gts_forecaster.cleaning.outliers import handle_outliers
from gts_forecaster.columns import DATE, ID, IS_FILLED, TARGET
from gts_forecaster.evaluation.metrics import get_metric
from gts_forecaster.features.known_future import known_columns_from_test
from gts_forecaster.models.base import PREDICTION, BaseForecaster
from gts_forecaster.splitting import backtest_splits

logger = logging.getLogger(__name__)

FOLD, MODEL = "fold", "model"
SCORE, N_POINTS = "score", "n_points"
SCORE_COLUMNS = [ID, FOLD, MODEL, SCORE, N_POINTS]
TIMING_COLUMNS = [FOLD, MODEL, "n_series", "fit_seconds", "predict_seconds", "status"]
FAILURE_COLUMNS = [FOLD, MODEL, "error"]
ON_ERROR = ("skip", "raise")
PREDICTION_COLUMNS = [ID, FOLD, MODEL, DATE, "horizon_step", TARGET, PREDICTION, IS_FILLED]


@dataclass
class BacktestResult:
    """What a backtest produced."""

    metric: str
    season_length: int
    cutoffs: dict[int, pd.Timestamp]  # fold number -> cutoff
    scores: pd.DataFrame  # unique_id, fold, model, score, n_points (NaN score = not defined)
    predictions: pd.DataFrame  # one row per (series, fold, model, date)
    train_y: dict[tuple[int, str], np.ndarray]  # (fold, unique_id) -> training values, for MASE
    series_lengths: pd.Series  # observations per series in the full data
    outliers: pd.DataFrame  # points replaced in the training data: fold, unique_id, ds, y_raw, ...
    # One row per (fold, model): series fitted, seconds spent, status ("ok", "no series", "failed").
    timings: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TIMING_COLUMNS))
    failures: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FAILURE_COLUMNS))


def run_backtest(
    df: pd.DataFrame,
    models: Mapping[str, BaseForecaster],
    *,
    horizon: int,
    pandas_freq: str,
    n_folds: int,
    step: int | None = None,
    metric: str = "mae",
    season_length: int = 1,
    known_columns: Sequence[str] = (),
    outlier_method: str = "none",
    outlier_threshold: float = 5.0,
    outlier_max_share: float = 0.1,
    outlier_action: str = "clip",
    on_error: str = "skip",
    forecast_floor: float | None = None,
) -> BacktestResult:
    """Backtest all `models` on `df` (model format, complete). Returns scores and predictions.

    `forecast_floor`: predictions below it are raised to it (0 for volumes), so the scores
    judge the forecasts that would really be delivered.
    `on_error="skip"` (default): a model that crashes is logged, recorded in `result.failures`
    and left out of that fold, so the other models still finish. `"raise"` stops at the first crash.
    """
    if not models:
        raise ValueError("no models to backtest")
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}, got {on_error!r}")
    metric_fn = get_metric(metric)
    splits = backtest_splits(df, horizon, pandas_freq, n_folds, step)
    if IS_FILLED not in df.columns:
        df = df.assign(**{IS_FILLED: False})

    score_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    train_y: dict[tuple[int, str], np.ndarray] = {}

    outlier_frames: list[pd.DataFrame] = []
    timing_rows: list[dict] = []
    failure_rows: list[dict] = []

    for split in splits:
        # Outliers are removed from the training data of each fold only, using that fold's own
        # history, so nothing from the test window leaks in. Scoring uses the raw actuals.
        cleaned = handle_outliers(
            split.train, method=outlier_method, threshold=outlier_threshold,
            max_share=outlier_max_share, season_length=season_length, action=outlier_action,
        )
        train = cleaned.data
        if len(cleaned.report):
            outlier_frames.append(cleaned.report.assign(fold=split.fold))
        lengths = split.train_lengths
        ordered_train = train.sort_values([ID, DATE], kind="stable")
        for uid, g in ordered_train.groupby(ID, sort=False):
            train_y[(split.fold, uid)] = g[TARGET].to_numpy(dtype=float)

        actual = split.test[[ID, DATE, "horizon_step", TARGET, IS_FILLED]]
        for number, (name, template) in enumerate(models.items(), start=1):
            eligible = lengths.index[lengths >= template.min_train_length]
            label = f"fold {split.fold}/{len(splits)} model {name} ({number}/{len(models)})"
            if len(eligible) == 0:
                logger.info("%s: no series long enough, skipped", label)
                timing_rows.append(_timing(split.fold, name, 0, 0.0, 0.0, "no series"))
                continue
            started = time.perf_counter()
            fit_done = started
            try:
                model = template.clone().fit(train[train[ID].isin(eligible)])
                fit_done = time.perf_counter()
                future = split.future[split.future[ID].isin(eligible)]
                future = known_columns_from_test(future, split.test, known_columns)
                predicted = model.predict(future)
            except Exception as exc:  # a model crash must not lose the other models' results
                if on_error == "raise":
                    raise
                seconds = time.perf_counter() - started
                message = f"{type(exc).__name__}: {exc}"
                logger.warning("%s FAILED after %.1fs, left out of this fold: %s", label, seconds, message)
                failure_rows.append({FOLD: split.fold, MODEL: name, "error": message})
                timing_rows.append(_timing(split.fold, name, len(eligible), seconds, 0.0, "failed"))
                continue
            if forecast_floor is not None:
                predicted[PREDICTION] = predicted[PREDICTION].clip(lower=forecast_floor)
            finished = time.perf_counter()
            timing_rows.append(
                _timing(split.fold, name, len(eligible), fit_done - started, finished - fit_done, "ok")
            )
            logger.info("%s: %d series, %.1fs", label, len(eligible), finished - started)

            merged = actual.merge(predicted, on=[ID, DATE], how="inner")
            merged.insert(1, FOLD, split.fold)
            merged.insert(2, MODEL, name)
            prediction_frames.append(merged[PREDICTION_COLUMNS])

            scored = merged[~merged[IS_FILLED]]
            for uid, g in scored.groupby(ID, sort=False):
                value = metric_fn(
                    g[TARGET].to_numpy(dtype=float),
                    g[PREDICTION].to_numpy(dtype=float),
                    train_y[(split.fold, uid)],
                    season_length,
                )
                score_rows.append(
                    {ID: uid, FOLD: split.fold, MODEL: name, SCORE: value, N_POINTS: len(g)}
                )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    return BacktestResult(
        metric=metric.lower(),
        season_length=season_length,
        cutoffs={s.fold: s.cutoff for s in splits},
        scores=pd.DataFrame(score_rows, columns=SCORE_COLUMNS),
        predictions=predictions,
        train_y=train_y,
        series_lengths=df.groupby(ID, sort=False).size(),
        outliers=(
            pd.concat(outlier_frames, ignore_index=True)[["fold", *OUTLIER_COLUMNS]]
            if outlier_frames
            else pd.DataFrame(columns=["fold", *OUTLIER_COLUMNS])
        ),
        timings=pd.DataFrame(timing_rows, columns=TIMING_COLUMNS),
        failures=pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS),
    )


def _timing(fold: int, model: str, n_series: int, fit: float, predict: float, status: str) -> dict:
    return {
        FOLD: fold, MODEL: model, "n_series": n_series,
        "fit_seconds": round(fit, 3), "predict_seconds": round(predict, 3), "status": status,
    }
