"""Produce forecasts: one column per model, then one combined forecast per series.

Step 1  `forecast_by_model`  fits every model on all the data and returns a wide table:
            unique_id, ds, <one column per model>
        (a model that cannot forecast a series, for example because it has too little history,
        leaves NaN for that series).
Step 2  `combine_forecasts`  takes, for each series, the models chosen in selection.json and
        returns the weighted average: ds, unique_id, forecast.
Series with a fallback (too much missing data) get the fallback value for every horizon date.

This is the internal shape of the result. The final output format (and its contract) is applied
later by a separate post-processing step.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import pandas as pd

from gts_forecaster.columns import DATE, ID
from gts_forecaster.models.base import PREDICTION, BaseForecaster
from gts_forecaster.splitting import Split

logger = logging.getLogger(__name__)

FORECAST = "forecast"
FORECAST_COLUMNS = [DATE, ID, FORECAST]


class ForecastError(ValueError):
    """A forecast could not be produced for every series."""


def forecast_by_model(
    split: Split,
    models: Mapping[str, BaseForecaster],
    *,
    on_error: str = "raise",
    failures: list[dict] | None = None,
    timings: list[dict] | None = None,
    fitted: Mapping[str, BaseForecaster] | None = None,
    forecast_floor: float | None = None,
) -> pd.DataFrame:
    """Fit each model on the training data of `split` (a forecast split) and predict `future`.

    `on_error="skip"`: a crashing model leaves NaN in its column (and a row in `failures`) so the
    other models still finish. `failures` and `timings`, if given, are lists that receive one
    dict per failed model / per model (model, n_series, fit_seconds, predict_seconds, status).

    `fitted` (frozen-models mode): models fitted earlier. A model that supports `update` takes in
    the newer history without refitting (status "updated"); any other model, or one that must
    forecast a series it was not fitted on, is refitted from its template (status "refitted").
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    wide = split.future[[ID, DATE]].copy()
    for number, (name, template) in enumerate(models.items(), start=1):
        label = f"model {name} ({number}/{len(models)})"
        eligible = split.train_lengths.index[split.train_lengths >= template.min_train_length]
        if len(eligible) == 0:
            logger.info("%s: no series long enough, column left empty", label)
            wide[name] = float("nan")
            _record(timings, name, 0, 0.0, 0.0, "no series")
            continue
        started = fit_done = time.perf_counter()
        status = "ok"
        try:
            history = split.train[split.train[ID].isin(eligible)]
            saved = fitted.get(name) if fitted else None
            if saved is None:
                model = template.clone().fit(history)
                if fitted is not None:
                    status = "refitted"
                    logger.warning("%s: no saved model, refitted", label)
            else:
                unseen = sorted(set(eligible) - set(saved.fitted_ids_))
                if saved.supports_update and not unseen:
                    model = saved.update(history)
                    status = "updated"
                else:
                    reason = (
                        f"{len(unseen)} series were not in the saved model" if unseen
                        else "this model cannot take new data without refitting"
                    )
                    logger.warning("%s: refitted (%s)", label, reason)
                    model = template.clone().fit(history)
                    status = "refitted"
            fit_done = time.perf_counter()
            predicted = model.predict(split.future[split.future[ID].isin(eligible)])
        except Exception as exc:
            if on_error == "raise":
                raise
            seconds = time.perf_counter() - started
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("%s FAILED after %.1fs, column left empty: %s", label, seconds, message)
            if failures is not None:
                failures.append({"model": name, "error": message})
            _record(timings, name, len(eligible), seconds, 0.0, "failed")
            wide[name] = float("nan")
            continue
        finished = time.perf_counter()
        _record(timings, name, len(eligible), fit_done - started, finished - fit_done, status)
        logger.info("%s: %d series, %.1fs", label, len(eligible), finished - started)
        if forecast_floor is not None:  # e.g. 0: volumes cannot be negative
            predicted[PREDICTION] = predicted[PREDICTION].clip(lower=forecast_floor)
        predicted = predicted.rename(columns={PREDICTION: name})
        wide = wide.merge(predicted, on=[ID, DATE], how="left")
    return wide.sort_values([ID, DATE], kind="stable").reset_index(drop=True)


def _record(timings: list[dict] | None, model: str, n: int, fit: float, predict: float, status: str):
    if timings is not None:
        timings.append({
            "model": model, "n_series": n, "fit_seconds": round(fit, 3),
            "predict_seconds": round(predict, 3), "status": status,
        })


def combine_forecasts(
    wide: pd.DataFrame,
    selection: Mapping[str, Any],
    fallback: pd.DataFrame | None = None,
    horizon_dates: pd.DatetimeIndex | None = None,
    *,
    drop_missing: bool = False,
    forecast_floor: float | None = None,
) -> pd.DataFrame:
    """Weighted average of each series' selected models. Returns ds, unique_id, forecast.

    `selection` is the content of selection.json. Raises ForecastError if a series has no entry,
    or a selected model has no forecast for it, rather than returning gaps.

    `drop_missing=True` is for runs where models may have crashed: a selected model without a
    forecast is left out (with a warning) and the remaining weights are rescaled to sum to 1.
    A series is still an error if none of its selected models has a forecast.
    """
    entries = selection["series"]
    frames = []
    for uid, g in wide.groupby(ID, sort=False):
        if uid not in entries:
            raise ForecastError(f"series {uid} has no entry in the selection")
        members = []
        for member in entries[uid]["models"]:
            name = member["name"]
            if name not in g.columns:
                raise ForecastError(f"series {uid}: selected model '{name}' was not run")
            if g[name].isna().any():
                if not drop_missing:
                    raise ForecastError(
                        f"series {uid}: selected model '{name}' has no forecast "
                        "(too little history or the model failed). "
                        "Choose another model in selection.json."
                    )
                logger.warning("series %s: selected model '%s' has no forecast, left out", uid, name)
                continue
            members.append((name, float(member["weight"])))
        if not members:
            raise ForecastError(f"series {uid}: none of the selected models produced a forecast")
        weight_sum = sum(w for _, w in members)
        total = pd.Series(0.0, index=g.index)
        for name, weight in members:
            total += g[name] * (weight / weight_sum if drop_missing else weight)
        frames.append(pd.DataFrame({DATE: g[DATE], ID: uid, FORECAST: total}))

    if fallback is not None and len(fallback):
        if horizon_dates is None:
            raise ForecastError("horizon_dates are needed to build fallback forecasts")
        for row in fallback.itertuples(index=False):
            frames.append(
                pd.DataFrame(
                    {DATE: horizon_dates, ID: getattr(row, ID), FORECAST: float(row.fallback_value)}
                )
            )
    if not frames:
        raise ForecastError("no forecasts were produced")
    out = pd.concat(frames, ignore_index=True)
    if forecast_floor is not None:  # also covers fallback values
        out[FORECAST] = out[FORECAST].clip(lower=forecast_floor)
    return out[FORECAST_COLUMNS].sort_values([ID, DATE], kind="stable").reset_index(drop=True)
