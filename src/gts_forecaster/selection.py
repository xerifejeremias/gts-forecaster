"""Choose the model (or ensemble) for each series from the backtest results.

For every series:
  1. Reference folds: the folds in which the reference model (`naive`, which needs the least
     data) was scored. Every model competes only on these same folds, and must have a valid
     score on all of them, so the comparison is fair.
  2. Rank the competing models by their mean score over the reference folds (lower is better).
  3. Take the best `top_n`. One model means the single best model; more means an ensemble whose
     members are weighted equally or by inverse error (config: ensemble_weights).
  4. A series with no reference fold (too little history to backtest) gets a fixed short-series
     ensemble: the first `short_series_n` models of `short_series_models` that it has enough
     data for, equally weighted.
  5. A series where no model has a valid score gets the reference model.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, IS_FILLED, TARGET
from gts_forecaster.evaluation.backtest import FOLD, MODEL, SCORE, BacktestResult
from gts_forecaster.evaluation.metrics import get_metric
from gts_forecaster.models.base import PREDICTION, BaseForecaster

logger = logging.getLogger(__name__)

REFERENCE_MODEL = "naive"
ENSEMBLE_ROW = "ensemble"
TABLE_COLUMNS = [ID, MODEL, "n_folds", SCORE, "rank", "weight", "note"]
_EPS = 1e-12


class SelectionError(ValueError):
    """A model selection could not be made."""


@dataclass(frozen=True)
class SeriesSelection:
    """The chosen model(s) for one series."""

    models: tuple[tuple[str, float], ...]  # (model name, weight); weights sum to 1
    note: str | None = None
    locked: bool = False


@dataclass
class Selection:
    series: dict[str, SeriesSelection]  # unique_id -> selection, in order of the data
    table: pd.DataFrame  # scores per series and model, for backtest_scores.csv


def _weights(scores: Sequence[float], how: str) -> list[float]:
    n = len(scores)
    if how == "equal" or n == 1:
        return [1.0 / n] * n
    if how == "inverse_error":
        inverse = np.array([1.0 / (s + _EPS) for s in scores])
        return list(inverse / inverse.sum())
    raise SelectionError(f"unknown ensemble weighting {how!r}")


def _ensemble_score(
    result: BacktestResult, uid: str, members: Sequence[str], weights: Sequence[float], folds: Sequence[int],
    predictions: pd.DataFrame | None = None,
) -> float:
    """Backtest score of the weighted combination of the members' forecasts. `predictions`: this
    series' rows of `result.predictions` (looked up once per series by the caller)."""
    metric_fn = get_metric(result.metric)
    p = result.predictions if predictions is None else predictions
    p = p[(p[ID] == uid) & p[MODEL].isin(members) & p[FOLD].isin(folds) & ~p[IS_FILLED]]
    wide = p.pivot_table(index=[FOLD, DATE], columns=MODEL, values=PREDICTION)
    actual = p.drop_duplicates([FOLD, DATE]).set_index([FOLD, DATE])[TARGET].reindex(wide.index)
    combined = sum(wide[m] * w for m, w in zip(members, weights, strict=True))
    values = []
    for fold in folds:
        mask = wide.index.get_level_values(FOLD) == fold
        if not mask.any():
            return float("nan")
        values.append(
            metric_fn(
                actual[mask].to_numpy(float),
                combined[mask].to_numpy(float),
                result.train_y[(fold, uid)],
                result.season_length,
            )
        )
    return float(np.mean(values))


def short_series_choice(
    uid: str, length: int, models: Mapping[str, BaseForecaster],
    short_series_models: Sequence[str], short_series_n: int,
) -> list[str]:
    """The fixed short-series ensemble of a series: the first `short_series_n` models of
    `short_series_models` that exist and that it has enough data for."""
    usable = [m for m in short_series_models if m in models and length >= models[m].min_train_length]
    chosen = usable[:short_series_n]
    if not chosen:
        raise SelectionError(
            f"series {uid} ({length} observations) cannot use any short-series model "
            f"{list(short_series_models)}"
        )
    return chosen


NOT_BACKTESTED_NOTE = "not in the last backtest (new or repaired series): fixed short-series ensemble"


def add_missing_series(
    selection: Mapping, lengths: pd.Series, models: Mapping[str, BaseForecaster],
    short_series_models: Sequence[str], short_series_n: int,
) -> tuple[dict, list[str]]:
    """A copy of a saved selection (selection.json content) with an entry for every series in
    `lengths` (unique_id -> training observations) that has none: series that are new since the
    last backtest, or that had the fallback then. They get the fixed short-series ensemble until
    the next backtest chooses for them. Returns the selection and the ids that were added."""
    series = dict(selection["series"])
    added = []
    for uid, length in lengths.items():
        if uid in series:
            continue
        chosen = short_series_choice(uid, int(length), models, short_series_models, short_series_n)
        series[uid] = {
            "locked": False,
            "models": [{"name": m, "weight": 1.0 / len(chosen)} for m in chosen],
            "note": NOT_BACKTESTED_NOTE,
        }
        added.append(uid)
    if added:
        logger.warning(
            "%d series are not in the saved selection (new since the last backtest, or no longer a "
            "fallback); they get the short-series ensemble until the next backtest, e.g. %s",
            len(added), added[:3],
        )
    return {**selection, "series": series}, added


def select_models(
    result: BacktestResult,
    models: Mapping[str, BaseForecaster],
    *,
    top_n: int,
    ensemble_weights: str = "equal",
    short_series_models: Sequence[str] = ("naive", "seasonal_naive", "drift"),
    short_series_n: int = 2,
) -> Selection:
    """Pick the model(s) for every series in the backtested data."""
    if REFERENCE_MODEL not in models:
        raise SelectionError(f"the model catalog must contain the reference model '{REFERENCE_MODEL}'")
    if top_n < 1 or short_series_n < 1:
        raise SelectionError("top_n and short_series_n must be >= 1")
    order = {name: k for k, name in enumerate(models)}

    # group once: filtering the whole tables per series grows with the square of the series count
    scores_by_id = {uid: g for uid, g in result.scores.groupby(ID, sort=False)}
    predictions_by_id = {uid: g for uid, g in result.predictions.groupby(ID, sort=False)}
    empty_scores = result.scores.iloc[0:0]
    series: dict[str, SeriesSelection] = {}
    rows: list[dict] = []

    for uid, length in result.series_lengths.items():
        mine = scores_by_id.get(uid, empty_scores)
        reference_folds = sorted(mine.loc[mine[MODEL] == REFERENCE_MODEL, FOLD].unique())

        # 4. too short to backtest -> fixed ensemble
        if not reference_folds:
            chosen = short_series_choice(uid, length, models, short_series_models, short_series_n)
            note = "too little history to backtest: fixed short-series ensemble"
            series[uid] = SeriesSelection(tuple((m, 1.0 / len(chosen)) for m in chosen), note)
            rows += [
                {ID: uid, MODEL: m, "n_folds": 0, SCORE: np.nan, "rank": np.nan, "weight": 1.0 / len(chosen), "note": note}
                for m in chosen
            ]
            continue

        # 1-2. rank the models that have a valid score on every reference fold
        competing = []
        for name in models:
            s = mine[(mine[MODEL] == name) & mine[FOLD].isin(reference_folds)]
            if len(s) == len(reference_folds) and s[SCORE].notna().all():
                competing.append((float(s[SCORE].mean()), order[name], name))
        competing.sort()

        # 5. nothing has a valid score
        if not competing:
            note = f"no valid backtest score; using '{REFERENCE_MODEL}'"
            series[uid] = SeriesSelection(((REFERENCE_MODEL, 1.0),), note)
            rows.append({ID: uid, MODEL: REFERENCE_MODEL, "n_folds": len(reference_folds), SCORE: np.nan, "rank": np.nan, "weight": 1.0, "note": note})
            continue

        # 3. best model or top-N ensemble
        top = competing[:top_n]
        names = [name for _, _, name in top]
        weights = _weights([score for score, _, _ in top], ensemble_weights)
        series[uid] = SeriesSelection(tuple(zip(names, weights, strict=True)))
        chosen_weight = dict(zip(names, weights, strict=True))
        for rank, (score, _, name) in enumerate(competing, start=1):
            rows.append({ID: uid, MODEL: name, "n_folds": len(reference_folds), SCORE: score, "rank": rank, "weight": chosen_weight.get(name, 0.0), "note": None})
        if len(names) > 1:
            rows.append({ID: uid, MODEL: ENSEMBLE_ROW, "n_folds": len(reference_folds), SCORE: _ensemble_score(result, uid, names, weights, reference_folds, predictions_by_id.get(uid)), "rank": np.nan, "weight": 1.0, "note": f"top {len(names)} models, {ensemble_weights} weights"})

    return Selection(series=series, table=pd.DataFrame(rows, columns=TABLE_COLUMNS))
