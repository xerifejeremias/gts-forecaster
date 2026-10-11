"""Sanity checks of the final forecasts: do the numbers make sense next to each series' history?

These are plausibility checks, not the output contract (which already guarantees the shape: no
gaps, no negative values, every series and date present). Each check flags a series whose forecast
looks wrong for that series; what happens next is the config key `forecast_check_action`:
  warn   (default) log a warning per flagged series and write outputs/<frequency>/forecast_checks.csv;
         the forecasts are still delivered
  raise  stop the forecast before anything is written (ForecastCheckError)
  off    do not run the checks

The checks (per series; series with the fallback forecast are skipped):
  outlier       a forecast step is more than `threshold` robust standard deviations away from the
                median of its two neighbours on each side, the score of the training-data outlier
                method (cleaning/outliers.extension_scores; no seasonal adjustment here), with the
                history's own spread as the yardstick. Catches spikes and dips.
  total_change  the forecast total differs from the actual total of the same number of periods just
                before the cutoff by more than `max_ratio` times (up or down). Catches runaway trends
                and collapses. Only when both totals are positive.
  start_jump    the first steps of the forecast (up to 3) differ by more than `max_ratio` times from
                BOTH the last actual values and the same periods one season earlier (only the last
                values when there is no season). Catches a forecast that starts at the wrong level;
                requiring both keeps a real seasonal turn, or one noisy reference, from flagging.
  zero          the whole forecast is 0 although the recent history is not (e.g. the floor at 0
                swallowed a forecast that went negative).
"""

from __future__ import annotations

import logging
from collections.abc import Collection

import numpy as np
import pandas as pd

from gts_forecaster.cleaning.outliers import MIN_OBSERVATIONS, extension_scores
from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.forecasting import FORECAST

logger = logging.getLogger(__name__)

CHECKS = ("outlier", "total_change", "start_jump", "zero")
CHECK_COLUMNS = [ID, "check", DATE, "value", "reference", "score", "threshold", "message"]
START_STEPS = 3  # forecast steps compared by start_jump
SCALE_FLOOR_SHARE = 0.01  # the outlier yardstick is at least 1% of the series' typical level
_MAX_LISTED = 5


class ForecastCheckError(DataContractError):
    """The forecasts failed the sanity checks and forecast_check_action is `raise`."""


def _ratio(value: float, reference: float) -> float | None:
    """value / reference when both are positive, otherwise None (no meaningful ratio)."""
    if value > 0 and reference > 0:
        return value / reference
    return None


def _flag(uid, check, ds, value, reference, score, threshold, message) -> dict:
    return {ID: uid, "check": check, DATE: ds, "value": value, "reference": reference,
            "score": score, "threshold": threshold, "message": message}


def check_series(
    uid: str, history: np.ndarray, dates: pd.DatetimeIndex, forecast: np.ndarray, *,
    season_length: int, threshold: float, max_ratio: float,
) -> list[dict]:
    """The flags for one series. `history` is the target up to the cutoff (oldest first),
    `forecast` the final forecast with its `dates`."""
    flags: list[dict] = []
    h, n, m = len(forecast), len(history), max(int(season_length), 1)

    if n >= MIN_OBSERVATIONS:
        scores = extension_scores(history, forecast, scale_floor_share=SCALE_FLOOR_SHARE)
        for k in np.flatnonzero(np.abs(np.nan_to_num(scores)) > threshold):
            flags.append(_flag(
                uid, "outlier", dates[k], float(forecast[k]), None, float(scores[k]), threshold,
                f"forecast step {k + 1} is {abs(scores[k]):.1f} robust standard deviations "
                f"{'above' if scores[k] > 0 else 'below'} its neighbours",
            ))

    if n >= h:
        total, before = float(forecast.sum()), float(history[-h:].sum())
        ratio = _ratio(total, before)
        if ratio is not None and not 1 / max_ratio <= ratio <= max_ratio:
            flags.append(_flag(
                uid, "total_change", None, total, before, ratio, max_ratio,
                f"forecast total is {ratio:.2f}x the actual total of the {h} periods before",
            ))

    k = min(START_STEPS, h)
    references = {f"the last {k} actual values": history[-k:]}
    if m > 1 and n >= m:
        references["the same periods one season earlier"] = history[n - m : n - m + k]
    if h and all(len(base) == k for base in references.values()):
        start = float(forecast[:k].mean())
        ratios = {where: _ratio(start, float(base.mean())) for where, base in references.items()}
        if all(r is not None and not 1 / max_ratio <= r <= max_ratio for r in ratios.values()):
            where, ratio = min(ratios.items(), key=lambda item: abs(np.log(item[1])))
            flags.append(_flag(
                uid, "start_jump", dates[0], start, float(references[where].mean()), ratio, max_ratio,
                f"first {k} forecast step(s) average {ratio:.2f}x {where}"
                + (" (and further from the other reference)" if len(ratios) > 1 else ""),
            ))

    recent = history[-max(m, START_STEPS):]
    if h and np.all(forecast == 0) and recent.mean() > 0:
        flags.append(_flag(
            uid, "zero", None, 0.0, float(recent.mean()), None, None,
            f"the whole forecast is 0, but the last {len(recent)} actual values average "
            f"{recent.mean():.4g}",
        ))
    return flags


def check_forecasts(
    history: pd.DataFrame, forecast: pd.DataFrame, *, season_length: int, threshold: float,
    max_ratio: float, skip_ids: Collection[str] = (),
) -> pd.DataFrame:
    """Run the checks on every series. `history` is the model frame (unique_id, ds, y),
    `forecast` the final forecast (ds, unique_id, forecast). Returns one row per flag
    (CHECK_COLUMNS); empty when everything looks plausible."""
    skip = set(skip_ids)
    past = {uid: g.sort_values(DATE)[TARGET].to_numpy(float) for uid, g in history.groupby(ID, sort=False)}
    flags: list[dict] = []
    for uid, g in forecast.groupby(ID, sort=False):
        if uid in skip or uid not in past:
            continue
        g = g.sort_values(DATE)
        flags.extend(check_series(
            uid, past[uid], pd.DatetimeIndex(g[DATE]), g[FORECAST].to_numpy(float),
            season_length=season_length, threshold=threshold, max_ratio=max_ratio,
        ))
    return pd.DataFrame(flags, columns=CHECK_COLUMNS)


def report_flags(flags: pd.DataFrame, action: str) -> None:
    """Log the flags; raise ForecastCheckError when `action` is raise and something was flagged."""
    if flags.empty:
        logger.info("forecast checks: no series flagged")
        return
    counts = flags.groupby("check")[ID].nunique()
    summary = ", ".join(f"{check} {counts[check]}" for check in CHECKS if check in counts)
    n_series = flags[ID].nunique()
    for uid, g in list(flags.groupby(ID, sort=False))[:_MAX_LISTED]:
        logger.warning("forecast check, uuid %s: %s", uid, "; ".join(g["message"]))
    more = f" (first {_MAX_LISTED} listed above)" if n_series > _MAX_LISTED else ""
    message = f"forecast checks flagged {n_series} series{more}: {summary}"
    if action == "raise":
        raise ForecastCheckError(
            message + ". Nothing was written. Look at those series, or set forecast_check_action: "
            "warn to deliver the forecasts with a warning."
        )
    logger.warning("%s. Details in forecast_checks.csv", message)
