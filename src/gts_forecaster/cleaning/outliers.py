"""Detect and replace severe outliers in the training data.

Method "mad" (robust, per series, target `y` only):
  1. Seasonality: when the series has at least three full seasons away from its ends, the typical offset of each season
     position from a rolling-median trend is removed, so ordinary seasonal peaks are not flagged.
  2. Each point is compared with the median of its two neighbours on each side (not including
     itself), which follows smooth trends and cycles without being pulled by the outlier. The
     first and last two points have no such comparison and are never flagged.
  3. A point is an outlier when that residual is more than `threshold` robust standard
     deviations (1.4826 x MAD of the residuals) from zero. The default of 5 flags only severe ones.
  4. Flagged points are replaced (`action`), so the series stays regular:
       clip         (default) the threshold bound that was crossed: expected value +/-
                    threshold x scale. A very high or very low value is kept as "high" or "low",
                    but no longer extreme enough to distort the fit.
       interpolate  linear interpolation between the nearest unflagged neighbours
       median       the median of the series' unflagged values
       mean         the mean of the series' unflagged values

Safeguards: periods filled in by the missing-data step are never flagged; a series with fewer than
`MIN_OBSERVATIONS` points is left alone; and if more than `max_share` of a series would be flagged,
the series is left unchanged with a warning (that is more likely a level shift than outliers).

Only training data is cleaned. Test windows are scored against the raw actual values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, IS_FILLED, TARGET

logger = logging.getLogger(__name__)

OUTLIER_METHODS = ("mad", "none")
OUTLIER_ACTIONS = ("clip", "interpolate", "median", "mean")
MIN_OBSERVATIONS = 8
IS_OUTLIER = "is_outlier"
REPORT_COLUMNS = [ID, DATE, "y_raw", "y_clean", "score"]
_MAD_TO_SD = 1.4826
_SCALE_FLOOR = 1e-8
_MAX_LISTED = 5
_NEIGHBOURHOOD = 2  # points this close to a stronger candidate are not flagged in the same pass


@dataclass
class OutlierResult:
    data: pd.DataFrame  # the input with `y` cleaned and a boolean column `is_outlier`
    report: pd.DataFrame  # one row per replaced point: unique_id, ds, y_raw, y_clean, score


def _seasonal_component(y: np.ndarray, season_length: int) -> np.ndarray:
    """Typical offset of each season position from the trend; zeros if the series is too short.

    Only points whose trend window is complete (not within half a window of either end) are used,
    because the trend is biased near the ends of a trending series.
    """
    n = len(y)
    if season_length < 2:
        return np.zeros(n)
    window = season_length if season_length % 2 else season_length + 1
    half = window // 2
    if n - 2 * half < 3 * season_length:  # the median needs at least three samples per position
        return np.zeros(n)
    trend = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    position = np.arange(n) % season_length
    interior = np.zeros(n, bool)
    interior[half : n - half] = True
    offsets = pd.Series((y - trend)[interior]).groupby(position[interior]).median()
    return offsets.reindex(range(season_length)).fillna(0.0).to_numpy()[position]


def _neighbour_prediction(y: np.ndarray) -> np.ndarray:
    """Median of the two neighbours on each side, excluding the point itself. NaN at the ends."""
    n = len(y)
    padded = np.full((n, 4), np.nan)
    for column, shift in enumerate((-2, -1, 1, 2)):
        lo, hi = max(0, -shift), min(n, n - shift)
        padded[lo:hi, column] = y[lo + shift : hi + shift]
    prediction = np.nanmedian(padded, axis=1)
    prediction[np.isnan(padded).any(axis=1)] = np.nan  # needs all four neighbours
    return prediction


def _score_details(y: np.ndarray, season_length: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust z-scores, the value expected at each point, and the robust scale (all NaN-aware).

    By construction  y = expected + score * scale.
    """
    n = len(y)
    if n < MIN_OBSERVATIONS:
        return np.full(n, np.nan), np.full(n, np.nan), float("nan")
    seasonal = _seasonal_component(y, season_length)
    deseasonalised = y - seasonal
    prediction = _neighbour_prediction(deseasonalised)
    resid = deseasonalised - prediction
    valid = resid[~np.isnan(resid)]
    centre = float(np.median(valid))
    scale = _MAD_TO_SD * np.median(np.abs(valid - centre))
    scale = max(scale, _SCALE_FLOOR * max(1.0, float(np.median(np.abs(y)))))
    return (resid - centre) / scale, prediction + seasonal + centre, float(scale)


def robust_scores(y: np.ndarray, season_length: int = 1) -> np.ndarray:
    """Robust z-score of every point (see module docstring). NaN where it cannot be computed,
    which includes the first and last two points of a series."""
    return _score_details(y, season_length)[0]


def _replacement(y: np.ndarray, flagged: np.ndarray, bound: np.ndarray, action: str) -> np.ndarray:
    """The series with the flagged points replaced according to `action`."""
    if action == "interpolate":
        return _interpolate(y, flagged)
    if action == "clip":
        return np.where(flagged, bound, y)
    reference = np.mean(y[~flagged]) if action == "mean" else np.median(y[~flagged])
    return np.where(flagged, reference, y)


def _interpolate(y: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    good = ~flagged
    index = np.arange(len(y))
    return np.where(flagged, np.interp(index, index[good], y[good]), y)


def _flag_series(
    y: np.ndarray, filled: np.ndarray, threshold: float, max_share: float, season_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag outliers in one series.

    Returns the flags, the score each flagged point had, and the value of the threshold bound it
    crossed (expected value +/- threshold x scale), which the `clip` action uses.

    Outliers are found in passes. A large outlier distorts the comparison values of its
    neighbours, so each pass accepts only the most extreme point of every group of nearby
    candidates, replaces it (by interpolation, for scoring only), and scores the series again.
    This stops the neighbours of one outlier from being flagged too.
    """
    n = len(y)
    flagged = np.zeros(n, bool)
    final_score = np.full(n, np.nan)
    bound = np.full(n, np.nan)
    work = y
    while True:
        scores, expected, scale = _score_details(work, season_length)
        candidate = (np.nan_to_num(np.abs(scores), nan=0.0) > threshold) & ~filled & ~flagged
        if not candidate.any():
            return flagged, final_score, bound
        accepted: list[int] = []
        for i in np.flatnonzero(candidate)[np.argsort(-np.abs(scores[candidate]))]:
            if all(abs(i - j) > _NEIGHBOURHOOD for j in accepted):
                accepted.append(int(i))
        flagged[accepted] = True
        final_score[accepted] = scores[accepted]
        bound[accepted] = expected[accepted] + np.sign(scores[accepted]) * threshold * scale
        if flagged.sum() > max_share * n:  # too many: the caller leaves the series alone
            return flagged, final_score, bound
        work = _interpolate(y, flagged)


def handle_outliers(
    train: pd.DataFrame,
    *,
    method: str = "mad",
    threshold: float = 5.0,
    max_share: float = 0.1,
    season_length: int = 1,
    action: str = "clip",
) -> OutlierResult:
    """Replace severe outliers in the `y` of every series of `train` (model format).

    `action` says what a flagged point is replaced with (see the module docstring).
    """
    if action not in OUTLIER_ACTIONS:
        raise ValueError(f"outlier action must be one of {list(OUTLIER_ACTIONS)}, got {action!r}")
    if method not in OUTLIER_METHODS:
        raise ValueError(f"outlier method must be one of {list(OUTLIER_METHODS)}, got {method!r}")
    if not threshold > 0:
        raise ValueError("outlier threshold must be > 0")
    if not 0 < max_share <= 1:
        raise ValueError("outlier max_share must be in (0, 1]")

    out = train.sort_values([ID, DATE], kind="stable").reset_index(drop=True)
    out[IS_OUTLIER] = False
    if method == "none" or out.empty:
        return OutlierResult(out, pd.DataFrame(columns=REPORT_COLUMNS))

    filled = out[IS_FILLED].to_numpy(bool) if IS_FILLED in out.columns else np.zeros(len(out), bool)
    y_all = out[TARGET].to_numpy(dtype=float).copy()
    flagged_all = np.zeros(len(out), bool)
    scores_all = np.full(len(out), np.nan)

    for uid, rows in out.groupby(ID, sort=False).indices.items():
        y = y_all[rows]
        flagged, scores, bound = _flag_series(y, filled[rows], threshold, max_share, season_length)
        if not flagged.any():
            continue
        if flagged.mean() > max_share:
            logger.warning(
                "uuid %s: %d of %d points look like outliers (more than %.0f%%), so the series "
                "is left unchanged. This looks like a level shift, not outliers.",
                uid, int(flagged.sum()), len(y), 100 * max_share,
            )
            continue
        flagged_all[rows] = flagged
        scores_all[rows] = scores
        y_all[rows] = _replacement(y, flagged, bound, action)

    report = pd.DataFrame(
        {
            ID: out.loc[flagged_all, ID].to_numpy(),
            DATE: out.loc[flagged_all, DATE].to_numpy(),
            "y_raw": out.loc[flagged_all, TARGET].to_numpy(dtype=float),
            "y_clean": y_all[flagged_all],
            "score": scores_all[flagged_all],
        },
        columns=REPORT_COLUMNS,
    )
    out[TARGET] = y_all
    out[IS_OUTLIER] = flagged_all
    if len(report):
        affected = report[ID].nunique()
        logger.warning("outliers: replaced %d point(s) in %d series", len(report), affected)
        for r in report.head(_MAX_LISTED).itertuples(index=False):
            logger.info("  uuid %s %s: %s -> %s", r[0], pd.Timestamp(r[1]).date(), r[2], round(r[3], 4))
    return OutlierResult(out, report)
