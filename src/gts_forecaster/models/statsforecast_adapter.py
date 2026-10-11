"""Adapters for Nixtla's StatsForecast: AutoETS, AutoARIMA, AutoTheta and MSTL.

StatsForecast fits every series on its own (a "local" model). One adapter class wraps one
StatsForecast model and exposes the common `fit(train)` / `predict(future)` interface. The
library is imported only when a model is fitted, so the project works without the optional
`stats` extra (`poetry install --extras stats`).

The StatsForecast forecast for step h of a series is matched to the row of `future` with the
same `horizon_step`, and the dates are checked, so a frequency or anchor mismatch is an error
rather than a silently shifted forecast.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.models.alignment import align_forecast
from gts_forecaster.models.base import HORIZON_STEP, BaseForecaster

logger = logging.getLogger(__name__)

_ALIAS = "gts_model"


def _import_statsforecast():
    try:
        import statsforecast
        from statsforecast import models as sf_models
    except ImportError as exc:
        raise ImportError(
            "This model needs the StatsForecast library. Install it with "
            "`poetry install --extras stats` (or `pip install statsforecast`)."
        ) from exc
    return statsforecast.StatsForecast, sf_models


class StatsForecastAdapter(BaseForecaster):
    """Base class of the StatsForecast adapters. Subclasses say which model to build.

    season_length  seasonal period (filled in from the frequency profile)
    freq           pandas frequency alias (filled in from the frequency profile)
    options        extra keyword arguments for the StatsForecast model, for example
                   {"model": "ZZZ"} for AutoETS; they override the defaults
    n_jobs         processes StatsForecast may use (1 = no multiprocessing, -1 or None = all cores;
                   on Windows more than 1 starts new processes, which works from the poetry commands)
    fallback       if StatsForecast fails on a series, use the naive forecast for it instead of
                   stopping the whole run
    min_length     override of the fewest observations a series needs
    """

    scope: ClassVar[str] = "local"
    requires_packages: ClassVar[tuple[str, ...]] = ("statsforecast",)
    install_extra: ClassVar[str | None] = "stats"

    _model_name: ClassVar[str] = ""  # name of the class in statsforecast.models
    _min_obs: ClassVar[int] = 4  # fewest observations, whatever the period

    def __init__(
        self,
        season_length: int = 1,
        freq: str = "MS",
        options: dict[str, Any] | None = None,
        n_jobs: int = 1,
        fallback: bool = True,
        min_length: int | None = None,
    ):
        self.season_length = season_length
        self.freq = freq
        self.options = options
        self.n_jobs = n_jobs
        self.fallback = fallback
        self.min_length = min_length

    @property
    def min_train_length(self) -> int:
        if self.min_length is not None:
            return int(self.min_length)
        m = int(self.season_length)
        # a seasonal model needs two full seasons to tell the season from the level
        return max(self._min_obs, 2 * m if m > 1 else 0)

    def _model_kwargs(self) -> dict[str, Any]:
        return {"season_length": int(self.season_length), "alias": _ALIAS, **(self.options or {})}

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        m = self.season_length
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"season_length must be an integer >= 1, got {m!r}")
        stats_forecast, sf_models = _import_statsforecast()
        model = getattr(sf_models, self._model_name)(**self._model_kwargs())
        fallback = sf_models.Naive(alias=_ALIAS) if self.fallback else None
        n_jobs = -1 if self.n_jobs is None else int(self.n_jobs)
        self.sf_ = stats_forecast(
            models=[model], freq=self.freq, n_jobs=n_jobs, fallback_model=fallback
        )
        self.sf_.fit(train[[ID, DATE, TARGET]].reset_index(drop=True))
        self.fallback_ids_ = self._fallback_series()
        if self.fallback_ids_:
            logger.warning(
                "%s failed on %d of %d series, which got the naive forecast instead, e.g. %s",
                type(self).__name__, len(self.fallback_ids_), len(series), self.fallback_ids_[:3],
            )

    def _fallback_series(self) -> list[str]:
        """Series where StatsForecast used the naive fallback. The library swaps the model
        silently (keeping our alias), so look at the class of each fitted model."""
        fitted = getattr(self.sf_, "fitted_", None)
        uids = getattr(self.sf_, "uids", None)
        if fitted is None or uids is None or self._model_name == "Naive":
            return []
        try:
            return [str(u) for u, row in zip(list(uids), fitted) if type(row[0]).__name__ == "Naive"]
        except (TypeError, IndexError):  # unexpected shape in a future version: just do not report
            return []

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        out = self.sf_.predict(h=int(future[HORIZON_STEP].max()))
        if ID not in out.columns:  # older versions return unique_id as the index
            out = out.reset_index()
        value_columns = [c for c in out.columns if c not in (ID, DATE)]
        if _ALIAS in value_columns:
            column = _ALIAS
        elif len(value_columns) == 1:
            column = value_columns[0]
        else:
            raise ValueError(f"cannot tell which column holds the forecast: {value_columns}")
        return align_forecast(
            future, out, value_column=column, owner=type(self).__name__, freq=self.freq
        )


class AutoETSForecaster(StatsForecastAdapter):
    """Exponential smoothing (ETS) with the best form chosen automatically per series."""

    _model_name = "AutoETS"
    _min_obs = 7  # statsforecast's automatic ETS search rejects 6 or fewer observations


class AutoARIMAForecaster(StatsForecastAdapter):
    """ARIMA with automatic order selection per series."""

    _model_name = "AutoARIMA"
    _min_obs = 6


class AutoThetaForecaster(StatsForecastAdapter):
    """Theta method with automatic choice among its variants per series."""

    _model_name = "AutoTheta"


class MSTLForecaster(StatsForecastAdapter):
    """Multiple seasonal-trend decomposition; the trend is forecast with ETS.

    Pass several periods with options, for example {"season_length": [7, 365]} for daily data."""

    requires_seasonality: ClassVar[bool] = True
    _model_name = "MSTL"

    def _periods(self) -> list[int]:
        periods = (self.options or {}).get("season_length", self.season_length)
        periods = list(periods) if isinstance(periods, (list, tuple)) else [periods]
        for p in periods:
            if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 2:
                raise ValueError(f"MSTL seasonal periods must be integers >= 2, got {periods!r}")
        return [int(p) for p in periods]

    @property
    def min_train_length(self) -> int:
        if self.min_length is not None:
            return int(self.min_length)
        # STL needs two full cycles of the LONGEST period (e.g. 730 days for [7, 365])
        return max(self._min_obs, 2 * max(self._periods()))

    def _model_kwargs(self) -> dict[str, Any]:
        kwargs = super()._model_kwargs()
        kwargs["season_length"] = self._periods()
        return kwargs
