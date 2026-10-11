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
from gts_forecaster.features.calendar_features import frequency_kind
from gts_forecaster.models.alignment import align_forecast
from gts_forecaster.models.base import HORIZON_STEP, BaseForecaster

logger = logging.getLogger(__name__)

_ALIAS = "gts_model"
_PERIODS_PER_YEAR = {"yearly": 1, "monthly": 12, "weekly": 52, "daily": 365}


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
    n_jobs         processes StatsForecast may use (default 1 = no multiprocessing, -1 = all cores).
                   On Windows more than 1 starts new processes for every fit, about 13 s each on the
                   project laptop, which only pays off with many long series (measured 2026-10-11)
    fallback       if StatsForecast fails on a series, use the naive forecast for it instead of
                   stopping the whole run
    min_length     override of the fewest observations a series needs
    max_history_years
                   fit on only the most recent years of each series, per frequency, for example
                   {"daily": 3}; a frequency not listed uses the whole series (default: all).
                   Only a speed measure for slow models on long series; forecasts start from the
                   last observation either way.
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
        max_history_years: dict[str, float] | None = None,
    ):
        self.season_length = season_length
        self.freq = freq
        self.options = options
        self.n_jobs = n_jobs
        self.fallback = fallback
        self.min_length = min_length
        self.max_history_years = max_history_years
        self._max_history()  # fail at start-up on a bad setting, not in the middle of a run

    def _max_history(self) -> int | None:
        """Most recent observations to fit on for this frequency, or None for all."""
        if not self.max_history_years:
            return None
        if not isinstance(self.max_history_years, dict):
            raise ValueError(
                f"max_history_years must map a frequency to years, e.g. {{'daily': 3}}, "
                f"got {self.max_history_years!r}"
            )
        unknown = sorted(set(self.max_history_years) - set(_PERIODS_PER_YEAR))
        if unknown:
            raise ValueError(
                f"max_history_years: unknown frequency {unknown}; use {list(_PERIODS_PER_YEAR)}"
            )
        for value in self.max_history_years.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"max_history_years must be positive numbers, got {value!r}")
        years = self.max_history_years.get(frequency_kind(self.freq))
        if years is None:
            return None
        periods = int(round(years * _PERIODS_PER_YEAR[frequency_kind(self.freq)]))
        if periods < self.min_train_length:
            raise ValueError(
                f"max_history_years {years} gives {periods} observations, fewer than the "
                f"{self.min_train_length} this model needs"
            )
        return periods

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
        data = train[[ID, DATE, TARGET]]
        cap = self._max_history()
        if cap is not None:
            data = data.groupby(ID, sort=False).tail(cap)
            logger.info("%s: fitting on the last %d observations of each series",
                        type(self).__name__, cap)
        self.sf_.fit(data.reset_index(drop=True))
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
    """ARIMA with automatic order selection per series.

    Seasonal ARIMA is far too slow for long seasons (weekly data, 52: tens of minutes per fold),
    so a season longer than `max_seasonal_period` gets a non-seasonal ARIMA, like AutoETS does
    above 24. MSTL, Theta, the MLForecast models and Chronos still cover the seasonality. To
    force a seasonal fit anyway, pass {"season_length": 52} in `options`."""

    _model_name = "AutoARIMA"
    _min_obs = 6
    max_seasonal_period: ClassVar[int] = 24

    def _effective_season(self) -> int:
        m = int(self.season_length)
        return 1 if m > self.max_seasonal_period else m

    @property
    def min_train_length(self) -> int:
        if self.min_length is not None or "season_length" in (self.options or {}):
            return super().min_train_length
        m = self._effective_season()
        return max(self._min_obs, 2 * m if m > 1 else 0)

    def _model_kwargs(self) -> dict[str, Any]:
        m = self._effective_season()
        if m != int(self.season_length) and "season_length" not in (self.options or {}):
            logger.info(
                "AutoARIMA: season length %d is over %d, fitting a non-seasonal ARIMA",
                self.season_length, self.max_seasonal_period,
            )
        return {"season_length": m, "alias": _ALIAS, **(self.options or {})}


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
