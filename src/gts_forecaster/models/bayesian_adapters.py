"""Prophet and Orbit: slow, per-series statistical models.

Both fit one model for every series, which takes seconds per series, so they are marked
`cost = "slow"` and are left out unless `fast_only` is switched off (`--no-fast-only` or
`fast_only: false` in config/default.yaml).

The models are fitted in `fit`, so a fitted model could be stored and reused later. If the library
fails on one series, that series gets the last observed value (`fallback: true`, the default) and a
warning names it; with `fallback: false` the error stops the run. Explanatory columns listed in
`known_future_columns` are used as regressors; other explanatory columns are ignored because their
future values are not known.

The libraries are imported only when a model is fitted (optional extras `prophet` and `orbit`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.features.calendar_features import frequency_kind
from gts_forecaster.models.base import BaseForecaster

logger = logging.getLogger(__name__)

# Prophet seasonalities that make sense for each frequency (yearly data has no seasonality inside the year)
_PROPHET_SEASONALITY = {
    "yearly": dict(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False),
    "monthly": dict(yearly_seasonality="auto", weekly_seasonality=False, daily_seasonality=False),
    "weekly": dict(yearly_seasonality="auto", weekly_seasonality=False, daily_seasonality=False),
    "daily": dict(yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False),
}


class PerSeriesForecaster(BaseForecaster):
    """Base class: one fitted model per series.

    season_length  seasonal period (filled in from the frequency profile)
    freq           pandas frequency alias (filled in from the frequency profile)
    known_columns  explanatory columns with known future values, used as regressors
    options        extra keyword arguments for the library model, over the defaults
    fallback       use the last observed value for a series the library fails on
    min_length     fewest observations a series needs (default: 10, or two seasons if more)
    """

    scope: ClassVar[str] = "local"
    cost: ClassVar[str] = "slow"

    def __init__(
        self,
        season_length: int = 1,
        freq: str = "MS",
        known_columns: Sequence[str] = (),
        options: dict[str, Any] | None = None,
        fallback: bool = True,
        min_length: int | None = None,
    ):
        self.season_length = season_length
        self.freq = freq
        self.known_columns = known_columns
        self.options = options
        self.fallback = fallback
        self.min_length = min_length
        m = season_length
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"season_length must be an integer >= 1, got {m!r}")

    @property
    def min_train_length(self) -> int:
        if self.min_length is not None:
            return int(self.min_length)
        return max(10, 2 * int(self.season_length))

    def _fit_one(self, frame: pd.DataFrame):
        raise NotImplementedError

    def _predict_one(self, model, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        known = list(self.known_columns)
        absent = [c for c in known if c not in train.columns]
        if absent:
            raise ValueError(f"known column(s) {absent} are not in the training data")
        self.last_ = {uid: float(y[-1]) for uid, y in series.items()}
        self.models_: dict[str, Any] = {}
        first_error: Exception | None = None
        for uid, g in train.groupby(ID, sort=False):
            frame = g[[DATE, TARGET, *known]].reset_index(drop=True)
            try:
                self.models_[uid] = self._fit_one(frame)
            except ImportError:
                raise
            except Exception as exc:
                if not self.fallback:
                    raise
                first_error = first_error or exc
                logger.warning(
                    "%s failed on series %s (%s); using its last value", type(self).__name__, uid, exc
                )
                self.models_[uid] = None
        if self.models_ and all(m is None for m in self.models_.values()):
            # Failing on EVERY series points at the setup (for example CmdStan not installed or
            # the Stan models not compiled), not at the data: stop instead of labelling a column
            # of last values as this model.
            raise RuntimeError(
                f"{type(self).__name__} failed on all {len(self.models_)} series; the library is "
                f"probably not set up correctly. First error: {type(first_error).__name__}: {first_error}"
            )

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        known = list(self.known_columns)
        absent = [c for c in known if c not in future.columns]
        if absent:
            raise ValueError(
                f"future is missing known column(s) {absent}: supply them with the known future "
                "features (known_future_columns / future_features_path)"
            )
        out = np.empty(len(future), dtype=float)
        ids = future[ID].to_numpy()
        for uid in pd.unique(ids):
            positions = np.flatnonzero(ids == uid)
            rows = future.iloc[positions][[DATE, *known]].reset_index(drop=True)
            model = self.models_[uid]
            if model is None:
                out[positions] = self.last_[uid]
                continue
            try:
                values = np.asarray(self._predict_one(model, rows), dtype=float)
                if values.shape != (len(rows),):
                    raise ValueError(f"expected {len(rows)} forecasts, got shape {values.shape}")
            except Exception as exc:
                if not self.fallback:
                    raise
                logger.warning(
                    "%s failed to predict series %s (%s); using its last value", type(self).__name__, uid, exc
                )
                values = np.full(len(rows), self.last_[uid])
            out[positions] = values
        return out


class ProphetForecaster(PerSeriesForecaster):
    """Facebook/Meta Prophet: trend with changepoints plus seasonality and regressors.

    Seasonalities are chosen from the frequency (none for yearly data); override them with options,
    for example {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.1}."""

    requires_packages: ClassVar[tuple[str, ...]] = ("prophet",)
    install_extra: ClassVar[str | None] = "prophet"
    # Prophet optimises with Newton's method below 100 points and allows 10,000 iterations: on a
    # (near) noise-free series it never converges and a fit took 36 s (measured 2026-10-11). Normal
    # series converge in a few dozen iterations, so a cap only stops the runaway cases.
    max_iterations: ClassVar[int] = 300

    def _fit_one(self, frame: pd.DataFrame):
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "This model needs the Prophet library. Install it with "
                "`poetry install --extras prophet` (or `pip install prophet`)."
            ) from exc
        for name in ("cmdstanpy", "prophet"):  # these loggers print a line per fit
            logging.getLogger(name).setLevel(logging.WARNING)
        settings = {**_PROPHET_SEASONALITY[frequency_kind(self.freq)], **(self.options or {})}
        model = Prophet(**settings)
        for column in self.known_columns:
            model.add_regressor(column)
        model.fit(frame, iter=self.max_iterations)
        return model

    def _predict_one(self, model, rows: pd.DataFrame) -> np.ndarray:
        return model.predict(rows)["yhat"].to_numpy()


class _OrbitForecaster(PerSeriesForecaster):
    """Uber's Orbit: Bayesian structural time series, estimated by maximum a posteriori (stan-map)."""

    requires_packages: ClassVar[tuple[str, ...]] = ("orbit",)
    install_extra: ClassVar[str | None] = "orbit"
    _model_name: ClassVar[str] = ""

    def _fit_one(self, frame: pd.DataFrame):
        try:
            import orbit.models as orbit_models
        except ImportError as exc:
            raise ImportError(
                "This model needs the Orbit library. Install it with "
                "`poetry install --extras orbit` (or `pip install orbit-ml`)."
            ) from exc
        settings = {
            "response_col": TARGET,
            "date_col": DATE,
            "regressor_col": list(self.known_columns) or None,
            "seasonality": int(self.season_length) if self.season_length > 1 else -1,
            "estimator": "stan-map",
            "n_bootstrap_draws": -1,  # point forecasts only (no uncertainty intervals)
            "seed": 8888,
            **(self.options or {}),
        }
        model = getattr(orbit_models, self._model_name)(**settings)
        model.fit(frame)
        return model

    def _predict_one(self, model, rows: pd.DataFrame) -> np.ndarray:
        predicted = model.predict(df=rows)
        return predicted.set_index(DATE)["prediction"].reindex(rows[DATE]).to_numpy()


class OrbitDLTForecaster(_OrbitForecaster):
    """Orbit DLT (damped local trend with seasonality and regressors)."""

    _model_name = "DLT"


class OrbitLGTForecaster(_OrbitForecaster):
    """Orbit LGT (local and global trend with seasonality and regressors).

    LGT does not accept negative values (such series get their last value), and Orbit warns that
    regressors in LGT will be deprecated: prefer DLT when known columns are used."""

    _model_name = "LGT"
