"""Common interface for every forecasting model.

Models follow the scikit-learn style (`fit`, `predict`, `get_params`, `set_params`, `clone`)
but work on a panel of series:

    model.fit(train)        train: unique_id, ds, y (+ explanatory columns)
    model.predict(future)   future: unique_id, ds, horizon_step (+ future explanatory columns)
                            returns: unique_id, ds, yhat, one row per row of `future`

The horizon is whatever `future` contains, so a model never needs to know the frequency or a
fixed horizon. Libraries with a different API (Prophet, Orbit, Nixtla) get an adapter that
exposes this interface.

A model that is fitted per series ("local") or once on all series ("global") is told apart
by `scope`. `min_train_length` is the least number of observations a series needs for this
model; the pipelines only hand a model the series that have enough.
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET

HORIZON_STEP = "horizon_step"
PREDICTION = "yhat"


class BaseForecaster:
    """Base class. Subclasses implement `_fit` and `_predict`."""

    scope: ClassVar[str] = "local"  # "local" (one model per series) or "global" (one for all)
    requires_seasonality: ClassVar[bool] = False  # needs a seasonal period of at least 2
    # "fast" models run by default; "slow" ones (Prophet, Orbit) only when `fast_only` is off.
    cost: ClassVar[str] = "fast"
    # Third-party packages the model imports, and the Poetry extra that installs them. The catalog
    # leaves the model out, with a warning, when a package is missing.
    requires_packages: ClassVar[tuple[str, ...]] = ()
    install_extra: ClassVar[str | None] = None
    # True when `update(train)` can take newer history without refitting (frozen-models mode).
    # Models that cannot are refitted in that mode, with a warning.
    supports_update: ClassVar[bool] = False

    # --- to implement ---------------------------------------------------------------------

    @property
    def min_train_length(self) -> int:
        """Fewest observations a series needs to be used by this model."""
        return 1

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        raise NotImplementedError

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # --- public API -----------------------------------------------------------------------

    def _prepare(self, train: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        missing = [c for c in (ID, DATE, TARGET) if c not in train.columns]
        if missing:
            raise ValueError(f"train is missing column(s) {missing}")
        if train.empty:
            raise ValueError("train is empty")
        if train[TARGET].isna().any():
            raise ValueError("train contains null values in y")
        ordered = train.sort_values([ID, DATE], kind="stable")
        series = {uid: g[TARGET].to_numpy(dtype=float) for uid, g in ordered.groupby(ID, sort=False)}
        too_short = {u: len(y) for u, y in series.items() if len(y) < self.min_train_length}
        if too_short:
            raise ValueError(
                f"{type(self).__name__} needs at least {self.min_train_length} observations per "
                f"series; too short: {dict(list(too_short.items())[:5])}"
            )
        return series, ordered

    def fit(self, train: pd.DataFrame) -> BaseForecaster:
        series, ordered = self._prepare(train)
        self._fit(series, ordered)
        self.fitted_ids_ = tuple(series)
        return self

    def update(self, train: pd.DataFrame) -> BaseForecaster:
        """Give an already fitted model newer history WITHOUT refitting its parameters.

        Only for models with `supports_update`. `train` may only contain series the model was
        fitted on. By default the history is simply taken in again (`_fit`), which is right for
        models whose whole state is the history (naive, zero-shot foundation models).
        """
        if not self.supports_update:
            raise NotImplementedError(f"{type(self).__name__} cannot be updated without refitting")
        if not hasattr(self, "fitted_ids_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit first")
        series, ordered = self._prepare(train)
        unknown = sorted(set(series) - set(self.fitted_ids_))
        if unknown:
            raise ValueError(f"update contains series the model was not fitted on: {unknown[:5]}")
        self._update(series, ordered)
        self.fitted_ids_ = tuple(series)
        return self

    def _update(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        self._fit(series, train)

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fitted_ids_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit first")
        missing = [c for c in (ID, DATE, HORIZON_STEP) if c not in future.columns]
        if missing:
            raise ValueError(f"future is missing column(s) {missing}")
        unknown = sorted(set(future[ID]) - set(self.fitted_ids_))
        if unknown:
            raise ValueError(f"future contains series the model was not fitted on: {unknown[:5]}")
        yhat = np.asarray(self._predict(future), dtype=float)
        if yhat.shape != (len(future),):
            raise ValueError("a model must return one forecast per row of `future`")
        return pd.DataFrame(
            {ID: future[ID].to_numpy(), DATE: future[DATE].to_numpy(), PREDICTION: yhat}
        )

    # --- scikit-learn style parameter handling --------------------------------------------

    def get_params(self) -> dict[str, Any]:
        parameters = inspect.signature(type(self).__init__).parameters.values()
        names = [
            p.name
            for p in parameters
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params: Any) -> BaseForecaster:
        valid = self.get_params()
        unknown = set(params) - set(valid)
        if unknown:
            raise ValueError(f"invalid parameter(s) {sorted(unknown)} for {type(self).__name__}")
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def clone(self) -> BaseForecaster:
        """A new, unfitted model with the same parameters."""
        return type(self)(**self.get_params())

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"
