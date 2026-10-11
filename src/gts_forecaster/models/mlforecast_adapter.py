"""Global machine-learning models through Nixtla's MLForecast.

One regressor (LightGBM, XGBoost, Ridge, Lasso, RandomForest) is trained on all series together
(a "global" model). MLForecast builds the lag and rolling-window features of the target and
forecasts recursively: each predicted value is fed back as the next lag.

Other columns the model can use:
  - calendar features and static (identifier) features, from a feature set (`features:` in
    models.yaml; the set must not define lags, MLForecast makes those). Calendar features are known
    for the future; static features stay constant per series.
  - explanatory columns listed in `known_future_columns` (their future values are supplied by the
    user). Explanatory columns that are not known in the future cannot be used by a model that
    forecasts several steps ahead, so they are ignored.

The target of each series is scaled (`target_transforms`), so series with very different levels
can share one model. The libraries are imported only when a model is fitted; they belong to the
optional extra `ml` (`poetry install --extras ml`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.features.builder import FeatureBuilder, resolve_lengths
from gts_forecaster.features.calendar_features import frequency_kind
from gts_forecaster.models.alignment import align_forecast
from gts_forecaster.models.base import HORIZON_STEP, BaseForecaster

_ALIAS = "gts_model"
ROLLING_STATS = {"mean": "RollingMean", "std": "RollingStd", "min": "RollingMin", "max": "RollingMax"}
TARGET_TRANSFORMS = ("local_standard", "local_minmax", "difference", "seasonal_difference")
# MLForecast date features used when no feature set is given (names of pandas date attributes)
_DEFAULT_DATE_FEATURES = {
    "yearly": [],
    "monthly": ["month", "quarter"],
    "weekly": ["month", "quarter", "dayofyear"],
    "daily": ["dayofweek", "month", "day", "dayofyear"],
}


def _import_mlforecast():
    try:
        from mlforecast import MLForecast
        from mlforecast import lag_transforms as lag_transforms
        from mlforecast import target_transforms as target_transforms
    except ImportError as exc:
        raise ImportError(
            "This model needs the MLForecast library. Install it with "
            "`poetry install --extras ml` (or `pip install mlforecast`)."
        ) from exc
    return MLForecast, lag_transforms, target_transforms


class MLForecastAdapter(BaseForecaster):
    """Base class of the MLForecast adapters. Subclasses say which regressor to build.

    season_length      seasonal period (filled in from the frequency profile)
    freq               pandas frequency alias (filled in from the frequency profile)
    lags               target lags: integers or `season`, `2*season`, ... (default 1, 2, 3, season).
                       Also per frequency: {monthly: [1, 2, 3, 12], daily: [...]}; a frequency that
                       is not listed gets the default
    rolling_windows    sizes (>= 2) of rolling statistics of the target, ending at lag 1
                       (default 3 and season); per frequency like `lags`
    rolling_stats      any of mean, std, min, max (default mean, std)
    date_features      MLForecast date features (pandas date attribute names). Default: a few per
                       frequency, or none when a feature set supplies the calendar
    target_transforms  how each series is scaled before training, any of local_standard,
                       local_minmax, difference, seasonal_difference (default local_standard)
    regressor_params   parameters of the regressor, over the defaults of the model
    known_columns      explanatory columns whose future values are known (filled in from
                       `known_future_columns`)
    feature_builder    calendar and static features (set by `features:` in models.yaml)
    num_threads        threads MLForecast may use to build features (default -1 = all cores)
    min_length         override of the fewest observations a series needs
    """

    scope: ClassVar[str] = "global"
    requires_packages: ClassVar[tuple[str, ...]] = ("mlforecast",)
    install_extra: ClassVar[str | None] = "ml"
    _default_params: ClassVar[dict[str, Any]] = {}
    # which `ml_features` group of the frequency profile gives this model its default lags/windows
    feature_group: ClassVar[str] = "trees"

    def __init__(
        self,
        season_length: int = 1,
        freq: str = "MS",
        lags: Sequence[int | str] | None = None,
        rolling_windows: Sequence[int | str] | None = None,
        rolling_stats: Sequence[str] = ("mean", "std"),
        date_features: Sequence[str] | None = None,
        target_transforms: Sequence[str] = ("local_standard",),
        regressor_params: dict[str, Any] | None = None,
        known_columns: Sequence[str] = (),
        feature_builder: FeatureBuilder | None = None,
        num_threads: int = -1,
        min_length: int | None = None,
    ):
        self.season_length = season_length
        self.freq = freq
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.rolling_stats = rolling_stats
        self.date_features = date_features
        self.target_transforms = target_transforms
        self.regressor_params = regressor_params
        self.known_columns = known_columns
        self.feature_builder = feature_builder
        self.num_threads = num_threads
        self.min_length = min_length
        self._check()

    # ---- configuration --------------------------------------------------------------------

    def _check(self) -> None:
        m = self.season_length
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"season_length must be an integer >= 1, got {m!r}")
        unknown = sorted(set(self.rolling_stats) - set(ROLLING_STATS))
        if unknown:
            raise ValueError(f"unknown rolling_stats {unknown}; choose from {list(ROLLING_STATS)}")
        unknown = sorted(set(self.target_transforms) - set(TARGET_TRANSFORMS))
        if unknown:
            raise ValueError(f"unknown target_transforms {unknown}; choose from {list(TARGET_TRANSFORMS)}")
        if self.feature_builder is not None and self.feature_builder.target_columns:
            raise ValueError(
                "the feature set of an MLForecast model must not define lags or rolling windows: "
                "MLForecast builds those itself (set `lags` and `rolling_windows` in the model params)"
            )
        if not self._resolved_lags():
            raise ValueError("a model needs at least one lag")

    def _for_frequency(self, value, what: str):
        """`value` itself, or its entry for this frequency when it is given per frequency."""
        if not isinstance(value, Mapping):
            return value
        unknown = sorted(set(value) - set(_DEFAULT_DATE_FEATURES))
        if unknown:
            raise ValueError(f"{what}: unknown frequency {unknown}; use {sorted(_DEFAULT_DATE_FEATURES)}")
        return value.get(frequency_kind(self.freq))

    def _resolved_lags(self) -> tuple[int, ...]:
        lags = self._for_frequency(self.lags, "lags")
        raw = [1, 2, 3, "season"] if lags is None else list(lags)
        return tuple(sorted(set(resolve_lengths(raw, int(self.season_length), "lags"))))

    def _resolved_windows(self) -> tuple[int, ...]:
        windows = self._for_frequency(self.rolling_windows, "rolling_windows")
        raw = [3, "season"] if windows is None else list(windows)
        windows = resolve_lengths(raw, int(self.season_length), "rolling_windows")
        return tuple(sorted({w for w in windows if w >= 2}))

    def _resolved_date_features(self) -> list[str]:
        if self.date_features is not None:
            return list(self.date_features)
        if self.feature_builder is not None:
            return []  # the feature set supplies the calendar
        return list(_DEFAULT_DATE_FEATURES[frequency_kind(self.freq)])

    @property
    def min_train_length(self) -> int:
        if self.min_length is not None:
            return int(self.min_length)
        history = max(max(self._resolved_lags()), max(self._resolved_windows(), default=0) + 1)
        return history + 3  # a few rows left to train on

    def _make_regressor(self):
        raise NotImplementedError

    def _regressor_settings(self) -> dict[str, Any]:
        return {**self._default_params, **(self.regressor_params or {})}

    # ---- columns --------------------------------------------------------------------------

    def _calendar_columns(self) -> list[str]:
        return self.feature_builder.calendar_columns() if self.feature_builder else []

    def _static_columns(self) -> list[str]:
        return self.feature_builder.static_columns if self.feature_builder else []

    # ---- fit and predict ------------------------------------------------------------------

    def _target_transforms(self, module):
        made = {
            "local_standard": lambda: module.LocalStandardScaler(),
            "local_minmax": lambda: module.LocalMinMaxScaler(),
            "difference": lambda: module.Differences([1]),
            "seasonal_difference": lambda: module.Differences([int(self.season_length)]),
        }
        return [made[name]() for name in self.target_transforms] or None

    def _lag_transforms(self, module):
        windows = self._resolved_windows()
        if not windows:
            return None
        return {1: [getattr(module, ROLLING_STATS[s])(window_size=w) for w in windows for s in self.rolling_stats]}

    supports_update = True  # new history is passed to MLForecast.predict(new_df=...)

    def _frames(self, train: pd.DataFrame):
        """(base, frame, static): the history, and the table handed to MLForecast."""
        known = list(self.known_columns)
        absent = [c for c in known if c not in train.columns]
        if absent:
            raise ValueError(f"known column(s) {absent} are not in the training data")
        base = train[[ID, DATE, TARGET, *known]].reset_index(drop=True)
        calendar, static = self._calendar_columns(), self._static_columns()
        if self.feature_builder is not None and (calendar or static):
            features = self.feature_builder.transform_history(base)
            frame = pd.concat([base, features[calendar + static]], axis=1)
        else:
            frame = base
        clash = sorted(set(self._resolved_date_features()) & set(frame.columns))
        if clash:  # MLForecast would add a second column with the same name
            raise ValueError(
                f"date_features {clash} clash with columns of the same name (feature set or known "
                "columns); remove them from date_features"
            )
        return base, frame, static

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        mlforecast, lag_module, target_module = _import_mlforecast()
        base, frame, static = self._frames(train)
        self.mlf_ = mlforecast(
            models={_ALIAS: self._make_regressor()},
            freq=self.freq,
            lags=list(self._resolved_lags()),
            lag_transforms=self._lag_transforms(lag_module),
            date_features=self._resolved_date_features(),
            target_transforms=self._target_transforms(target_module),
            num_threads=int(self.num_threads),
        )
        self.mlf_.fit(
            frame, id_col=ID, time_col=DATE, target_col=TARGET, static_features=static, dropna=True
        )
        self.train_ = base[[ID, DATE, TARGET]]
        self.new_df_ = None

    def _update(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        """Keep the fitted regressor; forecast from the newer history (predict(new_df=...)).

        Note: MLForecast refits the target transforms (for example the per-series scaling of
        `local_standard`) on the new history, and the lags / rolling windows are computed from it.
        Only the regressor itself stays frozen."""
        base, frame, _ = self._frames(train)
        self.train_ = base[[ID, DATE, TARGET]]
        self.new_df_ = frame

    def _future_exogenous(self, future: pd.DataFrame, horizon: int) -> pd.DataFrame | None:
        """The future values of the calendar and known columns, for every series and step."""
        known = list(self.known_columns)
        absent = [c for c in known if c not in future.columns]
        if absent:
            raise ValueError(
                f"future is missing known column(s) {absent}: supply them with the known future "
                "features (known_future_columns / future_features_path)"
            )
        calendar = self._calendar_columns()
        if not calendar and not known:
            return None
        counts = future.groupby(ID).size()
        steps_ok = future.groupby(ID)[HORIZON_STEP].apply(
            lambda s: sorted(s.tolist()) == list(range(1, horizon + 1))
        )
        if (counts != horizon).any() or not steps_ok.all():
            # MLForecast matches the rows of X_df to forecast steps by position, so every series
            # needs exactly the steps 1..horizon
            raise ValueError(
                f"{type(self).__name__} needs the future steps 1..{horizon} for every series when it "
                "uses calendar or known columns"
            )
        frame = future[[ID, DATE, HORIZON_STEP, *known]].reset_index(drop=True)
        if calendar:
            features = self.feature_builder.transform_future(self.train_, frame)
            frame = pd.concat([frame, features[calendar]], axis=1)
        return frame[[ID, DATE, *known, *calendar]]

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        horizon = int(future[HORIZON_STEP].max())
        ids = list(pd.unique(future[ID]))
        extra = {"new_df": self.new_df_} if getattr(self, "new_df_", None) is not None else {}
        out = self.mlf_.predict(
            h=horizon, X_df=self._future_exogenous(future, horizon), ids=ids, **extra
        )
        if ID not in out.columns:
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


class LightGBMForecaster(MLForecastAdapter):
    """Gradient boosted trees (LightGBM)."""

    requires_packages: ClassVar[tuple[str, ...]] = ("mlforecast", "lightgbm")
    _default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1,
        "n_jobs": -1, "random_state": 0,
    }

    def _make_regressor(self):
        import lightgbm

        return lightgbm.LGBMRegressor(**self._regressor_settings())


class XGBoostForecaster(MLForecastAdapter):
    """Gradient boosted trees (XGBoost, histogram method)."""

    requires_packages: ClassVar[tuple[str, ...]] = ("mlforecast", "xgboost")
    _default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "tree_method": "hist",
        "n_jobs": -1, "random_state": 0,
    }

    def _make_regressor(self):
        import xgboost

        return xgboost.XGBRegressor(**self._regressor_settings())


class RandomForestForecaster(MLForecastAdapter):
    """Random forest (scikit-learn)."""

    requires_packages: ClassVar[tuple[str, ...]] = ("mlforecast", "sklearn")
    _default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 200, "min_samples_leaf": 3, "max_features": 0.5, "n_jobs": -1,
        "random_state": 0,
    }

    def _make_regressor(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**self._regressor_settings())


class _LinearForecaster(MLForecastAdapter):
    """Linear model on standardised features (missing values filled with the median)."""

    requires_packages: ClassVar[tuple[str, ...]] = ("mlforecast", "sklearn")
    feature_group: ClassVar[str] = "linear"

    def _estimator(self, **params):
        raise NotImplementedError

    def _make_regressor(self):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), self._estimator(**self._regressor_settings())
        )


class RidgeForecaster(_LinearForecaster):
    """Ridge regression."""

    _default_params: ClassVar[dict[str, Any]] = {"alpha": 1.0}

    def _estimator(self, **params):
        from sklearn.linear_model import Ridge

        return Ridge(**params)


class LassoForecaster(_LinearForecaster):
    """Lasso regression."""

    _default_params: ClassVar[dict[str, Any]] = {"alpha": 0.001, "max_iter": 5000}

    def _estimator(self, **params):
        from sklearn.linear_model import Lasso

        return Lasso(**params)
