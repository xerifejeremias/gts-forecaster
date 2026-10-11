"""Pretrained foundation models from Amazon's Chronos family (zero-shot, CPU).

There is no training: `fit` only stores the recent history of each series, and `predict` hands it to
the pretrained model. The model weights are downloaded from Hugging Face the first time a model is
used and then cached. The package `chronos-forecasting` (with PyTorch) belongs to the optional extra
`foundation` (`poetry install --extras foundation`).

  chronos2   Chronos-2 (`amazon/chronos-2`, 120M parameters; `autogluon/chronos-2-small`, 28M).
             Can use explanatory columns: those known in the future (`known_future_columns`) and,
             with `use_past_covariates`, the other explanatory columns as past-only covariates.
  chronos_bolt  Chronos-Bolt (`amazon/chronos-bolt-tiny`, `-mini`, `-small`, `-base`). Univariate and
             much faster. Ignores explanatory columns.

Each backtest fold clones a model, so the loaded network is cached and shared between clones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.models.alignment import align_forecast
from gts_forecaster.models.base import HORIZON_STEP, BaseForecaster

_QUANTILE = 0.5  # the point forecast is the median
_PIPELINES: dict[tuple[str, str, str], Any] = {}  # (kind, model id, device) -> loaded pipeline


def _import_chronos():
    try:
        import chronos
    except ImportError as exc:
        raise ImportError(
            "This model needs the Chronos library. Install it with "
            "`poetry install --extras foundation` (or `pip install chronos-forecasting`)."
        ) from exc
    return chronos


def clear_pipeline_cache() -> None:
    """Forget the loaded networks (frees memory; used by the tests)."""
    _PIPELINES.clear()


class ChronosAdapter(BaseForecaster):
    """Base class of the Chronos adapters.

    model_id     Hugging Face id of the pretrained model
    season_length  seasonal period (filled in from the frequency profile; not used by the model)
    freq         pandas frequency alias (filled in from the frequency profile)
    device_map   "cpu" (default), or for example "cuda" if the computer has a GPU
    max_context  most recent observations of each series given to the model (keeps CPU runs fast)
    min_length   fewest observations a series needs
    """
    supports_update = True  # zero-shot: the history is all there is to take in

    scope: ClassVar[str] = "local"  # every series is forecast on its own
    requires_packages: ClassVar[tuple[str, ...]] = ("chronos", "torch")
    install_extra: ClassVar[str | None] = "foundation"
    _default_model: ClassVar[str] = ""
    _kind: ClassVar[str] = ""

    def __init__(
        self,
        model_id: str | None = None,
        season_length: int = 1,
        freq: str = "MS",
        device_map: str = "cpu",
        max_context: int = 1024,
        min_length: int = 5,
    ):
        self.model_id = model_id or self._default_model
        self.season_length = season_length
        self.freq = freq
        self.device_map = device_map
        self.max_context = max_context
        self.min_length = min_length
        self._check()

    def _check(self) -> None:
        for name in ("max_context", "min_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")

    @property
    def min_train_length(self) -> int:
        return int(self.min_length)

    # ---- model loading --------------------------------------------------------------------

    def _load(self):
        raise NotImplementedError

    def _pipeline(self):
        key = (self._kind, self.model_id, self.device_map)
        if key not in _PIPELINES:
            _PIPELINES[key] = self._load()
        return _PIPELINES[key]

    # ---- fit --------------------------------------------------------------------------------

    def _context_columns(self) -> list[str]:
        return []

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        columns = self._context_columns()
        absent = [c for c in columns if c not in train.columns]
        if absent:
            raise ValueError(f"column(s) {absent} are not in the training data")
        recent = train.groupby(ID, sort=False, group_keys=False).tail(int(self.max_context))
        self.context_ = recent[[ID, DATE, TARGET, *columns]].reset_index(drop=True)
        self._pipeline()  # load now, so a missing library or model shows up at fit time


class Chronos2Forecaster(ChronosAdapter):
    """Chronos-2, with optional covariates.

    known_columns        explanatory columns whose future values are known (filled in from
                         `known_future_columns`)
    explanatory_columns  all explanatory columns (filled in from the config)
    use_past_covariates  also give the explanatory columns that are NOT known in the future to the
                         model as past-only covariates (default off)
    """

    _default_model: ClassVar[str] = "amazon/chronos-2"
    _kind: ClassVar[str] = "chronos2"

    def __init__(
        self,
        model_id: str | None = None,
        season_length: int = 1,
        freq: str = "MS",
        device_map: str = "cpu",
        max_context: int = 1024,
        min_length: int = 5,
        known_columns: Sequence[str] = (),
        explanatory_columns: Sequence[str] = (),
        use_past_covariates: bool = False,
    ):
        self.known_columns = known_columns
        self.explanatory_columns = explanatory_columns
        self.use_past_covariates = use_past_covariates
        super().__init__(model_id, season_length, freq, device_map, max_context, min_length)

    def _past_columns(self) -> list[str]:
        if not self.use_past_covariates:
            return []
        known = set(self.known_columns)
        return [c for c in self.explanatory_columns if c not in known]

    def _context_columns(self) -> list[str]:
        return [*self.known_columns, *self._past_columns()]

    def _load(self):
        return _import_chronos().Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device_map)

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        horizon = int(future[HORIZON_STEP].max())
        known = list(self.known_columns)
        future_df = None
        if known:
            absent = [c for c in known if c not in future.columns]
            if absent:
                raise ValueError(
                    f"future is missing known column(s) {absent}: supply them with the known future "
                    "features (known_future_columns / future_features_path)"
                )
            if (future.groupby(ID).size() != horizon).any():
                raise ValueError(
                    f"{type(self).__name__} needs the same number of future steps ({horizon}) for every "
                    "series when it uses known columns"
                )
            future_df = future[[ID, DATE, *known]].reset_index(drop=True)
        context = self.context_[self.context_[ID].isin(set(future[ID]))]
        out = self._pipeline().predict_df(
            context,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=[_QUANTILE],
            id_column=ID,
            timestamp_column=DATE,
            target=TARGET,
            freq=self.freq,  # otherwise inferred per series, which fails on a series with a gap
        )
        column = "predictions" if "predictions" in out.columns else str(_QUANTILE)
        if column not in out.columns:
            raise ValueError(f"cannot find the forecast in the columns {list(out.columns)}")
        return align_forecast(
            future, out, value_column=column, owner=type(self).__name__, freq=self.freq
        )


class ChronosBoltForecaster(ChronosAdapter):
    """Chronos-Bolt: fast univariate forecasts, no explanatory columns."""

    _default_model: ClassVar[str] = "amazon/chronos-bolt-small"
    _kind: ClassVar[str] = "chronos_bolt"

    def _load(self):
        return _import_chronos().BaseChronosPipeline.from_pretrained(
            self.model_id, device_map=self.device_map
        )

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        import torch

        horizon = int(future[HORIZON_STEP].max())
        ids = list(pd.unique(future[ID]))
        history = {uid: g[TARGET].to_numpy(dtype="float32") for uid, g in self.context_.groupby(ID, sort=False)}
        contexts = [torch.tensor(history[uid], dtype=torch.float32) for uid in ids]
        # The first argument is called `inputs` in chronos-forecasting 2.x (the model cards still
        # show the old name `context`), so pass it by position.
        quantiles, _mean = self._pipeline().predict_quantiles(
            contexts, prediction_length=horizon, quantile_levels=[_QUANTILE]
        )
        values = np.asarray(quantiles, dtype=float)[:, :, 0]  # series x steps, the median
        last = self.context_.groupby(ID)[DATE].max()
        frames = []
        for uid, row in zip(ids, values, strict=True):
            dates = pd.date_range(last[uid], periods=horizon + 1, freq=self.freq)[1:]
            frames.append(pd.DataFrame({ID: uid, DATE: dates, "value": row}))
        return align_forecast(
            future, pd.concat(frames, ignore_index=True), value_column="value",
            owner=type(self).__name__, freq=self.freq,
        )
