"""Baseline models. Always part of the candidates, and the benchmark for the others."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gts_forecaster.columns import ID
from gts_forecaster.models.base import HORIZON_STEP, BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Every future value equals the last observed value."""

    supports_update = True

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        self.last_ = {uid: y[-1] for uid, y in series.items()}

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        return future[ID].map(self.last_).to_numpy(dtype=float)


class SeasonalNaiveForecaster(BaseForecaster):
    """Every future value equals the value one season earlier (the last full season repeats)."""

    requires_seasonality = True
    supports_update = True

    def __init__(self, season_length: int = 1):
        self.season_length = season_length

    @property
    def min_train_length(self) -> int:
        return int(self.season_length)

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        m = self.season_length
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"season_length must be an integer >= 1, got {m!r}")
        self.last_season_ = {uid: y[-m:] for uid, y in series.items()}

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        m = int(self.season_length)
        steps = future[HORIZON_STEP].to_numpy()
        return np.array(
            [self.last_season_[uid][(h - 1) % m] for uid, h in zip(future[ID], steps, strict=True)],
            dtype=float,
        )


class DriftForecaster(BaseForecaster):
    """Continues the straight line between the first and the last observation."""

    supports_update = True

    @property
    def min_train_length(self) -> int:
        return 2

    def _fit(self, series: dict[str, np.ndarray], train: pd.DataFrame) -> None:
        self.last_ = {uid: y[-1] for uid, y in series.items()}
        self.slope_ = {uid: (y[-1] - y[0]) / (len(y) - 1) for uid, y in series.items()}

    def _predict(self, future: pd.DataFrame) -> np.ndarray:
        last = future[ID].map(self.last_).to_numpy(dtype=float)
        slope = future[ID].map(self.slope_).to_numpy(dtype=float)
        return last + slope * future[HORIZON_STEP].to_numpy(dtype=float)
