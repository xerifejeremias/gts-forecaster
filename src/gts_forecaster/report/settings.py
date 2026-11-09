"""The run settings shown in the report's FAQ tab, so its answers quote the values really used.

Only modelling settings are included: no paths, no SQL query or connection (a connection string can
hold a password), nothing that identifies where the data came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# (key, attribute of RunConfig); the profile values are added separately
_KEYS = (
    "frequency", "horizon", "metric", "n_folds", "backtest_step", "top_n", "ensemble_weights",
    "selection_rule", "blend_share", "short_series_models", "short_series_n", "max_fill_periods",
    "fallback", "outlier_method", "outlier_threshold", "outlier_action", "outlier_max_share",
    "known_future_columns", "static_columns", "explanatory_columns", "fast_only", "forecast_mode",
    "forecast_models", "forecast_floor", "forecast_check_action", "forecast_check_threshold",
    "forecast_check_max_ratio", "on_model_error",
)


def _plain(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def run_settings(
    config, selection: Mapping[str, Any] | None = None, horizon: int | None = None
) -> dict[str, Any]:
    """Settings of this run (and of the backtest that made the selection, where they can differ).

    `horizon` is the length of the forecast shown in the report, which wins over the config value
    (the report command may run without the --horizon the forecast used)."""
    out = {key: _plain(getattr(config, key, None)) for key in _KEYS}
    meta = (selection or {}).get("meta") or {}
    if horizon is not None:
        out["horizon"] = int(horizon)
    if out["backtest_step"] is None:  # null step = the backtest's horizon
        out["backtest_step"] = meta.get("horizon") or out["horizon"]
    profile = config.profile
    out["season_length"] = profile.season_length
    out["min_series_length"] = profile.min_series_length
    out["pandas_freq"] = profile.pandas_freq
    # the selection was made by an earlier backtest: its own settings decide what the forecast uses
    out["selection"] = {k: meta.get(k) for k in (
        "created", "metric", "n_folds", "top_n", "ensemble_weights", "selection_rule", "horizon",
        "data_cutoff", "includes_slow_models", "variant",
    ) if k in meta}
    return out
