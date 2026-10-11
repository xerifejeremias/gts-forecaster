"""Model catalog: builds the candidate models from config/models.yaml.

Each entry in models.yaml is a model name (used in selection.json) with a `type`, optional
`params` and optional `features` (the name of a feature set in config/features.yaml, for models that
use explicit feature columns; the model class then takes a `feature_builder` argument). Adapters for other libraries are registered in MODEL_TYPES when they are added.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from gts_forecaster.config import ConfigError, project_root
from gts_forecaster.features.builder import FeatureBuilder
from gts_forecaster.features.builder import build_feature_builders, load_feature_config
from gts_forecaster.features.errors import FeatureError
from gts_forecaster.models.base import BaseForecaster
from gts_forecaster.models.bayesian_adapters import OrbitDLTForecaster, OrbitLGTForecaster, ProphetForecaster
from gts_forecaster.models.chronos_adapter import ChronosBoltForecaster, Chronos2Forecaster
from gts_forecaster.models.mlforecast_adapter import (
    LassoForecaster,
    LightGBMForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    XGBoostForecaster,
)
from gts_forecaster.models.naive import DriftForecaster, NaiveForecaster, SeasonalNaiveForecaster
from gts_forecaster.models.statsforecast_adapter import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    AutoThetaForecaster,
    MSTLForecaster,
)

logger = logging.getLogger(__name__)

MODEL_TYPES: dict[str, type[BaseForecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "drift": DriftForecaster,
    "auto_ets": AutoETSForecaster,
    "auto_arima": AutoARIMAForecaster,
    "auto_theta": AutoThetaForecaster,
    "mstl": MSTLForecaster,
    "lightgbm": LightGBMForecaster,
    "xgboost": XGBoostForecaster,
    "random_forest": RandomForestForecaster,
    "ridge": RidgeForecaster,
    "lasso": LassoForecaster,
    "chronos2": Chronos2Forecaster,
    "chronos_bolt": ChronosBoltForecaster,
    "prophet": ProphetForecaster,
    "orbit_dlt": OrbitDLTForecaster,
    "orbit_lgt": OrbitLGTForecaster,
}


def _package_available(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def load_model_specs(
    config_dir: Path | None = None, types: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Read config/models.yaml: {model name: {"type": ..., "params": {...}}}.

    `types` keeps only the models of those types (for example the baselines)."""
    path = (Path(config_dir) if config_dir else project_root() / "config") / "models.yaml"
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{path} must list at least one model")
    specs = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or "type" not in entry:
            raise ConfigError(f"model '{name}' in {path} needs a 'type'")
        unknown = set(entry) - {"type", "params", "features", "skip_without_static"}
        if unknown:
            raise ConfigError(f"model '{name}': unknown key(s) {sorted(unknown)}")
        if entry["type"] not in MODEL_TYPES:
            raise ConfigError(
                f"model '{name}': unknown type '{entry['type']}'. Known types: {sorted(MODEL_TYPES)}"
            )
        params = entry.get("params") or {}
        if not isinstance(params, dict):
            raise ConfigError(f"model '{name}': params must be a mapping")
        features = entry.get("features")
        if features is not None and not isinstance(features, str):
            raise ConfigError(f"model '{name}': features must be the name of a feature set")
        skip = entry.get("skip_without_static", False)
        if not isinstance(skip, bool):
            raise ConfigError(f"model '{name}': skip_without_static must be true or false")
        specs[str(name)] = {
            "type": entry["type"], "params": params, "features": features, "skip_without_static": skip,
        }
    if types is not None:
        wanted = set(types)
        specs = {name: spec for name, spec in specs.items() if spec["type"] in wanted}
    return specs


def build_models(
    specs: dict[str, dict[str, Any]],
    season_length: int,
    feature_builders: Mapping[str, FeatureBuilder] | None = None,
    *,
    pandas_freq: str | None = None,
    fast_only: bool = True,
    known_columns: Sequence[str] = (),
    explanatory_columns: Sequence[str] = (),
) -> dict[str, BaseForecaster]:
    """Create the candidate models for one frequency.

    `season_length` comes from the frequency profile and is passed to models that take it.
    Models that need a seasonal period are left out when the frequency has none (< 2).
    `pandas_freq` is passed to models that take a `freq`, `known_columns` (the explanatory columns
    known in the future) to models that take `known_columns`, and `explanatory_columns` (all of them) to models that take that. With `fast_only`, models marked slow
    are left out. A model whose library is not installed is left out with a warning.
    """
    models: dict[str, BaseForecaster] = {}
    for name, spec in specs.items():
        cls = MODEL_TYPES[spec["type"]]
        if cls.requires_seasonality and season_length < 2:
            logger.info("model '%s' skipped: the frequency has no seasonal period", name)
            continue
        if fast_only and cls.cost == "slow":
            logger.info("model '%s' skipped: it is slow and fast_only is on", name)
            continue
        missing = [pkg for pkg in cls.requires_packages if not _package_available(pkg)]
        if missing:
            hint = f" Install it with `poetry install --extras {cls.install_extra}`." if cls.install_extra else ""
            logger.warning("model '%s' skipped: package %s is not installed.%s", name, ", ".join(missing), hint)
            continue
        params = dict(spec["params"])
        feature_set = spec.get("features")
        if feature_set is not None:
            if "feature_builder" not in inspect.signature(cls.__init__).parameters:
                raise ConfigError(
                    f"model '{name}': type '{spec['type']}' does not use feature sets, remove `features`"
                )
            if feature_builders is None or feature_set not in feature_builders:
                raise ConfigError(f"model '{name}': no feature builder for set '{feature_set}'")
            params["feature_builder"] = feature_builders[feature_set]
        init_parameters = inspect.signature(cls.__init__).parameters
        if "season_length" in init_parameters:
            params.setdefault("season_length", season_length)
        if "freq" in init_parameters:
            if pandas_freq is None and "freq" not in params:
                raise ConfigError(f"model '{name}' needs the pandas frequency; pass pandas_freq")
            if pandas_freq is not None:
                params.setdefault("freq", pandas_freq)
        if "known_columns" in init_parameters:
            params.setdefault("known_columns", tuple(known_columns))
        if "explanatory_columns" in init_parameters:
            params.setdefault("explanatory_columns", tuple(explanatory_columns))
        try:
            models[name] = cls(**params)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"model '{name}': invalid params {params}: {exc}") from exc
    return models


REFERENCE_MODEL = "naive"  # the benchmark every other model is compared with


def filter_specs(
    specs: dict[str, dict[str, Any]],
    include: Sequence[str] | None,
    exclude: Sequence[str],
    protected: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Keep the models named in `include` (all if empty) and drop those in `exclude`.

    `protected` models (the reference model and the short-series models) always stay: the
    selection needs them. Naming an unknown model, or excluding a protected one, is an error."""
    protected = {name for name in protected if name in specs}
    unknown = sorted((set(include or ()) | set(exclude)) - set(specs))
    if unknown:
        raise ConfigError(f"unknown model name(s) {unknown}. Defined in models.yaml: {sorted(specs)}")
    clash = sorted(set(exclude) & protected)
    if clash:
        raise ConfigError(
            f"cannot exclude {clash}: the selection needs them (naive and the short_series_models)"
        )
    keep = set(specs) if not include else set(include) | protected
    extra = sorted(keep - set(include or keep))
    if include and extra:
        logger.info("models added because the selection needs them: %s", extra)
    return {name: spec for name, spec in specs.items() if name in keep and name not in exclude}


def build_models_from_config(
    cfg, config_dir: Path | None = None, static=None, apply_model_filter: bool = True
) -> dict[str, BaseForecaster]:
    """The candidate models for a run: models.yaml plus the feature sets they use.

    `static` is the table of static values per series (PreparedInput.static), needed only by
    feature sets that use identifier features. `cfg.models` / `cfg.exclude_models` restrict the
    candidates unless `apply_model_filter` is false."""
    specs = load_model_specs(config_dir)
    if apply_model_filter and (cfg.models or cfg.exclude_models):
        specs = filter_specs(
            specs, cfg.models, cfg.exclude_models, {REFERENCE_MODEL, *cfg.short_series_models}
        )
    feature_config = load_feature_config(config_dir)
    has_static = static is not None and any(c != "unique_id" for c in static.columns)
    if not has_static:
        for name, spec in list(specs.items()):
            feature_set = feature_config.sets.get(spec.get("features") or "")
            if spec.get("skip_without_static") and feature_set is not None and not feature_set.static.is_empty:
                logger.warning(
                    "model '%s' skipped: its feature set '%s' uses static features, but the input "
                    "has no static_columns", name, spec["features"],
                )
                del specs[name]
    wanted = [spec["features"] for spec in specs.values() if spec.get("features")]
    builders = None
    if wanted:
        try:
            builders = build_feature_builders(
                feature_config, wanted,
                pandas_freq=cfg.profile.pandas_freq, season_length=cfg.profile.season_length,
                static=static,
            )
        except FeatureError as exc:
            raise ConfigError(str(exc)) from exc
    return build_models(
        specs, cfg.profile.season_length, builders,
        pandas_freq=cfg.profile.pandas_freq, fast_only=cfg.fast_only,
        known_columns=cfg.known_future_columns, explanatory_columns=cfg.explanatory_columns,
    )
