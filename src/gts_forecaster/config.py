"""Run configuration: loads config/default.yaml and config/frequency_profiles.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gts_forecaster.cleaning.outliers import OUTLIER_ACTIONS, OUTLIER_METHODS
from gts_forecaster.columns import TARGET
from gts_forecaster.evaluation.metrics import METRICS

FALLBACKS = ("last_value", "zero")
RESERVED_COLUMNS = ("uuid", "date", "target", "unique_id", "ds", "y", "is_filled")
ENSEMBLE_WEIGHTS = ("equal", "inverse_error")
ON_MODEL_ERROR = ("skip", "raise")
FORECAST_MODES = ("frozen_selection", "frozen_models")
FORECAST_CHECK_ACTIONS = ("warn", "raise", "off")
_ALLOWED_KEYS = {
    "frequency",
    "horizon",
    "data_path",
    "explanatory_columns",
    "max_fill_periods",
    "fallback",
    "metric",
    "n_folds",
    "backtest_step",
    "top_n",
    "ensemble_weights",
    "short_series_models",
    "short_series_n",
    "registry_dir",
    "backtesting_dir",
    "outlier_method",
    "outlier_threshold",
    "outlier_max_share",
    "outlier_action",
    "known_future_columns",
    "future_features_path",
    "static_columns",
    "fast_only",
    "models",
    "exclude_models",
    "max_series",
    "on_model_error",
    "forecast_mode",
    "fitted_models_dir",
    "output_file",
    "output_rename",
    "forecast_floor",
    "runs_dir",
    "forecast_check_action",
    "forecast_check_threshold",
    "forecast_check_max_ratio",
}
_OPTIONAL_KEYS = {
    "horizon", "backtest_step", "known_future_columns", "future_features_path", "static_columns",
    "fast_only", "models", "exclude_models", "max_series", "on_model_error",
    "forecast_mode", "fitted_models_dir", "output_file", "output_rename",
    "forecast_floor", "runs_dir",
    "forecast_check_action", "forecast_check_threshold", "forecast_check_max_ratio",
}


class ConfigError(ValueError):
    """The configuration is invalid."""


@dataclass(frozen=True)
class FrequencyProfile:
    name: str
    pandas_freq: str
    min_series_length: int
    default_horizon: int
    season_length: int


@dataclass(frozen=True)
class RunConfig:
    frequency: str
    horizon: int
    profile: FrequencyProfile
    data_path: Path
    explanatory_columns: tuple[str, ...]
    max_fill_periods: int
    fallback: str
    metric: str
    n_folds: int
    backtest_step: int | None  # None = use the horizon
    top_n: int
    ensemble_weights: str
    short_series_models: tuple[str, ...]
    short_series_n: int
    registry_dir: Path
    backtesting_dir: Path  # forecasts vs actuals of every backtest window
    outlier_method: str = "mad"  # mad | none
    outlier_threshold: float = 5.0  # robust standard deviations
    outlier_action: str = "clip"  # clip | interpolate | median | mean
    outlier_max_share: float = 0.1  # leave a series alone if more of its points would be flagged
    # Explanatory columns whose future values the user supplies. Default: none are known.
    known_future_columns: tuple[str, ...] = ()
    future_features_path: Path | None = None  # CSV with the future values of those columns
    # Input columns that identify a series (market, product category, company ...): one text value
    # per uuid. They are kept apart from the time series and can be turned into features.
    static_columns: tuple[str, ...] = ()
    # True: models marked "slow" (Prophet, Orbit) are neither backtested nor run.
    fast_only: bool = True
    # Run controls. `models` / `exclude_models` pick candidate models by name (the baselines that the
    # selection needs always run); `max_series` runs a spread-out sample of that many series.
    models: tuple[str, ...] | None = None
    exclude_models: tuple[str, ...] = ()
    max_series: int | None = None
    on_model_error: str = "skip"  # skip: a model that fails is left out and reported; raise: stop
    # frozen_selection: refit the selected models on the newest data at every forecast.
    # frozen_models: use the models saved by `train`, without refitting (see pipelines/train.py).
    forecast_mode: str = "frozen_selection"
    fitted_models_dir: Path = Path("fitted_models")  # <fitted_models_dir>/<frequency>/
    # Final file for the receiving platform (long format), written to outputs/<frequency>/.
    output_file: str = "forecast_output.csv"
    # Optional new names for the output columns: {uuid: unique_id, date: ds, market: market_code}.
    output_rename: tuple[tuple[str, str], ...] = ()
    # Lowest allowed forecast (volumes cannot be negative). Applied to every model's predictions in
    # the backtest and the forecast, and to the final forecast. None = no floor.
    forecast_floor: float | None = 0.0
    # `run-pipeline` keeps one folder per run here: <runs_dir>/<frequency>/<run id>/
    runs_dir: Path = Path("runs")
    # Sanity checks of the final forecasts (forecast_checks.py): warn | raise | off. The outlier
    # check uses forecast_check_threshold robust standard deviations (None: outlier_threshold); the
    # total_change and start_jump checks flag changes beyond forecast_check_max_ratio times.
    forecast_check_action: str = "warn"
    forecast_check_threshold: float = 5.0
    forecast_check_max_ratio: float = 3.0

    @property
    def is_partial_run(self) -> bool:
        """True when only some models or series run, so the results must not replace a full run."""
        return bool(self.models) or bool(self.exclude_models) or self.max_series is not None

    @property
    def value_columns(self) -> list[str]:
        """Target (`y`) followed by the explanatory columns, as named in the model format."""
        return [TARGET, *self.explanatory_columns]


def project_root() -> Path:
    """Project root: the folder that contains config/, data/ and src/."""
    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ConfigError(f"Config file must contain a mapping at the top level: {path}")
    return content


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_profiles(config_dir: Path | None = None) -> dict[str, FrequencyProfile]:
    """Read config/frequency_profiles.yaml."""
    config_dir = config_dir or project_root() / "config"
    raw = _read_yaml(config_dir / "frequency_profiles.yaml")
    profiles = {}
    for name, entry in raw.items():
        try:
            profile = FrequencyProfile(
                name=name,
                pandas_freq=entry["pandas_freq"],
                min_series_length=entry["min_series_length"],
                default_horizon=entry["default_horizon"],
                season_length=entry["season_length"],
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"Frequency profile '{name}' is incomplete or malformed: {exc}") from exc
        if not isinstance(profile.pandas_freq, str) or not profile.pandas_freq:
            raise ConfigError(f"Frequency profile '{name}': pandas_freq must be a non-empty string")
        for key in ("min_series_length", "default_horizon", "season_length"):
            if not _is_int(getattr(profile, key)) or getattr(profile, key) < 1:
                raise ConfigError(f"Frequency profile '{name}': {key} must be an integer >= 1")
        profiles[name] = profile
    return profiles


def load_config(
    overrides: dict[str, Any] | None = None, config_dir: Path | None = None
) -> RunConfig:
    """Build the run configuration.

    `overrides` replaces values from default.yaml. A value of None in `overrides` means
    "not given" and is ignored (so command line defaults do not wipe out the config).
    """
    config_dir = Path(config_dir) if config_dir else project_root() / "config"
    root = config_dir.parent
    raw = _read_yaml(config_dir / "default.yaml")
    for key, value in (overrides or {}).items():
        if value is not None:
            raw[key] = value

    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ConfigError(f"Unknown config key(s): {sorted(unknown)}")
    missing = _ALLOWED_KEYS - set(raw) - _OPTIONAL_KEYS
    if missing:
        raise ConfigError(f"Missing config key(s): {sorted(missing)}")

    profiles = load_profiles(config_dir)
    frequency = raw["frequency"]
    if frequency not in profiles:
        raise ConfigError(f"Unknown frequency '{frequency}'. Choose one of {sorted(profiles)}")
    profile = profiles[frequency]

    horizon = raw.get("horizon")
    if horizon is None:
        horizon = profile.default_horizon
    if not _is_int(horizon) or horizon < 1:
        raise ConfigError(f"horizon must be an integer >= 1, got {horizon!r}")

    max_fill = raw["max_fill_periods"]
    if not _is_int(max_fill) or max_fill < 0:
        raise ConfigError(f"max_fill_periods must be an integer >= 0, got {max_fill!r}")

    fallback = raw["fallback"]
    if fallback not in FALLBACKS:
        raise ConfigError(f"fallback must be one of {list(FALLBACKS)}, got {fallback!r}")

    columns = raw["explanatory_columns"]
    if not isinstance(columns, (list, tuple)) or not all(isinstance(c, str) and c for c in columns):
        raise ConfigError("explanatory_columns must be a list of non-empty strings")
    if len(set(columns)) != len(columns):
        raise ConfigError("explanatory_columns contains duplicates")
    clash = set(columns) & set(RESERVED_COLUMNS)
    if clash:
        raise ConfigError(f"explanatory_columns cannot use reserved names: {sorted(clash)}")

    metric = str(raw["metric"]).lower()
    if metric not in METRICS:
        raise ConfigError(f"metric must be one of {sorted(METRICS)}, got {raw['metric']!r}")

    n_folds = raw["n_folds"]
    if not _is_int(n_folds) or n_folds < 1:
        raise ConfigError(f"n_folds must be an integer >= 1, got {n_folds!r}")
    step = raw.get("backtest_step")
    if step is not None and (not _is_int(step) or step < 1):
        raise ConfigError(f"backtest_step must be null or an integer >= 1, got {step!r}")
    top_n = raw["top_n"]
    if not _is_int(top_n) or top_n < 1:
        raise ConfigError(f"top_n must be an integer >= 1, got {top_n!r}")
    weights = raw["ensemble_weights"]
    if weights not in ENSEMBLE_WEIGHTS:
        raise ConfigError(f"ensemble_weights must be one of {list(ENSEMBLE_WEIGHTS)}, got {weights!r}")
    short_models = raw["short_series_models"]
    if (
        not isinstance(short_models, (list, tuple))
        or not short_models
        or not all(isinstance(m, str) and m for m in short_models)
        or len(set(short_models)) != len(short_models)
    ):
        raise ConfigError("short_series_models must be a non-empty list of distinct model names")
    short_n = raw["short_series_n"]
    if not _is_int(short_n) or short_n < 1:
        raise ConfigError(f"short_series_n must be an integer >= 1, got {short_n!r}")

    outlier_method = raw["outlier_method"]
    if outlier_method not in OUTLIER_METHODS:
        raise ConfigError(f"outlier_method must be one of {list(OUTLIER_METHODS)}, got {outlier_method!r}")
    outlier_action = raw["outlier_action"]
    if outlier_action not in OUTLIER_ACTIONS:
        raise ConfigError(f"outlier_action must be one of {list(OUTLIER_ACTIONS)}, got {outlier_action!r}")
    threshold = raw["outlier_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not threshold > 0:
        raise ConfigError(f"outlier_threshold must be a number > 0, got {threshold!r}")
    max_share = raw["outlier_max_share"]
    if isinstance(max_share, bool) or not isinstance(max_share, (int, float)) or not 0 < max_share <= 1:
        raise ConfigError(f"outlier_max_share must be a number in (0, 1], got {max_share!r}")

    static = raw.get("static_columns")
    static = [] if static is None else static
    if not isinstance(static, (list, tuple)) or not all(isinstance(c, str) and c for c in static):
        raise ConfigError("static_columns must be a list of column names")
    if len(set(static)) != len(static):
        raise ConfigError("static_columns contains duplicates")
    bad_static = sorted((set(static) & set(RESERVED_COLUMNS)) | (set(static) & set(columns)))
    if bad_static:
        raise ConfigError(
            f"static_columns {bad_static} clash with reserved names or explanatory_columns"
        )

    fast_only = raw.get("fast_only", True)
    if not isinstance(fast_only, bool):
        raise ConfigError(f"fast_only must be true or false, got {fast_only!r}")

    names = {}
    for key in ("models", "exclude_models"):
        value = raw.get(key)
        if value is not None and (
            not isinstance(value, (list, tuple)) or not all(isinstance(n, str) and n for n in value)
        ):
            raise ConfigError(f"{key} must be a list of model names")
        names[key] = value
    max_series = raw.get("max_series")
    if max_series is not None and (not _is_int(max_series) or max_series < 1):
        raise ConfigError(f"max_series must be null or an integer >= 1, got {max_series!r}")
    on_error = raw.get("on_model_error", "skip")
    if on_error not in ON_MODEL_ERROR:
        raise ConfigError(f"on_model_error must be one of {list(ON_MODEL_ERROR)}, got {on_error!r}")

    output_file = raw.get("output_file") or "forecast_output.csv"
    if not isinstance(output_file, str) or Path(output_file).name != output_file or not output_file.endswith(".csv"):
        raise ConfigError(f"output_file must be a plain .csv file name, got {output_file!r}")
    output_rename = raw.get("output_rename") or {}
    if not isinstance(output_rename, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v for k, v in output_rename.items()
    ):
        raise ConfigError("output_rename must map output column names to new names")
    final_names = [output_rename.get(c, c) for c in ["uuid", "date", *(raw.get("static_columns") or []), "forecast"]]
    unknown_keys = sorted(set(output_rename) - {"uuid", "date", "forecast", *(raw.get("static_columns") or [])})
    if unknown_keys:
        raise ConfigError(f"output_rename names unknown output column(s) {unknown_keys}")
    if len(set(final_names)) != len(final_names):
        raise ConfigError(f"output_rename gives duplicate column names: {final_names}")

    forecast_floor = raw.get("forecast_floor", 0.0)
    if forecast_floor is not None and (
        isinstance(forecast_floor, bool) or not isinstance(forecast_floor, (int, float))
    ):
        raise ConfigError(f"forecast_floor must be a number or null, got {forecast_floor!r}")

    check_action = raw.get("forecast_check_action", "warn")
    if check_action not in FORECAST_CHECK_ACTIONS:
        raise ConfigError(
            f"forecast_check_action must be one of {list(FORECAST_CHECK_ACTIONS)}, got {check_action!r}"
        )
    check_threshold = raw.get("forecast_check_threshold")
    check_threshold = threshold if check_threshold is None else check_threshold
    if isinstance(check_threshold, bool) or not isinstance(check_threshold, (int, float)) or not check_threshold > 0:
        raise ConfigError(f"forecast_check_threshold must be a number > 0 or null, got {check_threshold!r}")
    check_ratio = raw.get("forecast_check_max_ratio", 3.0)
    if isinstance(check_ratio, bool) or not isinstance(check_ratio, (int, float)) or not check_ratio > 1:
        raise ConfigError(f"forecast_check_max_ratio must be a number > 1, got {check_ratio!r}")

    forecast_mode = raw.get("forecast_mode", "frozen_selection")
    if forecast_mode not in FORECAST_MODES:
        raise ConfigError(f"forecast_mode must be one of {list(FORECAST_MODES)}, got {forecast_mode!r}")

    known = raw.get("known_future_columns")
    known = [] if known is None else known
    if not isinstance(known, (list, tuple)) or not all(isinstance(c, str) and c for c in known):
        raise ConfigError("known_future_columns must be a list of column names")
    if len(set(known)) != len(known):
        raise ConfigError("known_future_columns contains duplicates")
    not_explanatory = [c for c in known if c not in columns]
    if not_explanatory:
        raise ConfigError(
            f"known_future_columns {not_explanatory} are not in explanatory_columns {list(columns)}"
        )
    future_path = raw.get("future_features_path")
    if future_path is not None:
        if not known:
            raise ConfigError("future_features_path was given but known_future_columns is empty")
        future_path = Path(str(future_path).format(frequency=frequency))
        if not future_path.is_absolute():
            future_path = root / future_path

    data_path = Path(str(raw["data_path"]).format(frequency=frequency))
    if not data_path.is_absolute():
        data_path = root / data_path
    registry_dir = Path(str(raw["registry_dir"]))
    if not registry_dir.is_absolute():
        registry_dir = root / registry_dir
    backtesting_dir = Path(str(raw["backtesting_dir"]))
    if not backtesting_dir.is_absolute():
        backtesting_dir = root / backtesting_dir
    runs_dir = Path(str(raw.get("runs_dir") or "runs"))
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir
    fitted_models_dir = Path(str(raw.get("fitted_models_dir") or "fitted_models"))
    if not fitted_models_dir.is_absolute():
        fitted_models_dir = root / fitted_models_dir

    return RunConfig(
        frequency=frequency,
        horizon=horizon,
        profile=profile,
        data_path=data_path,
        explanatory_columns=tuple(columns),
        max_fill_periods=max_fill,
        fallback=fallback,
        metric=metric,
        n_folds=n_folds,
        backtest_step=step,
        top_n=top_n,
        ensemble_weights=weights,
        short_series_models=tuple(short_models),
        short_series_n=short_n,
        registry_dir=registry_dir,
        backtesting_dir=backtesting_dir,
        outlier_method=outlier_method,
        outlier_threshold=float(threshold),
        outlier_action=outlier_action,
        outlier_max_share=float(max_share),
        static_columns=tuple(static),
        known_future_columns=tuple(known),
        future_features_path=future_path,
        fast_only=fast_only,
        models=tuple(names['models']) if names['models'] else None,
        exclude_models=tuple(names['exclude_models'] or ()),
        max_series=max_series,
        on_model_error=on_error,
        forecast_mode=forecast_mode,
        fitted_models_dir=fitted_models_dir,
        output_file=output_file,
        output_rename=tuple(output_rename.items()),
        forecast_floor=None if forecast_floor is None else float(forecast_floor),
        runs_dir=runs_dir,
        forecast_check_action=check_action,
        forecast_check_threshold=float(check_threshold),
        forecast_check_max_ratio=float(check_ratio),
    )
