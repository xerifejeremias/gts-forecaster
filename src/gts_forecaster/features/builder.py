"""Feature sets and the FeatureBuilder that turns them into columns.

A feature set is a named recipe in config/features.yaml (calendar features, lags, rolling
statistics). A model asks for one in config/models.yaml with `features: <set name>`. Models that
generate their own features (Nixtla's MLForecast and StatsForecast, Prophet, Orbit) simply do
not ask for any.

    builder = FeatureBuilder(feature_set, pandas_freq="MS", season_length=12)
    train_x = builder.transform_history(train)           # unique_id, ds, y, ... + feature columns
    future_x = builder.transform_future(train, future)   # unique_id, ds, horizon_step + features

Explanatory columns (known future values and so on) are passed through unchanged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.features import static as stc
from gts_forecaster.features import target as tgt
from gts_forecaster.features.calendar_features import calendar_features, frequency_kind
from gts_forecaster.features.errors import FeatureError

logger = logging.getLogger(__name__)

_SET_KEYS = {
    "calendar", "fourier_order", "lags", "rolling_windows", "rolling_stats", "rolling_lag", "static",
}
_STATIC_KEYS = {
    "identifiers", "code_parts", "encoding", "group_stats", "series_stats",
}
_SEASON = re.compile(r"^(?:(\d+)\s*\*\s*)?season$")


@dataclass(frozen=True)
class FeatureSet:
    """A recipe. `lags` and `rolling_windows` may contain integers or the text `season`
    (one season length) or `2*season`, `3*season`, ... which are resolved per frequency."""

    name: str
    calendar: str | tuple[str, ...] = ()  # "all", a list of names, or empty for none
    fourier_order: int = 0
    lags: tuple[int | str, ...] = ()
    rolling_windows: tuple[int | str, ...] = ()
    rolling_stats: tuple[str, ...] = ("mean", "std")
    rolling_lag: int = 1
    static: stc.StaticSpec = stc.StaticSpec()  # features that identify the series (market, company ...)


@dataclass(frozen=True)
class CalendarSettings:
    country: str | None = None  # holidays package country code, for example "PT" or "US"
    extra_holidays: tuple[str, ...] = ()  # additional YYYY-MM-DD dates
    anchor: str = "start"  # does the date mark the start or the end of its period?


@dataclass(frozen=True)
class FeatureConfig:
    sets: Mapping[str, FeatureSet] = field(default_factory=dict)
    calendar: CalendarSettings = CalendarSettings()


def _resolve(values: Sequence[int | str], season_length: int, what: str) -> tuple[int, ...]:
    out: list[int] = []
    for v in values:
        if isinstance(v, str):
            match = _SEASON.match(v.strip().lower())
            if not match:
                raise FeatureError(f"{what}: {v!r} is not an integer, 'season' or 'N*season'")
            v = int(match.group(1) or 1) * season_length
        if v not in out:
            out.append(v)
    return tuple(out)


def holiday_dates(settings: CalendarSettings, years: Sequence[int]) -> list[pd.Timestamp]:
    """Holiday dates for the given years: the country's public holidays plus the extra dates."""
    dates = [pd.Timestamp(d) for d in settings.extra_holidays]
    if settings.country:
        try:
            import holidays as holidays_package
        except ImportError as exc:
            raise FeatureError(
                "holidays.country is set but the 'holidays' package is not installed. "
                "Install it with: poetry install --extras holidays"
            ) from exc
        calendar = holidays_package.country_holidays(settings.country, years=list(years))
        dates += [pd.Timestamp(d) for d in calendar]
    return sorted(set(dates))


class FeatureBuilder:
    """Creates the feature columns of a FeatureSet for one frequency."""

    def __init__(
        self,
        feature_set: FeatureSet,
        *,
        pandas_freq: str,
        season_length: int = 1,
        calendar: CalendarSettings = CalendarSettings(),
        static: pd.DataFrame | None = None,
    ):
        self.feature_set = feature_set
        self.static = static
        self.pandas_freq = pandas_freq
        self.season_length = season_length
        self.calendar_settings = calendar
        self.kind = frequency_kind(pandas_freq)

        self.lags = _resolve(feature_set.lags, season_length, f"{feature_set.name}: lags")
        windows = _resolve(feature_set.rolling_windows, season_length, f"{feature_set.name}: windows")
        dropped = [w for w in windows if w < 2]
        if dropped:
            logger.info("feature set '%s': rolling window(s) %s dropped (need >= 2)", feature_set.name, dropped)
        self.windows = tuple(w for w in windows if w >= 2)
        self.stats = tuple(feature_set.rolling_stats)
        self.rolling_lag = feature_set.rolling_lag
        tgt.check_target_spec(self.lags, self.windows, self.stats, self.rolling_lag)

        spec = feature_set.static
        self._static_names: list[str] = []
        if not spec.is_empty:
            if static is None or static.empty:
                raise FeatureError(
                    f"feature set '{feature_set.name}' uses static features, but the input has no "
                    "static columns. List them in `static_columns` in config/default.yaml."
                )
            self._static_names = stc.feature_names(spec, stc.check_spec(spec, static))

    # ---- columns ------------------------------------------------------------------------------

    @property
    def target_columns(self) -> list[str]:
        return tgt.target_feature_names(self.lags, self.windows, self.stats)

    def calendar_columns(self, dates: pd.Series | None = None) -> list[str]:
        """Names of the calendar columns (what `transform_*` will add)."""
        probe = pd.DataFrame({DATE: pd.to_datetime(["2000-01-03"])})
        return list(self._calendar(probe, years=[2000]).columns)

    @property
    def static_columns(self) -> list[str]:
        return list(self._static_names)

    @property
    def feature_columns(self) -> list[str]:
        return self.calendar_columns() + self.static_columns + self.target_columns

    def usable_at_step(self, step: int) -> list[str]:
        """Feature columns whose values are known at forecast step `step` (calendar always)."""
        return self.calendar_columns() + self.static_columns + tgt.usable_at_step(
            self.lags, self.windows, self.stats, self.rolling_lag, step
        )

    # ---- transforms ---------------------------------------------------------------------------

    def _calendar(self, df: pd.DataFrame, years: Sequence[int] | None = None) -> pd.DataFrame:
        spec = self.feature_set
        if not spec.calendar and not spec.fourier_order:
            return pd.DataFrame(index=df.index)
        dates = pd.to_datetime(df[DATE])
        if years is None:
            years = range(int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 2)
        holidays = holiday_dates(self.calendar_settings, list(years))
        names: Any = spec.calendar if spec.calendar else ()
        if names == "all":
            table = calendar_features(
                df, pandas_freq=self.pandas_freq, names="all",
                fourier_order=spec.fourier_order, holidays=holidays,
                anchor=self.calendar_settings.anchor,
            )
        else:
            wanted = list(names)
            table = calendar_features(
                df, pandas_freq=self.pandas_freq, names=wanted,
                fourier_order=spec.fourier_order, holidays=holidays,
                anchor=self.calendar_settings.anchor,
            )
        return table

    def _check_collisions(self, df: pd.DataFrame) -> None:
        clash = sorted(set(df.columns) & set(self.feature_columns))
        if clash:
            raise FeatureError(
                f"feature set '{self.feature_set.name}' would overwrite existing column(s) {clash}"
            )

    def _static_features(self, frame: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """Static feature values for the rows of `frame`, computed from `train`."""
        if not self._static_names:
            return pd.DataFrame(index=frame.index)
        table = stc.static_feature_table(train, self.static, self.feature_set.static, self.season_length)
        missing = sorted(set(frame[ID]) - set(table.index))
        if missing:
            raise FeatureError(f"no static features for series {missing[:5]}: they are not in the training data")
        return table.reindex(frame[ID]).set_axis(frame.index)

    def transform_history(self, train: pd.DataFrame) -> pd.DataFrame:
        """`train` with the feature columns added (same rows and order)."""
        self._check_collisions(train)
        parts = [train, self._calendar(train), self._static_features(train, train)]
        if self.target_columns:
            parts.append(tgt.history_target_features(train, self.lags, self.windows, self.stats, self.rolling_lag))
        return pd.concat(parts, axis=1)

    def transform_future(self, train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """`future` (unique_id, ds, horizon_step, ...) with the feature columns added.

        Target features use `train` only; those that would need values from inside the forecast
        window are NaN (see `usable_at_step`).
        """
        self._check_collisions(future)
        parts = [future, self._calendar(future), self._static_features(future, train)]
        if self.target_columns:
            parts.append(
                tgt.future_target_features(train, future, self.lags, self.windows, self.stats, self.rolling_lag)
            )
        return pd.concat(parts, axis=1)


# ---- config file ------------------------------------------------------------------------------


def _config_dir(config_dir: Path | None) -> Path:
    from gts_forecaster.config import project_root

    return Path(config_dir) if config_dir else project_root() / "config"


def _parse_static(set_name: str, raw: Any) -> stc.StaticSpec:
    if not raw:
        return stc.StaticSpec()
    if not isinstance(raw, dict) or set(raw) - _STATIC_KEYS:
        extra = sorted(set(raw) - _STATIC_KEYS) if isinstance(raw, dict) else raw
        raise FeatureError(f"feature set '{set_name}': unknown or invalid static section {extra}")
    groups_raw = raw.get("group_stats") or {}
    if set(groups_raw) - {"by", "stats"}:
        raise FeatureError(f"feature set '{set_name}': group_stats takes only 'by' and 'stats'")
    group_by = []
    for entry in groups_raw.get("by") or []:
        columns = tuple([entry] if isinstance(entry, str) else entry)
        if not columns or not all(isinstance(c, str) for c in columns):
            raise FeatureError(f"feature set '{set_name}': each group in 'by' is a column name or a list of them")
        group_by.append(columns)
    code_parts = {}
    for column, lengths in (raw.get("code_parts") or {}).items():
        if not isinstance(lengths, (list, tuple)):
            raise FeatureError(f"feature set '{set_name}': code_parts.{column} must be a list of lengths")
        code_parts[str(column)] = tuple(lengths)
    return stc.StaticSpec(
        identifiers=tuple(raw.get("identifiers") or ()),
        code_parts=code_parts,
        encoding=raw.get("encoding", "ordinal"),
        group_by=tuple(group_by),
        group_stats=tuple(groups_raw.get("stats") or ("mean",)),
        series_stats=tuple(raw.get("series_stats") or ()),
    )


def load_feature_config(config_dir: Path | None = None) -> FeatureConfig:
    """Read config/features.yaml. A missing file means no feature sets."""
    path = _config_dir(config_dir) / "features.yaml"
    if not path.is_file():
        return FeatureConfig()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict) or set(raw) - {"calendar", "sets"}:
        raise FeatureError(f"{path}: top-level keys must be 'calendar' and 'sets'")
    cal_raw = raw.get("calendar") or {}
    if set(cal_raw) - {"holiday_country", "extra_holidays", "period_anchor"}:
        raise FeatureError(f"{path}: unknown key(s) in 'calendar': {sorted(set(cal_raw) - {'holiday_country', 'extra_holidays', 'period_anchor'})}")
    extra = tuple(str(d) for d in (cal_raw.get("extra_holidays") or []))
    try:
        for d in extra:
            pd.Timestamp(d)
    except ValueError as exc:
        raise FeatureError(f"{path}: extra_holidays must be YYYY-MM-DD dates: {exc}") from exc
    anchor = cal_raw.get("period_anchor", "start")
    if anchor not in ("start", "end"):
        raise FeatureError(f"{path}: calendar.period_anchor must be 'start' or 'end'")
    settings = CalendarSettings(cal_raw.get("holiday_country"), extra, anchor)

    sets = {}
    for name, entry in (raw.get("sets") or {}).items():
        entry = entry or {}
        unknown = set(entry) - _SET_KEYS
        if unknown:
            raise FeatureError(f"feature set '{name}': unknown key(s) {sorted(unknown)}")
        calendar = entry.get("calendar") or ()
        if calendar != "all":
            if not isinstance(calendar, (list, tuple)):
                raise FeatureError(f"feature set '{name}': calendar must be 'all' or a list of names")
            calendar = tuple(calendar)
        sets[str(name)] = FeatureSet(
            name=str(name),
            calendar=calendar,
            fourier_order=int(entry.get("fourier_order", 0)),
            lags=tuple(entry.get("lags") or ()),
            rolling_windows=tuple(entry.get("rolling_windows") or ()),
            rolling_stats=tuple(entry.get("rolling_stats") or ("mean", "std")),
            rolling_lag=entry.get("rolling_lag", 1),
            static=_parse_static(str(name), entry.get("static")),
        )
    return FeatureConfig(sets, settings)


def build_feature_builders(
    config: FeatureConfig,
    names: Sequence[str],
    *,
    pandas_freq: str,
    season_length: int,
    static: pd.DataFrame | None = None,
) -> dict[str, FeatureBuilder]:
    """A FeatureBuilder for each requested feature set name."""
    builders = {}
    for name in dict.fromkeys(names):
        if name not in config.sets:
            raise FeatureError(
                f"feature set '{name}' is not defined in features.yaml. Defined: {sorted(config.sets)}"
            )
        builders[name] = FeatureBuilder(
            config.sets[name], pandas_freq=pandas_freq, season_length=season_length,
            calendar=config.calendar, static=static,
        )
    return builders


def resolve_lengths(values: Sequence[int | str], season_length: int, what: str) -> tuple[int, ...]:
    """Lags / window sizes given as integers or `season`, `2*season`, ... as whole numbers."""
    return _resolve(values, season_length, what)
