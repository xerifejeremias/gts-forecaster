"""Static features: describe WHICH series a row belongs to, so a global model can learn patterns
that are shared by all series of a market, a product category, a company and so on.

The input has "static columns" (config `static_columns`), for example market `DE04`, product
category `F0101`, company `company A`: one text value per series. From them a feature set can ask for

  identifiers   the column itself, encoded for models (`encoding: ordinal` gives one integer code
                per category, `onehot` gives one 0/1 column per category). Columns: `id_<column>`
                or `<column>_<category>`.
  code_parts    pieces of structured codes as new static columns: `market: [2]` turns "DE04"
                into `market_p2` = "DE"; `product: [1, 3]` turns "F0101" into `product_p1` = "F"
                and `product_p3` = "F01". The new columns can be used as identifiers or groups.
  group_stats   statistics of the target for a group of series that share the same value, for
                example every series of one market, or every series of the same market AND
                product category. For group G and series i (mu_i = mean of y, sigma_i = std,
                g_i = relative trend per period):
                  grp_<G>_mean      average of mu_i over the series of the group
                  grp_<G>_median    median of mu_i
                  grp_<G>_std       average of sigma_i
                  grp_<G>_n_series  number of series in the group
                  grp_<G>_rel_level mu_i / group mean: how large this series is within its group
                  grp_<G>_growth    average of g_i: how fast the group is growing
                Each series counts once, so long series do not dominate a group.
  series_stats  descriptors of the series itself: ser_mean, ser_std, ser_cv, ser_length,
                ser_trend (relative slope per period), ser_zero_share, ser_season_strength
                (share of the variance explained by the position in the season).

All statistics are computed from the training data passed in, so inside a backtest fold they only
use that fold's history. They are constant per series. Note that, like any target encoding, a
statistic of a group with very few series is close to the series' own history; groups with several
series generalise better.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, TARGET
from gts_forecaster.features.errors import FeatureError

ENCODINGS = ("ordinal", "onehot")
GROUP_STATS = ("mean", "median", "std", "n_series", "rel_level", "growth")
SERIES_STATS = ("mean", "std", "cv", "length", "trend", "zero_share", "season_strength")


@dataclass(frozen=True)
class StaticSpec:
    identifiers: tuple[str, ...] = ()
    code_parts: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    encoding: str = "ordinal"
    group_by: tuple[tuple[str, ...], ...] = ()  # each entry: the columns that define a group
    group_stats: tuple[str, ...] = ("mean",)
    series_stats: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.identifiers or self.group_by or self.series_stats)


def part_name(column: str, length: int) -> str:
    return f"{column}_p{length}"


def group_label(columns: Sequence[str]) -> str:
    return "_x_".join(columns)


def _safe(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")


def with_code_parts(static: pd.DataFrame, code_parts: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    """The static table plus a text column for every requested code prefix."""
    out = static.copy()
    for column, lengths in code_parts.items():
        if column not in static.columns:
            raise FeatureError(f"code_parts: static column {column!r} not found. Available: {_available(static)}")
        for length in lengths:
            if not isinstance(length, int) or isinstance(length, bool) or length < 1:
                raise FeatureError(f"code_parts for {column!r}: lengths must be integers >= 1, got {length!r}")
            out[part_name(column, length)] = static[column].astype(str).str[:length]
    return out


def _available(static: pd.DataFrame) -> list[str]:
    return [c for c in static.columns if c != ID]


def check_spec(spec: StaticSpec, static: pd.DataFrame) -> pd.DataFrame:
    """Validate the spec against the static table. Returns the table with the code parts added."""
    if spec.encoding not in ENCODINGS:
        raise FeatureError(f"static encoding must be one of {list(ENCODINGS)}, got {spec.encoding!r}")
    bad = [s for s in spec.group_stats if s not in GROUP_STATS]
    if bad:
        raise FeatureError(f"unknown group statistic(s) {bad}. Known: {list(GROUP_STATS)}")
    bad = [s for s in spec.series_stats if s not in SERIES_STATS]
    if bad:
        raise FeatureError(f"unknown series statistic(s) {bad}. Known: {list(SERIES_STATS)}")
    table = with_code_parts(static, spec.code_parts)
    wanted = list(spec.identifiers) + [c for group in spec.group_by for c in group]
    missing = sorted({c for c in wanted if c not in table.columns})
    if missing:
        raise FeatureError(
            f"static feature(s) refer to unknown column(s) {missing}. Available: {_available(table)}"
        )
    return table


def _categories(table: pd.DataFrame, column: str) -> list[str]:
    return sorted(table[column].astype(str).unique())


def identifier_names(spec: StaticSpec, table: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in spec.identifiers:
        if spec.encoding == "ordinal":
            names.append(f"id_{column}")
        else:
            names += [f"{column}_{_safe(c)}" for c in _categories(table, column)]
    return names


def feature_names(spec: StaticSpec, table: pd.DataFrame) -> list[str]:
    names = identifier_names(spec, table)
    names += [f"ser_{s}" for s in spec.series_stats]
    for group in spec.group_by:
        names += [f"grp_{group_label(group)}_{s}" for s in spec.group_stats]
    return names


def _series_stats(train: pd.DataFrame, season_length: int) -> pd.DataFrame:
    rows = {}
    for uid, g in train.sort_values(DATE, kind="stable").groupby(ID, sort=False):
        y = g[TARGET].to_numpy(dtype=float)
        n = len(y)
        mean = float(np.mean(y))
        std = float(np.std(y, ddof=1)) if n > 1 else 0.0
        trend = np.nan
        if n >= 3 and mean != 0:
            x = np.arange(n) - (n - 1) / 2
            trend = float(np.sum(x * (y - mean)) / np.sum(x * x)) / abs(mean)
        strength = np.nan
        if season_length >= 2 and n >= 2 * season_length and np.var(y) > 0:
            position = np.arange(n) % season_length
            profile = pd.Series(y).groupby(position).transform("mean").to_numpy()
            strength = float(np.clip(np.var(profile) / np.var(y), 0.0, 1.0))
        rows[uid] = {
            "mean": mean,
            "std": std,
            "cv": std / abs(mean) if mean != 0 else np.nan,
            "length": float(n),
            "trend": trend,
            "zero_share": float(np.mean(y == 0)),
            "season_strength": strength,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def static_feature_table(
    train: pd.DataFrame, static: pd.DataFrame, spec: StaticSpec, season_length: int = 1
) -> pd.DataFrame:
    """The static feature columns for every series in `train`, indexed by unique_id.

    `static` is the table of static values (unique_id + static columns). Statistics use `train` only.
    """
    table = check_spec(spec, static).set_index(ID)
    ids = list(pd.unique(train[ID]))
    absent = [u for u in ids if u not in table.index]
    if absent:
        raise FeatureError(f"no static values for series {absent[:5]}{' ...' if len(absent) > 5 else ''}")
    stats = _series_stats(train, season_length)
    out = pd.DataFrame(index=pd.Index(ids, name=ID))

    for column in spec.identifiers:
        values = table.loc[ids, column].astype(str)
        categories = _categories(table.reset_index(), column)
        if spec.encoding == "ordinal":
            out[f"id_{column}"] = values.map({c: i for i, c in enumerate(categories)}).astype(float)
        else:
            for c in categories:
                out[f"{column}_{_safe(c)}"] = (values == c).astype(float)

    for s in spec.series_stats:
        out[f"ser_{s}"] = stats.loc[ids, s]

    for group in spec.group_by:
        keys = table.loc[ids, list(group)].astype(str).agg("\x1f".join, axis=1)
        frame = pd.DataFrame(
            {"key": keys, "mean": stats.loc[ids, "mean"], "std": stats.loc[ids, "std"],
             "trend": stats.loc[ids, "trend"]}
        )
        by = frame.groupby("key")
        group_mean = by["mean"].transform("mean")
        label = group_label(group)
        columns = {
            "mean": group_mean,
            "median": by["mean"].transform("median"),
            "std": by["std"].transform("mean"),
            "n_series": by["mean"].transform("size").astype(float),
            "rel_level": (frame["mean"] / group_mean.where(group_mean != 0)),
            "growth": by["trend"].transform("mean"),
        }
        for s in spec.group_stats:
            out[f"grp_{label}_{s}"] = columns[s]
    return out
