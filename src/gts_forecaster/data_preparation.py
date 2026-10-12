"""Load, validate and repair the input data. Shared first step of both pipelines.

Run as a command to check an input file:  poetry run validate-input --frequency monthly
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from gts_forecaster.cleaning.missing import handle_missing_data, split_short_series
from gts_forecaster.columns import ID, to_model_format
from gts_forecaster.config import FALLBACKS, ConfigError, RunConfig, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.frequency_checks import check_complete_panel, check_model_frame
from gts_forecaster.contracts.input_schema import validate_input_structure
from gts_forecaster.contracts.static_checks import extract_static_table
from gts_forecaster.io.loaders import load_input

logger = logging.getLogger(__name__)


@dataclass
class PreparedInput:
    """The input data, validated and repaired, in the model format.

    `data` has the columns unique_id, ds, y, the explanatory columns and is_filled.
    """

    data: pd.DataFrame  # complete series for the models
    fallback: pd.DataFrame  # series that get a fallback forecast (unique_id, last value, ...)
    report: pd.DataFrame  # missing-data report, one row per missing run
    cutoff: pd.Timestamp  # common last date of all series
    config: RunConfig
    # One row per series (unique_id + the static columns: market, product, company ...), or empty.
    static: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({ID: []}))


def prepare_input_data(config: RunConfig) -> PreparedInput:
    """Load the input file, enforce the input contract and repair missing periods.

    The input file has the columns uuid, date, target and the explanatory columns. The result
    is renamed to the model format unique_id, ds, y. Raises DataContractError if the data
    cannot be loaded or breaks the contract. Series with fewer than the profile's
    `min_series_length` rows pass the structural checks but then get the last-value fallback
    (with a warning) instead of stopping the run.
    """
    raw = load_input(config)
    validated = validate_input_structure(
        raw,
        config.explanatory_columns,
        pandas_freq=config.profile.pandas_freq,
        min_series_length=1,  # too-short series get a fallback below instead of stopping the run
        static_columns=config.static_columns,
        id_format=config.id_format,
    )
    static = to_model_format(extract_static_table(validated, config.static_columns))
    long_enough, short_fallback = split_short_series(
        to_model_format(validated.drop(columns=list(config.static_columns))),
        config.profile.min_series_length,
    )
    if long_enough.empty:
        raise DataContractError(
            f"Every series has fewer than {config.profile.min_series_length} rows: nothing to model."
        )
    result = handle_missing_data(
        long_enough,
        pandas_freq=config.profile.pandas_freq,
        value_columns=config.value_columns,
        max_fill_periods=config.max_fill_periods,
        fallback=config.fallback,
    )
    check_complete_panel(
        result.data, config.profile.pandas_freq, config.value_columns, result.cutoff
    )
    check_model_frame(result.data, config.explanatory_columns)
    fallback = result.fallback
    if not short_fallback.empty:
        fallback = pd.concat([f for f in (fallback, short_fallback) if not f.empty], ignore_index=True)
    return PreparedInput(result.data, fallback, result.report, result.cutoff, config, static)


def limit_series(prepared: PreparedInput, n: int | None) -> PreparedInput:
    """Keep an evenly spaced sample of `n` series (a quick trial run on a big file).

    Series are sorted by (length, id) and picked at even steps, so short and long series are
    both represented and the same `n` always gives the same series. Fallback series and static
    rows are filtered to match. `n=None` or `n >= number of series` returns `prepared` unchanged.
    """
    if n is None:
        return prepared
    if n < 1:
        raise ValueError(f"max_series must be at least 1, got {n}")
    lengths = prepared.data.groupby(ID, sort=False).size()
    if n >= len(lengths):
        return prepared
    ordered = sorted(lengths.index, key=lambda uid: (lengths[uid], str(uid)))
    positions = sorted({round(k * (len(ordered) - 1) / (n - 1)) for k in range(n)} if n > 1 else {0})
    keep = {ordered[i] for i in positions}
    logger.info("max_series: using %d of %d series", len(keep), len(lengths))
    data = prepared.data[prepared.data[ID].isin(keep)].reset_index(drop=True)
    fallback = prepared.fallback[prepared.fallback[ID].isin(keep)].reset_index(drop=True)
    static = prepared.static
    if len(static):
        static = static[static[ID].isin(keep)].reset_index(drop=True)
    return replace(prepared, data=data, fallback=fallback, static=static)


def load_prepared(frequency: str, **overrides: object) -> PreparedInput:
    """Convenience: read the input for one frequency and return it in the model format.

        prepared = load_prepared("weekly")
        prepared = load_prepared("daily", max_fill_periods=5, fallback="zero")
        prepared.data      # unique_id, ds, y, explanatory columns, is_filled

    Any config option (horizon, data_path, ...) can be passed as a keyword.
    """
    return prepare_input_data(load_config({"frequency": frequency, **overrides}))


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: check an input file and print a summary."""
    parser = argparse.ArgumentParser(description="Validate and prepare the input data.")
    parser.add_argument("--frequency", help="yearly | monthly | weekly | daily")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--data-path", help="input CSV (default: from config)")
    parser.add_argument("--max-fill-periods", type=int)
    parser.add_argument("--static-columns", nargs="*", help="input columns that identify a series")
    parser.add_argument("--fallback", choices=FALLBACKS)
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(
            {
                "frequency": args.frequency,
                "horizon": args.horizon,
                "data_path": args.data_path,
                "max_fill_periods": args.max_fill_periods,
                "fallback": args.fallback,
                "static_columns": args.static_columns,
            },
            config_dir=args.config_dir,
        )
        prepared = prepare_input_data(config)
    except (ConfigError, DataContractError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    n_model = prepared.data[ID].nunique()
    print(f"OK: {config.frequency} data from {config.data_path}")
    print(f"  common cutoff:            {prepared.cutoff:%Y-%m-%d}")
    print(f"  series for the models:    {n_model}")
    print(f"  series with fallback:     {len(prepared.fallback)}")
    print(f"  rows for the models:      {len(prepared.data)}")
    print(f"  columns:                  {', '.join(prepared.data.columns)}")
    if config.static_columns:
        print(f"  static columns:           {', '.join(config.static_columns)}")
    print(f"  missing runs repaired:    {int((prepared.report['action'] != 'fallback').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
