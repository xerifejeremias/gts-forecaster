"""Build the data of the forecast report: one plain dictionary (JSON-ready).

This dictionary is the only thing the page needs, so it is also what a future web app or
dashboard can read:

    {
      "meta":   {frequency, horizon, cutoff, generated, metric, dimensions: [...], ...},
      "dates":  ["2024-01-01", ...],             history dates followed by forecast dates
      "n_history": 36,                            how many of `dates` are history
      "series": [{"id", "dims": {...}, "hist": [...], "fc": [...], "models": [...], "fallback",
                  "model_fc": {"<model>": [...]},    `fc` is the final (ensemble) forecast
                  "flags": [{"check", "message", "ds"}]}]   forecast sanity checks (may be empty)
    }

`meta.checks_ran` says whether forecast_checks.csv was found (the checks ran for these
forecasts); `meta.n_flagged` counts the series with at least one flag.

`hist` has one value per history date (null = not observed), `fc` one per forecast date.
Totals over markets, products or companies are computed by the page from these series.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from gts_forecaster.contracts.errors import DataContractError

_DECIMALS = 4


def _number(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), _DECIMALS)


def build_report_data(
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    pandas_freq: str,
    frequency: str,
    static_columns: Sequence[str] = (),
    selection: Mapping[str, Any] | None = None,
    fallback_ids: Sequence[str] = (),
    by_model: pd.DataFrame | None = None,
    meta: Mapping[str, Any] | None = None,
    checks: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """`history`: uuid, date, target (+ static columns), as in the input file.
    `forecasts`: ds, uuid, forecast (the forecast pipeline's forecasts.csv, user-facing names).
    `by_model`: optional forecasts_by_model.csv (uuid, ds, one column per model): adds a line per model.
    `checks`: optional forecast_checks.csv (uuid, check, ds, message, ...): the flags of each series."""
    for name, frame, cols in (
        ("history", history, ["uuid", "date", "target"]),
        ("forecasts", forecasts, ["ds", "uuid", "forecast"]),
    ):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise DataContractError(f"report: {name} is missing column(s) {missing}")
    if forecasts.empty:
        raise DataContractError("report: the forecasts table is empty")
    missing_static = [c for c in static_columns if c not in history.columns]
    if missing_static:
        raise DataContractError(f"report: input data has no column(s) {missing_static}")

    history = history.assign(date=pd.to_datetime(history["date"]))
    forecasts = forecasts.assign(ds=pd.to_datetime(forecasts["ds"]))
    last_history = history["date"].max()
    forecast_dates = pd.DatetimeIndex(sorted(forecasts["ds"].unique()))
    if forecast_dates.min() <= last_history:
        raise DataContractError(
            f"report: the forecasts start {forecast_dates.min():%Y-%m-%d}, which is not after the "
            f"last history date {last_history:%Y-%m-%d}. Were they made from this input file?"
        )
    history_dates = pd.date_range(history["date"].min(), last_history, freq=pandas_freq)

    dims = history.drop_duplicates("uuid").set_index("uuid")[list(static_columns)].astype(str)
    hist_wide = history.pivot_table(index="uuid", columns="date", values="target", aggfunc="last")
    hist_wide = hist_wide.reindex(columns=history_dates)
    fc_wide = forecasts.pivot_table(index="uuid", columns="ds", values="forecast", aggfunc="last")
    fc_wide = fc_wide.reindex(columns=forecast_dates)

    model_wide: dict[str, pd.DataFrame] = {}
    if by_model is not None and len(by_model):
        for column in ("uuid", "ds"):
            if column not in by_model.columns:
                raise DataContractError(f"report: forecasts_by_model is missing column {column!r}")
        by_model = by_model.assign(ds=pd.to_datetime(by_model["ds"]))
        names = [c for c in by_model.columns if c not in ("uuid", "ds", *static_columns)]
        for name in names:
            if by_model[name].notna().any():  # a model that forecast nothing has no line
                model_wide[name] = by_model.pivot_table(
                    index="uuid", columns="ds", values=name, aggfunc="last"
                ).reindex(columns=forecast_dates)

    flags: dict[str, list[dict[str, Any]]] = {}
    if checks is not None and len(checks):
        missing = [c for c in ("uuid", "check", "message") if c not in checks.columns]
        if missing:
            raise DataContractError(f"report: forecast_checks is missing column(s) {missing}")
        for row in checks.itertuples(index=False):
            ds = getattr(row, "ds", None)
            ds = None if ds is None or pd.isna(ds) else pd.Timestamp(ds).strftime("%Y-%m-%d")
            flags.setdefault(str(row.uuid), []).append(
                {"check": str(row.check), "message": str(row.message), "ds": ds}
            )

    entries = (selection or {}).get("series", {})
    fallback = set(fallback_ids)
    ids = sorted(set(hist_wide.index) | set(fc_wide.index))
    series = []
    for uid in ids:
        h = hist_wide.loc[uid].tolist() if uid in hist_wide.index else [None] * len(history_dates)
        f = fc_wide.loc[uid].tolist() if uid in fc_wide.index else [None] * len(forecast_dates)
        model_fc = {}
        for name, wide in model_wide.items():
            if uid in wide.index and wide.loc[uid].notna().any():
                model_fc[name] = [_number(v) for v in wide.loc[uid].tolist()]
        series.append(
            {
                "id": uid,
                "dims": dims.loc[uid].to_dict() if uid in dims.index else {},
                "hist": [_number(v) for v in h],
                "fc": [_number(v) for v in f],
                "models": [
                    {"name": m["name"], "weight": round(float(m["weight"]), 4)}
                    for m in entries.get(uid, {}).get("models", [])
                ],
                "fallback": uid in fallback,
                "model_fc": model_fc,
                "flags": flags.get(uid, []),
            }
        )
    dates = [d.strftime("%Y-%m-%d") for d in history_dates.append(forecast_dates)]
    return {
        "meta": {
            "frequency": frequency,
            "horizon": len(forecast_dates),
            "cutoff": last_history.strftime("%Y-%m-%d"),
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "dimensions": list(static_columns),
            "n_series": len(series),
            "n_fallback": sum(1 for s in series if s["fallback"]),
            "model_names": list(model_wide),
            "checks_ran": checks is not None,
            "n_flagged": sum(1 for s in series if s["flags"]),
            **dict(meta or {}),
        },
        "dates": dates,
        "n_history": len(history_dates),
        "series": series,
    }
