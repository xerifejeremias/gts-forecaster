"""Data for the report's Accuracy tab: forecasts next to the actuals that came true.

Two sources, both turned into the same row layout (one row per series, origin and date):

  backtest  the newest backtest's out-of-sample forecasts (backtesting/<frequency>/
            backtest_forecasts_*.csv): every model on every fold. Origin = the fold's cutoff.
  live      forecasts of earlier `run-pipeline` runs (runs/<frequency>/<id>/outputs/forecast/)
            whose dates now have actuals in the input file. Origin = the data cutoff of that run.
            This is the drift check: how the forecasts that were really delivered performed.

Both carry a "final" forecast: in the backtest it is rebuilt per series from the selection
weights (note: the selection was chosen on these same folds, so the backtest view of the final
forecast is optimistic); in live runs it is what was delivered.

Columnar layout, so the page stays small:
    {"models": [...], "dates": [ISO ...],
     "sources": {"backtest": {"origins": [ISO ...], "s": [series index], "o": [origin index],
                              "d": [date index], "h": [horizon step], "a": [actual],
                              "f": {"<model>": [value|null], ..., "__final__": [...]}},
                 "live": {...}}}
Series indexes point into the report's `series` list. All metrics are computed in the page.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FINAL_KEY = "__final__"
_DECIMALS = 4


def _numbers(values) -> list[float | None]:
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), _DECIMALS)
            for v in values]


def backtest_rows(table: pd.DataFrame, selection: Mapping[str, Any] | None) -> pd.DataFrame:
    """Long backtest file (uuid, model, fold, cutoff, ds, horizon_step, actual, forecast, is_filled)
    -> one row per (uuid, cutoff, ds) with a column per model and the rebuilt final forecast."""
    needed = {"uuid", "model", "cutoff", "ds", "horizon_step", "actual", "forecast"}
    missing = needed - set(table.columns)
    if missing:
        raise ValueError(f"backtest forecasts file is missing column(s) {sorted(missing)}")
    if "is_filled" in table.columns:  # filled-in periods are not real actuals
        table = table[~table["is_filled"].astype(str).str.lower().isin(["true", "1"])]
    table = table.assign(uuid=table["uuid"].astype(str), cutoff=pd.to_datetime(table["cutoff"]),
                         ds=pd.to_datetime(table["ds"]))
    keys = ["uuid", "cutoff", "ds"]
    wide = table.pivot_table(index=keys, columns="model", values="forecast", aggfunc="last")
    wide.columns = [str(c) for c in wide.columns]
    base = table.groupby(keys, sort=True)[["actual", "horizon_step"]].first()
    rows = base.join(wide).reset_index()
    rows[FINAL_KEY] = _final_from_selection(rows, list(wide.columns), selection)
    return rows


def _final_from_selection(rows: pd.DataFrame, models: list[str], selection) -> np.ndarray:
    final = np.full(len(rows), np.nan)
    if not selection:
        return final
    entries = selection.get("series", {})
    for uid, idx in rows.groupby("uuid", sort=False).groups.items():
        members = entries.get(uid, {}).get("models", [])
        if not members or any(m["name"] not in models for m in members):
            continue
        part = rows.loc[idx]
        total = sum(part[m["name"]].to_numpy(dtype=float) * float(m["weight"]) for m in members)
        final[rows.index.get_indexer(idx)] = total  # NaN where a member has no forecast
    return final


def live_rows(runs_folder: Path, history: pd.DataFrame) -> pd.DataFrame:
    """Forecasts of earlier run-pipeline runs that now have actuals. One origin per data cutoff
    (the newest run wins when several runs used the same data)."""
    runs_folder = Path(runs_folder)
    if not runs_folder.is_dir():
        return pd.DataFrame()
    by_cutoff: dict[str, Path] = {}
    for run_dir in sorted(p for p in runs_folder.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.json"
        forecast_dir = run_dir / "outputs" / "forecast"
        if not manifest_path.is_file() or not (forecast_dir / "forecasts.csv").is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cutoff = (manifest.get("input") or {}).get("data_cutoff")
        if cutoff:
            by_cutoff[cutoff[:10]] = forecast_dir  # sorted by run id: the newest run wins
    actual = history.assign(uuid=history["uuid"].astype(str), date=pd.to_datetime(history["date"]))
    actual = actual.dropna(subset=["target"]).set_index(["uuid", "date"])["target"]
    frames = []
    for cutoff, forecast_dir in sorted(by_cutoff.items()):
        final = pd.read_csv(forecast_dir / "forecasts.csv", dtype={"uuid": str}, parse_dates=["ds"])
        frame = final[["uuid", "ds", "forecast"]].rename(columns={"forecast": FINAL_KEY})
        by_model_path = forecast_dir / "forecasts_by_model.csv"
        if by_model_path.is_file():
            by_model = pd.read_csv(by_model_path, dtype={"uuid": str}, parse_dates=["ds"])
            # model columns are numeric; the identifier columns (market, product ...) are not
            keep = ["uuid", "ds"] + [
                c for c in by_model.columns
                if c not in ("uuid", "ds") and pd.api.types.is_numeric_dtype(by_model[c])
            ]
            frame = frame.merge(by_model[keep], on=["uuid", "ds"], how="left")
        frame["actual"] = actual.reindex(pd.MultiIndex.from_frame(frame[["uuid", "ds"]])).to_numpy()
        frame = frame.dropna(subset=["actual"])
        if frame.empty:
            continue  # no actuals yet for this run's forecast dates
        frame["cutoff"] = pd.Timestamp(cutoff)
        frame["horizon_step"] = frame.groupby("uuid")["ds"].rank(method="first").astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def encode(sources: Mapping[str, pd.DataFrame], series_ids: Sequence[str], max_values: int = 3_000_000,
           keep_models: Sequence[str] = ()) -> dict[str, Any] | None:
    """Columnar encoding of the row tables. Rows of series not in the report are dropped. If the
    data is very large, only `keep_models` (and the final forecast) are kept."""
    index = {uid: k for k, uid in enumerate(series_ids)}
    sources = {name: rows[rows["uuid"].isin(index)] for name, rows in sources.items() if rows is not None and len(rows)}
    sources = {name: rows for name, rows in sources.items() if len(rows)}
    if not sources:
        return None
    reserved = {"uuid", "cutoff", "ds", "horizon_step", "actual", FINAL_KEY}
    models: list[str] = []
    for rows in sources.values():
        models += [c for c in rows.columns if c not in reserved and c not in models]
    size = sum(len(rows) * (len(models) + 2) for rows in sources.values())
    note = None
    if size > max_values and keep_models:
        kept = [m for m in models if m in set(keep_models)]
        note = f"large data: only the models used in the selection are included ({len(kept)} of {len(models)})"
        logger.warning("accuracy view: %s", note)
        models = kept
    dates = sorted({d for rows in sources.values() for d in rows["ds"]})
    date_index = {d: k for k, d in enumerate(dates)}
    encoded = {}
    for name, rows in sources.items():
        rows = rows.sort_values(["cutoff", "uuid", "ds"], kind="stable")
        origins = sorted(rows["cutoff"].unique())
        origin_index = {o: k for k, o in enumerate(origins)}
        forecasts = {m: _numbers(rows[m].to_numpy(dtype=float)) for m in models if m in rows.columns}
        forecasts[FINAL_KEY] = _numbers(rows[FINAL_KEY].to_numpy(dtype=float))
        encoded[name] = {
            "origins": [pd.Timestamp(o).strftime("%Y-%m-%d") for o in origins],
            "s": [index[u] for u in rows["uuid"]],
            "o": [origin_index[o] for o in rows["cutoff"]],
            "d": [date_index[d] for d in rows["ds"]],
            "h": [int(v) for v in rows["horizon_step"]],
            "a": _numbers(rows["actual"].to_numpy(dtype=float)),
            "f": forecasts,
        }
    return {
        "models": models,
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "sources": encoded,
        "note": note,
    }


def newest_backtest_file(folder: Path, selection: Mapping[str, Any] | None) -> Path | None:
    """The backtest file that produced the selection (named in its meta), else the newest one."""
    folder = Path(folder)
    named = ((selection or {}).get("meta") or {}).get("backtest_forecasts_file")
    if named and (folder / named).is_file():
        return folder / named
    files = sorted(folder.glob("backtest_forecasts_*.csv")) if folder.is_dir() else []
    return files[-1] if files else None
