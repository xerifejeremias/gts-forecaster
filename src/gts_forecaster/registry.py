"""Read and write the registry: selection.json and backtest_scores.csv.

Layout: <registry_dir>/<frequency>/selection.json and backtest_scores.csv
Files are written atomically (a temporary file is moved into place), so a failed write never
leaves a half-written selection behind.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.selection_schema import validate_selection
from gts_forecaster.selection import Selection

# Two selections come out of one backtest (decided 2026-10-06):
#   selection_fast.json  chosen among the fast models only; used when fast_only is on (default)
#   selection.json       chosen among every model that was backtested (incl. Prophet / Orbit);
#                        used when fast_only is off
# So a forecast never asks for a model that its fast_only setting leaves out.
SELECTION_FILE = "selection.json"
SELECTION_FAST_FILE = "selection_fast.json"
SCORES_FILE = "backtest_scores.csv"
SCORES_FAST_FILE = "backtest_scores_fast.csv"


def selection_file(fast_only: bool) -> str:
    """The selection a run with this fast_only setting uses."""
    return SELECTION_FAST_FILE if fast_only else SELECTION_FILE


def scores_file(fast_only: bool) -> str:
    return SCORES_FAST_FILE if fast_only else SCORES_FILE
_DECIMALS = 6


def registry_path(registry_dir: Path, frequency: str) -> Path:
    return Path(registry_dir) / frequency


def _rounded_weights(weights: list[float]) -> list[float]:
    """Round to a readable precision and make sure the weights still sum to exactly 1."""
    rounded = [round(w, _DECIMALS) for w in weights]
    rounded[-1] = round(1.0 - sum(rounded[:-1]), _DECIMALS)
    return rounded


def build_selection_document(selection: Selection, meta: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON-ready document for a Selection."""
    series = {}
    for uid, chosen in selection.series.items():
        names = [name for name, _ in chosen.models]
        weights = _rounded_weights([w for _, w in chosen.models])
        entry: dict[str, Any] = {
            "locked": chosen.locked,
            "models": [{"name": n, "weight": w} for n, w in zip(names, weights, strict=True)],
        }
        if chosen.note:
            entry["note"] = chosen.note
        series[uid] = entry
    return {"meta": dict(meta), "series": series}


def merge_locked(new: dict[str, Any], existing: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Keep the entries marked `locked` in the existing file. Returns the merged document and the kept ids."""
    if not existing:
        return new, []
    kept = []
    for uid, entry in existing.get("series", {}).items():
        if entry.get("locked") is True and uid in new["series"]:
            new["series"][uid] = entry
            kept.append(uid)
    return new, kept


def write_json_atomic(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_scores_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out = table.rename(columns={"unique_id": "uuid"})  # user-facing file uses the input name
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_selection(
    path: Path, valid_models: Collection[str], expected_ids: Collection[str] | None = None
) -> dict[str, Any]:
    """Load selection.json and check it against the contract."""
    path = Path(path)
    if not path.is_file():
        raise DataContractError(f"Selection file not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Could not read selection file {path}: {exc}") from exc
    validate_selection(doc, valid_models, expected_ids)
    return doc


def read_existing_selection(path: Path, valid_models: Collection[str]) -> dict[str, Any] | None:
    """The current selection file, or None if there is none yet. An invalid file is an error."""
    return read_selection(path, valid_models) if Path(path).is_file() else None


BACKTEST_FORECASTS_PREFIX = "backtest_forecasts"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M"  # no colon: it is not allowed in Windows file names
BACKTEST_FORECAST_COLUMNS = [
    "uuid", "model", "fold", "cutoff", "ds", "horizon_step", "actual", "forecast", "is_filled"
]


def backtest_forecasts_table(result) -> pd.DataFrame:
    """Forecast vs actual for every series, model, fold and date of a backtest, one row each.

    `cutoff` is the last training date of the fold. Filled-in periods are flagged (`is_filled`)
    because they are not scored.
    """
    p = result.predictions
    out = pd.DataFrame(
        {
            "uuid": p["unique_id"],
            "model": p["model"],
            "fold": p["fold"],
            "cutoff": p["fold"].map(result.cutoffs),
            "ds": p["ds"],
            "horizon_step": p["horizon_step"],
            "actual": p["y"],
            "forecast": p["yhat"],
            "is_filled": p["is_filled"],
        }
    )
    return out[BACKTEST_FORECAST_COLUMNS].sort_values(
        ["uuid", "model", "fold", "ds"], kind="stable"
    ).reset_index(drop=True)


def new_backtest_forecasts_path(folder: Path, run_time: datetime) -> Path:
    """backtest_forecasts_YYYYMMDD_HHMM.csv in `folder`. Never an existing file: a second run in
    the same minute gets a _2, _3, ... suffix instead of overwriting."""
    stem = f"{BACKTEST_FORECASTS_PREFIX}_{run_time.strftime(TIMESTAMP_FORMAT)}"
    path = Path(folder) / f"{stem}.csv"
    n = 2
    while path.exists():
        path = Path(folder) / f"{stem}_{n}.csv"
        n += 1
    return path


def write_backtest_forecasts_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    os.replace(tmp, path)
