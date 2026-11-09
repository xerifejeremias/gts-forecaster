"""Data for the report's Models tab: run time per model and which models the selection uses.

The accuracy of each model is computed in the page from the backtest rows of the Accuracy tab
(so it follows the filters); this module adds what does not depend on the filters:

    {"top_n": 3, "has_ranks": true, "timings_file": "...",
     "models": [{"name", "seconds", "fit_seconds", "predict_seconds", "n_folds", "n_series",
                 "status", "top_n_series", "used_series"}, ...]}

seconds       mean fit + predict time per backtest fold (None if the model never ran)
top_n_series  series where the model ranks in the top_n of the backtest (own ranking)
used_series   series whose final forecast uses the model (with the blend rule this also counts
              the common mix, so a model can be used without being in any series' own top_n)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from gts_forecaster.selection import ENSEMBLE_ROW


def timings_file_for(backtest_file: Path | None) -> Path | None:
    """The timings file written by the same backtest (the files share a time stamp)."""
    if backtest_file is None:
        return None
    name = backtest_file.name.replace("backtest_forecasts", "backtest_timings", 1)
    path = backtest_file.with_name(name)
    return path if path.is_file() else None


def _round(value: float) -> float | None:
    return None if value is None or math.isnan(value) else round(float(value), 3)


def model_summary(
    selection: Mapping[str, Any] | None,
    scores: pd.DataFrame | None,
    timings: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """One entry per candidate model: run time, own top-N count and ensemble use."""
    meta = (selection or {}).get("meta") or {}
    top_n = int(meta.get("top_n") or 3)
    names: list[str] = list(meta.get("models") or [])
    for frame in (scores, timings):
        if frame is not None and len(frame):
            names += [m for m in frame["model"].astype(str).unique()
                      if m not in names and m != ENSEMBLE_ROW]
    if not names:
        return None

    used: dict[str, int] = {}
    for entry in ((selection or {}).get("series") or {}).values():
        for m in entry.get("models", []):
            used[m["name"]] = used.get(m["name"], 0) + 1

    top: dict[str, int] = {}
    if scores is not None and len(scores):
        ranked = scores[pd.to_numeric(scores["rank"], errors="coerce") <= top_n]
        top = ranked.groupby("model")["uuid"].nunique().to_dict()

    timing: dict[str, dict[str, Any]] = {}
    if timings is not None and len(timings):
        for name, g in timings.groupby("model"):
            ok = g[g["status"] == "ok"]
            status = "ok" if len(ok) else str(g["status"].iloc[-1])
            fit, predict = ok["fit_seconds"].astype(float), ok["predict_seconds"].astype(float)
            timing[str(name)] = {
                "seconds": _round((fit + predict).mean()) if len(ok) else None,
                "fit_seconds": _round(fit.mean()) if len(ok) else None,
                "predict_seconds": _round(predict.mean()) if len(ok) else None,
                "n_folds": int(len(ok)),
                "n_series": _round(ok["n_series"].astype(float).mean()) if len(ok) else None,
                "status": status,
            }

    models = []
    for name in names:
        t = timing.get(name, {"seconds": None, "fit_seconds": None, "predict_seconds": None,
                              "n_folds": 0, "n_series": None, "status": "not run"})
        models.append({"name": name, **t, "top_n_series": int(top.get(name, 0)),
                       "used_series": int(used.get(name, 0))})
    return {"top_n": top_n, "has_ranks": bool(top), "models": models}
