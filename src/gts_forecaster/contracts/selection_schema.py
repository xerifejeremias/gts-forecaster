"""Contract for selection.json: the saved choice of model(s) per series.

The file is meant to be edited by hand (for example to swap one model name for another), so it
is checked every time it is read, and before it is written.

    {
      "meta":   {...},
      "series": {"<uuid>": {"locked": false, "models": [{"name": "naive", "weight": 1.0}], "note": "..."}}
    }
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any

from gts_forecaster.contracts.errors import DataContractError

WEIGHT_TOLERANCE = 1e-6
_ENTRY_KEYS = {"locked", "models", "note"}
_MAX_LISTED = 10


def validate_selection(
    doc: Any,
    valid_models: Collection[str],
    expected_ids: Collection[str] | None = None,
) -> None:
    """Raise DataContractError if `doc` is not a valid selection document.

    `valid_models` are the model names that exist in the catalog. If `expected_ids` is given,
    every one of those series must have an entry (extra entries are allowed).
    """
    problems: list[str] = []

    if not isinstance(doc, Mapping):
        raise DataContractError("selection must be a JSON object with 'meta' and 'series'")
    if set(doc) != {"meta", "series"}:
        problems.append(f"top-level keys must be exactly 'meta' and 'series', got {sorted(doc)}")
    meta, series = doc.get("meta"), doc.get("series")
    if not isinstance(meta, Mapping):
        problems.append("'meta' must be an object")
    if not isinstance(series, Mapping):
        raise DataContractError("Selection is invalid: " + "; ".join([*problems, "'series' must be an object"]))

    valid = set(valid_models)
    for uid, entry in series.items():
        where = f"series {uid}"
        if not isinstance(entry, Mapping):
            problems.append(f"{where}: entry must be an object")
            continue
        unknown_keys = set(entry) - _ENTRY_KEYS
        if unknown_keys or "models" not in entry or "locked" not in entry:
            problems.append(f"{where}: keys must be 'locked', 'models' and optionally 'note' (got {sorted(entry)})")
            continue
        if not isinstance(entry["locked"], bool):
            problems.append(f"{where}: 'locked' must be true or false")
        if "note" in entry and entry["note"] is not None and not isinstance(entry["note"], str):
            problems.append(f"{where}: 'note' must be text")
        models = entry["models"]
        if not isinstance(models, list) or not models:
            problems.append(f"{where}: 'models' must be a non-empty list")
            continue
        names, total = [], 0.0
        for item in models:
            if not isinstance(item, Mapping) or set(item) != {"name", "weight"}:
                problems.append(f"{where}: each model needs exactly 'name' and 'weight'")
                continue
            name, weight = item["name"], item["weight"]
            names.append(name)
            if name not in valid:
                problems.append(f"{where}: unknown model {name!r} (valid: {sorted(valid)})")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                problems.append(f"{where}: weight of {name!r} must be a number > 0, got {weight!r}")
            else:
                total += weight
        if len(set(map(str, names))) != len(names):
            problems.append(f"{where}: a model is listed more than once")
        if abs(total - 1.0) > WEIGHT_TOLERANCE and not any(p.startswith(where) and "weight" in p for p in problems):
            problems.append(f"{where}: weights sum to {total:.6f}, expected 1")

    if expected_ids is not None:
        missing = [u for u in expected_ids if u not in series]
        if missing:
            shown = ", ".join(map(str, missing[:_MAX_LISTED])) + (" ..." if len(missing) > _MAX_LISTED else "")
            problems.append(f"{len(missing)} series have no entry: {shown}")

    if problems:
        raise DataContractError("Selection is invalid:\n  - " + "\n  - ".join(problems))
