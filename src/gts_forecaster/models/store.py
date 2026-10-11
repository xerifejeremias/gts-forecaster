"""Save and load fitted models (frozen-models mode).

Layout: <fitted_models_dir>/<frequency>/<fast|full>/manifest.json and one <model>.pkl per model.
"fast" holds the models of selection_fast.json (fast_only on, the default), "full" those of
selection.json (fast_only off), so train and forecast always use the same set.

The pickle files are Python objects: only load files this project wrote itself.
A model that cannot be pickled (some Prophet / Orbit objects) is simply not saved, and is
refitted when the forecast runs. Files are written atomically, the manifest last.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.models.base import BaseForecaster

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
_PRIMITIVES = (str, int, float, bool, type(None))


def fitted_models_path(fitted_models_dir: Path, frequency: str, fast_only: bool = True) -> Path:
    return Path(fitted_models_dir) / frequency / ("fast" if fast_only else "full")


def params_signature(model: BaseForecaster) -> str:
    """A stable text for a model's settings, to notice that models.yaml changed since training.

    Objects that are not plain values (for example a feature builder) are reduced to their
    class name, so a changed features.yaml is not detected; run `train` again after changing it.
    """

    def plain(value: Any) -> Any:
        if isinstance(value, _PRIMITIVES):
            return value
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        return type(value).__name__

    return json.dumps({"class": type(model).__name__, "params": plain(model.get_params())}, sort_keys=True)


def _library_versions(model: BaseForecaster) -> dict[str, str]:
    versions = {}
    for package in model.requires_packages:
        try:
            from importlib.metadata import version

            versions[package] = version(package)
        except Exception:  # not installed under that distribution name
            continue
    return versions


def save_fitted_models(
    folder: Path, fitted: Mapping[str, BaseForecaster], meta: Mapping[str, Any]
) -> list[str]:
    """Write the models and a manifest. Returns the names that could not be saved."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    not_saved: list[str] = []
    for name, model in fitted.items():
        path = folder / f"{name}.pkl"
        tmp = path.with_suffix(".pkl.tmp")
        try:
            with open(tmp, "wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception as exc:  # pickling can fail for models that wrap native objects
            tmp.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            logger.warning("model '%s' could not be saved (%s: %s); it will be refitted", name, type(exc).__name__, exc)
            not_saved.append(name)
            continue
        entries[name] = {
            "file": path.name,
            "class": type(model).__name__,
            "signature": params_signature(model),
            "supports_update": bool(model.supports_update),
            "n_series": len(getattr(model, "fitted_ids_", ())),
            "libraries": _library_versions(model),
        }
    for stale in folder.glob("*.pkl"):  # models of an earlier training that are not wanted now
        if stale.name not in {e["file"] for e in entries.values()}:
            stale.unlink()
    manifest = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        **dict(meta),
        "models": entries,
        "not_saved": not_saved,
    }
    tmp = folder / (MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, folder / MANIFEST_FILE)
    return not_saved


def load_manifest(folder: Path) -> dict[str, Any]:
    path = Path(folder) / MANIFEST_FILE
    if not path.is_file():
        raise DataContractError(
            f"No fitted models found in {folder}. Run `poetry run train` first, or use "
            "forecast_mode: frozen_selection."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Could not read {path}: {exc}") from exc


def load_fitted_model(folder: Path, manifest: Mapping[str, Any], name: str) -> BaseForecaster | None:
    """The saved model `name`, or None if it was not saved or the file cannot be read."""
    entry = manifest.get("models", {}).get(name)
    if entry is None:
        return None
    path = Path(folder) / entry["file"]
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)  # noqa: S301 - files written by this project only
    except Exception as exc:
        logger.warning("saved model '%s' could not be loaded (%s: %s); it will be refitted", name, type(exc).__name__, exc)
        return None
