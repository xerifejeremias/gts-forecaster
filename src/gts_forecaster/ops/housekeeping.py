"""Clean-up after a job: old run folders and old backtest files, per frequency.

Never deleted: the newest run (and the one named in runs/<frequency>/latest.txt), and the backtest files
named in the current selections (the report's Accuracy tab reads them).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^\d{8}_\d{6}(_\d+)?$")
_BACKTEST_FILE = re.compile(r"^backtest_(forecasts|timings|outliers|failures)(_\d{8}_\d{4}(?:_\d+)?)\.csv$")


def clean_runs(runs_dir: Path, frequency: str, keep: int) -> list[str]:
    """Delete all but the `keep` newest run folders of a frequency. Returns the deleted run ids."""
    folder = Path(runs_dir) / frequency
    if not folder.is_dir():
        return []
    runs = sorted(p.name for p in folder.iterdir() if p.is_dir() and _RUN_ID.match(p.name))
    latest_file = folder / "latest.txt"
    protected = {latest_file.read_text(encoding="utf-8").strip()} if latest_file.is_file() else set()
    deleted = []
    for run_id in runs[: max(len(runs) - keep, 0)]:
        if run_id in protected:
            continue
        shutil.rmtree(folder / run_id, ignore_errors=True)
        deleted.append(run_id)
    if deleted:
        logger.info("clean-up: deleted %d old run folder(s) of %s, kept %d", len(deleted), frequency, keep)
    return deleted


def _selected_backtest_files(registry_dir: Path, frequency: str) -> set[str]:
    names = set()
    for name in ("selection_fast.json", "selection.json"):
        path = Path(registry_dir) / frequency / name
        try:
            meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
        except (OSError, ValueError):
            continue
        if meta.get("backtest_forecasts_file"):
            names.add(meta["backtest_forecasts_file"])
    return names


def clean_backtests(backtesting_dir: Path, registry_dir: Path, frequency: str, keep: int) -> list[str]:
    """Delete the files of all but the `keep` newest backtests (forecasts, timings, outliers, failures
    share a time stamp). Backtests named in the current selections are kept. Returns the deleted stamps."""
    folder = Path(backtesting_dir) / frequency
    if not folder.is_dir():
        return []
    by_stamp: dict[str, list[Path]] = {}
    for path in folder.iterdir():
        match = _BACKTEST_FILE.match(path.name)
        if match:
            by_stamp.setdefault(match.group(2), []).append(path)
    protected = {_BACKTEST_FILE.match(n).group(2) for n in _selected_backtest_files(registry_dir, frequency)
                 if _BACKTEST_FILE.match(n)}
    stamps = sorted(by_stamp)
    deleted = []
    for stamp in stamps[: max(len(stamps) - keep, 0)]:
        if stamp in protected:
            continue
        for path in by_stamp[stamp]:
            path.unlink(missing_ok=True)
        deleted.append(stamp.lstrip("_"))
    if deleted:
        logger.info("clean-up: deleted %d old backtest(s) of %s, kept %d", len(deleted), frequency, keep)
    return deleted
