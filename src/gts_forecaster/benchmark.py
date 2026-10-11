"""Timing benchmark: how long would backtest and forecast take at your real data size?

Run on the computer that will run the pipeline (the numbers depend on its CPU):

    poetry run benchmark --frequency daily --series 3000
    poetry run benchmark --frequency monthly --series 800 --history 72 --no-fast-only

It makes synthetic series of the given frequency and history length (trend + seasonality + noise,
with the configured market/product/company columns), then for every model in config/models.yaml
times fit and predict on a small and a bigger sample (`--sizes`, default 20 and 80 series). From
those two points it extrapolates linearly to `--series` and estimates:

  backtest            n_folds x (fit + predict)
  forecast, refit     fit + predict once        (forecast_mode frozen_selection)
  forecast, frozen    predict only for models that can stay frozen, refit for the others

It prints a table and writes benchmarks/benchmark_<frequency>_<time>.csv. These are estimates:
real data, a different history length or other processes on the machine change them. A first
warm-up call per model (library loading, Chronos weight download) is not counted. A model whose
bigger sample would exceed --max-seconds is stopped after the small sample.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gts_forecaster.columns import DATE, ID, IS_FILLED, TARGET
from gts_forecaster.config import RunConfig, load_config
from gts_forecaster.models.base import BaseForecaster
from gts_forecaster.models.catalog import build_models_from_config
from gts_forecaster.splitting import forecast_split

logger = logging.getLogger(__name__)

DEFAULT_HISTORY = {"yearly": 15, "monthly": 60, "weekly": 156, "daily": 730}
COLUMNS = [
    "model", "cost", "supports_update", "status", "sizes", "fit_seconds", "predict_seconds",
    "est_fit_predict_s", "est_predict_s", "est_backtest_s", "est_forecast_refit_s", "est_forecast_frozen_s",
    "note",
]


def synthetic_panel(cfg: RunConfig, n_series: int, n_periods: int, seed: int = 0):
    """Model-format data (unique_id, ds, y, explanatory columns, is_filled) and the static table."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_periods, freq=cfg.profile.pandas_freq)
    m = max(int(cfg.profile.season_length), 1)
    t = np.arange(n_periods)
    frames, static_rows = [], []
    for i in range(n_series):
        level = rng.uniform(50, 1000)
        trend = rng.normal(0.001, 0.002)
        season = 1 + (0.25 * np.sin(2 * np.pi * (t + rng.integers(m)) / m) if m > 1 else 0)
        noise = rng.normal(1, 0.1, n_periods)
        y = np.maximum(level * (1 + trend * t) * season * noise, 0)
        uid = f"bench-{i:06d}"
        frame = pd.DataFrame({ID: uid, DATE: dates, TARGET: y})
        for k, column in enumerate(cfg.explanatory_columns):
            frame[column] = rng.normal(0, 1, n_periods) + k
        frame[IS_FILLED] = False
        frames.append(frame)
        static_rows.append({ID: uid, **{c: f"{c}_{(i // (k + 1)) % 3}" for k, c in enumerate(cfg.static_columns)}})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(static_rows, columns=[ID, *cfg.static_columns])


def _time(model: BaseForecaster, data: pd.DataFrame, horizon: int, pandas_freq: str):
    split = forecast_split(data, horizon, pandas_freq)
    eligible = split.train_lengths.index[split.train_lengths >= model.min_train_length]
    if len(eligible) == 0:
        raise ValueError(f"the synthetic history is too short for this model (needs {model.min_train_length})")
    fitted = model.clone()
    t0 = time.perf_counter()
    fitted.fit(split.train[split.train[ID].isin(eligible)])
    t1 = time.perf_counter()
    fitted.predict(split.future[split.future[ID].isin(eligible)])
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def _extrapolate(sizes: list[int], seconds: list[float], target: int) -> float:
    """Straight line through the measured points (a fixed cost plus a cost per series)."""
    if len(sizes) == 1 or sizes[-1] == sizes[0]:
        return seconds[-1] * target / sizes[-1]
    slope = (seconds[-1] - seconds[0]) / (sizes[-1] - sizes[0])
    if slope <= 0:  # noise on fast models: use the cost per series of the bigger sample
        return seconds[-1] * target / sizes[-1]
    fixed = max(seconds[-1] - slope * sizes[-1], 0.0)
    return fixed + slope * target


def run_benchmark(
    cfg: RunConfig,
    *,
    target_series: int,
    history: int,
    sizes: list[int],
    max_seconds: float = 600.0,
    config_dir: Path | None = None,
) -> pd.DataFrame:
    sizes = sorted(set(int(s) for s in sizes))
    if not sizes or sizes[0] < 1:
        raise ValueError("sizes must be positive numbers of series")
    data, static = synthetic_panel(cfg, max(sizes), history)
    models = build_models_from_config(cfg, config_dir, static)
    horizon, freq = cfg.horizon, cfg.profile.pandas_freq
    ids = list(pd.unique(data[ID]))
    rows = []
    for number, (name, model) in enumerate(models.items(), start=1):
        row = {"model": name, "cost": model.cost, "supports_update": bool(model.supports_update),
               "status": "ok", "note": ""}
        logger.info("model %s (%d/%d)", name, number, len(models))
        try:
            _time(model, data[data[ID].isin(ids[: min(2, len(ids))])], horizon, freq)  # warm-up
            measured, fits, predicts = [], [], []
            for size in sizes:
                fit_s, predict_s = _time(model, data[data[ID].isin(ids[:size])], horizon, freq)
                measured.append(size)
                fits.append(fit_s)
                predicts.append(predict_s)
                logger.info("  %d series: fit %.2fs, predict %.2fs", size, fit_s, predict_s)
                nxt = next((s for s in sizes if s > size), None)
                if nxt is not None and (fit_s + predict_s) * nxt / size > max_seconds:
                    row["note"] = f"stopped after {size} series (next sample would exceed {max_seconds:.0f}s)"
                    break
        except Exception as exc:  # report the model as failed, keep going with the others
            row.update(status="failed", note=f"{type(exc).__name__}: {exc}")
            logger.warning("  failed: %s", row["note"])
            rows.append(row)
            continue
        total = [f + p for f, p in zip(fits, predicts, strict=True)]
        est_total = _extrapolate(measured, total, target_series)
        est_predict = _extrapolate(measured, predicts, target_series)
        row.update(
            sizes=" ".join(map(str, measured)),
            fit_seconds=" ".join(f"{s:.2f}" for s in fits),
            predict_seconds=" ".join(f"{s:.2f}" for s in predicts),
            est_fit_predict_s=round(est_total, 1),
            est_predict_s=round(est_predict, 1),
            est_backtest_s=round(cfg.n_folds * est_total, 1),
            est_forecast_refit_s=round(est_total, 1),
            # frozen: models that can stay frozen only predict (plus a cheap update); others refit
            est_forecast_frozen_s=round(est_predict if model.supports_update else est_total, 1),
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def summary_lines(table: pd.DataFrame, cfg: RunConfig, target_series: int, history: int) -> list[str]:
    ok = table[table["status"] == "ok"]
    fast = ok[ok["cost"] != "slow"]

    def minutes(frame, column):
        return frame[column].astype(float).sum() / 60

    lines = [
        f"Estimates for {target_series} {cfg.frequency} series with {history} periods of history, "
        f"horizon {cfg.horizon}, {cfg.n_folds} backtest folds (one process; minutes):",
        f"  {'model':22} {'backtest':>9} {'fcst refit':>11} {'fcst frozen':>12}",
    ]
    for _, r in ok.sort_values("est_backtest_s", ascending=False).iterrows():
        lines.append(
            f"  {r['model']:22} {r['est_backtest_s'] / 60:9.1f} {r['est_forecast_refit_s'] / 60:11.1f} "
            f"{r['est_forecast_frozen_s'] / 60:12.1f}" + ("   (slow model)" if r["cost"] == "slow" else "")
        )
    lines += [
        f"  {'TOTAL fast models':22} {minutes(fast, 'est_backtest_s'):9.1f} "
        f"{minutes(fast, 'est_forecast_refit_s'):11.1f} {minutes(fast, 'est_forecast_frozen_s'):12.1f}",
    ]
    if len(fast) < len(ok):
        lines.append(
            f"  {'TOTAL all models':22} {minutes(ok, 'est_backtest_s'):9.1f} "
            f"{minutes(ok, 'est_forecast_refit_s'):11.1f} {minutes(ok, 'est_forecast_frozen_s'):12.1f}"
        )
    lines.append("  (forecast columns: all models; a real forecast only runs the models in the selection)")
    failed = table[table["status"] != "ok"]
    for _, r in failed.iterrows():
        lines.append(f"  FAILED {r['model']}: {r['note']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate backtest and forecast times at your data size.")
    parser.add_argument("--frequency", help="yearly | monthly | weekly | daily")
    parser.add_argument("--series", type=int, default=1000, help="number of series to estimate for")
    parser.add_argument("--history", type=int, help="periods of history per series (default by frequency)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 80], help="sample sizes to time")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="time budget per model and sample")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--n-folds", type=int)
    parser.add_argument("--models", nargs="+", help="only these models (plus the baselines)")
    parser.add_argument("--exclude-models", nargs="+")
    parser.add_argument(
        "--fast-only", action=argparse.BooleanOptionalAction, default=None,
        help="--no-fast-only also times Prophet / Orbit",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    overrides = {
        k: v for k, v in vars(args).items()
        if k in ("frequency", "horizon", "n_folds", "models", "exclude_models", "fast_only")
    }
    try:
        cfg = load_config(overrides, config_dir=args.config_dir)
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    history = args.history or DEFAULT_HISTORY.get(cfg.frequency, 60)
    table = run_benchmark(
        cfg, target_series=args.series, history=history, sizes=args.sizes,
        max_seconds=args.max_seconds, config_dir=args.config_dir,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"benchmark_{cfg.frequency}_{datetime.now():%Y%m%d_%H%M}.csv"
    table.to_csv(path, index=False)
    print("\n".join(summary_lines(table, cfg, args.series, history)))
    print(f"  details: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
