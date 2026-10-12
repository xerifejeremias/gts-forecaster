"""Fair accuracy check of the saved backtests. Run from the project root:

    python -m poetry run python scripts/holdout_check.py [yearly monthly weekly daily]

For every frequency it reads the backtest file named in registry/<frequency>/selection_fast.json and
replays the pipeline's own model choice (`selection.select_models`) out of sample: for each test fold
k >= 2 the models are chosen on folds < k only (expanding window) and scored on fold k, which the choice
never saw. Both selection rules are compared ("per_series" and "blend", see selection.py), with naive as
the reference. Slow models (Prophet, Orbit) are left out, like the default fast selection.
The scores use the selection's metric (MASE falls back to MAE: the training history is not in the file).
"""

import json
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gts_forecaster.evaluation.metrics import get_metric
from gts_forecaster.models.catalog import load_model_specs
from gts_forecaster.selection import REFERENCE_MODEL, select_models

RULES = ("per_series", "blend")
SLOW = ("prophet", "orbit")


def wape(a, f):
    return np.abs(f - a).sum() / np.abs(a).sum() * 100


def internal(bt):
    """The backtest file in the internal column names select_models expects."""
    return bt.rename(columns={"uuid": "unique_id", "ds": "ds", "actual": "y", "forecast": "yhat"})


def scores_table(pred, metric_fn):
    rows = []
    scored = pred[~pred["is_filled"]]
    for (uid, fold, model), g in scored.groupby(["unique_id", "fold", "model"], sort=False):
        value = metric_fn(g["y"].to_numpy(float), g["yhat"].to_numpy(float), None, 1)
        rows.append({"unique_id": uid, "fold": fold, "model": model, "score": value, "n_points": len(g)})
    return pd.DataFrame(rows)


def check(freq):
    selection_meta = json.load(open(f"registry/{freq}/selection_fast.json"))["meta"]
    metric = selection_meta.get("metric", "mae")
    if metric == "mase":
        metric = "mae"
    top_n = int(selection_meta.get("top_n", 3))
    share = float(selection_meta.get("blend_share", 0.5))
    bt = pd.read_csv(f"backtesting/{freq}/{selection_meta['backtest_forecasts_file']}", dtype={"uuid": str})
    bt = bt[~bt["model"].str.startswith(SLOW)].dropna(subset=["forecast"])
    bt["is_filled"] = bt["is_filled"].astype(str).str.lower().isin(["true", "1"])
    pred = internal(bt)
    pred["ds"] = pd.to_datetime(pred["ds"])
    catalog = [m for m in load_model_specs() if m in set(pred["model"])]
    models = {m: SimpleNamespace(min_train_length=1) for m in catalog}  # order = models.yaml order
    scores = scores_table(pred, get_metric(metric))
    folds = sorted(pred["fold"].unique())
    test_rows = {rule: [] for rule in RULES}
    per_series = []
    for k in folds[1:]:
        choose = [f for f in folds if f < k]
        result = SimpleNamespace(
            metric=metric, season_length=1, scores=scores[scores["fold"].isin(choose)],
            predictions=pred[pred["fold"].isin(choose)], train_y=defaultdict(lambda: None),
            series_lengths=pd.Series(10_000, index=pred["unique_id"].unique()),
        )
        test = pred[(pred["fold"] == k) & ~pred["is_filled"]]
        wide = test.pivot_table(index=["unique_id", "ds"], columns="model", values="yhat")
        actual = test.drop_duplicates(["unique_id", "ds"]).set_index(["unique_id", "ds"])["y"]
        judged = set(scores.loc[(scores["model"] == REFERENCE_MODEL) & scores["fold"].isin(choose), "unique_id"])
        judged &= set(test.loc[test["model"] == REFERENCE_MODEL, "unique_id"])
        chosen = {rule: select_models(result, models, top_n=top_n, rule=rule, blend_share=share).series
                  for rule in RULES}
        for uid in sorted(judged):
            a = actual.loc[uid]
            maes, ok = {}, True
            for rule in RULES:
                members = chosen[rule][uid].models
                w = wide.loc[uid]
                if any(name not in w.columns or w[name].isna().any() for name, _ in members):
                    ok = False
                    break
                f = sum(w[name] * weight for name, weight in members)
                test_rows[rule].append(pd.DataFrame({"a": a, "f": f.reindex(a.index)}))
                maes[rule] = (f.reindex(a.index) - a).abs().mean()
            if ok:
                naive = wide.loc[uid][REFERENCE_MODEL].reindex(a.index)
                test_rows.setdefault("naive", []).append(pd.DataFrame({"a": a, "f": naive}))
                per_series.append(maes)
            else:  # keep the rules on the same points
                for rule in RULES:
                    if len(test_rows[rule]) > len(per_series):
                        test_rows[rule].pop()
    p = pd.DataFrame(per_series)
    w = {k: wape(pd.concat(v).a, pd.concat(v).f) for k, v in test_rows.items()}
    gain = (p["per_series"] - p["blend"]) / p["per_series"] * 100
    print(f"== {freq}: {len(p)} series x test folds (choose on earlier folds, test folds {folds[1:]})")
    print(f"   WAPE  per_series {w['per_series']:.1f}%   blend {w['blend']:.1f}%   naive {w['naive']:.1f}%   "
          f"| blend vs per_series, series x folds >5% better / worse: {(gain > 5).sum()} / {(gain < -5).sum()}")


if __name__ == "__main__":
    for frequency in sys.argv[1:] or ["yearly", "monthly", "weekly", "daily"]:
        check(frequency)
