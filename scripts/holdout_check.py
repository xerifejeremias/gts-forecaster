"""Fair accuracy check of the saved backtests. Run from the project root:

    python -m poetry run python scripts/holdout_check.py

For every frequency, the models are chosen per series on the earlier backtest folds only (same
rules as selection.py: mean MAE, top 3, equal weights) and scored on the last fold, which the
choice never saw. Compared with naive and seasonal naive, overall and per predictability tier.
Reads registry/<frequency>/selection_fast.json, the backtest forecasts file it names, and
data/<frequency>/series_metadata.csv (tiers of the synthetic data).
"""
import json
import numpy as np, pandas as pd

TOP_N = 3
def wape(a, f): return np.abs(f - a).sum() / np.abs(a).sum() * 100

for freq in ["yearly", "monthly", "weekly", "daily"]:
    meta = json.load(open(f"registry/{freq}/selection_fast.json"))["meta"]
    bt = pd.read_csv(f"backtesting/{freq}/{meta['backtest_forecasts_file']}", dtype={"uuid": str})
    bt = bt[~bt["is_filled"].astype(bool)].dropna(subset=["forecast"])
    tiers = pd.read_csv(f"data/{freq}/series_metadata.csv", dtype={"uuid": str}).set_index("uuid")["predictability"]
    last = bt["fold"].max()
    err = (bt.assign(ae=(bt.forecast - bt.actual).abs())
             .groupby(["uuid", "model", "fold"])["ae"].mean().unstack("fold"))
    order = list(dict.fromkeys(bt["model"]))  # models.yaml order, for ties
    rows, test = [], bt[bt.fold == last]
    for uid, e in err.groupby(level="uuid"):
        e = e.droplevel("uuid")
        if "naive" not in e.index or pd.isna(e.loc["naive", last]):
            continue
        early = [f for f in e.columns if f != last and pd.notna(e.loc["naive", f])]
        if not early:
            continue  # only one fold: cannot choose without looking at the test fold
        cand = e[e[early].notna().all(axis=1) & e[last].notna()]
        mean = cand[early].mean(axis=1).to_frame("s")
        mean["o"] = [order.index(m) for m in mean.index]
        chosen = mean.sort_values(["s", "o"]).index[:TOP_N].tolist()
        t = test[test.uuid == uid].pivot_table(index="ds", columns="model", values="forecast")
        a = test[(test.uuid == uid) & (test.model == "naive")].set_index("ds")["actual"].reindex(t.index)
        rows.append(pd.DataFrame({"uuid": uid, "actual": a, "pipeline": t[chosen].mean(axis=1),
                                  "best1": t[chosen[0]], "naive": t["naive"],
                                  "snaive": t.get("seasonal_naive", pd.Series(np.nan, index=t.index))}))
    r = pd.concat(rows)
    per = r.groupby("uuid").apply(lambda g: pd.Series({
        "p_mae": (g.pipeline - g.actual).abs().mean(), "n_mae": (g.naive - g.actual).abs().mean()}), include_groups=False)
    per["tier"] = tiers.reindex(per.index).values
    print(f"\n== {freq}: {per.shape[0]} series scored on the last fold (cutoff never seen by the choice)")
    print(f"   WAPE  pipeline {wape(r.actual, r.pipeline):5.1f}%   best single {wape(r.actual, r.best1):5.1f}%   "
          f"naive {wape(r.actual, r.naive):5.1f}%   seasonal naive "
          + (f"{wape(r.dropna(subset=['snaive']).actual, r.dropna(subset=['snaive']).snaive):5.1f}%" if r.snaive.notna().any() else "  n/a"))
    print(f"   pipeline beats naive on {(per.p_mae < per.n_mae).mean()*100:.0f}% of series")
    for tier, g in per.groupby("tier"):
        rr = r[r.uuid.isin(g.index)]
        print(f"     {tier:6s} {len(g):3d} series: WAPE pipeline {wape(rr.actual, rr.pipeline):5.1f}% vs naive "
              f"{wape(rr.actual, rr.naive):5.1f}%, beats naive on {(g.p_mae < g.n_mae).mean()*100:3.0f}%")
