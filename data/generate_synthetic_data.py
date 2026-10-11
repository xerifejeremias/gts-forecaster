"""Generate synthetic multi-series datasets for GTS Forecaster.

Writes, for each frequency, into data/<freq>/:
  - <freq>_synthetic.csv      long format: uuid, date, target, x_temp, x_promo, x_price
  - series_metadata.csv       one row per series: length group, length and predictability tier

Run from the project root:  python data/generate_synthetic_data.py
"""
import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SERIES_PER_GROUP = 5
LARGE_GROUP_SERIES = 15   # the 2 largest (last 2) groups of each frequency get more series
N_LARGE_GROUPS = 2

# Each frequency has "length groups". A group is either an exact length (int) or an
# open range (lo, hi) from which SERIES_PER_GROUP lengths are spread geometrically
# (strictly inside the range).
FREQS = {
    "yearly":  dict(alias="YS",    period=8,  extra=None, end="2025-01-01",
                    groups=[7, 15, 35, (40, 70)]),
    "monthly": dict(alias="MS",    period=12, extra=None, end="2025-09-01",
                    groups=[7, 15, 35, (50, 140)]),
    "weekly":  dict(alias="W-MON", period=52, extra=None, end="2025-09-29",
                    groups=[7, 15, 35, (52, 104), (104, 208), (208, 502)]),
    "daily":   dict(alias="D",     period=7,  extra=365,  end="2025-10-04",
                    groups=[7, 15, 35, (50, 90), (90, 365), (365, 730), (730, 10000)]),
}
TIER_CYCLE = ["high", "medium", "low"]


def group_label(g):
    return f"{g} obs" if isinstance(g, int) else f"{g[0]}-{g[1]} obs"


def group_lengths(g, k):
    if isinstance(g, int):
        return [g] * k
    lo, hi = g
    return [int(round(x)) for x in np.geomspace(lo, hi, k + 2)[1:-1]]


# noise sd as a multiple of the clean signal's std
NOISE = {"high": 0.10, "medium": 0.70, "low": 3.00}


def make_series(rng, n, period, extra, tier):
    t = np.arange(n)
    amp = rng.uniform(10, 100)
    level = amp * rng.uniform(5, 10)
    trend = amp * rng.uniform(-0.01, 0.03) * (period / 12 if period > 12 else 1)
    phase = rng.uniform(0, 2 * np.pi)

    # explanatory variables
    x_temp = 15 + 8 * np.cos(2 * np.pi * t / period + phase) + rng.normal(0, 1.5, n)
    x_promo = (rng.random(n) < 0.15).astype(int)
    x_price = 10 + np.cumsum(rng.normal(0, 0.15, n))

    signal = level + trend * t + amp * np.cos(2 * np.pi * t / period + phase)
    if extra:
        signal += 0.6 * amp * np.cos(2 * np.pi * t / extra + rng.uniform(0, 2 * np.pi))
    signal += 0.02 * amp * (x_temp - 15)          # weak temperature effect
    signal += 0.5 * amp * x_promo                 # promo uplift
    signal -= 0.05 * amp * (x_price - 10)         # price sensitivity

    noise_sd = NOISE[tier] * signal.std() if n > 1 else 0
    eps = rng.normal(0, noise_sd, n)
    if tier == "low":                              # persistent noise, harder to model
        for i in range(1, n):
            eps[i] += 0.3 * eps[i - 1]
    target = signal + eps
    return target, x_temp, x_promo, x_price


def main():
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)
    for name, cfg in FREQS.items():
        frames, meta = [], []
        for gi, g in enumerate(cfg["groups"]):
            k = LARGE_GROUP_SERIES if gi >= len(cfg["groups"]) - N_LARGE_GROUPS else SERIES_PER_GROUP
            for j, n in enumerate(group_lengths(g, k)):
                tier = TIER_CYCLE[(gi + j) % 3]
                uid = str(uuid.UUID(int=py_rng.getrandbits(128), version=4))
                dates = pd.date_range(end=cfg["end"], periods=n, freq=cfg["alias"])
                y, xt, xp, xpr = make_series(rng, n, cfg["period"], cfg["extra"], tier)
                frames.append(pd.DataFrame({
                    "uuid": uid, "date": dates.strftime("%Y-%m-%d"),
                    "target": y.round(3), "x_temp": xt.round(3),
                    "x_promo": xp, "x_price": xpr.round(3),
                }))
                meta.append(dict(uuid=uid, length_group=group_label(g), n_obs=n,
                                 predictability=tier,
                                 start_date=dates[0].strftime("%Y-%m-%d"),
                                 end_date=dates[-1].strftime("%Y-%m-%d")))
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(out / f"{name}_synthetic.csv", index=False)
        pd.DataFrame(meta).to_csv(out / "series_metadata.csv", index=False)
        print(name, "ok")


if __name__ == "__main__":
    main()
