"""Simulate earlier forecast runs, to try the report's Accuracy tab "Live" view before real runs exist.

    python -m poetry run python scripts/simulate_past_runs.py --frequency monthly --horizon 18 --back 12 9 6 3

For every number in --back, the input file is cut that many periods before its last date and
`run-pipeline --steps forecast` runs on the cut data, as if the forecast had been made back then.
Each run is recorded in runs/<frequency>/<run id>/ with the cut date as its data cutoff, which is
what the Live view reads. The normal outputs/ folder is not touched (the runs write to a scratch
folder), and neither is registry/: every simulated run uses today's selection.json, which was chosen
with data the earlier runs would not have had, so these runs look somewhat better than real ones.
Afterwards run `report --frequency <frequency>` to rebuild the report with the Live view filled.
Delete the simulated runs from runs/<frequency>/ before going live: their manifest.json names an
input file under runs/_simulated_inputs/ (the cut inputs; runs/_simulated_outputs/ is scratch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gts_forecaster.config import load_config
from gts_forecaster.pipelines.run import main as run_pipeline_main

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate earlier forecast runs on cut input data.")
    parser.add_argument("--frequency", required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--back", type=int, nargs="+", default=[12, 9, 6, 3],
                        help="periods before the last date to cut the data (one run each)")
    args = parser.parse_args(argv)

    cfg = load_config({"frequency": args.frequency})
    data_path = Path(cfg.data_path)
    data = pd.read_csv(data_path, dtype=str)
    dates = pd.to_datetime(data["date"])
    all_dates = sorted(dates.unique())
    inputs = ROOT / "runs" / "_simulated_inputs" / args.frequency
    inputs.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "runs" / "_simulated_outputs"

    failed = 0
    for back in sorted(args.back, reverse=True):  # oldest first, like real runs
        if back < 1 or back >= len(all_dates):
            print(f"skipping --back {back}: the data has {len(all_dates)} dates")
            continue
        cut = pd.Timestamp(all_dates[-1 - back])
        path = inputs / f"{data_path.stem}_cut_{cut:%Y%m%d}.csv"
        # series that are very new at this date get the last-value fallback, as in a real run
        data.loc[dates <= cut].to_csv(path, index=False)
        print(f"\n=== simulated run with data up to {cut:%Y-%m-%d} ({back} periods back) ===")
        argv_run = ["--frequency", args.frequency, "--steps", "forecast",
                    "--data-path", str(path), "--output-dir", str(scratch)]
        if args.horizon:
            argv_run += ["--horizon", str(args.horizon)]
        failed += run_pipeline_main(argv_run) != 0
    print(f"\nDone: {len(args.back) - failed} run(s) recorded in runs/{args.frequency}/. "
          f"Now run: python -m poetry run report --frequency {args.frequency}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
