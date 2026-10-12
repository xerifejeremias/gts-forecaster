# GTS Forecaster

General time series forecasting for many series at once (`uuid`, `date`, `target`, explanatory variables).
Goals, design and decisions are in [OVERVIEW.md](OVERVIEW.md).

## Setup

```
poetry install
```

This creates the virtual environment, installs the packages and writes `poetry.lock`.
**Python 3.12 or 3.13 is required.** If Poetry picks an older Python, point it at a newer one first,
for example `poetry env use 3.12` (or the full path to `python.exe`; `py -0p` lists the installed ones).

The forecasting models come in optional groups. Install the ones you want:

```
poetry install --extras "stats ml foundation"     # StatsForecast, MLForecast (LightGBM, XGBoost, ...), Chronos
poetry install --extras all-models                # also Prophet (Orbit separately: --extras orbit)
```

| Extra | Models |
|---|---|
| `stats` | auto_ets, auto_arima, auto_theta, mstl |
| `ml` | lightgbm, xgboost, random_forest, ridge, lasso (and the `_static` variants) |
| `foundation` | chronos2, chronos2_small, chronos_bolt_small, chronos_bolt_tiny (weights download on first use) |
| `prophet`, `orbit` | prophet, orbit_dlt, orbit_lgt (slow, see below) |

Orbit is compiled when it is installed (it downloads CmdStan and builds C++ code), so on Windows it needs a
C++ toolchain (MinGW / RTools); that is why it is not in `all-models`. Prophet installs without a compiler.

A model whose library is missing is skipped with a warning that names the extra. The baselines (naive,
seasonal_naive, drift) always run.

## Your own data

Set these in `config/default.yaml` (example for data with `ds`, `unique_id`, `y` and market / product
category / company columns; other columns in the file are ignored):

```yaml
data_path: "data/{frequency}/volumes_{frequency}.parquet"   # .csv or .parquet; {frequency} is replaced
input_columns: {id: unique_id, date: ds, target: y}
static_columns: [MARKET_CODE, PRODUCT_CATEGORY_CODE, TMO]
explanatory_columns: []                 # e.g. [price] once there is one
output_rename: {uuid: unique_id, date: ds}   # output columns named like the input
```

- **CSV or Parquet:** chosen from the file suffix (`data_source: auto`), or set `data_source: csv | parquet`.
- **SQL:** `sql_query: "SELECT ds, unique_id, y, MARKET_CODE, PRODUCT_CATEGORY_CODE, TMO FROM volumes_{frequency}"`
  (or `sql_query_file`), and the SQLAlchemy URL in the environment variable `GTS_SQL_URL` (keeps the password
  out of the config; `sql_connection` works too). Install with `--extras sql` plus your database's driver
  (e.g. `psycopg` for PostgreSQL, `pyodbc` for SQL Server).
- **Series ids** can be any text (`id_format: uuid` checks for UUIDs). **Extra columns** are dropped with a log
  line (`extra_columns: error` stops instead).
- **Dates** must sit on the frequency's grid: monthly data on the first of the month (`pandas_freq: MS` in
  `config/frequency_profiles.yaml`). For month-end dates use `pandas_freq: ME` and `calendar.period_anchor: end`.
- **Zeros and negative values** are accepted. Forecasts are floored at 0 (`forecast_floor: 0`; `null` allows
  negative forecasts). Models that need positive values (Tweedie, multiplicative ETS) fall back cleanly.
- **Future values** of explanatory columns (e.g. planned prices): `known_future_columns` and
  `future_features_path` (see OVERVIEW.md, "Explanatory variables in the future").

Check a file before running anything: `poetry run validate-input --frequency monthly`.

## Run

Check an input file against the input contract and repair missing periods:

```
poetry run validate-input --frequency monthly
poetry run validate-input --frequency daily --max-fill-periods 5 --fallback zero
```

In Python:

```python
from gts_forecaster.data_preparation import load_prepared

prepared = load_prepared("weekly")   # or "yearly", "monthly", "daily"
prepared.data      # unique_id, ds, y, explanatory columns, is_filled
prepared.fallback  # series that get a fallback forecast instead of models (long gaps, or fewer
                   # than min_series_length rows: those always get their last value)
prepared.report    # every missing run that was repaired or sent to the fallback
```

Backtest the candidate models and write the selection, then forecast with it:

```
poetry run backtest --frequency monthly --horizon 12 --metric mae --top-n 3
poetry run forecast --frequency monthly --horizon 12
```

Prophet and Orbit are slow, so they are left out by default (`fast_only: true`). Include them with
`--no-fast-only` (on both commands) or `fast_only: false` in `config/default.yaml`. Which models exist is
set in `config/models.yaml`; calendar and identifier features in `config/features.yaml`. Backtest
forecasts versus actuals are saved in `backtesting/<frequency>/`, the selection in `registry/<frequency>/`
and the forecasts in `outputs/<frequency>/`.

### One command for a whole run

```
poetry run run-pipeline --frequency monthly                          # backtest, (train), forecast, report
poetry run run-pipeline --frequency monthly --steps forecast report  # quick run with the saved selection
poetry run run-pipeline --frequency monthly --no-fast-only --steps backtest   # let Prophet/Orbit compete
```

The input file is read and checked once for all steps. `train` is included by default only when
`forecast_mode` is `frozen_models`. If a step fails, the run stops there and exits with code 1.

Every run gets its own folder `runs/<frequency>/<run id>/` (the id is the start time, e.g.
`20261006_101500`), and `runs/<frequency>/latest.txt` names the newest one:

| File | Content |
|---|---|
| `run.log` | everything the steps logged, including errors with their traceback |
| `manifest.json` | status, time and error per step, the full settings, library versions, and a fingerprint of the input file (size, sha256, rows) |
| `config/` | copy of the config files used |
| `outputs/<step>/` | copy of what each step produced (selections, scores, forecasts, `forecast_output.csv`, report) |

The single commands (`backtest`, `train`, `forecast`, `report`) still work as before.

### How long will it take? (timing benchmark)

```
poetry run benchmark --frequency daily --series 3000                    # your real number of series
poetry run benchmark --frequency monthly --series 800 --history 72 --no-fast-only
```

Run it on the computer that will run the pipeline. It times every model on 20 and 80 synthetic
series (`--sizes`) with the given history length and extrapolates to `--series`, printing the
estimated minutes for the backtest, a forecast that refits, and a frozen-models forecast, per model
and in total. Details go to `benchmarks/benchmark_<frequency>_<time>.csv`. `--models` / `--exclude-models`
limit the models, `--max-seconds` (default 600) stops a model that is too slow for the bigger sample.
The estimates are rough (linear extrapolation, one process) but show which models dominate the time.

### Output for the receiving platform

`forecast` writes `outputs/<frequency>/forecast_output.csv` in long format, one row per series and date:
`uuid, date, market, product, company, forecast` (the final forecast only, no prediction intervals). It is
checked against the output contract before it is written. Rename columns with `output_rename` in
`config/default.yaml`, for example `{uuid: unique_id, date: ds, market: market_code}`. Negative forecasts are
set to 0 (`forecast_floor: 0`; this applies to every model in the backtest too, so model choice reflects it).

### Sanity checks of the forecasts

Before writing, `forecast` checks every series' final forecast against its own history and lists what looks
wrong in `outputs/<frequency>/forecast_checks.csv` (also logged): `outlier` (a forecast step far from its
neighbouring steps, in robust standard deviations of the history, like the training outliers), `total_change`
(forecast total more than 3x up or down vs the same number of periods before), `start_jump` (first steps more
than 3x off both the last actuals and the same periods a season earlier) and `zero` (all-zero forecast after
non-zero history). Settings in `config/default.yaml`: `forecast_check_action` (`warn` default: deliver and
warn; `raise`: stop before anything is written; `off`), `forecast_check_threshold` (null = `outlier_threshold`)
and `forecast_check_max_ratio` (3).

### HTML report

A reference copy of how it should look is in `docs/examples/` (`forecast_report_DEMO.html` and screenshots); the details are in OVERVIEW.md, section "Forecast report".

```
poetry run report --frequency monthly
```

Writes `outputs/<frequency>/forecast_report.html`, one file that opens in any browser (no server, no
internet). Actual values and the forecast are drawn as one continuous line (blue = actual, orange =
forecast). Choose a market, product category or company (or any combination; "All" leaves a level
open) and either **Total of selection** (the series that match are added up, history and forecast) or
**Single series** (pick the uuid, or step through them). A slider under the time axis sets how much history is shown: drag right for less (for example the latest two years), left for more; "Show all history" resets it. Hover for values, switch to the table, or
download the CSV of what is shown. The **Forecast lines** dropdown shows the final (ensemble) forecast and, optionally, any number of individual models at once ("Ensemble members" picks the models behind the selected series; in a total, a series without a forecast from a model uses its final forecast for that model's line, and the menu says how many). `--no-model-lines` leaves the per-model lines out for a smaller file. The same data is saved as `forecast_report_data.json` for a future
web app. `forecasts.csv` and `forecasts_by_model.csv` now also carry the identifier columns (market,
product, company ...) on every row. Run `forecast` first; the report never runs a model.

### Accuracy tab in the report

The report has a second tab, **Accuracy**, with forecasts next to the actuals that came true. It uses the
same market / product / company filters, single-series switch and "Forecast lines" model menu as the
Forecast tab.

- **Forecasts:** *Backtest* (out-of-sample forecasts of the newest backtest, every model on every fold)
  or *Live* (forecasts delivered by earlier `run-pipeline` runs whose dates now have actuals).
- **Metric buttons:** MAE, RMSE, MAPE, sMAPE, WAPE (default), Bias (+ = over-forecast).
- **Totals per:** day / week / month / quarter / year (only levels coarser than the data). Actuals and
  forecasts are added up per period, and per selection, before the error is computed, so "Month" on
  daily data compares February's total actuals with February's total forecast.
- **Actuals vs forecast** chart for the chosen lines, **Error over time** (the metric per backtest
  cutoff or per earlier run: a rising line is drift; in Live, dashed lines show the backtest level of
  the same models, so a line clearly above its dashed line says "run the backtest again and refit"),
  and **Error by group**: total, by market, by product, by company or by series (worst first, best line
  in bold).
- Live data builds up over time: only runs made with `run-pipeline` are kept, and only dates whose
  actuals are in the input file count. `--no-accuracy` leaves the tab out.

### Fast and full selections

One backtest scores every model and writes two selections to `registry/<frequency>/`:
`selection_fast.json` (best among the fast models) and `selection.json` (best among all models, including
Prophet and Orbit if they were backtested). `forecast`, `train` and `report` use the fast one by default
and the full one with `--no-fast-only`, so a forecast never needs a model that is switched off. To let
Prophet / Orbit compete, run the backtest once with `--no-fast-only` (slow); forecasts stay fast by default.
A later backtest without `--no-fast-only` updates only `selection_fast.json`: it never replaces a
`selection.json` in which Prophet / Orbit competed. Edit and lock entries in the file you use.

### Frozen models (no refitting at forecast time)

```
poetry run backtest --frequency monthly      # choose the models (selection.json)
poetry run train    --frequency monthly      # fit the selected models once and save them
poetry run forecast --frequency monthly --mode frozen_models
```

`forecast_mode` in `config/default.yaml` (default `frozen_selection`: refit the selected models on every
run) can also be set to `frozen_models`. The saved models go to `fitted_models/<frequency>/`. With newer
data, naive-type models, the machine-learning models (LightGBM, XGBoost, random forest, ridge, lasso) and
Chronos take in the new history without refitting. StatsForecast, Prophet and Orbit models cannot, so they
are refitted at forecast time (a warning says so; the timings file shows `updated` or `refitted`). A model
is also refitted if models.yaml changed since `train`, or if a series was not in the saved model. Run
`train` again after a new backtest, a changed models.yaml / features.yaml, or new series. The data must be
at least as recent as the training data.

### Trial runs and failures

```
poetry run backtest --frequency monthly --max-series 8 --models lightgbm ridge
poetry run backtest --frequency monthly --exclude-models chronos2 --on-model-error raise
```

- `--models a b` runs only those candidates (the baselines the selection needs always run);
  `--exclude-models a b` leaves some out; `--max-series N` uses an evenly spread sample of N series.
  The same keys exist in `config/default.yaml` (`models`, `exclude_models`, `max_series`).
- Such a **partial run** writes its selection to `registry/<frequency>_partial/`, so your real
  `selection.json` is never replaced by a trial.
- `--on-model-error skip` (default): a model that crashes is logged and left out, the others finish;
  `raise` stops at the first crash. `forecast` also accepts `--max-series` and `--on-model-error`
  (a failed selected model is dropped and the remaining weights are rescaled, with a warning).
- Every run writes timings per fold and model (`backtest_timings_<time>.csv`, and
  `outputs/<frequency>/model_timings.csv` for forecasts); failures go to `backtest_failures_<time>.csv`
  / `model_failures.csv`. Progress lines ("fold 2/3 model lightgbm (4/9)") are printed while running.

Run settings live in `config/default.yaml`; frequency specific settings in `config/frequency_profiles.yaml`.

## Test

```
poetry run pytest
```

Tests for the input contract are in `tests/contracts/`, unit tests in `tests/unit/`, and a check that the
files in `data/` pass the contract in `tests/data/`.

## Regenerate the synthetic data

```
poetry run python scripts/generate_synthetic_data.py
```
