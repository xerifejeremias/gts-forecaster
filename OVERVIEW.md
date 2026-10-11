# GTS Forecaster: General Time Series Forecasting

**Status:** Built; first real runs on the laptop done 2026-10-11 (all four frequencies run end to end)
**Created:** 2026-10-05 | **Last updated:** 2026-10-11
**Location:** Desktop\Remote Access Projects\GTS Forecaster

## Goal

A Python tool built on pandas that takes a dataframe holding many time series and runs a full machine learning forecasting pipeline on it, then presents the forecasts in a dashboard.

## Input data

One dataframe in long format, with these columns:

| Column | Role |
|---|---|
| uuid | Identifier of each individual time series |
| date | Timestamp of the observation |
| target | Variable to forecast |
| explanatory variables | Any number of additional columns used as predictors |
| static columns (optional) | Columns that identify a series, such as market, product category or company: one text value per uuid. Listed in `static_columns` in the config (see "Static (identifier) features") |

Each uuid is a separate series, so one dataframe can contain many series.

## Global configuration (decided)

The pipeline is driven by global configs set when running it:

- **Date frequency** of the data (for example daily, weekly, monthly).
- **Forecast horizon**: how many periods ahead to forecast.

## Modelling approach (decided)

Two families of models are supported:

- **Global models**: one model trained on all series together.
- **Local models**: one model trained separately for each uuid.

Backtesting decides what is used for each series. The pipeline picks either:

- the single best model, or
- an **ensemble of the top N models** (for example the 3 best). The ensemble can mix any combination, such as 1 global model and 2 local models, or any other combination, depending on what backtests best for that time series.

Model selection is therefore done per uuid, based on backtest performance.

## Pipeline architecture (agreed 2026-10-05)

Two entry points share the same building blocks.

- **Backtest pipeline**: run a couple of times per year. Evaluates every candidate model per uuid, picks the best model or ensemble per uuid, and saves the selection.
- **Forecast pipeline**: run as often as needed. Loads the saved selection, produces forecasts, and feeds the dashboard.

### Shared building blocks

1. **Config and frequency profile.** One config picks the frequency (data folder), horizon, performance metric, number of backtest folds, ensemble size and forecast mode. A frequency profile holds defaults that depend on frequency (seasonal periods, lag sets, calendar features, minimum series lengths, default horizon). Switching frequency or horizon is a config change only.
2. **Load and validate.** Schema (uuid, date, target, explanatory columns), duplicates, regular dates for the chosen frequency.
3. **Clean.** Missing values and gaps, plus outlier detection and removal (see below). Everything learned here is fitted on training data only.
4. **Split by cutoff.** One function takes a cutoff date and a horizon and returns the training data (up to the cutoff) and, when it exists, the test window (the next `h` periods). The backtest calls it with several cutoffs (rolling origin). The forecast pipeline calls it once with the cutoff at the last observed date, so there is no test window.
5. **Feature engineering.** A function of the cutoff, so nothing after the cutoff leaks into the features. Covers calendar, lag and rolling features, and the explanatory variables.
6. **Models** behind one interface (see below).
7. **Evaluation.** A metric registry; the metric is chosen in the config.

### Model interface and adapters

- Every model exposes `fit` and `predict`, in the spirit of the scikit-learn API (including `get_params` / `set_params`). Because forecasting needs a horizon and future values of explanatory variables, the signature is `fit(train)` and `predict(future)` instead of sklearn's `predict(X)`. `train` is the model-format frame (unique_id, ds, y, explanatory columns, is_filled). `future` has `unique_id`, `ds`, `horizon_step` (1..horizon) and, later, the explanatory columns; `predict` returns `unique_id`, `ds`, `yhat`, one row per row of `future`. The horizon is therefore never an argument of the model. Models also expose `min_train_length` (observations a series needs), `requires_seasonality` and `clone()`; all are in `models/base.py` (`BaseForecaster`). Models are declared by name in `config/models.yaml` (name -> `type` + optional `params`) and built by `models/catalog.py`.
- Libraries with a different API (Prophet, Orbit, and also Nixtla's own libraries) are wrapped with the **adapter pattern** so the pipeline only ever sees the common interface.
- **Nixtla libraries:** MLForecast gives scikit-learn-style regressors (LightGBM, XGBoost, RandomForest, Lasso, Ridge and so on) as forecasting models, and StatsForecast provides classical statistical models.
- Global models fit once on all series; local models fit per uuid.
- Naive baselines (naive, seasonal naive) are always included, both as candidates and as benchmarks.

### Model catalog (decided 2026-10-05; all families built)

Rules: CPU only (no model that needs a GPU), volumes data (no price-specific models), private production data (so backtest leakage from public pretraining is not a concern). Each family is an optional Poetry extra, so installing the project never pulls in everything.

| Family | Models | Default run |
|---|---|---|
| Own baselines | naive, seasonal_naive, drift | yes (built) |
| StatsForecast (`stats` extra) | AutoETS, AutoARIMA, AutoTheta, MSTL | yes |
| MLForecast (`ml` extra) | LightGBM, XGBoost, Ridge, Lasso, RandomForest; lags by MLForecast, calendar/static features from a feature set | yes |
| Pretrained foundation (`foundation` extra, CPU torch) | Chronos-2 (`amazon/chronos-2`, 120M; uses known future covariates), Chronos-2 small (`autogluon/chronos-2-small`, 28M), Chronos-Bolt (tiny/small, univariate, very fast) | yes (zero-shot: `fit` is a no-op) |
| Prophet (`prophet` extra) | Prophet | **no**, only when "fast only" is off |
| Orbit (`orbit` extra) | DLT, LGT | **no**, only when "fast only" is off |

- **"Fast only" switch** (config key, default on): Prophet and Orbit are neither backtested nor run. Turning it off includes them.
- **Considered and left out:** NeuralForecast models (NHITS, NBEATS, TFT, PatchTST: GPU wanted); TimesFM 3.0 (weights carry a non-commercial licence, the production use is commercial); TimesFM 2.5, Toto and Moirai-2 (pin NumPy 1.26 or other old versions, so they conflict with the rest of the stack); TiRex (community licence still to be checked, original version is univariate only); Kronos (financial candlestick data); TinyTimeMixer, Lag-Llama and others (weaker benchmark results). Any of them can be added later as a catalog entry plus an adapter.
- Model weights download once from Hugging Face on first use, so the first foundation run needs internet.

### Explanatory variables in the future (decided 2026-10-05)

- An explanatory variable is **unknown** in the future unless the user lists it. **The default is that none are known**; models then work without explanatory variables in the forecast window.
- Config: `known_future_columns` (a subset of `explanatory_columns`) and `future_features_path` (CSV; also `--future-features` and `--known-columns` on the command line, or a DataFrame passed in code).
- The table has `date`, the known columns, and optionally `uuid` (without `uuid`, the same values apply to all series, for example a promotion calendar). It must cover every series and every forecast date with no missing or infinite values; otherwise the forecast stops with a message naming the first gaps. Extra columns are ignored with a warning.
- The `future` frame passed to `predict` then carries exactly the known columns, and never the unknown ones. In a backtest the known columns' actual values in the test window are used, as they are known in advance by definition.
- Consequence for models that use explanatory variables: they may only use the known columns, in training as well as in the forecast. Unknown columns are not usable for forecasting (they could only be used if forecast separately, which is not planned).
- Built in `features/known_future.py`.

### Features (built 2026-10-05)

Features are **opt-in per model**. Models that generate their own features (Nixtla's MLForecast and StatsForecast, Prophet, Orbit) ask for none. Models that need explicit columns ask for a named **feature set** in `config/models.yaml` (`features: <set name>`); the model class then receives a `FeatureBuilder` through a `feature_builder` constructor argument. A model that does not take that argument and asks for a set is rejected at start-up. Sets are defined in `config/features.yaml`; the shipped sets are `calendar_only` and `calendar_and_history`. Code is in `features/` (`calendar_features.py`, `target.py`, `builder.py`).

- **Builder API:** `FeatureBuilder(feature_set, pandas_freq=..., season_length=...)` with `transform_history(train)` (adds the feature columns to the training frame, same rows) and `transform_future(train, future)` (adds them to the `future` frame from the training data only), `feature_columns`, and `usable_at_step(h)`. Explanatory columns, including the known future ones, pass through unchanged. It refuses to overwrite an existing column.
- **Calendar features** (known for any date, so also in the future): `n_days`, `n_business_days`, `share_business_days`, `n_holidays`, `n_mondays` .. `n_sundays` (monthly), `is_business_day`, `is_holiday`, `day_of_week`, `day_of_month`, `day_of_year`, `is_month_start`, `is_month_end` (daily), `month`, `quarter`, `week_of_year`, and Fourier terms `year_sin_k/year_cos_k` (and `week_sin_k/week_cos_k` for daily data, `k` up to `fourier_order`). `calendar: all` takes everything that exists for the frequency; a named feature that does not exist for the current frequency is skipped, so one set serves yearly, monthly, weekly and daily data. For monthly data, `n_business_days` is the number of Monday to Friday days in the month that are not holidays (January 2024: 23).
- **Business days and holidays:** weekends are Saturday and Sunday. Holidays come from `calendar.extra_holidays` (dates you list) and, optionally, `calendar.holiday_country` (for example `PT`), which needs the optional `holidays` package (`poetry install --extras holidays`). Without a country only the listed dates count.
- **Period anchor:** by default a date marks the **start** of its period (a monthly date 2024-03-01 is March; a weekly date starts a 7-day week). If your dates mark the end of the period, set `calendar.period_anchor: end`. Check this against your real data.
- **Target features:** `lag_k` is the target k periods earlier; `roll_<stat>_<w>` is the mean, std, min, max or median of the `w` values ending `rolling_lag` periods before the row. Lags and windows can be integers or `season` / `2*season` (monthly 12, weekly 52, daily 7). A row never sees its own value or later ones (checked by a test).
- **Multi-step forecasting and leakage:** at forecast step `h` a value from `k` periods earlier is only known when `k >= h`. Features that would need values from inside the forecast window are NaN in `transform_future`, and `usable_at_step(h)` lists the known ones. Models that predict a whole horizon directly should therefore use lags and a `rolling_lag` at least as large as the horizon; recursive models build their own lags and do not need these columns. A test checks that the future features equal the history features wherever they are known.
- Features are computed on the outlier-cleaned training data.

**Static (identifier) features for global models (built 2026-10-05).** The input file can carry columns that identify a series, for example a market code (`DE04`, `PT04`), a product category code (`F0101`, `M0101`) and a company (`company A`). The user lists their names in `static_columns` in `config/default.yaml` (default in `default.yaml`: `[market, product, company]`; use `[]` for data without them. The synthetic data is a full grid: markets DE04, PT04, IT04 × products F0101, M0101 × companies company A, company B, so every combination has several series). The rules: each series (uuid) has exactly one non-empty text value in each static column (the run stops otherwise); the columns are read as text (a code like `0042` keeps its zeros); they are kept apart from the time series, in `PreparedInput.static` (one row per series: `unique_id` + the static columns), so the model data frame and its contract are unchanged. A feature set switches them on in a `static:` section (set `global_static` in `features.yaml` is a ready example), and models get the static table through `build_models_from_config(..., static)`; a feature set that uses static features without any `static_columns` is refused with a clear message. What a set can ask for (code in `features/static.py`):
  - `identifiers`: a static column as a model feature. `encoding: ordinal` gives one integer code per category (`id_market`, stable across folds because the codes come from the full static table); `onehot` gives one 0/1 column per category (for linear models).
  - `code_parts`: pieces of structured codes as new static columns, e.g. `market: [2]` turns `DE04` into `market_p2 = DE`, and `product: [1, 3]` turns `F0101` into `product_p1 = F` and `product_p3 = F01`. These can be used as identifiers and as groups.
  - `group_stats`: statistics of the target for a group of series that share a value, with `by:` listing the groups (a column name, or a list of columns for a combination such as market x product group) and `stats:` among `mean`, `median`, `std`, `n_series`, `rel_level` (series mean / group mean, how big the series is within its group) and `growth` (average relative trend of the group). Columns are named `grp_<group>_<stat>`. Each series counts once in its group.
  - `series_stats`: descriptors of the series itself: `mean`, `std`, `cv`, `length`, `trend` (relative slope per period), `zero_share`, `season_strength`. Columns `ser_<stat>`.
  - All statistics are computed from the training data given to the builder, so in a backtest fold only from that fold's history. They are constant per series, and like any target encoding a group with very few series is close to the series' own history; groups with several series generalise better.
  - Not built: leave-one-out group statistics (excluding the series itself), scaling the target per series for global models, and static features of series that are not in the training data (new series).
- Not built yet: lags of the known explanatory columns, and a model that consumes the features (they arrive with the adapters).

### Outlier handling (built 2026-10-05)

Severe outliers in the **training data** are detected and replaced so forecasts are robust to them. Built in `cleaning/outliers.py`.
- **Method `mad`** (robust, per series, target `y` only): (1) when a series has at least three full seasons away from its ends, the typical offset of each season position is removed, so seasonal peaks are not flagged; (2) each point is compared with the median of its two neighbours on each side, not including itself, which follows trends and smooth cycles; (3) a point is an outlier when that residual is more than `outlier_threshold` robust standard deviations (1.4826 x MAD) away. Default 5, meant for severe outliers only.
- Outliers are found in passes: only the most extreme point of a group of nearby candidates is flagged per pass, then it is replaced and the series is scored again, so the neighbours of one big outlier are not flagged as well.
- What a flagged point is replaced with is the user's choice, `outlier_action` (also `--outlier-action`), so the series stays regular. The original value is kept in the report:
  - `clip` (**default**): the threshold bound that was crossed, that is the expected value plus or minus `outlier_threshold` x the robust spread. A value above the upper bound becomes the upper bound, a value below the lower bound becomes the lower bound. A very high or very low value stays high or low (it is not assumed to be a mistake), but is no longer extreme enough to distort the fit.
  - `interpolate`: linear interpolation between the nearest unflagged neighbours.
  - `median` / `mean`: the median or mean of the series' unflagged values.
- Safeguards: the first and last two points of a series are never flagged (a recent jump cannot be told apart from a level change); periods filled by the missing-data step are never flagged; series with fewer than 8 observations are left alone; if more than `outlier_max_share` (default 10%) of a series would be flagged, the series is left unchanged with a warning, since that looks like a level shift.
- **Applied to training data only**, separately inside every backtest fold (using only that fold's own history, so nothing leaks from the test window) and in the forecast pipeline. **Test windows are scored against the raw actual values** (decided; this closes the open question about scoring).
- Config: `outlier_method` (`mad` or `none`), `outlier_action`, `outlier_threshold`, `outlier_max_share`; command line `--outlier-method`, `--outlier-action`, `--outlier-threshold`.
- Reports for review: `backtesting/<frequency>/backtest_outliers_YYYYMMDD_HHMM.csv` (same timestamp as the backtest forecasts file, with the fold) and `outputs/<frequency>/outliers.csv` for the forecast. Columns: uuid, ds, y_raw, y_clean, score.
- Checked on the synthetic data (which has no injected outliers): 0 points flagged in yearly and monthly, 2 to 3 in weekly and daily. With spikes of 6 standard deviations injected into one series each, 87% to 98% were found.

### Backtest pipeline

For each cutoff: fit all eligible candidates, predict the horizon, score each against the actuals. Then rank per uuid using the **configured metric** (MAE, RMSE, MAPE, sMAPE, MASE, WAPE and so on, switchable in the config), choose the best model or the top-N ensemble for each uuid, and write the registry.

**How the backtest is implemented (built):** `evaluation/backtest.py` runs `n_folds` rolling-origin folds (cutoffs from `splitting.py`). For each fold and model it fits a fresh `clone()` on the series that have at least `min_train_length` observations before the cutoff, predicts the window and scores each series. Periods with `is_filled` are not scored. Metrics (`evaluation/metrics.py`): `mae`, `rmse`, `mape`, `smape`, `wape`, `mase`; lower is better, NaN where undefined (for example MAPE with an actual value of 0). Switch with `metric` in the config or `--metric`.

**How the selection is made (built, `selection.py`):** the reference model is `naive`. Per series, all models compete only on the folds where `naive` was scored, and must have a valid score on every one of them. Models are ranked by mean score (ties go to the earlier model in `models.yaml`). The best `top_n` are chosen: 1 = single best model, more = ensemble with `equal` or `inverse_error` weights. If no model has a valid score, the series gets `naive`. The score file also has an `ensemble` row with the backtest score of the combined forecast.

### Short series

Series too short to backtest or to support local models use a fixed ensemble: **two naive models plus a global model**. **Built so far:** a series with no `naive` fold (too little history to backtest) gets the first `short_series_n` (default 2) models of `short_series_models` (default naive, seasonal_naive, drift) that it has enough data for, equally weighted; the global model is added when global models exist. Seasonal naive needs at least one full season of data (and is left out for yearly data, season length 1); where a series is shorter than that, the second naive model is a simple alternative (to be decided when building, for example drift).

### Registry (saved selection)

- **Two selections from one backtest (decided with the user 2026-10-06).** The backtest scores every model once. From those scores it writes `selection_fast.json` (best models among the fast models only) and `selection.json` (best among every model that was backtested, including Prophet / Orbit when the backtest ran with `--no-fast-only`), each with its `backtest_scores_fast.csv` / `backtest_scores.csv`. A model's score does not depend on which other models compete, so the fast selection needs no second backtest. `forecast`, `train` and `report` read `selection_fast.json` when `fast_only` is on (the default) and `selection.json` when it is off (`registry.selection_file(fast_only)`), so a forecast never needs a model its setting leaves out. If the backtest ran with `fast_only` on, the slow models were not scored: `selection_fast.json` is updated, and `selection.json` is **left untouched when it comes from an earlier backtest that did score them** (`meta.includes_slow_models: true`; decided by the user 2026-10-06, `BacktestOutput.full_selection_kept`); otherwise (no full file yet, or it was also fast-only) it is written equal to the fast one with `includes_slow_models: false`. The summary says which happened. Recommended: run the backtest once with `--no-fast-only`, then forecast fast by default. Locked entries are kept per file. Frozen models are saved per variant too: `fitted_models/<frequency>/fast/` and `.../full/`.
- **`selection.json`**: human-editable, one entry per uuid. Changing the model for a series means editing one name. Model names must exist in the model catalog; the pipeline checks this when loading and reports unknown names. Entries marked `locked` are kept when a new backtest runs, so manual overrides are not overwritten. Planned shape:

```json
{
  "meta": {"frequency": "monthly", "horizon": 12, "metric": "MAE", "backtest_date": "2026-10-05"},
  "series": {
    "<uuid>": {
      "locked": false,
      "models": [
        {"name": "lightgbm_global", "weight": 0.5},
        {"name": "ets_local", "weight": 0.3},
        {"name": "seasonal_naive", "weight": 0.2}
      ]
    }
  }
}
```

- **`backtest_scores` file**: per uuid and model, for context when editing the selection.
- **Fitted model files**: kept separately, since they are binary and not meant to be edited by hand.

### Forecast pipeline and forecast mode

Load the registry, build features at the latest cutoff, predict, write forecasts, then update the dashboard. Config key `forecast_mode` (CLI `--mode`):
- **`frozen_selection`** (default): keep the saved choice of models (selection.json) but refit them on the newest data on each run.
- **`frozen_models`** (built 2026-10-05): `poetry run train` (`pipelines/train.py`) fits only the models named in selection.json on all data and saves them with `models/store.py` to `<fitted_models_dir>/<frequency>/` (`<model>.pkl` + `manifest.json` with data cutoff, settings signature, library versions; pickles are trusted files written by this project only). `forecast --mode frozen_models` loads them and calls `BaseForecaster.update(history)` (no refit) for models with `supports_update`: naive/seasonal_naive/drift, the MLForecast adapters (regressor kept, history passed as `predict(new_df=...)`) and Chronos (zero-shot). Models without it (StatsForecast, Prophet, Orbit), models whose settings signature changed (models.yaml; a changed features.yaml is NOT detected), models that failed to pickle, and series not in the saved model are refitted with a warning (timings status `refitted`; `updated` otherwise). Data older than the training cutoff is an error. The MLForecast `new_df` path and the target-transform behaviour with new data are **tested only against a stand-in**; check them on the first real run (compare `frozen_models` and `frozen_selection` forecasts).

### Dashboard and report

Reads saved outputs only (forecasts, selection, the input file for history), so it never runs models.
- **Static HTML report (built 2026-10-05, first dashboard step)**: `poetry run report` (`pipelines/report.py`, package `report/`). `report/data.py` builds one JSON-ready dictionary (`meta`, `dates`, `n_history`, `series[{id, dims, hist, fc, models, fallback}]`); `report/render.py` embeds it in `report/template.html` (vanilla JS + inline SVG, no external libraries, light/dark theme) and also writes `forecast_report_data.json`. The page lets the user filter by every static column (market, product, company ...) and show either the total of the matching series (summed in the browser, history and forecast) or a single series; the chart draws history and forecast as one continuous line in two colours with a hover tooltip, KPI tiles (last actual, forecast sum, previous-period sum, change), a history-window slider under the x axis (state.start; the y axis rescales to the visible data), a table view and CSV download, and a "Forecast lines" multi-select dropdown (final/ensemble forecast plus any individual models from `forecasts_by_model.csv`, stored per series as `model_fc`; totals substitute the final forecast for series lacking that model; `--no-model-lines` drops them). Fallback series are flagged; totals note when some series lack values in a period. `forecasts.csv` / `forecasts_by_model.csv` now carry the static columns. Converting to a web app later means reusing `forecast_report_data.json` and the template's JS. Series with gaps in the input stay gaps (history comes from the raw input file). Checked in a headless browser (desktop and phone width); the JS has no automated test, only the data builder and renderer do.
- Possible next steps: a comparison between forecast runs (no prediction intervals: decided against). Accuracy views: built 2026-10-07, see "Accuracy tab of the report".

## Deliverables

- Python code (pandas-based) implementing the two pipelines and the shared building blocks.
- A dashboard presenting the forecasts.

## Still to decide while building

- Which candidate models and hyperparameters to keep, the ensemble size N (`top_n`, default 3) and equal vs inverse-error weights: judge on real data with the Accuracy tab.
- Backtest scheme on real data (number of folds, step between cutoffs).
- Final dashboard technology (the static HTML report is built; `forecast_report_data.json` is meant to feed a later web app).
- How often to backtest, train and forecast (drives scheduling and clean-up of `runs/`).
- Whether the outlier threshold of 5 and the 10% safeguard suit the real data (starting values).
- Decided: prediction intervals not wanted (2026-10-05); future values of explanatory variables come from the user (`known_future_columns`, default none).

## Project structure (current)

```
GTS Forecaster/
  OVERVIEW.md        this document (goals and outline)
  scripts/
    generate_synthetic_data.py   (re)creates the synthetic datasets in data/
  data/
    yearly/          input time series with yearly frequency
    monthly/         input time series with monthly frequency
    weekly/          input time series with weekly frequency
    daily/           input time series with daily frequency
    data_viewer.html             self-contained page to browse every series (open in a browser)
  (the project has grown well beyond this sketch: see "Folder layout" and "Build status"; docs/examples/ holds the reference forecast report)
```

Each data folder holds input datasets in the long format described above (uuid, date, target, explanatory variables). More folders (for example for code, configs and outputs) will be added when building starts.

## Code layout and data contracts (proposed 2026-10-05, to be confirmed)

### Environment and running

- Environment and packages managed with **Poetry** (`pyproject.toml` and `poetry.lock` in the project root). Install with `poetry install`; run with `poetry run ...`.
- Heavy or fragile libraries (Prophet, Orbit) go in **optional Poetry dependency groups**, so the core pipeline installs without them. Python version range is pinned to what all chosen libraries support (to verify when dependencies are added).
- All code lives in `.py` files. Pipelines are run from the command line, for example `poetry run backtest --frequency monthly --horizon 12`, through script entry points defined in `pyproject.toml`.
- Dev tools in a dev group: `pytest`, plus a linter/formatter (for example `ruff`).

### Installing the model libraries

Every family is an optional Poetry extra, so the core pipeline installs without them (`poetry install`). Add what you want:

| Extra | Installs | Models it enables |
|---|---|---|
| `stats` | statsforecast | auto_ets, auto_arima, auto_theta, mstl |
| `ml` | mlforecast, lightgbm, xgboost, scikit-learn | lightgbm, lightgbm_static, xgboost, xgboost_static, random_forest, ridge, lasso |
| `foundation` | chronos-forecasting (+ PyTorch) | chronos2, chronos2_small, chronos_bolt_small, chronos_bolt_tiny |
| `prophet` | prophet | prophet (slow) |
| `orbit` | orbit-ml | orbit_dlt, orbit_lgt (slow) |
| `holidays` | holidays | country holidays in the calendar features |
| `all-models` | everything above except `holidays` and `orbit` | all but Orbit (Orbit compiles C++ on install: needs a toolchain on Windows) |

Example: `poetry install --extras "stats ml foundation"`. A model whose library is missing is skipped with a warning that names the extra. Prophet and Orbit only run with `poetry run backtest --frequency monthly --no-fast-only` (the same flag works for `forecast`). `poetry lock` resolves all extras together. Python must be 3.12 or 3.13 (see "Library versions"). Chronos downloads its weights from Hugging Face on first use.

### Library versions (checked against releases and source code, 2026-10-05)

Checked by reading each library's current source/docs (not by running them; the sandbox has no PyPI access). Latest releases at that date: statsforecast 2.1.1 (Jul 2026; 2.1 dropped numba), mlforecast 1.1.0, lightgbm 4.7.0, xgboost 3.4.1, scikit-learn 1.9.1, chronos-forecasting 2.3.2, prophet 1.5.0, orbit-ml 1.1.5.1, pandera 0.34.0, numpy 2.5.3, pandas 3.0.6, holidays 0.105.

- **Python 3.12–3.13** (`python = ">=3.12,<3.14"`): numpy ≥2.3, xgboost 3.4, scikit-learn 1.9, arviz 1.x and orbit-ml 1.1.5.1 need ≥3.12 (or ≥3.11). Poetry only uses releases whose requires-python covers the whole range, so a range starting at 3.10 would quietly lock old versions.
- **pandas < 3**: statsforecast and mlforecast require it. Before lifting it, also change `contracts/input_schema.py` (`datetime64[ns]` and `str` dtypes behave differently in pandas 3).
- **Orbit** is not in `all-models`: it compiles C++ (CmdStan + Stan models) when installed, so on Windows it needs MinGW/RTools. Prophet's Windows wheel includes CmdStan. If `from orbit.models import DLT, LGT` fails after install with an arviz error, pin `arviz<1`.
- **torch** (for Chronos): PyPI has only CPU builds for Windows, so no CUDA download there. torch 2.14 needs `VCRUNTIME140_THREADS.dll`, which comes with the Microsoft Visual C++ Redistributable 14.40 or newer (`winget install Microsoft.VCRedist.2015+.x64`); with an older one every Chronos model fails with `WinError 126 ... shm.dll`.

Adapter fixes made from this check: Chronos-Bolt passes the context by position (the argument was renamed `context` → `inputs`; the old name crashed); Chronos-2 gets `freq=` (otherwise it infers the frequency per series and fails on gaps); AutoETS needs 7 observations (the library refuses fewer); MSTL's minimum history follows the longest period given in `options`; the adapters now log which series StatsForecast silently switched to the naive fallback (`fallback_ids_`); Prophet/Orbit stop with an error when they fail on every series (a setup problem such as missing CmdStan) instead of returning last values labelled as the model; MLForecast checks that each series has exactly the future steps 1..h (X_df rows are matched by position) and refuses date features that clash with other columns. Behaviour to know: in frozen-models mode MLForecast refits the target scaling (`local_standard`) on the new history, only the regressor stays frozen; AutoETS skips seasonal forms when the period is over 24 (weekly data with 52 becomes non-seasonal); **AutoARIMA does the same since 2026-10-11** (decided by the user after the first laptop run: seasonal ARIMA with season 52 ran over 24 minutes on one weekly fold without finishing). Above `AutoARIMAForecaster.max_seasonal_period` (24) it fits a non-seasonal ARIMA and needs only 6 observations; `options: {season_length: 52}` forces the seasonal fit; Orbit LGT refuses negative values and its regressors are being deprecated (prefer DLT with known columns); `n_jobs > 1` for StatsForecast starts new processes on Windows (fine from the poetry commands).

### Folder layout

```
GTS Forecaster/
  pyproject.toml, poetry.lock, README.md, .gitignore
  OVERVIEW.md
  config/
    default.yaml              run settings: frequency, horizon, metric, folds, top N, forecast mode, paths
    frequency_profiles.yaml   per-frequency defaults (seasonal periods, lags, min lengths)
    models.yaml               model catalog: name -> adapter + hyperparameters
  data/                       input data per frequency (existing; includes market, product, company columns)
  scripts/                    generate_synthetic_data.py (writes into data/), first_run.bat, check_report_in_browser.py
  registry/<frequency>/       selection_fast.json + backtest_scores_fast.csv (fast models, default), selection.json +
                              backtest_scores.csv (all models) (<frequency>_partial/ for trial runs)
  fitted_models/<frequency>/fast|full/  models saved by `train` (*.pkl + manifest.json), used by --mode frozen_models
  backtesting/<frequency>/    backtest_forecasts_*.csv, backtest_timings_*.csv, backtest_failures_*.csv
  outputs/<frequency>/        forecasts.csv, forecasts_by_model.csv, model_timings.csv, outliers.csv,
                              forecast_report.html, forecast_report_data.json  (current; no run_id folders yet)
  docs/examples/              forecast_report_DEMO.html + screenshots: reference look of the report
  runs/<frequency>/<run id>/  record of each run-pipeline run: run.log, manifest.json, config/, outputs/<step>/
  benchmarks/                 benchmark_<frequency>_<time>.csv from `poetry run benchmark`
  src/gts_forecaster/
    config.py                 load and validate config
    contracts/                input_schema.py (pandera), frequency_checks.py, output_schema.py,
                              selection_schema.py, errors.py
    io/                       loaders.py, writers.py (atomic writes)
    data_preparation.py       load -> input contract -> missing-data repair -> contract again
    cleaning/                 missing.py, outliers.py
    splitting.py              cutoff-based train/test split (shared)
    features/                 calendar_features.py, target.py (lags, rolling), builder.py (feature sets), known_future.py
    models/                   base.py (common interface), naive.py, catalog.py,
                              mlforecast_adapter.py, statsforecast_adapter.py,
                              prophet_adapter.py, orbit_adapter.py
    evaluation/               metrics.py (metric registry), backtest.py
    selection.py              ranking and ensemble choice per uuid
    registry.py               read/write selection.json and model files
    forecasting.py            produce forecasts from the registry
    report/                   data.py, render.py, template.html: the static HTML report (see "Forecast report")
    models/store.py           save/load fitted models for frozen-models mode
    pipelines/                backtest.py, train.py, forecast.py, report.py, run.py (run-pipeline: all steps + run record)
  tests/
    conftest.py, fixtures/    small hand-made datasets, valid and invalid
    contracts/                input, output and selection contract tests
    unit/                     mirrors src/gts_forecaster/
    integration/              small end-to-end backtest and forecast runs
    data/                     checks the files in data/ against the input contract
```

### Input data contract (strict)

Checked in two steps. Any violation raises a `DataContractError` and the pipeline stops with no output written.

**Step 1: structure, checked right after loading.** These are never repaired.

- Exactly the expected columns: `uuid`, `date`, `target`, plus the explanatory columns and the static columns declared in the config. No missing and no unexpected columns.
- Static columns: text, no empty values, and exactly one value per uuid (checked in plain pandas, `contracts/static_checks.py`).
- Correct types: `uuid` is a valid UUID string, `date` is a date without time zone, `target` and explanatory variables are numeric.
- No nulls in `uuid` or `date`, and no infinite values in any column.
- No duplicate (`uuid`, `date`) rows.
- Dates are aligned to the chosen frequency (for example month starts for monthly).
- Data is not empty. A series shorter than the minimum length in the frequency profile no longer stops the run (changed 2026-10-11): it gets the last observed value as its forecast, with a warning that names it (see "Too-short series get the last value").

**Step 2: missing periods, repaired by the rules below, then checked again.** A period counts as missing if its row is absent or its `target` or an explanatory variable is null. After the repair the data must be fully valid: no nulls, no gaps in any series, and all series ending on the same date (the common cutoff, which is the latest date in the data).

### Missing data handling

All thresholds are per run of consecutive missing periods in one series. Defaults: `max_fill_periods = 3` and `fallback = "last_value"`, both settable as function arguments and in the config.

| Where the missing run is | Run is `max_fill_periods` or shorter | Run is longer |
|---|---|---|
| At the end of the series | Forward fill the target and explanatory variables up to the common cutoff | Fallback forecast (below) |
| In the middle of the series | Linear interpolation | Fallback forecast (below) |

**Fallback forecast.** The series is not given to any model. Its forecast for every horizon step is the **last observed value** (`fallback = "last_value"`, the default) or **0** (`fallback = "zero"`). The last observed value is the last real value, never a filled one.

For every affected uuid the pipeline prints and logs a warning that says which dates are missing (first and last missing date and how many periods), what was done (forward fill, interpolation, or fallback), and, for the fallback, why the models were skipped and which fallback value is used.

Filled and interpolated rows are marked in the cleaning report, and an `is_filled` flag is kept internally so that backtest scoring can ignore them instead of treating them as real actuals. Series that use the fallback are left out of the backtest and are not listed in `selection.json`, because the decision depends on the current state of the data, not on a backtest. Points removed as outliers are interpolated by the outlier step and do not trigger the fallback.

### Output (forecast) contract (decided and built 2026-10-05)

Decisions (user, 2026-10-05): **no prediction intervals**, point forecasts only. The receiving platform gets a **long-format** table, not the wide per-model table.

`poetry run forecast` writes `outputs/<frequency>/forecast_output.csv` (name: config `output_file`), one row per series and forecast date:

```
uuid, date, market, product, company, forecast
```

- The identifier columns are the configured `static_columns`, in that order, with the values from the input file. `forecast` is the final forecast (the weighted ensemble from `selection.json`, or the fallback value). Dates are written as `YYYY-MM-DD`.
- The platform's exact column names are not confirmed yet ("UUID or unique_id", "ds or date", "Market_code" ...). Change them without code via `output_rename` in `config/default.yaml`, e.g. `{uuid: unique_id, date: ds, market: market_code, product: product_code}`.
- Code: `output.py` (`build_output`, `write_output`, the post-processor) and `contracts/output_schema.py` (`validate_output`, plain pandas). Checks: exact columns and order; no empty values; numeric finite forecast; no duplicated (uuid, date); every expected series (model and fallback series) present and nothing extra; each series has exactly the horizon dates; identifier values constant within a series. Any breach is a DataContractError and nothing is written. Tests: `tests/unit/test_output.py`.
- Internal files stay as they were and are not for the platform: `forecasts_by_model.csv` (wide, one column per model) and `forecasts.csv` (ds, uuid, forecast + identifier columns), used by the report.
- **Negative forecasts become 0** (decided by the user 2026-10-05, volume data). Config `forecast_floor: 0` (`null` switches it off). Applied to every model's predictions in the backtest (`run_backtest(forecast_floor=...)`, so the scores and the model choice judge what is really delivered), to every model column in `forecast_by_model`, and to the final forecast in `combine_forecasts` (which also covers fallback values). The output contract rejects any value below the floor. Tests: `tests/unit/test_forecast_floor.py`.

**Internal result shape.** Step 1 gives a wide table `unique_id, ds, <one column per model>`. Step 2 reduces it to `ds, unique_id, forecast`: for each series, the weighted average of the models listed for it in `selection.json`; a fallback series gets its fallback value for every horizon date. Model names in `selection.json` are exactly the column names of the wide table. Step 3 (`output.py`) produces the platform file above.

### Selection contract

`selection.json` is checked on load: every uuid covered, model names exist in the catalog, weights are positive and sum to 1.

### Validation library

Proposed: **pandera** for the dataframe contracts, in strict mode, plus custom checks for the rules above. The tests in `tests/contracts/` cover both valid and invalid fixtures, so each rule has a test that proves it can fail.

## Production run and run records (built 2026-10-06)

`poetry run run-pipeline --frequency <f> [--steps backtest train forecast report]` (`pipelines/run.py`, `run_pipeline(cfg, steps=..., prepare=...)`) runs the steps in pipeline order with one input preparation shared by all steps. Default steps: all, `train` only when `forecast_mode` is `frozen_models`. A failing step (including the input check) is recorded, the later steps are marked `skipped`, the run's status is `failed` and the exit code 1; `run_pipeline` itself does not raise for step failures. Each run writes `<runs_dir>/<frequency>/<run id>/` (config `runs_dir`, default `runs`; run id `YYYYMMDD_HHMMSS`, suffix `_1` if two runs start in the same second): `run.log` (file handler on the root logger for the run), `manifest.json` (written after every step; run id, status, steps with seconds/error/output paths, resolved config, fast_only, forecast_mode, partial flag, input fingerprint with sha256 and line count, number of model series and data cutoff, Python/platform/library versions), `config/` (copy of the config folder) and `outputs/<step>/` (copies of the files each step produced; fitted model pickles are not copied, only the train manifest). `<runs_dir>/<frequency>/latest.txt` holds the newest run id. The normal folders (registry/, outputs/, backtesting/, fitted_models/) are still written as before. Tests: `tests/unit/test_run_pipeline.py`. Not done: deleting old runs (they accumulate), scheduling.

## Accuracy tab of the report (built 2026-10-07)

Requested by the user: metric buttons, model multi-select, Actuals vs Forecast chart, totals per coarser period (week / month / quarter / year: cumulative error of the whole period), errors by market / product / company, and error over time to see drift and know when to backtest and refit again.

- Data (`report/accuracy.py`, added to the report payload as `accuracy`; `--no-accuracy` / `run_report_pipeline(accuracy=False)` leaves it out): **backtest** rows from the backtest forecasts file named in the selection's meta (`backtest_forecasts_file`, written by the backtest since 2026-10-07; otherwise the newest `backtest_forecasts_*.csv`), pivoted to one row per (uuid, cutoff, ds) with a column per model, filled-in periods left out, and the final forecast rebuilt from the selection weights; **live** rows from `runs/<frequency>/<id>/outputs/forecast/forecasts.csv` (final) and `forecasts_by_model.csv` of earlier run-pipeline runs, matched with actuals from the current input file; origin = the run's data cutoff (`manifest.input.data_cutoff`; newest run wins per cutoff). Columnar encoding (`encode`) with series indexes into the report's series list; above ~3M values only the models used in the selection (+ naive) are kept (`note`).
- Page (`template.html`, tab "Accuracy"; filters, single series and the "Forecast lines" menu are shared with the Forecast tab): metrics MAE, RMSE, MAPE (actual 0 left out), sMAPE, WAPE (default), Bias (Σ(f−a)/Σ|a|, + = over-forecast). For every view, actuals and forecasts are first summed per cell (group × origin × period bucket over the selected series) and the metric is computed over those cells: errors of totals, not averages of per-series errors. A model without a forecast for a row (e.g. too little history) uses the final forecast there, and a note says how often. Chart 1: actuals vs forecasts per period (newest origin per date when "All folds/runs"); chart 2: metric per origin (drift), in Live with dashed lines at the backtest level of the same models; table: Total / by each static column / by series (top 300, worst first, best line bold). Week buckets start on Monday; for weekly data, months are assigned by the week's start date.
- Caveat shown in the page: in the backtest the final forecast uses the current selection, chosen on the same folds, so it looks better than it will live.
- Tests: `tests/unit/test_report_accuracy.py`; the JS was checked in a headless browser (desktop and phone width, all buttons) with simulated earlier runs; `scripts/check_report_in_browser.py` now also clicks through the Accuracy tab.

## Timing benchmark (built 2026-10-06)

`poetry run benchmark --frequency <f> --series <N> [--history H] [--sizes 20 80] [--no-fast-only]` (`src/gts_forecaster/benchmark.py`). Builds synthetic series (trend, seasonality, noise; the configured explanatory and static columns; ids `bench-NNNNNN`), builds the models from models.yaml exactly like the pipelines (so the catalog filters and fast_only apply), and for each model does an untimed warm-up on 2 series (library import, Chronos weight download), then times fit and predict on each sample size. The time is extrapolated linearly (fixed cost + cost per series) to N: backtest = n_folds × (fit + predict); forecast refit = fit + predict; forecast frozen = predict only for models with `supports_update`, otherwise refit. Prints a per-model table (minutes) and totals for the fast models and all models; writes `benchmarks/benchmark_<f>_<time>.csv`. A failing model is recorded (status failed) and the rest continue; `--max-seconds` stops a model after the small sample. Purpose: decide on the laptop, before the first real backtest, which models are affordable at the real scale (daily data with many series is the concern: per-series StatsForecast and Chronos on CPU). Tests: `tests/unit/test_benchmark.py`. The estimates were only checked with stand-in libraries in the sandbox.

## Forecast report (static HTML dashboard) — read this before touching it

**What it is.** One self-contained HTML file per frequency that shows actuals and forecasts interactively. It is the first dashboard step; the user likes the current look and behaviour, so keep them when changing it. It needs no server and no internet, and it never runs a model.

**Where it lives.**
- Generate it: `poetry run report --frequency monthly` (after `poetry run forecast`). Output: `outputs/<frequency>/forecast_report.html` and `forecast_report_data.json` (the same data, for a future web app). `--no-model-lines` makes a smaller file without the per-model lines.
- Code: `src/gts_forecaster/pipelines/report.py` (CLI, reads files), `src/gts_forecaster/report/data.py` (builds the data dictionary), `src/gts_forecaster/report/render.py` (embeds it in the template), `src/gts_forecaster/report/template.html` (**the whole UI**: CSS + vanilla JS + inline SVG; no libraries, no CDN). The data is injected at the placeholder `/*__GTS_REPORT_DATA__*/null`.
- Inputs it reads: the input CSV (history, incl. the static columns), `outputs/<frequency>/forecasts.csv` (needs the static columns, which `forecast` writes), `forecasts_by_model.csv` (model lines, optional), `registry/<frequency>/selection.json` (model names and weights, optional; series with a forecast but no selection entry are flagged as fallback series).
- Tests: `tests/unit/test_report.py` (data builder, renderer, pipeline). The JavaScript has no unit tests: after changing `template.html` run `scripts/check_report_in_browser.py` (headless Chromium; reports JS errors and sideways scrolling at phone width and saves screenshots).
- **Reference of the intended look:** `docs/examples/forecast_report_DEMO.html` (open it in a browser) and `docs/examples/shots/*.png` (1-3 Forecast tab, 4-5 Accuracy tab). Regenerated 2026-10-07 from the synthetic monthly data with three simulated earlier runs (data cut 9, 6 and 3 months earlier) so the Accuracy tab's Live view has data; only naive / seasonal_naive / drift are real models in it, the other lines come from test stand-ins, so their shapes and errors mean nothing.

**What the page does.**
- Filters for every static column (market, product, company ...; "All" leaves a level open) and a switch between *Total of selection* (the matching series are summed in the browser, history and forecast) and *Single series* (select by uuid or step with previous/next). Totals note when some series lack values in a period (hover shows "n of N series") and when fallback series are included.
- One continuous line: blue = actual, orange = forecast, joined at the last actual value, with a dashed divider and a shaded forecast area. Hover tooltip, KPI tiles (last actual, forecast sum, previous-period sum, change), Table view, CSV download of what is shown, light/dark/auto theme.
- **Forecast lines** multi-select dropdown: the final (ensemble) forecast, plus any number of individual models at once (colours from the categorical palette, dashes after six). "Ensemble members" selects the models behind the chosen series. In a total, a series without a forecast from a model uses its final forecast in that model's line (coverage shown in the menu and in a note).
- **History slider** under the x axis: drag right to show less history, left for more; the y axis rescales to the visible data; "Show all history" resets. The time axis also starts at the series' first value.

**Accuracy tab:** see "Accuracy tab of the report" (payload key `accuracy`).

**Data contract (`forecast_report_data.json`).** `{meta: {frequency, horizon, cutoff, generated, metric, forecast_mode, dimensions, n_series, n_fallback, model_names}, dates: [ISO dates, history then forecast], n_history, series: [{id, dims: {market, ...}, hist: [value|null per history date], fc: [final forecast per forecast date], models: [{name, weight}], fallback, model_fc: {model: [values]}}]}`. NaN is written as null. A future web app should read this file instead of re-deriving anything.

**Design notes.** Colours are CSS tokens with light and dark values (palette from the dataviz skill: blue `#2a78d6` / orange `#eb6834` for actual / forecast, six more for models). Layout works down to phone width (checked at 390 px). Not done / ideas (no prediction intervals: decided against; accuracy views built, see "Accuracy tab of the report"): quarter- or year-aligned x ticks, comparing two forecast runs, a web-app version (reuse the JSON and the JS in the template).

## Synthetic example data (generated 2026-10-05, regenerated twice: new length groups, then 15 series in the 2 largest groups)

Each frequency folder contains `<freq>_synthetic.csv` (long format: `uuid`, `date`, `target`, `x_temp`, `x_promo`, `x_price`) and `series_metadata.csv` (uuid, length_group, n_obs, predictability tier, start/end date). Reproducible with a fixed seed via `scripts/generate_synthetic_data.py`.

**Length groups.** The history length differs by frequency to cover a wide spectrum of training data size. Each group has 5 series, except the 2 largest groups of every frequency, which have 15. Ranges are spread geometrically inside the range; the exact-length groups use that exact length for all their series.

| Frequency | Groups (observations) | Series |
|---|---|---|
| yearly | 7, 15, 35 (15 series), 40-70 (15 series) | 40 |
| monthly | 7, 15, 35 (15 series), 50-140 (15 series) | 40 |
| weekly | 7, 15, 35, 52-104, 104-208 (15 series), 208-502 (15 series) | 50 |
| daily | 7, 15, 35, 50-90, 90-365, 365-730 (15 series), 730-10000 (15 series) | 55 |

**Predictability tiers** rotate across series so every group has a mix: **high** (almost all signal, cosine seasonality plus trend), **medium**, **low** (mostly noise with persistent noise, hard to detect a pattern).

- Explanatory variables: `x_temp` (seasonal, continuous), `x_promo` (binary, uplift on the target), `x_price` (random walk, negative effect on the target).
- All series in a file end on the same date, so shorter series have later start dates. No missing values. (The pipeline itself accepts series that end early, see "Series that end early".)
- Seasonal periods used: yearly 8, monthly 12, weekly 52, daily 7 plus 365.

## Data viewer

`data/data_viewer.html` is a single self-contained page (data embedded, no server needed). It shows a grid of mini charts for every series per frequency, grouped by length group and filterable by signal level. Clicking a series opens a full chart with hover values, drag-to-zoom, optional explanatory-variable panels and a table of rows. It embeds a snapshot of the data, so regenerate it if the CSVs change.

## Build status

Built on 2026-10-05:
- `pyproject.toml` (Poetry), `config/default.yaml`, `config/frequency_profiles.yaml`, `config.py`.
- Input contract: `contracts/input_schema.py` (pandera, structure) and `contracts/frequency_checks.py` (alignment, minimum length, completeness after repair).
- Missing data handling: `cleaning/missing.py`.
- Loader: `io/loaders.py`; full input path: `data_preparation.py`, runnable as `poetry run validate-input`. In Python: `load_prepared("weekly")` returns the repaired data for one frequency.
- Column naming (`columns.py`): input files use `uuid`, `date`, `target`; once the input contract has passed, everything is renamed to the Nixtla names `unique_id`, `ds`, `y` (explanatory columns keep their names). The prepared frame has exactly: `unique_id`, `ds`, `y`, the explanatory columns, `is_filled`, and is checked by `check_model_frame` (types, no nulls, no duplicates, sorted).
- Cutoff-based split (`splitting.py`), shared by both pipelines. The horizon is always an argument (normally `config.horizon`), with no default and no link to a frequency. `forecast_split(...)`: cutoff at the last date, no test window, plus a `future` table of the dates to forecast. `backtest_splits(...)`: rolling-origin folds, oldest first, with `n_folds`, `step` (default: the horizon, so test windows do not overlap) and `min_train_periods`. Each `Split` has `train`, `test` (with `horizon_step`), `future`, `horizon_dates`, `skipped_ids` (series with no data before the cutoff) and `train_lengths`.
- Model interface and naive models: `models/base.py`, `models/naive.py` (naive, seasonal_naive, drift), `models/catalog.py`, `config/models.yaml`.
- Backtest and selection: `evaluation/metrics.py`, `evaluation/backtest.py`, `selection.py`, `contracts/selection_schema.py`, `registry.py`, and the entry point `pipelines/backtest.py`: `poetry run backtest --frequency monthly --horizon 12 --metric mae --top-n 3`. It writes `registry/<frequency>/selection.json` and `backtest_scores.csv` (columns uuid, model, n_folds, score, rank, weight, note). Entries with `"locked": true` in an existing `selection.json` are kept on the next backtest. An invalid existing `selection.json` stops the run. New config keys: `metric`, `n_folds`, `backtest_step`, `top_n`, `ensemble_weights`, `short_series_models`, `short_series_n`, `registry_dir`; the frequency profiles gained `season_length` (yearly 1, monthly 12, weekly 52, daily 7).
- Known future explanatory variables: `features/known_future.py`, config keys `known_future_columns` and `future_features_path`, wired into both pipelines (see "Explanatory variables in the future").
- Features: `features/` (calendar, lags, rolling statistics, feature sets in `config/features.yaml`), connected to the model catalog (see "Features").
- Outlier handling: `cleaning/outliers.py`, wired into the backtest (per fold) and the forecast pipeline (see "Outlier handling").
- Backtest forecasts vs actuals: every backtest run also writes `backtesting/<frequency>/backtest_forecasts_YYYYMMDD_HHMM.csv` (folder set by `backtesting_dir`, `--backtesting-dir`; no colon in the time because Windows does not allow it). One row per series, model, fold and date, columns `uuid, model, fold, cutoff, ds, horizon_step, actual, forecast, is_filled`, so what any model forecast for any backtest window (for example June 2024) can be filtered and plotted. Every backtest run writes a new file and never overwrites an older one (two runs in the same minute get a `_2` suffix). `selection.json` and `backtest_scores.csv` are still overwritten, since the forecast pipeline reads the current selection. Every model added later appears in it automatically.
- Forecast layer: `forecasting.py` (`forecast_by_model`, `combine_forecasts`) and `pipelines/forecast.py`: `poetry run forecast --frequency monthly --horizon 12`. Frozen-selection mode: refits every model on all data, reads `selection.json` (it must cover every series; otherwise the run stops) and writes `outputs/<frequency>/forecasts_by_model.csv` (uuid, ds, one column per model) and `forecasts.csv` (ds, uuid, forecast). The horizon may differ from the backtest horizon (a warning is logged).
- StatsForecast adapters (built 2026-10-05): `models/statsforecast_adapter.py` with `auto_ets`, `auto_arima`, `auto_theta`, `mstl` (local models). The library is imported only at fit time (optional extra `stats`: `poetry install --extras stats`); a model whose library is missing is skipped with a warning that names the extra. Forecast step h is matched to the row with the same `horizon_step` and the dates are checked (a freq mismatch is an error, not a shifted forecast). Model params: `options` (extra keyword arguments for the library model, e.g. `{model: ZZZ}`), `n_jobs` (default 1), `fallback` (naive forecast for a series the library fails on, default true), `min_length`. Default minimum history: seasonal models need two full seasons (monthly 24), non-seasonal at least 4 (ARIMA 6); `mstl` is skipped when there is no seasonal period. The catalog passes `season_length` and `freq` from the frequency profile. **Tested only against a stand-in for the library** (it is not installed in the sandbox), so the first real run may show API differences.
- MLForecast adapters (built 2026-10-05): `models/mlforecast_adapter.py` with `lightgbm`, `xgboost`, `random_forest`, `ridge`, `lasso` (global models: one regressor for all series; extra `ml`: `poetry install --extras ml`). MLForecast builds the lags and rolling windows of the target and forecasts recursively. Params: `lags` (default 1, 2, 3, season), `rolling_windows` (3, season; end at lag 1), `rolling_stats` (mean, std; also min, max), `date_features`, `target_transforms` (default `local_standard`: each series is scaled so one model can serve series with different levels; also `local_minmax`, `difference`, `seasonal_difference`), `regressor_params` (over the defaults: boosting 200 trees, learning rate 0.05, one thread, fixed seed), `num_threads`, `min_length`. A series needs `max(lags, longest window + 1) + 3` observations (monthly 16). `features:` in models.yaml gives a feature set that adds calendar features (known in the future) and static identifier features (constant per series); the set must not define lags (refused). Explanatory columns listed in `known_future_columns` are used automatically (their future values are required); other explanatory columns are ignored. Ridge and Lasso run on standardised features with median imputation. Default models.yaml entries: `lightgbm`, `xgboost`, `random_forest`, `ridge`, `lasso` (feature set `calendar_only`), plus `lightgbm_static` and `xgboost_static` (feature set `global_static`, with `skip_without_static: true`: left out with a warning when the input has no `static_columns`; without that key a static set without static columns is an error). All series in a prediction must have the same number of future steps when calendar or known columns are used. The shared date matching for adapters is in `models/alignment.py`. **Tested only against a stand-in for mlforecast, lightgbm and xgboost** (scikit-learn is real); the first real run may show API differences.
- Chronos adapters (built 2026-10-05): `models/chronos_adapter.py` with `chronos2` (Chronos-2, `amazon/chronos-2`, 120M) and `chronos_bolt` (univariate, fast); models.yaml entries `chronos2`, `chronos2_small` (`autogluon/chronos-2-small`, 28M), `chronos_bolt_small`, `chronos_bolt_tiny`. Zero-shot: `fit` stores the last `max_context` (default 1024) observations of each series and loads the network; there is no training. Extra `foundation` (`poetry install --extras foundation`, package `chronos-forecasting` + PyTorch). CPU by default (`device_map: cpu`). The weights download from Hugging Face on first use (internet needed once) and the loaded network is cached across backtest folds and clones. The point forecast is the median. Chronos-2 sends the columns in `known_future_columns` as future covariates (all series need the same number of future steps then) and, with `use_past_covariates: true`, the other explanatory columns as past-only covariates (default off); Chronos-Bolt ignores explanatory columns. A series needs `min_length` (default 5) observations. **Tested only against a stand-in for the chronos library and PyTorch**; the real call signatures (`Chronos2Pipeline.predict_df`, `BaseChronosPipeline.predict_quantiles`) were taken from the library's README and may differ in detail.
- Prophet and Orbit adapters (built 2026-10-05): `models/bayesian_adapters.py` with `prophet`, `orbit_dlt`, `orbit_lgt`. Per-series models marked `cost = "slow"`, so they run only with `--no-fast-only` / `fast_only: false` (the models.yaml entries stay in place). One model is fitted per series in `fit`. If the library fails on a series that series gets its last value, with a warning naming it (`fallback`, default true; `fallback: false` raises instead). Known columns (`known_future_columns`) are regressors. Prophet seasonalities follow the frequency (none for yearly; yearly for monthly and weekly; yearly and weekly for daily); override with `options`. Orbit uses `stan-map` with no bootstrap draws (point forecasts), seed 8888, `seasonality = season_length` (off when there is none); override with `options`. Default minimum history: 10 observations or two seasons. Extras `prophet` (`prophet`) and `orbit` (`orbit-ml`). **Tested only against stand-ins for both libraries**; the Orbit calls follow its DLT tutorial (`predict(df)` returns `prediction`).
- Run controls (built 2026-10-05): config keys `models`, `exclude_models` (`filter_specs` in `models/catalog.py`; `naive` and the short-series models are always kept, unknown names are a ConfigError), `max_series` (`data_preparation.limit_series`: even sample by length then id, deterministic), `on_model_error` (`skip` default | `raise`). `run_backtest` isolates a crashing model per fold (`BacktestResult.failures`, `.timings`); `forecast_by_model` does the same (NaN column) and `combine_forecasts(drop_missing=True)` drops a failed selected model and rescales the weights (error if none is left). A partial run (`cfg.is_partial_run`) writes its registry to `registry/<frequency>_partial/`. Timings/failures CSVs are written next to the backtest forecasts and in `outputs/<frequency>/`. The forecast pipeline ignores the `models`/`exclude_models` filter because the saved selection decides.
- `fast_only` (config key, default true, CLI `--no-fast-only`): models whose class has `cost = "slow"` (Prophet, Orbit when added) are left out of the backtest and the forecast. `BaseForecaster` also has `requires_packages` / `install_extra` for the missing-library skip.
- Tests in `tests/` for all of the above, including a test that the files in `data/` pass the contract.

Notes:
- The profile values `min_series_length` (3) and `default_horizon` (yearly 3, monthly 12, weekly 12, daily 14) are starting values to review.
- Linear interpolation is applied to every value column, including the binary `x_promo`, which can produce fractions such as 0.5 in a filled gap.
- Which series are eligible for which model in a given fold (a series that starts late has little training data in early folds) is left to the model interface and the backtest.
- Early backtest folds skip series that do not have enough history yet, so a series may be scored on fewer folds than others; the number of folds used is in `backtest_scores.csv` (`n_folds`).
- Without the `stats` extra installed only the baselines run. Install it to get the StatsForecast models in the selection.
- Built later the same day: run controls, frozen-models mode (`train`), the static HTML report (see "Forecast report"). Not yet built: a final dashboard technology (web app), accuracy views.

## First real run on the laptop (2026-10-11)

Windows 10, Python 3.13.0, Poetry 2.1.1 (run as `python -m poetry`, it is not on PATH), 4 logical / 2
physical cores. The environment is in the project (`.venv`, Poetry `virtualenvs.in-project`).
- `poetry lock` and `poetry install --extras "stats ml foundation"` worked first time (statsforecast 2.1.1,
  mlforecast 1.1.0, lightgbm 4.7.0, xgboost 3.4.1, scikit-learn 1.9.1, chronos-forecasting 2.3.2,
  torch 2.14.1+cpu, pandera 0.34.1, pandas 2.x). `poetry.lock` is now in the repo.
- Tests: 572 of 575 passed at once. The 3 failures were tests that assumed the ML libraries were not
  installed (they built the whole catalog); fixed by limiting them to the models they test. Full suite
  ~3.5 min.
- The real StatsForecast, MLForecast and Chronos APIs matched the adapters: no adapter change was needed.
  Frozen-models forecasts equal the refit forecasts on the same data (monthly).
- Chronos needed a newer Visual C++ Redistributable (see "Library versions").
- Pipeline times (`run-pipeline`, synthetic data, all 18 fast models): yearly 1.8 min, monthly 1.8 min
  (before Chronos worked), weekly 4.5 min, daily 23.5 min (backtest 18 min, forecast 5.5 min). Daily: AutoARIMA took 621 s of the 1021 s backtest model time (61%) but is in only 4 of 55 selections (ranked first in 2); Chronos-2 is in 28 (synthetic data, so only indicative). Without AutoARIMA (skip_frequencies, below) daily takes 9.1 min (backtest 6.8, forecast 2.2) with the same hold-out accuracy (WAPE 22.8% either way).
- **Decisions made with the user on 2026-10-11:**
  - AutoARIMA fits a non-seasonal ARIMA when the season is over 24 (weekly): seasonal ARIMA with season
    52 had not finished one fold after 24 minutes (see "Library versions").
  - `max_history_years` (StatsForecast adapters, per frequency): AutoARIMA `{daily: 3}`, AutoTheta
    `{daily: 5}` in models.yaml. All other models and frequencies use the whole series. Measured on long
    daily series: Theta 34 s -> 2 s for 3 series; ARIMA 36 s -> 12 s. Still, ARIMA is the slowest daily
    model (about 5 s per series per fit, ~60% of the daily backtest time).
  - Threads: the MLForecast models use all cores by default (`num_threads: -1`, regressor `n_jobs: -1`;
    random forest 64 s -> 26 s, LightGBM 5.5 s -> 0.8 s on daily). StatsForecast stays at `n_jobs: 1`:
    on Windows every parallel fit starts new processes (~13 s each), which made monthly fits 7-25x slower,
    gave nothing on daily, and made the test suite 17 min instead of 3.5. Revisit with many real series.
    Ridge, Lasso and the baselines have no thread option; torch (Chronos) uses all physical cores itself.

### Fair accuracy check (2026-10-11, `scripts/holdout_check.py`)

The models are chosen per series on the earlier backtest folds only (same rules: mean MAE, top 3,
equal weights) and scored on the last fold, which the choice never saw. WAPE on that fold:

| Frequency | Series | Pipeline | Best single model | Naive | Seasonal naive | Beats naive (series) |
|---|---|---|---|---|---|---|
| yearly (h=3, before Chronos worked) | 40 | 17.5% | 19.3% | 21.9% | n/a | 60% |
| monthly (h=18) | 15 | 12.1% | 12.2% | 17.5% | 16.5% | 80% |
| weekly (h=12) | 40 | 24.8% | 25.6% | 31.2% | 38.4% | 65% |
| daily (h=14, with ARIMA) | 45 | 22.8% | 23.2% | 29.1% | 29.5% | 89% |

The pipeline beats naive at every frequency, and the top-3 ensemble beats the single best model.
On the high-predictability synthetic series the error is 2-10%, on the low ones 25-60% (mostly noise
by design). Only 15 monthly series could be checked: with h=18 and 3 folds most monthly series
have fewer than two folds.

**Daily AutoARIMA left out (decided by the user 2026-10-11):** new models.yaml key
`skip_frequencies` (any model; `catalog.skip_for_frequency`, applied in `build_models_from_config`;
naive and the short-series models cannot be skipped). `auto_arima` has `skip_frequencies: [daily]`;
its `max_history_years: {daily: 3}` was removed with it (the comment in models.yaml says how to
bring it back). After skipping or removing a model, run a new backtest for that frequency: `forecast`
and `train` refuse a selection.json that names a model which is no longer a candidate. The backtest
itself accepts the old file (`registry.read_existing_selection`, changed 2026-10-11 after the first
daily rerun failed on it): unlocked entries naming such a model are dropped with a warning (the new
backtest replaces them); a locked entry naming one is still an error.

### Simulated earlier runs for the Live view (2026-10-11, `scripts/simulate_past_runs.py`)

`python -m poetry run python scripts/simulate_past_runs.py --frequency monthly --horizon 18 --back 12 9 6 3`
cuts the input 12/9/6/3 periods before its last date and runs `run-pipeline --steps forecast` on each
cut (cut inputs in `runs/_simulated_inputs/`, outputs to the scratch `runs/_simulated_outputs/`, run
records in `runs/<frequency>/`, which the Live view reads; outputs/ and registry/ are untouched).
Every simulated run uses today's selection.json, so it looks better than a real past run would. Then
`report --frequency monthly`. Done for monthly: Live view shows 4 runs (data to Sep 2024, Dec 2024,
Mar 2025, Jun 2025), WAPE of the total 2.3-3.2% per run vs a backtest level of 3.5% (the Jun 2025
run has only 3 months of actuals). Delete those run folders before going live (`runs/monthly/
20261011_1701*`, `..._1702*`, `..._1703*`, `..._1704*`; their manifests name an input under
`runs/_simulated_inputs/`); `runs/monthly/latest.txt` was set back to the real run 20261011_151141.

**Too-short series get the last value (decided by the user 2026-10-11).** Before, a series with fewer
than `min_series_length` rows (profile: 3) stopped the whole run at the input contract, so one
brand-new product would have blocked every forecast (found while simulating: 5 synthetic series had 1
row at the 6-months-back cut). Now `prepare_input_data` runs the structural checks on every series
(types, duplicates, alignment still stop the run), then `cleaning/missing.split_short_series` moves
the too-short series to `PreparedInput.fallback` with the last observed target as the forecast for the
whole horizon (always last value, whatever `fallback` says) and a warning per series ("only N row(s),
fewer than min_series_length=3, so no model is applied ..."). They never reach a model or the
backtest, and appear in the output like the other fallback series. If every series is too short the
run still stops. `check_min_length` itself is unchanged (still used with 1). Tests:
`tests/contracts/test_data_preparation.py`. Checked end to end on the monthly cut at Mar 2025: 40
series in the output, the 5 new ones flat at their last value.

## Forecast sanity checks (built 2026-10-11, requested by the user)

`forecast_checks.py`, run by the forecast pipeline right after `combine_forecasts` and before any file
is written. Per series (fallback series skipped), the final forecast against the series' own history:
- `outlier`: a forecast step more than `forecast_check_threshold` (null = `outlier_threshold`, 5) robust
  standard deviations from the median of its neighbouring forecast steps (up to 2 each side, at least 2).
  The yardstick is the spread of the history's neighbour residuals, as in the training-outlier method
  (`cleaning/outliers.extension_scores`), floored at 1% of the series' median level. Differences from the
  training method, both found on the real outputs: no seasonal adjustment (estimated across history and
  forecast it made a flat forecast of a seasonal daily series look spiky), and history points are not
  neighbours (a forecast at another level is a level question, flagged by the next checks, not a spike).
- `total_change`: forecast total vs the actual total of the same number of periods before the cutoff,
  more than `forecast_check_max_ratio` (3) times up or down; only when both totals are positive.
- `start_jump`: mean of the first 3 forecast steps more than 3x off BOTH the last 3 actuals and the same
  periods one season earlier (only the last actuals without a season). Requiring both stopped two false
  alarms on noisy series whose values a season earlier were near zero or negative.
- `zero`: the whole forecast is 0 while the last max(season, 3) actuals average above 0.
`forecast_check_action`: `warn` (default; log one line per flagged series, at most 5, plus a summary, and
write `outputs/<frequency>/forecast_checks.csv`: uuid, check, ds, value, reference, score, threshold,
message, static columns; written even when empty, copied into the run record), `raise`
(`ForecastCheckError`, a DataContractError: nothing is written) or `off`. On the current synthetic
outputs (185 series, 4 frequencies) one series is flagged: weekly `c8fe3ccd` (high-predictability series
declining towards 0; total_change 0.10x), a real case to look at. Tests: `tests/unit/test_forecast_checks.py`
(injected spike, wrong level, all-zero, noisy reference, flat seasonal forecast, settings) and
`tests/unit/test_forecasting.py` (file written; `raise` writes nothing).

**Flags in the HTML report (built 2026-10-11, requested by the user).** `pipelines/report.py` reads
`outputs/<frequency>/forecast_checks.csv` when it exists (`CHECKS_FILE`, defined in forecast_checks.py);
`report/data.py` adds `flags: [{check, message, ds}]` to every series and `meta.checks_ran` /
`meta.n_flagged`. The page (template.html; tokens `--warn` / `--warn-bg` in all three theme blocks):
a "Checks" filter next to market/product/company ("All series" / "Flagged only (N)", shared by both tabs,
disabled when nothing is flagged; only shown when the checks ran); "· N flagged by the checks" in the
count; in a total, a note "k series of the n in this selection were flagged" with a "Show flagged only"
button; when filtered, the box names the flagged series (up to 30, `FLAG_LIST_MAX`) with their checks, each name a link to that series in the single-series view, plus "Show all series" (asked by the user after the first version); for a single series a "Flagged" badge next to its name, a
box listing each check and message, "— flagged" in the series dropdown, an amber ring on dated flagged
forecast points (outlier) with a legend entry, and the flag in the hover tooltip. The fallback text now
says "too many missing periods or too little history". Checked in the browser (desktop, 375 px dark,
no console errors, no sideways scroll) on the weekly report, whose series c8fe3ccd is flagged; the ring
was checked with a flag injected in the page (no real forecast has an outlier flag). Tests in
`tests/unit/test_report.py`. Not regenerated: `docs/examples/forecast_report_DEMO.html`.

## Tree-model settings (built 2026-10-11, requested by the user)

The user asked for 8-10 LightGBM and XGBoost settings and 4-5 random forest settings, with settings that
suit each frequency, a couple of Tweedie variants, a check that no setting makes fit time explode, and a
check of n_jobs. All are entries in `config/models.yaml` (comments there say what each is for);
`skip_frequencies` keeps a setting to the frequencies it suits. Rows per backtest fold: yearly ~1.4k,
monthly ~1.9k, weekly ~8k, daily ~60k (all series together: the models are global).

| Setting | Idea | Frequencies |
|---|---|---|
| lightgbm / xgboost | baseline (200 trees, lr 0.05, 31 leaves / depth 6) | all |
| lightgbm_static / xgboost_static | baseline + market/product/company | all |
| lgbm_shallow / xgb_shallow | small trees (7 leaves, depth 3), slow learning, bagging | all |
| lgbm_regularized / xgb_regularized | medium trees, L1/L2 penalties, bagging, column sampling | all |
| lgbm_deep / xgb_deep | 500 trees, 63 leaves / depth 8 | weekly, daily |
| lgbm_l1 / xgb_l1 | absolute-error loss (median; robust, matches the MAE metric) | all |
| lgbm_tweedie / xgb_tweedie | Tweedie, variance power 1.2, `local_minmax` target scaling | all |
| lgbm_tweedie_static / xgb_tweedie_static | Tweedie 1.5 + static features | all |
| lgbm_daily_lags / xgb_daily_lags | lags 1-7, 14, 21, 28; rolling windows 7 and 28 | daily |
| random_forest | baseline (200 trees, leaf >= 3, half the features) | not daily |
| rf_fast | 100 trees, depth 10, half the rows per tree | all |
| rf_shallow | 300 trees, depth 6, leaf >= 10 | not daily |
| rf_static | static features, 30% of the rows per tree, leaf >= 5 | all |
| rf_large | 300 trees, leaf >= 2, 70% of the features | not daily |

Tweedie (LightGBM and XGBoost) refuses negative targets, and the default per-series scaling
(`local_standard`) produces negative values (some synthetic series are negative too), so the Tweedie
settings use `local_minmax` (each series scaled to 0..1). Per frequency the catalog now has 19 (yearly,
monthly), 21 (weekly) and 20 (daily) tree models, plus ridge and lasso.

**Backtest of all settings (partial runs, `registry/<f>_partial`; real selections untouched).** Compared
on the points every setting forecast. WAPE of the best / worst tree setting vs naive: yearly 15.8-18.9%
(naive 21.0%), monthly 14.3-17.9% (19.5%), weekly 28.6-30.9% (34.9%), daily 24.2-26.9% (32.1%): every
setting beats naive everywhere. Findings: the `_static` settings are near the top at every frequency
(rf_static best on yearly and monthly); on daily the daily-lag settings are best (xgb 24.2%, lgbm 24.5%);
Tweedie without static features is at the bottom on yearly/monthly and mid-field on weekly/daily, Tweedie
with static features is among the best on weekly/daily; the deep settings are weak on weekly and mid-field
on daily (kept: real data has more rows). These are synthetic data: re-judge on real data (Accuracy tab,
`scripts/holdout_check.py`).

**Fit time.** Total backtest time of all tree settings (3 folds): yearly ~46 s, monthly ~74 s, weekly
~113 s, daily 259 s (415 s before the changes below). Random forests are the risk on daily data:
random_forest and the first rf_static took ~37 s per fit on all cores (~114-120 s per daily backtest for
55 series; ~45 min per forest per backtest at 1000 daily series). Decided: random_forest and rf_shallow
skip daily (rf_fast is better there at 7 s per fit; rf_shallow was last), rf_large skips daily, and
rf_static uses max_samples 0.3 / min_samples_leaf 5 (was 0.7 / 3): same accuracy (daily 25.1% vs 25.0%,
monthly 14.4% vs 14.3%, yearly 15.8% vs 16.0%), daily backtest 120 s -> 54 s. Boosting settings take 1-8 s
per daily fit.

**n_jobs.** Each setting fitted on the full daily data with 1 thread and with all cores (4 logical / 2
physical): LightGBM 1.3-1.7x faster with all cores (5x for the first fit), XGBoost 1.2-1.4x, random
forests 2.4-2.5x. These libraries use threads, not processes, so there is no start-up cost as with
StatsForecast on Windows: all cores stays the default for the tree models (`n_jobs: -1`, `num_threads: -1`).

Also changed: `models` / `exclude_models` (e.g. `backtest --models ...`) ignore names that exist in
models.yaml but are skipped for the run's frequency, so one list serves every frequency (it was an
"unknown model" error); naming only skipped models is an error. Pipeline tests now build the catalog without
the tuning variants (`tests/helpers.use_core_catalog`, names starting `lgbm_`, `xgb_`, `rf_`), otherwise the
suite took 9.5 min instead of 3.5.

**Full runs with the new settings (2026-10-11).** `run-pipeline` for all four frequencies (monthly with
--horizon 18): yearly 3.9 min, monthly 3.6, weekly 5.5, daily 11.7 (was 9.1). New settings in the top 3 /
ranked first: yearly 13 / 7 of 40 series, monthly 12 / 5 of 30, weekly 13 / 10 of 45, daily 10 / 5 of 50.
Most chosen: lgbm_tweedie_static and xgb_tweedie_static (all frequencies), xgb_l1, the daily-lag settings
(daily). In the backtest, the best new setting beat the best previous model by more than 5% in 6 yearly, 1
monthly, 0 weekly and 0 daily series. **Out of sample it does not help:** on the same backtest, choosing per
series on the earlier folds and scoring on the last fold (scratchpad script, same rules as
`scripts/holdout_check.py`), with only the previous models vs with the new settings too: yearly WAPE 15.9% ->
16.9%, monthly 12.1% -> 12.3%, weekly 24.8% -> 24.9%, daily 22.8% -> 22.8%; series more than 5% better /
worse: yearly 3 / 7, monthly 0 / 2, weekly 2 / 8, daily 3 / 3 (most series keep the same choice). Reading:
with 2-3 folds per series, many similar candidates mean some setting wins the backtest by luck (selection
overfitting); the synthetic series (trend + sine seasonality + noise) also leave little for tuning to find.
Real data (promotions, intermittent volumes, more history) may differ, Tweedie especially. **Decided by
the user (2026-10-11): keep all settings competing for now; decide later (on real data) whether to prune.**
Options noted for then: prune to the consistently strong ones (lgbm/xgb_tweedie_static, rf_static, the
daily-lag settings), or reduce selection noise (more folds, or a new setting must beat the previous best by
a margin).

## Lag and rolling-window features (2026-10-11, requested by the user)

Until now every MLForecast model used lags 1, 2, 3, season and rolling windows 3, season (mean, std) without
any check. The user asked to test alternative sets; agreed approach (the per-series selection already
overfits with many candidates): choose the feature set once per frequency and model family, on the backtest
of the family baselines (lightgbm, xgboost, rf_fast, ridge, lasso; same regressor settings), not as extra
per-series candidates. StatsForecast models and Chronos do not use lag features. The user chose
"Alternative A: denser short-term" (a long-memory set B was proposed too; long lags exclude most synthetic
series: a model needs max lag + 3 observations):

| Frequency | Alternative A lags | Rolling windows |
|---|---|---|
| yearly | 1-5 | 3, 5 |
| monthly | 1, 2, 3, 6, 12, 24 | 3, 6, 12 |
| weekly | 1, 2, 3, 4, 8, 13, 52 | 4, 13, 52 |
| daily | 1-7, 14, 21, 28 | 7, 28 |

Results (partial backtests, compared on the points both versions forecast; change in WAPE, test folds won by A):
- daily: every family better: trees -6.0% to -6.9%, ridge/lasso -16.2%, 3 of 3 folds, 82-90% of series.
  Coverage 40 of 45 backtested series (needs 32 days) instead of 45.
- yearly: trees better (lightgbm -4.5%, xgboost -8.0%, rf_fast -6.3%; 2-3 of 3 folds); ridge/lasso worse
  (+4.3%, 0 of 3 folds).
- monthly: small gains (-1.1% to -5.2%) but measured on only 15 of 30 series, because lag 24 needs 27 months:
  coverage halves. Not a clear win: kept the default.
- weekly: within +-1.4%: kept the default.

**Adopted (2026-10-11, the plan agreed with the user: switch only where A clearly wins):** daily, all
MLForecast models; yearly, tree models only. Mechanism: optional `ml_features` per frequency profile in
`config/frequency_profiles.yaml` (`trees` / `linear` groups, each `lags` and/or `rolling_windows`;
validated in `config._ml_features`); `build_models(..., ml_features=)` fills them in as the default for
every MLForecast model of that group (`feature_group` class attribute: "trees" for LightGBM/XGBoost/random
forest, "linear" for ridge/lasso) that does not set its own lags in models.yaml. Per-frequency `lags` /
`rolling_windows` mappings are also accepted in models.yaml now (`{monthly: [...], daily: [...]}`). The
experiment entries (`*_lags_a`) were removed. Side effect: on daily, lgbm_daily_lags / xgb_daily_lags now have
the same features as the baselines and differ only in their regressor settings. Tests: test_config.py,
test_mlforecast_adapter.py.

Full runs afterwards (yearly 2.8 min, daily 12.8 min; fair hold-out check, `scripts/holdout_check.py`, vs the
run just before): yearly pipeline WAPE 16.9% -> 16.7%; daily 22.8% -> 23.2%. So at pipeline level the
change is neutral, although each ML model on its own is 6-16% better on daily: on daily only 14 of 55 series
have an ML model in their selection (Chronos and the statistical models win most series), 5 short daily
series lose their ML candidates (32 days needed), and a few tenths of a point are within the noise of the
per-series choice. Yearly: ML models in 23 of 40 selections. Kept (better models, same pipeline result);
re-judge on real data.

## Slow models: Prophet and Orbit (tried 2026-10-11, asked by the user)

- **Prophet** installs with `poetry install --extras "stats ml foundation prophet"` (1.5.0, with
  cmdstanpy 1.3.0; the Windows wheel ships CmdStan) and works. A first test took 36 s per fit: with fewer
  than 100 points Prophet uses Newton's method, and on a noise-free test series the fit never converged
  (10,000 iterations). On realistic data a fit takes 0.2-0.8 s.
- **Orbit cannot be installed as things stand:** orbit-ml 1.1.5.1 (needed for Python 3.12+) is only on PyPI
  as source, and the older wheels do not install on this Python / Windows. Building it needs a C++ toolchain
  (RTools: the installer `RTools40.exe` is in `~/.cmdstan` but was never run; ~1 GB) and a CmdStan build.
  Not done: left for the user to decide.
- **Backtest with `--no-fast-only`** (Prophet in, Orbit skipped with a warning) on all four frequencies;
  one backtest writes both `selection_fast.json` (used by default) and `selection.json` (with Prophet,
  `includes_slow_models: true`). Prophet's backtest time / total: yearly 36 / 199 s, monthly 25 / 169 s,
  weekly 34 / 285 s, daily 70 / 623 s. Prophet in the top 3 / ranked first: yearly 2 / 1 of 40 series,
  monthly 1 / 0 of 30 (only 9 series have the two years Prophet needs in every fold), weekly 7 / 4 of 45,
  daily 12 / 7 of 50. Beats the best fast model by more than 5%: 0, 0, 2 and 1 series. **Fair hold-out**
  (same backtest, choice with vs without the slow models): yearly 16.7% -> 16.7%, monthly 12.3% -> 12.3%,
  weekly 24.9% -> 24.9%, daily 23.2% -> 23.3%. Conclusion on the synthetic data: not worth the extra time;
  `fast_only: true` stays the default. Real data with holidays, promotions or trend changes (Prophet's
  strengths) may differ: worth one `--no-fast-only` backtest there.

## AutoETS and AutoTheta versions (2026-10-11/12, requested by the user)

Eight versions added to models.yaml (comments there), 5 per model with the defaults: `ets_damped` (ZZZ,
damped trend), `ets_no_trend` (ZNZ), `ets_additive_season` (ZZA), `ets_multiplicative_season` (ZZM);
`theta_additive` (additive decomposition), `theta_standard` (STM), `theta_optimized` (OTM), `theta_dynamic`
(DOTM); the Theta versions keep `max_history_years: {daily: 5}`. Checked in StatsForecast 2.1 first: forcing
a seasonal ETS form with season length 1 is an error ("Nonseasonal data"), with season 52 it silently fits a
non-seasonal model, so the two seasonal ETS versions skip yearly and weekly; `theta_additive` skips yearly
(no decomposition there); a multiplicative ETS season fails on series with values <= 0 ("no model able to
be fitted"), which then get the adapter's naive fallback (17 of ~40 synthetic daily series, 1 monthly).
Pipeline tests leave the versions out (`ets_`, `theta_` added to `tests/helpers.TUNING_VARIANT_PREFIXES`).

Partial backtests (compared on common points): the versions are within ~1 point of each other at every
frequency (yearly 19.5-20.8%, monthly 12.7-13.9%, weekly 28.4-28.9%, daily ETS 23.7% for all four non-
multiplicative versions, Theta 24.4-24.5%; ets_multiplicative_season 30.0% on daily because of the
fallbacks). The fixed versions cost far less than the automatic search for the same accuracy: daily
ets_no_trend 21 s vs auto_ets 115 s, theta_standard 13 s vs auto_theta 121 s (3 folds, 45 series); weekly
ets_no_trend 8.5 s vs auto_ets 35 s.

Full runs (yearly 3.2 min, monthly 3.6, weekly 6.0, daily 19.8 instead of 12.8): a version is in the top 3 /
ranked first in yearly 2 / 1 of 40 series, monthly 6 / 1 of 30, weekly 2 / 1 of 45, daily 25 / 10 of 50.
Fair hold-out with vs without the versions (same backtest): yearly 16.7% -> 16.9%, monthly 12.3% -> 12.5%,
weekly 24.9% -> 24.9%, daily 23.2% -> 23.2% (series >5% better / worse: 0/1, 0/1, 0/1, 1/5). Same lesson as
the tree settings: more similar candidates do not help out of sample with 2-3 folds per series.
**Decided by the user (2026-10-12): keep all versions competing for now, as with the tree settings; decide
on real data.** Options noted for then: remove them, or replace auto_ets / auto_theta by a cheaper fixed
version (e.g. ets_damped, theta_standard) where the accuracy is the same (check with the fair hold-out).

## Next steps (updated 2026-10-11)

1. **Timing at the real scale:** `poetry run benchmark --frequency <f> --series <real number>`. Daily
   AutoARIMA is the model to watch (about 5 s per series per fit with 3 years); options are less history,
   leaving it out for daily (needs a per-frequency model list), or parallel StatsForecast with many series.
2. **Real data input:** waiting for the user to describe the real file (column names, number of series,
   frequencies, how the data arrives); then add a column mapping to the config if the names differ.
3. **Decisions for the user:** how often to backtest / train / forecast (then scheduling and deleting old
   `runs/` folders); confirm the platform's column names (`output_rename`); check the outlier threshold,
   minimum history, ensemble size and number of folds on real data with the Accuracy tab.
4. Optional: Prophet (`--extras prophet`, then `--no-fast-only`); Orbit needs a C++ toolchain on Windows.
4b. On real data: decide whether to prune the tree-model settings and the ETS/Theta versions (see "Tree-model
    settings" and "AutoETS and AutoTheta versions"); both add run time without a gain on the synthetic data.
5. **Independent code review** (the user wants it last). Known: `ruff check` reports older line-length and
   import-order issues across src and tests; the number of pytest warnings varies between identical full runs
   (124 to 2882, almost all attributed to tests/unit/test_outliers.py, none when that file runs alone).

## Note for other LLMs / collaborators

This file is the source of truth for the project's goals and design. Read it before working on the project, and update it when decisions are made.
