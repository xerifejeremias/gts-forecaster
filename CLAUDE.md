# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `OVERVIEW.md` first: it is the source of truth for the design, every decision the user made
(with dates), the build status and what is still open. `README.md` has the user-facing commands.

## Where things stand (2026-10-12)
- First real runs on the laptop are done: install, tests and `run-pipeline` work for all four
  frequencies (details and the decisions made in OVERVIEW "First real run on the laptop").
- Code review done 2026-10-12 (OVERVIEW "Code review"). Lint clean-up is still open: `ruff check`
  reports older line-length and import-order issues, so a non-clean ruff run is not your regression.
- Real-data input (CSV, Parquet, SQL, column mapping) is built; still waiting on the user's real file.
- Scheduled tasks are built but **not installed** (user decision 2026-10-12): jobs run only on request.
- Poetry is not on PATH here: use `python -m poetry ...` (e.g. `python -m poetry run pytest -q`).
  The venv is in the project (`.venv`). Set `PYTHONUTF8=1` if output has encoding errors.
- Python 3.12 or 3.13 is required (pyproject). pandas must stay < 3.

## Commands

```
poetry install --extras "stats ml foundation"   # core + StatsForecast + MLForecast + Chronos
poetry run pytest -q                            # whole suite
poetry run pytest tests/unit/test_features.py -q             # one file
poetry run pytest tests/unit/test_features.py::test_name -q  # one test
poetry run pytest -k "outlier or floor" -q                   # by name
poetry run ruff check src tests                 # lint (dev dep, line length 100, E/F/I/B/UP)
poetry run ruff format src tests                # format
```

Pipeline entry points (all take `--frequency yearly|monthly|weekly|daily`; README has the full flags):
`validate-input`, `backtest`, `train`, `forecast`, `report`, `run-pipeline`, `benchmark`.
Automatic runs: `schedule show|install|remove` and `scheduled-run --frequency f --job forecast|backtest`
(`config/schedule.yaml`, package `ops/`; README "Automatic runs").

Fast feedback while developing a model or a feature:
`poetry run backtest --frequency monthly --max-series 8 --models lightgbm ridge` — a partial run,
which writes to `registry/monthly_partial/` and never touches the real `selection.json`.

No `conftest.py`: `pyproject.toml` puts `src` and `tests` on `pythonpath`, so tests do
`from helpers import make_panel` (`tests/helpers.py` builds valid panels in input or model format).

## Architecture

Config-driven, cutoff-driven, one model interface. Switching frequency, horizon, metric or model set
is a config change, never a code change.

**Two names for the same data.** Input files use `uuid, date, target`. The moment the input contract
has passed, `columns.py` renames to the Nixtla names `unique_id, ds, y`, and everything downstream
uses those. Explanatory and static columns keep their names. Do not reintroduce the raw names past
that boundary.

**One path into the data.** `data_preparation.prepare_input_data(cfg)` returns
`PreparedInput(data, fallback, report, cutoff, config, static)`. It loads (via
`io/loaders.load_input`: `data_source: auto|csv|parquet|sql`; `input_columns` maps the user's column
names to `uuid, date, target` on reading, so the contract never changes), enforces the strict input
contract (`contracts/`), repairs missing periods (`cleaning/missing.py`), then checks the contract
again. Series whose gaps are too long never reach a model: they land in `fallback` and get a
last-value or zero forecast. `static` is one row per series (market, product, company) and is kept
out of the model frame on purpose, so the model-frame contract stays fixed.

**Everything downstream takes a cutoff.** `splitting.py` is the only place that decides what is
train and what is test: `backtest_splits` for rolling origins, `forecast_split` for the single cutoff
at the last date. Outlier cleaning and feature building are both functions of the cutoff and run
inside each fold, so nothing after the cutoff can leak. `is_filled` rows are never scored.

**Models share one interface, `models/base.py:BaseForecaster`.** `fit(train)` / `predict(future)` on
a panel; `future` carries `horizon_step`, so no model ever knows the horizon or the frequency. Class
attributes drive the pipelines: `scope` (local/global), `cost` ("slow" is excluded while `fast_only`
is on), `requires_packages` + `install_extra` (a missing library means the model is skipped with a
warning that names the extra), `supports_update` (frozen-models mode can take newer history without
refitting), `min_train_length`. Every third-party library sits behind an adapter
(`statsforecast_adapter`, `mlforecast_adapter`, `chronos_adapter`, `bayesian_adapters` for Prophet
and Orbit) and is imported only at fit time. `config/models.yaml` maps a name to a type plus params;
`models/catalog.py` builds them. A new model is an adapter plus a catalog entry: backtest, selection,
report and timings pick it up automatically.

**Features are opt-in per model.** A model asks for a named feature set from `config/features.yaml`
and receives a `FeatureBuilder`; models that generate their own features (Nixtla, Prophet, Orbit) ask
for none. `usable_at_step(h)` is what keeps multi-step forecasts leakage-free.

**Backtest, then selection, then forecast.** `evaluation/backtest.py` scores every model per series
and fold; `selection.py` ranks per series against the `naive` reference and builds an ensemble (default
`selection_rule: blend`: half the series' own top 3, half the common top 3 over all series;
`per_series` is the older rule); `registry.py` writes it. One backtest writes **two** selections,
`selection_fast.json` and `selection.json`, and a fast-only backtest never overwrites a
`selection.json` in which the slow models actually competed. `forecasting.py` then refits
(`frozen_selection`) or loads saved models (`frozen_models`, via `models/store.py`), `output.py`
writes the platform CSV, and `report/` builds the static HTML. The report only reads saved files; it
never runs a model.

**Contracts stop the run.** Input, model frame, output and selection each have a contract; a breach
raises `DataContractError` and nothing is written. If you change a frame's shape, change its contract
and its test in `tests/contracts/` with it.

**`run-pipeline` is the production entry point** (`pipelines/run.py`): it prepares the input once,
runs backtest, train, forecast and report in order, and records each run under
`runs/<frequency>/<run id>/` (`run.log`, `manifest.json` with settings, library versions and an input
fingerprint, a config copy and copies of the outputs). It never raises for a failing step; it records
it, marks the rest skipped, and exits 1.

## Conventions
- Windows laptop; the user works partly from a phone. Keep answers short and plain.
- Tests in `tests/` (pytest). Run the full suite after every change.
- After a change, update `OVERVIEW.md` (and `README.md` for commands) so another LLM can pick up
  the project without the chat history. Record user decisions with their date.
- The HTML report (`src/gts_forecaster/report/template.html`) has a look the user likes; keep it.
  Reference: `docs/examples/forecast_report_DEMO.html`. After changing the template, run
  `scripts/check_report_in_browser.py` (needs playwright, not a project dependency).
- Defaults decided by the user: fast_only on; two selections from one backtest
  (selection_fast.json / selection.json); negative forecasts set to 0; no prediction intervals;
  platform output in long format (`forecast_output.csv`).

## Two traps
- **Git: this folder has its own repo** (set up 2026-10-11, branch `main`, remote `origin` =
  GitHub `xerifejeremias/gts-forecaster`; push only when the user asks). There is
  also an unrelated repo rooted at `C:\Users\Admin`; it does not track project files. Commit only
  when the user asks. Never commit real (production) data.
- **The "Folder layout" sketch in OVERVIEW.md is partly aspirational.** `io/writers.py`,
  `prophet_adapter.py`, `orbit_adapter.py`, `tests/conftest.py`, `tests/fixtures/` and
  `tests/integration/` do not exist (Prophet and Orbit live in `models/bayesian_adapters.py`).
  Trust the files, not the sketch.
