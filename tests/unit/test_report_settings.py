"""FAQ tab data: the run settings quoted in the answers."""

import json
from dataclasses import replace

from gts_forecaster.config import load_config
from gts_forecaster.report.settings import run_settings


def test_settings_hold_the_modelling_values_and_the_profile():
    cfg = load_config({"frequency": "monthly", "horizon": 18})
    out = run_settings(cfg, {"meta": {"metric": "mae", "top_n": 3, "models": ["naive"]}})
    assert out["horizon"] == 18 and out["backtest_step"] == 18  # null step = the horizon
    assert out["season_length"] == 12 and out["pandas_freq"] == "MS"
    assert out["outlier_threshold"] == cfg.outlier_threshold
    assert out["selection"] == {"metric": "mae", "top_n": 3}
    json.dumps(out)  # goes into the report as JSON


def test_the_forecast_shown_and_the_backtest_decide_horizon_and_step():
    cfg = load_config({"frequency": "monthly"})  # default horizon 12
    out = run_settings(cfg, {"meta": {"horizon": 18}}, horizon=18)
    assert out["horizon"] == 18 and out["backtest_step"] == 18


def test_settings_never_carry_connection_details_or_paths():
    cfg = replace(load_config({"frequency": "monthly"}),
                  sql_connection="postgresql://user:secret@host/db", sql_query="select 1")
    text = json.dumps(run_settings(cfg, None))
    assert "secret" not in text and "select 1" not in text and "data_path" not in text
