import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.selection_schema import validate_selection
from gts_forecaster.evaluation.backtest import run_backtest
from gts_forecaster.models.catalog import build_models, load_model_specs
from gts_forecaster.registry import (
    build_selection_document,
    merge_locked,
    read_selection,
    write_json_atomic,
)
from gts_forecaster.selection import SelectionError, select_models
from helpers import BASELINES, make_uuid

N = 24


def _panel():
    """linear (drift is perfect), constant (everything is perfect), short (3 observations)."""
    dates = pd.date_range("2022-01-01", periods=N, freq="MS")
    frames = [
        pd.DataFrame({"unique_id": "linear", "ds": dates, "y": np.arange(1.0, N + 1)}),
        pd.DataFrame({"unique_id": "const", "ds": dates, "y": 5.0}),
        pd.DataFrame({"unique_id": "short", "ds": dates[-3:], "y": [1.0, 2.0, 3.0]}),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["is_filled"] = False
    return df


def _models(season=12):
    return build_models(load_model_specs(types=BASELINES), season)


def _backtest(metric="mae", n_folds=2, horizon=3):
    return run_backtest(
        _panel(), _models(), horizon=horizon, pandas_freq="MS", n_folds=n_folds,
        metric=metric, season_length=12,
    )


def test_backtest_scores_are_correct():
    r = _backtest()
    s = r.scores.set_index(["unique_id", "model", "fold"])["score"]
    for fold in r.cutoffs:
        assert s[("linear", "drift", fold)] == pytest.approx(0.0)
        assert s[("linear", "naive", fold)] == pytest.approx(2.0)  # last value lags by 1, 2, 3
        assert s[("const", "naive", fold)] == pytest.approx(0.0)


def test_backtest_skips_series_without_enough_history():
    r = _backtest()
    assert "short" not in set(r.scores["unique_id"])
    assert r.series_lengths["short"] == 3


def test_backtest_metric_is_switchable():
    mae = _backtest("mae").scores
    rmse = _backtest("rmse").scores
    a = mae[(mae.unique_id == "linear") & (mae.model == "naive")]["score"].iloc[0]
    b = rmse[(rmse.unique_id == "linear") & (rmse.model == "naive")]["score"].iloc[0]
    assert a != b


def test_filled_periods_are_not_scored():
    df = _panel()
    df.loc[(df.unique_id == "linear") & (df.ds == df.ds.max()), "is_filled"] = True
    r = run_backtest(df, _models(), horizon=3, pandas_freq="MS", n_folds=1, season_length=12)
    n = r.scores.query("unique_id == 'linear' and model == 'naive'")["n_points"].iloc[0]
    assert n == 2


def test_selection_picks_best_model_per_series():
    r = _backtest()
    sel = select_models(r, _models(), top_n=1)
    assert sel.series["linear"].models == (("drift", 1.0),)
    assert sel.series["const"].models[0][1] == 1.0  # ties go to the first model in the catalog
    assert sel.series["const"].models[0][0] == "naive"


def test_top_n_builds_weighted_ensemble():
    r = _backtest()
    equal = select_models(r, _models(), top_n=2, ensemble_weights="equal").series["linear"]
    assert [w for _, w in equal.models] == [0.5, 0.5]
    inv = select_models(r, _models(), top_n=2, ensemble_weights="inverse_error").series["linear"]
    weights = dict(inv.models)
    assert weights["drift"] > weights["naive"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_ensemble_row_in_score_table():
    sel = select_models(_backtest(), _models(), top_n=2)
    assert "ensemble" in set(sel.table["model"])


def test_short_series_gets_fixed_ensemble():
    sel = select_models(_backtest(), _models(), top_n=3, short_series_n=2).series["short"]
    assert [n for n, _ in sel.models] == ["naive", "seasonal_naive"] or [
        n for n, _ in sel.models
    ] == ["naive", "drift"]
    assert sum(w for _, w in sel.models) == pytest.approx(1.0)
    assert sel.note


def test_short_series_without_usable_model_fails():
    with pytest.raises(SelectionError):
        select_models(_backtest(), _models(), top_n=1, short_series_models=["seasonal_naive"])


def test_selection_requires_reference_model():
    models = _models()
    del models["naive"]
    with pytest.raises(SelectionError):
        select_models(_backtest(), models, top_n=1)


def test_every_series_is_selected():
    sel = select_models(_backtest(), _models(), top_n=3)
    assert set(sel.series) == {"linear", "const", "short"}


# ---- selection document, contract, registry -------------------------------------------------


def _doc():
    sel = select_models(_backtest(), _models(), top_n=2)
    return build_selection_document(sel, {"metric": "mae"})


def test_document_is_valid_and_weights_sum_to_one():
    doc = _doc()
    validate_selection(doc, _models(), ["linear", "const", "short"])
    for entry in doc["series"].values():
        assert sum(m["weight"] for m in entry["models"]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["series"]["linear"]["models"].__setitem__(0, {"name": "nope", "weight": 1.0}),
        lambda d: d["series"]["linear"].__setitem__("models", []),
        lambda d: d["series"]["linear"]["models"][0].__setitem__("weight", 0.3),
        lambda d: d["series"]["linear"]["models"][0].__setitem__("weight", -1),
        lambda d: d["series"]["linear"].__setitem__("locked", "yes"),
        lambda d: d["series"].pop("const"),
        lambda d: d.__setitem__("extra", 1),
        lambda d: d["series"]["linear"].__setitem__("surprise", 1),
    ],
)
def test_invalid_selection_is_rejected(mutate):
    doc = _doc()
    mutate(doc)
    with pytest.raises(DataContractError):
        validate_selection(doc, _models(), ["linear", "const", "short"])


def test_non_object_selection_is_rejected():
    with pytest.raises(DataContractError):
        validate_selection([], _models())


def test_locked_entries_survive_a_new_backtest():
    old, new = _doc(), _doc()
    old["series"]["linear"] = {"locked": True, "models": [{"name": "naive", "weight": 1.0}]}
    merged, kept = merge_locked(new, old)
    assert kept == ["linear"]
    assert merged["series"]["linear"]["models"] == [{"name": "naive", "weight": 1.0}]
    assert merged["series"]["const"] == new["series"]["const"]


def test_unlocked_entries_are_replaced_and_missing_file_is_fine():
    old, new = _doc(), _doc()
    old["series"]["linear"]["models"] = [{"name": "naive", "weight": 1.0}]
    assert merge_locked(new, old)[1] == []
    assert merge_locked(new, None)[1] == []


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "sub" / "selection.json"
    write_json_atomic(path, _doc())
    assert not list(path.parent.glob("*.tmp"))
    assert read_selection(path, _models())["series"].keys() == _doc()["series"].keys()


def test_read_rejects_missing_and_broken_files(tmp_path):
    with pytest.raises(DataContractError):
        read_selection(tmp_path / "missing.json", _models())
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(DataContractError):
        read_selection(bad, _models())


def test_hand_edit_swapping_a_model_name_is_valid(tmp_path):
    path = tmp_path / "selection.json"
    doc = _doc()
    doc["series"]["linear"]["models"] = [{"name": "seasonal_naive", "weight": 1.0}]
    path.write_text(json.dumps(doc))
    assert read_selection(path, _models())["series"]["linear"]["models"][0]["name"] == "seasonal_naive"


# ---- pipeline -------------------------------------------------------------------------------


def _prepared(tmp_path):
    cfg = load_config(
        {"frequency": "monthly", "horizon": 3, "n_folds": 2, "top_n": 2, "registry_dir": tmp_path, "backtesting_dir": tmp_path / "bt"}
    )
    df = _panel()
    return SimpleNamespace(data=df, config=cfg, cutoff=df["ds"].max())


def test_pipeline_writes_registry_files(tmp_path):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    out = run_backtest_pipeline(_prepared(tmp_path))
    # fast_only is on by default, so the main selection is the fast one; both are written
    assert out.selection_path == tmp_path / "monthly" / "selection_fast.json"
    assert (tmp_path / "monthly" / "selection.json").is_file()
    assert (tmp_path / "monthly" / "backtest_scores.csv").is_file()
    assert out.scores_path == tmp_path / "monthly" / "backtest_scores_fast.csv"
    assert out.scores_path.is_file()
    # the selection names the backtest file it came from (read by the report's accuracy tab)
    assert json.loads(out.selection_path.read_text())["meta"]["backtest_forecasts_file"] == out.forecasts_path.name
    doc = read_selection(out.selection_path, _models(), ["linear", "const", "short"])
    assert doc["meta"]["horizon"] == 3 and doc["meta"]["metric"] == "mae"
    assert "uuid" in pd.read_csv(out.scores_path).columns


def test_pipeline_keeps_locked_entry_on_rerun(tmp_path):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    first = run_backtest_pipeline(_prepared(tmp_path))
    doc = json.loads(first.selection_path.read_text())
    doc["series"]["linear"] = {"locked": True, "models": [{"name": "naive", "weight": 1.0}]}
    first.selection_path.write_text(json.dumps(doc))
    second = run_backtest_pipeline(_prepared(tmp_path))
    assert second.kept_locked == ["linear"]
    assert second.document["series"]["linear"]["models"][0]["name"] == "naive"


def test_pipeline_refuses_an_invalid_existing_selection(tmp_path):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    path = tmp_path / "monthly" / "selection_fast.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(DataContractError):
        run_backtest_pipeline(_prepared(tmp_path))


def test_backtest_forecast_file_has_forecast_vs_actual_for_every_model_and_window(tmp_path):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    from datetime import datetime

    out = run_backtest_pipeline(_prepared(tmp_path), run_time=datetime(2026, 10, 5, 12, 31))
    assert out.forecasts_path == tmp_path / "bt" / "monthly" / "backtest_forecasts_20261005_1231.csv"
    t = pd.read_csv(out.forecasts_path, parse_dates=["cutoff", "ds"])
    assert list(t.columns) == [
        "uuid", "model", "fold", "cutoff", "ds", "horizon_step", "actual", "forecast", "is_filled"
    ]
    # linear series, drift, newest fold: 3 dates, forecast continues the line, actuals are known
    rows = t[(t.uuid == "linear") & (t.model == "drift") & (t.fold == t.fold.max())]
    assert len(rows) == 3
    assert rows["actual"].tolist() == [22.0, 23.0, 24.0]
    assert rows["forecast"].tolist() == pytest.approx([22.0, 23.0, 24.0])
    assert (rows["ds"] > rows["cutoff"]).all()
    assert set(t.model) >= {"naive", "drift"}
    assert not t.duplicated(["uuid", "model", "fold", "ds"]).any()
    assert t[["actual", "forecast"]].notna().all().all()


def test_backtest_forecast_files_are_never_overwritten(tmp_path):
    from datetime import datetime

    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    same_minute = datetime(2026, 10, 5, 12, 31)
    a = run_backtest_pipeline(_prepared(tmp_path), run_time=same_minute)
    b = run_backtest_pipeline(_prepared(tmp_path), run_time=same_minute)
    c = run_backtest_pipeline(_prepared(tmp_path), run_time=datetime(2026, 10, 6, 9, 0))
    assert len({a.forecasts_path, b.forecasts_path, c.forecasts_path}) == 3
    assert b.forecasts_path.name == "backtest_forecasts_20261005_1231_2.csv"
    assert len(list(a.forecasts_path.parent.glob("backtest_forecasts_*.csv"))) == 3
    assert not any(":" in p.name for p in a.forecasts_path.parent.iterdir())
