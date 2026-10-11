import json
import re

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.pipelines.report import run_report_pipeline
from gts_forecaster.report.data import build_report_data
from gts_forecaster.report.render import render_html, write_report

DIMS = ("market", "company")


def _history():
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    frames = []
    for uid, market, company, start in (("a", "DE04", "A", 0), ("b", "PT04", "A", 2), ("c", "DE04", "B", 0)):
        d = dates[start:]
        frames.append(pd.DataFrame({
            "uuid": uid, "date": d, "target": np.arange(1.0, len(d) + 1), "market": market, "company": company,
        }))
    h = pd.concat(frames, ignore_index=True)
    return h[~((h.uuid == "c") & (h.date == dates[3]))]  # a gap in series c


def _forecasts(ids=("a", "b", "c"), start="2024-07-01", periods=3):
    ds = pd.date_range(start, periods=periods, freq="MS")
    return pd.concat([pd.DataFrame({"ds": ds, "uuid": u, "forecast": 100.0 + k}) for k, u in enumerate(ids)], ignore_index=True)


def _data(**kw):
    return build_report_data(
        _history(), _forecasts(), pandas_freq="MS", frequency="monthly", static_columns=DIMS, **kw
    )


def test_dates_cover_history_and_forecast():
    d = _data()
    assert d["dates"][:6] == [f"2024-{m:02d}-01" for m in range(1, 7)]
    assert d["dates"][6:] == ["2024-07-01", "2024-08-01", "2024-09-01"]
    assert d["n_history"] == 6 and d["meta"]["horizon"] == 3 and d["meta"]["cutoff"] == "2024-06-01"
    assert d["meta"]["dimensions"] == list(DIMS) and d["meta"]["n_series"] == 3


def test_series_values_are_aligned_to_the_date_grid():
    s = {x["id"]: x for x in _data()["series"]}
    assert s["a"]["hist"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert s["b"]["hist"] == [None, None, 1.0, 2.0, 3.0, 4.0]  # starts later
    assert s["c"]["hist"][3] is None  # the missing period stays empty
    assert s["a"]["fc"] == [100.0] * 3 and s["c"]["fc"] == [102.0] * 3
    assert s["b"]["dims"] == {"market": "PT04", "company": "A"}


def test_selection_models_and_fallback_flag():
    selection = {"series": {"a": {"models": [{"name": "naive", "weight": 0.5}, {"name": "drift", "weight": 0.5}]}}}
    s = {x["id"]: x for x in _data(selection=selection, fallback_ids=["c"])["series"]}
    assert s["a"]["models"][0] == {"name": "naive", "weight": 0.5}
    assert s["c"]["fallback"] and not s["a"]["fallback"]


def test_series_without_forecast_or_history_still_listed():
    d = build_report_data(
        _history(), _forecasts(ids=("a", "b")), pandas_freq="MS", frequency="monthly", static_columns=DIMS
    )
    c = next(x for x in d["series"] if x["id"] == "c")
    assert c["fc"] == [None] * 3


def test_nan_is_never_written_as_nan():
    h = _history()
    f = _forecasts()
    f.loc[0, "forecast"] = np.nan
    d = build_report_data(h, f, pandas_freq="MS", frequency="monthly", static_columns=DIMS)
    json.dumps(d, allow_nan=False)  # raises on NaN


def test_forecasts_must_follow_the_history():
    with pytest.raises(DataContractError, match="not after"):
        build_report_data(_history(), _forecasts(start="2024-05-01"), pandas_freq="MS",
                          frequency="monthly", static_columns=DIMS)


def test_missing_columns_are_reported():
    with pytest.raises(DataContractError, match="forecasts"):
        build_report_data(_history(), _forecasts().drop(columns="forecast"), pandas_freq="MS", frequency="monthly")
    with pytest.raises(DataContractError, match="no column"):
        build_report_data(_history().drop(columns="market"), _forecasts(), pandas_freq="MS",
                          frequency="monthly", static_columns=DIMS)


def test_without_static_columns_report_still_builds():
    d = build_report_data(_history(), _forecasts(), pandas_freq="MS", frequency="monthly")
    assert d["meta"]["dimensions"] == [] and d["series"][0]["dims"] == {}


# ---- rendering --------------------------------------------------------------------------------

def _embedded(html):
    m = re.search(r"const DATA = (.*?);\n\(function", html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_html_embeds_the_data_and_is_self_contained():
    d = _data()
    html = render_html(d)
    assert _embedded(html) == d
    assert "__GTS_REPORT_DATA__" not in html
    assert not re.search(r'(src|href)="https?://', html)  # works offline


def test_data_cannot_break_out_of_the_script_tag():
    d = _data()
    d["series"][0]["dims"]["market"] = "</script><script>alert(1)</script>"
    html = render_html(d)
    assert html.count("</script>") == 1
    assert _embedded(html)["series"][0]["dims"]["market"].startswith("</script>")


def test_write_report_writes_both_files(tmp_path):
    write_report(_data(), tmp_path / "x" / "r.html", tmp_path / "r.json")
    assert (tmp_path / "x" / "r.html").is_file()
    assert json.loads((tmp_path / "r.json").read_text())["n_history"] == 6


# ---- pipeline --------------------------------------------------------------------------------

def test_report_pipeline_reads_saved_files(tmp_path):
    input_csv = tmp_path / "in.csv"
    _history().to_csv(input_csv, index=False)
    out = tmp_path / "outputs" / "monthly"
    out.mkdir(parents=True)
    _forecasts().to_csv(out / "forecasts.csv", index=False)
    registry = tmp_path / "registry" / "monthly"
    registry.mkdir(parents=True)
    (registry / "selection_fast.json").write_text(json.dumps({
        "meta": {"metric": "mae"},
        "series": {u: {"locked": False, "models": [{"name": "naive", "weight": 1.0}]} for u in ("a", "b")},
    }))
    cfg = load_config({"data_path": str(input_csv), "static_columns": list(DIMS), "registry_dir": tmp_path / "registry"})
    result = run_report_pipeline(cfg, output_dir=tmp_path / "outputs")
    data = json.loads(result.json_path.read_text())
    assert result.html_path.is_file() and result.n_series == 3
    assert data["meta"]["metric"] == "mae"
    assert [s["id"] for s in data["series"] if s["fallback"]] == ["c"]  # no selection entry


def test_report_pipeline_without_forecasts_says_what_to_do(tmp_path):
    cfg = load_config({})
    with pytest.raises(DataContractError, match="forecast"):
        run_report_pipeline(cfg, output_dir=tmp_path)


# ---- per-model lines -------------------------------------------------------------------------

def _by_model():
    f = _forecasts()
    out = f[["uuid", "ds"]].copy()
    out["naive"] = 1.0
    out["drift"] = np.where(out.uuid == "c", np.nan, 2.0)  # drift has no forecast for series c
    out["never"] = np.nan  # a model that forecast nothing at all
    return out.assign(market=out.uuid.map({"a": "DE04", "b": "PT04", "c": "DE04"}))


def test_model_lines_per_series():
    d = build_report_data(
        _history(), _forecasts(), pandas_freq="MS", frequency="monthly", static_columns=DIMS,
        by_model=_by_model(),
    )
    s = {x["id"]: x for x in d["series"]}
    assert d["meta"]["model_names"] == ["naive", "drift"]  # "never" has no line; static columns are not models
    assert s["a"]["model_fc"] == {"naive": [1.0] * 3, "drift": [2.0] * 3}
    assert list(s["c"]["model_fc"]) == ["naive"]
    json.dumps(d, allow_nan=False)


def test_without_by_model_there_are_no_model_lines():
    d = _data()
    assert d["meta"]["model_names"] == [] and all(s["model_fc"] == {} for s in d["series"])


def test_by_model_needs_uuid_and_ds():
    with pytest.raises(DataContractError, match="uuid"):
        build_report_data(_history(), _forecasts(), pandas_freq="MS", frequency="monthly",
                          by_model=_by_model().drop(columns="uuid"))


def test_report_pipeline_includes_model_lines_unless_switched_off(tmp_path):
    input_csv = tmp_path / "in.csv"
    _history().to_csv(input_csv, index=False)
    out = tmp_path / "outputs" / "monthly"
    out.mkdir(parents=True)
    _forecasts().to_csv(out / "forecasts.csv", index=False)
    _by_model().to_csv(out / "forecasts_by_model.csv", index=False)
    cfg = load_config({"data_path": str(input_csv), "static_columns": list(DIMS), "registry_dir": tmp_path / "none"})
    with_lines = json.loads(run_report_pipeline(cfg, output_dir=tmp_path / "outputs").json_path.read_text())
    assert with_lines["meta"]["model_names"] == ["naive", "drift"]
    without = json.loads(run_report_pipeline(cfg, output_dir=tmp_path / "outputs", model_lines=False).json_path.read_text())
    assert without["meta"]["model_names"] == []


# ---- forecast check flags ---------------------------------------------------------------------


def _checks():
    return pd.DataFrame({
        "uuid": ["b", "b", "c"], "check": ["outlier", "total_change", "zero"],
        "ds": ["2024-08-01", None, None],
        "message": ["forecast step 2 is 6.0 robust standard deviations above its neighbours",
                    "forecast total is 4.00x the actual total", "the whole forecast is 0"],
    })


def test_flags_are_attached_per_series():
    data = _data(checks=_checks())
    flags = {s["id"]: s["flags"] for s in data["series"]}
    assert flags["a"] == []
    assert [f["check"] for f in flags["b"]] == ["outlier", "total_change"]
    assert flags["b"][0]["ds"] == "2024-08-01" and flags["b"][1]["ds"] is None
    assert data["meta"]["checks_ran"] is True and data["meta"]["n_flagged"] == 2


def test_without_checks_file_the_report_says_checks_did_not_run():
    data = _data()
    assert data["meta"]["checks_ran"] is False and data["meta"]["n_flagged"] == 0
    assert all(s["flags"] == [] for s in data["series"])
    assert _data(checks=_checks().iloc[:0])["meta"]["checks_ran"] is True  # ran, nothing flagged


def test_checks_need_uuid_check_and_message():
    with pytest.raises(DataContractError, match="forecast_checks"):
        _data(checks=_checks().drop(columns=["message"]))


def test_report_pipeline_reads_the_checks_file(tmp_path):
    input_csv = tmp_path / "in.csv"
    _history().to_csv(input_csv, index=False)
    out = tmp_path / "outputs" / "monthly"
    out.mkdir(parents=True)
    _forecasts().to_csv(out / "forecasts.csv", index=False)
    _checks().to_csv(out / "forecast_checks.csv", index=False)
    cfg = load_config({"data_path": str(input_csv), "static_columns": list(DIMS), "registry_dir": tmp_path / "registry"})
    data = json.loads(run_report_pipeline(cfg, output_dir=tmp_path / "outputs").json_path.read_text())
    assert data["meta"]["n_flagged"] == 2
    assert "flagbox" in (out / "forecast_report.html").read_text(encoding="utf-8")
