"""Full input path: load -> contract -> repair -> contract again. Uses pandera."""

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.data_preparation import load_prepared, main, prepare_input_data
from helpers import EXPLANATORY, drop_rows, make_panel, make_uuid


def config_for(tmp_path, df, **overrides):
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    options = {
        "frequency": "monthly",
        "data_path": str(path),
        "explanatory_columns": list(EXPLANATORY),
        "static_columns": [],  # the test files have none unless a test adds them
    }
    options.update(overrides)
    return load_config(options)


def test_complete_file_is_prepared(tmp_path):
    prepared = prepare_input_data(config_for(tmp_path, make_panel()))
    assert prepared.data["unique_id"].nunique() == 2
    assert prepared.fallback.empty and prepared.report.empty


def test_series_ending_early_is_forward_filled_to_the_cutoff(tmp_path):
    df = drop_rows(make_panel(), make_uuid(1), ["2024-08-01"])
    prepared = prepare_input_data(config_for(tmp_path, df))
    assert prepared.data.groupby("unique_id")["ds"].max().nunique() == 1
    assert prepared.report["action"].tolist() == ["forward_fill"]


def test_long_gap_gives_a_fallback_not_a_failure(tmp_path):
    df = drop_rows(make_panel(), make_uuid(0), ["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"])
    prepared = prepare_input_data(config_for(tmp_path, df))
    assert prepared.fallback["unique_id"].tolist() == [make_uuid(0)]


def test_too_short_series_gets_the_last_value_not_a_failure(tmp_path, caplog):
    # a brand-new series with 2 months of data (min_series_length is 3 for monthly)
    df = make_panel(3)
    new = make_uuid(2)
    df = df[(df["uuid"] != new) | (df["date"] >= df["date"].max() - pd.DateOffset(months=1))]
    last = df.loc[df["uuid"] == new].sort_values("date")["target"].iloc[-1]
    prepared = prepare_input_data(config_for(tmp_path, df))
    assert set(prepared.data["unique_id"]) == {make_uuid(0), make_uuid(1)}
    row = prepared.fallback.set_index("unique_id").loc[new]
    assert row["fallback_value"] == pytest.approx(last) and "fewer than" in row["reason"]
    assert "no model is applied" in caplog.text


def test_only_too_short_series_stops_the_run(tmp_path):
    with pytest.raises(DataContractError, match="Every series has fewer than"):
        prepare_input_data(config_for(tmp_path, make_panel(2, 2)))


def test_missing_file_stops_the_pipeline(tmp_path):
    config = load_config({"frequency": "monthly", "data_path": str(tmp_path / "nope.csv")})
    with pytest.raises(DataContractError, match="not found"):
        prepare_input_data(config)


def test_broken_data_stops_the_pipeline(tmp_path):
    df = make_panel()
    df.loc[1, "target"] = np.inf
    with pytest.raises(DataContractError):
        prepare_input_data(config_for(tmp_path, df))


def test_command_line_reports_ok_and_failure(tmp_path, capsys):
    good = tmp_path / "good.csv"
    args = ["--frequency", "monthly", "--data-path", str(good), "--static-columns"]
    # the default config expects x_temp, x_promo, x_price (and no static columns here), so build a matching file
    df = make_panel(explanatory=("x_temp", "x_promo", "x_price"))
    df.to_csv(good, index=False)
    assert main(args) == 0
    assert "OK" in capsys.readouterr().out

    df.loc[0, "target"] = np.inf
    df.to_csv(good, index=False)
    assert main(args) == 1
    assert "FAILED" in capsys.readouterr().err


def test_prepared_data_is_in_the_model_format(tmp_path):
    prepared = prepare_input_data(config_for(tmp_path, make_panel()))
    assert list(prepared.data.columns) == ["unique_id", "ds", "y", *EXPLANATORY, "is_filled"]
    assert str(prepared.data["ds"].dtype) == "datetime64[ns]"
    assert prepared.data["y"].dtype == "float64"
    assert not {"uuid", "date", "target"} & set(prepared.data.columns)


def test_load_prepared_reads_the_file_of_the_chosen_frequency():
    for frequency in ["yearly", "monthly", "weekly", "daily"]:
        prepared = load_prepared(frequency)
        assert prepared.config.frequency == frequency
        assert prepared.config.data_path.name == f"{frequency}_synthetic.csv"
        assert list(prepared.data.columns[:3]) == ["unique_id", "ds", "y"]


def test_static_columns_are_kept_apart_from_the_series(tmp_path):
    df = make_panel()
    df["market"] = df["uuid"].map({make_uuid(0): "DE04", make_uuid(1): "PT04"})
    df["company"] = "company A"
    prepared = prepare_input_data(config_for(tmp_path, df, static_columns=["market", "company"]))
    assert list(prepared.data.columns) == ["unique_id", "ds", "y", *EXPLANATORY, "is_filled"]
    assert list(prepared.static.columns) == ["unique_id", "market", "company"]
    assert prepared.static["market"].tolist() == ["DE04", "PT04"]


def test_static_value_that_changes_within_a_series_stops_the_run(tmp_path):
    df = make_panel()
    df["market"] = "DE04"
    df.loc[3, "market"] = "PT04"
    with pytest.raises(DataContractError, match="one value per uuid"):
        prepare_input_data(config_for(tmp_path, df, static_columns=["market"]))


def test_missing_static_column_stops_the_run(tmp_path):
    with pytest.raises(DataContractError):
        prepare_input_data(config_for(tmp_path, make_panel(), static_columns=["market"]))
