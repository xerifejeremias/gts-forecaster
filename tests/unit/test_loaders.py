import pandas as pd
import pytest

from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.io.loaders import load_input_csv
from helpers import make_panel


def test_loads_valid_csv_with_expected_types(tmp_path):
    path = tmp_path / "in.csv"
    make_panel().to_csv(path, index=False)
    df = load_input_csv(path)
    assert list(df.columns) == ["uuid", "date", "target", "x1", "x2"]
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert df["uuid"].map(type).eq(str).all()
    assert len(df) == 16


def test_missing_file_is_a_contract_error(tmp_path):
    with pytest.raises(DataContractError, match="not found"):
        load_input_csv(tmp_path / "nope.csv")


def test_file_without_date_column_is_a_contract_error(tmp_path):
    path = tmp_path / "in.csv"
    pd.DataFrame({"uuid": ["a"], "target": [1.0]}).to_csv(path, index=False)
    with pytest.raises(DataContractError, match="Could not read"):
        load_input_csv(path)


def test_empty_file_is_a_contract_error(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("")
    with pytest.raises(DataContractError, match="Could not read"):
        load_input_csv(path)


def test_unparseable_dates_are_left_for_the_contract_to_reject(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("uuid,date,target\na,not-a-date,1\n")
    df = load_input_csv(path)
    assert not pd.api.types.is_datetime64_any_dtype(df["date"])


def test_static_columns_are_read_as_text(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("uuid,date,target,market,company\na,2024-01-01,1.0,0042,7\n")
    df = load_input_csv(path, ["market", "company"])
    assert df["market"].iloc[0] == "0042" and df["company"].iloc[0] == "7"
    assert load_input_csv(path)["market"].iloc[0] == 42  # without the declaration pandas would guess a number
