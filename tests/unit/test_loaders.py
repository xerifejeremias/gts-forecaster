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


# ---- load_input: column mapping, extra columns, CSV / Parquet / SQL ------------------------------


def _real_like(n_series=3, n_periods=12):
    """Input shaped like the first real data: unique_id / ds / y, upper-case static columns, ids that
    are not UUIDs, and a column the pipeline does not use."""
    df = make_panel(n_series, n_periods, explanatory=())
    codes = {u: f"DE04_F0101_TMO{k}" for k, u in enumerate(df["uuid"].unique())}
    df["uuid"] = df["uuid"].map(codes)
    df = df.rename(columns={"uuid": "unique_id", "date": "ds", "target": "y"})
    df["MARKET_CODE"], df["PRODUCT_CATEGORY_CODE"] = "DE04", "F0101"
    df["TMO"] = df["unique_id"].str[-4:]
    df["COMMENT"] = "booked late"  # not used by the pipeline
    df.loc[df.index[3], "y"] = -5.0  # a restatement
    return df


def _real_config(tmp_path, data_path, **overrides):
    from gts_forecaster.config import load_config

    options = {"frequency": "monthly", "data_path": str(data_path), "explanatory_columns": [],
               "static_columns": ["MARKET_CODE", "PRODUCT_CATEGORY_CODE", "TMO"],
               "input_columns": {"id": "unique_id", "date": "ds", "target": "y"}, **overrides}
    return load_config(options)


def _check_prepared(cfg):
    from gts_forecaster.data_preparation import prepare_input_data

    prepared = prepare_input_data(cfg)
    assert prepared.data["unique_id"].nunique() == 3 and (prepared.data["y"] < 0).any()
    assert list(prepared.static.columns) == ["unique_id", "MARKET_CODE", "PRODUCT_CATEGORY_CODE", "TMO"]
    assert prepared.data["unique_id"].str.startswith("DE04_F0101_TMO").all()  # no UUID needed


def test_csv_with_mapped_columns_and_an_extra_column(tmp_path, caplog):
    import logging

    from gts_forecaster.io.loaders import load_input

    path = tmp_path / "real.csv"
    _real_like().to_csv(path, index=False)
    cfg = _real_config(tmp_path, path)
    with caplog.at_level(logging.INFO):
        df = load_input(cfg)
    assert list(df.columns)[:3] == ["uuid", "date", "target"] and "COMMENT" not in df.columns
    assert "COMMENT" in caplog.text
    _check_prepared(cfg)


def test_parquet_input(tmp_path):
    path = tmp_path / "real.parquet"
    _real_like().to_parquet(path, index=False)
    cfg = _real_config(tmp_path, path)  # data_source auto: from the suffix
    _check_prepared(cfg)


def test_sql_input_with_the_connection_in_an_environment_variable(tmp_path, monkeypatch):
    import sqlalchemy

    db = tmp_path / "volumes.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    _real_like().to_sql("volumes_monthly", engine, index=False)
    engine.dispose()
    monkeypatch.setenv("GTS_SQL_URL", f"sqlite:///{db}")
    cfg = _real_config(tmp_path, tmp_path / "unused.csv", sql_query="SELECT * FROM volumes_{frequency}")
    _check_prepared(cfg)


def test_sql_without_a_connection_says_what_to_set(tmp_path, monkeypatch):
    from gts_forecaster.io.loaders import load_input

    monkeypatch.delenv("GTS_SQL_URL", raising=False)
    cfg = _real_config(tmp_path, tmp_path / "x.csv", sql_query="SELECT 1")
    with pytest.raises(DataContractError, match="GTS_SQL_URL"):
        load_input(cfg)


def test_extra_columns_error_and_missing_mapped_column(tmp_path):
    from gts_forecaster.io.loaders import load_input

    path = tmp_path / "real.csv"
    _real_like().to_csv(path, index=False)
    with pytest.raises(DataContractError, match="COMMENT"):
        load_input(_real_config(tmp_path, path, extra_columns="error"))
    with pytest.raises(DataContractError, match="volume"):
        load_input(_real_config(tmp_path, path, input_columns={"target": "volume"}))


def test_id_format_uuid_rejects_other_ids(tmp_path):
    from gts_forecaster.data_preparation import prepare_input_data

    path = tmp_path / "real.csv"
    _real_like().to_csv(path, index=False)
    with pytest.raises(DataContractError, match="uuid"):
        prepare_input_data(_real_config(tmp_path, path, id_format="uuid"))


def test_input_settings_are_validated(tmp_path):
    from gts_forecaster.config import ConfigError, load_config

    for bad in ({"data_source": "excel"}, {"input_columns": {"weight": "w"}},
                {"input_columns": {"id": "x", "date": "x"}}, {"extra_columns": "keep"}, {"id_format": "int"}):
        with pytest.raises(ConfigError):
            load_config({"frequency": "monthly", **bad})


def test_password_is_masked():
    from gts_forecaster.io.loaders import masked_url

    assert masked_url("postgresql://pedro:s3cret@db.local:5432/sales") == "postgresql://pedro:***@db.local:5432/sales"
    assert masked_url("sqlite:///volumes.db") == "sqlite:///volumes.db"
