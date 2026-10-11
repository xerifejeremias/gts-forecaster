"""The synthetic datasets shipped in data/ must pass the full input contract."""

import pandas as pd
import pytest

from gts_forecaster.config import load_config
from gts_forecaster.data_preparation import prepare_input_data

EXPECTED_SERIES = {"yearly": 40, "monthly": 40, "weekly": 50, "daily": 55}


@pytest.mark.parametrize("frequency", sorted(EXPECTED_SERIES))
def test_synthetic_file_passes_the_input_contract(frequency):
    config = load_config({"frequency": frequency})
    if not config.data_path.is_file():
        pytest.skip(f"{config.data_path} not present")
    prepared = prepare_input_data(config)

    assert prepared.fallback.empty and prepared.report.empty  # synthetic data is complete
    assert not prepared.data["is_filled"].any()
    assert prepared.data["unique_id"].nunique() == EXPECTED_SERIES[frequency]

    metadata = pd.read_csv(config.data_path.parent / "series_metadata.csv")
    counts = prepared.data.groupby("unique_id").size()
    assert counts.loc[metadata["uuid"]].tolist() == metadata["n_obs"].tolist()


@pytest.mark.parametrize("frequency", sorted(EXPECTED_SERIES))
def test_synthetic_files_carry_static_columns(frequency):
    config = load_config({"frequency": frequency})
    if not config.data_path.is_file():
        pytest.skip(f"{config.data_path} not present")
    prepared = prepare_input_data(config)
    assert list(prepared.static.columns) == ["unique_id", *config.static_columns]
    assert prepared.static["unique_id"].nunique() == len(prepared.static) == EXPECTED_SERIES[frequency]
    assert not set(config.static_columns) & set(prepared.data.columns)  # kept apart from the series
    assert prepared.static["market"].str.match(r"^[A-Z]{2}04$").all()


@pytest.mark.parametrize("frequency", ["yearly", "monthly", "weekly", "daily"])
def test_synthetic_files_cover_every_market_product_company(frequency):
    """Sample data is a full grid: 3 markets x 2 products x 2 companies, each with series."""
    config = load_config({"frequency": frequency})
    if not config.data_path.is_file():
        pytest.skip(f"{config.data_path} not present")
    import pandas as pd

    series = pd.read_csv(config.data_path, dtype=str).drop_duplicates("uuid")
    assert set(series["market"]) == {"DE04", "PT04", "IT04"}
    assert set(series["product"]) == {"F0101", "M0101"}
    assert set(series["company"]) == {"company A", "company B"}
    counts = series.groupby(["market", "product", "company"]).size()
    assert len(counts) == 12 and counts.min() >= 1
