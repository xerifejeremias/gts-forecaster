import pytest

from gts_forecaster.config import ConfigError, load_config, load_profiles, project_root


def test_defaults_load_from_repo_config():
    config = load_config()
    assert config.frequency == "monthly"
    assert config.horizon == config.profile.default_horizon == 12
    assert config.max_fill_periods == 3
    assert config.fallback == "last_value"
    assert config.value_columns == ["y", "x_temp", "x_promo", "x_price"]
    assert config.data_path == project_root() / "data" / "monthly" / "monthly_synthetic.csv"


def test_all_four_frequencies_have_profiles():
    profiles = load_profiles()
    assert set(profiles) == {"yearly", "monthly", "weekly", "daily"}
    assert profiles["weekly"].pandas_freq == "W-MON"


def test_frequency_and_horizon_overrides():
    config = load_config({"frequency": "daily", "horizon": 30})
    assert config.frequency == "daily"
    assert config.horizon == 30
    assert config.data_path.name == "daily_synthetic.csv"


def test_none_override_means_not_given():
    config = load_config({"frequency": None, "horizon": None, "fallback": None})
    assert config.frequency == "monthly"
    assert config.fallback == "last_value"


def test_fallback_zero_and_max_fill_override():
    config = load_config({"fallback": "zero", "max_fill_periods": 0})
    assert config.fallback == "zero"
    assert config.max_fill_periods == 0


@pytest.mark.parametrize(
    "override",
    [
        {"frequency": "hourly"},
        {"horizon": 0},
        {"horizon": -3},
        {"horizon": 2.5},
        {"horizon": True},
        {"fallback": "mean"},
        {"max_fill_periods": -1},
        {"max_fill_periods": 1.5},
        {"explanatory_columns": ["x1", "x1"]},
        {"explanatory_columns": ["target"]},
        {"explanatory_columns": ["y"]},
        {"explanatory_columns": ["ds"]},
        {"explanatory_columns": ["unique_id"]},
        {"explanatory_columns": ["is_filled"]},
        {"explanatory_columns": "x1"},
        {"explanatory_columns": [""]},
        {"unknown_option": 1},
    ],
)
def test_invalid_config_is_rejected(override):
    with pytest.raises(ConfigError):
        load_config(override)


def test_missing_config_dir_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(config_dir=tmp_path / "nope")


def test_backtesting_dir_defaults_to_project_folder():
    cfg = load_config({"frequency": "monthly", "horizon": 3})
    assert cfg.backtesting_dir.name == "backtesting" and cfg.backtesting_dir.is_absolute()


# ---- ml_features in the frequency profiles ---------------------------------------------------


def test_shipped_profiles_ml_features():
    profiles = load_profiles()
    assert profiles["daily"].ml_features["trees"]["lags"] == [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    assert profiles["daily"].ml_features["linear"]["rolling_windows"] == [7, 28]
    assert set(profiles["yearly"].ml_features) == {"trees"}  # ridge/lasso keep the adapter default
    assert profiles["monthly"].ml_features == {} and profiles["weekly"].ml_features == {}


@pytest.mark.parametrize("bad", [
    "  ml_features: {forest: {lags: [1]}}\n",
    "  ml_features: {trees: {lags: [1], depth: 3}}\n",
    "  ml_features: {trees: {lags: [0, 1]}}\n",
    "  ml_features: {trees: {lags: []}}\n",
    "  ml_features: {trees: {lags: season}}\n",
    "  ml_features: {trees: {rolling_windows: [1, 7]}}\n",  # a rolling statistic needs >= 2 values
])
def test_invalid_ml_features_are_rejected(tmp_path, bad):
    text = open(project_root() / "config" / "frequency_profiles.yaml", encoding="utf-8").read()
    # replace the shipped yearly ml_features with the bad one
    start = text.index("yearly:")
    end = text.index("monthly:")
    yearly = "".join(l + "\n" for l in text[start:end].splitlines() if not l.startswith(("  ml_features", "    trees")))
    (tmp_path / "frequency_profiles.yaml").write_text(text[:start] + yearly + bad + text[end:], encoding="utf-8")
    with pytest.raises(ConfigError, match="ml_features"):
        load_profiles(tmp_path)
