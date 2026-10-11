"""Frozen-models mode: update without refit, saving/loading, train + forecast pipelines."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.forecasting import forecast_by_model
from gts_forecaster.models.base import BaseForecaster
from gts_forecaster.models.catalog import build_models, load_model_specs
from gts_forecaster.models.naive import DriftForecaster, NaiveForecaster, SeasonalNaiveForecaster
from gts_forecaster.models.store import (
    fitted_models_path, load_fitted_model, load_manifest, params_signature, save_fitted_models,
)
from gts_forecaster.splitting import forecast_split
from helpers import BASELINES

N = 24


def _panel(n=N, ids=("a", "b"), offset=0.0):
    dates = pd.date_range("2022-01-01", periods=n, freq="MS")
    frames = [
        pd.DataFrame({"unique_id": u, "ds": dates, "y": np.arange(1.0, n + 1) * (k + 1) + offset})
        for k, u in enumerate(ids)
    ]
    df = pd.concat(frames, ignore_index=True)
    df["is_filled"] = False
    return df


def _models():
    return build_models(load_model_specs(types=BASELINES), 12)


class Refits(BaseForecaster):
    """A model that cannot take new data without refitting; counts its fits."""

    fits = 0

    def _fit(self, series, train):
        type(self).fits += 1
        self.last_ = {u: y[-1] for u, y in series.items()}

    def _predict(self, future):
        return future["unique_id"].map(self.last_).to_numpy(dtype=float)


# ---- update ----------------------------------------------------------------------------------

def test_naive_family_supports_update():
    for model in (NaiveForecaster(), DriftForecaster(), SeasonalNaiveForecaster(season_length=12)):
        assert model.supports_update


def test_update_uses_newer_history_without_refit_for_naive():
    model = NaiveForecaster().fit(_panel(N))
    model.update(_panel(N + 3))
    future = forecast_split(_panel(N + 3), 2, "MS").future
    out = model.predict(future)
    assert out[out.unique_id == "a"]["yhat"].tolist() == [N + 3.0, N + 3.0]


def test_update_rejects_unknown_series_and_unsupported_models():
    model = NaiveForecaster().fit(_panel(N, ids=("a",)))
    with pytest.raises(ValueError, match="not fitted on"):
        model.update(_panel(N, ids=("a", "zzz")))
    with pytest.raises(NotImplementedError):
        Refits().fit(_panel(N)).update(_panel(N))
    with pytest.raises(RuntimeError):
        NaiveForecaster().update(_panel(N))


def test_update_checks_history_length():
    model = DriftForecaster().fit(_panel(N))
    short = _panel(1)
    with pytest.raises(ValueError, match="at least"):
        model.update(short)


# ---- forecast_by_model with fitted models ----------------------------------------------------

def test_fitted_models_are_updated_not_refitted():
    old = {n: m.clone().fit(_panel(N)) for n, m in _models().items()}
    split = forecast_split(_panel(N + 3), 2, "MS")
    timings = []
    wide = forecast_by_model(split, _models(), fitted=old, timings=timings)
    assert {t["status"] for t in timings} == {"updated"}
    assert wide[wide.unique_id == "a"]["naive"].iloc[0] == N + 3.0


def test_frozen_matches_refit_for_history_only_models():
    split = forecast_split(_panel(N + 3), 3, "MS")
    old = {n: m.clone().fit(_panel(N)) for n, m in _models().items()}
    frozen = forecast_by_model(split, _models(), fitted=old)
    refit = forecast_by_model(split, _models())
    pd.testing.assert_frame_equal(frozen, refit)


def test_model_without_update_is_refitted_with_a_warning(caplog):
    Refits.fits = 0
    models = {"refits": Refits()}
    fitted = {"refits": Refits().fit(_panel(N))}
    assert Refits.fits == 1
    timings = []
    with caplog.at_level(logging.WARNING):
        forecast_by_model(forecast_split(_panel(N + 1), 2, "MS"), models, fitted=fitted, timings=timings)
    assert Refits.fits == 2 and timings[0]["status"] == "refitted"
    assert "cannot take new data" in caplog.text


def test_series_not_in_saved_model_forces_refit(caplog):
    fitted = {"naive": NaiveForecaster().fit(_panel(N, ids=("a",)))}
    timings = []
    with caplog.at_level(logging.WARNING):
        wide = forecast_by_model(
            forecast_split(_panel(N, ids=("a", "b")), 2, "MS"), {"naive": NaiveForecaster()},
            fitted=fitted, timings=timings,
        )
    assert timings[0]["status"] == "refitted" and wide["naive"].notna().all()
    assert "not in the saved model" in caplog.text


def test_missing_saved_model_is_refitted():
    timings = []
    forecast_by_model(
        forecast_split(_panel(N), 2, "MS"), {"naive": NaiveForecaster()}, fitted={}, timings=timings
    )
    assert timings[0]["status"] == "refitted"


def test_without_fitted_status_stays_ok():
    timings = []
    forecast_by_model(forecast_split(_panel(N), 2, "MS"), {"naive": NaiveForecaster()}, timings=timings)
    assert timings[0]["status"] == "ok"


# ---- store -----------------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    fitted = {n: m.clone().fit(_panel(N)) for n, m in _models().items()}
    folder = fitted_models_path(tmp_path, "monthly")
    assert save_fitted_models(folder, fitted, {"frequency": "monthly", "data_cutoff": "2023-12-01"}) == []
    manifest = load_manifest(folder)
    assert manifest["frequency"] == "monthly" and set(manifest["models"]) == set(fitted)
    assert manifest["models"]["naive"]["supports_update"] is True
    loaded = load_fitted_model(folder, manifest, "drift")
    future = forecast_split(_panel(N), 3, "MS").future
    pd.testing.assert_frame_equal(loaded.predict(future), fitted["drift"].predict(future))
    assert load_fitted_model(folder, manifest, "nope") is None


def test_unpicklable_model_is_reported_not_fatal(tmp_path):
    model = NaiveForecaster().fit(_panel(N))
    model.handle = lambda: None  # lambdas cannot be pickled
    folder = tmp_path / "m"
    not_saved = save_fitted_models(folder, {"naive": model, "drift": DriftForecaster().fit(_panel(N))}, {})
    assert not_saved == ["naive"]
    manifest = load_manifest(folder)
    assert set(manifest["models"]) == {"drift"} and manifest["not_saved"] == ["naive"]
    assert not (folder / "naive.pkl").exists()


def test_stale_files_are_removed(tmp_path):
    folder = tmp_path / "m"
    save_fitted_models(folder, {"naive": NaiveForecaster().fit(_panel(N)), "drift": DriftForecaster().fit(_panel(N))}, {})
    save_fitted_models(folder, {"naive": NaiveForecaster().fit(_panel(N))}, {})
    assert sorted(p.name for p in folder.glob("*.pkl")) == ["naive.pkl"]


def test_missing_manifest_gives_clear_error(tmp_path):
    with pytest.raises(DataContractError, match="train"):
        load_manifest(tmp_path / "none")


def test_corrupt_pickle_returns_none(tmp_path):
    folder = tmp_path / "m"
    save_fitted_models(folder, {"naive": NaiveForecaster().fit(_panel(N))}, {})
    (folder / "naive.pkl").write_bytes(b"garbage")
    assert load_fitted_model(folder, load_manifest(folder), "naive") is None


def test_signature_changes_with_params():
    a, b = SeasonalNaiveForecaster(season_length=12), SeasonalNaiveForecaster(season_length=4)
    assert params_signature(a) != params_signature(b)
    assert params_signature(a) == params_signature(a.clone())
    json.loads(params_signature(a))


# ---- config ----------------------------------------------------------------------------------

def test_forecast_mode_defaults_and_validation():
    cfg = load_config()
    assert cfg.forecast_mode == "frozen_selection"
    assert cfg.fitted_models_dir.name == "fitted_models"
    assert load_config({"forecast_mode": "frozen_models"}).forecast_mode == "frozen_models"
    with pytest.raises(ConfigError):
        load_config({"forecast_mode": "magic"})


# ---- pipeline helpers ------------------------------------------------------------------------

def _saved(tmp_path, cutoff="2023-12-01", frequency="monthly"):
    from gts_forecaster.pipelines.forecast import load_frozen_models  # noqa: F401

    fitted = {n: m.clone().fit(_panel(N)) for n, m in _models().items()}
    cfg = load_config({"fitted_models_dir": tmp_path})
    save_fitted_models(
        fitted_models_path(tmp_path, "monthly"), fitted,
        {"frequency": frequency, "data_cutoff": cutoff},
    )
    return cfg


def _prepared(cfg, cutoff):
    from types import SimpleNamespace

    return SimpleNamespace(config=cfg, cutoff=pd.Timestamp(cutoff))


def test_load_frozen_models_returns_matching_models(tmp_path):
    from gts_forecaster.pipelines.forecast import load_frozen_models

    cfg = _saved(tmp_path)
    loaded = load_frozen_models(_prepared(cfg, "2024-03-01"), _models())
    assert set(loaded) == set(_models())


def test_load_frozen_models_rejects_older_data(tmp_path):
    from gts_forecaster.pipelines.forecast import load_frozen_models

    cfg = _saved(tmp_path)
    with pytest.raises(DataContractError, match="before"):
        load_frozen_models(_prepared(cfg, "2023-01-01"), _models())


def test_load_frozen_models_skips_models_whose_settings_changed(tmp_path, caplog):
    from gts_forecaster.pipelines.forecast import load_frozen_models

    cfg = _saved(tmp_path)
    changed = _models()
    changed["seasonal_naive"] = SeasonalNaiveForecaster(season_length=4)
    with caplog.at_level(logging.WARNING):
        loaded = load_frozen_models(_prepared(cfg, "2024-03-01"), changed)
    assert "seasonal_naive" not in loaded and "naive" in loaded
    assert "settings changed" in caplog.text


def test_load_frozen_models_without_training_explains_what_to_do(tmp_path):
    from gts_forecaster.pipelines.forecast import load_frozen_models

    cfg = load_config({"fitted_models_dir": tmp_path})
    with pytest.raises(DataContractError, match="train"):
        load_frozen_models(_prepared(cfg, "2024-03-01"), _models())


def test_selected_model_names_in_first_use_order():
    from gts_forecaster.pipelines.forecast import selected_model_names

    sel = {"series": {
        "a": {"models": [{"name": "drift", "weight": 0.5}, {"name": "naive", "weight": 0.5}]},
        "b": {"models": [{"name": "naive", "weight": 1.0}]},
    }}
    assert selected_model_names(sel) == ["drift", "naive"]
