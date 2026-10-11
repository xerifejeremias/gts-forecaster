from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gts_forecaster.config import ConfigError, load_config
from gts_forecaster.contracts.errors import DataContractError
from gts_forecaster.contracts.static_checks import check_static_columns, extract_static_table
from gts_forecaster.features.builder import (
    FeatureBuilder,
    FeatureSet,
    build_feature_builders,
    load_feature_config,
)
from gts_forecaster.features.errors import FeatureError
from gts_forecaster.features.static import (
    StaticSpec,
    check_spec,
    feature_names,
    static_feature_table,
    with_code_parts,
)
from gts_forecaster.models.base import BaseForecaster
from gts_forecaster.models.catalog import MODEL_TYPES, build_models_from_config

# four series: two markets x two products, one company each
STATIC = pd.DataFrame(
    {
        "unique_id": ["s1", "s2", "s3", "s4"],
        "market": ["DE04", "DE04", "PT04", "PT04"],
        "product": ["F0101", "M0118", "F0101", "F0115"],
        "company": ["company A", "company B", "company A", "company A"],
    }
)
LEVELS = {"s1": 100.0, "s2": 300.0, "s3": 50.0, "s4": 150.0}


def _train(n=24, levels=LEVELS):
    ds = pd.date_range("2022-01-01", periods=n, freq="MS")
    frames = []
    for uid, level in levels.items():
        y = level + np.arange(n) * 0.0 + (np.arange(n) % 2) * 2.0  # mean = level + 1
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": y}))
    return pd.concat(frames, ignore_index=True)


# ---- input side -------------------------------------------------------------------------------


def _raw():
    ds = pd.date_range("2024-01-01", periods=3, freq="MS")
    return pd.DataFrame(
        {
            "uuid": ["a"] * 3 + ["b"] * 3,
            "date": list(ds) * 2,
            "market": ["DE04"] * 3 + ["PT04"] * 3,
            "company": ["company A"] * 6,
        }
    )


def test_static_values_must_be_constant_per_series_and_not_blank():
    check_static_columns(_raw(), ["market", "company"])
    changing = _raw()
    changing.loc[1, "market"] = "IT04"
    with pytest.raises(DataContractError, match="one value per uuid"):
        check_static_columns(changing, ["market"])
    for bad in (None, "", "   "):
        blank = _raw()
        blank.loc[0, "market"] = bad
        with pytest.raises(DataContractError, match="empty values"):
            check_static_columns(blank, ["market"])


def test_static_table_has_one_row_per_series_in_order():
    table = extract_static_table(_raw().assign(market=lambda d: d.market + " "), ["market", "company"])
    assert table["uuid"].tolist() == ["a", "b"]
    assert table["market"].tolist() == ["DE04", "PT04"]  # whitespace trimmed


@pytest.mark.parametrize(
    "overrides",
    [
        {"static_columns": ["x_temp"]},  # clashes with an explanatory column
        {"static_columns": ["target"]},  # reserved name
        {"static_columns": ["market", "market"]},
        {"static_columns": "market"},
    ],
)
def test_static_columns_config_is_validated(overrides):
    with pytest.raises(ConfigError):
        load_config({"frequency": "monthly", **overrides})


def test_static_columns_config_default_in_project_and_override():
    assert load_config({"frequency": "monthly"}).static_columns == ("market", "product", "company")
    assert load_config({"frequency": "monthly", "static_columns": ["market"]}).static_columns == ("market",)


# ---- features ---------------------------------------------------------------------------------


def test_code_parts_split_structured_codes():
    parts = with_code_parts(STATIC, {"market": [2], "product": [1, 3]})
    assert parts["market_p2"].tolist() == ["DE", "DE", "PT", "PT"]
    assert parts["product_p1"].tolist() == ["F", "M", "F", "F"]
    assert parts["product_p3"].tolist() == ["F01", "M01", "F01", "F01"]
    with pytest.raises(FeatureError):
        with_code_parts(STATIC, {"nope": [1]})
    with pytest.raises(FeatureError):
        with_code_parts(STATIC, {"market": [0]})


def test_ordinal_identifiers_are_stable_integer_codes():
    spec = StaticSpec(identifiers=("market", "product"))
    table = static_feature_table(_train(), STATIC, spec)
    assert table["id_market"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert table["id_product"].tolist() == [0.0, 2.0, 0.0, 1.0]  # F0101, F0115, M0118 sorted
    # a fold that only contains some series still uses the codes of the full static table
    part = static_feature_table(_train(levels={"s3": 50.0, "s2": 300.0}), STATIC, spec)
    assert part.loc["s3", "id_market"] == 1.0 and part.loc["s2", "id_product"] == 2.0


def test_onehot_identifiers():
    spec = StaticSpec(identifiers=("company",), encoding="onehot")
    table = static_feature_table(_train(), STATIC, spec)
    assert list(table.columns) == ["company_company_A", "company_company_B"]
    assert table["company_company_A"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert (table.sum(axis=1) == 1).all()
    assert feature_names(spec, check_spec(spec, STATIC)) == list(table.columns)


def test_group_statistics_use_each_series_once():
    spec = StaticSpec(group_by=(("market",),), group_stats=("mean", "median", "n_series", "rel_level"))
    t = static_feature_table(_train(), STATIC, spec)
    # series means are level + 1
    assert t.loc["s1", "grp_market_mean"] == pytest.approx((101 + 301) / 2)
    assert t.loc["s3", "grp_market_mean"] == pytest.approx((51 + 151) / 2)
    assert t["grp_market_n_series"].tolist() == [2, 2, 2, 2]
    assert t.loc["s1", "grp_market_rel_level"] == pytest.approx(101 / 201)
    assert t.loc["s2", "grp_market_rel_level"] == pytest.approx(301 / 201)


def test_long_series_do_not_dominate_a_group():
    train = pd.concat([_train(24, {"s1": 100.0}), _train(6, {"s2": 300.0})], ignore_index=True)
    spec = StaticSpec(group_by=(("market",),), group_stats=("mean",))
    t = static_feature_table(train, STATIC, spec)
    assert t.loc["s1", "grp_market_mean"] == pytest.approx((101 + 301) / 2)


def test_combined_groups_and_code_part_groups():
    spec = StaticSpec(
        code_parts={"product": (1,)},
        group_by=(("market", "product_p1"), ("product_p1",)),
        group_stats=("n_series", "mean"),
    )
    t = static_feature_table(_train(), STATIC, spec)
    assert list(t.columns) == [
        "grp_market_x_product_p1_n_series", "grp_market_x_product_p1_mean",
        "grp_product_p1_n_series", "grp_product_p1_mean",
    ]
    assert t["grp_market_x_product_p1_n_series"].tolist() == [1, 1, 2, 2]  # (DE,F) (DE,M) (PT,F) (PT,F)
    assert t["grp_product_p1_n_series"].tolist() == [3, 1, 3, 3]
    assert t.loc["s3", "grp_market_x_product_p1_mean"] == pytest.approx((51 + 151) / 2)


def test_series_statistics():
    n = 24
    ds = pd.date_range("2022-01-01", periods=n, freq="MS")
    y = 100 + 5.0 * np.arange(n)
    seasonal = 50 + 20 * np.tile([1.0, -1.0, 0.0, 0.5], n // 4)
    zeros = np.where(np.arange(n) % 4 == 0, 0.0, 10.0)
    train = pd.concat(
        [pd.DataFrame({"unique_id": u, "ds": ds, "y": v}) for u, v in
         (("s1", y), ("s2", seasonal), ("s3", zeros), ("s4", np.full(n, 7.0)))],
        ignore_index=True,
    )
    spec = StaticSpec(series_stats=("mean", "std", "cv", "length", "trend", "zero_share", "season_strength"))
    t = static_feature_table(train, STATIC, spec, season_length=4)
    assert t.loc["s1", "ser_mean"] == pytest.approx(y.mean())
    assert t.loc["s1", "ser_trend"] == pytest.approx(5.0 / y.mean())
    assert t.loc["s1", "ser_length"] == 24 and t.loc["s1", "ser_std"] == pytest.approx(np.std(y, ddof=1))
    assert t.loc["s3", "ser_zero_share"] == pytest.approx(0.25)
    assert t.loc["s2", "ser_season_strength"] == pytest.approx(1.0)
    assert t.loc["s1", "ser_season_strength"] < 0.2
    assert np.isnan(t.loc["s4", "ser_season_strength"]) and t.loc["s4", "ser_std"] == 0.0


def test_statistics_only_use_the_training_data_passed_in():
    spec = StaticSpec(group_by=(("market",),), series_stats=("mean",))
    full = _train(24)
    later = full.copy()
    later.loc[later.ds >= "2023-07-01", "y"] += 1000  # a change in the second half
    a = static_feature_table(full[full.ds < "2023-07-01"], STATIC, spec)
    b = static_feature_table(later[later.ds < "2023-07-01"], STATIC, spec)
    pd.testing.assert_frame_equal(a, b)  # nothing from after the cutoff is used


@pytest.mark.parametrize(
    "spec",
    [
        StaticSpec(identifiers=("country",)),
        StaticSpec(group_by=(("market", "country"),)),
        StaticSpec(identifiers=("market",), encoding="hash"),
        StaticSpec(group_by=(("market",),), group_stats=("mode",)),
        StaticSpec(series_stats=("kurtosis",)),
    ],
)
def test_invalid_static_specs_are_rejected(spec):
    with pytest.raises(FeatureError):
        static_feature_table(_train(), STATIC, spec)


def test_series_without_static_values_are_reported():
    with pytest.raises(FeatureError, match="no static values"):
        static_feature_table(_train(levels={"s1": 1.0, "zzz": 2.0}), STATIC, StaticSpec(identifiers=("market",)))


# ---- builder ------------------------------------------------------------------------------------


def _builder(**spec):
    fs = FeatureSet(name="g", calendar=("month",), lags=(1,), static=StaticSpec(**spec))
    return FeatureBuilder(fs, pandas_freq="MS", season_length=12, static=STATIC)


def test_builder_adds_static_columns_to_history_and_future():
    b = _builder(identifiers=("market",), group_by=(("market",),), group_stats=("mean",))
    train = _train()
    out = b.transform_history(train)
    assert b.static_columns == ["id_market", "grp_market_mean"]
    assert out.columns.tolist()[-4:] == ["month", "id_market", "grp_market_mean", "lag_1"]
    assert len(out) == len(train)
    s3 = out[out.unique_id == "s3"]
    assert (s3["id_market"] == 1.0).all() and (s3["grp_market_mean"] == s3["grp_market_mean"].iloc[0]).all()
    future = pd.DataFrame({"unique_id": ["s1", "s3"], "ds": pd.to_datetime(["2024-01-01"] * 2), "horizon_step": 1})
    fx = b.transform_future(train, future)
    assert fx["id_market"].tolist() == [0.0, 1.0]
    assert fx["grp_market_mean"].tolist() == pytest.approx([201.0, 101.0])


def test_static_columns_are_usable_at_every_step():
    b = _builder(identifiers=("market",))
    assert "id_market" in b.usable_at_step(1) and "id_market" in b.usable_at_step(50)
    assert b.feature_columns == ["month", "id_market", "lag_1"]


def test_builder_needs_the_static_table_when_the_set_uses_static_features():
    fs = FeatureSet(name="g", static=StaticSpec(identifiers=("market",)))
    for missing in (None, pd.DataFrame({"unique_id": []})):
        with pytest.raises(FeatureError, match="static_columns"):
            FeatureBuilder(fs, pandas_freq="MS", season_length=12, static=missing)
    with pytest.raises(FeatureError, match="unknown column"):
        FeatureBuilder(FeatureSet(name="g", static=StaticSpec(identifiers=("zzz",))),
                       pandas_freq="MS", season_length=12, static=STATIC)


def test_sets_without_static_features_work_without_a_static_table():
    b = FeatureBuilder(FeatureSet(name="c", calendar=("month",)), pandas_freq="MS", season_length=12)
    assert b.static_columns == []
    assert "month" in b.transform_history(_train()).columns


def test_shipped_global_static_set_builds_for_the_synthetic_columns():
    cfg = load_feature_config()
    assert "global_static" in cfg.sets
    spec = cfg.sets["global_static"].static
    assert spec.code_parts == {"market": (2,), "product": (1, 3)}
    assert spec.group_by[-1] == ("market", "product_p1")
    static = pd.DataFrame({"unique_id": ["a", "b"], "market": ["DE04", "PT04"],
                           "product": ["F0101", "M0118"], "company": ["company A", "company B"]})
    for freq, season in (("YS", 1), ("MS", 12), ("W-MON", 52), ("D", 7)):
        b = build_feature_builders(cfg, ["global_static"], pandas_freq=freq, season_length=season,
                                   static=static)["global_static"]
        ds = pd.date_range("2022-01-01", periods=30, freq=freq)
        train = pd.concat([pd.DataFrame({"unique_id": u, "ds": ds, "y": np.arange(30.0) + i})
                           for i, u in enumerate(["a", "b"])], ignore_index=True)
        out = b.transform_history(train)
        assert set(b.static_columns) <= set(out.columns) and len(b.static_columns) == 23


def test_feature_file_rejects_bad_static_sections(tmp_path):
    base = "sets:\n  g:\n    static:\n"
    for body in ("      colour: [market]\n", "      group_stats:\n        how: [market]\n",
                 "      code_parts:\n        market: 2\n", "      group_stats:\n        by: [[3]]\n"):
        (tmp_path / "features.yaml").write_text(base + body)
        with pytest.raises(FeatureError):
            load_feature_config(tmp_path)


# ---- models and pipelines -----------------------------------------------------------------------


class _StaticUser(BaseForecaster):
    def __init__(self, feature_builder=None):
        self.feature_builder = feature_builder

    def _fit(self, series, train):
        self.train_ = train
        self.last_ = {u: y[-1] for u, y in series.items()}

    def _predict(self, future):
        self.seen_ = self.feature_builder.transform_future(self.train_, future)
        return future["unique_id"].map(self.last_).to_numpy(dtype=float)


def test_pipeline_gives_the_static_table_to_the_feature_builders(tmp_path, monkeypatch):
    from gts_forecaster.pipelines.backtest import run_backtest_pipeline

    monkeypatch.setitem(MODEL_TYPES, "static_user", _StaticUser)
    for name in ("default.yaml", "frequency_profiles.yaml", "features.yaml"):
        (tmp_path / name).write_text(open(f"config/{name}").read())
    (tmp_path / "models.yaml").write_text(
        "naive:\n  type: naive\ndrift:\n  type: drift\n"
        "global_a:\n  type: static_user\n  features: global_static\n"
    )
    cfg = load_config({"frequency": "monthly", "horizon": 3, "n_folds": 1, "top_n": 1,
                       "registry_dir": tmp_path / "reg", "backtesting_dir": tmp_path / "bt"},
                      config_dir=tmp_path)
    static = STATIC.assign(product=["F0101", "M0118", "F0101", "F0115"])
    df = _train(36).assign(is_filled=False)
    prepared = SimpleNamespace(data=df, config=cfg, cutoff=df["ds"].max(), static=static,
                               fallback=pd.DataFrame(columns=["unique_id"]))
    out = run_backtest_pipeline(prepared, config_dir=tmp_path)
    assert "global_a" in set(out.result.predictions["model"])
    # without a static table the same configuration is refused with a clear message
    prepared.static = pd.DataFrame({"unique_id": []})
    with pytest.raises(ConfigError, match="static_columns"):
        run_backtest_pipeline(prepared, config_dir=tmp_path)
