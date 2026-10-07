"""End-to-end checks of src/data_prep.py on a small SYNTHETIC file in the real format.

They guard the properties the brief grades (no leakage columns, no listing in two
splits, threshold and encoder fitted on TRAIN only, deterministic) without needing
the Kaggle data in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_prep as dp
from tests.fake_house_sale import make_fake_house_sale


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "house_sale.csv"
    make_fake_house_sale(800).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def cleaned(csv_path):
    return dp.clean_data(csv_path)


def test_cleaning_removes_duplicates_leakage_and_bad_rows(cleaned):
    clean, cleaning_log, outlier_log = cleaned
    assert clean.isna().sum().sum() == 0
    assert not clean["listing_id"].duplicated().any()
    leaking = set(dp.LEAKAGE_COLUMNS) | {"owner_name", "address", "views", "description"}
    assert not leaking & set(clean.columns), "leakage/identifier columns must not survive"
    assert clean["lat"].between(*dp.AZ_LAT_RANGE).all()
    assert list(cleaning_log.columns) == ["step", "rows_removed", "rows_left"]
    assert cleaning_log["rows_left"].iloc[-1] == len(clean)
    assert outlier_log["rows_flagged"].sum() > 0


def test_clean_on_a_dataframe_matches_clean_data(csv_path, cleaned):
    from_df = dp.clean(dp.load_raw(csv_path))
    pd.testing.assert_frame_equal(from_df, cleaned[0])


def test_split_is_leakage_free(cleaned):
    clean = cleaned[0]
    train, val, test, threshold = dp.split_data(clean)
    ids = [set(part["listing_id"]) for part in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert len(train) + len(val) + len(test) == len(clean)
    assert abs(len(train) / len(clean) - 0.70) < 0.02
    assert threshold == pytest.approx(train["price"].median()), "threshold from TRAIN only"
    for part in (train, val, test):
        assert ((part["price"] > threshold).astype(int) == part["price_tier"]).all()


def test_encoder_layout_comes_from_train(cleaned):
    train, val, test, _ = dp.split_data(cleaned[0])
    encoder = dp.FeatureEncoder().fit(train)
    counts = train["location"].value_counts()
    assert encoder.frequent_locations_ == set(counts[counts >= dp.MIN_LOCATION_COUNT].index)
    for part in (val, test):
        assert encoder.transform(part).columns.tolist() == encoder.feature_names_


def test_eda_figures_are_written(csv_path, tmp_path):
    from src import eda

    stats = eda.make_figures(csv_path, tmp_path)
    for name in ("fig1_price_distribution", "fig2_area_rooms_vs_price",
                 "fig3_location_effects", "fig4_class_balance"):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000
    assert stats["rows"] > 0 and stats["tier_threshold"] > 0


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_load_splits_shapes_scaling_and_determinism(csv_path, task):
    a = dp.load_splits(task, scaled=True, data_path=csv_path)
    b = dp.load_splits(task, scaled=True, data_path=csv_path)
    n_features = len(a["feature_names"])
    for split in ("train", "val", "test"):
        assert a[f"X_{split}"].shape == (len(a[f"y_{split}"]), n_features)
        assert np.array_equal(a[f"X_{split}"], b[f"X_{split}"]), "same seed, same arrays"
    assert np.allclose(a["X_train"].mean(axis=0), 0.0, atol=1e-9)
    if task == "classification":
        assert set(np.unique(a["y_train"])) <= {0, 1}
    else:
        assert np.allclose(a["y_train"], np.log(a["train_df"]["price"]))
