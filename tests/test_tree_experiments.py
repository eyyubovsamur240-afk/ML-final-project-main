"""
Tests for the Task A "price per m2 / sot" target in tree_experiments.py.

Run from the repo root:
    python -m pytest tests -q
"""

import numpy as np

from src.decision_tree import DecisionTree
from src.tree_experiments import log_size, per_size_target, task_a_splits

NAMES = ["area_m2", "land_area_sot", "rooms"]


def test_log_size_uses_floor_area_then_land_area():
    X = np.array([
        [100.0, 0.0, 3],    # flat: floor area
        [150.0, 6.0, 4],    # house: floor area, not its land
        [0.0, 12.0, 0],     # land plot: area_m2 = 0 -> land area in sot
        [0.0, 0.0, 0],      # no size at all -> 0, keeps plain log(price)
    ])
    np.testing.assert_allclose(log_size(X, NAMES), np.log([100.0, 150.0, 12.0, 1.0]))


def test_errors_are_the_same_on_both_scales():
    # prediction on log(price) = prediction on log(price per m2) + log(size),
    # so the residuals (and RMSE) do not change
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(30, 200, 50), np.zeros(50), rng.integers(1, 5, 50)])
    y = rng.normal(12, 0.5, 50)
    _, y_per_size, *_ = per_size_target((X, y, X, y, X, y), NAMES)
    pred_per_size = rng.normal(7, 0.5, 50)
    pred = pred_per_size + log_size(X, NAMES)
    np.testing.assert_allclose(pred - y, pred_per_size - y_per_size)


def test_per_size_target_helps_a_tree_with_area():
    # price = 2000 AZN per m2 exactly: on log(price) a shallow tree can only
    # approximate area in steps, on log(price per m2) one leaf is exact
    X = np.column_stack([np.linspace(30, 300, 200), np.zeros(200), np.ones(200)])
    y = np.log(2000 * X[:, 0])
    splits = (X, y, X, y, X, y)
    plain = DecisionTree("regression", "mse", max_depth=2).fit(X, y)
    X_tr, y_tr, *_ = per_size_target(splits, NAMES)
    per_m2 = DecisionTree("regression", "mse", max_depth=2).fit(X_tr, y_tr)
    plain_rmse = np.sqrt(np.mean((plain.predict(X) - y) ** 2))
    per_m2_rmse = np.sqrt(np.mean((per_m2.predict(X) + log_size(X, NAMES) - y) ** 2))
    assert per_m2_rmse < 1e-9 < plain_rmse


def test_falls_back_without_size_columns():
    X, y = np.ones((4, 2)), np.arange(4.0)
    splits = (X, y, X, y, X, y)
    out, target = task_a_splits(splits, ["a", "b"])
    assert target == "log(price)" and out is splits
    out, target = task_a_splits(splits, ["area_m2", "land_area_sot"], per_size=False)
    assert target == "log(price)" and out is splits
