"""
Tests for our from-scratch RandomForest (bonus Problem 9).

Run from the repo root:
    python -m pytest tests -q
"""

import numpy as np

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest


def _noisy_classification(seed=1, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.8 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _noisy_regression(seed=2, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = 2 * X[:, 0] + np.sin(2 * X[:, 1]) + X[:, 2] * X[:, 3] + 0.5 * rng.normal(size=n)
    return X, y


def test_one_tree_without_randomness_is_just_a_tree():
    # 1 tree, no bootstrap, all features -> exactly our DecisionTree
    X, y = _noisy_classification()
    forest = RandomForest(n_estimators=1, bootstrap=False, max_features=None).fit(X, y)
    tree = DecisionTree().fit(X, y)
    np.testing.assert_array_equal(forest.predict(X), tree.predict(X))
    np.testing.assert_allclose(forest.predict_proba(X), tree.predict_proba(X))


def test_same_seed_same_forest():
    X, y = _noisy_classification()
    a = RandomForest(n_estimators=10, random_state=3).fit(X, y)
    b = RandomForest(n_estimators=10, random_state=3).fit(X, y)
    c = RandomForest(n_estimators=10, random_state=4).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))
    assert not np.allclose(a.predict_proba(X), c.predict_proba(X))


def test_forest_beats_single_tree_on_held_out_data():
    for make_data, task in ((_noisy_classification, "classification"), (_noisy_regression, "regression")):
        X, y = make_data(n=600)
        X_tr, y_tr, X_te, y_te = X[:400], y[:400], X[400:], y[400:]
        tree = DecisionTree(task=task, criterion="gini" if task == "classification" else "mse").fit(X_tr, y_tr)
        forest = RandomForest(task=task, n_estimators=50, random_state=0).fit(X_tr, y_tr)
        if task == "classification":
            assert np.mean(forest.predict(X_te) == y_te) > np.mean(tree.predict(X_te) == y_te)
        else:
            forest_mse = np.mean((forest.predict(X_te) - y_te) ** 2)
            tree_mse = np.mean((tree.predict(X_te) - y_te) ** 2)
            assert forest_mse < tree_mse


def test_staged_predict_ends_at_predict():
    for make_data, task in ((_noisy_classification, "classification"), (_noisy_regression, "regression")):
        X, y = make_data()
        forest = RandomForest(task=task, n_estimators=7, random_state=0).fit(X, y)
        stages = list(forest.staged_predict(X))
        assert len(stages) == 7
        np.testing.assert_array_equal(stages[-1], forest.predict(X))
        # first stage = the first tree alone
        np.testing.assert_array_equal(stages[0], forest.classes_[forest.trees[0].predict(X)]
                                      if task == "classification" else forest.trees[0].predict(X))


def test_class_missing_from_a_bootstrap_sample():
    # a rare class (2 rows out of 40) is missing from many bootstrap samples;
    # probabilities must still line up with forest.classes_
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.array(["a"] * 19 + ["b"] * 19 + ["rare"] * 2)
    forest = RandomForest(n_estimators=30, random_state=1).fit(X, y)
    assert any(len(t.classes_) < 3 for t in forest.trees)
    proba = forest.predict_proba(X)
    assert proba.shape == (40, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(forest.predict(X)) <= {"a", "b", "rare"}


def test_feature_importances():
    X, y = _noisy_regression()
    forest = RandomForest(task="regression", n_estimators=20, random_state=0).fit(X, y)
    assert np.isclose(forest.feature_importances_.sum(), 1.0)
    # x0 has by far the strongest effect in _noisy_regression
    assert np.argmax(forest.feature_importances_) == 0
