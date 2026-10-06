"""
API contract tests — the interface the rest of the pipeline relies on.

These follow the docstrings in the starter code. Each test SKIPS while the
function it needs is still a TODO stub, and starts running (and must pass)
as soon as it is implemented. If the team deliberately changes an interface,
update the test in the same PR.

These are deliberately generic. The brief also asks for your own tests,
e.g. that your tree matches scikit-learn's splits on a tiny, fully-determined
example (Problem 2c) — put those in their own files (tests/test_tree_vs_sklearn.py, ...).
"""

from __future__ import annotations

import numpy as np
import pytest

from src import data_prep, evaluate
from src.decision_tree import DecisionTree
from src.svm import PegasosSVM


# --- toy data -----------------------------------------------------------------
def toy_classification(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def toy_separable(n=100, seed=0):
    """Two well-separated blobs: any working linear SVM gets ~100% train accuracy."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2.0, 0.5, size=(n // 2, 2)), rng.normal(2.0, 0.5, size=(n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def toy_regression(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.1 * X[:, 1]
    return X, y


def test_pipeline_modules_import():
    from src import run_all

    assert callable(run_all.main)


def test_seed_is_fixed():
    assert isinstance(data_prep.SEED, int)


# --- decision tree ------------------------------------------------------------
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tree_classifier_fit_predict(run_or_skip, criterion):
    X, y = toy_classification()
    tree = DecisionTree(task="classification", criterion=criterion, max_depth=3)
    assert run_or_skip(tree.fit, X, y) is tree, "fit() should return self"
    pred = np.asarray(run_or_skip(tree.predict, X))
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= set(np.unique(y))


def test_tree_classifier_unlimited_depth_memorises_train(run_or_skip):
    X, y = toy_classification()  # continuous features, no duplicate rows
    tree = DecisionTree(task="classification", criterion="gini", max_depth=None)
    run_or_skip(tree.fit, X, y)
    pred = np.asarray(run_or_skip(tree.predict, X))
    assert (pred == y).mean() == 1.0


def test_tree_predict_proba_rows_sum_to_one(run_or_skip):
    X, y = toy_classification()
    tree = DecisionTree(task="classification", max_depth=2)
    run_or_skip(tree.fit, X, y)
    proba = np.asarray(run_or_skip(tree.predict_proba, X))
    assert proba.shape[0] == len(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_tree_regressor_fit_predict(run_or_skip):
    X, y = toy_regression()
    tree = DecisionTree(task="regression", criterion="mse", max_depth=4)
    run_or_skip(tree.fit, X, y)
    pred = np.asarray(run_or_skip(tree.predict, X), dtype=float)
    assert pred.shape == (len(X),)
    assert np.mean((pred - y) ** 2) < np.var(y), "a depth-4 tree should beat predicting the mean"


def test_tree_is_deterministic(run_or_skip):
    X, y = toy_classification()
    a = DecisionTree(task="classification", max_depth=3, random_state=0)
    b = DecisionTree(task="classification", max_depth=3, random_state=0)
    run_or_skip(a.fit, X, y)
    run_or_skip(b.fit, X, y)
    assert np.array_equal(run_or_skip(a.predict, X), run_or_skip(b.predict, X))


# --- Pegasos SVM --------------------------------------------------------------
def test_svm_fit_predict_keeps_original_labels(run_or_skip):
    X, y = toy_separable()
    svm = PegasosSVM(lambda_=1e-2, random_state=0)
    assert run_or_skip(svm.fit, X, y) is svm, "fit() should return self"
    pred = np.asarray(run_or_skip(svm.predict, X))
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= {0, 1}, "predict() should map back to the input labels"
    assert (pred == y).mean() >= 0.95


def test_svm_decision_function_shape_and_sign(run_or_skip):
    X, y = toy_separable()
    svm = PegasosSVM(lambda_=1e-2, random_state=0)
    run_or_skip(svm.fit, X, y)
    scores = np.asarray(run_or_skip(svm.decision_function, X), dtype=float)
    assert scores.shape == (len(X),)
    assert (scores[y == 1] > 0).mean() >= 0.95 and (scores[y == 0] < 0).mean() >= 0.95


def test_svm_is_deterministic_for_a_fixed_seed(run_or_skip):
    X, y = toy_classification()
    a = PegasosSVM(lambda_=1e-2, n_iters=2_000, random_state=7)
    b = PegasosSVM(lambda_=1e-2, n_iters=2_000, random_state=7)
    run_or_skip(a.fit, X, y)
    run_or_skip(b.fit, X, y)
    assert np.allclose(run_or_skip(a.decision_function, X), run_or_skip(b.decision_function, X))


# --- metrics ------------------------------------------------------------------
def test_regression_metrics_on_perfect_and_known_predictions(run_or_skip):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert run_or_skip(evaluate.rmse, y, y) == pytest.approx(0.0)
    assert run_or_skip(evaluate.mae, y, y) == pytest.approx(0.0)
    assert run_or_skip(evaluate.r2, y, y) == pytest.approx(1.0)
    off = y + np.array([1.0, -1.0, 1.0, -1.0])
    assert run_or_skip(evaluate.rmse, y, off) == pytest.approx(1.0)
    assert run_or_skip(evaluate.mae, y, off) == pytest.approx(1.0)


def test_confusion_matrix_layout(run_or_skip):
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 1])
    cm = np.asarray(run_or_skip(evaluate.confusion_matrix, y_true, y_pred))
    assert cm.tolist() == [[1, 2], [1, 3]], "expected [[TN, FP], [FN, TP]]"


def test_precision_recall_f1_known_values(run_or_skip):
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 1])
    p, r, f1 = run_or_skip(evaluate.precision_recall_f1, y_true, y_pred)
    assert p == pytest.approx(3 / 5)
    assert r == pytest.approx(3 / 4)
    assert f1 == pytest.approx(2 * (3 / 5) * (3 / 4) / (3 / 5 + 3 / 4))


def test_roc_auc_extremes_and_ties(run_or_skip):
    y = np.array([0, 0, 1, 1])
    assert run_or_skip(evaluate.roc_auc, y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert run_or_skip(evaluate.roc_auc, y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert run_or_skip(evaluate.roc_auc, y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


# --- data prep (array-level contract; no dataset needed) ---------------------
def test_split_is_deterministic_disjoint_and_complete(run_or_skip):
    n = 200
    X = np.arange(n, dtype=float).reshape(-1, 1)  # row id as the only feature
    y = np.arange(n, dtype=float)
    a = run_or_skip(data_prep.train_val_test_split, X, y, seed=0)
    b = run_or_skip(data_prep.train_val_test_split, X, y, seed=0)
    X_tr, y_tr, X_val, y_val, X_te, y_te = a
    ids = [np.asarray(part).ravel() for part in (X_tr, X_val, X_te)]
    assert sum(len(i) for i in ids) == n
    assert len(np.unique(np.concatenate(ids))) == n, "splits must not overlap"
    assert np.array_equal(np.asarray(y_tr), np.asarray(X_tr).ravel()), "X and y rows misaligned"
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(np.asarray(left), np.asarray(right)), "same seed, same split"


def test_tier_label_uses_given_threshold_and_train_median(run_or_skip):
    y_train = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    tier, thr = run_or_skip(data_prep.make_tier_label, y_train)
    assert thr == pytest.approx(300.0), "threshold defaults to the TRAIN median"
    assert np.asarray(tier).tolist() == [0, 0, 0, 1, 1], "premium = price > threshold"
    tier_te, thr_te = run_or_skip(data_prep.make_tier_label, np.array([250.0, 350.0]), thr)
    assert thr_te == thr and np.asarray(tier_te).tolist() == [0, 1]


def test_standardize_uses_train_statistics_only(run_or_skip):
    rng = np.random.default_rng(0)
    X_tr = rng.normal(5.0, 2.0, size=(50, 3))
    X_te = rng.normal(-1.0, 4.0, size=(20, 3))
    Z_tr, Z_te = run_or_skip(data_prep.standardize, X_tr, X_te)
    assert np.allclose(np.asarray(Z_tr).mean(axis=0), 0.0, atol=1e-8)
    expected = (X_te - X_tr.mean(axis=0)) / X_tr.std(axis=0)
    assert np.allclose(np.asarray(Z_te), expected, atol=1e-6), "test split must use TRAIN mean/std"
