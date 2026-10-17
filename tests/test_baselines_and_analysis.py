"""Baselines are matched to our hyperparameters, and the error-analysis helpers add up."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import baselines, error_analysis, evaluate


# --- baselines ----------------------------------------------------------------
def test_sklearn_trees_get_our_hyperparameters():
    params = {"criterion": "entropy", "max_depth": 4, "min_samples_split": 6,
              "min_samples_leaf": 3, "min_impurity_decrease": 0.0}
    clf = baselines.sk_tree_classifier(params, seed=7)
    assert (clf.criterion, clf.max_depth, clf.min_samples_split, clf.min_samples_leaf,
            clf.random_state) == ("entropy", 4, 6, 3, 7)
    reg = baselines.sk_tree_regressor({"max_depth": 5}, seed=1)
    assert reg.criterion == "squared_error" and reg.max_depth == 5


def test_svm_baselines_use_the_pegasos_objective():
    sgd = baselines.sk_sgd_hinge(lambda_=1e-3, epochs=4, seed=0)[-1]
    assert (sgd.loss, sgd.alpha, sgd.learning_rate, sgd.max_iter) == ("hinge", 1e-3, "optimal", 4)
    svc = baselines.sk_linear_svc(lambda_=1e-3, n_train=500)[-1]
    assert svc.C == pytest.approx(1 / (1e-3 * 500))
    assert baselines.epochs_for(n_iters=1_001, n_train=500) == 3


def test_sized_to_fit_data_matches_c_to_the_fit_rows():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 3)), np.repeat([0, 1], 100)
    model = baselines.SizedToFitData(lambda n: baselines.sk_linear_svc(1e-2, n))
    assert not hasattr(model, "decision_function"), "nothing to forward before fit"
    model.fit(X[:50], np.tile([0, 1], 25))
    assert model.model[-1].C == pytest.approx(1 / (1e-2 * 50))
    model.fit(X, y)
    assert model.model[-1].C == pytest.approx(1 / (1e-2 * 200))
    assert model.decision_function(X).shape == (200,) and hasattr(model, "steps")


def test_trivial_baselines():
    X = np.zeros((4, 2))
    assert baselines.MeanRegressor().fit(X, [1.0, 2.0, 3.0, 6.0]).predict(X).tolist() == [3.0] * 4
    clf = baselines.MajorityClassifier().fit(X, [0, 1, 1, 1])
    assert clf.predict(X).tolist() == [1] * 4
    assert clf.predict_proba(X)[:, 1].tolist() == [1.0] * 4


class _Recorder:
    """Stands in for our SVM: remembers what it was trained on."""

    def __init__(self):
        self.w = np.array([1.0, -1.0])

    def fit(self, X, y):
        self.seen = X
        return self

    def predict(self, X):
        return (X[:, 0] > 0).astype(int)

    def decision_function(self, X):
        return X[:, 0]


def test_standard_scaled_uses_fit_statistics_only():
    rng = np.random.default_rng(0)
    X_fit = rng.normal(10.0, 3.0, size=(50, 2))
    X_fit[:, 1] = 4.0  # a constant column must not divide by zero
    model = baselines.StandardScaled(_Recorder()).fit(X_fit, np.zeros(50))
    assert np.allclose(model.model.seen[:, 0].mean(), 0.0) and np.allclose(model.model.seen[:, 0].std(), 1.0)
    assert np.all(model.model.seen[:, 1] == 0.0)
    X_new = np.array([[10.0, 4.0], [X_fit[:, 0].mean(), 4.0]])
    assert model.decision_function(X_new)[1] == pytest.approx(0.0)
    assert model.w.tolist() == [1.0, -1.0], "attributes of the wrapped model stay reachable"
    assert not hasattr(baselines.StandardScaled(object()), "decision_function")


# --- error analysis -------------------------------------------------------------
def test_price_band_and_district_coverage():
    bands = error_analysis.price_band([50_000, 100_000, 250_000, 2_000_000])
    assert list(bands.astype(str)) == ["0k-100k", "100k-200k", "200k-300k", ">= 1,000k"]
    cov = error_analysis.district_coverage(["a", "b", "z"], ["a"] * 40 + ["b"] * 5)
    assert list(cov.astype(str)) == ["30-99", "< 30 (sparse)", "< 30 (sparse)"]


def test_regression_breakdown_matches_overall_metrics():
    rng = np.random.default_rng(1)
    log_true = rng.normal(12, 0.5, 200)
    log_pred = log_true + rng.normal(0.1, 0.2, 200)
    one_group = error_analysis.regression_breakdown(np.array(["all"] * 200), log_true, log_pred)
    assert one_group.loc[0, "n"] == 200
    assert one_group.loc[0, "rmse_log"] == pytest.approx(evaluate.rmse(log_true, log_pred))
    assert one_group.loc[0, "bias_log"] == pytest.approx(np.mean(log_pred - log_true))
    two = error_analysis.regression_breakdown(np.where(np.arange(200) < 50, "x", "y"),
                                              log_true, log_pred)
    assert two["n"].tolist() == [50, 150]


def test_classification_breakdown_rates():
    groups = np.array(["g"] * 6)
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 0, 1])
    row = error_analysis.classification_breakdown(groups, y_true, y_pred).iloc[0]
    assert row["error_rate"] == pytest.approx(3 / 6)
    assert row["missed_premium_rate"] == pytest.approx(2 / 3)
    assert row["false_premium_rate"] == pytest.approx(1 / 3)


def test_threshold_margin_bins():
    labels = error_analysis.threshold_margin([100.0, 104.0, 150.0, 400.0], threshold=100.0)
    assert list(labels.astype(str)) == ["0-0.05", "0-0.05", "0.25-0.5", ">= 0.5"]


def test_feature_groups_keep_one_hot_columns_together():
    names = ["area_m2", "rooms", "category_Ofis", "category_Yeni tikili", "location_Səbail r.",
             "repair_yes"]
    groups = error_analysis.feature_groups(names)
    assert groups == {"area_m2": [0], "rooms": [1], "category": [2, 3], "location": [4],
                      "repair": [5]}


def test_permutation_importance_finds_the_used_feature():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 3))
    y = 3 * X[:, 1]
    imp = error_analysis.permutation_importance(
        lambda Xs, ys: -evaluate.rmse(ys, 3 * Xs[:, 1]), X, y,
        {"a": [0], "b": [1], "c": [2]}, n_repeats=2, seed=0)
    assert imp.iloc[0]["feature"] == "b" and imp.iloc[0]["importance"] > 1.0
    assert imp.set_index("feature").loc[["a", "c"], "importance"].abs().max() == 0.0


def test_grouped_importances_and_top_weights():
    groups = {"x": [0], "cat": [1, 2]}
    table = error_analysis.grouped_importances([0.2, 0.5, 0.3], groups)
    assert table.iloc[0].to_dict() == {"feature": "cat", "importance": 0.8}
    w = error_analysis.top_weights([0.1, -2.0, 0.5], ["a", "b", "c"], k=2)
    assert w["feature"].tolist() == ["b", "c"]


def test_worst_predictions_lists_the_biggest_misses():
    df = pd.DataFrame({"category": ["x", "y", "z"], "location": ["l"] * 3, "area_m2": [50, 60, 70],
                       "rooms": [1, 2, 3]})
    log_true = np.log([100_000.0, 100_000.0, 100_000.0])
    log_pred = np.log([110_000.0, 300_000.0, 90_000.0])
    worst = error_analysis.worst_predictions(df, log_true, log_pred, n=2)
    assert worst["category"].tolist() == ["y", "z"]
    assert worst.loc[0, "error_pct"] == pytest.approx(200.0)
