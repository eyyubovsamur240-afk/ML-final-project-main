"""
baselines.py — what our from-scratch models are compared against (Problem 5).

  * scikit-learn models built with the SAME hyperparameters as ours, so the
    comparison is matched. Our models are never built from these.
  * two trivial NumPy baselines (predict the train mean / the majority class):
    a model that cannot beat them has learned nothing.
  * StandardScaled: the train-only standardization our Pegasos SVM needs,
    wrapped like a scikit-learn Pipeline so CV can refit it per fold.
  * SizedToFitData: builds a baseline at fit time from the number of fit rows,
    so the SVM baselines stay matched on CV folds and scaling-study subsets.

Matching notes (worth a sentence in the report):
  * Pegasos minimises  lambda/2 ||w||^2 + (1/n) sum hinge.  SGDClassifier with
    loss="hinge" and alpha=lambda minimises the same objective, except that it
    leaves the intercept unregularised. Its "optimal" step 1/(alpha*(t+t0)) decays
    like Pegasos' 1/(lambda*t) but starts smaller (t0 is about 1,000 here), and it
    shuffles once per epoch instead of sampling rows with replacement. It gets the
    same number of updates as Pegasos (epochs_for).
  * LinearSVC minimises 1/2 ||w||^2 + C sum hinge, the same problem when
    C = 1/(lambda*n), with n the number of FIT rows. On the full training split it
    stops at max_iter with a ConvergenceWarning (recorded by run_all), but its test
    accuracy is unchanged at 5x and 15x more iterations (0.8916 vs 0.8915).
  * scikit-learn weights min_impurity_decrease by the node's share of the
    training rows (N_t/N); if our tree uses the unweighted gain, the same number
    prunes differently. Leave it at 0.0 for a like-for-like comparison.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Our criterion names -> scikit-learn's.
_SK_CRITERION = {"mse": "squared_error", "gini": "gini", "entropy": "entropy"}


def _tree_kwargs(params: dict, seed: int) -> dict:
    return {
        "criterion": _SK_CRITERION[params.get("criterion", "gini")],
        "max_depth": params.get("max_depth"),
        "min_samples_split": params.get("min_samples_split", 2),
        "min_samples_leaf": params.get("min_samples_leaf", 1),
        "min_impurity_decrease": params.get("min_impurity_decrease", 0.0),
        "random_state": seed,
    }


def sk_tree_regressor(params: dict, seed: int = 42) -> DecisionTreeRegressor:
    """DecisionTreeRegressor with our tree's hyperparameters (criterion "mse")."""
    return DecisionTreeRegressor(**_tree_kwargs({"criterion": "mse", **params}, seed))


def sk_tree_classifier(params: dict, seed: int = 42) -> DecisionTreeClassifier:
    """DecisionTreeClassifier with our tree's hyperparameters (criterion "gini"/"entropy")."""
    return DecisionTreeClassifier(**_tree_kwargs(params, seed))


def sk_sgd_hinge(lambda_: float, epochs: int, seed: int = 42):
    """Pegasos' twin: scaled features -> SGD on the hinge loss with alpha = lambda."""
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="hinge", penalty="l2", alpha=lambda_, learning_rate="optimal",
                      max_iter=max(1, int(epochs)), tol=None, random_state=seed),
    )


def sk_linear_svc(lambda_: float, n_train: int, seed: int = 42, max_iter: int = 20_000):
    """The exact linear soft-margin SVM (liblinear), C = 1 / (lambda * n_train)."""
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0 / (lambda_ * n_train), loss="hinge", dual=True, max_iter=max_iter,
                  random_state=seed),
    )


def epochs_for(n_iters: int, n_train: int) -> int:
    """Pegasos steps -> passes over the data, to give SGDClassifier the same budget."""
    return max(1, math.ceil(n_iters / max(1, n_train)))


class MeanRegressor:
    """Predicts the training mean for every row (the floor any regressor must beat)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class MajorityClassifier:
    """Predicts the most frequent training class (ties -> the larger label)."""

    def fit(self, X, y):
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        self.classes_ = labels
        self.majority_ = labels[counts == counts.max()][-1]
        self.share_ = float(counts.max() / counts.sum())
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, int(np.flatnonzero(self.classes_ == self.majority_)[0])] = 1.0
        return proba


class SizedToFitData:
    """A scikit-learn baseline built when fit() is called: build(n_fit_rows) -> model.

    LinearSVC's C = 1/(lambda*n) and SGD's epoch budget depend on n, so a model
    built once for the full training split would solve a different problem on a
    CV fold or on a scaling-study subsample.
    """

    def __init__(self, build):
        self.build = build

    def fit(self, X, y):
        self.model = self.build(len(y))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def __getattr__(self, name):
        # decision_function, steps, ... come from the fitted model.
        if name in ("build", "model"):
            raise AttributeError(name)
        return getattr(self.model, name)


class StandardScaled:
    """Standardize with the FIT data's mean/std, then delegate to `model`.

    The scaling statistics come only from the rows passed to fit(), so inside
    cross-validation each fold is scaled with its own training rows.
    Constant columns keep std = 1, as in data_prep.standardize.
    """

    def __init__(self, model):
        self.model = model

    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        self.model.fit(self._scale(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self._scale(X))

    def __getattr__(self, name):
        # decision_function, w, b, history_, ... come from the wrapped model.
        if name in ("model", "mean_", "std_"):
            raise AttributeError(name)
        attr = getattr(self.model, name)
        if callable(attr) and name in ("decision_function", "predict_proba"):
            return lambda X: attr(self._scale(X))
        return attr
