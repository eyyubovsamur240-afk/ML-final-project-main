"""
random_forest.py — a random forest built FROM SCRATCH on top of our own
DecisionTree (bonus Problem 9). No sklearn inside, NumPy only.

Each tree is trained on
  - a bootstrap sample: n rows drawn WITH replacement from the n training rows
  - feature subsampling: at every node the tree only looks at `max_features`
    randomly chosen features (DecisionTree's max_features)
so the trees make different mistakes, and averaging them cuts the variance
of a single deep tree while keeping its low bias.

Prediction:
  - classification: average the trees' predict_proba (soft voting), argmax
  - regression:     average the trees' predictions

Defaults follow sklearn's RandomForest* so the benchmark is matched:
max_features="sqrt" for classification, 1.0 (all features) for regression,
fully grown trees (max_depth=None).
"""

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        n_estimators: int = 100,
        criterion: str | None = None,   # default: "gini" / "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features="default",         # "default" -> "sqrt" (clf) / 1.0 (reg)
        bootstrap: bool = True,
        random_state: int = 42,
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees: list[DecisionTree] = []

    def _tree_params(self):
        criterion = self.criterion or ("mse" if self.task == "regression" else "gini")
        max_features = self.max_features
        if max_features == "default":
            max_features = "sqrt" if self.task == "classification" else None
        return dict(task=self.task, criterion=criterion, max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_features=max_features)

    def fit(self, X, y):
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

        if self.task == "classification":
            # encode labels as 0..K-1 once, so every tree uses the same class
            # numbering even if its bootstrap sample misses a class
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            y = y.astype(float)

        n = len(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        params = self._tree_params()
        self.trees = []
        for _ in range(self.n_estimators):
            rows = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            # each tree gets its own seed for the feature draws, taken from
            # the forest's rng, so the whole forest is reproducible
            seed = int(rng.integers(2**31 - 1))
            tree = DecisionTree(**params, random_state=seed).fit(X[rows], y[rows])
            self.trees.append(tree)

        # sklearn's definition: mean of the (normalised) tree importances
        imp = np.mean([t.feature_importances_ for t in self.trees], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    # --- prediction ----------------------------------------------------------
    def _tree_outputs(self, X):
        """
        One output per tree, stacked:
          classification -> (n_trees, n_rows, n_classes) probabilities
          regression     -> (n_trees, n_rows) predictions
        """
        if not self.trees:
            raise RuntimeError("call fit() before predicting")
        X = np.asarray(X, dtype=float)
        if self.task == "regression":
            return np.stack([t.predict(X) for t in self.trees])
        out = np.zeros((len(self.trees), X.shape[0], self.n_classes_))
        for i, t in enumerate(self.trees):
            # t.classes_ are the encoded labels this tree saw (maybe a subset)
            out[i][:, t.classes_] = t.predict_proba(X)
        return out

    def _to_prediction(self, avg):
        if self.task == "regression":
            return avg
        return self.classes_[np.argmax(avg, axis=1)]

    def predict(self, X):
        return self._to_prediction(self._tree_outputs(X).mean(axis=0))

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        return self._tree_outputs(X).mean(axis=0)

    def staged_predict(self, X):
        """
        Predictions using only the first 1, 2, ..., n_estimators trees.
        Used for the "error vs number of trees" plot. Running sums mean every
        tree is evaluated once, not once per stage.
        """
        outputs = self._tree_outputs(X)
        running = np.cumsum(outputs, axis=0)
        for m in range(len(self.trees)):
            yield self._to_prediction(running[m] / (m + 1))

    def get_depth(self) -> int:
        return max(t.get_depth() for t in self.trees)

    def get_n_leaves(self) -> int:
        return sum(t.get_n_leaves() for t in self.trees)
