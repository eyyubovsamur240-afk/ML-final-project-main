"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Supports BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT used here. This is our own code.

How it works:
  - impurity of a node:
        gini     G = 1 - sum_k p_k^2
        entropy  H = - sum_k p_k log2 p_k
        mse      I = (1/N) sum_i (y_i - mean(y))^2
  - a split is scored by the impurity decrease (information gain)
        IG = I(parent) - sum_c (N_c / N) I(child_c)
  - best split search: for every feature, sort the node's values and try a
    threshold halfway between every pair of neighbouring distinct values.
    Prefix sums over the sorted labels give the left/right impurity of ALL
    candidate thresholds at once, so there is no Python loop over thresholds.
  - rows with x[feature] <= threshold go left, the rest go right
    (same convention as sklearn, so the trees can be compared node by node).
  - leaves predict the majority class / class frequencies (classification)
    or the mean target (regression).

Stopping rules follow sklearn's definitions so that matched hyperparameters
mean the same thing in the benchmark:
  - max_depth             : never split a node at this depth
  - min_samples_split     : need at least this many rows to try a split
  - min_samples_leaf      : every child must keep at least this many rows
  - min_impurity_decrease : split only if  N_t/N * IG  >= this value
                            (N_t = rows in the node, N = rows in training)
"""

from __future__ import annotations

import numpy as np

# Neighbouring values closer than this are treated as equal when making
# thresholds (sklearn uses the same constant).
FEATURE_THRESHOLD = 1e-7
EPSILON = np.finfo(float).eps

CLASSIFICATION_CRITERIA = ("gini", "entropy")
REGRESSION_CRITERIA = ("mse",)


def _plog2p(p):
    """p * log2(p) elementwise, with 0 * log2(0) = 0."""
    return p * np.log2(np.where(p > 0, p, 1.0))


class Node:
    """One node of the tree."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node: rows with x[feature] <= threshold
        self.right = None        # Node: rows with x[feature] >  threshold
        self.value = None        # class frequencies (clf) or mean (reg)
        self.n_samples = 0       # training rows that reached this node
        self.impurity = 0.0      # impurity of those rows
        self.depth = 0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features=None,              # None | int | float in (0, 1] | "sqrt"
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        # max_features is only needed for the random forest bonus: each node
        # then looks at a random subset of features. None = all features.
        self.max_features = max_features
        self.random_state = random_state
        self.root: Node | None = None

    def _check_params(self):
        if self.task == "classification":
            if self.criterion not in CLASSIFICATION_CRITERIA:
                raise ValueError(f"classification needs criterion in {CLASSIFICATION_CRITERIA}, got {self.criterion!r}")
        elif self.task == "regression":
            if self.criterion not in REGRESSION_CRITERIA:
                raise ValueError(f"regression needs criterion in {REGRESSION_CRITERIA}, got {self.criterion!r}")
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got {self.task!r}")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be >= 0")

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        if self.task == "regression":
            return float(np.mean((y - y.mean()) ** 2))
        p = np.bincount(y, minlength=self.n_classes_) / len(y)
        if self.criterion == "gini":
            return float(1.0 - np.sum(p ** 2))
        p = p[p > 0]   # 0 * log(0) = 0
        return float(-np.sum(p * np.log2(p)))

    def _split_scores(self, y_sorted):
        """
        Weighted child impurity  (N_L * I_L + N_R * I_R) / N  for every split
        position of every feature at once.

        y_sorted is (n, d): column f is the node's target sorted by feature f.
        Row i-1 of the result is the split that sends the first i rows left
        (i = 1..n-1), so the result is (n-1, d). Prefix sums give the left
        side, total minus prefix gives the right side.
        """
        n = y_sorted.shape[0]
        n_left = np.arange(1, n)[:, None]
        n_right = n - n_left

        if self.task == "classification":
            # accumulate sum_k p_k^2 (gini) or -sum_k p_k log2 p_k (entropy),
            # one class at a time so memory stays at (n, d)
            acc_left = np.zeros((n - 1, y_sorted.shape[1]))
            acc_right = np.zeros_like(acc_left)
            for k in range(self.n_classes_):
                cum_k = np.cumsum(y_sorted == k, axis=0)
                left_k = cum_k[:-1]
                right_k = cum_k[-1] - left_k
                p_left, p_right = left_k / n_left, right_k / n_right
                if self.criterion == "gini":
                    acc_left += p_left ** 2
                    acc_right += p_right ** 2
                else:
                    acc_left -= _plog2p(p_left)
                    acc_right -= _plog2p(p_right)
            if self.criterion == "gini":
                imp_left, imp_right = 1.0 - acc_left, 1.0 - acc_right
            else:
                imp_left, imp_right = acc_left, acc_right
        else:
            # variance = E[y^2] - E[y]^2. Centering first keeps the numbers
            # small so the subtraction doesn't lose precision on raw prices.
            # (every column holds the same values, so one mean works for all)
            yc = y_sorted - y_sorted.mean()
            cum = np.cumsum(yc, axis=0)
            cum_sq = np.cumsum(yc ** 2, axis=0)
            sum_l, sq_l = cum[:-1], cum_sq[:-1]
            sum_r, sq_r = cum[-1] - sum_l, cum_sq[-1] - sq_l
            imp_left = sq_l / n_left - (sum_l / n_left) ** 2
            imp_right = sq_r / n_right - (sum_r / n_right) ** 2

        return (n_left * imp_left + n_right * imp_right) / n

    def _candidate_features(self, X):
        n_features = self.n_features_in_
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif isinstance(self.max_features, float):
            k = int(self.max_features * n_features)
        else:
            k = int(self.max_features)
        k = min(max(k, 1), n_features)
        # Draw features in random order and keep the first k that are NOT
        # constant in this node (sklearn does the same). Otherwise small nodes
        # with lots of all-zero one-hot columns would often stop splitting
        # just because the draw only hit constant columns.
        order = self._rng.permutation(n_features)
        varies = np.ptp(X[:, order], axis=0) > FEATURE_THRESHOLD
        return np.sort(order[varies][:k])

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split exists (every feature constant, or every
        threshold would leave a child smaller than min_samples_leaf).

        Ties are broken by taking the first feature (lowest index) and the
        smallest threshold. sklearn breaks feature ties in a random order,
        which is one reason the two trees can differ on real data.
        """
        n = len(y)
        if n < 2:
            return None
        features = self._candidate_features(X)
        if len(features) == 0:
            return None
        Xf = X[:, features]

        # sort every feature column at once; a Python loop over features
        # made fully grown trees ~6x slower
        order = np.argsort(Xf, axis=0, kind="stable")
        x_sorted = np.take_along_axis(Xf, order, axis=0)
        scores = self._split_scores(y[order])           # (n-1, d)

        # row i-1 = "first i rows go left". It is a valid split only where
        # the value actually changes, and both children keep at least
        # min_samples_leaf rows.
        valid = x_sorted[1:] > x_sorted[:-1] + FEATURE_THRESHOLD
        n_left = np.arange(1, n)
        valid &= ((n_left >= self.min_samples_leaf) & (n - n_left >= self.min_samples_leaf))[:, None]
        if not valid.any():
            return None
        scores = np.where(valid, scores, np.inf)

        # argmin over the transposed array walks feature by feature, so on a
        # tie we keep the lowest feature index and then the smallest threshold
        col, row = divmod(int(np.argmin(scores.T)), n - 1)

        lo, hi = x_sorted[row, col], x_sorted[row + 1, col]
        threshold = lo / 2.0 + hi / 2.0
        # halfway point can round up to `hi` when the values are very close,
        # then `hi` would go left by mistake
        if threshold == hi or not np.isfinite(threshold):
            threshold = lo

        decrease = self._impurity(y) - float(scores[row, col])
        return int(features[col]), float(threshold), decrease

    # --- grow / predict -----------------------------------------------------
    def _make_node(self, y, depth: int) -> Node:
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        node.depth = depth
        if self.task == "classification":
            node.value = np.bincount(y, minlength=self.n_classes_) / len(y)
        else:
            node.value = float(y.mean())
        return node

    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node = self._make_node(y, depth)
        n = len(y)

        is_leaf = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or node.impurity <= EPSILON         # node is already pure
        )
        if is_leaf:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, decrease = split

        # min_impurity_decrease is compared with the decrease weighted by the
        # share of training rows in this node (sklearn's definition).
        weighted_decrease = n / self.n_samples_fit_ * decrease
        if weighted_decrease + EPSILON < self.min_impurity_decrease:
            return node

        go_left = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[go_left], y[go_left], depth + 1)
        node.right = self._grow(X[~go_left], y[~go_left], depth + 1)

        # feature importance = total (N_t * decrease) a feature produced
        self._importance_sums[feature] += n * decrease
        return node

    def fit(self, X, y):
        self._check_params()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if len(y) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if np.isnan(X).any():
            raise ValueError("X contains NaN, impute missing values in data_prep first")

        if self.task == "classification":
            # work with class indices 0..K-1 internally, map back in predict
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            y = y.astype(float)

        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = len(y)
        self._rng = np.random.default_rng(self.random_state)
        self._importance_sums = np.zeros(self.n_features_in_)

        self.root = self._grow(X, y, depth=0)

        total = self._importance_sums.sum()
        if total > 0:
            self.feature_importances_ = self._importance_sums / total
        else:
            self.feature_importances_ = self._importance_sums.copy()  # single leaf tree
        return self

    def _check_fitted(self, X):
        if self.root is None:
            raise RuntimeError("call fit() before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected X with {self.n_features_in_} columns, got shape {X.shape}")
        return X

    def _predict_one(self, x, node: Node):
        """Follow one row down to its leaf and return that leaf."""
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def _leaf_values(self, X):
        """
        Leaf value for every row. Going row by row with _predict_one was slow
        on the full dataset, so this sends whole index arrays down the tree.
        """
        n = X.shape[0]
        out = np.empty((n, self.n_classes_)) if self.task == "classification" else np.empty(n)
        stack = [(self.root, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node.is_leaf():
                out[idx] = node.value
                continue
            go_left = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out

    def predict(self, X):
        X = self._check_fitted(X)
        values = self._leaf_values(X)
        if self.task == "regression":
            return values
        return self.classes_[np.argmax(values, axis=1)]

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        X = self._check_fitted(X)
        return self._leaf_values(X)

    # --- inspection (report / demo / sklearn comparison) ---------------------
    def nodes(self):
        """All nodes in pre-order (node, left subtree, right subtree).
        This is the same order sklearn numbers its tree_ arrays."""
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf():
                stack.append(node.right)
                stack.append(node.left)
        return out

    def apply(self, X):
        """Index of the leaf (position in nodes()) that each row ends up in,
        like sklearn's apply()."""
        X = self._check_fitted(X)
        index = {id(node): i for i, node in enumerate(self.nodes())}
        return np.array([index[id(self._predict_one(x, self.root))] for x in X])

    def get_depth(self) -> int:
        return max(node.depth for node in self.nodes())

    def get_n_leaves(self) -> int:
        return sum(node.is_leaf() for node in self.nodes())

    def export_text(self, feature_names=None, decimals: int = 3) -> str:
        """Readable dump of the tree, similar to sklearn.tree.export_text."""
        def name(f):
            return feature_names[f] if feature_names is not None else f"x[{f}]"

        def leaf_text(node):
            if self.task == "regression":
                return f"value: {node.value:.{decimals}f}"
            return f"class: {self.classes_[np.argmax(node.value)]}"

        lines = []

        def walk(node, indent):
            if node.is_leaf():
                lines.append(f"{indent}|--- {leaf_text(node)}  (n={node.n_samples})")
                return
            thr = f"{node.threshold:.{decimals}f}"
            lines.append(f"{indent}|--- {name(node.feature)} <= {thr}")
            walk(node.left, indent + "|   ")
            lines.append(f"{indent}|--- {name(node.feature)} >  {thr}")
            walk(node.right, indent + "|   ")

        walk(self.root, "")
        return "\n".join(lines)
