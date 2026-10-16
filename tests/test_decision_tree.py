"""
Tests for our from-scratch DecisionTree.

The main one (Problem 2c): on a tiny, fully-determined example our tree must
make exactly the same splits as sklearn. "Fully determined" means there are
no ties between candidate splits, so sklearn builds the same tree for every
random_state -- the test checks that first, otherwise a mismatch could just
be a different tie-break.

Run from the repo root:
    python -m pytest tests -q
or without pytest:
    python -m tests.test_decision_tree
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree

# 10 rows, 2 features, chosen so that no two candidate splits tie
X_TINY = np.array([
    [1, 7.5], [2, 3.0], [3, 8.0], [4, 1.0], [5, 6.0],
    [6, 2.5], [7, 5.0], [8, 4.0], [9, 9.0], [10, 0.5],
])
Y_TINY_CLF = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
Y_TINY_REG = np.array([53, 47, 51, 36, 50, 26, 32, 49, 16, 25], dtype=float)


def our_structure(tree):
    """(feature, threshold, n_samples) per node in pre-order, -2 for leaves
    (the same encoding sklearn uses in tree_.feature / tree_.threshold)."""
    return [
        (-2, -2.0, n.n_samples) if n.is_leaf() else (n.feature, n.threshold, n.n_samples)
        for n in tree.nodes()
    ]


def sklearn_structure(model):
    t = model.tree_
    return [(int(t.feature[i]), float(t.threshold[i]), int(t.n_node_samples[i])) for i in range(t.node_count)]


def fit_sklearn_checked(model_class, X, y, n_seeds=20, **params):
    """Fit sklearn with several seeds and make sure they all agree (= no ties)."""
    models = [model_class(random_state=seed, **params).fit(X, y) for seed in range(n_seeds)]
    structures = {tuple(sklearn_structure(m)) for m in models}
    assert len(structures) == 1, "example is not fully determined (sklearn tie-breaks differ)"
    return models[0]


# --- tiny examples vs sklearn ------------------------------------------------
def test_tiny_classification_matches_sklearn():
    for criterion in ("gini", "entropy"):
        sk = fit_sklearn_checked(DecisionTreeClassifier, X_TINY, Y_TINY_CLF, criterion=criterion)
        ours = DecisionTree(task="classification", criterion=criterion).fit(X_TINY, Y_TINY_CLF)

        assert our_structure(ours) == sklearn_structure(sk), criterion
        np.testing.assert_array_equal(ours.predict(X_TINY), sk.predict(X_TINY))
        np.testing.assert_allclose(ours.predict_proba(X_TINY), sk.predict_proba(X_TINY))
        np.testing.assert_allclose(ours.feature_importances_, sk.feature_importances_)


def test_tiny_regression_matches_sklearn():
    sk = fit_sklearn_checked(DecisionTreeRegressor, X_TINY, Y_TINY_REG, min_samples_leaf=2)
    ours = DecisionTree(task="regression", criterion="mse", min_samples_leaf=2).fit(X_TINY, Y_TINY_REG)

    assert our_structure(ours) == sklearn_structure(sk)
    np.testing.assert_allclose(ours.predict(X_TINY), sk.predict(X_TINY))
    np.testing.assert_allclose(ours.feature_importances_, sk.feature_importances_)


def test_hand_computed_root_split():
    # Root of the tiny classification tree, by hand (gini):
    #   parent: 5 vs 5 -> G = 0.5
    #   x1 <= 5.5: left = 6 rows (1 zero, 5 ones), right = 4 rows (all zeros)
    #   G_left = 1 - (1/6)^2 - (5/6)^2 = 10/36, G_right = 0
    #   IG = 0.5 - 6/10 * 10/36 = 1/3
    tree = DecisionTree(criterion="gini")
    tree.n_classes_ = 2
    tree.n_features_in_ = 2
    feature, threshold, decrease = tree._best_split(X_TINY, Y_TINY_CLF)
    assert (feature, threshold) == (1, 5.5)
    assert np.isclose(decrease, 1 / 3)


# --- random data: same leaves as sklearn whenever sklearn has no ties --------
def leaf_groups(leaf_ids):
    """Training rows grouped by the leaf they land in, as a set of sets."""
    groups = {}
    for row, leaf in enumerate(leaf_ids):
        groups.setdefault(leaf, set()).add(row)
    return {frozenset(g) for g in groups.values()}


def test_random_data_matches_sklearn_without_ties():
    # On random data two different features sometimes cut off exactly the
    # same rows (e.g. one point that is the min of x3 and the max of x1).
    # Then the gain is identical, and which feature gets picked is just
    # tie-breaking / float rounding. So here we compare WHICH training rows
    # end up together in a leaf, not the feature numbers themselves.
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(12):
        n, d = 200, 4
        # float32-representable values, because sklearn casts X to float32
        X = rng.normal(size=(n, d)).astype(np.float32).astype(float)
        params = dict(max_depth=[None, 3, 5][trial % 3],
                      min_samples_split=[2, 10][trial % 2],
                      min_samples_leaf=[1, 5, 3][trial % 3],
                      min_impurity_decrease=[0.0, 0.005][trial % 2])
        if trial % 2 == 0:
            y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
            sk_class = DecisionTreeClassifier
            ours = DecisionTree("classification", "gini", **params).fit(X, y)
        else:
            y = 3 * X[:, 0] + np.sin(X[:, 1]) + rng.normal(size=n)
            sk_class = DecisionTreeRegressor
            ours = DecisionTree("regression", "mse", **params).fit(X, y)

        sk_models = [sk_class(random_state=s, **params).fit(X, y) for s in range(5)]
        sk_groups = [leaf_groups(m.apply(X)) for m in sk_models]
        if any(g != sk_groups[0] for g in sk_groups):
            continue    # sklearn itself is not unique here (ties), skip
        ours_groups = leaf_groups(ours.apply(X))
        assert ours_groups == sk_groups[0], (trial, params)
        np.testing.assert_allclose(ours.predict(X), sk_models[0].predict(X))
        checked += 1
    assert checked >= 4, "too few tie-free cases were checked"


# --- hyperparameters must actually change the tree ---------------------------
def _noisy_data(seed=1, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.8 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_max_depth():
    X, y = _noisy_data()
    full = DecisionTree(max_depth=None).fit(X, y)
    for depth in (0, 1, 3):
        tree = DecisionTree(max_depth=depth).fit(X, y)
        assert tree.get_depth() <= depth
        assert tree.get_n_leaves() < full.get_n_leaves()


def test_min_samples_leaf():
    X, y = _noisy_data()
    full = DecisionTree().fit(X, y)
    tree = DecisionTree(min_samples_leaf=10).fit(X, y)
    assert all(n.n_samples >= 10 for n in tree.nodes() if n.is_leaf())
    assert tree.get_n_leaves() < full.get_n_leaves()


def test_min_samples_split():
    X, y = _noisy_data()
    full = DecisionTree().fit(X, y)
    tree = DecisionTree(min_samples_split=30).fit(X, y)
    assert all(n.n_samples >= 30 for n in tree.nodes() if not n.is_leaf())
    assert tree.get_n_leaves() < full.get_n_leaves()


def test_min_impurity_decrease():
    X, y = _noisy_data()
    full = DecisionTree().fit(X, y)
    tree = DecisionTree(min_impurity_decrease=0.01).fit(X, y)
    assert tree.get_n_leaves() < full.get_n_leaves()
    # a huge value means the root can't be split at all
    assert DecisionTree(min_impurity_decrease=10.0).fit(X, y).get_n_leaves() == 1


def test_criterion_changes_tree():
    X, y = _noisy_data()
    gini = DecisionTree(criterion="gini").fit(X, y)
    entropy = DecisionTree(criterion="entropy").fit(X, y)
    assert our_structure(gini) != our_structure(entropy)


# --- API / edge cases --------------------------------------------------------
def test_string_labels_and_proba():
    y = np.where(Y_TINY_CLF == 1, "premium", "standard")
    tree = DecisionTree(max_depth=1).fit(X_TINY, y)
    proba = tree.predict_proba(X_TINY)
    assert list(tree.classes_) == ["premium", "standard"]
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(tree.predict(X_TINY), tree.classes_[proba.argmax(axis=1)])


def test_pure_and_constant_data_give_single_leaf():
    X = np.ones((5, 2))
    assert DecisionTree().fit(X, [0, 1, 0, 1, 0]).get_n_leaves() == 1      # nothing to split on
    assert DecisionTree().fit(X_TINY, np.zeros(10)).get_n_leaves() == 1     # already pure
    reg = DecisionTree(task="regression", criterion="mse").fit(X, [1., 2., 3., 4., 5.])
    np.testing.assert_allclose(reg.predict(X), 3.0)


def test_predict_one_agrees_with_predict():
    X, y = _noisy_data()
    tree = DecisionTree(max_depth=6).fit(X, y)
    one_by_one = [tree.classes_[np.argmax(tree._predict_one(x, tree.root).value)] for x in X]
    np.testing.assert_array_equal(tree.predict(X), one_by_one)


def test_max_features_is_seeded():
    # used later by the random forest bonus: same seed -> same tree
    X, y = _noisy_data()
    a = DecisionTree(max_features="sqrt", random_state=0).fit(X, y)
    b = DecisionTree(max_features="sqrt", random_state=0).fit(X, y)
    assert our_structure(a) == our_structure(b)


def test_bad_inputs_raise():
    import pytest
    with pytest.raises(ValueError):
        DecisionTree(task="classification", criterion="mse").fit(X_TINY, Y_TINY_CLF)
    with pytest.raises(ValueError):
        DecisionTree(task="regression", criterion="gini").fit(X_TINY, Y_TINY_REG)
    with pytest.raises(ValueError):
        DecisionTree().fit(np.array([[np.nan, 1.0], [0.0, 1.0]]), [0, 1])
    with pytest.raises(AttributeError):
        DecisionTree(task="regression", criterion="mse").fit(X_TINY, Y_TINY_REG).predict_proba(X_TINY)
    with pytest.raises(RuntimeError):
        DecisionTree().predict(X_TINY)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
