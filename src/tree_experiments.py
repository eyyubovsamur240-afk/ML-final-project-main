"""
tree_experiments.py — tuning + sklearn benchmark for OUR decision tree.

  Problem 4a: Gini vs entropy (Task B), max_depth sweep, min_samples_leaf
              sweep, learning curves -> bias/variance story.
  Problem 5a: our tree vs DecisionTreeRegressor / DecisionTreeClassifier
              with matched hyperparameters (scores, structure, speed).
  Problem 9 (bonus): random forest from our own trees, error vs number of
              trees, vs sklearn's RandomForest*.

Everything takes ready-made arrays, so run_all.py only has to pass the
splits from data_prep. Hyperparameters are picked on the VALIDATION split
only; the test split is touched once, in the final comparison.

Usage from run_all.py:
    from . import tree_experiments
    tree_results = tree_experiments.run(task_a_splits, task_b_splits, feature_names)
    forest_results = tree_experiments.run_forest(task_a_splits, task_b_splits,
                                                 feature_names, tree_results)
where task_x_splits = (X_tr, y_tr, X_val, y_val, X_te, y_te).
For Task A pass log(price) as y (heavy tailed target); RMSE is then in log units.
By default the Task A models are trained on log(price per m2 / sot) and the
area is added back (per_size=True, see per_size_target); that does not change
what the RMSE means.

Quick check without the real data (two small datasets that ship with sklearn):
    python -m src.tree_experiments
"""

from __future__ import annotations

import os
import time

import numpy as np
# Figure objects directly (no pyplot): nothing opens a window and no backend
# has to be picked, so this also runs on CI / without a display
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .decision_tree import DecisionTree
from .random_forest import RandomForest

SEED = 42
FIG_DIR = "report/figures"

DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
MIN_LEAF = [1, 2, 5, 10, 20, 50, 100]


# --- metrics -----------------------------------------------------------------
# TODO: switch to evaluate.rmse / evaluate.precision_recall_f1 once
# evaluate.py is done, so every model is scored by the same code.
def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _f1(y_true, y_pred, positive=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    return float(2 * tp / (2 * tp + fp + fn)) if tp > 0 else 0.0


def score(task, y_true, y_pred):
    """RMSE for regression (lower = better), F1 of class 1 for classification."""
    return _rmse(y_true, y_pred) if task == "regression" else _f1(y_true, y_pred)


def metric_name(task):
    return "RMSE" if task == "regression" else "F1"


def is_better(task, a, b):
    """True if score a beats score b."""
    return a < b if task == "regression" else a > b


def default_params(task):
    return dict(task=task, criterion="mse" if task == "regression" else "gini",
                random_state=SEED)


# --- sweeps ------------------------------------------------------------------
def validation_curve(param, values, X_tr, y_tr, X_val, y_val, base_params):
    """Fit one tree per value of `param`; train + validation score for each."""
    task = base_params["task"]
    out = {"param": param, "values": list(values), "train": [], "val": [],
           "fit_time": [], "n_leaves": []}
    for v in values:
        t0 = time.perf_counter()
        tree = DecisionTree(**{**base_params, param: v}).fit(X_tr, y_tr)
        out["fit_time"].append(time.perf_counter() - t0)
        out["train"].append(score(task, y_tr, tree.predict(X_tr)))
        out["val"].append(score(task, y_val, tree.predict(X_val)))
        out["n_leaves"].append(tree.get_n_leaves())
    return out


def best_value(curve, task):
    """Value with the best validation score. Values are listed from simple to
    complex, so on a tie the simpler tree wins."""
    best = 0
    for i, s in enumerate(curve["val"]):
        if is_better(task, s, curve["val"][best]):
            best = i
    return curve["values"][best]


def learning_curve(params, X_tr, y_tr, X_val, y_val,
                   fractions=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0), seed=SEED):
    """Train on growing random subsets of the training split (same seed every time)."""
    task = params["task"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_tr))
    out = {"sizes": [], "train": [], "val": []}
    for frac in fractions:
        idx = perm[: max(int(frac * len(y_tr)), 2)]
        tree = DecisionTree(**params).fit(X_tr[idx], y_tr[idx])
        out["sizes"].append(len(idx))
        out["train"].append(score(task, y_tr[idx], tree.predict(X_tr[idx])))
        out["val"].append(score(task, y_val, tree.predict(X_val)))
    return out


# --- sklearn comparison (Problem 5a) -----------------------------------------
def sklearn_twin(params):
    """sklearn tree with the same hyperparameters as our DecisionTree params."""
    shared = dict(max_depth=params.get("max_depth"),
                  min_samples_split=params.get("min_samples_split", 2),
                  min_samples_leaf=params.get("min_samples_leaf", 1),
                  min_impurity_decrease=params.get("min_impurity_decrease", 0.0),
                  random_state=params.get("random_state", SEED))
    if params["task"] == "regression":
        return DecisionTreeRegressor(criterion="squared_error", **shared)
    return DecisionTreeClassifier(criterion=params.get("criterion", "gini"), **shared)


def _structure(tree):
    return [(-2, -2.0) if n.is_leaf() else (n.feature, n.threshold) for n in tree.nodes()]


def _sk_structure(model):
    t = model.tree_
    return [(int(t.feature[i]), float(t.threshold[i])) for i in range(t.node_count)]


def leaf_groups(leaf_ids):
    """Rows grouped by the leaf they land in, as a set of sets. Two trees
    with the same groups split the data the same way, even if they used
    different (equally good) features or thresholds to get there."""
    groups = {}
    for row, leaf in enumerate(leaf_ids):
        groups.setdefault(leaf, set()).add(row)
    return {frozenset(g) for g in groups.values()}


def _same_tree(a, b):
    return len(a) == len(b) and all(u[0] == v[0] and np.isclose(u[1], v[1])
                                    for u, v in zip(a, b, strict=True))


def compare_with_sklearn(params, X_tr, y_tr, X_eval, y_eval, n_repeats=3, seeds_to_try=30):
    """
    Fit our tree and sklearn's with matched hyperparameters on the same data.
    Returns scores, fit/predict times (best of n_repeats), tree size, how
    often the predictions agree, the first node where the trees differ, and
    whether they still put the same training rows together in the leaves.
    If the trees differ, sklearn is refit with other random_state values: if
    one of them builds exactly our tree, the difference is only tie-breaking
    (sklearn visits features in a random order, we go by feature index).
    """
    task = params["task"]
    res = {}
    for name, make in (("ours", lambda: DecisionTree(**params)), ("sklearn", lambda: sklearn_twin(params))):
        fit_times, pred_times = [], []
        for _ in range(n_repeats):
            model = make()
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            pred = model.predict(X_eval)
            pred_times.append(time.perf_counter() - t0)
        res[name] = dict(model=model, pred=pred, score=score(task, y_eval, pred),
                         fit_time=min(fit_times), predict_time=min(pred_times),
                         depth=model.get_depth(), n_leaves=model.get_n_leaves())

    ours, sk = res["ours"], res["sklearn"]
    if task == "regression":
        res["max_abs_pred_diff"] = float(np.max(np.abs(ours["pred"] - sk["pred"])))
        res["pred_agreement"] = float(np.mean(np.isclose(ours["pred"], sk["pred"])))
    else:
        res["pred_agreement"] = float(np.mean(ours["pred"] == sk["pred"]))

    a, b = _structure(ours["model"]), _sk_structure(sk["model"])
    # first pre-order node where the trees disagree (None = identical trees)
    res["first_diff_node"] = next((i for i, (u, v) in enumerate(zip(a, b, strict=False))
                                   if u[0] != v[0] or not np.isclose(u[1], v[1])),
                                  None if len(a) == len(b) else min(len(a), len(b)))
    res["same_structure"] = res["first_diff_node"] is None
    res["same_leaf_groups"] = (leaf_groups(ours["model"].apply(X_tr))
                               == leaf_groups(sk["model"].apply(X_tr)))
    res["sklearn_seed_matching_ours"] = None
    if not res["same_structure"]:
        for seed in range(seeds_to_try):
            twin = sklearn_twin({**params, "random_state": seed}).fit(X_tr, y_tr)
            if _same_tree(a, _sk_structure(twin)):
                res["sklearn_seed_matching_ours"] = seed
                break
    return res


def print_comparison(title, res, task):
    m = metric_name(task)
    print(f"\n{title}")
    print(f"  {'':10s} {m:>8s} {'fit [s]':>9s} {'predict [s]':>12s} {'depth':>6s} {'leaves':>7s}")
    for name in ("ours", "sklearn"):
        r = res[name]
        print(f"  {name:10s} {r['score']:8.4f} {r['fit_time']:9.3f} {r['predict_time']:12.4f} "
              f"{r['depth']:6d} {r['n_leaves']:7d}")
    print(f"  prediction agreement: {100 * res['pred_agreement']:.2f}%"
          + (f", max |diff| = {res['max_abs_pred_diff']:.4g}" if "max_abs_pred_diff" in res else ""))
    seed = res["sklearn_seed_matching_ours"]
    if res["same_structure"]:
        print("  trees are identical node by node")
    elif seed is not None:
        print(f"  trees first differ at pre-order node {res['first_diff_node']}, but sklearn with "
              f"random_state={seed} builds exactly our tree -> only tie-breaking differs")
    elif res["same_leaf_groups"]:
        print(f"  trees first differ at pre-order node {res['first_diff_node']}, but every leaf "
              "holds the same training rows -> equally good splits, broken differently (ties)")
    else:
        print(f"  trees first differ at pre-order node {res['first_diff_node']} and the leaves differ "
              "(ties between splits early on, or sklearn's float32 copy of X merging close values)")


# --- plots -------------------------------------------------------------------
def _save(fig, name, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return path


def plot_validation_curves(curves, labels, task, title, name, fig_dir=FIG_DIR):
    """curves: list of validation_curve() outputs over the same values."""
    fig = Figure(figsize=(5, 3.4))
    ax = fig.subplots()
    values = curves[0]["values"]
    x = np.arange(len(values))
    colors = ("tab:blue", "tab:orange", "tab:green")[: len(curves)]
    for curve, label, color in zip(curves, labels, colors, strict=True):
        ax.plot(x, curve["train"], "--", color=color, label=f"{label} train")
        ax.plot(x, curve["val"], "-o", ms=3, color=color, label=f"{label} validation")
    ax.set_xticks(x)
    ax.set_xticklabels(["None" if v is None else str(v) for v in values])
    ax.set_xlabel(curves[0]["param"])
    ax.set_ylabel(metric_name(task))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, name, fig_dir)


def plot_learning_curve(curve, task, title, name, fig_dir=FIG_DIR):
    fig = Figure(figsize=(5, 3.4))
    ax = fig.subplots()
    ax.plot(curve["sizes"], curve["train"], "--o", ms=3, label="train")
    ax.plot(curve["sizes"], curve["val"], "-o", ms=3, label="validation")
    ax.set_xlabel("training rows")
    ax.set_ylabel(metric_name(task))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, name, fig_dir)


def plot_feature_importances(tree, feature_names, title, name, top_k=15, fig_dir=FIG_DIR):
    imp = tree.feature_importances_
    top = np.argsort(imp)[::-1][:top_k]
    top = top[imp[top] > 0][::-1]       # skip unused features, biggest at the top
    fig = Figure(figsize=(6.5, 0.25 * len(top) + 1.2))
    ax = fig.subplots()
    ax.barh([feature_names[i] for i in top], imp[top])
    ax.set_xlabel("impurity-decrease importance")
    # figure-level title: long feature names (Azerbaijani districts) push the
    # axes to the right, and an axes title then runs off the image
    fig.suptitle(title, fontsize=11)
    return _save(fig, name, fig_dir)


# --- Task A target: price per m2 / per sot -----------------------------------
def log_size(X, feature_names):
    """
    log of each listing's size: floor area (m2) for buildings, land area (sot)
    for land plots (their area_m2 is 0). Rows without any size get 0, so they
    just keep log(price) as their target.
    """
    area = X[:, feature_names.index("area_m2")]
    land = X[:, feature_names.index("land_area_sot")]
    size = np.where(area > 0, area, land)
    return np.log(np.where(size > 0, size, 1.0))


def per_size_target(task_a, feature_names):
    """
    Task A splits with y = log(price) - log(size), i.e. log(price per m2 / sot).

    A tree can only model the effect of area in steps (one leaf per range of
    areas). With this target it predicts the price per m2 and the area goes
    back in exactly: log(price) = prediction + log(size).

    log(size) is a known feature of the listing, so y_hat - y is the same on
    both scales: RMSE here IS the RMSE of log(price), comparable with the plain
    target. Price per m2 is only the training target, never a feature (as a
    feature it would be the unit_price leakage). We chose this target on the
    validation split, where it beat plain log(price) with the same tree.
    """
    X_tr, y_tr, X_va, y_va, X_te, y_te = task_a
    return (X_tr, y_tr - log_size(X_tr, feature_names),
            X_va, y_va - log_size(X_va, feature_names),
            X_te, y_te - log_size(X_te, feature_names))


def task_a_splits(task_a, feature_names, per_size=True):
    """Task A splits the way the models see them (see per_size_target)."""
    if not per_size:
        return task_a, "log(price)"
    if "area_m2" not in feature_names or "land_area_sot" not in feature_names:
        print("  (no area_m2 / land_area_sot columns, Task A target stays log(price))")
        return task_a, "log(price)"
    return per_size_target(task_a, feature_names), "log(price per m2 / sot)"


# --- the whole tree study ----------------------------------------------------
def tune(task, X_tr, y_tr, X_val, y_val, criterion=None):
    """
    Pick max_depth, then min_samples_leaf (with that depth), on validation.
    Returns (best_params, depth_curve, leaf_curve).
    """
    params = default_params(task)
    if criterion is not None:
        params["criterion"] = criterion
    depth_curve = validation_curve("max_depth", DEPTHS, X_tr, y_tr, X_val, y_val, params)
    params["max_depth"] = best_value(depth_curve, task)
    leaf_curve = validation_curve("min_samples_leaf", MIN_LEAF, X_tr, y_tr, X_val, y_val, params)
    params["min_samples_leaf"] = best_value(leaf_curve, task)
    return params, depth_curve, leaf_curve


def run(task_a, task_b, feature_names, fig_dir=FIG_DIR, per_size=True):
    """
    Full decision-tree study for both tasks. task_a / task_b are
    (X_tr, y_tr, X_val, y_val, X_te, y_te), with y = log(price) for Task A.
    per_size=True trains Task A on log(price per m2 / sot), see
    per_size_target (RMSE is still the RMSE of log(price)).
    Saves figures to fig_dir and returns the final params, test scores and
    sklearn comparisons.
    """
    results = {}

    # Task B: Gini vs entropy over depth
    Xtr, ytr, Xva, yva, Xte, yte = task_b
    print("[tree] Task B: gini vs entropy")
    crit_curves = {}
    for crit in ("gini", "entropy"):
        params, depth_curve, leaf_curve = tune("classification", Xtr, ytr, Xva, yva, criterion=crit)
        crit_curves[crit] = (params, depth_curve, leaf_curve)
        best_val = depth_curve["val"][depth_curve["values"].index(params["max_depth"])]
        print(f"  {crit:8s} best max_depth={params['max_depth']}, "
              f"min_samples_leaf={params['min_samples_leaf']}, val F1 at that depth={best_val:.4f}")
    plot_validation_curves([crit_curves["gini"][1], crit_curves["entropy"][1]], ["gini", "entropy"],
                           "classification", "Task B: tree depth (gini vs entropy)",
                           "tree_b_depth_gini_entropy.png", fig_dir)

    # keep whichever criterion did better on validation (after the leaf sweep)
    best_crit = max(("gini", "entropy"), key=lambda c: max(crit_curves[c][2]["val"]))
    params_b, _, leaf_curve_b = crit_curves[best_crit]
    plot_validation_curves([leaf_curve_b], [best_crit], "classification",
                           f"Task B: min_samples_leaf (max_depth={params_b['max_depth']})",
                           "tree_b_min_samples_leaf.png", fig_dir)
    plot_learning_curve(learning_curve(params_b, Xtr, ytr, Xva, yva), "classification",
                        "Task B: learning curve (final config)", "tree_b_learning_curve.png", fig_dir)
    print(f"  final Task B config: {params_b}")

    cmp_b = compare_with_sklearn(params_b, Xtr, ytr, Xte, yte)
    print_comparison("[tree] Task B test set: ours vs DecisionTreeClassifier", cmp_b, "classification")
    plot_feature_importances(cmp_b["ours"]["model"], feature_names,
                             "Task B: our tree, feature importance", "tree_b_importance.png", fig_dir=fig_dir)
    results["task_b"] = dict(params=params_b, comparison=cmp_b,
                             criterion_curves={c: v[1] for c, v in crit_curves.items()})

    # Task A: regression
    print("\n[tree] Task A: regression")
    task_a, target = task_a_splits(task_a, feature_names, per_size)
    print(f"  target: {target} (RMSE below = RMSE of log(price))")
    Xtr, ytr, Xva, yva, Xte, yte = task_a
    params_a, depth_curve_a, leaf_curve_a = tune("regression", Xtr, ytr, Xva, yva)
    plot_validation_curves([depth_curve_a], ["mse"], "regression", "Task A: tree depth",
                           "tree_a_depth.png", fig_dir)
    plot_validation_curves([leaf_curve_a], ["mse"], "regression",
                           f"Task A: min_samples_leaf (max_depth={params_a['max_depth']})",
                           "tree_a_min_samples_leaf.png", fig_dir)
    plot_learning_curve(learning_curve(params_a, Xtr, ytr, Xva, yva), "regression",
                        "Task A: learning curve (final config)", "tree_a_learning_curve.png", fig_dir)
    print(f"  final Task A config: {params_a}")

    cmp_a = compare_with_sklearn(params_a, Xtr, ytr, Xte, yte)
    print_comparison("[tree] Task A test set: ours vs DecisionTreeRegressor", cmp_a, "regression")
    plot_feature_importances(cmp_a["ours"]["model"], feature_names,
                             "Task A: our tree, feature importance", "tree_a_importance.png", fig_dir=fig_dir)
    results["task_a"] = dict(params=params_a, comparison=cmp_a, depth_curve=depth_curve_a,
                             target=target)

    return results


# --- bonus: random forest from our own trees (Problem 9) ---------------------
FOREST_PARAMS = {
    # sklearn's RandomForestClassifier defaults
    "classification": dict(n_estimators=100, max_features="sqrt", min_samples_leaf=1),
    # a third of the features per split and leaves of >= 5 rows (Breiman's
    # suggestion for regression, also R randomForest's default). Fully grown
    # regression trees with 1-row leaves took ~2.8 s each in our NumPy code.
    "regression": dict(n_estimators=100, max_features=1 / 3, min_samples_leaf=5),
}


def sklearn_forest_twin(task, params):
    """sklearn forest with the same settings (single core, like ours)."""
    shared = dict(n_estimators=params["n_estimators"], max_features=params["max_features"],
                  min_samples_leaf=params["min_samples_leaf"], random_state=SEED, n_jobs=1)
    if task == "regression":
        return RandomForestRegressor(**shared)
    return RandomForestClassifier(**shared)


def plot_forest_curve(n_trees, val_curve, test_curve, single_tree, sk_score, task, title, name,
                      fig_dir=FIG_DIR):
    fig = Figure(figsize=(5, 3.4))
    ax = fig.subplots()
    ax.plot(n_trees, val_curve, label="our forest, validation")
    ax.plot(n_trees, test_curve, label="our forest, test")
    ax.axhline(single_tree, color="gray", ls="--", lw=1, label="our single tuned tree, test")
    ax.axhline(sk_score, color="tab:red", ls=":", lw=1.2, label="sklearn RandomForest, test")
    ax.set_xlabel("number of trees")
    ax.set_ylabel(metric_name(task))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, name, fig_dir)


def run_forest(task_a, task_b, feature_names, tree_results=None, fig_dir=FIG_DIR, per_size=True):
    """
    Random forest of our own trees on both tasks: validation/test error as
    trees are added, compared with a single tree and with sklearn's forest.
    tree_results = output of run(), used for the single-tree baseline (if
    None, a fully grown single tree is used instead). per_size: same Task A
    target as in run() -- keep it the same in both calls.
    """
    results = {}
    task_a, _ = task_a_splits(task_a, feature_names, per_size)
    for task, splits, key, label in (("classification", task_b, "task_b", "Task B"),
                                     ("regression", task_a, "task_a", "Task A")):
        Xtr, ytr, Xva, yva, Xte, yte = splits
        params = FOREST_PARAMS[task]
        print(f"\n[forest] {label}: {params['n_estimators']} trees, "
              f"max_features={params['max_features'] if isinstance(params['max_features'], str) else round(params['max_features'], 2)}, min_samples_leaf={params['min_samples_leaf']}")

        t0 = time.perf_counter()
        forest = RandomForest(task, random_state=SEED, **params).fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        test_score = score(task, yte, forest.predict(Xte))
        predict_time = time.perf_counter() - t0

        val_curve = [score(task, yva, p) for p in forest.staged_predict(Xva)]
        test_curve = [score(task, yte, p) for p in forest.staged_predict(Xte)]

        if tree_results is not None:
            single = tree_results[key]["comparison"]["ours"]["score"]
        else:
            single = score(task, yte, DecisionTree(**default_params(task)).fit(Xtr, ytr).predict(Xte))

        sk = sklearn_forest_twin(task, params)
        t0 = time.perf_counter()
        sk.fit(Xtr, ytr)
        sk_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        sk_score = score(task, yte, sk.predict(Xte))
        sk_pred = time.perf_counter() - t0

        m = metric_name(task)
        print(f"  {'':22s} {m:>8s} {'fit [s]':>9s} {'predict [s]':>12s}")
        print(f"  {'our single tree':22s} {single:8.4f}")
        print(f"  {'our forest':22s} {test_score:8.4f} {fit_time:9.2f} {predict_time:12.3f}")
        print(f"  {'sklearn RandomForest':22s} {sk_score:8.4f} {sk_fit:9.2f} {sk_pred:12.3f}")
        print(f"  test {m} with 1 / 10 / {len(test_curve)} trees: "
              f"{test_curve[0]:.4f} / {test_curve[min(9, len(test_curve) - 1)]:.4f} / {test_curve[-1]:.4f}")

        plot_forest_curve(np.arange(1, len(val_curve) + 1), val_curve, test_curve, single, sk_score,
                          task, f"{label}: random forest of our trees", f"forest_{key[-1]}_n_trees.png", fig_dir)
        plot_feature_importances(forest, feature_names, f"{label}: our forest, feature importance",
                                 f"forest_{key[-1]}_importance.png", fig_dir=fig_dir)
        results[key] = dict(params=params, test_score=test_score, fit_time=fit_time,
                            val_curve=val_curve, test_curve=test_curve,
                            sklearn_score=sk_score, sklearn_fit_time=sk_fit)
    return results


# --- quick check on a small built-in dataset ---------------------------------
def _demo(fig_dir="results/tree_demo"):
    """
    Runs the whole tree + forest study on sklearn's diabetes dataset (442
    rows, 10 features, ships with sklearn, no download), only to check that
    everything works end to end before the bina.az pipeline is ready.
    Not part of the report. It mirrors our two tasks:
      Task A: regression on the target
      Task B: above / below the TRAINING median of the target
    """
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X, y = data.data, data.target
    idx = np.random.default_rng(SEED).permutation(len(y))
    tr, va, te = idx[:265], idx[265:354], idx[354:]      # 60 / 20 / 20
    threshold = np.median(y[tr])                          # train only
    tier = (y > threshold).astype(int)

    task_a = (X[tr], y[tr], X[va], y[va], X[te], y[te])
    task_b = (X[tr], tier[tr], X[va], tier[va], X[te], tier[te])

    print("=== demo on sklearn's diabetes dataset (stand-in for bina.az) ===")
    t0 = time.perf_counter()
    tree_results = run(task_a, task_b, list(data.feature_names), fig_dir)
    forest_results = run_forest(task_a, task_b, list(data.feature_names), tree_results, fig_dir)
    print(f"\nfigures saved to {fig_dir}/  ({time.perf_counter() - t0:.1f}s)")
    return tree_results, forest_results


if __name__ == "__main__":
    _demo()
