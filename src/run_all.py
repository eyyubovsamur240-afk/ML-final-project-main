"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all                       # full run
    python -m src.run_all --fast                # quick smoke run on a training subsample
    python -m src.run_all --data path/to.csv    # dataset somewhere other than data/

What it does:
  1. data: clean, split 70/15/15, encode (src/data_prep.py, Problem 1)
  1b. the tree study (src/tree_experiments.py, Problems 4a, 5a, 9): tunes our
     tree on the validation split, compares it with scikit-learn's node by
     node and grows the random forest bonus; the tuned settings are the ones
     the trees use below (--no-tree-study keeps TREE_REGRESSION / _CLASSIFICATION
     and writes to no_tree_study/ subfolders, so the report's outputs are kept)
  2. Task A, log(price): our decision tree vs DecisionTreeRegressor vs the train mean
  3. Task B, price tier: our decision tree and our Pegasos SVM vs scikit-learn's
     DecisionTreeClassifier, SGDClassifier (hinge) and LinearSVC, and the majority class
  4. every model: k-fold CV on the training split (mean ± std), validation
     metrics, test metrics with 95% bootstrap intervals, fit and predict times,
     and how closely our tree reproduces scikit-learn's (Problem 5)
  5. error analysis, feature importance, fit-time scaling, figures (Problem 6),
     plus the data-exploration figures from src/eda.py when it exists
  6. writes results/metrics.json, results/results.md, report/tables/*.tex
     (results, timing, tree study and forest, error analysis) and
     report/figures/eval_*.pdf, and prints the comparison tables
     (--fast writes to results/fast/, report/tables/fast/, report/figures/fast/)

A model whose code is still a TODO stub is reported as "not implemented yet"
and skipped, so this runs end to end at every stage of the project.
Everything is seeded (data_prep.SEED): two runs give identical numbers
(apart from the timings).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import baselines, data_prep, error_analysis, evaluate, plots, tree_experiments
from .decision_tree import DecisionTree
from .svm import PegasosSVM

SEED = data_prep.SEED
DATA_PATH = getattr(data_prep, "DATA_PATH", "data/house_sale.csv")
RESULTS_DIR = "results"
TABLES_DIR = os.path.join("report", "tables")
FIGURES_DIR = plots.FIGURES_DIR

# --- Final hyperparameters (Problem 4c) ----------------------------------------
# A normal run takes the tree settings from the tree study, which tunes them on
# the VALIDATION split (tree_experiments.run); TREE_REGRESSION and
# TREE_CLASSIFICATION are only the --no-tree-study fallback. SVM is fixed: no
# code tunes lambda_ or n_iters yet (Problem 4b). The scikit-learn baselines
# reuse the same values, which is what makes the comparison in Problem 5 a
# matched one.
TREE_REGRESSION = {"criterion": "mse", "max_depth": 12, "min_samples_split": 2,
                   "min_samples_leaf": 5, "min_impurity_decrease": 0.0}
TREE_CLASSIFICATION = {"criterion": "gini", "max_depth": 10, "min_samples_split": 2,
                       "min_samples_leaf": 5, "min_impurity_decrease": 0.0}
SVM = {"lambda_": 1e-4, "n_iters": 200_000}

# Model names are shared by both tasks so the headline table lines them up.
OUR_TREE = "Ours: decision tree"
OUR_SVM = "Ours: SVM (Pegasos)"
SK_TREE = "sklearn: decision tree"
SK_SGD = "sklearn: SGD (hinge)"
SK_LINEAR_SVC = "sklearn: LinearSVC"
TRIVIAL = "Trivial baseline"


@dataclass
class ModelSpec:
    name: str
    make: Callable[[], object]   # returns a fresh, unfitted model
    ours: bool = False


def task_a_models(tree: dict | None = None, feature_names: list[str] | None = None,
                  per_size: bool = False) -> list[ModelSpec]:
    """per_size=True fits both trees on log(price per m2 / sot) and predicts
    log(price) (baselines.PerSizeTarget), the target the tree study tuned them on."""
    tree = TREE_REGRESSION if tree is None else tree

    def wrap(make):
        if not per_size:
            return make
        return lambda: baselines.PerSizeTarget(make(), feature_names)

    return [
        ModelSpec(OUR_TREE, wrap(lambda: DecisionTree(task="regression", random_state=SEED,
                                                      **tree)), ours=True),
        ModelSpec(SK_TREE, wrap(lambda: baselines.sk_tree_regressor(tree, SEED))),
        ModelSpec(TRIVIAL, baselines.MeanRegressor),
    ]


def task_b_models(tree: dict | None = None) -> list[ModelSpec]:
    tree = TREE_CLASSIFICATION if tree is None else tree

    # SGD's epochs and LinearSVC's C depend on the number of fit rows, so they are
    # built at fit time: matched on the full split, on every CV fold and in the scaling study.
    def sgd(n):
        return baselines.sk_sgd_hinge(SVM["lambda_"], baselines.epochs_for(SVM["n_iters"], n), SEED)

    def linear_svc(n):
        return baselines.sk_linear_svc(SVM["lambda_"], n, SEED)

    return [
        ModelSpec(OUR_TREE, lambda: DecisionTree(task="classification", random_state=SEED,
                                                 **tree), ours=True),
        # The hinge loss is scale-sensitive: standardize with the fit rows' statistics.
        ModelSpec(OUR_SVM, lambda: baselines.StandardScaled(PegasosSVM(random_state=SEED, **SVM)),
                  ours=True),
        ModelSpec(SK_TREE, lambda: baselines.sk_tree_classifier(tree, SEED)),
        ModelSpec(SK_SGD, lambda: baselines.SizedToFitData(sgd)),
        ModelSpec(SK_LINEAR_SVC, lambda: baselines.SizedToFitData(linear_svc)),
        ModelSpec(TRIVIAL, baselines.MajorityClassifier),
    ]


# --- 1. Data --------------------------------------------------------------------
def load_data(path: str = DATA_PATH, fast: bool = False) -> dict:
    """Arrays for both tasks from data_prep.load_splits (unscaled; scaling happens per model).

    Task A targets are log(price); Task B targets are the price tier (1 = premium),
    with the threshold set on the TRAINING prices by data_prep.
    """
    if not hasattr(data_prep, "load_splits"):
        raise SystemExit("run_all needs data_prep.load_splits(): merge the data-pipeline "
                         "branch (ali/data-pipeline) first.")
    s = data_prep.load_splits("regression", scaled=False, data_path=path, seed=SEED)
    data = {
        "X": {"train": s["X_train"], "val": s["X_val"], "test": s["X_test"]},
        "log_price": {"train": s["y_train"], "val": s["y_val"], "test": s["y_test"]},
        "tier": {part: s[f"{part}_df"]["price_tier"].to_numpy()
                 for part in ("train", "val", "test")},
        "df": {"train": s["train_df"], "val": s["val_df"], "test": s["test_df"]},
        "feature_names": list(s["feature_names"]),
        "threshold": float(s["tier_threshold"]),
    }
    if fast:
        data = subsample_train(data, 3_000)
    return data


def subsample_train(data: dict, n: int) -> dict:
    """Keep a seeded random n rows of the TRAINING split (validation and test untouched)."""
    n_train = len(data["log_price"]["train"])
    if n >= n_train:
        return data
    rows = np.sort(np.random.default_rng(SEED).choice(n_train, size=n, replace=False))
    out = {key: dict(value) if isinstance(value, dict) else value for key, value in data.items()}
    for key in ("X", "log_price", "tier"):
        out[key]["train"] = evaluate.take_rows(data[key]["train"], rows)
    out["df"]["train"] = data["df"]["train"].iloc[rows]
    return out


# --- 2-4. Fitting and scoring ------------------------------------------------------
def _is_todo(exc: Exception) -> bool:
    return isinstance(exc, NotImplementedError) and str(exc).startswith("TODO")


def score_regression(model, X, y) -> dict:
    return evaluate.price_metrics(y, model.predict(X))


def score_classification(model, X, y) -> dict:
    return evaluate.classification_metrics(y, model.predict(X), evaluate.model_scores(model, X))


def _summarise_warnings(caught) -> list[str]:
    """One line per distinct warning, with a count (e.g. LinearSVC not converging per fold)."""
    counts = Counter(f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught)
    return [msg if n == 1 else f"{msg} (x{n})" for msg, n in counts.items()]


def run_model(spec: ModelSpec, task: str, data: dict, folds: int, n_boot: int) -> tuple[dict, dict]:
    """Fit on train, score on validation and test, cross-validate on train.

    Returns (metrics for metrics.json, in-memory outputs for the analysis step).
    A model whose fit or predict is still a TODO stub is reported as "not
    implemented yet". If only predict_proba / decision_function is a stub, the
    model is still scored on its predictions and just has no AUC or ROC curve
    (n/a in the tables). Warnings raised meanwhile (e.g. a ConvergenceWarning)
    are recorded in the row instead of being printed in the progress output.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            row, outputs = _fit_and_score(spec, task, data, folds, n_boot)
        except NotImplementedError as exc:
            if not _is_todo(exc):
                raise
            row = {"model": spec.name, "ours": spec.ours, "status": "not implemented yet",
                   "reason": str(exc)}
            outputs = {}
    if caught:
        row["warnings"] = _summarise_warnings(caught)
    return row, outputs


def _fit_and_score(spec: ModelSpec, task: str, data: dict, folds: int,
                   n_boot: int) -> tuple[dict, dict]:
    target = "log_price" if task == "A" else "tier"
    X, y = data["X"], data[target]
    score = score_regression if task == "A" else score_classification
    row = {"model": spec.name, "ours": spec.ours}
    model = spec.make()
    _, fit_s = evaluate.timed(model.fit, X["train"], y["train"])
    pred_test, predict_s = evaluate.timed(model.predict, X["test"])
    scores_test = evaluate.model_scores(model, X["test"]) if task == "B" else None
    row.update(status="ok", fit_seconds=fit_s, predict_seconds=predict_s,
               n_train=len(y["train"]),
               validation=score(model, X["val"], y["val"]),
               test=score(model, X["test"], y["test"]))
    # Our SVM can record its objective on every training row while it trains (history_);
    # that is part of its fit time, so the timing caption says how often it happened.
    history = getattr(model, "history_", None)
    if isinstance(history, (list, tuple, np.ndarray)) and len(history):
        row["objective_evaluations"] = len(history)

    # 95% bootstrap intervals on the test set for the headline metrics.
    if task == "A":
        row["test_ci"] = {"rmse_log": evaluate.bootstrap_ci(evaluate.rmse, y["test"], pred_test,
                                                            n_boot=n_boot, seed=SEED)}
    else:
        f1 = lambda t, p: evaluate.precision_recall_f1(t, p)[2]  # noqa: E731
        row["test_ci"] = {"f1": evaluate.bootstrap_ci(f1, y["test"], pred_test,
                                                      n_boot=n_boot, seed=SEED)}
        if scores_test is not None:
            row["test_ci"]["roc_auc"] = evaluate.bootstrap_ci(
                evaluate.roc_auc, y["test"], scores_test, n_boot=n_boot, seed=SEED)

    if folds >= 2:
        row["cv"] = evaluate.cross_validate(
            spec.make, X["train"], y["train"], score, k=folds, seed=SEED,
            stratify=y["train"] if task == "B" else None)
    outputs = {"model": model, "pred": np.asarray(pred_test), "scores": scores_test}
    return row, outputs


def run_task(task: str, specs: list[ModelSpec], data: dict, folds: int, n_boot: int):
    rows, outputs = [], {}
    for spec in specs:
        print(f"  [{task}] {spec.name} ...", flush=True)
        row, out = run_model(spec, task, data, folds, n_boot)
        rows.append(row)
        if out:
            outputs[spec.name] = out
        else:
            print(f"        skipped: {row['reason']}")
        for message in row.get("warnings", []):
            print(f"        warning: {message}")
    return rows, outputs


# --- 5. Analysis -----------------------------------------------------------------------
def _linear_weights(model):
    """w of our SVM (through the scaler wrapper) or coef_ of a scikit-learn pipeline."""
    w = getattr(model, "w", None)
    if w is None and hasattr(model, "steps"):
        w = getattr(model.steps[-1][1], "coef_", None)
    return None if w is None else np.asarray(w, dtype=float).ravel()


def analyse(data: dict, outputs_a: dict, outputs_b: dict, perm_repeats: int) -> dict:
    """Error breakdowns and feature importance (Problem 6). Nothing here is fitted."""
    test, train = data["df"]["test"], data["df"]["train"]
    log_true, tier_true = data["log_price"]["test"], data["tier"]["test"]
    groups = error_analysis.feature_groups(data["feature_names"])
    bands = error_analysis.price_band(test["price"])
    coverage = error_analysis.district_coverage(test["location"], train["location"])
    margin = error_analysis.threshold_margin(test["price"], data["threshold"])
    out: dict = {"task_a": {}, "task_b": {}, "importance": {}, "svm_weights": {}, "notes": []}

    for name, o in outputs_a.items():
        if name == TRIVIAL:
            continue
        out["task_a"][name] = {
            "by_price_band": error_analysis.regression_breakdown(bands, log_true, o["pred"]),
            "by_category": error_analysis.regression_breakdown(test["category"], log_true, o["pred"]),
            "by_district_coverage": error_analysis.regression_breakdown(coverage, log_true,
                                                                        o["pred"]),
            "worst": error_analysis.worst_predictions(
                test, log_true, o["pred"],
                columns=("category", "location", "area_m2", "land_area_sot", "rooms")),
        }
    for name, o in outputs_b.items():
        if name == TRIVIAL:
            continue
        out["task_b"][name] = {
            "by_category": error_analysis.classification_breakdown(test["category"], tier_true,
                                                                   o["pred"]),
            "by_threshold_margin": error_analysis.classification_breakdown(margin, tier_true,
                                                                           o["pred"]),
            "by_district_coverage": error_analysis.classification_breakdown(coverage, tier_true,
                                                                            o["pred"]),
        }

    # Feature importance. Permutation importance runs on the VALIDATION split.
    X_val = data["X"]["val"]
    for task, outputs, y_val in (("A", outputs_a, data["log_price"]["val"]),
                                 ("B", outputs_b, data["tier"]["val"])):
        for name, o in outputs.items():
            if name == TRIVIAL:
                continue
            model = o["model"]
            if task == "A":
                def score(X, y, m=model):
                    return -evaluate.rmse(y, m.predict(X))
            else:
                def score(X, y, m=model):
                    return evaluate.precision_recall_f1(y, m.predict(X))[2]
            entry = {"permutation": error_analysis.permutation_importance(
                score, X_val, y_val, groups, n_repeats=perm_repeats, seed=SEED)}
            if getattr(model, "feature_importances_", None) is not None:
                entry["impurity"] = error_analysis.grouped_importances(
                    model.feature_importances_, groups)
            elif name == OUR_TREE:
                out["notes"].append(f"{task}: {OUR_TREE} has no feature_importances_, so its "
                                    "impurity-decrease importance is missing (Problem 6)")
            out["importance"][f"{task}: {name}"] = entry
            w = _linear_weights(model)
            if w is not None and len(w) == len(data["feature_names"]):
                out["svm_weights"][name] = error_analysis.top_weights(w, data["feature_names"], 15)
    return out


def _tree_size(model) -> dict:
    """Depth and leaf count, if the model exposes scikit-learn's get_depth/get_n_leaves."""
    out = {}
    for attr in ("get_depth", "get_n_leaves"):
        method = getattr(model, attr, None)
        try:
            out[attr[4:]] = int(method()) if callable(method) else None
        except NotImplementedError as exc:
            if not _is_todo(exc):
                raise
            out[attr[4:]] = None
    return out


def tree_vs_sklearn(outputs_a: dict, outputs_b: dict) -> dict:
    """How closely our tree reproduces scikit-learn's with the same hyperparameters (Problem 5a).

    Both are CART with the same settings on the same rows, so most test predictions
    should match; the rest come from tie-breaking between equally good splits and
    from where the threshold sits between two feature values.
    """
    out = {}
    for task, outputs in (("task_a", outputs_a), ("task_b", outputs_b)):
        ours, sk = outputs.get(OUR_TREE), outputs.get(SK_TREE)
        if ours is None or sk is None:
            continue
        if task == "task_a":
            diff = np.abs(ours["pred"] - sk["pred"])
            entry = {"mean_abs_diff_log": float(diff.mean()), "max_abs_diff_log": float(diff.max()),
                     "share_identical": float(np.mean(diff < 1e-9))}
        else:
            entry = {"share_identical": float(np.mean(ours["pred"] == sk["pred"]))}
        entry["ours"] = _tree_size(ours["model"])
        entry["sklearn"] = _tree_size(sk["model"])
        out[task] = entry
    return out


def scaling_study(task: str, specs: list[ModelSpec], data: dict, names: set,
                  fractions=(0.125, 0.25, 0.5, 1.0), repeats: int = 3) -> list[dict]:
    """Fit time vs training-set size for the given implemented models (Problem 5c).

    Each point is the median of `repeats` fits on the same nested subsample,
    after one untimed warm-up fit, so every point is measured the same way.
    """
    target = "log_price" if task == "A" else "tier"
    X, y = data["X"]["train"], data[target]["train"]
    order = np.random.default_rng(SEED).permutation(len(y))
    subsets = [np.sort(order[: max(50, int(frac * len(y)))]) for frac in fractions]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # already reported in each model's row
        for spec in specs:
            if spec.name == TRIVIAL or spec.name not in names:
                continue
            spec.make().fit(evaluate.take_rows(X, subsets[0]), y[subsets[0]])  # warm-up
            for idx in subsets:
                X_sub = evaluate.take_rows(X, idx)  # same memory layout as the full split
                seconds = [evaluate.timed(spec.make().fit, X_sub, y[idx])[1]
                           for _ in range(repeats)]
                rows.append({"task": task, "model": spec.name, "n_train": len(idx),
                             "fit_seconds": float(np.median(seconds))})
    return rows


# --- 6. Output ----------------------------------------------------------------------
def result_rows(rows: list[dict], task: str) -> list[dict]:
    """The results table of one task: CV mean ± std, test value [95% CI] (RMSE; F1 and
    AUC) and the other test metrics, one row per implemented model, the trivial floor last."""
    ok = [r for r in rows if r.get("status") == "ok"]
    ok = [r for r in ok if r["model"] != TRIVIAL] + [r for r in ok if r["model"] == TRIVIAL]
    table = []
    for r in ok:
        test, cv = r["test"], r.get("cv", {})
        if task == "A":
            table.append({"model": r["model"], "cv_rmse": cv.get("rmse_log"),
                          "test_rmse": r["test_ci"]["rmse_log"], "cv_r2": cv.get("r2_log"),
                          "test_r2": test["r2_log"], "test_mae": test["mae_log"],
                          "test_mape": test["median_ape"]})
        else:
            table.append({"model": r["model"], "cv_f1": cv.get("f1"), "test_f1": r["test_ci"]["f1"],
                          "cv_auc": cv.get("roc_auc"),
                          "test_auc": r["test_ci"].get("roc_auc", test.get("roc_auc")),
                          "test_precision": test["precision"], "test_recall": test["recall"],
                          "test_accuracy": test["accuracy"]})
    return table


def headline_rows(table_a: list[dict], table_b: list[dict]) -> list[dict]:
    """Every model on both tasks in one table (the results table Problem 6 asks for).

    Test RMSE(log) for Task A and test F1 for Task B, each with its 95% bootstrap
    interval; None (n/a) where a model does not do that task, e.g. the SVM on Task A.
    """
    rmse = {r["model"]: r["test_rmse"] for r in table_a}
    f1 = {r["model"]: r["test_f1"] for r in table_b}
    names = [name for name in dict.fromkeys([*f1, *rmse]) if name != TRIVIAL]
    if TRIVIAL in rmse or TRIVIAL in f1:
        names.append(TRIVIAL)
    return [{"model": name, "task_a_rmse": rmse.get(name), "task_b_f1": f1.get(name)}
            for name in names]


TABLE_COLUMNS = {
    "A": ["model", "cv_rmse", "test_rmse", "cv_r2", "test_r2", "test_mae", "test_mape"],
    "B": ["model", "cv_f1", "test_f1", "cv_auc", "test_auc", "test_precision", "test_recall",
          "test_accuracy"],
    "headline": ["model", "task_a_rmse", "task_b_f1"],
}
TABLE_HEADERS = {
    "A": ["Model", "CV RMSE(log)", "test RMSE(log) [95% CI]", "CV R2(log)", "test R2(log)",
          "test MAE(log)", "test median APE (fraction)"],
    "B": ["Model", "CV F1", "test F1 [95% CI]", "CV AUC", "test AUC [95% CI]", "test precision",
          "test recall", "test accuracy"],
    "headline": ["Model", "Task A: test RMSE(log) [95% CI]", "Task B: test F1 [95% CI]"],
}
LATEX_HEADERS = {
    "A": ["Model", "CV RMSE", "Test RMSE [95\\% CI]", "CV $R^2$", "Test $R^2$", "Test MAE",
          "Median APE"],
    "B": ["Model", "CV F1", "Test F1 [95\\% CI]", "CV AUC", "Test AUC [95\\% CI]", "Precision",
          "Recall", "Accuracy"],
    "headline": ["Model", "Task A: RMSE [95\\% CI]", "Task B: F1 [95\\% CI]"],
}


def agreement_rows(agreement: dict) -> list[dict]:
    """tree_vs_sklearn() as table rows: share of identical test predictions, and size."""
    rows = []
    for task, entry in agreement.items():
        rows.append({"task": "A" if task == "task_a" else "B",
                     "share_identical": entry["share_identical"],
                     "mean_abs_diff_log": entry.get("mean_abs_diff_log"),
                     "depth_ours": entry["ours"].get("depth"),
                     "depth_sklearn": entry["sklearn"].get("depth"),
                     "leaves_ours": entry["ours"].get("n_leaves"),
                     "leaves_sklearn": entry["sklearn"].get("n_leaves")})
    return rows


def timing_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    out = []
    for task, rows in (("A", rows_a), ("B", rows_b)):
        for r in rows:
            if r.get("status") == "ok":
                out.append({"task": task, "model": r["model"], "n_train": r["n_train"],
                            "fit_seconds": r["fit_seconds"], "predict_seconds": r["predict_seconds"],
                            "cv_fit_seconds": r["cv"]["fit_seconds"] if "cv" in r else None})
    return out


def tree_study_rows(study: dict) -> list[dict]:
    """The tree study as a table: one row per setting or score, one column per task."""
    a, b = study["task_a"], study["task_b"]

    def depth(s):
        return "unlimited" if s["settings"]["max_depth"] is None else s["settings"]["max_depth"]

    def size(s, who):
        return f"{s[who]['depth']:,} / {s[who]['n_leaves']:,}"

    def same_tree(s):
        if s["same_structure"]:
            return "yes"
        if s.get("sklearn_seed_matching_ours") is not None:
            return "ties only"
        return "same leaves" if s["same_leaf_groups"] else "no"

    items = [
        ("Target", a.get("target", "log(price)"), "price tier"),
        ("Score", study["metric"]["task_a"], study["metric"]["task_b"]),
        ("Criterion", a["settings"]["criterion"], b["settings"]["criterion"]),
        ("max_depth", depth(a), depth(b)),
        ("min_samples_leaf", a["settings"]["min_samples_leaf"], b["settings"]["min_samples_leaf"]),
        ("Validation, our tree", a["validation"], b["validation"]),
        ("Test, our tree", a["test_ours"], b["test_ours"]),
        ("Test, scikit-learn's tree", a["test_sklearn"], b["test_sklearn"]),
        ("Depth / leaves, ours", size(a, "size_ours"), size(b, "size_ours")),
        ("Depth / leaves, scikit-learn", size(a, "size_sklearn"), size(b, "size_sklearn")),
        ("Same test predictions (share)", a["prediction_agreement"], b["prediction_agreement"]),
        ("Same tree, node by node", same_tree(a), same_tree(b)),
        ("Test, our forest", a["forest_test"], b["forest_test"]),
        ("Test, scikit-learn's forest", a["sklearn_forest_test"], b["sklearn_forest_test"]),
    ]
    return [{"item": item, "task_a": value_a, "task_b": value_b} for item, value_a, value_b in items]


def _records(table) -> list[dict]:
    """An error-analysis table (a DataFrame, or its records once loaded from JSON) as dicts."""
    if table is None:
        return []
    return table.to_dict("records") if hasattr(table, "to_dict") else list(table)


# Error-analysis groupings shown in the report, and the columns each of our models fills.
ERROR_GROUPINGS = (("by_category", "By property category"),
                   ("by_district_coverage", "By training listings in the district"))
ERROR_COLUMNS = {("task_a", OUR_TREE): {"median_ape": "a_ape", "bias_log": "a_bias"},
                 ("task_b", OUR_TREE): {"error_rate": "b_tree"},
                 ("task_b", OUR_SVM): {"error_rate": "b_svm"}}


def error_rows(analysis: dict, grouping: str) -> list[dict]:
    """Test errors per group: Task A median APE and bias of our tree, Task B error rate of
    our tree and our SVM (missing = n/a, e.g. for a model that is not implemented yet)."""
    rows: dict[str, dict] = {}
    for (task, model), columns in ERROR_COLUMNS.items():
        for rec in _records(analysis[task].get(model, {}).get(grouping)):
            row = rows.setdefault(rec["group"], {"group": rec["group"], "n": rec["n"]})
            row.update({new: rec[old] for old, new in columns.items()})
    return list(rows.values())


def worst_rows(analysis: dict, n: int = 8) -> list[dict]:
    """The test listings with the largest |log error| of our tree on Task A (as ranked by
    error_analysis.worst_predictions; area 0 = not recorded)."""
    def recorded(value):
        return value is not None and np.isfinite(value) and value > 0

    def floor_area(value):
        return int(round(value)) if recorded(value) else None

    def land(value):  # plots of a fraction of a sot are common, so keep one decimal
        return f"{round(float(value), 1):g}" if recorded(value) else None

    return [{"category": rec.get("category"), "location": rec.get("location"),
             "area_m2": floor_area(rec.get("area_m2")),
             "land_area_sot": land(rec.get("land_area_sot")),
             "price": int(rec["price"]), "predicted": int(rec["predicted"]),
             "error": f"{rec['error_pct']:+.0f}%"}
            for rec in _records(analysis["task_a"].get(OUR_TREE, {}).get("worst"))[:n]]


ERROR_TABLE = (["group", "n", "a_ape", "a_bias", "b_tree", "b_svm"],
               ["Group", "n", "A: median APE, our tree", "A: bias (log), our tree",
                "B: error rate, our tree", "B: error rate, our SVM"],
               ["Group", "$n$", "A: median APE", "A: bias", "B: error, tree", "B: error, SVM"])
WORST_TABLE = (["category", "location", "area_m2", "land_area_sot", "price", "predicted", "error"],
               ["Category", "Location", "Area (m2)", "Land (sot)", "Price (AZN)", "Predicted (AZN)",
                "Error"],
               ["Category", "Location", "Area (m$^2$)", "Land (sot)", "Price", "Predicted",
                "Error"])


def _panel_table(panels: list[tuple[str, list[dict]]], columns: list[str], headers: list[str],
                 caption: str, label: str, align: str, wide: bool = False) -> str:
    """evaluate.latex_table with its rows in titled panels (e.g. by category, then by district)."""
    rows = [row for _, panel in panels for row in panel]
    lines = evaluate.latex_table(rows, columns, headers, caption, label, align=align,
                                 wide=wide).splitlines()
    first, last = lines.index(r"    \midrule") + 1, lines.index(r"    \bottomrule")
    out, body, start = lines[:first - 1], lines[first:last], 0
    for title, panel in panels:
        if not panel:
            continue
        out += [r"    \midrule",
                rf"    \multicolumn{{{len(columns)}}}{{@{{}}l}}{{\textit{{{title}}}}} \\",
                *body[start:start + len(panel)]]
        start += len(panel)
    return "\n".join(out + lines[last:]) + "\n"


def _jsonable(obj):
    """numpy / pandas values -> plain JSON types (used as json.dump's default)."""
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        return obj.to_dict(orient="records")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _finite(obj):
    """NaN / inf -> None, so metrics.json is strict JSON (e.g. an AUC that is undefined)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {key: _finite(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_finite(value) for value in obj]
    return obj


def _captions(config: dict, skipped: dict | None = None, n_train: int | None = None,
              objective_evals: dict | None = None) -> dict:
    """Table captions that say how the numbers were produced.

    skipped = {"A": [...], "B": [...]}: models still "not implemented yet", which have
    no row in that task's table; the captions name them so no model silently disappears.
    objective_evals = {model: k}: models whose fit evaluates their objective on every
    training row k times (our SVM's history_), which the timing caption has to say.
    """
    prefix = "SMOKE RUN (training split subsampled to 3,000 rows). " if config["fast"] else ""
    cv = (f"CV = {config['folds']}-fold mean $\\pm$ std on the training split; "
          if config["folds"] >= 2 else "")
    test = (f"test = held-out test split, with a 95\\% bootstrap interval "
            f"({config['n_boot']:,} resamples).")
    skipped = {task: [evaluate.latex_escape(name) for name in names]
               for task, names in (skipped or {}).items()}

    def not_shown(task):
        names = skipped.get(task, [])
        return f" Not implemented yet, so not shown: {', '.join(names)}." if names else ""

    by_model = {}
    for task in ("A", "B"):
        for name in skipped.get(task, []):
            by_model.setdefault(name, []).append(task)
    headline_skipped = "; ".join(f"{name} (Task {' and '.join(tasks)})"
                                 for name, tasks in by_model.items())
    floor_b = ("The trivial floor predicts the training majority class for every listing, so "
               "its premium-class precision, recall and F1 are 0 when that class is standard.")
    folds = config["folds"]
    cv_fit = (f", per CV fold (mean $\\pm$ std over the {folds} folds, each fitted on "
              f"{folds - 1}/{folds} of those rows)" if folds >= 2 else "")
    rows = f" ({n_train:,} rows)" if n_train else ""
    logged = "".join(f" The fit time of {evaluate.latex_escape(name)} includes evaluating its "
                     f"objective on every training row {k:,} times, so it grows with the "
                     "training-set size." for name, k in (objective_evals or {}).items())
    per_size = (" Both trees are fitted on log(price per m$^2$, or per sot for land) and the "
                "listing's log area is added back, so every metric is on log(price)."
                if config.get("tree_task_a_target", "log(price)") != "log(price)" else "")
    return {
        "A": f"{prefix}Task A, regression on log(price). {cv}{test} Median APE = median of "
             "$|$predicted $-$ actual$|$ / actual price in AZN, as a fraction (0.12 = 12\\%)."
             + per_size + not_shown("A"),
        "B": f"{prefix}Task B, premium vs standard (premium = above the training median "
             f"price). Precision, recall and F1 are for the premium class. {cv}{test} "
             f"{floor_b} n/a = not available (e.g. no AUC for a model without continuous "
             "scores yet)." + not_shown("B"),
        "timing": f"{prefix}Fit time on the full training split{rows} (one run){cv_fit}, and "
                  f"predict time on the test split, in seconds.{logged}",
        "headline": f"{prefix}Headline results on the held-out test split: Task A RMSE on "
                    "log(price), Task B F1 for the premium class, each with a 95\\% bootstrap "
                    f"interval ({config['n_boot']:,} resamples). Tables~\\ref{{tab:task-a}} and "
                    "\\ref{tab:task-b} add the cross-validated scores and the other metrics. "
                    "Trivial baseline: the training mean (Task A) and the training majority "
                    "class (Task B). n/a = no result for that task (the SVMs are classifiers "
                    "only)."
                    + (f" Not implemented yet: {headline_skipped}." if headline_skipped else ""),
    }


def _max_features(value) -> str:
    """A forest's max_features as a share of the d feature columns, for a caption."""
    if value == "sqrt":
        return r"$\sqrt{d}$"
    if isinstance(value, float):
        return f"${value:.2g}\\,d$"
    return str(value)


def _analysis_captions(config: dict, study: dict | None, n_worst: int) -> dict:
    """Captions of the tree-study, error-analysis and worst-listings tables."""
    prefix = "SMOKE RUN (training split subsampled to 3,000 rows). " if config["fast"] else ""
    out = {
        "errors": f"{prefix}Where our models fail on the test split. Per group: the number of "
                  "test listings, Task~A median $|$predicted $-$ actual$|$ / actual price and "
                  "mean error of $\\log(\\text{price})$ (bias; $>0$ = over-priced) of our tree, "
                  "and the Task~B error rate of our tree and our SVM. District coverage counts "
                  "the training listings in the listing's district. n/a = no result for that "
                  "model.",
        "worst": f"{prefix}The {n_worst} test listings our tree misprices most on Task~A, "
                 "ranked by $|\\log(\\text{predicted} / \\text{actual})|$, which counts "
                 "over- and under-pricing by the same factor alike, with the predicted price "
                 "in AZN and error = predicted / actual $-$ 1. n/a = not recorded (land plots "
                 "have a land area in sot instead of a floor area, flats have no separate "
                 "land).",
    }
    if study:
        def grid(values):
            return ", ".join("unlimited" if v is None else str(v) for v in values)

        forest_a, forest_b = study["task_a"]["forest"], study["task_b"]["forest"]
        trees = (str(forest_b["n_estimators"]) if forest_a["n_estimators"] == forest_b["n_estimators"]
                 else f"{forest_b['n_estimators']} (Task~B) or {forest_a['n_estimators']} (Task~A)")
        per_size = ("; for Task~A both trees are fitted on log(price per m$^2$, or per sot for "
                    "land) and the log area is added back, so the scores are on log(price)"
                    if study["task_a"].get("target", "log(price)") != "log(price)" else "")
        out["tree_study"] = (
            f"{prefix}Tree study (\\texttt{{src/tree\\_experiments.py}}). Our tree's max\\_depth "
            f"$\\in$ \\{{{grid(tree_experiments.GRID_DEPTHS)}\\}} and min\\_samples\\_leaf $\\in$ "
            f"\\{{{grid(tree_experiments.GRID_LEAF)}\\}} are tuned together on the validation "
            "split (for Task~B with both Gini and entropy; on a tie the simpler tree wins), then "
            "scored once on the test split. scikit-learn's tree gets the same settings and rows"
            f"{per_size}. Ties only: scikit-learn builds exactly our tree with another "
            "random\\_state, so only the order in which equally good splits are tried differs; "
            "same leaves: the trees differ but every leaf holds the same training rows. Forest "
            f"(Problem~9): {trees} of our trees, each on a bootstrap sample of the training rows, "
            f"trying {_max_features(forest_b['max_features'])} (Task~B) or "
            f"{_max_features(forest_a['max_features'])} (Task~A) of the $d$ feature columns at "
            f"each split, with leaves of at least {forest_b['min_samples_leaf']} (Task~B) or "
            f"{forest_a['min_samples_leaf']} (Task~A) rows, against scikit-learn's RandomForest "
            "with the same settings.")
    return out


def write_outputs(report: dict, results_dir: str, tables_dir: str) -> list[str]:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    paths = []

    path = os.path.join(results_dir, "metrics.json")
    plain = _finite(json.loads(json.dumps(report, default=_jsonable)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plain, f, indent=2, allow_nan=False)
    paths.append(path)

    skipped = {task: [r["model"] for r in report[f"task_{task.lower()}"]
                      if r.get("status") != "ok"] for task in ("A", "B")}
    objective_evals = {r["model"]: r["objective_evaluations"]
                       for r in report["task_a"] + report["task_b"]
                       if r.get("objective_evaluations")}
    captions = _captions(report["config"], skipped, report["data"]["train"], objective_evals)
    title = "# Results" + (" (SMOKE RUN, --fast)" if report["config"]["fast"] else "")
    md = [f"{title}, seed {SEED}", ""]
    for task, name in (("headline", "Headline: both tasks, test split"),
                       ("A", "Task A: log(price)"), ("B", "Task B: premium vs standard")):
        md += [f"## {name}", "", "```",
               evaluate.format_table(report["tables"][task], TABLE_COLUMNS[task],
                                     TABLE_HEADERS[task]), "```", ""]
    md += ["## Timing (seconds)", "", "```",
           evaluate.format_table(report["timing"], ["task", "model", "n_train", "fit_seconds",
                                                    "predict_seconds", "cv_fit_seconds"]), "```"]
    if report["scaling"]:
        md += ["", "## Fit time vs training-set size (seconds, median of the repeats)", "", "```",
               evaluate.format_table(report["scaling"], ["task", "model", "n_train",
                                                         "fit_seconds"]), "```"]
    if report["tree_vs_sklearn"]:
        md += ["", "## Our tree vs scikit-learn's, same hyperparameters (test split)", "", "```",
               evaluate.format_table(agreement_rows(report["tree_vs_sklearn"]),
                                     ["task", "share_identical", "mean_abs_diff_log",
                                      "depth_ours", "depth_sklearn", "leaves_ours",
                                      "leaves_sklearn"]), "```"]
    study = report.get("tree_study")
    if study:
        md += ["", "## Tree study: settings tuned on validation, test scores, random forest", "",
               "```", evaluate.format_table(tree_study_rows(study), ["item", "task_a", "task_b"],
                                            ["", "Task A", "Task B"]), "```"]
    analysis = report["analysis"]
    errors = [(title, error_rows(analysis, key)) for key, title in ERROR_GROUPINGS]
    worst = worst_rows(analysis)
    for title, rows in errors:
        if rows:
            md += ["", f"## Test errors of our models, {title.lower()}", "", "```",
                   evaluate.format_table(rows, ERROR_TABLE[0], ERROR_TABLE[1]), "```"]
    if worst:
        md += ["", "## Task A: largest test errors of our tree", "", "```",
               evaluate.format_table(worst, WORST_TABLE[0], WORST_TABLE[1]), "```"]
    path = os.path.join(results_dir, "results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    paths.append(path)

    with_cv = report["config"]["folds"] >= 2
    timing_columns = ["task", "model", "fit_seconds",
                      *(["cv_fit_seconds"] if with_cv else []), "predict_seconds"]
    timing_headers = ["Task", "Model", "Fit (s)", *(["CV fit (s)"] if with_cv else []),
                      "Predict (s)"]
    tex = {
        "headline.tex": evaluate.latex_table(
            report["tables"]["headline"], TABLE_COLUMNS["headline"], LATEX_HEADERS["headline"],
            captions["headline"], label="tab:results",
            align=r"l@{\hspace{8pt}}c@{\hspace{8pt}}c"),  # fits one column of the report
        "task_a.tex": evaluate.latex_table(
            report["tables"]["A"], TABLE_COLUMNS["A"], LATEX_HEADERS["A"], captions["A"],
            label="tab:task-a", align="l" + "c" * (len(TABLE_COLUMNS["A"]) - 1), wide=True),
        "task_b.tex": evaluate.latex_table(
            report["tables"]["B"], TABLE_COLUMNS["B"], LATEX_HEADERS["B"], captions["B"],
            label="tab:task-b", align="l" + "c" * (len(TABLE_COLUMNS["B"]) - 1), wide=True),
        "timing.tex": evaluate.latex_table(
            report["timing"], timing_columns, timing_headers, captions["timing"],
            label="tab:timing", align="ll" + "r" * (len(timing_columns) - 2)),
    }
    more = _analysis_captions(report["config"], study, len(worst))
    if study:
        tex["tree_study.tex"] = evaluate.latex_table(
            tree_study_rows(study), ["item", "task_a", "task_b"], ["", "Task A", "Task B"],
            more["tree_study"], label="tab:tree-study", align="lcc")
    if any(rows for _, rows in errors):
        tex["errors.tex"] = _panel_table(errors, ERROR_TABLE[0], ERROR_TABLE[2], more["errors"],
                                         label="tab:errors", align="lrcccc", wide=True)
    if worst:
        tex["worst.tex"] = evaluate.latex_table(
            worst, WORST_TABLE[0], WORST_TABLE[2], more["worst"], label="tab:worst",
            align="llrrrrr", wide=True)
    for name, text in tex.items():
        path = os.path.join(tables_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(path)
    return paths


def _importance_label(panel: str) -> str:
    return "increase in RMSE(log)" if panel.startswith("A:") else "drop in F1"


def make_figures(data: dict, outputs_a: dict, outputs_b: dict, analysis: dict,
                 scaling: list[dict], out_dir: str) -> list[str]:
    paths = []
    real_a = {k: v for k, v in outputs_a.items() if k != TRIVIAL}
    real_b = {k: v for k, v in outputs_b.items() if k != TRIVIAL}
    if real_a:
        paths.append(plots.predicted_vs_actual(data["log_price"]["test"],
                                               {k: v["pred"] for k, v in real_a.items()}, out_dir))
        paths.append(plots.error_by_group(
            {k: v["by_price_band"] for k, v in analysis["task_a"].items()}, "median_ape",
            "Task A: median relative price error by price band (test)", "median |error| / price",
            out_dir, "eval_error_by_price_band.pdf"))
        paths.append(plots.error_by_group(
            {k: v["by_district_coverage"] for k, v in analysis["task_a"].items()}, "median_ape",
            "Task A: error by number of training listings in the district",
            "median |error| / price", out_dir, "eval_error_by_district_coverage.pdf"))
    if real_b:
        y = data["tier"]["test"]
        paths.append(plots.confusion_matrices({k: (y, v["pred"]) for k, v in real_b.items()},
                                              out_dir))
        with_scores = {k: v["scores"] for k, v in real_b.items() if v["scores"] is not None}
        if with_scores:
            paths.append(plots.roc_curves(with_scores, y, out_dir))
        paths.append(plots.error_by_group(
            {k: v["by_threshold_margin"] for k, v in analysis["task_b"].items()}, "error_rate",
            "Task B: error rate by distance from the tier threshold, |log(price / threshold)|",
            "error rate", out_dir, "eval_error_by_threshold_margin.pdf"))
    if analysis["importance"]:
        paths.append(plots.importance_bars(
            {k: v["permutation"] for k, v in analysis["importance"].items()},
            title="Permutation importance on the validation split (drop in score)",
            out_dir=out_dir, xlabel=_importance_label))
        impurity = {k: v["impurity"] for k, v in analysis["importance"].items() if "impurity" in v}
        if impurity:
            paths.append(plots.importance_bars(
                impurity, title="Tree impurity-decrease importance (training split)",
                out_dir=out_dir, name="eval_tree_impurity_importance.pdf",
                xlabel="share of total impurity decrease"))
    if analysis["svm_weights"]:
        paths.append(plots.importance_bars(analysis["svm_weights"], value="weight",
                                           title="Largest linear-SVM weights (standardized features)",
                                           out_dir=out_dir, name="eval_svm_weights.pdf",
                                           xlabel="weight (standardized features)"))
    if scaling:
        paths.append(plots.fit_time_scaling(
            [{**r, "model": f"{r['task']}: {r['model']}"} for r in scaling], out_dir))
    return paths


def eda_figures(data_path: str, out_dir: str):
    """Figures 1-4 from src/eda.py (the data owner's module), if the project has one.

    Returns the numbers it reports, None when there is no src/eda.py yet, or
    {"status": "failed", "reason": ...} when it raises: a problem in those figures
    must not throw away the results this run has already computed.
    """
    if importlib.util.find_spec(f"{__package__}.eda") is None:
        return None
    try:
        from . import eda
        return eda.make_eda_figures(data_path, out_dir)
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _print_analysis(analysis: dict, agreement: dict) -> None:
    for name, tables in analysis["task_a"].items():
        print(evaluate.format_table(tables["by_price_band"].to_dict("records"),
                                    ["group", "n", "rmse_log", "median_ape", "bias_log"],
                                    title=f"\nTask A error by price band: {name}"))
    for name, tables in analysis["task_b"].items():
        print(evaluate.format_table(tables["by_threshold_margin"].to_dict("records"),
                                    ["group", "n", "error_rate", "false_premium_rate",
                                     "missed_premium_rate"],
                                    title=f"\nTask B error by |log(price/threshold)|: {name}"))
    for task, entry in agreement.items():
        same = entry["share_identical"]
        print(f"\nOur tree vs scikit-learn's ({task}): identical predictions on {same:.1%} of "
              f"the test rows; depth/leaves ours {entry['ours']}, sklearn {entry['sklearn']}")
    for note in analysis["notes"]:
        print(f"\nNote: {note}")


TREE_SETTINGS = ("criterion", "max_depth", "min_samples_split", "min_samples_leaf",
                 "min_impurity_decrease")


def _curve(curve: dict) -> dict:
    """A tree_experiments validation curve without its fit times, which differ between runs."""
    return {key: curve[key] for key in ("param", "values", "train", "val", "n_leaves")
            if key in curve}


def tree_study(data: dict, fig_dir: str) -> dict:
    """Tune, benchmark and ensemble our tree (tree_experiments, Problems 4a, 5a, 9).

    Tuning only looks at the validation split; the tuned settings are returned
    for the trees in the headline tables, so both report the same tree. The
    summary also keeps the evidence behind them (the validation grid and curves,
    the comparison with scikit-learn, the forest's curves) for metrics.json.
    """
    X, parts = data["X"], ("train", "val", "test")
    task_a = tuple(a for p in parts for a in (X[p], data["log_price"][p]))
    task_b = tuple(a for p in parts for a in (X[p], data["tier"][p]))
    names = data["feature_names"]
    trees = tree_experiments.run(task_a, task_b, names, fig_dir=fig_dir)
    forest = tree_experiments.run_forest(task_a, task_b, names, trees, fig_dir=fig_dir)
    summary = {}
    for key in ("task_a", "task_b"):
        params, cmp, grid = trees[key]["params"], trees[key]["comparison"], trees[key]["grid"]
        ours, sk = cmp["ours"], cmp["sklearn"]
        summary[key] = {
            "settings": {k: params.get(k, d) for k, d in zip(
                TREE_SETTINGS, ("mse", None, 2, 1, 0.0), strict=True)},
            "validation": grid[(params["max_depth"], params["min_samples_leaf"])],
            "grid": [{"max_depth": depth, "min_samples_leaf": leaf, "validation": score}
                     for (depth, leaf), score in grid.items()],
            "test_ours": ours["score"], "test_sklearn": sk["score"],
            "size_ours": {"depth": ours["depth"], "n_leaves": ours["n_leaves"]},
            "size_sklearn": {"depth": sk["depth"], "n_leaves": sk["n_leaves"]},
            "prediction_agreement": cmp["pred_agreement"],
            "same_structure": bool(cmp["same_structure"]),
            "same_leaf_groups": bool(cmp["same_leaf_groups"]),
            "sklearn_seed_matching_ours": cmp["sklearn_seed_matching_ours"],
            "ours_fit_seconds": ours["fit_time"], "sklearn_fit_seconds": sk["fit_time"],
            "forest": dict(forest[key]["params"]),
            "forest_test": forest[key]["test_score"],
            "sklearn_forest_test": forest[key]["sklearn_score"],
            "forest_val_curve": forest[key]["val_curve"],
            "forest_test_curve": forest[key]["test_curve"],
            "forest_fit_seconds": forest[key]["fit_time"],
            "sklearn_forest_fit_seconds": forest[key]["sklearn_fit_time"],
        }
    summary["task_a"]["target"] = trees["task_a"]["target"]
    summary["task_a"]["depth_curve"] = _curve(trees["task_a"]["depth_curve"])
    summary["task_b"]["criterion_curves"] = {
        criterion: _curve(curve) for criterion, curve in trees["task_b"]["criterion_curves"].items()}
    summary["metric"] = {"task_a": "RMSE of log(price)", "task_b": "F1 (premium)"}
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Reproduce every number and figure.")
    parser.add_argument("--data", default=DATA_PATH, help="path to house_sale.csv")
    parser.add_argument("--fast", action="store_true",
                        help="smoke run: 3,000 training rows, 3 folds, 200 bootstrap samples; "
                             "writes to fast/ subfolders so the real outputs are kept")
    parser.add_argument("--folds", type=int, default=None, help="CV folds (0 = no CV; default 5)")
    parser.add_argument("--boot", type=int, default=None, help="bootstrap samples (default 1000)")
    parser.add_argument("--no-scaling", action="store_true", help="skip the fit-time scaling study")
    parser.add_argument("--no-tree-study", action="store_true",
                        help="skip the tree tuning / sklearn comparison / forest study and use "
                             "the fixed TREE_REGRESSION and TREE_CLASSIFICATION settings; "
                             "writes to no_tree_study/ subfolders so the real outputs are kept")
    parser.add_argument("--no-figures", action="store_true", help="skip writing figures")
    parser.add_argument("--results-dir", default=None, help=f"default {RESULTS_DIR}")
    parser.add_argument("--tables-dir", default=None, help=f"default {TABLES_DIR}")
    parser.add_argument("--figures-dir", default=None, help=f"default {FIGURES_DIR}")
    args = parser.parse_args(argv)
    folds = args.folds if args.folds is not None else (3 if args.fast else 5)
    n_boot = args.boot if args.boot is not None else (200 if args.fast else 1000)
    if folds == 1 or folds < 0:
        parser.error("--folds must be 0 (no cross-validation) or at least 2")
    if n_boot < 1:
        parser.error("--boot must be at least 1")
    perm_repeats = 1 if args.fast else 3
    # A smoke run, or a run without the tree study, gets its own subfolders: neither
    # makes the report's numbers, so neither may overwrite them.
    sub = "fast" if args.fast else ("no_tree_study" if args.no_tree_study else "")
    out_dirs = {}
    for key, default in (("results", RESULTS_DIR), ("tables", TABLES_DIR),
                         ("figures", FIGURES_DIR)):
        given = getattr(args, f"{key}_dir")
        out_dirs[key] = given or (os.path.join(default, sub) if sub else default)

    # 1. data
    data = load_data(args.data, fast=args.fast)
    sizes = {part: len(data["log_price"][part]) for part in ("train", "val", "test")}
    print(f"Data: {sizes}, {len(data['feature_names'])} features, "
          f"tier threshold = {data['threshold']:,.0f} AZN (train median)")

    # 1b. tree study: its validation-tuned settings and Task A target are the
    # ones the trees use in the tables below
    tree_a, tree_b, per_size, study = TREE_REGRESSION, TREE_CLASSIFICATION, False, None
    if not args.no_tree_study:
        print("Tree study (tuning on validation, scikit-learn comparison, random forest) ...",
              flush=True)
        if args.no_figures:
            with tempfile.TemporaryDirectory() as scratch:
                study = tree_study(data, scratch)
        else:
            study = tree_study(data, out_dirs["figures"])
        tree_a = study["task_a"]["settings"]
        tree_b = study["task_b"]["settings"]
        per_size = study["task_a"]["target"] != "log(price)"

    # 2-4. models
    print("Task A: regression on log(price)"
          + (" (trees fitted on log(price per m2 / sot), area added back)" if per_size else ""))
    specs_a = task_a_models(tree_a, data["feature_names"], per_size)
    rows_a, outputs_a = run_task("A", specs_a, data, folds, n_boot)
    print("Task B: premium vs standard")
    specs_b = task_b_models(tree_b)
    rows_b, outputs_b = run_task("B", specs_b, data, folds, n_boot)

    # 5. analysis
    analysis = analyse(data, outputs_a, outputs_b, perm_repeats)
    agreement = tree_vs_sklearn(outputs_a, outputs_b)
    scaling = []
    if not args.no_scaling:
        print("Fit-time scaling study ...", flush=True)
        for task, specs, outputs in (("A", specs_a, outputs_a), ("B", specs_b, outputs_b)):
            scaling += scaling_study(task, specs, data, set(outputs),
                                     repeats=1 if args.fast else 3)

    # 6. output
    config = {"tree_regression": tree_a, "tree_classification": tree_b,
              "tree_task_a_target": "log(price per m2 / sot)" if per_size else "log(price)",
              "tree_settings_from": "fixed" if study is None else "tree study (validation)",
              "svm": SVM, "folds": folds, "n_boot": n_boot, "fast": args.fast}
    tables = {"A": result_rows(rows_a, "A"), "B": result_rows(rows_b, "B")}
    tables["headline"] = headline_rows(tables["A"], tables["B"])
    report = {
        "seed": SEED,
        "config": config,
        "data": {**sizes, "n_features": len(data["feature_names"]),
                 "tier_threshold_azn": data["threshold"],
                 "test_premium_share": float(np.mean(data["tier"]["test"]))},
        "task_a": rows_a,
        "task_b": rows_b,
        "tables": tables,
        "timing": timing_rows(rows_a, rows_b),
        "scaling": scaling,
        "tree_vs_sklearn": agreement,
        "tree_study": study,
        "analysis": analysis,
    }
    for task, name in (("A", "Task A: log(price)"), ("B", "Task B: premium vs standard")):
        print()
        print(evaluate.format_table(report["tables"][task], TABLE_COLUMNS[task],
                                    TABLE_HEADERS[task],
                                    title=f"{name} (CV on train: mean ± std; test: value "
                                          "[95% bootstrap CI])"))
    print()
    print(evaluate.format_table(report["timing"], ["task", "model", "n_train", "fit_seconds",
                                                   "predict_seconds"], title="Timing (seconds)"))
    _print_analysis(analysis, agreement)
    skipped = [f"{t}: {r['model']}" for t, rows in (("A", rows_a), ("B", rows_b))
               for r in rows if r.get("status") != "ok"]
    if skipped:
        print("\nNot implemented yet (skipped): " + ", ".join(skipped))

    if not args.no_figures:
        eda = eda_figures(args.data, out_dirs["figures"])
        if eda is not None:
            report["eda"] = eda
        eda_failed = isinstance(eda, dict) and eda.get("status") == "failed"
        if eda_failed:
            print("\nWarning: src/eda.py failed, so Figures 1-4 may be missing or out of date "
                  f"({eda['reason']})")
    written = write_outputs(report, out_dirs["results"], out_dirs["tables"])
    if not args.no_figures:
        written += make_figures(data, outputs_a, outputs_b, analysis, scaling,
                                out_dirs["figures"])
        if "eda" in report and not eda_failed:
            written.append(os.path.join(out_dirs["figures"], "fig1-4 (data exploration, src/eda.py)"))
    print("\nWrote:\n  " + "\n  ".join(written))
    return report


if __name__ == "__main__":
    main()
