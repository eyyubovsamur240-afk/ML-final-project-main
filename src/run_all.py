"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all                       # full run
    python -m src.run_all --fast                # quick smoke run on a training subsample
    python -m src.run_all --data path/to.csv    # dataset somewhere other than data/

What it does:
  1. data: clean, split 70/15/15, encode (src/data_prep.py, Problem 1)
  2. Task A, log(price): our decision tree vs DecisionTreeRegressor vs the train mean
  3. Task B, price tier: our decision tree and our Pegasos SVM vs scikit-learn's
     DecisionTreeClassifier, SGDClassifier (hinge) and LinearSVC, and the majority class
  4. every model: k-fold CV on the training split (mean ± std), validation
     metrics, test metrics with 95% bootstrap intervals, fit and predict times,
     and how closely our tree reproduces scikit-learn's (Problem 5)
  5. error analysis, feature importance, fit-time scaling, figures (Problem 6),
     plus the data-exploration figures from src/eda.py when it exists
  6. writes results/metrics.json, results/results.md, report/tables/*.tex and
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
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import baselines, data_prep, error_analysis, evaluate, plots
from .decision_tree import DecisionTree
from .svm import PegasosSVM

SEED = data_prep.SEED
DATA_PATH = getattr(data_prep, "DATA_PATH", "data/house_sale.csv")
RESULTS_DIR = "results"
TABLES_DIR = os.path.join("report", "tables")
FIGURES_DIR = plots.FIGURES_DIR

# --- Final hyperparameters (Problem 4c) ----------------------------------------
# The tree and SVM owners set these from the VALIDATION results only (never the
# test set). The scikit-learn baselines reuse the same values, which is what
# makes the comparison in Problem 5 a matched one.
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


def task_a_models() -> list[ModelSpec]:
    return [
        ModelSpec(OUR_TREE, lambda: DecisionTree(task="regression", random_state=SEED,
                                                 **TREE_REGRESSION), ours=True),
        ModelSpec(SK_TREE, lambda: baselines.sk_tree_regressor(TREE_REGRESSION, SEED)),
        ModelSpec(TRIVIAL, baselines.MeanRegressor),
    ]


def task_b_models() -> list[ModelSpec]:
    # SGD's epochs and LinearSVC's C depend on the number of fit rows, so they are
    # built at fit time: matched on the full split, on every CV fold and in the scaling study.
    def sgd(n):
        return baselines.sk_sgd_hinge(SVM["lambda_"], baselines.epochs_for(SVM["n_iters"], n), SEED)

    def linear_svc(n):
        return baselines.sk_linear_svc(SVM["lambda_"], n, SEED)

    return [
        ModelSpec(OUR_TREE, lambda: DecisionTree(task="classification", random_state=SEED,
                                                 **TREE_CLASSIFICATION), ours=True),
        # The hinge loss is scale-sensitive: standardize with the fit rows' statistics.
        ModelSpec(OUR_SVM, lambda: baselines.StandardScaled(PegasosSVM(random_state=SEED, **SVM)),
                  ours=True),
        ModelSpec(SK_TREE, lambda: baselines.sk_tree_classifier(TREE_CLASSIFICATION, SEED)),
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
    A model that hits a TODO stub anywhere (fit, predict, ...) is reported as
    "not implemented yet". Warnings raised meanwhile (e.g. a ConvergenceWarning)
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
            "worst": error_analysis.worst_predictions(test, log_true, o["pred"]),
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
    """The results table of one task: CV mean ± std, test value [95% CI] and the other
    test metrics, one row per implemented model, the trivial floor last."""
    ok = [r for r in rows if r.get("status") == "ok"]
    ok = [r for r in ok if r["model"] != TRIVIAL] + [r for r in ok if r["model"] == TRIVIAL]
    table = []
    for r in ok:
        test, cv = r["test"], r.get("cv", {})
        if task == "A":
            table.append({"model": r["model"], "cv_rmse": cv.get("rmse_log"),
                          "test_rmse": r["test_ci"]["rmse_log"], "test_mae": test["mae_log"],
                          "test_r2": test["r2_log"], "test_mape": test["median_ape"]})
        else:
            table.append({"model": r["model"], "cv_f1": cv.get("f1"), "test_f1": r["test_ci"]["f1"],
                          "test_precision": test["precision"], "test_recall": test["recall"],
                          "test_accuracy": test["accuracy"], "test_auc": test.get("roc_auc")})
    return table


TABLE_COLUMNS = {
    "A": ["model", "cv_rmse", "test_rmse", "test_mae", "test_r2", "test_mape"],
    "B": ["model", "cv_f1", "test_f1", "test_precision", "test_recall", "test_accuracy",
          "test_auc"],
}
TABLE_HEADERS = {
    "A": ["Model", "CV RMSE(log)", "test RMSE(log) [95% CI]", "test MAE(log)", "test R2(log)",
          "test median APE"],
    "B": ["Model", "CV F1", "test F1 [95% CI]", "test precision", "test recall", "test accuracy",
          "test AUC"],
}
LATEX_HEADERS = {
    "A": ["Model", "CV RMSE", "Test RMSE [95\\% CI]", "Test MAE", "Test $R^2$",
          "Median APE"],
    "B": ["Model", "CV F1", "Test F1 [95\\% CI]", "Precision", "Recall", "Accuracy", "AUC"],
}


def timing_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    out = []
    for task, rows in (("A", rows_a), ("B", rows_b)):
        for r in rows:
            if r.get("status") == "ok":
                out.append({"task": task, "model": r["model"], "n_train": r["n_train"],
                            "fit_seconds": r["fit_seconds"], "predict_seconds": r["predict_seconds"],
                            "cv_fit_seconds": r["cv"]["fit_seconds"] if "cv" in r else None})
    return out


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


def _captions(config: dict) -> dict:
    """Table captions that say how the numbers were produced."""
    prefix = "SMOKE RUN (training split subsampled to 3,000 rows). " if config["fast"] else ""
    cv = (f"CV = {config['folds']}-fold mean $\\pm$ std on the training split; "
          if config["folds"] >= 2 else "")
    test = (f"test = held-out test split, with a 95\\% bootstrap interval "
            f"({config['n_boot']:,} resamples).")
    return {
        "A": f"{prefix}Task A, regression on log(price). {cv}{test} Median APE = median "
             "absolute percentage error of the price in AZN.",
        "B": f"{prefix}Task B, premium vs standard (premium = above the training median "
             f"price). Precision, recall and F1 are for the premium class. {cv}{test}",
        "timing": f"{prefix}Fit time on the training split and predict time on the test split "
                  "(seconds, one run).",
    }


def write_outputs(report: dict, results_dir: str, tables_dir: str) -> list[str]:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    paths = []

    path = os.path.join(results_dir, "metrics.json")
    plain = _finite(json.loads(json.dumps(report, default=_jsonable)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plain, f, indent=2, allow_nan=False)
    paths.append(path)

    captions = _captions(report["config"])
    title = "# Results" + (" (SMOKE RUN, --fast)" if report["config"]["fast"] else "")
    md = [f"{title}, seed {SEED}", ""]
    for task, name in (("A", "Task A: log(price)"), ("B", "Task B: premium vs standard")):
        md += [f"## {name}", "", "```",
               evaluate.format_table(report["tables"][task], TABLE_COLUMNS[task],
                                     TABLE_HEADERS[task]), "```", ""]
    md += ["## Timing (seconds)", "", "```",
           evaluate.format_table(report["timing"], ["task", "model", "n_train", "fit_seconds",
                                                    "predict_seconds", "cv_fit_seconds"]), "```"]
    path = os.path.join(results_dir, "results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    paths.append(path)

    tex = {
        "task_a.tex": evaluate.latex_table(
            report["tables"]["A"], TABLE_COLUMNS["A"], LATEX_HEADERS["A"], captions["A"],
            label="tab:task-a", align="lccccc", wide=True),
        "task_b.tex": evaluate.latex_table(
            report["tables"]["B"], TABLE_COLUMNS["B"], LATEX_HEADERS["B"], captions["B"],
            label="tab:task-b", align="lcccccc", wide=True),
        "timing.tex": evaluate.latex_table(
            report["timing"], ["task", "model", "n_train", "fit_seconds", "predict_seconds"],
            ["Task", "Model", "Train rows", "Fit (s)", "Predict (s)"], captions["timing"],
            label="tab:timing", align="llrrr"),
    }
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


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Reproduce every number and figure.")
    parser.add_argument("--data", default=DATA_PATH, help="path to house_sale.csv")
    parser.add_argument("--fast", action="store_true",
                        help="smoke run: 3,000 training rows, 3 folds, 200 bootstrap samples; "
                             "writes to fast/ subfolders so the real outputs are kept")
    parser.add_argument("--folds", type=int, default=None, help="CV folds (0 = no CV; default 5)")
    parser.add_argument("--boot", type=int, default=None, help="bootstrap samples (default 1000)")
    parser.add_argument("--no-scaling", action="store_true", help="skip the fit-time scaling study")
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
    out_dirs = {}
    for key, default in (("results", RESULTS_DIR), ("tables", TABLES_DIR),
                         ("figures", FIGURES_DIR)):
        given = getattr(args, f"{key}_dir")
        out_dirs[key] = given or (os.path.join(default, "fast") if args.fast else default)

    # 1. data
    data = load_data(args.data, fast=args.fast)
    sizes = {part: len(data["log_price"][part]) for part in ("train", "val", "test")}
    print(f"Data: {sizes}, {len(data['feature_names'])} features, "
          f"tier threshold = {data['threshold']:,.0f} AZN (train median)")

    # 2-4. models
    print("Task A: regression on log(price)")
    specs_a = task_a_models()
    rows_a, outputs_a = run_task("A", specs_a, data, folds, n_boot)
    print("Task B: premium vs standard")
    specs_b = task_b_models()
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
    config = {"tree_regression": TREE_REGRESSION, "tree_classification": TREE_CLASSIFICATION,
              "svm": SVM, "folds": folds, "n_boot": n_boot, "fast": args.fast}
    report = {
        "seed": SEED,
        "config": config,
        "data": {**sizes, "n_features": len(data["feature_names"]),
                 "tier_threshold_azn": data["threshold"],
                 "test_premium_share": float(np.mean(data["tier"]["test"]))},
        "task_a": rows_a,
        "task_b": rows_b,
        "tables": {"A": result_rows(rows_a, "A"), "B": result_rows(rows_b, "B")},
        "timing": timing_rows(rows_a, rows_b),
        "scaling": scaling,
        "tree_vs_sklearn": agreement,
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
            print(f"\nWarning: src/eda.py failed, Figures 1-4 not written ({eda['reason']})")
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
