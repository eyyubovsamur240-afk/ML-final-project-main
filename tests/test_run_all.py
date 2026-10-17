"""run_all.py end to end on a small synthetic dataset (no Kaggle file needed).

The data loader is replaced by a fake one with the same shape as
data_prep.load_splits. Two cases: with the team's real tree and SVM (which
may still be TODO stubs, then they must be reported as skipped), and with
small stand-in models, so every code path for "our" models is exercised now.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import src
from src import run_all

CATEGORIES = ["Yeni tikili", "Köhnə tikili", "Obyekt"]
LOCATIONS = ["Səbail r.", "Nəsimi r.", "Xırdalan", "Binə q."]

# src/tree_experiments.py comes with the decision tree. Without it run_all skips the
# tree study, and these tests check that fallback instead of the study's outputs.
HAVE_TREE_STUDY = importlib.util.find_spec("src.tree_experiments") is not None
needs_tree_study = pytest.mark.skipif(not HAVE_TREE_STUDY,
                                      reason="no src/tree_experiments.py on this checkout yet")


def fake_data(n: int = 420, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "category": rng.choice(CATEGORIES, n),
        "location": rng.choice(LOCATIONS, n, p=[0.5, 0.3, 0.15, 0.05]),
        "area_m2": rng.uniform(30, 250, n).round(),
        "rooms": rng.integers(1, 6, n),
    })
    log_price = 9.0 + 0.9 * np.log(df["area_m2"]) + 0.3 * (df["location"] == "Səbail r.")
    df["price"] = np.exp(log_price + rng.normal(0, 0.15, n)).round()
    df["log_price"] = np.log(df["price"])
    one_hot = pd.get_dummies(df[["category", "location"]], dtype=float)
    X = pd.concat([df[["area_m2", "rooms"]], one_hot], axis=1)
    part = np.array(["train"] * int(0.7 * n) + ["val"] * int(0.15 * n)
                    + ["test"] * (n - int(0.7 * n) - int(0.15 * n)))
    threshold = float(df.loc[part == "train", "price"].median())
    df["price_tier"] = (df["price"] > threshold).astype(int)
    split = {p: df[part == p].reset_index(drop=True) for p in ("train", "val", "test")}
    return {
        "X": {p: X[part == p].to_numpy() for p in split},
        "log_price": {p: split[p]["log_price"].to_numpy() for p in split},
        "tier": {p: split[p]["price_tier"].to_numpy() for p in split},
        "df": split,
        "feature_names": list(X.columns),
        "threshold": threshold,
    }


class StandInTree:
    """Our tree's constructor signature, scikit-learn inside (test only)."""

    def __init__(self, task="classification", criterion="gini", random_state=42, **params):
        cls = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
        crit = "squared_error" if criterion == "mse" else criterion
        self.inner = cls(criterion=crit, random_state=random_state, **params)

    def fit(self, X, y):
        self.inner.fit(X, y)
        self.feature_importances_ = self.inner.feature_importances_
        return self

    def predict(self, X):
        return self.inner.predict(X)

    def predict_proba(self, X):
        return self.inner.predict_proba(X)

    def get_depth(self):
        return self.inner.get_depth()

    def get_n_leaves(self):
        return self.inner.get_n_leaves()


class HalfDoneTree(StandInTree):
    """fit and predict work, predict_proba is still a TODO stub."""

    def predict_proba(self, X):
        raise NotImplementedError("TODO: leaf class shares")


class StandInSVM:
    """Mini-batch hinge sub-gradient, our SVM's constructor signature (test only).

    The batches are drawn from random_state, like Pegasos' samples, so a run that
    stopped passing the seed would no longer reproduce its numbers.
    """

    def __init__(self, lambda_=1e-4, n_iters=1_000, random_state=42, **settings):
        # our SVM's other settings (n_epochs, batch_size, ...) are accepted and ignored
        self.lambda_, self.n_iters = lambda_, min(n_iters, 300)
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        y_pm = np.where(np.asarray(y) == 1, 1.0, -1.0)
        Xa = np.hstack([X, np.ones((len(X), 1))])
        w = np.zeros(Xa.shape[1])
        self.history_ = []  # the objective on every row every 100 steps, like our Pegasos
        for t in range(1, self.n_iters + 1):
            batch = rng.choice(len(Xa), size=max(1, len(Xa) // 2), replace=False)
            Xb, yb = Xa[batch], y_pm[batch]
            active = yb * (Xb @ w) < 1
            grad = self.lambda_ * w - (yb[active, None] * Xb[active]).sum(0) / len(Xb)
            w -= grad / (self.lambda_ * t + 10)
            if t % 100 == 0:
                hinge = np.maximum(0, 1 - y_pm * (Xa @ w)).mean()
                self.history_.append(self.lambda_ / 2 * w[:-1] @ w[:-1] + hinge)
        self.w, self.b = w[:-1], w[-1]
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def _fake_load_data(path, fast=False):
    """load_data on the synthetic set; --fast subsamples it like the real 3,000-row smoke run."""
    data = fake_data()
    return run_all.subsample_train(data, 200) if fast else data


@pytest.fixture(autouse=True)
def small_svm_study(monkeypatch):
    """The SVM study's grids cut down, so each run_all test stays quick, and no
    cleaning log from a real CSV that may sit in data/ (the data here is fake)."""
    from src import svm_study

    monkeypatch.setattr(run_all, "cleaning_summary", lambda path: None)
    monkeypatch.setattr(svm_study, "LAMBDAS", [1e-3, 1e-2])
    monkeypatch.setattr(svm_study, "EPOCHS", [5, 10])
    monkeypatch.setattr(svm_study, "SWEEP_EPOCHS", 10)
    monkeypatch.setattr(svm_study, "FIRST_VERSION", {**svm_study.FIRST_VERSION, "n_iters": 500})
    monkeypatch.setattr(svm_study, "RFF_GAMMAS", [0.03, 0.1])
    monkeypatch.setattr(svm_study, "RFF_LAMBDAS", [1e-4, 1e-3])
    monkeypatch.setattr(svm_study, "RFF_COMPONENTS", 64)
    monkeypatch.setattr(svm_study, "RFF_COMPONENT_SWEEP", [16, 64])
    monkeypatch.setattr(run_all, "SVM_RFF", {**run_all.SVM_RFF, "n_components": 64})


def _run(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(run_all, "load_data", _fake_load_data)
    monkeypatch.setattr(run_all, "eda_figures", lambda path, out_dir: None)  # needs the real CSV
    return run_all.main(["--folds", "3", "--boot", "30",
                         "--results-dir", str(tmp_path / "results"),
                         "--tables-dir", str(tmp_path / "tables"),
                         "--figures-dir", str(tmp_path / "figures"), *extra])


def test_run_all_with_the_team_models(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch)
    for task in ("task_a", "task_b"):
        for row in report[task]:
            assert row["status"] in ("ok", "not implemented yet"), row
            if not row["ours"]:
                assert row["status"] == "ok", "baselines always run"
    assert (tmp_path / "results" / "metrics.json").exists()
    for task, table in (("task_a", "task_a.tex"), ("task_b", "task_b.tex"),
                        ("task_b", "headline.tex")):
        tex = (tmp_path / "tables" / table).read_text(encoding="utf-8")
        for row in report[task]:
            if row["status"] != "ok":
                assert row["model"] in tex, f"{table} names the model it leaves out"


def test_run_all_with_stand_in_models_exercises_every_step(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "DecisionTree", StandInTree)
    monkeypatch.setattr(run_all, "PegasosSVM", StandInSVM)
    report = _run(tmp_path, monkeypatch)

    names_a = {r["model"]: r for r in report["task_a"]}
    names_b = {r["model"]: r for r in report["task_b"]}
    assert all(r["status"] == "ok" for r in [*names_a.values(), *names_b.values()])
    tree_a = names_a[run_all.OUR_TREE]
    assert len(tree_a["cv"]["rmse_log"]["folds"]) == 3
    assert tree_a["test_ci"]["rmse_log"]["lo"] <= tree_a["test"]["rmse_log"] <= \
        tree_a["test_ci"]["rmse_log"]["hi"]
    # The stand-in tree is scikit-learn with the same settings, so they must agree exactly.
    assert tree_a["test"]["rmse_log"] == pytest.approx(names_a[run_all.SK_TREE]["test"]["rmse_log"])
    assert names_b[run_all.OUR_SVM]["test"]["f1"] > 0.6
    assert "roc_auc" in names_b[run_all.OUR_SVM]["test_ci"]
    assert names_b[run_all.OUR_SVM]["objective_evaluations"] == 3, "300 steps, every 100"
    assert "objective_evaluations" not in names_b[run_all.SK_SGD]

    analysis = report["analysis"]
    assert run_all.OUR_TREE in analysis["task_a"] and run_all.TRIVIAL not in analysis["task_a"]
    assert "impurity" in analysis["importance"][f"A: {run_all.OUR_TREE}"]
    assert run_all.OUR_SVM in analysis["svm_weights"]
    assert {r["model"] for r in report["scaling"]} >= {run_all.OUR_TREE, run_all.OUR_SVM}
    assert max(r["n_train"] for r in report["scaling"]) == 294, "the full split is a point too"

    # The stand-in tree is scikit-learn's, so the two trees must agree everywhere.
    agreement = report["tree_vs_sklearn"]
    assert agreement["task_a"]["share_identical"] == 1.0
    assert agreement["task_b"]["share_identical"] == 1.0
    assert agreement["task_b"]["ours"] == agreement["task_b"]["sklearn"]

    table_a = [r["model"] for r in report["tables"]["A"]]
    table_b = {r["model"]: r for r in report["tables"]["B"]}
    assert run_all.OUR_SVM not in table_a, "the SVM has no Task A entry"
    assert table_a[-1] == run_all.TRIVIAL, "the floor is listed last"
    assert table_b[run_all.OUR_TREE]["test_precision"] is not None
    assert table_b[run_all.TRIVIAL]["test_accuracy"] > 0.4

    headline = {r["model"]: r for r in report["tables"]["headline"]}
    assert list(headline)[-1] == run_all.TRIVIAL and len(headline) == 8, "each model once"
    assert headline[run_all.OUR_SVM]["task_a_rmse"] is None, "the SVM has no Task A entry"
    assert headline[run_all.OUR_TREE]["task_a_rmse"] == tree_a["test_ci"]["rmse_log"]
    assert headline[run_all.OUR_TREE]["task_b_f1"] == table_b[run_all.OUR_TREE]["test_f1"]

    saved = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["data"]["train"] == 294 and saved["config"]["folds"] == 3
    md = (tmp_path / "results" / "results.md").read_text(encoding="utf-8")
    assert "## Headline" in md and "## Fit time vs training-set size" in md
    assert "## Our tree vs scikit-learn's" in md
    tex = (tmp_path / "tables" / "headline.tex").read_text(encoding="utf-8")
    assert r"\label{tab:results}" in tex and r"\begin{table}" in tex
    assert tex.count("& n/a &") == 5, "the SVMs have no Task A result"
    assert "Not implemented yet" not in tex, "every model ran"
    assert "Trivial baseline: the training mean (Task A)" in tex
    tex = (tmp_path / "tables" / "task_a.tex").read_text(encoding="utf-8")
    assert "CV $R^2$" in tex
    tex = (tmp_path / "tables" / "task_b.tex").read_text(encoding="utf-8")
    assert r"\label{tab:task-b}" in tex and "Ours: SVM (Pegasos)" in tex
    assert r"\begin{table*}" in tex and "3-fold" in tex and "CV AUC" in tex
    svm_row = next(line for line in tex.splitlines() if line.strip().startswith(run_all.OUR_SVM))
    assert svm_row.count("[") == 2, "test F1 and test AUC both have an interval"
    tex = (tmp_path / "tables" / "timing.tex").read_text(encoding="utf-8")
    assert "CV fit (s)" in tex and "294 rows" in tex
    study_tables = ("tree_study.tex",) if HAVE_TREE_STUDY else ()
    for name in ("headline.tex", "task_a.tex", "task_b.tex", "timing.tex", *study_tables,
                 "errors.tex", "worst.tex", "svm_study.tex", "svm_convergence.tex",
                 "svm_rff.tex", "data.tex"):
        lines = (tmp_path / "tables" / name).read_text(encoding="utf-8").splitlines()
        spec = next(line for line in lines if r"\begin{tabular}" in line).split("}{", 1)[1]
        spec = re.sub(r"@\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)  # drop @{...} spacing groups
        spec = re.sub(r">\{[^{}]*\}", "", spec)       # and >{...} column formats
        spec = re.sub(r"p\{[^{}]*\}", "l", spec)      # a wrapped p{width} column is one column
        n_columns = sum(spec.count(c) for c in "lcr")
        header = lines[lines.index(r"    \toprule") + 1]
        assert header.count("&") + 1 == n_columns, f"{name}: column spec matches the header"
    assert tex.count("includes evaluating its objective") == 2, "our two SVMs"
    assert "Ours: SVM (Pegasos) includes evaluating its objective on every training row 3 times" \
        in tex
    figures = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {"eval_pred_vs_actual.pdf", "eval_confusion_matrices.pdf", "eval_roc_curves.pdf",
            "eval_svm_weights.pdf", "eval_fit_time_scaling.pdf",
            "eval_tree_impurity_importance.pdf"} <= figures

    # The tree study, the forest and the error analysis reach the report as tables.
    study = saved["tree_study"]
    if HAVE_TREE_STUDY:
        assert {"tree_a_grid.png", "tree_a_depth.png", "tree_b_depth_gini_entropy.png",
                "forest_a_n_trees.png", "forest_b_n_trees.png"} <= figures, "the report uses these"
        for key in ("task_a", "task_b"):
            settings = study[key]["settings"]
            chosen = [g["validation"] for g in study[key]["grid"]
                      if (g["max_depth"], g["min_samples_leaf"]) == (
                          settings["max_depth"], settings["min_samples_leaf"])]
            assert chosen == [study[key]["validation"]], "the validation grid is saved"
            assert len(study[key]["forest_test_curve"]) == study[key]["forest"]["n_estimators"]
        assert set(study["task_b"]["criterion_curves"]) == {"gini", "entropy"}
        tex = (tmp_path / "tables" / "tree_study.tex").read_text(encoding="utf-8")
        assert r"\label{tab:tree-study}" in tex and "Test, our forest" in tex
        assert "max\\_depth" in tex and "tuned together on the validation split" in tex
        assert "## Tree study" in md
    else:
        assert study is None and saved["config"]["tree_settings_from"] == "fixed"
        assert saved["config"]["tree_regression"] == run_all.TREE_REGRESSION
        assert not (tmp_path / "tables" / "tree_study.tex").exists()
        assert "## Tree study" not in md
    # The SVM study picks the SVMs' settings on validation and reaches the report.
    svm = saved["svm_study"]
    assert saved["config"]["svm_settings_from"] == "SVM study (validation)"
    assert saved["config"]["svm"] == svm["linear"]["settings"]
    assert saved["config"]["svm_rff"] == svm["rff"]["settings"]
    assert svm["linear"]["settings"]["lambda_"] in [r["lambda_"] for r in
                                                    svm["linear"]["lambda_sweep"]]
    assert {"svm_lambda.png", "svm_objective.png", "svm_rff_grid.png",
            "svm_rff_components.png"} <= figures
    assert svm["figures"] == ["svm_lambda.png", "svm_objective.png", "svm_rff_grid.png",
                              "svm_rff_components.png"], "file names only"
    assert names_b[run_all.OUR_RFF]["svm"]["margin"] > 0, "margin of our RFF SVM recorded"
    for name, label in (("svm_study.tex", "tab:svm-study"),
                        ("svm_convergence.tex", "tab:svm-convergence"),
                        ("svm_rff.tex", "tab:svm-rff"), ("data.tex", "tab:data")):
        tex = (tmp_path / "tables" / name).read_text(encoding="utf-8")
        assert rf"\label{{{label}}}" in tex, name
        assert "TEXCELL" not in tex, "every LaTeX cell is put back"
    tex = (tmp_path / "tables" / "svm_study.tex").read_text(encoding="utf-8")
    assert r"\ensuremath{" in tex and r"$1/(\lambda(t+t_0))$" in tex, "math cells stay LaTeX"
    tex = (tmp_path / "tables" / "numbers.tex").read_text(encoding="utf-8")
    assert r"\providecommand{\defval}" in tex
    for name in ("NTrain", "SvmLambda", "SvmBFOne", "RffBFOne", "TreeBFOne", "PctCrossSplitDup"):
        assert rf"\defval{{{name}}}" in tex, name
    assert r"\defval{NTrain}{294}" in tex
    assert "## SVM study" in md and "## Data" in md
    tex = (tmp_path / "tables" / "errors.tex").read_text(encoding="utf-8")
    assert r"\textit{By property category}" in tex and r"\begin{table*}" in tex
    assert r"\textit{By training listings in the district}" in tex
    for category in set(report["analysis"]["task_a"][run_all.OUR_TREE]["by_category"]["group"]):
        assert run_all.evaluate.latex_escape(category) in tex
    tex = (tmp_path / "tables" / "worst.tex").read_text(encoding="utf-8")
    assert r"\label{tab:worst}" in tex and tex.count(r"\%") == 8, "eight listings, error in %"
    for heading in ("## Test errors of our models, by property category",
                    "## Task A: largest test errors of our tree"):
        assert heading in md


def _without_timings(obj):
    """Drop every *_seconds entry: wall-clock times may differ between runs, nothing else may."""
    if isinstance(obj, dict):
        return {k: _without_timings(v) for k, v in obj.items() if not k.endswith("_seconds")}
    if isinstance(obj, list):
        return [_without_timings(v) for v in obj]
    return obj


@pytest.mark.parametrize("fast", [False, True])
def test_two_runs_give_identical_numbers(tmp_path, monkeypatch, fast):
    """Everything is seeded, so a second run reproduces every number but the timings (Problem 7).

    With --fast, the training rows are also subsampled (with the seed) before anything is fitted.
    """
    monkeypatch.setattr(run_all, "DecisionTree", StandInTree)
    monkeypatch.setattr(run_all, "PegasosSVM", StandInSVM)
    runs = [tmp_path / "first", tmp_path / "second"]
    for out in runs:
        _run(out, monkeypatch, "--no-figures", *(["--fast"] if fast else []))
    first, second = (json.loads((out / "results" / "metrics.json").read_text(encoding="utf-8"))
                     for out in runs)
    assert first["scaling"] and first["task_b"][0]["cv"]["fit_seconds"], "timings were recorded"
    assert first["data"]["train"] == (200 if fast else 294)
    assert _without_timings(first) == _without_timings(second)
    for name in ("headline.tex", "task_a.tex", "task_b.tex"):
        assert (runs[0] / "tables" / name).read_text(encoding="utf-8") == \
            (runs[1] / "tables" / name).read_text(encoding="utf-8")


def test_a_half_done_model_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "DecisionTree", HalfDoneTree)
    report = _run(tmp_path, monkeypatch, "--no-figures", "--no-scaling")
    tree_b = {r["model"]: r for r in report["task_b"]}[run_all.OUR_TREE]
    assert tree_b["status"] == "ok", "scored on its predictions"
    assert "roc_auc" not in tree_b["test"], "no scores yet, so no AUC"


def test_a_model_whose_predict_is_a_stub_is_skipped(tmp_path, monkeypatch):
    class PredictStub(StandInTree):
        def predict(self, X):
            raise NotImplementedError("TODO: walk the tree")

    monkeypatch.setattr(run_all, "DecisionTree", PredictStub)
    report = _run(tmp_path, monkeypatch, "--no-figures", "--no-scaling")
    tree_a = {r["model"]: r for r in report["task_a"]}[run_all.OUR_TREE]
    assert tree_a["status"] == "not implemented yet" and "walk the tree" in tree_a["reason"]


def test_fast_run_keeps_the_real_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_all, "load_data", lambda path, fast=False: fake_data())
    monkeypatch.setattr(run_all, "eda_figures", lambda path, out_dir: None)
    run_all.main(["--fast", "--no-figures", "--no-scaling", "--boot", "20"])
    assert (tmp_path / "results" / "fast" / "results.md").exists()
    assert not (tmp_path / "results" / "results.md").exists()
    assert "SMOKE RUN" in (tmp_path / "report" / "tables" / "fast" / "task_a.tex").read_text(
        encoding="utf-8")


def test_a_run_without_the_tree_study_keeps_the_real_outputs(tmp_path, monkeypatch):
    """--no-tree-study scores other trees than the report's, so it writes to its own folders."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_all, "load_data", lambda path, fast=False: fake_data())
    monkeypatch.setattr(run_all, "eda_figures", lambda path, out_dir: None)
    report = run_all.main(["--no-tree-study", "--no-figures", "--no-scaling", "--folds", "0",
                           "--boot", "20"])
    assert report["tree_study"] is None and report["config"]["tree_settings_from"] == "fixed"
    assert (tmp_path / "results" / "no_tree_study" / "results.md").exists()
    assert (tmp_path / "report" / "tables" / "no_tree_study" / "task_a.tex").exists()
    assert not (tmp_path / "results" / "results.md").exists()
    assert not (tmp_path / "report" / "tables" / "task_a.tex").exists()
    assert not (tmp_path / "report" / "tables" / "no_tree_study" / "tree_study.tex").exists()


def test_a_run_without_the_svm_study_keeps_the_real_outputs(tmp_path, monkeypatch):
    """--no-svm-study scores other SVMs than the report's, so it writes to its own folders."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_all, "load_data", lambda path, fast=False: fake_data())
    monkeypatch.setattr(run_all, "eda_figures", lambda path, out_dir: None)
    report = run_all.main(["--no-svm-study", "--no-figures", "--no-scaling", "--folds", "0",
                           "--boot", "20"])
    assert report["svm_study"] is None and report["config"]["svm_settings_from"] == "fixed"
    assert report["config"]["svm"] == run_all.SVM
    tables = tmp_path / "report" / "tables" / "no_svm_study"
    assert (tables / "task_b.tex").exists() and not (tables / "svm_study.tex").exists()
    assert not (tmp_path / "report" / "tables" / "task_b.tex").exists()
    numbers = (tables / "numbers.tex").read_text(encoding="utf-8")
    assert r"\defval{SvmBFOne}" in numbers and r"\defval{SvmLambda}" not in numbers
    run_all.main(["--no-svm-study", "--no-tree-study", "--no-figures", "--no-scaling",
                  "--folds", "0", "--boot", "20"])
    assert (tmp_path / "results" / "no_tree_study-no_svm_study" / "results.md").exists()


def test_cross_split_duplicates_count_validation_and_test_twins():
    def frame(rows):
        return pd.DataFrame(rows, columns=["category", "price", "area_m2"])

    dfs = {"train": frame([("flat", 100, 50), ("flat", 200, 60)]),
           "val": frame([("flat", 100, 50), ("house", 300, 90)]),
           "test": frame([("house", 300, 90), ("flat", 400, 70), ("flat", 200, 61)])}
    dup = run_all.cross_split_duplicates(dfs)
    assert dup["key"] == ["category", "price", "area_m2"], "only the columns the data has"
    # val (flat, 100, 50) has a twin in train; (house, 300, 90) is in val and test
    assert (dup["holdout_rows"], dup["with_twin"], dup["with_twin_in_train"]) == (5, 3, 1)
    assert dup["share"] == pytest.approx(3 / 5)


def test_a_failing_eda_module_does_not_lose_the_results(tmp_path, monkeypatch):
    def broken(data_path, out_dir):
        raise FileNotFoundError("data/house_sale.csv")

    fake_eda = types.ModuleType("src.eda")
    fake_eda.__spec__ = importlib.machinery.ModuleSpec("src.eda", None)
    fake_eda.make_eda_figures = broken
    monkeypatch.setitem(sys.modules, "src.eda", fake_eda)
    monkeypatch.setattr(src, "eda", fake_eda, raising=False)
    monkeypatch.setattr(run_all, "load_data", lambda path, fast=False: fake_data())
    report = run_all.main(["--folds", "0", "--boot", "20", "--no-scaling",
                           "--results-dir", str(tmp_path / "results"),
                           "--tables-dir", str(tmp_path / "tables"),
                           "--figures-dir", str(tmp_path / "figures")])
    assert report["eda"]["status"] == "failed" and "house_sale.csv" in report["eda"]["reason"]
    saved = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["eda"]["status"] == "failed", "the run's own results are still written"
    assert (tmp_path / "figures" / "eval_confusion_matrices.pdf").exists()


def test_cv_and_timing_fits_get_the_training_matrix_layout():
    """CV folds and scaling subsets keep the column-major layout of the full split,
    so their fit times are comparable with the main fit (evaluate.take_rows)."""
    seen = []

    class Recorder:
        def fit(self, X, y):
            seen.append(X.flags.f_contiguous)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    data = fake_data()
    data["X"] = {part: np.asfortranarray(X) for part, X in data["X"].items()}
    run_all.scaling_study("A", [run_all.ModelSpec("recorder", Recorder)], data, {"recorder"},
                          repeats=1)
    run_all.evaluate.cross_validate(Recorder, data["X"]["train"], data["log_price"]["train"],
                                    lambda m, X, y: {"zero": 0.0}, k=3)
    assert len(seen) == 5 + 3 and all(seen)
    assert run_all.subsample_train(data, 100)["X"]["train"].flags.f_contiguous


def test_bad_fold_count_is_rejected(monkeypatch):
    monkeypatch.setattr(run_all, "load_data", lambda path, fast=False: fake_data())
    with pytest.raises(SystemExit):
        run_all.main(["--folds", "1"])


def test_subsample_train_keeps_validation_and_test(monkeypatch):
    data = fake_data()
    small = run_all.subsample_train(data, 100)
    assert len(small["X"]["train"]) == len(small["df"]["train"]) == 100
    assert len(small["X"]["test"]) == len(data["X"]["test"])
    assert len(data["X"]["train"]) == 294, "the original dict is not modified"


def _fake_data_with_land_area(path=None, fast=False):
    """fake_data plus data_prep's land_area_sot column (0 = no separate land),
    so the tree study trains Task A on log(price per m2) as on the real data."""
    data = fake_data()
    data["X"] = {p: np.hstack([X, np.zeros((len(X), 1))]) for p, X in data["X"].items()}
    data["feature_names"] = data["feature_names"] + ["land_area_sot"]
    return data


@needs_tree_study
def test_headline_trees_are_the_tree_studys_trees(tmp_path, monkeypatch):
    """The tables score the tree the study tuned on validation, on the same target,
    so the report has one Task A number for our tree, not two."""
    monkeypatch.setattr(run_all, "load_data", _fake_data_with_land_area)
    monkeypatch.setattr(run_all, "eda_figures", lambda path, out_dir: None)
    report = run_all.main(["--folds", "0", "--boot", "20", "--no-scaling", "--no-figures",
                           "--results-dir", str(tmp_path / "results"),
                           "--tables-dir", str(tmp_path / "tables"),
                           "--figures-dir", str(tmp_path / "figures")])
    study, config = report["tree_study"], report["config"]
    assert study["task_a"]["target"] == config["tree_task_a_target"] == "log(price per m2 / sot)"
    assert config["tree_regression"] == study["task_a"]["settings"]
    assert config["tree_classification"] == study["task_b"]["settings"]
    tree_a = {r["model"]: r for r in report["task_a"]}[run_all.OUR_TREE]
    assert tree_a["test"]["rmse_log"] == pytest.approx(study["task_a"]["test_ours"])
    assert "added back" in (tmp_path / "tables" / "task_a.tex").read_text(encoding="utf-8")
    assert not (tmp_path / "figures").exists(), "--no-figures writes no tree-study figures"


@needs_tree_study
def test_per_size_target_predicts_log_price():
    from src.baselines import PerSizeTarget

    data = _fake_data_with_land_area()
    names, X, y = data["feature_names"], data["X"], data["log_price"]
    model = PerSizeTarget(DecisionTreeRegressor(max_depth=3, random_state=0), names).fit(
        X["train"], y["train"])
    log_area = np.log(X["test"][:, names.index("area_m2")])
    inner = model.model.predict(X["test"])
    np.testing.assert_allclose(model.predict(X["test"]), inner + log_area)
    # R2 on log(price) is what the tables report; on the per-m2 scale it is lower
    r2 = run_all.evaluate.r2
    assert r2(y["test"], model.predict(X["test"])) > r2(y["test"] - log_area, inner)
    assert model.get_depth() == 3 and model.feature_importances_.shape == (len(names),)


def test_worst_rows_keep_fractional_land_and_mark_missing_areas():
    worst = pd.DataFrame({"category": ["Torpaq", "Yeni tikili"],
                          "location": ["Nərimanov r.", "Sahil m."],
                          "area_m2": [0.0, 75.4], "land_area_sot": [0.4, 0.0],
                          "price": [23_000.0, 75_900.0], "predicted": [50_176.0, 146_246.0],
                          "error_pct": [118.2, 92.7]})
    rows = run_all.worst_rows({"task_a": {run_all.OUR_TREE: {"worst": worst}}})
    assert rows[0]["land_area_sot"] == "0.4" and rows[0]["area_m2"] is None
    assert rows[1]["area_m2"] == 75 and rows[1]["land_area_sot"] is None
    assert [r["error"] for r in rows] == ["+118%", "+93%"]
