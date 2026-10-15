"""run_all.py end to end on a small synthetic dataset (no Kaggle file needed).

The data loader is replaced by a fake one with the same shape as
data_prep.load_splits. Two cases: with the team's real tree and SVM (which
may still be TODO stubs, then they must be reported as skipped), and with
small stand-in models, so every code path for "our" models is exercised now.
"""

from __future__ import annotations

import importlib.machinery
import json
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

    def __init__(self, lambda_=1e-4, n_iters=1_000, random_state=42):
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
    assert list(headline)[-1] == run_all.TRIVIAL and len(headline) == 6, "each model once"
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
    assert tex.count("& n/a &") == 3, "the SVMs have no Task A result"
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
    assert tex.count("includes evaluating its objective") == 1
    assert "Ours: SVM (Pegasos) includes evaluating its objective on every training row 3 times" \
        in tex
    figures = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {"eval_pred_vs_actual.pdf", "eval_confusion_matrices.pdf", "eval_roc_curves.pdf",
            "eval_svm_weights.pdf", "eval_fit_time_scaling.pdf",
            "eval_tree_impurity_importance.pdf"} <= figures


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
