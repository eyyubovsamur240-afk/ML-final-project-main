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
    """Full-batch hinge sub-gradient, our SVM's constructor signature (test only)."""

    def __init__(self, lambda_=1e-4, n_iters=1_000, random_state=42):
        self.lambda_, self.n_iters = lambda_, min(n_iters, 300)

    def fit(self, X, y):
        y_pm = np.where(np.asarray(y) == 1, 1.0, -1.0)
        Xa = np.hstack([X, np.ones((len(X), 1))])
        w = np.zeros(Xa.shape[1])
        for t in range(1, self.n_iters + 1):
            active = y_pm * (Xa @ w) < 1
            grad = self.lambda_ * w - (y_pm[active, None] * Xa[active]).sum(0) / len(X)
            w -= grad / (self.lambda_ * t + 10)
        self.w, self.b = w[:-1], w[-1]
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def _run(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(run_all, "load_data", lambda path, fast=False: fake_data())
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
    assert (tmp_path / "tables" / "task_a.tex").exists()


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

    saved = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["data"]["train"] == 294 and saved["config"]["folds"] == 3
    tex = (tmp_path / "tables" / "task_b.tex").read_text(encoding="utf-8")
    assert r"\label{tab:task-b}" in tex and "Ours: SVM (Pegasos)" in tex
    assert r"\begin{table*}" in tex and "3-fold" in tex
    figures = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {"eval_pred_vs_actual.pdf", "eval_confusion_matrices.pdf", "eval_roc_curves.pdf",
            "eval_svm_weights.pdf", "eval_fit_time_scaling.pdf",
            "eval_tree_impurity_importance.pdf"} <= figures


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
