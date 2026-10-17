"""The SVM study (Problem 4b) on a small synthetic problem: the selection rules, the
convergence check against LinearSVC and the figures."""

from __future__ import annotations

import numpy as np
import pytest

from src import svm_study


@pytest.fixture
def small_grids(monkeypatch):
    monkeypatch.setattr(svm_study, "LAMBDAS", [1e-3, 1e-2, 1e-1])
    monkeypatch.setattr(svm_study, "EPOCHS", [2, 20])
    monkeypatch.setattr(svm_study, "SWEEP_EPOCHS", 20)
    monkeypatch.setattr(svm_study, "FIRST_VERSION", {**svm_study.FIRST_VERSION, "n_iters": 400})
    monkeypatch.setattr(svm_study, "RFF_GAMMAS", [0.1, 1.0])
    monkeypatch.setattr(svm_study, "RFF_LAMBDAS", [1e-4, 1e-3])
    monkeypatch.setattr(svm_study, "RFF_COMPONENTS", 128)
    monkeypatch.setattr(svm_study, "RFF_COMPONENT_SWEEP", [16, 128])


def circle(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * [1, 1, 5] + [0, 0, 10]   # a column far from 0, wide
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 + 0.2 * rng.normal(size=n) > 1.4).astype(int)
    return X, y


def test_study_picks_settings_on_validation_and_checks_convergence(small_grids, tmp_path):
    X_tr, y_tr = circle(400, 0)
    X_val, y_val = circle(200, 1)
    s = svm_study.run(X_tr, y_tr, X_val, y_val, fig_dir=str(tmp_path))

    sweep = s["linear"]["lambda_sweep"]
    best = max(r["val_f1"] for r in sweep)
    assert s["linear"]["settings"]["lambda_"] == max(r["lambda_"] for r in sweep
                                                     if r["val_f1"] == best), "ties: larger"
    margins = [r["margin"] for r in sorted(sweep, key=lambda r: r["lambda_"])]
    assert margins[-1] > margins[0], "a larger lambda gives a wider margin"

    conv = s["convergence"]
    averaged = {r["epochs"]: r for r in conv["rows"] if r["average"]}
    within = [e for e, r in averaged.items() if r["gap"] <= conv["tolerance"]]
    assert conv["chosen_epochs"] == (min(within) if within else max(averaged))
    assert s["linear"]["settings"]["n_epochs"] == conv["chosen_epochs"]
    assert averaged[20]["gap"] < averaged[2]["gap"], "more epochs, closer to LinearSVC"
    first = next(r for r in conv["rows"] if r["batch_size"] == 1)
    assert first["schedule"] == "pegasos" and first["n_steps"] == 400
    assert {r["schedule"] for r in s["schedules"]} == {"pegasos", "offset", "constant"}

    rff = s["rff"]
    best = max(r["val_f1"] for r in rff["grid"])
    ties = [r for r in rff["grid"] if r["val_f1"] == best]
    top = max(ties, key=lambda r: (r["lambda_"], -r["gamma"]))
    assert (rff["settings"]["gamma"], rff["settings"]["lambda_"]) == (top["gamma"], top["lambda_"])
    assert rff["validation"]["f1"] > s["linear"]["validation"]["f1"], "the circle is non-linear"
    errors = [r["kernel_error"] for r in rff["components"]]
    assert errors[0] > errors[-1], "more features, a closer kernel"

    assert {p.name for p in tmp_path.iterdir()} == {
        "svm_lambda.png", "svm_objective.png", "svm_rff_grid.png", "svm_rff_components.png"}


def test_study_never_sees_the_test_split():
    import inspect

    params = list(inspect.signature(svm_study.run).parameters)
    assert params == ["X_tr", "y_tr", "X_val", "y_val", "fig_dir"]
