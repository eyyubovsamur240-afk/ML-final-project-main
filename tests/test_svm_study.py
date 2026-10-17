"""The SVM study runs end to end on small synthetic data and picks from its grids."""

from __future__ import annotations

import os

import numpy as np

from src import svm_study
from src.rff import make_rings


def test_study_runs_and_picks_settings_from_the_grids(tmp_path):
    X, y = make_rings(n=300, seed=1)
    order = np.random.default_rng(0).permutation(len(y))
    X, y = X[order], y[order]
    res = svm_study.run(X[:200], y[:200], X[200:], y[200:], fig_dir=str(tmp_path),
                        lambdas=[1e-3, 1e-2, 1e-1], gammas=[0.3, 1.0], epochs=[1, 5],
                        n_epochs=5, n_components=100, verbose=False)
    assert res["best_lambda"] in (1e-3, 1e-2, 1e-1)
    assert res["best_gamma"] in (0.3, 1.0)
    assert len(res["lambda_sweep"]) == 3 and len(res["rff_sweep"]) == 2
    assert set(res["schedules"]) == {"pegasos", "invsqrt", "constant"}
    margins = [r["margin"] for r in res["lambda_sweep"]]
    assert margins == sorted(margins)           # bigger lambda -> wider street
    assert max(r["val_f1"] for r in res["rff_sweep"]) > max(
        r["val_f1"] for r in res["lambda_sweep"])   # rings need the non-linear model
    for path in res["figures"]:
        assert os.path.exists(path)


def test_pick_best_breaks_ties_towards_the_simpler_model():
    rows = [{"lambda": 0.01, "val_f1": 0.8}, {"lambda": 0.1, "val_f1": 0.8}]
    assert svm_study.pick_best(rows, "lambda") == 0.1
    rows = [{"gamma": 0.1, "val_f1": 0.8}, {"gamma": 1.0, "val_f1": 0.8}]
    assert svm_study.pick_best(rows, "gamma", prefer_larger=False) == 0.1
