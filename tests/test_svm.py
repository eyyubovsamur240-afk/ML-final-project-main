"""Our own tests for the Pegasos SVM and the random-Fourier-feature extension (Problem 3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.rff import RandomFourierFeatures, RFFSVM, make_rings
from src.svm import SCHEDULES, PegasosSVM


def blobs(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2, 1, (n // 2, 2)), rng.normal(2, 1, (n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_hand_computed_scores_and_predictions():
    svm = PegasosSVM()
    _, svm.classes_ = svm._to_pm1(np.array([0, 1]))
    svm.w, svm.b = np.array([2.0, 1.0]), -4.0
    X = np.array([[3, 1], [1, 1], [2, 0]])
    assert np.allclose(svm.decision_function(X), [3.0, -1.0, 0.0])
    assert list(svm.predict(X)) == [1, 0, 1]          # score 0 -> positive class


def test_objective_by_hand():
    svm = PegasosSVM(lambda_=0.1)
    svm.w, svm.b = np.array([2.0, 1.0]), -4.0
    X = np.array([[3, 1], [1, 1], [2, 0], [2, 1]])
    # hinges 0, 0, 1, 2 -> mean 0.75; lambda/2 ||w||^2 = 0.05 * 5 = 0.25
    assert svm._objective(X, np.array([1, -1, 1, -1])) == pytest.approx(1.0)


def test_labels_must_have_two_classes():
    with pytest.raises(ValueError):
        PegasosSVM._to_pm1(np.array([1, 1, 1]))


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_schedule_learns_blobs_and_lowers_the_objective(schedule):
    X, y = blobs()
    svm = PegasosSVM(lambda_=0.01, n_epochs=50, schedule=schedule).fit(X, y)
    assert (svm.predict(X) == y).mean() >= 0.97
    assert svm.history_[-1] < svm.history_[0]


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        PegasosSVM(schedule="fast").fit(*blobs())


def test_n_epochs_sets_the_number_of_steps():
    X, y = blobs(n=60)
    svm = PegasosSVM(lambda_=0.01, n_epochs=3).fit(X, y)
    assert svm.n_iters_ == 3 * 60
    assert PegasosSVM(lambda_=0.01, n_iters=500).fit(X, y).n_iters_ == 500


def test_bigger_lambda_gives_a_wider_margin():
    X, y = blobs()
    widths = [PegasosSVM(lambda_=lam, n_epochs=30).fit(X, y).margin_width()
              for lam in (1e-3, 1e-2, 1e-1, 1.0)]
    assert widths == sorted(widths)
    assert widths[0] == pytest.approx(2 / PegasosSVM(lambda_=1e-3, n_epochs=30)
                                      .fit(X, y).weight_norm())


def test_support_vectors_are_exactly_the_rows_inside_the_street():
    X, y = blobs()
    svm = PegasosSVM(lambda_=0.01, n_epochs=30).fit(X, y)
    y_pm = np.where(y == svm.classes_[1], 1, -1)
    margins = y_pm * svm.decision_function(X)
    inside = np.zeros(len(y), dtype=bool)
    inside[svm.support_] = True
    assert inside.any()
    assert np.all(margins[inside] <= 1) and np.all(margins[~inside] > 1)


def test_rff_dot_products_approximate_the_rbf_kernel():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    Z = RandomFourierFeatures(n_components=5000, gamma=0.5, random_state=0).fit_transform(X)
    K = np.exp(-0.5 * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    assert np.abs(Z @ Z.T - K).max() < 0.1


def test_rff_is_deterministic_for_a_fixed_seed():
    X = np.random.default_rng(0).normal(size=(10, 4))
    a = RandomFourierFeatures(50, 1.0, random_state=3).fit_transform(X)
    b = RandomFourierFeatures(50, 1.0, random_state=3).fit_transform(X)
    assert a.shape == (10, 50) and np.array_equal(a, b)


def test_rff_svm_separates_rings_a_linear_svm_cannot():
    X, y = make_rings(n=400, seed=0)
    linear = PegasosSVM(lambda_=1e-3, n_epochs=20).fit(X, y)
    rbf = RFFSVM(n_components=300, gamma=1.0, lambda_=1e-3, n_epochs=20).fit(X, y)
    assert (linear.predict(X) == y).mean() < 0.8
    assert (rbf.predict(X) == y).mean() >= 0.95
    assert set(np.unique(rbf.predict(X))) <= {0, 1}
