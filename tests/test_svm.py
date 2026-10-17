"""Our Pegasos SVM: the update, the objective, mini-batches, averaging, schedules,
margin and support vectors (Problem 3), on small data where the answer is known."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from src.svm import PegasosSVM


def blobs(n=200, seed=0, shift=2.0):
    """Two Gaussian clouds, centred away from the origin so the bias matters."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-shift, 1, (n // 2, 2)), rng.normal(shift, 1, (n // 2, 2))]) + 3
    return X, np.repeat([0, 1], n // 2)


def linear_svc_objective(lambda_, X, y):
    """J of LinearSVC's solution: the same objective (bias regularised too)."""
    svc = LinearSVC(C=1 / (lambda_ * len(y)), loss="hinge", dual=True, max_iter=200_000,
                    tol=1e-8).fit(X, y)
    ref = PegasosSVM(lambda_=lambda_)
    ref.classes_, ref.w, ref.b = svc.classes_, svc.coef_.ravel(), float(svc.intercept_[0])
    return ref.objective(X, y), ref


def test_decision_function_predict_and_objective_by_hand():
    m = PegasosSVM(lambda_=0.1)
    y_pm, m.classes_ = m._to_pm1(np.array([0, 1, 1, 0]))
    assert y_pm.tolist() == [-1, 1, 1, -1] and m.classes_.tolist() == [0, 1]
    m.w, m.b = np.array([2.0, 1.0]), -4.0
    X = np.array([[3, 1], [1, 1], [2, 0]], dtype=float)
    assert m.decision_function(X).tolist() == [3.0, -1.0, 0.0]
    assert m.predict(X).tolist() == [1, 0, 1], "a score of 0 goes to the positive class"
    # J = 0.1/2 (4 + 1 + 16) + mean hinge of margins [3, 1, 0, -1] on y = [1, -1, 1, -1]
    X4 = np.array([[3, 1], [1, 1], [2, 0], [2, 1]], dtype=float)
    hinge = np.mean(np.maximum(0, 1 - np.array([1, -1, 1, -1]) * np.array([3, -1, 0, 1])))
    assert m.objective(X4, np.array([1, 0, 1, 0])) == pytest.approx(0.05 * 21 + hinge)


def test_one_row_steps_reproduce_the_first_version():
    """batch_size=1 with 1/(lambda t) is the original single-row Pegasos step."""
    X, y = blobs(100)
    m = PegasosSVM(lambda_=0.01, n_iters=5_000, random_state=42).fit(X, y)
    assert (m.predict(X) == y).mean() >= 0.97
    assert len(m.history_) == 50, "once per epoch of 100 single-row steps"
    assert m.history_[-1] < m.history_[0]


@pytest.mark.parametrize("schedule", ["pegasos", "offset", "constant"])
def test_minibatch_averaged_pegasos_reaches_the_linear_svc_optimum(schedule):
    X, y = blobs(400, seed=1, shift=1.0)  # overlapping clouds: a real soft margin
    lam = 1e-2
    best, _ = linear_svc_objective(lam, X, y)
    m = PegasosSVM(lambda_=lam, n_epochs=200, batch_size=16, average=True, schedule=schedule,
                   random_state=0).fit(X, y)
    assert m.n_iter_ == 200 * 25, "an epoch is ceil(n / batch) steps"
    assert m.history_[-1] == pytest.approx(m.objective(X, y)), "history ends at the returned w"
    assert m.history_[-1] / best - 1 < 0.01, "within 1% of LinearSVC's objective"


def test_averaging_lowers_the_objective_of_the_last_iterate():
    X, y = blobs(400, seed=2, shift=1.0)
    common = dict(lambda_=1e-3, n_epochs=20, batch_size=8, schedule="constant", eta0=0.5,
                  random_state=3)
    last = PegasosSVM(average=False, **common).fit(X, y).history_[-1]
    averaged = PegasosSVM(average=True, **common).fit(X, y).history_[-1]
    assert averaged < last


def test_margin_and_support_vectors():
    X, y = blobs(400, seed=4, shift=1.0)
    m = PegasosSVM(lambda_=1e-2, n_epochs=100, batch_size=16, average=True,
                   schedule="offset").fit(X, y)
    assert m.w_norm_ == pytest.approx(np.linalg.norm(m.w))
    assert m.margin_ == pytest.approx(2 / np.linalg.norm(m.w))
    y_pm = np.where(y == 1, 1, -1)
    inside = y_pm * m.decision_function(X) <= 1
    assert m.support_.tolist() == np.flatnonzero(inside).tolist()
    assert m.n_support_ == inside.sum() > 0
    # a larger lambda shrinks w: a wider margin and more rows on or inside it
    wide = PegasosSVM(lambda_=1.0, n_epochs=100, batch_size=16, average=True,
                      schedule="offset").fit(X, y)
    assert wide.margin_ > m.margin_ and wide.n_support_ > m.n_support_


def test_the_offset_schedule_starts_at_eta0_and_decays_like_pegasos():
    m = PegasosSVM(lambda_=0.1, schedule="offset", eta0=0.5)
    m.eta0_ = 0.5
    assert m._learning_rate(1) == pytest.approx(1 / (0.1 + 2), rel=1e-12)  # ~ eta0 at t = 1
    assert m._learning_rate(10**6) * 0.1 * 10**6 == pytest.approx(1, rel=1e-4)
    assert PegasosSVM(lambda_=0.1)._learning_rate(5) == pytest.approx(2.0)
    X, y = blobs(50)
    fitted = PegasosSVM(lambda_=0.1, n_epochs=1, schedule="offset").fit(X, y)
    assert fitted.eta0_ == pytest.approx(1 / np.mean((X ** 2).sum(1))), "default eta0"
    with pytest.raises(ValueError):
        PegasosSVM(schedule="adam").fit(X, y)


def test_float32_input_is_not_copied_to_float64_and_matches():
    X, y = blobs(200, seed=5)
    a = PegasosSVM(lambda_=1e-2, n_epochs=5, batch_size=8, random_state=1).fit(X, y)
    b = PegasosSVM(lambda_=1e-2, n_epochs=5, batch_size=8, random_state=1).fit(
        X.astype(np.float32), y)
    np.testing.assert_allclose(a.w, b.w, rtol=1e-4, atol=1e-5)
    assert b.decision_function(X.astype(np.float32)).dtype == np.float32


def test_same_seed_same_model_other_seed_other_model():
    X, y = blobs(200, seed=6, shift=1.0)
    fit = lambda seed: PegasosSVM(lambda_=1e-2, n_epochs=3, batch_size=4,  # noqa: E731
                                  random_state=seed).fit(X, y)
    assert np.array_equal(fit(7).w, fit(7).w)
    assert not np.array_equal(fit(7).w, fit(8).w)
