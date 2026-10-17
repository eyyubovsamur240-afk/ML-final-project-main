"""Random Fourier features approximate the RBF kernel, and the RFF SVM separates
data no linear SVM can (Problem 3d)."""

from __future__ import annotations

import numpy as np
import pytest

from src.rff import RandomFourierFeatures, RFFSVM, kernel_approximation_error, rbf_kernel
from src.svm import PegasosSVM


def rings(n=600, seed=0):
    """An inner disc against a surrounding ring: not linearly separable."""
    rng = np.random.default_rng(seed)
    radius = np.r_[rng.uniform(0, 1, n // 2), rng.uniform(2, 3, n // 2)]
    angle = rng.uniform(0, 2 * np.pi, n)
    X = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    return X, np.repeat([0, 1], n // 2)


def test_features_approximate_the_rbf_kernel_better_with_more_of_them():
    X = np.random.default_rng(1).normal(size=(200, 5))
    errors = [kernel_approximation_error(X, gamma=0.2, n_components=D, n_rows=200)
              for D in (16, 256, 4096)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.02
    z = RandomFourierFeatures(4096, gamma=0.2, random_state=0).fit_transform(X)
    np.testing.assert_allclose(z @ z.T, rbf_kernel(X, X, 0.2), atol=0.1)


def test_feature_map_is_seeded_float32_and_chunked():
    X = np.random.default_rng(2).normal(size=(50, 3))
    a = RandomFourierFeatures(64, 0.5, random_state=3).fit(X)
    b = RandomFourierFeatures(64, 0.5, random_state=3, chunk_rows=7).fit(X)
    assert a.transform(X).dtype == np.float32 and a.transform(X).shape == (50, 64)
    np.testing.assert_allclose(a.transform(X), b.transform(X), rtol=1e-6)
    assert np.all(np.abs(a.transform(X)) <= np.sqrt(2 / 64) + 1e-6)
    assert np.var(a.W_) == pytest.approx(2 * 0.5, rel=0.1), "W ~ N(0, 2 gamma)"


def test_rff_svm_separates_rings_a_linear_svm_cannot():
    X, y = rings()
    linear = PegasosSVM(lambda_=1e-3, n_epochs=50, batch_size=16, average=True,
                        schedule="offset").fit(X, y)
    rff = RFFSVM(gamma=0.5, n_components=256, lambda_=1e-4, n_epochs=50, batch_size=16,
                 random_state=0).fit(X, y)
    assert (linear.predict(X) == y).mean() < 0.75
    assert (rff.predict(X) == y).mean() > 0.95
    assert rff.decision_function(X).shape == (len(y),)
    # the linear SVM's attributes come through, in the space of the random features
    assert rff.w.shape == (256,) and rff.margin_ > 0 and rff.n_support_ < len(y)
    assert rff.history_[-1] < rff.history_[0]
    assert rff.objective(X, y) == pytest.approx(rff.history_[-1], rel=1e-5)


def test_rff_svm_attributes_before_fit_raise_attribute_error():
    model = RFFSVM()
    assert not hasattr(model, "margin_")
    assert getattr(model, "w", None) is None
