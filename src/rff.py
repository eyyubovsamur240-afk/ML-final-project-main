"""
rff.py — random Fourier features (RFF) + our linear Pegasos SVM (Problem 3d).

Why: a linear SVM can only draw a straight line. Some classes are separated
by a curve (e.g. one class inside a ring of the other). The RBF kernel
    K(x, z) = exp(-gamma * ||x - z||^2)
measures similarity and lets an SVM draw curved boundaries, but kernel SVMs
need an n x n matrix. Random Fourier features (Rahimi & Recht, 2007) build an
explicit map z(x) with D new features such that
    z(x) . z(x')  ~=  K(x, x')
so a LINEAR SVM on z(x) behaves like an RBF-kernel SVM on x, and our Pegasos
code can be reused unchanged.

The map:
    W ~ Normal(0, 2*gamma) of shape (d, D)     (random frequencies)
    c ~ Uniform(0, 2*pi)   of shape (D,)       (random phases)
    z(x) = sqrt(2 / D) * cos(x W + c)

gamma = how quickly similarity drops with distance: large gamma -> very local,
wiggly boundaries (overfitting risk); small gamma -> smooth, almost linear.
D = number of features: larger D -> better kernel approximation, slower.

We chose RFF over kernelised Pegasos because it keeps the solver linear
(the same tested PegasosSVM) and costs O(n * D) memory instead of O(n^2).

NumPy only (golden rule): this file is checked by tools/check_golden_rule.py.
"""

from __future__ import annotations

import numpy as np

from .svm import PegasosSVM


class RandomFourierFeatures:
    """Map X (n, d) -> Z (n, D) whose dot products approximate the RBF kernel."""

    def __init__(self, n_components: int = 300, gamma: float = 1.0, random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.W_ = None   # random frequencies, shape (d, D)
        self.c_ = None   # random phases, shape (D,)

    def fit(self, X):
        """Draw the random frequencies and phases (they depend only on d, not on the rows)."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(n_features, self.n_components))
        self.c_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W_ + self.c_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RFFSVM:
    """Random Fourier features followed by our linear PegasosSVM: an approximate RBF SVM.

    Same fit / decision_function / predict API as PegasosSVM. Features should be
    standardized first (gamma is then on a comparable scale for every feature).
    """

    def __init__(self, n_components: int = 300, gamma: float = 1.0, lambda_: float = 1e-4,
                 n_iters: int = 20_000, random_state: int = 42, **svm_kwargs):
        self.n_components = n_components
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.svm_kwargs = svm_kwargs   # e.g. n_epochs, schedule, eta0 -> passed to PegasosSVM

    def fit(self, X, y):
        self.features_ = RandomFourierFeatures(self.n_components, self.gamma, self.random_state)
        Z = self.features_.fit_transform(X)
        self.svm_ = PegasosSVM(lambda_=self.lambda_, n_iters=self.n_iters,
                               random_state=self.random_state, **self.svm_kwargs)
        self.svm_.fit(Z, y)
        self.classes_ = self.svm_.classes_
        self.history_ = self.svm_.history_
        return self

    def decision_function(self, X):
        return self.svm_.decision_function(self.features_.transform(X))

    def predict(self, X):
        return self.svm_.predict(self.features_.transform(X))


def make_rings(n: int = 400, noise: float = 0.15, seed: int = 0):
    """Toy data no straight line can separate: class 1 is a ring around class 0."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.where(np.arange(n) < n // 2, 1.0, 2.5) + rng.normal(0, noise, n)
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = (np.arange(n) >= n // 2).astype(int)
    return X, y


if __name__ == "__main__":
    X, y = make_rings()
    linear = PegasosSVM(lambda_=1e-3, n_epochs=20).fit(X, y)
    rbf = RFFSVM(n_components=300, gamma=1.0, lambda_=1e-3, n_epochs=20).fit(X, y)
    print(f"rings, linear SVM accuracy: {(linear.predict(X) == y).mean():.2f}")
    print(f"rings, RFF SVM accuracy:    {(rbf.predict(X) == y).mean():.2f}")
