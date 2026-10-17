"""
rff.py — the non-linear SVM: random Fourier features + linear Pegasos (NumPy only).

An RBF-kernel SVM, k(x, x') = exp(-gamma ||x - x'||^2), approximated with random
Fourier features (Rahimi & Recht, 2007):

    z(x) = sqrt(2/D) * cos(W^T x + c),   W_jk ~ N(0, 2 gamma),  c_j ~ U(0, 2 pi)

so that E[z(x) . z(x')] = k(x, x'). A LINEAR SVM on z(x) is then an approximate
kernel SVM: f(x) = w . z(x) + b, trained with the same Pegasos code (svm.py).

Why this and not kernelised Pegasos: kernel Pegasos keeps one coefficient per
training row, and every step and every prediction evaluates the kernel against
all rows with a non-zero coefficient, O(n d) each. With tens of thousands of
listings that is far slower than O(D d) per row here, and the solver, its
convergence check and the margin stay those of the linear SVM.

Features must be standardized first (gamma is then on a known scale);
run_all wraps RFFSVM in baselines.StandardScaled like the linear SVM.
"""

from __future__ import annotations

import numpy as np

from .svm import PegasosSVM


class RandomFourierFeatures:
    """x -> z(x) with D = n_components features; the random draw is seeded."""

    def __init__(self, n_components: int = 1024, gamma: float = 1e-2, random_state: int = 42,
                 chunk_rows: int = 8192):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.chunk_rows = chunk_rows

    def fit(self, X, y=None):
        d = np.asarray(X).shape[1]
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0.0, np.sqrt(2 * self.gamma), size=(d, self.n_components))
        self.c_ = rng.uniform(0.0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        """z(X) in float32, which halves the memory and time of float64 and is
        accurate far below the sampling error of the approximation itself.
        Computed in chunks of rows, so no large intermediate is ever held."""
        X = np.asarray(X, dtype=np.float32)
        W, c = self.W_.astype(np.float32), self.c_.astype(np.float32)
        scale = np.float32(np.sqrt(2.0 / self.n_components))
        out = np.empty((len(X), self.n_components), dtype=np.float32)
        for start in range(0, len(X), self.chunk_rows):
            rows = slice(start, start + self.chunk_rows)
            block = X[rows] @ W
            block += c
            np.cos(block, out=block)
            block *= scale
            out[rows] = block
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def rbf_kernel(A, B, gamma):
    """The exact kernel exp(-gamma ||a - b||^2) between the rows of A and B."""
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def kernel_approximation_error(X, gamma, n_components, random_state=42, n_rows=500):
    """Mean |z(x) . z(x') - k(x, x')| over all pairs of n_rows rows of X (seeded sample)."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    rows = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    sample = X[rows]
    z = RandomFourierFeatures(n_components, gamma, random_state).fit_transform(sample)
    approx = z.astype(float) @ z.T.astype(float)
    return float(np.mean(np.abs(approx - rbf_kernel(sample, sample, gamma))))


class RFFSVM:
    """Random Fourier features followed by our linear Pegasos SVM."""

    def __init__(self, gamma: float = 1e-2, n_components: int = 1024, lambda_: float = 1e-5,
                 n_epochs: int = 50, batch_size: int = 32, average: bool = True,
                 schedule: str = "offset", eta0: float | None = None, random_state: int = 42):
        self.gamma = gamma
        self.n_components = n_components
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.average = average
        self.schedule = schedule
        self.eta0 = eta0
        self.random_state = random_state

    def fit(self, X, y):
        self.features_ = RandomFourierFeatures(self.n_components, self.gamma, self.random_state)
        Z = self.features_.fit_transform(X)
        self.svm_ = PegasosSVM(lambda_=self.lambda_, n_epochs=self.n_epochs,
                               batch_size=self.batch_size, average=self.average,
                               schedule=self.schedule, eta0=self.eta0,
                               random_state=self.random_state).fit(Z, y)
        return self

    def decision_function(self, X):
        return self.svm_.decision_function(self.features_.transform(X))

    def predict(self, X):
        return self.svm_.predict(self.features_.transform(X))

    def objective(self, X, y):
        return self.svm_.objective(self.features_.transform(X), y)

    def __getattr__(self, name):
        # w, b, classes_, history_, margin_, w_norm_, n_support_, ... of the linear SVM on z(x)
        if name in ("features_", "svm_") or name.startswith("__"):
            raise AttributeError(name)
        if "svm_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.svm_, name)
