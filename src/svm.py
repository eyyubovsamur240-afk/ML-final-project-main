"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate eta_t:

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

Design decisions (explained in the report):
  * The bias is FOLDED into w: every x gets an extra constant 1, so the last
    entry of the weight vector plays the role of b. A separate, unregularised
    b update took steps of size eta_t (huge at the start) and drifted far
    from its best value on a toy problem; folding it keeps b stable. Cost: b
    is shrunk a little too, harmless on standardized features (best b ~ 0).
  * Three learning-rate schedules (Problem 3b), chosen with `schedule`:
      "pegasos"  eta_t = 1 / (lambda * t)      (the original Pegasos schedule)
      "invsqrt"  eta_t = eta0 / sqrt(t)
      "constant" eta_t = eta0
  * Training length can be given as steps (n_iters) or as epochs (n_epochs):
    one epoch = n_samples steps, i.e. on average every row is seen once.
  * After fit, the margin is reported (Problem 3c): ||w||, the street width
    2/||w||, and the indices of the support vectors (rows with y*f <= 1).

The non-linear extension (random Fourier features) lives in src/rff.py.
scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np

SCHEDULES = ("pegasos", "invsqrt", "constant")


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
        n_epochs: float | None = None,  # if given, overrides n_iters: n_epochs * n_samples steps
        schedule: str = "pegasos",      # learning-rate schedule, one of SCHEDULES
        eta0: float = 0.1,              # starting step size for "invsqrt" and "constant"
        record_every: int = 100,        # store the objective J every this many steps
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.schedule = schedule
        self.eta0 = eta0
        self.record_every = record_every
        self.w = None              # weight vector (without the bias)
        self.b = 0.0               # bias
        self.classes_ = None       # original labels: classes_[0] -> -1, classes_[1] -> +1
        self.history_ = None       # objective J, recorded every `record_every` steps
        self.n_iters_ = None       # number of steps actually taken
        self.support_ = None       # indices of the training support vectors (y*f <= 1)

    @staticmethod
    def _to_pm1(y):
        """Map {0,1} (or any 2-class labels) to {-1, +1}. Store the mapping."""
        y = np.asarray(y)
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(f"Need exactly 2 classes, got {labels}")
        updated_y = np.where(y == labels[0], -1, 1)
        return updated_y, labels

    def _learning_rate(self, t):
        """Step size eta_t at step t (t starts at 1, so 1/(lambda*t) never divides by 0)."""
        if self.schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.schedule == "invsqrt":
            return self.eta0 / np.sqrt(t)
        return self.eta0   # "constant"

    def fit(self, X, y):
        """
        Run Pegasos. Sample one example per step, use the chosen eta_t,
        and apply the update above. Return self.
        """
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        X_aug = np.hstack([X, np.ones((n_samples, 1))])   # folded bias: last column is 1
        y_pm, self.classes_ = self._to_pm1(y)
        rng = np.random.default_rng(self.random_state)
        w_aug = np.zeros(n_features + 1)                    # (w_1, ..., w_d, b)
        self.history_ = []

        if self.n_epochs is not None:
            self.n_iters_ = max(1, int(round(self.n_epochs * n_samples)))
        else:
            self.n_iters_ = self.n_iters

        for t in range(1, self.n_iters_ + 1):
            learning_rate = self._learning_rate(t)
            i = rng.integers(n_samples)
            x_i = X_aug[i]
            y_i = y_pm[i]

            score = w_aug @ x_i
            margin = y_i * score
            if margin < 1:      # inside the street or wrong side: shrink + push
                w_aug = (1 - learning_rate * self.lambda_) * w_aug + learning_rate * y_i * x_i
            else:               # safely outside the street: only shrink
                w_aug = (1 - learning_rate * self.lambda_) * w_aug

            if t % self.record_every == 0:
                self.w, self.b = w_aug[:-1], w_aug[-1]
                self.history_.append(self._objective(X, y_pm))

        self.w, self.b = w_aug[:-1], w_aug[-1]

        # Support vectors (Problem 3c): training rows on or inside the street.
        train_margins = y_pm * self.decision_function(X)
        self.support_ = np.flatnonzero(train_margins <= 1)
        return self

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        signed_margins = np.dot(X, self.w) + self.b
        return signed_margins

    def _objective(self, X, y_pm):
        """J(w, b) = lambda/2 ||w||^2 + mean hinge, for labels y_pm in {-1, +1}."""
        scores = self.decision_function(X)
        all_margins = np.multiply(y_pm, scores)
        reg = (self.lambda_ / 2) * np.dot(self.w, self.w)
        hinge = np.maximum(0, 1 - all_margins)
        return reg + np.mean(hinge)

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        # score == 0 (exactly on the boundary) -> positive class, same as sklearn
        return np.where(self.decision_function(X) < 0, self.classes_[0], self.classes_[1])

    # --- margin report (Problem 3c) ----------------------------------------------
    def weight_norm(self) -> float:
        """||w|| (without the bias: b only shifts the line, it does not change the street)."""
        return float(np.linalg.norm(self.w))

    def margin_width(self) -> float:
        """Street width 2 / ||w||; infinite if w = 0 (every point gets the same score)."""
        norm = self.weight_norm()
        return float("inf") if norm == 0 else 2.0 / norm


if __name__ == "__main__":
    # Toy test: two separated clouds; any working linear SVM gets ~100% here.
    rng = np.random.default_rng(0)
    X_toy = np.vstack([rng.normal(-2, 1, (50, 2)), rng.normal(2, 1, (50, 2))])
    y_toy = np.array([0] * 50 + [1] * 50)
    for sched in SCHEDULES:
        m = PegasosSVM(lambda_=0.01, n_epochs=50, schedule=sched).fit(X_toy, y_toy)
        print(f"{sched:9s} accuracy={(m.predict(X_toy) == y_toy).mean():.2f} "
              f"J {m.history_[0]:.4f} -> {m.history_[-1]:.4f}  "
              f"||w||={m.weight_norm():.3f} margin={m.margin_width():.3f} "
              f"support vectors={len(m.support_)}")
