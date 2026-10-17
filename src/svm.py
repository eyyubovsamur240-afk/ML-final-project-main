"""
svm.py — a soft-margin linear SVM trained with Pegasos (NumPy only).

Pegasos minimises the regularised hinge-loss primal by stochastic sub-gradient
descent (Shalev-Shwartz et al., 2011):

    J(w, b) = (lambda/2) * (||w||^2 + b^2)
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. The bias gets the same shrinking step as w, so it
is regularised too, as in liblinear (which appends a constant feature): J is
then exactly what scikit-learn's LinearSVC minimises with C = 1/(lambda*n),
so LinearSVC's J is a reference optimum for the convergence check.

One step on a mini-batch B_t of B rows drawn at random, with learning rate eta_t:

    active = rows of B_t with y_i (w . x_i + b) < 1          (hinge active)
    w <- (1 - eta_t*lambda) w + (eta_t/B) * sum_{active} y_i x_i
    b <- (1 - eta_t*lambda) b + (eta_t/B) * sum_{active} y_i

With B = 1 this is the original Pegasos step. Learning-rate schedules:
  "pegasos"   eta_t = 1 / (lambda * t)                        (the paper's)
  "offset"    eta_t = 1 / (lambda * (t + t0)), t0 = 1/(lambda*eta0), so the
              first step is about eta0 instead of 1/lambda, which for a small
              lambda throws w far off and takes many steps to recover from
  "constant"  eta_t = eta0
eta0 defaults to 1/(mean ||x_i||^2) of the training rows: a step of that size on
one row changes its margin y_i f(x_i) by about 1 at most, on any feature scale.
With average=True the returned weights are the mean of the iterates over the
second half of the steps (suffix averaging, Rakhlin et al., 2012), which
removes most of the noise of the last few stochastic steps.

After fit: w_norm_ = ||w||, margin_ = 2/||w|| (the width of the margin band,
in the units of the features), support_ = training rows on or inside the
margin (y_i f(x_i) <= 1, misclassified rows included) and history_ = J on all
training rows, recorded every `record_every` steps (default once per epoch).

Features must be standardized before training (run_all wraps this model in
baselines.StandardScaled). The non-linear extension is in rff.py.
"""

from __future__ import annotations

import math

import numpy as np

SCHEDULES = ("pegasos", "offset", "constant")


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (C = 1/(lambda*n))
        n_iters: int = 20_000,     # number of sub-gradient steps (if n_epochs is None)
        random_state: int = 42,
        batch_size: int = 1,       # rows per step
        n_epochs: int | None = None,  # passes over the data; overrides n_iters
        schedule: str = "pegasos",
        eta0: float | None = None, # first step of "offset", every step of "constant";
                                   # None = 1 / (mean squared row norm of X)
        average: bool = False,     # suffix average over the second half of the steps
        record_every: int | None = None,  # steps between objective records (None = per epoch)
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.schedule = schedule
        self.eta0 = eta0
        self.average = average
        self.record_every = record_every
        self.w = None              # weight vector
        self.b = 0.0               # bias
        self.classes_=None
        self.history_=None

    @staticmethod
    def _to_pm1(y):
        """Map {0,1} (or any 2-class labels) to {-1, +1}. Store the mapping."""
        labels=np.unique(y)
        if len(labels)!=2:
            raise ValueError("Only two class labels are accepted")
        updated_y=np.where(y==labels[0],-1,1)
        return updated_y,labels

    def _learning_rate(self, t):
        if self.schedule == "pegasos":
            return 1 / (self.lambda_ * t)
        if self.schedule == "offset":
            return 1 / (self.lambda_ * t + 1 / self.eta0_)  # = 1/(lambda (t + t0))
        if self.schedule == "constant":
            return self.eta0_
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")

    def fit(self, X, y):
        """Run Pegasos on mini-batches of batch_size rows and return self."""
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        X=np.asarray(X)
        if X.dtype not in (np.float32, np.float64):
            X=X.astype(float)
        n_samples,n_features=X.shape
        y_pm,self.classes_=self._to_pm1(y)
        batch=min(self.batch_size,n_samples)
        self.steps_per_epoch_=math.ceil(n_samples/batch)
        if self.n_epochs is not None:
            n_steps=self.n_epochs*self.steps_per_epoch_
        else:
            n_steps=self.n_iters
        record_every=self.record_every or self.steps_per_epoch_
        # eta0 = 1/mean ||x||^2: a step of eta0 on one row changes its margin y f(x) by
        # about 1 at most, whatever the scale of the features (standardized or RFF).
        self.eta0_=self.eta0 if self.eta0 is not None else 1/float(np.mean(np.einsum("ij,ij->i",X,X)))
        avg_start=n_steps//2+1 if self.average else n_steps+1   # first averaged step

        rng=np.random.default_rng(self.random_state)
        y_pm=y_pm.astype(X.dtype)
        w=np.zeros(n_features)
        b=0.0
        w_sum=np.zeros(n_features)
        b_sum=0.0
        self.history_=[]

        for t in range(1,n_steps+1):
            if (t-1)%record_every==0:
                # draw the rows of the next record_every steps at once (same seeded stream)
                rows=rng.integers(n_samples,size=(min(record_every,n_steps-t+1),batch))
            idx=rows[(t-1)%record_every]
            learning_rate=self._learning_rate(t)
            X_b=X[idx]
            y_b=y_pm[idx]

            # w in X's dtype: a float32 batch (rff.py) times a float64 w would copy the batch
            active=y_b*(X_b@w.astype(X.dtype,copy=False)+b)<1
            shrink=1-learning_rate*self.lambda_
            w*=shrink
            b*=shrink
            if active.any():
                coef=np.where(active,y_b,0)*(learning_rate/batch)   # y_i eta/B on active rows
                w+=coef@X_b
                b+=float(coef.sum())

            if t>=avg_start:
                w_sum+=w
                b_sum+=b
            if t%record_every==0 or t==n_steps:
                # the objective of the weights fit() would return if it stopped here
                self._set_weights(w,b,w_sum,b_sum,t-avg_start+1)
                self.history_.append(self._objective(X,y_pm))

        self._set_weights(w,b,w_sum,b_sum,n_steps-avg_start+1)
        self.n_iter_=n_steps
        scores=y_pm*self.decision_function(X)
        self.support_=np.flatnonzero(scores<=1)
        self.n_support_=len(self.support_)
        self.w_norm_=float(np.linalg.norm(self.w))
        self.margin_=2/self.w_norm_ if self.w_norm_>0 else math.inf
        return self

    def _set_weights(self,w,b,w_sum,b_sum,n_averaged):
        if n_averaged>0:
            self.w,self.b=w_sum/n_averaged,b_sum/n_averaged
        else:
            self.w,self.b=w.copy(),b

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        X=np.asarray(X)
        w=self.w.astype(X.dtype) if X.dtype==np.float32 else self.w  # no float64 copy of X
        signed_margins=np.dot(X,w)+self.b
        return signed_margins

    def _objective(self,X,y_pm):
        scores=self.decision_function(X)
        all_margins=np.multiply(y_pm,scores)
        reg=(self.lambda_/2)*(np.dot(self.w,self.w)+self.b**2)
        hinge=np.maximum(0,1-all_margins)

        return float(reg+np.mean(hinge))

    def objective(self, X, y):
        """J(w, b) of the fitted weights on (X, y), y in the original labels."""
        y_pm=np.where(np.asarray(y)==self.classes_[0],-1,1)
        return self._objective(np.asarray(X),y_pm)

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""

        return np.where(self.decision_function(X)<0,self.classes_[0],self.classes_[1]) ##score=0 -> positive class is predicted: same as sklearn
