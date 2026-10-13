"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
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


    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t), and apply the update above.
        Return self.
        """
        X=np.asarray(X,dtype=float)
        n_samples,n_features=X.shape
        X_aug=np.hstack([X,np.ones((n_samples,1))])
        y_pm,self.classes_=self._to_pm1(y)
        rng=np.random.default_rng(self.random_state)
        w_aug=np.zeros(n_features+1)
        self.history_=[]
        
        for t in range(1,self.n_iters+1):
            learning_rate=1/(self.lambda_*t)
            i=rng.integers(n_samples)
            x_i=X_aug[i]
            y_i=y_pm[i]

            score=w_aug @ x_i
            margin=y_i*score
            if margin<1: 
                w_aug=(1-learning_rate*self.lambda_)*w_aug+learning_rate*y_i*x_i
            else:
                w_aug=(1-learning_rate*self.lambda_)*w_aug

            if t%100==0: 
                self.w,self.b=w_aug[:-1],w_aug[-1]
                self.history_.append(self._objective(X,y_pm))

 
    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        signed_margins=np.dot(X,self.w)+self.b
        return signed_margins

    def _objective(self,X,y_pm):
        raise NotImplementedError("No enough place")


from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
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
        

    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t), and apply the update above.
        Return self.
        """
        X=np.asarray(X,dtype=float)
        n_samples,n_features=X.shape
        X_aug=np.hstack([X,np.ones((n_samples,1))])
        y_pm,self.classes_=self._to_pm1(y)
        rng=np.random.default_rng(self.random_state)
        w_aug=np.zeros(n_features+1)
        self.history_=[]
        
        for t in range(1,self.n_iters+1):
            learning_rate=1/(self.lambda_*t)
            i=rng.integers(n_samples)
            x_i=X_aug[i]
            y_i=y_pm[i]

            score=w_aug @ x_i
            margin=y_i*score
            if margin<1: 
                w_aug=(1-learning_rate*self.lambda_)*w_aug+learning_rate*y_i*x_i
            else:
                w_aug=(1-learning_rate*self.lambda_)*w_aug

            if t%100==0: 
                self.w,self.b=w_aug[:-1],w_aug[-1]
                self.history_.append(self._objective(X,y_pm))

        self.w,self.b=w_aug[:-1],w_aug[-1]
        return self
            

    
    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        signed_margins=np.dot(X,self.w)+self.b
        return signed_margins

    def _objective(self,X,y_pm):
        scores=self.decision_function(X)
        all_margins=np.multiply(y_pm,scores)
        reg=(self.lambda_/2)*np.dot(self.w,self.w)
        hinge=np.maximum(0,1-all_margins)

        return reg+np.mean(hinge)
        

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""

        return np.where(self.decision_function(X)<0,self.classes_[0],self.classes_[1]) ##score=0 -> positive class is predicted: same as sklearn
        

# if __name__ == "__main__":
#     # --- Test 1: decision_function + predict (hand-set w, b) ---
#     m = PegasosSVM()
#     y_pm, m.classes_ = m._to_pm1(np.array([0, 1, 1, 0]))
#     print(y_pm, m.classes_)             # expect [-1  1  1 -1] [0 1]
#     m.w = np.array([2., 1.]); m.b = -4.
#     X = np.array([[3, 1], [1, 1], [2, 0]])
#     print(m.decision_function(X))      # expect [ 3. -1.  0.]
#     print(m.predict(X))                # expect [1 0 1]

#     # --- Test 2: objective J by hand ---
#     m = PegasosSVM(lambda_=0.1)
#     m.w = np.array([2., 1.]); m.b = -4.
#     X = np.array([[3, 1], [1, 1], [2, 0], [2, 1]])
#     y_pm = np.array([1, -1, 1, -1])
#     print(m._objective(X, y_pm))       # expect 1.0

#     # --- Test 3: TOY TEST - train on two separated clouds ---
#     rng = np.random.default_rng(0)
#     X_toy = np.vstack([rng.normal(-2, 1, (50, 2)), rng.normal(2, 1, (50, 2))])
#     y_toy = np.array([0]*50 + [1]*50)
#     m = PegasosSVM(lambda_=0.01, n_iters=5000).fit(X_toy, y_toy)
#     print("accuracy:", (m.predict(X_toy) == y_toy).mean())     # expect 1.0
#     print("J first/last:", m.history_[0], m.history_[-1])     # expect ~0.0748 -> ~0.0221
#     print("w, b:", m.w, m.b)      
