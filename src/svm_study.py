"""
svm_study.py — tuning and diagnosis of OUR Pegasos SVM (Problems 3b, 3c, 4b, 4c).

  * lambda sweep  : for each lambda, the margin (||w||, 2/||w||), the share of
                    support vectors and the train / validation F1. Shows the
                    trade-off: big lambda -> wide street but underfit; small
                    lambda -> narrow street, fits the training rows harder.
  * schedules     : the objective J during training for the three learning-rate
                    schedules (convergence plot, Problem 3a/3b).
  * epochs        : validation F1 and final J vs number of epochs (Problem 3b).
  * RFF gamma     : validation F1 of the random-Fourier-feature SVM for each
                    kernel bandwidth gamma (Problem 3d/4b).

Selection uses the VALIDATION split only; the test split is never touched here
(Problem 4c). Features are standardized with TRAIN statistics (hinge loss is
scale-sensitive).

Usage from run_all.py (or a notebook):
    from . import svm_study
    res = svm_study.run(X_tr, y_tr, X_val, y_val)   # y = price tier (0/1)
    res["best_lambda"], res["best_gamma"]

Quick check without the real data (synthetic data, writes to results/svm_demo/):
    python -m src.svm_study
"""

from __future__ import annotations

import os
import time

import numpy as np
# Figure objects directly (no pyplot), as in tree_experiments: runs without a display.
from matplotlib.figure import Figure

from . import evaluate
from .data_prep import standardize
from .rff import RFFSVM, make_rings
from .svm import SCHEDULES, PegasosSVM

SEED = 42
FIG_DIR = "report/figures"

LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3]
EPOCHS = [1, 2, 5, 10, 20]
N_EPOCHS = 10          # training length for the lambda / gamma sweeps
RFF_COMPONENTS = 300   # D, number of random Fourier features


def _f1(y_true, y_pred) -> float:
    return evaluate.precision_recall_f1(y_true, y_pred)[2]


def _record_every(n_rows: int, n_epochs: float, points: int = 200) -> int:
    """Record J about `points` times per run: enough for a plot, cheap on big data."""
    return max(1, int(n_rows * n_epochs) // points)


def _fit(model, X_tr, y_tr):
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    return time.perf_counter() - start


# --- sweeps ------------------------------------------------------------------
def lambda_sweep(X_tr, y_tr, X_val, y_val, lambdas=LAMBDAS, n_epochs=N_EPOCHS, seed=SEED):
    """One row per lambda: margin, support vectors, objective, train / val F1, fit time."""
    rows = []
    for lam in lambdas:
        svm = PegasosSVM(lambda_=lam, n_epochs=n_epochs, random_state=seed,
                         record_every=_record_every(len(y_tr), n_epochs))
        seconds = _fit(svm, X_tr, y_tr)
        rows.append({
            "lambda": lam,
            "w_norm": svm.weight_norm(),
            "margin": svm.margin_width(),
            "support_share": len(svm.support_) / len(y_tr),
            "objective": svm.history_[-1],
            "train_f1": _f1(y_tr, svm.predict(X_tr)),
            "val_f1": _f1(y_val, svm.predict(X_val)),
            "fit_seconds": seconds,
        })
    return rows


def schedule_study(X_tr, y_tr, X_val, y_val, lambda_, n_epochs=N_EPOCHS, eta0=0.1, seed=SEED):
    """Objective history for each learning-rate schedule (same lambda, same samples)."""
    out = {}
    every = _record_every(len(y_tr), n_epochs)
    for schedule in SCHEDULES:
        svm = PegasosSVM(lambda_=lambda_, n_epochs=n_epochs, schedule=schedule, eta0=eta0,
                         random_state=seed, record_every=every)
        _fit(svm, X_tr, y_tr)
        out[schedule] = {"steps": every * np.arange(1, len(svm.history_) + 1),
                         "history": np.asarray(svm.history_),
                         "val_f1": _f1(y_val, svm.predict(X_val)),
                         "margin": svm.margin_width()}
    return out


def epoch_sweep(X_tr, y_tr, X_val, y_val, lambda_, epochs=EPOCHS, seed=SEED):
    """Validation F1 and final objective as training gets longer."""
    rows = []
    for e in epochs:
        svm = PegasosSVM(lambda_=lambda_, n_epochs=e, random_state=seed,
                         record_every=_record_every(len(y_tr), e))
        _fit(svm, X_tr, y_tr)
        rows.append({"epochs": e, "objective": svm.history_[-1] if svm.history_ else np.nan,
                     "val_f1": _f1(y_val, svm.predict(X_val)), "margin": svm.margin_width()})
    return rows


def rff_sweep(X_tr, y_tr, X_val, y_val, lambda_, gammas=GAMMAS, n_components=RFF_COMPONENTS,
              n_epochs=N_EPOCHS, seed=SEED):
    """One row per gamma for the random-Fourier-feature SVM."""
    rows = []
    for gamma in gammas:
        model = RFFSVM(n_components=n_components, gamma=gamma, lambda_=lambda_,
                       random_state=seed, n_epochs=n_epochs,
                       record_every=_record_every(len(y_tr), n_epochs))
        seconds = _fit(model, X_tr, y_tr)
        rows.append({"gamma": gamma, "train_f1": _f1(y_tr, model.predict(X_tr)),
                     "val_f1": _f1(y_val, model.predict(X_val)), "fit_seconds": seconds})
    return rows


def pick_best(rows, key, prefer_larger=True):
    """Highest validation F1. On a tie, the simpler model: the LARGER lambda (more
    regularisation) or the SMALLER gamma (smoother boundary) -> prefer_larger=False."""
    sign = 1 if prefer_larger else -1
    best = max(rows, key=lambda r: (round(r["val_f1"], 6), sign * r[key]))
    return best[key]


# --- plots -------------------------------------------------------------------
def _save(fig, name, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.tight_layout()
    fig.savefig(path)
    return path


def plot_lambda(rows, best, fig_dir=FIG_DIR, name="svm_lambda_sweep.pdf"):
    """Left: train / val F1 vs lambda. Right: margin width 2/||w|| vs lambda."""
    lam = [r["lambda"] for r in rows]
    fig = Figure(figsize=(8, 3.2))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(lam, [r["train_f1"] for r in rows], "o-", label="train F1")
    ax1.plot(lam, [r["val_f1"] for r in rows], "s-", label="validation F1")
    ax1.axvline(best, color="grey", ls="--", lw=1, label=f"chosen lambda = {best:g}")
    ax1.set(xscale="log", xlabel="lambda", ylabel="F1 (premium)", title="Fit vs regularisation")
    ax1.legend(fontsize=8)
    ax2.plot(lam, [r["margin"] for r in rows], "o-", color="tab:green")
    ax2.set(xscale="log", yscale="log", xlabel="lambda", ylabel="margin width 2/||w||",
            title="Bigger lambda -> wider street")
    return _save(fig, name, fig_dir)


def plot_convergence(schedules, fig_dir=FIG_DIR, name="svm_convergence.pdf"):
    """Objective J during training, one line per learning-rate schedule."""
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    for schedule, res in schedules.items():
        ax.plot(res["steps"], res["history"], label=f"{schedule} (val F1 {res['val_f1']:.3f})")
    ax.set(yscale="log", xlabel="step t", ylabel="objective J(w, b)",
           title="Pegasos convergence by schedule")
    ax.legend(fontsize=8)
    return _save(fig, name, fig_dir)


def plot_epochs(rows, fig_dir=FIG_DIR, name="svm_epochs.pdf"):
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    ax.plot([r["epochs"] for r in rows], [r["val_f1"] for r in rows], "o-", label="validation F1")
    ax.set(xscale="log", xlabel="epochs", ylabel="validation F1", title="Training length")
    ax2 = ax.twinx()
    ax2.plot([r["epochs"] for r in rows], [r["objective"] for r in rows], "s--",
             color="tab:red", label="final J")
    ax2.set_ylabel("final objective J", color="tab:red")
    return _save(fig, name, fig_dir)


def plot_rff(rows, linear_val_f1, best, fig_dir=FIG_DIR, name="svm_rff_gamma.pdf"):
    g = [r["gamma"] for r in rows]
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    ax.plot(g, [r["train_f1"] for r in rows], "o-", label="RFF train F1")
    ax.plot(g, [r["val_f1"] for r in rows], "s-", label="RFF validation F1")
    ax.axhline(linear_val_f1, color="grey", ls=":", label="linear SVM, validation F1")
    ax.axvline(best, color="grey", ls="--", lw=1, label=f"chosen gamma = {best:g}")
    ax.set(xscale="log", xlabel="gamma (RBF bandwidth)", ylabel="F1",
           title="Random Fourier features")
    ax.legend(fontsize=8)
    return _save(fig, name, fig_dir)


# --- driver ------------------------------------------------------------------
def run(X_tr, y_tr, X_val, y_val, fig_dir=FIG_DIR, lambdas=LAMBDAS, gammas=GAMMAS,
        epochs=EPOCHS, n_epochs=N_EPOCHS, n_components=RFF_COMPONENTS, seed=SEED, verbose=True):
    """Full SVM study on train / validation only. Returns every table, the choices and figures."""
    X_tr, X_val = standardize(X_tr, X_val)       # TRAIN mean / std only
    y_tr, y_val = np.asarray(y_tr), np.asarray(y_val)

    lam_rows = lambda_sweep(X_tr, y_tr, X_val, y_val, lambdas, n_epochs, seed)
    best_lambda = pick_best(lam_rows, "lambda")
    schedules = schedule_study(X_tr, y_tr, X_val, y_val, best_lambda, n_epochs, seed=seed)
    epoch_rows = epoch_sweep(X_tr, y_tr, X_val, y_val, best_lambda, epochs, seed)
    rff_rows = rff_sweep(X_tr, y_tr, X_val, y_val, best_lambda, gammas, n_components,
                         n_epochs, seed)
    best_gamma = pick_best(rff_rows, "gamma", prefer_larger=False)
    linear_val_f1 = next(r["val_f1"] for r in lam_rows if r["lambda"] == best_lambda)

    figures = [plot_lambda(lam_rows, best_lambda, fig_dir),
               plot_convergence(schedules, fig_dir),
               plot_epochs(epoch_rows, fig_dir),
               plot_rff(rff_rows, linear_val_f1, best_gamma, fig_dir)]

    if verbose:
        print(evaluate.format_table(
            lam_rows, ["lambda", "w_norm", "margin", "support_share", "objective",
                       "train_f1", "val_f1", "fit_seconds"],
            ["lambda", "||w||", "2/||w||", "SV share", "J", "train F1", "val F1", "fit s"],
            title="SVM: lambda sweep (validation picks lambda)", digits=4))
        print(f"chosen lambda = {best_lambda:g}\n")
        for schedule, res in schedules.items():
            print(f"schedule {schedule:9s} final J = {res['history'][-1]:.4f}  "
                  f"val F1 = {res['val_f1']:.4f}")
        print()
        print(evaluate.format_table(epoch_rows, ["epochs", "objective", "val_f1", "margin"],
                                    ["epochs", "final J", "val F1", "2/||w||"],
                                    title="SVM: training length", digits=4))
        print()
        print(evaluate.format_table(rff_rows, ["gamma", "train_f1", "val_f1", "fit_seconds"],
                                    ["gamma", "train F1", "val F1", "fit s"],
                                    title=f"RFF SVM (D = {n_components}): gamma sweep", digits=4))
        print(f"chosen gamma = {best_gamma:g}   (linear SVM val F1 = {linear_val_f1:.4f})")

    return {"lambda_sweep": lam_rows, "best_lambda": best_lambda,
            "schedules": {k: {"final_objective": float(v["history"][-1]),
                              "val_f1": v["val_f1"]} for k, v in schedules.items()},
            "epochs": epoch_rows, "rff_sweep": rff_rows, "best_gamma": best_gamma,
            "figures": figures}


def _demo(fig_dir="results/svm_demo"):
    """Rings: the linear SVM cannot separate them, the RFF SVM can (no real data needed)."""
    X, y = make_rings(n=600, seed=SEED)
    order = np.random.default_rng(SEED).permutation(len(y))
    X, y = X[order], y[order]
    cut = int(0.7 * len(y))
    return run(X[:cut], y[:cut], X[cut:], y[cut:], fig_dir=fig_dir,
               lambdas=[1e-4, 1e-3, 1e-2, 1e-1], gammas=[0.1, 0.3, 1.0, 3.0],
               epochs=[1, 5, 20], n_epochs=10)


if __name__ == "__main__":
    _demo()
