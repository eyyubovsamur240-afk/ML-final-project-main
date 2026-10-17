"""
svm_study.py — tuning and convergence study of our Pegasos SVM and its RFF
extension (Problems 3a-3d and 4b).

  1. lambda sweep (linear SVM): validation F1 / AUC, training F1, ||w||, the
     margin 2/||w||, the share of support vectors and the objective J for each
     lambda; lambda is chosen by validation F1, a tie going to the larger lambda
     (the wider margin, the simpler model).
  2. convergence at the chosen lambda: the first version's setting (one row per
     step, eta = 1/(lambda t), 200,000 steps, the last iterate) against
     mini-batches of 32 rows with suffix averaging for 5-100 epochs, each against
     LinearSVC's J at the same lambda (the same objective, see svm.py). The final
     model gets the fewest epochs that come within 1% of it.
  3. learning-rate schedules at the chosen lambda and epochs: J, margin and
     validation F1 of 1/(lambda t), 1/(lambda (t + t0)) and a constant step.
     Everything else uses the offset schedule 1/(lambda (t + t0)): with 1/(lambda t)
     the first steps are of size 1/lambda, which for a small lambda throws w far
     off and takes far more than our step budget to recover from.
  4. random Fourier features: gamma x lambda grid at D = 1,024 chosen by
     validation F1 (ties: larger lambda, then smaller gamma, the smoother
     kernel), then the number of features D at the chosen pair, with the mean
     error of the kernel approximation.

Only the training and validation splits are passed in, so nothing here can
look at the test split; run_all scores the chosen settings on test once.
The scikit-learn models here are references (LinearSVC's optimum), not ours.

Usage from run_all.py:
    summary = svm_study.run(X_train, y_train, X_val, y_val, fig_dir=...)
    summary["linear"]["settings"], summary["rff"]["settings"]  # -> PegasosSVM, RFFSVM
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatter
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from . import evaluate
from .rff import RandomFourierFeatures, kernel_approximation_error
from .svm import PegasosSVM

SEED = 42
LAMBDAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
BATCH = 32
SWEEP_EPOCHS = 50          # epochs of the lambda sweep (the RFF grid gets the chosen epochs)
EPOCHS = [5, 10, 25, 50, 100]
GAP_TOLERANCE = 0.01       # J within 1% of LinearSVC's
SCHEDULE = "offset"         # eta0 = svm.py's default, 1 / mean ||x||^2 of the training rows
FIRST_VERSION = {"batch_size": 1, "n_iters": 200_000, "average": False, "schedule": "pegasos"}
SCHEDULES = ["pegasos", "offset", "constant"]
SCHEDULE_LABELS = {"pegasos": "1/(lambda t)", "offset": "1/(lambda (t + t0))",
                   "constant": "constant step"}
RFF_GAMMAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
RFF_LAMBDAS = [1e-6, 1e-5, 1e-4]
RFF_COMPONENTS = 1024
RFF_COMPONENT_SWEEP = [32, 128, 512, 1024, 2048]
LINEAR_SVC_MAX_ITER = 20_000   # a time cap, as for the LinearSVC baseline


def _standardize(X_tr, X_val):
    """Training mean/std (constant columns keep std 1), as baselines.StandardScaled."""
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_tr - mean) / std, (X_val - mean) / std


def _f1(y, pred):
    return float(evaluate.precision_recall_f1(y, pred)[2])


def _scores(model, X_tr, y_tr, X_val, y_val) -> dict:
    """What every fitted SVM in the study reports (validation and training rows only)."""
    return {"val_f1": _f1(y_val, model.predict(X_val)),
            "val_auc": float(evaluate.roc_auc(y_val, model.decision_function(X_val))),
            "train_f1": _f1(y_tr, model.predict(X_tr)),
            "objective": float(model.history_[-1]),
            "w_norm": model.w_norm_, "margin": model.margin_,
            "support_share": model.n_support_ / len(y_tr), "n_steps": int(model.n_iter_),
            "eta0": float(model.eta0_)}


def fit_svm(params, X_tr, y_tr, X_val, y_val) -> tuple[dict, PegasosSVM]:
    start = time.perf_counter()
    model = PegasosSVM(random_state=SEED, **params).fit(X_tr, y_tr)
    seconds = time.perf_counter() - start
    return {**_scores(model, X_tr, y_tr, X_val, y_val), "fit_seconds": seconds}, model


def linear_svc_reference(lambda_, X_tr, y_tr) -> dict:
    """LinearSVC with C = 1/(lambda n): its J (our formula; liblinear regularises the
    bias like ours), margin and support vectors, as the optimum Pegasos is aiming at."""
    svc = LinearSVC(C=1.0 / (lambda_ * len(y_tr)), loss="hinge", dual=True,
                    max_iter=LINEAR_SVC_MAX_ITER, random_state=SEED)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        svc.fit(X_tr, y_tr)
    seconds = time.perf_counter() - start
    ref = PegasosSVM(lambda_=lambda_)
    ref.classes_ = svc.classes_
    ref.w, ref.b = svc.coef_.ravel().astype(float), float(svc.intercept_[0])
    w_norm = float(np.linalg.norm(ref.w))
    margins = np.where(y_tr == ref.classes_[0], -1, 1) * ref.decision_function(X_tr)
    return {"objective": ref.objective(X_tr, y_tr), "w_norm": w_norm, "margin": 2 / w_norm,
            "support_share": float(np.mean(margins <= 1)), "fit_seconds": seconds,
            "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            "max_iter": LINEAR_SVC_MAX_ITER}


def _pick(rows, key_order):
    """The row with the best validation F1; rows are tried in key_order (simplest first),
    so an exact tie keeps the simpler setting."""
    best = None
    for row in sorted(rows, key=key_order):
        if best is None or row["val_f1"] > best["val_f1"]:
            best = row
    return best


def lambda_sweep(X_tr, y_tr, X_val, y_val, lambdas=None, epochs=None) -> dict:
    lambdas = LAMBDAS if lambdas is None else lambdas
    epochs = SWEEP_EPOCHS if epochs is None else epochs
    rows = []
    for lam in lambdas:
        res, _ = fit_svm({"lambda_": lam, "n_epochs": epochs, "batch_size": BATCH,
                          "average": True, "schedule": SCHEDULE}, X_tr, y_tr, X_val, y_val)
        rows.append({"lambda_": lam, **res})
    best = _pick(rows, key_order=lambda r: -r["lambda_"])   # larger lambda first
    return {"rows": rows, "chosen_lambda": best["lambda_"], "epochs": epochs}


def convergence(lambda_, X_tr, y_tr, X_val, y_val, epochs_grid=None,
                first_version=None) -> dict:
    """Objective gap to LinearSVC for the first version's setting and for mini-batch +
    averaging at several epoch counts; histories are kept for the objective figure."""
    epochs_grid = EPOCHS if epochs_grid is None else epochs_grid
    first_version = FIRST_VERSION if first_version is None else first_version
    ref = linear_svc_reference(lambda_, X_tr, y_tr)

    def gap(objective):
        return objective / ref["objective"] - 1

    settings = [("first version: 1 row per step, 1/(lambda t), last iterate",
                 {"lambda_": lambda_, **first_version})]
    settings.append((f"batch {BATCH}, last iterate, {SWEEP_EPOCHS} epochs",
                     {"lambda_": lambda_, "n_epochs": SWEEP_EPOCHS, "batch_size": BATCH,
                      "average": False, "schedule": SCHEDULE}))
    settings += [(f"batch {BATCH}, averaged, {e} epochs",
                  {"lambda_": lambda_, "n_epochs": e, "batch_size": BATCH, "average": True,
                   "schedule": SCHEDULE})
                 for e in epochs_grid]
    rows = []
    for label, params in settings:
        res, model = fit_svm(params, X_tr, y_tr, X_val, y_val)
        rows.append({"setting": label, "epochs": params.get("n_epochs"),
                     "batch_size": params["batch_size"], "average": params["average"],
                     "schedule": params["schedule"], **res,
                     "gap": gap(res["objective"]), "history": [float(j) for j in model.history_],
                     "steps_per_record": model.steps_per_epoch_})
    averaged = [r for r in rows if r["average"]]
    within = [r for r in averaged if r["gap"] <= GAP_TOLERANCE]
    chosen = min(within, key=lambda r: r["epochs"]) if within else max(averaged,
                                                                        key=lambda r: r["epochs"])
    return {"reference": ref, "rows": rows, "chosen_epochs": chosen["epochs"],
            "tolerance": GAP_TOLERANCE}


def schedule_study(lambda_, epochs, reference_objective, X_tr, y_tr, X_val, y_val,
                   schedules=None) -> list[dict]:
    rows = []
    for schedule in SCHEDULES if schedules is None else schedules:
        params = {"lambda_": lambda_, "n_epochs": epochs, "batch_size": BATCH, "average": True,
                  "schedule": schedule}
        res, model = fit_svm(params, X_tr, y_tr, X_val, y_val)
        rows.append({"schedule": schedule, **res,
                     "gap": res["objective"] / reference_objective - 1,
                     "history": [float(j) for j in model.history_]})
    return rows


def rff_study(X_tr, y_tr, X_val, y_val, epochs, gammas=None, lambdas=None,
              n_components=None, component_sweep=None) -> dict:
    """The features depend only on (gamma, D, seed), so they are drawn once per gamma and
    shared by the lambdas: the same numbers as fitting rff.RFFSVM for every cell."""
    gammas = RFF_GAMMAS if gammas is None else gammas
    lambdas = RFF_LAMBDAS if lambdas is None else lambdas
    n_components = RFF_COMPONENTS if n_components is None else n_components
    component_sweep = RFF_COMPONENT_SWEEP if component_sweep is None else component_sweep
    def fit_on_features(gamma, lam, D):
        features = RandomFourierFeatures(D, gamma, SEED)
        start = time.perf_counter()
        Z_tr = features.fit_transform(X_tr)
        transform_s = time.perf_counter() - start
        Z_val = features.transform(X_val)
        out = []
        for one_lambda in np.atleast_1d(lam):
            res, _ = fit_svm({"lambda_": float(one_lambda), "n_epochs": epochs,
                              "batch_size": BATCH, "average": True, "schedule": SCHEDULE},
                             Z_tr, y_tr, Z_val, y_val)
            res["fit_seconds"] += transform_s
            out.append({"gamma": gamma, "lambda_": float(one_lambda), "n_components": D, **res})
        return out

    grid = [row for gamma in gammas for row in fit_on_features(gamma, lambdas, n_components)]
    best = _pick(grid, key_order=lambda r: (-r["lambda_"], r["gamma"]))
    components = []
    for D in component_sweep:
        row = fit_on_features(best["gamma"], best["lambda_"], D)[0]
        row["kernel_error"] = kernel_approximation_error(X_tr, best["gamma"], D, SEED)
        components.append(row)
    return {"grid": grid, "components": components,
            "settings": {"gamma": best["gamma"], "lambda_": best["lambda_"],
                         "n_components": n_components, "n_epochs": epochs,
                         "batch_size": BATCH, "average": True, "schedule": SCHEDULE},
            "validation": {"f1": best["val_f1"], "auc": best["val_auc"]},
            "kernel_error": next(r["kernel_error"] for r in components
                                 if r["n_components"] == n_components)
            if n_components in component_sweep else None}


# --- figures --------------------------------------------------------------------
def _save(fig, name, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return path


def plot_lambda(sweep, fig_dir):
    """Top: validation and training F1 against lambda (log x). Bottom: the margin
    2/||w|| (log) and the support-vector share."""
    rows = sweep["rows"]
    lam = [r["lambda_"] for r in rows]
    fig = Figure(figsize=(5, 4.6))
    top, bottom = fig.subplots(2, 1, sharex=True)
    top.plot(lam, [r["val_f1"] for r in rows], "-o", ms=3, color="tab:blue",
             label="validation F1")
    top.plot(lam, [r["train_f1"] for r in rows], "--", color="tab:blue", label="training F1")
    top.set_ylabel("F1 (premium)")
    top.legend(fontsize=7, loc="lower left")
    top.set_title(f"Linear SVM: {sweep['epochs']} epochs, batch {BATCH}, averaged", fontsize=9)
    bottom.plot(lam, [r["margin"] for r in rows], "-s", ms=3, color="tab:red",
                label=r"margin $2/\|w\|$")
    bottom.set_yscale("log")
    plain = dict(labelOnlyBase=False, minor_thresholds=(2, 0.5))   # 1.5, not 1.5 x 10^0
    bottom.yaxis.set_major_formatter(LogFormatter(**plain))
    bottom.yaxis.set_minor_formatter(LogFormatter(**plain))
    bottom.set_ylabel(r"margin $2/\|w\|$")
    share = bottom.twinx()
    share.plot(lam, [100 * r["support_share"] for r in rows], "-^", ms=3, color="tab:green",
               label="support vectors")
    share.set_ylabel("support vectors (% of rows)")
    lines = bottom.get_lines() + share.get_lines()
    bottom.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="upper left")
    bottom.set_xscale("log")
    bottom.set_xlabel(r"$\lambda$")
    for ax in (top, bottom):
        ax.axvline(sweep["chosen_lambda"], color="grey", ls=":", lw=1)
        ax.grid(alpha=0.3)
    return _save(fig, "svm_lambda.png", fig_dir)


def plot_objective(conv, schedules, fig_dir):
    """J per epoch for the convergence settings and the schedules against LinearSVC's
    J. Top: all of it (log-log). Bottom: the gap to LinearSVC in %, zoomed in."""
    ref = conv["reference"]["objective"]
    curves = []
    for r in conv["rows"]:
        if r["average"] and r["epochs"] != conv["chosen_epochs"]:
            continue   # the averaged runs differ only in length: show the chosen one
        steps = np.arange(1, len(r["history"]) + 1) * r["steps_per_record"]
        epochs = np.minimum(steps, r["n_steps"]) * r["batch_size"] / conv["n_train"]
        curves.append((epochs, np.asarray(r["history"]), r["setting"], "-"))
    for r in schedules:
        if r["schedule"] == SCHEDULE:
            continue   # the same run as the chosen averaged one above
        curves.append((np.arange(1, len(r["history"]) + 1), np.asarray(r["history"]),
                       f"batch {BATCH}, averaged, {SCHEDULE_LABELS[r['schedule']]}", "--"))
    colors = ["tab:gray", "tab:brown", "tab:blue", "tab:orange", "tab:green", "tab:purple",
              "tab:pink", "tab:olive", "tab:cyan", "black"]
    fig = Figure(figsize=(5, 5))
    top, bottom = fig.subplots(2, 1, sharex=True)
    for (epochs, history, label, ls), color in zip(curves, colors, strict=False):
        top.plot(epochs, history, color=color, lw=1.2, ls=ls, label=label)
        bottom.plot(epochs, 100 * (history / ref - 1), color=color, lw=1.2, ls=ls)
    top.axhline(ref, color="red", ls=":", lw=1.2, label="LinearSVC (same objective)")
    bottom.axhline(0, color="red", ls=":", lw=1.2)
    top.set_yscale("log")
    top.set_ylabel(r"objective $J(w, b)$")
    top.set_title(rf"Pegasos objective on the training split, $\lambda$ = {conv['lambda_']:g}",
                  fontsize=9)
    top.legend(fontsize=6)
    low = min(-1.0, min(100 * (h[-1] / ref - 1) for _, h, _, _ in curves) - 0.5)
    bottom.set_ylim(low, 10)
    bottom.set_ylabel("gap to LinearSVC (%)")
    bottom.set_xscale("log")
    bottom.set_xlabel("epochs (passes over the training rows)")
    for ax in (top, bottom):
        ax.grid(alpha=0.3, which="both")
    return _save(fig, "svm_objective.png", fig_dir)


def plot_rff_grid(rff, fig_dir):
    gammas = sorted({r["gamma"] for r in rff["grid"]})
    lambdas = sorted({r["lambda_"] for r in rff["grid"]})
    f1 = {(r["gamma"], r["lambda_"]): r["val_f1"] for r in rff["grid"]}
    scores = np.array([[f1[(g, lam)] for g in gammas] for lam in lambdas])
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    im = ax.imshow(scores, cmap="Blues", aspect="auto")
    for i in range(len(lambdas)):
        for j in range(len(gammas)):
            dark = im.norm(scores[i, j]) > 0.6
            ax.text(j, i, f"{scores[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if dark else "black")
    best = rff["settings"]
    bi, bj = lambdas.index(best["lambda_"]), gammas.index(best["gamma"])
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    ax.set_xticks(range(len(gammas)), labels=[f"{g:g}" for g in gammas])
    ax.set_yticks(range(len(lambdas)), labels=[f"{lam:g}" for lam in lambdas])
    ax.set_xlabel(r"$\gamma$ (RBF bandwidth)")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(f"RFF SVM, D = {best['n_components']:,}: validation F1", fontsize=9)
    fig.colorbar(im, ax=ax, label="validation F1")
    return _save(fig, "svm_rff_grid.png", fig_dir)


def plot_rff_components(rff, fig_dir):
    rows = rff["components"]
    D = [r["n_components"] for r in rows]
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    ax.plot(D, [r["val_f1"] for r in rows], "-o", ms=3, color="tab:blue", label="validation F1")
    ax.plot(D, [r["val_auc"] for r in rows], "-s", ms=3, color="tab:orange",
            label="validation AUC")
    ax.set_xscale("log", base=2)
    ax.set_xticks(D, labels=[str(d) for d in D])
    ax.set_xlabel("number of random features D")
    ax.set_ylabel("validation score")
    right = ax.twinx()
    right.plot(D, [r["kernel_error"] for r in rows], "-^", ms=3, color="tab:red",
               label="kernel approximation error")
    right.set_yscale("log")
    right.set_ylabel(r"mean $|z(x)^\top z(x') - k(x, x')|$")
    lines = ax.get_lines() + right.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="center right")
    s = rff["settings"]
    ax.set_title(rf"RFF SVM at $\gamma$ = {s['gamma']:g}, $\lambda$ = {s['lambda_']:g}",
                 fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "svm_rff_components.png", fig_dir)


# --- the whole study -------------------------------------------------------------
def run(X_tr, y_tr, X_val, y_val, fig_dir=None) -> dict:
    """The study on unscaled features (standardized here with training statistics)."""
    X_tr, X_val = _standardize(np.asarray(X_tr, dtype=float), np.asarray(X_val, dtype=float))
    y_tr, y_val = np.asarray(y_tr), np.asarray(y_val)

    print("  SVM: lambda sweep ...", flush=True)
    sweep = lambda_sweep(X_tr, y_tr, X_val, y_val)
    lam = sweep["chosen_lambda"]
    print(f"  SVM: convergence at lambda = {lam:g} ...", flush=True)
    conv = convergence(lam, X_tr, y_tr, X_val, y_val)
    conv["lambda_"], conv["n_train"] = lam, len(y_tr)
    epochs = conv["chosen_epochs"]
    schedules = schedule_study(lam, epochs, conv["reference"]["objective"],
                               X_tr, y_tr, X_val, y_val)
    print("  SVM: random Fourier features ...", flush=True)
    rff = rff_study(X_tr, y_tr, X_val, y_val, epochs)

    final = next(r for r in conv["rows"] if r["average"] and r["epochs"] == epochs)
    summary = {
        "linear": {"settings": {"lambda_": lam, "n_epochs": epochs, "batch_size": BATCH,
                                "average": True, "schedule": SCHEDULE},
                   "validation": {"f1": final["val_f1"], "auc": final["val_auc"]},
                   "margin": final["margin"], "w_norm": final["w_norm"],
                   "support_share": final["support_share"], "objective": final["objective"],
                   "objective_gap": final["gap"], "lambda_sweep": sweep["rows"],
                   "sweep_epochs": sweep["epochs"]},
        "convergence": {**{k: v for k, v in conv.items() if k != "rows"},
                        "rows": conv["rows"]},
        "schedules": schedules,
        "rff": rff,
        "grids": {"lambdas": LAMBDAS, "epochs": EPOCHS, "batch_size": BATCH,
                  "rff_gammas": RFF_GAMMAS, "rff_lambdas": RFF_LAMBDAS,
                  "rff_components": RFF_COMPONENT_SWEEP, "first_version": FIRST_VERSION,
                  "tolerance": GAP_TOLERANCE, "schedule": SCHEDULE},
    }
    if fig_dir is not None:
        summary["figures"] = [plot_lambda(sweep, fig_dir), plot_objective(conv, schedules, fig_dir),
                              plot_rff_grid(rff, fig_dir), plot_rff_components(rff, fig_dir)]
    return summary


def _first_version_gap(summary) -> float:
    return next(r["gap"] for r in summary["convergence"]["rows"] if r["batch_size"] == 1)


def _demo(fig_dir="results/svm_demo"):
    """Quick check without the real data: a small non-linear synthetic problem."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3_000, 6))
    y = ((X[:, 0] ** 2 + X[:, 1] - 0.5 * X[:, 2] + 0.3 * rng.normal(size=3_000)) > 1).astype(int)
    summary = run(X[:2_000], y[:2_000], X[2_000:], y[2_000:], fig_dir=fig_dir)
    lin, rff = summary["linear"], summary["rff"]
    print(f"linear: {lin['settings']}, validation F1 {lin['validation']['f1']:.3f}, "
          f"objective gap {lin['objective_gap']:+.2%} (first version "
          f"{_first_version_gap(summary):+.2%})")
    print(f"RFF: {rff['settings']}, validation F1 {rff['validation']['f1']:.3f}, "
          f"kernel error {rff['kernel_error']:.4f}")
    print("figures:", ", ".join(summary["figures"]))
    return summary


if __name__ == "__main__":
    _demo()
