"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.

What is in here (Problem 6, plus the uncertainty part of Problem 5c):
  - metrics:        rmse, mae, r2 | confusion_matrix, precision_recall_f1,
                    accuracy, roc_auc, roc_curve, precision_recall_curve,
                    average_precision (PR-AUC)
  - per-task dicts: regression_metrics, classification_metrics
  - uncertainty:    kfold_indices, cross_validate (mean ± std over folds,
                    plus fit/predict time)
  - comparison:     format_table, compare (the required results table,
                    as plain text, Markdown or LaTeX)
  - inspection:     feature_importance_table (tree), svm_weight_table and
                    margin_summary (SVM)
  - error analysis: top_errors, errors_by_group (e.g. district),
                    errors_by_bins (e.g. price deciles -> luxury outliers)
  - figures:        plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
                    plot_pred_vs_true, plot_bar

Every metric is checked against sklearn.metrics in tests/test_evaluate.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data_prep import SEED


def _pair(y_true, y_pred):
    """Return y_true, y_pred as equal-length 1-D float arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
    if y_true.size == 0:
        raise ValueError("empty input")
    return y_true, y_pred


def _binary(y, pos_label):
    """Map labels to a boolean 'is positive (premium)' mask."""
    return np.asarray(y).ravel() == pos_label


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:  # constant target: same convention as sklearn
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred, inverse_transform=None) -> dict:
    """
    RMSE, MAE and R2 for Task A.

    If the model was trained on a transformed target (e.g. log(price)), pass
    the inverse (e.g. np.exp or np.expm1) and the same three metrics are also
    reported on the original price scale, as "RMSE (price)" etc.
    """
    out = {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred)}
    if inverse_transform is not None:
        p_true = inverse_transform(np.asarray(y_true, dtype=float))
        p_pred = inverse_transform(np.asarray(y_pred, dtype=float))
        out["RMSE (price)"] = rmse(p_true, p_pred)
        out["MAE (price)"] = mae(p_true, p_pred)
        out["R2 (price)"] = r2(p_true, p_pred)
    return out


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred, pos_label=1):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    t, p = _binary(y_true, pos_label), _binary(y_pred, pos_label)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: {t.shape[0]} vs {p.shape[0]}")
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred, pos_label=1):
    """Return (precision, recall, f1) for the positive (premium) class.

    An undefined ratio (no predicted / no actual positives) is reported as 0.0,
    like sklearn's zero_division=0.
    """
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred, pos_label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
    return float(np.mean(y_true == y_pred))


def _average_ranks(x):
    """1-based ranks of x, with tied values sharing their average rank."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)               # rank of the last copy of each value
    return (upper - (counts - 1) / 2.0)[inverse]


def roc_auc(y_true, scores, pos_label=1) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.

    AUC = P(score of a random positive > score of a random negative), ties
    counting 1/2. With average ranks R of the positives:
        AUC = (sum(R_pos) - n_pos (n_pos + 1) / 2) / (n_pos * n_neg)
    """
    t = _binary(y_true, pos_label)
    s = np.asarray(scores, dtype=float).ravel()
    if t.shape != s.shape:
        raise ValueError(f"length mismatch: {t.shape[0]} vs {s.shape[0]}")
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC needs both classes in y_true")
    ranks = _average_ranks(s)
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ranked_counts(y_true, scores, pos_label):
    """Cumulative TP/FP counts at each distinct threshold, highest score first."""
    t = _binary(y_true, pos_label)
    s = np.asarray(scores, dtype=float).ravel()
    if t.shape != s.shape:
        raise ValueError(f"length mismatch: {t.shape[0]} vs {s.shape[0]}")
    order = np.argsort(-s, kind="mergesort")
    s, t = s[order], t[order]
    last_of_each = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    tps = np.cumsum(t)[last_of_each]
    fps = (last_of_each + 1) - tps
    return tps, fps, s[last_of_each], int(t.sum())


def roc_curve(y_true, scores, pos_label=1):
    """Return (fpr, tpr, thresholds), starting at (0, 0) like sklearn."""
    tps, fps, thr, n_pos = _ranked_counts(y_true, scores, pos_label)
    n_neg = fps[-1]
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC curve needs both classes in y_true")
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    return fpr, tpr, np.r_[np.inf, thr]


def precision_recall_curve(y_true, scores, pos_label=1):
    """Return (precision, recall, thresholds) in sklearn's order: thresholds rising, ending at (P=1, R=0)."""
    tps, fps, thr, n_pos = _ranked_counts(y_true, scores, pos_label)
    if n_pos == 0:
        raise ValueError("PR curve needs at least one positive in y_true")
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return np.r_[precision[::-1], 1.0], np.r_[recall[::-1], 0.0], thr[::-1]


def average_precision(y_true, scores, pos_label=1) -> float:
    """PR-AUC as average precision: sum_n (R_n - R_{n-1}) P_n (sklearn's definition)."""
    precision, recall, _ = precision_recall_curve(y_true, scores, pos_label)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def classification_metrics(y_true, y_pred, scores=None, pos_label=1) -> dict:
    """Accuracy, precision, recall, F1 and (with scores) ROC-AUC and PR-AUC for Task B."""
    p, r, f1 = precision_recall_f1(y_true, y_pred, pos_label)
    out = {"Accuracy": accuracy(y_true, y_pred), "Precision": p, "Recall": r, "F1": f1}
    if scores is not None:
        out["ROC-AUC"] = roc_auc(y_true, scores, pos_label)
        out["PR-AUC"] = average_precision(y_true, scores, pos_label)
    return out


def scores_of(model, X):
    """Continuous score for the positive class: SVM margin, else P(class 1)."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)[:, -1]
    return None


# --- Uncertainty: k-fold cross-validation ----------------------------------
def kfold_indices(y, k=5, seed=SEED, stratify=False):
    """
    Yield (train_idx, val_idx) for k shuffled folds. With stratify=True each
    fold keeps the class proportions of y (use it for Task B).
    """
    n = len(y)
    if not 2 <= k <= n:
        raise ValueError(f"need 2 <= k <= n_samples, got k={k}, n={n}")
    rng = np.random.default_rng(seed)
    fold_of = np.empty(n, dtype=int)
    if stratify:
        y = np.asarray(y)
        offset = 0
        for label in np.unique(y):
            idx = rng.permutation(np.nonzero(y == label)[0])
            # deal each class round-robin, continuing where the last class stopped
            fold_of[idx] = (np.arange(idx.size) + offset) % k
            offset += idx.size
    else:
        fold_of[rng.permutation(n)] = np.arange(n) % k
    for f in range(k):
        yield np.nonzero(fold_of != f)[0], np.nonzero(fold_of == f)[0]


def _standardize(X_tr, X_val):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_tr - mu) / sd, (X_val - mu) / sd


def cross_validate(make_model, X, y, task, k=5, seed=SEED, scale=False,
                   inverse_transform=None, pos_label=1) -> dict:
    """
    k-fold CV of a fresh model per fold; returns {metric: (mean, std)}.

    make_model        : zero-argument callable returning an unfitted model with
                        fit/predict (and decision_function or predict_proba for
                        AUC), e.g. lambda: PegasosSVM(lambda_=1e-3).
    task              : "regression" (Task A) or "classification" (Task B,
                        stratified folds).
    scale             : standardize inside each fold with that fold's TRAIN
                        statistics (needed for the SVM; no leakage into the fold).
                        For sklearn baselines a Pipeline does the same.
    inverse_transform : see regression_metrics (e.g. np.exp for a log target).

    "fit time (s)" and "predict time (s)" are reported too (Problem 5c).
    Run it on the train+validation data only; the test set stays untouched.
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    per_fold: dict[str, list[float]] = {}
    for tr, val in kfold_indices(y, k, seed, stratify=task == "classification"):
        X_tr, X_val = (_standardize(X[tr], X[val]) if scale else (X[tr], X[val]))
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X_tr, y[tr])
        t1 = time.perf_counter()
        pred = model.predict(X_val)
        t2 = time.perf_counter()
        if task == "regression":
            fold = regression_metrics(y[val], pred, inverse_transform)
        else:
            fold = classification_metrics(y[val], pred, scores_of(model, X_val), pos_label)
        fold["fit time (s)"] = t1 - t0
        fold["predict time (s)"] = t2 - t1
        for name, value in fold.items():
            per_fold.setdefault(name, []).append(value)
    return {name: (float(np.mean(v)), float(np.std(v))) for name, v in per_fold.items()}


# --- Comparison helper ------------------------------------------------------
def _cell(value, precision):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, tuple):
        mean, std = value
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.{precision}f}"
    return str(value)


def format_table(name_to_metrics: dict, columns=None, fmt="text", precision=4,
                 index_name="Model") -> str:
    """
    Render {row name: {column: value}} as a table string.

    A value can be a number, a (mean, std) tuple (printed "mean ± std") or
    None / missing (printed "n/a", e.g. the SVM on Task A). fmt is "text",
    "markdown" or "latex" (booktabs rows, for report/report.tex). index_name
    heads the first column (e.g. "District" for an errors_by_group table).
    """
    if columns is None:
        columns = []
        for metrics in name_to_metrics.values():
            columns += [c for c in metrics if c not in columns]
    header = [index_name, *columns]
    rows = [[str(name), *(_cell(m.get(c), precision) for c in columns)]
            for name, m in name_to_metrics.items()]

    if fmt == "markdown":
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    if fmt == "latex":
        def tex(s):
            return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
                    .replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
                    .replace("±", r"$\pm$"))
        lines = [r"\begin{tabular}{@{}l" + "c" * len(columns) + "@{}}", r"\toprule",
                 " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
        lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)
    if fmt != "text":
        raise ValueError(f"fmt must be 'text', 'markdown' or 'latex', got {fmt!r}")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    def line(cells):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths, strict=True)))
    rule = "-" * len(line(header))
    return "\n".join([line(header), rule, *(line(r) for r in rows)])


def compare(name_to_metrics: dict, columns=None, title=None, precision=4,
            save_to=None, index_name="Model") -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.

    save_to: optional path; ".tex" writes the LaTeX tabular, ".md" Markdown,
    anything else plain text (e.g. report/results_table.tex for the report).
    """
    if title:
        print(f"\n== {title} ==")
    print(format_table(name_to_metrics, columns, "text", precision, index_name))
    if save_to is not None:
        path = Path(save_to)
        fmt = {".tex": "latex", ".md": "markdown"}.get(path.suffix, "text")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_table(name_to_metrics, columns, fmt, precision, index_name) + "\n",
                        encoding="utf-8")


# --- Model inspection -------------------------------------------------------
def feature_importance_table(importances, feature_names, top=15):
    """
    [(feature, share)] sorted by importance, normalised to sum to 1.

    importances: the tree's impurity-decrease importances (for each feature,
    the sum over its splits of N_node/N * impurity decrease), e.g.
    tree.feature_importances_.
    """
    imp = np.asarray(importances, dtype=float).ravel()
    if imp.size != len(feature_names):
        raise ValueError(f"{imp.size} importances for {len(feature_names)} feature names")
    total = imp.sum()
    if total > 0:
        imp = imp / total
    order = np.argsort(-imp, kind="mergesort")[:top]
    return [(feature_names[i], float(imp[i])) for i in order]


def svm_weight_table(w, feature_names, top=15):
    """[(feature, weight)] sorted by |weight|; sign = pushes toward premium (+) or standard (-).

    Only comparable across features when the SVM was trained on standardized X.
    """
    w = np.asarray(w, dtype=float).ravel()
    if w.size != len(feature_names):
        raise ValueError(f"{w.size} weights for {len(feature_names)} feature names")
    order = np.argsort(-np.abs(w), kind="mergesort")[:top]
    return [(feature_names[i], float(w[i])) for i in order]


def margin_summary(w, b, X, y, pos_label=1) -> dict:
    """||w||, margin width 2/||w|| and the support vectors (y_i (w.x_i + b) <= 1)."""
    w = np.asarray(w, dtype=float).ravel()
    y_pm = np.where(_binary(y, pos_label), 1.0, -1.0)
    margins = y_pm * (np.asarray(X, dtype=float) @ w + b)
    norm = float(np.linalg.norm(w))
    sv = np.nonzero(margins <= 1.0)[0]
    return {"||w||": norm, "margin width": 2.0 / norm if norm > 0 else float("inf"),
            "n support vectors": int(sv.size), "support vector idx": sv}


# --- Error analysis ---------------------------------------------------------
def top_errors(y_true, y_pred, k=10):
    """Indices of the k largest absolute errors, worst first."""
    y_true, y_pred = _pair(y_true, y_pred)
    return np.argsort(-np.abs(y_true - y_pred), kind="mergesort")[:k]


def errors_by_group(y_true, y_pred, groups, task="regression", min_count=1,
                    pos_label=1) -> dict:
    """
    Error broken down by a grouping column (district, building type, ...).

    Returns {group: metrics}, worst group first, ready for compare()/format_table.
    regression    : n, RMSE, MAE, bias (mean of pred - true; > 0 = overestimates)
    classification: n, error rate, FP, FN
    Groups with fewer than min_count rows are pooled into "(other)" — the
    sparse districts — so they still show up.
    """
    groups = np.asarray(groups).ravel()
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if not (len(groups) == len(y_true) == len(y_pred)):
        raise ValueError("y_true, y_pred and groups must have the same length")
    labels, counts = np.unique(groups, return_counts=True)
    rare = set(labels[counts < min_count].tolist())
    keys = np.array(["(other)" if g in rare else g for g in groups.tolist()], dtype=object)

    rows = {}
    for g in dict.fromkeys(keys.tolist()):
        m = keys == g
        if task == "regression":
            t, p = _pair(y_true[m], y_pred[m])
            rows[g] = {"n": int(m.sum()), "RMSE": rmse(t, p), "MAE": mae(t, p),
                       "bias": float(np.mean(p - t))}
        else:
            (_, fp), (fn, _) = confusion_matrix(y_true[m], y_pred[m], pos_label)
            rows[g] = {"n": int(m.sum()), "error rate": float(np.mean(y_true[m] != y_pred[m])),
                       "FP": int(fp), "FN": int(fn)}
    key = "RMSE" if task == "regression" else "error rate"
    return dict(sorted(rows.items(), key=lambda kv: -kv[1][key]))


def errors_by_bins(y_true, y_pred, by=None, n_bins=5, task="regression", pos_label=1) -> dict:
    """
    Error per quantile bin of `by` (default: y_true itself), in bin order.

    E.g. errors_by_bins(price_te, pred_te, n_bins=10) shows whether the top
    price decile (luxury listings) drives the RMSE; for Task B pass the price
    as `by` to see which price band gets misclassified.
    """
    by = np.asarray(y_true if by is None else by, dtype=float).ravel()
    edges = np.unique(np.quantile(by, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, by, side="right") - 1, 0, len(edges) - 2)
    names = [f"Q{i + 1} [{edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in range(len(edges) - 1)]
    rows = errors_by_group(y_true, y_pred, np.array(names, dtype=object)[idx], task,
                           pos_label=pos_label)
    return {name: rows[name] for name in names if name in rows}


# --- Figures (matplotlib imported lazily; files go to report/figures/) ------
def _plt():
    import matplotlib

    matplotlib.use("Agg")  # no display needed: CI and `python -m src.run_all`
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _plt().close(fig)
    return path


def plot_confusion_matrix(cm, path, title="Confusion matrix",
                          labels=("standard", "premium")):
    plt = _plt()
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, path)


def plot_roc_curves(name_to_scores: dict, y_true, path, title="ROC curve (test)"):
    """name_to_scores: {model name: scores on the same y_true}."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for name, s in name_to_scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path)


def plot_pr_curves(name_to_scores: dict, y_true, path, title="Precision-recall (test)"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for name, s in name_to_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, s)
        ax.step(rec, prec, where="post", label=f"{name} (AP {average_precision(y_true, s):.3f})")
    ax.axhline(np.mean(np.asarray(y_true) == 1), color="k", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower left")
    return _save(fig, path)


def plot_pred_vs_true(y_true, y_pred, path, title="Predicted vs true", label="log(price)"):
    plt = _plt()
    y_true, y_pred = _pair(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(title)
    return _save(fig, path)


def plot_bar(pairs, path, title, xlabel):
    """Horizontal bars from [(name, value)], e.g. feature_importance_table / svm_weight_table."""
    plt = _plt()
    names = [n for n, _ in pairs][::-1]
    values = np.array([v for _, v in pairs][::-1], dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 0.28 * len(pairs) + 1.0))
    ax.barh(names, values, color=np.where(values >= 0, "tab:blue", "tab:red"))
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _save(fig, path)
