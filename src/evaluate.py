"""
evaluate.py — metrics, uncertainty and comparison helpers (NumPy only).

Every metric here is written from scratch. scikit-learn appears only in the
tests (tests/test_evaluate.py), to check these numbers against its own.

    1. Regression metrics (Task A)       rmse, mae, r2, regression_metrics, price_metrics
    2. Classification metrics (Task B)   confusion_matrix, precision_recall_f1, roc_auc, ...
    3. Uncertainty                       kfold_indices, cross_validate, bootstrap_ci
    4. Tables                            format_table, compare, latex_table

Conventions: the positive class is 1 ("premium"); a model's "scores" are its
decision_function (SVM margins) or predict_proba[:, 1] (tree), so higher means
"more premium".
"""

from __future__ import annotations

import math
import time

import numpy as np


def _pair(y_true, y_pred):
    """Both inputs as flat float arrays of the same length."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {len(y_true)} true values vs {len(y_pred)} predictions")
    if len(y_true) == 0:
        raise ValueError("cannot compute a metric on zero samples")
    return y_true, y_pred


# --- 1. Regression (Task A: price) ------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """1 - SS_res / SS_tot. A constant target gives 1.0 if predicted exactly, else 0.0."""
    y_true, y_pred = _pair(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def price_metrics(log_true, log_pred) -> dict:
    """Task A metrics on both scales.

    The models are trained on log(price), so the log-scale numbers are the ones
    the models optimise. The AZN numbers (after exp) are easier to read:
    median_ape is the median absolute percentage error, e.g. 0.15 = "half the
    listings are priced within 15%".
    """
    log_true, log_pred = _pair(log_true, log_pred)
    price, price_hat = np.exp(log_true), np.exp(log_pred)
    return {
        "rmse_log": rmse(log_true, log_pred),
        "mae_log": mae(log_true, log_pred),
        "r2_log": r2(log_true, log_pred),
        "rmse_azn": rmse(price, price_hat),
        "mae_azn": mae(price, price_hat),
        "r2_azn": r2(price, price_hat),
        "median_ape": float(np.median(np.abs(price_hat - price) / price)),
    }


# --- 2. Classification (Task B: price tier) ---------------------------------
def _labels(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions")
    return y_true == positive, y_pred == positive


def confusion_matrix(y_true, y_pred, positive=1):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]] (rows = truth, columns = prediction)."""
    t, p = _labels(y_true, y_pred, positive)
    tn = int(np.sum(~t & ~p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tp = int(np.sum(t & p))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions")
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, positive=1):
    """Return (precision, recall, f1) for the positive (premium) class.

    A ratio with a zero denominator (e.g. no positive predictions) is 0.0,
    the same convention as scikit-learn's zero_division=0.
    """
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred, positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _average_ranks(x) -> np.ndarray:
    """Ranks 1..n of x; tied values share the average of the ranks they span."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    starts_run = np.r_[True, sorted_x[1:] != sorted_x[:-1]]   # first element of each tie run
    run_id = np.cumsum(starts_run) - 1
    run_start = np.flatnonzero(starts_run)                     # 0-based positions
    run_end = np.r_[run_start[1:], len(x)]                     # exclusive
    run_rank = (run_start + 1 + run_end) / 2.0                 # mean of ranks start+1 .. end
    ranks = np.empty(len(x))
    ranks[order] = run_rank[run_id]
    return ranks


def roc_auc(y_true, scores, positive=1) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney): the probability that a random premium listing
    scores higher than a random standard one, ties counting one half.
    Returns nan when only one class is present (AUC is undefined then).
    """
    y_true = np.asarray(y_true).ravel() == positive
    scores = np.asarray(scores, dtype=float).ravel()
    if y_true.shape != scores.shape:
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(scores)} scores")
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = _average_ranks(scores)[y_true].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y_true, scores, positive=1):
    """Return (fpr, tpr, thresholds), one point per distinct score, from (0, 0) to (1, 1)."""
    y_true = np.asarray(y_true).ravel() == positive
    scores = np.asarray(scores, dtype=float).ravel()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve needs both classes in y_true")
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], y_true[order]
    last_of_run = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]   # last index of each tie run
    tps = np.cumsum(t)[last_of_run]
    fps = (last_of_run + 1) - tps
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    thresholds = np.r_[np.inf, s[last_of_run]]
    return fpr, tpr, thresholds


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    """Accuracy, precision/recall/F1 for premium, the confusion counts and (with scores) ROC-AUC."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if scores is not None:
        out["roc_auc"] = roc_auc(y_true, scores)
    return out


def model_scores(model, X):
    """Continuous "how premium" scores: decision_function, else predict_proba[:, 1], else None."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        return proba[:, -1] if proba.ndim == 2 else proba.ravel()
    return None


# --- 3. Uncertainty ----------------------------------------------------------
def kfold_indices(n: int, k: int = 5, seed: int = 42, stratify=None):
    """Shuffled k-fold split of range(n): a list of (train_idx, val_idx) pairs.

    With `stratify` (one label per row), every fold gets the same class mix:
    each class is shuffled and dealt round-robin over the folds. Fold sizes
    differ by at most one.
    """
    if not 2 <= k <= n:
        raise ValueError(f"need 2 <= k <= n, got k={k}, n={n}")
    rng = np.random.default_rng(seed)
    groups = np.zeros(n, dtype=int) if stratify is None else np.asarray(stratify).ravel()
    if len(groups) != n:
        raise ValueError("stratify must have one label per row")
    fold_of = np.empty(n, dtype=int)
    offset = 0  # continue the round-robin across classes so totals stay balanced
    for value in np.unique(groups):
        rows = rng.permutation(np.flatnonzero(groups == value))
        fold_of[rows] = (offset + np.arange(len(rows))) % k
        offset += len(rows)
    every_row = np.arange(n)
    return [(every_row[fold_of != f], every_row[fold_of == f]) for f in range(k)]


def mean_std(values) -> dict:
    """{"mean", "std"} of a list of fold scores (sample std, ddof=1; 0.0 for one value)."""
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(np.mean(values)), "std": std}


def cross_validate(make_model, X, y, score, k: int = 5, seed: int = 42, stratify=None,
                   preprocess=None) -> dict:
    """k-fold cross-validation of a fresh model per fold.

    make_model()                  -> an unfitted model with fit/predict
    score(model, X_val, y_val)    -> dict of metric name -> float
    preprocess(X_fit, X_val)      -> (X_fit, X_val), fitted on X_fit only
                                     (e.g. standardization, so no fold leaks into another)

    Returns {metric: {"mean", "std", "folds"}} plus fit/predict seconds per fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    per_metric: dict[str, list] = {}
    fit_s, predict_s = [], []
    for train_idx, val_idx in kfold_indices(len(y), k=k, seed=seed, stratify=stratify):
        X_fit, X_val = X[train_idx], X[val_idx]
        if preprocess is not None:
            X_fit, X_val = preprocess(X_fit, X_val)
        model = make_model()
        start = time.perf_counter()
        model.fit(X_fit, y[train_idx])
        fit_s.append(time.perf_counter() - start)
        start = time.perf_counter()
        fold_scores = score(model, X_val, y[val_idx])
        predict_s.append(time.perf_counter() - start)
        for name, value in fold_scores.items():
            per_metric.setdefault(name, []).append(float(value))
    out = {name: {**mean_std(values), "folds": values} for name, values in per_metric.items()}
    out["fit_seconds"] = {**mean_std(fit_s), "folds": fit_s}
    out["score_seconds"] = {**mean_std(predict_s), "folds": predict_s}
    return out


def bootstrap_ci(metric, *arrays, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Point estimate and percentile bootstrap interval of metric(*arrays).

    The rows of all arrays are resampled together (with replacement), so
    bootstrap_ci(f1, y_test, y_pred) gives a 95% interval for the test F1.
    Resamples where the metric is undefined (nan) are ignored.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same length")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric(*(a[idx] for a in arrays))
    stats = stats[~np.isnan(stats)]
    lo, hi = (np.quantile(stats, [alpha / 2, 1 - alpha / 2]) if len(stats) else (np.nan, np.nan))
    return {"value": float(metric(*arrays)), "lo": float(lo), "hi": float(hi),
            "level": 1 - alpha, "n_boot": int(n_boot)}


def timed(fn, *args, **kwargs):
    """Run fn and return (its result, wall-clock seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


# --- 4. Tables -----------------------------------------------------------------
def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_cell(value, digits: int = 3, latex: bool = False) -> str:
    """One table cell: numbers rounded, {"mean","std"} as mean ± std, {"value","lo","hi"} as value [lo, hi]."""
    pm = r" $\pm$ " if latex else " ± "
    if _is_missing(value):
        return "n/a"
    if isinstance(value, dict) and "mean" in value:
        return f"{value['mean']:.{digits}f}{pm}{value['std']:.{digits}f}"
    if isinstance(value, dict) and "value" in value:
        return f"{value['value']:.{digits}f} [{value['lo']:.{digits}f}, {value['hi']:.{digits}f}]"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    text = str(value)
    return latex_escape(text) if latex else text


def format_table(rows: list[dict], columns: list[str], headers: list[str] | None = None,
                 title: str = "", digits: int = 3) -> str:
    """Plain-text table: one row per dict, one column per key in `columns`."""
    headers = headers or columns
    body = [[format_cell(row.get(col), digits) for col in columns] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join("-" * w for w in widths)
    lines = [title] if title else []
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    lines.append(line)
    for r in body:
        cells = [r[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(r[1:], widths[1:], strict=True)]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def compare(name_to_metrics: dict, columns: list[str] | None = None, title: str = "",
            digits: int = 3) -> str:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.

    name_to_metrics = {"Ours: decision tree": {"rmse": 0.23, ...}, ...}.
    Prints the table and returns it as a string.
    """
    if columns is None:
        columns = []
        for metrics in name_to_metrics.values():
            columns += [c for c in metrics if c not in columns]
    rows = [{"model": name, **metrics} for name, metrics in name_to_metrics.items()]
    table = format_table(rows, ["model", *columns], title=title, digits=digits)
    print(table)
    return table


_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                  "^": r"\textasciicircum{}"}


def latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(text))


def latex_table(rows: list[dict], columns: list[str], headers: list[str], caption: str,
                label: str, digits: int = 3, align: str | None = None) -> str:
    """A booktabs table (same style as report/report.tex) ready for \\input{...}."""
    align = align or "l" + "c" * (len(columns) - 1)
    out = [r"\begin{table}[t]", r"  \centering", rf"  \caption{{{caption}}}",
           rf"  \label{{{label}}}", r"  \footnotesize",
           rf"  \begin{{tabular}}{{@{{}}{align}@{{}}}}", r"    \toprule",
           "    " + " & ".join(headers) + r" \\", r"    \midrule"]
    for row in rows:
        out.append("    " + " & ".join(format_cell(row.get(c), digits, latex=True) for c in columns)
                   + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"
