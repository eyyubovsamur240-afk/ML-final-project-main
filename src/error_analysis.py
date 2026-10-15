"""
error_analysis.py — where and why the models fail (Problem 6).

All functions take the TEST rows (`test_df`, as returned by data_prep) and a
model's test predictions, and return small pandas tables ready to print or to
turn into report tables.

  Task A (log price):  errors by price band, by category, by how well a district
                       is covered in training; the worst individual listings
  Task B (tier):       error rates by category and by distance from the tier threshold
  Features:            grouped permutation importance (any model), top SVM weights

Nothing here is fitted on the test set: price bands are fixed AZN edges, and
district coverage is counted on the TRAINING rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate

# Fixed price bands (AZN). Defined up front, not learned from any split.
PRICE_BANDS = [0, 100_000, 200_000, 300_000, 500_000, 1_000_000, np.inf]
# Districts by number of TRAINING listings: "sparse" ones are where we expect trouble.
COVERAGE_BINS = [0, 30, 100, 300, np.inf]
COVERAGE_LABELS = ["< 30 (sparse)", "30-99", "100-299", ">= 300"]
# Distance from the tier threshold in log-price units (0.1 ~ 10% away).
MARGIN_BINS = [0.0, 0.05, 0.1, 0.25, 0.5, np.inf]
# One-hot columns are named "<feature>_<value>" (see data_prep.FeatureEncoder).
CATEGORICAL = ("category", "repair", "location")


def _band_labels(edges, unit=1_000, suffix="k"):
    def fmt(x):
        return "inf" if np.isinf(x) else f"{x / unit:,.0f}{suffix}"
    return [f"{fmt(lo)}-{fmt(hi)}" if not np.isinf(hi) else f">= {fmt(lo)}"
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


def price_band(price) -> pd.Categorical:
    """Fixed AZN band of each listing: 0-100k, 100k-200k, ... , >= 1,000k."""
    return pd.cut(np.asarray(price, dtype=float), PRICE_BANDS, right=False,
                  labels=_band_labels(PRICE_BANDS))


def district_coverage(test_locations, train_locations) -> pd.Categorical:
    """How many TRAINING listings each test row's district has, binned."""
    counts = pd.Series(train_locations).value_counts()
    n_train = pd.Series(test_locations).map(counts).fillna(0).to_numpy()
    return pd.cut(n_train, COVERAGE_BINS, right=False, labels=COVERAGE_LABELS)


# --- Task A -------------------------------------------------------------------
def regression_breakdown(groups, log_true, log_pred) -> pd.DataFrame:
    """Per group: n, RMSE and MAE on log(price), median % error, and the bias.

    bias = mean(log_pred - log_true): > 0 means the model over-prices the group.
    """
    log_true = np.asarray(log_true, dtype=float)
    log_pred = np.asarray(log_pred, dtype=float)
    frame = pd.DataFrame({"group": groups, "true": log_true, "pred": log_pred})
    rows = []
    for name, part in frame.groupby("group", observed=True, sort=True):
        ape = np.abs(np.exp(part["pred"]) - np.exp(part["true"])) / np.exp(part["true"])
        rows.append({
            "group": str(name),
            "n": len(part),
            "rmse_log": evaluate.rmse(part["true"], part["pred"]),
            "mae_log": evaluate.mae(part["true"], part["pred"]),
            "median_ape": float(np.median(ape)),
            "bias_log": float(np.mean(part["pred"] - part["true"])),
        })
    return pd.DataFrame(rows)


def worst_predictions(test_df: pd.DataFrame, log_true, log_pred, n: int = 10,
                      columns=("category", "location", "area_m2", "rooms")) -> pd.DataFrame:
    """The n listings with the largest |log error|, with a few descriptive columns."""
    err = np.asarray(log_pred, dtype=float) - np.asarray(log_true, dtype=float)
    top = np.argsort(-np.abs(err), kind="mergesort")[:n]
    keep = [c for c in columns if c in test_df.columns]
    out = test_df.iloc[top][keep].reset_index(drop=True)
    out["price"] = np.exp(np.asarray(log_true, dtype=float)[top]).round()
    out["predicted"] = np.exp(np.asarray(log_pred, dtype=float)[top]).round()
    out["error_pct"] = (100 * (out["predicted"] / out["price"] - 1)).round(1)
    return out


# --- Task B -------------------------------------------------------------------
def classification_breakdown(groups, y_true, y_pred) -> pd.DataFrame:
    """Per group: n, share premium, error rate, false-premium and missed-premium rates."""
    frame = pd.DataFrame({"group": groups, "true": np.asarray(y_true), "pred": np.asarray(y_pred)})
    rows = []
    for name, part in frame.groupby("group", observed=True, sort=True):
        t, p = part["true"].to_numpy() == 1, part["pred"].to_numpy() == 1
        rows.append({
            "group": str(name),
            "n": len(part),
            "premium_share": float(t.mean()),
            "error_rate": float(np.mean(t != p)),
            "false_premium_rate": float(np.mean(p[~t])) if (~t).any() else float("nan"),
            "missed_premium_rate": float(np.mean(~p[t])) if t.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def threshold_margin(price, threshold) -> pd.Categorical:
    """How far each listing's price is from the tier threshold, |log(price / threshold)|, binned."""
    dist = np.abs(np.log(np.asarray(price, dtype=float) / float(threshold)))
    labels = [f"{lo:g}-{hi:g}" if not np.isinf(hi) else f">= {lo:g}"
              for lo, hi in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:], strict=True)]
    return pd.cut(dist, MARGIN_BINS, right=False, labels=labels)


# --- Features -----------------------------------------------------------------
def feature_groups(feature_names, categorical=CATEGORICAL) -> dict[str, list[int]]:
    """Map each original feature to its column indices (all one-hot columns of a categorical together)."""
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        base = next((c for c in categorical if str(name).startswith(c + "_")), str(name))
        groups.setdefault(base, []).append(i)
    return groups


def permutation_importance(score, X, y, groups: dict[str, list[int]], n_repeats: int = 3,
                           seed: int = 42) -> pd.DataFrame:
    """Drop in score when a feature's columns are shuffled together (higher = more important).

    score(X, y) -> float where higher is better (e.g. -RMSE, or F1). The rows of
    a feature group are permuted jointly, so a one-hot encoded district counts
    as one feature. Works for any model, ours or scikit-learn's.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    base = float(score(X, y))
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - float(score(X_perm, y)))
        rows.append({"feature": name, "importance": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)


def grouped_importances(importances, groups: dict[str, list[int]]) -> pd.DataFrame:
    """Sum per-column importances (e.g. a tree's impurity decrease) into original features."""
    importances = np.asarray(importances, dtype=float)
    rows = [{"feature": name, "importance": float(importances[cols].sum())}
            for name, cols in groups.items()]
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)


def top_weights(weights, feature_names, k: int = 10) -> pd.DataFrame:
    """The k largest |w| of a linear model on standardized features (sign = towards premium)."""
    weights = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(-np.abs(weights), kind="mergesort")[:k]
    return pd.DataFrame({"feature": [str(feature_names[i]) for i in order],
                         "weight": weights[order]})
