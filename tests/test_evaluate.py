"""
Our NumPy metrics vs sklearn.metrics on random synthetic data, plus the
CV / table / error-analysis helpers. sklearn is used here only as the reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from src import evaluate as ev


def _regression(seed, n=300):
    rng = np.random.default_rng(seed)
    y = rng.lognormal(11.0, 0.6, size=n)  # heavy-tailed like real prices
    return y, y * rng.lognormal(0.0, 0.3, size=n)


def _classification(seed, n=300, ties=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    scores = rng.normal(size=n) + 1.2 * y
    if ties:
        scores = np.round(scores, 0)  # many tied scores
    return y, scores, (scores > 0.3).astype(int)


@pytest.mark.parametrize("seed", range(5))
def test_regression_metrics_match_sklearn(seed):
    y, pred = _regression(seed)
    assert ev.rmse(y, pred) == pytest.approx(np.sqrt(skm.mean_squared_error(y, pred)))
    assert ev.mae(y, pred) == pytest.approx(skm.mean_absolute_error(y, pred))
    assert ev.r2(y, pred) == pytest.approx(skm.r2_score(y, pred))


def test_regression_metrics_on_log_scale_report_price_too():
    y, pred = _regression(0)
    out = ev.regression_metrics(np.log(y), np.log(pred), inverse_transform=np.exp)
    assert out["RMSE"] == pytest.approx(ev.rmse(np.log(y), np.log(pred)))
    assert out["RMSE (price)"] == pytest.approx(ev.rmse(y, pred))
    assert out["R2 (price)"] == pytest.approx(skm.r2_score(y, pred))


def test_r2_constant_target():
    assert ev.r2([3, 3, 3], [3, 3, 3]) == 1.0
    assert ev.r2([3, 3, 3], [2, 3, 4]) == 0.0


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ties", [False, True])
def test_classification_metrics_match_sklearn(seed, ties):
    y, scores, pred = _classification(seed, ties=ties)
    assert ev.confusion_matrix(y, pred).tolist() == skm.confusion_matrix(y, pred).tolist()
    p, r, f1 = ev.precision_recall_f1(y, pred)
    assert p == pytest.approx(skm.precision_score(y, pred))
    assert r == pytest.approx(skm.recall_score(y, pred))
    assert f1 == pytest.approx(skm.f1_score(y, pred))
    assert ev.accuracy(y, pred) == pytest.approx(skm.accuracy_score(y, pred))
    assert ev.roc_auc(y, scores) == pytest.approx(skm.roc_auc_score(y, scores))
    assert ev.average_precision(y, scores) == pytest.approx(
        skm.average_precision_score(y, scores))


@pytest.mark.parametrize("ties", [False, True])
def test_curves_match_sklearn(ties):
    y, scores, _ = _classification(1, ties=ties)
    fpr, tpr, _ = ev.roc_curve(y, scores)
    assert np.trapz(tpr, fpr) == pytest.approx(skm.roc_auc_score(y, scores))
    sk_fpr, sk_tpr, _ = skm.roc_curve(y, scores, drop_intermediate=False)
    assert np.allclose(fpr, sk_fpr) and np.allclose(tpr, sk_tpr)
    prec, rec, thr = ev.precision_recall_curve(y, scores)
    sk_prec, sk_rec, sk_thr = skm.precision_recall_curve(y, scores)
    assert np.allclose(prec, sk_prec) and np.allclose(rec, sk_rec) and np.allclose(thr, sk_thr)


def test_zero_division_and_label_order():
    assert ev.precision_recall_f1([0, 0, 1], [0, 0, 0]) == (0.0, 0.0, 0.0)
    # scores only need to rank; a monotone transform leaves AUC unchanged
    y, s, _ = _classification(2)
    assert ev.roc_auc(y, s) == pytest.approx(ev.roc_auc(y, np.exp(s)))
    with pytest.raises(ValueError):
        ev.roc_auc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        ev.rmse([1, 2], [1, 2, 3])


@pytest.mark.parametrize("stratify", [False, True])
def test_kfold_partitions_and_stratifies(stratify):
    y = np.array([0] * 70 + [1] * 30)
    folds = list(ev.kfold_indices(y, k=5, seed=0, stratify=stratify))
    val = np.concatenate([v for _, v in folds])
    assert sorted(val.tolist()) == list(range(100)), "each row is validated exactly once"
    for tr, v in folds:
        assert not set(tr) & set(v)
        if stratify:
            assert y[v].sum() == 6, "30 positives over 5 folds"
    again = list(ev.kfold_indices(y, k=5, seed=0, stratify=stratify))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True))


def test_cross_validate_reports_mean_std_and_times():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    out = ev.cross_validate(lambda: LogisticRegression(), X, y, "classification", k=4,
                            scale=True)
    for key in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
                "fit time (s)", "predict time (s)"):
        mean, std = out[key]
        assert np.isfinite(mean) and std >= 0
    assert out["ROC-AUC"][0] > 0.95

    yr = X[:, 0] ** 2 + rng.normal(scale=0.1, size=200)
    out = ev.cross_validate(lambda: DecisionTreeRegressor(max_depth=4, random_state=0),
                            X, yr, "regression", k=4)
    assert out["R2"][0] > 0.7
    again = ev.cross_validate(lambda: DecisionTreeRegressor(max_depth=4, random_state=0),
                              X, yr, "regression", k=4)
    assert again["RMSE"] == out["RMSE"]


def test_format_table_text_markdown_latex(tmp_path, capsys):
    table = {
        "Ours: decision tree": {"RMSE": (0.31, 0.02), "F1": 0.8},
        "Ours: SVM (Pegasos)": {"RMSE": None, "F1": (0.82, 0.01)},
    }
    text = ev.format_table(table, precision=2)
    assert "0.31 ± 0.02" in text and "n/a" in text
    md = ev.format_table(table, fmt="markdown", precision=2)
    assert md.splitlines()[0] == "| Model | RMSE | F1 |"
    tex = ev.format_table(table, fmt="latex", precision=2)
    assert r"0.31 $\pm$ 0.02" in tex and r"\toprule" in tex
    ev.compare(table, title="Results", save_to=tmp_path / "results.tex")
    assert "Results" in capsys.readouterr().out
    assert (tmp_path / "results.tex").read_text(encoding="utf-8").startswith(r"\begin{tabular}")


def test_inspection_tables():
    names = ["area", "rooms", "floor"]
    imp = ev.feature_importance_table([2.0, 6.0, 2.0], names)
    assert imp[0] == ("rooms", pytest.approx(0.6))
    assert sum(v for _, v in imp) == pytest.approx(1.0)
    w = ev.svm_weight_table([0.5, -2.0, 1.0], names, top=2)
    assert w == [("rooms", -2.0), ("floor", 1.0)]

    X = np.array([[2.0], [0.5], [-0.5], [-3.0]])
    s = ev.margin_summary(np.array([1.0]), 0.0, X, np.array([1, 1, 0, 0]))
    assert s["margin width"] == pytest.approx(2.0)
    assert s["support vector idx"].tolist() == [1, 2]


def test_error_analysis():
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    assert ev.top_errors(y, pred, k=1).tolist() == [4]

    groups = np.array(["Yasamal", "Yasamal", "Nasimi", "Nasimi", "Sabail"])
    rows = ev.errors_by_group(y, pred, groups, min_count=2)
    assert list(rows) == ["(other)", "Yasamal", "Nasimi"], "worst first; rare pooled"
    assert rows["(other)"]["bias"] == pytest.approx(-90.0)

    bins = ev.errors_by_bins(y, pred, n_bins=2)
    assert list(bins)[-1].startswith("Q2") and list(bins.values())[-1]["RMSE"] > 0

    rows = ev.errors_by_group([0, 1, 1, 0], [1, 1, 0, 0], ["a", "a", "b", "b"],
                              task="classification")
    assert rows["a"] == {"n": 2, "error rate": 0.5, "FP": 1, "FN": 0}


def test_plots_write_files(tmp_path):
    y, scores, pred = _classification(0)
    paths = [
        ev.plot_confusion_matrix(ev.confusion_matrix(y, pred), tmp_path / "cm.png"),
        ev.plot_roc_curves({"a": scores, "b": -scores}, y, tmp_path / "roc.png"),
        ev.plot_pr_curves({"a": scores}, y, tmp_path / "pr.png"),
        ev.plot_pred_vs_true(*_regression(0), tmp_path / "pvt.png"),
        ev.plot_bar([("area", 0.7), ("rooms", -0.3)], tmp_path / "imp.png", "w", "weight"),
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
