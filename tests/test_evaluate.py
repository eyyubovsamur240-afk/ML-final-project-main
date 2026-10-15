"""Our from-scratch metrics agree with scikit-learn's, and the uncertainty helpers behave.

scikit-learn is used here only as a reference implementation to check against.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from src import evaluate


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    y = rng.normal(12.0, 0.8, size=300)
    return y, y + rng.normal(0.0, 0.3, size=300)


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    scores = y + rng.normal(0.0, 0.9, size=400)
    scores[::7] = np.round(scores[::7], 1)  # create ties on purpose
    return y, (scores > 0.5).astype(int), scores


def test_regression_metrics_match_sklearn(reg_data):
    y, pred = reg_data
    assert evaluate.rmse(y, pred) == pytest.approx(np.sqrt(skm.mean_squared_error(y, pred)))
    assert evaluate.mae(y, pred) == pytest.approx(skm.mean_absolute_error(y, pred))
    assert evaluate.r2(y, pred) == pytest.approx(skm.r2_score(y, pred))


def test_r2_constant_target_follows_sklearn_convention():
    y = np.full(5, 3.0)
    assert evaluate.r2(y, y) == 1.0
    assert evaluate.r2(y, y + 1) == 0.0


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate.rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        evaluate.confusion_matrix([0, 1], [0])


def test_price_metrics_report_both_scales():
    log_true = np.log(np.array([100_000.0, 200_000.0, 400_000.0]))
    log_pred = np.log(np.array([110_000.0, 180_000.0, 400_000.0]))
    m = evaluate.price_metrics(log_true, log_pred)
    assert m["mae_azn"] == pytest.approx((10_000 + 20_000 + 0) / 3)
    assert m["median_ape"] == pytest.approx(0.1)  # errors are 10%, 10%, 0%
    assert m["rmse_log"] == pytest.approx(evaluate.rmse(log_true, log_pred))


def test_classification_metrics_match_sklearn(clf_data):
    y, pred, scores = clf_data
    assert evaluate.confusion_matrix(y, pred).tolist() == skm.confusion_matrix(y, pred).tolist()
    p, r, f1 = evaluate.precision_recall_f1(y, pred)
    assert p == pytest.approx(skm.precision_score(y, pred))
    assert r == pytest.approx(skm.recall_score(y, pred))
    assert f1 == pytest.approx(skm.f1_score(y, pred))
    assert evaluate.accuracy(y, pred) == pytest.approx(skm.accuracy_score(y, pred))
    assert evaluate.roc_auc(y, scores) == pytest.approx(skm.roc_auc_score(y, scores))


def test_precision_is_zero_without_positive_predictions():
    assert evaluate.precision_recall_f1([0, 1, 1], [0, 0, 0]) == (0.0, 0.0, 0.0)


def test_roc_auc_is_nan_for_a_single_class():
    assert np.isnan(evaluate.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_roc_curve_matches_sklearn(clf_data):
    y, _, scores = clf_data
    fpr, tpr, _ = evaluate.roc_curve(y, scores)
    sk_fpr, sk_tpr, _ = skm.roc_curve(y, scores, drop_intermediate=False)
    assert np.allclose(fpr, sk_fpr) and np.allclose(tpr, sk_tpr)
    assert np.trapz(tpr, fpr) == pytest.approx(evaluate.roc_auc(y, scores))


def test_classification_metrics_bundle(clf_data):
    y, pred, scores = clf_data
    m = evaluate.classification_metrics(y, pred, scores)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y)
    assert set(m) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert "roc_auc" not in evaluate.classification_metrics(y, pred)


@pytest.mark.parametrize("stratify", [False, True])
def test_kfold_partitions_every_row_once(stratify):
    labels = np.array([0] * 70 + [1] * 30)
    folds = evaluate.kfold_indices(100, k=5, seed=3, stratify=labels if stratify else None)
    val = np.concatenate([v for _, v in folds])
    assert sorted(val.tolist()) == list(range(100)), "each row is validated exactly once"
    for train, v in folds:
        assert len(np.intersect1d(train, v)) == 0
        assert len(train) + len(v) == 100
        assert len(v) in (20, 21, 19)
        if stratify:
            assert labels[v].sum() == 6, "every fold keeps the 70/30 class mix"
    again = evaluate.kfold_indices(100, k=5, seed=3, stratify=labels if stratify else None)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True))


def test_kfold_rejects_bad_k():
    with pytest.raises(ValueError):
        evaluate.kfold_indices(10, k=1)
    with pytest.raises(ValueError):
        evaluate.kfold_indices(3, k=5)


class _MeanModel:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_cross_validate_reports_mean_std_and_times(reg_data):
    y, _ = reg_data
    X = np.zeros((len(y), 1))
    seen = []

    def preprocess(X_fit, X_val):
        seen.append((len(X_fit), len(X_val)))
        return X_fit, X_val

    cv = evaluate.cross_validate(
        _MeanModel, X, y, lambda m, Xv, yv: evaluate.regression_metrics(yv, m.predict(Xv)),
        k=4, seed=0, preprocess=preprocess,
    )
    assert len(cv["rmse"]["folds"]) == 4
    assert cv["rmse"]["mean"] == pytest.approx(np.mean(cv["rmse"]["folds"]))
    assert cv["rmse"]["std"] == pytest.approx(np.std(cv["rmse"]["folds"], ddof=1))
    assert cv["fit_seconds"]["mean"] >= 0
    assert seen == [(225, 75)] * 4, "preprocess is fitted per fold on the fold's training rows"


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible(clf_data):
    y, pred, _ = clf_data
    f1 = lambda a, b: evaluate.precision_recall_f1(a, b)[2]  # noqa: E731
    ci = evaluate.bootstrap_ci(f1, y, pred, n_boot=200, seed=0)
    assert ci["lo"] <= ci["value"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.2
    assert ci == evaluate.bootstrap_ci(f1, y, pred, n_boot=200, seed=0)


def test_tables_format_mean_std_and_missing_values():
    rows = [{"model": "Ours: tree_1", "rmse": {"mean": 0.2346, "std": 0.0123}, "f1": None},
            {"model": "sklearn", "rmse": 0.25, "f1": 0.9}]
    text = evaluate.format_table(rows, ["model", "rmse", "f1"])
    assert "0.235 ± 0.012" in text and "n/a" in text
    header, rule, first, second = text.splitlines()
    assert first.startswith("Ours: tree_1") and second.startswith("sklearn ")
    assert first.endswith("n/a") and second.endswith("0.900"), "numbers are right-aligned"
    assert evaluate.format_cell(-0.0001) == "0.000", "no negative zero"
    tex = evaluate.latex_table(rows, ["model", "rmse", "f1"], ["Model", "RMSE", "F1"],
                               caption="Results", label="tab:x")
    assert r"0.235 $\pm$ 0.012" in tex and r"Ours: tree\_1" in tex and r"\label{tab:x}" in tex


def test_compare_prints_one_row_per_model(capsys):
    table = evaluate.compare({"A": {"rmse": 1.0}, "B": {"rmse": 2.0, "mae": 1.5}})
    printed = capsys.readouterr().out
    assert table in printed
    assert printed.count("\n") >= 4 and "mae" in table


def test_bad_inputs_raise_instead_of_giving_a_plausible_number():
    y = [0, 1, 1, 0]
    with pytest.raises(ValueError, match="NaN"):
        evaluate.roc_auc(y, [np.nan, 0.9, 0.8, 0.3])
    with pytest.raises(ValueError, match="NaN"):
        evaluate.roc_curve(y, [np.nan, 0.9, 0.8, 0.3])
    with pytest.raises(ValueError, match="inf"):
        evaluate.roc_curve(y, [np.inf, np.inf, 0.8, -np.inf])
    with pytest.raises(ValueError, match="length"):
        evaluate.roc_curve(y, [0.1, 0.9])
    with pytest.raises(ValueError, match="NaN"):
        evaluate.kfold_indices(6, k=2, stratify=[0, 1, np.nan, 0, 1, 1])


def test_model_scores_is_none_while_the_score_method_is_a_stub():
    class Stub:
        def predict_proba(self, X):
            raise NotImplementedError("TODO: class shares")

    class Broken:
        def predict_proba(self, X):
            raise NotImplementedError("a real error, not a TODO stub")

    assert evaluate.model_scores(Stub(), np.zeros((2, 1))) is None
    with pytest.raises(NotImplementedError):
        evaluate.model_scores(Broken(), np.zeros((2, 1)))
