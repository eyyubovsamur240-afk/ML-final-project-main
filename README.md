# MLE-AI-201 Machine Learning — Final Project (Starter Pack)

Build **two models from scratch** (a decision tree and an SVM) on the
**bina.az** real-estate dataset, then benchmark honestly against
scikit-learn. Read the full brief in the project statement PDF handed out
with this pack.

- **Released:** Sun 04 Oct 2026
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)** — two weeks, no late submissions.
- **Demo:** defended in the session after the deadline.

## What you submit (on Moodle)
Exactly **three** things; **one member submits** for the whole team:
1. **GitHub link** to your repo (tagged `v1.0-final`).
2. **Report PDF** (IEEE two-column — use `report/report.tex`).
3. **Slides PDF**.

Keep `contribution_report.md` in the repo root (checked at the demo).

## The two tasks
- **Task A — Regression:** predict `price` (consider `log(price)`).
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## Golden rule
Your **decision tree** and your **SVM** must be **your own NumPy code**.
scikit-learn is for the comparison baselines (and the optional bonuses)
only. For the SVM you do **not** need a QP/SMO solver — use the
**Pegasos** sub-gradient method (see the statement and `src/svm.py`).

## Repo layout
```
.
├── README.md               # this file
├── requirements.txt        # pin your versions
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/                    # YOUR from-scratch code
│   ├── data_prep.py        #   loading, cleaning, split, tier label
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE)
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient)
│   ├── evaluate.py         #   metrics, k-fold CV, bootstrap intervals, tables
│   ├── baselines.py        #   scikit-learn baselines with matched settings, trivial floors
│   ├── error_analysis.py   #   error breakdowns, feature importance
│   ├── plots.py            #   evaluation figures
│   └── run_all.py          #   ONE command reproduces every number/figure
├── tests/                  # pytest suite (run by CI)
├── tools/                  # CI checks: golden rule, no data, submission readiness
├── results/                # metrics.json, results.md (generated, git-ignored)
├── report/                 # IEEE report (self-contained, no IEEEtran.cls)
│   ├── report.tex
│   ├── tables/             #   generated .tex tables (git-ignored until final)
│   └── figures/
└── presentation/           # your slides PDF
```

## Reproducibility (graded)
- Pin dependencies in `requirements.txt`; fix all random seeds.
- `python -m src.run_all` must reproduce **every** headline number and
  figure from a clean checkout.
- Do **not** commit the raw data archive (see `.gitignore`).

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.11, see CONTRIBUTING.md
pip install -r requirements.txt
# put the dataset in data/ (see data/README.md), then:
python -m src.run_all
```

## Reproducing the results
One command produces every evaluation number, table and figure from a clean checkout:

```bash
# 1. the dataset (public on Kaggle; no account needed for this download link)
curl -L -o data/binaaz.zip \
  https://www.kaggle.com/api/v1/datasets/download/sehriyarmemmedli/binaaz-sale-project
unzip -o data/binaaz.zip -d data/            # -> data/house_sale.csv

# 2. everything else
python -m src.run_all            # full run (a few minutes)
python -m src.run_all --fast     # smoke run: 3,000 training rows, 3 folds; writes to fast/ subfolders
```

`run_all` cleans and splits the data (seed 42) and trains every model on the
training split. For each model it reports 5-fold cross-validation on the training
split (mean ± std), validation metrics, test metrics with 95% bootstrap intervals,
and fit/predict times:

| | Task A: log(price) | Task B: premium vs standard |
|---|---|---|
| Ours | decision tree | decision tree, Pegasos SVM |
| scikit-learn, same hyperparameters | `DecisionTreeRegressor` | `DecisionTreeClassifier`, `SGDClassifier` (hinge), `LinearSVC` |
| Floor | predict the training mean | predict the majority class |

It also measures how often our tree's test predictions match scikit-learn's,
fit time against training-set size, and errors broken down by price band,
district and distance from the tier threshold. It then writes:

- `results/metrics.json`: every number (including per-fold scores and error breakdowns)
- `results/results.md`: the results and timing tables as text
- `report/tables/task_a.tex`, `task_b.tex`, `timing.tex`: tables for the report
  (`\input{tables/task_a.tex}`)
- `report/figures/eval_*.pdf`: predicted vs actual, error by price band and by district
  coverage, confusion matrices, ROC curves, error vs distance from the tier threshold,
  permutation importance, tree impurity importance, SVM weights, fit time vs
  training-set size
- `report/figures/fig1-4`: the data-exploration figures, once `src/eda.py` is merged

These outputs are git-ignored: they are regenerated, never edited by hand. Once the
numbers are final, commit the report's tables and figures explicitly
(`git add -f report/tables/*.tex report/figures/<name>.pdf`, see CONTRIBUTING.md) so
the report builds in CI. A model whose `fit` or `predict` is still a `TODO` stub is
listed as "not implemented yet" and skipped; if only `predict_proba` /
`decision_function` is a stub, the model is scored without an AUC (n/a in the tables). The final hyperparameters live at the top of
`src/run_all.py` (`TREE_REGRESSION`, `TREE_CLASSIFICATION`, `SVM`) and are chosen on
the validation split only.

## Team workflow
How we branch, review, test and submit: **[CONTRIBUTING.md](CONTRIBUTING.md)**.
In short: branch → PR → green CI + one teammate's approval → squash-merge.

```bash
pip install -r requirements.txt -r requirements-dev.txt   # Python 3.11
python -m pytest                                          # tests (skipped = still TODO)
python tools/check_submission.py --warn-only              # what's left before v1.0-final
```
