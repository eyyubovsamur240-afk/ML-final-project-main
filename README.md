# Pricing bina.az listings with a from-scratch decision tree and SVM

Our final project for MLE-AI-201 Machine Learning. On the **bina.az**
real-estate dataset we predict a listing's sale price (Task A, with our
decision tree) and whether it is premium or standard (Task B, with the tree
and our Pegasos SVM). Both models are written in NumPy and benchmarked
against scikit-learn. The brief is
`PROJECT_STATEMENT.pdf`; who did what is in `contribution_report.md`.

- **Released:** Sun 04 Oct 2026
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)** — two weeks, no late submissions.
- **Demo:** defended in the session after the deadline.

## Submission
One of us submits three things on Moodle for the team:
1. the **GitHub link** to this repo, tagged `v1.0-final`;
2. the **report PDF**, built from `report/report.tex` (IEEE two-column);
3. the **slides PDF**.

`contribution_report.md` stays in the repo root; it is checked at the demo.

## The two tasks
- **Task A — Regression:** predict `log(price)`; errors are also reported in AZN.
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## From scratch
The **decision tree** and the **SVM** are our own NumPy code; scikit-learn
appears only in the comparison baselines, and CI checks that
(`tools/check_golden_rule.py`). The SVM is trained with the **Pegasos**
sub-gradient method rather than a QP/SMO solver (`src/svm.py`: mini-batches,
suffix averaging, three step-size schedules, margin and support vectors), and its
non-linear version maps the features with **random Fourier features** first
(`src/rff.py`, an approximate RBF-kernel SVM).

## Repo layout
```
.
├── README.md               # this file
├── requirements.txt        # pinned versions
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/                    # our code (tree and SVM from scratch in NumPy)
│   ├── data_prep.py        #   loading, cleaning, split, tier label
│   ├── eda.py              #   data-exploration figures 1-4
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE)
│   ├── random_forest.py    #   random forest of our trees (bonus)
│   ├── tree_experiments.py #   tree tuning, comparison with scikit-learn, forest study
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient, mini-batches, averaging)
│   ├── rff.py              #   random Fourier features + Pegasos (non-linear SVM)
│   ├── svm_study.py        #   SVM tuning: lambda sweep, convergence, schedules, RFF grid
│   ├── evaluate.py         #   metrics, k-fold CV, bootstrap intervals, tables
│   ├── baselines.py        #   scikit-learn baselines with matched settings, trivial floors
│   ├── error_analysis.py   #   error breakdowns, feature importance
│   ├── plots.py            #   evaluation figures
│   └── run_all.py          #   ONE command reproduces every number/figure
├── eda.ipynb               # data exploration notebook
├── tests/                  # pytest suite (run by CI)
├── tools/                  # CI checks: golden rule, no data, submission readiness
├── results/                # metrics.json, results.md (generated, git-ignored)
├── report/                 # IEEE report (self-contained, no IEEEtran.cls)
│   ├── report.tex
│   ├── tables/             #   generated .tex tables (git-ignored until final)
│   └── figures/
└── presentation/           # slides PDF
```

## Reproducibility (graded)
- The packages we use, and scikit-learn's numeric dependencies, are pinned in
  `requirements.txt` (Python 3.11).
- Every random step is seeded (`data_prep.SEED = 42`), and a test
  (`tests/test_run_all.py`) checks that two runs give identical numbers apart
  from the timings.
- `python -m src.run_all` reproduces **every** headline number and figure
  from a clean checkout (see below).
- The raw data archive is never committed (see `.gitignore`).

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
python -m src.run_all            # full run (about an hour: the tree and SVM studies)
python -m src.run_all --fast     # smoke run: 3,000 training rows, 3 folds; writes to fast/ subfolders
```

`run_all` cleans and splits the data (seed 42), then runs the tree study
(`src/tree_experiments.py`): it tunes our tree's depth and leaf size (and Gini vs
entropy for Task B) on the validation split, compares it with scikit-learn's node
by node, and grows the random-forest bonus; its figures go to `report/figures/`.
The tuned settings are the ones the trees use in every table below. For Task A both
trees are fitted on log(price per m², or per sot for land plots) and the log area is
added back, so they still predict log(price) and every metric is on that scale.
`--no-tree-study` is a quicker check: it skips the study, uses the fixed fallback
settings in `run_all.py` on plain log(price) and writes to `no_tree_study/`
subfolders, so it never replaces the report's tables.
Next comes the SVM study (`src/svm_study.py`), on the training and validation splits
only: a λ sweep (validation F1, margin 2/‖w‖, support vectors), a convergence check
against LinearSVC's objective at the chosen λ (which picks the number of epochs), the
three step-size schedules, and the γ × λ grid and number of features of the RFF SVM.
Both SVMs in the tables use the settings it picks. `--no-svm-study` uses the fixed
`SVM` / `SVM_RFF` settings instead and writes to `no_svm_study/` subfolders.
It then trains every model on the training split. For each model it reports 5-fold cross-validation on the training
split (mean ± std), validation metrics, test metrics with 95% bootstrap intervals,
and fit/predict times:

| | Task A: log(price) | Task B: premium vs standard |
|---|---|---|
| Ours | decision tree | decision tree, Pegasos SVM, RFF SVM |
| scikit-learn, same hyperparameters | `DecisionTreeRegressor` | `DecisionTreeClassifier`, `SGDClassifier` (hinge), `LinearSVC`, `RBFSampler` + `SGDClassifier` |
| Floor | predict the training mean | predict the majority class |

It also measures how often our tree's test predictions match scikit-learn's,
fit time against training-set size, and errors broken down by price band,
property category, district and distance from the tier threshold. It then writes:

- `results/metrics.json`: every number (including per-fold scores, error breakdowns
  and the tree and SVM studies' validation grids and curves)
- `results/results.md`: the results, timing and scaling tables, how closely our tree
  matches scikit-learn's, the tree and SVM studies, the dataset summary and the error
  analysis, as text
- `report/tables/headline.tex` (every model on both tasks), `task_a.tex`, `task_b.tex`,
  `timing.tex`, `tree_study.tex` (tuned settings, ours vs scikit-learn, random forest),
  `errors.tex` (errors by category and district coverage), `worst.tex` (the listings
  our tree misprices most), `svm_study.tex`, `svm_convergence.tex`, `svm_rff.tex` (the
  SVM study), `data.tex` (cleaning steps and split sizes): tables for the report
  (`\input{tables/headline.tex}`); `numbers.tex`: every number the report's text
  quotes, as `\defval{Name}{value}` for `\val{Name}`
- `report/figures/eval_*.pdf`: predicted vs actual, error by price band and by district
  coverage, confusion matrices, ROC curves, error vs distance from the tier threshold,
  permutation importance, tree impurity importance, SVM weights, fit time vs
  training-set size
- `report/figures/tree_*.png`, `forest_*.png`: the tree study's validation grids and
  curves and the forest's score as trees are added (`src/tree_experiments.py`)
- `report/figures/svm_*.png`: the SVM study's λ sweep, objective curves and RFF grid
  and number of features (`src/svm_study.py`)
- `report/figures/fig1_*.png` to `fig4_*.png`: the data-exploration figures (`src/eda.py`)

These outputs are git-ignored: they are regenerated, never edited by hand. Once the
numbers are final, commit the files the report uses explicitly
(`git add -f report/tables/*.tex report/figures/<name>.pdf report/figures/<name>.png`,
see CONTRIBUTING.md; `python tools/check_submission.py` lists any that are missing) so
the report builds in CI. A model whose `fit` or `predict` is still a `TODO` stub is
listed as "not implemented yet" and skipped; if only `predict_proba` /
`decision_function` is a stub, the model is scored without an AUC (n/a in the tables).
The trees' final settings come from the tree study (printed, and saved in
`metrics.json` under `tree_study`); `TREE_REGRESSION` and `TREE_CLASSIFICATION` at the
top of `src/run_all.py` are only the `--no-tree-study` fallback. The same holds for
the SVMs: their settings come from the SVM study (`metrics.json` under `svm_study`),
and `SVM` / `SVM_RFF` are only the `--no-svm-study` fallback.

## Team workflow
How we branch, review, test and submit: **[CONTRIBUTING.md](CONTRIBUTING.md)**.
In short: branch → PR → green CI + one teammate's approval → squash-merge.

```bash
pip install -r requirements.txt -r requirements-dev.txt   # Python 3.11
python -m pytest                                          # tests (skipped = still TODO)
python tools/check_submission.py --warn-only              # what's left before v1.0-final
```
