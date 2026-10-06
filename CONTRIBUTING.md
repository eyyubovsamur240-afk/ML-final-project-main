# How we work

The team's working agreement for the MLE-AI-201 final project.
**Deadline: Sun 18 Oct 2026, 23:59 (Baku).** Everyone reads this once, fully.

> **The short version**
> 1. Never commit to `main`. Make a branch → push → open a pull request (PR).
> 2. Every PR needs **green CI** and **one approval from a teammate** before merging.
> 3. Keep PRs small and merge often: at least every 1–2 days, never one big dump at the end.
> 4. Your decision tree and SVM use **NumPy only**, and you **never commit the dataset**. CI checks both.
> 5. Commit with **your own GitHub email**. Your grade partly depends on Git history.
> 6. Review means *understand*: at the demo, anyone can be asked about any part.

---

## 1. One-time setup (every member)

**Python 3.11.** The pinned versions in `requirements.txt` (numpy 1.26.4 etc.)
have no ready-made wheels for Python 3.13+, so a newer Python will fail to install
them. Install 3.11 from python.org (or with `pyenv` or `uv`).

```bash
git clone https://github.com/eyyubovsamur240-afk/ML-final-project-main.git
cd ML-final-project-main

# create and activate a virtual environment
python3.11 -m venv .venv            # Windows: py -3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt -r requirements-dev.txt
pre-commit install                  # runs the CI checks locally on every commit
python -m pytest                    # should pass (most tests "skipped" = still TODO)
```

**Use the email from your GitHub account** so your commits are credited to you.
The brief grades contribution "by Git history":

```bash
git config --global user.name  "Name Surname"
git config --global user.email "the-email-on-your-github-account@example.com"
```

Check that it worked: on GitHub, your commits should show your avatar, not a grey icon.

**Dataset:** download it and put it in `data/`; see [`data/README.md`](data/README.md).
It is git-ignored and **must never be committed**.

## 2. Daily loop

```bash
git switch main && git pull                 # start from the latest main
git switch -c aysel/tree-split-search       # new branch: <your-name>/<topic>
# ... work, run tests ...
git add -p                                  # review what you're adding
git commit -m "Add Gini and entropy split search"
git push -u origin aysel/tree-split-search  # then open a PR on GitHub
```

When `main` moves while your branch is open, bring it in with
`git pull origin main` (a merge; no rebasing or force-pushing needed).

**Branches** are short-lived: one task, 1–2 days, then merged and deleted.
Name them `<name>/<topic>`, for example `murad/pegasos-fit` or `rustam/eda-plots`.

**Commits:** each one is small and does one thing. Write the subject line in the imperative mood
("Add …", "Fix …", "Refactor …"), under ~70 characters. Mention the issue number when
there is one (`Fix tier threshold leak (#12)`).

**Pair programming?** Credit your partner by ending the commit message with a
trailer line, so both of you show up in Git history:

```
Co-authored-by: Name Surname <their-github-email@example.com>
```

## 3. Pull requests and reviews

1. Open the PR early (as a **draft** if it isn't finished) so others can see what you're doing.
2. Fill in the PR template; link the issue (`Closes #12`).
3. Wait for **CI to be green**. If it's red, open the failed check and read the error;
   it usually says exactly what is wrong.
4. Ask a teammate for review. **Reviewers respond within 24 hours** (a deadline this close can't absorb slow reviews).
5. The **author** merges, using **"Squash and merge"**. Each PR then becomes one clean commit
   on `main`, credited to you, and the branch is deleted automatically.

**As a reviewer:** run the code if you can, ask "why" until you could explain it
at the demo, and check the project-specific traps:
- Is anything fitted on validation or test data (scaler, tier threshold, hyperparameters)?
- Are all random seeds fixed?
- Did a library model sneak into the tree or SVM?
- Do the numbers make sense? Too good is suspicious: look for leakage columns like `unit_price`.

Approve when it's correct and understandable, not perfect. Use "Request changes" only
for real problems; label taste-level suggestions as `nit:`.

## 4. Definition of done

A task is done when it is **merged to `main`**, with:
- tests for the new behaviour (or a clear reason why not), all green in CI;
- seeds fixed; no data or large outputs committed;
- `python -m src.run_all` still running end to end (once the pipeline exists);
- any numbers or figures it changes updated in the README and report.

Code that sits on a branch doesn't count; the brief explicitly penalises
"code committed hours before the deadline that was never integrated, reviewed, or understood".

## 5. What CI checks, and why

Every PR and every push to `main` runs these GitHub Actions workflows (`.github/workflows/`):

| Check | What it enforces | Brief's rule |
|---|---|---|
| **Lint & repo rules**: golden rule | `src/decision_tree.py` and `src/svm.py` import only NumPy and the standard library | "Library model as a core model" scores zero |
| **Lint & repo rules**: no data | no dataset, archives or model dumps, and no file over 5 MB | "do not commit the raw archive" |
| **Lint & repo rules**: ruff | real bugs (undefined names, syntax errors), not style | — |
| **Tests** | `pytest` on Python 3.11 with the pinned requirements | "clean, tested fit/predict" |
| Submission readiness *(info only)* | template text left in the report, empty contribution report, unpinned deps, TODO stubs | "Leftover template text", "missing contribution report" |
| Contribution summary *(on `main`)* | commits per author, shown on the run's Summary page | "under-contribution … by Git history" |
| Report PDF *(when `report/` changes)* | `report.tex` compiles; the PDF can be downloaded from the run's **Artifacts** | — |

**About the tests.** `tests/test_api_contract.py` contains contract tests written from the
docstrings in `src/`. While a function is still a TODO stub its tests show as
*skipped*. They switch on automatically once someone implements the function,
and must pass from then on. If you deliberately change an interface, update the
test in the same PR. Add your own tests next to it. The brief explicitly requires
one showing that your tree matches scikit-learn's splits on a tiny, fully determined example (Problem 2c).

If the golden-rule check flags a package that is *not* a model or optimiser
(unlikely), the team can agree to add it to `ALLOWED_THIRD_PARTY` in
`tools/check_golden_rule.py`. If you build bonus models from your own trees
(e.g. `src/ensemble.py`), add those files to `PROTECTED` in the same script.

Run everything locally before pushing:

```bash
python -m pytest
ruff check .
python tools/check_submission.py --warn-only
```

## 6. Planning: issues, owners, board

- Every piece of work is a **GitHub issue** (template "Task"), with **one owner** (Assignee) and a target date.
- Track them on a **GitHub Project board** (Todo → In progress → In review → Done).
  Create it from the repo's **Projects** tab.
- Each module has an **owner** and a **second person who reviews it**, so nothing is understood by only one member.
  Once the split is decided, record it in `.github/CODEOWNERS`.

A suggested split for 3–4 people (adjust it to your team; everyone writes some code *and* some report):

| Member | Owns | Reviews |
|---|---|---|
| A | P1 data cleaning, EDA, leakage, splits (`data_prep.py`) | evaluation |
| B | P2 decision tree, P4a tree studies | data |
| C | P3 Pegasos SVM, P4b SVM studies | tree |
| D (or shared) | P5 benchmark, P6 metrics and error analysis (`evaluate.py`, `run_all.py`) | SVM |
| everyone | the report section for what they built, plus slides | someone else's section |

## 7. Schedule

| When | Milestone |
|---|---|
| **Tue 6 Oct** | Repo and CI set up. Everyone has cloned, installed and run `pytest`. Issues created, owners assigned. |
| **Thu 8 Oct** | Data loads and is cleaned; leakage columns identified; split and tier label done (P1). |
| **Sun 11 Oct** *(course milestone)* | Tree fits and predicts; SVM trains with a **decreasing loss**; both pass a sanity test vs scikit-learn on a toy split. All merged to `main`. |
| **Wed 14 Oct** | Tuning, benchmark and error analysis done (P4–P6); `python -m src.run_all` reproduces every number and figure. **Feature freeze**: bonuses only if the core is solid. |
| **Fri 16 Oct** | **Code freeze**: bug fixes only. Report draft complete; slides drafted. |
| **Sat 17 Oct** | Report and slides final. Full rehearsal of the demo: each member explains someone *else's* module. |
| **Sun 18 Oct, by 18:00** | Tag `v1.0-final` and submit on Moodle, leaving ~6 hours of buffer before 23:59. |

**Daily check-in** (async, in the team chat, 2 minutes): *yesterday / today / blocked by*.
If you're blocked for more than half a day, say so; don't wait silently.
Record decisions (e.g. "we drop `unit_price` because …") in the issue or PR, so they can be cited in the report.

## 8. Report, slides and contribution report

- `report/report.tex` is edited through PRs like code; each member writes the section for what they built.
  CI compiles it, and you can download the PDF from the run's **Artifacts**.
- The report's numbers and figures must come from `python -m src.run_all` (outputs go to `report/figures/`).
  That folder is git-ignored by default; once figures are final, add them explicitly
  (`git add -f report/figures/<name>.pdf`) so the report compiles for everyone and in CI.
- Keep `contribution_report.md` **current every week**, not only at the end. It must match Git history
  (CI's contribution summary helps you check). Claims that contradict Git history are treated as an integrity issue.
- The report must include an **AI-assistance disclosure**. Note as you go where you used AI tools and for what.
  Whatever you use, you must understand and be able to explain the code.

## 9. Submitting (the last day)

1. Run `python tools/check_submission.py`; it must report no ❌.
2. From a fresh clone, follow the README exactly and run `python -m src.run_all`. Check that the numbers match the report.
3. Put the slides PDF in `presentation/` (and optionally the compiled report PDF in `report/`).
4. Tag the final commit on `main`:
   ```bash
   git switch main && git pull
   git tag -a v1.0-final -m "Final submission"
   git push origin v1.0-final
   ```
5. The **Release** workflow re-runs CI, builds the PDF, runs the strict readiness check and
   publishes a GitHub Release. If it fails, fix the problem on `main`, then move the tag:
   `git push origin :refs/tags/v1.0-final && git tag -d v1.0-final`, and tag again.
   Do this **only before the deadline**: never move the tag afterwards.
6. **One** member submits on Moodle: the repo link (tag `v1.0-final`), the report PDF and the slides PDF.
   Make sure the graders can access the repo (it's public, so the link is enough, unless
   the course says otherwise).

## 10. Repo settings (repo owner, once)

**Settings → Collaborators → Add people**: add every teammate with the **Write** role.

**Settings → General → Pull Requests:**
- allow **only "Squash merging"** (untick merge commits and rebase merging);
- tick **"Automatically delete head branches"**.

**Settings → Rules → Rulesets → New branch ruleset** (target: the default branch `main`, enforcement *Active*):
- **Restrict deletions** and **Block force pushes**;
- **Require a pull request before merging**, with 1 required approval;
- **Require status checks to pass**, adding `Lint & repo rules` and `Tests` (they appear in the list
  after CI has run once). Tick "Require branches to be up to date before merging" only if merge races become a problem.

This repository is **public**, so rulesets are enforced on the free plan, and GitHub Actions
minutes are unlimited. (If it is ever made private again, rulesets need **GitHub Pro** on the
owner's account, which is free with the [GitHub Student Developer Pack](https://education.github.com/pack).
Without Pro, the rules are not enforced, and only CI plus the `no-commit-to-branch` pre-commit hook remain.)

Because the repo is public, anyone can read it: never commit the dataset, credentials or personal data.
To keep your email out of future commits, turn on **Settings → Emails → Keep my email address private**
on your GitHub account and use the `…@users.noreply.github.com` address it shows as your `git config user.email`.
