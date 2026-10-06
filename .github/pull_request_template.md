<!-- Title: imperative and specific, e.g. "Add Gini/entropy split search to DecisionTree" -->

## What & why
<!-- 1–3 sentences. What does this change and why is it needed? -->

Closes #<!-- issue number -->

## How I checked it
<!-- Tests added/updated, numbers before → after, a plot, a toy example... -->

## Checklist
- [ ] CI is green (lint & repo rules, tests)
- [ ] Tests added or updated for new behaviour (`python -m pytest`)
- [ ] Tree / SVM code is our own NumPy (scikit-learn only in baselines)
- [ ] No test-set peeking: thresholds, scalers and hyperparameters come from train/validation only
- [ ] Randomness is seeded (uses `SEED` / `random_state`)
- [ ] No data, model dumps or large outputs committed
- [ ] If results changed: `python -m src.run_all` still runs end to end and README/report numbers are updated
- [ ] `contribution_report.md` updated if this is a significant piece of my work

## For the reviewer
<!-- Where should they look first? Anything you're unsure about?
     Reviewing = understanding: at the demo every member must be able to explain this code. -->
