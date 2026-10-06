"""Repo rules from the brief, checked on every `pytest` run (and in CI)."""

from __future__ import annotations

from pathlib import Path

from tools import check_golden_rule, check_no_data


def test_golden_rule_holds_for_protected_files():
    for rel in check_golden_rule.PROTECTED:
        path = check_golden_rule.ROOT / rel
        assert path.exists(), f"{rel} is missing; update PROTECTED in tools/check_golden_rule.py"
        assert check_golden_rule.find_violations(path) == [], f"{rel} breaks the golden rule"


def test_golden_rule_guard_catches_library_models(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import numpy as np\n"
        "from sklearn.svm import SVC\n"
        "import scipy.optimize\n"
        "import importlib\n"
        "m = importlib.import_module('sklearn.tree')\n"
        "from .decision_tree import DecisionTree\n",
        encoding="utf-8",
    )
    lines = [line for line, _ in check_golden_rule.find_violations(bad)]
    assert lines == [2, 3, 5]


def test_no_dataset_or_large_files_tracked():
    problems = {f: check_no_data.problems_for(f) for f in check_no_data.tracked_files()}
    assert {f: p for f, p in problems.items() if p} == {}
