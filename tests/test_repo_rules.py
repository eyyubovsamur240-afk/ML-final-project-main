"""Repo rules from the brief, checked on every `pytest` run (and in CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_golden_rule_allows_own_modules_but_follows_them(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "helpers.py").write_text("import pandas as pd\nfrom sklearn.metrics import f1_score\n")
    (src / "wrapper.py").write_text("from sklearn.svm import LinearSVC\n")
    monkeypatch.setattr(check_golden_rule, "SRC", src)
    monkeypatch.setattr(check_golden_rule, "ROOT", tmp_path)

    ok = src / "ok.py"
    ok.write_text("import numpy as np\nfrom .helpers import pd\nfrom src import helpers\n")
    assert check_golden_rule.find_violations(ok) == []

    sneaky = src / "sneaky.py"
    sneaky.write_text("import numpy as np\nfrom src.wrapper import LinearSVC\n")
    assert [line for line, _ in check_golden_rule.find_violations(sneaky)] == [2]

    # a wrapper hidden in a sub-package, reached through a relative import inside it
    (src / "helpers").mkdir()
    (src / "helpers" / "__init__.py").write_text("")
    (src / "helpers" / "svmwrap.py").write_text("from .base import LinearSVC\n")
    (src / "helpers" / "base.py").write_text("from sklearn.svm import LinearSVC\n")
    nested = src / "nested.py"
    nested.write_text("from .helpers.svmwrap import LinearSVC\n")
    assert [line for line, _ in check_golden_rule.find_violations(nested)] == [1]

    # a UTF-8 BOM (some Windows editors add one) is valid Python, not a syntax error
    (src / "bom.py").write_bytes(b"\xef\xbb\xbfimport numpy as np\n")
    uses_bom = src / "uses_bom.py"
    uses_bom.write_text("from .bom import np\n")
    assert check_golden_rule.find_violations(uses_bom) == []


def test_no_dataset_or_large_files_tracked():
    if not (check_no_data.ROOT / ".git").exists():
        pytest.skip("not a git checkout (e.g. a downloaded ZIP)")
    problems = {f: check_no_data.problems_for(f) for f in check_no_data.tracked_files()}
    assert {f: p for f, p in problems.items() if p} == {}
