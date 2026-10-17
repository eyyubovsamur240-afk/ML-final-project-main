"""Repo rules from the brief, checked on every `pytest` run (and in CI)."""

from __future__ import annotations

import shutil
import subprocess
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


def test_submission_check_wants_the_report_tables_committed(tmp_path: Path, monkeypatch):
    from tools import check_submission

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    (tmp_path / "report" / "tables").mkdir(parents=True)
    (tmp_path / "report" / "figures").mkdir()
    (tmp_path / "report" / "report.tex").write_text(
        "\\newcommand{\\generatedtable}[1]{\\IfFileExists{#1}{\\input{#1}}{}}\n"
        "\\graphicspath{{figures/}}\n"
        "% \\input{tables/old.tex} is commented out\n"
        "50\\% of rows \\\\% \\input{tables/also_old.tex} is a comment after a line break\n"
        "\\generatedtable{tables/headline.tex}\n"
        "\\input{tables/task_a}\n"                                  # LaTeX adds .tex
        "\\includegraphics[width=\\linewidth]{figures/eval_a}\n"     # ... and .pdf
        "\\includegraphics{eval_b.pdf}\n", encoding="utf-8")          # found via graphicspath
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_submission, "ROOT", tmp_path)

    def report_problems():
        errors, warnings = check_submission.check()
        return [p for p in errors + warnings if "report.tex" in p]

    files = {"tables/headline.tex": "\\begin{table}\\label{tab:results}\\end{table}\n",
             "tables/task_a.tex": "\\begin{table}\\label{tab:task-a}\\end{table}\n",
             "figures/eval_a.pdf": "%PDF", "figures/eval_b.pdf": "%PDF"}
    problems = report_problems()
    assert any("tables/headline.tex" in p and "eval_b.pdf" in p for p in problems)
    assert not any("old.tex" in p for p in problems), "commented-out includes are ignored"
    for rel, text in files.items():
        (tmp_path / "report" / rel).write_text(text, encoding="utf-8")
    assert any("not in the repo" in p for p in report_problems()), "generated, not committed"
    subprocess.run(["git", "add", "-f", *files], cwd=tmp_path / "report", check=True)
    assert report_problems() == [], "committed, and their \\label counts as the results table"


def test_submission_check_flags_placeholders_and_undefined_numbers(tmp_path: Path, monkeypatch):
    from tools import check_submission

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    (tmp_path / "report" / "tables").mkdir(parents=True)
    report = tmp_path / "report" / "report.tex"
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_submission, "ROOT", tmp_path)

    def report_problems():
        errors, _ = check_submission.check()
        return [p for p in errors if "report.tex" in p and "not in the repo" not in p]

    report.write_text("\\IfFileExists{tables/numbers.tex}{\\input{tables/numbers.tex}}{}\n"
                      "RMSE \\val{TreeARmse}, F1 \\val{SvmBFOne}.\n"
                      "% \\val{OldName} in a comment\n"
                      "\\textbf{[TO FILL after the full run: the takeaway.]}\n", encoding="utf-8")
    problems = report_problems()
    assert any("1 `[TO FILL" in p for p in problems)
    assert not any("numbers.tex` does not define" in p for p in problems), "no numbers yet"
    (tmp_path / "report" / "tables" / "numbers.tex").write_text(
        "\\defval{TreeARmse}{0.250}\n", encoding="utf-8")
    undefined = [p for p in report_problems() if "does not define" in p]
    assert len(undefined) == 1 and "SvmBFOne" in undefined[0] and "OldName" not in undefined[0]
    report.write_text(report.read_text(encoding="utf-8").replace(
        "\\textbf{[TO FILL after the full run: the takeaway.]}", "Done."), encoding="utf-8")
    (tmp_path / "report" / "tables" / "numbers.tex").write_text(
        "\\defval{TreeARmse}{0.250}\n\\defval{SvmBFOne}{0.810}\n", encoding="utf-8")
    assert report_problems() == []
