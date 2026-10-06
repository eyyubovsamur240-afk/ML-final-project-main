"""
check_submission.py — catch the brief's automatic deductions before graders do.

Looks for the things the project statement penalises that a script can see:
  - leftover template text in report/report.tex,
  - an unfilled contribution_report.md,
  - unpinned dependencies in requirements.txt,
  - TODO stubs (`raise NotImplementedError("TODO ...")`) still left in src/,
  - no results table in the report,
  - no slides PDF in presentation/ (warning only: the PDF is also on Moodle).

Run from the repo root:
    python tools/check_submission.py              # strict: exit 1 on any error
    python tools/check_submission.py --warn-only  # always exit 0 (used on every CI run)

CI appends the Markdown output to the job summary.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Phrases that only exist in the untouched starter templates.
REPORT_TEMPLATE_MARKERS = [
    "Your Project Title Here",
    "A Clear, Specific Subtitle",
    r"Aliyeva, Aysel\quad Mammadov, Murad",
    "github.com/your-team/your-repo",
    "One paragraph, roughly 150--250 words",
    "Write it last.",
    "Motivate the problem and state your contributions",
    "Briefly place your work in context",
    "Describe the bina.az dataset: size, key columns",
    "Describe your impurity criteria",
    "State the objective and the Pegasos update you implemented",
    "Explain the learning-rate schedule",
    "hyperparameter search protocol, seeds, and the",
    "Lead with a results table; back every claim",
    "Replace with a figure",
    "One clear figure beats ten decorative ones",
    "Where and why each model fails",
    "One short paragraph: what you found and what you would do next.",
    "Disclose any AI-assistant use",
]
CONTRIB_TEMPLATE_MARKERS = [
    "**Surname, Name —**",
    "2–4 sentences: what you designed",
    "| ---                    | ---           | ---                   |",
    "| ---    | ---       |",
]


def check() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    report = ROOT / "report" / "report.tex"
    if report.exists():
        tex = report.read_text(encoding="utf-8")
        left = [m for m in REPORT_TEMPLATE_MARKERS if m in tex]
        if left:
            errors.append(
                f"`report/report.tex` still contains template text ({len(left)} spots), "
                f"e.g. “{left[0]}”"
            )
        placeholders = len(re.findall(r"&\s*---", tex))
        if placeholders:
            errors.append(
                f"`report/report.tex` has {placeholders} table cells still set to `---` "
                "(write n/a for cells that don't apply, as the template does)"
            )
        if r"\label{tab:results}" not in tex:
            errors.append("`report/report.tex` has no results table (`\\label{tab:results}`)")
    else:
        errors.append("`report/report.tex` is missing")

    contrib = ROOT / "contribution_report.md"
    if contrib.exists():
        text = contrib.read_text(encoding="utf-8")
        if any(m in text for m in CONTRIB_TEMPLATE_MARKERS):
            errors.append("`contribution_report.md` still has unfilled template rows")
    else:
        errors.append("`contribution_report.md` is missing from the repo root")

    reqs = ROOT / "requirements.txt"
    if reqs.exists():
        unpinned = []
        for raw in reqs.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line and not line.startswith("-") and "==" not in line:
                unpinned.append(line)
        if unpinned:
            errors.append(f"`requirements.txt` has unpinned packages: {', '.join(unpinned)}")
    else:
        errors.append("`requirements.txt` is missing")

    stubs, deliberate = [], []
    for py in sorted((ROOT / "src").glob("**/*.py")):
        code = py.read_text(encoding="utf-8")
        todo = len(re.findall(r"raise NotImplementedError\(\s*[\'\"]TODO", code))
        other = code.count("raise NotImplementedError") - todo
        rel = py.relative_to(ROOT).as_posix()
        if todo:
            stubs.append(f"{rel} ({todo})")
        if other:
            deliberate.append(f"{rel} ({other})")
    if stubs:
        errors.append(f"unimplemented TODO stubs left in: {', '.join(stubs)}")
    if deliberate:
        warnings.append(
            f"other `raise NotImplementedError` in: {', '.join(deliberate)} — deliberate?"
        )

    if not list((ROOT / "presentation").glob("*.pdf")):
        warnings.append("no slides PDF in `presentation/` yet")

    readme = ROOT / "README.md"
    if readme.exists() and "(Starter Pack)" in readme.read_text(encoding="utf-8"):
        warnings.append("`README.md` is still the starter README — describe *your* project")

    return errors, warnings


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):  # emoji below; Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    warn_only = "--warn-only" in argv
    errors, warnings = check()
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"

    print("## Submission readiness\n")
    if not errors and not warnings:
        print("All automatic checks pass. Still read the checklist in CONTRIBUTING.md.")
    for e in errors:
        print(f"- ❌ {e}")
    for w in warnings:
        print(f"- ⚠️ {w}")
    if errors and warn_only:
        print("\n_Expected while the project is in progress — must be clean before tagging "
              "`v1.0-final`._")

    if in_ci:
        level = "warning" if warn_only else "error"
        for e in errors:
            print(f"::{level} title=Submission readiness::{_plain(e)}", file=sys.stderr)
        for w in warnings:
            print(f"::warning title=Submission readiness::{_plain(w)}", file=sys.stderr)

    return 0 if (warn_only or not errors) else 1


def _plain(s: str) -> str:
    return s.replace("`", "")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
