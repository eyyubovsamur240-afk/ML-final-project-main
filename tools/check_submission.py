"""
check_submission.py — catch the brief's automatic deductions before graders do.

Looks for the things the project statement penalises that a script can see:
  - leftover template text in report/report.tex,
  - an unfilled contribution_report.md,
  - unpinned dependencies in requirements.txt,
  - TODO stubs (`raise NotImplementedError("TODO ...")`) still left in src/,
  - no results table in the report, or a generated table/figure it uses that
    is not in the repo (src.run_all writes them; they are git-ignored until final),
  - no slides PDF in presentation/ (warning only: the PDF is also on Moodle).

Run from the repo root:
    python tools/check_submission.py              # strict: exit 1 on any error
    python tools/check_submission.py --warn-only  # always exit 0 (used on every CI run)

CI appends the Markdown output to the job summary.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
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
]
DASH_ROW = re.compile(r"^\s*\|(\s*-{3,}\s*\|)+\s*$")  # | --- | --- |  (any spacing)
# Files report.tex pulls in: \input{..}, \includegraphics[..]{..} and the report's
# \generatedtable{..} / \generatedfigure{..}{width} wrappers around them.
INCLUDED_FILE = re.compile(
    r"\\(input|generatedtable|generatedfigure|includegraphics)(?:\[[^\]]*\])?\{([^}#]+)\}")
GRAPHICS_PATH = re.compile(r"\\graphicspath\{((?:\{[^}]*\})+)\}")
GRAPHICS_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")


def _strip_comments(tex: str) -> str:
    """Drop LaTeX comments: from a % that is not escaped (\\% is a literal percent sign)."""
    lines = []
    for line in tex.splitlines():
        i = 0
        while i < len(line) and line[i] != "%":
            i += 2 if line[i] == "\\" else 1  # skip the character after a backslash
        lines.append(line[:i])
    return "\n".join(lines)


def _candidates(command: str, name: str, graphics_path: list[str]) -> list[str]:
    """The files LaTeX would try for one include, in its order (extension added if missing)."""
    if command in ("input", "generatedtable"):
        return [name] if Path(name).suffix else [name + ".tex", name]
    names = [name] if Path(name).suffix else [name + ext for ext in GRAPHICS_EXTENSIONS]
    return [folder + n for folder in ["", *graphics_path] for n in names]


@functools.lru_cache(maxsize=None)
def _own_git_checkout(root: Path) -> bool:
    """root is the top level of a git checkout that git can read (not a ZIP download)."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                             capture_output=True, text=True)
    except FileNotFoundError:  # git not installed
        return False
    return top.returncode == 0 and Path(top.stdout.strip()).resolve() == root.resolve()


def _committed(path: Path) -> bool:
    """Present and tracked by git (just present when this is not a git checkout).

    Generated tables and figures are git-ignored, so a local run_all creates them
    without putting them in the repo the graders clone.
    """
    if not path.exists():
        return False
    if not _own_git_checkout(ROOT):
        return True
    git = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT,
                         capture_output=True)
    return git.returncode != 1  # 1 = untracked; any other failure is git's, not an answer


def _unfilled_rows(text: str) -> int:
    """Table rows made only of `---` cells, not counting each table's header separator."""
    lines = text.splitlines()
    count = 0
    for i, line in enumerate(lines):
        if not DASH_ROW.match(line):
            continue
        header_above = i > 0 and lines[i - 1].lstrip().startswith("|") and not (
            i > 1 and lines[i - 2].lstrip().startswith("|")
        )
        if not header_above:  # the separator sits right under the table's first row
            count += 1
    return count


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
        body = _strip_comments(tex)
        graphics_path = [folder for group in GRAPHICS_PATH.findall(body)
                         for folder in re.findall(r"\{([^}]*)\}", group)]
        missing = []
        for command, name in sorted(set(INCLUDED_FILE.findall(body))):
            files = [report.parent / f for f in _candidates(command, name, graphics_path)]
            if not any(_committed(f) for f in files):
                missing.append(name)
            tex += "".join(f.read_text(encoding="utf-8") for f in files[:1]
                           if f.suffix == ".tex" and f.exists())
        if missing:
            errors.append(
                "`report/report.tex` uses generated files that are not in the repo: "
                f"{', '.join(missing)} — run `python -m src.run_all`, then "
                "`git add -f` them (see README)"
            )
        if not re.search(r"\\label\{tab:(?!data\})[^}]*\}", tex):
            warnings.append(
                "`report/report.tex`: no results table found (no `\\label{tab:...}` "
                "besides the dataset table) — a results table is required"
            )
    else:
        errors.append("`report/report.tex` is missing")

    contrib = ROOT / "contribution_report.md"
    if contrib.exists():
        text = contrib.read_text(encoding="utf-8")
        if any(m in text for m in CONTRIB_TEMPLATE_MARKERS) or _unfilled_rows(text):
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
