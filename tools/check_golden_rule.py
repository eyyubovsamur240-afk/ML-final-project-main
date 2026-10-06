"""
check_golden_rule.py — enforce the project's golden rule in CI.

    "Your decision tree and your SVM must be your own NumPy implementations."

Every file in PROTECTED may import only NumPy, the Python standard library,
and other modules of this package (relative imports such as
`from .decision_tree import DecisionTree`). Anything else (scikit-learn,
SciPy, cvxpy, libsvm, ...) is reported, as is any dynamic import
(`importlib.import_module`, `__import__`) that could sneak one in.

Run from the repo root:
    python tools/check_golden_rule.py               # check every PROTECTED file
    python tools/check_golden_rule.py src/svm.py    # check specific files

Exit code 0 = clean, 1 = violation(s) found.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files that must be your OWN NumPy code. If you build bonus models from your
# own trees (e.g. src/ensemble.py for a random forest), add them here too.
PROTECTED = [
    "src/decision_tree.py",
    "src/svm.py",
]

# Third-party top-level packages allowed inside PROTECTED files, besides the
# standard library. Only extend this after the team agrees the package is NOT
# a model/optimiser library.
ALLOWED_THIRD_PARTY = {"numpy"}

STDLIB = set(sys.stdlib_module_names)


def _is_allowed(module: str) -> bool:
    top = module.split(".")[0]
    return top in ALLOWED_THIRD_PARTY or top in STDLIB


def find_violations(path: Path) -> list[tuple[int, str]]:
    """Return (line, message) for every disallowed import in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed(alias.name):
                    problems.append((node.lineno, f"import of '{alias.name}' is not allowed"))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # relative import of our own code: fine
                continue
            if node.module and not _is_allowed(node.module):
                problems.append((node.lineno, f"import from '{node.module}' is not allowed"))
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name in {"__import__", "import_module"}:
                problems.append((node.lineno, f"dynamic import via {name}() is not allowed"))
    return problems


def main(argv: list[str]) -> int:
    targets = argv or PROTECTED
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    failed = False
    for rel in targets:
        path = ROOT / rel
        if not path.exists():
            print(f"{rel}: protected file is missing — update PROTECTED in {Path(__file__).name}")
            failed = True
            continue
        for line, msg in find_violations(path):
            failed = True
            text = (
                f"{msg}. Golden rule: the decision tree and the SVM must be your own "
                "NumPy code (scikit-learn is for the baselines only)."
            )
            if in_ci:
                print(f"::error file={rel},line={line},title=Golden rule::{text}")
            print(f"{rel}:{line}: {text}")
    if not failed:
        print(f"Golden rule OK: {', '.join(targets)} use only NumPy + the standard library.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
