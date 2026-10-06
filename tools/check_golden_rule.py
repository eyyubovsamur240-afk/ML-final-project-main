"""
check_golden_rule.py — enforce the project's golden rule in CI.

    "Your decision tree and your SVM must be your own NumPy implementations."

Every file in PROTECTED may import only NumPy, the Python standard library,
and the project's own modules (`from .evaluate import ...`, `from src.data_prep
import ...`). Anything else (scikit-learn, SciPy, cvxpy, libsvm, ...) is
reported, as is any dynamic import (`importlib.import_module`, `__import__`)
that could sneak one in.

Own modules imported by a protected file are followed (recursively): they may
use pandas, matplotlib or sklearn.metrics, but must not pull in a model or
optimiser (MODEL_MODULES); otherwise a wrapper module would bypass the rule.

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

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python 3.10+ needed (the team uses 3.11, see CONTRIBUTING.md); "
        f"this is {sys.version.split()[0]}"
    )

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

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

# Models/optimisers that must not reach a protected file through one of our own
# modules (a module and everything below it, e.g. "sklearn.svm" covers sklearn.svm._classes).
MODEL_MODULES = {
    "sklearn.svm", "sklearn.tree", "sklearn.linear_model", "sklearn.ensemble",
    "sklearn.neighbors", "sklearn.neural_network", "sklearn.kernel_ridge",
    "scipy.optimize", "statsmodels", "cvxpy", "cvxopt", "libsvm", "svmlight",
    "torch", "tensorflow", "keras", "jax", "xgboost", "lightgbm", "catboost",
}

STDLIB = set(sys.stdlib_module_names)
OWN_PACKAGE = "src"


def _module_files(base: Path, parts: list[str]) -> list[Path]:
    """Files Python runs to import base/<parts>: packages on the way, then the module."""
    files, cur = [], base
    for i, part in enumerate(parts):
        if i > 0 and (cur / "__init__.py").is_file():
            files.append(cur / "__init__.py")
        cur = cur / part
    if cur.with_suffix(".py").is_file():
        files.append(cur.with_suffix(".py"))
    elif (cur / "__init__.py").is_file():
        files.append(cur / "__init__.py")
    return files


def _own_targets(node: ast.Import | ast.ImportFrom, alias: ast.alias,
                 importer: Path) -> list[Path] | None:
    """Files of our own package that this import runs, or None if it is not our package."""
    if isinstance(node, ast.ImportFrom):
        parts = [p for p in (node.module or "").split(".") if p]
        if node.level > 0:  # relative: resolved from the importing file's package
            base = importer.parent
            for _ in range(node.level - 1):
                base = base.parent
        elif parts and parts[0] == OWN_PACKAGE:
            base, parts = SRC, parts[1:]
        else:
            return None
        files = _module_files(base, parts) if parts else []
        # `from . import evaluate` / `from .pkg import mod`: the name may be a module too
        return files + [f for f in _module_files(base, parts + [alias.name]) if f not in files]
    parts = alias.name.split(".")
    if parts[0] != OWN_PACKAGE:
        return None
    return _module_files(SRC, parts[1:]) if len(parts) > 1 else []


def find_violations(path: Path, _seen: set[Path] | None = None,
                    _followed: bool = False) -> list[tuple[int, str]]:
    """Return (line, message) for every disallowed import in `path`.

    In a protected file every import other than NumPy, the standard library and our
    own modules is disallowed. Our own modules are followed; there, only the
    MODEL_MODULES are disallowed.
    """
    seen = _seen if _seen is not None else set()
    seen.add(path.resolve())
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))  # bytes: BOM/coding cookie OK
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"syntax error ({exc.msg}); fix it first")]

    problems: list[tuple[int, str]] = []

    def check_module(line: int, module: str, names: list[str]) -> None:
        if _followed:
            # `from sklearn import svm` imports the model module "sklearn.svm"
            full = [module] + [f"{module}.{n}" for n in names]
            bad = any(f == m or f.startswith(m + ".") for f in full for m in MODEL_MODULES)
        else:
            top = module.split(".")[0]
            bad = top not in ALLOWED_THIRD_PARTY and top not in STDLIB
        if bad:
            problems.append((line, f"import of '{module}' is not allowed"))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                targets = _own_targets(node, alias, path)
                if targets is None:
                    if isinstance(node, ast.ImportFrom):  # one check per `from X import a, b`
                        check_module(node.lineno, node.module or "", [a.name for a in node.names])
                        break
                    check_module(node.lineno, alias.name, [])
                    continue
                for target in targets:
                    if target.resolve() in seen:
                        continue
                    rel = target.relative_to(ROOT).as_posix()
                    for sub_line, msg in find_violations(target, seen, _followed=True):
                        problems.append((node.lineno, f"{rel}:{sub_line} (imported here): {msg}"))
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
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    targets = argv or PROTECTED
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    failed = False
    for rel in targets:
        rel = rel.replace("\\", "/")
        path = ROOT / rel
        if not path.exists():
            print(f"{rel}: protected file is missing — update PROTECTED in {Path(__file__).name}")
            failed = True
            continue
        for line, msg in find_violations(path):
            failed = True
            text = msg if "syntax error" in msg else (
                f"{msg}. Golden rule: the decision tree and the SVM must be your own "
                "NumPy code (scikit-learn is for the baselines only)."
            )
            if in_ci:
                print(f"::error file={rel},line={line},title=Golden rule::{text}")
            print(f"{rel}:{line}: {text}")
    if not failed:
        print(
            f"Golden rule OK ({', '.join(targets)}): only NumPy, the standard library "
            "and our own modules (which import no model library)."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
