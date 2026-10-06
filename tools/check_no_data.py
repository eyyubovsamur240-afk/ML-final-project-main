"""
check_no_data.py — keep the dataset and other big/binary outputs out of Git.

The brief says: do not commit the raw archive. `.gitignore` already ignores
`data/*`, but `git add -f` (or a CSV saved somewhere else) bypasses it. This
check fails if any tracked file:
  - lives in data/ (other than data/README.md and .gitkeep files),
  - has a dataset / model-dump extension (.csv, .zip, .parquet, .pkl, ...),
    except small fixtures under tests/,
  - is larger than MAX_MB.

Run from the repo root:
    python tools/check_no_data.py            # check every tracked file
    python tools/check_no_data.py a.csv b.py # check specific files (pre-commit)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_MB = 5
TEST_FIXTURE_MAX_KB = 100

DATA_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather", ".json.gz",
    ".zip", ".gz", ".7z", ".rar", ".tar", ".bz2", ".xz",
    ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".h5", ".hdf5",
    ".db", ".sqlite",
}
DATA_DIR_ALLOWED = {"data/README.md"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8")
    return [f for f in out.split("\0") if f]


def problems_for(rel: str) -> list[str]:
    rel = rel.replace("\\", "/")
    path = ROOT / rel
    if not path.is_file():  # deleted in the working tree: nothing to check
        return []
    found = []
    size = path.stat().st_size
    name = path.name.lower()
    has_data_ext = any(name.endswith(ext) for ext in DATA_EXTENSIONS)

    if rel.startswith("data/") and rel not in DATA_DIR_ALLOWED and name != ".gitkeep":
        found.append("files in data/ must not be committed (the dataset stays local)")
    elif has_data_ext and not (rel.startswith("tests/") and size <= TEST_FIXTURE_MAX_KB * 1024):
        found.append(
            "looks like a dataset or model dump; regenerate it with code instead of committing it"
        )
    if size > MAX_MB * 1024 * 1024:
        found.append(f"file is {size / 1024 / 1024:.1f} MB (limit {MAX_MB} MB)")
    return found


def main(argv: list[str]) -> int:
    files = argv or tracked_files()
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    failed = False
    for rel in files:
        for msg in problems_for(rel):
            failed = True
            if in_ci:
                print(f"::error file={rel},title=No data in Git::{msg}")
            print(f"{rel}: {msg}")
    if failed:
        print("\nIf you staged it by accident: git rm --cached <file>  (keeps your local copy)")
    else:
        print(f"No-data check OK ({len(files)} files checked).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
