"""
contrib_stats.py — who has committed what, straight from Git history.

The brief grades individual contribution partly "by Git history" and treats
claims that contradict it as an integrity issue. CI prints this table on
every push to main so the team can keep contribution_report.md honest and
spot an imbalance early (not the night before the deadline).

Run from the repo root (needs full history: `git fetch --unshallow` in CI):
    python tools/contrib_stats.py

Counts non-merge commits on the current branch after the course starter pack
(STARTER), which is course-provided code, not team work. Rows are per author
name; add a .mailmap to merge one person's different names/emails.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEP = "\x1f"
STARTER = "dd5e816"  # "Add ML final project starter pack": it and earlier commits are excluded


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8", errors="replace")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    fmt = f"@@{SEP}%aN{SEP}%ad{SEP}%(trailers:key=Co-authored-by,valueonly,separator=;)"
    try:
        git("cat-file", "-e", f"{STARTER}^{{commit}}")
        revs, note = [f"{STARTER}..HEAD"], f"_Course starter pack ({STARTER}) and earlier excluded._"
    except subprocess.CalledProcessError:
        revs, note = ["HEAD"], f"_Starter commit {STARTER} not found: counting the whole history._"
    log = git("log", *revs, "--no-merges", "--numstat", f"--format={fmt}", "--date=short")
    stats: dict[str, dict] = defaultdict(
        lambda: {"commits": 0, "co": 0, "added": 0, "deleted": 0, "days": set()}
    )
    author = None
    for line in log.splitlines():
        if line.startswith("@@" + SEP):
            _, author, day, coauthors = line.split(SEP)
            stats[author]["commits"] += 1
            stats[author]["days"].add(day)
            for co in filter(None, (c.strip() for c in coauthors.split(";"))):
                name = co.split("<")[0].strip()
                stats[name]["co"] += 1
                stats[name]["days"].add(day)
        elif line.strip() and author:
            added, deleted, _path = line.split("\t", 2)
            if added != "-":  # binary files report "-"
                stats[author]["added"] += int(added)
                stats[author]["deleted"] += int(deleted)

    total = sum(s["commits"] for s in stats.values()) or 1
    print("## Contributions on this branch (from Git history)\n")
    print(note + "\n")
    print("| Author | Commits | Share | Co-authored | Lines + / − | Active days |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["commits"]):
        print(
            f"| {name} | {s['commits']} | {100 * s['commits'] / total:.0f}% | {s['co']} "
            f"| +{s['added']} / −{s['deleted']} | {len(s['days'])} |"
        )
    print(
        "\n_Lines are a rough signal, not a score — a careful 50-line fix can matter more than "
        "a 500-line dump. Same person under two names? Use one `git config user.name` "
        "everywhere, and add a `.mailmap` line `Real Name <email>` to merge past commits._"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
