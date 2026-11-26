"""Pre-publish hygiene scan.

Scans the content of every tracked file and the full patch history
(`git log -p --all`) for secrets and for private names.

Patterns:
  - the Anthropic key prefix
  - the API key variable name followed by "=" and a non-space character
  - whole-word, case-insensitive terms from the env var ``RAG_DENYLIST``
    (comma-separated; supplied at run time, never stored in the repo)

Exit code 1 on any hit (location and line are printed), 0 if clean.
"""
import os
import re
import subprocess
import sys


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def build_patterns() -> list[tuple[str, re.Pattern]]:
    pats = [
        # Built from fragments so this file never matches its own patterns.
        ("key-prefix", re.compile("sk-" + "ant")),
        ("key-assignment", re.compile("ANTHROPIC_API" + r"_KEY=\S")),
    ]
    for term in os.environ.get("RAG_DENYLIST", "").split(","):
        term = term.strip()
        if term:
            pats.append(
                ("denylist", re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE))
            )
    return pats


# Lines that merely *describe* the scan (spec, plan, a test denylist entry) are
# not findings. Matched case-sensitively against the line text.
BENIGN = re.compile(r"scan for secrets|No secrets in git|hygiene_scan|RAG_DENYLIST|^\+?DENYLIST = ")


def scan_lines(location: str, text: str, pats, hits: list[str]) -> None:
    for n, line in enumerate(text.splitlines(), start=1):
        for label, rx in pats:
            if rx.search(line) and not (label != "denylist" and BENIGN.search(line)):
                # Denylist terms are not echoed in the label, only the line.
                hits.append(f"{location}:{n} [{label}] {line.strip()[:200]}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pats = build_patterns()
    if len(pats) == 2:
        print("warning: RAG_DENYLIST is empty; only secret patterns are checked", file=sys.stderr)
    hits: list[str] = []

    for path in git("ls-files", "-z").decode("utf-8").split("\0"):
        if not path:
            continue
        try:
            text = git("show", f":{path}").decode("utf-8", errors="replace")
        except subprocess.CalledProcessError:
            continue
        scan_lines(path, text, pats, hits)

    commit = "?"
    log = git("log", "-p", "--all", "--format=commit %H").decode("utf-8", errors="replace")
    for n, line in enumerate(log.splitlines(), start=1):
        if line.startswith("commit ") and len(line) == 47:
            commit = line[7:15]
            continue
        for label, rx in pats:
            if rx.search(line) and not (label != "denylist" and BENIGN.search(line)):
                hits.append(f"history commit {commit}:{n} [{label}] {line.strip()[:200]}")

    if hits:
        print("\n".join(hits))
        print(f"{len(hits)} hit(s)")
        return 1
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
