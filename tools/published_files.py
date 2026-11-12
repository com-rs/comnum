"""No document of a reference in the repository: what Git tracks is what is published.

    python tools/published_files.py --check

The papers and the standards stay local (`references/`, ignored), and so do the reading
cards written from them (`validation/readings/`, ignored) and the captures of signals
(`captures/`). The check fails on a tracked file that is:

- under `references/`, `validation/readings/`, `captures/` or `.claude/`;
- a document by its extension (PDF, PostScript, EPUB, DjVu, Word);
- a PDF by its content (a file that starts with `%PDF-`, whatever its name);
- a reading card by its content (the section titles of a card).

`.gitignore` keeps these files out of a commit; this check also catches a file added
with `git add -f`, a renamed PDF and a card written elsewhere, in CI as well.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from _common import ROOT

FOLDERS = ("references/", "validation/readings/", "captures/", ".claude/")
EXTENSIONS = re.compile(r"\.(pdf|ps|eps|epub|djvu|doc|docx)$", re.IGNORECASE)
# the section titles of a reading card (its template)
CARD = re.compile(
    rb"^## (What the document is about|Reproducible items|Missing or unclear in the document)",
    re.MULTILINE,
)


def tracked() -> list[str]:
    """The files Git tracks, relative to the root."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def problems(paths: list[str]) -> list[str]:
    """The tracked files that must stay local, each with its reason."""
    out = []
    for p in paths:
        if any(p.startswith(f) or f"/{f}" in p for f in FOLDERS):
            out.append(f"{p}: a local folder (references, reading cards, captures)")
            continue
        if EXTENSIONS.search(p):
            out.append(f"{p}: a document")
            continue
        path = ROOT / p
        if not path.is_file():
            continue
        with path.open("rb") as f:
            head = f.read(1 << 16)
        if head.startswith(b"%PDF-"):
            out.append(f"{p}: a PDF by its content")
        elif p.endswith(".md") and CARD.search(head):
            out.append(f"{p}: a reading card by its section titles")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on a file that must stay local")
    parser.parse_args()
    found = problems(tracked())
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print(
            f"{len(found)} tracked file(s) must stay local: `git rm --cached` them",
            file=sys.stderr,
        )
        return 1
    print("no document of a reference is tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
