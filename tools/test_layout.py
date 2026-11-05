"""The layout of the tests: every test in the place `docs/ARCHITECTURE.md` gives it.

    python tools/test_layout.py --check

The check fails on:

- a file of `validation/tests/` outside a theme folder, or a folder that is not a theme;
- a theme whose `main.rs` does not declare exactly the files of its folder (one
  `mod <key>;` each), or a file with a `fn main` (an exploration is an example of
  `validation/examples/`, not a test);
- an ignored Rust test without its reason (`#[ignore = "..."]`);
- a Python test outside `tests/oracle/` that reads the recorded oracles, or a file of
  `tests/oracle/` that is neither a test, the package, nor a recording.

It also warns, without failing, on a Rust source file of the library or of the bindings
longer than `LONG` lines: a file that size is split into a folder of submodules by notion
(`multipath/`, `nlse/`, `ccsds/`), the public paths kept by re-exports.
"""

from __future__ import annotations

import argparse
import re
import sys

from _common import ROOT

THEMES = {
    "coding",
    "modulation",
    "multicarrier",
    "mimo",
    "synchronisation",
    "wireless_channels",
    "fiber",
    "optical_wireless",
    "hardware",
    "systems",
    "simulation",
}
MOD = re.compile(r"^mod ([a-z0-9_]+);$", re.MULTILINE)
IGNORE = re.compile(r"^\s*#\[ignore\]", re.MULTILINE)
MAIN = re.compile(r"^\s*fn main\(", re.MULTILINE)
ORACLE = re.compile(r"^\s*(from oracle import|import oracle)", re.MULTILINE)
RECORDINGS = {".json", ".npz"}
# the length (lines) above which a source file of the library is reported
LONG = 1200
# the Python tests that read the package `oracle` without comparing with a tool
READERS = {"test_cards.py": "the expected values of the cards, read with `oracle.recorded`"}


def problems() -> list[str]:
    out = []
    tests = ROOT / "validation" / "tests"
    for path in sorted(tests.iterdir()):
        rel = path.relative_to(ROOT)
        if path.is_file():
            out.append(f"{rel}: a validation test lies in a theme folder of validation/tests/")
        elif path.name not in THEMES:
            out.append(f"{rel}: not a theme ({', '.join(sorted(THEMES))})")
    for theme in sorted(THEMES):
        folder = tests / theme
        main = folder / "main.rs"
        if not main.exists():
            out.append(f"{main.relative_to(ROOT)} is missing")
            continue
        declared = set(MOD.findall(main.read_text(encoding="utf-8")))
        files = {p.stem for p in folder.glob("*.rs") if p.name != "main.rs"}
        for key in sorted(files - declared):
            out.append(f"{main.relative_to(ROOT)}: `mod {key};` is missing")
        for key in sorted(declared - files):
            out.append(f"{main.relative_to(ROOT)}: no file {key}.rs for `mod {key};`")
        for p in sorted(folder.rglob("*.rs")):
            text = p.read_text(encoding="utf-8")
            if MAIN.search(text):
                out.append(
                    f"{p.relative_to(ROOT)}: a `fn main` (an exploration goes to validation/examples/)"
                )
            if IGNORE.search(text):
                out.append(f"{p.relative_to(ROOT)}: an `#[ignore]` without its reason")
    oracle = ROOT / "tests" / "oracle"
    for p in sorted((ROOT / "tests").glob("*.py")):
        if p.name not in READERS and ORACLE.search(p.read_text(encoding="utf-8")):
            out.append(
                f"{p.relative_to(ROOT)}: reads the recorded oracles, so it lies in tests/oracle/"
            )
    for p in sorted(oracle.iterdir()):
        ok = p.suffix in RECORDINGS or p.name == "__init__.py" or p.name == "__pycache__"
        ok = ok or (p.suffix == ".py" and p.name.startswith("test_"))
        if not ok:
            out.append(
                f"{p.relative_to(ROOT)}: neither an oracle test, the package, nor a recording"
            )
    return out


def long_files() -> list[str]:
    files = [*(ROOT / "crates").rglob("src/**/*.rs"), *(ROOT / "bindings").rglob("src/**/*.rs")]
    out = []
    for p in sorted(files):
        n = len(p.read_text(encoding="utf-8").splitlines())
        if n > LONG:
            out.append(f"{p.relative_to(ROOT)}: {n} lines (more than {LONG}: split it by notion)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on a test out of place")
    parser.parse_args()
    for line in long_files():
        print(f"warning: {line}", file=sys.stderr)
    found = problems()
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print(
            f"{len(found)} test(s) out of place: see docs/ARCHITECTURE.md, 'Where a test goes'",
            file=sys.stderr,
        )
        return 1
    print("the tests are in their places")
    return 0


if __name__ == "__main__":
    sys.exit(main())
