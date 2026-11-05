"""The validation cards of `validation/cards/`: check them, list them, run one.

    python tools/cards.py --check     # every card consistent with the repository
    python tools/cards.py list        # id, runner, tier, literature entry, evidence
    python tools/cards.py run <id>    # rerun the comparison of one card

A card is run by its runner: a function of `comnum` (`pytest tests/test_cards.py -k <id>`),
a Rust test of `validation/tests/<theme>/<module>.rs` (`cargo test --release -p
comnum-validation --test <theme> -- <module>::<test>`, ignored tests included) or a
specification of `specs/validation/` (`comnum verify`). The check makes sure that
the evidence of a recording comes from a tool, that every tolerance but an exact one
has its `why`, that the runner exists and points back at the card (a Rust test loads
it with `Card::load("<id>")`), that the literature entry exists, that the reference file is in
a theme folder of `references/` (the local folder of the documents, not distributed), that every card a Rust test loads exists, and that a
test named after an entry of `validation/literature.yaml` loads a card (the tests of
`validation/cards/uncarded.txt` apart, a list that only shrinks).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml
from _common import ROOT, defines

CARDS = ROOT / "validation" / "cards"
# The theme folders of references/ (the documents themselves are not in the repository:
# a card names its file under one of these folders)
THEMES = {
    "coding",
    "fiber",
    "hardware",
    "isac",
    "mimo",
    "modulation",
    "multicarrier",
    "multiple-access",
    "optical-wireless",
    "sensing",
    "simulation",
    "standards",
    "synchronisation",
    "systems",
    "wireless-channels",
}
UNCARDED = CARDS / "uncarded.txt"
LOAD = re.compile(r'Card::load\("([a-z0-9_]+)"\)')


def cards() -> list[dict]:
    return [
        yaml.safe_load(p.read_text(encoding="utf-8")) | {"_file": p}
        for p in sorted(CARDS.glob("*.yaml"))
    ]


def runner(card: dict) -> tuple[str, str]:
    ((kind, target),) = card["run"].items()
    return kind, target


def problems() -> list[str]:
    out = []
    entries = yaml.safe_load((ROOT / "validation" / "literature.yaml").read_text(encoding="utf-8"))
    literature = {e["id"] for e in entries}
    themes = THEMES
    known = set()
    for card in cards():
        cid = card["id"]
        known.add(cid)
        if card["_file"].stem != cid:
            out.append(f"{card['_file'].name}: the id is {cid}")
        if card.get("literature") and card["literature"] not in literature:
            out.append(f"{cid}: no entry {card['literature']} in validation/literature.yaml")
        file = card["source"].get("file")
        if file and file.split("/")[0] not in themes:
            out.append(f"{cid}: {file} is not in a theme folder of references/")
        if ("recording" in card.get("evidence", [])) != (card["source"]["kind"] == "tool"):
            out.append(
                f"{cid}: the evidence recording goes with source.kind tool, and only with it"
            )
        for i, case in enumerate(card.get("cases") or []):
            tolerance = case.get("tolerance") or card.get("tolerance") or {}
            ranged = isinstance(case["expected"], dict) and set(case["expected"]) <= {"min", "max"}
            if tolerance and not tolerance.get("exact") and not ranged:
                if not (case.get("why") or card.get("why")):
                    out.append(f"{cid}: case {i} has a tolerance without its why")
        kind, target = runner(card)
        if kind == "test":
            path, _, name = target.partition("::")
            if not (ROOT / path).is_file() or (name and not defines(ROOT / path, name)):
                out.append(f"{cid}: no test {target}")
            elif f'Card::load("{cid}")' not in (ROOT / path).read_text(encoding="utf-8"):
                out.append(f"{cid}: {path} does not load the card")
        elif kind == "spec" and not (ROOT / target).is_file():
            out.append(f"{cid}: no specification {target}")
    tests = ROOT / "validation" / "tests"
    uncarded = {
        line.split("#")[0].strip()
        for line in UNCARDED.read_text(encoding="utf-8").splitlines()
        if line.split("#")[0].strip()
    }
    for test in sorted(tests.rglob("*.rs")):
        loads = LOAD.findall(test.read_text(encoding="utf-8"))
        for cid in loads:
            if cid not in known:
                out.append(f"{test.relative_to(ROOT)}: no card {cid}")
        key = test.relative_to(tests).as_posix()
        if test.stem in literature and not loads and key not in uncarded:
            out.append(
                f"{test.relative_to(ROOT)}: compares with {test.stem} but loads no card "
                "(put the published values and the tolerances in a card)"
            )
        if key in uncarded and loads:
            out.append(f"{key} loads a card: remove it from {UNCARDED.relative_to(ROOT)}")
    for key in sorted(uncarded):
        if not (tests / key).is_file():
            out.append(f"{UNCARDED.relative_to(ROOT)}: no test {key}")
    return out


def command(card: dict) -> list[str]:
    kind, target = runner(card)
    if kind == "function":
        return [sys.executable, "-m", "pytest", "-q", "tests/test_cards.py", "-k", card["id"]]
    if kind == "test":
        path, _, name = target.partition("::")
        p = Path(path)
        # validation/tests/<theme>/<module>.rs: the binary <theme>, the test <module>::<name>
        only = ["--exact", f"{p.stem}::{name}"] if name else [f"{p.stem}::"]
        return ["cargo", "test", "--release", "-p", "comnum-validation", "--test", p.parent.name, "--",
                "--include-ignored", *only, "--nocapture"]  # fmt: skip
    return [
        "cargo",
        "run",
        "--release",
        "-p",
        "comnum-spec",
        "--bin",
        "comnum",
        "--",
        "verify",
        target,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check every card")
    parser.add_argument("action", nargs="?", choices=["list", "run"])
    parser.add_argument("id", nargs="?")
    args = parser.parse_args()
    if args.check:
        bad = problems()
        for p in bad:
            print(p, file=sys.stderr)
        return 1 if bad else 0
    if args.action == "list":
        for card in cards():
            kind, target = runner(card)
            print(
                f"{card['id']:38} {kind:8} {card.get('tier', 'ci'):9} {card.get('literature', '-'):15} "
                f"{','.join(card['evidence']):24} {target}"
            )
        return 0
    if args.action == "run" and args.id:
        match = [c for c in cards() if c["id"] == args.id]
        if not match:
            print(f"no card {args.id}", file=sys.stderr)
            return 1
        cmd = command(match[0])
        print(" ".join(cmd))
        return subprocess.call(cmd, cwd=ROOT)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
