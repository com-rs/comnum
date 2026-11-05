"""Time AFF3CT on the recorded configurations: one thread, a fixed budget per point,
at one Eb/N0 (the middle of the recorded sweep), decoder throughput and chain throughput."""

import glob
import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
binary = sys.argv[2]
out = pathlib.Path(sys.argv[3])
seconds = float(sys.argv[4])
results = {}
for f in sorted(glob.glob(str(root / "validation/aff3ct/*.json"))):
    d = json.load(open(f))
    pts = d["data"]
    mid = pts[len(pts) // 2]["ebn0_db"]
    cmd = d["command"].split(" ", 1)[1]
    # replace the sweep by one point, a time budget and one thread; keep AZCW/paths
    cmd = re.sub(
        r"-m \S+ -M \S+ -s \S+ -e \d+ -S \d+ -p \d+",
        f"-m {mid} -M {mid} -s 1 -e 100000000 -S 1 -p 64 -t 1 --sim-stop-time {int(seconds)} --sim-stats",
        cmd,
    )
    argv = [binary] + [a.strip("'") for a in re.findall(r"'[^']*'|\S+", cmd)]
    r = subprocess.run(argv, capture_output=True, text=True, cwd=root)
    dec = chain = None
    frames = None
    for line in r.stdout.splitlines():
        m = re.search(
            r"\|\s+(decode\w*)\s+\|.*?\|\|\s*(\d+)\s*\|\s*([\d.e+-]+)\s*\|\s*[\d.]+\s*\|\|\s*([\d.e+-]+)",
            line,
        )
        if m and "Decoder" in line and dec is None:
            dec = float(m.group(4))
            frames = int(m.group(2))
        m = re.search(r"\|\s+TOTAL\s+\|.*?\|\|\s*(\d+)\s*\|\s*([\d.e+-]+)\s*\|", line)
        if m:
            chain = (int(m.group(1)), float(m.group(2)))
    kbits = (
        d["bits_per_frame"]
        if "-sim-coded" not in cmd
        else int(re.search(r"-N (\d+)", cmd).group(1))
    )
    chain_mbps = (chain[0] * kbits / chain[1] / 1e6) if chain else None
    results[d["name"]] = {
        "ebn0_db": mid,
        "decoder_mbps": dec,
        "chain_mbps": chain_mbps,
        "frames": frames,
        "bits_per_frame": kbits,
        "returncode": r.returncode,
    }
    print(d["name"], results[d["name"]], flush=True)
    if r.returncode:
        print(r.stderr[-800:], flush=True)
out.write_text(json.dumps(results, indent=1))
