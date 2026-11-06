#!/usr/bin/env python3
"""Record the split-step Fourier propagation of Optilux that comnum is compared with.

Optilux (GPL-3.0, Paolo Serena, University of Parma, <https://optilux.sourceforge.io/>,
the release of April 2021) is a MATLAB/Octave toolbox of optical communications. This
script runs its ``fiber`` and ``ampliflat`` under Octave (``tools/record_optilux.m``) on
the input field of the split-step scenario of the library benchmark
(``benchmarks/libraries/s5_ssfm.py``: 16-QAM at 32 GBd, root-raised cosine of roll-off
0.1, 4 samples per symbol, 4 dBm, 2¹¹ symbols, seed 1) over the link of that scenario
(80 km spans of standard fibre, 1 km steps, noiseless amplification), and records the
output field of three cases in ``tests/oracle/optilux_ssfm.npz`` (the provenance in the
``.json``), read by ``tests/oracle/test_optilux_oracle.py``:

- ``linear_1span``: one span without Kerr effect (dispersion and attenuation);
- ``kerr_1span``: one span with gamma = 1.3 /W/km;
- ``kerr_10spans``: the ten spans of the benchmark.

It records the same three cases on a shorter field (2⁸ symbols) as plain JSON in
``validation/optilux/ssfm_256.json`` (15 significant digits, one ``[re, im]`` pair per
sample), read by the Rust test of the validation card ``optilux_ssfm``
(``validation/tests/fiber/optilux.rs``, ``just card optilux_ssfm``), which has no reader
of ``.npz`` archives.

It also runs the amplified link of Optilux's example 21 (``tools/record_optilux_ase.m``:
polarisation-multiplexed QPSK through 50 spans of 150 km of lossy fibre and noisy
amplifiers, the bit error rate and the SNR after every span) and records its results in
``validation/optilux/ase_link.json`` for the card ``optilux_ase_link``.

Optilux's fields are returned in comnum's convention: Optilux writes the equation with
the opposite sign of the imaginary unit, so the driver propagates the conjugate of the
input and conjugates the result. Optilux is not a dependency of comnum: download
``optilux.zip`` from SourceForge (it is never committed: GPL-3.0 code) and run::

    python tools/record_optilux.py --optilux /path/to/optilux

The files are written with the functions of the recorded oracles
(``tests/oracle/__init__.py``): recording again with the same version produces no diff.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _common import ROOT

sys.path.insert(0, str(ROOT / "tests"))
from oracle import DIR as OUT
from oracle import archive, dumps

NAME = "optilux_ssfm"
SEED = 1
N_SYMBOLS = 2**11
N_SYMBOLS_CARD = 2**8
CARD_FIXTURE = ROOT / "validation" / "optilux" / "ssfm_256.json"
ASE_FIXTURE = ROOT / "validation" / "optilux" / "ase_link.json"
SYMBOL_RATE, SPS, ROLLOFF, N_TAPS = 32e9, 4, 0.1, 1025
POWER_DBM = 4.0
CASES = [("linear_1span", 1, 0.0), ("kerr_1span", 1, 1.0), ("kerr_10spans", 10, 1.0)]
LINK = {
    "alpha_db_km": 0.2,
    "d_ps_nm_km": 17.0,
    "beta3": 0.0,
    "gamma_w_km": 1.3,
    "fc_hz": 193.1e12,
    "span_km": 80.0,
    "step_km": 1.0,
    "amplification": "ideal (noiseless, alpha L at the end of every span)",
    "fs_hz": SYMBOL_RATE * SPS,
}


def input_field(n_symbols: int = N_SYMBOLS) -> np.ndarray:
    """The input field of ``benchmarks/libraries/s5_ssfm.py`` (at its quick size by
    default)."""
    import comnum

    rng = comnum.Rng(SEED)
    c = comnum.Constellation("qam", 16).normalized()
    _, s = comnum.symbols(c, n_symbols, rng)
    h = comnum.pulse_shape("rrc", SPS, n_taps=N_TAPS, rolloff=ROLLOFF)
    x = comnum.pnorm(comnum.fir_filter(h, comnum.upsample(s, SPS)))
    return x * math.sqrt(comnum.dbm2w(POWER_DBM))


def run_optilux(optilux: Path, x: np.ndarray, n_spans: int, gamma_scale: float) -> tuple:
    """The field after ``n_spans`` spans of Optilux, in comnum's convention, and the
    number of steps of every span."""
    script = ROOT / "tools" / "record_optilux.m"
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        xc = np.conj(x)
        np.savetxt(src, np.c_[xc.real, xc.imag], delimiter=",", fmt="%.17g")
        cmd = [
            "octave", "--no-gui", "--quiet", str(script), str(optilux), str(src), str(dst),
            str(n_spans), repr(gamma_scale),
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = np.loadtxt(dst, delimiter=",")
    steps = [int(v) for v in re.findall(r"\d+", proc.stdout.strip().splitlines()[-1])]
    return np.conj(out[:, 0] + 1j * out[:, 1]), steps


def run_ase_link(optilux: Path) -> list[dict]:
    """The rows of ``tools/record_optilux_ase.m``: one per span."""
    script = ROOT / "tools" / "record_optilux_ase.m"
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "out.csv"
        cmd = ["octave", "--no-gui", "--quiet", str(script), str(optilux), str(dst)]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        rows = np.loadtxt(dst, delimiter=",")
    keys = ["span", "ber", "ber_std", "bits", "snr_db", "snr_theory_db", "ber_theory"]
    out = []
    for row in rows:
        d = dict(zip(keys, row.tolist(), strict=True))
        d["span"], d["bits"] = int(d["span"]), int(d["bits"])
        out.append(d)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--optilux", required=True, type=Path, help="the directory of Optilux")
    args = p.parse_args()
    readme = (args.optilux / "README").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Optilux,\s*([A-Za-z]+ \d{4})", readme)
    version = m.group(1) if m else "unknown"
    octave = subprocess.run(["octave", "--version"], capture_output=True, text=True, check=True)
    octave_version = octave.stdout.splitlines()[0]

    x = input_field()
    arrays = {"input": x}
    steps: dict[str, list[int]] = {}
    for name, n_spans, gamma_scale in CASES:
        y, steps[name] = run_optilux(args.optilux, x, n_spans, gamma_scale)
        arrays[name] = y
        print(f"{name}: {len(y)} samples, steps per span {steps[name]}")
    provenance = {
        "name": NAME,
        "tool": "Optilux",
        "version": version,
        "repository": "https://optilux.sourceforge.io/",
        "license": "GPL-3.0 (the outputs, not the code)",
        "octave": octave_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "script": "tools/record_optilux.m (fiber with steptype symm, dzmax 1000 m, "
        "dphi1fwm false, stepupd nlp, dphimax 1e9; ampliflat with the gain alpha L)",
        "signal": f"16-QAM, {SYMBOL_RATE / 1e9:g} GBd, RRC roll-off {ROLLOFF}, {SPS} samples "
        f"per symbol, {POWER_DBM} dBm, {N_SYMBOLS} symbols, seed {SEED} (comnum)",
        "link": LINK,
        "cases": {name: {"spans": n, "gamma_scale": g} for name, n, g in CASES},
        "steps_per_span": steps,
        "convention": "the fields are conj(optilux(conj(x))): Optilux writes the NLSE "
        "with the opposite sign of the imaginary unit",
        "units": "sqrt(W)",
    }
    doc = {**provenance, "name": NAME, "npz": f"{NAME}.npz"}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{NAME}.json").write_text(dumps(doc), encoding="utf-8")
    (OUT / f"{NAME}.npz").write_bytes(archive(arrays))
    print(f"wrote {OUT / NAME}.json and .npz")

    # the shorter field of the validation card, as JSON for the Rust test
    x = input_field(N_SYMBOLS_CARD)
    fields = {"input": x}
    card_steps = {}
    for name, n_spans, gamma_scale in CASES:
        fields[name], card_steps[name] = run_optilux(args.optilux, x, n_spans, gamma_scale)
    card_doc = {
        **provenance,
        "name": CARD_FIXTURE.stem,
        "signal": provenance["signal"].replace(f"{N_SYMBOLS} symbols", f"{N_SYMBOLS_CARD} symbols"),
        "steps_per_span": card_steps,
        "digits": 15,
    }
    # the fields as compact rows, 15 significant digits (relative 1e-15 per sample)
    text = json.dumps(card_doc, sort_keys=True, indent=1)
    rows = []
    for key in sorted(fields):
        pairs = ",".join(f"[{z.real:.15g},{z.imag:.15g}]" for z in fields[key])
        rows.append(f'  "{key}": [{pairs}]')
    text = text[: text.rindex("}")].rstrip() + ',\n "fields": {\n' + ",\n".join(rows) + "\n }\n}\n"
    CARD_FIXTURE.parent.mkdir(exist_ok=True)
    CARD_FIXTURE.write_text(text, encoding="utf-8")
    print(f"wrote {CARD_FIXTURE} ({CARD_FIXTURE.stat().st_size // 1024} KB)")

    # the amplified link of example 21
    spans = run_ase_link(args.optilux)
    ase_doc = {
        "name": ASE_FIXTURE.stem,
        "tool": "Optilux",
        "version": version,
        "repository": "https://optilux.sourceforge.io/",
        "license": "GPL-3.0 (the outputs, not the code)",
        "octave": octave_version,
        "script": "tools/record_optilux_ase.m: the chain of Examples/ex21_opt_modulation_ase_ber.m "
        "with the stopping rule 3 % at 95 % confidence or 400 000 bits",
        "link": {
            "spans": 50,
            "span_km": 150.0,
            "alpha_db_km": 0.2,
            "dispersion": 0.0,
            "kerr": 0.0,
            "amplifier_gain_db": 30.0,
            "noise_figure_db": 6.0,
            "wavelength_nm": 1550.0,
        },
        "signal": "polarisation-multiplexed QPSK, 10 GBd, root-raised cosine of roll-off 0.01, "
        "32 samples per symbol, 0 dBm in total (0.5 mW per polarisation), 1024 symbols per "
        "polarisation, seed 1 (Optilux's pattern)",
        "receiver": "no optical filter; the root-raised-cosine electrical filter of bandwidth 0.5 "
        "symbol rate; data-aided timing; the SNR of samp2snr (Es/N0 per polarisation)",
        "theory": "the example's: SNR = P / (F G h nu n R) over the symbol rate R, with the "
        "total power P, the noise figure F, the gain G and n spans; BER = 0.5 erfc(sqrt(SNR/2))",
        "estimation": "the example re-draws the noise of the last span only: the previous "
        "spans keep one realisation, so a BER estimate carries the bias of that realisation",
        "spans_recorded": spans,
    }
    ASE_FIXTURE.write_text(dumps(ase_doc), encoding="utf-8")
    print(f"wrote {ASE_FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
