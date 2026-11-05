"""The three figures of README.md, in ``docs/python/_static/readme/``.

    python tools/readme_figures.py [--recording BASE --carrier-hz F]

1. ``coded_link.png``: ``specs/bicm_ldpc.yaml`` run on the Rust runner (LDPC (1024, 512)
   BICM with 16-QAM over AWGN) against the uncoded 16-QAM closed form and the BICM
   limit of the code rate.
2. ``optical_link.png``: ``specs/ssfm_dbp.yaml`` (16-QAM at 32 GBd over 10 × 80 km:
   dispersion compensation against digital back-propagation).
3. ``lte_cell.png``: the cell search and the MIB of an LTE capture. With
   ``--recording`` (a SigMF recording of a software radio and the centre frequency of
   the carrier), the figure is drawn from the measurement; without, from a cell
   simulated by :func:`comnum.lte_frame` through a frequency offset and noise, so that
   the figure can always be regenerated. The committed figure comes from a HackRF One
   capture (band 28, not committed).

Needs ``matplotlib`` (``pip install --group test``) and the extension built
(``just develop``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import comnum

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "python" / "_static" / "readme"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def coded_link() -> None:
    res = comnum.run_spec(ROOT / "specs" / "bicm_ldpc.yaml", progress=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for metric, color, marker, label in (
        ("coded_ber", BLUE, "o", "LDPC (1024, 512), 16-QAM BICM: bit error rate"),
        ("bler", AQUA, "s", "block error rate"),
    ):
        x, v, lo, hi = res.series("ebn0_db", metric, interval=True)
        ax.errorbar(
            x, v, [v - lo, hi - v], color=color, marker=marker, ms=5, lw=2, capsize=3, label=label
        )
    xx = np.linspace(0, 10, 60)
    ax.semilogy(
        xx,
        comnum.theory_ber(16, xx, "qam"),
        color=INK2,
        lw=1.2,
        ls="--",
        label="uncoded 16-QAM (closed form)",
    )
    # the BICM limit of 16-QAM at 2 bits per symbol, in Eb/N0 (Caire, Taricco & Biglieri 1998)
    c16 = comnum.Constellation("qam", 16).normalized()
    lo_db, hi_db = 0.0, 15.0
    for _ in range(40):
        mid = (lo_db + hi_db) / 2
        lo_db, hi_db = (mid, hi_db) if comnum.theory_bicm_capacity(c16, mid) < 2.0 else (lo_db, mid)
    limit = lo_db - 10 * np.log10(2.0)
    ax.axvline(
        limit, color=ORANGE, lw=1.2, ls=":", label=f"BICM limit at rate 1/2 ({limit:.2f} dB)"
    )
    ax.set(xlabel="$E_b/N_0$ (dB)", ylabel="error rate", xlim=(0, 10), ylim=(1e-6, 1))
    ax.set_title(
        "A coded link: specs/bicm_ldpc.yaml on the parallel runner", fontsize=10, color=INK
    )
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "coded_link.png", dpi=110)


def optical_link() -> None:
    res = comnum.run_spec(ROOT / "specs" / "ssfm_dbp.yaml", progress=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = {
        "cdc": "dispersion compensation",
        "dbp1": "DBP, 1 step per span",
        "dbp4": "DBP, 4 steps",
        "dbp20": "DBP, 20 steps",
    }
    for (name, label), color, marker in zip(
        labels.items(), [INK2, YELLOW, AQUA, BLUE], "osD^", strict=True
    ):
        x, snr = res.series("power_dbm", "snr_db", receiver=name)
        ax.plot(x, snr, color=color, marker=marker, ms=5, lw=2, label=label)
    ax.set(xlabel="launch power per channel (dBm)", ylabel="effective SNR (dB)")
    ax.set_title(
        "An optical link: 16-QAM at 32 GBd over 10 × 80 km (specs/ssfm_dbp.yaml)",
        fontsize=10,
        color=INK,
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "optical_link.png", dpi=110)


def lte_cell(recording: Path | None, carrier_hz: float) -> None:
    if recording is not None:
        x, fs, meta = comnum.read_sigmf(recording, channel=0)
        tuning = meta["captures"][0]["core:frequency"]
        source = f"HackRF One, {fs / 1e6:.2f} MS/s, tuned to {tuning / 1e6:.1f} MHz; carrier at {carrier_hz / 1e6:.1f} MHz"
    else:
        rng = comnum.Rng(55)
        fs, tuning, carrier_hz = 15.36e6, 768.5e6, 770.5e6
        frames = [
            comnum.lte_frame(55, sfn=s, n_rb=50, ports=2, n_fft=1024, rng=rng)
            for s in range(700, 720)
        ]
        h = np.array([0.8 + 0.3j, -0.2 + 0.5j])
        x = np.concatenate([h @ f for f in frames])
        x = comnum.resample_rational(x, 7.68e6, fs, 3.5e6)
        n = np.arange(len(x))
        x = x * np.exp(2j * np.pi * (carrier_hz - tuning - 13.1e3) * n / fs)
        gen = np.random.default_rng(55)
        x = x + np.sqrt(np.mean(np.abs(x) ** 2) / 10) * (
            gen.standard_normal(len(x)) + 1j * gen.standard_normal(len(x))
        ) / np.sqrt(2)
        source = "a simulated cell (comnum.lte_frame), 10 dB of SNR, the oscillator 17 ppm off"
    y = comnum.select_channel(x, fs, carrier_hz - tuning, 1.92e6, 0.6e6)
    cell = comnum.lte_cell_search(y, max_cfo_hz=25e3)[0]
    frames = comnum.lte_decode_mib(y, cell)
    # the PSS correlation over the first 20 ms, after the offset is removed
    yy = y * np.exp(-2j * np.pi * cell["cfo_hz"] * np.arange(len(y)) / 1.92e6)
    pss = comnum.lte_pss(cell["n_cell_id"] % 3)
    grid = np.concatenate([[0], pss[31:], np.zeros(128 - 63), pss[:31]])
    replica = np.fft.ifft(grid)
    seg = yy[: int(0.0205 * 1.92e6)]
    c = np.abs(np.correlate(seg, replica, "valid")) ** 2
    e = np.convolve(np.abs(seg) ** 2, np.ones(128), "valid") * np.vdot(replica, replica).real
    fig, (ax, tab) = plt.subplots(1, 2, figsize=(10.5, 4.0), width_ratios=[3, 2])
    ax.plot(np.arange(len(c)) / 1.92e3, c / e, color=BLUE, lw=1)
    ax.set(xlabel="time (ms)", ylabel="normalised PSS correlation", ylim=(0, 1))
    ax.set_title(
        f"cell {cell['n_cell_id']}: the PSS every 5 ms, offset {cell['cfo_hz'] / 1e3:+.2f} kHz",
        fontsize=10,
        color=INK,
    )
    tab.axis("off")
    total = (len(y) - cell["frame_start"] - 1920) // 19_200 + 1
    lines = [f"MIB decoded in {len(frames)} of {total} frames", ""]
    for f in frames[:6]:
        lines.append(
            f"SFN {f['sfn']:4d}   {f['ports']} ports   {f['n_rb']:3d} RB   PHICH N_g = {f['phich_ng']:.3f}"
        )
    if len(frames) > 6:
        lines.append("…")
    tab.text(0.0, 0.95, "\n".join(lines), family="monospace", fontsize=9.5, color=INK, va="top")
    fig.suptitle(f"A measured signal: {source}", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "lte_cell.png", dpi=110)
    print(
        f"LTE: cell {cell['n_cell_id']}, {len(frames)} MIB of {total} frames, offset {cell['cfo_hz']:.0f} Hz"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--recording", type=Path, help="a SigMF recording (base name or .sigmf-meta)")
    p.add_argument(
        "--carrier-hz", type=float, default=0.0, help="the centre of the LTE carrier (Hz)"
    )
    p.add_argument("--only", choices=["coded", "optical", "lte"], help="draw one figure")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "coded"):
        coded_link()
    if a.only in (None, "optical"):
        optical_link()
    if a.only in (None, "lte"):
        lte_cell(a.recording, a.carrier_hz)


if __name__ == "__main__":
    main()
