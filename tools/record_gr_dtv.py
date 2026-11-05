#!/usr/bin/env python3
"""Record the DVB-T and DVB-S2 transmitters of GNU Radio (gr-dtv) that comnum is compared with.

gr-dtv (GNU Radio, GPL-3.0, <https://github.com/gnuradio/gnuradio>, the ``dtv`` module)
implements the DVB-T transmitter of ETSI EN 300 744 block by block. This script runs
its chain on random transport packets (seed ``SEED``), as the example flowgraph
``examples/dtv/dvbt_tx_2k.grc`` of GNU Radio connects it, and records three taps:

- the bytes after the outer coding (``dvbt_energy_dispersal``,
  ``dvbt_reed_solomon_enc``, ``dvbt_convolutional_interleaver``), common to DVB-T and
  DVB-S (EN 300 421 Secs. 4.4.1 to 4.4.3);
- the output of ``dvbt_inner_coder`` for QPSK at the five code rates: two punctured
  bits per byte, the pairs that DVB-S sends on I and Q (EN 300 421 Table 2);
- the OFDM symbols after ``dvbt_reference_signals`` (data cells mapped by ``dvbt_map``,
  pilots and TPS) of the configurations ``DVBT``, as their ``K`` cells: the FFT of every
  symbol, each cell recorded as the index of its value in an alphabet of the values
  that occur (the cells of a symbol take a few dozen values: the constellation, the
  pilots, the TPS), and the samples of the first symbols of the first configuration
  after ``digital.ofdm_cyclic_prefixer``.

It also runs the DVB-S2 transmitter of ETSI EN 302 307-1 as ``examples/dtv/dvbs2_tx.grc``
connects it (``dvb_bbheader_bb`` on the transport stream, ``dvb_bbscrambler_bb``,
``dvb_bch_bb``, ``dvb_ldpc_bb``, ``dvbs2_interleaver_bb``, ``dvbs2_modulator_bc``,
``dvbs2_physical_cc``) for the configurations of ``dvbs2_configs`` and records the first
scrambled BBFRAME and the first PLFRAME of each (its symbols as indices in an alphabet).

GNU Radio is not a dependency of comnum: the script runs with the Python of an
environment that has it (conda-forge, ``gnuradio``) and writes
``tests/oracle/gr_dtv.{npz,json}`` and ``tests/oracle/gr_dtv_dvbs2.{npz,json}`` (the
provenance in the JSON documents), read by
``tests/oracle/test_gr_dtv.py``::

    micromamba create -n gnuradio -c conda-forge gnuradio
    ~/mamba/envs/gnuradio/bin/python tools/record_gr_dtv.py

The files are written with the functions of the recorded oracles (``tests/oracle/__init__.py``):
recording again with the same version produces no diff.
"""

from __future__ import annotations

import platform
import sys
from fractions import Fraction

import numpy as np
from _common import ROOT

sys.path.insert(0, str(ROOT / "tests"))
from oracle import DIR as OUT
from oracle import archive, dumps

SEED = 2026
RATES = ["1/2", "2/3", "3/4", "5/6", "7/8"]
# mode, guard interval, constellation, code rate, OFDM symbols recorded
DVBT = [
    ("2K", "1/32", "QPSK", "1/2", 100),
    ("2K", "1/8", "16-QAM", "3/4", 20),
    ("2K", "1/4", "64-QAM", "2/3", 20),
    ("8K", "1/16", "16-QAM", "5/6", 6),
    ("8K", "1/32", "64-QAM", "7/8", 6),
]
TIME_SYMBOLS = 4  # samples (guard interval included) of the first symbols of DVBT[0]
DVBS_PACKETS = 48
DVBS2_PACKETS = 400
# the MODCODs of EN 302 307-1 Table 12
MODCODS = [
    ("QPSK", ["1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "5/6", "8/9", "9/10"]),
    ("8PSK", ["3/5", "2/3", "3/4", "5/6", "8/9", "9/10"]),
    ("16APSK", ["2/3", "3/4", "4/5", "5/6", "8/9", "9/10"]),
    ("32APSK", ["3/4", "4/5", "5/6", "8/9", "9/10"]),
]
# K_bch of Tables 5a and 5b
DVBS2_K = {
    "normal": {"1/4": 16008, "1/3": 21408, "2/5": 25728, "1/2": 32208, "3/5": 38688,
               "2/3": 43040, "3/4": 48408, "4/5": 51648, "5/6": 53840, "8/9": 57472,
               "9/10": 58192},
    "short": {"1/4": 3072, "1/3": 5232, "2/5": 6312, "1/2": 7032, "3/5": 9552, "2/3": 10632,
              "3/4": 11712, "4/5": 12432, "5/6": 13152, "8/9": 14232},
}  # fmt: skip


def packets(n: int) -> np.ndarray:
    """``n`` transport packets: the sync byte 0x47 and 187 random bytes."""
    p = np.random.default_rng(SEED).integers(0, 256, (n, 188), dtype=np.uint8)
    p[:, 0] = 0x47
    return p.ravel()


def alphabet(x: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    """The values that occur in ``x`` (complex, float32 rounding) and the index of every
    element among them: the compact form of a signal built from a few values."""
    key = np.round(x / np.abs(x).max() * 1e3)
    keys, index = np.unique(key.ravel(), return_inverse=True)
    values = np.array([x.ravel()[index == j].mean() for j in range(len(keys))])
    spread = np.abs(x.ravel() - values[index]).max() / np.abs(x).max()
    if len(values) > 256 or spread > 1e-5:
        raise RuntimeError(f"{what}: {len(values)} values, spread {spread}")
    return values.astype(np.complex128), index.reshape(x.shape).astype(np.uint8)


def main() -> None:
    dvbt_dvbs()
    dvbs2()


def dvbt_dvbs() -> None:
    from gnuradio import blocks, digital, dtv, gr

    constellations = {"QPSK": dtv.MOD_QPSK, "16-QAM": dtv.MOD_16QAM, "64-QAM": dtv.MOD_64QAM}
    bits = {"QPSK": 2, "16-QAM": 4, "64-QAM": 6}
    rates = dict(zip(RATES, [dtv.C1_2, dtv.C2_3, dtv.C3_4, dtv.C5_6, dtv.C7_8], strict=True))
    modes = {"2K": (dtv.T2k, 2048, 1705, 1512), "8K": (dtv.T8k, 8192, 6817, 6048)}
    guards = {"1/32": dtv.GI_1_32, "1/16": dtv.GI_1_16, "1/8": dtv.GI_1_8, "1/4": dtv.GI_1_4}

    def outer() -> list:
        # the parameters of examples/dtv/dvbt_tx_2k.grc: the shortened RS(204, 188) as
        # RS(255, 239) with 51 leading zeros, groups of 8 packets, I = 12, M = 17
        return [
            dtv.dvbt_energy_dispersal(1),
            dtv.dvbt_reed_solomon_enc(2, 8, 0x11D, 255, 239, 8, 51, 8),
            dtv.dvbt_convolutional_interleaver(136, 12, 17),
        ]

    def dvbt(mode: str, guard: str, c: str, r: str) -> list:
        # fresh blocks for every run: a block keeps its state (registers, counters)
        tm, n_fft, _, n_data = modes[mode]
        return [
            *outer(),
            dtv.dvbt_inner_coder(1, n_data, constellations[c], dtv.NH, rates[r]),
            dtv.dvbt_bit_inner_interleaver(n_data, constellations[c], dtv.NH, tm),
            dtv.dvbt_symbol_inner_interleaver(n_data, tm, 1),
            dtv.dvbt_map(n_data, constellations[c], dtv.NH, tm, 1.0),
            dtv.dvbt_reference_signals(
                gr.sizeof_gr_complex, n_data, n_fft, constellations[c], dtv.NH,
                rates[r], rates[r], guards[guard], tm, 0, 0,
            ),
        ]  # fmt: skip

    def run(src: np.ndarray, chain: list, sink) -> np.ndarray:
        tb = gr.top_block()
        tb.connect(blocks.vector_source_b(src.tolist(), False), *chain, sink)
        tb.run()
        return np.array(sink.data())

    def need(bytes_per_symbol: float, n_symbols: int) -> int:
        # whole groups of 8 packets, with margin for the latency of the flowgraph
        n = int(np.ceil((n_symbols + 2) * bytes_per_symbol / 188)) + 3 * 12
        return 8 * ((n + 7) // 8)

    arrays: dict[str, np.ndarray] = {}
    n_max = max(
        need(cells * bits[c] * float(Fraction(r)) / 8, n)
        for mode, _, c, r, n in DVBT
        for cells in [modes[mode][3]]
    )
    src = packets(max(n_max, DVBS_PACKETS))
    arrays["packets"] = src

    arrays["outer"] = run(src[: DVBS_PACKETS * 188], outer(), blocks.vector_sink_b(1))
    for r in RATES:
        coder = dtv.dvbt_inner_coder(1, 1512, dtv.MOD_QPSK, dtv.NH, rates[r])
        out = run(src[: DVBS_PACKETS * 188], [*outer(), coder], blocks.vector_sink_b(1512))
        arrays[f"inner.{r.replace('/', '_')}"] = out.astype(np.uint8)

    configs = []
    for i, (mode, guard, c, r, n_symbols) in enumerate(DVBT):
        _, n_fft, k, n_data = modes[mode]
        n_packets = need(n_data * bits[c] * float(Fraction(r)) / 8, n_symbols)

        time = run(src[: n_packets * 188], dvbt(mode, guard, c, r), blocks.vector_sink_c(n_fft))
        if len(time) < n_symbols * n_fft:
            raise RuntimeError(f"{mode} {c} {r}: {len(time) // n_fft} symbols out of {n_symbols}")
        spectrum = np.fft.fft(time[: n_symbols * n_fft].reshape(n_symbols, n_fft), axis=1)
        # the K carriers, carrier (K - 1)/2 at the centre frequency (bin 0)
        cells = spectrum[:, (np.arange(k) - (k - 1) // 2) % n_fft]
        name = f"dvbt.{i}"
        values, index = alphabet(cells, f"{mode} {c} {r}")
        arrays[f"{name}.alphabet"] = values
        arrays[f"{name}.cells"] = index
        if i == 0:
            prefixer = digital.ofdm_cyclic_prefixer(n_fft, n_fft + n_fft // int(guard[2:]), 0, "")
            samples = run(
                src[: n_packets * 188],
                [*dvbt(mode, guard, c, r), prefixer],
                blocks.vector_sink_c(1),
            )
            arrays[f"{name}.samples"] = samples[: TIME_SYMBOLS * (n_fft + n_fft // int(guard[2:]))]
        configs.append(
            {"mode": mode, "guard": guard, "constellation": c, "rate": r,
             "symbols": n_symbols, "packets": n_packets}
        )  # fmt: skip

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gr_dtv.npz").write_bytes(archive(arrays))
    doc = {
        "tool": "GNU Radio gr-dtv",
        "version": gr.version(),
        "repository": "https://github.com/gnuradio/gnuradio",
        "license": "GPL-3.0-or-later (the outputs, not the code)",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": SEED,
        "flowgraph": "examples/dtv/dvbt_tx_2k.grc: energy dispersal, RS(255, 239) shortened by 51,"
        " convolutional interleaver (12, 17), inner coder, bit and symbol inner interleavers,"
        " map (gain 1), reference signals (no cell id), cyclic prefixer",
        "dvbs_packets": DVBS_PACKETS,
        "dvbs_rates": RATES,
        "dvbt": configs,
        "time_symbols": TIME_SYMBOLS,
    }
    (OUT / "gr_dtv.json").write_text(dumps(doc), encoding="utf-8")


def dvbs2() -> None:
    from gnuradio import blocks, dtv, gr

    arrays: dict[str, np.ndarray] = {}
    configs = []
    src = packets(DVBS2_PACKETS)
    for i, (modulation, rate, frame, pilots, gold) in enumerate(dvbs2_configs()):
        f = dtv.FECFRAME_NORMAL if frame == "normal" else dtv.FECFRAME_SHORT
        m = getattr(dtv, f"MOD_{modulation}")
        r = getattr(dtv, f"C{rate.replace('/', '_')}")
        s2 = dtv.STANDARD_DVBS2
        # the parameters of examples/dtv/dvbs2_tx.grc: transport stream, roll-off 0.20
        header = dtv.dvb_bbheader_bb(
            s2, f, r, dtv.RO_0_20, dtv.INPUTMODE_NORMAL, dtv.INBAND_OFF, 168, 4000000
        )
        scrambler = dtv.dvb_bbscrambler_bb(s2, f, r)
        chain = [
            header, scrambler, dtv.dvb_bch_bb(s2, f, r), dtv.dvb_ldpc_bb(s2, f, r, dtv.MOD_OTHER),
            dtv.dvbs2_interleaver_bb(f, r, m), dtv.dvbs2_modulator_bc(f, r, m, dtv.INTERPOLATION_OFF),
            dtv.dvbs2_physical_cc(f, r, m, dtv.PILOTS_ON if pilots else dtv.PILOTS_OFF, gold),
        ]  # fmt: skip
        tb = gr.top_block()
        bbframes, symbols = blocks.vector_sink_b(), blocks.vector_sink_c()
        tb.connect(blocks.vector_source_b(src.tolist(), False), *chain, symbols)
        tb.connect(scrambler, bbframes)
        tb.run()
        # the physical layer writes every symbol followed by a zero (for its filter)
        x = np.array(symbols.data())
        if np.abs(x[1::2]).max() != 0:
            raise RuntimeError(f"{modulation} {rate} {frame}: odd samples are not zero")
        x = x[::2]
        bb = np.array(bbframes.data(), np.uint8)
        k_bch = DVBS2_K[frame][rate]
        n_symbols = plframe_len(frame, modulation, pilots)
        if len(bb) < k_bch or len(x) < n_symbols:
            raise RuntimeError(f"{modulation} {rate} {frame}: no whole frame")
        name = f"dvbs2.{i}"
        arrays[f"{name}.bbframe"] = np.packbits(bb[:k_bch])
        values, index = alphabet(x[:n_symbols], f"{modulation} {rate} {frame}")
        arrays[f"{name}.alphabet"] = values
        arrays[f"{name}.plframe"] = index
        configs.append(
            {"modulation": modulation, "rate": rate, "frame": frame, "pilots": pilots,
             "gold_code": gold, "k_bch": k_bch, "symbols": n_symbols}
        )  # fmt: skip

    (OUT / "gr_dtv_dvbs2.npz").write_bytes(archive(arrays))
    doc = {
        "tool": "GNU Radio gr-dtv",
        "version": gr.version(),
        "repository": "https://github.com/gnuradio/gnuradio",
        "license": "GPL-3.0-or-later (the outputs, not the code)",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": SEED,
        "flowgraph": "examples/dtv/dvbs2_tx.grc: BBHEADER (transport stream, roll-off 0.20),"
        " BB scrambler, BCH, LDPC, bit interleaver, modulator, physical layer (PLHEADER,"
        " pilots, scrambling); the first BBFRAME (scrambled) and the first PLFRAME",
        "dvbs2": configs,
    }
    (OUT / "gr_dtv_dvbs2.json").write_text(dumps(doc), encoding="utf-8")


def dvbs2_configs() -> list[tuple[str, str, str, bool, int]]:
    """Every MODCOD of Table 12 in normal frames without pilots; every one of short
    frames (no 9/10) with pilots; three normal frames with pilots and a Gold code."""
    out = [(m, r, "normal", False, 0) for m, rates in MODCODS for r in rates]
    out += [(m, r, "short", True, 0) for m, rates in MODCODS for r in rates if r != "9/10"]
    out += [("QPSK", "1/4", "normal", True, 1), ("8PSK", "3/5", "normal", True, 17),
            ("32APSK", "9/10", "normal", True, 262141)]  # fmt: skip
    return out


def plframe_len(frame: str, modulation: str, pilots: bool) -> int:
    """Symbols of a PLFRAME: 90 (S + 1) + 36 floor((S - 1)/16) with pilots (Sec. 5.5)."""
    bits = {"QPSK": 2, "8PSK": 3, "16APSK": 4, "32APSK": 5}[modulation]
    slots = (64800 if frame == "normal" else 16200) // bits // 90
    return 90 * (slots + 1) + (36 * ((slots - 1) // 16) if pilots else 0)


if __name__ == "__main__":
    main()
