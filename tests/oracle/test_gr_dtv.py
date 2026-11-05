"""comnum against the DVB-T and DVB-S2 transmitters of GNU Radio (gr-dtv, GPL-3.0), recorded.

``tools/record_gr_dtv.py`` runs the transmitter of gr-dtv (GNU Radio 3.10, the chain of
its example ``dvbt_tx_2k.grc``) on random transport packets and records, in
``tests/oracle/gr_dtv.npz`` (provenance in the ``.json``):

- the bytes after the outer coding (energy dispersal, RS(204, 188), convolutional
  interleaver I = 12, M = 17), common to DVB-T (EN 300 744 Secs. 4.3.1 to 4.3.2) and
  DVB-S (EN 300 421 Secs. 4.4.1 to 4.4.3);
- the punctured inner code of QPSK at the five rates, the pairs of bits that DVB-S
  sends on I and Q (EN 300 421 Table 2, the serial order of EN 300 744 Figure 6 read
  two by two);
- the cells of the OFDM symbols (data, scattered and continual pilots, TPS) of five
  configurations, 2K and 8K, QPSK to 64-QAM, rates 1/2 to 7/8, over 100 symbols for
  the first (the TPS of the second frame included), and the samples of its first four
  symbols, guard intervals included.

Conventions: gr-dtv's inner coder writes the pair of bits of a QPSK cell as one byte
``2 b_0 + b_1``; its cells carry the gain of its mapper and of its unnormalised inverse
FFT, a real positive factor that the comparison fits (one per configuration) and then
checks to be the same on every cell; the residual is the float32 rounding of GNU Radio.
"""

import json

import numpy as np
import pytest

import comnum
from oracle import DIR, missing

if not (DIR / "gr_dtv.npz").exists():
    missing("tests/oracle/gr_dtv.npz is missing: run tools/record_gr_dtv.py", module=True)
V = dict(np.load(DIR / "gr_dtv.npz"))
META = json.loads((DIR / "gr_dtv.json").read_text(encoding="utf-8"))
PACKETS = V["packets"]
# float32 samples of GNU Radio: relative error of a cell after the common gain
RTOL = 1e-5


def scaled_error(reference: np.ndarray, ours: np.ndarray) -> tuple[complex, float]:
    """The least-squares gain from ``ours`` to ``reference`` and the largest error
    relative to the largest value."""
    gain = np.vdot(ours, reference) / np.vdot(ours, ours)
    return gain, float(np.abs(reference - gain * ours).max() / np.abs(reference).max())


def test_outer_coding_is_that_of_gr_dtv():
    n = META["dvbs_packets"]
    ours = comnum.Dvbs("1/2").outer(PACKETS[: n * 188])
    theirs = V["outer"]
    assert len(theirs) >= (n - 12) * 204
    np.testing.assert_array_equal(ours[: len(theirs)], theirs)


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4", "5/6", "7/8"])
def test_dvbs_inner_code_is_that_of_gr_dtv(rate):
    n = META["dvbs_packets"]
    link = comnum.Dvbs(rate)
    symbols = link.inner(link.outer(PACKETS[: n * 188]))
    # Figure 5 of EN 300 421: a bit 0 is the positive level on its axis
    ours = 2 * (symbols.real < 0) + (symbols.imag < 0)
    theirs = V[f"inner.{rate.replace('/', '_')}"]
    assert len(theirs) > 10_000
    np.testing.assert_array_equal(ours[: len(theirs)], theirs)


@pytest.mark.parametrize("i", range(len(META["dvbt"])))
def test_dvbt_cells_are_those_of_gr_dtv(i):
    c = META["dvbt"][i]
    link = comnum.Dvbt(c["mode"], c["guard"], c["constellation"], c["rate"])
    data = comnum.Dvbs("1/2").outer(PACKETS[: c["packets"] * 188])
    ours = link.cells(data).T[: c["symbols"]]
    theirs = V[f"dvbt.{i}.alphabet"][V[f"dvbt.{i}.cells"]]
    gain, err = scaled_error(theirs, ours)
    assert gain.real > 0 and abs(gain.imag) < RTOL * gain.real
    assert err < RTOL, f"{c}: error {err:.2e}"


def test_dvbt_samples_are_those_of_gr_dtv():
    c = META["dvbt"][0]
    link = comnum.Dvbt(c["mode"], c["guard"], c["constellation"], c["rate"])
    theirs = V["dvbt.0.samples"]
    ours = link.modulate(PACKETS[: c["packets"] * 188])[: len(theirs)]
    gain, err = scaled_error(theirs, ours)
    assert gain.real > 0 and abs(gain.imag) < RTOL * gain.real
    assert err < RTOL, f"error {err:.2e}"


# DVB-S2 (EN 302 307-1): the first BBFRAME of the transport stream (BBHEADER of
# gr-dtv, then the BB scrambler) and its PLFRAME, for every MODCOD of Table 12 in
# normal frames, every one of short frames with pilots, and three Gold codes.
S2 = json.loads((DIR / "gr_dtv_dvbs2.json").read_text(encoding="utf-8"))["dvbs2"]
V2 = dict(np.load(DIR / "gr_dtv_dvbs2.npz"))


def s2_name(c):
    pilots = "pilots" if c["pilots"] else "no pilots"
    return f"{c['modulation']} {c['rate']} {c['frame']} {pilots} gold {c['gold_code']}"


@pytest.mark.parametrize("i", range(len(S2)), ids=[s2_name(c) for c in S2])
def test_dvbs2_plframe_is_that_of_gr_dtv(i):
    c = S2[i]
    link = comnum.Dvbs2(
        c["modulation"],
        c["rate"],
        frame=c["frame"],
        pilots=c["pilots"],
        scrambling_code=c["gold_code"],
    )
    assert link.k == c["k_bch"] and link.plframe_len == c["symbols"]
    bbframe = np.unpackbits(V2[f"dvbs2.{i}.bbframe"])[: c["k_bch"]]
    # the BBHEADER of gr-dtv (transport stream) passes comnum's CRC-8 once descrambled
    assert comnum.dvbs2_bbframe_data(bbframe)[1]
    theirs = V2[f"dvbs2.{i}.alphabet"][V2[f"dvbs2.{i}.plframe"]]
    ours = link.modulate(bbframe)
    gain, err = scaled_error(theirs, ours)
    assert abs(gain - 1) < RTOL, f"gain {gain}"
    assert err < RTOL, f"error {err:.2e}"
