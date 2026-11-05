"""The PDSCH transmitter of comnum against two independent implementations of 5G NR.

The transmitter of ``comnum.NrTransportBlock`` and ``comnum.NrPdsch`` (TS 38.212 Sec.
7.2, TS 38.211 Sec. 7.3.1, TS 38.214 Sec. 5.1.3) is compared bit for bit with recorded
outputs (``tests/oracle/__init__.py``):

- Sionna 2.2.0 (NVIDIA, Apache-2.0): its ``TBEncoder`` with ``channel_type="PDSCH"``,
  the transport block CRC, the segmentation, the LDPC code of both base graphs, the rate
  matching with the bit interleaver and the scrambling of ``c_init = n_RNTI 2^15 + n_ID``,
  for the transport blocks of three FRCs of TS 38.101-4 (R.PDSCH.1-1.2, 1-2.1 and 1-4.1:
  TBS 480 to 45096, one to six code blocks, QPSK to 256-QAM; R.PDSCH.1-1.1, base graph 1
  at R = 0.30, is beyond its encoder, which needs a rate above 1/3 for base graph 1);
- py3gpp 0.6.0 (AGPL-3.0, a reimplementation of the 5G functions of MATLAB): ``nrTBS``
  on a grid of allocations and MCS, the codewords of ``nrCRCEncode``,
  ``nrCodeBlockSegmentLDPC``, ``nrLDPCEncode`` and ``nrRateMatchLDPC`` for the four
  redundancy versions (base graph 1), and the symbols of ``nrPDSCH`` (scrambling and the
  modulation mapper, QPSK to 256-QAM).

Two functions of py3gpp 0.6.0 are not used, for errors of their own: its LDPC encoder of
base graph 2 at ``Z_c = 64`` gives a word that is not a codeword (its default algorithm
fails on the dimensions; the other gives parity bits that the parity checks of TS 38.212
reject, where Sionna agrees with comnum to the bit), and ``nrPDSCHIndices`` removes
whole DM-RS symbols (as with two CDM groups without data) counted from the first
allocated symbol. The order of the REs is checked by the channel bits of the FRCs (1584
and 1512 for R.PDSCH.1-1.2), which need the data on the odd subcarriers of the DM-RS
symbols.

Record with::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_nr_pdsch_oracle.py
"""

import numpy as np
import pytest

import comnum
from oracle import oracle

# (TBS, R x 1024, G, Q_m) of R.PDSCH.1-1.2, 1-1.1, 1-2.1, 1-4.1 (TS 38.101-4 Annex A.3.2)
FRCS = [(480, 308, 1584, 2), (3904, 308, 13104, 2), (13064, 490, 27456, 4), (45096, 841, 54912, 8)]
# the FRCs that Sionna's encoder accepts (base graph 1 needs R > 1/3)
SIONNA_FRCS = [f for f in FRCS if f[0] != 3904]
MODULATION = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
RNTI, N_ID = 1, 0


def _sionna():
    import torch
    from sionna.phy.nr import TBEncoder

    out = {}
    rng = np.random.default_rng(38_212)
    for tbs, r, g, qm in SIONNA_FRCS:
        bits = rng.integers(0, 2, tbs).astype(np.float64)
        row = {"tb": bits.astype(np.uint8)}
        for scrambled in (False, True):
            enc = TBEncoder(
                tbs, g, r / 1024, qm, 1, n_rnti=RNTI, n_id=N_ID, channel_type="PDSCH",
                use_scrambler=scrambled, precision="double",
            )  # fmt: skip
            assert enc.tb_size == tbs
            c = enc(torch.tensor(bits[None, :])).numpy().ravel().astype(np.uint8)
            row["scrambled" if scrambled else "coded"] = c
        out[str(tbs)] = row
    return out


def _py3gpp():
    from py3gpp.nrCodeBlockSegmentLDPC import nrCodeBlockSegmentLDPC
    from py3gpp.nrCRCEncode import nrCRCEncode
    from py3gpp.nrDLSCHInfo import nrDLSCHInfo
    from py3gpp.nrLDPCEncode import nrLDPCEncode
    from py3gpp.nrPDSCH import nrPDSCH
    from py3gpp.nrRateMatchLDPC import nrRateMatchLDPC
    from py3gpp.nrTBS import nrTBS

    rng = np.random.default_rng(38_214)
    tbs = {
        f"{qm}_{n_prb}_{n_re}_{r}": int(nrTBS(MODULATION[qm], 1, n_prb, n_re, r / 1024))
        for qm, r in [(2, 120), (2, 308), (4, 490), (6, 517), (6, 948), (8, 841)]
        for n_prb in [1, 6, 25, 52, 106, 273]
        for n_re in [132, 126, 120]
    }
    codewords = {}
    for a, r, g, qm in FRCS[1:]:  # base graph 1 only (see the module)
        info = nrDLSCHInfo(a, r / 1024)
        bits = rng.integers(0, 2, a).astype(np.int8)
        cbs = nrCodeBlockSegmentLDPC(nrCRCEncode(bits, info["CRC"]), info["BGN"])
        enc = nrLDPCEncode(cbs, info["BGN"], algo="thangaraj")
        row = {"tb": bits.astype(np.uint8), "bgn": int(info["BGN"])}
        for rv in range(4):
            row[f"rv{rv}"] = (
                np.asarray(nrRateMatchLDPC(enc, g, rv, MODULATION[qm], 1)).ravel().astype(np.uint8)
            )
        codewords[str(a)] = row
    symbols = {}
    for qm in (2, 4, 6, 8):
        bits = rng.integers(0, 2, 792 * qm).astype(np.uint8)
        sym = np.asarray(nrPDSCH([bits.astype(int)], [MODULATION[qm]], 1, N_ID, RNTI)).ravel()
        symbols[str(qm)] = {"bits": bits, "symbols": sym}
    return {"tbs": tbs, "codewords": codewords, "symbols": symbols}


@pytest.fixture(scope="module")
def sionna():
    return oracle("sionna_nr_pdsch", _sionna, tool="Sionna")


@pytest.fixture(scope="module")
def py3gpp():
    return oracle("nr_pdsch_py3gpp", _py3gpp, tool="py3gpp")


@pytest.mark.parametrize("frc", SIONNA_FRCS, ids=[str(f[0]) for f in SIONNA_FRCS])
def test_the_transport_blocks_are_those_of_sionna(sionna, frc):
    tbs, r, g, qm = frc
    ref = sionna[str(tbs)]
    tb = comnum.NrTransportBlock(tbs, r / 1024)
    coded = tb.encode(ref["tb"], g, qm)
    np.testing.assert_array_equal(coded, ref["coded"])
    scrambled = comnum.Scrambler((RNTI << 15) + N_ID).scramble(coded)
    np.testing.assert_array_equal(scrambled, ref["scrambled"])


def test_the_tbs_is_that_of_py3gpp(py3gpp):
    for key, want in py3gpp["tbs"].items():
        qm, n_prb, n_re, r = (int(v) for v in key.split("_"))
        table = "qam256" if qm == 8 else "qam64"
        # the MCS index of the (Q_m, R) of the key in its table
        index = next(i for i in range(29) if comnum.nr_mcs(table, i) == (qm, float(r)))
        # N'_RE = 12 * 12 - n_dmrs with 12 symbols
        assert comnum.nr_tbs(n_prb, 12, 144 - n_re, table, index) == want, key


@pytest.mark.parametrize("tbs", [str(f[0]) for f in FRCS[1:]])
def test_the_redundancy_versions_are_those_of_py3gpp(py3gpp, tbs):
    a, r, g, qm = next(f for f in FRCS if str(f[0]) == tbs)
    ref = py3gpp["codewords"][tbs]
    tb = comnum.NrTransportBlock(a, r / 1024)
    assert tb.base_graph == ref["bgn"] == 1
    for rv in range(4):
        np.testing.assert_array_equal(
            tb.encode(ref["tb"], g, qm, rv=rv), ref[f"rv{rv}"], err_msg=f"rv {rv}"
        )


@pytest.mark.parametrize("qm", [2, 4, 6, 8])
def test_the_pdsch_symbols_are_those_of_py3gpp(py3gpp, qm):
    ref = py3gpp["symbols"][str(qm)]
    p = comnum.NrPdsch(52, 23, 6, qm, dmrs_additional_position=1, rnti=RNTI, n_id=N_ID)
    grid = p.transmit(ref["bits"])[0]
    res = p.data_res()
    np.testing.assert_allclose(grid[res[:, 0], res[:, 1]], ref["symbols"], atol=1e-12)
