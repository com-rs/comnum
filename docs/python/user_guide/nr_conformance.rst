5G NR PDSCH: the conformance tests and the industry
====================================================

The PDSCH of comnum is checked three ways, each answering a question the others
cannot: is the transmitter the one of the standard (bit for bit against two other
implementations), does the link meet the requirement of the standard (TS 38.101-4), and
does it perform like the simulators of the industry (the RAN4 alignment that set that
requirement). This page states what is simulated, what each comparison found, and what
remains idealised.

What is simulated
-----------------

Test 1-2 of TS 38.101-4 V19.4.0 (clause 5.2.2.1.1, FDD, normal performance), the
conditions of its Table 5.2.2.1.1-2 and of the fixed reference channel R.PDSCH.1-1.2:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Part
     - What comnum does
   * - Transport block
     - 480 random bits per slot, TBS of TS 38.214 Sec. 5.1.3.2 (MCS 4 of Table 5.1.3.1-1,
       6 PRBs, 12 symbols, 12 DM-RS REs per PRB); CRC16, one code block, LDPC base graph 2,
       rate matching of TS 38.212 Sec. 5.4.2 with the redundancy version of the
       transmission (:class:`comnum.NrTransportBlock`)
   * - Slot
     - 14 symbols at 15 kHz, PDSCH on symbols 2 to 13, DM-RS of port 1000 on symbols 2 and
       11 (one CDM group without data, the odd subcarriers carry data), Gold scrambling,
       QPSK of TS 38.211 Sec. 5.1; no PDSCH in slot 0 of every 20 (the SS/PBCH block), the
       TRS of slots 10 and 11 around which the data are rate-matched (:class:`comnum.NrPdsch`)
   * - Precoding
     - 2 transmit antennas, a Type I precoder of TS 38.214 Table 5.2.2.2.1-1 drawn at
       random for every bundle of 2 PRBs at every slot; the receiver does not know it
   * - HARQ
     - 4 processes served in turn, at most 4 transmissions of redundancy versions 0, 2,
       3, 1, the soft buffers combined (:class:`comnum.NrHarqProcess`)
   * - Channel
     - TDLC300 of TS 38.101-4 Table B.2.1.1-4 (12 taps, 300 ns), Clarke/Jakes Doppler of
       100 Hz, independent links (ULA Low), applied per subcarrier at the middle of every
       OFDM symbol with the delays of the table; white noise of
       :math:`10^{-\mathrm{SNR}/10}` per RE and receive antenna
   * - Receiver
     - the precoded channel estimated on the DM-RS of every bundle by the 2-D Wiener filter
       of :mod:`comnum` (Hoeher, Kaiser & Robertson 1997) designed for delays up to the
       cyclic prefix and a Doppler of 100 Hz; maximum ratio combining; exact LLRs with the
       noise and the estimation error of every RE
   * - Metric
     - the fraction of the PDSCH slots that deliver their transport block; the SNR where
       it crosses 70 %

The transmitter, bit for bit
----------------------------

The transport blocks and the PDSCH symbols are compared with recorded outputs of two
independent implementations (``tests/oracle/test_nr_pdsch_oracle.py``): the ``TBEncoder`` of
Sionna 2.2.0 (CRC, segmentation, LDPC of both base graphs, rate matching, scrambling;
three FRCs, one to six code blocks, QPSK to 256-QAM) and py3gpp 0.6.0 (the TBS of 108
allocations, the four redundancy versions of base graph 1, the scrambled symbols). They
agree to the bit. The comparison found two errors of py3gpp 0.6.0 (its encoder of base
graph 2 at :math:`Z_c = 64`, its ``nrPDSCHIndices``), not of comnum.

The requirement and the industry
--------------------------------

The requirements of TS 38.101-4 were set by RAN4 from simulations of the companies,
each with its own simulator under common assumptions (R4-1811723), plus an
implementation margin of about 2 dB for what a handset loses. The four results of the
alignment of RAN4 #88bis (Chengdu, October 2018; contributions R4-1812164, R4-1812461,
R4-1813439 and R4-1813632, `3GPP archive
<https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_88Bis/Docs/>`_) are anonymised here as
companies 1 to 4, numbered by their result with 2 receive antennas.

.. list-table:: SNR of 70 % of the maximum throughput (dB), test 1-2, TDLC300-100
   :header-rows: 1
   :widths: 46 27 27

   * - Receiver
     - 2 TX x 2 RX
     - 2 TX x 4 RX
   * - comnum, the channel known
     - −3.88
     - −7.40
   * - comnum, Wiener estimator matched to TDLC300 (2.6 µs)
     - −2.48
     - −5.37
   * - companies 1 to 4 (RAN4 #88bis)
     - −2.33 to −1.84
     - −5.45 to −4.87
   * - comnum, robust Wiener estimator (cyclic prefix)
     - −1.89
     - −4.89
   * - requirement, TS 38.101-4
     - 0.2
     - −2.9

.. figure:: ../auto_examples/systems/images/sphx_glr_plot_06_nr_pdsch_conformance_001.png
   :width: 100%

   The three receivers of comnum, the 70 % points of the four companies and the curves of
   two of them (:ref:`sphx_glr_auto_examples_systems_plot_06_nr_pdsch_conformance.py`).

What the comparison shows:

- the four companies lie between the robust and the matched estimators of comnum: their
  spread, 0.5 to 0.6 dB, is that of the designs of their channel estimators, and every
  one of them is 1.5 to 2.5 dB above a known channel, the cost of estimating the
  channel on the DM-RS;
- the robust receiver of comnum is at the conservative edge of the industry, 0.15 and
  0.24 dB above the mean of the companies, and its curves follow those of two of them
  within 0.035 and 0.05 of throughput, a little steeper;
- under the conditions of the alignment rather than those of the final standard (the
  channel of TR 38.901 without simplification, a transmitter EVM of 6 %, the TRS in
  symbols 4 and 8) the result moves by less than the Monte Carlo, about 0.1 dB;
- the comparison found an error of comnum, since fixed: its LLRs counted the estimation
  error once per receive antenna (−1.75 dB before, −1.89 dB after, with 2 antennas).

What remains idealised: the receiver knows the noise variance and is perfectly
synchronised, and the channel has no inter-carrier interference (100 Hz over 15 kHz).
Only test 1-2 is run; the tests at 400 Hz, in 256-QAM and with two layers are not.

Reproducing it
--------------

The card ``validation/cards/ts38101_4_pdsch.yaml`` holds every value of this page and its
tolerance; the campaign measures them and records the curves of the figure::

    just card ts38101_4_pdsch

(about seven minutes in release). ``cargo run --release -p comnum-validation --example ts38101_4_alignment`` runs the
variants of the alignment conditions (an exploration, not a validation).
