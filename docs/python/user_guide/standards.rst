Standards
=========

The functions below follow a standard clause by clause: the tables are transcribed
from the documents, the values the documents print are checked by tests, and every
misprint found in a document is recorded with the test that shows it
(:doc:`../maturity`). The satellite link of DVB-S2 is in :doc:`wireless`.

Test sequences: ITU-T O.150
---------------------------

:func:`comnum.o150_sequence` gives the eight pseudo-random test sequences of the
bit-error-rate test sets, ITU-T Recommendation O.150 (05/96), Sec. 5: a shift
register of :math:`n` stages fed back from the stages :math:`k` and :math:`n`, the
polynomial :math:`x^n + x^k + 1`, period :math:`2^n - 1`.

=========  ========  ===  ===  ======================  ======================
Name       Clause    n    k    Output                  Longest run of zeros
=========  ========  ===  ===  ======================  ======================
prbs9      5.1       9    5    non-inverted            8
prbs11     5.2       11   9    non-inverted            10
prbs15     5.3       15   14   inverted                15
prbs20     5.4       20   3    non-inverted            19
prbs20z    5.5       20   17   zero suppression        14
prbs23     5.6       23   18   inverted                23
prbs29     5.7       29   27   inverted                29
prbs31     5.8       31   28   inverted                31
=========  ========  ===  ===  ======================  ======================

An inverted sequence is the negated register output: its longest run of zeros is
:math:`n` instead of :math:`n - 1` (Sec. 4). ``prbs20z`` forces its output to one
before 14 zeros (Sec. 5.5). Every register starts with its stages at one, the phase
the Recommendation gives for the 511-bit sequence ("the first ONE of 9 consecutive
ONEs"); an inverted sequence then begins with its :math:`n` zeros. A test set
synchronises to the received sequence, so the phase does not matter for a
measurement:

.. code-block:: python

   import numpy as np
   import comnum

   tx = comnum.o150_sequence("prbs15", 32767)        # one period, the O.151 sequence
   assert tx[:15].tolist() == [0] * 15                # inverted: the 15 zeros first

   rx = np.roll(tx, 1234)                              # the received stream, any phase
   rx[[10, 500, 20000]] ^= 1                           # three bit errors
   lag = int(np.argmax(np.correlate(1 - 2.0 * rx, 1 - 2.0 * tx, "full"))) - (len(tx) - 1)
   errors = int(np.count_nonzero(np.roll(tx, lag) != rx))   # 3

:func:`comnum.prbs` produces the same polynomials from any register state and never
inverts: a phase of the same maximal-length sequence, for a source that only needs
the statistics.

The eight sequences are checked over a whole period, up to :math:`2^{31} - 1` bits,
against Table 1 of the Recommendation (card ``o150_sec5``): the period, from the order
of :math:`x` modulo each polynomial; the longest run of zeros; the :math:`n - 1`
consecutive ones of an inverted sequence (Sec. 3); and the first bits of Sec. 5.1. The
Recommendation prints two misprints, both corrected by the text of the same clause:
the zero-suppression formula of Sec. 5.5 ends its OR with :math:`Q_6` instead of
:math:`Q_{19}`, and Sec. 5.8 calls the 31-stage register "twenty-nine-stage".

Satellite: DVB-S (ETSI EN 300 421)
----------------------------------

:class:`comnum.Dvbs` is the transmitter of ETSI EN 300 421 V1.1.2 and a receiver aligned
on the received burst, at one sample per symbol (the root-raised cosine of roll-off 0.35
of Sec. 4.5 is a :func:`comnum.pulse_shape` away):

=====================================  ==========  ===============================================
Function                               Clause      What it does
=====================================  ==========  ===============================================
energy dispersal                       4.4.1       the PRBS :math:`1 + X^{14} + X^{15}` loaded with
                                                   ``100101010000000`` every 8 packets, added to
                                                   every byte but the sync bytes; the first sync
                                                   byte of a group inverted to ``0xB8``
outer code                             4.4.2       RS(204, 188, T = 8), 16 parity bytes per packet
outer interleaver                      4.4.2       convolutional, 12 branches of depths
                                                   :math:`17 j` bytes (Figure 4)
inner code                             4.4.3       constraint length 7, :math:`X` from
                                                   :math:`171_8`, :math:`Y` from :math:`133_8`,
                                                   punctured by Table 2
mapping                                4.5         Gray QPSK of Figure 5, absolute, unit energy
=====================================  ==========  ===============================================

The input is a burst of whole MPEG-2 transport packets of 188 bytes, each starting with
the sync byte ``0x47`` (:func:`comnum.dvb_packets` draws random ones):

.. code-block:: python

   import numpy as np
   import comnum

   link = comnum.Dvbs("3/4")                      # 1/2, 2/3, 3/4, 5/6 or 7/8
   packets = comnum.dvb_packets(40, comnum.Rng(1))
   x = link.modulate(packets)                      # unit-energy QPSK symbols

   ebn0_db = 5.0                                   # per useful bit, as in Table 3
   noise_var = 1 / comnum.db2lin(link.esn0_db(ebn0_db))
   rng = np.random.default_rng(1)
   y = x + np.sqrt(noise_var / 2) * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))

   out, corrected, failed = link.decode(y, noise_var)   # soft Viterbi, RS(204, 188)
   assert np.array_equal(out, packets)

The receiver is split in two steps so that the error rate after the Viterbi decoder, the
quantity of Table 3, can be measured: :meth:`~comnum.Dvbs.outer` gives the bytes
entering the inner code (dispersal, RS, interleaver), :meth:`~comnum.Dvbs.inner` their
symbols, :meth:`~comnum.Dvbs.inner_decode` the bytes decided by the Viterbi decoder:

.. code-block:: python

   tx = link.outer(packets)
   rx = link.inner_decode(link.inner(tx) + w, noise_var)     # w: the noise
   ber_after_viterbi = np.unpackbits(tx ^ rx).mean()

The broadcast is continuous; a burst here starts with an empty interleaver flushed by 11
packets of zeros (its latency of 2244 bytes) and a trellis returned to state 0, so that
every packet of the burst decodes. The punctured bits leave in time order, :math:`X`
before :math:`Y`, alternately on I and Q, which gives the I and Q rows of Table 2 (rate
3/4: I = :math:`X_1 Y_2`, Q = :math:`Y_1 X_3`).

**Validation.** Table 3 of the standard gives the :math:`E_b/N_0` per useful bit of a
BER of :math:`2 \cdot 10^{-4}` after the Viterbi decoder, a modem implementation margin
of 0.8 dB included. The whole chain reaches it 0.10 to 0.22 dB below Table 3 minus the
margin (card ``dvbs_table3``, :math:`10^6` bits per point):

=====  ========  =================  =================
Rate   Table 3   Table 3 − 0.8 dB   comnum (ideal)
=====  ========  =================  =================
1/2    4.5 dB    3.7 dB             3.60 dB
2/3    5.0 dB    4.2 dB             4.04 dB
3/4    5.5 dB    4.7 dB             4.50 dB
5/6    6.0 dB    5.2 dB             5.00 dB
7/8    6.4 dB    5.6 dB             5.38 dB
=====  ========  =================  =================

Exact checks cover the first PRBS byte printed under Figure 2 (``0x03``), the 8-packet
groups, the sync bytes in the branch without delay of the interleaver, the patterns,
I and Q rows and free distances (10, 6, 5, 4, 3) of Table 2, the labels of Figure 5 and
the useful bit rates of Tables C.1 and D.1. The masks of Annex A, the satellite chain of
Annexes C and D and the acquisition of Annex B are not implemented. The example
:doc:`../auto_examples/systems/plot_03_dvbs` draws the waterfalls.

Terrestrial: DVB-T (ETSI EN 300 744)
------------------------------------

:class:`comnum.Dvbt` is the non-hierarchical transmitter of ETSI EN 300 744 V1.6.2 and a
receiver that knows the channel (the perfect channel estimation of Annex A):

=====================================  ============  ===============================================
Function                               Clause        What it does
=====================================  ============  ===============================================
outer coding                           4.3.1–4.3.2   as DVB-S: energy dispersal, RS(204, 188),
                                                     convolutional interleaver
inner code                             4.3.3         (171, 133), punctured by Table 2, continuous
inner interleaver                      4.3.4         demultiplexer into :math:`v` = 2, 4, 6 streams,
                                                     six bit interleavers of 126 bits, the symbol
                                                     interleaver :math:`H(q)` of the mode
mapping                                4.3.5         Figure 9, normalised by Table 6
frame                                  4.4–4.5       2K (1705 carriers) or 8K (6817), 68 symbols,
                                                     guard 1/4 to 1/32, scattered and continual
                                                     pilots at 4/3
TPS                                    4.6           68 bits per frame: sync word, parameters,
                                                     BCH(67, 53), differential BPSK
=====================================  ============  ===============================================

The cells of the symbols are arrays of shape ``(K, n_symbols)``. The signal is at one
sample per elementary period :math:`T = 7/(8B)` (7/64 µs at 8 MHz):

.. code-block:: python

   import numpy as np
   import comnum

   link = comnum.Dvbt("8K", "1/4", "64-QAM", "2/3")
   print(link.packets_per_superframe, link.useful_bit_rate(8e6))   # 4032, 19.91e6

   packets = comnum.dvb_packets(1000, comnum.Rng(1))   # 69 symbols: one frame and more
   x = link.modulate(packets)                    # the OFDM signal, guard intervals included
   cells = link.demodulate(x)                    # (6817, n_symbols)
   print(link.decode_tps(cells[:, :68]))         # the parameters signalled by the TPS

   # the portable channel P1 of Annex B at a C/N of 21 dB over the 6817 carriers
   h = comnum.dvbt_channel_gains("P1", link.carrier_frequencies())
   noise_var = link.carrier_noise_var(21.0)
   rng = np.random.default_rng(1)
   w = np.sqrt(noise_var / 2) * (rng.normal(size=cells.shape) + 1j * rng.normal(size=cells.shape))
   out, corrected, failed = link.decode(cells * h[:, None] + w, 1000, noise_var, channel=h)

The C/N of Annex A is "calculated over the transmitted 6 817 OFDM carriers": the mean
power of the carriers, the boosted pilots included, over the noise of a carrier
(:meth:`~comnum.Dvbt.carrier_noise_var`). A data cell then sees an :math:`E_s/N_0` 0.33 dB
below the C/N. The static channels F1 and P1 act carrier by carrier, since every echo of
Table B.1 is shorter than the guard interval.

**Validation.** Table C.1 of Annex C, the interleaving example (the input bit numbers of
the six bits of every carrier of the first symbol, 2K, 64-QAM), is reproduced bit for bit;
so are the positions of Tables 7 and 8, the constellations of Figure 9, the TPS code and
the packet counts and bit rates of Tables 16 and 17. Table A.1 gives the C/N of a BER of
:math:`2 \cdot 10^{-4}` after the Viterbi decoder; the card ``dvbt_tablea1`` measures the
45 values in the 8K mode:

======================  =================================  ====================
Channel                 Gap to Table A.1 (15 values)       Tolerance of the card
======================  =================================  ====================
Gaussian                within 0.12 dB                     0.2 dB
F1 (fixed, Ricean)      within 0.26 dB                     0.3 dB
P1 (portable)           0.1 to 0.7 dB below                0.8 dB
======================  =================================  ====================

These are gaps between thresholds: the curve of comnum crosses the BER of
:math:`2 \cdot 10^{-4}` at a C/N that differs from the printed one by that much, the same
BER on both sides. A gap of 0.12 dB is a factor 1.03 in signal-to-noise ratio. The table
is printed to 0.1 dB in italics ("approximate"), so the Gaussian column is reproduced to
its precision. The P1 column is the weak point: a static channel's threshold depends on
the realisation of :math:`|H(f)|` over the band, and shifting the frequency reference of
the echoes by the 32/7 MHz of the centre carrier named in Annex A moves the P1 threshold
of 16-QAM 3/4 by 0.6 to 1.5 dB. The standard does not fix that reference; the baseband
reference :math:`k'/T_U` used here is the closest to the table of the three tried.
Hierarchical transmission (Tables A.2 and A.3) is not implemented beyond its
constellations (:math:`\alpha` = 2 and 4). The example
:doc:`../auto_examples/systems/plot_04_dvbt` draws the frame and the waterfalls.


Space telemetry: CCSDS 131.0-B-5
--------------------------------

CCSDS 131.0-B-5 (TM Synchronization and Channel Coding, Blue Book, 2023) specifies the
channel codes of space telemetry and how their codewords are framed on the channel.
comnum implements every code of the standard and the framing constants:

=====================================  ============  ===============================================
Function                               Clause        What it does
=====================================  ============  ===============================================
:class:`comnum.CcsdsConvolutional`     3             :math:`K = 7`, :math:`G_1 = 171_8` then
                                                     :math:`G_2 = 133_8`, :math:`G_2` inverted at
                                                     rate 1/2, punctured by Table 3-1
:class:`comnum.CcsdsRs`                4             RS(255, 255 − 2E), E = 8 or 16, depths
                                                     I = 1–5 and 8, virtual fill, dual basis
:class:`comnum.CcsdsTurbo`             6             16 states, k = 1784 to 8920, rates 1/2 to 1/6
:class:`comnum.CcsdsLdpc`              7             (8160, 7136) code, nine AR4JA codes
                                                     (k = 1024, 4096, 16384; rates 1/2, 2/3, 4/5)
:func:`comnum.ccsds_marker`            9             the attached sync markers
:func:`comnum.ccsds_randomizer`        10            the pseudo-randomizers (periods 131 071, 255)
:func:`comnum.ccsds_cadu`              9–10          marker, then the randomized codeblock
=====================================  ============  ===============================================

The Reed–Solomon codeblock works on octets, as they are transmitted: the frame, then the
:math:`2EI` check octets, the codeword :math:`i` holding the octets :math:`i, i + I,
i + 2I, \dots`. The octets are in the dual basis of Annex F; the conventional code of
:class:`comnum.ReedSolomonCode` with ``poly=0x187, first_root=128 - E, root_step=11`` is
the same code in the conventional basis.

.. code-block:: python

   import numpy as np
   import comnum

   rs = comnum.CcsdsRs(16, 5)                     # E = 16, I = 5 (Sec. 4.3.8)
   print(rs.frame_len, rs.codeblock_len)          # 1115, 1275 octets
   frame = np.arange(rs.frame_len) % 256
   block = rs.encode_bytes(frame)
   block[:80] ^= 0xFF                             # 16 octets in error in every codeword
   decoded, corrected = rs.decode_bytes(block)    # corrected = 80

The LDPC and turbo codes take and return bits, as every code of the package; their
decoders take LLRs, positive for a 0:

.. code-block:: python

   code = comnum.CcsdsLdpc("1/2", 1024)           # AR4JA, (2048, 1024), Table 7-5
   b = comnum.bits(code.k, comnum.Rng(1))
   c = code.encode(b)                             # information first, M bits punctured
   assert code.ldpc.is_codeword(code.full_codeword(b))   # H of Sec. 7.4.2, (K + 3) M bits

   noise_var = comnum.bpsk_noise_variance(1.5, code.k / code.n)    # Eb/N0 = 1.5 dB
   y = 1.0 - 2.0 * c + np.sqrt(noise_var) * np.random.default_rng(1).normal(size=code.n)
   b_hat = code.decode(comnum.bpsk_llr(y, noise_var))

   turbo = comnum.CcsdsTurbo(8920, "1/6")         # n = (k + 4) 6 = 53 544 (Table 6-2)

On the channel, a codeblock follows its marker and is randomized (the marker is not); the
receiver removes the sequence by negating the LLRs where it is one:

.. code-block:: python

   asm = comnum.ccsds_marker("asm")               # 1ACFFC1D
   bits = comnum.ccsds_cadu(asm, c, "long")       # 32 marker bits, then c randomized
   llr = 4.0 * (1.0 - 2.0 * bits[32:])            # the channel LLRs of the codeword
   llr *= 1.0 - 2.0 * comnum.ccsds_randomizer("long", code.n)      # derandomized

**Validation.** The standard prints tables, not curves, and comnum reproduces every value
it prints exactly (card ``ccsds131_exact``, 52 values): the 256 rows of Table F-1 and the
two examples of Annex F, the generators of Annex G, the sizes of the codeblock of
Sec. 4.3.8, the output sequences of Table 3-1, the lengths of Tables 6-2 and 7-5, the parity
checks of Table 7-1 and Sec. 7.4.2 on random codewords of the ten LDPC encoders, the first
40 bits of both randomizers (Sec. 10.4.3, NOTE 2) and the markers of Figures 9-1 to 9-6.
The error rates of the decoders are not validated: their curves are in the Green Book
CCSDS 130.1-G. The example :doc:`../auto_examples/systems/plot_05_ccsds` draws them next to
the normal approximation, a check of self-consistency (the three codes reach a frame error
rate of :math:`10^{-2}` 0.6 to 0.8 dB to its right). The turbo permutation is checked by
its first values computed from Sec. 6.3 g), since the standard does not list it. The
slicing of a stream into LDPC codewords (Sec. 8) and the frame synchronization (left to the
implementer by Sec. 9.2) are not implemented.

Cellular, broadcast and maritime: LTE (3GPP TS 36.211, 36.212), FM (ITU-R BS.450, RDS), AIS (ITU-R M.1371)
-------------------------------------------------------------------------------------------------------------

The LTE downlink as a terminal reads it first, and FM sound broadcasting with its
radio data system, are the receivers of measured signals: :func:`comnum.lte_frame`
builds the frames of a cell (PSS, SSS, CRS, PBCH, transmit diversity),
:func:`comnum.lte_cell_search` and :func:`comnum.lte_decode_mib` read them back from a
capture at 1.92 MS/s; :func:`comnum.fm_multiplex` and :func:`comnum.fm_stereo_decode`
build and decode the stereophonic multiplex, :func:`comnum.rds_receive` the RDS of its
57 kHz subcarrier. The page :doc:`recordings` shows them on a HackRF capture, and the
tutorial :doc:`A measured signal <../auto_tutorials/plot_07_measured_signal>` on a
simulated one.

The automatic identification system of ships (ITU-R M.1371-5) is the third receiver
of measured signals: :func:`comnum.ais_transmit` builds the packets of Annex 2 (the
training sequence, the HDLC frame with its frame check sequence and bit stuffing, NRZI,
GMSK with ``BT = 0.4`` at 9600 bit/s) and :func:`comnum.ais_receive` reads them back
from a channel at baseband (161.975 or 162.025 MHz through :func:`comnum.select_channel`
at 153.6 kHz: the polar discriminator, an integrator over half a bit, the frequency
offset of the transmitter removed, a bit clock from the zero crossings, the frames whose
frame check sequence is correct). :func:`comnum.ais_decode` reads the fields of the
position reports (messages 1, 2, 3, 18), the base station report (4), the static and
voyage data (5) and the class B static data (24) of Annex 8, and
:func:`comnum.ais_aivdm` and :func:`comnum.ais_messages` write and read the ``!AIVDM``
sentences of NMEA 0183. The fields are checked against the recorded decoding of pyais
(``tests/oracle/test_ais_oracle.py``); over a 25 kHz channel of white noise the receiver
reads 9 packets of 10 at 10 dB of SNR, where the standard asks for 8 of 10 at its
sensitivity level. The Rust example ``ais`` runs the receiver on a SigMF recording of a
software radio (``hackrf_transfer -r ais.sigmf-data -f 162000000 -s 2000000``).
