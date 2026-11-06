Comparisons with other libraries
================================

comnum gathers blocks that usually live in separate tools: channel codes, wireless
waveforms and channels, fibre propagation, optical devices, receiver DSP, recordings.
Each domain has its reference implementations, and comnum is compared with them on the
same inputs. The outputs of the tools are recorded once (``tests/oracle/``,
``validation/aff3ct/``), with the version that produced them, so that the tests run
without the tools; none of their code is copied (see ``THIRD_PARTY.md``). The
:doc:`maturity <maturity>` page lists, module by module, the tests named below.

Agreeing with a tool is evidence, not proof: two implementations of a standard can
share a misreading. Where a tool and comnum differ, the difference is traced to the
standard or the paper, and documented.

.. list-table::
   :header-rows: 1
   :widths: 18 34 22 26

   * - Library
     - What is compared
     - Agreement
     - Differences found
   * - `OptiCommPy <https://github.com/edsonportosilva/OptiCommPy>`_ 0.11
       (GPL-3.0)
     - Gray mapping and demapping, closed-form error rates, pulse shaping and
       filters, resampling, linear fibre, split-step Fourier and Manakov propagation,
       WDM grid and SNR, Mach–Zehnder and IQ modulators, photodiodes, optical hybrid
       and coherent receiver, adaptive equalisers (CMA, RDE, NLMS, DD-LMS, 2×2),
       carrier recovery (fourth power, Viterbi–Viterbi, blind phase search, DD-PLL),
       Gardner clock recovery, BER, SNR, EVM, MI and GMI
       (``tests/oracle/test_oracle.py``)
     - to machine precision for the deterministic blocks (10⁻⁸ relative on the
       propagated fields); statistically for the Monte Carlo estimators
     - the BER of BPSK from ``theoryBER`` is twice the exact value; its ``edc``
       limits the SNR of a noiseless link to 23 dB; its ``edfa`` re-seeds its
       generator on every call; the trapezoidal nonlinear phase of ``manakovSSF``
       smooths the cross-phase modulation of channels that walk off within a step (45 %
       of the XPM of Chiang et al. 1996 with steps of 0.5 km) and its 0.02 rad per step
       leaves 0.45 dB of error on WDM links, so the Manakov solver of comnum uses other
       defaults (``docs/adr/0013-manakov-defaults.md``; all the fibre comparisons:
       :doc:`user_guide/fibre_validation`)
   * - `Optilux <https://optilux.sourceforge.io/>`_ Apr 2021 (GPL-3.0, under
       Octave)
     - split-step Fourier propagation of 16-QAM at 32 GBd over 10 × 80 km with 1 km
       steps, and one span with and without the Kerr effect, on the field of the
       benchmark scenario (``tests/oracle/test_optilux_oracle.py``, the card
       ``optilux_ssfm``); the amplified link of its example 21, PDM QPSK through 50
       spans of 150 km with amplifiers of 30 dB gain and 6 dB noise figure (the card
       ``optilux_ase_link``)
     - to machine precision without the Kerr effect (NMSE 3·10⁻²⁸); 1.2·10⁻¹⁰ after
       one span and 1.5·10⁻⁸ after ten with it; the SNR after 10 to 50 amplified
       spans within 0.15 dB of Optilux's estimates and 0.03 dB of the example's
       formula, the BER within 3 % of the formula
     - the nonlinear step: Optilux applies the phase over the effective length of the
       step, comnum over its length; at 1 km Optilux's field is the closer to the
       step-converged solution (1.1·10⁻⁸ against 3.7·10⁻⁸). Optilux writes the equation
       with the opposite sign of the imaginary unit, which the recording undoes. Its
       example 21 re-draws the noise of the last span only, so its BER estimates carry
       the bias of one realisation of the previous spans (1.45·10⁻³ after 20 spans
       against 8.7·10⁻⁴ of its own formula); its SNR estimates do not
   * - `GNPy <https://github.com/Telecominfraproject/oopt-gnpy>`_ 3.0
       (BSD-3-Clause)
     - Raman power profiles with co- and counter-propagating pumps, their spontaneous
       noise, the generalised GN model with power profiles
       (``tests/oracle/test_gnpy_oracle.py``)
     - 0.02 dB on the NLI; 0.004 dB on the profiles at a 5 m step
     - GNPy integrates by Euler's method (first order in its step); at its default
       tolerances it underestimates the NLI of counter-pumped spans by about 1 dB
   * - `AFF3CT <https://github.com/aff3ct/aff3ct>`_ v4.7.1 (MIT)
     - bit and frame error rates of convolutional (Viterbi), LTE turbo (log-MAP and
       max-log-MAP), LDPC (sum-product, min-sum variants, flooding and layered),
       polar (SC, CA-SCL), Reed–Solomon and BCH codes, the short LDPC code of DVB-S2
       (``validation/tests/coding/aff3ct.rs``)
     - within the confidence intervals of both simulations
     - none on the decoders; the polar construction is that of comnum on both sides
   * - `Sionna <https://github.com/NVlabs/sionna>`_ (Apache-2.0), its test vectors
     - the CRCs of 3GPP (CRC6 to CRC24C); convolutional codes of rates 1/2, 1/3, 1/4,
       encoded and Viterbi-decoded; the LTE turbo code at K = 40, 112, 168, 432,
       encoded and decoded at 0 dB; the 5G LDPC code with rate matching (28 sizes, 64
       to 8448 bits, both base graphs) and the 5G polar code of the uplink control
       information (``tests/oracle/test_sionna_vectors.py``)
     - bit for bit, the decisions included (errors and all)
     - none; the vectors brought the full QPP table of LTE (188 block sizes) to
       ``TurboCode.lte``; its tables of the LDPC base graphs and of the polar
       sequence are identical to those extracted from TS 38.212
   * - `py3gpp <https://github.com/catkira/py3gpp>`_ 0.6 (AGPL-3.0)
     - 5G NR: PSS, SSS, PBCH DMRS and their positions, the PDSCH DMRS of 42
       configurations, the OFDM modulation of a subframe
       (``tests/oracle/test_nr_oracle.py``)
     - bit for bit for sequences and positions; 10⁻¹⁴ for the waveforms
     - py3gpp lengthens the cyclic prefix of every slot instead of every half
       subframe (its subframes no longer last 1 ms from 60 kHz); its DMRS table adds
       a symbol to 8- and 9-symbol allocations without additional position
   * - `sigmf-python <https://github.com/sigmf/sigmf-python>`_ 1.x (LGPL-3.0)
     - SigMF recordings both ways: the 28 sample formats, several channels, captures,
       annotations, archives (``tests/test_sigmf.py``)
     - identical samples and metadata; the recordings of comnum pass its validation
     - sigmf-python scales 32-bit integers in float32 (an error of 3·10⁻⁸)

Other open implementations of the standards can serve the same way, through their
outputs: srsRAN and OpenAirInterface (LTE, 5G NR), the ``gr-dtv`` module of GNU Radio
(DVB-T, DVB-T2) and ``gr-ieee802-11`` (Wi-Fi). Their licences (GPL, AGPL and others)
forbid copying their code into comnum, which is MIT/Apache-2.0, but not comparing with
their results.

Decoding throughput against AFF3CT
----------------------------------

The decoders of comnum timed against those of AFF3CT v4.7.1 (commit d4d715e, the
version of the recorded error rates, built with clang 21 and ``-march=native``, NEON)
on the seventeen configurations of ``validation/aff3ct/``: one thread, the same code,
the same :math:`E_b/N_0` (the middle of the recorded sweep), a budget of 8 s per point,
the decoder alone (AFF3CT: the task ``decode_siho`` of ``--sim-stats``; comnum: the
``decode_block`` of the code on frames of LLRs drawn beforehand), in Mb/s of the bits
AFF3CT counts (information bits, or coded bits for the LDPC codes it runs with
``--sim-coded``). AFF3CT runs its ``STD``/``NAIVE`` implementations, as in the
recordings: its inter-frame SIMD decoders (``FAST``, ``AVX``, ``NEON``) are faster still
on the polar and LDPC codes. One run per point, on an Apple M6;
``tools/speed_aff3ct.py`` and ``validation/src/bin/speed_fec.rs`` reproduce it, the
raw results are in ``benchmarks/aff3ct/``.

.. list-table::
   :header-rows: 1
   :widths: 40 10 12 12 10 16

   * - Code and decoder
     - :math:`E_b/N_0` (dB)
     - AFF3CT (Mb/s)
     - comnum (Mb/s)
     - comnum / AFF3CT
     - comnum, whole chain (Mb/s)
   * - BCH (63, 45), t = 3, hard decisions
     - 5.0
     - 70.23
     - 142.71
     - 2.03
     - 31.95
   * - BCH (127, 92), t = 5
     - 5.0
     - 51.49
     - 107.91
     - 2.10
     - 33.01
   * - BCH (255, 223), t = 4
     - 5.5
     - 78.69
     - 178.68
     - 2.27
     - 43.12
   * - RS (15, 11), t = 2
     - 5.5
     - 349.52
     - 119.32
     - 0.34
     - 35.23
   * - RS (63, 55), t = 4
     - 5.75
     - 417.44
     - 238.38
     - 0.57
     - 57.08
   * - RS (255, 239), t = 8
     - 6.25
     - 441.83
     - 255.39
     - 0.58
     - 62.28
   * - convolutional K = 7, 64 bits, soft Viterbi
     - 2.5
     - 3.77
     - 3.93
     - 1.04
     - 3.18
   * - LDPC PEG (576, 288), flooding SPA, 50 it.
     - 2.0
     - 2.52
     - 2.79
     - 1.11
     - 2.79
   * - LDPC PEG (576, 288), flooding OMS 0.15, 20 it.
     - 2.25
     - 19.79
     - 7.51
     - 0.38
     - 6.39
   * - LDPC PEG (576, 288), layered NMS 0.75, 20 it.
     - 2.0
     - 23.58
     - 10.69
     - 0.45
     - 8.53
   * - LDPC PEG (648, 540), layered SPA, 20 it.
     - 3.75
     - 13.18
     - 14.28
     - 1.08
     - 10.84
   * - LDPC DVB-S2 (16200, 7200), layered SPA, 50 it.
     - 0.7
     - 0.92
     - 1.13
     - 1.23
     - 1.14
   * - polar (256, 128), SC
     - 3.0
     - 38.40
     - 12.80
     - 0.33
     - 9.69
   * - polar (1024, 512), SC
     - 2.5
     - 34.38
     - 13.23
     - 0.38
     - 9.93
   * - polar (256, 128) + CRC-11, SCL L = 8
     - 2.0
     - 1.72
     - 2.14
     - 1.24
     - 2.00
   * - turbo LTE K = 40, log-MAP, 8 it.
     - 2.5
     - 0.27
     - 0.21
     - 0.78
     - 0.21
   * - turbo LTE K = 1024, max-log-MAP, 6 it.
     - 1.0
     - 9.65
     - 1.93
     - 0.20
     - 1.81

comnum is level with AFF3CT on the sum-product, list and Viterbi decoders and twice as
fast on the BCH codes; it is 2 to 3 times slower where AFF3CT has specialised paths
(the min-sum LDPC rules, which here share the code of the sum-product; the successive
cancellation of polar codes, which AFF3CT prunes on the rate-0, rate-1 and repetition
nodes; the Reed–Solomon arithmetic) and 5 times slower on the max-log-MAP turbo decoder
of 1024 bits (a scalar BCJR in double precision). The whole chain of comnum (bits,
encoding, BPSK, noise, LLRs, decoding, error count) is also given: for the short codes
the channel dominates it.

Deviations from OptiCommPy
--------------------------

The blocks follow the layout, names and conventions of OptiCommPy, so that a user of
one finds the same blocks in the other, and the oracle tests compare them to machine
precision. The deviations are deliberate and tested as such:

- the declarative metrics compute MI and GMI with the least-squares channel gain;
  :func:`comnum.monte_carlo_mi` and :func:`comnum.monte_carlo_gmi` keep OptiCommPy's
  power normalisation (which underestimates the rates by 0.06 bit at 0 dB);
- :func:`comnum.evm` follows ``calcEVM``, which normalises before the rotation, unlike
  ``fastBERcalc``;
- :func:`comnum.theory_ber` of BPSK is the exact :math:`Q(\sqrt{2E_b/N_0})`;
  OptiCommPy's general PSK formula doubles it;
- :func:`comnum.monte_carlo_mi` snaps the normalised transmitted symbols to the
  constellation, so that the estimate never exceeds :math:`\log_2 M`, and both
  estimators return :math:`\log_2 M` without noise instead of 0/0;
- :func:`comnum.pulse_shape` uses a symmetric time grid, equal to OptiCommPy's for an
  even length; an odd length puts the taps on the sampling instants, while OptiCommPy's
  odd grid is shifted by about half a sample;
- the clock recovery computes in double precision; OptiCommPy stores its output in
  ``complex64``;
- with shaped symbols, :func:`comnum.theory_gmi` is :math:`H(X) - \sum_k H(B_k|Y)`,
  the rate of bit-metric decoding and the limit of the Monte Carlo estimators;
  OptiCommPy's ``theoryGMI`` sums the bit-wise informations :math:`\sum_k I(B_k;Y)`,
  which can exceed the entropy of the source (both agree for uniform symbols);
- with a seed, OptiCommPy's ``edfa`` re-seeds its generator on every call, so that
  ``manakovSSF`` adds the same ASE to both polarisations and after every span; the
  comparisons of noisy links leave its seed unset, and :class:`comnum.ManakovSsfm`
  draws independent noise;
- OptiCommPy's ``edc`` limits the SNR of a linear, noiseless link to 23 dB; the oracle
  tests compensate dispersion with its linear fibre of opposite dispersion.
