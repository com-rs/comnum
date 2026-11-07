Getting started
===============

Installation
------------

The package is on PyPI, with wheels for Linux (glibc and musl, x86-64 and ARM),
macOS (Intel and Apple silicon) and Windows, for CPython 3.11 and later; NumPy is its
only dependency:

.. code-block:: bash

   pip install comnum            # or: pip install "comnum[plot]", with matplotlib

From a clone, ``comnum`` is built with `maturin <https://www.maturin.rs>`_ and a Rust
toolchain (1.85 or later):

.. code-block:: bash

   git clone https://github.com/com-rs/comnum
   cd comnum
   python -m venv .venv && source .venv/bin/activate
   pip install maturin numpy
   maturin develop --release

The development dependencies are dependency groups of ``pyproject.toml`` (pip 25.1 or
later): ``pip install --group test`` for the tests (they read the recorded outputs of
the oracles), ``pip install --group lint`` for ruff, ``pip install --group docs`` for
this documentation. The package is typed (``py.typed`` and a stub of the extension).

Conventions
-----------

Arrays
   Signals are 1-D ``complex128`` arrays (array-likes are converted), bits are
   ``uint8`` arrays of 0 and 1, symbol indices ``int64`` arrays. Several modes
   (polarisations) are 2-D arrays ``(modes, samples)``.

Randomness
   There is no global generator: every function that draws takes a :class:`comnum.Rng`
   and advances it, so a simulation is reproducible from its seed. The generator is the
   one of the Rust crates — the same seed gives the same draws in Rust and in Python —
   and :meth:`comnum.Rng.stream` derives the independent generator of every trial of a
   Monte Carlo campaign.

Units and names
   Physical quantities are in SI units unless the name says otherwise: ``ebn0_db``,
   ``esn0_db``, ``snr_db``, ``power_dbm``, ``length_km``, ``alpha_db_km``,
   ``d_ps_nm_km``… :math:`E_b/N_0` and :math:`E_s/N_0` are always distinguished by
   name. ``fs`` is a sampling rate, ``fc_hz`` a carrier frequency, ``noise_var`` a noise
   variance :math:`E|w|^2`; MIMO matrices are ``(n_rx, n_tx)``. The tuning parameters
   are keyword-only (``comnum.pulse_shape("rrc", 4, rolloff=0.25)``).

Closed forms
   The ``theory_*`` functions and the conversions take a scalar (and return a float) or
   an array-like (and return an array of the same shape):
   ``comnum.theory_ber(16, np.arange(0, 12))``.

Errors
   Invalid parameters raise :class:`ValueError`, numerical failures
   :class:`ArithmeticError`.

Conventions of the blocks
   Gray labels, pulse grids, metrics and fibre models follow
   `OptiCommPy <https://github.com/edsonportosilva/OptiCommPy>`_, which serves as the
   test oracle: a user of one finds the same blocks under the same names in the other.

A first link
------------

16-QAM over AWGN, decided symbol by symbol, against the closed-form bit error rate:

.. code-block:: python

   import comnum

   rng = comnum.Rng(2024)
   c = comnum.Constellation("qam", 16).normalized()   # Gray labels, unit energy

   bits = comnum.bits(4 * 200_000, rng)                # 4 bits per symbol
   x = c.map_bits(bits)
   y = comnum.awgn_ebn0(x, 10.0, 4, rng)               # Eb/N0 = 10 dB, 4 bits/symbol

   print(comnum.ber(c.demap_bits(y), bits))            # 1.80e-3
   print(comnum.theory_ber(16, 10.0, "qam"))           # 1.75e-3

.. _getting-started-signals:

Seeing the signals along the chain
----------------------------------

The specifications of ``specs/tutorial/`` run a single trial and place ``scope`` blocks
along the link; in the `playground <https://com-rs.github.io/comnum/playground/>`_
(the group ``specs/tutorial`` of the explorer), **Run** opens one tab per scope, and
the selector **traced** switches between the conditions of the grid (a receiver, a
roll-off, a number of antennas…), which all see the same channel when the channel
block has a ``seed``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Specification
     - What it shows
   * - ``01_awgn.yaml``
     - 16-QAM over AWGN: the points, the spectrum and the eye of the root-raised-cosine
       signal, before and after the channel and the matched filter
   * - ``02_ofdm.yaml``
     - 16-QAM OFDM over the ETU channel of 3GPP TS 36.101: the notches of the received
       spectrum, and the subcarriers without an equaliser (``estimation: none``, every
       one rotated and scaled by its own :math:`H_{k}`), with zero forcing on the known
       channel, and on the channel estimated by LMMSE from the pilots
   * - ``03_mimo.yaml``
     - 4 × 4 spatial multiplexing of 16-QAM over one Rayleigh matrix: the received
       antennas, each a sum of the four streams, and the points after the detectors
       ``mr``, ``zf``, ``mmse``, ``zf_sic`` and ``sphere`` (maximum likelihood)
   * - ``04_nyquist.yaml``
     - the raised cosine for roll-offs from 0.05 to 1: the band it occupies, the eye
       open at its centre without intersymbol interference, narrower for a small roll-off
   * - ``05_constellations.yaml``
     - BPSK, QPSK, 8-PSK, 16-QAM and 64-QAM at the same :math:`E_{b}/N_{0}` of 12 dB, and
       their symbol error rates against the closed forms
   * - ``06_carrier_sync.yaml``
     - a carrier frequency offset (the constellation turns into a ring) and phase noise
       (arcs), corrected by the fourth-power estimator and the blind phase search
   * - ``07_timing_sync.yaml``
     - a sampling clock 100 ppm slow: the drifting eye and points, with and without the
       Gardner loop
   * - ``08_equalisation.yaml``
     - a multipath channel of three echoes: the closed eye of intersymbol interference,
       and the points after the adaptive equaliser (CMA, then decision-directed LMS)
   * - ``09_diversity.yaml``
     - QPSK over Rayleigh fading on one, two and four antennas with maximal-ratio
       combining
   * - ``10_iq_imbalance.yaml``
     - the gain and phase imbalance of the I and Q branches of the transmitter: the
       stretched and sheared constellation
   * - ``11_adc.yaml``
     - an ADC of 1 to 6 bits: the staircase of the time signal and the points after the
       matched filter
   * - ``12_cpm.yaml``
     - MSK and GMSK (BT = 0.5 and 0.3): the constant envelope, the narrower spectrum of the
       Gaussian pulse, and the maximum-likelihood detector over the phase trellis

From the command line, ``comnum run specs/tutorial/02_ofdm.yaml`` gives the symbol
error rate and the EVM of the trial for every receiver. The classical error-rate curves
are campaigns of ``specs/`` (:doc:`user_guide/specifications`): ``qam_awgn`` (QAM over
AWGN against the closed forms), ``mimo_diversity`` (the order of diversity of Alamouti
and zero forcing), ``ofdm_eva`` (OFDM with estimated channels) and ``bicm_ldpc`` (coded
against uncoded).

The :doc:`gallery <auto_examples/index>` turns such snippets into the classical curves,
and the :doc:`user guide <user_guide/index>` shows how to run them as Monte Carlo
campaigns with stopping rules and confidence intervals.
