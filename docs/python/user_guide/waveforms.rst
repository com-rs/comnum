Waveforms beyond OFDM
=====================

CP-OFDM turns a frequency-selective channel into flat subcarriers at the price of a
prefix, of sidelobes that fall slowly, and of interference between subcarriers when
the channel changes within a symbol. ``comnum`` implements the waveforms proposed to
relax each of these limits, with the same conventions as :class:`~comnum.Ofdm`:
``modulate`` takes the symbols of whole blocks and returns the samples, ``demodulate``
inverts it, and the round trip is exact without a channel. The gallery page
:doc:`../auto_examples/waveforms/plot_08_filtered_multicarrier_otfs` compares their
spectra.

FBMC/OQAM
---------

:class:`~comnum.Fbmc` shapes every subcarrier with the PHYDYAS prototype of overlapping
factor 2, 3 or 4 and drops the prefix. Orthogonality holds in the real sense only, so
each subcarrier carries a real symbol every half symbol: :func:`~comnum.oqam_stagger`
splits complex symbols into them and :func:`~comnum.oqam_destagger` takes the real part
back.

.. code-block:: python

   import comnum, numpy as np

   fbmc = comnum.Fbmc(64, overlap=4)
   c = comnum.Constellation("qam", 16).normalized().map_bits(comnum.bits(4 * 64 * 20, comnum.Rng(1)))
   x = fbmc.modulate(comnum.oqam_stagger(c, 64))
   c_hat = comnum.oqam_destagger(fbmc.demodulate(x).real, 64)

The prototype spans :math:`K` symbols, so a burst is :math:`K - 1/2` symbols longer
than its data, and PHYDYAS is orthogonal only nearly: the round trip leaves a residual of
about :math:`10^{-3}`, far below any noise of interest. In a doubly-selective channel
the residual imaginary interference adds to the noise; ``validation/tests/multicarrier/nissel2017.rs``
compares FBMC with OFDM and CP-OFDM in the Vehicular A channel of Nissel & Rupp (2017).

UFMC
----

:class:`~comnum.Ufmc` groups the active subcarriers in sub-bands and filters each one
with a Dolph–Chebyshev filter of length :math:`L` (:func:`~comnum.chebyshev_window`), as
Schaich & Wild (2014) propose: the filter's ramps replace the prefix, a symbol is
:math:`N + L - 1` samples, and the receiver takes a DFT of :math:`2N` points and keeps
the even bins. The filter attenuates the subcarriers at the edges of each sub-band, which
the one-tap equaliser undoes.

OTFS
----

:class:`~comnum.Otfs` places :math:`MN` symbols on a grid of :math:`M` delays and
:math:`N` Doppler shifts, maps them to time and frequency by the inverse symplectic
Fourier transform and transmits them as OFDM symbols with one prefix per frame. A
channel of paths of whole delays and Doppler bins is a 2-D circular convolution on that
grid: :meth:`~comnum.Otfs.effective_channel` builds it (Raviteja et al. 2019, Theorem 1),
and :func:`~comnum.delay_doppler_paths` draws paths from a power-delay profile with
Jakes' Doppler.

.. code-block:: python

   otfs = comnum.Otfs(32, 16, n_cp=8)
   fs = 32 * 15e3
   paths = comnum.delay_doppler_paths("eva", 500.0, fs, comnum.Rng(2))
   # paths: (gain, delay in samples, Doppler in Hz); with whole bins
   # (Doppler multiple of fs / (M N)), H @ x is the demodulated frame
   H = otfs.effective_channel([(1.0, 0, 0.0), (0.5, 3, fs / 512)], fs)

Every path then reaches every symbol: a detector on the whole frame (linear MMSE on
``H``, or message passing) collects the full diversity of the channel, where one-tap
OFDM loses it (``validation/tests/multicarrier/raviteja2019.rs``).

Chirp spread spectrum
---------------------

:class:`~comnum.Css` is the modulation of LoRa: :math:`2^{SF}` cyclic shifts of an
up-chirp, detected by dechirping and an FFT. :func:`~comnum.theory_css_ser` and
:func:`~comnum.theory_css_ber` give the error rates over AWGN, the ``css_*`` functions the
spectrum and the interference between two signals; the gallery page
:doc:`../auto_examples/waveforms/plot_07_lora` compares them with three papers.

Direct-sequence spread spectrum and CDMA
----------------------------------------

:class:`~comnum.Dsss` spreads the symbols of one or several synchronous users on
signature codes of :math:`N` chips: :func:`~comnum.m_sequence`,
:func:`~comnum.gold_codes`, :func:`~comnum.kasami_codes`, :func:`~comnum.walsh_codes` and
:func:`~comnum.barker_code`, whose correlations :func:`~comnum.periodic_correlation` and
:func:`~comnum.aperiodic_correlation` give. One symbol period is :math:`y = SAb + w`,
the model of a MIMO link with the signatures :math:`H = SA` (``Dsss.signatures``) in
place of the channel, so the multiuser receivers are the detectors of
:func:`~comnum.mimo_detect`:

.. code-block:: python

   cdma = comnum.Dsss(comnum.gold_codes(5)[1:9], amplitudes=[1, 1, 2, 2, 1, 1, 3, 1])
   chips = cdma.modulate(symbols)                    # (8, n) symbols -> 31 n chips
   chips += rng.complex_noise(chips.size, noise_var)
   soft = cdma.demodulate(chips)                     # the bank of matched filters
   labels = cdma.detect(chips, bpsk, noise_var, "mmse")   # or "mr", "zf", "mmse_sic", "ml"
   sinr = cdma.sinr(noise_var, "mmse")               # of every user, for these codes

``"mr"`` is the conventional receiver, limited by the other users (the near–far
problem), ``"zf"`` the decorrelator of Lupas and Verdú (*IEEE Trans. Inf. Theory* 35(1),
1989), whose error rate does not depend on their powers. :func:`~comnum.theory_ber_cdma`
gives the exact error rate of both for the signatures at hand and
:func:`~comnum.cdma_near_far_resistance` the loss of the decorrelator. For random
signatures, :func:`~comnum.cdma_sinr_large_system` and
:func:`~comnum.cdma_sinr_mmse_large_system` are the limits of the SINR at the load
:math:`K/N` (Tse and Hanly, *IEEE Trans. Inf. Theory* 45(2), 1999, eqs. (6) and (9)) and
:func:`~comnum.cdma_spectral_efficiency` the spectral efficiency of Verdú and Shamai
(same issue, eqs. (9)–(12)). Over a multipath channel, ``Dsss.rake`` combines one
finger per path. The users are synchronous and sampled at the chip rate, with one
period of the code per symbol. The gallery page
:doc:`../auto_examples/waveforms/plot_09_spread_spectrum` goes through the codes, a tone
jammer, the near–far problem, large systems and the rake receiver.

Frequency hopping
-----------------

:class:`~comnum.Fhss` is slow frequency hopping with noncoherent :math:`M`-FSK: a band
of :math:`Q` channels of :math:`M` orthogonal tones, :math:`QM` samples per symbol, the
channel changed along a pattern (``Fhss.random_pattern``) that the receiver follows
with a DFT. :func:`~comnum.partial_band_noise` is a Gaussian jammer on a part of the
same grid. A jammer of the density :math:`N_J` over the whole band that concentrates
its power on the fraction :math:`\rho` hits a hop with the probability :math:`\rho`,
:math:`1/\rho` times denser: :func:`~comnum.theory_ber_fh_partial_band` is the
resulting error rate and :func:`~comnum.fh_worst_case_partial_band` the fraction that
maximises it, :math:`e^{-1}N_J/E_b` for binary FSK. Only whole symbols per hop are
modelled, without coding.
