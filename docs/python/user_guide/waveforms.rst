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
