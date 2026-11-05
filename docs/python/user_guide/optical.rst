Optical links: WDM and nonlinearity
===================================

A WDM link is a chain of the blocks ``WdmTx`` (the channel under test and its
neighbours), ``Manakov`` (dual-polarisation nonlinear propagation over the spans, with
the EDFAs), ``WdmRx`` (the channel under test at baseband) and the receiver DSP. With
``Source(pols=2)`` the two polarisations carry independent symbols; the blocks written
for one polarisation process each of them, ``AdaptiveEqualizer`` is the 2 × 2
butterfly, and the metrics pool both.

.. code-block:: python

   import comnum
   from comnum import blocks as B

   fiber = {"n_spans": 10, "span_km": 80, "d_ps_nm_km": 16, "gamma_w_km": 1.3}
   link = comnum.Chain(
       tx=B.Source(constellation="16-QAM", n_symbols=8192, symbol_rate=32e9, pols=2),
       wdm=B.WdmTx(n_channels=5, spacing_hz=50e9, sps=8,
                   power_dbm=comnum.var("power_dbm"), rolloff=0.1),
       fiber=B.Manakov(),
       rx=B.WdmRx(sps=2),
       comp=B.Cdc(),
       mf=B.MatchedFilter(rolloff=0.1),
       dec=B.Decimate(),
       cpr=B.Bps(n=32, b=32),
       sync=B.FrameSync(start=2048, length=512),
   )
   results = link.run(grid={"power_dbm": range(-6, 7, 2)},
                      metrics={"report": ["snr_db", "ber", "gmi"], "skip_start": 512,
                               "skip_end": 512},
                      stop={"max_trials": 8}, fiber=fiber)

   # dual-polarisation digital back-propagation instead of dispersion compensation
   dbp = link.replace(comp=B.ManakovDbp(steps_per_span=4))

The propagation
---------------

``Manakov`` solves the Manakov equation by the symmetric split-step method:
:class:`comnum.ManakovSsfm` runs it on arrays. Two numerical choices fix its accuracy,
and its defaults differ from OptiCommPy's ``manakovSSF`` on both
(``docs/adr/0013-manakov-defaults.md``):

- **the nonlinear step** (``nonlinear_step``): ``"midpoint"`` (the default) takes the
  phase of the power at the middle of the step, as the scalar solver ``Ssfm``;
  ``"trapezoidal"`` is OptiCommPy's mean of the power at the two ends of the step,
  iterated. When channels walk off within a step the mean smooths the intensity that
  drives the cross-phase modulation: with steps of 0.5 km, a probe 125 GHz from a
  modulated pump receives 45 % of its XPM, and a five-channel link 0.9 dB too little
  interference;
- **the step rule** (``step_rule``): ``"fwm"`` (the default) is the rule of Musetti,
  Serena and Bononi (2018): the first step of every span bounds the phase mismatch of
  the four-wave mixing across the signal bandwidth to ``fwm_phase`` (20 rad, about
  0.01 dB of error on the SNR whatever the power), the next ones grow with the loss, and
  no step exceeds ``max_nl_phase`` (0.02 rad) of nonlinear phase, which a single channel
  at high power needs. The bandwidth is measured on the launched field, or given
  (``bandwidth_hz``). ``"nonlinear_phase"`` is OptiCommPy's rule, ``max_nl_phase`` alone:
  it depends on the power and ignores the bandwidth. ``"logarithmic"`` takes
  ``steps_per_span`` steps per span distributed as Bosco et al. (*IEEE Photon. Technol.
  Lett.* 12(5), 2000, eq. (6)): every step carries the same integrated power, so the
  steps are short at the start of the span and long at its end, and the spurious
  four-wave mixing of the split step stays near :math:`1/K` of the true one instead of
  the peaks of equal steps (their Fig. 3, checked in closed form). ``step_km`` fixes the
  step. ``Ssfm(fiber, steps_per_span=K)`` uses the logarithmic steps too (double
  precision).

**Steps of the back-propagation.** ``Ssfm.backpropagate(..., step_distribution=
"logarithmic")`` and the ``dbp`` block (``step_distribution: logarithmic``) distribute
the steps of a span logarithmically and apply them from its end. On one span of 80 km
at 6 dBm the residual of the back-propagation falls from 0.090 to 0.070 with two steps
and from 0.045 to 0.032 with four, at the full nonlinear phase (:math:`\xi = 1`, where
the equal steps need :math:`\xi \approx 0.9`); with sixteen steps the equal steps are
better (0.0015 against 0.0058): the last logarithmic steps are long, and Sinkin et al.
(*J. Lightw. Technol.* 21(1), 2003) find the method efficient only in some regimes.
``step_distribution="modified_logarithmic"`` (``step_factor`` 0.4) is the heuristic of
Zhang, Li & Dong (2013) that Häger & Pfister use (*IEEE J. Sel. Areas Commun.* 39(1),
2021): Bosco's steps for the attenuation times the factor, less concentrated at the
start of the span. On Fig. 8 of Häger & Pfister (32 GBd over 10 × 100 km) it reproduces
the DBP with 2 and 4 steps per span within 0.15 dB up to 4 dBm, where equal steps and
Bosco's steps fall 1.7 to 4.9 dB below (``specs/validation/hager2021_fig8.yaml``).
Above 4 dBm the DBP with 4 steps of comnum is up to 1.1 dB better than the figure: its
nonlinear step integrates the power over the step, where the code of the paper takes the
power at the middle of the step times the step. ``nonlinear_length="step"`` selects the
paper's convention, which reproduces the figure within 0.33 dB up to 8 dBm; the default,
``"effective"``, is the more accurate one.

**The sampling rate** must hold the four-wave mixing that falls on the channel under
test: at least :math:`\frac{3}{2} B_{\mathrm{WDM}} + \frac{1}{2} B_{\mathrm{CUT}}`, the
bandwidth of the comb and of the channel (Musetti et al. 2018, Table I; :math:`2
B_{\mathrm{WDM}}` for every channel, :math:`3 B_{\mathrm{WDM}}` for all the first-order
mixing). Five channels of 32 GBd on 50 GHz with roll-off 0.1 need 370 GHz: with 8
samples per symbol (256 GHz) the aliased mixing adds 0.3 dB of NLI to the centre
channel; the WDM specifications use 12. The block must also be longer than the walk-off
across the bandwidth over the whole link, :math:`|\beta_2| \, 2\pi B \, L` (Musetti et
al. 2018, Appendix B), or the circular convolution of the dispersion wraps the fast
channels onto the slow ones. Both conditions are checked on the launched field before
every propagation (``check_sampling``, ``True`` by default): a field that fails them is
refused with the rate or the block it needs, and ``check_sampling=False`` runs anyway
(the numerical tests on full-band signals).

**Third-order dispersion.** ``FiberParams(slope_ps_nm2_km=0.057)`` gives the dispersion
slope :math:`S` of the fibre, from which the solvers take
:math:`\beta_3 = (\lambda / 2\pi c)^2 (S \lambda^2 + 2 \lambda D)`, about 0.13 ps³/km
for standard fibre (Agrawal 2013, eq. (1.2.11)); the default is 0, no slope. The term
adds a cubic phase :math:`\beta_3 \omega^3 L / 6` to the linear step of every propagator,
and the back-propagation of :class:`comnum.ManakovSsfm` and ``Ssfm`` undoes it.

``Manakov(step_rule="nonlinear_phase", nonlinear_step="trapezoidal")`` is OptiCommPy's
solver: on the same field the two implementations agree to :math:`10^{-8}`
(``tests/oracle/test_oracle.py``).

The solvers are compared with exact soliton solutions, the theory of cross-phase
modulation, the step accuracy of Musetti et al., the GN and EGN models, a published
system (Häger & Pfister 2021) and other simulators: :doc:`fibre_validation` gathers
these comparisons, their results, what they changed in the defaults (with the accuracy
and time of the defaults against OptiCommPy's scheme) and what is not validated.

The GN model
------------

:func:`comnum.gn_model` computes the nonlinear interference of a WDM signal by the
Gaussian-noise model of Poggiolini (the double integral with the coherent accumulation
of the spans), the ASE of the amplifiers, the SNR of a channel and the optimal launch
power; :func:`comnum.gn_closed_form` is Poggiolini's closed form and
:func:`comnum.gn_isrs_closed_form` the closed form of the ISRS GN model of Semrau, Killey
& Bayvel (2019): the SPM and the XPM of every interferer with the first-order tilt of
inter-channel stimulated Raman scattering, in microseconds (within 0.3 dB of the
dashed curves of their Figs. 4a, 4b and 5 away from the band edges). The model assumes a
Gaussian signal: ``specs/wdm_gn.yaml`` checks that the Manakov simulation of Gaussian
symbols follows it within 0.4 dB from the ASE-limited regime to 4 dB above the optimum.
Specifications can use it as an expected result (``theory: {function: gn_snr_db, …}``).

QAM symbols generate less interference than Gaussian ones. With ``format`` (a
constellation, ``"16-QAM"``, with ``px`` for a shaped source), :func:`comnum.gn_model`
returns the EGN model of Carena et al. (2014): the GN model corrected for the fourth and
sixth moments of the symbols (:func:`comnum.format_moments`), for the self-channel
interference and the cross-channel interference of every channel. In
``specs/wdm_gn.yaml`` it follows the simulated 16-QAM within 0.3 dB, where the GN model
is up to 1.7 dB off. ``validation/tests/fiber/carena2014.rs`` checks it against Fig. 1(a) of the
paper and against a Monte Carlo evaluation of the first-order perturbation.
Specifications use it with ``theory: {function: gn_snr_db, format: 16-QAM, …}``.

Raman amplification
-------------------

A ``raman`` mapping in the fibre parameters pumps every span: any number of pumps,
co-propagating (launched with the signals) or counter-propagating (launched at the end
of the span), given by ``wavelength_nm`` or ``frequency_thz`` and ``power_mw``, with
their own attenuation ``alpha_db_km`` if needed. The channels also exchange power
among themselves (inter-channel stimulated Raman scattering, ``isrs: true`` by
default), which tilts a wide WDM spectrum towards low frequencies even without pumps.

.. code-block:: python

   fiber = {
       "n_spans": 3, "span_km": 100, "nf_db": 5,
       "raman": {"pumps": [
           {"wavelength_nm": 1425, "power_mw": 300, "direction": "counter"},
           {"wavelength_nm": 1455, "power_mw": 250, "direction": "counter"},
           {"frequency_thz": 205.0, "power_mw": 100, "direction": "co"},
       ]},
   }
   results = link.run(grid={"power_dbm": range(-10, 3, 2)},
                      metrics={"report": ["snr_db"]}, fiber=fiber)

   # the power profiles of one span (channels, then pumps) and the gains
   p = comnum.raman_profile(comnum.FiberParams(span_km=100), comnum.wdm_grid(40, 100e9),
                           0.0, fiber["raman"])
   p["on_off_gain_db"], p["net_gain_db"], p["ase_psd"]

:func:`comnum.raman_profile` solves the coupled power equations of stimulated Raman
scattering (Bromage 2004) along the span: RK4 integration, forward for the waves that
travel with the signals and backward for the counter-propagating pumps, repeated until
the profiles converge. The coefficients are those of the measured gain spectrum of
standard single-mode fibre (``gain`` replaces it), scaled with the pump frequency, and
conserve the number of photons. The spontaneous Raman scattering of the pumps adds a
noise whose thermal factor grows near the pumps.

``Ssfm`` and ``Manakov`` apply the profile of every span to the field as a gain that
depends on the position and on the frequency. The profile is solved for slices of the
spectrum of the field (``slice_hz``, 25 GHz) at the input of the span. After the span
they add the spontaneous noise; the amplifier (``amplification``) then restores every
frequency to its launch power, as an EDFA with an ideal gain-flattening filter that adds
its ASE only where the Raman gain did not compensate the loss. :func:`comnum.gn_model`
takes the same ``raman`` mapping. It integrates the power profiles of the channels in
the link function of the GN model, the generalised GN model of Semrau, Killey & Bayvel
(2019), and counts the noise of the hybrid amplifiers. ``specs/raman_wdm.yaml`` checks
the Manakov simulation against it, without and with a counter-pump, within 0.21 dB.
``tests/oracle/test_gnpy_oracle.py`` checks the profiles, the noise and the NLI against GNPy.
The digital back-propagation keeps the profile of lumped amplifiers.

Polarisation demultiplexing
---------------------------

``Pmd`` rotates the polarisations at random (a uniform SU(2) rotation) and adds
polarisation-mode dispersion of a mean differential group delay ``mean_dgd_ps`` (the
coarse-step model of Marcuse, Menyuk and Wai), a new realisation every trial;
``theta_deg`` gives a fixed real rotation instead. ``AdaptiveEqualizer`` with two
polarisations is the 2 × 2 butterfly of OptiCommPy's ``mimoAdaptEqualizer`` (equal to it
to :math:`10^{-6}` on the same input), with one step size per stage of the schedule.
Blind criteria such as the CMA admit a singular solution, both outputs on the same
polarisation; ``kikuchi=n`` is Kikuchi's two-stage initialisation (output x alone for
``n`` symbols, then output y as its unitary complement), which removes it. The block
records ``<label>.correlation`` (of the two outputs) and ``<label>.singular`` (above
0.5); ``FrameSync`` restores the order of the outputs from the known symbols.

.. code-block:: python

   link = comnum.Chain(
       tx=B.Source(constellation="16-QAM", n_symbols=16384, symbol_rate=32e9, pols=2),
       shape=B.PulseShape(sps=2, rolloff=0.2),
       pmd=B.Pmd(mean_dgd_ps=20.0),
       noise=B.Awgn(snr_db=20.0),
       mf=B.MatchedFilter(rolloff=0.2),
       eq=B.AdaptiveEqualizer(algorithm=[["cma", 6000], ["rde", 10**9]],
                              mu=[2e-3, 1e-4], kikuchi=3000),
       sync=B.FrameSync(start=12384, length=512, symmetry=0),
   )

``specs/polmux_cma.yaml`` and ``specs/polmux_cma_45deg.yaml`` check it: from the
identity, half of the trials of a 45° rotation end singular; with Kikuchi's
initialisation none does, and the SNR after the equaliser is within 1 dB of the
channel's under rotation and DGD up to 20 ps.
