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
  ``"effective_length"`` is Optilux's scheme, the power at the start of the nonlinear
  block over the effective length of the step, which integrates the attenuation within
  the step exactly (the most accurate of the three on one channel, :doc:`fibre_validation`,
  section 3); ``"trapezoidal"`` is OptiCommPy's mean of the power at the two ends of
  the step, iterated. When channels walk off within a step the mean smooths the intensity that
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

Reproducing published curves
----------------------------

The defaults of the solvers are chosen for accuracy, not to match a given paper. A
published curve is reproduced when the simulation takes the paper's own choices: its
step distribution, its nonlinear step, its sampling, its number of runs. Each example
below gives the link of the paper, the settings that decide the curve, what changes
with other settings, the agreement obtained and the file that runs it. The numbers come
from the validation specifications (``specs/validation/``) and cards
(``validation/cards/``) named in each example; :doc:`fibre_validation` explains the
comparisons in full.

.. list-table::
   :header-rows: 1
   :widths: 22 30 26 22

   * - Paper
     - The settings that decide the curve
     - Agreement
     - Run
   * - Häger & Pfister 2021, Fig. 5 (DBP, one step per span)
     - ``dbp`` with ``steps_per_span: 1`` and the default effective length
     - within 0.20 dB at 25 points
     - ``specs/validation/hager2021_fig5.yaml`` (CI)
   * - Häger & Pfister 2021, Fig. 8 (DBP, 2 to 500 steps per span)
     - ``step_distribution: modified_logarithmic``, ``step_factor: 0.4``;
       ``nonlinear_length: step`` above 4 dBm
     - within 0.15 dB up to 6 dBm, 0.33 dB at 8 dBm
     - ``specs/validation/hager2021_fig8.yaml`` (150 s)
   * - Häger & Pfister 2021, Fig. 8 (the forward steps)
     - the reference, 500 uniform steps per span; every technique against it
     - within 0.07 dB; 0.01 dB from 22 uniform steps
     - the gallery (60 s), ``just card hager2021_convergence`` (CI)
   * - Musetti, Serena & Bononi 2018, Fig. 7 (step accuracy)
     - ``step_rule: fwm`` at the figure's four-wave-mixing phase
     - within 2.4 dB from −72 to −7 dB
     - ``just card musetti2018_fig7`` (63 s)
   * - Dar et al. 2014, Sec. 3 (NLI of QPSK and 16-QAM)
     - ``nonlinear_step: midpoint`` (the default), steps of 0.5 km, 64 runs
     - 9.95 and 5.78 dB against 10 and 6 dB
     - ``just card dar2014_ssfm`` (80 s)
   * - Shao, Liang & Kumar 2014, Table 2 (error of one step)
     - ``nonlinear_step: midpoint`` (scheme I) or ``effective_length`` (scheme II)
     - scheme I within 12 % at 10 GBd; scheme II not reproduced
     - ``just card shao2014_table2`` (3 min)
   * - Optilux and OptiCommPy (the same field)
     - Optilux: ``effective_length``; OptiCommPy: ``step_rule: nonlinear_phase``,
       ``nonlinear_step: trapezoidal``
     - :math:`10^{-26}` and :math:`10^{-8}`
     - ``tests/oracle/``

Häger & Pfister 2021, Fig. 5: back-propagation with one step per span
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

C. Häger and H. D. Pfister, *IEEE J. Sel. Areas Commun.* 39(1), 2021, Fig. 5: one
channel of Gaussian symbols at 10.7 GBd over 25 × 80 km of standard fibre
(:math:`\beta_2 = -21.683` ps²/km at 194.6 THz, :math:`\gamma = 1.3` /W/km, 0.2 dB/km,
noise figure 5 dB), root-raised cosine of roll-off 0.1, propagation at 6 samples per
symbol, receivers at 2. The lines of the link that matter:

.. code-block:: yaml

   fiber: {n_spans: 25, span_km: 80, alpha_db_km: 0.2, gamma_w_km: 1.3, d_ps_nm_km: 17.20934, fc_hz: 194.6e12, nf_db: 5.0}
   link:
     - {type: pulse_shape, sps: 6, rolloff: 0.1, n_taps: 6145}
     - {type: ssfm, step_km: 0.4, precision: f32}
     - {type: resample, sps: 2}
     - {type: dbp, steps_per_span: 1, nonlinear: true}   # nonlinear: false, linear equalisation

- The forward propagation is converged: the paper's 500 logarithmic steps per span
  (``{type: ssfm, steps_per_span: 500}``) move every point by 0.01 dB at most, for six
  times the computation.
- The back-propagation with one step takes the nonlinear phase over the effective
  length of the span, the default. This example found that the coarse step used to take
  the step length (60 % too much phase with one step per span); it is fixed.
- Both curves are within 0.20 dB of the 25 points of the figure (0.15 dB below 2 dBm).
  CI runs the specification; the gallery draws it:
  :doc:`../auto_examples/optical/plot_03_hager2021_fig5`.

.. code-block:: bash

   comnum verify specs/validation/hager2021_fig5.yaml

Häger & Pfister 2021, Fig. 8: the steps of the back-propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same paper, Fig. 8: 32 GBd over 10 × 100 km, linear equalisation and DBP with 2, 4
and 500 steps per span, frames of 1024 symbols, the forward link at 500 logarithmic
steps per span (its Table I). This figure is the one where the settings decide the
curve:

.. code-block:: yaml

   link:
     - {type: ssfm, steps_per_span: 500}
     - {type: resample, sps: 2}
     - {type: dbp, steps_per_span: 4, step_distribution: modified_logarithmic, step_factor: 0.4, nonlinear_length: step}

- **The steps of the DBP.** The paper uses the modified logarithmic steps of Zhang, Li &
  Dong (2013) with the factor 0.4 (its Sec. IV-C). With equal steps the DBP with 2 steps
  per span falls 2.6 to 4.9 dB below the figure from 2 to 6 dBm, and with Bosco's
  logarithmic steps 1.7 to 3.1 dB below.
- **The nonlinear length of a coarse step.** The code of the paper takes the power at
  the middle of the step times the step, ``nonlinear_length: step``. comnum's default,
  ``effective``, integrates the power over the step and back-propagates better: with it
  the DBP with 4 steps is 0.29, 0.54, 1.07 and 1.03 dB above the figure from 5 to 8 dBm.
  With ``step`` it is within 0.09 dB up to 6 dBm, 0.24 and 0.33 dB above at 7 and 8 dBm.
- Linear equalisation is within 0.06 dB, the DBP with 2 steps within 0.14 dB, with 500
  steps within 0.15 dB. The specification runs both conventions (the cases
  ``dbp2_step`` and ``dbp4_step``); about 150 s on 12 threads, not in CI.

.. code-block:: bash

   comnum verify specs/validation/hager2021_fig8.yaml

How many steps: the forward model of Fig. 8, and the other techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reference is the split step with uniform steps and many of them: 500 steps per span
of 200 m on the link of Fig. 8. With an error in :math:`h^2` (Sinkin et al., *J. Lightw.
Technol.* 21(1), 2003, Sec. III-E), more steps change nothing: 2000 steps per span move
the effective SNR with DBP by :math:`10^{-7}` dB at 6 dBm. The reference checks the
steps and nothing else. A wrong sign or unit in the equation converges just as well,
which is why the solvers are also compared with exact solitons and with the fields of
Optilux and OptiCommPy.

With this reference in the forward model, the specification of Fig. 8 follows the
published curves within 0.04 dB with linear equalisation and 0.07 dB with DBP of 500
steps per span, up to 6 dBm (8 frames per point). With fewer forward steps the DBP curve
falls, because a back-propagation that knows the fibre exactly cannot undo the error of
a coarse simulation:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20

   * - Forward model
     - DBP at 6 dBm
     - DBP at 8 dBm
   * - 10 uniform steps per span
     - −0.30 dB
     - −0.88 dB
   * - 10 logarithmic steps per span
     - −1.07 dB
     - −2.13 dB

Linear equalisation barely sees these errors: its own nonlinear interference is much
larger. The DBP receiver is the strict test, and it decides how many steps a
simulation needs. At 6 dBm, the peak of the DBP curve, the SNR lost against the
reference with every technique, and the computation time of the propagation on the
machine of the documentation (10 spans, 6144 samples):

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - Technique
     - Steps per span for 0.01 dB
     - Time for 0.01 dB
     - Lost with 10 steps per span
   * - uniform, midpoint phase (``nonlinear_step: midpoint``, the default)
     - about 22
     - 12 ms
     - 0.21 dB
   * - uniform, effective length (``effective_length``, Optilux)
     - about 20
     - 10 ms
     - 0.16 dB
   * - uniform, single precision (``precision: f32``)
     - about 22
     - 7 ms
     - 0.21 dB
   * - logarithmic (``steps_per_span``, Bosco; Häger & Pfister)
     - about 45
     - 30 ms
     - 1.12 dB
   * - uniform, trapezoidal (``trapezoidal``, OptiCommPy)
     - about 50
     - 110 ms
     - 4.13 dB
   * - four-wave-mixing rule (``manakov``, ``step_rule: fwm``, Musetti et al.)
     - about 16
     - 30 ms
     - adaptive
   * - nonlinear-phase rule (``manakov``, ``step_rule: nonlinear_phase``, OptiCommPy)
     - about 100
     - 420 ms
     - adaptive

- **The paper's 500 steps are more than enough.** They lose less than
  :math:`10^{-4}` dB, in 325 ms. Uniform steps reach 0.01 dB with about 22 steps per
  span, in 12 ms: a twenty-fifth of the computation.
- **Every technique converges as the square of the step.** Only the constant in front
  differs, and with it the number of steps.
- **The effective length is the most accurate fixed step.** It loses a little less than
  the midpoint phase for the same time. The trapezoidal mean iterates, four times the
  time per step, and needs more steps.
- **Logarithmic steps need twice as many steps for one channel.** They keep the spurious
  four-wave mixing of the split step low in WDM (Bosco et al. 2000). They are not built
  for the error of one channel, which their long steps at the end of the span decide.
- **Single precision** costs half the time and has a floor near
  :math:`2 \cdot 10^{-4}` dB.
- **The adaptive rules** run in the Manakov solver, which carries two polarisations, so
  their times are about twice those of the scalar solver per step. The
  four-wave-mixing rule needs the fewest steps. The default of ``manakov``, 20 rad with
  at most 0.02 rad per step, takes 107 steps per span and loses less than
  :math:`10^{-4}` dB.

To compare the techniques on the same noise, this table propagates without noise and
adds the ASE of the link at the receiver. Its losses with 10 steps per span are those
of the specification within 0.1 dB (0.21 and 1.12 dB against 0.30 and 1.07 dB). The
gallery draws both figures: :doc:`../auto_examples/optical/plot_08_split_step_techniques`.
CI checks the convergence of every technique on two spans of the same link (card
``hager2021_convergence``):

.. code-block:: bash

   just card hager2021_convergence

Musetti, Serena & Bononi 2018, Fig. 7: how accurate a step rule is
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

S. Musetti, P. Serena and A. Bononi, *J. Lightw. Technol.* 36(23), 2018, Fig. 7: seven
dual-polarisation 16-QAM channels of 49 GBd spaced by 50 GHz, 0 dBm per channel, one
span of 100 km, the error of the split step against a reference, for the step rules of
the paper. The test calls the Manakov solver with the four-wave-mixing rule alone
(``step_rule: fwm``, the phase ``fwm_phase_rad`` of the figure), the phase at the middle
of the step and Nyquist pulses without slope (the paper: roll-off 0.01 and a slope of
0.057 ps/(nm²·km)).

- The error follows the figure within 2.4 dB from −72 to −7 dB: −70.5, −40.2, −21.3 and
  −7.3 dB at 4.5, 9, 18 and 36 rad against −72, −40, −21 and −7.
- The rule alone is not enough for one channel at high power: a DP-16QAM channel of
  32 GBd over 10 × 80 km gets 0.28 to 0.66 dB too much SNR from 2 to 8 dBm with 20 rad.
  The default of ``manakov`` therefore also bounds the nonlinear phase of a step
  (``max_nl_phase_rad``, 0.02 rad); keep it when the link is not the paper's.

.. code-block:: bash

   just card musetti2018_fig7

Dar et al. 2014: the nonlinear step in WDM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

R. Dar, M. Feder, A. Mecozzi and M. Shtaif, *Opt. Express*, 2014, Sec. 3: five channels
of Nyquist pulses at 32 GBd spaced by 50 GHz, 500 km of lossless fibre, −16 dBm per
channel, no noise. The paper finds that the GN model overestimates the interference of
QPSK by about 10 dB, and that QPSK and 16-QAM differ by about 6 dB.

.. code-block:: yaml

   - {type: manakov, step_km: 0.5, nonlinear_step: midpoint}

- With the phase at the middle of the step, the scalar solver gives 9.95 and 5.78 dB,
  and the Manakov solver follows the GN model within 0.17 dB with Gaussian symbols.
- With OptiCommPy's trapezoidal step at the same 0.5 km, the Manakov solver finds 0.90
  dB (Gaussian) and 1.29 dB (QPSK) less interference: the mean of the power at the ends
  of the step smooths the cross-phase modulation of channels that walk off.
- The runs matter: 16 samples per symbol, 4096 symbols per channel and 64 runs per
  format. The first two seeds alone give 5.3 dB instead of 10.

.. code-block:: bash

   just card dar2014_ssfm
   just card dar2014_manakov

Shao, Liang & Kumar 2014, Table 2: two nonlinear steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

J. Shao, X. Liang and S. Kumar, *IEEE Photonics J.* 6(4), 2014, Table 2: the error of a
single step of 0.5 to 60 km on 32-QAM at 10 and 25 GBd, with the loss in the dispersion
(scheme I) or in the nonlinearity (scheme II). Scheme I is ``nonlinear_step:
midpoint``, scheme II ``nonlinear_step: effective_length``, Optilux's.

- Scheme I is within 12 % of the table at every step at 10 GBd and within a factor 2.7
  at 25 GBd, with the order :math:`h^6` of the table.
- Scheme II is 2 to 4.9 times more accurate than the table, which is not explained: it
  is the scheme of the paper's eq. (21), and it gives Optilux's fields to
  :math:`10^{-28}`. Do not expect this table to come out for scheme II.

.. code-block:: bash

   just card shao2014_table2

Another simulator on the same field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare with another simulator, take its scheme, its steps and its input field, and
undo its sign convention:

- **Optilux** (``fiber``): ``nonlinear_step="effective_length"`` at its fixed step
  (``dzmax``); Optilux writes the equation with the opposite sign, so compare with the
  conjugate of its output. NMSE :math:`3 \cdot 10^{-26}` after ten spans, and
  :math:`1.5 \cdot 10^{-8}` with the default midpoint step.
- **OptiCommPy** (``manakovSSF``): ``Manakov(step_rule="nonlinear_phase",
  nonlinear_step="trapezoidal")``. Relative difference :math:`10^{-8}`.
- **Sionna**: the default midpoint step at the same fixed step, on the conjugate field.
  NMSE :math:`4.8 \cdot 10^{-26}` (:doc:`../benchmarks`).

Not reproduced
~~~~~~~~~~~~~~

Two published results are not reproduced, and the cards say so:

- **Ip & Kahn 2008, Fig. 12** (``validation/cards/ip2008_fig12.yaml``): the phase error
  of the paper is larger than comnum's at every distance (linear equalisation 0.114
  against 0.204 rad at 25 spans), with the model that reproduces Häger & Pfister's Fig. 5.
- **Sinkin et al. 2003, Fig. 2** (``validation/cards/sinkin2003_fig2.yaml``): the error
  of constant steps on a soliton collision is 1.5 to 2 times the figure's, with the same
  order 2; the exact two-soliton solution confirms comnum's values.

When a curve does not match
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check these, in this order; each one moved a published curve by a decibel or more in
the examples above:

1. **The steps of the back-propagation**: equal, logarithmic or modified logarithmic
   (``step_distribution``), and the nonlinear length of a coarse step
   (``nonlinear_length``).
2. **The nonlinear step of the forward propagation** (``nonlinear_step``): the
   trapezoidal mean loses about 1 dB of interference in WDM at steps of 0.5 km.
3. **The step rule**: a rule on the four-wave mixing alone is too coarse for one
   channel at high power; a rule on the nonlinear phase alone depends on the power.
4. **The sampling**: the four-wave mixing that falls on the channel under test needs
   :math:`\frac{3}{2} B_{\mathrm{WDM}} + \frac{1}{2} B_{\mathrm{CUT}}`; keep
   ``check_sampling`` on.
5. **The number of runs**: the interference of a few blocks varies by several decibels.
6. **The coefficient**: the Manakov equation applies :math:`\frac{8}{9}\gamma`. To
   reproduce a scalar result with the Manakov solver and one polarisation empty, scale
   :math:`\gamma` by :math:`\frac{9}{8}`; to compare the scalar solver with a Manakov
   simulation, give it :math:`\frac{8}{9}\gamma`.

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
