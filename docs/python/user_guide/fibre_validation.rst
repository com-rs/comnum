Fibre propagation: how the split step is validated
==================================================

The fibre solvers of comnum, :class:`comnum.Ssfm` (one polarisation),
:class:`comnum.ManakovSsfm` (two) and the digital back-propagation built on them, are
compared with about fifteen references. Each one checks something the others cannot,
and none is enough on its own. An exact solution proves the scheme but has no
dispersion map or WDM comb; a perturbation model covers a WDM link but neglects the
higher orders; another simulator shows that two codes agree, not that either is right.
This page orders them by what they check, from the numerical scheme up to a published
system, and states what each found.

.. list-table:: Which reference answers which question
   :header-rows: 1
   :widths: 34 36 30

   * - Question
     - Reference
     - Answer
   * - Is the scheme right, and of second order?
     - exact soliton collisions: Hirota (1973) on the system of Sinkin et al. (2003);
       Radhakrishnan et al. (1997)
     - exact to :math:`3 \cdot 10^{-9}`; order 2.005 and 1.999
   * - Do the polarisations couple correctly?
     - Radhakrishnan, Lakshmanan & Hietarinta (1997), Fig. 1
     - 0.40 % of energy switched, 0.39 % by eq. (16)
   * - Is the cross-phase modulation between channels right?
     - Chiang et al. (1996), eq. (9)
     - within 0.25 % from 0.125 to 2 GHz
   * - Are the steps small enough?
     - Musetti, Serena & Bononi (2018), Fig. 7
     - within 2.4 dB from −72 to −7 dB of accuracy
   * - Is the interference of a WDM link right?
     - GN model (Poggiolini 2012), EGN model (Carena et al. 2014), the simulations of
       Dar et al. (2014)
     - within 0.08 to 0.25 dB (GN); 9.95 and 5.78 dB against 10 and 6 dB (Dar)
   * - Does a published system come out the same?
     - Häger & Pfister (2021), Fig. 5
     - within 0.20 dB
   * - Does another simulator give the same field?
     - OptiCommPy 0.11, Sionna, GNPy 3.0
     - to :math:`10^{-8}` (OptiCommPy, its own options); NMSE :math:`5 \cdot 10^{-26}`
       (Sionna); 0.02 dB (GNPy, Raman)

Each result below comes from a validation card of ``validation/cards/``, which holds
the published values, the tolerance and its reason, and the findings; ``just card
<id>`` reruns one. The :doc:`maturity <../maturity>` page grades the module, and
``docs/LITERATURE.md`` lists every reference with its badge.

1. The scheme: exact solutions
------------------------------

The nonlinear Schrödinger and Manakov equations have exact multi-soliton solutions.
Propagating one with the solver and comparing with the formula checks the scheme
itself: the symmetric split step (half dispersion, nonlinear phase, half dispersion),
its order, and the time and frequency grids, with nothing approximated on the
reference side.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Reference
     - Setup
     - Result
   * - the exact two-soliton solution (Hirota, *J. Math. Phys.* 14(7), 1973) on the
       collision of O. V. Sinkin, R. Holzlöhner, J. Zweck & C. R. Menyuk, *J. Lightw.
       Technol.* 21(1), 2003, Fig. 2
     - two solitons 800 GHz apart colliding over 400 km, constant steps
     - the scalar solver within :math:`2.95 \cdot 10^{-9}` with steps of 1 m, order
       2.005; the Manakov solver also second order, with 6.4 times the error at
       equal steps (card ``sinkin2003_exact``)
   * - Sinkin et al. (2003), Fig. 2: the global error against the number of FFTs
     - the same collision
     - the same slope as the figure, 1.5 to 2 times its error at every point, **not
       explained** (card ``sinkin2003_fig2``, badge orange)
   * - the exact bright two-soliton solution of the Manakov system: R.
       Radhakrishnan, M. Lakshmanan & J. Hietarinta, *Phys. Rev. E* 56(2), 1997,
       eqs. (7)–(16) and Fig. 1
     - an inelastic collision that switches energy between the polarisations
     - 0.40 % of soliton 1 left in component 1 (midpoint step), 0.38 %
       (trapezoidal), against 0.39 % by eq. (16); second order down to
       :math:`2.4 \cdot 10^{-6}`, no floor (card ``radhakrishnan1997_fig1``)

Unit tests check the invariants: the energy of a lossless propagation is conserved,
the fundamental soliton keeps its shape, and the back-propagation inverts the forward
propagation (``crates/comnum-math/src/nlse/mod.rs``,
``crates/comnum-channel/src/optical/fiber.rs``).

2. The interaction of channels: cross-phase modulation
------------------------------------------------------

In a WDM link, channels at different frequencies travel at different group
velocities, so the phase one imprints on another is filtered by their walk-off. The
small-signal theory of T.-K. Chiang, N. Kagi, M. E. Marhic & L. G. Kazovsky (*J.
Lightw. Technol.* 14(3), 1996, eqs. (8)–(9)) gives this filter in closed form for a
continuous probe and a sinusoidally modulated pump.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Reference
     - Setup
     - Result
   * - Chiang et al. (1996), eq. (9)
     - probe 125 GHz from the pump, 80 km, modulation from 0.125 to 2 GHz, where
       the walk-off divides the XPM by 4.5
     - the scalar solver and the Manakov solver (midpoint step) within 0.25 % with
       steps of 0.5 km; OptiCommPy's trapezoidal step gives 0.447 of the XPM at
       those steps and needs 0.02 km to reach 0.1 % (card ``chiang1996_eq9``)

This comparison is the origin of the default nonlinear step of the Manakov solver
(section 7).

3. The step: how small is small enough
--------------------------------------

S. Musetti, P. Serena & A. Bononi (*J. Lightw. Technol.* 36(23), 2018) bound the
error of the split step by the phase mismatch of the four-wave mixing across the
signal bandwidth, and derive the step rule that comnum uses by default
(``step_rule="fwm"``, eq. (13)). Their Fig. 7 gives the accuracy obtained for a given
bound.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Reference
     - Setup
     - Result
   * - Musetti et al. (2018), Fig. 7
     - 7 channels, one span, symmetric step, FWM-CLE and FWM-NLP variants
     - within 2.4 dB over an accuracy of −72 to −7 dB: −70.5, −40.2, −21.3 and
       −7.3 dB at 4.5, 9, 18 and 36 rad, against −72, −40, −21 and −7 dB read on the
       figure (card ``musetti2018_fig7``)

The rule alone is not enough for one channel at high power: one DP-16QAM channel of
32 GBd over 10 × 80 km gets 0.28 to 0.66 dB too much SNR from 2 to 8 dBm, because the
steps grow beyond its dispersion length. The default rule therefore also bounds the
nonlinear phase of a step to 0.02 rad (``max_nl_phase``), which gives the result of
steps of 0.1 km.

The same paper gives the sampling rate and the block length that a WDM simulation
needs (Table I and Appendix B). The solvers check both on the launched field before
every propagation (``check_sampling``, ``docs/adr/0014-one-fibre-propagator.md``).

4. The link: perturbation models
--------------------------------

Over a whole WDM link the reference is no longer a closed form of the field but a
model of the nonlinear interference (NLI). The models are first order in the
nonlinearity and assume a Gaussian (GN) or a given (EGN) symbol distribution, so they
are compared in the regime where they hold.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Reference
     - Setup
     - Result
   * - the GN model: P. Poggiolini, *J. Lightw. Technol.* 30(24), 2012, the PSD of
       the NLI integrated over the band of the channel
     - Gaussian symbols, Manakov solver, at −4 and 2 dBm
     - self-channel interference within 0.08 dB, total within 0.08 dB,
       cross-channel within 0.25 dB (card ``poggiolini2012_band``)
   * - the EGN model: A. Carena, G. Bosco, V. Curri, Y. Jiang, P. Poggiolini & F.
       Forghieri, *Opt. Express* 22(13), 2014, Fig. 1(a)
     - the self-channel interference of PM-QPSK at 1, 10, 20 and 50 spans (models
       only)
     - GN within 0.2 dB, EGN within 0.25 dB of the curves (card
       ``carena2014_fig1a``)
   * - R. Dar, M. Feder, A. Mecozzi & M. Shtaif, *Opt. Express* 22(12), 2014,
       Sec. 3: their split-step simulation of 500 km of distributed amplification
     - QPSK, 16-QAM and Gaussian symbols, scalar solver
     - the error of the GN model for QPSK 9.95 dB and QPSK against 16-QAM 5.78 dB,
       against the paper's 10 and 6 dB; Gaussian symbols within 0.16 dB of the GN
       model (card ``dar2014_ssfm``)
   * - Dar et al. (2014), the same link with two polarisations
     - Manakov solver, midpoint step
     - Gaussian symbols 0.17 dB below the GN model, QPSK 0.35 dB below the EGN
       model; with OptiCommPy's trapezoidal step at the same steps, 0.90 and 1.29 dB
       below (card ``dar2014_manakov``)

The ``specs/wdm_gn.yaml`` specification checks the Manakov simulation against the GN
model from the ASE-limited regime to 4 dB above the optimal power (within 0.4 dB),
and 16-QAM against the EGN model (within 0.3 dB, where the GN model is up to 1.7 dB
off). The Raman-amplified links are compared with the generalised GN model in
``specs/raman_wdm.yaml`` (within 0.21 dB).

5. The system: a published result
---------------------------------

C. Häger & H. D. Pfister (*IEEE J. Sel. Areas Commun.* 39(1), 2021, Fig. 5) give the
effective SNR of linear equalisation and of digital back-propagation (DBP) with one
step per span over 25 × 80 km, against the launch power. comnum reproduces both
curves within 0.20 dB of the 25 digitised points (``specs/validation/hager2021_fig5.yaml``,
card ``hager2021_fig5``, run in CI; gallery
:doc:`../auto_examples/optical/plot_03_hager2021_fig5`).

This comparison found a bug: the back-propagation with coarse steps applied the
nonlinear phase over the length of the step instead of its effective length. It is
fixed in ``Ssfm::backpropagate``.

6. Other simulators
-------------------

Agreeing with another simulator shows that both solve the same equation the same way;
it does not show that the way is accurate. These comparisons are therefore run with
the same scheme and the same steps on both sides, on the same input field, and their
outputs are recorded once (``tests/oracle/``), so the tools are not dependencies.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Simulator
     - What is compared
     - Result
   * - `OptiCommPy <https://github.com/edsonportosilva/OptiCommPy>`_ 0.11
     - the linear fibre, ``ssfm`` and ``manakovSSF`` with WDM, on the same field
       (``tests/oracle/test_oracle.py``)
     - relative difference :math:`10^{-8}` with OptiCommPy's options
       (``step_rule="nonlinear_phase"``, ``nonlinear_step="trapezoidal"``)
   * - `Sionna <https://github.com/NVlabs/sionna>`_ and OptiCommPy, benchmark scenario
       5
     - 16-QAM, 32 GBd, 10 × 80 km, steps of 1 km, float64
       (``benchmarks/libraries/s5_ssfm.py``)
     - NMSE of the fields :math:`5.7 \cdot 10^{-26}` (OptiCommPy) and
       :math:`4.8 \cdot 10^{-26}` (Sionna); all three are :math:`3.7 \cdot 10^{-8}`
       from the reference at 0.05 km (:doc:`../benchmarks`). Sionna's sign
       convention gives the complex conjugate, which the comparison undoes.
   * - `Optilux <https://optilux.sourceforge.io/>`_ Apr 2021, under Octave
     - the same scenario at its quick size (2¹¹ symbols), and one span with and
       without the Kerr effect (``tests/oracle/test_optilux_oracle.py``,
       recorded by ``tools/record_optilux.py``)
     - NMSE :math:`3 \cdot 10^{-28}` without the Kerr effect, :math:`1.2 \cdot 10^{-10}`
       after one span and :math:`1.5 \cdot 10^{-8}` after ten with it: Optilux's
       nonlinear step uses the effective length of the step, comnum its length;
       Optilux is :math:`1.1 \cdot 10^{-8}` from the solution at 0.05 km, comnum
       :math:`3.7 \cdot 10^{-8}`. Optilux's sign convention is the conjugate too.
   * - `GNPy <https://github.com/Telecominfraproject/oopt-gnpy>`_ 3.0
     - Raman power profiles with co- and counter-pumps, their spontaneous noise,
       the generalised GN model (``tests/oracle/test_gnpy_oracle.py``)
     - 0.02 dB on the NLI, 0.004 dB on the profiles; GNPy at its default tolerance
       underestimates the NLI of counter-pumped spans by about 1 dB (Euler
       integration)

7. What the comparisons changed
-------------------------------

The solvers were first a transcription of OptiCommPy, equal to it to :math:`10^{-8}`.
The references of sections 1 to 5 found three problems, which changed the defaults
(``docs/adr/0013-manakov-defaults.md``):

- **the nonlinear step.** OptiCommPy takes the mean of the power at the two ends of a
  step, iterated. When channels walk off within a step, the mean smooths the intensity
  that drives the XPM: 0.447 of the XPM of Chiang et al. at steps of 0.5 km, and 0.9
  to 1.3 dB too little NLI on the link of Dar et al. The default is now the power at
  the middle of the step (``nonlinear_step="midpoint"``), as in the scalar solver;
- **the step rule.** OptiCommPy bounds the nonlinear phase of a step (0.02 rad); the
  rule depends on the power and ignores the bandwidth, and left 0.45 dB of error on
  the WDM specifications. The default is now the four-wave-mixing rule of Musetti et
  al., with the bound on the nonlinear phase kept for a single channel at high power;
- **the sampling.** With 8 samples per symbol, the four-wave mixing of a five-channel
  comb aliased onto the centre channel and added 0.3 dB of NLI; the WDM
  specifications now use 12, and the solvers refuse a field that is under-sampled or
  too short (``docs/adr/0014-one-fibre-propagator.md``).

At equal accuracy the defaults are 6 to 11 times faster than OptiCommPy's scheme on
the same Rust code. The error is measured against fixed steps of 0.25 km:

.. list-table::
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Case
     - OptiCommPy's choices
     - OptiCommPy's scheme, steps of 0.1 km
     - Defaults
     - Reference (0.25 km)
   * - ``specs/wdm_gn.yaml`` (5 channels, 5 × 80 km)
     - 0.44 dB, 18 s
     - 0.02 dB, 291 s
     - 0.02 dB, 27 s
     - 61 s
   * - ``specs/raman_wdm.yaml`` (counter-pumped Raman, 3 × 100 km)
     - 0.45 dB, 7 s
     - 0.02 dB, 135 s
     - 0.00 dB, 21 s
     - 31 s
   * - one DP-16QAM channel, 10 × 80 km, 2 to 8 dBm
     - 0.03 dB, 1.7 s
     -
     - 0.00 dB, 1.4 s
     - 7.4 s

Times on an Apple M6 for the whole specification, measured with 8 samples per symbol
(before the specifications moved to 12, which affects every column alike).
OptiCommPy's solver remains available with
``Manakov(step_rule="nonlinear_phase", nonlinear_step="trapezoidal")``.

8. What is not validated
------------------------

- **Fig. 2 of Sinkin et al.**: the error is 1.5 to 2 times the figure's at every point,
  with the right slope, and is not explained. The exact solution of the same collision
  is reproduced to :math:`3 \cdot 10^{-9}`, and Fig. 7(e) of the same paper contradicts
  its Fig. 2 by a factor 10.
- **Ultra-wideband links** (P. Serena et al., *J. Lightw. Technol.* 38(5), 2020):
  the accuracy of the split step and the inter-channel Raman scattering at
  C+L-band widths are not compared (badge red).
- **PMD**: the coarse-step model of D. Marcuse, C. R. Menyuk & P. K. A. Wai (*J.
  Lightw. Technol.* 15(9), 1997) is checked for unitarity only (badge red).
- **Third-order dispersion** is checked against its cubic phase and the textbook
  value of :math:`\beta_3` (``nlse/mod.rs::the_third_order_dispersion_adds_a_cubic_phase``,
  ``fiber.rs::beta3_from_the_slope_is_the_textbook_value``), not against a published
  propagation.
- **Lumped amplification in Dar et al.**: only the distributed case is simulated; the
  lumped cases are compared through the EGN model, which lies 0.07 to 0.72 dB below
  the paper.
- **Carena et al., Fig. 1(a), 2 to 5 spans**: the published EGN curve lies up to
  0.9 dB below the formulas of the same paper; comnum follows the formulas.

Rerunning the comparisons
-------------------------

Every card reruns from the root of the repository; the campaigns take from a few
seconds to 8 minutes in release mode:

.. code-block:: bash

   just card sinkin2003_exact
   just card radhakrishnan1997_fig1
   just card chiang1996_eq9
   just card musetti2018_fig7
   just card poggiolini2012_band
   just card dar2014_ssfm
   just card dar2014_manakov
   just card hager2021_fig5

The comparisons with OptiCommPy and GNPy are part of the Python tests
(``python -m pytest tests/oracle/test_oracle.py tests/oracle/test_gnpy_oracle.py``), and the
benchmark of section 6 reruns with ``python benchmarks/libraries/s5_ssfm.py``.
