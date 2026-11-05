Why the results can be trusted
==============================

A block of comnum is not only tested: it is compared with something outside the
library, and the comparison is kept as data that anyone can rerun. This page explains
the four kinds of evidence, where they are recorded, and how to read the grades of the
:doc:`maturity <maturity>` page.

The four kinds of evidence
--------------------------

**A closed form.** The simplest and the strongest: the error rate of QAM over AWGN
(Proakis & Salehi), the capacity of BICM (Caire, Taricco & Biglieri 1998), the SNR of
the GN model (Poggiolini 2012). The Monte Carlo result must fall inside its confidence
interval around the formula, or the formula inside the interval of the simulation.

**A table of a standard.** The constellations of DVB-S2, the clusters of the 3GPP
channel models, the SSS indices of LTE, the generator polynomials of the CRCs: the
tables are transcribed from the document and compared verbatim, and a chain built on
them reproduces the performance the standard requires (the Es/N0 of the 28 MODCODs of
DVB-S2 at a FER of 10 %, the throughput of the 5G NR conformance tests).

**A published figure.** The points of a figure are read off the paper with the
precision of the reading (the grid of the figure, the marker size), and the simulation
is run with the parameters of the paper. The tolerance states why it is what it is.
When comnum and the paper disagree, the cause is found before anything is loosened: a
parameter, a misprint, a figure that contradicts its own text, or a bug in comnum. The
inconsistencies found in the papers are recorded next to the comparison.

**The recorded output of a reference tool.** OptiCommPy, GNPy, AFF3CT, Sionna, py3gpp
and sigmf-python are run once, on inputs the tests generate from seeded generators,
and their outputs are stored (``tests/oracle/``, ``validation/aff3ct/``) with the
version of the tool and the command. The tests compare with the recording: the tools
are never dependencies, and none of their code is copied. Agreeing with a tool is
evidence, not proof: two implementations of a standard can share a misreading, so the
differences found are traced to the document and documented
(:doc:`comparisons <comparisons>`).

Validation cards
----------------

A validation card is one comparison written as data: the source (a paper with its
equation, table or figure; a standard with its section; a reference implementation
with its version), the expected values with their tolerance and why, the differences
found, and the runner that measures the values again. Rerunning a comparison does not
need the paper: ``just card tal2015_fig1``.

.. code-block:: yaml

   id: chang2002_fig4_floors
   literature: chang2002
   source: {kind: paper, reference: "Chang & Su, IEEE TWC 1(4), 2002", locator: "Fig. 4, read at 50 dB"}
   run: {function: comnum.theory_ber_imperfect_csi}
   tolerance: {rtol: 0.3}
   why: "values read on the figure, to about 10 %"
   cases:
     - inputs: {modulation: 16qam, ebn0_db: 50.0, rho1: 0.999}
       expected: 3.5e-3

A card names one of three runners:

- ``function``: a closed form or a deterministic function of the package; the card is
  the test, run by ``tests/test_cards.py`` at every ``pytest``;
- ``spec``: a declarative campaign that the blocks can build, run by ``comnum verify``
  (the expected results of a :doc:`specification <user_guide/specifications>`); CI
  runs the short ones;
- ``test``: a Rust test of ``validation/tests/``, for a chain that needs code or a
  numerical property; the test measures and the card compares.

The cards live in ``validation/cards/`` with their schema; the long ones are tiered
``campaign`` and run before a release.

Two registries, three generated pages
-------------------------------------

``validation/literature.yaml`` has one entry per document (and per reference tool):
what was reproduced, with the numbers, the tests that do it, a green, orange or red
badge, what is not covered, and the remarks on the document itself. It renders
`docs/LITERATURE.md <https://github.com/com-rs/comnum/blob/main/docs/LITERATURE.md>`_.

``validation/maturity.yaml`` has one entry per module a user sees: its evidence, one
item per kind, and the regimes not covered yet. The level is computed from the kinds,
not written: **A** when an external reference is reproduced (a recording, a table, a
curve or a claim of a document), **B** when only a closed form is, **C** when the
module is self-consistent only (round trips, invariants). It renders the
:doc:`maturity <maturity>` page, the table of the README and ``docs/MATURITY.md``.

Both registries name their tests, and ``just check`` fails when a named test does
not exist, when a card points at a missing runner or document, or when a generated
page is out of date: the pages cannot drift from the evidence.

What the validation found
-------------------------

Comparing is how bugs are found, in comnum and in the references. Among the
findings, both fixed in comnum:

- coarse back-propagation applied the nonlinear phase over the step length instead of
  the effective length: with one step per span it applied 60 % of the phase
  (``specs/validation/hager2021_fig5.yaml`` now agrees with Häger & Pfister 2021 within
  0.13 dB);
- OptiCommPy's estimators of MI and GMI normalise the power of the noisy signal,
  which underestimates the rates by 0.06 bit at 0 dB; the declarative metrics use the
  least-squares channel gain, and the OptiCommPy-compatible functions are kept for
  the oracle tests.

And in the references: the sign misprint of eq. (58) of Afisiadis et al. (2020), the
"more than 3 dB" of Caire et al. (1998) that their own equations put at 2.55 dB, the
prefix of the first symbol that py3gpp lengthens in every slot where TS 38.211 says
every half subframe, the base graph 2 at :math:`Z_c = 64` that py3gpp 0.6 encodes outside the
code. Every one is listed with the comparison that found it.

Reading a disagreement
----------------------

A disagreement with a reference is a bug report, whether the error is in comnum or in
the document. The issue template *validation discrepancy* asks for the reference with
its locator, the values, and the specification or the card that reproduces the gap.
`Contributing a validation <https://github.com/com-rs/comnum/blob/main/CONTRIBUTING.md>`_
is the shortest way into the project: a paper to reproduce is a self-contained task with a clear end.
