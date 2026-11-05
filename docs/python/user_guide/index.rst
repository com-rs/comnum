User guide
==========

A link is a chain of blocks applied to NumPy arrays; a *campaign* evaluates that link
on a grid of conditions (an SNR, a launch power, a modulation order…), as many times
as a stopping rule requires, and reports every metric with a confidence interval.
``comnum`` offers two ways to run one:

- :doc:`monte_carlo`: the trial is a Python function, run by :mod:`comnum.sim` —
  the quickest way to prototype;
- :doc:`specifications`: the whole campaign is a YAML file run by the parallel Rust
  runner — the way to produce the numbers of a study.

A :doc:`chain <chains>` of named blocks bridges the two: built and inspected in
Python, it runs on the Rust runner and converts to and from a specification. The
blocks cover :doc:`wireless and coded links <wireless>` (antennas, fading, OFDM, MIMO
detection, channel codes), the :doc:`waveforms beyond OFDM <waveforms>` (FBMC/OQAM,
UFMC, OTFS, chirp spread spectrum), :doc:`optical links <optical>` (WDM, dual
polarisation, nonlinear propagation, back-propagation, and :doc:`how the fibre
solvers are validated <fibre_validation>`) and :doc:`optical wireless links
<optical_wireless>` (visible light and infrared indoors, free-space optics), and the
functions that follow a :doc:`standard <standards>` clause by clause.

Both draw trial ``t`` of condition ``i`` from the same stream
``Rng.stream(seed, name, i, t)``, so a scenario gives the same draws in both, and the
results never depend on the number of threads.

.. toctree::
   :maxdepth: 1

   monte_carlo
   chains
   analysis
   wireless
   waveforms
   optical
   fibre_validation
   optical_wireless
   standards
   nr_conformance
   specifications
   recordings
