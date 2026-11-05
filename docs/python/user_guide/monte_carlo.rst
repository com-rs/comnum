Monte Carlo campaigns in Python
===============================

:mod:`comnum.sim` runs Python trials on the Rust runner (the crate ``comnum-sim``), with
the vocabulary of PyTorch Lightning: a **trial** is what one random experiment does, a
**record** collects its metrics, a **stopping rule** says when a condition has enough
trials. The runner calls the Python code under the GIL, with one thread; its
statistics, stopping rules, statuses, condition labels and CSV and JSON reports are
those of the Rust campaigns.

The trial
---------

A trial receives the condition (a ``dict``), its own generator and a
:class:`~comnum.sim.Record`. It reports error counts, estimated as rates with Wilson
score intervals, or values, estimated as means with normal intervals:

.. code-block:: python

   import comnum
   from comnum import sim

   c = comnum.Constellation("qam", 16).normalized()

   def trial(cond, rng, rec):
       bits = comnum.bits(4 * 4096, rng)
       y = comnum.awgn_ebn0(c.map_bits(bits), cond["ebn0_db"], 4, rng)
       rec.errors("ber", int((c.demap_bits(y) != bits).sum()), bits.size)
       rec.value("evm", comnum.evm(y, c))

The grid and the stopping rule
------------------------------

:func:`~comnum.sim.grid` builds the Cartesian product of its axes (the last one varies
fastest). A :class:`~comnum.sim.Stop` is checked after every batch of trials; rules
combine with ``|`` (or) and ``&`` (and), and a rule must be bounded by
``max_trials`` or ``max_time``. The ``stop`` mapping of a specification works too
(``stop={"min_errors": {"ber": 200}, "max_trials": 2000}``, parsed by
:meth:`Stop.from_spec <comnum.sim.Stop.from_spec>` as the Rust runner parses it):

.. code-block:: python

   stop = sim.Stop.min_errors("ber", 200) | sim.Stop.max_trials(2_000)
   results = sim.run(trial, sim.grid(ebn0_db=range(0, 15, 2)), name="qam16",
                     seed=1, stop=stop)

   for r in results:
       lo, hi = r.metrics["ber"].interval(0.95)
       print(r.condition["ebn0_db"], r.metrics["ber"].value, (lo, hi), r.trials)

``csv_path=`` and ``json_path=`` write the reports of the Rust loggers. An exception of
a trial is raised again unchanged, with a note naming the trial and the condition
(``trial 3 of (ebn0_db=4) failed``).

Statistics
----------

An error count of :math:`k` errors over :math:`n` trials is reported with the Wilson
score interval

.. math::

   \frac{\hat p + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat p (1 - \hat p)}{n} + \frac{z^2}{4n^2}}}
        {1 + \frac{z^2}{n}}, \qquad \hat p = \frac{k}{n},

which stays inside :math:`[0, 1]` and is accurate with few errors; the relative
standard error of a rate is about :math:`1/\sqrt{k}`, so 100 errors give about 10 %
and 400 errors about 5 %. ``Stop.precision(metric, relative)`` stops when the relative
half-width of the interval falls below a target instead.

The Wilson interval is the one Brown, Cai and DasGupta recommend for small and moderate
numbers of trials (*Statistical Science* 16(2), 2001): the textbook (Wald) interval
:math:`\hat p \pm z\sqrt{\hat p(1 - \hat p)/n}` covers the true rate far less often
than its nominal level, 0.891 instead of 0.95 at :math:`n = 10` and :math:`p = 1/2`
(their Table 1), and not at all without errors. :meth:`ErrorRate.interval
<comnum.sim.ErrorRate.interval>` also gives the Wald, Agresti–Coull, Jeffreys and
Clopper–Pearson intervals, and :func:`~comnum.sim.interval_coverage` and
:func:`~comnum.sim.interval_length` their exact coverage and expected length for
:math:`X \sim \mathrm{Bin}(n, p)`, sums over the binomial law without simulation:

.. code-block:: python

   from comnum import sim

   r = sim.ErrorRate(errors=3, total=7)
   r.interval()                          # Wilson, the default: (0.158, 0.750)
   r.interval(method="clopper_pearson")  # "exact", conservative: (0.099, 0.816)
   sim.interval_coverage("wald", 10, 0.5)    # 0.891
   sim.interval_coverage("wilson", 10, 0.5)  # 0.979

``validation/tests/simulation/brown2001.rs`` reproduces the coverages of their Tables 1–3, the
limits of their Sec. 3.2 and the 252 Jeffreys limits of their Table 5
(:doc:`../maturity`; gallery :doc:`../auto_examples/simulation/plot_04_brown2001_fig5`).

Reproducibility
---------------

Trial ``t`` of condition ``i`` of the scenario ``name`` draws from
``comnum.Rng.stream(seed, name, i, t)``: a result is reproducible from the seed, a
single trial can be replayed alone, and the Rust runner draws exactly the same numbers
for the same scenario. For long campaigns, write the link with the blocks of the
registry (a native :class:`~comnum.Chain` or a :doc:`specification <specifications>`
runs on all cores) or in Rust: Python trials run on one thread.
