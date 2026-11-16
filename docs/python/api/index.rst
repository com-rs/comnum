API reference
=============

comnum is used in two ways, and the reference has a part for each:

- :doc:`functions`: the functions and objects of the package, which take and return
  NumPy arrays (``comnum.awgn_ebn0(x, ...)``, ``comnum.Constellation("qam", 16)``,
  ``comnum.Ssfm(...)``), grouped by theme in the order of a link;
- :doc:`blocks`: the blocks of :mod:`comnum.blocks`, assembled into a :class:`~comnum.Chain`
  or a declarative specification and run by the Monte Carlo runner, grouped by their
  role in the link.

Every name of the first part, and :class:`~comnum.Chain` and the runners of the second,
are importable from the package: ``import comnum``; the blocks from ``comnum.blocks``.

.. toctree::
   :maxdepth: 2

   functions
   blocks
