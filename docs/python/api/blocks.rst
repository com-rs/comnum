Blocks and chains
=================

.. currentmodule:: comnum

The second way to use comnum: a link written as a :class:`Chain` of named blocks, in
the spirit of ``torch.nn.Sequential``. A block is a class of :mod:`comnum.blocks`, one
per block of the declarative specifications (``awgn_ebn0`` → :class:`~comnum.blocks.AwgnEbn0`),
with the parameters of the block as keyword arguments; :func:`var` makes one a grid
variable. The same chain runs once to inspect its signals, runs a campaign on the
parallel Rust runner, and converts to and from a specification. The functions and
objects of the first part work on NumPy arrays; a block works on a :class:`Frame` and
calls the same Rust code. The guide is :doc:`../user_guide/chains`.


Chains
------

.. autosummary::
   :toctree: generated

   Chain
   Block
   Native
   Var
   Frame
   LinkBlock
   frame_metrics

``var`` (the function that makes a :class:`Var`) is documented here rather than on a
page of its own, whose file name would clash with that of :class:`Var` on
case-insensitive file systems:

.. autosummary::

   var

.. autofunction:: var

Monte Carlo runner
------------------

.. autosummary::
   :toctree: generated

   sim

Declarative campaigns
---------------------

.. autosummary::
   :toctree: generated

   run_spec
   verify_spec
   search_spec
   spec_conditions
   spec_blocks
   spec_schema
   spec_check_link
   Tolerance

The blocks, by role
-------------------

.. automodule:: comnum.blocks
   :no-members:

The blocks in the order of a link; ``needs`` and ``gives`` of each class are the facts
of the frame it reads and adds (:func:`spec_check_link` checks a link with them), and
``role`` its place in the link.

.. the blocks by role, written by tools/gen_blocks.py: do not edit below

.. currentmodule:: comnum.blocks

Sources
~~~~~~~

.. autosummary::
   :toctree: generated

   Source
   Capture
   TextSource

Coding and mapping
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   PasEncode
   PasDecode
   FecEncode
   Map
   Demap
   FecDecode

Waveforms
~~~~~~~~~

.. autosummary::
   :toctree: generated

   PulseShape
   CeOfdmTx
   CeOfdmRx
   CpmTx
   CpmRx
   FskTx
   FskRx
   WdmTx
   WdmRx
   MimoTx
   OfdmTx
   OfdmRx
   MimoRx

Wireless channels
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Awgn
   AwgnEbn0
   Fir
   Rayleigh
   FlatFading
   Tdl
   Cdl

Optical channels
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Fiber
   Ssfm
   Edfa
   Manakov
   Pmd

Hardware impairments
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   PhaseNoise
   Cfo
   ClockOffset
   RxIq
   TxIq
   Adc
   Dac

Receiver DSP
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   ReferenceSync
   Cdc
   Dbp
   Normalize
   Resample
   MatchedFilter
   Decimate
   Gardner
   AdaptiveEqualizer
   Foe
   Bps
   Viterbi
   Ddpll
   FrameSync
   ManakovDbp

Measurements and analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Probe
   Dump
   Scope
   NoiseEstimate
   Classify

Systems and standards
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Dvbs2Tx
   Dvbs2Rx

.. currentmodule:: comnum

.. end of the blocks by role
