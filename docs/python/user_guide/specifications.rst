Declarative campaigns
=====================

A specification describes a whole campaign — the grid, the link as a list of blocks,
the metrics, the stopping rule, the outputs and the expected results — in a YAML,
JSON or TOML file (or a Python ``dict``). :func:`comnum.run_spec` runs it on the
parallel Rust runner; the same file runs from the command line with ``comnum run``.

.. code-block:: python

   import comnum

   results = comnum.run_spec("""
   name: qam_awgn
   seed: 2024
   grid:
     m: [4, 16, 64]
     ebn0_db: {from: 0, to: 20, step: 2}
   link:
     - {type: source, constellation: {kind: qam, m: $m}, n_symbols: 4096}
     - {type: awgn_ebn0, ebn0_db: $ebn0_db}
   metrics: {report: [ber, ser]}
   stop: {min_errors: {ber: 200}, min_trials: 128, max_trials: 4000}
   callbacks:
     skip_below: {metric: ber, threshold: 1e-6, along: ebn0_db}
   """)
   results[0]["condition"], results[0]["metrics"]["ber"]

Every result is a ``dict`` with the condition, the status, the number of trials and
every metric with its value and confidence interval (``value``, ``lo``, ``hi``).

- ``profile: true`` (or ``run_spec(spec, profile=True)``, ``comnum run --profile``)
  times every block, reported as the metrics ``<label>.time_ms`` (see
  :doc:`analysis`);
- ``$name`` in a block is the value of the grid variable ``name``;
- :func:`comnum.spec_blocks` lists the blocks of the links and their parameters, with
  units and defaults; :func:`comnum.spec_conditions` the conditions of a grid;
- the JSON report records the library version, its commit, the seed and the
  specification itself, so that every number can be traced back to what produced it.

Expected results
----------------

An ``expect`` section lists results the campaign must reproduce — a value, a published
curve, or a closed form — with its source and a tolerance. :func:`comnum.verify_spec`
runs the campaign and checks them:

.. code-block:: yaml

   expect:
     - name: SER of square QAM (exact)
       source: Proakis & Salehi, Digital Communications, 5th ed., Sec. 4.3
       metric: ser
       theory: {function: ser, kind: qam, m: $m, ebn0_db: $ebn0_db}
       interval: 0.999          # the closed form lies in the 99.9 % interval

The specifications of ``specs/validation/`` reproduce Proakis' error rates, the
information rates of QAM by quadrature and figures of the literature; they run in CI.
The closed forms available to ``theory`` are the error rates over AWGN (``ber``,
``ser``), the information rates (``mi``, ``gmi``), the BER of BPSK and QPSK with
maximal-ratio combining of Rayleigh branches (``ber_rayleigh``, with ``branches`` and
``offset_db``) and the SNR of a WDM link by the GN model (``gn_snr_db``): see
``specs/mimo_diversity.yaml``, ``specs/ofdm_eva.yaml`` and ``specs/wdm_gn.yaml``.

Operating points
----------------

A ``search`` section replaces the sweep by the search of the value of a variable at
which a metric reaches a target — the :math:`E_b/N_0` required for a BER of
:math:`10^{-3}`, say — for every combination of the other axes
(:func:`comnum.search_spec`).

The specifications of the repository
------------------------------------

``specs/tutorial/`` holds one-trial specifications that look at the signals along the
chain (:ref:`getting-started-signals`), ``specs/`` the campaigns of the examples and of
the user guide, and ``specs/validation/`` those whose expected results come from the
literature (every
one has a :doc:`validation card <../validation>`; CI verifies the short ones):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Specification
     - Content
   * - ``qam_awgn.yaml``
     - BER of QAM over AWGN, the example ``ber_awgn``
   * - ``coherent_link.yaml``
     - the coherent optical link and its DSP chain, the example ``coherent_link``
   * - ``ssfm_dbp.yaml``
     - launch-power sweep over 10 × 80 km, dispersion compensation against
       back-propagation, the example ``ssfm_dbp``
   * - ``capture_16qam.yaml``
     - the receiver DSP chain on a recorded capture (:doc:`recordings`)
   * - ``mimo_diversity.yaml``
     - Alamouti and zero forcing over 2 × 2 Rayleigh fading, checked against
       maximal-ratio combining
   * - ``ofdm_eva.yaml``
     - OFDM over EVA with known, LMMSE and LS channels, checked against Rayleigh fading
       per subcarrier
   * - ``bicm_ldpc.yaml``
     - LDPC (1024, 512) BICM with 16-QAM over AWGN: BER, coded BER and BLER
   * - ``shaping_awgn.yaml``, ``pas_64qam.yaml``
     - MI and GMI of Maxwell–Boltzmann 64-QAM, and probabilistic amplitude shaping
       (CCDM, LDPC of rate 2/3) at 3 bits per symbol, checked against the closed forms
       with the priors
   * - ``wdm_gn.yaml``, ``raman_wdm.yaml``, ``wdm_dp16qam.yaml``
     - five DP channels over 5 × 80 km by the Manakov equation against the GN and EGN
       models, the same with a counter-propagating Raman pump, and the practical
       receiver (BPS) against the ASE-limited SNR
   * - ``polmux_cma.yaml``, ``polmux_cma_45deg.yaml``
     - blind 2 × 2 polarisation demultiplexing (CMA then RDE) under rotation and PMD,
       with and without Kikuchi's initialisation against the singularity
   * - ``validation/proakis_awgn.yaml``, ``validation/required_snr.yaml``
     - the error rates of QAM, PSK and PAM over AWGN (47 checks), and the Eb/N0 of a
       BER of 10⁻³ by search against the inverted closed forms (within 0.02 dB)
   * - ``validation/information_rates.yaml``
     - MI and GMI by Gauss–Hermite quadrature (Alvarado et al. 2018), within 0.02 bit
   * - ``validation/hager2021_fig5.yaml``
     - Häger & Pfister 2021, Fig. 5: linear equalisation and DBP over 25 × 80 km,
       within 0.20 dB at the 25 points of the figure
   * - ``validation/msk_bpsk.yaml``, ``validation/thompson2008_eq15.yaml``
     - precoded MSK with maximum-likelihood detection has the BER of BPSK; the BER of
       CE-OFDM in the linear regime (Thompson et al. 2008, eq. (15))
   * - ``validation/dvbs2_table13.yaml``
     - ETSI EN 302 307-1, Table 13: the Es/N0 of the 28 MODCODs at a FER of 10 %,
       all within [threshold − 0.8 dB, threshold]

Every one runs from the command line (``comnum run``, ``comnum verify``) and from
Python (:func:`comnum.run_spec`, :func:`comnum.verify_spec`).
