# Changelog

All notable changes to comnum are documented here, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crates share one version
and follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The Raman solver takes the attenuation and the effective area of the fibre against
  frequency, `Raman::alpha_spectrum_db_km` and `Raman::aeff_spectrum_um2` (tables of
  `(Hz, value)` points, linear between them and constant beyond; in a specification and
  in Python the `raman` keys `alpha_spectrum: {frequency_thz, alpha_db_km}` and
  `aeff_spectrum: {frequency_thz, aeff_um2}`), and `Raman::alpha_db_km_at` and
  `Raman::aeff_um2_at`. The attenuation applies to the signals and to the pumps without
  their own `alpha_db_km`, and two waves couple through the mean of their effective areas
  (exact for Gaussian modes; GNPy's `effective_area_overlap`). The power profiles carry
  them to `ssfm`, `manakov`, the generalised GN model and `design_pumps`. Compared with
  GNPy 3.0.1 on an S+C+L span (161 channels from 1490 to 1620 nm, a co-pump and four
  counter-pumps, GNPy's attenuation table and effective areas; recorded in
  `tests/oracle/raman_profile_scl_*`, GNPy's Euler steps of 0.5 and 0.25 m extrapolated
  to a zero step): net gains and pumps within 2e-6 dB, spontaneous noise within 3e-5 dB
  (GNPy alone at 0.25 m: 3e-4 dB on the gains, 8e-4 dB on the pumps); the photon flux of a lossless S+C+L span is conserved within 1e-9 (co-pumps) and
  1e-5 (counter-pumps). Empty tables, the default, change no result.
- `Ofdm` takes optional complex weights of its active subcarriers, `weights` (Rust:
  `Option<Vec<C64>>`, Python: `Ofdm(..., weights=w)` and the property `Ofdm.weights`), for
  power loading or a precoder per subcarrier: the transmitter sends $w_{k} X_{k}$
  (`Ofdm::to_time`), the receiver divides by $w_{k}$ (`Ofdm::to_active`, 0 on a subcarrier of
  zero weight), the power of the samples is the mean of $|w_{k}|^{2}$. The default, `None`,
  changes no result; a struct literal that moved a configuration with `..ofdm` and used it
  afterwards now needs `..ofdm.clone()`.
- `ofdm_rx` takes `estimation: none`: no equaliser, the channel taken as the identity, so
  that the detector sees the subcarriers as received, $Y_{k} = H_{k} X_{k} + W_{k}$; and
  `tdl` and `rayleigh` take a `seed`, which draws the channel from its own stream, the same
  realisation in every condition and trial, to compare receivers on one channel. The
  specification `specs/tutorial/02_ofdm.yaml` looks at one trial of 16-QAM OFDM over the ETU
  channel along the chain, without an equaliser (a symbol error rate of 0.91), with zero
  forcing on the known channel (an EVM of 8.6e-4, no error out of 2400 symbols) and on
  the channel estimated by LMMSE (1.1e-3); `specs/tutorial/03_mimo.yaml` looks at 4 x 4
  spatial multiplexing of 16-QAM over one Rayleigh matrix with the detectors mr, zf,
  mmse, zf_sic and sphere (symbol error rates of 0.76, 0.068, 0.053, 0.0028 and 0 at
  18 dB); the playground has a selector of the traced condition above the scope tabs.
  These one-trial specifications are grouped with `01_awgn` in `specs/tutorial/`, with nine
  more of the figures of a course in digital communications: `04_nyquist` (the raised
  cosine, roll-offs from 0.05 to 1), `05_constellations` (BPSK to 64-QAM at the same
  Eb/N0), `06_carrier_sync` (frequency offset and phase noise, foe and bps),
  `07_timing_sync` (a clock 100 ppm slow, with and without the Gardner loop),
  `08_equalisation` (three echoes, CMA then DD-LMS), `09_diversity` (Rayleigh fading
  on one, two and four antennas), `10_iq_imbalance`, `11_adc` (1 to 6 bits) and
  `12_cpm` (MSK and GMSK); they are first in the explorer of the playground and listed
  in Getting started ("Seeing the signals along the chain").
- A convergence card for the split step, `hager2021_convergence` (CI): on the link of
  Fig. 8 of Häger & Pfister (2021) cut to two spans, every technique of `Ssfm` (the
  midpoint phase, the effective length, the trapezoidal mean, Bosco's logarithmic steps,
  single precision) against the solver's own solution with steps of 25 m, which halving
  the step changes by 1.2e-14 (`validation/tests/fiber/hager2021.rs`). Every scheme is of
  order 2 (2.01 to 2.10) and tends to the same field; the paper's 500 logarithmic steps
  per span leave an error 66 dB below the noise of the amplifiers in the band of the
  channel; logarithmic steps have 20 dB more field error than uniform ones at 100 steps
  per span. The gallery example `plot_08_split_step_techniques` takes 500 uniform steps
  per span as the reference on the full link of Fig. 8 (the specification then follows
  the published curves within 0.07 dB up to 6 dBm) and gives the effective SNR that every
  technique loses against it with the DBP receiver at 6 dBm, with its computation time:
  uniform steps lose 0.01 dB with about 22 steps per span (12 ms), the logarithmic steps
  need about 45, the trapezoidal mean about 50 at four times the time per step; the user
  guide (optical, "How many steps") gives the table.
- Validation of the multi-pump Raman solver against Kidorf, Rottwitt, Nissov, Ma &
  Rabarijaona (*IEEE Photon. Technol. Lett.* 11(5), 1999): eight counter-pumps from 1416
  to 1502 nm in 45 km of pure silica core fibre for 100 signals from 1520 to 1619 nm
  (`validation/tests/fiber/kidorf1999.rs`, cards `kidorf1999_fig5`, `kidorf1999_fig4`,
  `kidorf1999_fig2`), with the gain spectrum of its Fig. 1 read and extended beyond
  19.1 THz, and the loss spectrum, which the paper does not give, assumed. The
  pump-to-pump transfer of Fig. 5 is reproduced (the 1502 nm pump ends 20.70 dB above the
  1416 nm one against 20.99 read), but every far-end pump power is 0.49 to 1.05 dB below
  the figure; the optical SNR in 100 GHz is 34.87 and 39.87 dB against 34.6 and 39.1;
  the ripple is 1.46 dB against 1.1 (Fig. 4) and 12.57 against 10.5 dB for evenly spaced
  pumps (Fig. 2). These three gaps are not explained (badge orange).
- Validation of the two nonlinear steps against Table 2 of Shao, Liang & Kumar (*IEEE
  Photonics J.* 6(4), 2014; the reference of Optilux's step): the error of a single step
  of 0.5 to 60 km on 32-QAM at 10 and 25 Gbaud (`validation/tests/fiber/shao2014.rs`,
  card `shao2014_table2`, campaign). The midpoint step reproduces the paper's scheme I
  within 12 % at 10 Gbaud and a factor 2.7 at 25 Gbaud, with its order in the step; the
  effective-length step, eq. (21) of the paper as written, has 2 to 4.9 times less error
  than the paper's scheme II, which is not explained (badge orange). The
  minimum-area-mismatch step distribution of its Table 1 is not implemented.
- `RamanProfile::ase_backward_psd`, the spontaneous Raman noise emitted towards the
  input of the span (Bromage 2004, eq. (6) along $-z$), returned by
  `comnum.raman_profile`; the mirror image of the forward noise of the reversed span
  (unit test). The conventions of the Raman module (the depolarised gain, the noise in
  both directions, the noise figures, the double Rayleigh backscattering) and the
  convergence of its solver (second order in the step: $5 \cdot 10^{-3}$, $1.3 \cdot
  10^{-3}$, $3.2 \cdot 10^{-4}$, $7.9 \cdot 10^{-5}$ and $1.9 \cdot 10^{-5}$ dB at 0.8,
  0.4, 0.2, 0.1 and 0.05 km on a span with three 500 mW counter-pumps and 40 channels)
  are written in the fibre validation guide (section 4).
- The pump-power optimiser `Raman::design_pumps` (`comnum_channel::optical::raman::design`):
  the launch powers of pumps at given wavelengths and directions that bring the on-off
  gain of the signals closest to a target per signal, by the two stages of Perlin &
  Winful (*J. Lightw. Technol.* 20(2), 2002) at fixed wavelengths (Hu, Marks & Menyuk,
  *J. Lightw. Technol.* 22(6), 2004): the integrated pump powers that fit the targets
  (`DesignObjective::LeastSquares`, non-negative least squares, or `Minimax`, the
  largest gain error, the flatness of both papers), then the launch powers whose
  solved profiles have these integrals (Newton's method on the logarithms of the powers
  with a finite-difference Jacobian), alternated until the integrals match within
  `DESIGN_TOLERANCE`. `PumpDesign` returns the pumps, the target integrals, the profile
  and the ripple (the peak-to-peak spread of the on-off gain minus its target).
  `comnum.raman_design` is the Python function. `comnum_math::linalg` gains `nnls`
  (Lawson & Hanson 1974, ch. 23), `nnls_minimax` (Lawson's iteratively reweighted least
  squares; Rice & Usow 1968) and `lstsq`. Validated against Hu et al. 2004, Table I,
  case I (`validation/cards/hu2004_table1.yaml`, the gain and loss profiles of its Fig. 1
  read from the rasters): 8 of the 9 ripples of its case I within 0.1 dB plus half the value (0.29, 1.35, 0.045, 0.43, 1.82, 0.021, 0.10 and 0.37 dB against 0.180, 1.015, 0.019, 0.435, 1.167, 0.008, 0.080 and 0.255), 2 pumps over 80 nm 5.7 dB against 2.77, not explained; and against Perlin & Winful 2002, Fig. 2
  (`validation/cards/perlin2002_fig2.yaml`), the 4- and 8-pump designs at the pump
  wavelengths read on the figure: the 4-pump design within 0.85 dB on every launch power (with the polarisation factor 1, not given by the paper), 503 mW of pump against about 530 and a ripple of 0.86 dB against 0.90; the 8-pump design 0.48 dB against 0.22, its three long-wavelength pumps ill-conditioned at the ±1 nm of the reading.
- `RamanProfile::noise_figure_db` (Bromage 2004, eq. (9): signal-spontaneous beating
  and shot noise from the spontaneous noise of the solver), `effective_noise_figure_db`
  (eq. (12), the noise figure minus the loss of the span, negative for a pumped span),
  `span_loss_db` and `integral_w_km` (the integrated power of a wave, the quantity of the
  gain identity of Hu, Marks & Menyuk 2004, eq. (2)); `comnum.raman_profile` returns
  `noise_figure_db` and `effective_noise_figure_db`. The Bromage and Hu tests reuse
  them. Validated against Islam, *IEEE J. Sel. Topics Quantum Electron.* 8(3), 2002:
  Sec. II-A (`validation/cards/islam2002_sec2a.yaml`): a lossless undepleted amplifier
  has the noise figure $2 - 1/G$ of a fully inverted amplifier within 10⁻³ dB (with the
  thermal factor at 300 K, $(1 + \eta) 2 (G - 1)/G + 1/G$); Fig. 3
  (`validation/cards/islam2002_fig3.yaml`): the output power of a depleted single-pump
  DCF amplifier (20 mW of signal, 0.1 to 1.15 W of pump, 13.5 dB/(W·km) over 5.64 km;
  counter-propagating and 0.63 dB/km of pump loss assumed, the paper gives neither)
  within 1 mW at 0.1 and 0.2 W and 3 to 5.6 % above the reading beyond, the conversion
  efficiency within 2.5 points of what the paper's panel (a) implies (its panel (b)
  reads 8 to 10 points lower at 0.1 and 0.2 W, inconsistent with (a)); a co-propagating
  pump saturates at 47 % of conversion against the 59 % of the figure.
- `Raman::solve` converges on strongly depleted amplifiers (a 1 W counter-pump on a
  20 mW signal in 5.64 km of DCF, 52 dB of small-signal gain): when the sweeps from the
  loss-only guess do not converge, it continues from a fraction of the pump powers
  (halved until the sweeps converge, then doubled with the previous solution as the
  guess). Converged results are unchanged (the sweeps are the same).
- Double Rayleigh backscattering of the Raman-pumped spans: `Raman::rayleigh_db_km`
  (the Rayleigh backscatter coupling of the signals, $10 \log_{10}$ of $\kappa = \alpha_{R}
  S$ in 1/km; `None`, the default, leaves it out) and, on every profile, the effective
  Rayleigh reflection coefficient `RamanProfile::double_rayleigh` (and
  `double_rayleigh_at` for another coupling), the closed form of Bromage 2004, eqs.
  (13)–(15), on the solved gain profile: $\kappa^{2} \int_{0}^{L} G^{-2}(z) \int_{z}^{L}
  G^{2}(z')\,dz'\,dz$. The links count it as a noise, the co-polarised fraction 5/9 of
  $R^{2} P$ (Bromage 2004, eq. (16); `MPI_COPOLARISED_FRACTION`): `Ssfm` and
  `ManakovSsfm` add the multiple-path interference of every slice with the
  spontaneous Raman noise (`RamanProfile::mpi_psd`), `GnModel::snr` divides by it too
  (`GnModel::mpi_power`). The `raman` mapping of the specifications and of Python gains
  the key `rayleigh_db_km`, `comnum.raman_profile` the output `double_rayleigh`.
  Validated against Islam, *IEEE J. Sel. Topics Quantum Electron.* 8(3), 2002, Fig.
  6(b) (`validation/cards/islam2002_fig6b.yaml`): the coefficient against the on-off
  gain of SMF-28 of 80 and 100 km and of two dispersion-compensating fibres within the
  reading of 1.5 dB at 16 of the 20 points, 1.6 to 3 dB off at four (not explained: the
  paper's model is not written). `Raman::coefficient` (the signed coefficient
  $C_{kj}$ of the power equations) and `RamanGain::flat` (a constant efficiency, the
  single number of the data sheets) are public. No existing result changes: without
  `rayleigh_db_km` the noise is zero.
- `comnum_math::nlse::NonlinearStep::EffectiveLength`, a third nonlinear step of the
  split-step engine next to `Midpoint` (the default, unchanged) and `Trapezoidal`:
  Optilux's scheme (Shao, Liang & Kumar, *IEEE Photonics J.* 6(4), 2014, cited by its
  `fiber`), the phase of the power at the start of the nonlinear block over the
  effective length `(1 - e^{-αh})/α` of the step, the attenuation of the whole step
  after it. The scalar `Ssfm` gains the field `nonlinear` (double precision only for
  the new step and the trapezoidal one), `comnum.Ssfm` and `comnum.ManakovSsfm` the
  string `nonlinear_step="effective_length"`, the blocks `ssfm` and `manakov` the same
  value. With it comnum reproduces Optilux's fields to machine precision
  (`tests/oracle/test_optilux_oracle.py`: NMSE 4·10⁻²⁸ after one span and 3·10⁻²⁶
  after 10 × 80 km at 1 km steps, against 1.2·10⁻¹⁰ and 1.5·10⁻⁸ with the midpoint
  step). On that field (16-QAM at 32 GBd, 4 dBm, 10 × 80 km) the NMSE against the
  step-converged solution (comnum at 0.05 km) makes it the most accurate of the three
  schemes at every step, 2.8 to 3.5 times below the midpoint step and 12 to 39 times
  below the trapezoidal one: 3.8·10⁻¹¹, 6.7·10⁻¹⁰, 1.1·10⁻⁸, 2.3·10⁻⁷ and 3.7·10⁻⁶ at 0.25,
  0.5, 1, 2 and 4 km (midpoint 1.3·10⁻¹⁰, 2.3·10⁻⁹, 3.7·10⁻⁸, 6.4·10⁻⁷ and 1.0·10⁻⁵;
  trapezoidal 1.5·10⁻⁹, 2.3·10⁻⁸, 3.5·10⁻⁷, 4.6·10⁻⁶ and 4.4·10⁻⁵). The default stays
  the midpoint step: no published result changes.
- Optilux's Manakov propagation recorded: `tools/record_optilux.py --mode manakov`
  runs its `fiber` in the unique-field mode with `ismanakov` (no PMD, no birefringence)
  on a dual-polarisation WDM field generated by comnum (five 16-QAM channels at 32 GBd
  spaced by 50 GHz, 2 dBm per channel, seed 1) over the same link, into
  `tests/oracle/optilux_manakov.{json,npz}`, compared by
  `tests/oracle/test_optilux_manakov_oracle.py`: the linear span to machine precision;
  with the effective-length step the Kerr spans too (NMSE 6·10⁻²⁷ after one span,
  7·10⁻²⁵ after ten: Optilux's dual-polarisation operator is comnum's Manakov rotation);
  with the midpoint step 6·10⁻¹⁰ and 7·10⁻⁸, the weighting of the attenuation within the
  step only. On this WDM field the step error (3.3·10⁻³ at 1 km against the solution
  at 0.05 km) comes from the walk-off of the channels, and the two weightings are within
  0.1 % of each other from 0.25 to 4 km; OptiCommPy's trapezoidal step is 0.39 from
  Optilux after ten spans. The recorder takes the sampling rate and the number of
  polarisations from the field; the scalar fixture re-recorded by the new script is
  unchanged (its `.json` names the equation and the layout).
- Playground: a **Share** button puts the specification of the editor, compressed, in
  the fragment of the address (`#yaml=…`); the link opens it in the editor, to run on
  **Run**, with nothing stored on a server.
- Block `scope` (`comnum-spec`): a view of the signal at a point of the link
  (constellation, spectrum, eye or time) taken in one traced trial and nothing in a
  campaign; `Spec::trace(condition, trial)` runs that trial of a campaign (the same
  random stream) and returns the signals of its scopes (`ScopeTrace`, `Frame::traces`), the
  spectrum computed by `comnum_dsp::spectrum::psd`. The playground draws one tab per
  scope and the chain of blocks; `specs/tutorial/01_awgn.yaml` is a one-trial example of
  16-QAM over AWGN.
- Browser playground (`playground/`): a static page where a specification is edited
  (CodeMirror) and run in the browser, the Rust engine compiled to WebAssembly by
  `wasm-pack` from the new crate `bindings/comnum-wasm` (not published), laid out as a
  workbench (an explorer of the specifications of `specs/` and `specs/validation/`, the
  editor and the plot side by side, one tab per curve when the grid gives several, a
  terminal panel, a status bar, light and dark, the
  colours of the logo), a Run button, the curves plotted
  (Plotly, log scale for the error rates, confidence intervals) as the conditions
  complete in a Web Worker, the published points and closed forms of the expectations
  drawn over the simulated curve, a terminal panel with the lines of `comnum run` and
  `comnum verify` as the conditions complete, the errors of the specification marked
  in the editor, and a probe that tells when the browser has installed its optimised
  code for the module (V8 runs a fresh module in its baseline tier, about five times
  slower, for some twenty seconds). `just playground` builds the
  module, `just playground-serve` serves the page, and the CI job `docs-python`
  publishes it under `/playground/` of the documentation site on every push to `main`. The engine is the native one on one
  thread: the runner of `comnum-sim` runs the trials of a batch one after the other on
  `wasm32-unknown-unknown` (no thread there), with the records merged in trial order as
  on every target, and the new module `comnum_sim::clock` (`Stopwatch`, `unix_time_s`,
  `set_clock`) replaces `std::time::Instant`, which has no clock in the browser, in the
  runner, the progress bar, the profile of a link and the search. No result changes;
  the `qam_awgn` example takes 3.6 s in Chrome against 2.0 s natively on one core.
  `comnum_spec::verify::Expectation` gains `name`, `source`, `metric` and
  `expected_on`, the expected value on a condition before any run.
- Library benchmark on a GPU: OptiCommPy's CuPy split-step (`optic.models.modelsGPU.ssfm`)
  runs in the mode `gpu` of scenario 5, in complex128 and complex64, next to Sionna on
  the device and comnum on the CPU; a scenario names its GPU-capable libraries in
  `GPU_LIBS`. `runpod.sh` installs OptiCommPy and the CuPy wheel of the image's CUDA
  version. Results on the laptop unchanged. The scenarios 2 and 3 name the Rayleigh
  block's `coherence_samples` (the field was renamed after the recorded run), and the
  GPU mode gives Sionna the device as `cuda:0`, the form its `config.device` accepts.
  Inside a container the thread pools of the modes `all` and `gpu` are sized to the CPU
  quota (cgroup `cpu.max`, the affinity mask) instead of the host's core count, which
  oversubscribed a RunPod pod; the quota is recorded as `cpus_available`.
  Sionna's `config.device` is set to `cpu` in the modes `1` and `all` (its own default
  is the first CUDA device when there is one: on the pod the CPU modes ran on the GPU,
  0.7 ms per 10⁶ bits) and to `cuda:0` in the mode `gpu`, without
  `torch.set_default_device`, whose device context slowed every operation.
  The results of two RunPod pods (`results/rtx4090/`, `results/h100/`) are recorded,
  and the benchmark page becomes an overview, `docs/python/benchmarks.rst` (the
  machines with their provenance, the CPU comparison with charts, then what a GPU
  changes, the limits), generated with its SVG charts by `tools/benchmarks_page.py`
  from the headline cases; every number moves to `docs/python/benchmarks_details.rst`.
- Optilux (Apr 2021, GPL-3.0, under Octave) as a recorded oracle of the split-step:
  `tools/record_optilux.py` runs its `fiber` and `ampliflat` (`tools/record_optilux.m`)
  on the field of the benchmark scenario and `tests/oracle/test_optilux_oracle.py`
  compares: machine precision without the Kerr effect, NMSE 1.2·10⁻¹⁰ after one span
  and 1.5·10⁻⁸ after 10 × 80 km at 1 km steps (Optilux's nonlinear step uses the
  effective length of the step), both solvers within 4·10⁻⁸ of the solution at 0.05 km.
  Two validation cards on Optilux, run by `validation/tests/fiber/optilux.rs` from the
  recordings of `validation/optilux/` (`tools/record_optilux.py`): `optilux_ssfm`, the
  same three cases on a field of 2⁸ symbols (NMSE 2·10⁻²⁸, 1.2·10⁻¹⁰, 1.4·10⁻⁸), and
  `optilux_ase_link`, the amplified link of its example 21 (`tools/record_optilux_ase.m`:
  PDM QPSK through 50 spans of 150 km with amplifiers of 30 dB gain and 6 dB noise
  figure), the SNR of comnum's chain within 0.15 dB of Optilux's estimates and 0.03 dB
  of the example's formula, the BER within 3 % of the formula; the example re-draws the
  noise of the last span only, so its own BER estimates are biased (noted on the card).

## [0.3.1] — 2026-11-06

### Fixed

- `validation/cards/goldsmith1997_adaptive_mqam.yaml` wrote the target BER of the two
  Fig. 10 cases as `1e-06`, which YAML 1.1 reads as a string: `tests/test_cards.py`
  failed in CI. The value is `1.0e-6`, unchanged.

## [0.3.0] — 2026-11-06

Eight references reproduced and recorded as validation cards (Gardner 1986, Zheng & Xiao
2003, Goldsmith & Chua 1997, Ding et al. 2014, ten Brink 2001, MacKay 1999, Savory 2010,
Nissel, Schwarz & Rupp 2017), the JSON output and the guide of `comnum` for programs and
agents, and the logarithmic step distributions of the split-step method. No existing
result changes.

### Added

- `comnum_waveform::sir`: the signal-to-interference ratio of any multicarrier scheme
  with a one-tap equaliser over a doubly-selective WSSUS channel with Jakes' Doppler, in
  closed form from its pulses (`DoublySelective::sir_qam`, `sir_oqam`; Nissel, Schwarz &
  Rupp, *IEEE J. Sel. Areas Commun.* 35(8), 2017, eqs. (35)-(40)); `Ofdm::pulse` and
  `Ofdm::window`; `ofdm::budget::jakes_ici_sir`, the exact ICI of CP-OFDM under Jakes'
  fading (eq. (42)) whose small-Doppler limit is `doppler_noise_power`;
  `fbmc::Prototype::Hermite` (Haas & Belfiore 1997, eqs. (10)-(11)) and
  `Prototype::localisation` (eqs. (4)-(5)); `comnum_math::{hyp1f2, hermite_h}`. Cards
  `nissel2017jsac_eq42` (CI: the CP-OFDM points of Figs. 5-8 at 500 km/h within 0.2 dB
  and the labelled spacings to the kHz) and `nissel2017jsac_figs5_8` (campaign:
  the SIR of OFDM, FBMC-OQAM and FBMC-QAM at the optimal spacing over four channels, 36 of 39 points at 100-500 km/h within 1 dB of Figs. 5-8). `nissel2017_fig4` gains the FBMC curve of Nissel & Rupp with the Hermite prototype (8.56e-3 against 8.5e-3 at 250 km/h).

- `comnum_fec::exit`: EXIT charts of parallel concatenated codes (ten Brink, *IEEE
  Trans. Commun.* 49(10), 2001): the J function and its inverse, the mutual information
  of LLRs, the transfer curve of a recursive systematic code measured with the BCJR
  (`ExitCurve`) and the pinch-off limit by bisection (`pinch_off_db`). Card
  `tenbrink2001_pinch_off` (campaign): Table IV within 0.02 dB on the diagonal and the
  pairs with C2, C2/C1 1.61 against 1.49 dB (the end of the chart is not given).

- `comnum_fec::density_evolution::correction_effect_threshold`: the zero of the
  correction effect of the simple decoder (MacKay, *IEEE Trans. Inf. Theory* 45(2), 1999,
  eq. (57)). Card `mackay1999` (campaign): Fig. 6(b) within 1.6 %, the BP transitions of
  Fig. 6, and the BER of the (504, 252) and (1008, 504) codes of Fig. 11 within a factor
  1.4 with PEG codes.

- `comnum_dsp::compensation::iq`: the quadrature imbalance of a coherent front end
  removed by Gram-Schmidt (`gram_schmidt`) or Löwdin (`lowdin`) orthonormalisation,
  and `iq_correlation` (Savory, *IEEE J. Sel. Topics Quantum Electron.* 16(5), 2010, eqs.
  (6)-(8)). Card `savory2010_eqs6_8` (CI): the correlation sin 2 theta, Gram-Schmidt
  leaving the signal turned by theta, Löwdin restoring it (Fig. 3).

- `comnum_mimo::noma`: downlink NOMA with users dropped at random in a disc (Ding, Yang,
  Fan & Poor, *IEEE Signal Process. Lett.* 21(12), 2014): the gain CDF (eq. (3)) and
  its Gauss-Chebyshev form (4), the SIC thresholds, the exact outage of the ordered
  users (eq. (9)) and its high-SNR form (11), the orthogonal random user; and
  `comnum_math::gauss_chebyshev`. Card `ding2014_outage`: the table of Fig. 3 (eq. (11)
  to four digits) and a Monte Carlo with SIC within 0.03 decade of Fig. 1(a).

- `comnum_mapping::link_adaptation`: variable-rate variable-power MQAM over Rayleigh
  fading (Goldsmith & Chua, *IEEE Trans. Commun.* 45(10), 1997): the BER bound (17) and
  the power gap K, the capacity with water-filling in time (eq. (7)), continuous-rate
  (eq. (25)) and discrete-rate (eqs. (31)-(32)) adaptive MQAM, the average fade region
  durations (eqs. (34)-(35)) and the BER under an SNR estimation error (eq. (42)), in
  closed form with exponential integrals. Card `goldsmith1997_adaptive_mqam` (CI): Table
  II within 1.3 %, Figs. 3, 5 and 10 within their reading.

- `comnum_channel::multipath::rayleigh_level_crossing_rate` and
  `rayleigh_average_fade_duration` (Zheng & Xiao, *IEEE Trans. Commun.* 51(6), 2003, eq.
  (19)). Card `zheng2003_statistics`: the sum-of-sinusoids fading of `Tdl` with 8
  sinusoids has the correlations of Theorem 1 (within 0.02 of J0) and the level-crossing
  rate and fade duration of eqs. (19) within 3.5 % from -20 to 0 dB (5 to 6 % at +5 dB,
  within 0.8 % with 64 sinusoids).

- `comnum_dsp::sync::timing::gardner_s_curve` and `gardner_gain`: the S-curve
  $(4/T) G_{R} \sin(2\pi\tau/T)$ and the gain $8\pi G_{R}/T^{2}$ of Gardner's detector for a
  raised-cosine pulse (Gardner, *IEEE Trans. Commun.* 34(5), 1986, eqs. (15) and (29)).
  Card `gardner1986_scurve`: the Monte Carlo mean of `gardner_ted` within 0.003 of the
  S-curve for roll-offs 0.35 to 1, and its independence from the carrier phase (eq. (33)).

- `comnum_fec::density_evolution`: density evolution of regular LDPC ensembles
  (Richardson & Urbanke, *IEEE Trans. Inf. Theory* 47(2), 2001): Gallager's algorithms
  A (eq. (6)) and B (Example 4) and the ternary algorithm E (Example 5) on the BSC
  (`hard_decision_error`, `hard_decision_threshold`), and discretized belief propagation
  on the BSC and the binary-input AWGN channel (`BpDensityEvolution`: the variable node
  by FFT, the check node pair by pair on the LLR grid), with thresholds by bisection.
  Validated against their Tables I and II (cards `richardson2001_table1`, in CI, and
  `richardson2001_bp`, a campaign): the twelve thresholds of belief propagation within
  the rounding of the tables ((3, 6): 0.8808 and 0.0841 against 0.88 and 0.084), 14 of
  the 18 hard-decision thresholds within the rounding of the table, the four others
  within 0.002 (A and B for (3, 6): 0.0395, printed 0.04).

- `comnum --json` for programs and agents (`comnum_spec::cli::agent`): `run`, `check`
  and `verify` print one compact JSON document per specification on standard output
  (the metrics with their confidence intervals, counts and totals; the provenance:
  version, seed, threads, confidence, a fingerprint of the specification and the
  reports written), and the errors as JSON with a kind, a hint and a suggestion for a
  misspelt block or parameter (or the block that takes a parameter given to another).
  `--max-seconds S` bounds the wall time of a run, 300 s by default with `--json`
  (`0` for none), spread over its conditions; a condition it cut short says
  `time_limited`. A specification `-` is read from standard input. Output without
  `--json` is unchanged.
- `comnum guide` and `crates/comnum-spec/GUIDE_AGENTS.md`: the guide for programs and
  agents (the loop blocks, check, run, verify; the format; the JSON; the pitfalls), its
  specifications checked by the tests; `llms.txt` at the root points to it.

- The logarithmic step-size distribution of Bosco et al. (*IEEE Photon. Technol. Lett.*
  12(5), 2000, eq. (6)): `comnum_math::nlse::logarithmic_steps`, the propagation rule
  `StepRule::Logarithmic(n)` (the `ssfm` block and `Ssfm(steps_per_span=)` in double
  precision, the `manakov` block and `ManakovSsfm(step_rule="logarithmic")`), and the
  back-propagation `Dbp::distribution` (`StepDistribution`, the `dbp` block and
  `Ssfm.backpropagate(step_distribution=)`). `SplitStep::with_lengths` runs steps of
  different lengths. Checked by eq. (6) to 1e-10 and by the floor 1/K of the
  fictitious four-wave mixing of eq. (5) and Fig. 3 (mean 0.92/K, where equal steps
  reach peaks of 1). On one span of 80 km at 6 dBm, the back-propagation with 2 and 4
  logarithmic steps leaves 22 % and 28 % less residual than equal steps, at the full
  nonlinear phase; with 16 steps equal steps are four times better. The forward model
  of Fig. 5 of Häger & Pfister (500 logarithmic steps per span) gives the curves of the
  200 equal steps of `specs/validation/hager2021_fig5.yaml` within 0.01 dB. No existing
  result changes; `Ssfm::dbp` no longer claims the fixed steps of `step_km` as the exact
  forward steps when another rule propagated the field.
- The modified logarithmic steps of Zhang, Li & Dong (2013) for the back-propagation:
  `StepDistribution::ModifiedLogarithmic(factor)`, the `dbp` block
  (`step_distribution: modified_logarithmic`, `step_factor`, 0.4 by default) and
  `Ssfm.backpropagate(step_distribution="modified_logarithmic", step_factor=)`; checked
  by Example 1 of Häger & Pfister (2021, Sec. IV-C: 2 steps over 100 km, about 30 and
  70 km). `specs/validation/hager2021_fig8.yaml` (a campaign of 100 s) reproduces Fig. 8
  of Häger & Pfister (32 GBd over 10 × 100 km): linear equalisation within 0.06 dB, DBP
  with 2, 4 and 500 steps per span within 0.15 dB, except the DBP with 4 steps from 5 to
  8 dBm, 0.3 to 1.1 dB above the figure. With uniform or Bosco's steps
  the DBP with 2 steps falls 1.7 to 4.9 dB below the figure: the steps decide the curve.
- `NonlinearLength::Step` of the back-propagation (`comnum_dsp::compensation::dbp`), the
  `dbp` block `nonlinear_length: step` and `Ssfm.backpropagate(nonlinear_length="step")`:
  the nonlinear step takes the power at the middle of the step times the step, the
  convention of the code of Häger & Pfister, instead of the power integrated over the
  step (`effective`, the default, unchanged). With it their Fig. 8 is reproduced: the
  DBP with 4 steps per span is within 0.09 dB up to 6 dBm and 0.24 and 0.33 dB above at
  7 and 8 dBm, against 0.29 to 1.07 dB above from 5 to 8 dBm with the default.

### Fixed

- `tools/publish_crates.sh` stopped at the first rate limit of crates.io instead of
  waiting: the error names a date ("try again after Wed, 04 Nov 2026 21:58:29 GMT"), the
  script looked for a number of seconds, and the `grep` that found nothing ended the
  script under `set -euo pipefail` (the first run of v0.2.0 published five crates). The
  delay is read from the date or the seconds, 10 minutes otherwise, and the reading
  cannot fail. The workflow `publish-crates.yml`, started by hand, publishes the crates
  of a tag already pushed with the script of the branch.

## [0.2.0] — 2026-11-04

The first public release: the Rust crates on crates.io, the Python package `comnum` on
PyPI and the command line `comnum`.

### Crates and layers

- Fourteen crates in five layers whose dependencies point downward only
  (`docs/ARCHITECTURE.md`, decisions in `docs/adr/`): `comnum-core` and `comnum-math` (0),
  `comnum-dsp` (1), `comnum-fec`, `comnum-mapping`, `comnum-waveform`, `comnum-mimo`, `comnum-channel`,
  `comnum-hardware`, `comnum-metrics` (2), `comnum-sim` (3), `comnum-spec`, `comnum-systems` and the
  facade `comnum` (4), with the features `optical` and `spec`.
- `comnum-core`: the multi-stream containers (`Bits`, `Llr`, `Symbols`, `Labels`,
  `Signal`, `DualPol`), the verb traits (`Encode`, `Decode`, `Map`, `Demap`, `Modulate`,
  `Demodulate`, `Precode`, `Detect`, `Propagate`, `Impair`), the unit newtypes, one
  error type for every crate, and the generator `SimRng` (xoshiro256\*\*) with one
  stream per `(seed, experiment, condition, trial)`.
- `comnum-math`: special functions, Gauss–Hermite quadrature, FFT plans, complex linear
  algebra (QR, Cholesky, SVD), nearest-point search, the split-step solvers of the NLSE
  and the Manakov equation, GF(2) and GF(2^m), exact integers, the Gray code,
  bisection and golden-section search.
- MSRV 1.85, edition 2024, licence MIT OR Apache-2.0; no `unsafe` in the library.

### Blocks

- Mapping: QAM, PSK, PAM, OOK and custom constellations with Gray labels, decisions,
  exact and max-log LLRs; sources (fair bits, PRBS, uniform, Maxwell–Boltzmann and
  Gaussian symbols); differential M-PSK; the closed forms of the error rates and of
  the information rates (MI, GMI, with priors).
- Coding: CRCs and the Gold scrambler of 5G NR; block, random and QPP interleavers;
  convolutional codes (Viterbi, BCJR) and their puncturing (IEEE 802.11, DVB-S); LTE
  turbo codes with the 188 QPP interleavers of TS 36.212; the LDPC code of 5G NR (both base
  graphs, lifting, filler bits, the circular buffer and the bit interleaver) and its
  polar code (the polar sequence, puncturing, shortening, repetition, parity-check bits,
  the input and coded-bit interleavers), from the tables of TS 38.212; LDPC codes (PEG, repeat-accumulate, alist files; sum-product,
  normalised and offset min-sum, flooding or layered); polar codes (SC, CRC-aided
  SCL); Reed–Solomon codes over any field with erasures; shortened binary BCH codes;
  constant-composition distribution matching; BI-AWGN capacity and the normal
  approximation.
- Waveforms: OFDM and DFT-spread OFDM with resource grids, pilots, LS and LMMSE
  channel estimation, Schmidl–Cox synchronisation and MIMO-OFDM detection; the
  distribution of the PAPR and its reduction (clipping and filtering, SLM, PTS, tone
  reservation, ACE, companding); constant-envelope OFDM (Thompson et al. 2008); 5G NR
  (and LTE) numerologies and slot modulation, the synchronisation signal block (PSS,
  SSS, PBCH DMRS), the cell search and the DMRS of the PDSCH over a slot, checked
  against py3gpp;
  continuous-phase modulation (CPFSK, MSK, GMSK, raised-cosine pulses) with sequence,
  quadrature and differential detection.
- MIMO: ZF, MMSE, SIC, ML, sphere and K-best detection with soft outputs; message
  passing for large systems, LAMA (Jeon et al. 2018) and OAMP (Ma & Ping 2017), with
  their state evolution and the recovery thresholds of LAMA (detectors `amp` and
  `oamp`); the receiver of a link, orthogonal space-time block codes, eigenmode and zero-forcing precoding,
  capacity.
- Channels: AWGN; flat Rayleigh and Rician fading, SISO and MIMO, with Kronecker
  correlation; tapped delay lines (EPA, EVA, ETU, ITU-R M.1225) with Doppler; the CDL
  profiles of 3GPP TR 38.901 between antenna arrays; optical fibre (linear, split-step
  Fourier in double or single precision, Manakov with PMD), EDFA, distributed Raman
  amplification and inter-channel stimulated Raman scattering, the GN and EGN models
  (their integration grid follows the width of the band, and the phased-array factor
  of the coherent accumulation is integrated exactly over every cell).
- Blind estimation: fourth-order cumulants, modulation classification by cumulants
  (Swami & Sadler 2000) and by likelihood (Wei & Mendel 2000), the M2M4 SNR (Pauluzzi &
  Beaulieu 2000), the symbol rate by cyclostationarity (Gardner 1991); the `classify`
  block.
- Hardware: power amplifiers (Rapp, Saleh, memory polynomial), Wiener phase noise,
  frequency offset, IQ imbalance; DAC and ADC (uniform quantisation with clipping,
  effective number of bits, clock jitter, bandwidth, zero-order hold) with the closed
  forms of their noise; lasers, phase, Mach–Zehnder and IQ modulators,
  photodiodes with shot and thermal noise, the 90° hybrid, the coherent receiver, the
  WDM transmitter.
- Receiver DSP: pulse shaping and FIR filtering, resampling, chromatic-dispersion
  compensation (exact, frequency sampling, Savory's FIR, the least-squares FIR of Eghbali
  et al. 2014 over the whole band or the band of the signal), digital back-propagation (one and two polarisations), the n×n
  adaptive equaliser (CMA, RDE, NLMS, DD-LMS, Kikuchi's initialisation), carrier
  recovery (fourth-power FOE, Viterbi–Viterbi, blind phase search, DD-PLL), Gardner
  clock recovery, frame synchronisation, digital predistortion.
- Metrics: error counts and rates, SNR, EVM, MI and GMI estimators, the rates of a
  receiver's posterior, PAPR and CCDF, Welch PSD, ACLR.
- Systems: the DVB-S2 physical layer (ETSI EN 302 307-1) from the BBFRAME to the
  scrambled PLFRAME, with its BCH and LDPC codes and the 28 MODCODs; probabilistic
  amplitude shaping (Böcherer, Steiner & Schulte 2015) with the bit mapper of the
  paper.

- The SNR-loss budget of an OFDM receiver of Speth, Fechtel, Fock & Meyr (*IEEE Trans.
  Commun.* 47(11), 1999): `comnum_waveform::ofdm::budget` (`estimator_loss_db`,
  `estimator_gain`, `noise_budget`, `timing_noise_power`, `cfo_noise_power`,
  `doppler_noise_power`, `max_cfo`, `max_doppler`), and the sampling clock offset
  `comnum_hardware::oscillator::{sampling_offset, ClockOffset}` (block `clock_offset`;
  Python `ofdm_budget`, `ofdm_timing_noise`, `clock_offset`, `blocks.ClockOffset`).
  Validated against the paper (`validation/tests/synchronisation/speth1999.rs`, card `speth1999`): Figs.
  2 and 7 within their reading precision, Fig. 3 (AWGN) and eqs. (40), (47) by
  simulation within 0.85 %. No existing result changes.

- The two-dimensional Wiener interpolation of pilots of Hoeher, Kaiser & Robertson
  (*Proc. IEEE ICASSP* 1997): `comnum_waveform::ofdm::wiener` (`Wiener2d` on any pilot
  set, the `n_taps` nearest pilots by their weighted distance, `Correlation2d` with a
  uniform or Jakes Doppler spectrum and a uniform delay spectrum or a
  `PowerDelayProfile`); Python `Wiener2d`. Validated against Figs. 2 and 4 of the paper
  (`validation/tests/synchronisation/hoeher1997.rs`, within 3.6 % on the 2-D filters). The robust
  estimator of Li, Cimini & Sollenberger (*IEEE Trans. Commun.* 46(7), 1998),
  `ofdm::wiener::RobustDesign` (the MSE of eq. (57), the L-tap filter of eqs. (69)–(74);
  Python `RobustDesign`), validated against their Figs. 3 and 4 within 0.26 dB
  (`validation/tests/synchronisation/li1998.rs`): the figures count the taps of the design delay spread
  as floor(K t_max / T_s) + 1 where eq. (56) prints the ceiling. No existing result
  changes.

- The adaptive algorithms share their pieces (`docs/adr/0015-adaptive-error-laws-and-loop-filters.md`):
  `comnum_dsp::adaptive` holds the error laws (`ErrorLaw`: `ConstantModulus`,
  `RadiusDirected`, `Reference`, `Decision`), the update rules (`UpdateRule`: LMS,
  normalised, sign-error) and the loop filters (`LoopFilter`: `ProportionalIntegral`,
  `BilinearPi`). `AdaptiveEqualizer` runs any law with any rule (`run_with`, the field
  `update_rule`, block parameter `update_rule`, Python `update_rule=`); `GardnerLoop` and
  `ddpll` use the loop filters. The outputs of the equaliser, the Gardner loop and the
  DD-PLL are bit-identical before and after (33 arrays on fixed inputs).

- The rank-p LMMSE channel estimator of Edfors et al. (*IEEE Trans. Commun.* 46(7), 1998):
  `comnum_dsp::estimation::{LowRank, ls_noise_factor}` (the estimator of eqs. (8)–(9),
  its MSE with design mismatch, eq. (10), and its floor, eq. (11); beta of a
  constellation) and the channel of random delays of their Sec. II,
  `comnum_waveform::ofdm::estimation::RandomDelayProfile`; Python `LowRank`,
  `ls_noise_factor`, `random_delay_correlation`. Validated against Figs. 4–6 of the
  conference version (`validation/tests/synchronisation/edfors1998.rs`, within 5.9 % on the MSE; the
  synchronized channel needs M = 5 impulses, not given by the paper). No existing result
  changes.

- FM sound broadcasting, for measured signals: `comnum_dsp::fm` (the modulator, the
  polar discriminator, the pre- and de-emphasis of ITU-R BS.450-4, the de-emphasis within
  0.1 dB of the RC curve at 15 kHz); `comnum_systems::fm::mpx` (the stereophonic
  multiplex of the pilot-tone system, BS.450-4 Sec. 2.2: transmitter and stereo decoder,
  40 dB of separation); `comnum_systems::fm::rds` (the RDS of NRSC-4 / EN 50067: the
  biphase symbols and their demodulator by the Gardner loop and the Viterbi–Viterbi phase
  of order 2, the block code with the offsets of Table A.1, the synchronisation, the
  programme service name, the radiotext and the clock time with the characters of Figure
  E.1 and the dates of Annex G). `comnum_dsp::frontend::{resample_rational,
  resampling_filter}`: the filter of `select_channel` for signals already at baseband.
  Python: `fm_modulate`, `fm_discriminator`, `fm_preemphasis`, `fm_deemphasis`,
  `fm_multiplex`, `fm_stereo_decode`, `rds_group_bits`, `rds_differential_encode`,
  `rds_receive`, `resample_rational`. Checked end to end on a simulated station; no
  measured station yet (the indoor capture is at 4 dB of CNR). No existing result
  changes.

- The LTE downlink as a terminal reads it first, for measured signals: the PSS
  (Zadoff–Chu, `comnum_math::sequences::zadoff_chu`), the SSS, the CRS, the PBCH elements,
  transmit diversity on 2 and 4 ports and the grid with its empty DC subcarrier of TS
  36.211 (`comnum_waveform::ofdm::lte`); the BCH of TS 36.212 Sec. 5.3.1 (CRC masked by
  the number of ports, the tail-biting code (133, 171, 165), the rate matching) and the
  fields of the MIB (`comnum_systems::lte::{bch, mib}`); the frames of a cell for
  simulation (`lte::cell::Downlink`); the cell search at 1.92 MS/s (PSS over a grid of
  frequency offsets, the offset refined on the prefixes, SSS) and the MIB of every frame,
  its timing tracked on the PSS against the drift of the sampling clock
  (`lte::search::{cell_search, decode_mib}`). Tail biting in `comnum_fec::conv`
  (`Trellis::encode_tail_biting`, the wrap-around Viterbi algorithm `viterbi_tail_biting`
  of Shao, Lin & Fossorier 2003). A front end for captures, `comnum_dsp::frontend::
  select_channel`: the DC offset removed, the channel shifted to 0 Hz and resampled by the
  exact rational ratio of the two rates under an 80 dB Chebyshev-windowed filter. Python:
  `select_channel`, `zadoff_chu`, `lte_pss`, `lte_frame`, `lte_cell_search`,
  `lte_decode_mib`; the example `lte_mib` reads a SigMF recording (a HackRF capture). On a
  capture of a 20 MHz cell of band 28 (HackRF One at 15.36 MS/s, not committed) the MIB
  of 100 frames of 100 decodes, and the oscillator error measured on the carrier and on
  the frame timing agree (+17.0 and +16.8 ppm). `Crc::crc16` is also the CRC16 of LTE. No
  existing result changes.

- The PDSCH of 5G NR, `comnum_systems::nr`: the modulation mapper of TS 38.211 Sec. 5.1
  (QPSK to 1024-QAM), the MCS tables and the transport block size of TS 38.214 Sec.
  5.1.3, the transport channel of TS 38.212 Sec. 7.2 (TB CRC, code block segmentation,
  LDPC, rate matching per code block) with the soft buffers of a HARQ process, and one
  slot of a single-layer PDSCH with its DM-RS, a precoder per PRG and a receiver that
  estimates the precoded channel by robust 2-D Wiener filtering; `NrLdpc::combine` and
  `NrLdpc::decode_buffer` (soft combining of redundancy versions); `Tdl::draw_at` (the
  taps at given instants). Checked against TS 38.101-4 test 1-2 (TDLC300-100, 2 and 4
  RX): 70 % of the maximum throughput 2.09 and 1.99 dB below the requirements, within the
  range of the four companies of the RAN4 alignment (2018, anonymised in the card), the
  curves of two of them within 0.035 and 0.05 of throughput;
  `Pdsch::receive_known`, the receiver of a known channel; the documentation page `user_guide/nr_conformance` and the gallery example
  `systems/plot_06_nr_pdsch_conformance.py` (the curves recorded in
  `validation/results/ts38101_4_pdsch.json`) (`validation/tests/systems/ts38101_4.rs`, card
  `ts38101_4_pdsch`). No existing result changes
  (`Tdl::draw` is bit-identical).

- Python bindings of the NR PDSCH: `nr_mcs`, `nr_tbs`, `nr_constellation`,
  `NrTransportBlock`, `NrHarqProcess`, `NrPdsch`; the transmitter compared bit for bit
  with Sionna 2.2.0 (`TBEncoder`, three FRCs of TS 38.101-4, both base graphs, the
  scrambling) and py3gpp 0.6.0 (`nrTBS` on 108 allocations, the four redundancy versions,
  `nrPDSCH`), recorded in `tests/oracle/test_nr_pdsch_oracle.py`. py3gpp 0.6.0 is wrong on base
  graph 2 at Z_c = 64 and in `nrPDSCHIndices` (not used, documented).

- The path loss, line-of-sight probability, shadow fading and outdoor-to-indoor
  penetration of 3GPP TR 38.901 §7.4: `comnum_channel::pathloss` (`path_loss_db`,
  `shadow_fading_std_db`, `los_probability`, `uma_environment_height`,
  `building_penetration_db`, `breakpoint_m`) for RMa, UMa, UMi-Street canyon, InH-Office and
  the four indoor-factory sub-scenarios (Tables 7.4.1-1, 7.4.2-1, 7.4.3-1, 7.4.3-2), the
  breakpoints with the 3.0e8 m/s of the TR; Python `path_loss_db`, `shadow_fading_std_db`,
  `los_probability`, `uma_environment_height`, `building_penetration_db`. Equal to Sionna 2.2.0
  at 112 points (recorded, `tests/oracle/test_sionna_pathloss.py`).

- The TDL profiles of 3GPP TR 38.901 (TDL-A to TDL-E, Tables 7.7.2-1 to 7.7.2-5, derived
  from the CDL tables and scaled to a desired RMS delay spread, §7.7.3) and of the
  conformance tests of TS 38.101-4 (TDLA30, TDLB100, TDLC300, TDLD30, Annex B.2.1):
  `comnum_channel::multipath::Profile::{TdlA, …, TdlE, TdlA30, TdlB100, TdlC300, TdlD30}`,
  `Profile::{specular, with_delay_spread, tdl_params}`; the Ricean first tap of TDL-D, TDL-E and
  TDLD30, a specular component of Doppler 0.7 f_D (`TdlParams::specular`,
  `Tdl::specular`, `LOS_DOPPLER_RATIO`); block `tdl` with the names `TDLA30` … and
  `TDL-A` … and the parameter `delay_spread_s`; Python `power_delay_profile(name,
  delay_spread_s=)`, `TdlChannel(..., delay_spread_s=)`, `TdlChannel.specular`. The nine
  tables equal those of Sionna 2.2.0 (recorded, `tests/oracle/test_sionna_tdl.py`). Existing
  profiles and their realisations are unchanged (the random draws of a channel without a
  specular component are those of before).

- Fixed: 20 cluster angles of the CDL tables of `comnum_channel::cdl` were mistyped
  (CDL-A clusters 5–7 and 22, CDL-B 23, CDL-D 8–13, CDL-E 13–14; the worst, CDL-D 12 and
  13, had all four angles wrong). They now equal TR 38.901 V19.5.0 Tables 7.7.1-1 to
  7.7.1-5, checked entry by entry against the PDF and against the tables of Sionna 2.2.0
  (found on reading the report). The delays and powers,
  hence the power-delay profiles, were right; the spatial correlation and the Doppler of
  the CDL realisations change for these profiles.

- Training-based estimation of a flat MIMO channel (Biguesh & Gershman, *IEEE Trans.
  Signal Process.* 54(3), 2006): `comnum_mimo::estimation` with the LS, scaled LS,
  LMMSE and relaxed MMSE estimators of $S = H P + V$ (eqs. (4), (16), (26), (48)), the
  LS-SLS and LS-RMMSE variants that estimate the trace of $R_H$ from the LS estimate
  (eq. (17)), the closed-form MSE of every estimator for any training and a mismatched
  prior, the DFT training of eq. (10) and the water-filled training of the LMMSE (eqs.
  (41)–(44)), and the BLUE combination of the LS estimates of several training periods
  (Lemma 4, eq. (59)); Python `MimoTraining`, `blue_weights`, `blue_combine`,
  `blue_mse`. Validated against Figs. 1–4 and the paper's equations
  (`validation/tests/mimo/biguesh2006.rs`: 121 readings within 7.4 %, the Monte Carlo within
  1.9 % of the closed forms; Fig. 4 at t = r = 2 below 6 dB and Fig. 5 contradict the
  paper's own equations and are reported). No existing result changes.

- Four repetitions removed, the results unchanged: the iterations of the turbo decoder
  are one function, `comnum_fec::turbo::iterate`, shared by `TurboCode` (LTE) and
  `CcsdsTurbo`; `Trellis::recursive` is the case of one polynomial of
  `Trellis::recursive_parities`; the single-stream checks of the `Modulate` and
  `Demodulate` blocks of the waveforms (OFDM, UFMC, OTFS, CE-OFDM, FBMC) are
  `one_stream` and `check_input` of the crate; the LMMSE frequency interpolation of
  `ofdm::estimation::estimate` is the plan of `ofdm::wiener::Wiener2d` (one
  factorisation per set of taps, `Wiener2d::plan`, which also speeds `Wiener2d` up).
  The turbo decoders, the trellises and `estimate` are bit-identical before and after;
  the taps of `Wiener2d` now follow the order of the pilots (the same linear system,
  the validations of Hoeher et al. unchanged to five digits).

### Simulation

- `comnum-sim`: scenarios run on every condition of a grid by a parallel runner whose
  results do not depend on the thread count; stopping rules on errors, precision,
  trials and time; Wilson and normal intervals; callbacks, CSV and JSON reports with
  the version, the commit and the configuration.
- The confidence intervals of a proportion of Brown, Cai & DasGupta (*Statistical
  Science* 16(2), 2001): `comnum_sim::stats::BinomialInterval` (Wald, Wilson,
  Agresti–Coull, Jeffreys, Clopper–Pearson), `ErrorRate::interval`, the exact
  `coverage` and `expected_length` for X ~ Bin(n, p); in Python
  `ErrorRate.interval(method=...)`, `sim.interval_coverage` and `sim.interval_length`.
  `comnum_math` gains `ln_binomial_pmf` (shared with the Reed–Solomon error
  probabilities), `beta_inc` and `beta_inv`. Validated against the paper
  (`validation/tests/simulation/brown2001.rs`, cards `brown2001_coverage` and `brown2001_table5`):
  the Wald coverages of Tables 1–2 to the printed digits, eq. (3), the limits of Sec.
  3.2 and 4.1.1, the crossings of Fig. 8 and the 252 Jeffreys limits of Table 5; the
  infima of the Wilson coverage over gamma >= 5 and 10 lie 0.001 to 0.002 below the
  paper (not explained). The runner still reports the Wilson interval: no existing
  result changes. Gallery `simulation/plot_04_brown2001_fig5.py`.

### Specifications and command line

- `comnum-spec`: campaigns written as YAML, JSON or TOML files (grid, link of blocks,
  metrics, stopping rule, outputs), 57 blocks with typed parameters and ports,
  expectations checked against closed forms and published curves, the search of
  operating points, the time of every block (`profile`), recordings (SigMF, raw IQ,
  NPY) as sources and as dumps, and a JSON Schema of the blocks. SigMF 1.x: every
  sample format (little and big endian; unsigned types centred by 2^(b-1), as
  sigmf-python), several channels, capture headers, trailing bytes, non-conforming
  datasets, `.sigmf` archives, centre frequencies and annotations; checked against
  sigmf-python.
- Blocks of other crates join the registry with `comnum_spec::blocks::register` (ADR 0006).
- Link theory: the bit error rates of equalisation with an imperfect channel estimate
  and of DPSK over Jakes' fading (Chang & Su 2002); the output series and in-band
  distortion of a nonlinear amplifier driven by OFDM (Banelli & Cacopardi 2000); the
  Bessel function J0; the exact Gray BER of PAM and rectangular QAM (Cho & Yoon 2002)
  and the density of the phase of a carrier in noise.
- CE-OFDM receivers and theory: the basic linear and enhanced receivers of Ahmed &
  Zeidler (2015) and the near-optimum GAMP receiver of López Morales et al. (2022); the
  spectrum and the powers of the intermodulation products (Guerreiro et al. 2015), the
  99 % bandwidth (Pacheco & Hatzinakos 2006), the optimum asymptotic gain (Guerreiro et
  al. 2016), the Bussgang analysis of the linear receivers per subcarrier and the effect
  of a cycle slip (Ahmed et al. 2012); the block `ce_ofdm_rx` takes a `receiver`.
  Reproduced against Figs. 1 of Guerreiro et al. 2016 and Pacheco & Hatzinakos 2006,
  Fig. 3 of López Morales et al. 2022 and Fig. 6 of Ahmed & Zeidler 2015
  (`validation/tests/multicarrier/ce_ofdm.rs`); a misprint in eq. (8) of Ahmed & Zeidler 2015.
- Chirp spread spectrum, the modulation of LoRa (`comnum_waveform::css`, Python `Css`):
  the chirps of Chiani & Elzanaty (2019) at any oversampling, with the waveform of
  Vangelista (2017) and FSK for comparison, the noncoherent DFT receiver, a same-SF
  interferer; the exact SER and its approximations, the cross-correlation, the spectrum
  in Fresnel functions and its lines, the interference pattern in closed form and the
  SER and FER under interference (Afisiadis et al. 2020). Reproduces Table I of Chiani
  & Elzanaty, Figs. 2 and 7 of Afisiadis et al. and Fig. 1 of Vangelista; a sign
  misprint in eq. (58) of Afisiadis et al. Adds `comnum_math::bessel_i0e`.
- Ergodic and outage capacity of Rayleigh MIMO channels in closed form
  (`comnum_mimo::capacity::{ergodic_rayleigh, ergodic_rayleigh_limit, outage_rayleigh,
  outage_capacity_rayleigh, wishart_eigenvalue_density}`, Python `ergodic_capacity`,
  `outage_probability`, `outage_capacity`), after Telatar (1999): Theorem 2, eqs. (13)
  and (16)–(18), Figs. 1–4. Adds `comnum_math::{erf, ln_gamma, gamma_p, expint_e1}`.
- Capacity and cutoff rate of CM and BICM over AWGN and Rayleigh fading
  (`comnum_mapping::bicm`, Python `cm_capacity`, `bicm_capacity`, `cm_cutoff_rate`,
  `bicm_cutoff_rate`) and set-partitioning labels (`Constellation::set_partitioning`),
  after Caire, Taricco & Biglieri (1998): Figs. 4–7; the "more than 3 dB" of Sec. III-C
  is 2.55 dB. The kernels of `theory::mi` and `theory::gmi` are shared with it.
- Pilot-based gain correction of OFDM against a cochannel interferer
  (`comnum_waveform::ofdm::correction`, Python `gain_limited_correction`,
  `sdr_gain_limited`, `sdr_optimum_correction`), after Cimini (1985): eqs. (20), (21),
  (25), Figs. 3–7 by simulation of his system; eq. (19) lacks a $-1$, the coherence
  bandwidth of eq. (23) is $1/(4\Delta)$, and the "6–7 dB" of the text is 4.5 dB on
  Fig. 5.
- The comb interpolations of Coleri et al. (2002): second order (eq. (19), with
  $\alpha = l/L$), not-a-knot cubic spline, the band-limited interpolator of MATLAB
  `interp` and the time-domain interpolation (`comnum_dsp::estimation::{
  quadratic_interpolation, spline_interpolation, lowpass_interpolation}`,
  `FrequencyInterpolation::{SecondOrder, Spline, LowPass, TimeDomain}`, Python
  `ofdm_estimate(method=...)`); their ranking of Sec. VI-B holds only when the comb
  oversamples the channel.
- Filtered multicarrier: FBMC/OQAM with the PHYDYAS prototype, polyphase modulator and
  demodulator, OQAM staggering and the intrinsic interference (`comnum_waveform::fbmc`,
  Python `Fbmc`, `oqam_stagger`, `oqam_destagger`), and UFMC with Dolph–Chebyshev
  sub-band filters (`comnum_waveform::ufmc`, `comnum_dsp::window::chebyshev`, Python `Ufmc`,
  `chebyshev_window`), after Farhang-Boroujeny (2011) and Schaich & Wild (2014).
  Bussgang's complex gain and distortion power of any memoryless amplifier
  (`comnum_hardware::gaussian_input::bussgang`, Python `amplifier_bussgang`). Reproduces
  Tables 1–3 and Figs. 7, 10, 11 of Bouhadda et al. (2014) (the curves of their Fig. 7
  are at an input back-off of 8 dB, not 4 dB) and Figs. 2 and 4 of Nissel & Rupp
  (2017) for OFDM and CP-OFDM; their eqs. (16)–(17) are the Rayleigh closed forms
  already in `comnum_mapping`.
- OTFS with rectangular pulses and one prefix per frame (`comnum_waveform::otfs`: ISFFT
  and Heisenberg transform, the sparse effective channel of Theorem 1 of Raviteja et al.
  2019 and the general one of their eq. (13)), and the delay–Doppler channel of
  discrete paths with one Doppler shift each (`comnum_channel::multipath::DelayDoppler`,
  realised as `Taps`); detection with the MMSE detector of `comnum-mimo`. Python `Otfs`,
  `delay_doppler_paths`.
- Optical wireless channels (`comnum_channel::owc`): the indoor channel of a rectangular
  room with Lambertian emitters and reflections of every order, computed recursively on
  the elements of the surfaces (Barry et al. 1993), with the illuminance of LED lights,
  the −3 dB bandwidth and the split of on–off keying into signal and interference
  (`owc::indoor`); free-space optics with log-normal, gamma–gamma and negative
  exponential turbulence, pointing errors and the attenuation of fog (`owc::atmospheric`,
  `Fso` implements `Propagate`); the PIN-FET receiver of Komine & Nakagawa (2004)
  (`comnum_hardware::optical::PinFetReceiver`), gamma variates (`SimRng::gamma`) and the
  modified Bessel function $K_{\nu}$ (`comnum_math::{bessel_k, ln_bessel_k}`). Reproduces
  the powers of the four configurations of Barry et al. (1993) and the bandwidths of
  three of them, Figs. 3, 4, 6, 8 and 9 of Komine & Nakagawa (2004), and Figs. 3 and 6–8
  of Al-Habash, Andrews & Phillips (2001), whose exponent 7/6 of eqs. (19) and (26) is
  5/6. Python `OwcRoom`, `OwcIllumination`, `OwcImpulseResponse`, `PinFetReceiver`,
  `Fso`, `GammaGamma`, `PointingError` and helpers.
- Validation cards (`validation/cards/`): one file per algorithm with its inputs, the
  expected outputs and their tolerance, and the exact source of the expectation, run by
  `tests/test_cards.py` and checked by a JSON Schema.
- DRY pass: `comnum_math::{gaussian_nodes, log_sum_exp_iter, normal_pdf, simpson}`,
  `comnum_math::fft::filter`, `comnum_dsp::signal::map_envelope` and
  `comnum_dsp::compensation::dpd::polynomial_term` replace copies across the crates.
- Comparisons with other libraries: the reference vectors of Sionna (CRC, convolutional
  and turbo codes, bit for bit), py3gpp (5G NR), sigmf-python, next to OptiCommPy,
  GNPy and AFF3CT; a documentation page gathers them.
- The command `comnum` (`cargo install comnum-spec`): `run`, `check`, `verify`,
  `conditions`, `blocks`, `schema`.
- `specs/`: the example campaigns, and `specs/validation/` checked against the
  literature.

### Validation

- The closed form of the ISRS GN model of Semrau, Killey & Bayvel (*JLT* 37(9), 2019),
  `comnum_channel::optical::gn::isrs_closed_form_eta` (eqs. (5), (10), (11): the SPM and
  the XPM of every interferer with the first-order tilt of inter-channel stimulated Raman
  scattering, the SPM accumulated as $n^{1 + \varepsilon}$) and the coherence factor
  $\varepsilon$ of Poggiolini (*JLT* 30(24), 2012, eq. (22)), `coherence_factor`;
  `comnum.gn_isrs_closed_form` in Python. Validated against the dashed curves of Figs.
  4a, 4b and 5 of the paper on its 251-channel, 10 THz link (cards `semrau2019_fig4a`,
  `semrau2019_fig4b`, `semrau2019_fig5`): within 0.12 dB without ISRS and 0.25 dB with
  it away from the band edges, the change by ISRS against the power transfer within
  0.14 dB, ε = 0.151 against 0.15. Found: the last term of eq. (10) of the preprint is
  not dimensionless and is left out (the closed form is 0.35 to 0.7 dB below the paper's
  at the band edges under ISRS); eq. (3) gives a power transfer of 6.5 dB where the text
  says 6.3. No existing result changes.
- The DGD of `PmdLink` against Foschini & Poole (*JLT* 9(11), 1991): Maxwellian (Sec. V,
  Fig. 3: the mean, the mean over the rms, the quartiles, the density at the mode within
  1 % of the figure), the components of the input PMD vector i.i.d., the mean DGD of
  `PmdLink::random` growing as the square root of the length (eq. (18)); cards
  `foschini1991_fig3` and `foschini1991_eq18`. Found: the output PMD vector of the model
  is not isotropic (no rotation follows the last section); the DGD is unaffected. The
  module "EDFA and PMD" reaches level A. Reading cards of Ip & Kahn 2008 (`ip2008`) and
  Saul 2004 (`saul2004`) for the next validations.
- `upfirdn` convolved the upsampled signal by FFT whatever its length: a 120 s capture
  at 2 MS/s through `select_channel` to 153.6 kHz (up 48, down 625) asked for 11.5 G
  samples and was killed. Beyond 64 Mi upsampled samples the direct polyphase sum runs
  instead, which allocates nothing (the capture in seconds); the results are unchanged
  below. Found by the AIS receiver on a capture.
- AIS, the automatic identification system of ships (ITU-R M.1371-5):
  `comnum_systems::ais` (the packets of Annex 2 through GMSK at 9600 bit/s with NRZI,
  a receiver for a channel of a capture: polar discriminator, half-bit integrator,
  frequency offset removed, a bit clock from the zero crossings; the messages 1-5, 18
  and 24 of Annex 8; the `!AIVDM` sentences of NMEA 0183) and `comnum_systems::hdlc`
  (the HDLC framing of ISO/IEC 13239: flag, bit stuffing, the CRC-16/X-25 frame check
  sequence, NRZI). Python `ais_transmit`, `ais_receive`, `ais_decode`,
  `ais_position_report`, `ais_aivdm`, `ais_messages`, `ais_front_end_rate`; the Rust
  example `ais` on a SigMF recording. Checked by the CRC catalogue (check value and
  residue), the recorded decoding of pyais 3.3.1 (`tests/oracle/test_ais_oracle.py`),
  and a simulated ship through a capture at 2 MS/s; 9 packets of 10 at 10 dB of SNR in
  the 25 kHz channel. No ship decoded from a capture yet (30 s indoors: no packet).
- Windowed OFDM and the output back-off of a Rapp amplifier, for Figs. 8 and 9 of Saul
  (*Proc. IEEE VTC2004-Fall*, 2004): `Ofdm::windowed` (the raised-cosine window of whole
  blocks over the guard interval, the postfix of a block overlapping the ramp of the next;
  Python `Ofdm.windowed`) and `Rapp::scale_for_output_backoff` (the input scale for an
  output back-off, eq. (3) of the paper, by bisection; Python `pa_rapp(obo_db=)`). Cards
  `saul2004_fig8` and `saul2004_fig9`, orange: the recursive-clipping curves (`ClipFilter`)
  of the PSD after the amplifier against the Eb/N0 loss are reproduced (QPSK within 0.2 dB
  from -31 to -78 dBc, 16-QAM at both ends), the PSD floors of ACE too (-67.4 and -47.0
  against -67 and -46.5 dBc), but comnum's clipping-based `Ace` reaches them at a third
  of the paper's loss: the paper's smart gradient projection extends the points further.
  `Card::inputs_matching` lets a test take the abscissae of a figure from its card. No
  existing result changes.
- The receiver of Ip & Kahn (*JLT* 26(20), 2008, Fig. 3(b)) and its Fig. 12 (BP with one
  step per span against the distance): `PulseShape::Rz` (the field of the 50 %
  sinusoidal carver, `pulse_shape("rz")`), `comnum_dsp::filter::Butterworth` (a low-pass
  of any order by the bilinear transform, second-order sections; Python
  `butterworth_lowpass`), `comnum_dsp::equalizer::ls_equalizer` (the fractionally
  spaced least-squares equaliser of a block on known symbols), and
  `comnum_metrics::phase_error_std` (the s.d. of the phase error after the constant
  gain and phase; Python `phase_error_std`). Card `ip2008_fig12`, orange: the AWGN limit
  of the ASE (closed form within 8 % of the figure, measured at 3 samples per symbol
  within 0.6 % of the closed form), BP-SSFM10 at M/K = 3 up to 50 spans and the claims
  of the text are reproduced; the nonlinear penalty of the figure is larger than
  comnum's at every distance (linear equalisation 0.114 against 0.204 rad at 25 spans),
  not explained: the model reproduces Häger & Pfister 2021 within 0.13 dB, the
  equaliser is not the cause, and no single scale of the nonlinearity reproduces the
  whole figure. No existing result changes.
- `GardnerLoop::run` panicked when a loud signal drove its timing back before its
  first two output samples (an index below zero; reachable from Python by
  `gardner_clock_recovery`): the output index now stays at 2 at least. Found by the RDS
  demodulator; the results of a signal of moderate level do not change.
- TS 36.211 V18.0.2 and TS 36.212 V18.1.0 downloaded: Table 6.11.2.1-1 (the 168 SSS
  indices) verbatim in `ofdm/lte.rs::the_sss_indices_are_table_6_11_2_1_1`; the 188 rows
  of the QPP table of the LTE turbo code (Table 5.1.3-3), transcribed from Sionna, are
  identical to those of the standard (literature entries `ts36211`, `ts36212`).
- No Rust panic reaches Python (ADR 0016): `tests/test_no_panic.py` calls every
  function and every method of the package with degenerate arguments and fails on a
  `PanicException`. It found two: `Constellation::energy_with` and `LowRank::mse_floor`
  now return `Result` (`Error::Shape` for probabilities or a correlation of another
  size; `ValueError` in Python). The length and index preconditions of the kernels
  (FFT, `matvec`, the split-step solvers) stay documented panics.
- A result is checked in one place (`docs/ARCHITECTURE.md`, "Where a test goes"):
  `tests/test_performance.py` no longer repeats what CI checks elsewhere, the error
  rates of QAM, PSK and PAM over AWGN and the mutual information of 16-QAM (the
  specifications `proakis_awgn` and `information_rates` of `ci_specs`) and the
  diversity of Alamouti and zero forcing, directly and as a chain
  (`specs/mimo_diversity.yaml`): 17 cases and 137 lines fewer. The Python tests marked
  `slow` (the WDM links, the dual-polarisation back-propagation, 79 of the 105 s of the
  suite) run on one runner of CI, Linux with Python 3.12, instead of nine.
- The levels of `docs/MATURITY.md` are computed, not written: every item of evidence of
  `validation/maturity.yaml` declares its kind (the evidence of the cards, and `self`
  for self-consistency), the level is A with a recording, table, curve or claim, else
  B with a closed form, else C; `maturity_table.py --check` fails on an item whose kind
  is none of the kinds of the cards it cites. The space-time block codes move from A to
  B (their evidence is the designs of Tarokh et al. and the closed form of MRC); the
  other 67 modules keep their level.
- Every validation card says what its expected values are, `evidence`: a closed form,
  a table (numbers printed verbatim), a curve (points read on a figure), a claim of
  the text (a range) or a recording of a tool, apart from where they come from
  (`source.kind`: paper, standard or tool; `closed_form` is no longer a source).
  `cards.py --check` holds a recording to a tool and every tolerance but an exact one
  to its `why`; `docs/LITERATURE.md` has a column *Evidence* (27 documents with a
  closed form, 19 with a table, 29 with a curve, 10 with a claim, 2 with a recording).
- The tree has one folder per notion at its root: `benches/` moves to
  `benchmarks/rust/` and the speed against OptiCommPy to `benchmarks/opticommpy/` (with
  `aff3ct/` and `libraries/`); the synthetic capture of `data/` to `examples/data/`; the
  recipes of the `justfile` and the examples write their reports under
  `target/results/` instead of `results/` at the root (`comnum run` keeps `results/` by
  default, in the directory where it is run).
- `docs/api-duplicates.txt` gives every accepted name the number of modules that may
  define it and its reason (the 100 names that had none reviewed: no copy among them);
  `api_index.py --duplicates --check` fails when a name is defined in more modules than
  accepted, so that a new copy of `decode` or `estimate` no longer passes unseen.
- `cards.py --check` fails on a validation test named after an entry of
  `validation/literature.yaml` that loads no card; the 15 tests that still keep their
  published values in the code are listed in `validation/cards/uncarded.txt`, a list
  that only shrinks.
- The four source files of more than 1 200 lines split by notion, the public paths
  unchanged (`docs/ARCHITECTURE.md`, "File size"; `tools/test_layout.py` warns above
  1 200 lines): `comnum_channel::multipath::{taps, tdl, delay_doppler, echoes}`,
  `comnum_math::nlse::manakov`, `comnum_fec::ccsds::{ldpc, turbo}`, and the MIMO
  bindings (`amp`, `estimation`); the same 667 unit tests.
- `just check` is three recipes (`check-rust`, `check-tests`, `check-python`) that the
  jobs of CI run as they are, so that the two cannot differ; the validation
  specifications of CI are listed once (`ci_specs` of the `justfile`); `just regen`
  rewrites every generated file; the Python environment is `uv sync --locked` at the
  versions of `uv.lock` (`just setup-py`), locally and in CI.
- One place per kind of test (`docs/ARCHITECTURE.md`, "Where a test goes"), checked by
  `tools/test_layout.py` in `just check`: the 72 files of `validation/tests/` grouped into
  eleven theme folders, each one test binary (`cargo test -p comnum-validation --test
  <theme>`), which halves the build of the validation tests (28.7 s to 12.8 s) with the
  same 188 tests; the harness of TS 38.101-4 moved to `comnum_validation::ts38101_4`, its
  exploration of the RAN4 alignment conditions to the example
  `validation/examples/ts38101_4_alignment.rs`; the Python tests that compare with a
  recorded tool gathered in `tests/oracle/` beside their recordings. The `mod literature`
  checks of one item against one document stay next to the code.
- `literature` unit tests check the tables, examples and closed forms of the documents
  of `references/` (ITU-R M.1225, 3GPP TS 38.212 and TR 38.901, IEEE 802.11-2020, ETSI
  EN 300 421 and EN 300 744, CCSDS 131.0-B, and the papers they cite).
- `specs/validation/`: Proakis & Salehi over AWGN, information rates by quadrature,
  Häger & Pfister 2021 Fig. 5, the required SNR, MSK against BPSK, Thompson et al.
  2008 eq. (15), and Table 13 of EN 302 307-1 for the 28 DVB-S2 MODCODs.
- `comnum-validation`: the channel codes against 17 recorded AFF3CT configurations; the
  diversity orders of the MIMO detectors against Yang & Hanzo 2015, Table X; the CCDM,
  the information rates of shaping and PAS on the DVB-S2 codes against Schulte &
  Böcherer 2016, Böcherer, Steiner & Schulte 2015 and Liang & Kahn 2025; the EGN model
  against Carena et al. 2014, Fig. 1(a), and against the fourth-order noise of Dar et
  al. 2014, Sec. 3 (the EGN correction of the cross-channel interference gives 0.07 to
  0.72 dB less, and the split-step simulation of the paper's link confirms its values); the massive-MIMO uplink against Ngo, Larsson & Marzetta 2013, Figs. 1
  and 4; the RIS against Basar et al. 2019, Figs. 6 and 7; the blind phase search
  against the linewidth tolerances of Pfau, Hoffmann & Noé 2009, Table I; the limits,
  uncoded baselines and the 64-state convolutional code against Costello & Forney 2007;
  the step rule of the Manakov solver against Fig. 7 of Musetti et al. 2018; the
  self- and cross-channel interference of the Manakov solver against the band integral of
  the GN model (Poggiolini 2012); the PMD vectors, mean delays and random walk of the
  PMD sections against Gordon & Kogelnik 2000, and their pulse broadening against
  Sunnerud, Karlsson & Andrekson 2000; the scalar
  and Manakov split-step solvers against the exact two-soliton collision
  (Hirota 1973) of Sinkin et al. 2003, and the Manakov solver against the exact
  inelastic collision of two Manakov solitons of Radhakrishnan et al. 1997, and their
  cross-phase modulation against the walk-off theory of Chiang et al. 1996.
- The Python tests compare the blocks with the recorded outputs of OptiCommPy 0.11 and
  GNPy 3.0 (`tests/oracle/`) and reproduce the classical results of the textbooks.
- The DVB-S and DVB-T transmitters against the recorded outputs of the gr-dtv module of
  GNU Radio 3.10.12 (`tools/record_gr_dtv.py`, `tests/oracle/test_gr_dtv.py`): the outer coding
  and the I and Q pairs of the five punctured rates bit for bit, the cells of five DVB-T
  configurations (2K and 8K, QPSK to 64-QAM, rates 1/2 to 7/8, pilots and TPS) and the
  samples of four 2K symbols within 6e-8 after one real gain; the DVB-S2 PLFRAMEs of
  the 28 MODCODs in normal frames, the 24 of short frames with pilots and three Gold
  codes within 4e-8 at unit gain. No existing result changes.
- `docs/MATURITY.md` grades the evidence of every module.

- MIMO: the maximum-ratio (matched-filter) detector, `Linear::mr` and `"mr"` of
  `DetectorKind`, `mimo_detect`, `mimo_llr` and `mimo_estimate`, with the variance of
  the interference of the other streams; `comnum_mimo::massive::Uplink`, the uplink of
  multi-user massive MIMO with perfect or estimated CSI and the closed-form bounds on
  the rates of MR and ZF of Ngo, Larsson & Marzetta (2013).
- Reconfigurable intelligent surface: `comnum_channel::ris` (the two Rayleigh hops of $N$
  elements without line of sight, the gain of a configuration, the MGF and the mean SNR
  of the co-phased elements in the Gaussian approximation), the co-phasing
  `comnum_mimo::precoding::ris_co_phasing`, and the SER of PSK and square QAM from the MGF
  of the SNR, `comnum_mapping::theory::ser_mgf` (Basar et al. 2019).
- `comnum_mapping::differential::QuadrantQam`: square QAM whose quadrant is
  differentially encoded and whose points inside a quadrant are Gray-labelled (Pfau et
  al. 2009, Sec. III-A), decoded whatever multiple of $\pi/2$ the carrier recovery
  leaves.
- `PmdLink::random` and `PmdLink::with_mean_dgd` divide out the spread of the section
  DGDs (mean square 1.04 times the square of their mean): the mean DGD of a link was 2 %
  above its target.
- `comnum_math::nlse::StepRule::Fwm`: the step rule of Musetti, Serena & Bononi (2018),
  the first step of a span bounding the four-wave-mixing phase across the signal
  bandwidth (measured by `signal_bandwidth`), the next ones growing with the loss
  (`StepUpdate`), with an optional bound on the nonlinear phase. The Manakov solver
  takes by default 20 rad and 0.02 rad (`StepRule::FWM_DEFAULT`) instead of OptiCommPy's
  0.02 rad of nonlinear phase alone, which leaves 0.45 dB of error on `wdm_gn` and
  `raman_wdm`; the parameters `step_rule`, `fwm_phase_rad` and `bandwidth_hz` of the
  block `manakov` and `step_rule`, `fwm_phase`, `bandwidth_hz` of `comnum.ManakovSsfm`
  (`docs/adr/0013-manakov-defaults.md`).
- `comnum_math::nlse::NonlinearStep`: the Manakov solver takes, by default, the nonlinear
  phase at the middle of the step (`Midpoint`, the scheme of the scalar solver), or the
  trapezoidal mean of OptiCommPy (`Trapezoidal`, the former default), which smooths
  the cross-phase modulation of channels that walk off within a step;
  `ManakovSsfm::nonlinear`, the parameter `nonlinear_step` of the block `manakov` and
  of `comnum.ManakovSsfm`. The results of the Manakov solver change: 0.9 dB more NLI
  than before on a five-channel link with steps of 0.5 km, within 0.17 dB of the GN
  model.
- `FiberParams::slope_ps_nm2_km` and `Medium::beta3_s3_km`: the dispersion slope and the
  third-order dispersion $\beta_{3} = (\lambda / 2\pi c)^{2} (S \lambda^{2} + 2 \lambda D)$
  it gives, a cubic phase in the linear step of every propagator of the engine (the
  scalar and Manakov split-step, the PMD link, the back-propagation); 0 by default, the
  former behaviour. The parameter `slope_ps_nm2_km` of the fibre of the specifications
  and of `comnum.FiberParams`.
- `comnum_math::nlse::SamplingCheck` and `sampling_check`: the sampling conditions of a
  split-step run (Musetti et al. 2018, Appendix B), a rate of at least $\frac{3}{2} B$
  and a block longer than the walk-off $|\beta_{2}| 2\pi B L$, checked before every
  propagation of `Ssfm` and `ManakovSsfm` (`check_sampling`, on by default; the parameter
  of the blocks `ssfm` and `manakov` and of the Python classes), which refuse a field
  that fails them with the rate or the block it needs.
- One split-step engine for every propagator (`docs/adr/0014-one-fibre-propagator.md`):
  `ManakovSplitStep::run_fields` runs one or two fields with the gain, the birefringent
  sections of a PMD link (`Birefringence`) and any step rule, and `Ssfm`, `ManakovSsfm`
  and `ManakovPmd` call it; the scalar solver with fixed steps gives the same samples as
  before, bit for bit. `Ssfm::rule` takes another step rule than the fixed steps.
- CCSDS 131.0-B-5, TM Synchronization and Channel Coding (2023): `comnum_fec::ccsds`
  with the interleaved Reed–Solomon codeblocks of Sec. 4 (`CcsdsRs`: E = 8 or 16, depths
  1 to 5 and 8, virtual fill, the dual basis of Annex F, `to_dual_basis`), the turbo
  codes of Sec. 6 (`CcsdsTurbo`: 16 states, the permutation of Sec. 6.3 g), rates 1/2 to
  1/6), the (8160, 7136) LDPC code of Sec. 7.3 and the nine AR4JA codes of Sec. 7.4
  (`CcsdsLdpc`, with the encoders of the standard, the virtual zeros and the punctured
  bits); `comnum_systems::ccsds` with the convolutional code of Sec. 3
  (`Convolutional`, $G_{2}$ inverted at rate 1/2, Table 3-1), the randomizers of
  Sec. 10.4, the markers of Sec. 9 and the channel access data unit (`cadu`). Python
  `comnum.CcsdsRs`, `CcsdsLdpc`, `CcsdsTurbo`, `CcsdsConvolutional`,
  `ccsds_randomizer`, `ccsds_marker`, `ccsds_cadu`, and `LdpcCode.is_codeword`.
  Validated exactly against Tables F-1, 3-1, 6-2, 7-5, Annex G, the parity checks of
  Table 7-1 and Sec. 7.4.2 and the bits printed for the randomizers and the markers
  (card `ccsds131_exact`); the standard gives no error-rate figure.
- `comnum_fec::conv::Trellis::recursive_parities` (a recursive systematic code with
  several parity outputs), `comnum_fec::ldpc::LdpcCode::without_encoder` (a decoder whose
  encoder is external) and `comnum_math::gf2::Matrix::inverse` and `mul_vec`.
- The `rs` block of the specifications takes the primitive polynomial `poly`: its CCSDS
  recipe built the code on the field of `0x11D` instead of $x^{8} + x^{7} + x^{2} + x + 1$
  (`0x187`); a specification that followed the recipe now builds the code of CCSDS.
- DVB-T, ETSI EN 300 744 V1.6.2 (`comnum_systems::dvbt::Dvbt`, Python `comnum.Dvbt`):
  the non-hierarchical transmitter (the outer coding shared with DVB-S, the inner code,
  the inner interleaver `InnerInterleaver` with the symbol interleaver
  `symbol_permutation`, the constellations of Figure 9 for alpha = 1, 2 and 4, the TPS
  with their BCH code, `Tps`) and a receiver that knows the channel; the OFDM frame in
  `comnum_waveform::ofdm::dvbt` (2K and 8K, guard intervals, the continual pilots and TPS
  carriers of Tables 7 and 8, the scattered pilots, the pilot PRBS, `DvbtFrame`); the
  static channels F1 and P1 of Annex B, `comnum_channel::multipath::Echoes` (Python
  `comnum.dvbt_channel_gains`). Validated bit for bit against Table C.1 and against
  Table A.1: the 15 C/N of the Gaussian channel within 0.12 dB, F1 within 0.26 dB, P1
  0.1 to 0.7 dB below (card `dvbt_tablea1`). Gallery `systems/plot_04_dvbt.py`.
- `comnum_systems::dvb::burst_encode` and `burst_decode`: the outer coding of a burst of
  packets with the flush of the interleaver, shared by DVB-S and DVB-T.
- DVB-S, ETSI EN 300 421 V1.1.2 (`comnum_systems::dvbs::Dvbs`, Python `comnum.Dvbs` and
  `comnum.dvb_packets`): the transmitter of Secs. 4.4 and 4.5 and a frame-aligned
  receiver, on bursts of transport packets. The coding it shares with DVB-T is
  `comnum_systems::dvb`: the energy dispersal (`energy_dispersal`, the PRBS of Figure 2,
  reset every 8 packets, the first sync byte inverted), RS(204, 188) (`outer_encode`,
  `outer_decode`), the convolutional interleaver I = 12, M = 17 and the inner code
  (171, 133) with $X$ sent before $Y$, punctured by Table 2 (`InnerRate`). Validated
  against Table 3 (the Eb/N0 of a BER of 2e-4 after the Viterbi decoder, 0.10 to
  0.22 dB below the table minus its 0.8 dB margin, card `dvbs_table3`), Table 2 (the
  patterns, the I and Q rows, the free distances), Figures 2, 4 and 5 and the rates of
  Tables C.1 and D.1. Gallery `systems/plot_03_dvbs.py`.
- `comnum_fec::interleaving::ConvolutionalInterleaver`: the convolutional (Forney)
  interleaver and deinterleaver of a stream, $I$ branches of depths $M j$ (Forney 1971).
- `dvbs2::bb_scramble` uses the PRBS of `comnum_systems::dvb::dispersal_prbs` (the same
  sequence; its output is unchanged).
- The DVB-S puncturers of `comnum_fec::puncture` cite Table 2 of EN 300 421 V1.1.2 (they
  cited Table 1, the system interfaces) and say that they send $Y$ before $X$.
- `comnum_mapping::source::O150Sequence`: the eight pseudo-random test sequences of
  ITU-T O.150 (05/96), Sec. 5, exactly (the polynomial, the inverted outputs of the
  sequences of 15, 23, 29 and 31 stages, the zero suppression of Sec. 5.5, the starting
  phase of Sec. 5.1), with `O150Sequence::iter` to stream a whole period; Python
  `comnum.o150_sequence(name, length)`. `prbs` takes the order 29. Validated over a whole
  period of every sequence against Table 1 (card `o150_sec5`); two misprints of the
  Recommendation (Secs. 5.5 and 5.8) in `validation/literature.yaml`.
- The blind phase search against Pfau, Hoffmann & Noé 2009, Fig. 7 (the penalty
  against the linewidth, 34 points) and Fig. 6 (the penalty against the number of test
  phases, the floors and the 4- and 16-QAM curves; the steep rise of 64- and 256-QAM at
  the smallest B is not reproduced): cards `pfau2009_fig7` and `pfau2009_fig6`. The
  link of `pfau2009_table1` starts its carrier at a random phase: on the grid of the
  test phases the quantisation of the phase cost nothing (256-QAM floor 0.34 dB,
  0.51 now, the paper 0.55).
- The first reading card, of Pfau et al. (2009): the extraction of a
  reference (model, equations, parameters, figures read) that the tests and cards
  are written from.
- `comnum_fec::bounds::gaussian_limit_ebn0_db`: the Shannon limit on $E_{b}/N_{0}$ with
  Gaussian inputs, $(2^{\eta} - 1)/\eta$, $-1.59$ dB as $\eta \to 0$.
- The GN integral of a lossless span (distributed amplification) takes the limit $L^{2}$
  where its link function is 0/0, instead of NaN.

### Python package

- `comnum`: NumPy bindings of the blocks with explicit generators, one vocabulary for
  the arguments (`fs`, `fc_hz`, `noise_var`, `esn0_db`, …), the closed forms
  `theory_*`, the codes as classes that encode and decode whole blocks, typed with a
  generated stub.
- `comnum.Chain`: links of native and Python blocks, traced block by block, run as
  campaigns on the Rust runner, converted to and from specifications; the generated
  classes of `comnum.blocks`; `comnum.sim` for Python trials on the same runner;
  `Results`, `compare`, probes, profiles and `comnum.plot`.
- abi3 wheels for CPython 3.11 and later on Linux (manylinux, musllinux), macOS and
  Windows.

### Consistency of the API

A review of the whole library before the release settled its conventions
(`docs/ARCHITECTURE.md`) and removed the duplicated kernels:

- invalid input returns an error instead of panicking: the interpolators of
  `comnum_dsp::estimation`, `Ofdm::to_time` and `to_active`, `oqam_stagger` and
  `oqam_destagger`, the LLRs of `Constellation::llr` (and `llr_with_variances` for one
  noise variance per symbol), `theory::ber` and `ser` outside the orders they cover,
  `ber_rayleigh_mrc` without a branch, the closed forms of chirp spread spectrum, CE-OFDM
  and OTFS prefixes longer than a block, FBMC demodulation of a signal of another length;
- one kernel per method, one layer down: decibels (`db2amp`, `amp2db`), `qfuncinv`,
  `diric`, `try_simpson`, `gauss_legendre`, `linalg::solve_tridiagonal` and the unitary
  and centred FFT in `comnum-math`; cyclic prefixes, rotations, interpolation
  (`comnum_dsp::interpolation`, gathered from the estimators and the timing recovery),
  spectral estimation (`comnum_dsp::spectrum`) and Bussgang's decomposition
  (`comnum_dsp::nonlinear`, shared by clipping and the amplifiers) in `comnum-dsp`;
  `hard_decisions` and the seeded families of sequences (`rng::member`) in `comnum-core`,
  their QPSK draw in `comnum_mapping::source::qpsk`; `add_noise` in `comnum-channel`; the frequency correlation of a power-delay
  profile; the PRBS on the shared LFSR;
- the channel stays in `comnum-channel`: the delay–Doppler channel of OTFS
  (`DelayDoppler`, `DelayDopplerModel`), the validation tests draw their channels there;
- one API for the mutual information over AWGN and Rayleigh fading
  (`theory::constellation_mi`, `constellation_gmi`, the CM and BICM capacities),
  `theory::ber_exact`, and `ser_orthogonal_noncoherent` shared by noncoherent FSK and CSS;
- the modems share their vocabulary (`n_fft`, `n_active`, `n_cp`, `block_len`, public
  configurations checked by `check`) and the `Modulate`/`Demodulate` traits (UFMC, OTFS);
  `BlockCode` for BCH and the DVB-S2 FEC;
- names: `_lin` on linear SNRs, `noise_var`, `from_name` for the parsers that take
  arguments, `water_filling_powers`, `Domain` (real or complex message passing),
  `decode_alamouti`, the argument order `(rx, tx)` of the metrics and `(kind, m, …)` of
  the closed forms; Python `bandwidth_hz`, `smoothness`, `n_fft`, keyword-only options;
- Python bindings of OTFS (`Otfs`, `delay_doppler_paths`), `fading` for `theory_mi` and
  `theory_gmi`.
- the documentation follows one taxonomy, in the order of a link (foundations, coding and
  shaping, waveforms, MIMO, wireless channels, optical, hardware impairments, receiver
  DSP, metrics and analysis, systems and standards, simulation and specifications): the
  Python API reference, now complete (a test fails if an exported name is missing), the
  gallery, one sub-gallery per theme with the reproduced figures in their theme,
  `docs/LITERATURE.md` and `docs/MATURITY.md`; the maturity tables of `MATURITY.md`, of
  the README and of the Python documentation are generated from
  `validation/maturity.yaml` (`just maturity`, checked by `just check`).

### Conventions of the API (plan of 30 October 2026)

The rules of `docs/ARCHITECTURE.md` (naming conventions, errors) applied to the
crates; the Python API follows in the next section.

- Units in the names: `Cdl::{fc_hz, velocity_mps, delay_spread_s(), max_doppler_hz()}`,
  `Ray::doppler_hz`, `WdmSpectrum::freqs_hz`, `WdmTransmitter::freqs_hz`,
  `Numerology::subcarrier_spacing_hz`, `Photodiode::{temperature_k, dark_current_a}`,
  `SyncRule::Optimum { snr_lin }`, and the parameters `snr_lin`, `es_n0_lin`, `ts_s`,
  `f_hz`, `spacing_hz`, `offset_hz`, `delay_s`, `bandwidth_hz`, `spacing_s`
  (`fftfreq`); the keys of the specifications `offset_hz`, `omega_rad` (`cfo`),
  `sigma_rad` (`phase_noise`), `tau1_s`, `tau2_s` (`ddpll`), `max_nl_phase_rad`
  (`manakov`), `spacing_wavelengths` (`cdl`) and `coherence_samples` (`rayleigh`).
- One word per notion: `sps`, `rolloff` (`rrc_taps`, `rc_taps`), `noise_var`,
  `CeOfdm::n_subcarriers` and `CeOfdm::sps`, `OfdmFrame::n_data_symbols`.
- `Jacobian { Exact, MaxLog }` (`comnum_core::traits`, re-exported as
  `comnum_math::Jacobian`) for `max_star`, `bcjr`, `TurboDecoder`, `LabelHalves`,
  `llr_from_metrics`, `Constellation::llr` and its variants, `Dvbs2::llr` and the
  `Demap` trait; the specifications keep their key `max_log`.
- Errors: `Error::Shape` for lengths and empty inputs, `Error::Config` for values, in
  `lama`, `oamp`, `welch`, the symbol metrics, the blind estimators, the information
  rates, the frequency and frame synchronisation and the AWGN channel.
- Parameter structs: `TdlParams`, `CdlParams` and `ClusterTable`, `PortEstimation`
  (`estimate_port`), `FrameDetection` (`OfdmFrame::detect`), `MessagePassing` (`lama`,
  `oamp` and their state evolutions); `Ufmc` is built from `Default` by struct update;
  `PilotPattern::{block, comb, lattice}`; the CE-OFDM closed forms read the receiver of
  the modem.
- Containers: `stream`/`stream_mut` are slices on every container, `stream_view` the
  `ndarray` view; `Signal::len` and `Signal::resize`.
- Names: `Constellation::{map_bits, demap_bits}` and `Dpsk::{map_bits, demap_bits}`,
  `Ccdm::match_bits`, `CeOfdm::demodulate_with_receiver`, `pmd::ManakovPmd`,
  `theory::RateChannel`, `theory::ser_esn0` and `theory::ber_orthogonal_noncoherent`,
  the CE-OFDM `theory::{ser_esn0, ber_esn0}`; the DVB-S puncturers of rates 2/3, 3/4
  and 5/6 are aliases of the IEEE 802.11 patterns.
- The optical propagators work in place and return `Result`: `Ssfm::propagate`
  (`propagate_f32`, `propagate_with`), `Edfa::apply`, `PmdLink::apply` (on a `DualPol`),
  as `ManakovSsfm` and `ManakovPmd`.

### Conventions of the Python API (plan of 30 October 2026)

The Python rules of `docs/ARCHITECTURE.md` and the names of the crates, applied to the
package (0.2.0 is unreleased: the former names are not kept).

- Names of the crates: `Constellation.map_bits`/`demap_bits` and the same for `Dpsk`
  (were `modulate`/`demodulate`), `Ccdm.match_bits` (`matched`), `ManakovPmd`
  (`Manakov`), `CeOfdm(n_subcarriers)`, `CeOfdm.n_subcarriers` and the parameter
  `n_subcarriers` of the blocks `ce_ofdm_tx` and `ce_ofdm_rx` (`n_data`),
  `CeOfdm.ser_esn0`/`ber_esn0` (`ser`/`ber`), `theory_ser_esn0` (new), `lagrange_cubic`
  (`interpolate_cubic`), `snr_crb_nmse(sps=)` (`samples_per_symbol`),
  `CdlChannel(spacing_wavelengths=)` (`spacing`), `TurboCode(max_log=False)`
  (`exact=True`), `Photodiode(temperature_k=, dark_current_a=)` and its attributes
  (`temperature_c` in °C, `dark_current`), `NrLdpc(k, e)` and `NrPolar(k, e)` (`n`),
  `NrLdpc(max_iterations=)` and `Dvbs2.max_iterations` (`iterations`: both stop on a
  codeword).
- Closed forms named `theory_*`: `theory_cm_capacity`, `theory_bicm_capacity`,
  `theory_cm_cutoff_rate`, `theory_bicm_cutoff_rate`, `theory_biawgn_capacity`,
  `theory_normal_approximation`, `theory_shannon_limit_ebn0_db`,
  `theory_ergodic_capacity`, `theory_outage_probability`, `theory_outage_capacity`,
  `theory_css_ser`, `theory_css_ber`, `theory_css_ser_interference`, `theory_msk_psd`,
  `theory_sdr_gain_limited`, `theory_sdr_optimum_correction`,
  `theory_differential_phase_tail` and `theory_ce_ofdm_ber_optimum`; arrays for the SNR
  of `theory_css_ser_interference` and `theory_outage_probability` (the rate a number),
  for the frequency of `theory_jitter_snr_db` and `zero_order_hold_gain` and for
  `enob_from_sinad_db`; the linear SNRs `theory_phase_pdf(snr_lin)` and
  `theory_differential_phase_tail(esn0_lin)`.
- Stream-major arrays: `mimo_detect`, `mimo_llr` and `mimo_estimate` take the received
  streams `(n_rx, n)` and return `(n_tx, n)` (the LLRs `(n_tx, n · log2 M)`, the LLR
  sequence of every stream); the dual-polarisation fields of `ManakovPmd` and
  `ManakovSsfm` (`propagate`, `backpropagate`, `backpropagate_with_pmd`,
  `equalize_pmd`) are `(2, n)` arrays.
- One return type per function: `Ofdm.modulate(data)` and
  `Ofdm.modulate_with(data, reduction)` (the samples and the side information),
  `Ofdm.demodulate(samples)` and `Ofdm.demodulate_with(samples, reduction, side=)`;
  `papr_db(x)` (a float) and `papr_db_blocks(x, block_len)`; `mimo_channel` (one
  matrix) and `mimo_channels(n, n_rx, n_tx, rng)` (was `mimo_channel(n=)`);
  `mimo_capacity` an array `(n,)`; `ChannelTaps.frequency_response` always
  `(n_bins, n_rx, n_tx)`; `read_sigmf` the first channel by default (1-D),
  `channel=None` for every channel `(n_channels, n)`; `ManakovSsfm.propagate` the field
  only, its number of steps in `ManakovSsfm.last_steps`.
- The 5G NR and OFDM grids: `(n_subcarriers, n_symbols)` arrays with `(k, l)` pairs
  for the resource elements: `nr_ssb_indices` and the `indices` of `nr_pdsch_dmrs`
  (were the linear indices `l · K + k`), `ResourceGrid.map` (the grid), `extract` and
  `ofdm_estimate` (the grid, or the symbols of `Ofdm.demodulate`), `Ofdm.modulate` and
  `modulate_with` (a grid `(data_len, n_blocks)` too); the key
  `subcarrier_spacing_hz` of `nr_numerology`.
- The generator last: `GammaGamma.draw(n, rng)`, `PointingError.draw(n, rng)`,
  `Fso.gains(n, rng)`; an optional generator by keyword: `Ssfm.propagate(x, fs, *,
  precision, rng)`, `ManakovPmd.propagate(u, fs, *, rng)`,
  `ManakovSsfm.propagate(u, fs, *, rng)`.
- Keyword-only options: `fading` of the CM and BICM rates, `branches` of
  `theory_ber_rayleigh_mrc`, `rho2` and `sigma_ratio` of `theory_ber_imperfect_csi`,
  `lag` and `fs` of `cyclic_autocorrelation`, `fs` of `psd` and `periodogram`, `n_taps`
  of `cd_savory_taps`, `pulse` of `Cpm`, `approximation` of `theory_css_ser`, `pbch` of
  `nr_ssb_grid`, `iterations` of `message_passing`, `receiver` of
  `CeOfdm.subcarrier_snr` and `ber_receiver`, the arguments of `Chain.run` and
  `Chain.to_spec` but the grid (they were in two orders), `ax` of
  `comnum.plot.constellation`, `eye`, `time` and `psd`.
- Units in the names: `aclr_db(bandwidth_hz, offset_hz)`, `delay_fft(delay_s)`,
  `freq_shift(offset_hz)`, `gn_closed_form(spacing_hz)`, `wdm_grid(spacing_hz)`,
  `estimate_symbol_rate(f_min_hz, f_max_hz)`, `theory_jitter_snr_db(f_hz)`,
  `zero_order_hold_gain(f_hz)`, `write_sigmf(fc_hz=)`.
- `comnum.sim.Stop.to_spec()`, the `stop` mapping of a rule (`comnum_spec::stop_mapping`):
  `Chain.run` given a `Stop` runs a native chain on every core when a specification can
  write the rule. `comnum.sim.grid` expands its axes with `comnum_spec::grid_conditions`
  instead of a specification built for the purpose.
- Every fallible binding documents its exceptions (`Raises:`), those of the runner and
  of the specifications included; the checks that the crates make are no longer
  repeated in the bindings (`ddpll` pilots, `phase_noise` rate, `ResourceGrid.extract`).
- The palette has one name per colour: `COBALT` (`BLUE`) and `GREY_WARM` (`GREY`) are
  removed.

### Waveforms and hardware

- Verb traits: CE-OFDM (its receiver a field) and FBMC (the OQAM staggering inside)
  implement `Modulate` and `Demodulate`; every modem (`Ofdm`, `Ufmc`, `Otfs`, `CeOfdm`,
  `Fbmc`, `Css`, `Cpm`) carries its `sample_rate` (1 Hz by default) and
  `demodulate_into` refuses an input at another rate.
- Waveforms of symbol indices (ADR 0012): the container `Labels` (streams × indices),
  the traits `Modulate<In = Symbols>` and `Demodulate<Out = Symbols>`; `Css` (the field
  `sample_rate`) and `Cpm` (`Cpm::with_sample_rate`, `Cpm::sample_rate`) implement
  `Modulate<Labels>` and `Demodulate<Labels>` (the noncoherent decisions of CSS, the
  maximum-likelihood sequence detector of CPM), bit for bit their slice methods.
- `Demap` with priors and the Jacobian: `Demapping { log_priors, jacobian }` and the
  required method `Demap::demap_with_into` (`demap_into`, `demap_with`, `demap`
  provided); `Constellation` implements it on the kernel of `llr_into`, allocation-free
  up to `demapper::STACK_POINTS` points, and the specification block `demap` calls it
  (one variance per symbol keeps `llr_with_variances`).
- Placement: `WdmTransmitter` moves to `comnum_systems::wdm`; `wdm_grid` and
  `wdm_demultiplex` to `comnum_dsp::signal`; `Path` to `comnum_core::channel` (re-exported by
  `comnum_channel::multipath`); the envelope limiter to `comnum_dsp::nonlinear::limit_envelope`
  and the static predistorter to `comnum_dsp::compensation::dpd::static_predistort`
  (`Rapp::predistort` calls it).
- One kernel each: `Laser::carrier` (the laser and the WDM multiplex), `phase_noise` on
  `increment_sd`, `shot_noise_variance` and `thermal_noise_variance`,
  `ResourceGrid::data_indices`, the OFDM link on `Ofdm::modulate`/`demodulate`,
  `comnum_math::argmax`, `comnum_dsp::filter::{normalize_energy, response_at}`,
  `Signal::map_streams_in_place` for the in-place `Impair` blocks.
- `Otfs::effective_channel(&[Path])` returns a sparse `EffectiveChannel` (P entries per
  row; `to_dense` for the detectors that take a matrix).
- New: `nr::Carrier` with `demodulate`; `Photodiode::detect_signal` and
  `coherent_receiver_signal` on a `Signal`; `Rapp::bussgang`, `Saleh::bussgang`.
- Checks instead of NaN, panics or silent changes: `ResourceGrid::check` (a spacing of
  zero), `PowerDelayProfile::new`, `Css::lora`, `Rapp`/`Saleh`/`Mzm::check`, the
  laser, the phase noise, `ResourceGrid::extract`, `Cpm::min_distance2` (bounded search);
  the length mismatches of NR return `Error::Shape`.
- Numerics: the chirps of CSS are tabulated (within 10⁻¹¹ of the direct evaluation,
  closer to the exact phase; the error counts of the validation links are unchanged);
  `Cpm::min_distance2` integrates by Simpson's rule (within 10⁻¹¹ of the closed form of
  CPFSK, 4·10⁻⁶ before); `Laser::field` at zero linewidth no longer draws its null
  phase increments (its random stream changes there only).
- Migration: `Css::lora`, `PowerDelayProfile::{new, uniform}`, `Laser::field`,
  `phase_noise`, `increment_sd`, `Rapp::{amplify, predistort}`, `Saleh::amplify`,
  `ResourceGrid::extract` and `Cpm::min_distance2` return `Result`;
  `nr::modulate(nr, grid, k, n_fft, first)` is `nr::Carrier { numerology: nr,
  n_subcarriers: k, n_fft, first_symbol: first }.modulate(grid)`;
  `Otfs::effective_channel(&tuples, fs)` is `Otfs { sample_rate: Hertz(fs), .. }
  .effective_channel(&paths)?.to_dense()`; a full literal of `Ofdm`, `Ufmc`, `Otfs` or
  `CeOfdm` adds `sample_rate` (and `receiver` for `CeOfdm`), or starts from `new` (from
  `Ufmc::default` for UFMC); the
  fields of `PowerDelayProfile` are getters; `comnum_hardware::optical::{WdmTransmitter,
  wdm_grid, wdm_demultiplex}` are `comnum_systems::wdm::WdmTransmitter` and
  `comnum_dsp::signal::{wdm_grid, wdm_demultiplex}`.

### Documentation

- The library benchmark has a GPU mode (`run_all.py --threads gpu`, Sionna on the
  CUDA device, the other libraries on the CPU; `--results` writes another machine's
  results in a subdirectory of `results/`), `benchmarks/libraries/runpod.sh` runs it on
  a RunPod pod without compiling anything there (comnum from PyPI at the version of the
  checkout, or a wheel built on the laptop by `just runpod-wheel`, or `--source`;
  Sionna and the few packages of `requirements-gpu.txt`, the PyTorch of the image
  kept), packs the results, and `just runpod-fetch` brings them back and renders the
  benchmarks page, which renders every result set, the machine's GPU recorded next to
  its CPU.
- The README is a landing page: install, three examples with their figures (a coded
  link on the parallel runner, an optical link, a measured LTE signal from a HackRF
  capture; `tools/readme_figures.py` regenerates them), what the library covers, the
  evidence, the positioning and the stability policy. Its reference material moved
  to the documentation: the specifications of the repository and the operating points
  (user guide, *Specifications*), the receiver posteriors (*Analysing a campaign*),
  the deviations from OptiCommPy (*Comparisons*), the validation cards and registries
  (the new page *Why the results can be trusted*), the test suites and the good first
  contributions (`CONTRIBUTING.md`). The crate table lives in `docs/ARCHITECTURE.md`.
- The documentation index follows the four kinds of pages of Diátaxis (tutorials,
  guides, explanations, reference); a seventh tutorial, *A measured signal*, takes a
  capture through the front end, the LTE cell search and the FM receiver.
- `docs/ARCHITECTURE.md`, *API stability*: the contract of the Python API and of the
  specifications from 0.2.0; CI runs `cargo-semver-checks` on every pull request.
- The Python documentation (Sphinx): getting started, six tutorials, a user guide (with
  pages on the waveforms beyond OFDM and on optical wireless links), a
  gallery of classical results with a section that reproduces named figures of the
  literature, the API reference and the maturity table.
- The Rust API on docs.rs, every formula in LaTeX; `docs/API.md`, the generated index
  of every public item; `docs/ARCHITECTURE.md` and the ADRs.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`,
  `THIRD_PARTY.md`.
- CI on Linux, macOS and Windows (`just check` runs its checks); a release workflow
  with a TestPyPI dry run; the long validations and the benchmarks in a workflow
  started by hand.
