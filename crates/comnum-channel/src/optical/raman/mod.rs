//! Distributed Raman amplification and inter-channel stimulated Raman scattering
//! (ISRS): the power profiles of the signals and of any number of co- and
//! counter-propagating pumps along a span, and the spontaneous Raman noise.
//!
//! # Model
//!
//! Every wave $k$ (a signal, a slice of the spectrum of a field, or a pump) of frequency
//! $f_{k}$ and power $P_{k}(z)$ obeys the coupled power equations of stimulated Raman
//! scattering (Bromage 2004, Eq. (1); Agrawal, *Nonlinear Fiber Optics*, ch. 8)
//!
//! $$
//! \pm \frac{d P_{k}}{d z} = P_{k} \left(-\alpha_{k} + \sum_{j} C_{kj} P_{j}\right),
//! $$
//!
//! with $+$ for the waves that travel with the signals and $-$ for the
//! counter-propagating pumps (the Raman gain of silica is the same in both directions).
//! The coefficient of a wave $k$ from a wave $j$ of higher frequency is the gain
//! efficiency of the fibre
//!
//! $$
//! C_{kj} = \frac{g_{R}(f_{j} - f_{k})}{A_{kj}} \frac{f_{j}}{f_{\mathrm{ref}}},
//! \qquad A_{kj} = \frac{A_{\mathrm{eff}}(f_{k}) + A_{\mathrm{eff}}(f_{j})}{2},
//! \qquad f_{j} > f_{k},
//! $$
//!
//! the measured gain spectrum $g_{R}$ of a pump at $f_{\mathrm{ref}}$ ([`RamanGain`])
//! scaled with the pump frequency, and the wave of higher frequency loses the photons
//! that the other gains: $C_{jk} = -(f_{j}/f_{k}) C_{kj}$ (photon-number
//! conservation; without loss, $\sum_{k} \pm P_{k}/f_{k}$ is constant along $z$).
//!
//! # Frequency-dependent loss and effective area
//!
//! Over the S, C and L bands the attenuation and the effective area of a fibre change
//! with the wavelength, and a single value of each leaves out the tilt that they give
//! over a span. [`Raman::alpha_spectrum_db_km`] gives the
//! attenuation $\alpha_{k} = \alpha(f_{k})$ of every wave (the signals and the pumps
//! without their own attenuation) and [`Raman::aeff_spectrum_um2`] the effective area,
//! both as tables of the frequency, linear between their points and constant beyond.
//! Two waves couple through the overlap area $A_{kj}$ above, the mean of their effective
//! areas: the overlap integral $\left(\int I_{k} I_{j}\, dA\right)^{-1}$ of two
//! Gaussian modes of radii $w_{k}$ and $w_{j}$ is $\pi (w_{k}^{2} + w_{j}^{2})/2$, and
//! $A_{\mathrm{eff}} = \pi w^{2}$ for each (the `effective_area_overlap` of GNPy). With
//! empty tables every wave has the attenuation of the fibre and the area
//! [`Raman::aeff_um2`]. The profile carries the loss of every frequency to the
//! split-step solvers and to the power profiles of the generalised GN model; the closed
//! forms of the GN model keep the attenuation of the fibre.
//!
//! The co-propagating waves are known at $z = 0$ and the counter-propagating pumps at
//! $z = L$: a two-point boundary-value problem. [`Raman::solve`] integrates the
//! logarithms of the powers by the classical fourth-order Runge-Kutta method on a
//! uniform grid, forward for the co-propagating waves with the counter-propagating
//! powers of the previous sweep (linear between the nodes, so that the scheme is of
//! second order when both directions are present: $10^{-6}$ dB with the default step
//! of 100 m), then backward for the counter-propagating ones, until
//! the logarithms change by less than a tolerance between sweeps (the relaxation
//! method of the multi-pump design literature; Perlin & Winful 2002), with
//! under-relaxation if the sweeps oscillate.
//!
//! # Spontaneous emission
//!
//! The spontaneous Raman scattering of the pumps adds to signal $k$ a noise of power
//! spectral density, both polarisations,
//!
//! $$
//! N_{k}(L) = \int_{0}^{L} 2 h f_{k} \sum_{j \in \mathrm{pumps}} C_{kj} P_{j}(z)\\,
//!   (1 + \eta_{kj})\\, \frac{G_{k}(L)}{G_{k}(z)}\\, dz,
//! \qquad \eta_{kj} = \frac{1}{e^{h (f_{j} - f_{k}) / (k_{B} T)} - 1},
//! $$
//!
//! with $G_{k}(z) = P_{k}(z)/P_{k}(0)$ and $\eta$ the phonon occupancy at the
//! temperature $T$ (Bromage 2004, Eq. (4); the spontaneous term of GNPy's
//! `RamanSolver`). As in GNPy, the spontaneous scattering of the signals themselves
//! (Stokes and anti-Stokes noise between channels, about 10 % of the noise of the
//! pumps for 40 channels of 1 mW) is neglected.
//!
//! # Double Rayleigh backscattering
//!
//! A fraction $\kappa = \alpha_{R} S$ (1/km) of the signal is backscattered per unit
//! length by Rayleigh scattering and recaptured by the mode; the light scattered twice
//! travels with the signal again, amplified three times by the pumped fibre (Bromage
//! 2004, Sec. V-A, eqs. (13)–(15)). With the gain profile $G_{k}(z)$ of signal $k$ the
//! doubly backscattered power at the end of the span is the closed form
//!
//! $$
//! R_{k}^{2} = \frac{P_{\mathrm{DRB}}(L)}{P_{k}(L)} = \kappa^{2} \int_{0}^{L}
//!   \frac{1}{G_{k}(z)^{2}} \int_{z}^{L} G_{k}(z')^{2}\\, dz'\\, dz,
//! $$
//!
//! the effective Rayleigh reflection coefficient (Islam 2002, Fig. 6(b)); without pump,
//! $\kappa^{2} [L - (1 - e^{-2 \alpha L})/(2 \alpha)]/(2 \alpha)$. It is computed when
//! [`Raman::rayleigh_db_km`] gives $\kappa$ ([`RamanProfile::double_rayleigh`]) and
//! then counted as a noise by the links: the part co-polarised with the signal,
//! $(5/9) R_{k}^{2} P_{k}$ after the birefringence of the fibre has depolarised the
//! scattered light (Bromage 2004, Sec. V-A and eq. (16)), the multiple-path interference
//! (MPI) of the split-step solvers (a Gaussian noise of that power over the slice) and
//! of the SNR of [`super::gn::GnModel`]. Typical $\kappa$: $10^{-4}$ km⁻¹ (−40 dB/km) for
//! transmission fibres, up to $5 \cdot 10^{-4}$ for dispersion-compensating fibre.
//!
//! # Use in the links
//!
//! [`super::fiber::Ssfm`], [`super::manakov::ManakovSsfm`] and [`super::gn::GnModel`]
//! take a [`Raman`] configuration: the split-step solvers apply the profile of
//! every span as a gain that depends on the position and on the frequency (the
//! spectrum of the field is cut into slices of [`Raman::slice_hz`] whose average powers
//! are the waves of the equations, the quasi-static treatment of ISRS in split-step
//! simulations), add the spontaneous noise at the end of the span and, after it, the
//! amplifier restores every frequency to its launch power (an EDFA with an ideal
//! gain-flattening filter, which adds its ASE only where the Raman gain did not
//! compensate the loss). Without pumps and with [`Raman::isrs`], the profile is the
//! tilt of the spectrum by ISRS alone.
//!
//! # Examples
//! ```
//! use comnum_channel::optical::fiber::FiberParams;
//! use comnum_channel::optical::raman::{Direction, Raman, RamanPump};
//! // a counter-propagating pump at 1450 nm of 400 mW in 100 km of standard fibre
//! let fiber = FiberParams { span_km: 100.0, ..FiberParams::default() };
//! let raman = Raman {
//!     pumps: vec![RamanPump::at_wavelength(1450.0, 0.4, Direction::Counter)],
//!     ..Raman::default()
//! };
//! let signal = [(fiber.fc_hz, 1e-3)];
//! let p = raman.solve(&fiber, &signal)?;
//! // the pump compensates 13 of the 20 dB of loss of the span
//! assert!((p.net_gain_db(0) + 20.0 - p.on_off_gain_db(0) ).abs() < 1e-9);
//! assert!(p.on_off_gain_db(0) > 10.0 && p.on_off_gain_db(0) < 16.0);
//! assert!(p.ase_psd[0] > 0.0);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. Bromage, "Raman amplification for fiber communications systems," *J. Lightw.
//!    Technol.*, 22(1), 79–93, 2004.
//! 2. V. E. Perlin and H. G. Winful, "Optimal design of flat-gain wide-band fiber Raman
//!    amplifiers," *J. Lightw. Technol.*, 20(2), 250–254, 2002.
//! 3. G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013, ch. 8.
//! 4. GNPy (Telecom Infra Project), `gnpy.core.science_utils.RamanSolver` and the gain
//!    spectrum of standard single-mode fibre of `gnpy.core.parameters`.
//! 5. D. Semrau, R. I. Killey and P. Bayvel, "A closed-form approximation of the
//!    Gaussian noise model in the presence of inter-channel stimulated Raman
//!    scattering," *J. Lightw. Technol.*, 37(9), 1924–1936, 2019.

mod design;
mod gain;
mod solver;

pub use design::{DESIGN_MAX_ITERATIONS, DESIGN_TOLERANCE, DesignObjective, PumpDesign};
pub use gain::RamanGain;

use super::edfa::ase_density;
use super::fiber::{Amplification, FiberParams};
use comnum_core::Result;
use comnum_core::rng::Rng;
use comnum_math::{C64, PLANCK, db2lin, fft, lin2db, nm2hz, np2db};

/// The direction of propagation of a pump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// With the signals (launched at $z = 0$).
    Co,
    /// Against the signals (launched at $z = L$).
    Counter,
}

/// A Raman pump.
#[derive(Clone, Debug, PartialEq)]
pub struct RamanPump {
    /// Frequency (Hz).
    pub frequency_hz: f64,
    /// Power launched into the fibre at its end of the span (W).
    pub power_w: f64,
    /// Direction of propagation.
    pub direction: Direction,
    /// Attenuation at the pump wavelength (dB/km); that of the fibre at the pump
    /// frequency if `None` ([`Raman::alpha_db_km_at`]).
    pub alpha_db_km: Option<f64>,
}

/// The fraction of the doubly backscattered light that is co-polarised with the signal
/// (Bromage 2004, Sec. V-A).
pub const MPI_COPOLARISED_FRACTION: f64 = 5.0 / 9.0;

impl RamanPump {
    /// A pump of `power_w` at the vacuum wavelength `wavelength_nm`, with the
    /// attenuation of the fibre.
    pub fn at_wavelength(wavelength_nm: f64, power_w: f64, direction: Direction) -> RamanPump {
        RamanPump { frequency_hz: nm2hz(wavelength_nm), power_w, direction, alpha_db_km: None }
    }
}

/// The Raman amplification and ISRS of the spans of a link (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Raman {
    /// The pumps of every span.
    pub pumps: Vec<RamanPump>,
    /// The gain spectrum of the fibre.
    pub gain: RamanGain,
    /// Effective area of the fibre (µm²), at every frequency when
    /// [`Raman::aeff_spectrum_um2`] is empty.
    pub aeff_um2: f64,
    /// The effective area against frequency: `(frequency (Hz), area (µm²))` points of
    /// increasing frequency, linear between them and constant beyond; empty: `aeff_um2`
    /// at every frequency. Two waves couple through the mean of their areas (see the
    /// module).
    pub aeff_spectrum_um2: Vec<(f64, f64)>,
    /// The attenuation of the fibre against frequency: `(frequency (Hz), attenuation
    /// (dB/km))` points of increasing frequency, linear between them and constant beyond,
    /// the loss of the signals and of the pumps without their own `alpha_db_km`; empty:
    /// the attenuation of the fibre ([`FiberParams::alpha_db_km`]) at every frequency.
    pub alpha_spectrum_db_km: Vec<(f64, f64)>,
    /// Temperature of the fibre (K), for the spontaneous emission.
    pub temperature_k: f64,
    /// The Raman interaction between the signals (ISRS); without it the signals only
    /// interact with the pumps (the pumps always interact with each other).
    pub isrs: bool,
    /// Step of the integration grid (km).
    pub step_km: f64,
    /// Convergence tolerance of the sweeps: the largest change of $\ln P$.
    pub tolerance: f64,
    /// Largest number of forward-backward sweeps.
    pub max_iterations: usize,
    /// Width of the slices of the spectrum of a field (Hz) that are the signal waves of
    /// the split-step solvers.
    pub slice_hz: f64,
    /// The Rayleigh backscatter coupling of the signals $\kappa = \alpha_{R} S$, as
    /// $10 \log_{10}$ of its value in 1/km (about −40 dB/km for transmission fibres,
    /// −33 dB/km for dispersion-compensating fibre); `None` leaves double Rayleigh
    /// backscattering out (see the module).
    pub rayleigh_db_km: Option<f64>,
}

impl Default for Raman {
    /// No pump, the gain of standard single-mode fibre, 80 µm² and the attenuation of the
    /// fibre at every frequency, 300 K, ISRS, steps of
    /// 100 m, tolerance $10^{-10}$, at most 500 sweeps, slices of 25 GHz, no double
    /// Rayleigh backscattering.
    fn default() -> Self {
        Raman {
            pumps: Vec::new(),
            gain: RamanGain::ssmf(),
            aeff_um2: 80.0,
            aeff_spectrum_um2: Vec::new(),
            alpha_spectrum_db_km: Vec::new(),
            temperature_k: 300.0,
            isrs: true,
            step_km: 0.1,
            tolerance: 1e-10,
            max_iterations: 500,
            slice_hz: 25e9,
            rayleigh_db_km: None,
        }
    }
}

/// The power profiles of a span (see [`Raman::solve`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RamanProfile {
    /// The positions of the grid (km), from 0 to the span length.
    pub z_km: Vec<f64>,
    /// The frequency of every wave (Hz): the signals in their order, then the pumps.
    pub freqs_hz: Vec<f64>,
    /// The direction of every wave.
    pub directions: Vec<Direction>,
    /// The attenuation of every wave (Np/km).
    pub alpha_np_km: Vec<f64>,
    /// The number of signals.
    pub n_signals: usize,
    /// $\ln P_{k}(z)$ (W) of every wave on the grid.
    pub log_power: Vec<Vec<f64>>,
    /// The spontaneous Raman noise of every signal at the end of the span, both
    /// polarisations (W/Hz).
    pub ase_psd: Vec<f64>,
    /// The spontaneous Raman noise emitted towards $-z$ at the frequency of every
    /// signal, at the start of the span, both polarisations (W/Hz): $\int_{0}^{L} 2 h f
    /// \sum_{j} C_{kj} P_{j}(z) (1 + \eta_{kj}) G_{k}(z)\\, dz$, the backward ASE of
    /// Bromage 2004, eq. (6), which a counter-pumped span sends back to its input.
    pub ase_backward_psd: Vec<f64>,
    /// The effective Rayleigh reflection coefficient $R_{k}^{2}$ of every signal, the
    /// doubly backscattered power over the signal power at the end of the span (see the
    /// module); 0 without [`Raman::rayleigh_db_km`].
    pub double_rayleigh: Vec<f64>,
    /// The number of forward-backward sweeps.
    pub iterations: usize,
}

impl RamanProfile {
    /// The power of wave `k` along the span (W).
    pub fn power_w(&self, k: usize) -> Vec<f64> {
        self.log_power[k].iter().map(|l| l.exp()).collect()
    }

    /// The power of wave `k` along the span (dBm).
    pub fn power_dbm(&self, k: usize) -> Vec<f64> {
        self.log_power[k].iter().map(|l| 10.0 * std::f64::consts::LOG10_E * l + 30.0).collect()
    }

    /// The net gain of signal `k` over the span, $P(L)/P(0)$ (dB).
    pub fn net_gain_db(&self, k: usize) -> f64 {
        let l = &self.log_power[k];
        np2db(l[l.len() - 1] - l[0])
    }

    /// The on-off gain of signal `k`: the net gain with the pumps and ISRS against the
    /// loss alone (dB).
    pub fn on_off_gain_db(&self, k: usize) -> f64 {
        self.net_gain_db(k) + self.span_loss_db(k)
    }

    /// The loss of wave `k` over the span without pumps, $\alpha_{k} L$ (dB).
    pub fn span_loss_db(&self, k: usize) -> f64 {
        np2db(self.alpha_np_km[k]) * self.z_km[self.z_km.len() - 1]
    }

    /// The noise figure of the span for signal `k` (dB), signal-spontaneous beating and
    /// shot noise (Bromage 2004, eq. (9)): $\mathrm{NF} = 2 P_{A}^{+}(L)/(h \nu G B) +
    /// 1/G$ with $P_{A}^{+}$ the spontaneous noise of one polarisation (half of
    /// [`RamanProfile::ase_psd`]) and $G$ the net gain. For a lossless, undepleted
    /// amplifier at 0 K it is $2 - 1/G$, the fully inverted amplifier of Islam 2002,
    /// Sec. II-A (3 dB at high gain). The multiple-path interference is not counted.
    pub fn noise_figure_db(&self, k: usize) -> f64 {
        let g = db2lin(self.net_gain_db(k));
        lin2db(self.ase_psd[k] / (PLANCK * self.freqs_hz[k] * g) + 1.0 / g)
    }

    /// The effective noise figure of the span for signal `k` (dB): the noise figure of
    /// a discrete amplifier of the on-off gain that, after the unpumped span, would give
    /// the same noise, $\mathrm{NF}_{\mathrm{span}} - \alpha_{k} L$ in dB (Bromage 2004,
    /// eq. (12)); negative for a pumped span.
    pub fn effective_noise_figure_db(&self, k: usize) -> f64 {
        self.noise_figure_db(k) - self.span_loss_db(k)
    }

    /// The integrated power of wave `k` over the span, $I_{k} = \int_{0}^{L} P_{k}\\, dz$
    /// (W·km), by the trapezoidal rule on the grid: the quantity of the gain identity
    /// of Hu, Marks & Menyuk 2004, eq. (2), $\ln G_{k} = -\alpha_{k} L + \sum_{j} C_{kj}
    /// I_{j}$.
    pub fn integral_w_km(&self, k: usize) -> f64 {
        let w = self.power_w(k);
        let h = self.z_km[self.z_km.len() - 1] / (w.len() - 1) as f64;
        h * (w.iter().sum::<f64>() - (w[0] + w[w.len() - 1]) / 2.0)
    }

    /// The effective Rayleigh reflection coefficient $R_{k}^{2}$ of signal `k` for a
    /// Rayleigh backscatter coupling of `rayleigh_db_km` ($10 \log_{10} \kappa$, 1/km):
    /// $\kappa^{2} \int_{0}^{L} G^{-2}(z) \int_{z}^{L} G^{2}(z')\\, dz'\\, dz$ by the
    /// trapezoidal rule on the grid (the inner integral accumulated from the end of the
    /// span). See the module.
    pub fn double_rayleigh_at(&self, k: usize, rayleigh_db_km: f64) -> f64 {
        let l = &self.log_power[k];
        let m = l.len();
        let h = self.z_km[m - 1] / (m - 1) as f64;
        let g2: Vec<f64> = l.iter().map(|v| (2.0 * (v - l[0])).exp()).collect();
        let mut inner = vec![0.0; m];
        for i in (0..m - 1).rev() {
            inner[i] = inner[i + 1] + h * (g2[i] + g2[i + 1]) / 2.0;
        }
        let f: Vec<f64> = (0..m).map(|i| inner[i] / g2[i]).collect();
        let outer = h * (f.iter().sum::<f64>() - (f[0] + f[m - 1]) / 2.0);
        let kappa = db2lin(rayleigh_db_km);
        kappa * kappa * outer
    }

    /// The multiple-path interference of every signal at the end of the span, both
    /// polarisations, as a power spectral density over `bandwidth_hz` (W/Hz): the
    /// co-polarised fraction [`MPI_COPOLARISED_FRACTION`] of $R_{k}^{2} P_{k}(L)$.
    pub fn mpi_psd(&self, bandwidth_hz: f64) -> Vec<f64> {
        (0..self.n_signals)
            .map(|k| {
                let l = &self.log_power[k];
                MPI_COPOLARISED_FRACTION * self.double_rayleigh[k] * l[l.len() - 1].exp() / bandwidth_hz
            })
            .collect()
    }

    /// $\ln (P_{k}(z)/P_{k}(0))$ of signal `k` at `z` (km), linear between the nodes.
    pub fn log_gain(&self, k: usize, z: f64) -> f64 {
        let (i, t) = self.node(z);
        let l = &self.log_power[k];
        let a = l[i] + t * (l[(i + 1).min(l.len() - 1)] - l[i]);
        a - l[0]
    }

    fn node(&self, z: f64) -> (usize, f64) {
        let m = self.z_km.len() - 1;
        let h = self.z_km[m] / m as f64;
        let x = (z / h).clamp(0.0, m as f64);
        let i = (x.floor() as usize).min(m.saturating_sub(1));
        (i, x - i as f64)
    }

    /// The weights that interpolate the signals linearly in frequency at `freqs_hz`
    /// (constant beyond the extreme signals).
    pub(crate) fn interpolator(&self, freqs_hz: &[f64]) -> SpectralInterp {
        let mut order: Vec<usize> = (0..self.n_signals).collect();
        order.sort_by(|a, b| self.freqs_hz[*a].total_cmp(&self.freqs_hz[*b]));
        let fs: Vec<f64> = order.iter().map(|k| self.freqs_hz[*k]).collect();
        let weights = freqs_hz
            .iter()
            .map(|f| {
                if fs.len() == 1 || *f <= fs[0] {
                    return (order[0], order[0], 0.0);
                }
                if *f >= fs[fs.len() - 1] {
                    let k = order[fs.len() - 1];
                    return (k, k, 0.0);
                }
                let i = fs.partition_point(|v| v <= f).clamp(1, fs.len() - 1);
                (order[i - 1], order[i], (f - fs[i - 1]) / (fs[i] - fs[i - 1]))
            })
            .collect();
        SpectralInterp { weights }
    }

    /// $\ln (P(z)/P(0))$ at `z` of every frequency of `w`, into `out`.
    pub(crate) fn log_gain_at(&self, w: &SpectralInterp, z: f64, out: &mut [f64]) {
        let g: Vec<f64> = (0..self.n_signals).map(|k| self.log_gain(k, z)).collect();
        for (o, (a, b, t)) in out.iter_mut().zip(&w.weights) {
            *o = g[*a] + t * (g[*b] - g[*a]);
        }
    }

    /// The spontaneous noise at the end of the span of every frequency of `w`, both
    /// polarisations (W/Hz).
    pub(crate) fn ase_at(&self, w: &SpectralInterp) -> Vec<f64> {
        Self::interpolate(&self.ase_psd, w)
    }

    /// The values `v` of the signals at every frequency of `w`.
    fn interpolate(v: &[f64], w: &SpectralInterp) -> Vec<f64> {
        w.weights.iter().map(|(a, b, t)| v[*a] + t * (v[*b] - v[*a])).collect()
    }
}

/// Linear interpolation over the signals of a [`RamanProfile`] at given frequencies.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpectralInterp {
    weights: Vec<(usize, usize, f64)>,
}

/// One span of a split-step simulation with Raman amplification: the profile of the
/// slices of the field and the gains of its frequency bins.
#[derive(Clone, Debug)]
pub(crate) struct FieldSpan {
    /// The profile of the span.
    pub(crate) profile: RamanProfile,
    interp: SpectralInterp,
    /// The absolute frequency of every bin (Hz, FFT order).
    freqs: Vec<f64>,
    length_km: f64,
    slice_hz: f64,
}

impl Raman {
    /// The coefficient $C_{kj}$ (1/(W·km)) of a wave at `f_k` from a wave at `f_j` (Hz):
    /// the gain efficiency of the fibre when `f_j > f_k`, the depletion
    /// $-(f_{k}/f_{j}) C_{jk}$ when `f_j < f_k` (photon-number conservation), 0 at equal
    /// frequencies (see the module).
    pub fn coupling(&self, f_k: f64, f_j: f64) -> f64 {
        let aeff = (self.aeff_um2_at(f_k) + self.aeff_um2_at(f_j)) / 2.0 * 1e-12;
        if f_j > f_k {
            self.gain.efficiency(f_k, f_j, aeff)
        } else if f_j < f_k {
            -(f_k / f_j) * self.gain.efficiency(f_j, f_k, aeff)
        } else {
            0.0
        }
    }

    /// The effective area at `f_hz` (µm²): [`Raman::aeff_spectrum_um2`] interpolated,
    /// or [`Raman::aeff_um2`] when it is empty.
    pub fn aeff_um2_at(&self, f_hz: f64) -> f64 {
        piecewise_linear(&self.aeff_spectrum_um2, f_hz).unwrap_or(self.aeff_um2)
    }

    /// The attenuation of the fibre at `f_hz` (dB/km): [`Raman::alpha_spectrum_db_km`]
    /// interpolated, or [`FiberParams::alpha_db_km`] when it is empty.
    pub fn alpha_db_km_at(&self, fiber: &FiberParams, f_hz: f64) -> f64 {
        piecewise_linear(&self.alpha_spectrum_db_km, f_hz).unwrap_or(fiber.alpha_db_km)
    }

    /// The slices of the spectrum of the polarisations `fields` sampled at `fs` around
    /// the carrier `fc` (Hz): the absolute centre frequency of every slice of
    /// [`Raman::slice_hz`] and its average power (W, all polarisations).
    pub(crate) fn spectral_slices(&self, fields: &[&[C64]], fs: f64, fc: f64) -> Vec<(f64, f64)> {
        let mut slices: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
        for x in fields {
            let n = x.len();
            if n == 0 {
                continue;
            }
            let spectrum = fft::fft(x);
            let nn = (n * n) as f64;
            for (v, f) in spectrum.iter().zip(fft::fftfreq(n, 1.0 / fs)) {
                *slices.entry((f / self.slice_hz).round() as i64).or_insert(0.0) += v.norm_sqr() / nn;
            }
        }
        slices.into_iter().map(|(q, p)| (fc + q as f64 * self.slice_hz, p)).collect()
    }

    /// The span of `fiber` for the polarisations `fields` sampled at `fs` (see
    /// [`FieldSpan`]).
    ///
    /// # Errors
    /// As [`Raman::solve`].
    pub(crate) fn field_span(&self, fiber: &FiberParams, fields: &[&[C64]], fs: f64) -> Result<FieldSpan> {
        let slices = self.spectral_slices(fields, fs, fiber.fc_hz);
        let profile = self.solve(fiber, &slices)?;
        let n = fields.first().map_or(0, |x| x.len());
        let freqs: Vec<f64> = fft::fftfreq(n, 1.0 / fs).iter().map(|f| fiber.fc_hz + f).collect();
        let interp = profile.interpolator(&freqs);
        Ok(FieldSpan { profile, interp, freqs, length_km: fiber.span_km, slice_hz: self.slice_hz })
    }
}

/// The value at `x` of the table `points` (increasing abscissae), linear between its
/// points and constant beyond; `None` for an empty table.
fn piecewise_linear(points: &[(f64, f64)], x: f64) -> Option<f64> {
    let (first, last) = (points.first()?, points[points.len() - 1]);
    if x <= first.0 {
        return Some(first.1);
    }
    if x >= last.0 {
        return Some(last.1);
    }
    let i = points.partition_point(|p| p.0 <= x).clamp(1, points.len() - 1);
    let ((x0, y0), (x1, y1)) = (points[i - 1], points[i]);
    Some(y0 + (x - x0) / (x1 - x0) * (y1 - y0))
}

impl FieldSpan {
    /// The gain of the field at every bin from `z0` to `z1` (km), $\sqrt{G(z_{1}, f) /
    /// G(z_{0}, f)}$: the [`comnum_math::nlse::FieldGain`] of the span.
    pub(crate) fn field_gain(&self, z0: f64, z1: f64, out: &mut [f64]) {
        let mut a = vec![0.0; out.len()];
        self.profile.log_gain_at(&self.interp, z0, &mut a);
        self.profile.log_gain_at(&self.interp, z1, out);
        for (o, a) in out.iter_mut().zip(&a) {
            *o = ((*o - a) / 2.0).exp();
        }
    }

    /// The power gain that restores every bin to its launch power at the end of the
    /// span, $1/G(L, f)$.
    pub(crate) fn restoring_gain(&self) -> Vec<f64> {
        let mut g = vec![0.0; self.freqs.len()];
        self.profile.log_gain_at(&self.interp, self.length_km, &mut g);
        g.iter().map(|l| (-l).exp()).collect()
    }

    /// The end of the span on one of the `pols` polarisations, `x`: the spontaneous
    /// Raman noise of one polarisation (half of [`RamanProfile::ase_psd`], whatever
    /// `pols`: a scalar field is one polarisation, as for the ASE of its EDFA), the
    /// multiple-path interference of double Rayleigh backscattering (a share `1/pols`
    /// of that of the slices, which follows the power of the signal), then the
    /// amplifier, and its ASE where its gain exceeds 1 (`Edfa`, noise figure `nf_db`):
    /// the noise is drawn from `rng`, none without one.
    pub(crate) fn amplify(
        &self,
        x: &mut [C64],
        fs: f64,
        pols: usize,
        amplification: &Amplification,
        nf_db: f64,
        rng: Option<&mut Rng>,
    ) {
        let n = x.len();
        if n == 0 {
            return;
        }
        let g = match amplification {
            Amplification::None => vec![1.0; n],
            Amplification::Edfa | Amplification::Ideal => self.restoring_gain(),
        };
        let p = fft::plan(n);
        p.forward(x);
        for (v, gb) in x.iter_mut().zip(&g) {
            *v *= gb.sqrt();
        }
        if let Some(rng) = rng {
            let ase = self.profile.ase_at(&self.interp);
            let mpi = RamanProfile::interpolate(&self.profile.mpi_psd(self.slice_hz), &self.interp);
            let share = 1.0 / pols as f64;
            let nf = db2lin(nf_db);
            let mut w: Vec<C64> = (0..n).map(|_| rng.complex_gaussian()).collect();
            p.forward(&mut w);
            for (b, (v, wb)) in x.iter_mut().zip(&w).enumerate() {
                let edfa = match amplification {
                    Amplification::Edfa if g[b] > 1.0 => ase_density(g[b], nf, self.freqs[b]),
                    _ => 0.0,
                };
                // two polarisations: the expression of the results recorded so far
                let raman = if pols == 2 { (ase[b] + mpi[b]) * share } else { 0.5 * ase[b] + mpi[b] * share };
                *v += wb * ((raman * g[b] + edfa) * fs).sqrt();
            }
        }
        p.inverse(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_math::nlse::NonlinearStep;
    use comnum_math::{BOLTZMANN, PLANCK, SPEED_OF_LIGHT, db2np};

    fn fiber(span_km: f64) -> FiberParams {
        FiberParams { span_km, ..FiberParams::default() }
    }

    #[test]
    fn without_pumps_and_isrs_the_profile_is_the_loss() {
        let raman = Raman { isrs: false, ..Raman::default() };
        let f = fiber(80.0);
        let p = raman.solve(&f, &[(193e12, 1e-3), (194e12, 2e-3)]).unwrap();
        for k in 0..2 {
            assert!((p.net_gain_db(k) + 16.0).abs() < 1e-9, "{}", p.net_gain_db(k));
            assert!(p.on_off_gain_db(k).abs() < 1e-9);
        }
        assert_eq!(p.ase_psd, vec![0.0, 0.0]);
        // 1 mW and 2 mW at the start of the span, 16 dB less at its end
        let dbm = p.power_dbm(1);
        assert!((dbm[0] - 10.0 * 2f64.log10()).abs() < 1e-12 && (dbm[0] - dbm[dbm.len() - 1] - 16.0).abs() < 1e-9);
    }

    #[test]
    fn an_undepleted_pump_gives_the_closed_form_gain() {
        // small signal: G_on-off = exp(C P_p L_eff,p) for a co- or a counter-pump
        let f = fiber(80.0);
        let gain = RamanGain::ssmf();
        let fp = SPEED_OF_LIGHT / 1450e-9;
        let fs = SPEED_OF_LIGHT / 1550e-9;
        let c = gain.efficiency(fs, fp, 80e-12);
        let ap = 0.25 / (10.0 * std::f64::consts::LOG10_E);
        let leff = (1.0 - (-ap * 80.0).exp()) / ap;
        for direction in [Direction::Co, Direction::Counter] {
            let pump = RamanPump { alpha_db_km: Some(0.25), ..RamanPump::at_wavelength(1450.0, 0.3, direction) };
            let raman = Raman { pumps: vec![pump], ..Raman::default() };
            let p = raman.solve(&f, &[(fs, 1e-9)]).unwrap();
            let expected = 10.0 * std::f64::consts::LOG10_E * c * 0.3 * leff;
            assert!(
                (p.on_off_gain_db(0) - expected).abs() < 1e-3,
                "{direction:?}: {} vs {expected}",
                p.on_off_gain_db(0)
            );
        }
    }

    #[test]
    fn photons_are_conserved_without_loss() {
        let f = FiberParams { alpha_db_km: 0.0, span_km: 50.0, ..FiberParams::default() };
        let pumps = vec![
            RamanPump::at_wavelength(1425.0, 0.4, Direction::Co),
            RamanPump::at_wavelength(1455.0, 0.3, Direction::Co),
        ];
        let raman = Raman { pumps, ..Raman::default() };
        let signals: Vec<(f64, f64)> = (0..8).map(|k| (190e12 + k as f64 * 0.6e12, 5e-3)).collect();
        let p = raman.solve(&f, &signals).unwrap();
        let photons =
            |i: usize| -> f64 { (0..p.freqs_hz.len()).map(|k| p.log_power[k][i].exp() / p.freqs_hz[k]).sum() };
        let last = p.z_km.len() - 1;
        assert!((photons(last) / photons(0) - 1.0).abs() < 1e-9);
        // and the pumps transferred power to the signals
        assert!(p.net_gain_db(0) > 3.0);
    }

    #[test]
    fn a_frequency_dependent_loss_is_the_loss_of_each_wave() {
        // no pump, no ISRS: every signal loses α(f) L, interpolated between the points
        let f = fiber(80.0);
        let table = vec![(186e12, 0.25), (196e12, 0.19), (206e12, 0.21)];
        let raman = Raman { isrs: false, alpha_spectrum_db_km: table, ..Raman::default() };
        let signals = [(180e12, 1e-3), (186e12, 1e-3), (191e12, 1e-3), (201e12, 1e-3), (210e12, 1e-3)];
        let p = raman.solve(&f, &signals).unwrap();
        for (k, alpha) in [0.25, 0.25, 0.22, 0.20, 0.21].iter().enumerate() {
            assert!((p.net_gain_db(k) + alpha * 80.0).abs() < 1e-9, "{k}: {}", p.net_gain_db(k));
            assert!(p.on_off_gain_db(k).abs() < 1e-9);
        }
        // a pump without its own attenuation takes that of the table
        let pump = RamanPump::at_wavelength(1455.0, 0.1, Direction::Counter);
        let q = Raman { pumps: vec![pump.clone()], ..raman.clone() }.solve(&f, &signals[..1]).unwrap();
        let expected = db2np(raman.alpha_db_km_at(&f, pump.frequency_hz));
        assert!((q.alpha_np_km[1] - expected).abs() < 1e-15);
        assert!((raman.alpha_db_km_at(&f, 191e12) - 0.22).abs() < 1e-15);
        assert!((Raman::default().alpha_db_km_at(&f, 191e12) - f.alpha_db_km).abs() < 1e-15);
    }

    #[test]
    fn two_waves_couple_through_the_mean_of_their_effective_areas() {
        let (fs, fp) = (nm2hz(1550.0), nm2hz(1450.0));
        let raman = Raman { aeff_spectrum_um2: vec![(fs, 85.0), (fp, 75.0)], ..Raman::default() };
        let gain = RamanGain::ssmf();
        assert!((raman.coupling(fs, fp) / gain.efficiency(fs, fp, 80e-12) - 1.0).abs() < 1e-12);
        assert!((raman.coupling(fp, fs) + fp / fs * gain.efficiency(fs, fp, 80e-12)).abs() < 1e-12);
        assert!((raman.aeff_um2_at(nm2hz(1500.0)) - 80.0).abs() < 0.2);
        assert!((raman.aeff_um2_at(1.0) - 85.0).abs() < 1e-12 && (raman.aeff_um2_at(1e15) - 75.0).abs() < 1e-12);
        // empty table: the single area
        let flat = Raman::default();
        assert!((flat.coupling(fs, fp) - gain.efficiency(fs, fp, 80e-12)).abs() < 1e-15);
    }

    #[test]
    fn photons_are_conserved_across_the_s_c_and_l_bands() {
        // 146 channels from 1480 to 1625 nm, an area that grows with the wavelength and
        // co- or counter-pumps from 1365 to 1485 nm, without loss: the photon flux
        // Σ± P/f is constant, to the error of the scheme (second order with counter-pumps)
        let f = FiberParams { alpha_db_km: 0.0, span_km: 80.0, ..FiberParams::default() };
        let signals: Vec<(f64, f64)> = (0..146).map(|k| (nm2hz(1625.0 - k as f64), 1e-3)).collect();
        let aeff_spectrum_um2 = vec![(nm2hz(1625.0), 86.0), (nm2hz(1365.0), 70.0)];
        for (direction, tolerance) in [(Direction::Co, 1e-9), (Direction::Counter, 1e-5)] {
            let pumps = [1365.0, 1395.0, 1425.0, 1455.0, 1485.0]
                .iter()
                .map(|w| RamanPump::at_wavelength(*w, 0.15, direction))
                .collect();
            let raman = Raman { pumps, aeff_spectrum_um2: aeff_spectrum_um2.clone(), ..Raman::default() };
            let p = raman.solve(&f, &signals).unwrap();
            let flux = |i: usize| -> f64 {
                (0..p.freqs_hz.len())
                    .map(|k| {
                        let sign = if p.directions[k] == Direction::Counter { -1.0 } else { 1.0 };
                        sign * p.log_power[k][i].exp() / p.freqs_hz[k]
                    })
                    .sum()
            };
            let last = p.z_km.len() - 1;
            let drift = ((flux(last) - flux(0)) / flux(0)).abs();
            assert!(drift < tolerance, "{direction:?}: {drift}");
            assert!(p.net_gain_db(0) > 1.0, "{direction:?}: {}", p.net_gain_db(0));
        }
    }

    #[test]
    fn invalid_spectra_are_rejected() {
        let f = fiber(80.0);
        for raman in [
            Raman { aeff_spectrum_um2: vec![(194e12, 80.0), (193e12, 80.0)], ..Raman::default() },
            Raman { aeff_spectrum_um2: vec![(193e12, 0.0)], ..Raman::default() },
            Raman { alpha_spectrum_db_km: vec![(193e12, -0.1)], ..Raman::default() },
            Raman { alpha_spectrum_db_km: vec![(193e12, f64::NAN)], ..Raman::default() },
        ] {
            assert!(raman.solve(&f, &[(193e12, 1e-3)]).is_err(), "{raman:?}");
        }
    }

    #[test]
    fn counter_pumps_meet_their_boundary_conditions_and_the_equations() {
        let f = fiber(100.0);
        let pumps = vec![
            RamanPump::at_wavelength(1424.0, 0.25, Direction::Counter),
            RamanPump::at_wavelength(1452.0, 0.2, Direction::Counter),
            RamanPump::at_wavelength(1480.0, 0.1, Direction::Co),
        ];
        let raman = Raman { pumps: pumps.clone(), ..Raman::default() };
        let signals: Vec<(f64, f64)> = (0..10).map(|k| (191e12 + k as f64 * 0.5e12, 1e-3)).collect();
        let p = raman.solve(&f, &signals).unwrap();
        let m = p.z_km.len() - 1;
        for (k, pump) in pumps.iter().enumerate() {
            let at = if pump.direction == Direction::Counter { m } else { 0 };
            assert!((p.log_power[10 + k][at].exp() / pump.power_w - 1.0).abs() < 1e-12);
        }
        // a finer grid gives the same profiles (to 10⁻⁵ dB)
        let fine = Raman { step_km: 0.025, ..raman }.solve(&f, &signals).unwrap();
        for k in 0..p.freqs_hz.len() {
            let d = (p.log_power[k][m] - fine.log_power[k][4 * m]).abs();
            assert!(d < 3e-6, "{k}: {d}");
        }
    }

    #[test]
    fn isrs_tilts_the_spectrum_towards_low_frequencies() {
        // 10 THz of signals at 1 mW per 50 GHz: the lowest channels gain from the highest
        let f = fiber(80.0);
        let raman = Raman::default();
        let signals: Vec<(f64, f64)> = (0..200).map(|k| (186e12 + k as f64 * 50e9, 1e-3)).collect();
        let p = raman.solve(&f, &signals).unwrap();
        let tilt = p.net_gain_db(0) - p.net_gain_db(199);
        // Zirngibl's triangular approximation: tilt ≈ 10 log10 e · C' P_tot L_eff B with
        // C' the slope of the gain efficiency
        let gain = RamanGain::ssmf();
        let slope = gain.efficiency(193e12, 193e12 + 5e12, 80e-12) / 5e12;
        let expected = 10.0 * std::f64::consts::LOG10_E * slope * 0.2 * f.effective_length_km() * 9.95e12;
        assert!(tilt > 0.0 && (tilt / expected - 1.0).abs() < 0.15, "{tilt} vs {expected}");
    }

    #[test]
    fn the_spontaneous_noise_matches_the_undepleted_integral() {
        let f = fiber(80.0);
        let pump = RamanPump { alpha_db_km: Some(0.25), ..RamanPump::at_wavelength(1450.0, 0.3, Direction::Counter) };
        let raman = Raman { pumps: vec![pump], ..Raman::default() };
        let fs = SPEED_OF_LIGHT / 1550e-9;
        let p = raman.solve(&f, &[(fs, 1e-9)]).unwrap();
        // independent quadrature with the closed-form profiles
        let fp = SPEED_OF_LIGHT / 1450e-9;
        let c = RamanGain::ssmf().efficiency(fs, fp, 80e-12);
        let (a, ap) = (f.alpha_np_km(), 0.25 / (10.0 * std::f64::consts::LOG10_E));
        let pp = |z: f64| 0.3 * (-ap * (80.0 - z)).exp();
        let lg = |z: f64| -a * z + c * 0.3 * ((-ap * (80.0 - z)).exp() - (-ap * 80.0).exp()) / ap;
        let eta = 1.0 / (PLANCK * (fp - fs) / (BOLTZMANN * 300.0)).exp_m1();
        let n = 80_000;
        let dz = 80.0 / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let z = (i as f64 + 0.5) * dz;
                c * pp(z) * (lg(80.0) - lg(z)).exp()
            })
            .sum::<f64>()
            * dz;
        let expected = 2.0 * PLANCK * fs * (1.0 + eta) * integral;
        assert!((p.ase_psd[0] / expected - 1.0).abs() < 1e-4, "{} vs {expected}", p.ase_psd[0]);
    }

    #[test]
    fn the_backward_spontaneous_noise_of_a_co_pumped_lossless_span_is_the_forward_one_of_a_counter_pumped_one() {
        // the mirror image: the backward noise of the co-pumped span at z = 0 equals the
        // forward noise of the counter-pumped span at z = L (same profile reversed)
        let f = FiberParams { alpha_db_km: 0.0, span_km: 30.0, ..FiberParams::default() };
        let fs = SPEED_OF_LIGHT / 1550e-9;
        let noise = |d: Direction| {
            let pump = RamanPump { alpha_db_km: Some(0.25), ..RamanPump::at_wavelength(1450.0, 0.3, d) };
            Raman { pumps: vec![pump], ..Raman::default() }.solve(&f, &[(fs, 1e-9)]).unwrap()
        };
        let (co, counter) = (noise(Direction::Co), noise(Direction::Counter));
        assert!((co.ase_backward_psd[0] / counter.ase_psd[0] - 1.0).abs() < 1e-6);
        assert!((counter.ase_backward_psd[0] / co.ase_psd[0] - 1.0).abs() < 1e-6);
        // without pump, no noise either way
        let none = Raman { isrs: false, ..Raman::default() }.solve(&f, &[(fs, 1e-3)]).unwrap();
        assert_eq!(none.ase_backward_psd, vec![0.0]);
    }

    #[test]
    fn the_relaxation_is_second_order_in_the_step_on_a_stiff_span() {
        // three strong counter-pumps and ISRS over 100 km: the error of the net gains
        // against the solution at 12.5 m, for steps of 0.8 to 0.05 km, falls by 4 per
        // halving (the counter waves are linear between the nodes of the forward sweep)
        let f = fiber(100.0);
        let pumps: Vec<RamanPump> =
            [1425.0, 1445.0, 1465.0].iter().map(|w| RamanPump::at_wavelength(*w, 0.5, Direction::Counter)).collect();
        let signals: Vec<(f64, f64)> = (0..40).map(|k| (188e12 + k as f64 * 250e9, 1e-3)).collect();
        let solve =
            |h: f64| Raman { pumps: pumps.clone(), step_km: h, ..Raman::default() }.solve(&f, &signals).unwrap();
        let reference = solve(0.0125);
        let errors: Vec<f64> = [0.8, 0.4, 0.2, 0.1, 0.05]
            .iter()
            .map(|h| {
                let p = solve(*h);
                (0..43).map(|k| (p.net_gain_db(k) - reference.net_gain_db(k)).abs()).fold(0.0, f64::max)
            })
            .collect();
        println!("errors (dB) at 0.8, 0.4, 0.2, 0.1, 0.05 km: {errors:?}");
        for w in errors.windows(2) {
            let ratio = w[0] / w[1];
            assert!(ratio > 3.0 && ratio < 5.5, "{errors:?}");
        }
        assert!(errors[3] < 1e-3, "{errors:?}");
    }

    #[test]
    fn without_pump_the_double_rayleigh_is_its_closed_form() {
        // κ²/(2α) [L − (1 − e^{−2αL})/(2α)]
        let f = fiber(80.0);
        let raman = Raman { isrs: false, rayleigh_db_km: Some(-40.0), ..Raman::default() };
        let p = raman.solve(&f, &[(193e12, 1e-3)]).unwrap();
        let (a, kappa) = (f.alpha_np_km(), 1e-4);
        let expected = kappa * kappa / (2.0 * a) * (80.0 - (1.0 - (-2.0 * a * 80.0).exp()) / (2.0 * a));
        assert!((p.double_rayleigh[0] / expected - 1.0).abs() < 2e-5, "{} vs {expected}", p.double_rayleigh[0]);
        assert!((p.double_rayleigh_at(0, -40.0) - p.double_rayleigh[0]).abs() < 1e-30);
        // none without a coefficient
        let q = Raman { rayleigh_db_km: None, ..raman }.solve(&f, &[(193e12, 1e-3)]).unwrap();
        assert_eq!(q.double_rayleigh, vec![0.0]);
    }

    #[test]
    fn the_double_rayleigh_of_a_pumped_span_matches_a_fine_double_integral() {
        let f = fiber(60.0);
        let pump = RamanPump { alpha_db_km: Some(0.25), ..RamanPump::at_wavelength(1450.0, 0.4, Direction::Counter) };
        let raman = Raman { pumps: vec![pump], isrs: false, rayleigh_db_km: Some(-40.0), ..Raman::default() };
        let fs = SPEED_OF_LIGHT / 1550e-9;
        let p = raman.solve(&f, &[(fs, 1e-9)]).unwrap();
        // the closed-form profile and a double sum on a grid of 10 m
        let c = RamanGain::ssmf().efficiency(fs, SPEED_OF_LIGHT / 1450e-9, 80e-12);
        let (a, ap) = (f.alpha_np_km(), 0.25 / (10.0 * std::f64::consts::LOG10_E));
        let lg = |z: f64| -a * z + c * 0.4 * ((-ap * (60.0 - z)).exp() - (-ap * 60.0).exp()) / ap;
        let n = 6000;
        let dz = 60.0 / n as f64;
        let g2: Vec<f64> = (0..=n).map(|i| (2.0 * lg(i as f64 * dz)).exp()).collect();
        let mut inner = vec![0.0; n + 1];
        for i in (0..n).rev() {
            inner[i] = inner[i + 1] + dz * (g2[i] + g2[i + 1]) / 2.0;
        }
        let outer: f64 = (0..=n).map(|i| inner[i] / g2[i] * if i == 0 || i == n { 0.5 } else { 1.0 }).sum::<f64>() * dz;
        let expected = 1e-8 * outer;
        assert!((p.double_rayleigh[0] / expected - 1.0).abs() < 1e-4, "{} vs {expected}", p.double_rayleigh[0]);
        assert!(p.on_off_gain_db(0) > 10.0);
    }

    #[test]
    fn the_split_step_adds_the_multiple_path_interference_of_every_slice() {
        use super::super::fiber::Ssfm;
        // a tone in a linear, lossless fibre with a strong Rayleigh coupling and no pump:
        // the noise added is (5/9) R² P (the spontaneous noise is zero without pump)
        let fiber =
            FiberParams { alpha_db_km: 0.0, gamma_w_km: 0.0, d_ps_nm_km: 0.0, span_km: 50.0, ..FiberParams::default() };
        let (fs, n) = (64e9, 1 << 15);
        let x = tones(n, fs, &[(0.0, 1e-3)]);
        let raman = Raman { isrs: false, rayleigh_db_km: Some(-20.0), ..Raman::default() };
        let ssfm = Ssfm {
            fiber: fiber.clone(),
            step_km: 5.0,
            amplification: Amplification::None,
            raman: Some(raman.clone()),
            rule: None,
            nonlinear: NonlinearStep::Midpoint,
            check_sampling: false,
        };
        let y = crate::optical::fiber::tests::propagated(&ssfm, &x, fs, Some(&mut Rng::new(5)));
        let noise = y.iter().zip(&x).map(|(u, v)| (u - v).norm_sqr()).sum::<f64>() / n as f64;
        // lossless: R² = κ² L²/2
        let r2 = 1e-4 * 50.0 * 50.0 / 2.0;
        let expected = MPI_COPOLARISED_FRACTION * r2 * 1e-3;
        assert!((noise / expected - 1.0).abs() < 0.05, "{noise} vs {expected}");
    }

    #[test]
    fn a_polarisation_gets_half_the_spontaneous_noise_whatever_their_number() {
        // a scalar field is one polarisation: half of the noise of both (it used to get
        // all of it, 3 dB more than a polarisation of the Manakov solver)
        let fiber = fiber(80.0);
        let (fs, n) = (64e9, 1 << 14);
        let x = tones(n, fs, &[(0.0, 1e-3)]);
        let raman = Raman {
            pumps: vec![RamanPump::at_wavelength(1450.0, 0.5, Direction::Counter)],
            rayleigh_db_km: None,
            ..Raman::default()
        };
        let span = raman.field_span(&fiber, &[&x], fs).unwrap();
        let ase = span.profile.ase_at(&span.interp);
        let expected = 0.5 * ase.iter().sum::<f64>() / ase.len() as f64 * fs;
        assert!(expected > 0.0);
        for pols in [1usize, 2] {
            let mut e = x.clone();
            span.amplify(&mut e, fs, pols, &Amplification::None, fiber.nf_db, Some(&mut Rng::new(9)));
            let noise = e.iter().zip(&x).map(|(u, v)| (u - v).norm_sqr()).sum::<f64>() / n as f64;
            assert!((noise / expected - 1.0).abs() < 0.05, "{pols} polarisations: {noise} vs {expected}");
        }
    }

    fn tones(n: usize, fs: f64, offsets: &[(f64, f64)]) -> Vec<C64> {
        // tones at the frequencies of the bins nearest to the offsets, of the given powers
        (0..n)
            .map(|t| {
                offsets
                    .iter()
                    .map(|(f, p)| {
                        let k = (f / fs * n as f64).round();
                        C64::from_polar(p.sqrt(), 2.0 * std::f64::consts::PI * k * t as f64 / n as f64)
                    })
                    .sum()
            })
            .collect()
    }

    fn tone_power(x: &[C64], fs: f64, f: f64) -> f64 {
        let n = x.len();
        let k = ((f / fs * n as f64).round() as i64).rem_euclid(n as i64) as usize;
        fft::fft(x)[k].norm_sqr() / (n * n) as f64
    }

    #[test]
    fn the_split_step_follows_the_profile_of_every_slice() {
        use super::super::fiber::Ssfm;
        // two tones 1.2 THz apart, linear fibre, a counter pump, no amplifier
        let fiber = FiberParams { gamma_w_km: 0.0, d_ps_nm_km: 0.0, span_km: 60.0, ..FiberParams::default() };
        let (fs, n) = (4.096e12, 4096);
        let x = tones(n, fs, &[(-0.6e12, 20e-3), (0.6e12, 20e-3)]);
        let raman = Raman {
            pumps: vec![RamanPump::at_wavelength(1450.0, 0.3, Direction::Counter)],
            slice_hz: 100e9,
            ..Raman::default()
        };
        let ssfm = Ssfm {
            fiber: fiber.clone(),
            step_km: 0.5,
            amplification: Amplification::None,
            raman: Some(raman.clone()),
            rule: None,
            nonlinear: NonlinearStep::Midpoint,
            check_sampling: false,
        };
        let y = crate::optical::fiber::tests::propagated(&ssfm, &x, fs, None);
        let p = raman.solve(&fiber, &raman.spectral_slices(&[&x], fs, fiber.fc_hz)).unwrap();
        for f in [-0.6e12, 0.6e12] {
            let k = p.freqs_hz[..p.n_signals].iter().position(|v| (v - fiber.fc_hz - f).abs() < 1.0).unwrap();
            let measured = comnum_math::lin2db(tone_power(&y, fs, f) / tone_power(&x, fs, f));
            assert!((measured - p.net_gain_db(k)).abs() < 1e-6, "{f}: {measured} vs {}", p.net_gain_db(k));
        }
    }

    #[test]
    fn without_pumps_nor_isrs_the_raman_links_are_the_edfa_links() {
        use super::super::edfa::Edfa;
        use super::super::fiber::Ssfm;
        use super::super::manakov::ManakovSsfm;
        use comnum_core::signal::DualPol;
        use comnum_math::nlse::StepRule;
        let fiber = FiberParams { n_spans: 2, span_km: 50.0, ..FiberParams::default() };
        let (fs, n) = (64e9, 1024);
        let mut r = Rng::new(3);
        let x: Vec<C64> = r.complex_noise(n, 1e-3);
        let none = Some(Raman { isrs: false, ..Raman::default() });
        let a = Ssfm { amplification: Amplification::Ideal, check_sampling: false, ..Ssfm::new(fiber.clone(), 0.5) };
        let b = Ssfm { raman: none.clone(), ..a.clone() };
        let (ya, yb) = (
            crate::optical::fiber::tests::propagated(&a, &x, fs, None),
            crate::optical::fiber::tests::propagated(&b, &x, fs, None),
        );
        let err = ya.iter().zip(&yb).map(|(u, v)| (u - v).norm()).fold(0.0, f64::max);
        assert!(err < 1e-9 * 0.03, "{err}");
        // the same steps in both: with Raman amplification the loss is in the profile, so the
        // steps of the four-wave-mixing rule would not grow with it
        let a = ManakovSsfm {
            check_sampling: false,
            amplification: Amplification::Ideal,
            rule: StepRule::NonlinearPhase(2e-2),
            ..ManakovSsfm::new(fiber.clone())
        };
        let b = ManakovSsfm { raman: none.clone(), ..a.clone() };
        let mut ua = DualPol::new(x.clone(), x.iter().map(|v| v * 0.5).collect()).unwrap();
        let mut ub = ua.clone();
        a.propagate(&mut ua, fs, None).unwrap();
        b.propagate(&mut ub, fs, None).unwrap();
        let err = ua.x.iter().zip(&ub.x).chain(ua.y.iter().zip(&ub.y)).map(|(u, v)| (u - v).norm()).fold(0.0, f64::max);
        assert!(err < 1e-9 * 0.03, "{err}");
        // and the noise of the EDFAs: the variance of the noise added to a silent field
        let silent = vec![C64::new(0.0, 0.0); 1 << 16];
        let e =
            Ssfm { raman: none, check_sampling: false, ..Ssfm::new(FiberParams { n_spans: 1, ..fiber.clone() }, 1.0) };
        let y = crate::optical::fiber::tests::propagated(&e, &silent, fs, Some(&mut Rng::new(9)));
        let var = y.iter().map(C64::norm_sqr).sum::<f64>() / y.len() as f64;
        let expected = Edfa { gain_db: 10.0, nf_db: fiber.nf_db, fc_hz: fiber.fc_hz }.noise_variance(fs);
        assert!((var / expected - 1.0).abs() < 0.02, "{var} vs {expected}");
    }
}

/// Checks against the published closed forms (`references/`).
#[cfg(test)]
mod literature {
    use super::*;
    use crate::optical::fiber::FiberParams;

    /// D. Semrau and P. Bayvel, "The Gaussian noise model in the presence of inter-channel
    /// stimulated Raman scattering," J. Lightw. Technol. 36(14), 2018 (arXiv 1801.02460),
    /// eq. (10) (p. 4): with a linear Raman gain slope `C_r`, the ISRS power transfer of
    /// a flat band of `B` after one span is `Δρ [dB] = 4.3 P_tot C_r L_eff B` between the
    /// band edges; the text reports the ODE and this closed form within 0.18 dB at
    /// 28 dBm (Fig. 1, 100 km, 0.2 dB/km). Checked at 19 and 22 dBm over 10 THz, within
    /// 5 % of the transfer plus 0.05 dB: the gain of the library scales with the pump
    /// frequency (`RamanGain::efficiency`), which the constant slope of the paper does
    /// not, and the band spans ±2.6 % of the carrier.
    #[test]
    fn semrau_2018_eq_10_isrs_power_transfer() {
        let fiber = FiberParams { span_km: 100.0, alpha_db_km: 0.2, ..FiberParams::default() };
        let c_r = 0.028; // 1/(W km THz), the fibre of the paper's Sec. IV
        let aeff_um2 = 80.0;
        // g_R A_eff (m/W) = C_r Δf A_eff / 1e3, linear up to 14 THz, zero beyond
        let gain =
            RamanGain::new(vec![0.0, 14e12], vec![0.0, c_r * 14.0 * aeff_um2 * 1e-12 / 1e3], fiber.fc_hz).unwrap();
        let raman = Raman { gain, aeff_um2, isrs: true, ..Raman::default() };
        let (n, spacing) = (101usize, 100e9);
        let l_eff = fiber.effective_length_km();
        for p_tot_dbm in [19.0, 22.0] {
            let p_ch = comnum_math::dbm2w(p_tot_dbm) / n as f64;
            let signals: Vec<(f64, f64)> = (0..n).map(|i| (fiber.fc_hz + (i as f64 - 50.0) * spacing, p_ch)).collect();
            let profile = raman.solve(&fiber, &signals).unwrap();
            let measured = profile.net_gain_db(0) - profile.net_gain_db(n - 1);
            let band_thz = (n - 1) as f64 * spacing * 1e-12;
            let paper = 4.3 * comnum_math::dbm2w(p_tot_dbm) * c_r * l_eff * band_thz;
            assert!(
                (measured - paper).abs() < 0.05 * paper + 0.05,
                "{p_tot_dbm} dBm: {measured:.3} dB vs {paper:.3} dB"
            );
        }
    }
}
