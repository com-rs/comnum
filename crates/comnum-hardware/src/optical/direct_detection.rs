//! Closed forms of direct-detection receivers: the Q factor and bit error rate of on–off
//! keying, the sensitivity of p-i-n and avalanche photodiode receivers and its quantum
//! limit, and the power penalties of a finite extinction ratio, of intensity noise, of
//! timing jitter, of dispersive pulse broadening and of the chirp of the source, as the
//! textbook treatment of G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed.,
//! Wiley, 2002, Secs. 2.4.2–2.4.3, 4.4–4.6 and 5.4 (the equation numbers below are its).
//!
//! # Model
//!
//! The receiver decides between the levels $I_{1}$ and $I_{0}$ of the photocurrent, each
//! with a Gaussian noise of standard deviation $\sigma_{1}$ and $\sigma_{0}$. At the
//! optimum threshold the bit error rate is set by the Q factor (eqs. (4.5.10), (4.5.11)):
//!
//! $$
//! Q = \frac{I_{1} - I_{0}}{\sigma_{1} + \sigma_{0}}, \qquad
//! \mathrm{BER} = \frac{1}{2}\operatorname{erfc}\!\left(\frac{Q}{\sqrt{2}}\right)
//! \approx \frac{e^{-Q^{2}/2}}{Q\sqrt{2\pi}},
//! $$
//!
//! $Q = 6$ for a BER of $10^{-9}$ ([`comnum_math::qfunc`] is the BER of a Q factor,
//! [`comnum_math::qfuncinv`] the Q factor of a BER). The noise of a photodiode of
//! responsivity $R$ with the avalanche gain $M$ (1 for a p-i-n) and excess noise factor
//! $F_{A}$ in the noise bandwidth $\Delta f$ is the shot noise of the current and the
//! thermal noise of the load $R_{L}$ at the temperature $T$, raised by the noise figure
//! $F_{n}$ of the amplifier (eqs. (4.4.17), (4.4.18), (4.4.9)):
//!
//! $$
//! \sigma_{s}^{2} = 2 q M^{2} F_{A}(M) R P \Delta f, \qquad
//! \sigma_{T}^{2} = \frac{4 k_{B} T}{R_{L}} F_{n} \Delta f, \qquad
//! F_{A}(M) = k_{A} M + (1 - k_{A})\left(2 - \frac{1}{M}\right),
//! $$
//!
//! with $k_{A}$ the ionisation-coefficient ratio of the avalanche photodiode. With no
//! power in the 0 bits, the average power that gives a Q factor $Q$ is the receiver
//! sensitivity (eq. (4.5.17), exact for this model),
//!
//! $$
//! \bar{P}_{\mathrm{rec}} = \frac{Q}{R}\left(q F_{A} \Delta f\, Q + \frac{\sigma_{T}}{M}\right),
//! $$
//!
//! minimised over $M$ at $M_{\mathrm{opt}}^{2} = \left[\sigma_{T}/(q \Delta f Q) - (1 -
//! k_{A})\right]/k_{A}$ (eq. (4.5.19)), where it is $(2 q \Delta f/R) Q^{2} (k_{A}
//! M_{\mathrm{opt}} + 1 - k_{A})$ (eq. (4.5.20)). A receiver that counts photons without
//! noise errs only when a 1 bit carries no photon, $\mathrm{BER} = \frac{1}{2} e^{-N_{p}}$
//! with $N_{p}$ photons per 1 bit (eq. (4.5.24)): the quantum limit, $\bar{N}_{p} =
//! N_{p}/2 \approx 10$ photons per bit on average for $10^{-9}$, the power $\bar{N}_{p} h
//! \nu B$ (eq. (4.5.25)).
//!
//! The power penalties are the extra average power that restores the Q factor of the
//! ideal receiver, in dB (eqs. (4.6.4), (4.6.11), (4.6.20), (5.4.4)):
//!
//! $$
//! \delta_{\mathrm{ex}} = 10 \log_{10}\frac{1 + r_{\mathrm{ex}}}{1 - r_{\mathrm{ex}}}, \qquad
//! \delta_{I} = -10 \log_{10}\left(1 - r_{I}^{2} Q^{2}\right), \qquad
//! \delta_{j} = 10 \log_{10}\frac{1 - b/2}{(1 - b/2)^{2} - b^{2} Q^{2}/2}, \qquad
//! \delta_{d} = 10 \log_{10} f_{b} = -5 \log_{10}\left[1 - (4 B L |D| \sigma_{\lambda})^{2}\right],
//! $$
//!
//! with $r_{\mathrm{ex}} = P_{0}/P_{1}$ the extinction ratio (thermal-noise-limited
//! receiver), $r_{I}$ the relative intensity noise of the source (its r.m.s. over the
//! power), $b = (4\pi^{2}/3 - 8)(B \tau_{j})^{2}$ for a Gaussian timing jitter of r.m.s.
//! $\tau_{j}$ on a raised-cosine pulse (eq. (4.6.17)), and $f_{b} = \sigma/\sigma_{0}$ the
//! broadening of a Gaussian pulse whose output width fills the bit, $4 B \sigma = 1$: for a
//! source of r.m.s. spectral width $\sigma_{\lambda}$ over $L$ of dispersion $D$,
//! $\sigma^{2} = \sigma_{0}^{2} + (D L \sigma_{\lambda})^{2}$ (eq. (2.4.24)). For a
//! chirped Gaussian pulse $\exp[-(1 + jC) t^{2}/(2 T_{0}^{2})]$ of a narrow-linewidth
//! source the r.m.s. width grows as (eq. (2.4.22), with $\sigma_{0} = T_{0}/\sqrt{2}$)
//!
//! $$
//! f_{b}^{2} = \left(1 + \frac{C \beta_{2} L}{T_{0}^{2}}\right)^{2} +
//! \left(\frac{\beta_{2} L}{T_{0}^{2}}\right)^{2} +
//! (1 + C^{2})^{2} \left(\frac{\beta_{3} L}{2 T_{0}^{3}}\right)^{2},
//! $$
//!
//! the chirp penalty of the worst-case pulse that fills the bit, $\sigma_{0} = 1/(4B)$, is
//! $\delta_{c} = 5 \log_{10}[(1 + 8 C \beta_{2} B^{2} L)^{2} + (8 \beta_{2} B^{2}
//! L)^{2}]$ (eq. (5.4.14)), and the dispersion-limited length of an unchirped pulse of
//! optimum width is $L \le 1/(16 B^{2} |\beta_{2}|)$ (eqs. (2.4.30), (5.2.3)).
//!
//! The numbers of the text (the sensitivities of Secs. 4.5.2 and 4.5.3, the penalties of
//! Figs. 4.20–4.22, 5.7 and 5.11) are the validation cards `agrawal2002_*`; the formulas
//! are also checked against each other and against a simulation of
//! [`Photodiode`](super::Photodiode) (the tests of this module).
//!
//! # Examples
//! ```
//! use comnum_hardware::optical::direct_detection::{DirectDetectionReceiver, extinction_ratio_penalty_db};
//! use comnum_math::{qfunc, w2dbm};
//! // a p-i-n receiver at 1 Gb/s: 1 A/W, 50 Ω at 300 K, 3 dB of noise figure, Δf = 500 MHz
//! let rx = DirectDetectionReceiver { bandwidth_hz: 0.5e9, ..DirectDetectionReceiver::pin(1.0) };
//! let p = rx.sensitivity_w(6.0).unwrap();                       // Q = 6: BER 1e-9
//! assert!((w2dbm(p) - (-24.6)).abs() < 0.05);
//! assert!((rx.q_factor(2.0 * p, 0.0).unwrap() - 6.0).abs() < 1e-9);  // the exact inverse
//! assert!((qfunc(6.0) - 1e-9).abs() < 1e-10);
//! assert!((extinction_ratio_penalty_db(0.1).unwrap() - 0.87).abs() < 0.01);
//! ```
//!
//! # References
//! 1. G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002,
//!    Secs. 2.4.2–2.4.3, 4.4–4.6, 5.2.2 and 5.4.

use super::{shot_noise_variance, thermal_noise_variance};
use comnum_core::{Result, ensure};
use comnum_math::{ELEMENTARY_CHARGE, PLANCK, db2lin, lin2db, qfunc, w2dbm};

/// The Q factor $(I_{1} - I_{0})/(\sigma_{1} + \sigma_{0})$ of two Gaussian levels.
///
/// # Errors
/// [`comnum_core::Error::Config`] if the standard deviations are negative or both zero.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::q_factor;
/// assert_eq!(q_factor(1.0, 0.0, 0.1, 0.1).unwrap(), 5.0);
/// ```
pub fn q_factor(i1: f64, i0: f64, sigma1: f64, sigma0: f64) -> Result<f64> {
    ensure(sigma1 >= 0.0 && sigma0 >= 0.0 && sigma1 + sigma0 > 0.0, || {
        format!("the noise deviations {sigma1} and {sigma0} must be non-negative and not both zero")
    })?;
    Ok((i1 - i0) / (sigma1 + sigma0))
}

/// The excess noise factor $F_{A}(M) = k_{A} M + (1 - k_{A})(2 - 1/M)$ of an avalanche
/// photodiode of gain `m` $M \ge 1$ and ionisation-coefficient ratio `k_a` $k_{A} \in [0,
/// 1]$ (Agrawal eq. (4.4.18)): 1 for $M = 1$, $M$ for $k_{A} = 1$.
///
/// # Errors
/// [`comnum_core::Error::Config`] for $M < 1$ or $k_{A}$ outside $[0, 1]$.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::excess_noise_factor;
/// assert_eq!(excess_noise_factor(1.0, 0.3).unwrap(), 1.0);
/// assert!((excess_noise_factor(10.0, 0.0).unwrap() - 1.9).abs() < 1e-12);
/// ```
pub fn excess_noise_factor(m: f64, k_a: f64) -> Result<f64> {
    ensure(m >= 1.0, || format!("the avalanche gain {m} must be at least 1"))?;
    ensure((0.0..=1.0).contains(&k_a), || format!("the ionisation ratio {k_a} must lie in [0, 1]"))?;
    Ok(k_a * m + (1.0 - k_a) * (2.0 - 1.0 / m))
}

/// A direct-detection receiver of on–off keying: a p-i-n or avalanche photodiode, its
/// load and amplifier, in the Gaussian noise model of the module.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectDetectionReceiver {
    /// Responsivity $R$ of the photodiode before the avalanche gain (A/W).
    pub responsivity: f64,
    /// Avalanche gain $M$ (1 for a p-i-n photodiode).
    pub apd_gain: f64,
    /// Ionisation-coefficient ratio $k_{A}$ of the avalanche photodiode (unused for $M = 1$).
    pub ionization_ratio: f64,
    /// Temperature $T$ (K).
    pub temperature_k: f64,
    /// Load resistance $R_{L}$ (Ω).
    pub load_ohm: f64,
    /// Noise figure $F_{n}$ of the amplifier (dB).
    pub noise_figure_db: f64,
    /// Noise bandwidth $\Delta f$ of the receiver (Hz), about half the bit rate.
    pub bandwidth_hz: f64,
}

impl DirectDetectionReceiver {
    /// A p-i-n receiver of responsivity `responsivity`: 50 Ω at 300 K, 3 dB of noise
    /// figure, 1 GHz of noise bandwidth.
    pub fn pin(responsivity: f64) -> DirectDetectionReceiver {
        DirectDetectionReceiver {
            responsivity,
            apd_gain: 1.0,
            ionization_ratio: 0.0,
            temperature_k: 300.0,
            load_ohm: 50.0,
            noise_figure_db: 3.0,
            bandwidth_hz: 1e9,
        }
    }

    /// The same receiver with the avalanche gain `m` and ionisation ratio `k_a`.
    pub fn with_apd(self, m: f64, k_a: f64) -> DirectDetectionReceiver {
        DirectDetectionReceiver { apd_gain: m, ionization_ratio: k_a, ..self }
    }

    /// Validate the parameters.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a non-positive responsivity, load, temperature
    /// or bandwidth, a gain below 1 or an ionisation ratio outside $[0, 1]$.
    pub fn check(&self) -> Result<()> {
        ensure(self.responsivity > 0.0, || format!("the responsivity {} must be positive", self.responsivity))?;
        ensure(self.temperature_k > 0.0, || format!("the temperature {} K must be positive", self.temperature_k))?;
        ensure(self.load_ohm > 0.0, || format!("the load {} Ω must be positive", self.load_ohm))?;
        ensure(self.bandwidth_hz > 0.0, || format!("the noise bandwidth {} Hz must be positive", self.bandwidth_hz))?;
        ensure(self.noise_figure_db.is_finite(), || "the noise figure must be finite".into())?;
        excess_noise_factor(self.apd_gain, self.ionization_ratio).map(|_| ())
    }

    /// The excess noise factor $F_{A}(M)$ of the photodiode.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::check`].
    pub fn excess_noise_factor(&self) -> Result<f64> {
        excess_noise_factor(self.apd_gain, self.ionization_ratio)
    }

    /// The variance $\sigma_{T}^{2} = 4 k_{B} T F_{n} \Delta f / R_{L}$ (A²) of the
    /// thermal noise.
    pub fn thermal_variance(&self) -> f64 {
        thermal_noise_variance(self.temperature_k, self.load_ohm, self.bandwidth_hz) * db2lin(self.noise_figure_db)
    }

    /// The variance $\sigma_{s}^{2} = 2 q M^{2} F_{A} R P \Delta f$ (A²) of the shot noise
    /// of the optical power `power_w` $P$.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::check`].
    pub fn shot_variance(&self, power_w: f64) -> Result<f64> {
        let m = self.apd_gain;
        Ok(m * m * self.excess_noise_factor()? * shot_noise_variance(self.responsivity * power_w, self.bandwidth_hz))
    }

    /// The Q factor of the powers `p1_w` and `p0_w` of the 1 and 0 bits (W):
    /// $I_{b} = M R P_{b}$, $\sigma_{b}^{2} = \sigma_{s}^{2}(P_{b}) + \sigma_{T}^{2}$.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::check`], or negative powers.
    pub fn q_factor(&self, p1_w: f64, p0_w: f64) -> Result<f64> {
        self.check()?;
        ensure(p1_w >= 0.0 && p0_w >= 0.0, || format!("the powers {p1_w} and {p0_w} W must be non-negative"))?;
        let t = self.thermal_variance();
        let gain = self.apd_gain * self.responsivity;
        q_factor(
            gain * p1_w,
            gain * p0_w,
            (self.shot_variance(p1_w)? + t).sqrt(),
            (self.shot_variance(p0_w)? + t).sqrt(),
        )
    }

    /// The bit error rate $\frac{1}{2}\operatorname{erfc}(Q/\sqrt{2})$ of the powers
    /// `p1_w` and `p0_w` at the optimum threshold.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::q_factor`].
    pub fn ber(&self, p1_w: f64, p0_w: f64) -> Result<f64> {
        Ok(qfunc(self.q_factor(p1_w, p0_w)?))
    }

    /// The receiver sensitivity (W): the average power $\bar{P}_{\mathrm{rec}} = (Q/R)(q
    /// F_{A} \Delta f Q + \sigma_{T}/M)$ of on–off keying with no power in the 0 bits that
    /// gives the Q factor `q` (Agrawal eq. (4.5.17)).
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::check`], or a non-positive `q`.
    pub fn sensitivity_w(&self, q: f64) -> Result<f64> {
        self.check()?;
        ensure(q > 0.0, || format!("the Q factor {q} must be positive"))?;
        let fa = self.excess_noise_factor()?;
        Ok(q / self.responsivity
            * (ELEMENTARY_CHARGE * fa * self.bandwidth_hz * q + self.thermal_variance().sqrt() / self.apd_gain))
    }

    /// [`DirectDetectionReceiver::sensitivity_w`] in dBm.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::sensitivity_w`].
    pub fn sensitivity_dbm(&self, q: f64) -> Result<f64> {
        Ok(w2dbm(self.sensitivity_w(q)?))
    }

    /// The avalanche gain that minimises the sensitivity at the Q factor `q`,
    /// $M_{\mathrm{opt}}^{2} = [\sigma_{T}/(q \Delta f Q) - (1 - k_{A})]/k_{A}$, at
    /// least 1 (Agrawal eq. (4.5.19), which also gives its leading term
    /// $\sqrt{\sigma_{T}/(k_{A} q \Delta f Q)}$); the sensitivity is then $(2 q \Delta
    /// f/R) Q^{2} (k_{A} M_{\mathrm{opt}} + 1 - k_{A})$ (eq. (4.5.20)).
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::check`]; an ionisation ratio of 0 has no finite
    /// optimum once the thermal noise exceeds $q \Delta f Q$ and is refused.
    pub fn optimum_apd_gain(&self, q: f64) -> Result<f64> {
        self.check()?;
        ensure(q > 0.0, || format!("the Q factor {q} must be positive"))?;
        let k = self.ionization_ratio;
        ensure(k > 0.0, || "an ionisation ratio of 0 gives no finite optimum gain".into())?;
        let ratio = self.thermal_variance().sqrt() / (ELEMENTARY_CHARGE * self.bandwidth_hz * q);
        Ok(((ratio - (1.0 - k)) / k).max(1.0).sqrt())
    }
}

/// The average number of photons per bit $\bar{N}_{p} = -\ln(2\,\mathrm{BER})/2$ of the
/// quantum limit of on–off keying (half the $N_{p}$ photons of a 1 bit; Agrawal
/// eq. (4.5.24)): 10.0 for a BER of $10^{-9}$.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a BER outside $(0, 1/2)$.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::quantum_limit_photons_per_bit;
/// assert!((quantum_limit_photons_per_bit(1e-9).unwrap() - 10.0).abs() < 0.02);
/// ```
pub fn quantum_limit_photons_per_bit(ber: f64) -> Result<f64> {
    ensure(ber > 0.0 && ber < 0.5, || format!("the BER {ber} must lie in (0, 1/2)"))?;
    Ok(-(2.0 * ber).ln() / 2.0)
}

/// The quantum limit of the receiver sensitivity (W), $\bar{N}_{p} h \nu B$ for the bit
/// rate `bit_rate_hz` $B$ at the optical frequency `frequency_hz` $\nu$ (Agrawal
/// eq. (4.5.25): 13 nW, $-48.9$ dBm, at 10 Gb/s and 1.55 µm).
///
/// # Errors
/// As [`quantum_limit_photons_per_bit`], or a non-positive rate or frequency.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::quantum_limit_w;
/// use comnum_math::{nm2hz, w2dbm};
/// // 10 Gb/s at 1550 nm: 10 photons per bit are -48.9 dBm
/// let p = quantum_limit_w(1e-9, 10e9, nm2hz(1550.0)).unwrap();
/// assert!((w2dbm(p) - (-48.9)).abs() < 0.05);
/// ```
pub fn quantum_limit_w(ber: f64, bit_rate_hz: f64, frequency_hz: f64) -> Result<f64> {
    ensure(bit_rate_hz > 0.0 && frequency_hz > 0.0, || {
        format!("the bit rate {bit_rate_hz} Hz and the frequency {frequency_hz} Hz must be positive")
    })?;
    Ok(quantum_limit_photons_per_bit(ber)? * PLANCK * frequency_hz * bit_rate_hz)
}

/// The power penalty (dB) of the extinction ratio `r_ex` $= P_{0}/P_{1} \in [0, 1)$ of a
/// thermal-noise-limited receiver, $10 \log_{10}[(1 + r_{\mathrm{ex}})/(1 -
/// r_{\mathrm{ex}})]$ (Agrawal eq. (4.6.4), Fig. 4.20): 0.87 dB at 0.1, 1 dB at 0.12,
/// 4.8 dB at 0.5.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a ratio outside $[0, 1)$.
pub fn extinction_ratio_penalty_db(r_ex: f64) -> Result<f64> {
    ensure((0.0..1.0).contains(&r_ex), || format!("the extinction ratio {r_ex} must lie in [0, 1)"))?;
    Ok(lin2db((1.0 + r_ex) / (1.0 - r_ex)))
}

/// The power penalty (dB) of the relative intensity noise `r_i` $= \sigma_{I}/P$ of the
/// source at the Q factor `q`, $-10 \log_{10}(1 - r_{I}^{2} Q^{2})$ (Agrawal eq. (4.6.11),
/// Fig. 4.21): the intensity noise $\sigma_{I} = r_{I} R P_{1}$ adds to the noise of the
/// 1 bits, and the penalty diverges at $r_{I} Q = 1$ (about 2 dB at $r_{I} = 0.1$ for
/// $Q = 6$).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a negative `r_i` or `q`, or $r_{I} Q \ge 1$.
pub fn intensity_noise_penalty_db(r_i: f64, q: f64) -> Result<f64> {
    ensure(r_i >= 0.0 && q >= 0.0, || format!("the intensity noise {r_i} and the Q factor {q} must be non-negative"))?;
    ensure(r_i * q < 1.0, || {
        format!("no power restores the Q factor {q} with an intensity noise of {r_i} (r_I Q >= 1)")
    })?;
    Ok(-lin2db(1.0 - r_i * r_i * q * q))
}

/// The power penalty (dB) of a Gaussian timing jitter of r.m.s. `b_tau_j` $B \tau_{j}$
/// bit periods on a thermal-noise-limited receiver of raised-cosine pulses $\cos^{2}(\pi
/// B t/2)$ at the Q factor `q` (Agrawal eqs. (4.6.17), (4.6.20), Fig. 4.22):
///
/// $$
/// \delta_{j} = 10 \log_{10}\frac{1 - b/2}{(1 - b/2)^{2} - b^{2} Q^{2}/2}, \qquad
/// b = \left(\frac{4\pi^{2}}{3} - 8\right)(B \tau_{j})^{2},
/// $$
///
/// about 2 dB at $B \tau_{j} = 0.16$ for $Q = 6$, infinite beyond about 0.2.
///
/// # Errors
/// [`comnum_core::Error::Config`] for negative arguments, or a jitter that no power
/// compensates ($(1 - b/2)^{2} \le b^{2} Q^{2}/2$).
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::timing_jitter_penalty_db;
/// assert!((timing_jitter_penalty_db(0.16, 6.0).unwrap() - 2.23).abs() < 0.01);
/// assert!(timing_jitter_penalty_db(0.0, 6.0).unwrap().abs() < 1e-12);
/// ```
pub fn timing_jitter_penalty_db(b_tau_j: f64, q: f64) -> Result<f64> {
    ensure(b_tau_j >= 0.0 && q >= 0.0, || format!("the jitter {b_tau_j} and the Q factor {q} must be non-negative"))?;
    let b = (4.0 * std::f64::consts::PI * std::f64::consts::PI / 3.0 - 8.0) * b_tau_j * b_tau_j;
    let top = 1.0 - b / 2.0;
    let bottom = top * top - b * b * q * q / 2.0;
    ensure(bottom > 0.0, || format!("no power restores the Q factor {q} with a jitter of {b_tau_j} bit periods"))?;
    Ok(lin2db(top / bottom))
}

/// The broadening factor $f_{b} = \sigma/\sigma_{0}$ of the r.m.s. width of a Gaussian
/// pulse of half-width `t0_s` $T_{0}$ (at $1/e$ of the intensity, $\sigma_{0} =
/// T_{0}/\sqrt{2}$) and chirp `chirp` $C$ after `length_km` of the dispersion
/// `beta2_s2_km` $\beta_{2}$ and third-order dispersion `beta3_s3_km` $\beta_{3}$ (Agrawal
/// eq. (2.4.22), a narrow-linewidth source):
///
/// $$
/// f_{b}^{2} = \left(1 + \frac{C \beta_{2} L}{T_{0}^{2}}\right)^{2} +
/// \left(\frac{\beta_{2} L}{T_{0}^{2}}\right)^{2} +
/// (1 + C^{2})^{2} \left(\frac{\beta_{3} L}{2 T_{0}^{3}}\right)^{2}.
/// $$
///
/// Without $\beta_{3}$ the pulse stays Gaussian and $f_{b}$ is also $T_{1}/T_{0}$ (eq.
/// (2.4.17)); $\beta_{3}$ alone gives eq. (2.4.31).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a non-positive `t0_s`.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::gaussian_broadening_factor;
/// // an unchirped pulse doubles its r.m.s. width squared when β₂ L = T₀²
/// let b = gaussian_broadening_factor(10e-12, 0.0, -20e-24, 0.0, 5.0).unwrap();
/// assert!((b - 2f64.sqrt()).abs() < 1e-12);
/// ```
pub fn gaussian_broadening_factor(
    t0_s: f64,
    chirp: f64,
    beta2_s2_km: f64,
    beta3_s3_km: f64,
    length_km: f64,
) -> Result<f64> {
    ensure(t0_s > 0.0, || format!("the pulse width {t0_s} s must be positive"))?;
    let x = beta2_s2_km * length_km / (t0_s * t0_s);
    let y = beta3_s3_km * length_km / (2.0 * t0_s * t0_s * t0_s);
    Ok(((1.0 + chirp * x).powi(2) + x * x + ((1.0 + chirp * chirp) * y).powi(2)).sqrt())
}

/// The chirp power penalty (dB) of a linearly chirped Gaussian pulse of chirp `chirp`
/// $C$ that fills the bit, $\sigma_{0} = 1/(4B)$, at the bit rate `bit_rate_hz` $B$ over
/// `length_km` $L$ of the dispersion `beta2_s2_km` $\beta_{2}$ (Agrawal eq. (5.4.14),
/// Fig. 5.11):
///
/// $$
/// \delta_{c} = 5 \log_{10}\left[(1 + 8 C \beta_{2} B^{2} L)^{2} + (8 \beta_{2} B^{2} L)^{2}\right],
/// $$
///
/// negative where $C \beta_{2} < 0$ compresses the pulse first. $C = 0$ is the dispersion
/// penalty of a chirp-free Gaussian pulse of a narrow-linewidth source.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a negative bit rate or length.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::chirp_penalty_db;
/// // |β₂| B² L = 0.2 at 10 Gb/s over 100 km of -20 ps²/km: more than 8 dB for C = -6
/// assert!(chirp_penalty_db(-6.0, -20e-24, 10e9, 100.0).unwrap() > 8.0);
/// assert!(chirp_penalty_db(2.0, -20e-24, 10e9, 10.0).unwrap() < 0.0);
/// ```
pub fn chirp_penalty_db(chirp: f64, beta2_s2_km: f64, bit_rate_hz: f64, length_km: f64) -> Result<f64> {
    ensure(bit_rate_hz >= 0.0 && length_km >= 0.0, || "the bit rate and length must be non-negative".into())?;
    let x = 8.0 * beta2_s2_km * bit_rate_hz * bit_rate_hz * length_km;
    Ok(5.0 * ((1.0 + chirp * x).powi(2) + x * x).log10())
}

/// The power penalty (dB) of the broadening factor `b_f` of a Gaussian pulse that fills
/// the bit, $10 \log_{10} f_{b}$ (Agrawal eq. (5.4.1)).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a factor below 1.
pub fn broadening_penalty_db(b_f: f64) -> Result<f64> {
    ensure(b_f >= 1.0, || format!("the broadening factor {b_f} must be at least 1"))?;
    Ok(lin2db(b_f))
}

/// The dispersion power penalty (dB) of a source of r.m.s. spectral width
/// `sigma_lambda_nm` $\sigma_{\lambda}$ at the bit rate `bit_rate_hz` $B$ over
/// `length_km` $L$ of the dispersion `d_ps_nm_km` $D$, $-5 \log_{10}[1 - (4 B L |D|
/// \sigma_{\lambda})^{2}]$ (Agrawal eq. (5.4.4), Fig. 5.7): the output pulse fills the
/// bit, $4 B \sigma = 1$, and the penalty diverges at $4 B L |D| \sigma_{\lambda} = 1$
/// (0.38 dB at $B L D \sigma_{\lambda} = 0.1$, 2.2 dB at 0.2).
///
/// # Errors
/// [`comnum_core::Error::Config`] for negative arguments or $4 B L |D| \sigma_{\lambda} \ge 1$.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::dispersion_penalty_db;
/// // 2.5 Gb/s over 50 km of 17 ps/(nm km) with a 0.05 nm source: 4BLDσ = 0.425, 0.43 dB
/// let d = dispersion_penalty_db(2.5e9, 50.0, 17.0, 0.05).unwrap();
/// assert!((d - 0.4326).abs() < 1e-3);
/// ```
pub fn dispersion_penalty_db(bit_rate_hz: f64, length_km: f64, d_ps_nm_km: f64, sigma_lambda_nm: f64) -> Result<f64> {
    ensure(bit_rate_hz >= 0.0 && length_km >= 0.0 && sigma_lambda_nm >= 0.0, || {
        "the bit rate, length and spectral width must be non-negative".into()
    })?;
    let x = 4.0 * bit_rate_hz * length_km * d_ps_nm_km.abs() * 1e-12 * sigma_lambda_nm;
    ensure(x < 1.0, || format!("the pulse overfills the bit: 4 B L |D| σ_λ = {x} >= 1"))?;
    Ok(-5.0 * (1.0 - x * x).log10())
}

/// The dispersion-limited length (km) of an unchirped Gaussian pulse of optimum width
/// at the bit rate `bit_rate_hz` $B$ over the dispersion `beta2_s2_km` $\beta_{2}$,
/// $L = 1/(16 B^{2} |\beta_{2}|)$ from the criterion $4 B \sigma \le 1$ (Agrawal
/// eqs. (2.4.30), (5.2.3)).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a non-positive bit rate or a zero dispersion.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::dispersion_limited_length_km;
/// // 10 Gb/s over standard fibre (β₂ = -21.7 ps²/km): about 29 km
/// let l = dispersion_limited_length_km(10e9, -21.7e-24).unwrap();
/// assert!((l - 28.8).abs() < 0.1);
/// ```
pub fn dispersion_limited_length_km(bit_rate_hz: f64, beta2_s2_km: f64) -> Result<f64> {
    ensure(bit_rate_hz > 0.0, || format!("the bit rate {bit_rate_hz} Hz must be positive"))?;
    ensure(beta2_s2_km != 0.0, || "a zero dispersion has no dispersion limit".into())?;
    Ok(1.0 / (16.0 * bit_rate_hz * bit_rate_hz * beta2_s2_km.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical::Photodiode;
    use comnum_core::rng::Rng;
    use comnum_math::{ZERO_CELSIUS, qfuncinv};

    fn rx() -> DirectDetectionReceiver {
        DirectDetectionReceiver { bandwidth_hz: 5e9, ..DirectDetectionReceiver::pin(0.8) }
    }

    #[test]
    fn the_sensitivity_is_the_exact_inverse_of_the_q_factor() {
        for q in [3.0, 6.0, 7.5] {
            for rx in [rx(), rx().with_apd(10.0, 0.3), rx().with_apd(40.0, 0.02)] {
                let p = rx.sensitivity_w(q).unwrap();
                assert!((rx.q_factor(2.0 * p, 0.0).unwrap() - q).abs() < 1e-9 * q, "{rx:?} at {q}");
                assert!((rx.ber(2.0 * p, 0.0).unwrap() - qfunc(q)).abs() < 1e-12 * qfunc(q));
            }
        }
    }

    #[test]
    fn the_thermal_and_shot_noise_limits_of_the_sensitivity() {
        let q = 6.0;
        // thermal limit (a narrow bandwidth, the shot noise falling as Δf and the thermal
        // noise as √Δf): P = Q σ_T / R to 4e-6
        let cold = DirectDetectionReceiver { bandwidth_hz: 1e3, ..rx() };
        let p = cold.sensitivity_w(q).unwrap();
        assert!((p / (q * cold.thermal_variance().sqrt() / cold.responsivity) - 1.0).abs() < 1e-5);
        // shot limit: a huge load kills the thermal noise, P = q Δf Q² / R
        let hot = DirectDetectionReceiver { load_ohm: 1e30, ..rx() };
        let p = hot.sensitivity_w(q).unwrap();
        assert!((p / (ELEMENTARY_CHARGE * hot.bandwidth_hz * q * q / hot.responsivity) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_optimum_gain_minimises_the_sensitivity_and_keeps_the_leading_term() {
        let q = 6.0;
        for k in [0.02, 0.1, 0.5] {
            let apd = rx().with_apd(1.0, k);
            let m = apd.optimum_apd_gain(q).unwrap();
            let best = apd.clone().with_apd(m, k).sensitivity_w(q).unwrap();
            for other in [1.0, 2.0, m * 0.8, m * 1.2, 5.0 * m] {
                assert!(
                    apd.clone().with_apd(other, k).sensitivity_w(q).unwrap() >= best * (1.0 - 1e-12),
                    "k {k}, M {other}"
                );
            }
            let leading = (apd.thermal_variance().sqrt() / (k * ELEMENTARY_CHARGE * apd.bandwidth_hz * q)).sqrt();
            assert!((m / leading - 1.0).abs() < 0.02, "k {k}: {m} against {leading}");
            // eq. (4.5.20): the minimum is (2 q Δf / R) Q² (k_A M_opt + 1 - k_A)
            let closed = 2.0 * ELEMENTARY_CHARGE * apd.bandwidth_hz / apd.responsivity * q * q * (k * m + 1.0 - k);
            assert!((best / closed - 1.0).abs() < 1e-12, "k {k}: {best} against {closed}");
        }
        assert!(rx().with_apd(1.0, 0.0).optimum_apd_gain(q).is_err());
        assert!(rx().with_apd(0.5, 0.3).check().is_err());
        assert!(rx().with_apd(2.0, 1.5).check().is_err());
    }

    #[test]
    fn the_quantum_limit_is_ten_photons_per_bit_at_1e_minus_9() {
        let n = quantum_limit_photons_per_bit(1e-9).unwrap();
        assert!((n - 10.01).abs() < 0.01, "{n}");
        assert!(quantum_limit_photons_per_bit(0.5).is_err());
        let p = quantum_limit_w(1e-9, 1e9, 193.1e12).unwrap();
        assert!((p / (n * PLANCK * 193.1e12 * 1e9) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_penalties_are_the_power_that_restores_the_q_factor() {
        let q = 6.0;
        // extinction ratio, thermal limit: the average power at r_ex against the one at 0
        let cold = DirectDetectionReceiver { bandwidth_hz: 1e3, ..rx() };
        let p = cold.sensitivity_w(q).unwrap();
        for r in [0.0, 0.05, 0.1, 0.3] {
            let avg = 2.0 * p * db2lin(extinction_ratio_penalty_db(r).unwrap()) / 2.0;
            let (p1, p0) = (2.0 * avg / (1.0 + r), 2.0 * avg * r / (1.0 + r));
            assert!((cold.q_factor(p1, p0).unwrap() - q).abs() < 1e-4 * q, "r_ex {r}");
        }
        assert!((extinction_ratio_penalty_db(0.1).unwrap() - 0.8715).abs() < 1e-3);
        assert!(extinction_ratio_penalty_db(1.0).is_err());
        // intensity noise: the 1 level carries r_I R P_1 more noise
        for r_i in [0.0, 0.05, 0.1] {
            let penalty = db2lin(intensity_noise_penalty_db(r_i, q).unwrap());
            let p1 = 2.0 * p * penalty;
            let i1 = cold.responsivity * p1;
            let s1 = (cold.thermal_variance() + (r_i * i1).powi(2)).sqrt();
            let s0 = cold.thermal_variance().sqrt();
            assert!((q_factor(i1, 0.0, s1, s0).unwrap() - q).abs() < 1e-4 * q, "r_I {r_i}");
        }
        assert!(intensity_noise_penalty_db(0.2, q).is_err());
    }

    #[test]
    fn the_dispersion_penalty_is_the_broadening_of_a_pulse_that_fills_the_bit() {
        let (b, l, d, sl): (f64, f64, f64, f64) = (2.5e9, 80.0, 17.0, 0.05);
        let x = 4.0 * b * l * d * 1e-12 * sl;
        // σ = 1/(4B) at the output: σ_0² = σ² - (D L σ_λ)², b_f = σ/σ_0
        let sigma = 1.0 / (4.0 * b);
        let sigma0 = (sigma * sigma - (d * 1e-12 * l * sl).powi(2)).sqrt();
        let expected = broadening_penalty_db(sigma / sigma0).unwrap();
        assert!((dispersion_penalty_db(b, l, d, sl).unwrap() - expected).abs() < 1e-12);
        assert!((dispersion_penalty_db(b, l, d, sl).unwrap() + 5.0 * (1.0 - x * x).log10()).abs() < 1e-12);
        assert_eq!(dispersion_penalty_db(b, 0.0, d, sl).unwrap(), 0.0);
        assert!(dispersion_penalty_db(b, l, d, 2.0).is_err());
        // a chirped pulse: C β₂ < 0 broadens from the start, C β₂ > 0 narrows first
        let beta2 = -21.7e-24;
        let (t0, l) = (20e-12, 5.0);
        let plain = gaussian_broadening_factor(t0, 0.0, beta2, 0.0, l).unwrap();
        assert!(gaussian_broadening_factor(t0, 2.0, beta2, 0.0, l).unwrap() < plain);
        assert!(gaussian_broadening_factor(t0, -2.0, beta2, 0.0, l).unwrap() > plain);
        assert_eq!(gaussian_broadening_factor(t0, 3.0, beta2, 0.0, 0.0).unwrap(), 1.0);
        // eq. (2.4.31): β₃ alone, σ² = σ₀² + (β₃ L / (4 σ₀²))² / 2 with σ₀ = T₀ / √2
        let beta3 = 0.13e-36;
        let sigma0 = t0 / 2f64.sqrt();
        let f = gaussian_broadening_factor(t0, 0.0, 0.0, beta3, 100.0).unwrap();
        let expected = (1.0 + (beta3 * 100.0 / (4.0 * sigma0 * sigma0)).powi(2) / 2.0 / (sigma0 * sigma0)).sqrt();
        assert!((f - expected).abs() < 1e-12 * expected, "{f} against {expected}");
        assert!(
            gaussian_broadening_factor(t0, 1.0, beta2, beta3, l).unwrap()
                > gaussian_broadening_factor(t0, 1.0, beta2, 0.0, l).unwrap()
        );
        // the dispersion limit: 4 B σ = 1 at the optimum width σ_0² = |β₂| L / 2
        let lim = dispersion_limited_length_km(10e9, beta2).unwrap();
        let t0 = (beta2.abs() * lim).sqrt(); // T₀² = |β₂| L minimises σ(L)
        let sigma = t0 / 2f64.sqrt() * gaussian_broadening_factor(t0, 0.0, beta2, 0.0, lim).unwrap();
        assert!((4.0 * 10e9 * sigma - 1.0).abs() < 1e-9);
        assert!(broadening_penalty_db(0.9).is_err());
        // eq. (5.4.14) is the broadening penalty of the pulse σ₀ = 1/(4B), T₀ = √2 σ₀
        let (b, l) = (10e9, 60.0);
        for c in [-6.0, -2.0, 0.0, 2.0] {
            let t0 = 2f64.sqrt() / (4.0 * b);
            let via = lin2db(gaussian_broadening_factor(t0, c, beta2, 0.0, l).unwrap());
            let direct = chirp_penalty_db(c, beta2, b, l).unwrap();
            assert!((via - direct).abs() < 1e-12, "C {c}: {via} against {direct}");
        }
        assert!(chirp_penalty_db(2.0, beta2, b, 10.0).unwrap() < 0.0);
        assert_eq!(chirp_penalty_db(-6.0, beta2, b, 0.0).unwrap(), 0.0);
        // the jitter penalty: none without jitter, infinite past about 0.2 bit periods
        assert!(timing_jitter_penalty_db(0.0, 6.0).unwrap().abs() < 1e-12);
        assert!(timing_jitter_penalty_db(0.1, 6.0).unwrap() < 0.5);
        assert!(timing_jitter_penalty_db(0.21, 6.0).is_err());
    }

    #[test]
    fn the_simulated_photodiode_has_the_bit_error_rate_of_its_q_factor() {
        // NRZ on–off keying integrated over the bit: `Photodiode` draws white noise of
        // variance σ²(B) f_s / (2B) per sample; averaged over the sps = f_s / R_b samples
        // of a bit it has the variance σ²(Δf) of the noise bandwidth Δf = R_b / 2
        let (bit_rate, sps, n_bits) = (10e9, 4usize, 200_000usize);
        let fs = bit_rate * sps as f64;
        let b = bit_rate / 2.0;
        let pd = Photodiode {
            responsivity: 0.8,
            temperature_k: 300.0,
            dark_current_a: 0.0,
            load_ohm: 50.0,
            bandwidth_hz: b,
            bandwidth_limitation: false,
            ..Photodiode::default()
        };
        let rx = DirectDetectionReceiver {
            responsivity: pd.responsivity,
            temperature_k: pd.temperature_k,
            load_ohm: pd.load_ohm,
            noise_figure_db: 0.0,
            bandwidth_hz: b,
            ..DirectDetectionReceiver::pin(pd.responsivity)
        };
        let q = 3.0;
        let p = rx.sensitivity_w(q).unwrap();
        let mut rng = Rng::new(11);
        let bits: Vec<bool> = (0..n_bits).map(|_| rng.gaussian() > 0.0).collect();
        let power: Vec<f64> =
            bits.iter().flat_map(|&b| std::iter::repeat_n(if b { 2.0 * p } else { 0.0 }, sps)).collect();
        let i = pd.detect_power(&power, fs, &mut rng).unwrap();
        let threshold = {
            // the optimum threshold of two Gaussian levels: σ_0 I_1 / (σ_0 + σ_1)
            let (s1, s0) =
                ((rx.shot_variance(2.0 * p).unwrap() + rx.thermal_variance()).sqrt(), rx.thermal_variance().sqrt());
            s0 * rx.responsivity * 2.0 * p / (s0 + s1)
        };
        let errors = bits
            .iter()
            .enumerate()
            .filter(|&(k, &b)| (i[k * sps..(k + 1) * sps].iter().sum::<f64>() / sps as f64 > threshold) != b)
            .count();
        let ber = errors as f64 / n_bits as f64;
        let expected = qfunc(q);
        let sampling = (expected / n_bits as f64).sqrt();
        assert!((ber - expected).abs() < 3.0 * sampling, "{ber} against {expected} ± {sampling}");
        assert!(ZERO_CELSIUS > 0.0 && qfuncinv(expected) > 2.99);
    }
}
