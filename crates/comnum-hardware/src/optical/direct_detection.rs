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
//! # Multilevel signalling
//!
//! The same model decides between the $M$ equiprobable, equally spaced power levels
//! $P_{0} \le \dots \le P_{M-1}$ of unipolar $M$-PAM (on–off keying is $M = 2$). The eye
//! between the adjacent levels $k$ and $k + 1$ has the Q factor $Q_{k} = (I_{k+1} -
//! I_{k})/(\sigma_{k+1} + \sigma_{k})$ at its optimum threshold, and the errors to the
//! adjacent levels give the symbol and, for Gray labels (one bit per adjacent error),
//! the bit error rates
//!
//! $$
//! \mathrm{SER} = \frac{2}{M} \sum_{k=0}^{M-2} \frac{1}{2}\operatorname{erfc}\!\left(\frac{Q_{k}}{\sqrt{2}}\right),
//! \qquad \mathrm{BER} = \frac{\mathrm{SER}}{\log_{2} M}.
//! $$
//!
//! For the average power $\bar{P}$ and the extinction ratio $r_{\mathrm{ex}} =
//! P_{0}/P_{M-1}$ the levels are $P_{M-1} = 2\bar{P}/(1 + r_{\mathrm{ex}})$, $P_{0} =
//! r_{\mathrm{ex}} P_{M-1}$ and $P_{k} = P_{0} + k\,(P_{M-1} - P_{0})/(M - 1)$; the eye
//! opening $(P_{M-1} - P_{0})/(M - 1)$ is $M - 1$ times smaller than that of on–off keying
//! of the same average power, so a thermal-noise-limited receiver pays $10 \log_{10}(M -
//! 1)$ dB at the same symbol rate (4.8 dB for 4-PAM), less $5 \log_{10} \log_{2} M$ when
//! the noise bandwidth shrinks with the symbol rate of the same bit rate (3.3 dB), and the
//! extinction-ratio penalty $\delta_{\mathrm{ex}}$ is that of on–off keying. The
//! sensitivity of $M$-PAM is the average power at which the smallest eye, the top one
//! under shot noise, reaches the Q factor: for $M = 2$ it is eq. (4.5.17). The curves of
//! Walklin & Conradi, *J. Lightw. Technol.* 17(11), 2235–2248, 1999, the reference of the
//! multilevel sensitivities, are not compared (the paper is not in `references/`): the
//! formulas are checked against their $M = 2$ case and against a simulated 4-PAM
//! [`Photodiode`](super::Photodiode) (the tests of this module). The optical modulation
//! amplitude $P_{M-1} - P_{0}$ and the extinction ratio are the quantities of the IEEE
//! 802.3 PAM4 specifications (Std 802.3-2022, 121.8.5).
//!
//! # Small-signal response of the dispersive link
//!
//! A chirped source, a dispersive fibre and a square-law photodiode make a linear channel
//! for a small intensity modulation (Wang & Petermann 1992; Devaux, Sorel & Kerdiles 1993).
//! For the power $P_{0}(1 + m \cos \Omega t)$, $m \ll 1$, of the laser of
//! [`DirectlyModulatedLaser`](super::DirectlyModulatedLaser) (linewidth enhancement
//! $\beta_{c}$, adiabatic coefficient $\kappa$), the two sidebands of the field are
//! $\frac{m}{4}(1 \mp a - j\beta_{c})$ with $a = \beta_{c}\kappa P_{0}/\Omega$, both
//! turned by $e^{j\theta}$, $\theta = \beta_{2}\Omega^{2}L/2$, by the fibre
//! ([`comnum_math::nlse`], "Convention"); the detected power is $P_{0}[1 + m\,\Re(H
//! e^{j\Omega t})]$ with
//!
//! $$
//! H(\Omega) = \cos\theta + \beta_{c}\sin\theta - j\,\frac{\beta_{c}\kappa P_{0}}{\Omega}\sin\theta,
//! $$
//!
//! which, on standard fibre ($\beta_{2} < 0$, $\theta < 0$), is the $\cos|\theta| -
//! \beta_{c}\sin|\theta|$ of Wang & Petermann: the transient chirp of a positive
//! $\beta_{c}$ brings the first fading closer and the adiabatic chirp fills the notches.
//! Without adiabatic chirp the response vanishes at the fading frequencies
//!
//! $$
//! f_{u}^{2} = \frac{1 + 2u - \frac{2}{\pi}\arctan\beta_{c}}{4\pi|\beta_{2}|L}
//! = \frac{c}{2 D L \lambda^{2}}\left(1 + 2u - \frac{2}{\pi}\arctan\beta_{c}\right),
//! \qquad u = 0, 1, \dots
//! $$
//!
//! (Devaux et al. 1993, the measurement of $D$ and $\beta_{c}$ from the notches). The
//! simulated chain, laser, `comnum_channel::optical::fiber::linear_fiber_channel` and
//! square-law detection, is compared with $H$ in the card `wang1992_small_signal`.
//!
//! # Examples
//! ```
//! use comnum_hardware::optical::direct_detection::{DirectDetectionReceiver, extinction_ratio_penalty_db};
//! use comnum_math::{qfunc, w2dbm};
//! // a p-i-n receiver at 1 Gb/s: 1 A/W, 50 Ω at 300 K, 3 dB of noise figure, Δf = 500 MHz
//! let rx = DirectDetectionReceiver { bandwidth_hz: 0.5e9, ..Default::default() };
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
//! 2. J. Wang, K. Petermann, "Small signal analysis for dispersive optical fiber
//!    communication systems," *J. Lightw. Technol.* 10(1), 96–100, 1992.
//! 3. F. Devaux, Y. Sorel, J. F. Kerdiles, "Simple measurement of fiber dispersion and of
//!    chirp parameter of intensity modulated light emitter," *J. Lightw. Technol.* 11(12),
//!    1937–1940, 1993.

use super::{shot_noise_variance, thermal_noise_variance};
use comnum_core::{Result, ensure};
use comnum_math::{C64, ELEMENTARY_CHARGE, PLANCK, bisect, db2lin, lin2db, qfunc, w2dbm};
use std::f64::consts::PI;

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

impl Default for DirectDetectionReceiver {
    /// A p-i-n receiver of 1 A/W: 50 Ω at 300 K, 3 dB of noise figure, 1 GHz of noise
    /// bandwidth.
    fn default() -> Self {
        DirectDetectionReceiver {
            responsivity: 1.0,
            apd_gain: 1.0,
            ionization_ratio: 0.0,
            temperature_k: 300.0,
            load_ohm: 50.0,
            noise_figure_db: 3.0,
            bandwidth_hz: 1e9,
        }
    }
}

impl DirectDetectionReceiver {
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

    /// The Q factor of every eye of the power levels `levels_w` (W, non-decreasing, at
    /// least two): $M - 1$ values $Q_{k} = (I_{k+1} - I_{k})/(\sigma_{k+1} + \sigma_{k})$
    /// with $I_{k} = M R P_{k}$ and $\sigma_{k}^{2} = \sigma_{s}^{2}(P_{k}) +
    /// \sigma_{T}^{2}$ (see the module, "Multilevel signalling").
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::check`]; [`comnum_core::Error::Config`] for fewer
    /// than two levels, a negative or a decreasing level.
    ///
    /// # Examples
    /// ```
    /// use comnum_hardware::optical::direct_detection::{DirectDetectionReceiver, pam_levels_w};
    /// let rx = DirectDetectionReceiver::default();
    /// let levels = pam_levels_w(4, 1e-5, 0.0).unwrap();
    /// let q = rx.eye_q_factors(&levels).unwrap();
    /// assert_eq!(q.len(), 3);
    /// assert!(q[0] > q[1] && q[1] > q[2]); // the shot noise grows with the level
    /// ```
    pub fn eye_q_factors(&self, levels_w: &[f64]) -> Result<Vec<f64>> {
        self.check()?;
        check_levels(levels_w)?;
        let t = self.thermal_variance();
        let gain = self.apd_gain * self.responsivity;
        let sigma =
            levels_w.iter().map(|&p| self.shot_variance(p).map(|s| (s + t).sqrt())).collect::<Result<Vec<f64>>>()?;
        (0..levels_w.len() - 1)
            .map(|k| q_factor(gain * levels_w[k + 1], gain * levels_w[k], sigma[k + 1], sigma[k]))
            .collect()
    }

    /// The symbol error rate of the equiprobable power levels `levels_w` at the optimum
    /// thresholds, counting the errors to the adjacent levels: $\frac{2}{M}
    /// \sum_{k} \frac{1}{2}\operatorname{erfc}(Q_{k}/\sqrt{2})$ over the eyes of
    /// [`DirectDetectionReceiver::eye_q_factors`]; [`DirectDetectionReceiver::ber`] for
    /// two levels.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::eye_q_factors`].
    pub fn ser_pam(&self, levels_w: &[f64]) -> Result<f64> {
        let q = self.eye_q_factors(levels_w)?;
        Ok(2.0 / levels_w.len() as f64 * q.iter().map(|&q| qfunc(q)).sum::<f64>())
    }

    /// The bit error rate of the Gray-labelled equiprobable power levels `levels_w`,
    /// [`DirectDetectionReceiver::ser_pam`] over $\log_{2} M$ (an error to an adjacent
    /// level flips one bit).
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::eye_q_factors`].
    pub fn ber_pam(&self, levels_w: &[f64]) -> Result<f64> {
        Ok(self.ser_pam(levels_w)? / (levels_w.len() as f64).log2())
    }

    /// The sensitivity (W) of unipolar `m`-PAM of extinction ratio `r_ex`: the average
    /// power of the levels of [`pam_levels_w`] at which the smallest eye reaches the Q
    /// factor `q`, found by bisection (the Q factor of every eye grows with the power);
    /// [`DirectDetectionReceiver::sensitivity_w`] for $M = 2$ and $r_{\mathrm{ex}} = 0$.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::check`] and [`pam_levels_w`], or a non-positive `q`.
    ///
    /// # Examples
    /// ```
    /// use comnum_hardware::optical::direct_detection::DirectDetectionReceiver;
    /// let rx = DirectDetectionReceiver::default();
    /// let ook = rx.sensitivity_pam_w(2, 6.0, 0.0).unwrap();
    /// assert!((ook - rx.sensitivity_w(6.0).unwrap()).abs() < 1e-9 * ook);
    /// // 4-PAM at the same symbol rate: about (M - 1) times the power, more with the shot noise
    /// let pam4 = rx.sensitivity_pam_w(4, 6.0, 0.0).unwrap();
    /// assert!(pam4 > 3.0 * ook && pam4 < 3.5 * ook);
    /// ```
    pub fn sensitivity_pam_w(&self, m: usize, q: f64, r_ex: f64) -> Result<f64> {
        self.check()?;
        ensure(q > 0.0, || format!("the Q factor {q} must be positive"))?;
        pam_levels_w(m, 1.0, r_ex)?;
        // the arguments are checked above: the closed forms cannot fail (NaN would stop the
        // bisection at its lower bound)
        let smallest_eye = |p_avg: f64| -> f64 {
            pam_levels_w(m, p_avg, r_ex)
                .and_then(|levels| self.eye_q_factors(&levels))
                .map_or(f64::NAN, |q| q.into_iter().fold(f64::INFINITY, f64::min))
        };
        // the thermal-noise limit is a lower bound: the shot noise only costs more power
        let eye_per_watt = 2.0 * (1.0 - r_ex) / ((1.0 + r_ex) * (m - 1) as f64);
        let lo = q * 2.0 * self.thermal_variance().sqrt() / (self.apd_gain * self.responsivity * eye_per_watt);
        let mut hi = 2.0 * lo;
        while smallest_eye(hi) < q {
            hi *= 2.0;
        }
        Ok(bisect(|p| smallest_eye(p) < q, lo, hi, 200))
    }

    /// [`DirectDetectionReceiver::sensitivity_pam_w`] in dBm.
    ///
    /// # Errors
    /// As [`DirectDetectionReceiver::sensitivity_pam_w`].
    pub fn sensitivity_pam_dbm(&self, m: usize, q: f64, r_ex: f64) -> Result<f64> {
        Ok(w2dbm(self.sensitivity_pam_w(m, q, r_ex)?))
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

fn check_levels(levels_w: &[f64]) -> Result<()> {
    ensure(levels_w.len() >= 2, || format!("{} power level(s): at least two are needed", levels_w.len()))?;
    ensure(levels_w.iter().all(|&p| p >= 0.0), || "the power levels must be non-negative".into())?;
    ensure(levels_w.windows(2).all(|w| w[1] >= w[0]), || "the power levels must be non-decreasing".into())
}

/// The $M$ equally spaced power levels (W) of unipolar `m`-PAM of average power
/// `p_avg_w` $\bar{P}$ and extinction ratio `r_ex` $= P_{0}/P_{M-1} \in [0, 1)$, in
/// increasing order: $P_{M-1} = 2\bar{P}/(1 + r_{\mathrm{ex}})$, $P_{0} = r_{\mathrm{ex}}
/// P_{M-1}$, $P_{k} = P_{0} + k\,(P_{M-1} - P_{0})/(M - 1)$ (see the module, "Multilevel
/// signalling"). Equiprobable levels have the average $\bar{P}$.
///
/// # Errors
/// [`comnum_core::Error::Config`] for $M < 2$, a negative power or a ratio outside $[0, 1)$.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::pam_levels_w;
/// assert_eq!(pam_levels_w(2, 1.0, 0.0).unwrap(), vec![0.0, 2.0]);
/// let p = pam_levels_w(4, 1.5, 0.5).unwrap(); // P_3 = 2, P_0 = 1
/// for (got, want) in p.iter().zip([1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0]) {
///     assert!((got - want).abs() < 1e-15);
/// }
/// ```
pub fn pam_levels_w(m: usize, p_avg_w: f64, r_ex: f64) -> Result<Vec<f64>> {
    ensure(m >= 2, || format!("{m} level(s): at least two are needed"))?;
    ensure(p_avg_w >= 0.0, || format!("the average power {p_avg_w} W must be non-negative"))?;
    ensure((0.0..1.0).contains(&r_ex), || format!("the extinction ratio {r_ex} must lie in [0, 1)"))?;
    let top = 2.0 * p_avg_w / (1.0 + r_ex);
    let bottom = r_ex * top;
    let step = (top - bottom) / (m - 1) as f64;
    Ok((0..m).map(|k| bottom + k as f64 * step).collect())
}

/// The optical modulation amplitude (W) of the power levels `levels_w`, $P_{M-1} -
/// P_{0}$: the `OMA` of on–off keying, the `OMA_outer` of the PAM4 specifications of
/// IEEE Std 802.3-2022 (121.8.5).
///
/// # Errors
/// [`comnum_core::Error::Config`] for fewer than two levels, a negative or a decreasing
/// level.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::{oma_w, pam_levels_w};
/// let p = pam_levels_w(4, 1.5, 0.5).unwrap();
/// assert!((oma_w(&p).unwrap() - 1.0).abs() < 1e-15);
/// ```
pub fn oma_w(levels_w: &[f64]) -> Result<f64> {
    check_levels(levels_w)?;
    Ok(levels_w[levels_w.len() - 1] - levels_w[0])
}

/// The power penalty (dB) of unipolar `m`-PAM against on–off keying of the same average
/// power and symbol rate on a thermal-noise-limited receiver, $10 \log_{10}(M - 1)$: the
/// eye opening is $M - 1$ times smaller (4.77 dB for 4-PAM, 8.45 dB for 8-PAM). At the
/// same bit rate the noise bandwidth shrinks by $\log_{2} M$ and the penalty by $5
/// \log_{10} \log_{2} M$ (3.27 dB for 4-PAM).
///
/// # Errors
/// [`comnum_core::Error::Config`] for $M < 2$.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::multilevel_penalty_db;
/// assert_eq!(multilevel_penalty_db(2).unwrap(), 0.0);
/// assert!((multilevel_penalty_db(4).unwrap() - 4.771).abs() < 1e-3);
/// ```
pub fn multilevel_penalty_db(m: usize) -> Result<f64> {
    ensure(m >= 2, || format!("{m} level(s): at least two are needed"))?;
    Ok(lin2db((m - 1) as f64))
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

/// The small-signal response $H(f)$ of the dispersive IM-DD link (see the module) at the
/// modulation frequency `f_hz`, for a laser of linewidth enhancement `chirp` $\beta_{c}$,
/// adiabatic coefficient `adiabatic_hz_w` $\kappa$ (Hz/W) and mean power `power_w`
/// $P_{0}$, over `length_km` of the dispersion `beta2_s2_km`; $H(0) = 1$.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a negative length, power or frequency, or for a
/// zero frequency with adiabatic chirp (the response has a pole there).
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::{dispersion_fading_response, fading_frequency_hz};
/// // standard fibre, β_c = 3: the response vanishes at the first fading frequency
/// let f0 = fading_frequency_hz(0, 3.0, -21.7e-24, 40.0).unwrap();
/// assert!((f0 / 1e9 - 4.3).abs() < 0.1);
/// assert!(dispersion_fading_response(f0, 3.0, 0.0, 1e-3, -21.7e-24, 40.0).unwrap().norm() < 1e-9);
/// // the adiabatic chirp fills the notch
/// assert!(dispersion_fading_response(f0, 3.0, 1e13, 1e-3, -21.7e-24, 40.0).unwrap().norm() > 0.3);
/// ```
pub fn dispersion_fading_response(
    f_hz: f64,
    chirp: f64,
    adiabatic_hz_w: f64,
    power_w: f64,
    beta2_s2_km: f64,
    length_km: f64,
) -> Result<C64> {
    ensure(f_hz >= 0.0 && power_w >= 0.0 && length_km >= 0.0, || {
        "the frequency, power and length must be non-negative".into()
    })?;
    let omega = 2.0 * PI * f_hz;
    let theta = beta2_s2_km / 2.0 * omega * omega * length_km;
    let (sin, cos) = theta.sin_cos();
    if adiabatic_hz_w == 0.0 {
        return Ok(C64::new(cos + chirp * sin, 0.0));
    }
    ensure(f_hz > 0.0, || "the response of an adiabatic chirp needs a positive frequency".into())?;
    Ok(C64::new(cos + chirp * sin, -chirp * adiabatic_hz_w * power_w / omega * sin))
}

/// The fading frequency $f_{u}$ (Hz) of order `u` of the dispersive IM-DD link for a
/// transient chirp `chirp` $\beta_{c}$ over `length_km` of the dispersion `beta2_s2_km`
/// (see the module; Devaux et al. 1993).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a zero dispersion or a non-positive length.
///
/// # Examples
/// ```
/// use comnum_hardware::optical::direct_detection::fading_frequency_hz;
/// // an unchirped source: the first notch at 1/(4 π |β₂| L)^(1/2), 9.6 GHz over 40 km of SSMF
/// let f0 = fading_frequency_hz(0, 0.0, -21.7e-24, 40.0).unwrap();
/// assert!((f0 / 1e9 - 9.58).abs() < 0.01);
/// // the chirp of a positive β_c brings it closer, a negative one pushes it away
/// assert!(fading_frequency_hz(0, 3.0, -21.7e-24, 40.0).unwrap() < f0);
/// assert!(fading_frequency_hz(0, -3.0, -21.7e-24, 40.0).unwrap() > f0);
/// ```
pub fn fading_frequency_hz(u: usize, chirp: f64, beta2_s2_km: f64, length_km: f64) -> Result<f64> {
    ensure(beta2_s2_km != 0.0 && length_km > 0.0, || "the fading needs a dispersion and a positive length".into())?;
    let order = 1.0 + 2.0 * u as f64 - 2.0 / PI * chirp.atan();
    Ok((order / (4.0 * PI * beta2_s2_km.abs() * length_km)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical::Photodiode;
    use comnum_core::rng::Rng;

    fn rx() -> DirectDetectionReceiver {
        DirectDetectionReceiver { responsivity: 0.8, bandwidth_hz: 5e9, ..Default::default() }
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
    fn the_fading_frequencies_are_the_zeros_of_the_small_signal_response() {
        let (b2, l) = (-21.7e-24, 40.0);
        for chirp in [0.0, 3.0, -3.0, 6.0] {
            for u in 0..3 {
                let f = fading_frequency_hz(u, chirp, b2, l).unwrap();
                let h = dispersion_fading_response(f, chirp, 0.0, 1e-3, b2, l).unwrap();
                assert!(h.norm() < 1e-9, "beta_c {chirp}, u {u}: |H| = {}", h.norm());
                // the next zero is the next root of tan θ = -1/β_c: no other zero between them
                let bound = (1.0 + chirp * chirp).sqrt();
                let h_mid = dispersion_fading_response(f * 1.2, chirp, 0.0, 1e-3, b2, l).unwrap().norm();
                assert!(h_mid > 0.0 && h_mid <= bound + 1e-12);
            }
        }
        // a notch of an unchirped source on anomalous and normal dispersion is the same
        let f = fading_frequency_hz(0, 0.0, b2, l).unwrap();
        assert_eq!(f, fading_frequency_hz(0, 0.0, -b2, l).unwrap());
        assert_eq!(dispersion_fading_response(0.0, 3.0, 0.0, 1e-3, b2, l).unwrap(), C64::new(1.0, 0.0));
        assert!(dispersion_fading_response(0.0, 3.0, 1e13, 1e-3, b2, l).is_err());
        assert!(fading_frequency_hz(0, 3.0, 0.0, l).is_err());
    }

    #[test]
    fn two_levels_are_on_off_keying_and_the_thermal_limit_pays_m_minus_one() {
        let rx = DirectDetectionReceiver { bandwidth_hz: 5e9, ..Default::default() };
        let q = 6.0;
        // M = 2, r_ex = 0: the sensitivity of eq. (4.5.17), its eye the Q factor
        let ook = rx.sensitivity_pam_w(2, q, 0.0).unwrap();
        assert!((ook - rx.sensitivity_w(q).unwrap()).abs() < 1e-9 * ook);
        let levels = pam_levels_w(2, ook, 0.0).unwrap();
        assert!((rx.eye_q_factors(&levels).unwrap()[0] - q).abs() < 1e-9);
        assert!((rx.ber_pam(&levels).unwrap() - rx.ber(levels[1], levels[0]).unwrap()).abs() < 1e-18);
        assert!((rx.ser_pam(&levels).unwrap() - qfunc(q)).abs() < 1e-18);
        // the extinction-ratio penalty of M levels is that of on-off keying (thermal limit)
        // a noisy amplifier makes the receiver thermal-noise limited
        let cold = DirectDetectionReceiver { noise_figure_db: 60.0, ..rx.clone() };
        for m in [2usize, 4, 8] {
            let ideal = cold.sensitivity_pam_w(m, q, 0.0).unwrap();
            let penalty = lin2db(cold.sensitivity_pam_w(m, q, 0.3).unwrap() / ideal);
            assert!((penalty - extinction_ratio_penalty_db(0.3).unwrap()).abs() < 1e-3, "{m}: {penalty}");
            // and M-PAM pays 10 log10(M - 1) against on-off keying at the same symbol rate
            let against_ook = lin2db(ideal / cold.sensitivity_pam_w(2, q, 0.0).unwrap());
            assert!((against_ook - multilevel_penalty_db(m).unwrap()).abs() < 1e-3, "{m}: {against_ook}");
        }
        // at the same bit rate the bandwidth shrinks by log2 M: 5 log10(log2 M) less
        let half = DirectDetectionReceiver { bandwidth_hz: 2.5e9, ..cold.clone() };
        let same_bit_rate =
            lin2db(half.sensitivity_pam_w(4, q, 0.0).unwrap() / cold.sensitivity_pam_w(2, q, 0.0).unwrap());
        assert!((same_bit_rate - (multilevel_penalty_db(4).unwrap() - 5.0 * 2f64.log10())).abs() < 1e-3);
        // with shot noise the top eye is the smallest and the sensitivity is where it reaches Q
        let p = rx.sensitivity_pam_w(4, q, 0.1).unwrap();
        let eyes = rx.eye_q_factors(&pam_levels_w(4, p, 0.1).unwrap()).unwrap();
        assert!(eyes[0] > eyes[1] && eyes[1] > eyes[2]);
        assert!((eyes[2] - q).abs() < 1e-9);
        assert!(p > 3.0 * ook);
        // the shape errors
        assert!(rx.eye_q_factors(&[1.0]).is_err());
        assert!(rx.eye_q_factors(&[1.0, 0.5]).is_err());
        assert!(oma_w(&[-1.0, 1.0]).is_err());
        assert!(pam_levels_w(1, 1.0, 0.0).is_err());
        assert!(pam_levels_w(4, 1.0, 1.0).is_err());
    }

    #[test]
    fn the_simulated_photodiode_has_the_symbol_error_rate_of_its_eyes_for_four_levels() {
        // 4-PAM integrated over the symbol, as the on-off keying test below
        let (symbol_rate, sps, n_symbols) = (10e9, 4usize, 100_000usize);
        let fs = symbol_rate * sps as f64;
        let b = symbol_rate / 2.0;
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
            ..Default::default()
        };
        let (q, r_ex) = (2.5, 0.1);
        let levels = pam_levels_w(4, rx.sensitivity_pam_w(4, q, r_ex).unwrap(), r_ex).unwrap();
        let mut rng = Rng::new(7);
        let symbols: Vec<usize> = (0..n_symbols).map(|_| (rng.bit() as usize) << 1 | rng.bit() as usize).collect();
        let power: Vec<f64> = symbols.iter().flat_map(|&k| std::iter::repeat_n(levels[k], sps)).collect();
        let i = pd.detect_power(&power, fs, &mut rng).unwrap();
        // the optimum thresholds of adjacent Gaussian levels: (σ_k I_{k+1} + σ_{k+1} I_k)/(σ_k + σ_{k+1})
        let t = rx.thermal_variance();
        let sigma: Vec<f64> = levels.iter().map(|&p| (rx.shot_variance(p).unwrap() + t).sqrt()).collect();
        let thresholds: Vec<f64> = (0..3)
            .map(|k| {
                rx.responsivity * (sigma[k] * levels[k + 1] + sigma[k + 1] * levels[k]) / (sigma[k] + sigma[k + 1])
            })
            .collect();
        let errors = symbols
            .iter()
            .enumerate()
            .filter(|&(n, &k)| {
                let sample = i[n * sps..(n + 1) * sps].iter().sum::<f64>() / sps as f64;
                thresholds.iter().filter(|&&th| sample > th).count() != k
            })
            .count();
        let ser = errors as f64 / n_symbols as f64;
        let expected = rx.ser_pam(&levels).unwrap();
        let sampling = (expected / n_symbols as f64).sqrt();
        assert!((ser - expected).abs() < 3.0 * sampling, "{ser} against {expected} ± {sampling}");
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
            ..Default::default()
        };
        let q = 3.0;
        let p = rx.sensitivity_w(q).unwrap();
        let mut rng = Rng::new(11);
        let bits: Vec<bool> = (0..n_bits).map(|_| rng.bit() == 1).collect();
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
    }
}
