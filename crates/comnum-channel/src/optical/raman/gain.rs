//! The Raman gain spectrum of a fibre: the measured spectrum of standard single-mode
//! fibre (GNPy), a table, a constant, or one of the response-function models of silica
//! (Blow & Wood 1989, Lin & Agrawal 2006, Hollenbeck & Cantrell 2002), scaled with the
//! pump frequency unless [`RamanGain::with_pump_scaling`]`(false)`.
//!
//! # Response-function models
//!
//! The delayed part of the nonlinear response of silica is the Raman response function
//! $h_{R}(t)$, causal and normalised to $\int_{0}^{\infty} h_{R}(t)\, dt = 1$; the Raman
//! gain spectrum is the imaginary part of its Fourier transform (Agrawal, *Nonlinear
//! Fiber Optics*, 5th ed., Sec. 2.3.2 and 8.3.1),
//!
//! $$
//! g_{R}(\Omega) \propto \operatorname{Im} \tilde{h}_{R}(\Omega),
//! \qquad
//! \tilde{h}_{R}(\Omega) = \int_{0}^{\infty} h_{R}(t)\, e^{i \Omega t}\, dt,
//! $$
//!
//! so that $\operatorname{Im} \tilde{h}_{R}(\Omega) = \int_{0}^{\infty} h_{R}(t) \sin(\Omega t)\,
//! dt$. The three models give $h_{R}(t)$ in closed form; the constructors evaluate the
//! integral on a grid of 0.5 fs over 4 ps (the response has decayed by more than
//! $e^{-40}$) at every 0.1 THz from 0 to 42 THz, and scale the spectrum so that its peak
//! is the $g_{R} A_{\mathrm{eff}}$ asked for: the models give the shape of the gain, the
//! peak comes from a measurement ([`RamanGain::ssmf`]`().peak()`, $3.33 \cdot 10^{-14}$
//! m/W for a pump at 1454 nm). The prefactor of $g_{R}$ grows with the pump frequency
//! (Agrawal, Sec. 8.1.2), the $f_{p}/f_{\mathrm{ref}}$ of [`RamanGain::efficiency`].
//!
//! | Model | $h_{R}(t)$ | Parameters | Keeps |
//! | --- | --- | --- | --- |
//! | [`RamanResponse::BlowWood`] | $h_{a}(t) = \frac{\tau_{1}^{2} + \tau_{2}^{2}}{\tau_{1} \tau_{2}^{2}} e^{-t/\tau_{2}} \sin(t/\tau_{1})$ | $\tau_{1} = 12.2$ fs, $\tau_{2} = 32$ fs | the main peak (a Lorentzian at $1/(2\pi\tau_{1}) = 13.05$ THz) |
//! | [`RamanResponse::LinAgrawal`] | $(1 - f_{b})\, h_{a}(t) + f_{b}\, \frac{2\tau_{b} - t}{\tau_{b}^{2}} e^{-t/\tau_{b}}$ | $f_{b} = 0.21$, $\tau_{b} = 96$ fs | the peak and the boson-peak shoulder below 5 THz |
//! | [`RamanResponse::HollenbeckCantrell`] | $\sum_{i=1}^{13} A_{i}\, e^{-\gamma_{i} t - \Gamma_{i}^{2} t^{2}/4} \sin(\omega_{i} t)$ | Table 1 of the paper | the whole measured spectrum to 40 THz |
//!
//! Where the parameters of the models are quoted the fractional Raman contribution $f_{R}$
//! of each (0.18, 0.245 and 0.2) multiplies $h_{R}$ in the nonlinear response; it does
//! not change the shape of the gain and is not used here.
//!
//! # References
//! - K. J. Blow and D. Wood, "Theoretical description of transient stimulated Raman
//!   scattering in optical fibers," *IEEE J. Quantum Electron.* 25(12), 2665–2673, 1989,
//!   doi 10.1109/3.40655.
//! - Q. Lin and G. P. Agrawal, "Raman response function for silica fibers," *Opt. Lett.*
//!   31(21), 3086–3088, 2006, doi 10.1364/OL.31.003086.
//! - D. Hollenbeck and C. D. Cantrell, "Multiple-vibrational-mode model for fiber-optic
//!   Raman gain spectrum and response function," *J. Opt. Soc. Am. B* 19(12), 2886–2892,
//!   2002, doi 10.1364/JOSAB.19.002886.
//! - R. H. Stolen, J. P. Gordon, W. J. Tomlinson and H. A. Haus, "Raman response
//!   function of silica-core fibers," *J. Opt. Soc. Am. B* 6(6), 1159–1166, 1989, doi
//!   10.1364/JOSAB.6.001159 (the measured spectrum the three models are fitted to).

use comnum_core::{Result, ensure, ensure_shape};
use comnum_math::{SPEED_OF_LIGHT, trapezoid_uniform};
use std::sync::OnceLock;

/// A model of the Raman response function $h_{R}(t)$ of silica, the shape of a gain
/// spectrum ([`RamanGain::model`]); the parameters and the references are in the module
/// documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RamanResponse {
    /// The single damped oscillator of Blow & Wood 1989 (Agrawal, *Nonlinear Fiber
    /// Optics*, Sec. 2.3.2), $h_{a}(t) = \frac{\tau_{1}^{2} + \tau_{2}^{2}}{\tau_{1}
    /// \tau_{2}^{2}} e^{-t/\tau_{2}} \sin(t/\tau_{1})$, $\tau_{1} = 12.2$ fs, $\tau_{2} =
    /// 32$ fs: the Lorentzian $\operatorname{Im} \tilde{h}_{a}(\Omega) \propto \Omega /
    /// \big[(\tau_{1}^{-2} + \tau_{2}^{-2} - \Omega^{2})^{2} + 4 \Omega^{2} /
    /// \tau_{2}^{2}\big]$, the peak of silica at 13.1 THz with its width, a quarter of the
    /// measured gain at 3 THz and a long tail above 15 THz. The default Raman term of the
    /// split-step solvers of the literature.
    BlowWood,
    /// Lin & Agrawal 2006 (Agrawal, Sec. 2.3.2): the oscillator plus the boson peak of
    /// the glass, $(1 - f_{b})\, h_{a}(t) + f_{b}\, \frac{2 \tau_{b} - t}{\tau_{b}^{2}}
    /// e^{-t/\tau_{b}}$, $f_{b} = 0.21$, $\tau_{b} = 96$ fs ($f_{a} = 0.75$ and $f_{c} =
    /// 0.04$ of the paper both multiply $h_{a}$). The shoulder below 5 THz is recovered,
    /// the one that drives the inter-channel Raman scattering of a C+L comb; the tail
    /// above 15 THz stays.
    LinAgrawal,
    /// The intermediate-broadening model of Hollenbeck & Cantrell 2002 (Table 1):
    /// thirteen vibrational modes of the Si–O–Si structure, each a damped oscillator with
    /// a Gaussian distribution of frequencies, $\sum_{i=1}^{13} A_{i}\, e^{-\gamma_{i}
    /// t}\, e^{-\Gamma_{i}^{2} t^{2}/4} \sin(\omega_{i} t)$ with $\omega_{i} = 2\pi c\,
    /// \nu_{i}$, $\Gamma_{i} = \pi c\, \Delta\nu_{G,i}$, $\gamma_{i} = \pi c\,
    /// \Delta\nu_{L,i}$ from the position, the Gaussian and the Lorentzian full widths
    /// (cm⁻¹) of the table, fitted to the measured spectrum of Stolen et al. 1989. The
    /// whole spectrum to 40 THz: the peak, the boson shoulder, the fall at 15 THz, the
    /// bands at 18, 24 and 34 THz.
    HollenbeckCantrell,
}

/// Table 1 of Hollenbeck & Cantrell 2002: position (cm⁻¹), amplitude, Gaussian FWHM
/// (cm⁻¹), Lorentzian FWHM (cm⁻¹) of the thirteen modes.
const HOLLENBECK_CANTRELL_MODES: [(f64, f64, f64, f64); 13] = [
    (56.25, 1.00, 52.10, 17.37),
    (100.00, 11.40, 110.42, 38.81),
    (231.25, 36.67, 175.00, 58.33),
    (362.50, 67.67, 162.50, 54.17),
    (463.00, 74.00, 135.33, 45.11),
    (497.00, 4.50, 24.50, 8.17),
    (611.50, 6.80, 41.50, 13.83),
    (691.67, 4.60, 155.00, 51.67),
    (793.67, 4.20, 59.50, 19.83),
    (835.50, 4.50, 64.30, 21.43),
    (930.00, 2.70, 150.00, 50.00),
    (1080.00, 3.10, 91.00, 30.33),
    (1215.00, 3.00, 160.00, 53.33),
];

impl RamanResponse {
    /// The names of the models in a specification (`model` of the `gain` mapping).
    pub const NAMES: &[&str] = &["blow_wood", "lin_agrawal", "hollenbeck_cantrell"];

    /// The damped oscillator $h_{a}(t)$ of Blow & Wood (1/s) at the time `t_s` (s).
    fn blow_wood(t_s: f64) -> f64 {
        const TAU_1_S: f64 = 12.2e-15;
        const TAU_2_S: f64 = 32e-15;
        (TAU_1_S * TAU_1_S + TAU_2_S * TAU_2_S) / (TAU_1_S * TAU_2_S * TAU_2_S)
            * (-t_s / TAU_2_S).exp()
            * (t_s / TAU_1_S).sin()
    }

    /// The response $h_{R}(t)$ of the model (1/s, up to the normalisation that the gain
    /// spectrum does not keep) at the time `t_s` (s).
    pub fn response(self, t_s: f64) -> f64 {
        match self {
            RamanResponse::BlowWood => RamanResponse::blow_wood(t_s),
            RamanResponse::LinAgrawal => {
                const F_B: f64 = 0.21;
                const TAU_B_S: f64 = 96e-15;
                let h_b = (2.0 * TAU_B_S - t_s) / (TAU_B_S * TAU_B_S) * (-t_s / TAU_B_S).exp();
                (1.0 - F_B) * RamanResponse::blow_wood(t_s) + F_B * h_b
            }
            RamanResponse::HollenbeckCantrell => {
                // c in cm/s turns a wavenumber (cm⁻¹) into a frequency (Hz)
                let c_cm_s = SPEED_OF_LIGHT * 1e2;
                HOLLENBECK_CANTRELL_MODES
                    .iter()
                    .map(|&(nu, a, dg, dl)| {
                        let omega = 2.0 * std::f64::consts::PI * c_cm_s * nu;
                        let big_gamma = std::f64::consts::PI * c_cm_s * dg;
                        let gamma = std::f64::consts::PI * c_cm_s * dl;
                        a * (-gamma * t_s - big_gamma * big_gamma * t_s * t_s / 4.0).exp() * (omega * t_s).sin()
                    })
                    .sum()
            }
        }
    }

    /// The gain spectrum of the model with a unit peak, computed once per process.
    fn unit_shape(self) -> &'static RamanGain {
        static SHAPES: [OnceLock<RamanGain>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
        let k = match self {
            RamanResponse::BlowWood => 0,
            RamanResponse::LinAgrawal => 1,
            RamanResponse::HollenbeckCantrell => 2,
        };
        SHAPES[k].get_or_init(|| {
            // a unit peak and a placeholder reference: `model` scales and sets them
            RamanGain::from_response(|t| self.response(t), 1.0, 1.0).unwrap_or_else(|_| RamanGain::ssmf())
        })
    }
}

impl std::str::FromStr for RamanResponse {
    type Err = comnum_core::Error;

    fn from_str(s: &str) -> Result<RamanResponse> {
        match s {
            "blow_wood" => Ok(RamanResponse::BlowWood),
            "lin_agrawal" => Ok(RamanResponse::LinAgrawal),
            "hollenbeck_cantrell" => Ok(RamanResponse::HollenbeckCantrell),
            _ => Err(comnum_core::Error::Config(format!(
                "the Raman response model must be one of {:?}, got {s:?}",
                RamanResponse::NAMES
            ))),
        }
    }
}

/// The Raman gain spectrum $g_{R}(\Delta f)$ of a fibre, as the product with the
/// effective area $g_{R} A_{\mathrm{eff}}$ (m/W), for a pump at a reference frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct RamanGain {
    offsets_hz: Vec<f64>,
    gain_m_w: Vec<f64>,
    reference_hz: f64,
    /// Whether the efficiency of a pump scales with its frequency, $f_{p}/f_{\mathrm{ref}}$
    /// (see [`RamanGain::efficiency`]).
    pump_scaling: bool,
}

/// Frequency offsets of the gain spectrum of standard single-mode fibre (THz).
const SSMF_OFFSETS_THZ: [f64; 90] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
    11.0, 11.5, 12.0, 12.5, 12.75, 13.0, 13.25, 13.5, 14.0, 14.5, 14.75, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
    18.25, 18.5, 18.75, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5,
    27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0,
    36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0,
];

/// $g_{R} A_{\mathrm{eff}}$ of standard single-mode fibre at the offsets above (m/W).
#[allow(clippy::unreadable_literal, reason = "the table of GNPy, digit for digit")]
const SSMF_GAIN_M_W: [f64; 90] = [
    0.0,
    8.524419934705497e-16,
    2.643567866245371e-15,
    4.410548410941305e-15,
    6.153422961291078e-15,
    7.484924703044943e-15,
    8.452060808349209e-15,
    9.101549322698156e-15,
    9.57837595158966e-15,
    1.0008642675474562e-14,
    1.0865773569905647e-14,
    1.1300776305865833e-14,
    1.2143238647099625e-14,
    1.3231065750676068e-14,
    1.4624900971525384e-14,
    1.6013330554840492e-14,
    1.7458119359310242e-14,
    1.9320241330434762e-14,
    2.1720395392873534e-14,
    2.4137337406734775e-14,
    2.628163218460466e-14,
    2.8041019963285974e-14,
    2.9723155447089933e-14,
    3.129353531005888e-14,
    3.251796163324624e-14,
    3.3198839487612773e-14,
    3.329527690685666e-14,
    3.313155691238456e-14,
    3.289013852154548e-14,
    3.2458917188506916e-14,
    3.060684277937575e-14,
    3.2660349473783173e-14,
    2.957419109657689e-14,
    2.518894321396672e-14,
    1.734560485857344e-14,
    9.902860761605233e-15,
    7.219176385099358e-15,
    6.079565990401311e-15,
    5.828373065963427e-15,
    7.20580801091692e-15,
    7.561924351387493e-15,
    7.621152352332206e-15,
    6.8859886780643254e-15,
    5.629181047471162e-15,
    3.679727598966185e-15,
    2.7555869742500355e-15,
    2.4810133942597675e-15,
    2.2160080532403624e-15,
    2.1440626024765557e-15,
    2.33873070799544e-15,
    2.557317929858713e-15,
    3.039839048226572e-15,
    4.8337165515610065e-15,
    5.4647431818257436e-15,
    5.229187813711269e-15,
    4.510768525811313e-15,
    3.3213473130607794e-15,
    2.2602577027996455e-15,
    1.969576495866441e-15,
    1.5179853954188527e-15,
    1.2953988551200156e-15,
    1.1304672156251838e-15,
    9.10004390675213e-16,
    8.432919922183503e-16,
    7.849224069008326e-16,
    7.827568196032024e-16,
    9.000514440646232e-16,
    1.3025926460013665e-15,
    1.5444108938497558e-15,
    1.8795594063060786e-15,
    1.7796130169921014e-15,
    1.5938159865046653e-15,
    1.1585522355108287e-15,
    8.507044444633358e-16,
    7.625404663756823e-16,
    8.14510750925789e-16,
    9.047944693473188e-16,
    9.636431901702084e-16,
    9.298633899602105e-16,
    8.349739503637023e-16,
    7.482901278066085e-16,
    6.240794767134268e-16,
    5.00652535687506e-16,
    3.553373263685851e-16,
    2.0344217706119682e-16,
    1.4267522642294203e-16,
    8.980016576743517e-17,
    2.9829068181832594e-17,
    1.4861959129014824e-17,
    7.404482113326137e-18,
];

impl RamanGain {
    /// The measured gain spectrum of standard single-mode fibre for a pump at 1454 nm
    /// (206.18 THz), the default of GNPy: peak $g_{R} A_{\mathrm{eff}} = 3.33 \cdot
    /// 10^{-14}$ m/W at 13.2 THz.
    pub fn ssmf() -> RamanGain {
        RamanGain {
            offsets_hz: SSMF_OFFSETS_THZ.iter().map(|f| f * 1e12).collect(),
            gain_m_w: SSMF_GAIN_M_W.to_vec(),
            reference_hz: 206.184_634_112_792e12,
            pump_scaling: true,
        }
    }

    /// A gain spectrum: $g_{R} A_{\mathrm{eff}}$ (m/W) at increasing non-negative
    /// offsets (Hz), for a pump at `reference_hz`; linear between the offsets, zero
    /// beyond the last.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] for tables of different lengths or fewer than two
    /// offsets, [`comnum_core::Error::Config`] for offsets that do not increase from a
    /// non-negative value, or a negative gain.
    pub fn new(offsets_hz: Vec<f64>, gain_m_w: Vec<f64>, reference_hz: f64) -> Result<RamanGain> {
        ensure_shape(offsets_hz.len() == gain_m_w.len() && offsets_hz.len() >= 2, || {
            format!(
                "the Raman gain needs at least two offsets and one gain per offset, got {} offsets and {} gains",
                offsets_hz.len(),
                gain_m_w.len()
            )
        })?;
        ensure(offsets_hz[0] >= 0.0 && offsets_hz.windows(2).all(|w| w[1] > w[0]), || {
            "the offsets of the Raman gain must increase from a non-negative value".into()
        })?;
        ensure(gain_m_w.iter().all(|g| *g >= 0.0), || "the Raman gain must be non-negative".into())?;
        ensure(reference_hz > 0.0, || "the reference frequency must be positive".into())?;
        Ok(RamanGain { offsets_hz, gain_m_w, reference_hz, pump_scaling: true })
    }

    /// The same gain with (`true`, the default) or without the scaling of the efficiency
    /// with the frequency of the pump, $f_{p}/f_{\mathrm{ref}}$. Without it the efficiency
    /// depends on the offset alone: the coupling of GNPy up to its version 2.4 (an
    /// interpolated table of the offset), which the papers computed with those versions
    /// assume. On the eight pumps from 203 to 219 THz of Soltani et al. (*J. Lightw.
    /// Technol.* 40(12), 2022, Table III) the two conventions differ by about 0.6 dB on
    /// the power of the signals after 80 km.
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::optical::raman::RamanGain;
    /// let g = RamanGain::ssmf();
    /// let flat = g.clone().with_pump_scaling(false);
    /// // at the reference pump the two agree; above it the scaled efficiency is larger
    /// let (fs, f_ref) = (193.1e12, 206.184_634_112_792e12);
    /// assert!((g.efficiency(fs, f_ref, 80e-12) / flat.efficiency(fs, f_ref, 80e-12) - 1.0).abs() < 1e-12);
    /// assert!(g.efficiency(fs + 6e12, f_ref + 6e12, 80e-12) > flat.efficiency(fs + 6e12, f_ref + 6e12, 80e-12));
    /// ```
    #[must_use]
    pub fn with_pump_scaling(mut self, on: bool) -> RamanGain {
        self.pump_scaling = on;
        self
    }

    /// The same spectrum times `factor`: a measured shape with another peak (Soltani et
    /// al. 2022 give the shape of standard silica and a peak efficiency of 0.4125
    /// 1/(W·km) at 80 µm², where that of [`RamanGain::ssmf`] is 0.416).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a factor negative or not finite.
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::optical::raman::RamanGain;
    /// let g = RamanGain::ssmf();
    /// assert_eq!(g.clone().scaled(0.5)?.peak().1, 0.5 * g.peak().1);
    /// assert!(g.scaled(-1.0).is_err());
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn scaled(mut self, factor: f64) -> Result<RamanGain> {
        ensure(factor.is_finite() && factor >= 0.0, || {
            format!("the factor of the Raman gain must be finite and non-negative, got {factor}")
        })?;
        for g in &mut self.gain_m_w {
            *g *= factor;
        }
        Ok(self)
    }

    /// A constant gain efficiency `c_r_w_km` (1/(W·km)) at every offset from 0 to 40 THz
    /// (zero beyond) for a fibre of `aeff_um2` and a pump at `reference_hz`: the single
    /// number of the data sheets and of the examples of Bromage 2004 and Islam 2002 (a
    /// "Raman gain coefficient" of $g$ dB/(W·km) is $C_{R} = g \ln(10)/10$).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a negative efficiency, a non-positive area or
    /// reference frequency.
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::optical::raman::RamanGain;
    /// use comnum_math::nm2hz;
    /// let g = RamanGain::flat(0.373, 80.0, nm2hz(1451.0))?;
    /// let c = g.efficiency(nm2hz(1550.0), nm2hz(1451.0), 80e-12);
    /// assert!((c - 0.373).abs() < 1e-12);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn flat(c_r_w_km: f64, aeff_um2: f64, reference_hz: f64) -> Result<RamanGain> {
        ensure(c_r_w_km >= 0.0 && aeff_um2 > 0.0, || {
            format!(
                "the flat Raman gain needs a non-negative efficiency and a positive area, got {c_r_w_km} and {aeff_um2}"
            )
        })?;
        let offsets: Vec<f64> = (0..=40).map(|k| f64::from(k) * 1e12).collect();
        let g = c_r_w_km * aeff_um2 * 1e-12 / 1e3;
        RamanGain::new(offsets, vec![g; 41], reference_hz)
    }

    /// The linear gain of the closed forms of the inter-channel stimulated Raman
    /// scattering (Semrau, Killey & Bayvel, *J. Lightw. Technol.* 36(14), 2018, Sec. II):
    /// $g_{R} A_{\mathrm{eff}} = C_{R}' \Delta f A_{\mathrm{eff}}$ up to 14 THz, zero
    /// beyond, for the slope `slope_w_km_thz` (1/(W·km·THz), 0.028 for standard fibre at
    /// 80 µm²), the area `aeff_um2` and a pump at `reference_hz`; the `raman_slope_w_km_thz`
    /// of the ISRS closed form of the GN model, as a gain spectrum the solvers can use.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a negative slope, a non-positive area or
    /// reference frequency.
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::optical::raman::RamanGain;
    /// let g = RamanGain::linear(0.028, 80.0, 193.1e12)?;
    /// assert!((g.efficiency(193.1e12 - 5e12, 193.1e12, 80e-12) - 0.14).abs() < 1e-12);
    /// assert_eq!(g.at(20e12), 0.0);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn linear(slope_w_km_thz: f64, aeff_um2: f64, reference_hz: f64) -> Result<RamanGain> {
        const CUTOFF_THZ: f64 = 14.0;
        ensure(slope_w_km_thz >= 0.0 && aeff_um2 > 0.0, || {
            format!(
                "the linear Raman gain needs a non-negative slope and a positive area, got {slope_w_km_thz} and {aeff_um2}"
            )
        })?;
        let g = slope_w_km_thz * CUTOFF_THZ * aeff_um2 * 1e-12 / 1e3;
        RamanGain::new(vec![0.0, CUTOFF_THZ * 1e12], vec![0.0, g], reference_hz)
    }

    /// The gain spectrum of a Raman response function $h_{R}(t)$ (1/s, of the time in
    /// seconds): $\operatorname{Im} \tilde{h}_{R}(2\pi \Delta f) = \int_{0}^{\infty}
    /// h_{R}(t) \sin(2\pi \Delta f\, t)\, dt$ at every 0.1 THz from 0 to 42 THz (the
    /// trapezoidal rule on 0.5 fs over 4 ps), scaled so that its peak is `peak_m_w`
    /// (m/W, $g_{R} A_{\mathrm{eff}}$), for a pump at `reference_hz`. The models of
    /// [`RamanResponse`] go through it ([`RamanGain::model`]); so does a user model
    /// (another set of oscillators, a measured response).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a peak negative or not finite, a non-positive
    /// reference frequency, or a response whose spectrum has no positive maximum.
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::optical::raman::RamanGain;
    /// // a single damped oscillator at 10 THz, 1 ps of decay
    /// let (f0, tau) = (10e12, 1e-12);
    /// let g = RamanGain::from_response(|t| (-t / tau).exp() * (2.0 * std::f64::consts::PI * f0 * t).sin(), 1e-14, 206e12)?;
    /// let (f_peak, g_peak) = g.peak();
    /// assert!((f_peak - 10e12).abs() <= 0.1e12 && (g_peak - 1e-14).abs() < 1e-30);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn from_response(h_r: impl Fn(f64) -> f64, peak_m_w: f64, reference_hz: f64) -> Result<RamanGain> {
        const DT_S: f64 = 0.5e-15;
        const N_T: usize = 8000;
        const DF_HZ: f64 = 0.1e12;
        const N_F: usize = 421;
        ensure(peak_m_w.is_finite() && peak_m_w >= 0.0, || {
            format!("the peak of the Raman gain must be finite and non-negative, got {peak_m_w}")
        })?;
        let h: Vec<f64> = (0..N_T).map(|k| h_r(k as f64 * DT_S)).collect();
        let mut product = vec![0.0; N_T];
        let offsets: Vec<f64> = (0..N_F).map(|k| k as f64 * DF_HZ).collect();
        let mut gain: Vec<f64> = offsets
            .iter()
            .map(|&df| {
                // sin(ω k Δt) by the recurrence s_{k+1} = 2 cos(ω Δt) s_k - s_{k-1}
                let step = 2.0 * std::f64::consts::PI * df * DT_S;
                let two_cos = 2.0 * step.cos();
                let (mut s_prev, mut s) = (-step.sin(), 0.0);
                for (p, hk) in product.iter_mut().zip(&h) {
                    *p = hk * s;
                    (s_prev, s) = (s, two_cos * s - s_prev);
                }
                trapezoid_uniform(&product, DT_S)
            })
            .collect();
        let max = gain.iter().copied().fold(0.0, f64::max);
        ensure(max > 0.0 && max.is_finite(), || "the Raman response must have a positive gain spectrum".into())?;
        for g in &mut gain {
            // the models give the shape of the gain; a small negative lobe of the
            // numerical integral (the tail of the time window) is a zero gain
            *g = (*g / max * peak_m_w).max(0.0);
        }
        RamanGain::new(offsets, gain, reference_hz)
    }

    /// The gain spectrum of a [`RamanResponse`] model of silica, scaled to the peak
    /// `peak_m_w` ($g_{R} A_{\mathrm{eff}}$, m/W) for a pump at `reference_hz`: the shape
    /// of each model is computed once per process, the call costs a copy.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a peak negative or not finite, or a
    /// non-positive reference frequency.
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::optical::raman::{RamanGain, RamanResponse};
    /// let ssmf = RamanGain::ssmf();
    /// let (f_ref, peak) = (ssmf.reference_hz(), ssmf.peak().1);
    /// let bw = RamanGain::model(RamanResponse::BlowWood, peak, f_ref)?;
    /// let la = RamanGain::model(RamanResponse::LinAgrawal, peak, f_ref)?;
    /// let hc = RamanGain::model(RamanResponse::HollenbeckCantrell, peak, f_ref)?;
    /// // every model has the peak of silica
    /// assert!((bw.peak().0 - 13.1e12).abs() <= 0.1e12 && bw.peak().1 == peak);
    /// // the Lorentzian misses the shoulder at 3 THz, the boson peak brings it within 25 %
    /// assert!(bw.at(3e12) < 0.6 * ssmf.at(3e12) && (la.at(3e12) / ssmf.at(3e12) - 1.0).abs() < 0.25);
    /// // the band at 24 THz above the dip at 21.5 THz, which a single Lorentzian does not have
    /// assert!(hc.at(24e12) > 1.5 * hc.at(21.5e12) && bw.at(24e12) < bw.at(21.5e12));
    /// // the names of the specifications
    /// assert_eq!("lin_agrawal".parse::<RamanResponse>()?, RamanResponse::LinAgrawal);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn model(kind: RamanResponse, peak_m_w: f64, reference_hz: f64) -> Result<RamanGain> {
        ensure(reference_hz > 0.0, || "the reference frequency must be positive".into())?;
        let mut g = kind.unit_shape().clone().scaled(peak_m_w)?;
        g.reference_hz = reference_hz;
        Ok(g)
    }

    /// $g_{R} A_{\mathrm{eff}}$ (m/W) at the offset `df_hz` for a pump at the reference
    /// frequency (zero outside the table).
    pub fn at(&self, df_hz: f64) -> f64 {
        let (x, y) = (&self.offsets_hz, &self.gain_m_w);
        if df_hz < x[0] || df_hz > x[x.len() - 1] {
            return 0.0;
        }
        let (i, t) = super::segment(x, df_hz, |v| *v);
        y[i] + t * (y[i + 1] - y[i])
    }

    /// The gain efficiency $C$ (1/(W·km)) of a wave at `f_s` from a pump at `f_p >
    /// f_s`, for the effective area `aeff_m2`: $g_{R}(f_{p} - f_{s}) f_{p} / (f_{\mathrm{ref}}
    /// A_{\mathrm{eff}})$, or without the factor $f_{p}/f_{\mathrm{ref}}$
    /// ([`RamanGain::with_pump_scaling`]).
    pub fn efficiency(&self, f_s: f64, f_p: f64, aeff_m2: f64) -> f64 {
        if self.pump_scaling {
            self.at(f_p - f_s) * f_p / self.reference_hz / aeff_m2 * 1e3
        } else {
            self.at(f_p - f_s) / aeff_m2 * 1e3
        }
    }

    /// The frequency of the pump of the measurement or of the peak, $f_{\mathrm{ref}}$ (Hz).
    pub fn reference_hz(&self) -> f64 {
        self.reference_hz
    }

    /// The offset of the peak of the gain (Hz) and its $g_{R} A_{\mathrm{eff}}$ (m/W).
    pub fn peak(&self) -> (f64, f64) {
        self.offsets_hz
            .iter()
            .zip(&self.gain_m_w)
            .fold((0.0, 0.0), |(fo, go), (f, g)| if *g > go { (*f, *g) } else { (fo, go) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rms error of a model against the table of GNPy over `[0, f_max]`, relative to
    /// the peak of the table.
    fn rms_error(model: &RamanGain, f_max_hz: f64) -> f64 {
        let ssmf = RamanGain::ssmf();
        let peak = ssmf.peak().1;
        let errors: Vec<f64> = ssmf
            .offsets_hz
            .iter()
            .zip(&ssmf.gain_m_w)
            .filter(|(f, _)| **f <= f_max_hz)
            .map(|(f, g)| (model.at(*f) - g) / peak)
            .collect();
        (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt()
    }

    #[test]
    fn the_numerical_spectrum_of_blow_wood_is_its_lorentzian() {
        let g = RamanGain::model(RamanResponse::BlowWood, 1.0, 206e12).unwrap();
        let (a, b) = (1.0 / 32e-15, 1.0 / 12.2e-15);
        let lorentzian = |w: f64| 2.0 * a * b * w / ((a * a + b * b - w * w).powi(2) + 4.0 * a * a * w * w);
        let w_peak = 2.0 * std::f64::consts::PI * g.peak().0;
        for df_thz in [1.0, 5.0, 10.0, 13.0, 16.0, 25.0, 40.0] {
            let w = 2.0 * std::f64::consts::PI * df_thz * 1e12;
            let expected = lorentzian(w) / lorentzian(w_peak);
            assert!(
                (g.at(df_thz * 1e12) - expected).abs() < 2e-3,
                "{df_thz} THz: {} vs {expected}",
                g.at(df_thz * 1e12)
            );
        }
    }

    #[test]
    fn each_model_keeps_more_of_the_measured_spectrum_than_the_previous_one() {
        let ssmf = RamanGain::ssmf();
        let (f_ref, peak) = (ssmf.reference_hz(), ssmf.peak().1);
        let models = [RamanResponse::BlowWood, RamanResponse::LinAgrawal, RamanResponse::HollenbeckCantrell];
        let rms: Vec<(f64, f64)> = models
            .iter()
            .map(|&kind| {
                let g = RamanGain::model(kind, peak, f_ref).unwrap();
                let (f, p) = g.peak();
                assert!((p - peak).abs() < 1e-28 && (f - 13.25e12).abs() <= 0.5e12, "{kind:?}: peak at {f}");
                assert_eq!(g.reference_hz(), f_ref);
                (rms_error(&g, 10e12), rms_error(&g, 42e12))
            })
            .collect();
        // the boson peak halves the error below 10 THz, the thirteen modes that of the whole spectrum
        assert!(rms[1].0 < 0.5 * rms[0].0 && rms[2].0 < 0.5 * rms[0].0, "{rms:?}");
        assert!(rms[2].1 < 0.5 * rms[1].1 && (rms[1].1 - rms[0].1).abs() < 0.01, "{rms:?}");
    }

    #[test]
    fn the_linear_gain_is_the_slope_of_the_closed_form_to_14_thz() {
        let g = RamanGain::linear(0.028, 80.0, 193.1e12).unwrap();
        assert!((g.efficiency(193.1e12 - 10e12, 193.1e12, 80e-12) - 0.28).abs() < 1e-12);
        assert!((g.at(14e12) / (0.028 * 14.0 * 80e-12 / 1e3) - 1.0).abs() < 1e-12 && g.at(14.1e12) == 0.0);
        assert!(RamanGain::linear(-0.1, 80.0, 193.1e12).is_err());
    }

    #[test]
    fn the_constructors_reject_a_bad_peak_and_a_bad_reference() {
        assert!(RamanGain::model(RamanResponse::BlowWood, -1.0, 206e12).is_err());
        assert!(RamanGain::model(RamanResponse::LinAgrawal, f64::NAN, 206e12).is_err());
        assert!(RamanGain::model(RamanResponse::HollenbeckCantrell, 1e-14, 0.0).is_err());
        assert!(RamanGain::from_response(|_| 0.0, 1e-14, 206e12).is_err());
        assert!("stolen".parse::<RamanResponse>().is_err());
    }
}
