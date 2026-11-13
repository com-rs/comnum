//! The closed forms of the GN model: Poggiolini's ([`closed_form_nli_psd`]) with the
//! coherence factor of the accumulation over the spans ([`coherence_factor`]), and that
//! of the ISRS GN model of Semrau, Killey and Bayvel ([`isrs_closed_form_terms`],
//! [`isrs_closed_form_eta`]). The model and the references are those of the parent
//! module.

use super::WdmSpectrum;
use crate::optical::fiber::FiberParams;
use comnum_core::{Result, ensure};
use std::f64::consts::PI;

/// Poggiolini's closed form of the NLI power spectral density at the centre of a WDM
/// comb of `n_channels` rectangular channels spaced by `spacing` (see the module),
/// incoherent accumulation (W/Hz).
pub fn closed_form_nli_psd(
    fiber: &FiberParams,
    n_channels: usize,
    symbol_rate: f64,
    spacing: f64,
    power_w: f64,
) -> f64 {
    let a = fiber.alpha_np_km();
    let b2 = fiber.beta2_s2_km().abs();
    let l_eff = fiber.effective_length_km();
    let l_eff_a = 1.0 / a;
    let g = power_w / symbol_rate;
    let arg = PI * PI / 2.0
        * b2
        * l_eff_a
        * symbol_rate
        * symbol_rate
        * (n_channels as f64).powf(2.0 * symbol_rate / spacing);
    fiber.n_spans as f64 * 8.0 / 27.0 * fiber.gamma_w_km.powi(2) * g.powi(3) * l_eff * l_eff * arg.asinh()
        / (PI * b2 * l_eff_a)
}

/// The coherence factor $\varepsilon$ of the accumulation of the self-channel
/// interference over $N_{s}$ spans, $\eta_{N_{s}} \approx \eta_{1} N_{s}^{1 +
/// \varepsilon}$, over a bandwidth $B$ (Poggiolini 2012, eq. (22), the Nyquist-WDM
/// approximation of eq. (21) with its eqs. (13), (17) and (20)):
///
/// $$
/// \varepsilon \approx \frac{1}{\ln N_{s}} \ln\left[1 + \frac{2 L_{\mathrm{eff},a}}{N_{s} L_{s}}
///   \frac{1 - N_{s} + N_{s} H_{N_{s} - 1}}{\operatorname{asinh}\left(\frac{\pi^{2}}{2}
///   |\beta_{2}| L_{\mathrm{eff},a} B^{2}\right)}\right],
/// $$
///
/// with $L_{\mathrm{eff},a} = 1/\alpha$ and $H_{k}$ the $k$-th harmonic number. The paper
/// writes it for the whole WDM band $B_{\mathrm{WDM}}$; Semrau, Killey & Bayvel 2019
/// (Sec. II-B) take it over the bandwidth $B_{i}$ of the channel, where only the SPM
/// accumulates coherently: for the link of their Table I (40 GBd, 6 × 100 km of SMF)
/// it gives 0.151, the 0.15 of their Sec. III-A. 0 for one span (where the factor
/// does not matter).
///
/// # References
///
/// 1. P. Poggiolini, "The GN model of non-linear propagation in uncompensated coherent
///    optical systems," *J. Lightw. Technol.*, 30(24), 3857–3879, 2012, eq. (22).
pub fn coherence_factor(fiber: &FiberParams, bandwidth_hz: f64) -> f64 {
    let ns = fiber.n_spans;
    if ns < 2 {
        return 0.0;
    }
    let n = ns as f64;
    let l_eff_a = 1.0 / fiber.alpha_np_km();
    let harmonic: f64 = (1..ns).map(|k| 1.0 / k as f64).sum();
    let arg = PI * PI / 2.0 * fiber.beta2_s2_km().abs() * l_eff_a * bandwidth_hz * bandwidth_hz;
    (1.0 + 2.0 * l_eff_a / (n * fiber.span_km) * (1.0 - n + n * harmonic) / arg.asinh()).ln() / n.ln()
}

/// The closed form of the ISRS GN model of one span (Semrau, Killey & Bayvel 2019): the
/// NLI coefficients $(\eta_{\mathrm{SPM}}, \eta_{\mathrm{XPM}})$ of channel `ch`
/// (1/W², $P_{\mathrm{NLI}} = \eta P_{i}^{3}$ in the bandwidth $B_{i}$ of the channel) of
/// eqs. (10) and (11), the XPM summed over the other channels of `spectrum`;
/// [`isrs_closed_form_eta`] combines them over the spans. Every channel has the
/// bandwidth $B = R_{s}(1 + \beta)$ of `spectrum`, $P_{\mathrm{tot}}$ is the sum of its
/// powers and $f$ the offset from the carrier of `fiber`, whose $\alpha$, $\beta_{2}$,
/// $\beta_{3}$ (from the slope) and $\gamma$ are used; `raman_slope_w_km_thz` is the
/// slope $C_{r}$ of the linear regression of the normalised Raman gain spectrum
/// (1/W/km/THz, 0.028 for SMF), 0 for no ISRS. With $T_{k} = 2 - f_{k} P_{\mathrm{tot}}
/// C_{r} / \alpha$, $\phi_{i} = 12 \pi^{2} (\beta_{2} + 2 \pi \beta_{3} f_{i})$ and
/// $\phi_{i,k} = 2 \pi^{2} (f_{k} - f_{i}) [\beta_{2} + \pi \beta_{3} (f_{i} + f_{k})]$
/// (both taken in absolute value: the XPM of an interferer does not depend on the side
/// it lies on, up to $\beta_{3}$):
///
/// $$
/// \eta_{\mathrm{SPM}}(f_{i}) \approx \frac{16}{27} \frac{\gamma^{2} \pi}{B_{i}^{2} \alpha
///   |\phi_{i}|} \left(T_{i}^{2} - \frac{4}{9}\right) \operatorname{asinh}\left(\frac{B_{i}^{2}
///   |\phi_{i}|}{16 \alpha}\right),
/// $$
///
/// $$
/// \eta_{\mathrm{XPM}}(f_{i}) \approx \frac{32}{27} \frac{\gamma^{2}}{\alpha} \sum_{k \ne i}
///   \left(\frac{P_{k}}{P_{i}}\right)^{2} \frac{1}{B_{k} |\phi_{i,k}|} \left[\frac{T_{k}^{2} - 1}{3}
///   \operatorname{atan}\left(\frac{B_{i} |\phi_{i,k}|}{\alpha}\right) + \frac{4 - T_{k}^{2}}{6}
///   \operatorname{atan}\left(\frac{B_{i} |\phi_{i,k}|}{2 \alpha}\right)\right].
/// $$
///
/// The assumptions of the paper (Sec. II-C): rectangular channel spectra, no four-wave
/// mixing between two interferers (the XPM assumption), $|f_{k} - f_{i}| \gg B_{k}/2$,
/// weak ISRS (the power profile to first order in $P_{\mathrm{tot}} C_{r}
/// L_{\mathrm{eff}}$, a power transfer $\Delta\rho(L)$ well below 26 dB, eq. (12)), a
/// profile that depends on $P_{\mathrm{tot}}$ only, and $e^{-\alpha L} \ll 1$ (the span
/// length does not enter). Eq. (10) of the preprint (arXiv 1808.07940v2) ends with a
/// term $B_{i}^{2}/(9 \alpha^{2})$ inside the bracket, which is not dimensionless; placed
/// outside the bracket, where it is, it adds $16 \gamma^{2} / (243 \alpha^{2})$ and
/// lifts the curves of Fig. 4a by 0.6 dB above the paper's own; it is left out, and the
/// SPM alone is then 0.6 dB below the integral of [`GnModel`](super::GnModel) of one channel, which the
/// XPM (9 dB above it) hides in the total: the closed form follows the dashed curves of
/// Figs. 4a, 4b and 5 within 0.3 dB away from the band edges
/// (`validation/tests/fiber/semrau2019.rs`).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a missing channel, a negative or non-finite slope,
/// two channels at the same frequency, or a channel at the zero-dispersion frequency
/// ($\phi_{i} = 0$); [`comnum_core::Error::Shape`] and [`comnum_core::Error::Config`] as
/// [`GnModel::nli_psd`](super::GnModel::nli_psd) for an invalid spectrum.
///
/// # Examples
/// ```
/// use comnum_channel::optical::fiber::FiberParams;
/// use comnum_channel::optical::gn::{WdmSpectrum, isrs_closed_form_terms};
/// // five 40 GBd channels of 1 mW on a 40.005 GHz grid, 100 km of SMF, no ISRS
/// let fiber = FiberParams { span_km: 100.0, gamma_w_km: 1.2, ..FiberParams::default() };
/// let freqs: Vec<f64> = (-2..=2).map(|k| k as f64 * 40.005e9).collect();
/// let s = WdmSpectrum::uniform(freqs, 1e-3, 40e9, 1e-4);
/// let (spm, xpm) = isrs_closed_form_terms(&fiber, &s, 2, 0.0).unwrap();
/// assert!(spm > 0.0 && xpm > spm);
/// ```
///
/// # References
///
/// 1. D. Semrau, R. I. Killey and P. Bayvel, "A closed-form approximation of the Gaussian
///    noise model in the presence of inter-channel stimulated Raman scattering," *J.
///    Lightw. Technol.*, 37(9), 1924–1936, 2019 (arXiv 1808.07940), eqs. (10)–(11).
pub fn isrs_closed_form_terms(
    fiber: &FiberParams,
    spectrum: &WdmSpectrum,
    ch: usize,
    raman_slope_w_km_thz: f64,
) -> Result<(f64, f64)> {
    spectrum.check()?;
    let n = spectrum.freqs_hz.len();
    ensure(ch < n, || format!("no channel {ch}: the spectrum has {n} channels"))?;
    ensure(raman_slope_w_km_thz.is_finite() && raman_slope_w_km_thz >= 0.0, || {
        format!("the Raman gain slope must be finite and non-negative, got {raman_slope_w_km_thz}")
    })?;
    // SI per metre: α (1/m), β2 (s²/m), β3 (s³/m), γ (1/W/m), C_r (1/W/m/Hz)
    let alpha = fiber.alpha_np_km() / 1e3;
    let beta2 = fiber.beta2_s2_km() / 1e3;
    let beta3 = fiber.beta3_s3_km() / 1e3;
    let gamma = fiber.gamma_w_km / 1e3;
    let c_r = raman_slope_w_km_thz / 1e3 / 1e12;
    let b = spectrum.symbol_rate * (1.0 + spectrum.rolloff);
    let p_tot: f64 = spectrum.powers_w.iter().sum();
    let f_i = spectrum.freqs_hz[ch];
    let p_i = spectrum.powers_w[ch];
    let t = |f: f64| 2.0 - f * p_tot * c_r / alpha;
    // SPM, eq. (10) without its last term (see above)
    let phi_i = (12.0 * PI * PI * (beta2 + 2.0 * PI * beta3 * f_i)).abs();
    ensure(phi_i > 0.0, || format!("channel {ch} at {f_i} Hz sits at the zero-dispersion frequency"))?;
    let t_i = t(f_i);
    let spm = 16.0 / 27.0 * gamma * gamma * PI / (b * b * alpha * phi_i)
        * (t_i * t_i - 4.0 / 9.0)
        * (b * b * phi_i / (16.0 * alpha)).asinh();
    // XPM, eq. (11)
    let mut xpm = 0.0;
    for (k, (&f_k, &p_k)) in spectrum.freqs_hz.iter().zip(&spectrum.powers_w).enumerate() {
        if k == ch {
            continue;
        }
        let phi_ik = (2.0 * PI * PI * (f_k - f_i) * (beta2 + PI * beta3 * (f_i + f_k))).abs();
        ensure(phi_ik > 0.0, || format!("channels {ch} and {k} have the same phase mismatch ({f_i} and {f_k} Hz)"))?;
        let t_k = t(f_k);
        let ratio = p_k / p_i;
        xpm += ratio * ratio / (b * phi_ik)
            * ((t_k * t_k - 1.0) / 3.0 * (b * phi_ik / alpha).atan()
                + (4.0 - t_k * t_k) / 6.0 * (b * phi_ik / (2.0 * alpha)).atan());
    }
    xpm *= 32.0 / 27.0 * gamma * gamma / alpha;
    Ok((spm, xpm))
}

/// The NLI coefficient $\eta_{n}(f_{i})$ (1/W²) of channel `ch` after the `n` spans of
/// `fiber` by the closed form of the ISRS GN model (Semrau, Killey & Bayvel 2019,
/// eq. (5)): the SPM of [`isrs_closed_form_terms`] accumulated coherently with the
/// coherence factor `epsilon` ([`coherence_factor`] over the bandwidth of the channel),
/// the XPM incoherently,
///
/// $$
/// \eta_{n}(f_{i}) \approx \eta_{\mathrm{SPM}}(f_{i})\\, n^{1 + \varepsilon} +
///   \eta_{\mathrm{XPM}}(f_{i})\\, n.
/// $$
///
/// The SNR of the channel is then $P_{i} / (P_{\mathrm{ASE}} + \eta_{n} P_{i}^{3})$
/// (eq. (1)); the PSD per W³ of [`GnModel::eta`](super::GnModel::eta) is $\eta_{n} / B_{i}$. The paper
/// neglects the change of $\varepsilon$ by ISRS (0.012 at 6.5 dB of power transfer).
///
/// # Errors
/// As [`isrs_closed_form_terms`], and [`comnum_core::Error::Config`] for a non-finite
/// `epsilon`.
///
/// # Examples
/// ```
/// use comnum_channel::optical::fiber::FiberParams;
/// use comnum_channel::optical::gn::{WdmSpectrum, coherence_factor, isrs_closed_form_eta};
/// let fiber = FiberParams { span_km: 100.0, n_spans: 6, gamma_w_km: 1.2, ..FiberParams::default() };
/// let freqs: Vec<f64> = (-2..=2).map(|k| k as f64 * 40.005e9).collect();
/// let s = WdmSpectrum::uniform(freqs, 1e-3, 40e9, 1e-4);
/// let eps = coherence_factor(&fiber, 40.004e9);
/// let eta = isrs_closed_form_eta(&fiber, &s, 2, 0.028, eps).unwrap();
/// assert!(eps > 0.1 && eps < 0.2 && eta > 0.0);
/// ```
///
/// # References
///
/// 1. D. Semrau, R. I. Killey and P. Bayvel, "A closed-form approximation of the Gaussian
///    noise model in the presence of inter-channel stimulated Raman scattering," *J.
///    Lightw. Technol.*, 37(9), 1924–1936, 2019 (arXiv 1808.07940), eq. (5).
pub fn isrs_closed_form_eta(
    fiber: &FiberParams,
    spectrum: &WdmSpectrum,
    ch: usize,
    raman_slope_w_km_thz: f64,
    epsilon: f64,
) -> Result<f64> {
    ensure(epsilon.is_finite(), || format!("the coherence factor must be finite, got {epsilon}"))?;
    let (spm, xpm) = isrs_closed_form_terms(fiber, spectrum, ch, raman_slope_w_km_thz)?;
    let n = fiber.n_spans as f64;
    Ok(spm * n.powf(1.0 + epsilon) + xpm * n)
}
