//! The GN and EGN models of the nonlinear interference, the closed form of Poggiolini,
//! the closed form of the ISRS GN model, the moments of a format, the Raman profiles of a
//! span and the WDM grid.

use crate::fiber::PyFiber;
use crate::{FArr, IntoPy, count, farr, fvec, json};
use comnum::channel::optical::gn::{
    FormatMoments, GnModel, WdmSpectrum, closed_form_nli_psd, coherence_factor, isrs_closed_form_eta,
    isrs_closed_form_terms,
};
use comnum::channel::optical::raman::{DesignObjective, RamanGain};
use comnum::math::dbm2w;
use comnum::spec::blocks::BuildContext;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The GN model of the nonlinear interference of a WDM link.
///
/// The NLI power spectral density at the receiver is the double integral of the GN
/// reference formula [1]_,
///
/// .. math::
///
///     G_{\mathrm{NLI}}(f) = \frac{16}{27} \gamma^2 \iint G(f_1) G(f_2) G(f_1 + f_2 - f)\,
///     |\rho(f_1, f_2, f)|^2\, \chi(f_1, f_2, f)\, df_1\, df_2,
///
/// with :math:`G` the power spectral density of the transmitted signal (both
/// polarisations), the span efficiency and the phased-array factor of the spans
///
/// .. math::
///
///     \begin{aligned}
///     \rho &= \frac{1 - e^{-\alpha L_s} e^{j 4 \pi^2 \beta_2 L_s \Delta}}
///     {\alpha - j 4 \pi^2 \beta_2 \Delta}, \qquad \Delta = (f_1 - f)(f_2 - f), \\
///     \chi &= \frac{\sin^2(2 N_s \pi^2 \beta_2 L_s \Delta)}{\sin^2(2 \pi^2 \beta_2 L_s \Delta)},
///     \end{aligned}
///
/// computed on a grid, with the coherent accumulation of the spans (``coherent=True``,
/// the phased-array factor :math:`\chi`) or the incoherent one (:math:`\chi = N_s`, :math:`N_s` times a
/// span). Channels have raised-cosine spectra of roll-off ``rolloff`` at the symbol rate
/// ``symbol_rate``, centred on ``freqs`` (Hz, relative to the carrier), with the launch
/// powers ``power_dbm`` (one per channel or one for all, both polarisations).
///
/// With ``raman`` (the keys of ``fiber.raman``, see :func:`raman_profile`), the power
/// profiles of the channels along the span replace the exponential decay in the link
/// function (the generalised GN model of [2]_), the noise is the
/// spontaneous Raman noise plus the ASE of the EDFA that restores every channel, and
/// the channels are restored to their launch power after every span.
///
/// With ``modulation`` (a constellation, a name such as ``"16-QAM"``, or
/// ``"gaussian"``) and optionally its probabilities ``px``, the NLI is that of the EGN
/// model [3]_: the GN model corrected for the fourth and sixth moments
/// of the symbols (see :func:`format_moments`), for the self-channel interference and
/// the cross-channel interference of every other channel.
///
/// Args:
///     fiber (FiberParams): the link.
///     freqs (array_like): centre frequencies of the channels relative to the carrier
///         (Hz).
///     power_dbm (float | array_like): launch power of every channel (dBm, both
///         polarisations), one for all or one per channel.
///     symbol_rate (float): symbol rate of every channel (Bd).
///     rolloff (float): roll-off of the raised-cosine spectra.
///     channel (int, optional): the channel of interest (the middle one by default).
///     coherent (bool): coherent accumulation of the spans.
///     resolution_hz (float, optional): step of the integration grid (Hz).
///     raman (dict, optional): Raman pumps and ISRS (see :func:`raman_profile`).
///     modulation (Constellation | str, optional): the modulation format of the EGN
///         model.
///     px (array_like, optional): the probabilities of its points.
///
/// Returns:
///     dict: ``nli_psd`` and ``ase_psd`` (W/Hz, both polarisations, at the frequency of
///     channel ``channel``), ``snr_db`` (after a matched filter), ``eta`` (NLI PSD per
///     W³ of launch power per channel) and ``p_opt_dbm`` (the optimal launch power per
///     channel for equal powers; with ``raman``, ``eta`` and ``p_opt_dbm`` are those of
///     the profiles of 1 mW per channel).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] P. Poggiolini, "The GN model of non-linear propagation in uncompensated
///        coherent optical systems," *J. Lightw. Technol.*, 30(24), 3857–3879, 2012.
///     .. [2] D. Semrau, R. I. Killey and P. Bayvel, "A closed-form approximation of the
///        Gaussian noise model in the presence of inter-channel stimulated Raman
///        scattering," *J. Lightw. Technol.*, 37(9), 1924–1936, 2019.
///     .. [3] A. Carena, G. Bosco, V. Curri, Y. Jiang, P. Poggiolini and F. Forghieri, "EGN
///        model of non-linear fiber propagation," *Opt. Express*, 22(13), 16335–16362,
///        2014.
#[pyfunction]
#[pyo3(signature = (fiber, freqs, power_dbm, symbol_rate, *, rolloff = 0.1, channel = None, coherent = true, resolution_hz = None, raman = None, modulation = None, px = None))]
#[allow(clippy::too_many_arguments, reason = "the link, the grid and the options of the model, by keyword")]
fn gn_model<'py>(
    py: Python<'py>,
    fiber: &PyFiber,
    freqs: &Bound<'py, PyAny>,
    power_dbm: &Bound<'py, PyAny>,
    symbol_rate: f64,
    rolloff: f64,
    channel: Option<usize>,
    coherent: bool,
    resolution_hz: Option<f64>,
    raman: Option<&Bound<'py, PyAny>>,
    modulation: Option<&Bound<'py, PyAny>>,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let freqs = fvec(freqs)?;
    let px = px.map(fvec).transpose()?;
    let n = freqs.len();
    let powers = powers_dbm(power_dbm, n)?;
    let egn = match modulation {
        None => None,
        Some(f) => Some(self::format_moments_of(py, f, px.as_deref())?),
    };
    let model =
        GnModel { fiber: fiber.inner.clone(), coherent, resolution_hz, raman: crate::fiber::raman(py, raman)?, egn };
    let spec =
        WdmSpectrum { freqs_hz: freqs, powers_w: powers.iter().map(|p| dbm2w(*p)).collect(), symbol_rate, rolloff };
    let s = py.detach(|| model.summary(&spec, channel)).py()?;
    let d = PyDict::new(py);
    d.set_item("nli_psd", s.nli_psd)?;
    d.set_item("ase_psd", s.ase_psd)?;
    d.set_item("snr_db", comnum::math::lin2db(s.snr))?;
    d.set_item("eta", s.eta)?;
    d.set_item("p_opt_dbm", comnum::math::w2dbm(s.optimal_power_w))?;
    Ok(d)
}

/// One launch power per channel: a scalar broadcast to `n` channels, or an array of `n`.
fn powers_dbm(power_dbm: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<f64>> {
    let powers: Vec<f64> = if let Ok(v) = power_dbm.extract::<f64>() { vec![v; n] } else { fvec(power_dbm)? };
    if powers.len() != n {
        return Err(PyValueError::new_err(format!("{} powers for {n} channels", powers.len())));
    }
    Ok(powers)
}

/// The EGN moments of a format: a :class:`Constellation`, a name (``"16-QAM"``) or
/// ``"gaussian"``, with the probabilities ``px`` of a shaped source.
fn format_moments_of(py: Python<'_>, f: &Bound<'_, PyAny>, px: Option<&[f64]>) -> PyResult<FormatMoments> {
    if let Ok(c) = f.extract::<PyRef<'_, crate::comm::PyConstellation>>() {
        return FormatMoments::of(c.inner.points(), px).py();
    }
    let j = json(py, f)?;
    if j.as_str().is_some_and(|s| s.eq_ignore_ascii_case("gaussian")) {
        return Ok(FormatMoments::GAUSSIAN);
    }
    let c = comnum::spec::blocks::parse_constellation(&j).py()?.normalized();
    FormatMoments::of(c.points(), px).py()
}

/// The moments of a modulation format that the EGN model needs.
///
/// In the EGN model [1]_, they are
///
/// .. math::
///
///     \Phi = \frac{\mathbb{E}|a|^4}{\mathbb{E}^2|a|^2} - 2, \qquad
///     \Psi = \frac{\mathbb{E}|a|^6}{\mathbb{E}^3|a|^2}
///          - 9\, \frac{\mathbb{E}|a|^4}{\mathbb{E}^2|a|^2} + 12,
///
/// both zero for Gaussian symbols.
///
/// Args:
///     modulation (Constellation | str): a constellation, a name such as ``"16-QAM"``,
///         or ``"gaussian"``.
///     px (array_like, optional): the probabilities of the points (a shaped source).
///
/// Returns:
///     tuple[float, float]: :math:`(\Phi, \Psi)`.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// References:
///     .. [1] A. Carena, G. Bosco, V. Curri, Y. Jiang, P. Poggiolini and F. Forghieri, "EGN
///        model of non-linear fiber propagation," *Opt. Express*, 22(13), 16335–16362,
///        2014.
///
/// Examples:
///     >>> import comnum
///     >>> [round(v, 4) for v in comnum.format_moments("16-QAM")]
///     [-0.68, 2.08]
#[pyfunction]
#[pyo3(signature = (modulation, *, px = None))]
fn format_moments(
    py: Python<'_>,
    modulation: &Bound<'_, PyAny>,
    px: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, f64)> {
    let px = px.map(fvec).transpose()?;
    let f = format_moments_of(py, modulation, px.as_deref())?;
    Ok((f.phi, f.psi))
}

/// The power profiles of a span with distributed Raman amplification: the signals, any
/// number of co- and counter-propagating pumps, the power they exchange and the spontaneous
/// Raman noise of every channel.
///
/// **The model.** Every wave :math:`k` (a channel, a slice of the spectrum of a field in
/// the split-step solvers, or a pump) of frequency :math:`f_k` and power :math:`P_k(z)` (W)
/// obeys the coupled power equations of stimulated Raman scattering ([1]_, eq. (1);
/// [2]_, ch. 8)
///
/// .. math::
///
///     \pm \frac{d P_k}{d z} = P_k \left(-\alpha_k + \sum_j C_{kj} P_j\right),
///
/// with :math:`+` for the waves that travel with the signals and :math:`-` for the
/// counter-propagating pumps, :math:`\alpha_k` the attenuation of the power at
/// :math:`f_k` (1/km) and :math:`C_{kj}` the Raman gain efficiency (1/(W·km)). A wave gains
/// power from the waves of higher frequency, at the rate of the gain spectrum of silica
/// at their frequency offset, and gives it to those of lower frequency:
///
/// .. math::
///
///     C_{kj} = \frac{g_R(f_j - f_k)}{A_{kj}}\, \frac{f_j}{f_{\mathrm{ref}}}, \qquad
///     A_{kj} = \frac{A_{\mathrm{eff}}(f_k) + A_{\mathrm{eff}}(f_j)}{2}, \qquad f_j > f_k,
///
/// with :math:`g_R` the measured gain spectrum of a pump at :math:`f_{\mathrm{ref}}`
/// (``gain``), scaled with the frequency of the pump, and :math:`A_{kj}` the overlap
/// area of the two modes. The wave of higher frequency loses the photons that the other
/// gains, :math:`C_{jk} = -(f_j/f_k)\, C_{kj}`: without loss,
/// :math:`\sum_k \pm P_k/f_k` (the photon flux) is constant along :math:`z`. With ``isrs``
/// the channels also pump each other (inter-channel stimulated Raman scattering), which
/// tilts the spectrum towards the low frequencies.
///
/// **The solution.** The co-propagating waves are known at :math:`z = 0` and the
/// counter-propagating pumps at :math:`z = L`, a two-point boundary-value problem: the
/// logarithms of the powers are integrated by the fourth-order Runge-Kutta method on a
/// grid of ``step_km``, forward for the co-propagating waves and backward for the
/// counter-propagating ones, sweep after sweep until they change by less than a tolerance
/// (the relaxation method of [3]_).
///
/// **The gains and the noise.** For channel :math:`k`, with
/// :math:`G_k(z) = P_k(z)/P_k(0)`, the net gain of the span is :math:`G_k(L)`, the on-off
/// gain is the net gain with the pumps and ISRS over that of the loss alone,
/// :math:`G_k(L)\, e^{\alpha_k L}`, and the spontaneous scattering of the pumps adds a noise
/// of power spectral density, both polarisations ([1]_, eq. (4)),
///
/// .. math::
///
///     N_k(L) = \int_0^L 2 h f_k \sum_{j \in \mathrm{pumps}} C_{kj} P_j(z)\,
///     (1 + \eta_{kj})\, \frac{G_k(L)}{G_k(z)}\, dz, \qquad
///     \eta_{kj} = \frac{1}{e^{h (f_j - f_k)/(k_B T)} - 1},
///
/// with :math:`\eta_{kj}` the phonon occupancy at the temperature :math:`T`: noise
/// generated at :math:`z` is amplified by the rest of the span. With ``rayleigh_db_km``, a
/// fraction :math:`\kappa` (1/km) of the signal is backscattered by Rayleigh scattering,
/// and the light scattered twice travels with the signal again; its power over that of the
/// signal at the end of the span is ([1]_, Sec. V-A)
///
/// .. math::
///
///     R_k^2 = \kappa^2 \int_0^L \frac{1}{G_k(z)^2} \int_z^L G_k(z')^2\, dz'\, dz.
///
/// Args:
///     fiber (FiberParams): the span (length, attenuation, carrier frequency).
///     freqs (array_like): centre frequencies of the channels relative to the carrier
///         (Hz).
///     power_dbm (float | array_like): launch power of every channel (dBm, both
///         polarisations).
///     raman (dict): the keys of ``fiber.raman`` in a specification: ``pumps`` (a list of
///         ``{wavelength_nm | frequency_thz, power_mw, direction: "co" | "counter",
///         alpha_db_km}``), ``aeff_um2`` (80), ``aeff_spectrum`` (``{frequency_thz,
///         aeff_um2}``, the effective area against frequency, linear between the points;
///         ``aeff_um2`` everywhere by default), ``alpha_spectrum`` (``{frequency_thz,
///         alpha_db_km}``, the attenuation of the signals and of the pumps without their
///         own against frequency; that of ``fiber`` by default), ``temperature_k`` (300),
///         ``isrs`` (True:
///         the channels also exchange power), ``step_km`` (0.1), ``slice_hz`` (25e9, for
///         the split-step solvers), ``gain`` (the gain spectrum :math:`g_R A_{\mathrm{eff}}`
///         of a pump at ``reference_thz``: a measured table ``{offsets_thz, gain_m_w}`` or a
///         response-function model of silica, ``model`` one of ``"blow_wood"`` (one damped
///         oscillator, the peak only), ``"lin_agrawal"`` (plus the boson peak below 5 THz)
///         and ``"hollenbeck_cantrell"`` (13 vibrational modes, the whole spectrum to 40
///         THz), scaled to ``peak_m_w`` (the peak of standard single-mode fibre, 3.33e-14,
///         by default); the efficiency scales with the frequency of the pump unless
///         ``pump_scaling`` is False; the measured spectrum of standard single-mode fibre
///         by default) and ``rayleigh_db_km`` (the Rayleigh
///         backscatter coupling of the signals, :math:`10 \log_{10}` of its value in 1/km,
///         about -40 for transmission fibre;
///         absent: no double Rayleigh backscattering).
///
/// Returns:
///     dict: ``z_km``, ``freqs_hz`` (absolute, the channels then the pumps),
///     ``power_dbm`` (one row per wave), ``net_gain_db`` and ``on_off_gain_db`` (per
///     channel), ``ase_psd`` (W/Hz, both polarisations, per channel at the end of the
///     span), ``ase_backward_psd`` (the spontaneous noise emitted towards the input of
///     the span, at its start), ``double_rayleigh`` (the effective Rayleigh reflection coefficient of every
///     channel, the doubly backscattered power over the signal power at the end of the
///     span, linear; 0 without ``rayleigh_db_km``), ``noise_figure_db`` (of the span,
///     [1]_, eq. (9)), ``effective_noise_figure_db`` (the noise figure minus the
///     loss of the span, eq. (12)) and ``iterations``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] J. Bromage, "Raman amplification for fiber communications systems," *J.
///        Lightw. Technol.*, 22(1), 79–93, 2004.
///     .. [2] G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013.
///     .. [3] V. E. Perlin and H. G. Winful, "Optimal design of flat-gain wide-band fiber
///        Raman amplifiers," *J. Lightw. Technol.*, 20(2), 250–254, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> p = comnum.raman_profile(comnum.FiberParams(span_km=100), [0.0], -10.0,
///     ...     {"pumps": [{"wavelength_nm": 1450, "power_mw": 400, "direction": "counter"}]})
///     >>> bool(10 < p["on_off_gain_db"][0] < 16)
///     True
#[pyfunction]
fn raman_profile<'py>(
    py: Python<'py>,
    fiber: &PyFiber,
    freqs: &Bound<'py, PyAny>,
    power_dbm: &Bound<'py, PyAny>,
    raman: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let freqs = fvec(freqs)?;
    let n = freqs.len();
    let powers = powers_dbm(power_dbm, n)?;
    let r = crate::fiber::raman(py, Some(raman))?.ok_or_else(|| PyValueError::new_err("raman must be a mapping"))?;
    let f = fiber.inner.clone();
    let signals: Vec<(f64, f64)> = freqs.iter().zip(&powers).map(|(fr, p)| (f.fc_hz + fr, dbm2w(*p))).collect();
    let p = py.detach(|| r.solve(&f, &signals)).py()?;
    let d = PyDict::new(py);
    let n_waves = p.log_power.len();
    let n_z = p.z_km.len();
    let flat: Vec<f64> = (0..n_waves).flat_map(|k| p.power_dbm(k)).collect();
    d.set_item("z_km", farr(py, p.z_km.clone()))?;
    d.set_item("freqs_hz", farr(py, p.freqs_hz.clone()))?;
    d.set_item(
        "power_dbm",
        numpy::ndarray::Array2::from_shape_vec((n_waves, n_z), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    d.set_item("net_gain_db", farr(py, (0..n).map(|k| p.net_gain_db(k)).collect()))?;
    d.set_item("on_off_gain_db", farr(py, (0..n).map(|k| p.on_off_gain_db(k)).collect()))?;
    d.set_item("ase_psd", farr(py, p.ase_psd.clone()))?;
    d.set_item("ase_backward_psd", farr(py, p.ase_backward_psd.clone()))?;
    d.set_item("double_rayleigh", farr(py, p.double_rayleigh.clone()))?;
    d.set_item("noise_figure_db", farr(py, (0..n).map(|k| p.noise_figure_db(k)).collect()))?;
    d.set_item("effective_noise_figure_db", farr(py, (0..n).map(|k| p.effective_noise_figure_db(k)).collect()))?;
    d.set_item("iterations", p.iterations)?;
    Ok(d)
}

/// The pump powers that flatten the on-off gain of the channels over one span.
///
/// The pump-power optimiser of [1]_ at fixed
/// pump wavelengths [2]_: the
/// integrated pump powers that fit the targets (non-negative least squares or the
/// minimax fit of Lawson's iteratively reweighted least squares), then the launch powers
/// whose profiles have these integrals (Newton's method on the solved profiles), until
/// the integrals match within ``1e-5``. See ``comnum_channel::optical::raman::design``.
///
/// The on-off gain of channel :math:`k` is linear in the integrated powers :math:`I_j`
/// of the waves,
///
/// .. math::
///
///     \ln G^{\mathrm{on/off}}_k = \sum_{j \in \mathrm{pumps}} C_{kj} I_j
///     + \sum_{s \in \mathrm{signals}} C_{ks} I_s,
///
/// with :math:`I_j = \int_0^L P_j(z)\, dz` the power of wave :math:`j` integrated over the
/// span (W·km) and :math:`C_{kj}` the gain efficiencies of :func:`raman_profile`: dividing
/// the equation of :math:`P_k` by :math:`P_k` and integrating it over the span gives
/// :math:`\ln G_k(L) = -\alpha_k L + \sum_j C_{kj} I_j`, and the on-off gain removes the
/// loss. The targets fix the integrals of the pumps (a linear fit), then the launch powers
/// are searched whose solved profiles have these integrals: the design splits in two
/// ([1]_, Sec. II).
///
/// Args:
///     fiber (FiberParams): the span (length, attenuation, carrier frequency).
///     freqs (array_like): centre frequencies of the channels relative to the carrier
///         (Hz).
///     power_dbm (float | array_like): launch power of every channel (dBm).
///     raman (dict): the keys of :func:`raman_profile`; the pumps give the wavelengths
///         and directions, their ``power_mw`` is the initial guess when positive.
///     target_on_off_db (float | array_like): the target on-off gain of every channel
///         (dB), one value for all or one per channel (the loss of each over the span for
///         a flat net gain).
///     objective (str): ``"minimax"`` (the largest gain error) or ``"least_squares"``.
///
/// Returns:
///     dict: ``pump_power_mw`` (the powers found, in the order of the pumps),
///     ``pump_freqs_hz``, ``integrals_w_km`` (the target integrated power of every
///     pump), ``ripple_db`` (the peak-to-peak spread of the on-off gain minus its target
///     over the channels), ``on_off_gain_db`` and ``net_gain_db`` (per channel) and
///     ``iterations``.
///
/// Raises:
///     ValueError: for an invalid parameter, no pump, or a design that does not
///         converge.
///
/// References:
///     .. [1] V. E. Perlin and H. G. Winful, "Optimal design of flat-gain wide-band fiber
///        Raman amplifiers," *J. Lightw. Technol.*, 20(2), 250–254, 2002.
///     .. [2] J. Hu, B. S. Marks and C. R. Menyuk, "Flat-gain fiber Raman amplifiers using
///        equally spaced pumps," *J. Lightw. Technol.*, 22(6), 1519–1522, 2004.
///
/// Examples:
///     >>> import comnum
///     >>> pumps = [{"wavelength_nm": w, "power_mw": 0, "direction": "counter"} for w in (1430, 1460)]
///     >>> d = comnum.raman_design(comnum.FiberParams(), comnum.wdm_grid(8, 500e9), -10.0,
///     ...     {"pumps": pumps}, 10.0)
///     >>> bool(d["ripple_db"] < 1.0)
///     True
#[pyfunction]
#[pyo3(signature = (fiber, freqs, power_dbm, raman, target_on_off_db, objective = "minimax"))]
fn raman_design<'py>(
    py: Python<'py>,
    fiber: &PyFiber,
    freqs: &Bound<'py, PyAny>,
    power_dbm: &Bound<'py, PyAny>,
    raman: &Bound<'py, PyAny>,
    target_on_off_db: &Bound<'py, PyAny>,
    objective: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let freqs = fvec(freqs)?;
    let n = freqs.len();
    let powers = powers_dbm(power_dbm, n)?;
    let targets = powers_dbm(target_on_off_db, n)?;
    let objective = match objective {
        "minimax" => DesignObjective::Minimax,
        "least_squares" => DesignObjective::LeastSquares,
        other => {
            return Err(PyValueError::new_err(format!("objective must be minimax or least_squares, got {other:?}")));
        }
    };
    let r = crate::fiber::raman(py, Some(raman))?.ok_or_else(|| PyValueError::new_err("raman must be a mapping"))?;
    let f = fiber.inner.clone();
    let signals: Vec<(f64, f64)> = freqs.iter().zip(&powers).map(|(fr, p)| (f.fc_hz + fr, dbm2w(*p))).collect();
    let d = py.detach(|| r.design_pumps(&f, &signals, &targets, objective)).py()?;
    let out = PyDict::new(py);
    out.set_item("pump_power_mw", farr(py, d.pumps.iter().map(|p| p.power_w * 1e3).collect()))?;
    out.set_item("pump_freqs_hz", farr(py, d.pumps.iter().map(|p| p.frequency_hz).collect()))?;
    out.set_item("integrals_w_km", farr(py, d.integrals_w_km.clone()))?;
    out.set_item("ripple_db", d.ripple_db)?;
    out.set_item("on_off_gain_db", farr(py, (0..n).map(|k| d.profile.on_off_gain_db(k)).collect()))?;
    out.set_item("net_gain_db", farr(py, (0..n).map(|k| d.profile.net_gain_db(k)).collect()))?;
    out.set_item("iterations", d.iterations)?;
    Ok(out)
}

/// The Raman gain spectrum of a ``gain`` mapping, as :func:`raman_profile`,
/// :func:`gn_model`, ``Ssfm`` and ``ManakovSsfm`` use it.
///
/// **The model.** The power that a wave at :math:`f_s` gains from a wave at
/// :math:`f_p > f_s` along the fibre is :math:`dP_s/dz = C\, P_p P_s`, with the gain
/// efficiency (1/(W·km))
///
/// .. math::
///
///     C(f_s, f_p) = \frac{g_R(f_p - f_s)}{A_{\mathrm{eff}}}\, \frac{f_p}{f_{\mathrm{ref}}},
///
/// :math:`g_R(\Delta f)` the Raman gain coefficient of silica (m/W) measured with a pump
/// at :math:`f_{\mathrm{ref}}`, a function of the frequency offset alone that peaks
/// near 13 THz, and :math:`A_{\mathrm{eff}}` the effective area [1]_. The library stores
/// the product :math:`g_R A_{\mathrm{eff}}` (m/W), independent of the fibre, and divides
/// by the area of the fibre simulated. The factor :math:`f_p/f_{\mathrm{ref}}` is the
/// growth of the gain coefficient with the pump frequency ([1]_, Sec. 8.1); GNPy up to
/// version 2.4 left it out (``pump_scaling: False``).
///
/// The ``gain`` mapping gives :math:`g_R A_{\mathrm{eff}}` in one of two ways:
///
/// - a measured table, ``{"offsets_thz": [...], "gain_m_w": [...], "reference_thz":
///   f}``, linear between its points and zero beyond the last; absent, the table of
///   standard single-mode fibre of GNPy (90 points from 0 to 42 THz, a pump at 1454 nm,
///   peak :math:`3.33 \cdot 10^{-14}` m/W at 13 THz, :math:`C = 0.42` 1/(W·km) at 80 µm²);
/// - a response-function model of silica, ``{"model": name, "reference_thz": f,
///   "peak_m_w": p}``. The gain spectrum is the imaginary part of the Fourier
///   transform of the Raman response function :math:`h_R(t)` of the glass [1]_, and
///   each model gives :math:`h_R(t)` in closed form, so that the shape of the spectrum
///   follows from a few parameters; its scale, ``peak_m_w``, comes from a measurement
///   (the peak of the table above by default). ``"blow_wood"`` is one damped oscillator,
///   :math:`h_a(t) \propto e^{-t/\tau_2} \sin(t/\tau_1)`, :math:`\tau_1 = 12.2` fs,
///   :math:`\tau_2 = 32` fs [2]_: a Lorentzian with the peak of silica, but a quarter
///   of the measured gain at 3 THz. ``"lin_agrawal"`` adds the boson peak of the glass,
///   :math:`(1 - f_b) h_a(t) + f_b (2\tau_b - t) e^{-t/\tau_b} / \tau_b^2`, :math:`f_b =
///   0.21`, :math:`\tau_b = 96` fs [3]_: the shoulder below 5 THz is recovered.
///   ``"hollenbeck_cantrell"`` sums thirteen vibrational modes of the Si–O–Si structure,
///   :math:`\sum_i A_i e^{-\gamma_i t - \Gamma_i^2 t^2/4} \sin(\omega_i t)` [4]_: the whole
///   spectrum to 40 THz. Against the table of standard single-mode fibre the rms errors
///   relative to the peak are 9, 3 and 3 % over 0–10 THz and 13, 13 and 4 % over 0–42 THz.
///
/// Args:
///     offsets_hz (array_like): frequency offsets :math:`f_p - f_s` (Hz, non-negative).
///     gain (dict, optional): the ``gain`` mapping above; None: standard single-mode fibre.
///     aeff_um2 (float): effective area for the efficiency (µm², 80 by default).
///
/// Returns:
///     dict: ``gain_m_w`` (:math:`g_R A_{\mathrm{eff}}` at the offsets, m/W),
///     ``efficiency_w_km`` (:math:`g_R A_{\mathrm{eff}} / A_{\mathrm{eff}}` for ``aeff_um2``,
///     1/(W·km), at the reference pump), ``peak_hz`` and ``peak_m_w`` (the maximum of the
///     spectrum) and ``reference_hz``.
///
/// Raises:
///     ValueError: for an invalid mapping (a model with a table, an unknown model, a
///         negative gain) or a non-positive area.
///
/// References:
///     .. [1] G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013,
///        Sec. 2.3.2 and 8.1.
///     .. [2] K. J. Blow and D. Wood, "Theoretical description of transient stimulated
///        Raman scattering in optical fibers," *IEEE J. Quantum Electron.* 25(12),
///        2665–2673, 1989.
///     .. [3] Q. Lin and G. P. Agrawal, "Raman response function for silica fibers," *Opt.
///        Lett.* 31(21), 3086–3088, 2006.
///     .. [4] D. Hollenbeck and C. D. Cantrell, "Multiple-vibrational-mode model for
///        fiber-optic Raman gain spectrum and response function," *J. Opt. Soc. Am. B*
///        19(12), 2886–2892, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> g = comnum.raman_gain([13.2e12])
///     >>> round(float(g["efficiency_w_km"][0]), 2)
///     0.41
///     >>> h = comnum.raman_gain([13.2e12], {"model": "hollenbeck_cantrell", "reference_thz": 206.18})
///     >>> abs(float(h["peak_m_w"]) - float(g["peak_m_w"])) < 1e-28
///     True
#[pyfunction]
#[pyo3(signature = (offsets_hz, gain = None, aeff_um2 = 80.0))]
fn raman_gain<'py>(
    py: Python<'py>,
    offsets_hz: &Bound<'py, PyAny>,
    gain: Option<&Bound<'py, PyAny>>,
    aeff_um2: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if aeff_um2.is_nan() || aeff_um2 <= 0.0 {
        return Err(PyValueError::new_err(format!("aeff_um2 must be positive, got {aeff_um2}")));
    }
    let offsets = fvec(offsets_hz)?;
    let g = match gain {
        None => RamanGain::ssmf(),
        Some(v) if v.is_none() => RamanGain::ssmf(),
        Some(v) => {
            let j = json(py, v)?;
            BuildContext::default()
                .raman(Some(&serde_json::json!({ "raman": { "gain": j } })))
                .py()?
                .ok_or_else(|| PyValueError::new_err("gain must be a mapping"))?
                .gain
        }
    };
    let values: Vec<f64> = offsets.iter().map(|df| g.at(*df)).collect();
    let (peak_hz, peak_m_w) = g.peak();
    let out = PyDict::new(py);
    out.set_item("efficiency_w_km", farr(py, values.iter().map(|v| v / (aeff_um2 * 1e-12) * 1e3).collect()))?;
    out.set_item("gain_m_w", farr(py, values))?;
    out.set_item("peak_hz", peak_hz)?;
    out.set_item("peak_m_w", peak_m_w)?;
    out.set_item("reference_hz", g.reference_hz())?;
    Ok(out)
}

/// The closed form of the NLI power spectral density at the centre of
/// ``n_channels`` rectangular channels spaced by ``spacing`` (incoherent accumulation).
///
/// The closed form of the GN model [1]_ is
///
/// .. math::
///
///     G_{\mathrm{NLI}} \approx N_s \frac{8}{27} \gamma^2 G_{\mathrm{ch}}^3 L_{\mathrm{eff}}^2
///     \frac{\operatorname{asinh}\left(\frac{\pi^2}{2} |\beta_2| L_{\mathrm{eff},a} R_s^2
///     N_{\mathrm{ch}}^{2 R_s / \Delta f}\right)}{\pi |\beta_2| L_{\mathrm{eff},a}},
///
/// with :math:`G_{\mathrm{ch}} = P_{\mathrm{ch}}/R_s`,
/// :math:`L_{\mathrm{eff}} = (1 - e^{-\alpha L_s})/\alpha` and
/// :math:`L_{\mathrm{eff},a} = 1/\alpha`.
///
/// Args:
///     fiber (FiberParams): the link.
///     n_channels (int): number of channels.
///     symbol_rate (float): symbol rate of every channel (Bd).
///     spacing_hz (float): channel spacing (Hz).
///     power_dbm (float): launch power per channel (dBm).
///
/// Returns:
///     float: the NLI power spectral density (W/Hz).
///
/// References:
///     .. [1] P. Poggiolini, "The GN model of non-linear propagation in uncompensated
///        coherent optical systems," *J. Lightw. Technol.*, 30(24), 3857–3879, 2012.
#[pyfunction]
fn gn_closed_form(fiber: &PyFiber, n_channels: usize, symbol_rate: f64, spacing_hz: f64, power_dbm: f64) -> f64 {
    closed_form_nli_psd(&fiber.inner, n_channels, symbol_rate, spacing_hz, dbm2w(power_dbm))
}

/// The closed form of the ISRS GN model.
///
/// The closed form of [1]_ gives the NLI coefficient of a channel of a WDM signal whose
/// power profile along the span is tilted by inter-channel stimulated Raman scattering,
/// in microseconds instead of the minutes of the integral of :func:`gn_model`. The SPM of
/// the channel (eq. (10)) and the XPM of every other channel (eq. (11)) are summed over
/// the :math:`n` spans by eq. (5), the SPM coherently, the XPM incoherently:
///
/// .. math::
///
///     \eta_n(f_i) \approx \eta_{\mathrm{SPM}}(f_i)\, n^{1 + \epsilon}
///     + \eta_{\mathrm{XPM}}(f_i)\, n;
///
/// the ISRS enters to first order in the total launch power times the
/// slope ``raman_slope_w_km_thz`` of the Raman gain spectrum (0.028 /W/km/THz for
/// standard single-mode fibre; 0 for no ISRS). Channels have rectangular spectra of
/// bandwidth ``symbol_rate * (1 + rolloff)``; ``fiber`` gives the loss, the dispersion
/// and its slope, the nonlinear coefficient and the number of spans (the span length
/// does not enter: :math:`\exp(-\alpha L) \ll 1`).
///
/// The paper's eq. (10) ends with a term that is not dimensionless; it is left out
/// (see ``comnum_channel::optical::gn::isrs_closed_form_terms``): the closed form
/// follows the dashed curves of Figs. 4a, 4b and 5 of the paper within 0.3 dB away from
/// the band edges (``validation/tests/fiber/semrau2019.rs``).
///
/// Args:
///     fiber (FiberParams): the link.
///     freqs (array_like): centre frequencies of the channels relative to the carrier
///         (Hz).
///     power_dbm (float | array_like): launch power of every channel (dBm, both
///         polarisations), one for all or one per channel.
///     symbol_rate (float): symbol rate of every channel (Bd).
///     rolloff (float): roll-off of the channels (the bandwidth is
///         ``symbol_rate * (1 + rolloff)``).
///     channel (int, optional): the channel of interest (the middle one by default).
///     raman_slope_w_km_thz (float): the slope :math:`C_r` of the linear regression of the
///         normalised Raman gain spectrum (1/W/km/THz).
///     epsilon (float, optional): the coherence factor of the SPM; by default eq. (22) of
///         [2]_ over the bandwidth of the channel.
///
/// Returns:
///     dict: ``eta`` (1/W²: the NLI power of the channel is :math:`\eta P^3`; divide by
///     the bandwidth for the PSD per W³ of :func:`gn_model`), ``eta_spm`` and
///     ``eta_xpm`` (the two terms, one span), ``epsilon`` and ``snr_db`` (with the ASE of
///     the EDFAs of ``fiber`` in the bandwidth of the channel).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] D. Semrau, R. I. Killey and P. Bayvel, "A closed-form approximation of the
///        Gaussian noise model in the presence of inter-channel stimulated Raman
///        scattering," *J. Lightw. Technol.*, 37(9), 1924–1936, 2019.
///     .. [2] P. Poggiolini, "The GN model of non-linear propagation in uncompensated
///        coherent optical systems," *J. Lightw. Technol.*, 30(24), 3857–3879, 2012.
///
/// Examples:
///     >>> import comnum
///     >>> fiber = comnum.FiberParams(span_km=100, n_spans=6, gamma_w_km=1.2, slope_ps_nm2_km=0.067)
///     >>> r = comnum.gn_isrs_closed_form(fiber, comnum.wdm_grid(5, 40.005e9), 0.0, 40e9, rolloff=1e-4)
///     >>> bool(0.1 < r["epsilon"] < 0.2 and r["eta_xpm"] > r["eta_spm"] > 0)
///     True
#[pyfunction]
#[pyo3(signature = (fiber, freqs, power_dbm, symbol_rate, *, rolloff = 0.1, channel = None, raman_slope_w_km_thz = 0.028, epsilon = None))]
#[allow(clippy::too_many_arguments, reason = "the link, the grid and the options of the model, by keyword")]
fn gn_isrs_closed_form<'py>(
    py: Python<'py>,
    fiber: &PyFiber,
    freqs: &Bound<'py, PyAny>,
    power_dbm: &Bound<'py, PyAny>,
    symbol_rate: f64,
    rolloff: f64,
    channel: Option<usize>,
    raman_slope_w_km_thz: f64,
    epsilon: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let freqs = fvec(freqs)?;
    let n = freqs.len();
    let powers = powers_dbm(power_dbm, n)?;
    let ch = channel.unwrap_or(n / 2);
    let spec =
        WdmSpectrum { freqs_hz: freqs, powers_w: powers.iter().map(|p| dbm2w(*p)).collect(), symbol_rate, rolloff };
    let b = symbol_rate * (1.0 + rolloff);
    let eps = epsilon.unwrap_or_else(|| coherence_factor(&fiber.inner, b));
    let (spm, xpm) = isrs_closed_form_terms(&fiber.inner, &spec, ch, raman_slope_w_km_thz).py()?;
    let eta = isrs_closed_form_eta(&fiber.inner, &spec, ch, raman_slope_w_km_thz, eps).py()?;
    let p = spec.powers_w[ch];
    let ase = GnModel::new(fiber.inner.clone()).ase_psd() * b;
    let d = PyDict::new(py);
    d.set_item("eta", eta)?;
    d.set_item("eta_spm", spm)?;
    d.set_item("eta_xpm", xpm)?;
    d.set_item("epsilon", eps)?;
    d.set_item("snr_db", comnum::math::lin2db(p / (ase + eta * p * p * p)))?;
    Ok(d)
}

/// The centre frequencies of ``n`` WDM channels, relative to the carrier: the grid of
/// OptiCommPy's ``simpleWDMTx``.
///
/// Args:
///     n (int): number of channels.
///     spacing_hz (float): channel spacing (Hz).
///
/// Returns:
///     numpy.ndarray: the ``n`` frequencies (Hz, float64).
///
/// Raises:
///     ValueError: for a negative count.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.wdm_grid(3, 50e9).tolist()
///     [-50000000000.0, 0.0, 50000000000.0]
#[pyfunction]
fn wdm_grid(py: Python<'_>, n: i64, spacing_hz: f64) -> PyResult<FArr<'_>> {
    Ok(farr(py, comnum::dsp::signal::wdm_grid(count(n, "n")?, spacing_hz)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(gn_model, m)?,
        wrap_pyfunction!(gn_closed_form, m)?,
        wrap_pyfunction!(gn_isrs_closed_form, m)?,
        wrap_pyfunction!(raman_design, m)?,
        wrap_pyfunction!(raman_gain, m)?,
        wrap_pyfunction!(raman_profile, m)?,
        wrap_pyfunction!(format_moments, m)?,
        wrap_pyfunction!(wdm_grid, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
