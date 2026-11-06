//! The GN and EGN models of the nonlinear interference, the closed form of Poggiolini,
//! the closed form of the ISRS GN model, the moments of a format, the Raman profiles of a
//! span and the WDM grid.

use crate::fiber::PyFiber;
use crate::{FArr, IntoPy, count, farr, fvec, json};
use comnum::channel::optical::gn::{
    FormatMoments, GnModel, WdmSpectrum, closed_form_nli_psd, coherence_factor, isrs_closed_form_eta,
    isrs_closed_form_terms,
};
use comnum::math::dbm2w;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The GN model of the nonlinear interference of a WDM link (Poggiolini 2012).
///
/// The NLI power spectral density at the receiver is the double integral of the GN
/// reference formula, computed on a grid, with the coherent accumulation of the spans
/// (``coherent=True``, the phased-array factor) or the incoherent one (``N_s`` times a
/// span). Channels have raised-cosine spectra of roll-off ``rolloff`` at the symbol rate
/// ``symbol_rate``, centred on ``freqs`` (Hz, relative to the carrier), with the launch
/// powers ``power_dbm`` (one per channel or one for all, both polarisations).
///
/// Returns:
///     dict: ``nli_psd`` and ``ase_psd`` (W/Hz, both polarisations, at the frequency of
///     channel ``channel``), ``snr_db`` (after a matched filter), ``eta`` (NLI PSD per
///     W³ of launch power per channel) and ``p_opt_dbm`` (the optimal launch power per
///     channel for equal powers; with ``raman``, ``eta`` and ``p_opt_dbm`` are those of
///     the profiles of 1 mW per channel).
///
/// With ``raman`` (the keys of ``fiber.raman``, see :func:`raman_profile`), the power
/// profiles of the channels along the span replace the exponential decay in the link
/// function (the generalised GN model of Semrau et al. 2019), the noise is the
/// spontaneous Raman noise plus the ASE of the EDFA that restores every channel, and
/// the channels are restored to their launch power after every span.
///
/// With ``modulation`` (a constellation, a name such as ``"16-QAM"``, or
/// ``"gaussian"``) and optionally its probabilities ``px``, the NLI is that of the EGN
/// model (Carena et al. 2014): the GN model corrected for the fourth and sixth moments
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
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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

/// The moments of a modulation format that the EGN model needs: ``Φ = E|a|⁴/E²|a|² − 2``
/// and ``Ψ = E|a|⁶/E³|a|² − 9 E|a|⁴/E²|a|² + 12`` (Carena et al. 2014), both zero for
/// Gaussian symbols.
///
/// Args:
///     modulation (Constellation | str): a constellation, a name such as ``"16-QAM"``,
///         or ``"gaussian"``.
///     px (array_like, optional): the probabilities of the points (a shaped source).
///
/// Returns:
///     tuple[float, float]: ``(Φ, Ψ)``.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
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

/// The power profiles of a span with distributed Raman amplification: the coupled
/// equations of stimulated Raman scattering for the co-propagating channels and any
/// number of co- and counter-propagating pumps (RK4, forward-backward sweeps), and the
/// spontaneous Raman noise of every channel (see ``comnum_channel::optical::raman``).
///
/// Args:
///     fiber (FiberParams): the span (length, attenuation, carrier frequency).
///     freqs (array_like): centre frequencies of the channels relative to the carrier
///         (Hz).
///     power_dbm (float | array_like): launch power of every channel (dBm, both
///         polarisations).
///     raman (dict): the keys of ``fiber.raman`` in a specification: ``pumps`` (a list of
///         ``{wavelength_nm | frequency_thz, power_mw, direction: "co" | "counter",
///         alpha_db_km}``), ``aeff_um2`` (80), ``temperature_k`` (300), ``isrs`` (True:
///         the channels also exchange power), ``step_km`` (0.1), ``slice_hz`` (25e9, for
///         the split-step solvers), ``gain`` (``{offsets_thz, gain_m_w,
///         reference_thz}``, the gain spectrum ``g_R A_eff``; standard single-mode fibre
///         by default) and ``rayleigh_db_km`` (the Rayleigh backscatter coupling of the
///         signals, 10 log10 of its value in 1/km, about -40 for transmission fibre;
///         absent: no double Rayleigh backscattering).
///
/// Returns:
///     dict: ``z_km``, ``freqs_hz`` (absolute, the channels then the pumps),
///     ``power_dbm`` (one row per wave), ``net_gain_db`` and ``on_off_gain_db`` (per
///     channel), ``ase_psd`` (W/Hz, both polarisations, per channel at the end of the
///     span), ``double_rayleigh`` (the effective Rayleigh reflection coefficient of every
///     channel, the doubly backscattered power over the signal power at the end of the
///     span, linear; 0 without ``rayleigh_db_km``), ``noise_figure_db`` (of the span,
///     Bromage 2004 eq. (9)), ``effective_noise_figure_db`` (the noise figure minus the
///     loss of the span, eq. (12)) and ``iterations``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
    d.set_item("double_rayleigh", farr(py, p.double_rayleigh.clone()))?;
    d.set_item("noise_figure_db", farr(py, (0..n).map(|k| p.noise_figure_db(k)).collect()))?;
    d.set_item("effective_noise_figure_db", farr(py, (0..n).map(|k| p.effective_noise_figure_db(k)).collect()))?;
    d.set_item("iterations", p.iterations)?;
    Ok(d)
}

/// Poggiolini's closed form of the NLI power spectral density at the centre of
/// ``n_channels`` rectangular channels spaced by ``spacing`` (incoherent accumulation).
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
#[pyfunction]
fn gn_closed_form(fiber: &PyFiber, n_channels: usize, symbol_rate: f64, spacing_hz: f64, power_dbm: f64) -> f64 {
    closed_form_nli_psd(&fiber.inner, n_channels, symbol_rate, spacing_hz, dbm2w(power_dbm))
}

/// The closed form of the ISRS GN model of Semrau, Killey & Bayvel (2019). It gives the
/// NLI coefficient (*J. Lightw. Technol.* 37(9), 1924–1936) of a channel of a WDM signal
/// whose power profile along the span is tilted by inter-channel stimulated Raman
/// scattering, in microseconds instead of the minutes of the integral of
/// :func:`gn_model`. The SPM of
/// the channel (eq. (10)) and the XPM of every other channel (eq. (11)) are summed over
/// the spans by eq. (5), the SPM coherently as ``n ** (1 + epsilon)``, the XPM
/// incoherently; the ISRS enters to first order in the total launch power times the
/// slope ``raman_slope_w_km_thz`` of the Raman gain spectrum (0.028 /W/km/THz for
/// standard single-mode fibre; 0 for no ISRS). Channels have rectangular spectra of
/// bandwidth ``symbol_rate * (1 + rolloff)``; ``fiber`` gives the loss, the dispersion
/// and its slope, the nonlinear coefficient and the number of spans (the span length
/// does not enter: ``exp(-alpha L) << 1``).
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
///     raman_slope_w_km_thz (float): the slope ``C_r`` of the linear regression of the
///         normalised Raman gain spectrum (1/W/km/THz).
///     epsilon (float, optional): the coherence factor of the SPM; by default Poggiolini's
///         eq. (22) (*J. Lightw. Technol.* 30(24), 2012) over the bandwidth of the channel.
///
/// Returns:
///     dict: ``eta`` (1/W²: the NLI power of the channel is ``eta * P**3``; divide by
///     the bandwidth for the PSD per W³ of :func:`gn_model`), ``eta_spm`` and
///     ``eta_xpm`` (the two terms, one span), ``epsilon`` and ``snr_db`` (with the ASE of
///     the EDFAs of ``fiber`` in the bandwidth of the channel).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
        wrap_pyfunction!(raman_profile, m)?,
        wrap_pyfunction!(format_moments, m)?,
        wrap_pyfunction!(wdm_grid, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
