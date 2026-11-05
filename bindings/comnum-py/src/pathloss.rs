//! The path loss, line-of-sight probability, shadow fading and penetration of 3GPP TR
//! 38.901 §7.4.

use crate::IntoPy;
use crate::rng::PyRng;
use comnum::channel::pathloss::{
    self as pl, InfClutter, InfSubScenario, LinkGeometry, O2iModel, OfficeLayout, Scenario,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The scenario of a name, with its default BS height.
#[allow(clippy::too_many_arguments, reason = "the parameters of the scenarios, all optional")]
fn scenario(
    name: &str,
    building_height_m: f64,
    street_width_m: f64,
    environment_height_m: f64,
    office: &str,
    clutter_size_m: f64,
    clutter_density: f64,
    clutter_height_m: f64,
) -> PyResult<(Scenario, f64)> {
    let layout = match office {
        "mixed" => OfficeLayout::Mixed,
        "open" => OfficeLayout::Open,
        _ => return Err(PyValueError::new_err(format!("office must be mixed or open, got {office:?}"))),
    };
    let inf = |sub| InfClutter { sub, size_m: clutter_size_m, density: clutter_density, height_m: clutter_height_m };
    Ok(match name.to_ascii_lowercase().as_str() {
        "rma" => (Scenario::RMa { building_height_m, street_width_m }, 35.0),
        "uma" => (Scenario::UMa { environment_height_m }, 25.0),
        "umi" => (Scenario::UMi, 10.0),
        "inh" => (Scenario::InhOffice(layout), 3.0),
        "inf-sl" => (Scenario::InF(inf(InfSubScenario::SparseLow)), 1.5),
        "inf-dl" => (Scenario::InF(inf(InfSubScenario::DenseLow)), 1.5),
        "inf-sh" => (Scenario::InF(inf(InfSubScenario::SparseHigh)), 8.0),
        "inf-dh" => (Scenario::InF(inf(InfSubScenario::DenseHigh)), 8.0),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scenario {other:?}: rma, uma, umi, inh, inf-sl, inf-dl, inf-sh, inf-dh"
            )));
        }
    })
}

/// The path loss of a link of 3GPP TR 38.901 (Table 7.4.1-1).
///
/// Args:
///     scenario (str): ``"rma"``, ``"uma"``, ``"umi"`` (street canyon), ``"inh"`` (office),
///         ``"inf-sl"``, ``"inf-dl"``, ``"inf-sh"``, ``"inf-dh"`` (indoor factory).
///     d2d_m (float): the horizontal BS-UT distance (m).
///     fc_hz (float): the carrier frequency (Hz), 0.5 to 100 GHz (30 GHz for RMa).
///     los (bool): with a line of sight.
///     h_bs_m (float | None): the BS height (m); by default 35 (RMa), 25 (UMa), 10 (UMi),
///         3 (InH), 1.5 (InF low BS), 8 (InF high BS).
///     h_ut_m (float): the UT height (m).
///     building_height_m (float): the average building height of RMa (m).
///     street_width_m (float): the average street width of RMa (m).
///     environment_height_m (float): the effective environment height of UMa (m),
///         ``uma_environment_height`` draws it.
///
/// Returns:
///     float: the path loss (dB), without shadow fading.
///
/// Raises:
///     ValueError: for an unknown scenario, a carrier or a distance outside the range of
///         the scenario, or a non-positive height.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.path_loss_db("uma", 200.0, 3.5e9), 2)
///     89.57
#[pyfunction]
#[pyo3(signature = (scenario, d2d_m, fc_hz, *, los=true, h_bs_m=None, h_ut_m=1.5, building_height_m=5.0, street_width_m=20.0, environment_height_m=1.0))]
#[allow(clippy::too_many_arguments, reason = "the parameters of TR 38.901, all but three optional")]
fn path_loss_db(
    scenario: &str,
    d2d_m: f64,
    fc_hz: f64,
    los: bool,
    h_bs_m: Option<f64>,
    h_ut_m: f64,
    building_height_m: f64,
    street_width_m: f64,
    environment_height_m: f64,
) -> PyResult<f64> {
    let (s, hb) =
        self::scenario(scenario, building_height_m, street_width_m, environment_height_m, "mixed", 10.0, 0.2, 2.0)?;
    pl::path_loss_db(&s, &LinkGeometry { d2d_m, h_bs_m: h_bs_m.unwrap_or(hb), h_ut_m }, fc_hz, los).py()
}

/// The standard deviation of the log-normal shadow fading of a link (Table 7.4.1-1).
///
/// Args:
///     scenario (str): as ``path_loss_db``.
///     d2d_m (float): the horizontal BS-UT distance (m), which selects the RMa LOS value.
///     fc_hz (float): the carrier frequency (Hz).
///     los (bool): with a line of sight.
///     h_bs_m (float | None): the BS height (m), the default of the scenario if None.
///     h_ut_m (float): the UT height (m).
///
/// Returns:
///     float: the standard deviation (dB).
///
/// Raises:
///     ValueError: for an unknown scenario.
#[pyfunction]
#[pyo3(signature = (scenario, d2d_m, fc_hz, *, los=true, h_bs_m=None, h_ut_m=1.5))]
fn shadow_fading_std_db(
    scenario: &str,
    d2d_m: f64,
    fc_hz: f64,
    los: bool,
    h_bs_m: Option<f64>,
    h_ut_m: f64,
) -> PyResult<f64> {
    let (s, hb) = self::scenario(scenario, 5.0, 20.0, 1.0, "mixed", 10.0, 0.2, 2.0)?;
    Ok(pl::shadow_fading_std_db(&s, &LinkGeometry { d2d_m, h_bs_m: h_bs_m.unwrap_or(hb), h_ut_m }, fc_hz, los))
}

/// The probability of a line of sight (TR 38.901 Table 7.4.2-1).
///
/// Args:
///     scenario (str): as ``path_loss_db``.
///     d2d_out_m (float): the outdoor horizontal distance (m); the indoor distance for
///         ``"inh"``.
///     h_ut_m (float): the UT height (m).
///     h_bs_m (float | None): the BS height (m) of the InF scenarios with a high BS.
///     office (str): ``"mixed"`` or ``"open"``, for ``"inh"``.
///     clutter_size_m (float): the clutter size of InF (m), Table 7.2-4.
///     clutter_density (float): the clutter density of InF (0 to 1).
///     clutter_height_m (float): the clutter height of InF (m).
///
/// Returns:
///     float: the probability.
///
/// Raises:
///     ValueError: for an unknown scenario or office.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.los_probability("umi", 10.0)
///     1.0
#[pyfunction]
#[pyo3(signature = (scenario, d2d_out_m, *, h_ut_m=1.5, h_bs_m=None, office="mixed", clutter_size_m=10.0, clutter_density=0.2, clutter_height_m=2.0))]
#[allow(clippy::too_many_arguments, reason = "the parameters of TR 38.901, all but two optional")]
fn los_probability(
    scenario: &str,
    d2d_out_m: f64,
    h_ut_m: f64,
    h_bs_m: Option<f64>,
    office: &str,
    clutter_size_m: f64,
    clutter_density: f64,
    clutter_height_m: f64,
) -> PyResult<f64> {
    let (s, hb) = self::scenario(scenario, 5.0, 20.0, 1.0, office, clutter_size_m, clutter_density, clutter_height_m)?;
    Ok(pl::los_probability(&s, d2d_out_m, h_ut_m, h_bs_m.unwrap_or(hb)))
}

/// The effective environment height of a UMa link (TR 38.901 Table 7.4.1-1, NOTE 1).
///
/// Args:
///     d2d_m (float): the horizontal BS-UT distance (m).
///     h_ut_m (float): the UT height (m).
///     rng (Rng): the generator.
///
/// Returns:
///     float: 1 m, or one of 12, 15, ..., h_UT - 1.5 m.
#[pyfunction]
fn uma_environment_height(d2d_m: f64, h_ut_m: f64, mut rng: PyRefMut<'_, PyRng>) -> f64 {
    pl::uma_environment_height(d2d_m, h_ut_m, &mut rng.inner)
}

/// The outdoor-to-indoor penetration of a building (TR 38.901 eq. (7.4-2), Table 7.4.3-2).
///
/// Args:
///     model (str): ``"low"``, ``"high"`` or ``"low-a"``.
///     fc_hz (float): the carrier frequency (Hz).
///     d2d_in_m (float): the indoor horizontal distance (m).
///
/// Returns:
///     tuple[float, float]: the mean loss ``PL_tw + PL_in`` and the standard deviation
///     ``sigma_P`` (dB).
///
/// Raises:
///     ValueError: for an unknown model.
#[pyfunction]
fn building_penetration_db(model: &str, fc_hz: f64, d2d_in_m: f64) -> PyResult<(f64, f64)> {
    let m = match model {
        "low" => O2iModel::LowLoss,
        "high" => O2iModel::HighLoss,
        "low-a" => O2iModel::LowLossA,
        _ => return Err(PyValueError::new_err(format!("model must be low, high or low-a, got {model:?}"))),
    };
    Ok(pl::building_penetration_db(m, fc_hz, d2d_in_m))
}

/// Register the path-loss bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(path_loss_db, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_fading_std_db, m)?)?;
    m.add_function(wrap_pyfunction!(los_probability, m)?)?;
    m.add_function(wrap_pyfunction!(uma_environment_height, m)?)?;
    m.add_function(wrap_pyfunction!(building_penetration_db, m)?)?;
    Ok(())
}
