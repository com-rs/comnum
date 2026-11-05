//! Path loss, line-of-sight probability, shadow fading and outdoor-to-indoor penetration
//! of 3GPP TR 38.901 §7.4 (Tables 7.4.1-1, 7.4.2-1, 7.4.3-1 and 7.4.3-2).
//!
//! # Model
//!
//! A link between a base station (BS) of height $h_{BS}$ and a terminal (UT) of height
//! $h_{UT}$, at the horizontal distance $d_{2D}$, has the 3-D distance
//! $d_{3D} = \sqrt{d_{2D}^{2} + (h_{BS} - h_{UT})^{2}}$ (Fig. 7.4.1-1). Its path loss
//! (dB), $f_{c}$ in GHz and distances in metres (Table 7.4.1-1, NOTE 6), is
//!
//! - RMa, LOS: $PL_{1} = 20\log_{10}(40\pi d_{3D} f_{c}/3) + \min(0.03 h^{1.72}, 10)
//!   \log_{10}(d_{3D}) - \min(0.044 h^{1.72}, 14.77) + 0.002\log_{10}(h)\\, d_{3D}$ up to the
//!   breakpoint $d_{BP} = 2\pi h_{BS} h_{UT} f_{c}/c$, then $PL_{2} = PL_{1}(d_{BP}) +
//!   40\log_{10}(d_{3D}/d_{BP})$; NLOS: $\max(PL_{\mathrm{LOS}}, PL')$ with
//!   $PL' = 161.04 - 7.1\log_{10} W + 7.5\log_{10} h - (24.37 - 3.7(h/h_{BS})^{2})\log_{10} h_{BS} +
//!   (43.42 - 3.1\log_{10} h_{BS})(\log_{10} d_{3D} - 3) + 20\log_{10} f_{c} - (3.2
//!   (\log_{10}(11.75 h_{UT}))^{2} - 4.97)$, $h$ the average building height and $W$ the
//!   street width;
//! - UMa, LOS: $PL_{1} = 28 + 22\log_{10} d_{3D} + 20\log_{10} f_{c}$ up to the effective
//!   breakpoint $d'_{BP} = 4 h'_{BS} h'_{UT} f_{c}/c$, $h' = h - h_{E}$, then
//!   $PL_{2} = 28 + 40\log_{10} d_{3D} + 20\log_{10} f_{c} - 9\log_{10}((d'_{BP})^{2} +
//!   (h_{BS} - h_{UT})^{2})$; NLOS: $\max(PL_{\mathrm{LOS}}, 13.54 + 39.08\log_{10} d_{3D} +
//!   20\log_{10} f_{c} - 0.6(h_{UT} - 1.5))$;
//! - UMi-Street canyon, LOS: $32.4 + 21\log_{10} d_{3D} + 20\log_{10} f_{c}$, then
//!   $32.4 + 40\log_{10} d_{3D} + 20\log_{10} f_{c} - 9.5\log_{10}((d'_{BP})^{2} +
//!   (h_{BS} - h_{UT})^{2})$ ($h_{E} = 1$ m); NLOS: $\max(PL_{\mathrm{LOS}}, 35.3\log_{10}
//!   d_{3D} + 22.4 + 21.3\log_{10} f_{c} - 0.3(h_{UT} - 1.5))$;
//! - InH-Office, LOS: $32.4 + 17.3\log_{10} d_{3D} + 20\log_{10} f_{c}$; NLOS:
//!   $\max(PL_{\mathrm{LOS}}, 38.3\log_{10} d_{3D} + 17.30 + 24.9\log_{10} f_{c})$;
//! - InF, LOS: $31.84 + 21.50\log_{10} d_{3D} + 19.00\log_{10} f_{c}$; NLOS by sub-scenario
//!   ([`InfClutter`]), each bounded below by the LOS loss (and InF-DL by InF-SL).
//!
//! $c = 3.0 \times 10^{8}$ m/s in the breakpoints, as NOTE 1 and NOTE 5 prescribe
//! ([`SPEED_OF_LIGHT_TR`]). The effective environment height $h_{E}$ of UMa is random
//! per link ([`uma_environment_height`], NOTE 1); [`Scenario::UMa`] holds the drawn value.
//! The shadow fading is log-normal of the standard deviation [`shadow_fading_std_db`].
//! The optional NLOS formulas and the SMa scenario (truncated in the converted copy of the
//! TR) are not implemented.
//!
//! The probability of a line of sight (Table 7.4.2-1) is [`los_probability`]; the
//! penetration of a building (Tables 7.4.3-1, 7.4.3-2, eq. (7.4-2)) is
//! [`building_penetration_db`].
//!
//! # Examples
//! ```
//! use comnum_channel::pathloss::{path_loss_db, LinkGeometry, Scenario};
//! // UMi at 3.5 GHz, 100 m, below the breakpoint: 32.4 + 21 log10(d3D) + 20 log10(3.5)
//! let link = LinkGeometry { d2d_m: 100.0, h_bs_m: 10.0, h_ut_m: 1.5 };
//! let pl = path_loss_db(&Scenario::UMi, &link, 3.5e9, true)?;
//! let d3d = (100.0f64.powi(2) + 8.5f64.powi(2)).sqrt();
//! assert!((pl - (32.4 + 21.0 * d3d.log10() + 20.0 * 3.5f64.log10())).abs() < 1e-12);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TR 38.901 V19.5.0, *Study on channel model for frequencies from 0.5 to 100
//!    GHz*, §7.4 (Tables 7.4.1-1, 7.4.2-1, 7.4.3-1, 7.4.3-2; eqs. (7.4-1), (7.4-2)).

use comnum_core::ensure;
use comnum_core::prelude::*;
use std::f64::consts::PI;

/// The speed of light $c = 3.0 \times 10^{8}$ m/s of the breakpoint distances of TR 38.901
/// (Table 7.4.1-1, NOTE 1 and NOTE 5); `comnum_math::SPEED_OF_LIGHT` is the exact value.
pub const SPEED_OF_LIGHT_TR: f64 = 3.0e8;

/// The layout of an indoor office, for its line-of-sight probability (Table 7.4.2-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OfficeLayout {
    /// Mixed office.
    Mixed,
    /// Open office.
    Open,
}

/// The sub-scenario of an indoor factory (TR 38.901 Table 7.2-4): sparse or dense
/// clutter, the BS low or high (above the clutter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfSubScenario {
    /// Sparse clutter, low BS.
    SparseLow,
    /// Dense clutter, low BS.
    DenseLow,
    /// Sparse clutter, high BS.
    SparseHigh,
    /// Dense clutter, high BS.
    DenseHigh,
}

/// The clutter of an indoor factory: its sub-scenario and the parameters of its line-of-sight
/// probability (Table 7.2-4: typical clutter size 10 m sparse and 2 m dense, density below
/// 40 % low and at least 40 % high, height 0 to 10 m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InfClutter {
    /// The sub-scenario.
    pub sub: InfSubScenario,
    /// Clutter size $d_{\mathrm{clutter}}$ (m).
    pub size_m: f64,
    /// Clutter density $r$ (0 to 1).
    pub density: f64,
    /// Clutter height $h_{c}$ (m).
    pub height_m: f64,
}

/// A scenario of TR 38.901 §7.4.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scenario {
    /// Rural macro: average building height $h$ and street width $W$ (defaults 5 m and
    /// 20 m, applicability 5 to 50 m).
    RMa {
        /// Average building height $h$ (m).
        building_height_m: f64,
        /// Average street width $W$ (m).
        street_width_m: f64,
    },
    /// Urban macro, with the effective environment height $h_{E}$ of the link (1 m, or
    /// drawn by [`uma_environment_height`]).
    UMa {
        /// Effective environment height $h_{E}$ (m).
        environment_height_m: f64,
    },
    /// Urban micro, street canyon ($h_{E} = 1$ m).
    UMi,
    /// Indoor hotspot, office.
    InhOffice(OfficeLayout),
    /// Indoor factory.
    InF(InfClutter),
}

impl Scenario {
    /// RMa with the default building height (5 m) and street width (20 m).
    pub const RMA: Scenario = Scenario::RMa { building_height_m: 5.0, street_width_m: 20.0 };
    /// UMa with $h_{E} = 1$ m (every link of a UT below 13 m, NOTE 1).
    pub const UMA: Scenario = Scenario::UMa { environment_height_m: 1.0 };
}

/// The geometry of a link: horizontal distance and antenna heights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkGeometry {
    /// Horizontal BS-UT distance $d_{2D}$ (m).
    pub d2d_m: f64,
    /// BS antenna height $h_{BS}$ (m).
    pub h_bs_m: f64,
    /// UT antenna height $h_{UT}$ (m).
    pub h_ut_m: f64,
}

impl LinkGeometry {
    /// The 3-D distance $d_{3D} = \sqrt{d_{2D}^{2} + (h_{BS} - h_{UT})^{2}}$ (m).
    pub fn d3d_m(&self) -> f64 {
        self.d2d_m.hypot(self.h_bs_m - self.h_ut_m)
    }

    fn check(&self) -> Result<()> {
        let LinkGeometry { d2d_m, h_bs_m, h_ut_m } = *self;
        ensure(d2d_m.is_finite() && d2d_m >= 0.0, || format!("the distance must be non-negative, got {d2d_m} m"))?;
        ensure(h_bs_m > 0.0 && h_ut_m > 0.0 && h_bs_m.is_finite() && h_ut_m.is_finite(), || {
            format!("the antenna heights must be positive, got {h_bs_m} m and {h_ut_m} m")
        })
    }
}

/// The breakpoint distance (m): $d_{BP} = 2\pi h_{BS} h_{UT} f_{c}/c$ of RMa (NOTE 5),
/// $d'_{BP} = 4 (h_{BS} - h_{E})(h_{UT} - h_{E}) f_{c}/c$ of UMa and UMi (NOTE 1); `None`
/// for the indoor scenarios, which have none.
pub fn breakpoint_m(scenario: &Scenario, link: &LinkGeometry, fc_hz: f64) -> Option<f64> {
    let (hb, hu) = (link.h_bs_m, link.h_ut_m);
    match *scenario {
        Scenario::RMa { .. } => Some(2.0 * PI * hb * hu * fc_hz / SPEED_OF_LIGHT_TR),
        Scenario::UMa { environment_height_m: he } => Some(4.0 * (hb - he) * (hu - he) * fc_hz / SPEED_OF_LIGHT_TR),
        Scenario::UMi => Some(4.0 * (hb - 1.0) * (hu - 1.0) * fc_hz / SPEED_OF_LIGHT_TR),
        Scenario::InhOffice(_) | Scenario::InF(_) => None,
    }
}

/// The path loss $PL$ (dB) of the link in the scenario at the carrier `fc_hz` (Hz), with
/// a line of sight (`los`) or without (Table 7.4.1-1; see the module).
///
/// # Errors
/// [`Error::Config`] for a carrier outside 0.5 to 100 GHz (30 GHz for RMa), a distance
/// outside the range of the scenario (outdoor: $d_{2D}$ from 10 m to 10 km for RMa LOS,
/// 5 km otherwise; indoor: $d_{3D}$ from 1 m to 150 m for InH, 600 m for InF), a
/// non-positive height, or a breakpoint of UMa or UMi that is not positive ($h_{UT}$ or
/// $h_{BS}$ not above $h_{E}$).
pub fn path_loss_db(scenario: &Scenario, link: &LinkGeometry, fc_hz: f64, los: bool) -> Result<f64> {
    link.check()?;
    let max_ghz = if matches!(scenario, Scenario::RMa { .. }) { 30.0 } else { 100.0 };
    let f = fc_hz / 1e9;
    ensure((0.5..=max_ghz).contains(&f), || format!("the carrier must lie in 0.5 to {max_ghz} GHz, got {f} GHz"))?;
    let (d2, d3, hb, hu) = (link.d2d_m, link.d3d_m(), link.h_bs_m, link.h_ut_m);
    let lg = f64::log10;
    let outdoor = |max: f64| {
        ensure((10.0..=max).contains(&d2), || format!("d2D must lie in 10 m to {max} m for this scenario, got {d2} m"))
    };
    let indoor = |max: f64| ensure((1.0..=max).contains(&d3), || format!("d3D must lie in 1 m to {max} m, got {d3} m"));
    Ok(match *scenario {
        Scenario::RMa { building_height_m: h, street_width_m: w } => {
            outdoor(if los { 10_000.0 } else { 5_000.0 })?;
            let dbp = 2.0 * PI * hb * hu * fc_hz / SPEED_OF_LIGHT_TR;
            let pl1 = |d: f64| {
                20.0 * lg(40.0 * PI * d * f / 3.0) + (0.03 * h.powf(1.72)).min(10.0) * lg(d)
                    - (0.044 * h.powf(1.72)).min(14.77)
                    + 0.002 * lg(h) * d
            };
            let pl_los = if d2 <= dbp { pl1(d3) } else { pl1(dbp) + 40.0 * lg(d3 / dbp) };
            if los {
                pl_los
            } else {
                let nlos = 161.04 - 7.1 * lg(w) + 7.5 * lg(h) - (24.37 - 3.7 * (h / hb).powi(2)) * lg(hb)
                    + (43.42 - 3.1 * lg(hb)) * (lg(d3) - 3.0)
                    + 20.0 * lg(f)
                    - (3.2 * lg(11.75 * hu).powi(2) - 4.97);
                pl_los.max(nlos)
            }
        }
        Scenario::UMa { .. } | Scenario::UMi => {
            outdoor(5_000.0)?;
            let umi = matches!(scenario, Scenario::UMi);
            let dbp = breakpoint_m(scenario, link, fc_hz).unwrap_or_default();
            ensure(dbp > 0.0, || format!("the breakpoint must be positive, got {dbp} m (heights above h_E)"))?;
            let (a, b1, b2) = if umi { (32.4, 21.0, 9.5) } else { (28.0, 22.0, 9.0) };
            let pl_los = if d2 <= dbp {
                a + b1 * lg(d3) + 20.0 * lg(f)
            } else {
                a + 40.0 * lg(d3) + 20.0 * lg(f) - b2 * lg(dbp * dbp + (hb - hu).powi(2))
            };
            if los {
                pl_los
            } else if umi {
                pl_los.max(35.3 * lg(d3) + 22.4 + 21.3 * lg(f) - 0.3 * (hu - 1.5))
            } else {
                pl_los.max(13.54 + 39.08 * lg(d3) + 20.0 * lg(f) - 0.6 * (hu - 1.5))
            }
        }
        Scenario::InhOffice(_) => {
            indoor(150.0)?;
            let pl_los = 32.4 + 17.3 * lg(d3) + 20.0 * lg(f);
            if los { pl_los } else { pl_los.max(38.3 * lg(d3) + 17.30 + 24.9 * lg(f)) }
        }
        Scenario::InF(c) => {
            indoor(600.0)?;
            let pl_los = 31.84 + 21.50 * lg(d3) + 19.00 * lg(f);
            let sl = 33.0 + 25.5 * lg(d3) + 20.0 * lg(f);
            if los {
                pl_los
            } else {
                match c.sub {
                    InfSubScenario::SparseLow => sl.max(pl_los),
                    InfSubScenario::DenseLow => (18.6 + 35.7 * lg(d3) + 20.0 * lg(f)).max(pl_los).max(sl),
                    InfSubScenario::SparseHigh => (32.4 + 23.0 * lg(d3) + 20.0 * lg(f)).max(pl_los),
                    InfSubScenario::DenseHigh => (33.63 + 21.9 * lg(d3) + 20.0 * lg(f)).max(pl_los),
                }
            }
        }
    })
}

/// The standard deviation $\sigma_{SF}$ (dB) of the log-normal shadow fading of the link
/// (Table 7.4.1-1): RMa LOS 4 dB below the breakpoint and 6 dB beyond, RMa NLOS 8 dB; UMa
/// 4 and 6 dB; UMi 4 and 7.82 dB; InH 3 and 8.03 dB; InF LOS 4.3 dB, NLOS 5.7 (SL), 7.2
/// (DL), 5.9 (SH), 4.0 dB (DH).
pub fn shadow_fading_std_db(scenario: &Scenario, link: &LinkGeometry, fc_hz: f64, los: bool) -> f64 {
    match (*scenario, los) {
        (Scenario::RMa { .. }, true) => {
            if breakpoint_m(scenario, link, fc_hz).is_some_and(|d| link.d2d_m < d) {
                4.0
            } else {
                6.0
            }
        }
        (Scenario::RMa { .. }, false) => 8.0,
        (Scenario::UMa { .. } | Scenario::UMi, true) => 4.0,
        (Scenario::UMa { .. }, false) => 6.0,
        (Scenario::UMi, false) => 7.82,
        (Scenario::InhOffice(_), true) => 3.0,
        (Scenario::InhOffice(_), false) => 8.03,
        (Scenario::InF(_), true) => 4.3,
        (Scenario::InF(c), false) => match c.sub {
            InfSubScenario::SparseLow => 5.7,
            InfSubScenario::DenseLow => 7.2,
            InfSubScenario::SparseHigh => 5.9,
            InfSubScenario::DenseHigh => 4.0,
        },
    }
}

/// The effective environment height $h_{E}$ (m) of a UMa link (Table 7.4.1-1, NOTE 1):
/// 1 m with probability $1/(1 + C(d_{2D}, h_{UT}))$, otherwise uniform over
/// $\\{12, 15, \ldots, h_{UT} - 1.5\\}$, with $C = 0$ for $h_{UT} < 13$ m and
/// $C = ((h_{UT} - 13)/10)^{1.5} g(d_{2D})$, $g = \frac{5}{4}(d_{2D}/100)^{3}
/// e^{-d_{2D}/150}$ beyond 18 m (0 below).
pub fn uma_environment_height(d2d_m: f64, h_ut_m: f64, rng: &mut SimRng) -> f64 {
    let g = if d2d_m <= 18.0 { 0.0 } else { 1.25 * (d2d_m / 100.0).powi(3) * (-d2d_m / 150.0).exp() };
    let c = if h_ut_m < 13.0 { 0.0 } else { ((h_ut_m - 13.0) / 10.0).powf(1.5) * g };
    if rng.uniform() < 1.0 / (1.0 + c) {
        return 1.0;
    }
    let n = (((h_ut_m - 1.5 - 12.0) / 3.0).floor() as usize + 1).max(1);
    let k = ((rng.uniform() * n as f64) as usize).min(n - 1);
    12.0 + 3.0 * k as f64
}

/// The probability of a line of sight (Table 7.4.2-1) at the outdoor horizontal distance
/// `d2d_out_m` (m) for a UT at `h_ut_m` (m): RMa, UMi, UMa (with $C'(h_{UT})$), the mixed
/// and open office (`d2d_out_m` is then the indoor distance), and the indoor factory
/// ($e^{-d_{2D}/k}$, $k = -d_{\mathrm{clutter}}/\ln(1 - r)$, times
/// $(h_{BS} - h_{UT})/(h_{c} - h_{UT})$ for a high BS of height `h_bs_m`).
pub fn los_probability(scenario: &Scenario, d2d_out_m: f64, h_ut_m: f64, h_bs_m: f64) -> f64 {
    let d = d2d_out_m;
    match *scenario {
        Scenario::RMa { .. } => {
            if d <= 10.0 {
                1.0
            } else {
                (-(d - 10.0) / 1000.0).exp()
            }
        }
        Scenario::UMi => {
            if d <= 18.0 {
                1.0
            } else {
                18.0 / d + (-d / 36.0).exp() * (1.0 - 18.0 / d)
            }
        }
        Scenario::UMa { .. } => {
            if d <= 18.0 {
                return 1.0;
            }
            let c = if h_ut_m <= 13.0 { 0.0 } else { ((h_ut_m - 13.0) / 10.0).powf(1.5) };
            (18.0 / d + (-d / 63.0).exp() * (1.0 - 18.0 / d))
                * (1.0 + c * 1.25 * (d / 100.0).powi(3) * (-d / 150.0).exp())
        }
        Scenario::InhOffice(OfficeLayout::Mixed) => {
            if d <= 1.2 {
                1.0
            } else if d < 6.5 {
                (-(d - 1.2) / 4.7).exp()
            } else {
                (-(d - 6.5) / 32.6).exp() * 0.32
            }
        }
        Scenario::InhOffice(OfficeLayout::Open) => {
            if d <= 5.0 {
                1.0
            } else if d <= 49.0 {
                (-(d - 5.0) / 70.8).exp()
            } else {
                (-(d - 49.0) / 211.7).exp() * 0.54
            }
        }
        Scenario::InF(c) => {
            let k = -c.size_m / (1.0 - c.density).ln();
            let k = match c.sub {
                InfSubScenario::SparseLow | InfSubScenario::DenseLow => k,
                InfSubScenario::SparseHigh | InfSubScenario::DenseHigh => k * (h_bs_m - h_ut_m) / (c.height_m - h_ut_m),
            };
            (-d / k).exp()
        }
    }
}

/// A building material of TR 38.901 Table 7.4.3-1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Material {
    /// Standard multi-pane glass, $2 + 0.2 f$ dB.
    Glass,
    /// Infrared-reflecting glass, $25.4 + 0.11 f$ dB (Rel-19).
    IrrGlass,
    /// Concrete, $5 + 4 f$ dB.
    Concrete,
    /// Plywood, $1.03 + 0.17 f$ dB.
    Plywood,
    /// Wood, $4.85 + 0.12 f$ dB.
    Wood,
}

impl Material {
    /// The penetration loss $L = a + b f$ (dB) at `fc_hz` (Hz), $f$ in GHz (Table 7.4.3-1).
    pub fn loss_db(self, fc_hz: f64) -> f64 {
        let f = fc_hz / 1e9;
        let (a, b) = match self {
            Material::Glass => (2.0, 0.2),
            Material::IrrGlass => (25.4, 0.11),
            Material::Concrete => (5.0, 4.0),
            Material::Plywood => (1.03, 0.17),
            Material::Wood => (4.85, 0.12),
        };
        a + b * f
    }
}

/// An outdoor-to-indoor penetration model of TR 38.901 Table 7.4.3-2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum O2iModel {
    /// 30 % standard glass, 70 % concrete; $\sigma_{P} = 4.4$ dB.
    LowLoss,
    /// 70 % IRR glass, 30 % concrete; $\sigma_{P} = 6.5$ dB.
    HighLoss,
    /// 30 % standard glass, 70 % plywood; $\sigma_{P} = 4.4$ dB.
    LowLossA,
}

/// The outdoor-to-indoor penetration (eq. (7.4-2), Table 7.4.3-2): the mean loss
/// $PL_{tw} + PL_{in}$ (dB), $PL_{tw} = 5 - 10\log_{10}\sum_{i} p_{i} 10^{-L_{i}/10}$ and
/// $PL_{in} = 0.5\\, d_{2D\text{-}in}$, and the standard deviation $\sigma_{P}$ (dB) of its
/// Gaussian part, for the indoor distance `d2d_in_m` (m) at `fc_hz` (Hz).
pub fn building_penetration_db(model: O2iModel, fc_hz: f64, d2d_in_m: f64) -> (f64, f64) {
    let ((p1, m1), (p2, m2), sigma) = match model {
        O2iModel::LowLoss => ((0.3, Material::Glass), (0.7, Material::Concrete), 4.4),
        O2iModel::HighLoss => ((0.7, Material::IrrGlass), (0.3, Material::Concrete), 6.5),
        O2iModel::LowLossA => ((0.3, Material::Glass), (0.7, Material::Plywood), 4.4),
    };
    let lin = |m: Material| 10f64.powf(-m.loss_db(fc_hz) / 10.0);
    let tw = 5.0 - 10.0 * (p1 * lin(m1) + p2 * lin(m2)).log10();
    (tw + 0.5 * d2d_in_m, sigma)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code: a failure is a test failure")]
    use super::*;

    fn link(d2d_m: f64, h_bs_m: f64, h_ut_m: f64) -> LinkGeometry {
        LinkGeometry { d2d_m, h_bs_m, h_ut_m }
    }

    #[test]
    fn the_uma_and_umi_los_losses_are_continuous_at_the_breakpoint() {
        for (s, hb) in [(Scenario::UMA, 25.0), (Scenario::UMi, 10.0)] {
            let probe = link(10.0, hb, 1.5);
            let dbp = breakpoint_m(&s, &probe, 3.5e9).unwrap();
            let below = path_loss_db(&s, &link(dbp * (1.0 - 1e-9), hb, 1.5), 3.5e9, true).unwrap();
            let above = path_loss_db(&s, &link(dbp * (1.0 + 1e-9), hb, 1.5), 3.5e9, true).unwrap();
            assert!((below - above).abs() < 1e-6, "{s:?}: {below} {above}");
        }
    }

    #[test]
    fn nlos_is_never_below_los_and_the_breakpoints_follow_the_notes() {
        let scenarios = [
            (Scenario::RMA, 35.0),
            (Scenario::UMA, 25.0),
            (Scenario::UMi, 10.0),
            (Scenario::InhOffice(OfficeLayout::Mixed), 3.0),
        ];
        for (s, hb) in scenarios {
            for d in [12.0, 40.0, 100.0] {
                let l = link(d, hb, 1.5);
                assert!(path_loss_db(&s, &l, 6e9, false).unwrap() >= path_loss_db(&s, &l, 6e9, true).unwrap());
            }
        }
        // NOTE 5 and NOTE 1, c = 3e8 m/s: RMa 2π 35 1.5 3.5e9 / 3e8, UMi 4 9 0.5 3.5e9 / 3e8
        let l = link(100.0, 35.0, 1.5);
        assert!((breakpoint_m(&Scenario::RMA, &l, 3.5e9).unwrap() - 2.0 * PI * 35.0 * 1.5 * 35.0 / 3.0).abs() < 1e-9);
        assert!(
            (breakpoint_m(&Scenario::UMi, &link(100.0, 10.0, 1.5), 3.5e9).unwrap() - 4.0 * 9.0 * 0.5 * 35.0 / 3.0)
                .abs()
                < 1e-9
        );
        assert!(path_loss_db(&Scenario::UMi, &link(5.0, 10.0, 1.5), 3.5e9, true).is_err());
        assert!(path_loss_db(&Scenario::RMA, &link(100.0, 35.0, 1.5), 40e9, true).is_err());
    }

    #[test]
    fn the_los_probabilities_of_table_7_4_2_1() {
        let open = Scenario::InhOffice(OfficeLayout::Open);
        assert_eq!(los_probability(&Scenario::UMi, 18.0, 1.5, 10.0), 1.0);
        assert!((los_probability(&Scenario::UMi, 36.0, 1.5, 10.0) - (0.5 + (-1f64).exp() * 0.5)).abs() < 1e-15);
        assert!((los_probability(&Scenario::RMA, 1010.0, 1.5, 35.0) - (-1f64).exp()).abs() < 1e-15);
        // UMa: C'(h_UT) raises the probability of high UTs
        assert!(los_probability(&Scenario::UMA, 100.0, 20.0, 25.0) > los_probability(&Scenario::UMA, 100.0, 1.5, 25.0));
        // the open office is not continuous at 49 m, as printed: exp(-44/70.8) then 0.54
        assert!((los_probability(&open, 49.0, 1.0, 3.0) - (-44.0f64 / 70.8).exp()).abs() < 1e-15);
        assert!((los_probability(&open, 49.0 + 1e-9, 1.0, 3.0) - 0.54).abs() < 1e-9);
        let sl = InfClutter { sub: InfSubScenario::SparseLow, size_m: 10.0, density: 0.2, height_m: 2.0 };
        let k = -10.0 / 0.8f64.ln();
        assert!((los_probability(&Scenario::InF(sl), 20.0, 1.5, 1.5) - (-20.0 / k).exp()).abs() < 1e-15);
    }

    #[test]
    fn the_uma_environment_height_is_one_metre_below_13_m_and_on_the_grid_above() {
        let mut rng = SimRng::new(4);
        assert!((0..100).all(|_| uma_environment_height(200.0, 12.0, &mut rng) == 1.0));
        let hs: Vec<f64> = (0..2000).map(|_| uma_environment_height(200.0, 22.5, &mut rng)).collect();
        assert!(hs.iter().all(|h| *h == 1.0 || [12.0, 15.0, 18.0, 21.0].contains(h)));
        // C = (0.95)^1.5 g(200) = 0.926 * 1.25 * 8 * e^{-4/3}: P(h_E = 1) = 1 / (1 + C)
        let c = 0.95f64.powf(1.5) * 1.25 * 8.0 * (-200.0f64 / 150.0).exp();
        let p1 = hs.iter().filter(|h| **h == 1.0).count() as f64 / 2000.0;
        assert!((p1 - 1.0 / (1.0 + c)).abs() < 0.04, "{p1}");
    }

    #[test]
    fn the_o2i_models_of_table_7_4_3_2() {
        // low loss at 2 GHz: 5 - 10 log10(0.3 10^{-0.24} + 0.7 10^{-1.3}) + 0.5 d
        let want = 5.0 - 10.0 * (0.3 * 10f64.powf(-0.24) + 0.7 * 10f64.powf(-1.3)).log10() + 5.0;
        let (pl, s) = building_penetration_db(O2iModel::LowLoss, 2e9, 10.0);
        assert!((pl - want).abs() < 1e-12 && s == 4.4);
        assert!(
            building_penetration_db(O2iModel::HighLoss, 2e9, 0.0).0
                > building_penetration_db(O2iModel::LowLoss, 2e9, 0.0).0
        );
        assert!((Material::Concrete.loss_db(28e9) - 117.0).abs() < 1e-12);
    }
}
