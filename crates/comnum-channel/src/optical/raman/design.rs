//! The pump-power optimiser: the launch powers of pumps at given wavelengths that
//! flatten the on-off gain over the signals.
//!
//! # Model
//!
//! By the gain identity of the coupled power equations (Perlin & Winful 2002, eq. (2);
//! Hu, Marks & Menyuk 2004, eq. (2)), the on-off gain of signal $k$ is linear in the
//! integrated powers $I_{j} = \int_{0}^{L} P_{j}\\, dz$ of the waves,
//!
//! $$
//! \ln G^{\mathrm{on/off}}_{k} = \sum_{j \in \mathrm{pumps}} C_{kj} I_{j} + \sum_{s \in \mathrm{signals}} C_{ks} I_{s},
//! $$
//!
//! so the design splits in two (Perlin & Winful 2002, Sec. II), with the pump
//! frequencies fixed (Hu, Marks & Menyuk 2004, the case of pumps specified a priori):
//!
//! 1. the integrals $I_{j}^{*} \ge 0$ that bring the first sum closest to the targets
//!    minus the signal–signal term, a linear fit: non-negative least squares
//!    ([`comnum_math::linalg::nnls`]) or the non-negative minimax fit of Lawson's
//!    iteratively reweighted least squares ([`comnum_math::linalg::nnls_minimax`]), the
//!    flatness criterion of both papers (the spread of the gain in dB);
//! 2. the launch powers whose solution of the equations has these integrals: Newton's
//!    method on the logarithms of the powers, the Jacobian $\partial \ln I_{j} /
//!    \partial \ln P_{l}$ by finite differences, recomputed when a step does not reduce
//!    the mismatch (the papers use an unspecified iterative correction and a shooting
//!    method).
//!
//! The two stages alternate until the integrals match their targets within
//! [`DESIGN_TOLERANCE`] (the outer iterations of Perlin & Winful, which update the
//! signal integrals of the Raman tilt). A pump whose target integral is zero is left at
//! zero power.
//!
//! The ripple is the peak-to-peak spread of $G^{\mathrm{on/off}}_{k}$ minus its target
//! over the signals (dB): Hu's "signal gain ripple" when the targets are the losses of
//! the signals (a flat net gain), Perlin's $F_{\mathrm{rel}}$ times the target when the
//! target is flat.
//!
//! # References
//!
//! 1. V. E. Perlin and H. G. Winful, "Optimal design of flat-gain wide-band fiber Raman
//!    amplifiers," *J. Lightw. Technol.*, 20(2), 250–254, 2002.
//! 2. J. Hu, B. S. Marks and C. R. Menyuk, "Flat-gain fiber Raman amplifiers using
//!    equally spaced pumps," *J. Lightw. Technol.*, 22(6), 1519–1522, 2004.

use super::{Raman, RamanProfile, RamanPump};
use crate::optical::fiber::FiberParams;
use comnum_core::{Error, Result, ensure};
use comnum_math::linalg::{lstsq, nnls, nnls_minimax};
use comnum_math::{db2np, np2db};
use ndarray::Array2;

/// What the fit of the integrated pump powers minimises over the signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DesignObjective {
    /// The sum of the squared gain errors (dB²).
    LeastSquares,
    /// The largest gain error (dB): the flatness of Perlin & Winful and of Hu et al.
    Minimax,
}

/// The convergence of the design: the largest $|\ln (I_{j}/I_{j}^{*})|$ of the pumps
/// with a non-zero target.
pub const DESIGN_TOLERANCE: f64 = 1e-5;
/// The largest number of outer iterations of the design.
pub const DESIGN_MAX_ITERATIONS: usize = 60;
/// The iterations of the minimax fit.
const MINIMAX_ITERATIONS: usize = 200;
/// The relative step of the finite differences of the Jacobian.
const JACOBIAN_STEP: f64 = 1e-3;
/// The largest Newton step on $\ln P$ (a factor $e$).
const MAX_STEP: f64 = 1.0;
/// The power of a pump that the design leaves out (W).
const OFF_W: f64 = 1e-12;

/// A design of [`Raman::design_pumps`].
#[derive(Clone, Debug, PartialEq)]
pub struct PumpDesign {
    /// The pumps with the powers found (W).
    pub pumps: Vec<RamanPump>,
    /// The target integrated power of every pump, $I_{j}^{*}$ (W·km).
    pub integrals_w_km: Vec<f64>,
    /// The profile of the span with the pumps found.
    pub profile: RamanProfile,
    /// The peak-to-peak spread of the on-off gain minus its target over the signals (dB).
    pub ripple_db: f64,
    /// The number of outer iterations.
    pub iterations: usize,
}

impl Raman {
    /// The powers of the pumps of `self` (their frequencies and directions kept, their
    /// powers the initial guess when positive) that bring the on-off gain of every
    /// signal of `signals` (absolute frequency in Hz, launch power in W) over one span of
    /// `fiber` closest to `target_on_off_db` (one value per signal), by `objective` (see
    /// the module).
    ///
    /// # Errors
    /// [`Error::Config`] for no pump or a target per signal missing, [`Error::Numerical`]
    /// when the design does not converge in [`DESIGN_MAX_ITERATIONS`] iterations, and as
    /// [`Raman::solve`].
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::optical::fiber::FiberParams;
    /// use comnum_channel::optical::raman::{DesignObjective, Direction, Raman, RamanPump};
    /// // two counter-pumps for a flat on-off gain of 10 dB over 8 channels of 80 km
    /// let fiber = FiberParams::default();
    /// let raman = Raman {
    ///     pumps: vec![
    ///         RamanPump::at_wavelength(1430.0, 0.0, Direction::Counter),
    ///         RamanPump::at_wavelength(1460.0, 0.0, Direction::Counter),
    ///     ],
    ///     ..Raman::default()
    /// };
    /// let signals: Vec<(f64, f64)> = (0..8).map(|k| (191e12 + k as f64 * 0.5e12, 1e-3)).collect();
    /// let d = raman.design_pumps(&fiber, &signals, &[10.0; 8], DesignObjective::Minimax)?;
    /// assert!(d.ripple_db < 1.0 && d.pumps.iter().all(|p| p.power_w > 0.0));
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn design_pumps(
        &self,
        fiber: &FiberParams,
        signals: &[(f64, f64)],
        target_on_off_db: &[f64],
        objective: DesignObjective,
    ) -> Result<PumpDesign> {
        let (m, n) = (signals.len(), self.pumps.len());
        ensure(n >= 1, || "raman design: no pump".into())?;
        ensure(target_on_off_db.len() == m, || {
            format!("raman design: {} targets for {m} signals", target_on_off_db.len())
        })?;
        let target_np: Vec<f64> = target_on_off_db.iter().map(|t| db2np(*t)).collect();
        let mut a = Array2::<f64>::zeros((m, n));
        for k in 0..m {
            for j in 0..n {
                a[[k, j]] = self.coupling(signals[k].0, self.pumps[j].frequency_hz);
            }
        }
        let fit = |rhs: &[f64]| -> Result<Vec<f64>> {
            match objective {
                DesignObjective::LeastSquares => nnls(&a.view(), rhs),
                DesignObjective::Minimax => nnls_minimax(&a.view(), rhs, MINIMAX_ITERATIONS),
            }
        };
        // the right-hand side of the fit for a profile: the targets minus the Raman tilt
        let rhs_of = |p: &RamanProfile| -> Vec<f64> {
            (0..m)
                .map(|k| {
                    let tilt: f64 = if self.isrs {
                        (0..m).map(|s| self.coupling(signals[k].0, signals[s].0) * p.integral_w_km(s)).sum()
                    } else {
                        0.0
                    };
                    target_np[k] - tilt
                })
                .collect()
        };
        let solve = |powers: &[f64]| -> Result<RamanProfile> {
            let pumps = self.pumps.iter().zip(powers).map(|(p, w)| RamanPump { power_w: *w, ..p.clone() }).collect();
            Raman { pumps, ..self.clone() }.solve(fiber, signals)
        };
        // the initial guess: the undepleted integrals of the loss-only profile
        let mut powers: Vec<f64> = self.pumps.iter().map(|p| p.power_w).collect();
        let unpumped = solve(&vec![OFF_W; n])?;
        let mut targets = fit(&rhs_of(&unpumped))?;
        for j in 0..n {
            if powers[j] <= 0.0 {
                let l_eff = unpumped.integral_w_km(m + j) / OFF_W;
                powers[j] = (targets[j] / l_eff).max(OFF_W);
            }
        }
        let mut jacobian: Option<Vec<Vec<f64>>> = None;
        let mut last = f64::INFINITY;
        let mut profile = solve(&powers)?;
        for iteration in 1..=DESIGN_MAX_ITERATIONS {
            targets = fit(&rhs_of(&profile))?;
            let active: Vec<usize> = (0..n).filter(|j| targets[*j] > 0.0).collect();
            for j in 0..n {
                if targets[j] <= 0.0 {
                    powers[j] = OFF_W;
                }
            }
            let mismatch: Vec<f64> = active.iter().map(|&j| (targets[j] / profile.integral_w_km(m + j)).ln()).collect();
            let residual = mismatch.iter().map(|v| v.abs()).fold(0.0, f64::max);
            if residual < DESIGN_TOLERANCE {
                return Ok(self.design(&powers, targets, profile, target_on_off_db, iteration));
            }
            if jacobian.as_ref().is_none_or(|j| j.len() != active.len()) || residual > 0.5 * last {
                let mut jac = vec![vec![0.0; active.len()]; active.len()];
                for (q, &l) in active.iter().enumerate() {
                    let mut perturbed = powers.clone();
                    perturbed[l] *= (JACOBIAN_STEP).exp();
                    let p = solve(&perturbed)?;
                    for (r, &j) in active.iter().enumerate() {
                        jac[r][q] = (p.integral_w_km(m + j) / profile.integral_w_km(m + j)).ln() / JACOBIAN_STEP;
                    }
                }
                jacobian = Some(jac);
            }
            let jac = jacobian.as_ref().ok_or_else(|| Error::Numerical("raman design: no Jacobian".into()))?;
            let mut flat = Array2::<f64>::zeros((active.len(), active.len()));
            for (r, row) in jac.iter().enumerate() {
                for (q, v) in row.iter().enumerate() {
                    flat[[r, q]] = *v;
                }
            }
            let step = lstsq(&flat.view(), &mismatch)?;
            for (q, &j) in active.iter().enumerate() {
                powers[j] *= step[q].clamp(-MAX_STEP, MAX_STEP).exp();
            }
            last = residual;
            profile = solve(&powers)?;
        }
        Err(Error::Numerical(format!("raman design: no convergence after {DESIGN_MAX_ITERATIONS} iterations")))
    }

    fn design(
        &self,
        powers: &[f64],
        integrals_w_km: Vec<f64>,
        profile: RamanProfile,
        target_on_off_db: &[f64],
        iterations: usize,
    ) -> PumpDesign {
        let errors: Vec<f64> =
            (0..target_on_off_db.len()).map(|k| profile.on_off_gain_db(k) - target_on_off_db[k]).collect();
        let ripple_db =
            errors.iter().copied().fold(f64::MIN, f64::max) - errors.iter().copied().fold(f64::MAX, f64::min);
        let pumps = self
            .pumps
            .iter()
            .zip(powers)
            .map(|(p, w)| RamanPump { power_w: if *w <= OFF_W { 0.0 } else { *w }, ..p.clone() })
            .collect();
        PumpDesign { pumps, integrals_w_km, profile, ripple_db: np2db(0.0) + ripple_db, iterations }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Direction, RamanGain};
    use super::*;
    use comnum_math::nm2hz;

    #[test]
    fn one_pump_reaches_the_target_of_one_signal() {
        let fiber = FiberParams { span_km: 50.0, ..FiberParams::default() };
        let raman = Raman {
            pumps: vec![RamanPump::at_wavelength(1450.0, 0.0, Direction::Counter)],
            gain: RamanGain::flat(0.4, 80.0, nm2hz(1450.0)).unwrap(),
            isrs: false,
            ..Raman::default()
        };
        for objective in [DesignObjective::LeastSquares, DesignObjective::Minimax] {
            let d = raman.design_pumps(&fiber, &[(nm2hz(1550.0), 1e-4)], &[12.0], objective).unwrap();
            assert!((d.profile.on_off_gain_db(0) - 12.0).abs() < 1e-4, "{}", d.profile.on_off_gain_db(0));
            assert!(d.ripple_db.abs() < 1e-12);
            // the undepleted closed form: P = ln G / (C L_eff)
            let a_p = fiber.alpha_np_km();
            let expected = db2np(12.0) / (0.4 * (1.0 - (-a_p * 50.0).exp()) / a_p);
            assert!((d.pumps[0].power_w / expected - 1.0).abs() < 2e-2, "{} vs {expected}", d.pumps[0].power_w);
        }
    }

    #[test]
    fn four_pumps_flatten_forty_channels_and_the_integrals_match() {
        let fiber = FiberParams { span_km: 80.0, ..FiberParams::default() };
        let raman = Raman {
            pumps: [1425.0, 1445.0, 1465.0, 1490.0]
                .iter()
                .map(|w| RamanPump::at_wavelength(*w, 0.0, Direction::Counter))
                .collect(),
            ..Raman::default()
        };
        let signals: Vec<(f64, f64)> = (0..40).map(|k| (187.5e12 + k as f64 * 250e9, 5e-4)).collect();
        let ls = raman.design_pumps(&fiber, &signals, &[16.0; 40], DesignObjective::LeastSquares).unwrap();
        let mm = raman.design_pumps(&fiber, &signals, &[16.0; 40], DesignObjective::Minimax).unwrap();
        assert!(mm.ripple_db <= ls.ripple_db + 1e-9, "{} vs {}", mm.ripple_db, ls.ripple_db);
        assert!(mm.ripple_db < 2.5, "{}", mm.ripple_db);
        for (j, i) in mm.integrals_w_km.iter().enumerate() {
            if *i > 0.0 {
                assert!((mm.profile.integral_w_km(40 + j) / i - 1.0).abs() < 2.0 * DESIGN_TOLERANCE);
            }
        }
        // the design reaches the mean target
        let mean: f64 = (0..40).map(|k| mm.profile.on_off_gain_db(k)).sum::<f64>() / 40.0;
        assert!((mean - 16.0).abs() < 1.0, "{mean}");
    }

    #[test]
    fn invalid_designs_are_errors() {
        let fiber = FiberParams::default();
        let none = Raman::default();
        assert!(none.design_pumps(&fiber, &[(193e12, 1e-3)], &[10.0], DesignObjective::Minimax).is_err());
        let one = Raman { pumps: vec![RamanPump::at_wavelength(1450.0, 0.1, Direction::Co)], ..Raman::default() };
        assert!(one.design_pumps(&fiber, &[(193e12, 1e-3)], &[10.0, 10.0], DesignObjective::Minimax).is_err());
    }
}
