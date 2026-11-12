//! The boundary-value problem of the coupled power equations: forward and backward
//! Runge-Kutta sweeps with relaxation, and the spontaneous Raman noise.

use super::{Direction, Raman, RamanProfile};
use crate::optical::fiber::FiberParams;
use comnum_core::{Error, Result, ensure};
use comnum_math::{BOLTZMANN, PLANCK, db2np, trapezoid_uniform};

/// The waves of the equations.
struct Wave {
    f_hz: f64,
    alpha_np_km: f64,
    p_in_w: f64,
    counter: bool,
    signal: bool,
}

/// The smallest power of a wave: the logarithms stay finite for empty slices.
const POWER_FLOOR_W: f64 = 1e-30;

/// The largest number of steps of the grid over a span (10 m steps over 10,000 km).
const MAX_CELLS: f64 = 1e6;

impl Raman {
    fn check(&self) -> Result<()> {
        ensure(self.aeff_um2 > 0.0, || format!("raman: aeff_um2 must be positive, got {}", self.aeff_um2))?;
        ensure(self.temperature_k > 0.0, || {
            format!("raman: temperature_k must be positive, got {}", self.temperature_k)
        })?;
        ensure(self.step_km > 0.0, || format!("raman: step_km must be positive, got {}", self.step_km))?;
        ensure(self.tolerance > 0.0 && self.max_iterations >= 1, || {
            "raman: tolerance must be positive and max_iterations at least 1".into()
        })?;
        ensure(self.slice_hz > 0.0, || format!("raman: slice_hz must be positive, got {}", self.slice_hz))?;
        for (name, table, floor) in [
            ("aeff_spectrum_um2", &self.aeff_spectrum_um2, f64::MIN_POSITIVE),
            ("alpha_spectrum_db_km", &self.alpha_spectrum_db_km, 0.0),
        ] {
            ensure(
                table.iter().all(|(f, v)| f.is_finite() && *f > 0.0 && v.is_finite() && *v >= floor)
                    && table.windows(2).all(|w| w[1].0 > w[0].0),
                || {
                    format!(
                        "raman: {name} needs positive, increasing frequencies and finite values (positive areas, non-negative attenuations), got {table:?}"
                    )
                },
            )?;
        }
        for p in &self.pumps {
            ensure(
                p.frequency_hz.is_finite()
                    && p.frequency_hz > 0.0
                    && p.power_w.is_finite()
                    && p.power_w >= 0.0
                    && p.alpha_db_km.is_none_or(|a| a.is_finite() && a >= 0.0),
                || {
                    format!(
                        "raman: a pump needs a finite positive frequency, a finite non-negative power and attenuation, got {p:?}"
                    )
                },
            )?;
        }
        ensure(self.rayleigh_db_km.is_none_or(f64::is_finite), || {
            format!("raman: rayleigh_db_km must be finite, got {:?}", self.rayleigh_db_km)
        })?;
        Ok(())
    }

    /// The power profiles of one span of `fiber` for the co-propagating `signals`
    /// (absolute frequency in Hz, launch power in W) and the pumps.
    ///
    /// # Errors
    /// [`Error::Config`] for invalid parameters or no signal, [`Error::Numerical`] if
    /// the sweeps do not converge.
    pub fn solve(&self, fiber: &FiberParams, signals: &[(f64, f64)]) -> Result<RamanProfile> {
        self.check()?;
        ensure(!signals.is_empty(), || "raman: no signal".into())?;
        ensure(fiber.span_km > 0.0 && fiber.span_km.is_finite(), || {
            "raman: the span must have a positive length".into()
        })?;
        ensure(signals.iter().all(|(f, p)| f.is_finite() && *f > 0.0 && p.is_finite() && *p >= 0.0), || {
            "raman: signals need finite positive frequencies and non-negative powers".into()
        })?;
        let alpha = |f: f64| db2np(self.alpha_db_km_at(fiber, f));
        let mut waves: Vec<Wave> = signals
            .iter()
            .map(|(f, p)| Wave {
                f_hz: *f,
                alpha_np_km: alpha(*f),
                p_in_w: p.max(POWER_FLOOR_W),
                counter: false,
                signal: true,
            })
            .collect();
        waves.extend(self.pumps.iter().map(|p| Wave {
            f_hz: p.frequency_hz,
            alpha_np_km: p.alpha_db_km.map_or_else(|| alpha(p.frequency_hz), db2np),
            p_in_w: p.power_w.max(POWER_FLOOR_W),
            counter: p.direction == Direction::Counter,
            signal: false,
        }));
        let n = waves.len();
        let mut c = vec![vec![0.0; n]; n];
        for k in 0..n {
            for j in 0..n {
                if !self.isrs && waves[k].signal && waves[j].signal {
                    continue;
                }
                c[k][j] = self.coupling(waves[k].f_hz, waves[j].f_hz);
            }
        }
        let length = fiber.span_km;
        let cells = (length / self.step_km).ceil();
        ensure(cells <= MAX_CELLS, || {
            format!("raman: {cells} steps of {} km over the span, more than {MAX_CELLS}", self.step_km)
        })?;
        let m = (cells as usize).max(1);
        let h = length / m as f64;
        let z: Vec<f64> = (0..=m).map(|i| i as f64 * h).collect();
        // the initial guess: the loss alone
        let mut y: Vec<Vec<f64>> = waves
            .iter()
            .map(|w| {
                z.iter().map(|zi| w.p_in_w.ln() - w.alpha_np_km * if w.counter { length - zi } else { *zi }).collect()
            })
            .collect();
        let co: Vec<usize> = (0..n).filter(|k| !waves[*k].counter).collect();
        let counter: Vec<usize> = (0..n).filter(|k| waves[*k].counter).collect();
        let sys = System { waves: &waves, c: &c, h };
        let mut iterations = 0;
        if counter.is_empty() {
            sys.sweep(&mut y, &co, true, 1.0);
            iterations = 1;
        } else {
            let pumps: Vec<usize> = (0..n).filter(|k| !waves[*k].signal).collect();
            // the loss-only guess, kept for the continuation (a failed relaxation leaves
            // its last sweeps in y)
            let guess = y.clone();
            if let Ok(it) = sys.relax(&mut y, &co, &counter, self.tolerance, self.max_iterations) {
                iterations = it;
            } else {
                // continuation from a fraction of the pump powers (a strongly depleted
                // amplifier: the sweeps from the loss-only guess oscillate): the fraction
                // is halved until the sweeps converge, then doubled from the previous
                // solution, whose pump rows are shifted by the logarithm of the ratio of
                // the fractions
                let (mut solved, mut f) = (0.0f64, 0.5f64);
                loop {
                    let (mut trial, shift) =
                        if solved == 0.0 { (guess.clone(), f.ln()) } else { (y.clone(), (f / solved).ln()) };
                    for k in &pumps {
                        for v in &mut trial[*k] {
                            *v += shift;
                        }
                    }
                    match sys.relax(&mut trial, &co, &counter, self.tolerance, self.max_iterations) {
                        Ok(it) => {
                            iterations += it;
                            y = trial;
                            solved = f;
                            if f >= 1.0 {
                                break;
                            }
                            f = (2.0 * f).min(1.0);
                        }
                        Err(e) => {
                            f = (solved + f) / 2.0;
                            if f - solved < 1.0 / 256.0 {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        let ase_psd = (0..signals.len()).map(|k| self.spontaneous(&waves, &c, &y, k, h, true)).collect();
        let ase_backward_psd = (0..signals.len()).map(|k| self.spontaneous(&waves, &c, &y, k, h, false)).collect();
        let mut profile = RamanProfile {
            z_km: z,
            freqs_hz: waves.iter().map(|w| w.f_hz).collect(),
            directions: waves.iter().map(|w| if w.counter { Direction::Counter } else { Direction::Co }).collect(),
            alpha_np_km: waves.iter().map(|w| w.alpha_np_km).collect(),
            n_signals: signals.len(),
            log_power: y,
            ase_psd,
            ase_backward_psd,
            double_rayleigh: vec![0.0; signals.len()],
            iterations,
        };
        if let Some(r) = self.rayleigh_db_km {
            profile.double_rayleigh = (0..signals.len()).map(|k| profile.double_rayleigh_at(k, r)).collect();
        }
        Ok(profile)
    }

    /// The spontaneous noise of signal `k` at the end of the span (`forward`) or, for the
    /// noise emitted towards $-z$, at its start (both polarisations), by the trapezoidal
    /// rule on the grid: the source $2 h f_{k} \sum_{j} C_{kj} P_{j}(z) (1 + \eta_{kj})$
    /// amplified by $G_{k}(L)/G_{k}(z)$ to the end, or by $G_{k}(z)$ back to the start
    /// (the Raman gain is the same in both directions).
    fn spontaneous(&self, waves: &[Wave], c: &[Vec<f64>], y: &[Vec<f64>], k: usize, h: f64, forward: bool) -> f64 {
        let fk = waves[k].f_hz;
        let sources: Vec<(usize, f64)> = (0..waves.len())
            .filter(|j| waves[*j].f_hz > fk && !waves[*j].signal && c[k][*j] > 0.0)
            .map(|j| {
                let x = PLANCK * (waves[j].f_hz - fk) / (BOLTZMANN * self.temperature_k);
                (j, c[k][j] * (1.0 + 1.0 / x.exp_m1()))
            })
            .collect();
        if sources.is_empty() {
            return 0.0;
        }
        let yk = &y[k];
        let end = yk[yk.len() - 1];
        let integrand: Vec<f64> = (0..yk.len())
            .map(|i| {
                let gain = if forward { end - yk[i] } else { yk[i] - yk[0] };
                sources.iter().map(|(j, cj)| cj * y[*j][i].exp()).sum::<f64>() * gain.exp()
            })
            .collect();
        2.0 * PLANCK * fk * trapezoid_uniform(&integrand, h)
    }
}

/// The equations on the grid.
struct System<'a> {
    waves: &'a [Wave],
    c: &'a [Vec<f64>],
    h: f64,
}

impl System<'_> {
    /// The forward-backward sweeps from the profiles `y` until the logarithms change by
    /// less than `tolerance`, with under-relaxation of the backward sweep when the
    /// changes grow: the number of sweeps.
    fn relax(
        &self,
        y: &mut [Vec<f64>],
        co: &[usize],
        counter: &[usize],
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<usize> {
        let mut omega: f64 = 1.0;
        let mut last = f64::INFINITY;
        let mut iterations = 0;
        loop {
            iterations += 1;
            let r_co = self.sweep(y, co, true, 1.0);
            let r_cnt = self.sweep(y, counter, false, omega);
            let residue = r_co.max(r_cnt / omega);
            if !residue.is_finite() {
                return Err(Error::Numerical("raman: the power profiles diverge".into()));
            }
            if residue < tolerance {
                return Ok(iterations);
            }
            if iterations >= max_iterations {
                return Err(Error::Numerical(format!(
                    "raman: no convergence after {iterations} sweeps (change {residue:.2e})"
                )));
            }
            if residue > last {
                omega = (omega / 2.0).max(1.0 / 64.0);
            }
            last = residue;
        }
    }

    /// $d \ln P_{k} / dz$ along $+z$ of the waves of `group` for their logarithms
    /// `own` and the logarithms `fixed` of every wave (the others are read there): the
    /// rate $-\alpha_{k} + \sum_{j} C_{kj} P_{j}$, opposite for the counter-propagating
    /// waves.
    fn rates(&self, group: &[usize], own: &[f64], fixed: &[f64], p: &mut [f64], out: &mut [f64]) {
        for (p, l) in p.iter_mut().zip(fixed) {
            *p = l.exp();
        }
        for (g, l) in group.iter().zip(own) {
            p[*g] = l.exp();
        }
        for (o, k) in out.iter_mut().zip(group) {
            let gain: f64 = self.c[*k].iter().zip(p.iter()).map(|(c, p)| c * p).sum();
            let r = -self.waves[*k].alpha_np_km + gain;
            *o = if self.waves[*k].counter { -r } else { r };
        }
    }

    /// Integrate the waves of `group` over the grid (forward from $z = 0$ or backward
    /// from $z = L$) by RK4 with the other waves fixed, and relax the new profiles by
    /// `omega`: the largest change of the logarithms.
    fn sweep(&self, y: &mut [Vec<f64>], group: &[usize], forward: bool, omega: f64) -> f64 {
        let m = y[0].len() - 1;
        let n = y.len();
        let g = group.len();
        let mut new: Vec<Vec<f64>> = group.iter().map(|k| y[*k].clone()).collect();
        let (start, step) = if forward { (0, 1i64) } else { (m, -1i64) };
        let h = if forward { self.h } else { -self.h };
        let mut s: Vec<f64> = group.iter().map(|k| y[*k][start]).collect();
        let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0; g], vec![0.0; g], vec![0.0; g], vec![0.0; g]);
        let mut tmp = vec![0.0; g];
        // the logarithms of every wave at both ends of the step and at its middle, and
        // their powers (the buffers of every step)
        let mut fa: Vec<f64> = (0..n).map(|k| y[k][start]).collect();
        let (mut fb, mut fm, mut p) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut i = start;
        for _ in 0..m {
            let next = (i as i64 + step) as usize;
            for k in 0..n {
                fb[k] = y[k][next];
                fm[k] = (fa[k] + fb[k]) / 2.0;
            }
            // the backward sweep integrates the same derivative along +z with h < 0
            self.rates(group, &s, &fa, &mut p, &mut k1);
            for q in 0..g {
                tmp[q] = s[q] + h / 2.0 * k1[q];
            }
            self.rates(group, &tmp, &fm, &mut p, &mut k2);
            for q in 0..g {
                tmp[q] = s[q] + h / 2.0 * k2[q];
            }
            self.rates(group, &tmp, &fm, &mut p, &mut k3);
            for q in 0..g {
                tmp[q] = s[q] + h * k3[q];
            }
            self.rates(group, &tmp, &fb, &mut p, &mut k4);
            for q in 0..g {
                s[q] += h / 6.0 * (k1[q] + 2.0 * k2[q] + 2.0 * k3[q] + k4[q]);
                new[q][next] = s[q];
            }
            std::mem::swap(&mut fa, &mut fb);
            i = next;
        }
        let mut change: f64 = 0.0;
        for (q, k) in group.iter().enumerate() {
            for (old, v) in y[*k].iter_mut().zip(&new[q]) {
                let relaxed = *old + omega * (v - *old);
                change = change.max((relaxed - *old).abs());
                *old = relaxed;
            }
        }
        change
    }
}
