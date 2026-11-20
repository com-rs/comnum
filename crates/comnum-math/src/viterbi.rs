//! Dynamic programming over a finite-state machine: the Viterbi algorithm (the path of
//! largest metric), its trace-back, and the soft-output Viterbi algorithm (the
//! reliability of every decision). The machine knows nothing of codes or channels: the
//! convolutional decoders of `comnum-fec` and the maximum-likelihood sequence detector
//! of `comnum-dsp` give it their branch metrics.
//!
//! # Model
//!
//! A [`StateMachine`] has $S$ states and $M$ inputs; input $u$ in state $s$ leads to
//! `next[s][u]`. A path over $T$ steps collects the branch metrics $\gamma_{t}(s, u)$ of
//! its transitions, and the Viterbi algorithm keeps, for every state and step, the
//! survivor of largest accumulated metric (add–compare–select):
//!
//! $$
//! \Gamma_{t+1}(s') = \max_{(s, u):\, \mathrm{next}\[s\]\[u\] = s'} \left[\Gamma_{t}(s) +
//! \gamma_{t}(s, u)\right],
//! $$
//!
//! so that the path that ends in the best final state is the one of largest metric
//! (Viterbi 1967; Forney 1973). Ties keep the first transition in the order $(s, u)$.
//!
//! The soft-output Viterbi algorithm of Hagenauer and Hoeher (1989) attaches to every
//! decision $\hat{u}_{t}$ of the maximum-likelihood path a reliability $\Delta_{t}$: at
//! every state of the path, the metric difference $\Delta$ between the survivor and the
//! best competing path that merges there is assigned to every earlier step where the
//! competitor decides differently, keeping the minimum,
//!
//! $$
//! \Delta_{j} \leftarrow \min(\Delta_{j}, \Delta) \quad \text{for } j \le t:\ u_{j}^{\mathrm{comp}}
//! \ne \hat{u}_{j},
//! $$
//!
//! traced back until the competitor merges with the path. For a binary input and the
//! branch metrics $\frac{1}{2}\sum (1 - 2 c) L$ of a convolutional code, the
//! log-likelihood ratio is $(1 - 2 \hat{u}_{t})\, \Delta_{t}$ (their eq. (10) and
//! Sec. III).
//!
//! # Examples
//! ```
//! use comnum_math::viterbi::{StateMachine, viterbi};
//! // a binary shift register of memory 1: the state is the previous input; the metric
//! // rewards the input 1 at even steps and 0 at odd steps
//! let sm = StateMachine::shift_register(2, 1)?;
//! let gamma = |t: usize, g: &mut [f64]| {
//!     for s in 0..2 { for u in 0..2 { g[2 * s + u] = if (t % 2 == 0) == (u == 1) { 1.0 } else { 0.0 }; } }
//! };
//! let (path, _) = viterbi(&sm, 6, &[0.0, f64::NEG_INFINITY], gamma, None)?;
//! assert_eq!(path, vec![1, 0, 1, 0, 1, 0]);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum
//!    decoding algorithm," *IEEE Trans. Inf. Theory* 13(2), 260–269, 1967.
//! 2. G. D. Forney, "The Viterbi algorithm," *Proc. IEEE* 61(3), 268–278, 1973.
//! 3. J. Hagenauer and P. Hoeher, "A Viterbi algorithm with soft-decision outputs and its
//!    applications," *Proc. IEEE GLOBECOM*, 1680–1686, 1989.

use comnum_core::{Result, ensure};

/// A finite-state machine with `n_states` states and `n_inputs` inputs: the trellis of
/// the Viterbi algorithm (see the module).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateMachine {
    n_states: usize,
    n_inputs: usize,
    /// `next[s * n_inputs + u]`: the state after input `u` in state `s`.
    next: Vec<usize>,
}

impl StateMachine {
    /// The machine of the transition table `next` (`n_states × n_inputs`, flat, state
    /// after input `u` in state `s` at `s * n_inputs + u`).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for no state or no input, a table of the wrong
    /// length, a next state out of range, or more than $2^{31}$ transitions (the
    /// survivors are stored on 32 bits).
    pub fn new(n_states: usize, n_inputs: usize, next: Vec<usize>) -> Result<StateMachine> {
        ensure(n_states >= 1 && n_inputs >= 1, || "a state machine needs a state and an input".into())?;
        ensure(next.len() == n_states * n_inputs, || {
            format!("{} transitions for {n_states} states and {n_inputs} inputs", next.len())
        })?;
        ensure(next.iter().all(|&s| s < n_states), || "a next state is out of range".into())?;
        ensure(n_states * n_inputs <= 1 << 31, || "more than 2^31 transitions".into())?;
        Ok(StateMachine { n_states, n_inputs, next })
    }

    /// A shift register of `memory` cells over an alphabet of `m` symbols: the state is
    /// the number $\sum_{i=1}^{\mathrm{memory}} u_{t-i}\, m^{\,i-1}$ of the last inputs
    /// (the most recent in the least significant digit), $m^{\mathrm{memory}}$ states.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for an alphabet of fewer than two symbols or more
    /// than $2^{31}$ transitions.
    pub fn shift_register(m: usize, memory: usize) -> Result<StateMachine> {
        ensure(m >= 2, || format!("an alphabet of {m} symbols has no trellis"))?;
        let n_states = m.checked_pow(memory as u32).filter(|&n| n.checked_mul(m).is_some_and(|t| t <= 1 << 31));
        let n_states = n_states.ok_or_else(|| {
            comnum_core::Error::Config(format!("a shift register of memory {memory} over {m} symbols is too large"))
        })?;
        let next = (0..n_states).flat_map(|s| (0..m).map(move |u| (s * m + u) % n_states)).collect();
        StateMachine::new(n_states, m, next)
    }

    /// Number of states $S$.
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Number of inputs $M$.
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// The state after input `u` in state `s`.
    pub fn next(&self, s: usize, u: usize) -> usize {
        self.next[s * self.n_inputs + u]
    }

    /// The digit of the input made `age` steps ago in the state `s` of a
    /// [`StateMachine::shift_register`] (`age` 1 is the most recent).
    pub fn digit(&self, s: usize, age: usize) -> usize {
        (s / self.n_inputs.pow(age as u32 - 1)) % self.n_inputs
    }
}

/// The add–compare–select recursion over `steps` steps from the path metrics `metric`
/// (updated in place to the final metrics). `gamma(t, g)` fills the branch metrics of
/// step `t` into `g[s * M + u]`; `from[t * S + s']` receives the survivor into every
/// state, `s * M + u`. With `delta` (same shape), the metric difference between the
/// survivor and the best competitor into every state (infinite when there is none)
/// goes to `delta[t * S + s']` and the competitor to `second[t * S + s']`.
pub fn forward(
    sm: &StateMachine,
    metric: &mut Vec<f64>,
    mut gamma: impl FnMut(usize, &mut [f64]),
    from: &mut [u32],
    mut soft: Option<(&mut [f64], &mut [u32])>,
) {
    let (ns, ni) = (sm.n_states, sm.n_inputs);
    let mut next = vec![f64::NEG_INFINITY; ns];
    let mut second = vec![f64::NEG_INFINITY; ns];
    let mut g = vec![0.0; ns * ni];
    for (t, row) in from.chunks_exact_mut(ns).enumerate() {
        gamma(t, &mut g);
        next.fill(f64::NEG_INFINITY);
        second.fill(f64::NEG_INFINITY);
        let soft_row = soft.as_mut().map(|(d, c)| (&mut d[t * ns..(t + 1) * ns], &mut c[t * ns..(t + 1) * ns]));
        let mut soft_row = soft_row;
        for s in 0..ns {
            if metric[s] == f64::NEG_INFINITY {
                continue;
            }
            for u in 0..ni {
                let s2 = sm.next[s * ni + u];
                let m = metric[s] + g[s * ni + u];
                let branch = (s * ni + u) as u32;
                if m > next[s2] {
                    second[s2] = next[s2];
                    if let Some((_, comp)) = soft_row.as_mut() {
                        comp[s2] = row[s2];
                    }
                    next[s2] = m;
                    row[s2] = branch;
                } else if m > second[s2] {
                    second[s2] = m;
                    if let Some((_, comp)) = soft_row.as_mut() {
                        comp[s2] = branch;
                    }
                }
            }
        }
        if let Some((d, _)) = soft_row.as_mut() {
            for s2 in 0..ns {
                d[s2] = next[s2] - second[s2];
            }
        }
        std::mem::swap(metric, &mut next);
    }
}

/// The state of the largest path metric (the first of equal metrics).
pub fn best_state(metric: &[f64]) -> usize {
    (0..metric.len()).max_by(|a, b| metric[*a].total_cmp(&metric[*b])).unwrap_or(0)
}

/// The inputs of the survivor of [`forward`] that ends in `end`, and the state it starts
/// from.
pub fn traceback(sm: &StateMachine, from: &[u32], end: usize) -> (Vec<usize>, usize) {
    let (ns, ni) = (sm.n_states, sm.n_inputs);
    let steps = from.len() / ns;
    let mut s = end;
    let mut inputs = vec![0usize; steps];
    for t in (0..steps).rev() {
        let su = from[t * ns + s] as usize;
        inputs[t] = su % ni;
        s = su / ni;
    }
    (inputs, s)
}

/// The Viterbi algorithm over `steps` steps from the initial path metrics `start`
/// ($-\infty$ excludes a state): the inputs of the path of largest metric ending in
/// `end`, or in the best final state when `None`, and the final metrics of every state.
///
/// # Errors
/// [`comnum_core::Error::Shape`] if `start` has not one metric per state, or `end` is
/// out of range.
pub fn viterbi(
    sm: &StateMachine,
    steps: usize,
    start: &[f64],
    gamma: impl FnMut(usize, &mut [f64]),
    end: Option<usize>,
) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut metric = checked_start(sm, start, end)?;
    let mut from = vec![0u32; steps * sm.n_states];
    forward(sm, &mut metric, gamma, &mut from, None);
    let end = end.unwrap_or_else(|| best_state(&metric));
    Ok((traceback(sm, &from, end).0, metric))
}

/// The soft-output Viterbi algorithm (Hagenauer & Hoeher 1989, see the module): the
/// inputs of the path of largest metric and the reliability $\Delta_{t} \ge 0$ of every
/// one of them (infinite when no competitor ever disagreed), from the initial metrics
/// `start` to `end` or the best final state. The competitor of every state of the path
/// is traced back until it merges with the path, so that the reliabilities are exact
/// for the whole block.
///
/// # Errors
/// As [`viterbi`].
pub fn sova(
    sm: &StateMachine,
    steps: usize,
    start: &[f64],
    gamma: impl FnMut(usize, &mut [f64]),
    end: Option<usize>,
) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut metric = checked_start(sm, start, end)?;
    let ns = sm.n_states;
    let mut from = vec![0u32; steps * ns];
    let mut delta = vec![f64::INFINITY; steps * ns];
    let mut second = vec![u32::MAX; steps * ns];
    forward(sm, &mut metric, gamma, &mut from, Some((&mut delta, &mut second)));
    let free_end = end.is_none();
    let end = end.unwrap_or_else(|| best_state(&metric));
    let (path, _) = traceback(sm, &from, end);
    // the states of the path: state[t] is the state before step t, state[steps] = end
    let mut states = vec![0usize; steps + 1];
    states[steps] = end;
    for t in (0..steps).rev() {
        states[t] = from[t * ns + states[t + 1]] as usize / sm.n_inputs;
    }
    let mut reliability = vec![f64::INFINITY; steps];
    // the competitor into every state of the path, traced back until it merges
    let mut compete = |t: usize, su: usize, d: f64| {
        let mut su = su;
        let mut j = t;
        loop {
            let (s, u) = (su / sm.n_inputs, su % sm.n_inputs);
            if u != path[j] && d < reliability[j] {
                reliability[j] = d;
            }
            if j == 0 || s == states[j] {
                break;
            }
            j -= 1;
            su = from[j * ns + s] as usize;
        }
    };
    for t in 0..steps {
        let s2 = states[t + 1];
        let d = delta[t * ns + s2];
        if d.is_finite() && second[t * ns + s2] != u32::MAX {
            compete(t, second[t * ns + s2] as usize, d);
        }
    }
    // a free end: the survivors of the other final states compete with the path too
    if free_end && steps > 0 {
        for s in (0..ns).filter(|&s| s != end && metric[s].is_finite()) {
            compete(steps - 1, from[(steps - 1) * ns + s] as usize, metric[end] - metric[s]);
        }
    }
    Ok((path, reliability))
}

fn checked_start(sm: &StateMachine, start: &[f64], end: Option<usize>) -> Result<Vec<f64>> {
    comnum_core::ensure_shape(start.len() == sm.n_states, || {
        format!("{} initial metrics for {} states", start.len(), sm.n_states)
    })?;
    comnum_core::ensure_shape(end.is_none_or(|e| e < sm.n_states), || "the final state is out of range".into())?;
    Ok(start.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register() -> StateMachine {
        StateMachine::shift_register(2, 2).unwrap()
    }

    #[test]
    fn the_shift_register_remembers_its_last_inputs() {
        let sm = register();
        assert_eq!(sm.n_states(), 4);
        // state 0b01 (last input 1, the one before 0), input 1 → 0b11
        assert_eq!(sm.next(1, 1), 3);
        assert_eq!(sm.next(3, 0), 2);
        assert_eq!(sm.digit(2, 1), 0);
        assert_eq!(sm.digit(2, 2), 1);
        assert!(StateMachine::shift_register(1, 2).is_err());
        assert!(StateMachine::shift_register(2, 40).is_err());
        let three = StateMachine::shift_register(3, 1).unwrap();
        assert_eq!(three.next(2, 1), 1);
    }

    #[test]
    fn the_best_path_is_found_by_exhaustive_comparison() {
        // random branch metrics on a 4-state binary trellis: the Viterbi path has the
        // largest metric among the 2^T paths from state 0
        let sm = register();
        let steps = 8;
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let g: Vec<f64> = (0..steps * 8).map(|_| rnd()).collect();
        let gamma = |t: usize, out: &mut [f64]| out.copy_from_slice(&g[t * 8..(t + 1) * 8]);
        let mut start = vec![f64::NEG_INFINITY; 4];
        start[0] = 0.0;
        let (path, metric) = viterbi(&sm, steps, &start, gamma, None).unwrap();
        let score = |inputs: &[usize]| {
            let mut s = 0;
            let mut m = 0.0;
            for (t, &u) in inputs.iter().enumerate() {
                m += g[t * 8 + s * 2 + u];
                s = sm.next(s, u);
            }
            (m, s)
        };
        let best = (0..1usize << steps)
            .map(|w| (0..steps).map(|t| (w >> t) & 1).collect::<Vec<_>>())
            .map(|p| score(&p).0)
            .fold(f64::NEG_INFINITY, f64::max);
        let (m, end) = score(&path);
        assert!((m - best).abs() < 1e-12);
        assert!((metric[end] - best).abs() < 1e-12);
        assert_eq!(metric.iter().copied().fold(f64::NEG_INFINITY, f64::max), best);
    }

    #[test]
    fn the_soft_output_follows_the_path_and_measures_the_competitors() {
        // a binary register of memory 1 driven by noisy ±1 observations of its input:
        // γ(u) = (1 - 2u) r, the path is the sign of r, the reliability of an isolated
        // decision is 2|r| (the competitor flips one input)
        let sm = StateMachine::shift_register(2, 1).unwrap();
        let r = [1.5, -0.2, 0.8, -1.1, 0.05, 0.9];
        let gamma = |t: usize, g: &mut [f64]| {
            for s in 0..2 {
                g[2 * s] = r[t];
                g[2 * s + 1] = -r[t];
            }
        };
        let (path, rel) = sova(&sm, r.len(), &[0.0, 0.0], gamma, None).unwrap();
        let (hard, _) = viterbi(&sm, r.len(), &[0.0, 0.0], gamma, None).unwrap();
        assert_eq!(path, hard);
        assert_eq!(path, vec![0, 1, 0, 1, 0, 0]);
        // the metrics are memoryless: every competitor differs in one decision and
        // merges at once, Δ_t = 2|r_t|
        for (t, &v) in r.iter().enumerate() {
            assert!((rel[t] - 2.0 * v.abs()).abs() < 1e-12, "{t}: {} against {}", rel[t], 2.0 * v.abs());
        }
        assert!(viterbi(&sm, 3, &[0.0], gamma, None).is_err());
        assert!(sova(&sm, 3, &[0.0, 0.0], gamma, Some(5)).is_err());
    }

    #[test]
    fn the_reliabilities_bound_the_max_log_differences_of_every_path() {
        // exhaustive: Δ_t is the metric difference with a competitor that disagrees at t,
        // so it is at least the difference with the best disagreeing path (max-log-MAP)
        let sm = register();
        let steps = 7;
        let mut seed = 777u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for trial in 0..20 {
            let g: Vec<f64> = (0..steps * 8).map(|_| rnd()).collect();
            let gamma = |t: usize, out: &mut [f64]| out.copy_from_slice(&g[t * 8..(t + 1) * 8]);
            let mut start = vec![f64::NEG_INFINITY; 4];
            start[0] = 0.0;
            let end = if trial % 2 == 0 { Some(0) } else { None };
            let (path, rel) = sova(&sm, steps, &start, gamma, end).unwrap();
            let score = |inputs: &[usize]| {
                let mut s = 0;
                let mut m = 0.0;
                for (t, &u) in inputs.iter().enumerate() {
                    m += g[t * 8 + s * 2 + u];
                    s = sm.next(s, u);
                }
                (m, s)
            };
            let paths: Vec<(Vec<usize>, f64)> = (0..1usize << steps)
                .map(|w| (0..steps).map(|t| (w >> t) & 1).collect::<Vec<_>>())
                .filter_map(|p| {
                    let (m, s) = score(&p);
                    end.is_none_or(|e| e == s).then_some((p, m))
                })
                .collect();
            let best = score(&path).0;
            for t in 0..steps {
                let other =
                    paths.iter().filter(|(p, _)| p[t] != path[t]).map(|(_, m)| *m).fold(f64::NEG_INFINITY, f64::max);
                assert!(rel[t] >= best - other - 1e-12, "trial {trial}, step {t}: {} against {}", rel[t], best - other);
            }
        }
    }

    #[test]
    fn a_forced_end_state_is_respected() {
        let sm = register();
        let gamma = |_: usize, g: &mut [f64]| g.fill(0.0);
        let (path, _) = viterbi(&sm, 5, &[0.0; 4], gamma, Some(3)).unwrap();
        assert_eq!(&path[3..], &[1, 1]);
    }
}
