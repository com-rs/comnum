//! Dynamic programming over a finite-state machine: the Viterbi algorithm (the path of
//! largest metric), its trace-back, and the soft-output Viterbi algorithm (the
//! reliability of every decision). The machine knows nothing of codes or channels: the
//! convolutional decoders of `comnum-fec` and the sequence detector of `comnum-dsp` give
//! it their branch metrics.
//!
//! # Model
//!
//! A [`StateMachine`] has $S$ states and $M$ inputs; input $u$ in state $s$ leads to
//! `next(s, u)`. A path over $T$ steps collects the branch metrics $\gamma_{t}(s, u)$ of
//! its transitions, and the Viterbi algorithm keeps, for every state and step, the
//! survivor of largest accumulated metric (add–compare–select):
//!
//! $$
//! \Gamma_{t+1}(s') = \max_{(s, u):\, \mathrm{next}(s, u) = s'} \left[\Gamma_{t}(s) +
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
//! traced back until the competitor merges with the path (the whole block, no window:
//! the reliabilities are exact, at a cost that grows with the depth of the merges). For
//! a binary input and the branch metrics $\frac{1}{2}\sum (1 - 2 c) L$ of a
//! convolutional code, the log-likelihood ratio is $(1 - 2 \hat{u}_{t})\, \Delta_{t}$
//! ([`binary_llr`]; their eq. (10) and Sec. III).
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

use comnum_core::{Result, ensure, ensure_shape};

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
    /// (the most recent in the least significant digit), $m^{\mathrm{memory}}$ states;
    /// [`StateMachine::digit`] reads them back.
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

    /// The input made `age` steps ago (1 the most recent) in the state `s` of a
    /// [`StateMachine::shift_register`]; a digit of `s` in base $M$ for any other machine.
    pub fn digit(&self, s: usize, age: usize) -> usize {
        (s / self.n_inputs.pow(age as u32 - 1)) % self.n_inputs
    }
}

/// The survivor tables of [`forward`] over a block: for every step and state, the
/// transition `s * M + u` that won, and, when asked for, the best competing transition
/// with the metric it lost by (the soft-output algorithm reads them).
#[derive(Clone, Debug, PartialEq)]
pub struct Survivors {
    n_states: usize,
    n_inputs: usize,
    from: Vec<u32>,
    /// `(delta, second)`: the metric difference with the best competitor into every
    /// state (infinite when there is none) and that competitor's transition.
    competitors: Option<(Vec<f64>, Vec<u32>)>,
}

/// A path through the survivors: its inputs and the states it visits (`states[0]` the
/// start, `states[t + 1]` the state after step `t`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurvivorPath {
    /// The input of every step.
    pub inputs: Vec<usize>,
    /// The `steps + 1` states.
    pub states: Vec<usize>,
}

impl Survivors {
    /// The tables of `steps` steps of `sm`, the survivors alone.
    pub fn new(sm: &StateMachine, steps: usize) -> Survivors {
        Survivors {
            n_states: sm.n_states,
            n_inputs: sm.n_inputs,
            from: vec![0; steps * sm.n_states],
            competitors: None,
        }
    }

    /// The tables of `steps` steps of `sm`, with the competitors.
    pub fn with_competitors(sm: &StateMachine, steps: usize) -> Survivors {
        let n = steps * sm.n_states;
        Survivors { competitors: Some((vec![f64::INFINITY; n], vec![0; n])), ..Survivors::new(sm, steps) }
    }

    /// Number of steps $T$.
    pub fn steps(&self) -> usize {
        self.from.len() / self.n_states
    }

    /// The transition `s * M + u` that won into state `s2` at step `t`.
    fn winner(&self, t: usize, s2: usize) -> usize {
        self.from[t * self.n_states + s2] as usize
    }

    /// The best competing transition into `s2` at step `t` and the metric it lost by,
    /// `None` without competitors or when none exists.
    fn competitor(&self, t: usize, s2: usize) -> Option<(usize, f64)> {
        let (delta, second) = self.competitors.as_ref()?;
        let d = delta[t * self.n_states + s2];
        d.is_finite().then(|| (second[t * self.n_states + s2] as usize, d))
    }

    /// The survivor that ends in `end`.
    pub fn traceback(&self, end: usize) -> SurvivorPath {
        let steps = self.steps();
        let mut states = vec![0usize; steps + 1];
        let mut inputs = vec![0usize; steps];
        states[steps] = end;
        for t in (0..steps).rev() {
            let su = self.winner(t, states[t + 1]);
            inputs[t] = su % self.n_inputs;
            states[t] = su / self.n_inputs;
        }
        SurvivorPath { inputs, states }
    }
}

/// The add–compare–select recursion over the steps of `surv` from the path metrics
/// `metric` (updated in place to the final metrics). `gamma(t, g)` fills the branch
/// metrics of step `t` into `g[s * M + u]`.
pub fn forward(sm: &StateMachine, metric: &mut [f64], gamma: impl FnMut(usize, &mut [f64]), surv: &mut Survivors) {
    match surv.competitors.take() {
        Some((mut delta, mut second)) => {
            acs::<true>(sm, metric, gamma, &mut surv.from, &mut delta, &mut second);
            surv.competitors = Some((delta, second));
        }
        None => acs::<false>(sm, metric, gamma, &mut surv.from, &mut [], &mut []),
    }
}

/// [`forward`] for one kind of table: the hard path pays nothing for the competitors.
fn acs<const SOFT: bool>(
    sm: &StateMachine,
    metric: &mut [f64],
    mut gamma: impl FnMut(usize, &mut [f64]),
    from: &mut [u32],
    delta: &mut [f64],
    comp: &mut [u32],
) {
    let (ns, ni) = (sm.n_states, sm.n_inputs);
    let mut next = vec![f64::NEG_INFINITY; ns];
    let mut second = vec![f64::NEG_INFINITY; if SOFT { ns } else { 0 }];
    let mut g = vec![0.0; ns * ni];
    for (t, row) in from.chunks_exact_mut(ns).enumerate() {
        gamma(t, &mut g);
        next.fill(f64::NEG_INFINITY);
        if SOFT {
            second.fill(f64::NEG_INFINITY);
        }
        for s in 0..ns {
            if metric[s] == f64::NEG_INFINITY {
                continue;
            }
            for u in 0..ni {
                let s2 = sm.next[s * ni + u];
                let m = metric[s] + g[s * ni + u];
                let branch = (s * ni + u) as u32;
                if m > next[s2] {
                    if SOFT {
                        second[s2] = next[s2];
                        comp[t * ns + s2] = row[s2];
                    }
                    next[s2] = m;
                    row[s2] = branch;
                } else if SOFT && m > second[s2] {
                    second[s2] = m;
                    comp[t * ns + s2] = branch;
                }
            }
        }
        if SOFT {
            for s2 in 0..ns {
                delta[t * ns + s2] = next[s2] - second[s2];
            }
        }
        metric.copy_from_slice(&next);
    }
}

/// The state of the largest path metric (the first of equal metrics).
pub fn best_state(metric: &[f64]) -> usize {
    (0..metric.len()).max_by(|a, b| metric[*a].total_cmp(&metric[*b])).unwrap_or(0)
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
    let mut surv = Survivors::new(sm, steps);
    forward(sm, &mut metric, gamma, &mut surv);
    let end = end.unwrap_or_else(|| best_state(&metric));
    Ok((surv.traceback(end).inputs, metric))
}

/// The soft-output Viterbi algorithm (Hagenauer & Hoeher 1989, see the module): the
/// inputs of the path of largest metric and the reliability $\Delta_{t} \ge 0$ of every
/// one of them (infinite when no competitor ever disagreed), from the initial metrics
/// `start` to `end` or the best final state (whose other final states then compete too).
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
    let mut surv = Survivors::with_competitors(sm, steps);
    forward(sm, &mut metric, gamma, &mut surv);
    let free_end = end.is_none();
    let end = end.unwrap_or_else(|| best_state(&metric));
    let path = surv.traceback(end);
    let mut reliability = vec![f64::INFINITY; steps];
    // a competitor into the state of the path at step t, traced back until it merges
    let mut compete = |t: usize, mut su: usize, d: f64| {
        let mut j = t;
        loop {
            let (s, u) = (su / sm.n_inputs, su % sm.n_inputs);
            if u != path.inputs[j] && d < reliability[j] {
                reliability[j] = d;
            }
            if j == 0 || s == path.states[j] {
                break;
            }
            j -= 1;
            su = surv.winner(j, s);
        }
    };
    for t in 0..steps {
        if let Some((su, d)) = surv.competitor(t, path.states[t + 1]) {
            compete(t, su, d);
        }
    }
    if free_end && steps > 0 {
        for s in (0..sm.n_states).filter(|&s| s != end && metric[s].is_finite()) {
            compete(steps - 1, surv.winner(steps - 1, s), metric[end] - metric[s]);
        }
    }
    Ok((path.inputs, reliability))
}

/// The log-likelihood ratios $\ln[P(u = 0)/P(u = 1)]$ of a binary path from its
/// `inputs` and `reliability` ([`sova`]): $(1 - 2 u_{t})\, \Delta_{t}$, positive for 0.
pub fn binary_llr(inputs: &[usize], reliability: &[f64]) -> Vec<f64> {
    inputs.iter().zip(reliability).map(|(&u, &d)| if u == 0 { d } else { -d }).collect()
}

fn checked_start(sm: &StateMachine, start: &[f64], end: Option<usize>) -> Result<Vec<f64>> {
    ensure_shape(start.len() == sm.n_states, || format!("{} initial metrics for {} states", start.len(), sm.n_states))?;
    ensure_shape(end.is_none_or(|e| e < sm.n_states), || "the final state is out of range".into())?;
    Ok(start.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    fn register() -> StateMachine {
        StateMachine::shift_register(2, 2).unwrap()
    }

    /// The metric and final state of `inputs` from state 0 under the metrics `g`.
    fn score(sm: &StateMachine, g: &[f64], inputs: &[usize]) -> (f64, usize) {
        let (ni, mut s, mut m) = (sm.n_inputs(), 0, 0.0);
        for (t, &u) in inputs.iter().enumerate() {
            m += g[t * sm.n_states() * ni + s * ni + u];
            s = sm.next(s, u);
        }
        (m, s)
    }

    #[test]
    fn the_shift_register_remembers_its_last_inputs() {
        let sm = register();
        assert_eq!((sm.n_states(), sm.n_inputs()), (4, 2));
        // state 0b01 (last input 1, the one before 0), input 1 → 0b11
        assert_eq!(sm.next(1, 1), 3);
        assert_eq!(sm.next(3, 0), 2);
        assert_eq!((sm.digit(2, 1), sm.digit(2, 2)), (0, 1));
        assert!(StateMachine::shift_register(1, 2).is_err());
        assert!(StateMachine::shift_register(2, 40).is_err());
        assert_eq!(StateMachine::shift_register(3, 1).unwrap().next(2, 1), 1);
    }

    #[test]
    fn the_best_path_is_found_by_exhaustive_comparison() {
        // random branch metrics on a 4-state binary trellis: the Viterbi path has the
        // largest metric among the 2^T paths from state 0
        let sm = register();
        let steps = 8;
        let mut rng = Rng::new(12);
        let g: Vec<f64> = (0..steps * 8).map(|_| rng.uniform()).collect();
        let gamma = |t: usize, out: &mut [f64]| out.copy_from_slice(&g[t * 8..(t + 1) * 8]);
        let mut start = vec![f64::NEG_INFINITY; 4];
        start[0] = 0.0;
        let (path, metric) = viterbi(&sm, steps, &start, gamma, None).unwrap();
        let best = (0..1usize << steps)
            .map(|w| (0..steps).map(|t| (w >> t) & 1).collect::<Vec<_>>())
            .map(|p| score(&sm, &g, &p).0)
            .fold(f64::NEG_INFINITY, f64::max);
        let (m, end) = score(&sm, &g, &path);
        assert!((m - best).abs() < 1e-12 && (metric[end] - best).abs() < 1e-12);
        assert!(viterbi(&sm, 3, &[0.0], gamma, None).is_err());
        assert!(sova(&sm, 3, &start, gamma, Some(5)).is_err());
    }

    #[test]
    fn the_soft_output_follows_the_path_and_measures_the_competitors() {
        // a binary register of memory 1 driven by noisy ±1 observations of its input:
        // γ(u) = (1 - 2u) r, the path is the sign of r, and since the metrics are
        // memoryless every competitor flips one decision and merges at once: Δ_t = 2|r_t|
        let sm = StateMachine::shift_register(2, 1).unwrap();
        let r = [1.5, -0.2, 0.8, -1.1, 0.05, 0.9];
        let gamma = |t: usize, g: &mut [f64]| {
            for s in 0..2 {
                g[2 * s] = r[t];
                g[2 * s + 1] = -r[t];
            }
        };
        let (path, rel) = sova(&sm, r.len(), &[0.0, 0.0], gamma, None).unwrap();
        assert_eq!(path, viterbi(&sm, r.len(), &[0.0, 0.0], gamma, None).unwrap().0);
        assert_eq!(path, vec![0, 1, 0, 1, 0, 0]);
        for (t, &v) in r.iter().enumerate() {
            assert!((rel[t] - 2.0 * v.abs()).abs() < 1e-12, "{t}: {} against {}", rel[t], 2.0 * v.abs());
        }
        let llr = binary_llr(&path, &rel);
        assert!(llr.iter().zip(&r).all(|(l, v)| (l - 2.0 * v).abs() < 1e-12));
    }

    #[test]
    fn the_reliabilities_bound_the_max_log_differences_of_every_path() {
        // exhaustive: Δ_t is the metric difference with a competitor that disagrees at t,
        // so it is at least the difference with the best disagreeing path (max-log-MAP)
        let sm = register();
        let steps = 7;
        let mut rng = Rng::new(77);
        for trial in 0..20 {
            let g: Vec<f64> = (0..steps * 8).map(|_| rng.uniform()).collect();
            let gamma = |t: usize, out: &mut [f64]| out.copy_from_slice(&g[t * 8..(t + 1) * 8]);
            let mut start = vec![f64::NEG_INFINITY; 4];
            start[0] = 0.0;
            let end = if trial % 2 == 0 { Some(0) } else { None };
            let (path, rel) = sova(&sm, steps, &start, gamma, end).unwrap();
            let paths: Vec<(Vec<usize>, f64)> = (0..1usize << steps)
                .map(|w| (0..steps).map(|t| (w >> t) & 1).collect::<Vec<_>>())
                .filter_map(|p| {
                    let (m, s) = score(&sm, &g, &p);
                    end.is_none_or(|e| e == s).then_some((p, m))
                })
                .collect();
            let best = score(&sm, &g, &path).0;
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
