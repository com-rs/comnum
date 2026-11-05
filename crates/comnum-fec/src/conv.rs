//! Convolutional codes: trellis, Viterbi and BCJR decoders.
//!
//! A [`Trellis`] describes a rate-$1/n$ code with $2^{K-1}$ states: from every state, an
//! input bit leads to a next state and emits $n$ coded bits. Two families are built:
//!
//! - feed-forward codes ([`Trellis::feedforward`]) from octal generators, the most
//!   significant bit tapping the current input — the $(133, 171)_{8}$ code of constraint
//!   length 7 ([`ConvolutionalCode::k7`]) is the classical one (802.11, DVB-S, LTE);
//! - recursive systematic codes ([`Trellis::recursive`]), the constituents of turbo
//!   codes, with a feedback and a feed-forward polynomial (LTE: $(13, 15)_{8}$).
//!
//! # Decoders
//!
//! - [`viterbi`]: maximum-likelihood sequence detection; with LLR inputs, the branch
//!   metric is the correlation $\sum_{i} (1 - 2 c_{i}) L_{i} / 2$ (soft decisions), with
//!   hard decisions it is the Hamming distance — about 2 dB apart (Viterbi 1967;
//!   Forney 1973);
//! - [`viterbi_tail_biting`]: a tail-biting code, whose path ends in the state it starts
//!   from ([`Trellis::encode_tail_biting`], the code of the PBCH of LTE), by the
//!   wrap-around Viterbi algorithm of Shao, Lin and Fossorier (2003);
//! - [`bcjr`]: the a posteriori LLR of every input bit (Bahl, Cocke, Jelinek & Raviv
//!   1974), in the log domain with the exact Jacobian logarithm (log-MAP) or its
//!   maximum (max-log-MAP).
//!
//! # Examples
//! ```
//! use comnum_fec::block::BlockCode;
//! use comnum_fec::conv::ConvolutionalCode;
//! let code = ConvolutionalCode::k7(100)?;
//! let info: Vec<u8> = (0..100).map(|i| ((i * 7) % 3 == 0) as u8).collect();
//! let c = code.encode_block(&info)?;
//! assert_eq!(c.len(), 2 * (100 + 6));                           // terminated
//! let llr: Vec<f64> = c.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect(); // noiseless
//! assert_eq!(code.decode_block(&llr)?, info);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum
//!    decoding algorithm," *IEEE Trans. Inf. Theory*, 13(2), 260–269, 1967.
//! 2. G. D. Forney, "The Viterbi algorithm," *Proc. IEEE*, 61(3), 268–278, 1973.
//! 3. L. R. Bahl, J. Cocke, F. Jelinek and J. Raviv, "Optimal decoding of linear codes
//!    for minimizing symbol error rate," *IEEE Trans. Inf. Theory*, 20(2), 284–287, 1974.
//! 4. P. Robertson, E. Villebrun and P. Hoeher, "A comparison of optimal and sub-optimal
//!    MAP decoding algorithms operating in the log domain," *Proc. IEEE ICC*, 1995.
//! 5. R. Y. Shao, S. Lin and M. P. C. Fossorier, "Two decoding algorithms for tailbiting
//!    codes," *IEEE Trans. Commun.* 51(10), 1658–1665, 2003 (tail biting, the WAVA).

use crate::block::{BlockCode, Jacobian, SoftDecode, container_traits, expect_len, max_star};
use comnum_core::{Result, ensure, ensure_shape};

fn parity(x: u32) -> u8 {
    (x.count_ones() & 1) as u8
}

/// The trellis of a rate-$1/n$ binary convolutional code, built by
/// [`Trellis::feedforward`] or [`Trellis::recursive`] (its fields are read through
/// getters, so that the tables always agree with the constraint length and the number
/// of outputs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trellis {
    /// Constraint length $K$ (memory $K - 1$).
    pub(crate) constraint: u32,
    /// Coded bits per input bit $n$.
    pub(crate) n_out: usize,
    /// `next[s][u]`: the state after input `u` in state `s`.
    pub(crate) next: Vec<[usize; 2]>,
    /// `out[s][u]`: the coded bits, first output in the most significant position.
    pub(crate) out: Vec<[u32; 2]>,
    /// `tail[s]`: the input that drives state `s` towards zero (0 for feed-forward codes).
    pub(crate) tail: Vec<u8>,
}

impl Trellis {
    /// Number of states $2^{K-1}$.
    pub fn n_states(&self) -> usize {
        self.next.len()
    }

    /// Constraint length $K$ (memory $K - 1$).
    pub fn constraint(&self) -> u32 {
        self.constraint
    }

    /// Coded bits per input bit $n$ (at least 1).
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// `next()[s][u]`: the state after input `u` in state `s`.
    pub fn next(&self) -> &[[usize; 2]] {
        &self.next
    }

    /// `out()[s][u]`: the coded bits of the branch, the first output in the most
    /// significant position.
    pub fn out(&self) -> &[[u32; 2]] {
        &self.out
    }

    /// `tail()[s]`: the input that drives state `s` towards zero (0 for feed-forward
    /// codes).
    pub fn tail(&self) -> &[u8] {
        &self.tail
    }

    /// The feed-forward code of constraint length `constraint` with the generators
    /// `generators` (octal digits written as integers, e.g. `0o133`).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a constraint length outside 1 to 24 or no generator.
    pub fn feedforward(constraint: u32, generators: &[u32]) -> Result<Trellis> {
        let m = Self::memory(constraint)?;
        ensure(!generators.is_empty(), || "a convolutional code needs at least one generator".into())?;
        let states = 1usize << m;
        let mut next = Vec::with_capacity(states);
        let mut out = Vec::with_capacity(states);
        for s in 0..states as u32 {
            let mut nx = [0usize; 2];
            let mut o = [0u32; 2];
            for u in 0..2u32 {
                let reg = (u << m) | s;
                nx[u as usize] = (reg >> 1) as usize;
                o[u as usize] = generators.iter().fold(0, |acc, g| (acc << 1) | u32::from(parity(reg & g)));
            }
            next.push(nx);
            out.push(o);
        }
        Ok(Trellis { constraint, n_out: generators.len(), next, out, tail: vec![0; states] })
    }

    /// The memory `constraint - 1` of a code, checked.
    fn memory(constraint: u32) -> Result<u32> {
        ensure((1..=24).contains(&constraint), || format!("the constraint length must be 1 to 24, got {constraint}"))?;
        Ok(constraint - 1)
    }

    /// The recursive systematic code with the feedback polynomial `feedback` and the
    /// feed-forward polynomial `forward` (octal, e.g. `0o13`, `0o15`): outputs the input
    /// bit, then the parity bit.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a constraint length outside 1 to 24.
    pub fn recursive(constraint: u32, feedback: u32, forward: u32) -> Result<Trellis> {
        Trellis::recursive_parities(constraint, feedback, &[forward])
    }

    /// The recursive systematic code with the feedback polynomial `feedback` and several
    /// feed-forward polynomials `forwards` (octal, the most significant bit the current
    /// register input $w_{t} = u_{t} \oplus \dots$): outputs the input bit, then one parity
    /// bit per polynomial, in their order ([`Trellis::recursive`] is the case of one
    /// polynomial); the constituent codes of the turbo codes of CCSDS 131.0-B (Sec. 6.3)
    /// have up to three.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a constraint length outside 1 to 24, no
    /// polynomial, or more than 31 outputs.
    ///
    /// # Examples
    /// ```
    /// use comnum_fec::conv::Trellis;
    /// // the 16-state code of CCSDS rate 1/6, encoder a: G0 = 10011, G1, G2, G3
    /// let t = Trellis::recursive_parities(5, 0o23, &[0o33, 0o25, 0o37])?;
    /// assert_eq!((t.n_states(), t.n_out()), (16, 4));
    /// assert_eq!(Trellis::recursive_parities(4, 0o13, &[0o15])?, Trellis::recursive(4, 0o13, 0o15)?);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn recursive_parities(constraint: u32, feedback: u32, forwards: &[u32]) -> Result<Trellis> {
        let m = Self::memory(constraint)?;
        ensure(!forwards.is_empty() && forwards.len() < 31, || {
            format!("a recursive code needs 1 to 30 feed-forward polynomials, got {}", forwards.len())
        })?;
        let states = 1usize << m;
        let fb = feedback & ((1 << m) - 1);
        let mut next = Vec::with_capacity(states);
        let mut out = Vec::with_capacity(states);
        let mut tail = Vec::with_capacity(states);
        for s in 0..states as u32 {
            let mut nx = [0usize; 2];
            let mut o = [0u32; 2];
            for u in 0..2u32 {
                let w = u ^ u32::from(parity(s & fb));
                let reg = (w << m) | s;
                nx[u as usize] = (reg >> 1) as usize;
                o[u as usize] = forwards.iter().fold(u, |acc, &g| (acc << 1) | u32::from(parity(reg & g)));
            }
            next.push(nx);
            out.push(o);
            tail.push(parity(s & fb));
        }
        Ok(Trellis { constraint, n_out: forwards.len() + 1, next, out, tail })
    }

    /// Encode `bits` from state 0; with `terminate`, append the $K - 1$ tail inputs that
    /// return to state 0. Returns the coded bits.
    pub fn encode(&self, bits: &[u8], terminate: bool) -> Vec<u8> {
        let tail = if terminate { self.constraint as usize - 1 } else { 0 };
        let mut out = Vec::with_capacity((bits.len() + tail) * self.n_out);
        let mut s = self.encode_from(0, bits, &mut out);
        for _ in 0..tail {
            let u = self.tail[s];
            s = self.encode_from(s, &[u], &mut out);
        }
        out
    }

    /// Encode `bits` with tail biting: the register starts in the state that its last
    /// $K - 1$ inputs leave it in, so that the path ends in the state it starts from and
    /// no tail is sent (the code of LTE, TS 36.212 Sec. 5.1.3.1). Returns the $n$ coded
    /// bits of every input bit.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a recursive code (its state depends on more than
    /// the last $K - 1$ inputs); [`comnum_core::Error::Shape`] for fewer than $K - 1$ bits.
    ///
    /// # Examples
    /// ```
    /// use comnum_fec::conv::{Trellis, viterbi_tail_biting};
    /// let t = Trellis::feedforward(7, &[0o133, 0o171, 0o165])?;   // the code of LTE
    /// let bits: Vec<u8> = (0..40).map(|i| ((i * 5) % 7 < 3) as u8).collect();
    /// let c = t.encode_tail_biting(&bits)?;
    /// assert_eq!(c.len(), 3 * 40);                                 // no tail
    /// let llr: Vec<f64> = c.iter().map(|&b| if b == 0 { 2.0 } else { -2.0 }).collect();
    /// assert_eq!(viterbi_tail_biting(&t, &llr, 4)?, bits);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn encode_tail_biting(&self, bits: &[u8]) -> Result<Vec<u8>> {
        self.check_feedforward()?;
        let m = self.constraint as usize - 1;
        ensure_shape(bits.len() >= m, || format!("tail biting needs at least K - 1 = {m} bits, got {}", bits.len()))?;
        let start = bits[bits.len() - m..].iter().fold(0, |s, &b| self.next[s][usize::from(b & 1)]);
        let mut out = Vec::with_capacity(bits.len() * self.n_out);
        self.encode_from(start, bits, &mut out);
        Ok(out)
    }

    /// Encode `bits` from the state `s` into `out`; returns the final state.
    fn encode_from(&self, mut s: usize, bits: &[u8], out: &mut Vec<u8>) -> usize {
        for &b in bits {
            let u = usize::from(b & 1);
            let o = self.out[s][u];
            for i in (0..self.n_out).rev() {
                out.push(((o >> i) & 1) as u8);
            }
            s = self.next[s][u];
        }
        s
    }

    /// Whether the next state is the register shifted by the input (a feed-forward code).
    fn check_feedforward(&self) -> Result<()> {
        let m = self.constraint - 1;
        let ff = self.next.iter().enumerate().all(|(s, nx)| (0..2).all(|u| nx[u] == ((u << m) | s) >> 1));
        ensure(ff, || "tail biting is defined here for feed-forward codes only".into())
    }

    /// The log-likelihood of a branch, $\sum_{i} \pm L_{i}/2$.
    fn branch(&self, o: u32, llr: &[f64]) -> f64 {
        llr.iter().enumerate().map(|(i, l)| if (o >> (self.n_out - 1 - i)) & 1 == 0 { l / 2.0 } else { -l / 2.0 }).sum()
    }

    /// The log-likelihoods of the branches of one step into `metrics[s][u]` (flat,
    /// `2 s + u`): through the $2^{n}$ output words when they are fewer than the
    /// branches, each computed once.
    fn branches(&self, llr: &[f64], words: &mut Vec<f64>, metrics: &mut [f64]) {
        let n_words = if self.n_out < 16 { 1usize << self.n_out } else { usize::MAX };
        if n_words <= metrics.len() {
            words.clear();
            words.extend((0..n_words as u32).map(|o| self.branch(o, llr)));
            for (m, o) in metrics.iter_mut().zip(self.out.iter().flatten()) {
                *m = words[*o as usize];
            }
        } else {
            for (m, o) in metrics.iter_mut().zip(self.out.iter().flatten()) {
                *m = self.branch(*o, llr);
            }
        }
    }
}

/// Viterbi decoding of `steps = llr.len() / n` trellis steps from state 0; with
/// `terminated`, the path ends in state 0 and the last $K - 1$ inputs (the tail) are
/// dropped. Returns the decided input bits.
///
/// # Errors
/// [`comnum_core::Error::Shape`] if the number of LLRs is not a multiple of $n$.
pub fn viterbi(trellis: &Trellis, llr: &[f64], terminated: bool) -> Result<Vec<u8>> {
    let steps = steps_of(trellis, llr)?;
    let ns = trellis.n_states();
    let mut metric = vec![f64::NEG_INFINITY; ns];
    metric[0] = 0.0;
    let mut from = vec![0u32; steps * ns];
    forward(trellis, llr, &mut metric, &mut from);
    let end = if terminated { 0 } else { best_state(&metric) };
    let (mut bits, _) = traceback(&from, ns, end);
    if terminated {
        bits.truncate(steps.saturating_sub(trellis.constraint as usize - 1));
    }
    Ok(bits)
}

/// Viterbi decoding of a tail-biting code ([`Trellis::encode_tail_biting`]) by the
/// wrap-around Viterbi algorithm (WAVA) of Shao, Lin and Fossorier (2003): a pass starts
/// from every state with the metrics that the previous pass ended with (all zero for
/// the first), and the decoding stops at the first pass whose best path starts and ends
/// in the same state; after `passes` passes, the best path among those that start and
/// end in the same state is kept (the best path if there is none). Returns the decided
/// input bits, one per step.
///
/// # Errors
/// [`comnum_core::Error::Shape`] if the number of LLRs is not a multiple of $n$;
/// [`comnum_core::Error::Config`] for `passes = 0`.
///
/// # References
/// R. Y. Shao, S. Lin and M. P. C. Fossorier, "Two decoding algorithms for tailbiting
/// codes," *IEEE Trans. Commun.* 51(10), 1658–1665, 2003, doi:10.1109/TCOMM.2003.818084
/// (Sec. III, the WAVA).
pub fn viterbi_tail_biting(trellis: &Trellis, llr: &[f64], passes: usize) -> Result<Vec<u8>> {
    ensure(passes > 0, || "the wrap-around Viterbi algorithm needs at least one pass".into())?;
    let steps = steps_of(trellis, llr)?;
    let ns = trellis.n_states();
    let mut metric = vec![0.0; ns];
    let mut from = vec![0u32; steps * ns];
    for _ in 1..passes {
        forward(trellis, llr, &mut metric, &mut from);
        let end = best_state(&metric);
        let (bits, first) = traceback(&from, ns, end);
        if first == end {
            return Ok(bits);
        }
        let top = metric[end];
        for m in &mut metric {
            *m -= top;
        }
    }
    let start = metric.clone();
    forward(trellis, llr, &mut metric, &mut from);
    let end = best_state(&metric);
    let (bits, first) = traceback(&from, ns, end);
    if first == end {
        return Ok(bits);
    }
    // the best tail-biting survivor of the last pass: the metric it gained in the pass
    let best = (0..ns)
        .map(|s| (s, traceback(&from, ns, s)))
        .filter(|(s, (_, first))| first == s)
        .max_by(|(a, (_, fa)), (b, (_, fb))| (metric[*a] - start[*fa]).total_cmp(&(metric[*b] - start[*fb])));
    Ok(best.map_or(bits, |(_, (b, _))| b))
}

/// The trellis steps of `llr`, checked.
fn steps_of(trellis: &Trellis, llr: &[f64]) -> Result<usize> {
    let n = trellis.n_out;
    ensure_shape(llr.len() % n == 0, || format!("{} LLRs is not a multiple of {n}", llr.len()))?;
    Ok(llr.len() / n)
}

/// The add-compare-select recursion over the steps of `llr`, from the path metrics
/// `metric` (updated in place); `from[t * ns + s]` receives the survivor of every step
/// and state, its predecessor and input `2 s + u` (the states number at most 2^23).
fn forward(trellis: &Trellis, llr: &[f64], metric: &mut Vec<f64>, from: &mut [u32]) {
    let (n, ns) = (trellis.n_out, trellis.n_states());
    let mut next = vec![f64::NEG_INFINITY; ns];
    let (mut words, mut gamma) = (Vec::new(), vec![0.0; 2 * ns]);
    for (t, row) in from.chunks_exact_mut(ns).enumerate() {
        trellis.branches(&llr[t * n..(t + 1) * n], &mut words, &mut gamma);
        next.fill(f64::NEG_INFINITY);
        for s in 0..ns {
            if metric[s] == f64::NEG_INFINITY {
                continue;
            }
            for u in 0..2 {
                let ns2 = trellis.next[s][u];
                let m = metric[s] + gamma[2 * s + u];
                if m > next[ns2] {
                    next[ns2] = m;
                    row[ns2] = (2 * s + u) as u32;
                }
            }
        }
        std::mem::swap(metric, &mut next);
    }
}

/// The state of the largest path metric.
fn best_state(metric: &[f64]) -> usize {
    (0..metric.len()).max_by(|a, b| metric[*a].total_cmp(&metric[*b])).unwrap_or(0)
}

/// The inputs of the survivor that ends in `end`, and the state it starts from.
fn traceback(from: &[u32], ns: usize, end: usize) -> (Vec<u8>, usize) {
    let steps = from.len() / ns;
    let mut s = end;
    let mut bits = vec![0u8; steps];
    for t in (0..steps).rev() {
        let su = from[t * ns + s] as usize;
        bits[t] = (su & 1) as u8;
        s = su >> 1;
    }
    (bits, s)
}

/// The BCJR algorithm in the log domain: the a posteriori LLR of every input bit,
/// given the channel LLRs of the coded bits (`n` per step) and the a priori LLRs of the
/// inputs (`apriori`, zero if empty). `jacobian` selects log-MAP ([`Jacobian::Exact`])
/// or max-log-MAP ([`Jacobian::MaxLog`]); `terminated` forces the final state to zero.
///
/// # Errors
/// [`comnum_core::Error::Shape`] for inconsistent lengths.
pub fn bcjr(trellis: &Trellis, llr: &[f64], apriori: &[f64], terminated: bool, jacobian: Jacobian) -> Result<Vec<f64>> {
    let mut app = Vec::new();
    bcjr_into(trellis, llr, apriori, terminated, jacobian, &mut BcjrWorkspace::default(), &mut app)?;
    Ok(app)
}

/// The arrays of [`bcjr`], kept between the calls of an iterative decoder.
#[derive(Default)]
pub(crate) struct BcjrWorkspace {
    /// $\gamma_{t}(s, u)$ at `(t ns + s) 2 + u`.
    gamma: Vec<f64>,
    /// $\alpha_{t}(s)$, then $\beta_{t}(s)$, at `t ns + s`, $t = 0, \dots, T$.
    alpha: Vec<f64>,
    beta: Vec<f64>,
    words: Vec<f64>,
}

/// [`bcjr`] into `app`, with the arrays of `ws`.
pub(crate) fn bcjr_into(
    trellis: &Trellis,
    llr: &[f64],
    apriori: &[f64],
    terminated: bool,
    jacobian: Jacobian,
    ws: &mut BcjrWorkspace,
    app: &mut Vec<f64>,
) -> Result<()> {
    let n = trellis.n_out;
    if llr.len() % n != 0 || (!apriori.is_empty() && apriori.len() != llr.len() / n) {
        return Err(comnum_core::Error::Shape("inconsistent channel and a priori LLRs".into()));
    }
    let steps = llr.len() / n;
    let ns = trellis.n_states();
    let ninf = f64::NEG_INFINITY;
    // the branch metrics, once per step and branch
    ws.gamma.resize(steps * 2 * ns, 0.0);
    for (t, g) in ws.gamma.chunks_exact_mut(2 * ns).enumerate() {
        trellis.branches(&llr[t * n..(t + 1) * n], &mut ws.words, g);
        let la = apriori.get(t).copied().unwrap_or(0.0);
        for (i, g) in g.iter_mut().enumerate() {
            *g += if i % 2 == 0 { la / 2.0 } else { -la / 2.0 };
        }
    }
    let gamma = &ws.gamma;
    let alpha = &mut ws.alpha;
    alpha.clear();
    alpha.resize((steps + 1) * ns, ninf);
    alpha[0] = 0.0;
    for t in 0..steps {
        let (now, later) = alpha[t * ns..(t + 2) * ns].split_at_mut(ns);
        for s in 0..ns {
            if now[s] == ninf {
                continue;
            }
            for u in 0..2 {
                let s2 = trellis.next[s][u];
                later[s2] = max_star(later[s2], now[s] + gamma[(t * ns + s) * 2 + u], jacobian);
            }
        }
    }
    let beta = &mut ws.beta;
    beta.clear();
    beta.resize((steps + 1) * ns, ninf);
    if terminated {
        beta[steps * ns] = 0.0;
    } else {
        beta[steps * ns..].fill(0.0);
    }
    for t in (0..steps).rev() {
        let (now, later) = beta[t * ns..(t + 2) * ns].split_at_mut(ns);
        for s in 0..ns {
            let mut b = ninf;
            for u in 0..2 {
                b = max_star(b, gamma[(t * ns + s) * 2 + u] + later[trellis.next[s][u]], jacobian);
            }
            now[s] = b;
        }
    }
    app.clear();
    for t in 0..steps {
        let mut l = [ninf; 2];
        for s in 0..ns {
            let a = alpha[t * ns + s];
            if a == ninf {
                continue;
            }
            for u in 0..2 {
                l[u] =
                    max_star(l[u], a + gamma[(t * ns + s) * 2 + u] + beta[(t + 1) * ns + trellis.next[s][u]], jacobian);
            }
        }
        app.push(l[0] - l[1]);
    }
    Ok(())
}

/// A terminated rate-$1/n$ convolutional code over blocks of `k` information bits,
/// decoded by soft-decision Viterbi.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvolutionalCode {
    /// The trellis.
    trellis: Trellis,
    /// Information bits per block.
    k: usize,
    /// Decode hard decisions (the signs of the LLRs) instead of the LLRs.
    pub hard_decision: bool,
}

impl ConvolutionalCode {
    /// The code of `trellis` terminated on blocks of `k` bits, decoded by soft-decision
    /// Viterbi.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for $k = 0$.
    pub fn new(trellis: Trellis, k: usize) -> Result<ConvolutionalCode> {
        ensure(k >= 1, || "a convolutional code needs at least one information bit per block".into())?;
        Ok(ConvolutionalCode { trellis, k, hard_decision: false })
    }

    /// The $(133, 171)_{8}$ code of constraint length 7, rate 1/2, on blocks of `k` bits.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for $k = 0$.
    pub fn k7(k: usize) -> Result<ConvolutionalCode> {
        let trellis = Trellis::feedforward(7, &[0o133, 0o171]).unwrap_or_else(|_| unreachable!("a valid code"));
        ConvolutionalCode::new(trellis, k)
    }

    /// The trellis.
    pub fn trellis(&self) -> &Trellis {
        &self.trellis
    }
}

impl BlockCode for ConvolutionalCode {
    fn k(&self) -> usize {
        self.k
    }

    fn n(&self) -> usize {
        self.trellis.n_out * (self.k + self.trellis.constraint as usize - 1)
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k, "bits")?;
        Ok(self.trellis.encode(info, true))
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        if self.hard_decision {
            // a zero LLR (a punctured bit) keeps a zero metric
            let h: Vec<f64> = llr
                .iter()
                .map(|l| {
                    if *l < 0.0 {
                        -1.0
                    } else if *l > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            viterbi(&self.trellis, &h, true)
        } else {
            viterbi(&self.trellis, llr, true)
        }
    }
}

impl SoftDecode for ConvolutionalCode {
    /// The a posteriori LLRs of the information bits by [`bcjr`] (log-MAP, the tail
    /// forced to state zero, its inputs without a priori); one iteration.
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        expect_len(out.len(), self.k, "a posteriori LLRs")?;
        let la: Vec<f64> = match apriori {
            None => Vec::new(),
            Some(a) => {
                expect_len(a.len(), self.k, "a priori LLRs")?;
                a.iter().copied().chain(std::iter::repeat_n(0.0, self.trellis.constraint as usize - 1)).collect()
            }
        };
        let app = bcjr(&self.trellis, llr, &la, true, Jacobian::Exact)?;
        out.copy_from_slice(&app[..self.k]);
        Ok(1)
    }
}

container_traits!(ConvolutionalCode);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::testing::awgn_llr;
    use comnum_core::rng::Rng;
    use comnum_core::signal::hard_decisions;

    #[test]
    fn the_k7_code_has_free_distance_10() {
        // the weight of the impulse response is the free distance of (133, 171)
        let t = Trellis::feedforward(7, &[0o133, 0o171]).unwrap();
        let c = t.encode(&[1, 0, 0, 0, 0, 0, 0], true);
        assert_eq!(c.iter().map(|b| u32::from(*b)).sum::<u32>(), 10);
    }

    #[test]
    fn viterbi_corrects_errors_and_bcjr_agrees() {
        let code = ConvolutionalCode::k7(200).unwrap();
        let mut r = Rng::new(1);
        let info: Vec<u8> = (0..200).map(|_| r.bit()).collect();
        let c = code.encode_block(&info).unwrap();
        let y = awgn_llr(&c, 0.6, &mut r);
        assert!(hard_decisions(&y).iter().zip(&c).filter(|(a, b)| a != b).count() > 5);
        assert_eq!(code.decode_block(&y).unwrap(), info);
        let app = bcjr(&code.trellis, &y, &[], true, Jacobian::Exact).unwrap();
        assert_eq!(hard_decisions(&app[..200]), info);
        let maxlog = bcjr(&code.trellis, &y, &[], true, Jacobian::MaxLog).unwrap();
        assert_eq!(hard_decisions(&maxlog[..200]), info);
    }

    #[test]
    fn branch_tables_and_reused_workspaces_change_nothing() {
        let mut r = Rng::new(5);
        // a table of the 2^n output words (k7, the turbo constituent) and the branches
        // computed one by one (five outputs, four states)
        let codes = [
            Trellis::feedforward(7, &[0o133, 0o171]).unwrap(),
            Trellis::recursive(4, 0o13, 0o15).unwrap(),
            Trellis::feedforward(3, &[5, 7, 7, 5, 6]).unwrap(),
        ];
        let (mut ws, mut app) = (BcjrWorkspace::default(), Vec::new());
        for t in &codes {
            let (ns, n) = (t.n_states(), t.n_out);
            let l: Vec<f64> = (0..n).map(|_| r.gaussian()).collect();
            let (mut words, mut m) = (Vec::new(), vec![0.0; 2 * ns]);
            t.branches(&l, &mut words, &mut m);
            for s in 0..ns {
                for u in 0..2 {
                    assert_eq!(m[2 * s + u].to_bits(), t.branch(t.out[s][u], &l).to_bits());
                }
            }
            for len in [30, 7, 50] {
                let bits: Vec<u8> = (0..len).map(|_| r.bit()).collect();
                let c = t.encode(&bits, true);
                let clean: Vec<f64> = c.iter().map(|b| 4.0 * (1.0 - 2.0 * f64::from(*b))).collect();
                assert_eq!(viterbi(t, &clean, true).unwrap(), bits);
                let y: Vec<f64> = clean.iter().map(|v| v + 2.0 * r.gaussian()).collect();
                let la: Vec<f64> = (0..y.len() / n).map(|_| r.gaussian()).collect();
                for (terminated, jacobian) in [(true, Jacobian::Exact), (false, Jacobian::MaxLog)] {
                    bcjr_into(t, &y, &la, terminated, jacobian, &mut ws, &mut app).unwrap();
                    let fresh = bcjr(t, &y, &la, terminated, jacobian).unwrap();
                    assert!(
                        app.iter().zip(&fresh).all(|(a, b)| a.to_bits() == b.to_bits()) && app.len() == fresh.len()
                    );
                }
            }
        }
    }

    #[test]
    fn the_soft_output_is_the_bcjr_of_the_information_bits() {
        let code = ConvolutionalCode::k7(60).unwrap();
        let mut r = Rng::new(3);
        let info: Vec<u8> = (0..60).map(|_| r.bit()).collect();
        let y = awgn_llr(&code.encode_block(&info).unwrap(), 0.8, &mut r);
        let la: Vec<f64> = (0..60).map(|_| r.gaussian()).collect();
        let former = bcjr(&code.trellis, &y, &[], true, Jacobian::Exact).unwrap();
        let app = code.app(&y, None).unwrap();
        assert!(app.iter().zip(&former[..60]).all(|(a, b)| a.to_bits() == b.to_bits()));
        let padded: Vec<f64> = la.iter().copied().chain([0.0; 6]).collect();
        let former = bcjr(&code.trellis, &y, &padded, true, Jacobian::Exact).unwrap();
        let app = code.app(&y, Some(&la)).unwrap();
        assert!(app.iter().zip(&former[..60]).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(code.app(&y[1..], None).is_err() && code.app(&y, Some(&la[1..])).is_err());
        assert!(ConvolutionalCode::k7(0).is_err());
    }

    fn tail_biting_ml(t: &Trellis, llr: &[f64]) -> Vec<u8> {
        // maximum likelihood: the best of the Viterbi paths forced to start and end in s
        let ns = t.n_states();
        let steps = llr.len() / t.n_out();
        let mut best = (f64::NEG_INFINITY, Vec::new());
        for s0 in 0..ns {
            let mut metric = vec![f64::NEG_INFINITY; ns];
            metric[s0] = 0.0;
            let mut from = vec![0u32; steps * ns];
            forward(t, llr, &mut metric, &mut from);
            if metric[s0] > best.0 {
                best = (metric[s0], traceback(&from, ns, s0).0);
            }
        }
        best.1
    }

    #[test]
    fn the_wava_decodes_the_lte_code_as_the_maximum_likelihood_decoder() {
        let t = Trellis::feedforward(7, &[0o133, 0o171, 0o165]).unwrap();
        let mut rng = comnum_core::rng::Rng::new(36_212);
        let (mut agree, mut errors, trials) = (0, 0, 300);
        for _ in 0..trials {
            let bits: Vec<u8> = (0..40).map(|_| (rng.uniform() < 0.5) as u8).collect();
            let c = t.encode_tail_biting(&bits).unwrap();
            // BPSK at Es/N0 = -2 dB (rate 1/3: Eb/N0 = 2.8 dB): LLR = 4 y Es/N0
            let es_n0 = 10f64.powf(-0.2);
            let llr: Vec<f64> = c
                .iter()
                .map(|&b| {
                    let y = 1.0 - 2.0 * f64::from(b) + rng.gaussian() / (2.0 * es_n0).sqrt();
                    4.0 * es_n0 * y
                })
                .collect();
            let wava = viterbi_tail_biting(&t, &llr, 4).unwrap();
            agree += usize::from(wava == tail_biting_ml(&t, &llr));
            errors += usize::from(wava != bits);
        }
        assert!(agree >= trials - 3, "{agree} of {trials}");
        assert!(errors < trials / 10, "{errors} block errors of {trials}");
    }

    #[test]
    fn tail_biting_starts_and_ends_in_the_state_of_the_last_inputs() {
        let t = Trellis::feedforward(7, &[0o133, 0o171, 0o165]).unwrap();
        let bits = [1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let c = t.encode_tail_biting(&bits).unwrap();
        // the same as encoding from state 0 the last six bits, then all of them
        let mut warm = bits[4..].to_vec();
        warm.extend_from_slice(&bits);
        assert_eq!(c, t.encode(&warm, false)[3 * 6..]);
        assert!(t.encode_tail_biting(&bits[..5]).is_err());
        assert!(Trellis::recursive(4, 0o13, 0o15).unwrap().encode_tail_biting(&bits).is_err());
        assert!(viterbi_tail_biting(&t, &[0.0; 4], 2).is_err());
        assert!(viterbi_tail_biting(&t, &[0.0; 3], 0).is_err());
    }

    #[test]
    fn recursive_codes_terminate_and_are_systematic() {
        let t = Trellis::recursive(4, 0o13, 0o15).unwrap();
        let bits = [1, 0, 1, 1, 0, 0, 1, 0];
        let c = t.encode(&bits, true);
        assert_eq!(c.len(), 2 * (8 + 3));
        // systematic: every other bit is the input
        assert_eq!((0..8).map(|i| c[2 * i]).collect::<Vec<_>>(), bits);
        // the tail returns to state 0
        let mut s = 0;
        for &b in &bits {
            s = t.next[s][usize::from(b)];
        }
        for _ in 0..3 {
            s = t.next[s][usize::from(t.tail[s])];
        }
        assert_eq!(s, 0);
    }
}
