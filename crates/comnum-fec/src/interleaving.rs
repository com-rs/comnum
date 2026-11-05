//! Interleavers: permutations of the coded bits (or of their LLRs).
//!
//! An [`Interleaver`] holds a permutation $\pi$: the interleaved sequence is
//! $y_{i} = x_{\pi(i)}$. [`Interleaver::block`] writes row by row and reads column by
//! column; [`Interleaver::random`] draws a uniform permutation from a seed; the
//! quadratic permutation polynomial of LTE turbo codes is
//! [`Interleaver::qpp`], $\pi(i) = (f_{1} i + f_{2} i^{2}) \bmod K$ (Sun & Takeshita 2005).
//!
//! A [`ConvolutionalInterleaver`] spreads a stream instead of a block (Forney 1971): a
//! commutator sends symbol $t$ to branch $j = t \bmod I$, a FIFO of $M j$ cells clocked
//! once every $I$ symbols, so that symbol $t$ leaves $I M j$ symbols late; its
//! deinterleaver has the depths $M (I - 1 - j)$, and every symbol crosses the pair in
//! $I M (I - 1)$ symbols. A burst of $b \le I$ consecutive channel errors lands, at the
//! output of the deinterleaver, in $b$ symbols at least $I M - 1$ apart. DVB-S and DVB-T
//! interleave the bytes of their Reed–Solomon codewords with $I = 12$, $M = 17$ (ETSI
//! EN 300 421, Sec. 4.4.2, Figure 4).
//!
//! # Examples
//! ```
//! use comnum_fec::interleaving::Interleaver;
//! let p = Interleaver::block(2, 3);
//! assert_eq!(p.interleave(&[0, 1, 2, 3, 4, 5]), vec![0, 3, 1, 4, 2, 5]);
//! assert_eq!(p.deinterleave(&p.interleave(&[9, 8, 7, 6, 5, 4])), vec![9, 8, 7, 6, 5, 4]);
//! ```
//!
//! # References
//!
//! 1. J. Sun and O. Y. Takeshita, "Interleavers for turbo codes using permutation
//!    polynomials over integer rings," *IEEE Trans. Inf. Theory*, 51(1), 101–119, 2005.
//! 2. G. D. Forney, "Burst-correcting codes for the classic bursty channel," *IEEE
//!    Trans. Commun. Technol.*, 19(5), 772–781, 1971.
//! 3. ETSI EN 300 421 V1.1.2, *DVB; Framing structure, channel coding and modulation for
//!    11/12 GHz satellite services*, 1997, Sec. 4.4.2.

use comnum_core::rng::Rng;
use comnum_core::{Error, Result, ensure};
use std::collections::VecDeque;

/// A permutation of the positions of a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interleaver {
    /// $\pi$: output position $i$ takes input position `perm[i]`.
    perm: Vec<usize>,
}

impl Interleaver {
    /// The interleaver of the permutation `perm`: output position $i$ takes input
    /// position `perm[i]`.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] if `perm` is not a permutation of $0, \dots, n - 1$.
    ///
    /// # Examples
    /// ```
    /// use comnum_fec::interleaving::Interleaver;
    /// let p = Interleaver::from_permutation(vec![2, 0, 1])?;
    /// assert_eq!(p.interleave(&['a', 'b', 'c']), ['c', 'a', 'b']);
    /// assert!(Interleaver::from_permutation(vec![0, 0, 1]).is_err());
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn from_permutation(perm: Vec<usize>) -> Result<Interleaver> {
        ensure_permutation(&perm, || "the interleaver is not a permutation".into())?;
        Ok(Interleaver { perm })
    }

    /// $\pi$: output position $i$ takes input position `perm()[i]`.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Write row by row into `rows × cols`, read column by column.
    pub fn block(rows: usize, cols: usize) -> Interleaver {
        Interleaver { perm: (0..rows * cols).map(|i| (i % rows) * cols + i / rows).collect() }
    }

    /// A uniform random permutation of `n` positions (seeded).
    pub fn random(n: usize, seed: u64) -> Interleaver {
        let mut perm: Vec<usize> = (0..n).collect();
        Rng::new(seed).shuffle(&mut perm);
        Interleaver { perm }
    }

    /// The quadratic permutation polynomial $\pi(i) = (f_{1} i + f_{2} i^{2}) \bmod K$.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] if the polynomial is not a permutation.
    pub fn qpp(k: usize, f1: usize, f2: usize) -> Result<Interleaver> {
        let perm: Vec<usize> = (0..k).map(|i| ((f1 * i) % k + (f2 * ((i * i) % k)) % k) % k).collect();
        ensure_permutation(&perm, || format!("({f1}, {f2}) is not a permutation polynomial modulo {k}"))?;
        Ok(Interleaver { perm })
    }

    /// The interleaver of a standard by name, for blocks of `k` bits: `lte`, the QPP
    /// interleaver of the LTE turbo code ([`crate::turbo::lte_qpp`], TS 36.212,
    /// Table 5.1.3-3), case-insensitive.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for another name, or a size `k` outside the table.
    pub fn from_name(name: &str, k: usize) -> Result<Interleaver> {
        match name.to_ascii_lowercase().as_str() {
            "lte" => {
                let (f1, f2) = crate::turbo::lte_qpp(k).ok_or_else(|| {
                    Error::Config(format!("K = {k} is not a block size of LTE (TS 36.212, Table 5.1.3-3)"))
                })?;
                Interleaver::qpp(k, f1, f2)
            }
            _ => Err(Error::Config(format!("unknown interleaver {name:?}: lte"))),
        }
    }

    /// Length of the block.
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// $y_{i} = x_{\pi(i)}$.
    ///
    /// # Panics
    /// If `x` is shorter than the permutation.
    pub fn interleave<T: Copy>(&self, x: &[T]) -> Vec<T> {
        self.perm.iter().map(|&p| x[p]).collect()
    }

    /// The inverse permutation.
    ///
    /// # Panics
    /// If `y` is shorter than the permutation.
    pub fn deinterleave<T: Copy + Default>(&self, y: &[T]) -> Vec<T> {
        let mut x = vec![T::default(); self.perm.len()];
        for (i, &p) in self.perm.iter().enumerate() {
            x[p] = y[i];
        }
        x
    }
}

/// [`comnum_core::Error::Config`] with the message of `msg` unless `perm` holds every
/// position below its length once.
fn ensure_permutation(perm: &[usize], msg: impl FnOnce() -> String) -> Result<()> {
    let mut seen = vec![false; perm.len()];
    for &p in perm {
        if p >= perm.len() || seen[p] {
            return ensure(false, msg);
        }
        seen[p] = true;
    }
    Ok(())
}

/// A convolutional (Forney) interleaver or deinterleaver of $I$ branches of unit depth
/// $M$ (see the module): a stream processed symbol by symbol, its FIFOs holding the
/// symbols still in flight.
///
/// Branch $j$ of the interleaver holds $M j$ cells; branch $j$ of the deinterleaver
/// holds $M (I - 1 - j)$ cells. Both start filled with `T::default()`; the commutator
/// starts on branch 0, so that the first symbol of a stream (the sync byte of DVB) takes
/// the branch without delay. Symbol $t$ leaves the interleaver $I M (t \bmod I)$ symbols
/// late, and the pair delays every symbol by [`ConvolutionalInterleaver::latency`].
///
/// # Examples
/// ```
/// use comnum_fec::interleaving::ConvolutionalInterleaver;
/// // the interleaver of DVB-S and DVB-T: I = 12, M = 17
/// let mut tx = ConvolutionalInterleaver::new(12, 17)?;
/// let mut rx = ConvolutionalInterleaver::deinterleaver(12, 17)?;
/// let data: Vec<u8> = (0..5000).map(|i| (i % 251) as u8).collect();
/// let y = rx.process(&tx.process(&data));
/// let lat = tx.latency();                          // 12 · 17 · 11 = 2244 symbols
/// assert_eq!(&y[lat..], &data[..data.len() - lat]);
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct ConvolutionalInterleaver<T> {
    unit: usize,
    deinterleave: bool,
    /// The FIFO of every branch (empty for a branch without delay).
    fifos: Vec<VecDeque<T>>,
    /// The branch of the next symbol.
    branch: usize,
}

impl<T: Copy + Default> ConvolutionalInterleaver<T> {
    /// The interleaver of `branches` branches of unit depth `unit` ($I$, $M$).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] without a branch.
    pub fn new(branches: usize, unit: usize) -> Result<Self> {
        Self::build(branches, unit, false)
    }

    /// The deinterleaver of [`ConvolutionalInterleaver::new`] with the same parameters.
    ///
    /// # Errors
    /// As [`ConvolutionalInterleaver::new`].
    pub fn deinterleaver(branches: usize, unit: usize) -> Result<Self> {
        Self::build(branches, unit, true)
    }

    fn build(branches: usize, unit: usize, deinterleave: bool) -> Result<Self> {
        ensure(branches >= 1, || "a convolutional interleaver needs at least one branch".into())?;
        let depth = |j: usize| unit * if deinterleave { branches - 1 - j } else { j };
        let fifos = (0..branches).map(|j| std::iter::repeat_n(T::default(), depth(j)).collect()).collect();
        Ok(ConvolutionalInterleaver { unit, deinterleave, fifos, branch: 0 })
    }

    /// The number of branches $I$.
    pub fn branches(&self) -> usize {
        self.fifos.len()
    }

    /// The unit depth $M$.
    pub fn unit_depth(&self) -> usize {
        self.unit
    }

    /// Whether this is the deinterleaver.
    pub fn is_deinterleaver(&self) -> bool {
        self.deinterleave
    }

    /// The delay (in symbols) of the symbols of branch `branch`: $I M j$ for the
    /// interleaver, $I M (I - 1 - j)$ for the deinterleaver.
    pub fn delay(&self, branch: usize) -> usize {
        self.branches() * self.fifos[branch % self.branches()].len()
    }

    /// The delay of the interleaver and deinterleaver together, $I M (I - 1)$ symbols.
    pub fn latency(&self) -> usize {
        let i = self.branches();
        i * self.unit * (i - 1)
    }

    /// Back to the initial state: the FIFOs filled with `T::default()`, the commutator on
    /// branch 0.
    pub fn reset(&mut self) {
        for f in &mut self.fifos {
            for v in f.iter_mut() {
                *v = T::default();
            }
        }
        self.branch = 0;
    }

    /// One symbol in, one symbol out.
    pub fn push(&mut self, x: T) -> T {
        let n = self.fifos.len();
        let fifo = &mut self.fifos[self.branch];
        self.branch = (self.branch + 1) % n;
        match fifo.pop_front() {
            Some(y) => {
                fifo.push_back(x);
                y
            }
            None => x,
        }
    }

    /// The symbols of `x` through the interleaver, into `out` (as many).
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] if `out` does not have the length of `x`.
    pub fn process_into(&mut self, x: &[T], out: &mut [T]) -> Result<()> {
        comnum_core::same_len(out.len(), x.len(), "interleaved symbols")?;
        for (o, &v) in out.iter_mut().zip(x) {
            *o = self.push(v);
        }
        Ok(())
    }

    /// The symbols of `x` through the interleaver.
    pub fn process(&mut self, x: &[T]) -> Vec<T> {
        x.iter().map(|&v| self.push(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lte_interleaver_is_read_by_name() {
        assert_eq!(Interleaver::from_name("LTE", 40).unwrap(), Interleaver::qpp(40, 3, 10).unwrap());
        assert!(Interleaver::from_name("lte", 41).is_err());
        assert!(Interleaver::from_name("umts", 40).is_err());
    }

    #[test]
    fn permutations_invert() {
        let x: Vec<u32> = (0..40).collect();
        for p in [Interleaver::random(40, 3), Interleaver::block(5, 8), Interleaver::qpp(40, 3, 10).unwrap()] {
            let mut sorted = p.perm.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, x.iter().map(|v| *v as usize).collect::<Vec<_>>());
            assert_eq!(p.deinterleave(&p.interleave(&x)), x);
        }
        // the QPP of LTE for K = 40 is (3, 10); (2, 10) is not a permutation
        assert!(Interleaver::qpp(40, 2, 10).is_err());
    }
    #[test]
    fn a_convolutional_interleaver_delays_branch_j_by_i_m_j_symbols() {
        // EN 300 421 Sec. 4.4.2: branch j holds M j cells, clocked once every I symbols
        let mut tx = ConvolutionalInterleaver::<u32>::new(12, 17).unwrap();
        let x: Vec<u32> = (1..=20_000).collect();
        let y = tx.process(&x);
        for t in 0..x.len() {
            let d = 12 * 17 * (t % 12);
            if t + d < y.len() {
                assert_eq!(y[t + d], x[t], "symbol {t} of branch {}", t % 12);
            }
        }
        assert_eq!((tx.delay(0), tx.delay(11)), (0, 12 * 17 * 11));
    }

    #[test]
    fn the_deinterleaver_undoes_the_interleaver_after_its_latency() {
        for (i, m) in [(12, 17), (5, 3), (1, 4)] {
            let mut tx = ConvolutionalInterleaver::new(i, m).unwrap();
            let mut rx = ConvolutionalInterleaver::deinterleaver(i, m).unwrap();
            let x: Vec<u16> = (0..3000).map(|v| (v * 7 + 1) as u16).collect();
            let y = rx.process(&tx.process(&x));
            let lat = tx.latency();
            assert_eq!(lat, i * m * (i - 1));
            assert_eq!(&y[lat..], &x[..x.len() - lat], "I = {i}, M = {m}");
            assert!(y[..lat].iter().all(|&v| v == 0), "the initial content");
        }
        assert!(ConvolutionalInterleaver::<u8>::new(0, 3).is_err());
    }

    #[test]
    fn a_burst_is_spread_over_symbols_i_m_apart() {
        // I consecutive channel errors land in I symbols of the stream, I M - 1 apart at least
        let (i, m) = (12, 17);
        let mut tx = ConvolutionalInterleaver::new(i, m).unwrap();
        let mut rx = ConvolutionalInterleaver::deinterleaver(i, m).unwrap();
        let mut y = tx.process(&vec![0u8; 20_000]);
        y[9_000..9_000 + i].iter_mut().for_each(|v| *v = 1);
        let out = rx.process(&y);
        let hit: Vec<usize> = (0..out.len()).filter(|&t| out[t] == 1).collect();
        assert_eq!(hit.len(), i);
        let gaps: Vec<usize> = hit.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|&g| g >= i * m - 1), "{gaps:?}");
    }

    #[test]
    fn reset_restores_the_initial_state() {
        let mut tx = ConvolutionalInterleaver::new(3, 2).unwrap();
        let a = tx.process(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9]);
        tx.reset();
        assert_eq!(tx.process(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9]), a);
    }
}
