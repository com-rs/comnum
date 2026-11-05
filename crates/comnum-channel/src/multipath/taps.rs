//! The realised taps of a tapped delay line: [`Taps`].

use comnum_core::prelude::*;
use comnum_core::same_len;
use comnum_math::cis;
use std::f64::consts::PI;

/// The channel taps of a realisation: for every tap its delay (samples) and its
/// coefficients, `[tap][sample][rx][tx]` flattened.
///
/// A time-invariant realisation ([`Taps::time_invariant`], a [`Tdl`](super::Tdl) without Doppler, a
/// static [`crate::cdl::Rays`]) stores one `n_rx × n_tx` matrix per tap instead of one per
/// sample: the accessors give the same coefficient at every sample. Equality compares
/// the coefficients, whatever the storage.
#[derive(Clone, Debug)]
pub struct Taps {
    pub(super) delays: Vec<usize>,
    pub(super) n_rx: usize,
    pub(super) n_tx: usize,
    pub(super) n_samples: usize,
    // `[sample][rx][tx]` per tap, or `[rx][tx]` when `invariant`
    pub(super) coeffs: Vec<Vec<C64>>,
    pub(super) invariant: bool,
}

impl PartialEq for Taps {
    fn eq(&self, other: &Taps) -> bool {
        let same_shape = self.delays == other.delays
            && (self.n_rx, self.n_tx, self.n_samples) == (other.n_rx, other.n_tx, other.n_samples);
        if !same_shape {
            return false;
        }
        if self.invariant == other.invariant {
            return self.coeffs == other.coeffs;
        }
        (0..self.delays.len()).all(|l| {
            (0..self.n_samples).all(|n| {
                (0..self.n_rx)
                    .all(|r| (0..self.n_tx).all(|t| self.coefficient(l, n, r, t) == other.coefficient(l, n, r, t)))
            })
        })
    }
}

impl Taps {
    /// The taps of delays `delays` (samples) between `n_tx` transmit and `n_rx` receive
    /// antennas, `coeffs[l]` holding the `n_samples × n_rx × n_tx` coefficients of tap
    /// `l`, `[sample][rx][tx]` flattened.
    ///
    /// # Errors
    /// [`Error::Shape`] if `coeffs` has not one entry per delay, or an entry of the
    /// wrong length.
    pub fn new(delays: Vec<usize>, n_rx: usize, n_tx: usize, n_samples: usize, coeffs: Vec<Vec<C64>>) -> Result<Taps> {
        same_len(coeffs.len(), delays.len(), "coefficients and delays")?;
        for c in &coeffs {
            same_len(c.len(), n_samples * n_rx * n_tx, "coefficients of a tap and samples × antennas")?;
        }
        Ok(Taps { delays, n_rx, n_tx, n_samples, coeffs, invariant: false })
    }

    /// The taps of a channel constant over its `n_samples` samples: `coeffs[l]` holds the
    /// `n_rx × n_tx` coefficients of tap `l`, `[rx][tx]` flattened, stored once
    /// ($n_{\mathrm{samples}}$ times less memory than [`Taps::new`]).
    ///
    /// # Errors
    /// [`Error::Shape`] if `coeffs` has not one entry per delay, or an entry of the
    /// wrong length.
    ///
    /// # Examples
    /// ```
    /// use comnum_channel::multipath::Taps;
    /// use comnum_math::C64;
    /// let h = Taps::time_invariant(vec![0, 2], 1, 1, 1000, vec![vec![C64::new(1.0, 0.0)], vec![C64::new(0.0, 0.5)]])?;
    /// assert!(h.is_time_invariant());
    /// assert_eq!(h.coefficient(1, 999, 0, 0), C64::new(0.0, 0.5));
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn time_invariant(
        delays: Vec<usize>,
        n_rx: usize,
        n_tx: usize,
        n_samples: usize,
        coeffs: Vec<Vec<C64>>,
    ) -> Result<Taps> {
        same_len(coeffs.len(), delays.len(), "coefficients and delays")?;
        for c in &coeffs {
            same_len(c.len(), n_rx * n_tx, "coefficients of a tap and antennas")?;
        }
        Ok(Taps { delays, n_rx, n_tx, n_samples, coeffs, invariant: true })
    }

    /// Whether the coefficients are stored once for every sample (a channel constant
    /// over the realisation, [`Taps::time_invariant`]).
    pub fn is_time_invariant(&self) -> bool {
        self.invariant
    }

    /// Delays of the taps (samples).
    pub fn delays(&self) -> &[usize] {
        &self.delays
    }

    /// Number of samples of the realisation.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Receive antennas.
    pub fn n_rx(&self) -> usize {
        self.n_rx
    }

    /// Transmit antennas.
    pub fn n_tx(&self) -> usize {
        self.n_tx
    }

    /// The `len × len` matrix of a single-antenna realisation on the samples
    /// `start .. start + len`: $H\[i, j\] = \sum_{l} h_{l}\[s + i\]\, \delta_{i - j, d_{l}}$, the
    /// samples before `start` being zero (the time-variant convolution matrix).
    ///
    /// # Errors
    /// [`Error::Shape`] for a MIMO realisation or samples beyond the realisation.
    pub fn matrix(&self, start: usize, len: usize) -> Result<comnum_math::linalg::CMat> {
        self.block_matrix(start, len, false)
    }

    /// [`Taps::matrix`] of a block preceded by a cyclic prefix at least as long as the
    /// largest delay: the delays wrap around the block,
    /// $H\[i, (i - d_{l}) \bmod L\] \mathrel{+}= h_{l}\[s + i\]$ (the channel of a block after
    /// the removal of its prefix, `start` its first sample).
    ///
    /// # Errors
    /// As [`Taps::matrix`].
    pub fn cyclic_matrix(&self, start: usize, len: usize) -> Result<comnum_math::linalg::CMat> {
        self.block_matrix(start, len, true)
    }

    fn block_matrix(&self, start: usize, len: usize, cyclic: bool) -> Result<comnum_math::linalg::CMat> {
        comnum_core::ensure_shape(self.n_rx == 1 && self.n_tx == 1, || {
            format!("a matrix of a single-antenna realisation, not {} × {}", self.n_rx, self.n_tx)
        })?;
        comnum_core::ensure_shape(start + len <= self.n_samples, || {
            format!("samples {start}..{} of a realisation of {}", start + len, self.n_samples)
        })?;
        let mut h = comnum_math::linalg::CMat::zeros((len, len));
        for (l, &d) in self.delays.iter().enumerate() {
            for i in 0..len {
                let j = if cyclic {
                    (i + len - d % len) % len
                } else if i >= d {
                    i - d
                } else {
                    continue;
                };
                h[[i, j]] += self.coefficient(l, start + i, 0, 0);
            }
        }
        Ok(h)
    }

    /// Apply the taps to `x` (`n_tx` streams of `n_samples` samples), the samples
    /// before the signal being zero: the `n_rx` received streams,
    /// $y_{r}\[n\] = \sum_{l} \sum_{t} h_{l, r, t}\[n\]\, x_{t}\[n - d_{l}\]$.
    ///
    /// # Errors
    /// [`Error::Shape`] if `x` does not have `n_tx` streams of `n_samples` samples.
    pub fn apply(&self, x: &[Vec<C64>]) -> Result<Vec<Vec<C64>>> {
        same_len(x.len(), self.n_tx, "streams and transmit antennas")?;
        for xs in x {
            same_len(xs.len(), self.n_samples, "signal and taps")?;
        }
        let n = self.n_samples;
        let mut out = vec![vec![C64::new(0.0, 0.0); n]; self.n_rx];
        for (r, row) in out.iter_mut().enumerate() {
            for (l, &d) in self.delays.iter().enumerate() {
                for (t, xs) in x.iter().enumerate() {
                    if self.invariant {
                        let h = self.coeffs[l][r * self.n_tx + t];
                        for i in d..n {
                            row[i] += h * xs[i - d];
                        }
                    } else {
                        for i in d..n {
                            row[i] += self.coefficient(l, i, r, t) * xs[i - d];
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Coefficient of tap `l` at sample `n` between transmit antenna `t` and receive
    /// antenna `r`.
    ///
    /// # Panics
    /// If `l` is not a tap or `n` not a sample of the realisation.
    pub fn coefficient(&self, l: usize, n: usize, r: usize, t: usize) -> C64 {
        if self.invariant {
            assert!(n < self.n_samples, "sample {n} of a realisation of {}", self.n_samples);
            self.coeffs[l][r * self.n_tx + t]
        } else {
            self.coeffs[l][(n * self.n_rx + r) * self.n_tx + t]
        }
    }

    /// The frequency response $H_{k} = \sum_{l} h_{l}\[n\] e^{-j 2\pi k d_{l} / N}$ at sample
    /// `n` for the FFT bins `bins` of a size-`fft_len` transform: `[bin][rx][tx]`
    /// flattened.
    pub fn frequency_response(&self, n: usize, bins: &[usize], fft_len: usize) -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); bins.len() * self.n_rx * self.n_tx];
        for (b, &k) in bins.iter().enumerate() {
            for (l, &d) in self.delays.iter().enumerate() {
                let e = cis(-2.0 * PI * (k * d) as f64 / fft_len as f64);
                for r in 0..self.n_rx {
                    for t in 0..self.n_tx {
                        out[(b * self.n_rx + r) * self.n_tx + t] += self.coefficient(l, n, r, t) * e;
                    }
                }
            }
        }
        out
    }
}
