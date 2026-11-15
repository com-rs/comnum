//! Orthogonal $M$-ary frequency-shift keying with a noncoherent receiver.
//!
//! # Model
//!
//! The symbol $a \in \{0, \dots, M - 1\}$ is a tone held for $N$ samples (`sps`), at
//! the frequency of $a \Delta$ cycles per symbol, $\Delta$ the `spacing`, a whole
//! number:
//!
//! $$
//! x\[n\] = e^{j 2\pi a \Delta n / N}, \qquad 0 \le n < N.
//! $$
//!
//! Every tone makes a whole number of cycles in a symbol, so the phase is continuous
//! from one symbol to the next and two tones are orthogonal over a symbol whatever
//! their phases: the receiver needs neither the phase nor the exact frequency of the
//! carrier. It correlates every symbol with the $M$ tones and keeps the largest
//! magnitude ([`Fsk::energies`], [`Fsk::demodulate`]). With white noise of variance
//! $\sigma^{2}$ per sample, $E_{s}/N_{0} = N/\sigma^{2}$ and the error rate is that of
//! $M$ orthogonal signals detected noncoherently
//! ([`comnum_mapping::theory::ber_orthogonal_noncoherent`]): $\tfrac{1}{2}
//! e^{-E_{b}/(2 N_{0})}$ for $M = 2$. The signal occupies about $M \Delta$ times the
//! symbol rate, from 0 upwards (the tones above half the sample rate are the negative
//! frequencies): $M \Delta \le N$.
//!
//! The receiver must know where the symbols start. [`Fsk::timing`] finds the offset,
//! among the $N$ of a symbol, that maximises the energy of the decisions: a window
//! that straddles two symbols splits its energy between two tones. Every symbol
//! starting at the phase 0, a window one sample late loses nothing: the offset is
//! known to within one sample.
//!
//! # Examples
//! ```
//! use comnum_waveform::fsk::Fsk;
//! let fsk = Fsk::new(4, 16, 2)?;
//! let x = fsk.modulate(&[0, 3, 1, 2, 2])?;
//! assert_eq!(x.len(), 80);
//! assert_eq!(fsk.demodulate(&x)?, [0, 3, 1, 2, 2]);
//! // the symbols found in a signal that starts five samples late, to within one sample
//! let late = [vec![comnum_math::C64::new(0.0, 0.0); 5], x].concat();
//! assert!([5, 6].contains(&fsk.timing(&late)?));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed. McGraw-Hill, 2008,
//!    Sec. 4.5 (noncoherent detection of orthogonal signals).

use crate::{check_input, check_rate};
use comnum_core::signal::{Labels, Signal};
use comnum_core::traits::{Demodulate, Modulate};
use comnum_core::units::Hertz;
use comnum_core::{Error, Result, ensure};
use comnum_math::{C64, argmax, cis};
use std::f64::consts::TAU;

/// An orthogonal $M$-FSK modem with a noncoherent receiver (see the module): a
/// configuration with public fields, checked by [`Fsk::check`] when it is used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fsk {
    /// The number of tones $M$.
    pub m: usize,
    /// Samples per symbol $N$.
    pub sps: usize,
    /// The spacing $\Delta$ of two adjacent tones, in cycles per symbol.
    pub spacing: usize,
    /// The sample rate (Hz; 1 by default, time in samples).
    pub sample_rate: Hertz,
}

impl Fsk {
    /// The modem of `m` tones `spacing` cycles per symbol apart, at `sps` samples per
    /// symbol.
    ///
    /// # Errors
    /// As [`Fsk::check`].
    pub fn new(m: usize, sps: usize, spacing: usize) -> Result<Fsk> {
        let fsk = Fsk { m, sps, spacing, sample_rate: Hertz(1.0) };
        fsk.check()?;
        Ok(fsk)
    }

    /// Check the configuration.
    ///
    /// # Errors
    /// [`Error::Config`] for fewer than two tones, a spacing of zero, or tones that do
    /// not fit in the sampled band ($M \Delta > N$).
    pub fn check(&self) -> Result<()> {
        ensure(self.m >= 2 && self.spacing >= 1, || {
            format!("at least two tones one cycle per symbol apart, got {} tones {} apart", self.m, self.spacing)
        })?;
        ensure(self.m * self.spacing <= self.sps, || {
            format!(
                "{} tones {} cycles apart need {} samples per symbol, got {}",
                self.m,
                self.spacing,
                self.m * self.spacing,
                self.sps
            )
        })?;
        check_rate(self.sample_rate)
    }

    /// The unit-envelope samples of a sequence of symbols, $N$ per symbol.
    ///
    /// # Errors
    /// As [`Fsk::check`]; [`Error::Config`] for a symbol outside $0 \dots M - 1$.
    pub fn modulate(&self, symbols: &[usize]) -> Result<Vec<C64>> {
        self.check()?;
        ensure(symbols.iter().all(|&a| a < self.m), || format!("the symbols lie in 0..{}", self.m))?;
        let table = self.table();
        let mut out = Vec::with_capacity(symbols.len() * self.sps);
        for &a in symbols {
            out.extend((0..self.sps).map(|n| table[a * self.spacing * n % self.sps]));
        }
        Ok(out)
    }

    /// The samples $e^{j 2\pi k / N}$, $k < N$, of which every tone is a decimation.
    fn table(&self) -> Vec<C64> {
        (0..self.sps).map(|k| cis(TAU * k as f64 / self.sps as f64)).collect()
    }

    /// The energies of the $M$ tones in the `sps` samples of `block`.
    fn correlate<'a>(&'a self, table: &'a [C64], block: &'a [C64]) -> impl Iterator<Item = f64> + 'a {
        (0..self.m).map(move |a| {
            let step = a * self.spacing;
            block.iter().enumerate().map(|(n, v)| v * table[step * n % self.sps].conj()).sum::<C64>().norm_sqr()
        })
    }

    /// The energy of the $M$ candidates of every whole symbol of `r`, the first symbol
    /// at its first sample: the squared magnitudes of the correlations with the tones
    /// (the samples after the last whole symbol are left out).
    ///
    /// # Errors
    /// As [`Fsk::check`].
    pub fn energies(&self, r: &[C64]) -> Result<Vec<f64>> {
        self.check()?;
        let table = self.table();
        Ok(r.chunks_exact(self.sps).flat_map(|block| self.correlate(&table, block).collect::<Vec<_>>()).collect())
    }

    /// The noncoherent decisions: the tone of largest energy in every whole symbol.
    ///
    /// # Errors
    /// As [`Fsk::check`].
    pub fn demodulate(&self, r: &[C64]) -> Result<Vec<usize>> {
        Ok(self.energies(r)?.chunks_exact(self.m).map(|e| argmax(e.iter().copied()).map_or(0, |(a, _)| a)).collect())
    }

    /// The offset in $0 \dots N - 1$ at which the symbols of `r` start: the one that
    /// maximises the sum over the symbols of the energy of the decided tone, to within
    /// one sample (see the module). Give it the part of a recording that holds the signal: the symbols of
    /// noise alone only blur the maximum.
    ///
    /// # Errors
    /// As [`Fsk::check`]; [`Error::Shape`] for fewer samples than two symbols.
    pub fn timing(&self, r: &[C64]) -> Result<usize> {
        self.check()?;
        if r.len() < 2 * self.sps {
            return Err(Error::Shape(format!("{} samples are less than two symbols of {}", r.len(), self.sps)));
        }
        let table = self.table();
        let metric = |offset: usize| -> f64 {
            r[offset..].chunks_exact(self.sps).map(|block| self.correlate(&table, block).fold(0.0, f64::max)).sum()
        };
        // every offset is given the same number of symbols
        let symbols = (r.len() - self.sps + 1) / self.sps;
        let trimmed = |offset: usize| {
            metric(offset) - if (r.len() - offset) / self.sps > symbols { self.last(r, offset, &table) } else { 0.0 }
        };
        Ok(argmax((0..self.sps).map(trimmed)).map_or(0, |(offset, _)| offset))
    }

    /// The energy of the decision of the last whole symbol of `r` from `offset`.
    fn last(&self, r: &[C64], offset: usize, table: &[C64]) -> f64 {
        r[offset..].chunks_exact(self.sps).last().map_or(0.0, |block| self.correlate(table, block).fold(0.0, f64::max))
    }
}

/// The transmitter as a [`Modulate`] block on one stream of labels.
impl Modulate<Labels> for Fsk {
    /// # Errors
    /// [`Error::Config`] for several streams, and the errors of [`Fsk::modulate`].
    fn modulate_into(&self, x: &Labels, out: &mut Signal) -> Result<()> {
        ensure(x.n_streams() == 1, || format!("FSK modulates one stream, not {}", x.n_streams()))?;
        out.set_stream(&Fsk::modulate(self, x.stream(0))?, self.sample_rate);
        Ok(())
    }
}

/// The noncoherent receiver as a [`Demodulate`] block on one stream of samples at
/// [`Fsk::sample_rate`], the first symbol at the first sample.
impl Demodulate<Labels> for Fsk {
    /// # Errors
    /// [`Error::Config`] for several streams or a signal at another sample rate, and
    /// the errors of [`Fsk::demodulate`].
    fn demodulate_into(&mut self, y: &Signal, out: &mut Labels) -> Result<()> {
        check_input(y, self.sample_rate, "FSK")?;
        out.set_stream(&Fsk::demodulate(self, y.stream(0))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;
    use comnum_mapping::theory::ber_orthogonal_noncoherent;

    #[test]
    fn the_tones_are_orthogonal_with_a_continuous_phase() {
        let fsk = Fsk::new(8, 32, 3).unwrap();
        let symbols: Vec<usize> = (0..8).collect();
        let x = fsk.modulate(&symbols).unwrap();
        assert!(x.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
        // every symbol starts at the phase 0: no jump at the boundaries
        assert!(x.iter().step_by(32).all(|v| (v - 1.0).norm() < 1e-12));
        for (i, e) in fsk.energies(&x).unwrap().chunks_exact(8).enumerate() {
            for (a, v) in e.iter().enumerate() {
                assert!((v - if a == i { 1024.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
        // a phase and a small frequency offset of the carrier do not change the decisions
        let rotated: Vec<C64> = x.iter().enumerate().map(|(n, v)| v * cis(1.0 + 0.01 * n as f64)).collect();
        assert_eq!(fsk.demodulate(&rotated).unwrap(), symbols);
    }

    #[test]
    fn the_simulated_error_rate_is_that_of_orthogonal_noncoherent_signals() {
        let mut rng = Rng::new(5);
        for (m, ebn0_db) in [(2usize, 8.0), (4, 6.0)] {
            let fsk = Fsk::new(m, 16, 1).unwrap();
            let bits_per_symbol = (m as f64).log2();
            let esn0_db = ebn0_db + comnum_math::lin2db(bits_per_symbol);
            let symbols: Vec<usize> = (0..200_000).map(|_| rng.below(m)).collect();
            let mut r = fsk.modulate(&symbols).unwrap();
            let noise = rng.complex_noise(r.len(), 16.0 / comnum_math::db2lin(esn0_db));
            r.iter_mut().zip(&noise).for_each(|(v, w)| *v += w);
            let decided = fsk.demodulate(&r).unwrap();
            let errors: u32 = decided.iter().zip(&symbols).map(|(a, b)| (a ^ b).count_ones()).sum();
            let ber = f64::from(errors) / (200_000.0 * bits_per_symbol);
            let theory = ber_orthogonal_noncoherent(m, esn0_db).unwrap();
            assert!((ber / theory - 1.0).abs() < 0.08, "M = {m}: {ber} against {theory}");
        }
    }

    #[test]
    fn the_timing_is_found_in_noise_at_every_offset() {
        let mut rng = Rng::new(6);
        let fsk = Fsk::new(4, 24, 2).unwrap();
        let symbols: Vec<usize> = (0..300).map(|_| rng.below(4)).collect();
        let x = fsk.modulate(&symbols).unwrap();
        for offset in [0usize, 1, 7, 12, 23] {
            let mut r = rng.complex_noise(offset + x.len() + 11, 1.0);
            x.iter().enumerate().for_each(|(n, v)| r[offset + n] += v);
            // one sample late costs nothing (the first sample of every symbol is 1)
            let found = fsk.timing(&r).unwrap();
            assert!(found == offset || found == (offset + 1) % 24, "{found} for {offset}");
            let decided = fsk.demodulate(&r[offset.max(found)..]).unwrap();
            assert_eq!(decided[..299], symbols[..299]);
        }
        assert!(fsk.timing(&x[..40]).is_err());
    }

    #[test]
    fn the_configuration_is_checked_and_the_blocks_carry_the_labels() {
        assert!(Fsk::new(1, 8, 1).is_err() && Fsk::new(4, 8, 0).is_err() && Fsk::new(4, 7, 2).is_err());
        let mut fsk = Fsk::new(2, 8, 4).unwrap();
        assert!(fsk.modulate(&[2]).is_err());
        let labels = Labels::from_stream(vec![1, 0, 1]).unwrap();
        let y = Modulate::modulate(&fsk, &labels).unwrap();
        assert_eq!(y.len(), 24);
        let back: Labels = Demodulate::demodulate(&mut fsk, &y).unwrap();
        assert_eq!(back.stream(0), [1, 0, 1]);
    }
}
