//! Frequency-hopping spread spectrum with noncoherent $M$-ary FSK (FH/MFSK).
//!
//! # Model
//!
//! The band is cut into $Q$ channels of $M$ orthogonal tones each. A symbol $a \in
//! \{0, \dots, M - 1\}$ sent while the link sits on the channel $q$ is the tone of
//! index $qM + a$ among the $N = QM$ tones of the band, over one symbol period of $N$
//! samples:
//!
//! $$
//! x\[n\] = e^{j 2\pi (qM + a) n / N}, \qquad 0 \le n < N,
//! $$
//!
//! and the channel changes every `symbols_per_hop` symbols along the hopping pattern,
//! known to both ends and repeated ([`Fhss::modulate`]; slow hopping). The receiver
//! follows the pattern: the DFT of the $N$ samples, of which it keeps the $M$ bins of
//! the current channel ([`Fhss::energies`]), and the largest wins
//! ([`Fhss::demodulate`]). The tone has the magnitude $N$ in its bin and white noise of
//! variance $\sigma^{2}$ per sample the variance $N\sigma^{2}$ in every bin, so
//! $E_{s}/N_{0} = N/\sigma^{2}$ and the error rate over white noise is that of $M$
//! orthogonal signals detected noncoherently
//! ([`comnum_mapping::theory::ber_orthogonal_noncoherent`]), whatever $Q$: hopping
//! changes nothing there.
//!
//! What it buys is that an interferer must cover $Q$ times the band of the message. A
//! jammer of power $J$ spread over the whole band is noise of the density $N_{J}$ with
//! $E_{b}/N_{J} = N / (J \log_{2} M)$, a processing gain of $Q$ against the same power
//! in one channel; one that concentrates it on a fraction $\rho$ of the band
//! (`comnum_channel::jamming::partial_band_noise`) hits a hop with the probability
//! $\rho$ and is the subject of [`theory`].
//!
//! # Examples
//! ```
//! use comnum_core::rng::Rng;
//! use comnum_waveform::fhss::Fhss;
//! // 16 channels of binary FSK, four symbols per hop
//! let pattern = Fhss::random_pattern(16, 100, &mut Rng::new(1));
//! let link = Fhss::new(16, 2, 4, pattern)?;
//! let symbols = [0, 1, 1, 0, 1, 0, 0, 0, 1];
//! let x = link.modulate(&symbols)?;
//! assert_eq!(x.len(), 9 * 32);
//! assert_eq!(link.demodulate(&x)?, symbols);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed. McGraw-Hill, 2008,
//!    Sec. 12.3 (frequency-hopped spread spectrum signals).
//! 2. A. J. Viterbi and I. M. Jacobs, "Advances in coding and modulation for
//!    noncoherent channels affected by fading, partial band, and multiple-access
//!    interference," in *Advances in Communication Systems*, vol. 4. New York: Academic
//!    Press, 1975, pp. 279–308.

pub mod theory;

use crate::{check_input, check_rate};
use comnum_core::rng::SimRng;
use comnum_core::signal::{Labels, Signal};
use comnum_core::traits::{Demodulate, Modulate};
use comnum_core::units::Hertz;
use comnum_core::{Error, Result, ensure};
use comnum_math::fft::fft;
use comnum_math::{C64, argmax, cis};
use std::f64::consts::TAU;

/// A frequency-hopping link with noncoherent $M$-FSK (see the module): a configuration
/// with public fields, checked by [`Fhss::check`] when it is used.
#[derive(Clone, Debug, PartialEq)]
pub struct Fhss {
    /// The number of channels $Q$ of the band.
    pub n_channels: usize,
    /// The number of tones $M$ of a channel, the size of the alphabet.
    pub m: usize,
    /// Symbols sent on a channel before the next hop.
    pub symbols_per_hop: usize,
    /// The hopping pattern, one channel per hop, repeated.
    pub pattern: Vec<usize>,
    /// The sample rate (Hz; 1 by default, time in samples): the whole band.
    pub sample_rate: Hertz,
}

impl Fhss {
    /// The link of `n_channels` channels of `m` tones that hops every
    /// `symbols_per_hop` symbols along `pattern`.
    ///
    /// # Errors
    /// As [`Fhss::check`].
    pub fn new(n_channels: usize, m: usize, symbols_per_hop: usize, pattern: Vec<usize>) -> Result<Fhss> {
        let link = Fhss { n_channels, m, symbols_per_hop, pattern, sample_rate: Hertz(1.0) };
        link.check()?;
        Ok(link)
    }

    /// A pattern of `len` hops drawn uniformly among `n_channels` channels.
    pub fn random_pattern(n_channels: usize, len: usize, rng: &mut SimRng) -> Vec<usize> {
        (0..len).map(|_| rng.below(n_channels)).collect()
    }

    /// Check the configuration.
    ///
    /// # Errors
    /// [`Error::Config`] without a channel, for fewer than two tones, no symbol per
    /// hop, an empty pattern or one that names a channel outside the band.
    pub fn check(&self) -> Result<()> {
        ensure(self.n_channels > 0 && self.m >= 2, || {
            format!("at least one channel of two tones, got {} of {}", self.n_channels, self.m)
        })?;
        ensure(self.symbols_per_hop > 0, || "at least one symbol per hop".to_string())?;
        ensure(!self.pattern.is_empty() && self.pattern.iter().all(|&q| q < self.n_channels), || {
            format!("a hopping pattern over the channels 0 to {}", self.n_channels - 1)
        })?;
        check_rate(self.sample_rate)
    }

    /// Samples per symbol, $N = QM$: the number of tones of the band.
    pub fn symbol_len(&self) -> usize {
        self.n_channels * self.m
    }

    /// The channel of the symbol of index `i`.
    pub fn hop(&self, i: usize) -> usize {
        self.pattern[(i / self.symbols_per_hop) % self.pattern.len()]
    }

    /// The unit-envelope samples of a sequence of symbols, $N$ per symbol, the first
    /// symbol at the start of the pattern.
    ///
    /// # Errors
    /// As [`Fhss::check`]; [`Error::Config`] for a symbol outside $0 \dots M - 1$.
    pub fn modulate(&self, symbols: &[usize]) -> Result<Vec<C64>> {
        self.check()?;
        ensure(symbols.iter().all(|&a| a < self.m), || format!("the symbols lie in 0..{}", self.m))?;
        let n = self.symbol_len();
        let tones: Vec<C64> = (0..n).map(|k| cis(TAU * k as f64 / n as f64)).collect();
        let mut out = Vec::with_capacity(symbols.len() * n);
        for (i, &a) in symbols.iter().enumerate() {
            let bin = self.hop(i) * self.m + a;
            out.extend((0..n).map(|t| tones[bin * t % n]));
        }
        Ok(out)
    }

    /// The energy of the $M$ candidates of every received symbol: the squared
    /// magnitudes of the DFT in the bins of the channel of the pattern.
    ///
    /// # Errors
    /// As [`Fhss::check`]; [`Error::Shape`] unless `r` holds whole symbols.
    pub fn energies(&self, r: &[C64]) -> Result<Vec<f64>> {
        self.check()?;
        let n = self.symbol_len();
        if r.len() % n != 0 {
            return Err(Error::Shape(format!("{} samples are not whole symbols of {n}", r.len())));
        }
        let mut out = Vec::with_capacity(r.len() / n * self.m);
        for (i, block) in r.chunks_exact(n).enumerate() {
            let first = self.hop(i) * self.m;
            out.extend(fft(block)[first..first + self.m].iter().map(C64::norm_sqr));
        }
        Ok(out)
    }

    /// The noncoherent decisions: the tone of largest energy in the channel of every
    /// symbol.
    ///
    /// # Errors
    /// As [`Fhss::energies`].
    pub fn demodulate(&self, r: &[C64]) -> Result<Vec<usize>> {
        Ok(self.energies(r)?.chunks_exact(self.m).map(|e| argmax(e.iter().copied()).map_or(0, |(a, _)| a)).collect())
    }
}

/// The transmitter as a [`Modulate`] block on one stream of labels.
impl Modulate<Labels> for Fhss {
    /// # Errors
    /// [`Error::Config`] for several streams, and the errors of [`Fhss::modulate`].
    fn modulate_into(&self, x: &Labels, out: &mut Signal) -> Result<()> {
        ensure(x.n_streams() == 1, || format!("FHSS modulates one stream, not {}", x.n_streams()))?;
        out.set_stream(&Fhss::modulate(self, x.stream(0))?, self.sample_rate);
        Ok(())
    }
}

/// The noncoherent receiver as a [`Demodulate`] block on one stream of samples at
/// [`Fhss::sample_rate`].
impl Demodulate<Labels> for Fhss {
    /// # Errors
    /// [`Error::Config`] for several streams or a signal at another sample rate, and
    /// the errors of [`Fhss::demodulate`].
    fn demodulate_into(&mut self, y: &Signal, out: &mut Labels) -> Result<()> {
        check_input(y, self.sample_rate, "FHSS")?;
        out.set_stream(&Fhss::demodulate(self, y.stream(0))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    #[test]
    fn the_tones_are_orthogonal_and_the_round_trip_is_exact() {
        let mut rng = Rng::new(1);
        let link = Fhss::new(8, 4, 3, Fhss::random_pattern(8, 50, &mut rng)).unwrap();
        let symbols: Vec<usize> = (0..200).map(|_| rng.below(4)).collect();
        let x = link.modulate(&symbols).unwrap();
        assert_eq!(x.len(), 200 * 32);
        assert!(x.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
        // all the energy of a symbol in its bin: N² there, nothing in the others
        let e = link.energies(&x).unwrap();
        for (i, tones) in e.chunks_exact(4).enumerate() {
            for (a, v) in tones.iter().enumerate() {
                let expected = if a == symbols[i] { 1024.0 } else { 0.0 };
                assert!((v - expected).abs() < 1e-6);
            }
        }
        assert_eq!(link.demodulate(&x).unwrap(), symbols);
    }

    #[test]
    fn the_link_follows_its_pattern() {
        let link = Fhss::new(4, 2, 2, vec![3, 0, 2]).unwrap();
        let channels: Vec<usize> = (0..8).map(|i| link.hop(i)).collect();
        assert_eq!(channels, [3, 3, 0, 0, 2, 2, 3, 3]);
        // the symbol 1 on the channel 3: the tone 7 of 8
        let x = link.modulate(&[1]).unwrap();
        let spectrum = fft(&x);
        assert!((spectrum[7].norm() - 8.0).abs() < 1e-9);
        // a receiver on another pattern sees nothing of it
        let other = Fhss { pattern: vec![1], ..link.clone() };
        assert!(other.energies(&x).unwrap().iter().all(|v| *v < 1e-12));
    }

    #[test]
    fn the_configuration_and_the_inputs_are_checked() {
        assert!(Fhss::new(0, 2, 1, vec![0]).is_err());
        assert!(Fhss::new(4, 1, 1, vec![0]).is_err());
        assert!(Fhss::new(4, 2, 0, vec![0]).is_err());
        assert!(Fhss::new(4, 2, 1, vec![]).is_err());
        assert!(Fhss::new(4, 2, 1, vec![4]).is_err());
        let mut link = Fhss::new(4, 2, 1, vec![0, 1]).unwrap();
        assert!(link.modulate(&[2]).is_err());
        assert!(link.energies(&[C64::new(1.0, 0.0); 9]).is_err());
        let labels = Labels::from_stream(vec![0, 1, 1]).unwrap();
        let y = Modulate::modulate(&link, &labels).unwrap();
        let back: Labels = Demodulate::demodulate(&mut link, &y).unwrap();
        assert_eq!(back.stream(0), [0, 1, 1]);
    }
}
