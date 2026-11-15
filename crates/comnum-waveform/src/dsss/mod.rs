//! Direct-sequence spread spectrum and synchronous code-division multiple access
//! (DS-CDMA): spreading by signature codes, the bank of matched filters, the multiuser
//! detectors and the rake receiver.
//!
//! # Model
//!
//! User $k$ of $K$ sends its symbols $b_{k}\[m\]$ (unit energy) on the signature
//! $s_{k} \in \mathbb{C}^{N}$, $\lVert s_{k} \rVert = 1$, with the amplitude $A_{k}$: $N$
//! chips per symbol, the spreading factor (processing gain). Synchronous users over
//! white noise give, for every symbol period, the vector of $N$ chips
//!
//! $$
//! y = \sum_{k=1}^{K} A_{k} b_{k} s_{k} + w = H b + w, \qquad H = S A \in
//! \mathbb{C}^{N \times K}, \quad w \sim \mathcal{CN}(0, \sigma^{2} I_{N}),
//! $$
//!
//! which is the model $y = H x + w$ of [`comnum_mimo::detection`] with the signatures in
//! place of the channel: the chips are the receive antennas, the users the transmit
//! streams. [`Dsss`] holds $H$ ([`Dsss::signatures`]); [`Dsss::modulate`] forms the chips
//! and every receiver is a detector of `comnum-mimo` applied to it ([`Dsss::detect`]):
//!
//! | Multiuser receiver | Detector | Remark |
//! |---|---|---|
//! | conventional (single-user matched filter) | [`Linear::mr`] | $s_{k}^{H} y$: limited by the interference $\rho_{jk} = s_{j}^{H} s_{k}$, not near–far resistant |
//! | decorrelator | [`Linear::zf`] | $(S^{H} S)^{-1} S^{H} y$: no interference, the noise enhanced by $(R^{-1})_{kk}$; needs $K \le N$ |
//! | linear MMSE | [`Linear::mmse`] | the largest SINR of the linear receivers |
//! | successive cancellation | [`comnum_mimo::detection::Sic`] | strongest user first |
//! | optimum (maximum likelihood) | [`comnum_mimo::detection::Ml`], [`comnum_mimo::detection::Sphere`] | $M^{K}$ hypotheses, or a tree search |
//!
//! [`Dsss::sinr`] is the signal-to-interference-plus-noise ratio of every user at the
//! output of a linear receiver for the signatures at hand, and [`theory`] holds the
//! exact error rates of the matched filter and of the decorrelator, the near–far
//! resistance, and the large-system limits for random signatures (Tse and Hanly 1999;
//! Verdú and Shamai 1999).
//!
//! With one user ($K = 1$) this is a direct-sequence link: [`Dsss::demodulate`], the
//! bank of matched filters $s_{k}^{H} y / A_{k}$, despreads, and an interferer of power
//! $J$ per chip uncorrelated with the code is left with the power $J \lvert s^{H} j
//! \rvert^{2}$, $J/N$ of the symbol energy on average: the processing gain.
//!
//! # Rake receiver
//!
//! Over a channel of $L$ resolvable paths, $r\[n\] = \sum_{l} g_{l}\\, x\[n - d_{l}\] +
//! w\[n\]$ (delays in chips), a finger despreads the path $l$, $z_{l}\[m\] = \sum_{i}
//! s_{k,i}^{\*}\\, r\[mN + i + d_{l}\]$, and [`Dsss::rake`] combines the fingers with the
//! weights $g_{l}^{\*}$ (Price and Green 1958):
//!
//! $$
//! \hat b_{k}\[m\] = \frac{\sum_{l} g_{l}^{\*} z_{l}\[m\]}{A_{k} \sum_{l} \lvert g_{l}
//! \rvert^{2}}.
//! $$
//!
//! The other paths leak through the off-peak autocorrelation of the code, $-1/N$ of the
//! peak for an m-sequence over consecutive identical symbols; without that leakage the
//! rake is maximal-ratio combining of $L$ branches, of the error rates of
//! [`comnum_mapping::theory::ber_binary_fading`] and
//! [`comnum_mapping::theory::ber_binary_mgf`].
//!
//! # Examples
//! ```
//! use comnum_mapping::constellation::Constellation;
//! use comnum_math::{C64, sequences::gold_codes};
//! use comnum_mimo::detection::Linear;
//! use comnum_waveform::dsss::Dsss;
//! // three users on Gold codes of length 31, the second 20 dB above the others
//! let cdma = Dsss::new(&gold_codes(5)?[2..5])?.with_amplitudes(&[1.0, 10.0, 1.0])?;
//! let bpsk = Constellation::psk(2)?;
//! let symbols: Vec<Vec<C64>> = [[0, 1, 1], [1, 1, 0], [0, 0, 1]]
//!     .iter()
//!     .map(|labels| labels.iter().map(|&l| bpsk.points()[l]).collect())
//!     .collect();
//! let chips = cdma.modulate(&symbols)?;
//! assert_eq!(chips.len(), 3 * 31);
//! let decided = cdma.detect(&Linear::zf(bpsk), &chips, 1e-2)?;
//! assert_eq!(decided, vec![vec![0, 1, 1], vec![1, 1, 0], vec![0, 0, 1]]);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. R. Lupas and S. Verdú, "Linear multiuser detectors for synchronous code-division
//!    multiple-access channels," *IEEE Trans. Inf. Theory* 35(1), 123–136, 1989.
//! 2. D. N. C. Tse and S. V. Hanly, "Linear multiuser receivers: effective interference,
//!    effective bandwidth and user capacity," *IEEE Trans. Inf. Theory* 45(2), 641–657,
//!    1999.
//! 3. S. Verdú and S. Shamai (Shitz), "Spectral efficiency of CDMA with random
//!    spreading," *IEEE Trans. Inf. Theory* 45(2), 622–640, 1999.
//! 4. R. Price and P. E. Green, "A communication technique for multipath channels,"
//!    *Proc. IRE* 46(3), 555–570, 1958.

pub mod theory;

use crate::{check_input, check_rate};
use comnum_core::signal::{Signal, Symbols};
use comnum_core::traits::{Demodulate, Modulate};
use comnum_core::units::Hertz;
use comnum_core::{Error, Result, ensure};
use comnum_math::C64;
use comnum_math::linalg::CMat;
use comnum_mimo::detection::{Linear, MimoDetector};
use ndarray::{Array2, ArrayView2, Axis};

/// A direct-sequence link of $K$ synchronous users (see the module): the matrix $H = S
/// A$ of the signatures scaled by the amplitudes, built by [`Dsss::new`] from binary
/// codes or by [`Dsss::from_signatures`].
#[derive(Clone, Debug, PartialEq)]
pub struct Dsss {
    /// $H = S A$, $N \times K$.
    h: CMat,
    /// The rate of the chips (Hz; 1 by default, time in chips).
    pub sample_rate: Hertz,
}

impl Dsss {
    /// The link of the binary `codes`, one per user and all of the same length $N$:
    /// the signatures $s_{k,i} = (-1)^{c_{k,i}} / \sqrt{N}$, unit amplitudes.
    ///
    /// # Errors
    /// [`Error::Config`] without a code, for an empty code or codes of different
    /// lengths.
    pub fn new(codes: &[Vec<u8>]) -> Result<Dsss> {
        let n = codes.first().map_or(0, Vec::len);
        ensure(n > 0 && codes.iter().all(|c| c.len() == n), || {
            "the spreading codes must be nonempty and of one length".to_string()
        })?;
        let scale = 1.0 / (n as f64).sqrt();
        let chip = |(i, k): (usize, usize)| C64::new(if codes[k][i] & 1 == 0 { scale } else { -scale }, 0.0);
        Dsss::from_signatures(Array2::from_shape_fn((n, codes.len()), chip))
    }

    /// The link of the signatures in the columns of `signatures` ($N \times K$, any
    /// complex chips), each normalised to a unit norm.
    ///
    /// # Errors
    /// [`Error::Config`] for an empty matrix or a signature without energy.
    pub fn from_signatures(mut signatures: CMat) -> Result<Dsss> {
        ensure(signatures.nrows() > 0 && signatures.ncols() > 0, || "no signature".to_string())?;
        for mut s in signatures.columns_mut() {
            let norm = s.iter().map(C64::norm_sqr).sum::<f64>().sqrt();
            ensure(norm > 0.0 && norm.is_finite(), || "a signature has no energy".to_string())?;
            s.mapv_inplace(|v| v / norm);
        }
        Ok(Dsss { h: signatures, sample_rate: Hertz(1.0) })
    }

    /// The same signatures received with the `amplitudes` $A_{k}$ in place of the
    /// current ones (the energy per symbol $A_{k}^{2}$).
    ///
    /// # Errors
    /// [`Error::Config`] unless there is one positive amplitude per user.
    pub fn with_amplitudes(mut self, amplitudes: &[f64]) -> Result<Dsss> {
        ensure(amplitudes.len() == self.n_users() && amplitudes.iter().all(|a| *a > 0.0 && a.is_finite()), || {
            format!("{} positive amplitudes, one per user", self.n_users())
        })?;
        let current = self.amplitudes();
        for ((mut s, a), old) in self.h.columns_mut().into_iter().zip(amplitudes).zip(current) {
            s.mapv_inplace(|v| v * (a / old));
        }
        Ok(self)
    }

    /// The spreading factor $N$, chips per symbol.
    pub fn spreading_factor(&self) -> usize {
        self.h.nrows()
    }

    /// The number of users $K$.
    pub fn n_users(&self) -> usize {
        self.h.ncols()
    }

    /// The amplitudes $A_{k}$.
    pub fn amplitudes(&self) -> Vec<f64> {
        self.h.columns().into_iter().map(|s| s.iter().map(C64::norm_sqr).sum::<f64>().sqrt()).collect()
    }

    /// The matrix $H = S A$ ($N \times K$): the channel matrix of the detectors of
    /// [`comnum_mimo::detection`].
    pub fn signatures(&self) -> ArrayView2<'_, C64> {
        self.h.view()
    }

    /// The chips of the symbols of every user: `symbols[k]` holds the $n$ symbols of
    /// user $k$, the result the $nN$ chips of their sum.
    ///
    /// # Errors
    /// [`Error::Shape`] unless there is one stream per user, all of the same length.
    pub fn modulate(&self, symbols: &[Vec<C64>]) -> Result<Vec<C64>> {
        let n = symbols.first().map_or(0, Vec::len);
        if symbols.len() != self.n_users() || symbols.iter().any(|s| s.len() != n) {
            return Err(Error::Shape(format!("{} streams of symbols of one length, one per user", self.n_users())));
        }
        let mut chips = vec![C64::new(0.0, 0.0); n * self.spreading_factor()];
        for (m, block) in chips.chunks_exact_mut(self.spreading_factor()).enumerate() {
            for (k, s) in self.h.columns().into_iter().enumerate() {
                block.iter_mut().zip(s).for_each(|(c, si)| *c += symbols[k][m] * si);
            }
        }
        Ok(chips)
    }

    /// The `chips` as the $N \times n$ block of the detectors, one symbol period per
    /// column.
    fn block(&self, chips: &[C64]) -> Result<Array2<C64>> {
        let n = self.spreading_factor();
        if chips.len() % n != 0 {
            return Err(Error::Shape(format!("{} chips are not whole symbols of {n} chips", chips.len())));
        }
        Ok(Array2::from_shape_fn((n, chips.len() / n), |(i, m)| chips[m * n + i]))
    }

    /// The bank of matched filters: the estimates $s_{k}^{H} y / A_{k}$ of the symbols
    /// of every user, `[k][m]` (one user: the despread symbols; several: with the
    /// interference of the others, see the module).
    ///
    /// # Errors
    /// [`Error::Shape`] unless `chips` holds whole symbols.
    pub fn demodulate(&self, chips: &[C64]) -> Result<Vec<Vec<C64>>> {
        let y = self.block(chips)?;
        let energy = self.amplitudes();
        let filter = |(k, s): (usize, ndarray::ArrayView1<'_, C64>)| {
            let w = s.mapv(|v| v.conj() / (energy[k] * energy[k]));
            y.columns().into_iter().map(|col| w.dot(&col)).collect()
        };
        Ok(self.h.columns().into_iter().enumerate().map(filter).collect())
    }

    /// The labels decided by the multiuser `detector` for every user, `[k][m]`, the
    /// noise of variance `noise_var` per chip ($\sigma^{2} = E\lvert w_{i} \rvert^{2}$).
    ///
    /// # Errors
    /// [`Error::Shape`] unless `chips` holds whole symbols, and the errors of the
    /// detector ([`MimoDetector::detect_block`]: more users than chips for the
    /// decorrelator, too many hypotheses for the exhaustive search).
    pub fn detect<D: MimoDetector + ?Sized>(
        &self,
        detector: &D,
        chips: &[C64],
        noise_var: f64,
    ) -> Result<Vec<Vec<usize>>> {
        let labels = detector.detect_block(&self.h.view(), &self.block(chips)?.view(), noise_var)?;
        Ok(labels.axis_iter(Axis(0)).map(|row| row.to_vec()).collect())
    }

    /// The signal-to-interference-plus-noise ratio of every user at the output of the
    /// `linear` receiver for these signatures, symbols of the energy of its
    /// constellation and the noise variance `noise_var` per chip: the conventional
    /// receiver ([`Linear::mr`]), the decorrelator ([`Linear::zf`]) or the MMSE
    /// receiver ([`Linear::mmse`]).
    ///
    /// # Errors
    /// As [`Linear::estimate`].
    pub fn sinr(&self, linear: &Linear, noise_var: f64) -> Result<Vec<f64>> {
        let silence = vec![C64::new(0.0, 0.0); self.spreading_factor()];
        let es = linear.constellation.energy();
        Ok(linear.estimate(&self.h.view(), &silence, noise_var)?.noise_var.iter().map(|v| es / v).collect())
    }

    /// The rake receiver of user `user` (see the module): the fingers `(delay in
    /// chips, gain)` of the paths combined with the conjugate gains, one estimate per
    /// whole symbol of `chips` (the chips beyond its end count as zero).
    ///
    /// # Errors
    /// [`Error::Config`] for a user that does not exist, without a finger or for
    /// fingers without energy.
    pub fn rake(&self, chips: &[C64], user: usize, fingers: &[(usize, C64)]) -> Result<Vec<C64>> {
        ensure(user < self.n_users(), || format!("user {user} of {}", self.n_users()))?;
        let gain: f64 = fingers.iter().map(|(_, g)| g.norm_sqr()).sum();
        ensure(gain > 0.0 && gain.is_finite(), || "the fingers of a rake receiver need energy".to_string())?;
        let s = self.h.column(user);
        let scale = gain * s.iter().map(C64::norm_sqr).sum::<f64>();
        let n = self.spreading_factor();
        let finger = |start: usize, delay: usize| -> C64 {
            s.iter().zip(chips.iter().skip(start + delay)).map(|(si, r)| si.conj() * r).sum()
        };
        Ok((0..chips.len() / n)
            .map(|m| fingers.iter().map(|&(d, g)| g.conj() * finger(m * n, d)).sum::<C64>() / scale)
            .collect())
    }
}

/// The transmitter as a [`Modulate`] block: one stream of symbols per user, one stream
/// of chips at [`Dsss::sample_rate`].
impl Modulate for Dsss {
    fn modulate_into(&self, x: &Symbols, out: &mut Signal) -> Result<()> {
        check_rate(self.sample_rate)?;
        let streams: Vec<Vec<C64>> = (0..x.n_streams()).map(|k| x.stream(k).to_vec()).collect();
        out.set_stream(&Dsss::modulate(self, &streams)?, self.sample_rate);
        Ok(())
    }
}

/// The bank of matched filters as a [`Demodulate`] block: one stream of estimates per
/// user ([`Dsss::demodulate`]).
impl Demodulate for Dsss {
    /// # Errors
    /// [`Error::Config`] for several streams or a signal at another sample rate, and
    /// the errors of [`Dsss::demodulate`].
    fn demodulate_into(&mut self, y: &Signal, out: &mut Symbols) -> Result<()> {
        check_input(y, self.sample_rate, "DSSS")?;
        *out = Symbols::from_streams(&Dsss::demodulate(self, y.stream(0))?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;
    use comnum_mapping::constellation::Constellation;
    use comnum_math::sequences::{gold_codes, m_sequence, walsh_codes};
    use comnum_mimo::detection::{Ml, Sic};

    fn bpsk() -> Constellation {
        Constellation::psk(2).unwrap()
    }

    /// `n` random labels per user and their symbols.
    fn random_symbols(c: &Constellation, users: usize, n: usize, rng: &mut Rng) -> (Vec<Vec<usize>>, Vec<Vec<C64>>) {
        let labels: Vec<Vec<usize>> =
            (0..users).map(|_| (0..n).map(|_| rng.below(c.points().len())).collect()).collect();
        let symbols = labels.iter().map(|l| l.iter().map(|&i| c.points()[i]).collect()).collect();
        (labels, symbols)
    }

    #[test]
    fn orthogonal_codes_separate_the_users_exactly() {
        let cdma =
            Dsss::new(&walsh_codes(8).unwrap()).unwrap().with_amplitudes(&[1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 1.0, 4.0]);
        let cdma = cdma.unwrap();
        let qpsk = Constellation::psk(4).unwrap();
        let (_, symbols) = random_symbols(&qpsk, 8, 20, &mut Rng::new(1));
        let chips = cdma.modulate(&symbols).unwrap();
        assert_eq!(chips.len(), 160);
        for (z, s) in cdma.demodulate(&chips).unwrap().iter().zip(&symbols) {
            assert!(z.iter().zip(s).all(|(a, b)| (a - b).norm() < 1e-12));
        }
        assert!(
            cdma.amplitudes().iter().zip([1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 1.0, 4.0]).all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn every_multiuser_detector_recovers_the_symbols_without_noise() {
        // a near–far case: the conventional receiver fails, the others do not
        let cdma =
            Dsss::new(&gold_codes(5).unwrap()[..6]).unwrap().with_amplitudes(&[1.0, 30.0, 30.0, 30.0, 30.0, 30.0]);
        let cdma = cdma.unwrap();
        let (labels, symbols) = random_symbols(&bpsk(), 6, 200, &mut Rng::new(2));
        let chips = cdma.modulate(&symbols).unwrap();
        assert_eq!(cdma.detect(&Linear::zf(bpsk()), &chips, 1e-6).unwrap(), labels);
        assert_eq!(cdma.detect(&Linear::mmse(bpsk()), &chips, 1e-6).unwrap(), labels);
        assert_eq!(cdma.detect(&Sic::mmse(bpsk()), &chips, 1e-6).unwrap(), labels);
        assert_eq!(cdma.detect(&Ml::new(bpsk()), &chips, 1e-6).unwrap(), labels);
        let conventional = cdma.detect(&Linear::mr(bpsk()), &chips, 1e-6).unwrap();
        assert_ne!(conventional[0], labels[0]);
        assert_eq!(conventional[1..], labels[1..]);
    }

    #[test]
    fn the_sinr_of_the_linear_receivers_is_the_one_measured() {
        let cdma = Dsss::new(&gold_codes(5).unwrap()[3..13]).unwrap();
        let noise_var = 0.1;
        let h = cdma.signatures();
        let g = comnum_math::linalg::gram(&h);
        // the conventional receiver: A_k² / (σ² + Σ_j A_j² |ρ_jk|²)
        let mr = cdma.sinr(&Linear::mr(bpsk()), noise_var).unwrap();
        for (k, v) in mr.iter().enumerate() {
            let interference: f64 = (0..10).filter(|&j| j != k).map(|j| g[[k, j]].norm_sqr()).sum();
            assert!((v - 1.0 / (noise_var + interference)).abs() < 1e-12);
        }
        // the decorrelator: 1 / (σ² (R⁻¹)_kk); the MMSE receiver is above both
        let inv = comnum_math::linalg::inverse_hpd(&g.view()).unwrap();
        let zf = cdma.sinr(&Linear::zf(bpsk()), noise_var).unwrap();
        let mmse = cdma.sinr(&Linear::mmse(bpsk()), noise_var).unwrap();
        for k in 0..10 {
            assert!((zf[k] - 1.0 / (noise_var * inv[[k, k]].re)).abs() < 1e-9);
            assert!(mmse[k] > zf[k] && mmse[k] > mr[k]);
        }
        // one user: the three are the SNR
        let one = Dsss::new(&[m_sequence(5).unwrap()]).unwrap().with_amplitudes(&[2.0]).unwrap();
        for d in [Linear::mr(bpsk()), Linear::zf(bpsk()), Linear::mmse(bpsk())] {
            assert!((one.sinr(&d, 0.5).unwrap()[0] - 8.0).abs() < 1e-12);
        }
    }

    #[test]
    fn despreading_divides_a_tone_jammer_by_the_processing_gain() {
        // a tone of power J per chip leaves J |s^H j|² after despreading: J/N on
        // average over its frequency (Parseval on the spectrum of the code)
        let link = Dsss::new(&[m_sequence(7).unwrap()]).unwrap();
        let n = link.spreading_factor();
        let mean: f64 = (0..n)
            .map(|f| {
                let tone: Vec<C64> =
                    (0..n).map(|i| comnum_math::cis(std::f64::consts::TAU * (f * i) as f64 / n as f64)).collect();
                link.demodulate(&tone).unwrap()[0][0].norm_sqr()
            })
            .sum::<f64>()
            / n as f64;
        assert!((mean - 1.0).abs() < 1e-9, "{mean}");
        // and (N + 1)/N² ≈ 1/N of the power per chip at every frequency but the DC
        let tone: Vec<C64> =
            (0..n).map(|i| comnum_math::cis(std::f64::consts::TAU * (5 * i) as f64 / n as f64)).collect();
        let left = link.demodulate(&tone).unwrap()[0][0].norm_sqr() / n as f64;
        assert!((left - (n as f64 + 1.0) / (n * n) as f64).abs() < 1e-12, "{left}");
    }

    #[test]
    fn the_rake_combines_the_paths_with_their_gains() {
        let link = Dsss::new(&[m_sequence(8).unwrap()]).unwrap();
        let n = link.spreading_factor();
        let (_, symbols) = random_symbols(&bpsk(), 1, 50, &mut Rng::new(3));
        let x = link.modulate(&symbols).unwrap();
        let paths = [(0usize, C64::new(0.6, 0.3)), (3, C64::new(-0.2, 0.5)), (11, C64::new(0.1, -0.4))];
        let mut r = vec![C64::new(0.0, 0.0); x.len() + 11];
        for &(d, g) in &paths {
            x.iter().enumerate().for_each(|(i, xi)| r[i + d] += g * xi);
        }
        // every finger: the symbol, and the other paths through the off-peak
        // autocorrelation of the m-sequence, at most 1/N of the peak each (2/N across a
        // symbol transition)
        let z = link.rake(&r, 0, &paths).unwrap();
        assert_eq!(z.len(), 50 + 11 / n);
        let worst = z.iter().zip(&symbols[0]).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
        assert!(worst < 0.1, "{worst}");
        // one finger on a single path is the matched filter
        let single = link.rake(&x, 0, &[(0, C64::new(0.0, 2.0))]).unwrap();
        let matched = &link.demodulate(&x).unwrap()[0];
        assert!(single.iter().zip(matched).all(|(a, b)| (a * C64::new(0.0, 2.0) - b).norm() < 1e-12));
        assert!(link.rake(&r, 1, &paths).is_err() && link.rake(&r, 0, &[]).is_err());
    }

    #[test]
    fn the_blocks_check_their_shapes_and_carry_the_streams() {
        let mut cdma = Dsss::new(&walsh_codes(4).unwrap()[1..3]).unwrap();
        assert_eq!((cdma.spreading_factor(), cdma.n_users()), (4, 2));
        assert!(Dsss::new(&[]).is_err() && Dsss::new(&[vec![0, 1], vec![0]]).is_err());
        assert!(cdma.clone().with_amplitudes(&[1.0]).is_err() && cdma.clone().with_amplitudes(&[1.0, 0.0]).is_err());
        assert!(cdma.modulate(&[vec![C64::new(1.0, 0.0)]]).is_err());
        assert!(cdma.demodulate(&[C64::new(1.0, 0.0); 5]).is_err());
        let (_, symbols) = random_symbols(&bpsk(), 2, 6, &mut Rng::new(4));
        let x = Symbols::from_streams(&symbols).unwrap();
        let y = Modulate::modulate(&cdma, &x).unwrap();
        assert_eq!((y.n_streams(), y.len()), (1, 24));
        let back: Symbols = Demodulate::demodulate(&mut cdma, &y).unwrap();
        assert!(back.as_array().iter().zip(symbols.iter().flatten()).all(|(a, b)| (a - b).norm() < 1e-12));
    }
}
