//! Jammers: interference added on purpose to a band.
//!
//! # Model
//!
//! A partial-band noise jammer spends its power $J$ on a part of the band only: on the
//! grid of a DFT of $N$ samples, Gaussian noise in $B$ consecutive bins and nothing
//! elsewhere ([`partial_band_noise`]),
//!
//! $$
//! j\[n\] = \frac{1}{N} \sum_{k=k_{0}}^{k_{0} + B - 1} J_{k}\\, e^{j 2\pi k n / N}, \qquad
//! J_{k} \sim \mathcal{CN}\\!\left(0, \frac{J N^{2}}{B}\right),
//! $$
//!
//! drawn afresh for every block of $N$ samples, so that $E\lvert j\[n\] \rvert^{2} = J$:
//! in the jammed bins it is white noise of the variance $J N / B$ per sample, $N/B$
//! times what the same power gives over the whole band. Against a frequency-hopping
//! link, a jammer that fills the fraction $\rho = B/N$ of the band hits a hop with the
//! probability $\rho$, with a noise density $1/\rho$ times larger
//! (`comnum_waveform::fhss`).
//!
//! # Examples
//! ```
//! use comnum_channel::jamming::partial_band_noise;
//! use comnum_core::rng::Rng;
//! // a quarter of the band of 64 bins, power 2 per sample
//! let j = partial_band_noise(64 * 500, 64, 16, 16, 2.0, &mut Rng::new(1))?;
//! let power = j.iter().map(|v| v.norm_sqr()).sum::<f64>() / j.len() as f64;
//! assert!((power - 2.0).abs() < 0.1);
//! # Ok::<(), comnum_core::Error>(())
//! ```

use comnum_core::rng::SimRng;
use comnum_core::{Result, ensure};
use comnum_math::C64;
use comnum_math::fft::ifft;

/// `n_samples` samples of a partial-band noise jammer of mean power `power` per sample
/// (see the module): Gaussian noise in the `n_bins` bins from `first_bin` (modulo
/// `n_fft`) of consecutive DFT blocks of `n_fft` samples, the last block cut short.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a DFT of size zero, a number of bins that is zero
/// or above `n_fft`, or a power that is negative or not finite.
pub fn partial_band_noise(
    n_samples: usize,
    n_fft: usize,
    first_bin: usize,
    n_bins: usize,
    power: f64,
    rng: &mut SimRng,
) -> Result<Vec<C64>> {
    ensure(n_fft > 0 && (1..=n_fft).contains(&n_bins), || format!("1 to {n_fft} jammed bins, got {n_bins}"))?;
    ensure(power >= 0.0 && power.is_finite(), || format!("the power of the jammer must be non-negative, got {power}"))?;
    let sigma = (power / n_bins as f64).sqrt() * n_fft as f64;
    let mut out = Vec::with_capacity(n_samples + n_fft);
    let mut spectrum = vec![C64::new(0.0, 0.0); n_fft];
    while out.len() < n_samples {
        for k in 0..n_bins {
            spectrum[(first_bin + k) % n_fft] = rng.complex_gaussian() * sigma;
        }
        out.extend(ifft(&spectrum));
    }
    out.truncate(n_samples);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;
    use comnum_math::fft::fft;

    #[test]
    fn the_jammer_fills_its_bins_only_with_its_power() {
        let (n, first, bins, power) = (32, 28, 8, 3.0);
        let j = partial_band_noise(n * 2000, n, first, bins, power, &mut Rng::new(7)).unwrap();
        let mut psd = vec![0.0; n];
        for block in j.chunks_exact(n) {
            fft(block).iter().zip(&mut psd).for_each(|(v, p)| *p += v.norm_sqr() / (2000 * n) as f64);
        }
        // the bins 28..32 and 0..4 (wrapped): white noise of variance J N / B per sample
        for (k, p) in psd.iter().enumerate() {
            let jammed = !(4..28).contains(&k);
            let expected = if jammed { power * n as f64 / bins as f64 } else { 0.0 };
            assert!((p - expected).abs() < 0.1 * expected + 1e-20, "bin {k}: {p}");
        }
        let mean = j.iter().map(C64::norm_sqr).sum::<f64>() / j.len() as f64;
        assert!((mean - power).abs() < 0.05 * power);
    }

    #[test]
    fn the_jammer_checks_its_parameters_and_cuts_the_last_block() {
        let rng = &mut Rng::new(1);
        assert_eq!(partial_band_noise(70, 32, 0, 32, 1.0, rng).unwrap().len(), 70);
        assert!(partial_band_noise(10, 32, 0, 4, 0.0, rng).unwrap().iter().all(|v| v.norm() == 0.0));
        assert!(partial_band_noise(10, 32, 0, 0, 1.0, rng).is_err());
        assert!(partial_band_noise(10, 32, 0, 33, 1.0, rng).is_err());
        assert!(partial_band_noise(10, 0, 0, 1, 1.0, rng).is_err());
        assert!(partial_band_noise(10, 32, 0, 4, -1.0, rng).is_err());
    }
}
