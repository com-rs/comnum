//! The error rates of FH/MFSK under partial-band noise jamming.
//!
//! # Model
//!
//! A jammer of power $J$ over the whole band is noise of the density $N_{J}$. Spent on
//! the fraction $\rho$ of the band, it hits a hop with the probability $\rho$ and is
//! then $1/\rho$ times denser. With $P_{b}(\gamma)$ the bit error rate of noncoherent
//! $M$-FSK at $E_{b}/N_{0} = \gamma$ ($\tfrac{1}{2} e^{-\gamma/2}$ for $M = 2$) and the
//! thermal noise $N_{0}$, a slow-hopping link without coding has
//! ([`ber_partial_band`])
//!
//! $$
//! P_{b}(\rho) = (1 - \rho)\\, P_{b}\\!\left(\frac{E_{b}}{N_{0}}\right) + \rho\\,
//! P_{b}\\!\left(\frac{E_{b}}{N_{0} + N_{J}/\rho}\right).
//! $$
//!
//! The jammer chooses $\rho$. For binary FSK without thermal noise, $\frac{\rho}{2}
//! e^{-\rho E_{b}/(2 N_{J})}$ is largest at $\rho^{\star} = 2 N_{J}/E_{b}$ (when $E_{b}/N_{J}
//! \ge 2$; the whole band otherwise), where
//!
//! $$
//! P_{b}^{\star} = \frac{e^{-1}}{E_{b}/N_{J}}:
//! $$
//!
//! the exponential decay of the error rate has become an inverse-linear one, as over
//! Rayleigh fading, and the jammer gains tens of dB at low error rates (Viterbi and
//! Jacobs 1975; Proakis and Salehi 2008, Sec. 12.3). [`worst_case_partial_band`] finds
//! the worst fraction for any $M$ and thermal noise. Coding with interleaving, or
//! several hops per symbol, restores the exponential decay; neither is modelled here.

use comnum_core::{Result, ensure};
use comnum_mapping::theory::ber_orthogonal_noncoherent;
use comnum_math::{db2lin, golden_max, lin2db};

/// The bit error rate of noncoherent `m`-FSK at the linear $E_{b}/N_{0}$ `ebn0` (0 for
/// an infinite ratio).
fn ber_fsk(m: usize, ebn0: f64) -> Result<f64> {
    if ebn0.is_infinite() {
        ensure(m >= 2, || format!("M orthogonal signals need M ≥ 2, got {m}"))?;
        return Ok(0.0);
    }
    ber_orthogonal_noncoherent(m, lin2db(ebn0 * (m as f64).log2()))
}

/// [`ber_partial_band`] on linear ratios, `n0` $= N_{0}/E_{b}$ and `nj` $= N_{J}/E_{b}$.
fn ber_at(m: usize, n0: f64, nj: f64, rho: f64) -> Result<f64> {
    Ok((1.0 - rho) * ber_fsk(m, 1.0 / n0)? + rho * ber_fsk(m, 1.0 / (n0 + nj / rho))?)
}

/// The linear $N_{0}/E_{b}$ of an optional $E_{b}/N_{0}$ in dB (none: no thermal noise).
fn thermal(ebn0_db: Option<f64>) -> f64 {
    ebn0_db.map_or(0.0, |e| 1.0 / db2lin(e))
}

/// The bit error rate of slow FH with noncoherent `m`-FSK under a noise jammer on the
/// fraction `rho` of the band, at `ebnj_db` $= E_{b}/N_{J}$ (dB, $N_{J}$ the density of
/// the jammer spread over the whole band) and the thermal `ebn0_db` (dB; `None`
/// without thermal noise): see the module.
///
/// # Errors
/// [`comnum_core::Error::Config`] for `m` below 2 or a fraction outside $(0, 1\]$.
///
/// # Examples
/// ```
/// use comnum_waveform::fhss::theory::ber_partial_band;
/// // binary FSK, the whole band jammed: e^{-Eb/(2 NJ)} / 2
/// let p = ber_partial_band(2, 10.0, 1.0, None)?;
/// assert!((p - 0.5 * (-5.0f64).exp()).abs() < 1e-9);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ber_partial_band(m: usize, ebnj_db: f64, rho: f64, ebn0_db: Option<f64>) -> Result<f64> {
    ensure(rho > 0.0 && rho <= 1.0, || format!("the jammed fraction lies in (0, 1], got {rho}"))?;
    ber_at(m, thermal(ebn0_db), 1.0 / db2lin(ebnj_db), rho)
}

/// The fraction $\rho^{\star}$ of the band that maximises [`ber_partial_band`] and the
/// bit error rate there, by a golden-section search on $\log \rho$ (see the module;
/// $2 N_{J}/E_{b}$ and $e^{-1} N_{J}/E_{b}$ for binary FSK without thermal noise).
///
/// # Errors
/// [`comnum_core::Error::Config`] for `m` below 2.
///
/// # Examples
/// ```
/// use comnum_waveform::fhss::theory::worst_case_partial_band;
/// let (rho, ber) = worst_case_partial_band(2, 20.0, None)?;
/// assert!((rho - 0.02).abs() < 1e-6 && (ber - (-1.0f64).exp() / 100.0).abs() < 1e-9);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn worst_case_partial_band(m: usize, ebnj_db: f64, ebn0_db: Option<f64>) -> Result<(f64, f64)> {
    let (n0, nj) = (thermal(ebn0_db), 1.0 / db2lin(ebnj_db));
    let whole = ber_at(m, n0, nj, 1.0)?;
    // the fraction of the maximum is of the order of NJ/Eb: twelve decades cover it
    let log_rho = golden_max(|x| ber_at(m, n0, nj, x.exp()).unwrap_or(0.0), -28.0, 0.0, 120);
    let rho = log_rho.exp();
    let ber = ber_at(m, n0, nj, rho)?;
    // the whole band unless a part of it is worse by more than the rounding of the search
    Ok(if ber > whole * (1.0 + 1e-12) { (rho, ber) } else { (1.0, whole) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_fsk_has_the_closed_form_and_its_worst_case() {
        for ebnj_db in [5.0, 10.0, 20.0, 30.0, 40.0] {
            let g = db2lin(ebnj_db);
            for rho in [1.0, 0.3, 0.01] {
                let p = ber_partial_band(2, ebnj_db, rho, None).unwrap();
                assert!((p - 0.5 * rho * (-rho * g / 2.0).exp()).abs() < 1e-9 * rho);
            }
            // the worst fraction 2 NJ/Eb and the error rate e^{-1} NJ/Eb
            let (rho, ber) = worst_case_partial_band(2, ebnj_db, None).unwrap();
            assert!((rho - 2.0 / g).abs() < 1e-5 * rho, "{ebnj_db}: {rho}");
            assert!((ber - (-1.0f64).exp() / g).abs() < 1e-9 * ber);
        }
        // below Eb/NJ = 2 the jammer fills the band
        let (rho, ber) = worst_case_partial_band(2, 0.0, None).unwrap();
        assert_eq!(rho, 1.0);
        assert!((ber - 0.5 * (-0.5f64).exp()).abs() < 1e-9);
    }

    #[test]
    fn the_thermal_noise_and_the_alphabet_enter_as_in_awgn() {
        // no jammer to speak of: the error rate of the thermal noise
        let quiet = ber_partial_band(4, 80.0, 0.5, Some(8.0)).unwrap();
        let awgn = ber_orthogonal_noncoherent(4, 8.0 + lin2db(2.0)).unwrap();
        assert!((quiet - awgn).abs() < 1e-6 * awgn);
        // the whole band jammed: the two noises add
        let both = ber_partial_band(2, 10.0, 1.0, Some(10.0)).unwrap();
        assert!((both - 0.5 * (-2.5f64).exp()).abs() < 1e-9);
        // the worst case is never below the full-band jammer, and the thermal noise
        // raises it
        for m in [2, 4, 16] {
            let (rho, worst) = worst_case_partial_band(m, 15.0, None).unwrap();
            assert!(rho < 1.0 && worst > ber_partial_band(m, 15.0, 1.0, None).unwrap());
            assert!(worst_case_partial_band(m, 15.0, Some(12.0)).unwrap().1 > worst);
        }
        assert!(ber_partial_band(2, 10.0, 0.0, None).is_err());
        assert!(ber_partial_band(1, 10.0, 0.5, None).is_err());
        assert!(worst_case_partial_band(1, 10.0, None).is_err());
    }
}
