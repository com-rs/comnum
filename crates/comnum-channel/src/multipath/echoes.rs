//! The static echoes of the DVB-T channels of ETSI EN 300 744 Annex B: [`Echoes`].

use super::rms_delay_spread;
use comnum_core::ensure;
use comnum_core::prelude::*;

/// A static channel of discrete echoes, each of its own amplitude, delay and phase, and
/// an optional direct path: the fixed (Ricean, F1) and portable (Rayleigh, P1) channels
/// of Annex B of ETSI EN 300 744 (DVB-T),
///
/// $$
/// y(t) = \frac{\rho_{0}\, x(t) + \sum_{i=1}^{N} \rho_{i}\, e^{-j\theta_{i}}\, x(t - \tau_{i})}{\sqrt{\sum_{i=0}^{N} \rho_{i}^{2}}},
/// $$
///
/// unit power gain, $\rho_{0} = 0$ without direct path. [`Echoes::frequency_response`]
/// evaluates $H(f)$ exactly at any frequency (the delays need not fall on a sampling
/// grid): an OFDM receiver whose guard interval is longer than the echoes sees $H$ at its
/// carriers.
///
/// # Examples
/// ```
/// use comnum_channel::multipath::Echoes;
/// let f1 = Echoes::dvbt_f1();
/// assert!((f1.k_factor_db() - 10.0).abs() < 1e-9);          // Annex B a): K = 10 dB
/// let h = f1.frequency_response(&[0.0, 1e6]);
/// assert_eq!(h.len(), 2);
/// assert!(Echoes::dvbt_p1().direct().is_none());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Echoes {
    /// The amplitude of the direct path $\rho_{0}$ (`None`: no direct path).
    direct: Option<f64>,
    /// The echoes `(amplitude, delay in s, phase in rad)`.
    echoes: Vec<(f64, f64, f64)>,
}

/// The 20 echoes of Table B.1 of EN 300 744: amplitude $\rho_{i}$, delay $\tau_{i}$ (µs),
/// phase $\theta_{i}$ (rad).
const DVBT_ECHOES: [(f64, f64, f64); 20] = [
    (0.057_662, 1.003_019, 4.855_121),
    (0.176_809, 5.422_091, 3.419_109),
    (0.407_163, 0.518_650, 5.864_470),
    (0.303_585, 2.751_772, 2.215_894),
    (0.258_782, 0.602_895, 3.758_058),
    (0.061_831, 1.016_585, 5.430_202),
    (0.150_340, 0.143_556, 3.952_093),
    (0.051_534, 0.153_832, 1.093_586),
    (0.185_074, 3.324_866, 5.775_198),
    (0.400_967, 1.935_570, 0.154_459),
    (0.295_723, 0.429_948, 5.928_383),
    (0.350_825, 3.228_872, 3.053_023),
    (0.262_909, 0.848_831, 0.628_578),
    (0.225_894, 0.073_883, 2.128_544),
    (0.170_996, 0.203_952, 1.099_463),
    (0.149_723, 0.194_207, 3.462_951),
    (0.240_140, 0.924_450, 3.664_773),
    (0.116_587, 1.381_320, 2.833_799),
    (0.221_155, 0.640_512, 3.334_290),
    (0.259_730, 1.368_671, 0.393_889),
];

impl Echoes {
    /// A channel of a direct path of amplitude `direct` (`None` for none) and of the
    /// echoes `(amplitude, delay in s, phase in rad)`.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a negative or non-finite amplitude, a negative
    /// delay, or a channel without power.
    pub fn new(direct: Option<f64>, echoes: Vec<(f64, f64, f64)>) -> Result<Echoes> {
        let valid = |a: f64| a.is_finite() && a >= 0.0;
        ensure(direct.is_none_or(valid), || "the direct path needs a finite non-negative amplitude".into())?;
        ensure(echoes.iter().all(|&(a, d, p)| valid(a) && valid(d) && p.is_finite()), || {
            "every echo needs a finite non-negative amplitude and delay and a finite phase".into()
        })?;
        let e = Echoes { direct, echoes };
        ensure(e.power() > 0.0, || "a channel of echoes needs some power".into())?;
        Ok(e)
    }

    /// The fixed-reception channel F1 of EN 300 744, Annex B a): the 20 echoes of Table B.1
    /// and a direct path $\rho_{0} = \sqrt{10 \sum_{i} \rho_{i}^{2}}$, a Ricean factor of
    /// 10 dB.
    pub fn dvbt_f1() -> Echoes {
        let echoes = Self::dvbt_echoes();
        let power: f64 = echoes.iter().map(|e| e.0 * e.0).sum();
        Echoes { direct: Some((10.0 * power).sqrt()), echoes }
    }

    /// The portable-reception channel P1 of EN 300 744, Annex B b): the 20 echoes of
    /// Table B.1 without direct path (static: the standard gives no Doppler).
    pub fn dvbt_p1() -> Echoes {
        Echoes { direct: None, echoes: Self::dvbt_echoes() }
    }

    fn dvbt_echoes() -> Vec<(f64, f64, f64)> {
        DVBT_ECHOES.iter().map(|&(a, d_us, p)| (a, d_us * 1e-6, p)).collect()
    }

    /// The amplitude of the direct path.
    pub fn direct(&self) -> Option<f64> {
        self.direct
    }

    /// The echoes `(amplitude, delay in s, phase in rad)`.
    pub fn echoes(&self) -> &[(f64, f64, f64)] {
        &self.echoes
    }

    /// $\sum_{i \ge 0} \rho_{i}^{2}$.
    fn power(&self) -> f64 {
        self.direct.map_or(0.0, |a| a * a) + self.echoes.iter().map(|e| e.0 * e.0).sum::<f64>()
    }

    /// The Ricean factor $K = \rho_{0}^{2} / \sum_{i \ge 1} \rho_{i}^{2}$ (dB), $-\infty$
    /// without direct path.
    pub fn k_factor_db(&self) -> f64 {
        let echoes: f64 = self.echoes.iter().map(|e| e.0 * e.0).sum();
        10.0 * (self.direct.map_or(0.0, |a| a * a) / echoes).log10()
    }

    /// The RMS delay spread (s), the direct path at delay 0.
    pub fn rms_delay_spread(&self) -> f64 {
        let taps: Vec<(f64, f64)> = self
            .direct
            .iter()
            .map(|&a| (0.0, a * a))
            .chain(self.echoes.iter().map(|&(a, d, _)| (d, a * a)))
            .map(|(d, p)| (d, 10.0 * p.log10()))
            .collect();
        rms_delay_spread(&taps)
    }

    /// $H(f) = \left(\rho_{0} + \sum_{i} \rho_{i} e^{-j\theta_{i}} e^{-j 2\pi f \tau_{i}}\right)
    /// / \sqrt{\sum_{i} \rho_{i}^{2}}$ at the baseband frequencies `freqs_hz`.
    pub fn frequency_response(&self, freqs_hz: &[f64]) -> Vec<C64> {
        let norm = self.power().sqrt();
        freqs_hz
            .iter()
            .map(|&f| {
                let direct = C64::new(self.direct.unwrap_or(0.0), 0.0);
                let echoes: C64 = self
                    .echoes
                    .iter()
                    .map(|&(a, d, p)| C64::from_polar(a, -p - 2.0 * std::f64::consts::PI * f * d))
                    .sum();
                (direct + echoes) / norm
            })
            .collect()
    }
}
