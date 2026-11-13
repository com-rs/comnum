//! The description of a channel realisation shared by the media that draw it and the
//! waveforms that take it.
//!
//! A doubly-dispersive channel of a few discrete paths, each with its complex gain
//! $h_{i}$, its delay $l_{i}$ (samples) and a single Doppler shift $\nu_{i}$ (Hz), acts on
//! a signal sampled at $f_{s}$ as
//!
//! $$
//! r\[n\] = \sum_{i=1}^{P} h_{i}\, e^{j 2\pi \nu_{i} (n - l_{i}) / f_{s}}\, s\[n - l_{i}\].
//! $$
//!
//! `comnum_channel::multipath::DelayDoppler` draws and applies it;
//! `comnum_waveform::otfs::Otfs::effective_channel` takes its paths. Both crates sit on the
//! same layer, so the type of a path lives here (it contains no algorithm).
//!
//! # Examples
//! ```
//! use comnum_core::channel::Path;
//! use comnum_core::C64;
//! let p = Path { gain: C64::new(1.0, 0.0), delay: 2, doppler_hz: 1e3 };
//! assert_eq!(p.delay, 2);
//! ```

use crate::C64;
use crate::error::{Error, Result, ensure};

/// One path of a delay–Doppler channel: a complex gain, a delay (samples) and a single
/// Doppler shift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Path {
    /// Complex gain $h_{i}$.
    pub gain: C64,
    /// Delay $l_{i}$ (samples).
    pub delay: usize,
    /// Doppler shift $\nu_{i}$ (Hz).
    pub doppler_hz: f64,
}

/// The distribution of the complex gain $h$ of a flat fading channel, of unit mean
/// power $\mathbb{E}|h|^{2} = 1$ (the amplitude densities of Table 2 of M. K. Simon and
/// M.-S. Alouini, *Proc. IEEE* 86(9), 1998), shared by the medium that draws the gains
/// (`comnum_channel::fading::flat_gains`) and the closed forms of the error rates over
/// it (`comnum_mapping::theory::SnrFading`):
///
/// - Rayleigh: $h \sim \mathcal{CN}(0, 1)$;
/// - Rice of factor $K$: a line of sight $\sqrt{K/(1 + K)}$ plus
///   $\mathcal{CN}(0, 1/(1 + K))$;
/// - Nakagami-m: $|h|^{2} \sim \Gamma(m)/m$ (a gamma variable of shape $m$ and mean 1)
///   with a uniform phase;
/// - Hoyt (Nakagami-q): a Gaussian of variance $1/(1 + q^{2})$ on the real axis and
///   $q^{2}/(1 + q^{2})$ on the imaginary one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlatFading {
    /// $h \sim \mathcal{CN}(0, 1)$.
    Rayleigh,
    /// A line of sight and a scattered term, of power ratio `k_factor` (linear): 0 is
    /// Rayleigh fading.
    Rice {
        /// $K \ge 0$ (linear).
        k_factor: f64,
    },
    /// $|h|^{2}$ a gamma variable of shape `m` and unit mean: $m = 1$ is Rayleigh
    /// fading, $m = 1/2$ the one-sided Gaussian.
    Nakagami {
        /// $m \ge 1/2$.
        m: f64,
    },
    /// Unequal variances on the two axes, of ratio $q^{2}$: $q = 1$ is Rayleigh fading,
    /// $q = 0$ the one-sided Gaussian.
    Hoyt {
        /// $0 \le q \le 1$.
        q: f64,
    },
}

impl FlatFading {
    /// The distribution of a name, `rayleigh`, `rice`, `nakagami` or `hoyt`
    /// (case-insensitive), with its parameter ($K$, $m$ or $q$) when it has one.
    ///
    /// # Errors
    /// [`Error::Config`] for another name, a missing or an invalid parameter.
    ///
    /// # Examples
    /// ```
    /// use comnum_core::channel::FlatFading;
    /// assert_eq!(FlatFading::from_name("Rice", Some(3.0))?, FlatFading::Rice { k_factor: 3.0 });
    /// assert!(FlatFading::from_name("nakagami", None).is_err());
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn from_name(name: &str, parameter: Option<f64>) -> Result<FlatFading> {
        let needs = |what: &str| {
            parameter.ok_or_else(|| Error::Config(format!("the {name} fading needs its parameter {what}")))
        };
        let f = match name.to_ascii_lowercase().as_str() {
            "rayleigh" => FlatFading::Rayleigh,
            "rice" | "rician" => FlatFading::Rice { k_factor: needs("K (linear)")? },
            "nakagami" => FlatFading::Nakagami { m: needs("m")? },
            "hoyt" => FlatFading::Hoyt { q: needs("q")? },
            _ => {
                return Err(Error::Config(format!(
                    "the fading must be rayleigh, rice, nakagami or hoyt, got {name:?}"
                )));
            }
        };
        f.check()?;
        Ok(f)
    }

    /// Its parameter is in its range.
    ///
    /// # Errors
    /// [`Error::Config`] for a negative or infinite $K$, $m$ below $1/2$, $q$ outside
    /// $[0, 1]$.
    pub fn check(self) -> Result<()> {
        match self {
            FlatFading::Rayleigh => Ok(()),
            FlatFading::Rice { k_factor } => ensure(k_factor >= 0.0 && k_factor.is_finite(), || {
                format!("the Rician factor must be finite and non-negative, got {k_factor}")
            }),
            FlatFading::Nakagami { m } => ensure(m >= 0.5 && m.is_finite(), || {
                format!("the Nakagami parameter must be finite and at least 1/2, got {m}")
            }),
            FlatFading::Hoyt { q } => {
                ensure((0.0..=1.0).contains(&q), || format!("the Hoyt parameter must be in [0, 1], got {q}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_and_the_ranges_of_the_flat_fading_distributions() {
        assert_eq!(FlatFading::from_name("RAYLEIGH", None).unwrap(), FlatFading::Rayleigh);
        assert_eq!(FlatFading::from_name("rician", Some(2.0)).unwrap(), FlatFading::Rice { k_factor: 2.0 });
        assert_eq!(FlatFading::from_name("nakagami", Some(0.5)).unwrap(), FlatFading::Nakagami { m: 0.5 });
        assert_eq!(FlatFading::from_name("hoyt", Some(1.0)).unwrap(), FlatFading::Hoyt { q: 1.0 });
        for (name, parameter) in [
            ("weibull", None),
            ("rice", None),
            ("rice", Some(-1.0)),
            ("rice", Some(f64::INFINITY)),
            ("nakagami", Some(0.4)),
            ("hoyt", Some(1.5)),
            ("hoyt", Some(f64::NAN)),
        ] {
            assert!(FlatFading::from_name(name, parameter).is_err(), "{name} {parameter:?}");
        }
    }
}
