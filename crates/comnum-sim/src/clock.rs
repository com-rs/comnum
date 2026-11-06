//! The wall clock of a run, on every target.
//!
//! The runner, the progress bar and the profile of a link time themselves with a
//! [`Stopwatch`]. On the native targets it is [`std::time::Instant`]. On
//! `wasm32-unknown-unknown` (the browser), where the standard library has no clock, it
//! reads the clock that the host registers with [`set_clock`] (`performance.now()`,
//! `Date.now()`), and stands still, at zero, until one is registered: the results of a
//! run never depend on it, only the wall times of the reports and the
//! [`Stop::max_time`](crate::stopping::Stop::max_time) rule do.
//!
//! # Examples
//! ```
//! use comnum_sim::clock::Stopwatch;
//! let t = Stopwatch::start();
//! let d = t.elapsed();
//! assert!(d <= t.elapsed());
//! ```

use std::time::Duration;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod imp {
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    #[derive(Clone, Copy, Debug)]
    pub(super) struct Now(Instant);

    pub(super) fn now() -> Now {
        Now(Instant::now())
    }

    pub(super) fn since(t: Now) -> Duration {
        t.0.elapsed()
    }

    pub(super) fn unix_time_s() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
    }

    pub(super) fn set(_clock: fn() -> f64) {}
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod imp {
    use std::sync::OnceLock;
    use std::time::Duration;

    /// The clock of the host: seconds, monotonic, from an origin of its choice.
    static CLOCK: OnceLock<fn() -> f64> = OnceLock::new();

    #[derive(Clone, Copy, Debug)]
    pub(super) struct Now(f64);

    pub(super) fn seconds() -> f64 {
        CLOCK.get().map_or(0.0, |f| f())
    }

    pub(super) fn now() -> Now {
        Now(seconds())
    }

    pub(super) fn since(t: Now) -> Duration {
        Duration::try_from_secs_f64((seconds() - t.0).max(0.0)).unwrap_or(Duration::ZERO)
    }

    pub(super) fn unix_time_s() -> u64 {
        0
    }

    pub(super) fn set(clock: fn() -> f64) {
        let _ = CLOCK.set(clock);
    }
}

/// A wall-clock timer: [`Instant`](std::time::Instant) on the native targets, the clock
/// of the host in the browser (see the [module](self)).
#[derive(Clone, Copy, Debug)]
pub struct Stopwatch(imp::Now);

impl Stopwatch {
    /// Start timing.
    pub fn start() -> Stopwatch {
        Stopwatch(imp::now())
    }

    /// The wall time since the start.
    pub fn elapsed(&self) -> Duration {
        imp::since(self.0)
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Stopwatch::start()
    }
}

/// The current Unix time in seconds (the `started_unix_s` of every report); zero where
/// the target has no calendar clock (the browser).
pub fn unix_time_s() -> u64 {
    imp::unix_time_s()
}

/// Register the clock of the host on `wasm32-unknown-unknown`: a function that returns
/// seconds, monotonic (`performance.now() / 1e3`). The first registration wins; without
/// one every [`Stopwatch`] reads zero there. On the native targets, which have a clock,
/// the registration is ignored.
///
/// # Examples
/// ```
/// comnum_sim::clock::set_clock(|| 0.0);
/// ```
pub fn set_clock(clock: fn() -> f64) {
    imp::set(clock);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stopwatch_measures_elapsed_time_and_the_unix_time_is_recent() {
        let t = Stopwatch::start();
        std::thread::sleep(Duration::from_millis(5));
        assert!(t.elapsed() >= Duration::from_millis(5));
        assert!(unix_time_s() > 1_700_000_000);
    }
}
