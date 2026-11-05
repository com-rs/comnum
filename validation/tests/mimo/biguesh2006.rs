//! The training-based MIMO channel estimators of `comnum::mimo::estimation` against M.
//! Biguesh and A. B. Gershman, "Training-based MIMO channel estimation: a study of
//! estimator tradeoffs and optimal training signals," *IEEE Trans. Signal Process.*
//! 54(3), 884–893, 2006, Figs. 1–5 and eqs. (12), (20), (28), (41), (49) and (59).
//!
//! The link of Sec. VIII: $S = H P + V$, $N = t$ training symbols, the entries of $H$ and
//! of $V$ circular complex Gaussian of unit variance ($\sigma_{n}^{2} = 1$), the rows of
//! $H$ independent of covariance $[\varepsilon^{|n-m|}]$ (`Kronecker` of the identity
//! and of `exponential_correlation`), so that $R_{H} = E\\{H^{H} H\\} = r
//! [\varepsilon^{|n-m|}]$ (eq. (60)); the abscissa $\mathcal{P}/\sigma_{n}^{2}$ in dB
//! (the paper prints no unit; only dB makes Fig. 1 follow eq. (12)). The orthogonal
//! training is the DFT matrix of eq. (10), the optimal training of the LMMSE the
//! water-filling of eqs. (41)–(44). The MSE is normalised by $rt$, 5000 independent
//! channels per point as in the paper.
//!
//! Three comparisons per point (the card `validation/cards/biguesh2006.yaml`):
//!
//! - `figure`: the Monte Carlo MSE against the value read on the figure;
//! - `closed_form`: the MSE of [`Training::mse`] against the paper's equation, evaluated
//!   independently on the card ((12) LS, (20) and (49) SLS and RMMSE, (28) LMMSE with
//!   orthogonal training, (41) with the water-filled one, (59) BLUE);
//! - `monte_carlo_over_closed_form`: the ratio of the two, 1 within the sampling error.
//!
//! LS-SLS (eq. (17)) has no closed form: only its figure readings. The paper gives no
//! $\varepsilon$ for Figs. 1, 2 and 5; LS, SLS and BLUE do not depend on it and LS-SLS
//! barely (1.464 at 1 dB, $t = 5$, with $\varepsilon = 0$ against 1.457 with 0.8):
//! $\varepsilon = 0$.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::fading::Mimo;
use comnum::channel::spatial::{Kronecker, exponential_correlation};
use comnum::math::C64;
use comnum::math::linalg::{CMat, identity, matmul};
use comnum::mimo::estimation::{ChannelEstimator, TracePrior, Training, blue_combine, blue_mse, blue_weights};
use comnum::prelude::*;
use comnum_validation::card::Card;
use serde_json::{Value, json};

const RUNS: usize = 5000;

/// One point of a figure.
struct Point {
    figure: u32,
    estimator: &'static str,
    t: usize,
    r: usize,
    eps: f64,
    training: &'static str,
    snr_db: f64,
    periods: Option<usize>,
}

impl Point {
    fn inputs(&self, quantity: &str) -> Value {
        let mut v = json!({
            "figure": self.figure, "estimator": self.estimator, "t": self.t, "r": self.r,
            "eps": self.eps, "training": self.training, "snr_db": self.snr_db, "quantity": quantity,
        });
        if let Some(k) = self.periods {
            v["periods"] = json!(k);
        }
        v
    }

    fn correlation(&self) -> CMat {
        exponential_correlation(self.t, C64::new(self.eps, 0.0)) * C64::from(self.r as f64)
    }

    fn training(&self) -> Training {
        let power = 10f64.powf(self.snr_db / 10.0);
        match self.training {
            "optimal" => Training::mmse_optimal(&self.correlation().view(), 1.0, self.r, power, self.t).unwrap(),
            _ => Training::orthogonal(self.t, self.t, power).unwrap(),
        }
    }

    fn estimator(&self) -> ChannelEstimator {
        let trace = TracePrior::Known((self.r * self.t) as f64);
        match self.estimator {
            "ls" | "blue" => ChannelEstimator::Ls,
            "sls" => ChannelEstimator::ScaledLs(trace),
            "ls_sls" => ChannelEstimator::ScaledLs(TracePrior::FromLs),
            "rmmse" => ChannelEstimator::RelaxedMmse(trace),
            _ => ChannelEstimator::Mmse(self.correlation()),
        }
    }

    /// The normalised MSE of [`Training::mse`] (or [`blue_mse`]).
    fn closed_form(&self) -> f64 {
        let p = self.training();
        let j = match self.periods {
            Some(k) => blue_mse(&vec![p; k], 1.0, self.r).unwrap(),
            None => p.mse(&self.estimator(), &self.correlation().view(), 1.0, self.r).unwrap(),
        };
        j / (self.r * self.t) as f64
    }

    /// The normalised Monte Carlo MSE over [`RUNS`] channels.
    fn monte_carlo(&self, seed: u64) -> f64 {
        let channel = Mimo {
            correlation: Some(
                Kronecker::new(identity(self.r), exponential_correlation(self.t, C64::new(self.eps, 0.0))).unwrap(),
            ),
            ..Mimo::rayleigh(self.r, self.t)
        };
        let (p, est) = (self.training(), self.estimator());
        let periods = self.periods.unwrap_or(1);
        let weights = match self.periods {
            Some(k) => blue_weights(&vec![p.clone(); k]).unwrap(),
            None => vec![1.0],
        };
        let mut rng = SimRng::new(seed);
        let mut acc = 0.0;
        for _ in 0..RUNS {
            let h = channel.draw(&mut rng).unwrap();
            let estimates: Vec<CMat> = (0..periods)
                .map(|_| {
                    let n = p.n_symbols();
                    let v = CMat::from_shape_vec((self.r, n), rng.complex_noise(self.r * n, 1.0)).unwrap();
                    let s = matmul(&h.view(), &p.symbols().view()) + v;
                    p.estimate(&s.view(), &est, 1.0).unwrap()
                })
                .collect();
            let h_hat = blue_combine(&estimates, &weights).unwrap();
            acc += (&h - &h_hat).iter().map(C64::norm_sqr).sum::<f64>();
        }
        acc / RUNS as f64 / (self.r * self.t) as f64
    }
}

/// Every point that the figures could hold; the card says which are compared.
fn points() -> Vec<Point> {
    let mut out = Vec::new();
    let mut push = |figure, estimator, t, r, eps, training, snr_db: f64, periods| {
        out.push(Point { figure, estimator, t, r, eps, training, snr_db, periods });
    };
    for t in 1..=5 {
        for x in [1.0, 5.0, 10.0, 15.0, 19.0] {
            push(1, "ls", t, 3, 0.0, "orthogonal", x, None);
        }
        for est in ["sls", "ls_sls"] {
            for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 19.0] {
                push(2, est, t, 3, 0.0, "orthogonal", x, None);
            }
        }
    }
    for eps in [0.4, 0.8] {
        for t in [1, 2, 3, 5] {
            for training in ["orthogonal", "optimal"] {
                for x in [3.0, 8.0, 10.0, 19.0] {
                    push(3, "mmse", t, 3, eps, training, x, None);
                }
            }
        }
    }
    for t in [2, 8] {
        for est in ["ls", "sls", "rmmse", "mmse"] {
            for x in [3.0, 6.0, 10.0, 19.0] {
                push(4, est, t, t, 0.8, "orthogonal", x, None);
            }
        }
    }
    for k in [1, 2, 4] {
        for x in [3.0, 10.0, 19.0] {
            push(5, "blue", 4, 2, 0.0, "orthogonal", x, Some(k));
        }
    }
    out
}

#[test]
fn biguesh_2006_figs_1_to_5() {
    let mut card = Card::load("biguesh2006").unwrap();
    for (i, pt) in points().iter().enumerate() {
        let [fig, closed, ratio] = ["figure", "closed_form", "monte_carlo_over_closed_form"].map(|q| pt.inputs(q));
        let wanted = [&fig, &closed, &ratio].map(|q| card.expected(q).is_some());
        if !wanted.iter().any(|w| *w) {
            continue;
        }
        let mc = (wanted[0] || wanted[2]).then(|| pt.monte_carlo(2006 + i as u64));
        let cf = (wanted[1] || wanted[2]).then(|| pt.closed_form());
        if let Some(m) = mc.filter(|_| wanted[0]) {
            card.check(&fig, m);
        }
        if let Some(c) = cf.filter(|_| wanted[1]) {
            card.check(&closed, c);
        }
        if wanted[2] {
            card.check(&ratio, mc.unwrap() / cf.unwrap());
        }
    }
    card.finish().unwrap();
}
