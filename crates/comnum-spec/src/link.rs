//! The link of a condition ([`Link`]) and a specification as a scenario of the runner
//! ([`SpecScenario`]).

use crate::blocks::{self, Block, BuildContext, Frame};
use crate::metrics::Metrics;
use crate::{Spec, params};
use comnum_core::rng::{Rng, fnv1a64, stream};
use comnum_core::{Error, Result};
use comnum_sim::clock::Stopwatch;
use comnum_sim::grid::Condition;
use comnum_sim::scenario::{Record, Scenario};

impl Spec {
    /// Build the link of a condition: substitute the variables, build the blocks, train
    /// the blocks that are trained once, parse the metrics.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid block or metric; the errors of a training.
    pub fn link_for(&self, cond: &Condition) -> Result<Link> {
        let ctx = BuildContext {
            fiber: self.fiber.clone(),
            base_dir: self.base_dir.clone(),
            name: Some(self.name.clone()),
            condition: Some(cond.label()),
        };
        let mut blocks: Vec<Box<dyn Block>> = Vec::with_capacity(self.link.len());
        let mut kinds = Vec::with_capacity(self.link.len());
        let mut labels = Vec::with_capacity(self.link.len());
        let mut parts = Vec::with_capacity(self.link.len());
        for b in &self.link {
            let (kind, label, p) = params::split_block(&params::substitute(b, cond)?)?;
            labels.push(label.clone().unwrap_or_else(|| kind.clone()));
            parts.push((kind, label, p));
        }
        // a block without a `label` is built with its unique name: two blocks of a type
        // record their values under `probe_3` and `probe_5`, not both under `probe` (a
        // `label` is kept as it is written: a probe may share that of the block it follows)
        let labels = number_labels(&labels);
        for (i, ((kind, given, p), numbered)) in parts.into_iter().zip(&labels).enumerate() {
            let label = given.unwrap_or_else(|| numbered.clone());
            let block = blocks::build_parts(kind.clone(), Some(label), p, &ctx)
                .map_err(|e| e.context(format!("link block {}", i + 1)))?;
            blocks.push(block);
            kinds.push(kind);
        }
        blocks::with_registry(|registry| {
            if registry.iter().any(|b| b.starts && b.name == kinds[0]) {
                return Ok(());
            }
            let starts: Vec<&str> = registry.iter().filter(|b| b.starts).map(|b| b.name).collect();
            Err(Error::Config(format!("the first block of the link must be one of {starts:?}")))
        })?;
        blocks::check_link(&kinds.iter().map(String::as_str).collect::<Vec<_>>())?;
        for i in 0..blocks.len() {
            if blocks[i].needs_training() {
                let mut rng = self.training_rng(cond, i);
                let mut s = Frame::default();
                for b in &blocks[..i] {
                    b.apply(&mut s, &mut rng)?;
                }
                blocks[i].train(&s, &mut rng)?;
            }
        }
        let metrics = Metrics::from_json(&params::substitute(&self.metrics, cond)?)?;
        Ok(Link { blocks, kinds, labels, metrics, profile: self.profile })
    }

    /// The signals of the `scope` blocks in trial `trial` of the condition of index
    /// `condition`: the frame that trial of a campaign gives (the same random stream,
    /// `(seed, name, condition, trial)`), its traces in the order of the link.
    ///
    /// # Errors
    /// [`Error::Config`] for a condition out of the grid; the errors of the link.
    ///
    /// # Examples
    /// ```
    /// let spec = comnum_spec::Spec::from_yaml(
    ///     "name: t\nlink: [{type: source, n_symbols: 64}, {type: awgn, snr_db: 20}, scope]\nstop: {max_trials: 1}",
    /// )
    /// .unwrap();
    /// let traces = spec.trace(0, 0).unwrap();
    /// assert_eq!((traces[0].label.as_str(), traces[0].streams[0].len()), ("scope", 64));
    /// ```
    pub fn trace(&self, condition: usize, trial: u64) -> Result<Vec<blocks::analysis::ScopeTrace>> {
        let conds = self.conditions();
        let cond = conds
            .get(condition)
            .ok_or_else(|| Error::Config(format!("condition {condition} out of the {} of the grid", conds.len())))?;
        let link = self.link_for(cond)?;
        let mut rng = stream(self.seed, &self.name, condition as u64, trial);
        let mut rec = Record::for_trial(trial);
        let s = link.trace(&mut rng, &mut rec)?;
        Ok(s.traces.unwrap_or_default())
    }

    fn training_rng(&self, cond: &Condition, block: usize) -> Rng {
        stream(self.seed, &format!("{}/train", self.name), fnv1a64(cond.label().as_bytes()), block as u64)
    }
}

/// The labels of the blocks of a link, as the profile names them (and the Python chains
/// their blocks): the `label` of a block, else its type, numbered by [`number_labels`].
///
/// # Errors
/// [`Error::Config`] for a block that is neither a name nor a mapping with a type.
pub fn link_labels(link: &[serde_json::Value]) -> Result<Vec<String>> {
    let labels = link
        .iter()
        .map(|b| params::split_block(b).map(|(kind, label, _)| label.unwrap_or(kind)))
        .collect::<Result<Vec<_>>>()?;
    Ok(number_labels(&labels))
}

/// Unique labels: a repeated label takes the position of its block, counted from 1
/// (`[source, awgn, awgn]` gives `[source, awgn_2, awgn_3]`).
pub fn number_labels(labels: &[String]) -> Vec<String> {
    labels
        .iter()
        .enumerate()
        .map(|(i, l)| if labels.iter().filter(|g| *g == l).count() > 1 { format!("{l}_{}", i + 1) } else { l.clone() })
        .collect()
}

/// The link of one condition: its blocks and its metrics.
pub struct Link {
    /// The blocks, in order.
    pub blocks: Vec<Box<dyn Block>>,
    /// The type of every block (the context of its errors).
    pub kinds: Vec<String>,
    /// The label of every block (its `label`, else its type; a repeated label takes
    /// the position of the block, `awgn_3`), the names of the profile.
    pub labels: Vec<String>,
    /// The metrics.
    pub metrics: Metrics,
    /// Time every block: the trial records `<label>.time_ms` for every block and
    /// `link.time_ms` for the whole link (the metrics excluded), reported as means.
    pub profile: bool,
}

impl Link {
    /// Run one trial: the blocks in order, then the metrics and the recorded values.
    ///
    /// # Errors
    /// The first error of a block or of the metrics.
    pub fn trial(&self, rng: &mut Rng, rec: &mut Record) -> Result<Frame> {
        self.run_frame(Frame { trial: rec.trial(), ..Frame::default() }, rng, rec)
    }

    /// Run one trial collecting the signals of its `scope` blocks: the frame of
    /// [`Link::trial`], with its traces.
    ///
    /// # Errors
    /// As [`Link::trial`].
    pub fn trace(&self, rng: &mut Rng, rec: &mut Record) -> Result<Frame> {
        self.run_frame(Frame { trial: rec.trial(), traces: Some(Vec::new()), ..Frame::default() }, rng, rec)
    }

    fn run_frame(&self, mut s: Frame, rng: &mut Rng, rec: &mut Record) -> Result<Frame> {
        let t_link = Stopwatch::start();
        for (i, (b, kind)) in self.blocks.iter().zip(&self.kinds).enumerate() {
            let t = Stopwatch::start();
            b.apply(&mut s, rng).map_err(|e| e.context(format!("link block {} ({kind})", i + 1)))?;
            if self.profile {
                rec.value(&format!("{}.time_ms", self.labels[i]), t.elapsed().as_secs_f64() * 1e3);
            }
        }
        if self.profile {
            rec.value("link.time_ms", t_link.elapsed().as_secs_f64() * 1e3);
        }
        self.metrics.record_frame(&s, rec)?;
        Ok(s)
    }
}

/// A specification as a [`Scenario`] of the runner.
pub struct SpecScenario<'a> {
    spec: &'a Spec,
    profile: bool,
}

impl<'a> SpecScenario<'a> {
    /// The scenario of a specification.
    pub fn new(spec: &'a Spec) -> SpecScenario<'a> {
        SpecScenario { spec, profile: spec.profile }
    }

    /// Time every block, whatever the specification says.
    #[must_use]
    pub fn profile(mut self, profile: bool) -> Self {
        self.profile = profile;
        self
    }
}

impl Scenario for SpecScenario<'_> {
    type State = Link;

    fn name(&self) -> &str {
        &self.spec.name
    }

    fn setup(&self, condition: &Condition) -> Result<Link> {
        let mut link = self.spec.link_for(condition)?;
        link.profile = self.profile;
        Ok(link)
    }

    fn trial(&self, link: &Link, _: &Condition, rng: &mut Rng, record: &mut Record) -> Result<()> {
        link.trial(rng, record).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_labels_take_the_position_of_the_block() {
        let spec = Spec::from_yaml(
            "name: twice\nlink: [{type: source, n_symbols: 64}, {type: awgn, snr_db: 10}, {type: awgn, snr_db: 10}, {type: awgn, snr_db: 10, label: last}]\nstop: {max_trials: 1}",
        )
        .unwrap();
        let link = spec.link_for(&spec.conditions()[0]).unwrap();
        assert_eq!(link.labels, ["source", "awgn_2", "awgn_3", "last"]);
    }

    #[test]
    fn a_trace_is_the_trial_of_the_campaign_and_a_scope_changes_nothing() {
        let text = "name: scoped\nseed: 4\ngrid: {snr_db: [5, 15]}\nlink: [{type: source, constellation: {kind: qam, m: 16}, n_symbols: 512}, {type: scope, label: tx}, {type: awgn, snr_db: $snr_db}, {type: scope, label: rx}, {type: scope, view: spectrum, nfft: 64}]\nmetrics: {report: [ser]}\nstop: {max_trials: 1}";
        let spec = Spec::from_yaml(text).unwrap();
        let traces = spec.trace(1, 0).unwrap();
        let labels: Vec<_> = traces.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["tx", "rx", "scope"]);
        // the samples of the scope after the noise are those of the trial of the campaign
        let cond = &spec.conditions()[1];
        let link = spec.link_for(cond).unwrap();
        let mut rng = stream(spec.seed, &spec.name, 1, 0);
        let s = link.trial(&mut rng, &mut Record::for_trial(0)).unwrap();
        assert_eq!(traces[1].streams[0], s.samples);
        assert!(s.traces.is_none());
        // the transmitted points are those of the constellation, the received ones are not
        let c = comnum_mapping::constellation::Constellation::from_name("16-QAM").unwrap().normalized();
        assert!(traces[0].streams[0].iter().all(|z| c.points().iter().any(|p| (z - p).norm() < 1e-12)));
        assert_ne!(traces[1].streams[0], traces[0].streams[0]);
        // the spectrum: 64 bins, positive, of one stream
        let (f, d) = traces[2].spectrum.as_ref().unwrap();
        assert_eq!((f.len(), d.len(), d[0].len()), (64, 1, 64));
        assert!(d[0].iter().all(|&v| v > 0.0));
        // the same specification without its scopes gives the same metrics
        let plain = Spec::from_yaml(
            &text
                .replace(", {type: scope, label: tx}", "")
                .replace(", {type: scope, label: rx}, {type: scope, view: spectrum, nfft: 64}", ""),
        )
        .unwrap();
        let r = crate::run(&spec, &crate::RunOptions::default()).unwrap();
        let q = crate::run(&plain, &crate::RunOptions::default()).unwrap();
        let ser = |row: Vec<(String, comnum_sim::grid::Value)>| {
            row.into_iter().filter(|(k, _)| k.starts_with("ser")).collect::<Vec<_>>()
        };
        assert_eq!(ser(r[1].row(0.95)), ser(q[1].row(0.95)));
        assert!(ser(r[1].row(0.95)).iter().any(|(k, _)| k == "ser"));
        assert!(spec.trace(2, 0).is_err());
    }

    #[test]
    fn two_blocks_of_a_type_without_label_record_under_their_own_names() {
        // 20 dB after the first noise, about 2 dB after the second: one metric each (they
        // were merged into one `probe.snr_db`, the mean of the two)
        let spec = Spec::from_yaml(
            "name: t\nseed: 1\nstop: {max_trials: 4}\nlink:\n  - {type: source, constellation: QPSK, n_symbols: 2000}\n  - {type: awgn, snr_db: 20}\n  - {type: probe, metrics: [snr_db]}\n  - {type: awgn, snr_db: 0}\n  - {type: probe, metrics: [snr_db]}\n",
        )
        .unwrap();
        let r = crate::run(&spec, &crate::RunOptions::default()).unwrap();
        let row: std::collections::BTreeMap<String, f64> = r[0]
            .row(0.95)
            .into_iter()
            .filter_map(|(k, v)| match v {
                comnum_sim::grid::Value::F64(x) => Some((k, x)),
                _ => None,
            })
            .collect();
        assert!(!row.contains_key("probe.snr_db"), "{row:?}");
        assert!((row["probe_3.snr_db"] - 20.0).abs() < 0.5, "{row:?}");
        assert!(row["probe_5.snr_db"] < 4.0, "{row:?}");
    }

    #[test]
    fn errors_name_their_place() {
        let e = Spec::from_yaml("name: t\nlink: [source]\nseeds: 3").unwrap_err().to_string();
        assert!(e.contains("seeds"), "{e}");
        let spec = Spec::from_yaml("name: t\nlink: [awgn_ebn0]").unwrap();
        let e = spec.link_for(&Condition::new()).err().unwrap().to_string();
        assert!(e.contains("link block 1"), "{e}");
        let spec = Spec::from_yaml("name: t\nlink: [{type: awgn_ebn0, ebn0_db: 3}, source]").unwrap();
        assert!(spec.link_for(&Condition::new()).err().unwrap().to_string().contains("first block"));
        let spec = Spec::from_yaml("name: t\nlink: [source]\nmetrics: {report: [fer]}").unwrap();
        assert!(spec.link_for(&Condition::new()).err().unwrap().to_string().contains("fer"));
    }
}
