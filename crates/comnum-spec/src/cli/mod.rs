//! The command-line tool `comnum`: run declarative Monte Carlo campaigns.
//!
//! ```text
//! comnum run SPEC... [--out DIR] [--threads N] [--fast-dev-run] [--profile] [--quiet] [--json] [--max-seconds S]
//! comnum check SPEC... [--json]
//! comnum verify SPEC... [--out DIR] [--threads N] [--quiet] [--json] [--max-seconds S]
//! comnum conditions SPEC
//! comnum blocks [NAME...] [--json]
//! comnum schema
//! comnum guide
//! ```
//!
//! `run` writes the CSV and JSON reports given by the specification, or
//! `DIR/<name>.csv` and `DIR/<name>.json` (`DIR` = `results` by default), and prints
//! every metric with its confidence interval (for a specification with a `search`, the
//! operating point of every combination of the other axes); `--profile` adds the mean
//! time of every block ([`profile`](crate::profile)). `check` builds the link of every
//! condition and runs one trial of each, without writing anything (for a specification
//! with a `search`, at the lower end of its range for every combination of the other
//! axes). `verify` runs the specifications and checks their expected results (closed
//! forms, published curves), prints every check, writes the reports of the run and
//! `DIR/<name>.verify.json`, and fails if a check fails. `conditions` lists the
//! conditions of the grid; `blocks` documents the blocks of the links (their parameters
//! with types, units and accepted values, and their ports; as JSON with `--json`);
//! `schema` prints the JSON Schema of a block of a link, for editors and graphical
//! front ends; `guide` prints the guide for programs and agents
//! (`GUIDE_AGENTS.md` of the crate).
//!
//! For programs and agents ([`agent`]): `--json` prints one compact JSON document per
//! specification on standard output (the results with their confidence intervals and
//! their provenance, the checks of `verify`) and the errors as JSON with a kind, a hint
//! and a suggestion for a misspelt block or parameter; `--max-seconds S` bounds the
//! wall time of a run (300 s by default with `--json`, 0 for none), spread over its
//! conditions, and marks the conditions it cut short (`time_limited`: their results
//! then depend on the machine). A specification `-` is read from standard input (YAML
//! or JSON).
//!
//! The binary `comnum` of this crate is [`main`] on the arguments of the process; a crate
//! that adds blocks ([`blocks::register`]) wraps the same function in its own binary,
//! so that its command lists, checks and runs them with the others.

use crate::{Format, RunOptions, Spec, blocks, run, search, verify};
use comnum_sim::stopping::Stop;
use serde_json::json;
use std::path::PathBuf;
use std::process::ExitCode;

pub mod agent;

const USAGE: &str = "usage:
  comnum run SPEC... [--out DIR] [--threads N] [--fast-dev-run] [--profile] [--quiet] [--json] [--max-seconds S]
  comnum check SPEC... [--json]
  comnum verify SPEC... [--out DIR] [--threads N] [--quiet] [--json] [--max-seconds S]
  comnum conditions SPEC
  comnum blocks [NAME...] [--json]
  comnum schema
  comnum guide

SPEC is a YAML, JSON or TOML specification (see the documentation of the crate comnum-spec), or -
for standard input. --json: one JSON document per specification on standard output, for programs
and agents (a time budget of 300 s unless --max-seconds is given; --max-seconds 0 for none).";

#[allow(clippy::struct_excessive_bools, reason = "the flags of the command line")]
struct Args {
    command: String,
    files: Vec<PathBuf>,
    out: PathBuf,
    threads: Option<usize>,
    fast_dev_run: bool,
    profile: bool,
    quiet: bool,
    json: bool,
    max_seconds: Option<f64>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut it = args.into_iter();
    let command = it.next().ok_or_else(|| USAGE.to_string())?;
    let mut a = Args {
        command,
        files: Vec::new(),
        out: PathBuf::from("results"),
        threads: None,
        fast_dev_run: false,
        profile: false,
        quiet: false,
        json: false,
        max_seconds: None,
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => a.out = PathBuf::from(it.next().ok_or("--out needs a directory")?),
            "--threads" => {
                let n = it.next().ok_or("--threads needs a number")?;
                a.threads = Some(n.parse().map_err(|_| format!("--threads: {n:?} is not a number"))?);
            }
            "--fast-dev-run" => a.fast_dev_run = true,
            "--profile" => a.profile = true,
            "--quiet" => a.quiet = true,
            "--json" => a.json = true,
            "--max-seconds" => {
                let s = it.next().ok_or("--max-seconds needs a number of seconds")?;
                let v: f64 = s.parse().map_err(|_| format!("--max-seconds: {s:?} is not a number"))?;
                if !(v.is_finite() && v >= 0.0) {
                    return Err(format!("--max-seconds: {s} must be a number of seconds >= 0"));
                }
                a.max_seconds = Some(v);
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n\n{USAGE}")),
            "-" => a.files.push(PathBuf::from("-")),
            s => a.files.push(PathBuf::from(s)),
        }
    }
    Ok(a)
}

/// The options of `run` and `verify`: the threads, the progress and the reports, those
/// of the specification or `DIR/<name>.csv` and `DIR/<name>.json`.
fn report_options(spec: &Spec, a: &Args) -> RunOptions {
    RunOptions {
        threads: a.threads,
        progress: Some(!a.quiet && !a.json),
        csv: Some(spec.csv.clone().unwrap_or_else(|| a.out.join(format!("{}.csv", spec.name)))),
        json: Some(spec.json.clone().unwrap_or_else(|| a.out.join(format!("{}.json", spec.name)))),
        ..RunOptions::default()
    }
}

/// Say where the reports went, unless `--quiet`.
fn written(opts: &RunOptions, a: &Args) {
    if let (false, false, Some(csv), Some(json)) = (a.quiet, a.json, &opts.csv, &opts.json) {
        eprintln!("written {} and {}", csv.display(), json.display());
    }
}

/// Read a specification from a file, or from standard input for `-`.
fn load(path: &PathBuf) -> comnum_core::Result<Spec> {
    if path.as_os_str() == "-" {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
        return Spec::parse(&text, Format::Yaml).map_err(|e| e.context("standard input"));
    }
    Spec::from_path(path).map_err(|e| match e {
        comnum_core::Error::Io(io) => {
            comnum_core::Error::Io(std::io::Error::new(io.kind(), format!("{}: {io}", path.display())))
        }
        other => other,
    })
}

/// The specification with the time budget of the options, and the rule of the
/// specification when a budget applies.
fn budgeted(path: &PathBuf, a: &Args) -> comnum_core::Result<(Spec, Option<Stop>)> {
    let mut spec = load(path)?;
    let original = agent::budget(a.max_seconds, a.json).map(|s| agent::apply_budget(&mut spec, s));
    Ok((spec, original))
}

fn run_one(path: &PathBuf, a: &Args) -> comnum_core::Result<()> {
    let (spec, original) = budgeted(path, a)?;
    let opts =
        RunOptions { fast_dev_run: a.fast_dev_run, profile: a.profile.then_some(true), ..report_options(&spec, a) };
    let started = comnum_sim::clock::Stopwatch::start();
    if a.json {
        let mut doc = json!({
            "ok": true,
            "command": "run",
            "spec": spec.name,
            "description": spec.description,
            "provenance": agent::provenance(&spec, &opts),
            "budget_s": agent::budget(a.max_seconds, a.json),
        });
        if spec.search.is_some() {
            let found = search::search(&spec, &opts)?;
            doc["operating_points"] = found.iter().map(agent::operating_point).collect();
        } else {
            let results = run(&spec, &opts)?;
            doc["conditions"] =
                results.iter().map(|r| agent::condition(r, spec.confidence, original.as_ref())).collect();
        }
        doc["wall_s"] = json!(started.elapsed().as_secs_f64());
        println!("{doc}");
        return Ok(());
    }
    if let Some(sp) = &spec.search {
        if !a.quiet {
            eprintln!(
                "{}: {} (search of {} for {} = {:e})",
                spec.name, spec.description, sp.along, sp.metric, sp.target
            );
        }
        for r in search::search(&spec, &opts)? {
            println!(
                "{} | {} = {:.4} [{:.4}, {:.4}] | {} evaluations{}",
                r.condition.label(),
                r.along,
                r.value,
                r.lo,
                r.hi,
                r.evaluations.len(),
                if r.converged { "" } else { " (not converged)" }
            );
        }
        written(&opts, a);
        return Ok(());
    }
    if !a.quiet {
        eprintln!("{}: {} ({} conditions)", spec.name, spec.description, spec.conditions().len());
    }
    let results = run(&spec, &opts)?;
    for r in &results {
        let metrics: Vec<String> = r
            .record
            .iter()
            .map(|(k, e)| {
                let (lo, hi) = e.interval(spec.confidence);
                format!("{k}={:.4e} [{lo:.3e}, {hi:.3e}]", e.value())
            })
            .collect();
        println!("{} | {} | {} trials | {}", r.condition.label(), r.status.as_str(), r.trials, metrics.join("  "));
    }
    let profile = crate::profile_of(&spec, &results)?;
    if !profile.is_empty() {
        println!("\nprofile: mean time per trial over {} conditions", results.len());
        for row in &profile {
            println!("{:>24}  {:>10.3} ms  {:>5.1} %", row.label, row.time_ms, 100.0 * row.share);
        }
    }
    written(&opts, a);
    Ok(())
}

/// Run a specification and check its expectations; returns the number of failures.
fn verify_one(path: &PathBuf, a: &Args) -> comnum_core::Result<usize> {
    let (spec, _) = budgeted(path, a)?;
    // the reports of the run, as with `run`
    let opts = report_options(&spec, a);
    let (_, checks) = verify::verify_spec(&spec, &opts)?;
    let failed = checks.iter().filter(|c| !c.passed).count();
    let report = a.out.join(format!("{}.verify.json", spec.name));
    verify::write_report(&report, &spec, &opts, &checks)?;
    if a.json {
        let doc = json!({
            "ok": failed == 0,
            "command": "verify",
            "spec": spec.name,
            "provenance": agent::provenance(&spec, &opts),
            "budget_s": agent::budget(a.max_seconds, a.json),
            "checks": checks.iter().map(agent::check_json).collect::<Vec<_>>(),
            "failed": failed,
            "report": report.display().to_string(),
        });
        println!("{doc}");
        return Ok(failed);
    }
    println!("{}: {} checks, {} failed", spec.name, checks.len(), failed);
    for c in &checks {
        println!(
            "  {} {} | {} | {}: expected {:.4e}, measured {:.4e} [{:.3e}, {:.3e}]",
            if c.passed { "ok  " } else { "FAIL" },
            c.name,
            c.condition,
            c.metric,
            c.expected,
            c.measured.value,
            c.measured.lo,
            c.measured.hi
        );
    }
    Ok(failed)
}

fn check_one(path: &PathBuf, json_out: bool) -> comnum_core::Result<()> {
    let spec = load(path)?;
    // a search: the first point it evaluates, for every combination of the other axes
    let mut spec = search::first_points(&spec).unwrap_or(spec);
    // nothing is written
    (spec.csv, spec.json) = (None, None);
    let opts = RunOptions { fast_dev_run: true, batch: Some(1), progress: Some(false), ..RunOptions::default() };
    let r = run(&spec, &opts)?;
    let names: Vec<&String> = r.first().map(|c| c.record.iter().map(|(k, _)| k).collect()).unwrap_or_default();
    if json_out {
        let doc = json!({
            "ok": true,
            "command": "check",
            "spec": spec.name,
            "conditions": r.len(),
            "blocks": spec.link.len(),
            "metrics": names,
        });
        println!("{doc}");
    } else {
        println!("{}: ok, {} conditions, {} blocks, metrics {:?}", path.display(), r.len(), spec.link.len(), names);
    }
    Ok(())
}

/// Pretty JSON, or the error as a configuration error.
fn to_json(v: &impl serde::Serialize) -> comnum_core::Result<String> {
    serde_json::to_string_pretty(v).map_err(|e| comnum_core::Error::Config(format!("cannot serialise the report: {e}")))
}

/// Run the command line on its arguments (without the program name); the exit code is
/// 0 on success, 1 on a failure and 2 on a usage error.
pub fn main(args: impl IntoIterator<Item = String>) -> ExitCode {
    let args: Vec<String> = args.into_iter().collect();
    let json_out = args.iter().any(|s| s == "--json");
    let usage = |msg: &str| {
        if json_out {
            let first = msg.lines().next().unwrap_or(msg);
            println!("{}", json!({"ok": false, "command": args.first(), "error": agent::error("usage", first)}));
        } else {
            eprintln!("{msg}");
        }
        ExitCode::from(2)
    };
    let a = match parse_args(args.clone()) {
        Ok(a) => a,
        Err(msg) => return usage(&msg),
    };
    if ["run", "check", "verify", "conditions"].contains(&a.command.as_str()) && a.files.is_empty() {
        return usage(&format!("no specification given\n\n{USAGE}"));
    }
    if a.json && ["run", "check", "verify"].contains(&a.command.as_str()) {
        return main_json(&a);
    }
    let result: comnum_core::Result<()> = match a.command.as_str() {
        "blocks" => {
            let all = blocks::registry();
            let wanted: Vec<String> = a.files.iter().map(|f| f.display().to_string()).collect();
            let shown: Vec<_> =
                all.iter().filter(|b| wanted.is_empty() || wanted.iter().any(|w| w == b.name)).collect();
            if shown.is_empty() {
                Err(comnum_core::Error::Config(format!("no block {wanted:?}")))
            } else if a.json {
                let all: Vec<_> = shown.iter().map(|b| b.to_json()).collect();
                to_json(&all).map(|j| println!("{j}"))
            } else {
                for b in shown {
                    println!("{}", b.describe());
                }
                Ok(())
            }
        }
        "schema" => to_json(&blocks::json_schema()).map(|j| println!("{j}")),
        "guide" => {
            print!("{}", include_str!("../../GUIDE_AGENTS.md"));
            Ok(())
        }
        "conditions" => a.files.iter().try_for_each(|f| {
            let spec = load(f)?;
            for (i, c) in spec.conditions().iter().enumerate() {
                println!("{i}: {}", c.label());
            }
            Ok(())
        }),
        "check" => a.files.iter().try_for_each(|f| check_one(f, a.json)),
        "run" => a.files.iter().try_for_each(|f| run_one(f, &a)),
        "verify" => {
            let mut failed = 0;
            let r = a.files.iter().try_for_each(|f| verify_one(f, &a).map(|n| failed += n));
            if r.is_ok() && failed > 0 {
                eprintln!("{failed} checks failed");
                return ExitCode::FAILURE;
            }
            r
        }
        other => return usage(&format!("unknown command {other:?}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `run`, `check` and `verify` with `--json`: one line per specification, an error as a
/// line of its own (the other specifications still run); the exit code is 1 if one
/// failed.
fn main_json(a: &Args) -> ExitCode {
    let mut failed = false;
    for f in &a.files {
        let r = match a.command.as_str() {
            "run" => run_one(f, a).map(|()| 0),
            "check" => check_one(f, true).map(|()| 0),
            _ => verify_one(f, a),
        };
        match r {
            Ok(n) => failed |= n > 0,
            Err(e) => {
                failed = true;
                let message = e.to_string();
                println!(
                    "{}",
                    json!({"ok": false, "command": a.command, "file": f.display().to_string(), "error": agent::error(agent::error_kind(&e), &message)})
                );
            }
        }
    }
    if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
