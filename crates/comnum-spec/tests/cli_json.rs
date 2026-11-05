//! `comnum --json`: the output for programs and agents, from the binary itself.
#![allow(clippy::unwrap_used, reason = "a test")]

use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};

const SPEC: &str = "name: qpsk
description: QPSK over AWGN
grid: {ebn0_db: [0, 8]}
link:
  - {type: source, constellation: qpsk, n_symbols: 500}
  - {type: awgn_ebn0, ebn0_db: $ebn0_db}
  - {type: demap}
metrics: {report: [ber]}
stop: {max_trials: 20}
";

/// Run `comnum ARGS` with `stdin`, return the exit code and the JSON lines it printed.
fn comnum(args: &[&str], stdin: &str) -> (i32, Vec<Value>) {
    let dir = std::env::temp_dir().join(format!("comnum-cli-json-{}", std::process::id()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_comnum"))
        .args(args)
        .args(["--out", dir.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let lines = String::from_utf8(out.stdout).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    (out.status.code().unwrap(), lines)
}

#[test]
fn run_json_prints_the_metrics_their_intervals_and_the_provenance() {
    let (code, lines) = comnum(&["run", "-", "--json"], SPEC);
    assert_eq!((code, lines.len()), (0, 1));
    let d = &lines[0];
    assert_eq!((d["ok"].as_bool(), d["spec"].as_str()), (Some(true), Some("qpsk")));
    assert_eq!(d["budget_s"], 300.0);
    assert_eq!(d["provenance"]["comnum"], env!("CARGO_PKG_VERSION"));
    assert_eq!(d["provenance"]["fingerprint"].as_str().unwrap().len(), 16);
    let c = &d["conditions"][0];
    assert_eq!((c["condition"]["ebn0_db"].as_i64(), c["trials"].as_u64()), (Some(0), Some(20)));
    assert_eq!(c["time_limited"], false);
    let ber = &c["metrics"]["ber"];
    let (v, lo, hi) = (ber["value"].as_f64().unwrap(), ber["lo"].as_f64().unwrap(), ber["hi"].as_f64().unwrap());
    // QPSK at 0 dB: 0.0786 by the closed form
    assert!(lo < v && v < hi && (v - 0.0786).abs() < 0.01, "{ber}");
}

#[test]
fn a_misspelt_block_is_a_json_error_with_the_closest_name() {
    let (code, lines) = comnum(&["check", "-", "--json"], &SPEC.replace("awgn_ebn0", "awgn_ebno"));
    assert_eq!(code, 1);
    let e = &lines[0]["error"];
    assert_eq!((lines[0]["ok"].as_bool(), e["kind"].as_str()), (Some(false), Some("config")));
    assert_eq!(e["suggestion"], "type: awgn_ebn0");
    let (code, lines) = comnum(&["run", "-", "--json", "--bogus"], SPEC);
    assert_eq!((code, lines[0]["error"]["kind"].as_str()), (2, Some("usage")));
}

#[test]
fn the_budget_cuts_a_long_run_and_says_so() {
    let (code, lines) = comnum(
        &["run", "-", "--json", "--max-seconds", "0.4"],
        &SPEC.replace("max_trials: 20", "max_trials: 100000000"),
    );
    assert_eq!(code, 0);
    for c in lines[0]["conditions"].as_array().unwrap() {
        assert_eq!(c["time_limited"], true);
        assert!(c["trials"].as_u64().unwrap() < 100_000_000 && c["wall_s"].as_f64().unwrap() < 2.0, "{c}");
    }
}

#[test]
fn every_specification_of_the_guide_passes_check() {
    let guide = include_str!("../GUIDE_AGENTS.md");
    let specs: Vec<&str> = guide.split("```yaml\n").skip(1).map(|b| b.split("```").next().unwrap()).collect();
    assert_eq!(specs.len(), 2);
    for spec in specs {
        let (code, lines) = comnum(&["check", "-", "--json"], spec);
        assert_eq!((code, lines[0]["ok"].as_bool()), (0, Some(true)), "{spec}\n{}", lines[0]);
    }
}

#[test]
fn guide_prints_the_guide() {
    let out = Command::new(env!("CARGO_BIN_EXE_comnum")).arg("guide").output().unwrap();
    assert!(
        out.status.success() && String::from_utf8(out.stdout).unwrap().starts_with("# comnum for programs and agents")
    );
}
