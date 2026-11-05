//! `comnum`: run declarative Monte Carlo campaigns (see `comnum_spec::cli`).

fn main() -> std::process::ExitCode {
    comnum_spec::cli::main(std::env::args().skip(1))
}
