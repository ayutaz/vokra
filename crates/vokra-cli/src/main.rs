//! `vokra-cli` — the Vokra umbrella command-line tool (M1-10a; FR-TL-*, NFR-PF-13).
//!
//! Subcommands:
//!
//! - `run` — load a GGUF via the public session/model APIs and run its task
//!   (VAD speech probabilities / ASR text / TTS audio);
//! - `convert` — delegate to the offline `vokra-convert` library
//!   (checkpoint → GGUF);
//! - `bench` — measure RTF / TTFA / jitter / p50-p95-p99 with `std::time` and,
//!   with `--baseline`, a >5% relative regression gate (NFR-PF-13).
//!
//! Argument parsing is hand-written (no clap/getopts — external deps are
//! forbidden, NFR-DS-02), mirroring `vokra-convert`. The whole crate is
//! std-only and depends only on first-party `vokra-*` crates.

mod bench;
mod convert;
mod engine;
mod report;
mod run;
mod wav;

use std::process::ExitCode;

const USAGE: &str = "\
vokra-cli — Vokra speech runtime CLI (M1-10a)

USAGE:
    vokra-cli <run|convert|bench> [options]

SUBCOMMANDS:
    run       load a GGUF and run its task (VAD probs / ASR text / TTS audio)
    convert   convert an upstream checkpoint to a Vokra GGUF (offline tool)
    bench     measure RTF / TTFA / jitter / p50-p95-p99, optional regression gate

Run `vokra-cli <subcommand> --help` for that subcommand's options.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((sub, rest)) = args.split_first() else {
        eprintln!("error: no subcommand given\n\n{USAGE}");
        return ExitCode::from(2);
    };

    let result = match sub.as_str() {
        "-h" | "--help" => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        "run" => run::main(rest),
        "convert" => convert::main(rest),
        "bench" => bench::main(rest),
        other => {
            eprintln!("error: unknown subcommand `{other}`\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    match result {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
