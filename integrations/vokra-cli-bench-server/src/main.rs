//! `vokra-cli-bench-server` binary — thin driver over the
//! [`vokra_cli_bench_server`] library.
//!
//! Responsibilities:
//! 1. Parse argv via [`vokra_cli_bench_server::parse_args`].
//! 2. Print [`vokra_cli_bench_server::cli::USAGE`] on `--help` and exit 0.
//! 3. Run the bench via [`vokra_cli_bench_server::run_bench`].
//! 4. Emit KV or JSON per `--format`.
//! 5. Exit with the right code (`ExitCode`).
//!
//! Everything else lives in the library so it can be exercised from
//! `tests/e2e.rs` without spawning a subprocess.

use std::io::{self, Write};
use std::process::ExitCode as ProcessExitCode;

use vokra_cli_bench_server::{
    ExitCode, OutputFormat, ParseError, cli::USAGE, emit_json, emit_kv, parse_args, run_bench,
};

fn main() -> ProcessExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(argv) {
        Ok(a) => a,
        Err(ParseError::Help) => {
            print!("{USAGE}");
            return ProcessExitCode::from(ExitCode::Ok as u8);
        }
        Err(e) => {
            let mut stderr = io::stderr().lock();
            let _ = writeln!(stderr, "vokra-cli-bench-server: {e}");
            let _ = writeln!(stderr, "run with `--help` for usage");
            return ProcessExitCode::from(ExitCode::BadArgs as u8);
        }
    };

    // Run the measurement window. `run_bench` returns Err only on
    // URL-parse failure (structural — `--server` is malformed); actual
    // transport failures during the window are captured in the summary
    // counters (FR-EX-08: no silent fallback, we still exit non-zero
    // if EVERY request failed the wire, see below).
    let summary = match run_bench(&args) {
        Ok(s) => s,
        Err(msg) => {
            let mut stderr = io::stderr().lock();
            let _ = writeln!(stderr, "vokra-cli-bench-server: {msg}");
            return ProcessExitCode::from(ExitCode::BadArgs as u8);
        }
    };

    let stdout = io::stdout();
    let mut w = stdout.lock();
    let emit_result = match args.format {
        OutputFormat::Kv => emit_kv(&mut w, &args, &summary),
        OutputFormat::Json => emit_json(&mut w, &args, &summary),
    };
    if let Err(e) = emit_result {
        // Broken pipe on stdout — rare but real (piping to `head`).
        let _ = writeln!(
            io::stderr().lock(),
            "vokra-cli-bench-server: emit failed: {e}"
        );
        return ProcessExitCode::from(ExitCode::BadArgs as u8);
    }

    // Every request in the window failed the wire → exit 3. NO
    // fallback verdict — the artifact still emitted above will show
    // `counters.transport_errors == iterations` and `verdict=NO_SUCCESS`
    // for the operator, but the shell exit code makes CI catch it
    // without parsing stdout (FR-EX-08).
    if summary.ok_2xx == 0
        && summary.over_capacity_503 == 0
        && summary.rate_limited_429 == 0
        && summary.client_error_4xx == 0
        && summary.server_error_5xx == 0
        && summary.transport_errors > 0
    {
        return ProcessExitCode::from(ExitCode::AllTransportFailed as u8);
    }

    ProcessExitCode::from(ExitCode::Ok as u8)
}
