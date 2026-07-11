//! CLI flag parsing.
//!
//! Hand-rolled (no `clap` / `structopt`) — the zero-dep posture (see
//! `Cargo.toml`) means we cannot depend on a parser crate. The parser
//! accepts both `--flag value` and `--flag=value`; `--help` prints usage
//! to stdout and returns [`ParseError::Help`] so the caller can exit 0.
//!
//! Every default matches
//! [`docs/m3-15-server-latency-handover.md` § 4](../../../docs/m3-15-server-latency-handover.md)
//! byte-for-byte AND the sibling `integrations/vokra-server-bench/` binary
//! so operators can swap the two without re-learning the flag set.

use std::fmt;

/// Parsed CLI arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    /// Server base URL. Default `http://127.0.0.1:8080`. Only `http://`
    /// is accepted; `https://` fails at URL-parse time (pure-std has no
    /// TLS). See [`crate::http::parse_target`].
    pub server: String,
    /// URL path appended to `--server`. Default `/api/tts`. Any path is
    /// accepted; the M3-15 handover documents `/api/tts` (piper-plus
    /// HTTP) and `/v1/audio/transcriptions` (OpenAI Whisper compat)
    /// as the two v0.9 targets.
    pub endpoint: String,
    /// Reference utterance body. Default `"Hello world"` (matches the
    /// in-process bench `SHORT_UTTERANCE` in
    /// `integrations/vokra-server/benches/tts_latency.rs`).
    pub text: String,
    /// Piper voice tag included in the JSON body. Default
    /// `"en_US-libritts-high"` (matches the handover runbook).
    pub voice: String,
    /// Measurement iterations distributed across the concurrent
    /// workers. Default 100.
    pub iters: usize,
    /// Warm-up iterations discarded before the measurement window,
    /// always run single-threaded on the main thread to amortise
    /// DNS lookup + connect-timeout skew. Default 10.
    pub warmup: usize,
    /// Number of concurrent worker threads (== sustained in-flight
    /// requests). Default 1.
    pub concurrent: usize,
    /// Output format. Default [`OutputFormat::Kv`].
    pub format: OutputFormat,
    /// Latency budget the artifact echoes and the verdict is computed
    /// against. Default 75 (NFR-PF-05 v0.9 value); operators
    /// backporting to v0.5 should pass `--budget-ms 90`.
    pub budget_ms: u64,
    /// Per-request HTTP timeout in seconds. Default 30. Guards against
    /// a hung server dragging the bench to infinity.
    pub timeout_secs: u64,
}

impl Args {
    /// Field-typed defaults, so callers building `Args` in tests do
    /// not need to duplicate the values.
    #[must_use]
    pub fn defaults() -> Self {
        Self {
            server: "http://127.0.0.1:8080".to_owned(),
            endpoint: "/api/tts".to_owned(),
            text: "Hello world".to_owned(),
            voice: "en_US-libritts-high".to_owned(),
            iters: 100,
            warmup: 10,
            concurrent: 1,
            format: OutputFormat::Kv,
            budget_ms: 75,
            timeout_secs: 30,
        }
    }

    /// Fully-composed URL (`server + endpoint`, with `/`-normalisation).
    ///
    /// Kept as a method so unit tests can assert on it without hitting
    /// the network.
    #[must_use]
    pub fn full_url(&self) -> String {
        let base = self.server.trim_end_matches('/');
        if self.endpoint.starts_with('/') {
            format!("{base}{}", self.endpoint)
        } else {
            format!("{base}/{}", self.endpoint)
        }
    }
}

/// Output format switch for `--format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Line-per-key `key=value` (default; grep-friendly for shell
    /// pipelines and the M3-15 handover results-report template).
    Kv,
    /// Single-line JSON blob matching the handover schema § 4
    /// byte-for-byte.
    Json,
}

/// Process exit codes.
///
/// Every non-zero exit corresponds to a specific failure category so
/// CI + owner scripts can react without reading stderr. Mirrors
/// [`vokra_server_bench::cli::ExitCode`] byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitCode {
    /// Measurement window ran to completion (verdict may still be
    /// FAIL — the tool never gates the process exit on verdict).
    /// Also used for `--help`.
    Ok = 0,
    /// CLI flag / value parse error, or invalid value (e.g. negative
    /// iters, zero concurrent).
    BadArgs = 2,
    /// Every request in the measurement window failed with a
    /// transport error (server unreachable, DNS failure, connect
    /// refused). Distinct from HTTP 5xx which is captured in stats
    /// counters.
    AllTransportFailed = 3,
}

/// CLI parse error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// `--help` (or `-h`) requested.
    Help,
    /// Unknown flag like `--foo`.
    UnknownFlag(String),
    /// `--flag` with no value following.
    MissingValue(String),
    /// `--flag NUMERIC_STR` failed to parse.
    InvalidNumber {
        /// Flag name (e.g. `"--iters"`).
        flag: String,
        /// The raw value that failed.
        value: String,
    },
    /// `--concurrent 0` / `--iters 0` etc. — must be positive.
    NotPositive(String),
    /// `--format` value not in `{kv, json}`.
    UnknownFormat(String),
    /// A positional argument was supplied (all args are flag-based).
    UnexpectedPositional(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Help => write!(f, "help requested"),
            Self::UnknownFlag(s) => write!(f, "unknown flag: {s}"),
            Self::MissingValue(s) => write!(f, "flag {s} requires a value"),
            Self::InvalidNumber { flag, value } => {
                write!(
                    f,
                    "flag {flag}: could not parse `{value}` as a positive integer"
                )
            }
            Self::NotPositive(s) => write!(f, "flag {s} must be a positive integer (> 0)"),
            Self::UnknownFormat(s) => {
                write!(f, "unknown --format value `{s}` (expected `kv` or `json`)")
            }
            Self::UnexpectedPositional(s) => write!(f, "unexpected positional argument: {s}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Human-readable usage message written to stdout when `--help` is
/// requested or `main.rs` wants to print an error tip. Byte-stable so
/// tests can assert on substrings.
pub const USAGE: &str = "\
vokra-cli-bench-server — zero-dep HTTP-boundary TTS/ASR latency benchmark

USAGE:
    vokra-cli-bench-server [OPTIONS]

OPTIONS:
    --server URL         Server base URL, http:// only (default http://127.0.0.1:8080)
    --endpoint PATH      URL path appended to --server (default /api/tts)
    --text TEXT          Reference utterance body (default \"Hello world\")
    --voice NAME         Piper voice tag (default en_US-libritts-high)
    --iters N            Total measurement iterations (default 100)
    --warmup N           Warmup iterations, discarded (default 10)
    --concurrent N       Concurrent worker threads (default 1)
    --format kv|json     Output format (default kv)
    --budget-ms N        Latency budget in ms echoed into artifact (default 75)
    --timeout-secs N     Per-request timeout in seconds (default 30)
    -h, --help           Print this usage and exit 0

EXIT CODES:
    0  measurement window completed (verdict may be FAIL — see docs)
    2  bad CLI args
    3  all measurement requests failed with a transport error
       (see docs/m3-15-server-latency-handover.md § 4 Option C)

NOTES:
    * Zero third-party deps: this binary talks HTTP/1.1 over std::net::TcpStream
      and emits hand-crafted JSON. Its Cargo.lock contains only the crate itself.
    * `https://` is NOT supported (pure-std has no TLS). Use the sibling
      `vokra-server-bench` binary (integrations/vokra-server-bench/) if you
      need TLS: it uses ureq + rustls.
";

/// Parse the argv slice into an [`Args`].
///
/// The first element of `argv` is expected to be the program name (as
/// with `std::env::args()`) and is discarded.
///
/// # Errors
///
/// Returns [`ParseError`] on unknown flags, missing values, bad numeric
/// values, or `--help` (which callers should treat as a clean exit).
pub fn parse_args<I, S>(argv: I) -> Result<Args, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut iter = argv.into_iter().map(Into::into);
    // Discard argv[0] (program name).
    let _prog = iter.next();
    let mut args = Args::defaults();

    while let Some(raw) = iter.next() {
        // Support `--flag=value` in one token.
        let (flag, inline_val): (String, Option<String>) = if let Some(eq_at) = raw.find('=') {
            if raw.starts_with("--") {
                (raw[..eq_at].to_string(), Some(raw[eq_at + 1..].to_string()))
            } else {
                (raw.clone(), None)
            }
        } else {
            (raw.clone(), None)
        };

        // Helper: pop the next value token, honouring `--flag=value`.
        let take_value = |inline: Option<String>,
                          it: &mut dyn Iterator<Item = String>,
                          f: &str|
         -> Result<String, ParseError> {
            if let Some(v) = inline {
                Ok(v)
            } else {
                it.next()
                    .ok_or_else(|| ParseError::MissingValue(f.to_owned()))
            }
        };
        // Helper: parse a positive-integer value.
        let take_pos_int = |inline: Option<String>,
                            it: &mut dyn Iterator<Item = String>,
                            f: &str|
         -> Result<usize, ParseError> {
            let v = take_value(inline, it, f)?;
            let n: usize = v.parse().map_err(|_| ParseError::InvalidNumber {
                flag: f.to_owned(),
                value: v.clone(),
            })?;
            if n == 0 {
                return Err(ParseError::NotPositive(f.to_owned()));
            }
            Ok(n)
        };
        // Helper: parse a non-negative u64 for --budget-ms (0 allowed
        // = "no budget" tag, so verdict is a no-op).
        let take_nonneg_u64 = |inline: Option<String>,
                               it: &mut dyn Iterator<Item = String>,
                               f: &str|
         -> Result<u64, ParseError> {
            let v = take_value(inline, it, f)?;
            v.parse::<u64>().map_err(|_| ParseError::InvalidNumber {
                flag: f.to_owned(),
                value: v.clone(),
            })
        };

        match flag.as_str() {
            "-h" | "--help" => return Err(ParseError::Help),
            "--server" => {
                args.server = take_value(inline_val, &mut iter, "--server")?;
            }
            "--endpoint" => {
                args.endpoint = take_value(inline_val, &mut iter, "--endpoint")?;
            }
            "--text" => {
                args.text = take_value(inline_val, &mut iter, "--text")?;
            }
            "--voice" => {
                args.voice = take_value(inline_val, &mut iter, "--voice")?;
            }
            "--iters" => {
                args.iters = take_pos_int(inline_val, &mut iter, "--iters")?;
            }
            "--warmup" => {
                // Warmup 0 is legal (skip warmup entirely).
                let v = take_value(inline_val, &mut iter, "--warmup")?;
                args.warmup = v.parse().map_err(|_| ParseError::InvalidNumber {
                    flag: "--warmup".to_owned(),
                    value: v.clone(),
                })?;
            }
            "--concurrent" => {
                args.concurrent = take_pos_int(inline_val, &mut iter, "--concurrent")?;
            }
            "--format" => {
                let v = take_value(inline_val, &mut iter, "--format")?;
                args.format = match v.as_str() {
                    "kv" => OutputFormat::Kv,
                    "json" => OutputFormat::Json,
                    other => return Err(ParseError::UnknownFormat(other.to_owned())),
                };
            }
            "--budget-ms" => {
                args.budget_ms = take_nonneg_u64(inline_val, &mut iter, "--budget-ms")?;
            }
            "--timeout-secs" => {
                args.timeout_secs = take_pos_int(inline_val, &mut iter, "--timeout-secs")? as u64;
            }
            other if other.starts_with("--") => {
                return Err(ParseError::UnknownFlag(other.to_owned()));
            }
            other => {
                return Err(ParseError::UnexpectedPositional(other.to_owned()));
            }
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Result<Args, ParseError> {
        let mut with_prog = vec!["vokra-cli-bench-server".to_string()];
        with_prog.extend(argv.iter().map(|s| (*s).to_string()));
        parse_args(with_prog)
    }

    #[test]
    fn defaults_from_empty_argv() {
        let args = parse(&[]).unwrap();
        assert_eq!(args, Args::defaults());
    }

    #[test]
    fn parses_long_flags_space_separated() {
        let args = parse(&[
            "--server",
            "http://x:9",
            "--endpoint",
            "/foo",
            "--text",
            "hi",
            "--voice",
            "ja_JP-a-b",
            "--iters",
            "5",
            "--warmup",
            "2",
            "--concurrent",
            "3",
            "--format",
            "json",
            "--budget-ms",
            "42",
            "--timeout-secs",
            "10",
        ])
        .unwrap();
        assert_eq!(args.server, "http://x:9");
        assert_eq!(args.endpoint, "/foo");
        assert_eq!(args.text, "hi");
        assert_eq!(args.voice, "ja_JP-a-b");
        assert_eq!(args.iters, 5);
        assert_eq!(args.warmup, 2);
        assert_eq!(args.concurrent, 3);
        assert_eq!(args.format, OutputFormat::Json);
        assert_eq!(args.budget_ms, 42);
        assert_eq!(args.timeout_secs, 10);
    }

    #[test]
    fn parses_long_flags_equals_form() {
        let args = parse(&["--server=http://x:9", "--iters=5", "--format=json"]).unwrap();
        assert_eq!(args.server, "http://x:9");
        assert_eq!(args.iters, 5);
        assert_eq!(args.format, OutputFormat::Json);
    }

    #[test]
    fn help_flag_returns_help_error() {
        assert_eq!(parse(&["--help"]), Err(ParseError::Help));
        assert_eq!(parse(&["-h"]), Err(ParseError::Help));
    }

    #[test]
    fn unknown_flag_rejected() {
        assert_eq!(
            parse(&["--foo"]),
            Err(ParseError::UnknownFlag("--foo".to_string())),
        );
    }

    #[test]
    fn missing_value_rejected() {
        assert_eq!(
            parse(&["--server"]),
            Err(ParseError::MissingValue("--server".to_string())),
        );
    }

    #[test]
    fn bad_int_rejected() {
        assert_eq!(
            parse(&["--iters", "not-a-number"]),
            Err(ParseError::InvalidNumber {
                flag: "--iters".to_string(),
                value: "not-a-number".to_string(),
            }),
        );
    }

    #[test]
    fn zero_iters_rejected() {
        assert_eq!(
            parse(&["--iters", "0"]),
            Err(ParseError::NotPositive("--iters".to_string())),
        );
    }

    #[test]
    fn zero_concurrent_rejected() {
        assert_eq!(
            parse(&["--concurrent", "0"]),
            Err(ParseError::NotPositive("--concurrent".to_string())),
        );
    }

    #[test]
    fn warmup_zero_is_legal() {
        let args = parse(&["--warmup", "0"]).unwrap();
        assert_eq!(args.warmup, 0);
    }

    #[test]
    fn budget_zero_is_legal() {
        let args = parse(&["--budget-ms", "0"]).unwrap();
        assert_eq!(args.budget_ms, 0);
    }

    #[test]
    fn unknown_format_rejected() {
        assert_eq!(
            parse(&["--format", "xml"]),
            Err(ParseError::UnknownFormat("xml".to_string())),
        );
    }

    #[test]
    fn positional_rejected() {
        assert_eq!(
            parse(&["extra"]),
            Err(ParseError::UnexpectedPositional("extra".to_string())),
        );
    }

    #[test]
    fn full_url_normalises_slashes() {
        let mut a = Args::defaults();
        a.server = "http://x:9/".into();
        a.endpoint = "/api/tts".into();
        assert_eq!(a.full_url(), "http://x:9/api/tts");

        a.server = "http://x:9".into();
        a.endpoint = "api/tts".into();
        assert_eq!(a.full_url(), "http://x:9/api/tts");

        a.server = "http://x:9".into();
        a.endpoint = "/api/tts".into();
        assert_eq!(a.full_url(), "http://x:9/api/tts");
    }

    #[test]
    fn usage_mentions_all_flags() {
        for flag in [
            "--server",
            "--endpoint",
            "--text",
            "--voice",
            "--iters",
            "--warmup",
            "--concurrent",
            "--format",
            "--budget-ms",
            "--timeout-secs",
        ] {
            assert!(USAGE.contains(flag), "USAGE missing {flag}");
        }
    }
}
