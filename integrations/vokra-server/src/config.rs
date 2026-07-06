//! Startup configuration: CLI flags, environment variables, optional TOML.
//!
//! Precedence (highest first): CLI flag > env var > TOML file > built-in
//! default. The scope of T03 is only what the startup foundation needs:
//! bind addresses for HTTP + Wyoming, and the optional config-file path.
//! Model registry / backend selection land in T04.
//!
//! We hand-roll the CLI parser to keep the surface tiny — this crate's whole
//! dependency footprint is HTTP + serde; adding `clap` would nearly double
//! the compile time for no user-visible benefit at T03.

use std::net::SocketAddr;
use std::path::PathBuf;

/// All startup knobs T03 needs.
#[derive(Debug, Clone)]
pub struct Config {
    /// HTTP (OpenAI / vLLM / piper-plus) listener address.
    /// Default: `127.0.0.1:8080` (loopback — external exposure is opt-in per T20).
    pub http_bind: SocketAddr,
    /// Wyoming Protocol TCP listener address.
    /// Default: `127.0.0.1:10300` (Home Assistant's Wyoming standard port).
    pub wyoming_bind: SocketAddr,
    /// Optional path to a TOML config file. When set, missing file is an error.
    pub config_file: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            http_bind: "127.0.0.1:8080".parse().expect("valid default"),
            wyoming_bind: "127.0.0.1:10300".parse().expect("valid default"),
            config_file: None,
        }
    }
}

/// Config parsing errors — surfaced from `parse_args`.
#[derive(Debug)]
pub enum ConfigError {
    /// `--help` was requested; caller should print help text and exit 0.
    HelpRequested,
    /// `--version` was requested; caller should print version and exit 0.
    VersionRequested,
    /// An unknown flag was passed on the CLI.
    UnknownFlag(String),
    /// A flag was passed without its required value.
    MissingValue(String),
    /// A flag's value failed to parse (e.g. bind address).
    InvalidValue {
        flag: String,
        value: String,
        reason: String,
    },
    /// The TOML config file was requested but could not be read.
    ConfigFileRead { path: PathBuf, error: String },
    /// The TOML config file failed to parse.
    ConfigFileParse { path: PathBuf, error: String },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HelpRequested => f.write_str("help requested"),
            Self::VersionRequested => f.write_str("version requested"),
            Self::UnknownFlag(s) => write!(f, "unknown flag: {s}"),
            Self::MissingValue(s) => write!(f, "flag {s} requires a value"),
            Self::InvalidValue {
                flag,
                value,
                reason,
            } => {
                write!(f, "flag {flag} value {value:?} invalid: {reason}")
            }
            Self::ConfigFileRead { path, error } => {
                write!(f, "cannot read config file {}: {error}", path.display())
            }
            Self::ConfigFileParse { path, error } => {
                write!(f, "cannot parse config file {}: {error}", path.display())
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// The `--help` text emitted for `vokra-server --help`.
pub const HELP_TEXT: &str = "\
vokra-server — Vokra single-binary API server (v0.5 / M2-09).

USAGE:
    vokra-server [OPTIONS]

OPTIONS:
    --http-bind <ADDR>       HTTP (OpenAI/vLLM/piper-plus) listener.
                             Default: 127.0.0.1:8080
    --wyoming-bind <ADDR>    Wyoming Protocol TCP listener.
                             Default: 127.0.0.1:10300
    --config <PATH>          Optional TOML config file (overrides defaults,
                             overridden by env and CLI flags).
    -h, --help               Print this help and exit.
    -V, --version            Print version and exit.

ENVIRONMENT:
    VOKRA_HTTP_BIND          Same as --http-bind.
    VOKRA_WYOMING_BIND       Same as --wyoming-bind.
    VOKRA_CONFIG             Same as --config.

Bind defaults are LOOPBACK. Expose to the network only via a reverse proxy
(nginx / Caddy) or by explicitly passing --http-bind 0.0.0.0:<port>.
See docs/tickets/m2/M2-09-vokra-server for the full scope.
";

/// Parse the CLI argument vector into a [`Config`]. `argv[0]` is treated as
/// the program name and ignored.
///
/// This is the T03 primitive tests exercise directly; production `main.rs`
/// calls it with `std::env::args`.
pub fn parse_args<I, S>(args: I) -> Result<Config, ConfigError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut it = args.into_iter().map(Into::into);
    let _program = it.next(); // discard argv[0]

    // Layer 1: built-in defaults.
    let mut cfg = Config::default();

    // Layer 2: env vars overlay (before CLI so CLI wins).
    if let Ok(v) = std::env::var("VOKRA_HTTP_BIND") {
        cfg.http_bind = parse_bind("VOKRA_HTTP_BIND", &v)?;
    }
    if let Ok(v) = std::env::var("VOKRA_WYOMING_BIND") {
        cfg.wyoming_bind = parse_bind("VOKRA_WYOMING_BIND", &v)?;
    }
    if let Ok(v) = std::env::var("VOKRA_CONFIG") {
        cfg.config_file = Some(PathBuf::from(v));
    }

    // Layer 3: CLI flags.
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(ConfigError::HelpRequested),
            "-V" | "--version" => return Err(ConfigError::VersionRequested),
            "--http-bind" => {
                let v = it
                    .next()
                    .ok_or_else(|| ConfigError::MissingValue("--http-bind".into()))?;
                cfg.http_bind = parse_bind("--http-bind", &v)?;
            }
            "--wyoming-bind" => {
                let v = it
                    .next()
                    .ok_or_else(|| ConfigError::MissingValue("--wyoming-bind".into()))?;
                cfg.wyoming_bind = parse_bind("--wyoming-bind", &v)?;
            }
            "--config" => {
                let v = it
                    .next()
                    .ok_or_else(|| ConfigError::MissingValue("--config".into()))?;
                cfg.config_file = Some(PathBuf::from(v));
            }
            // T03 accepts unknown flags nowhere: silent-ignore would violate
            // FR-EX-08 (no silent fallback) and mask typos.
            other => return Err(ConfigError::UnknownFlag(other.to_string())),
        }
    }

    // Layer 4: TOML file (if requested) — merged UNDER CLI/env choices.
    // We apply it here (last) but only for fields NOT already explicitly set
    // by CLI/env. To keep T03 minimal we detect "explicitly set" by
    // comparing against defaults; a richer three-way merge lands with T04.
    if let Some(path) = cfg.config_file.clone() {
        let overlay = load_toml_overlay(&path)?;
        if cfg.http_bind == Config::default().http_bind {
            if let Some(v) = overlay.http_bind {
                cfg.http_bind = v;
            }
        }
        if cfg.wyoming_bind == Config::default().wyoming_bind {
            if let Some(v) = overlay.wyoming_bind {
                cfg.wyoming_bind = v;
            }
        }
    }

    Ok(cfg)
}

fn parse_bind(flag: &str, value: &str) -> Result<SocketAddr, ConfigError> {
    value
        .parse::<SocketAddr>()
        .map_err(|e| ConfigError::InvalidValue {
            flag: flag.to_string(),
            value: value.to_string(),
            reason: e.to_string(),
        })
}

/// Fields the TOML overlay may contain.
#[derive(Default)]
struct TomlOverlay {
    http_bind: Option<SocketAddr>,
    wyoming_bind: Option<SocketAddr>,
}

/// Minimal hand-rolled TOML subset: `key = "value"` lines, `#` comments,
/// blank lines. Full TOML would drag in `toml`/`toml_edit`; T03 needs only
/// two string keys, and we can extend to serde_json/toml when T04 adds
/// registries. This subset keeps the surface small AND avoids any C
/// locale-sensitive parser reaching our path (NFR-RL-01 defence-in-depth).
fn load_toml_overlay(path: &PathBuf) -> Result<TomlOverlay, ConfigError> {
    let text = std::fs::read_to_string(path).map_err(|e| ConfigError::ConfigFileRead {
        path: path.clone(),
        error: e.to_string(),
    })?;
    let mut out = TomlOverlay::default();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| ConfigError::ConfigFileParse {
                path: path.clone(),
                error: format!("line {}: expected `key = value`", i + 1),
            })?;
        let key = key.trim();
        let value = value.trim().trim_matches('"');
        match key {
            "http_bind" => {
                out.http_bind = Some(parse_bind("http_bind", value).map_err(|e| {
                    ConfigError::ConfigFileParse {
                        path: path.clone(),
                        error: e.to_string(),
                    }
                })?);
            }
            "wyoming_bind" => {
                out.wyoming_bind = Some(parse_bind("wyoming_bind", value).map_err(|e| {
                    ConfigError::ConfigFileParse {
                        path: path.clone(),
                        error: e.to_string(),
                    }
                })?);
            }
            other => {
                // Unknown keys are an error, not silently ignored — same
                // reasoning as unknown CLI flags (FR-EX-08).
                return Err(ConfigError::ConfigFileParse {
                    path: path.clone(),
                    error: format!("line {}: unknown key {other:?}", i + 1),
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_env() {
        // Test-local env hygiene; parse_args reads these. `remove_var` is
        // `unsafe` under Rust edition 2024 because process env is a shared
        // resource — safe here because tests run single-threaded per
        // process crate default and no other thread reads these keys.
        // SAFETY: only this test module touches VOKRA_* env keys, and it
        // does so before spawning any observer thread.
        unsafe {
            std::env::remove_var("VOKRA_HTTP_BIND");
            std::env::remove_var("VOKRA_WYOMING_BIND");
            std::env::remove_var("VOKRA_CONFIG");
        }
    }

    #[test]
    fn startup_defaults_are_loopback() {
        clear_env();
        let cfg = parse_args(["vokra-server"]).unwrap();
        assert_eq!(cfg.http_bind.to_string(), "127.0.0.1:8080");
        assert_eq!(cfg.wyoming_bind.to_string(), "127.0.0.1:10300");
        assert!(cfg.config_file.is_none());
    }

    #[test]
    fn startup_help_flag_is_reported() {
        clear_env();
        let err = parse_args(["vokra-server", "--help"]).unwrap_err();
        assert!(matches!(err, ConfigError::HelpRequested));
    }

    #[test]
    fn startup_cli_overrides_defaults() {
        clear_env();
        let cfg = parse_args([
            "vokra-server",
            "--http-bind",
            "127.0.0.1:0",
            "--wyoming-bind",
            "127.0.0.1:0",
        ])
        .unwrap();
        assert_eq!(cfg.http_bind.port(), 0);
        assert_eq!(cfg.wyoming_bind.port(), 0);
    }

    #[test]
    fn startup_unknown_flag_is_error() {
        clear_env();
        let err = parse_args(["vokra-server", "--nope"]).unwrap_err();
        assert!(matches!(err, ConfigError::UnknownFlag(ref s) if s == "--nope"));
    }

    #[test]
    fn startup_bad_bind_is_error() {
        clear_env();
        let err = parse_args(["vokra-server", "--http-bind", "not-an-addr"]).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidValue { .. }));
    }
}
