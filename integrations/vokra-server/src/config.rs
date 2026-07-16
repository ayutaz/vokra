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

    // ---- Model registry paths (T04 / M4-19-T02). ----
    //
    // Every field is optional. When ALL of them are unset the server boots
    // health-only + Wyoming-discovery-only (the M2-09 default). When ANY is
    // set the server builds the full [`crate::service::InferenceService`]
    // registry; the required minimum is `whisper_base_gguf` AND
    // `piper_plus_gguf` (a half-wired config is a hard startup error, never a
    // silent partial boot — FR-EX-08). These paths map 1:1 onto
    // [`crate::service::ServiceConfig`].
    /// Whisper base GGUF. Required once any model path is configured.
    pub whisper_base_gguf: Option<PathBuf>,
    /// Whisper base tokenizer sidecar (base ships one per M0-06; large-v3
    /// embeds its vocab so it needs none).
    pub whisper_base_tokenizer: Option<PathBuf>,
    /// Whisper large-v3 GGUF (M2-06). Absent ⇒ `whisper-large-v3` requests
    /// are rejected, never silently routed to base.
    pub whisper_large_v3_gguf: Option<PathBuf>,
    /// Whisper large-v3 tokenizer sidecar (rarely needed — large-v3 embeds
    /// its vocab). Set without `whisper_large_v3_gguf` is a config error
    /// surfaced by [`crate::service::InferenceService::build`].
    pub whisper_large_v3_tokenizer: Option<PathBuf>,
    /// piper-plus voice GGUF. Required once any model path is configured.
    pub piper_plus_gguf: Option<PathBuf>,
    /// Kokoro-82M voice GGUF (M2-07). Advertised when present.
    pub kokoro_gguf: Option<PathBuf>,
    /// Voxtral (Mistral) GGUF (M3-10). Enables the `voxtral*` aliases.
    pub voxtral_gguf: Option<PathBuf>,
    /// Silero VAD v5 GGUF — optional Wyoming chunk-boundary helper.
    pub silero_vad_gguf: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            http_bind: "127.0.0.1:8080".parse().expect("valid default"),
            wyoming_bind: "127.0.0.1:10300".parse().expect("valid default"),
            config_file: None,
            whisper_base_gguf: None,
            whisper_base_tokenizer: None,
            whisper_large_v3_gguf: None,
            whisper_large_v3_tokenizer: None,
            piper_plus_gguf: None,
            kokoro_gguf: None,
            voxtral_gguf: None,
            silero_vad_gguf: None,
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

MODELS (all optional; unset ⇒ health-only + Wyoming discovery-only. When ANY
is set, --whisper-base AND --piper-plus are the required minimum — a partial
config is a hard startup error, never a silent partial boot):
    --whisper-base <PATH>            Whisper base GGUF.
    --whisper-base-tokenizer <PATH>  Whisper base tokenizer sidecar.
    --whisper-large-v3 <PATH>        Whisper large-v3 GGUF.
    --whisper-large-v3-tokenizer <PATH>
                                     Whisper large-v3 tokenizer sidecar.
    --piper-plus <PATH>              piper-plus voice GGUF.
    --kokoro <PATH>                  Kokoro-82M voice GGUF.
    --voxtral <PATH>                 Voxtral (Mistral) GGUF.
    --silero-vad <PATH>              Silero VAD v5 GGUF.

ENVIRONMENT:
    VOKRA_HTTP_BIND                  Same as --http-bind.
    VOKRA_WYOMING_BIND               Same as --wyoming-bind.
    VOKRA_CONFIG                     Same as --config.
    VOKRA_WHISPER_BASE               Same as --whisper-base.
    VOKRA_WHISPER_BASE_TOKENIZER     Same as --whisper-base-tokenizer.
    VOKRA_WHISPER_LARGE_V3           Same as --whisper-large-v3.
    VOKRA_WHISPER_LARGE_V3_TOKENIZER Same as --whisper-large-v3-tokenizer.
    VOKRA_PIPER_PLUS                 Same as --piper-plus.
    VOKRA_KOKORO                     Same as --kokoro.
    VOKRA_VOXTRAL                    Same as --voxtral.
    VOKRA_SILERO_VAD                 Same as --silero-vad.

Precedence (highest first): CLI flag > env var > TOML file > built-in default.
TOML keys mirror the flag names with underscores (e.g. `whisper_base`,
`piper_plus`, `silero_vad`).

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
    // Model paths from env (each overridden by the matching CLI flag below).
    if let Ok(v) = std::env::var("VOKRA_WHISPER_BASE") {
        cfg.whisper_base_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_BASE_TOKENIZER") {
        cfg.whisper_base_tokenizer = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_LARGE_V3") {
        cfg.whisper_large_v3_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_LARGE_V3_TOKENIZER") {
        cfg.whisper_large_v3_tokenizer = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_PIPER_PLUS") {
        cfg.piper_plus_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_KOKORO") {
        cfg.kokoro_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_VOXTRAL") {
        cfg.voxtral_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_SILERO_VAD") {
        cfg.silero_vad_gguf = Some(PathBuf::from(v));
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
            // Model registry paths (T04). Each takes a value; a missing value
            // is an error, never a silent skip.
            "--whisper-base" => {
                cfg.whisper_base_gguf =
                    Some(PathBuf::from(take_flag_value(&mut it, "--whisper-base")?));
            }
            "--whisper-base-tokenizer" => {
                cfg.whisper_base_tokenizer = Some(PathBuf::from(take_flag_value(
                    &mut it,
                    "--whisper-base-tokenizer",
                )?));
            }
            "--whisper-large-v3" => {
                cfg.whisper_large_v3_gguf = Some(PathBuf::from(take_flag_value(
                    &mut it,
                    "--whisper-large-v3",
                )?));
            }
            "--whisper-large-v3-tokenizer" => {
                cfg.whisper_large_v3_tokenizer = Some(PathBuf::from(take_flag_value(
                    &mut it,
                    "--whisper-large-v3-tokenizer",
                )?));
            }
            "--piper-plus" => {
                cfg.piper_plus_gguf =
                    Some(PathBuf::from(take_flag_value(&mut it, "--piper-plus")?));
            }
            "--kokoro" => {
                cfg.kokoro_gguf = Some(PathBuf::from(take_flag_value(&mut it, "--kokoro")?));
            }
            "--voxtral" => {
                cfg.voxtral_gguf = Some(PathBuf::from(take_flag_value(&mut it, "--voxtral")?));
            }
            "--silero-vad" => {
                cfg.silero_vad_gguf =
                    Some(PathBuf::from(take_flag_value(&mut it, "--silero-vad")?));
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
        // Model paths: TOML fills only the slots still unset by CLI/env, so
        // precedence stays CLI > env > TOML > default. `Option::or` keeps the
        // CLI/env value when present and otherwise adopts the TOML one.
        cfg.whisper_base_gguf = cfg.whisper_base_gguf.or(overlay.whisper_base_gguf);
        cfg.whisper_base_tokenizer = cfg
            .whisper_base_tokenizer
            .or(overlay.whisper_base_tokenizer);
        cfg.whisper_large_v3_gguf = cfg.whisper_large_v3_gguf.or(overlay.whisper_large_v3_gguf);
        cfg.whisper_large_v3_tokenizer = cfg
            .whisper_large_v3_tokenizer
            .or(overlay.whisper_large_v3_tokenizer);
        cfg.piper_plus_gguf = cfg.piper_plus_gguf.or(overlay.piper_plus_gguf);
        cfg.kokoro_gguf = cfg.kokoro_gguf.or(overlay.kokoro_gguf);
        cfg.voxtral_gguf = cfg.voxtral_gguf.or(overlay.voxtral_gguf);
        cfg.silero_vad_gguf = cfg.silero_vad_gguf.or(overlay.silero_vad_gguf);
    }

    Ok(cfg)
}

/// Pull the value token for a value-taking CLI flag, or return
/// [`ConfigError::MissingValue`] when the flag is the last argument. Keeps
/// the model-path flag arms in [`parse_args`] to a single line each.
fn take_flag_value<I: Iterator<Item = String>>(
    it: &mut I,
    flag: &str,
) -> Result<String, ConfigError> {
    it.next()
        .ok_or_else(|| ConfigError::MissingValue(flag.into()))
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
    whisper_base_gguf: Option<PathBuf>,
    whisper_base_tokenizer: Option<PathBuf>,
    whisper_large_v3_gguf: Option<PathBuf>,
    whisper_large_v3_tokenizer: Option<PathBuf>,
    piper_plus_gguf: Option<PathBuf>,
    kokoro_gguf: Option<PathBuf>,
    voxtral_gguf: Option<PathBuf>,
    silero_vad_gguf: Option<PathBuf>,
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
            // Model registry paths (T04). Values are paths, not parsed further
            // here; `InferenceService::build` is the one that opens them and
            // fails hard on a missing / broken file (FR-EX-08).
            "whisper_base" => out.whisper_base_gguf = Some(PathBuf::from(value)),
            "whisper_base_tokenizer" => out.whisper_base_tokenizer = Some(PathBuf::from(value)),
            "whisper_large_v3" => out.whisper_large_v3_gguf = Some(PathBuf::from(value)),
            "whisper_large_v3_tokenizer" => {
                out.whisper_large_v3_tokenizer = Some(PathBuf::from(value))
            }
            "piper_plus" => out.piper_plus_gguf = Some(PathBuf::from(value)),
            "kokoro" => out.kokoro_gguf = Some(PathBuf::from(value)),
            "voxtral" => out.voxtral_gguf = Some(PathBuf::from(value)),
            "silero_vad" => out.silero_vad_gguf = Some(PathBuf::from(value)),
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
        // does so before spawning any observer thread. Clearing the model
        // keys too keeps a developer's ambient VOKRA_WHISPER_BASE (etc.) out
        // of the CLI/TOML parse tests below.
        unsafe {
            std::env::remove_var("VOKRA_HTTP_BIND");
            std::env::remove_var("VOKRA_WYOMING_BIND");
            std::env::remove_var("VOKRA_CONFIG");
            std::env::remove_var("VOKRA_WHISPER_BASE");
            std::env::remove_var("VOKRA_WHISPER_BASE_TOKENIZER");
            std::env::remove_var("VOKRA_WHISPER_LARGE_V3");
            std::env::remove_var("VOKRA_WHISPER_LARGE_V3_TOKENIZER");
            std::env::remove_var("VOKRA_PIPER_PLUS");
            std::env::remove_var("VOKRA_KOKORO");
            std::env::remove_var("VOKRA_VOXTRAL");
            std::env::remove_var("VOKRA_SILERO_VAD");
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

    #[test]
    fn startup_no_model_flags_leaves_paths_unset() {
        clear_env();
        let cfg = parse_args(["vokra-server"]).unwrap();
        assert!(cfg.whisper_base_gguf.is_none());
        assert!(cfg.piper_plus_gguf.is_none());
        assert!(cfg.kokoro_gguf.is_none());
        assert!(cfg.voxtral_gguf.is_none());
        assert!(cfg.silero_vad_gguf.is_none());
    }

    #[test]
    fn startup_cli_sets_model_paths() {
        use std::path::Path;
        clear_env();
        let cfg = parse_args([
            "vokra-server",
            "--whisper-base",
            "/m/whisper-base.gguf",
            "--whisper-base-tokenizer",
            "/m/whisper-base.tok",
            "--whisper-large-v3",
            "/m/large.gguf",
            "--piper-plus",
            "/m/piper.gguf",
            "--kokoro",
            "/m/kokoro.gguf",
            "--voxtral",
            "/m/voxtral.gguf",
            "--silero-vad",
            "/m/vad.gguf",
        ])
        .unwrap();
        assert_eq!(
            cfg.whisper_base_gguf.as_deref(),
            Some(Path::new("/m/whisper-base.gguf"))
        );
        assert_eq!(
            cfg.whisper_base_tokenizer.as_deref(),
            Some(Path::new("/m/whisper-base.tok"))
        );
        assert_eq!(
            cfg.whisper_large_v3_gguf.as_deref(),
            Some(Path::new("/m/large.gguf"))
        );
        assert_eq!(
            cfg.piper_plus_gguf.as_deref(),
            Some(Path::new("/m/piper.gguf"))
        );
        assert_eq!(
            cfg.kokoro_gguf.as_deref(),
            Some(Path::new("/m/kokoro.gguf"))
        );
        assert_eq!(
            cfg.voxtral_gguf.as_deref(),
            Some(Path::new("/m/voxtral.gguf"))
        );
        assert_eq!(
            cfg.silero_vad_gguf.as_deref(),
            Some(Path::new("/m/vad.gguf"))
        );
    }

    #[test]
    fn startup_model_flag_missing_value_is_error() {
        clear_env();
        // `--piper-plus` as the last token has no value → hard error, not a
        // silent skip (FR-EX-08).
        let err = parse_args(["vokra-server", "--piper-plus"]).unwrap_err();
        assert!(matches!(err, ConfigError::MissingValue(ref s) if s == "--piper-plus"));
    }

    /// Writes a throwaway TOML file to a unique temp path, runs `f` with its
    /// path, and removes it afterwards. Hand-rolled to avoid a `tempfile`
    /// dependency in the excluded-workspace lockfile (NFR-DS-02 spirit).
    fn with_temp_toml(name: &str, body: &str, f: impl FnOnce(&std::path::Path)) {
        let path = std::env::temp_dir().join(format!(
            "vokra-server-config-test-{}-{name}.toml",
            std::process::id()
        ));
        std::fs::write(&path, body).expect("write temp toml");
        f(&path);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn startup_toml_sets_model_paths() {
        use std::path::Path;
        clear_env();
        with_temp_toml(
            "models",
            "whisper_base = \"/t/base.gguf\"\npiper_plus = \"/t/piper.gguf\"\nkokoro = \"/t/kokoro.gguf\"\n",
            |p| {
                let cfg = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap();
                assert_eq!(
                    cfg.whisper_base_gguf.as_deref(),
                    Some(Path::new("/t/base.gguf"))
                );
                assert_eq!(
                    cfg.piper_plus_gguf.as_deref(),
                    Some(Path::new("/t/piper.gguf"))
                );
                assert_eq!(
                    cfg.kokoro_gguf.as_deref(),
                    Some(Path::new("/t/kokoro.gguf"))
                );
                assert!(cfg.voxtral_gguf.is_none());
            },
        );
    }

    #[test]
    fn startup_cli_model_path_overrides_toml() {
        use std::path::Path;
        clear_env();
        with_temp_toml("override", "whisper_base = \"/toml/base.gguf\"\n", |p| {
            let cfg = parse_args([
                "vokra-server",
                "--config",
                p.to_str().unwrap(),
                "--whisper-base",
                "/cli/base.gguf",
            ])
            .unwrap();
            // CLI must win over TOML (precedence CLI > env > TOML > default).
            assert_eq!(
                cfg.whisper_base_gguf.as_deref(),
                Some(Path::new("/cli/base.gguf"))
            );
        });
    }

    #[test]
    fn startup_toml_unknown_model_key_is_error() {
        clear_env();
        with_temp_toml("badkey", "whisper_xl = \"/x.gguf\"\n", |p| {
            let err = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap_err();
            assert!(matches!(err, ConfigError::ConfigFileParse { .. }));
        });
    }
}
