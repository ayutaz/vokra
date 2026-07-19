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

use vokra_core::BackendKind;

use crate::service::{BackendOverrides, ModelSlot};

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
    /// Whisper small GGUF (cc-39, 2026-07-19 M4-residual audit). Absent ⇒
    /// `whisper-small` requests are rejected, never silently routed to
    /// another size.
    pub whisper_small_gguf: Option<PathBuf>,
    /// Whisper small tokenizer sidecar. Rarely needed — every converter-B
    /// GGUF embeds `vokra.tokenizer.model` (verified on the real
    /// `whisper-small.gguf`, 2026-07-19); kept for the converter-A path,
    /// exactly like the large-v3 sidecar above.
    pub whisper_small_tokenizer: Option<PathBuf>,
    /// Whisper medium GGUF (cc-39). Same no-substitution rule as small.
    pub whisper_medium_gguf: Option<PathBuf>,
    /// Whisper medium tokenizer sidecar (see [`Self::whisper_small_tokenizer`]).
    pub whisper_medium_tokenizer: Option<PathBuf>,
    /// Whisper large-v3-turbo GGUF (cc-39). Same no-substitution rule.
    pub whisper_turbo_gguf: Option<PathBuf>,
    /// Whisper turbo tokenizer sidecar (see [`Self::whisper_small_tokenizer`]).
    pub whisper_turbo_tokenizer: Option<PathBuf>,
    /// piper-plus voice GGUF. Required once any model path is configured.
    pub piper_plus_gguf: Option<PathBuf>,
    /// Kokoro-82M voice GGUF (M2-07). Advertised when present.
    pub kokoro_gguf: Option<PathBuf>,
    /// Voxtral (Mistral) GGUF (M3-10). Enables the `voxtral*` aliases.
    pub voxtral_gguf: Option<PathBuf>,
    /// Silero VAD v5 GGUF — optional Wyoming chunk-boundary helper.
    pub silero_vad_gguf: Option<PathBuf>,

    /// Enable the real 8-language piper-plus G2P text front-end
    /// (campaign-2 P1 fix). `None` = not configured = **off** (the
    /// startup default): the TTS surfaces accept only raw phoneme ids /
    /// `[[symbol]]` literals through the `PassthroughPhonemizer`, and
    /// plain text stays an explicit error (FR-EX-08) exactly as before.
    ///
    /// `Some(true)` makes the startup path build a
    /// `vokra_piper_g2p::PiperPlusG2p` from the loaded piper voice (the
    /// GGUF's own `vokra.piper.phoneme_symbols` / `language_codes`; the
    /// JA dictionary is compile-time bundled — no data dir needed) and
    /// inject it via `InferenceService::with_phonemizer`, so plain-text
    /// TTS works on `/api/tts` and Wyoming `synthesize`.
    ///
    /// NOTE the deliberate behaviour change when enabled: request text
    /// is then interpreted as natural language by the real G2P, so
    /// raw-phoneme-id payloads (`"1 30 2"`) are no longer parsed as ids.
    /// Deployments that feed raw ids must keep this off — which is why
    /// this is opt-in and never auto-enabled.
    ///
    /// Kept as `Option<bool>` (not `bool`) so the CLI > env > TOML
    /// precedence merge can distinguish "explicitly set to false" from
    /// "unset" — same pattern as the model paths above.
    pub piper_g2p: Option<bool>,

    /// Multi-session concurrency cap (`--max-concurrent-sessions`,
    /// `VOKRA_MAX_CONCURRENT_SESSIONS`, TOML `max_concurrent_sessions`).
    ///
    /// The value feeds BOTH the session registry and the scheduler permit
    /// pool — `Scheduler::new` requires `n_stream == max_concurrent_sessions`,
    /// so a single knob drives both (splitting them would let the registry
    /// admit a session the scheduler can never run).
    ///
    /// Unset = [`DEFAULT_MAX_CONCURRENT_SESSIONS`] (4). `0` is rejected as
    /// an invalid value rather than silently meaning "unlimited" — an
    /// accidental `=0` must not disable the DoS ceiling (FR-EX-08).
    ///
    /// [`DEFAULT_MAX_CONCURRENT_SESSIONS`]: crate::server::DEFAULT_MAX_CONCURRENT_SESSIONS
    pub max_concurrent_sessions: Option<usize>,

    /// Default backend every pre-warmed engine runs on (`--backend`,
    /// `VOKRA_BACKEND`, TOML `backend`). Unset = [`BackendKind::Cpu`].
    ///
    /// cc-30 (2026-07-19 M4-residual audit): before this, the server had no
    /// backend knob at all — [`crate::service::ServiceConfig::minimum`]
    /// hard-coded `BackendKind::Cpu`, so a GPU-capable build could not be
    /// selected at runtime.
    ///
    /// **A backend that is not compiled into this binary is a hard startup
    /// error**, never a silent CPU fall back (FR-EX-08). See
    /// [`crate::service::ensure_backend_available`] and the Cargo-feature
    /// passthrough note in [`HELP_TEXT`].
    pub backend: Option<BackendKind>,

    /// Per-model backend overrides (`--model-backend <SLOT>=<BACKEND>`,
    /// repeatable; `VOKRA_MODEL_BACKENDS`; TOML `model_backends`).
    ///
    /// cc-30: this is the "per-model backend override is a T03 follow-up"
    /// note that sat on `ServiceConfig::backend` since M2-09-T04. A slot
    /// without an override runs on [`Self::backend`].
    pub model_backends: BackendOverrides,
}

impl Config {
    /// Whether the real piper-plus G2P front-end was requested
    /// ([`Config::piper_g2p`]; unset = off).
    pub fn piper_g2p_enabled(&self) -> bool {
        self.piper_g2p.unwrap_or(false)
    }

    /// The effective multi-session cap: the configured value, or the
    /// built-in default when unset.
    pub fn max_concurrent_sessions_or_default(&self) -> usize {
        self.max_concurrent_sessions
            .unwrap_or(crate::server::DEFAULT_MAX_CONCURRENT_SESSIONS)
    }

    /// The effective default backend (cc-30): the configured value, or
    /// [`BackendKind::Cpu`] when unset. Per-slot overrides in
    /// [`Self::model_backends`] take precedence over this.
    pub fn backend_or_default(&self) -> BackendKind {
        self.backend.unwrap_or(BackendKind::Cpu)
    }
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
            whisper_small_gguf: None,
            whisper_small_tokenizer: None,
            whisper_medium_gguf: None,
            whisper_medium_tokenizer: None,
            whisper_turbo_gguf: None,
            whisper_turbo_tokenizer: None,
            piper_plus_gguf: None,
            kokoro_gguf: None,
            voxtral_gguf: None,
            silero_vad_gguf: None,
            piper_g2p: None,
            max_concurrent_sessions: None,
            backend: None,
            model_backends: BackendOverrides::default(),
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
    --max-concurrent-sessions <N>
                             Multi-session cap shared by the session
                             registry and the scheduler permit pool.
                             Default: 4. Must be >= 1 (0 is rejected —
                             it would remove the concurrency ceiling).

MODELS (all optional; unset ⇒ health-only + Wyoming discovery-only. When ANY
is set, --whisper-base AND --piper-plus are the required minimum — a partial
config is a hard startup error, never a silent partial boot):
    --whisper-base <PATH>            Whisper base GGUF.
    --whisper-base-tokenizer <PATH>  Whisper base tokenizer sidecar.
    --whisper-small <PATH>           Whisper small GGUF.
    --whisper-small-tokenizer <PATH> Whisper small tokenizer sidecar.
    --whisper-medium <PATH>          Whisper medium GGUF.
    --whisper-medium-tokenizer <PATH>
                                     Whisper medium tokenizer sidecar.
    --whisper-turbo <PATH>           Whisper large-v3-turbo GGUF.
    --whisper-turbo-tokenizer <PATH> Whisper turbo tokenizer sidecar.
    --whisper-large-v3 <PATH>        Whisper large-v3 GGUF.
    --whisper-large-v3-tokenizer <PATH>
                                     Whisper large-v3 tokenizer sidecar.
    --piper-plus <PATH>              piper-plus voice GGUF.
    --kokoro <PATH>                  Kokoro-82M voice GGUF.
    --voxtral <PATH>                 Voxtral (Mistral) GGUF.
    --silero-vad <PATH>              Silero VAD v5 GGUF.
    --piper-g2p                      Enable the real 8-language piper-plus G2P
                                     text front-end for TTS (requires
                                     --piper-plus; dictionaries are built into
                                     the binary — no data dir). Default: OFF —
                                     TTS text is then raw phoneme ids /
                                     [[symbol]] literals only, and plain text
                                     is an explicit error. NOTE: with G2P on,
                                     request text is natural language; raw
                                     phoneme-id payloads are no longer parsed.

BACKEND (cc-30):
    --backend <NAME>                 Default backend for every pre-warmed
                                     engine: cpu | metal | cuda | vulkan.
                                     Default: cpu.
    --model-backend <SLOT>=<NAME>    Per-model override, repeatable. SLOT is
                                     one of: whisper-base, whisper-small,
                                     whisper-medium, whisper-turbo,
                                     whisper-large-v3, piper-plus, kokoro,
                                     voxtral, silero-vad. Repeating the same
                                     SLOT is an error (ambiguous config).

    A backend is only SELECTABLE if it was COMPILED INTO this binary. The
    server's GPU backends are Cargo features that forward to vokra-models:

        cargo build --release --features metal    # macOS / iOS
        cargo build --release --features cuda     # Windows / Linux + NVIDIA
        cargo build --release --features vulkan   # Linux / Android / Windows

    Requesting a backend that was not compiled in (or whose device cannot be
    opened) is a HARD STARTUP ERROR, never a silent CPU fall back (FR-EX-08).
    The default build is CPU-only, so `--backend metal` on it fails fast with
    the rebuild instruction rather than serving CPU results under a GPU label.

ENVIRONMENT:
    VOKRA_HTTP_BIND                  Same as --http-bind.
    VOKRA_WYOMING_BIND               Same as --wyoming-bind.
    VOKRA_CONFIG                     Same as --config.
    VOKRA_WHISPER_BASE               Same as --whisper-base.
    VOKRA_WHISPER_BASE_TOKENIZER     Same as --whisper-base-tokenizer.
    VOKRA_WHISPER_SMALL              Same as --whisper-small.
    VOKRA_WHISPER_SMALL_TOKENIZER    Same as --whisper-small-tokenizer.
    VOKRA_WHISPER_MEDIUM             Same as --whisper-medium.
    VOKRA_WHISPER_MEDIUM_TOKENIZER   Same as --whisper-medium-tokenizer.
    VOKRA_WHISPER_TURBO              Same as --whisper-turbo.
    VOKRA_WHISPER_TURBO_TOKENIZER    Same as --whisper-turbo-tokenizer.
    VOKRA_WHISPER_LARGE_V3           Same as --whisper-large-v3.
    VOKRA_WHISPER_LARGE_V3_TOKENIZER Same as --whisper-large-v3-tokenizer.
    VOKRA_PIPER_PLUS                 Same as --piper-plus.
    VOKRA_KOKORO                     Same as --kokoro.
    VOKRA_VOXTRAL                    Same as --voxtral.
    VOKRA_SILERO_VAD                 Same as --silero-vad.
    VOKRA_PIPER_G2P                  Same as --piper-g2p (1/true/0/false).
    VOKRA_MAX_CONCURRENT_SESSIONS    Same as --max-concurrent-sessions.
    VOKRA_BACKEND                    Same as --backend.
    VOKRA_MODEL_BACKENDS             Comma-separated --model-backend list,
                                     e.g. `whisper-base=metal,kokoro=cpu`.

Precedence (highest first): CLI flag > env var > TOML file > built-in default.
TOML keys mirror the flag names with underscores (e.g. `whisper_base`,
`piper_plus`, `silero_vad`, `piper_g2p = true`, `backend = \"metal\"`,
`model_backends = \"whisper-base=metal,kokoro=cpu\"`).

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
    if let Ok(v) = std::env::var("VOKRA_WHISPER_SMALL") {
        cfg.whisper_small_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_SMALL_TOKENIZER") {
        cfg.whisper_small_tokenizer = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_MEDIUM") {
        cfg.whisper_medium_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_MEDIUM_TOKENIZER") {
        cfg.whisper_medium_tokenizer = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_TURBO") {
        cfg.whisper_turbo_gguf = Some(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("VOKRA_WHISPER_TURBO_TOKENIZER") {
        cfg.whisper_turbo_tokenizer = Some(PathBuf::from(v));
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
    if let Ok(v) = std::env::var("VOKRA_MAX_CONCURRENT_SESSIONS") {
        cfg.max_concurrent_sessions = Some(parse_session_cap("VOKRA_MAX_CONCURRENT_SESSIONS", &v)?);
    }
    if let Ok(v) = std::env::var("VOKRA_PIPER_G2P") {
        cfg.piper_g2p = Some(parse_bool("VOKRA_PIPER_G2P", &v)?);
    }
    if let Ok(v) = std::env::var("VOKRA_BACKEND") {
        cfg.backend = Some(parse_backend_name("VOKRA_BACKEND", &v)?);
    }
    // Per-slot overrides are accumulated per LAYER, then merged slot-by-slot
    // (CLI > env > TOML) at the end — the same precedence the model paths
    // get. Merging eagerly here would make a CLI `--model-backend x=cpu`
    // collide with an env `x=metal` and be reported as a duplicate, when the
    // operator's intent is plainly "CLI wins".
    let mut env_backends = BackendOverrides::default();
    if let Ok(v) = std::env::var("VOKRA_MODEL_BACKENDS") {
        apply_model_backend_list("VOKRA_MODEL_BACKENDS", &v, &mut env_backends)?;
    }
    let mut cli_backends = BackendOverrides::default();
    let mut toml_backends = BackendOverrides::default();

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
            // cc-39: the three M4-14 Whisper sizes. Exact mirror of the
            // large-v3 pattern above — including the (rarely needed)
            // tokenizer sidecar, since a converter-A GGUF without an
            // embedded `vokra.tokenizer.model` still needs one.
            "--whisper-small" => {
                cfg.whisper_small_gguf =
                    Some(PathBuf::from(take_flag_value(&mut it, "--whisper-small")?));
            }
            "--whisper-small-tokenizer" => {
                cfg.whisper_small_tokenizer = Some(PathBuf::from(take_flag_value(
                    &mut it,
                    "--whisper-small-tokenizer",
                )?));
            }
            "--whisper-medium" => {
                cfg.whisper_medium_gguf =
                    Some(PathBuf::from(take_flag_value(&mut it, "--whisper-medium")?));
            }
            "--whisper-medium-tokenizer" => {
                cfg.whisper_medium_tokenizer = Some(PathBuf::from(take_flag_value(
                    &mut it,
                    "--whisper-medium-tokenizer",
                )?));
            }
            "--whisper-turbo" => {
                cfg.whisper_turbo_gguf =
                    Some(PathBuf::from(take_flag_value(&mut it, "--whisper-turbo")?));
            }
            "--whisper-turbo-tokenizer" => {
                cfg.whisper_turbo_tokenizer = Some(PathBuf::from(take_flag_value(
                    &mut it,
                    "--whisper-turbo-tokenizer",
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
            // Boolean switch (takes no value): enable the real 8-language
            // piper-plus G2P text front-end. There is deliberately no
            // `--no-piper-g2p` — off is the default, and an env/TOML `true`
            // can only be overridden by removing it (keeping the CLI surface
            // one honest switch).
            "--piper-g2p" => {
                cfg.piper_g2p = Some(true);
            }
            "--max-concurrent-sessions" => {
                let v = take_flag_value(&mut it, "--max-concurrent-sessions")?;
                cfg.max_concurrent_sessions =
                    Some(parse_session_cap("--max-concurrent-sessions", &v)?);
            }
            // cc-30: default backend + repeatable per-model override.
            "--backend" => {
                let v = take_flag_value(&mut it, "--backend")?;
                cfg.backend = Some(parse_backend_name("--backend", &v)?);
            }
            "--model-backend" => {
                let v = take_flag_value(&mut it, "--model-backend")?;
                // One `SLOT=BACKEND` pair per occurrence (the comma-list form
                // is the env/TOML spelling); repeating a SLOT on the CLI is a
                // duplicate error, not last-wins (FR-EX-08 — an ambiguous
                // config must not silently pick one).
                apply_model_backend_pair("--model-backend", &v, &mut cli_backends)?;
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
        cfg.whisper_small_gguf = cfg.whisper_small_gguf.or(overlay.whisper_small_gguf);
        cfg.whisper_small_tokenizer = cfg
            .whisper_small_tokenizer
            .or(overlay.whisper_small_tokenizer);
        cfg.whisper_medium_gguf = cfg.whisper_medium_gguf.or(overlay.whisper_medium_gguf);
        cfg.whisper_medium_tokenizer = cfg
            .whisper_medium_tokenizer
            .or(overlay.whisper_medium_tokenizer);
        cfg.whisper_turbo_gguf = cfg.whisper_turbo_gguf.or(overlay.whisper_turbo_gguf);
        cfg.whisper_turbo_tokenizer = cfg
            .whisper_turbo_tokenizer
            .or(overlay.whisper_turbo_tokenizer);
        cfg.piper_plus_gguf = cfg.piper_plus_gguf.or(overlay.piper_plus_gguf);
        cfg.kokoro_gguf = cfg.kokoro_gguf.or(overlay.kokoro_gguf);
        cfg.voxtral_gguf = cfg.voxtral_gguf.or(overlay.voxtral_gguf);
        cfg.silero_vad_gguf = cfg.silero_vad_gguf.or(overlay.silero_vad_gguf);
        // `Option<bool>` keeps the same CLI > env > TOML precedence as the
        // paths: an explicit env `VOKRA_PIPER_G2P=0` beats a TOML `true`.
        cfg.piper_g2p = cfg.piper_g2p.or(overlay.piper_g2p);
        cfg.max_concurrent_sessions = cfg
            .max_concurrent_sessions
            .or(overlay.max_concurrent_sessions);
        cfg.backend = cfg.backend.or(overlay.backend);
        toml_backends = overlay.model_backends;
    }

    // Per-slot backend overrides: merge the three layers slot-by-slot so
    // precedence matches every other knob (CLI > env > TOML). Duplicates
    // WITHIN a layer were already rejected at parse time.
    cfg.model_backends = cli_backends.or(&env_backends).or(&toml_backends);

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

/// Parse a boolean env/TOML value. Accepts `1`/`true` and `0`/`false`
/// (ASCII case-insensitive). Anything else is a hard
/// [`ConfigError::InvalidValue`] — a typo like `VOKRA_PIPER_G2P=yes`
/// must never be silently read as "off" (FR-EX-08).
fn parse_bool(flag: &str, value: &str) -> Result<bool, ConfigError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => Err(ConfigError::InvalidValue {
            flag: flag.to_string(),
            value: value.to_string(),
            reason: "expected one of: 1, true, 0, false".to_string(),
        }),
    }
}

/// Parse a backend name (cc-30). Mirrors `vokra_cli::bench::parse_backend`
/// verbatim so the CLI and the server accept exactly the same vocabulary.
///
/// Every [`BackendKind`] variant parses here regardless of which Cargo
/// features this binary was built with — *selectability* is a separate,
/// later check ([`crate::service::ensure_backend_available`]) so the error
/// message can distinguish "no such backend" (typo) from "that backend was
/// not compiled into this build" (rebuild instruction). Collapsing the two
/// into one error would tell an operator to fix the wrong thing.
fn parse_backend_name(flag: &str, value: &str) -> Result<BackendKind, ConfigError> {
    match value.trim() {
        "cpu" => Ok(BackendKind::Cpu),
        "metal" => Ok(BackendKind::Metal),
        "cuda" => Ok(BackendKind::Cuda),
        "vulkan" => Ok(BackendKind::Vulkan),
        other => Err(ConfigError::InvalidValue {
            flag: flag.to_string(),
            value: other.to_string(),
            reason: "expected one of: cpu, metal, cuda, vulkan".to_string(),
        }),
    }
}

/// Apply one `SLOT=BACKEND` pair to `out`, rejecting an unknown slot, an
/// unknown backend, a malformed pair, and a repeated slot (FR-EX-08 — an
/// ambiguous override must not silently resolve to one of the two values).
fn apply_model_backend_pair(
    flag: &str,
    pair: &str,
    out: &mut BackendOverrides,
) -> Result<(), ConfigError> {
    let (slot_raw, backend_raw) =
        pair.split_once('=')
            .ok_or_else(|| ConfigError::InvalidValue {
                flag: flag.to_string(),
                value: pair.to_string(),
                reason: format!(
                    "expected `SLOT=BACKEND` (slots: {})",
                    ModelSlot::ALL_NAMES.join(", ")
                ),
            })?;
    let slot = ModelSlot::parse(slot_raw.trim()).ok_or_else(|| ConfigError::InvalidValue {
        flag: flag.to_string(),
        value: slot_raw.trim().to_string(),
        reason: format!(
            "unknown model slot (expected one of: {})",
            ModelSlot::ALL_NAMES.join(", ")
        ),
    })?;
    let backend = parse_backend_name(flag, backend_raw)?;
    if out.get(slot).is_some() {
        return Err(ConfigError::InvalidValue {
            flag: flag.to_string(),
            value: pair.to_string(),
            reason: format!(
                "model slot `{}` already has a backend override in this layer; \
                 specify it once (refusing to silently pick one of two values)",
                slot.as_str()
            ),
        });
    }
    out.set(slot, backend);
    Ok(())
}

/// Apply a comma-separated `SLOT=BACKEND` list (the env / TOML spelling of
/// the repeatable `--model-backend` flag). Empty entries are skipped so a
/// trailing comma is tolerated; everything else routes through
/// [`apply_model_backend_pair`].
fn apply_model_backend_list(
    flag: &str,
    list: &str,
    out: &mut BackendOverrides,
) -> Result<(), ConfigError> {
    for entry in list.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        apply_model_backend_pair(flag, entry, out)?;
    }
    Ok(())
}

/// Parse the multi-session cap. Must be a positive integer: `0` would
/// disable the DoS ceiling that the registry and scheduler share, so it is
/// an explicit error rather than a silent "unlimited" (FR-EX-08).
fn parse_session_cap(flag: &str, value: &str) -> Result<usize, ConfigError> {
    let n = value
        .trim()
        .parse::<usize>()
        .map_err(|e| ConfigError::InvalidValue {
            flag: flag.to_string(),
            value: value.to_string(),
            reason: e.to_string(),
        })?;
    if n == 0 {
        return Err(ConfigError::InvalidValue {
            flag: flag.to_string(),
            value: value.to_string(),
            reason: "must be >= 1 (0 would remove the concurrency ceiling)".to_string(),
        });
    }
    Ok(n)
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
    whisper_small_gguf: Option<PathBuf>,
    whisper_small_tokenizer: Option<PathBuf>,
    whisper_medium_gguf: Option<PathBuf>,
    whisper_medium_tokenizer: Option<PathBuf>,
    whisper_turbo_gguf: Option<PathBuf>,
    whisper_turbo_tokenizer: Option<PathBuf>,
    piper_plus_gguf: Option<PathBuf>,
    kokoro_gguf: Option<PathBuf>,
    voxtral_gguf: Option<PathBuf>,
    silero_vad_gguf: Option<PathBuf>,
    piper_g2p: Option<bool>,
    max_concurrent_sessions: Option<usize>,
    backend: Option<BackendKind>,
    model_backends: BackendOverrides,
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
            "whisper_small" => out.whisper_small_gguf = Some(PathBuf::from(value)),
            "whisper_small_tokenizer" => out.whisper_small_tokenizer = Some(PathBuf::from(value)),
            "whisper_medium" => out.whisper_medium_gguf = Some(PathBuf::from(value)),
            "whisper_medium_tokenizer" => out.whisper_medium_tokenizer = Some(PathBuf::from(value)),
            "whisper_turbo" => out.whisper_turbo_gguf = Some(PathBuf::from(value)),
            "whisper_turbo_tokenizer" => out.whisper_turbo_tokenizer = Some(PathBuf::from(value)),
            "piper_plus" => out.piper_plus_gguf = Some(PathBuf::from(value)),
            "kokoro" => out.kokoro_gguf = Some(PathBuf::from(value)),
            "voxtral" => out.voxtral_gguf = Some(PathBuf::from(value)),
            "silero_vad" => out.silero_vad_gguf = Some(PathBuf::from(value)),
            "piper_g2p" => {
                out.piper_g2p = Some(parse_bool("piper_g2p", value).map_err(|e| {
                    ConfigError::ConfigFileParse {
                        path: path.clone(),
                        error: e.to_string(),
                    }
                })?);
            }
            "max_concurrent_sessions" => {
                out.max_concurrent_sessions = Some(
                    parse_session_cap("max_concurrent_sessions", value).map_err(|e| {
                        ConfigError::ConfigFileParse {
                            path: path.clone(),
                            error: e.to_string(),
                        }
                    })?,
                );
            }
            "backend" => {
                out.backend = Some(parse_backend_name("backend", value).map_err(|e| {
                    ConfigError::ConfigFileParse {
                        path: path.clone(),
                        error: e.to_string(),
                    }
                })?);
            }
            "model_backends" => {
                apply_model_backend_list("model_backends", value, &mut out.model_backends)
                    .map_err(|e| ConfigError::ConfigFileParse {
                        path: path.clone(),
                        error: e.to_string(),
                    })?;
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
            std::env::remove_var("VOKRA_PIPER_G2P");
            // cc-39 sizes + cc-30 backend knobs: same hygiene reason as above.
            std::env::remove_var("VOKRA_WHISPER_SMALL");
            std::env::remove_var("VOKRA_WHISPER_SMALL_TOKENIZER");
            std::env::remove_var("VOKRA_WHISPER_MEDIUM");
            std::env::remove_var("VOKRA_WHISPER_MEDIUM_TOKENIZER");
            std::env::remove_var("VOKRA_WHISPER_TURBO");
            std::env::remove_var("VOKRA_WHISPER_TURBO_TOKENIZER");
            std::env::remove_var("VOKRA_BACKEND");
            std::env::remove_var("VOKRA_MODEL_BACKENDS");
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

    // ---- --piper-g2p (campaign-2 P1: real G2P opt-in) ----

    /// Unset keeps the built-in default; the accessor is what the startup
    /// path reads, so it is pinned alongside the raw field (P2 cc-29).
    #[test]
    fn startup_max_concurrent_sessions_defaults_to_the_builtin() {
        clear_env();
        let cfg = parse_args(["vokra-server"]).unwrap();
        assert_eq!(cfg.max_concurrent_sessions, None, "unset must stay None");
        assert_eq!(
            cfg.max_concurrent_sessions_or_default(),
            crate::server::DEFAULT_MAX_CONCURRENT_SESSIONS,
        );
    }

    #[test]
    fn startup_max_concurrent_sessions_cli_flag_overrides() {
        clear_env();
        let cfg = parse_args(["vokra-server", "--max-concurrent-sessions", "16"]).unwrap();
        assert_eq!(cfg.max_concurrent_sessions, Some(16));
        assert_eq!(cfg.max_concurrent_sessions_or_default(), 16);
    }

    /// `0` must be a hard error, not a silent "unlimited": the value caps
    /// BOTH the session registry and the scheduler permit pool, so zeroing
    /// it would remove the DoS ceiling (FR-EX-08). Same rationale as the
    /// bool-typo rejection above; the env layer routes through the same
    /// helper (not set in-process — see the note on the bool test).
    #[test]
    fn startup_max_concurrent_sessions_rejects_zero_and_garbage() {
        assert_eq!(
            parse_session_cap("--max-concurrent-sessions", "1").unwrap(),
            1
        );
        assert_eq!(
            parse_session_cap("VOKRA_MAX_CONCURRENT_SESSIONS", " 32 ").unwrap(),
            32,
            "surrounding whitespace is trimmed"
        );
        for raw in ["0", "-1", "", "many", "4.5"] {
            let err = parse_session_cap("--max-concurrent-sessions", raw)
                .expect_err("value {raw:?} must be rejected");
            assert!(
                matches!(err, ConfigError::InvalidValue { .. }),
                "value {raw:?} must be InvalidValue, got {err:?}"
            );
        }
    }

    #[test]
    fn startup_max_concurrent_sessions_cli_beats_toml() {
        clear_env();
        with_temp_toml(
            "max-sessions",
            "whisper_base = \"/t/base.gguf\"\nmax_concurrent_sessions = 9\n",
            |path| {
                let from_toml =
                    parse_args(["vokra-server", "--config", path.to_str().unwrap()]).unwrap();
                assert_eq!(from_toml.max_concurrent_sessions, Some(9), "TOML applies");

                let cli_wins = parse_args([
                    "vokra-server",
                    "--config",
                    path.to_str().unwrap(),
                    "--max-concurrent-sessions",
                    "2",
                ])
                .unwrap();
                assert_eq!(cli_wins.max_concurrent_sessions, Some(2), "CLI > TOML");
            },
        );
    }

    #[test]
    fn startup_piper_g2p_defaults_off() {
        clear_env();
        let cfg = parse_args(["vokra-server"]).unwrap();
        assert_eq!(cfg.piper_g2p, None, "unset must stay None");
        assert!(!cfg.piper_g2p_enabled(), "unset must mean OFF");
    }

    #[test]
    fn startup_piper_g2p_cli_switch_enables() {
        clear_env();
        let cfg = parse_args(["vokra-server", "--piper-g2p"]).unwrap();
        assert_eq!(cfg.piper_g2p, Some(true));
        assert!(cfg.piper_g2p_enabled());
    }

    /// The env layer (`VOKRA_PIPER_G2P`) routes through `parse_bool` — this
    /// pins the accepted/rejected value set directly. NOTE: deliberately NOT
    /// exercised by *setting* the env var in-process: the suite runs
    /// multi-threaded and every other config test calls `clear_env()`, so a
    /// set-var here races (observed flaky before this rewrite). The repo
    /// precedent is that tests only ever REMOVE `VOKRA_*` keys; process-level
    /// env wiring is covered by the release-binary e2e instead.
    #[test]
    fn startup_piper_g2p_bool_values_parse_and_reject() {
        for (raw, want) in [
            ("1", true),
            ("true", true),
            ("TRUE", true),
            ("0", false),
            ("false", false),
            (" true ", true), // trimmed
        ] {
            assert_eq!(
                parse_bool("VOKRA_PIPER_G2P", raw).unwrap(),
                want,
                "value {raw:?}"
            );
        }
        // A typo'd bool must be a hard error, never silently read as "off"
        // (FR-EX-08).
        for raw in ["yes", "no", "on", "off", "2", ""] {
            let err = parse_bool("VOKRA_PIPER_G2P", raw).unwrap_err();
            assert!(
                matches!(err, ConfigError::InvalidValue { ref flag, .. } if flag == "VOKRA_PIPER_G2P"),
                "value {raw:?} must be InvalidValue, got {err}"
            );
        }
    }

    #[test]
    fn startup_piper_g2p_toml_sets_and_cli_overrides() {
        clear_env();
        with_temp_toml("g2p", "piper_g2p = false\n", |p| {
            // TOML alone → explicit false (distinct from unset None).
            let cfg = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap();
            assert_eq!(cfg.piper_g2p, Some(false), "TOML false must be explicit");
            assert!(!cfg.piper_g2p_enabled());

            // CLI switch must beat TOML (precedence CLI > env > TOML).
            let cfg = parse_args([
                "vokra-server",
                "--config",
                p.to_str().unwrap(),
                "--piper-g2p",
            ])
            .unwrap();
            assert_eq!(
                cfg.piper_g2p,
                Some(true),
                "CLI --piper-g2p must override TOML false"
            );
        });
        with_temp_toml("g2p-on", "piper_g2p = true\n", |p| {
            let cfg = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap();
            assert_eq!(cfg.piper_g2p, Some(true), "TOML true must enable");
        });
    }

    #[test]
    fn startup_piper_g2p_toml_garbage_is_parse_error() {
        clear_env();
        with_temp_toml("g2p-bad", "piper_g2p = maybe\n", |p| {
            let err = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap_err();
            assert!(
                matches!(err, ConfigError::ConfigFileParse { .. }),
                "got {err}"
            );
        });
    }

    // ---- cc-39: whisper small / medium / turbo slots (2026-07-19 audit) ----

    /// Every new size flag parses independently, and the sidecar variants
    /// mirror the large-v3 pattern exactly.
    #[test]
    fn startup_cli_sets_whisper_size_paths() {
        use std::path::Path;
        clear_env();
        let cfg = parse_args([
            "vokra-server",
            "--whisper-small",
            "/m/small.gguf",
            "--whisper-small-tokenizer",
            "/m/small.tok",
            "--whisper-medium",
            "/m/medium.gguf",
            "--whisper-medium-tokenizer",
            "/m/medium.tok",
            "--whisper-turbo",
            "/m/turbo.gguf",
            "--whisper-turbo-tokenizer",
            "/m/turbo.tok",
        ])
        .unwrap();
        assert_eq!(
            cfg.whisper_small_gguf.as_deref(),
            Some(Path::new("/m/small.gguf"))
        );
        assert_eq!(
            cfg.whisper_small_tokenizer.as_deref(),
            Some(Path::new("/m/small.tok"))
        );
        assert_eq!(
            cfg.whisper_medium_gguf.as_deref(),
            Some(Path::new("/m/medium.gguf"))
        );
        assert_eq!(
            cfg.whisper_medium_tokenizer.as_deref(),
            Some(Path::new("/m/medium.tok"))
        );
        assert_eq!(
            cfg.whisper_turbo_gguf.as_deref(),
            Some(Path::new("/m/turbo.gguf"))
        );
        assert_eq!(
            cfg.whisper_turbo_tokenizer.as_deref(),
            Some(Path::new("/m/turbo.tok"))
        );
        // Unset sizes stay unset — no cross-contamination between slots.
        assert!(cfg.whisper_large_v3_gguf.is_none());
    }

    /// A dangling size flag is a hard error, exactly like `--piper-plus`.
    #[test]
    fn startup_whisper_size_flag_missing_value_is_error() {
        clear_env();
        for flag in ["--whisper-small", "--whisper-medium", "--whisper-turbo"] {
            let err = parse_args(["vokra-server", flag]).unwrap_err();
            assert!(
                matches!(err, ConfigError::MissingValue(ref s) if s == flag),
                "{flag} without a value must be MissingValue, got {err:?}"
            );
        }
    }

    /// TOML keys mirror the flag names with underscores, and CLI still wins.
    #[test]
    fn startup_toml_sets_whisper_sizes_and_cli_overrides() {
        use std::path::Path;
        clear_env();
        with_temp_toml(
            "sizes",
            "whisper_small = \"/t/small.gguf\"\nwhisper_medium = \"/t/medium.gguf\"\n\
             whisper_turbo = \"/t/turbo.gguf\"\n",
            |p| {
                let cfg = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap();
                assert_eq!(
                    cfg.whisper_small_gguf.as_deref(),
                    Some(Path::new("/t/small.gguf"))
                );
                assert_eq!(
                    cfg.whisper_medium_gguf.as_deref(),
                    Some(Path::new("/t/medium.gguf"))
                );
                assert_eq!(
                    cfg.whisper_turbo_gguf.as_deref(),
                    Some(Path::new("/t/turbo.gguf"))
                );

                let cli_wins = parse_args([
                    "vokra-server",
                    "--config",
                    p.to_str().unwrap(),
                    "--whisper-turbo",
                    "/cli/turbo.gguf",
                ])
                .unwrap();
                assert_eq!(
                    cli_wins.whisper_turbo_gguf.as_deref(),
                    Some(Path::new("/cli/turbo.gguf")),
                    "CLI > TOML for the turbo slot"
                );
            },
        );
    }

    /// The help text is the operator's contract: every new flag and env key
    /// must be discoverable there (a flag that works but is undocumented is
    /// how the base+large-v3-only gap survived this long).
    #[test]
    fn startup_help_text_documents_every_new_flag() {
        for token in [
            "--whisper-small",
            "--whisper-small-tokenizer",
            "--whisper-medium",
            "--whisper-medium-tokenizer",
            "--whisper-turbo",
            "--whisper-turbo-tokenizer",
            "VOKRA_WHISPER_SMALL",
            "VOKRA_WHISPER_MEDIUM",
            "VOKRA_WHISPER_TURBO",
            "--backend",
            "--model-backend",
            "VOKRA_BACKEND",
            "VOKRA_MODEL_BACKENDS",
            "--features metal",
        ] {
            assert!(
                HELP_TEXT.contains(token),
                "HELP_TEXT must document {token:?}"
            );
        }
    }

    // ---- cc-30: backend selection (2026-07-19 audit) ----

    #[test]
    fn startup_backend_defaults_to_cpu() {
        clear_env();
        let cfg = parse_args(["vokra-server"]).unwrap();
        assert_eq!(cfg.backend, None, "unset must stay None");
        assert_eq!(cfg.backend_or_default(), BackendKind::Cpu);
        assert!(cfg.model_backends.is_empty());
    }

    /// Every backend name the CLI accepts parses here too — including ones
    /// this build may not have compiled in. Selectability is checked later,
    /// with a different (rebuild-instruction) error.
    #[test]
    fn startup_backend_names_parse_and_reject_unknown() {
        clear_env();
        for (name, want) in [
            ("cpu", BackendKind::Cpu),
            ("metal", BackendKind::Metal),
            ("cuda", BackendKind::Cuda),
            ("vulkan", BackendKind::Vulkan),
        ] {
            let cfg = parse_args(["vokra-server", "--backend", name]).unwrap();
            assert_eq!(cfg.backend, Some(want), "--backend {name}");
            assert_eq!(cfg.backend_or_default(), want);
        }
        for bad in ["npu", "", "Metal", "gpu"] {
            let err = parse_args(["vokra-server", "--backend", bad]).unwrap_err();
            assert!(
                matches!(err, ConfigError::InvalidValue { ref flag, .. } if flag == "--backend"),
                "--backend {bad:?} must be InvalidValue, got {err:?}"
            );
        }
    }

    #[test]
    fn startup_model_backend_pairs_parse_per_slot() {
        clear_env();
        let cfg = parse_args([
            "vokra-server",
            "--model-backend",
            "whisper-base=metal",
            "--model-backend",
            "piper-plus=cpu",
            "--model-backend",
            "whisper-turbo=cuda",
        ])
        .unwrap();
        assert_eq!(
            cfg.model_backends.get(ModelSlot::WhisperBase),
            Some(BackendKind::Metal)
        );
        assert_eq!(
            cfg.model_backends.get(ModelSlot::PiperPlus),
            Some(BackendKind::Cpu)
        );
        assert_eq!(
            cfg.model_backends.get(ModelSlot::WhisperTurbo),
            Some(BackendKind::Cuda)
        );
        // Untouched slots stay unset (they inherit the global default).
        assert_eq!(cfg.model_backends.get(ModelSlot::Kokoro), None);
    }

    /// Malformed pairs, unknown slots, unknown backends, and a repeated slot
    /// are each an explicit error — never a dropped or arbitrarily-resolved
    /// override (FR-EX-08).
    #[test]
    fn startup_model_backend_rejects_malformed_unknown_and_duplicate() {
        clear_env();
        for bad in [
            "whisper-base",         // no `=`
            "whisper-xl=metal",     // unknown slot
            "whisper-base=quantum", // unknown backend
            "=metal",               // empty slot
        ] {
            let err = parse_args(["vokra-server", "--model-backend", bad]).unwrap_err();
            assert!(
                matches!(err, ConfigError::InvalidValue { .. }),
                "--model-backend {bad:?} must be InvalidValue, got {err:?}"
            );
        }
        let err = parse_args([
            "vokra-server",
            "--model-backend",
            "whisper-base=metal",
            "--model-backend",
            "whisper-base=cpu",
        ])
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(err, ConfigError::InvalidValue { .. })
                && msg.contains("already has a backend"),
            "a repeated slot must be rejected, got {msg}"
        );
    }

    /// TOML carries the comma-list spelling; a CLI flag for the SAME slot
    /// must override it rather than trip the duplicate check (the layers are
    /// merged, not concatenated).
    #[test]
    fn startup_model_backends_toml_list_and_cli_precedence() {
        clear_env();
        with_temp_toml(
            "backends",
            "backend = \"cpu\"\nmodel_backends = \"whisper-base=metal,kokoro=cuda\"\n",
            |p| {
                let cfg = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap();
                assert_eq!(cfg.backend, Some(BackendKind::Cpu));
                assert_eq!(
                    cfg.model_backends.get(ModelSlot::WhisperBase),
                    Some(BackendKind::Metal)
                );
                assert_eq!(
                    cfg.model_backends.get(ModelSlot::Kokoro),
                    Some(BackendKind::Cuda)
                );

                // Same slot on the CLI: overrides, does NOT error as duplicate.
                let cli_wins = parse_args([
                    "vokra-server",
                    "--config",
                    p.to_str().unwrap(),
                    "--model-backend",
                    "whisper-base=cpu",
                    "--backend",
                    "vulkan",
                ])
                .unwrap();
                assert_eq!(
                    cli_wins.model_backends.get(ModelSlot::WhisperBase),
                    Some(BackendKind::Cpu),
                    "CLI override must beat the TOML list for the same slot"
                );
                assert_eq!(
                    cli_wins.model_backends.get(ModelSlot::Kokoro),
                    Some(BackendKind::Cuda),
                    "slots the CLI did not mention keep their TOML value"
                );
                assert_eq!(cli_wins.backend, Some(BackendKind::Vulkan), "CLI > TOML");
            },
        );
    }

    /// A duplicate WITHIN the comma-list is still ambiguous, so it is still
    /// an error (the cross-layer merge is the only place replacement is OK).
    #[test]
    fn startup_model_backends_list_rejects_intra_layer_duplicate() {
        clear_env();
        with_temp_toml(
            "backends-dup",
            "model_backends = \"whisper-base=metal,whisper-base=cpu\"\n",
            |p| {
                let err =
                    parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap_err();
                assert!(
                    matches!(err, ConfigError::ConfigFileParse { .. }),
                    "got {err}"
                );
            },
        );
        // A trailing comma is tolerated (it is not ambiguous).
        with_temp_toml(
            "backends-trailing",
            "model_backends = \"whisper-base=metal,\"\n",
            |p| {
                let cfg = parse_args(["vokra-server", "--config", p.to_str().unwrap()]).unwrap();
                assert_eq!(
                    cfg.model_backends.get(ModelSlot::WhisperBase),
                    Some(BackendKind::Metal)
                );
            },
        );
    }

    /// Every `ModelSlot` must be reachable by its documented name — a slot
    /// that exists in the enum but parses to `None` would be an override the
    /// operator can never express.
    #[test]
    fn startup_every_model_slot_name_round_trips() {
        for (slot, name) in ModelSlot::ALL.iter().zip(ModelSlot::ALL_NAMES.iter()) {
            assert_eq!(slot.as_str(), *name, "ALL and ALL_NAMES must stay aligned");
            assert_eq!(
                ModelSlot::parse(name),
                Some(*slot),
                "slot name {name:?} must parse back"
            );
        }
        assert_eq!(ModelSlot::parse("whisper-xl"), None);
    }
}
