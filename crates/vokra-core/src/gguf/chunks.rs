//! `vokra.*` GGUF metadata key namespace (chunk naming specification).
//!
//! This module is the in-code home of the "`vokra.*` chunk specification"
//! (M0-03-T08). Audio-specific metadata is stored under a **`vokra.` key
//! prefix** so it can never collide with llama.cpp's own namespaces
//! (`general.*`, `<arch>.*`, `tokenizer.*`), satisfying FR-LD-02 / IF-07 and
//! the rationale in CLAUDE.md design note 3.
//!
//! # Prefix rule
//!
//! Every Vokra-specific key begins with `vokra.`. Two sub-namespaces are
//! defined in M0:
//!
//! - `vokra.model.*` — model identification (**proposal**).
//! - `vokra.frontend.*` — the `frontend_spec` (front-end feature-extraction
//!   parameters), one key per [`crate::gguf::FrontendSpec`] field.
//!
//! # `vokra.frontend.*` keys and value types
//!
//! The 13 `frontend_spec` fields are transcribed verbatim from CLAUDE.md /
//! FR-LD-03: `{n_fft, hop, win_length, window_type, mel_norm, htk_mode, fmin,
//! fmax, n_mels, pad_mode, dc_offset_removal, pre_emphasis, sample_rate}`.
//! The GGUF value type chosen for each key is a **proposal** of this ticket
//! (M0-03-T08) — the field list is transcribed, the type mapping is designed
//! here:
//!
//! | key                              | GGUF value type |
//! |----------------------------------|-----------------|
//! | `vokra.frontend.n_fft`           | `UINT32`        |
//! | `vokra.frontend.hop`             | `UINT32`        |
//! | `vokra.frontend.win_length`      | `UINT32`        |
//! | `vokra.frontend.window_type`     | `STRING`        |
//! | `vokra.frontend.mel_norm`        | `STRING`        |
//! | `vokra.frontend.htk_mode`        | `BOOL`          |
//! | `vokra.frontend.fmin`            | `FLOAT32`       |
//! | `vokra.frontend.fmax`            | `FLOAT32`       |
//! | `vokra.frontend.n_mels`          | `UINT32`        |
//! | `vokra.frontend.pad_mode`        | `STRING`        |
//! | `vokra.frontend.dc_offset_removal` | `BOOL`        |
//! | `vokra.frontend.pre_emphasis`    | `FLOAT32`       |
//! | `vokra.frontend.sample_rate`     | `UINT32`        |
//!
//! # M0 / M1 scope boundary
//!
//! M0 covers only **writing and reading** these chunks. The *inspection* of
//! `frontend_spec` — the bit-exact match check that must `warn`/`fail` when a
//! model's front-end does not match the runtime's (FR-LD-03) — is a v0.1 MVP
//! concern owned by **M1-03** and is deliberately absent here. The
//! provenance / license metadata of FR-CP-05 (M1) is expected to land in this
//! same `vokra.*` namespace later (e.g. `vokra.provenance.*`).
//!
//! # Silero VAD note
//!
//! Silero VAD has no STFT/mel front-end that Vokra controls — its internal
//! pseudo-STFT is an implementation detail hidden inside the 1:1 subgraph
//! (FR-LD-06, M0-05). Its GGUF therefore carries only `vokra.model.*` keys and
//! **omits the `vokra.frontend.*` chunk** entirely (the converter must not
//! invent front-end values it cannot source).

/// Prefix shared by every Vokra-specific metadata key.
pub const VOKRA_PREFIX: &str = "vokra.";

/// Model architecture tag, e.g. `"whisper"` (**proposal**, `STRING`).
pub const KEY_MODEL_ARCH: &str = "vokra.model.arch";

/// Human-readable model name, e.g. `"whisper-base"` (**proposal**, `STRING`).
pub const KEY_MODEL_NAME: &str = "vokra.model.name";

/// `vokra.frontend.n_fft` — FFT window size (`UINT32`).
pub const KEY_FRONTEND_N_FFT: &str = "vokra.frontend.n_fft";
/// `vokra.frontend.hop` — hop length between frames (`UINT32`).
pub const KEY_FRONTEND_HOP: &str = "vokra.frontend.hop";
/// `vokra.frontend.win_length` — analysis window length (`UINT32`).
pub const KEY_FRONTEND_WIN_LENGTH: &str = "vokra.frontend.win_length";
/// `vokra.frontend.window_type` — window function name (`STRING`).
pub const KEY_FRONTEND_WINDOW_TYPE: &str = "vokra.frontend.window_type";
/// `vokra.frontend.mel_norm` — mel filterbank normalization mode (`STRING`).
pub const KEY_FRONTEND_MEL_NORM: &str = "vokra.frontend.mel_norm";
/// `vokra.frontend.htk_mode` — HTK vs. Slaney mel scale (`BOOL`).
pub const KEY_FRONTEND_HTK_MODE: &str = "vokra.frontend.htk_mode";
/// `vokra.frontend.fmin` — lowest mel band edge, Hz (`FLOAT32`).
pub const KEY_FRONTEND_FMIN: &str = "vokra.frontend.fmin";
/// `vokra.frontend.fmax` — highest mel band edge, Hz (`FLOAT32`).
pub const KEY_FRONTEND_FMAX: &str = "vokra.frontend.fmax";
/// `vokra.frontend.n_mels` — number of mel bands (`UINT32`).
pub const KEY_FRONTEND_N_MELS: &str = "vokra.frontend.n_mels";
/// `vokra.frontend.pad_mode` — signal padding mode (`STRING`).
pub const KEY_FRONTEND_PAD_MODE: &str = "vokra.frontend.pad_mode";
/// `vokra.frontend.dc_offset_removal` — remove DC offset before framing (`BOOL`).
pub const KEY_FRONTEND_DC_OFFSET_REMOVAL: &str = "vokra.frontend.dc_offset_removal";
/// `vokra.frontend.pre_emphasis` — pre-emphasis coefficient, `0.0` = off (`FLOAT32`).
pub const KEY_FRONTEND_PRE_EMPHASIS: &str = "vokra.frontend.pre_emphasis";
/// `vokra.frontend.sample_rate` — input sample rate, Hz (`UINT32`).
pub const KEY_FRONTEND_SAMPLE_RATE: &str = "vokra.frontend.sample_rate";

/// Standard GGUF key for the global tensor-data alignment (`UINT32`).
///
/// Not a `vokra.*` key: this is the upstream `general.alignment` field. When
/// absent the alignment defaults to `32` (see the GGUF spec).
pub const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

/// Returns `true` if `key` lies in the `vokra.*` namespace.
pub fn is_vokra_key(key: &str) -> bool {
    key.starts_with(VOKRA_PREFIX)
}
