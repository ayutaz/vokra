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
//! concern owned by **M1-03** and is deliberately absent here.
//!
//! # `vokra.provenance.*` keys (FR-CP-05 / FR-CP-03, M2-13)
//!
//! The provenance / license metadata of FR-CP-05 lives in the
//! `vokra.provenance.*` sub-namespace. These keys let the runtime classify a
//! model's **weight license** and enforce the CC-BY-NC research-flag gate
//! (FR-CP-03; see [`crate::compliance`]). They record the *weight* license,
//! which is independent of the crate/source-code license (a model can be
//! MIT-code but CC-BY-NC-weight, e.g. F5-TTS / EnCodec — see
//! `docs/license-audit.md` §3).
//!
//! | key                              | GGUF value type | meaning |
//! |----------------------------------|-----------------|---------|
//! | `vokra.provenance.weight_license`| `STRING`        | resolved [`LicenseClass`](crate::compliance::LicenseClass) canonical name (e.g. `"non-commercial"`) — an explicit, highest-priority override |
//! | `vokra.provenance.license`       | `STRING`        | raw weight license string (e.g. `"CC-BY-NC-4.0"`, `"MIT"`) |
//! | `vokra.provenance.model_id`      | `STRING`        | model identifier used for the built-in registry lookup (e.g. `"f5-tts"`) |
//! | `vokra.provenance.source`        | `STRING`        | free-form upstream source note (URL / repo), advisory only |
//!
//! The converter side of this chunk (writing it) is minimal in M2-13: see
//! [`crate::compliance::stamp_provenance`]. When the chunk is absent the
//! runtime falls back to the built-in registry keyed on `vokra.model.*`, and
//! finally to [`LicenseClass::Unknown`](crate::compliance::LicenseClass) —
//! which is **fail-closed** (gate required), never a silent pass.
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

/// `vokra.provenance.weight_license` — explicit resolved weight
/// [`LicenseClass`](crate::compliance::LicenseClass) canonical name (`STRING`).
///
/// Highest-priority signal for the compliance gate (FR-CP-03): when present and
/// parseable it wins over the raw license string and the registry. Written by
/// [`crate::compliance::stamp_provenance`].
pub const KEY_PROVENANCE_WEIGHT_LICENSE: &str = "vokra.provenance.weight_license";

/// `vokra.provenance.license` — raw weight license string, e.g.
/// `"CC-BY-NC-4.0"` / `"MIT"` (`STRING`, FR-CP-05).
pub const KEY_PROVENANCE_LICENSE: &str = "vokra.provenance.license";

/// `vokra.provenance.model_id` — model identifier for the built-in license
/// registry lookup, e.g. `"f5-tts"` / `"encodec"` (`STRING`, FR-CP-05).
pub const KEY_PROVENANCE_MODEL_ID: &str = "vokra.provenance.model_id";

/// `vokra.provenance.source` — advisory upstream source note (URL / repo),
/// not used for classification (`STRING`, FR-CP-05).
pub const KEY_PROVENANCE_SOURCE: &str = "vokra.provenance.source";

/// Standard GGUF key for the global tensor-data alignment (`UINT32`).
///
/// Not a `vokra.*` key: this is the upstream `general.alignment` field. When
/// absent the alignment defaults to `32` (see the GGUF spec).
pub const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

/// Returns `true` if `key` lies in the `vokra.*` namespace.
pub fn is_vokra_key(key: &str) -> bool {
    key.starts_with(VOKRA_PREFIX)
}
