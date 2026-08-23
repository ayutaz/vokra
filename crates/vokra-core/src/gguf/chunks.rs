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

/// `vokra.provenance.attribution` — the human-readable attribution text an
/// `AttributionRequired` weight (e.g. Moshi / Mimi CC-BY 4.0) obliges the
/// deployer to display (`STRING`, FR-MD-09 — M4-06). Written by
/// [`crate::compliance::stamp_attribution`]; read back through
/// [`crate::compliance::resolve_attribution`], which also supplies a
/// registry fallback so an attribution-required weight is never left
/// without a displayable string.
pub const KEY_PROVENANCE_ATTRIBUTION: &str = "vokra.provenance.attribution";

// `vokra.silero.*` — Silero VAD subgraph metadata (M0-05 / vokra-vad-micro).
//
// The Silero VAD subgraph is architecturally stable across v5 and v6.2.1
// (identical topology per upstream `snakers4/silero-vad` `tinygrad_model.py`
// and `utils_vad.py` at both tags — same `[bins, frames]` conv shapes, same
// `LSTMCell(128, 128)`, same context prepending, same state layout `[2,1,128]`).
// What differs is the **trained weights** and the release provenance. This key
// carries the release tag so consumers (parity fixtures, license attribution,
// forward-branch validation, future architecture divergence) can distinguish
// artifacts. Absent means "produced before variant tagging existed" and is
// treated as [`crate::gguf::silero::SileroVariant::V5`] for backward
// compatibility — the pre-2026-07-29 fixture GGUF and every converter run
// before this key existed omit it.

/// `vokra.silero.version` — Silero VAD release tag (`STRING`, e.g. `"v5"` or
/// `"v6.2.1"`).
///
/// Written by [`crate::gguf::silero::stamp_variant`], read back by
/// [`crate::gguf::silero::SileroVariant::from_gguf`]. Unknown values are a
/// fail-closed [`crate::VokraError::ModelLoad`] (FR-EX-08) — never a silent
/// V5 fallback, because a tag we do not recognize may imply a topology change
/// this build cannot honor.
pub const KEY_SILERO_VERSION: &str = "vokra.silero.version";

/// `vokra.schema.version` — Vokra GGUF **schema** generation (`UINT32`).
///
/// Hand-bumped ([`crate::gguf::SCHEMA_VERSION`]) whenever a converter starts emitting a
/// metadata or tensor group that loaders may rely on. Its whole purpose is to
/// make a **stale artifact** — a GGUF produced by an older converter, missing a
/// group newer code expects — visible at load instead of degrading quietly.
///
/// Absent means "produced before schema stamping existed", which is a real
/// answer, not an error: every GGUF converted up to that point lacks the key.
/// Read it through [`crate::gguf::schema::schema_version`].
///
/// **Not** the crate version: `CARGO_PKG_VERSION` stayed `0.1.0-alpha.0`
/// across M0…M5 and now advances only with releases, so it cannot distinguish
/// converter generations within one release. That is exactly the case this
/// key exists for.
pub const KEY_SCHEMA_VERSION: &str = "vokra.schema.version";

/// `vokra.schema.producer` — the converter build that wrote the file, e.g.
/// `"vokra-convert 0.1.0"` (`STRING`).
///
/// Diagnostic companion to [`KEY_SCHEMA_VERSION`]: it identifies the build for
/// a bug report but must never gate behaviour, because the version string does
/// is not a schema compatibility contract.
pub const KEY_SCHEMA_PRODUCER: &str = "vokra.schema.producer";

// `vokra.quant.*` — quantization policy metadata (M2-08, FR-QT-02).
//
// The runtime reads its quantization policy **only** from GGUF metadata (no
// config-file / TOML / serde path exists — NFR-DS-02 zero-dep). The offline
// converter writes these keys via `QuantPolicy::write_to_gguf_builder`, and
// the runtime reads them back via `QuantPolicy::from_gguf`.

/// `vokra.quant.default_scheme` — canonical alias of the policy's fallback
/// [`QuantScheme`](crate::quant::QuantScheme) applied when no rule matches
/// (`STRING`, e.g. `"fp16"`, `"w4a16-q4k"`).
pub const KEY_QUANT_DEFAULT_SCHEME: &str = "vokra.quant.default_scheme";

/// `vokra.quant.rule_count` — number of ordered
/// [`QuantRule`](crate::quant::QuantRule) entries stored under
/// `vokra.quant.rule.{i}.*` (`UINT64`).
pub const KEY_QUANT_RULE_COUNT: &str = "vokra.quant.rule_count";

/// Prefix under which each rule's three fields (`pattern_kind`, `pattern`,
/// `scheme`) are stored, e.g. `vokra.quant.rule.0.pattern_kind`.
pub const PREFIX_QUANT_RULE: &str = "vokra.quant.rule.";

/// `vokra.quant.hifigan_int8_opt_in` — HiFi-GAN INT8 opt-in gate flag
/// (`BOOL`, default `false`; FR-QT-03 / M2-08-T10).
pub const KEY_QUANT_HIFIGAN_INT8_OPT_IN: &str = "vokra.quant.hifigan_int8_opt_in";

/// `vokra.quant.hifigan_int8_calibration_ref` — opaque handle to the HiFi-GAN
/// INT8 calibration blob (`STRING`, optional; required when
/// [`KEY_QUANT_HIFIGAN_INT8_OPT_IN`] is `true`).
pub const KEY_QUANT_HIFIGAN_INT8_CALIBRATION_REF: &str = "vokra.quant.hifigan_int8_calibration_ref";

/// `vokra.quant.min_dtype_enforced` — audit record listing op-kind identifiers
/// the converter validated against the built-in
/// [`MinDtypeRegistry`](crate::quant) (`Array<String>`, optional; M2-08-T08).
pub const KEY_QUANT_MIN_DTYPE_ENFORCED: &str = "vokra.quant.min_dtype_enforced";

/// Standard GGUF key for the global tensor-data alignment (`UINT32`).
///
/// Not a `vokra.*` key: this is the upstream `general.alignment` field. When
/// absent the alignment defaults to `32` (see the GGUF spec).
pub const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

/// Returns `true` if `key` lies in the `vokra.*` namespace.
pub fn is_vokra_key(key: &str) -> bool {
    key.starts_with(VOKRA_PREFIX)
}

// Audit 2026-08-10 (Rank 14, test-coverage-audit workflow): every KEY_*
// string constant in this module is part of the on-disk `vokra.*` wire
// contract. A silent rename would compile clean but break every previously
// published Vokra GGUF (readers look up by string literal). None had a
// direct test — coverage was purely incidental via converter round-trips.
// This module pins each constant against its documented literal and asserts
// the `vokra.` prefix invariant so a rename cannot pass CI unnoticed.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_contract_model_keys() {
        assert_eq!(KEY_MODEL_ARCH, "vokra.model.arch");
        assert_eq!(KEY_MODEL_NAME, "vokra.model.name");
    }

    #[test]
    fn wire_contract_frontend_keys() {
        assert_eq!(KEY_FRONTEND_N_FFT, "vokra.frontend.n_fft");
        assert_eq!(KEY_FRONTEND_HOP, "vokra.frontend.hop");
        assert_eq!(KEY_FRONTEND_WIN_LENGTH, "vokra.frontend.win_length");
        assert_eq!(KEY_FRONTEND_WINDOW_TYPE, "vokra.frontend.window_type");
        assert_eq!(KEY_FRONTEND_MEL_NORM, "vokra.frontend.mel_norm");
        assert_eq!(KEY_FRONTEND_HTK_MODE, "vokra.frontend.htk_mode");
        assert_eq!(KEY_FRONTEND_FMIN, "vokra.frontend.fmin");
        assert_eq!(KEY_FRONTEND_FMAX, "vokra.frontend.fmax");
        assert_eq!(KEY_FRONTEND_N_MELS, "vokra.frontend.n_mels");
        assert_eq!(KEY_FRONTEND_PAD_MODE, "vokra.frontend.pad_mode");
        assert_eq!(
            KEY_FRONTEND_DC_OFFSET_REMOVAL,
            "vokra.frontend.dc_offset_removal",
        );
        assert_eq!(KEY_FRONTEND_PRE_EMPHASIS, "vokra.frontend.pre_emphasis");
        assert_eq!(KEY_FRONTEND_SAMPLE_RATE, "vokra.frontend.sample_rate");
    }

    #[test]
    fn wire_contract_provenance_keys() {
        assert_eq!(
            KEY_PROVENANCE_WEIGHT_LICENSE,
            "vokra.provenance.weight_license",
        );
        assert_eq!(KEY_PROVENANCE_LICENSE, "vokra.provenance.license");
        assert_eq!(KEY_PROVENANCE_MODEL_ID, "vokra.provenance.model_id");
        assert_eq!(KEY_PROVENANCE_SOURCE, "vokra.provenance.source");
        assert_eq!(KEY_PROVENANCE_ATTRIBUTION, "vokra.provenance.attribution",);
    }

    #[test]
    fn wire_contract_schema_and_quant_keys() {
        assert_eq!(KEY_SILERO_VERSION, "vokra.silero.version");
        assert_eq!(KEY_SCHEMA_VERSION, "vokra.schema.version");
        assert_eq!(KEY_SCHEMA_PRODUCER, "vokra.schema.producer");
        assert_eq!(KEY_QUANT_DEFAULT_SCHEME, "vokra.quant.default_scheme");
        assert_eq!(KEY_QUANT_RULE_COUNT, "vokra.quant.rule_count");
        assert_eq!(
            KEY_QUANT_HIFIGAN_INT8_OPT_IN,
            "vokra.quant.hifigan_int8_opt_in",
        );
        assert_eq!(
            KEY_QUANT_HIFIGAN_INT8_CALIBRATION_REF,
            "vokra.quant.hifigan_int8_calibration_ref",
        );
        assert_eq!(
            KEY_QUANT_MIN_DTYPE_ENFORCED,
            "vokra.quant.min_dtype_enforced",
        );
    }

    #[test]
    fn general_alignment_key_is_upstream_ggml_not_vokra() {
        // `general.alignment` is the upstream ggml spec key (not `vokra.*`).
        // is_vokra_key MUST reject it — otherwise a Vokra reader would
        // treat every ggml alignment header as a Vokra-owned key.
        assert_eq!(KEY_GENERAL_ALIGNMENT, "general.alignment");
        assert!(!is_vokra_key(KEY_GENERAL_ALIGNMENT));
    }

    #[test]
    fn is_vokra_key_positive_and_negative() {
        // Positive: every KEY_* const under the vokra.* namespace must
        // match. Sampling a few is enough — the wire-contract pins above
        // catch drift on the literals themselves.
        assert!(is_vokra_key(KEY_FRONTEND_N_FFT));
        assert!(is_vokra_key(KEY_PROVENANCE_LICENSE));
        assert!(is_vokra_key(KEY_SCHEMA_VERSION));
        // Negative: unrelated / llama.cpp-owned keys must NOT match.
        assert!(!is_vokra_key("general.name"));
        assert!(!is_vokra_key("general.architecture"));
        assert!(!is_vokra_key("llama.attention.head_count"));
        assert!(!is_vokra_key(""));
    }

    #[test]
    fn every_vokra_key_starts_with_vokra_prefix() {
        // Cross-check: the wire-contract tests pin literal values, and
        // this test asserts the prefix invariant on all of them. If a new
        // `vokra.*` key is added without a wire-contract pin, is_vokra_key
        // should still say true; if a `general.*` key is added, it must
        // fail here so the reviewer notices.
        let vokra_keys: [&str; 26] = [
            KEY_MODEL_ARCH,
            KEY_MODEL_NAME,
            KEY_FRONTEND_N_FFT,
            KEY_FRONTEND_HOP,
            KEY_FRONTEND_WIN_LENGTH,
            KEY_FRONTEND_WINDOW_TYPE,
            KEY_FRONTEND_MEL_NORM,
            KEY_FRONTEND_HTK_MODE,
            KEY_FRONTEND_FMIN,
            KEY_FRONTEND_FMAX,
            KEY_FRONTEND_N_MELS,
            KEY_FRONTEND_PAD_MODE,
            KEY_FRONTEND_DC_OFFSET_REMOVAL,
            KEY_FRONTEND_PRE_EMPHASIS,
            KEY_FRONTEND_SAMPLE_RATE,
            KEY_PROVENANCE_WEIGHT_LICENSE,
            KEY_PROVENANCE_LICENSE,
            KEY_PROVENANCE_MODEL_ID,
            KEY_PROVENANCE_SOURCE,
            KEY_PROVENANCE_ATTRIBUTION,
            KEY_SILERO_VERSION,
            KEY_SCHEMA_VERSION,
            KEY_SCHEMA_PRODUCER,
            KEY_QUANT_DEFAULT_SCHEME,
            KEY_QUANT_RULE_COUNT,
            KEY_QUANT_HIFIGAN_INT8_OPT_IN,
        ];
        for key in vokra_keys {
            assert!(
                is_vokra_key(key),
                "vokra.* key {key:?} did not pass is_vokra_key",
            );
        }
    }
}
