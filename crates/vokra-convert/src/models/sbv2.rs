//! Style-Bert-VITS2 v2 (SBV2) official checkpoint → GGUF conversion
//! (SBV2 v2 plan Task 25, 2026-07-26).
//!
//! Input: an upstream SBV2 v2 safetensors checkpoint (`litagin02/
//! style_bert_vits2` family) plus an optional JSON config side-car.
//! Output: a GGUF carrying every float tensor plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks, and — only when
//! a config side-car is supplied — the `vokra.sbv2.*` hparam chunk group
//! `SbV2Model::from_gguf` (Task 24, `crates/vokra-models/src/sbv2/mod.rs`)
//! is written to read.
//!
//! # References (permissive only)
//!
//! - VITS paper: arXiv:2106.06103 (Kim et al. 2021)
//! - jaywalnut310/vits (MIT): VITS core reference
//! - VITS2 paper: arXiv:2307.16430
//! - SBV2 model card / `config.json` / `.safetensors` header: information
//!   feed only (tensor names / shapes / dtypes) — no code is read from it
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//! - Any AGPL derivative of the above.
//!
//! # Why this lives in `vokra-convert`, not `vokra-models`
//!
//! The SBV2 v2 design doc
//! (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §4) originally
//! drew this converter as `vokra-models::sbv2::converter` plus a
//! `vokra-convert::models::sbv2` "thin shim" that calls back into it for
//! the `ModelKind` dispatch wiring. That is the exact shape Task 11's
//! DeBERTa converter rejected, for the same reason: the converter needs
//! `vokra-convert`'s own crate-private safetensors reader
//! (`crate::safetensors::SafetensorsFile`) and JSON parser (`crate::json`)
//! — neither is re-exported from this crate's public API — so putting the
//! body in `vokra-models` would force `vokra-models` to gain `vokra-convert`
//! as a *normal* (non-dev) dependency. `vokra-models` already depends on
//! `vokra-convert` **only as a dev-dependency** (M4-04-T10/T11's roundtrip
//! tests, `crates/vokra-models/Cargo.toml`); if `vokra-convert`'s shim then
//! depended on `vokra-models` to call back in (as the original shim plan
//! requires), the two *normal* dependency edges would form a genuine cycle
//! (`vokra-models -> vokra-convert -> vokra-models`), which Cargo rejects
//! outright. This module resolves the cycle exactly like `deberta_v2.rs`
//! did: the real implementation lives directly in `vokra-convert` (which
//! already owns every other model's converter), so `vokra-models` gains no
//! new dependency and the `ModelKind::SbV2` dispatch wiring needs only a
//! `pub use` re-export here, not a new module in `vokra-models`. Task
//! 26/27/28's own tests can reach [`convert_sbv2_file`] directly via
//! `vokra-models`'s existing `vokra-convert` dev-dependency — no shim is
//! needed there either.
//!
//! # BF16 pass-through — mirror of `deberta_v2` / `funcodec` / `wespeaker`
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their upstream
//! safetensors names (GGUF types 0 / 1 / 30). No convert-time widening —
//! the runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # TODO(owner): tensor name mapping needs a real checkpoint (Task 30)
//!
//! Every tensor is emitted under its **upstream safetensors name,
//! verbatim** — this converter does not rename e.g. an upstream
//! `dec.ups.0.weight`-shaped name to the `sbv2.decoder.upsample.0.weight`
//! name `SbV2Model::from_gguf` reads (see that method's doc, on
//! `SbV2Model` in `crates/vokra-models/src/sbv2/mod.rs`, for the complete
//! target tensor-name hierarchy this converter's Task 30 follow-up must
//! produce). Building that mapping table honestly requires a real
//! checkpoint header dump (`tools/parity/sbv2_prepare_checkpoint.py` +
//! `tools/parity/sbv2_dump_reference.py`, design doc §10) — until it
//! lands, a GGUF this converter produces is a provenance-correct,
//! byte-faithful **staging artifact**, not yet loadable by `from_gguf`,
//! mirroring the "Wiring status" posture `deberta_v2.rs` / `funcodec.rs` /
//! `wespeaker.rs` already carry for their own not-yet-consumed outputs. No
//! rename table is guessed here in its place.
//!
//! # M6 refactor (2026-08-06): `language_embed` replaces `wb_embed`
//!
//! The Task 30 rename table, when it lands, must map the real upstream
//! tensor `enc_p.language_emb.weight` (shape `[3, 192]` — verified on
//! `litagin/Style-Bert-VITS2-2.0-base-JP-Extra`) to the loader's
//! `sbv2.text_encoder.language_embed`. The pre-M6 design-doc §7 assumed
//! `enc_p.word_boundary_emb.weight [2, 192]` at this slot; no such tensor
//! exists on the real checkpoint. The runtime side of this refactor
//! (`crates/vokra-models/src/sbv2/text_encoder.rs`
//! `SbV2TextEncoder::language_embed`) is already updated; the loader
//! now reads `sbv2.text_encoder.language_embed` (not `.wb_embed`) and
//! cross-checks its length against `N_LANGUAGES = 3` times `d_model`.
//! This converter's hparam block additionally stamps
//! `vokra.sbv2.n_languages = 3` (see [`write_hparams`]) as a
//! forward-looking metadata anchor for that cross-check.
//!
//! # Hparams — config-side-car-driven, never invented
//!
//! `SbV2Model::from_gguf` requires 22 `vokra.sbv2.*` metadata keys (13
//! top-level dims + 3 decoder scalars + 6 decoder arrays) plus one optional
//! key (`decoder.leaky_relu_slope`, default `0.1`) — 23 keys total, none of
//! which is recoverable from a generic tensor-shape scan the way
//! `deberta_v2.rs`'s `infer_vocab_and_d_model` recovers DeBERTa's
//! `vocab_size` / `d_model` (SBV2's real upstream tensor-naming convention
//! is unknown pending Task 30 — see above — so no shape-scanning heuristic
//! can be trusted yet, and several keys, e.g. `n_tones` / `d_ff` / the
//! decoder's upsample/resblock arrays, are pure architecture choices no
//! tensor shape encodes at all).
//!
//! Rather than invent placeholder numbers — which `SbV2Model::from_gguf`
//! would **not** reliably reject at load (only `d_z` gets an explicit
//! non-zero/even check there; every other scalar field is a bare
//! "key present" check, so a `0` sentinel would silently produce
//! degenerate zero-sized tensors instead of a clean, loud
//! `VokraError::ModelLoad`) — this converter takes the honest path:
//!
//! - `config_side_car = None`: the `vokra.sbv2.*` chunk is **not written at
//!   all** (tensors still pass through). `SbV2Model::from_gguf` then fails
//!   loudly on the first missing key, exactly as designed
//!   ([`ConvertReport::hparams_written`] is `false`).
//! - `config_side_car = Some(path)`: every one of the 22 required keys must
//!   be present in the JSON or this function returns [`ConvertError::Parse`]
//!   naming the missing field (mirrors `models::dac::DacConfig::parse`'s
//!   `req()` closure). [`SbV2Config::parse`] additionally cross-checks the
//!   same internal-consistency invariants `SbV2Model::from_gguf` itself
//!   checks at load time (`d_z` non-zero + even; the three decoder array
//!   lengths agreeing with `upsample_rates`/`resblock_kernel_sizes`; the
//!   flattened dilation array's length matching the sum of the per-branch
//!   dilation counts) — catching a config-authoring mistake at convert time
//!   with a clear message, rather than only much later inside the loader.
//!
//! # No ONNX (permanent)
//!
//! SBV2 ships as an HF-style safetensors checkpoint; this converter never
//! touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::json::{self, JsonValue};
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for SBV2 GGUFs.
pub(crate) const ARCH: &str = "sbv2";
/// `vokra.model.name` — short slug (design doc §9 SKU table:
/// `vokra/sbv2-v2-multilingual-base`), distinct from the full HF
/// `org/repo` path in [`UPSTREAM_HF`] (mirrors the `funcodec` /
/// `wespeaker` / `deberta_v2` convention).
pub(crate) const NAME: &str = "sbv2-v2-multilingual-base";
/// Upstream source family — provenance breadcrumb. Not a single pinned HF
/// repo id: `litagin02`'s SBV2 v2 releases span several checkpoint repos
/// under this account (design doc §2/§9).
pub(crate) const UPSTREAM_HF: &str = "litagin02/style_bert_vits2";
/// Upstream declared weight license (SPDX id, lower-case per
/// `docs/license-audit.md` §3.1). `agpl-3.0` classifies as
/// [`LicenseClass::Copyleft`] (design doc §9 — redistribution is permitted
/// with the original licence preserved, never relabelled; see
/// `LicenseClass::from_license_str`'s share-alike/copyleft ordering
/// rationale).
pub(crate) const DEFAULT_LICENSE: &str = "agpl-3.0";

// --- vokra.sbv2.* metadata keys ------------------------------------------
// The runtime side lives in `crates/vokra-models/src/sbv2/mod.rs`
// (`SbV2Model::from_gguf`) — the two crates share only `vokra-core`, so
// the cross-crate constant duplication rule the CSM / Kokoro / VitsJa
// family converters use applies. Every key name below is copied verbatim
// from that method's own doc (the schema's source of truth, ~23 keys).

// Top-level dims (13 upstream-driven + 1 fixed-architecture = 14 total,
// see `KEY_N_LANGUAGES` below).
const KEY_D_MODEL: &str = "vokra.sbv2.d_model";
const KEY_D_BERT: &str = "vokra.sbv2.d_bert";
const KEY_D_SPEAKER: &str = "vokra.sbv2.d_speaker";
const KEY_N_SPEAKERS: &str = "vokra.sbv2.n_speakers";
const KEY_D_STYLE: &str = "vokra.sbv2.d_style";
const KEY_D_Z: &str = "vokra.sbv2.d_z";
const KEY_N_VOCAB: &str = "vokra.sbv2.n_vocab";
const KEY_N_TONES: &str = "vokra.sbv2.n_tones";
const KEY_D_FF: &str = "vokra.sbv2.d_ff";
const KEY_N_TEXT_LAYERS: &str = "vokra.sbv2.n_text_layers";
const KEY_N_FLOW_LAYERS: &str = "vokra.sbv2.n_flow_layers";
const KEY_N_SDP_LAYERS: &str = "vokra.sbv2.n_sdp_layers";
const KEY_SAMPLE_RATE: &str = "vokra.sbv2.sample_rate";

// M6 refactor (2026-08-06): the SBV2 v2 base checkpoint's real
// `enc_p.language_emb.weight` table is `[3, 192]` (JA/EN/ZH). This value
// is a fixed architectural constant, not a config-authored one — it
// mirrors `crates/vokra-models/src/sbv2/text_encoder.rs`'s
// `N_LANGUAGES = 3`. See the module doc's "M6 refactor" section for the
// primary-source verification behind it and why the metadata is stamped
// forward-looking (the loader's own `language_embed` length check already
// gates on the same value even without this metadata being present).
const KEY_N_LANGUAGES: &str = "vokra.sbv2.n_languages";
/// The fixed value stamped under [`KEY_N_LANGUAGES`] — a hard-coded
/// architectural anchor rather than a config-side-car field, so
/// [`SbV2Config`] does not carry it and [`SbV2Config::parse`] does not
/// read it. Mirrors `vokra_models::sbv2::N_LANGUAGES`.
const N_LANGUAGES: u32 = 3;

// Decoder scalars (3).
const KEY_DECODER_INITIAL_CHANNEL: &str = "vokra.sbv2.decoder.initial_channel";
const KEY_DECODER_CONV_PRE_KERNEL: &str = "vokra.sbv2.decoder.conv_pre_kernel";
const KEY_DECODER_CONV_POST_KERNEL: &str = "vokra.sbv2.decoder.conv_post_kernel";

// Decoder arrays (6).
const KEY_DECODER_UPSAMPLE_RATES: &str = "vokra.sbv2.decoder.upsample_rates";
const KEY_DECODER_UPSAMPLE_KERNEL_SIZES: &str = "vokra.sbv2.decoder.upsample_kernel_sizes";
const KEY_DECODER_UPSAMPLE_OUT_CHANNELS: &str = "vokra.sbv2.decoder.upsample_out_channels";
const KEY_DECODER_RESBLOCK_KERNEL_SIZES: &str = "vokra.sbv2.decoder.resblock_kernel_sizes";
const KEY_DECODER_RESBLOCK_DILATION_COUNTS: &str = "vokra.sbv2.decoder.resblock_dilation_counts";
const KEY_DECODER_RESBLOCK_DILATIONS_FLAT: &str = "vokra.sbv2.decoder.resblock_dilations_flat";

// Optional (1).
const KEY_DECODER_LEAKY_RELU_SLOPE: &str = "vokra.sbv2.decoder.leaky_relu_slope";

/// Default `decoder.leaky_relu_slope` when the config side-car omits it —
/// mirrors `SbV2Model::from_gguf`'s own `unwrap_or(0.1)` fallback (the
/// universal jik876/hifi-gan `LRELU_SLOPE` every sibling decoder in this
/// codebase uses — `vits_ja::VITS_JA_LEAKY_RELU_SLOPE`, piper-plus's
/// `LRELU_SLOPE`).
const DEFAULT_LEAKY_RELU_SLOPE: f32 = 0.1;

/// `JsonValue` → `f64` accepting both int and float literals (a side-car
/// may write `0.1` as a float or a whole-number slope like `1` as an int).
/// Mirror of `models::utmos::json_f64`.
fn json_f64(v: &JsonValue) -> Option<f64> {
    match v {
        JsonValue::Int(i) => Some(*i as f64),
        JsonValue::Float(f) => Some(*f),
        _ => None,
    }
}

/// Emit a `u32` array under `key`. Follows the CSM / VibeVoice / VitsJa /
/// distil-whisper pattern (`add_metadata(GgufMetadataValue::Array(...))`)
/// — the builder does not carry a typed `add_*_array` shortcut.
fn add_u32_array(b: &mut GgufBuilder, key: &str, values: &[u32]) {
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().map(|&v| GgufMetadataValue::U32(v)).collect(),
        }),
    );
}

/// Parsed `vokra.sbv2.*` config side-car — every field mirrors one
/// `SbV2Model::from_gguf` metadata key 1:1 (see this crate's `sbv2` module
/// doc "Hparams" section for the field-by-field rationale, and that
/// method's own doc for the authoritative schema). Field names are the
/// JSON side-car's own key spelling: the metadata key's final segment,
/// flattened (e.g. `"d_model"` for `vokra.sbv2.d_model`,
/// `"decoder_upsample_rates"` for `vokra.sbv2.decoder.upsample_rates`) —
/// deliberately **not nested**, so [`Self::parse`] makes no assumption
/// about whether the shared `vokra_core::json` parser's lookup supports
/// dotted-path or nested-object traversal (it does not need to: `JsonValue
/// ::get` is a single-level object lookup — mirrors `models::dac::DacConfig`
/// / `models::utmos::UtmosConvertConfig`'s flat-key convention).
#[derive(Debug, Clone)]
pub(crate) struct SbV2Config {
    pub(crate) d_model: u32,
    pub(crate) d_bert: u32,
    pub(crate) d_speaker: u32,
    pub(crate) n_speakers: u32,
    pub(crate) d_style: u32,
    pub(crate) d_z: u32,
    pub(crate) n_vocab: u32,
    pub(crate) n_tones: u32,
    pub(crate) d_ff: u32,
    pub(crate) n_text_layers: u32,
    pub(crate) n_flow_layers: u32,
    pub(crate) n_sdp_layers: u32,
    pub(crate) sample_rate: u32,
    pub(crate) decoder_initial_channel: u32,
    pub(crate) decoder_conv_pre_kernel: u32,
    pub(crate) decoder_conv_post_kernel: u32,
    pub(crate) decoder_upsample_rates: Vec<u32>,
    pub(crate) decoder_upsample_kernel_sizes: Vec<u32>,
    pub(crate) decoder_upsample_out_channels: Vec<u32>,
    pub(crate) decoder_resblock_kernel_sizes: Vec<u32>,
    pub(crate) decoder_resblock_dilation_counts: Vec<u32>,
    pub(crate) decoder_resblock_dilations_flat: Vec<u32>,
    pub(crate) decoder_leaky_relu_slope: f32,
}

impl SbV2Config {
    /// Parses a JSON config side-car. Every field is required except
    /// `decoder_leaky_relu_slope` (defaults to [`DEFAULT_LEAKY_RELU_SLOPE`]
    /// when the key is absent, mirroring `SbV2Model::from_gguf`'s own
    /// fallback).
    ///
    /// # Errors
    ///
    /// [`ConvertError::Parse`] if the bytes are not valid JSON, if any
    /// required field is missing or not the expected type, or if the
    /// parsed config fails an internal-consistency check (`d_z` zero or
    /// odd; a decoder array length disagreeing with its sibling — see the
    /// module doc's "Hparams" section for the full list, all mirroring
    /// `SbV2Model::from_gguf`'s own load-time checks).
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
        let req_u32 = |key: &str| -> Result<u32, ConvertError> {
            root.get(key)
                .and_then(JsonValue::as_u64)
                .map(|u| u as u32)
                .ok_or_else(|| {
                    ConvertError::Parse(format!(
                        "sbv2 config: required non-negative integer field `{key}` missing or \
                         not a number (see SbV2Model::from_gguf's doc for the full \
                         vokra.sbv2.* schema)"
                    ))
                })
        };
        let req_u32_array = |key: &str| -> Result<Vec<u32>, ConvertError> {
            let val = root.get(key).ok_or_else(|| {
                ConvertError::Parse(format!("sbv2 config: required array field `{key}` missing"))
            })?;
            let arr = val.as_array().ok_or_else(|| {
                ConvertError::Parse(format!("sbv2 config: field `{key}` is not an array"))
            })?;
            arr.iter()
                .map(|v| {
                    v.as_u64().map(|u| u as u32).ok_or_else(|| {
                        ConvertError::Parse(format!(
                            "sbv2 config: an element of array field `{key}` is not a \
                             non-negative integer"
                        ))
                    })
                })
                .collect()
        };

        let cfg = Self {
            d_model: req_u32("d_model")?,
            d_bert: req_u32("d_bert")?,
            d_speaker: req_u32("d_speaker")?,
            n_speakers: req_u32("n_speakers")?,
            d_style: req_u32("d_style")?,
            d_z: req_u32("d_z")?,
            n_vocab: req_u32("n_vocab")?,
            n_tones: req_u32("n_tones")?,
            d_ff: req_u32("d_ff")?,
            n_text_layers: req_u32("n_text_layers")?,
            n_flow_layers: req_u32("n_flow_layers")?,
            n_sdp_layers: req_u32("n_sdp_layers")?,
            sample_rate: req_u32("sample_rate")?,
            decoder_initial_channel: req_u32("decoder_initial_channel")?,
            decoder_conv_pre_kernel: req_u32("decoder_conv_pre_kernel")?,
            decoder_conv_post_kernel: req_u32("decoder_conv_post_kernel")?,
            decoder_upsample_rates: req_u32_array("decoder_upsample_rates")?,
            decoder_upsample_kernel_sizes: req_u32_array("decoder_upsample_kernel_sizes")?,
            decoder_upsample_out_channels: req_u32_array("decoder_upsample_out_channels")?,
            decoder_resblock_kernel_sizes: req_u32_array("decoder_resblock_kernel_sizes")?,
            decoder_resblock_dilation_counts: req_u32_array("decoder_resblock_dilation_counts")?,
            decoder_resblock_dilations_flat: req_u32_array("decoder_resblock_dilations_flat")?,
            decoder_leaky_relu_slope: root
                .get("decoder_leaky_relu_slope")
                .and_then(json_f64)
                .map(|f| f as f32)
                .unwrap_or(DEFAULT_LEAKY_RELU_SLOPE),
        };

        // Internal-consistency checks — every one mirrors an explicit,
        // documented `SbV2Model::from_gguf` load-time check (module doc
        // "Hparams" section). Failing here gives a clearer, earlier error
        // than letting a malformed config reach the loader.
        if cfg.d_z == 0 || cfg.d_z % 2 != 0 {
            return Err(ConvertError::Parse(format!(
                "sbv2 config: `d_z` must be non-zero and even (VITS2 affine coupling splits \
                 the flow latent into two equal channel halves — see SbV2Model::from_gguf's \
                 doc), got {}",
                cfg.d_z
            )));
        }
        // n_text_layers / n_flow_layers / n_sdp_layers may legitimately be
        // 0 (an exercised empty-stack configuration — SbV2Model::from_gguf's
        // own doc); every other scalar dimension must be positive for the
        // architecture to describe a working model.
        if cfg.d_model == 0
            || cfg.d_bert == 0
            || cfg.d_speaker == 0
            || cfg.n_speakers == 0
            || cfg.d_style == 0
            || cfg.n_vocab == 0
            || cfg.n_tones == 0
            || cfg.d_ff == 0
            || cfg.sample_rate == 0
            || cfg.decoder_initial_channel == 0
            || cfg.decoder_conv_pre_kernel == 0
            || cfg.decoder_conv_post_kernel == 0
        {
            return Err(ConvertError::Parse(
                "sbv2 config: d_model / d_bert / d_speaker / n_speakers / d_style / n_vocab / \
                 n_tones / d_ff / sample_rate / decoder_initial_channel / \
                 decoder_conv_pre_kernel / decoder_conv_post_kernel must all be > 0 \
                 (n_text_layers / n_flow_layers / n_sdp_layers are the only dims where 0 is a \
                 legitimate empty-stack configuration)"
                    .to_owned(),
            ));
        }
        if cfg.decoder_upsample_kernel_sizes.len() != cfg.decoder_upsample_rates.len() {
            return Err(ConvertError::Parse(format!(
                "sbv2 config: decoder_upsample_kernel_sizes.len() ({}) != \
                 decoder_upsample_rates.len() ({})",
                cfg.decoder_upsample_kernel_sizes.len(),
                cfg.decoder_upsample_rates.len()
            )));
        }
        if cfg.decoder_upsample_out_channels.len() != cfg.decoder_upsample_rates.len() {
            return Err(ConvertError::Parse(format!(
                "sbv2 config: decoder_upsample_out_channels.len() ({}) != \
                 decoder_upsample_rates.len() ({})",
                cfg.decoder_upsample_out_channels.len(),
                cfg.decoder_upsample_rates.len()
            )));
        }
        if cfg.decoder_resblock_dilation_counts.len() != cfg.decoder_resblock_kernel_sizes.len() {
            return Err(ConvertError::Parse(format!(
                "sbv2 config: decoder_resblock_dilation_counts.len() ({}) != \
                 decoder_resblock_kernel_sizes.len() ({})",
                cfg.decoder_resblock_dilation_counts.len(),
                cfg.decoder_resblock_kernel_sizes.len()
            )));
        }
        let expected_flat: usize = cfg
            .decoder_resblock_dilation_counts
            .iter()
            .map(|&c| c as usize)
            .sum();
        if cfg.decoder_resblock_dilations_flat.len() != expected_flat {
            return Err(ConvertError::Parse(format!(
                "sbv2 config: decoder_resblock_dilations_flat.len() ({}) != \
                 sum(decoder_resblock_dilation_counts) ({expected_flat})",
                cfg.decoder_resblock_dilations_flat.len(),
            )));
        }

        Ok(cfg)
    }
}

/// Outcome of an SBV2 v2 conversion.
#[derive(Debug, Default)]
pub struct ConvertReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time today, so
    /// this arm is unreachable in practice; kept for parity with the
    /// sibling converters' counter shape).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16 →
    /// f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    pub bf16_passthrough: usize,
    /// Whether `config_side_car` was supplied to [`convert_sbv2_file`] and
    /// the `vokra.sbv2.*` hparam chunk (22 required + 1 optional keys) was
    /// written. `false` means tensors still passed through but
    /// `SbV2Model::from_gguf` will fail loudly on the first missing
    /// `vokra.sbv2.*` key — see this module's doc "Hparams" section for
    /// why that is preferred over inventing placeholder values.
    pub hparams_written: bool,
}

/// Converts an SBV2 v2 safetensors checkpoint at `input` into a Vokra GGUF
/// at `output`.
///
/// `config_side_car`, when `Some`, points at a JSON file supplying every
/// `vokra.sbv2.*` hparam (see [`SbV2Config::parse`] for the schema); when
/// `None`, tensors still pass through but the `vokra.sbv2.*` chunk is
/// omitted entirely rather than filled with invented placeholders (module
/// doc "Hparams" section). `license` overrides the upstream `agpl-3.0`
/// stamp (mirror of the `convert_file --license <spdx>` boundary in
/// `lib.rs`).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` / `config_side_car`
/// or writing `output`; [`ConvertError::Parse`] for malformed safetensors
/// input, or a malformed/incomplete config side-car (see
/// [`SbV2Config::parse`]'s doc for the full list of required fields and
/// consistency checks); [`ConvertError::Gguf`] if the GGUF serialization
/// fails.
pub fn convert_sbv2_file(
    input: &Path,
    output: &Path,
    config_side_car: Option<&Path>,
    license: Option<&str>,
) -> Result<ConvertReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);

    let mut report = ConvertReport::default();

    // Tensors — verbatim pass-through under the upstream safetensors name;
    // no renaming (module doc "TODO(owner)" section — Task 30 fixup).
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => report.skipped_non_float += 1,
        }
    }

    // Hparams — config-side-car-driven only, never invented (module doc
    // "Hparams" section).
    if let Some(config_path) = config_side_car {
        let config_bytes = std::fs::read(config_path)?;
        let cfg = SbV2Config::parse(&config_bytes)?;
        write_hparams(&mut b, &cfg);
        report.hparams_written = true;
    }

    let spdx = license.unwrap_or(DEFAULT_LICENSE);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_HF));

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;

    Ok(report)
}

/// Writes the 22 required + 1 optional `vokra.sbv2.*` keys from a parsed
/// [`SbV2Config`] — one `add_*` call per field, in the same order as
/// `SbV2Model::from_gguf`'s own read sequence.
///
/// Additionally stamps [`KEY_N_LANGUAGES`] = [`N_LANGUAGES`] — a fixed
/// architectural constant not carried on [`SbV2Config`] (see that const's
/// own doc and the module doc's "M6 refactor" section).
fn write_hparams(b: &mut GgufBuilder, cfg: &SbV2Config) {
    b.add_u32(KEY_D_MODEL, cfg.d_model);
    b.add_u32(KEY_D_BERT, cfg.d_bert);
    b.add_u32(KEY_D_SPEAKER, cfg.d_speaker);
    b.add_u32(KEY_N_SPEAKERS, cfg.n_speakers);
    b.add_u32(KEY_D_STYLE, cfg.d_style);
    b.add_u32(KEY_D_Z, cfg.d_z);
    b.add_u32(KEY_N_VOCAB, cfg.n_vocab);
    b.add_u32(KEY_N_TONES, cfg.n_tones);
    b.add_u32(KEY_D_FF, cfg.d_ff);
    b.add_u32(KEY_N_TEXT_LAYERS, cfg.n_text_layers);
    b.add_u32(KEY_N_FLOW_LAYERS, cfg.n_flow_layers);
    b.add_u32(KEY_N_SDP_LAYERS, cfg.n_sdp_layers);
    b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
    // Fixed-architecture: [`N_LANGUAGES`] = 3 (JA/EN/ZH); see its own doc
    // and the module doc's "M6 refactor" section.
    b.add_u32(KEY_N_LANGUAGES, N_LANGUAGES);

    b.add_u32(KEY_DECODER_INITIAL_CHANNEL, cfg.decoder_initial_channel);
    b.add_u32(KEY_DECODER_CONV_PRE_KERNEL, cfg.decoder_conv_pre_kernel);
    b.add_u32(KEY_DECODER_CONV_POST_KERNEL, cfg.decoder_conv_post_kernel);

    add_u32_array(b, KEY_DECODER_UPSAMPLE_RATES, &cfg.decoder_upsample_rates);
    add_u32_array(
        b,
        KEY_DECODER_UPSAMPLE_KERNEL_SIZES,
        &cfg.decoder_upsample_kernel_sizes,
    );
    add_u32_array(
        b,
        KEY_DECODER_UPSAMPLE_OUT_CHANNELS,
        &cfg.decoder_upsample_out_channels,
    );
    add_u32_array(
        b,
        KEY_DECODER_RESBLOCK_KERNEL_SIZES,
        &cfg.decoder_resblock_kernel_sizes,
    );
    add_u32_array(
        b,
        KEY_DECODER_RESBLOCK_DILATION_COUNTS,
        &cfg.decoder_resblock_dilation_counts,
    );
    add_u32_array(
        b,
        KEY_DECODER_RESBLOCK_DILATIONS_FLAT,
        &cfg.decoder_resblock_dilations_flat,
    );

    b.add_f32(KEY_DECODER_LEAKY_RELU_SLOPE, cfg.decoder_leaky_relu_slope);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn safetensors_multi(entries: &[(&str, &str, &[u64], Vec<u8>)]) -> Vec<u8> {
        let mut parts = Vec::new();
        let mut body = Vec::new();
        let mut cursor: usize = 0;
        for (name, dtype, shape, payload) in entries {
            let start = cursor;
            let end = cursor + payload.len();
            let shape_str = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            parts.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[{start},{end}]}}"#
            ));
            body.extend_from_slice(payload);
            cursor = end;
        }
        let header = format!("{{{}}}", parts.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Distinctive BF16 payload — top 16 bits of exact f32 values (mirrors
    /// `deberta_v2.rs`'s `bf16_bytes`).
    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn temp_path(label: &str, ext: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vokra-sbv2-{label}-{}.{ext}", std::process::id()));
        p
    }

    fn base_fixture() -> Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> {
        vec![
            ("enc_p.emb.weight", "F32", &[6, 4], f32_bytes(&[0.01; 24])),
            ("dec.ups.0.weight", "BF16", &[4, 4], bf16_bytes(&[0.04; 16])),
        ]
    }

    /// Minimal but complete config JSON covering every required field plus
    /// a legitimate value for `decoder_leaky_relu_slope`. Two upsample
    /// stages, one resblock branch with two dilations — small but internally
    /// consistent (exercises every array-length cross-check with non-trivial
    /// lengths).
    fn valid_config_json() -> Vec<u8> {
        br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 192, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8, 8],
            "decoder_upsample_kernel_sizes": [16, 16],
            "decoder_upsample_out_channels": [256, 128],
            "decoder_resblock_kernel_sizes": [3],
            "decoder_resblock_dilation_counts": [2],
            "decoder_resblock_dilations_flat": [1, 3],
            "decoder_leaky_relu_slope": 0.2
        }"#
        .to_vec()
    }

    // ---- convert_sbv2_file: no config side-car --------------------------

    #[test]
    fn tensors_pass_through_without_config_and_hparams_are_absent() {
        let blob = safetensors_multi(&base_fixture());
        let input = temp_path("no-config-in", "safetensors");
        let output = temp_path("no-config-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and BF16 both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);
        assert!(!report.hparams_written, "no config side-car was supplied");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        // Tensors round-trip verbatim under their upstream names.
        assert!(file.tensor_info("enc_p.emb.weight").is_some());
        let bf16_info = file
            .tensor_info("dec.ups.0.weight")
            .expect("BF16 tensor present under its verbatim upstream name");
        assert_eq!(
            bf16_info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16 (type 30)"
        );

        // vokra.sbv2.* is entirely absent — not filled with placeholder 0s.
        assert!(file.get(KEY_D_MODEL).is_none());
        assert!(file.get(KEY_D_Z).is_none());
        assert!(file.get(KEY_DECODER_UPSAMPLE_RATES).is_none());
        // The fixed-architecture `n_languages` stamp is also absent when
        // the hparam-writing path is not triggered — it belongs to the
        // same chunk group as everything else the loader reads, so all or
        // none applies. See the module doc's "Hparams" section for the
        // preference for missing-key-loud-fail over placeholder pollution.
        assert!(file.get(KEY_N_LANGUAGES).is_none());

        // arch / name / provenance still stamped.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Copyleft.as_str()),
            "agpl-3.0 must classify as Copyleft"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // ---- convert_sbv2_file: with config side-car -------------------------

    #[test]
    fn hparams_written_and_round_trip_with_config_side_car() {
        let blob = safetensors_multi(&base_fixture());
        let input = temp_path("with-config-in", "safetensors");
        let config = temp_path("with-config-cfg", "json");
        let output = temp_path("with-config-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        std::fs::write(&config, valid_config_json()).expect("write config");

        let report = convert_sbv2_file(&input, &output, Some(&config), None).expect("convert");
        assert!(report.hparams_written);

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        let get_u32 = |key: &str| -> u64 {
            file.get(key)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("{key}: missing or not u32"))
        };
        assert_eq!(get_u32(KEY_D_MODEL), 192);
        assert_eq!(get_u32(KEY_D_BERT), 1024);
        assert_eq!(get_u32(KEY_D_SPEAKER), 512);
        assert_eq!(get_u32(KEY_N_SPEAKERS), 3);
        assert_eq!(get_u32(KEY_D_STYLE), 256);
        assert_eq!(get_u32(KEY_D_Z), 192);
        assert_eq!(get_u32(KEY_N_VOCAB), 178);
        assert_eq!(get_u32(KEY_N_TONES), 3);
        assert_eq!(get_u32(KEY_D_FF), 768);
        assert_eq!(get_u32(KEY_N_TEXT_LAYERS), 6);
        assert_eq!(get_u32(KEY_N_FLOW_LAYERS), 4);
        assert_eq!(get_u32(KEY_N_SDP_LAYERS), 4);
        assert_eq!(get_u32(KEY_SAMPLE_RATE), 44_100);
        assert_eq!(get_u32(KEY_DECODER_INITIAL_CHANNEL), 512);
        assert_eq!(get_u32(KEY_DECODER_CONV_PRE_KERNEL), 7);
        assert_eq!(get_u32(KEY_DECODER_CONV_POST_KERNEL), 7);
        // M6 refactor (2026-08-06): `n_languages` is a fixed
        // architectural constant (JA/EN/ZH = 3), not a config field, and
        // is always stamped as long as the config side-car triggers the
        // hparam-writing path. See the module doc's "M6 refactor" section
        // for the primary-source verification.
        assert_eq!(get_u32(KEY_N_LANGUAGES), u64::from(N_LANGUAGES));

        let get_u32_array = |key: &str| -> Vec<u32> {
            file.get(key)
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("{key}: not an array"))
                .values
                .iter()
                .map(|v| match v {
                    GgufMetadataValue::U32(x) => *x,
                    other => panic!("{key}: array elem not u32 ({other:?})"),
                })
                .collect()
        };
        assert_eq!(get_u32_array(KEY_DECODER_UPSAMPLE_RATES), vec![8, 8]);
        assert_eq!(
            get_u32_array(KEY_DECODER_UPSAMPLE_KERNEL_SIZES),
            vec![16, 16]
        );
        assert_eq!(
            get_u32_array(KEY_DECODER_UPSAMPLE_OUT_CHANNELS),
            vec![256, 128]
        );
        assert_eq!(get_u32_array(KEY_DECODER_RESBLOCK_KERNEL_SIZES), vec![3]);
        assert_eq!(get_u32_array(KEY_DECODER_RESBLOCK_DILATION_COUNTS), vec![2]);
        assert_eq!(
            get_u32_array(KEY_DECODER_RESBLOCK_DILATIONS_FLAT),
            vec![1, 3]
        );

        let slope = match file.get(KEY_DECODER_LEAKY_RELU_SLOPE) {
            Some(GgufMetadataValue::F32(f)) => *f,
            other => panic!("leaky_relu_slope: unexpected {other:?}"),
        };
        assert!((slope - 0.2).abs() < 1e-6);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&config).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn leaky_relu_slope_defaults_when_omitted() {
        let blob = safetensors_multi(&base_fixture());
        let input = temp_path("default-slope-in", "safetensors");
        let output = temp_path("default-slope-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        // Same as valid_config_json but without decoder_leaky_relu_slope.
        let config_bytes = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 192, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8, 8],
            "decoder_upsample_kernel_sizes": [16, 16],
            "decoder_upsample_out_channels": [256, 128],
            "decoder_resblock_kernel_sizes": [3],
            "decoder_resblock_dilation_counts": [2],
            "decoder_resblock_dilations_flat": [1, 3]
        }"#;
        let config = temp_path("default-slope-cfg", "json");
        std::fs::write(&config, config_bytes).expect("write config");

        convert_sbv2_file(&input, &output, Some(&config), None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        let slope = match file.get(KEY_DECODER_LEAKY_RELU_SLOPE) {
            Some(GgufMetadataValue::F32(f)) => *f,
            other => panic!("leaky_relu_slope: unexpected {other:?}"),
        };
        assert!((slope - DEFAULT_LEAKY_RELU_SLOPE).abs() < 1e-6);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&config).ok();
        std::fs::remove_file(&output).ok();
    }

    // ---- SbV2Config::parse error paths -----------------------------------

    #[test]
    fn missing_required_field_is_a_loud_error() {
        let bytes = br#"{"d_model": 192}"#;
        let err = SbV2Config::parse(bytes).expect_err("must fail loudly");
        let msg = err.to_string();
        assert!(matches!(err, ConvertError::Parse(_)));
        assert!(msg.contains("d_bert"), "message must name the field: {msg}");
    }

    #[test]
    fn d_z_zero_is_rejected() {
        let bytes = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 0, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8], "decoder_upsample_kernel_sizes": [16],
            "decoder_upsample_out_channels": [256],
            "decoder_resblock_kernel_sizes": [3], "decoder_resblock_dilation_counts": [1],
            "decoder_resblock_dilations_flat": [1]
        }"#;
        let err = SbV2Config::parse(bytes).expect_err("d_z=0 must be rejected");
        assert!(err.to_string().contains("d_z"));
    }

    #[test]
    fn d_z_odd_is_rejected() {
        let bytes = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 191, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8], "decoder_upsample_kernel_sizes": [16],
            "decoder_upsample_out_channels": [256],
            "decoder_resblock_kernel_sizes": [3], "decoder_resblock_dilation_counts": [1],
            "decoder_resblock_dilations_flat": [1]
        }"#;
        let err = SbV2Config::parse(bytes).expect_err("odd d_z must be rejected");
        assert!(err.to_string().contains("d_z"));
    }

    #[test]
    fn zero_stack_depths_are_accepted_empty_stack_is_legitimate() {
        let bytes = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 192, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 0, "n_flow_layers": 0, "n_sdp_layers": 0,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8], "decoder_upsample_kernel_sizes": [16],
            "decoder_upsample_out_channels": [256],
            "decoder_resblock_kernel_sizes": [3], "decoder_resblock_dilation_counts": [1],
            "decoder_resblock_dilations_flat": [1]
        }"#;
        let cfg = SbV2Config::parse(bytes).expect("0 stack depths must be accepted");
        assert_eq!(cfg.n_text_layers, 0);
        assert_eq!(cfg.n_flow_layers, 0);
        assert_eq!(cfg.n_sdp_layers, 0);
    }

    #[test]
    fn mismatched_upsample_array_lengths_are_rejected() {
        let bytes = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 192, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8, 8],
            "decoder_upsample_kernel_sizes": [16],
            "decoder_upsample_out_channels": [256, 128],
            "decoder_resblock_kernel_sizes": [3], "decoder_resblock_dilation_counts": [1],
            "decoder_resblock_dilations_flat": [1]
        }"#;
        let err = SbV2Config::parse(bytes).expect_err("length mismatch must be rejected");
        assert!(err.to_string().contains("decoder_upsample_kernel_sizes"));
    }

    #[test]
    fn mismatched_resblock_dilations_flat_length_is_rejected() {
        let bytes = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 192, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8], "decoder_upsample_kernel_sizes": [16],
            "decoder_upsample_out_channels": [256],
            "decoder_resblock_kernel_sizes": [3, 7],
            "decoder_resblock_dilation_counts": [2, 1],
            "decoder_resblock_dilations_flat": [1, 3]
        }"#;
        let err = SbV2Config::parse(bytes).expect_err("flat-length mismatch must be rejected");
        assert!(err.to_string().contains("resblock_dilations_flat"));
    }

    #[test]
    fn missing_config_side_car_file_is_an_io_error() {
        let blob = safetensors_multi(&base_fixture());
        let input = temp_path("missing-config-in", "safetensors");
        let output = temp_path("missing-config-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        let missing_config = temp_path("does-not-exist", "json");

        let err = convert_sbv2_file(&input, &output, Some(&missing_config), None)
            .expect_err("missing config file must error");
        assert!(matches!(err, ConvertError::Io(_)));
        assert!(!output.exists(), "no partial GGUF must be left behind");

        std::fs::remove_file(&input).ok();
    }

    // ---- license override -------------------------------------------------

    #[test]
    fn license_override_replaces_default() {
        let blob = safetensors_multi(&base_fixture());
        let input = temp_path("license-override-in", "safetensors");
        let output = temp_path("license-override-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        convert_sbv2_file(&input, &output, None, Some("apache-2.0")).expect("convert");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // ---- misc identity checks ----------------------------------------------

    #[test]
    fn arch_tag_is_distinct_from_neighboring_tts_archs() {
        assert_eq!(ARCH, "sbv2");
        assert_ne!(ARCH, "piper-plus-mb-istft-vits2");
        assert_ne!(ARCH, "vits-ja");
        assert_ne!(ARCH, "deberta_v2");
        assert_ne!(ARCH, "deberta_v3");
    }
}
