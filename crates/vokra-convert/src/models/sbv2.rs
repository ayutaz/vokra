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
//! # Task 30 rename table (2026-08-06)
//!
//! Every upstream tensor is now classified by [`classify_tensor`] into one
//! of four buckets before being written to the output GGUF:
//!
//! 1. **Rename** — the upstream name maps 1:1 to a `sbv2.*` name that
//!    `SbV2Model::from_gguf` reads (see that method's own doc for the full
//!    target hierarchy). Bytes are copied verbatim under the new name.
//!    Applies to: text encoder embeddings (`enc_p.emb` / `enc_p.tone_emb` /
//!    `enc_p.language_emb`), BERT projection (`enc_p.bert_proj.*`), decoder
//!    conv_pre / conv_post, decoder upsample bias, and the reconstructed
//!    decoder MRF `bias` tensors.
//! 2. **`WeightNorm`** — a `.weight_g` sibling for a `.weight_v` is paired
//!    at scan time; the reconstructed weight `W[i, ...] = weight_g[i] *
//!    (weight_v[i, ...] / ||weight_v[i, ...]||_2)` is emitted under the
//!    renamed target. Applies to: `dec.ups.{i}.weight_g/v` and
//!    `dec.resblocks.{flat_i}.convs1.{j}.weight_g/v`. The `weight_v` on
//!    disk is `PyTorch nn.utils.weight_norm(dim=0)`-split — see the
//!    `restore_weight_norm_f32` docstring for the formula.
//! 3. **`PassThrough`** — bytes emitted **verbatim under the upstream
//!    name**, no rename. Data is preserved for a future architecture wave
//!    that will teach `SbV2Model::from_gguf` (or a companion loader) to
//!    consume it. Applies to: `flow.*` (VITS2 TransformerCouplingBlock —
//!    "Blocker 2"), `enc_p.encoder.*` (SBV2 relative-position transformer
//!    stack that does not match the simplified Rust `SbV2TextEncoder`),
//!    `sdp.pre / .proj / .convs / .flows / .cond` (production SDP path
//!    using DDS-net + rational-quadratic-spline `ConvFlow` — the Rust
//!    `SbV2SDP` is a scalar-affine simplification), `dec.cond.*` (decoder
//!    speaker conditioning), `enc_p.proj.*` (VITS output projection to
//!    `(mu, log_sigma)`), `enc_p.encoder.spk_emb_linear.*` ("Blocker 3"
//!    speaker-conditioning projection), and `dec.resblocks.*.convs2.*`
//!    (undilated residual convs the Rust simplified HiFi-GAN does not
//!    consume). See [`classify_tensor`] for the per-family reason strings.
//! 4. **`Skip`** — tensor is dropped with a stderr log line. Applies to
//!    training-side tensors that `SbV2Model::from_gguf` never loads:
//!    `enc_q.*` (posterior encoder), `dp.*` (deterministic duration
//!    predictor, VITS1-legacy), and `sdp.post_*` (SDP training-side
//!    inverse path — post_pre, post_proj, post_convs, post_flows). The log
//!    line satisfies FR-EX-08's "no silent drop" rule (`eprintln!` to
//!    stderr, one line per skipped tensor).
//!
//! # What Rust `SbV2Model::from_gguf` will still fail loudly on
//!
//! Even after this rename table lands, the loader will fail — as designed
//! — on tensor names it needs that this converter cannot yet produce:
//!
//! - `sbv2.text_encoder.layer.{i}.attn.{q,k,v,o}.weight` etc. — the Rust
//!   text encoder is a simplified prenorm transformer, not the SBV2
//!   relative-position `MultiHeadAttention` + Conv1D-FFN stack; the
//!   upstream `enc_p.encoder.*` tensors are preserved verbatim for a
//!   future architecture wave.
//! - `sbv2.flow.layer.{i}.{scale,shift,style_proj,speaker_proj}` — the
//!   Rust flow is a simplified WaveNet-residual affine coupling, not the
//!   VITS2 `TransformerCouplingBlock` upstream ships. See "Blocker 2".
//! - `sbv2.speaker.table` — the base checkpoint has no per-speaker
//!   embedding table; it uses an external `[512]` speaker vector projected
//!   through `enc_p.encoder.spk_emb_linear` (preserved verbatim). See
//!   "Blocker 3".
//! - `sbv2.sdp.flow_layer.{i}.{proj_weight,proj_bias}` and
//!   `sbv2.sdp.{tone_embed,tone_bias}` — the Rust SDP is a scalar-affine
//!   simplification of the DDS-net + ConvFlow production SDP. Upstream
//!   `sdp.*` tensors are preserved verbatim.
//! - `sbv2.decoder.conv_post.bias` — upstream has `dec.conv_post.weight`
//!   with no bias; the Rust loader unconditionally reads `conv_post.bias`.
//!   This is a Rust-side latent gap, not something the converter can fill.
//!
//! Each of these is a signalled "loud fail" the design accepts as the
//! next-wave gate. This converter's Task 30 job is to make every tensor
//! the loader **can** consume land under its target name, and to preserve
//! every other tensor's bytes so no data is silently discarded.
//!
//! # Wave audit trail (upstream base checkpoint)
//!
//! For `litagin/Style-Bert-VITS2-2.0-base-JP-Extra` `G_0.safetensors`
//! (1264 tensors, all F32), the classification breakdown is (verified
//! end-to-end on 2026-08-06 against the real checkpoint):
//!
//! - Rename emits: 58 (`enc_p.emb` / `enc_p.tone_emb` /
//!   `enc_p.language_emb` = 3, `enc_p.bert_proj.{weight,bias}` = 2,
//!   `dec.conv_pre.{weight,bias}` = 2, `dec.conv_post.weight` = 1,
//!   `dec.ups.{0..4}.bias` = 5, `dec.resblocks.{0..14}.convs1.{0..2}.bias`
//!   = 45)
//! - WeightNorm reconstructions: 50 (`dec.ups.{0..4}` = 5 +
//!   `dec.resblocks.{0..14}.convs1.{0..2}` = 45)
//! - Verbatim pass-through: 849 (`flow.*` = 456 +
//!   `enc_p.encoder.*` = 110 + `sdp.*` production (not `post_`) = 144 +
//!   `dec.resblocks.*.convs2.*` = 135 + `enc_p.proj.*` = 2 +
//!   `dec.cond.*` = 2)
//! - **Total written to output GGUF: 957** = 58 + 50 + 849
//! - Skipped training-side: 257 (`enc_q.*` = 103 + `dp.*` = 12 +
//!   `sdp.post_*` = 142)
//! - `weight_v` consumed as pair sibling (read but not written): 50
//! - **Input-side balance**: 1264 = 957 (written) + 257 (skipped) + 50
//!   (weight_v consumed) ✓
//!
//! The [`ConvertReport`] counters expose the same breakdown per-conversion
//! so callers (and tests) can spot-check this partition.
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

use std::collections::HashSet;
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
///
/// The `read` counter is the ground-truth input-side count (every
/// safetensors entry the reader saw). The remaining counters partition
/// it exactly under **two** invariants:
///
/// - **Input-side partition**: `written + skipped_non_float +
///   skipped_training + weight_norm_v_consumed == read`. Every input
///   tensor lands in exactly one bucket: written to output, refused
///   non-float, dropped training-side, or consumed as the `weight_v`
///   sibling of a `weight_g` (folded into a single reconstructed weight
///   in the output).
/// - **Output-side partition**: `renamed + weight_norm_reconstructed +
///   passed_through_verbatim == written`. Every written tensor arrived
///   via exactly one of the three "how it got emitted" paths.
///
/// See the module doc's "Wave audit trail" section for the
/// base-checkpoint numbers these partitions produce.
#[derive(Debug, Default)]
pub struct ConvertReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written to the output GGUF (renamed + weight_norm-
    /// reconstructed + pass-through-verbatim).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time today, so
    /// this arm is unreachable in practice; kept for parity with the
    /// sibling converters' counter shape).
    pub skipped_non_float: usize,
    /// Training-side tensors intentionally dropped: `enc_q.*` (posterior
    /// encoder), `dp.*` (deterministic duration predictor), `sdp.post_*`
    /// (SDP training-side inverse path). Each drop emits a stderr log
    /// line satisfying FR-EX-08's "no silent drop" rule. See the module
    /// doc's "Task 30 rename table" section for the classification list.
    pub skipped_training: usize,
    /// Input-side counter for `weight_v` tensors folded into a
    /// [`Self::weight_norm_reconstructed`] emit alongside their
    /// `weight_g` sibling. Each pair contributes `read += 2`,
    /// `weight_norm_v_consumed += 1`, `written += 1`,
    /// `weight_norm_reconstructed += 1` — the input-side partition (see
    /// the struct doc) accounts for the "read but not emitted"
    /// `weight_v` here. Equal to [`Self::weight_norm_reconstructed`] in
    /// normal (non-orphan) checkpoints.
    pub weight_norm_v_consumed: usize,
    /// Of the tensors in [`Self::written`], how many landed via a rename
    /// (fixed 1:1 rewrite, no data transform). Subset counter.
    pub renamed: usize,
    /// Of the tensors in [`Self::written`], how many landed via
    /// `weight_norm` reconstruction (a `weight_g` + `weight_v` pair
    /// consumed to produce one output weight). Subset counter.
    pub weight_norm_reconstructed: usize,
    /// Of the tensors in [`Self::written`], how many passed through
    /// verbatim under their upstream name (no rename, data preserved for
    /// a future Rust wave). Subset counter.
    pub passed_through_verbatim: usize,
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

    // Parse config side-car early so the resblock stage/branch math has
    // `n_branches` available during tensor classification. (The hparam
    // chunk is still written at the very end of the tensor loop, so the
    // final metadata layout is unchanged from before Task 30.)
    let cfg = if let Some(config_path) = config_side_car {
        let config_bytes = std::fs::read(config_path)?;
        Some(SbV2Config::parse(&config_bytes)?)
    } else {
        None
    };
    let n_resblock_branches = cfg.as_ref().map(|c| c.decoder_resblock_kernel_sizes.len());

    // Pre-scan: tensors that will be consumed as a weight_v sibling of a
    // weight_g pair. We must skip them when the main loop reaches them
    // (otherwise they would be emitted twice or as an orphan pass-through).
    let mut consumed_weight_v: HashSet<String> = HashSet::new();
    for t in st.tensors() {
        if let TensorClass::WeightNorm { sibling_v, .. } =
            classify_tensor(&t.name, n_resblock_branches)
        {
            if st.tensor_info(&sibling_v).is_some() {
                consumed_weight_v.insert(sibling_v);
            }
        }
    }

    // Main tensor loop — Task 30 rename table.
    for t in st.tensors() {
        report.read += 1;

        // Non-float safetensors dtypes are refused at parse time today (the
        // vokra-core reader only accepts F32/F16/BF16), so this arm is
        // defensive parity with the sibling converters' counter shape.
        if !matches!(t.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            report.skipped_non_float += 1;
            continue;
        }

        // A weight_v whose weight_g pair was already scheduled: consumed —
        // its bytes will be folded into the reconstructed weight below when
        // we visit the paired weight_g. Do not emit it a second time.
        if consumed_weight_v.contains(&t.name) {
            report.weight_norm_v_consumed += 1;
            continue;
        }

        match classify_tensor(&t.name, n_resblock_branches) {
            TensorClass::Skip { reason } => {
                // FR-EX-08: never silently drop — one stderr line per drop.
                eprintln!(
                    "convert_sbv2: skipping training-side tensor `{}` ({reason})",
                    t.name
                );
                report.skipped_training += 1;
            }
            TensorClass::Rename(target) => {
                let data = st.tensor_bytes(t).to_vec();
                let is_bf16 = t.dtype == GgmlType::BF16;
                b.add_tensor(&target, t.dtype, t.shape.clone(), data)?;
                report.written += 1;
                report.renamed += 1;
                if is_bf16 {
                    report.bf16_passthrough += 1;
                }
            }
            TensorClass::PassThrough { .. } => {
                let data = st.tensor_bytes(t).to_vec();
                let is_bf16 = t.dtype == GgmlType::BF16;
                b.add_tensor(&t.name, t.dtype, t.shape.clone(), data)?;
                report.written += 1;
                report.passed_through_verbatim += 1;
                if is_bf16 {
                    report.bf16_passthrough += 1;
                }
            }
            TensorClass::WeightNorm {
                sibling_v,
                target_name,
            } => {
                // Look up the sibling weight_v; if missing (orphan
                // weight_g), fall back to pass-through under the upstream
                // name (data preserved for inspection). Missing-sibling is
                // rare on real checkpoints but possible on hand-authored
                // fixtures — mirror what our own tests exercise.
                let Some(v_info) = st.tensor_info(&sibling_v) else {
                    let data = st.tensor_bytes(t).to_vec();
                    let is_bf16 = t.dtype == GgmlType::BF16;
                    b.add_tensor(&t.name, t.dtype, t.shape.clone(), data)?;
                    report.written += 1;
                    report.passed_through_verbatim += 1;
                    if is_bf16 {
                        report.bf16_passthrough += 1;
                    }
                    continue;
                };

                // Both siblings must be F32 for the reconstruction
                // arithmetic below (all base-checkpoint weight_norm pairs
                // are F32 in practice; defensively refuse otherwise so a
                // future BF16 checkpoint surfaces as a clean error rather
                // than a silent numerical narrowing).
                if t.dtype != GgmlType::F32 || v_info.dtype != GgmlType::F32 {
                    return Err(ConvertError::Parse(format!(
                        "sbv2 Task 30: weight_norm reconstruction supports F32 only \
                         today, got weight_g={:?} + weight_v={:?} for `{}` — file a \
                         Task 30 follow-up if a real checkpoint ships this pair non-F32",
                        t.dtype, v_info.dtype, t.name
                    )));
                }

                let weight_g = st.tensor_f32(&t.name).map_err(|e| {
                    ConvertError::Parse(format!(
                        "sbv2 Task 30: reading weight_g `{}` failed: {e}",
                        t.name
                    ))
                })?;
                let weight_v = st.tensor_f32(&sibling_v).map_err(|e| {
                    ConvertError::Parse(format!(
                        "sbv2 Task 30: reading weight_v `{sibling_v}` failed: {e}"
                    ))
                })?;

                let restored_bytes = restore_weight_norm_f32(&weight_g, &weight_v, &v_info.shape)
                    .map_err(|msg| {
                    ConvertError::Parse(format!(
                        "sbv2 Task 30: weight_norm reconstruction for `{}` + `{sibling_v}` \
                                 -> `{target_name}` failed: {msg}",
                        t.name
                    ))
                })?;

                b.add_tensor(
                    &target_name,
                    GgmlType::F32,
                    v_info.shape.clone(),
                    restored_bytes,
                )?;
                report.written += 1;
                report.weight_norm_reconstructed += 1;
            }
        }
    }

    // Hparams — config-side-car-driven only, never invented (module doc
    // "Hparams" section).
    if let Some(cfg) = cfg.as_ref() {
        write_hparams(&mut b, cfg);
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

// =====================================================================
// Task 30: upstream → GGUF tensor rename table + weight_norm restoration
// =====================================================================

/// Classification of one upstream safetensors tensor, produced by
/// [`classify_tensor`] and consumed by [`convert_sbv2_file`]'s main loop.
///
/// See the module doc's "Task 30 rename table" section for the four
/// buckets and the reasoning behind them.
#[derive(Debug, PartialEq, Eq)]
enum TensorClass {
    /// Rename this tensor's name to the given GGUF path; bytes are copied
    /// verbatim (no data transform).
    Rename(String),
    /// This tensor is a `.weight_g` whose paired `.weight_v` (`sibling_v`)
    /// exists in the same safetensors file: the caller reconstructs
    /// `weight = weight_g * (weight_v / ||weight_v||)` and emits it under
    /// `target_name`, then marks `sibling_v` as consumed so the main loop
    /// does not emit it a second time. If the sibling is missing (orphan
    /// `weight_g` — rare on real checkpoints, possible on hand-authored
    /// fixtures), the caller falls back to pass-through under the upstream
    /// name.
    WeightNorm {
        sibling_v: String,
        target_name: String,
    },
    /// Pass through under the upstream name; no rename, bytes copied
    /// verbatim. Data is preserved for a future architecture wave.
    /// `reason` is included for observability / future debug logging;
    /// nothing else consumes it today.
    PassThrough {
        #[allow(dead_code)]
        reason: &'static str,
    },
    /// Skip this tensor entirely. `reason` is emitted to stderr by the
    /// main loop (FR-EX-08 "no silent drop").
    Skip { reason: &'static str },
}

/// Classifies one upstream tensor into a [`TensorClass`].
///
/// `n_resblock_branches`, when `Some`, unlocks the flat `dec.resblocks.i`
/// → `mrf.{stage}.{branch}` remapping (`stage = i / n_branches`,
/// `branch = i % n_branches`). Without the config side-car we cannot
/// safely split the flat index (a wrong `n_branches` would silently
/// mis-shuffle every ResBlock), so those tensors fall back to
/// [`TensorClass::PassThrough`] to keep the data available for a later
/// wave.
///
/// # Ordering (important)
///
/// Training-side prefixes are checked **before** the family-specific
/// arms so that e.g. `sdp.post_flows.*` — which starts with `sdp.` — is
/// correctly skipped rather than being caught by the production `sdp.*`
/// pass-through arm.
fn classify_tensor(name: &str, n_resblock_branches: Option<usize>) -> TensorClass {
    // --------------------------------------------------------------
    // 1) Training-side skips (checked first; most specific prefixes)
    // --------------------------------------------------------------
    if name.starts_with("enc_q.") {
        return TensorClass::Skip {
            reason: "training-side posterior encoder",
        };
    }
    if name.starts_with("dp.") {
        return TensorClass::Skip {
            reason: "training-side deterministic duration predictor \
                     (VITS1 legacy, unused by SDP)",
        };
    }
    if name.starts_with("sdp.post_") {
        return TensorClass::Skip {
            reason: "training-side SDP inverse (post_pre / post_proj / \
                     post_convs / post_flows)",
        };
    }

    // --------------------------------------------------------------
    // 2) Fixed 1:1 renames
    // --------------------------------------------------------------
    match name {
        "enc_p.emb.weight" => {
            return TensorClass::Rename("sbv2.text_encoder.phoneme_embed".into());
        }
        "enc_p.tone_emb.weight" => {
            return TensorClass::Rename("sbv2.text_encoder.tone_embed".into());
        }
        "enc_p.language_emb.weight" => {
            return TensorClass::Rename("sbv2.text_encoder.language_embed".into());
        }
        "enc_p.bert_proj.weight" => {
            return TensorClass::Rename("sbv2.bert_bridge.conv.weight".into());
        }
        "enc_p.bert_proj.bias" => {
            return TensorClass::Rename("sbv2.bert_bridge.conv.bias".into());
        }
        "dec.conv_pre.weight" => {
            return TensorClass::Rename("sbv2.decoder.conv_pre.weight".into());
        }
        "dec.conv_pre.bias" => {
            return TensorClass::Rename("sbv2.decoder.conv_pre.bias".into());
        }
        "dec.conv_post.weight" => {
            return TensorClass::Rename("sbv2.decoder.conv_post.weight".into());
        }
        _ => {}
    }

    // --------------------------------------------------------------
    // 3) dec.ups.{i}.{weight_g | weight_v | bias | weight}
    // --------------------------------------------------------------
    if let Some(rest) = name.strip_prefix("dec.ups.") {
        if let Some((idx_str, tail)) = rest.split_once('.') {
            if let Ok(i) = idx_str.parse::<usize>() {
                match tail {
                    "weight_g" => {
                        return TensorClass::WeightNorm {
                            sibling_v: format!("dec.ups.{i}.weight_v"),
                            target_name: format!("sbv2.decoder.upsample.{i}.weight"),
                        };
                    }
                    "weight_v" => {
                        // Sibling of a weight_g that the main loop has
                        // already scheduled — the pre-scan added this
                        // name to `consumed_weight_v` and the main loop
                        // will `continue` before it lands here. This arm
                        // is reached only if a `weight_v` shows up
                        // without its `weight_g` sibling — treat as
                        // orphan and pass through so no data is lost.
                        return TensorClass::PassThrough {
                            reason: "orphan dec.ups weight_v (weight_g missing)",
                        };
                    }
                    "bias" => {
                        return TensorClass::Rename(format!("sbv2.decoder.upsample.{i}.bias"));
                    }
                    "weight" => {
                        // Fixture / non-weight_norm variant — some
                        // hand-authored test blobs use a bare `.weight`
                        // instead of the `.weight_g/.weight_v` split.
                        // Rename directly so the fixture round-trips
                        // under the target name.
                        return TensorClass::Rename(format!("sbv2.decoder.upsample.{i}.weight"));
                    }
                    _ => {}
                }
            }
        }
    }

    // --------------------------------------------------------------
    // 4) dec.resblocks.{flat_i}.{convs1|convs2}.{j}.{tail}
    // --------------------------------------------------------------
    if let Some(rest) = name.strip_prefix("dec.resblocks.") {
        let parts: Vec<&str> = rest.splitn(4, '.').collect();
        if parts.len() == 4 {
            if let (Ok(flat_i), Ok(j)) = (parts[0].parse::<usize>(), parts[2].parse::<usize>()) {
                let convs = parts[1];
                let tail = parts[3];
                if convs == "convs2" {
                    // Rust's simplified HiFi-GAN keeps only one Conv1d
                    // per dilation (the dilated one, `convs1`). The
                    // undilated residual convs — `convs2` in upstream —
                    // have no landing spot yet. Preserve verbatim so a
                    // later architecture wave can flip them on.
                    return TensorClass::PassThrough {
                        reason: "convs2 (undilated residual conv) — Rust simplified HiFi-GAN \
                                 has no consumer yet",
                    };
                }
                if convs == "convs1" {
                    let n_branches = match n_resblock_branches {
                        Some(n) if n > 0 => n,
                        _ => {
                            // No config side-car (or a degenerate
                            // n_branches=0) → we cannot honestly split
                            // the flat index. Preserve verbatim.
                            return TensorClass::PassThrough {
                                reason: "resblock stage/branch split requires config side-car's \
                                         decoder_resblock_kernel_sizes.len() (n_branches)",
                            };
                        }
                    };
                    let stage = flat_i / n_branches;
                    let branch = flat_i % n_branches;
                    let target_base = format!("sbv2.decoder.mrf.{stage}.{branch}.layer.{j}");
                    match tail {
                        "weight_g" => {
                            return TensorClass::WeightNorm {
                                sibling_v: format!("dec.resblocks.{flat_i}.convs1.{j}.weight_v"),
                                target_name: format!("{target_base}.weight"),
                            };
                        }
                        "weight_v" => {
                            return TensorClass::PassThrough {
                                reason: "orphan dec.resblocks convs1 weight_v (weight_g missing)",
                            };
                        }
                        "bias" => {
                            return TensorClass::Rename(format!("{target_base}.bias"));
                        }
                        "weight" => {
                            return TensorClass::Rename(format!("{target_base}.weight"));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------
    // 5) Verbatim pass-through with per-family reason (default)
    // --------------------------------------------------------------
    let reason: &'static str = if name.starts_with("flow.") {
        "flow VITS2 TransformerCouplingBlock — Blocker 2 wave"
    } else if name.starts_with("enc_p.encoder.spk_emb_linear") {
        "external speaker-vector projection — Blocker 3 (Rust API needs 512-d input path)"
    } else if name.starts_with("enc_p.encoder.") {
        "SBV2 relative-position transformer stack — Rust text_encoder simplified, awaiting \
         architecture wave"
    } else if name.starts_with("enc_p.proj.") {
        "VITS output projection to (mu, log_sigma) — no Rust text_encoder field yet"
    } else if name.starts_with("sdp.") {
        "production SBV2 SDP path (DDS-net + rational-quadratic-spline ConvFlow) — Rust \
         duration.rs simplified"
    } else if name.starts_with("dec.cond.") {
        "decoder speaker conditioning — no Rust HifiGanAttrs field yet"
    } else {
        "unrecognized upstream tensor — passed through verbatim so no data is silently discarded"
    };
    TensorClass::PassThrough { reason }
}

/// Reconstructs a PyTorch `nn.utils.weight_norm(dim=0)`-split weight
/// tensor back into a single dense weight, returning its little-endian
/// f32 byte buffer suitable for [`GgufBuilder::add_tensor`].
///
/// # Formula
///
/// PyTorch's `weight_norm(dim=0)` decomposes `W` (shape
/// `[rows, ...trailing]`) into two learned tensors:
///
/// - `weight_g`, shape `[rows, 1, ...]` (one scalar per row of `W`),
/// - `weight_v`, same shape as `W` (the direction),
///
/// with
///
/// ```text
/// W[i, ...] = weight_g[i] * (weight_v[i, ...] / ||weight_v[i, ...]||_2)
/// ```
///
/// where the L2 norm is taken across the trailing dims. This mirrors the
/// PyTorch reference formula (`torch.nn.utils.weight_norm` docstring,
/// v2.x — no code reference, formula only).
///
/// # Zero-norm rows
///
/// If a row of `weight_v` has zero L2 norm, PyTorch's forward returns
/// zero for that row (no NaN); this function does the same, gated by a
/// literal `> 0.0` check on the squared norm.
///
/// # Errors
///
/// Returns `Err(String)` (surface stringly-typed to avoid pulling
/// `ConvertError` into a pure numerics helper) if:
/// - `v_shape` is empty (a 0-dim `weight_v` cannot be `dim=0`-normed);
/// - `weight_v.len()` disagrees with the product of `v_shape`;
/// - `weight_g.len()` is not exactly the first-dim length of `v_shape`
///   (the reader has already flattened both from safetensors byte form,
///   so this only fires on genuinely corrupt inputs).
fn restore_weight_norm_f32(
    weight_g: &[f32],
    weight_v: &[f32],
    v_shape: &[u64],
) -> Result<Vec<u8>, String> {
    if v_shape.is_empty() {
        return Err("weight_v shape is empty (cannot dim=0-normalize a 0-dim tensor)".to_owned());
    }
    let n_rows = v_shape[0] as usize;
    // `iter().product()` on an empty slice returns 1 — perfect: a
    // shape `[N]` (1-D `weight_v`) then reduces to `row_len = 1`.
    let row_len: usize = v_shape[1..].iter().map(|&d| d as usize).product();
    let expected_v_len = n_rows.saturating_mul(row_len);
    if weight_v.len() != expected_v_len {
        return Err(format!(
            "weight_v shape {v_shape:?} expects {expected_v_len} f32 elements but got {}",
            weight_v.len()
        ));
    }
    if weight_g.len() != n_rows {
        return Err(format!(
            "weight_g flat length {} does not match v_shape[0]={n_rows} (weight_g must be \
             [{n_rows}, 1, ...])",
            weight_g.len()
        ));
    }

    let mut out = Vec::with_capacity(weight_v.len() * 4);
    for (row, &g_scalar) in weight_g.iter().enumerate() {
        let start = row * row_len;
        let end = start + row_len;
        let row_slice = &weight_v[start..end];
        // L2 norm across the trailing dims (row-wise).
        let norm_sq: f32 = row_slice.iter().map(|&x| x * x).sum();
        let scale = if norm_sq > 0.0 {
            g_scalar / norm_sq.sqrt()
        } else {
            0.0
        };
        for &v in row_slice {
            out.extend_from_slice(&(v * scale).to_le_bytes());
        }
    }
    Ok(out)
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
    fn tensors_land_under_renamed_targets_without_config_and_hparams_are_absent() {
        let blob = safetensors_multi(&base_fixture());
        let input = temp_path("no-config-in", "safetensors");
        let output = temp_path("no-config-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and BF16 both land under new names");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.skipped_training, 0,
            "fixture has no training-side tensors"
        );
        assert_eq!(
            report.renamed, 2,
            "both fixture tensors match fixed-rename entries (Task 30)"
        );
        assert_eq!(report.weight_norm_reconstructed, 0);
        assert_eq!(
            report.passed_through_verbatim, 0,
            "no unclassified tensor in this fixture"
        );
        assert_eq!(report.bf16_passthrough, 1);
        assert!(!report.hparams_written, "no config side-car was supplied");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        // Task 30: upstream `enc_p.emb.weight` now lands as
        // `sbv2.text_encoder.phoneme_embed`, and `dec.ups.0.weight`
        // (bare-weight fixture variant, not weight_norm-split) lands as
        // `sbv2.decoder.upsample.0.weight`. The upstream names must be
        // absent — a stale converter would still stamp them and the
        // Rust loader would then load nothing.
        assert!(
            file.tensor_info("sbv2.text_encoder.phoneme_embed")
                .is_some(),
            "enc_p.emb.weight must land under its renamed target"
        );
        assert!(
            file.tensor_info("enc_p.emb.weight").is_none(),
            "upstream name must not leak through when a rename applies"
        );
        let bf16_info = file
            .tensor_info("sbv2.decoder.upsample.0.weight")
            .expect("BF16 tensor present under its renamed target");
        assert!(
            file.tensor_info("dec.ups.0.weight").is_none(),
            "upstream name must not leak through when a rename applies"
        );
        assert_eq!(
            bf16_info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16 (type 30) \
             even under the renamed target"
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

    // ====================================================================
    // Task 30: classify_tensor unit tests
    // ====================================================================

    #[test]
    fn classify_fixed_renames_hit_expected_target_names() {
        // Every entry below is a bit-exact anchor to the Task 30 rename
        // table's "Fixed 1:1 renames" arm — if the target string here
        // drifts from the classify_tensor arm, a stale converter would
        // silently stamp under the wrong name and Rust load would then
        // load nothing.
        assert_eq!(
            classify_tensor("enc_p.emb.weight", None),
            TensorClass::Rename("sbv2.text_encoder.phoneme_embed".into())
        );
        assert_eq!(
            classify_tensor("enc_p.tone_emb.weight", None),
            TensorClass::Rename("sbv2.text_encoder.tone_embed".into())
        );
        assert_eq!(
            classify_tensor("enc_p.language_emb.weight", None),
            TensorClass::Rename("sbv2.text_encoder.language_embed".into())
        );
        assert_eq!(
            classify_tensor("enc_p.bert_proj.weight", None),
            TensorClass::Rename("sbv2.bert_bridge.conv.weight".into())
        );
        assert_eq!(
            classify_tensor("enc_p.bert_proj.bias", None),
            TensorClass::Rename("sbv2.bert_bridge.conv.bias".into())
        );
        assert_eq!(
            classify_tensor("dec.conv_pre.weight", None),
            TensorClass::Rename("sbv2.decoder.conv_pre.weight".into())
        );
        assert_eq!(
            classify_tensor("dec.conv_pre.bias", None),
            TensorClass::Rename("sbv2.decoder.conv_pre.bias".into())
        );
        assert_eq!(
            classify_tensor("dec.conv_post.weight", None),
            TensorClass::Rename("sbv2.decoder.conv_post.weight".into())
        );
    }

    #[test]
    fn classify_dec_ups_weight_norm_pair_and_bias() {
        assert_eq!(
            classify_tensor("dec.ups.0.weight_g", None),
            TensorClass::WeightNorm {
                sibling_v: "dec.ups.0.weight_v".into(),
                target_name: "sbv2.decoder.upsample.0.weight".into(),
            }
        );
        assert_eq!(
            classify_tensor("dec.ups.3.weight_g", None),
            TensorClass::WeightNorm {
                sibling_v: "dec.ups.3.weight_v".into(),
                target_name: "sbv2.decoder.upsample.3.weight".into(),
            }
        );
        assert_eq!(
            classify_tensor("dec.ups.0.bias", None),
            TensorClass::Rename("sbv2.decoder.upsample.0.bias".into())
        );
        // Bare .weight (fixture / non-weight_norm variant) is still
        // renamed directly — this is what the base_fixture BF16 tensor
        // exercises.
        assert_eq!(
            classify_tensor("dec.ups.0.weight", None),
            TensorClass::Rename("sbv2.decoder.upsample.0.weight".into())
        );
    }

    #[test]
    fn classify_dec_resblocks_needs_config_for_stage_branch_split() {
        // No config → PassThrough (we cannot safely split flat index
        // without knowing n_branches; a wrong split would mis-shuffle
        // every ResBlock).
        let no_cfg = classify_tensor("dec.resblocks.0.convs1.0.weight_g", None);
        assert!(matches!(no_cfg, TensorClass::PassThrough { .. }));

        // With n_branches=3 (base checkpoint layout: 5 stages × 3
        // branches = 15 flat resblocks), flat_i=0 → (stage=0, branch=0)
        // and flat_i=4 → (stage=1, branch=1).
        assert_eq!(
            classify_tensor("dec.resblocks.0.convs1.0.weight_g", Some(3)),
            TensorClass::WeightNorm {
                sibling_v: "dec.resblocks.0.convs1.0.weight_v".into(),
                target_name: "sbv2.decoder.mrf.0.0.layer.0.weight".into(),
            }
        );
        assert_eq!(
            classify_tensor("dec.resblocks.4.convs1.2.weight_g", Some(3)),
            TensorClass::WeightNorm {
                sibling_v: "dec.resblocks.4.convs1.2.weight_v".into(),
                target_name: "sbv2.decoder.mrf.1.1.layer.2.weight".into(),
            }
        );
        assert_eq!(
            classify_tensor("dec.resblocks.14.convs1.1.weight_g", Some(3)),
            TensorClass::WeightNorm {
                sibling_v: "dec.resblocks.14.convs1.1.weight_v".into(),
                target_name: "sbv2.decoder.mrf.4.2.layer.1.weight".into(),
            }
        );
        // convs1 bias / bare weight also remap (with math).
        assert_eq!(
            classify_tensor("dec.resblocks.7.convs1.0.bias", Some(3)),
            TensorClass::Rename("sbv2.decoder.mrf.2.1.layer.0.bias".into())
        );
    }

    #[test]
    fn classify_dec_resblocks_convs2_always_passes_through() {
        // Rust's simplified HiFi-GAN has no consumer for convs2
        // (undilated residual conv). Even with a valid config that would
        // let us do the stage/branch math for convs1, convs2 stays
        // pass-through until a later architecture wave lands.
        for cfg in [None, Some(3usize)] {
            let out = classify_tensor("dec.resblocks.0.convs2.0.weight_g", cfg);
            assert!(
                matches!(out, TensorClass::PassThrough { .. }),
                "convs2 must be PassThrough (cfg={cfg:?}) but got {out:?}"
            );
        }
    }

    #[test]
    fn classify_training_side_families_are_skipped() {
        for name in [
            "enc_q.pre.weight",
            "enc_q.enc.in_layers.0.weight_v",
            "dp.conv_1.weight",
            "dp.norm_1.gamma",
            "sdp.post_pre.weight",
            "sdp.post_flows.1.pre.bias",
            "sdp.post_convs.convs_1x1.0.weight",
        ] {
            assert!(
                matches!(classify_tensor(name, Some(3)), TensorClass::Skip { .. }),
                "{name} must be Skip"
            );
        }
        // Production sdp.* (non-post_*) is PassThrough, not Skip — the
        // "post_" boundary is the delineator.
        for name in [
            "sdp.pre.weight",
            "sdp.proj.bias",
            "sdp.convs.convs_1x1.0.weight",
            "sdp.flows.0.m",
            "sdp.flows.2.pre.weight",
            "sdp.cond.weight",
        ] {
            assert!(
                matches!(
                    classify_tensor(name, Some(3)),
                    TensorClass::PassThrough { .. }
                ),
                "{name} must be PassThrough (production SDP path), not Skip"
            );
        }
    }

    #[test]
    fn classify_blocker_families_are_pass_through_with_reason() {
        for name in [
            "flow.flows.0.enc.attn_layers.0.conv_q.weight",
            "flow.flows.3.post.bias",
            "enc_p.encoder.attn_layers.0.conv_q.weight",
            "enc_p.encoder.spk_emb_linear.weight", // Blocker 3
            "enc_p.encoder.ffn_layers.0.conv_1.bias",
            "enc_p.encoder.norm_layers_1.0.gamma",
            "enc_p.proj.weight",
            "dec.cond.weight",
            "dec.cond.bias",
        ] {
            assert!(
                matches!(
                    classify_tensor(name, Some(3)),
                    TensorClass::PassThrough { .. }
                ),
                "{name} must be PassThrough"
            );
        }
    }

    #[test]
    fn classify_unknown_name_falls_through_to_pass_through_default() {
        // Preserves data (with a reason string) rather than silently
        // dropping — FR-EX-08. Real checkpoints must not add nameless
        // tensors, but a future upstream refactor could plausibly land
        // one; verbatim preservation gives an owner a chance to inspect.
        let cls = classify_tensor("something_new_upstream_added.weight", None);
        assert!(matches!(cls, TensorClass::PassThrough { .. }));
    }

    // ====================================================================
    // Task 30: restore_weight_norm_f32 unit tests
    // ====================================================================

    #[test]
    fn restore_weight_norm_single_row_normalizes_to_unit_and_scales() {
        // Row [3, 4] has L2 norm 5; weight_g=1 must give [3/5, 4/5].
        let g = [1.0_f32];
        let v = [3.0_f32, 4.0_f32];
        let out = restore_weight_norm_f32(&g, &v, &[1, 2]).expect("restore ok");
        assert_eq!(out.len(), 8, "2 f32 → 8 bytes");
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((out_f32[0] - 0.6).abs() < 1e-6);
        assert!((out_f32[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn restore_weight_norm_scales_by_g_when_v_is_unit_direction() {
        // Row [1, 0] has L2 norm 1; weight_g=7 must give [7, 0].
        let g = [7.0_f32];
        let v = [1.0_f32, 0.0_f32];
        let out = restore_weight_norm_f32(&g, &v, &[1, 2]).expect("restore ok");
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((out_f32[0] - 7.0).abs() < 1e-6);
        assert!(out_f32[1].abs() < 1e-6);
    }

    #[test]
    fn restore_weight_norm_zero_row_yields_zero_not_nan() {
        // A zero-norm row must produce a zero output (matches PyTorch
        // weight_norm's forward, which gates on norm > 0). Silent NaN
        // would poison downstream MRF outputs.
        let g = [42.0_f32];
        let v = [0.0_f32, 0.0_f32];
        let out = restore_weight_norm_f32(&g, &v, &[1, 2]).expect("restore ok");
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(out_f32.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn restore_weight_norm_multi_row_normalizes_per_row_independently() {
        // Two rows: [3,4] with weight_g=1 → [0.6, 0.8], and [0,5] with
        // weight_g=2 → [0, 2]. Interleaved in v as row-major flat.
        let g = [1.0_f32, 2.0_f32];
        let v = [3.0_f32, 4.0_f32, 0.0_f32, 5.0_f32];
        let out = restore_weight_norm_f32(&g, &v, &[2, 2]).expect("restore ok");
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((out_f32[0] - 0.6).abs() < 1e-6);
        assert!((out_f32[1] - 0.8).abs() < 1e-6);
        assert!(out_f32[2].abs() < 1e-6);
        assert!((out_f32[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn restore_weight_norm_matches_real_conv1d_shape_512x256x3() {
        // Shape [512, 256, 3] mirrors dec.resblocks.*.convs1.*.weight_v
        // (Conv1d with out_ch=256... wait, actually resblocks are 256×256×3;
        // this test uses a scaled-down proxy). Verifies the row_len math
        // works for 3-D shapes: row_len = 256*3 = 768.
        let n_rows = 4_usize;
        let row_len = 256 * 3;
        let g: Vec<f32> = (0..n_rows).map(|i| (i + 1) as f32).collect();
        // Fill each row with the same distinctive pattern so we can
        // check the per-row scale independently.
        let mut v = Vec::with_capacity(n_rows * row_len);
        for _ in 0..n_rows {
            for k in 0..row_len {
                v.push(if k == 0 { 1.0 } else { 0.0 });
            }
        }
        let out = restore_weight_norm_f32(&g, &v, &[n_rows as u64, 256, 3]).expect("restore ok");
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Each row's L2 norm = 1 (only k=0 nonzero); so scale = g[row]/1
        // = g[row], and only position 0 of each row should be nonzero
        // and equal to g[row].
        for (row, &g_scalar) in g.iter().enumerate() {
            let base = row * row_len;
            assert!((out_f32[base] - g_scalar).abs() < 1e-6, "row {row} col 0");
            for k in 1..row_len {
                assert!(out_f32[base + k].abs() < 1e-6, "row {row} col {k}");
            }
        }
    }

    #[test]
    fn restore_weight_norm_rejects_empty_shape() {
        let err = restore_weight_norm_f32(&[1.0], &[1.0], &[]).expect_err("empty shape must fail");
        assert!(err.contains("empty"), "message: {err}");
    }

    #[test]
    fn restore_weight_norm_rejects_v_len_shape_mismatch() {
        let err = restore_weight_norm_f32(&[1.0], &[1.0, 2.0, 3.0], &[1, 2])
            .expect_err("v_len != product(shape) must fail");
        assert!(err.contains("expects"), "message: {err}");
    }

    #[test]
    fn restore_weight_norm_rejects_g_len_row_mismatch() {
        let err = restore_weight_norm_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1, 2])
            .expect_err("g_len != rows must fail");
        assert!(err.contains("weight_g"), "message: {err}");
    }

    // ====================================================================
    // Task 30: end-to-end convert_sbv2_file smoke tests
    // ====================================================================

    #[test]
    fn training_side_tensors_are_dropped_with_stderr_log() {
        // Build a fixture containing one renamed tensor + one
        // training-side tensor (enc_q.pre.weight). The training-side
        // tensor must be dropped (skipped_training=1), a stderr line
        // emitted (validated via non-panic behavior only — we do not
        // capture stderr in tests, but the counter increment is the
        // observable invariant), and the renamed tensor must still land.
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            ("enc_p.emb.weight", "F32", &[6, 4], f32_bytes(&[0.01; 24])),
            (
                "enc_q.pre.weight",
                "F32",
                &[8, 4],
                f32_bytes(&[0.5_f32; 32]),
            ),
            (
                "dp.conv_1.weight",
                "F32",
                &[2, 4, 3],
                f32_bytes(&[0.5_f32; 24]),
            ),
            (
                "sdp.post_pre.weight",
                "F32",
                &[4, 4, 1],
                f32_bytes(&[0.5_f32; 16]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("training-skip-in", "safetensors");
        let output = temp_path("training-skip-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 4);
        assert_eq!(report.written, 1, "only enc_p.emb.weight survives");
        assert_eq!(report.renamed, 1);
        assert_eq!(
            report.skipped_training, 3,
            "enc_q.* + dp.* + sdp.post_* all dropped"
        );
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        assert!(
            file.tensor_info("sbv2.text_encoder.phoneme_embed")
                .is_some()
        );
        assert!(file.tensor_info("enc_q.pre.weight").is_none());
        assert!(file.tensor_info("dp.conv_1.weight").is_none());
        assert!(file.tensor_info("sdp.post_pre.weight").is_none());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn weight_norm_pair_reconstructs_to_renamed_target() {
        // dec.ups.0.weight_g [1] = 5.0, weight_v [1,2] = [3, 4]:
        // norm=5, scale=5/5=1 → restored [3, 4] under
        // sbv2.decoder.upsample.0.weight.
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            ("dec.ups.0.weight_g", "F32", &[1, 1, 1], f32_bytes(&[5.0])),
            ("dec.ups.0.weight_v", "F32", &[1, 2], f32_bytes(&[3.0, 4.0])),
            ("dec.ups.0.bias", "F32", &[4], f32_bytes(&[0.1; 4])),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("wnorm-pair-in", "safetensors");
        let output = temp_path("wnorm-pair-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 3);
        assert_eq!(
            report.written, 2,
            "weight_g+weight_v folded → 1 emit + 1 bias"
        );
        assert_eq!(report.weight_norm_reconstructed, 1);
        assert_eq!(report.renamed, 1, "bias uses the direct rename arm");

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        // Both upstream weight_g and weight_v must be absent.
        assert!(file.tensor_info("dec.ups.0.weight_g").is_none());
        assert!(file.tensor_info("dec.ups.0.weight_v").is_none());
        // Reconstructed weight lands under the renamed target with the
        // shape of weight_v.
        let w = file
            .tensor_info("sbv2.decoder.upsample.0.weight")
            .expect("reconstructed weight");
        assert_eq!(w.dimensions, vec![1_u64, 2]);
        assert_eq!(w.dtype, GgmlType::F32);
        // And bias lands separately.
        assert!(file.tensor_info("sbv2.decoder.upsample.0.bias").is_some());
        // Reconstructed values: [3, 4] (weight_g=5 / ||v||=5 = 1x).
        let w_bytes = file.tensor_bytes(w);
        let w_f32: Vec<f32> = w_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((w_f32[0] - 3.0).abs() < 1e-6);
        assert!((w_f32[1] - 4.0).abs() < 1e-6);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn orphan_weight_g_falls_back_to_verbatim_pass_through() {
        // A weight_g without a sibling weight_v must not silently drop —
        // fall back to verbatim pass-through under the upstream name so
        // an owner can inspect the corrupt / partial checkpoint.
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            (
                "dec.ups.0.weight_g",
                "F32",
                &[3, 1, 1],
                f32_bytes(&[1.0; 3]),
            ),
            // no sibling dec.ups.0.weight_v
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("orphan-g-in", "safetensors");
        let output = temp_path("orphan-g-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(
            report.passed_through_verbatim, 1,
            "orphan weight_g becomes verbatim pass-through, not reconstruction"
        );
        assert_eq!(report.weight_norm_reconstructed, 0);
        assert_eq!(report.renamed, 0);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        assert!(
            file.tensor_info("dec.ups.0.weight_g").is_some(),
            "orphan weight_g must survive under its upstream name"
        );
        assert!(
            file.tensor_info("sbv2.decoder.upsample.0.weight").is_none(),
            "no target emit without a real weight_v to reconstruct from"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn flow_and_encoder_families_pass_through_verbatim_when_no_rename_applies() {
        // Preservation invariant: flow.* + enc_p.encoder.* + sdp.* +
        // dec.cond.* stay under their upstream names so a future Rust
        // wave can consume them without reconverting the checkpoint.
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            (
                "flow.flows.0.enc.attn_layers.0.conv_q.weight",
                "F32",
                &[192, 192, 1],
                f32_bytes(&[0.01_f32; 192 * 192]),
            ),
            (
                "enc_p.encoder.attn_layers.0.conv_q.weight",
                "F32",
                &[192, 192, 1],
                f32_bytes(&[0.02_f32; 192 * 192]),
            ),
            (
                "sdp.flows.1.pre.weight",
                "F32",
                &[192, 1, 1],
                f32_bytes(&[0.03_f32; 192]),
            ),
            (
                "dec.cond.weight",
                "F32",
                &[512, 512, 1],
                f32_bytes(&[0.04_f32; 512 * 512]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("verbatim-pt-in", "safetensors");
        let output = temp_path("verbatim-pt-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 4);
        assert_eq!(report.written, 4);
        assert_eq!(report.passed_through_verbatim, 4);
        assert_eq!(report.renamed, 0);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        for name in [
            "flow.flows.0.enc.attn_layers.0.conv_q.weight",
            "enc_p.encoder.attn_layers.0.conv_q.weight",
            "sdp.flows.1.pre.weight",
            "dec.cond.weight",
        ] {
            assert!(
                file.tensor_info(name).is_some(),
                "{name}: must land under upstream name"
            );
        }

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn dec_resblocks_convs2_pass_through_verbatim_never_lands_under_mrf() {
        // convs2 is the "convs1 with dilation=1" residual conv in
        // HiFi-GAN's ResBlock1; Rust's simplified impl has no landing
        // spot for it. Even with a valid config, it must stay verbatim
        // rather than being folded into the mrf.*.layer.* path.
        let cfg_bytes = valid_config_json();
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            (
                "dec.resblocks.0.convs2.0.weight_g",
                "F32",
                &[8, 1, 1],
                f32_bytes(&[1.0_f32; 8]),
            ),
            (
                "dec.resblocks.0.convs2.0.weight_v",
                "F32",
                &[8, 8, 3],
                f32_bytes(&[0.5_f32; 8 * 8 * 3]),
            ),
            (
                "dec.resblocks.0.convs2.0.bias",
                "F32",
                &[8],
                f32_bytes(&[0.1_f32; 8]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("convs2-pt-in", "safetensors");
        let config = temp_path("convs2-pt-cfg", "json");
        let output = temp_path("convs2-pt-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        std::fs::write(&config, &cfg_bytes).expect("write cfg");

        let report = convert_sbv2_file(&input, &output, Some(&config), None).expect("convert");
        // All 3 tensors verbatim pass-through (bias, weight_g, weight_v).
        assert_eq!(report.read, 3);
        assert_eq!(
            report.passed_through_verbatim, 3,
            "convs2 stays fully verbatim under upstream names"
        );
        assert_eq!(report.renamed, 0);
        assert_eq!(report.weight_norm_reconstructed, 0);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        assert!(
            file.tensor_info("dec.resblocks.0.convs2.0.weight_g")
                .is_some(),
            "convs2 weight_g preserved under upstream name"
        );
        // Confirm we did NOT accidentally emit anything under the mrf.*
        // path — no convs2-driven target should land at mrf.0.0.layer.0.
        assert!(
            file.tensor_info("sbv2.decoder.mrf.0.0.layer.0.weight")
                .is_none(),
            "convs2 must never land under the mrf.*.layer.*.weight target"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&config).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn dec_resblocks_convs1_lands_under_mrf_with_config_stage_branch_split() {
        // With n_branches=3, flat_i=4 → (stage=1, branch=1). Build a
        // fixture with dec.resblocks.4.convs1.2.{weight_g,weight_v,bias}
        // and verify the mrf.1.1.layer.2.{weight,bias} landing.
        //
        // The config has decoder_resblock_kernel_sizes=[3, 5, 7] so the
        // stage/branch math activates (n_branches=3).
        let cfg_bytes = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 192, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8, 8, 2, 2, 2],
            "decoder_upsample_kernel_sizes": [16, 16, 4, 4, 4],
            "decoder_upsample_out_channels": [256, 128, 64, 32, 16],
            "decoder_resblock_kernel_sizes": [3, 5, 7],
            "decoder_resblock_dilation_counts": [3, 3, 3],
            "decoder_resblock_dilations_flat": [1, 3, 5, 1, 3, 5, 1, 3, 5]
        }"#;
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            (
                "dec.resblocks.4.convs1.2.weight_g",
                "F32",
                &[8, 1, 1],
                f32_bytes(&[1.0_f32; 8]),
            ),
            (
                "dec.resblocks.4.convs1.2.weight_v",
                "F32",
                &[8, 8, 3],
                f32_bytes(&[0.5_f32; 8 * 8 * 3]),
            ),
            (
                "dec.resblocks.4.convs1.2.bias",
                "F32",
                &[8],
                f32_bytes(&[0.1_f32; 8]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("convs1-mrf-in", "safetensors");
        let config = temp_path("convs1-mrf-cfg", "json");
        let output = temp_path("convs1-mrf-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        std::fs::write(&config, cfg_bytes).expect("write cfg");

        let report = convert_sbv2_file(&input, &output, Some(&config), None).expect("convert");
        assert_eq!(report.read, 3);
        assert_eq!(report.written, 2, "weight pair + bias → 2 emits");
        assert_eq!(report.weight_norm_reconstructed, 1);
        assert_eq!(report.renamed, 1);
        assert!(report.hparams_written);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        // flat_i=4, n_branches=3 → stage=1, branch=1, layer=2.
        assert!(
            file.tensor_info("sbv2.decoder.mrf.1.1.layer.2.weight")
                .is_some(),
            "reconstructed weight lands at mrf.1.1.layer.2.weight"
        );
        assert!(
            file.tensor_info("sbv2.decoder.mrf.1.1.layer.2.bias")
                .is_some(),
            "bias lands at mrf.1.1.layer.2.bias"
        );
        // Upstream names must be gone.
        assert!(
            file.tensor_info("dec.resblocks.4.convs1.2.weight_g")
                .is_none()
        );
        assert!(
            file.tensor_info("dec.resblocks.4.convs1.2.weight_v")
                .is_none()
        );
        assert!(file.tensor_info("dec.resblocks.4.convs1.2.bias").is_none());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&config).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn write_counters_partition_read_counter_exactly() {
        // Invariant: `written + skipped_non_float + skipped_training ==
        // read`, and `renamed + weight_norm_reconstructed +
        // passed_through_verbatim == written`. Mix all four buckets in
        // one fixture to lock the partition.
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            // Rename
            ("enc_p.emb.weight", "F32", &[6, 4], f32_bytes(&[0.01; 24])),
            // Skip (training-side)
            (
                "enc_q.pre.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[0.5_f32; 16]),
            ),
            // PassThrough (production SDP)
            (
                "sdp.pre.weight",
                "F32",
                &[4, 4, 1],
                f32_bytes(&[0.03_f32; 16]),
            ),
            // WeightNorm pair (bias renamed, weight reconstructed)
            (
                "dec.ups.0.weight_g",
                "F32",
                &[1, 1, 1],
                f32_bytes(&[5.0_f32]),
            ),
            (
                "dec.ups.0.weight_v",
                "F32",
                &[1, 2],
                f32_bytes(&[3.0_f32, 4.0_f32]),
            ),
            ("dec.ups.0.bias", "F32", &[4], f32_bytes(&[0.1_f32; 4])),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("partition-in", "safetensors");
        let output = temp_path("partition-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let r = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(r.read, 6);
        assert_eq!(
            r.written + r.skipped_non_float + r.skipped_training + r.weight_norm_v_consumed,
            r.read,
            "input-side partition: written + skipped_non_float + skipped_training + \
             weight_norm_v_consumed == read"
        );
        assert_eq!(
            r.renamed + r.weight_norm_reconstructed + r.passed_through_verbatim,
            r.written,
            "output-side partition: renamed + wnorm_reconstructed + verbatim == written"
        );
        // Concrete values: enc_p.emb.weight=Rename,
        // enc_q.pre.weight=Skip, sdp.pre.weight=PassThrough,
        // (dec.ups.0.weight_g + dec.ups.0.weight_v)=WeightNorm — the
        // weight_g reconstructs (+1 written, +1 wnorm), the weight_v is
        // consumed (+1 weight_norm_v_consumed), and dec.ups.0.bias=Rename.
        assert_eq!(r.renamed, 2);
        assert_eq!(r.weight_norm_reconstructed, 1);
        assert_eq!(r.weight_norm_v_consumed, 1, "one weight_v folded");
        assert_eq!(r.passed_through_verbatim, 1);
        assert_eq!(r.skipped_training, 1);
        assert_eq!(r.written, 4);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
