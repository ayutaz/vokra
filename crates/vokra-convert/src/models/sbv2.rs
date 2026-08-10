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
//!    `(mu, log_sigma)`), and `enc_p.encoder.spk_emb_linear.*` ("Blocker
//!    3" speaker-conditioning projection). Pre-Wave-2 (2026-08-09) also
//!    included `dec.resblocks.*.convs2.*` — that arm now
//!    [`Rename`](TensorClass::Rename)s to `sbv2.decoder.mrf.<s>.<b>.layer.<l>.{weight,bias}_c2`
//!    as part of the HGAN-01 fix; see [`classify_tensor`]'s convs2
//!    arm. See [`classify_tensor`] for the per-family reason strings.
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
//! - Rename emits (post-Wave-2): 604 = 58 (pre-Blocker-2b) + 456 (Blocker
//!   2b flow family: `flow.flows.{0,2,4,6}.*` = 4 blocks × 114 tensors/block,
//!   see [`classify_flow_block_tensor`] for the per-family breakdown) +
//!   90 (HGAN-01 convs2 bare-weight + bias renames — 15 flat resblocks ×
//!   3 layers × 2 tail slots)
//! - WeightNorm reconstructions (post-Wave-2): 95 (`dec.ups.{0..4}` = 5 +
//!   `dec.resblocks.{0..14}.convs1.{0..2}` = 45 +
//!   `dec.resblocks.{0..14}.convs2.{0..2}` = 45 — HGAN-01 added the
//!   convs2 weight_g/weight_v pairs)
//! - Verbatim pass-through (post-Wave-2): 258 = 393 (pre-Wave-2)
//!   − 135 (`dec.resblocks.*.convs2.*` now renamed) = `enc_p.encoder.spk_emb_linear.*` = 2 +
//!   `sdp.*` production (not `post_`) = 144 +
//!   `enc_p.proj.*` = 2 +
//!   `dec.cond.*` = 2 + auxiliary residual
//! - **Total written to output GGUF: 957** = 604 + 95 + 258 (unchanged
//!   from pre-Wave-2 — the same 957 tensors land in the output, just
//!   under new target names for the convs2 family)
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
//! `SbV2Model::from_gguf` requires 23 `vokra.sbv2.*` metadata keys (13
//! top-level dims + 3 decoder scalars + 6 decoder arrays + the
//! `decoder.leaky_relu_slope` `f32`; WP-13 promoted the slope from
//! optional-with-`0.1`-fallback to required per FR-EX-08) — none of
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
//!   `req()` closure). `SbV2Config::parse` additionally cross-checks the
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
// Post-M6 (2026-08-06) relative-position transformer hparams — the SBV2
// v2 real base checkpoint has a 6-layer relative-position transformer
// stack under `enc_p.encoder.*` (see the module doc's "M6 refactor"
// section) and the Rust loader (`SbV2Model::from_gguf`) requires all
// three to construct `RelPositionMHA` / `PositionWiseFFN`. Every
// architecture-parameterizing hparam is stamped as its own key rather
// than being pinned to a design-doc constant, so future SBV2 SKUs
// (varying `n_heads` / `window` / `kernel_ffn`) round-trip through the
// converter without a code change on the runtime side.
const KEY_N_HEADS: &str = "vokra.sbv2.n_heads";
const KEY_WINDOW_SIZE: &str = "vokra.sbv2.window_size";
const KEY_KERNEL_FFN: &str = "vokra.sbv2.kernel_ffn";
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
/// [`SbV2Config`] does not carry it and `SbV2Config::parse` does not
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

// Optional at the JSON side-car level only (WP-13 promoted the loader-side
// read to required per FR-EX-08 — see `SbV2Model::from_gguf`'s doc). The
// converter always emits this key downstream, using
// `DEFAULT_LEAKY_RELU_SLOPE` when the side-car omits it, so no Vokra-produced
// GGUF ever ends up with the key missing.
const KEY_DECODER_LEAKY_RELU_SLOPE: &str = "vokra.sbv2.decoder.leaky_relu_slope";

/// Default `decoder.leaky_relu_slope` when the config side-car omits it —
/// the universal jik876/hifi-gan `LRELU_SLOPE` every sibling decoder in
/// this codebase compiles in (`vits_ja::VITS_JA_LEAKY_RELU_SLOPE`,
/// piper-plus's `LRELU_SLOPE`). This defaulting is a **converter-side
/// convenience only** (it lets a stock config side-car omit the field);
/// the loader (`SbV2Model::from_gguf`) requires the emitted GGUF metadata
/// key unconditionally (WP-13, FR-EX-08).
const DEFAULT_LEAKY_RELU_SLOPE: f32 = 0.1;

/// Post-M6 (2026-08-06) relative-position transformer defaults for the
/// text encoder. Values pin the SBV2 v2 base checkpoint
/// (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`) as verified from the
/// real `enc_p.encoder.attn_layers.0.*` tensor shapes:
///
/// - `emb_rel_k`/`emb_rel_v` shape `[1, 9, 96]` → `n_heads = 2`
///   (`d_model=192 / d_head=96`), `window_size = 4` (`2*4+1=9`).
/// - `ffn_layers.0.conv_1.weight` shape `[768, 192, 3]` → `kernel_ffn = 3`.
///
/// Used by [`SbV2Config`] as the default when a JSON side-car omits
/// `n_heads` / `window_size` / `kernel_ffn`, matching the base value all
/// existing SBV2 v2 SKUs use; a config that varies the architecture (a
/// hypothetical `n_heads = 4` future SKU) overrides these by supplying
/// its own values in the side-car.
const DEFAULT_N_HEADS: u32 = 2;
const DEFAULT_WINDOW_SIZE: u32 = 4;
const DEFAULT_KERNEL_FFN: u32 = 3;

// Blocker 2b (2026-08-06) — flow's inner transformer / coupling
// hparams. Values pin the SBV2 v2 base checkpoint's real per-block flow
// tensor shapes verified via `/tmp/sbv2-fixtures/sbv2-prep/G_0.safetensors`:
//
// - `flow.flows.0.enc.attn_layers.0..5.*` = 6 layers per coupling.
// - `flow.flows.0.enc.ffn_layers.0.conv_1.weight` shape `[768, 192, 5]`
//   → `kernel_ffn_flow = 5` (distinct from text encoder's `3`).
// - `flow.flows.0.enc.spk_emb_linear.weight` shape `[192, 512]`
//   → `gin_channels = 512`.
// - `flow.flows.0.post.weight` shape `[96, 192, 1]` → `post_out_dim =
//   half_d_z` (mean_only=True; the same shape under mean_only=False
//   would be `[192, 192, 1]`).
//
// Used by [`SbV2Config`] as the default when a JSON side-car omits
// these keys, matching the base value every existing SBV2 v2 SKU uses;
// a config that varies the flow architecture overrides these by
// supplying its own values in the side-car.
const DEFAULT_FLOW_N_ENCODER_LAYERS: u32 = 6;
const DEFAULT_FLOW_KERNEL_FFN: u32 = 5;
const DEFAULT_FLOW_GIN_CHANNELS: u32 = 512;
const DEFAULT_FLOW_MEAN_ONLY: bool = true;

// Blocker 2b (2026-08-06) — flow hparam metadata keys. All read by
// `SbV2Model::from_gguf` from the `vokra.sbv2.flow.*` chunk group,
// only when `vokra.sbv2.n_flow_layers > 0` (an empty flow stack skips
// them entirely to stay backward-compatible with pre-Blocker-2b
// GGUFs).
const KEY_FLOW_N_ENCODER_LAYERS: &str = "vokra.sbv2.flow.n_encoder_layers";
const KEY_FLOW_KERNEL_FFN: &str = "vokra.sbv2.flow.kernel_ffn";
const KEY_FLOW_GIN_CHANNELS: &str = "vokra.sbv2.flow.gin_channels";
const KEY_FLOW_MEAN_ONLY: &str = "vokra.sbv2.flow.mean_only";

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

/// Wave-4 WORKFLOW-SHAPE-FIXUP (2026-08-09): recovers `d_speaker` from
/// the input safetensors' `enc_p.encoder.spk_emb_linear.weight` tensor
/// shape (`[d_model, d_speaker]`) — the definitive per-checkpoint source.
/// Fallback candidates: `emb_g.weight[1]` (speaker embedding table's
/// per-row width). Returns `None` when neither tensor is present (e.g.
/// single-speaker fine-tunes that ship no speaker weights).
///
/// The pre-Wave-4 pattern lived inline in `.github/workflows/parity-sbv2-
/// real.yml`'s `python3 - <<PYEOF` block; moving it into the converter
/// itself so the recovery runs everywhere `convert_sbv2_file` runs, not
/// only inside CI.
fn infer_d_speaker(st: &SafetensorsFile) -> Option<u32> {
    // Primary: enc_p.encoder.spk_emb_linear.weight has shape
    // [d_model, d_speaker] — Vokra loader's own dimensionality contract.
    for cand in [
        "enc_p.encoder.spk_emb_linear.weight",
        // Fallback: some fine-tunes host spk_emb_linear elsewhere. Add
        // aliases here as they surface; every candidate MUST have a
        // shape whose LAST dim is d_speaker to be a valid match.
    ] {
        if let Some(info) = st.tensor_info(cand)
            && let Some(&last) = info.shape.last()
            && let Ok(u) = u32::try_from(last)
        {
            return Some(u);
        }
    }
    // Fallback: emb_g.weight shape [n_speakers, d_speaker].
    if let Some(info) = st.tensor_info("emb_g.weight")
        && info.shape.len() == 2
        && let Ok(u) = u32::try_from(info.shape[1])
    {
        return Some(u);
    }
    None
}

/// Wave-4 WORKFLOW-SHAPE-FIXUP: recovers `n_speakers` from `emb_g.weight`
/// shape (`[n_speakers, d_speaker]`) — the definitive per-checkpoint
/// source (mirrors the sibling `infer_d_speaker`'s fallback path but
/// reads the first-axis extent). Single-speaker fine-tunes that ship no
/// `emb_g` return `None`.
fn infer_n_speakers(st: &SafetensorsFile) -> Option<u32> {
    st.tensor_info("emb_g.weight")
        .filter(|info| info.shape.len() == 2)
        .and_then(|info| u32::try_from(info.shape[0]).ok())
}

/// Wave-4 WORKFLOW-SHAPE-FIXUP: recovers the per-stage HiFi-GAN upsample
/// kernel sizes from `dec.ups.<i>.weight_v` (weight-normed) or
/// `dec.ups.<i>.weight` (plain) tensor shape's LAST axis. Real SBV2 v2
/// JP-Extra base ships `[16, 16, 8, 2, 2]`, not the HiFi-GAN 2*stride
/// default `[16, 16, 4, 4, 4]` the config prep-script's
/// `--clean-room-defaults` emits. Returns `None` when any stage's
/// tensor is missing (incomplete recovery aborts to keep the config's
/// declared value in play).
///
/// Follows the same 5-stage 44.1 kHz probe pattern the CI workflow's
/// inline Python used; extended here to the same probe logic in native
/// Rust so `convert_sbv2_file` is self-contained.
fn infer_decoder_upsample_kernel_sizes(st: &SafetensorsFile) -> Option<Vec<u32>> {
    const MAX_STAGES: usize = 8; // HiFi-GAN ladder cap — 5 stages typical
    let mut out = Vec::with_capacity(MAX_STAGES);
    for i in 0..MAX_STAGES {
        let key_v = format!("dec.ups.{i}.weight_v");
        let key_plain = format!("dec.ups.{i}.weight");
        let info = st
            .tensor_info(&key_v)
            .or_else(|| st.tensor_info(&key_plain));
        let Some(info) = info else {
            // Stop on first absent stage; that's how many stages exist.
            break;
        };
        // Every ConvTranspose1d weight is 3-D [in_ch, out_ch, kernel].
        if info.shape.len() != 3 {
            return None;
        }
        let Ok(kernel) = u32::try_from(info.shape[2]) else {
            return None;
        };
        out.push(kernel);
    }
    if out.is_empty() { None } else { Some(out) }
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
    /// Post-M6 relative-position transformer hparam: attention head
    /// count for the SBV2 text encoder's `enc_p.encoder.*` stack.
    /// Optional in the JSON side-car — defaults to [`DEFAULT_N_HEADS`]
    /// (`2`, the SBV2 v2 base value), mirroring the Rust loader's
    /// hard-fail on missing `vokra.sbv2.n_heads` metadata (a config
    /// side-car missing `n_heads` still stamps the default, so a real
    /// checkpoint whose config side-car omits this key still round-trips
    /// through the loader without change).
    pub(crate) n_heads: u32,
    /// Post-M6 relative-position transformer hparam: half-window (`w`)
    /// for the relative-position bias table (`emb_rel_k/v` have shape
    /// `[1, 2*w+1, d_head]`). Optional in the JSON side-car — defaults
    /// to [`DEFAULT_WINDOW_SIZE`] (`4`, matching the SBV2 v2 base value
    /// verified from `enc_p.encoder.attn_layers.0.emb_rel_k` shape
    /// `[1, 9, 96]`).
    pub(crate) window_size: u32,
    /// Post-M6 relative-position transformer hparam: FFN Conv1d kernel
    /// width (SBV2 v2 uses same-padded kernel=3). Optional in the JSON
    /// side-car — defaults to [`DEFAULT_KERNEL_FFN`] (`3`).
    pub(crate) kernel_ffn: u32,
    pub(crate) n_flow_layers: u32,
    /// Blocker 2b (2026-08-06): flow inner transformer stack depth
    /// (6 on the SBV2 v2 base — see [`DEFAULT_FLOW_N_ENCODER_LAYERS`]).
    /// Optional in the JSON side-car; the runtime hard-fails on missing
    /// `vokra.sbv2.flow.n_encoder_layers` when `n_flow_layers > 0`, so
    /// the converter always emits this key alongside `n_flow_layers`.
    pub(crate) flow_n_encoder_layers: u32,
    /// Blocker 2b (2026-08-06): flow FFN Conv1d kernel width (`5` on
    /// the SBV2 v2 base, distinct from the text encoder's `kernel_ffn
    /// = 3`). Optional in the JSON side-car — defaults to
    /// [`DEFAULT_FLOW_KERNEL_FFN`].
    pub(crate) flow_kernel_ffn: u32,
    /// Blocker 2b (2026-08-06): flow per-block `spk_emb_linear` input
    /// width — upstream `TransformerCouplingLayer.spk_emb_linear`'s
    /// `gin_channels` argument (`512` on the SBV2 v2 base). Optional in
    /// the JSON side-car — defaults to [`DEFAULT_FLOW_GIN_CHANNELS`].
    pub(crate) flow_gin_channels: u32,
    /// Blocker 2b (2026-08-06): flow coupling's `mean_only` flag
    /// (`true` on the SBV2 v2 base). Optional in the JSON side-car —
    /// defaults to [`DEFAULT_FLOW_MEAN_ONLY`].
    pub(crate) flow_mean_only: bool,
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
    /// when the key is absent — a converter-side convenience only; the
    /// loader `SbV2Model::from_gguf` requires the emitted GGUF metadata key
    /// unconditionally per WP-13 / FR-EX-08).
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
            // Post-M6 relative-position transformer hparams — optional
            // in the JSON side-car so a config authored before the M6
            // refactor (pre-`n_heads` / `window_size` / `kernel_ffn`
            // schema) still round-trips through the converter for the
            // SBV2 v2 base value (the defaults). A hypothetical future
            // SBV2 SKU with `n_heads = 4` etc. must supply the override
            // explicitly in its side-car.
            n_heads: root
                .get("n_heads")
                .and_then(JsonValue::as_u64)
                .map(|u| u as u32)
                .unwrap_or(DEFAULT_N_HEADS),
            window_size: root
                .get("window_size")
                .and_then(JsonValue::as_u64)
                .map(|u| u as u32)
                .unwrap_or(DEFAULT_WINDOW_SIZE),
            kernel_ffn: root
                .get("kernel_ffn")
                .and_then(JsonValue::as_u64)
                .map(|u| u as u32)
                .unwrap_or(DEFAULT_KERNEL_FFN),
            n_flow_layers: req_u32("n_flow_layers")?,
            // Blocker 2b (2026-08-06) — flow's own hparams. Optional in
            // the JSON side-car so a pre-Blocker-2b config still round-
            // trips (the defaults pin the SBV2 v2 base checkpoint's
            // values, which every existing SKU uses). A hypothetical
            // future SKU with different flow hparams overrides these by
            // supplying its own values in the side-car.
            flow_n_encoder_layers: root
                .get("flow_n_encoder_layers")
                .and_then(JsonValue::as_u64)
                .map(|u| u as u32)
                .unwrap_or(DEFAULT_FLOW_N_ENCODER_LAYERS),
            flow_kernel_ffn: root
                .get("flow_kernel_ffn")
                .and_then(JsonValue::as_u64)
                .map(|u| u as u32)
                .unwrap_or(DEFAULT_FLOW_KERNEL_FFN),
            flow_gin_channels: root
                .get("flow_gin_channels")
                .and_then(JsonValue::as_u64)
                .map(|u| u as u32)
                .unwrap_or(DEFAULT_FLOW_GIN_CHANNELS),
            flow_mean_only: root
                .get("flow_mean_only")
                .and_then(|v| match v {
                    JsonValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(DEFAULT_FLOW_MEAN_ONLY),
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
        // Post-M6 relative-position transformer hparam consistency (mirrors
        // the identical loud-fail path `SbV2Model::from_gguf` runs at
        // GGUF-load time — every check here matches one on that side).
        if cfg.n_heads == 0 || cfg.window_size == 0 || cfg.kernel_ffn == 0 {
            return Err(ConvertError::Parse(format!(
                "sbv2 config: n_heads ({}) / window_size ({}) / kernel_ffn ({}) must all be > 0",
                cfg.n_heads, cfg.window_size, cfg.kernel_ffn,
            )));
        }
        if cfg.d_model % cfg.n_heads != 0 {
            return Err(ConvertError::Parse(format!(
                "sbv2 config: d_model ({}) must be divisible by n_heads ({}) — VITS \
                 MultiHeadAttention requires d_head = d_model / n_heads to be an exact integer",
                cfg.d_model, cfg.n_heads,
            )));
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
    /// reconstructed + pass-through-verbatim + SDP-rewritten). Includes the
    /// SDP tensors that got rewritten (see [`Self::sdp_rewritten`]) since
    /// they still made it into the emitted GGUF, just under different names.
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
    /// Of the tensors in [`Self::written`], how many were rewritten from
    /// their upstream `sdp.<x>` name to the `sbv2.sdp.<remap-of-x>` name
    /// `SbV2Model::from_gguf` reads. Real SBV2 v2 base has 144 such
    /// tensors (see the module doc's tensor-count summary and
    /// [`convert_sbv2_file`]'s arm-by-arm doc); other upstream tensors
    /// (`enc_p.*` / `dec.*` / `flow.*` / ...) pass through under their
    /// original upstream names for now (a separate Blocker 2 sub-item
    /// tracks those rewriters).
    pub sdp_rewritten: usize,
    /// Number of training-side `sdp.post_*` tensors intentionally
    /// **not** emitted into the GGUF (the ~142-tensor training scaffolding
    /// that the inference-time SDP forward pass never reads — see
    /// [`convert_sbv2_file`]'s arm 1 doc). Distinct from
    /// [`Self::skipped_non_float`] so a caller can tell "unsupported
    /// dtype" apart from "intentionally-not-loaded training-side" at a
    /// glance.
    pub sdp_post_skipped: usize,
    /// Whether `config_side_car` was supplied to [`convert_sbv2_file`] and
    /// the `vokra.sbv2.*` hparam chunk (22 required + 1 optional keys) was
    /// written. `false` means tensors still passed through but
    /// `SbV2Model::from_gguf` will fail loudly on the first missing
    /// `vokra.sbv2.*` key — see this module's doc "Hparams" section for
    /// why that is preferred over inventing placeholder values.
    pub hparams_written: bool,
    /// Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS: count of `converter_zero_default`
    /// tensors emitted at the end of [`convert_sbv2_file`] to replace the
    /// pre-Wave-4 loader-side silent fallback path for optional slots
    /// (`sbv2.decoder.conv_post.bias`, `sbv2.style_injector.proj_*`,
    /// `sbv2.speaker.table`). Zero on ckpts that already ship every
    /// optional slot; up to 3 on the SBV2 v2 base ckpt which ships none.
    pub converter_zero_defaults_emitted: usize,
}

/// Rewrites the tail of an `sdp.<tail>` upstream tensor name into its
/// `sbv2.sdp.*` runtime equivalent — mirror of the schema
/// `SbV2Model::from_gguf` reads (see `crates/vokra-models/src/sbv2/mod.rs`
/// `sbv2.sdp.*` tensor-path section). Deterministic and total; every
/// `sdp.*` production tensor lands on exactly one arm below:
///
/// - `flows.0.m|logs` → `ea.m|logs` (`ElementwiseAffine` at flow slot 0)
/// - `flows.<odd>.<w>` → `flow.<(odd-1)/2>.<w>` for `odd ∈ {1,3,5,7}`
///   (dense re-index of upstream's sparse `flows` list — see
///   [`convert_sbv2_file`]'s arm 2 doc for the full derivation)
/// - anything else (`pre.<w>`, `proj.<w>`, `cond.<w>`, `convs.<x>`, ...) →
///   verbatim under `sbv2.sdp.<tail>`
///
/// Never called with a `tail` that itself starts with `post_` — the
/// caller filters those out first (`convert_sbv2_file`'s arm 1).
fn rewrite_sdp_tensor_name(tail: &str) -> String {
    // Handle the two flow-index arms.
    if let Some(rest) = tail.strip_prefix("flows.") {
        // Split `rest` at the first `.` to isolate the index.
        if let Some((idx_str, tail_after_idx)) = rest.split_once('.') {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx == 0 && (tail_after_idx == "m" || tail_after_idx == "logs") {
                    return format!("sbv2.sdp.ea.{tail_after_idx}");
                }
                if idx % 2 == 0 {
                    // Even index (2/4/6/8) is a `Flip` slot — upstream
                    // stores no learnable parameters there, so no tensor
                    // ever has this prefix in practice. Preserve
                    // verbatim under `sbv2.sdp.flows.<idx>.<tail>` so an
                    // out-of-spec safetensors is loud-detected downstream
                    // (as an unrecognised tensor in `from_gguf`) rather
                    // than silently coalesced into `flow.<mapped>.*`.
                    return format!("sbv2.sdp.flows.{idx}.{tail_after_idx}");
                }
                // Odd index (1/3/5/7) is a `ConvFlow`. Densify: 1→0, 3→1,
                // 5→2, 7→3.
                let dense = (idx - 1) / 2;
                return format!("sbv2.sdp.flow.{dense}.{tail_after_idx}");
            }
        }
    }
    // Every other `sdp.<tail>` (pre/proj/cond/convs) is a verbatim rewrite
    // — same tail, just under `sbv2.sdp.`.
    format!("sbv2.sdp.{tail}")
}

/// Converts an SBV2 v2 safetensors checkpoint at `input` into a Vokra GGUF
/// at `output`.
///
/// `config_side_car`, when `Some`, points at a JSON file supplying every
/// `vokra.sbv2.*` hparam (see `SbV2Config::parse` for the schema); when
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
/// `SbV2Config::parse`'s doc for the full list of required fields and
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
    let mut cfg = if let Some(config_path) = config_side_car {
        let config_bytes = std::fs::read(config_path)?;
        Some(SbV2Config::parse(&config_bytes)?)
    } else {
        None
    };

    // Blocker 2c follow-up (2026-08-08): shape-recover `n_sdp_layers` from
    // the actual `sdp.flows.<odd>.pre.weight` tensor set instead of trusting
    // the config side-car. Upstream `StochasticDurationPredictor.__init__(...,
    // n_flows=4)` stores 4 ConvFlows at `sdp.flows.{1, 3, 5, 7}`. The prep
    // script (`tools/parity/sbv2_prepare_checkpoint.py`) shipped a fallback
    // `n_sdp_layers = 3` — meant for the DDS-net inner depth (`n_layers_dp`)
    // but wired to the ConvFlow-count metadata slot, off by one — so before
    // this shape-recovery landed, the real base ckpt's fourth ConvFlow
    // (`sdp.flows.7.*`) was never loaded, and Rust `SbV2SDP::sample`'s
    // reverse walk had one fewer layer than upstream, producing runaway
    // durations (max=16066, sum=47914 on a 8-phoneme "テスト" test — see
    // that fn's flow-order comment for the full trace). Follows the same
    // shape-recovery pattern the workflow already uses for `d_speaker` /
    // `n_speakers` / `decoder_upsample_kernel_sizes` (see
    // `.github/workflows/parity-sbv2-real.yml`'s "shape-recover" step).
    if let Some(cfg_mut) = cfg.as_mut() {
        // Count `sdp.flows.<odd>.pre.weight` entries — the definitive
        // per-ConvFlow marker (every ConvFlow has exactly one `pre.weight`
        // tensor; even indices are `Flip` layers with no parameters, so
        // they never appear; the odd-index sparse indexing collapses to a
        // dense count via `n_flows = odd_index_count`).
        let mut sdp_flow_count = 0_u32;
        for t in st.tensors() {
            let name = t.name.as_str();
            let Some(rest) = name.strip_prefix("sdp.flows.") else {
                continue;
            };
            let Some((idx_str, tail)) = rest.split_once('.') else {
                continue;
            };
            if tail != "pre.weight" {
                continue;
            }
            let Ok(idx) = idx_str.parse::<usize>() else {
                continue;
            };
            if idx % 2 == 1 {
                sdp_flow_count += 1;
            }
        }
        if sdp_flow_count > 0 && sdp_flow_count != cfg_mut.n_sdp_layers {
            eprintln!(
                "convert_sbv2: shape-recovering vokra.sbv2.n_sdp_layers: config \
                 declared {} but the input safetensors carries {sdp_flow_count} \
                 `sdp.flows.<odd>.pre.weight` tensors — overriding to {sdp_flow_count} \
                 so all ConvFlows are loaded (upstream `StochasticDurationPredictor` \
                 `n_flows`, see docstring).",
                cfg_mut.n_sdp_layers,
            );
            cfg_mut.n_sdp_layers = sdp_flow_count;
        }

        // Wave-4 WORKFLOW-SHAPE-FIXUP (2026-08-09): shape-recover
        // `d_speaker` + `n_speakers` + `decoder_upsample_kernel_sizes`
        // from the input safetensors. This mirrors the inline-Python
        // shape-recovery `.github/workflows/parity-sbv2-real.yml`
        // previously did AFTER prep but BEFORE convert — moving it into
        // the converter itself so the recovery runs everywhere convert
        // runs (CI, local dev, offline sidecar), not only inside the CI
        // workflow. Same "recover from shape, loud-warn on config
        // disagreement" contract as `n_sdp_layers` above.
        //
        // d_speaker <- enc_p.encoder.spk_emb_linear.weight[1] (or fallback
        // sources — see `infer_d_speaker` doc).
        if let Some(new_d_speaker) = infer_d_speaker(&st)
            && new_d_speaker != cfg_mut.d_speaker
        {
            eprintln!(
                "convert_sbv2: shape-recovering vokra.sbv2.d_speaker: config \
                 declared {} but tensor shape implies {new_d_speaker} — \
                 overriding to {new_d_speaker} so downstream loaders can \
                 cross-check spk_emb_linear + emb_g dimensions correctly.",
                cfg_mut.d_speaker,
            );
            cfg_mut.d_speaker = new_d_speaker;
        }
        // n_speakers <- emb_g.weight[0]
        if let Some(new_n_speakers) = infer_n_speakers(&st)
            && new_n_speakers != cfg_mut.n_speakers
        {
            eprintln!(
                "convert_sbv2: shape-recovering vokra.sbv2.n_speakers: config \
                 declared {} but `emb_g.weight` shape implies {new_n_speakers} — \
                 overriding to {new_n_speakers}.",
                cfg_mut.n_speakers,
            );
            cfg_mut.n_speakers = new_n_speakers;
        }
        // decoder_upsample_kernel_sizes <- per-stage dec.ups.<i>.weight_v last dim
        if let Some(new_kernels) = infer_decoder_upsample_kernel_sizes(&st)
            && cfg_mut.decoder_upsample_kernel_sizes != new_kernels
        {
            eprintln!(
                "convert_sbv2: shape-recovering vokra.sbv2.decoder_upsample_kernel_sizes: \
                 config declared {:?} but `dec.ups.<i>.weight_v` shape implies {:?} — \
                 overriding so HiFi-GAN loader's per-stage kernel cross-check succeeds.",
                cfg_mut.decoder_upsample_kernel_sizes, new_kernels,
            );
            cfg_mut.decoder_upsample_kernel_sizes = new_kernels;
        }
    }

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

    // Main tensor loop — Task 30 rename table (Blocker 2b) + SDP rewriter
    // (Blocker 2c: `sdp.post_*` skip and `sdp.<x>` → `sbv2.sdp.<remap>`).
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

        // Blocker 2c: `sdp.post_*` (142 training-side inverse-flow tensors).
        // `models.py::StochasticDurationPredictor.forward(reverse=True)`
        // (upstream) walks only `self.flows` (production), not
        // `self.post_flows`, so these are pure training scaffolding.
        // Counter tracked separately from `skipped_training` so the caller
        // can tell "SDP inverse-flow skipped by design" from other
        // training-side drops. Emit a stderr line to satisfy FR-EX-08 too.
        if let Some(rest) = t.name.strip_prefix("sdp.post_") {
            let _ = rest;
            report.sdp_post_skipped += 1;
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
                // Blocker 2c: `sdp.<x>` (production) → `sbv2.sdp.<remap>`
                // via `rewrite_sdp_tensor_name`. Every other pass-through
                // tensor keeps its upstream name verbatim. The `sdp.post_*`
                // arm is already filtered above (guaranteed non-`sdp.post_*`
                // here).
                let write_name = t.name.strip_prefix("sdp.").map(rewrite_sdp_tensor_name);
                let effective_name = write_name.as_deref().unwrap_or(&t.name);
                b.add_tensor(effective_name, t.dtype, t.shape.clone(), data)?;
                report.written += 1;
                if write_name.is_some() {
                    report.sdp_rewritten += 1;
                } else {
                    report.passed_through_verbatim += 1;
                }
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

    // Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS (2026-08-09): emit explicit
    // all-zero tensors for the three optional slots that the SBV2 v2 base
    // ckpt ships as absent (see the helper's doc for the full list). The
    // pre-Wave-4 loader silently fabricated these; post-Wave-4 the
    // converter emits them explicitly with a provenance-metadata trail.
    // Idempotent: only emits slots the main tensor loop above did not
    // already write.
    emit_converter_zero_defaults(&mut b, cfg.as_ref(), &mut report)?;

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
    // Post-M6 relative-position transformer hparams (see this fn's
    // caller's config-side-car doc, and the `KEY_*` const-block comment
    // above for the primary-source rationale). Stamped every time the
    // hparam chunk group is written — the loader's cross-check in
    // `SbV2Model::from_gguf` requires all three, so all or none is
    // consistent with every other hparam in this file.
    b.add_u32(KEY_N_HEADS, cfg.n_heads);
    b.add_u32(KEY_WINDOW_SIZE, cfg.window_size);
    b.add_u32(KEY_KERNEL_FFN, cfg.kernel_ffn);
    b.add_u32(KEY_N_FLOW_LAYERS, cfg.n_flow_layers);
    // Blocker 2b (2026-08-06) — flow's own hparams. Stamped
    // unconditionally alongside `n_flow_layers` so the runtime cross-
    // check in `SbV2Model::from_gguf` (which requires all four when
    // `n_flow_layers > 0`) never sees a partial stamp. For a config
    // with `n_flow_layers = 0` the runtime skips these keys entirely,
    // so stamping the defaults costs nothing and keeps the metadata
    // shape uniform across configs — matches the "all or none" rule
    // the text encoder's own hparam group already follows.
    b.add_u32(KEY_FLOW_N_ENCODER_LAYERS, cfg.flow_n_encoder_layers);
    b.add_u32(KEY_FLOW_KERNEL_FFN, cfg.flow_kernel_ffn);
    b.add_u32(KEY_FLOW_GIN_CHANNELS, cfg.flow_gin_channels);
    b.add_bool(KEY_FLOW_MEAN_ONLY, cfg.flow_mean_only);
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

/// Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS metadata key: stamps the source
/// for `converter_zero_default` tensor emissions. When present at load
/// time, signals a Wave-4-or-later converter emitted explicit zero
/// placeholders rather than leaving the slots empty; the loader can
/// then honor its no-silent-fallback contract (a mismatch = converter
/// bug, not a legitimate absent tensor).
pub(crate) const KEY_CONVERTER_ZERO_DEFAULTS: &str = "vokra.sbv2.converter_zero_defaults";

/// Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS (2026-08-09): emits explicit
/// all-zero tensors for three optional slots that the SBV2 v2 base
/// checkpoint (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`) ships as
/// absent from the upstream safetensors:
///
/// (a) `sbv2.decoder.conv_post.bias` — upstream `dec.conv_post` has
///     `bias=False`; a `[1]` zero bias is the identity behavior the
///     Rust HiFi-GAN decoder expects.
/// (b) `sbv2.style_injector.proj_{scale,bias}` — style is learned per
///     fine-tune; base ckpt inference is an all-zero identity injector.
/// (c) `sbv2.speaker.table` — base uses per-utterance external speaker
///     vector via `enc_p.encoder.spk_emb_linear`; the table slot is a
///     shape-valid placeholder never reached at request time.
///
/// The pre-Wave-4 loader silently fabricated these three tensors when
/// absent from the GGUF. Post-Wave-4 the converter emits them
/// explicitly with a provenance-metadata trail
/// ([`KEY_CONVERTER_ZERO_DEFAULTS`] = comma-separated list of slots)
/// so the loader's no-silent-fallback contract stays honest.
///
/// Idempotent: does not emit a slot that any earlier converter arm
/// already wrote (the main tensor loop's `Rename` / `PassThrough`
/// arms take precedence via [`GgufBuilder::has_tensor`]).
fn emit_converter_zero_defaults(
    b: &mut GgufBuilder,
    cfg: Option<&SbV2Config>,
    report: &mut ConvertReport,
) -> Result<(), ConvertError> {
    // Only emit when we have a config — need `d_model` / `d_style` /
    // `d_speaker` for shape. Without a config the loader would fail
    // loudly on the missing hparam chunk long before reaching a tensor
    // absence anyway.
    let Some(cfg) = cfg else { return Ok(()) };
    let d_model = cfg.d_model as usize;
    let d_style = cfg.d_style as usize;
    let d_speaker = cfg.d_speaker as usize;

    let mut emitted: Vec<&'static str> = Vec::new();

    // (a) sbv2.decoder.conv_post.bias — [out_channels=1] zero.
    if !b.has_tensor("sbv2.decoder.conv_post.bias") {
        let bias = [0.0_f32; 1];
        let bytes: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
        b.add_tensor("sbv2.decoder.conv_post.bias", GgmlType::F32, vec![1], bytes)?;
        emitted.push("sbv2.decoder.conv_post.bias");
        report.written += 1;
        report.converter_zero_defaults_emitted += 1;
    }

    // (b) sbv2.style_injector.proj_scale + proj_bias — [d_model, d_style]
    // zero. Only emitted when BOTH are absent (all-or-nothing per the
    // loader's FR-EX-08 contract).
    let scale_present = b.has_tensor("sbv2.style_injector.proj_scale");
    let bias_present = b.has_tensor("sbv2.style_injector.proj_bias");
    if !scale_present && !bias_present && d_style > 0 && d_model > 0 {
        let n = d_model * d_style;
        let zeros = vec![0.0_f32; n];
        let bytes: Vec<u8> = zeros.iter().flat_map(|v| v.to_le_bytes()).collect();
        b.add_tensor(
            "sbv2.style_injector.proj_scale",
            GgmlType::F32,
            vec![d_model as u64, d_style as u64],
            bytes.clone(),
        )?;
        b.add_tensor(
            "sbv2.style_injector.proj_bias",
            GgmlType::F32,
            vec![d_model as u64, d_style as u64],
            bytes,
        )?;
        emitted.push("sbv2.style_injector.proj_scale");
        emitted.push("sbv2.style_injector.proj_bias");
        report.written += 2;
        report.converter_zero_defaults_emitted += 2;
    }

    // (c) sbv2.speaker.table — [1, d_speaker] shape-valid placeholder
    // that is never reached at request time (synthesize dispatches
    // through the projection path when `spk_emb_linear` is present).
    if !b.has_tensor("sbv2.speaker.table") && d_speaker > 0 {
        let zeros = vec![0.0_f32; d_speaker];
        let bytes: Vec<u8> = zeros.iter().flat_map(|v| v.to_le_bytes()).collect();
        b.add_tensor(
            "sbv2.speaker.table",
            GgmlType::F32,
            vec![1, d_speaker as u64],
            bytes,
        )?;
        emitted.push("sbv2.speaker.table");
        report.written += 1;
        report.converter_zero_defaults_emitted += 1;
    }

    // Emit the provenance trail iff at least one slot was zero-defaulted.
    // Comma-separated list of full tensor names so a future audit can
    // reconstruct exactly what the converter fabricated.
    if !emitted.is_empty() {
        let trail = emitted.join(",");
        b.add_string(KEY_CONVERTER_ZERO_DEFAULTS, &trail);
        eprintln!(
            "convert_sbv2: emitted {} converter_zero_default tensor(s): {trail}",
            emitted.len()
        );
    }

    Ok(())
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

/// Post-M6 (2026-08-06) `enc_p.encoder.*` → `sbv2.text_encoder.layer.<i>.*`
/// rename helper. Returns `Some(new_name)` for every per-layer attn /
/// ffn / norm tensor and `None` for anything under `enc_p.encoder.*`
/// that doesn't match the per-layer pattern (e.g. `spk_emb_linear` —
/// Blocker 3 external speaker-vector projection, which stays pass-
/// through until the Rust API gains a `512-d input path`).
///
/// `rest` is the substring **after** stripping `enc_p.encoder.`; e.g.
/// upstream `enc_p.encoder.attn_layers.0.conv_q.weight` → `rest =
/// "attn_layers.0.conv_q.weight"` → returned rename
/// `"sbv2.text_encoder.layer.0.attn.conv_q.weight"`.
///
/// The four upstream sub-namespaces this handles map to the four SBV2
/// per-layer sub-modules the M6 refactor gave `SbV2TransformerBlock`:
///
/// - `attn_layers.<i>.conv_{q,k,v,o}.{weight,bias}` → `layer.<i>.attn.conv_{q,k,v,o}.{weight,bias}`
/// - `attn_layers.<i>.emb_rel_{k,v}` → `layer.<i>.attn.rel_pos_{k,v}`
/// - `ffn_layers.<i>.conv_{1,2}.{weight,bias}` → `layer.<i>.ffn.conv_{1,2}.{weight,bias}`
/// - `norm_layers_1.<i>.{gamma,beta}` → `layer.<i>.norm1.{gamma,beta}`
/// - `norm_layers_2.<i>.{gamma,beta}` → `layer.<i>.norm2.{gamma,beta}`
/// Blocker 2b (2026-08-06) — `flow.flows.<2i>.*` (upstream
/// `TransformerCouplingLayer` blocks; upstream indices 0/2/4/6 are the
/// parameterized couplings, 1/3/5/7 are parameter-free `Flip` modules
/// with no tensors) → `sbv2.flow.layer.<i>.*` (target indices 0/1/2/3).
/// Returns `Some(target_name)` for every per-block tensor and `None` for
/// odd upstream indices (Flip has no tensors, so its slot never emits an
/// upstream `flow.flows.<odd>.*` name — if one shows up it falls through
/// to the pass-through arm with the "flow VITS2" reason for post-mortem
/// visibility rather than being silently dropped).
///
/// The seven per-block upstream sub-namespaces this handles map to the
/// per-block target names `SbV2Model::from_gguf` reads:
///
/// - `pre.{weight,bias}` → `layer.<i>.pre.{weight,bias}`
/// - `post.{weight,bias}` → `layer.<i>.post.{weight,bias}`
/// - `enc.spk_emb_linear.{weight,bias}` → `layer.<i>.spk_emb.{weight,bias}`
/// - `enc.attn_layers.<j>.conv_{q,k,v,o}.{weight,bias}` → `layer.<i>.enc.<j>.attn.conv_{q,k,v,o}.{weight,bias}`
/// - `enc.attn_layers.<j>.emb_rel_{k,v}` → `layer.<i>.enc.<j>.attn.rel_pos_{k,v}`
/// - `enc.ffn_layers.<j>.conv_{1,2}.{weight,bias}` → `layer.<i>.enc.<j>.ffn.conv_{1,2}.{weight,bias}`
/// - `enc.norm_layers_{1,2}.<j>.{gamma,beta}` → `layer.<i>.enc.<j>.norm{1,2}.{gamma,beta}`
///
/// The upstream index divide-by-2 (`0/2/4/6 → 0/1/2/3`) mirrors upstream
/// `TransformerCouplingBlock.__init__` at `n_flows = 4`, which stores 8
/// entries in `nn.ModuleList` alternating coupling and Flip; the target
/// side counts couplings only, and `SbV2Model::from_gguf` interleaves
/// `FlowLayer::Flip` after each loaded coupling.
fn classify_flow_block_tensor(rest: &str) -> Option<String> {
    // rest = e.g. "0.pre.weight" or "2.enc.attn_layers.3.conv_q.weight".
    let (idx_str, tail) = rest.split_once('.')?;
    let upstream_i: usize = idx_str.parse().ok()?;
    // Only even upstream indices are TransformerCouplingLayer blocks;
    // odd indices are parameter-free `Flip` slots with no tensors.
    if upstream_i % 2 != 0 {
        return None;
    }
    let target_i = upstream_i / 2;

    // pre / post — 1×1 Conv1d each.
    if let Some(sub) = tail.strip_prefix("pre.") {
        let mapped = match sub {
            "weight" => "pre.weight",
            "bias" => "pre.bias",
            _ => return None,
        };
        return Some(format!("sbv2.flow.layer.{target_i}.{mapped}"));
    }
    if let Some(sub) = tail.strip_prefix("post.") {
        let mapped = match sub {
            "weight" => "post.weight",
            "bias" => "post.bias",
            _ => return None,
        };
        return Some(format!("sbv2.flow.layer.{target_i}.{mapped}"));
    }

    // enc.spk_emb_linear.{weight,bias} → layer.<i>.spk_emb.{weight,bias}
    // (target name shortened to `spk_emb` — matches the loader's field
    // naming; upstream's fuller `spk_emb_linear` says the same thing).
    if let Some(sub) = tail.strip_prefix("enc.spk_emb_linear.") {
        let mapped = match sub {
            "weight" => "spk_emb.weight",
            "bias" => "spk_emb.bias",
            _ => return None,
        };
        return Some(format!("sbv2.flow.layer.{target_i}.{mapped}"));
    }

    // enc.attn_layers.<j>.* — reuse the text encoder's attn tail mapping
    // literally (same sub-modules, same target-tail naming), just under
    // the flow's `layer.<i>.enc.<j>` path instead of `layer.<j>` directly.
    if let Some(after) = tail.strip_prefix("enc.attn_layers.") {
        let (j_str, sub) = after.split_once('.')?;
        let j: usize = j_str.parse().ok()?;
        let mapped_tail = match sub {
            "conv_q.weight" => "attn.conv_q.weight",
            "conv_q.bias" => "attn.conv_q.bias",
            "conv_k.weight" => "attn.conv_k.weight",
            "conv_k.bias" => "attn.conv_k.bias",
            "conv_v.weight" => "attn.conv_v.weight",
            "conv_v.bias" => "attn.conv_v.bias",
            "conv_o.weight" => "attn.conv_o.weight",
            "conv_o.bias" => "attn.conv_o.bias",
            "emb_rel_k" => "attn.rel_pos_k",
            "emb_rel_v" => "attn.rel_pos_v",
            _ => return None,
        };
        return Some(format!("sbv2.flow.layer.{target_i}.enc.{j}.{mapped_tail}"));
    }

    // enc.ffn_layers.<j>.conv_{1,2}.{weight,bias}
    if let Some(after) = tail.strip_prefix("enc.ffn_layers.") {
        let (j_str, sub) = after.split_once('.')?;
        let j: usize = j_str.parse().ok()?;
        let mapped_tail = match sub {
            "conv_1.weight" => "ffn.conv_1.weight",
            "conv_1.bias" => "ffn.conv_1.bias",
            "conv_2.weight" => "ffn.conv_2.weight",
            "conv_2.bias" => "ffn.conv_2.bias",
            _ => return None,
        };
        return Some(format!("sbv2.flow.layer.{target_i}.enc.{j}.{mapped_tail}"));
    }

    // enc.norm_layers_{1,2}.<j>.{gamma,beta}
    for (upstream_prefix, dst_prefix) in [
        ("enc.norm_layers_1.", "norm1"),
        ("enc.norm_layers_2.", "norm2"),
    ] {
        if let Some(after) = tail.strip_prefix(upstream_prefix) {
            let (j_str, sub) = after.split_once('.')?;
            let j: usize = j_str.parse().ok()?;
            let mapped_tail = match sub {
                "gamma" => format!("{dst_prefix}.gamma"),
                "beta" => format!("{dst_prefix}.beta"),
                _ => return None,
            };
            return Some(format!("sbv2.flow.layer.{target_i}.enc.{j}.{mapped_tail}"));
        }
    }
    None
}

fn classify_encoder_layer_tensor(rest: &str) -> Option<String> {
    // attn_layers
    if let Some(after) = rest.strip_prefix("attn_layers.") {
        let (idx_str, tail) = after.split_once('.')?;
        let i: usize = idx_str.parse().ok()?;
        let mapped_tail = match tail {
            "conv_q.weight" => "attn.conv_q.weight",
            "conv_q.bias" => "attn.conv_q.bias",
            "conv_k.weight" => "attn.conv_k.weight",
            "conv_k.bias" => "attn.conv_k.bias",
            "conv_v.weight" => "attn.conv_v.weight",
            "conv_v.bias" => "attn.conv_v.bias",
            "conv_o.weight" => "attn.conv_o.weight",
            "conv_o.bias" => "attn.conv_o.bias",
            "emb_rel_k" => "attn.rel_pos_k",
            "emb_rel_v" => "attn.rel_pos_v",
            _ => return None,
        };
        return Some(format!("sbv2.text_encoder.layer.{i}.{mapped_tail}"));
    }
    // ffn_layers
    if let Some(after) = rest.strip_prefix("ffn_layers.") {
        let (idx_str, tail) = after.split_once('.')?;
        let i: usize = idx_str.parse().ok()?;
        let mapped_tail = match tail {
            "conv_1.weight" => "ffn.conv_1.weight",
            "conv_1.bias" => "ffn.conv_1.bias",
            "conv_2.weight" => "ffn.conv_2.weight",
            "conv_2.bias" => "ffn.conv_2.bias",
            _ => return None,
        };
        return Some(format!("sbv2.text_encoder.layer.{i}.{mapped_tail}"));
    }
    // norm_layers_1 / norm_layers_2
    for (upstream_prefix, dst_prefix) in [("norm_layers_1.", "norm1"), ("norm_layers_2.", "norm2")]
    {
        if let Some(after) = rest.strip_prefix(upstream_prefix) {
            let (idx_str, tail) = after.split_once('.')?;
            let i: usize = idx_str.parse().ok()?;
            let mapped_tail = match tail {
                "gamma" => format!("{dst_prefix}.gamma"),
                "beta" => format!("{dst_prefix}.beta"),
                _ => return None,
            };
            return Some(format!("sbv2.text_encoder.layer.{i}.{mapped_tail}"));
        }
    }
    None
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
        // Blocker 3: external speaker projection (`spk_emb_linear [192, 512]`)
        // → renamed to the Rust loader's expected name so
        // `SbV2Model::from_gguf` can construct an ExternalSpeakerProjection
        // from it. The base checkpoint carries no `emb_g` table; this is
        // the entire speaker-conditioning input path.
        "enc_p.encoder.spk_emb_linear.weight" => {
            return TensorClass::Rename("sbv2.text_encoder.spk_emb_linear.weight".into());
        }
        "enc_p.encoder.spk_emb_linear.bias" => {
            return TensorClass::Rename("sbv2.text_encoder.spk_emb_linear.bias".into());
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
        // Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS (2026-08-09): fine-tune SKUs
        // that DO ship `dec.conv_post.bias` (upstream base ckpt does not)
        // must land at the canonical Vokra name so the loader consumes
        // the real value, not the converter's zero-default fallback.
        // Pre-Wave-4 this passed through under its upstream name and the
        // loader's silent-fabrication path swallowed the difference.
        "dec.conv_post.bias" => {
            return TensorClass::Rename("sbv2.decoder.conv_post.bias".into());
        }
        // HGAN-05-GIN-COND (2026-08-09): rename the decoder's
        // speaker-conditioning cond layer. Upstream `dec.cond` is
        // `Conv1d(gin_channels, initial_channel, 1)` — present on
        // SBV2 v2 multi-speaker base ckpt (`gin_channels = 512`),
        // absent on single-speaker fixtures. Pre-HGAN-05 these were
        // PassThrough with the reason "no Rust HifiGanAttrs field
        // yet"; now the runtime loader binds them into the
        // `HifiGanWeights::cond` slot and drives upstream's `x = x +
        // self.cond(g)` broadcast-add.
        "dec.cond.weight" => {
            return TensorClass::Rename("sbv2.decoder.cond.weight".into());
        }
        "dec.cond.bias" => {
            return TensorClass::Rename("sbv2.decoder.cond.bias".into());
        }
        _ => {}
    }

    // --------------------------------------------------------------
    // 2b) enc_p.encoder.{attn_layers|ffn_layers|norm_layers_1|norm_layers_2}.<i>.*
    //     (Post-M6 relative-position transformer stack — the M6 refactor
    //     wired the Rust runtime to consume these under the SBV2 layer
    //     path convention `sbv2.text_encoder.layer.<i>.{attn|ffn|norm1|
    //     norm2}.*`. See the module doc's "M6 refactor" section for the
    //     primary-source rationale and the tensor-shape trail.)
    // --------------------------------------------------------------
    if let Some(rest) = name.strip_prefix("enc_p.encoder.") {
        if let Some(remapped) = classify_encoder_layer_tensor(rest) {
            return TensorClass::Rename(remapped);
        }
        // `enc_p.encoder.spk_emb_linear.{weight,bias}` and any other
        // `enc_p.encoder.*` that is not part of the per-layer attn/ffn/
        // norm stack (Blocker 3 — external speaker vector projection)
        // stays on the pass-through path below rather than being renamed
        // here, keeping the "fall through to pass-through with a
        // per-family reason" convention this file's classification
        // relies on.
    }

    // --------------------------------------------------------------
    // 2c) flow.flows.<2i>.* → sbv2.flow.layer.<i>.* (Blocker 2b,
    //     2026-08-06). Upstream `TransformerCouplingBlock` stores 8
    //     entries alternating `TransformerCouplingLayer` (even indices
    //     0/2/4/6) and parameter-free `Flip` (odd indices 1/3/5/7). We
    //     rename only the parameterized-coupling tensors — `Flip` has no
    //     tensors — and `SbV2Model::from_gguf` reconstructs the
    //     alternating `[TCL, Flip, ...]` layout at load time.
    // --------------------------------------------------------------
    if let Some(rest) = name.strip_prefix("flow.flows.") {
        if let Some(remapped) = classify_flow_block_tensor(rest) {
            return TensorClass::Rename(remapped);
        }
        // A `flow.flows.<odd>.*` (an unexpected Flip-slot tensor — none
        // exist on the base checkpoint) or any other `flow.flows.*` that
        // doesn't match the seven per-block sub-namespaces falls through
        // to the pass-through arm below with the "flow VITS2" reason so
        // no data is silently dropped and a future audit surfaces it.
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
                    // Wave-2 HGAN-01 fix (2026-08-09): pre-fix this
                    // arm was `PassThrough` because the Rust runtime
                    // had no `weight_c2` / `bias_c2` slot to bind
                    // convs2 into. `mrf_branch_forward` now runs the
                    // upstream `for (c1, c2) in zip(convs1, convs2)`
                    // chain (V1 topology), and the from_gguf loader
                    // requires both convs1 + convs2 tensors — a
                    // converter that continued to drop convs2 would
                    // fail the loader's FR-EX-08 gate loudly.
                    let n_branches = match n_resblock_branches {
                        Some(n) if n > 0 => n,
                        _ => {
                            // Without a config side-car we cannot
                            // honestly split the flat index — preserve
                            // verbatim so no data is dropped, and the
                            // downstream loader's shape check surfaces
                            // the config gap.
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
                                sibling_v: format!("dec.resblocks.{flat_i}.convs2.{j}.weight_v"),
                                target_name: format!("{target_base}.weight_c2"),
                            };
                        }
                        "weight_v" => {
                            return TensorClass::PassThrough {
                                reason: "orphan dec.resblocks convs2 weight_v (weight_g missing)",
                            };
                        }
                        "bias" => {
                            return TensorClass::Rename(format!("{target_base}.bias_c2"));
                        }
                        "weight" => {
                            return TensorClass::Rename(format!("{target_base}.weight_c2"));
                        }
                        _ => {}
                    }
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
        // Post-Blocker-2b (2026-08-06) — the per-block `flow.flows.<2i>.*`
        // tensors are now renamed via `classify_flow_block_tensor` above
        // and never reach this pass-through arm. What lands here is any
        // `flow.*` that doesn't match the `flow.flows.<even>.` pattern
        // (currently: nothing on the real base checkpoint; a future SKU
        // might add auxiliary tensors here and this reason keeps them
        // preserved for a follow-up wave rather than silently dropped).
        "SBV2 flow auxiliary tensor — no matching per-block rename in \
         classify_flow_block_tensor (preserved verbatim for a follow-up wave)"
    } else if name.starts_with("enc_p.encoder.spk_emb_linear") {
        "external speaker-vector projection — Blocker 3 (Rust API needs 512-d input path)"
    } else if name.starts_with("enc_p.encoder.") {
        // Post-M6 (2026-08-06) — the per-layer attn / ffn / norm tensors
        // are now renamed via `classify_encoder_layer_tensor` above and
        // never reach this pass-through arm. What lands here is anything
        // else under `enc_p.encoder.*` (currently: nothing on the real
        // base checkpoint; a future SKU might add auxiliary tensors here
        // and this reason keeps them preserved for a follow-up wave
        // rather than silently dropped).
        "SBV2 encoder auxiliary tensor — no matching per-layer rename in \
         classify_encoder_layer_tensor (preserved verbatim for a follow-up wave)"
    } else if name.starts_with("enc_p.proj.") {
        // SBV2-INFO-01-ENC-P-PROJ (2026-08-09): upstream VITS prior head
        // projects the text_encoder output to `(mu, log_sigma)` — the
        // prior mean and log-standard-deviation used by the FLOW-NOISE-
        // SCALE reparameterization (`z_p = mu + torch.randn * exp(logs)
        // * noise_scale`). The Rust scaffold currently treats
        // `mel_hidden` as the mean directly and `logs = 0`
        // (`exp(0) = 1`), which is arithmetically equivalent when
        // upstream's `enc_p.proj` weights are near-identity for `mu`
        // and near-zero for `log_sigma`. Real fine-tune ckpts may
        // ship `enc_p.proj` weights that meaningfully diverge from
        // that assumption.
        //
        // Preserving verbatim means a future SbV2 wave can implement
        // `PriorHead::from_gguf` + `mel_hidden.mean, mel_hidden.logstd
        // = prior_head(text_hidden)` without a re-conversion. Owner
        // decision is required to (a) transcribe upstream forward
        // topology under a licensing-cleared source, or (b) trace a
        // real fine-tune ckpt's arithmetic through the AGPL upstream,
        // or (c) accept the current mean=mel_hidden approximation as
        // permanent (base-ckpt agnostic behavior is unchanged by
        // either choice).
        "VITS output projection to (mu, log_sigma) — no Rust text_encoder field yet \
         (SBV2-INFO-01-ENC-P-PROJ: owner decision pending on prior-head implementation)"
    } else if name.starts_with("sdp.") {
        "production SBV2 SDP path (DDS-net + rational-quadratic-spline ConvFlow) — Rust \
         duration.rs simplified"
    } else if name.starts_with("dec.cond.") {
        // HGAN-05-GIN-COND (2026-08-09): `dec.cond.weight` and
        // `dec.cond.bias` are Rename-mapped above (see `dec.cond.*`
        // arms in the earlier `match`), so this arm should be
        // unreachable in practice. A tensor named `dec.cond.<other>`
        // (unknown sub-name a future SBV2 SKU might add) still lands
        // here rather than being silently dropped.
        "decoder speaker conditioning — auxiliary sub-tensor beyond the HGAN-05 loader's \
         .weight / .bias slots (preserved verbatim for a follow-up wave)"
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
        // Blocker 2b (2026-08-06) — flow's own hparams. Stamped every
        // time the hparam chunk group is written (see `write_hparams`),
        // with defaults matching the SBV2 v2 base checkpoint's real
        // per-block flow tensor shapes.
        assert_eq!(get_u32(KEY_FLOW_N_ENCODER_LAYERS), 6);
        assert_eq!(get_u32(KEY_FLOW_KERNEL_FFN), 5);
        assert_eq!(get_u32(KEY_FLOW_GIN_CHANNELS), 512);
        let mean_only = match file.get(KEY_FLOW_MEAN_ONLY) {
            Some(GgufMetadataValue::Bool(b)) => *b,
            other => panic!("flow.mean_only: unexpected {other:?}"),
        };
        assert!(
            mean_only,
            "flow.mean_only default must be true (SBV2 v2 base)"
        );
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

    /// Blocker 2b TDD-hardening (2026-08-10) — converter-side flow-key
    /// spelling-contract pin. The existing
    /// `hparams_written_and_round_trip_with_config_side_car` reads back
    /// the four flow-hparam keys through the module-private `KEY_FLOW_*`
    /// constants; if any of those constants has a typo (say,
    /// `"vokra.sbv2.flow.n_encoder_layer"` missing the trailing `s`),
    /// both the write path AND the assertion path go through the same
    /// buggy constant and the test wrongly passes.
    ///
    /// This test asserts each flow-hparam key is present in the emitted
    /// GGUF under its HARDCODED spelling — the strings live only here,
    /// not through any converter-side constant. A converter typo now
    /// surfaces as a missing-key panic here (this test fails) rather
    /// than as a runtime "tensor not found" or an arithmetic-wrong
    /// output downstream (per FR-EX-08).
    ///
    /// The four hardcoded strings here MUST match the four hardcoded
    /// strings in the loader's `require_u32` / `.and_then(|v| v.as_bool())`
    /// reads (`crates/vokra-models/src/sbv2/mod.rs` lines 2341-2352) and
    /// the four hardcoded strings pinned by the corresponding
    /// `from_gguf_positive_n_flow_layers_missing_flow_*_fails_loudly`
    /// tests in `crates/vokra-models/tests/sbv2_gguf_loader.rs`. Any
    /// drift among the three sites breaks the flow-hparam contract.
    #[test]
    fn write_hparams_stamps_flow_keys_under_exact_hardcoded_string_spellings() {
        let blob = safetensors_multi(&base_fixture());
        let input = temp_path("spelling-contract-in", "safetensors");
        let config = temp_path("spelling-contract-cfg", "json");
        let output = temp_path("spelling-contract-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        std::fs::write(&config, valid_config_json()).expect("write config");

        let report = convert_sbv2_file(&input, &output, Some(&config), None).expect("convert");
        assert!(report.hparams_written);

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        // Hardcoded strings — the "external ground truth" for the
        // converter/loader spelling contract.
        for u32_key in [
            "vokra.sbv2.flow.n_encoder_layers",
            "vokra.sbv2.flow.kernel_ffn",
            "vokra.sbv2.flow.gin_channels",
        ] {
            let val = file.get(u32_key).unwrap_or_else(|| {
                panic!(
                    "converter must stamp `{u32_key}` — a typo in one of the `KEY_FLOW_*` \
                     constants (`crates/vokra-convert/src/models/sbv2.rs` lines 378-381) \
                     would land here as a missing key. The loader (`crates/vokra-models/src/\
                     sbv2/mod.rs`) reads this exact spelled string."
                )
            });
            assert!(
                val.as_u64().is_some(),
                "`{u32_key}` must be a UINT32-typed value in the emitted GGUF"
            );
        }
        // `mean_only` is a `bool`, distinct type — hand-check.
        match file.get("vokra.sbv2.flow.mean_only") {
            Some(GgufMetadataValue::Bool(_)) => (),
            Some(other) => panic!(
                "`vokra.sbv2.flow.mean_only` must be BOOL in the emitted GGUF, got {other:?}"
            ),
            None => panic!(
                "converter must stamp `vokra.sbv2.flow.mean_only` — a typo in the \
                 `KEY_FLOW_MEAN_ONLY` constant would land here as a missing key. The \
                 loader reads this exact spelled string."
            ),
        }

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

    // ---- Blocker 2c: sdp.* rewriter + sdp.post_* skip ---------------------

    #[test]
    fn sdp_tensor_name_rewriter_maps_every_arm() {
        // Body components → verbatim tail under `sbv2.sdp.`.
        assert_eq!(rewrite_sdp_tensor_name("pre.weight"), "sbv2.sdp.pre.weight");
        assert_eq!(rewrite_sdp_tensor_name("pre.bias"), "sbv2.sdp.pre.bias");
        assert_eq!(
            rewrite_sdp_tensor_name("proj.weight"),
            "sbv2.sdp.proj.weight"
        );
        assert_eq!(
            rewrite_sdp_tensor_name("cond.weight"),
            "sbv2.sdp.cond.weight"
        );
        assert_eq!(rewrite_sdp_tensor_name("cond.bias"), "sbv2.sdp.cond.bias");
        assert_eq!(
            rewrite_sdp_tensor_name("convs.convs_sep.0.weight"),
            "sbv2.sdp.convs.convs_sep.0.weight"
        );
        assert_eq!(
            rewrite_sdp_tensor_name("convs.norms_2.2.beta"),
            "sbv2.sdp.convs.norms_2.2.beta"
        );

        // Flow slot 0 = ElementwiseAffine.
        assert_eq!(rewrite_sdp_tensor_name("flows.0.m"), "sbv2.sdp.ea.m");
        assert_eq!(rewrite_sdp_tensor_name("flows.0.logs"), "sbv2.sdp.ea.logs");

        // Flow slots 1/3/5/7 = ConvFlow — densified to 0/1/2/3.
        assert_eq!(
            rewrite_sdp_tensor_name("flows.1.pre.weight"),
            "sbv2.sdp.flow.0.pre.weight"
        );
        assert_eq!(
            rewrite_sdp_tensor_name("flows.3.convs.convs_1x1.0.weight"),
            "sbv2.sdp.flow.1.convs.convs_1x1.0.weight"
        );
        assert_eq!(
            rewrite_sdp_tensor_name("flows.5.proj.bias"),
            "sbv2.sdp.flow.2.proj.bias"
        );
        assert_eq!(
            rewrite_sdp_tensor_name("flows.7.pre.bias"),
            "sbv2.sdp.flow.3.pre.bias"
        );

        // Even indices (Flip slots; no learnable params upstream, but if
        // one appeared it must land somewhere loud, not silently coalesced
        // into `flow.<x>.*`).
        assert_eq!(
            rewrite_sdp_tensor_name("flows.2.something"),
            "sbv2.sdp.flows.2.something"
        );
        assert_eq!(
            rewrite_sdp_tensor_name("flows.4.other"),
            "sbv2.sdp.flows.4.other"
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
    fn classify_dec_resblocks_convs2_renames_when_config_available() {
        // Wave-2 HGAN-01 (2026-08-09): pre-fix this arm was
        // PassThrough because the Rust runtime had no `weight_c2` slot.
        // Now `mrf_branch_forward` runs the full V1 `convs1 + convs2`
        // chain and the from_gguf loader demands both — the converter
        // must emit convs2.* under the sibling `sbv2.decoder.mrf.<s>.<b>.layer.<l>.weight_c2 / bias_c2`
        // names so the loader can bind them.
        //
        // Without a config (cannot do stage/branch math) it still falls
        // back to PassThrough so no data is silently lost — the
        // downstream loader will then loud-fail on the missing weight_c2
        // tensor, which is the honest FR-EX-08 outcome.
        let no_cfg = classify_tensor("dec.resblocks.0.convs2.0.weight_g", None);
        assert!(
            matches!(no_cfg, TensorClass::PassThrough { .. }),
            "convs2 without config must PassThrough (config-derivable field), got {no_cfg:?}"
        );

        // With n_branches=3 and flat_i=0 → (stage=0, branch=0).
        assert_eq!(
            classify_tensor("dec.resblocks.0.convs2.0.weight_g", Some(3)),
            TensorClass::WeightNorm {
                sibling_v: "dec.resblocks.0.convs2.0.weight_v".into(),
                target_name: "sbv2.decoder.mrf.0.0.layer.0.weight_c2".into(),
            }
        );
        // convs2 bias also remaps to `bias_c2` (distinct from convs1's
        // `bias` — the loader has two named slots per layer).
        assert_eq!(
            classify_tensor("dec.resblocks.7.convs2.0.bias", Some(3)),
            TensorClass::Rename("sbv2.decoder.mrf.2.1.layer.0.bias_c2".into())
        );
        // Bare weight (post-weight-norm collapsed) also renames.
        assert_eq!(
            classify_tensor("dec.resblocks.14.convs2.1.weight", Some(3)),
            TensorClass::Rename("sbv2.decoder.mrf.4.2.layer.1.weight_c2".into())
        );
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
        // Post-Blocker-2b/3 (2026-08-06): the VITS2 flow's per-block
        // `flow.flows.<even>.*` tensors are now Rename-mapped, and
        // `spk_emb_linear` (Blocker 3) is renamed to
        // `sbv2.text_encoder.spk_emb_linear.*` for the
        // `ExternalSpeakerProjection` loader path — see
        // `classify_encoder_spk_emb_linear_now_renamed_blocker3`. What
        // still passes through with a per-family reason: `enc_p.proj.*`
        // (VITS output projection to (mu, log_sigma), no Rust
        // text_encoder field yet — tracked as SBV2-INFO-01-ENC-P-PROJ).
        //
        // HGAN-05-GIN-COND (2026-08-09): `dec.cond.{weight,bias}` moved
        // OUT of this PassThrough list and INTO Rename because the
        // runtime `HifiGanWeights::cond` slot now consumes them. See
        // `classify_dec_cond_now_renamed_hgan05` below.
        {
            // Wave-4 (2026-08-09): unrolled from a single-element `for` to
            // satisfy clippy `single_element_loop -D warnings`. Kept as a
            // block so the list can grow back if HGAN-05 gets follow-up
            // "still-PassThrough" names.
            let name = "enc_p.proj.weight";
            assert!(
                matches!(
                    classify_tensor(name, Some(3)),
                    TensorClass::PassThrough { .. }
                ),
                "{name} must be PassThrough"
            );
        }
    }

    // HGAN-05-GIN-COND regression pin (2026-08-09): `dec.cond.*` are
    // now consumed by the runtime via `HifiGanWeights::cond`. A
    // converter that dropped the Rename entries would leave those
    // tensors under upstream names and the loader's shape check
    // would fail with "tensor not found".
    #[test]
    fn classify_dec_cond_now_renamed_hgan05() {
        assert_eq!(
            classify_tensor("dec.cond.weight", Some(3)),
            TensorClass::Rename("sbv2.decoder.cond.weight".into())
        );
        assert_eq!(
            classify_tensor("dec.cond.bias", Some(3)),
            TensorClass::Rename("sbv2.decoder.cond.bias".into())
        );
    }

    // -----------------------------------------------------------------
    // Blocker 2b (2026-08-06): flow block per-family rename tests
    // -----------------------------------------------------------------
    //
    // Each entry pins one upstream `flow.flows.<2i>.*` tensor to its
    // target GGUF name under `sbv2.flow.layer.<i>.*`. A stale converter
    // that drops the Blocker 2b rename table would silently leave
    // upstream names on disk and the Rust loader would then fail with
    // "tensor not found". The 4 real upstream indices (0, 2, 4, 6) map
    // to target indices (0, 1, 2, 3) via divide-by-2 — see
    // `classify_flow_block_tensor`'s doc for the upstream layout.

    #[test]
    fn classify_flow_pre_and_post_rename_by_block_index_halving() {
        for (upstream_i, target_i) in [(0usize, 0usize), (2, 1), (4, 2), (6, 3)] {
            for (tail, mapped) in [
                ("pre.weight", "pre.weight"),
                ("pre.bias", "pre.bias"),
                ("post.weight", "post.weight"),
                ("post.bias", "post.bias"),
            ] {
                let input = format!("flow.flows.{upstream_i}.{tail}");
                let expected = format!("sbv2.flow.layer.{target_i}.{mapped}");
                assert_eq!(
                    classify_tensor(&input, Some(3)),
                    TensorClass::Rename(expected.clone()),
                    "{input} must rename to {expected}"
                );
            }
        }
    }

    #[test]
    fn classify_flow_spk_emb_linear_renames_to_spk_emb() {
        for upstream_i in [0usize, 2, 4, 6] {
            let target_i = upstream_i / 2;
            for (tail, mapped) in [
                ("enc.spk_emb_linear.weight", "spk_emb.weight"),
                ("enc.spk_emb_linear.bias", "spk_emb.bias"),
            ] {
                let input = format!("flow.flows.{upstream_i}.{tail}");
                let expected = format!("sbv2.flow.layer.{target_i}.{mapped}");
                assert_eq!(
                    classify_tensor(&input, Some(3)),
                    TensorClass::Rename(expected),
                );
            }
        }
    }

    #[test]
    fn classify_flow_encoder_attn_ffn_norm_all_rename() {
        // Per-block encoder-layer sub-namespace rename. Uses
        // upstream_i = 0 (target_i = 0) and encoder-layer j = 0, 3 —
        // enough to pin the numeric substitution without exhausting
        // every combination.
        let upstream_i = 0usize;
        for j in [0usize, 3, 5] {
            for (tail_in, tail_out) in [
                ("conv_q.weight", "attn.conv_q.weight"),
                ("conv_q.bias", "attn.conv_q.bias"),
                ("conv_k.weight", "attn.conv_k.weight"),
                ("conv_v.weight", "attn.conv_v.weight"),
                ("conv_o.weight", "attn.conv_o.weight"),
                ("emb_rel_k", "attn.rel_pos_k"),
                ("emb_rel_v", "attn.rel_pos_v"),
            ] {
                let input = format!("flow.flows.{upstream_i}.enc.attn_layers.{j}.{tail_in}");
                let expected = format!("sbv2.flow.layer.0.enc.{j}.{tail_out}");
                assert_eq!(
                    classify_tensor(&input, Some(3)),
                    TensorClass::Rename(expected.clone()),
                    "{input} must rename to {expected}"
                );
            }
            for (tail_in, tail_out) in [
                ("conv_1.weight", "ffn.conv_1.weight"),
                ("conv_2.bias", "ffn.conv_2.bias"),
            ] {
                let input = format!("flow.flows.{upstream_i}.enc.ffn_layers.{j}.{tail_in}");
                let expected = format!("sbv2.flow.layer.0.enc.{j}.{tail_out}");
                assert_eq!(
                    classify_tensor(&input, Some(3)),
                    TensorClass::Rename(expected)
                );
            }
            for (norm_i, dst_prefix) in [(1usize, "norm1"), (2, "norm2")] {
                for (tail_in, tail_out) in [("gamma", "gamma"), ("beta", "beta")] {
                    let input =
                        format!("flow.flows.{upstream_i}.enc.norm_layers_{norm_i}.{j}.{tail_in}");
                    let expected = format!("sbv2.flow.layer.0.enc.{j}.{dst_prefix}.{tail_out}");
                    assert_eq!(
                        classify_tensor(&input, Some(3)),
                        TensorClass::Rename(expected)
                    );
                }
            }
        }
    }

    #[test]
    fn classify_flow_odd_upstream_index_falls_through_to_pass_through() {
        // Upstream odd indices are `Flip` slots — no tensors on the
        // real base checkpoint, but if a hand-authored fixture emits
        // one it must NOT be renamed (that would silently misroute a
        // real coupling's tensor if the index accidentally lined up).
        // Instead, fall through to the pass-through arm with the
        // "flow VITS2" reason for post-mortem visibility.
        for upstream_i in [1usize, 3, 5, 7] {
            let input = format!("flow.flows.{upstream_i}.pre.weight");
            assert!(
                matches!(
                    classify_tensor(&input, Some(3)),
                    TensorClass::PassThrough { .. }
                ),
                "{input} (odd upstream index = Flip slot) must fall through to PassThrough"
            );
        }
    }

    /// Blocker 2b TDD-hardening (2026-08-10) — full-block exhaustive
    /// enumeration pin: for each of the four upstream coupling-block
    /// indices `0/2/4/6`, enumerate every tensor of every sub-namespace
    /// the base SBV2 v2 checkpoint carries, assert (a) every one lands as
    /// `TensorClass::Rename` with the exact `sbv2.flow.layer.<i>.*` target
    /// name the loader (`SbV2Model::from_gguf`) reads, and (b) the total
    /// count is exactly `4 * 114 = 456` — the base checkpoint's own tensor
    /// count for the flow block (`4 blocks × 114 tensors/block`, matching
    /// the module doc's line-141 accounting: 4 blocks × (2 `pre` + 2
    /// `post` + 2 `spk_emb_linear` + 6 encoder layers × (8 attn conv +
    /// 2 attn rel_pos + 4 ffn conv + 4 norm) = 4 × (6 + 6 × 18) = 4 ×
    /// 114 = 456).
    ///
    /// The three existing per-family tests
    /// (`classify_flow_pre_and_post_rename_by_block_index_halving`,
    /// `classify_flow_spk_emb_linear_renames_to_spk_emb`,
    /// `classify_flow_encoder_attn_ffn_norm_all_rename`) each cover **one
    /// sub-namespace at a time**; this test walks all seven together with a
    /// per-tensor rename equality *and* a total-count assertion, so a
    /// regression that (i) silently drops a sub-namespace or (ii) mis-counts
    /// how many tensors a block should have surfaces here as a single
    /// clearly-labelled fail. This is the "flow tensor rename table
    /// contract" pin — every tensor the real base checkpoint carries
    /// must have a stable target name the loader knows how to find.
    #[test]
    fn classify_flow_full_block_renames_all_114_tensors_per_upstream_block() {
        // (upstream_i, target_i) pairs for the four base-checkpoint
        // coupling blocks; upstream indices 1/3/5/7 are `Flip` slots
        // (no tensors — covered by
        // `classify_flow_odd_upstream_index_falls_through_to_pass_through`
        // above).
        let block_pairs = [(0usize, 0usize), (2, 1), (4, 2), (6, 3)];
        // Base SBV2 v2 hparams (design doc §7 + upstream config):
        //   flow.n_encoder_layers = 6
        // (kernel_ffn = 5 / gin_channels = 512 / mean_only = true are
        // architectural but do not enter the rename table — they are
        // metadata-only, covered by the sbv2_gguf_loader.rs pin tests).
        const N_ENCODER_LAYERS: usize = 6;
        // Assemble (upstream_name, expected_target_name) pairs for one
        // upstream block; the outer loop below iterates over the four
        // (upstream_i, target_i) pairs and prepends the block prefix.
        let per_block_pairs: Vec<(String, String)> = {
            let mut v: Vec<(String, String)> = Vec::new();
            // pre / post — 1×1 Conv1d each.
            for (tail_in, tail_out) in [
                ("pre.weight", "pre.weight"),
                ("pre.bias", "pre.bias"),
                ("post.weight", "post.weight"),
                ("post.bias", "post.bias"),
            ] {
                v.push((tail_in.to_string(), tail_out.to_string()));
            }
            // enc.spk_emb_linear — Blocker 3-style g projection at block level.
            for (tail_in, tail_out) in [
                ("enc.spk_emb_linear.weight", "spk_emb.weight"),
                ("enc.spk_emb_linear.bias", "spk_emb.bias"),
            ] {
                v.push((tail_in.to_string(), tail_out.to_string()));
            }
            // Per-encoder-layer j: attn (10) + ffn (4) + norm (4) = 18 tensors.
            for j in 0..N_ENCODER_LAYERS {
                // Attention: 4 conv q/k/v/o × {weight, bias} + 2 emb_rel_{k, v}
                for (tail_in, tail_out) in [
                    ("conv_q.weight", "attn.conv_q.weight"),
                    ("conv_q.bias", "attn.conv_q.bias"),
                    ("conv_k.weight", "attn.conv_k.weight"),
                    ("conv_k.bias", "attn.conv_k.bias"),
                    ("conv_v.weight", "attn.conv_v.weight"),
                    ("conv_v.bias", "attn.conv_v.bias"),
                    ("conv_o.weight", "attn.conv_o.weight"),
                    ("conv_o.bias", "attn.conv_o.bias"),
                    ("emb_rel_k", "attn.rel_pos_k"),
                    ("emb_rel_v", "attn.rel_pos_v"),
                ] {
                    v.push((
                        format!("enc.attn_layers.{j}.{tail_in}"),
                        format!("enc.{j}.{tail_out}"),
                    ));
                }
                // FFN: 2 conv × {weight, bias}
                for (tail_in, tail_out) in [
                    ("conv_1.weight", "ffn.conv_1.weight"),
                    ("conv_1.bias", "ffn.conv_1.bias"),
                    ("conv_2.weight", "ffn.conv_2.weight"),
                    ("conv_2.bias", "ffn.conv_2.bias"),
                ] {
                    v.push((
                        format!("enc.ffn_layers.{j}.{tail_in}"),
                        format!("enc.{j}.{tail_out}"),
                    ));
                }
                // Norm: 2 layers × {gamma, beta}
                for (norm_i, dst_prefix) in [(1usize, "norm1"), (2, "norm2")] {
                    for (tail_in, tail_out) in [("gamma", "gamma"), ("beta", "beta")] {
                        v.push((
                            format!("enc.norm_layers_{norm_i}.{j}.{tail_in}"),
                            format!("enc.{j}.{dst_prefix}.{tail_out}"),
                        ));
                    }
                }
            }
            v
        };
        // Sanity: per-block enumeration is exactly 114 (4 + 2 + 6 × 18 = 114).
        // A drift here (e.g. one of the sub-namespace enumerations goes
        // stale) would silently reduce the total-count check below.
        assert_eq!(
            per_block_pairs.len(),
            114,
            "per-block enumeration must be exactly 114 tensors — got {}",
            per_block_pairs.len()
        );

        let mut total_renamed = 0usize;
        for (upstream_i, target_i) in block_pairs {
            for (tail_in, tail_out) in &per_block_pairs {
                let input = format!("flow.flows.{upstream_i}.{tail_in}");
                let expected = format!("sbv2.flow.layer.{target_i}.{tail_out}");
                let got = classify_tensor(&input, Some(3));
                assert_eq!(
                    got,
                    TensorClass::Rename(expected.clone()),
                    "{input} must rename to {expected}"
                );
                total_renamed += 1;
            }
        }
        assert_eq!(
            total_renamed, 456,
            "4 upstream blocks × 114 tensors/block must yield exactly 456 Rename \
             classifications — got {total_renamed}"
        );
    }

    /// Blocker 2b TDD-hardening (2026-08-10) — malformed / unknown
    /// per-block sub-namespaces must fall through to `PassThrough` with
    /// the `flow.` per-family reason, NOT be silently dropped. Complements
    /// `classify_flow_odd_upstream_index_falls_through_to_pass_through`
    /// (which covers `Flip`-slot indices) by covering typo / rename-drift
    /// on the tail side (a stale upstream config might rename
    /// `emb_rel_k` → `emb_rel_key` in a future revision — the converter
    /// must surface that as a preserved-verbatim tensor, not silently drop
    /// it or panic).
    #[test]
    fn classify_flow_unknown_per_block_subname_falls_through_to_pass_through() {
        // Each of these is a plausible-looking tail that does NOT match
        // any classify_flow_block_tensor arm.
        let unknown_tails = [
            "pre.gamma",                       // pre has {weight, bias} only
            "post.beta",                       // post has {weight, bias} only
            "enc.spk_emb_linear.rel",          // spk_emb_linear has {weight, bias} only
            "enc.attn_layers.0.conv_x.weight", // no `conv_x` — {q,k,v,o} only
            "enc.attn_layers.0.emb_rel_q",     // {k, v} only
            "enc.ffn_layers.0.conv_3.weight",  // {1, 2} only
            "enc.norm_layers_3.0.gamma",       // {1, 2} only
            "enc.attn_layers.0.conv_q.scale",  // {weight, bias} only
            "enc.random_new_sub_module.0",     // whole new sub-namespace
        ];
        for tail in unknown_tails {
            let input = format!("flow.flows.0.{tail}");
            match classify_tensor(&input, Some(3)) {
                TensorClass::PassThrough { reason } => {
                    // FR-EX-08: the fall-through must land on the
                    // `flow.` per-family reason (the "SBV2 flow auxiliary
                    // tensor" arm at classify_tensor's default 5-arm),
                    // not the generic "unrecognized upstream tensor" arm
                    // — a future maintainer needs the family label to
                    // triage.
                    assert!(
                        reason.starts_with("SBV2 flow auxiliary tensor"),
                        "{input} must fall through to the flow.* per-family PassThrough \
                         reason, got: {reason}"
                    );
                }
                other => panic!(
                    "{input} (unknown per-block sub-namespace) must fall through to \
                     PassThrough, got {other:?}"
                ),
            }
        }
    }

    // Post-M6 (2026-08-06) encoder-layer rename tests. Each entry pins
    // one upstream `enc_p.encoder.*` tensor to its target GGUF name
    // under `sbv2.text_encoder.layer.<i>.*` — a stale converter that
    // drops the M6 rename table would silently leave upstream names on
    // disk and the Rust loader would then fail with "tensor not found".
    #[test]
    fn classify_encoder_attn_layers_rename_to_layer_attn_conv_paths() {
        for (i, tail_in, tail_out) in [
            (0usize, "conv_q.weight", "attn.conv_q.weight"),
            (0, "conv_q.bias", "attn.conv_q.bias"),
            (0, "conv_k.weight", "attn.conv_k.weight"),
            (0, "conv_k.bias", "attn.conv_k.bias"),
            (0, "conv_v.weight", "attn.conv_v.weight"),
            (0, "conv_v.bias", "attn.conv_v.bias"),
            (0, "conv_o.weight", "attn.conv_o.weight"),
            (0, "conv_o.bias", "attn.conv_o.bias"),
            (5, "conv_q.weight", "attn.conv_q.weight"),
        ] {
            let input = format!("enc_p.encoder.attn_layers.{i}.{tail_in}");
            let expected = format!("sbv2.text_encoder.layer.{i}.{tail_out}");
            assert_eq!(
                classify_tensor(&input, Some(3)),
                TensorClass::Rename(expected.clone()),
                "attn_layers.{i}.{tail_in} must rename to {expected}"
            );
        }
    }

    #[test]
    fn classify_encoder_attn_layers_emb_rel_renames_to_rel_pos() {
        // Real SBV2 v2 base has `emb_rel_k`/`emb_rel_v` shape `[1, 9, 96]`
        // — the M6 refactor renames these to `rel_pos_k`/`rel_pos_v` so
        // the Rust `RelPositionMHA::new` field names read naturally
        // (`rel_pos_*` matches the mathematical convention while the
        // upstream `emb_rel_*` is a compact abbreviation).
        for (i, tail_in, tail_out) in [
            (0usize, "emb_rel_k", "attn.rel_pos_k"),
            (0, "emb_rel_v", "attn.rel_pos_v"),
            (5, "emb_rel_k", "attn.rel_pos_k"),
        ] {
            let input = format!("enc_p.encoder.attn_layers.{i}.{tail_in}");
            let expected = format!("sbv2.text_encoder.layer.{i}.{tail_out}");
            assert_eq!(
                classify_tensor(&input, Some(3)),
                TensorClass::Rename(expected)
            );
        }
    }

    #[test]
    fn classify_encoder_ffn_and_norm_layers_rename() {
        for (input, expected) in [
            (
                "enc_p.encoder.ffn_layers.0.conv_1.weight",
                "sbv2.text_encoder.layer.0.ffn.conv_1.weight",
            ),
            (
                "enc_p.encoder.ffn_layers.0.conv_1.bias",
                "sbv2.text_encoder.layer.0.ffn.conv_1.bias",
            ),
            (
                "enc_p.encoder.ffn_layers.0.conv_2.weight",
                "sbv2.text_encoder.layer.0.ffn.conv_2.weight",
            ),
            (
                "enc_p.encoder.ffn_layers.0.conv_2.bias",
                "sbv2.text_encoder.layer.0.ffn.conv_2.bias",
            ),
            (
                "enc_p.encoder.norm_layers_1.0.gamma",
                "sbv2.text_encoder.layer.0.norm1.gamma",
            ),
            (
                "enc_p.encoder.norm_layers_1.0.beta",
                "sbv2.text_encoder.layer.0.norm1.beta",
            ),
            (
                "enc_p.encoder.norm_layers_2.5.gamma",
                "sbv2.text_encoder.layer.5.norm2.gamma",
            ),
            (
                "enc_p.encoder.norm_layers_2.5.beta",
                "sbv2.text_encoder.layer.5.norm2.beta",
            ),
        ] {
            assert_eq!(
                classify_tensor(input, Some(3)),
                TensorClass::Rename(expected.into()),
                "{input} must rename to {expected}"
            );
        }
    }

    #[test]
    fn classify_encoder_spk_emb_linear_now_renamed_blocker3() {
        // Post-Blocker-3 (2026-08-06): the external speaker projection is
        // consumed by `ExternalSpeakerProjection`, so `spk_emb_linear` must
        // now rename to the Rust loader's expected name (not pass-through).
        assert_eq!(
            classify_tensor("enc_p.encoder.spk_emb_linear.weight", Some(3)),
            TensorClass::Rename("sbv2.text_encoder.spk_emb_linear.weight".into())
        );
        assert_eq!(
            classify_tensor("enc_p.encoder.spk_emb_linear.bias", Some(3)),
            TensorClass::Rename("sbv2.text_encoder.spk_emb_linear.bias".into())
        );
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
        // Post-Blocker-2c (2026-08-06): sdp.post_* has its own dedicated
        // counter (sdp_post_skipped), distinct from the general
        // skipped_training counter (enc_q + dp only). Both are dropped
        // with an FR-EX-08 stderr line either way.
        assert_eq!(
            report.skipped_training, 2,
            "enc_q.* + dp.* dropped via skipped_training"
        );
        assert_eq!(
            report.sdp_post_skipped, 1,
            "sdp.post_* dropped via dedicated Blocker-2c counter"
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
        // Preservation invariant: `enc_p.proj.*` (VITS output projection
        // to (mu, log_sigma), no Rust text_encoder field yet — tracked
        // as SBV2-INFO-01-ENC-P-PROJ) stays under its upstream names so
        // a future Rust wave can consume it without reconverting the
        // checkpoint. Post-Blocker-2b (2026-08-06) the
        // `flow.flows.<even>.*` per-block tensors are Rename-mapped
        // (see `classify_flow_pre_and_post_rename_by_block_index_
        // halving` and siblings), and post-Blocker-3 (2026-08-06)
        // `enc_p.encoder.spk_emb_linear.*` is Rename-mapped to
        // `sbv2.text_encoder.spk_emb_linear.*`, so both no longer
        // belong in this pass-through-invariant set. Note `sdp.*` is
        // now sdp-rewritten (Blocker 2c) rather than pass-through.
        //
        // HGAN-05-GIN-COND (2026-08-09): `dec.cond.*` moved OUT of
        // this pass-through-invariant set (it is now Rename → the
        // HifiGanWeights::cond loader path). See
        // `classify_dec_cond_now_renamed_hgan05` for the per-tensor
        // pin.
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            (
                "enc_p.proj.weight",
                "F32",
                &[384, 192, 1],
                f32_bytes(&[0.02_f32; 384 * 192]),
            ),
            (
                "enc_p.proj.bias",
                "F32",
                &[384],
                f32_bytes(&[0.03_f32; 384]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("verbatim-pt-in", "safetensors");
        let output = temp_path("verbatim-pt-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.passed_through_verbatim, 2);
        assert_eq!(report.renamed, 0);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        for name in ["enc_p.proj.weight", "enc_p.proj.bias"] {
            assert!(
                file.tensor_info(name).is_some(),
                "{name}: must land under upstream name"
            );
        }

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Blocker 2b (2026-08-06): end-to-end rename check — a fixture
    /// containing one representative per-block flow tensor for each of
    /// the 7 per-family branches must land under its target GGUF name
    /// (not the upstream name).
    #[test]
    fn flow_family_tensors_land_under_renamed_targets_end_to_end() {
        let entries: Vec<(&str, &str, &[u64], Vec<u8>)> = vec![
            (
                "flow.flows.0.pre.weight",
                "F32",
                &[192, 96, 1],
                f32_bytes(&[0.01_f32; 192 * 96]),
            ),
            (
                "flow.flows.2.post.bias",
                "F32",
                &[96],
                f32_bytes(&[0.02_f32; 96]),
            ),
            (
                "flow.flows.4.enc.spk_emb_linear.weight",
                "F32",
                &[192, 512],
                f32_bytes(&[0.03_f32; 192 * 512]),
            ),
            (
                "flow.flows.6.enc.attn_layers.0.conv_q.weight",
                "F32",
                &[192, 192, 1],
                f32_bytes(&[0.04_f32; 192 * 192]),
            ),
            (
                "flow.flows.0.enc.attn_layers.5.emb_rel_k",
                "F32",
                &[9, 96],
                f32_bytes(&[0.05_f32; 9 * 96]),
            ),
            (
                "flow.flows.2.enc.ffn_layers.3.conv_1.weight",
                "F32",
                &[768, 192, 5],
                f32_bytes(&[0.06_f32; 768 * 192 * 5]),
            ),
            (
                "flow.flows.4.enc.norm_layers_2.5.gamma",
                "F32",
                &[192, 1],
                f32_bytes(&[0.07_f32; 192]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("flow-rename-in", "safetensors");
        let output = temp_path("flow-rename-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 7);
        assert_eq!(report.written, 7);
        assert_eq!(
            report.renamed, 7,
            "every one of the 7 per-family entries must go through the Rename branch"
        );
        assert_eq!(report.passed_through_verbatim, 0);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        // Every entry must land under its target name (block index
        // divided by 2: 0/2/4/6 → 0/1/2/3).
        for target in [
            "sbv2.flow.layer.0.pre.weight",
            "sbv2.flow.layer.1.post.bias",
            "sbv2.flow.layer.2.spk_emb.weight",
            "sbv2.flow.layer.3.enc.0.attn.conv_q.weight",
            "sbv2.flow.layer.0.enc.5.attn.rel_pos_k",
            "sbv2.flow.layer.1.enc.3.ffn.conv_1.weight",
            "sbv2.flow.layer.2.enc.5.norm2.gamma",
        ] {
            assert!(
                file.tensor_info(target).is_some(),
                "{target}: must land under its target GGUF name post-Blocker-2b"
            );
        }
        // And none of the upstream names must survive.
        for upstream in [
            "flow.flows.0.pre.weight",
            "flow.flows.2.post.bias",
            "flow.flows.4.enc.spk_emb_linear.weight",
            "flow.flows.6.enc.attn_layers.0.conv_q.weight",
            "flow.flows.0.enc.attn_layers.5.emb_rel_k",
            "flow.flows.2.enc.ffn_layers.3.conv_1.weight",
            "flow.flows.4.enc.norm_layers_2.5.gamma",
        ] {
            assert!(
                file.tensor_info(upstream).is_none(),
                "{upstream}: upstream name must NOT survive after Blocker 2b rename"
            );
        }

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn dec_resblocks_convs2_lands_under_mrf_layer_c2_slots() {
        // Wave-2 HGAN-01 (2026-08-09) — pre-fix this test asserted
        // convs2 stayed as PassThrough because the Rust runtime had no
        // consumer for it. Now `mrf_branch_forward` runs the full V1
        // `for (c1, c2) in zip(convs1, convs2)` chain and the loader
        // demands both convs1 + convs2 tensors, so the converter now
        // renames convs2.{weight_g/weight_v pairs, bias, bare weight}
        // to sibling `sbv2.decoder.mrf.<s>.<b>.layer.<l>.{weight_c2,bias_c2}`.
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
        let input = temp_path("convs2-mrf-c2-in", "safetensors");
        let config = temp_path("convs2-mrf-c2-cfg", "json");
        let output = temp_path("convs2-mrf-c2-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        std::fs::write(&config, &cfg_bytes).expect("write cfg");

        let report = convert_sbv2_file(&input, &output, Some(&config), None).expect("convert");
        // 3 tensors read; weight_g + weight_v pair reconstructs to one
        // weight_c2 (1 weight-norm), bias renames to bias_c2 (1 rename).
        assert_eq!(report.read, 3);
        assert_eq!(report.weight_norm_reconstructed, 1);
        assert_eq!(report.renamed, 1);
        assert_eq!(
            report.passed_through_verbatim, 0,
            "convs2 must no longer PassThrough — it renames to weight_c2/bias_c2"
        );

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        assert!(
            file.tensor_info("sbv2.decoder.mrf.0.0.layer.0.weight_c2")
                .is_some(),
            "convs2 weight lands under sbv2.decoder.mrf.0.0.layer.0.weight_c2"
        );
        assert!(
            file.tensor_info("sbv2.decoder.mrf.0.0.layer.0.bias_c2")
                .is_some(),
            "convs2 bias lands under sbv2.decoder.mrf.0.0.layer.0.bias_c2"
        );
        // The upstream names must not leak into the output.
        assert!(
            file.tensor_info("dec.resblocks.0.convs2.0.weight_g")
                .is_none(),
            "post-Wave-2: upstream convs2 name must not survive"
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
        // Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS (2026-08-09): the
        // config-carrying path now also emits 4 converter_zero_default
        // tensors for the optional slots the fixture's input does not
        // supply (conv_post.bias + style_injector.proj_{scale,bias} +
        // speaker.table). Bias + weight_norm pair = 2, + 4 zero defaults
        // = 6.
        assert_eq!(
            report.written, 6,
            "weight pair + bias + 4 zero-defaults = 6 emits"
        );
        assert_eq!(
            report.converter_zero_defaults_emitted, 4,
            "config-driven runs emit 4 zero-default tensors when the input fixture ships none of the optional slots"
        );
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
        // Invariant: `written + skipped_non_float + skipped_training +
        // sdp_post_skipped + weight_norm_v_consumed == read`, and
        // `renamed + weight_norm_reconstructed + passed_through_verbatim +
        // sdp_rewritten == written`. Mix all buckets in one fixture to
        // lock the partition.
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
            // PassThrough (real pass-through: enc_p.proj.* has no Rust
            // text_encoder field yet, still verbatim)
            (
                "enc_p.proj.weight",
                "F32",
                &[8, 4, 1],
                f32_bytes(&[0.03_f32; 32]),
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
        // Input-side partition — every read tensor lands in exactly one
        // input bucket (Blocker 2c added sdp_post_skipped as a distinct
        // input bucket, separate from skipped_training).
        assert_eq!(
            r.written
                + r.skipped_non_float
                + r.skipped_training
                + r.sdp_post_skipped
                + r.weight_norm_v_consumed,
            r.read,
            "input-side partition: written + skipped_non_float + skipped_training + \
             sdp_post_skipped + weight_norm_v_consumed == read"
        );
        // Output-side partition — every written tensor lands in exactly
        // one output bucket (Blocker 2c added sdp_rewritten as a distinct
        // output bucket, separate from renamed/verbatim).
        assert_eq!(
            r.renamed + r.weight_norm_reconstructed + r.passed_through_verbatim + r.sdp_rewritten,
            r.written,
            "output-side partition: renamed + wnorm_reconstructed + verbatim + \
             sdp_rewritten == written"
        );
        // Concrete values: enc_p.emb.weight=Rename,
        // enc_q.pre.weight=Skip, enc_p.proj.weight=PassThrough,
        // (dec.ups.0.weight_g + dec.ups.0.weight_v)=WeightNorm — the
        // weight_g reconstructs (+1 written, +1 wnorm), the weight_v is
        // consumed (+1 weight_norm_v_consumed), and dec.ups.0.bias=Rename.
        assert_eq!(r.renamed, 2);
        assert_eq!(r.weight_norm_reconstructed, 1);
        assert_eq!(r.weight_norm_v_consumed, 1, "one weight_v folded");
        assert_eq!(r.passed_through_verbatim, 1);
        assert_eq!(r.skipped_training, 1);
        assert_eq!(r.sdp_post_skipped, 0);
        assert_eq!(r.sdp_rewritten, 0);
        assert_eq!(r.written, 4);
    }

    #[test]
    fn sdp_convert_rewrites_production_and_skips_post() {
        let blob = safetensors_multi(&[
            // 3 production-side SDP tensors — each on a different arm of
            // the rewriter (body, EA flow slot 0, ConvFlow flow slot 1).
            ("sdp.pre.weight", "F32", &[4, 4, 1], f32_bytes(&[0.1; 16])),
            ("sdp.flows.0.m", "F32", &[2, 1], f32_bytes(&[0.2; 2])),
            (
                "sdp.flows.1.proj.weight",
                "F32",
                &[29, 4, 1],
                f32_bytes(&[0.3; 116]),
            ),
            // 2 training-side tensors — both must be skipped.
            (
                "sdp.post_pre.weight",
                "F32",
                &[4, 1, 1],
                f32_bytes(&[0.4; 4]),
            ),
            ("sdp.post_flows.0.m", "F32", &[2, 1], f32_bytes(&[0.5; 2])),
        ]);
        let input = temp_path("sdp-rewrite-in", "safetensors");
        let output = temp_path("sdp-rewrite-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");

        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.read, 5, "5 input tensors observed");
        assert_eq!(report.written, 3, "3 production tensors written");
        assert_eq!(
            report.sdp_rewritten, 3,
            "3 production tensors got rewritten names"
        );
        assert_eq!(report.sdp_post_skipped, 2, "2 sdp.post_ tensors skipped");
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");

        // Production tensors present under their remapped names, absent
        // under their upstream names.
        assert!(file.tensor_info("sbv2.sdp.pre.weight").is_some());
        assert!(file.tensor_info("sdp.pre.weight").is_none());
        assert!(file.tensor_info("sbv2.sdp.ea.m").is_some());
        assert!(file.tensor_info("sdp.flows.0.m").is_none());
        assert!(file.tensor_info("sbv2.sdp.flow.0.proj.weight").is_some());
        assert!(file.tensor_info("sdp.flows.1.proj.weight").is_none());

        // Training-side tensors absent altogether.
        assert!(file.tensor_info("sdp.post_pre.weight").is_none());
        assert!(file.tensor_info("sbv2.sdp.post_pre.weight").is_none());
        assert!(file.tensor_info("sdp.post_flows.0.m").is_none());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // =====================================================================
    // Wave-4 WORKFLOW-SHAPE-FIXUP (2026-08-09)
    // =====================================================================

    /// `infer_d_speaker` primary path: `enc_p.encoder.spk_emb_linear.weight`
    /// shape's last axis IS d_speaker. Real SBV2 v2 JP-Extra base ships
    /// `[192, 512]` -> d_speaker = 512 (not the VITS default 256 the
    /// clean-room fallback emits).
    #[test]
    fn infer_d_speaker_recovers_from_spk_emb_linear() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![(
            "enc_p.encoder.spk_emb_linear.weight",
            "F32",
            &[192u64, 512],
            f32_bytes(&vec![0.0; 192 * 512]),
        )];
        let blob = safetensors_multi(&entries);
        let st = SafetensorsFile::parse(blob).expect("parse safetensors");
        assert_eq!(
            infer_d_speaker(&st),
            Some(512),
            "d_speaker must be recovered as 512 from spk_emb_linear.weight[1]"
        );
    }

    /// `infer_d_speaker` fallback path: no `spk_emb_linear` but
    /// `emb_g.weight` is present (some fine-tunes strip the linear).
    #[test]
    fn infer_d_speaker_falls_back_to_emb_g() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![(
            "emb_g.weight",
            "F32",
            &[7u64, 384],
            f32_bytes(&vec![0.0; 7 * 384]),
        )];
        let blob = safetensors_multi(&entries);
        let st = SafetensorsFile::parse(blob).expect("parse safetensors");
        assert_eq!(
            infer_d_speaker(&st),
            Some(384),
            "d_speaker must be recovered as 384 from emb_g.weight[1]"
        );
    }

    /// `infer_d_speaker` returns None when both primary and fallback
    /// tensors are absent (single-speaker fine-tune with no speaker
    /// weights at all) — the recovery block MUST then leave the config's
    /// declared value in place.
    #[test]
    fn infer_d_speaker_returns_none_when_no_speaker_tensors() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![(
            // A totally unrelated tensor.
            "enc_p.emb.weight",
            "F32",
            &[10u64, 192],
            f32_bytes(&vec![0.0; 10 * 192]),
        )];
        let blob = safetensors_multi(&entries);
        let st = SafetensorsFile::parse(blob).expect("parse safetensors");
        assert!(infer_d_speaker(&st).is_none());
    }

    /// `infer_n_speakers` reads `emb_g.weight[0]`.
    #[test]
    fn infer_n_speakers_reads_emb_g_first_axis() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![(
            "emb_g.weight",
            "F32",
            &[5u64, 512],
            f32_bytes(&vec![0.0; 5 * 512]),
        )];
        let blob = safetensors_multi(&entries);
        let st = SafetensorsFile::parse(blob).expect("parse safetensors");
        assert_eq!(infer_n_speakers(&st), Some(5));
    }

    /// `infer_decoder_upsample_kernel_sizes`: real SBV2 v2 JP-Extra base
    /// ships `[16, 16, 8, 2, 2]`, not the HiFi-GAN 2*stride default
    /// `[16, 16, 4, 4, 4]` that the config side-car's
    /// `--clean-room-defaults` emits. Every stage's `dec.ups.<i>.weight`
    /// is `[in_ch, out_ch, kernel]`; the helper reads the last axis.
    #[test]
    fn infer_decoder_upsample_kernel_sizes_recovers_full_ladder() {
        let kernels = [16u64, 16, 8, 2, 2];
        let mut entries: Vec<(String, &'static str, Vec<u64>, Vec<u8>)> = Vec::new();
        for (i, &k) in kernels.iter().enumerate() {
            // Shape [in_ch=64, out_ch=32, kernel=k] — the SBV2 loader only
            // cares about the LAST axis for kernel-size recovery.
            entries.push((
                format!("dec.ups.{i}.weight"),
                "F32",
                vec![64u64, 32, k],
                f32_bytes(&vec![0.0; 64 * 32 * (k as usize)]),
            ));
        }
        let borrowed: Vec<(&str, &'static str, &[u64], Vec<u8>)> = entries
            .iter()
            .map(|(n, d, s, p)| (n.as_str(), *d, s.as_slice(), p.clone()))
            .collect();
        let blob = safetensors_multi(&borrowed);
        let st = SafetensorsFile::parse(blob).expect("parse safetensors");
        assert_eq!(
            infer_decoder_upsample_kernel_sizes(&st),
            Some(vec![16, 16, 8, 2, 2]),
            "must recover the real SBV2 v2 JP-Extra base 5-stage ladder"
        );
    }

    /// Weight-normed variant: real HF-hosted checkpoints often store
    /// `dec.ups.<i>.weight_v` alongside `.weight_g`; the recovery must
    /// prefer `.weight_v` (the raw pre-norm weight) when present.
    #[test]
    fn infer_decoder_upsample_kernel_sizes_prefers_weight_v() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![
            (
                "dec.ups.0.weight_v",
                "F32",
                &[64u64, 32, 20],
                f32_bytes(&vec![0.0; 64 * 32 * 20]),
            ),
            // A stray `dec.ups.0.weight` with a DIFFERENT last dim would
            // silently mismatch — if the helper picked the wrong one, the
            // assertion below fails.
            (
                "dec.ups.0.weight",
                "F32",
                &[64u64, 32, 4],
                f32_bytes(&vec![0.0; 64 * 32 * 4]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let st = SafetensorsFile::parse(blob).expect("parse safetensors");
        assert_eq!(
            infer_decoder_upsample_kernel_sizes(&st),
            Some(vec![20]),
            "weight_v (the raw pre-norm weight) must be preferred over the folded weight"
        );
    }

    /// A stage gap (e.g. dec.ups.0 + dec.ups.1 present, then dec.ups.2
    /// absent) stops the probe cleanly at the last-present stage. That's
    /// the correct behavior: the ladder simply has fewer stages, not a
    /// corrupt checkpoint.
    #[test]
    fn infer_decoder_upsample_kernel_sizes_stops_at_first_gap() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![
            (
                "dec.ups.0.weight",
                "F32",
                &[64u64, 32, 16],
                f32_bytes(&vec![0.0; 64 * 32 * 16]),
            ),
            (
                "dec.ups.1.weight",
                "F32",
                &[64u64, 32, 8],
                f32_bytes(&vec![0.0; 64 * 32 * 8]),
            ),
            // dec.ups.2.* absent.
        ];
        let blob = safetensors_multi(&entries);
        let st = SafetensorsFile::parse(blob).expect("parse safetensors");
        assert_eq!(
            infer_decoder_upsample_kernel_sizes(&st),
            Some(vec![16, 8]),
            "probe must stop at first missing stage, not extrapolate"
        );
    }

    // =====================================================================
    // Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS (2026-08-09)
    // =====================================================================

    /// A minimal config side-car covering only the fields
    /// `emit_converter_zero_defaults` reads (d_model / d_style / d_speaker)
    /// plus the required-fields the parser cross-checks. Every other field
    /// is set to a valid nonzero value.
    fn minimal_cfg_json(d_model: u32, d_style: u32, d_speaker: u32) -> String {
        format!(
            r#"{{
                "sample_rate": 44100,
                "n_speakers": 1,
                "n_vocab": 112,
                "n_tones": 12,
                "d_model": {d_model},
                "d_z": {d_model},
                "d_speaker": {d_speaker},
                "d_ff": 768,
                "d_style": {d_style},
                "d_bert": 1024,
                "n_text_layers": 1,
                "n_flow_layers": 1,
                "n_sdp_layers": 1,
                "n_heads": 2,
                "window_size": 4,
                "kernel_ffn": 3,
                "decoder_initial_channel": 32,
                "decoder_conv_pre_kernel": 7,
                "decoder_conv_post_kernel": 7,
                "decoder_upsample_rates": [8, 8, 2, 2, 2],
                "decoder_upsample_kernel_sizes": [16, 16, 4, 4, 4],
                "decoder_upsample_out_channels": [16, 8, 4, 2, 1],
                "decoder_resblock_kernel_sizes": [3, 7, 11],
                "decoder_resblock_dilation_counts": [3, 3, 3],
                "decoder_resblock_dilations_flat": [1, 3, 5, 1, 3, 5, 1, 3, 5],
                "decoder_leaky_relu_slope": 0.1
            }}"#
        )
    }

    /// Given an input safetensors that ships NONE of the three optional
    /// slots, the converter emits all four zero-default tensors +
    /// stamps `vokra.sbv2.converter_zero_defaults` with a comma-joined
    /// trail listing every emitted slot.
    #[test]
    fn emit_converter_zero_defaults_full_trail_when_no_optional_slots_present() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![(
            // Something non-conflicting the main loop passes through
            "enc_p.emb.weight",
            "F32",
            &[112u64, 8],
            f32_bytes(&vec![0.0; 112 * 8]),
        )];
        let blob = safetensors_multi(&entries);
        let input = temp_path("cvz-full-in", "safetensors");
        let cfg = temp_path("cvz-full-cfg", "json");
        let output = temp_path("cvz-full-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        std::fs::write(&cfg, minimal_cfg_json(8, 4, 16).as_bytes()).expect("write cfg");

        let report = convert_sbv2_file(&input, &output, Some(&cfg), None).expect("convert");
        // 4 zero defaults: conv_post.bias + proj_scale + proj_bias + speaker.table.
        assert_eq!(report.converter_zero_defaults_emitted, 4);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");

        // Every zero-default slot is present in the emitted GGUF.
        assert!(file.tensor_info("sbv2.decoder.conv_post.bias").is_some());
        assert!(file.tensor_info("sbv2.style_injector.proj_scale").is_some());
        assert!(file.tensor_info("sbv2.style_injector.proj_bias").is_some());
        assert!(file.tensor_info("sbv2.speaker.table").is_some());

        // Provenance trail lists all four slots in emission order.
        let trail = file
            .get(KEY_CONVERTER_ZERO_DEFAULTS)
            .and_then(|v| v.as_str())
            .expect("KEY_CONVERTER_ZERO_DEFAULTS must be stamped");
        for name in [
            "sbv2.decoder.conv_post.bias",
            "sbv2.style_injector.proj_scale",
            "sbv2.style_injector.proj_bias",
            "sbv2.speaker.table",
        ] {
            assert!(
                trail.contains(name),
                "trail `{trail}` must mention `{name}`"
            );
        }

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&cfg).ok();
        std::fs::remove_file(&output).ok();
    }

    /// When the input safetensors ships one of the optional slots (e.g.
    /// `dec.conv_post.bias` — a fine-tune SKU), the converter's main loop
    /// passes it through; `emit_converter_zero_defaults` does NOT overwrite
    /// it (idempotent contract) and the trail lists only the still-missing
    /// slots.
    #[test]
    fn emit_converter_zero_defaults_skips_slots_already_written() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![
            (
                "enc_p.emb.weight",
                "F32",
                &[112u64, 8],
                f32_bytes(&vec![0.0; 112 * 8]),
            ),
            (
                // Real fine-tune value; main loop renames this to
                // `sbv2.decoder.conv_post.bias` via classify_tensor.
                "dec.conv_post.bias",
                "F32",
                &[1u64],
                f32_bytes(&[0.5]),
            ),
        ];
        let blob = safetensors_multi(&entries);
        let input = temp_path("cvz-partial-in", "safetensors");
        let cfg = temp_path("cvz-partial-cfg", "json");
        let output = temp_path("cvz-partial-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        std::fs::write(&cfg, minimal_cfg_json(8, 4, 16).as_bytes()).expect("write cfg");

        let report = convert_sbv2_file(&input, &output, Some(&cfg), None).expect("convert");
        // 3 zero defaults (conv_post.bias came from input; other 3 emitted).
        assert_eq!(report.converter_zero_defaults_emitted, 3);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");

        // conv_post.bias present with the FINE-TUNE value (0.5), NOT
        // overwritten by zero-default (0.0).
        let bias = file
            .tensor_f32("sbv2.decoder.conv_post.bias")
            .expect("bias present");
        assert_eq!(
            bias,
            vec![0.5],
            "input fine-tune bias must not be overwritten by zero-default"
        );

        // Trail lists the 3 slots emitted, NOT the 1 already-present slot.
        let trail = file
            .get(KEY_CONVERTER_ZERO_DEFAULTS)
            .and_then(|v| v.as_str())
            .expect("KEY_CONVERTER_ZERO_DEFAULTS must be stamped");
        assert!(!trail.contains("sbv2.decoder.conv_post.bias"));
        assert!(trail.contains("sbv2.style_injector.proj_scale"));
        assert!(trail.contains("sbv2.style_injector.proj_bias"));
        assert!(trail.contains("sbv2.speaker.table"));

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&cfg).ok();
        std::fs::remove_file(&output).ok();
    }

    /// When NO config side-car is supplied, `emit_converter_zero_defaults`
    /// is a no-op (needs `d_model`/`d_style`/`d_speaker` for shape). The
    /// pre-Wave-4 loader-side fallback path stays reachable for those
    /// (config-less) GGUFs.
    #[test]
    fn emit_converter_zero_defaults_no_op_without_config() {
        let entries: Vec<(&'static str, &'static str, &'static [u64], Vec<u8>)> = vec![(
            "enc_p.emb.weight",
            "F32",
            &[112u64, 8],
            f32_bytes(&vec![0.0; 112 * 8]),
        )];
        let blob = safetensors_multi(&entries);
        let input = temp_path("cvz-noop-in", "safetensors");
        let output = temp_path("cvz-noop-out", "gguf");
        std::fs::write(&input, &blob).expect("write input");
        // No config passed.
        let report = convert_sbv2_file(&input, &output, None, None).expect("convert");
        assert_eq!(report.converter_zero_defaults_emitted, 0);

        let out_bytes = std::fs::read(&output).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");
        assert!(file.get(KEY_CONVERTER_ZERO_DEFAULTS).is_none());
        // The three optional slots stay absent from the GGUF too, so
        // the pre-Wave-4 loader-side fallback path is what the loader
        // would use if it opened this file.
        assert!(file.tensor_info("sbv2.decoder.conv_post.bias").is_none());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
