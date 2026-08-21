//! **EAT** (`cwx-worst-one/EAT`, MIT) — self-supervised audio encoder
//! runtime binder for the `eat` converter arch (Wave C2, 2026-08-15;
//! ViT wave, 2026-08-15).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/eat.rs` (SSL audio-encoder wave,
//! 2026-08-13) stamps `vokra.model.arch = "eat"` onto every GGUF it
//! produces, but a workspace-wide grep proved that **nothing read that
//! arch string back** — weights converted, and then no code path could
//! load them. This module is that consumer.
//!
//! # What EAT is
//!
//! EAT is a self-supervised audio encoder trained with a
//! bootstrap / self-distillation objective and **inverse block masking**
//! over an utterance-level Transformer, pre-trained on AudioSet-2M with
//! MAE-style masked reconstruction (Chen et al. 2024,
//! [arXiv:2401.03497]). The `eat-base` size point is ~86 M parameters
//! (~350 MB PyTorch checkpoint). It is positioned upstream as an
//! efficient alternative to BEATs / AST for downstream audio tagging and
//! general audio-embedding tasks.
//!
//! **Naming note** (recorded so a reader is not confused by a
//! cross-file discrepancy): the acronym is expanded as *Efficient Audio
//! Transformer* in the upstream paper title, while the converter's
//! module docstring writes *Effective Audio Transformer*. Both refer to
//! the same release, [`UPSTREAM_URL`] / [`PRIMARY_SOURCE_PAPER`]. This
//! binder does not adjudicate the spelling; it only cites both anchors.
//!
//! # This is a feature extractor, not an end-task model
//!
//! EAT emits **representations**, not labels: a sequence of hidden
//! states over the patchified spectrogram plus an utterance-level
//! embedding. The upstream release ships downstream task heads
//! (AudioSet tagging, ESC-50, SPC-2 fine-tunes) **separately** from the
//! pre-trained encoder, so this binder deliberately exposes only
//! [`Eat::encode`] (frame/patch hidden states) and
//! [`Eat::embed_utterance`] (the utterance-level embedding). **No
//! classification head is invented here** — the pre-training checkpoint
//! this converter targets does not contain one, and fabricating a
//! label space would be exactly the "fake-complete" failure CLAUDE.md
//! 教訓 (a) warns about.
//!
//! # Runtime layout
//!
//! ```text
//! PCM (mono f32)
//!   -> Kaldi-fbank front-end                            ← **loud-partial**
//!        (the ARGUMENTS are stamped in full — see
//!         `vokra.eat.fbank_*` and [`EatConfig`] — but
//!         `vokra_ops::kaldi_fbank` hard-codes the Povey
//!         window while EAT passes `window_type='hanning'`,
//!         and the op has no window selector yet.)
//!   -> 2-D patch embedding over the mel plane           ← **real**
//!        ([`vokra_ops::vit::vit_patch_embed`], driven by
//!         [`EatConfig::to_vit_attrs`].)
//!   -> pre-norm Transformer encoder stack               ← **real**
//!        ([`vokra_ops::vit::ViTEncoder`], weights bound by
//!         [`Eat::bind_vit_weights`] from a caller-supplied
//!         [`EatVitTensorNames`].)
//!   -> per-patch hidden states  ............... `Eat::encode`
//!   -> utterance-level pooling / CLS read-out  ← **loud-partial**
//!        ([`vokra_ops::vit::ViTPooling`] can express either
//!         convention, but which one EAT trained is not
//!         transcribed.)
//!   -> utterance embedding  ................... `Eat::embed_utterance`
//! ```
//!
//! # The `vokra.eat.*` topology chunk group IS read
//!
//! An earlier landing of this module recorded that "the EAT converter
//! stamps **no** topology axes at all". That was true when it was
//! written and is **no longer true**: the converter now stamps a
//! 38-key `vokra.eat.*` group covering the ViT-B backbone, the patch
//! grid, the pre-training decoder and the complete Kaldi-fbank argument
//! set. [`EatConfig::from_gguf`] reads every one of them **strictly**,
//! in the `vokra.wavlm.*` posture: a missing key is a loud
//! [`VokraError::ModelLoad`] naming it, never a fallback to a
//! primary-source constant, because a silent default would let a
//! mismatched artifact bind a topology it does not carry (FR-EX-08).
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (as of this landing)**:
//!   - [`Eat::from_gguf`] with **strict** `vokra.model.arch == "eat"`
//!     verification, a `vokra.model.category` cross-check, the tensor
//!     manifest, and the full [`EatConfig`] axis group.
//!   - [`EatConfig::to_vit_attrs`], which maps the stamped axes onto
//!     [`vokra_ops::vit::ViTAttrs`], **derives** the patch stride and
//!     checks it against the independently stamped grid rather than
//!     assuming it, and refuses a multi-channel patch stem.
//!   - [`Eat::bind_vit_weights`] / [`Eat::bind_vit_encoder`], which
//!     decode a real [`vokra_ops::vit::ViTWeights`] out of the GGUF
//!     through [`EatWeights::require_tensor`] /
//!     [`EatWeights::require_tensor_dims`], so every absent or
//!     wrong-shaped tensor names itself.
//!   - Weight-license surfacing that fail-closes to
//!     [`LicenseClass::Unknown`] when the stamp is absent.
//!
//! - **Loud-partial (still)**: [`Eat::encode`] and
//!   [`Eat::embed_utterance`] return [`VokraError::UnsupportedOp`].
//!   They take **PCM**, and three things stand between PCM and a
//!   defensible embedding: the front-end window mismatch, the absence
//!   of any verified tensor-name manifest, and the un-reconciled
//!   `layer_norm_first` flag. See [`Eat::encode`] for each in full.
//!   **No fabricated hidden states or embeddings are ever emitted**
//!   (FR-EX-08 — no silent partial output).
//!
//! # Sibling family distinctness (SSL audio-encoder neighbourhood)
//!
//! [`ARCH`] = `"eat"` is deliberately distinct from every sibling SSL
//! audio-encoder arch tag landed in the converter tree — `beats`
//! (iterative acoustic-tokenizer SSL), `dasheng` (universal MAE),
//! `atst` (teacher-student patchout), `m2d` (masked-modeling duo),
//! `mert` / `muq` (music-domain SSL), `ast` (supervised audio
//! spectrogram Transformer, not self-supervised), `hubert` (masked
//! cluster prediction over raw waveform). They share a family
//! resemblance but not a topology: silently aliasing arch would let
//! runtime dispatch bind, say, an MAE decoder over an utterance-level
//! checkpoint and produce shape-valid garbage instead of a loud error
//! (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_URL`] /
//! [`DEFAULT_LICENSE_SPDX`] and every `GGUF_KEY_*` spelling are
//! **mirrors of the converter's constants**, not imports — the same
//! rule every sibling binder follows so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF
//! reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`. The tests pin every mirrored value so a converter-side
//! rename must land here in the same commit or fail.
//!
//! # Licensing
//!
//! Upstream `github.com/cwx-worst-one/EAT` reports `spdx_id: MIT` via
//! the GitHub license API (converter task input, 2026-08-13), so the
//! converter stamps `mit` → [`LicenseClass::Permissive`]. This binder
//! only **surfaces** whatever class the artifact carries and
//! fail-closes to [`LicenseClass::Unknown`] when the stamp is missing.
//! `docs/license-audit.md` §3.1 sign-off stays **blank** — owner-only
//! per memory `[[feedback-license-signoff-primary-source]]`; Claude
//! Code does not sign.
//!
//! # No ONNX / no pickle (permanent)
//!
//! EAT ships upstream as a PyTorch `.pt` pickle from the GitHub
//! releases page; neither this runtime nor the converter ever touches
//! ONNX or pickle (FR-LD-05 / NFR-DS-02). The `.pt` → safetensors
//! bridge is an offline, uv-managed Python 3.12 sidecar (memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`),
//! mirroring the DAC / Kokoro / UTMOSv2 pattern.
//!
//! [arXiv:2401.03497]: https://arxiv.org/abs/2401.03497

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::vit::{
    GeluKind, PatchEmbedWeights, PosEmbedPolicy, ViTAttnWeights, ViTAttrs, ViTBlockWeights,
    ViTEncoder, ViTMlpWeights, ViTWeights, patch_grid,
};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/eat.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model eat`.
///
/// Distinct from every sibling SSL audio-encoder arch tag — `beats`,
/// `dasheng`, `atst`, `m2d`, `mert`, `muq`, `ast`, `hubert`. Silently
/// sharing an arch would misroute runtime dispatch onto a loader whose
/// tensor walk expects a different topology (FR-EX-08).
pub const ARCH: &str = "eat";

/// Expected `vokra.model.name` value written by the converter — the
/// canonical `eat-base` size point.
///
/// The upstream releases page also carries an `eat-large` variant; per
/// the converter's docstring that is published under its own `NAME` via
/// a separate `ModelKind` (the `snac_24khz` / `snac_44khz` precedent),
/// so this binder pins the base point only.
pub const NAME: &str = "eat-base";

/// Expected `vokra.model.category` value — general audio embedding.
///
/// Consumed by the model-card generator and the zoo-manifest tier gate
/// so an audio-embedding release is never advertised as an ASR / TTS
/// model.
pub const CATEGORY: &str = "audio-embedding";

/// Upstream source tree. EAT is **not** hosted on HuggingFace, so the
/// converter stamps `vokra.provenance.upstream_url` rather than
/// `upstream_hf` (the `nsnet2` / `beats` posture); the model-card
/// generator accepts either.
pub const UPSTREAM_URL: &str = "github.com/cwx-worst-one/EAT";

/// SPDX identifier the converter stamps by default.
///
/// Upstream `cwx-worst-one/EAT` LICENSE reports `spdx_id: MIT` via the
/// GitHub license API (converter task input, 2026-08-13). A caller with
/// a different attestation may override at the converter boundary
/// (`--license <spdx>`), so this binder never *asserts* the class — it
/// reads back whatever was stamped.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// `vokra.model.category` metadata key (mirror of the converter's
/// private constant — not exported by `vokra_core::gguf::chunks`).
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` metadata key (mirror of the
/// converter's private constant — not exported by
/// `vokra_core::gguf::chunks`).
pub const GGUF_KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Primary-source anchor: the paper (Chen et al. 2024).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2401.03497";

/// Tensor-name prefix of the ViT-style encoder blocks, as exercised by
/// the converter's own round-trip test (`blocks.0.attn.qkv.weight`).
///
/// Used **only** for pure-observation structure discovery
/// ([`EatWeights::observed_block_count`]); it is never a load gate,
/// because the upstream state-dict naming has not been transcribed
/// anywhere in-repo and a real fairseq/data2vec2-lineage checkpoint may
/// well use a different prefix.
pub const BLOCK_PREFIX: &str = "blocks.";

/// Tensor-name prefix of the 2-D patch-embedding stem, as exercised by
/// the converter's own round-trip test (`patch_embed.proj.weight`).
///
/// Observation only — see [`BLOCK_PREFIX`] for why this is not a gate.
pub const PATCH_EMBED_PREFIX: &str = "patch_embed.";

// ---------------------------------------------------------------------------
// `vokra.eat.*` chunk keys — byte-identical mirrors of the private
// `KEY_EAT_*` constants in `crates/vokra-convert/src/models/eat.rs`.
// The tests pin every spelling, so a converter-side rename must land
// here in the same commit or fail.
// ---------------------------------------------------------------------------

/// `vokra.eat.embed_dim` — Transformer width (`UINT32`).
pub const GGUF_KEY_EMBED_DIM: &str = "vokra.eat.embed_dim";
/// `vokra.eat.depth` — encoder block count (`UINT32`).
pub const GGUF_KEY_DEPTH: &str = "vokra.eat.depth";
/// `vokra.eat.num_heads` — attention head count (`UINT32`).
pub const GGUF_KEY_NUM_HEADS: &str = "vokra.eat.num_heads";
/// `vokra.eat.mlp_ratio` — feed-forward expansion ratio (`FLOAT32`).
pub const GGUF_KEY_MLP_RATIO: &str = "vokra.eat.mlp_ratio";
/// `vokra.eat.norm_eps` — LayerNorm epsilon (`FLOAT32`).
pub const GGUF_KEY_NORM_EPS: &str = "vokra.eat.norm_eps";
/// `vokra.eat.layer_norm_first` — transcribed `layer_norm_first` flag (`BOOL`).
pub const GGUF_KEY_LAYER_NORM_FIRST: &str = "vokra.eat.layer_norm_first";
/// `vokra.eat.patch_size` — square patch side in spectrogram cells (`UINT32`).
pub const GGUF_KEY_PATCH_SIZE: &str = "vokra.eat.patch_size";
/// `vokra.eat.in_chans` — patch-embedding input channels (`UINT32`).
pub const GGUF_KEY_IN_CHANS: &str = "vokra.eat.in_chans";
/// `vokra.eat.target_length` — fixed spectrogram length in frames (`UINT32`).
pub const GGUF_KEY_TARGET_LENGTH: &str = "vokra.eat.target_length";
/// `vokra.eat.n_mels` — mel-bin count / frequency extent (`UINT32`).
pub const GGUF_KEY_N_MELS: &str = "vokra.eat.n_mels";
/// `vokra.eat.patch_grid_time` — time-axis patch count (`UINT32`).
pub const GGUF_KEY_PATCH_GRID_TIME: &str = "vokra.eat.patch_grid_time";
/// `vokra.eat.patch_grid_freq` — frequency-axis patch count (`UINT32`).
pub const GGUF_KEY_PATCH_GRID_FREQ: &str = "vokra.eat.patch_grid_freq";
/// `vokra.eat.num_patches` — patch tokens per clip (`UINT32`).
pub const GGUF_KEY_NUM_PATCHES: &str = "vokra.eat.num_patches";
/// `vokra.eat.num_extra_tokens` — prepended non-patch tokens (`UINT32`).
pub const GGUF_KEY_NUM_EXTRA_TOKENS: &str = "vokra.eat.num_extra_tokens";
/// `vokra.eat.pos_embed_max_length` — positional-grid height (`UINT32`).
pub const GGUF_KEY_POS_EMBED_MAX_LENGTH: &str = "vokra.eat.pos_embed_max_length";
/// `vokra.eat.decoder_dim` — pre-training decoder width (`UINT32`).
pub const GGUF_KEY_DECODER_DIM: &str = "vokra.eat.decoder_dim";
/// `vokra.eat.decoder_groups` — pre-training decoder conv groups (`UINT32`).
pub const GGUF_KEY_DECODER_GROUPS: &str = "vokra.eat.decoder_groups";
/// `vokra.eat.decoder_kernel` — pre-training decoder conv kernel (`UINT32`).
pub const GGUF_KEY_DECODER_KERNEL: &str = "vokra.eat.decoder_kernel";
/// `vokra.eat.decoder_layers` — pre-training decoder depth (`UINT32`).
pub const GGUF_KEY_DECODER_LAYERS: &str = "vokra.eat.decoder_layers";
/// `vokra.eat.fbank_sample_rate` — front-end sample rate, Hz (`UINT32`).
pub const GGUF_KEY_FBANK_SAMPLE_RATE: &str = "vokra.eat.fbank_sample_rate";
/// `vokra.eat.fbank_frame_length_ms` — analysis frame length, ms (`UINT32`).
pub const GGUF_KEY_FBANK_FRAME_LENGTH_MS: &str = "vokra.eat.fbank_frame_length_ms";
/// `vokra.eat.fbank_frame_shift_ms` — frame hop, ms (`UINT32`).
pub const GGUF_KEY_FBANK_FRAME_SHIFT_MS: &str = "vokra.eat.fbank_frame_shift_ms";
/// `vokra.eat.fbank_window_type` — analysis window name (`STRING`).
pub const GGUF_KEY_FBANK_WINDOW_TYPE: &str = "vokra.eat.fbank_window_type";
/// `vokra.eat.fbank_htk_compat` — Kaldi `htk_compat` argument (`BOOL`).
pub const GGUF_KEY_FBANK_HTK_COMPAT: &str = "vokra.eat.fbank_htk_compat";
/// `vokra.eat.fbank_use_energy` — Kaldi `use_energy` argument (`BOOL`).
pub const GGUF_KEY_FBANK_USE_ENERGY: &str = "vokra.eat.fbank_use_energy";
/// `vokra.eat.fbank_dither` — Kaldi `dither` argument (`FLOAT32`).
pub const GGUF_KEY_FBANK_DITHER: &str = "vokra.eat.fbank_dither";
/// `vokra.eat.fbank_low_freq` — low mel band edge, Hz (`FLOAT32`).
pub const GGUF_KEY_FBANK_LOW_FREQ: &str = "vokra.eat.fbank_low_freq";
/// `vokra.eat.fbank_high_freq` — high mel band edge, Kaldi encoding (`FLOAT32`).
pub const GGUF_KEY_FBANK_HIGH_FREQ: &str = "vokra.eat.fbank_high_freq";
/// `vokra.eat.fbank_preemph_coeff` — pre-emphasis coefficient (`FLOAT32`).
pub const GGUF_KEY_FBANK_PREEMPH_COEFF: &str = "vokra.eat.fbank_preemph_coeff";
/// `vokra.eat.fbank_remove_dc_offset` — per-frame DC removal (`BOOL`).
pub const GGUF_KEY_FBANK_REMOVE_DC_OFFSET: &str = "vokra.eat.fbank_remove_dc_offset";
/// `vokra.eat.fbank_round_to_power_of_two` — FFT-size rounding (`BOOL`).
pub const GGUF_KEY_FBANK_ROUND_TO_POWER_OF_TWO: &str = "vokra.eat.fbank_round_to_power_of_two";
/// `vokra.eat.fbank_snip_edges` — snip-edges framing (`BOOL`).
pub const GGUF_KEY_FBANK_SNIP_EDGES: &str = "vokra.eat.fbank_snip_edges";
/// `vokra.eat.fbank_use_power` — power vs. magnitude spectrum (`BOOL`).
pub const GGUF_KEY_FBANK_USE_POWER: &str = "vokra.eat.fbank_use_power";
/// `vokra.eat.fbank_use_log` — log mel energies (`BOOL`).
pub const GGUF_KEY_FBANK_USE_LOG: &str = "vokra.eat.fbank_use_log";
/// `vokra.eat.fbank_subtract_mean` — per-utterance CMN (`BOOL`).
pub const GGUF_KEY_FBANK_SUBTRACT_MEAN: &str = "vokra.eat.fbank_subtract_mean";
/// `vokra.eat.fbank_norm_mean` — feature normalisation mean (`FLOAT32`).
pub const GGUF_KEY_FBANK_NORM_MEAN: &str = "vokra.eat.fbank_norm_mean";
/// `vokra.eat.fbank_norm_std` — feature normalisation std (`FLOAT32`).
pub const GGUF_KEY_FBANK_NORM_STD: &str = "vokra.eat.fbank_norm_std";
/// `vokra.eat.fbank_norm_std_multiplier` — divisor multiplier (`FLOAT32`).
pub const GGUF_KEY_FBANK_NORM_STD_MULTIPLIER: &str = "vokra.eat.fbank_norm_std_multiplier";

// ---------------------------------------------------------------------------
// Strict metadata readers. Each names the absent key and the repro
// command; none of them falls back to a constant (FR-EX-08).
// ---------------------------------------------------------------------------

/// The shared tail of every missing-key message.
fn missing_key(key: &str, kind: &str) -> VokraError {
    VokraError::ModelLoad(format!(
        "eat: GGUF is missing required {kind} chunk `{key}` — a converter-produced \
         artifact always carries the full `vokra.eat.*` topology group, every value of \
         which `crates/vokra-convert/src/models/eat.rs` transcribes from the upstream \
         source tree ({UPSTREAM_URL}). This binder refuses to fall back to a \
         primary-source constant (FR-EX-08): a silent default would let an artifact bind \
         a topology it does not actually carry. Re-run `vokra-cli convert --model eat` \
         against an upstream release flattened to safetensors by the offline uv-managed \
         Python 3.12 sidecar. (A key stamped under the wrong GGUF type also reads back as \
         absent here.)"
    ))
}

/// Reads a required `u32`-valued chunk.
fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| missing_key(key, "u32"))
}

/// Reads a required `f32`-valued chunk.
///
/// `GgufMetadataValue::as_f64` widens an on-disk `FLOAT32` losslessly,
/// and narrowing it back is exact, so the value round-trips bit-for-bit
/// through this reader.
fn req_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    gguf.get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .ok_or_else(|| missing_key(key, "f32"))
}

/// Reads a required `bool`-valued chunk.
fn req_bool(gguf: &GgufFile, key: &str) -> Result<bool> {
    gguf.get(key)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| missing_key(key, "bool"))
}

/// Reads a required string-valued chunk.
fn req_string(gguf: &GgufFile, key: &str) -> Result<String> {
    gguf.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| missing_key(key, "string"))
}

// ---------------------------------------------------------------------------
// EatConfig — the `vokra.eat.*` axis group.
// ---------------------------------------------------------------------------

/// The EAT topology + front-end axes as they ride the `vokra.eat.*`
/// chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader in the
/// `vokra.wavlm.*` posture: every one of the 38 stamped keys is
/// required, and a missing one is a loud [`VokraError::ModelLoad`]
/// naming it. There is deliberately no primary-source constant
/// fallback — the converter stamps the whole group unconditionally, so
/// a partial group means a mis-produced or hand-edited artifact, and a
/// silent default would let it bind a topology it does not carry
/// (FR-EX-08).
///
/// [`eat_base_reference`](Self::eat_base_reference) exists for tests and
/// diagnostics only; the loader never consults it.
#[derive(Debug, Clone, PartialEq)]
pub struct EatConfig {
    /// Transformer embedding width (`vokra.eat.embed_dim`).
    pub embed_dim: u32,
    /// Transformer encoder block count (`vokra.eat.depth`).
    pub depth: u32,
    /// Attention head count (`vokra.eat.num_heads`).
    pub num_heads: u32,
    /// Feed-forward expansion ratio (`vokra.eat.mlp_ratio`).
    pub mlp_ratio: f32,
    /// LayerNorm epsilon (`vokra.eat.norm_eps`).
    pub norm_eps: f32,
    /// Transcribed upstream `layer_norm_first` flag
    /// (`vokra.eat.layer_norm_first`).
    ///
    /// Carried verbatim, **not** interpreted: the converter records
    /// that upstream `models/modules.py AltBlock` is what gives this
    /// flag its meaning and that the file has not been transcribed. See
    /// [`EatConfig::to_vit_attrs`] for why this matters.
    pub layer_norm_first: bool,
    /// Square patch side, in spectrogram cells (`vokra.eat.patch_size`).
    pub patch_size: u32,
    /// Patch-embedding input channel count (`vokra.eat.in_chans`).
    pub in_chans: u32,
    /// Fixed spectrogram length in frames (`vokra.eat.target_length`).
    pub target_length: u32,
    /// Mel-bin count, i.e. the plane's frequency extent
    /// (`vokra.eat.n_mels`).
    pub n_mels: u32,
    /// Time-axis patch count (`vokra.eat.patch_grid_time`).
    pub patch_grid_time: u32,
    /// Frequency-axis patch count (`vokra.eat.patch_grid_freq`).
    pub patch_grid_freq: u32,
    /// Patch-token count per clip (`vokra.eat.num_patches`).
    pub num_patches: u32,
    /// Prepended non-patch (CLS) token count
    /// (`vokra.eat.num_extra_tokens`).
    pub num_extra_tokens: u32,
    /// Positional-table height in time patches
    /// (`vokra.eat.pos_embed_max_length`).
    pub pos_embed_max_length: u32,
    /// Pre-training MAE decoder width (`vokra.eat.decoder_dim`).
    pub decoder_dim: u32,
    /// Pre-training MAE decoder conv groups (`vokra.eat.decoder_groups`).
    pub decoder_groups: u32,
    /// Pre-training MAE decoder conv kernel (`vokra.eat.decoder_kernel`).
    pub decoder_kernel: u32,
    /// Pre-training MAE decoder depth (`vokra.eat.decoder_layers`).
    pub decoder_layers: u32,
    /// Front-end sample rate in Hz (`vokra.eat.fbank_sample_rate`).
    pub fbank_sample_rate: u32,
    /// Analysis frame length in ms (`vokra.eat.fbank_frame_length_ms`).
    pub fbank_frame_length_ms: u32,
    /// Frame hop in ms (`vokra.eat.fbank_frame_shift_ms`).
    pub fbank_frame_shift_ms: u32,
    /// Analysis window name (`vokra.eat.fbank_window_type`).
    pub fbank_window_type: String,
    /// Kaldi `htk_compat` argument (`vokra.eat.fbank_htk_compat`).
    pub fbank_htk_compat: bool,
    /// Kaldi `use_energy` argument (`vokra.eat.fbank_use_energy`).
    pub fbank_use_energy: bool,
    /// Kaldi `dither` argument (`vokra.eat.fbank_dither`).
    pub fbank_dither: f32,
    /// Low mel band edge in Hz (`vokra.eat.fbank_low_freq`).
    pub fbank_low_freq: f32,
    /// High mel band edge in Kaldi's own encoding, where a non-positive
    /// value means "Nyquist + high_freq" (`vokra.eat.fbank_high_freq`).
    pub fbank_high_freq: f32,
    /// Pre-emphasis coefficient (`vokra.eat.fbank_preemph_coeff`).
    pub fbank_preemph_coeff: f32,
    /// Per-frame DC removal (`vokra.eat.fbank_remove_dc_offset`).
    pub fbank_remove_dc_offset: bool,
    /// FFT-size rounding (`vokra.eat.fbank_round_to_power_of_two`).
    pub fbank_round_to_power_of_two: bool,
    /// Snip-edges framing (`vokra.eat.fbank_snip_edges`).
    pub fbank_snip_edges: bool,
    /// Power vs. magnitude spectrum (`vokra.eat.fbank_use_power`).
    pub fbank_use_power: bool,
    /// Log mel energies (`vokra.eat.fbank_use_log`).
    pub fbank_use_log: bool,
    /// Per-utterance cepstral mean normalisation
    /// (`vokra.eat.fbank_subtract_mean`).
    pub fbank_subtract_mean: bool,
    /// Feature normalisation mean (`vokra.eat.fbank_norm_mean`).
    pub fbank_norm_mean: f32,
    /// Feature normalisation standard deviation
    /// (`vokra.eat.fbank_norm_std`).
    pub fbank_norm_std: f32,
    /// Multiplier applied to the normalisation divisor — upstream
    /// normalises as `(feats - mean) / (std * multiplier)`
    /// (`vokra.eat.fbank_norm_std_multiplier`).
    pub fbank_norm_std_multiplier: f32,
}

impl EatConfig {
    /// The `eat-base` axes as the converter's own transcribed constants
    /// record them.
    ///
    /// Diagnostic and test reference **only**: [`Self::from_gguf`] does
    /// not fall back to these, it reads the stamped values and fails
    /// loud on any missing chunk. Mirrors the
    /// `WavlmSvConfig::base_plus_default` posture.
    #[must_use]
    pub fn eat_base_reference() -> Self {
        Self {
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            norm_eps: 1e-6,
            layer_norm_first: false,
            patch_size: 16,
            in_chans: 1,
            target_length: 1024,
            n_mels: 128,
            patch_grid_time: 64,
            patch_grid_freq: 8,
            num_patches: 512,
            num_extra_tokens: 1,
            pos_embed_max_length: 768,
            decoder_dim: 768,
            decoder_groups: 16,
            decoder_kernel: 3,
            decoder_layers: 6,
            fbank_sample_rate: 16_000,
            fbank_frame_length_ms: 25,
            fbank_frame_shift_ms: 10,
            fbank_window_type: "hanning".to_owned(),
            fbank_htk_compat: true,
            fbank_use_energy: false,
            fbank_dither: 0.0,
            fbank_low_freq: 20.0,
            fbank_high_freq: 0.0,
            fbank_preemph_coeff: 0.97,
            fbank_remove_dc_offset: true,
            fbank_round_to_power_of_two: true,
            fbank_snip_edges: true,
            fbank_use_power: true,
            fbank_use_log: true,
            fbank_subtract_mean: false,
            fbank_norm_mean: -4.268,
            fbank_norm_std: 4.569,
            fbank_norm_std_multiplier: 2.0,
        }
    }

    /// Reads every `vokra.eat.*` chunk from `gguf`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the first absent key. A key
    ///   stamped under the wrong GGUF value type reads back as absent
    ///   and produces the same loud error, which is the intended
    ///   behaviour: a type mismatch is a mis-produced artifact.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Ok(Self {
            embed_dim: req_u32(gguf, GGUF_KEY_EMBED_DIM)?,
            depth: req_u32(gguf, GGUF_KEY_DEPTH)?,
            num_heads: req_u32(gguf, GGUF_KEY_NUM_HEADS)?,
            mlp_ratio: req_f32(gguf, GGUF_KEY_MLP_RATIO)?,
            norm_eps: req_f32(gguf, GGUF_KEY_NORM_EPS)?,
            layer_norm_first: req_bool(gguf, GGUF_KEY_LAYER_NORM_FIRST)?,
            patch_size: req_u32(gguf, GGUF_KEY_PATCH_SIZE)?,
            in_chans: req_u32(gguf, GGUF_KEY_IN_CHANS)?,
            target_length: req_u32(gguf, GGUF_KEY_TARGET_LENGTH)?,
            n_mels: req_u32(gguf, GGUF_KEY_N_MELS)?,
            patch_grid_time: req_u32(gguf, GGUF_KEY_PATCH_GRID_TIME)?,
            patch_grid_freq: req_u32(gguf, GGUF_KEY_PATCH_GRID_FREQ)?,
            num_patches: req_u32(gguf, GGUF_KEY_NUM_PATCHES)?,
            num_extra_tokens: req_u32(gguf, GGUF_KEY_NUM_EXTRA_TOKENS)?,
            pos_embed_max_length: req_u32(gguf, GGUF_KEY_POS_EMBED_MAX_LENGTH)?,
            decoder_dim: req_u32(gguf, GGUF_KEY_DECODER_DIM)?,
            decoder_groups: req_u32(gguf, GGUF_KEY_DECODER_GROUPS)?,
            decoder_kernel: req_u32(gguf, GGUF_KEY_DECODER_KERNEL)?,
            decoder_layers: req_u32(gguf, GGUF_KEY_DECODER_LAYERS)?,
            fbank_sample_rate: req_u32(gguf, GGUF_KEY_FBANK_SAMPLE_RATE)?,
            fbank_frame_length_ms: req_u32(gguf, GGUF_KEY_FBANK_FRAME_LENGTH_MS)?,
            fbank_frame_shift_ms: req_u32(gguf, GGUF_KEY_FBANK_FRAME_SHIFT_MS)?,
            fbank_window_type: req_string(gguf, GGUF_KEY_FBANK_WINDOW_TYPE)?,
            fbank_htk_compat: req_bool(gguf, GGUF_KEY_FBANK_HTK_COMPAT)?,
            fbank_use_energy: req_bool(gguf, GGUF_KEY_FBANK_USE_ENERGY)?,
            fbank_dither: req_f32(gguf, GGUF_KEY_FBANK_DITHER)?,
            fbank_low_freq: req_f32(gguf, GGUF_KEY_FBANK_LOW_FREQ)?,
            fbank_high_freq: req_f32(gguf, GGUF_KEY_FBANK_HIGH_FREQ)?,
            fbank_preemph_coeff: req_f32(gguf, GGUF_KEY_FBANK_PREEMPH_COEFF)?,
            fbank_remove_dc_offset: req_bool(gguf, GGUF_KEY_FBANK_REMOVE_DC_OFFSET)?,
            fbank_round_to_power_of_two: req_bool(gguf, GGUF_KEY_FBANK_ROUND_TO_POWER_OF_TWO)?,
            fbank_snip_edges: req_bool(gguf, GGUF_KEY_FBANK_SNIP_EDGES)?,
            fbank_use_power: req_bool(gguf, GGUF_KEY_FBANK_USE_POWER)?,
            fbank_use_log: req_bool(gguf, GGUF_KEY_FBANK_USE_LOG)?,
            fbank_subtract_mean: req_bool(gguf, GGUF_KEY_FBANK_SUBTRACT_MEAN)?,
            fbank_norm_mean: req_f32(gguf, GGUF_KEY_FBANK_NORM_MEAN)?,
            fbank_norm_std: req_f32(gguf, GGUF_KEY_FBANK_NORM_STD)?,
            fbank_norm_std_multiplier: req_f32(gguf, GGUF_KEY_FBANK_NORM_STD_MULTIPLIER)?,
        })
    }

    /// Token count one fixed-length clip produces:
    /// `num_extra_tokens + num_patches`.
    #[must_use]
    pub fn tokens_per_clip(&self) -> usize {
        self.num_extra_tokens as usize + self.num_patches as usize
    }

    /// Row count of the **stored** positional table:
    /// `num_extra_tokens + pos_embed_max_length * patch_grid_freq`.
    ///
    /// The converter's `POS_EMBED_MAX_LENGTH` docs transcribe the
    /// upstream sizing rule — the fixed 2-D sin-cos table is built over
    /// a `(max_length, patch_grid_freq)` grid, deliberately taller than
    /// the [`Self::tokens_per_clip`] rows one clip consumes, which is
    /// what lets the encoder accept variable-length spectrograms.
    ///
    /// A caller choosing a [`PosEmbedPolicy`] needs both numbers: the
    /// stored table is larger than the runtime sequence, so
    /// [`PosEmbedPolicy::RequireExact`] applies to an **offline-sliced**
    /// table, not to the raw on-disk one.
    #[must_use]
    pub fn pos_embed_table_rows(&self) -> usize {
        self.num_extra_tokens as usize
            + self.pos_embed_max_length as usize * self.patch_grid_freq as usize
    }

    /// Maps the stamped axes onto [`vokra_ops::vit::ViTAttrs`].
    ///
    /// # Where each `ViTAttrs` field comes from
    ///
    /// - `embed_dim` / `depth` / `n_heads` / `mlp_ratio` /
    ///   `layer_norm_eps` — stamped directly as `vokra.eat.embed_dim`,
    ///   `depth`, `num_heads`, `mlp_ratio`, `norm_eps`.
    /// - `patch_h` / `patch_w` — both from the single stamped
    ///   `vokra.eat.patch_size`, which the converter records as square.
    /// - `stride_h` / `stride_w` — **derived**, not stamped. EAT stamps
    ///   the grid instead, and the grid is only consistent with
    ///   non-overlapping tiling, so this maps stride to `patch_size`
    ///   and then *checks* that choice against the independently
    ///   stamped `patch_grid_freq` / `patch_grid_time` / `num_patches`.
    ///   A disagreement means overlapping patches, whose stride is not
    ///   stamped anywhere, and is refused rather than guessed.
    /// - `n_prepended_tokens` — stamped `vokra.eat.num_extra_tokens`.
    /// - `gelu` / `pos_embed_policy` — **not in the stamped group**, so
    ///   they are parameters of this call. The sibling `atst` binder
    ///   resolves both from metadata because its converter stamps
    ///   `act_layer` and `pos_type`; EAT's converter stamps neither, and
    ///   inventing them here would silently pick a different function
    ///   than the checkpoint was trained with (the GELU flavours differ
    ///   by ~1e-3, and an interpolating positional policy resamples a
    ///   table that upstream slices).
    ///
    /// # Axis orientation
    ///
    /// `vokra_ops::vit` walks a `[n_mels, n_frames]` plane, so its
    /// `grid_h` is the **frequency** axis and its `grid_w` is the
    /// **time** axis. This mapping therefore checks `grid_h` against
    /// `patch_grid_freq` and `grid_w` against `patch_grid_time`, not
    /// the other way around.
    ///
    /// # This does NOT adjudicate norm order
    ///
    /// [`vokra_ops::vit::ViTEncoder`] is pre-norm **by construction**
    /// and its own module docs call the post-norm ordering "a different
    /// function whose outputs are shape-valid and numerically wrong".
    /// `vokra.eat.layer_norm_first` is stamped `false` for `eat-base`,
    /// and the converter is explicit that this is a transcribed config
    /// value rather than an assertion about where the norms sit — that
    /// is decided by upstream `models/modules.py AltBlock`, which is not
    /// transcribed in-repo. `ViTAttrs` has no norm-order axis to carry
    /// the flag into, so this mapping cannot express the difference and
    /// does not pretend to: reconciling it is one of the blockers
    /// [`Eat::encode`] still names.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any axis needed for the mapping
    ///   is stamped zero, when `in_chans` is not 1, when the derived
    ///   non-overlapping stride disagrees with the stamped grid, or when
    ///   `num_patches` contradicts the stamped grid.
    /// - [`VokraError::InvalidArgument`] propagated from
    ///   [`ViTAttrs::validate`].
    pub fn to_vit_attrs(
        &self,
        gelu: GeluKind,
        pos_embed_policy: PosEmbedPolicy,
    ) -> Result<ViTAttrs> {
        // --- Zero guards before any division. `ViTAttrs::validate` catches
        // --- most of these too, but dividing first would panic before it ran.
        for (key, value) in [
            (GGUF_KEY_EMBED_DIM, self.embed_dim),
            (GGUF_KEY_DEPTH, self.depth),
            (GGUF_KEY_NUM_HEADS, self.num_heads),
            (GGUF_KEY_PATCH_SIZE, self.patch_size),
            (GGUF_KEY_N_MELS, self.n_mels),
            (GGUF_KEY_TARGET_LENGTH, self.target_length),
        ] {
            if value == 0 {
                return Err(VokraError::ModelLoad(format!(
                    "eat: `{key}` is stamped 0, which cannot describe a real encoder. \
                     Refusing to bind a degenerate topology (FR-EX-08). Primary sources: \
                     {UPSTREAM_URL} + {PRIMARY_SOURCE_PAPER}"
                )));
            }
        }

        // --- The ViT primitive patch-embeds a single-channel plane: its
        // --- `PatchEmbedWeights::proj_w` is `[embed_dim, patch_h * patch_w]`,
        // --- i.e. a `Conv2d(1, embed_dim, ...)` flattened over its trailing
        // --- dims. A multi-channel stem carries a wider weight and a
        // --- different flattening, so it is refused rather than reshaped.
        if self.in_chans != 1 {
            return Err(VokraError::ModelLoad(format!(
                "eat: `{GGUF_KEY_IN_CHANS}` is {chans}, but `vokra_ops::vit` patch-embeds a \
                 single-channel `[n_mels, n_frames]` plane. Upstream selects EAT's audio \
                 geometry with exactly the `in_chans == 1` branch of `models/images.py`, so \
                 a value other than 1 is either the 3-channel ImageNet modality or a \
                 mis-produced artifact. Refusing to reshape a multi-channel patch stem \
                 through the single-channel path (FR-EX-08). Primary source: {UPSTREAM_URL}",
                chans = self.in_chans,
            )));
        }

        // --- Stride derivation, checked against the independently stamped
        // --- grid rather than assumed (see the doc comment).
        let patch = self.patch_size as usize;
        let grid_freq = self.patch_grid_freq as usize;
        let grid_time = self.patch_grid_time as usize;
        let tiled_freq = self.n_mels as usize / patch;
        let tiled_time = self.target_length as usize / patch;
        if grid_freq != tiled_freq || grid_time != tiled_time {
            return Err(VokraError::ModelLoad(format!(
                "eat: stamped patch grid is freq={grid_freq}, time={grid_time}, but \
                 non-overlapping tiling of a {mels}x{frames} plane by a {patch}x{patch} \
                 patch gives freq={tiled_freq}, time={tiled_time}. Upstream derives the \
                 grid as `img_size[1] // patch_size[1]` and `img_size[0] // patch_size[0]`, \
                 so the two must agree; a disagreement means this artifact uses overlapping \
                 patches, whose stride is NOT stamped anywhere in `vokra.eat.*`. Refusing \
                 to guess a stride (FR-EX-08). Primary source: {UPSTREAM_URL}",
                mels = self.n_mels,
                frames = self.target_length,
            )));
        }
        let expected_patches = grid_freq * grid_time;
        if self.num_patches as usize != expected_patches {
            return Err(VokraError::ModelLoad(format!(
                "eat: `{GGUF_KEY_NUM_PATCHES}` is {n} but the stamped grid freq={grid_freq} \
                 x time={grid_time} implies {expected_patches}. Refusing to bind \
                 contradictory axes (FR-EX-08). Primary source: {UPSTREAM_URL}",
                n = self.num_patches,
            )));
        }

        let attrs = ViTAttrs {
            embed_dim: self.embed_dim as usize,
            depth: self.depth as usize,
            n_heads: self.num_heads as usize,
            mlp_ratio: self.mlp_ratio,
            patch_h: patch,
            patch_w: patch,
            // Non-overlapping tiling, verified against the stamped grid above.
            stride_h: patch,
            stride_w: patch,
            n_prepended_tokens: self.num_extra_tokens as usize,
            layer_norm_eps: self.norm_eps,
            gelu,
            pos_embed_policy,
        };
        attrs.validate()?;

        // --- Final cross-check against the primitive that will actually
        // --- consume these attrs, so the derivation is verified by the
        // --- consumer rather than only by arithmetic restated here.
        let grid = patch_grid(self.n_mels as usize, self.target_length as usize, &attrs)?;
        if grid.grid_h != grid_freq
            || grid.grid_w != grid_time
            || grid.dropped_rows != 0
            || grid.dropped_cols != 0
        {
            return Err(VokraError::ModelLoad(format!(
                "eat: `vokra_ops::vit::patch_grid` walks the stamped {mels}x{frames} plane \
                 into grid_h={gh}, grid_w={gw} (dropping {dr} mel bin(s) and {dc} frame(s)), \
                 but `vokra.eat.*` stamps freq={grid_freq}, time={grid_time} with nothing \
                 dropped. The derived stride does not reproduce the stamped grid under the \
                 primitive that will consume it, so binding would patchify a different \
                 plane than upstream did (FR-EX-08). Primary source: {UPSTREAM_URL}",
                mels = self.n_mels,
                frames = self.target_length,
                gh = grid.grid_h,
                gw = grid.grid_w,
                dr = grid.dropped_rows,
                dc = grid.dropped_cols,
            )));
        }
        Ok(attrs)
    }
}

// ---------------------------------------------------------------------------
// Caller-supplied ViT tensor-name manifest.
// ---------------------------------------------------------------------------

/// The `state_dict` names of one Transformer block's tensors.
///
/// The field *set* follows the ViT block structure
/// [`vokra_ops::vit::ViTBlockWeights`] consumes — two LayerNorms, a
/// fused QKV projection, an output projection and a two-layer MLP. The
/// **strings** are the caller's to supply; see [`EatVitTensorNames`] for
/// why this type ships no default.
///
/// # Fused QKV contract
///
/// [`Self::qkv_weight`] names a single `[3 * embed_dim, embed_dim]`
/// row-major tensor which this binder splits into thirds in **q, k, v**
/// row order, and [`Self::qkv_bias`] a matching `[3 * embed_dim]`
/// vector. That is the layout the one EAT tensor name present anywhere
/// in this repository implies (`blocks.0.attn.qkv.weight`, from the
/// converter's own round-trip test) and the same contract the sibling
/// `atst` manifest uses. A checkpoint that stores q, k and v separately
/// must be fused by the offline sidecar before conversion — this
/// runtime will not guess which of the two conventions a payload is in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EatVitBlockTensorNames {
    /// Pre-attention LayerNorm gain, `[embed_dim]`.
    pub norm1_weight: String,
    /// Pre-attention LayerNorm bias, `[embed_dim]`.
    pub norm1_bias: String,
    /// Fused QKV projection weight, `[3 * embed_dim, embed_dim]`.
    pub qkv_weight: String,
    /// Fused QKV projection bias, `[3 * embed_dim]` — `None` for a
    /// bias-free projection.
    pub qkv_bias: Option<String>,
    /// Attention output projection weight, `[embed_dim, embed_dim]`.
    pub proj_weight: String,
    /// Attention output projection bias, `[embed_dim]`, when present.
    pub proj_bias: Option<String>,
    /// Pre-MLP LayerNorm gain, `[embed_dim]`.
    pub norm2_weight: String,
    /// Pre-MLP LayerNorm bias, `[embed_dim]`.
    pub norm2_bias: String,
    /// MLP first linear weight, `[mlp_dim, embed_dim]`.
    pub fc1_weight: String,
    /// MLP first linear bias, `[mlp_dim]`, when present.
    pub fc1_bias: Option<String>,
    /// MLP second linear weight, `[embed_dim, mlp_dim]`.
    pub fc2_weight: String,
    /// MLP second linear bias, `[embed_dim]`, when present.
    pub fc2_bias: Option<String>,
}

/// A full EAT ViT tensor-name manifest, supplied by the caller.
///
/// # There is deliberately no `Default` and no `eat_base()`
///
/// This is the first of the blockers [`Eat::encode`] names, made
/// explicit in the type system. **Nothing in this repository records
/// EAT's real `state_dict` keys.** The converter passes upstream
/// safetensors names through verbatim, and the only EAT tensor names
/// present anywhere in-repo are the converter's own round-trip
/// fixtures — `patch_embed.proj.weight` and `blocks.0.attn.qkv.weight`
/// — which that test's own comment labels merely "realistic". EAT
/// descends from fairseq data2vec2 (its config dataclass is
/// `Data2VecMultiConfig`), whose modality-specific parameters live under
/// a `modality_encoders.*` tree instead, so the two plausible
/// conventions disagree precisely where it matters.
///
/// Shipping a guessed default would let callers bind the wrong tensors
/// **without failing**, so the caller must supply names it can defend,
/// exactly as [`vokra_ops::vit::ViTAttrs`] refuses to default its axes.
/// Feed the result to [`Eat::bind_vit_weights`] or
/// [`Eat::bind_vit_encoder`], which shape-check every name against the
/// dims [`EatConfig::to_vit_attrs`] derives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EatVitTensorNames {
    /// Patch projection weight. Accepted either as the 4-D `Conv2d`
    /// form `[embed_dim, 1, patch_size, patch_size]` or already
    /// flattened to `[embed_dim, patch_size * patch_size]` — the two
    /// carry identical row-major payloads, so this binder validates the
    /// element count rather than forcing one spelling of the dims.
    pub patch_embed_weight: String,
    /// Patch projection bias, `[embed_dim]`, when present.
    pub patch_embed_bias: Option<String>,
    /// Learned prepended (CLS) token parameter. `None` only when the
    /// config prepends no tokens; a mismatch between this and
    /// `num_extra_tokens` is refused loudly.
    pub prepended_tokens: Option<String>,
    /// Positional table. Its row count is validated as a whole number of
    /// `embed_dim`-wide rows here; whether that row count is *usable* is
    /// decided by the [`PosEmbedPolicy`] the caller chose (see
    /// [`EatConfig::pos_embed_table_rows`]).
    pub pos_embed: String,
    /// One entry per Transformer block, in depth order. Its length must
    /// equal the stamped `vokra.eat.depth`.
    pub blocks: Vec<EatVitBlockTensorNames>,
    /// Final LayerNorm gain, applied after the whole stack.
    pub final_norm_weight: String,
    /// Final LayerNorm bias.
    pub final_norm_bias: String,
}

// ---------------------------------------------------------------------------
// EatWeights — the tensor manifest, with a non-empty gate, loud lookups,
// and pure-observation structure discovery.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an EAT GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step — a GGUF carrying zero tensors is refused rather than silently
/// binding an all-zero forward (FR-EX-08).
///
/// The struct stores tensor names and their GGUF-side dims. Payloads are
/// decoded on demand by [`Eat::bind_vit_weights`], which drives every
/// lookup through [`require_tensor`](Self::require_tensor) /
/// [`require_tensor_dims`](Self::require_tensor_dims) so a missing or
/// wrong-shaped tensor names itself.
#[derive(Debug)]
pub struct EatWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims.
    tensors: Vec<(String, Vec<usize>)>,
}

impl EatWeights {
    /// Scans `gguf` for the EAT `state_dict` tensors.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "eat: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate EAT checkpoint is ~86 M parameters \
                 (arch={ARCH}, name={NAME}): a 2-D patch-embedding stem plus a \
                 Transformer encoder stack carry hundreds of Linear / LayerNorm / Conv \
                 tensors, so zero tensors always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model eat` against an upstream `{UPSTREAM_URL}` \
                 release flattened to safetensors. Primary source: {PRIMARY_SOURCE_PAPER}"
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors discovered on disk.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Every discovered tensor name, in on-disk order.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// GGUF dimensions of `name`, or `None` when it is absent.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// How many discovered tensors start with `prefix`.
    ///
    /// A pure observation over what is on disk — it asserts **no**
    /// naming scheme (the upstream EAT state-dict naming is not
    /// transcribed anywhere in-repo).
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .count()
    }

    /// `true` when the manifest carries at least one tensor under
    /// [`PATCH_EMBED_PREFIX`].
    ///
    /// Observation only: `false` is **not** an error — it means the
    /// checkpoint uses a naming scheme this repo has not transcribed,
    /// not that the checkpoint is invalid.
    #[must_use]
    pub fn has_patch_embed(&self) -> bool {
        self.count_with_prefix(PATCH_EMBED_PREFIX) > 0
    }

    /// Encoder depth **as observed from the manifest**: one past the
    /// largest `<i>` seen in a `blocks.<i>.…` tensor name, or `None`
    /// when no such tensor exists.
    ///
    /// This is deliberately derived from data on disk rather than from
    /// the stamped `vokra.eat.depth`, so the two can be *compared*.
    /// `None` is a normal outcome for a checkpoint whose state-dict uses
    /// a different prefix; callers must treat it as "unknown", never as
    /// "zero layers".
    #[must_use]
    pub fn observed_block_count(&self) -> Option<u32> {
        let mut max_idx: Option<u32> = None;
        for (name, _) in &self.tensors {
            let Some(rest) = name.strip_prefix(BLOCK_PREFIX) else {
                continue;
            };
            let Ok(idx) = rest.split('.').next().unwrap_or("").parse::<u32>() else {
                continue;
            };
            max_idx = Some(max_idx.map_or(idx, |m: u32| m.max(idx)));
        }
        max_idx.map(|m| m + 1)
    }

    /// Looks up `name`, failing loud when it is absent.
    ///
    /// The error names the missing tensor and lists up to five sibling
    /// names sharing its first dotted segment (or, failing that, the
    /// first five names on disk) so a reader diagnosing a manifest
    /// mismatch can see what the artifact *does* contain without
    /// dumping the whole GGUF.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the missing tensor.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        if let Some(dims) = self.tensor_dims(name) {
            return Ok(dims);
        }
        let segment = name.split('.').next().unwrap_or(name);
        let mut near: Vec<&str> = self
            .tensors
            .iter()
            .filter(|(n, _)| n.starts_with(segment))
            .map(|(n, _)| n.as_str())
            .take(5)
            .collect();
        if near.is_empty() {
            near = self
                .tensors
                .iter()
                .map(|(n, _)| n.as_str())
                .take(5)
                .collect();
        }
        Err(VokraError::ModelLoad(format!(
            "eat: required tensor `{name}` is absent from the GGUF ({count} tensors \
             present; nearest names on disk: {near:?}). The converter passes upstream \
             safetensors names through verbatim, so a mismatch means either the \
             checkpoint was flattened with a different prefix convention or the \
             `EatVitTensorNames` manifest in hand was written for a different EAT size \
             point (`eat-base` vs `eat-large`). Refusing to substitute a zero tensor \
             (FR-EX-08). Primary sources: {UPSTREAM_URL} + {PRIMARY_SOURCE_PAPER}",
            count = self.tensors.len(),
        )))
    }

    /// Looks up `name` and checks its dimensions against `expected`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the tensor when it is absent
    ///   (via [`Self::require_tensor`]).
    /// - [`VokraError::ModelLoad`] naming the tensor plus **both** the
    ///   expected and the actual dims on a shape mismatch — never a
    ///   silent reshape or truncation (FR-EX-08).
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<()> {
        let actual = self.require_tensor(name)?;
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "eat: tensor `{name}` has dims {actual:?} but the stamped `vokra.eat.*` \
                 axes imply {expected:?} — refusing to reshape or truncate silently \
                 (FR-EX-08). Either the GGUF was produced from a different EAT size point \
                 (`eat-base` vs `eat-large`) or the `EatVitTensorNames` manifest points at \
                 the wrong tensor. Primary sources: {UPSTREAM_URL} + {PRIMARY_SOURCE_PAPER}"
            )));
        }
        Ok(())
    }

    /// Decodes a tensor payload to `f32`, mapping a decode failure to a
    /// loud [`VokraError::ModelLoad`] naming the tensor.
    fn decode(&self, file: &GgufFile, name: &str) -> Result<Vec<f32>> {
        file.tensor_f32(name).map_err(|e| {
            VokraError::ModelLoad(format!(
                "eat: tensor `{name}` is listed in the manifest but its payload failed to \
                 decode: {e}"
            ))
        })
    }

    /// Shape-gates `name` against `dims`, then decodes it.
    fn read_dims(&self, file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        self.require_tensor_dims(name, dims)?;
        self.decode(file, name)
    }

    /// Optional-slot form of [`Self::read_dims`].
    fn read_dims_opt(
        &self,
        file: &GgufFile,
        name: Option<&String>,
        dims: &[usize],
    ) -> Result<Option<Vec<f32>>> {
        match name {
            Some(n) => Ok(Some(self.read_dims(file, n, dims)?)),
            None => Ok(None),
        }
    }

    /// Decodes `name` and checks its **element count** rather than its
    /// dims, for slots whose leading singleton axes are a checkpoint
    /// convention rather than a fact the stamped axes settle.
    fn read_elems(&self, file: &GgufFile, name: &str, want: usize, why: &str) -> Result<Vec<f32>> {
        let dims = self.require_tensor(name)?;
        let values = self.decode(file, name)?;
        if values.len() != want {
            return Err(VokraError::ModelLoad(format!(
                "eat: tensor `{name}` decodes to {got} element(s) but the stamped \
                 `vokra.eat.*` axes imply {want} ({why}; on-disk dims {dims:?}). Refusing \
                 to pad or truncate silently (FR-EX-08). Primary source: {UPSTREAM_URL}",
                got = values.len(),
            )));
        }
        Ok(values)
    }

    /// Decodes a positional table, checking only that it is a whole,
    /// non-zero number of `embed_dim`-wide rows. Whether that row count
    /// is usable is the [`PosEmbedPolicy`]'s decision.
    fn read_rows(&self, file: &GgufFile, name: &str, embed_dim: usize) -> Result<Vec<f32>> {
        let dims = self.require_tensor(name)?;
        let values = self.decode(file, name)?;
        if values.is_empty() || values.len() % embed_dim != 0 {
            return Err(VokraError::ModelLoad(format!(
                "eat: positional table `{name}` decodes to {got} element(s), which is not a \
                 non-zero multiple of embed_dim {embed_dim} (on-disk dims {dims:?}). A \
                 positional table must be a whole number of rows (FR-EX-08). Primary \
                 source: {UPSTREAM_URL}",
                got = values.len(),
            )));
        }
        Ok(values)
    }
}

// ---------------------------------------------------------------------------
// Eat — the runtime binder handle.
// ---------------------------------------------------------------------------

/// EAT (`cwx-worst-one/EAT`, MIT) self-supervised audio-encoder runtime
/// binder.
///
/// Bind with [`from_gguf`](Self::from_gguf). The stamped topology is
/// then available through [`config`](Self::config), the ViT axis set
/// through [`EatConfig::to_vit_attrs`], and a real encoder through
/// [`bind_vit_encoder`](Self::bind_vit_encoder) once the caller supplies
/// an [`EatVitTensorNames`]. [`encode`](Self::encode) and
/// [`embed_utterance`](Self::embed_utterance) — the PCM-in surfaces —
/// remain loud-partial; see [`encode`](Self::encode) for exactly what
/// still blocks them.
///
/// This is a **feature extractor**: it exposes representations only.
/// The upstream downstream task heads (AudioSet tagging, ESC-50,
/// SPC-2) ship as separate fine-tunes and are not part of the
/// checkpoint this converter targets.
#[derive(Debug)]
pub struct Eat {
    config: EatConfig,
    weights: EatWeights,
    weight_license: LicenseClass,
}

impl Eat {
    /// Binds an EAT GGUF: verifies arch strictly, cross-checks the
    /// category stamp, discovers the tensor manifest, reads the full
    /// `vokra.eat.*` axis group, and surfaces the stamped weight-license
    /// class for the compliance-gate cross-checks (FR-CP-03).
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key so a reader diagnosing a mis-produced GGUF
    /// has exactly one place to walk (FR-EX-08 — never a silent partial
    /// bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is not
    ///   `"eat"` — a sibling SSL audio-encoder GGUF (`beats` /
    ///   `dasheng` / `atst` / `m2d` / `mert` / `muq` / `ast` /
    ///   `hubert`) handed here by mistake fails with a message naming
    ///   both tags instead of a downstream missing-tensor error.
    /// - [`VokraError::ModelLoad`] when `vokra.model.category` is
    ///   present but disagrees with [`CATEGORY`].
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   (via [`EatWeights::from_gguf`]).
    /// - [`VokraError::ModelLoad`] naming any absent `vokra.eat.*` key
    ///   (via [`EatConfig::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first, so a mis-typed model handed here
        //    fails with a specific message rather than a downstream
        //    missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "eat: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model eat`? Note that the sibling \
                     SSL audio-encoder arch tags — `beats` (iterative acoustic-tokenizer \
                     SSL), `dasheng` (universal MAE), `atst` (teacher-student patchout), \
                     `m2d` (masked-modeling duo), `mert` / `muq` (music-domain SSL), \
                     `ast` (supervised audio spectrogram Transformer, not \
                     self-supervised) and `hubert` (masked cluster prediction over raw \
                     waveform) all live in the same neighbourhood but are distinct \
                     topologies. EAT's utterance-level Transformer trained with inverse \
                     block masking has no analog among them, so binding one manifest \
                     with another's loader would produce shape-valid garbage instead of \
                     a loud error — FR-EX-08, no silent partial load.)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "eat: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native eat GGUF (was it produced by `vokra-cli convert \
                     --model eat`?)"
                        .to_owned(),
                ));
            }
        }

        // 2. Category cross-check. The converter ALWAYS stamps
        //    `audio-embedding`, so a disagreeing value signals a
        //    hand-edited or mis-produced artifact and must not pass
        //    silently. Absence is tolerated: hand-assembled fixtures
        //    need not carry the full chunk set (same tolerance the
        //    sibling binders extend to the provenance stamp).
        if let Some(cat) = file.get(GGUF_KEY_MODEL_CATEGORY).and_then(|v| v.as_str())
            && cat != CATEGORY
        {
            return Err(VokraError::ModelLoad(format!(
                "eat: GGUF `{GGUF_KEY_MODEL_CATEGORY}` is `{cat}`, expected \
                 `{CATEGORY}` — the converter stamps `{CATEGORY}` unconditionally, so a \
                 disagreeing value means a hand-edited or mis-produced artifact. \
                 Refusing to advertise an audio-embedding encoder under a foreign \
                 category (FR-EX-08); the model-card generator and the zoo-manifest \
                 tier gate both key off this value."
            )));
        }

        // 3. Tensor manifest with the non-emptiness gate. Ordered before
        //    the axis group so an artifact carrying neither reports the
        //    more fundamental problem first.
        let weights = EatWeights::from_gguf(file)?;

        // 4. The `vokra.eat.*` topology group, strictly.
        let config = EatConfig::from_gguf(file)?;

        // 5. Provenance surfacing — read the stamped weight-license
        //    class for the compliance-gate cross-checks. The EAT
        //    converter stamps `Permissive` (MIT) by default; a GGUF
        //    missing the stamp reads back as `Unknown`, which is
        //    fail-closed at the M2-13 gate (memory
        //    `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            weights,
            weight_license,
        })
    }

    /// The stamped `vokra.eat.*` topology and front-end axes.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &EatConfig {
        &self.config
    }

    /// The bound tensor manifest, for callers that need loud slot
    /// lookups ([`EatWeights::require_tensor`] /
    /// [`EatWeights::require_tensor_dims`]).
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &EatWeights {
        &self.weights
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk.
    ///
    /// The EAT converter stamps [`LicenseClass::Permissive`] by default
    /// (`mit`); a GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Encoder depth as observed from the tensor manifest, or `None`
    /// when the checkpoint's naming scheme carries no
    /// [`BLOCK_PREFIX`] tensors. See
    /// [`EatWeights::observed_block_count`] for why `None` is a normal
    /// outcome and must never be read as "zero layers".
    #[inline]
    #[must_use]
    pub fn observed_block_count(&self) -> Option<u32> {
        self.weights.observed_block_count()
    }

    /// Decodes a [`vokra_ops::vit::ViTWeights`] out of the GGUF, using
    /// `names` for every slot and `attrs` for every expected shape.
    ///
    /// Each slot goes through [`EatWeights::require_tensor`] /
    /// [`EatWeights::require_tensor_dims`], so an absent or
    /// wrong-shaped tensor fails loud naming itself and, on a shape
    /// mismatch, naming both the expected and the actual dims. Nothing
    /// is reshaped, padded or substituted (FR-EX-08).
    ///
    /// Two slots are validated by **element count** rather than dims,
    /// because their leading singleton axes are a checkpoint convention
    /// the stamped axes do not settle: the patch projection (`[D, 1, P,
    /// P]` and `[D, P*P]` carry identical row-major payloads) and the
    /// prepended-token parameter. The positional table is validated only
    /// as a whole number of rows — how many rows are *usable* is the
    /// [`PosEmbedPolicy`] in `attrs` deciding, inside
    /// [`ViTEncoder`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] propagated from
    ///   [`ViTAttrs::validate`].
    /// - [`VokraError::ModelLoad`] when `names.blocks.len()` disagrees
    ///   with `attrs.depth`, when the prepended-token slot disagrees
    ///   with `attrs.n_prepended_tokens`, or when any tensor is absent,
    ///   wrong-shaped, or fails to decode.
    pub fn bind_vit_weights(
        &self,
        file: &GgufFile,
        names: &EatVitTensorNames,
        attrs: &ViTAttrs,
    ) -> Result<ViTWeights> {
        attrs.validate()?;
        let w = &self.weights;
        let d = attrs.embed_dim;
        let mlp_dim = attrs.mlp_dim();
        let patch_len = attrs.patch_h * attrs.patch_w;
        let n_prepended = attrs.n_prepended_tokens;

        if names.blocks.len() != attrs.depth {
            return Err(VokraError::ModelLoad(format!(
                "eat: the supplied `EatVitTensorNames` carries {got} block name set(s) but \
                 the stamped `{GGUF_KEY_DEPTH}` is {depth}. Refusing to bind a stack of a \
                 different depth than the artifact declares (FR-EX-08).",
                got = names.blocks.len(),
                depth = attrs.depth,
            )));
        }

        let proj_w = w.read_elems(
            file,
            &names.patch_embed_weight,
            d * patch_len,
            "embed_dim * patch_h * patch_w, i.e. a Conv2d [D, 1, P, P] or a flattened \
             [D, P*P]",
        )?;
        let proj_b = w.read_dims_opt(file, names.patch_embed_bias.as_ref(), &[d])?;

        let prepended_tokens = match (&names.prepended_tokens, n_prepended) {
            (Some(name), k) if k > 0 => w.read_elems(
                file,
                name,
                k * d,
                "num_extra_tokens * embed_dim, i.e. the learned CLS parameter",
            )?,
            (Some(_), _) => {
                return Err(VokraError::ModelLoad(format!(
                    "eat: the supplied `EatVitTensorNames` names a prepended-token tensor \
                     but `{GGUF_KEY_NUM_EXTRA_TOKENS}` is stamped 0, so the encoder \
                     prepends nothing. Refusing to bind a token the topology says does not \
                     exist (FR-EX-08)."
                )));
            }
            (None, 0) => Vec::new(),
            (None, k) => {
                return Err(VokraError::ModelLoad(format!(
                    "eat: `{GGUF_KEY_NUM_EXTRA_TOKENS}` is stamped {k}, so the encoder \
                     prepends {k} learned token(s), but the supplied \
                     `EatVitTensorNames::prepended_tokens` is `None`. Refusing to \
                     substitute zeros for a learned parameter (FR-EX-08)."
                )));
            }
        };

        let pos_embed = w.read_rows(file, &names.pos_embed, d)?;

        let mut blocks = Vec::with_capacity(attrs.depth);
        for names_i in &names.blocks {
            let ln1_gamma = w.read_dims(file, &names_i.norm1_weight, &[d])?;
            let ln1_beta = w.read_dims(file, &names_i.norm1_bias, &[d])?;
            let ln2_gamma = w.read_dims(file, &names_i.norm2_weight, &[d])?;
            let ln2_beta = w.read_dims(file, &names_i.norm2_bias, &[d])?;

            let wo = w.read_dims(file, &names_i.proj_weight, &[d, d])?;
            let bo = w.read_dims_opt(file, names_i.proj_bias.as_ref(), &[d])?;

            // Fused QKV, split into thirds in q, k, v row order — the
            // contract documented on `EatVitBlockTensorNames`.
            let qkv = w.read_dims(file, &names_i.qkv_weight, &[3 * d, d])?;
            let (q_rows, rest) = qkv.split_at(d * d);
            let (k_rows, v_rows) = rest.split_at(d * d);
            let (bq, bk, bv) = match &names_i.qkv_bias {
                Some(name) => {
                    let bias = w.read_dims(file, name, &[3 * d])?;
                    (
                        Some(bias[..d].to_vec()),
                        Some(bias[d..2 * d].to_vec()),
                        Some(bias[2 * d..].to_vec()),
                    )
                }
                None => (None, None, None),
            };

            let w1 = w.read_dims(file, &names_i.fc1_weight, &[mlp_dim, d])?;
            let b1 = w.read_dims_opt(file, names_i.fc1_bias.as_ref(), &[mlp_dim])?;
            let w2 = w.read_dims(file, &names_i.fc2_weight, &[d, mlp_dim])?;
            let b2 = w.read_dims_opt(file, names_i.fc2_bias.as_ref(), &[d])?;

            blocks.push(ViTBlockWeights {
                ln1_gamma,
                ln1_beta,
                attn: ViTAttnWeights {
                    wq: q_rows.to_vec(),
                    bq,
                    wk: k_rows.to_vec(),
                    bk,
                    wv: v_rows.to_vec(),
                    bv,
                    wo,
                    bo,
                },
                ln2_gamma,
                ln2_beta,
                mlp: ViTMlpWeights { w1, b1, w2, b2 },
            });
        }

        Ok(ViTWeights {
            patch_embed: PatchEmbedWeights { proj_w, proj_b },
            prepended_tokens,
            pos_embed,
            blocks,
            final_ln_gamma: w.read_dims(file, &names.final_norm_weight, &[d])?,
            final_ln_beta: w.read_dims(file, &names.final_norm_bias, &[d])?,
        })
    }

    /// Maps the stamped axes onto [`ViTAttrs`], binds the weights, and
    /// builds a ready-to-run [`ViTEncoder`].
    ///
    /// `gelu` and `pos_embed_policy` are caller-supplied because the
    /// `vokra.eat.*` group stamps neither — see
    /// [`EatConfig::to_vit_attrs`] for why inventing them here would be
    /// silently wrong.
    ///
    /// # This does not make [`Self::encode`] real
    ///
    /// The returned encoder runs the patch stem and the pre-norm stack
    /// over a mel plane the **caller** supplies. It carries no claim of
    /// parity with upstream EAT: the norm-order flag is still
    /// unreconciled, and the caller vouched for `names`. Those are two
    /// of the three blockers [`Self::encode`] names; the third is the
    /// front-end, which this entry point does not touch at all because
    /// it takes features, not PCM.
    ///
    /// # Errors
    ///
    /// - Propagates [`EatConfig::to_vit_attrs`] and
    ///   [`Self::bind_vit_weights`].
    /// - [`VokraError::InvalidArgument`] from [`ViTEncoder::new`] when a
    ///   bound buffer is non-finite or the positional table cannot be
    ///   applied under `pos_embed_policy`.
    pub fn bind_vit_encoder(
        &self,
        file: &GgufFile,
        names: &EatVitTensorNames,
        gelu: GeluKind,
        pos_embed_policy: PosEmbedPolicy,
    ) -> Result<ViTEncoder> {
        let attrs = self.config.to_vit_attrs(gelu, pos_embed_policy)?;
        let weights = self.bind_vit_weights(file, names, &attrs)?;
        ViTEncoder::new(attrs, weights)
    }

    /// Encodes a PCM waveform into the sequence of per-patch encoder
    /// hidden states, shaped `[n_patches][hidden]`.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. The config half of this
    /// binder is now real — the `vokra.eat.*` group is read strictly and
    /// [`EatConfig::to_vit_attrs`] maps it onto
    /// [`vokra_ops::vit::ViTAttrs`] — and `vokra-ops` now supplies both
    /// the 2-D patch embedding and the pre-norm Transformer stack this
    /// module used to lack. Three things still stand between **PCM** and
    /// a defensible representation:
    ///
    /// 1. **No verified tensor-name manifest.** The converter passes
    ///    upstream safetensors names through verbatim and nothing
    ///    in-repo transcribes them; the only EAT tensor names present
    ///    anywhere are the converter's own round-trip fixtures, which
    ///    that test labels merely "realistic". EAT descends from
    ///    fairseq data2vec2, whose modality-specific parameters live
    ///    under a `modality_encoders.*` tree instead. A caller that can
    ///    defend a naming supplies it as [`EatVitTensorNames`] and calls
    ///    [`Self::bind_vit_encoder`]; this surface has no such manifest
    ///    to reach for.
    /// 2. **The Kaldi-fbank window.** The front-end arguments are
    ///    stamped in full, and `vokra.eat.fbank_window_type` is
    ///    `"hanning"`, but `vokra_ops::kaldi_fbank` hard-codes the Povey
    ///    window (Hann^0.85) and exposes no selector. Two different
    ///    windows desync every feature, and the stamp is what makes the
    ///    mismatch detectable instead of invisible.
    /// 3. **Norm order unreconciled.** `vokra.eat.layer_norm_first` is
    ///    stamped `false`, recorded by the converter as a transcribed
    ///    config value and explicitly *not* as an assertion about where
    ///    the norms sit. [`vokra_ops::vit::ViTEncoder`] is pre-norm by
    ///    construction and its docs call post-norm "a different function
    ///    whose outputs are shape-valid and numerically wrong". Upstream
    ///    `models/modules.py AltBlock` settles it and is not transcribed
    ///    in-repo.
    ///
    /// The message additionally reports the mapped axes and what the
    /// manifest actually contains, so the follow-up wave can see whether
    /// the checkpoint in hand matches. **No fabricated hidden states are
    /// ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred PCM-in forward.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind explicitly so a future accidental removal of the
        // parameter cannot hide behind an unused-variable warning; the
        // real implementation will consume it.
        let _ = pcm;
        Err(forward_loud_partial(
            "eat encode",
            None,
            &self.config,
            &self.weights,
        ))
    }

    /// Encodes a PCM waveform into a single utterance-level embedding.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. In addition to the three
    /// blockers [`Self::encode`] lists, the read-out convention itself
    /// is un-transcribed: [`vokra_ops::vit::ViTPooling`] can now express
    /// either form and the embedding width **is** stamped, but which
    /// form EAT's utterance-level objective uses — and at which index
    /// the CLS token sits — is not recorded anywhere in-repo. The
    /// converter stamps `num_extra_tokens` and explicitly declines to
    /// stamp the token's position. **No fabricated embedding is ever
    /// emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred PCM-in forward plus the un-transcribed read-out.
    pub fn embed_utterance(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(forward_loud_partial(
            "eat embed_utterance",
            Some(
                "(iv) the utterance-level READ-OUT convention — `vokra_ops::vit::ViTPooling` \
                 can now express either form (a prepended-token read-out or the mean over \
                 patch tokens) and the embedding width IS stamped as \
                 `vokra.eat.embed_dim`, but WHICH form EAT's utterance-level objective uses, \
                 and at which index the CLS token sits, are not transcribed in-repo: the \
                 converter stamps `vokra.eat.num_extra_tokens` and explicitly declines to \
                 stamp the token's POSITION, leaving it to be read off upstream \
                 `models/images.py`.",
            ),
            &self.config,
            &self.weights,
        ))
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Eat::encode`] and [`Eat::embed_utterance`].
///
/// `surface` names the calling method; `extra_piece` appends a
/// surface-specific blocker (the utterance read-out) when present.
///
/// Every claim in the message is checked against the tree it describes:
/// the axis group it says is stamped is the one [`EatConfig::from_gguf`]
/// reads, the window it says is hard-coded is
/// `crates/vokra-ops/src/kaldi_fbank.rs`, and the pre-norm construction
/// it cites is `crates/vokra-ops/src/vit.rs`. An error message that
/// still named a resolved blocker would mislead the next reader, which
/// is the failure this rewrite exists to remove (CLAUDE.md 教訓 (a):
/// "loud-partial は fake-complete より honest" — but only while every
/// clause of the loud part is still true).
fn forward_loud_partial(
    surface: &str,
    extra_piece: Option<&str>,
    cfg: &EatConfig,
    weights: &EatWeights,
) -> VokraError {
    let blocks = weights.observed_block_count().map_or_else(
        || "unknown (no `blocks.<i>.` tensors on disk)".to_owned(),
        |n| n.to_string(),
    );
    let axes = format!(
        "embed_dim={ed}, depth={dp}, n_heads={nh}, patch {ps}x{ps}, grid freq={gf} x \
         time={gt} = {np} patch token(s) + {ex} prepended = {tot} token(s)",
        ed = cfg.embed_dim,
        dp = cfg.depth,
        nh = cfg.num_heads,
        ps = cfg.patch_size,
        gf = cfg.patch_grid_freq,
        gt = cfg.patch_grid_time,
        np = cfg.num_patches,
        ex = cfg.num_extra_tokens,
        tot = cfg.tokens_per_clip(),
    );
    let extra = extra_piece.unwrap_or("");
    VokraError::UnsupportedOp(format!(
        "{surface} (loud-partial): the CONFIG half of this binder is real — the \
         `vokra.eat.*` group is read strictly and maps onto `vokra_ops::vit::ViTAttrs` as \
         {axes} — and `vokra_ops::vit::ViTEncoder` now supplies the 2-D patch embedding and \
         the pre-norm Transformer stack this module used to lack. Three things still stand \
         between PCM and a defensible representation. \
         (i) NO VERIFIED TENSOR-NAME MANIFEST: the converter passes upstream safetensors \
         names through verbatim and nothing in-repo transcribes them — the only EAT tensor \
         names present anywhere are the converter's own round-trip fixtures \
         (`{PATCH_EMBED_PREFIX}proj.weight`, `{BLOCK_PREFIX}0.attn.qkv.weight`), which that \
         test labels merely 'realistic', and EAT descends from fairseq data2vec2 \
         (`Data2VecMultiConfig`), whose modality-specific parameters live under a \
         `modality_encoders.*` tree instead. A caller that can defend a naming supplies it \
         as `EatVitTensorNames` and calls `Eat::bind_vit_encoder` — that path IS real and \
         shape-gated against the stamped axes — but this PCM-in surface has no such \
         manifest to reach for and will not guess one. \
         (ii) THE KALDI-FBANK WINDOW: the front-end arguments ARE stamped in full, and \
         `{GGUF_KEY_FBANK_WINDOW_TYPE}` is `{window}`, but the checked \
         `KaldiFbankWindow` selector exposes Povey and Hamming, not Hanning, so every \
         feature would desync. The stamp makes this detectable rather than invisible. \
         (iii) NORM ORDER UNRECONCILED: `{GGUF_KEY_LAYER_NORM_FIRST}` is stamped \
         `{lnf}` as a transcribed config value, explicitly NOT as an assertion about where \
         the norms sit; `vokra_ops::vit::ViTEncoder` is pre-norm BY CONSTRUCTION and its own \
         docs call post-norm 'a different function whose outputs are shape-valid and \
         numerically wrong'. Upstream `models/modules.py AltBlock` settles this and is not \
         transcribed in-repo. \
         {extra} \
         Two axes `ViTAttrs` needs are also absent from the stamped group and are therefore \
         parameters of `EatConfig::to_vit_attrs` rather than invented here: the GELU flavour \
         and the positional-embedding policy (the sibling `atst` binder derives both from \
         metadata because its converter stamps `act_layer` and `pos_type`; EAT's stamps \
         neither). \
         Observed on disk: tensor_count={count}, observed_block_count={blocks}, \
         has_patch_embed={patch_present}. \
         Primary sources: {UPSTREAM_URL} + {PRIMARY_SOURCE_PAPER} (arch={ARCH}, \
         name={NAME}, category={CATEGORY}). Runtime cannot fabricate hidden states or \
         an embedding (FR-EX-08 — no silent partial output).",
        window = cfg.fbank_window_type,
        lnf = cfg.layer_norm_first,
        count = weights.tensor_count(),
        patch_present = weights.has_patch_embed(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the EAT runtime binder — contract-constant pins,
    //! strict axis-group round-trip, manifest observation, real ViT
    //! binding, and negative-space round-trip on every loud gate.
    //!
    //! # What is and is not asserted
    //!
    //! The ViT binding tests run a real forward over a **synthetic**
    //! checkpoint and assert only mechanics: expected shape, finiteness
    //! and determinism. They deliberately assert **no** numerical parity
    //! against upstream EAT — there is no reference dump in-repo, and
    //! inventing an expected value would be fabrication (CLAUDE.md
    //! 教訓 (a)). Parity belongs to the wave that lands a real
    //! checkpoint plus a `tools/parity/` reference.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Every `vokra.eat.*` key the converter stamps, in converter order.
    /// The count is pinned so a converter-side addition that this reader
    /// does not learn about fails here.
    const AXIS_KEYS: [&str; 38] = [
        GGUF_KEY_EMBED_DIM,
        GGUF_KEY_DEPTH,
        GGUF_KEY_NUM_HEADS,
        GGUF_KEY_MLP_RATIO,
        GGUF_KEY_NORM_EPS,
        GGUF_KEY_LAYER_NORM_FIRST,
        GGUF_KEY_PATCH_SIZE,
        GGUF_KEY_IN_CHANS,
        GGUF_KEY_TARGET_LENGTH,
        GGUF_KEY_N_MELS,
        GGUF_KEY_PATCH_GRID_TIME,
        GGUF_KEY_PATCH_GRID_FREQ,
        GGUF_KEY_NUM_PATCHES,
        GGUF_KEY_NUM_EXTRA_TOKENS,
        GGUF_KEY_POS_EMBED_MAX_LENGTH,
        GGUF_KEY_DECODER_DIM,
        GGUF_KEY_DECODER_GROUPS,
        GGUF_KEY_DECODER_KERNEL,
        GGUF_KEY_DECODER_LAYERS,
        GGUF_KEY_FBANK_SAMPLE_RATE,
        GGUF_KEY_FBANK_FRAME_LENGTH_MS,
        GGUF_KEY_FBANK_FRAME_SHIFT_MS,
        GGUF_KEY_FBANK_WINDOW_TYPE,
        GGUF_KEY_FBANK_HTK_COMPAT,
        GGUF_KEY_FBANK_USE_ENERGY,
        GGUF_KEY_FBANK_DITHER,
        GGUF_KEY_FBANK_LOW_FREQ,
        GGUF_KEY_FBANK_HIGH_FREQ,
        GGUF_KEY_FBANK_PREEMPH_COEFF,
        GGUF_KEY_FBANK_REMOVE_DC_OFFSET,
        GGUF_KEY_FBANK_ROUND_TO_POWER_OF_TWO,
        GGUF_KEY_FBANK_SNIP_EDGES,
        GGUF_KEY_FBANK_USE_POWER,
        GGUF_KEY_FBANK_USE_LOG,
        GGUF_KEY_FBANK_SUBTRACT_MEAN,
        GGUF_KEY_FBANK_NORM_MEAN,
        GGUF_KEY_FBANK_NORM_STD,
        GGUF_KEY_FBANK_NORM_STD_MULTIPLIER,
    ];

    /// Deterministic uniform-ish source in `[-1, 1)`, so no fixture
    /// bytes are committed and every synthetic weight reproduces on
    /// every platform (the `vokra_ops::vit` test-module convention).
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / 8_388_608.0 - 1.0
        }

        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    fn put_u32(b: &mut GgufBuilder, skip: Option<&str>, key: &str, value: u32) {
        if skip != Some(key) {
            b.add_u32(key, value);
        }
    }

    fn put_f32(b: &mut GgufBuilder, skip: Option<&str>, key: &str, value: f32) {
        if skip != Some(key) {
            b.add_f32(key, value);
        }
    }

    fn put_bool(b: &mut GgufBuilder, skip: Option<&str>, key: &str, value: bool) {
        if skip != Some(key) {
            b.add_bool(key, value);
        }
    }

    fn put_str(b: &mut GgufBuilder, skip: Option<&str>, key: &str, value: &str) {
        if skip != Some(key) {
            b.add_string(key, value);
        }
    }

    /// Stamps the whole `vokra.eat.*` group from `cfg`, optionally
    /// omitting exactly one key so the strict reader can be probed.
    fn stamp_axes(b: &mut GgufBuilder, cfg: &EatConfig, skip: Option<&str>) {
        put_u32(b, skip, GGUF_KEY_EMBED_DIM, cfg.embed_dim);
        put_u32(b, skip, GGUF_KEY_DEPTH, cfg.depth);
        put_u32(b, skip, GGUF_KEY_NUM_HEADS, cfg.num_heads);
        put_f32(b, skip, GGUF_KEY_MLP_RATIO, cfg.mlp_ratio);
        put_f32(b, skip, GGUF_KEY_NORM_EPS, cfg.norm_eps);
        put_bool(b, skip, GGUF_KEY_LAYER_NORM_FIRST, cfg.layer_norm_first);
        put_u32(b, skip, GGUF_KEY_PATCH_SIZE, cfg.patch_size);
        put_u32(b, skip, GGUF_KEY_IN_CHANS, cfg.in_chans);
        put_u32(b, skip, GGUF_KEY_TARGET_LENGTH, cfg.target_length);
        put_u32(b, skip, GGUF_KEY_N_MELS, cfg.n_mels);
        put_u32(b, skip, GGUF_KEY_PATCH_GRID_TIME, cfg.patch_grid_time);
        put_u32(b, skip, GGUF_KEY_PATCH_GRID_FREQ, cfg.patch_grid_freq);
        put_u32(b, skip, GGUF_KEY_NUM_PATCHES, cfg.num_patches);
        put_u32(b, skip, GGUF_KEY_NUM_EXTRA_TOKENS, cfg.num_extra_tokens);
        put_u32(
            b,
            skip,
            GGUF_KEY_POS_EMBED_MAX_LENGTH,
            cfg.pos_embed_max_length,
        );
        put_u32(b, skip, GGUF_KEY_DECODER_DIM, cfg.decoder_dim);
        put_u32(b, skip, GGUF_KEY_DECODER_GROUPS, cfg.decoder_groups);
        put_u32(b, skip, GGUF_KEY_DECODER_KERNEL, cfg.decoder_kernel);
        put_u32(b, skip, GGUF_KEY_DECODER_LAYERS, cfg.decoder_layers);
        put_u32(b, skip, GGUF_KEY_FBANK_SAMPLE_RATE, cfg.fbank_sample_rate);
        put_u32(
            b,
            skip,
            GGUF_KEY_FBANK_FRAME_LENGTH_MS,
            cfg.fbank_frame_length_ms,
        );
        put_u32(
            b,
            skip,
            GGUF_KEY_FBANK_FRAME_SHIFT_MS,
            cfg.fbank_frame_shift_ms,
        );
        put_str(b, skip, GGUF_KEY_FBANK_WINDOW_TYPE, &cfg.fbank_window_type);
        put_bool(b, skip, GGUF_KEY_FBANK_HTK_COMPAT, cfg.fbank_htk_compat);
        put_bool(b, skip, GGUF_KEY_FBANK_USE_ENERGY, cfg.fbank_use_energy);
        put_f32(b, skip, GGUF_KEY_FBANK_DITHER, cfg.fbank_dither);
        put_f32(b, skip, GGUF_KEY_FBANK_LOW_FREQ, cfg.fbank_low_freq);
        put_f32(b, skip, GGUF_KEY_FBANK_HIGH_FREQ, cfg.fbank_high_freq);
        put_f32(
            b,
            skip,
            GGUF_KEY_FBANK_PREEMPH_COEFF,
            cfg.fbank_preemph_coeff,
        );
        put_bool(
            b,
            skip,
            GGUF_KEY_FBANK_REMOVE_DC_OFFSET,
            cfg.fbank_remove_dc_offset,
        );
        put_bool(
            b,
            skip,
            GGUF_KEY_FBANK_ROUND_TO_POWER_OF_TWO,
            cfg.fbank_round_to_power_of_two,
        );
        put_bool(b, skip, GGUF_KEY_FBANK_SNIP_EDGES, cfg.fbank_snip_edges);
        put_bool(b, skip, GGUF_KEY_FBANK_USE_POWER, cfg.fbank_use_power);
        put_bool(b, skip, GGUF_KEY_FBANK_USE_LOG, cfg.fbank_use_log);
        put_bool(
            b,
            skip,
            GGUF_KEY_FBANK_SUBTRACT_MEAN,
            cfg.fbank_subtract_mean,
        );
        put_f32(b, skip, GGUF_KEY_FBANK_NORM_MEAN, cfg.fbank_norm_mean);
        put_f32(b, skip, GGUF_KEY_FBANK_NORM_STD, cfg.fbank_norm_std);
        put_f32(
            b,
            skip,
            GGUF_KEY_FBANK_NORM_STD_MULTIPLIER,
            cfg.fbank_norm_std_multiplier,
        );
    }

    /// A deliberately tiny but self-consistent topology, so a full ViT
    /// forward fits in a unit test. The fbank half is inherited from the
    /// transcribed `eat-base` reference because these tests never touch
    /// the front-end.
    ///
    /// Consistency check, mirroring the real `eat-base` arithmetic:
    /// `n_mels 4 / patch 2 = 2` freq patches, `target_length 6 / patch 2
    /// = 3` time patches, `2 * 3 = 6` patch tokens, `+ 1` prepended.
    fn tiny_config() -> EatConfig {
        EatConfig {
            embed_dim: 8,
            depth: 2,
            num_heads: 2,
            mlp_ratio: 2.0,
            patch_size: 2,
            target_length: 6,
            n_mels: 4,
            patch_grid_time: 3,
            patch_grid_freq: 2,
            num_patches: 6,
            num_extra_tokens: 1,
            pos_embed_max_length: 4,
            ..EatConfig::eat_base_reference()
        }
    }

    /// Builds a synthetic EAT GGUF carrying the arch / name / category
    /// stamps, the full axis group from `cfg`, an optional
    /// weight-license class, and the two representative tensor names the
    /// converter's own round-trip tests exercise across `n_blocks`
    /// encoder blocks.
    fn eat_gguf(weight_license_class: Option<LicenseClass>, n_blocks: u32) -> GgufFile {
        let cfg = EatConfig::eat_base_reference();
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        stamp_axes(&mut b, &cfg, None);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // The 2-D patch-embedding stem: a Conv2d weight is 4-D upstream;
        // dims here are placeholders, because these fixtures exercise the
        // manifest observers rather than the shape-gated ViT binding.
        // The `* 1` is the in-channel axis of `[2, 2, 1, 4]`, kept so the byte
        // count reads as the shape above it rather than as a folded constant
        // that no longer tracks it.
        #[allow(
            clippy::identity_op,
            reason = "the factor is a shape axis, not arithmetic padding"
        )]
        let patch_embed_bytes = vec![0u8; 2 * 2 * 1 * 4 * 4];
        b.add_tensor(
            "patch_embed.proj.weight",
            GgmlType::F32,
            vec![2, 2, 1, 4],
            patch_embed_bytes,
        )
        .expect("add_tensor patch_embed");
        for i in 0..n_blocks {
            b.add_tensor(
                &format!("{BLOCK_PREFIX}{i}.attn.qkv.weight"),
                GgmlType::F32,
                vec![12, 4],
                vec![0u8; 12 * 4 * 4],
            )
            .expect("add_tensor block");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// Appends an `F32` tensor whose payload is `values` in row-major
    /// order under `dims`.
    fn add_f32_tensor(b: &mut GgufBuilder, name: &str, dims: Vec<u64>, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        b.add_tensor(name, GgmlType::F32, dims, bytes)
            .expect("add_tensor");
    }

    /// A tensor-name manifest matching [`tiny_gguf`]'s fixtures.
    ///
    /// Test-local on purpose: shipping this as a production constructor
    /// would be exactly the guessed default `EatVitTensorNames` refuses
    /// to provide.
    fn tiny_names(depth: usize) -> EatVitTensorNames {
        let blocks: Vec<EatVitBlockTensorNames> = (0..depth)
            .map(|i| EatVitBlockTensorNames {
                norm1_weight: format!("blocks.{i}.norm1.weight"),
                norm1_bias: format!("blocks.{i}.norm1.bias"),
                qkv_weight: format!("blocks.{i}.attn.qkv.weight"),
                qkv_bias: Some(format!("blocks.{i}.attn.qkv.bias")),
                proj_weight: format!("blocks.{i}.attn.proj.weight"),
                proj_bias: Some(format!("blocks.{i}.attn.proj.bias")),
                norm2_weight: format!("blocks.{i}.norm2.weight"),
                norm2_bias: format!("blocks.{i}.norm2.bias"),
                fc1_weight: format!("blocks.{i}.mlp.fc1.weight"),
                fc1_bias: Some(format!("blocks.{i}.mlp.fc1.bias")),
                fc2_weight: format!("blocks.{i}.mlp.fc2.weight"),
                fc2_bias: Some(format!("blocks.{i}.mlp.fc2.bias")),
            })
            .collect();
        EatVitTensorNames {
            patch_embed_weight: "patch_embed.proj.weight".to_owned(),
            patch_embed_bias: Some("patch_embed.proj.bias".to_owned()),
            prepended_tokens: Some("cls_token".to_owned()),
            pos_embed: "pos_embed".to_owned(),
            blocks,
            final_norm_weight: "norm.weight".to_owned(),
            final_norm_bias: "norm.bias".to_owned(),
        }
    }

    /// Builds a GGUF whose payloads actually satisfy [`tiny_config`],
    /// so the ViT binding can be driven end to end.
    fn tiny_gguf(cfg: &EatConfig) -> GgufFile {
        let d = cfg.embed_dim as usize;
        let mlp = (cfg.embed_dim as f32 * cfg.mlp_ratio).round() as usize;
        let patch_len = (cfg.patch_size * cfg.patch_size) as usize;
        let n_tokens = cfg.tokens_per_clip();

        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        stamp_axes(&mut b, cfg, None);

        let dd = d as u64;
        add_f32_tensor(
            &mut b,
            "patch_embed.proj.weight",
            vec![dd, 1, u64::from(cfg.patch_size), u64::from(cfg.patch_size)],
            &rng.vec(d * patch_len),
        );
        add_f32_tensor(&mut b, "patch_embed.proj.bias", vec![dd], &rng.vec(d));
        add_f32_tensor(&mut b, "cls_token", vec![1, 1, dd], &rng.vec(d));
        add_f32_tensor(
            &mut b,
            "pos_embed",
            vec![1, n_tokens as u64, dd],
            &rng.vec(n_tokens * d),
        );
        for i in 0..cfg.depth {
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.norm1.weight"),
                vec![dd],
                &rng.vec(d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.norm1.bias"),
                vec![dd],
                &rng.vec(d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.attn.qkv.weight"),
                vec![3 * dd, dd],
                &rng.vec(3 * d * d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.attn.qkv.bias"),
                vec![3 * dd],
                &rng.vec(3 * d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.attn.proj.weight"),
                vec![dd, dd],
                &rng.vec(d * d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.attn.proj.bias"),
                vec![dd],
                &rng.vec(d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.norm2.weight"),
                vec![dd],
                &rng.vec(d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.norm2.bias"),
                vec![dd],
                &rng.vec(d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.mlp.fc1.weight"),
                vec![mlp as u64, dd],
                &rng.vec(mlp * d),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.mlp.fc1.bias"),
                vec![mlp as u64],
                &rng.vec(mlp),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.mlp.fc2.weight"),
                vec![dd, mlp as u64],
                &rng.vec(d * mlp),
            );
            add_f32_tensor(
                &mut b,
                &format!("blocks.{i}.mlp.fc2.bias"),
                vec![dd],
                &rng.vec(d),
            );
        }
        add_f32_tensor(&mut b, "norm.weight", vec![dd], &rng.vec(d));
        add_f32_tensor(&mut b, "norm.bias", vec![dd], &rng.vec(d));

        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Contract-constant pin (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        // Pinned against `crates/vokra-convert/src/models/eat.rs`. A
        // converter-side rename must land here in the same commit or
        // fail this test.
        assert_eq!(ARCH, "eat", "eat arch tag pin");
        assert_eq!(NAME, "eat-base", "canonical eat-base size-point pin");
        assert_eq!(
            CATEGORY, "audio-embedding",
            "EAT is an audio-embedding release, not ASR / TTS"
        );
        assert_eq!(
            UPSTREAM_URL, "github.com/cwx-worst-one/EAT",
            "EAT is not on HuggingFace — provenance rides `upstream_url`"
        );
        assert_eq!(DEFAULT_LICENSE_SPDX, "mit", "upstream SPDX pin");
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            GGUF_KEY_PROVENANCE_UPSTREAM_URL,
            "vokra.provenance.upstream_url"
        );
        assert_eq!(PRIMARY_SOURCE_PAPER, "arxiv.org/abs/2401.03497");
    }

    #[test]
    fn axis_key_spellings_mirror_the_converter() {
        // Byte-identical mirrors of the converter's private `KEY_EAT_*`
        // constants. `vokra-models` must not depend on `vokra-convert`,
        // so these spellings are duplicated; pinning them here makes a
        // rename on either side a same-commit failure.
        assert_eq!(GGUF_KEY_EMBED_DIM, "vokra.eat.embed_dim");
        assert_eq!(GGUF_KEY_DEPTH, "vokra.eat.depth");
        assert_eq!(GGUF_KEY_NUM_HEADS, "vokra.eat.num_heads");
        assert_eq!(GGUF_KEY_MLP_RATIO, "vokra.eat.mlp_ratio");
        assert_eq!(GGUF_KEY_NORM_EPS, "vokra.eat.norm_eps");
        assert_eq!(GGUF_KEY_LAYER_NORM_FIRST, "vokra.eat.layer_norm_first");
        assert_eq!(GGUF_KEY_PATCH_SIZE, "vokra.eat.patch_size");
        assert_eq!(GGUF_KEY_IN_CHANS, "vokra.eat.in_chans");
        assert_eq!(GGUF_KEY_TARGET_LENGTH, "vokra.eat.target_length");
        assert_eq!(GGUF_KEY_N_MELS, "vokra.eat.n_mels");
        assert_eq!(GGUF_KEY_PATCH_GRID_TIME, "vokra.eat.patch_grid_time");
        assert_eq!(GGUF_KEY_PATCH_GRID_FREQ, "vokra.eat.patch_grid_freq");
        assert_eq!(GGUF_KEY_NUM_PATCHES, "vokra.eat.num_patches");
        assert_eq!(GGUF_KEY_NUM_EXTRA_TOKENS, "vokra.eat.num_extra_tokens");
        assert_eq!(
            GGUF_KEY_POS_EMBED_MAX_LENGTH,
            "vokra.eat.pos_embed_max_length"
        );
        assert_eq!(GGUF_KEY_DECODER_DIM, "vokra.eat.decoder_dim");
        assert_eq!(GGUF_KEY_DECODER_GROUPS, "vokra.eat.decoder_groups");
        assert_eq!(GGUF_KEY_DECODER_KERNEL, "vokra.eat.decoder_kernel");
        assert_eq!(GGUF_KEY_DECODER_LAYERS, "vokra.eat.decoder_layers");
        assert_eq!(GGUF_KEY_FBANK_SAMPLE_RATE, "vokra.eat.fbank_sample_rate");
        assert_eq!(
            GGUF_KEY_FBANK_FRAME_LENGTH_MS,
            "vokra.eat.fbank_frame_length_ms"
        );
        assert_eq!(
            GGUF_KEY_FBANK_FRAME_SHIFT_MS,
            "vokra.eat.fbank_frame_shift_ms"
        );
        assert_eq!(GGUF_KEY_FBANK_WINDOW_TYPE, "vokra.eat.fbank_window_type");
        assert_eq!(GGUF_KEY_FBANK_HTK_COMPAT, "vokra.eat.fbank_htk_compat");
        assert_eq!(GGUF_KEY_FBANK_USE_ENERGY, "vokra.eat.fbank_use_energy");
        assert_eq!(GGUF_KEY_FBANK_DITHER, "vokra.eat.fbank_dither");
        assert_eq!(GGUF_KEY_FBANK_LOW_FREQ, "vokra.eat.fbank_low_freq");
        assert_eq!(GGUF_KEY_FBANK_HIGH_FREQ, "vokra.eat.fbank_high_freq");
        assert_eq!(
            GGUF_KEY_FBANK_PREEMPH_COEFF,
            "vokra.eat.fbank_preemph_coeff"
        );
        assert_eq!(
            GGUF_KEY_FBANK_REMOVE_DC_OFFSET,
            "vokra.eat.fbank_remove_dc_offset"
        );
        assert_eq!(
            GGUF_KEY_FBANK_ROUND_TO_POWER_OF_TWO,
            "vokra.eat.fbank_round_to_power_of_two"
        );
        assert_eq!(GGUF_KEY_FBANK_SNIP_EDGES, "vokra.eat.fbank_snip_edges");
        assert_eq!(GGUF_KEY_FBANK_USE_POWER, "vokra.eat.fbank_use_power");
        assert_eq!(GGUF_KEY_FBANK_USE_LOG, "vokra.eat.fbank_use_log");
        assert_eq!(
            GGUF_KEY_FBANK_SUBTRACT_MEAN,
            "vokra.eat.fbank_subtract_mean"
        );
        assert_eq!(GGUF_KEY_FBANK_NORM_MEAN, "vokra.eat.fbank_norm_mean");
        assert_eq!(GGUF_KEY_FBANK_NORM_STD, "vokra.eat.fbank_norm_std");
        assert_eq!(
            GGUF_KEY_FBANK_NORM_STD_MULTIPLIER,
            "vokra.eat.fbank_norm_std_multiplier"
        );
        assert_eq!(
            AXIS_KEYS.len(),
            38,
            "the converter stamps 38 `vokra.eat.*` keys; a change on either side must land \
             on both"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Arch distinctness pin — no collision with any sibling SSL
    //    audio-encoder arch tag
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_ssl_encoder_arches() {
        for sibling in [
            "beats", "dasheng", "atst", "m2d", "mert", "muq", "ast", "hubert",
        ] {
            assert_ne!(
                ARCH, sibling,
                "eat and {sibling} are distinct SSL audio-encoder topologies — sharing \
                 an arch tag would mis-route runtime dispatch (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. The stamped axis group round-trips field by field
    // -----------------------------------------------------------------------

    #[test]
    fn axis_group_round_trips_every_stamped_field() {
        let cfg = EatConfig::eat_base_reference();
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        stamp_axes(&mut b, &cfg, None);
        b.add_tensor("probe", GgmlType::F32, vec![1], vec![0u8; 4])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let read = EatConfig::from_gguf(&file).expect("the full axis group must parse");
        assert_eq!(read, cfg, "every stamped field must round-trip");

        // Spot-check the transcribed values against the converter's own
        // constants, so a silent edit on either side is caught by value
        // and not only by struct equality.
        assert_eq!(read.embed_dim, 768);
        assert_eq!(read.depth, 12);
        assert_eq!(read.num_heads, 12);
        assert!((read.mlp_ratio - 4.0).abs() < f32::EPSILON);
        assert!((read.norm_eps - 1e-6).abs() < f32::EPSILON);
        assert!(!read.layer_norm_first);
        assert_eq!(read.patch_size, 16);
        assert_eq!(read.in_chans, 1);
        assert_eq!(read.target_length, 1024);
        assert_eq!(read.n_mels, 128);
        assert_eq!(read.patch_grid_time, 64);
        assert_eq!(read.patch_grid_freq, 8);
        assert_eq!(read.num_patches, 512);
        assert_eq!(read.num_extra_tokens, 1);
        assert_eq!(read.pos_embed_max_length, 768);
        assert_eq!(read.fbank_window_type, "hanning");
        assert!(read.fbank_htk_compat);
        assert!(!read.fbank_use_energy);
        assert!((read.fbank_norm_mean - -4.268).abs() < 1e-6);
        assert!((read.fbank_norm_std - 4.569).abs() < 1e-6);
        assert!((read.fbank_norm_std_multiplier - 2.0).abs() < f32::EPSILON);

        // Derived helpers.
        assert_eq!(read.tokens_per_clip(), 1 + 512);
        assert_eq!(read.pos_embed_table_rows(), 1 + 768 * 8);
    }

    #[test]
    fn every_stamped_key_is_required_and_names_itself_when_absent() {
        let cfg = EatConfig::eat_base_reference();
        for key in AXIS_KEYS {
            let mut b = GgufBuilder::new();
            b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
            stamp_axes(&mut b, &cfg, Some(key));
            b.add_tensor("probe", GgmlType::F32, vec![1], vec![0u8; 4])
                .expect("add_tensor");
            let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

            let Err(err) = EatConfig::from_gguf(&file) else {
                panic!("expected a loud ModelLoad when `{key}` is absent");
            };
            match err {
                VokraError::ModelLoad(m) => {
                    assert!(
                        m.contains(key),
                        "message must NAME the missing key `{key}`, got `{m}`"
                    );
                    assert!(
                        m.contains("FR-EX-08"),
                        "message must cite the no-fallback clause for `{key}`, got `{m}`"
                    );
                }
                other => panic!("expected VokraError::ModelLoad for `{key}`, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // 4. The config maps onto ViTAttrs
    // -----------------------------------------------------------------------

    #[test]
    fn config_maps_onto_validated_vit_attrs() {
        let cfg = EatConfig::eat_base_reference();
        let attrs = cfg
            .to_vit_attrs(GeluKind::Erf, PosEmbedPolicy::RequireExact)
            .expect("the transcribed eat-base axes must map onto ViTAttrs");
        attrs.validate().expect("the mapped attrs must validate");

        assert_eq!(attrs.embed_dim, 768);
        assert_eq!(attrs.depth, 12);
        assert_eq!(attrs.n_heads, 12);
        assert_eq!(attrs.head_dim(), 64);
        assert_eq!(attrs.mlp_dim(), 3072);
        assert_eq!(attrs.patch_h, 16);
        assert_eq!(attrs.patch_w, 16);
        // Stride is DERIVED from the stamped grid, not stamped; the
        // mapping verifies it reproduces `patch_grid_freq` /
        // `patch_grid_time` before returning.
        assert_eq!(attrs.stride_h, 16);
        assert_eq!(attrs.stride_w, 16);
        assert_eq!(attrs.n_prepended_tokens, 1);
        assert_eq!(attrs.gelu, GeluKind::Erf);
        assert_eq!(attrs.pos_embed_policy, PosEmbedPolicy::RequireExact);

        // The tiny test topology must map too, or the binding tests
        // below would be exercising an unreachable configuration.
        let tiny = tiny_config()
            .to_vit_attrs(GeluKind::Tanh, PosEmbedPolicy::RequireExact)
            .expect("the tiny topology must map onto ViTAttrs");
        tiny.validate().expect("tiny attrs must validate");
        assert_eq!(tiny.mlp_dim(), 16);
    }

    #[test]
    fn to_vit_attrs_refuses_a_grid_that_the_derived_stride_cannot_reproduce() {
        // Overlapping patches would produce a denser grid than
        // non-overlapping tiling. The stride is not stamped anywhere, so
        // the mapping must refuse rather than guess one.
        let cfg = EatConfig {
            patch_grid_time: 127,
            num_patches: 127 * 8,
            ..EatConfig::eat_base_reference()
        };
        let Err(err) = cfg.to_vit_attrs(GeluKind::Erf, PosEmbedPolicy::RequireExact) else {
            panic!("expected a loud ModelLoad when the stamped grid implies a stride");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("overlapping"),
                    "message must name the overlapping-patch case, got `{m}`"
                );
                assert!(
                    m.contains("Refusing to guess a stride"),
                    "message must refuse to guess, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn to_vit_attrs_refuses_a_multi_channel_patch_stem() {
        let cfg = EatConfig {
            in_chans: 3,
            ..EatConfig::eat_base_reference()
        };
        let Err(err) = cfg.to_vit_attrs(GeluKind::Erf, PosEmbedPolicy::RequireExact) else {
            panic!("expected a loud ModelLoad for a multi-channel patch stem");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_IN_CHANS),
                    "message must name the offending key, got `{m}`"
                );
                assert!(
                    m.contains("single-channel"),
                    "message must explain the primitive's constraint, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5. Real ViT binding over a synthetic checkpoint
    // -----------------------------------------------------------------------

    #[test]
    fn bind_vit_encoder_runs_a_forward_over_a_synthetic_checkpoint() {
        // MECHANICS ONLY. There is no upstream reference in-repo, so
        // this asserts shape, finiteness and determinism — never a
        // numerical value, which would be fabricated.
        let cfg = tiny_config();
        let file = tiny_gguf(&cfg);
        let model = Eat::from_gguf(&file).expect("the synthetic checkpoint must bind");
        assert_eq!(model.config(), &cfg, "the stamped axes must round-trip");

        let names = tiny_names(cfg.depth as usize);
        let encoder = model
            .bind_vit_encoder(&file, &names, GeluKind::Erf, PosEmbedPolicy::RequireExact)
            .expect("a shape-consistent synthetic checkpoint must bind into a ViTEncoder");

        let n_mels = cfg.n_mels as usize;
        let n_frames = cfg.target_length as usize;
        let mut rng = Lcg(0x0BAD_C0DE_0000_0007);
        let mel = rng.vec(n_mels * n_frames);

        let (hidden, grid) = encoder
            .forward(&mel, n_mels, n_frames)
            .expect("the bound encoder must run");

        assert_eq!(grid.grid_h, cfg.patch_grid_freq as usize);
        assert_eq!(grid.grid_w, cfg.patch_grid_time as usize);
        assert_eq!(grid.n_patches, cfg.num_patches as usize);
        assert_eq!(
            hidden.len(),
            cfg.tokens_per_clip() * cfg.embed_dim as usize,
            "output must be [n_tokens, embed_dim] row-major"
        );
        assert!(
            hidden.iter().all(|v| v.is_finite()),
            "a finite input over finite weights must stay finite"
        );

        // Determinism: the encoder holds no interior mutability and the
        // forward draws no randomness, so two calls must agree exactly.
        let (again, _) = encoder
            .forward(&mel, n_mels, n_frames)
            .expect("second forward");
        assert_eq!(hidden, again, "the forward must be deterministic");
    }

    #[test]
    fn bind_vit_weights_names_a_wrong_shaped_tensor() {
        let cfg = tiny_config();
        let file = tiny_gguf(&cfg);
        let model = Eat::from_gguf(&file).unwrap();
        let attrs = cfg
            .to_vit_attrs(GeluKind::Erf, PosEmbedPolicy::RequireExact)
            .unwrap();

        // Point one slot at a tensor that exists but has the wrong shape.
        let mut names = tiny_names(cfg.depth as usize);
        names.blocks[0].fc1_weight = "blocks.0.attn.proj.weight".to_owned();

        let Err(err) = model.bind_vit_weights(&file, &names, &attrs) else {
            panic!("expected a loud ModelLoad on a wrong-shaped MLP weight");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("blocks.0.attn.proj.weight"),
                    "message must NAME the offending tensor, got `{m}`"
                );
                assert!(
                    m.contains("[8, 8]") && m.contains("[16, 8]"),
                    "message must report BOTH the actual and the expected dims, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the no-silent-reshape clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn bind_vit_weights_rejects_a_depth_mismatched_manifest() {
        let cfg = tiny_config();
        let file = tiny_gguf(&cfg);
        let model = Eat::from_gguf(&file).unwrap();
        let attrs = cfg
            .to_vit_attrs(GeluKind::Erf, PosEmbedPolicy::RequireExact)
            .unwrap();

        // One block short of the stamped depth.
        let names = tiny_names(1);
        let Err(err) = model.bind_vit_weights(&file, &names, &attrs) else {
            panic!("expected a loud ModelLoad when the manifest depth disagrees");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_DEPTH),
                    "message must name the stamped depth key, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the refusal clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6. Synthetic GGUF with the observer fixtures binds
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_synthetic_checkpoint_and_surfaces_license() {
        let file = eat_gguf(Some(LicenseClass::Permissive), 2);
        let m = Eat::from_gguf(&file).expect("a well-formed eat GGUF must bind");
        // Permissive is what the converter stamps for `mit`.
        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "the Permissive stamp must round-trip"
        );
        // 1 patch-embed tensor + 2 block tensors.
        assert_eq!(m.tensor_count(), 3);
        assert!(m.weights().has_patch_embed());
        assert_eq!(m.observed_block_count(), Some(2));
        // The stamped depth and the observed depth are independent
        // facts, and here they deliberately disagree: the fixture
        // carries 2 blocks while `eat-base` stamps 12. The binder
        // reports both rather than reconciling them.
        assert_eq!(m.config().depth, 12);
        // A loud slot lookup finds a real tensor and returns its dims.
        let dims = m
            .weights()
            .require_tensor("blocks.1.attn.qkv.weight")
            .expect("tensor present");
        assert_eq!(dims, &[12, 4]);
        m.weights()
            .require_tensor_dims("blocks.1.attn.qkv.weight", &[12, 4])
            .expect("dims match");
    }

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        let file = eat_gguf(None, 1);
        let m = Eat::from_gguf(&file).expect("license stamp is not a bind gate");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "a missing provenance stamp must fail closed to Unknown, never be assumed \
             Permissive"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Manifest observation — never a fabricated topology
    // -----------------------------------------------------------------------

    #[test]
    fn observed_structure_is_none_for_a_foreign_naming_scheme() {
        // A checkpoint flattened under a different prefix convention
        // (fairseq / data2vec2 lineage) is NOT invalid — the binder must
        // report "unknown", not a fabricated zero-layer topology.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        stamp_axes(&mut b, &EatConfig::eat_base_reference(), None);
        b.add_tensor(
            "modality_encoders.AUDIO.local_encoder.proj.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 4 * 4 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let m = Eat::from_gguf(&file).expect("a foreign naming scheme still binds");
        assert_eq!(
            m.observed_block_count(),
            None,
            "no `blocks.<i>.` tensors means UNKNOWN depth, never zero layers"
        );
        assert!(!m.weights().has_patch_embed());
        assert_eq!(m.tensor_count(), 1);
        assert_eq!(m.weights().count_with_prefix("modality_encoders."), 1);
        assert_eq!(
            m.weights().tensor_names(),
            vec!["modality_encoders.AUDIO.local_encoder.proj.weight"]
        );
    }

    // -----------------------------------------------------------------------
    // 8. Loud negative space — arch metadata absent
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor(
            "some.tensor",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 2 * 2 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Eat::from_gguf(&file) else {
            panic!("expected a loud ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native eat GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 9. Loud negative space — foreign arch names BOTH tags
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // A sibling SSL audio-encoder GGUF (`beats`) handed to the EAT
        // binder must fail loud rather than silently mis-binding.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "beats");
        b.add_string(chunks::KEY_MODEL_NAME, "beats-iter3-plus");
        b.add_tensor(
            "beats.probe",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 4 * 4 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Eat::from_gguf(&file) else {
            panic!("expected a loud ModelLoad on a foreign arch tag");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`beats`"),
                    "message must name the ACTUAL arch tag, got `{m}`"
                );
                assert!(
                    m.contains("`eat`"),
                    "message must name the EXPECTED arch tag, got `{m}`"
                );
                // The whole sibling neighbourhood must be enumerated so
                // the reader knows which loader they actually wanted.
                for sibling in ["dasheng", "atst", "m2d", "mert", "muq", "ast", "hubert"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling `{sibling}` disambiguation in error: {m}"
                    );
                }
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 10. Loud negative space — category stamped but wrong
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_category() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, "asr");
        b.add_tensor(
            "patch_embed.proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 2 * 2 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Eat::from_gguf(&file) else {
            panic!("expected a loud ModelLoad when the category stamp disagrees");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`asr`") && m.contains("`audio-embedding`"),
                    "message must name BOTH the actual and expected category, got `{m}`"
                );
                assert!(
                    m.contains(GGUF_KEY_MODEL_CATEGORY),
                    "message must name the offending key, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 11. Loud negative space — empty tensor manifest
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Eat::from_gguf(&file) else {
            panic!("expected a loud ModelLoad on a zero-tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model eat"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 12. Loud negative space — a missing tensor names the tensor
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = eat_gguf(Some(LicenseClass::Permissive), 2);
        let m = Eat::from_gguf(&file).unwrap();
        let Err(err) = m.weights().require_tensor("blocks.11.attn.qkv.weight") else {
            panic!("expected a loud ModelLoad when the requested tensor is absent");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("blocks.11.attn.qkv.weight"),
                    "message must NAME the missing tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("3 tensors present"),
                    "message should report how many tensors the artifact does carry, \
                     got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-zero-substitution clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn require_tensor_dims_names_expected_and_actual() {
        let file = eat_gguf(Some(LicenseClass::Permissive), 1);
        let m = Eat::from_gguf(&file).unwrap();
        let Err(err) = m
            .weights()
            .require_tensor_dims("blocks.0.attn.qkv.weight", &[768, 2304])
        else {
            panic!("expected a loud ModelLoad on a dims mismatch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("blocks.0.attn.qkv.weight"),
                    "message must name the tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("[12, 4]"),
                    "message must report the ACTUAL dims, got `{msg}`"
                );
                assert!(
                    msg.contains("[768, 2304]"),
                    "message must report the EXPECTED dims, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-silent-reshape clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 13. Loud-partial — encode names ONLY the blockers that are real today
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partials_naming_only_live_blockers() {
        let file = eat_gguf(Some(LicenseClass::Permissive), 2);
        let m = Eat::from_gguf(&file).unwrap();
        // 1 s of legitimately-shaped mono PCM so the loud-partial gate
        // fires, not some pre-encode length validation.
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial — it cannot emit real hidden states yet");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("eat encode"), "surface must be named: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // --- The blockers that were resolved under this landing must
                // --- NOT be claimed any more. A stale claim in an error
                // --- message actively misleads the next reader, which is
                // --- precisely what this rewrite removes.
                assert!(
                    !msg.contains("vokra.frontend.*"),
                    "the frontend spec is no longer the blocker — the Kaldi-fbank argument \
                     set IS stamped under `vokra.eat.fbank_*`. Stale claim in: {msg}"
                );
                assert!(
                    !msg.contains("patchifier"),
                    "`vokra_ops::vit::vit_patch_embed` exists now; the missing-patchifier \
                     claim is stale. Got: {msg}"
                );
                assert!(
                    !msg.contains("no plain ViT encoder"),
                    "`vokra_ops::vit::ViTEncoder` exists now; the missing-encoder claim is \
                     stale. Got: {msg}"
                );
                assert!(
                    !msg.contains("not known to the runtime"),
                    "the topology axes ARE known now — the converter stamps 38 \
                     `vokra.eat.*` keys and this binder reads them. Got: {msg}"
                );

                // --- The blockers that ARE real today must be named.
                assert!(
                    msg.contains("NO VERIFIED TENSOR-NAME MANIFEST"),
                    "message must name the manifest blocker, got `{msg}`"
                );
                assert!(
                    msg.contains("EatVitTensorNames"),
                    "message must point at the type that resolves it, got `{msg}`"
                );
                assert!(
                    msg.contains("modality_encoders."),
                    "message must name the competing fairseq naming convention, got `{msg}`"
                );
                assert!(
                    msg.contains("KaldiFbankWindow") && msg.contains("Povey"),
                    "message must name the window mismatch and the op that hard-codes it, \
                     got `{msg}`"
                );
                assert!(
                    msg.contains(GGUF_KEY_FBANK_WINDOW_TYPE) && msg.contains("hanning"),
                    "message must quote the stamped window the op cannot honour, got \
                     `{msg}`"
                );
                assert!(
                    msg.contains(GGUF_KEY_LAYER_NORM_FIRST) && msg.contains("pre-norm"),
                    "message must name the unreconciled norm-order flag, got `{msg}`"
                );
                assert!(
                    msg.contains("AltBlock"),
                    "message must name the upstream file that settles norm order, got \
                     `{msg}`"
                );

                // --- The mapped axes, so a reader can see what IS known.
                assert!(
                    msg.contains("embed_dim=768") && msg.contains("depth=12"),
                    "message must report the axes it successfully mapped, got `{msg}`"
                );

                // --- Observed manifest facts.
                assert!(
                    msg.contains("tensor_count=3"),
                    "message must report the observed tensor count, got `{msg}`"
                );
                assert!(
                    msg.contains("observed_block_count=2"),
                    "message must report the observed block count, got `{msg}`"
                );
                assert!(
                    msg.contains("has_patch_embed=true"),
                    "message must report patch-embed presence, got `{msg}`"
                );

                // --- Primary sources + the FR-EX-08 rationale.
                assert!(
                    msg.contains(UPSTREAM_URL),
                    "message must cite the upstream repo, got `{msg}`"
                );
                assert!(
                    msg.contains(PRIMARY_SOURCE_PAPER),
                    "message must cite the paper anchor, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-fabrication clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 14. Loud-partial — embed_utterance adds the read-out blocker
    // -----------------------------------------------------------------------

    #[test]
    fn embed_utterance_loud_partials_naming_the_readout_blocker() {
        let file = eat_gguf(Some(LicenseClass::Permissive), 2);
        let m = Eat::from_gguf(&file).unwrap();
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.embed_utterance(&pcm) else {
            panic!("embed_utterance must loud-partial — no embedding can be fabricated");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("eat embed_utterance"),
                    "surface must be named: {msg}"
                );
                // The utterance-specific fourth blocker, narrowed to what
                // is actually still unknown.
                assert!(
                    msg.contains("READ-OUT convention"),
                    "message must name the deferred read-out convention, got `{msg}`"
                );
                assert!(
                    msg.contains("ViTPooling"),
                    "message must name the primitive that CAN express it now, got `{msg}`"
                );
                assert!(
                    msg.contains("CLS token sits"),
                    "message must name the un-stamped CLS index, got `{msg}`"
                );
                // The embedding width is no longer unknown — it is stamped.
                assert!(
                    msg.contains("embedding width IS stamped"),
                    "message must record that the width is now known, got `{msg}`"
                );
                // It still carries the three shared blockers.
                assert!(
                    msg.contains("NO VERIFIED TENSOR-NAME MANIFEST")
                        && msg.contains("Povey")
                        && msg.contains("NORM ORDER UNRECONCILED"),
                    "message must still name the shared blockers, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-fabrication clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
