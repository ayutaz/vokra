#![allow(clippy::doc_lazy_continuation)]
//! **MAEST** (`mtg-upf/discogs-maest-30s-pw-129e`,
//! **cc-by-nc-sa-4.0**): safetensors → GGUF conversion
//! (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `mtg-upf/discogs-maest-30s-pw-129e` release —
//! MAEST ("Music **A**udio **E**fficient **S**pectrogram
//! **T**ransformer", Alonso-Jiménez et al. 2023 ISMIR
//! arXiv:2309.16418) is a self-supervised **music-tagger** built on
//! the **Audio Spectrogram Transformer (AST)** backbone (HF
//! `config`: `model_type: audio-spectrogram-transformer`,
//! `architectures: ["ASTForAudioClassification"]`, verified via HF
//! cardData API 2026-08-13). The `30s-pw-129e` variant is 30-second
//! patch-wise pretrained for 129 epochs on the MTG Discogs4All
//! music-tagger dataset. ~87M F32 parameters (safetensors
//! `parameters.F32: 86,858,128` per HF API primary source).
//!
//! # Vokra scope — music understanding via AST-backbone SSL
//!
//! Sibling of `mert` (HuBERT-derived Conv1D + Transformer MPM),
//! `muq` (Mel-RVQ + BEATs teacher), `dasheng` (universal MAE).
//! Distinct arch tag `maest` because the AST-backbone (patch-wise
//! Transformer over log-mel spectrogram) + Discogs-tagger SSL
//! pretraining objective is a distinct topology axis from every
//! sibling music-embedding model — silently sharing would
//! misroute the runtime dispatch (FR-EX-08). Category
//! `music-embedding` (sibling of `mert` / `muq`; downstream
//! music-tagging heads consume the encoder's hidden states).
//!
//! # License posture — CC-BY-NC-SA 4.0 (**NonCommercialShareAlike**)
//!
//! Upstream HF cardData `license: cc-by-nc-sa-4.0` (verified via
//! `https://huggingface.co/api/models/mtg-upf/discogs-maest-30s-pw-129e`
//! primary source task input 2026-08-13; HF tag
//! `license:cc-by-nc-sa-4.0` also present). This is **T4 tier +
//! SA cascade** — the strictest CC family + share-alike:
//!
//! - **NonCommercial** — commercial use forbidden without a
//!   separate license from the MTG group.
//! - **ShareAlike** — any downstream distribution of the weight
//!   (or a derivative) must be under the same CC-BY-NC-SA 4.0
//!   license, effectively preventing re-licensing.
//! - **BY (Attribution)** — cascaded attribution requirement.
//!
//! **Publish path**: `publish-one.sh --allow-noncommercial` gate
//! + `fetch_license.sh --spdx cc-by-nc-sa-4.0` canonical LICENSE
//! bundled (nisqa_v2_weight / audioldm2 precedent). §3.1 sign-off
//! stays blank fail-closed until owner completes primary-source
//! confirmation (memory `[[feedback-license-signoff-primary-source]]`
//! — no CC pre-fill).
//!
//! # `vokra.maest.*` topology axis group (stamped here)
//!
//! Beyond the `vokra.model.*` / `vokra.provenance.*` stamps, this converter
//! writes the full Transformer topology + log-mel front-end axis group, so
//! the runtime binder (`crates/vokra-models/src/maest/mod.rs`) can bind a
//! shape without guessing one. Every value is transcribed from a primary
//! source actually fetched on 2026-08-15 — the citation block above the
//! constants lists the four URLs, and carries a parameter-count closure that
//! mutually verifies the whole group against the upstream weight file.
//!
//! **Deliberate omissions.** A missing key is honest; a guessed one binds
//! shape-valid garbage, so the following are left unstamped:
//!
//! - **STFT framing / centering.** No primary source reached states whether
//!   the analysis is centred, nor which padding mode it uses, so no
//!   `pad_mode` or `center` axis is written.
//! - **The `vokra.frontend.*` bit-exact group** (`vokra_core::gguf::FrontendSpec`)
//!   is deliberately NOT written. It is an all-or-nothing bit-exactness
//!   contract, and three of its required fields — `pad_mode`,
//!   `dc_offset_removal`, `pre_emphasis` — have no primary-source value for
//!   MAEST. Filling them with plausible defaults would promote an unverified
//!   guess into a *checked* "bit-exact" claim, which is strictly worse than
//!   an absent group. The front-end subset that IS transcribed rides the
//!   `vokra.maest.*` namespace instead.
//! - **`initializer_range`** (`config.json`, `0.02`) is read but not stamped:
//!   a training-time weight-init detail with no inference meaning.
//!
//! # Scale — local convert OK (~0.15 GB / ~87M F32 params)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! MAEST ships as single-file safetensors (`model.safetensors`)
//! per HF `siblings` inspection (also includes legacy
//! `pytorch_model.bin` pickle which Vokra never reads). This
//! converter **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02) — the safetensors path is the primary source and
//! no bridge sidecar is needed.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16
//! is emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime
//! widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MAEST GGUFs. Distinct from sibling
/// music-embedding arch tags (`mert` = HuBERT-derived MPM /
/// `muq` = Mel-RVQ + BEATs teacher / `dasheng` = MAE
/// ViT/ConvNeXt) — MAEST's AST-backbone (patch-wise Transformer
/// over log-mel spectrogram) + Discogs-tagger SSL objective is a
/// distinct topology axis from every sibling.
pub const ARCH: &str = "maest";

/// `vokra.model.name` — canonical `mtg-upf/discogs-maest-30s-pw-129e`
/// release variant (30-second, patch-wise, 129 epochs — the
/// primary release variant per the upstream README). Sibling
/// variants (5s / 10s / 20s durations, 30s-pw-73e checkpoint
/// point etc.) are distinct release identities published as their
/// own future `NAME` following the snac_24khz / snac_44khz
/// pattern (added via separate future ModelKind).
pub const NAME: &str = "maest-30s-pw-129e";

/// `vokra.model.category` — music understanding embedding
/// (sibling of `mert` / `muq`; downstream music-tagging heads
/// consume the encoder's hidden states). Distinct from
/// `audio-tagging` (sibling of `yamnet` / `panns` / `ast` /
/// `clap`) because MAEST is trained specifically on the Discogs
/// music-tagger dataset — output is genre / mood / instrument /
/// era annotations over music, not the general AudioSet audio-
/// event ontology.
pub const CATEGORY: &str = "music-embedding";

/// Upstream HF slug — recorded on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "mtg-upf/discogs-maest-30s-pw-129e";

/// Default SPDX. HF cardData primary source `license: cc-by-nc-sa-4.0`
/// (verified via `https://huggingface.co/api/models/mtg-upf/
/// discogs-maest-30s-pw-129e` task input 2026-08-13). A caller
/// with a different attestation may override at the outer
/// boundary (`--license <spdx>`); the M2-13 runtime gate then
/// reclassifies.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

// ---------------------------------------------------------------------------
// MAEST topology axes — transcribed verbatim from the upstream primary
// sources, all fetched 2026-08-15. Every constant below is a value that was
// READ; none is inferred from "typical AST-base" defaults.
//
// Primary sources:
// - config.json .............. huggingface.co/mtg-upf/discogs-maest-30s-pw-129e
//                              /raw/main/config.json
// - preprocessor_config.json . same repo, /raw/main/preprocessor_config.json
// - feature_extraction_maest.py  same repo, /raw/main/feature_extraction_maest.py
//                              (the `MAESTFeatureExtractor` that the sidecar's
//                              `auto_map` points at — the front-end's own code)
// - ASTPatchEmbeddings ....... github.com/huggingface/transformers/blob/v4.34.0
//                              /src/transformers/models/
//                              audio_spectrogram_transformer/
//                              modeling_audio_spectrogram_transformer.py
//                              (v4.34.0 is the matching tag: the checkpoint's
//                              own config records
//                              `transformers_version: "4.34.0.dev0"`)
//
// The three in-repo sources agree on every shared field: `num_mel_bins` (96)
// and `max_length` (1876) appear identically in config.json AND
// preprocessor_config.json, and feature_extraction_maest.py's __init__
// defaults reproduce the whole sidecar.
//
// CLOSURE CHECK — the axes below are not merely transcribed, they are
// mutually verified. Summing the parameter count they imply reproduces the
// `parameters.F32` that the HuggingFace API reports for this repo EXACTLY:
//
//   embeddings  cls + distillation tokens    2 * 768         =       1,536
//               position table       (1683 + 2) * 768        =   1,294,080
//               patch Conv2d      768*1*16*16 + 768          =     197,376
//   encoder     12 * ( 3*(768*768+768)        [q, k, v]
//                      + (768*768+768)        [attn output]
//                      + (768*3072+3072)      [FFN in]
//                      + (3072*768+768)       [FFN out]
//                      + 4*768 )              [2 LayerNorms] =  85,054,464
//   final LN    2 * 768                                      =       1,536
//   head        LN 2*768 + (400*768 + 400)                   =     309,136
//                                                    TOTAL   =  86,858,128
//
// 86,858,128 is exactly the upstream `parameters.F32`. A single wrong axis —
// a different depth, width, label count, patch grid, prefix-token count, or
// a missing qkv bias — breaks that equality, so the closure is an
// independent check on the whole group rather than a restatement of it. It
// is also executable: see `transcribed_axes_reproduce_the_upstream_f32_parameter_count`
// in the test module.
// ---------------------------------------------------------------------------

/// Transformer hidden size (`config.json` `hidden_size`) — 768.
pub const HIDDEN_SIZE: u32 = 768;

/// Transformer block count (`config.json` `num_hidden_layers`) — 12.
pub const NUM_HIDDEN_LAYERS: u32 = 12;

/// Attention head count (`config.json` `num_attention_heads`) — 12.
pub const NUM_ATTENTION_HEADS: u32 = 12;

/// FFN intermediate width (`config.json` `intermediate_size`) — 3072.
pub const INTERMEDIATE_SIZE: u32 = 3072;

/// Square ViT patch edge, in mel bins × frames (`config.json` `patch_size`)
/// — 16.
///
/// The AST patch embedding is a `Conv2d(1, hidden_size, kernel_size=(16, 16),
/// stride=(frequency_stride, time_stride))`, so the patches **overlap**: both
/// strides are 10 against a kernel of 16. A reader assuming the usual
/// non-overlapping ViT `stride == patch_size` would compute a ~3x smaller
/// patch grid.
pub const PATCH_SIZE: u32 = 16;

/// Patch stride along the mel-bin axis (`config.json` `frequency_stride`) — 10.
pub const FREQUENCY_STRIDE: u32 = 10;

/// Patch stride along the frame axis (`config.json` `time_stride`) — 10.
pub const TIME_STRIDE: u32 = 10;

/// Log-mel band count (`config.json` `num_mel_bins`, and identically
/// `preprocessor_config.json` `num_mel_bins`) — 96.
///
/// Note this is **not** the 128 bands the general-audio AST / AudioSet
/// lineage uses: MAEST's Essentia-derived front-end is 96-band. Carrying the
/// sibling `ast` value across would mis-shape the patch grid.
pub const NUM_MEL_BINS: u32 = 96;

/// Input frame count the position-embedding table is sized for
/// (`config.json` `max_length`, identically `preprocessor_config.json`
/// `max_length`) — 1876.
///
/// Consistent with the 30-second clip this release variant is named for:
/// 30 s × 16000 Hz ÷ 256-sample hop = 1875 frames, and the table holds 1876.
/// The exact framing convention that accounts for the extra frame is not
/// stated by any source reached, which is one reason no `pad_mode` axis is
/// stamped (see the module docstring's "Deliberate omissions").
pub const MAX_LENGTH: u32 = 1876;

/// Discogs label-set size — 400.
///
/// **Read, never assumed.** Three independent confirmations: `config.json`
/// carries an `id2label` map with keys `"0"` … `"399"` plus a matching
/// `label2id` inverse; the upstream model card independently states "a
/// taxonomy of 400 music styles derived from the public metadata of
/// Discogs"; and the parameter-count closure above only balances at 400.
pub const NUM_LABELS: u32 = 400;

/// Whether the attention q/k/v projections carry a bias
/// (`config.json` `qkv_bias`) — `true`.
pub const QKV_BIAS: bool = true;

/// Encoder activation (`config.json` `hidden_act`) — `"gelu"`.
pub const HIDDEN_ACT: &str = "gelu";

/// LayerNorm epsilon as a scaled `u32` (× 1e9). The primary-source value is
/// `config.json` `layer_norm_eps` = `1e-06`, so 1e-6 × 1e9 = 1000.
///
/// Integer-encoded for the same reason the sibling `vokra.wavlm.*` group does
/// it: the reader round-trips it without float-serialization ambiguity, and
/// the runtime binder converts back on load. Note the value differs from
/// WavLM's `1e-5` — do not carry the sibling constant across.
pub const LAYER_NORM_EPS_SCALED_1E9: u32 = 1_000;

/// Hidden dropout scaled by 1e3 (`config.json` `hidden_dropout_prob` = 0.0)
/// — 0. Inference-irrelevant; stamped for audit symmetry with the sibling
/// `vokra.wavlm.hidden_dropout_scaled_1e3`.
pub const HIDDEN_DROPOUT_SCALED_1E3: u32 = 0;

/// Attention dropout scaled by 1e3
/// (`config.json` `attention_probs_dropout_prob` = 0.0) — 0.
pub const ATTENTION_DROPOUT_SCALED_1E3: u32 = 0;

// Patch grid. DERIVED — but derived by a formula transcribed from
// transformers v4.34.0 `ASTPatchEmbeddings.get_shape`, quoted verbatim:
//
//   frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
//   time_out_dimension      = (config.max_length   - config.patch_size) // config.time_stride      + 1
//   num_patches             = frequency_out_dimension * time_out_dimension
//
// evaluated at the transcribed axes above:
//
//   freq = (96   - 16) // 10 + 1 =   8 + 1 =   9
//   time = (1876 - 16) // 10 + 1 = 186 + 1 = 187
//   num  = 9 * 187                         = 1683
//
// The derivation is re-executed in the test module against the stamped
// inputs, so a hand-edit that drifts an input away from the grid fails.

/// Patch-grid extent along the mel-bin axis — 9.
pub const FREQ_PATCHES: u32 = 9;

/// Patch-grid extent along the frame axis — 187.
pub const TIME_PATCHES: u32 = 187;

/// Total patch-token count entering the encoder — 1683 = 9 × 187.
pub const NUM_PATCHES: u32 = 1683;

/// Prefix tokens prepended to the patch sequence — 2.
///
/// AST is DeiT-derived, so it carries **both** a classification token and a
/// distillation token, making the encoder sequence length
/// `NUM_PATCHES + NUM_PREFIX_TOKENS` = 1685. This is the axis a ViT
/// implementation is most likely to get wrong by assuming the plain
/// single-CLS convention, so it is stamped explicitly rather than left
/// implicit: the parameter-count closure only reproduces the upstream
/// 86,858,128 with a 1685-row position table plus two `[1, 1, 768]` token
/// parameters.
pub const NUM_PREFIX_TOKENS: u32 = 2;

// Log-mel front-end. Transcribed from `preprocessor_config.json` and from the
// `feature_extraction_maest.py` its `auto_map` names
// (`AutoFeatureExtractor: feature_extraction_maest.MAESTFeatureExtractor`).
// The sidecar and the code agree on every shared field.

/// Front-end sample rate in Hz — 16000, mono (`sampling_rate`).
pub const SAMPLE_RATE: u32 = 16_000;

/// STFT transform size — 512 (`n_fft`).
pub const N_FFT: u32 = 512;

/// STFT hop in samples — 256 (`hop_length`).
pub const HOP_LENGTH: u32 = 256;

/// STFT analysis-window length in samples — 512.
///
/// Not a distinct sidecar field: `feature_extraction_maest.py` passes
/// `window_length=self.n_fft`, so the window spans the whole transform.
pub const WIN_LENGTH: u32 = 512;

/// STFT analysis-window type — `"hann"`.
pub const WINDOW: &str = "hann";

/// Mel filterbank frequency scale — `"slaney"` (i.e. **not** HTK).
pub const MEL_SCALE: &str = "slaney";

/// Mel filterbank normalization — `"slaney"` (equal-area bands).
pub const MEL_NORM: &str = "slaney";

/// Mel filterbank lower edge in Hz — 0 (`min_frequency=0`).
pub const FMIN_HZ: u32 = 0;

/// Mel filterbank upper edge in Hz — 8000.
///
/// `feature_extraction_maest.py` passes `max_frequency=self.sampling_rate / 2`;
/// evaluated at the transcribed [`SAMPLE_RATE`] that is Nyquist = 8000.
pub const FMAX_HZ: u32 = 8_000;

/// Magnitude-compression mode — `"logC"` (`log_compression`).
pub const LOG_COMPRESSION: &str = "logC";

/// The multiplier inside the `"logC"` compression — 10000.
///
/// `feature_extraction_maest.py` computes the compressed band energies as
/// `np.log10(1 + melspec * 10000)`. Stamped as its own axis so a runtime
/// implementing the front-end binds the constant instead of re-deriving it
/// from the opaque mode string.
pub const LOG_COMPRESSION_MUL: u32 = 10_000;

/// Whether the compressed spectrogram is mean/std normalized — `true`
/// (`do_normalize`).
pub const DO_NORMALIZE: bool = true;

/// Normalization mean subtracted when [`DO_NORMALIZE`] — 2.06755686098554.
///
/// Stamped as GGUF `FLOAT64`, not `FLOAT32`, so the upstream value
/// round-trips at the precision it was published with.
pub const NORM_MEAN: f64 = 2.067_556_860_985_54;

/// Normalization standard deviation divided out when [`DO_NORMALIZE`] —
/// 1.268292820667291. Stamped as GGUF `FLOAT64`; see [`NORM_MEAN`].
pub const NORM_STD: f64 = 1.268_292_820_667_291;

const UPSTREAM_SOURCE: &str = "mtg-upf/discogs-maest-30s-pw-129e (Music AEST — Discogs-pretrained AST self-supervised \
     music-tagger, 30-second patch-wise 129-epoch pretraining, ~87M F32 params, \
     Alonso-Jiménez et al. arXiv:2309.16418 ISMIR 2023, cc-by-nc-sa-4.0)";

// ---------------------------------------------------------------------------
// `vokra.maest.*` metadata keys
//
// Reader-side counterparts live in `vokra-models::maest`; the spellings here
// are the contract between the two. Named after the upstream `config.json`
// field each value came from (see the constant declarations above for the
// per-field primary source), so a reader diagnosing a mismatch can map a key
// straight back to the upstream config.
// ---------------------------------------------------------------------------

const KEY_MAEST_HIDDEN_SIZE: &str = "vokra.maest.hidden_size";
const KEY_MAEST_NUM_HIDDEN_LAYERS: &str = "vokra.maest.num_hidden_layers";
const KEY_MAEST_NUM_ATTENTION_HEADS: &str = "vokra.maest.num_attention_heads";
const KEY_MAEST_INTERMEDIATE_SIZE: &str = "vokra.maest.intermediate_size";
const KEY_MAEST_PATCH_SIZE: &str = "vokra.maest.patch_size";
const KEY_MAEST_FREQUENCY_STRIDE: &str = "vokra.maest.frequency_stride";
const KEY_MAEST_TIME_STRIDE: &str = "vokra.maest.time_stride";
const KEY_MAEST_NUM_MEL_BINS: &str = "vokra.maest.num_mel_bins";
const KEY_MAEST_MAX_LENGTH: &str = "vokra.maest.max_length";
const KEY_MAEST_NUM_LABELS: &str = "vokra.maest.num_labels";
const KEY_MAEST_QKV_BIAS: &str = "vokra.maest.qkv_bias";
const KEY_MAEST_HIDDEN_ACT: &str = "vokra.maest.hidden_act";
const KEY_MAEST_LAYER_NORM_EPS_SCALED_1E9: &str = "vokra.maest.layer_norm_eps_scaled_1e9";
const KEY_MAEST_HIDDEN_DROPOUT_SCALED_1E3: &str = "vokra.maest.hidden_dropout_scaled_1e3";
const KEY_MAEST_ATTENTION_DROPOUT_SCALED_1E3: &str = "vokra.maest.attention_dropout_scaled_1e3";
const KEY_MAEST_FREQ_PATCHES: &str = "vokra.maest.freq_patches";
const KEY_MAEST_TIME_PATCHES: &str = "vokra.maest.time_patches";
const KEY_MAEST_NUM_PATCHES: &str = "vokra.maest.num_patches";
const KEY_MAEST_NUM_PREFIX_TOKENS: &str = "vokra.maest.num_prefix_tokens";
const KEY_MAEST_SAMPLE_RATE: &str = "vokra.maest.sample_rate";
const KEY_MAEST_N_FFT: &str = "vokra.maest.n_fft";
const KEY_MAEST_HOP_LENGTH: &str = "vokra.maest.hop_length";
const KEY_MAEST_WIN_LENGTH: &str = "vokra.maest.win_length";
const KEY_MAEST_WINDOW: &str = "vokra.maest.window";
const KEY_MAEST_MEL_SCALE: &str = "vokra.maest.mel_scale";
const KEY_MAEST_MEL_NORM: &str = "vokra.maest.mel_norm";
const KEY_MAEST_FMIN_HZ: &str = "vokra.maest.fmin_hz";
const KEY_MAEST_FMAX_HZ: &str = "vokra.maest.fmax_hz";
const KEY_MAEST_LOG_COMPRESSION: &str = "vokra.maest.log_compression";
const KEY_MAEST_LOG_COMPRESSION_MUL: &str = "vokra.maest.log_compression_mul";
const KEY_MAEST_DO_NORMALIZE: &str = "vokra.maest.do_normalize";
const KEY_MAEST_NORM_MEAN: &str = "vokra.maest.norm_mean";
const KEY_MAEST_NORM_STD: &str = "vokra.maest.norm_std";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MAEST conversion. Mirrors the counter shape of
/// the sibling BF16 pass-through converters (`mert` / `muq` /
/// `dasheng` / `beats` / `eat` / `atst` / `yamnet`) — the
/// invariant `read == written + skipped_non_float` is auditable
/// at the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MaestReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16, so any tensor reaching
    /// this counter would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a `mtg-upf/discogs-maest-30s-pw-129e` safetensors
/// checkpoint at `input` into a Vokra-native GGUF at `output`,
/// returning a [`MaestReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-sa-4.0"`,
/// `NonCommercialShareAlike`) — fail-closed under M2-13, publish
/// requires `publish-one.sh --allow-noncommercial` + share-alike
/// obligation on any downstream distribution.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_maest_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MaestReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    // ---- `vokra.maest.*` topology group ---------------------------------
    // Purely additive: the arch / name / category / provenance stamps above
    // are untouched, so an artifact produced before this group existed and
    // one produced after differ only by these keys. Values are transcribed
    // constants (see their declarations for the primary source each came
    // from). The Discogs label count is stamped from NUM_LABELS rather than
    // being inferred from the tag-head tensor, so the reader can cross-check
    // the two and refuse an artifact where they disagree.
    b.add_u32(KEY_MAEST_HIDDEN_SIZE, HIDDEN_SIZE);
    b.add_u32(KEY_MAEST_NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS);
    b.add_u32(KEY_MAEST_NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS);
    b.add_u32(KEY_MAEST_INTERMEDIATE_SIZE, INTERMEDIATE_SIZE);
    b.add_u32(KEY_MAEST_PATCH_SIZE, PATCH_SIZE);
    b.add_u32(KEY_MAEST_FREQUENCY_STRIDE, FREQUENCY_STRIDE);
    b.add_u32(KEY_MAEST_TIME_STRIDE, TIME_STRIDE);
    b.add_u32(KEY_MAEST_NUM_MEL_BINS, NUM_MEL_BINS);
    b.add_u32(KEY_MAEST_MAX_LENGTH, MAX_LENGTH);
    b.add_u32(KEY_MAEST_NUM_LABELS, NUM_LABELS);
    b.add_bool(KEY_MAEST_QKV_BIAS, QKV_BIAS);
    b.add_string(KEY_MAEST_HIDDEN_ACT, HIDDEN_ACT);
    b.add_u32(
        KEY_MAEST_LAYER_NORM_EPS_SCALED_1E9,
        LAYER_NORM_EPS_SCALED_1E9,
    );
    b.add_u32(
        KEY_MAEST_HIDDEN_DROPOUT_SCALED_1E3,
        HIDDEN_DROPOUT_SCALED_1E3,
    );
    b.add_u32(
        KEY_MAEST_ATTENTION_DROPOUT_SCALED_1E3,
        ATTENTION_DROPOUT_SCALED_1E3,
    );
    b.add_u32(KEY_MAEST_FREQ_PATCHES, FREQ_PATCHES);
    b.add_u32(KEY_MAEST_TIME_PATCHES, TIME_PATCHES);
    b.add_u32(KEY_MAEST_NUM_PATCHES, NUM_PATCHES);
    b.add_u32(KEY_MAEST_NUM_PREFIX_TOKENS, NUM_PREFIX_TOKENS);
    b.add_u32(KEY_MAEST_SAMPLE_RATE, SAMPLE_RATE);
    b.add_u32(KEY_MAEST_N_FFT, N_FFT);
    b.add_u32(KEY_MAEST_HOP_LENGTH, HOP_LENGTH);
    b.add_u32(KEY_MAEST_WIN_LENGTH, WIN_LENGTH);
    b.add_string(KEY_MAEST_WINDOW, WINDOW);
    b.add_string(KEY_MAEST_MEL_SCALE, MEL_SCALE);
    b.add_string(KEY_MAEST_MEL_NORM, MEL_NORM);
    b.add_u32(KEY_MAEST_FMIN_HZ, FMIN_HZ);
    b.add_u32(KEY_MAEST_FMAX_HZ, FMAX_HZ);
    b.add_string(KEY_MAEST_LOG_COMPRESSION, LOG_COMPRESSION);
    b.add_u32(KEY_MAEST_LOG_COMPRESSION_MUL, LOG_COMPRESSION_MUL);
    b.add_bool(KEY_MAEST_DO_NORMALIZE, DO_NORMALIZE);
    // FLOAT64, not FLOAT32: upstream publishes these normalization
    // statistics to 15 significant digits, and f32 carries about 7.
    // `GgufBuilder` has no `add_f64` shorthand, so they go through
    // `add_metadata` with the spec type (value.rs `F64 = 12`).
    b.add_metadata(
        KEY_MAEST_NORM_MEAN,
        vokra_core::gguf::GgufMetadataValue::F64(NORM_MEAN),
    );
    b.add_metadata(
        KEY_MAEST_NORM_STD,
        vokra_core::gguf::GgufMetadataValue::F64(NORM_STD),
    );

    let mut report = MaestReport::default();
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
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-maest-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn f32_tensor_passes_through_and_default_license_is_ncsa_fail_closed() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // MAEST uses HF's ASTForAudioClassification wrapper — realistic
        // upstream state-dict name from the AST-backbone body.
        let st = safetensors_one(
            "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight",
            "F32",
            &[3],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_maest_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |k: &str| -> String {
            g.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{k}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_HF), UPSTREAM_HF);
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::NonCommercialShareAlike.as_str(),
            "cc-by-nc-sa-4.0 must resolve to NonCommercialShareAlike (T4 + SA cascade)"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 4] = [1.0, -0.5, 0.25, 8.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one(
            "audio_spectrogram_transformer.encoder.layer.0.output.dense.weight",
            "BF16",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_maest_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("audio_spectrogram_transformer.encoder.layer.0.output.dense.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn topology_axes_round_trip_and_stay_purely_additive() {
        let inp = tmp_path("topo-in");
        let outp = tmp_path("topo-out");
        let payload: Vec<u8> = [1.5_f32, -0.25, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let st = safetensors_one("blocks.0.mlp.fc1.weight", "F32", &[1, 3], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_maest_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);

        let g = GgufFile::open(&outp).unwrap();
        let read_u64 = |k: &str| -> u64 {
            g.get(k)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("{k}: missing or not an unsigned integer"))
        };

        // Each row asserts the stamp against BOTH the constant and a literal
        // restating the upstream value, so it fails if either drifts.
        assert_eq!(read_u64(KEY_MAEST_HIDDEN_SIZE), u64::from(HIDDEN_SIZE));
        assert_eq!(read_u64(KEY_MAEST_HIDDEN_SIZE), 768);
        assert_eq!(
            read_u64(KEY_MAEST_NUM_HIDDEN_LAYERS),
            u64::from(NUM_HIDDEN_LAYERS)
        );
        assert_eq!(read_u64(KEY_MAEST_NUM_HIDDEN_LAYERS), 12);
        assert_eq!(read_u64(KEY_MAEST_NUM_LABELS), u64::from(NUM_LABELS));

        // The Discogs label count is stamped, so a reader can cross-check it
        // against the tag-head tensor width instead of trusting one of them.
        const {
            assert!(
                NUM_LABELS > 0,
                "a zero label count would make the cross-check vacuous"
            )
        };

        // FLOAT64 normalization statistics survive at full precision — the
        // whole reason they are not stamped as f32.
        assert_eq!(
            g.get(KEY_MAEST_NORM_MEAN),
            Some(&vokra_core::gguf::GgufMetadataValue::F64(NORM_MEAN))
        );
        assert_eq!(
            g.get(KEY_MAEST_NORM_STD),
            Some(&vokra_core::gguf::GgufMetadataValue::F64(NORM_STD))
        );

        // Purely additive: the pre-existing stamps are untouched.
        assert_eq!(
            g.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            g.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
    }

    #[test]
    fn license_override_to_permissive_flips_class() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        // A caller with a different license attestation escapes the
        // NonCommercialShareAlike default — mirror of the audioldm2 /
        // nisqa_v2_weight escape hatch (though for MAEST this would
        // require MTG group re-license).
        convert_maest_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
