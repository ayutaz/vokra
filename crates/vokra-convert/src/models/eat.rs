#![allow(clippy::doc_lazy_continuation)]
//! **EAT** (`cwx-worst-one/EAT`, **mit**): safetensors → GGUF
//! conversion (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `cwx-worst-one/EAT` release — EAT
//! ("Effective Audio Transformer") is a self-supervised audio
//! encoder that combines an **utterance-level Transformer** with
//! **inverse block masking** and a self-distillation objective
//! (Chen et al. 2024, arXiv:2401.03497). Trained on AudioSet-2M
//! with MAE-style masked reconstruction, positioned as an
//! efficient alternative to BEATs / AST for downstream audio
//! tagging and general audio-embedding tasks. ~86M parameters
//! base variant (~350 MB PyTorch checkpoint).
//!
//! # Vokra scope — SSL audio encoder (2026-07-30 scope expansion)
//!
//! Sibling of `beats` (iterative-tokenizer SSL), `dasheng`
//! (universal MAE), `atst` (teacher-student patchout), `m2d`
//! (masked-modeling-duo). Distinct arch tag `eat` because the
//! utterance-level Transformer + inverse-block-masking topology
//! is a distinct axis from every sibling SSL encoder — silently
//! sharing would misroute the runtime dispatch and try to bind
//! e.g. a MAE decoder over an utterance-level checkpoint
//! (FR-EX-08). Category `audio-embedding`.
//!
//! # License posture — mit (**Permissive**)
//!
//! Upstream `github.com/cwx-worst-one/EAT` LICENSE reports
//! `spdx_id: MIT` via GitHub API `/repos/cwx-worst-one/EAT/license`
//! (task input 2026-08-13). No HuggingFace mirror exists as of
//! 2026-08-13 (search of `EAT` audio-tagged models returned no
//! matches beyond unrelated finetunes). §3.1 sign-off stays
//! blank fail-closed until owner completes primary-source
//! confirmation (memory `[[feedback-license-signoff-primary-source]]`
//! — no CC pre-fill).
//!
//! # Scale — local convert OK (~0.35 GB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! EAT ships as PyTorch `.pt` pickle from the upstream GitHub
//! release; this converter **never** touches ONNX or pickle
//! (FR-LD-05 / NFR-DS-02). Callers pre-flatten via a future
//! `tools/parity/eat_prepare_checkpoint.py` uv-managed Python
//! 3.12 sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) mirroring the DAC / Kokoro /
//! UTMOSv2 bridge pattern.
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

/// `vokra.model.arch` for EAT GGUFs. Distinct from sibling SSL
/// audio-encoder arch tags (`beats` / `dasheng` / `atst` / `m2d` /
/// `mert` / `muq`) — EAT's utterance-level Transformer +
/// inverse-block-masking training target is a distinct topology
/// axis from every sibling.
pub const ARCH: &str = "eat";

/// `vokra.model.name` — canonical `eat-base` size point.
/// Sibling `eat-large` release exists in the upstream releases
/// page but is a distinct arch variant published as its own
/// `NAME` following the snac_24khz / snac_44khz pattern (added
/// via a separate future ModelKind).
pub const NAME: &str = "eat-base";

/// `vokra.model.category` — general audio-embedding (sibling of
/// `dasheng` / `beats` / `atst` / `m2d`; downstream music-tagging
/// / audio-classification / sound-event heads feed from the
/// encoder's hidden states).
pub const CATEGORY: &str = "audio-embedding";

/// `vokra.provenance.upstream_url` value — the GitHub tree the
/// release ships from. EAT is not hosted on HuggingFace, so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either. Sibling of `nsnet2::UPSTREAM_URL` /
/// `emotion2vec` / `beats::UPSTREAM_URL` posture.
pub const UPSTREAM_URL: &str = "github.com/cwx-worst-one/EAT";

/// Default SPDX. Upstream `cwx-worst-one/EAT` LICENSE via GitHub
/// API `/repos/cwx-worst-one/EAT/license` returns
/// `spdx_id: MIT` (task input 2026-08-13). A caller with a
/// different attestation may override at the outer boundary
/// (`--license <spdx>`).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// ---------------------------------------------------------------------------
// `vokra.eat.*` — EAT-base topology + Kaldi-fbank front-end axes.
//
// Every constant below is TRANSCRIBED from a primary source that was actually
// read (all fetched 2026-08-15). None is inferred from a sibling model, and any
// axis no reachable primary source states is OMITTED rather than guessed — see
// "Deliberate omissions" at the end of this block. A missing key is honest; a
// guessed one binds shape-valid garbage.
//
// Sources, referred to by letter in the per-constant docs:
//
//   [A] github.com/cwx-worst-one/EAT/blob/main/config/pretraining_AS2M.yaml
//       The repo README names this "the default configuration file for
//       pre-training"; it carries the task / model / modality overrides used
//       for the AudioSet-2M pre-trained `eat-base` release.
//   [B] github.com/cwx-worst-one/EAT/blob/main/models/EAT_pretraining.py
//       `Data2VecMultiConfig` dataclass defaults — the backbone axes that [A]
//       does not override.
//   [C] github.com/cwx-worst-one/EAT/blob/main/models/images.py
//       `D2vImageConfig` dataclass plus the `in_chans == 1` audio branch that
//       selects the spectrogram geometry.
//   [D] github.com/cwx-worst-one/EAT/blob/main/data/raw_audio_dataset.py
//       The training-side `torchaudio.compliance.kaldi.fbank(...)` call site
//       and the per-dataset normalisation profiles.
//   [E] github.com/cwx-worst-one/EAT/blob/main/feature_extract/feature_extract.py
//       The INFERENCE-side front-end: the same fbank arguments with
//       `sample_frequency=16000`, plus the `--norm_mean` / `--norm_std`
//       defaults. This is the call path a Vokra runtime must reproduce.
//   [F] docs.pytorch.org/audio/main/generated/torchaudio.compliance.kaldi.fbank.html
//       Documented defaults, cited ONLY for arguments [D]/[E] leave unset — so
//       the stamped value is upstream's effective value, not a Vokra choice.
//   [G] arXiv:2401.03497 (Chen et al. 2024, "EAT: Self-Supervised Pre-Training
//       with Efficient Audio Transformer"). Title + abstract only: the abstract
//       states NO numeric axis, so no constant here rests on it.
//
// Value encoding follows the sibling `vokra.wavlm.*` group: `u32` for counts
// and lengths, `f32` for genuine floats (`GgufBuilder::add_f32` round-trips
// IEEE-754 exactly, the same encoding `FrontendSpec` uses for `fmin` / `fmax` /
// `pre_emphasis`), `bool` for flags, `string` for the window name. Unlike
// `vokra.wavlm.*` there is no indexed axis-array: EAT's encoder is a uniform
// ViT-B stack with no per-layer axis, so manufacturing an array would add
// structure the upstream config does not have.
// ---------------------------------------------------------------------------

/// Transformer embedding width — 768 (`Data2VecMultiConfig.embed_dim` [B]).
///
/// Not overridden by [A]. Consistent with the README's "EAT-base: 88M
/// parameters (ViT-B backbone)", but transcribed from the dataclass rather
/// than inferred from the ViT-B label.
pub const EMBED_DIM: u32 = 768;

/// Transformer encoder depth — 12 blocks.
///
/// Stated twice and identically: `Data2VecMultiConfig.depth = 12` [B] and the
/// explicit `model.depth: 12` override in [A].
pub const DEPTH: u32 = 12;

/// Attention head count — 12 (`Data2VecMultiConfig.num_heads` [B]).
///
/// Not overridden by [A]. With [`EMBED_DIM`] = 768 this gives a 64-wide head,
/// but the head width itself is not stamped: it is a quotient, and the loader
/// should compute it rather than read two keys that can drift apart.
pub const NUM_HEADS: u32 = 12;

/// Feed-forward expansion ratio — 4.0 (`Data2VecMultiConfig.mlp_ratio` [B],
/// declared `float = 4`). Not overridden by [A].
pub const MLP_RATIO: f32 = 4.0;

/// LayerNorm epsilon — 1e-6.
///
/// Stated twice and identically: `Data2VecMultiConfig.norm_eps = 1e-6` [B] and
/// the explicit `model.norm_eps: 1e-6` override in [A]. Stamped as a real
/// `f32` (not the scaled-integer dance the older `vokra.wavlm.*` group uses)
/// because GGUF `FLOAT32` round-trips the IEEE-754 bits exactly.
pub const NORM_EPS: f32 = 1e-6;

/// `layer_norm_first` — `false` (`Data2VecMultiConfig.layer_norm_first` [B],
/// not overridden by [A]).
///
/// Stamped as the transcribed configuration value, NOT as an assertion about
/// where the norms sit inside a block: `models/modules.py AltBlock` is what
/// interprets this flag and that file has not been transcribed here. The
/// runtime binder's prose currently describes the stack as "ViT-style
/// pre-norm", which is in tension with a `false` flag — a forward wave must
/// reconcile the two against `AltBlock` rather than trusting either
/// description. Recorded here so the tension is visible in the artifact
/// instead of being resolved by guesswork at bind time.
pub const LAYER_NORM_FIRST: bool = false;

/// 2-D patch side, in spectrogram cells — 16 (`D2vImageConfig.patch_size` [C]).
/// Square: the same 16 applies to the time and frequency axes.
pub const PATCH_SIZE: u32 = 16;

/// Patch-embedding input channel count — 1 (`modalities.image.in_chans: 1`
/// in [A]).
///
/// This is an override, and a load-bearing one: `D2vImageConfig.in_chans` [C]
/// defaults to `3` for the ImageNet modality, and it is precisely the `== 1`
/// test in `models/images.py` that switches the geometry onto the audio branch
/// described by [`TARGET_LENGTH`] / [`N_MELS`].
pub const IN_CHANS: u32 = 1;

/// Fixed spectrogram length in frames — 1024.
///
/// Stated twice and identically: `task.target_length: 1024` in [A] and
/// `D2vImageConfig.target_length: int = 1024` [C]. At [`FBANK_FRAME_SHIFT_MS`]
/// = 10 ms this is the 10-second AudioSet clip the README describes. [D] pads
/// short clips with zeros and crops long ones to this length.
pub const TARGET_LENGTH: u32 = 1024;

/// Mel-bin count — 128.
///
/// One key for one number, deliberately, because upstream uses the same 128 in
/// two places that MUST agree: the `num_mel_bins=128` fbank argument in
/// [D]/[E], and the literal in `img_size = (modality_cfg.target_length, 128)`
/// in [C] that gives the patch grid its frequency extent. Stamping it once
/// makes it impossible for a reader to bind a front-end and a patch grid that
/// disagree.
pub const N_MELS: u32 = 128;

/// Time-axis patch count — 64.
///
/// Derived by UPSTREAM's own formula, not by Vokra: [C] builds
/// `img_size = (target_length, 128)` on the `in_chans == 1` branch and takes
/// `img_size[0] // patch_size[0]`, i.e. `1024 / 16`. The test module pins the
/// arithmetic against [`TARGET_LENGTH`] and [`PATCH_SIZE`] so the three cannot
/// drift apart.
pub const PATCH_GRID_TIME: u32 = 64;

/// Frequency-axis patch count — 8.
///
/// Same upstream formula as [`PATCH_GRID_TIME`], on the other axis:
/// `img_size[1] // patch_size[1]` = `128 / 16`. Pinned against [`N_MELS`] and
/// [`PATCH_SIZE`] in the test module.
pub const PATCH_GRID_FREQ: u32 = 8;

/// Patch-token count per clip — 512.
///
/// [C] computes this as `(img_size[1] // patch_size[1]) * (img_size[0] //
/// patch_size[0])` and its own inline comment states the result: "number of
/// patch -> 512". Transcribed rather than derived, and cross-checked against
/// [`PATCH_GRID_TIME`] × [`PATCH_GRID_FREQ`] in the test module.
pub const NUM_PATCHES: u32 = 512;

/// Extra (non-patch) tokens prepended to the sequence — 1
/// (`modalities.image.num_extra_tokens: 1` in [A]).
///
/// This is the CLS token that carries EAT's utterance-level half of the
/// Utterance-Frame Objective; [A] also sets `model.cls_loss: 1`, which is what
/// trains it. Its **position** in the sequence is deliberately NOT stamped:
/// that must be read off `models/images.py` [C] by the forward wave, and this
/// converter will not assert an index it has not transcribed.
pub const NUM_EXTRA_TOKENS: u32 = 1;

/// Positional-embedding grid height, in time patches — 768
/// (`D2vImageConfig.max_length` [C]).
///
/// [C] sizes the fixed 2-D sin-cos positional embedding over a
/// `(max_length, patch_grid_freq)` grid — a buffer larger than the
/// [`PATCH_GRID_TIME`] = 64 rows a 1024-frame clip actually consumes, which is
/// what lets the encoder accept variable-length spectrograms. Stamped so a
/// loader allocates the same buffer instead of sizing it to the fixed clip and
/// silently breaking on any other length.
pub const POS_EMBED_MAX_LENGTH: u32 = 768;

// --- Pre-training decoder (`modalities.image.decoder` in [A]) --------------
//
// The MAE-style reconstruction decoder is used ONLY during self-supervised
// pre-training; a feature-extraction forward stops at the encoder. These four
// axes are stamped so a loader walking the tensor manifest can positively
// identify the decoder tensors and skip them, rather than guessing which
// prefix belongs to the encoder path.

/// Pre-training decoder width — 768 (`decoder.decoder_dim: 768` in [A]).
pub const DECODER_DIM: u32 = 768;
/// Pre-training decoder convolution groups — 16 (`decoder.decoder_groups: 16`
/// in [A]).
pub const DECODER_GROUPS: u32 = 16;
/// Pre-training decoder convolution kernel — 3 (`decoder.decoder_kernel: 3`
/// in [A]).
pub const DECODER_KERNEL: u32 = 3;
/// Pre-training decoder depth — 6 (`decoder.decoder_layers: 6` in [A]).
pub const DECODER_LAYERS: u32 = 6;

// --- Kaldi fbank front-end --------------------------------------------------
//
// EAT's front-end is `torchaudio.compliance.kaldi.fbank`, NOT the librosa-style
// STFT+mel the `vokra.frontend.*` chunk group describes. The constants below
// are the exact argument set of the [D]/[E] call site, plus the [F] defaults
// for the arguments upstream leaves unset. They map 1:1 onto
// `vokra_ops::kaldi_fbank::KaldiFbankOpts`, which is what the forward wave will
// construct from them.
//
// WINDOW DIVERGENCE — read this before wiring the forward: upstream passes
// `window_type='hanning'` ([`FBANK_WINDOW_TYPE`]), while
// `vokra_ops::kaldi_fbank` currently hard-codes the Povey window (Hann^0.85).
// Those are different windows and would desync every feature. The op needs a
// window selector before it can serve EAT; stamping the upstream value makes
// that mismatch detectable at load instead of invisible in the output.

/// Front-end sample rate — 16 kHz.
///
/// [A] sets `task.downsr_16hz: true`, which resamples to `new_freq=16000` in
/// [D] before the fbank call; [E] hard-codes `sample_frequency=16000` on the
/// inference path. (The 32 kHz figure in `data/mae_image_dataset.py` is the
/// *file load* rate feeding that resample, not the front-end rate.)
pub const FBANK_SAMPLE_RATE: u32 = 16_000;

/// Analysis frame length — 25 ms (= 400 samples at [`FBANK_SAMPLE_RATE`]).
///
/// Upstream passes no `frame_length`, so the effective value is torchaudio's
/// documented default `frame_length: float = 25.0` [F].
pub const FBANK_FRAME_LENGTH_MS: u32 = 25;

/// Frame hop — 10 ms (= 160 samples at [`FBANK_SAMPLE_RATE`]). Passed
/// explicitly as `frame_shift=10` in [D] and [E].
pub const FBANK_FRAME_SHIFT_MS: u32 = 10;

/// Analysis window — `"hanning"`, passed explicitly as `window_type='hanning'`
/// in [D] and [E].
///
/// See the WINDOW DIVERGENCE note above: `vokra_ops::kaldi_fbank` hard-codes
/// Povey today, so this value is the one that must drive a window selector.
pub const FBANK_WINDOW_TYPE: &str = "hanning";

/// `htk_compat` — `true`, passed explicitly in [D] and [E].
///
/// Stamped verbatim rather than interpreted. Its effect inside
/// `torchaudio.compliance.kaldi.fbank` interacts with `use_energy` (which
/// upstream sets to `false`, see [`FBANK_USE_ENERGY`]); the torchaudio *source*
/// has not been transcribed here, so this converter records the argument
/// upstream actually passes and leaves its semantics to be read off [F] and
/// the torchaudio implementation by the forward wave.
pub const FBANK_HTK_COMPAT: bool = true;

/// `use_energy` — `false`, passed explicitly in [D] and [E].
///
/// Load-bearing for feature WIDTH: with energy off, the frame vector is
/// exactly [`N_MELS`] = 128 wide, which is what makes the `128 / 16 = 8`
/// frequency-patch grid come out whole.
pub const FBANK_USE_ENERGY: bool = false;

/// `dither` — `0.0`, passed explicitly in [D] and [E]. Dithering off is what
/// makes the front-end deterministic and therefore parity-testable.
pub const FBANK_DITHER: f32 = 0.0;

/// `low_freq` — 20.0 Hz, torchaudio's documented default [F] (upstream passes
/// nothing).
pub const FBANK_LOW_FREQ: f32 = 20.0;

/// `high_freq` — `0.0`, torchaudio's documented default [F] (upstream passes
/// nothing).
///
/// Stamped RAW, in Kaldi's own encoding: a non-positive `high_freq` means
/// "Nyquist + high_freq", so `0.0` selects the Nyquist frequency. The same
/// convention is already documented on `KaldiFbankOpts::high_freq`, so the raw
/// value drops straight in. Resolving it to 8000 here would bake a derivation
/// into the artifact and break the moment [`FBANK_SAMPLE_RATE`] changes.
pub const FBANK_HIGH_FREQ: f32 = 0.0;

/// `preemphasis_coefficient` — 0.97, torchaudio's documented default [F]
/// (upstream passes nothing).
pub const FBANK_PREEMPH_COEFF: f32 = 0.97;

/// `remove_dc_offset` — `true`, torchaudio's documented default [F] (upstream
/// passes nothing). Kaldi removes the DC offset per frame, not per utterance.
pub const FBANK_REMOVE_DC_OFFSET: bool = true;

/// `round_to_power_of_two` — `true`, torchaudio's documented default [F]
/// (upstream passes nothing).
///
/// This is why no `n_fft` key exists in this group: Kaldi derives the FFT size
/// from the frame length, so at 25 ms / 16 kHz the 400-sample frame is padded
/// to 512. Stamping a separate `n_fft` would let it drift out of step with
/// [`FBANK_FRAME_LENGTH_MS`].
pub const FBANK_ROUND_TO_POWER_OF_TWO: bool = true;

/// `snip_edges` — `true`, torchaudio's documented default [F] (upstream passes
/// nothing). Snip-edges framing means no centre padding, so the frame count is
/// `1 + (n_samples - frame_length) / frame_shift`.
pub const FBANK_SNIP_EDGES: bool = true;

/// `use_power` — `true`, torchaudio's documented default [F] (upstream passes
/// nothing): the mel filters see `|X|²`, not `|X|`.
pub const FBANK_USE_POWER: bool = true;

/// `use_log_fbank` — `true`, torchaudio's documented default [F] (upstream
/// passes nothing).
pub const FBANK_USE_LOG: bool = true;

/// `subtract_mean` — `false`, torchaudio's documented default [F] (upstream
/// passes nothing).
///
/// Stamped explicitly because the in-repo sibling front-end
/// `KaldiFbankOpts::camplus()` sets this to `true` (per-utterance CMN). EAT
/// does NOT do CMN — it applies the fixed affine normalisation described by
/// [`FBANK_NORM_MEAN`] instead. Copying the CAM++ preset would silently shift
/// every EAT feature.
pub const FBANK_SUBTRACT_MEAN: bool = false;

/// Feature normalisation mean — `-4.268` (the AudioSet profile).
///
/// [D] carries three profiles and [E] exposes the AudioSet one as its
/// `--norm_mean` default; `eat-base` is the AudioSet-2M pre-trained release
/// ([`NAME`]), so that is the profile stamped. The other two profiles are
/// dataset-specific fine-tuning constants recorded in [D] for completeness:
/// ESC-50 uses `-6.627` / `5.359` and SPC-v2 uses `-6.846` / `5.565`. They are
/// deliberately NOT stamped — this artifact describes one release, and a
/// downstream fine-tune must carry its own.
pub const FBANK_NORM_MEAN: f32 = -4.268;

/// Feature normalisation standard deviation — `4.569` (the AudioSet profile;
/// see [`FBANK_NORM_MEAN`] for the other profiles and why they are omitted).
pub const FBANK_NORM_STD: f32 = 4.569;

/// Divisor multiplier applied to [`FBANK_NORM_STD`] — `2.0`.
///
/// Upstream normalises as `(feats - norm_mean) / (norm_std * 2)` in BOTH [D]
/// and [E] — note the `* 2`. This key exists so the full formula is
/// recoverable from metadata alone: a reader who assumed the conventional
/// `(x - mean) / std` would be off by exactly a factor of two, with no shape
/// error to reveal it.
pub const FBANK_NORM_STD_MULTIPLIER: f32 = 2.0;

// --- Deliberate omissions ---------------------------------------------------
//
// Recorded so a later reader can tell "not yet transcribed" from "checked, and
// upstream does not state it":
//
//   * `n_fft` — not an upstream axis at all. Kaldi derives the FFT size from
//     the frame length via `round_to_power_of_two`; see
//     [`FBANK_ROUND_TO_POWER_OF_TWO`].
//   * `mel_norm` / `pad_mode` — two of the thirteen `vokra.frontend.*` fields
//     have no Kaldi counterpart. Kaldi applies no Slaney-style area
//     normalisation, and `snip_edges=true` means there is no padding mode to
//     name. This is why the front-end is stamped as the raw Kaldi argument set
//     under `vokra.eat.fbank_*` rather than being translated into a synthetic
//     `vokra.frontend.*` spec: CLAUDE.md requires that spec to be bit-exact,
//     and a translated one is exactly the silent desync it guards against.
//   * `input_size` (= 224 in [C]) — the ImageNet branch only. The `in_chans ==
//     1` audio branch replaces it with `(target_length, 128)`, so stamping it
//     would advertise a geometry EAT never uses for audio.
//   * head width — a quotient of [`EMBED_DIM`] and [`NUM_HEADS`]; see
//     [`NUM_HEADS`].
//   * masking axes (`inverse_mask`, `mask_prob`, `mask_length`, …), EMA decay,
//     `clone_batch`, `average_top_k_layers` — pre-training-only knobs. The
//     inference encoder sees every patch, so stamping them would imply the
//     runtime honours them.
//   * downstream label spaces (AudioSet / ESC-50 / SPC-2) — the upstream
//     release ships those heads as separate fine-tunes; this converter targets
//     the pre-trained encoder, which contains none of them.
//   * `eat-large` axes — a different size point published under its own
//     [`NAME`] via a separate `ModelKind` (see [`NAME`]'s docs).

const UPSTREAM_SOURCE: &str = "cwx-worst-one/EAT (Effective Audio Transformer, utterance-level Transformer + \
     inverse block masking self-supervised audio encoder, ~86M params base, Chen et al. \
     arXiv:2401.03497, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// `vokra.eat.*` chunk keys. Private here and mirrored on the runtime side (the
// `vokra.wavlm.*` posture): `vokra-models` must not gain a dependency edge onto
// `vokra-convert`, so the binder re-declares these spellings and its tests pin
// them, which makes a rename on either side a same-commit failure.

/// `vokra.eat.embed_dim` — Transformer width (`UINT32`).
const KEY_EAT_EMBED_DIM: &str = "vokra.eat.embed_dim";
/// `vokra.eat.depth` — encoder block count (`UINT32`).
const KEY_EAT_DEPTH: &str = "vokra.eat.depth";
/// `vokra.eat.num_heads` — attention head count (`UINT32`).
const KEY_EAT_NUM_HEADS: &str = "vokra.eat.num_heads";
/// `vokra.eat.mlp_ratio` — feed-forward expansion ratio (`FLOAT32`).
const KEY_EAT_MLP_RATIO: &str = "vokra.eat.mlp_ratio";
/// `vokra.eat.norm_eps` — LayerNorm epsilon (`FLOAT32`).
const KEY_EAT_NORM_EPS: &str = "vokra.eat.norm_eps";
/// `vokra.eat.layer_norm_first` — transcribed `layer_norm_first` flag (`BOOL`).
const KEY_EAT_LAYER_NORM_FIRST: &str = "vokra.eat.layer_norm_first";
/// `vokra.eat.patch_size` — square patch side in spectrogram cells (`UINT32`).
const KEY_EAT_PATCH_SIZE: &str = "vokra.eat.patch_size";
/// `vokra.eat.in_chans` — patch-embedding input channels (`UINT32`).
const KEY_EAT_IN_CHANS: &str = "vokra.eat.in_chans";
/// `vokra.eat.target_length` — fixed spectrogram length in frames (`UINT32`).
const KEY_EAT_TARGET_LENGTH: &str = "vokra.eat.target_length";
/// `vokra.eat.n_mels` — mel-bin count / frequency extent (`UINT32`).
const KEY_EAT_N_MELS: &str = "vokra.eat.n_mels";
/// `vokra.eat.patch_grid_time` — time-axis patch count (`UINT32`).
const KEY_EAT_PATCH_GRID_TIME: &str = "vokra.eat.patch_grid_time";
/// `vokra.eat.patch_grid_freq` — frequency-axis patch count (`UINT32`).
const KEY_EAT_PATCH_GRID_FREQ: &str = "vokra.eat.patch_grid_freq";
/// `vokra.eat.num_patches` — patch tokens per clip (`UINT32`).
const KEY_EAT_NUM_PATCHES: &str = "vokra.eat.num_patches";
/// `vokra.eat.num_extra_tokens` — prepended non-patch tokens (`UINT32`).
const KEY_EAT_NUM_EXTRA_TOKENS: &str = "vokra.eat.num_extra_tokens";
/// `vokra.eat.pos_embed_max_length` — positional-grid height (`UINT32`).
const KEY_EAT_POS_EMBED_MAX_LENGTH: &str = "vokra.eat.pos_embed_max_length";
/// `vokra.eat.decoder_dim` — pre-training decoder width (`UINT32`).
const KEY_EAT_DECODER_DIM: &str = "vokra.eat.decoder_dim";
/// `vokra.eat.decoder_groups` — pre-training decoder conv groups (`UINT32`).
const KEY_EAT_DECODER_GROUPS: &str = "vokra.eat.decoder_groups";
/// `vokra.eat.decoder_kernel` — pre-training decoder conv kernel (`UINT32`).
const KEY_EAT_DECODER_KERNEL: &str = "vokra.eat.decoder_kernel";
/// `vokra.eat.decoder_layers` — pre-training decoder depth (`UINT32`).
const KEY_EAT_DECODER_LAYERS: &str = "vokra.eat.decoder_layers";
/// `vokra.eat.fbank_sample_rate` — front-end sample rate, Hz (`UINT32`).
const KEY_EAT_FBANK_SAMPLE_RATE: &str = "vokra.eat.fbank_sample_rate";
/// `vokra.eat.fbank_frame_length_ms` — analysis frame length, ms (`UINT32`).
const KEY_EAT_FBANK_FRAME_LENGTH_MS: &str = "vokra.eat.fbank_frame_length_ms";
/// `vokra.eat.fbank_frame_shift_ms` — frame hop, ms (`UINT32`).
const KEY_EAT_FBANK_FRAME_SHIFT_MS: &str = "vokra.eat.fbank_frame_shift_ms";
/// `vokra.eat.fbank_window_type` — analysis window name (`STRING`).
const KEY_EAT_FBANK_WINDOW_TYPE: &str = "vokra.eat.fbank_window_type";
/// `vokra.eat.fbank_htk_compat` — Kaldi `htk_compat` argument (`BOOL`).
const KEY_EAT_FBANK_HTK_COMPAT: &str = "vokra.eat.fbank_htk_compat";
/// `vokra.eat.fbank_use_energy` — Kaldi `use_energy` argument (`BOOL`).
const KEY_EAT_FBANK_USE_ENERGY: &str = "vokra.eat.fbank_use_energy";
/// `vokra.eat.fbank_dither` — Kaldi `dither` argument (`FLOAT32`).
const KEY_EAT_FBANK_DITHER: &str = "vokra.eat.fbank_dither";
/// `vokra.eat.fbank_low_freq` — low mel band edge, Hz (`FLOAT32`).
const KEY_EAT_FBANK_LOW_FREQ: &str = "vokra.eat.fbank_low_freq";
/// `vokra.eat.fbank_high_freq` — high mel band edge in Kaldi encoding
/// (`FLOAT32`).
const KEY_EAT_FBANK_HIGH_FREQ: &str = "vokra.eat.fbank_high_freq";
/// `vokra.eat.fbank_preemph_coeff` — pre-emphasis coefficient (`FLOAT32`).
const KEY_EAT_FBANK_PREEMPH_COEFF: &str = "vokra.eat.fbank_preemph_coeff";
/// `vokra.eat.fbank_remove_dc_offset` — per-frame DC removal (`BOOL`).
const KEY_EAT_FBANK_REMOVE_DC_OFFSET: &str = "vokra.eat.fbank_remove_dc_offset";
/// `vokra.eat.fbank_round_to_power_of_two` — FFT-size rounding (`BOOL`).
const KEY_EAT_FBANK_ROUND_TO_POWER_OF_TWO: &str = "vokra.eat.fbank_round_to_power_of_two";
/// `vokra.eat.fbank_snip_edges` — snip-edges framing (`BOOL`).
const KEY_EAT_FBANK_SNIP_EDGES: &str = "vokra.eat.fbank_snip_edges";
/// `vokra.eat.fbank_use_power` — power vs. magnitude spectrum (`BOOL`).
const KEY_EAT_FBANK_USE_POWER: &str = "vokra.eat.fbank_use_power";
/// `vokra.eat.fbank_use_log` — log mel energies (`BOOL`).
const KEY_EAT_FBANK_USE_LOG: &str = "vokra.eat.fbank_use_log";
/// `vokra.eat.fbank_subtract_mean` — per-utterance CMN (`BOOL`).
const KEY_EAT_FBANK_SUBTRACT_MEAN: &str = "vokra.eat.fbank_subtract_mean";
/// `vokra.eat.fbank_norm_mean` — feature normalisation mean (`FLOAT32`).
const KEY_EAT_FBANK_NORM_MEAN: &str = "vokra.eat.fbank_norm_mean";
/// `vokra.eat.fbank_norm_std` — feature normalisation std (`FLOAT32`).
const KEY_EAT_FBANK_NORM_STD: &str = "vokra.eat.fbank_norm_std";
/// `vokra.eat.fbank_norm_std_multiplier` — divisor multiplier (`FLOAT32`).
const KEY_EAT_FBANK_NORM_STD_MULTIPLIER: &str = "vokra.eat.fbank_norm_std_multiplier";

/// Outcome of an EAT conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`beats` / `dasheng` /
/// `mert` / `muq` / `yamnet`) — the invariant
/// `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EatReport {
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

/// Converts an EAT safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning an [`EatReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"mit"`,
/// `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_eat_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<EatReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    // ---- `vokra.eat.*` topology group -----------------------------------
    // Purely additive: the arch / name / category / provenance stamps above
    // are untouched, so an artifact produced before this group existed and
    // one produced after differ only by these keys. The values are
    // transcribed constants (see their declarations for the primary source
    // each came from) — the reader side in `vokra-models::eat` looks up
    // exactly these key spellings.
    b.add_u32(KEY_EAT_EMBED_DIM, EMBED_DIM);
    b.add_u32(KEY_EAT_DEPTH, DEPTH);
    b.add_u32(KEY_EAT_NUM_HEADS, NUM_HEADS);
    b.add_f32(KEY_EAT_MLP_RATIO, MLP_RATIO);
    b.add_f32(KEY_EAT_NORM_EPS, NORM_EPS);
    b.add_bool(KEY_EAT_LAYER_NORM_FIRST, LAYER_NORM_FIRST);
    b.add_u32(KEY_EAT_PATCH_SIZE, PATCH_SIZE);
    b.add_u32(KEY_EAT_IN_CHANS, IN_CHANS);
    b.add_u32(KEY_EAT_TARGET_LENGTH, TARGET_LENGTH);
    b.add_u32(KEY_EAT_N_MELS, N_MELS);
    b.add_u32(KEY_EAT_PATCH_GRID_TIME, PATCH_GRID_TIME);
    b.add_u32(KEY_EAT_PATCH_GRID_FREQ, PATCH_GRID_FREQ);
    b.add_u32(KEY_EAT_NUM_PATCHES, NUM_PATCHES);
    b.add_u32(KEY_EAT_NUM_EXTRA_TOKENS, NUM_EXTRA_TOKENS);
    b.add_u32(KEY_EAT_POS_EMBED_MAX_LENGTH, POS_EMBED_MAX_LENGTH);
    b.add_u32(KEY_EAT_DECODER_DIM, DECODER_DIM);
    b.add_u32(KEY_EAT_DECODER_GROUPS, DECODER_GROUPS);
    b.add_u32(KEY_EAT_DECODER_KERNEL, DECODER_KERNEL);
    b.add_u32(KEY_EAT_DECODER_LAYERS, DECODER_LAYERS);
    b.add_u32(KEY_EAT_FBANK_SAMPLE_RATE, FBANK_SAMPLE_RATE);
    b.add_u32(KEY_EAT_FBANK_FRAME_LENGTH_MS, FBANK_FRAME_LENGTH_MS);
    b.add_u32(KEY_EAT_FBANK_FRAME_SHIFT_MS, FBANK_FRAME_SHIFT_MS);
    b.add_string(KEY_EAT_FBANK_WINDOW_TYPE, FBANK_WINDOW_TYPE);
    b.add_bool(KEY_EAT_FBANK_HTK_COMPAT, FBANK_HTK_COMPAT);
    b.add_bool(KEY_EAT_FBANK_USE_ENERGY, FBANK_USE_ENERGY);
    b.add_f32(KEY_EAT_FBANK_DITHER, FBANK_DITHER);
    b.add_f32(KEY_EAT_FBANK_LOW_FREQ, FBANK_LOW_FREQ);
    b.add_f32(KEY_EAT_FBANK_HIGH_FREQ, FBANK_HIGH_FREQ);
    b.add_f32(KEY_EAT_FBANK_PREEMPH_COEFF, FBANK_PREEMPH_COEFF);
    b.add_bool(KEY_EAT_FBANK_REMOVE_DC_OFFSET, FBANK_REMOVE_DC_OFFSET);
    b.add_bool(
        KEY_EAT_FBANK_ROUND_TO_POWER_OF_TWO,
        FBANK_ROUND_TO_POWER_OF_TWO,
    );
    b.add_bool(KEY_EAT_FBANK_SNIP_EDGES, FBANK_SNIP_EDGES);
    b.add_bool(KEY_EAT_FBANK_USE_POWER, FBANK_USE_POWER);
    b.add_bool(KEY_EAT_FBANK_USE_LOG, FBANK_USE_LOG);
    b.add_bool(KEY_EAT_FBANK_SUBTRACT_MEAN, FBANK_SUBTRACT_MEAN);
    b.add_f32(KEY_EAT_FBANK_NORM_MEAN, FBANK_NORM_MEAN);
    b.add_f32(KEY_EAT_FBANK_NORM_STD, FBANK_NORM_STD);
    b.add_f32(KEY_EAT_FBANK_NORM_STD_MULTIPLIER, FBANK_NORM_STD_MULTIPLIER);

    let mut report = EatReport::default();
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
            "vokra-convert-eat-{tag}-{}-{n}",
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
    fn f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // EAT uses a `patch_embed` conv + Transformer encoder blocks
        // — realistic upstream state-dict name from the utterance-level
        // Transformer body.
        let st = safetensors_one("patch_embed.proj.weight", "F32", &[1, 3], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_eat_file(&inp, &outp, None).expect("convert F32");
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
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_URL), UPSTREAM_URL);
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Permissive.as_str(),
            "mit must resolve to Permissive"
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
        let st = safetensors_one("blocks.0.attn.qkv.weight", "BF16", &[2, 2], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_eat_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("blocks.0.attn.qkv.weight")
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

        let r = convert_eat_file(&inp, &outp, None).expect("convert F32");
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
        assert_eq!(read_u64(KEY_EAT_EMBED_DIM), u64::from(EMBED_DIM));
        assert_eq!(read_u64(KEY_EAT_EMBED_DIM), 768);
        assert_eq!(read_u64(KEY_EAT_DEPTH), u64::from(DEPTH));
        assert_eq!(read_u64(KEY_EAT_DEPTH), 12);
        assert_eq!(read_u64(KEY_EAT_NUM_HEADS), u64::from(NUM_HEADS));
        assert_eq!(read_u64(KEY_EAT_NUM_HEADS), 12);

        // Purely additive: the pre-existing stamps are untouched.
        assert_eq!(
            g.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            g.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            g.get(KEY_PROVENANCE_UPSTREAM_URL).and_then(|v| v.as_str()),
            Some(UPSTREAM_URL)
        );
    }

    #[test]
    fn license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        convert_eat_file(&inp, &outp, Some("apache-2.0")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
