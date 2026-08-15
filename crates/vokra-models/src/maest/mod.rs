//! **MAEST** — "Music Audio Efficient Spectrogram Transformer"
//! (`mtg-upf/discogs-maest-30s-pw-129e`, **cc-by-nc-sa-4.0**) — runtime binder
//! for the `maest` converter arch (Wave C2 2026-08-15, loud-partial per the
//! `atst` / `m2d` / `emotion2vec` / `wavlm` / `panns` / `redimnet` precedent —
//! CLAUDE.md 教訓 (a):「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/maest.rs` has been stamping
//! `vokra.model.arch = "maest"` since the 2026-08-13 SSL audio-encoder wave,
//! but a workspace-wide grep proved that **nothing read that arch string
//! back** — a converted MAEST checkpoint was unloadable. This module is that
//! consumer.
//!
//! # Primary sources
//!
//! Every fact below is transcribed from the converter's own module docstring
//! (`crates/vokra-convert/src/models/maest.rs`) and its [`ModelKind`] entry in
//! `crates/vokra-convert/src/lib.rs`, which together are this repository's
//! primary-source record for MAEST. Nothing here is re-derived from memory.
//!
//! - Upstream release: <https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e>
//! - Paper: Alonso-Jiménez et al. 2023, ISMIR — <https://arxiv.org/abs/2309.16418>
//! - Backbone: the HF `config` records `model_type:
//!   audio-spectrogram-transformer` and `architectures:
//!   ["ASTForAudioClassification"]` (verified via the HF cardData API on
//!   2026-08-13 and recorded by the converter).
//! - Scale: safetensors `parameters.F32: 86,858,128` per the HF API
//!   ([`UPSTREAM_PARAM_COUNT_F32`]).
//! - Licence: HF cardData `license: cc-by-nc-sa-4.0` → the **T4 tier +
//!   ShareAlike cascade**, i.e. [`LicenseClass::NonCommercialShareAlike`].
//!
//! [`ModelKind`]: https://docs.rs/vokra-convert
//!
//! # What MAEST is — and why it is the music-domain member of the SSL fleet
//!
//! MAEST is a self-supervised **music** encoder: an AST (Audio Spectrogram
//! Transformer) backbone — a ViT-style patch-wise Transformer over a log-mel
//! spectrogram — pretrained on the MTG Discogs4All **music-tagger** dataset.
//! The `30s-pw-129e` variant this [`NAME`] tracks is 30-second, patch-wise, 129
//! epochs.
//!
//! Unlike its general-audio siblings (`atst` / `eat` / `m2d` / `dasheng`, all
//! of which ship a bare encoder and no task head), MAEST's upstream
//! `architectures` string is `ASTForAudioClassification` — i.e. the release
//! **does** carry a tagging head over a Discogs label taxonomy. The converter
//! is a verbatim float pass-through, so if that head is in the checkpoint its
//! tensors ride through under their upstream `state_dict` names and land on
//! disk. This binder therefore exposes a tag surface — but see the honesty
//! constraint in "Label taxonomy" below.
//!
//! ```text
//! PCM (mono f32)
//!   -> log-mel spectrogram front-end                    ← **loud-partial**
//!        (every axis IS stamped — sample rate, n_fft, hop, win, window, mel
//!         scale / norm, fmin / fmax, logC + multiplier, mean / std — EXCEPT
//!         the STFT framing / centering convention, which the converter
//!         deliberately omits because no primary source states it).
//!   -> log-mel plane [num_mel_bins, n_frames]           ── caller-supplied
//!   -> 2-D patch embedding over the mel plane           ── **real**
//!   -> pre-norm Transformer encoder (~87M-param AST)    ── **real**
//!   -> final LayerNorm -> token hidden states           ── [`MaestEncoder::encode_mel`]
//!   -> mean(CLS, distillation) -> clip embedding        ── [`MaestEncoder::embed_mel`]
//!   -> ASTMLPHead -> Discogs tag logits                 ── [`MaestEncoder::tag_mel`]
//! ```
//!
//! # What changed, and what is still missing
//!
//! This module was written when three things were true and are no longer:
//!
//! 1. The converter stamped no `vokra.maest.*` axis group. It now stamps the
//!    full topology + front-end group, so [`MaestConfig`] reads the axes off
//!    the artifact instead of the binder guessing them.
//! 2. `vokra-ops` had no ViT primitive. [`vokra_ops::vit`] now supplies the 2-D
//!    patch embedding, prepended tokens, additive positional table, pre-norm
//!    Transformer stack and final norm — the gap that was shared across the
//!    whole SSL fleet.
//! 3. The upstream `state_dict` names were unverified. They are now transcribed
//!    from [`PRIMARY_SOURCE_HF_AST_MODELING`], the HuggingFace AST modelling
//!    file at the tag the checkpoint's own config names
//!    (`transformers_version: "4.34.0.dev0"` → `v4.34.0`).
//!
//! **What remains** is one axis, and the loud-partial names only that: the
//! **STFT framing / centering convention** of the log-mel front end. The
//! converter records this as a deliberate omission — no primary source it
//! reached states whether the analysis is centred or which padding mode it
//! uses — and it writes no `vokra.frontend.*` bit-exact group for the same
//! reason. Choosing centred vs non-centred shifts every frame by half a
//! window: shape-valid, numerically wrong, and silent. So the PCM-in surfaces
//! [`Maest::encode`] / [`Maest::embed`] / [`Maest::tag`] stay loud-partial,
//! while the mel-plane-in surfaces on [`MaestEncoder`] are real.
//!
//! # Label taxonomy — the count has two witnesses, the names have none
//!
//! The converter stamps the label **count** (`vokra.maest.num_labels`, from
//! `config.json`'s `id2label` cardinality) but **no label list**. This module
//! still contains no taxonomy constant of its own: [`Maest::label_count`] reads
//! the head projection's leading dimension off disk — a PyTorch `nn.Linear`
//! weight is `[out_features, in_features]` — so the stamp and the payload are
//! independent witnesses that the head binding cross-checks. Any ambiguous
//! shape on disk yields `None`, never a fallback number.
//!
//! The label **names** are unrecoverable from the artifact. That does not block
//! [`MaestEncoder::tag_mel`], whose return type is logits, but it does mean a
//! caller cannot map logit index `i` onto a Discogs genre / mood / instrument /
//! era string from the GGUF alone.
//!
//! # Real / loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! **Real**:
//!
//! - [`Maest::from_gguf`] with **strict** `vokra.model.arch == "maest"`
//!   verification. A sibling SSL-encoder GGUF handed here by mistake fails with
//!   a message naming **both** tags and enumerating the whole
//!   audio/music-embedding neighbourhood — `ast` most sharply, since MAEST
//!   shares its *backbone* but not its objective, its domain or its taxonomy
//!   (FR-EX-08 — never a silent misroute).
//! - [`MaestConfig::from_gguf`] **strict** axis-group reading: every stamped
//!   key required, a missing one a loud [`VokraError::ModelLoad`] naming it, and
//!   no primary-source constant fallback.
//! - [`MaestConfig::vit_attrs`] mapping onto [`vokra_ops::vit::ViTAttrs`], with
//!   the `mlp_ratio` round-trip and [`ViTAttrs::validate`] both enforced.
//! - [`MaestWeights::from_gguf`] tensor-manifest binding over the verbatim
//!   upstream `state_dict` names the converter passes through, with a non-empty
//!   gate plus [`MaestWeights::require_tensor`] /
//!   [`MaestWeights::require_tensor_dims`] lookups that name the missing
//!   tensor, or **both** the expected and the actual dims, and
//!   [`MaestWeights::detect_tensor_prefix`] probing which `state_dict` prefix
//!   the artifact actually uses.
//! - [`Maest::encoder`] weight binding and the [`MaestEncoder`] forward:
//!   patch embedding, prepended CLS + distillation tokens, positional table,
//!   12-block pre-norm stack, final norm, DeiT-style pooling and the
//!   `ASTMLPHead` tagging head.
//! - Tag-head discovery from disk: [`MaestWeights::tag_head_tensors`] /
//!   [`MaestWeights::label_count_from_disk`], reporting only what the artifact
//!   contains.
//! - Metadata surfacing: [`Maest::name`] / [`Maest::category`] /
//!   [`Maest::upstream_hf`] / [`Maest::model_id`] / [`Maest::source`] read back
//!   the converter's stamps.
//! - Weight-licence + FR-MD-09 attribution surfacing, fail-closing to
//!   [`LicenseClass::Unknown`] when the artifact carries no stamp, and the
//!   compliance-gated [`Maest::from_gguf_with_policy`] / [`Maest::from_path`] /
//!   [`Maest::from_path_with_policy`] entry points.
//!
//! **Loud-partial** — [`Maest::encode`], [`Maest::embed`] and [`Maest::tag`]
//! return [`VokraError::UnsupportedOp`] for the unstamped STFT framing
//! convention described above, and for nothing else.
//!
//! No fabricated hidden states, embeddings or tag logits are ever emitted
//! (FR-EX-08 — no silent partial output). A follow-up wave flips the last
//! switch by establishing the framing convention from a primary source — most
//! likely by reading `feature_extraction_maest.py`'s framing call against a
//! real checkpoint — and teaching the converter to stamp it.
//!
//! # Numerical parity is NOT claimed
//!
//! The forward is transcribed from upstream, but no parity run against a real
//! MAEST checkpoint has happened in this repository: the weights are gated
//! CC-BY-NC-SA 4.0 and no fixture exists. The tests below therefore assert
//! structure, shape, determinism and finiteness — never an expected numeric
//! value, since inventing one would be fabrication with the appearance of
//! verification. Two axes in particular would survive a shape-only check while
//! being numerically wrong, and are called out at their binding sites: the
//! pre-norm vs post-norm LayerNorm ordering, and the erf vs tanh GELU flavour.
//!
//! # Sibling family distinctness (SSL audio/music-embedding neighbourhood)
//!
//! [`ARCH`] = `"maest"` is deliberately distinct from every sibling:
//!
//! - `ast` — the **same AST backbone**, but supervised, fine-tuned on AudioSet,
//!   general-audio, and published under a different licence tier
//!   (`bsd-3-clause`). Backbone identity is not topology identity: the
//!   objective, domain, head and taxonomy all differ, so this is the sharpest
//!   confusable pair in the fleet.
//! - `atst` — BYOL-style teacher-student patchout (general audio);
//! - `beats` — iterative acoustic tokenizer + masked acoustic modelling;
//! - `eat` — utterance-level MAE with efficient inverse block masking;
//! - `dasheng` — universal MAE;
//! - `m2d` — masked modelling **duo** (dual online + target branch);
//! - `mert` — HuBERT-derived masked prediction (music);
//! - `muq` — Mel-RVQ + BEATs teacher (music);
//! - `yamnet` / `panns` — supervised audio-tagging CNNs, not SSL at all;
//! - `clap` — contrastive language-audio pretraining (text tower attached);
//! - `hubert` / `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec` — the wav2vec2
//!   lineage, whose encoders sit on a **raw-waveform 1-D conv stem** rather
//!   than a log-mel patch grid.
//!
//! Sharing an arch tag would let runtime dispatch bind, say, an AudioSet
//! 527-class head or a raw-waveform conv stem over a Discogs music-tagger
//! checkpoint (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] are **mirrors of the converter's constants** — the
//! same rule every sibling binder (`atst` / `m2d` / `emotion2vec` / `wavlm` /
//! `panns` / `redimnet` / `canary_1b_flash`) follows so `vokra-models` does not
//! gain a dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # Licence posture — T4 + ShareAlike, fail-closed
//!
//! The converter stamps `cc-by-nc-sa-4.0` → [`LicenseClass::NonCommercialShareAlike`],
//! whose [`LicenseClass::requires_research_flag`] is `true`. A correctly
//! stamped MAEST artifact is therefore **refused** under
//! [`CompliancePolicy::strict`] and loads only with an explicit research opt-in
//! ([`CompliancePolicy::with_research_license`], `VOKRA_ALLOW_RESEARCH_LICENSE=1`,
//! or [`ComplianceLevel::Research`]) — that refusal is the correct behaviour,
//! not a bug. Three obligations cascade: **NonCommercial** (no commercial use
//! without a separate licence from the MTG group), **ShareAlike** (any
//! downstream distribution stays CC-BY-NC-SA 4.0 —
//! [`LicenseClass::requires_license_preserved`]), and **BY** (attribution —
//! [`LicenseClass::requires_attribution`]).
//!
//! Note that the converter's `stamp_provenance` call writes weight-licence,
//! SPDX, model id and source but **not** `vokra.provenance.attribution`, so
//! [`Maest::attribution`] reads `None` on a converter-produced artifact today
//! even though the BY cascade obliges a downstream to display credit. That is a
//! recorded gap, surfaced rather than papered over.
//!
//! This binder only *surfaces* whatever class the artifact carries;
//! `docs/license-audit.md` §3.1 sign-off stays **blank** (owner-only per memory
//! `[[feedback-license-signoff-primary-source]]` — CC does not sign, and does
//! not treat a converter default as a sign-off).
//!
//! [`ComplianceLevel::Research`]: vokra_core::ComplianceLevel::Research
//!
//! # No ONNX / no pickle (permanent)
//!
//! MAEST ships as single-file safetensors (`model.safetensors`); the upstream
//! repo also carries a legacy `pytorch_model.bin` pickle which Vokra never
//! reads. This runtime **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};
use vokra_ops::vit::{
    GeluKind, PatchEmbedWeights, PatchGrid, PosEmbedPolicy, ViTAttnWeights, ViTAttrs,
    ViTBlockWeights, ViTEncoder, ViTMlpWeights, ViTWeights,
};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/maest.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model maest-30s-pw-129e`.
///
/// Distinct from every sibling SSL audio/music-embedding arch tag (`ast` /
/// `atst` / `beats` / `eat` / `dasheng` / `m2d` / `mert` / `muq` / `yamnet` /
/// `panns` / `clap`) and from the wav2vec2 lineage (`hubert` /
/// `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec`). The `ast` pair is the sharpest
/// one: MAEST shares AST's backbone but not its objective (Discogs music-tagger
/// SSL vs supervised AudioSet fine-tuning), its domain, its head or its
/// taxonomy — silently sharing a tag would misroute runtime dispatch (FR-EX-08,
/// see the module docstring "Sibling family distinctness" section).
pub const ARCH: &str = "maest";

/// Expected `vokra.model.name` value written by the converter — the canonical
/// `30s-pw-129e` release variant (30-second, patch-wise, 129 epochs).
///
/// Sibling duration / epoch variants (`5s` / `10s` / `20s`, `30s-pw-73e`, …)
/// are distinct release identities that the converter publishes under their own
/// `NAME` following the `snac_24khz` / `snac_44khz` pattern, so this value is
/// **surfaced, not gated** — see [`Maest::name`].
pub const NAME: &str = "maest-30s-pw-129e";

/// Expected `vokra.model.category` value — `music-embedding`, shared with the
/// sibling music SSL encoders (`mert` / `muq`).
///
/// Deliberately **not** `audio-tagging` (the `yamnet` / `panns` / `ast` /
/// `clap` category): MAEST is trained on the Discogs music-tagger dataset, so
/// its outputs are genre / mood / instrument / era annotations over music
/// rather than the general AudioSet audio-event ontology. Consumed by the
/// model-card generator and the zoo-manifest tier gate.
pub const CATEGORY: &str = "music-embedding";

/// Upstream HuggingFace slug — stamped on `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "mtg-upf/discogs-maest-30s-pw-129e";

/// Default SPDX stamped by the converter — the **weight** tier.
///
/// Resolves to [`LicenseClass::NonCommercialShareAlike`] (T4 + ShareAlike
/// cascade). A caller with a different attestation may override at the
/// converter boundary (`--license <spdx>`), which is why this binder *surfaces*
/// rather than *asserts* the class.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

/// Metadata key holding [`CATEGORY`] (not part of `vokra_core::gguf::chunks`,
/// so mirrored here from the converter).
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key holding [`UPSTREAM_HF`] (not part of
/// `vokra_core::gguf::chunks`, so mirrored here from the converter).
pub const GGUF_KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// F32 parameter count of the upstream release, as reported by the HuggingFace
/// API (`parameters.F32: 86,858,128`) and recorded by the converter's module
/// docstring on 2026-08-13.
///
/// Used only to make the empty-manifest refusal concrete — it is never used to
/// validate a payload, because a sibling duration variant legitimately differs.
pub const UPSTREAM_PARAM_COUNT_F32: usize = 86_858_128;

/// `state_dict` name prefix under which the upstream tagging head sits.
///
/// **Transcribed, not assumed.** Upstream's HF `config` declares
/// `architectures: ["ASTForAudioClassification"]`, and that class holds
/// `self.classifier = ASTMLPHead(config)` — so a `state_dict` saved from it
/// carries the head under `classifier.`, alongside
/// `classifier.layernorm.{weight,bias}` and `classifier.dense.{weight,bias}`
/// (see [`PRIMARY_SOURCE_HF_AST_MODELING`]).
///
/// An artifact with no matching tensor is still **not** rejected: a
/// bare-`ASTModel` encoder export carries no head at all, which is a legitimate
/// artifact. It simply has no [`MaestEncoder::tag_mel`] surface.
pub const TAG_HEAD_PREFIX: &str = "classifier.";

// Primary-source anchors, cited inside the loud-partial error so a reader
// diagnosing the gap has fully specified places to walk.

/// Primary-source anchor: the upstream HuggingFace release.
pub const PRIMARY_SOURCE_UPSTREAM_HF: &str = "huggingface.co/mtg-upf/discogs-maest-30s-pw-129e";

/// Primary-source anchor: Alonso-Jiménez et al. 2023 (ISMIR) — the MAEST paper.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2309.16418";

/// Primary-source anchor: the HuggingFace `transformers` AST modelling file at
/// the tag the checkpoint's own config names.
///
/// The upstream `config.json` records `transformers_version: "4.34.0.dev0"`, so
/// `v4.34.0` is the matching tag. Every `state_dict` name this module walks —
/// and the pre-norm block ordering, the token concatenation order, the plane
/// orientation and the pooling rule — is transcribed from this file rather than
/// inferred from a naming convention.
pub const PRIMARY_SOURCE_HF_AST_MODELING: &str = "github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py";

// ---------------------------------------------------------------------------
// `vokra.maest.*` metadata keys — byte-identical mirrors of the converter's
// private `KEY_MAEST_*` constants in
// `crates/vokra-convert/src/models/maest.rs`. The spellings ARE the cross-crate
// contract, so they are pinned by a test rather than merely copied.
// ---------------------------------------------------------------------------

/// Transformer hidden width (`config.json` `hidden_size`).
pub const GGUF_KEY_HIDDEN_SIZE: &str = "vokra.maest.hidden_size";
/// Transformer block count (`config.json` `num_hidden_layers`).
pub const GGUF_KEY_NUM_HIDDEN_LAYERS: &str = "vokra.maest.num_hidden_layers";
/// Attention head count (`config.json` `num_attention_heads`).
pub const GGUF_KEY_NUM_ATTENTION_HEADS: &str = "vokra.maest.num_attention_heads";
/// FFN intermediate width (`config.json` `intermediate_size`).
pub const GGUF_KEY_INTERMEDIATE_SIZE: &str = "vokra.maest.intermediate_size";
/// Square ViT patch edge in mel bins × frames (`config.json` `patch_size`).
pub const GGUF_KEY_PATCH_SIZE: &str = "vokra.maest.patch_size";
/// Patch stride along the mel-bin axis (`config.json` `frequency_stride`).
pub const GGUF_KEY_FREQUENCY_STRIDE: &str = "vokra.maest.frequency_stride";
/// Patch stride along the frame axis (`config.json` `time_stride`).
pub const GGUF_KEY_TIME_STRIDE: &str = "vokra.maest.time_stride";
/// Log-mel band count (`config.json` `num_mel_bins`).
pub const GGUF_KEY_NUM_MEL_BINS: &str = "vokra.maest.num_mel_bins";
/// Frame count the position table is sized for (`config.json` `max_length`).
pub const GGUF_KEY_MAX_LENGTH: &str = "vokra.maest.max_length";
/// Discogs label-set size (`config.json` `id2label` cardinality).
pub const GGUF_KEY_NUM_LABELS: &str = "vokra.maest.num_labels";
/// Whether q/k/v projections carry a bias (`config.json` `qkv_bias`).
pub const GGUF_KEY_QKV_BIAS: &str = "vokra.maest.qkv_bias";
/// Encoder activation name (`config.json` `hidden_act`).
pub const GGUF_KEY_HIDDEN_ACT: &str = "vokra.maest.hidden_act";
/// LayerNorm epsilon as a `u32` scaled by 1e9 (`config.json` `layer_norm_eps`).
pub const GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9: &str = "vokra.maest.layer_norm_eps_scaled_1e9";
/// Hidden dropout scaled by 1e3 — inference-inert, stamped for audit.
pub const GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3: &str = "vokra.maest.hidden_dropout_scaled_1e3";
/// Attention dropout scaled by 1e3 — inference-inert, stamped for audit.
pub const GGUF_KEY_ATTENTION_DROPOUT_SCALED_1E3: &str = "vokra.maest.attention_dropout_scaled_1e3";
/// Patch-grid extent along the mel-bin axis.
pub const GGUF_KEY_FREQ_PATCHES: &str = "vokra.maest.freq_patches";
/// Patch-grid extent along the frame axis.
pub const GGUF_KEY_TIME_PATCHES: &str = "vokra.maest.time_patches";
/// Total patch-token count entering the encoder.
pub const GGUF_KEY_NUM_PATCHES: &str = "vokra.maest.num_patches";
/// Learned tokens prepended ahead of the patch tokens (AST: CLS + distillation).
pub const GGUF_KEY_NUM_PREFIX_TOKENS: &str = "vokra.maest.num_prefix_tokens";
/// Front-end sample rate in Hz.
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.maest.sample_rate";
/// STFT transform size.
pub const GGUF_KEY_N_FFT: &str = "vokra.maest.n_fft";
/// STFT hop in samples.
pub const GGUF_KEY_HOP_LENGTH: &str = "vokra.maest.hop_length";
/// STFT analysis-window length in samples.
pub const GGUF_KEY_WIN_LENGTH: &str = "vokra.maest.win_length";
/// STFT analysis-window type.
pub const GGUF_KEY_WINDOW: &str = "vokra.maest.window";
/// Mel filterbank frequency scale.
pub const GGUF_KEY_MEL_SCALE: &str = "vokra.maest.mel_scale";
/// Mel filterbank normalization.
pub const GGUF_KEY_MEL_NORM: &str = "vokra.maest.mel_norm";
/// Mel filterbank lower edge in Hz.
pub const GGUF_KEY_FMIN_HZ: &str = "vokra.maest.fmin_hz";
/// Mel filterbank upper edge in Hz.
pub const GGUF_KEY_FMAX_HZ: &str = "vokra.maest.fmax_hz";
/// Magnitude-compression mode.
pub const GGUF_KEY_LOG_COMPRESSION: &str = "vokra.maest.log_compression";
/// Multiplier inside the `logC` compression.
pub const GGUF_KEY_LOG_COMPRESSION_MUL: &str = "vokra.maest.log_compression_mul";
/// Whether the compressed spectrogram is mean/std normalized.
pub const GGUF_KEY_DO_NORMALIZE: &str = "vokra.maest.do_normalize";
/// Normalization mean, stamped `FLOAT64`.
pub const GGUF_KEY_NORM_MEAN: &str = "vokra.maest.norm_mean";
/// Normalization standard deviation, stamped `FLOAT64`.
pub const GGUF_KEY_NORM_STD: &str = "vokra.maest.norm_std";

/// The `hidden_act` value this binder can map onto a [`GeluKind`].
///
/// Upstream `ACT2FN["gelu"]` is `GELUActivation`, the **exact erf** formulation
/// (`x · 0.5 · (1 + erf(x / √2))`). The tanh approximation is registered under
/// *different* keys upstream (`gelu_new`, `gelu_pytorch_tanh`, `gelu_fast`,
/// `gelu_accurate`), so a value other than this one is a genuinely different
/// activation and is refused rather than silently folded onto the erf form —
/// the two differ by up to ~1e-3, which is shape-valid and numerically wrong.
pub const SUPPORTED_HIDDEN_ACT: &str = "gelu";

/// The `state_dict` prefix an `ASTForAudioClassification` export carries.
///
/// That class holds the backbone as `self.audio_spectrogram_transformer`, so
/// every encoder tensor is nested under this prefix.
pub const TENSOR_PREFIX_CLASSIFICATION: &str = "audio_spectrogram_transformer.";

/// The `state_dict` prefix a bare `ASTModel` export carries — none.
///
/// Both spellings occur in the wild depending on which class was saved, so the
/// prefix is **discovered** from the manifest on disk rather than assumed; see
/// [`MaestWeights::detect_tensor_prefix`].
pub const TENSOR_PREFIX_BARE: &str = "";

/// Prefix-relative name of the position table, used to probe which
/// `state_dict` prefix an artifact was saved under.
const PROBE_SUFFIX_POSITION_EMBEDDINGS: &str = "embeddings.position_embeddings";

// ---------------------------------------------------------------------------
// MaestConfig — the `vokra.maest.*` topology + front-end axis group
// ---------------------------------------------------------------------------

/// MAEST topology and log-mel front-end axes, as they ride the
/// `vokra.maest.*` chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every stamped key is
/// required and a missing one is a loud [`VokraError::ModelLoad`] naming it.
/// There is deliberately **no** primary-source constant fallback — the
/// converter transcribes each of these from the upstream `config.json` /
/// `preprocessor_config.json` and stamps them, so a partially stamped artifact
/// signals a mis-produced GGUF, and defaulting would let it through while
/// binding fabricated axes (FR-EX-08). Same posture as the sibling
/// `vokra.wavlm.*` reader.
#[derive(Debug, Clone, PartialEq)]
pub struct MaestConfig {
    /// Transformer hidden width `D`.
    pub hidden_size: u32,
    /// Transformer block count.
    pub num_hidden_layers: u32,
    /// Attention head count; must divide [`Self::hidden_size`].
    pub num_attention_heads: u32,
    /// FFN intermediate width.
    pub intermediate_size: u32,
    /// Square ViT patch edge, in mel bins × frames.
    pub patch_size: u32,
    /// Patch stride along the mel-bin axis. **Smaller than
    /// [`Self::patch_size`]** for MAEST — the patches overlap.
    pub frequency_stride: u32,
    /// Patch stride along the frame axis, likewise overlapping.
    pub time_stride: u32,
    /// Log-mel band count the front-end produces.
    pub num_mel_bins: u32,
    /// Frame count the position table was trained at.
    pub max_length: u32,
    /// Discogs label-set size.
    pub num_labels: u32,
    /// Whether the q/k/v projections carry a bias.
    pub qkv_bias: bool,
    /// Encoder activation name; see [`SUPPORTED_HIDDEN_ACT`].
    pub hidden_act: String,
    /// LayerNorm epsilon scaled by 1e9.
    pub layer_norm_eps_scaled_1e9: u32,
    /// Hidden dropout scaled by 1e3 (inference-inert).
    pub hidden_dropout_scaled_1e3: u32,
    /// Attention dropout scaled by 1e3 (inference-inert).
    pub attention_dropout_scaled_1e3: u32,
    /// Patch-grid extent along the mel-bin axis, at the trained length.
    pub freq_patches: u32,
    /// Patch-grid extent along the frame axis, at the trained length.
    pub time_patches: u32,
    /// Total patch tokens at the trained length.
    pub num_patches: u32,
    /// Learned tokens prepended ahead of the patch tokens.
    pub num_prefix_tokens: u32,
    /// Front-end sample rate in Hz.
    pub sample_rate: u32,
    /// STFT transform size.
    pub n_fft: u32,
    /// STFT hop in samples.
    pub hop_length: u32,
    /// STFT analysis-window length in samples.
    pub win_length: u32,
    /// STFT analysis-window type.
    pub window: String,
    /// Mel filterbank frequency scale.
    pub mel_scale: String,
    /// Mel filterbank normalization.
    pub mel_norm: String,
    /// Mel filterbank lower edge in Hz.
    pub fmin_hz: u32,
    /// Mel filterbank upper edge in Hz.
    pub fmax_hz: u32,
    /// Magnitude-compression mode.
    pub log_compression: String,
    /// Multiplier inside the `logC` compression.
    pub log_compression_mul: u32,
    /// Whether the compressed spectrogram is mean/std normalized.
    pub do_normalize: bool,
    /// Normalization mean, at the `FLOAT64` precision it was published with.
    pub norm_mean: f64,
    /// Normalization standard deviation, likewise `FLOAT64`.
    pub norm_std: f64,
}

impl MaestConfig {
    /// Reads every `vokra.maest.*` chunk from `gguf`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any stamped key is absent or carries
    ///   the wrong metadata type — the message names the key (FR-EX-08, no
    ///   primary-source constant fallback).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Ok(Self {
            hidden_size: req_u32(gguf, GGUF_KEY_HIDDEN_SIZE)?,
            num_hidden_layers: req_u32(gguf, GGUF_KEY_NUM_HIDDEN_LAYERS)?,
            num_attention_heads: req_u32(gguf, GGUF_KEY_NUM_ATTENTION_HEADS)?,
            intermediate_size: req_u32(gguf, GGUF_KEY_INTERMEDIATE_SIZE)?,
            patch_size: req_u32(gguf, GGUF_KEY_PATCH_SIZE)?,
            frequency_stride: req_u32(gguf, GGUF_KEY_FREQUENCY_STRIDE)?,
            time_stride: req_u32(gguf, GGUF_KEY_TIME_STRIDE)?,
            num_mel_bins: req_u32(gguf, GGUF_KEY_NUM_MEL_BINS)?,
            max_length: req_u32(gguf, GGUF_KEY_MAX_LENGTH)?,
            num_labels: req_u32(gguf, GGUF_KEY_NUM_LABELS)?,
            qkv_bias: req_bool(gguf, GGUF_KEY_QKV_BIAS)?,
            hidden_act: req_string(gguf, GGUF_KEY_HIDDEN_ACT)?,
            layer_norm_eps_scaled_1e9: req_u32(gguf, GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9)?,
            hidden_dropout_scaled_1e3: req_u32(gguf, GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3)?,
            attention_dropout_scaled_1e3: req_u32(gguf, GGUF_KEY_ATTENTION_DROPOUT_SCALED_1E3)?,
            freq_patches: req_u32(gguf, GGUF_KEY_FREQ_PATCHES)?,
            time_patches: req_u32(gguf, GGUF_KEY_TIME_PATCHES)?,
            num_patches: req_u32(gguf, GGUF_KEY_NUM_PATCHES)?,
            num_prefix_tokens: req_u32(gguf, GGUF_KEY_NUM_PREFIX_TOKENS)?,
            sample_rate: req_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            n_fft: req_u32(gguf, GGUF_KEY_N_FFT)?,
            hop_length: req_u32(gguf, GGUF_KEY_HOP_LENGTH)?,
            win_length: req_u32(gguf, GGUF_KEY_WIN_LENGTH)?,
            window: req_string(gguf, GGUF_KEY_WINDOW)?,
            mel_scale: req_string(gguf, GGUF_KEY_MEL_SCALE)?,
            mel_norm: req_string(gguf, GGUF_KEY_MEL_NORM)?,
            fmin_hz: req_u32(gguf, GGUF_KEY_FMIN_HZ)?,
            fmax_hz: req_u32(gguf, GGUF_KEY_FMAX_HZ)?,
            log_compression: req_string(gguf, GGUF_KEY_LOG_COMPRESSION)?,
            log_compression_mul: req_u32(gguf, GGUF_KEY_LOG_COMPRESSION_MUL)?,
            do_normalize: req_bool(gguf, GGUF_KEY_DO_NORMALIZE)?,
            norm_mean: req_f64(gguf, GGUF_KEY_NORM_MEAN)?,
            norm_std: req_f64(gguf, GGUF_KEY_NORM_STD)?,
        })
    }

    /// LayerNorm epsilon, un-scaling the stamped `× 1e9` integer encoding.
    #[inline]
    #[must_use]
    pub fn layer_norm_eps(&self) -> f32 {
        // Divide in f64, then narrow once — the stamped integer is exact, so
        // this is the only rounding step.
        (f64::from(self.layer_norm_eps_scaled_1e9) / 1.0e9) as f32
    }

    /// MLP hidden width as a multiple of the hidden width.
    ///
    /// [`ViTAttrs`] carries the *ratio* rather than the absolute width, so this
    /// divides the two stamped axes. [`Self::vit_attrs_with_pos_embed`]
    /// verifies that the ratio rounds back to the stamped
    /// [`Self::intermediate_size`] exactly, so the conversion cannot silently
    /// lose a unit.
    #[inline]
    #[must_use]
    pub fn mlp_ratio(&self) -> f32 {
        self.intermediate_size as f32 / self.hidden_size as f32
    }

    /// Encoder sequence length at the trained input length:
    /// `num_prefix_tokens + num_patches`.
    #[inline]
    #[must_use]
    pub fn encoder_sequence_len(&self) -> usize {
        self.num_prefix_tokens as usize + self.num_patches as usize
    }

    /// Maps the stamped [`Self::hidden_act`] onto a [`GeluKind`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] for any value other than
    ///   [`SUPPORTED_HIDDEN_ACT`] — the tanh-family activations carry distinct
    ///   upstream names, so folding them onto the erf form would be silently
    ///   wrong rather than loud.
    pub fn gelu_kind(&self) -> Result<GeluKind> {
        if self.hidden_act == SUPPORTED_HIDDEN_ACT {
            return Ok(GeluKind::Erf);
        }
        Err(VokraError::ModelLoad(format!(
            "maest: `{GGUF_KEY_HIDDEN_ACT}` is `{act}`, but this binder maps only \
             `{SUPPORTED_HIDDEN_ACT}` (upstream `ACT2FN[\"gelu\"]` = `GELUActivation`, the exact \
             erf formulation `x · 0.5 · (1 + erf(x / √2))`). The tanh approximation is registered \
             upstream under DIFFERENT keys (`gelu_new`, `gelu_pytorch_tanh`, `gelu_fast`, \
             `gelu_accurate`), so `{act}` is a genuinely different activation — the two differ by \
             up to ~1e-3, which stays shape-valid while being numerically wrong. Refusing to fold \
             it onto the erf form (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF_AST_MODELING}",
            act = self.hidden_act,
        )))
    }

    /// The positional-embedding policy that resizes the **stamped** table grid
    /// to whatever grid the runtime plane produces.
    ///
    /// Use this when encoding a clip whose frame count differs from the trained
    /// [`Self::max_length`]. Note the resize is **bilinear**, whereas ViT-audio
    /// implementations generally resize positional tables bicubically — see
    /// [`PosEmbedPolicy::InterpolateGridBilinear`]. For numerical parity against
    /// upstream, resize offline and use [`PosEmbedPolicy::RequireExact`].
    #[inline]
    #[must_use]
    pub fn stamped_grid_pos_embed_policy(&self) -> PosEmbedPolicy {
        PosEmbedPolicy::InterpolateGridBilinear {
            table_grid_h: self.freq_patches as usize,
            table_grid_w: self.time_patches as usize,
        }
    }

    /// Maps the stamped axes onto [`ViTAttrs`] under
    /// [`PosEmbedPolicy::RequireExact`].
    ///
    /// # Errors
    ///
    /// See [`Self::vit_attrs_with_pos_embed`].
    pub fn vit_attrs(&self) -> Result<ViTAttrs> {
        self.vit_attrs_with_pos_embed(PosEmbedPolicy::RequireExact)
    }

    /// Maps the stamped axes onto [`ViTAttrs`] under an explicit
    /// positional-embedding policy.
    ///
    /// Where each [`ViTAttrs`] field comes from — every one is a stamped value,
    /// none is a "typical AST-base" default:
    ///
    /// - `embed_dim` ← [`Self::hidden_size`].
    /// - `depth` ← [`Self::num_hidden_layers`].
    /// - `n_heads` ← [`Self::num_attention_heads`].
    /// - `mlp_ratio` ← [`Self::intermediate_size`] ÷ [`Self::hidden_size`],
    ///   checked to round back to `intermediate_size` exactly.
    /// - `patch_h` / `patch_w` ← [`Self::patch_size`], which upstream passes to
    ///   `Conv2d` as the square `kernel_size=(patch_size, patch_size)`.
    /// - `stride_h` ← [`Self::frequency_stride`] and `stride_w` ←
    ///   [`Self::time_stride`], matching upstream's
    ///   `stride=(frequency_stride, time_stride)`. Because upstream transposes
    ///   the plane to `[num_mel_bins, max_length]` before the convolution, the
    ///   *first* stride walks the mel-bin axis — which is exactly `vokra-ops`'
    ///   `stride_h`. Both are 10 against a kernel of 16, so the patches overlap.
    /// - `n_prepended_tokens` ← [`Self::num_prefix_tokens`].
    /// - `layer_norm_eps` ← [`Self::layer_norm_eps`].
    /// - `gelu` ← [`Self::gelu_kind`].
    /// - `pos_embed_policy` ← the caller's `policy`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when [`Self::gelu_kind`] rejects the stamped
    ///   activation, or when the `mlp_ratio` division does not round back to
    ///   the stamped `intermediate_size`.
    /// - [`VokraError::InvalidArgument`] from [`ViTAttrs::validate`] when an
    ///   axis is zero or `hidden_size` is not divisible by
    ///   `num_attention_heads`.
    pub fn vit_attrs_with_pos_embed(&self, policy: PosEmbedPolicy) -> Result<ViTAttrs> {
        let attrs = ViTAttrs {
            embed_dim: self.hidden_size as usize,
            depth: self.num_hidden_layers as usize,
            n_heads: self.num_attention_heads as usize,
            mlp_ratio: self.mlp_ratio(),
            patch_h: self.patch_size as usize,
            patch_w: self.patch_size as usize,
            stride_h: self.frequency_stride as usize,
            stride_w: self.time_stride as usize,
            n_prepended_tokens: self.num_prefix_tokens as usize,
            layer_norm_eps: self.layer_norm_eps(),
            gelu: self.gelu_kind()?,
            pos_embed_policy: policy,
        };
        attrs.validate()?;

        // The ratio is a lossy re-encoding of two integers, so verify it lands
        // back on the stamped width instead of trusting the division.
        let resolved = attrs.mlp_dim();
        if resolved != self.intermediate_size as usize {
            return Err(VokraError::ModelLoad(format!(
                "maest: `{GGUF_KEY_INTERMEDIATE_SIZE}` is {stamped} but the `ViTAttrs` \
                 mlp_ratio ({ratio}) derived from it and `{GGUF_KEY_HIDDEN_SIZE}` ({hidden}) \
                 rounds the MLP hidden width to {resolved}. Refusing to bind an FFN of the \
                 wrong width (FR-EX-08).",
                stamped = self.intermediate_size,
                ratio = attrs.mlp_ratio,
                hidden = self.hidden_size,
            )));
        }
        Ok(attrs)
    }
}

/// Reads a required `u32`-ish metadata value, naming the key on failure.
fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| missing_axis(key, "unsigned integer"))
}

/// Reads a required boolean metadata value, naming the key on failure.
fn req_bool(gguf: &GgufFile, key: &str) -> Result<bool> {
    gguf.get(key)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| missing_axis(key, "boolean"))
}

/// Reads a required string metadata value, naming the key on failure.
fn req_string(gguf: &GgufFile, key: &str) -> Result<String> {
    gguf.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| missing_axis(key, "string"))
}

/// Reads a required float metadata value, naming the key on failure.
fn req_f64(gguf: &GgufFile, key: &str) -> Result<f64> {
    gguf.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| missing_axis(key, "float"))
}

/// The one loud message shape for an absent or wrongly typed `vokra.maest.*`
/// axis.
fn missing_axis(key: &str, want: &str) -> VokraError {
    VokraError::ModelLoad(format!(
        "maest: GGUF is missing required {want} chunk `{key}` — the upstream `{UPSTREAM_HF}` \
         release carries a first-class `config.json` + `preprocessor_config.json`, and the \
         converter transcribes every axis from them and stamps the whole `vokra.maest.*` group, \
         so a proper conversion always carries this key. This binder refuses to fall back to a \
         primary-source constant, because a silent default would let a mismatched artifact bind \
         against a fabricated axis (FR-EX-08). Re-run `vokra-cli convert --model \
         maest-30s-pw-129e` against the upstream safetensors release. Primary source: \
         {PRIMARY_SOURCE_UPSTREAM_HF}"
    ))
}

// ---------------------------------------------------------------------------
// MaestWeights — the tensor manifest, with loud lookups
// ---------------------------------------------------------------------------

/// Weight tensors bound from a MAEST GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification step.
/// A GGUF that carries zero tensors is rejected with [`VokraError::ModelLoad`]
/// (FR-EX-08 — an ~87M-parameter AST never converts to an empty manifest, so
/// zero tensors always signals a mis-produced artifact, and binding it would
/// silently run an all-zero forward).
///
/// Under the current landing this struct stores the tensor names and their
/// GGUF-side dims. The payload is deliberately not dequantised: the forward is
/// loud-partial (see [`Maest::encode`]), and the follow-up wave sizes its
/// dequant per its kernel needs. [`require_tensor`](Self::require_tensor) /
/// [`require_tensor_dims`](Self::require_tensor_dims) are already in place so
/// that wave walks a manifest that fails loudly rather than substituting zeros.
#[derive(Debug, Clone)]
pub struct MaestWeights {
    /// Tensors discovered on disk, in file order, as
    /// `(upstream state_dict name, GGUF-side dims)`.
    tensors: Vec<(String, Vec<usize>)>,
}

impl MaestWeights {
    /// Scans `gguf` for the MAEST `state_dict` tensors.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   (FR-EX-08 — refusing to bind an all-zero forward).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tensors: Vec<(String, Vec<usize>)> = gguf
            .tensors()
            .iter()
            .map(|info| {
                let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                (info.name.clone(), dims)
            })
            .collect();

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "maest: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate MAEST checkpoint is an AST-backbone Transformer of \
                 roughly {UPSTREAM_PARAM_COUNT_F32} F32 parameters (arch={ARCH}, name={NAME}) \
                 and always converts to hundreds of Linear / LayerNorm tensors, so an empty \
                 manifest always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model maest-30s-pw-129e` against the upstream \
                 `{UPSTREAM_HF}` safetensors release."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The upstream `state_dict` names discovered on disk, in file order.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// GGUF-side dims of `name`, or `None` when the tensor is absent.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// How many tensors have a name starting with `prefix`.
    ///
    /// A plain string count over what is actually on disk — it asserts nothing
    /// about the upstream naming convention.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .count()
    }

    /// The tensors on disk whose name starts with [`TAG_HEAD_PREFIX`], as
    /// `(name, dims)` pairs in file order.
    ///
    /// Pure disk reporting with no interpretation: an empty result means either
    /// the artifact is a bare-encoder export or the upstream head sits under a
    /// prefix this repository has not transcribed (see [`TAG_HEAD_PREFIX`]).
    /// Neither case is an error.
    #[must_use]
    pub fn tag_head_tensors(&self) -> Vec<(&str, &[usize])> {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(TAG_HEAD_PREFIX))
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect()
    }

    /// The label-set size **read off the artifact**, or `None`.
    ///
    /// Returns `Some(dims[0])` when exactly one tensor under
    /// [`TAG_HEAD_PREFIX`] is 2-D — a PyTorch `nn.Linear` weight is
    /// `[out_features, in_features]`, and the converter passes the safetensors
    /// shape through verbatim, so that leading dimension *is* the number of
    /// Discogs labels in whatever checkpoint the caller converted.
    ///
    /// Returns `None` in every other case (no head on disk, or an ambiguous
    /// shape layout). It never falls back to a constant — not even to the
    /// stamped [`MaestConfig::num_labels`], deliberately: the stamp comes from
    /// `config.json` and this value comes from the payload, so keeping them
    /// independent is what lets the head binding cross-check them. Collapsing
    /// one onto the other would turn two witnesses into one.
    #[must_use]
    pub fn label_count_from_disk(&self) -> Option<usize> {
        let mut two_d = self
            .tag_head_tensors()
            .into_iter()
            .filter(|(_, dims)| dims.len() == 2);
        let first = two_d.next()?;
        if two_d.next().is_some() {
            // More than one 2-D tensor under the head prefix: which one is the
            // label projection is ambiguous, so report nothing rather than pick.
            return None;
        }
        first.1.first().copied()
    }

    /// Dims of a **required** tensor, failing loudly when it is absent.
    ///
    /// The error names the missing tensor, the manifest size, and up to five
    /// nearby names on disk so a caller diagnosing a prefix mismatch has
    /// something concrete to compare against.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `name` is not present (FR-EX-08 —
    ///   never substitute a zero tensor).
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
            "maest: required tensor `{name}` is absent from the GGUF ({count} tensors \
             present; nearest names on disk: {near:?}). The converter passes upstream \
             `state_dict` names through verbatim, so a mismatch means either the checkpoint \
             was flattened with a different prefix policy (upstream wraps the backbone as \
             `ASTForAudioClassification`, whose body sits under an \
             `audio_spectrogram_transformer.` prefix that a re-export may strip) or the \
             caller is walking a manifest transcribed from a different MAEST release variant \
             (the 5s / 10s / 20s durations and the `30s-pw-73e` checkpoint point are \
             published under their own names). Refusing to substitute a zero tensor \
             (FR-EX-08). Primary source: {PRIMARY_SOURCE_UPSTREAM_HF}",
            count = self.tensors.len(),
        )))
    }

    /// Asserts that a required tensor is present **and** has exactly `expected`
    /// dims.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the tensor is absent (see
    ///   [`Self::require_tensor`]) or when its dims differ — the message names
    ///   **both** the expected and the actual dims (FR-EX-08 — never reshape or
    ///   truncate silently).
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<()> {
        let actual = self.require_tensor(name)?;
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "maest: tensor `{name}` has dims {actual:?} but the caller expects \
                 {expected:?} — refusing to reshape or truncate silently (FR-EX-08). The \
                 expected shape is derived from the stamped `vokra.maest.*` axis group, so a \
                 disagreement here means the payload and the stamped topology describe different \
                 checkpoints (a different duration variant, a different epoch checkpoint, or a \
                 re-export with a different label-set size). Primary sources: \
                 {PRIMARY_SOURCE_UPSTREAM_HF}, {PRIMARY_SOURCE_HF_AST_MODELING}"
            )));
        }
        Ok(())
    }

    /// Discovers which `state_dict` prefix this artifact was saved under.
    ///
    /// Returns [`TENSOR_PREFIX_CLASSIFICATION`] for an
    /// `ASTForAudioClassification` export (the class holds the backbone as
    /// `self.audio_spectrogram_transformer`, so every encoder tensor is nested
    /// under it) or [`TENSOR_PREFIX_BARE`] for a bare `ASTModel` export. Both
    /// occur in the wild, so the prefix is **probed against the manifest on
    /// disk** rather than assumed from the upstream `architectures` string —
    /// a re-export can legitimately strip it.
    ///
    /// The probe is the position table, which every AST export carries exactly
    /// once.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when neither spelling is present — the
    ///   message names **both** candidates it looked for (FR-EX-08 — never pick
    ///   a prefix that is not actually on disk).
    pub fn detect_tensor_prefix(&self) -> Result<&'static str> {
        for prefix in [TENSOR_PREFIX_CLASSIFICATION, TENSOR_PREFIX_BARE] {
            let probe = format!("{prefix}{PROBE_SUFFIX_POSITION_EMBEDDINGS}");
            if self.tensor_dims(&probe).is_some() {
                return Ok(prefix);
            }
        }
        let sample: Vec<&str> = self
            .tensors
            .iter()
            .map(|(n, _)| n.as_str())
            .take(5)
            .collect();
        Err(VokraError::ModelLoad(format!(
            "maest: could not find the position table under either known `state_dict` prefix — \
             looked for `{TENSOR_PREFIX_CLASSIFICATION}{PROBE_SUFFIX_POSITION_EMBEDDINGS}` (an \
             `ASTForAudioClassification` export, which nests the backbone under \
             `self.audio_spectrogram_transformer`) and \
             `{PROBE_SUFFIX_POSITION_EMBEDDINGS}` (a bare `ASTModel` export). The GGUF holds \
             {count} tensor(s); first names on disk: {sample:?}. Refusing to guess a third \
             prefix (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF_AST_MODELING}",
            count = self.tensors.len(),
        )))
    }
}

// ---------------------------------------------------------------------------
// Maest — the runtime binder handle
// ---------------------------------------------------------------------------

/// MAEST (Music Audio Efficient Spectrogram Transformer) self-supervised music
/// encoder with a Discogs tagging head.
///
/// Bind with [`from_gguf`](Self::from_gguf) — or the compliance-gated
/// [`from_gguf_with_policy`](Self::from_gguf_with_policy) /
/// [`from_path`](Self::from_path) / [`from_path_with_policy`](Self::from_path_with_policy).
///
/// Binding is cheap: it reads the metadata, the strict [`MaestConfig`] axis
/// group and the tensor **manifest**, but decodes no payload. To run the
/// encoder, call [`encoder`](Self::encoder), which binds the weights into a
/// [`MaestEncoder`] and gives you [`MaestEncoder::encode_mel`] /
/// [`MaestEncoder::embed_mel`] / [`MaestEncoder::tag_mel`].
///
/// The PCM-in surfaces [`encode`](Self::encode) / [`embed`](Self::embed) /
/// [`tag`](Self::tag) remain **loud-partial**: the log-mel front-end's framing
/// convention is the one axis the converter deliberately does not stamp. See
/// the module docstring.
#[derive(Debug, Clone)]
pub struct Maest {
    name: Option<String>,
    category: Option<String>,
    upstream_hf: Option<String>,
    model_id: Option<String>,
    source: Option<String>,
    config: MaestConfig,
    weights: MaestWeights,
    weight_license: LicenseClass,
    attribution: Option<String>,
}

impl Maest {
    /// Binds a MAEST GGUF: verifies the arch strictly, binds the tensor
    /// manifest, and surfaces the converter's metadata + licence stamps.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the missing
    /// or wrong key, so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// `vokra.model.name` is deliberately **surfaced, not gated**: the duration
    /// and epoch sibling variants share this arch under different names, so a
    /// hard name check would make a legitimate future artifact unloadable. See
    /// [`Self::name`].
    ///
    /// This entry point performs **no licence gate** — use
    /// [`Self::from_gguf_with_policy`] for that. MAEST is
    /// [`LicenseClass::NonCommercialShareAlike`], so the gated route refuses
    /// under [`CompliancePolicy::strict`] by design.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is not `"maest"` —
    ///   the message names both the found and the expected tag and enumerates
    ///   the SSL audio/music-embedding neighbourhood.
    /// - [`VokraError::ModelLoad`] when any `vokra.maest.*` axis is absent
    ///   ([`MaestConfig::from_gguf`] is strict — no constant fallback).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`MaestWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch first, so a mis-routed artifact reports the arch mismatch
        //    (the actionable fact) instead of a downstream missing-tensor trail.
        verify_arch(file)?;

        // 2. Metadata surfacing. Soft: a converter-produced artifact always
        //    carries these, but they are diagnostics, not load gates.
        let read_str = |key: &str| -> Option<String> {
            file.get(key).and_then(|v| v.as_str()).map(str::to_owned)
        };
        let name = read_str(chunks::KEY_MODEL_NAME);
        let category = read_str(GGUF_KEY_MODEL_CATEGORY);
        let upstream_hf = read_str(GGUF_KEY_PROVENANCE_UPSTREAM_HF);
        let model_id = read_str(chunks::KEY_PROVENANCE_MODEL_ID);
        let source = read_str(chunks::KEY_PROVENANCE_SOURCE);

        // 3. Strict topology + front-end axes from the `vokra.maest.*` group.
        //    Ahead of the manifest so a pre-axis-group artifact reports the
        //    missing axis (the actionable fact) rather than a tensor trail.
        let config = MaestConfig::from_gguf(file)?;

        // 4. Tensor manifest with the non-emptiness gate.
        let weights = MaestWeights::from_gguf(file)?;

        // 5. Provenance surfacing. The converter stamps
        //    `NonCommercialShareAlike` (cc-by-nc-sa-4.0); an artifact missing
        //    the stamp reads back as `Unknown` — fail-closed at the M2-13 gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let attribution = read_str(chunks::KEY_PROVENANCE_ATTRIBUTION);

        Ok(Self {
            name,
            category,
            upstream_hf,
            model_id,
            source,
            config,
            weights,
            weight_license,
            attribution,
        })
    }

    /// Loads a MAEST GGUF from raw bytes under `policy` (the M2-13
    /// weight-licence gate).
    ///
    /// MAEST ships **CC-BY-NC-SA 4.0** → [`LicenseClass::NonCommercialShareAlike`],
    /// whose [`LicenseClass::requires_research_flag`] is `true`, so a correctly
    /// stamped artifact is **refused** under [`CompliancePolicy::strict`] and
    /// loads only with an explicit research opt-in. An artifact with no
    /// provenance stamp resolves to [`LicenseClass::Unknown`] and is refused for
    /// the same reason — fail-closed, never a silent substitution.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on GGUF parse failure, or on a wrong /
    ///   missing `vokra.model.arch`.
    /// - `VokraError::ResearchLicenseRequired` from the compliance gate when the
    ///   weight class is gated and `policy` grants no research opt-in — the
    ///   expected outcome for MAEST under a strict policy.
    /// - See [`Self::from_gguf`] for the remaining bind errors.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("maest GGUF: {e}")))?;
        // Arch before the compliance gate so a mis-routed artifact reports the
        // arch mismatch rather than a licence verdict about a model the caller
        // never meant to load.
        verify_arch(&file)?;
        check_weight_license(&file, policy)?;
        Self::from_gguf(&file)
    }

    /// Loads a MAEST GGUF from a path under [`CompliancePolicy::strict`].
    ///
    /// Because MAEST is non-commercial, this route **refuses** a correctly
    /// stamped artifact — that is the fail-closed default working as intended.
    /// Callers with a research/evaluation basis should use
    /// [`Self::from_path_with_policy`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Loads a MAEST GGUF from a path under an explicit `policy`.
    ///
    /// The route a research/evaluation consumer takes:
    /// `CompliancePolicy::strict().with_research_license(true)` unlocks the
    /// non-commercial gate and emits the mandatory research-only warning. The
    /// ShareAlike and attribution obligations are not waived by that opt-in —
    /// see [`Self::weight_license`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path_with_policy(
        path: impl AsRef<std::path::Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, policy)
    }

    /// The stamped `vokra.model.name`, if present.
    ///
    /// [`NAME`] (`"maest-30s-pw-129e"`) for the release this module tracks; the
    /// duration / epoch sibling variants share [`ARCH`] under different names,
    /// which is why this is surfaced rather than gated.
    #[inline]
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The stamped `vokra.model.category`, if present — [`CATEGORY`]
    /// (`"music-embedding"`) for a converter-produced artifact.
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_hf`, if present — [`UPSTREAM_HF`]
    /// for a converter-produced artifact.
    #[inline]
    #[must_use]
    pub fn upstream_hf(&self) -> Option<&str> {
        self.upstream_hf.as_deref()
    }

    /// The stamped `vokra.provenance.model_id`, if present — the converter
    /// passes [`NAME`] into `stamp_provenance`, and the compliance gate uses
    /// this same key when naming a refused model.
    #[inline]
    #[must_use]
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// The stamped `vokra.provenance.source`, if present — the converter's
    /// free-text upstream description (release slug, objective, scale, paper).
    #[inline]
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &MaestWeights {
        &self.weights
    }

    /// The strict topology + front-end axes read from the `vokra.maest.*`
    /// chunk group.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &MaestConfig {
        &self.config
    }

    /// Binds the encoder weights, producing a runnable [`MaestEncoder`].
    ///
    /// This is the expensive step [`Self::from_gguf`] deliberately skips: it
    /// decodes every encoder tensor through `GgufFile::tensor_f32` (so a
    /// K-quantised artifact dequantises on the way in) and shape-checks each
    /// one against the stamped axes. `file` must be the same GGUF this handle
    /// was bound from.
    ///
    /// `pos_embed_policy` decides what happens when the plane you later encode
    /// produces a different patch grid than the table on disk was trained at.
    /// [`PosEmbedPolicy::RequireExact`] is the parity-safe choice;
    /// [`MaestConfig::stamped_grid_pos_embed_policy`] builds the interpolating
    /// alternative from the stamped grid.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the `state_dict` prefix cannot be
    ///   discovered ([`MaestWeights::detect_tensor_prefix`]), when a required
    ///   tensor is absent or has unexpected dims, when a payload fails to
    ///   decode, or when the stamped `num_prefix_tokens` is not the 2 that an
    ///   AST checkpoint's CLS + distillation pair implies.
    /// - [`VokraError::InvalidArgument`] from `ViTEncoder::new` when a weight
    ///   buffer is the wrong length or holds a non-finite value.
    pub fn encoder(
        &self,
        file: &GgufFile,
        pos_embed_policy: PosEmbedPolicy,
    ) -> Result<MaestEncoder> {
        MaestEncoder::bind(file, &self.config, &self.weights, pos_embed_policy)
    }

    /// Number of tensors discovered on disk.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Whether the artifact carries any tensor under [`TAG_HEAD_PREFIX`].
    ///
    /// **Diagnostic only** — it gates nothing. `false` is not an error: a
    /// bare-encoder export is legitimate, and so is a head under a prefix this
    /// repository has not transcribed.
    #[inline]
    #[must_use]
    pub fn has_tag_head(&self) -> bool {
        self.weights.count_with_prefix(TAG_HEAD_PREFIX) > 0
    }

    /// The Discogs label-set size **read off this artifact**, or `None`.
    ///
    /// Delegates to [`MaestWeights::label_count_from_disk`]: the value comes
    /// from the head projection's leading dimension on disk, never from a
    /// taxonomy constant (this module contains none — see the module docstring
    /// "Label taxonomy" section).
    #[inline]
    #[must_use]
    pub fn label_count(&self) -> Option<usize> {
        self.weights.label_count_from_disk()
    }

    /// The weight-licence class surfaced from
    /// `vokra.provenance.weight_license`.
    ///
    /// [`LicenseClass::NonCommercialShareAlike`] for a correctly stamped MAEST
    /// artifact (cc-by-nc-sa-4.0), carrying three cascading obligations —
    /// non-commercial use only, ShareAlike on any downstream distribution, and
    /// attribution. [`LicenseClass::Unknown`] when the stamp is absent
    /// (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The FR-MD-09 attribution text stamped under
    /// `vokra.provenance.attribution`, if any.
    ///
    /// CC-BY-NC-SA 4.0 carries a BY cascade, so a consumer shipping
    /// MAEST-derived output must render credit. The converter's
    /// `stamp_provenance` call does **not** currently write this key, so a
    /// converter-produced artifact reads `None` — a recorded gap, surfaced here
    /// rather than papered over with invented wording.
    #[inline]
    #[must_use]
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }

    /// Encodes a mono `f32` PCM slice into the encoder's token hidden states.
    ///
    /// # Loud-partial — the log-mel **framing convention**, and only that
    ///
    /// Returns [`VokraError::UnsupportedOp`]. The encoder itself is **no longer
    /// a blocker**: [`Self::encoder`] binds it and
    /// [`MaestEncoder::encode_mel`] runs it. What is missing is the PCM → mel
    /// step, and specifically one axis of it — the STFT framing / centering
    /// convention. Every other front-end axis *is* stamped and readable off
    /// [`Self::config`] (sample rate, `n_fft`, hop, window length and type, mel
    /// scale and normalization, `fmin` / `fmax`, the `logC` compression and its
    /// multiplier, and the normalization mean / std). The converter
    /// deliberately does not stamp `center` or `pad_mode` because no primary
    /// source it reached states them, and choosing wrongly shifts every frame
    /// by half a window — shape-valid, numerically wrong, and silent.
    ///
    /// Callers holding their own log-mel plane should use
    /// [`MaestEncoder::encode_mel`], which is real today.
    ///
    /// **No fabricated hidden states are ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   unstamped front-end framing convention.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind explicitly so a future accidental removal of the parameter is
        // not masked by an unused-variable warning (mirror of the atst / m2d /
        // emotion2vec loud-partial signature discipline).
        let _ = pcm;
        Err(front_end_loud_partial(
            "encode",
            "token hidden states",
            &self.config,
        ))
    }

    /// Encodes a mono `f32` PCM slice into the **pooled clip embedding**.
    ///
    /// # Loud-partial — same single blocker as [`Self::encode`]
    ///
    /// Returns [`VokraError::UnsupportedOp`] for the unstamped log-mel framing
    /// convention. Unlike the previous landing, the **width** of the vector is
    /// no longer unknown: it is [`MaestConfig::hidden_size`]. Callers holding
    /// their own mel plane should use [`MaestEncoder::embed_mel`], which is
    /// real today.
    ///
    /// **No fabricated embedding is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   unstamped front-end framing convention.
    pub fn embed(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(front_end_loud_partial(
            "embed",
            "pooled clip embedding",
            &self.config,
        ))
    }

    /// Runs the Discogs tagging head over a mono `f32` PCM slice, returning one
    /// logit per label.
    ///
    /// # Loud-partial — same single blocker as [`Self::encode`]
    ///
    /// Returns [`VokraError::UnsupportedOp`] for the unstamped log-mel framing
    /// convention. Callers holding their own mel plane should use
    /// [`MaestEncoder::tag_mel`], which is real today.
    ///
    /// Note what is **not** a blocker here: the head produces *logits*, and
    /// this method's return type is logits, so the absent label taxonomy does
    /// not stand in its way. The taxonomy gap is real but narrower than it once
    /// read — the artifact carries no label *names*, so a caller cannot map
    /// those logits onto human-readable Discogs genre / mood / instrument / era
    /// strings. The label **count** is available twice over and cross-checked:
    /// stamped as [`MaestConfig::num_labels`] and read off the head projection
    /// on disk via [`Self::label_count`].
    ///
    /// **No fabricated logits are ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   unstamped front-end framing convention.
    pub fn tag(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(front_end_loud_partial(
            "tag",
            "Discogs tag logits",
            &self.config,
        ))
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Strict `vokra.model.arch` verification.
///
/// Refuses a foreign GGUF loudly, naming **both** the found and the expected tag
/// and enumerating the SSL audio/music-embedding neighbourhood plus the wav2vec2
/// lineage, so a reader who handed the wrong artifact over knows immediately
/// which loader they wanted (FR-EX-08 — never a silent misroute).
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "maest: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF produced by \
             `vokra-cli convert --model maest-30s-pw-129e`?). MAEST is an AST-backbone \
             self-supervised MUSIC tagger pretrained on the MTG Discogs4All dataset. The \
             confusable neighbours: `ast` shares the very same Audio Spectrogram Transformer \
             backbone but is SUPERVISED, fine-tuned on AudioSet, general-audio, and carries a \
             different label taxonomy — backbone identity is not topology identity; `atst` \
             (BYOL-style teacher-student patchout), `beats` (iterative acoustic tokenizer + \
             masked acoustic modelling), `eat` (utterance-level MAE with inverse block \
             masking), `dasheng` (universal MAE), `m2d` (masked modelling duo, dual online + \
             target branch), `mert` (HuBERT-derived masked prediction, music), `muq` (Mel-RVQ \
             + BEATs teacher, music) differ in the pre-training objective that shapes the \
             topology; `yamnet` and `panns` are supervised audio-tagging CNNs, not SSL at all; \
             `clap` bolts a text tower on for contrastive pretraining; and the wav2vec2 \
             lineage (`hubert`, `wav2vec2_ctc`, `wavlm_sv`, `emotion2vec`) sits on a \
             raw-waveform 1-D conv stem rather than a log-mel patch grid. Binding any of them \
             here would walk a foreign topology over a MAEST payload (FR-EX-08 — no silent \
             partial load). Primary source: {PRIMARY_SOURCE_UPSTREAM_HF}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "maest: GGUF is missing `vokra.model.arch` — this is not a Vokra-native maest \
             GGUF (was it produced by `vokra-cli convert --model maest-30s-pw-129e`?). \
             Refusing to guess the arch from the tensor manifest (FR-EX-08). Primary source: \
             {PRIMARY_SOURCE_UPSTREAM_HF}"
        ))),
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by the
/// PCM-in surfaces [`Maest::encode`] / [`Maest::embed`] / [`Maest::tag`].
///
/// `surface` is the method name and `output` is what that method would have
/// returned. The message states the **one** remaining blocker — the unstamped
/// log-mel framing convention — and, so a reader can see the boundary rather
/// than infer it, enumerates the front-end axes that ARE stamped and points at
/// the real mel-plane entry point.
///
/// # Why this message shrank
///
/// It once named four blockers. Three are resolved and were removed rather than
/// left standing, because a stale claim in an error message misleads whoever
/// reads it next: the converter now stamps the full `vokra.maest.*` axis group;
/// `vokra_ops::vit` now supplies the 2-D patch embedding + pre-norm Transformer
/// encoder; and the `state_dict` manifest is now transcribed from
/// [`PRIMARY_SOURCE_HF_AST_MODELING`]. The fourth (the absent label taxonomy)
/// was never a blocker on *logits* — see [`Maest::tag`].
fn front_end_loud_partial(surface: &str, output: &str, cfg: &MaestConfig) -> VokraError {
    // Bound to a `let` rather than nested inside the outer `format!` args
    // (clippy: no `format!` inside another `format!`'s arguments).
    let stamped_axes = format!(
        "sample_rate={sr}, n_fft={n_fft}, hop_length={hop}, win_length={win}, window={window}, \
         num_mel_bins={mels}, mel_scale={scale}, mel_norm={norm}, fmin_hz={fmin}, \
         fmax_hz={fmax}, log_compression={comp} (multiplier {mul}), do_normalize={do_norm}, \
         norm_mean={mean}, norm_std={std}",
        sr = cfg.sample_rate,
        n_fft = cfg.n_fft,
        hop = cfg.hop_length,
        win = cfg.win_length,
        window = cfg.window,
        mels = cfg.num_mel_bins,
        scale = cfg.mel_scale,
        norm = cfg.mel_norm,
        fmin = cfg.fmin_hz,
        fmax = cfg.fmax_hz,
        comp = cfg.log_compression,
        mul = cfg.log_compression_mul,
        do_norm = cfg.do_normalize,
        mean = cfg.norm_mean,
        std = cfg.norm_std,
    );

    VokraError::UnsupportedOp(format!(
        "maest {surface} (loud-partial): the PCM -> log-mel front end is incomplete, so no \
         {output} can be produced FROM PCM. Exactly ONE axis is missing: the STFT FRAMING / \
         CENTERING CONVENTION. The converter deliberately stamps no `center` and no `pad_mode` \
         key, and writes no `vokra.frontend.*` bit-exact group, because no primary source it \
         reached states them; picking centred vs non-centred shifts every frame by half a \
         window, which stays shape-valid while being numerically wrong, so it is refused rather \
         than guessed. Every OTHER front-end axis IS stamped and readable via \
         `Maest::config`: {stamped_axes}. \
         THE ENCODER ITSELF IS NOT A BLOCKER — bind it with `Maest::encoder` and run \
         `MaestEncoder::encode_mel` / `MaestEncoder::embed_mel` / `MaestEncoder::tag_mel` over a \
         log-mel plane you supply as [num_mel_bins, n_frames] row-major. That path is real: the \
         `vokra.maest.*` axis group is stamped, `vokra_ops::vit` supplies the 2-D patch \
         embedding + pre-norm Transformer encoder, and the upstream `state_dict` names are \
         transcribed from {modeling}. \
         Primary sources: upstream release {upstream}, paper (Alonso-Jiménez et al., ISMIR \
         2023) {paper}, HF AST modelling file {modeling}. The runtime cannot fabricate {output} \
         (FR-EX-08 — no silent partial output; CLAUDE.md 教訓 (a) 'loud-partial は \
         fake-complete より honest').",
        upstream = PRIMARY_SOURCE_UPSTREAM_HF,
        paper = PRIMARY_SOURCE_PAPER,
        modeling = PRIMARY_SOURCE_HF_AST_MODELING,
    ))
}

// ---------------------------------------------------------------------------
// MaestEncoder — the bound, runnable encoder
// ---------------------------------------------------------------------------

/// The Discogs tagging head (`ASTMLPHead`): a LayerNorm followed by a linear
/// projection onto the label set.
///
/// Transcribed from `ASTMLPHead.forward`, which is
/// `dense(layernorm(hidden_state))`.
#[derive(Debug, Clone)]
pub struct MaestTagHead {
    /// `[hidden_size]` LayerNorm gain (`classifier.layernorm.weight`).
    ln_gamma: Vec<f32>,
    /// `[hidden_size]` LayerNorm bias (`classifier.layernorm.bias`).
    ln_beta: Vec<f32>,
    /// Row-major `[num_labels, hidden_size]` (`classifier.dense.weight`).
    dense_w: Vec<f32>,
    /// `[num_labels]` (`classifier.dense.bias`).
    dense_b: Vec<f32>,
}

impl MaestTagHead {
    /// Number of labels this head projects onto.
    #[inline]
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.dense_b.len()
    }
}

/// A MAEST encoder with its weights bound and validated — the runnable handle.
///
/// Produced by [`Maest::encoder`]. The forward entry points take a **log-mel
/// plane**, not PCM, because the PCM → mel framing convention is the one axis
/// the converter does not stamp (see [`Maest::encode`]).
///
/// # Plane orientation — the axis order is load-bearing
///
/// `mel` is `[num_mel_bins, n_frames]` **row-major**: index
/// `bin * n_frames + frame`. Upstream's own feature extractor produces
/// `[n_frames, num_mel_bins]` and `ASTPatchEmbeddings.forward` transposes it
/// (`input_values.unsqueeze(1).transpose(2, 3)`) precisely so the convolution
/// walks mel bins as its first spatial axis. Handing this method the untransposed
/// plane is shape-plausible whenever the two extents coincide and silently wrong
/// otherwise, so the mel-bin extent is checked against the stamped
/// [`MaestConfig::num_mel_bins`] rather than inferred.
#[derive(Debug, Clone)]
pub struct MaestEncoder {
    config: MaestConfig,
    encoder: ViTEncoder,
    head: Option<MaestTagHead>,
    tensor_prefix: &'static str,
}

impl MaestEncoder {
    /// Binds and validates every encoder tensor. See [`Maest::encoder`].
    fn bind(
        file: &GgufFile,
        config: &MaestConfig,
        manifest: &MaestWeights,
        pos_embed_policy: PosEmbedPolicy,
    ) -> Result<Self> {
        let attrs = config.vit_attrs_with_pos_embed(pos_embed_policy)?;
        let prefix = manifest.detect_tensor_prefix()?;

        // AST is DeiT-derived: `ASTEmbeddings` allocates a cls token AND a
        // distillation token, and sizes the position table `num_patches + 2`.
        // A different prefix-token count is a different topology, not a knob.
        if config.num_prefix_tokens != 2 {
            return Err(VokraError::ModelLoad(format!(
                "maest: `{GGUF_KEY_NUM_PREFIX_TOKENS}` is {got}, but the AST backbone this \
                 binder walks is DeiT-derived and prepends exactly two learned tokens — \
                 `cls_token` and `distillation_token` — with its position table sized \
                 `num_patches + 2`. Refusing to bind a prefix block of a different width \
                 (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF_AST_MODELING}",
                got = config.num_prefix_tokens,
            )));
        }

        let hidden = config.hidden_size as usize;
        let inter = config.intermediate_size as usize;
        let patch = config.patch_size as usize;
        let seq_len = config.encoder_sequence_len();

        let embeddings = format!("{prefix}embeddings.");
        let load = |name: &str, expected: &[usize]| -> Result<Vec<f32>> {
            load_tensor(file, manifest, name, expected)
        };

        // ---- embeddings -------------------------------------------------
        // `nn.Parameter(torch.zeros(1, 1, hidden_size))` for both tokens, and
        // `torch.zeros(1, num_patches + 2, hidden_size)` for the table.
        let cls = load(&format!("{embeddings}cls_token"), &[1, 1, hidden])?;
        let distillation = load(&format!("{embeddings}distillation_token"), &[1, 1, hidden])?;
        // NOT `{embeddings}{PROBE_SUFFIX_POSITION_EMBEDDINGS}`: the probe
        // suffix is rooted at the `state_dict` prefix and already carries its
        // own `embeddings.` segment, so composing it with `embeddings` (which
        // is `{prefix}embeddings.`) yields a doubled `embeddings.embeddings.`
        // that matches nothing.
        let pos_embed = load(
            &format!("{embeddings}position_embeddings"),
            &[1, seq_len, hidden],
        )?;

        // `ASTEmbeddings.forward` concatenates `(cls, distillation, patches)`
        // along the token axis, so the prepended block is in that order and the
        // position table's leading rows line up with it.
        let mut prepended_tokens = Vec::with_capacity(2 * hidden);
        prepended_tokens.extend_from_slice(&cls);
        prepended_tokens.extend_from_slice(&distillation);

        // `Conv2d(1, hidden, kernel_size=(patch, patch))` weight is
        // `[hidden, 1, patch, patch]`; flattening its trailing dims gives the
        // `[embed_dim, patch_h * patch_w]` row-major layout `vokra_ops::vit`
        // wants, and because the channel dim is 1 the buffer is already in
        // that order.
        let proj = format!("{embeddings}patch_embeddings.projection.");
        let proj_w = load(&format!("{proj}weight"), &[hidden, 1, patch, patch])?;
        let proj_b = load(&format!("{proj}bias"), &[hidden])?;

        // ---- encoder blocks ---------------------------------------------
        let mut blocks = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer in 0..config.num_hidden_layers as usize {
            let base = format!("{prefix}encoder.layer.{layer}.");
            // `ASTLayer.forward` norms BEFORE each branch:
            //   hidden = attention(layernorm_before(x)) + x
            //   out    = ASTOutput(intermediate(layernorm_after(hidden)), hidden)
            // so `layernorm_before` is ln1 and `layernorm_after` is ln2. Reading
            // them the other way round is the post-norm function: shape-valid,
            // numerically wrong, silent.
            let ln1 = format!("{base}layernorm_before.");
            let ln2 = format!("{base}layernorm_after.");
            let attn = format!("{base}attention.attention.");
            let attn_out = format!("{base}attention.output.dense.");
            let mlp_in = format!("{base}intermediate.dense.");
            let mlp_out = format!("{base}output.dense.");

            // `nn.Linear(hidden, all_head_size, bias=config.qkv_bias)` — the
            // q/k/v biases exist only when the stamped flag says so.
            let qkv_bias = |name: &str| -> Result<Option<Vec<f32>>> {
                if config.qkv_bias {
                    Ok(Some(load(name, &[hidden])?))
                } else {
                    Ok(None)
                }
            };

            blocks.push(ViTBlockWeights {
                ln1_gamma: load(&format!("{ln1}weight"), &[hidden])?,
                ln1_beta: load(&format!("{ln1}bias"), &[hidden])?,
                attn: ViTAttnWeights {
                    wq: load(&format!("{attn}query.weight"), &[hidden, hidden])?,
                    bq: qkv_bias(&format!("{attn}query.bias"))?,
                    wk: load(&format!("{attn}key.weight"), &[hidden, hidden])?,
                    bk: qkv_bias(&format!("{attn}key.bias"))?,
                    wv: load(&format!("{attn}value.weight"), &[hidden, hidden])?,
                    bv: qkv_bias(&format!("{attn}value.bias"))?,
                    // `ASTSelfOutput.dense` is a plain `nn.Linear`, so its bias
                    // is unconditional — it does NOT follow `qkv_bias`.
                    wo: load(&format!("{attn_out}weight"), &[hidden, hidden])?,
                    bo: Some(load(&format!("{attn_out}bias"), &[hidden])?),
                },
                ln2_gamma: load(&format!("{ln2}weight"), &[hidden])?,
                ln2_beta: load(&format!("{ln2}bias"), &[hidden])?,
                mlp: ViTMlpWeights {
                    w1: load(&format!("{mlp_in}weight"), &[inter, hidden])?,
                    b1: Some(load(&format!("{mlp_in}bias"), &[inter])?),
                    w2: load(&format!("{mlp_out}weight"), &[hidden, inter])?,
                    b2: Some(load(&format!("{mlp_out}bias"), &[hidden])?),
                },
            });
        }

        // `ASTModel.forward` applies `self.layernorm` to the whole sequence
        // after the stack — the final norm `ViTEncoder` owns.
        let final_ln = format!("{prefix}layernorm.");
        let weights = ViTWeights {
            patch_embed: PatchEmbedWeights {
                proj_w,
                proj_b: Some(proj_b),
            },
            prepended_tokens,
            pos_embed,
            blocks,
            final_ln_gamma: load(&format!("{final_ln}weight"), &[hidden])?,
            final_ln_beta: load(&format!("{final_ln}bias"), &[hidden])?,
        };
        let encoder = ViTEncoder::new(attrs, weights)?;

        // ---- optional tagging head --------------------------------------
        // A bare `ASTModel` export legitimately carries none.
        let head = if manifest.count_with_prefix(TAG_HEAD_PREFIX) > 0 {
            Some(bind_tag_head(file, manifest, config)?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            encoder,
            head,
            tensor_prefix: prefix,
        })
    }

    /// The axes this encoder was bound against.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &MaestConfig {
        &self.config
    }

    /// The underlying `vokra-ops` ViT encoder.
    #[inline]
    #[must_use]
    pub fn vit(&self) -> &ViTEncoder {
        &self.encoder
    }

    /// The `state_dict` prefix this artifact was discovered to use — either
    /// [`TENSOR_PREFIX_CLASSIFICATION`] or [`TENSOR_PREFIX_BARE`].
    #[inline]
    #[must_use]
    pub fn tensor_prefix(&self) -> &'static str {
        self.tensor_prefix
    }

    /// The bound tagging head, or `None` for a bare-encoder artifact.
    #[inline]
    #[must_use]
    pub fn tag_head(&self) -> Option<&MaestTagHead> {
        self.head.as_ref()
    }

    /// The patch grid a `[num_mel_bins, n_frames]` plane produces.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when the plane is smaller than one
    ///   patch along either axis.
    pub fn patch_grid(&self, n_mels: usize, n_frames: usize) -> Result<PatchGrid> {
        self.encoder.patch_grid(n_mels, n_frames)
    }

    /// Runs the encoder over a log-mel plane, returning one row per token.
    ///
    /// The result is upstream's `sequence_output`: row `0` is the CLS token,
    /// row `1` the distillation token, and rows `2..` the patch tokens in grid
    /// row-major order (mel-bin major, then frame). Every row is
    /// [`MaestConfig::hidden_size`] wide and has already passed through the
    /// final LayerNorm, matching `ASTModel.forward`'s
    /// `sequence_output = self.layernorm(encoder_outputs[0])`.
    ///
    /// See the type docs for the plane's required orientation.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `n_mels` disagrees with the
    ///   stamped [`MaestConfig::num_mel_bins`], when
    ///   `mel.len() != n_mels * n_frames`, when the plane holds a non-finite
    ///   value, when no patch fits, or when the positional table cannot be
    ///   applied under the configured [`PosEmbedPolicy`].
    pub fn encode_mel(&self, mel: &[f32], n_mels: usize, n_frames: usize) -> Result<Vec<Vec<f32>>> {
        if n_mels != self.config.num_mel_bins as usize {
            return Err(VokraError::InvalidArgument(format!(
                "maest encode_mel: the plane has {n_mels} mel bin(s) but the artifact stamps \
                 `{GGUF_KEY_NUM_MEL_BINS}` = {want}. The plane must be \
                 [num_mel_bins, n_frames] row-major; upstream's feature extractor emits \
                 [n_frames, num_mel_bins] and `ASTPatchEmbeddings.forward` transposes it, so a \
                 caller passing the untransposed plane lands here. Refusing to reinterpret the \
                 axes (FR-EX-08).",
                want = self.config.num_mel_bins,
            )));
        }
        let (hidden, _grid) = self.encoder.forward(mel, n_mels, n_frames)?;
        let width = self.config.hidden_size as usize;
        Ok(hidden.chunks(width).map(<[f32]>::to_vec).collect())
    }

    /// Runs the encoder and pools it into the clip embedding.
    ///
    /// The pooling is upstream's, verbatim:
    /// `pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2` —
    /// the **mean of the CLS and distillation tokens**. That is a DeiT-style
    /// rule rather than either of the two conventions `vokra_ops::vit`'s
    /// [`vokra_ops::vit::ViTPooling`] offers, so it is computed here instead of
    /// approximated with a CLS-only or mean-over-patches variant, both of which
    /// would return a different vector without failing.
    ///
    /// # Errors
    ///
    /// - See [`Self::encode_mel`].
    pub fn embed_mel(&self, mel: &[f32], n_mels: usize, n_frames: usize) -> Result<Vec<f32>> {
        let tokens = self.encode_mel(mel, n_mels, n_frames)?;
        // `encode_mel` returns `n_prefix + n_patches` rows and `bind` pinned
        // `n_prefix == 2`, so both rows exist; the grid check inside `forward`
        // already refused a plane too small to produce any patch token.
        let (cls, distillation) = (&tokens[0], &tokens[1]);
        Ok(cls
            .iter()
            .zip(distillation.iter())
            .map(|(a, b)| (a + b) * 0.5)
            .collect())
    }

    /// Runs the encoder, pools it, and applies the Discogs tagging head,
    /// returning one **logit** per label.
    ///
    /// The head is `ASTMLPHead.forward`: `dense(layernorm(pooled))`. No softmax
    /// or sigmoid is applied — the return value is raw logits, and which
    /// activation is appropriate is the caller's modelling decision.
    ///
    /// The logits are **unnamed**: the artifact carries no label list, so
    /// mapping index `i` onto a Discogs genre / mood / instrument / era string
    /// is not possible from the GGUF alone.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] when the artifact carries no tagging
    ///   head (a bare-`ASTModel` export) — the message says so rather than
    ///   returning an empty or zero-filled vector.
    /// - Otherwise see [`Self::encode_mel`].
    pub fn tag_mel(&self, mel: &[f32], n_mels: usize, n_frames: usize) -> Result<Vec<f32>> {
        let Some(head) = self.head.as_ref() else {
            return Err(VokraError::UnsupportedOp(format!(
                "maest tag_mel: this artifact carries no tagging head — no tensor on disk sits \
                 under `{TAG_HEAD_PREFIX}`, which is what a bare `ASTModel` export looks like \
                 (only an `ASTForAudioClassification` export carries \
                 `{TAG_HEAD_PREFIX}layernorm.*` + `{TAG_HEAD_PREFIX}dense.*`). The encoder \
                 itself bound fine: use `MaestEncoder::encode_mel` or \
                 `MaestEncoder::embed_mel`. Refusing to return empty or zero-filled logits \
                 (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF_AST_MODELING}"
            )));
        };
        let pooled = self.embed_mel(mel, n_mels, n_frames)?;
        let hidden = self.config.hidden_size as usize;
        let eps = self.config.layer_norm_eps();

        // `ASTMLPHead.layernorm` is built with `eps=config.layer_norm_eps`, the
        // same epsilon the encoder's norms use.
        let mut normed = pooled;
        let n = hidden as f32;
        let mean = normed.iter().sum::<f32>() / n;
        let var = normed
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n;
        let inv = 1.0 / (var + eps).sqrt();
        for ((slot, &g), &b) in normed
            .iter_mut()
            .zip(head.ln_gamma.iter())
            .zip(head.ln_beta.iter())
        {
            *slot = (*slot - mean) * inv * g + b;
        }

        // `nn.Linear` weight is `[out_features, in_features]` row-major.
        Ok(head
            .dense_b
            .iter()
            .enumerate()
            .map(|(o, bias)| {
                let row = &head.dense_w[o * hidden..(o + 1) * hidden];
                let dot: f32 = row.iter().zip(normed.iter()).map(|(a, b)| a * b).sum();
                bias + dot
            })
            .collect())
    }
}

/// Shape-checks a tensor against the stamped topology, then decodes it to
/// `f32` through the canonical `GgufFile::tensor_f32` path (so K-quantised
/// artifacts dequantise on the way in).
fn load_tensor(
    file: &GgufFile,
    manifest: &MaestWeights,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>> {
    manifest.require_tensor_dims(name, expected)?;
    file.tensor_f32(name).map_err(|e| {
        VokraError::ModelLoad(format!(
            "maest: tensor `{name}` is present with the expected dims {expected:?} but its \
             payload failed to decode: {e}. Refusing to substitute zeros (FR-EX-08)."
        ))
    })
}

/// Binds `ASTMLPHead` — `classifier.layernorm.*` + `classifier.dense.*`.
///
/// The label count is cross-checked against the stamped
/// [`MaestConfig::num_labels`]: the converter stamps it from `config.json`'s
/// `id2label` cardinality while the head projection carries it as its leading
/// dimension, so the two are independent witnesses and a disagreement means the
/// payload and the stamps describe different checkpoints.
fn bind_tag_head(
    file: &GgufFile,
    manifest: &MaestWeights,
    config: &MaestConfig,
) -> Result<MaestTagHead> {
    let hidden = config.hidden_size as usize;
    let labels = config.num_labels as usize;
    let ln = format!("{TAG_HEAD_PREFIX}layernorm.");
    let dense = format!("{TAG_HEAD_PREFIX}dense.");
    Ok(MaestTagHead {
        ln_gamma: load_tensor(file, manifest, &format!("{ln}weight"), &[hidden])?,
        ln_beta: load_tensor(file, manifest, &format!("{ln}bias"), &[hidden])?,
        dense_w: load_tensor(file, manifest, &format!("{dense}weight"), &[labels, hidden])?,
        dense_b: load_tensor(file, manifest, &format!("{dense}bias"), &[labels])?,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the MAEST runtime binder — contract-constant pins against the
    //! converter, metadata round-trip, loud negative space on every stated
    //! blocker, arch-tag distinctness, and the read-not-guessed label count.
    //!
    //! # What "round-trip" means here
    //!
    //! On real audio this would be `encode(...)` returning hidden states, but
    //! the MAEST forward is loud-partial (see the module doc). Fabricating an
    //! output would violate CLAUDE.md 教訓 (a)
    //! (「loud-partial は fake-complete より honest」). The round-trips we *can*
    //! honestly test:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` / `DEFAULT_LICENSE_SPDX` and the two metadata keys match
    //!    the converter exactly, so a converter drift without a binder-side
    //!    follow-through fails here.
    //! 2. **Metadata round-trip** — a synthetic GGUF shaped like the
    //!    converter's output binds, and every stamp reads back.
    //! 3. **Loud negative space** — missing arch, foreign arch, empty manifest,
    //!    missing tensor, wrong dims, and all three forward surfaces fire at
    //!    their documented surface point in their documented variant.
    //! 4. **Arch distinctness pin** — the tag differs from every sibling arch,
    //!    `ast` included.
    //! 5. **Label count is data, not a constant** — two synthetic artifacts
    //!    with different head widths report different counts, which is only
    //!    possible if the value is read rather than hardcoded.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// The upstream MAEST axes, transcribed by the converter from
    /// `config.json` + `preprocessor_config.json`.
    ///
    /// Pinned here as literals rather than imported from `vokra-convert`, so
    /// that a converter-side drift fails a test instead of propagating: this
    /// crate deliberately has no dependency edge onto the writer.
    fn upstream_config() -> MaestConfig {
        MaestConfig {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            patch_size: 16,
            frequency_stride: 10,
            time_stride: 10,
            num_mel_bins: 96,
            max_length: 1876,
            num_labels: 400,
            qkv_bias: true,
            hidden_act: "gelu".to_owned(),
            layer_norm_eps_scaled_1e9: 1_000,
            hidden_dropout_scaled_1e3: 0,
            attention_dropout_scaled_1e3: 0,
            freq_patches: 9,
            time_patches: 187,
            num_patches: 1683,
            num_prefix_tokens: 2,
            sample_rate: 16_000,
            n_fft: 512,
            hop_length: 256,
            win_length: 512,
            window: "hann".to_owned(),
            mel_scale: "slaney".to_owned(),
            mel_norm: "slaney".to_owned(),
            fmin_hz: 0,
            fmax_hz: 8_000,
            log_compression: "logC".to_owned(),
            log_compression_mul: 10_000,
            do_normalize: true,
            norm_mean: 2.067_556_860_985_54,
            norm_std: 1.268_292_820_667_291,
        }
    }

    /// A deliberately tiny topology, so a full synthetic weight set stays small
    /// enough to build, bind and run inside a unit test.
    ///
    /// The patch grid closes on itself the same way the real one does:
    /// `freq_patches = (4 - 2) / 1 + 1 = 3`, `time_patches = (5 - 2) / 1 + 1 = 4`,
    /// `num_patches = 12`, sequence length `12 + 2 = 14`.
    fn tiny_config() -> MaestConfig {
        MaestConfig {
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 8,
            patch_size: 2,
            frequency_stride: 1,
            time_stride: 1,
            num_mel_bins: 4,
            max_length: 5,
            num_labels: 3,
            freq_patches: 3,
            time_patches: 4,
            num_patches: 12,
            ..upstream_config()
        }
    }

    /// Writes the full `vokra.maest.*` axis group, with the same key spellings
    /// and metadata types the converter uses.
    fn stamp_axis_group(b: &mut GgufBuilder, cfg: &MaestConfig) {
        // No real key is the empty string, so nothing is withheld.
        stamp_axis_group_except(b, cfg, "");
    }

    /// Writes the axis group with exactly one key withheld — the fixture behind
    /// the per-key missing-axis test.
    fn stamp_axis_group_except(b: &mut GgufBuilder, cfg: &MaestConfig, withheld: &str) {
        let u32s: [(&str, u32); 24] = [
            (GGUF_KEY_HIDDEN_SIZE, cfg.hidden_size),
            (GGUF_KEY_NUM_HIDDEN_LAYERS, cfg.num_hidden_layers),
            (GGUF_KEY_NUM_ATTENTION_HEADS, cfg.num_attention_heads),
            (GGUF_KEY_INTERMEDIATE_SIZE, cfg.intermediate_size),
            (GGUF_KEY_PATCH_SIZE, cfg.patch_size),
            (GGUF_KEY_FREQUENCY_STRIDE, cfg.frequency_stride),
            (GGUF_KEY_TIME_STRIDE, cfg.time_stride),
            (GGUF_KEY_NUM_MEL_BINS, cfg.num_mel_bins),
            (GGUF_KEY_MAX_LENGTH, cfg.max_length),
            (GGUF_KEY_NUM_LABELS, cfg.num_labels),
            (
                GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9,
                cfg.layer_norm_eps_scaled_1e9,
            ),
            (
                GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3,
                cfg.hidden_dropout_scaled_1e3,
            ),
            (
                GGUF_KEY_ATTENTION_DROPOUT_SCALED_1E3,
                cfg.attention_dropout_scaled_1e3,
            ),
            (GGUF_KEY_FREQ_PATCHES, cfg.freq_patches),
            (GGUF_KEY_TIME_PATCHES, cfg.time_patches),
            (GGUF_KEY_NUM_PATCHES, cfg.num_patches),
            (GGUF_KEY_NUM_PREFIX_TOKENS, cfg.num_prefix_tokens),
            (GGUF_KEY_SAMPLE_RATE, cfg.sample_rate),
            (GGUF_KEY_N_FFT, cfg.n_fft),
            (GGUF_KEY_HOP_LENGTH, cfg.hop_length),
            (GGUF_KEY_WIN_LENGTH, cfg.win_length),
            (GGUF_KEY_FMIN_HZ, cfg.fmin_hz),
            (GGUF_KEY_FMAX_HZ, cfg.fmax_hz),
            (GGUF_KEY_LOG_COMPRESSION_MUL, cfg.log_compression_mul),
        ];
        for (key, value) in u32s {
            if key != withheld {
                b.add_u32(key, value);
            }
        }

        let strings: [(&str, &str); 5] = [
            (GGUF_KEY_HIDDEN_ACT, cfg.hidden_act.as_str()),
            (GGUF_KEY_WINDOW, cfg.window.as_str()),
            (GGUF_KEY_MEL_SCALE, cfg.mel_scale.as_str()),
            (GGUF_KEY_MEL_NORM, cfg.mel_norm.as_str()),
            (GGUF_KEY_LOG_COMPRESSION, cfg.log_compression.as_str()),
        ];
        for (key, value) in strings {
            if key != withheld {
                b.add_string(key, value);
            }
        }

        let bools: [(&str, bool); 2] = [
            (GGUF_KEY_QKV_BIAS, cfg.qkv_bias),
            (GGUF_KEY_DO_NORMALIZE, cfg.do_normalize),
        ];
        for (key, value) in bools {
            if key != withheld {
                b.add_bool(key, value);
            }
        }

        let f64s: [(&str, f64); 2] = [
            (GGUF_KEY_NORM_MEAN, cfg.norm_mean),
            (GGUF_KEY_NORM_STD, cfg.norm_std),
        ];
        for (key, value) in f64s {
            if key != withheld {
                b.add_metadata(key, vokra_core::gguf::GgufMetadataValue::F64(value));
            }
        }
    }

    /// Copies every tensor out of `source` under a fresh axis-group stamp.
    ///
    /// Used to build artifacts where the stamps and the payload deliberately
    /// disagree, which is how the cross-check assertions are exercised.
    fn restamped(source: &GgufFile, cfg: &MaestConfig) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        stamp_axis_group(&mut b, cfg);
        for info in source.tensors() {
            b.add_tensor(
                &info.name,
                info.dtype,
                info.dimensions.clone(),
                source.tensor_bytes(info).to_vec(),
            )
            .expect("add_tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// Deterministic small-value source. No committed fixtures: every synthetic
    /// weight below is generated in-test from a fixed seed, so the tests
    /// reproduce on every platform and nothing here pretends to be upstream
    /// data.
    struct Lcg(u64);

    impl Lcg {
        fn new() -> Self {
            Self(0x2545_F491_4F6C_DD1D)
        }

        /// Values in `[-0.5, 0.5)`, small enough that a 1-block forward stays
        /// comfortably inside `f32` range.
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((self.0 >> 40) as f32) / ((1u32 << 24) as f32);
            unit - 0.5
        }
    }

    /// Adds one deterministic F32 tensor of the given shape.
    fn add_f32(b: &mut GgufBuilder, rng: &mut Lcg, name: &str, dims: &[u64]) {
        let n: u64 = dims.iter().product();
        let mut bytes = Vec::with_capacity((n * 4) as usize);
        for _ in 0..n {
            bytes.extend_from_slice(&rng.next_f32().to_le_bytes());
        }
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), bytes)
            .expect("add_tensor");
    }

    /// Builds a GGUF carrying a COMPLETE synthetic weight set for `cfg`, under
    /// the given `state_dict` prefix, optionally including the tagging head.
    ///
    /// Every tensor name and shape here is the one `MaestEncoder::bind` walks,
    /// so this fixture doubles as an executable statement of the transcribed
    /// manifest.
    fn synthetic_weights_gguf(cfg: &MaestConfig, prefix: &str, with_head: bool) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::NonCommercialShareAlike.as_str(),
        );
        stamp_axis_group(&mut b, cfg);

        let mut rng = Lcg::new();
        let hidden = u64::from(cfg.hidden_size);
        let inter = u64::from(cfg.intermediate_size);
        let patch = u64::from(cfg.patch_size);
        let seq = cfg.encoder_sequence_len() as u64;

        let emb = format!("{prefix}embeddings.");
        add_f32(
            &mut b,
            &mut rng,
            &format!("{emb}cls_token"),
            &[1, 1, hidden],
        );
        add_f32(
            &mut b,
            &mut rng,
            &format!("{emb}distillation_token"),
            &[1, 1, hidden],
        );
        add_f32(
            &mut b,
            &mut rng,
            &format!("{emb}position_embeddings"),
            &[1, seq, hidden],
        );
        add_f32(
            &mut b,
            &mut rng,
            &format!("{emb}patch_embeddings.projection.weight"),
            &[hidden, 1, patch, patch],
        );
        add_f32(
            &mut b,
            &mut rng,
            &format!("{emb}patch_embeddings.projection.bias"),
            &[hidden],
        );

        for layer in 0..cfg.num_hidden_layers {
            let base = format!("{prefix}encoder.layer.{layer}.");
            for suffix in ["layernorm_before", "layernorm_after"] {
                add_f32(
                    &mut b,
                    &mut rng,
                    &format!("{base}{suffix}.weight"),
                    &[hidden],
                );
                add_f32(&mut b, &mut rng, &format!("{base}{suffix}.bias"), &[hidden]);
            }
            for proj in ["query", "key", "value"] {
                add_f32(
                    &mut b,
                    &mut rng,
                    &format!("{base}attention.attention.{proj}.weight"),
                    &[hidden, hidden],
                );
                if cfg.qkv_bias {
                    add_f32(
                        &mut b,
                        &mut rng,
                        &format!("{base}attention.attention.{proj}.bias"),
                        &[hidden],
                    );
                }
            }
            add_f32(
                &mut b,
                &mut rng,
                &format!("{base}attention.output.dense.weight"),
                &[hidden, hidden],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{base}attention.output.dense.bias"),
                &[hidden],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{base}intermediate.dense.weight"),
                &[inter, hidden],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{base}intermediate.dense.bias"),
                &[inter],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{base}output.dense.weight"),
                &[hidden, inter],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{base}output.dense.bias"),
                &[hidden],
            );
        }

        add_f32(
            &mut b,
            &mut rng,
            &format!("{prefix}layernorm.weight"),
            &[hidden],
        );
        add_f32(
            &mut b,
            &mut rng,
            &format!("{prefix}layernorm.bias"),
            &[hidden],
        );

        if with_head {
            let labels = u64::from(cfg.num_labels);
            add_f32(
                &mut b,
                &mut rng,
                &format!("{TAG_HEAD_PREFIX}layernorm.weight"),
                &[hidden],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{TAG_HEAD_PREFIX}layernorm.bias"),
                &[hidden],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{TAG_HEAD_PREFIX}dense.weight"),
                &[labels, hidden],
            );
            add_f32(
                &mut b,
                &mut rng,
                &format!("{TAG_HEAD_PREFIX}dense.bias"),
                &[labels],
            );
        }

        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// A deterministic `[num_mel_bins, n_frames]` row-major plane.
    fn mel_plane(cfg: &MaestConfig, n_frames: usize) -> Vec<f32> {
        let mut rng = Lcg::new();
        (0..cfg.num_mel_bins as usize * n_frames)
            .map(|_| rng.next_f32())
            .collect()
    }

    /// Tensor-name samples shaped like the converter's own test module's
    /// "realistic upstream state-dict name" choices (the HF
    /// `ASTForAudioClassification` wrapper places the body under an
    /// `audio_spectrogram_transformer.` prefix). Used only to give the manifest
    /// something to hold in the metadata-level tests; the *real* transcribed
    /// manifest is exercised by [`synthetic_weights_gguf`].
    const SAMPLE_TENSORS: [(&str, [u64; 2]); 3] = [
        (
            "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight",
            [4, 12],
        ),
        (
            "audio_spectrogram_transformer.encoder.layer.0.output.dense.weight",
            [4, 6],
        ),
        ("audio_spectrogram_transformer.layernorm.weight", [4, 1]),
    ];

    /// Builds a GGUF shaped like `convert_maest_file`'s output: arch + name +
    /// category + upstream HF slug, an optional weight-licence class, an
    /// optional FR-MD-09 attribution stamp, the sample tensor manifest, and —
    /// when `head_labels` is `Some(n)` — a tagging head of `n` labels over a
    /// width-4 hidden, shaped like the HF `ASTForAudioClassification` wrapper
    /// (a 1-D LayerNorm pair plus the 2-D `[n, 4]` projection and its 1-D bias).
    fn maest_builder(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
        head_labels: Option<u64>,
    ) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            "mtg-upf/discogs-maest-30s-pw-129e (test)",
        );
        stamp_axis_group(&mut b, &upstream_config());
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        }
        if attribution {
            b.add_string(
                chunks::KEY_PROVENANCE_ATTRIBUTION,
                "MAEST (mtg-upf/discogs-maest-30s-pw-129e) weights, licensed CC BY-NC-SA 4.0.",
            );
        }
        if with_tensors {
            for (name, dims) in SAMPLE_TENSORS {
                let elems = dims[0] * dims[1];
                b.add_tensor(
                    name,
                    GgmlType::F32,
                    dims.to_vec(),
                    vec![0u8; (elems * 4) as usize],
                )
                .expect("add_tensor");
            }
        }
        if let Some(labels) = head_labels {
            // A PyTorch `nn.Linear` weight is `[out_features, in_features]`, so
            // the leading dim is the label-set size. The LayerNorm siblings are
            // 1-D and must NOT be mistaken for the projection.
            b.add_tensor(
                "classifier.layernorm.weight",
                GgmlType::F32,
                vec![4],
                vec![0u8; 16],
            )
            .expect("add_tensor");
            b.add_tensor(
                "classifier.dense.weight",
                GgmlType::F32,
                vec![labels, 4],
                vec![0u8; (labels * 4 * 4) as usize],
            )
            .expect("add_tensor");
            b.add_tensor(
                "classifier.dense.bias",
                GgmlType::F32,
                vec![labels],
                vec![0u8; (labels * 4) as usize],
            )
            .expect("add_tensor");
        }
        b
    }

    /// Parses a `maest_builder` result into a `GgufFile`.
    fn maest_gguf(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
        head_labels: Option<u64>,
    ) -> GgufFile {
        let b = maest_builder(weight_license_class, attribution, with_tensors, head_labels);
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Contract-constant pin (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        // Mirrors of `crates/vokra-convert/src/models/maest.rs`. A converter
        // drift without a binder-side follow-through lands here in the same
        // commit or fails this test.
        assert_eq!(ARCH, "maest", "arch tag pin");
        assert_eq!(NAME, "maest-30s-pw-129e", "canonical variant name pin");
        assert_eq!(
            CATEGORY, "music-embedding",
            "category pin — MAEST is music-domain, NOT the general `audio-tagging` bucket"
        );
        assert_eq!(
            UPSTREAM_HF, "mtg-upf/discogs-maest-30s-pw-129e",
            "upstream HF slug pin"
        );
        assert_eq!(
            DEFAULT_LICENSE_SPDX, "cc-by-nc-sa-4.0",
            "T4 tier + ShareAlike cascade"
        );
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            GGUF_KEY_PROVENANCE_UPSTREAM_HF,
            "vokra.provenance.upstream_hf"
        );
        assert_eq!(
            UPSTREAM_PARAM_COUNT_F32, 86_858_128,
            "HF API `parameters.F32` as recorded by the converter on 2026-08-13"
        );

        // The weight SPDX must resolve to the class the converter stamps, and
        // that class must carry all three obligations.
        assert_eq!(
            LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX),
            LicenseClass::NonCommercialShareAlike,
            "cc-by-nc-sa-4.0 must classify as NonCommercialShareAlike"
        );
        assert!(
            !LicenseClass::NonCommercialShareAlike.commercial_ok(),
            "NC: commercial use forbidden"
        );
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_license_preserved(),
            "SA: share-alike cascade on any downstream distribution"
        );
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_attribution(),
            "BY: attribution cascade"
        );
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_research_flag(),
            "T4 tier must be gated at the M2-13 compliance gate"
        );
        assert!(
            !LicenseClass::NonCommercialShareAlike.redistributable(),
            "T4 tier must be refused at publish without an explicit opt-in"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Arch-tag distinctness pin
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_encoder_arches() {
        // Every sibling below is a real converter arch tag. Sharing one would
        // let runtime dispatch bind a foreign topology over a MAEST payload
        // (FR-EX-08).
        for sibling in [
            // Same AST backbone, different objective / domain / taxonomy.
            "ast",
            // SSL audio/music-embedding neighbourhood.
            "atst",
            "beats",
            "eat",
            "dasheng",
            "m2d",
            "mert",
            "muq",
            // Supervised tagging CNNs / contrastive text-audio.
            "yamnet",
            "panns",
            "clap",
            // wav2vec2 lineage — raw-waveform 1-D conv stem, not a log-mel
            // patch grid.
            "hubert",
            "wav2vec2_ctc",
            "wavlm_sv",
            "emotion2vec",
        ] {
            assert_ne!(
                ARCH, sibling,
                "maest must not share an arch tag with `{sibling}` — a different objective, \
                 domain or head means a different topology (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Metadata round-trip on a synthetic converter-shaped GGUF
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_a_synthetic_converter_shaped_gguf() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            true,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("a converter-shaped GGUF must bind");

        // Metadata surfaces round-trip.
        assert_eq!(m.name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_hf(), Some(UPSTREAM_HF));
        assert_eq!(m.model_id(), Some(NAME));
        assert!(m.source().is_some(), "provenance source must surface");

        // Tensor manifest.
        assert_eq!(m.tensor_count(), SAMPLE_TENSORS.len());
        assert_eq!(m.weights().tensor_names().len(), SAMPLE_TENSORS.len());
        // Dims round-trip verbatim and are NOT reversed by the writer/reader
        // pair — pinned with an asymmetric shape so a future reversal is caught.
        assert_eq!(
            m.weights().tensor_dims(
                "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight"
            ),
            Some([4usize, 12].as_slice())
        );

        // No head in this artifact: honest `None`, not a taxonomy fallback.
        assert!(!m.has_tag_head());
        assert_eq!(m.label_count(), None);

        // Licence + FR-MD-09 attribution surfaces.
        assert_eq!(m.weight_license(), LicenseClass::NonCommercialShareAlike);
        let attr = m.attribution().expect("attribution stamp must surface");
        assert!(
            attr.contains("CC BY-NC-SA 4.0"),
            "attribution text must name the licence: {attr}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Missing arch fails loud
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Maest::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native maest GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5. Foreign arch fails loud, naming BOTH tags
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // An `ast` GGUF handed to the MAEST binder by mistake — the sharpest
        // confusable in the fleet, because MAEST is literally built on the AST
        // backbone. A silent bind would look plausible right up until the
        // numbers (and the label taxonomy) are wrong: exactly the misroute
        // FR-EX-08 forbids.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "ast");
        b.add_string(chunks::KEY_MODEL_NAME, "ast-finetuned-audioset");
        b.add_tensor("ast.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Maest::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH the actual and the expected tag.
                assert!(
                    m.contains("`ast`"),
                    "message must name the arch actually found, got `{m}`"
                );
                assert!(
                    m.contains("`maest`"),
                    "message must name the expected arch, got `{m}`"
                );
                // The neighbourhood must be enumerated so the reader knows
                // which loader they actually wanted.
                for sibling in [
                    "atst",
                    "beats",
                    "eat",
                    "dasheng",
                    "m2d",
                    "mert",
                    "muq",
                    "yamnet",
                    "panns",
                    "clap",
                    "hubert",
                    "wav2vec2_ctc",
                    "wavlm_sv",
                    "emotion2vec",
                ] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling `{sibling}` enumerated in the error: {m}"
                    );
                }
                assert!(
                    m.contains("backbone identity is not topology identity"),
                    "message must explain why sharing AST's backbone is not sharing its \
                     topology, got `{m}`"
                );
                assert!(
                    m.contains("Discogs"),
                    "message should state what makes MAEST distinct, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6. Empty tensor manifest fails loud (never binds an all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        // Correct arch + full metadata but zero tensors.
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            false,
            None,
        );
        let Err(err) = Maest::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
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
                    m.contains("vokra-cli convert --model maest-30s-pw-129e"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. require_tensor names the missing tensor
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");

        let missing = "audio_spectrogram_transformer.encoder.layer.11.output.dense.weight";
        let Err(err) = m.weights().require_tensor(missing) else {
            panic!("expected ModelLoad for a tensor that is not on disk");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(missing),
                    "message must name the missing tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("nearest names on disk"),
                    "message must offer the nearby names, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-zero-substitution clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // A tensor that IS on disk resolves.
        assert_eq!(
            m.weights()
                .require_tensor("audio_spectrogram_transformer.layernorm.weight")
                .expect("present tensor must resolve"),
            [4usize, 1].as_slice()
        );
    }

    // -----------------------------------------------------------------------
    // 8. require_tensor_dims names BOTH expected and actual dims
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_dims_names_expected_and_actual() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");
        let name = "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight";

        // Exact match passes.
        m.weights()
            .require_tensor_dims(name, &[4, 12])
            .expect("matching dims must pass");

        // Mismatch fails loud, naming both sides.
        let Err(err) = m.weights().require_tensor_dims(name, &[4, 36]) else {
            panic!("expected ModelLoad on a dims mismatch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("[4, 12]"),
                    "message must name the ACTUAL dims, got `{msg}`"
                );
                assert!(
                    msg.contains("[4, 36]"),
                    "message must name the EXPECTED dims, got `{msg}`"
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
    // 9. encode loud-partials on the framing convention — and ONLY on that
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partial_names_the_framing_gap_and_drops_the_resolved_ones() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");

        // A legitimately shaped buffer, so the loud-partial gate is what fires
        // (not some pre-encode length validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("maest encode"), "surface must be named: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // ---- the blocker that is REAL today --------------------
                assert!(
                    msg.contains("FRAMING") && msg.contains("CENTERING"),
                    "must name the STFT framing / centering convention as the gap: {msg}"
                );
                for key in ["center", "pad_mode", "vokra.frontend.*"] {
                    assert!(
                        msg.contains(key),
                        "must name the unstamped front-end key `{key}`: {msg}"
                    );
                }
                // It must say WHY guessing is refused, not merely that it is.
                assert!(
                    msg.contains("half a window"),
                    "must explain that a wrong framing choice is silently wrong: {msg}"
                );

                // ---- the boundary: what IS available -------------------
                // The stamped front-end axes are echoed so a reader can see
                // that only one axis is missing rather than the whole group.
                for stamped in [
                    "sample_rate=16000",
                    "n_fft=512",
                    "hop_length=256",
                    "win_length=512",
                    "window=hann",
                    "num_mel_bins=96",
                    "mel_scale=slaney",
                    "mel_norm=slaney",
                    "fmax_hz=8000",
                    "log_compression=logC",
                ] {
                    assert!(
                        msg.contains(stamped),
                        "must echo the stamped front-end axis `{stamped}`: {msg}"
                    );
                }
                // And the real path must be pointed at by name.
                for path in [
                    "Maest::encoder",
                    "MaestEncoder::encode_mel",
                    "vokra_ops::vit",
                ] {
                    assert!(
                        msg.contains(path),
                        "must point at the real entry point `{path}`: {msg}"
                    );
                }

                // ---- the RESOLVED blockers must be GONE ----------------
                // This is the heart of the test: a stale claim in an error
                // message actively misleads whoever reads it next.
                assert!(
                    !msg.contains("NO `vokra.maest.*` AXIS CHUNK GROUP"),
                    "the axis group IS stamped now — that claim must not survive: {msg}"
                );
                assert!(
                    !msg.contains("NO ViT-STYLE ENCODER PRIMITIVE"),
                    "`vokra_ops::vit` exists now — that claim must not survive: {msg}"
                );
                assert!(
                    !msg.contains("NO VERIFIED TENSOR-NAME MANIFEST"),
                    "the manifest is transcribed now — that claim must not survive: {msg}"
                );
                assert!(
                    !msg.contains("SAME SHARED GAP"),
                    "the shared SSL-fleet primitive gap is closed: {msg}"
                );
                for stale in [
                    "vokra_ops::conformer",
                    "vokra_ops::ebranchformer",
                    "vokra_ops::zipformer",
                ] {
                    assert!(
                        !msg.contains(stale),
                        "must not still argue why `{stale}` is not a substitute: {msg}"
                    );
                }

                // Primary sources, including the newly cited modelling file.
                for url in [
                    PRIMARY_SOURCE_UPSTREAM_HF,
                    PRIMARY_SOURCE_PAPER,
                    PRIMARY_SOURCE_HF_AST_MODELING,
                ] {
                    assert!(msg.contains(url), "expected primary source `{url}`: {msg}");
                }

                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-fabricated-output clause: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 10. embed loud-partials on the same single blocker
    // -----------------------------------------------------------------------

    #[test]
    fn embed_loud_partials_on_the_same_framing_gap() {
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");

        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.embed(&pcm) else {
            panic!("embed must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("maest embed"), "surface must be named: {msg}");
                assert!(
                    msg.contains("pooled clip embedding"),
                    "must name the output it refuses to fabricate: {msg}"
                );
                assert!(
                    msg.contains("FRAMING") && msg.contains("CENTERING"),
                    "must name the framing gap: {msg}"
                );
                assert!(
                    msg.contains("MaestEncoder::embed_mel"),
                    "must point at the real mel-plane entry point: {msg}"
                );
                assert!(
                    !msg.contains("NO ViT-STYLE ENCODER PRIMITIVE"),
                    "the resolved primitive blocker must not survive: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-fabricated-output clause: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 11. tag loud-partials on the framing gap, NOT on the taxonomy
    // -----------------------------------------------------------------------

    #[test]
    fn tag_loud_partials_on_the_framing_gap_not_the_taxonomy() {
        // Give this artifact a head, so the refusal is unambiguously about the
        // deferred forward rather than about a missing classifier.
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            Some(11),
        );
        let m = Maest::from_gguf(&file).expect("bind");
        assert!(m.has_tag_head(), "fixture must carry a head");

        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.tag(&pcm) else {
            panic!("tag must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("maest tag"), "surface must be named: {msg}");
                assert!(
                    msg.contains("Discogs tag logits"),
                    "must name the output it refuses to fabricate: {msg}"
                );
                // The one real blocker.
                assert!(
                    msg.contains("FRAMING") && msg.contains("CENTERING"),
                    "must name the framing gap: {msg}"
                );
                assert!(
                    msg.contains("MaestEncoder::tag_mel"),
                    "must point at the real mel-plane entry point: {msg}"
                );
                // The taxonomy is NOT claimed as a blocker on logits any more:
                // `tag` returns logits, and the head produces logits. The gap
                // that remains is naming them, which lives in the rustdoc.
                assert!(
                    !msg.contains("NO LABEL TAXONOMY"),
                    "the taxonomy never blocked LOGITS — that claim must not survive: {msg}"
                );
                assert!(
                    !msg.contains("SAME SHARED GAP"),
                    "the shared SSL-fleet primitive gap is closed: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-fabricated-output clause: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 12. The label count is READ from the artifact, never a constant
    // -----------------------------------------------------------------------

    #[test]
    fn label_count_is_read_from_disk_never_a_constant() {
        // Two artifacts, two different head widths. A hardcoded taxonomy size
        // could not satisfy both, so this test is what makes "read, never
        // guessed" mechanically true rather than merely documented.
        for labels in [7u64, 13] {
            let file = maest_gguf(
                Some(LicenseClass::NonCommercialShareAlike),
                false,
                true,
                Some(labels),
            );
            let m = Maest::from_gguf(&file).expect("bind");
            assert!(m.has_tag_head(), "head must be discovered for {labels}");
            assert_eq!(
                m.label_count(),
                Some(labels as usize),
                "label count must track the head projection's leading dim on disk"
            );
            // The head tensors are reported verbatim, LayerNorm siblings
            // included — three under the prefix in this fixture.
            assert_eq!(m.weights().tag_head_tensors().len(), 3);
        }
    }

    #[test]
    fn label_count_is_none_without_a_head() {
        // A bare-encoder export is legitimate: no head, no count, no error, and
        // above all no fallback taxonomy number.
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");
        assert!(!m.has_tag_head());
        assert!(m.weights().tag_head_tensors().is_empty());
        assert_eq!(
            m.label_count(),
            None,
            "no head on disk must yield None, never a guessed taxonomy size"
        );
    }

    #[test]
    fn label_count_is_none_when_the_head_layout_is_ambiguous() {
        // Two 2-D tensors under the head prefix: which one is the label
        // projection cannot be decided from shape alone, so report nothing.
        let mut b = maest_builder(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            Some(9),
        );
        b.add_tensor(
            "classifier.extra_projection.weight",
            GgmlType::F32,
            vec![5, 4],
            vec![0u8; 80],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let m = Maest::from_gguf(&file).expect("bind");

        assert!(m.has_tag_head());
        assert_eq!(
            m.label_count(),
            None,
            "an ambiguous head layout must report None rather than pick a dim"
        );
    }

    // -----------------------------------------------------------------------
    // 13. Missing licence stamp fails closed to Unknown
    // -----------------------------------------------------------------------

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        // No provenance licence stamp at all: the binder still binds (arch +
        // manifest are the load gates), but the licence surface must fail
        // closed.
        let file = maest_gguf(None, false, true, None);
        let m = Maest::from_gguf(&file).expect("arch + manifest are the load gates");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an absent weight-licence stamp must fail closed to Unknown"
        );
        assert!(m.attribution().is_none(), "no stamp => no attribution");
        assert!(
            LicenseClass::Unknown.requires_research_flag(),
            "Unknown must be gated at the M2-13 compliance gate"
        );
    }

    // -----------------------------------------------------------------------
    // 14. Compliance gate: NonCommercialShareAlike is REFUSED under strict
    // -----------------------------------------------------------------------

    #[test]
    fn compliance_gate_refuses_non_commercial_under_strict_and_allows_research_opt_in() {
        let stamped = maest_builder(
            Some(LicenseClass::NonCommercialShareAlike),
            true,
            true,
            None,
        )
        .to_bytes()
        .expect("serialize");

        // Strict refuses — MAEST is T4, so this refusal is the fail-closed
        // default working as intended, NOT a bug.
        let Err(err) = Maest::from_gguf_with_policy(&stamped, &CompliancePolicy::strict()) else {
            panic!("a cc-by-nc-sa-4.0 artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for a NonCommercialShareAlike weight, got {err:?}"
        );

        // An explicit research opt-in unlocks it (and emits the mandatory
        // research-only warning inside the gate).
        let research = CompliancePolicy::strict().with_research_license(true);
        let m = Maest::from_gguf_with_policy(&stamped, &research)
            .expect("the research opt-in must unlock a T4 weight");
        assert_eq!(m.weight_license(), LicenseClass::NonCommercialShareAlike);

        // An unstamped artifact resolves to Unknown and is refused too —
        // fail-closed, never a silent substitution.
        let unstamped = maest_builder(None, false, true, None)
            .to_bytes()
            .expect("serialize");
        let Err(err) = Maest::from_gguf_with_policy(&unstamped, &CompliancePolicy::strict()) else {
            panic!("an unstamped artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for an Unknown weight class, got {err:?}"
        );

        // The gate must not mask an arch mismatch: a foreign artifact reports
        // the arch, which is the actionable fact.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "mert");
        b.add_tensor("mert.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let foreign = b.to_bytes().expect("serialize");
        let Err(err) = Maest::from_gguf_with_policy(&foreign, &CompliancePolicy::strict()) else {
            panic!("a foreign arch must be refused");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains("`mert`") && msg.contains("`maest`"),
                "arch mismatch must be reported ahead of any licence verdict: {msg}"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 15. The stamped axis group round-trips into MaestConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_round_trips_every_stamped_axis() {
        let want = upstream_config();
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");
        let got = m.config();

        // Whole-struct equality first, so a field added later without a
        // reader-side follow-through fails here.
        assert_eq!(*got, want, "every stamped axis must round-trip");

        // Then field-by-field against literals restating the upstream value, so
        // the test fails if the shared `upstream_config()` helper itself drifts.
        assert_eq!(got.hidden_size, 768);
        assert_eq!(got.num_hidden_layers, 12);
        assert_eq!(got.num_attention_heads, 12);
        assert_eq!(got.intermediate_size, 3072);
        assert_eq!(got.patch_size, 16);
        // The strides are SMALLER than the patch — MAEST's patches overlap, and
        // a reader assuming the usual non-overlapping ViT convention would
        // compute a ~3x smaller grid.
        assert_eq!(got.frequency_stride, 10);
        assert_eq!(got.time_stride, 10);
        // 96, not the 128 the general-audio AST / AudioSet lineage uses.
        assert_eq!(got.num_mel_bins, 96);
        assert_eq!(got.max_length, 1876);
        assert_eq!(got.num_labels, 400);
        assert!(got.qkv_bias);
        assert_eq!(got.hidden_act, "gelu");
        assert_eq!(got.layer_norm_eps_scaled_1e9, 1_000);
        assert_eq!(got.freq_patches, 9);
        assert_eq!(got.time_patches, 187);
        assert_eq!(got.num_patches, 1683);
        assert_eq!(got.num_prefix_tokens, 2);
        assert_eq!(got.sample_rate, 16_000);
        assert_eq!(got.n_fft, 512);
        assert_eq!(got.hop_length, 256);
        assert_eq!(got.win_length, 512);
        assert_eq!(got.window, "hann");
        assert_eq!(got.mel_scale, "slaney");
        assert_eq!(got.mel_norm, "slaney");
        assert_eq!(got.fmin_hz, 0);
        assert_eq!(got.fmax_hz, 8_000);
        assert_eq!(got.log_compression, "logC");
        assert_eq!(got.log_compression_mul, 10_000);
        assert!(got.do_normalize);

        // The normalization statistics are stamped FLOAT64 precisely so they do
        // NOT lose precision; assert exact equality rather than a tolerance.
        assert_eq!(got.norm_mean, 2.067_556_860_985_54);
        assert_eq!(got.norm_std, 1.268_292_820_667_291);

        // Derived views.
        assert_eq!(got.layer_norm_eps(), 1.0e-6);
        assert_eq!(got.mlp_ratio(), 4.0);
        assert_eq!(got.encoder_sequence_len(), 1685);

        // The patch grid the stamped axes imply is the one upstream's
        // `get_shape` computes: (96-16)/10+1 = 9, (1876-16)/10+1 = 187.
        assert_eq!(
            got.freq_patches * got.time_patches,
            got.num_patches,
            "the stamped grid must close on the stamped patch count"
        );
    }

    // -----------------------------------------------------------------------
    // 16. A missing stamped key is loud and names itself
    // -----------------------------------------------------------------------

    #[test]
    fn missing_axis_key_is_loud_and_names_the_key() {
        // Rebuild the axis group with exactly one key withheld, for EVERY key
        // in the group — so a future key added to the writer without a
        // reader-side `req_*` call cannot slip through.
        let cfg = upstream_config();
        for withheld in [
            GGUF_KEY_HIDDEN_SIZE,
            GGUF_KEY_NUM_HIDDEN_LAYERS,
            GGUF_KEY_NUM_ATTENTION_HEADS,
            GGUF_KEY_INTERMEDIATE_SIZE,
            GGUF_KEY_PATCH_SIZE,
            GGUF_KEY_FREQUENCY_STRIDE,
            GGUF_KEY_TIME_STRIDE,
            GGUF_KEY_NUM_MEL_BINS,
            GGUF_KEY_MAX_LENGTH,
            GGUF_KEY_NUM_LABELS,
            GGUF_KEY_QKV_BIAS,
            GGUF_KEY_HIDDEN_ACT,
            GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9,
            GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3,
            GGUF_KEY_ATTENTION_DROPOUT_SCALED_1E3,
            GGUF_KEY_FREQ_PATCHES,
            GGUF_KEY_TIME_PATCHES,
            GGUF_KEY_NUM_PATCHES,
            GGUF_KEY_NUM_PREFIX_TOKENS,
            GGUF_KEY_SAMPLE_RATE,
            GGUF_KEY_N_FFT,
            GGUF_KEY_HOP_LENGTH,
            GGUF_KEY_WIN_LENGTH,
            GGUF_KEY_WINDOW,
            GGUF_KEY_MEL_SCALE,
            GGUF_KEY_MEL_NORM,
            GGUF_KEY_FMIN_HZ,
            GGUF_KEY_FMAX_HZ,
            GGUF_KEY_LOG_COMPRESSION,
            GGUF_KEY_LOG_COMPRESSION_MUL,
            GGUF_KEY_DO_NORMALIZE,
            GGUF_KEY_NORM_MEAN,
            GGUF_KEY_NORM_STD,
        ] {
            let mut b = GgufBuilder::new();
            b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
            stamp_axis_group_except(&mut b, &cfg, withheld);
            b.add_tensor("probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
                .expect("add_tensor");
            let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");

            let Err(err) = Maest::from_gguf(&file) else {
                panic!("expected ModelLoad when `{withheld}` is withheld");
            };
            match err {
                VokraError::ModelLoad(msg) => {
                    assert!(
                        msg.contains(withheld),
                        "message must name the missing key `{withheld}`, got `{msg}`"
                    );
                    assert!(
                        msg.contains("FR-EX-08"),
                        "message must cite the no-fallback clause for `{withheld}`: {msg}"
                    );
                }
                other => panic!("expected VokraError::ModelLoad for `{withheld}`, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // 17. The config maps onto ViTAttrs, and that ViTAttrs validates
    // -----------------------------------------------------------------------

    #[test]
    fn config_maps_onto_validating_vit_attrs() {
        let cfg = upstream_config();
        let attrs = cfg
            .vit_attrs()
            .expect("the stamped axes must map onto ViTAttrs");
        attrs.validate().expect("the mapped ViTAttrs must validate");

        assert_eq!(attrs.embed_dim, 768);
        assert_eq!(attrs.depth, 12);
        assert_eq!(attrs.n_heads, 12);
        assert_eq!(attrs.head_dim(), 64);
        // The ratio is a re-encoding of two stamped integers, so what matters
        // is that it lands back on the stamped intermediate width.
        assert_eq!(attrs.mlp_dim(), cfg.intermediate_size as usize);
        assert_eq!(attrs.patch_h, 16);
        assert_eq!(attrs.patch_w, 16);
        // Upstream transposes the plane to [num_mel_bins, max_length] BEFORE the
        // convolution, so `frequency_stride` is the mel-axis stride — which is
        // `vokra-ops`' `stride_h`. Swapping these is silently wrong whenever the
        // two happen to differ, so pin the assignment.
        assert_eq!(attrs.stride_h, cfg.frequency_stride as usize);
        assert_eq!(attrs.stride_w, cfg.time_stride as usize);
        assert_eq!(attrs.n_prepended_tokens, 2);
        assert_eq!(attrs.layer_norm_eps, 1.0e-6);
        // `hidden_act: "gelu"` is upstream's EXACT erf GELU, not the tanh
        // approximation (which carries distinct upstream keys).
        assert_eq!(attrs.gelu, GeluKind::Erf);
        assert_eq!(attrs.pos_embed_policy, PosEmbedPolicy::RequireExact);

        // The patch grid the primitive computes from the stamped plane size must
        // agree with the stamped grid — an independent check on the mapping.
        let grid =
            vokra_ops::vit::patch_grid(cfg.num_mel_bins as usize, cfg.max_length as usize, &attrs)
                .expect("the stamped plane must produce a grid");
        assert_eq!(grid.grid_h, cfg.freq_patches as usize);
        assert_eq!(grid.grid_w, cfg.time_patches as usize);
        assert_eq!(grid.n_patches, cfg.num_patches as usize);
        assert_eq!(grid.n_tokens(attrs.n_prepended_tokens), 1685);

        // The interpolating policy is built from the stamped grid, not invented.
        assert_eq!(
            cfg.stamped_grid_pos_embed_policy(),
            PosEmbedPolicy::InterpolateGridBilinear {
                table_grid_h: 9,
                table_grid_w: 187,
            }
        );
    }

    #[test]
    fn unsupported_hidden_act_is_refused_rather_than_folded_onto_erf() {
        // `gelu_pytorch_tanh` is a REAL upstream activation key naming the tanh
        // approximation. Folding it onto the erf form would stay shape-valid
        // while differing by ~1e-3 — exactly the silent wrongness FR-EX-08
        // forbids.
        let mut cfg = upstream_config();
        cfg.hidden_act = "gelu_pytorch_tanh".to_owned();

        let Err(err) = cfg.gelu_kind() else {
            panic!("a tanh-family activation must be refused");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("gelu_pytorch_tanh"),
                    "must name the activation it found: {msg}"
                );
                assert!(
                    msg.contains(GGUF_KEY_HIDDEN_ACT),
                    "must name the key it read: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
        // And it must propagate rather than being swallowed by the mapping.
        assert!(
            cfg.vit_attrs().is_err(),
            "vit_attrs must propagate the refusal"
        );
    }

    // -----------------------------------------------------------------------
    // 18. The transcribed manifest binds, and the forward runs
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_binds_the_transcribed_manifest_under_either_prefix() {
        let cfg = tiny_config();
        for prefix in [TENSOR_PREFIX_CLASSIFICATION, TENSOR_PREFIX_BARE] {
            let file = synthetic_weights_gguf(&cfg, prefix, false);
            let m = Maest::from_gguf(&file).expect("bind");
            let enc = m
                .encoder(&file, PosEmbedPolicy::RequireExact)
                .expect("a complete synthetic weight set must bind");

            assert_eq!(
                enc.tensor_prefix(),
                prefix,
                "the prefix must be DISCOVERED from disk, not assumed"
            );
            assert_eq!(*enc.config(), cfg);
            // A bare-encoder export has no head, and that is not an error.
            assert!(enc.tag_head().is_none());
        }
    }

    #[test]
    fn missing_encoder_tensor_is_loud_and_names_itself() {
        // An artifact whose stamps claim more layers than the payload carries.
        // Layer 1's tensors are therefore required and absent, and the binder
        // must name one rather than substituting zeros.
        let cfg = tiny_config();
        let one_layer = synthetic_weights_gguf(&cfg, TENSOR_PREFIX_CLASSIFICATION, false);
        let mut deeper = cfg.clone();
        deeper.num_hidden_layers = 2;
        let file = restamped(&one_layer, &deeper);

        let m = Maest::from_gguf(&file).expect("bind");
        let Err(err) = m.encoder(&file, PosEmbedPolicy::RequireExact) else {
            panic!("a manifest missing layer 1 must not bind");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("encoder.layer.1."),
                    "message must name the missing layer-1 tensor: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-zero-substitution clause: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn encode_mel_produces_finite_output_of_the_expected_shape_and_is_deterministic() {
        // NOTE: this asserts SHAPE, FINITENESS and DETERMINISM only. There is
        // no upstream reference to compare against here — the weights are gated
        // CC-BY-NC-SA 4.0 and no fixture exists — so asserting a specific
        // numeric value would be fabrication dressed as verification.
        let cfg = tiny_config();
        let file = synthetic_weights_gguf(&cfg, TENSOR_PREFIX_CLASSIFICATION, true);
        let m = Maest::from_gguf(&file).expect("bind");
        let enc = m
            .encoder(&file, PosEmbedPolicy::RequireExact)
            .expect("bind weights");

        let n_frames = cfg.max_length as usize;
        let n_mels = cfg.num_mel_bins as usize;
        let mel = mel_plane(&cfg, n_frames);

        let tokens = enc
            .encode_mel(&mel, n_mels, n_frames)
            .expect("the forward must run over a stamped-size plane");

        // One row per token: 2 prefix + 12 patch = 14, each `hidden_size` wide.
        assert_eq!(tokens.len(), cfg.encoder_sequence_len());
        assert_eq!(tokens.len(), 14);
        for row in &tokens {
            assert_eq!(row.len(), cfg.hidden_size as usize);
            for v in row {
                assert!(v.is_finite(), "every hidden state must be finite, got {v}");
            }
        }

        // Deterministic: the same plane through the same weights twice.
        let again = enc.encode_mel(&mel, n_mels, n_frames).expect("second run");
        assert_eq!(tokens, again, "the forward must be deterministic");

        // The grid is reported, because it cannot be recovered from the token
        // count alone.
        let grid = enc.patch_grid(n_mels, n_frames).expect("grid");
        assert_eq!(grid.grid_h, cfg.freq_patches as usize);
        assert_eq!(grid.grid_w, cfg.time_patches as usize);

        // Pooling is the DeiT rule — mean of the CLS and distillation tokens —
        // so it must equal that mean computed from the token rows, and must NOT
        // equal either token alone.
        let pooled = enc.embed_mel(&mel, n_mels, n_frames).expect("embed");
        assert_eq!(pooled.len(), cfg.hidden_size as usize);
        for (i, v) in pooled.iter().enumerate() {
            assert!(v.is_finite(), "pooled element {i} must be finite, got {v}");
            let want = (tokens[0][i] + tokens[1][i]) * 0.5;
            assert_eq!(*v, want, "pooling must be mean(CLS, distillation)");
        }

        // Tag logits: one per stamped label, finite and deterministic.
        let logits = enc.tag_mel(&mel, n_mels, n_frames).expect("tag");
        assert_eq!(logits.len(), cfg.num_labels as usize);
        for (i, v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit {i} must be finite, got {v}");
        }
        assert_eq!(
            logits,
            enc.tag_mel(&mel, n_mels, n_frames).expect("tag again"),
            "the head must be deterministic"
        );
    }

    #[test]
    fn encode_mel_refuses_a_transposed_plane() {
        // The plane must be [num_mel_bins, n_frames]. Upstream's extractor emits
        // the transpose and `ASTPatchEmbeddings.forward` flips it, so a caller
        // handing over the untransposed plane is a real mistake — and one that
        // would be shape-plausible rather than loud if the extent were not
        // checked against the stamped band count.
        let cfg = tiny_config();
        let file = synthetic_weights_gguf(&cfg, TENSOR_PREFIX_CLASSIFICATION, false);
        let m = Maest::from_gguf(&file).expect("bind");
        let enc = m
            .encoder(&file, PosEmbedPolicy::RequireExact)
            .expect("bind weights");

        let n_frames = cfg.max_length as usize;
        let n_mels = cfg.num_mel_bins as usize;
        let mel = mel_plane(&cfg, n_frames);

        // Swap the two extents: same buffer length, wrong interpretation.
        let Err(err) = enc.encode_mel(&mel, n_frames, n_mels) else {
            panic!("a transposed plane must be refused");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains(GGUF_KEY_NUM_MEL_BINS),
                    "must name the stamped band-count key: {msg}"
                );
                assert!(
                    msg.contains("num_mel_bins, n_frames"),
                    "must state the required orientation: {msg}"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn a_shorter_clip_needs_the_interpolating_policy() {
        // The position table on disk is sized for the trained frame count, so a
        // shorter clip produces fewer patch tokens than the table has rows.
        // Under RequireExact that is loud; the interpolating policy built from
        // the stamped grid handles it.
        let cfg = tiny_config();
        let file = synthetic_weights_gguf(&cfg, TENSOR_PREFIX_CLASSIFICATION, false);
        let m = Maest::from_gguf(&file).expect("bind");
        let n_mels = cfg.num_mel_bins as usize;
        let short_frames = cfg.max_length as usize - 1;
        let mel = mel_plane(&cfg, short_frames);

        let strict = m
            .encoder(&file, PosEmbedPolicy::RequireExact)
            .expect("bind weights");
        let Err(err) = strict.encode_mel(&mel, n_mels, short_frames) else {
            panic!("RequireExact must refuse a table/token-count mismatch");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );

        let lenient = m
            .encoder(&file, cfg.stamped_grid_pos_embed_policy())
            .expect("bind weights");
        let tokens = lenient
            .encode_mel(&mel, n_mels, short_frames)
            .expect("the interpolating policy must accept a shorter clip");
        let grid = lenient.patch_grid(n_mels, short_frames).expect("grid");
        assert_eq!(tokens.len(), grid.n_tokens(cfg.num_prefix_tokens as usize));
        for row in &tokens {
            assert!(row.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn tag_mel_refuses_a_bare_encoder_artifact() {
        // No head on disk: the encoder still runs, but the tag surface must say
        // so rather than returning empty or zero-filled logits.
        let cfg = tiny_config();
        let file = synthetic_weights_gguf(&cfg, TENSOR_PREFIX_CLASSIFICATION, false);
        let m = Maest::from_gguf(&file).expect("bind");
        let enc = m
            .encoder(&file, PosEmbedPolicy::RequireExact)
            .expect("bind weights");

        let n_frames = cfg.max_length as usize;
        let n_mels = cfg.num_mel_bins as usize;
        let mel = mel_plane(&cfg, n_frames);

        // The encoder half works.
        assert!(enc.encode_mel(&mel, n_mels, n_frames).is_ok());

        let Err(err) = enc.tag_mel(&mel, n_mels, n_frames) else {
            panic!("a bare-encoder artifact must refuse the tag surface");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains(TAG_HEAD_PREFIX),
                    "must name the prefix it looked under: {msg}"
                );
                assert!(
                    msg.contains("MaestEncoder::encode_mel"),
                    "must point at the surfaces that DO work: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-fabricated-output clause: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn head_label_count_must_agree_with_the_stamped_count() {
        // The stamp comes from `config.json` and the projection comes from the
        // payload: two independent witnesses. A disagreement means the two
        // describe different checkpoints, so binding must refuse.
        let cfg = tiny_config();
        let mut lying = cfg.clone();
        lying.num_labels = cfg.num_labels + 1;

        // Weights carry `cfg.num_labels`; the stamps claim one more.
        let honest = synthetic_weights_gguf(&cfg, TENSOR_PREFIX_CLASSIFICATION, true);
        let file = restamped(&honest, &lying);

        let m = Maest::from_gguf(&file).expect("bind");
        // The disk still reports the honest count.
        assert_eq!(m.label_count(), Some(cfg.num_labels as usize));

        let Err(err) = m.encoder(&file, PosEmbedPolicy::RequireExact) else {
            panic!("a stamped/payload label-count disagreement must not bind");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("classifier.dense.weight"),
                    "must name the head projection: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-silent-reshape clause: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn a_non_ast_prefix_is_loud_and_names_both_candidates() {
        // A manifest with tensors but no position table under either known
        // prefix: refuse rather than invent a third spelling.
        let file = maest_gguf(
            Some(LicenseClass::NonCommercialShareAlike),
            false,
            true,
            None,
        );
        let m = Maest::from_gguf(&file).expect("bind");
        let Err(err) = m.weights().detect_tensor_prefix() else {
            panic!("expected ModelLoad when no position table is on disk");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(TENSOR_PREFIX_CLASSIFICATION),
                    "must name the ASTForAudioClassification candidate: {msg}"
                );
                assert!(
                    msg.contains(PROBE_SUFFIX_POSITION_EMBEDDINGS),
                    "must name the probe it used: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "must cite the no-guessing clause: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }
}
