//! **ATST** — "Audio Teacher-Student Transformer" (`Audio-WestlakeU/audiossl`,
//! code MIT / **weight CC-BY-4.0**) — runtime binder for the `atst` converter
//! arch (Wave C2 2026-08-15, loud-partial per the `emotion2vec` / `wavlm` /
//! `panns` / `redimnet` / `storm` precedent — CLAUDE.md 教訓 (a):
//! 「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/atst.rs` has been stamping
//! `vokra.model.arch = "atst"` since the 2026-08-13 SSL-encoder wave, but a
//! workspace-wide grep proved that **nothing read that arch string back** — a
//! converted ATST checkpoint was unloadable. This module is that consumer.
//!
//! # Primary sources
//!
//! Every fact below is transcribed from the converter's own module docstring
//! (`crates/vokra-convert/src/models/atst.rs`), which is this repository's
//! primary-source record for ATST. Nothing here is re-derived from memory.
//!
//! - Upstream tree: <https://github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst>
//! - Paper (utterance-level, this [`NAME`]): Li et al. 2022, INTERSPEECH,
//!   *"ATST: Audio Representation Learning with Teacher-Student Transformer"*
//!   — <https://arxiv.org/abs/2204.12076>
//! - Paper (frame-level `atst-frame` sibling): Li et al. 2023, TASLP —
//!   <https://arxiv.org/abs/2306.04186>
//! - Licence split (upstream `LICENSE` file text, quoted verbatim by the
//!   converter): *"The pretrained checkpoints hyper-linked in this repo are
//!   licensed under CC BY 4.0. … audiossl is licenced under MIT Licence."*
//!   Vokra stamps the **weight** tier, so [`DEFAULT_LICENSE_SPDX`] is
//!   `cc-by-4.0` → [`LicenseClass::AttributionRequired`].
//!
//! # What ATST is (and what this binder therefore exposes)
//!
//! ATST is a **feature extractor**, not an end-task model: it is a
//! self-supervised audio encoder trained with a BYOL-style EMA
//! teacher + student-patchout objective over a log-mel spectrogram, and it
//! maps audio to a sequence of hidden states (plus, for the utterance-level
//! 2022 variant this [`NAME`] tracks, a pooled utterance embedding).
//!
//! **The checkpoint carries no task head.** Downstream sound-event-detection
//! / audio-tagging / speaker heads are trained separately by consumers and are
//! not part of the ATST release, so this module deliberately exposes no
//! classifier and invents no class-label list.
//!
//! ```text
//! PCM (mono f32, 16 kHz — `sample_rate`)
//!   -> log-mel spectrogram front-end                    ← axes now stamped
//!        (n_fft / hop / win_length / n_mels / f_min / f_max / top_db and the
//!         min-max normalisation constants all ride [`AtstConfig`]).
//!   -> 2-D patch embedding over the mel plane           ← primitive exists
//!        ([`vokra_ops::vit::vit_patch_embed`]; the 64x4 patch and the grid it
//!         tiles ride [`AtstConfig`]).
//!   -> pre-norm Transformer encoder (~86M-param base)   ← primitive exists
//!        ([`vokra_ops::vit::ViTEncoder`]; [`AtstConfig::to_vit_attrs`] maps
//!         the stamped axes onto [`vokra_ops::vit::ViTAttrs`]).
//!   -> per-patch hidden states     ── [`Atst::encode`]  ← **loud-partial**
//!   -> pooled utterance embedding  ── [`Atst::embed`]   ← **loud-partial**
//!        (both still blocked on the branch choice and the tensor names).
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! **Real (this WP)** — everything up to and including the axis set:
//!
//! - [`Atst::from_gguf`] with **strict** `vokra.model.arch == "atst"`
//!   verification. A sibling SSL-encoder GGUF handed here by mistake fails
//!   with a message naming **both** tags and enumerating the whole
//!   audio/music-embedding neighbourhood (FR-EX-08 — never a silent
//!   misroute into a foreign topology).
//! - [`AtstConfig::from_gguf`] reads the converter's topology group. Every
//!   stamped key is **required**: a missing one is a loud
//!   [`VokraError::ModelLoad`] naming it, with no primary-source constant
//!   fallback (the `wavlm` posture — a silent default would let a mismatched
//!   artifact through).
//! - [`AtstConfig::to_vit_attrs`] maps that config onto
//!   [`vokra_ops::vit::ViTAttrs`], cross-checking the independently stamped
//!   axes against each other (patch grid against `spec_h / patch_h`,
//!   `pos_embed_len` against `num_patches + use_cls`, `mlp_hidden_dim`
//!   against the primitive's own rounding rule) so an internally inconsistent
//!   artifact fails loudly instead of forwarding.
//! - [`AtstConfig::vit_tensor_shapes`] derives the exact dims every ViT
//!   weight must carry, and [`Atst::verify_vit_tensor_shapes`] walks a
//!   **caller-supplied** [`AtstVitTensorNames`] through
//!   [`AtstWeights::require_tensor_dims`].
//! - [`AtstWeights::from_gguf`] tensor-manifest binding over the verbatim
//!   upstream `state_dict` names the converter passes through, with a
//!   non-empty gate plus [`AtstWeights::require_tensor`] /
//!   [`AtstWeights::require_tensor_dims`] lookups that name the missing
//!   tensor, or **both** the expected and the actual dims.
//! - Metadata surfacing: [`Atst::name`] / [`Atst::category`] /
//!   [`Atst::upstream_url`] read back the converter's stamps.
//! - Weight-licence + FR-MD-09 attribution surfacing, fail-closing to
//!   [`LicenseClass::Unknown`] when the artifact carries no stamp, and the
//!   compliance-gated [`Atst::from_gguf_with_policy`] / [`Atst::from_path`]
//!   entry points.
//!
//! **Loud-partial (this WP)** — [`Atst::encode`] and [`Atst::embed`] return
//! [`VokraError::UnsupportedOp`] naming the **two** blockers that remain. Both
//! are facts about a real checkpoint that no file in this repository records,
//! and both are stated as still-open by the converter's own
//! "Deliberate omissions" block:
//!
//! 1. **Teacher/student branch selection is unresolved.** A BYOL-style EMA
//!    checkpoint carries **both** branches. Picking the wrong one yields a
//!    shape-valid but numerically different embedding — a silent misroute of
//!    exactly the kind FR-EX-08 forbids — so the branch that upstream's own
//!    inference entry point uses must be read off the upstream tree before a
//!    forward may run. [`AtstBranch`] + [`Atst::branch_tensor_count`] exist
//!    today only as *diagnostics* over what is actually on disk; they gate
//!    nothing.
//! 2. **No verified tensor-name manifest.** The converter copies every float
//!    tensor under its verbatim upstream `state_dict` name, and nothing
//!    in-repo transcribes ATST's naming. The converter records that the real
//!    key chain runs `ATSTLightningModule.model` -> `ATST.student` /
//!    `.teacher` -> `MultiCropWrapper` -> the `AST` encoder, so the prefix is
//!    at least `model.student.` and **not** the bare `student.` this module's
//!    fixtures use — but no checkpoint key listing has been read, so the
//!    exact strings are unknown. Walking guessed names into typed slots would
//!    bind the wrong tensors without failing, which is why
//!    [`AtstVitTensorNames`] deliberately has **no** `Default` and no
//!    `atst_base()` constructor: the caller must supply names it can defend,
//!    exactly as [`vokra_ops::vit::ViTAttrs`] refuses to default its axes.
//!
//! # What changed on 2026-08-15 (two blockers closed)
//!
//! This module previously named **four** blockers. Two of them are now facts
//! about the world rather than about this repository, and a stale claim in an
//! error message actively misleads whoever reads it next:
//!
//! - *"No `vokra.atst.*` axis chunk group."* **Resolved.**
//!   `crates/vokra-convert/src/models/atst.rs` now stamps the full topology
//!   group — Transformer width / depth / heads, the patch grid and position
//!   table, and the whole log-mel front-end — each value transcribed from the
//!   upstream source tree with its file and line recorded. [`AtstConfig`] is
//!   the consumer.
//! - *"No ViT-style encoder primitive in `vokra-ops`."* **Resolved.**
//!   [`vokra_ops::vit`] landed with 2-D patch embedding over a mel plane,
//!   learned prepended tokens, an additive positional table, a pre-norm
//!   Transformer stack, a final norm and pooling.
//!
//! No fabricated hidden states or embeddings are ever emitted (FR-EX-08 — no
//! silent partial output). A follow-up wave flips the remaining switch by
//! reading a real checkpoint's key listing: that single act resolves both
//! remaining blockers at once, since the branch prefix and the tensor names
//! are the same listing.
//!
//! # Sibling family distinctness (SSL audio/music-embedding neighbourhood)
//!
//! [`ARCH`] = `"atst"` is deliberately distinct from every sibling. All of
//! these are self-supervised audio encoders that emit hidden states, and all
//! of them differ in the pre-training objective that shapes the topology:
//!
//! - `beats` — iterative acoustic tokenizer + masked acoustic modelling;
//! - `eat` — utterance-level MAE with efficient inverse block masking;
//! - `dasheng` — universal MAE;
//! - `m2d` — masked modelling **duo** (dual online + target branch);
//! - `maest` — AST backbone with a Discogs music-tagger SSL objective;
//! - `mert` — HuBERT-derived masked prediction (music);
//! - `muq` — Mel-RVQ + BEATs teacher (music);
//! - `yamnet` — supervised audio-tagging CNN, not SSL at all;
//! - `hubert` / `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec` — the
//!   wav2vec2 lineage, whose encoders sit on a **raw-waveform 1-D conv
//!   stem** rather than a log-mel patch grid.
//!
//! Sharing an arch tag would let runtime dispatch bind, say, an MAE decoder
//! or a raw-waveform conv stem over a teacher-student log-mel checkpoint
//! (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_URL`] /
//! [`DEFAULT_LICENSE_SPDX`] are **mirrors of the converter's constants** —
//! the same rule every sibling binder (`emotion2vec` / `wavlm` / `panns` /
//! `redimnet` / `canary_1b_flash`) follows so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # Licence posture
//!
//! The converter stamps `cc-by-4.0` → [`LicenseClass::AttributionRequired`],
//! which is commercially permitted, so a correctly stamped artifact loads
//! under [`CompliancePolicy::strict`] without a research opt-in — but CC-BY
//! 4.0 obliges a downstream to **display attribution**, so
//! [`Atst::attribution`] surfaces the stamped text rather than burying it.
//! An unstamped artifact resolves to [`LicenseClass::Unknown`] and is refused
//! by the gate (fail-closed, never a silent substitution). This binder only
//! *surfaces* whatever class the artifact carries;
//! `docs/license-audit.md` §3.1 sign-off stays **blank** (owner-only per
//! memory `[[feedback-license-signoff-primary-source]]` — CC does not sign,
//! and does not treat a converter default as a sign-off).
//!
//! # No ONNX / no pickle (permanent)
//!
//! ATST ships upstream as a PyTorch `.ckpt` pickle; this runtime **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02). The bridge is an offline
//! uv-managed Python 3.12 sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), exactly as the converter docstring specifies.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};
use vokra_ops::vit::{GeluKind, PosEmbedPolicy, ViTAttrs};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/atst.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model atst-base`.
///
/// Distinct from every sibling SSL audio/music-embedding arch tag (`beats` /
/// `eat` / `dasheng` / `m2d` / `maest` / `mert` / `muq` / `yamnet`) and from
/// the wav2vec2 lineage (`hubert` / `wav2vec2_ctc` / `wavlm_sv` /
/// `emotion2vec`): ATST's BYOL-style teacher-student patchout objective over
/// a log-mel patch grid is a distinct topology axis from all of them, and
/// silently sharing a tag would misroute runtime dispatch (FR-EX-08 — see the
/// module docstring "Sibling family distinctness" section).
pub const ARCH: &str = "atst";

/// Expected `vokra.model.name` value written by the converter — the canonical
/// `atst-base` size point (the utterance-level INTERSPEECH 2022 release).
///
/// The frame-level TASLP 2023 sibling `atst-frame` is published by the
/// converter under its own `NAME` following the `snac_24khz` / `snac_44khz`
/// pattern, so this value is **surfaced, not gated** — see [`Atst::name`].
pub const NAME: &str = "atst-base";

/// Expected `vokra.model.category` value — `audio-embedding`, shared with the
/// sibling general-audio SSL encoders (`beats` / `eat` / `dasheng` / `m2d`).
/// Consumed by the model-card generator and the zoo-manifest tier gate so a
/// feature extractor is never advertised as an ASR / TTS release.
pub const CATEGORY: &str = "audio-embedding";

/// Upstream source tree. ATST is **not** hosted on HuggingFace, so the
/// converter stamps [`GGUF_KEY_PROVENANCE_UPSTREAM_URL`] rather than
/// `vokra.provenance.upstream_hf` (the sibling `beats` / `eat` / `nsnet2`
/// posture); the model-card generator picks up either.
pub const UPSTREAM_URL: &str =
    "github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst";

/// Default SPDX stamped by the converter — the **weight** tier.
///
/// The upstream `LICENSE` file separates code (`mit`) from pretrained
/// checkpoints (`cc-by-4.0`); `vokra.provenance.weight_license` tracks the
/// weight, so this resolves to [`LicenseClass::AttributionRequired`]. A
/// caller with a different attestation may override at the converter boundary
/// (`--license <spdx>`), which is why this binder *surfaces* rather than
/// *asserts* the class.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-4.0";

/// Metadata key holding [`CATEGORY`] (not part of `vokra_core::gguf::chunks`,
/// so mirrored here from the converter).
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key holding [`UPSTREAM_URL`] (not part of
/// `vokra_core::gguf::chunks`, so mirrored here from the converter).
pub const GGUF_KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// Primary-source anchors, cited inside the loud-partial error so a reader
// diagnosing the gap has fully specified places to walk.

/// Primary-source anchor: the upstream ATST source tree.
pub const PRIMARY_SOURCE_UPSTREAM: &str = UPSTREAM_URL;
/// Primary-source anchor: Li et al. 2022 (INTERSPEECH) — the utterance-level
/// ATST this [`NAME`] tracks.
pub const PRIMARY_SOURCE_PAPER_2022: &str = "arxiv.org/abs/2204.12076";
/// Primary-source anchor: Li et al. 2023 (TASLP) — the frame-level
/// `atst-frame` extension.
pub const PRIMARY_SOURCE_PAPER_2023: &str = "arxiv.org/abs/2306.04186";

// ---------------------------------------------------------------------------
// `vokra.atst.*` chunk keys — byte-identical mirrors of the converter's
// `KEY_ATST_*` constants (`crates/vokra-convert/src/models/atst.rs`). The
// converter is the writer, this module is the reader; a spelling drift on
// either side turns every conversion into a missing-key load failure, which
// is why both sides carry the literal and the tests pin them.
// ---------------------------------------------------------------------------

/// `vokra.atst.embed_dim` — Transformer width `D`.
pub const GGUF_KEY_EMBED_DIM: &str = "vokra.atst.embed_dim";
/// `vokra.atst.depth` — stacked pre-norm block count.
pub const GGUF_KEY_DEPTH: &str = "vokra.atst.depth";
/// `vokra.atst.num_heads` — attention head count.
pub const GGUF_KEY_NUM_HEADS: &str = "vokra.atst.num_heads";
/// `vokra.atst.mlp_ratio_scaled_1e3` — FFN expansion ratio times 1000.
pub const GGUF_KEY_MLP_RATIO_SCALED_1E3: &str = "vokra.atst.mlp_ratio_scaled_1e3";
/// `vokra.atst.mlp_hidden_dim` — resolved FFN hidden width.
pub const GGUF_KEY_MLP_HIDDEN_DIM: &str = "vokra.atst.mlp_hidden_dim";
/// `vokra.atst.layer_norm_eps` — LayerNorm epsilon (f32-typed chunk).
pub const GGUF_KEY_LAYER_NORM_EPS: &str = "vokra.atst.layer_norm_eps";
/// `vokra.atst.qkv_bias` — whether the fused QKV projection carries a bias.
pub const GGUF_KEY_QKV_BIAS: &str = "vokra.atst.qkv_bias";
/// `vokra.atst.use_cls` — whether a CLS token is prepended.
pub const GGUF_KEY_USE_CLS: &str = "vokra.atst.use_cls";
/// `vokra.atst.act_layer` — FFN activation name.
pub const GGUF_KEY_ACT_LAYER: &str = "vokra.atst.act_layer";
/// `vokra.atst.in_chans` — input channel count of the mel plane.
pub const GGUF_KEY_IN_CHANS: &str = "vokra.atst.in_chans";
/// `vokra.atst.num_classes` — classifier width (`0` = no task head ships).
pub const GGUF_KEY_NUM_CLASSES: &str = "vokra.atst.num_classes";
/// `vokra.atst.drop_path_rate_scaled_1e3` — stochastic depth times 1000.
pub const GGUF_KEY_DROP_PATH_RATE_SCALED_1E3: &str = "vokra.atst.drop_path_rate_scaled_1e3";
/// `vokra.atst.patch_h` — patch extent along the mel-bin axis.
pub const GGUF_KEY_PATCH_H: &str = "vokra.atst.patch_h";
/// `vokra.atst.patch_w` — patch extent along the frame axis.
pub const GGUF_KEY_PATCH_W: &str = "vokra.atst.patch_w";
/// `vokra.atst.spec_h` — mel-plane height the position table is built for.
pub const GGUF_KEY_SPEC_H: &str = "vokra.atst.spec_h";
/// `vokra.atst.spec_w` — mel-plane width the position table is built for.
pub const GGUF_KEY_SPEC_W: &str = "vokra.atst.spec_w";
/// `vokra.atst.num_patches` — patch count the position table is sized for.
pub const GGUF_KEY_NUM_PATCHES: &str = "vokra.atst.num_patches";
/// `vokra.atst.pos_embed_len` — position-table row count.
pub const GGUF_KEY_POS_EMBED_LEN: &str = "vokra.atst.pos_embed_len";
/// `vokra.atst.pos_type` — position-table resize policy.
pub const GGUF_KEY_POS_TYPE: &str = "vokra.atst.pos_type";
/// `vokra.atst.patch_embed_in_features` — patch-projection input width.
pub const GGUF_KEY_PATCH_EMBED_IN_FEATURES: &str = "vokra.atst.patch_embed_in_features";
/// `vokra.atst.patch_embed_kind` — how patches are embedded.
pub const GGUF_KEY_PATCH_EMBED_KIND: &str = "vokra.atst.patch_embed_kind";
/// `vokra.atst.patch_order` — patch flattening order over the grid.
pub const GGUF_KEY_PATCH_ORDER: &str = "vokra.atst.patch_order";
/// `vokra.atst.patch_grid` — axis-array prefix, stamped as `_0` / `_1`.
pub const GGUF_KEY_PATCH_GRID_PREFIX: &str = "vokra.atst.patch_grid";
/// `vokra.atst.sample_rate` — front-end sample rate in Hz.
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.atst.sample_rate";
/// `vokra.atst.n_fft` — front-end FFT size.
pub const GGUF_KEY_N_FFT: &str = "vokra.atst.n_fft";
/// `vokra.atst.hop_length` — front-end hop in samples.
pub const GGUF_KEY_HOP_LENGTH: &str = "vokra.atst.hop_length";
/// `vokra.atst.win_length` — front-end window length in samples.
pub const GGUF_KEY_WIN_LENGTH: &str = "vokra.atst.win_length";
/// `vokra.atst.n_mels` — mel band count.
pub const GGUF_KEY_N_MELS: &str = "vokra.atst.n_mels";
/// `vokra.atst.f_min` — mel low edge in Hz.
pub const GGUF_KEY_F_MIN: &str = "vokra.atst.f_min";
/// `vokra.atst.f_max` — mel high edge in Hz.
pub const GGUF_KEY_F_MAX: &str = "vokra.atst.f_max";
/// `vokra.atst.amp_to_db_top_db` — `AmplitudeToDB` dynamic-range clamp.
pub const GGUF_KEY_AMP_TO_DB_TOP_DB: &str = "vokra.atst.amp_to_db_top_db";
/// `vokra.atst.amp_to_db_stype` — `AmplitudeToDB` input scale.
pub const GGUF_KEY_AMP_TO_DB_STYPE: &str = "vokra.atst.amp_to_db_stype";
/// `vokra.atst.norm_min` — min-max normalisation floor in dB (f32 chunk).
pub const GGUF_KEY_NORM_MIN: &str = "vokra.atst.norm_min";
/// `vokra.atst.norm_max` — min-max normalisation ceiling in dB (f32 chunk).
pub const GGUF_KEY_NORM_MAX: &str = "vokra.atst.norm_max";

/// Length of the `vokra.atst.patch_grid_{i}` axis array — `[rows, cols]`.
pub const PATCH_GRID_AXIS_LEN: usize = 2;

/// The only [`AtstConfig::act_layer`] value this binder can map — upstream
/// `audiossl/modules/transformer.py` passes a bare `act_layer=nn.GELU`.
pub const ACT_LAYER_GELU: &str = "gelu";

/// The only [`AtstConfig::pos_type`] value this binder can map — see
/// [`AtstConfig::cut_pos_embed`] for what "cut" means and who performs it.
pub const POS_TYPE_CUT: &str = "cut";

/// The only [`AtstConfig::patch_embed_kind`] value this binder can map.
///
/// `PatchEmbed_v2` is a `Rearrange` followed by
/// `nn.Linear(patch_h * patch_w, embed_dim)`, **not** the `Conv2d` stem most
/// ViT ports assume — which is exactly why the converter stamps the axis.
pub const PATCH_EMBED_KIND_LINEAR: &str = "linear";

/// [`AtstConfig::patch_order`] value meaning **width-major, then height** —
/// upstream's `Rearrange('b c (h p1) (w p2) -> b (w h) (p1 p2 c)')`.
pub const PATCH_ORDER_WH: &str = "wh";

/// [`AtstConfig::patch_order`] value meaning **height-major, then width** —
/// the order [`vokra_ops::vit`] itself emits tokens in.
pub const PATCH_ORDER_HW: &str = "hw";

// ---------------------------------------------------------------------------
// AtstConfig — the topology axes read from the `vokra.atst.*` chunk group.
//
// STRICT: every stamped key is required (FR-EX-08). There is deliberately no
// primary-source constant fallback. The converter transcribed each of these
// from the upstream source tree and stamps all of them, so a partial group
// means the artifact was produced by a different (older, or foreign) writer —
// and silently substituting a constant would let that artifact bind a
// topology it does not actually carry. This is the `wavlm` posture
// (`crates/vokra-models/src/wavlm/mod.rs`), for the same reason.
// ---------------------------------------------------------------------------

/// ATST topology axes as they ride the `vokra.atst.*` chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every field below
/// corresponds to a key the converter stamps, and every one of them is
/// required. A GGUF missing any key is rejected with a
/// [`VokraError::ModelLoad`] naming that key.
///
/// Scaled-integer fields (`*_scaled_1e3`) carry a float as an integer so it
/// round-trips without serialization ambiguity; the accessors
/// [`mlp_ratio`](Self::mlp_ratio) / [`drop_path_rate`](Self::drop_path_rate)
/// undo the scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct AtstConfig {
    /// Transformer width `D` (`AST_base`: 768).
    pub embed_dim: u32,
    /// Stacked pre-norm Transformer block count (`AST_base`: 12).
    pub depth: u32,
    /// Attention head count (`AST_base`: 12).
    pub num_heads: u32,
    /// FFN expansion ratio scaled by 1000 (`mlp_ratio=4.` → 4000).
    pub mlp_ratio_scaled_1e3: u32,
    /// Resolved FFN hidden width, `int(embed_dim * mlp_ratio)` (768*4 = 3072).
    pub mlp_hidden_dim: u32,
    /// LayerNorm epsilon (`partial(nn.LayerNorm, eps=1e-6)`).
    pub layer_norm_eps: f32,
    /// Whether the fused QKV projection carries a bias (`AST_base`: false).
    pub qkv_bias: bool,
    /// Whether a CLS token is prepended (`AST_base`: true).
    pub use_cls: bool,
    /// FFN activation name — see [`ACT_LAYER_GELU`].
    pub act_layer: String,
    /// Input channel count of the mel plane (`in_chans`: 1).
    pub in_chans: u32,
    /// Classifier width. `0` is load-bearing: the release ships **no** task
    /// head, so a consumer must never look for a classifier.
    pub num_classes: u32,
    /// Stochastic-depth rate scaled by 1000 (training-time only).
    pub drop_path_rate_scaled_1e3: u32,
    /// Patch extent along the mel-bin axis (`AST_base`: 64).
    pub patch_h: u32,
    /// Patch extent along the frame axis (`AST_base`: 4).
    pub patch_w: u32,
    /// Mel-plane height the position table is built for (`spec_h`: 64).
    pub spec_h: u32,
    /// Mel-plane width the position table is built for (`spec_w`: 1001).
    ///
    /// A **maximum**, not a fixed input length — see [`Self::cut_pos_embed`].
    pub spec_w: u32,
    /// Patch count the position table is sized for (`num_patches`: 250).
    pub num_patches: u32,
    /// Position-table row count, `num_patches + use_cls` (251).
    pub pos_embed_len: u32,
    /// Position-table policy — see [`POS_TYPE_CUT`].
    pub pos_type: String,
    /// Patch-projection input width, `patch_h * patch_w` (256).
    pub patch_embed_in_features: u32,
    /// How patches are embedded — see [`PATCH_EMBED_KIND_LINEAR`].
    pub patch_embed_kind: String,
    /// Patch flattening order — see [`PATCH_ORDER_WH`] / [`PATCH_ORDER_HW`].
    pub patch_order: String,
    /// Patch grid as `[rows, cols]` (`AST_base`: `[1, 250]`).
    pub patch_grid: [u32; PATCH_GRID_AXIS_LEN],
    /// Front-end sample rate in Hz (16000, mono).
    pub sample_rate: u32,
    /// Front-end FFT size (1024).
    pub n_fft: u32,
    /// Front-end hop in samples (160 = 10 ms at 16 kHz).
    pub hop_length: u32,
    /// Front-end window length in samples (1024).
    pub win_length: u32,
    /// Mel band count (64) — must equal [`Self::spec_h`].
    pub n_mels: u32,
    /// Mel low edge in Hz (60).
    pub f_min: u32,
    /// Mel high edge in Hz (7800).
    pub f_max: u32,
    /// `AmplitudeToDB` dynamic-range clamp in dB (80).
    pub amp_to_db_top_db: u32,
    /// `AmplitudeToDB` input scale — `"power"`, not `"magnitude"` (the two
    /// differ by a factor of two in the dB conversion).
    pub amp_to_db_stype: String,
    /// Min-max normalisation floor in dB (-79.6482). A dataset-fitted
    /// constant: it must be reproduced exactly or every embedding shifts.
    pub norm_min: f32,
    /// Min-max normalisation ceiling in dB (50.6842). See [`Self::norm_min`].
    pub norm_max: f32,
}

impl AtstConfig {
    /// Reads every `vokra.atst.*` chunk from `gguf`.
    ///
    /// A missing key is a loud [`VokraError::ModelLoad`] naming it — there is
    /// no primary-source constant fallback, because the converter stamps the
    /// full group and a silent default would let a mismatched artifact bind a
    /// topology it does not carry (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any stamped key is absent, or is
    ///   present under a type this reader cannot decode (a bool stamped as an
    ///   integer, say, reads back as absent and is reported as such).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Ok(Self {
            embed_dim: req_u32(gguf, GGUF_KEY_EMBED_DIM)?,
            depth: req_u32(gguf, GGUF_KEY_DEPTH)?,
            num_heads: req_u32(gguf, GGUF_KEY_NUM_HEADS)?,
            mlp_ratio_scaled_1e3: req_u32(gguf, GGUF_KEY_MLP_RATIO_SCALED_1E3)?,
            mlp_hidden_dim: req_u32(gguf, GGUF_KEY_MLP_HIDDEN_DIM)?,
            layer_norm_eps: req_f32(gguf, GGUF_KEY_LAYER_NORM_EPS)?,
            qkv_bias: req_bool(gguf, GGUF_KEY_QKV_BIAS)?,
            use_cls: req_bool(gguf, GGUF_KEY_USE_CLS)?,
            act_layer: req_string(gguf, GGUF_KEY_ACT_LAYER)?,
            in_chans: req_u32(gguf, GGUF_KEY_IN_CHANS)?,
            num_classes: req_u32(gguf, GGUF_KEY_NUM_CLASSES)?,
            drop_path_rate_scaled_1e3: req_u32(gguf, GGUF_KEY_DROP_PATH_RATE_SCALED_1E3)?,
            patch_h: req_u32(gguf, GGUF_KEY_PATCH_H)?,
            patch_w: req_u32(gguf, GGUF_KEY_PATCH_W)?,
            spec_h: req_u32(gguf, GGUF_KEY_SPEC_H)?,
            spec_w: req_u32(gguf, GGUF_KEY_SPEC_W)?,
            num_patches: req_u32(gguf, GGUF_KEY_NUM_PATCHES)?,
            pos_embed_len: req_u32(gguf, GGUF_KEY_POS_EMBED_LEN)?,
            pos_type: req_string(gguf, GGUF_KEY_POS_TYPE)?,
            patch_embed_in_features: req_u32(gguf, GGUF_KEY_PATCH_EMBED_IN_FEATURES)?,
            patch_embed_kind: req_string(gguf, GGUF_KEY_PATCH_EMBED_KIND)?,
            patch_order: req_string(gguf, GGUF_KEY_PATCH_ORDER)?,
            patch_grid: req_u32_grid(gguf, GGUF_KEY_PATCH_GRID_PREFIX)?,
            sample_rate: req_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            n_fft: req_u32(gguf, GGUF_KEY_N_FFT)?,
            hop_length: req_u32(gguf, GGUF_KEY_HOP_LENGTH)?,
            win_length: req_u32(gguf, GGUF_KEY_WIN_LENGTH)?,
            n_mels: req_u32(gguf, GGUF_KEY_N_MELS)?,
            f_min: req_u32(gguf, GGUF_KEY_F_MIN)?,
            f_max: req_u32(gguf, GGUF_KEY_F_MAX)?,
            amp_to_db_top_db: req_u32(gguf, GGUF_KEY_AMP_TO_DB_TOP_DB)?,
            amp_to_db_stype: req_string(gguf, GGUF_KEY_AMP_TO_DB_STYPE)?,
            norm_min: req_f32(gguf, GGUF_KEY_NORM_MIN)?,
            norm_max: req_f32(gguf, GGUF_KEY_NORM_MAX)?,
        })
    }

    /// FFN expansion ratio, undoing the 1000x integer scaling.
    #[inline]
    #[must_use]
    pub fn mlp_ratio(&self) -> f32 {
        self.mlp_ratio_scaled_1e3 as f32 / 1000.0
    }

    /// Stochastic-depth rate, undoing the 1000x integer scaling.
    ///
    /// Training-time regularization only — inactive at inference, surfaced for
    /// audit completeness.
    #[inline]
    #[must_use]
    pub fn drop_path_rate(&self) -> f32 {
        self.drop_path_rate_scaled_1e3 as f32 / 1000.0
    }

    /// How many learned tokens sit ahead of the patch tokens: `1` when
    /// [`Self::use_cls`], else `0`.
    #[inline]
    #[must_use]
    pub const fn n_prepended_tokens(&self) -> usize {
        if self.use_cls { 1 } else { 0 }
    }

    /// Maps the stamped axes onto [`vokra_ops::vit::ViTAttrs`].
    ///
    /// # Where each ViT field comes from
    ///
    /// | `ViTAttrs` field | Source |
    /// |---|---|
    /// | `embed_dim` | [`Self::embed_dim`], stamped |
    /// | `depth` | [`Self::depth`], stamped |
    /// | `n_heads` | [`Self::num_heads`], stamped |
    /// | `mlp_ratio` | [`Self::mlp_ratio`], stamped (unscaled here) |
    /// | `patch_h` / `patch_w` | [`Self::patch_h`] / [`Self::patch_w`], stamped |
    /// | `stride_h` / `stride_w` | **derived** — equal to the patch extents |
    /// | `n_prepended_tokens` | **derived** from [`Self::use_cls`] |
    /// | `layer_norm_eps` | [`Self::layer_norm_eps`], stamped |
    /// | `gelu` | **derived** from [`Self::act_layer`] |
    /// | `pos_embed_policy` | **derived** from [`Self::pos_type`] |
    ///
    /// The three derivations, and why none of them is a guess:
    ///
    /// - **Stride.** Upstream's `PatchEmbed_v2` is a `Rearrange` that
    ///   *partitions* the plane (`'b c (h p1) (w p2) -> ...'`), and
    ///   `get_num_patches` is `(h // patch_h) * (w // patch_w)` — the
    ///   non-overlapping tiling rule, i.e. stride equals the patch extent.
    ///   Rather than take that on trust this method **checks it against the
    ///   independently stamped grid**: an overlapping variant would stamp a
    ///   `patch_grid` that the floor-division formula cannot reproduce, and
    ///   that disagreement is a loud error rather than a silent re-tiling.
    /// - **GELU flavour.** `act_layer=nn.GELU` is the *bare* constructor, and
    ///   [`vokra_ops::vit::GeluKind::Erf`] documents itself as "what a bare
    ///   `torch.nn.GELU()` computes". The tanh approximation would need an
    ///   explicit `approximate="tanh"`, which upstream does not pass.
    /// - **Positional policy.** `pos_type="cut"` means slice, never
    ///   interpolate, so neither of the primitive's resizing policies applies
    ///   directly. The binder performs the cut itself (see
    ///   [`Self::cut_pos_embed`]) and then asks the primitive for
    ///   [`PosEmbedPolicy::RequireExact`], which is exact by construction.
    ///   Choosing `InterpolateGridBilinear` here would silently substitute a
    ///   bilinear resample for upstream's slice.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when a stamped string names a variant this
    ///   binder cannot map (`act_layer`, `pos_type`, `patch_embed_kind`,
    ///   `patch_order`), when a stamped axis is zero, or when independently
    ///   stamped axes contradict each other. Every such case is an internally
    ///   inconsistent artifact, and forwarding it would produce shape-valid
    ///   numbers from a topology the checkpoint does not have (FR-EX-08).
    /// - [`VokraError::InvalidArgument`] propagated from
    ///   [`ViTAttrs::validate`].
    pub fn to_vit_attrs(&self) -> Result<ViTAttrs> {
        // --- String axes: map or refuse. A default here would silently pick a
        // --- different function than the one the checkpoint was trained with.
        if self.patch_embed_kind != PATCH_EMBED_KIND_LINEAR {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_PATCH_EMBED_KIND}` is `{kind}`, but this binder can only \
                 map `{PATCH_EMBED_KIND_LINEAR}` — upstream's `PatchEmbed_v2` is a \
                 `Rearrange` plus `nn.Linear(patch_h * patch_w, embed_dim)`. A `conv2d` \
                 stem would carry a 4-D weight and a different flattening, so binding it \
                 through the linear path would reshape the wrong tensor (FR-EX-08). \
                 Primary source: {UPSTREAM_URL}",
                kind = self.patch_embed_kind,
            )));
        }
        if self.act_layer != ACT_LAYER_GELU {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_ACT_LAYER}` is `{act}`, but this binder can only map \
                 `{ACT_LAYER_GELU}`. Substituting a different activation is silently \
                 wrong — outputs stay finite and correctly shaped (FR-EX-08). Primary \
                 source: {UPSTREAM_URL}",
                act = self.act_layer,
            )));
        }
        if self.pos_type != POS_TYPE_CUT {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_POS_TYPE}` is `{pos}`, but this binder can only map \
                 `{POS_TYPE_CUT}` (allocate the table for `spec_w` frames and slice it \
                 down, never interpolate). An interpolating policy would resample the \
                 positional table instead of slicing it, shifting every patch embedding \
                 (FR-EX-08). Primary source: {UPSTREAM_URL}",
                pos = self.pos_type,
            )));
        }

        // --- Zero guards before any division. `ViTAttrs::validate` catches
        // --- these too, but dividing first would panic before it ran.
        for (key, value) in [
            (GGUF_KEY_PATCH_H, self.patch_h),
            (GGUF_KEY_PATCH_W, self.patch_w),
            (GGUF_KEY_EMBED_DIM, self.embed_dim),
            (GGUF_KEY_DEPTH, self.depth),
            (GGUF_KEY_NUM_HEADS, self.num_heads),
        ] {
            if value == 0 {
                return Err(VokraError::ModelLoad(format!(
                    "atst: `{key}` is stamped 0, which cannot describe a real encoder. \
                     Refusing to bind a degenerate topology (FR-EX-08). Primary source: \
                     {UPSTREAM_URL}"
                )));
            }
        }

        // --- Stride derivation, checked against the independently stamped
        // --- grid rather than assumed (see the doc comment).
        let grid_h = self.patch_grid[0] as usize;
        let grid_w = self.patch_grid[1] as usize;
        let tiled_h = (self.spec_h / self.patch_h) as usize;
        let tiled_w = (self.spec_w / self.patch_w) as usize;
        if grid_h != tiled_h || grid_w != tiled_w {
            return Err(VokraError::ModelLoad(format!(
                "atst: stamped patch grid is [{grid_h}, {grid_w}] but the non-overlapping \
                 tiling of a {sh}x{sw} plane by a {ph}x{pw} patch gives [{tiled_h}, \
                 {tiled_w}]. Upstream's `PatchEmbed_v2` partitions the plane and \
                 `get_num_patches` is `(h // patch_h) * (w // patch_w)`, so the two must \
                 agree; a disagreement means this artifact uses overlapping patches, whose \
                 stride is NOT stamped anywhere. Refusing to guess a stride (FR-EX-08). \
                 Primary source: {UPSTREAM_URL}",
                sh = self.spec_h,
                sw = self.spec_w,
                ph = self.patch_h,
                pw = self.patch_w,
            )));
        }

        // --- Token ordering. The primitive emits tokens in grid row-major
        // --- (height-major) order. Upstream ATST is width-major ("wh"), which
        // --- coincides only while the grid is one row tall.
        if self.patch_order == PATCH_ORDER_WH {
            if grid_h != 1 {
                return Err(VokraError::ModelLoad(format!(
                    "atst: `{GGUF_KEY_PATCH_ORDER}` is `{PATCH_ORDER_WH}` (width-major, \
                     then height) with a {grid_h}-row patch grid. `vokra_ops::vit` emits \
                     tokens in grid row-major (height-major) order, so the two orders \
                     coincide only when the grid is one row tall. Binding this artifact \
                     would transpose the token sequence against its positional table — \
                     shape-valid and silently wrong (FR-EX-08). Primary source: \
                     {UPSTREAM_URL}"
                )));
            }
        } else if self.patch_order != PATCH_ORDER_HW {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_PATCH_ORDER}` is `{order}`, but this binder knows only \
                 `{PATCH_ORDER_WH}` (upstream's `Rearrange` order) and `{PATCH_ORDER_HW}` \
                 (the order `vokra_ops::vit` emits). Refusing to guess a token ordering \
                 (FR-EX-08). Primary source: {UPSTREAM_URL}",
                order = self.patch_order,
            )));
        }

        // --- Cross-checks between axes the converter stamped independently.
        // --- Each pair is derivable from the other upstream, so a mismatch is
        // --- a mis-produced artifact rather than a variant this binder should
        // --- try to accommodate.
        let n_prepended = self.n_prepended_tokens();
        let expected_patches = grid_h * grid_w;
        if self.num_patches as usize != expected_patches {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_NUM_PATCHES}` is {n} but the stamped grid [{grid_h}, \
                 {grid_w}] implies {expected_patches}. Refusing to bind contradictory axes \
                 (FR-EX-08). Primary source: {UPSTREAM_URL}",
                n = self.num_patches,
            )));
        }
        if self.pos_embed_len as usize != expected_patches + n_prepended {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_POS_EMBED_LEN}` is {len} but {expected_patches} patch \
                 token(s) plus {n_prepended} prepended token(s) implies {implied}. \
                 Upstream allocates `torch.zeros(1, num_patches + 1, embed_dim)` with the \
                 `+ 1` being the CLS slot, so these must agree (FR-EX-08). Primary source: \
                 {UPSTREAM_URL}",
                len = self.pos_embed_len,
                implied = expected_patches + n_prepended,
            )));
        }
        let patch_len = self.patch_h as usize * self.patch_w as usize;
        if self.patch_embed_in_features as usize != patch_len {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_PATCH_EMBED_IN_FEATURES}` is {got} but `patch_h * \
                 patch_w` is {patch_len}. The patch projection reads one flattened patch, \
                 so these must agree (FR-EX-08). Primary source: {UPSTREAM_URL}",
                got = self.patch_embed_in_features,
            )));
        }
        if self.spec_h != self.n_mels {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_SPEC_H}` is {sh} but `{GGUF_KEY_N_MELS}` is {nm}. The \
                 mel-plane height the encoder is built for must equal the band count the \
                 front-end produces, or the patch stem cannot tile the input (FR-EX-08). \
                 Primary source: {UPSTREAM_URL}",
                sh = self.spec_h,
                nm = self.n_mels,
            )));
        }

        let attrs = ViTAttrs {
            embed_dim: self.embed_dim as usize,
            depth: self.depth as usize,
            n_heads: self.num_heads as usize,
            mlp_ratio: self.mlp_ratio(),
            patch_h: self.patch_h as usize,
            patch_w: self.patch_w as usize,
            // Non-overlapping tiling, verified against the stamped grid above.
            stride_h: self.patch_h as usize,
            stride_w: self.patch_w as usize,
            n_prepended_tokens: n_prepended,
            layer_norm_eps: self.layer_norm_eps,
            gelu: GeluKind::Erf,
            pos_embed_policy: PosEmbedPolicy::RequireExact,
        };
        attrs.validate()?;

        // The primitive resolves the FFN width itself by rounding
        // `embed_dim * mlp_ratio`. The converter stamps the width upstream
        // computes with `int(dim * mlp_ratio)`. If the two disagree, one of
        // them would size the FFN weights wrongly.
        if attrs.mlp_dim() != self.mlp_hidden_dim as usize {
            return Err(VokraError::ModelLoad(format!(
                "atst: `{GGUF_KEY_MLP_HIDDEN_DIM}` is {stamped} but `ViTAttrs::mlp_dim()` \
                 resolves embed_dim {d} times mlp_ratio {ratio} to {resolved}. The FFN \
                 weight shapes follow from this number, so a disagreement would bind \
                 mis-sized tensors (FR-EX-08). Primary source: {UPSTREAM_URL}",
                stamped = self.mlp_hidden_dim,
                d = self.embed_dim,
                ratio = self.mlp_ratio(),
                resolved = attrs.mlp_dim(),
            )));
        }
        Ok(attrs)
    }

    /// Applies upstream's `pos_type="cut"` rule: keep the prepended rows, then
    /// the **first** `n_patch_tokens` patch rows of the stamped table.
    ///
    /// Upstream allocates the positional table for [`Self::spec_w`] frames —
    /// a maximum, not a fixed input length — and slices it down to whatever
    /// the runtime grid needs. Doing that here, rather than asking the
    /// primitive to resize, is what makes
    /// [`PosEmbedPolicy::RequireExact`] the correct policy in
    /// [`Self::to_vit_attrs`]: after the cut the table has exactly one row per
    /// token, by construction.
    ///
    /// A prefix cut is only equivalent to upstream's slice while the grid is
    /// one row tall, because the patch rows are then laid out purely along the
    /// frame axis. That is the `atst-base` case; a taller grid is refused
    /// rather than cut in the wrong order.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `table` is not
    ///   `pos_embed_len * embed_dim` long, when the stamped grid has more than
    ///   one row, or when `n_patch_tokens` exceeds the patch rows the table
    ///   actually holds (the cut would run off the end, and padding it would
    ///   fabricate positions).
    pub fn cut_pos_embed(&self, table: &[f32], n_patch_tokens: usize) -> Result<Vec<f32>> {
        let d = self.embed_dim as usize;
        let rows = self.pos_embed_len as usize;
        if d == 0 || table.len() != rows * d {
            return Err(VokraError::InvalidArgument(format!(
                "atst::cut_pos_embed: expected a {rows} x {d} table ({} values), got {}",
                rows * d,
                table.len()
            )));
        }
        if self.patch_grid[0] != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "atst::cut_pos_embed: the stamped patch grid is {} row(s) tall, so a prefix \
                 cut is not upstream's slice — the patch rows are not laid out purely along \
                 the frame axis. Refusing to cut in the wrong order.",
                self.patch_grid[0]
            )));
        }
        let n_prepended = self.n_prepended_tokens();
        let available = rows - n_prepended;
        if n_patch_tokens > available {
            return Err(VokraError::InvalidArgument(format!(
                "atst::cut_pos_embed: asked for {n_patch_tokens} patch row(s) but the table \
                 holds only {available} (it is sized for spec_w = {sw} frames). Refusing to \
                 pad the table, which would fabricate positions.",
                sw = self.spec_w,
            )));
        }
        let keep = (n_prepended + n_patch_tokens) * d;
        Ok(table[..keep].to_vec())
    }

    /// Derives the exact dims every ViT weight tensor must carry.
    ///
    /// Pure arithmetic over the stamped axes — it needs no checkpoint, which
    /// is why it is real while the tensor *names* are not. Pair it with
    /// [`Atst::verify_vit_tensor_shapes`] once a name manifest is known.
    #[must_use]
    pub fn vit_tensor_shapes(&self) -> AtstVitShapes {
        let d = self.embed_dim as usize;
        AtstVitShapes {
            patch_embed_weight: [d, self.patch_embed_in_features as usize],
            patch_embed_bias: [d],
            pos_embed: [1, self.pos_embed_len as usize, d],
            prepended_tokens_elems: self.n_prepended_tokens() * d,
            block_qkv_weight: [3 * d, d],
            block_qkv_bias: if self.qkv_bias { Some([3 * d]) } else { None },
            block_proj_weight: [d, d],
            block_proj_bias: [d],
            block_fc1_weight: [self.mlp_hidden_dim as usize, d],
            block_fc1_bias: [self.mlp_hidden_dim as usize],
            block_fc2_weight: [d, self.mlp_hidden_dim as usize],
            block_fc2_bias: [d],
            norm_weight: [d],
            norm_bias: [d],
        }
    }
}

/// Required dims of each ViT weight tensor, derived from [`AtstConfig`].
///
/// Shapes are in **upstream torch order** (`[out_features, in_features]` for a
/// `nn.Linear`), because the converter passes each safetensors shape through
/// verbatim and the GGUF reader hands it back unchanged.
///
/// # The fused QKV projection
///
/// Upstream is `nn.Linear(dim, dim * 3, bias=qkv_bias)` — a **single** tensor
/// of `[3 * embed_dim, embed_dim]`, whereas
/// [`vokra_ops::vit::ViTAttnWeights`] wants separate `wq` / `wk` / `wv` of
/// `[embed_dim, embed_dim]`. Binding therefore has to split the fused rows
/// into three, and the split order is part of the unresolved tensor-name work:
/// nothing read so far states whether the rows run Q, K, V or are interleaved
/// per head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtstVitShapes {
    /// Patch projection weight, `[embed_dim, patch_h * patch_w]`.
    pub patch_embed_weight: [usize; 2],
    /// Patch projection bias, `[embed_dim]`. `nn.Linear` defaults to
    /// `bias=True` and upstream passes no `bias=` argument, but the flag is
    /// not stamped, so treat this as the shape to check *if* a bias is named.
    pub patch_embed_bias: [usize; 1],
    /// Positional table, `[1, pos_embed_len, embed_dim]` — upstream allocates
    /// `nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))`.
    pub pos_embed: [usize; 3],
    /// Element count of the learned prepended (CLS) tokens,
    /// `n_prepended_tokens * embed_dim`.
    ///
    /// An **element count, not a shape**: nothing read so far transcribes the
    /// rank upstream allocates the CLS parameter with, so asserting `[1, 1,
    /// embed_dim]` would be a guess.
    pub prepended_tokens_elems: usize,
    /// Fused QKV weight, `[3 * embed_dim, embed_dim]` — see the type docs.
    pub block_qkv_weight: [usize; 2],
    /// Fused QKV bias, `[3 * embed_dim]`, or `None` when `qkv_bias` is false
    /// (in which case no bias tensor exists at all).
    pub block_qkv_bias: Option<[usize; 1]>,
    /// Attention output projection weight, `[embed_dim, embed_dim]`.
    pub block_proj_weight: [usize; 2],
    /// Attention output projection bias, `[embed_dim]`.
    pub block_proj_bias: [usize; 1],
    /// FFN first linear weight, `[mlp_hidden_dim, embed_dim]`.
    pub block_fc1_weight: [usize; 2],
    /// FFN first linear bias, `[mlp_hidden_dim]`.
    pub block_fc1_bias: [usize; 1],
    /// FFN second linear weight, `[embed_dim, mlp_hidden_dim]`.
    pub block_fc2_weight: [usize; 2],
    /// FFN second linear bias, `[embed_dim]`.
    pub block_fc2_bias: [usize; 1],
    /// Any LayerNorm gain, `[embed_dim]` — `norm1` / `norm2` / the final norm
    /// all share this shape.
    pub norm_weight: [usize; 1],
    /// Any LayerNorm bias, `[embed_dim]`.
    pub norm_bias: [usize; 1],
}

/// The `state_dict` names of one Transformer block's tensors.
///
/// Field names follow the structure the converter transcribed from
/// `audiossl/modules/transformer.py` (a fused `qkv` projection, an output
/// `proj`, `norm1` / `norm2`, and a two-layer `Mlp`). The **strings** are the
/// caller's to supply — see [`AtstVitTensorNames`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtstVitBlockTensorNames {
    /// Pre-attention LayerNorm gain.
    pub norm1_weight: String,
    /// Pre-attention LayerNorm bias.
    pub norm1_bias: String,
    /// Fused QKV projection weight.
    pub qkv_weight: String,
    /// Fused QKV projection bias — `None` when `qkv_bias` is false.
    pub qkv_bias: Option<String>,
    /// Attention output projection weight.
    pub proj_weight: String,
    /// Attention output projection bias, when present.
    pub proj_bias: Option<String>,
    /// Pre-MLP LayerNorm gain.
    pub norm2_weight: String,
    /// Pre-MLP LayerNorm bias.
    pub norm2_bias: String,
    /// FFN first linear weight.
    pub fc1_weight: String,
    /// FFN first linear bias, when present.
    pub fc1_bias: Option<String>,
    /// FFN second linear weight.
    pub fc2_weight: String,
    /// FFN second linear bias, when present.
    pub fc2_bias: Option<String>,
}

/// A full ATST ViT tensor-name manifest, supplied by the caller.
///
/// # There is deliberately no `Default` and no `atst_base()`
///
/// This is the second of the two remaining blockers, made explicit in the type
/// system. Nothing in this repository records ATST's real `state_dict` keys:
/// the converter establishes that the chain runs `ATSTLightningModule.model`
/// -> `ATST.student` / `.teacher` -> `MultiCropWrapper` -> the `AST` encoder,
/// so the prefix is at least `model.student.` and **not** the bare `student.`
/// this module's fixtures use — but no checkpoint key listing has been read.
///
/// Shipping a guessed default would let callers bind the wrong tensors
/// without failing, so the caller must supply names it can defend, exactly as
/// [`vokra_ops::vit::ViTAttrs`] refuses to default its axes. Feed the result
/// to [`Atst::verify_vit_tensor_shapes`], which checks every name against the
/// dims [`AtstConfig::vit_tensor_shapes`] derives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtstVitTensorNames {
    /// Patch projection weight.
    pub patch_embed_weight: String,
    /// Patch projection bias, when present.
    pub patch_embed_bias: Option<String>,
    /// Learned prepended (CLS) token parameter — `None` when the config
    /// prepends no tokens.
    pub prepended_tokens: Option<String>,
    /// Positional table.
    pub pos_embed: String,
    /// One entry per Transformer block, in depth order.
    pub blocks: Vec<AtstVitBlockTensorNames>,
    /// Final LayerNorm gain, applied after the whole stack.
    pub final_norm_weight: String,
    /// Final LayerNorm bias.
    pub final_norm_bias: String,
}

// ---------------------------------------------------------------------------
// Strict metadata readers. Each names the absent key and the repro command;
// none of them falls back to a constant (FR-EX-08).
// ---------------------------------------------------------------------------

/// The shared tail of every missing-key message.
fn missing_key(key: &str, kind: &str) -> VokraError {
    VokraError::ModelLoad(format!(
        "atst: GGUF is missing required {kind} chunk `{key}` — a converter-produced \
         artifact always carries the full `vokra.atst.*` topology group, every value of \
         which `crates/vokra-convert/src/models/atst.rs` transcribes from the upstream \
         source tree. This binder refuses to fall back to a primary-source constant \
         (FR-EX-08): a silent default would let an artifact bind a topology it does not \
         actually carry. Re-run `vokra-cli convert --model atst-base` against an upstream \
         `{UPSTREAM_URL}` checkpoint flattened to safetensors by the offline uv-managed \
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

/// Reads a required `f32`-valued chunk. GGUF widens F32 to f64 exactly, so
/// narrowing back is lossless for anything the converter stamped.
fn req_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    gguf.get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .ok_or_else(|| missing_key(key, "f32"))
}

/// Reads a required bool-valued chunk.
///
/// A bool stamped as an integer reads back as absent, which is the honest
/// report: the converter stamps `add_bool`, so an integer means a foreign
/// writer produced the artifact.
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

/// Reads the indexed `vokra.atst.patch_grid_{i}` axis array as `[rows, cols]`.
fn req_u32_grid(gguf: &GgufFile, prefix: &str) -> Result<[u32; PATCH_GRID_AXIS_LEN]> {
    let mut out = [0u32; PATCH_GRID_AXIS_LEN];
    for (i, slot) in out.iter_mut().enumerate() {
        let key = format!("{prefix}_{i}");
        *slot = req_u32(gguf, &key)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// AtstBranch — the BYOL duo selector (diagnostic only, gates nothing)
// ---------------------------------------------------------------------------

/// Which branch of the BYOL-style teacher-student duo a caller is asking
/// about.
///
/// ATST is trained with an EMA **teacher** tracking a **student**, so a
/// released checkpoint can carry both sets of weights. Which one upstream's
/// own inference entry point uses is **not** recorded anywhere in this
/// repository, and choosing wrongly produces a shape-valid but numerically
/// different embedding — a silent misroute. That is the branch-selection
/// blocker of the [`Atst::encode`] loud-partial.
///
/// Consequently this enum is a **diagnostic** only: [`Atst::branch_tensor_count`]
/// counts tensors whose name starts with [`Self::prefix`] so an operator can
/// see what a given artifact actually contains. It gates nothing, and a
/// checkpoint carrying neither prefix is **not** rejected — the prefix
/// convention below is transcribed from the converter's own test module and
/// has not been verified against a real upstream checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtstBranch {
    /// The student branch — the one that receives gradient updates and takes
    /// the patchout augmentation during training.
    Student,
    /// The teacher branch — the exponential-moving-average copy of the
    /// student.
    Teacher,
}

impl AtstBranch {
    /// The `state_dict` name prefix associated with this branch.
    ///
    /// **Unverified convention.** These prefixes are transcribed from the
    /// sample tensor names in the converter's own test module
    /// (`student.encoder.blocks.0.norm1.weight` /
    /// `teacher.encoder.blocks.0.attn.qkv.weight`), which that module labels
    /// "realistic upstream state-dict name" — they have not been checked
    /// against a real ATST checkpoint. Nothing in this binder *depends* on
    /// them; they only shape the counts reported by
    /// [`Atst::branch_tensor_count`].
    #[inline]
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Student => "student.",
            Self::Teacher => "teacher.",
        }
    }

    /// Both branches, in a stable order, for callers that want to report on
    /// each in turn.
    #[inline]
    #[must_use]
    pub const fn all() -> [Self; 2] {
        [Self::Student, Self::Teacher]
    }
}

// ---------------------------------------------------------------------------
// AtstWeights — the tensor manifest, with loud lookups
// ---------------------------------------------------------------------------

/// Weight tensors bound from an ATST GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification step.
/// A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an ~86M-parameter Transformer never
/// converts to an empty manifest, so zero tensors always signals a
/// mis-produced artifact, and binding it would silently run an all-zero
/// forward).
///
/// Under the current landing this struct stores the tensor names and their
/// GGUF-side dims. The payload is deliberately not dequantised: the forward is
/// loud-partial (see [`Atst::encode`]), and the follow-up wave sizes its
/// dequant per its kernel needs. [`require_tensor`](Self::require_tensor) /
/// [`require_tensor_dims`](Self::require_tensor_dims) are already in place so
/// that wave walks a manifest that fails loudly rather than substituting
/// zeros.
#[derive(Debug, Clone)]
pub struct AtstWeights {
    /// Tensors discovered on disk, in file order, as
    /// `(upstream state_dict name, GGUF-side dims)`.
    tensors: Vec<(String, Vec<usize>)>,
}

impl AtstWeights {
    /// Scans `gguf` for the ATST `state_dict` tensors.
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
                "atst: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate ATST checkpoint is an ~86M-parameter \
                 Transformer (arch={ARCH}, name={NAME}) and always converts to hundreds \
                 of Linear / LayerNorm tensors, so an empty manifest always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model atst-base` against \
                 an upstream `{UPSTREAM_URL}` checkpoint flattened to safetensors by the \
                 offline uv-managed Python 3.12 sidecar."
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
    /// A plain string count over what is actually on disk — it asserts
    /// nothing about the upstream naming convention. See
    /// [`AtstBranch::prefix`] for the (unverified) prefixes this pairs with.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .count()
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
            "atst: required tensor `{name}` is absent from the GGUF ({count} tensors \
             present; nearest names on disk: {near:?}). The converter passes upstream \
             `state_dict` names through verbatim, so a mismatch means either the \
             checkpoint was flattened with a different prefix policy (e.g. a branch \
             prefix such as `{student}` / `{teacher}` was stripped) or the caller is \
             walking a manifest transcribed from a different ATST variant \
             (`atst-frame`, the frame-level TASLP 2023 sibling, is published under its \
             own name). Refusing to substitute a zero tensor (FR-EX-08). Primary \
             source: {UPSTREAM_URL}",
            count = self.tensors.len(),
            student = AtstBranch::Student.prefix(),
            teacher = AtstBranch::Teacher.prefix(),
        )))
    }

    /// Asserts that a required tensor is present **and** has exactly
    /// `expected` dims.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the tensor is absent (see
    ///   [`Self::require_tensor`]) or when its dims differ — the message names
    ///   **both** the expected and the actual dims (FR-EX-08 — never reshape
    ///   or truncate silently).
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<()> {
        let actual = self.require_tensor(name)?;
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "atst: tensor `{name}` has dims {actual:?} but the caller expects \
                 {expected:?} — refusing to reshape or truncate silently (FR-EX-08). \
                 When the expected dims came from `AtstConfig::vit_tensor_shapes` they \
                 are derived from the artifact's own stamped topology group, so a \
                 disagreement means the tensor NAME resolved to the wrong tensor rather \
                 than that the axes are wrong — check the manifest against a real \
                 checkpoint's key listing. A hand-written expectation may instead be \
                 transcribed from a different size point, or from the frame-level \
                 `atst-frame` sibling. Primary source: {UPSTREAM_URL}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Atst — the runtime binder handle
// ---------------------------------------------------------------------------

/// ATST (Audio Teacher-Student Transformer) self-supervised audio encoder.
///
/// Bind with [`from_gguf`](Self::from_gguf) — or the compliance-gated
/// [`from_gguf_with_policy`](Self::from_gguf_with_policy) /
/// [`from_path`](Self::from_path) — then call [`encode`](Self::encode) for
/// per-patch hidden states or [`embed`](Self::embed) for the pooled utterance
/// embedding. Both forwards are **loud-partial** today; see the module
/// docstring for the two remaining blockers and the FR-EX-08 contract.
///
/// The topology axes are **not** among them: [`Self::config`] reads them and
/// [`Self::vit_attrs`] maps them onto [`vokra_ops::vit::ViTAttrs`].
#[derive(Debug, Clone)]
pub struct Atst {
    name: Option<String>,
    category: Option<String>,
    upstream_url: Option<String>,
    config: AtstConfig,
    weights: AtstWeights,
    weight_license: LicenseClass,
    attribution: Option<String>,
}

impl Atst {
    /// Binds an ATST GGUF: verifies the arch strictly, binds the tensor
    /// manifest, and surfaces the converter's metadata + licence stamps.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing or wrong key, so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// `vokra.model.name` is deliberately **surfaced, not gated**: the
    /// frame-level `atst-frame` sibling shares this arch under a different
    /// name, so a hard name check would make a legitimate future artifact
    /// unloadable. See [`Self::name`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is not `"atst"` —
    ///   the message names both the found and the expected tag and enumerates
    ///   the SSL audio/music-embedding neighbourhood.
    /// - [`VokraError::ModelLoad`] when any `vokra.atst.*` topology chunk is
    ///   absent ([`AtstConfig::from_gguf`]) — the message names the key.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`AtstWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch first, so a mis-routed artifact reports the arch mismatch
        //    (the actionable fact) instead of a downstream missing-tensor
        //    trail.
        verify_arch(file)?;

        // 2. Metadata surfacing. Soft: a converter-produced artifact always
        //    carries these, but they are diagnostics, not load gates.
        let read_str = |key: &str| -> Option<String> {
            file.get(key).and_then(|v| v.as_str()).map(str::to_owned)
        };
        let name = read_str(chunks::KEY_MODEL_NAME);
        let category = read_str(GGUF_KEY_MODEL_CATEGORY);
        let upstream_url = read_str(GGUF_KEY_PROVENANCE_UPSTREAM_URL);

        // 3. Topology group. STRICT — every stamped key is required, with no
        //    primary-source constant fallback. An artifact that predates the
        //    converter's topology group cannot be forwarded anyway, and
        //    binding it half-way would be exactly the silent partial load
        //    this module refuses (FR-EX-08).
        let config = AtstConfig::from_gguf(file)?;

        // 4. Tensor manifest with the non-emptiness gate.
        let weights = AtstWeights::from_gguf(file)?;

        // 5. Provenance surfacing. The converter stamps `AttributionRequired`
        //    (cc-by-4.0); an artifact missing the stamp reads back as
        //    `Unknown` — fail-closed at the M2-13 gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let attribution = read_str(chunks::KEY_PROVENANCE_ATTRIBUTION);

        Ok(Self {
            name,
            category,
            upstream_url,
            config,
            weights,
            weight_license,
            attribution,
        })
    }

    /// Loads an ATST GGUF from raw bytes under `policy` (the M2-13
    /// weight-licence gate).
    ///
    /// ATST ships **CC-BY 4.0** → [`LicenseClass::AttributionRequired`],
    /// which is commercially permitted, so a correctly stamped artifact passes
    /// under [`CompliancePolicy::strict`] without a research opt-in. An
    /// artifact with no provenance stamp resolves to
    /// [`LicenseClass::Unknown`] and is refused — fail-closed, never a silent
    /// substitution.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on GGUF parse failure, or on a wrong /
    ///   missing `vokra.model.arch`.
    /// - `VokraError::ResearchLicenseRequired` from the compliance gate when
    ///   the weight class is gated and `policy` grants no research opt-in.
    /// - See [`Self::from_gguf`] for the remaining bind errors.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("atst GGUF: {e}")))?;
        // Arch before the compliance gate so a mis-routed artifact reports the
        // arch mismatch rather than a licence verdict about a model the caller
        // never meant to load.
        verify_arch(&file)?;
        check_weight_license(&file, policy)?;
        Self::from_gguf(&file)
    }

    /// Loads an ATST GGUF from a path under [`CompliancePolicy::strict`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
    }

    /// The stamped `vokra.model.name`, if present.
    ///
    /// [`NAME`] (`"atst-base"`) for the utterance-level 2022 release this
    /// module tracks; the frame-level `atst-frame` sibling shares [`ARCH`]
    /// under a different name, which is why this is surfaced rather than
    /// gated.
    #[inline]
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The stamped `vokra.model.category`, if present — [`CATEGORY`]
    /// (`"audio-embedding"`) for a converter-produced artifact.
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The stamped `vokra.provenance.upstream_url`, if present —
    /// [`UPSTREAM_URL`] for a converter-produced artifact. ATST is not on
    /// HuggingFace, so there is no `upstream_hf` counterpart.
    #[inline]
    #[must_use]
    pub fn upstream_url(&self) -> Option<&str> {
        self.upstream_url.as_deref()
    }

    /// The topology axes read from the artifact's `vokra.atst.*` group.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &AtstConfig {
        &self.config
    }

    /// The stamped axes mapped onto [`vokra_ops::vit::ViTAttrs`].
    ///
    /// # Errors
    ///
    /// Propagates [`AtstConfig::to_vit_attrs`].
    pub fn vit_attrs(&self) -> Result<ViTAttrs> {
        self.config.to_vit_attrs()
    }

    /// Required dims of every ViT weight, derived from [`Self::config`].
    #[inline]
    #[must_use]
    pub fn vit_tensor_shapes(&self) -> AtstVitShapes {
        self.config.vit_tensor_shapes()
    }

    /// Checks a caller-supplied tensor-name manifest against the dims
    /// [`Self::vit_tensor_shapes`] derives, using
    /// [`AtstWeights::require_tensor_dims`] so any absent or wrong-shaped
    /// tensor names itself.
    ///
    /// This is the "flip the switch" verification for the remaining
    /// tensor-name blocker: once a real checkpoint's key listing is read, this
    /// call proves the transcribed names actually resolve to tensors of the
    /// shapes the artifact's own topology group implies — before any forward
    /// runs. It reads names and dims only; no payload is dequantised.
    ///
    /// Optional names are checked only when supplied, because the upstream
    /// bias flags for the patch projection, the attention output projection
    /// and the FFN are not stamped. `qkv_bias` is the exception: it **is**
    /// stamped, so supplying a QKV bias name when the config says
    /// `qkv_bias = false` is refused rather than checked.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the block count differs from the
    ///   stamped depth, when a named tensor is absent or has the wrong dims,
    ///   when a prepended-token name is supplied for a config that prepends
    ///   none (or omitted for one that does), or when a QKV bias name
    ///   contradicts the stamped `qkv_bias` flag.
    pub fn verify_vit_tensor_shapes(&self, names: &AtstVitTensorNames) -> Result<()> {
        let shapes = self.vit_tensor_shapes();
        let w = &self.weights;
        let depth = self.config.depth as usize;
        if names.blocks.len() != depth {
            return Err(VokraError::ModelLoad(format!(
                "atst: the supplied tensor-name manifest describes {got} block(s) but the \
                 artifact stamps depth {depth}. Refusing to bind a partial stack \
                 (FR-EX-08). Primary source: {UPSTREAM_URL}",
                got = names.blocks.len(),
            )));
        }

        w.require_tensor_dims(&names.patch_embed_weight, &shapes.patch_embed_weight)?;
        if let Some(bias) = &names.patch_embed_bias {
            w.require_tensor_dims(bias, &shapes.patch_embed_bias)?;
        }
        w.require_tensor_dims(&names.pos_embed, &shapes.pos_embed)?;

        // Prepended tokens: presence and element count only. The rank upstream
        // allocates the CLS parameter with is not transcribed anywhere, so
        // asserting a shape would be a guess (see `AtstVitShapes`).
        match (&names.prepended_tokens, shapes.prepended_tokens_elems) {
            (Some(name), elems) if elems > 0 => {
                let dims = w.require_tensor(name)?;
                let actual: usize = dims.iter().product();
                if actual != elems {
                    return Err(VokraError::ModelLoad(format!(
                        "atst: prepended-token tensor `{name}` holds {actual} element(s) \
                         but the stamped topology implies {elems} \
                         (n_prepended_tokens x embed_dim). Its rank is deliberately not \
                         asserted — upstream's CLS allocation is not transcribed — but the \
                         element count must match (FR-EX-08). Primary source: \
                         {UPSTREAM_URL}"
                    )));
                }
            }
            (Some(name), _) => {
                return Err(VokraError::ModelLoad(format!(
                    "atst: the manifest names a prepended-token tensor `{name}` but the \
                     artifact stamps `{GGUF_KEY_USE_CLS}` false, so no such parameter \
                     exists. Refusing to bind a token the topology does not declare \
                     (FR-EX-08). Primary source: {UPSTREAM_URL}"
                )));
            }
            (None, elems) if elems > 0 => {
                return Err(VokraError::ModelLoad(format!(
                    "atst: the artifact stamps `{GGUF_KEY_USE_CLS}` true, so a learned \
                     prepended token of {elems} element(s) must be bound, but the manifest \
                     names none. Refusing to prepend a zero token (FR-EX-08). Primary \
                     source: {UPSTREAM_URL}"
                )));
            }
            (None, _) => {}
        }

        for (i, block) in names.blocks.iter().enumerate() {
            w.require_tensor_dims(&block.norm1_weight, &shapes.norm_weight)?;
            w.require_tensor_dims(&block.norm1_bias, &shapes.norm_bias)?;
            w.require_tensor_dims(&block.qkv_weight, &shapes.block_qkv_weight)?;
            match (&block.qkv_bias, shapes.block_qkv_bias) {
                (Some(name), Some(dims)) => w.require_tensor_dims(name, &dims)?,
                (Some(name), None) => {
                    return Err(VokraError::ModelLoad(format!(
                        "atst: block {i} names a QKV bias `{name}` but the artifact stamps \
                         `{GGUF_KEY_QKV_BIAS}` false, so `nn.Linear(dim, dim * 3, \
                         bias=False)` carries no bias tensor at all. Refusing to bind a \
                         tensor the topology denies (FR-EX-08). Primary source: \
                         {UPSTREAM_URL}"
                    )));
                }
                (None, Some(_)) => {
                    return Err(VokraError::ModelLoad(format!(
                        "atst: the artifact stamps `{GGUF_KEY_QKV_BIAS}` true but block {i} \
                         names no QKV bias. Refusing to substitute a zero bias (FR-EX-08). \
                         Primary source: {UPSTREAM_URL}"
                    )));
                }
                (None, None) => {}
            }
            w.require_tensor_dims(&block.proj_weight, &shapes.block_proj_weight)?;
            if let Some(bias) = &block.proj_bias {
                w.require_tensor_dims(bias, &shapes.block_proj_bias)?;
            }
            w.require_tensor_dims(&block.norm2_weight, &shapes.norm_weight)?;
            w.require_tensor_dims(&block.norm2_bias, &shapes.norm_bias)?;
            w.require_tensor_dims(&block.fc1_weight, &shapes.block_fc1_weight)?;
            if let Some(bias) = &block.fc1_bias {
                w.require_tensor_dims(bias, &shapes.block_fc1_bias)?;
            }
            w.require_tensor_dims(&block.fc2_weight, &shapes.block_fc2_weight)?;
            if let Some(bias) = &block.fc2_bias {
                w.require_tensor_dims(bias, &shapes.block_fc2_bias)?;
            }
        }

        w.require_tensor_dims(&names.final_norm_weight, &shapes.norm_weight)?;
        w.require_tensor_dims(&names.final_norm_bias, &shapes.norm_bias)?;
        Ok(())
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &AtstWeights {
        &self.weights
    }

    /// Number of tensors discovered on disk.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// How many tensors on disk carry `branch`'s name prefix.
    ///
    /// **Diagnostic only** — it gates nothing, and `0` for both branches is
    /// not an error (the prefix convention is unverified; see
    /// [`AtstBranch::prefix`]). It exists so an operator inspecting a
    /// converted BYOL checkpoint can see whether it carries one branch or
    /// both, which is the branch-selection blocker of the [`Self::encode`]
    /// loud-partial.
    #[inline]
    #[must_use]
    pub fn branch_tensor_count(&self, branch: AtstBranch) -> usize {
        self.weights.count_with_prefix(branch.prefix())
    }

    /// The weight-licence class surfaced from
    /// `vokra.provenance.weight_license`.
    ///
    /// [`LicenseClass::AttributionRequired`] for a correctly stamped ATST
    /// artifact (cc-by-4.0); [`LicenseClass::Unknown`] when the stamp is
    /// absent (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The FR-MD-09 attribution text stamped under
    /// `vokra.provenance.attribution`, if any.
    ///
    /// CC-BY 4.0 obliges a downstream to display attribution alongside output
    /// derived from the weights, so this is surfaced rather than buried: a
    /// consumer shipping ATST-derived embeddings must render this string.
    /// `None` means the artifact carries no stamp (for example it was
    /// converted with an explicit `--license` override, which suppresses the
    /// CC-BY wording).
    #[inline]
    #[must_use]
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }

    /// Encodes a mono `f32` PCM slice into the ATST encoder's **per-patch
    /// hidden states** (`[n_patches][embed_dim]`).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the **two** blockers that
    /// remain: unresolved teacher/student branch selection, and no verified
    /// tensor-name manifest. Both are facts about a real checkpoint that no
    /// file in this repository records, and reading one key listing resolves
    /// both. All three primary sources are cited so a reader diagnosing the
    /// gap has exactly three places to walk. **No fabricated hidden states are
    /// ever emitted** (FR-EX-08 — no silent partial output).
    ///
    /// The axis group and the ViT primitive are **no longer blockers** — see
    /// [`AtstConfig`] and [`vokra_ops::vit`].
    ///
    /// `pcm` is treated as mono `f32` in `[-1, 1]`. Its required sample rate
    /// is now stamped and surfaced as
    /// [`config().sample_rate`](AtstConfig::sample_rate); it is not asserted
    /// here because the forward does not run, so there is nothing yet for a
    /// rate check to protect.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind explicitly so a future accidental removal of the parameter is
        // not masked by an unused-variable warning (mirror of the
        // emotion2vec / wavlm loud-partial signature discipline).
        let _ = pcm;
        Err(forward_loud_partial("encode", "per-patch hidden states"))
    }

    /// Encodes a mono `f32` PCM slice into the **pooled utterance embedding**
    /// the utterance-level ATST ([`NAME`]) defines.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] for the same two blockers as
    /// [`Self::encode`] — the pooled embedding is the encoder output reduced
    /// over the patch axis, so it cannot exist before the encoder does. The
    /// **width** of the vector this would return is now known
    /// ([`config().embed_dim`](AtstConfig::embed_dim)); what is missing is the
    /// content, and **no fabricated embedding is ever emitted** (FR-EX-08).
    ///
    /// Which pooling upstream applies — the CLS token or a mean over the patch
    /// tokens ([`vokra_ops::vit::ViTPooling`] offers both) — follows from the
    /// same inference entry point that decides the branch, so it is folded
    /// into the branch-selection blocker rather than guessed here.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward.
    pub fn embed(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(forward_loud_partial("embed", "pooled utterance embedding"))
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Strict `vokra.model.arch` verification.
///
/// Refuses a foreign GGUF loudly, naming **both** the found and the expected
/// tag and enumerating the SSL audio/music-embedding neighbourhood plus the
/// wav2vec2 lineage, so a reader who handed the wrong artifact over knows
/// immediately which loader they wanted (FR-EX-08 — never a silent misroute).
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "atst: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF produced by \
             `vokra-cli convert --model atst-base`?). ATST is a BYOL-style \
             teacher-student patchout SSL encoder over a log-mel patch grid; the sibling \
             SSL audio/music-embedding arches differ in the pre-training objective that \
             shapes the topology — `beats` (iterative acoustic tokenizer + masked \
             acoustic modelling), `eat` (utterance-level MAE with inverse block masking), \
             `dasheng` (universal MAE), `m2d` (masked modelling duo, dual online + target \
             branch), `maest` (AST backbone, Discogs music-tagger objective), `mert` \
             (HuBERT-derived masked prediction), `muq` (Mel-RVQ + BEATs teacher), \
             `yamnet` (supervised audio-tagging CNN, not SSL) — and the wav2vec2 lineage \
             (`hubert`, `wav2vec2_ctc`, `wavlm_sv`, `emotion2vec`) sits on a raw-waveform \
             1-D conv stem rather than a log-mel patch grid. Binding any of them here \
             would walk a foreign topology over an ATST payload (FR-EX-08 — no silent \
             partial load). Primary source: {UPSTREAM_URL}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "atst: GGUF is missing `vokra.model.arch` — this is not a Vokra-native atst \
             GGUF (was it produced by `vokra-cli convert --model atst-base`?). Refusing \
             to guess the arch from the tensor manifest (FR-EX-08). Primary source: \
             {UPSTREAM_URL}"
        ))),
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Atst::encode`] / [`Atst::embed`] until the ATST forward wave lands.
///
/// `surface` is the method name, `output` is what that method would have
/// returned. The message names the **two** blockers that remain and cites
/// **three** primary sources so a reader diagnosing the gap has fully
/// specified anchors (`emotion2vec` / `wavlm` / `panns` / `redimnet`
/// loud-partial-message precedent — CLAUDE.md 教訓 (a)).
///
/// # Every claim here must be true on the day it is read
///
/// This message previously named four blockers, two of which are now resolved
/// (the converter stamps the topology group; [`vokra_ops::vit`] exists). A
/// stale claim in an error message actively misleads whoever reads it next —
/// it sends them to close a gap that is already closed — so the resolved two
/// are stated as *resolved* rather than merely deleted, and the message
/// deliberately does not name the axis-group key prefix as missing.
fn forward_loud_partial(surface: &str, output: &str) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "atst {surface} (loud-partial): the ATST encoder forward is deferred, so no \
         {output} can be produced. TWO blockers remain, and both are facts about a real \
         checkpoint that no file in this repository records: \
         (i) TEACHER/STUDENT BRANCH SELECTION IS UNRESOLVED — a BYOL-style EMA \
         checkpoint carries both branches (`{student}` / `{teacher}` prefixes), and \
         picking the wrong one yields a shape-valid but numerically different embedding, \
         so which branch upstream's own inference entry point uses must be read off the \
         upstream tree first; the same entry point also settles which pooling the \
         utterance embedding uses (CLS token vs mean over patch tokens); \
         (ii) NO VERIFIED TENSOR-NAME MANIFEST — the converter copies every float tensor \
         under its verbatim upstream `state_dict` name and nothing in-repo transcribes \
         ATST's naming. The key chain is known to run `ATSTLightningModule.model` -> \
         `ATST.student` / `.teacher` -> `MultiCropWrapper` -> the `AST` encoder, so the \
         real prefix is at least `model.student.` and NOT the bare `{student}` used by \
         this module's fixtures — but no checkpoint key listing has been read, so the \
         exact strings are unknown and walking guessed names into typed slots would bind \
         shape-valid garbage. Reading one real key listing resolves BOTH blockers at \
         once, since the branch prefix and the tensor names are the same listing; \
         `AtstVitTensorNames` + `Atst::verify_vit_tensor_shapes` are already in place to \
         check that listing against this artifact's own stamped topology. \
         ALREADY RESOLVED, so do not go looking: the topology axes ARE stamped and are \
         read by `AtstConfig::from_gguf` (Transformer width / depth / heads, the patch \
         grid and position table, and the whole log-mel front-end), and the ViT patch \
         encoder DOES exist — `vokra_ops::vit::ViTEncoder`, onto which \
         `AtstConfig::to_vit_attrs` maps those axes. Primary sources: upstream tree \
         {upstream}, paper (utterance-level, INTERSPEECH 2022) {p2022}, paper \
         (frame-level `atst-frame`, TASLP 2023) {p2023}. The runtime cannot fabricate \
         {output} (FR-EX-08 — no silent partial output; CLAUDE.md 教訓 (a) \
         'loud-partial は fake-complete より honest').",
        student = AtstBranch::Student.prefix(),
        teacher = AtstBranch::Teacher.prefix(),
        upstream = PRIMARY_SOURCE_UPSTREAM,
        p2022 = PRIMARY_SOURCE_PAPER_2022,
        p2023 = PRIMARY_SOURCE_PAPER_2023,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the ATST runtime binder — contract-constant pins against the
    //! converter, metadata round-trip, loud negative space on every stated
    //! blocker, and arch-tag distinctness.
    //!
    //! # What "round-trip" means here
    //!
    //! On real audio this would be `encode(...)` returning hidden states, but
    //! the ATST forward is loud-partial (two blockers, see the module doc).
    //! Fabricating an output would violate CLAUDE.md 教訓 (a)
    //! (「loud-partial は fake-complete より honest」). The round-trips we
    //! *can* honestly test:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` and the two metadata keys
    //!    match the converter exactly, so a converter drift without a
    //!    binder-side follow-through fails here.
    //! 2. **Metadata round-trip** — a synthetic GGUF shaped like the
    //!    converter's output binds, and every stamp reads back.
    //! 3. **Loud negative space** — missing arch, foreign arch, empty
    //!    manifest, missing tensor, wrong dims, a withheld topology key, an
    //!    internally contradictory axis set, and both forward surfaces all
    //!    fire at their documented surface point in their documented variant.
    //! 4. **Arch distinctness pin** — the tag differs from every sibling SSL
    //!    audio/music-embedding arch and from the wav2vec2 lineage.
    //! 5. **Topology round-trip** — the stamped `vokra.atst.*` group reads
    //!    back into `AtstConfig` field for field, maps onto `ViTAttrs`, and
    //!    those attrs pass the primitive's own `validate()`.
    //! 6. **Derived-shape walk** — the dims `AtstConfig::vit_tensor_shapes`
    //!    derives are checked against a miniature synthetic artifact through
    //!    `require_tensor_dims`.
    //!
    //! # What is deliberately NOT asserted
    //!
    //! No numerical parity against upstream. There is no reference to compare
    //! against here, the forward does not run, and inventing an expected value
    //! would be exactly the fabrication this module exists to refuse. Nor are
    //! the tensor NAMES used below a transcription — see `small_vit_names`.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// A tensor-name sample shaped like the converter's own test module's
    /// "realistic upstream state-dict name" choices. Unverified against a real
    /// checkpoint — used here only to give the manifest something to hold.
    const SAMPLE_TENSORS: [(&str, [u64; 2]); 4] = [
        ("student.encoder.blocks.0.norm1.weight", [4, 1]),
        ("student.encoder.blocks.0.attn.qkv.weight", [4, 12]),
        ("teacher.encoder.blocks.0.norm1.weight", [4, 1]),
        ("teacher.encoder.blocks.0.attn.qkv.weight", [4, 12]),
    ];

    /// The ATST-base axis set the converter stamps, transcribed from
    /// `crates/vokra-convert/src/models/atst.rs`'s constants block (which in
    /// turn records the upstream file and line for each value). Pinned here so
    /// a converter drift without a binder-side follow-through fails.
    fn atst_base_config() -> AtstConfig {
        AtstConfig {
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio_scaled_1e3: 4_000,
            mlp_hidden_dim: 3_072,
            layer_norm_eps: 1e-6,
            qkv_bias: false,
            use_cls: true,
            act_layer: ACT_LAYER_GELU.to_owned(),
            in_chans: 1,
            num_classes: 0,
            drop_path_rate_scaled_1e3: 100,
            patch_h: 64,
            patch_w: 4,
            spec_h: 64,
            spec_w: 1001,
            num_patches: 250,
            pos_embed_len: 251,
            pos_type: POS_TYPE_CUT.to_owned(),
            patch_embed_in_features: 256,
            patch_embed_kind: PATCH_EMBED_KIND_LINEAR.to_owned(),
            patch_order: PATCH_ORDER_WH.to_owned(),
            patch_grid: [1, 250],
            sample_rate: 16_000,
            n_fft: 1024,
            hop_length: 160,
            win_length: 1024,
            n_mels: 64,
            f_min: 60,
            f_max: 7800,
            amp_to_db_top_db: 80,
            amp_to_db_stype: "power".to_owned(),
            norm_min: -79.6482,
            norm_max: 50.6842,
        }
    }

    /// A miniature but **internally consistent** ATST-shaped axis set, so the
    /// tensor-shape tests can build a whole ViT weight set without allocating
    /// the ~86M-parameter real one. Every cross-check `to_vit_attrs` performs
    /// holds: grid = [4/4, 9/2] = [1, 4], num_patches = 4,
    /// pos_embed_len = 4 + 1, patch_embed_in_features = 4*2, mlp_hidden = 8*4.
    fn small_config() -> AtstConfig {
        AtstConfig {
            embed_dim: 8,
            depth: 2,
            num_heads: 2,
            mlp_ratio_scaled_1e3: 4_000,
            mlp_hidden_dim: 32,
            patch_h: 4,
            patch_w: 2,
            spec_h: 4,
            spec_w: 9,
            num_patches: 4,
            pos_embed_len: 5,
            patch_embed_in_features: 8,
            patch_grid: [1, 4],
            n_mels: 4,
            ..atst_base_config()
        }
    }

    /// Stamps a whole `vokra.atst.*` group from `cfg`, optionally omitting one
    /// key so the strict reader's missing-key path can be exercised.
    fn stamp_axis_group(b: &mut GgufBuilder, cfg: &AtstConfig, skip: Option<&str>) {
        let keep = |k: &str| match skip {
            Some(s) => s != k,
            None => true,
        };
        if keep(GGUF_KEY_EMBED_DIM) {
            b.add_u32(GGUF_KEY_EMBED_DIM, cfg.embed_dim);
        }
        if keep(GGUF_KEY_DEPTH) {
            b.add_u32(GGUF_KEY_DEPTH, cfg.depth);
        }
        if keep(GGUF_KEY_NUM_HEADS) {
            b.add_u32(GGUF_KEY_NUM_HEADS, cfg.num_heads);
        }
        if keep(GGUF_KEY_MLP_RATIO_SCALED_1E3) {
            b.add_u32(GGUF_KEY_MLP_RATIO_SCALED_1E3, cfg.mlp_ratio_scaled_1e3);
        }
        if keep(GGUF_KEY_MLP_HIDDEN_DIM) {
            b.add_u32(GGUF_KEY_MLP_HIDDEN_DIM, cfg.mlp_hidden_dim);
        }
        if keep(GGUF_KEY_LAYER_NORM_EPS) {
            b.add_f32(GGUF_KEY_LAYER_NORM_EPS, cfg.layer_norm_eps);
        }
        if keep(GGUF_KEY_QKV_BIAS) {
            b.add_bool(GGUF_KEY_QKV_BIAS, cfg.qkv_bias);
        }
        if keep(GGUF_KEY_USE_CLS) {
            b.add_bool(GGUF_KEY_USE_CLS, cfg.use_cls);
        }
        if keep(GGUF_KEY_ACT_LAYER) {
            b.add_string(GGUF_KEY_ACT_LAYER, &cfg.act_layer);
        }
        if keep(GGUF_KEY_IN_CHANS) {
            b.add_u32(GGUF_KEY_IN_CHANS, cfg.in_chans);
        }
        if keep(GGUF_KEY_NUM_CLASSES) {
            b.add_u32(GGUF_KEY_NUM_CLASSES, cfg.num_classes);
        }
        if keep(GGUF_KEY_DROP_PATH_RATE_SCALED_1E3) {
            b.add_u32(
                GGUF_KEY_DROP_PATH_RATE_SCALED_1E3,
                cfg.drop_path_rate_scaled_1e3,
            );
        }
        if keep(GGUF_KEY_PATCH_H) {
            b.add_u32(GGUF_KEY_PATCH_H, cfg.patch_h);
        }
        if keep(GGUF_KEY_PATCH_W) {
            b.add_u32(GGUF_KEY_PATCH_W, cfg.patch_w);
        }
        if keep(GGUF_KEY_SPEC_H) {
            b.add_u32(GGUF_KEY_SPEC_H, cfg.spec_h);
        }
        if keep(GGUF_KEY_SPEC_W) {
            b.add_u32(GGUF_KEY_SPEC_W, cfg.spec_w);
        }
        if keep(GGUF_KEY_NUM_PATCHES) {
            b.add_u32(GGUF_KEY_NUM_PATCHES, cfg.num_patches);
        }
        if keep(GGUF_KEY_POS_EMBED_LEN) {
            b.add_u32(GGUF_KEY_POS_EMBED_LEN, cfg.pos_embed_len);
        }
        if keep(GGUF_KEY_POS_TYPE) {
            b.add_string(GGUF_KEY_POS_TYPE, &cfg.pos_type);
        }
        if keep(GGUF_KEY_PATCH_EMBED_IN_FEATURES) {
            b.add_u32(
                GGUF_KEY_PATCH_EMBED_IN_FEATURES,
                cfg.patch_embed_in_features,
            );
        }
        if keep(GGUF_KEY_PATCH_EMBED_KIND) {
            b.add_string(GGUF_KEY_PATCH_EMBED_KIND, &cfg.patch_embed_kind);
        }
        if keep(GGUF_KEY_PATCH_ORDER) {
            b.add_string(GGUF_KEY_PATCH_ORDER, &cfg.patch_order);
        }
        for (i, &v) in cfg.patch_grid.iter().enumerate() {
            let key = format!("{GGUF_KEY_PATCH_GRID_PREFIX}_{i}");
            if keep(key.as_str()) {
                b.add_u32(key.as_str(), v);
            }
        }
        if keep(GGUF_KEY_SAMPLE_RATE) {
            b.add_u32(GGUF_KEY_SAMPLE_RATE, cfg.sample_rate);
        }
        if keep(GGUF_KEY_N_FFT) {
            b.add_u32(GGUF_KEY_N_FFT, cfg.n_fft);
        }
        if keep(GGUF_KEY_HOP_LENGTH) {
            b.add_u32(GGUF_KEY_HOP_LENGTH, cfg.hop_length);
        }
        if keep(GGUF_KEY_WIN_LENGTH) {
            b.add_u32(GGUF_KEY_WIN_LENGTH, cfg.win_length);
        }
        if keep(GGUF_KEY_N_MELS) {
            b.add_u32(GGUF_KEY_N_MELS, cfg.n_mels);
        }
        if keep(GGUF_KEY_F_MIN) {
            b.add_u32(GGUF_KEY_F_MIN, cfg.f_min);
        }
        if keep(GGUF_KEY_F_MAX) {
            b.add_u32(GGUF_KEY_F_MAX, cfg.f_max);
        }
        if keep(GGUF_KEY_AMP_TO_DB_TOP_DB) {
            b.add_u32(GGUF_KEY_AMP_TO_DB_TOP_DB, cfg.amp_to_db_top_db);
        }
        if keep(GGUF_KEY_AMP_TO_DB_STYPE) {
            b.add_string(GGUF_KEY_AMP_TO_DB_STYPE, &cfg.amp_to_db_stype);
        }
        if keep(GGUF_KEY_NORM_MIN) {
            b.add_f32(GGUF_KEY_NORM_MIN, cfg.norm_min);
        }
        if keep(GGUF_KEY_NORM_MAX) {
            b.add_f32(GGUF_KEY_NORM_MAX, cfg.norm_max);
        }
    }

    /// Builds a GGUF shaped like `convert_atst_file`'s output: arch + name +
    /// category + upstream URL, the full `vokra.atst.*` topology group, an
    /// optional weight-licence class, an optional FR-MD-09 attribution stamp,
    /// and the sample tensor manifest.
    fn atst_builder(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
    ) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        stamp_axis_group(&mut b, &atst_base_config(), None);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        }
        if attribution {
            b.add_string(
                chunks::KEY_PROVENANCE_ATTRIBUTION,
                "ATST (Audio-WestlakeU/audiossl) weights, licensed CC BY 4.0.",
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
        b
    }

    /// Parses an `atst_builder` result into a `GgufFile`.
    fn atst_gguf(
        weight_license_class: Option<LicenseClass>,
        attribution: bool,
        with_tensors: bool,
    ) -> GgufFile {
        let b = atst_builder(weight_license_class, attribution, with_tensors);
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Contract-constant pin (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        // Mirrors of `crates/vokra-convert/src/models/atst.rs`. A converter
        // drift without a binder-side follow-through lands here in the same
        // commit or fails this test.
        assert_eq!(ARCH, "atst", "arch tag pin");
        assert_eq!(NAME, "atst-base", "canonical size-point name pin");
        assert_eq!(CATEGORY, "audio-embedding", "category pin");
        assert_eq!(
            UPSTREAM_URL, "github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst",
            "upstream source tree pin (ATST is not on HuggingFace)"
        );
        assert_eq!(
            DEFAULT_LICENSE_SPDX, "cc-by-4.0",
            "the WEIGHT tier is cc-by-4.0 even though the CODE is mit"
        );
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            GGUF_KEY_PROVENANCE_UPSTREAM_URL,
            "vokra.provenance.upstream_url"
        );
        // The weight SPDX must resolve to the class the converter stamps.
        assert_eq!(
            LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX),
            LicenseClass::AttributionRequired,
            "cc-by-4.0 must classify as AttributionRequired"
        );
        // ... which is commercially usable but carries a display obligation.
        assert!(LicenseClass::AttributionRequired.commercial_ok());
        assert!(LicenseClass::AttributionRequired.requires_attribution());
        // Branch prefixes are stable.
        assert_eq!(AtstBranch::Student.prefix(), "student.");
        assert_eq!(AtstBranch::Teacher.prefix(), "teacher.");
        assert_eq!(
            AtstBranch::all(),
            [AtstBranch::Student, AtstBranch::Teacher]
        );
    }

    // -----------------------------------------------------------------------
    // 2. Arch-tag distinctness pin
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_ssl_encoder_arches() {
        // Every sibling below is a real converter arch tag. Sharing one would
        // let runtime dispatch bind a foreign topology over an ATST payload
        // (FR-EX-08).
        for sibling in [
            // SSL audio/music-embedding neighbourhood.
            "beats",
            "eat",
            "dasheng",
            "m2d",
            "maest",
            "mert",
            "muq",
            // Supervised audio tagging (not SSL at all).
            "yamnet",
            // wav2vec2 lineage — raw-waveform 1-D conv stem, not a log-mel
            // patch grid.
            "hubert",
            "wav2vec2_ctc",
            "wavlm_sv",
            "emotion2vec",
        ] {
            assert_ne!(
                ARCH, sibling,
                "atst must not share an arch tag with `{sibling}` — different \
                 pre-training objective means a different topology (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Metadata round-trip on a synthetic converter-shaped GGUF
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_a_synthetic_converter_shaped_gguf() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), true, true);
        let m = Atst::from_gguf(&file).expect("a converter-shaped GGUF must bind");

        // Metadata surfaces round-trip.
        assert_eq!(m.name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_url(), Some(UPSTREAM_URL));

        // Tensor manifest.
        assert_eq!(m.tensor_count(), SAMPLE_TENSORS.len());
        assert_eq!(m.weights().tensor_names().len(), SAMPLE_TENSORS.len());
        assert_eq!(
            m.weights()
                .tensor_dims("student.encoder.blocks.0.attn.qkv.weight"),
            Some([4usize, 12].as_slice())
        );

        // BYOL duo diagnostic: the sample manifest carries both branches.
        assert_eq!(m.branch_tensor_count(AtstBranch::Student), 2);
        assert_eq!(m.branch_tensor_count(AtstBranch::Teacher), 2);

        // Licence + FR-MD-09 attribution surfaces.
        assert_eq!(m.weight_license(), LicenseClass::AttributionRequired);
        let attr = m.attribution().expect("attribution stamp must surface");
        assert!(
            attr.contains("CC BY 4.0"),
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

        let Err(err) = Atst::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native atst GGUF"),
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
        // A `beats` GGUF (iterative acoustic tokenizer + MAM) handed to the
        // ATST binder by mistake. Both are `audio-embedding` SSL encoders, so
        // a silent bind would look plausible right up until the numbers are
        // wrong — exactly the misroute FR-EX-08 forbids.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "beats");
        b.add_string(chunks::KEY_MODEL_NAME, "beats-iter3-plus-as2m");
        b.add_tensor("beats.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Atst::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH the actual and the expected tag.
                assert!(
                    m.contains("`beats`"),
                    "message must name the arch actually found, got `{m}`"
                );
                assert!(
                    m.contains("`atst`"),
                    "message must name the expected arch, got `{m}`"
                );
                // The neighbourhood must be enumerated so the reader knows
                // which loader they actually wanted.
                for sibling in [
                    "eat",
                    "dasheng",
                    "m2d",
                    "maest",
                    "mert",
                    "muq",
                    "yamnet",
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
                    m.contains("teacher-student"),
                    "message should state what makes ATST distinct, got `{m}`"
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
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, false);
        let Err(err) = Atst::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model atst-base"),
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
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        let Err(err) = m
            .weights()
            .require_tensor("student.encoder.blocks.11.mlp.fc2.weight")
        else {
            panic!("expected ModelLoad for a tensor that is not on disk");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("student.encoder.blocks.11.mlp.fc2.weight"),
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
                .require_tensor("teacher.encoder.blocks.0.norm1.weight")
                .expect("present tensor must resolve"),
            [4usize, 1].as_slice()
        );
    }

    // -----------------------------------------------------------------------
    // 8. require_tensor_dims names BOTH expected and actual dims
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_dims_names_expected_and_actual() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        // Exact match passes.
        m.weights()
            .require_tensor_dims("student.encoder.blocks.0.attn.qkv.weight", &[4, 12])
            .expect("matching dims must pass");

        // Mismatch fails loud, naming both sides.
        let Err(err) = m
            .weights()
            .require_tensor_dims("student.encoder.blocks.0.attn.qkv.weight", &[4, 36])
        else {
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
    // 9. encode loud-partials, naming the missing primitive
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partials_naming_the_missing_primitive() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        // A legitimately shaped buffer, so the loud-partial gate is what
        // fires (not some pre-encode length validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("atst encode"), "surface must be named: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // --- The two RESOLVED blockers must be gone. A stale claim
                // --- sends the next reader to close an already-closed gap.
                assert!(
                    !msg.contains("vokra.atst.*"),
                    "the axis-group blocker is RESOLVED (the converter stamps the group \
                     and `AtstConfig` reads it) — the message must not still name it as \
                     missing: {msg}"
                );
                for resolved in [
                    "vokra_ops::conformer",
                    "vokra_ops::ebranchformer",
                    "vokra_ops::zipformer",
                ] {
                    assert!(
                        !msg.contains(resolved),
                        "the missing-primitive blocker is RESOLVED (`vokra_ops::vit` \
                         landed), so the message must no longer argue that `{resolved}` \
                         is not a substitute: {msg}"
                    );
                }
                assert!(
                    msg.contains("TWO blockers remain"),
                    "must state how many blockers actually remain: {msg}"
                );

                // --- Remaining blocker 1: the BYOL branch ambiguity.
                assert!(
                    msg.contains("student.") && msg.contains("teacher."),
                    "must name both BYOL branch prefixes: {msg}"
                );
                // --- Remaining blocker 2: no verified tensor-name manifest.
                assert!(
                    msg.contains("state_dict"),
                    "must name the unverified tensor-name manifest: {msg}"
                );
                assert!(
                    msg.contains("model.student."),
                    "must state what IS known about the real prefix, so the reader knows \
                     the fixtures' bare `student.` is not it: {msg}"
                );
                // --- And it must point at what already exists to check a
                // --- listing against, once one is read.
                assert!(
                    msg.contains("AtstConfig::from_gguf")
                        && msg.contains("vokra_ops::vit::ViTEncoder"),
                    "must name the pieces that ARE in place: {msg}"
                );

                // All three primary sources.
                for url in [
                    PRIMARY_SOURCE_UPSTREAM,
                    PRIMARY_SOURCE_PAPER_2022,
                    PRIMARY_SOURCE_PAPER_2023,
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
    // 10. embed loud-partials on the same blockers
    // -----------------------------------------------------------------------

    #[test]
    fn embed_loud_partials_on_the_same_blockers() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");

        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.embed(&pcm) else {
            panic!("embed must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("atst embed"), "surface must be named: {msg}");
                assert!(
                    msg.contains("pooled utterance embedding"),
                    "must name the output it refuses to fabricate: {msg}"
                );
                assert!(
                    !msg.contains("vokra.atst.*"),
                    "the axis-group blocker is RESOLVED — the message must not still name \
                     it as missing: {msg}"
                );
                assert!(
                    msg.contains("state_dict") && msg.contains("teacher."),
                    "must name the two blockers that actually remain: {msg}"
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
    // 11. Missing licence stamp fails closed to Unknown
    // -----------------------------------------------------------------------

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        // No provenance stamp at all: the binder still binds (arch + manifest
        // are the load gates), but the licence surface must fail closed.
        let file = atst_gguf(None, false, true);
        let m = Atst::from_gguf(&file).expect("arch + manifest are the load gates");
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
    // 12. Compliance gate: AttributionRequired passes strict, Unknown does not
    // -----------------------------------------------------------------------

    #[test]
    fn compliance_gate_passes_attribution_required_and_refuses_unknown() {
        // cc-by-4.0 -> AttributionRequired is commercially permitted, so a
        // correctly stamped artifact loads under the strict policy.
        let stamped = atst_builder(Some(LicenseClass::AttributionRequired), true, true)
            .to_bytes()
            .expect("serialize");
        let m = Atst::from_gguf_with_policy(&stamped, &CompliancePolicy::strict())
            .expect("AttributionRequired must pass the strict gate");
        assert_eq!(m.weight_license(), LicenseClass::AttributionRequired);

        // An unstamped artifact resolves to Unknown and is refused —
        // fail-closed, never a silent substitution.
        let unstamped = atst_builder(None, false, true)
            .to_bytes()
            .expect("serialize");
        let Err(err) = Atst::from_gguf_with_policy(&unstamped, &CompliancePolicy::strict()) else {
            panic!("an unstamped artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for an Unknown weight class, got {err:?}"
        );

        // The gate must not mask an arch mismatch: a foreign artifact reports
        // the arch, which is the actionable fact.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "dasheng");
        b.add_tensor("dasheng.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let foreign = b.to_bytes().expect("serialize");
        let Err(err) = Atst::from_gguf_with_policy(&foreign, &CompliancePolicy::strict()) else {
            panic!("a foreign arch must be refused");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains("`dasheng`") && msg.contains("`atst`"),
                "arch mismatch must be reported ahead of any licence verdict: {msg}"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 13. The stamped axis group round-trips into AtstConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_round_trips_every_stamped_axis() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");
        let cfg = m.config();
        let want = atst_base_config();

        // Whole-struct equality first: if a field is added to AtstConfig and
        // the reader forgets it, this fails before the per-field pins below.
        assert_eq!(cfg, &want, "every stamped axis must read back");

        // Per-field pins against the values the converter transcribes, so a
        // reader can see WHICH axis drifted rather than just "not equal".
        assert_eq!(cfg.embed_dim, 768);
        assert_eq!(cfg.depth, 12);
        assert_eq!(cfg.num_heads, 12);
        assert_eq!(cfg.mlp_ratio_scaled_1e3, 4_000);
        assert_eq!(cfg.mlp_hidden_dim, 3_072);
        assert_eq!(cfg.layer_norm_eps, 1e-6);
        assert!(!cfg.qkv_bias, "AST_base passes qkv_bias=False");
        assert!(
            cfg.use_cls,
            "use_cls=True is what makes pos_embed_len = 251"
        );
        assert_eq!(cfg.act_layer, ACT_LAYER_GELU);
        assert_eq!(cfg.in_chans, 1);
        assert_eq!(cfg.num_classes, 0, "no task head ships with the checkpoint");
        assert_eq!(cfg.drop_path_rate_scaled_1e3, 100);
        assert_eq!(cfg.patch_h, 64);
        assert_eq!(cfg.patch_w, 4);
        assert_eq!(cfg.spec_h, 64);
        assert_eq!(cfg.spec_w, 1001);
        assert_eq!(cfg.num_patches, 250);
        assert_eq!(cfg.pos_embed_len, 251);
        assert_eq!(cfg.pos_type, POS_TYPE_CUT);
        assert_eq!(cfg.patch_embed_in_features, 256);
        assert_eq!(cfg.patch_embed_kind, PATCH_EMBED_KIND_LINEAR);
        assert_eq!(cfg.patch_order, PATCH_ORDER_WH);
        assert_eq!(cfg.patch_grid, [1, 250]);
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.n_fft, 1024);
        assert_eq!(cfg.hop_length, 160);
        assert_eq!(cfg.win_length, 1024);
        assert_eq!(cfg.n_mels, 64);
        assert_eq!(cfg.f_min, 60);
        assert_eq!(cfg.f_max, 7800);
        assert_eq!(cfg.amp_to_db_top_db, 80);
        assert_eq!(cfg.amp_to_db_stype, "power");
        assert_eq!(cfg.norm_min, -79.6482_f32);
        assert_eq!(cfg.norm_max, 50.6842_f32);

        // Scaled-integer accessors undo the encoding.
        assert_eq!(cfg.mlp_ratio(), 4.0);
        assert_eq!(cfg.drop_path_rate(), 0.1);
        assert_eq!(cfg.n_prepended_tokens(), 1);
    }

    // -----------------------------------------------------------------------
    // 14. A GGUF missing ANY stamped key is loud and names that key
    // -----------------------------------------------------------------------

    /// A converter-shaped GGUF with exactly one topology key withheld.
    fn atst_gguf_missing(key: &str) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        stamp_axis_group(&mut b, &atst_base_config(), Some(key));
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
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    #[test]
    fn missing_topology_key_is_loud_and_names_it() {
        // One key from each GGUF type the group uses (u32 / f32 / bool /
        // string / indexed array), so a type-specific reader bug cannot hide.
        for key in [
            GGUF_KEY_EMBED_DIM,
            GGUF_KEY_DEPTH,
            GGUF_KEY_NUM_HEADS,
            GGUF_KEY_MLP_HIDDEN_DIM,
            GGUF_KEY_LAYER_NORM_EPS,
            GGUF_KEY_QKV_BIAS,
            GGUF_KEY_USE_CLS,
            GGUF_KEY_ACT_LAYER,
            GGUF_KEY_POS_TYPE,
            GGUF_KEY_PATCH_EMBED_KIND,
            GGUF_KEY_PATCH_ORDER,
            GGUF_KEY_PATCH_H,
            GGUF_KEY_SPEC_W,
            GGUF_KEY_POS_EMBED_LEN,
            GGUF_KEY_SAMPLE_RATE,
            GGUF_KEY_N_MELS,
            GGUF_KEY_NORM_MIN,
            GGUF_KEY_AMP_TO_DB_STYPE,
            "vokra.atst.patch_grid_0",
            "vokra.atst.patch_grid_1",
        ] {
            let file = atst_gguf_missing(key);
            let Err(err) = Atst::from_gguf(&file) else {
                panic!("expected ModelLoad when `{key}` is withheld");
            };
            match err {
                VokraError::ModelLoad(msg) => {
                    assert!(
                        msg.contains(key),
                        "the message must name the absent key `{key}`, got `{msg}`"
                    );
                    assert!(
                        msg.contains("FR-EX-08"),
                        "must cite the no-fallback clause for `{key}`, got `{msg}`"
                    );
                    assert!(
                        msg.contains("refuses to fall back"),
                        "must say it will not substitute a constant for `{key}`, got \
                         `{msg}`"
                    );
                }
                other => panic!("expected VokraError::ModelLoad for `{key}`, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // 15. The config maps onto ViTAttrs, and those attrs validate
    // -----------------------------------------------------------------------

    #[test]
    fn config_maps_onto_vit_attrs_that_validate() {
        let file = atst_gguf(Some(LicenseClass::AttributionRequired), false, true);
        let m = Atst::from_gguf(&file).expect("bind");
        let attrs = m
            .vit_attrs()
            .expect("the stamped axes must map onto ViTAttrs");

        // The primitive's own validator is the authority on whether this axis
        // set is coherent.
        attrs.validate().expect("mapped ViTAttrs must validate");

        // Stamped axes, carried straight across.
        assert_eq!(attrs.embed_dim, 768);
        assert_eq!(attrs.depth, 12);
        assert_eq!(attrs.n_heads, 12);
        assert_eq!(attrs.mlp_ratio, 4.0);
        assert_eq!(attrs.patch_h, 64);
        assert_eq!(attrs.patch_w, 4);
        assert_eq!(attrs.layer_norm_eps, 1e-6);

        // Derived axes.
        assert_eq!(
            (attrs.stride_h, attrs.stride_w),
            (64, 4),
            "upstream tiles the plane without overlap, so stride == patch"
        );
        assert_eq!(
            attrs.n_prepended_tokens, 1,
            "use_cls=True prepends exactly one token"
        );
        assert_eq!(
            attrs.gelu,
            GeluKind::Erf,
            "a bare nn.GELU is the exact erf formulation, not the tanh approximation"
        );
        assert_eq!(
            attrs.pos_embed_policy,
            PosEmbedPolicy::RequireExact,
            "pos_type=cut means slice; the binder cuts and the primitive then requires \
             an exact table"
        );

        // Consequences the primitive derives must agree with the stamps.
        assert_eq!(attrs.head_dim(), 64, "768 / 12");
        assert_eq!(
            attrs.mlp_dim(),
            3_072,
            "must equal the stamped mlp_hidden_dim"
        );
    }

    // -----------------------------------------------------------------------
    // 16. Internally inconsistent axis sets are refused, not accommodated
    // -----------------------------------------------------------------------

    /// Asserts `cfg.to_vit_attrs()` fails with a `ModelLoad` containing `needle`.
    fn assert_attrs_rejected(cfg: &AtstConfig, needle: &str) {
        let Err(err) = cfg.to_vit_attrs() else {
            panic!("expected ModelLoad for an artifact whose axes disagree ({needle})");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains(needle),
                "message should explain `{needle}`, got `{msg}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn to_vit_attrs_refuses_contradictory_axes() {
        // A grid the non-overlapping tiling formula cannot reproduce means
        // overlapping patches, whose stride is not stamped anywhere.
        assert_attrs_rejected(
            &AtstConfig {
                patch_grid: [1, 200],
                ..small_config()
            },
            "Refusing to guess a stride",
        );
        // num_patches must be the grid product.
        assert_attrs_rejected(
            &AtstConfig {
                num_patches: 99,
                ..small_config()
            },
            GGUF_KEY_NUM_PATCHES,
        );
        // pos_embed_len must be num_patches + the prepended block.
        assert_attrs_rejected(
            &AtstConfig {
                pos_embed_len: 4,
                ..small_config()
            },
            GGUF_KEY_POS_EMBED_LEN,
        );
        // The patch projection reads one flattened patch.
        assert_attrs_rejected(
            &AtstConfig {
                patch_embed_in_features: 7,
                ..small_config()
            },
            GGUF_KEY_PATCH_EMBED_IN_FEATURES,
        );
        // The encoder's plane height must equal the front-end's band count.
        assert_attrs_rejected(
            &AtstConfig {
                n_mels: 5,
                ..small_config()
            },
            GGUF_KEY_N_MELS,
        );
        // The stamped FFN width must match what the primitive resolves.
        assert_attrs_rejected(
            &AtstConfig {
                mlp_hidden_dim: 31,
                ..small_config()
            },
            GGUF_KEY_MLP_HIDDEN_DIM,
        );
        // Variants this binder cannot map are refused rather than defaulted.
        assert_attrs_rejected(
            &AtstConfig {
                act_layer: "silu".to_owned(),
                ..small_config()
            },
            GGUF_KEY_ACT_LAYER,
        );
        assert_attrs_rejected(
            &AtstConfig {
                pos_type: "interpolate".to_owned(),
                ..small_config()
            },
            GGUF_KEY_POS_TYPE,
        );
        assert_attrs_rejected(
            &AtstConfig {
                patch_embed_kind: "conv2d".to_owned(),
                ..small_config()
            },
            GGUF_KEY_PATCH_EMBED_KIND,
        );
        assert_attrs_rejected(
            &AtstConfig {
                patch_order: "zigzag".to_owned(),
                ..small_config()
            },
            GGUF_KEY_PATCH_ORDER,
        );
        // A zero axis cannot describe a real encoder.
        assert_attrs_rejected(
            &AtstConfig {
                patch_h: 0,
                ..small_config()
            },
            GGUF_KEY_PATCH_H,
        );
    }

    /// Upstream's `wh` token order and the primitive's `hw` order coincide
    /// only while the grid is one row tall. A taller grid must be refused, not
    /// silently transposed.
    #[test]
    fn to_vit_attrs_refuses_width_major_order_on_a_multi_row_grid() {
        // grid = [4/2, 8/2] = [2, 4] -> 8 patches, 9 table rows.
        let tall = AtstConfig {
            embed_dim: 8,
            depth: 2,
            num_heads: 2,
            mlp_hidden_dim: 32,
            patch_h: 2,
            patch_w: 2,
            spec_h: 4,
            spec_w: 8,
            num_patches: 8,
            pos_embed_len: 9,
            patch_embed_in_features: 4,
            patch_grid: [2, 4],
            n_mels: 4,
            ..atst_base_config()
        };
        assert_eq!(tall.patch_order, PATCH_ORDER_WH, "the base config is wh");
        assert_attrs_rejected(&tall, "transpose the token sequence");

        // The very same axes in the primitive's own order are fine, which
        // proves the refusal is about ORDER and not about the grid shape.
        let hw = AtstConfig {
            patch_order: PATCH_ORDER_HW.to_owned(),
            ..tall
        };
        let attrs = hw.to_vit_attrs().expect("hw order maps on any grid");
        assert_eq!((attrs.stride_h, attrs.stride_w), (2, 2));
    }

    // -----------------------------------------------------------------------
    // 17. cut_pos_embed implements pos_type="cut"
    // -----------------------------------------------------------------------

    #[test]
    fn cut_pos_embed_keeps_the_prepended_rows_then_a_patch_prefix() {
        let cfg = small_config();
        let d = cfg.embed_dim as usize;
        let rows = cfg.pos_embed_len as usize; // 5 = 1 CLS + 4 patch
        // Row r is filled with the value r, so the cut is easy to read off.
        let mut table: Vec<f32> = Vec::with_capacity(rows * d);
        for r in 0..rows {
            for _ in 0..d {
                table.push(r as f32);
            }
        }

        // Ask for 2 of the 4 patch rows.
        let cut = cfg.cut_pos_embed(&table, 2).expect("cut must succeed");
        assert_eq!(cut.len(), 3 * d, "1 prepended + 2 patch rows");
        assert_eq!(cut[0], 0.0, "the CLS row is carried through unchanged");
        assert_eq!(cut[d], 1.0, "then the FIRST patch row");
        assert_eq!(cut[2 * d], 2.0, "then the second");

        // Asking for the whole table is the identity.
        let all = cfg.cut_pos_embed(&table, 4).expect("full-length cut");
        assert_eq!(all, table);

        // Over-long requests are refused rather than padded, which would
        // fabricate positions.
        let Err(err) = cfg.cut_pos_embed(&table, 5) else {
            panic!("expected InvalidArgument when the cut runs off the end");
        };
        match err {
            VokraError::InvalidArgument(msg) => assert!(
                msg.contains("Refusing to pad"),
                "must refuse to pad, got `{msg}`"
            ),
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }

        // A wrong-length table is refused.
        let Err(err) = cfg.cut_pos_embed(&table[..d], 1) else {
            panic!("expected InvalidArgument on a wrong-length table");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // -----------------------------------------------------------------------
    // 18. Derived ViT tensor shapes
    // -----------------------------------------------------------------------

    #[test]
    fn vit_tensor_shapes_follow_from_the_stamped_axes() {
        let shapes = atst_base_config().vit_tensor_shapes();
        assert_eq!(shapes.patch_embed_weight, [768, 256], "nn.Linear(256, 768)");
        assert_eq!(shapes.patch_embed_bias, [768]);
        assert_eq!(
            shapes.pos_embed,
            [1, 251, 768],
            "torch.zeros(1, 250+1, 768)"
        );
        assert_eq!(
            shapes.prepended_tokens_elems, 768,
            "one CLS token of width D"
        );
        assert_eq!(
            shapes.block_qkv_weight,
            [2_304, 768],
            "nn.Linear(dim, dim*3) is ONE fused tensor, not three"
        );
        assert_eq!(
            shapes.block_qkv_bias, None,
            "qkv_bias=False means no bias tensor exists at all"
        );
        assert_eq!(shapes.block_proj_weight, [768, 768]);
        assert_eq!(shapes.block_fc1_weight, [3_072, 768]);
        assert_eq!(shapes.block_fc2_weight, [768, 3_072]);
        assert_eq!(shapes.norm_weight, [768]);
        assert_eq!(shapes.norm_bias, [768]);

        // The bias arm is reachable: flipping the stamped flag produces one.
        let biased = AtstConfig {
            qkv_bias: true,
            ..atst_base_config()
        };
        assert_eq!(biased.vit_tensor_shapes().block_qkv_bias, Some([2_304]));
    }

    // -----------------------------------------------------------------------
    // 19. verify_vit_tensor_shapes walks a caller-supplied name manifest
    // -----------------------------------------------------------------------
    //
    // The names below are INVENTED FOR THIS TEST. They are not a transcription
    // of ATST's real `state_dict` keys — that listing is the remaining blocker,
    // and the `t.` prefix is deliberately implausible so nobody mistakes these
    // for upstream truth. What the test proves is that the *walk* is correct:
    // given any manifest, every tensor is checked against the dims the
    // artifact's own stamped topology implies.

    /// An invented, self-consistent name manifest for [`small_config`].
    fn small_vit_names() -> AtstVitTensorNames {
        let blocks = (0..small_config().depth)
            .map(|i| AtstVitBlockTensorNames {
                norm1_weight: format!("t.blocks.{i}.norm1.weight"),
                norm1_bias: format!("t.blocks.{i}.norm1.bias"),
                qkv_weight: format!("t.blocks.{i}.attn.qkv.weight"),
                qkv_bias: None,
                proj_weight: format!("t.blocks.{i}.attn.proj.weight"),
                proj_bias: Some(format!("t.blocks.{i}.attn.proj.bias")),
                norm2_weight: format!("t.blocks.{i}.norm2.weight"),
                norm2_bias: format!("t.blocks.{i}.norm2.bias"),
                fc1_weight: format!("t.blocks.{i}.mlp.fc1.weight"),
                fc1_bias: Some(format!("t.blocks.{i}.mlp.fc1.bias")),
                fc2_weight: format!("t.blocks.{i}.mlp.fc2.weight"),
                fc2_bias: Some(format!("t.blocks.{i}.mlp.fc2.bias")),
            })
            .collect();
        AtstVitTensorNames {
            patch_embed_weight: "t.patch_embed.proj.weight".to_owned(),
            patch_embed_bias: Some("t.patch_embed.proj.bias".to_owned()),
            prepended_tokens: Some("t.cls_token".to_owned()),
            pos_embed: "t.pos_embed".to_owned(),
            blocks,
            final_norm_weight: "t.norm.weight".to_owned(),
            final_norm_bias: "t.norm.bias".to_owned(),
        }
    }

    /// `(name, dims)` pairs matching [`small_vit_names`] at the shapes
    /// [`small_config`] implies.
    fn small_vit_manifest() -> Vec<(String, Vec<u64>)> {
        /// `[usize]` shape to the GGUF-side `Vec<u64>` the builder wants.
        fn gd(dims: &[usize]) -> Vec<u64> {
            dims.iter().map(|&d| d as u64).collect()
        }
        /// One `(name, dims)` entry.
        fn e(name: &str, dims: &[usize]) -> (String, Vec<u64>) {
            (name.to_owned(), gd(dims))
        }
        /// Unwraps an optional name the fixture always supplies.
        fn req(name: &Option<String>, what: &str) -> String {
            name.clone().expect(what)
        }

        let cfg = small_config();
        let s = cfg.vit_tensor_shapes();
        let names = small_vit_names();
        let mut out: Vec<(String, Vec<u64>)> = vec![
            e(&names.patch_embed_weight, &s.patch_embed_weight),
            e(
                &req(&names.patch_embed_bias, "a patch-embed bias"),
                &s.patch_embed_bias,
            ),
            e(&names.pos_embed, &s.pos_embed),
            // Rank 3 on purpose: the walk asserts the ELEMENT COUNT of the
            // prepended tokens, never a rank, because upstream's CLS
            // allocation is not transcribed.
            e(
                &req(&names.prepended_tokens, "a CLS token"),
                &[1, 1, cfg.embed_dim as usize],
            ),
        ];
        for b in &names.blocks {
            out.push(e(&b.norm1_weight, &s.norm_weight));
            out.push(e(&b.norm1_bias, &s.norm_bias));
            out.push(e(&b.qkv_weight, &s.block_qkv_weight));
            out.push(e(&b.proj_weight, &s.block_proj_weight));
            out.push(e(&req(&b.proj_bias, "a proj bias"), &s.block_proj_bias));
            out.push(e(&b.norm2_weight, &s.norm_weight));
            out.push(e(&b.norm2_bias, &s.norm_bias));
            out.push(e(&b.fc1_weight, &s.block_fc1_weight));
            out.push(e(&req(&b.fc1_bias, "an fc1 bias"), &s.block_fc1_bias));
            out.push(e(&b.fc2_weight, &s.block_fc2_weight));
            out.push(e(&req(&b.fc2_bias, "an fc2 bias"), &s.block_fc2_bias));
        }
        out.push(e(&names.final_norm_weight, &s.norm_weight));
        out.push(e(&names.final_norm_bias, &s.norm_bias));
        out
    }

    /// Binds a miniature ATST artifact carrying `manifest`.
    fn small_atst(manifest: &[(String, Vec<u64>)]) -> Atst {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        stamp_axis_group(&mut b, &small_config(), None);
        for (name, dims) in manifest {
            let elems: u64 = dims.iter().product();
            b.add_tensor(
                name,
                GgmlType::F32,
                dims.clone(),
                vec![0u8; (elems * 4) as usize],
            )
            .expect("add_tensor");
        }
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        Atst::from_gguf(&file).expect("the miniature artifact must bind")
    }

    #[test]
    fn verify_vit_tensor_shapes_accepts_a_consistent_manifest() {
        let m = small_atst(&small_vit_manifest());
        assert_eq!(m.config(), &small_config());
        m.vit_attrs()
            .expect("the miniature axis set must map onto ViTAttrs")
            .validate()
            .expect("and validate");
        m.verify_vit_tensor_shapes(&small_vit_names())
            .expect("a manifest matching the derived shapes must verify");
    }

    #[test]
    fn verify_vit_tensor_shapes_rejects_a_wrong_shaped_tensor() {
        // Transpose one FFN weight: [8, 32] -> [32, 8]. Same element count, so
        // only a dims check catches it — precisely the silent reshape FR-EX-08
        // forbids.
        let target = "t.blocks.1.mlp.fc2.weight";
        let manifest: Vec<(String, Vec<u64>)> = small_vit_manifest()
            .into_iter()
            .map(|(n, d)| {
                if n == target {
                    (n, vec![32, 8])
                } else {
                    (n, d)
                }
            })
            .collect();
        let m = small_atst(&manifest);

        let Err(err) = m.verify_vit_tensor_shapes(&small_vit_names()) else {
            panic!("expected ModelLoad when a bound tensor is transposed");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains(target), "must name the tensor, got `{msg}`");
                assert!(msg.contains("[32, 8]"), "must name the ACTUAL dims: {msg}");
                assert!(
                    msg.contains("[8, 32]"),
                    "must name the EXPECTED dims: {msg}"
                );
                assert!(msg.contains("FR-EX-08"), "must cite the clause: {msg}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn verify_vit_tensor_shapes_rejects_an_absent_tensor() {
        let target = "t.blocks.0.attn.qkv.weight";
        let manifest: Vec<(String, Vec<u64>)> = small_vit_manifest()
            .into_iter()
            .filter(|(n, _)| n != target)
            .collect();
        let m = small_atst(&manifest);

        let Err(err) = m.verify_vit_tensor_shapes(&small_vit_names()) else {
            panic!("expected ModelLoad when a named tensor is not on disk");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains(target), "must name the tensor, got `{msg}`");
                assert!(
                    msg.contains("nearest names on disk"),
                    "must offer nearby names: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn verify_vit_tensor_shapes_honours_the_stamped_qkv_bias_flag() {
        // The artifact stamps qkv_bias=false, so naming a QKV bias must be
        // refused rather than checked: upstream's Linear carries none.
        let mut names = small_vit_names();
        names.blocks[0].qkv_bias = Some("t.blocks.0.attn.qkv.bias".to_owned());
        let m = small_atst(&small_vit_manifest());

        let Err(err) = m.verify_vit_tensor_shapes(&names) else {
            panic!("expected ModelLoad when a manifest names a bias the topology denies");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains(GGUF_KEY_QKV_BIAS) && msg.contains("bias=False"),
                "must point at the stamped flag, got `{msg}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn verify_vit_tensor_shapes_rejects_a_short_block_list() {
        let mut names = small_vit_names();
        names.blocks.truncate(1);
        let m = small_atst(&small_vit_manifest());

        let Err(err) = m.verify_vit_tensor_shapes(&names) else {
            panic!("expected ModelLoad when the manifest is shorter than the stack");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains("1 block(s)") && msg.contains("depth 2"),
                "must name both counts, got `{msg}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn verify_vit_tensor_shapes_requires_a_prepended_token_when_use_cls_is_set() {
        let mut names = small_vit_names();
        names.prepended_tokens = None;
        let m = small_atst(&small_vit_manifest());

        let Err(err) = m.verify_vit_tensor_shapes(&names) else {
            panic!("expected ModelLoad when use_cls is set but no CLS tensor is named");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains(GGUF_KEY_USE_CLS) && msg.contains("Refusing to prepend a zero"),
                "must refuse to prepend a zero token, got `{msg}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }
}
