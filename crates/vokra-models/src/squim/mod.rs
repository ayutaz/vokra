//! **TorchAudio-SQUIM** (`pytorch/audio`, BSD-2-Clause code) — runtime binder
//! for the `torchaudio_squim` converter arch (Wave A 2026-08-15, loud-partial
//! per the `dnsmos_p808_p835` / `emotion2vec` / `sepformer` / RMVPE precedent —
//! CLAUDE.md 教訓 (a): "loud-partial は fake-complete より honest").
//!
//! Before this module existed the converter
//! (`crates/vokra-convert/src/models/torchaudio_squim.rs`) produced a GGUF that
//! **nothing in the workspace could read** — the `torchaudio_squim` arch string
//! had no consumer, so every converted bundle was unloadable. This module is
//! that consumer.
//!
//! # Primary sources
//!
//! - Paper: Kumar, Tan, Ni, Manocha, Zhang, Henderson & Xu, *"TorchAudio-Squim:
//!   Reference-less Speech Quality and Intelligibility Measures in TorchAudio"*,
//!   ICASSP 2023 (<https://arxiv.org/abs/2304.01448>).
//! - Reference implementation (upstream module tree):
//!   <https://github.com/pytorch/audio/blob/main/src/torchaudio/models/squim/objective.py>
//!   and `.../squim/subjective.py`.
//! - Repository LICENSE (code): BSD-2-Clause,
//!   <https://github.com/pytorch/audio/blob/main/LICENSE>.
//! - In-repo bundle contract: `tools/parity/torchaudio_squim_prepare_checkpoint.py`
//!   (the offline pickle-bridge sidecar; FR-LD-05 — no torch, no pickle, no
//!   ONNX ever enters the runtime). Every hyperparameter and tensor-prefix
//!   claim below is transcribed from that sidecar, which records having
//!   verified them against the two upstream `squim/{objective,subjective}.py`
//!   factory functions on 2026-08-04.
//!
//! # Two heads, two different tasks — deliberately NOT conflated
//!
//! SQUIM ships as **two independently-trained bundles**. They are separate
//! entry points here ([`Squim::estimate_objective`] /
//! [`Squim::estimate_subjective`]) because they answer different questions and
//! take different inputs:
//!
//! | Head | Upstream bundle | Input | Output |
//! |---|---|---|---|
//! | **Objective** | `squim_objective_dns2020.pth` | degraded waveform **only** | STOI + PESQ + SI-SDR |
//! | **Subjective** | `squim_subjective_bvcc_daps.pth` | degraded waveform **+ a non-matching reference** | MOS |
//!
//! *Objective* is fully reference-free. *Subjective* is **not** reference-free
//! in the same sense: it needs a **non-matching reference** (NMR) — an
//! unrelated clean utterance that supplies a quality anchor without being the
//! paired clean source of the degraded input. A caller that passes the clean
//! source as the NMR is outside the trained regime; a caller that passes
//! nothing at all cannot use this head. Folding the two into one "score()" call
//! would erase that distinction, which is why they stay separate.
//!
//! # Relationship to `vokra-eval` (complementary, not duplicated)
//!
//! The sibling `vokra-eval` work landing in this same wave adds **computed**
//! SI-SNR / SI-SDR / STOI: those take a hypothesis waveform **and its paired
//! clean reference** and evaluate a closed-form formula. SQUIM Objective
//! **estimates** the very same quantities from the degraded waveform **alone**,
//! with a trained network standing in for the missing reference.
//!
//! They are therefore complements, not duplicates, and the pairing is useful:
//! on a corpus where a clean reference *does* exist, `vokra-eval`'s computed
//! SI-SDR / STOI are the ground truth against which this estimator's error can
//! be measured — that is exactly the calibration experiment Kumar et al. 2023
//! report. Nothing here re-implements the closed-form metrics, and nothing in
//! `vokra-eval` re-implements the estimator.
//!
//! # Runtime layout
//!
//! ```text
//! Objective head (STOI + PESQ + SI-SDR):
//!   degraded PCM (mono f32, 16 kHz)
//!     -> learnable 1-D encoder                          ← **loud-partial**
//!     -> DPRNN block stack                              ← **loud-partial**
//!          (segment/chunk -> intra-chunk RNN -> inter-chunk
//!           RNN -> overlap-add; num_blocks = 2, chunk_size
//!           = 71, feat_dim = 256 per the sidecar's recorded
//!           `squim_objective_base()` defaults. The RNN CELL
//!           TYPE and the per-block composition are NOT
//!           recorded anywhere in-repo and must be read off
//!           upstream `objective.py`.)
//!     -> 3 transformer metric heads                     ← **loud-partial**
//!          (d_model = 256, nhead = 4; one head per metric in
//!           the load-bearing order [`OBJECTIVE_METRICS`].)
//!     -> (stoi, pesq, si_sdr)
//!
//! Subjective head (MOS):
//!   degraded PCM + non-matching-reference PCM (both mono f32, 16 kHz)
//!     -> wav2vec2_base SSL encoder                      ← **loud-partial**
//!          (shared gap with `crate::emotion2vec`, `crate::wavlm`
//!           and the wav2vec2-lineage fleet.)
//!     -> attentive pooling (att_dim = 5)                ← **loud-partial**
//!     -> linear projector (feat_dim 768 -> proj_dim 32)  ← **loud-partial**
//!     -> MOS scalar
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`Squim::from_gguf`] with **strict** `vokra.model.arch ==
//!     "torchaudio_squim"` verification. A foreign GGUF — including every
//!     sibling `category = "eval"` quality-metric arch (`utmos` / `utmosv2` /
//!     `dnsmos` / `nisqa_v2_weight`) — is refused with a message naming both
//!     the observed and expected tags plus the whole sibling fleet, never
//!     silently mis-bound (FR-EX-08).
//!   - [`SquimWeights::from_gguf`]: real bundle-prefix routing over the tensor
//!     manifest (`objective.` / `subjective.`), an empty-manifest refusal, a
//!     no-head-discoverable refusal, and a refusal naming any tensor that
//!     escapes both prefixes (the sidecar's prefix invariant is load-bearing —
//!     see "Tensor naming contract" below).
//!   - [`SquimConfig::from_gguf`]: optional `vokra.squim.sample_rate`
//!     validation against [`EXPECTED_SAMPLE_RATE`], with an explicit
//!     [`ConfigSource`] record so a caller can tell a *stamped* config from a
//!     *factory-default* one rather than being silently told they are the same.
//!   - Per-head reachability gates: asking a subjective-only bundle for STOI /
//!     PESQ / SI-SDR is a loud [`VokraError::InvalidArgument`] naming the
//!     missing `objective.` prefix — never a fabricated zero.
//!   - Input validation: an empty PCM slice (either argument) is rejected
//!     before the loud-partial fires, so the argument contract is enforced
//!     today and will not change shape when the forward lands.
//!   - Weight-license class surfacing, fail-closed to
//!     [`LicenseClass::Unknown`] when the stamp is absent.
//!
//! - **Loud-partial (this WP)**: [`Squim::estimate_objective`] and
//!   [`Squim::estimate_subjective`] each return
//!   [`VokraError::UnsupportedOp`] naming their own missing primitives, their
//!   own primary-source file, and the `vokra.squim.*` metadata chunk a
//!   follow-up wave must stamp. **No score value is ever fabricated** — no
//!   `0.0` STOI, no `1.0` MOS, no clamped placeholder (FR-EX-08).
//!
//! ## Why the forwards are deferred rather than best-guessed
//!
//! Neither head's op sequence is primary-source-transcribable from anything
//! currently in this repository. The sidecar records **scalar
//! hyperparameters** (`feat_dim`, `d_model`, `nhead`, `num_blocks`,
//! `chunk_size`, `ssl_type`, `proj_dim`, `att_dim`) but **not** the op order,
//! and the converter is a dtype-agnostic pass-through that never inspects a
//! graph. Writing a best-guess DPRNN or a best-guess attentive-pool would be
//! silent-wrong: it would return plausible floats that are not SQUIM's, and
//! nothing downstream could tell. A loud refusal that names the exact file to
//! read is strictly more useful, and the surrounding scaffold (routing,
//! validation, licence surfacing) is real and testable today.
//!
//! Separately, the DPRNN stack needs a **general RNN primitive that
//! `vokra-ops` does not have**: the only recurrent kernels in the tree are the
//! DFN3-specific GRU stack embedded in `vokra_ops::denoise`, which is bound to
//! DeepFilterNet3's own tensor layout and is not a reusable cell. See
//! [`missing_primitive_note`] for the note carried in the error text.
//!
//! # Tensor naming contract (the sidecar is authoritative)
//!
//! `tools/parity/torchaudio_squim_prepare_checkpoint.py` merges the two
//! upstream `state_dict`s into one safetensors bundle, prefixing every key with
//! its bundle tag — `objective.<upstream_name>` / `subjective.<upstream_name>` —
//! exactly analogous to the DNSMOS `p808.` / `p835.` convention. Its own
//! "Prefix policy" note reads: *"The Rust binder walks the leading dot-segment
//! to route the tensor — never rewrite these because any mangling would need to
//! travel with the binder to stay consistent."* This module is that binder, and
//! [`SquimHead::tensor_prefix`] is that dot-segment.
//!
//! **Known doc drift (follow-up, not a code bug)**: the converter's own module
//! docstring describes tensor names as `ssl_encoder.*` / `head.*`, omitting the
//! bundle prefix the sidecar adds, and its unit-test fixtures use unprefixed
//! synthetic names. Those fixtures exercise the converter's dtype pass-through
//! (which is name-agnostic), so they are not wrong about what the converter
//! does — but they do not describe a real bundle. The sidecar wins because it
//! is the tool that actually writes the names. A follow-up should correct the
//! converter docstring to `objective.<upstream>` / `subjective.<upstream>`.
//!
//! # `vokra.squim.*` chunk group — reserved, not yet stamped
//!
//! The converter stamps only `vokra.model.{arch,name,category}` and
//! `vokra.provenance.*`. It writes **no** `vokra.squim.*` chunk, so the
//! sidecar's `--config-out` JSON (sample rate + factory hyperparameters) never
//! reaches the GGUF. The key names are reserved here
//! ([`KEY_SQUIM_BUNDLE`], [`KEY_SQUIM_SAMPLE_RATE`],
//! [`KEY_SQUIM_OBJECTIVE_TOPOLOGY`], [`KEY_SQUIM_SUBJECTIVE_TOPOLOGY`]) and are
//! read **optionally** so that stamping them later is a purely additive change
//! — mirroring the DNSMOS `vokra.dnsmos.{p808,p835}.topology` posture.
//!
//! # Weight licence divergence (surfaced, NOT signed off)
//!
//! The *code* is BSD-2-Clause, and that is what the converter stamps by
//! default. The *weights* are not uniform: the sidecar records
//! `squim_objective_dns2020.pth` as **CC-BY-4.0** (attribution) and
//! `squim_subjective_bvcc_daps.pth` as **CC-BY-NC-4.0** (non-commercial), per
//! the upstream tutorial page as it verified them on 2026-08-04.
//!
//! That divergence matters for *distribution*, not for loading: a bundle
//! carrying the subjective head is non-commercial-encumbered even though its
//! stamp reads `bsd-2-clause`. [`Squim::weight_license`] surfaces whatever the
//! GGUF actually claims and [`Squim::has_subjective`] tells a publish gate that
//! the NC-encumbered head is present. **This module makes no licence
//! determination** — `docs/license-audit.md` §3.1 sign-off is owner-only and is
//! deliberately left untouched.
//!
//! # Why `MosScorerEngine` is deliberately NOT implemented
//!
//! `vokra_core::engines::MosScorerEngine` exists and this model does emit a
//! MOS, so implementing it looks tempting. It is refused on purpose:
//! `MosScore`'s fields are DNSMOS-shaped (`p808` / `sig` / `bak` / `ovrl`),
//! each naming a specific ITU-T listening-test protocol. SQUIM Subjective's MOS
//! is trained on BVCC + DAPS and is **not** a P.808 score, so surfacing it as
//! `MosScore::p808` would be a silent misrepresentation of which protocol
//! produced the number — precisely the class of quiet lie FR-EX-08 exists to
//! prevent. Worse, the trait's `score(&self, pcm16k)` signature has nowhere to
//! put the non-matching reference the head requires, so an impl would have to
//! invent one. If a shared MOS surface is wanted later, the correct move is to
//! widen `MosScore` with a protocol-tagged variant, not to squat an existing
//! field.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_URL`] /
//! [`DEFAULT_LICENSE_SPDX`] mirror the converter's `pub const` surface. Same
//! rule every sibling binder follows so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layering `vokra-ops →
//! nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
//! binder`, `vokra-convert → GGUF writer`.
//!
//! # Why this binder lives in `vokra-models`, not `vokra-eval`
//!
//! The sidecar's docstring anticipates a `vokra_eval::squim::from_gguf` binder.
//! It landed here instead, alongside its closest sibling
//! [`crate::dnsmos_p808_p835`] — which is also `category = "eval"` and also
//! binds in `vokra-models`. The split is by *layer*, not by category:
//! `vokra-models` binds GGUF-backed neural models, `vokra-eval` holds the
//! `Metric` traits and the weight-free algorithmic metrics. A weight-driven
//! estimator belongs on the model side of that line.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/torchaudio_squim.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model torchaudio-squim`.
///
/// Deliberately distinct from every sibling `category = "eval"` quality-metric
/// arch tag (`utmos` / `utmosv2` / `dnsmos` / `nisqa_v2_weight`): SQUIM's
/// reference-free multi-metric topology has no analogue in any of them, and
/// silently sharing an arch would mis-route runtime dispatch onto a loader
/// expecting a completely different tensor tree (FR-EX-08).
pub const ARCH: &str = "torchaudio_squim";

/// Expected `vokra.model.name` value written by the converter.
pub const NAME: &str = "torchaudio_squim";

/// Expected `vokra.model.category` value — `"eval"`, the reference-free
/// quality-metric tier shared with `utmos` / `dnsmos`.
pub const CATEGORY: &str = "eval";

/// Primary redistribution source. SQUIM ships via `torch.hub`, not the HF hub,
/// so the converter stamps [`KEY_PROVENANCE_UPSTREAM_URL`] rather than
/// `vokra.provenance.upstream_hf`.
pub const UPSTREAM_URL: &str = "github.com/pytorch/audio";

/// Default upstream **code** licence (SPDX) the converter stamps. See the
/// module docstring's "Weight licence divergence" section — the *weights* are
/// not uniformly BSD-2-Clause.
pub const DEFAULT_LICENSE_SPDX: &str = "bsd-2-clause";

// ---- GGUF metadata keys ---------------------------------------------------

/// GGUF metadata key: model category tag (mirror of the converter's local
/// constant — not yet centralised in `vokra_core::gguf::chunks`).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// GGUF metadata key: primary redistribution source URL, used instead of
/// `vokra.provenance.upstream_hf` for models with no HF mirror.
pub const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Reserved GGUF metadata key: bundle inventory (`Array<String>`, canonical
/// order `["objective", "subjective"]`). **Not stamped by the current
/// converter** — head presence is derived from the tensor prefixes instead.
/// Reserved so a follow-up wave can stamp it additively.
pub const KEY_SQUIM_BUNDLE: &str = "vokra.squim.bundle";

/// Reserved GGUF metadata key: PCM sample rate (u32 Hz). **Optional today** —
/// when present it is validated against [`EXPECTED_SAMPLE_RATE`]; when absent
/// the config records [`ConfigSource::FactoryDefaults`].
pub const KEY_SQUIM_SAMPLE_RATE: &str = "vokra.squim.sample_rate";

/// Reserved GGUF metadata key: objective-head op-token sequence
/// (`Array<u32>`). Absent today; stamping it is the first half of flipping
/// [`Squim::estimate_objective`] out of loud-partial. Mirrors the DNSMOS
/// `vokra.dnsmos.p808.topology` posture.
pub const KEY_SQUIM_OBJECTIVE_TOPOLOGY: &str = "vokra.squim.objective.topology";

/// Reserved GGUF metadata key: subjective-head op-token sequence
/// (`Array<u32>`). Same role as [`KEY_SQUIM_OBJECTIVE_TOPOLOGY`] for the MOS
/// head.
pub const KEY_SQUIM_SUBJECTIVE_TOPOLOGY: &str = "vokra.squim.subjective.topology";

// ---- Model constants ------------------------------------------------------

/// Sample rate both SQUIM bundles are trained at (Hz). Recorded as the
/// canonical value by the in-repo sidecar, which additionally refuses to emit a
/// bundle whose two halves disagree on it.
///
/// A differently-rated GGUF is either mis-produced or a non-canonical fork —
/// resample upstream rather than binding it here (FR-EX-08).
pub const EXPECTED_SAMPLE_RATE: u32 = 16_000;

/// The three metrics the objective head predicts, in the **load-bearing**
/// order the sidecar's config JSON records for `squim_objective_base()`.
///
/// The order is load-bearing because the head is a 3-way branch: index `i`
/// corresponds to `OBJECTIVE_METRICS[i]`, and a silent reorder would swap a
/// dB-scale SI-SDR with a unitless STOI without any type change to catch it.
pub const OBJECTIVE_METRICS: [&str; 3] = ["stoi", "pesq", "sisdr"];

/// The single metric the subjective head predicts.
pub const SUBJECTIVE_METRICS: [&str; 1] = ["mos"];

/// Objective head: encoder bottleneck width (`feat_dim`), from the sidecar's
/// recorded `squim_objective_base()` factory defaults.
pub const OBJECTIVE_FEAT_DIM: u32 = 256;

/// Objective head: transformer metric-head width (`d_model`).
pub const OBJECTIVE_D_MODEL: u32 = 256;

/// Objective head: transformer metric-head attention heads (`nhead`).
pub const OBJECTIVE_NHEAD: u32 = 4;

/// Objective head: number of DPRNN blocks (`num_blocks`).
pub const OBJECTIVE_NUM_BLOCKS: u32 = 2;

/// Objective head: DPRNN chunking factor (`chunk_size`).
pub const OBJECTIVE_CHUNK_SIZE: u32 = 71;

/// Subjective head: SSL feature extractor type (`ssl_type`). The same
/// wav2vec2-base lineage `crate::emotion2vec` and `crate::wavlm` also defer on.
pub const SUBJECTIVE_SSL_TYPE: &str = "wav2vec2_base";

/// Subjective head: SSL feature width (`feat_dim`).
pub const SUBJECTIVE_FEAT_DIM: u32 = 768;

/// Subjective head: projector output width (`proj_dim`).
pub const SUBJECTIVE_PROJ_DIM: u32 = 32;

/// Subjective head: attentive-pool width (`att_dim`).
pub const SUBJECTIVE_ATT_DIM: u32 = 5;

// ---- Primary-source anchors (cited verbatim in loud-partial errors) -------

/// Primary-source anchor: the ICASSP 2023 paper.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2304.01448";

/// Primary-source anchor: the upstream objective-head module.
pub const PRIMARY_SOURCE_CODE_OBJECTIVE: &str =
    "github.com/pytorch/audio/blob/main/src/torchaudio/models/squim/objective.py";

/// Primary-source anchor: the upstream subjective-head module.
pub const PRIMARY_SOURCE_CODE_SUBJECTIVE: &str =
    "github.com/pytorch/audio/blob/main/src/torchaudio/models/squim/subjective.py";

/// In-repo anchor: the offline pickle-bridge sidecar that defines the bundle
/// prefix contract and records the factory hyperparameters.
pub const PREPARE_CHECKPOINT_SIDECAR: &str = "tools/parity/torchaudio_squim_prepare_checkpoint.py";

/// Sibling `category = "eval"` arch tags enumerated in the wrong-arch
/// diagnostic so a reader who mis-routed a quality-metric GGUF sees the whole
/// neighbourhood at once.
pub const SIBLING_EVAL_ARCHES: [&str; 4] = ["utmos", "utmosv2", "dnsmos", "nisqa_v2_weight"];

// ---------------------------------------------------------------------------
// SquimHead
// ---------------------------------------------------------------------------

/// Which of the two SQUIM bundles a tensor or a call refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SquimHead {
    /// `squim_objective_dns2020.pth` — reference-**free** STOI + PESQ + SI-SDR
    /// estimation from a degraded waveform alone.
    Objective,
    /// `squim_subjective_bvcc_daps.pth` — MOS estimation against a
    /// **non-matching reference** (an unrelated clean utterance).
    Subjective,
}

impl SquimHead {
    /// Canonical short name for logs, metadata and diagnostics.
    #[must_use]
    pub const fn short(self) -> &'static str {
        match self {
            Self::Objective => "objective",
            Self::Subjective => "subjective",
        }
    }

    /// The GGUF tensor-name prefix this head's weights carry.
    ///
    /// This is the leading dot-segment the sidecar prepends
    /// (`objective.<upstream_name>` / `subjective.<upstream_name>`); see the
    /// module docstring's "Tensor naming contract" section for why it must
    /// never be rewritten on either side.
    #[must_use]
    pub const fn tensor_prefix(self) -> &'static str {
        match self {
            Self::Objective => "objective.",
            Self::Subjective => "subjective.",
        }
    }

    /// The upstream checkpoint filename this head is flattened from — echoed
    /// in diagnostics so a reader can identify which `.pth` is missing.
    #[must_use]
    pub const fn upstream_checkpoint(self) -> &'static str {
        match self {
            Self::Objective => "squim_objective_dns2020.pth",
            Self::Subjective => "squim_subjective_bvcc_daps.pth",
        }
    }

    /// The reserved topology metadata key that will pin this head's op-token
    /// sequence (see [`KEY_SQUIM_OBJECTIVE_TOPOLOGY`]).
    #[must_use]
    pub const fn topology_key(self) -> &'static str {
        match self {
            Self::Objective => KEY_SQUIM_OBJECTIVE_TOPOLOGY,
            Self::Subjective => KEY_SQUIM_SUBJECTIVE_TOPOLOGY,
        }
    }

    /// The upstream reference-implementation file to transcribe when flipping
    /// this head out of loud-partial.
    #[must_use]
    pub const fn primary_source_code(self) -> &'static str {
        match self {
            Self::Objective => PRIMARY_SOURCE_CODE_OBJECTIVE,
            Self::Subjective => PRIMARY_SOURCE_CODE_SUBJECTIVE,
        }
    }
}

// ---------------------------------------------------------------------------
// Score types
// ---------------------------------------------------------------------------

/// The objective head's three reference-free estimates.
///
/// No range clamping or normalisation is applied on the (future) real path:
/// these are whatever the upstream heads emit, in the upstream units —
/// `stoi` is unitless short-time objective intelligibility, `pesq` is on the
/// wide-band PESQ MOS-LQO scale, and `si_sdr` is in dB. Deriving a verdict
/// from them is the caller's concern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquimObjectiveScores {
    /// Estimated short-time objective intelligibility (unitless).
    pub stoi: f32,
    /// Estimated perceptual evaluation of speech quality (wide-band MOS-LQO
    /// scale).
    pub pesq: f32,
    /// Estimated scale-invariant signal-to-distortion ratio (dB).
    pub si_sdr: f32,
}

/// Where a bound [`SquimConfig`]'s values came from.
///
/// Recorded explicitly so a caller can distinguish "the GGUF told me 16 kHz"
/// from "nobody told me anything and I assumed the documented default". Those
/// are different epistemic states and collapsing them would be exactly the kind
/// of quiet assumption FR-EX-08 targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    /// The `vokra.squim.*` chunk group was present and was read verbatim.
    GgufChunk,
    /// The chunk group was absent (the current converter stamps none of it),
    /// so the documented `squim_{objective,subjective}_base()` factory
    /// defaults were used.
    FactoryDefaults,
}

// ---------------------------------------------------------------------------
// SquimConfig
// ---------------------------------------------------------------------------

/// Runtime config for a bound SQUIM bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SquimConfig {
    /// PCM sample rate both heads expect (Hz). Always [`EXPECTED_SAMPLE_RATE`]
    /// — a stamped value that disagrees is refused at load.
    pub sample_rate: u32,
    /// Whether the objective (STOI + PESQ + SI-SDR) head is present.
    pub has_objective: bool,
    /// Whether the subjective (MOS) head is present.
    pub has_subjective: bool,
    /// Whether the above came from a stamped chunk or from documented
    /// defaults.
    pub source: ConfigSource,
}

impl SquimConfig {
    /// Validates the config loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when neither head is present, or when the
    ///   sample rate is not [`EXPECTED_SAMPLE_RATE`].
    pub fn validate(&self) -> Result<()> {
        if !self.has_objective && !self.has_subjective {
            return Err(VokraError::ModelLoad(format!(
                "torchaudio_squim: the bundle advertises neither head — a GGUF with no \
                 `objective.` and no `subjective.` tensors is not a SQUIM bundle. Re-run \
                 `{PREPARE_CHECKPOINT_SIDECAR}` (it refuses to emit an empty bundle) and \
                 then `vokra-cli convert --model torchaudio-squim` (FR-EX-08 forbids \
                 binding a headless bundle)."
            )));
        }
        if self.sample_rate != EXPECTED_SAMPLE_RATE {
            return Err(VokraError::ModelLoad(format!(
                "torchaudio_squim: `{KEY_SQUIM_SAMPLE_RATE}` = {} — both SQUIM bundles are \
                 trained at {EXPECTED_SAMPLE_RATE} Hz only (the sidecar \
                 `{PREPARE_CHECKPOINT_SIDECAR}` pins the canonical rate and refuses a \
                 bundle whose halves disagree). Resample the audio upstream rather than \
                 emitting a different-rate GGUF (FR-EX-08).",
                self.sample_rate,
            )));
        }
        Ok(())
    }

    /// Reads the config from a parsed GGUF.
    ///
    /// The `vokra.squim.*` chunk group is **optional today** — the current
    /// converter stamps none of it, so an absent [`KEY_SQUIM_SAMPLE_RATE`]
    /// yields [`ConfigSource::FactoryDefaults`] rather than an error. A
    /// *present but wrong* value is still a hard failure: an explicit claim
    /// that contradicts the model is worse than no claim at all.
    ///
    /// `has_objective` / `has_subjective` are supplied by the caller from the
    /// tensor-prefix walk ([`SquimWeights::from_gguf`]) because tensor presence
    /// — not metadata — is the ground truth for which heads can actually run.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when a stamped sample rate does not fit in
    ///   `u32`, or when [`Self::validate`] rejects the result.
    pub fn from_gguf(gguf: &GgufFile, has_objective: bool, has_subjective: bool) -> Result<Self> {
        let (sample_rate, source) = match gguf.get(KEY_SQUIM_SAMPLE_RATE).and_then(|v| v.as_u64()) {
            Some(raw) => {
                let sr = u32::try_from(raw).map_err(|_| {
                    VokraError::ModelLoad(format!(
                        "torchaudio_squim: GGUF metadata `{KEY_SQUIM_SAMPLE_RATE}` = {raw} does \
                         not fit in u32"
                    ))
                })?;
                (sr, ConfigSource::GgufChunk)
            }
            None => (EXPECTED_SAMPLE_RATE, ConfigSource::FactoryDefaults),
        };

        let cfg = Self {
            sample_rate,
            has_objective,
            has_subjective,
            source,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

// ---------------------------------------------------------------------------
// SquimWeights — bundle-prefix routing over the tensor manifest.
// ---------------------------------------------------------------------------

/// The tensor manifest of a bound SQUIM bundle, routed by bundle prefix.
///
/// Weights are held **by name and dims only**: the forwards are loud-partial,
/// so eagerly dequantising ~102 M parameters (the sidecar records ~7.4 M for
/// the objective head and ~94.4 M for the subjective one) would burn memory to
/// produce nothing. The follow-up wave that lands a real forward chooses its
/// own caching shape from the topology metadata; what it needs from today is a
/// routed, validated name list, which is what this is.
#[derive(Debug, Clone)]
pub struct SquimWeights {
    objective: Vec<(String, Vec<usize>)>,
    subjective: Vec<(String, Vec<usize>)>,
}

impl SquimWeights {
    /// Routes every tensor in `gguf` onto its head by leading dot-segment.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries **zero** tensors (an
    ///   empty manifest is never a valid SQUIM bundle — the smaller head alone
    ///   is millions of parameters).
    /// - [`VokraError::ModelLoad`] naming the offending tensor when a tensor
    ///   carries **neither** bundle prefix. The sidecar guarantees the prefix
    ///   on every key, so an unprefixed name means the bundle was assembled by
    ///   some other path and its routing cannot be trusted — silently ignoring
    ///   it would drop weights from a head that then "loads fine" (FR-EX-08).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let infos = gguf.tensors();
        if infos.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "torchaudio_squim: GGUF carries zero tensors — refusing to bind an empty \
                 bundle (FR-EX-08). A legitimate SQUIM bundle carries millions of \
                 parameters per head (arch={ARCH}, name={NAME}). Re-run \
                 `{PREPARE_CHECKPOINT_SIDECAR}` then `vokra-cli convert --model \
                 torchaudio-squim`."
            )));
        }

        let mut objective: Vec<(String, Vec<usize>)> = Vec::new();
        let mut subjective: Vec<(String, Vec<usize>)> = Vec::new();
        let mut unrouted: Vec<String> = Vec::new();

        let obj_prefix = SquimHead::Objective.tensor_prefix();
        let sub_prefix = SquimHead::Subjective.tensor_prefix();
        for info in infos {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            if let Some(rest) = info.name.strip_prefix(obj_prefix) {
                objective.push((rest.to_owned(), dims));
            } else if let Some(rest) = info.name.strip_prefix(sub_prefix) {
                subjective.push((rest.to_owned(), dims));
            } else {
                unrouted.push(info.name.clone());
            }
        }

        if !unrouted.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "torchaudio_squim: {n} tensor(s) carry neither the `{op}` nor the `{sp}` \
                 bundle prefix — first offender `{first}`. `{PREPARE_CHECKPOINT_SIDECAR}` \
                 prefixes every upstream state_dict key with its bundle tag \
                 (`objective.<upstream_name>` / `subjective.<upstream_name>`, the DNSMOS \
                 `p808.` / `p835.` convention) and the binder routes on that leading \
                 dot-segment, so an unprefixed name means this GGUF was assembled by \
                 some other path and its routing cannot be trusted. Silently ignoring \
                 it would drop weights from a head that then appears to load (FR-EX-08). \
                 Re-run the sidecar rather than renaming tensors on either side.",
                n = unrouted.len(),
                op = SquimHead::Objective.tensor_prefix(),
                sp = SquimHead::Subjective.tensor_prefix(),
                first = unrouted[0],
            )));
        }

        Ok(Self {
            objective,
            subjective,
        })
    }

    /// The objective head's tensors, with the bundle prefix **stripped** (i.e.
    /// upstream `state_dict` keys as they appear in `objective.py`).
    #[must_use]
    pub fn objective_tensors(&self) -> &[(String, Vec<usize>)] {
        &self.objective
    }

    /// The subjective head's tensors, with the bundle prefix **stripped**.
    #[must_use]
    pub fn subjective_tensors(&self) -> &[(String, Vec<usize>)] {
        &self.subjective
    }

    /// Tensors routed onto `head`.
    #[must_use]
    pub fn tensors_for(&self, head: SquimHead) -> &[(String, Vec<usize>)] {
        match head {
            SquimHead::Objective => &self.objective,
            SquimHead::Subjective => &self.subjective,
        }
    }

    /// Total number of tensors bound across both heads.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.objective.len() + self.subjective.len()
    }
}

// ---------------------------------------------------------------------------
// Squim — the runtime binder handle.
// ---------------------------------------------------------------------------

/// TorchAudio-SQUIM runtime binder — reference-free speech quality and
/// intelligibility estimation.
///
/// Bind with [`from_gguf`](Self::from_gguf) or [`from_path`](Self::from_path),
/// then call [`estimate_objective`](Self::estimate_objective) (STOI + PESQ +
/// SI-SDR, degraded waveform alone) or
/// [`estimate_subjective`](Self::estimate_subjective) (MOS, degraded waveform +
/// non-matching reference). See the module docs for the implementation-status
/// matrix and the FR-EX-08 loud-error contract on the deferred forwards.
#[derive(Debug, Clone)]
pub struct Squim {
    cfg: SquimConfig,
    weights: SquimWeights,
    weight_license: LicenseClass,
    /// Built once at bind time so [`Squim::heads`] can hand back a stable
    /// `&'static` slice without allocating per call (the DNSMOS `variants`
    /// pattern).
    heads: &'static [&'static str],
}

impl Squim {
    /// Binds a SQUIM bundle from a parsed GGUF (FR-LD-01).
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming exactly
    /// what is wrong, so a reader diagnosing a mis-produced GGUF has one place
    /// to walk — never a silent partial bind (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent, or is not
    ///   `"torchaudio_squim"` (a sibling eval-family GGUF —
    ///   [`SIBLING_EVAL_ARCHES`] — handed here by mistake fails with a specific
    ///   message rather than a downstream missing-tensor error).
    /// - [`VokraError::ModelLoad`] from [`SquimWeights::from_gguf`] on an empty
    ///   manifest or an unroutable tensor name.
    /// - [`VokraError::ModelLoad`] from [`SquimConfig::from_gguf`] on a bad
    ///   stamped sample rate or a headless bundle.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // 1. Arch check FIRST — a UTMOS / DNSMOS / NISQA GGUF handed here by
        //    mistake must fail with a clear message, not with a confusing
        //    "no tensors under the `objective.` prefix" three steps later.
        match gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "torchaudio_squim: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model torchaudio-squim`?). Note \
                     that the sibling `category=\"eval\"` quality-metric arches — \
                     `{s0}` (UTMOS single-scalar MOS predictor), `{s1}` (UTMOSv2), `{s2}` \
                     (DNSMOS P.808 / P.835), `{s3}` (NISQA v2) — all live in the same \
                     reference-free-quality neighbourhood but each has a different tensor \
                     tree and a different output arity. SQUIM is the only member with a \
                     two-bundle layout (objective STOI+PESQ+SI-SDR / subjective MOS), so \
                     silently aliasing arch would mis-route runtime dispatch onto a loader \
                     that cannot express it (FR-EX-08 — no silent partial load).",
                    s0 = SIBLING_EVAL_ARCHES[0],
                    s1 = SIBLING_EVAL_ARCHES[1],
                    s2 = SIBLING_EVAL_ARCHES[2],
                    s3 = SIBLING_EVAL_ARCHES[3],
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "torchaudio_squim: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native torchaudio_squim GGUF (was it produced by `vokra-cli \
                     convert --model torchaudio-squim`? see `{PREPARE_CHECKPOINT_SIDECAR}` \
                     for the upstream bridge)."
                )));
            }
        }

        // 2. Route the tensor manifest onto the two heads. Tensor presence —
        //    not metadata — is the ground truth for which heads can run.
        let weights = SquimWeights::from_gguf(gguf)?;
        let has_objective = !weights.objective_tensors().is_empty();
        let has_subjective = !weights.subjective_tensors().is_empty();

        // 3. Config (sample rate + truthful head inventory).
        let cfg = SquimConfig::from_gguf(gguf, has_objective, has_subjective)?;

        // 4. Provenance surfacing. The converter stamps the *code* licence
        //    (bsd-2-clause -> Permissive); the per-head *weight* licences
        //    diverge (see the module docstring). Fail-closed to Unknown when
        //    the stamp is absent, per
        //    `[[feedback-license-signoff-primary-source]]`.
        let weight_license = gguf
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        let heads: &'static [&'static str] = match (has_objective, has_subjective) {
            (true, true) => &["objective", "subjective"],
            (true, false) => &["objective"],
            (false, true) => &["subjective"],
            // SquimConfig::validate refuses a headless bundle above.
            (false, false) => unreachable!("SquimConfig::validate rejects a headless bundle"),
        };

        Ok(Self {
            cfg,
            weights,
            weight_license,
            heads,
        })
    }

    /// Opens and binds a SQUIM bundle from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Propagates the GGUF open/parse error, then everything
    /// [`Self::from_gguf`] can return.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The bound config.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &SquimConfig {
        &self.cfg
    }

    /// The routed tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &SquimWeights {
        &self.weights
    }

    /// The heads this bundle actually carries, in canonical order — the
    /// truthful subset, never a fixed `["objective", "subjective"]` regardless
    /// of contents (FR-EX-08). Mirrors the DNSMOS `variants()` shape.
    #[inline]
    #[must_use]
    pub fn heads(&self) -> &[&'static str] {
        self.heads
    }

    /// Whether the objective (STOI + PESQ + SI-SDR) head is present.
    #[inline]
    #[must_use]
    pub const fn has_objective(&self) -> bool {
        self.cfg.has_objective
    }

    /// Whether the subjective (MOS) head is present.
    ///
    /// Also the flag a publish gate should read: per the sidecar's record, the
    /// subjective weights are CC-BY-NC-4.0 while the code is BSD-2-Clause, so a
    /// bundle carrying this head is non-commercial-encumbered regardless of the
    /// stamped SPDX string. This module states that fact and makes **no**
    /// determination from it — `docs/license-audit.md` §3.1 is owner-only.
    #[inline]
    #[must_use]
    pub const fn has_subjective(&self) -> bool {
        self.cfg.has_subjective
    }

    /// The weight-licence class stamped in `vokra.provenance.weight_license`,
    /// or [`LicenseClass::Unknown`] when absent (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Estimate STOI, PESQ and SI-SDR from a degraded waveform **alone** — no
    /// reference of any kind.
    ///
    /// `pcm16k` is mono `f32` in `[-1, 1]` at [`EXPECTED_SAMPLE_RATE`].
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the deferred learnable 1-D
    /// encoder, the DPRNN block stack (and the general RNN primitive
    /// `vokra-ops` lacks), and the three transformer metric heads, plus the
    /// upstream file to transcribe and the metadata chunk to stamp. **No score
    /// is fabricated** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// Checks fire in a documented order, so the error a caller sees is the
    /// most specific one available:
    ///
    /// 1. [`VokraError::InvalidArgument`] when this bundle carries no
    ///    `objective.` tensors — a subjective-only bundle cannot produce STOI /
    ///    PESQ / SI-SDR, and returning zeros would be a lie.
    /// 2. [`VokraError::InvalidArgument`] when `pcm16k` is empty.
    /// 3. [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn estimate_objective(&self, pcm16k: &[f32]) -> Result<SquimObjectiveScores> {
        self.require_head(SquimHead::Objective)?;
        require_non_empty_pcm(pcm16k, "pcm16k")?;
        Err(forward_loud_partial(SquimHead::Objective))
    }

    /// Estimate MOS for `pcm16k` against a **non-matching reference**.
    ///
    /// `non_matching_reference` is an *unrelated clean utterance* — not the
    /// paired clean source of `pcm16k`. It supplies the quality anchor the
    /// subjective head was trained against (Kumar et al. 2023). Both slices are
    /// mono `f32` in `[-1, 1]` at [`EXPECTED_SAMPLE_RATE`]; they need not be
    /// the same length.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the deferred
    /// `wav2vec2_base` SSL encoder, the attentive pool and the linear
    /// projector, plus the upstream file to transcribe and the metadata chunk
    /// to stamp. **No MOS is fabricated** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// 1. [`VokraError::InvalidArgument`] when this bundle carries no
    ///    `subjective.` tensors.
    /// 2. [`VokraError::InvalidArgument`] when either slice is empty — the NMR
    ///    is a required input, so an empty one is a caller bug, not a
    ///    "reference-free" mode.
    /// 3. [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn estimate_subjective(
        &self,
        pcm16k: &[f32],
        non_matching_reference: &[f32],
    ) -> Result<f32> {
        self.require_head(SquimHead::Subjective)?;
        require_non_empty_pcm(pcm16k, "pcm16k")?;
        require_non_empty_pcm(non_matching_reference, "non_matching_reference")?;
        Err(forward_loud_partial(SquimHead::Subjective))
    }

    /// Refuses a call against a head this bundle does not carry, naming the
    /// prefix that came up empty and the upstream checkpoint that would supply
    /// it.
    fn require_head(&self, head: SquimHead) -> Result<()> {
        let present = match head {
            SquimHead::Objective => self.cfg.has_objective,
            SquimHead::Subjective => self.cfg.has_subjective,
        };
        if present {
            return Ok(());
        }
        let metrics: &[&str] = match head {
            SquimHead::Objective => &OBJECTIVE_METRICS,
            SquimHead::Subjective => &SUBJECTIVE_METRICS,
        };
        Err(VokraError::InvalidArgument(format!(
            "torchaudio_squim: cannot estimate {metrics:?} — this bundle carries no tensors \
             under the `{prefix}` prefix, so the `{short}` head is absent (advertised heads: \
             {heads:?}). The sidecar `{PREPARE_CHECKPOINT_SIDECAR}` emits partial bundles on \
             purpose (`--objective-only` / `--subjective-only`), so a truthful subset is a \
             valid artefact — but a missing head must be scored as *absent*, never as a \
             fabricated value (FR-EX-08). Re-run the sidecar including \
             `{ckpt}` to get this head.",
            prefix = head.tensor_prefix(),
            short = head.short(),
            heads = self.heads,
            ckpt = head.upstream_checkpoint(),
        )))
    }
}

/// Rejects an empty PCM slice, naming the offending argument.
///
/// A real check that holds today and will not change shape when the forwards
/// land: an empty waveform has no frames to encode, so every downstream path
/// would either divide by zero or silently return a default.
fn require_non_empty_pcm(pcm: &[f32], arg: &str) -> Result<()> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "torchaudio_squim: `{arg}` is empty — SQUIM needs at least one sample of mono \
             f32 PCM at {EXPECTED_SAMPLE_RATE} Hz. An empty waveform has no frames to \
             encode; returning a default score for it would be a fabricated measurement \
             (FR-EX-08)."
        )));
    }
    Ok(())
}

/// The note about the recurrent primitive `vokra-ops` does not yet have,
/// carried inside the objective head's loud-partial message.
///
/// Exposed as a function rather than inlined so the module docs can point at
/// one authoritative wording, and so a follow-up wave that lands a general RNN
/// op has a single grep target to delete.
#[must_use]
pub fn missing_primitive_note() -> &'static str {
    "the DPRNN stack additionally needs a GENERAL recurrent primitive that `vokra-ops` does \
     not currently expose: the only recurrent kernels in the tree are the DeepFilterNet3- \
     specific GRU stack embedded in `vokra_ops::denoise` (bound to DFN3's own tensor layout \
     and grouped-linear skips, not a reusable cell). Landing this head therefore means \
     either adding a reusable RNN op to `vokra-ops` or transcribing SQUIM's cell inline"
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Squim::estimate_objective`] / [`Squim::estimate_subjective`] until the
/// respective forward lands.
///
/// Each message names (a) the head and its concrete missing stages, (b) the
/// upstream reference file to transcribe, (c) the paper, (d) the reserved
/// `vokra.squim.*` topology chunk to stamp, and (e) why no value is returned.
/// A reader hitting this error knows exactly where to flip the switch.
fn forward_loud_partial(head: SquimHead) -> VokraError {
    let body = match head {
        SquimHead::Objective => format!(
            "three stages are missing: (1) the learnable 1-D encoder that projects raw audio \
             into the latent the DPRNN attends over; (2) the DPRNN block stack \
             (segment/chunk -> intra-chunk RNN -> inter-chunk RNN -> overlap-add; \
             num_blocks={nb}, chunk_size={cs}, feat_dim={fd} per the factory defaults \
             recorded in `{sidecar}` — but the RNN CELL TYPE and the per-block composition \
             are NOT recorded anywhere in this repository and must be read off the upstream \
             module, so best-guessing them would be silent-wrong); (3) the three transformer \
             metric heads (d_model={dm}, nhead={nh}) emitting {metrics:?} in that \
             load-bearing order. Note that {primitive}.",
            nb = OBJECTIVE_NUM_BLOCKS,
            cs = OBJECTIVE_CHUNK_SIZE,
            fd = OBJECTIVE_FEAT_DIM,
            dm = OBJECTIVE_D_MODEL,
            nh = OBJECTIVE_NHEAD,
            metrics = OBJECTIVE_METRICS,
            sidecar = PREPARE_CHECKPOINT_SIDECAR,
            primitive = missing_primitive_note(),
        ),
        SquimHead::Subjective => format!(
            "three stages are missing: (1) the `{ssl}` SSL encoder walk (feat_dim={fd}) — the \
             same wav2vec2-lineage gap `crate::emotion2vec` and `crate::wavlm` also defer on, \
             so landing it once unblocks all three; (2) attentive pooling (att_dim={ad}) over \
             the encoder time axis; (3) the linear projector ({fd} -> proj_dim={pd}) and the \
             pairing of the degraded utterance with its NON-MATCHING REFERENCE embedding into \
             a single {metrics:?} scalar. The exact composition is not recorded in this \
             repository — `{sidecar}` captures only the scalar hyperparameters — so \
             best-guessing it would be silent-wrong.",
            ssl = SUBJECTIVE_SSL_TYPE,
            fd = SUBJECTIVE_FEAT_DIM,
            ad = SUBJECTIVE_ATT_DIM,
            pd = SUBJECTIVE_PROJ_DIM,
            metrics = SUBJECTIVE_METRICS,
            sidecar = PREPARE_CHECKPOINT_SIDECAR,
        ),
    };

    VokraError::UnsupportedOp(format!(
        "torchaudio_squim {short} head (loud-partial): the forward is deferred — {body} \
         To flip this switch: transcribe {code}, extend `{sidecar}` to stamp `{topo}` (a u32 \
         op-token array pinning the op sequence), re-run `vokra-cli convert --model \
         torchaudio-squim`, then wire the forward against that token stream. Paper: {paper}. \
         Until then this is a loud pending — the runtime cannot fabricate a quality score \
         (FR-EX-08 no silent partial output).",
        short = head.short(),
        code = head.primary_source_code(),
        sidecar = PREPARE_CHECKPOINT_SIDECAR,
        topo = head.topology_key(),
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the TorchAudio-SQUIM runtime binder.
    //!
    //! # What can honestly be tested today
    //!
    //! On a real bundle the headline test would be `estimate_objective(...)`
    //! returning STOI / PESQ / SI-SDR within tolerance of the upstream
    //! `torchaudio` reference. Both forwards are deferred (see the module doc),
    //! and fabricating a score would violate CLAUDE.md 教訓 (a)
    //! ("loud-partial は fake-complete より honest"). What *is* real, and is
    //! what these tests pin:
    //!
    //! 1. **Contract-constant pins** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_URL` mirror the converter, and the factory hyperparameters
    //!    + metric ordering mirror the sidecar. A one-sided drift fails here.
    //! 2. **Arch verification** — absent and foreign arch tags are refused with
    //!    messages that name the sibling eval fleet.
    //! 3. **Tensor routing** — a synthetic bundle routes onto both heads;
    //!    empty and unprefixed manifests are refused, the latter naming the
    //!    offending tensor.
    //! 4. **Truthful partial bundles** — an objective-only bundle advertises
    //!    only `objective`, and asking it for MOS is a loud refusal naming the
    //!    missing prefix rather than a fabricated value.
    //! 5. **Two separately-reachable heads** — each entry point loud-partials
    //!    with its own missing primitives, its own primary source and its own
    //!    reserved topology chunk; neither message leaks the other's.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Number of bytes for an `[a, b]`-shaped F32 tensor.
    fn f32_bytes(a: usize, b: usize) -> Vec<u8> {
        vec![0u8; a * b * 4]
    }

    /// Builds a synthetic SQUIM GGUF.
    ///
    /// `objective` / `subjective` select which heads get tensors; the names
    /// mirror the sidecar's contract (`<bundle>.<upstream_name>`) using
    /// plausible upstream sub-module names. They are placeholders for *shape*,
    /// not a claim about the real upstream `state_dict` keys — the binder
    /// routes on the leading dot-segment only, which is the part the sidecar
    /// actually guarantees.
    fn squim_gguf(
        objective: bool,
        subjective: bool,
        weight_license: Option<LicenseClass>,
        sample_rate: Option<u32>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        if let Some(cls) = weight_license {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        if let Some(sr) = sample_rate {
            b.add_u32(KEY_SQUIM_SAMPLE_RATE, sr);
        }
        if objective {
            b.add_tensor(
                "objective.encoder.conv.weight",
                GgmlType::F32,
                vec![4, 4],
                f32_bytes(4, 4),
            )
            .expect("add objective tensor");
            b.add_tensor(
                "objective.branches.0.layers.0.weight",
                GgmlType::F32,
                vec![2, 3],
                f32_bytes(2, 3),
            )
            .expect("add objective tensor");
        }
        if subjective {
            b.add_tensor(
                "subjective.ssl_model.encoder.layers.0.attention.k_proj.weight",
                GgmlType::F32,
                vec![3, 3],
                f32_bytes(3, 3),
            )
            .expect("add subjective tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1 — contract-constant pins
    // -----------------------------------------------------------------------

    #[test]
    fn converter_contract_constants_are_mirrored_exactly() {
        assert_eq!(ARCH, "torchaudio_squim", "arch tag pin");
        assert_eq!(NAME, "torchaudio_squim", "model name pin");
        assert_eq!(CATEGORY, "eval", "category pin (sibling of utmos / dnsmos)");
        assert_eq!(UPSTREAM_URL, "github.com/pytorch/audio", "upstream URL pin");
        assert_eq!(DEFAULT_LICENSE_SPDX, "bsd-2-clause", "code SPDX pin");
        assert_eq!(KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            KEY_PROVENANCE_UPSTREAM_URL, "vokra.provenance.upstream_url",
            "SQUIM has no HF mirror, so provenance uses the URL key",
        );
        // The arch must stay distinct from every sibling eval-family tag.
        for sibling in SIBLING_EVAL_ARCHES {
            assert_ne!(
                ARCH, sibling,
                "SQUIM must not share an arch tag with `{sibling}`"
            );
        }
    }

    #[test]
    fn sidecar_factory_defaults_and_metric_order_are_pinned() {
        // Objective: `squim_objective_base()` defaults as recorded by
        // tools/parity/torchaudio_squim_prepare_checkpoint.py.
        assert_eq!(OBJECTIVE_FEAT_DIM, 256);
        assert_eq!(OBJECTIVE_D_MODEL, 256);
        assert_eq!(OBJECTIVE_NHEAD, 4);
        assert_eq!(OBJECTIVE_NUM_BLOCKS, 2);
        assert_eq!(OBJECTIVE_CHUNK_SIZE, 71);
        // Load-bearing order: index i is metric i in the 3-way branch.
        assert_eq!(OBJECTIVE_METRICS, ["stoi", "pesq", "sisdr"]);
        // Subjective: `squim_subjective_base()` defaults.
        assert_eq!(SUBJECTIVE_SSL_TYPE, "wav2vec2_base");
        assert_eq!(SUBJECTIVE_FEAT_DIM, 768);
        assert_eq!(SUBJECTIVE_PROJ_DIM, 32);
        assert_eq!(SUBJECTIVE_ATT_DIM, 5);
        assert_eq!(SUBJECTIVE_METRICS, ["mos"]);
        assert_eq!(EXPECTED_SAMPLE_RATE, 16_000);
    }

    #[test]
    fn head_prefixes_match_the_sidecar_prefix_policy() {
        // The sidecar prefixes every key `objective.` / `subjective.` and
        // states the binder must walk that leading dot-segment. Renaming
        // either side silently un-routes every tensor.
        assert_eq!(SquimHead::Objective.tensor_prefix(), "objective.");
        assert_eq!(SquimHead::Subjective.tensor_prefix(), "subjective.");
        assert_eq!(SquimHead::Objective.short(), "objective");
        assert_eq!(SquimHead::Subjective.short(), "subjective");
        assert_eq!(
            SquimHead::Objective.upstream_checkpoint(),
            "squim_objective_dns2020.pth"
        );
        assert_eq!(
            SquimHead::Subjective.upstream_checkpoint(),
            "squim_subjective_bvcc_daps.pth"
        );
        assert_eq!(
            SquimHead::Objective.topology_key(),
            KEY_SQUIM_OBJECTIVE_TOPOLOGY
        );
        assert_eq!(
            SquimHead::Subjective.topology_key(),
            KEY_SQUIM_SUBJECTIVE_TOPOLOGY
        );
        // The reserved chunk group namespace must not collide with DNSMOS's.
        assert!(KEY_SQUIM_BUNDLE.starts_with("vokra.squim."));
        assert!(KEY_SQUIM_SAMPLE_RATE.starts_with("vokra.squim."));
    }

    // -----------------------------------------------------------------------
    // 2 — arch verification (MUST refuse foreign GGUFs)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor(
            "objective.encoder.conv.weight",
            GgmlType::F32,
            vec![2, 2],
            f32_bytes(2, 2),
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Squim::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model torchaudio-squim"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_the_sibling_eval_fleet() {
        // A DNSMOS GGUF handed to the SQUIM binder by mistake. Both are
        // `category = "eval"` reference-free quality metrics, so this is the
        // realistic mis-route, and it must fail loud rather than proceed to a
        // confusing "no tensors under `objective.`" error.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "dnsmos");
        b.add_string(chunks::KEY_MODEL_NAME, "dnsmos-p808-p835");
        b.add_tensor("p808.conv1", GgmlType::F32, vec![2, 2], f32_bytes(2, 2))
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Squim::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch tag");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`dnsmos`") && m.contains("`torchaudio_squim`"),
                    "message must name both the observed and expected arch, got `{m}`"
                );
                for sibling in SIBLING_EVAL_ARCHES {
                    assert!(
                        m.contains(sibling),
                        "message must enumerate sibling `{sibling}`, got `{m}`"
                    );
                }
                assert!(
                    m.contains("two-bundle"),
                    "message should explain why SQUIM cannot share an arch, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 3 — tensor routing
    // -----------------------------------------------------------------------

    #[test]
    fn synthetic_bundle_loads_and_routes_both_heads() {
        let file = squim_gguf(true, true, Some(LicenseClass::Permissive), None);
        let m = Squim::from_gguf(&file).expect("a well-formed SQUIM GGUF must bind");

        assert!(m.has_objective(), "objective head must be discovered");
        assert!(m.has_subjective(), "subjective head must be discovered");
        assert_eq!(m.heads(), &["objective", "subjective"][..]);
        assert_eq!(m.weights().objective_tensors().len(), 2);
        assert_eq!(m.weights().subjective_tensors().len(), 1);
        assert_eq!(m.weights().total_count(), 3);

        // The bundle prefix is stripped, so what surfaces is the upstream
        // state_dict key a follow-up wave will match against objective.py.
        assert_eq!(
            m.weights().objective_tensors()[0].0,
            "encoder.conv.weight",
            "the `objective.` bundle prefix must be stripped on the way out"
        );
        assert_eq!(m.weights().objective_tensors()[0].1, vec![4, 4]);
        assert!(
            m.weights().subjective_tensors()[0]
                .0
                .starts_with("ssl_model."),
            "the `subjective.` bundle prefix must be stripped on the way out"
        );
        assert_eq!(
            m.weights().tensors_for(SquimHead::Subjective).len(),
            1,
            "tensors_for must agree with the per-head accessor"
        );

        // Config: nothing stamped the chunk group, so this is a documented
        // default and says so.
        assert_eq!(m.config().sample_rate, EXPECTED_SAMPLE_RATE);
        assert_eq!(
            m.config().source,
            ConfigSource::FactoryDefaults,
            "an unstamped `vokra.squim.*` group must be reported as a default, not as a \
             stamped value"
        );
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
    }

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // NO tensors.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Squim::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(m.contains("zero tensors"), "must name the gap: {m}");
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
                assert!(
                    m.contains(PREPARE_CHECKPOINT_SIDECAR),
                    "must point at the sidecar: {m}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_a_tensor_that_escapes_both_bundle_prefixes() {
        // A tensor with no bundle prefix cannot be routed. Silently ignoring
        // it would drop weights from a head that then appears to load, so the
        // binder refuses and NAMES the offender.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_tensor(
            "objective.encoder.conv.weight",
            GgmlType::F32,
            vec![2, 2],
            f32_bytes(2, 2),
        )
        .expect("add_tensor");
        b.add_tensor(
            "ssl_encoder.layers.0.attn.wq.weight",
            GgmlType::F32,
            vec![2, 3],
            f32_bytes(2, 3),
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Squim::from_gguf(&file) else {
            panic!("expected ModelLoad on an unroutable tensor name");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("ssl_encoder.layers.0.attn.wq.weight"),
                    "the error must NAME the offending tensor, got `{m}`"
                );
                assert!(
                    m.contains("objective.") && m.contains("subjective."),
                    "the error must name both expected prefixes, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn stamped_sample_rate_is_honoured_and_a_wrong_one_is_refused() {
        // Present and correct -> reported as a stamped value.
        let ok = squim_gguf(true, false, None, Some(EXPECTED_SAMPLE_RATE));
        let m = Squim::from_gguf(&ok).expect("16 kHz stamp must bind");
        assert_eq!(m.config().source, ConfigSource::GgufChunk);
        assert_eq!(m.config().sample_rate, EXPECTED_SAMPLE_RATE);

        // Present and wrong -> hard failure. An explicit claim that
        // contradicts the model is worse than no claim at all.
        let bad = squim_gguf(true, false, None, Some(24_000));
        let Err(err) = Squim::from_gguf(&bad) else {
            panic!("expected ModelLoad on a non-16 kHz stamp");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains("24000"), "must echo the observed rate: {msg}");
                assert!(msg.contains("16000"), "must name the expected rate: {msg}");
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        let file = squim_gguf(true, true, None, None);
        let m = Squim::from_gguf(&file).expect("license stamp is not a bind gate");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an absent weight-license stamp must fail closed, never default to Permissive"
        );
    }

    // -----------------------------------------------------------------------
    // 4 — truthful partial bundles
    // -----------------------------------------------------------------------

    #[test]
    fn objective_only_bundle_advertises_only_objective_and_refuses_mos() {
        let file = squim_gguf(true, false, Some(LicenseClass::Permissive), None);
        let m = Squim::from_gguf(&file).expect("a partial bundle is a valid artefact");
        assert!(m.has_objective());
        assert!(!m.has_subjective());
        assert_eq!(
            m.heads(),
            &["objective"],
            "a partial bundle must advertise the truthful subset only"
        );

        let Err(err) = m.estimate_subjective(&[0.0; 16], &[0.0; 16]) else {
            panic!("a subjective-head call on an objective-only bundle must fail");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("subjective."),
                    "must name the missing prefix, got `{msg}`"
                );
                assert!(
                    msg.contains("squim_subjective_bvcc_daps.pth"),
                    "must name the checkpoint that would supply it, got `{msg}`"
                );
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn subjective_only_bundle_advertises_only_subjective_and_refuses_objective() {
        let file = squim_gguf(false, true, Some(LicenseClass::NonCommercial), None);
        let m = Squim::from_gguf(&file).expect("a partial bundle is a valid artefact");
        assert!(!m.has_objective());
        assert!(m.has_subjective());
        assert_eq!(m.heads(), &["subjective"][..]);
        // The NC-encumbered head is the one a publish gate must notice; the
        // binder surfaces the class it was handed and makes no determination.
        assert_eq!(m.weight_license(), LicenseClass::NonCommercial);

        let Err(err) = m.estimate_objective(&[0.0; 16]) else {
            panic!("an objective-head call on a subjective-only bundle must fail");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("objective."),
                    "must name the missing prefix, got `{msg}`"
                );
                assert!(
                    msg.contains("stoi") && msg.contains("pesq") && msg.contains("sisdr"),
                    "must name the metrics that cannot be produced, got `{msg}`"
                );
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn empty_pcm_is_refused_before_the_loud_partial() {
        let file = squim_gguf(true, true, None, None);
        let m = Squim::from_gguf(&file).expect("bind");

        // Objective: empty waveform.
        let Err(err) = m.estimate_objective(&[]) else {
            panic!("empty PCM must be refused");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("`pcm16k` is empty"),
                    "names the argument: {msg}"
                );
            }
            other => panic!("expected InvalidArgument (not the loud-partial), got {other:?}"),
        }

        // Subjective: the NMR is a required input, so an empty one is a caller
        // bug rather than a "reference-free" mode.
        let Err(err) = m.estimate_subjective(&[0.0; 8], &[]) else {
            panic!("an empty non-matching reference must be refused");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("`non_matching_reference` is empty"),
                    "names the argument: {msg}"
                );
            }
            other => panic!("expected InvalidArgument (not the loud-partial), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5 — the two heads are separately reachable and each loud-partials with
    //     its OWN missing primitives
    // -----------------------------------------------------------------------

    #[test]
    fn objective_head_loud_partials_naming_the_dprnn_and_its_missing_primitive() {
        let file = squim_gguf(true, true, Some(LicenseClass::Permissive), None);
        let m = Squim::from_gguf(&file).expect("bind");

        // 1 s of silence at 16 kHz — a shape-legitimate input.
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.estimate_objective(&pcm) else {
            panic!("estimate_objective must loud-partial, never return a fabricated score");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("objective head"), "names the surface: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");
                assert!(msg.contains("DPRNN"), "names the deferred stack: {msg}");
                assert!(
                    msg.contains("1-D encoder"),
                    "names the deferred encoder: {msg}"
                );
                assert!(
                    msg.contains("transformer metric heads"),
                    "names the deferred heads: {msg}"
                );
                // The missing vokra-ops primitive is named explicitly.
                assert!(
                    msg.contains("vokra-ops") && msg.contains("recurrent"),
                    "must name the missing recurrent primitive: {msg}"
                );
                assert!(
                    msg.contains("vokra_ops::denoise"),
                    "must say which recurrent kernels DO exist so the reader is not sent \
                     looking for nothing: {msg}"
                );
                // Anchors to walk.
                assert!(
                    msg.contains(PRIMARY_SOURCE_CODE_OBJECTIVE),
                    "must cite the upstream objective module: {msg}"
                );
                assert!(
                    msg.contains(PRIMARY_SOURCE_PAPER),
                    "must cite the paper: {msg}"
                );
                assert!(
                    msg.contains(KEY_SQUIM_OBJECTIVE_TOPOLOGY),
                    "must name the reserved topology chunk to stamp: {msg}"
                );
                assert!(
                    msg.contains(PREPARE_CHECKPOINT_SIDECAR),
                    "must name the sidecar to extend: {msg}"
                );
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
                // It must NOT bleed the subjective head's gaps.
                assert!(
                    !msg.contains(PRIMARY_SOURCE_CODE_SUBJECTIVE),
                    "the objective message must not cite subjective.py: {msg}"
                );
                assert!(
                    !msg.contains("attentive pooling"),
                    "the objective message must not name the subjective head's stages: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn subjective_head_loud_partials_naming_wav2vec2_attentive_pool_and_the_nmr() {
        let file = squim_gguf(true, true, Some(LicenseClass::Permissive), None);
        let m = Squim::from_gguf(&file).expect("bind");

        let degraded = vec![0.0_f32; 16_000];
        let nmr = vec![0.0_f32; 8_000];
        let Err(err) = m.estimate_subjective(&degraded, &nmr) else {
            panic!("estimate_subjective must loud-partial, never return a fabricated MOS");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("subjective head"), "names the surface: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");
                assert!(
                    msg.contains(SUBJECTIVE_SSL_TYPE),
                    "names the deferred SSL encoder: {msg}"
                );
                assert!(
                    msg.contains("attentive pooling"),
                    "names the deferred pool: {msg}"
                );
                assert!(
                    msg.contains("linear projector"),
                    "names the deferred projector: {msg}"
                );
                assert!(
                    msg.contains("NON-MATCHING REFERENCE"),
                    "must call out that this head is NOT reference-free: {msg}"
                );
                // The shared wav2vec2-lineage gap is cross-referenced so a
                // follow-up wave knows landing it once unblocks three binders.
                assert!(
                    msg.contains("emotion2vec"),
                    "must cross-reference the shared wav2vec2-lineage gap: {msg}"
                );
                assert!(
                    msg.contains(PRIMARY_SOURCE_CODE_SUBJECTIVE),
                    "must cite the upstream subjective module: {msg}"
                );
                assert!(
                    msg.contains(KEY_SQUIM_SUBJECTIVE_TOPOLOGY),
                    "must name the reserved topology chunk to stamp: {msg}"
                );
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
                // It must NOT bleed the objective head's gaps.
                assert!(
                    !msg.contains(PRIMARY_SOURCE_CODE_OBJECTIVE),
                    "the subjective message must not cite objective.py: {msg}"
                );
                assert!(
                    !msg.contains("DPRNN"),
                    "the subjective message must not name the objective head's stack: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn the_two_heads_produce_distinct_loud_partial_messages() {
        // Both heads are reachable on a full bundle, and neither is a rename
        // of the other: a single shared "not implemented" string would hide
        // that these are two independently-deferred forwards.
        let file = squim_gguf(true, true, None, None);
        let m = Squim::from_gguf(&file).expect("bind");
        let pcm = vec![0.0_f32; 128];

        let Err(obj) = m.estimate_objective(&pcm) else {
            panic!("objective must loud-partial");
        };
        let Err(sub) = m.estimate_subjective(&pcm, &pcm) else {
            panic!("subjective must loud-partial");
        };
        let (o, s) = match (&obj, &sub) {
            (VokraError::UnsupportedOp(o), VokraError::UnsupportedOp(s)) => (o, s),
            _ => panic!("both heads must loud-partial with UnsupportedOp, got {obj:?} / {sub:?}"),
        };
        assert_ne!(
            o, s,
            "each head must name its own missing primitives, not share one message"
        );
    }

    #[test]
    fn config_validate_refuses_a_headless_bundle() {
        // Reachable only by constructing the config directly — from_gguf can
        // no longer produce this state because an empty manifest is refused
        // earlier. Pinned anyway so the invariant survives a refactor that
        // reorders the gates.
        let cfg = SquimConfig {
            sample_rate: EXPECTED_SAMPLE_RATE,
            has_objective: false,
            has_subjective: false,
            source: ConfigSource::FactoryDefaults,
        };
        let Err(err) = cfg.validate() else {
            panic!("a headless bundle must be refused");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("neither head"),
                    "must name the condition: {msg}"
                );
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }
}
