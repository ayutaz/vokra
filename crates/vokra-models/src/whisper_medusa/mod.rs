//! **aiola Whisper-Medusa-v1** — runtime binder for the
//! `whisper-medusa-v1` arch (Wave C1 2026-08-15 coverage-gap closure).
//!
//! Closes a real gap: `crates/vokra-convert/src/models/whisper_medusa_v1.rs`
//! (coverage-audit wave-b, 2026-08-03) produces a GGUF stamped
//! `vokra.model.arch = "whisper-medusa-v1"` that a workspace-wide grep proved
//! **nothing read back** — every converted Whisper-Medusa checkpoint was
//! unloadable. This module is that consumer.
//!
//! # Primary sources
//!
//! Only sources the converter itself records are cited here — nothing is
//! added from memory (CLAUDE.md「ハルシネーション厳禁」):
//!
//! - HF release: <https://huggingface.co/aiola/whisper-medusa-v1>
//!   (recorded by the converter as `UPSTREAM_HF`).
//! - Method paper: Cai et al. 2024, *"Medusa: Simple LLM Inference
//!   Acceleration Framework with Multiple Decoding Heads"*,
//!   <https://arxiv.org/abs/2401.10774> (cited verbatim in the converter
//!   docstring).
//! - In-repo ticket:
//!   `docs/tickets/coverage-audit-2026-08-03/wave-b/whisper-medusa-v1.md`
//!   (cited by the converter; its §Converter section explicitly scopes the
//!   runtime binding — i.e. *this module* — to a follow-up).
//!
//! # License status — UNVERIFIED, owner follow-up (do not upgrade to a claim)
//!
//! The converter stamps `apache-2.0` → [`LicenseClass::Permissive`] **by
//! default**, and its own docstring is explicit that this is the
//! *ticket-header* value ("the aiola precedent") and **not** a transcription
//! of the primary-source model-card front matter. The 2026-08-03 audit
//! flagged the row `要一次 (Apache-2.0 想定)` — literally "primary source
//! required (Apache-2.0 assumed)".
//!
//! Therefore:
//!
//! - this module makes **no license claim**. [`CONVERTER_DEFAULT_LICENSE`] is
//!   named as *what the converter writes when no `--license` override is
//!   given*, never as "the license of `aiola/whisper-medusa-v1`";
//! - [`WhisperMedusa::weight_license`] only **surfaces** whatever class the
//!   artifact carries, and fail-closes to [`LicenseClass::Unknown`] when the
//!   stamp is absent;
//! - `docs/license-audit.md` §3.1 sign-off stays **BLANK**. Owner-only per
//!   `[[feedback-license-signoff-primary-source]]` — CC does not sign, and
//!   does not treat the converter's assumed default as a sign-off. Until an
//!   owner reads `huggingface.co/aiola/whisper-medusa-v1`'s front matter and
//!   signs, publishing a converted Whisper-Medusa GGUF stays blocked at the
//!   §3.1 gate.
//!
//! # What Whisper-Medusa is
//!
//! An OpenAI Whisper backbone (encoder + decoder, unmodified) plus **N Medusa
//! speculative-decoding heads**: extra LM heads that each predict one further
//! future token from the same final decoder hidden state, so a step proposes
//! several candidate continuations at once which the base decoder then
//! *verifies* in a single forward. The published gain is a per-step
//! throughput multiplier at unchanged output quality — the accepted prefix is
//! by construction what plain greedy/beam decoding would have produced.
//!
//! The consequence that matters for a binder: **the base tower is ordinary
//! Whisper**. Speculation is a decode-loop optimisation, not a different
//! model. So this module deliberately does **not** fork the Whisper
//! encoder/decoder — see the next section.
//!
//! # Reuse posture — no fork of the Whisper tower
//!
//! The base tower loads through the existing, real
//! [`crate::whisper::WhisperAsr`] path. Two facts make that work:
//!
//! 1. **Tensor names already line up.** The Medusa converter passes every
//!    float tensor through under its verbatim upstream safetensors name, and
//!    [`crate::whisper::WhisperWeights`] binds on exactly those HF-Transformers
//!    names (`model.encoder.layers.{i}.self_attn.q_proj.weight`,
//!    `model.decoder.layers.{i}.encoder_attn_layer_norm.bias`, …). No rename
//!    layer is needed.
//! 2. **`WhisperAsr::from_gguf` does not gate on arch.** The arch gate
//!    (`whisper::verify_arch`, whose [`crate::whisper::ACCEPTED_ARCHS`] list
//!    *deliberately excludes* `whisper-medusa-v1`) is applied by
//!    [`crate::whisper::WhisperSession`], not by the `WhisperAsr` load path
//!    that `distil-whisper` / `kotoba-whisper` already delegate through.
//!
//! **Do not "fix" this by adding `whisper-medusa-v1` to
//! [`crate::whisper::ACCEPTED_ARCHS`].** That list is documented as excluding
//! this arch precisely so a bare `WhisperSession` cannot bind a Medusa
//! checkpoint and silently drop the heads. Arch ownership stays here; the
//! *tower* is borrowed, the *identity* is not.
//!
//! # The metadata gap this binder exposes (loud, not silent)
//!
//! The Medusa converter is a pure pass-through: it writes
//! `vokra.model.*` + `vokra.provenance.*` + `vokra.schema.*` and **nothing
//! else**. In particular it stamps neither
//!
//! - the `vokra.whisper.*` hyper-parameter chunk
//!   ([`crate::whisper::WhisperConfig`] requires it), nor
//! - the `vokra.frontend.*` chunk (the FR-LD-03 bit-exact front-end check
//!   requires it), nor
//! - `vokra.tokenizer.model` (detokenization requires it, or an explicitly
//!   attached [`WhisperTokenizer`]).
//!
//! So against a GGUF produced by today's converter, the base-tower delegation
//! **fails at load**, and this binder records *why*: [`WhisperMedusa::from_gguf`]
//! still succeeds (the artifact genuinely is a valid Medusa GGUF per the
//! converter contract — refusing it would re-open the very gap this module
//! closes), the Medusa heads are really bound, and every transcription entry
//! point then returns a loud [`VokraError::UnsupportedOp`] quoting the
//! underlying [`crate::whisper`] error verbatim. Nothing degrades silently
//! (FR-EX-08).
//!
//! The fix is METADATA, not weights — the tensor names already line up. The
//! real route is to teach `vokra-convert::models::whisper_medusa_v1` to stamp
//! the `vokra.whisper.*` + `vokra.frontend.*` chunks the way its sibling
//! `vokra-convert::models::whisper` already does; the moment it does, the
//! *same* code path here transcribes for real with zero changes to this
//! module, and [`WhisperMedusa::base_status`] is how a caller checks which of
//! the two worlds it is in. That converter change is a follow-up and is NOT
//! done here — today's Medusa converter stamps only `vokra.model.*`,
//! `vokra.provenance.*` and `vokra.schema.*`.
//!
//! Restamping is NOT a route to those chunks. `restamp_provenance` lives in
//! **`vokra-convert`** (`vokra_convert::restamp_provenance`), not in
//! `vokra-core`, and it rewrites only the `vokra.provenance.*` group: it
//! carries every other metadata key through verbatim from the input, so it
//! can never introduce a `vokra.whisper.*` or `vokra.frontend.*` chunk that
//! the input did not already carry.
//!
//! # Implementation-status matrix
//!
//! **REAL (this WP)**
//!
//! - Strict `vokra.model.arch == "whisper-medusa-v1"` verification that
//!   refuses a foreign GGUF loudly, naming BOTH the expected and the actual
//!   arch and enumerating the whole vanilla-Whisper-topology sibling fleet
//!   (`whisper` / `crisper-whisper` / `distil-whisper` / `kotoba-whisper`) —
//!   those four share the tensor topology verbatim, so only the arch tag can
//!   disambiguate them, and mis-routing one into this binder would look for
//!   Medusa heads that do not exist.
//! - Medusa-head discovery ([`MedusaHeads::from_gguf`]): a real prefix walk
//!   over the GGUF tensor manifest that groups tensors per head index,
//!   refuses a malformed head index by name, refuses a non-contiguous head
//!   index set by naming the first gap, and refuses an artifact carrying no
//!   Medusa tensors at all (which would be plain Whisper mis-stamped).
//! - Base-tower presence gate: an artifact with Medusa heads but no Whisper
//!   encoder tensors is refused by name.
//! - Optional all-or-nothing `vokra.medusa.*` hyper-parameter group
//!   ([`MedusaConfig`]) — absent → [`WhisperMedusa::config`] is `None` and the
//!   checkpoint still binds; partially stamped → loud, naming the missing key;
//!   `0` sentinel → loud; and a **cross-check** that a stamped
//!   `vokra.medusa.num_heads` agrees with the head count actually discovered
//!   on disk.
//! - Base Whisper tower delegation through [`crate::whisper::WhisperAsr`],
//!   with the load outcome recorded rather than swallowed.
//! - Weight-license surfacing, fail-closed to [`LicenseClass::Unknown`].
//!
//! **LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
//! honest」)**
//!
//! - [`WhisperMedusa::transcribe_speculative`] — **always** returns
//!   [`VokraError::UnsupportedOp`], even when the base tower bound
//!   successfully. Three concrete blockers, none of them a matter of
//!   effort-in-this-file:
//!   1. **No speculative decode loop exists anywhere in the workspace.**
//!      `vokra_core::decode` carries `beam_search` / `sample_sequence` /
//!      `ctc_decode`-family drivers; there is no draft→verify→accept driver
//!      and no tree/sparse attention mask op in `vokra-ops` to run candidate
//!      continuations in one batched decoder forward.
//!   2. **The Medusa tree topology is not recorded anywhere.** Upstream
//!      Medusa selects candidates with a `medusa_choices` tree (which head
//!      feeds which continuation at which top-k rank). The converter stamps
//!      no `vokra.medusa.*` group at all, so the tree is invisible in the
//!      artifact; guessing it would produce a shape-valid decode that
//!      silently accepts wrong tokens.
//!   3. **The per-head sub-module layout needs a real-checkpoint walk.**
//!      This binder groups by head index (real, verifiable), but the
//!      residual-block depth *inside* each head, and whether the final
//!      projection carries a bias, cannot be transcribed from any source the
//!      converter records.
//!
//!   Because the accepted prefix of a correct Medusa decode is *identical* to
//!   what plain decoding produces, the honest fallback is explicit: callers
//!   wanting output today use [`WhisperMedusa::transcribe`] (base decode, no
//!   speedup) — the loud error says so. **No fabricated speculative tokens,
//!   and no silently-non-speculative "speculative" call, are ever emitted**
//!   (FR-EX-08).
//! - [`WhisperMedusa::transcribe`] / [`WhisperMedusa::transcribe_tokens`] —
//!   real when the base tower bound; loud `UnsupportedOp` quoting the
//!   underlying `vokra.whisper.*` / `vokra.frontend.*` load error otherwise.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] mirror the converter's
//! constants — the same deliberate two-copies convention every sibling binder
//! uses so `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`, preserving the layered convention `vokra-ops → nothing
//! GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! `aiola/whisper-medusa-v1` ships `model.safetensors` driven by a Python /
//! Transformers pipeline; this runtime **never** touches ONNX or pickle
//! (FR-LD-05 / NFR-DS-02). The pipeline is re-implemented natively
//! (whisper.cpp 型, CLAUDE.md 設計判断 4).

use vokra_core::engines::AsrEngine;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::tasks::Transcription;
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};

use crate::whisper::{WhisperAsr, WhisperTokenizer};

// ---------------------------------------------------------------------------
// Contract constants — mirror of
// `crates/vokra-convert/src/models/whisper_medusa_v1.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model whisper-medusa-v1`.
///
/// Deliberately distinct from `"whisper"` — and deliberately **absent** from
/// [`crate::whisper::ACCEPTED_ARCHS`] — because the Medusa head family is not
/// present in a base Whisper checkpoint. Admitting this arch into the vanilla
/// binder would bind the base tower and silently drop the heads (FR-EX-08).
pub const ARCH: &str = "whisper-medusa-v1";

/// Expected `vokra.model.name` value written by the converter.
pub const NAME: &str = "whisper-medusa-v1";

/// Expected `vokra.model.category` value — the same `"asr"` tier as vanilla
/// Whisper / distil-whisper / kotoba-whisper / canary / parakeet. The
/// speculative-decoding axis is resolved by the arch tag, not by the category.
pub const CATEGORY: &str = "asr";

/// Upstream HuggingFace slug, recorded so loud diagnostics can point at the
/// serving location without re-fetching a manifest.
pub const UPSTREAM_HF: &str = "aiola/whisper-medusa-v1";

/// The SPDX id the converter stamps **when no `--license` override is given**.
///
/// This is the converter's ticket-header-derived *assumption*, NOT a
/// transcription of the upstream model card. See the module docstring's
/// license section: the 2026-08-03 audit flagged this row `要一次` (primary
/// source required) and `docs/license-audit.md` §3.1 sign-off is owner-only.
/// Never present this constant as "the license of `aiola/whisper-medusa-v1`".
pub const CONVERTER_DEFAULT_LICENSE: &str = "apache-2.0";

/// Tensor-name prefixes under which the Medusa heads are discovered.
///
/// Upstream Medusa exposes the heads as an `nn.ModuleList` attribute, so every
/// head tensor is `"<attr>.{head_index}.…"`. The converter's own round-trip
/// fixture pins the singular spelling (`medusa_head.0.linear.weight`); the
/// plural is accepted as well because the exact attribute spelling in the
/// `aiola/whisper-medusa-v1` release cannot be transcribed from any source the
/// converter records, and rejecting a real checkpoint over an `s` would be a
/// worse failure than tolerating both. A GGUF carrying **both** spellings is
/// ambiguous and is refused loudly (see [`MedusaHeads::from_gguf`]).
pub const MEDUSA_HEAD_PREFIXES: [&str; 2] = ["medusa_head.", "medusa_heads."];

/// Tensor-name prefixes that identify the **base Whisper encoder** inside a
/// Medusa artifact.
///
/// HF Whisper exports name encoder blocks `model.encoder.layers.{i}.…`; a
/// checkpoint saved from the inner module rather than the
/// `…ForConditionalGeneration` wrapper drops the `model.` segment. Both are
/// accepted; at least one must match or the artifact is not a Whisper-backed
/// Medusa checkpoint at all.
pub const BASE_ENCODER_PREFIXES: [&str; 2] = ["model.encoder.layers.", "encoder.layers."];

/// Optional `vokra.medusa.num_heads` — the head count a future converter
/// revision should stamp. Absent in every artifact today's converter produces.
pub const KEY_MEDUSA_NUM_HEADS: &str = "vokra.medusa.num_heads";

/// Optional `vokra.medusa.num_layers` — the residual-block depth inside each
/// head. Absent in every artifact today's converter produces.
pub const KEY_MEDUSA_NUM_LAYERS: &str = "vokra.medusa.num_layers";

/// Primary-source anchor: the HF release (recorded by the converter).
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/aiola/whisper-medusa-v1";

/// Primary-source anchor: the Medusa paper (Cai et al. 2024), cited verbatim
/// in the converter docstring.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2401.10774";

/// Primary-source anchor: the in-repo ticket the converter cites, whose
/// §Converter section scopes this runtime binding as the follow-up.
pub const PRIMARY_SOURCE_TICKET: &str =
    "docs/tickets/coverage-audit-2026-08-03/wave-b/whisper-medusa-v1.md";

/// The sibling arch tags that share the vanilla Whisper tensor topology
/// verbatim — enumerated in the arch-mismatch error so a mis-routed load has
/// fully specified anchors. Mirror of [`crate::whisper::ACCEPTED_ARCHS`].
const VANILLA_WHISPER_SIBLINGS: [&str; 4] = [
    "whisper",
    "crisper-whisper",
    "distil-whisper",
    "kotoba-whisper",
];

// ---------------------------------------------------------------------------
// MedusaConfig — optional, all-or-nothing `vokra.medusa.*` group.
// ---------------------------------------------------------------------------

/// The optional `vokra.medusa.*` hyper-parameter group.
///
/// **Absent in every artifact today's converter produces** — the converter is
/// a pure pass-through. It is read all-or-nothing so a half-stamped artifact
/// (a converter revision that grew one key and forgot the other) fails loudly
/// at load rather than silently defaulting a geometry axis.
///
/// Deliberately **no** `upstream_default()` constructor: the head count and
/// residual depth of `aiola/whisper-medusa-v1` are not stated in any source
/// the converter records, so a default would be invented numbers wearing an
/// authoritative face (CLAUDE.md「ハルシネーション厳禁」). The same posture
/// the sibling `ten_vad` / `firered_vad` binders take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MedusaConfig {
    /// Number of Medusa heads (`vokra.medusa.num_heads`).
    pub num_heads: usize,
    /// Residual-block depth inside each head (`vokra.medusa.num_layers`).
    pub num_layers: usize,
}

impl MedusaConfig {
    /// Reads the all-or-nothing `vokra.medusa.*` group.
    ///
    /// Returns `Ok(None)` when **no** key of the group is present (the
    /// universal case today). Returns an error when the group is partially
    /// stamped, when a value is not an unsigned integer, or when a value is
    /// the `0` sentinel.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the missing key when the group is
    ///   partially stamped.
    /// - [`VokraError::ModelLoad`] naming the key when a value is present but
    ///   not an unsigned integer, or is `0`.
    pub fn from_gguf(file: &GgufFile) -> Result<Option<Self>> {
        let raw_heads = file.get(KEY_MEDUSA_NUM_HEADS);
        let raw_layers = file.get(KEY_MEDUSA_NUM_LAYERS);

        match (raw_heads, raw_layers) {
            (None, None) => return Ok(None),
            (Some(_), None) => {
                return Err(VokraError::ModelLoad(format!(
                    "whisper-medusa: the `vokra.medusa.*` group is partially stamped — \
                     `{KEY_MEDUSA_NUM_HEADS}` is present but `{KEY_MEDUSA_NUM_LAYERS}` is \
                     missing. The group is read all-or-nothing so a half-stamped artifact \
                     fails at load instead of silently defaulting a geometry axis \
                     (FR-EX-08). Either stamp both keys or neither."
                )));
            }
            (None, Some(_)) => {
                return Err(VokraError::ModelLoad(format!(
                    "whisper-medusa: the `vokra.medusa.*` group is partially stamped — \
                     `{KEY_MEDUSA_NUM_LAYERS}` is present but `{KEY_MEDUSA_NUM_HEADS}` is \
                     missing. The group is read all-or-nothing so a half-stamped artifact \
                     fails at load instead of silently defaulting a geometry axis \
                     (FR-EX-08). Either stamp both keys or neither."
                )));
            }
            (Some(_), Some(_)) => {}
        }

        let num_heads = read_positive_count(file, KEY_MEDUSA_NUM_HEADS)?;
        let num_layers = read_positive_count(file, KEY_MEDUSA_NUM_LAYERS)?;
        Ok(Some(Self {
            num_heads,
            num_layers,
        }))
    }
}

/// Reads `key` as a strictly-positive unsigned integer, loud on every other
/// shape. `0` is refused because both axes are counts whose zero value would
/// describe a model that is not a Medusa model at all.
fn read_positive_count(file: &GgufFile, key: &str) -> Result<usize> {
    let raw = file.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "whisper-medusa: `{key}` is present but is not an unsigned integer \
             (FR-EX-08 — refusing to guess a geometry axis)."
        ))
    })?;
    if raw == 0 {
        return Err(VokraError::ModelLoad(format!(
            "whisper-medusa: `{key}` is 0 — a zero count describes a checkpoint that is \
             not a Medusa model at all. Refusing the `0` sentinel rather than binding a \
             degenerate geometry (FR-EX-08)."
        )));
    }
    usize::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "whisper-medusa: `{key}` is {raw}, which does not fit in this platform's \
             `usize` — refusing rather than truncating a geometry axis (FR-EX-08)."
        ))
    })
}

// ---------------------------------------------------------------------------
// MedusaHeads — the real prefix walk over the tensor manifest.
// ---------------------------------------------------------------------------

/// One Medusa head's tensors, as discovered on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MedusaHead {
    /// Head index parsed out of the tensor name (`medusa_head.{index}.…`).
    pub index: usize,
    /// The head's tensors: verbatim upstream name plus GGUF-side dims.
    pub tensors: Vec<(String, Vec<usize>)>,
}

/// The Medusa speculative-decoding head family bound from a GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification step.
/// Every rejection names the offending tensor or index so a mis-produced
/// artifact has exactly one place to walk (FR-EX-08 — never a silent partial
/// bind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MedusaHeads {
    /// Which of [`MEDUSA_HEAD_PREFIXES`] this artifact actually uses.
    prefix: &'static str,
    /// Heads, sorted ascending by index and verified contiguous from 0.
    heads: Vec<MedusaHead>,
    /// Number of base Whisper encoder tensors observed (presence gate).
    base_encoder_tensors: usize,
}

impl MedusaHeads {
    /// Walks `gguf`'s tensor manifest and groups the Medusa head tensors.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the artifact mixes both spellings from
    ///   [`MEDUSA_HEAD_PREFIXES`] (ambiguous grouping).
    /// - [`VokraError::ModelLoad`] when a tensor carries a Medusa prefix but no
    ///   parseable head index — the offending tensor is named.
    /// - [`VokraError::ModelLoad`] when no Medusa tensor is present at all
    ///   (a plain Whisper checkpoint mis-stamped as `whisper-medusa-v1`).
    /// - [`VokraError::ModelLoad`] when the head index set is not contiguous
    ///   from 0 — the first missing index is named.
    /// - [`VokraError::ModelLoad`] when no base Whisper encoder tensor is
    ///   present — the expected prefixes are named.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // 1. Decide which spelling this artifact uses, and refuse a mix.
        let mut seen_prefixes: Vec<&'static str> = Vec::new();
        for candidate in MEDUSA_HEAD_PREFIXES {
            if gguf.tensors().iter().any(|t| t.name.starts_with(candidate)) {
                seen_prefixes.push(candidate);
            }
        }
        if seen_prefixes.len() > 1 {
            return Err(VokraError::ModelLoad(format!(
                "whisper-medusa: GGUF carries BOTH Medusa head spellings \
                 ({seen:?}) — grouping heads by index would be ambiguous, so this is \
                 refused rather than silently preferring one (FR-EX-08). A legitimate \
                 `{UPSTREAM_HF}` checkpoint uses exactly one `nn.ModuleList` attribute \
                 name; a GGUF carrying both is a merge artifact. Primary source: \
                 {PRIMARY_SOURCE_HF}.",
                seen = seen_prefixes,
            )));
        }

        // 2. Group the head tensors, refusing a malformed index by name.
        let mut prefix: &'static str = MEDUSA_HEAD_PREFIXES[0];
        let mut grouped: Vec<MedusaHead> = Vec::new();
        if let Some(&found) = seen_prefixes.first() {
            prefix = found;
            for info in gguf.tensors() {
                let Some(rest) = info.name.strip_prefix(prefix) else {
                    continue;
                };
                let segment = rest.split('.').next().unwrap_or("");
                let Ok(index) = segment.parse::<usize>() else {
                    return Err(VokraError::ModelLoad(format!(
                        "whisper-medusa: tensor `{name}` carries the Medusa prefix \
                         `{prefix}` but its head index segment `{segment}` is not a \
                         non-negative integer. Upstream Medusa names every head tensor \
                         `{prefix}{{head_index}}.…` (an `nn.ModuleList`), so an \
                         unparseable segment means this GGUF was produced by something \
                         other than `vokra-cli convert --model whisper-medusa-v1`, or the \
                         upstream layout changed. Refusing rather than guessing which head \
                         this tensor belongs to (FR-EX-08). Primary source: \
                         {PRIMARY_SOURCE_HF}.",
                        name = info.name,
                    )));
                };
                let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                match grouped.iter_mut().find(|h| h.index == index) {
                    Some(head) => head.tensors.push((info.name.clone(), dims)),
                    None => grouped.push(MedusaHead {
                        index,
                        tensors: vec![(info.name.clone(), dims)],
                    }),
                }
            }
        }

        // 3. No Medusa tensors at all → this is not a Medusa checkpoint.
        if grouped.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "whisper-medusa: GGUF carries no tensor under any Medusa head prefix \
                 ({prefixes:?}) — refusing to bind (FR-EX-08). The whole point of the \
                 `{ARCH}` arch tag (as opposed to plain `whisper`) is the extra Medusa \
                 head tensor family; an artifact without it is either a vanilla Whisper \
                 checkpoint mis-stamped as `{ARCH}` (re-convert it with \
                 `vokra-cli convert --model whisper`) or a truncated conversion. \
                 Re-run `vokra-cli convert --model whisper-medusa-v1` against an upstream \
                 `{UPSTREAM_HF}` safetensors checkpoint. Primary source: \
                 {PRIMARY_SOURCE_HF}.",
                prefixes = MEDUSA_HEAD_PREFIXES,
            )));
        }

        // 4. Contiguity: heads must be 0..n with no gap.
        grouped.sort_by_key(|h| h.index);
        for (expected, head) in grouped.iter().enumerate() {
            if head.index != expected {
                return Err(VokraError::ModelLoad(format!(
                    "whisper-medusa: Medusa head indices are not contiguous — head \
                     {expected} is missing (next present index is {actual}, discovered \
                     indices {found:?}). Upstream Medusa heads are an `nn.ModuleList`, so \
                     indices always run 0..N-1; a gap means tensors were dropped during \
                     conversion or merge. Refusing rather than binding a partial head \
                     family and reporting a wrong head count (FR-EX-08).",
                    actual = head.index,
                    found = grouped.iter().map(|h| h.index).collect::<Vec<_>>(),
                )));
            }
        }

        // 5. Base tower presence: Medusa heads without a Whisper encoder are
        //    not a runnable checkpoint (the heads read the decoder's hidden
        //    state; there is nothing to read without the tower).
        let base_encoder_tensors = gguf
            .tensors()
            .iter()
            .filter(|t| BASE_ENCODER_PREFIXES.iter().any(|p| t.name.starts_with(p)))
            .count();
        if base_encoder_tensors == 0 {
            return Err(VokraError::ModelLoad(format!(
                "whisper-medusa: GGUF carries {n} Medusa head(s) but no base Whisper \
                 encoder tensor (expected a tensor named under one of {prefixes:?}). \
                 Medusa heads read the base decoder's final hidden state — without the \
                 Whisper tower there is nothing for them to read, so this artifact cannot \
                 transcribe. Refusing rather than binding heads that can never run \
                 (FR-EX-08). Re-run `vokra-cli convert --model whisper-medusa-v1` against \
                 a complete `{UPSTREAM_HF}` safetensors checkpoint.",
                n = grouped.len(),
                prefixes = BASE_ENCODER_PREFIXES,
            )));
        }

        Ok(Self {
            prefix,
            heads: grouped,
            base_encoder_tensors,
        })
    }

    /// Which of [`MEDUSA_HEAD_PREFIXES`] this artifact uses.
    #[inline]
    #[must_use]
    pub const fn prefix(&self) -> &'static str {
        self.prefix
    }

    /// Number of Medusa heads discovered (contiguous `0..len`).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    /// Always `false` — [`from_gguf`](Self::from_gguf) refuses an empty head
    /// family. Present so clippy's `len_without_is_empty` is satisfied and so
    /// callers can write the idiomatic check.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// The discovered heads, ascending by index.
    #[inline]
    #[must_use]
    pub fn heads(&self) -> &[MedusaHead] {
        &self.heads
    }

    /// Number of base Whisper encoder tensors observed alongside the heads.
    #[inline]
    #[must_use]
    pub const fn base_encoder_tensors(&self) -> usize {
        self.base_encoder_tensors
    }
}

// ---------------------------------------------------------------------------
// Base tower delegation state.
// ---------------------------------------------------------------------------

/// Outcome of delegating the base Whisper tower load to
/// [`crate::whisper::WhisperAsr::from_gguf`].
///
/// Recorded rather than swallowed: today's converter stamps no
/// `vokra.whisper.*` / `vokra.frontend.*` chunk, so `Unavailable` is the
/// universal case, and every transcription entry point turns it into a loud
/// error quoting `reason` verbatim (FR-EX-08).
enum BaseTower {
    /// The base Whisper tower bound — real transcription is available.
    Bound(Box<WhisperAsr>),
    /// The base tower did not bind; `.0` is the underlying error text.
    Unavailable(String),
}

/// Whether the base Whisper tower bound, for callers that want to branch
/// before calling a transcription entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseTowerStatus {
    /// Real base transcription is available on this handle.
    Bound,
    /// The base tower did not load; carries the underlying error text.
    Unavailable(String),
}

// ---------------------------------------------------------------------------
// WhisperMedusa — the runtime binder handle.
// ---------------------------------------------------------------------------

/// aiola **Whisper-Medusa-v1** (`aiola/whisper-medusa-v1`) runtime binder.
///
/// Bind with [`from_gguf`](Self::from_gguf). Base (non-speculative)
/// transcription runs through the shared, real [`crate::whisper`] tower;
/// speculative decoding is loud-partial — see the module docstring's
/// implementation-status matrix.
///
/// Not `Debug`: it holds a [`WhisperAsr`], which owns an `Arc<WhisperModel>`
/// that does not derive `Debug`. Use let-else (`let Err(e) = … else { … }`) in
/// tests rather than `unwrap_err()`.
pub struct WhisperMedusa {
    heads: MedusaHeads,
    config: Option<MedusaConfig>,
    base: BaseTower,
    weight_license: LicenseClass,
}

impl WhisperMedusa {
    /// Binds a Whisper-Medusa-v1 GGUF: validates arch, discovers the Medusa
    /// head family, reads the optional `vokra.medusa.*` group, delegates the
    /// base Whisper tower load, and surfaces the stamped weight-license class.
    ///
    /// Succeeding when the base tower fails to load is **deliberate**: the
    /// artifact genuinely is a valid Medusa GGUF per the converter contract
    /// (which stamps no `vokra.whisper.*`), and refusing it here would re-open
    /// the very "converted but unloadable" gap this module closes. The failure
    /// is not swallowed — it is recorded and re-raised loudly at every
    /// transcription entry point. Check up front with
    /// [`base_status`](Self::base_status).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent, or is
    ///   present but not [`ARCH`].
    /// - [`VokraError::ModelLoad`] from [`MedusaHeads::from_gguf`] — no Medusa
    ///   tensors, mixed prefix spellings, malformed head index, non-contiguous
    ///   head indices, or no base Whisper encoder tensor.
    /// - [`VokraError::ModelLoad`] from [`MedusaConfig::from_gguf`] — a
    ///   partially stamped or malformed `vokra.medusa.*` group.
    /// - [`VokraError::ModelLoad`] when a stamped `vokra.medusa.num_heads`
    ///   disagrees with the head count discovered on disk.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch gate first, so a mis-routed GGUF fails with a specific
        //    message instead of a downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "whisper-medusa: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model whisper-medusa-v1`?). \
                     The vanilla-Whisper-topology fleet — {siblings:?} — shares this \
                     model's encoder/decoder tensor names verbatim and differs only in \
                     provenance, license and decoder depth, so ONLY the arch tag can tell \
                     them apart. None of them carries the Medusa speculative-decoding \
                     head family, so binding one here would look for `{head_prefix}*` \
                     tensors that do not exist; conversely a `{ARCH}` GGUF handed to the \
                     vanilla Whisper binder would bind the base tower and SILENTLY DROP \
                     the heads — which is exactly why `{ARCH}` is excluded from \
                     `whisper::ACCEPTED_ARCHS` (FR-EX-08 — no silent partial load). \
                     Primary source: {PRIMARY_SOURCE_HF}.",
                    siblings = VANILLA_WHISPER_SIBLINGS,
                    head_prefix = MEDUSA_HEAD_PREFIXES[0],
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "whisper-medusa: GGUF is missing `{key}` — this is not a Vokra-native \
                     whisper-medusa GGUF (was it produced by `vokra-cli convert --model \
                     whisper-medusa-v1`?). Refusing to bind an arch-unlabeled artifact \
                     (FR-EX-08).",
                    key = chunks::KEY_MODEL_ARCH,
                )));
            }
        }

        // 2. Medusa head family (real prefix walk, loud on every anomaly).
        let heads = MedusaHeads::from_gguf(file)?;

        // 3. Optional all-or-nothing `vokra.medusa.*` group, cross-checked
        //    against what is actually on disk.
        let config = MedusaConfig::from_gguf(file)?;
        if let Some(cfg) = config {
            if cfg.num_heads != heads.len() {
                return Err(VokraError::ModelLoad(format!(
                    "whisper-medusa: `{KEY_MEDUSA_NUM_HEADS}` is stamped as {stamped} but \
                     {found} Medusa head(s) were discovered on disk under the `{prefix}` \
                     prefix. The metadata and the tensor manifest disagree, so one of them \
                     is wrong; refusing rather than picking a winner and silently decoding \
                     with the wrong head count (FR-EX-08).",
                    stamped = cfg.num_heads,
                    found = heads.len(),
                    prefix = heads.prefix(),
                )));
            }
        }

        // 4. Base Whisper tower — reuse, never fork. `WhisperAsr::from_gguf`
        //    is the arch-agnostic load path (the arch gate lives on
        //    `WhisperSession`), and it binds on exactly the HF-verbatim tensor
        //    names this converter passes through. Today's converter stamps no
        //    `vokra.whisper.*` / `vokra.frontend.*`, so this normally fails —
        //    the error is RECORDED, not swallowed, and re-raised loudly by
        //    every transcription entry point.
        let base = match WhisperAsr::from_gguf(file) {
            Ok(asr) => BaseTower::Bound(Box::new(asr)),
            Err(e) => BaseTower::Unavailable(e.to_string()),
        };

        // 5. Provenance surfacing — fail-closed to `Unknown` when unstamped.
        //    This only REPORTS the artifact's stamp; it is not a license
        //    determination (see the module docstring — §3.1 sign-off is
        //    owner-only and this row is flagged 要一次 / primary source
        //    required).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            heads,
            config,
            base,
            weight_license,
        })
    }

    /// Attaches a Whisper detokenizer to the base tower.
    ///
    /// The Medusa converter embeds no `vokra.tokenizer.model` blob, so
    /// [`transcribe`](Self::transcribe) needs one attached here (or an
    /// artifact that carries the blob) before it can render text;
    /// [`transcribe_tokens`](Self::transcribe_tokens) needs none.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] when the base tower did not bind —
    ///   attaching a tokenizer to a tower that does not exist would silently
    ///   drop it, so this is loud instead (FR-EX-08).
    pub fn with_tokenizer(self, tokenizer: WhisperTokenizer) -> Result<Self> {
        let Self {
            heads,
            config,
            base,
            weight_license,
        } = self;
        let base = match base {
            BaseTower::Bound(asr) => BaseTower::Bound(Box::new(asr.with_tokenizer(tokenizer))),
            BaseTower::Unavailable(reason) => {
                return Err(base_tower_loud_partial("with_tokenizer", &reason));
            }
        };
        Ok(Self {
            heads,
            config,
            base,
            weight_license,
        })
    }

    /// Selects the backend the base tower's forward runs on.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] when the base tower did not bind.
    pub fn with_backend(self, backend: BackendKind) -> Result<Self> {
        let Self {
            heads,
            config,
            base,
            weight_license,
        } = self;
        let base = match base {
            BaseTower::Bound(asr) => BaseTower::Bound(Box::new(asr.with_backend(backend))),
            BaseTower::Unavailable(reason) => {
                return Err(base_tower_loud_partial("with_backend", &reason));
            }
        };
        Ok(Self {
            heads,
            config,
            base,
            weight_license,
        })
    }

    /// The Medusa head family discovered on disk.
    #[inline]
    #[must_use]
    pub const fn heads(&self) -> &MedusaHeads {
        &self.heads
    }

    /// Number of Medusa heads discovered on disk.
    #[inline]
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// The optional `vokra.medusa.*` group, `None` when unstamped (the
    /// universal case with today's converter).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> Option<MedusaConfig> {
        self.config
    }

    /// Whether the base Whisper tower bound, and why not when it did not.
    #[must_use]
    pub fn base_status(&self) -> BaseTowerStatus {
        match &self.base {
            BaseTower::Bound(_) => BaseTowerStatus::Bound,
            BaseTower::Unavailable(reason) => BaseTowerStatus::Unavailable(reason.clone()),
        }
    }

    /// The base Whisper engine, for callers that want the full Whisper surface
    /// (beam search, n-best, sampling) on the Medusa checkpoint's base tower.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] when the base tower did not bind,
    ///   quoting the underlying load error.
    pub fn base_asr(&self) -> Result<&WhisperAsr> {
        match &self.base {
            BaseTower::Bound(asr) => Ok(asr),
            BaseTower::Unavailable(reason) => Err(base_tower_loud_partial("base_asr", reason)),
        }
    }

    /// The stamped weight-license class surfaced from
    /// `vokra.provenance.weight_license`, fail-closed to
    /// [`LicenseClass::Unknown`] when the stamp is absent.
    ///
    /// This **reports** the artifact's stamp; it is not a license
    /// determination. The converter's default
    /// ([`CONVERTER_DEFAULT_LICENSE`]) is a ticket-header assumption and the
    /// `docs/license-audit.md` §3.1 sign-off for this row is owner-only and
    /// still blank.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Base (non-speculative) greedy transcription to raw token ids, through
    /// the shared [`crate::whisper`] tower.
    ///
    /// Real when the base tower bound. Speculation is **not** applied — this
    /// is ordinary Whisper decoding on the Medusa checkpoint's base weights,
    /// which by Medusa's own correctness argument produces exactly the token
    /// sequence a correct speculative decode would accept, only without the
    /// throughput gain.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] when the base tower did not bind,
    ///   quoting the underlying load error.
    /// - Whatever [`crate::whisper::WhisperAsr::transcribe_tokens`] returns.
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        match &self.base {
            BaseTower::Bound(asr) => asr.transcribe_tokens(pcm),
            BaseTower::Unavailable(reason) => {
                Err(base_tower_loud_partial("transcribe_tokens", reason))
            }
        }
    }

    /// Base (non-speculative) transcription to text.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] when the base tower did not bind.
    /// - Whatever the underlying [`crate::whisper::WhisperAsr`] returns
    ///   (including a missing-detokenizer error — see
    ///   [`with_tokenizer`](Self::with_tokenizer)).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        match &self.base {
            BaseTower::Bound(asr) => asr.transcribe(pcm),
            BaseTower::Unavailable(reason) => Err(base_tower_loud_partial("transcribe", reason)),
        }
    }

    /// **Loud-partial**: speculative (Medusa) decoding.
    ///
    /// Always returns [`VokraError::UnsupportedOp`], including when the base
    /// tower bound successfully — a "speculative" call that silently ran plain
    /// decoding would misreport what the runtime did, which is exactly the
    /// fake-complete failure mode FR-EX-08 forbids.
    ///
    /// The error names three blockers (no draft→verify→accept driver and no
    /// tree-attention mask op anywhere in the workspace; the `medusa_choices`
    /// candidate tree is recorded in no `vokra.medusa.*` metadata; the per-head
    /// sub-module layout needs a real-checkpoint walk), points at the primary
    /// sources, and names [`transcribe`](Self::transcribe) as the honest
    /// same-output fallback. **No fabricated speculative tokens are ever
    /// emitted.**
    ///
    /// # Errors
    ///
    /// - Always [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn transcribe_speculative(&self, _pcm: &[f32]) -> Result<Vec<u32>> {
        // Bind explicitly so an unused-parameter lint cannot mask a future
        // accidental removal of the argument (sibling loud-partial discipline).
        let _ = _pcm;
        Err(speculative_decode_loud_partial(self.heads.len()))
    }
}

impl AsrEngine for WhisperMedusa {
    /// Base (non-speculative) transcription — the same body as the inherent
    /// [`WhisperMedusa::transcribe`].
    ///
    /// Deliberately **not** written as `Self::transcribe(self, pcm)`: with an
    /// inherent method of the same name in scope, that path form is resolved
    /// by inherent-impl-first precedence, which is a subtle rule to rely on
    /// when getting it wrong yields silent infinite recursion. The four-line
    /// duplication buys an unambiguous call.
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        match &self.base {
            BaseTower::Bound(asr) => asr.transcribe(pcm),
            BaseTower::Unavailable(reason) => Err(base_tower_loud_partial("transcribe", reason)),
        }
    }

    /// Asks the base tower rather than storing a second copy: the backend is
    /// set through [`WhisperMedusa::with_backend`], which forwards to the
    /// bound [`WhisperAsr`], so a duplicate field here could disagree with
    /// the engine that actually runs.
    ///
    /// The unavailable arm reports `Cpu`, which cannot mislead in the way
    /// the trait warns about: without a bound tower every transcription
    /// entry point is a loud partial, so nothing executes anywhere for the
    /// answer to contradict.
    fn backend(&self) -> BackendKind {
        match &self.base {
            BaseTower::Bound(asr) => asr.backend(),
            BaseTower::Unavailable(_) => BackendKind::Cpu,
        }
    }
}

/// The loud error raised whenever the base Whisper tower is needed but did not
/// bind. Quotes the underlying [`crate::whisper`] error verbatim so the reader
/// sees the real cause (normally: the Medusa converter stamps no
/// `vokra.whisper.*` hyper-parameter chunk).
fn base_tower_loud_partial(surface: &str, reason: &str) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "whisper-medusa {surface} (loud-partial): the base Whisper tower did not bind, so \
         there is nothing to decode with. Underlying error: {reason}. \
         Root cause (expected): `crates/vokra-convert/src/models/whisper_medusa_v1.rs` is a \
         pure tensor pass-through — it stamps `vokra.model.*` + `vokra.provenance.*` + \
         `vokra.schema.*` and NOT the `vokra.whisper.*` hyper-parameter chunk that \
         `whisper::WhisperConfig::from_gguf` requires, nor the `vokra.frontend.*` chunk the \
         FR-LD-03 bit-exact front-end check requires, nor a `vokra.tokenizer.model` blob. \
         The tensor NAMES already line up (the converter passes through HF-verbatim names \
         and `whisper::WhisperWeights` binds on exactly those), so the fix is metadata, not \
         weights: extend the converter to stamp `vokra.whisper.*` + `vokra.frontend.*` \
         (mirror of `crates/vokra-convert/src/models/whisper.rs`), then this binder \
         transcribes with no change here. Runtime cannot fabricate a transcript \
         (FR-EX-08 — no silent partial output). Primary sources: {PRIMARY_SOURCE_HF}, \
         {PRIMARY_SOURCE_PAPER}; in-repo ticket {PRIMARY_SOURCE_TICKET}."
    ))
}

/// The loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`WhisperMedusa::transcribe_speculative`].
///
/// Mirror of the sibling loud-partial message precedent (CLAUDE.md 教訓 (a)):
/// name the surface, name every missing piece by exact identifier, cite the
/// primary sources, and state the honest fallback.
fn speculative_decode_loud_partial(num_heads: usize) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "whisper-medusa transcribe_speculative (loud-partial): {num_heads} Medusa head(s) \
         are bound, but the speculative decode loop is not implemented — three pieces must \
         land before real speculative tokens can be emitted: \
         (1) a draft-verify-accept DRIVER: `vokra_core::decode` carries `beam_search` / \
         `sample_sequence` and the CTC/RNN-T family, but no speculative driver, and \
         `vokra-ops` carries no tree/sparse attention mask op, so the candidate \
         continuations cannot be verified in one batched decoder forward; \
         (2) the MEDUSA CANDIDATE TREE (`medusa_choices` upstream — which head feeds which \
         continuation at which top-k rank) is recorded NOWHERE in the artifact: the \
         converter stamps no `vokra.medusa.*` group at all, and a guessed tree yields a \
         shape-valid decode that silently ACCEPTS WRONG TOKENS; \
         (3) the PER-HEAD SUB-MODULE LAYOUT (residual-block depth inside each head, and \
         whether the final projection carries a bias) cannot be transcribed from any \
         source the converter records — this binder groups tensors by head index, which is \
         real and verifiable, but does not know the shape of a head's interior. \
         HONEST FALLBACK: Medusa is a throughput optimisation, not a different model — the \
         prefix a correct speculative decode accepts is by construction identical to plain \
         decoding, so call `WhisperMedusa::transcribe` / `transcribe_tokens` for the SAME \
         output today (base tower, no speedup). This entry point deliberately does NOT \
         silently forward to it, because reporting plain decoding as speculative would \
         misstate what the runtime did. No fabricated speculative tokens are ever emitted \
         (FR-EX-08 — no silent partial output). Primary sources: {PRIMARY_SOURCE_HF}, \
         {PRIMARY_SOURCE_PAPER}; in-repo ticket {PRIMARY_SOURCE_TICKET}."
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the Whisper-Medusa-v1 runtime binder.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real `aiola/whisper-medusa-v1` checkpoint this would be
    //! `transcribe(...)` returning text. Today's converter stamps no
    //! `vokra.whisper.*` / `vokra.frontend.*` chunk, so no synthetic fixture
    //! this file can build reaches a real forward, and fabricating one would
    //! violate CLAUDE.md 教訓 (a)「loud-partial は fake-complete より honest」.
    //!
    //! The semantics that CAN be tested honestly, and are:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` match the converter's values exactly, and `ARCH` is
    //!    NOT in `whisper::ACCEPTED_ARCHS` (the anti-regression pin for the
    //!    "do not admit Medusa into the vanilla binder" rule).
    //! 2. **Metadata round-trip** — a synthetic GGUF with the right arch,
    //!    Medusa head tensors and base encoder tensors binds, reports the head
    //!    count, and surfaces the license stamp (Permissive when stamped,
    //!    Unknown when absent).
    //! 3. **Loud negative space** — every documented rejection fires at its
    //!    documented surface, in the documented variant, naming the offending
    //!    tensor / index / key.
    //! 4. **Loud-partial round-trip** — the base-tower and speculative gates
    //!    both fire and both name their missing pieces.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a synthetic Whisper-Medusa GGUF.
    ///
    /// `head_indices` chooses which Medusa head indices exist, `prefix` which
    /// spelling they use, and `with_base_encoder` whether a base Whisper
    /// encoder tensor is present. No `vokra.whisper.*` / `vokra.frontend.*` is
    /// written — matching what the real converter produces, so the base tower
    /// deliberately does not bind (see the module doc).
    fn medusa_gguf(
        arch: Option<&str>,
        head_indices: &[usize],
        prefix: &str,
        with_base_encoder: bool,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.add_string(chunks::KEY_MODEL_ARCH, a);
        }
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        b.add_string("vokra.provenance.upstream_hf", UPSTREAM_HF);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        if with_base_encoder {
            // The HF-verbatim name `whisper::WhisperWeights` also binds on.
            b.add_tensor(
                "model.encoder.layers.0.self_attn.q_proj.weight",
                GgmlType::F32,
                vec![4, 4],
                vec![0u8; 4 * 4 * 4],
            )
            .expect("add_tensor");
        }
        for &i in head_indices {
            b.add_tensor(
                &format!("{prefix}{i}.linear.weight"),
                GgmlType::F32,
                vec![4, 4],
                vec![0u8; 4 * 4 * 4],
            )
            .expect("add_tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// The canonical happy-path fixture: arch stamped, 3 contiguous heads,
    /// base encoder present, Permissive license stamped.
    fn valid_medusa_gguf() -> GgufFile {
        medusa_gguf(
            Some(ARCH),
            &[0, 1, 2],
            MEDUSA_HEAD_PREFIXES[0],
            true,
            Some(LicenseClass::Permissive),
        )
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin, including the anti-regression pin that
    //          keeps `whisper-medusa-v1` OUT of the vanilla Whisper binder.
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_pin_the_converter_values() {
        assert_eq!(ARCH, "whisper-medusa-v1", "arch tag pin");
        assert_eq!(NAME, "whisper-medusa-v1", "model name pin");
        assert_eq!(CATEGORY, "asr", "category tier pin");
        assert_eq!(UPSTREAM_HF, "aiola/whisper-medusa-v1", "upstream slug pin");
        assert_eq!(
            CONVERTER_DEFAULT_LICENSE, "apache-2.0",
            "the converter's ticket-header-derived DEFAULT (an assumption, not a \
             verified license — docs/license-audit.md §3.1 sign-off is owner-only)"
        );
        assert_eq!(MEDUSA_HEAD_PREFIXES[0], "medusa_head.");

        // Anti-regression: the vanilla Whisper binder must never accept this
        // arch — doing so would bind the base tower and silently drop the
        // Medusa heads, which `whisper::ACCEPTED_ARCHS`'s docstring calls out
        // explicitly.
        assert!(
            !crate::whisper::ACCEPTED_ARCHS.contains(&ARCH),
            "`{ARCH}` must stay OUT of whisper::ACCEPTED_ARCHS — admitting it would \
             silently drop the Medusa heads (FR-EX-08)"
        );
        // Conversely every sibling this binder enumerates really is served by
        // the vanilla binder, so the mismatch message is accurate.
        for sibling in VANILLA_WHISPER_SIBLINGS {
            assert!(
                crate::whisper::ACCEPTED_ARCHS.contains(&sibling),
                "`{sibling}` is enumerated as a vanilla-Whisper sibling but is not in \
                 whisper::ACCEPTED_ARCHS — the arch-mismatch message would be wrong"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 — Happy path: a synthetic GGUF with the right tensors binds.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_a_well_formed_medusa_gguf() {
        let file = valid_medusa_gguf();
        let Ok(m) = WhisperMedusa::from_gguf(&file) else {
            panic!("a well-formed whisper-medusa GGUF must bind");
        };
        assert_eq!(m.num_heads(), 3, "three contiguous Medusa heads discovered");
        assert_eq!(m.heads().prefix(), MEDUSA_HEAD_PREFIXES[0]);
        assert_eq!(
            m.heads().base_encoder_tensors(),
            1,
            "the base Whisper encoder tensor must be counted by the presence gate"
        );
        assert!(!m.heads().is_empty());
        assert_eq!(
            m.heads()
                .heads()
                .iter()
                .map(|h| h.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2],
            "heads must be sorted ascending and contiguous from 0"
        );
        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "the stamped class must round-trip"
        );
        assert_eq!(
            m.config(),
            None,
            "today's converter stamps no `vokra.medusa.*` group"
        );
    }

    /// The plural spelling is tolerated (see [`MEDUSA_HEAD_PREFIXES`]).
    #[test]
    fn from_gguf_accepts_the_plural_head_prefix_spelling() {
        let file = medusa_gguf(Some(ARCH), &[0, 1], MEDUSA_HEAD_PREFIXES[1], true, None);
        let Ok(m) = WhisperMedusa::from_gguf(&file) else {
            panic!("the plural `medusa_heads.` spelling must also bind");
        };
        assert_eq!(m.num_heads(), 2);
        assert_eq!(m.heads().prefix(), MEDUSA_HEAD_PREFIXES[1]);
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an unstamped artifact must fail-closed to Unknown"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 — arch metadata absent → loud ModelLoad.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let file = medusa_gguf(None, &[0], MEDUSA_HEAD_PREFIXES[0], true, None);
        let Err(err) = WhisperMedusa::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("vokra.model.arch"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native whisper-medusa GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — foreign arch → loud ModelLoad naming BOTH expected and actual.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // Plain `whisper` is the most dangerous confusion: identical tensor
        // topology, no Medusa heads.
        let file = medusa_gguf(Some("whisper"), &[0], MEDUSA_HEAD_PREFIXES[0], true, None);
        let Err(err) = WhisperMedusa::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`whisper`"),
                    "message must name the ACTUAL arch, got `{m}`"
                );
                assert!(
                    m.contains("`whisper-medusa-v1`"),
                    "message must name the EXPECTED arch, got `{m}`"
                );
                for sibling in VANILLA_WHISPER_SIBLINGS {
                    assert!(
                        m.contains(sibling),
                        "message must enumerate the sibling `{sibling}`: {m}"
                    );
                }
                assert!(
                    m.contains("SILENTLY DROP"),
                    "message must explain why the vanilla binder must not take this \
                     arch, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — missing tensor family → loud error naming the tensor prefix.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_a_gguf_with_no_medusa_head_tensors() {
        // Right arch, base tower present, but zero Medusa tensors: a vanilla
        // Whisper checkpoint mis-stamped as whisper-medusa-v1.
        let file = medusa_gguf(Some(ARCH), &[], MEDUSA_HEAD_PREFIXES[0], true, None);
        let Err(err) = WhisperMedusa::from_gguf(&file) else {
            panic!("expected ModelLoad when no Medusa head tensor is present");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("medusa_head."),
                    "message must name the expected tensor prefix, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model whisper-medusa-v1"),
                    "message must include the repro command, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// Medusa heads present but the base Whisper tower's tensors absent — the
    /// heads have nothing to read, so this is refused by name.
    #[test]
    fn from_gguf_rejects_medusa_heads_without_a_base_encoder() {
        let file = medusa_gguf(Some(ARCH), &[0, 1], MEDUSA_HEAD_PREFIXES[0], false, None);
        let Err(err) = WhisperMedusa::from_gguf(&file) else {
            panic!("expected ModelLoad when the base Whisper encoder is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("model.encoder.layers."),
                    "message must name the expected base-encoder prefix, got `{m}`"
                );
                assert!(
                    m.contains("no base Whisper"),
                    "message must name the missing family, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// A gap in the head index set is a dropped-tensor signal; the first
    /// missing index must be named.
    #[test]
    fn from_gguf_rejects_non_contiguous_head_indices() {
        let file = medusa_gguf(Some(ARCH), &[0, 2], MEDUSA_HEAD_PREFIXES[0], true, None);
        let Err(err) = WhisperMedusa::from_gguf(&file) else {
            panic!("expected ModelLoad on a head-index gap");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("not contiguous"),
                    "message must name the contiguity gap, got `{m}`"
                );
                assert!(
                    m.contains("head 1 is missing"),
                    "message must name the FIRST missing index, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// A tensor under the Medusa prefix whose head-index segment does not
    /// parse must be refused BY NAME rather than silently skipped.
    #[test]
    fn from_gguf_rejects_a_malformed_head_index_naming_the_tensor() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_tensor(
            "model.encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        b.add_tensor(
            "medusa_head.oops.linear.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = WhisperMedusa::from_gguf(&file) else {
            panic!("expected ModelLoad on a malformed head index");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("medusa_head.oops.linear.weight"),
                    "message must NAME the offending tensor, got `{m}`"
                );
                assert!(
                    m.contains("`oops`"),
                    "message must name the unparseable segment, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// Both head spellings in one artifact is a merge artifact — ambiguous, so
    /// refused rather than silently preferring one.
    #[test]
    fn from_gguf_rejects_mixed_head_prefix_spellings() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_tensor(
            "model.encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        for name in [
            "medusa_head.0.linear.weight",
            "medusa_heads.0.linear.weight",
        ] {
            b.add_tensor(name, GgmlType::F32, vec![2, 2], vec![0u8; 16])
                .expect("add_tensor");
        }
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = WhisperMedusa::from_gguf(&file) else {
            panic!("expected ModelLoad on mixed head prefixes");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("BOTH Medusa head spellings"),
                    "message must name the ambiguity, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — the optional `vokra.medusa.*` group.
    // -----------------------------------------------------------------------

    #[test]
    fn medusa_config_group_is_all_or_nothing_and_cross_checked() {
        // (a) Absent → Ok(None), the checkpoint still binds.
        let file = valid_medusa_gguf();
        assert_eq!(MedusaConfig::from_gguf(&file).expect("absent is Ok"), None);

        // (b) Partially stamped → loud, naming the missing key.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(KEY_MEDUSA_NUM_HEADS, 3);
        let partial = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = MedusaConfig::from_gguf(&partial) else {
            panic!("expected ModelLoad on a partially stamped group");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains(KEY_MEDUSA_NUM_LAYERS) && m.contains("partially stamped"),
                "message must name the missing key, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (c) `0` sentinel → loud, naming the key.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(KEY_MEDUSA_NUM_HEADS, 0);
        b.add_u32(KEY_MEDUSA_NUM_LAYERS, 1);
        let zeroed = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = MedusaConfig::from_gguf(&zeroed) else {
            panic!("expected ModelLoad on the 0 sentinel");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains(KEY_MEDUSA_NUM_HEADS) && m.contains("is 0"),
                "message must name the zeroed key, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (d) Fully stamped and agreeing with disk → binds, `config()` set.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(KEY_MEDUSA_NUM_HEADS, 2);
        b.add_u32(KEY_MEDUSA_NUM_LAYERS, 1);
        b.add_tensor(
            "model.encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        for i in 0..2 {
            b.add_tensor(
                &format!("medusa_head.{i}.linear.weight"),
                GgmlType::F32,
                vec![2, 2],
                vec![0u8; 16],
            )
            .expect("add_tensor");
        }
        let agreeing = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Ok(m) = WhisperMedusa::from_gguf(&agreeing) else {
            panic!("an agreeing `vokra.medusa.*` group must bind");
        };
        assert_eq!(
            m.config(),
            Some(MedusaConfig {
                num_heads: 2,
                num_layers: 1
            })
        );

        // (e) Stamped head count disagreeing with disk → loud.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(KEY_MEDUSA_NUM_HEADS, 5);
        b.add_u32(KEY_MEDUSA_NUM_LAYERS, 1);
        b.add_tensor(
            "model.encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        b.add_tensor(
            "medusa_head.0.linear.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let disagreeing = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = WhisperMedusa::from_gguf(&disagreeing) else {
            panic!("expected ModelLoad when metadata and manifest disagree");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains(KEY_MEDUSA_NUM_HEADS) && m.contains("disagree"),
                "message must name the disagreement, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — base-tower loud-partial: today's converter output cannot
    //          transcribe, and says exactly why.
    // -----------------------------------------------------------------------

    #[test]
    fn transcribe_loud_partials_when_the_base_tower_did_not_bind() {
        let file = valid_medusa_gguf();
        let Ok(m) = WhisperMedusa::from_gguf(&file) else {
            panic!("the fixture must bind — the base-tower gap is not a bind failure");
        };

        // The status accessor must report the gap up front.
        match m.base_status() {
            BaseTowerStatus::Unavailable(reason) => assert!(
                !reason.is_empty(),
                "the recorded reason must quote the underlying whisper error"
            ),
            BaseTowerStatus::Bound => panic!(
                "a converter-shaped GGUF carries no `vokra.whisper.*` chunk, so the base \
                 tower cannot have bound — if this ever passes, the converter grew the \
                 stamp and this test should be updated to assert real transcription"
            ),
        }

        // 1 s of silence at 16 kHz mono (the Whisper input convention).
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.transcribe_tokens(&pcm) else {
            panic!("transcribe_tokens must loud-partial without a base tower");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("whisper-medusa transcribe_tokens"),
                    "surface must be named: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");
                assert!(
                    msg.contains("vokra.whisper.*"),
                    "message must name the MISSING METADATA CHUNK (the actual blocker): {msg}"
                );
                assert!(
                    msg.contains("vokra.frontend.*"),
                    "message must name the missing front-end chunk: {msg}"
                );
                assert!(
                    msg.contains("the fix is metadata, not weights"),
                    "message must state that tensor names already line up: {msg}"
                );
                for anchor in [
                    PRIMARY_SOURCE_HF,
                    PRIMARY_SOURCE_PAPER,
                    PRIMARY_SOURCE_TICKET,
                ] {
                    assert!(msg.contains(anchor), "expected anchor '{anchor}': {msg}");
                }
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }

        // The trait method must compose to the same loud gate.
        let Err(err) = AsrEngine::transcribe(&m, &pcm) else {
            panic!("AsrEngine::transcribe must loud-partial too");
        };
        assert!(matches!(err, VokraError::UnsupportedOp(_)));

        // `base_asr` / builders must be loud rather than silently no-op.
        let Err(err) = m.base_asr() else {
            panic!("base_asr must loud-partial without a base tower");
        };
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    // -----------------------------------------------------------------------
    // Test 8 — speculative loud-partial: always fires, names all three
    //          blockers and the honest fallback.
    // -----------------------------------------------------------------------

    #[test]
    fn transcribe_speculative_always_loud_partials() {
        let file = valid_medusa_gguf();
        let Ok(m) = WhisperMedusa::from_gguf(&file) else {
            panic!("the fixture must bind");
        };
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.transcribe_speculative(&pcm) else {
            panic!("transcribe_speculative must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("whisper-medusa transcribe_speculative"),
                    "surface must be named: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // The head count actually discovered is echoed, so the reader
                // can see the heads really were bound.
                assert!(
                    msg.contains("3 Medusa head(s) are bound"),
                    "message must echo the discovered head count: {msg}"
                );

                // Blocker (1): no driver / no tree-attention op.
                assert!(
                    msg.contains("draft-verify-accept DRIVER"),
                    "message must name the missing decode driver: {msg}"
                );
                assert!(
                    msg.contains("vokra-ops"),
                    "message must name where the tree-attention op is missing: {msg}"
                );
                // Blocker (2): the candidate tree is unrecorded.
                assert!(
                    msg.contains("medusa_choices"),
                    "message must name the missing candidate tree: {msg}"
                );
                assert!(
                    msg.contains("vokra.medusa.*"),
                    "message must name the missing metadata group: {msg}"
                );
                assert!(
                    msg.contains("ACCEPTS WRONG TOKENS"),
                    "message must state the silent-wrong hazard of guessing: {msg}"
                );
                // Blocker (3): per-head interior needs a real-checkpoint walk.
                assert!(
                    msg.contains("PER-HEAD SUB-MODULE LAYOUT"),
                    "message must name the untranscribable head interior: {msg}"
                );

                // The honest fallback must be spelled out, and the deliberate
                // refusal to silently forward to it.
                assert!(
                    msg.contains("WhisperMedusa::transcribe"),
                    "message must name the same-output fallback: {msg}"
                );
                assert!(
                    msg.contains("deliberately does NOT silently forward"),
                    "message must explain why it does not just call the fallback: {msg}"
                );

                for anchor in [
                    PRIMARY_SOURCE_HF,
                    PRIMARY_SOURCE_PAPER,
                    PRIMARY_SOURCE_TICKET,
                ] {
                    assert!(msg.contains(anchor), "expected anchor '{anchor}': {msg}");
                }
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
