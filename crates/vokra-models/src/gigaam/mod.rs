//! **Sber GigaAM** (Sberbank / SberDevices, `salute-developers/GigaAM`) —
//! runtime binder for the `sber_gigaam_v3` **and** `gigaam_multilingual`
//! converter arches (Wave C1 2026-08-15 audit follow-up, loud-partial per
//! the `firered_vad` / `emotion2vec` / `sensevoicesmall_runtime` / `panns` /
//! `redimnet` precedent — CLAUDE.md 教訓 (a):
//! 「loud-partial は fake-complete より honest」).
//!
//! # The gap this module closes
//!
//! Two converters existed with **no consumer anywhere in the workspace**:
//!
//! - `crates/vokra-convert/src/models/sber_gigaam_v3.rs` stamps
//!   `vokra.model.arch = "sber_gigaam_v3"`;
//! - `crates/vokra-convert/src/models/sber_gigaam_multilingual.rs` stamps
//!   `vokra.model.arch = "gigaam_multilingual"`.
//!
//! A workspace-wide grep proved nothing read either arch string back, so
//! every converted GigaAM checkpoint was **unloadable** — the weights could
//! be produced and then nothing could bind them. This module is that
//! missing consumer, for both halves of the family at once.
//!
//! # Why one module for two arch tags
//!
//! The Wave B tickets
//! (`docs/tickets/coverage-audit-2026-08-03/wave-b/sber-gigaam-v3.md` and
//! `.../sber-gigaam-multilingual.md`) both call for a single
//! `ModelKind::GigaAm` carrying a **variant enum**, and the multilingual
//! converter's own module doc records that the split into two standalone
//! converters was a deliberate *worktree-isolation* decision, not an
//! architectural one:
//!
//! > a future refactor MAY absorb both siblings under a single
//! > `ModelKind::GigaAm` with a variant enum (`_v3` / `_multilingual` — the
//! > sibling ticket's audit note), but that refactor is deliberately
//! > deferred so this Wave B ticket can land independently
//!
//! The runtime side has no such isolation constraint: one binder that
//! accepts [`ACCEPTED_ARCHS`] and distinguishes the halves with
//! [`GigaamVariant`] is strictly better than two near-identical modules,
//! and it matches the `whisper` binder's existing `ACCEPTED_ARCHS` family
//! precedent (`whisper` / `crisper-whisper` / `distil-whisper` /
//! `kotoba-whisper` share one reader).
//!
//! What actually differs between the two variants is the **vocabulary**
//! (Russian-only char space vs a 70+-language char space) and the
//! provenance key (`vokra.provenance.upstream_hf` for v3, whose primary
//! redistribution surface is the `ai-sage` HF collection, vs
//! `vokra.provenance.upstream_url` for multilingual, whose primary surface
//! is the GitHub release because the audit ticket flags the HF mirror as
//! 要 mirror URL 確認). Both are recorded per-variant on [`GigaamVariant`].
//!
//! # Primary sources
//!
//! - Upstream repository (both variants):
//!   <https://github.com/salute-developers/GigaAM>
//! - HF surface for v3: <https://huggingface.co/ai-sage/GigaAM-v3>
//! - HF mirror for multilingual: `ai-sage/GigaAM-Multilingual` — flagged
//!   **要 mirror URL 確認** in the Wave B ticket
//!   (`proprietary-mirror-risk: medium`), which is exactly why the
//!   multilingual converter stamps the GitHub URL rather than an HF slug.
//! - License: **MIT** for both, per the upstream GitHub `LICENSE`
//!   (transcribed by both converters as `DEFAULT_LICENSE_SPDX = "mit"` →
//!   [`LicenseClass::Permissive`]).
//!
//! # Architecture (as far as primary sources in-repo state it)
//!
//! GigaAM is a **Conformer encoder + char-wise CTC head** ASR family; the
//! v3 half is a Russian / Central-Asian fine-tune, the multilingual half
//! widens the char label space to 70+ languages. The v3 converter's own
//! provenance string additionally records a `CTC/RNN-T head`, so an RNN-T
//! flavour exists in the release family.
//!
//! ```text
//! PCM (mono f32)
//!   -> log-mel / Kaldi-fbank front end       <- **loud-partial** (axes not stamped)
//!   -> Conformer encoder stack               <- **loud-partial** (tensor mapping absent)
//!   -> char-wise CTC head                    <- **loud-partial** (vocabulary absent)
//!   -> CTC greedy / prefix-beam blank fold
//!   -> Russian (v3) or 70+-language (multilingual) text
//! ```
//!
//! **The blockers are not missing kernels.** Vokra already carries every
//! primitive this composition needs — `vokra_ops::conformer` (the encoder
//! body, shared with the Parakeet-CTC / Canary binders),
//! `vokra_ops::ctc_decode_greedy` / `vokra_ops::ctc_decode_beam`, and the
//! `vokra_ops::mel` / `vokra_ops::kaldi_fbank` front ends. What is missing
//! is the **wiring information**, and it is missing from the GGUF contract
//! itself. See [`transcribe_loud_partial`].
//!
//! # Loud-partial classification (CLAUDE.md 教訓 (a))
//!
//! **REAL in this wave** — everything the GGUF contract actually supports:
//!
//! - Strict [`ACCEPTED_ARCHS`] verification that refuses a foreign GGUF
//!   loudly, naming both the observed and the expected arch tags and
//!   enumerating the sibling `category = "asr"` fleet (`parakeet-ctc` /
//!   `canary` / `omniasr-ctc` / `whisper` / `sensevoicesmall` / …) — the
//!   shared category tag alone cannot disambiguate them, only the arch tag
//!   can (FR-EX-08).
//! - A crossed-wires check: a GGUF whose arch says one variant while
//!   `vokra.model.name` says the *other* variant's name is refused, because
//!   the two disagree about which vocabulary the CTC head carries.
//! - [`GigaamWeights`] — a non-empty tensor-manifest gate, a by-name
//!   [`dims`](GigaamWeights::dims) lookup that **names** an absent tensor
//!   instead of handing back a `None` for a caller to swallow, and an
//!   optional [`KEY_REQUIRED_TENSORS`] declaration that turns a truncated
//!   or mis-merged upload into a **load-time** failure naming the first
//!   missing tensor.
//! - [`GigaamTopology`] — a real structural probe over the bound tensor
//!   names. It discovers every `<root>.layers.<i>.<leaf>` stack present in
//!   the checkpoint, counts its depth, and enforces the two invariants a
//!   homogeneous transformer stack must satisfy: **contiguity** (indices
//!   `0..n` with no hole) and **uniformity** (every layer carries the same
//!   leaf-tensor set as layer 0). A violation is a loud error naming the
//!   exact absent tensor. This is derived entirely from data present in the
//!   GGUF — nothing about it is guessed.
//! - Weight-license surfacing that fail-closes to
//!   [`LicenseClass::Unknown`] when nothing is stamped.
//!
//! **LOUD-PARTIAL in this wave**: [`Gigaam::transcribe`] returns
//! [`VokraError::UnsupportedOp`] naming three concrete blockers — see
//! [`transcribe_loud_partial`] for the full text. No fabricated
//! transcript is ever emitted (FR-EX-08 — no silent partial output).
//!
//! # Why no `GigaamConfig::upstream_default()`
//!
//! Deliberately absent, for the same reason the sibling `firered_vad` and
//! `ten_vad` binders omit theirs: **neither converter stamps a
//! `vokra.gigaam.*` hyper-parameter group**, and no in-repo primary source
//! transcribes GigaAM's encoder geometry, front-end axes or sample rate.
//! Inventing `n_layer` / `d_model` / `n_mels` / `sample_rate` numbers and
//! dressing them in an authoritative-looking constant would be exactly the
//! hallucination CLAUDE.md forbids (「ハルシネーション厳禁」), and — worse
//! — a mis-guessed geometry produces a **shape-valid but quietly wrong**
//! transcript rather than a crash. What this module exposes instead is
//! [`GigaamTopology`], which *measures* the checkpoint rather than
//! asserting anything about it.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH_V3`] / [`ARCH_MULTILINGUAL`] / the `NAME_*` / [`CATEGORY`] /
//! `UPSTREAM_*` / `DEFAULT_LICENSE_SPDX` constants mirror the two
//! converters' own constants. `vokra-models` deliberately does **not**
//! depend on `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`. The
//! converter owns the writer contract; this module owns the reader
//! contract, and the two copies are pinned against each other by tests on
//! both sides.
//!
//! Note that the arch tags use `_` while the model names use `-`, and that
//! the v3 arch (`sber_gigaam_v3`) carries the vendor prefix while the
//! multilingual arch (`gigaam_multilingual`) does not. Both asymmetries are
//! load-bearing on the wire and are pinned separately by tests.
//!
//! # Licensing posture
//!
//! Both converters stamp `mit` → [`LicenseClass::Permissive`] on the basis
//! of the upstream GitHub `LICENSE`. This binder only **surfaces** whatever
//! class the GGUF carries and fail-closes to [`LicenseClass::Unknown`] when
//! nothing is stamped. The `docs/license-audit.md` §3.1 sign-off row stays
//! **BLANK** — owner-only per `[[feedback-license-signoff-primary-source]]`;
//! CC does not sign, and does not treat a converter default as a sign-off.
//! Both Wave B tickets additionally flag open corpus-provenance questions
//! (Sber-internal corpus disclosure for v3; a Common Voice / MLS /
//! VoxPopuli / FLEURS rights chain for the 70+-language multilingual
//! variant) that are owner audit items, not runtime concerns.
//!
//! # No ONNX / no pickle (permanent)
//!
//! The upstream release ships torch pickles; the offline sidecars
//! [`SIDECAR_PATH_V3`] / [`SIDECAR_PATH_MULTILINGUAL`] bridge them to
//! safetensors outside the runtime. This module never touches ONNX or
//! pickle (FR-LD-05 / NFR-DS-02).

use std::collections::{BTreeMap, BTreeSet};

use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirrors of the two converters' constants.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// `vokra.model.arch` written by `vokra-cli convert --model sber-gigaam-v3`.
///
/// Mirror of `vokra-convert::models::sber_gigaam_v3::ARCH`. Carries the
/// vendor prefix (`sber_`), unlike its multilingual sibling — an asymmetry
/// that is load-bearing on the wire and pinned by a test.
pub const ARCH_V3: &str = "sber_gigaam_v3";

/// `vokra.model.arch` written by
/// `vokra-cli convert --model sber-gigaam-multilingual`.
///
/// Mirror of `vokra-convert::models::sber_gigaam_multilingual::ARCH`. Note
/// it does **not** carry the `sber_` vendor prefix its v3 sibling uses.
pub const ARCH_MULTILINGUAL: &str = "gigaam_multilingual";

/// Every `vokra.model.arch` value this binder serves.
///
/// Both members are the same Conformer + char-wise-CTC family and differ
/// only in vocabulary breadth, so one reader serves both — the
/// `whisper` binder's `ACCEPTED_ARCHS` family precedent. Anything else is
/// refused loudly: the binder would otherwise bind whatever verbatim
/// upstream tensor names happen to overlap and transcribe noise (FR-EX-08).
pub const ACCEPTED_ARCHS: &[&str] = &[ARCH_V3, ARCH_MULTILINGUAL];

/// `vokra.model.name` the v3 converter writes.
pub const NAME_V3: &str = "gigaam-v3";

/// `vokra.model.name` the multilingual converter writes.
pub const NAME_MULTILINGUAL: &str = "sber-gigaam-multilingual";

/// `vokra.model.category` both converters write.
///
/// Both tickets label the models more narrowly (`asr/russian`,
/// `asr/70+lang`) but both converters deliberately stamp the first-word
/// `asr` so runtime dispatch and model-card grouping do not multiply
/// category labels by distinctions the arch tag already carries.
pub const CATEGORY: &str = "asr";

/// `vokra.provenance.upstream_hf` the **v3** converter writes — the
/// `ai-sage` collection is the canonical HF redistribution surface for the
/// Russian-specific release.
pub const UPSTREAM_HF_V3: &str = "ai-sage/GigaAM-v3";

/// `vokra.provenance.upstream_url` the **multilingual** converter writes.
///
/// The GitHub release is the primary redistribution source because the
/// Wave B ticket flags the HF mirror `ai-sage/GigaAM-Multilingual` as
/// 要 mirror URL 確認 (`proprietary-mirror-risk: medium` — Sber has
/// maintained HF mirrors independently of the primary in the past).
pub const UPSTREAM_URL_MULTILINGUAL: &str = "github.com/salute-developers/GigaAM";

/// Default upstream weight licence SPDX id both converters stamp, per the
/// upstream `github.com/salute-developers/GigaAM/LICENSE` (standard MIT →
/// [`LicenseClass::Permissive`]).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Primary-source anchor for the upstream repository (both variants).
pub const PRIMARY_SOURCE_REPO: &str = "github.com/salute-developers/GigaAM";

/// Primary-source anchor for the v3 HF surface.
pub const PRIMARY_SOURCE_HF_V3: &str = "huggingface.co/ai-sage/GigaAM-v3";

/// Converter that produces [`ARCH_V3`] GGUFs.
pub const CONVERTER_PATH_V3: &str = "crates/vokra-convert/src/models/sber_gigaam_v3.rs";

/// Converter that produces [`ARCH_MULTILINGUAL`] GGUFs.
pub const CONVERTER_PATH_MULTILINGUAL: &str =
    "crates/vokra-convert/src/models/sber_gigaam_multilingual.rs";

/// Offline pickle → safetensors sidecar for the v3 release.
pub const SIDECAR_PATH_V3: &str = "tools/parity/sber_gigaam_v3_prepare_checkpoint.py";

/// Offline pickle → safetensors sidecar for the multilingual release.
pub const SIDECAR_PATH_MULTILINGUAL: &str =
    "tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py";

/// `vokra.model.category` metadata key.
///
/// Local mirror per the established converter-side convention — not yet
/// centralised in `vokra_core::gguf::chunks`.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` — the key the **v3** converter stamps.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// `vokra.provenance.upstream_url` — the key the **multilingual** converter
/// stamps (its canonical release is not on the HF hub).
pub const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Optional `Array<String>` declaration naming the tensors the producer
/// asserts it wrote.
///
/// **Neither converter stamps this today.** When present it upgrades a
/// truncated / mis-merged / partially-uploaded GGUF from "surprises a
/// forward halfway through" to "fails at load naming the first missing
/// tensor" — see [`GigaamWeights::require_all`]. Absent is fine and is the
/// normal case; a *present but empty* list is always a producer bug and is
/// refused.
pub const KEY_REQUIRED_TENSORS: &str = "vokra.gigaam.required_tensors";

/// The infix that marks a homogeneous layer stack in an upstream
/// state-dict key, e.g. the `.layers.` in
/// `encoder.layers.7.self_attn.linear_q.weight`.
///
/// [`GigaamTopology`] discovers stacks by scanning for this infix rather
/// than by asserting any particular upstream prefix, so the probe works
/// without a verified upstream tensor-name manifest.
pub const LAYER_STACK_INFIX: &str = ".layers.";

// ---------------------------------------------------------------------------
// GigaamVariant
// ---------------------------------------------------------------------------

/// Which half of the GigaAM family a bound GGUF is.
///
/// The two share the Conformer + char-wise-CTC topology and differ in the
/// **vocabulary** the CTC head emits over — which is precisely why the
/// converters keep distinct arch tags rather than aliasing: a runtime that
/// decoded a 70+-language checkpoint through a Russian-only vocabulary
/// would emit shape-valid nonsense rather than fail (FR-EX-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GigaamVariant {
    /// `ai-sage/GigaAM-v3` — the Russian / Central-Asian fine-tune.
    /// Arch [`ARCH_V3`], name [`NAME_V3`].
    V3,
    /// The 2026 `salute-developers/GigaAM` multilingual variant covering
    /// 70+ languages through a widened char label space. Arch
    /// [`ARCH_MULTILINGUAL`], name [`NAME_MULTILINGUAL`].
    Multilingual,
}

impl GigaamVariant {
    /// Resolves a `vokra.model.arch` string to a variant.
    ///
    /// Returns `None` for anything outside [`ACCEPTED_ARCHS`]; callers that
    /// want the loud diagnostic use [`Gigaam::from_gguf`] instead.
    #[must_use]
    pub fn from_arch(arch: &str) -> Option<Self> {
        if arch == ARCH_V3 {
            Some(Self::V3)
        } else if arch == ARCH_MULTILINGUAL {
            Some(Self::Multilingual)
        } else {
            None
        }
    }

    /// The `vokra.model.arch` string this variant is stamped with.
    #[inline]
    #[must_use]
    pub const fn arch(self) -> &'static str {
        match self {
            Self::V3 => ARCH_V3,
            Self::Multilingual => ARCH_MULTILINGUAL,
        }
    }

    /// The `vokra.model.name` string the producing converter writes.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::V3 => NAME_V3,
            Self::Multilingual => NAME_MULTILINGUAL,
        }
    }

    /// The `vokra.provenance.*` key this variant's converter stamps its
    /// upstream location under.
    ///
    /// v3 uses [`KEY_PROVENANCE_UPSTREAM_HF`] (the `ai-sage` collection is
    /// its canonical redistribution surface); multilingual uses
    /// [`KEY_PROVENANCE_UPSTREAM_URL`] (its HF mirror is unverified, so the
    /// GitHub release is primary).
    #[inline]
    #[must_use]
    pub const fn upstream_key(self) -> &'static str {
        match self {
            Self::V3 => KEY_PROVENANCE_UPSTREAM_HF,
            Self::Multilingual => KEY_PROVENANCE_UPSTREAM_URL,
        }
    }

    /// The upstream location value this variant's converter stamps.
    #[inline]
    #[must_use]
    pub const fn upstream_value(self) -> &'static str {
        match self {
            Self::V3 => UPSTREAM_HF_V3,
            Self::Multilingual => UPSTREAM_URL_MULTILINGUAL,
        }
    }

    /// The `vokra-cli convert --model <arg>` spelling that produces this
    /// variant's GGUF — echoed in load-failure diagnostics so a reader has
    /// a runnable repro.
    #[inline]
    #[must_use]
    pub const fn converter_arg(self) -> &'static str {
        match self {
            Self::V3 => "sber-gigaam-v3",
            Self::Multilingual => "sber-gigaam-multilingual",
        }
    }

    /// Path to the converter that writes this variant.
    #[inline]
    #[must_use]
    pub const fn converter_path(self) -> &'static str {
        match self {
            Self::V3 => CONVERTER_PATH_V3,
            Self::Multilingual => CONVERTER_PATH_MULTILINGUAL,
        }
    }

    /// Path to the offline pickle → safetensors sidecar for this variant.
    #[inline]
    #[must_use]
    pub const fn sidecar_path(self) -> &'static str {
        match self {
            Self::V3 => SIDECAR_PATH_V3,
            Self::Multilingual => SIDECAR_PATH_MULTILINGUAL,
        }
    }

    /// Human-readable description of the vocabulary this variant's CTC head
    /// emits over. Used in the loud-partial message to make the
    /// consequence of a mis-routed load concrete.
    #[inline]
    #[must_use]
    pub const fn vocabulary_scope(self) -> &'static str {
        match self {
            Self::V3 => "Russian / Central-Asian char space",
            Self::Multilingual => "70+-language char space",
        }
    }

    /// The other half of the family — used by the crossed-wires check in
    /// [`Gigaam::from_gguf`].
    #[inline]
    #[must_use]
    pub const fn sibling(self) -> Self {
        match self {
            Self::V3 => Self::Multilingual,
            Self::Multilingual => Self::V3,
        }
    }
}

// ---------------------------------------------------------------------------
// Arch verification
// ---------------------------------------------------------------------------

/// Sibling `category = "asr"` arch tags enumerated in the wrong-arch
/// diagnostic.
///
/// Every one of these shares `vokra.model.category = "asr"` with GigaAM, so
/// the category tag alone cannot disambiguate them — only the arch tag can.
/// Naming them in the error turns "this GGUF did not load" into "you handed
/// the GigaAM binder a Parakeet checkpoint".
const ASR_SIBLING_ARCHS: &[&str] = &[
    "parakeet-ctc",
    "canary",
    "omniasr-ctc",
    "whisper",
    "distil-whisper",
    "kotoba-whisper",
    "sensevoicesmall",
];

/// Resolves and validates the `vokra.model.arch` chunk of `file`.
///
/// A *loud* validation step (FR-EX-08): both GigaAM converters copy every
/// tensor under its verbatim upstream state-dict name, so a foreign
/// checkpoint that happens to share some of those names would bind a
/// partial model and — once the forward lands — transcribe noise rather
/// than crash.
///
/// # Errors
///
/// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
/// - [`VokraError::ModelLoad`] when it is not in [`ACCEPTED_ARCHS`], naming
///   both the observed tag and the expected set.
pub fn verify_arch(file: &GgufFile) -> Result<GigaamVariant> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) => GigaamVariant::from_arch(a).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "gigaam: GGUF arch is `{a}`, expected one of {ACCEPTED_ARCHS:?} \
                 (`{ARCH_V3}` = the Russian / Central-Asian fine-tune produced by \
                 `vokra-cli convert --model sber-gigaam-v3`; `{ARCH_MULTILINGUAL}` = \
                 the 70+-language variant produced by \
                 `vokra-cli convert --model sber-gigaam-multilingual`). Those two \
                 share the Conformer + char-wise-CTC topology and are the only \
                 arches this binder serves. Sibling `{CATEGORY}`-category arch tags \
                 — {ASR_SIBLING_ARCHS:?} — carry the same `vokra.model.category` \
                 value but completely different encoder geometries and head \
                 contracts, so the category tag alone cannot disambiguate them; only \
                 the arch tag can. Binding one here would match whatever verbatim \
                 upstream tensor names happen to overlap and transcribe noise \
                 (FR-EX-08 — no silent partial load)."
            ))
        }),
        None => Err(VokraError::ModelLoad(format!(
            "gigaam: GGUF is missing `{key}` — this is not a Vokra-native GigaAM \
             GGUF (was it produced by `vokra-cli convert --model sber-gigaam-v3` or \
             `vokra-cli convert --model sber-gigaam-multilingual`?). Both converters \
             stamp the arch chunk unconditionally, so its absence means the file \
             came from somewhere else entirely (FR-EX-08 — no silent partial load).",
            key = chunks::KEY_MODEL_ARCH,
        ))),
    }
}

// ---------------------------------------------------------------------------
// Optional required-tensor declaration
// ---------------------------------------------------------------------------

/// Reads the optional [`KEY_REQUIRED_TENSORS`] declaration.
///
/// Returns `Ok(None)` when the key is absent — the normal case, since
/// neither converter stamps it today. Refuses a wrong container type, a
/// wrong element type, a non-string element, or an empty list: an empty
/// declaration asserts nothing, so stamping one is always a producer bug.
///
/// # Errors
///
/// - [`VokraError::ModelLoad`] on any of the malformed shapes above.
fn read_required_tensors(gguf: &GgufFile) -> Result<Option<Vec<String>>> {
    let Some(value) = gguf.get(KEY_REQUIRED_TENSORS) else {
        return Ok(None);
    };
    let arr = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "gigaam: GGUF metadata `{KEY_REQUIRED_TENSORS}` is not an array \
             (expected Array<String> naming the tensors the producer wrote), got \
             {:?}",
            value.value_type()
        ))
    })?;
    if arr.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "gigaam: GGUF metadata `{KEY_REQUIRED_TENSORS}` has element_type {:?}, \
             expected String",
            arr.element_type
        )));
    }
    let mut out = Vec::with_capacity(arr.values.len());
    for (i, v) in arr.values.iter().enumerate() {
        match v {
            GgufMetadataValue::String(s) => out.push(s.clone()),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "gigaam: GGUF metadata `{KEY_REQUIRED_TENSORS}[{i}]` is not a \
                     string (got {:?})",
                    other.value_type()
                )));
            }
        }
    }
    if out.is_empty() {
        return Err(VokraError::ModelLoad(format!(
            "gigaam: GGUF metadata `{KEY_REQUIRED_TENSORS}` is an empty list — an \
             empty required-tensor declaration asserts nothing, so stamping it is \
             always a producer bug. Omit the key entirely, or list the tensor names \
             the converter actually wrote (FR-EX-08)."
        )));
    }
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// GigaamWeights — the bound tensor manifest.
// ---------------------------------------------------------------------------

/// Tensor manifest bound from a GigaAM GGUF.
///
/// Both converters emit every F32 / F16 / BF16 tensor under its **verbatim
/// upstream state-dict key**, so the names here are the upstream names.
/// This struct holds those names with their GGUF-side dims: enough for the
/// structural [`GigaamTopology`] probe and for the follow-up forward wave
/// to walk without re-parsing the GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF carrying zero tensors is refused rather than silently
/// binding an all-zero forward (FR-EX-08).
#[derive(Debug, Clone)]
pub struct GigaamWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl GigaamWeights {
    /// Scans `gguf` for its tensor manifest.
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
                "gigaam: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate GigaAM checkpoint carries the \
                 front-end stem plus every Conformer block's attention / \
                 feed-forward / convolution / normalisation parameters and the \
                 char-wise CTC head (arch is one of {ACCEPTED_ARCHS:?}); zero \
                 tensors always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model sber-gigaam-v3` (or \
                 `--model sber-gigaam-multilingual`) against an upstream \
                 `{PRIMARY_SOURCE_REPO}` safetensors checkpoint."
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

    /// The bound tensor names with their GGUF-side dims.
    #[inline]
    #[must_use]
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        &self.tensors
    }

    /// `true` when `name` is present in the bound manifest.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.tensors.iter().any(|(n, _)| n.as_str() == name)
    }

    /// Looks a tensor's GGUF-side dims up by name.
    ///
    /// Returns a loud [`VokraError::ModelLoad`] **naming the absent
    /// tensor** rather than `None`, so a caller cannot swallow the miss
    /// with `unwrap_or_default()` and proceed on an implicit zero shape
    /// (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `name` is not in the manifest.
    pub fn dims(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .map(|(_, d)| d.as_slice())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "gigaam: tensor `{name}` is absent from the GGUF manifest \
                     ({count} tensors present). GigaAM GGUFs carry the upstream \
                     safetensors names verbatim (see `{CONVERTER_PATH_V3}` / \
                     `{CONVERTER_PATH_MULTILINGUAL}`), so a miss means either a \
                     mis-produced GGUF or a stale name in the caller (FR-EX-08 — no \
                     silent zero-shape fallback).",
                    count = self.tensors.len()
                ))
            })
    }

    /// Verifies every name in a [`KEY_REQUIRED_TENSORS`] declaration is
    /// present, failing loud on the **first** absent one.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the first missing tensor.
    pub fn require_all(&self, names: &[String]) -> Result<()> {
        for name in names {
            if !self.has(name) {
                return Err(VokraError::ModelLoad(format!(
                    "gigaam: required tensor `{name}` is declared in \
                     `{KEY_REQUIRED_TENSORS}` but absent from the GGUF manifest \
                     ({count} tensors present, {declared} declared). The producer \
                     asserted it wrote this tensor, so the GGUF is truncated, \
                     mis-merged or partially uploaded — refusing at load time rather \
                     than surprising a forward halfway through (FR-EX-08).",
                    count = self.tensors.len(),
                    declared = names.len(),
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GigaamTopology — a measured structural probe (nothing here is guessed).
// ---------------------------------------------------------------------------

/// One homogeneous layer stack discovered in a GigaAM checkpoint.
///
/// A "stack" is any family of tensors named
/// `<root><index>.<leaf>` where `<root>` ends in [`LAYER_STACK_INFIX`] —
/// e.g. root `encoder.layers.`, index `7`, leaf
/// `self_attn.linear_q.weight`.
///
/// Discovery is purely name-driven, so it needs **no verified upstream
/// tensor-name manifest**: whatever the checkpoint actually contains is
/// what gets measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GigaamLayerStack {
    root: String,
    n_layer: usize,
    leaf_suffixes: Vec<String>,
}

impl GigaamLayerStack {
    /// The stack root, including the trailing [`LAYER_STACK_INFIX`] —
    /// e.g. `"encoder.layers."`.
    #[inline]
    #[must_use]
    pub fn root(&self) -> &str {
        &self.root
    }

    /// Depth of the stack: indices `0..n_layer` are all present (enforced
    /// at probe time).
    #[inline]
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// The leaf tensor suffixes every layer in this stack carries, sorted.
    ///
    /// Uniformity across all layers is enforced at probe time, so this set
    /// describes every layer, not merely layer 0.
    #[inline]
    #[must_use]
    pub fn leaf_suffixes(&self) -> &[String] {
        &self.leaf_suffixes
    }

    /// Number of tensors per layer in this stack.
    #[inline]
    #[must_use]
    pub fn tensors_per_layer(&self) -> usize {
        self.leaf_suffixes.len()
    }

    /// Reconstructs the full tensor name for `layer` / `leaf`.
    ///
    /// Convenience for the follow-up forward wave, which walks a stack by
    /// index rather than by string surgery.
    #[must_use]
    pub fn tensor_name(&self, layer: usize, leaf: &str) -> String {
        format!("{}{}.{}", self.root, layer, leaf)
    }
}

/// The measured structure of a bound GigaAM checkpoint.
///
/// This is the honest replacement for a fabricated `GigaamConfig`: rather
/// than asserting `n_layer = 16` / `d_model = 768` from nowhere, the probe
/// **counts** what the checkpoint contains and enforces the two invariants
/// a homogeneous transformer stack must satisfy.
///
/// # What is enforced
///
/// 1. **Contiguity** — a stack whose indices are `{0, 1, 3}` has a hole at
///    2 and is refused, naming the absent index.
/// 2. **Uniformity** — every layer must carry the same leaf-tensor set as
///    layer 0. A violation is refused naming the **exact** absent tensor
///    (e.g. `encoder.layers.7.conv_module.depthwise_conv.weight`).
///
/// Both are load-time gates rather than warnings, on the FR-EX-08 posture
/// the rest of this crate takes: a truncated stack that binds successfully
/// would surprise a forward halfway through, which is strictly worse than
/// a loud refusal that names the tensor. If a legitimate upstream release
/// ever ships a genuinely non-uniform stack, the correct response is to
/// extend this probe to model that asymmetry — not to relax the gate.
///
/// # What is *not* asserted
///
/// Nothing about `d_model`, `n_mels`, head count, sample rate or vocabulary
/// size. None of those are recoverable from a name-only manifest without a
/// verified upstream mapping, and guessing them is what
/// [`Gigaam::transcribe`] loud-partials over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GigaamTopology {
    stacks: Vec<GigaamLayerStack>,
    non_stack_tensors: usize,
}

impl GigaamTopology {
    /// Probes `weights` for its layer stacks.
    ///
    /// Stacks are returned sorted by root, so the result is deterministic
    /// regardless of GGUF tensor ordering.
    ///
    /// An empty stack list is **not** an error: a checkpoint that does not
    /// use `.layers.` naming is unusual but not malformed, and refusing it
    /// would re-open the very "converted weights are unloadable" gap this
    /// module closes.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when a stack has an index hole, naming
    ///   the absent index.
    /// - [`VokraError::ModelLoad`] when a stack is non-uniform, naming the
    ///   exact absent tensor.
    pub fn probe(weights: &GigaamWeights) -> Result<Self> {
        // root -> index -> {leaf suffix}. BTree* throughout so iteration
        // order (and therefore every diagnostic) is deterministic.
        let mut found: BTreeMap<String, BTreeMap<usize, BTreeSet<String>>> = BTreeMap::new();
        let mut non_stack_tensors = 0usize;

        for (name, _dims) in weights.tensors() {
            match split_stack_name(name) {
                Some((root, index, leaf)) => {
                    found
                        .entry(root.to_owned())
                        .or_default()
                        .entry(index)
                        .or_default()
                        .insert(leaf.to_owned());
                }
                None => non_stack_tensors += 1,
            }
        }

        let mut stacks = Vec::with_capacity(found.len());
        for (root, layers) in found {
            // `layers` is non-empty by construction (an entry is only
            // created when a tensor lands in it).
            let max_index = layers
                .keys()
                .next_back()
                .copied()
                .expect("stack entry is created only when a tensor lands in it");
            let n_layer = max_index + 1;

            // 1. Contiguity.
            for i in 0..n_layer {
                if !layers.contains_key(&i) {
                    return Err(VokraError::ModelLoad(format!(
                        "gigaam: layer stack `{root}` has a hole — index {i} carries \
                         no tensors, but index {max_index} does (so the stack is \
                         {n_layer} layers deep). A homogeneous Conformer stack is \
                         contiguous by construction, so a hole means the GGUF is \
                         truncated, mis-merged or partially uploaded. Refusing at \
                         load time rather than surprising a forward halfway through \
                         (FR-EX-08 — no silent partial load)."
                    )));
                }
            }

            // 2. Uniformity against layer 0 (present after the contiguity
            //    check above).
            let reference = layers
                .get(&0)
                .expect("contiguity check guarantees index 0 is present");
            for (&i, leaves) in &layers {
                if i == 0 {
                    continue;
                }
                // Absent from layer i but present in layer 0.
                if let Some(missing) = reference.difference(leaves).next() {
                    return Err(VokraError::ModelLoad(format!(
                        "gigaam: layer stack `{root}` is non-uniform — tensor \
                         `{root}{i}.{missing}` is absent, but layer 0 carries \
                         `{root}0.{missing}`. Every block of a homogeneous Conformer \
                         stack carries the same parameter set, so a gap means the \
                         GGUF is truncated, mis-merged or partially uploaded. \
                         Refusing at load time rather than surprising a forward \
                         halfway through (FR-EX-08 — no silent partial load)."
                    )));
                }
                // Present in layer i but absent from layer 0 — the same
                // defect seen from the other side; report it against
                // layer 0 so the named tensor is the one that is missing.
                if let Some(extra) = leaves.difference(reference).next() {
                    return Err(VokraError::ModelLoad(format!(
                        "gigaam: layer stack `{root}` is non-uniform — tensor \
                         `{root}0.{extra}` is absent, but layer {i} carries \
                         `{root}{i}.{extra}`. Every block of a homogeneous Conformer \
                         stack carries the same parameter set, so a gap means the \
                         GGUF is truncated, mis-merged or partially uploaded. \
                         Refusing at load time rather than surprising a forward \
                         halfway through (FR-EX-08 — no silent partial load)."
                    )));
                }
            }

            stacks.push(GigaamLayerStack {
                root,
                n_layer,
                leaf_suffixes: reference.iter().cloned().collect(),
            });
        }

        Ok(Self {
            stacks,
            non_stack_tensors,
        })
    }

    /// Every layer stack discovered, sorted by root.
    ///
    /// May be empty — see [`probe`](Self::probe).
    #[inline]
    #[must_use]
    pub fn stacks(&self) -> &[GigaamLayerStack] {
        &self.stacks
    }

    /// Looks a stack up by its root (including the trailing
    /// [`LAYER_STACK_INFIX`]).
    #[must_use]
    pub fn stack(&self, root: &str) -> Option<&GigaamLayerStack> {
        self.stacks.iter().find(|s| s.root == root)
    }

    /// Count of bound tensors that are **not** part of any layer stack —
    /// the front-end stem, the CTC head, top-level norms and so on.
    #[inline]
    #[must_use]
    pub fn non_stack_tensors(&self) -> usize {
        self.non_stack_tensors
    }
}

/// Splits `name` into `(root_including_infix, index, leaf_suffix)` when it
/// looks like a layer-stack entry.
///
/// Returns `None` when the name has no [`LAYER_STACK_INFIX`], when the
/// segment after the infix is not a decimal index, or when nothing follows
/// the index (a bare `encoder.layers.0` with no leaf is not a stack entry).
fn split_stack_name(name: &str) -> Option<(&str, usize, &str)> {
    let infix_at = name.find(LAYER_STACK_INFIX)?;
    let root_end = infix_at + LAYER_STACK_INFIX.len();
    let root = &name[..root_end];
    let rest = &name[root_end..];
    let dot_at = rest.find('.')?;
    let index: usize = rest[..dot_at].parse().ok()?;
    let leaf = &rest[dot_at + 1..];
    if leaf.is_empty() {
        return None;
    }
    Some((root, index, leaf))
}

// ---------------------------------------------------------------------------
// Gigaam — the runtime binder handle.
// ---------------------------------------------------------------------------

/// Sber GigaAM runtime binder — the consumer of the `sber_gigaam_v3` and
/// `gigaam_multilingual` converter arches.
///
/// Bind with [`from_gguf`](Self::from_gguf) or
/// [`from_path`](Self::from_path), inspect the measured structure through
/// [`topology`](Self::topology), and call [`transcribe`](Self::transcribe)
/// on a mono f32 PCM waveform.
///
/// [`transcribe`](Self::transcribe) is a loud-partial today; see
/// [`transcribe_loud_partial`] for exactly which three pieces are missing
/// and why guessing them would be silent-wrong.
#[derive(Debug, Clone)]
pub struct Gigaam {
    variant: GigaamVariant,
    weights: GigaamWeights,
    topology: GigaamTopology,
    weight_license: LicenseClass,
    license_spdx: Option<String>,
    model_name: Option<String>,
    category: Option<String>,
    upstream: Option<String>,
}

impl Gigaam {
    /// Binds a GigaAM GGUF.
    ///
    /// Steps, in order, each a loud gate:
    ///
    /// 1. [`verify_arch`] — resolves the [`GigaamVariant`] or refuses.
    /// 2. Crossed-wires check — an arch/name pair naming *different*
    ///    variants is refused (they disagree about which vocabulary the
    ///    CTC head carries).
    /// 3. [`GigaamWeights::from_gguf`] — non-empty tensor gate.
    /// 4. [`KEY_REQUIRED_TENSORS`] — honoured when stamped.
    /// 5. [`GigaamTopology::probe`] — contiguity + uniformity gates.
    /// 6. Provenance surfacing — weight-license class fail-closes to
    ///    [`LicenseClass::Unknown`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the arch chunk is absent or is not
    ///   in [`ACCEPTED_ARCHS`].
    /// - [`VokraError::ModelLoad`] when arch and `vokra.model.name` name
    ///   different variants.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    /// - [`VokraError::ModelLoad`] when a [`KEY_REQUIRED_TENSORS`]
    ///   declaration is malformed or names an absent tensor.
    /// - [`VokraError::ModelLoad`] when a layer stack has an index hole or
    ///   is non-uniform.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch first, so a mis-typed model fails with a specific
        //    message rather than a downstream missing-tensor error.
        let variant = verify_arch(file)?;

        // 2. Crossed wires: arch says one variant, name says the other.
        let model_name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        if let Some(name) = model_name.as_deref() {
            let sibling = variant.sibling();
            if name == sibling.name() {
                return Err(VokraError::ModelLoad(format!(
                    "gigaam: GGUF arch is `{arch}` (the {vocab}) but \
                     `{name_key}` is `{name}`, which is the sibling variant \
                     `{sibling_arch}` (the {sibling_vocab}). The two chunks \
                     disagree about which vocabulary the char-wise CTC head emits \
                     over, so one of them is wrong and this file was assembled by \
                     crossing two conversions. Decoding a checkpoint through the \
                     wrong variant's vocabulary produces shape-valid nonsense rather \
                     than an error, so this is refused at load time (FR-EX-08).",
                    arch = variant.arch(),
                    vocab = variant.vocabulary_scope(),
                    name_key = chunks::KEY_MODEL_NAME,
                    sibling_arch = sibling.arch(),
                    sibling_vocab = sibling.vocabulary_scope(),
                )));
            }
        }

        // 3. Tensor manifest with the non-emptiness gate.
        let weights = GigaamWeights::from_gguf(file)?;

        // 4. Optional producer-declared required-tensor manifest.
        if let Some(required) = read_required_tensors(file)? {
            weights.require_all(&required)?;
        }

        // 5. Measured structure (contiguity + uniformity gates inside).
        let topology = GigaamTopology::probe(&weights)?;

        // 6. Provenance surfacing. A GGUF missing the stamp reads back as
        //    `Unknown` (fail-closed per
        //    `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let license_spdx = file
            .get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let category = file
            .get(KEY_MODEL_CATEGORY)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        // Each variant's converter stamps its upstream under a different
        // key (HF slug vs raw URL); read the variant's own key.
        let upstream = file
            .get(variant.upstream_key())
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        Ok(Self {
            variant,
            weights,
            topology,
            weight_license,
            license_spdx,
            model_name,
            category,
            upstream,
        })
    }

    /// Binds a GigaAM GGUF from a filesystem path.
    ///
    /// # Errors
    ///
    /// - Any [`GgufFile::open`] failure, plus everything
    ///   [`from_gguf`](Self::from_gguf) can return.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Which half of the GigaAM family this checkpoint is.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> GigaamVariant {
        self.variant
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &GigaamWeights {
        &self.weights
    }

    /// The measured layer-stack structure of this checkpoint.
    #[inline]
    #[must_use]
    pub const fn topology(&self) -> &GigaamTopology {
        &self.topology
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The stamped weight-license class.
    ///
    /// Both converters stamp `mit` → [`LicenseClass::Permissive`]. A GGUF
    /// missing the stamp reads back as [`LicenseClass::Unknown`]
    /// (fail-closed at the M2-13 compliance gate). This binder only
    /// **surfaces** the class; the `docs/license-audit.md` §3.1 sign-off is
    /// owner-only.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The raw SPDX id stamped under `vokra.provenance.license`, when
    /// present (`"mit"` for a default conversion).
    #[inline]
    #[must_use]
    pub fn license_spdx(&self) -> Option<&str> {
        self.license_spdx.as_deref()
    }

    /// The `vokra.model.name` chunk, when present.
    #[inline]
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// The `vokra.model.category` chunk, when present ([`CATEGORY`] for a
    /// default conversion).
    #[inline]
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// The upstream location stamped under this variant's provenance key —
    /// an HF slug for [`GigaamVariant::V3`], a raw URL for
    /// [`GigaamVariant::Multilingual`].
    #[inline]
    #[must_use]
    pub fn upstream(&self) -> Option<&str> {
        self.upstream.as_deref()
    }

    /// Transcribes a mono f32 PCM waveform.
    ///
    /// # Loud-partial (this wave)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — see
    /// [`transcribe_loud_partial`] for the three blockers and why a
    /// best-guess implementation would be silent-wrong rather than merely
    /// incomplete. **No fabricated transcript is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `pcm` is empty — an empty
    ///   waveform is a caller bug, and returning `Ok(String::new())` for it
    ///   would be indistinguishable from a successful silent transcription.
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "gigaam transcribe: empty PCM buffer. An empty waveform cannot be \
                 transcribed, and returning an empty transcript would be \
                 indistinguishable from a successful recognition of silence \
                 (FR-EX-08)."
                    .to_owned(),
            ));
        }
        Err(transcribe_loud_partial(self.variant))
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Gigaam::transcribe`].
///
/// The message names **three concrete blockers**, all of which are
/// properties of the GGUF contract rather than of the kernel library:
///
/// 1. **The missing front-end spec.** Neither converter stamps a
///    `vokra.gigaam.*` or `vokra.frontend.*` chunk — both write only
///    `vokra.model.{arch,name,category}` plus the `vokra.provenance.*`
///    group. So the sample rate, mel-bin count, hop, window and
///    normalisation convention are all unknown, and Vokra guarantees
///    bit-exact front ends precisely because these differ silently between
///    librosa / torchaudio / Kaldi conventions.
/// 2. **The missing encoder tensor-name mapping.** Both converters pass
///    every tensor through under its verbatim upstream state-dict key and
///    both explicitly record real-weight binding as a follow-up "gated on
///    the upstream tensor-name manifest fetch". Nothing in-repo transcribes
///    which upstream name is the depthwise convolution, which is the
///    macaron feed-forward, or how QKV is packed. A best-guess mapping
///    yields a **shape-valid but quietly wrong** transcript — the worst
///    failure mode available.
/// 3. **The missing CTC vocabulary.** Neither converter embeds a
///    tokenizer / vocab chunk (contrast the Whisper converter, which embeds
///    `vokra.tokenizer.model` as a raw U8 array). GigaAM is char-wise CTC,
///    so without the char table the frame-argmax indices cannot be mapped
///    to Russian Cyrillic (v3) or 70+-language (multilingual) characters at
///    all.
///
/// The message deliberately also states what is **not** missing: the
/// composition needs no new primitive. `vokra_ops::conformer` already
/// carries the encoder body (it backs the Parakeet-CTC and Canary
/// binders), `vokra_ops::ctc_decode_greedy` / `vokra_ops::ctc_decode_beam`
/// already carry the blank-fold and prefix-beam decoders, and
/// `vokra_ops::mel` / `vokra_ops::kaldi_fbank` already carry the front
/// ends. The fix is to extend the converters to stamp the axes and the
/// vocabulary — not to write a kernel.
///
/// The cited "primary sources" are the upstream repository plus **this
/// variant's own** stamped upstream location, deliberately rather than a
/// fixed HF slug: the multilingual half's HF mirror is flagged
/// 要 mirror URL 確認 in its Wave B ticket, so pointing a reader at it as
/// though it were authoritative would be wrong.
#[must_use]
pub fn transcribe_loud_partial(variant: GigaamVariant) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "gigaam transcribe ({arch}, loud-partial): the full forward is deferred; \
         three pieces must land before a real transcript can be emitted. \
         (1) MISSING FRONT-END SPEC — `{converter}` stamps only \
         `vokra.model.{{arch,name,category}}` plus the `vokra.provenance.*` group, \
         so there is no `vokra.gigaam.*` / `vokra.frontend.*` chunk carrying the \
         sample rate, mel-bin count, hop, window or normalisation convention, and \
         those differ silently between librosa / torchaudio / Kaldi conventions. \
         (2) MISSING ENCODER TENSOR-NAME MAPPING — the converter copies every \
         tensor under its verbatim upstream state-dict key and records real-weight \
         binding as a follow-up gated on the upstream tensor-name manifest, so \
         nothing in-repo says which upstream name is the depthwise convolution, \
         which is the macaron feed-forward, or how QKV is packed; a best-guess \
         mapping would emit a SHAPE-VALID but quietly wrong transcript rather than \
         crash. (3) MISSING CTC VOCABULARY — no tokenizer / vocab chunk is embedded \
         (contrast the Whisper converter's `vokra.tokenizer.model` U8 array), and \
         GigaAM is char-wise CTC, so frame-argmax indices cannot be mapped to the \
         {vocab} at all. NOTE the blockers are METADATA, not kernels: \
         `vokra_ops::conformer` (the encoder body shared with the Parakeet-CTC / \
         Canary binders), `vokra_ops::ctc_decode_greedy` / \
         `vokra_ops::ctc_decode_beam`, and `vokra_ops::mel` / \
         `vokra_ops::kaldi_fbank` all already exist — the fix is to extend \
         `{converter}` (and the offline sidecar `{sidecar}`) to stamp the axes and \
         the vocabulary. Primary sources: upstream repository {repo}, plus this \
         variant's own stamped upstream location (`{ukey}` = `{uval}`). Runtime \
         cannot fabricate a transcript (FR-EX-08 — no silent partial output).",
        arch = variant.arch(),
        converter = variant.converter_path(),
        sidecar = variant.sidecar_path(),
        vocab = variant.vocabulary_scope(),
        repo = PRIMARY_SOURCE_REPO,
        ukey = variant.upstream_key(),
        uval = variant.upstream_value(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the GigaAM family binder.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real 16 kHz waveform this would be `transcribe(...)` returning
    //! Russian (or multilingual) text, but the front-end spec, the encoder
    //! tensor-name mapping and the CTC vocabulary are all absent from the
    //! GGUF contract (see [`transcribe_loud_partial`]). Fabricating a
    //! transcript would violate CLAUDE.md 教訓 (a)
    //! 「loud-partial は fake-complete より honest」.
    //!
    //! What is honestly testable, and is tested below:
    //!
    //! 1. **Contract-constant pins** — the arch / name / category /
    //!    upstream / SPDX constants match the two converters exactly, and
    //!    the `_` vs `-` and vendor-prefix asymmetries are pinned.
    //! 2. **Both-variant metadata round-trip** — each arch binds, resolves
    //!    to the right variant, and surfaces its own provenance key.
    //! 3. **Measured topology** — layer stacks are discovered, counted and
    //!    checked for contiguity + uniformity on synthetic manifests.
    //! 4. **Loud negative space** — every stated gate (missing arch /
    //!    foreign arch / crossed arch+name / empty manifest / declared but
    //!    absent tensor / index hole / non-uniform stack / absent lookup /
    //!    empty PCM / deferred forward) fires at its documented surface in
    //!    its documented variant, and names what a reader needs.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufArray, GgufBuilder};

    /// One F32 tensor's worth of zero payload for `dims`.
    fn payload(dims: &[u64]) -> Vec<u8> {
        let n: u64 = dims.iter().product();
        vec![0u8; (n as usize) * 4]
    }

    /// Builds a GGUF with the given arch + name and a caller-supplied
    /// tensor-name list (all `[2, 3]` F32), optionally stamping the
    /// weight-license class.
    fn gguf_with(
        arch: &str,
        name: Option<&str>,
        tensor_names: &[&str],
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, arch);
        if let Some(n) = name {
            b.add_string(chunks::KEY_MODEL_NAME, n);
        }
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        }
        for tn in tensor_names {
            b.add_tensor(tn, GgmlType::F32, vec![2, 3], payload(&[2, 3]))
                .expect("add_tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// A two-layer, two-leaf-per-layer encoder stack plus a front-end stem
    /// and a CTC head — the shape a well-formed GigaAM manifest has, at
    /// toy scale.
    const WELL_FORMED_TENSORS: &[&str] = &[
        "encoder.pre_encode.conv.0.weight",
        "encoder.layers.0.self_attn.linear_q.weight",
        "encoder.layers.0.conv_module.depthwise_conv.weight",
        "encoder.layers.1.self_attn.linear_q.weight",
        "encoder.layers.1.conv_module.depthwise_conv.weight",
        "head.decoder_layers.0.weight",
    ];

    /// Builds a well-formed v3 GGUF, stamping the v3 upstream HF slug.
    fn v3_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_V3);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_V3);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF_V3);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        }
        for tn in WELL_FORMED_TENSORS {
            b.add_tensor(tn, GgmlType::F32, vec![2, 3], payload(&[2, 3]))
                .expect("add_tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// Builds a well-formed multilingual GGUF, stamping the GitHub URL
    /// under the `upstream_url` key its converter actually uses.
    fn multilingual_gguf() -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_MULTILINGUAL);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_MULTILINGUAL);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL_MULTILINGUAL);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
        for tn in WELL_FORMED_TENSORS {
            b.add_tensor(tn, GgmlType::F32, vec![2, 3], payload(&[2, 3]))
                .expect("add_tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1 — Contract-constant pins (cross-crate consistency with BOTH
    //     converters).
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_are_stable() {
        // Arch tags — note the deliberate asymmetry: v3 carries the `sber_`
        // vendor prefix, multilingual does not. Both are load-bearing on
        // the wire.
        assert_eq!(ARCH_V3, "sber_gigaam_v3");
        assert_eq!(ARCH_MULTILINGUAL, "gigaam_multilingual");
        assert!(
            ARCH_V3.starts_with("sber_"),
            "v3 arch keeps a vendor prefix"
        );
        assert!(
            !ARCH_MULTILINGUAL.starts_with("sber_"),
            "multilingual arch deliberately has no vendor prefix"
        );

        // Names — `-` separated where the arches are `_` separated.
        assert_eq!(NAME_V3, "gigaam-v3");
        assert_eq!(NAME_MULTILINGUAL, "sber-gigaam-multilingual");
        assert!(!ARCH_V3.contains('-'), "arch tags use `_`");
        assert!(!NAME_V3.contains('_'), "model names use `-`");

        assert_eq!(CATEGORY, "asr");
        assert_eq!(UPSTREAM_HF_V3, "ai-sage/GigaAM-v3");
        assert_eq!(
            UPSTREAM_URL_MULTILINGUAL,
            "github.com/salute-developers/GigaAM"
        );
        assert_eq!(DEFAULT_LICENSE_SPDX, "mit");
        assert_eq!(KEY_REQUIRED_TENSORS, "vokra.gigaam.required_tensors");
        assert_eq!(LAYER_STACK_INFIX, ".layers.");
        assert_eq!(PRIMARY_SOURCE_REPO, UPSTREAM_URL_MULTILINGUAL);
        assert!(
            PRIMARY_SOURCE_HF_V3.ends_with(UPSTREAM_HF_V3),
            "the v3 HF anchor must be the HF host plus the stamped slug, so the two \
             cannot drift apart"
        );
        assert_eq!(CONVERTER_PATH_V3, GigaamVariant::V3.converter_path());
        assert_eq!(
            CONVERTER_PATH_MULTILINGUAL,
            GigaamVariant::Multilingual.converter_path()
        );
        assert_eq!(SIDECAR_PATH_V3, GigaamVariant::V3.sidecar_path());
        assert_eq!(
            SIDECAR_PATH_MULTILINGUAL,
            GigaamVariant::Multilingual.sidecar_path()
        );

        // `converter_arg` mirrors `ModelKind::as_arg` on the writer side
        // (`crates/vokra-convert/src/lib.rs`, the `Self::SberGigaamV3` /
        // `Self::SberGigaamMultilingual` arms). A drift here would print a
        // repro command that does not exist.
        assert_eq!(GigaamVariant::V3.converter_arg(), "sber-gigaam-v3");
        assert_eq!(
            GigaamVariant::Multilingual.converter_arg(),
            "sber-gigaam-multilingual"
        );

        // The accepted set is exactly the two family members, and nothing
        // from the sibling ASR fleet leaked into it.
        assert_eq!(ACCEPTED_ARCHS.len(), 2, "exactly the two family members");
        assert_eq!(ACCEPTED_ARCHS[0], ARCH_V3);
        assert_eq!(ACCEPTED_ARCHS[1], ARCH_MULTILINGUAL);
        for sibling in ASR_SIBLING_ARCHS {
            assert!(
                !ACCEPTED_ARCHS.contains(sibling),
                "sibling ASR arch `{sibling}` must NOT be accepted by the GigaAM binder"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2 — Variant resolution + per-variant metadata.
    // -----------------------------------------------------------------------

    #[test]
    fn variant_round_trips_through_arch_and_carries_distinct_metadata() {
        for v in [GigaamVariant::V3, GigaamVariant::Multilingual] {
            assert_eq!(
                GigaamVariant::from_arch(v.arch()),
                Some(v),
                "arch string must round-trip back to its variant"
            );
            assert_eq!(v.sibling().sibling(), v, "sibling() must be an involution");
        }
        assert_eq!(GigaamVariant::from_arch("parakeet-ctc"), None);
        assert_eq!(GigaamVariant::from_arch(""), None);

        // The two halves differ in every identity-bearing field...
        assert_ne!(GigaamVariant::V3.arch(), GigaamVariant::Multilingual.arch());
        assert_ne!(GigaamVariant::V3.name(), GigaamVariant::Multilingual.name());
        assert_ne!(
            GigaamVariant::V3.upstream_key(),
            GigaamVariant::Multilingual.upstream_key(),
            "v3 stamps upstream_hf, multilingual stamps upstream_url"
        );
        assert_ne!(
            GigaamVariant::V3.vocabulary_scope(),
            GigaamVariant::Multilingual.vocabulary_scope()
        );
        // ...and agree on the ones the converters share.
        assert_eq!(GigaamVariant::V3.upstream_key(), KEY_PROVENANCE_UPSTREAM_HF);
        assert_eq!(
            GigaamVariant::Multilingual.upstream_key(),
            KEY_PROVENANCE_UPSTREAM_URL
        );
        assert_eq!(GigaamVariant::V3.upstream_value(), UPSTREAM_HF_V3);
        assert_eq!(
            GigaamVariant::Multilingual.upstream_value(),
            UPSTREAM_URL_MULTILINGUAL
        );
    }

    // -----------------------------------------------------------------------
    // 3 — Both arches bind, with the right variant + provenance surfaced.
    // -----------------------------------------------------------------------

    #[test]
    fn v3_gguf_binds_and_surfaces_provenance() {
        let file = v3_gguf(Some(LicenseClass::Permissive));
        let m = Gigaam::from_gguf(&file).expect("well-formed v3 GGUF must bind");

        assert_eq!(m.variant(), GigaamVariant::V3);
        assert_eq!(m.model_name(), Some(NAME_V3));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(
            m.upstream(),
            Some(UPSTREAM_HF_V3),
            "v3 upstream must be read from the `upstream_hf` key"
        );
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
        assert_eq!(m.license_spdx(), Some(DEFAULT_LICENSE_SPDX));
        assert_eq!(m.tensor_count(), WELL_FORMED_TENSORS.len());
    }

    #[test]
    fn multilingual_gguf_binds_and_reads_the_url_provenance_key() {
        let file = multilingual_gguf();
        let m = Gigaam::from_gguf(&file).expect("well-formed multilingual GGUF must bind");

        assert_eq!(m.variant(), GigaamVariant::Multilingual);
        assert_eq!(m.model_name(), Some(NAME_MULTILINGUAL));
        assert_eq!(
            m.upstream(),
            Some(UPSTREAM_URL_MULTILINGUAL),
            "multilingual upstream must be read from the `upstream_url` key, not `upstream_hf`"
        );
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
    }

    // -----------------------------------------------------------------------
    // 4 — The measured topology probe.
    // -----------------------------------------------------------------------

    #[test]
    fn topology_measures_the_layer_stack() {
        let file = v3_gguf(Some(LicenseClass::Permissive));
        let m = Gigaam::from_gguf(&file).expect("bind");
        let topo = m.topology();

        let stack = topo
            .stack("encoder.layers.")
            .expect("the encoder stack must be discovered");
        assert_eq!(stack.n_layer(), 2, "two layer indices are present");
        assert_eq!(stack.tensors_per_layer(), 2);
        let leaves: Vec<&str> = stack.leaf_suffixes().iter().map(String::as_str).collect();
        assert_eq!(
            leaves,
            [
                "conv_module.depthwise_conv.weight",
                "self_attn.linear_q.weight",
            ],
            "leaf suffixes are reported sorted, so diagnostics are deterministic"
        );
        assert_eq!(
            stack.tensor_name(1, "self_attn.linear_q.weight"),
            "encoder.layers.1.self_attn.linear_q.weight",
            "tensor_name must reconstruct the exact on-disk name"
        );

        // Neither `encoder.pre_encode.conv.0.weight` nor
        // `head.decoder_layers.0.weight` carries the `.layers.` infix, so both
        // are counted as non-stack tensors rather than silently dropped. The
        // second is a deliberate NEAR MISS: it contains `_layers.`, and a probe
        // that searched for the bare substring `layers.` instead of `.layers.`
        // would wrongly fold the CTC head into the encoder stack.
        assert_eq!(topo.non_stack_tensors(), 2);
        assert_eq!(topo.stacks().len(), 1);
        assert!(
            topo.stack("head.decoder_layers.").is_none(),
            "`head.decoder_layers.0.weight` must not be mistaken for a layer stack"
        );
    }

    #[test]
    fn topology_probe_tolerates_a_manifest_with_no_layer_stack() {
        // A checkpoint that does not use `.layers.` naming is unusual but
        // not malformed — refusing it would re-open the "converted weights
        // are unloadable" gap this module exists to close.
        let file = gguf_with(
            ARCH_V3,
            Some(NAME_V3),
            &["encoder.weight", "head.bias"],
            Some(LicenseClass::Permissive),
        );
        let m = Gigaam::from_gguf(&file).expect("a stackless manifest must still bind");
        assert!(m.topology().stacks().is_empty());
        assert_eq!(m.topology().non_stack_tensors(), 2);
    }

    #[test]
    fn split_stack_name_rejects_non_stack_shapes() {
        assert_eq!(
            split_stack_name("encoder.layers.3.self_attn.linear_q.weight"),
            Some(("encoder.layers.", 3, "self_attn.linear_q.weight"))
        );
        // No infix at all.
        assert_eq!(split_stack_name("encoder.weight"), None);
        // Infix present but the segment after it is not an index.
        assert_eq!(split_stack_name("encoder.layers.foo.weight"), None);
        // Index present but nothing follows it.
        assert_eq!(split_stack_name("encoder.layers.0"), None);
    }

    // -----------------------------------------------------------------------
    // 5 — Loud negative space: arch gates.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "something-else");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native GigaAM GGUF"),
                    "message must name the surface, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_expected_and_actual() {
        // A Parakeet-CTC GGUF handed to the GigaAM binder by mistake. Both
        // are Conformer + CTC ASR with `category = "asr"`, so the category
        // tag cannot disambiguate them — only the arch tag can.
        let file = gguf_with(
            "parakeet-ctc",
            Some("parakeet-ctc-1.1b"),
            &["encoder.layers.0.probe.weight"],
            None,
        );
        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // Names the ACTUAL arch...
                assert!(
                    m.contains("`parakeet-ctc`"),
                    "message must name the observed arch, got `{m}`"
                );
                // ...and BOTH expected arches.
                assert!(
                    m.contains(ARCH_V3) && m.contains(ARCH_MULTILINGUAL),
                    "message must name both expected arch tags, got `{m}`"
                );
                // Enumerates the sibling ASR fleet so the reader knows why
                // the category tag was not enough.
                for sibling in ["canary", "omniasr-ctc", "whisper"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling `{sibling}` in the disambiguation: {m}"
                    );
                }
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_crossed_arch_and_name() {
        // arch says v3, name says multilingual — the two chunks disagree
        // about which vocabulary the CTC head emits over.
        let file = gguf_with(
            ARCH_V3,
            Some(NAME_MULTILINGUAL),
            &["encoder.layers.0.probe.weight"],
            Some(LicenseClass::Permissive),
        );
        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad when arch and name name different variants");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(ARCH_V3) && m.contains(NAME_MULTILINGUAL),
                    "message must name both crossed values, got `{m}`"
                );
                assert!(
                    m.contains("vocabulary"),
                    "message must explain the consequence, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // The matching pair must still bind — the gate is precise, not a
        // blanket name check (a renamed-but-consistent GGUF is fine).
        let ok = gguf_with(
            ARCH_V3,
            Some("a-downstream-repack-name"),
            &["encoder.layers.0.probe.weight"],
            Some(LicenseClass::Permissive),
        );
        assert!(
            Gigaam::from_gguf(&ok).is_ok(),
            "a name that is not the SIBLING's name must not trip the crossed-wires gate"
        );
    }

    // -----------------------------------------------------------------------
    // 6 — Loud negative space: tensor-manifest gates.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_V3);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_V3);
        // No tensors at all.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(m.contains("zero tensors"), "message must name the gap: {m}");
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
                assert!(
                    m.contains("vokra-cli convert --model sber-gigaam-v3"),
                    "message must include a runnable repro: {m}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn dims_lookup_names_the_absent_tensor() {
        let file = v3_gguf(Some(LicenseClass::Permissive));
        let m = Gigaam::from_gguf(&file).expect("bind");

        // A present tensor resolves to its GGUF-side dims.
        assert_eq!(
            m.weights()
                .dims("encoder.layers.0.self_attn.linear_q.weight")
                .expect("present tensor"),
            &[2, 3]
        );
        assert!(m.weights().has("head.decoder_layers.0.weight"));

        // An absent one is a loud error NAMING the tensor — never a
        // `None` a caller could swallow.
        let Err(err) = m.weights().dims("encoder.layers.9.does_not_exist.weight") else {
            panic!("expected ModelLoad for an absent tensor");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("encoder.layers.9.does_not_exist.weight"),
                    "message must name the absent tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn declared_required_tensor_that_is_absent_fails_at_load() {
        // A producer that stamps `KEY_REQUIRED_TENSORS` asserts it wrote
        // those tensors; a truncated upload must therefore fail at LOAD
        // time naming the first missing one.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_V3);
        b.add_string(chunks::KEY_MODEL_NAME, NAME_V3);
        b.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: vec![
                    GgufMetadataValue::String("encoder.layers.0.self_attn.linear_q.weight".into()),
                    GgufMetadataValue::String("head.ctc_projection.weight".into()),
                ],
            }),
        );
        // Only the FIRST declared tensor is actually written.
        b.add_tensor(
            "encoder.layers.0.self_attn.linear_q.weight",
            GgmlType::F32,
            vec![2, 3],
            payload(&[2, 3]),
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad when a declared required tensor is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("head.ctc_projection.weight"),
                    "message must name the missing declared tensor, got `{m}`"
                );
                assert!(
                    m.contains(KEY_REQUIRED_TENSORS),
                    "message must name the declaring key, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn empty_required_tensor_declaration_is_a_producer_bug() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_V3);
        b.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: Vec::new(),
            }),
        );
        b.add_tensor(
            "encoder.weight",
            GgmlType::F32,
            vec![2, 3],
            payload(&[2, 3]),
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty required-tensor declaration");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("empty list") || m.contains("empty required-tensor"),
                    "message must name the empty-declaration bug, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7 — Loud negative space: structural topology gates.
    // -----------------------------------------------------------------------

    #[test]
    fn non_uniform_layer_stack_is_refused_naming_the_missing_tensor() {
        // Layer 0 carries two leaves, layer 1 carries only one — the
        // signature of a truncated / mis-merged conversion.
        let file = gguf_with(
            ARCH_V3,
            Some(NAME_V3),
            &[
                "encoder.layers.0.self_attn.linear_q.weight",
                "encoder.layers.0.conv_module.depthwise_conv.weight",
                "encoder.layers.1.self_attn.linear_q.weight",
            ],
            Some(LicenseClass::Permissive),
        );
        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad on a non-uniform layer stack");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("encoder.layers.1.conv_module.depthwise_conv.weight"),
                    "message must name the EXACT absent tensor, got `{m}`"
                );
                assert!(
                    m.contains("non-uniform"),
                    "message must name the defect class, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn layer_index_hole_is_refused_naming_the_index() {
        // Indices {0, 2} — index 1 is missing entirely.
        let file = gguf_with(
            ARCH_MULTILINGUAL,
            Some(NAME_MULTILINGUAL),
            &[
                "encoder.layers.0.self_attn.linear_q.weight",
                "encoder.layers.2.self_attn.linear_q.weight",
            ],
            Some(LicenseClass::Permissive),
        );
        let Err(err) = Gigaam::from_gguf(&file) else {
            panic!("expected ModelLoad on a layer-index hole");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("encoder.layers."),
                    "message must name the stack root, got `{m}`"
                );
                assert!(
                    m.contains("index 1"),
                    "message must name the absent index, got `{m}`"
                );
                assert!(m.contains("hole"), "message must name the defect: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 8 — Fail-closed licensing.
    // -----------------------------------------------------------------------

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        let file = gguf_with(
            ARCH_V3,
            Some(NAME_V3),
            &["encoder.layers.0.self_attn.linear_q.weight"],
            None,
        );
        let m = Gigaam::from_gguf(&file).expect("license is a surface, not a bind gate");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an unstamped weight-license must fail CLOSED to Unknown"
        );
        assert_eq!(m.license_spdx(), None);
    }

    // -----------------------------------------------------------------------
    // 9 — The loud-partial forward.
    // -----------------------------------------------------------------------

    #[test]
    fn transcribe_loud_partials_naming_all_three_blockers() {
        let file = v3_gguf(Some(LicenseClass::Permissive));
        let m = Gigaam::from_gguf(&file).expect("bind");

        // A legitimate PCM shape: 1 s of silence at 16 kHz mono.
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.transcribe(&pcm) else {
            panic!("transcribe must loud-partial — never fabricate a transcript");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("gigaam transcribe"), "surface: {msg}");
                assert!(msg.contains("loud-partial"), "posture label: {msg}");
                assert!(
                    msg.contains(ARCH_V3),
                    "message must name the variant: {msg}"
                );

                // Blocker 1 — the front-end spec.
                assert!(
                    msg.contains("MISSING FRONT-END SPEC"),
                    "blocker 1 must be named: {msg}"
                );
                assert!(
                    msg.contains("vokra.gigaam.*"),
                    "blocker 1 must name the absent metadata group: {msg}"
                );

                // Blocker 2 — the encoder tensor-name mapping.
                assert!(
                    msg.contains("MISSING ENCODER TENSOR-NAME MAPPING"),
                    "blocker 2 must be named: {msg}"
                );
                assert!(
                    msg.contains("SHAPE-VALID"),
                    "blocker 2 must state why guessing is silent-wrong: {msg}"
                );

                // Blocker 3 — the CTC vocabulary.
                assert!(
                    msg.contains("MISSING CTC VOCABULARY"),
                    "blocker 3 must be named: {msg}"
                );

                // The composition needs no new kernel — say so, and name
                // the primitives that already exist.
                for op in [
                    "vokra_ops::conformer",
                    "vokra_ops::ctc_decode_greedy",
                    "vokra_ops::kaldi_fbank",
                ] {
                    assert!(
                        msg.contains(op),
                        "expected existing primitive `{op}`: {msg}"
                    );
                }

                // Actionable anchors: the converter to extend, the sidecar,
                // and the primary sources.
                assert!(
                    msg.contains(CONVERTER_PATH_V3),
                    "message must name the converter to extend: {msg}"
                );
                assert!(
                    msg.contains(SIDECAR_PATH_V3),
                    "message must name the offline sidecar: {msg}"
                );
                assert!(
                    msg.contains(PRIMARY_SOURCE_REPO),
                    "message must cite the upstream repository: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn multilingual_loud_partial_names_its_own_variant_and_converter() {
        let file = multilingual_gguf();
        let m = Gigaam::from_gguf(&file).expect("bind");
        let Err(err) = m.transcribe(&[0.0_f32; 512]) else {
            panic!("transcribe must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains(ARCH_MULTILINGUAL),
                    "message must name the multilingual arch: {msg}"
                );
                assert!(
                    msg.contains(CONVERTER_PATH_MULTILINGUAL),
                    "message must point at the multilingual converter: {msg}"
                );
                assert!(
                    msg.contains("70+-language"),
                    "message must name the multilingual vocabulary scope: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_rejects_an_empty_waveform() {
        let file = v3_gguf(Some(LicenseClass::Permissive));
        let m = Gigaam::from_gguf(&file).expect("bind");
        let Err(err) = m.transcribe(&[]) else {
            panic!("an empty PCM buffer must be refused, not silently transcribed to \"\"");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("empty PCM"),
                    "message must name the defect, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }
}
