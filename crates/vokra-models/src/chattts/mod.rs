//! **ChatTTS** (`2Noise/ChatTTS`, **cc-by-nc-4.0**) — runtime binder for the
//! `chattts` converter arch (Wave G 2026-08-15, loud-partial per the `maest` /
//! `atst` / `m2d` / `emotion2vec` / `panns` / `redimnet` precedent —
//! CLAUDE.md 教訓 (a):「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/chattts.rs` has been stamping
//! `vokra.model.arch = "chattts"` since the coverage-audit-2026-08-03 Wave D
//! T4 landing, and a workspace-wide grep proved that **nothing read that arch
//! string back** — a converted ChatTTS artifact was unloadable. This module is
//! that consumer.
//!
//! # Primary sources
//!
//! Every fact below is transcribed from a file in this repository. Nothing is
//! re-derived from memory (CLAUDE.md「ハルシネーション厳禁」).
//!
//! - The converter itself, `crates/vokra-convert/src/models/chattts.rs` — the
//!   authority for [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//!   [`DEFAULT_LICENSE_SPDX`] and for the verbatim-passthrough tensor-naming
//!   contract.
//! - The audit ticket,
//!   `docs/tickets/coverage-audit-2026-08-03/wave-d/chattts.md` — the authority
//!   for the module bundle breakdown, the parameter scale, the upstream
//!   packaging and the ELVIS Act flag.
//! - `docs/license-audit.md` §3.1 — records `cc-by-nc-4.0`, T4 tier,
//!   `--allow-noncommercial` required to publish.
//! - Upstream code: <https://github.com/2noise/ChatTTS>
//! - Upstream weights: <https://huggingface.co/2Noise/ChatTTS>
//!
//! # What ChatTTS is
//!
//! A **dialogue-oriented** TTS: a GPT-style autoregressive decoder over
//! discrete speech tokens, SFT-tuned on conversational audio so it produces the
//! disfluencies real speech carries — laughter, pauses, hesitation — driven by
//! inline tags (`[laugh]`, `[uv_break]`). Per the audit ticket it is a
//! **three-module bundle** plus two side assets:
//!
//! ```text
//! text (+ inline dialogue tags)
//!   -> GPT-style autoregressive backbone  (~200M)   ← **loud-partial**
//!        emits discrete speech tokens.
//!   -> DVAE                                (~100M)  ← **loud-partial**
//!        decodes speech tokens to an acoustic representation.
//!   -> Vocos vocoder head                  (~100M)  ← **loud-partial**
//!        ConvNeXt-V2 + iSTFT to a waveform.
//!   -> waveform (sample rate NOT recoverable from the artifact — see below)
//! ```
//!
//! plus `spk_stat` (the 30-d speaker-prompt statistics) and a GPT-2-lineage
//! tokenizer. Roughly 400M parameters across the bundle.
//!
//! # Why the forward is loud-partial — the axis group does not exist
//!
//! This is the whole reason, and it is a property of the artifact rather than a
//! gap in ambition. The converter is a **pure float pass-through**: it stamps
//! [`ARCH`], [`NAME`], [`CATEGORY`], [`UPSTREAM_HF`] and the
//! `vokra.provenance.*` group, and then copies every F32 / F16 / BF16 tensor
//! through verbatim. It stamps **no `vokra.chattts.*` topology group at all** —
//! no layer count, no hidden width, no head count, no vocab size, no codebook
//! layout, no hop length, not even an output sample rate.
//!
//! So there is no honest way to compose a forward here. Every axis would have
//! to be guessed, and a guessed axis is the specific failure mode this
//! repository has already been bitten by: shape-valid, numerically wrong, and
//! silent. Nor can the shapes be back-derived from the manifest, because the
//! second half of the contract is missing too — the upstream ships the four
//! modules as **separate torch pickles**, and the flattening script the ticket
//! specifies (`tools/parity/chattts_prepare_checkpoint.py`) has not been
//! written, so the module-namespace convention that would let a binder walk
//! `state_dict` names is not yet pinned by anything.
//!
//! Two things must land before [`ChatTts::synthesize`] can return real audio:
//! that prep script (fixing the naming convention), and a converter change
//! stamping a `vokra.chattts.*` axis group. Until then this binder refuses
//! rather than fabricates (FR-EX-08).
//!
//! # The published artifact is GPT-only — which is why module census is real
//!
//! `docs/tickets/coverage-audit-2026-08-03/INDEX.md` (UPDATE 2026-08-04 #3)
//! records that the published `vokra/chattts` repository was built from
//! `asset/gpt/model.safetensors` **alone** (~814 MB), with the rest of the
//! ~2.2 GB bundle explicitly deferred to "runtime binder 実装時" — i.e. to this
//! wave. So the artifact a caller is most likely to hold today carries the GPT
//! backbone and neither the DVAE nor the Vocos head.
//!
//! A binder that ignored that would bind happily and then die deep in the
//! vocoder with a missing-tensor trail. Instead [`ChatTtsWeights::module_census`]
//! probes the manifest for the four module namespaces and
//! [`ChatTts::synthesize`] names **which modules are absent from the artifact in
//! the caller's hand**. That part is real, disk-derived and tested — see the
//! honesty caveat on [`MODULE_PREFIX_GPT`] about what the probe does and does
//! not prove.
//!
//! # Real / loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! **Real**:
//!
//! - [`ChatTts::from_gguf`] with **strict** `vokra.model.arch == "chattts"`
//!   verification, naming **both** the expected and the actual tag and
//!   enumerating the TTS neighbourhood (FR-EX-08 — never a silent misroute).
//! - [`ChatTtsWeights::from_gguf`] manifest binding with a non-empty gate, plus
//!   [`ChatTtsWeights::require_tensor`] / [`ChatTtsWeights::require_tensor_dims`]
//!   lookups that name the missing tensor, or **both** the expected and the
//!   actual dims.
//! - [`ChatTtsWeights::module_census`] — the on-disk module probe described
//!   above.
//! - Metadata surfacing: [`ChatTts::name`] / [`ChatTts::category`] /
//!   [`ChatTts::upstream_hf`] / [`ChatTts::model_id`] / [`ChatTts::source`].
//! - Weight-licence surfacing, fail-closing to [`LicenseClass::Unknown`] when
//!   the artifact carries no stamp, and the compliance-gated
//!   [`ChatTts::from_gguf_with_policy`] / [`ChatTts::from_path`] /
//!   [`ChatTts::from_path_with_policy`] entry points.
//!
//! **Loud-partial**: [`ChatTts::synthesize`] returns
//! [`VokraError::UnsupportedOp`] naming the three deferred modules, the missing
//! `vokra.chattts.*` axis group, the un-written prep script, the on-disk module
//! census and all three primary sources. No fabricated waveform is ever emitted
//! (FR-EX-08 — no silent partial output).
//!
//! # Licence posture — T4 NonCommercial, fail-closed
//!
//! The converter's [`DEFAULT_LICENSE_SPDX`] is `cc-by-nc-4.0`, which
//! `LicenseClass::from_license_str` resolves to [`LicenseClass::NonCommercial`],
//! whose [`LicenseClass::requires_research_flag`] is `true`. A correctly stamped
//! ChatTTS artifact is therefore **refused** under [`CompliancePolicy::strict`]
//! and loads only with an explicit research opt-in
//! ([`CompliancePolicy::with_research_license`],
//! `VOKRA_ALLOW_RESEARCH_LICENSE=1`, or `ComplianceLevel::Research`).
//!
//! **That refusal is the fail-closed default working as intended, not a
//! defect** — and it is tested from both sides. The in-tree precedent is the
//! `maest` binder (`cc-by-nc-sa-4.0`); the publish-side precedent is X-Codec-2,
//! the first T4 tier release, which is why `docs/license-audit.md` §3.1 records
//! `--allow-noncommercial` as mandatory for this model.
//!
//! This binder only *surfaces* whatever class the artifact carries.
//! `docs/license-audit.md` §3.1 sign-off stays **owner-only** per memory
//! `[[feedback-license-signoff-primary-source]]` — CC does not sign, and does
//! not treat a converter default as a sign-off.
//!
//! # ELVIS Act — deliberately no speaker-prompt injection surface
//!
//! The audit ticket flags ChatTTS as **borderline** for the ELVIS Act
//! voice-cloning-tool test: the official release offers only "voice random
//! sampling" (the 30-d `spk_emb` is derived from a seed), but technically an
//! arbitrary 30-d vector can be substituted, which is exactly the
//! "primary purpose or effect" question CLAUDE.md 設計判断 8 turns on. The
//! ticket routes that to an owner ADR, and the ADR has not been made.
//!
//! Accordingly this module exposes **no** speaker-embedding injection entry
//! point — no `synthesize_with_speaker`, no `spk_emb` setter — and will not
//! grow one before that ADR lands. [`ChatTtsModuleCensus::speaker_stats`] only
//! *reports whether a `spk_stat` group is present on disk*, which is plain
//! manifest reporting and is precisely the input the owner ADR needs; reporting
//! presence is categorically different from providing the injection path, and
//! only the latter is the trigger.
//!
//! # Sibling family distinctness (TTS neighbourhood)
//!
//! [`ARCH`] = `"chattts"` is deliberately distinct from every sibling TTS arch
//! tag in this workspace, each of which is a different topology behind a
//! different loader — `piper-plus-mb-istft-vits2`, `kokoro-82m-istftnet`,
//! `cosyvoice2`, `cosyvoice3`, `chatterbox`, `styletts2`, `vibevoice`, `dia`,
//! `zonos`, `qwen3_tts`, `sbv2`, `voxcpm2`, `melotts`, `irodori-tts`, `csm`,
//! `moshi`. The sharpest confusable is **`vocos`**: ChatTTS's vocoder head *is*
//! Vocos, so a bundle artifact legitimately contains Vocos-shaped tensors — but
//! a bare `vocos` GGUF is a standalone vocoder with no GPT backbone and no
//! DVAE, so binding one here would produce a handle that can never synthesise.
//! Sharing an arch tag would let runtime dispatch bind the wrong loader
//! (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] are **mirrors of the converter's constants** — the
//! same rule every sibling binder (`maest` / `atst` / `m2d` / `emotion2vec` /
//! `panns` / `redimnet`) follows so `vokra-models` does not gain a dependency
//! edge onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # Numerical parity is NOT claimed
//!
//! No parity run against a real ChatTTS checkpoint has happened in this
//! repository. The tests below assert structure, metadata round-trip and
//! loud-error negative space — never an expected numeric value, since inventing
//! one would be fabrication wearing the costume of verification.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Upstream ships torch pickles per module; the converter accepts **safetensors
//! only** and this runtime **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02). The pickle→safetensors bridge is the owner-side, uv-managed
//! Python 3.12 prep script the ticket specifies — it never enters the runtime.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/chattts.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model chattts`.
///
/// Distinct from every sibling TTS arch tag — see the module docstring's
/// "Sibling family distinctness" section, and note the `vocos` pair in
/// particular: ChatTTS *contains* a Vocos head, so shape overlap is real while
/// loader identity is not (FR-EX-08).
pub const ARCH: &str = "chattts";

/// Expected `vokra.model.name` value written by the converter for the canonical
/// `2Noise/ChatTTS` release.
///
/// The converter's `NAME` equals its `ARCH` for this model (single release, no
/// variant enum yet — the ticket reserves `turbo` / `mini` for a future enum),
/// so this constant is deliberately identical to [`ARCH`]. It is nevertheless
/// **surfaced, not gated** by [`ChatTts::from_gguf`], so a future variant
/// publishing under this arch with its own name stays loadable.
pub const NAME: &str = "chattts";

/// Expected `vokra.model.category` value — `tts`, shared with the whole TTS
/// family. Consumed by the model-card generator and the zoo-manifest tier gate.
///
/// The audit ticket labels the model `tts/dialogue`; the converter folds that to
/// the plain `tts` the runtime dispatch uses, matching the sibling uniform
/// posture.
pub const CATEGORY: &str = "tts";

/// Upstream HuggingFace slug — stamped on `vokra.provenance.upstream_hf`.
///
/// Note the capitalisation: the GitHub organisation is `2noise` but the
/// HuggingFace repository is `2Noise`, and the converter records the latter.
pub const UPSTREAM_HF: &str = "2Noise/ChatTTS";

/// Default SPDX stamped by the converter — the **weight** tier.
///
/// Resolves to [`LicenseClass::NonCommercial`] (T4). A caller with a different
/// attestation may override at the converter boundary (`--license <spdx>`),
/// which is why this binder *surfaces* rather than *asserts* the class.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// Metadata key holding [`CATEGORY`] (not part of `vokra_core::gguf::chunks`,
/// so mirrored here from the converter).
pub const GGUF_KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key holding [`UPSTREAM_HF`] (not part of
/// `vokra_core::gguf::chunks`, so mirrored here from the converter).
pub const GGUF_KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// ---------------------------------------------------------------------------
// Primary-source anchors — cited inside the loud-partial error so a reader
// diagnosing the gap has fully specified places to walk.
// ---------------------------------------------------------------------------

/// Primary-source anchor: the upstream weight release.
pub const PRIMARY_SOURCE_UPSTREAM_HF: &str = "huggingface.co/2Noise/ChatTTS";

/// Primary-source anchor: the upstream reference implementation.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/2noise/ChatTTS";

/// Primary-source anchor: the in-repo audit ticket, which is this repository's
/// record of the module breakdown, the parameter scale and the ELVIS Act flag.
pub const PRIMARY_SOURCE_TICKET: &str = "docs/tickets/coverage-audit-2026-08-03/wave-d/chattts.md";

/// The prep script the audit ticket specifies and which **has not been
/// written**: it must merge the four upstream torch pickles into one
/// safetensors before the converter can see a whole bundle.
///
/// Named in the loud-partial error because it is the first of the two things
/// that must land before a real forward is possible (the second being a
/// converter change that stamps a `vokra.chattts.*` axis group).
pub const PREP_SCRIPT_PATH: &str = "tools/parity/chattts_prepare_checkpoint.py";

/// The `vokra.*` metadata prefix a future converter change must stamp before a
/// real forward can be composed. **No key under it exists today** — see the
/// module docstring's "Why the forward is loud-partial" section.
pub const AXIS_GROUP_PREFIX: &str = "vokra.chattts.";

// ---------------------------------------------------------------------------
// Module namespace probes.
// ---------------------------------------------------------------------------

/// Manifest prefix probed for the GPT-style autoregressive backbone (~200M
/// parameters per the audit ticket).
///
/// **Honesty caveat, shared by all four `MODULE_PREFIX_*` constants.** These
/// are the four upstream module names the audit ticket records (`GPT`, `DVAE`,
/// `vocos`, `spk_stat`), lower-cased into the namespace form a flattening script
/// would most plainly produce, and corroborated only weakly — by the
/// `gpt.`-prefixed sample names the converter's own test module uses, and by the
/// `asset/gpt/model.safetensors` path the publish log records. Because the prep
/// script that *fixes* the flattening convention does not exist yet (see
/// [`PREP_SCRIPT_PATH`]), a probe miss proves nothing about the artifact.
///
/// [`ChatTtsWeights::module_census`] therefore **reports** matches and never
/// errors on a miss: a zero count means "no tensor here starts with this
/// prefix", not "this module is absent". The distinction is load-bearing and is
/// restated wherever a census value is consumed.
pub const MODULE_PREFIX_GPT: &str = "gpt.";

/// Manifest prefix probed for the DVAE that decodes speech tokens to an
/// acoustic representation (~100M parameters per the audit ticket). See
/// [`MODULE_PREFIX_GPT`] for the honesty caveat.
pub const MODULE_PREFIX_DVAE: &str = "dvae.";

/// Manifest prefix probed for the Vocos vocoder head (~100M parameters per the
/// audit ticket). See [`MODULE_PREFIX_GPT`] for the honesty caveat.
///
/// The sibling standalone `vocos` converter arch exists in this workspace; a
/// ChatTTS bundle legitimately carries Vocos-shaped tensors *under this
/// module*, which is why arch identity — not tensor shape — is what routes the
/// loader.
pub const MODULE_PREFIX_VOCOS: &str = "vocos.";

/// Manifest prefix probed for the 30-d speaker-prompt statistics.
///
/// Deliberately carries **no trailing dot**, because the ticket describes
/// `spk_stat` as a side asset rather than a module tree, so it may land as a
/// single tensor rather than a namespace; the dot-less probe matches both
/// spellings. See [`MODULE_PREFIX_GPT`] for the honesty caveat, and the module
/// docstring's ELVIS Act section for why presence is *reported* while no
/// injection entry point is *provided*.
pub const MODULE_PREFIX_SPEAKER_STATS: &str = "spk_stat";

/// Human-readable label for the GPT backbone, used when listing which synthesis
/// modules are missing from an artifact.
pub const MODULE_LABEL_GPT: &str = "GPT autoregressive backbone";

/// Human-readable label for the DVAE, used when listing which synthesis modules
/// are missing from an artifact.
pub const MODULE_LABEL_DVAE: &str = "DVAE speech-token decoder";

/// Human-readable label for the Vocos head, used when listing which synthesis
/// modules are missing from an artifact.
pub const MODULE_LABEL_VOCOS: &str = "Vocos vocoder head";

// ---------------------------------------------------------------------------
// ChatTtsModuleCensus — pure on-disk reporting.
// ---------------------------------------------------------------------------

/// How many tensors on disk sit under each ChatTTS module namespace.
///
/// Pure manifest reporting with no interpretation. Every field is a count of
/// tensor names starting with the corresponding `MODULE_PREFIX_*` probe, and a
/// zero is **not** an assertion that the module is absent from the upstream
/// bundle — only that nothing on disk carries that prefix. See
/// [`MODULE_PREFIX_GPT`] for why that distinction matters until
/// [`PREP_SCRIPT_PATH`] lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsModuleCensus {
    /// Tensors under [`MODULE_PREFIX_GPT`].
    pub gpt: usize,
    /// Tensors under [`MODULE_PREFIX_DVAE`].
    pub dvae: usize,
    /// Tensors under [`MODULE_PREFIX_VOCOS`].
    pub vocos: usize,
    /// Tensors under [`MODULE_PREFIX_SPEAKER_STATS`].
    ///
    /// Reported so an owner making the ELVIS Act call has the fact in hand. No
    /// speaker-prompt injection surface exists on this type or any other in
    /// this module — see the module docstring.
    pub speaker_stats: usize,
    /// Total tensors in the manifest, whether or not they matched a probe.
    pub total_tensors: usize,
}

impl ChatTtsModuleCensus {
    /// Whether any probe matched at all.
    ///
    /// `false` means the artifact uses a namespace convention this repository
    /// has not transcribed — expected until [`PREP_SCRIPT_PATH`] fixes one, and
    /// never on its own an error.
    #[inline]
    #[must_use]
    pub const fn matched_any(&self) -> bool {
        self.gpt > 0 || self.dvae > 0 || self.vocos > 0 || self.speaker_stats > 0
    }

    /// Whether all three synthesis modules (GPT, DVAE, Vocos) matched.
    ///
    /// A `true` here means the artifact *looks like* a whole bundle rather than
    /// the GPT-only slice that was actually published; it does not by itself
    /// make a forward possible, because the `vokra.chattts.*` axis group is
    /// still missing (see the module docstring).
    #[inline]
    #[must_use]
    pub const fn synthesis_chain_complete(&self) -> bool {
        self.gpt > 0 && self.dvae > 0 && self.vocos > 0
    }

    /// The human-readable labels of whichever synthesis modules had no matching
    /// tensor, in pipeline order.
    ///
    /// `spk_stat` is excluded: it is a side asset, not a stage of the synthesis
    /// chain.
    #[must_use]
    pub fn missing_synthesis_modules(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.gpt == 0 {
            missing.push(MODULE_LABEL_GPT);
        }
        if self.dvae == 0 {
            missing.push(MODULE_LABEL_DVAE);
        }
        if self.vocos == 0 {
            missing.push(MODULE_LABEL_VOCOS);
        }
        missing
    }
}

// ---------------------------------------------------------------------------
// ChatTtsWeights — the tensor manifest, with loud lookups.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a ChatTTS GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification step. A
/// GGUF that carries zero tensors is rejected with [`VokraError::ModelLoad`]
/// (FR-EX-08 — a ~400M-parameter bundle never converts to an empty manifest, so
/// zero tensors always signals a mis-produced artifact, and binding it would
/// silently run an all-zero forward).
///
/// The payload is deliberately not dequantised: the forward is loud-partial
/// (see [`ChatTts::synthesize`]), and the follow-up wave sizes its dequant per
/// its kernel needs. [`require_tensor`](Self::require_tensor) /
/// [`require_tensor_dims`](Self::require_tensor_dims) are already in place so
/// that wave walks a manifest that fails loudly rather than substituting zeros.
#[derive(Debug, Clone)]
pub struct ChatTtsWeights {
    /// Tensors discovered on disk, in file order, as
    /// `(upstream state_dict name, GGUF-side dims)`.
    tensors: Vec<(String, Vec<usize>)>,
}

impl ChatTtsWeights {
    /// Scans `gguf` for the ChatTTS `state_dict` tensors.
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
                "chattts: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate ChatTTS artifact is a GPT backbone + DVAE + Vocos \
                 head bundle of roughly 400M parameters (arch={ARCH}, name={NAME}), and even \
                 the GPT-only slice that was actually published is ~814 MB, so an empty \
                 manifest always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model chattts` against an upstream `{UPSTREAM_HF}` \
                 safetensors checkpoint. Primary source: {PRIMARY_SOURCE_UPSTREAM_HF}"
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

    /// Probes the manifest for the four ChatTTS module namespaces.
    ///
    /// Never fails: a probe that matches nothing yields a zero count, because
    /// the flattening convention is not yet pinned (see [`MODULE_PREFIX_GPT`]).
    #[must_use]
    pub fn module_census(&self) -> ChatTtsModuleCensus {
        ChatTtsModuleCensus {
            gpt: self.count_with_prefix(MODULE_PREFIX_GPT),
            dvae: self.count_with_prefix(MODULE_PREFIX_DVAE),
            vocos: self.count_with_prefix(MODULE_PREFIX_VOCOS),
            speaker_stats: self.count_with_prefix(MODULE_PREFIX_SPEAKER_STATS),
            total_tensors: self.tensors.len(),
        }
    }

    /// Dims of a **required** tensor, failing loudly when it is absent.
    ///
    /// The error names the missing tensor, the manifest size, and up to five
    /// nearby names on disk so a caller diagnosing a namespace mismatch has
    /// something concrete to compare against.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `name` is not present (FR-EX-08 — never
    ///   substitute a zero tensor).
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
            "chattts: required tensor `{name}` is absent from the GGUF ({count} tensors \
             present; nearest names on disk: {near:?}). The converter passes upstream \
             safetensors keys through verbatim, so a mismatch means either the artifact is \
             the GPT-only slice (the published `vokra/chattts` repository was built from \
             `asset/gpt/model.safetensors` alone, with the DVAE / Vocos / spk_stat assets \
             deferred) or the four upstream modules were flattened under a different \
             namespace convention — which is not yet pinned by anything, because the prep \
             script `{PREP_SCRIPT_PATH}` has not been written. Refusing to substitute a zero \
             tensor (FR-EX-08). Primary sources: {PRIMARY_SOURCE_UPSTREAM_HF}, \
             {PRIMARY_SOURCE_TICKET}",
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
                "chattts: tensor `{name}` has dims {actual:?} but the caller expects \
                 {expected:?} — refusing to reshape or truncate silently (FR-EX-08). Note \
                 that the converter stamps no `{AXIS_GROUP_PREFIX}*` axis group, so any \
                 expected shape a caller holds today was derived outside the artifact and a \
                 disagreement here may mean the expectation is wrong rather than the \
                 payload. Primary sources: {PRIMARY_SOURCE_UPSTREAM_HF}, \
                 {PRIMARY_SOURCE_TICKET}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ChatTts — the runtime binder handle.
// ---------------------------------------------------------------------------

/// ChatTTS (`2Noise/ChatTTS`, CC-BY-NC-4.0) dialogue-oriented TTS runtime
/// binder.
///
/// Bind with [`from_gguf`](Self::from_gguf) — or, to run the M2-13 weight-licence
/// gate, the compliance-gated [`from_gguf_with_policy`](Self::from_gguf_with_policy)
/// / [`from_path`](Self::from_path) /
/// [`from_path_with_policy`](Self::from_path_with_policy). Because ChatTTS is
/// non-commercial, the gated routes **refuse** a correctly stamped artifact
/// unless the caller supplies a research opt-in; that is the fail-closed default
/// working as intended.
///
/// Binding is cheap: it reads the metadata and the tensor **manifest**, and
/// decodes no payload. [`synthesize`](Self::synthesize) is **loud-partial** —
/// see the module docstring for exactly which two things must land first.
#[derive(Debug, Clone)]
pub struct ChatTts {
    name: Option<String>,
    category: Option<String>,
    upstream_hf: Option<String>,
    model_id: Option<String>,
    source: Option<String>,
    weights: ChatTtsWeights,
    weight_license: LicenseClass,
    attribution: Option<String>,
}

impl ChatTts {
    /// Binds a ChatTTS GGUF: verifies the arch strictly, binds the tensor
    /// manifest, and surfaces the converter's metadata + licence stamps.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the missing
    /// or wrong key, so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// `vokra.model.name` is deliberately **surfaced, not gated**: the audit
    /// ticket reserves `turbo` / `mini` variants that would share this arch
    /// under different names, so a hard name check would make a legitimate
    /// future artifact unloadable. See [`Self::name`].
    ///
    /// This entry point performs **no licence gate** — use
    /// [`Self::from_gguf_with_policy`] for that.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is not `"chattts"` —
    ///   the message names both the found and the expected tag and enumerates
    ///   the TTS neighbourhood.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`ChatTtsWeights::from_gguf`]).
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

        // 3. Tensor manifest with the non-emptiness gate. There is deliberately
        //    no axis-group step between (2) and (3): the converter stamps no
        //    `vokra.chattts.*` group at all, so inventing a strict reader here
        //    would reject every artifact that exists.
        let weights = ChatTtsWeights::from_gguf(file)?;

        // 4. Provenance surfacing. The converter stamps `NonCommercial`
        //    (cc-by-nc-4.0); an artifact missing the stamp reads back as
        //    `Unknown` — fail-closed at the M2-13 gate either way, since both
        //    classes require the research flag.
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
            weights,
            weight_license,
            attribution,
        })
    }

    /// Loads a ChatTTS GGUF from raw bytes under `policy` (the M2-13
    /// weight-licence gate).
    ///
    /// ChatTTS ships **CC-BY-NC-4.0** → [`LicenseClass::NonCommercial`], whose
    /// [`LicenseClass::requires_research_flag`] is `true`, so a correctly
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
    ///   expected outcome for ChatTTS under a strict policy.
    /// - See [`Self::from_gguf`] for the remaining bind errors.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("chattts GGUF: {e}")))?;
        // Arch before the compliance gate so a mis-routed artifact reports the
        // arch mismatch rather than a licence verdict about a model the caller
        // never meant to load.
        verify_arch(&file)?;
        check_weight_license(&file, policy)?;
        Self::from_gguf(&file)
    }

    /// Loads a ChatTTS GGUF from a path under [`CompliancePolicy::strict`].
    ///
    /// Because ChatTTS is non-commercial, this route **refuses** a correctly
    /// stamped artifact — the fail-closed default working as intended, not a
    /// defect. Callers with a research/evaluation basis should use
    /// [`Self::from_path_with_policy`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Loads a ChatTTS GGUF from a path under an explicit `policy`.
    ///
    /// The route a research/evaluation consumer takes:
    /// `CompliancePolicy::strict().with_research_license(true)` unlocks the
    /// non-commercial gate and emits the mandatory research-only warning. The
    /// attribution obligation CC-BY-NC carries is not waived by that opt-in —
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

    /// The stamped `vokra.model.name`, if present — [`NAME`] for a
    /// converter-produced artifact.
    #[inline]
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The stamped `vokra.model.category`, if present — [`CATEGORY`] (`"tts"`)
    /// for a converter-produced artifact.
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
    /// free-text upstream description.
    #[inline]
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// The stamped `vokra.provenance.attribution`, if present.
    ///
    /// CC-BY-NC carries a **BY** obligation, so a downstream displaying this
    /// model owes credit. The converter's `stamp_provenance` call writes
    /// weight-licence, SPDX, model id and source but **not** attribution, so
    /// this reads `None` on a converter-produced artifact today even though the
    /// obligation stands. That is a recorded gap, surfaced rather than papered
    /// over.
    #[inline]
    #[must_use]
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &ChatTtsWeights {
        &self.weights
    }

    /// The on-disk module census — which of the four ChatTTS module namespaces
    /// have tensors in this artifact.
    ///
    /// Read the honesty caveat on [`MODULE_PREFIX_GPT`] before treating a zero
    /// as proof of absence.
    #[inline]
    #[must_use]
    pub fn module_census(&self) -> ChatTtsModuleCensus {
        self.weights.module_census()
    }

    /// The stamped weight-licence class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk.
    ///
    /// [`LicenseClass::NonCommercial`] for a converter-produced artifact
    /// (`cc-by-nc-4.0`); [`LicenseClass::Unknown`] when the stamp is absent.
    /// Both require the research flag, so both fail closed at the M2-13 gate.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Synthesises a waveform from `text` (with ChatTTS's inline dialogue tags,
    /// e.g. `[laugh]` / `[uv_break]`).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. The full ChatTTS forward needs the
    /// GPT-style autoregressive backbone, the DVAE speech-token decoder and the
    /// Vocos vocoder head, and **none of them can be composed from this
    /// artifact**: the converter stamps no `vokra.chattts.*` axis group at all,
    /// so every topology axis — layer count, hidden width, head count, vocab
    /// size, codebook layout, even the output sample rate — would have to be
    /// guessed, and a guessed axis is shape-valid, numerically wrong and
    /// silent.
    ///
    /// The error names both blockers ([`PREP_SCRIPT_PATH`] and the missing
    /// [`AXIS_GROUP_PREFIX`] group), reports the on-disk module census so the
    /// reader can see whether they even hold a whole bundle, and cites all three
    /// primary sources. **No fabricated waveform is ever emitted** (FR-EX-08 —
    /// no silent partial output).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate described above.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        // Bind explicitly so an unused-variable warning cannot mask a future
        // accidental removal of the parameter (mirror of the sibling
        // loud-partial signature discipline).
        let _ = text;
        Err(synthesize_loud_partial(&self.module_census()))
    }
}

/// Verifies `vokra.model.arch == "chattts"`, naming **both** tags on a
/// mismatch and enumerating the TTS neighbourhood.
///
/// Split out so [`ChatTts::from_gguf_with_policy`] can run it *ahead* of the
/// compliance gate: a caller who hands over the wrong file wants to hear about
/// the arch, not a licence verdict on a model they never meant to load.
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "chattts: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF produced by \
             `vokra-cli convert --model chattts`?). Every sibling TTS arch tag in this \
             workspace is a different topology behind a different loader — \
             `piper-plus-mb-istft-vits2`, `kokoro-82m-istftnet`, `cosyvoice2`, `cosyvoice3`, \
             `chatterbox`, `styletts2`, `vibevoice`, `dia`, `zonos`, `qwen3_tts`, `sbv2`, \
             `voxcpm2`, `melotts`, `irodori-tts`, `csm`, `moshi` — and the sharpest \
             confusable is `vocos`: ChatTTS's vocoder head IS Vocos, so a bundle artifact \
             legitimately carries Vocos-shaped tensors, but a bare `vocos` GGUF is a \
             standalone vocoder with no GPT backbone and no DVAE and could never \
             synthesise. Arch identity, not tensor shape, is what routes the loader \
             (FR-EX-08 — no silent partial load)."
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "chattts: GGUF is missing `vokra.model.arch` — this is not a Vokra-native \
             chattts GGUF (was it produced by `vokra-cli convert --model chattts`?). \
             Expected `{ARCH}`. Primary source: {PRIMARY_SOURCE_UPSTREAM_HF}"
        ))),
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`ChatTts::synthesize`] until the prep script and the `vokra.chattts.*` axis
/// group land.
///
/// Names both blockers, the three deferred modules, the on-disk census and all
/// three primary sources, so a reader diagnosing the gap has fully specified
/// places to walk. Mirror of the `maest` / `emotion2vec` / `panns`
/// loud-partial-message precedent (CLAUDE.md 教訓 (a)).
fn synthesize_loud_partial(census: &ChatTtsModuleCensus) -> VokraError {
    let missing = census.missing_synthesis_modules();
    let mut census_note = String::new();
    if missing.is_empty() {
        census_note.push_str(
            "all three synthesis modules matched a namespace probe on disk, so the blocker \
             here is the absent axis group rather than the artifact",
        );
    } else {
        census_note.push_str("no tensor on disk matched the namespace probe for: ");
        census_note.push_str(&missing.join(", "));
        census_note.push_str(
            " — which is expected if this is the GPT-only slice (the published \
             `vokra/chattts` repository was built from `asset/gpt/model.safetensors` alone, \
             with the remaining assets deferred to this wave), and is NOT on its own proof \
             of absence, because the flattening convention is not yet pinned",
        );
    }

    VokraError::UnsupportedOp(format!(
        "chattts synthesize (loud-partial): the full forward is deferred. ChatTTS needs \
         three modules in series — {gpt}, {dvae}, {vocos} — and none can be composed from \
         this artifact, because the converter stamps NO `{AXIS_GROUP_PREFIX}*` axis group at \
         all: layer count, hidden width, head count, vocab size, DVAE codebook layout and \
         the output sample rate are every one of them unrecoverable, and guessing any of \
         them yields a forward that is shape-valid, numerically wrong and silent. Two \
         things must land first: (1) the prep script `{PREP_SCRIPT_PATH}`, which merges the \
         four upstream torch pickles (GPT / DVAE / vocos / spk_stat) into one safetensors \
         and thereby pins the module-namespace convention a binder must walk; and (2) a \
         converter change stamping the `{AXIS_GROUP_PREFIX}*` group transcribed from the \
         upstream config. On-disk module census: gpt={c_gpt}, dvae={c_dvae}, \
         vocos={c_vocos}, spk_stat={c_spk}, total tensors={c_total} — {census_note}. \
         Primary sources: weights {hf}, reference code {code}, audit ticket {ticket}. \
         Runtime cannot fabricate a waveform (FR-EX-08 no silent partial output).",
        gpt = MODULE_LABEL_GPT,
        dvae = MODULE_LABEL_DVAE,
        vocos = MODULE_LABEL_VOCOS,
        c_gpt = census.gpt,
        c_dvae = census.dvae,
        c_vocos = census.vocos,
        c_spk = census.speaker_stats,
        c_total = census.total_tensors,
        hf = PRIMARY_SOURCE_UPSTREAM_HF,
        code = PRIMARY_SOURCE_CODE,
        ticket = PRIMARY_SOURCE_TICKET,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the ChatTTS runtime binder — contract-constant pins, metadata
    //! round-trip, module-census reporting, loud-error negative space on every
    //! stated blocker, and the T4 licence posture from **both** sides.
    //!
    //! # What is deliberately NOT asserted
    //!
    //! No expected numeric value appears anywhere below. The forward is
    //! loud-partial and no parity run against a real ChatTTS checkpoint has
    //! happened in this repository, so any number here would be fabrication
    //! wearing the costume of verification (CLAUDE.md 教訓 (a)).
    //!
    //! Nor do the fixtures claim to carry real upstream tensor names: they use
    //! the module-namespace spellings this module probes for, which is exactly
    //! what the census contract is about, and nothing more.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// One F32 tensor's worth of zero payload, sized for `dims`.
    fn zeros_for(dims: &[u64]) -> Vec<u8> {
        let elems: u64 = dims.iter().product();
        vec![0u8; elems as usize * 4]
    }

    /// Builds a GGUF carrying the converter's metadata stamps plus whichever
    /// tensor names the caller lists.
    ///
    /// `weight_license_class` of `None` omits the provenance stamp entirely, so
    /// the binder's fail-closed `Unknown` path is exercised.
    fn chattts_builder(
        weight_license_class: Option<LicenseClass>,
        tensor_names: &[&str],
    ) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
        b.add_string(chunks::KEY_PROVENANCE_SOURCE, "2Noise/ChatTTS test fixture");
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        for &name in tensor_names {
            let dims = vec![2u64, 3];
            let payload = zeros_for(&dims);
            b.add_tensor(name, GgmlType::F32, dims, payload)
                .expect("add_tensor");
        }
        b
    }

    /// The tensor names of a fixture that looks like the whole three-module
    /// bundle plus the speaker side asset.
    const FULL_BUNDLE_TENSORS: [&str; 5] = [
        "gpt.layers.0.attn.wq.weight",
        "gpt.embed_tokens.weight",
        "dvae.decoder.conv_in.weight",
        "vocos.backbone.convnext.0.dwconv.weight",
        "spk_stat",
    ];

    /// The tensor names of a fixture shaped like the artifact that was actually
    /// published: the GPT backbone alone.
    const GPT_ONLY_TENSORS: [&str; 2] = ["gpt.layers.0.attn.wq.weight", "gpt.embed_tokens.weight"];

    fn parse(b: &GgufBuilder) -> GgufFile {
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1 — Contract-constant pin (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "chattts", "chattts arch tag pin");
        assert_eq!(NAME, "chattts", "converter NAME equals ARCH for this model");
        assert_eq!(CATEGORY, "tts", "ChatTTS is a TTS release");
        assert_eq!(
            UPSTREAM_HF, "2Noise/ChatTTS",
            "upstream HF slug pin — note the capital N, the GitHub org is `2noise`"
        );
        assert_eq!(
            DEFAULT_LICENSE_SPDX, "cc-by-nc-4.0",
            "converter default weight SPDX pin (T4 tier)"
        );
        assert_eq!(GGUF_KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(
            GGUF_KEY_PROVENANCE_UPSTREAM_HF,
            "vokra.provenance.upstream_hf"
        );
        // The arch tag must not collide with any sibling TTS loader.
        for sibling in [
            "piper-plus-mb-istft-vits2",
            "kokoro-82m-istftnet",
            "cosyvoice2",
            "cosyvoice3",
            "chatterbox",
            "styletts2",
            "vibevoice",
            "dia",
            "zonos",
            "qwen3_tts",
            "sbv2",
            "voxcpm2",
            "melotts",
            "irodori-tts",
            "csm",
            "moshi",
            "vocos",
        ] {
            assert_ne!(ARCH, sibling, "chattts must not alias sibling `{sibling}`");
        }
    }

    // -----------------------------------------------------------------------
    // 2 — The SPDX the converter stamps really does resolve to NonCommercial
    // -----------------------------------------------------------------------

    #[test]
    fn default_spdx_resolves_to_non_commercial_and_is_research_gated() {
        let class = LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX);
        assert_eq!(
            class,
            LicenseClass::NonCommercial,
            "cc-by-nc-4.0 must resolve to NonCommercial (T4 tier)"
        );
        assert!(
            class.requires_research_flag(),
            "a NonCommercial weight must be research-gated, which is what makes the strict \
             refusal below correct rather than a defect"
        );
        assert!(
            !class.commercial_ok(),
            "NonCommercial must not read as commercially usable"
        );
    }

    // -----------------------------------------------------------------------
    // 3 — MANDATORY: arch absent -> loud ModelLoad
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = parse(&b);
        let Err(err) = ChatTts::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native chattts GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(
                    m.contains(ARCH),
                    "message must state the expected arch, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 4 — MANDATORY: foreign arch -> loud ModelLoad naming BOTH tags
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        // `vocos` is the sharpest confusable: ChatTTS's vocoder head IS Vocos,
        // so a bare Vocos GGUF has genuinely overlapping tensor shapes while
        // being a completely different loader identity.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "vocos");
        b.add_string(chunks::KEY_MODEL_NAME, "vocos");
        b.add_tensor("vocos.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = parse(&b);
        let Err(err) = ChatTts::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH tags, expected and actual.
                assert!(
                    m.contains("`vocos`"),
                    "message must name the ACTUAL arch found, got `{m}`"
                );
                assert!(
                    m.contains("`chattts`"),
                    "message must name the EXPECTED arch, got `{m}`"
                );
                // The neighbourhood enumeration gives the reader anchors.
                for sibling in ["cosyvoice2", "chatterbox", "kokoro-82m-istftnet", "moshi"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling `{sibling}` in the disambiguation, got `{m}`"
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
    // 5 — Empty tensor manifest fails loud (never an all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let b = chattts_builder(Some(LicenseClass::NonCommercial), &[]);
        let file = parse(&b);
        let Err(err) = ChatTts::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model chattts"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6 — MANDATORY: a synthetic GGUF with the right tensors binds
    // -----------------------------------------------------------------------

    #[test]
    fn full_bundle_gguf_binds_and_surfaces_metadata_and_census() {
        let b = chattts_builder(Some(LicenseClass::NonCommercial), &FULL_BUNDLE_TENSORS);
        let file = parse(&b);
        let m = ChatTts::from_gguf(&file).expect("a well-formed chattts GGUF must bind");

        // Metadata round-trip.
        assert_eq!(m.name(), Some(NAME));
        assert_eq!(m.category(), Some(CATEGORY));
        assert_eq!(m.upstream_hf(), Some(UPSTREAM_HF));
        assert_eq!(m.model_id(), Some(NAME));
        assert!(m.source().is_some(), "the converter stamps a source string");
        // The converter does not stamp attribution, so this is a recorded gap.
        assert_eq!(
            m.attribution(),
            None,
            "the converter's stamp_provenance call writes no attribution key"
        );

        // Licence surface.
        assert_eq!(m.weight_license(), LicenseClass::NonCommercial);

        // Manifest + census.
        assert_eq!(m.weights().tensor_count(), FULL_BUNDLE_TENSORS.len());
        let census = m.module_census();
        assert_eq!(census.gpt, 2, "two `gpt.` tensors in the fixture");
        assert_eq!(census.dvae, 1);
        assert_eq!(census.vocos, 1);
        assert_eq!(census.speaker_stats, 1);
        assert_eq!(census.total_tensors, FULL_BUNDLE_TENSORS.len());
        assert!(census.matched_any());
        assert!(
            census.synthesis_chain_complete(),
            "all three synthesis namespaces are present in this fixture"
        );
        assert!(census.missing_synthesis_modules().is_empty());

        // A present tensor resolves through the loud lookup, with its dims.
        let dims = m
            .weights()
            .require_tensor("gpt.embed_tokens.weight")
            .expect("a present tensor must resolve");
        assert_eq!(dims, [2usize, 3].as_slice());
        m.weights()
            .require_tensor_dims("gpt.embed_tokens.weight", &[2, 3])
            .expect("dims must match the fixture");
    }

    // -----------------------------------------------------------------------
    // 7 — MANDATORY: a missing tensor -> loud error naming it
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let b = chattts_builder(Some(LicenseClass::NonCommercial), &FULL_BUNDLE_TENSORS);
        let file = parse(&b);
        let m = ChatTts::from_gguf(&file).expect("bind");

        let wanted = "gpt.layers.99.attn.wo.weight";
        let Err(err) = m.weights().require_tensor(wanted) else {
            panic!("expected ModelLoad when a required tensor is absent");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(wanted),
                    "message must NAME the missing tensor, got `{msg}`"
                );
                // Nearest names on disk give the reader something to compare.
                assert!(
                    msg.contains("gpt.embed_tokens.weight"),
                    "message must list nearby names on disk, got `{msg}`"
                );
                assert!(
                    msg.contains(PREP_SCRIPT_PATH),
                    "message must name the un-written prep script, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // A dims mismatch names BOTH the expected and the actual shape.
        let Err(err) = m
            .weights()
            .require_tensor_dims("gpt.embed_tokens.weight", &[7, 11])
        else {
            panic!("expected ModelLoad on a dims mismatch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains("[2, 3]"), "must name the ACTUAL dims: {msg}");
                assert!(
                    msg.contains("[7, 11]"),
                    "must name the EXPECTED dims: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 8 — MANDATORY: the forward loud-partials, naming the missing primitives
    // -----------------------------------------------------------------------

    #[test]
    fn synthesize_loud_partials_naming_the_missing_primitives() {
        let b = chattts_builder(Some(LicenseClass::NonCommercial), &FULL_BUNDLE_TENSORS);
        let file = parse(&b);
        let m = ChatTts::from_gguf(&file).expect("bind");

        let Err(err) = m.synthesize("hello [laugh] world") else {
            panic!("synthesize must loud-partial — no real forward exists yet");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("chattts synthesize"),
                    "the surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // The three missing primitives, by name.
                assert!(
                    msg.contains(MODULE_LABEL_GPT),
                    "must name the GPT backbone: {msg}"
                );
                assert!(msg.contains(MODULE_LABEL_DVAE), "must name the DVAE: {msg}");
                assert!(
                    msg.contains(MODULE_LABEL_VOCOS),
                    "must name the Vocos head: {msg}"
                );

                // The two concrete blockers.
                assert!(
                    msg.contains(AXIS_GROUP_PREFIX),
                    "must name the absent axis group: {msg}"
                );
                assert!(
                    msg.contains(PREP_SCRIPT_PATH),
                    "must name the un-written prep script: {msg}"
                );

                // All three primary sources.
                for url in [
                    PRIMARY_SOURCE_UPSTREAM_HF,
                    PRIMARY_SOURCE_CODE,
                    PRIMARY_SOURCE_TICKET,
                ] {
                    assert!(msg.contains(url), "expected primary source `{url}`: {msg}");
                }

                assert!(
                    msg.contains("FR-EX-08"),
                    "expected the FR-EX-08 rationale for emitting no waveform: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 9 — The GPT-only artifact (what was actually published) is reported
    //     honestly rather than dying in a missing-tensor trail
    // -----------------------------------------------------------------------

    #[test]
    fn gpt_only_artifact_binds_and_the_forward_names_what_is_absent() {
        let b = chattts_builder(Some(LicenseClass::NonCommercial), &GPT_ONLY_TENSORS);
        let file = parse(&b);
        let m = ChatTts::from_gguf(&file).expect("the published GPT-only slice must still bind");

        let census = m.module_census();
        assert_eq!(census.gpt, GPT_ONLY_TENSORS.len());
        assert_eq!(census.dvae, 0);
        assert_eq!(census.vocos, 0);
        assert_eq!(census.speaker_stats, 0);
        assert!(census.matched_any());
        assert!(
            !census.synthesis_chain_complete(),
            "a GPT-only slice cannot complete the synthesis chain"
        );
        assert_eq!(
            census.missing_synthesis_modules(),
            vec![MODULE_LABEL_DVAE, MODULE_LABEL_VOCOS],
            "the census must list the absent stages in pipeline order"
        );

        let Err(err) = m.synthesize("hello") else {
            panic!("synthesize must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("gpt=2"),
                    "the census must be reported in the error: {msg}"
                );
                assert!(msg.contains("dvae=0"), "census must show the gap: {msg}");
                assert!(msg.contains("vocos=0"), "census must show the gap: {msg}");
                assert!(
                    msg.contains("asset/gpt/model.safetensors"),
                    "the message must explain the GPT-only publish, which is the likeliest \
                     reason a caller sees this: {msg}"
                );
                assert!(
                    msg.contains("NOT on its own proof of absence"),
                    "the message must not overclaim what a probe miss proves: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 10 — Licence posture, BOTH halves: strict refuses, research opt-in loads
    // -----------------------------------------------------------------------

    #[test]
    fn compliance_gate_refuses_non_commercial_under_strict_and_allows_research_opt_in() {
        let stamped = chattts_builder(Some(LicenseClass::NonCommercial), &FULL_BUNDLE_TENSORS)
            .to_bytes()
            .expect("serialize");

        // Half one: strict REFUSES. ChatTTS is T4, so this is the fail-closed
        // default working as intended, NOT a defect.
        let Err(err) = ChatTts::from_gguf_with_policy(&stamped, &CompliancePolicy::strict()) else {
            panic!("a cc-by-nc-4.0 artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for a NonCommercial weight, got {err:?}"
        );

        // Half two: an explicit research opt-in unlocks it (and emits the
        // mandatory research-only warning inside the gate).
        let research = CompliancePolicy::strict().with_research_license(true);
        let m = ChatTts::from_gguf_with_policy(&stamped, &research)
            .expect("the research opt-in must unlock a T4 weight");
        assert_eq!(m.weight_license(), LicenseClass::NonCommercial);
        assert_eq!(m.name(), Some(NAME));
    }

    // -----------------------------------------------------------------------
    // 11 — An unstamped artifact fails closed to Unknown, and is refused too
    // -----------------------------------------------------------------------

    #[test]
    fn unstamped_artifact_fails_closed_to_unknown() {
        let b = chattts_builder(None, &FULL_BUNDLE_TENSORS);

        // Un-gated bind still works and surfaces the fail-closed class.
        let file = parse(&b);
        let m = ChatTts::from_gguf(&file).expect("arch + manifest are the bind gates");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "a missing provenance stamp must fail closed to Unknown, never to Permissive"
        );

        // The gated route refuses it for the same reason a NonCommercial weight
        // is refused: Unknown also requires the research flag.
        let bytes = b.to_bytes().expect("serialize");
        let Err(err) = ChatTts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()) else {
            panic!("an unstamped artifact must be refused by the strict gate");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired for an Unknown weight class, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 12 — The gate must not mask an arch mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn arch_mismatch_is_reported_ahead_of_any_licence_verdict() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "cosyvoice2");
        b.add_string(chunks::KEY_MODEL_NAME, "cosyvoice2");
        b.add_tensor("cosyvoice2.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let foreign = b.to_bytes().expect("serialize");

        let Err(err) = ChatTts::from_gguf_with_policy(&foreign, &CompliancePolicy::strict()) else {
            panic!("a foreign arch must be refused");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains("`cosyvoice2`") && msg.contains("`chattts`"),
                "the arch mismatch is the actionable fact and must be reported ahead of a \
                 licence verdict about a model the caller never meant to load: {msg}"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }
}
