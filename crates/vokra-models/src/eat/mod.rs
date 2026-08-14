//! **EAT** (`cwx-worst-one/EAT`, MIT) — self-supervised audio encoder
//! runtime binder for the `eat` converter arch (Wave C2, 2026-08-15).
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
//! # Runtime layout (loud-partial per CLAUDE.md 教訓 (a))
//!
//! ```text
//! PCM (mono f32)
//!   -> log-mel spectrogram front-end                    ← **loud-partial**
//!        (the mel primitives DO exist in-repo —
//!         `vokra_ops::mel` / `kaldi_fbank` / `fused_logmel` —
//!         but the EAT converter stamps NO `vokra.frontend.*`
//!         chunk group, so n_fft / hop / n_mels / mel_norm /
//!         htk_mode are unknown. CLAUDE.md requires the
//!         frontend spec to be bit-exact and metadata-stamped;
//!         guessing it would silently desync every downstream
//!         embedding.)
//!   -> 2-D patch embedding (Conv2d patchifier)          ← **loud-partial**
//!        (`patch_embed.proj.*` per the converter's own
//!         state-dict sample. `vokra-ops` has no reusable
//!         ViT-style 2-D patchifier op today — its conv2d code
//!         is private to `denoise` / `conformer`.)
//!   -> ViT-style pre-norm Transformer encoder stack     ← **loud-partial**
//!        (`blocks.<i>.attn.qkv.*` per the converter's own
//!         state-dict sample. `vokra-ops` ships `conformer` /
//!         `ebranchformer` / `zipformer` — all ASR-specific,
//!         convolution-augmented, none with a prepended CLS
//!         token — but no plain ViT encoder block.)
//!   -> per-patch hidden states  ............... `Eat::encode`
//!   -> utterance-level pooling / CLS read-out  ← **loud-partial**
//!   -> utterance embedding  ................... `Eat::embed_utterance`
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`Eat::from_gguf`] with **strict** `vokra.model.arch == "eat"`
//!     verification. A foreign GGUF fails loud naming **both** the
//!     expected and the actual tag, and enumerates the sibling SSL
//!     audio-encoder neighbourhood so the reader knows which loader
//!     they wanted (FR-EX-08 — never a silent mis-route).
//!   - A `vokra.model.category` cross-check: **present-but-wrong**
//!     fails loud (the converter always stamps
//!     [`CATEGORY`] = `"audio-embedding"`, so a disagreeing value means
//!     a hand-edited or mis-produced artifact). Absent is tolerated so
//!     hand-assembled test fixtures need not stamp the full chunk set.
//!   - [`EatWeights::from_gguf`]: a real tensor manifest over the
//!     verbatim upstream state-dict names the converter passes through,
//!     with a non-empty gate, [`EatWeights::require_tensor`] /
//!     [`EatWeights::require_tensor_dims`] lookups that NAME the
//!     missing tensor (or BOTH the expected and actual dims), and
//!     **pure-observation** structure discovery
//!     ([`EatWeights::observed_block_count`] /
//!     [`EatWeights::has_patch_embed`] /
//!     [`EatWeights::count_with_prefix`]).
//!   - Weight-license surfacing that fail-closes to
//!     [`LicenseClass::Unknown`] when the stamp is absent.
//!
//! - **Loud-partial (this WP)**: [`Eat::encode`] and
//!   [`Eat::embed_utterance`] return [`VokraError::UnsupportedOp`]
//!   naming every missing piece, the primitives that would have to be
//!   added to `vokra-ops`, the un-stamped frontend spec, both primary
//!   sources, and the manifest facts actually observed on disk. **No
//!   fabricated hidden states or embeddings are ever emitted**
//!   (FR-EX-08 — no silent partial output).
//!
//! # No `vokra.eat.*` topology chunk group exists
//!
//! Unlike `wavlm_sv` (which reads a strict `vokra.wavlm.*` axis group),
//! the EAT converter stamps **no** topology axes at all — only
//! `vokra.model.*` and `vokra.provenance.*`. This binder therefore
//! reads no axis group and, critically, **invents no defaults**:
//! hidden width, depth, head count, patch size and mel-bin count are
//! simply *not known* to the runtime today. What structure can be
//! observed is observed from the tensor manifest itself
//! ([`EatWeights::observed_block_count`]), which is data actually on
//! disk rather than a transcribed constant. A future converter
//! revision that transcribes the upstream config should add a
//! `vokra.eat.*` group and this binder should grow a strict reader for
//! it, following [`crate::wavlm`].
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
//! [`DEFAULT_LICENSE_SPDX`] are **mirrors of the converter's
//! constants**, not imports — the same rule every sibling binder
//! follows so `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`. The
//! tests pin every mirrored value so a converter-side rename must land
//! here in the same commit or fail.
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
// EatWeights — the tensor manifest, with a non-empty gate, loud lookups,
// and pure-observation structure discovery.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an EAT GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step — a GGUF carrying zero tensors is refused rather than silently
/// binding an all-zero forward (FR-EX-08).
///
/// Under this landing the struct stores the tensor names and their
/// GGUF-side dims. The forward is deferred (see [`Eat::encode`]), so no
/// payload is dequantised yet; the follow-up wave sizes its dequant per
/// its kernel needs and uses [`require_tensor`](Self::require_tensor) /
/// [`require_tensor_dims`](Self::require_tensor_dims) to bind slots
/// loudly.
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
    /// a transcribed constant — the converter stamps no topology axes
    /// (see the module docstring), and inventing a depth would be
    /// fabrication. `None` is a normal outcome for a checkpoint whose
    /// state-dict uses a different prefix; callers must treat it as
    /// "unknown", never as "zero layers".
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
             checkpoint was flattened with a different prefix convention or the caller \
             is walking a manifest transcribed from a different EAT size point \
             (`eat-base` vs `eat-large`). Refusing to substitute a zero tensor \
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
                "eat: tensor `{name}` has dims {actual:?} but the caller expects \
                 {expected:?} — refusing to reshape or truncate silently (FR-EX-08). \
                 Either the GGUF was produced from a different EAT size point \
                 (`eat-base` vs `eat-large`) or the caller's transcribed axes disagree \
                 with the payload. The converter stamps no `vokra.eat.*` topology \
                 group, so the runtime has no stamped axes to arbitrate with. Primary \
                 sources: {UPSTREAM_URL} + {PRIMARY_SOURCE_PAPER}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Eat — the runtime binder handle.
// ---------------------------------------------------------------------------

/// EAT (`cwx-worst-one/EAT`, MIT) self-supervised audio-encoder runtime
/// binder.
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`encode`](Self::encode) for per-patch hidden states or
/// [`embed_utterance`](Self::embed_utterance) for the utterance-level
/// embedding. See the module doc for the implementation-status matrix
/// and the FR-EX-08 loud-error contract on the deferred forward.
///
/// This is a **feature extractor**: it exposes representations only.
/// The upstream downstream task heads (AudioSet tagging, ESC-50,
/// SPC-2) ship as separate fine-tunes and are not part of the
/// checkpoint this converter targets.
#[derive(Debug)]
pub struct Eat {
    weights: EatWeights,
    weight_license: LicenseClass,
}

impl Eat {
    /// Binds an EAT GGUF: verifies arch strictly, cross-checks the
    /// category stamp, discovers the tensor manifest, and surfaces the
    /// stamped weight-license class for the compliance-gate
    /// cross-checks (FR-CP-03).
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

        // 3. Tensor manifest with the non-emptiness gate.
        let weights = EatWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
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
            weights,
            weight_license,
        })
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

    /// Number of tensors bound from the GGUF. Diagnostic accessor — the
    /// follow-up forward wave uses it to size its expectations.
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

    /// Encodes a PCM waveform into the sequence of per-patch encoder
    /// hidden states, shaped `[n_patches][hidden]`.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. Three pieces are missing
    /// and none can be synthesized honestly today:
    ///
    /// 1. The **log-mel front-end spec**. The mel primitives exist
    ///    in-repo (`vokra_ops::mel` / `kaldi_fbank` / `fused_logmel`),
    ///    but the EAT converter stamps **no** `vokra.frontend.*` chunk
    ///    group, so n_fft / hop / n_mels / mel_norm / htk_mode are
    ///    unknown. CLAUDE.md requires the frontend spec to be bit-exact
    ///    and metadata-stamped precisely because librosa / torchaudio /
    ///    TF mel filterbanks are not bit-identical — guessing it would
    ///    silently desync every embedding this model ever produces.
    /// 2. A **2-D patch embedding (Conv2d patchifier)** primitive.
    ///    `vokra-ops` has no reusable ViT-style patchifier op; its
    ///    conv2d code is private to `denoise` / `conformer`.
    /// 3. A **ViT-style pre-norm Transformer encoder block**.
    ///    `vokra-ops` ships `conformer` / `ebranchformer` / `zipformer`
    ///    — all ASR-specific and convolution-augmented, none with a
    ///    prepended CLS token — but no plain ViT encoder stack.
    ///
    /// The message additionally reports what the manifest actually
    /// contains (tensor count, observed block count, patch-embed
    /// presence) so the follow-up wave can see whether the checkpoint
    /// in hand matches the naming the converter's tests exercise.
    /// **No fabricated hidden states are ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind explicitly so a future accidental removal of the
        // parameter cannot hide behind an unused-variable warning; the
        // real implementation will consume it.
        let _ = pcm;
        Err(forward_loud_partial("eat encode", None, &self.weights))
    }

    /// Encodes a PCM waveform into a single utterance-level embedding.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. In addition to the three
    /// pieces [`Self::encode`] lists, the utterance read-out itself is
    /// un-transcribed: EAT's utterance-level objective is defined over
    /// a CLS-token / pooled read-out whose exact form must be walked
    /// from the upstream repo, and the embedding width is not stamped
    /// anywhere in the GGUF. **No fabricated embedding is ever
    /// emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder forward plus the deferred utterance read-out.
    pub fn embed_utterance(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let _ = pcm;
        Err(forward_loud_partial(
            "eat embed_utterance",
            Some(
                "(iv) the utterance-level read-out — EAT's utterance objective is \
                 defined over a CLS-token / pooled read-out whose exact form must be \
                 walked from the upstream repo, and the embedding width is stamped \
                 nowhere in the GGUF, so no output length can even be shaped",
            ),
            &self.weights,
        ))
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Eat::encode`] and [`Eat::embed_utterance`] until the EAT forward
/// wave lands.
///
/// `surface` names the calling method; `extra_piece` appends a
/// surface-specific blocker (the utterance read-out) when present.
/// The message names the missing `vokra-ops` primitives, the un-stamped
/// frontend spec, both primary sources, and the manifest facts observed
/// on disk — the emotion2vec / wavlm / panns / redimnet loud-partial
/// precedent (CLAUDE.md 教訓 (a): "loud-partial は fake-complete より
/// honest").
fn forward_loud_partial(
    surface: &str,
    extra_piece: Option<&str>,
    weights: &EatWeights,
) -> VokraError {
    let blocks = weights.observed_block_count().map_or_else(
        || "unknown (no `blocks.<i>.` tensors on disk)".to_owned(),
        |n| n.to_string(),
    );
    let extra = extra_piece.unwrap_or("");
    VokraError::UnsupportedOp(format!(
        "{surface} (loud-partial): the EAT forward is deferred — three pieces must land \
         before real representations can be emitted. \
         (i) the log-mel FRONT-END SPEC: the mel primitives exist in-repo \
         (`vokra_ops::mel` / `vokra_ops::kaldi_fbank` / `vokra_ops::fused_logmel`), but \
         the EAT converter stamps NO `vokra.frontend.*` chunk group, so n_fft / hop / \
         n_mels / mel_norm / htk_mode are unknown; CLAUDE.md requires the frontend spec \
         to be bit-exact and metadata-stamped because librosa / torchaudio / TF mel \
         filterbanks are not bit-identical, so guessing it would silently desync every \
         embedding. \
         (ii) a 2-D PATCH EMBEDDING (Conv2d patchifier) primitive — `vokra-ops` exposes \
         no reusable ViT-style patchifier op today (its conv2d code is private to \
         `denoise` / `conformer`); the converter's own state-dict sample names \
         `{PATCH_EMBED_PREFIX}proj.weight`. \
         (iii) a ViT-style pre-norm TRANSFORMER ENCODER block — `vokra-ops` ships \
         `conformer` / `ebranchformer` / `zipformer`, all ASR-specific and \
         convolution-augmented with no prepended CLS token, but no plain ViT encoder \
         stack; the converter's own state-dict sample names \
         `{BLOCK_PREFIX}0.attn.qkv.weight`. \
         {extra} \
         No topology axes are available to arbitrate with: the converter stamps no \
         `vokra.eat.*` group, so hidden width / depth / head count / patch size are \
         simply not known to the runtime, and this binder refuses to invent them. \
         Observed on disk instead: tensor_count={count}, observed_block_count={blocks}, \
         has_patch_embed={patch}. \
         Primary sources: {UPSTREAM_URL} + {PRIMARY_SOURCE_PAPER} (arch={ARCH}, \
         name={NAME}, category={CATEGORY}). Runtime cannot fabricate hidden states or \
         an embedding (FR-EX-08 — no silent partial output).",
        count = weights.tensor_count(),
        patch = weights.has_patch_embed(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the EAT runtime binder — contract-constant pins,
    //! metadata round-trip, manifest observation, and negative-space
    //! round-trip on every loud gate.
    //!
    //! # What "round-trip" means here
    //!
    //! On real PCM this would be `encode(...)` returning real hidden
    //! states, but the EAT forward is deferred (see the module doc and
    //! [`Eat::encode`]). Fabricating a real-PCM output would violate
    //! CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
    //! honest"). What can honestly be tested:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` match the converter.
    //! 2. **Arch distinctness pin** — the tag collides with no sibling
    //!    SSL audio-encoder arch.
    //! 3. **Metadata round-trip** — a synthetic GGUF carrying the
    //!    converter's own sample tensor names binds, and the license
    //!    stamp surfaces (Permissive when stamped, Unknown when not).
    //! 4. **Manifest observation** — block count and patch-embed
    //!    presence are derived from what is on disk, and `None` is
    //!    produced (not a fabricated zero) for a foreign naming scheme.
    //! 5. **Loud negative space** — missing arch, foreign arch, wrong
    //!    category, empty manifest, missing tensor, wrong dims, and the
    //!    two loud-partial forward surfaces each fire at their
    //!    documented point in their documented variant.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a synthetic EAT GGUF carrying the arch / name / category
    /// stamps, an optional weight-license class, and the two
    /// representative tensor names the converter's own round-trip tests
    /// exercise (`patch_embed.proj.weight`, `blocks.<i>.attn.qkv.weight`)
    /// across `n_blocks` encoder blocks.
    fn eat_gguf(weight_license_class: Option<LicenseClass>, n_blocks: u32) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(GGUF_KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // The 2-D patch-embedding stem: a Conv2d weight is 4-D upstream;
        // dims here are placeholders (the converter stamps no axes, so
        // nothing in the binder validates them).
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
    // 3. Synthetic GGUF with the right tensors binds
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
    // 4. Manifest observation — never a fabricated topology
    // -----------------------------------------------------------------------

    #[test]
    fn observed_structure_is_none_for_a_foreign_naming_scheme() {
        // A checkpoint flattened under a different prefix convention
        // (fairseq / data2vec2 lineage) is NOT invalid — the binder must
        // report "unknown", not a fabricated zero-layer topology.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
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
    // 5. Loud negative space — arch metadata absent
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
    // 6. Loud negative space — foreign arch names BOTH tags
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
    // 7. Loud negative space — category stamped but wrong
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
    // 8. Loud negative space — empty tensor manifest
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
    // 9. Loud negative space — a missing tensor names the tensor
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
    // 10. Loud-partial — encode names every missing primitive
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partials_naming_the_missing_primitives() {
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

                // The three missing pieces, each by exact identifier.
                assert!(
                    msg.contains("vokra.frontend.*"),
                    "message must name the un-stamped frontend spec, got `{msg}`"
                );
                assert!(
                    msg.contains("vokra_ops::mel"),
                    "message must name the mel primitives that DO exist, got `{msg}`"
                );
                assert!(
                    msg.contains("2-D PATCH EMBEDDING") && msg.contains("Conv2d patchifier"),
                    "message must name the missing patchifier primitive, got `{msg}`"
                );
                assert!(
                    msg.contains("TRANSFORMER ENCODER") && msg.contains("ViT-style"),
                    "message must name the missing ViT encoder primitive, got `{msg}`"
                );
                assert!(
                    msg.contains("conformer")
                        && msg.contains("ebranchformer")
                        && msg.contains("zipformer"),
                    "message must name the encoder ops that DO exist so the reader can \
                     see why they do not substitute, got `{msg}`"
                );
                assert!(
                    msg.contains("vokra.eat.*"),
                    "message must state that no topology chunk group is stamped, got \
                     `{msg}`"
                );

                // Observed manifest facts, so the follow-up wave can see
                // whether the checkpoint in hand matches.
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

                // Primary sources + the FR-EX-08 rationale.
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
    // 11. Loud-partial — embed_utterance adds the read-out blocker
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
                // The utterance-specific fourth blocker.
                assert!(
                    msg.contains("utterance-level read-out"),
                    "message must name the deferred utterance read-out, got `{msg}`"
                );
                assert!(
                    msg.contains("CLS-token"),
                    "message must name the CLS-token read-out form, got `{msg}`"
                );
                assert!(
                    msg.contains("embedding width"),
                    "message must state that the output width is unknown, got `{msg}`"
                );
                // It still carries the three shared blockers.
                assert!(
                    msg.contains("2-D PATCH EMBEDDING") && msg.contains("TRANSFORMER ENCODER"),
                    "message must still name the shared encoder blockers, got `{msg}`"
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
