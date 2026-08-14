//! **NVIDIA Sortformer-Diar-4spk-v1** (`nvidia/diar_sortformer_4spk-v1`,
//! CC-BY-NC-4.0 — T4 tier) — end-to-end 4-speaker diarization runtime
//! binder for the `sortformer` converter arch (2026-08-14 audit
//! follow-up Wave 4).
//!
//! # Primary source
//!
//! - HF model card: <https://huggingface.co/nvidia/diar_sortformer_4spk-v1>
//! - NeMo reference:
//!   <https://github.com/NVIDIA/NeMo> (`nemo/collections/asr/models/sortformer_diar_models.py`
//!   + `nemo/collections/asr/modules/sortformer_modules.py`).
//! - Paper: Park et al., *"Sortformer: Seamless Integration of Speaker
//!   Diarization and ASR by Bridging Timestamps and Tokens"*, ICASSP
//!   2025 (arXiv:2409.06656).
//! - Weight license: **CC-BY-NC-4.0** (HF cardData primary source,
//!   `docs/license-audit.md` §3.1 row 449 ☑ Research-only 2026-08-04
//!   yousan, T4 tier — X-Codec-2 precedent).
//!
//! # Architecture (transcribed from primary sources)
//!
//! ```text
//! PCM (mono f32, 16 kHz)
//!   -> preprocessor (log-mel front-end, 128 filters — NEST-Fast-Conformer front-end)
//!   -> NEST FastConformer encoder                      ← **loud-partial**
//!        (18 layers, hidden=192, MHA n_head=8,
//!         Stacking subsampling factor=8;
//!         `vokra_ops::conformer::ConformerEncoder` with
//!         `ConvSubsampleKind::Stacking { factor: 8 }` covers the
//!         primitive but the tensor-name walk from upstream
//!         state_dict prefixes → `ConformerEncoder::forward`
//!         inputs has NOT been pinned pending the upstream
//!         tensor-name manifest fetch.)
//!   -> 18-layer plain Transformer encoder              ← **loud-partial**
//!        (hidden=192, MHA n_head=8, standard pre-norm block —
//!         composable from Vokra's existing softmax + GEMM +
//!         LayerNorm primitives; no new op needed, only the
//!         composition + tensor-name walk are pending.)
//!   -> 4-way per-frame diarization head                ← **loud-partial**
//!        (per-frame Linear(hidden, 4) + sigmoid, one column per
//!         target speaker; arrival-order sort loss is training-only
//!         per Park et al. so inference is per-frame threshold /
//!         argmax over the 4 sigmoid probabilities.)
//!   -> per-frame [0,1]^4 activity matrix
//!   -> contiguous-region grouping                      ← **loud-partial**
//!        (per-speaker activity → contiguous `SpeakerSegment`
//!         list; region-merging port is pending — a follow-up
//!         wave lands the Rust port of the NeMo `postprocessing.py`
//!         helpers alongside the tensor-name walk.)
//!   -> Vec<SpeakerSegment>
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`SortformerDiar::from_gguf`] with strict
//!   `vokra.model.arch == "sortformer"` validation +
//!   [`SortformerConfig::from_gguf`] with primary-source constant
//!   fallback (the Sortformer converter does NOT currently stamp the
//!   `vokra.sortformer.*` chunk group — only arch / name / category /
//!   upstream_hf / provenance — so a *strict* reader would refuse the
//!   already-published `huggingface.co/vokra/sortformer-diar-4spk-v1`
//!   GGUF. Primary source is well-established (HF card + NeMo YAML +
//!   paper) so fallback does not fabricate axes; a future converter
//!   sub-wave that starts stamping the chunk group upgrades this to
//!   real-stamped reads seamlessly — mirror of
//!   `PyanNetConfig::from_gguf` pattern), [`SortformerWeights::from_gguf`]
//!   with a floor of non-empty tensor count enforced loud (a GGUF that
//!   carries zero tensors is refused rather than silently running an
//!   all-zero forward — FR-EX-08), [`SpeakerSegment`] public surface
//!   pin (fields match task-hint spelling and are distinct from sibling
//!   `pyannote::rttm::DiarizationSegment { start_s, duration_s,
//!   speaker_id }` which uses `duration_s` semantics), and
//!   weight-license class surfacing (defaults to
//!   [`LicenseClass::NonCommercial`] per the converter's stamped
//!   `cc-by-nc-4.0` — T4 tier fail-closed at the runtime compliance
//!   gate M2-13).
//! - **Loud-partial (this WP)**: [`SortformerDiar::diarize`] returns
//!   [`VokraError::UnsupportedOp`] naming **both** deferred pieces:
//!   1. the composition of `vokra_ops::conformer::ConformerEncoder`
//!      (primitive **exists** — it covers the NEST FastConformer via
//!      `Stacking { factor: 8 }`) + an 18-layer plain Transformer
//!      (hidden=192, composable from existing softmax + GEMM +
//!      LayerNorm — no new op needed) + the per-frame 4-sigmoid
//!      diarization head, PLUS the tensor-name walk from the upstream
//!      `nvidia/diar_sortformer_4spk-v1` state_dict prefixes to the
//!      encoder's `ConformerEncoder::forward` inputs (pending the
//!      upstream tensor-name manifest fetch);
//!   2. the contiguous-region grouping from per-frame 4-sigmoid
//!      logits to a `Vec<SpeakerSegment>` list (Rust port of the NeMo
//!      inference-side region-merging helpers).
//!
//! The error names the primary source URLs (HF card + NeMo repo +
//! paper) so a reader diagnosing this gap has exactly three places to
//! walk — mirror of `Mt3::transcribe` / `BeatThis::analyze` / `RMVPE`
//! Wave 3 loud-partial-message precedent.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 loud-partial precedent, CLAUDE.md 教訓 (a)): the
//! surrounding scaffold + `from_gguf` chunk-group validation +
//! `SpeakerSegment` surface + FR-EX-08 loud-fails land today so a
//! follow-up wave can flip the switch by (i) landing the tensor-name
//! walk against a real Sortformer state_dict via
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12
//! sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) + (ii) wiring the encoder + Transformer
//! + 4-sigmoid head composition + (iii) porting the region-merging
//!   helpers. The primitive `vokra_ops::conformer::ConformerEncoder`
//!   already exists so the follow-up wave is composition + tensor walk
//!   only, NOT a greenfield kernel.
//!
//! # `vokra.sortformer.*` chunk group (read here — fallback-friendly)
//!
//! The Sortformer converter
//! (`crates/vokra-convert/src/models/sortformer_diar_4spk_v1.rs`)
//! currently stamps only the arch / name / category / upstream_hf /
//! provenance chunks. The topology chunk group is READ by this binder
//! but any absent key falls back to the primary-source constant so an
//! already-published GGUF loads correctly. A future converter sub-wave
//! that adds `vokra.sortformer.*` stamps will override the fallback
//! automatically per-key.
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"sortformer"`).
//!   Deliberately distinct from every sibling ASR arch that also
//!   consumes a FastConformer encoder (`parakeet-tdt` / `parakeet-ctc` /
//!   `parakeet-unified` / `canary`) — silently sharing would misroute a
//!   diarization checkpoint onto an ASR loader (FR-EX-08). A future
//!   Sortformer-8spk / Sortformer-16spk stays classifier-compatible
//!   under the same arch tag.
//! - `vokra.model.name` (`String`): `"sortformer-diar-4spk-v1"` — the
//!   versioned identifier that matches the `huggingface.co/vokra/`
//!   publish slug.
//! - `vokra.sortformer.{d_model, n_heads, num_nest_layers,
//!   num_transformer_layers, num_speakers, subsampling_factor}` (`u32`
//!   each): the composite topology axes. Fallback constants
//!   transcribed from the HF card + paper + NeMo YAML (see the
//!   `DEFAULT_*` constants for the primary-source anchors).
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03 / M2-13) can classify the
//!   artifact without re-inspecting the safetensors provenance. The
//!   Sortformer converter stamps `NonCommercial` by default per the HF
//!   card's `license: cc-by-nc-4.0` — a caller who legitimately holds
//!   the weight under a distinct SPDX overrides at
//!   `vokra-cli convert --license <spdx>` and the stamped class
//!   re-derives via `LicenseClass::from_license_str`.
//!
//! # Cross-crate constant duplication (mirror of the converter's
//! [`ARCH`] / [`NAME`] / topology keys) — same rule the sibling BF16
//! pass-through binders (`pyannote` / `snac` / `hifigan` / `beat_this` /
//! `mt3`) use so `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Sortformer is distributed as `.nemo` (NGC) or safetensors (HF); this
//! runtime **never** touches ONNX (FR-LD-05 / NFR-DS-02). The `.nemo`
//! → safetensors bridge lives in `tools/parity/nemo_pt_to_safetensors.py`
//! (an offline uv-managed Python 3.12 sidecar — not part of the
//! runtime), mirroring the Parakeet / Canary / Parakeet-CTC bridge
//! pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/sortformer_diar_4spk_v1.rs`. See
// module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model sortformer-diar-4spk-v1`.
///
/// Deliberately distinct from every sibling FastConformer-based ASR
/// arch tag (`parakeet-tdt` / `parakeet-ctc` / `parakeet-unified` /
/// `canary`). Silently sharing would let runtime dispatch bind a
/// FastConformer-ASR loader over a Sortformer diarization checkpoint
/// (or vice versa): an ASR loader would look for `joint.*` /
/// `decoder.*` tensors the diarization-head-only Sortformer never
/// emits, and Sortformer's per-frame 4-sigmoid diarization head has
/// no ASR-side analog — FR-EX-08 forbids the silent-wrong shape
/// mismatch. Version-neutral (a future Sortformer-8spk / -16spk keeps
/// the tag; only [`NAME`] is versioned).
pub const ARCH: &str = "sortformer";

/// Expected `vokra.model.name` value — matches the
/// `huggingface.co/vokra/sortformer-diar-4spk-v1` publish slug.
pub const NAME: &str = "sortformer-diar-4spk-v1";

/// `vokra.sortformer.d_model` — plain Transformer hidden dim (also
/// the NEST FastConformer encoder hidden dim, per Park et al. and the
/// NeMo YAML). Primary-source default: 192.
pub const GGUF_KEY_D_MODEL: &str = "vokra.sortformer.d_model";
/// `vokra.sortformer.n_heads` — multi-head attention head count for
/// both the NEST encoder and the plain Transformer post-encoder.
/// Primary-source default: 8.
pub const GGUF_KEY_N_HEADS: &str = "vokra.sortformer.n_heads";
/// `vokra.sortformer.num_nest_layers` — NEST FastConformer encoder
/// depth (Park et al.: 18).
pub const GGUF_KEY_NUM_NEST_LAYERS: &str = "vokra.sortformer.num_nest_layers";
/// `vokra.sortformer.num_transformer_layers` — plain Transformer
/// post-encoder depth (Park et al.: 18).
pub const GGUF_KEY_NUM_TRANSFORMER_LAYERS: &str = "vokra.sortformer.num_transformer_layers";
/// `vokra.sortformer.num_speakers` — output column count of the
/// per-frame sigmoid diarization head. Fixed at 4 for the v1 release
/// (`-diar-4spk-v1` in the model slug); a future 8-speaker release
/// would use the same arch tag with a distinct name + a bumped
/// value here.
pub const GGUF_KEY_NUM_SPEAKERS: &str = "vokra.sortformer.num_speakers";
/// `vokra.sortformer.subsampling_factor` — NEST Fast-Conformer
/// subsampling factor (8× per the NeMo YAML —
/// `ConvSubsampleKind::Stacking { factor: 8 }`).
pub const GGUF_KEY_SUBSAMPLING_FACTOR: &str = "vokra.sortformer.subsampling_factor";

// Primary-source constants transcribed from the HF model card
// (huggingface.co/nvidia/diar_sortformer_4spk-v1), the NeMo repository
// (github.com/NVIDIA/NeMo `nemo/collections/asr/models/sortformer_diar_models.py`
// + `sortformer_modules.py`), and the paper (arXiv:2409.06656 §3.2
// "Model Architecture", fetched 2026-08-14 — CLAUDE.md「ハルシネーション
// 厳禁」).

/// NEST FastConformer + plain Transformer hidden dim. Primary source:
/// HF card + NeMo config + Park et al. §3.2.
pub const DEFAULT_D_MODEL: u32 = 192;
/// Multi-head attention head count. Primary source: NeMo config.
pub const DEFAULT_N_HEADS: u32 = 8;
/// NEST FastConformer encoder depth. Primary source: HF card
/// ("18-layer NEST encoder based on Fast-Conformer") + Park et al.
/// §3.2.
pub const DEFAULT_NUM_NEST_LAYERS: u32 = 18;
/// Plain Transformer post-encoder depth. Primary source: HF card
/// ("18-layer Transformer encoder") + Park et al. §3.2.
pub const DEFAULT_NUM_TRANSFORMER_LAYERS: u32 = 18;
/// Fixed at 4 for this v1 release — the `-diar-4spk-v1` in the model
/// slug.
pub const DEFAULT_NUM_SPEAKERS: u32 = 4;
/// NEST FastConformer subsampling factor (8× per the NeMo YAML —
/// matches `vokra_ops::conformer::ConvSubsampleKind::Stacking { factor: 8 }`).
pub const DEFAULT_SUBSAMPLING_FACTOR: u32 = 8;

/// Primary-source anchor for the HF model card. Cited in the
/// loud-partial error so a reader diagnosing the gap knows the
/// definitive artifact source.
const PRIMARY_SOURCE_HF_CARD: &str = "huggingface.co/nvidia/diar_sortformer_4spk-v1";
/// Primary-source anchor for the NeMo reference implementation. Cited
/// in the loud-partial error so a reader diagnosing the gap knows the
/// tensor-name walk anchor.
const PRIMARY_SOURCE_NEMO_REPO: &str = "github.com/NVIDIA/NeMo";
/// Paper anchor (Park et al. ICASSP 2025) — cited alongside the two
/// artefact URLs so a reader has the theoretical context as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2409.06656";

// ---------------------------------------------------------------------------
// SortformerConfig — the composite topology axes read from the
// `vokra.sortformer.*` chunk group, with primary-source constant
// fallback (the Sortformer converter does not currently stamp this
// chunk group — the fallback is honest because the primary source is
// well-established; a future converter sub-wave that adds the stamps
// upgrades this reader to real-stamped reads seamlessly). Mirror of
// [`crate::pyannote::PyanNetConfig::from_gguf`].
// ---------------------------------------------------------------------------

/// Sortformer hyperparameters as they ride the `vokra.sortformer.*`
/// chunk group.
///
/// [`from_gguf`](Self::from_gguf) reads the chunk with primary-source
/// constant fallback per key — a GGUF that never carried the chunk
/// still loads with the upstream defaults transcribed from the HF card
/// + NeMo YAML + paper. Every numeric axis is `u32` in the GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortformerConfig {
    /// NEST FastConformer + plain Transformer hidden dim
    /// (default 192, HF card + NeMo YAML + Park et al. §3.2).
    pub d_model: u32,
    /// Multi-head attention head count (default 8, NeMo YAML).
    pub n_heads: u32,
    /// NEST FastConformer encoder depth (default 18, HF card + paper).
    pub num_nest_layers: u32,
    /// Plain Transformer post-encoder depth (default 18, HF card +
    /// paper).
    pub num_transformer_layers: u32,
    /// Diarization-head output column count / number of target
    /// speakers (fixed 4 for v1, model slug `-4spk-v1`).
    pub num_speakers: u32,
    /// NEST FastConformer subsampling factor (default 8, matches
    /// `vokra_ops::conformer::ConvSubsampleKind::Stacking { factor: 8 }`).
    pub subsampling_factor: u32,
}

impl Default for SortformerConfig {
    /// The primary-source-transcribed Sortformer-diar-4spk-v1 axes.
    /// Used by [`Self::from_gguf`] as the per-key fallback.
    fn default() -> Self {
        Self {
            d_model: DEFAULT_D_MODEL,
            n_heads: DEFAULT_N_HEADS,
            num_nest_layers: DEFAULT_NUM_NEST_LAYERS,
            num_transformer_layers: DEFAULT_NUM_TRANSFORMER_LAYERS,
            num_speakers: DEFAULT_NUM_SPEAKERS,
            subsampling_factor: DEFAULT_SUBSAMPLING_FACTOR,
        }
    }
}

impl SortformerConfig {
    /// The primary-source-transcribed Sortformer-diar-4spk-v1 axes as
    /// a `const` — an alias for the [`Default`] impl useful in
    /// contexts that need a `const` (e.g. `const` initializers, doc
    /// examples). Never used silently by the loader; every axis
    /// passes through [`Self::from_gguf`]'s per-key fallback path.
    #[must_use]
    pub const fn v1_default() -> Self {
        Self {
            d_model: DEFAULT_D_MODEL,
            n_heads: DEFAULT_N_HEADS,
            num_nest_layers: DEFAULT_NUM_NEST_LAYERS,
            num_transformer_layers: DEFAULT_NUM_TRANSFORMER_LAYERS,
            num_speakers: DEFAULT_NUM_SPEAKERS,
            subsampling_factor: DEFAULT_SUBSAMPLING_FACTOR,
        }
    }

    /// Reads every `vokra.sortformer.*` chunk from `gguf`, falling
    /// back to the primary-source [`Default`] constants per absent
    /// key.
    ///
    /// The Sortformer converter does not currently stamp this chunk
    /// group (only arch / name / category / upstream_hf / provenance),
    /// so on an already-published GGUF every axis falls through to
    /// its primary-source default. A future converter sub-wave that
    /// adds the stamps upgrades this reader to real-stamped reads
    /// per-key with no runtime code change.
    ///
    /// Mirror of [`crate::pyannote::PyanNetConfig::from_gguf`] — the
    /// same fallback pattern used for pyannote because the pyannote
    /// converter's chunk group is likewise post-launch. Distinct from
    /// [`crate::mt3::Mt3Config::from_gguf`] which is strict (fails
    /// loud on any missing chunk) because MT3's upstream release ships
    /// no first-class config anywhere and fallback would fabricate.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let default = Self::default();
        Self {
            d_model: gguf
                .get(GGUF_KEY_D_MODEL)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.d_model),
            n_heads: gguf
                .get(GGUF_KEY_N_HEADS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.n_heads),
            num_nest_layers: gguf
                .get(GGUF_KEY_NUM_NEST_LAYERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.num_nest_layers),
            num_transformer_layers: gguf
                .get(GGUF_KEY_NUM_TRANSFORMER_LAYERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.num_transformer_layers),
            num_speakers: gguf
                .get(GGUF_KEY_NUM_SPEAKERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.num_speakers),
            subsampling_factor: gguf
                .get(GGUF_KEY_SUBSAMPLING_FACTOR)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.subsampling_factor),
        }
    }
}

// ---------------------------------------------------------------------------
// SortformerWeights — bound the tensor manifest with a non-emptiness
// gate. Under the loud-partial WP the weights are counted but the
// NEST FastConformer + plain Transformer + 4-sigmoid head forward is
// deferred. Mirror of `Mt3Weights` / `BeatThisWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a Sortformer GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid Sortformer checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave sizes its
/// dequant per its kernel needs — today only the count + names are
/// consumed so a future
/// `SortformerWeights::bind_conformer_encoder_weights` tensor walk
/// can find its inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct SortformerWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// NEST-encoder-forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl SortformerWeights {
    /// Scans `gguf` for the Sortformer state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid Sortformer checkpoint).
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
            return Err(VokraError::ModelLoad(
                "sortformer: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model sortformer-diar-4spk-v1` \
                 against an upstream safetensors checkpoint (either the direct \
                 `nvidia/diar_sortformer_4spk-v1` safetensors on HF or the `.nemo` \
                 tarball flattened via `tools/parity/nemo_pt_to_safetensors.py`)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Load-time shape gate — validates that at least one bound
    /// tensor has an axis matching `config.d_model`. Under the
    /// current landing this is a **soft** gate (mismatch is silently
    /// ignored) because the NEST FastConformer + plain Transformer +
    /// head tensor-name walk has not yet been pinned pending the
    /// upstream tensor-name manifest fetch — a hard shape assertion
    /// today would fail against every legitimate future manifest.
    ///
    /// The follow-up wave will replace this soft accessor with a
    /// hard pin against the primary-source-verified tensor-name walk
    /// (mirror of `pyannote::PyanNetWeights::verify_core_shapes`).
    ///
    /// Kept as a `#[must_use]` accessor so the read is deliberate.
    #[must_use]
    pub fn matches_config(&self, config: &SortformerConfig) -> bool {
        let d = config.d_model as usize;
        self.tensors.iter().any(|(_, dims)| dims.contains(&d))
    }
}

// ---------------------------------------------------------------------------
// SpeakerSegment — the public output surface for `SortformerDiar::diarize`
// once the tensor-name walk + head composition + region-merging port
// land. Defined here per the task hint ("Ship `SortformerDiar::diarize(pcm)
// -> Result<Vec<SpeakerSegment>>`") — pinned as the surface a future
// forward wave binds against.
//
// Note: distinct semantic from the sibling
// `crate::pyannote::rttm::DiarizationSegment { start_s, duration_s,
// speaker_id }` which encodes span end as `duration_s` (RTTM
// convention). Sortformer's task-hinted surface uses `end_s` as an
// absolute timestamp — silently unifying would break the pyannote RTTM
// contract *and* the task-listed field spelling here.
// ---------------------------------------------------------------------------

/// A single-speaker time span emitted by Sortformer's diarization
/// head after per-frame threshold decoding + contiguous-region
/// grouping.
///
/// Fields match the task-hint spelling — pinned as a **surface pin**:
/// a rename or field-shape change would need to land here in the same
/// commit or fail the surface pin test at the bottom of this module.
///
/// A single utterance emits multiple [`SpeakerSegment`] entries, one
/// per contiguous activity region per target speaker; the caller can
/// filter / merge / render to RTTM (`pyannote::rttm`) or JSON as
/// needed.
///
/// `speaker_id` is a **zero-based dense index** into the 4 target
/// speakers Sortformer resolves (`num_speakers == 4` for the v1
/// release). Sortformer's arrival-order sort loss is a *training-side*
/// device (Park et al. §3.3) — at inference time the four output
/// columns have no per-column identity; downstream clustering /
/// speaker verification is the caller's responsibility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerSegment {
    /// Zero-based index into Sortformer's 4 output columns
    /// (0..num_speakers).
    pub speaker_id: usize,
    /// Absolute segment start time (seconds from utterance start).
    pub start_s: f32,
    /// Absolute segment end time (seconds from utterance start).
    /// Distinct semantics from `pyannote::rttm::DiarizationSegment`
    /// which uses `duration_s` — that pairs with the RTTM writer's
    /// on-wire format, but the task hint here calls for absolute
    /// `end_s`. `duration = end_s - start_s` at the callsite.
    pub end_s: f32,
}

// ---------------------------------------------------------------------------
// SortformerDiar — the runtime binder handle
// ---------------------------------------------------------------------------

/// NVIDIA Sortformer end-to-end 4-speaker diarization runtime binder
/// (`nvidia/diar_sortformer_4spk-v1`, CC-BY-NC-4.0 T4 tier).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`diarize`](Self::diarize) on a PCM buffer to obtain a
/// `Vec<SpeakerSegment>`. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract
/// on the NEST FastConformer + plain Transformer + 4-sigmoid head +
/// region-merging composition.
#[derive(Debug)]
pub struct SortformerDiar {
    config: SortformerConfig,
    // The bound weights are held (real, counted) but the encoder +
    // Transformer + head + region-merging composition is a follow-up
    // wave; the field is deliberately `#[allow(dead_code)]` until the
    // composition lands so a reader is not misled by an unused field.
    // Same posture as RMVPE / pyannote / mt3 / beat_this.
    #[allow(dead_code)]
    weights: SortformerWeights,
    weight_license: LicenseClass,
}

impl SortformerDiar {
    /// Binds a Sortformer GGUF: validates arch, reads the topology
    /// chunk group (with primary-source constant fallback per key),
    /// discovers tensors, and surfaces the stamped weight-license
    /// class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"sortformer"` (a `parakeet-tdt` / `parakeet-ctc` /
    ///   `parakeet-unified` / `canary` GGUF handed to us by mistake
    ///   fails with a clear message instead of a downstream missing-
    ///   tensor — every sibling arch shares the FastConformer encoder
    ///   primitive but has a different terminal head, so the runtime
    ///   dispatch discipline forbids silent aliasing).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero
    ///   tensors ([`SortformerWeights::from_gguf`] refuses to bind an
    ///   all-zero forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "sortformer: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model sortformer-diar-4spk-v1`? Note \
                     that the sibling FastConformer-encoder-based arches — `parakeet-tdt` \
                     (English streaming ASR + TDT joint head), `parakeet-ctc` (English ASR \
                     + Linear CTC head), `parakeet-unified` (unified ASR + AST), `canary` \
                     (25-language ASR / AST + Transformer AED decoder) — all share the \
                     FastConformer encoder primitive but have completely different terminal \
                     heads; Sortformer's per-frame 4-sigmoid diarization head has no ASR-side \
                     analog and silently aliasing arch would misroute the runtime dispatch, \
                     FR-EX-08)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "sortformer: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native sortformer GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Topology axes from the `vokra.sortformer.*` chunk group
        //    (fallback-friendly — see the module doc for the
        //    Sortformer converter's stamp posture).
        let config = SortformerConfig::from_gguf(file);

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = SortformerWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The Sortformer
        //    converter defaults to `NonCommercial` per the HF card's
        //    `license: cc-by-nc-4.0`; a caller override at
        //    `--license <spdx>` re-derives the class. Missing
        //    provenance falls back to `Unknown` which is fail-closed
        //    at the M2-13 compliance gate — same posture as MT3.
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

    /// The bound topology axes (from `vokra.sortformer.*` chunk group
    /// with primary-source constant fallback).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &SortformerConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The Sortformer
    /// converter stamps `NonCommercial` by default per the HF card's
    /// `license: cc-by-nc-4.0` (T4 tier — fail-closed at the M2-13
    /// compliance gate; owner must pass `--allow-noncommercial` to
    /// publish and the runtime refuses commercial-mode load). A GGUF
    /// missing the stamp reads back as [`LicenseClass::Unknown`]
    /// which is also fail-closed.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Diarizes a mono PCM buffer (16 kHz per the NEST FastConformer
    /// front-end spec) into a `Vec<SpeakerSegment>`.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — Sortformer's inference
    /// path requires **two** deferred pieces:
    ///
    /// 1. **Encoder + head composition + tensor walk**:
    ///    `vokra_ops::conformer::ConformerEncoder` with
    ///    `ConvSubsampleKind::Stacking { factor: 8 }` already exists
    ///    as the NEST FastConformer primitive, but the tensor-name
    ///    walk from the upstream `nvidia/diar_sortformer_4spk-v1`
    ///    state_dict prefixes to `ConformerEncoder::forward` inputs
    ///    has NOT been pinned pending the upstream tensor-name
    ///    manifest fetch. The 18-layer plain Transformer post-encoder
    ///    (hidden=192) is composable from Vokra's existing softmax +
    ///    GEMM + LayerNorm primitives (no new op needed), and the
    ///    per-frame Linear(hidden, 4) + sigmoid diarization head is
    ///    trivial (one Linear + elementwise sigmoid).
    /// 2. **Region-merging**: mapping per-frame `[0,1]^4` activity to
    ///    contiguous `SpeakerSegment` entries — a Rust port of the
    ///    NeMo `postprocessing.py` region-merging helpers. Sortformer's
    ///    arrival-order sort loss is training-only per Park et al.
    ///    §3.3, so inference is per-frame threshold / argmax over the
    ///    4 sigmoid probabilities → contiguous span extraction, but
    ///    the port itself is deferred.
    ///
    /// The error names **three** primary source URLs (HF card + NeMo
    /// repo + paper) so a reader diagnosing this gap has exactly
    /// three places to walk. **No fabricated segment stream is ever
    /// emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred encoder-head composition + region-merging port.
    pub fn diarize(&self, pcm: &[f32]) -> Result<Vec<SpeakerSegment>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future
        // real implementation will consume it.
        let _ = pcm;
        Err(diarize_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`SortformerDiar::diarize`] until the tensor-name walk +
/// encoder-head composition + region-merging port land.
///
/// Names **all three** primary source URLs (HF card + NeMo repo +
/// paper) so a reader diagnosing the gap has exactly three places to
/// walk. Mirrors the MT3 / beat_this / RMVPE / pyannote / snac /
/// hifigan Wave 3 loud-partial-message precedent — CLAUDE.md 教訓
/// (a).
fn diarize_forward_loud_partial(cfg: &SortformerConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "sortformer diarize: NEST FastConformer encoder + plain Transformer + \
         4-sigmoid diarization head composition + region-merging port pending. \
         The FastConformer primitive `vokra_ops::conformer::ConformerEncoder` \
         with `ConvSubsampleKind::Stacking {{ factor: {subsample} }}` already \
         exists — what is missing is (a) the tensor-name walk from the upstream \
         `nvidia/diar_sortformer_4spk-v1` state_dict prefixes to \
         `ConformerEncoder::forward` inputs (pending the upstream tensor-name \
         manifest fetch — same posture as pyannote / Charsiu real-weight bind), \
         (b) the {n_transformer}-layer plain Transformer post-encoder composition \
         (hidden={d_model} — composable from existing softmax + GEMM + LayerNorm \
         primitives, no new op needed), (c) the per-frame Linear({d_model}, \
         {num_speakers}) + sigmoid diarization head (trivial), and (d) the \
         region-merging port that maps per-frame [0,1]^{num_speakers} activity to \
         contiguous `SpeakerSegment` entries (Rust port of the NeMo `postprocessing.py` \
         helpers — Sortformer's arrival-order sort loss is training-only per Park \
         et al. §3.3, so inference is per-frame threshold over the {num_speakers} \
         sigmoid probabilities → contiguous span extraction). Config: \
         num_nest_layers={n_nest}, num_transformer_layers={n_transformer}, \
         d_model={d_model}, n_heads={n_heads}, num_speakers={num_speakers}, \
         subsampling_factor={subsample}. Primary sources: {hf_card} + {nemo_repo} + \
         {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
         より honest') — no silent fabricated SpeakerSegment stream ever emitted \
         (FR-EX-08).",
        n_nest = cfg.num_nest_layers,
        n_transformer = cfg.num_transformer_layers,
        d_model = cfg.d_model,
        n_heads = cfg.n_heads,
        num_speakers = cfg.num_speakers,
        subsample = cfg.subsampling_factor,
        hf_card = PRIMARY_SOURCE_HF_CARD,
        nemo_repo = PRIMARY_SOURCE_NEMO_REPO,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the Sortformer runtime binder — round-trip on the
    //! topology chunk group + negative-space round-trip on the
    //! loud-partial gates + `SpeakerSegment` surface pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would
    //! be `diarize(...)` returning real speaker-segment streams, but
    //! the tensor-name walk + encoder-head composition + region-merging
    //! port are all deferred (see the module doc +
    //! [`SortformerDiar::diarize`] rustdoc). Fabricating a real-PCM
    //! output would violate CLAUDE.md 教訓 (a) ("loud-partial は
    //! fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config round-trip**: `from_gguf` reads every axis
    //!    stamped by the converter, and falls back cleanly to the
    //!    primary-source defaults for any absent key.
    //! 2. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / empty tensor list /
    //!    unsupported forward surface) fires at its documented
    //!    surface point, in the documented error variant.
    //! 3. **SpeakerSegment surface pin**: the field shape matches the
    //!    task-hint spelling.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a Sortformer GGUF carrying the arch tag + name + one
    /// representative encoder tensor whose outer dim matches
    /// `d_model`. The topology chunk group is optionally stamped
    /// (`stamp_topology = true`) — when omitted the runtime binder
    /// falls back to the primary-source defaults per key.
    ///
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn sortformer_gguf(
        cfg: SortformerConfig,
        stamp_topology: bool,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        if stamp_topology {
            b.add_u32(GGUF_KEY_D_MODEL, cfg.d_model);
            b.add_u32(GGUF_KEY_N_HEADS, cfg.n_heads);
            b.add_u32(GGUF_KEY_NUM_NEST_LAYERS, cfg.num_nest_layers);
            b.add_u32(GGUF_KEY_NUM_TRANSFORMER_LAYERS, cfg.num_transformer_layers);
            b.add_u32(GGUF_KEY_NUM_SPEAKERS, cfg.num_speakers);
            b.add_u32(GGUF_KEY_SUBSAMPLING_FACTOR, cfg.subsampling_factor);
        }
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative encoder tensor so the non-emptiness
        // gate passes and the shape-consistency accessor has
        // something to walk. The `d_model` dim is deliberately at
        // axis 0 so `matches_config` returns true.
        let d = cfg.d_model as u64;
        b.add_tensor(
            "encoder.layers.0.self_attn.linear_q.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. SortformerConfig default matches primary-source Sortformer-4spk-v1
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_config_default_matches_primary_source_v1_axes() {
        // Pin the primary-source axes transcribed from the HF card +
        // NeMo YAML + Park et al. §3.2. A rename or axis-value change
        // would land here in the same commit or fail this test.
        let cfg = SortformerConfig::v1_default();
        assert_eq!(cfg.d_model, 192);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.num_nest_layers, 18);
        assert_eq!(cfg.num_transformer_layers, 18);
        assert_eq!(cfg.num_speakers, 4);
        assert_eq!(cfg.subsampling_factor, 8);
        // Sortformer-specific invariant: the two encoder stack depths
        // are independent axes (NEST + plain Transformer), not
        // derived from each other. A future "simplification" that
        // silently collapses them into a single `num_layers` field
        // would break the primary-source topology.
        assert_eq!(
            cfg.num_nest_layers, cfg.num_transformer_layers,
            "primary source has both stacks at depth 18 — this equality is a \
             coincidence of the v1 release, NOT a derivable invariant; a future \
             8-speaker or larger release may split them"
        );
        // Sanity: `Default` matches `v1_default` (both must be
        // primary-source-transcribed constants; no silent divergence).
        assert_eq!(SortformerConfig::default(), cfg);
    }

    // -----------------------------------------------------------------------
    // 2. from_gguf full topology chunk-group round-trip (stamped path)
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_from_gguf_round_trips_stamped_chunk_group() {
        let cfg = SortformerConfig::v1_default();
        let file = sortformer_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let sf = SortformerDiar::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip — every stamped axis reads back into
        // the same SortformerConfig value (converter follow-up sub-
        // wave path).
        assert_eq!(*sf.config(), cfg);
        // NC weight license is the primary-source default per the
        // HF card (`license: cc-by-nc-4.0`) — the runtime must
        // surface it verbatim from the provenance chunk. The M2-13
        // compliance gate refuses this artifact in commercial mode
        // (T4 tier — `--allow-noncommercial` opt-in required).
        assert_eq!(sf.weight_license(), LicenseClass::NonCommercial);
        assert!(sf.tensor_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // 3. from_gguf falls back to primary-source constants on absent chunks
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_config_from_gguf_falls_back_to_primary_source_defaults() {
        // The Sortformer converter does NOT currently stamp the
        // `vokra.sortformer.*` chunk group (only arch / name /
        // category / upstream_hf / provenance). An already-published
        // GGUF must still load — the fallback path reads the
        // primary-source constants transcribed from the HF card + NeMo
        // YAML + paper. Mirror of PyanNetConfig::from_gguf fallback
        // pattern.
        let cfg = SortformerConfig::v1_default();
        let file = sortformer_gguf(
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let sf = SortformerDiar::from_gguf(&file).expect("chunk-free GGUF must bind via fallback");
        // Every axis fell through to its primary-source default —
        // the loader returns the same values as v1_default().
        assert_eq!(sf.config().d_model, DEFAULT_D_MODEL);
        assert_eq!(sf.config().n_heads, DEFAULT_N_HEADS);
        assert_eq!(sf.config().num_nest_layers, DEFAULT_NUM_NEST_LAYERS);
        assert_eq!(
            sf.config().num_transformer_layers,
            DEFAULT_NUM_TRANSFORMER_LAYERS
        );
        assert_eq!(sf.config().num_speakers, DEFAULT_NUM_SPEAKERS);
        assert_eq!(sf.config().subsampling_factor, DEFAULT_SUBSAMPLING_FACTOR);
    }

    // -----------------------------------------------------------------------
    // 4. from_gguf honors stamped chunks over defaults (converter forward-compat)
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_config_from_gguf_honors_stamped_chunks_over_defaults() {
        // Simulate a future converter sub-wave that starts stamping
        // the chunk group — a fixture with a deliberately non-default
        // value must round-trip through SortformerConfig::from_gguf
        // and override the primary-source default. This is the
        // forward-compat contract that lets the converter add stamps
        // without a runtime code change.
        let mut cfg = SortformerConfig::v1_default();
        cfg.d_model = 384; // arbitrary non-default (e.g. hypothetical Sortformer-8spk-v2)
        cfg.num_nest_layers = 24;
        let file = sortformer_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let sf = SortformerDiar::from_gguf(&file).expect("stamped GGUF must bind");
        assert_eq!(sf.config().d_model, 384);
        assert_eq!(sf.config().num_nest_layers, 24);
    }

    // -----------------------------------------------------------------------
    // 5. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_from_gguf_rejects_wrong_arch() {
        // A `parakeet-tdt` GGUF handed to the Sortformer binder by
        // mistake must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08). Both `parakeet-tdt` and
        // `sortformer` share the FastConformer encoder primitive but
        // have completely different terminal heads (TDT joint vs
        // 4-sigmoid diarization), so silent aliasing would misroute.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "parakeet-tdt");
        b.add_u32(GGUF_KEY_D_MODEL, 192);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SortformerDiar::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`parakeet-tdt`") && m.contains("`sortformer`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("4-sigmoid"),
                    "message should disambiguate Sortformer's diarization-head topology \
                     to help the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6. from_gguf rejects missing arch chunk
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_from_gguf_rejects_missing_arch() {
        // A GGUF that carries no `vokra.model.arch` at all (e.g. a
        // hand-assembled fixture from an unrelated pipeline) must
        // fail loud rather than mis-bind.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "not-sortformer");
        // No `vokra.model.arch`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SortformerDiar::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("sortformer"),
                    "message must name the sortformer binder so a reader knows \
                     which loader complained, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. Empty tensor manifest fails loud (never binds all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_from_gguf_rejects_empty_tensor_list() {
        // Correct arch + full chunk group but zero tensors — the
        // SortformerWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_D_MODEL, DEFAULT_D_MODEL);
        b.add_u32(GGUF_KEY_N_HEADS, DEFAULT_N_HEADS);
        b.add_u32(GGUF_KEY_NUM_NEST_LAYERS, DEFAULT_NUM_NEST_LAYERS);
        b.add_u32(
            GGUF_KEY_NUM_TRANSFORMER_LAYERS,
            DEFAULT_NUM_TRANSFORMER_LAYERS,
        );
        b.add_u32(GGUF_KEY_NUM_SPEAKERS, DEFAULT_NUM_SPEAKERS);
        b.add_u32(GGUF_KEY_SUBSAMPLING_FACTOR, DEFAULT_SUBSAMPLING_FACTOR);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SortformerDiar::from_gguf(&file) else {
            panic!("expected ModelLoad on empty tensor manifest");
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
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 8. diarize returns UnsupportedOp with primary-source anchors
    // -----------------------------------------------------------------------

    #[test]
    fn sortformer_diarize_loud_partial_returns_unsupported_op_with_primary_source_urls() {
        let cfg = SortformerConfig::v1_default();
        let file = sortformer_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let sf = SortformerDiar::from_gguf(&file).unwrap();
        // 1 second of 16 kHz mono silence — legitimate input shape,
        // so the loud-partial gate fires (not some pre-diarize
        // validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = sf.diarize(&pcm) else {
            panic!("diarize must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("sortformer diarize"),
                    "message must call out the sortformer diarize surface, got `{m}`"
                );
                assert!(
                    m.contains("ConformerEncoder"),
                    "message must name the reusable primitive `ConformerEncoder` so \
                     the follow-up wave knows the composition anchor, got `{m}`"
                );
                assert!(
                    m.contains("region-merging"),
                    "message must name the region-merging port so the follow-up \
                     wave knows the post-processing gap, got `{m}`"
                );
                // Both primary-source URLs must be cited — the task's
                // hint requires the message contain the primary
                // source URLs.
                assert!(
                    m.contains(PRIMARY_SOURCE_HF_CARD),
                    "message must contain the HF card URL substring \
                     (huggingface.co/nvidia/diar_sortformer_4spk-v1), got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_NEMO_REPO),
                    "message must contain the NeMo repo URL substring \
                     (github.com/NVIDIA/NeMo), got `{m}`"
                );
                assert!(
                    m.contains(PRIMARY_SOURCE_PAPER),
                    "message must contain the paper URL substring \
                     (arxiv.org/abs/2409.06656), got `{m}`"
                );
                // Every config axis must be echoed so the reader
                // can cross-check what topology the follow-up wave
                // targets.
                assert!(
                    m.contains("num_nest_layers=18"),
                    "num_nest_layers axis missing: {m}"
                );
                assert!(
                    m.contains("num_transformer_layers=18"),
                    "num_transformer_layers axis missing: {m}"
                );
                assert!(m.contains("d_model=192"), "d_model axis missing: {m}");
                assert!(m.contains("n_heads=8"), "n_heads axis missing: {m}");
                assert!(
                    m.contains("num_speakers=4"),
                    "num_speakers axis missing: {m}"
                );
                assert!(
                    m.contains("subsampling_factor=8"),
                    "subsampling_factor axis missing: {m}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 9. SpeakerSegment surface pin — fields match task-hint spelling
    // -----------------------------------------------------------------------

    #[test]
    fn speaker_segment_surface_pin() {
        // Surface pin: field names + types + derives must match the
        // task-hint spelling `{speaker_id: usize, start_s: f32, end_s:
        // f32}`. A rename or shape change would land here in the same
        // commit or fail this test.
        let seg = SpeakerSegment {
            speaker_id: 0,
            start_s: 1.5,
            end_s: 3.25,
        };
        // Field-access + type-check smoke.
        let _sid: usize = seg.speaker_id;
        let _start: f32 = seg.start_s;
        let _end: f32 = seg.end_s;
        // duration = end - start (contract with `pyannote::rttm::DiarizationSegment`
        // which uses `duration_s` — pin the difference explicitly).
        let duration = seg.end_s - seg.start_s;
        assert!(
            (duration - 1.75).abs() < 1e-6,
            "duration = end_s - start_s must equal 1.75 for the pinned fixture, got {duration}"
        );
        // Derives smoke: Debug + Clone + Copy + PartialEq — Copy
        // implies the type is memcpy-safe, PartialEq lets tests
        // compare slices of segments.
        let cloned: SpeakerSegment = seg;
        assert_eq!(seg, cloned);
        let dbg = format!("{seg:?}");
        assert!(
            dbg.contains("SpeakerSegment") && dbg.contains("speaker_id"),
            "Debug output must render the field spellings, got `{dbg}`"
        );
        // Distinct semantics from pyannote::rttm::DiarizationSegment
        // (which uses `duration_s`, RTTM convention). The task-hint
        // asks for `end_s` here — pin the distinction so a future
        // consolidation-refactor does not silently swap the semantic.
        // We assert this by *presence of the `end_s` field* at type
        // level; the assignment above already exercises that.
        // Sortformer's 4-speaker range: speaker_id ∈ 0..4.
        for spk in 0..DEFAULT_NUM_SPEAKERS {
            let s = SpeakerSegment {
                speaker_id: spk as usize,
                start_s: 0.0,
                end_s: 0.5,
            };
            assert!(s.speaker_id < DEFAULT_NUM_SPEAKERS as usize);
        }
    }

    // -----------------------------------------------------------------------
    // 10. Default weight license is NonCommercial (T4 tier fail-closed)
    // -----------------------------------------------------------------------

    #[test]
    fn default_weight_license_stamps_noncommercial_t4_tier() {
        // The Sortformer converter's DEFAULT_LICENSE_SPDX is
        // `cc-by-nc-4.0` → LicenseClass::NonCommercial. The runtime
        // must surface this verbatim so the M2-13 compliance gate
        // refuses commercial-mode load (T4 tier — the same fail-
        // closed posture as X-Codec-2 precedent 2026-07-23).
        let cfg = SortformerConfig::v1_default();
        let file = sortformer_gguf(
            cfg,
            /*stamp_topology=*/ false,
            Some(LicenseClass::NonCommercial),
        );
        let sf = SortformerDiar::from_gguf(&file).expect("bind");
        assert_eq!(
            sf.weight_license(),
            LicenseClass::NonCommercial,
            "the Sortformer converter defaults to NonCommercial per the HF card's \
             `license: cc-by-nc-4.0` — the runtime binder must surface it so the \
             M2-13 compliance gate can refuse commercial-mode load (T4 tier)"
        );
        // Missing provenance stamp falls back to Unknown (also
        // fail-closed at the gate).
        let file_no_license = sortformer_gguf(cfg, /*stamp_topology=*/ false, None);
        let sf_no_license =
            SortformerDiar::from_gguf(&file_no_license).expect("bind without license stamp");
        assert_eq!(
            sf_no_license.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed)"
        );
    }

    // -----------------------------------------------------------------------
    // 11. Structural pin — arch tag is stable and distinct from sibling
    //     FastConformer-encoder ASR arches
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_fastconformer_asr_arches() {
        // Pin the arch string so a rename would land here in the
        // same commit or fail this test. The sibling FastConformer-
        // based ASR arches MUST NOT collide with ours — they share
        // the encoder primitive but have completely different
        // terminal heads.
        assert_eq!(ARCH, "sortformer");
        assert_eq!(NAME, "sortformer-diar-4spk-v1");
        // Direct string comparisons against the sibling arch tags to
        // document the "which sibling should NOT be aliased" contract
        // at test time (a future rename of any sibling arch would
        // land here in the same commit or fail this test).
        assert_ne!(
            ARCH, "parakeet-tdt",
            "sortformer (diarization) and parakeet-tdt (ASR + TDT joint) share the \
             FastConformer encoder but have different terminal heads — sharing arch \
             would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "parakeet-ctc",
            "sortformer (diarization) and parakeet-ctc (ASR + CTC head) share the \
             FastConformer encoder but have different terminal heads — sharing arch \
             would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "canary",
            "sortformer (diarization) and canary (ASR / AST + Transformer AED \
             decoder) share the FastConformer encoder but have different terminal \
             heads — sharing arch would mis-route (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // 12. matches_config soft accessor honestly reflects shape presence
    // -----------------------------------------------------------------------

    #[test]
    fn matches_config_soft_accessor_finds_d_model_axis() {
        // The soft accessor should return true when at least one
        // bound tensor has an axis matching `d_model`. The fixture
        // encoder tensor's rows/cols are both `d_model` so this must
        // pass.
        let cfg = SortformerConfig::v1_default();
        let file = sortformer_gguf(
            cfg,
            /*stamp_topology=*/ true,
            Some(LicenseClass::NonCommercial),
        );
        let gguf = file;
        let sf = SortformerDiar::from_gguf(&gguf).unwrap();
        assert!(
            sf.weights.matches_config(sf.config()),
            "at least one bound tensor must have an axis matching config.d_model"
        );
        // Sanity: a stale config (bogus d_model) does NOT match the
        // fixture — pins the accessor as a real check (not a stub
        // that always returns true).
        let stale = SortformerConfig {
            d_model: 99999,
            ..cfg
        };
        assert!(
            !sf.weights.matches_config(&stale),
            "matches_config must return false for a d_model with no matching axis"
        );
    }
}
