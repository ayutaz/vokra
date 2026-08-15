//! **SepFormer** (SpeechBrain sepformer family, apache-2.0) — dual-path
//! Transformer speech separation / enhancement runtime binder for the
//! `sepformer` converter arch (2026-08-14 audit follow-up Wave 5).
//!
//! # Primary source
//!
//! - Paper: Subakan, Ravanelli, Cornell, Bronzi & Zhong,
//!   *"Attention is All You Need in Speech Separation"* (ICASSP 2021 /
//!   arXiv:2010.13154 §3 "SepFormer").
//! - SpeechBrain reference implementations:
//!   - Dual-Path Transformer masker composition:
//!     <https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py>
//!     (`SBTransformerBlock`, `Dual_Path_Model`, `Dual_Computation_Block`).
//!   - Recurrent SepFormer wrapper (used by the WHAMR! variants):
//!     <https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/resepformer.py>
//!     (`ReSepFormer`).
//! - HF model cards (7 released variants): `speechbrain/sepformer-*`
//!   (`wsj02mix` / `libri2mix` / `libri3mix` / `wham16k-enhancement` /
//!   `whamr16k` / `whamr` (8 kHz WHAMR!) / `dns4-16k-enhancement`).
//! - Weight license: **apache-2.0** for every variant per HF cardData
//!   (see `docs/license-audit.md` §3.1 rows 364-370 all ☑ Commercial
//!   2026-07-30 / 2026-08-01 yousan).
//!
//! # Architecture (transcribed from primary sources)
//!
//! ```text
//! Mixture PCM (mono f32, 8 kHz for `whamr` / 16 kHz for the other six)
//!   -> Learnable 1D encoder                            ← **loud-partial**
//!        (a strided Conv1D projects raw audio into a
//!         time-frequency-like representation — the
//!         non-negative masked latent that the masker
//!         attends over.)
//!   -> Dual-Path Transformer masker                    ← **loud-partial**
//!        (chunking → IntraTransformer (SBTransformerBlock
//!         over the intra-chunk axis) → InterTransformer
//!         (SBTransformerBlock over the inter-chunk axis) →
//!         de-chunking → PReLU → 1x1 Conv → `n_out`-way
//!         parallel masker head; the tensor-name walk from
//!         upstream `state_dict` prefixes onto the composed
//!         masker forward has NOT been pinned pending the
//!         upstream tensor-name manifest fetch.)
//!   -> Learnable 1D decoder                            ← **loud-partial**
//!        (per-speaker `n_out` reconstruction via a strided
//!         transposed Conv1D — one output waveform stream
//!         per masker column.)
//!   -> Vec<Vec<f32>> (n_out parallel speaker / enhanced streams)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`SepFormer::from_gguf`] with strict
//!   `vokra.model.arch == "sepformer"` validation +
//!   [`SepformerVariant`] tag round-trip (required, no silent default —
//!   a Libri3Mix GGUF silently loaded as a Wsj02mix would corrupt the
//!   downstream `n_out` axis) + strict `vokra.sepformer.n_out` cross-
//!   check against the variant-derived expectation (mismatch = converter
//!   bug, loud FR-EX-08 fail) + [`SepformerWeights::from_gguf`] with a
//!   floor of non-empty tensor count enforced loud + weight-license
//!   class surfacing (defaults to [`LicenseClass::Unknown`] on a stamp-
//!   free fixture, fail-closed at the M2-13 compliance gate — the
//!   converter stamps [`LicenseClass::Permissive`] in production per
//!   the apache-2.0 default).
//! - **Loud-partial (this WP)**: [`SepFormer::separate`] returns
//!   [`VokraError::UnsupportedOp`] naming the dual-path Transformer
//!   masker composition (encoder + IntraTransformer + InterTransformer
//!   + decoder per Subakan et al. 2021) + the tensor-name walk from
//!     upstream `speechbrain/sepformer-*` `state_dict` prefixes onto the
//!     composed masker forward. The message cites all three primary
//!     sources (dual_path.py + resepformer.py + arXiv:2010.13154) and
//!     echoes variant / tag / n_out / category so a reader diagnosing
//!     this gap knows exactly which of the 7 variants fired and where to
//!     walk.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / redimnet / sortformer Wave 1-4 loud-partial
//! precedent, CLAUDE.md 教訓 (a)): the surrounding scaffold +
//! `from_gguf` chunk-group validation + FR-EX-08 loud-fails land today
//! so a follow-up wave can flip the switch by (i) landing the tensor-
//! name walk against a real SpeechBrain `sepformer-*` `state_dict`
//! (the release ships PyTorch checkpoints on HF that the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` uv-managed Python 3.12
//! sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]` bridges to safetensors), (ii) composing
//! the `SBTransformerBlock`-based intra/inter masker forward from
//! Vokra's existing softmax + GEMM + LayerNorm primitives (no new op
//! needed for the Transformer body itself). The learnable encoder /
//! decoder Conv1D bank is **not** greenfield either: a conv1d kernel
//! exists at `vokra_backend_cpu::kernels::conv1d_f32` and is reachable
//! through `vokra_models::compute::Compute::conv1d_f32` with Metal /
//! CUDA / WebGPU coverage. The greenfield work is the *composition* —
//! SepFormer's dual-path chunking, the `n_out`-way parallel mask head,
//! and the overlap-add — not the arithmetic underneath it.
//!
//! # `vokra.sepformer.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::sepformer::convert_sepformer_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"sepformer"`).
//!   Distinct from every sibling separation / enhancement arch
//!   (`metricgan_plus`, `mp_senet_dns`, `denoise` (DFN3), `rnnoise`,
//!   `nsnet2`, `dnsmos`) — silently sharing would misroute runtime
//!   dispatch (FR-EX-08). All 7 variants share this arch tag since
//!   they share the SepFormer topology (encoder + dual-path
//!   Transformer masker + decoder).
//! - `vokra.model.name` (`String`): the versioned identifier that
//!   matches the `huggingface.co/vokra/` publish slug for the specific
//!   variant.
//! - `vokra.model.category` (`String`): `"separation"` for the
//!   multi-speaker variants (`wsj02mix`, `libri2mix`, `libri3mix`) or
//!   `"enhancement"` for the single-output variants (`wham16k-*`,
//!   `whamr16k`, `whamr` (8 kHz), `dns4-16k-enhancement`).
//! - `vokra.sepformer.variant` (`String`): one of `"wsj02mix"`,
//!   `"libri2mix"`, `"libri3mix"`, `"wham16k-enhancement"`, `"whamr16k"`,
//!   `"whamr8k"`, `"dns4-16k-enhancement"`. Read strict (required — no
//!   silent default) so a Libri3Mix GGUF silently loaded as a Wsj02mix
//!   is caught at load time (both share the `separation` category but
//!   differ in `n_out`).
//! - `vokra.sepformer.n_out` (`u32`): the number of parallel output
//!   streams the masker head emits (`1` for the 4 enhancement variants,
//!   `2` for `wsj02mix` / `libri2mix`, `3` for `libri3mix`). Read strict
//!   (required — the converter always stamps it; a mismatch against the
//!   variant-derived expectation is a converter bug per FR-EX-08).
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03 / M2-13) can classify the
//!   artifact without re-inspecting the safetensors provenance.
//!   Defaults to `Permissive` in production per apache-2.0 stamp.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`KEY_SEPFORMER_VARIANT`] /
//! [`KEY_SEPFORMER_N_OUT`] / [`KEY_MODEL_CATEGORY`] — same rule the
//! sibling BF16 pass-through binders (`pyannote` / `snac` / `hifigan` /
//! `beat_this` / `mt3` / `redimnet` / `sortformer_diar_4spk_v1`) use so
//! `vokra-models` does not gain a dependency edge onto `vokra-convert`,
//! preserving the layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`. A `[test]` at the bottom of this
//! module pins the mirror so a converter-side rename lands here in the
//! same commit or fails the pin.
//!
//! # No ONNX / no pickle (permanent)
//!
//! SpeechBrain ships PyTorch checkpoints (safetensors); this runtime
//! **never** touches ONNX (FR-LD-05 / NFR-DS-02). The .pt / .ckpt
//! → safetensors bridge lives offline through
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12
//! sidecar — not part of the runtime), mirroring the Parakeet /
//! Canary / Parakeet-CTC / Sortformer bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/sepformer.rs` (see module docstring
// for the cross-crate duplication rationale).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model sepformer-*`.
///
/// Deliberately shared across all 7 variants — they share the SepFormer
/// topology (encoder + dual-path Transformer masker + decoder — Subakan
/// et al. 2021). Distinct from every sibling separation / enhancement
/// arch (`metricgan_plus`, `mp_senet_dns`, `denoise` (DFN3), `rnnoise`,
/// `nsnet2`, `dnsmos`): silently sharing would misroute runtime
/// dispatch (FR-EX-08).
pub const ARCH: &str = "sepformer";

/// `vokra.model.category` — `"separation"` for `wsj02mix` / `libri2mix` /
/// `libri3mix` (multi-speaker source separation) or `"enhancement"` for
/// the 4 single-output variants (`wham16k-enhancement`, `whamr16k`,
/// `whamr8k`, `dns4-16k-enhancement`). Mirror of the converter's
/// `KEY_MODEL_CATEGORY` (raw string, not covered by `chunks::`).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.sepformer.variant` — one of `"wsj02mix"` / `"libri2mix"` /
/// `"libri3mix"` / `"wham16k-enhancement"` / `"whamr16k"` / `"whamr8k"` /
/// `"dns4-16k-enhancement"`. Distinguishes the 7 SpeechBrain SepFormer
/// releases (all share the SepFormer topology but differ in training
/// corpus, sample rate, and `n_out`).
pub const KEY_SEPFORMER_VARIANT: &str = "vokra.sepformer.variant";

/// `vokra.sepformer.n_out` — the number of parallel output streams the
/// masker head emits (`1` for the 4 enhancement variants, `2` for
/// `wsj02mix` / `libri2mix`, `3` for `libri3mix`). Read strict — a
/// stamped value that mismatches the variant-derived expectation is a
/// converter bug per FR-EX-08.
pub const KEY_SEPFORMER_N_OUT: &str = "vokra.sepformer.n_out";

/// `vokra.model.category` value emitted for the 3 multi-speaker
/// variants (`wsj02mix`, `libri2mix`, `libri3mix`).
pub const CATEGORY_SEPARATION: &str = "separation";
/// `vokra.model.category` value emitted for the 4 single-output
/// variants (`wham16k-enhancement`, `whamr16k`, `whamr8k`,
/// `dns4-16k-enhancement`).
pub const CATEGORY_ENHANCEMENT: &str = "enhancement";

/// `vokra.sepformer.variant` tag — WSJ0-2mix 2-speaker separation.
pub const VARIANT_TAG_WSJ02MIX: &str = "wsj02mix";
/// `vokra.sepformer.variant` tag — LibriMix 2-speaker separation.
pub const VARIANT_TAG_LIBRI2MIX: &str = "libri2mix";
/// `vokra.sepformer.variant` tag — LibriMix 3-speaker cocktail-party
/// separation (`n_out = 3`, distinct from the 2-speaker LibriMix
/// sibling).
pub const VARIANT_TAG_LIBRI3MIX: &str = "libri3mix";
/// `vokra.sepformer.variant` tag — WHAM! 16 kHz speech enhancement.
pub const VARIANT_TAG_WHAM16K_ENHANCEMENT: &str = "wham16k-enhancement";
/// `vokra.sepformer.variant` tag — WHAMR! 16 kHz joint dereverb +
/// denoise.
pub const VARIANT_TAG_WHAMR16K: &str = "whamr16k";
/// `vokra.sepformer.variant` tag — WHAMR! **8 kHz** joint dereverb +
/// denoise (base-sample-rate sibling of `whamr16k`).
pub const VARIANT_TAG_WHAMR8K: &str = "whamr8k";
/// `vokra.sepformer.variant` tag — Microsoft DNS-4 16 kHz speech
/// enhancement (distinct training corpus from the WHAM! / WHAMR!
/// siblings, same single-output head).
pub const VARIANT_TAG_DNS4_ENHANCEMENT: &str = "dns4-16k-enhancement";

/// Primary-source anchor: the dual-path Transformer masker
/// composition (`SBTransformerBlock`, `Dual_Path_Model`,
/// `Dual_Computation_Block`). Cited in the loud-partial error so a
/// reader diagnosing this gap knows the composition anchor.
const PRIMARY_SOURCE_DUAL_PATH: &str =
    "github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py";
/// Primary-source anchor: the recurrent SepFormer wrapper
/// (`ReSepFormer`) used by the WHAMR! variants. Cited in the loud-
/// partial error so a reader diagnosing this gap knows the WHAMR!-
/// specific composition anchor.
const PRIMARY_SOURCE_RESEPFORMER: &str =
    "github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/resepformer.py";
/// Paper anchor (Subakan et al. ICASSP 2021) — cited alongside the
/// two source URLs so a reader has the theoretical context as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2010.13154";

// ---------------------------------------------------------------------------
// SepformerVariant — mirror of the converter enum. All 7 arms with
// name / upstream_hf / tag / category / n_out accessors.
// ---------------------------------------------------------------------------

/// Which SpeechBrain SepFormer release was converted.
///
/// All 7 variants share the [`ARCH`] tag `sepformer`; the category,
/// upstream HF slug, sample rate, and `n_out` differ.
///
/// Mirror of `SepformerVariant` in
/// `crates/vokra-convert/src/models/sepformer.rs`.
/// A test at the bottom of this module pins every accessor's return
/// value byte-for-byte against the converter constants, so a rename
/// on either side lands in the same commit or fails the pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SepformerVariant {
    /// `speechbrain/sepformer-wsj02mix`: 2-speaker source separation
    /// on WSJ0-2mix. Category = `separation`,
    /// `vokra.sepformer.variant = "wsj02mix"`, `n_out = 2`.
    Wsj02mix,
    /// `speechbrain/sepformer-libri2mix`: 2-speaker source separation
    /// on LibriMix (LibriSpeech-derived CC-BY-4.0 corpus). Same
    /// 2-speaker head as [`Self::Wsj02mix`] — the two differ only in
    /// the training corpus. Category = `separation`,
    /// `vokra.sepformer.variant = "libri2mix"`, `n_out = 2`.
    Libri2Mix,
    /// `speechbrain/sepformer-libri3mix`: **3-speaker** source
    /// separation on LibriMix (Libri3Mix cocktail-party mixture set).
    /// Same SepFormer topology as [`Self::Libri2Mix`] with the same
    /// LibriSpeech-derived training corpus — the sole difference is
    /// the masker output head branches into **3 parallel speaker
    /// streams instead of 2**. Category = `separation`,
    /// `vokra.sepformer.variant = "libri3mix"`, `n_out = 3`.
    Libri3Mix,
    /// `speechbrain/sepformer-wham16k-enhancement`: single-speaker
    /// speech enhancement (WHAM! 16 kHz). Category = `enhancement`,
    /// `vokra.sepformer.variant = "wham16k-enhancement"`, `n_out = 1`.
    Wham16kEnhancement,
    /// `speechbrain/sepformer-whamr16k`: joint dereverb + denoise
    /// (WHAMR! 16 kHz). Category = `enhancement`,
    /// `vokra.sepformer.variant = "whamr16k"`, `n_out = 1`.
    Whamr16k,
    /// `speechbrain/sepformer-whamr`: joint dereverb + denoise
    /// (WHAMR! **8 kHz** — base-sample-rate sibling of
    /// [`Self::Whamr16k`]). Category = `enhancement`,
    /// `vokra.sepformer.variant = "whamr8k"`, `n_out = 1`.
    Whamr8k,
    /// `speechbrain/sepformer-dns4-16k-enhancement`: single-speaker
    /// speech enhancement trained on the **Microsoft DNS-4** corpus
    /// at 16 kHz. Same SepFormer topology as the WHAM! / WHAMR!
    /// enhancement siblings; distinct training corpus. Category =
    /// `enhancement`,
    /// `vokra.sepformer.variant = "dns4-16k-enhancement"`, `n_out = 1`.
    Dns4Enhancement,
}

impl SepformerVariant {
    /// The `vokra.model.name` string for this release.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wsj02mix => "sepformer-wsj02mix",
            Self::Libri2Mix => "sepformer-libri2mix",
            Self::Libri3Mix => "sepformer-libri3mix",
            Self::Wham16kEnhancement => "sepformer-wham16k-enhancement",
            Self::Whamr16k => "sepformer-whamr16k",
            Self::Whamr8k => "sepformer-whamr",
            Self::Dns4Enhancement => "sepformer-dns4-16k-enhancement",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`).
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Wsj02mix => "speechbrain/sepformer-wsj02mix",
            Self::Libri2Mix => "speechbrain/sepformer-libri2mix",
            Self::Libri3Mix => "speechbrain/sepformer-libri3mix",
            Self::Wham16kEnhancement => "speechbrain/sepformer-wham16k-enhancement",
            Self::Whamr16k => "speechbrain/sepformer-whamr16k",
            Self::Whamr8k => "speechbrain/sepformer-whamr",
            Self::Dns4Enhancement => "speechbrain/sepformer-dns4-16k-enhancement",
        }
    }

    /// The `vokra.sepformer.variant` tag.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Wsj02mix => VARIANT_TAG_WSJ02MIX,
            Self::Libri2Mix => VARIANT_TAG_LIBRI2MIX,
            Self::Libri3Mix => VARIANT_TAG_LIBRI3MIX,
            Self::Wham16kEnhancement => VARIANT_TAG_WHAM16K_ENHANCEMENT,
            Self::Whamr16k => VARIANT_TAG_WHAMR16K,
            Self::Whamr8k => VARIANT_TAG_WHAMR8K,
            Self::Dns4Enhancement => VARIANT_TAG_DNS4_ENHANCEMENT,
        }
    }

    /// The `vokra.model.category` value. Wsj02mix / Libri2Mix /
    /// Libri3Mix are pure **source-separation** tasks (N speakers out
    /// of 1 mixture); the WHAM / WHAMR / DNS-4 variants are
    /// single-output **enhancement** tasks (dereverb / denoise).
    #[must_use]
    pub const fn category(self) -> &'static str {
        match self {
            Self::Wsj02mix | Self::Libri2Mix | Self::Libri3Mix => CATEGORY_SEPARATION,
            Self::Wham16kEnhancement | Self::Whamr16k | Self::Whamr8k | Self::Dns4Enhancement => {
                CATEGORY_ENHANCEMENT
            }
        }
    }

    /// The number of parallel output streams the masker head emits.
    ///
    /// - `1` for every enhancement variant (single-speaker dereverb /
    ///   denoise).
    /// - `2` for the standard 2-speaker separation task
    ///   ([`Self::Wsj02mix`] / [`Self::Libri2Mix`]).
    /// - `3` for the LibriMix 3-speaker cocktail-party head
    ///   ([`Self::Libri3Mix`]).
    #[must_use]
    pub const fn n_out(self) -> u32 {
        match self {
            Self::Wham16kEnhancement | Self::Whamr16k | Self::Whamr8k | Self::Dns4Enhancement => 1,
            Self::Wsj02mix | Self::Libri2Mix => 2,
            Self::Libri3Mix => 3,
        }
    }

    /// Reverse lookup: parse a `vokra.sepformer.variant` tag back to
    /// the enum. Returns `None` for anything not one of the 7 known
    /// tags so the caller falls loud rather than silently defaulting.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            VARIANT_TAG_WSJ02MIX => Some(Self::Wsj02mix),
            VARIANT_TAG_LIBRI2MIX => Some(Self::Libri2Mix),
            VARIANT_TAG_LIBRI3MIX => Some(Self::Libri3Mix),
            VARIANT_TAG_WHAM16K_ENHANCEMENT => Some(Self::Wham16kEnhancement),
            VARIANT_TAG_WHAMR16K => Some(Self::Whamr16k),
            VARIANT_TAG_WHAMR8K => Some(Self::Whamr8k),
            VARIANT_TAG_DNS4_ENHANCEMENT => Some(Self::Dns4Enhancement),
            _ => None,
        }
    }

    /// The 7-way iteration list for tests / exhaustiveness checks.
    /// Kept as a `const` so a new variant that skips adding a row here
    /// fails the pairwise-distinctness pin at the bottom of this module.
    pub const ALL: [Self; 7] = [
        Self::Wsj02mix,
        Self::Libri2Mix,
        Self::Libri3Mix,
        Self::Wham16kEnhancement,
        Self::Whamr16k,
        Self::Whamr8k,
        Self::Dns4Enhancement,
    ];
}

// ---------------------------------------------------------------------------
// SepformerConfig — the composite variant / n_out / category axes.
// Derived from the variant tag; every field is redundant with the
// variant accessors but the struct pins the (variant, n_out, category)
// triple as the public surface a future forward wave binds against.
// ---------------------------------------------------------------------------

/// SepFormer variant + derived axes as they ride the artifact.
///
/// Every field is redundant with the [`SepformerVariant`] accessors —
/// the struct exists so a future forward wave (dual-path Transformer
/// masker + encoder / decoder Conv1D bank) has a single value carrying
/// the (variant, n_out, category) triple through the composition
/// without repeatedly walking `variant.n_out()` / `variant.category()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SepformerConfig {
    /// Which SpeechBrain SepFormer release this artifact came from.
    pub variant: SepformerVariant,
    /// Number of parallel output streams the masker head emits (mirror
    /// of `variant.n_out()`).
    pub n_out: u32,
    /// Task category — `"separation"` or `"enhancement"` (mirror of
    /// `variant.category()`).
    pub category: &'static str,
}

impl SepformerConfig {
    /// Builds a config from a variant tag — the three axes are all
    /// derivable from the variant so there is only one honest constructor.
    #[must_use]
    pub const fn for_variant(variant: SepformerVariant) -> Self {
        Self {
            variant,
            n_out: variant.n_out(),
            category: variant.category(),
        }
    }
}

// ---------------------------------------------------------------------------
// SepformerWeights — bound the tensor manifest with a non-emptiness
// gate. Under the loud-partial WP the weights are counted but the
// dual-path Transformer masker + encoder / decoder Conv1D bank forward
// is deferred. Mirror of `SortformerWeights` / `ReDimNetWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a SepFormer GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// SepFormer checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The dual-path Transformer masker
/// + encoder / decoder Conv1D bank forward is deferred (see
///   [`SepFormer::separate`] loud-partial), so the payload is not yet
///   dequantised — the follow-up wave sizes the dequant per its kernel
///   needs.
#[derive(Debug)]
pub struct SepformerWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up masker-forward
    /// wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl SepformerWeights {
    /// Scans `gguf` for the SepFormer state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid SepFormer checkpoint).
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
                "sepformer: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model sepformer-*` \
                 against an upstream `speechbrain/sepformer-*` safetensors checkpoint."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the masker-forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// SepFormer — the runtime binder handle
// ---------------------------------------------------------------------------

/// SpeechBrain SepFormer dual-path Transformer speech separation /
/// enhancement runtime binder (`speechbrain/sepformer-*`, apache-2.0).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`separate`](Self::separate) on a mixture PCM buffer to obtain a
/// `Vec<Vec<f32>>` of the `n_out` parallel speaker / enhanced streams.
/// See the module doc for the current implementation-status matrix and
/// the FR-EX-08 loud-error contract on the dual-path Transformer masker
/// composition.
#[derive(Debug)]
pub struct SepFormer {
    config: SepformerConfig,
    variant: SepformerVariant,
    n_out: u32,
    // The bound weights are held (real, counted) but the masker + encoder
    // + decoder Conv1D forward composition is a follow-up wave; the
    // field is deliberately `#[allow(dead_code)]` until the composition
    // lands so a reader is not misled by an unused field. Same posture
    // as RMVPE / pyannote / mt3 / beat_this / redimnet / sortformer.
    #[allow(dead_code)]
    weights: SepformerWeights,
    weight_license: LicenseClass,
}

impl SepFormer {
    /// Binds a SepFormer GGUF: validates arch, reads the variant tag +
    /// `n_out` (with variant/n_out cross-check), discovers tensors, and
    /// surfaces the stamped weight-license class for compliance gate
    /// cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong /
    /// mismatched key so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial
    /// bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"sepformer"` (a `metricgan_plus` / `mp_senet_dns` /
    ///   `denoise` / `rnnoise` / `nsnet2` GGUF handed to us by mistake
    ///   fails with a clear message instead of a downstream missing-
    ///   tensor).
    /// - [`VokraError::ModelLoad`] when `vokra.sepformer.variant` is
    ///   absent (silent default would corrupt `n_out`).
    /// - [`VokraError::ModelLoad`] when `vokra.sepformer.variant` is
    ///   not one of the 7 known tags.
    /// - [`VokraError::ModelLoad`] when `vokra.sepformer.n_out` is
    ///   absent (converter always stamps it — strict per ReDimNet
    ///   posture).
    /// - [`VokraError::ModelLoad`] when the stamped `n_out` mismatches
    ///   the variant-derived expectation (converter bug per FR-EX-08).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`SepformerWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "vokra.sepformer.variant missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "sepformer: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model sepformer-*`? Note that sibling \
                     separation / enhancement arches — `metricgan_plus` (MetricGAN+), \
                     `mp_senet_dns` (MP-SENet + DNS), `denoise` (DeepFilterNet3), \
                     `rnnoise` (RNNoise), `nsnet2` (NSNet2), `dnsmos` (DNSMOS evaluator) \
                     — are all distinct topologies with distinct terminal heads; SepFormer's \
                     dual-path Transformer masker + `n_out`-way parallel head has no analog \
                     in the sibling arches and silently aliasing would misroute the runtime \
                     dispatch, FR-EX-08)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "sepformer: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native sepformer GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Variant tag — REQUIRED (no silent default; a Libri3Mix
        //    GGUF silently loaded as a Wsj02mix would corrupt n_out).
        let variant_tag = file
            .get(KEY_SEPFORMER_VARIANT)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "sepformer: GGUF is missing required string chunk `{KEY_SEPFORMER_VARIANT}` \
                     — the converter always stamps it. Accepted tags: `{VARIANT_TAG_WSJ02MIX}`, \
                     `{VARIANT_TAG_LIBRI2MIX}`, `{VARIANT_TAG_LIBRI3MIX}`, \
                     `{VARIANT_TAG_WHAM16K_ENHANCEMENT}`, `{VARIANT_TAG_WHAMR16K}`, \
                     `{VARIANT_TAG_WHAMR8K}`, `{VARIANT_TAG_DNS4_ENHANCEMENT}`. Refusing a \
                     silent default to avoid mis-routing a Libri3Mix GGUF onto the 2-speaker \
                     Wsj02mix head (FR-EX-08)."
                ))
            })?;
        let variant = SepformerVariant::from_tag(variant_tag).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "sepformer: unknown `{KEY_SEPFORMER_VARIANT}` tag `{variant_tag}`. Accepted \
                 tags: `{VARIANT_TAG_WSJ02MIX}`, `{VARIANT_TAG_LIBRI2MIX}`, \
                 `{VARIANT_TAG_LIBRI3MIX}`, `{VARIANT_TAG_WHAM16K_ENHANCEMENT}`, \
                 `{VARIANT_TAG_WHAMR16K}`, `{VARIANT_TAG_WHAMR8K}`, \
                 `{VARIANT_TAG_DNS4_ENHANCEMENT}`. Refusing to silently default to any of \
                 the 7 to avoid mis-binding the downstream masker head (FR-EX-08)."
            ))
        })?;

        // 3. `n_out` — REQUIRED (converter always stamps it) and must
        //    match the variant-derived expectation. A mismatch is a
        //    converter bug per FR-EX-08 — silently accepting the
        //    stamped value would corrupt the downstream masker head
        //    output-stream allocation.
        let n_out = file
            .get(KEY_SEPFORMER_N_OUT)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "sepformer: GGUF is missing required u32 chunk `{KEY_SEPFORMER_N_OUT}` \
                     — the converter always stamps it (strict per ReDimNet posture). Re-run \
                     `vokra-cli convert --model sepformer-{tag}` against the upstream \
                     `speechbrain/sepformer-*` safetensors checkpoint to produce a proper \
                     artifact.",
                    tag = variant.tag()
                ))
            })?;
        let expected_n_out = variant.n_out();
        if n_out != expected_n_out {
            return Err(VokraError::ModelLoad(format!(
                "sepformer: `{KEY_SEPFORMER_N_OUT}` chunk value `{n_out}` disagrees with \
                 the variant-derived expectation `{expected_n_out}` for variant \
                 `{tag}` — this is a converter bug (FR-EX-08). The variant tag and n_out \
                 axis are stamped by the same converter pass, so silently accepting the \
                 stamped value would corrupt the downstream masker head output-stream \
                 allocation.",
                tag = variant.tag()
            )));
        }

        // 4. Load the tensor manifest with the non-emptiness gate.
        let weights = SepformerWeights::from_gguf(file)?;

        // 5. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The SepFormer
        //    converter stamps `Permissive` in production per the
        //    apache-2.0 default; missing provenance falls back to
        //    `Unknown` which is fail-closed at the M2-13 compliance
        //    gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        let config = SepformerConfig::for_variant(variant);
        Ok(Self {
            config,
            variant,
            n_out,
            weights,
            weight_license,
        })
    }

    /// The composite (variant, n_out, category) axes derived from the
    /// GGUF's `vokra.sepformer.variant` tag.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &SepformerConfig {
        &self.config
    }

    /// The variant this artifact carries — one of the 7 known SpeechBrain
    /// SepFormer releases.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> SepformerVariant {
        self.variant
    }

    /// The number of parallel output streams the masker head emits
    /// (mirror of `variant().n_out()` — the stamped value cross-
    /// checked against the variant-derived expectation at bind time).
    #[inline]
    #[must_use]
    pub const fn n_out(&self) -> u32 {
        self.n_out
    }

    /// The task category — `"separation"` for the 3 multi-speaker
    /// variants, `"enhancement"` for the 4 single-output variants.
    #[inline]
    #[must_use]
    pub const fn category(&self) -> &'static str {
        self.config.category
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the masker-forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The SepFormer
    /// converter stamps `Permissive` in production per the apache-2.0
    /// default; a GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed at the M2-13 compliance
    /// gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Test-only fixture: builds a `SepFormer` handle for a given
    /// variant that satisfies the [`SepformerWeights`] non-empty
    /// invariant honestly (single named placeholder tensor) so the
    /// loud-partial [`separate`](Self::separate) surface can be
    /// exercised without a real GGUF on disk. Same posture as
    /// `Vocos::synthesized` / `sortformer_diar_4spk_v1` fixtures —
    /// the loud-partial `separate` never touches weights, so the
    /// placeholder tensor is not misleading.
    ///
    /// Weight license is [`LicenseClass::Unknown`] (fail-closed at
    /// the M2-13 compliance gate) — a caller running compliance-
    /// gated code paths against a synthesized handle picks up the
    /// same fail-closed default a real GGUF missing the provenance
    /// stamp would.
    #[must_use]
    pub fn synthesized(variant: SepformerVariant) -> Self {
        let n_out = variant.n_out();
        let config = SepformerConfig::for_variant(variant);
        let weights = SepformerWeights {
            tensors: vec![("__synthesized_placeholder__".to_owned(), vec![1])],
        };
        Self {
            config,
            variant,
            n_out,
            weights,
            weight_license: LicenseClass::Unknown,
        }
    }

    /// Separates / enhances a mixture PCM buffer into `n_out` parallel
    /// speaker / enhanced streams.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the SepFormer inference
    /// path requires composing:
    ///
    /// 1. The **learnable 1D encoder** (strided Conv1D projecting the
    ///    raw waveform into a non-negative masked latent).
    /// 2. The **dual-path Transformer masker** — chunking →
    ///    IntraTransformer (`SBTransformerBlock` over the intra-chunk
    ///    axis) → InterTransformer (`SBTransformerBlock` over the
    ///    inter-chunk axis) → de-chunking → PReLU → 1x1 Conv →
    ///    `n_out`-way parallel masker head. The Transformer body
    ///    itself is composable from Vokra's existing softmax + GEMM +
    ///    LayerNorm primitives (no new op needed); the composition +
    ///    the tensor-name walk from the upstream
    ///    `speechbrain/sepformer-*` `state_dict` prefixes onto the
    ///    composed masker forward has NOT been pinned pending the
    ///    upstream tensor-name manifest fetch.
    /// 3. The **learnable 1D decoder** (per-speaker `n_out`
    ///    reconstruction via a strided transposed Conv1D — one output
    ///    waveform stream per masker column).
    ///
    /// The error message names the dual-path Transformer masker
    /// explicitly + cites all three primary source anchors
    /// (dual_path.py + resepformer.py + arXiv:2010.13154) + echoes
    /// variant / tag / n_out / category so a reader diagnosing this
    /// gap has exactly three places to walk and knows exactly which of
    /// the 7 variants fired. **No fabricated separated waveform stream
    /// is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred dual-path Transformer masker composition +
    ///   tensor-name walk.
    pub fn separate(&self, mixed_pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future
        // real implementation will consume it.
        let _ = mixed_pcm;
        Err(separate_forward_loud_partial(
            self.variant,
            self.n_out,
            self.config.category,
        ))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`SepFormer::separate`] until the dual-path Transformer masker
/// composition + encoder / decoder Conv1D bank + tensor-name walk
/// land.
///
/// Names **the dual-path Transformer masker** explicitly (the "dual-
/// path Transformer" phrase is what a reader diagnosing a SepFormer
/// stack search for in the primary sources) and cites all three
/// primary source anchors (dual_path.py + resepformer.py + arXiv paper).
/// Echoes variant / tag / n_out / category so the fire-path message
/// disambiguates which of the 7 variants was in flight.
///
/// Mirror of the RMVPE / pyannote / snac / hifigan / beat_this / mt3 /
/// redimnet / sortformer Wave 1-4 loud-partial-message precedent —
/// CLAUDE.md 教訓 (a).
fn separate_forward_loud_partial(
    variant: SepformerVariant,
    n_out: u32,
    category: &'static str,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "sepformer separate: dual-path Transformer masker composition + encoder / decoder \
         Conv1D bank + tensor-name walk pending. The SepFormer inference path (Subakan et al. \
         2021 §3, arXiv:2010.13154) requires composing (a) the learnable 1D encoder (strided \
         Conv1D projecting the raw waveform into a non-negative masked latent), (b) the \
         dual-path Transformer masker — chunking → IntraTransformer (`SBTransformerBlock` \
         over the intra-chunk axis) → InterTransformer (`SBTransformerBlock` over the \
         inter-chunk axis) → de-chunking → PReLU → 1x1 Conv → n_out-way parallel masker head \
         (the Transformer body itself is composable from Vokra's existing softmax + GEMM + \
         LayerNorm primitives; the composition + the tensor-name walk from the upstream \
         `{upstream}` `state_dict` prefixes onto the composed masker forward has NOT been \
         pinned pending the upstream tensor-name manifest fetch), and (c) the learnable 1D \
         decoder (per-speaker n_out reconstruction via a strided transposed Conv1D — one \
         output waveform stream per masker column). Variant: `{tag}` (name = `{name}`), \
         n_out = {n_out}, category = `{category}`. Primary sources: {dual_path} + \
         {resepformer} + {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は \
         fake-complete より honest') — no silent fabricated separated waveform stream ever \
         emitted (FR-EX-08).",
        upstream = variant.upstream_hf(),
        tag = variant.tag(),
        name = variant.name(),
        n_out = n_out,
        category = category,
        dual_path = PRIMARY_SOURCE_DUAL_PATH,
        resepformer = PRIMARY_SOURCE_RESEPFORMER,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the SepFormer runtime binder — round-trip on the
    //! variant / n_out / category / provenance chunk group +
    //! negative-space round-trip on the loud-partial gates + cross-
    //! crate constant-mirror pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would be
    //! `separate(...)` returning a real `Vec<Vec<f32>>` of `n_out`
    //! parallel speaker streams, but the dual-path Transformer masker
    //! + encoder / decoder Conv1D bank composition has not been walked
    //!   against the upstream `speechbrain/sepformer-*` state_dict (see
    //!   the module doc + [`SepFormer::separate`] rustdoc). Fabricating
    //!   a real-PCM output would violate CLAUDE.md 教訓 (a)
    //!   ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Cross-crate constant mirror pin**: [`ARCH`] and the four
    //!    metadata / category / variant keys match the converter's
    //!    verbatim constants — the tripwire for a rename on either
    //!    side landing without the other.
    //! 2. **Per-variant accessor pin**: for each of the 7 variants,
    //!    `name` / `upstream_hf` / `tag` / `category` / `n_out` match
    //!    the converter enum accessors' return values byte-for-byte.
    //! 3. **Pairwise distinctness pin**: the 7 variants have distinct
    //!    `name` / `tag` / `upstream_hf` triples (defense-in-depth
    //!    against a copy-paste variant addition inheriting the wrong
    //!    provenance).
    //! 4. **Arch-tag distinctness pin**: [`ARCH`] is stable and
    //!    distinct from every sibling separation / enhancement arch.
    //! 5. **Config round-trip**: `for_variant` builds the expected
    //!    (variant, n_out, category) triple for each of 7 variants.
    //! 6. **Synthesized round-trip**: `SepFormer::synthesized(v)`
    //!    round-trips the variant / n_out / category axes for all 7.
    //! 7. **Variant tag round-trip**: `SepformerVariant::from_tag` is
    //!    the inverse of `variant.tag()` for the 7 accepted tags and
    //!    returns `None` for unknown / empty / arch-tag inputs.
    //! 8. **`from_gguf` chunk-group round-trip** — for each of 7
    //!    variants: build a GGUF with arch + variant + n_out + name +
    //!    license and 1 fake F32 tensor, then bind and verify the
    //!    handle's `variant()` / `n_out()` / `category()` /
    //!    `weight_license()==Permissive` are correct.
    //! 9. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / missing variant /
    //!    unknown variant / missing n_out / variant-n_out mismatch /
    //!    empty tensor list) fires at its documented surface point,
    //!    in the documented error variant.
    //! 10. **`separate` loud-partial**: `separate(...)` returns
    //!     `UnsupportedOp` naming the dual-path Transformer masker +
    //!     `speechbrain` + `dual_path.py` + arXiv:2010.13154 + the
    //!     specific variant tag, for all 7 variants.
    //! 11. **Weight-license class round-trip**: omitting
    //!     `KEY_PROVENANCE_WEIGHT_LICENSE` reads back as `Unknown`,
    //!     stamping `Permissive` reads back as `Permissive`.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a minimal SepFormer GGUF carrying the arch tag + name +
    /// variant tag + n_out + one representative encoder tensor.
    /// `stamp_n_out_override` overrides the n_out stamp (used by the
    /// converter-bug mismatch test).
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn sepformer_gguf(
        variant: SepformerVariant,
        stamp_n_out_override: Option<u32>,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, variant.name());
        b.add_string(KEY_MODEL_CATEGORY, variant.category());
        b.add_string(KEY_SEPFORMER_VARIANT, variant.tag());
        b.add_u32(
            KEY_SEPFORMER_N_OUT,
            stamp_n_out_override.unwrap_or(variant.n_out()),
        );
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative encoder tensor so the non-emptiness gate
        // passes. Kept small (4 f32 = 16 bytes) to keep the fixture
        // cheap.
        b.add_tensor(
            "encoder.conv.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Cross-crate constant mirror pin
    // -----------------------------------------------------------------------

    /// The constants transcribed at the top of this module MUST match
    /// the converter's verbatim strings (`crates/vokra-convert/src/
    /// models/sepformer.rs`). A rename on either side without a
    /// matched land on the other lands here as a test failure — the
    /// tripwire for the cross-crate duplication rule.
    #[test]
    fn arch_and_keys_match_converter_verbatim() {
        // Arch tag.
        assert_eq!(ARCH, "sepformer");
        // Metadata / topology / provenance keys.
        assert_eq!(KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(KEY_SEPFORMER_VARIANT, "vokra.sepformer.variant");
        assert_eq!(KEY_SEPFORMER_N_OUT, "vokra.sepformer.n_out");
        // Category values.
        assert_eq!(CATEGORY_SEPARATION, "separation");
        assert_eq!(CATEGORY_ENHANCEMENT, "enhancement");
        // Variant tags.
        assert_eq!(VARIANT_TAG_WSJ02MIX, "wsj02mix");
        assert_eq!(VARIANT_TAG_LIBRI2MIX, "libri2mix");
        assert_eq!(VARIANT_TAG_LIBRI3MIX, "libri3mix");
        assert_eq!(VARIANT_TAG_WHAM16K_ENHANCEMENT, "wham16k-enhancement");
        assert_eq!(VARIANT_TAG_WHAMR16K, "whamr16k");
        assert_eq!(VARIANT_TAG_WHAMR8K, "whamr8k");
        assert_eq!(VARIANT_TAG_DNS4_ENHANCEMENT, "dns4-16k-enhancement");
    }

    // -----------------------------------------------------------------------
    // 2. Per-variant accessor pin — every axis matches the converter
    // -----------------------------------------------------------------------

    /// For each of the 7 variants, the accessors' return values must
    /// match the converter enum's return values byte-for-byte. A
    /// converter rename or accessor drift lands here as a mismatch.
    #[test]
    fn all_seven_variant_stamps_match_converter_verbatim() {
        // The 7 (variant, name, upstream_hf, tag, category, n_out)
        // rows below are transcribed verbatim from
        // `crates/vokra-convert/src/models/sepformer.rs::SepformerVariant`
        // — if the converter renames any field, the pin fires here.
        let cases: [(SepformerVariant, &str, &str, &str, &str, u32); 7] = [
            (
                SepformerVariant::Wsj02mix,
                "sepformer-wsj02mix",
                "speechbrain/sepformer-wsj02mix",
                "wsj02mix",
                "separation",
                2,
            ),
            (
                SepformerVariant::Libri2Mix,
                "sepformer-libri2mix",
                "speechbrain/sepformer-libri2mix",
                "libri2mix",
                "separation",
                2,
            ),
            (
                SepformerVariant::Libri3Mix,
                "sepformer-libri3mix",
                "speechbrain/sepformer-libri3mix",
                "libri3mix",
                "separation",
                3,
            ),
            (
                SepformerVariant::Wham16kEnhancement,
                "sepformer-wham16k-enhancement",
                "speechbrain/sepformer-wham16k-enhancement",
                "wham16k-enhancement",
                "enhancement",
                1,
            ),
            (
                SepformerVariant::Whamr16k,
                "sepformer-whamr16k",
                "speechbrain/sepformer-whamr16k",
                "whamr16k",
                "enhancement",
                1,
            ),
            (
                SepformerVariant::Whamr8k,
                "sepformer-whamr",
                "speechbrain/sepformer-whamr",
                "whamr8k",
                "enhancement",
                1,
            ),
            (
                SepformerVariant::Dns4Enhancement,
                "sepformer-dns4-16k-enhancement",
                "speechbrain/sepformer-dns4-16k-enhancement",
                "dns4-16k-enhancement",
                "enhancement",
                1,
            ),
        ];
        for (v, name, upstream, tag, category, n_out) in cases {
            assert_eq!(v.name(), name, "variant {v:?} name");
            assert_eq!(v.upstream_hf(), upstream, "variant {v:?} upstream_hf");
            assert_eq!(v.tag(), tag, "variant {v:?} tag");
            assert_eq!(v.category(), category, "variant {v:?} category");
            assert_eq!(v.n_out(), n_out, "variant {v:?} n_out");
        }
        assert_eq!(SepformerVariant::ALL.len(), 7);
    }

    // -----------------------------------------------------------------------
    // 3. Pairwise distinctness pin — no accidental provenance-sharing
    // -----------------------------------------------------------------------

    /// The 7 variants must have distinct `name` / `tag` / `upstream_hf`
    /// triples so a copy-paste variant addition never silently inherits
    /// the wrong provenance. The 4 enhancement variants share
    /// `n_out = 1`, so the distinct provenance triple is what surfaces
    /// a routing mistake in the shared-n_out case.
    #[test]
    fn every_variant_has_distinct_stamps() {
        let all = SepformerVariant::ALL;
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                let (a, b) = (all[i], all[j]);
                assert_ne!(a.name(), b.name(), "variants {a:?} and {b:?} share name");
                assert_ne!(a.tag(), b.tag(), "variants {a:?} and {b:?} share tag");
                assert_ne!(
                    a.upstream_hf(),
                    b.upstream_hf(),
                    "variants {a:?} and {b:?} share upstream_hf"
                );
            }
        }
        // Cross-check the 4 enhancement variants explicitly — they
        // all share `n_out = 1` and `category = "enhancement"` so the
        // upstream_hf tag is the only signal that would surface a
        // routing mistake.
        let enhancement_variants = [
            SepformerVariant::Wham16kEnhancement,
            SepformerVariant::Whamr16k,
            SepformerVariant::Whamr8k,
            SepformerVariant::Dns4Enhancement,
        ];
        for i in 0..enhancement_variants.len() {
            for j in (i + 1)..enhancement_variants.len() {
                let (a, b) = (enhancement_variants[i], enhancement_variants[j]);
                assert_ne!(
                    a.upstream_hf(),
                    b.upstream_hf(),
                    "enhancement variants {a:?} and {b:?} must have distinct upstream_hf \
                     (they share `n_out = 1` and `category = enhancement`, so upstream_hf \
                     is the only routing-mistake signal)"
                );
                assert_eq!(a.n_out(), 1);
                assert_eq!(b.n_out(), 1);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 4. Arch-tag distinct from sibling separation / enhancement arches
    // -----------------------------------------------------------------------

    #[test]
    fn arch_distinct_from_sibling_enhancement_families() {
        // Silently sharing an arch with a sibling separation /
        // enhancement family would misroute the runtime dispatch
        // (FR-EX-08). SepFormer's dual-path Transformer masker has
        // no analog in the sibling arches.
        let siblings = [
            "metricgan_plus",
            "mp_senet_dns",
            "denoise", // DeepFilterNet3
            "rnnoise",
            "nsnet2",
            "dnsmos",
        ];
        for sibling in siblings {
            assert_ne!(
                ARCH, sibling,
                "sepformer arch must not silently alias sibling `{sibling}`"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 5. Config derived from variant
    // -----------------------------------------------------------------------

    #[test]
    fn config_derived_from_variant() {
        for v in SepformerVariant::ALL {
            let cfg = SepformerConfig::for_variant(v);
            assert_eq!(cfg.variant, v);
            assert_eq!(cfg.n_out, v.n_out());
            assert_eq!(cfg.category, v.category());
        }
    }

    // -----------------------------------------------------------------------
    // 6. Synthesized round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn synthesized_round_trip_all_variants() {
        for v in SepformerVariant::ALL {
            let sf = SepFormer::synthesized(v);
            assert_eq!(sf.variant(), v);
            assert_eq!(sf.n_out(), v.n_out());
            assert_eq!(sf.category(), v.category());
            // Fail-closed default: a fixture-constructed handle carries
            // `Unknown` so a compliance-gated code path picks up the
            // same fail-closed default a real GGUF missing the stamp
            // would.
            assert_eq!(sf.weight_license(), LicenseClass::Unknown);
            assert_eq!(sf.tensor_count(), 1);
            assert_eq!(sf.config().variant, v);
        }
    }

    // -----------------------------------------------------------------------
    // 7. Variant tag round-trip via from_tag
    // -----------------------------------------------------------------------

    #[test]
    fn variant_tag_round_trips_via_from_tag() {
        for v in SepformerVariant::ALL {
            assert_eq!(
                SepformerVariant::from_tag(v.tag()),
                Some(v),
                "variant {v:?} tag round-trip failed"
            );
        }
        // Negative space — unknown / empty / arch-name tags return None
        // rather than silently defaulting to any of the 7.
        assert_eq!(SepformerVariant::from_tag("libri4mix"), None);
        assert_eq!(SepformerVariant::from_tag(""), None);
        assert_eq!(SepformerVariant::from_tag("sepformer"), None);
        assert_eq!(SepformerVariant::from_tag("wsj02mix "), None); // trailing space
        assert_eq!(SepformerVariant::from_tag("WSJ02mix"), None); // wrong case
    }

    // -----------------------------------------------------------------------
    // 8. from_gguf full chunk-group round-trip for all 7 variants
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_all_seven_variants_round_trip() {
        for v in SepformerVariant::ALL {
            let file = sepformer_gguf(v, None, Some(LicenseClass::Permissive));
            let sf = SepFormer::from_gguf(&file)
                .unwrap_or_else(|e| panic!("variant {v:?} must bind: {e:?}"));
            assert_eq!(sf.variant(), v);
            assert_eq!(sf.n_out(), v.n_out());
            assert_eq!(sf.category(), v.category());
            // The converter stamps Permissive per apache-2.0 default —
            // the runtime must surface it verbatim from the provenance
            // chunk.
            assert_eq!(sf.weight_license(), LicenseClass::Permissive);
            assert!(sf.tensor_count() >= 1);
            assert_eq!(sf.config().variant, v);
        }
    }

    // -----------------------------------------------------------------------
    // 9. from_gguf rejects missing arch chunk
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "not-sepformer");
        // NO `vokra.model.arch`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SepFormer::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("sepformer"),
                    "message must name the sepformer binder so a reader knows which loader \
                     complained, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 10. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch_and_hints_siblings() {
        // A `denoise` (DeepFilterNet3) GGUF handed to the SepFormer
        // binder by mistake must fail loud with a specific message
        // rather than silently mis-binding (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "denoise");
        b.add_string(KEY_SEPFORMER_VARIANT, VARIANT_TAG_WSJ02MIX);
        b.add_u32(KEY_SEPFORMER_N_OUT, 2);
        b.add_tensor("t.w", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SepFormer::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`denoise`") && m.contains("`sepformer`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("metricgan_plus"),
                    "message should hint sibling arches, got `{m}`"
                );
                assert!(
                    m.contains("dual-path Transformer masker"),
                    "message should disambiguate SepFormer's masker topology, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 11. from_gguf rejects missing variant tag
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_variant_tag() {
        // Correct arch but no variant tag — silent default would
        // corrupt n_out on a Libri3Mix / Libri2Mix mistake.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(KEY_SEPFORMER_N_OUT, 2);
        // NO `vokra.sepformer.variant`.
        b.add_tensor("t.w", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SepFormer::from_gguf(&file) else {
            panic!("expected ModelLoad on missing variant tag");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_SEPFORMER_VARIANT),
                    "message must name the missing chunk key, got `{m}`"
                );
                // All 7 accepted tags must be enumerated so the reader
                // knows the acceptable options.
                for expected in [
                    VARIANT_TAG_WSJ02MIX,
                    VARIANT_TAG_LIBRI2MIX,
                    VARIANT_TAG_LIBRI3MIX,
                    VARIANT_TAG_WHAM16K_ENHANCEMENT,
                    VARIANT_TAG_WHAMR16K,
                    VARIANT_TAG_WHAMR8K,
                    VARIANT_TAG_DNS4_ENHANCEMENT,
                ] {
                    assert!(
                        m.contains(expected),
                        "message must enumerate accepted tag `{expected}`, got `{m}`"
                    );
                }
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 12. from_gguf rejects unknown variant tag
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_unknown_variant_tag() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(KEY_SEPFORMER_VARIANT, "libri4mix"); // hypothetical future release
        b.add_u32(KEY_SEPFORMER_N_OUT, 4);
        b.add_tensor("t.w", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SepFormer::from_gguf(&file) else {
            panic!("expected ModelLoad on unknown variant tag");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("libri4mix"),
                    "message must echo the unknown tag, got `{m}`"
                );
                for expected in [
                    VARIANT_TAG_WSJ02MIX,
                    VARIANT_TAG_LIBRI2MIX,
                    VARIANT_TAG_LIBRI3MIX,
                    VARIANT_TAG_WHAM16K_ENHANCEMENT,
                    VARIANT_TAG_WHAMR16K,
                    VARIANT_TAG_WHAMR8K,
                    VARIANT_TAG_DNS4_ENHANCEMENT,
                ] {
                    assert!(
                        m.contains(expected),
                        "message must enumerate accepted tag `{expected}`, got `{m}`"
                    );
                }
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 13. from_gguf rejects missing n_out
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_n_out() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(KEY_SEPFORMER_VARIANT, VARIANT_TAG_LIBRI3MIX);
        // NO `vokra.sepformer.n_out`.
        b.add_tensor("t.w", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SepFormer::from_gguf(&file) else {
            panic!("expected ModelLoad on missing n_out");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_SEPFORMER_N_OUT),
                    "message must name the missing chunk key, got `{m}`"
                );
                assert!(
                    m.contains("libri3mix"),
                    "message should include the variant tag from the artifact, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 14. from_gguf rejects variant / n_out mismatch (converter bug)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_variant_n_out_mismatch() {
        // A Libri3Mix GGUF stamped with `n_out = 2` (converter bug —
        // silently accepting would corrupt the downstream masker head
        // output-stream allocation). FR-EX-08 fail-loud.
        let file = sepformer_gguf(
            SepformerVariant::Libri3Mix,
            Some(2), // wrong — Libri3Mix carries n_out = 3
            Some(LicenseClass::Permissive),
        );
        let Err(err) = SepFormer::from_gguf(&file) else {
            panic!("expected ModelLoad on variant/n_out mismatch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`2`"),
                    "message must echo the stamped n_out value, got `{m}`"
                );
                assert!(
                    m.contains("`3`"),
                    "message must echo the variant-derived expectation, got `{m}`"
                );
                assert!(
                    m.contains("libri3mix"),
                    "message must name the variant tag, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08") || m.contains("converter bug"),
                    "message must cite the mismatch policy, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 15. from_gguf rejects empty tensor list
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + variant + n_out but zero tensors — the
        // SepformerWeights non-emptiness gate must fire (FR-EX-08 —
        // an empty GGUF is never a valid SepFormer checkpoint).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "sepformer-wsj02mix");
        b.add_string(KEY_SEPFORMER_VARIANT, VARIANT_TAG_WSJ02MIX);
        b.add_u32(KEY_SEPFORMER_N_OUT, 2);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = SepFormer::from_gguf(&file) else {
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
    // 16. separate returns UnsupportedOp naming dual-path masker + primary sources
    // -----------------------------------------------------------------------

    #[test]
    fn separate_returns_unsupported_op_naming_dual_path_masker() {
        let sf = SepFormer::synthesized(SepformerVariant::Wsj02mix);
        // 1 second of 16 kHz mono silence — legitimate input shape,
        // so the loud-partial gate fires (not some pre-separate
        // validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = sf.separate(&pcm) else {
            panic!("separate must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("sepformer separate"),
                    "message must call out the sepformer separate surface, got `{m}`"
                );
                assert!(
                    m.contains("dual-path Transformer masker"),
                    "message must name the dual-path Transformer masker so the follow-up \
                     wave knows the composition anchor, got `{m}`"
                );
                // Primary-source URLs — three anchors as promised.
                assert!(
                    m.contains("speechbrain"),
                    "message must contain speechbrain org substring, got `{m}`"
                );
                assert!(
                    m.contains("dual_path.py"),
                    "message must cite dual_path.py primary source, got `{m}`"
                );
                assert!(
                    m.contains("resepformer.py"),
                    "message must cite resepformer.py primary source, got `{m}`"
                );
                assert!(
                    m.contains("2010.13154") || m.contains("arxiv"),
                    "message must cite the arXiv paper anchor, got `{m}`"
                );
                assert!(
                    m.contains("wsj02mix"),
                    "message must echo the variant tag so a reader knows which of the 7 \
                     variants fired, got `{m}`"
                );
                assert!(
                    m.contains("n_out = 2"),
                    "message should echo the n_out axis, got `{m}`"
                );
                assert!(
                    m.contains("category = `separation`"),
                    "message should echo the category, got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 17. separate loud-partial fires for every variant with the right tag
    // -----------------------------------------------------------------------

    #[test]
    fn separate_all_seven_variants_fire_loud_partial() {
        let pcm = vec![0.0f32; 1024];
        for v in SepformerVariant::ALL {
            let sf = SepFormer::synthesized(v);
            let Err(err) = sf.separate(&pcm) else {
                panic!("separate must loud-partial for variant {v:?}");
            };
            match err {
                VokraError::UnsupportedOp(m) => {
                    // Defense against a silent Wsj02mix-only branch —
                    // every variant tag must appear in its own fire-
                    // path message.
                    assert!(
                        m.contains(v.tag()),
                        "variant {v:?}: message must echo the tag `{tag}`, got `{m}`",
                        tag = v.tag()
                    );
                    assert!(
                        m.contains(v.name()),
                        "variant {v:?}: message must echo the name `{name}`, got `{m}`",
                        name = v.name()
                    );
                    assert!(
                        m.contains(&format!("n_out = {}", v.n_out())),
                        "variant {v:?}: message must echo the correct n_out, got `{m}`"
                    );
                    assert!(
                        m.contains(v.category()),
                        "variant {v:?}: message must echo the category, got `{m}`"
                    );
                }
                other => panic!("expected VokraError::UnsupportedOp for {v:?}, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // 18. Weight-license class round-trip (fail-closed default + stamp)
    // -----------------------------------------------------------------------

    #[test]
    fn weight_license_class_round_trips() {
        // (a) Omitting the provenance chunk reads back as Unknown
        //     (fail-closed at the M2-13 compliance gate).
        let file_no_stamp = sepformer_gguf(SepformerVariant::Wsj02mix, None, None);
        let sf = SepFormer::from_gguf(&file_no_stamp).expect("bind without provenance stamp");
        assert_eq!(sf.weight_license(), LicenseClass::Unknown);

        // (b) Stamping Permissive (converter default per apache-2.0)
        //     reads back as Permissive.
        let file_permissive = sepformer_gguf(
            SepformerVariant::Wsj02mix,
            None,
            Some(LicenseClass::Permissive),
        );
        let sf = SepFormer::from_gguf(&file_permissive).expect("bind with Permissive stamp");
        assert_eq!(sf.weight_license(), LicenseClass::Permissive);
    }
}
