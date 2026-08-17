//! NVIDIA **Parakeet-TDT-1.1B** — runtime binder for the
//! `parakeet-tdt-1_1b` GGUF arch (Wave C1 2026-08-15 coverage-gap
//! follow-up, loud-partial per the emotion2vec / panns / redimnet / storm
//! precedent — CLAUDE.md 教訓 (a): "loud-partial は fake-complete より
//! honest").
//!
//! # The gap this closes
//!
//! `crates/vokra-convert/src/models/parakeet_tdt_1_1b.rs` has stamped
//! `vokra.model.arch = "parakeet-tdt-1_1b"` since the 2026-08-03 Wave B
//! coverage audit, but **no code in the workspace read that arch string**
//! — a converted Parakeet-TDT-1.1B GGUF could be produced and then never
//! loaded. This module is the reader half of that handshake.
//!
//! Note the spelling: the arch tag uses an **underscore**
//! (`parakeet-tdt-1_1b`) while the model name uses a **dot**
//! (`parakeet-tdt-1.1b`). Both spellings are load-bearing on the wire and
//! are pinned separately by the test suite (mirror of the `firered_vad`
//! arch/name split note in `lib.rs`).
//!
//! # Primary sources
//!
//! - HF release: <https://huggingface.co/nvidia/parakeet-tdt-1.1b>
//!   (weight license **CC-BY 4.0** — attribution required, commercial use
//!   permitted; transcribed from the converter's `DEFAULT_LICENSE`, whose
//!   value was primary-source verified per the wave-b ticket
//!   `docs/tickets/coverage-audit-2026-08-03/wave-b/parakeet-tdt-1.1b.md`).
//! - Reference implementation: NVIDIA NeMo (Apache-2.0),
//!   <https://github.com/NVIDIA/NeMo> — the TDT decoding semantics this
//!   module forwards to are ported in
//!   [`vokra_ops::rnnt_decode`](mod@vokra_ops::rnnt_decode), which cites
//!   `nemo/collections/asr/parts/submodules/tdt_beam_decoding.py` and
//!   `rnnt_greedy_decoding.py` line-by-line.
//!
//! No arXiv identifier is cited here: the TDT paper's id is not recorded
//! anywhere in this repository and inventing one would be a hallucinated
//! citation wearing an authoritative face (CLAUDE.md「ハルシネーション厳禁」).
//!
//! # What TDT is
//!
//! TDT (**T**oken-**a**nd-**D**uration **T**ransducer) is an RNN-T variant
//! whose joint head emits **two** distributions per step: the usual vocab
//! distribution over `V + 1` (blank inclusive) **and** a duration
//! distribution over `D` bins. The chosen duration decides how far the
//! frame pointer jumps, so a TDT model skips frames instead of walking
//! every one — this is what makes the Parakeet TDT variants fast.
//!
//! ```text
//! PCM (mono f32, 16 kHz)
//!   -> log-mel front-end                              <- loud-partial (axes deferred)
//!   -> FastConformer encoder (vokra_ops::conformer)   <- loud-partial (axes deferred)
//!   -> RNN-T prediction network (LSTM)                <- loud-partial (axes deferred)
//!   -> joint: enc_proj + dec_proj -> act -> {vocab head, duration head}
//!                                                     <- loud-partial (axes deferred)
//!   -> TDT decode                                     <- **REAL, WIRED**
//!        vokra_ops::rnnt_decode(RnntDecoderKind::Tdt { duration_bins })
//!        reachable today via [`ParakeetTdt11b::decode_tdt`]
//!   -> SentencePiece detokenize                       <- loud-partial (no vocab in GGUF)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! **Real (this WP)**:
//!
//! - [`ParakeetTdt11b::from_gguf`] — strict `vokra.model.arch ==
//!   "parakeet-tdt-1_1b"` verification with *specific* sibling-misroute
//!   diagnostics for the two neighbours that share the FastConformer
//!   encoder body but not the head (`parakeet-tdt` = the 0.6B-v3 TDT
//!   variant with different axes; `parakeet-ctc` = the 1.1B CTC variant
//!   with no prediction net / joint / duration head at all).
//! - [`ParakeetTdt11bWeights::from_gguf`] — tensor manifest discovery with
//!   a loud non-empty floor (a zero-tensor GGUF is refused rather than
//!   silently running an all-zero forward, FR-EX-08).
//! - Weight-license class surfacing, fail-closed to
//!   [`LicenseClass::Unknown`] when the stamp is absent.
//! - **[`ParakeetTdt11b::decode_tdt`] — the TDT decode leg, genuinely
//!   wired to [`vokra_ops::rnnt_decode`](mod@vokra_ops::rnnt_decode)'s `Tdt` mode.** The primitive
//!   already exposes a first-class TDT decoder (vocab argmax + duration
//!   argmax per frame, duration-driven frame skip, zero-duration multi-emit
//!   capped by `max_symbols_per_step`); this module forwards to it rather
//!   than re-implementing transducer decoding. A caller that materializes
//!   joint log-probs by any means (an external joint, a parity fixture, a
//!   future in-crate joint) gets a real decode today.
//!
//! **Loud-partial (this WP)**: [`ParakeetTdt11b::transcribe`] returns
//! [`VokraError::UnsupportedOp`] naming every missing piece. The blocker is
//! **not** a missing primitive on the decode side — it is that the
//! converter is, by design, a BF16 pass-through skeleton that stamps **no
//! `vokra.parakeet_tdt_1_1b.*` hparam chunk group**. Its module docstring
//! states the 1.1B axes are "transcribed by owner from the upstream
//! `config.json` when the first real weight arrives". Without `d_model`,
//! `n_layer`, `n_head`, `num_mel_bins`, `subsampling_factor`,
//! `attention_bias`, `vocab_size`, `blank_token_id` and the `durations`
//! bin list, the encoder / prediction net / joint cannot be shaped at all,
//! and **guessing them from the 0.6B-v3 sibling would be fabrication** —
//! the two releases are known to differ (0.6B-v3 uses 24 layers / 128 mel
//! bins / `attention_bias=false`; the 1.1B CTC sibling uses 42 layers /
//! 80 mel bins / `attention_bias=true`, so the 1.1B TDT axes are
//! genuinely unknown until transcribed).
//!
//! This is why no `ParakeetTdt11bConfig::parakeet_tdt_1_1b()` constant
//! exists in this module: there is nothing honest to put in it yet. The
//! sibling [`crate::parakeet`] and [`crate::parakeet_ctc`] modules DO carry
//! such constants because their `config.json` files were fetched and
//! transcribed verbatim (2026-07-24).
//!
//! # Sibling family distinctness
//!
//! [`ARCH`] is deliberately distinct from both Parakeet siblings:
//!
//! - `parakeet-tdt` — Parakeet-TDT-0.6B-v3. **Same head topology** (RNN-T
//!   prediction net + joint + duration bins) but different axes (24 layers,
//!   `d_model=1024`, 128 mel bins, `attention_bias=false`, vocab 8193,
//!   blank 8192). Silently aliasing would bind the 1.1B weights against
//!   0.6B shapes.
//! - `parakeet-ctc` — Parakeet-CTC-1.1B. **Different head topology
//!   entirely**: a single CTC vocab head, no prediction network, no joint
//!   projection, no duration bins. Silently aliasing would route a TDT
//!   checkpoint into a CTC blank-fold decode.
//!
//! FR-EX-08 forbids the silent shape misroute across these arches.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE`] are **mirrors of the converter's constants** — same
//! rule every sibling binder uses so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Parakeet ships `.nemo` (a tar.gz of yaml + ckpt), flattened offline to
//! safetensors by `tools/parity/nemo_pt_to_safetensors.py` (uv-managed
//! Python 3.12 per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) before the converter runs. The runtime never
//! sees ONNX, Python or torch (FR-LD-05 / NFR-DS-02).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{RnntAttrs, RnntDecoderKind, RnntHypothesis, rnnt_decode};

// ---------------------------------------------------------------------------
// Contract constants — mirror of
// `crates/vokra-convert/src/models/parakeet_tdt_1_1b.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model parakeet-tdt-1.1b`.
///
/// **Underscore**, not a dot — the converter chose `_1_1b` so the SKU is
/// pinned on the arch tag itself and a downstream reader can dispatch
/// without a second hparam lookup. Distinct from `parakeet-tdt` (the
/// 0.6B-v3 variant, different axes) and `parakeet-ctc` (the 1.1B CTC
/// variant, different head topology). Silent aliasing with either would
/// misroute the runtime dispatch (FR-EX-08 — see the module docstring's
/// "Sibling family distinctness" section).
pub const ARCH: &str = "parakeet-tdt-1_1b";

/// Expected `vokra.model.name` value written by the converter. **Dot**, not
/// an underscore — this matches the `huggingface.co/vokra/parakeet-tdt-1.1b`
/// publish slug and the `--model parakeet-tdt-1.1b` CLI argument.
pub const NAME: &str = "parakeet-tdt-1.1b";

/// Expected `vokra.model.category` value — the third `asr` entry in the
/// Parakeet family (after `parakeet-tdt-0.6b-v3` and `parakeet-ctc-1.1b`).
/// Consumed by the model-card generator + zoo manifest tier gate.
pub const CATEGORY: &str = "asr";

/// Ad-hoc metadata key for the model category. Mirror of the converter's
/// local constant — kept local (not a `chunks::KEY_*` alias) until a
/// sibling `category` consumer lands in `vokra-core`.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Metadata key under which the converter records the upstream repository
/// slug. Mirror of the converter's local constant.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Canonical NVIDIA HuggingFace slug recorded under
/// [`KEY_PROVENANCE_UPSTREAM_HF`]. Echoed in loud diagnostics so a reader
/// has a fully specified anchor without re-fetching a manifest.
pub const UPSTREAM_HF: &str = "nvidia/parakeet-tdt-1.1b";

/// Canonical weight-license SPDX the converter stamps by default —
/// `cc-by-4.0`, an [`LicenseClass::AttributionRequired`] class. Attribution
/// required, commercial use permitted; the FR-MD-09 attribution surface
/// activates so a downstream must display the NVIDIA attribution.
///
/// A caller may override the SPDX at convert time (`--license`), so this
/// binder **surfaces** whatever class the artifact carries rather than
/// asserting this value — see [`ParakeetTdt11b::weight_license`].
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";

/// Primary-source anchor for the HF release (cited in loud diagnostics).
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/nvidia/parakeet-tdt-1.1b";

/// Primary-source anchor for the reference implementation (NVIDIA NeMo,
/// Apache-2.0). The TDT decoding semantics this module forwards to are
/// ported in `vokra_ops::rnnt_decode`, whose module docs cite the exact
/// NeMo submodule paths and line ranges.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/NVIDIA/NeMo";

/// PCM sample rate the Parakeet family expects (16 kHz mono), per the
/// sibling [`crate::parakeet`] / [`crate::parakeet_ctc`] model cards. Not
/// stamped in the GGUF by this converter — recorded here as the audio
/// boundary a caller must satisfy, and echoed in the loud-partial message.
pub const PARAKEET_TDT_1_1B_SAMPLE_RATE: u32 = 16_000;

/// Duration-bin list used by the **0.6B-v3 sibling**, transcribed verbatim
/// from `huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json`
/// (fetched 2026-07-24 — see [`crate::parakeet`]).
///
/// # This is a reference value, NOT an assertion about the 1.1B release
///
/// It is exposed so a caller who has independently transcribed the 1.1B
/// `config.json` can compare, and so tests have a realistically-shaped bin
/// list. It is **never** used as a default anywhere in this module:
/// [`TdtDecodeParams`] requires the caller to supply the bins explicitly,
/// because the 1.1B axes have not been transcribed (the converter defers
/// them to the owner) and the two releases are known to differ in other
/// axes. Defaulting to this list would be fabrication.
pub const PARAKEET_TDT_0_6B_V3_REFERENCE_DURATIONS: [u32; 5] = [0, 1, 2, 3, 4];

/// NeMo's greedy `max_symbols_per_step` default (the zero-duration emission
/// cap). Sourced from [`vokra_ops::rnnt_decode`](mod@vokra_ops::rnnt_decode), whose `RnntAttrs`
/// constructors document `10` as the NeMo greedy default. Used by
/// [`TdtDecodeParams::nemo_defaults`].
pub const NEMO_DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

// ---------------------------------------------------------------------------
// TdtDecodeParams — caller-supplied decode axes (never defaulted).
// ---------------------------------------------------------------------------

/// Axes required to run the TDT decode leg.
///
/// Every field is **caller-supplied**. This module deliberately provides no
/// "the Parakeet-TDT-1.1B values" constructor: the converter stamps no
/// hparam chunk group for this SKU (its docstring defers the axis
/// transcription to the owner), so a built-in constant would be invented
/// numbers wearing an authoritative face. See the module docstring.
///
/// Conventions match [`vokra_ops::rnnt_decode::RnntAttrs`] exactly:
///
/// * `vocab_size` **excludes** the blank — the vocab head is `vocab_size +
///   1` wide. NeMo checkpoints express this as a head width; the ops-side
///   value is `head_width - 1` (e.g. the 0.6B-v3 head width 8193 becomes
///   `vocab_size = 8192`).
/// * `blank_id` must be inside `[0, vocab_size]`. NeMo puts the blank at
///   the tail (`blank_id == vocab_size`).
/// * `duration_bins` lists the bin values in head-output order, and must
///   contain at least one non-zero entry (an all-zero set would deadlock
///   the frame pointer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtDecodeParams {
    /// Number of encoder frames the joint has been materialized over.
    pub num_timesteps: usize,
    /// Vocabulary size **excluding** the blank symbol.
    pub vocab_size: usize,
    /// Blank symbol index inside `[0, vocab_size]`.
    pub blank_id: u32,
    /// Duration bins in head-output order.
    pub duration_bins: Vec<u32>,
    /// Zero-duration emission cap (NeMo `max_symbols_per_step`).
    pub max_symbols_per_step: usize,
}

impl TdtDecodeParams {
    /// Builds params with NeMo's structural defaults: blank at the tail of
    /// the vocab head (`blank_id = vocab_size`) and
    /// `max_symbols_per_step = `[`NEMO_DEFAULT_MAX_SYMBOLS_PER_STEP`].
    ///
    /// `duration_bins` stays caller-supplied — there is no honest default
    /// for this SKU (see the type-level docs).
    #[must_use]
    pub fn nemo_defaults(num_timesteps: usize, vocab_size: usize, duration_bins: Vec<u32>) -> Self {
        Self {
            num_timesteps,
            vocab_size,
            blank_id: vocab_size as u32,
            duration_bins,
            max_symbols_per_step: NEMO_DEFAULT_MAX_SYMBOLS_PER_STEP,
        }
    }

    /// Per-timestep float count of the joint log-prob buffer the TDT
    /// decoder expects: `(vocab_size + 1) + duration_bins.len()`.
    ///
    /// Exposed so a caller materializing joint outputs can lay them out
    /// correctly without re-deriving the layout from the ops-crate docs.
    #[must_use]
    pub fn joint_frame_stride(&self) -> usize {
        self.vocab_size + 1 + self.duration_bins.len()
    }

    /// Total float count of a well-formed joint buffer
    /// (`num_timesteps * joint_frame_stride()`), or `None` on overflow.
    #[must_use]
    pub fn expected_joint_len(&self) -> Option<usize> {
        self.num_timesteps.checked_mul(self.joint_frame_stride())
    }
}

// ---------------------------------------------------------------------------
// Weights — tensor manifest with a loud non-empty floor.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a Parakeet-TDT-1.1B GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF carrying zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// Parakeet checkpoint; the FastConformer encoder alone carries hundreds of
/// Linear / LayerNorm / Conv1D parameters, so an empty manifest always
/// signals a mis-produced GGUF).
///
/// The converter passes every upstream safetensors tensor through under its
/// **verbatim upstream name**, so this store keeps the names + GGUF-side
/// dims exactly as found. The follow-up wave that binds the real
/// FastConformer / prediction-net / joint tensors walks these same names
/// once the axis transcription lands.
#[derive(Debug)]
pub struct ParakeetTdt11bWeights {
    /// Tensors discovered on disk, indexed by verbatim upstream
    /// `state_dict` name with their GGUF-side dims.
    tensors: Vec<(String, Vec<usize>)>,
}

impl ParakeetTdt11bWeights {
    /// Scans `gguf` for the Parakeet-TDT-1.1B state_dict tensors. Refuses
    /// to bind if the GGUF carries zero tensors (FR-EX-08).
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
                "parakeet-tdt-1.1b: GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). A legitimate Parakeet-TDT-1.1B checkpoint \
                 carries hundreds of FastConformer encoder + RNN-T prediction network + \
                 joint / duration head parameters (arch={ARCH}, name={NAME}); zero \
                 tensors always signals a mis-produced GGUF. Re-run `vokra-cli convert \
                 --model parakeet-tdt-1.1b` against an upstream `{UPSTREAM_HF}` \
                 safetensors checkpoint (flatten the upstream `.nemo` first with \
                 `tools/parity/nemo_pt_to_safetensors.py`)."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Verbatim upstream tensor names discovered in the GGUF, in file
    /// order. Diagnostic accessor for the follow-up wave that maps these
    /// onto FastConformer / prediction-net / joint slots.
    // No `#[must_use]`: `impl Iterator` already carries it, and a bare
    // duplicate is a clippy error.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    /// GGUF-side dims of `name`, or `None` when absent.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// Whether `name` is present in the bound manifest.
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|(n, _)| n == name)
    }

    /// Looks up `name`, returning a loud [`VokraError::ModelLoad`] that
    /// echoes the missing tensor name when it is absent.
    ///
    /// This is the accessor the follow-up FastConformer / joint binding
    /// wave uses, so a checkpoint missing an expected slot fails naming
    /// exactly which one (FR-EX-08 — never a silent zero-fill).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming `name` when it is absent.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        self.tensor_dims(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "parakeet-tdt-1.1b: GGUF is missing required tensor `{name}` \
                 ({bound} tensors bound, arch={ARCH}). The converter passes every \
                 upstream safetensors tensor through under its verbatim upstream name, \
                 so a missing slot means the input checkpoint did not carry it — \
                 re-run `vokra-cli convert --model parakeet-tdt-1.1b` against a \
                 complete `{UPSTREAM_HF}` checkpoint (FR-EX-08 — no silent zero-fill).",
                bound = self.tensors.len(),
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// ParakeetTdt11b — the runtime binder handle.
// ---------------------------------------------------------------------------

/// Parakeet-TDT-1.1B (`nvidia/parakeet-tdt-1.1b`, CC-BY 4.0) runtime
/// binder.
///
/// Bind with [`from_gguf`](Self::from_gguf). The TDT **decode** leg is real
/// and reachable through [`decode_tdt`](Self::decode_tdt); the full PCM →
/// text [`transcribe`](Self::transcribe) path is a loud-partial until the
/// 1.1B hparam axes are transcribed (see the module doc).
#[derive(Debug)]
pub struct ParakeetTdt11b {
    weights: ParakeetTdt11bWeights,
    weight_license: LicenseClass,
}

impl ParakeetTdt11b {
    /// Binds a Parakeet-TDT-1.1B GGUF: validates arch strictly, discovers
    /// the tensor manifest, and surfaces the stamped weight-license class
    /// for the M2-13 compliance-gate cross-checks.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing / wrong key so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` names a Parakeet
    ///   sibling (`parakeet-tdt` = the 0.6B-v3 TDT variant with different
    ///   axes; `parakeet-ctc` = the 1.1B CTC variant with no prediction net
    ///   / joint / duration head), with a message routing the caller to the
    ///   correct binder.
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is any other
    ///   value, naming **both** the expected and the actual tag.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first — a mis-typed model handed here must fail
        //    with a specific message, never a downstream missing-tensor
        //    error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            // Sibling A: the 0.6B-v3 TDT variant. SAME head topology
            // (prediction net + joint + duration bins) but different axes —
            // this is the dangerous one, because a naive binder would
            // "successfully" bind 1.1B weights against 0.6B shapes.
            Some(sib @ ("parakeet-tdt" | "parakeet-tdt-0.6b-v3" | "parakeet-tdt-0.6b")) => {
                return Err(VokraError::ModelLoad(format!(
                    "parakeet-tdt-1.1b: GGUF arch is `{sib}` (Parakeet-TDT-0.6B-v3), \
                     expected `{ARCH}` (Parakeet-TDT-1.1B). These share the TDT head \
                     topology (RNN-T prediction network + joint projection + duration \
                     bins) but NOT their axes — 0.6B-v3 is 24 layers / d_model 1024 / \
                     128 mel bins / attention_bias=false / vocab 8193 / blank 8192, \
                     while the 1.1B axes are a separate transcription. Binding one \
                     against the other's shapes would corrupt every projection. Route \
                     this GGUF through the sibling `parakeet::ParakeetAsr` binder \
                     (`crates/vokra-models/src/parakeet/mod.rs`) instead (FR-EX-08 — \
                     no silent partial load)."
                )));
            }
            // Sibling B: the 1.1B CTC variant. Same scale, completely
            // different head.
            Some(sib @ ("parakeet-ctc" | "parakeet-ctc-1.1b" | "parakeet-ctc-1_1b")) => {
                return Err(VokraError::ModelLoad(format!(
                    "parakeet-tdt-1.1b: GGUF arch is `{sib}` (Parakeet-CTC-1.1B), \
                     expected `{ARCH}` (Parakeet-TDT-1.1B). Same 1.1B scale and the \
                     same FastConformer encoder body, but a completely different head: \
                     CTC has a single vocab head and NO prediction network, NO joint \
                     projection and NO duration bins, so it decodes by blank-folding \
                     rather than by TDT frame-skipping. Route this GGUF through the \
                     sibling `parakeet_ctc::ParakeetCtcAsr::from_gguf` binder \
                     (`crates/vokra-models/src/parakeet_ctc/mod.rs`) instead \
                     (FR-EX-08 — no silent partial load)."
                )));
            }
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "parakeet-tdt-1.1b: GGUF arch is `{other}`, expected `{ARCH}` (was \
                     this GGUF produced by `vokra-cli convert --model \
                     parakeet-tdt-1.1b`? Note the arch tag spells the SKU with an \
                     UNDERSCORE — `{ARCH}` — while the model name uses a dot, \
                     `{NAME}`. Sibling ASR arches — `whisper`, `voxtral`, `canary`, \
                     `parakeet-tdt`, `parakeet-ctc` — are different topologies and \
                     must not be aliased here, FR-EX-08)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "parakeet-tdt-1.1b: GGUF is missing `vokra.model.arch` — this is \
                     not a Vokra-native parakeet-tdt-1.1b GGUF (was it produced by \
                     `vokra-cli convert --model parakeet-tdt-1.1b`? expected arch \
                     `{ARCH}`)."
                )));
            }
        }

        // 2. Tensor manifest with the non-emptiness gate.
        let weights = ParakeetTdt11bWeights::from_gguf(file)?;

        // 3. Provenance surfacing. The converter stamps
        //    `AttributionRequired` by default (cc-by-4.0), but the SPDX is
        //    overridable at convert time, so we SURFACE whatever the
        //    artifact carries rather than asserting. A GGUF missing the
        //    stamp reads back as `Unknown` (fail-closed per memory
        //    `[[feedback-license-signoff-primary-source]]`); the outer
        //    M2-13 gate does the strict enforcement.
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

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk.
    ///
    /// A converter-produced artifact surfaces
    /// [`LicenseClass::AttributionRequired`] (CC-BY 4.0 — the NVIDIA
    /// attribution must be displayed, FR-MD-09). A GGUF missing the stamp
    /// reads back as [`LicenseClass::Unknown`] (fail-closed at the M2-13
    /// gate). This binder never asserts the class — the `--license`
    /// override at convert time is a supported path.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The bound weight manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &ParakeetTdt11bWeights {
        &self.weights
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Runs the **TDT decode leg** over a caller-materialized joint
    /// log-prob buffer.
    ///
    /// This is genuinely wired to [`vokra_ops::rnnt_decode`](mod@vokra_ops::rnnt_decode)'s
    /// [`RnntDecoderKind::Tdt`] mode — Vokra's transducer primitive already
    /// implements TDT (per-frame vocab argmax over `V + 1` **and** duration
    /// argmax over `D`, duration-driven frame skip, zero-duration multi-emit
    /// capped by `max_symbols_per_step`, blank-with-zero-duration force
    /// advance), ported against NeMo's `tdt_beam_decoding.py` /
    /// `rnnt_greedy_decoding.py`. Nothing is re-implemented here.
    ///
    /// # Buffer layout
    ///
    /// `joint_logprobs` is row-major, one contiguous frame per timestep,
    /// with stride [`TdtDecodeParams::joint_frame_stride`] =
    /// `(vocab_size + 1) + duration_bins.len()`. Within a frame the vocab
    /// head occupies the first `vocab_size + 1` floats (blank at
    /// `blank_id`) and the duration head the trailing
    /// `duration_bins.len()`. Values are treated as (log-)probabilities and
    /// are **not** log-softmaxed by the decoder — pass whatever the joint
    /// head produced.
    ///
    /// # Why this takes a buffer rather than PCM
    ///
    /// Producing that buffer requires the FastConformer encoder + RNN-T
    /// prediction network + joint, which cannot be shaped until the 1.1B
    /// hparam axes are transcribed (see [`Self::transcribe`] and the module
    /// doc). Exposing the decode leg separately means the half that *is*
    /// implementable today is real and usable — by a parity harness, by an
    /// externally-driven joint, or by the follow-up wave once it lands.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from
    ///   [`vokra_ops::rnnt_decode()`] on any degenerate input: zero
    ///   timesteps, zero vocab, `blank_id > vocab_size`, zero
    ///   `max_symbols_per_step`, empty or all-zero `duration_bins`, a
    ///   `joint_logprobs` length that does not match
    ///   `num_timesteps * stride`, or a `NaN` anywhere in the buffer.
    /// - [`VokraError::InvalidArgument`] if the decoder returns no
    ///   hypothesis (defensive — the `Tdt` arm always yields exactly one).
    pub fn decode_tdt(
        &self,
        joint_logprobs: &[f32],
        params: &TdtDecodeParams,
    ) -> Result<RnntHypothesis> {
        let attrs = RnntAttrs {
            num_timesteps: params.num_timesteps,
            vocab_size: params.vocab_size,
            blank_id: params.blank_id,
            max_symbols_per_step: params.max_symbols_per_step,
            kind: RnntDecoderKind::Tdt {
                duration_bins: params.duration_bins.clone(),
            },
        };
        // `rnnt_decode` performs the full structural validation (attrs,
        // buffer length, NaN scan) and returns loud `InvalidArgument`s —
        // no need to duplicate those checks here, and duplicating them
        // would risk drifting from the primitive's contract.
        let hyps = rnnt_decode(joint_logprobs, &attrs)?;
        hyps.into_iter().next().ok_or_else(|| {
            VokraError::InvalidArgument(
                "parakeet-tdt-1.1b decode_tdt: vokra_ops::rnnt_decode returned no \
                 hypothesis for RnntDecoderKind::Tdt (the Tdt arm is documented to \
                 always yield exactly one) — this indicates an ops-crate contract \
                 change, not a caller error."
                    .to_owned(),
            )
        })
    }

    /// Transcribes a mono `f32` PCM slice at
    /// [`PARAKEET_TDT_1_1B_SAMPLE_RATE`].
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. The blocker is **not** the
    /// decode algorithm — [`vokra_ops::rnnt_decode`](mod@vokra_ops::rnnt_decode) already implements TDT
    /// and is wired through [`Self::decode_tdt`]. The blocker is that the
    /// converter is a BF16 pass-through skeleton that stamps **no**
    /// `vokra.parakeet_tdt_1_1b.*` hparam chunk group (its docstring defers
    /// the 1.1B axis transcription to the owner, pending the first real
    /// weight). Without `d_model` / `n_layer` / `n_head` / `num_mel_bins` /
    /// `subsampling_factor` / `attention_bias` / `vocab_size` /
    /// `blank_token_id` / `durations`, the log-mel front-end, FastConformer
    /// encoder, prediction network and joint cannot be shaped, and copying
    /// the 0.6B-v3 sibling's axes would be fabrication (the releases are
    /// known to differ). **No fabricated token sequence is ever emitted**
    /// (FR-EX-08 — no silent partial output).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::UnsupportedOp`] otherwise — the loud-partial gate.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt-1.1b transcribe: pcm slice is empty".to_owned(),
            ));
        }
        Err(transcribe_loud_partial())
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`ParakeetTdt11b::transcribe`].
///
/// Names every missing piece, states explicitly which piece is **not**
/// missing (the TDT decode leg, already wired to `vokra_ops::rnnt_decode`),
/// identifies the root blocker (the absent `vokra.parakeet_tdt_1_1b.*`
/// hparam chunk group) and cites both primary sources. Mirror of the
/// emotion2vec / panns / redimnet / storm loud-partial message precedent
/// (CLAUDE.md 教訓 (a)).
fn transcribe_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "parakeet-tdt-1.1b transcribe (loud-partial): the full PCM -> text forward is \
         deferred. ROOT BLOCKER: the converter \
         (crates/vokra-convert/src/models/parakeet_tdt_1_1b.rs) is a BF16 pass-through \
         skeleton that stamps NO `vokra.parakeet_tdt_1_1b.*` hparam chunk group — its \
         docstring defers the 1.1B axis transcription to the owner, pending the first \
         real weight. Missing axes: d_model, n_layer, n_head, n_head_kv, ffn_dim, \
         conv_kernel_size, num_mel_bins, subsampling_factor, attention_bias, \
         convolution_bias, scale_input, vocab_size, blank_token_id, durations. Without \
         them these stages cannot be shaped: (1) log-mel front-end (num_mel_bins \
         unknown — the 0.6B-v3 sibling uses 128, the 1.1B CTC sibling uses 80, so the \
         1.1B TDT value is genuinely unknown); (2) FastConformer encoder \
         (vokra_ops::conformer needs n_layer / d_model / n_head / ffn_dim / \
         conv_kernel_size / subsampling_factor); (3) RNN-T prediction network (LSTM \
         width + layer count unknown); (4) joint projection + vocab head + duration \
         head (vocab_size / durations unknown); (5) SentencePiece detokenize (the \
         converter embeds no tokenizer chunk). NOT missing: the TDT DECODE leg — \
         vokra_ops::rnnt_decode already implements RnntDecoderKind::Tdt (per-frame \
         vocab argmax over V+1 plus duration argmax over D, duration-driven frame \
         skip, zero-duration multi-emit capped by max_symbols_per_step) and is wired \
         and callable today via ParakeetTdt11b::decode_tdt on a caller-materialized \
         joint buffer. Audio boundary when the forward lands: {sr} Hz mono f32. \
         Primary sources: HF release {hf}, reference implementation {code} (the NeMo \
         submodule paths and line ranges the decoder was ported against are cited in \
         vokra_ops::rnnt_decode's module docs). Copying the parakeet-tdt-0.6b-v3 axes \
         would be fabrication — the releases are known to differ. Runtime cannot \
         fabricate a token sequence (FR-EX-08 no silent partial output).",
        sr = PARAKEET_TDT_1_1B_SAMPLE_RATE,
        hf = PRIMARY_SOURCE_HF,
        code = PRIMARY_SOURCE_CODE,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the Parakeet-TDT-1.1B runtime binder.
    //!
    //! # What is honestly testable here
    //!
    //! The full `transcribe` path is a loud-partial (see the module doc),
    //! so no test asserts a transcript — fabricating one would violate
    //! CLAUDE.md 教訓 (a). What IS tested:
    //!
    //! 1. **Contract-constant pins** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` / `DEFAULT_LICENSE` mirror the converter exactly,
    //!    including the load-bearing underscore-vs-dot split between the
    //!    arch tag and the model name.
    //! 2. **Metadata round-trip** — a synthetic GGUF built with the
    //!    converter's exact chunk keys binds, and the license stamp
    //!    round-trips (with a fail-closed `Unknown` when absent).
    //! 3. **Loud negative space** — missing arch / foreign arch / each
    //!    Parakeet sibling arch / empty tensor manifest / missing tensor
    //!    each fire at their documented surface in their documented
    //!    variant, naming the offending value.
    //! 4. **The TDT decode leg, for real** — `decode_tdt` is exercised
    //!    end-to-end against hand-built joint frames whose expected output
    //!    is derivable by hand, proving the wiring to
    //!    `vokra_ops::rnnt_decode`'s `Tdt` mode is live rather than
    //!    nominal.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a GGUF carrying the converter's exact chunk set. Tensor names
    /// mirror the converter's own test fixtures (`encoder.blocks.0.*` /
    /// `decoder.pred_net.*`), which model the FastConformer + RNN-T
    /// prediction-net topology.
    fn parakeet_tdt_11b_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        b.add_tensor(
            "encoder.blocks.0.attn.qkv_proj.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor qkv");
        b.add_tensor(
            "decoder.pred_net.embed.weight",
            GgmlType::F32,
            vec![4, 2],
            vec![0u8; 4 * 2 * 4],
        )
        .expect("add_tensor embed");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// Builds a GGUF with an arbitrary arch tag and one tensor, so the arch
    /// gate is reached before the tensor gate.
    fn gguf_with_arch(arch: &str) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, arch);
        b.add_tensor("probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1 — Contract-constant pins (cross-crate consistency with the converter)
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "parakeet-tdt-1_1b", "arch tag pin (UNDERSCORE)");
        assert_eq!(NAME, "parakeet-tdt-1.1b", "model name pin (DOT)");
        assert_eq!(CATEGORY, "asr");
        assert_eq!(UPSTREAM_HF, "nvidia/parakeet-tdt-1.1b");
        assert_eq!(DEFAULT_LICENSE, "cc-by-4.0");
        assert_eq!(KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(KEY_PROVENANCE_UPSTREAM_HF, "vokra.provenance.upstream_hf");
        assert_eq!(PARAKEET_TDT_1_1B_SAMPLE_RATE, 16_000);
        assert_eq!(NEMO_DEFAULT_MAX_SYMBOLS_PER_STEP, 10);
    }

    /// The arch/name spelling split is load-bearing on the wire: the arch
    /// tag uses `_1_1b`, the publish slug / CLI argument uses `-1.1b`. A
    /// silent unification in either direction would break the handshake.
    #[test]
    fn arch_uses_underscore_while_name_uses_dot() {
        assert!(
            ARCH.contains("1_1b"),
            "arch must spell the SKU with `_`: {ARCH}"
        );
        assert!(!ARCH.contains("1.1b"), "arch must NOT use a dot: {ARCH}");
        assert!(
            NAME.contains("1.1b"),
            "name must spell the SKU with `.`: {NAME}"
        );
        assert_ne!(ARCH, NAME, "arch and name are distinct strings on the wire");
    }

    /// The arch tag must not alias either Parakeet sibling — silently
    /// sharing one would misroute runtime dispatch onto a different
    /// topology (FR-EX-08).
    #[test]
    fn arch_is_distinct_from_parakeet_siblings() {
        assert_ne!(
            ARCH,
            crate::parakeet::EXPECTED_ARCH,
            "must not alias the 0.6B-v3 TDT arch tag"
        );
        assert_ne!(
            ARCH,
            crate::parakeet_ctc::EXPECTED_ARCH,
            "must not alias the 1.1B CTC arch tag"
        );
    }

    // -----------------------------------------------------------------------
    // 2 — Metadata round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_synthetic_checkpoint() {
        let file = parakeet_tdt_11b_gguf(Some(LicenseClass::AttributionRequired));
        let m = ParakeetTdt11b::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            m.weight_license(),
            LicenseClass::AttributionRequired,
            "cc-by-4.0 => AttributionRequired must round-trip (FR-MD-09 attribution \
             surface activates)"
        );
        assert_eq!(m.tensor_count(), 2, "both fixture tensors must bind");
        assert!(
            m.weights()
                .has_tensor("encoder.blocks.0.attn.qkv_proj.weight"),
            "verbatim upstream tensor names must be preserved"
        );
        assert_eq!(
            m.weights()
                .tensor_dims("decoder.pred_net.embed.weight")
                .expect("dims present"),
            &[4, 2],
            "GGUF-side dims must round-trip"
        );
        let names: Vec<&str> = m.weights().tensor_names().collect();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("arch + tensors are the bind gates");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "absent stamp must fail-closed to Unknown, never to a permissive default"
        );
    }

    // -----------------------------------------------------------------------
    // 3 — Loud negative space
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "something-else");
        b.add_tensor("probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ParakeetTdt11b::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must name the missing key: {m}"
                );
                assert!(m.contains(ARCH), "message must name the expected arch: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_both_tags() {
        let file = gguf_with_arch("whisper");
        let Err(err) = ParakeetTdt11b::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`whisper`"),
                    "message must name the ACTUAL arch: {m}"
                );
                assert!(m.contains(ARCH), "message must name the EXPECTED arch: {m}");
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause: {m}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// The dangerous sibling: same TDT head topology, different axes. The
    /// message must route the caller to the 0.6B-v3 binder rather than
    /// letting a 1.1B checkpoint bind against 0.6B shapes.
    #[test]
    fn from_gguf_rejects_parakeet_tdt_0_6b_sibling_with_routing_hint() {
        for sibling in ["parakeet-tdt", "parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b"] {
            let file = gguf_with_arch(sibling);
            let Err(err) = ParakeetTdt11b::from_gguf(&file) else {
                panic!("expected ModelLoad on sibling arch `{sibling}`");
            };
            match err {
                VokraError::ModelLoad(m) => {
                    assert!(m.contains(sibling), "must name the actual arch: {m}");
                    assert!(m.contains(ARCH), "must name the expected arch: {m}");
                    assert!(
                        m.contains("parakeet::ParakeetAsr"),
                        "must route the caller to the 0.6B-v3 binder: {m}"
                    );
                    assert!(m.contains("axes"), "must explain that the axes differ: {m}");
                }
                other => panic!("expected VokraError::ModelLoad, got {other:?}"),
            }
        }
    }

    /// The other sibling: same scale, no prediction net / joint / duration
    /// head at all.
    #[test]
    fn from_gguf_rejects_parakeet_ctc_sibling_with_routing_hint() {
        for sibling in ["parakeet-ctc", "parakeet-ctc-1.1b", "parakeet-ctc-1_1b"] {
            let file = gguf_with_arch(sibling);
            let Err(err) = ParakeetTdt11b::from_gguf(&file) else {
                panic!("expected ModelLoad on sibling arch `{sibling}`");
            };
            match err {
                VokraError::ModelLoad(m) => {
                    assert!(m.contains(sibling), "must name the actual arch: {m}");
                    assert!(m.contains(ARCH), "must name the expected arch: {m}");
                    assert!(
                        m.contains("parakeet_ctc::ParakeetCtcAsr::from_gguf"),
                        "must route the caller to the CTC binder: {m}"
                    );
                    assert!(
                        m.contains("duration bins"),
                        "must explain the head-topology divergence: {m}"
                    );
                }
                other => panic!("expected VokraError::ModelLoad, got {other:?}"),
            }
        }
    }

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // No tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ParakeetTdt11b::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(m.contains("zero tensors"), "must name the gap: {m}");
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
                assert!(
                    m.contains("vokra-cli convert --model parakeet-tdt-1.1b"),
                    "must include the repro command: {m}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// A missing tensor must fail naming exactly which one (the accessor
    /// the follow-up FastConformer binding wave uses).
    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");
        // Present.
        assert_eq!(
            m.weights()
                .require_tensor("encoder.blocks.0.attn.qkv_proj.weight")
                .expect("present tensor must resolve"),
            &[2, 3]
        );
        // Absent.
        let Err(err) = m.weights().require_tensor("joint.duration_head.weight") else {
            panic!("expected ModelLoad for an absent tensor");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("joint.duration_head.weight"),
                    "message must name the missing tensor: {msg}"
                );
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");
        let Err(err) = m.transcribe(&[]) else {
            panic!("expected InvalidArgument on empty PCM");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "empty PCM must be InvalidArgument, got {err:?}"
        );
    }

    /// The loud-partial must name every missing stage, identify the ROOT
    /// blocker (the absent hparam chunk group), state that the TDT decode
    /// leg is NOT the blocker, and cite both primary sources.
    #[test]
    fn transcribe_loud_partials_naming_missing_primitives() {
        let file = parakeet_tdt_11b_gguf(Some(LicenseClass::AttributionRequired));
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");
        let pcm = vec![0.0_f32; 16_000]; // 1 s of silence at 16 kHz mono.
        let Err(err) = m.transcribe(&pcm) else {
            panic!("transcribe must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("loud-partial"), "posture label: {msg}");
                // Root blocker named precisely.
                assert!(
                    msg.contains("vokra.parakeet_tdt_1_1b.*"),
                    "must name the absent hparam chunk group as the root blocker: {msg}"
                );
                // Every deferred stage named.
                for stage in [
                    "log-mel",
                    "FastConformer encoder",
                    "RNN-T prediction network",
                    "duration head",
                    "SentencePiece",
                ] {
                    assert!(msg.contains(stage), "must name stage `{stage}`: {msg}");
                }
                // The primitive that is NOT missing is called out, with the
                // reachable entry point.
                assert!(
                    msg.contains("rnnt_decode"),
                    "must name the wired decode primitive: {msg}"
                );
                assert!(
                    msg.contains("ParakeetTdt11b::decode_tdt"),
                    "must name the reachable decode entry point: {msg}"
                );
                // Anti-fabrication rationale + primary sources.
                assert!(
                    msg.contains("fabrication"),
                    "must state why the 0.6B axes are not copied: {msg}"
                );
                for url in [PRIMARY_SOURCE_HF, PRIMARY_SOURCE_CODE] {
                    assert!(msg.contains(url), "must cite `{url}`: {msg}");
                }
                assert!(msg.contains("FR-EX-08"), "must cite FR-EX-08: {msg}");
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 4 — The TDT decode leg, for real
    // -----------------------------------------------------------------------

    #[test]
    fn decode_params_layout_matches_ops_stride() {
        let p = TdtDecodeParams::nemo_defaults(3, 3, vec![0, 1, 2]);
        // (vocab_size + 1) + D = 4 + 3 = 7.
        assert_eq!(p.joint_frame_stride(), 7);
        assert_eq!(p.expected_joint_len(), Some(21));
        // NeMo structural defaults.
        assert_eq!(
            p.blank_id, 3,
            "NeMo puts blank at the tail of the vocab head"
        );
        assert_eq!(p.max_symbols_per_step, NEMO_DEFAULT_MAX_SYMBOLS_PER_STEP);
    }

    /// End-to-end proof that the wiring to `vokra_ops::rnnt_decode`'s `Tdt`
    /// mode is live: the duration head drives a frame skip, so frame 1 is
    /// never visited even though its vocab logits are the largest in the
    /// buffer.
    ///
    /// Layout: `vocab_size = 3` (head width 4, blank = 3) + `D = 3` bins
    /// `[0, 1, 2]`, stride 7.
    ///
    /// * t=0 — vocab argmax = 1 (4.0); duration argmax = bin index 2 => 2.
    ///   Emit token 1 at t=0, jump to t=2.
    /// * t=1 — SKIPPED. Its vocab logits are all 9.0, so if the decoder
    ///   walked every frame it would emit here and the assertion would
    ///   fail. This is what makes the test prove frame-skipping.
    /// * t=2 — vocab argmax = 2 (3.0); duration argmax = bin index 1 => 1.
    ///   Emit token 2 at t=2, advance to t=3 = num_timesteps, done.
    ///
    /// Score = (4.0 + 2.0) + (3.0 + 1.0) = 10.0 (both heads contribute).
    #[test]
    fn decode_tdt_uses_the_duration_head_to_skip_frames() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");

        #[rustfmt::skip]
        let joint: Vec<f32> = vec![
            // t=0: vocab[4]                  durations[3]
            0.0, 4.0, 0.0, 0.0,               0.0, 1.0, 2.0,
            // t=1: skipped by the duration jump (large logits on purpose)
            9.0, 9.0, 9.0, 9.0,               0.0, 0.0, 0.0,
            // t=2
            0.0, 0.0, 3.0, 0.0,               0.0, 1.0, 0.0,
        ];
        let params = TdtDecodeParams::nemo_defaults(3, 3, vec![0, 1, 2]);
        assert_eq!(
            joint.len(),
            params.expected_joint_len().expect("no overflow"),
            "fixture must match the documented stride"
        );

        let hyp = m.decode_tdt(&joint, &params).expect("decode must succeed");
        assert_eq!(
            hyp.tokens,
            vec![1, 2],
            "frame 1 must be skipped by the duration jump — a decoder that walked \
             every frame would have emitted its argmax here"
        );
        assert_eq!(hyp.timestamps, vec![0, 2]);
        assert!(
            (hyp.score - 10.0).abs() < 1e-6,
            "score must sum both the vocab and the duration head: {}",
            hyp.score
        );
        assert_eq!(hyp.last_frame, 3);
    }

    /// A blank-dominant buffer emits nothing but still terminates cleanly
    /// (the primitive force-advances on blank-with-zero-duration).
    #[test]
    fn decode_tdt_all_blank_yields_empty_token_sequence() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");

        #[rustfmt::skip]
        let joint: Vec<f32> = vec![
            0.0, 0.0, 0.0, 2.0,   3.0, 0.0, // blank, duration bin 0 (= 0)
            0.0, 0.0, 0.0, 2.0,   3.0, 0.0,
        ];
        let params = TdtDecodeParams::nemo_defaults(2, 3, vec![0, 1]);
        let hyp = m.decode_tdt(&joint, &params).expect("decode must succeed");
        assert!(hyp.tokens.is_empty(), "all-blank must emit nothing");
        assert!(hyp.timestamps.is_empty());
        assert_eq!(hyp.last_frame, 2, "must terminate, not deadlock");
    }

    /// A buffer whose length does not match `num_timesteps * stride` must
    /// fail loud through the primitive's validation (FR-EX-08 — never a
    /// silent truncation).
    #[test]
    fn decode_tdt_rejects_buffer_shape_mismatch() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");
        let params = TdtDecodeParams::nemo_defaults(3, 3, vec![0, 1, 2]);
        // 20 floats where 21 are required.
        let joint = vec![0.0_f32; 20];
        let Err(err) = m.decode_tdt(&joint, &params) else {
            panic!("expected InvalidArgument on a short joint buffer");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("length 20"),
                    "must name the actual length: {msg}"
                );
                assert!(msg.contains("expected"), "must name the expectation: {msg}");
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    /// All-zero duration bins would deadlock the frame pointer; the
    /// primitive rejects them and this binder must surface that rather
    /// than hanging.
    #[test]
    fn decode_tdt_rejects_all_zero_duration_bins() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");
        let params = TdtDecodeParams::nemo_defaults(1, 3, vec![0, 0]);
        let joint = vec![0.0_f32; 6];
        let Err(err) = m.decode_tdt(&joint, &params) else {
            panic!("expected InvalidArgument on all-zero duration bins");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "all-zero bins must be InvalidArgument, got {err:?}"
        );
    }

    /// `NaN` anywhere in the joint buffer is a loud error, not a silent
    /// argmax surprise.
    #[test]
    fn decode_tdt_rejects_nan_in_joint_buffer() {
        let file = parakeet_tdt_11b_gguf(None);
        let m = ParakeetTdt11b::from_gguf(&file).expect("bind");
        let params = TdtDecodeParams::nemo_defaults(2, 3, vec![0, 1]);
        let mut joint = vec![0.0_f32; 12];
        joint[7] = f32::NAN;
        let Err(err) = m.decode_tdt(&joint, &params) else {
            panic!("expected InvalidArgument on NaN");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("NaN"), "must name the NaN: {msg}");
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    /// The 0.6B-v3 duration list is exposed as a REFERENCE only. This pins
    /// its value (so a drift is caught) while documenting that it is not a
    /// default anywhere — `TdtDecodeParams` has no constructor that
    /// supplies bins on the caller's behalf.
    #[test]
    fn reference_duration_bins_are_pinned_but_never_defaulted() {
        assert_eq!(PARAKEET_TDT_0_6B_V3_REFERENCE_DURATIONS, [0, 1, 2, 3, 4]);
        // The reference list is well-formed for the primitive (non-empty,
        // at least one non-zero) — but the caller must pass it explicitly.
        let params =
            TdtDecodeParams::nemo_defaults(1, 3, PARAKEET_TDT_0_6B_V3_REFERENCE_DURATIONS.to_vec());
        assert_eq!(params.duration_bins.len(), 5);
        assert_eq!(params.joint_frame_stride(), 4 + 5);
    }
}
