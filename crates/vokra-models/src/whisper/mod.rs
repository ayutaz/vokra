//! Whisper base — native encoder / decoder / beam search (M0-06).
//!
//! whisper.cpp-style native implementation: the model *definition* lives here
//! and only the upstream **checkpoint** is consumed, converted offline to
//! GGUF by `vokra-convert` (M0-03). No ONNX graph is loaded at runtime
//! (FR-LD-05, permanent). Hyperparameters come from the `vokra.*` GGUF
//! metadata, never hard-coded (FR-LD-02 / FR-MD-02).
//!
//! # Layout (M0-06)
//!
//! - [`config`] — [`WhisperConfig`], read from `vokra.whisper.*` metadata;
//! - [`weights`] — GGUF tensors bound to typed weight structs (owned f32; the
//!   `unsafe`-free reason is documented there);
//! - [`mel`] — the PCM → log-mel front-end (reuses the `vokra-ops` STFT + mel
//!   filter bank);
//! - [`nn`] — small forward helpers (linear / layer-norm / attention) built on
//!   the M0-08 `vokra-backend-cpu` kernels;
//! - [`encoder`] — conv stem + positional embedding + self-attention stack;
//! - [`decoder`] — token/positional embedding + causal self-attention (KV
//!   cache) + cross-attention + tied logits head;
//! - [`tokenizer`] — id ↔ text (byte-level BPE) for detokenization;
//! - [`greedy`] — greedy decode loop (special-token prefix, stop condition);
//! - [`asr`] — the [`vokra_core::engines::AsrEngine`] wired to
//!   `session.asr().transcribe()`.
//!
//! Search (`beam_search`) itself is model-independent and lives in
//! [`vokra_core::decode`]; this module supplies a `BeamScorer` from the
//! decoder (see [`decoder`]).
//!
//! # Operator inventory and gap analysis (M0-06-T02/T03)
//!
//! Every operator Whisper base needs was already available, so **no new
//! `vokra-ops` op or M0-08 kernel had to be added** — the gap list is empty:
//!
//! | need | provided by |
//! |------|-------------|
//! | STFT, mel filter bank | `vokra-ops` (M0-04): [`vokra_ops::stft()`], [`vokra_ops::mel_filterbank`] |
//! | matmul / linear (bias) | `vokra-backend-cpu` (M0-08) `gemm_f32` |
//! | softmax, layer-norm | `vokra-backend-cpu` `softmax_f32`, `layer_norm_f32` |
//! | exact (erf) GELU | `vokra-backend-cpu` `gelu_f32` |
//! | conv1d (stem) | `vokra-backend-cpu` `conv1d_f32` (im2col + GEMM) |
//! | residual add | `vokra-backend-cpu` `add_f32` |
//! | embedding lookup, transpose, head split | plain indexing in [`nn`] / [`decoder`] (memory-bound, intentionally not kernels — M0-08 boundary note) |
//! | log-mel post-processing (log10 / clamp / range) | [`mel`] (Whisper-specific, not a general op) |
//! | causal / cross attention, KV cache, logits head | assembled here from the above |
//! | beam search | [`vokra_core::decode::beam_search()`] (host-side, FR-OP-40) |
//!
//! The Whisper-specific `k_proj`-has-no-bias detail and the tied logits head
//! are handled in [`weights`] / [`decoder`], not as new ops.
//!
//! # Scope boundary
//!
//! - whisper.cpp-style native reimplementation: only the upstream safetensors
//!   checkpoint is consumed (FR-MD-02 / IF-06); no ONNX at runtime (FR-LD-05);
//! - the KV cache is a **model-internal** detail here; promoting it to a
//!   first-class session state (FR-EX-02) is M1-04;
//! - `frontend_spec` bit-exact **checking** (FR-LD-03) landed in **M1-03**:
//!   [`WhisperModel::from_gguf`] validates the `vokra.frontend.*` chunk via
//!   [`mel::check_frontend_spec`]; `resample` (FR-OP-04) is M1-06 and the input
//!   is still expected to already be at the model sample rate here;
//! - word-level timestamps are a `beam_search` attribute (FR-OP-40) but not
//!   implemented in M0 (WP completion = demo + parity).

pub mod asr;
pub mod beam_glue;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod greedy;
pub mod mel;
pub mod nn;
/// Reusable per-forward scratch buffers (FR-EX-05, hot-path malloc elimination);
/// internal to the whisper module.
// `pub(crate)`: the Voxtral audio tower (`crate::voxtral::audio_encoder`)
// reuses `EncoderScratch` + `encoder_block` for its (Whisper-identical)
// pre-norm stack — same audited scratch discipline, one implementation.
pub(crate) mod scratch;
pub mod session;
pub mod tokenizer;
pub mod weights;

pub use asr::WhisperAsr;
pub use config::WhisperConfig;
pub use session::WhisperSession;
pub use tokenizer::WhisperTokenizer;
pub use weights::{QuantBindReport, WhisperLoadOptions, WhisperWeights};

#[cfg(feature = "coreml")]
pub use vokra_backend_coreml::{CoreMlArtifact, CoreMlBackend, CoreMlComputePrecision};

use std::sync::Arc;

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{
    BackendKind, DelegateBackend, DelegateSubmodel, FrontendPolicy, Result, Tensor, VokraError,
};

use crate::compute::{Compute, HotOp};
use encoder::EncoderOutput;

/// Every `vokra.model.arch` value the Whisper binder legitimately serves.
///
/// Whisper is one of the few binders that **correctly** accepts more than
/// one arch tag: four distinct upstream release families share the vanilla
/// Whisper tensor topology, the `vokra.whisper.*` hparam schema and the
/// detokenizer verbatim, differing only in provenance / license / decoder
/// depth. All four therefore load through the same code path:
///
/// - `whisper` — vanilla `openai/whisper` (MIT).
///   (`crates/vokra-convert/src/models/whisper.rs::ARCH`)
/// - `crisper-whisper` — `nyrahealth/CrisperWhisper`, a large-v3
///   verbatim-word-timestamps fine-tune (cc-by-nc-4.0). Byte-identical
///   architecture to whisper-large-v3; only the stamp, license class and
///   provenance differ.
///   (`…/whisper.rs::ARCH_CRISPERWHISPER`, via `WhisperVariant`)
/// - `distil-whisper` — `distil-whisper/distil-large-v3.5`. Architecturally
///   a Whisper checkpoint whose only difference is
///   `n_text_layer < n_audio_layer`; [`crate::distil_whisper::DistilWhisperAsr::from_gguf`]
///   delegates the whole load to [`WhisperAsr::from_gguf`] and then enforces
///   that distil invariant on top.
///   (`crates/vokra-convert/src/models/distil_whisper.rs::ARCH`)
/// - `kotoba-whisper` — `kotoba-tech/kotoba-whisper-v2.0` (Apache-2.0),
///   the same shallow-decoder shape distilled on a Japanese corpus.
///   [`crate::kotoba_whisper::KotobaWhisperAsr::from_gguf`] delegates
///   identically.
///   (`crates/vokra-convert/src/models/kotoba_whisper.rs::ARCH`)
///
/// Deliberately **excluded**: `whisper-medusa-v1`
/// (`crates/vokra-convert/src/models/whisper_medusa_v1.rs::ARCH`) — it
/// carries extra Medusa residual heads owned by
/// [`crate::whisper_medusa::WhisperMedusa`]. Admitting it here would bypass
/// that strict binder and silently drop the official module-0 output
/// transform.
///
/// These strings mirror the converter constants — the converter owns the
/// writer contract, this module owns the reader contract (the deliberate
/// two-copies convention [`crate::pyannote`] documents; a compile-time
/// check would need `vokra-convert` in `vokra-models`'s dependency graph,
/// which the workspace pins forbid).
pub const ACCEPTED_ARCHS: &[&str] = &[
    "whisper",
    "crisper-whisper",
    "distil-whisper",
    "kotoba-whisper",
];

/// Rejects a GGUF whose `vokra.model.arch` is absent or is not one of
/// [`ACCEPTED_ARCHS`].
///
/// A *loud* validation step (FR-EX-08): the Whisper weight binder matches
/// on HF-verbatim tensor names, so a foreign checkpoint that happens to
/// share some of them would bind a partial model and transcribe noise.
///
/// # Scope note (follow-up)
///
/// This is currently called from [`WhisperSession::from_gguf`] /
/// [`WhisperSession::from_gguf_on`]. Extending it to
/// [`WhisperModel::from_gguf`] and [`WhisperAsr::from_gguf`] — the paths
/// `vokra-cli` / `vokra-capi` / `vokra-server` take — is a follow-up that
/// also has to stamp the synthetic fixtures in
/// [`crate::distil_whisper`]'s and [`crate::kotoba_whisper`]'s delegation
/// tests, which today build unstamped shape-only GGUFs and assert on the
/// *front-end chunk* message they currently get.
pub(crate) fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if ACCEPTED_ARCHS.contains(&a) => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "whisper: GGUF arch is `{other}`, expected one of {ACCEPTED_ARCHS:?}. Those four \
             families share the vanilla Whisper topology + `vokra.whisper.*` schema verbatim \
             and are the only ones this binder serves. `whisper-medusa-v1` is deliberately \
             NOT accepted (its Medusa heads belong to `whisper_medusa::WhisperMedusa`; \
             binding it here would silently drop the module-0 output transform). Any other arch would bind whatever \
             HF-verbatim tensor names happen to overlap and transcribe noise (FR-EX-08 — no \
             silent partial load)."
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "whisper: GGUF is missing `{}` — this is not a Vokra-native Whisper GGUF (was it \
             produced by `vokra-cli convert --model whisper`?)",
            chunks::KEY_MODEL_ARCH,
        ))),
    }
}

/// The backend hot ops the Whisper forward dispatches. Unlike CAM++ / piper
/// (GEMM only), Whisper also routes softmax / layer-norm / GELU / conv1d / GEMV
/// through the backend, so a backend must cover **all six** to run Whisper. The
/// Metal covers all six through the imperative compute seam, including the
/// beam/sample replay and word-alignment second forward. A backend missing any
/// required op is rejected explicitly — never a silent CPU fall back.
pub(crate) const WHISPER_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
];

/// A loaded Whisper model: validated config plus bound weights.
///
/// Construct with [`WhisperModel::from_gguf`]. The high-level transcription
/// entry point is [`WhisperAsr`] (the [`AsrEngine`](vokra_core::AsrEngine)
/// implementation); this type exposes the encoder / decoder forwards used by
/// the parity tests and by the search integration.
pub struct WhisperModel {
    config: WhisperConfig,
    weights: WhisperWeights,
}

/// Same-model, same-feature CPU/delegate Whisper encoder measurement.
///
/// `cpu_seconds` and `delegate_seconds` contain one sample per measured
/// iteration. Model loading, delegate compilation, and log-mel extraction are
/// excluded: callers bind one delegate session first, this helper warms both
/// paths, then alternates their order to reduce thermal/order bias. Numerical
/// error is accumulated across every measured output element and iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperEncoderBakeoff {
    /// Delegate identity reported by [`DelegateBackend::delegate_name`].
    pub delegate_name: String,
    /// Timed Rust CPU encoder samples, in seconds.
    pub cpu_seconds: Vec<f64>,
    /// Timed delegate encoder samples, in seconds.
    pub delegate_seconds: Vec<f64>,
    /// Largest absolute CPU/delegate output difference across all iterations.
    pub max_abs_error: f32,
    /// Mean absolute CPU/delegate output difference across all iterations.
    pub mean_abs_error: f64,
    /// Total output values compared (`n_audio_ctx * d_model * iterations`).
    pub compared_values: usize,
    /// Comparison tolerance supplied by the caller.
    pub comparison_atol: f32,
    /// Number of compared values whose absolute error exceeded `comparison_atol`.
    pub values_over_atol: usize,
    /// Measured iteration containing the maximum error.
    pub max_error_iteration: usize,
    /// Flat `[n_audio_ctx, d_model]` index containing the maximum error.
    pub max_error_index: usize,
    /// CPU oracle value at `max_error_index`.
    pub max_error_cpu_value: f32,
    /// Delegate value at `max_error_index`.
    pub max_error_delegate_value: f32,
}

impl WhisperModel {
    /// Loads config (`vokra.whisper.*`) and every weight tensor from `file`.
    ///
    /// # Front-end check (FR-LD-03, M1-03)
    ///
    /// After the config is read, the model's declared `vokra.frontend.*` chunk
    /// is validated bit-for-bit against the runtime Whisper front-end
    /// ([`mel::runtime_frontend_spec`]) under the default
    /// [`FrontendPolicy::Fail`](vokra_core::FrontendPolicy) — a mismatched or
    /// missing chunk aborts the load *before* the (larger) weight tensors are
    /// bound. Use [`mel::check_frontend_spec`] directly for a lenient
    /// (`Warn`) load.
    ///
    /// # Errors
    ///
    /// [`vokra_core::VokraError::ModelLoad`] if a hyperparameter key or a weight tensor is
    /// missing, mistyped or mis-shaped, or the `vokra.frontend.*` chunk is
    /// absent; [`vokra_core::VokraError::FrontendMismatch`] if the
    /// declared front-end differs from the runtime's.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with(file, WhisperLoadOptions::default())
    }

    /// [`from_gguf`](Self::from_gguf) with the M5-15 fused-quant load options.
    ///
    /// With [`WhisperLoadOptions::fused_quant_weights`], K-quantized
    /// projections keep their super-blocks and run the fused INT8 kernels.
    /// This is **CPU-only** and **not bit-identical** to the dequant path — see
    /// the option's docs and `docs/adr/M5-15-quant.md`. [`Self::quant_report`]
    /// says how many weights actually took each route.
    ///
    /// # Errors
    ///
    /// As [`from_gguf`](Self::from_gguf).
    pub fn from_gguf_with(file: &GgufFile, opts: WhisperLoadOptions) -> Result<Self> {
        let config = WhisperConfig::from_gguf(file)?;
        // Whisper declares a front-end chunk; check it bit-exact before the
        // heavier weight load. VAD / piper-plus loaders deliberately skip this
        // (they write no `vokra.frontend.*`) — the gating is per-model, by caller.
        mel::check_frontend_spec(file, config.n_mels, FrontendPolicy::Fail)?;
        let weights = WhisperWeights::load_with(file, &config, opts)?;
        Ok(Self { config, weights })
    }

    /// What the fused-quant binding did on this load (all-zero for a default
    /// [`from_gguf`](Self::from_gguf)).
    pub fn quant_report(&self) -> QuantBindReport {
        self.weights.quant_report()
    }

    /// The model hyperparameters.
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Runs the log-mel front-end on mono `pcm` at the model sample rate.
    ///
    /// Returns the `[n_mels, n_frames]` log-mel features (row-major). See
    /// [`mel::log_mel`] for the algorithm and its parity guarantees.
    pub fn log_mel(&self, pcm: &[f32]) -> Vec<f32> {
        mel::log_mel(pcm, self.config.n_mels)
    }

    /// Encodes `[n_mels, n_frames]` log-mel features into the encoder hidden
    /// states `[n_audio_ctx, d_model]` on the CPU backend.
    pub fn encode(&self, log_mel: &[f32], n_frames: usize) -> Result<EncoderOutput> {
        self.encode_with(&Compute::cpu(), log_mel, n_frames)
    }

    /// [`encode`](Self::encode) on an explicit [`Compute`] (M2-01 Phase 3). The
    /// CPU dispatcher reproduces the pre-seam kernel calls bit-for-bit.
    pub fn encode_with(
        &self,
        compute: &Compute,
        log_mel: &[f32],
        n_frames: usize,
    ) -> Result<EncoderOutput> {
        encoder::encode(
            compute,
            &self.config,
            &self.weights.encoder,
            log_mel,
            n_frames,
        )
    }

    /// Executes the complete Whisper encoder through a declared submodel
    /// delegate and validates its data contract before exposing the result to
    /// the decoder.
    ///
    /// This is intentionally separate from [`Compute`]'s per-op surface. The
    /// delegate must claim and execute [`DelegateSubmodel::WhisperEncoder`] as
    /// one indivisible graph. Missing support, wrong arity, wrong shape, or a
    /// non-f32 output is an explicit error; this method never falls back to the
    /// Rust CPU encoder.
    pub fn encode_with_delegate(
        &self,
        delegate: &dyn DelegateBackend,
        log_mel: &[f32],
        n_frames: usize,
    ) -> Result<EncoderOutput> {
        let expected_frames = self.config.n_audio_ctx.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument(
                "whisper delegate input frame count overflows usize".to_owned(),
            )
        })?;
        if n_frames != expected_frames {
            return Err(VokraError::InvalidArgument(format!(
                "whisper delegate requires the fixed full encoder window of {expected_frames} frames, got {n_frames}"
            )));
        }
        let expected_input_len = self.config.n_mels.checked_mul(n_frames).ok_or_else(|| {
            VokraError::InvalidArgument(
                "whisper delegate input element count overflows usize".to_owned(),
            )
        })?;
        if log_mel.len() != expected_input_len {
            return Err(VokraError::InvalidArgument(format!(
                "whisper delegate log-mel len {} != n_mels*n_frames {expected_input_len}",
                log_mel.len()
            )));
        }
        if !delegate.supports_submodel(DelegateSubmodel::WhisperEncoder) {
            return Err(VokraError::UnsupportedOp(format!(
                "{} does not support the complete WhisperEncoder submodel (no silent CPU fallback, FR-EX-08)",
                delegate.delegate_name()
            )));
        }
        let input = Tensor::host_f32(vec![1, self.config.n_mels, n_frames], log_mel.to_vec())?;
        let inputs = [&input];
        let mut outputs = delegate.execute_submodel(DelegateSubmodel::WhisperEncoder, &inputs)?;
        if outputs.len() != 1 {
            return Err(VokraError::ModelLoad(format!(
                "{} WhisperEncoder returned {} outputs, expected exactly one",
                delegate.delegate_name(),
                outputs.len()
            )));
        }
        let output = outputs.pop().expect("length checked");
        let expected_shape = [1, self.config.n_audio_ctx, self.config.d_model];
        if output.shape.as_slice() != expected_shape {
            return Err(VokraError::ModelLoad(format!(
                "{} WhisperEncoder output shape {:?} != {expected_shape:?}",
                delegate.delegate_name(),
                output.shape
            )));
        }
        Ok(EncoderOutput {
            hidden: output.as_f32()?.to_vec(),
            n_ctx: self.config.n_audio_ctx,
            d_model: self.config.d_model,
        })
    }

    /// Measures the CPU encoder against one already-bound whole-submodel
    /// delegate on identical log-mel features.
    ///
    /// This is the hardware bakeoff seam used by CoreML/ANE and QNN/Hexagon.
    /// It deliberately accepts a live delegate rather than constructing one,
    /// so load/compile time is outside the samples and every iteration uses the
    /// same model and delegate session. The measured order alternates between
    /// CPU-first and delegate-first. No backend fallback exists in this path.
    pub fn bakeoff_encoder_delegate(
        &self,
        delegate: &dyn DelegateBackend,
        log_mel: &[f32],
        n_frames: usize,
        warmup: usize,
        iterations: usize,
        atol: f32,
    ) -> Result<WhisperEncoderBakeoff> {
        if warmup == 0 {
            return Err(VokraError::InvalidArgument(
                "whisper delegate bakeoff requires at least one warm-up iteration".to_owned(),
            ));
        }
        if iterations == 0 {
            return Err(VokraError::InvalidArgument(
                "whisper delegate bakeoff requires at least one measured iteration".to_owned(),
            ));
        }
        if !atol.is_finite() || atol < 0.0 {
            return Err(VokraError::InvalidArgument(
                "whisper delegate bakeoff atol must be finite and non-negative".to_owned(),
            ));
        }

        for _ in 0..warmup {
            let cpu = self.encode(log_mel, n_frames)?;
            std::hint::black_box(&cpu.hidden);
            let delegated = self.encode_with_delegate(delegate, log_mel, n_frames)?;
            std::hint::black_box(&delegated.hidden);
        }

        let mut cpu_seconds = Vec::with_capacity(iterations);
        let mut delegate_seconds = Vec::with_capacity(iterations);
        let mut max_abs_error = 0.0f32;
        let mut sum_abs_error = 0.0f64;
        let mut compared_values = 0usize;
        let mut values_over_atol = 0usize;
        let mut max_error_iteration = 0usize;
        let mut max_error_index = 0usize;
        let mut max_error_cpu_value = 0.0f32;
        let mut max_error_delegate_value = 0.0f32;

        for iteration in 0..iterations {
            let timed_cpu = || -> Result<(EncoderOutput, f64)> {
                let started = std::time::Instant::now();
                let output = self.encode(log_mel, n_frames)?;
                let seconds = started.elapsed().as_secs_f64();
                std::hint::black_box(&output.hidden);
                Ok((output, seconds))
            };
            let timed_delegate = || -> Result<(EncoderOutput, f64)> {
                let started = std::time::Instant::now();
                let output = self.encode_with_delegate(delegate, log_mel, n_frames)?;
                let seconds = started.elapsed().as_secs_f64();
                std::hint::black_box(&output.hidden);
                Ok((output, seconds))
            };
            let ((cpu, cpu_elapsed), (delegated, delegate_elapsed)) = if iteration % 2 == 0 {
                (timed_cpu()?, timed_delegate()?)
            } else {
                let delegated = timed_delegate()?;
                let cpu = timed_cpu()?;
                (cpu, delegated)
            };
            cpu_seconds.push(cpu_elapsed);
            delegate_seconds.push(delegate_elapsed);

            if cpu.hidden.len() != delegated.hidden.len() {
                return Err(VokraError::ModelLoad(format!(
                    "{} Whisper encoder output length {} != CPU length {}",
                    delegate.delegate_name(),
                    delegated.hidden.len(),
                    cpu.hidden.len()
                )));
            }
            for (index, (&cpu_value, &delegate_value)) in
                cpu.hidden.iter().zip(&delegated.hidden).enumerate()
            {
                if !cpu_value.is_finite() || !delegate_value.is_finite() {
                    return Err(VokraError::ModelLoad(format!(
                        "{} Whisper encoder bakeoff produced a non-finite CPU/delegate value",
                        delegate.delegate_name()
                    )));
                }
                let difference = (cpu_value - delegate_value).abs();
                if difference > max_abs_error {
                    max_abs_error = difference;
                    max_error_iteration = iteration;
                    max_error_index = index;
                    max_error_cpu_value = cpu_value;
                    max_error_delegate_value = delegate_value;
                }
                if difference > atol {
                    values_over_atol += 1;
                }
                sum_abs_error += f64::from(difference);
            }
            compared_values = compared_values
                .checked_add(cpu.hidden.len())
                .ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "whisper delegate bakeoff comparison count overflowed usize".to_owned(),
                    )
                })?;
        }

        Ok(WhisperEncoderBakeoff {
            delegate_name: delegate.delegate_name().to_owned(),
            cpu_seconds,
            delegate_seconds,
            max_abs_error,
            mean_abs_error: sum_abs_error / compared_values as f64,
            compared_values,
            comparison_atol: atol,
            values_over_atol,
            max_error_iteration,
            max_error_index,
            max_error_cpu_value,
            max_error_delegate_value,
        })
    }

    /// Convenience: PCM → log-mel → encoder hidden states (CPU backend).
    pub fn encode_pcm(&self, pcm: &[f32]) -> Result<EncoderOutput> {
        self.encode_pcm_with(&Compute::cpu(), pcm)
    }

    /// [`encode_pcm`](Self::encode_pcm) on an explicit [`Compute`] — the entry
    /// [`WhisperAsr`] uses to run the encoder on the selected backend.
    pub fn encode_pcm_with(&self, compute: &Compute, pcm: &[f32]) -> Result<EncoderOutput> {
        let n_frames = mel::N_FRAMES;
        let feats = self.log_mel(pcm);
        self.encode_with(compute, &feats, n_frames)
    }

    /// Creates a decoder run bound to `encoder`, with fresh KV caches, on the
    /// CPU backend. Used by the greedy / beam drivers and by the decoder parity
    /// tests.
    ///
    /// Takes `&Arc<Self>` and clones the `Arc` into the returned
    /// [`DecoderState`](decoder::DecoderState), which therefore owns the model
    /// and carries no lifetime (so it is `Send` and can outlive this borrow).
    pub fn decoder(self: &Arc<Self>, encoder: &EncoderOutput) -> Result<decoder::DecoderState> {
        decoder::DecoderState::new(Arc::clone(self), encoder)
    }

    /// [`decoder`](Self::decoder) on an explicit backend (M2-01 Phase 3). On
    /// the CPU backend this is identical to [`decoder`](Self::decoder); Metal
    /// covers the complete Whisper hot-op set.
    pub fn decoder_with_backend(
        self: &Arc<Self>,
        encoder: &EncoderOutput,
        backend_kind: BackendKind,
    ) -> Result<decoder::DecoderState> {
        decoder::DecoderState::new_with_backend(Arc::clone(self), encoder, backend_kind)
    }

    /// Borrows the decoder weights / config for the [`decoder`] forward and the
    /// [`greedy`] / search drivers.
    pub(crate) fn decoder_state(&self) -> (&WhisperConfig, &weights::DecoderWeights) {
        (&self.config, &self.weights.decoder)
    }

    /// Test-only constructor from already-built parts, so the synthetic decoder
    /// tests can assemble a tiny model without a GGUF fixture.
    #[cfg(test)]
    pub(crate) fn new_for_test(config: WhisperConfig, weights: WhisperWeights) -> Self {
        Self { config, weights }
    }
}

#[cfg(test)]
mod delegate_tests {
    use super::*;

    struct FixedDelegate {
        output_shape: Vec<usize>,
        supported: bool,
    }

    impl DelegateBackend for FixedDelegate {
        fn delegate_name(&self) -> &str {
            "fixed-test-delegate"
        }

        fn supports_submodel(&self, submodel: DelegateSubmodel) -> bool {
            self.supported && matches!(submodel, DelegateSubmodel::WhisperEncoder)
        }

        fn execute_submodel(
            &self,
            submodel: DelegateSubmodel,
            inputs: &[&Tensor],
        ) -> Result<Vec<Tensor>> {
            assert_eq!(submodel, DelegateSubmodel::WhisperEncoder);
            assert_eq!(inputs.len(), 1);
            assert_eq!(inputs[0].shape, vec![1, 80, 8]);
            let len = self.output_shape.iter().product();
            Ok(vec![Tensor::host_f32(
                self.output_shape.clone(),
                (0..len).map(|index| index as f32).collect(),
            )?])
        }
    }

    #[test]
    fn whole_encoder_delegate_contract_is_data_carrying_and_fail_loud() {
        let model = decoder::test_support::tiny_model(1);
        let log_mel = vec![0.25; 80 * 8];
        let delegate = FixedDelegate {
            output_shape: vec![1, 4, 2],
            supported: true,
        };
        let output = model
            .encode_with_delegate(&delegate, &log_mel, 8)
            .expect("declared whole encoder delegate");
        assert_eq!(output.n_ctx, 4);
        assert_eq!(output.d_model, 2);
        assert_eq!(
            output.hidden,
            (0..8).map(|index| index as f32).collect::<Vec<_>>()
        );

        let wrong_shape = FixedDelegate {
            output_shape: vec![1, 2, 4],
            supported: true,
        };
        let err = model
            .encode_with_delegate(&wrong_shape, &log_mel, 8)
            .expect_err("layout-compatible element count with wrong axes must fail");
        assert!(matches!(err, VokraError::ModelLoad(_)));

        let unsupported = FixedDelegate {
            output_shape: vec![1, 4, 2],
            supported: false,
        };
        let err = model
            .encode_with_delegate(&unsupported, &log_mel, 8)
            .expect_err("unsupported delegate must not invoke the CPU encoder");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    #[test]
    fn encoder_bakeoff_uses_identical_features_and_reports_numerical_error() {
        let model = decoder::test_support::tiny_model(1);
        let log_mel = vec![0.25; 80 * 8];
        let expected = model.encode(&log_mel, 8).expect("CPU oracle");
        struct OracleDelegate {
            hidden: Vec<f32>,
        }
        impl DelegateBackend for OracleDelegate {
            fn delegate_name(&self) -> &str {
                "oracle-delegate"
            }

            fn supports_submodel(&self, submodel: DelegateSubmodel) -> bool {
                matches!(submodel, DelegateSubmodel::WhisperEncoder)
            }

            fn execute_submodel(
                &self,
                _submodel: DelegateSubmodel,
                _inputs: &[&Tensor],
            ) -> Result<Vec<Tensor>> {
                Ok(vec![Tensor::host_f32(vec![1, 4, 2], self.hidden.clone())?])
            }
        }
        let delegate = OracleDelegate {
            hidden: expected.hidden,
        };
        let report = model
            .bakeoff_encoder_delegate(&delegate, &log_mel, 8, 1, 2, 0.01)
            .expect("same-feature bakeoff");
        assert_eq!(report.delegate_name, "oracle-delegate");
        assert_eq!(report.cpu_seconds.len(), 2);
        assert_eq!(report.delegate_seconds.len(), 2);
        assert_eq!(report.compared_values, 16);
        assert_eq!(report.max_abs_error, 0.0);
        assert_eq!(report.mean_abs_error, 0.0);
        assert_eq!(report.values_over_atol, 0);

        let err = model
            .bakeoff_encoder_delegate(&delegate, &log_mel, 8, 1, 0, 0.01)
            .expect_err("zero measured iterations must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}

/// Public-surface `whisper::quant_load` tests (spec test path — M2-08 T07 / c06).
///
/// The unit-level coverage lives in [`session`]; this module mounts the
/// integration-shaped assertions at `whisper::quant_load::*` so the spec's
/// exact `cargo test -p vokra-models whisper::quant_load` filter selects
/// them.
#[cfg(test)]
mod quant_load {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_core::quant::{QuantPolicy, QuantScheme};
    use vokra_core::{BackendKind, VokraError};

    /// Builds a GGUF carrying a valid `vokra.whisper.*` hyperparameter chunk
    /// (no front-end, no weights) — enough for `WhisperModel::from_gguf` to
    /// reach the front-end check (which then fails on the missing chunk).
    /// The session ctor's *quant* gate fires **before** the model load only
    /// if we skip weights and want to observe policy loading in isolation,
    /// but c06 is scoped to the session ctor which runs the model load
    /// first; so the public surface test we run is a compilable-shape check
    /// on the constructor error type — the deep behaviour is covered under
    /// [`session::quant_load`].
    ///
    /// Carries the `vokra.model.arch` stamp: the session ctor gates arch
    /// before the model load (FR-EX-08), so an unstamped fixture would
    /// short-circuit there and the ordering assertion below would pass for
    /// the wrong reason.
    fn write_valid_config(b: &mut GgufBuilder) {
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        b.add_u32("vokra.whisper.n_mels", 80);
        b.add_u32("vokra.whisper.n_audio_ctx", 1500);
        b.add_u32("vokra.whisper.n_audio_state", 512);
        b.add_u32("vokra.whisper.n_audio_head", 8);
        b.add_u32("vokra.whisper.n_audio_layer", 6);
        b.add_u32("vokra.whisper.n_text_ctx", 448);
        b.add_u32("vokra.whisper.n_text_state", 512);
        b.add_u32("vokra.whisper.n_text_head", 8);
        b.add_u32("vokra.whisper.n_text_layer", 6);
        b.add_u32("vokra.whisper.n_vocab", 51865);
        b.add_u32("vokra.whisper.ffn_dim", 2048);
        b.add_u32("vokra.whisper.eot", 50257);
        b.add_metadata(
            "vokra.whisper.decoder_start_ids",
            vokra_core::gguf::GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::U32,
                values: [50258u32, 50259, 50359, 50363]
                    .iter()
                    .map(|&id| vokra_core::gguf::GgufMetadataValue::U32(id))
                    .collect(),
            }),
        );
    }

    #[test]
    fn from_gguf_on_reports_model_load_before_touching_the_quant_gate() {
        // A config-only GGUF triggers `ModelLoad` from the front-end check
        // (weights aren't reached, quant gate isn't reached). Confirms the
        // ordering: model validation runs before the c06 activation gate,
        // so a broken model surfaces the model error, not a policy error.
        let mut b = GgufBuilder::new();
        write_valid_config(&mut b);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let result = session::WhisperSession::from_gguf_on(&file, BackendKind::Cpu);
        match result {
            Err(VokraError::ModelLoad(_)) => {}
            Err(other) => panic!("expected ModelLoad, got {other:?}"),
            Ok(_) => panic!("expected model load to fail on missing weights"),
        }
    }

    #[test]
    fn quant_policy_default_is_vocoder_safe_fp16() {
        // c06 contract, pinned at the public API: when the `vokra.quant.*`
        // chunk is absent (every GGUF today), the loaded policy is
        // vocoder-safe FP16.
        assert_eq!(
            vokra_core::quant::resolve::default_vocoder_safe().default_scheme(),
            QuantScheme::Fp16,
            "the safe default must never resolve to Int8"
        );
        // Confirm the alias is not the INT8 one.
        assert_ne!(QuantScheme::Fp16.as_str(), QuantScheme::W8A8Int8.as_str());
        // Silence unused-import warning on `QuantPolicy` when the type is
        // only referenced through its associated free-function preset above.
        let _: &'static str = std::any::type_name::<QuantPolicy>();
    }

    #[test]
    fn unsupported_quant_path_carries_op_scheme_backend() {
        // c06 error shape, FR-EX-08 audit trail — verified on the variant
        // directly so callers can special-case without string-matching.
        let err = VokraError::UnsupportedQuantPath {
            op: "whisper::gemm".to_owned(),
            scheme: "w8a8".to_owned(),
            backend: "cpu".to_owned(),
        };
        match &err {
            VokraError::UnsupportedQuantPath {
                op,
                scheme,
                backend,
            } => {
                assert_eq!(op, "whisper::gemm");
                assert_eq!(scheme, "w8a8");
                assert_eq!(backend, "cpu");
            }
            other => panic!("expected UnsupportedQuantPath, got {other:?}"),
        }
        // Display must name FR-EX-08 so log readers can trace the reject to
        // the requirement.
        assert!(err.to_string().contains("FR-EX-08"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::VokraError;
    use vokra_core::gguf::GgufBuilder;

    /// Writes a full, valid `vokra.whisper.*` hyperparameter chunk (n_mels = 80).
    fn write_valid_config(b: &mut GgufBuilder) {
        b.add_u32("vokra.whisper.n_mels", 80);
        b.add_u32("vokra.whisper.n_audio_ctx", 1500);
        b.add_u32("vokra.whisper.n_audio_state", 512);
        b.add_u32("vokra.whisper.n_audio_head", 8);
        b.add_u32("vokra.whisper.n_audio_layer", 6);
        b.add_u32("vokra.whisper.n_text_ctx", 448);
        b.add_u32("vokra.whisper.n_text_state", 512);
        b.add_u32("vokra.whisper.n_text_head", 8);
        b.add_u32("vokra.whisper.n_text_layer", 6);
        b.add_u32("vokra.whisper.n_vocab", 51865);
        b.add_u32("vokra.whisper.ffn_dim", 2048);
        b.add_u32("vokra.whisper.eot", 50257);
        b.add_metadata(
            "vokra.whisper.decoder_start_ids",
            vokra_core::gguf::GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::U32,
                values: [50258u32, 50259, 50359, 50363]
                    .iter()
                    .map(|&id| vokra_core::gguf::GgufMetadataValue::U32(id))
                    .collect(),
            }),
        );
    }

    #[test]
    fn from_gguf_aborts_on_a_mismatched_frontend_before_loading_weights() {
        // A valid config plus a front-end chunk that differs in one field. The
        // GGUF carries NO weight tensors — so reaching a FrontendMismatch (rather
        // than a weight ModelLoad) proves the front-end check runs first and the
        // wiring in `from_gguf` is live (FR-LD-03).
        let mut b = GgufBuilder::new();
        write_valid_config(&mut b);
        let mut declared = mel::runtime_frontend_spec(80);
        declared.n_fft = 512;
        declared.write_into(&mut b);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        assert!(matches!(
            WhisperModel::from_gguf(&file),
            Err(VokraError::FrontendMismatch(_))
        ));
    }

    #[test]
    fn from_gguf_reports_a_missing_frontend_chunk() {
        // Whisper requires the chunk; a config-only GGUF (no `vokra.frontend.*`)
        // fails the check as a ModelLoad, again before any weight is touched.
        let mut b = GgufBuilder::new();
        write_valid_config(&mut b);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        assert!(matches!(
            WhisperModel::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }
}
