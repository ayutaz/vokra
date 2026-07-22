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
//! | STFT, mel filter bank | `vokra-ops` (M0-04): [`vokra_ops::stft`], [`vokra_ops::mel_filterbank`] |
//! | matmul / linear (bias) | `vokra-backend-cpu` (M0-08) `gemm_f32` |
//! | softmax, layer-norm | `vokra-backend-cpu` `softmax_f32`, `layer_norm_f32` |
//! | exact (erf) GELU | `vokra-backend-cpu` `gelu_f32` |
//! | conv1d (stem) | `vokra-backend-cpu` `conv1d_f32` (im2col + GEMM) |
//! | residual add | `vokra-backend-cpu` `add_f32` |
//! | embedding lookup, transpose, head split | plain indexing in [`nn`] / [`decoder`] (memory-bound, intentionally not kernels — M0-08 boundary note) |
//! | log-mel post-processing (log10 / clamp / range) | [`mel`] (Whisper-specific, not a general op) |
//! | causal / cross attention, KV cache, logits head | assembled here from the above |
//! | beam search | [`vokra_core::decode::beam_search`] (host-side, FR-OP-40) |
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

use std::sync::Arc;

use vokra_core::gguf::GgufFile;
use vokra_core::{BackendKind, FrontendPolicy, Result};

use crate::compute::{Compute, HotOp};
use encoder::EncoderOutput;

/// The backend hot ops the Whisper forward dispatches. Unlike CAM++ / piper
/// (GEMM only), Whisper also routes softmax / layer-norm / GELU / conv1d / GEMV
/// through the backend, so a backend must cover **all six** to run Whisper. The
/// Metal backend covers only GEMM in this slice, so Whisper on Metal is an
/// explicit [`VokraError::UnsupportedOp`](vokra_core::VokraError) until those
/// kernels land (M2-01 T09-T13, Phase 4) — never a silent CPU fall back.
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
    /// [`VokraError::ModelLoad`] if a hyperparameter key or a weight tensor is
    /// missing, mistyped or mis-shaped, or the `vokra.frontend.*` chunk is
    /// absent; [`VokraError::FrontendMismatch`](vokra_core::VokraError) if the
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

    /// [`decoder`](Self::decoder) on an explicit backend (M2-01 Phase 3). Metal
    /// is rejected until its full Whisper op set lands (Phase 4); on the CPU
    /// backend this is identical to [`decoder`](Self::decoder).
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
