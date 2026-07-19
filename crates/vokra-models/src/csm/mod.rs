//! Sesame CSM-1B — native S2S speech-generation model (M4-05, FR-MD-08).
//!
//! # What CSM is (ADR M4-05 §D1-(b), upstream-verified)
//!
//! CSM (Conversational Speech Model, Sesame AI Labs, Apache 2.0 code +
//! weight) is a **speech generation** model conditioned on a dialog context
//! of text + audio plus a speaker id and the reply text. It is *not* an
//! ASR and does *not* generate reply text — the caller supplies
//! `DialogRequest::reply_text` (an upstream LLM / human), and CSM speaks
//! it in context. Whisper-ASR + text-LLM + CSM is the upstream-recommended
//! pipeline shape, composed above this engine.
//!
//! # Architecture (whisper.cpp-style native re-implementation)
//!
//! Two Llama-3.2-flavor transformers over Mimi RVQ audio tokens
//! (ADR M4-05 §D2 — every value transcribed from `SesameAILabs/csm`
//! `models.py` / `generator.py`, never invented):
//!
//! - [`CsmBackbone`] — `llama3_2_1B` flavor; one sequence position = one
//!   33-slot frame (32 audio codebooks + 1 text slot, masked-sum
//!   embedding); GQA + Llama-3 **scaled** RoPE ([`rope`]) + SwiGLU;
//!   multi-stream **paged KV** (M3-03, `BlockSize::Two` — FR-EX-03);
//!   `codebook0_head` samples the zeroth codebook directly.
//! - [`CsmDepthTransformer`] — `llama3_2_100M` flavor (upstream呼称
//!   `decoder`; FR-MD-08 internal term **depth transformer**);
//!   per-frame codebook-axis autoregression, reset every frame.
//! - [`CsmModel`] — the combined frame generator
//!   (backbone step → c0 sample → depth → RVQ frame; EOS = all-zero
//!   frame).
//!
//! The RVQ codes decode to PCM through `vokra_ops::mimi_rvq`
//! (codes → features, M3-06/M4-04) plus the shared Mimi neural decoder
//! (`crate::mimi`, features → 24 kHz PCM — T31〜T34; consumed by both CSM
//! and M4-06 Moshi).
//!
//! # Honest loading state (FR-EX-08)
//!
//! Real-checkpoint weight binding is gated on the T29 owner hand-off
//! (tensor manifest). Until then every `from_gguf` weight path returns
//! [`vokra_core::VokraError::NotImplemented`] and the deterministic
//! synthesized fixtures (`SplitMix64` + Xavier) drive shape / stability /
//! property tests — never a silent zero-fill.

pub mod aec_front;
pub mod audio;
pub mod backbone;
pub mod config;
pub mod depth;
pub mod engine;
pub mod frame;
pub mod rope;
pub mod session_cuda;
pub mod session_metal;
pub mod streaming;
pub mod tokenizer;

pub use aec_front::{AecFront, EchoPath};
pub use audio::{CsmAudioDecodeChain, CsmAudioDecodeState};
pub use backbone::{
    CSM_FROM_GGUF_DEFAULT_SEED, CsmBackbone, CsmBackboneState, CsmBackboneWeights, CsmFrame,
};
pub use config::{CsmConfig, CsmRopeScaling, CsmTransformerConfig};
pub use depth::{CsmDepthState, CsmDepthTransformer, CsmDepthWeights};
pub use engine::{CsmEngine, pad_to_whole_frames};
pub use frame::{CsmFrameKind, CsmGenerationState, CsmModel};
#[cfg(all(feature = "cuda", any(unix, windows)))]
pub use session_cuda::CsmCudaDecodeSession;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use session_metal::CsmMetalDecodeSession;
pub use streaming::{CsmInterruptHandle, CsmStream, CsmStreamConfig, CsmStreamStop};
pub use tokenizer::{CsmTextTokenizer, FixtureByteTokenizer, GgufCsmTokenizer};

/// Probes whether `backend` can host the CSM hot-op set through the
/// Compute seam — the runtime-checkable face of the T21/T22 GPU sessions
/// for builds where the session types do not exist (feature off). A
/// disabled feature / absent device / coverage gap is an explicit error
/// (FR-EX-08 — the M3-10 off-GPU negative-test band pattern).
///
/// # Errors
///
/// [`vokra_core::VokraError::BackendUnavailable`] /
/// [`vokra_core::VokraError::UnsupportedOp`] verbatim from the seam.
pub fn gpu_backend_probe(backend: vokra_core::BackendKind) -> vokra_core::Result<()> {
    crate::compute::Compute::for_backend(backend, backbone::CSM_HOT_OPS).map(|_| ())
}

/// `vokra.model.arch` a CSM GGUF must carry. Written by
/// `vokra-convert::models::csm::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `sesame-csm` / `csm-1b` as
/// `Permissive` (Apache 2.0 / Apache 2.0 — docs/license-audit.md), so a
/// stock CSM GGUF passes the M2-13 gate without a research flag.
pub const EXPECTED_ARCH: &str = "csm";
