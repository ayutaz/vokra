//! # vokra-core
//!
//! Core of **Vokra**, a speech-first inference runtime built as an ONNX /
//! ONNX Runtime alternative for speech AI (TTS / ASR / S2S / VC /
//! Speaker-ID / VAD).
//!
//! Per the crate layout recorded in `docs/adr/0001-crate-layout.md`
//! (SRS §1.3), this crate hosts the *IR and execution engine* side:
//!
//! - the **audio graph descriptor IR** ([`ir`], FR-EX-01): a ggml-style flat
//!   op enum — [`DType`], [`Dim`] (fixed / symbolic axis extents for
//!   variable-length I/O), [`TensorDesc`], [`OpKind`], [`AudioGraph`],
//!   [`GraphBuilder`];
//! - the **backend abstraction** ([`backend`]): [`Backend`] /
//!   [`BackendKind`] with uniform op coverage (FR-EX-08);
//! - the **graph evaluator** ([`runtime`]): the data-carrying [`Tensor`] and
//!   [`run_graph`], which threads real values through an [`AudioGraph`] on a
//!   backend in topological order (one graph = one backend, no silent
//!   fallback);
//! - the **decoder KV cache** ([`cache`], FR-EX-02): [`KvCache`], an ownable,
//!   `Send` key/value cache promoted out of the models so a decode can be
//!   moved across threads (the M1-08 streaming foundation), and its M3-03
//!   sibling [`PagedKvCache`] (FR-EX-03) with a `[time, stream, codebook]`
//!   3D logical address and a session-lifetime page arena that keeps the hot
//!   path free of system allocations (FR-EX-05);
//! - the **complex value type** ([`complex`], FR-EX-09): [`Complex32`], the
//!   host pair-of-`f32` behind the [`DType::Complex64`] IR dtype, shared with
//!   the audio ops and their FFT core;
//! - the **error type** [`VokraError`] and the [`Result`] alias
//!   (FR-API-02);
//! - the **public Rust API skeleton** (FR-API-02): [`Session`] /
//!   [`SessionBuilder`], the task facades [`Asr`] / [`Tts`] / [`S2s`],
//!   the declarative [`AudioPipeline`], and [`Stream`] handles.
//!
//! Speech operators live in `vokra-ops`, concrete backends in
//! `vokra-backend-*`, native model implementations in `vokra-models`, and
//! the C ABI in `vokra-capi`.
//!
//! # Design red lines (permanent constraints)
//!
//! - **No ONNX at runtime** (FR-LD-05): ONNX models are handled only by the
//!   offline conversion tool; this runtime never loads ONNX graphs and
//!   never depends on protobuf / abseil / onnx (NFR-DS-02).
//! - **No silent CPU fallback** (FR-EX-08): an op a backend does not
//!   support is an explicit error.
//! - **No NNAPI backend, ever** (FR-BE-07).
//! - **Memory safety** (NFR-RL-07, SRS §5-(1)): this crate is 100% safe
//!   Rust (`unsafe_code = "deny"` via the workspace lints). `unsafe` +
//!   SIMD intrinsics are permitted only inside operator / backend / C ABI
//!   crates, always behind safe public APIs.
//!
//! # Numeric parsing policy (NFR-RL-01)
//!
//! String-to-number conversion MUST use Rust's locale-independent
//! [`str::parse`]. The C function `strtod` — and any other
//! `LC_NUMERIC`-sensitive parser — is forbidden across the workspace: under
//! European comma-decimal locales it misparses or crashes. The guard script
//! `scripts/check-forbidden-symbols.sh` enforces this for all sources under
//! `crates/`.
//!
//! # Examples
//!
//! Session construction and task facades (FR-API-02 shapes; stubs in M0,
//! wired by M0-05/06/07):
//!
//! ```no_run
//! use vokra_core::{BackendKind, Session};
//!
//! let session = Session::from_file("model.gguf").with_backend(BackendKind::Cpu)?;
//!
//! let _text = session.asr().transcribe(&[0.0f32; 16_000]);
//! let _audio = session.tts().synthesize("hello vokra");
//! let _turn = session.s2s().dialog(&[0.0f32; 16_000]);
//!
//! let stream = session.open_stream()?;
//! println!("opened stream {}", stream.id());
//! # Ok::<(), vokra_core::VokraError>(())
//! ```
//!
//! Declarative pipeline (FR-API-02 shape, verbatim):
//!
//! ```
//! use vokra_core::AudioPipeline;
//!
//! let pipeline = AudioPipeline::new().vad().asr().llm().tts().build()?;
//! assert_eq!(pipeline.stages().len(), 4);
//! # Ok::<(), vokra_core::VokraError>(())
//! ```

pub mod backend;
pub mod cache;
pub mod complex;
pub mod compliance;
pub mod decode;
pub mod engines;
pub mod error;
pub mod gguf;
pub mod ir;
pub mod json;
pub mod kv_quant;
// M4-20 T14: reserved op-kind anchors for the M5-residual audio ops (declared,
// never registered — the KOKORO_ISTFT_HEAD_OP pattern; ADR M4-20 §D-6).
pub mod m5_residual_ops;
pub mod pipeline;
pub mod prenorm;
pub mod quant;
pub mod rng;
pub mod runtime;
pub mod safetensors;
pub mod session;
pub mod stream;
pub mod tasks;

pub use backend::{Backend, BackendKind};
pub use cache::KvCache;
pub use cache::paged::{
    AllocatorSnapshot, BlockSize, GpuPagedKvCacheOps, KvDims, KvElement, KvSlot, PageId,
    PagedKvCache, TimeRangeIter,
};
pub use cache::paged_quant::{
    AllocatorSnapshot as QuantAllocatorSnapshot, AnyBlock, QuantizedPagedKvCache,
};
pub use complex::Complex32;
pub use compliance::{
    ComplianceConfig, ComplianceLevel, CompliancePolicy, DisclosureConfig, LicenseClass,
    LicenseResolution, ResolutionSource, SpeakerEmbeddingPolicy, VoiceCloningPolicy,
    WatermarkBackendStatus, WatermarkConfig, check_weight_license, registry_lookup,
    resolve_license_class, stamp_provenance,
};
pub use decode::{
    CfgMode, DecodeStepper, LogitsSource, Sampler, SamplerConfig, TOKEN_FLAG_EOT, apply_cfg,
    apply_cfg_inplace, argmax, sample_sequence,
};
pub use engines::{AsrEngine, SynthesisRequest, TtsEngine, VadEngine, VadStreamHandle};
pub use error::{Result, VokraError};
pub use gguf::{
    FieldMismatch, FrontendPolicy, FrontendSpec, GgmlType, GgufBuilder, GgufError, GgufFile,
    GgufTensorInfo,
};
pub use ir::{AudioGraph, DType, Dim, GraphBuilder, Node, OpKind, TensorDesc, TensorId};
pub use kv_quant::{
    BlockQ4_0, BlockQ5_0, BlockQ8_0, F16Bits, KV_QUANT_BLOCK_SIZE, KvQuant, KvQuantBlock,
    KvQuantDequantGemvOps, QuantKind, dequantize_bytes, pack_slice, unpack_slice,
    validate_dequant_gemv,
};
pub use pipeline::{AudioPipeline, Pipeline, PipelineStage};
pub use prenorm::{DecoderLayerView, PrenormLayer};
pub use rng::SplitMix64;
pub use runtime::{Tensor, run_graph};
pub use safetensors::{SafeTensorInfo, SafetensorsError, SafetensorsFile};
pub use session::{Session, SessionBuilder};
pub use stream::{
    EventPoller, EventSink, InterruptHandle, RawEvent, RingConsumer, RingFull, RingProducer,
    Stream, StreamEvent, StreamState, StreamStep, channel,
};
pub use tasks::{Asr, DialogTurn, S2s, SynthesizedAudio, Transcription, Tts};
