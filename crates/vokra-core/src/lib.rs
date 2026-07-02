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
//!   op enum — [`DType`], [`TensorDesc`], [`OpKind`], [`AudioGraph`],
//!   [`GraphBuilder`];
//! - the **backend abstraction** ([`backend`]): [`Backend`] /
//!   [`BackendKind`] with uniform op coverage (FR-EX-08);
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
pub mod error;
pub mod ir;
pub mod pipeline;
pub mod session;
pub mod stream;
pub mod tasks;

pub use backend::{Backend, BackendKind};
pub use error::{Result, VokraError};
pub use ir::{AudioGraph, DType, GraphBuilder, Node, OpKind, TensorDesc, TensorId};
pub use pipeline::{AudioPipeline, Pipeline, PipelineStage};
pub use session::{Session, SessionBuilder};
pub use stream::{Stream, StreamState};
pub use tasks::{Asr, DialogTurn, S2s, SynthesizedAudio, Transcription, Tts};
