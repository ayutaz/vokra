//! HTTP + Wyoming route modules.
//!
//! T01 skeleton only — no route wiring here yet. Four API surfaces, each in
//! its own submodule per T01 spec:
//!
//! | Module         | Compat target        | Ticket window   | Requirement |
//! |----------------|----------------------|-----------------|-------------|
//! | [`openai`]     | OpenAI Audio API     | T06 / T07 / T08 | FR-SV-02    |
//! | [`vllm`]       | vLLM OpenAI-shape    | T09 / T10       | FR-SV-03    |
//! | [`piper_http`] | piper-plus HTTP API  | T11 / T12 / T13 | FR-SV-04    |
//! | [`wyoming`]    | Wyoming JSONL/TCP    | T14–T17         | FR-SV-05    |
//!
//! # Cross-cutting invariants (apply to every surface)
//!
//! * All handlers go through the panic-catch layer landed in T05 so a
//!   handler panic becomes a 500 JSON error, never a runtime abort
//!   (NFR-RL-07).
//! * `UnsupportedOp` from the engine layer is surfaced as HTTP 501 with
//!   `type: "unsupported_op"`. No silent CPU fallback (FR-EX-08).
//! * No numeric kernels are implemented here — handlers call
//!   `service::InferenceService` and shape the request/response only.
//! * No ONNX / protobuf / gRPC (FR-LD-05, NFR-DS-02).
//! * Watermark firing is disabled at v0.5 (2026-07-04 owner drop). TTS
//!   surfaces (`piper_http`, `wyoming`) accept a `WatermarkConfig` field
//!   for forward-compat but never embed anything at v0.5.

pub mod openai;
pub mod piper_http;
pub mod vllm;
pub mod wyoming;
