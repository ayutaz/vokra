//! Vokra IR — the *audio graph descriptor* (FR-EX-01).
//!
//! The MVP IR is a ggml-style flat op enum ([`OpKind`]) over a simple graph
//! container ([`AudioGraph`]); MLIR audio dialect + StableHLO are only
//! re-evaluated in v1.5+ (FR-EX-01). The IR is Vokra's own definition:
//! no ONNX graph is ever loaded at runtime (FR-LD-05, permanent constraint)
//! and no protobuf / abseil / onnx dependency exists (NFR-DS-02).

pub mod fusion;
pub mod graph;
pub mod tensor;

pub use fusion::FusedOp;
pub use graph::{AudioGraph, GraphBuilder, Node, OpKind};
pub use tensor::{DType, Dim, TensorDesc, TensorId};
