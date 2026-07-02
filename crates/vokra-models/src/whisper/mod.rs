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
//! - config / weights / log-mel front-end (reusing `vokra-ops` STFT + mel);
//! - encoder (self-attention stack) and decoder (self + cross attention,
//!   KV cache) and tokenizer / detokenizer;
//! - greedy + `beam_search` (the search itself is model-independent and lives
//!   in [`vokra_core::decode`]; this module supplies the `BeamScorer`).
//!
//! The public surface is a [`vokra_core::engines::AsrEngine`] implementation
//! wired to `session.asr().transcribe()`.
//!
//! Implementation lands with M0-06 (T01–T27).
