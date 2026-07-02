//! piper-plus native TTS — MB-iSTFT-VITS2 (M0-07). Vokra's first native TTS.
//!
//! Native re-implementation of the piper-plus (MB-iSTFT-VITS2) inference core
//! — text encoder / duration predictor / flow / MB-iSTFT decoder — in the
//! whisper.cpp style (client decision 2026-07-02; the former wrap approach is
//! abolished, ADR-0002). The voice model is converted offline to GGUF by
//! `vokra-convert` (M0-03); no ONNX runs at runtime (FR-LD-05). G2P (8
//! languages) is bridged through the `vokra-piper-plus` crate for now.
//!
//! # Layout (M0-07)
//!
//! - model definition + GGUF load (config, phoneme table, iSTFT params);
//! - text encoder, (stochastic) duration predictor + length regulation,
//!   flow (residual coupling), MB-iSTFT decoder — the decoder is the first
//!   real consumer of the `vokra-ops` `istft` op (M0-04);
//! - a [`vokra_core::engines::TtsEngine`] implementation wired to
//!   `session.tts().synthesize()`, with deterministic (noise-off) synthesis
//!   for reference parity against piper-plus onnxruntime.
//!
//! Implementation lands with M0-07 (T06–T24); see `docs/piper-plus-integration.md`.
