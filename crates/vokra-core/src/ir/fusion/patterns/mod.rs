//! Fusion pattern registry (M2-04-T04+).
//!
//! Each submodule declares one [`FusionPattern`](super::FusionPattern)
//! implementation. Patterns are individually testable against synthetic
//! [`AudioGraph`](crate::ir::AudioGraph) inputs — none of them require an
//! in-tree consumer model to exercise their matcher plumbing.
//!
//! Registered patterns (M2-04):
//!
//! - [`logmel::LogMelPattern`] — Whisper log-mel front-end
//!   `Stft → (mul-squared-magnitude proxy) → MelFilterbank → (log proxy)`.
//!
//! Scaffolded patterns (M2-04-T09/T10, kernel bodies co-delivered with
//! the FR-OP-11/13/14 consumer model — see [`snake`] module docs):
//!
//! - [`snake::Conv1dSnakePattern`] — `Conv1d → Snake`.
//! - [`snake::UpsampleSnakeResidualPattern`] — `Upsample → Snake → Add(residual)`
//!   (BigVGAN AMP block).
//!
//! The Snake / BigVGAN patterns' active matchers live behind the
//! `fusion-snake-stub` cargo feature (default OFF); the marker types
//! ([`snake::Conv1dSnakePattern`] / [`snake::UpsampleSnakeResidualPattern`])
//! are always visible so the follow-up PR can hang the real
//! [`super::FusionPattern`] impls off them without churning the module
//! surface.

pub mod logmel;
pub mod snake;
