//! Speaker encoding (M0-08): the native CAM++ (3D-Speaker) speaker encoder.
//!
//! Turns a reference utterance's 80-d Kaldi fbank into a 192-d speaker
//! embedding for zero-shot voice cloning — the input the piper v7 `spk_proj`
//! expects, replacing the zero-embedding fallback. The network is re-implemented
//! from scratch in Rust (whisper.cpp pattern): the verified `campplus.onnx`
//! topology is hard-coded and only weight *values* are loaded from GGUF; no ONNX
//! is touched at runtime (FR-LD-05).
//!
//! - [`camplus`] — the forward pass ([`SpeakerEncoder`]);
//! - `weights` — GGUF binding + BatchNorm fold (internal).
//!
//! The audio→fbank front-end (Kaldi fbank + CMN) is a separate work item and is
//! validated once an offline Kaldi-fbank oracle exists; this module's
//! fbank→embedding network is fully validated against onnxruntime fixtures
//! (`tests/parity/camplus/`) by the [`parity`] submodule.

pub mod camplus;
mod weights;

#[cfg(test)]
mod parity;

pub use camplus::{EMBED_DIM, SpeakerEncoder};
