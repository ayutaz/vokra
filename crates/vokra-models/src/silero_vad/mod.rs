//! Silero VAD v5 as a 1:1-preserved dedicated subgraph (M0-05).
//!
//! # Design red lines (permanent)
//!
//! - **1:1 preservation (FR-LD-06 / FR-OP-50)**: Silero VAD is kept as a
//!   dedicated subgraph, *not* lowered to generic audio-dialect ops. Its
//!   internal recurrent state (LSTM `h`/`c`), the 64-sample carried context
//!   and the learned pseudo-STFT are all hidden behind the stream handle.
//! - **No librosa/FFT STFT approximation (NFR-QL-05)**: the pseudo-STFT is a
//!   *learned* `Conv1d(1, 258, k=256, stride=128)`. Replacing it with a
//!   standard `stft` op (FR-OP-01) would corrupt the meaning of the weights,
//!   so this module must not call into the `vokra-ops` `stft` path.
//!
//! # Layout (M0-05)
//!
//! Weights load from the `vokra.*` GGUF produced by `vokra-convert`
//! (M0-03). The public surface is a [`vokra_core::engines::VadEngine`]
//! implementation whose stream handle carries all hidden state.
//!
//! Implementation lands with M0-05 (T01–T12).
