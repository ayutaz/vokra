//! # vokra-vad-micro
//!
//! The Silero VAD v5 forward core (M0-05 subgraph) as a `#![no_std]` (+ `alloc`)
//! subset, so it cross-compiles for bare-metal **Cortex-M55** (`thumbv8m-none`,
//! IoT Tier 3 / NFR-PT-03) — the crate that realises "Cortex-M55 で Silero VAD
//! が動作" (SRS §6, M5-03).
//!
//! # Why a separate crate (ADR M5-03-iot-tier3-nostd §(a), 案1)
//!
//! `vokra-models` depends non-optionally on `vokra-ops` + `vokra-backend-cpu`
//! (both std-heavy), so building `vokra-models` for `thumbv8m` is impossible even
//! though `silero_vad` itself imports neither. This crate lifts the
//! no_std-capable *numeric forward* — GGUF weight binding, the learned
//! pseudo-STFT, the encoder conv stack, the LSTM cell + head — out of
//! `vokra-models::silero_vad`, naming only `vokra-core` (its no_std subset:
//! error + GGUF reader). The std `vokra-models::silero_vad` wrapper (the
//! `VadEngine` impl, the streaming handle, the WAV reader, the `open()` file
//! constructor) depends on this crate and re-exports [`SampleRate`]. There is
//! therefore **one** forward, and the std and no_std builds are **bit-identical
//! by construction** (same source; the transcendentals come from the shared
//! [`scalar`] module — M5-03 T08/T11).
//!
//! # Design red lines (inherited from `vokra-models::silero_vad`)
//!
//! - **1:1 preservation (FR-LD-06 / FR-OP-50 / NFR-QL-05)**: Silero VAD is a
//!   dedicated subgraph, not lowered to generic audio-dialect ops. The
//!   pseudo-STFT is a *learned* `Conv1d`, reproduced op-for-op — never a DSP
//!   `stft` (see [`pseudo_stft`]).
//! - **Zero external deps (NFR-DS-02)**: only `vokra-core`. The
//!   transcendentals ([`scalar`]) are a from-scratch scalar port — **no `libm`**
//!   (deny.toml bans it).
//! - **No `unsafe` (NFR-RL-07)**: the whole crate is safe Rust (`#![forbid`-class
//!   workspace lint `unsafe_code = "deny"`). The `sqrt` route is Newton–Raphson
//!   (portable, no `asm!`); an FP-armv8 `vsqrt` accel is an owner follow-up
//!   (ADR §(d)/(e), T18).
//!
//! # no_std construction path (T19)
//!
//! The no_std subset has no filesystem. Load GGUF from an in-memory /
//! flash-mapped `&[u8]` — `vokra_core::gguf::GgufFile::from_external` (or
//! `parse`) — then [`SileroWeights::from_gguf`]; a single frame runs through
//! [`SileroWeights::forward_chunk`]. A library is allocator-agnostic; the
//! downstream binary installs the `#[global_allocator]`.

// M5-03-T09: `#![no_std]` whenever the default `std` feature is off (Cortex-M55
// Tier 3). With the default feature set this attribute is inert — the crate is a
// normal std library (NFR-PT-01 cross-build non-interference).
#![cfg_attr(not(feature = "std"), no_std)]

// The forward is alloc-dependent (owned `Vec<f32>` weights, `String` error
// messages), not alloc-free. `extern crate alloc` links it in both modes; under
// `std` it is already present, so this is harmless there.
extern crate alloc;

pub mod scalar;

mod encoder;
mod lstm;
mod math;
mod pseudo_stft;
mod vad;
mod weights;

pub use encoder::{EncoderOut, encode};
pub use lstm::LstmState;
pub use pseudo_stft::{Magnitude, pseudo_stft, stft_conv};
pub use vad::{SampleRate, run_frame};
pub use weights::{RateWeights, SileroWeights};
