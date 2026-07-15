//! CSM Metal decode session — thin Compute-seam wrapper (M4-05-T21).
//!
//! The Voxtral `VoxtralMetalDecodeSession` posture (M3-10 Wave 9/10),
//! applied to the CSM backbone + depth transformer:
//!
//! - **Explicit backend selection + FR-EX-08 gating**: construction
//!   probes `Compute::for_backend(Metal, CSM_HOT_OPS)`; a missing device
//!   or a coverage gap is a loud [`VokraError`] — never a silent CPU
//!   fall back. Off the `metal` feature this type does not exist
//!   (cfg-gated file); runtime probing without the type goes through
//!   [`super::gpu_backend_probe`].
//! - **Real GPU dispatch through the seam**: every GEMM / GEMV / softmax
//!   the two stacks emit routes through the `Compute::Metal` arm. Weight
//!   residency / per-step command-buffer fusion is the follow-up slot
//!   (M3-10 ResidencyMode precedent) — **not** taken here, and **no
//!   FlashAttention v3** (M4-07 red-line).
//! - **CPU parity within FP32 rounding**: the device-gated test band pins
//!   `atol ≤ 5e-4` on the hidden state plus greedy code-sequence
//!   identity against the CPU path on the same synthesized weights
//!   (M1 iMac local verification — CLAUDE.md dev environment).
//!
//! # `!Send` / `!Sync`
//!
//! The per-call `Compute` holds a live `MetalContext` on the stack only
//! (the piper-plus pattern), so the session itself stays plain data; it
//! is still confined to one thread by convention, matching the Voxtral
//! session's documented usage.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use vokra_core::{BackendKind, Result};

use super::backbone::{CSM_HOT_OPS, CsmFrame};
use super::frame::{CsmFrameKind, CsmGenerationState, CsmModel};
use crate::compute::Compute;

/// The Metal-backed CSM decode session (module docs).
pub struct CsmMetalDecodeSession {
    model: CsmModel,
    generation: CsmGenerationState,
}

impl std::fmt::Debug for CsmMetalDecodeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmMetalDecodeSession")
            .field("generation", &self.generation)
            .finish()
    }
}

impl CsmMetalDecodeSession {
    /// Builds the session from a model, re-routing its hot ops through
    /// the Metal arm. Fails loudly when no Metal device is present
    /// (FR-EX-08 — construction is the gate).
    ///
    /// # Errors
    ///
    /// [`vokra_core::VokraError::BackendUnavailable`] /
    /// [`vokra_core::VokraError::UnsupportedOp`] from the probe;
    /// state-allocation errors verbatim.
    pub fn new_from_model(model: CsmModel) -> Result<Self> {
        // Probe = the same construction the forward path performs; a
        // session that constructs will also dispatch.
        Compute::for_backend(BackendKind::Metal, CSM_HOT_OPS)?;
        let model = model.with_backend(BackendKind::Metal);
        let generation = CsmGenerationState::new(model.config())?;
        Ok(Self { model, generation })
    }

    /// Primes the dialog context (same contract as
    /// [`CsmModel::prime`]).
    ///
    /// # Errors
    ///
    /// Propagated verbatim.
    pub fn prime(&mut self, frames: &[CsmFrame]) -> Result<()> {
        self.model.prime(&mut self.generation, frames)
    }

    /// One frame step on the GPU seam (same contract as
    /// [`CsmModel::generate_frame_into`]).
    ///
    /// # Errors
    ///
    /// Propagated verbatim — a Metal dispatch failure is loud, never a
    /// CPU retry.
    pub fn step(
        &mut self,
        sample: &mut dyn FnMut(&mut [f32]) -> u32,
        codes_out: &mut [u32],
    ) -> Result<CsmFrameKind> {
        self.model
            .generate_frame_into(&mut self.generation, sample, codes_out)
    }

    /// Context frames consumed so far.
    #[must_use]
    pub fn context_len(&self) -> usize {
        self.generation.context_len()
    }

    /// Rewinds for a fresh turn (pre-allocated arena retained).
    pub fn reset(&mut self) {
        self.generation.reset();
    }

    /// The backend label (always `"metal"` for this type).
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        "metal"
    }
}
