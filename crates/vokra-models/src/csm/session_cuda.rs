//! CSM CUDA decode session — thin Compute-seam wrapper (M4-05-T22).
//!
//! **API-symmetric with [`super::session_metal::CsmCudaDecodeSession`]**
//! (`new_from_model` / `prime` / `step` / `reset` / `context_len` /
//! `backend_name` — the Voxtral Metal/CUDA session symmetry, M3-10). The
//! CUDA arm rides the M2-03 raw-FFI base: `libcuda` + NVRTC dlopen'd at
//! runtime (EULA install model, zero-dep — NFR-DS-02).
//!
//! - **Explicit backend selection + FR-EX-08 gating**: construction
//!   probes `Compute::for_backend(Cuda, CSM_HOT_OPS)`; a missing driver /
//!   device is a loud [`VokraError`] — never a silent CPU fall back. Off
//!   the `cuda` feature this type does not exist (cfg-gated file);
//!   runtime probing without the type goes through
//!   [`super::gpu_backend_probe`] (the off-GPU negative test band —
//!   M3-10's 3-band pattern).
//! - **Real GPU dispatch through the seam**; residency / fusion =
//!   follow-up slot, **no FlashAttention v3** (M4-07 red-line).
//! - **Device parity is vast.ai-gated**: real-GPU verification runs on a
//!   spot RTX 4090 (CLAUDE.md dev environment) — CI skips cleanly when
//!   no device is present (never a fabricated pass).

#![cfg(all(feature = "cuda", any(unix, windows)))]

use vokra_core::{BackendKind, Result};

use super::backbone::{CSM_HOT_OPS, CsmFrame};
use super::frame::{CsmFrameKind, CsmGenerationState, CsmModel};
use crate::compute::Compute;

/// The Metal-backed CSM decode session (module docs).
pub struct CsmCudaDecodeSession {
    model: CsmModel,
    generation: CsmGenerationState,
}

impl std::fmt::Debug for CsmCudaDecodeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmCudaDecodeSession")
            .field("generation", &self.generation)
            .finish()
    }
}

impl CsmCudaDecodeSession {
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
        Compute::for_backend(BackendKind::Cuda, CSM_HOT_OPS)?;
        let model = model.with_backend(BackendKind::Cuda);
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
    /// Propagated verbatim — a CUDA dispatch failure is loud, never a
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

    /// The backend label (always `"cuda"` for this type).
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        "cuda"
    }
}
