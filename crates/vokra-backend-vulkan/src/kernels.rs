//! Typed per-kernel dispatch entry points on [`VulkanBackend`]
//! (M4-13-T03〜T08).
//!
//! Each method pairs a host-portable [`crate::plan`] constructor (shape
//! validation + push-constant packing + workgroup math — explicit
//! [`VokraError::InvalidArgument`](vokra_core::VokraError::InvalidArgument)
//! on bad shapes, testable on any host) with the gated generic dispatch
//! chain (`context::dispatch_kernel`, Vulkan targets only). The
//! placeholder-then-swap seam applies to every method: while the owner has
//! not yet committed a kernel's glslc-produced `.spv` (M4-13-T16), dispatch
//! surfaces the explicit
//! [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp) that
//! `spirv::require_blob` formats — never a silent CPU fall back (FR-EX-08).
//! Once the blob lands, the method lights up with no code change here.
//!
//! # Two dispatch surfaces (M4-13-T01)
//!
//! Only three of these kernels back a graph [`OpKind`](vokra_core::OpKind)
//! arm (`gemm_*` → `MatMul`, `elementwise` → `Mul` (+ duplicate `Add`
//! coverage), `softmax` → `Softmax` — see `crate::eval`). The rest —
//! `gemv` / `softmax_causal` / `layer_norm` / `gelu` / `conv1d` /
//! `activation` / `transpose` / `gather` — have **no** corresponding
//! `OpKind` variant (`OpKind::Gemv` / `LayerNorm` / `Gelu` / `Conv1D` /
//! `SoftmaxCausal` / `Transpose` / `Gather` do not exist); they are the
//! **imperative Whisper-base encoder / decoder primitives** exercised by the
//! M4-13-T12/T13 model-level parity harness, exactly like the Metal / CUDA
//! backends' non-graph kernels.
//!
//! # Off-target stubs
//!
//! The module compiles on every target: on non-Vulkan targets / default
//! (feature-off) builds the same method signatures exist but return the
//! explicit
//! [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
//! — the `smoke_dispatch_*` precedent — so integration tests stay
//! host-portable (they skip at `VulkanBackend::new()` anyway, which fails
//! off-target before any method could be reached).

use vokra_core::Result;

use crate::backend::{GemmPipelinePreference, VulkanBackend};
// `plan` is consumed only by the gated dispatch bodies; off-target builds use
// none of it (their stubs never validate — they are unreachable, see docs).
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
use crate::plan;

// ---------------------------------------------------------------------------
// Real dispatch bodies (Vulkan targets + feature on).
// ---------------------------------------------------------------------------

#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
mod imp {
    use vokra_core::Result;

    use crate::context::{KernelInvocation, dispatch_kernel};
    use crate::plan::KernelPlan;

    /// LE-encode a f32 slice for SSBO upload (matches the smoke impls).
    pub(super) fn f32s_to_le_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for f in v {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    /// LE-encode a u32 slice for SSBO upload (`gather` indices).
    pub(super) fn u32s_to_le_bytes(v: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for u in v {
            out.extend_from_slice(&u.to_le_bytes());
        }
        out
    }

    /// Decode a readback SSBO into f32s (inverse of [`f32s_to_le_bytes`]).
    pub(super) fn le_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
        debug_assert_eq!(bytes.len() % 4, 0, "SSBO readback is f32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// A 4-byte all-zero dummy SSBO for optional bindings the shader never
    /// reads (`bias_present = 0` paths). Vulkan requires every declared
    /// binding bound; `dispatch_kernel` additionally rejects empty inputs.
    pub(super) const DUMMY_SSBO: [u8; 4] = [0u8; 4];

    impl super::VulkanBackend {
        /// Executes one [`KernelPlan`] against this backend's device and
        /// decodes the output SSBO as f32s. The shared tail of every typed
        /// kernel method below.
        pub(super) fn run_plan_f32(&self, plan: &KernelPlan, inputs: &[&[u8]]) -> Result<Vec<f32>> {
            let inv = KernelInvocation {
                name: plan.shader,
                inputs,
                output_byte_len: plan.output_byte_len(),
                push_constants: &plan.push_constants,
                spec_constants: &plan.spec_constants,
                workgroups: plan.workgroups,
            };
            let bytes = dispatch_kernel(self.device(), &inv)?;
            Ok(le_bytes_to_f32s(&bytes))
        }
    }
}

#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
impl VulkanBackend {
    /// `out[m,n] = Σ_k a[m,k]·b[k,n]` on the GEMM pipeline this device
    /// selects for `pref` (M4-13-T03). Capability-driven variant selection:
    /// `PreferCoopMatrix` binds `gemm_coopmat` only when the probe reports
    /// the cooperative-matrix preconditions, otherwise `gemm_subgroup` — the
    /// op still runs on the GPU either way (never a silent CPU fall back).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`](vokra_core::VokraError::InvalidArgument)
    /// on shape mismatch;
    /// [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp)
    /// while the selected variant's `.spv` blob is not committed
    /// (M4-13-T16 owner task lights it up).
    pub fn gemm_f32(
        &self,
        pref: GemmPipelinePreference,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        let variant = self.select_gemm_pipeline_variant(pref);
        let plan = plan::plan_gemm(variant, m, n, k, a.len(), b.len())?;
        self.run_plan_f32(
            &plan,
            &[&imp::f32s_to_le_bytes(a), &imp::f32s_to_le_bytes(b)],
        )
    }

    /// `y[i] = Σ_j a[i,j]·x[j] (+ bias[i])` (M4-13-T04) — the Whisper
    /// decoder-step hot path (tied logits head). **Not a graph op**
    /// (`OpKind::Gemv` does not exist): this is a model-level primitive
    /// exercised by the M4-13-T12/T13 Whisper parity harness.
    ///
    /// # Errors
    ///
    /// See [`VulkanBackend::gemm_f32`] — same contract, `gemv.spv` blob.
    pub fn gemv_f32(
        &self,
        m: usize,
        n: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let plan = plan::plan_gemv(m, n, a.len(), x.len(), bias.map(<[f32]>::len))?;
        let a_bytes = imp::f32s_to_le_bytes(a);
        let x_bytes = imp::f32s_to_le_bytes(x);
        let b_bytes = match bias {
            Some(b) => imp::f32s_to_le_bytes(b),
            // The shader never reads binding 2 when bias_present = 0, but
            // Vulkan requires the binding bound — 4-byte dummy.
            None => imp::DUMMY_SSBO.to_vec(),
        };
        self.run_plan_f32(&plan, &[&a_bytes, &x_bytes, &b_bytes])
    }
}

// ---------------------------------------------------------------------------
// Off-target stubs (macOS / iOS / WASM hosts, or feature `vulkan` off).
// ---------------------------------------------------------------------------

#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
impl VulkanBackend {
    /// Off-target stub — see the module docs. Unreachable in practice
    /// ([`VulkanBackend::new`] fails first) but keeps the API surface
    /// identical across targets so integration tests compile everywhere.
    ///
    /// # Errors
    ///
    /// Always
    /// [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable).
    pub fn gemm_f32(
        &self,
        _pref: GemmPipelinePreference,
        _m: usize,
        _n: usize,
        _k: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> Result<Vec<f32>> {
        Err(stub_unavailable("gemm"))
    }

    /// Off-target stub — see [`VulkanBackend::gemm_f32`].
    ///
    /// # Errors
    ///
    /// Always
    /// [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable).
    pub fn gemv_f32(
        &self,
        _m: usize,
        _n: usize,
        _a: &[f32],
        _x: &[f32],
        _bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        Err(stub_unavailable("gemv"))
    }
}

#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
fn stub_unavailable(kernel: &str) -> vokra_core::VokraError {
    vokra_core::VokraError::BackendUnavailable(format!(
        "vokra-backend-vulkan `{kernel}` kernel: compiled without the `vulkan` feature or on a \
         non-Vulkan target (macOS / iOS / WASM); rebuild with --features vulkan on Linux / \
         Android / Windows (FR-EX-08: explicit error, never a silent CPU substitute)."
    ))
}
