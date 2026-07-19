//! Imperative compute dispatcher for the native models (Phase 3 of the GPU
//! execution architecture; see `scratchpad/graph-engine-plan.md` §3).
//!
//! The graph evaluator ([`vokra_core::run_graph`]) drives one op at a time via
//! [`Backend::eval_op`](vokra_core::Backend); it is the right shape for new /
//! fused / graph-first models. The **existing** models (Whisper, piper-plus,
//! CAM++) are imperative: they call the compute kernels directly in a
//! zero-malloc hot path (`out: &mut [f32]`, caller-owned scratch — FR-EX-05).
//! Rewriting them onto the graph engine would add a large op surface and risk
//! the numeric parity for no speed gain (same kernels). Instead this module adds
//! a thin, typed seam — [`Compute`] — that those call sites dispatch through, so
//! the same GEMM the CPU backend runs can instead run on the GPU by swapping one
//! enum arm.
//!
//! # One kernel per (backend, op); two entry shapes
//!
//! [`Compute::gemm_f32`] on the CPU arm calls the very same
//! [`vokra_backend_cpu::kernels::gemm_f32`] that
//! [`Backend::eval_op`](vokra_core::Backend) does, and on the Metal arm the very
//! same `MetalContext::gemm_f32` — there is no second kernel. So the imperative
//! `Compute` path and the graph `eval_op` path stay bit-for-bit consistent on a
//! given backend, and a `Compute::cpu()` run reproduces the pre-seam output
//! **exactly** (the parity suites stay green).
//!
//! # One model = one backend, no silent fallback (FR-EX-08)
//!
//! [`Compute::for_backend`] takes the model's *required* hot-op set and refuses
//! to build a backend that does not cover **every** op in it — an explicit
//! [`VokraError::UnsupportedOp`], never a per-op CPU fall back. As of Phase 4
//! (M2-01 T09-T13) the Metal backend has a real GPU kernel for every hot op
//! (GEMM / GEMV / softmax / layer-norm / GELU / conv1d), so not only the
//! GEMM-only models (CAM++, piper-plus) but the **full Whisper forward** runs on
//! Metal through this seam. A backend that genuinely could not cover an op would
//! still be an explicit `UnsupportedOp` rather than a silent CPU fall back;
//! selecting the CPU instead is the caller's *explicit* [`BackendKind::Cpu`]
//! choice.
//!
//! # `!Send` `MetalContext`, `Send + Sync` engines
//!
//! `MetalContext` is `!Send` / `!Sync` (thread-affine `id` handles), whereas the
//! engine traits (`AsrEngine` / `TtsEngine` / …) are `Send + Sync`. So a model
//! **engine** must not *hold* a live backend; it holds a [`BackendKind`]
//! (`Copy`) and builds a `Compute` on the stack at each transcribe / synthesize
//! entry, threading `&Compute` down. That keeps the engines `Send + Sync` while
//! the `!Send` context lives only for the call.

use vokra_backend_cpu::kernels;
use vokra_backend_cpu::kernels::KQuantDtype;
use vokra_core::backend::BackendKind;
use vokra_core::{Backend, DecoderLayerView, PrenormLayer, Result, VokraError};
// M3-06 mimi_rvq (+ M4-04 dac_rvq / encodec_rvq, + M4-16 FSQ family
// wavtokenizer_vq / xcodec2_fsq) codec decode wired into the imperative
// Compute seam. The CPU arms delegate to the vokra-ops runtime functions;
// the Metal / CUDA arms return `VokraError::UnsupportedOp` until the GPU
// kernels land (no silent CPU fall back, FR-EX-08). See
// `Compute::mimi_rvq_f32` / `dac_rvq_f32` / `encodec_rvq_f32` /
// `wavtokenizer_vq_f32` / `xcodec2_fsq_f32` below.
use vokra_ops::{
    CodebookTable, DacOutProj, DacRvqAttrs, EncodecRvqAttrs, FsqOutProj, MimiRvqAttrs,
    WavTokenizerVqAttrs, Xcodec2FsqAttrs, dac_rvq_decode, encodec_rvq_decode, mimi_rvq_decode,
    wavtokenizer_vq_decode, xcodec2_fsq_decode,
};

/// A backend-dispatched hot op — the operators the imperative models route
/// through a backend (as opposed to the model-internal scalar glue like
/// LeakyReLU, embedding lookup or transpose, which always stays on the host and
/// is *not* a backend op, so is never a silent fall back).
///
/// A model declares the set it needs (`*_HOT_OPS`) so [`Compute::for_backend`]
/// can enforce whole-model backend coverage before running anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotOp {
    /// Row-major GEMM (`gemm_f32`) — the dominant matmul / linear / conv (via
    /// im2col) cost. The first op wired onto the GPU (M2-01 slice).
    Gemm,
    /// Row-major matrix-vector product (`gemv_f32`) — Whisper's tied logits head.
    Gemv,
    /// Row-wise softmax (`softmax_f32`) — attention.
    Softmax,
    /// Affine layer normalisation (`layer_norm_f32`) — Whisper pre-norm blocks.
    LayerNorm,
    /// Exact (erf) GELU (`gelu_f32`) — Whisper MLP / conv stem.
    Gelu,
    /// 1-D convolution (`conv1d_f32`) — Whisper encoder stem.
    Conv1d,
    /// Mimi (Kyutai) residual vector quantization codec decode
    /// (`mimi_rvq_decode`) — the M3-06 RVQ codec op family. The heterogeneous
    /// signature (u32 `codes` + `Vec<CodebookTable>` → `Vec<f32>`) drives the
    /// [`Compute::mimi_rvq_f32`] method shape (heap-returning, not
    /// `out: &mut [f32]`), which is the reason `mimi_rvq_decode` is a
    /// runtime function in `vokra-ops` rather than an [`vokra_core::OpKind`]
    /// variant (module docs in `vokra_ops::mimi_rvq`).
    ///
    /// **CPU-only through the imperative seam today.** The Metal / CUDA arms
    /// of [`Compute::mimi_rvq_f32`] return an explicit
    /// [`VokraError::UnsupportedOp`]; the M3-06 T14 (Metal) / T15 (CUDA) GPU
    /// kernels are deferred to the M3-09 mimi_bridge upgrade past stub. This
    /// variant therefore has `covered_by_metal` / `covered_by_cuda` /
    /// `covered_by_vulkan` return `false`, so any model that lists `MimiRvq`
    /// in its required set will fail `for_backend(Metal|Cuda|Vulkan, …)` with
    /// a coverage `UnsupportedOp` — never a silent CPU fall back (FR-EX-08).
    MimiRvq,
    /// DAC (Descript) factorized residual VQ codec decode
    /// (`dac_rvq_decode`) — M4-04, FR-OP-30. Same heterogeneous-signature /
    /// heap-returning shape as [`HotOp::MimiRvq`] plus the per-quantizer
    /// projection operands ([`DacOutProj`]). Kept a **separate variant** from
    /// `MimiRvq` so the coverage table stays honest per op (ADR M4-04 §D-e).
    ///
    /// **CPU-only through the imperative seam today** — the Metal / CUDA arms
    /// of [`Compute::dac_rvq_f32`] return an explicit
    /// [`VokraError::UnsupportedOp`] (the M4-04 GPU kernels are deferred; the
    /// naive gather + GEMV + fold layout note in `vokra_ops::mimi_rvq`
    /// L104-106 applies to all three RVQ ops). `covered_by_*` return `false`
    /// so the coverage gate rejects GPU listings (FR-EX-08).
    DacRvq,
    /// EnCodec residual VQ codec decode (`encodec_rvq_decode`) — M4-04,
    /// FR-OP-30 op / FR-OP-32 permanent weight exclusion. The op rides the
    /// shape-generic gather + FP32 fold; **pretrained EnCodec weights never
    /// ship** (the official zoo excludes them permanently; the M2-13 gate
    /// refuses them without a research flag). Separate variant for honest
    /// per-op coverage (ADR M4-04 §D-e); CPU-only today like
    /// [`HotOp::DacRvq`].
    EncodecRvq,
    /// WavTokenizer single-codebook VQ decode (`wavtokenizer_vq_decode`) —
    /// M4-16, FR-OP-31 **FSQ family** (single-stage, *separate subgraph from
    /// the RVQ family* — no cross-codebook residual sum, no paged variant;
    /// module docs in `vokra_ops::fsq_codec`). Heterogeneous-signature /
    /// heap-returning shape like the RVQ seam methods, but the table operand
    /// is a *singular* [`CodebookTable`].
    ///
    /// **CPU-only through the imperative seam today** — the Metal / CUDA
    /// arms of [`Compute::wavtokenizer_vq_f32`] return an explicit
    /// [`VokraError::UnsupportedOp`] (the M4-16 GPU kernels are deferred;
    /// being single-stage GEMV/gather bound they will reuse the existing
    /// M2-01 / M2-03 kernels). `covered_by_*` return `false` so the coverage
    /// gate rejects GPU listings (FR-EX-08).
    WavTokenizerVq,
    /// X-Codec 2 FSQ dequant (`xcodec2_fsq_decode`) — M4-16, FR-OP-31 FSQ
    /// family sibling of [`HotOp::WavTokenizerVq`]. Implicit per-dimension
    /// grid (no codebook tensor) + one out-projection GEMV per timestep.
    /// Separate variant for honest per-op coverage; CPU-only today —
    /// `covered_by_*` return `false` and the Metal / CUDA arms of
    /// [`Compute::xcodec2_fsq_f32`] are explicit
    /// [`VokraError::UnsupportedOp`] (FR-EX-08).
    Xcodec2Fsq,
}

impl HotOp {
    /// Whether the Metal backend's imperative [`Compute`] seam covers this op.
    ///
    /// Kept in sync with the Metal arms of the [`Compute`] methods below; the
    /// `metal_coverage_is_consistent` test pins the two together. As of Phase 4
    /// (M2-01 T09-T13) the whole Whisper hot-op set (GEMM / GEMV / softmax /
    /// layer_norm / GELU / conv1d) has a `MetalContext` kernel, so the whole
    /// Whisper forward runs on the GPU through this seam. [`HotOp::MimiRvq`]
    /// remains uncovered on Metal — the M3-06 T14 MSL kernel is deferred to
    /// the M3-09 mimi_bridge upgrade past stub, and until it lands the Metal
    /// arm of [`Compute::mimi_rvq_f32`] returns an explicit
    /// [`VokraError::UnsupportedOp`] (never a silent CPU fall back, FR-EX-08).
    /// (The *graph* backend `MetalBackend::supports` / `eval_op` is a separate
    /// path and still covers only `MatMul` — the two coverage surfaces are
    /// intentionally independent.)
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    fn covered_by_metal(self) -> bool {
        // Phase 4 wired the six Whisper hot ops; MimiRvq is deferred to the
        // M3-06 T14 MSL kernel (M3-09 follow-up). Any model listing MimiRvq
        // in its required set therefore fails `for_backend(Metal, …)` with a
        // coverage `UnsupportedOp` (FR-EX-08 — no silent CPU fall back).
        matches!(
            self,
            HotOp::Gemm
                | HotOp::Gemv
                | HotOp::Softmax
                | HotOp::LayerNorm
                | HotOp::Gelu
                | HotOp::Conv1d
        )
    }

    /// Whether the CUDA backend's imperative [`Compute`] seam covers this op.
    ///
    /// Kept in sync with the `Be::Cuda` arms of the [`Compute`] methods below;
    /// the `cuda_coverage_is_consistent` test pins the two together. As of
    /// Phase 4 (M2-03 T10-T14) the whole Whisper hot-op set (GEMM / GEMV /
    /// softmax / layer_norm / GELU / conv1d) has a real NVRTC-compiled kernel,
    /// so the whole Whisper forward runs on the GPU through this seam.
    /// [`HotOp::MimiRvq`] remains uncovered on CUDA — the M3-06 T15 NVRTC
    /// kernel is deferred to the M3-09 mimi_bridge upgrade past stub, and
    /// until it lands the CUDA arm of [`Compute::mimi_rvq_f32`] returns an
    /// explicit [`VokraError::UnsupportedOp`] (never a silent CPU fall back,
    /// FR-EX-08). (The *graph* backend `CudaBackend::supports` / `eval_op` is
    /// a separate path and still covers only `MatMul` — the two coverage
    /// surfaces are independent.)
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    fn covered_by_cuda(self) -> bool {
        // Phase 4 wired the six Whisper hot ops; MimiRvq is deferred to the
        // M3-06 T15 NVRTC kernel (M3-09 follow-up). Any model listing MimiRvq
        // in its required set therefore fails `for_backend(Cuda, …)` with a
        // coverage `UnsupportedOp` (FR-EX-08 — no silent CPU fall back).
        matches!(
            self,
            HotOp::Gemm
                | HotOp::Gemv
                | HotOp::Softmax
                | HotOp::LayerNorm
                | HotOp::Gelu
                | HotOp::Conv1d
        )
    }

    /// Whether the Vulkan backend's imperative [`Compute`] seam covers this op.
    ///
    /// **M3-02 foundation slice (2026-07-09):** no SPIR-V kernel is wired yet
    /// (`crates/vokra-backend-vulkan/kernels/precompiled/` ships no `.spv`
    /// blob), so **every** hot op is uncovered — including [`HotOp::MimiRvq`]
    /// — and `covered_by_vulkan(_) = false` for every variant. As T14〜T22
    /// land, this method flips to `true` op-by-op, in lock-step with the
    /// `Be::Vulkan` arms of the `Compute` methods below (the
    /// `vulkan_coverage_is_consistent` test pins the two together). MimiRvq
    /// on Vulkan is not on the M3-02 track at all — it lands with the M3-06
    /// GPU kernels' Vulkan sibling (M4+), so the `false` here holds through
    /// the whole Vulkan T14〜T22 rollout.
    ///
    /// The consequence today is that `Compute::for_backend(BackendKind::Vulkan,
    /// &required)` returns an explicit [`VokraError::UnsupportedOp`] for every
    /// non-empty `required` — never a silent CPU fall back (FR-EX-08).
    #[cfg(all(
        feature = "vulkan",
        any(target_os = "linux", target_os = "android", target_os = "windows")
    ))]
    fn covered_by_vulkan(self) -> bool {
        // Foundation slice: the Vulkan backend has NO wired kernels. This is
        // the honest state — as ticket M3-02-T14 ships the GEMM `.spv`, its
        // arm becomes `HotOp::Gemm => true`; T15 flips GEMV, and so on. Note
        // that MimiRvq is off the M3-02 T14〜T22 track (it needs the M3-06 GPU
        // kernels' Vulkan sibling, which is an M4+ item), so this method will
        // still return `false` for `HotOp::MimiRvq` after T22 lands.
        let _ = self;
        false
    }

    /// Whether the WebGPU backend's imperative [`Compute`] seam covers this
    /// op (M4-01-T16).
    ///
    /// Kept in sync with the `Be::WebGpu` arms of the [`Compute`] methods
    /// below (the wasm32-only `webgpu_coverage_is_consistent` test pins the
    /// two together; the Node harness `tools/wasm/run-kernel-parity.mjs`
    /// exercises the runtime side). The whole Whisper hot-op set (GEMM /
    /// GEMV / softmax / layer_norm / GELU / conv1d) has a WGSL kernel from
    /// the M4-01 slice (T12〜T15), so the whole Whisper forward runs on
    /// WebGPU through this seam. The RVQ codec ops ([`HotOp::MimiRvq`] /
    /// [`HotOp::DacRvq`] / [`HotOp::EncodecRvq`]) remain uncovered — the
    /// same posture as Metal / CUDA / Vulkan — so any model listing them
    /// fails `for_backend(WebGpu, …)` with a coverage `UnsupportedOp`, never
    /// a silent CPU fall back (FR-EX-08).
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    fn covered_by_webgpu(self) -> bool {
        matches!(
            self,
            HotOp::Gemm
                | HotOp::Gemv
                | HotOp::Softmax
                | HotOp::LayerNorm
                | HotOp::Gelu
                | HotOp::Conv1d
        )
    }
}

/// A typed, zero-malloc compute dispatcher the imperative model hot path calls
/// instead of the `vokra_backend_cpu::kernels::*` free functions directly.
///
/// Build one at a model entry point with [`Compute::for_backend`] (or the
/// infallible [`Compute::cpu`]) and thread `&Compute` down; the `out: &mut [f32]`
/// method shape preserves the zero-allocation hot path (FR-EX-05). It is a plain
/// `enum` dispatch (not `&dyn`), so the CPU per-call cost over calling the kernel
/// directly is a single branch.
pub struct Compute {
    be: Be,
}

/// The explicit refusal every GPU arm of [`Compute::gemm_q_f32`] returns
/// (M5-15-T27). Compiled only when at least one GPU arm exists on this target,
/// so it never becomes dead code.
#[cfg(any(
    all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    all(feature = "cuda", any(unix, windows)),
    all(feature = "webgpu", target_arch = "wasm32"),
))]
fn unsupported_quant_gemm(backend: &str) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "fused K-quant GEMM has no {backend} kernel (M5-15 is CPU-only; GPU fused K-quant is a \
         separate WP). Vokra does not dequantize behind your back, nor silently run this op on \
         the CPU (FR-EX-08) — load the model without \
         `WhisperLoadOptions::fused_quant_weights` to get dequantized weights this backend can \
         use, or select BackendKind::Cpu."
    ))
}

/// The live backend behind a [`Compute`]. The `Metal` arm owns a `!Send`
/// `MetalContext`, which is why a `Compute` is built at a call entry and never
/// stored on a `Send + Sync` engine.
enum Be {
    /// CPU kernels (`vokra_backend_cpu::kernels`). Covers every [`HotOp`].
    Cpu,
    /// Metal GPU context. Covers every [`HotOp`] (Phase 4).
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    Metal(vokra_backend_metal::MetalContext),
    /// CUDA GPU context. Covers every [`HotOp`] (Phase 4). `Box`ed because
    /// `CudaContext` embeds the whole `CudaDriver` (≈20 dlopen'd fn pointers) by
    /// value, which would make the `Be` enum's inline size dwarf the other arms
    /// (`clippy::large_enum_variant`); the heap alloc is negligible — a `Compute`
    /// is built once per model entry, after a far costlier dlopen + NVRTC compile.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    Cuda(Box<vokra_backend_cuda::CudaContext>),
    /// WebGPU context (browser WASM, M4-01). Covers the six Whisper hot ops
    /// through per-op WGSL dispatches (upload → dispatch → readback; whole-
    /// run device residency is the M4-02+ follow-up). `!Send` like the Metal
    /// arm — glue handles are realm-affine.
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    WebGpu(vokra_backend_webgpu::WebGpuContext),
}

impl Compute {
    /// A CPU-backed dispatcher. Infallible: the CPU backend covers every op, and
    /// its methods reproduce the pre-seam kernel calls bit-for-bit.
    #[must_use]
    pub fn cpu() -> Self {
        Compute { be: Be::Cpu }
    }

    /// Builds a dispatcher for `kind`, requiring it to cover every op in
    /// `required` (one model = one backend, FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] if `kind` is a real backend that does not
    ///   cover some op in `required` (e.g. Metal for a model that needs softmax)
    ///   — never a per-op CPU fall back.
    /// - [`VokraError::BackendUnavailable`] if `kind` is not built into this
    ///   binary (e.g. `Metal` without the `metal` feature, or off an Apple
    ///   target), or if the device probe fails (no Metal device).
    pub fn for_backend(kind: BackendKind, required: &[HotOp]) -> Result<Self> {
        // `required` is consulted only by the Metal / CUDA / Vulkan coverage
        // gates; without any GPU arm compiled in, the CPU / unavailable arms do
        // not read it.
        #[cfg(not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows)),
            all(
                feature = "vulkan",
                any(target_os = "linux", target_os = "android", target_os = "windows")
            ),
            all(feature = "webgpu", target_arch = "wasm32")
        )))]
        let _ = required;
        match kind {
            BackendKind::Cpu => Ok(Compute::cpu()),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            BackendKind::Metal => {
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_metal()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "metal backend does not cover {op:?} in this slice; the model requires \
                         {required:?}. One model = one backend — Vokra does not silently run the \
                         uncovered ops on the CPU (FR-EX-08). Select BackendKind::Cpu, or wait for \
                         the Metal {op:?} kernel (M2-01)."
                    )));
                }
                Ok(Compute {
                    be: Be::Metal(vokra_backend_metal::MetalContext::new()?),
                })
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            BackendKind::Cuda => {
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_cuda()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "cuda backend does not cover {op:?} in this slice; the model requires \
                         {required:?}. One model = one backend — Vokra does not silently run the \
                         uncovered ops on the CPU (FR-EX-08). Select BackendKind::Cpu, or wait for \
                         the CUDA {op:?} kernel (M2-03 T10–T14)."
                    )));
                }
                Ok(Compute {
                    be: Be::Cuda(Box::new(vokra_backend_cuda::CudaContext::new()?)),
                })
            }
            #[cfg(all(
                feature = "vulkan",
                any(target_os = "linux", target_os = "android", target_os = "windows")
            ))]
            BackendKind::Vulkan => {
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_vulkan()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "vulkan backend has no wired kernel for {op:?} in the M3-02 foundation \
                         slice; the model requires {required:?}. \
                         `crates/vokra-backend-vulkan/kernels/precompiled/` ships no .spv blob \
                         yet — every hot op is uncovered. One model = one backend — Vokra does \
                         not silently run the uncovered ops on the CPU (FR-EX-08). Select \
                         BackendKind::Cpu, or wait for the SPIR-V kernels (M3-02-T14〜T22)."
                    )));
                }
                // `required` is empty AND every hot op is uncovered — the
                // foundation slice cannot construct a useful `Compute::Vulkan`
                // dispatcher (no callable kernel). Surface an explicit error
                // rather than pretending a coverage-empty dispatcher is usable.
                // Once T14+ lands, this branch becomes an
                // `Ok(Compute { be: Be::Vulkan(...) })` — the same shape as the
                // Metal / CUDA arms above.
                Err(VokraError::UnsupportedOp(
                    "vulkan Compute path has no wired kernels in the M3-02 foundation slice — \
                     no covered required set exists. Wait for M3-02-T14+ SPIR-V kernels."
                        .to_owned(),
                ))
            }
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            BackendKind::WebGpu => {
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_webgpu()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "webgpu backend does not cover {op:?}; the model requires {required:?}. \
                         One model = one backend — Vokra does not silently run the uncovered ops \
                         on the CPU (FR-EX-08). Select BackendKind::Cpu explicitly for the WASM \
                         SIMD128/scalar path."
                    )));
                }
                Ok(Compute {
                    be: Be::WebGpu(vokra_backend_webgpu::WebGpuContext::new()?),
                })
            }
            other => Err(VokraError::BackendUnavailable(format!(
                "{other:?} backend is not built into vokra-models (build with the `metal` feature \
                 on macOS / iOS for Metal, the `cuda` feature on Windows / Linux for CUDA, the \
                 `vulkan` feature on Linux / Android / Windows for Vulkan, or the `webgpu` \
                 feature on wasm32 for browser WebGPU)"
            ))),
        }
    }

    /// The backend this dispatcher runs on (`"cpu"` or `"metal"`).
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        match &self.be {
            Be::Cpu => "cpu",
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => "metal",
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => "cuda",
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => "webgpu",
        }
    }

    /// Whether this backend has the Phase-5 fused non-causal attention
    /// ([`Self::attn_f32`]): `true` on the GPU arms (Metal / CUDA), `false` on
    /// CPU.
    ///
    /// The caller (`whisper::nn::attention_from_kv_into`) gates the fused fast
    /// path on this: only a GPU backend routes a non-causal block through
    /// `attn_f32`; the CPU always runs the per-op head loop. This keeps the CPU
    /// arm of `attn_f32` an explicit [`VokraError::UnsupportedOp`] that correct
    /// code never reaches (no silent fall back, FR-EX-08), while `compute.rs`
    /// hosts **zero** duplicated attention math (nn.rs is the single source of
    /// truth for the head loop).
    #[must_use]
    pub fn attention_is_fused(&self) -> bool {
        match &self.be {
            Be::Cpu => false,
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => true,
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => true,
            // No fused-attention WGSL chain in the M4-01 slice: the caller
            // runs the per-op head loop (standard GEMM + softmax — also the
            // FA v3 red-line posture). Honest `false`, not a stub `true`.
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => false,
        }
    }

    /// Whether this backend has the Phase-5-follow-on device-resident whole-encoder
    /// stack ([`Self::encode_prenorm_encoder`]): `true` on the GPU arms (Metal /
    /// CUDA), `false` on CPU.
    ///
    /// The caller (`whisper::encoder::encode`) gates the fused encoder on this:
    /// only a GPU backend routes the whole pre-norm block stack through
    /// `encode_prenorm_encoder` (one submission for the encoder); the CPU always
    /// runs the per-op `encoder_block` loop. This keeps the CPU arm of
    /// `encode_prenorm_encoder` an explicit [`VokraError::UnsupportedOp`] correct
    /// code never reaches (no silent fall back, FR-EX-08), while the block math
    /// lives in exactly one place (the CPU `encoder_block` loop is the single
    /// source of truth — `compute.rs` hosts no duplicated encoder loop).
    #[must_use]
    pub fn prenorm_stack_is_fused(&self) -> bool {
        match &self.be {
            Be::Cpu => false,
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => true,
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => true,
            // No device-resident encoder chain in the M4-01 slice (per-op
            // upload/dispatch/readback; residency is the M4-02+ follow-up).
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => false,
        }
    }

    /// Row-major GEMM with optional per-column bias
    /// (`out[i,j] = bias[j] + Σ_l a[i,l]·b[l,j]`); `a` is `m×k`, `b` is `k×n`.
    ///
    /// The GPU-accelerated op in this slice: the CPU arm calls
    /// [`kernels::gemm_f32`], the Metal arm the identically-typed
    /// `MetalContext::gemm_f32` (drop-in, M2-01-T18).
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set (matches kernels::gemm_f32)
    pub fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gemm_f32(m, n, k, a, b, bias, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.gemm_f32(m, n, k, a, b, bias, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.gemm_f32(m, n, k, a, b, bias, out),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => ctx.gemm_f32(m, n, k, a, b, bias, out),
        }
    }

    /// Row-major GEMM against a **K-quantized** weight
    /// (`out[t,j] = bias[j] + Σ_l a[t,l]·dequant(wq[j,l])`), the fused
    /// dequant-dot counterpart of [`Self::gemm_f32`] (M5-15-T27/T33).
    ///
    /// `a` is `[m, k]` and `out` is `[m, n]` exactly as for `gemm_f32`, but
    /// `wq` is the **untransposed** `[n, k]` GGUF payload — the layout the
    /// INT8 kernels want — so the quant route skips the `[out, in] → [in, out]`
    /// transpose the f32 loader pays. `m == 1` (the decoder step) routes into
    /// the single-activation GEMV kernel inside the driver, so this one entry
    /// serves both the GEMV and GEMM shapes that `whisper::nn::linear_apply`
    /// produces.
    ///
    /// # Backends
    ///
    /// **CPU only.** Every GPU arm is an explicit [`VokraError::UnsupportedOp`]:
    /// there is no fused K-quant kernel in Metal / CUDA / WebGPU in this WP,
    /// and silently dequantizing (or silently running on the CPU) is exactly
    /// the fallback FR-EX-08 forbids. Callers avoid this arm by loading
    /// without `WhisperLoadOptions::fused_quant_weights` on a GPU backend; the
    /// arm exists so a mistake is *noticed*.
    #[allow(clippy::too_many_arguments)] // mirrors gemm_f32 plus the weight dtype
    pub fn gemm_q_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        wq: &[u8],
        dtype: KQuantDtype,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gemm_q_f32(m, n, k, a, wq, dtype, bias, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => Err(unsupported_quant_gemm("metal")),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(unsupported_quant_gemm("cuda")),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(unsupported_quant_gemm("webgpu")),
        }
    }

    /// Row-major matrix-vector product with optional per-row bias
    /// (`out[i] = bias[i] + Σ_l a[i,l]·x[l]`); `a` is `m×k`.
    pub fn gemv_f32(
        &self,
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gemv_f32(m, k, a, x, bias, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.gemv_f32(m, k, a, x, bias, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.gemv_f32(m, k, a, x, bias, out),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => ctx.gemv_f32(m, k, a, x, bias, out),
        }
    }

    /// Row-wise softmax over the innermost axis of a `rows × cols` buffer.
    pub fn softmax_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::softmax_f32(input, out, rows, cols),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.softmax_f32(input, out, rows, cols),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.softmax_f32(input, out, rows, cols),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => ctx.softmax_f32(input, out, rows, cols),
        }
    }

    /// Affine layer normalisation over the innermost axis of a `rows × cols`
    /// buffer (`gamma` / `beta` length `cols`).
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set (matches kernels::layer_norm_f32)
    pub fn layer_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
        }
    }

    /// Element-wise exact (erf) GELU (`x` and `out` equal length).
    pub fn gelu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gelu_f32(x, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.gelu_f32(x, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.gelu_f32(x, out),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => ctx.gelu_f32(x, out),
        }
    }

    /// 1-D convolution via im2col + GEMM (`input` is `in_ch × in_len`, `weight`
    /// is `out_ch × in_ch × kernel`, `out` is `out_ch × out_len`).
    #[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set (matches kernels::conv1d_f32)
    pub fn conv1d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            ),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            ),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => ctx.conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            ),
        }
    }

    /// Mimi (Kyutai) residual vector quantization codec decode — the M3-06
    /// codec op wired into the imperative `Compute` seam.
    ///
    /// Given a `[time, n_codebooks]` row-major slice of `u32` `codes` and one
    /// [`CodebookTable`] per codebook (each `[codebook_size, d_model]`
    /// row-major), returns a fresh `[time, d_model]` row-major `Vec<f32>` of
    /// feature vectors reconstructed by summing every codebook's contribution
    /// in FP32 (see [`vokra_ops::mimi_rvq_decode`] for the algorithm).
    ///
    /// # Heterogeneous shape (owned `Vec<f32>`, not `out: &mut [f32]`)
    ///
    /// Unlike the other seam methods (which take `out: &mut [f32]` for the
    /// zero-alloc reserve, FR-EX-05), this method returns a freshly-allocated
    /// `Vec<f32>`. The reason is baked into [`vokra_ops::mimi_rvq_decode`]:
    /// the op is a codebook-table fold shaped by `Vec<CodebookTable>`
    /// (heterogeneous width across callers) rather than a plain M×N GEMM,
    /// which is also why `mimi_rvq_decode` is a runtime function in
    /// `vokra-ops` and not an [`vokra_core::OpKind`] variant (see the module
    /// docs in `vokra_ops::mimi_rvq`). The heap alloc is negligible because
    /// M3-09 (CosyVoice2) calls this at chunk granularity, not at the
    /// per-token hot-path granularity the GEMM seam serves.
    ///
    /// # CPU-only through this seam today (Metal / CUDA arms return `UnsupportedOp`)
    ///
    /// The CPU arm delegates verbatim to [`vokra_ops::mimi_rvq_decode`]
    /// (M3-06 T04 kernel; bit-for-bit reproduces a direct kernel call, so a
    /// `Compute::cpu()` run reproduces the pre-seam output exactly). The
    /// **Metal** and **CUDA** arms return an explicit
    /// [`VokraError::UnsupportedOp`] because the M3-06 T14 (MSL) / T15 (NVRTC)
    /// GPU kernels are still deferred to the M3-09 mimi_bridge upgrade past
    /// stub — this is the honest state today and is *never* a silent CPU
    /// fall back (FR-EX-08). The coverage gate on
    /// [`Compute::for_backend`] additionally rejects any model that lists
    /// [`HotOp::MimiRvq`] against Metal / CUDA / Vulkan, so a well-behaved
    /// consumer never reaches this method through those arms; the explicit
    /// error here is the belt-and-braces defence for any consumer that
    /// bypassed the coverage gate (e.g. built a `Compute::for_backend(Metal,
    /// &[])` with an empty required set and then reached for
    /// `mimi_rvq_f32`).
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`vokra_ops::mimi_rvq_decode`] raises (shape mismatch, out-of-range
    ///   codebook index; never a silent 0-clamp — FR-EX-08).
    /// - Metal / CUDA arms: explicit [`VokraError::UnsupportedOp`] until the
    ///   M3-06 T14 / T15 GPU kernels land.
    pub fn mimi_rvq_f32(
        &self,
        codes: &[u32],
        time: usize,
        codebook_tables: &[CodebookTable],
        attrs: &MimiRvqAttrs,
    ) -> Result<Vec<f32>> {
        match &self.be {
            Be::Cpu => mimi_rvq_decode(codes, time, codebook_tables, attrs),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => Err(VokraError::UnsupportedOp(
                "mimi_rvq_f32 has no wired Metal MSL kernel; the M3-06 T14 GPU arm is deferred to \
                 the M3-09 mimi_bridge upgrade past stub. Select BackendKind::Cpu (which \
                 delegates to vokra_ops::mimi_rvq_decode), or wait for the Metal kernel — \
                 Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "mimi_rvq_f32 has no wired CUDA NVRTC kernel; the M3-06 T15 GPU arm is deferred \
                 to the M3-09 mimi_bridge upgrade past stub. Select BackendKind::Cpu (which \
                 delegates to vokra_ops::mimi_rvq_decode), or wait for the CUDA kernel — \
                 Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "mimi_rvq_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six Whisper hot \
                 ops only; the RVQ codec GPU arms are deferred like Metal/CUDA). Select \
                 BackendKind::Cpu — Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// DAC (Descript) factorized residual VQ codec decode — the M4-04 op
    /// wired into the imperative `Compute` seam (mirror of
    /// [`Compute::mimi_rvq_f32`], plus the per-quantizer projection
    /// operands).
    ///
    /// Given `[time, n_codebooks]` `codes`, one low-dim [`CodebookTable`] and
    /// one [`DacOutProj`] per quantizer, returns a fresh `[time, d_model]`
    /// `Vec<f32>`: `out[t,:] = Σ_cb (W_cb @ codebook_cb[codes[t,cb]] + b_cb)`
    /// in FP32 (see [`vokra_ops::dac_rvq_decode`]). Heap-returning for the
    /// same heterogeneous-signature reason as `mimi_rvq_f32` (chunk
    /// granularity, not per-token hot path).
    ///
    /// # CPU-only through this seam today
    ///
    /// The CPU arm delegates verbatim to [`vokra_ops::dac_rvq_decode`]
    /// (bit-for-bit vs a direct kernel call); the **Metal** / **CUDA** arms
    /// return an explicit [`VokraError::UnsupportedOp`] — the M4-04 GPU
    /// kernels are deferred, and Vokra never silently substitutes the CPU
    /// (FR-EX-08). The [`Compute::for_backend`] coverage gate additionally
    /// rejects any model listing [`HotOp::DacRvq`] against Metal / CUDA /
    /// Vulkan.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates [`vokra_ops::dac_rvq_decode`]'s
    ///   [`VokraError::InvalidArgument`] (shape mismatch, out-of-range index).
    /// - Metal / CUDA arms: explicit [`VokraError::UnsupportedOp`].
    pub fn dac_rvq_f32(
        &self,
        codes: &[u32],
        time: usize,
        codebook_tables: &[CodebookTable],
        out_projs: &[DacOutProj],
        attrs: &DacRvqAttrs,
    ) -> Result<Vec<f32>> {
        match &self.be {
            Be::Cpu => dac_rvq_decode(codes, time, codebook_tables, out_projs, attrs),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => Err(VokraError::UnsupportedOp(
                "dac_rvq_f32 has no wired Metal MSL kernel; the M4-04 GPU arm is deferred (naive \
                 gather + GEMV + fold layout, same follow-up as mimi_rvq). Select \
                 BackendKind::Cpu (which delegates to vokra_ops::dac_rvq_decode) — Vokra does \
                 not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "dac_rvq_f32 has no wired CUDA NVRTC kernel; the M4-04 GPU arm is deferred (naive \
                 gather + GEMV + fold layout, same follow-up as mimi_rvq). Select \
                 BackendKind::Cpu (which delegates to vokra_ops::dac_rvq_decode) — Vokra does \
                 not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "dac_rvq_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six Whisper hot \
                 ops only). Select BackendKind::Cpu — no silent CPU fall back (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// EnCodec residual VQ codec decode — the M4-04 engine-only op wired into
    /// the imperative `Compute` seam (FR-OP-32: the op exists, the pretrained
    /// weights are permanently zoo-excluded — see `vokra_ops::encodec_rvq`
    /// module docs).
    ///
    /// Same shape-generic gather + FP32 fold as [`Compute::mimi_rvq_f32`];
    /// the CPU arm delegates verbatim to [`vokra_ops::encodec_rvq_decode`],
    /// the Metal / CUDA arms are explicit [`VokraError::UnsupportedOp`]
    /// (FR-EX-08 — no silent CPU fall back), and the coverage gate rejects
    /// [`HotOp::EncodecRvq`] against every GPU backend.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates [`vokra_ops::encodec_rvq_decode`]'s
    ///   [`VokraError::InvalidArgument`].
    /// - Metal / CUDA arms: explicit [`VokraError::UnsupportedOp`].
    pub fn encodec_rvq_f32(
        &self,
        codes: &[u32],
        time: usize,
        codebook_tables: &[CodebookTable],
        attrs: &EncodecRvqAttrs,
    ) -> Result<Vec<f32>> {
        match &self.be {
            Be::Cpu => encodec_rvq_decode(codes, time, codebook_tables, attrs),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => Err(VokraError::UnsupportedOp(
                "encodec_rvq_f32 has no wired Metal MSL kernel; the M4-04 GPU arm is deferred. \
                 Select BackendKind::Cpu (which delegates to vokra_ops::encodec_rvq_decode) — \
                 Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "encodec_rvq_f32 has no wired CUDA NVRTC kernel; the M4-04 GPU arm is deferred. \
                 Select BackendKind::Cpu (which delegates to vokra_ops::encodec_rvq_decode) — \
                 Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "encodec_rvq_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six Whisper \
                 hot ops only). Select BackendKind::Cpu — no silent CPU fall back (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// WavTokenizer single-codebook VQ decode — the M4-16 **FSQ-family** op
    /// wired into the imperative `Compute` seam (FR-OP-31: single-stage,
    /// deliberately a *separate subgraph* from the RVQ family — module docs
    /// in `vokra_ops::fsq_codec`).
    ///
    /// Given `[time]` codes and **one** [`CodebookTable`] (singular — the
    /// signature-level distinction from the RVQ methods' `&[CodebookTable]`),
    /// returns a fresh `[time, d_model]` `Vec<f32>` of gathered embedding
    /// rows (bit-exact single gather per timestep; see
    /// [`vokra_ops::wavtokenizer_vq_decode`]). Heap-returning for the same
    /// heterogeneous-signature reason as [`Compute::mimi_rvq_f32`] (chunk
    /// granularity, not the per-token GEMM hot path).
    ///
    /// # CPU-only through this seam today
    ///
    /// The CPU arm delegates verbatim to
    /// [`vokra_ops::wavtokenizer_vq_decode`] (bit-for-bit vs a direct call);
    /// the **Metal** / **CUDA** arms return an explicit
    /// [`VokraError::UnsupportedOp`] — the M4-16 GPU kernels are deferred
    /// (single-stage gather/GEMV bound: they will reuse the existing M2-01 /
    /// M2-03 kernels), and Vokra never silently substitutes the CPU
    /// (FR-EX-08). The [`Compute::for_backend`] coverage gate additionally
    /// rejects any model listing [`HotOp::WavTokenizerVq`] against Metal /
    /// CUDA / Vulkan.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates [`vokra_ops::wavtokenizer_vq_decode`]'s
    ///   [`VokraError::InvalidArgument`] (shape mismatch, out-of-range code).
    /// - Metal / CUDA arms: explicit [`VokraError::UnsupportedOp`].
    pub fn wavtokenizer_vq_f32(
        &self,
        codes: &[u32],
        time: usize,
        codebook_table: &CodebookTable,
        attrs: &WavTokenizerVqAttrs,
    ) -> Result<Vec<f32>> {
        match &self.be {
            Be::Cpu => wavtokenizer_vq_decode(codes, time, codebook_table, attrs),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => Err(VokraError::UnsupportedOp(
                "wavtokenizer_vq_f32 has no wired Metal MSL kernel; the M4-16 GPU arm is \
                 deferred (single-stage gather — reuses the M2-01 kernels when it lands). \
                 Select BackendKind::Cpu (which delegates to \
                 vokra_ops::wavtokenizer_vq_decode) — Vokra does not silently run the op on \
                 the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "wavtokenizer_vq_f32 has no wired CUDA NVRTC kernel; the M4-16 GPU arm is \
                 deferred (single-stage gather — reuses the M2-03 kernels when it lands). \
                 Select BackendKind::Cpu (which delegates to \
                 vokra_ops::wavtokenizer_vq_decode) — Vokra does not silently run the op on \
                 the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "wavtokenizer_vq_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six \
                 Whisper hot ops only; the FSQ codec GPU arms are deferred like Metal/CUDA). \
                 Select BackendKind::Cpu (which delegates to \
                 vokra_ops::wavtokenizer_vq_decode) — Vokra does not silently run the op on \
                 the CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// X-Codec 2 FSQ dequant — the M4-16 FSQ-family sibling of
    /// [`Compute::wavtokenizer_vq_f32`] (FR-OP-31 single-stage GEMV bound;
    /// implicit per-dimension grid, **no codebook tensor**, one
    /// out-projection GEMV per timestep — see
    /// [`vokra_ops::xcodec2_fsq_decode`]).
    ///
    /// # CPU-only through this seam today
    ///
    /// The CPU arm delegates verbatim to [`vokra_ops::xcodec2_fsq_decode`];
    /// the **Metal** / **CUDA** arms are explicit
    /// [`VokraError::UnsupportedOp`] (FR-EX-08 — no silent CPU fall back),
    /// and the coverage gate rejects [`HotOp::Xcodec2Fsq`] against every GPU
    /// backend.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates [`vokra_ops::xcodec2_fsq_decode`]'s
    ///   [`VokraError::InvalidArgument`].
    /// - Metal / CUDA arms: explicit [`VokraError::UnsupportedOp`].
    pub fn xcodec2_fsq_f32(
        &self,
        codes: &[u32],
        time: usize,
        out_proj: Option<&FsqOutProj>,
        attrs: &Xcodec2FsqAttrs,
    ) -> Result<Vec<f32>> {
        match &self.be {
            Be::Cpu => xcodec2_fsq_decode(codes, time, out_proj, attrs),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => Err(VokraError::UnsupportedOp(
                "xcodec2_fsq_f32 has no wired Metal MSL kernel; the M4-16 GPU arm is deferred \
                 (single-stage GEMV — reuses the M2-01 kernels when it lands). Select \
                 BackendKind::Cpu (which delegates to vokra_ops::xcodec2_fsq_decode) — Vokra \
                 does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "xcodec2_fsq_f32 has no wired CUDA NVRTC kernel; the M4-16 GPU arm is deferred \
                 (single-stage GEMV — reuses the M2-03 kernels when it lands). Select \
                 BackendKind::Cpu (which delegates to vokra_ops::xcodec2_fsq_decode) — Vokra \
                 does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "xcodec2_fsq_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six Whisper \
                 hot ops only; the FSQ codec GPU arms are deferred like Metal/CUDA). Select \
                 BackendKind::Cpu (which delegates to vokra_ops::xcodec2_fsq_decode) — Vokra \
                 does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// Fused MLP `fc2(gelu(fc1(x)))` — the Phase-5 device-residency slice.
    ///
    /// `x` is `[t, d]`; `fc1` maps `d → ffn` (`fc1_w` is `[d, ffn]`, bias
    /// `[ffn]`); `fc2` maps `ffn → d` (`fc2_w` is `[ffn, d]`, bias `[d]`); `out`
    /// is `[t, d]`. `mlp_h` / `mlp_a` are the two `[t, ffn]` intermediates.
    ///
    /// On the **CPU** arm this is the identical three-kernel sequence
    /// (`gemm_f32` → `gelu_f32` → `gemm_f32`, into `mlp_h` / `mlp_a`) the
    /// pre-fusion `whisper::nn::mlp_into` ran, so it is **bit-for-bit** the
    /// pre-seam result (the parity suites stay green). On the **Metal / CUDA**
    /// arms the same three kernels run in ONE GPU submission with the two
    /// `[t, ffn]` intermediates resident on the device — only `out` is read back
    /// — which is bit-identical to three separate GPU ops but pays one readback /
    /// one sync instead of three. `mlp_h` / `mlp_a` are unused on the GPU arms
    /// (the device holds those intermediates); the caller still sizes them so the
    /// CPU arm and the zero-alloc reserve are unaffected.
    #[allow(clippy::too_many_arguments)] // fused-MLP operand set (two Linears + scratch + dims)
    pub fn mlp_f32(
        &self,
        t: usize,
        d: usize,
        ffn: usize,
        x: &[f32],
        fc1_w: &[f32],
        fc1_bias: Option<&[f32]>,
        fc2_w: &[f32],
        fc2_bias: Option<&[f32]>,
        mlp_h: &mut [f32],
        mlp_a: &mut [f32],
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => {
                // Bit-identical to the former `mlp_into`: fc1 GEMM → GELU → fc2
                // GEMM through the same CPU kernels, in the same order, into the
                // caller's scratch.
                kernels::gemm_f32(t, ffn, d, x, fc1_w, fc1_bias, mlp_h)?;
                kernels::gelu_f32(mlp_h, mlp_a)?;
                kernels::gemm_f32(t, d, ffn, mlp_a, fc2_w, fc2_bias, out)
            }
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.mlp_f32(t, d, ffn, x, fc1_w, fc1_bias, fc2_w, fc2_bias, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.mlp_f32(t, d, ffn, x, fc1_w, fc1_bias, fc2_w, fc2_bias, out),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => {
                // Same fc1 GEMM → GELU → fc2 GEMM chain as the CPU arm, per-op
                // through the WGSL kernels into the caller's scratch (no fused
                // MLP kernel in the M4-01 slice — honest per-op mode).
                ctx.gemm_f32(t, ffn, d, x, fc1_w, fc1_bias, mlp_h)?;
                ctx.gelu_f32(mlp_h, mlp_a)?;
                ctx.gemm_f32(t, d, ffn, mlp_a, fc2_w, fc2_bias, out)
            }
        }
    }

    /// Fused **non-causal** multi-head attention — the Phase-5 device-residency
    /// slice (the sibling of [`Self::mlp_f32`]).
    ///
    /// `xq` is `[t_q, d]`; `k` / `v` are the pre-projected `[t_kv, d]` keys /
    /// values; `q_w` / `out_w` are `[d, d]` (both projections `d → d`), biases
    /// `[d]`; `scale = head_dim^-0.5` (the caller folds the query scale in);
    /// `out` is `[t_q, d]`.
    ///
    /// **GPU-only.** On the Metal / CUDA arms this runs the q-proj → per-head
    /// {gather, QKᵀ, softmax, A·V, scatter} → out-proj chain in ONE GPU
    /// submission with every intermediate resident on the device (bit-identical
    /// to the per-op path, one readback instead of many). The **CPU arm is an
    /// explicit [`VokraError::UnsupportedOp`]**: the CPU never fuses attention —
    /// it runs the per-op head loop in `whisper::nn::attention_from_kv_into`,
    /// which gates this call behind [`Self::attention_is_fused`], so correct code
    /// never hits the CPU arm. This keeps the attention math in exactly one place
    /// (nn.rs) with no silent CPU fall back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] on the CPU arm; the GPU arms return
    /// [`VokraError::InvalidArgument`] on a shape mismatch and
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[allow(clippy::too_many_arguments)]
    // fused-attention operand set (two Linears + K/V + dims)
    // Without a GPU arm compiled in, only the CPU arm (which reads none of the
    // operands, just returns UnsupportedOp) remains, so every operand is unused —
    // exactly as `for_backend` cfg-silences its `required` argument.
    #[cfg_attr(
        not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows))
        )),
        allow(unused_variables)
    )]
    pub fn attn_f32(
        &self,
        t_q: usize,
        t_kv: usize,
        d: usize,
        n_head: usize,
        xq: &[f32],
        q_w: &[f32],
        q_bias: Option<&[f32]>,
        k: &[f32],
        v: &[f32],
        out_w: &[f32],
        out_bias: Option<&[f32]>,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => Err(VokraError::UnsupportedOp(
                "attn_f32 is the GPU fused attention path; the CPU uses the per-op attention loop \
                 (whisper::nn::attention_from_kv_into gates it behind Compute::attention_is_fused)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.attn_f32(
                t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.attn_f32(
                t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, out,
            ),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "webgpu has no fused-attention chain in the M4-01 slice; correct code never \
                 reaches this arm because `attention_is_fused()` is false for WebGPU — the \
                 caller runs the per-op head loop (standard GEMM + softmax; FA v3 red line). \
                 No silent CPU fall back (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// Device-resident **whole pre-norm encoder** (Phase-5 follow-on): runs
    /// `n × [ln → attn → residual → ln → mlp → residual]` + a final LayerNorm with
    /// the hidden state and every intermediate kept on the GPU across all blocks,
    /// so the encoder pays ONE submission instead of the per-op path's `6·N + 1`.
    ///
    /// `hidden` is the `[t, d]` post-conv-stem input, `out` the `[t, d]` normed
    /// output; `layers` are the per-block weight slices; `n_head` splits `d`;
    /// `eps` is the LayerNorm epsilon (the caller passes the CPU-kernel constant,
    /// which the backend cannot import).
    ///
    /// **GPU-only.** On the Metal / CUDA arms this is bit-identical to running the
    /// blocks per-op on the GPU (same kernels, order, geometry) and matches the CPU
    /// within the FP32 bound. The **CPU arm is an explicit
    /// [`VokraError::UnsupportedOp`]**: the CPU never fuses the encoder — it runs
    /// the per-op `encoder_block` loop in `whisper::encoder`, which gates this call
    /// behind [`Self::prenorm_stack_is_fused`], so correct code never hits the CPU
    /// arm (no silent CPU fall back, FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] on the CPU arm; the GPU arms return
    /// [`VokraError::InvalidArgument`] on a shape mismatch and
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[allow(clippy::too_many_arguments)] // whole-encoder operand set (dims + weights + I/O)
    #[cfg_attr(
        not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows))
        )),
        allow(unused_variables)
    )]
    pub fn encode_prenorm_encoder(
        &self,
        t: usize,
        d: usize,
        ff: usize,
        n_head: usize,
        eps: f32,
        hidden: &[f32],
        layers: &[PrenormLayer<'_>],
        final_ln_gamma: &[f32],
        final_ln_beta: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => Err(VokraError::UnsupportedOp(
                "encode_prenorm_encoder is the GPU device-resident encoder path; the CPU uses the \
                 per-op encoder_block loop (whisper::encoder::encode gates it behind \
                 Compute::prenorm_stack_is_fused)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.encode_prenorm_stack(
                t,
                d,
                ff,
                n_head,
                eps,
                hidden,
                layers,
                final_ln_gamma,
                final_ln_beta,
                out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.encode_prenorm_stack(
                t,
                d,
                ff,
                n_head,
                eps,
                hidden,
                layers,
                final_ln_gamma,
                final_ln_beta,
                out,
            ),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "webgpu has no device-resident pre-norm encoder chain in the M4-01 slice; \
                 correct code never reaches this arm because `prenorm_stack_is_fused()` is \
                 false for WebGPU — the caller runs the per-op encoder_block loop. Whole-run \
                 residency is the M4-02+ follow-up. No silent CPU fall back (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// Whether this backend has the Phase-3 device-resident **decoder-step**
    /// session ([`Self::new_decoder_step_session`] on Metal (Phase 3a) and CUDA
    /// (Phase 3b)).
    ///
    /// The caller (`whisper::decoder::DecoderState`) gates the whole-step device
    /// path on this: only a session-backed backend builds a
    /// [`DecoderStepSession`] at construction and routes every step through it;
    /// CPU keeps the per-op step loop untouched. This keeps
    /// [`Self::new_decoder_step_session`]'s CPU arm an explicit
    /// [`VokraError::UnsupportedOp`] correct code never hits (no silent fall
    /// back, FR-EX-08), with zero duplicated decode-block math in `compute.rs`.
    #[must_use]
    pub fn decoder_step_is_session_backed(&self) -> bool {
        match &self.be {
            Be::Cpu => false,
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => true,
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => true,
            // No device-resident decoder-step session in the M4-01 slice —
            // the decoder runs the per-op CPU-shaped loop through the WGSL
            // kernels (honest per-op mode; session residency is follow-up).
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => false,
        }
    }

    /// Builds a device-resident **decoder-step session** — the Phase-3 device-
    /// residency slice sibling of [`Self::encode_prenorm_encoder`], for the
    /// autoregressive decode (weights uploaded once, self-attention K/V kept on
    /// the GPU and appended each step, cross-attention K/V uploaded once from
    /// the pre-projected slices in `layers`).
    ///
    /// `dims` names the model shape; `layers` carries every decoder block's
    /// weight slices (row-major, `[in, out]` layout — the same layout the CPU
    /// per-op path uses) plus the pre-projected cross-K/V; `token_emb` is the
    /// tied-head / embedding table `[n_vocab, d]`; `ln_post_gamma` /
    /// `ln_post_beta` are the final decoder LayerNorm.
    ///
    /// **GPU-only.** On the Metal (Phase 3a) or CUDA (Phase 3b) arm this
    /// returns a session ready for [`DecoderStepSession::step`] (one GPU
    /// submission + one full `[t, n_vocab]` logits readback per step;
    /// bit-identical to running the step per-op on the GPU). The **CPU arm is
    /// an explicit [`VokraError::UnsupportedOp`]** — the CPU never fuses the
    /// decoder step; the model layer gates this call behind
    /// [`Self::decoder_step_is_session_backed`], so correct code never hits the
    /// CPU arm. No silent CPU fall back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] on the CPU arm; the Metal / CUDA arms
    /// return [`VokraError::InvalidArgument`] on a shape mismatch and
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[cfg_attr(
        not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows))
        )),
        allow(unused_variables)
    )]
    pub fn new_decoder_step_session(
        &self,
        dims: DecoderStepDims,
        layers: &[DecoderLayerView<'_>],
        token_emb: &[f32],
        ln_post_gamma: &[f32],
        ln_post_beta: &[f32],
    ) -> Result<DecoderStepSession> {
        match &self.be {
            Be::Cpu => Err(VokraError::UnsupportedOp(
                "new_decoder_step_session is the GPU device-resident decoder-step driver; the CPU \
                 runs the per-op step loop (whisper::decoder::DecoderState gates it behind \
                 Compute::decoder_step_is_session_backed)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => {
                // The session owns its own `MetalContext` (weights + KV live inside
                // it; the outer `Compute`'s context is used only for the cross-KV
                // precompute at construction and dropped afterwards). Bit-identical
                // to the per-op Metal path within the FP32 bound: same kernels,
                // same launch geometry, one command-buffer submission per step.
                let s = vokra_backend_metal::MetalDecodeSession::new(
                    dims.d,
                    dims.n_head,
                    dims.ff,
                    dims.n_text_ctx,
                    dims.n_vocab,
                    dims.n_ctx,
                    dims.max_t_q,
                    dims.eps,
                    layers,
                    token_emb,
                    ln_post_gamma,
                    ln_post_beta,
                )?;
                Ok(DecoderStepSession::Metal(Box::new(s)))
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => {
                // Same construction contract as the Metal arm: the session owns
                // its own `CudaContext` (weights + KV live inside it; the outer
                // `Compute`'s context is used only for the cross-KV precompute
                // at construction and dropped afterwards). Bit-identical to the
                // per-op CUDA path within the FP32 bound — same NVRTC kernels,
                // same launch geometry — with ONE `cuStreamSynchronize` per step
                // instead of the per-op path's per-op synchronise.
                let s = vokra_backend_cuda::CudaDecodeSession::new(
                    dims.d,
                    dims.n_head,
                    dims.ff,
                    dims.n_text_ctx,
                    dims.n_vocab,
                    dims.n_ctx,
                    dims.max_t_q,
                    dims.eps,
                    layers,
                    token_emb,
                    ln_post_gamma,
                    ln_post_beta,
                )?;
                Ok(DecoderStepSession::Cuda(Box::new(s)))
            }
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "webgpu has no device-resident decoder-step session in the M4-01 slice; correct \
                 code never reaches this arm because `decoder_step_is_session_backed()` is \
                 false for WebGPU. No silent CPU fall back (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }
}

/// Immutable model shape for a device-resident decoder-step session
/// ([`Compute::new_decoder_step_session`]).
///
/// Names the dims the backend needs at build time to size its resident buffers
/// once (`n_text_ctx` bounds the self-attention KV cache; `max_t_q` bounds the
/// per-step scratch and the tied-head logits buffer; `n_ctx` matches the encoder
/// output width the pre-projected cross-K/V is `[n_ctx, d]` rows of). `eps` is
/// the LayerNorm epsilon (the caller passes the CPU-kernel constant, which the
/// backend cannot import).
#[derive(Clone, Copy, Debug)]
pub struct DecoderStepDims {
    /// Hidden width.
    pub d: usize,
    /// Attention head count (must divide `d`).
    pub n_head: usize,
    /// MLP inner width.
    pub ff: usize,
    /// Max decoder-context length (the hard self-attention KV cache bound).
    pub n_text_ctx: usize,
    /// Vocabulary size (the tied logits head output width).
    pub n_vocab: usize,
    /// Encoder context length (the cross-attention key window; the
    /// pre-projected `cross_k` / `cross_v` in each [`DecoderLayerView`] are
    /// `[n_ctx, d]` rows).
    pub n_ctx: usize,
    /// Widest single decode step's query length (the forced-prefix width;
    /// steady-state greedy decodes one token). Bounds the per-step scratch and
    /// the `[max_t_q, n_vocab]` logits buffer.
    pub max_t_q: usize,
    /// LayerNorm epsilon (the backend cannot import the CPU-kernel constant).
    pub eps: f32,
}

/// A backend-specific device-resident **decoder-step session** — the
/// autoregressive-decode sibling of [`Compute::encode_prenorm_encoder`].
///
/// Built once at [`Compute::new_decoder_step_session`] (Metal Phase 3a and
/// CUDA Phase 3b). Each [`Self::step`] runs the whole decode step device-
/// resident in ONE GPU submission, then reads back the full `[t, n_vocab]`
/// logits so the model layer can compare against the CPU decoder's row-major
/// output (not only the greedy last row). The session owns its own backend
/// context (weights + KV live inside it), so a Metal / CUDA
/// [`DecoderStepSession`] holds Metal / CUDA handles — see the
/// `unsafe impl Send` notes on the backend types for why the model layer can
/// still hold it inside a `Send` `DecoderState`.
pub enum DecoderStepSession {
    /// Metal (M2 Phase 3a) device-resident decoder-step session.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    Metal(Box<vokra_backend_metal::MetalDecodeSession>),
    /// CUDA (M2 Phase 3b) device-resident decoder-step session.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    Cuda(Box<vokra_backend_cuda::CudaDecodeSession>),
}

// `DecoderStepSession` is `Send` because every variant boxes a backend session
// (`MetalDecodeSession` / `CudaDecodeSession`) the backend crate declares `Send`
// via `unsafe impl` (see each backend crate for the SAFETY rationale — Metal
// handles are documented thread-safe; CUDA context / stream / module handles
// are transferable via the driver's `cuCtxSetCurrent` contract). Those unsafe
// impls live behind each backend's `#![allow(unsafe_code)]` opt-out because
// `vokra-models` stays under the workspace `unsafe_code = deny`. The model-
// layer `DecoderState` therefore stays `Send` (its `assert_send::<DecoderState>()`
// compile-time bound + the cross-thread decode test both continue to hold
// under `--features cuda`) without either reuploading every weight per step or
// duplicating attention math in `compute.rs`.

impl DecoderStepSession {
    /// Advances the decode by the `t` tokens whose `[t, d]` token+positional
    /// embedding is `embedded`, starting at the committed position `start`.
    /// Runs the whole step device-resident in ONE GPU submission + ONE
    /// `[t, n_vocab]` logits readback (bit-identical to running the step
    /// per-op on the GPU).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a bad `t` / `start` / `embedded`
    /// length; [`VokraError::BackendUnavailable`] on a device failure.
    #[cfg_attr(
        not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows))
        )),
        allow(unused_variables)
    )]
    pub fn step(&mut self, embedded: &[f32], t: usize, start: usize) -> Result<()> {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.step(embedded, t, start),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            DecoderStepSession::Cuda(s) => s.step(embedded, t, start),
            // Off every session-backed build the enum is uninhabited (no
            // variants) and `Option<DecoderStepSession>` is only ever `None`;
            // the model-layer caller (`whisper::decoder::DecoderState`) never
            // constructs a session and so never calls this. `Self` still
            // contains fields (`&mut self` binding), so the match falls through
            // the empty never-reachable wildcard.
            #[cfg(not(any(
                all(feature = "metal", any(target_os = "macos", target_os = "ios")),
                all(feature = "cuda", any(unix, windows))
            )))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal/CUDA build"),
        }
    }

    /// Rewinds the decode position to 0 for a fresh decode of the same audio
    /// (resident weights + cross-KV stay valid; the self-attention KV rows are
    /// simply overwritten from row 0). Mirrors [`vokra_core::KvCache::reset`].
    pub fn reset(&mut self) {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.reset(),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            DecoderStepSession::Cuda(s) => s.reset(),
            #[cfg(not(any(
                all(feature = "metal", any(target_os = "macos", target_os = "ios")),
                all(feature = "cuda", any(unix, windows))
            )))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal/CUDA build"),
        }
    }

    /// Committed token positions in the self-attention cache (the causal query
    /// offset for the next [`Self::step`]).
    #[must_use]
    pub fn positions(&self) -> usize {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.positions(),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            DecoderStepSession::Cuda(s) => s.positions(),
            #[cfg(not(any(
                all(feature = "metal", any(target_os = "macos", target_os = "ios")),
                all(feature = "cuda", any(unix, windows))
            )))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal/CUDA build"),
        }
    }

    /// The last decoded row of the last [`Self::step`] — `[n_vocab]` logits,
    /// the greedy / argmax read. Empty before the first step.
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.last_logits(),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            DecoderStepSession::Cuda(s) => s.last_logits(),
            #[cfg(not(any(
                all(feature = "metal", any(target_os = "macos", target_os = "ios")),
                all(feature = "cuda", any(unix, windows))
            )))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal/CUDA build"),
        }
    }

    /// All `[t, n_vocab]` rows the last [`Self::step`] wrote, row-major
    /// (row `i` at offset `i·n_vocab`). This is the full-row output the model-
    /// layer path compares against the CPU decoder's `[t, n_vocab]` logits (not
    /// only the greedy last row). Empty before the first step.
    #[must_use]
    pub fn all_logits(&self) -> &[f32] {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.all_logits(),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            DecoderStepSession::Cuda(s) => s.all_logits(),
            #[cfg(not(any(
                all(feature = "metal", any(target_os = "macos", target_os = "ios")),
                all(feature = "cuda", any(unix, windows))
            )))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal/CUDA build"),
        }
    }
}

/// Builds a boxed [`Backend`] for the graph evaluator ([`vokra_core::run_graph`])
/// — the assembly-layer factory (§2.4). Distinct from [`Compute`], which is the
/// imperative seam; both ultimately drive the same per-(backend, op) kernels.
///
/// # Errors
///
/// [`VokraError::BackendUnavailable`] if `kind` is not built into this binary or
/// (for Metal) has no device.
pub fn make_backend(kind: BackendKind) -> Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(vokra_backend_cpu::CpuBackend::new())),
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        BackendKind::Metal => Ok(Box::new(vokra_backend_metal::MetalBackend::new()?)),
        #[cfg(all(feature = "cuda", any(unix, windows)))]
        BackendKind::Cuda => Ok(Box::new(vokra_backend_cuda::CudaBackend::new()?)),
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        BackendKind::Vulkan => Ok(Box::new(vokra_backend_vulkan::VulkanBackend::new()?)),
        #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
        BackendKind::WebGpu => Ok(Box::new(vokra_backend_webgpu::WebGpuBackend::new()?)),
        other => Err(VokraError::BackendUnavailable(format!(
            "{other:?} backend is not built into vokra-models (build with the `metal` feature on \
             macOS / iOS for Metal, the `cuda` feature on Windows / Linux for CUDA, the \
             `vulkan` feature on Linux / Android / Windows for Vulkan, or the `webgpu` feature \
             on wasm32 for browser WebGPU)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_compute_matches_direct_kernel_bit_for_bit() {
        // The whole point of the seam: `Compute::cpu()` must reproduce the direct
        // kernel call exactly (atol = 0), so the model parity suites stay green.
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let bias = [0.5f32, -0.5];

        let mut via_compute = vec![0.0f32; 4];
        Compute::cpu()
            .gemm_f32(2, 2, 3, &a, &b, Some(&bias), &mut via_compute)
            .unwrap();

        let mut direct = vec![0.0f32; 4];
        kernels::gemm_f32(2, 2, 3, &a, &b, Some(&bias), &mut direct).unwrap();

        assert_eq!(via_compute, direct, "Compute::cpu gemm != direct kernel");
    }

    #[test]
    fn cpu_mimi_rvq_f32_matches_direct_kernel_bit_for_bit() {
        // The M3-06 seam contract: `Compute::cpu().mimi_rvq_f32(...)` must
        // reproduce `vokra_ops::mimi_rvq_decode(...)` byte-identically, so a
        // future consumer switching from the free function to the seam pays
        // zero numeric cost. (Same guarantee `cpu_compute_matches_direct_kernel
        // _bit_for_bit` gives for `gemm_f32`.)
        let attrs = MimiRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            d_model: 4,
        };
        // Codebook 0: rows [0..4], [4..8], [8..12].
        let cb0 = CodebookTable::new(3, 4, (0..12).map(|i| i as f32).collect()).unwrap();
        // Codebook 1: rows [100..104], [104..108], [108..112] — distinct so
        // the fold across codebooks distinguishes them.
        let cb1 = CodebookTable::new(3, 4, (100..112).map(|i| i as f32).collect()).unwrap();
        let tables = vec![cb0, cb1];
        // time=3, n_cb=2 → codes.len() = 6.
        let codes = vec![0u32, 1, 2, 0, 1, 2];
        let time = 3;

        let via_compute = Compute::cpu()
            .mimi_rvq_f32(&codes, time, &tables, &attrs)
            .expect("cpu mimi_rvq_f32");
        let direct =
            mimi_rvq_decode(&codes, time, &tables, &attrs).expect("direct mimi_rvq_decode");
        assert_eq!(
            via_compute, direct,
            "Compute::cpu().mimi_rvq_f32 must byte-match vokra_ops::mimi_rvq_decode",
        );
    }

    #[test]
    fn cpu_mimi_rvq_f32_propagates_input_error() {
        // The seam does not wrap the kernel's `InvalidArgument` in anything —
        // it propagates verbatim so callers can special-case shape / index
        // mismatches without string-matching on a wrapped message.
        let attrs = MimiRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            d_model: 4,
        };
        let cb0 = CodebookTable::new(3, 4, vec![0.0; 12]).unwrap();
        let cb1 = CodebookTable::new(3, 4, vec![0.0; 12]).unwrap();
        let tables = vec![cb0, cb1];
        // Out-of-range codebook index (silent-clamp is forbidden — FR-EX-08).
        let codes = vec![0u32, /* out of range */ 42];
        let err = Compute::cpu()
            .mimi_rvq_f32(&codes, 1, &tables, &attrs)
            .expect_err("out-of-range codebook index must be an explicit error");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}",
        );
    }

    #[test]
    fn cpu_dac_rvq_f32_matches_direct_kernel_bit_for_bit() {
        // M4-04 T09 seam contract: `Compute::cpu().dac_rvq_f32(...)` must
        // reproduce `vokra_ops::dac_rvq_decode(...)` byte-identically.
        let attrs = DacRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            codebook_dim: 2,
            d_model: 4,
        };
        let tables = vec![
            CodebookTable::new(3, 2, (0..6).map(|i| i as f32).collect()).unwrap(),
            CodebookTable::new(3, 2, (10..16).map(|i| i as f32).collect()).unwrap(),
        ];
        let projs = vec![
            DacOutProj::new(
                4,
                2,
                (0..8).map(|i| i as f32 * 0.25).collect(),
                vec![0.5; 4],
            )
            .unwrap(),
            DacOutProj::new(
                4,
                2,
                (0..8).map(|i| 2.0 - i as f32 * 0.125).collect(),
                vec![-0.25; 4],
            )
            .unwrap(),
        ];
        let codes = vec![0u32, 2, 1, 0];
        let time = 2;

        let via_compute = Compute::cpu()
            .dac_rvq_f32(&codes, time, &tables, &projs, &attrs)
            .expect("cpu dac_rvq_f32");
        let direct =
            dac_rvq_decode(&codes, time, &tables, &projs, &attrs).expect("direct dac_rvq_decode");
        assert_eq!(
            via_compute, direct,
            "Compute::cpu().dac_rvq_f32 must byte-match vokra_ops::dac_rvq_decode",
        );
    }

    #[test]
    fn cpu_encodec_rvq_f32_matches_direct_kernel_bit_for_bit() {
        let attrs = EncodecRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            d_model: 4,
        };
        let tables = vec![
            CodebookTable::new(3, 4, (0..12).map(|i| i as f32).collect()).unwrap(),
            CodebookTable::new(3, 4, (100..112).map(|i| i as f32).collect()).unwrap(),
        ];
        let codes = vec![0u32, 1, 2, 0];
        let time = 2;

        let via_compute = Compute::cpu()
            .encodec_rvq_f32(&codes, time, &tables, &attrs)
            .expect("cpu encodec_rvq_f32");
        let direct =
            encodec_rvq_decode(&codes, time, &tables, &attrs).expect("direct encodec_rvq_decode");
        assert_eq!(
            via_compute, direct,
            "Compute::cpu().encodec_rvq_f32 must byte-match vokra_ops::encodec_rvq_decode",
        );
    }

    #[test]
    fn cpu_dac_and_encodec_rvq_f32_propagate_input_errors() {
        // Same verbatim-propagation contract as `mimi_rvq_f32` (FR-EX-08 —
        // out-of-range codes are explicit `InvalidArgument`, never a clamp).
        let dac_attrs = DacRvqAttrs {
            n_codebooks: 1,
            codebook_size: 2,
            codebook_dim: 2,
            d_model: 3,
        };
        let dac_tables = vec![CodebookTable::new(2, 2, vec![0.0; 4]).unwrap()];
        let dac_projs = vec![DacOutProj::new(3, 2, vec![0.0; 6], vec![0.0; 3]).unwrap()];
        let err = Compute::cpu()
            .dac_rvq_f32(&[9u32], 1, &dac_tables, &dac_projs, &dac_attrs)
            .expect_err("out-of-range DAC code must be an explicit error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));

        let enc_attrs = EncodecRvqAttrs {
            n_codebooks: 1,
            codebook_size: 2,
            d_model: 3,
        };
        let enc_tables = vec![CodebookTable::new(2, 3, vec![0.0; 6]).unwrap()];
        let err = Compute::cpu()
            .encodec_rvq_f32(&[7u32], 1, &enc_tables, &enc_attrs)
            .expect_err("out-of-range EnCodec code must be an explicit error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn cpu_wavtokenizer_vq_f32_matches_direct_kernel_bit_for_bit() {
        // M4-16 T09 seam contract: `Compute::cpu().wavtokenizer_vq_f32(...)`
        // must reproduce `vokra_ops::wavtokenizer_vq_decode(...)`
        // byte-identically (same guarantee the RVQ-family seam methods give).
        let attrs = WavTokenizerVqAttrs {
            vocab_size: 5,
            d_model: 3,
        };
        // Single codebook (FSQ family: singular table, not a slice).
        let table = CodebookTable::new(5, 3, (0..15).map(|i| i as f32).collect()).unwrap();
        let codes = vec![4u32, 0, 2];
        let time = 3;

        let via_compute = Compute::cpu()
            .wavtokenizer_vq_f32(&codes, time, &table, &attrs)
            .expect("cpu wavtokenizer_vq_f32");
        let direct = wavtokenizer_vq_decode(&codes, time, &table, &attrs)
            .expect("direct wavtokenizer_vq_decode");
        assert_eq!(
            via_compute, direct,
            "Compute::cpu().wavtokenizer_vq_f32 must byte-match \
             vokra_ops::wavtokenizer_vq_decode",
        );
    }

    #[test]
    fn cpu_xcodec2_fsq_f32_matches_direct_kernel_bit_for_bit() {
        // M4-16 T09 seam contract for the FSQ dequant + out-projection GEMV.
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 3,
        };
        let proj = FsqOutProj::new(
            3,
            2,
            (0..6).map(|i| i as f32 * 0.5 - 1.0).collect(),
            vec![0.25, -0.25, 0.5],
        )
        .unwrap();
        let codes = vec![7u32, 0, 15];
        let time = 3;

        let via_compute = Compute::cpu()
            .xcodec2_fsq_f32(&codes, time, Some(&proj), &attrs)
            .expect("cpu xcodec2_fsq_f32");
        let direct = xcodec2_fsq_decode(&codes, time, Some(&proj), &attrs)
            .expect("direct xcodec2_fsq_decode");
        assert_eq!(
            via_compute, direct,
            "Compute::cpu().xcodec2_fsq_f32 must byte-match vokra_ops::xcodec2_fsq_decode",
        );
    }

    #[test]
    fn cpu_fsq_family_f32_propagates_input_errors() {
        // Verbatim-propagation contract (FR-EX-08 — out-of-range codes are
        // explicit `InvalidArgument`, never a clamp), mirror of the RVQ
        // propagate tests above.
        let wt_attrs = WavTokenizerVqAttrs {
            vocab_size: 2,
            d_model: 2,
        };
        let table = CodebookTable::new(2, 2, vec![0.0; 4]).unwrap();
        let err = Compute::cpu()
            .wavtokenizer_vq_f32(&[9u32], 1, &table, &wt_attrs)
            .expect_err("out-of-range WavTokenizer code must be an explicit error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));

        let fsq_attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 2,
        };
        let err = Compute::cpu()
            .xcodec2_fsq_f32(&[16u32], 1, None, &fsq_attrs)
            .expect_err("out-of-range FSQ code must be an explicit error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// On a Metal build the `wavtokenizer_vq_f32` / `xcodec2_fsq_f32` Metal
    /// arms are explicit `UnsupportedOp` — the M4-16 GPU kernels are deferred
    /// (single-stage GEMV bound: the future kernels reuse the M2-01 gemv /
    /// gather kernels), and a consumer that bypasses the coverage gate still
    /// hits the method-level defence (FR-EX-08, mirror of the RVQ tests).
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_fsq_family_arms_are_unsupported_no_silent_fallback() {
        let compute = match Compute::for_backend(BackendKind::Metal, &[]) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; fsq family Metal arm test skipped");
                return;
            }
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        };
        let wt_attrs = WavTokenizerVqAttrs {
            vocab_size: 1,
            d_model: 1,
        };
        let table = CodebookTable::new(1, 1, vec![0.0]).unwrap();
        assert!(matches!(
            compute.wavtokenizer_vq_f32(&[0u32], 1, &table, &wt_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));

        let fsq_attrs = Xcodec2FsqAttrs {
            levels: vec![2],
            d_model: 1,
        };
        assert!(matches!(
            compute.xcodec2_fsq_f32(&[0u32], 1, None, &fsq_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    /// On a CUDA build the `wavtokenizer_vq_f32` / `xcodec2_fsq_f32` CUDA
    /// arms are explicit `UnsupportedOp` (FR-EX-08); skips when no CUDA
    /// loader exists.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    #[test]
    fn cuda_fsq_family_arms_are_unsupported_no_silent_fallback() {
        let compute = match Compute::for_backend(BackendKind::Cuda, &[]) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no CUDA loader; fsq family CUDA arm test skipped");
                return;
            }
            Err(e) => panic!("unexpected CUDA for_backend error: {e}"),
        };
        let wt_attrs = WavTokenizerVqAttrs {
            vocab_size: 1,
            d_model: 1,
        };
        let table = CodebookTable::new(1, 1, vec![0.0]).unwrap();
        assert!(matches!(
            compute.wavtokenizer_vq_f32(&[0u32], 1, &table, &wt_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));

        let fsq_attrs = Xcodec2FsqAttrs {
            levels: vec![2],
            d_model: 1,
        };
        assert!(matches!(
            compute.xcodec2_fsq_f32(&[0u32], 1, None, &fsq_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    /// Off the Metal build, `for_backend(Metal, [WavTokenizerVq|Xcodec2Fsq])`
    /// is an explicit `BackendUnavailable` — never a silent CPU substitute
    /// (FR-EX-08; mirror of `metal_mimi_rvq_off_metal_is_backend_unavailable`).
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn metal_fsq_family_off_metal_is_backend_unavailable() {
        for op in [HotOp::WavTokenizerVq, HotOp::Xcodec2Fsq] {
            let err = match Compute::for_backend(BackendKind::Metal, &[op]) {
                Ok(_) => panic!(
                    "Metal must fail explicitly when not compiled in — never a silent CPU \
                     substitute",
                ),
                Err(e) => e,
            };
            assert!(
                matches!(err, VokraError::BackendUnavailable(_)),
                "expected BackendUnavailable for {op:?}, got {err:?}",
            );
        }
    }

    /// On a Metal build the `dac_rvq_f32` / `encodec_rvq_f32` Metal arms are
    /// explicit `UnsupportedOp` — the M4-04 GPU kernels are deferred, and a
    /// consumer that bypasses the coverage gate still hits the method-level
    /// defence (FR-EX-08, mirror of the mimi test above).
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_dac_and_encodec_rvq_arms_are_unsupported_no_silent_fallback() {
        let compute = match Compute::for_backend(BackendKind::Metal, &[]) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; dac/encodec rvq Metal arm test skipped");
                return;
            }
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        };
        let dac_attrs = DacRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            codebook_dim: 1,
            d_model: 1,
        };
        let dac_tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        let dac_projs = vec![DacOutProj::new(1, 1, vec![0.0], vec![0.0]).unwrap()];
        assert!(matches!(
            compute.dac_rvq_f32(&[0u32], 1, &dac_tables, &dac_projs, &dac_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));

        let enc_attrs = EncodecRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            d_model: 1,
        };
        let enc_tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        assert!(matches!(
            compute.encodec_rvq_f32(&[0u32], 1, &enc_tables, &enc_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    /// On a CUDA build the `dac_rvq_f32` / `encodec_rvq_f32` CUDA arms are
    /// explicit `UnsupportedOp` (FR-EX-08); skips when no CUDA loader exists.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    #[test]
    fn cuda_dac_and_encodec_rvq_arms_are_unsupported_no_silent_fallback() {
        let compute = match Compute::for_backend(BackendKind::Cuda, &[]) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no CUDA loader; dac/encodec rvq CUDA arm test skipped");
                return;
            }
            Err(e) => panic!("unexpected CUDA for_backend error: {e}"),
        };
        let dac_attrs = DacRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            codebook_dim: 1,
            d_model: 1,
        };
        let dac_tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        let dac_projs = vec![DacOutProj::new(1, 1, vec![0.0], vec![0.0]).unwrap()];
        assert!(matches!(
            compute.dac_rvq_f32(&[0u32], 1, &dac_tables, &dac_projs, &dac_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));

        let enc_attrs = EncodecRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            d_model: 1,
        };
        let enc_tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        assert!(matches!(
            compute.encodec_rvq_f32(&[0u32], 1, &enc_tables, &enc_attrs),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    /// On a Metal build the `mimi_rvq_f32` Metal arm is an explicit
    /// `UnsupportedOp` — no silent CPU fall back for the deferred M3-06 T14
    /// MSL kernel (FR-EX-08). A consumer that bypasses the `for_backend`
    /// coverage gate (e.g. by requesting an empty required set) still hits
    /// this belt-and-braces defence at the method level.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_mimi_rvq_arm_is_unsupported_no_silent_fallback() {
        // Build a Metal `Compute` with an empty required set (which the
        // coverage gate accepts). If the Metal device is absent this skips
        // — we cannot exercise the arm at all off a device.
        let compute = match Compute::for_backend(BackendKind::Metal, &[]) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; mimi_rvq_f32 Metal arm test skipped");
                return;
            }
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        };
        assert_eq!(compute.backend_name(), "metal");

        // Any inputs are fine — the arm returns early with UnsupportedOp
        // before touching the codes / tables (the M3-06 T14 kernel is
        // deferred to the M3-09 mimi_bridge upgrade past stub).
        let attrs = MimiRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            d_model: 1,
        };
        let tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        let err = compute
            .mimi_rvq_f32(&[0u32], 1, &tables, &attrs)
            .expect_err("Metal arm of mimi_rvq_f32 must be UnsupportedOp");
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp, got {err:?}",
        );
    }

    /// Off the Metal build (or off Apple), the coverage gate blocks
    /// `for_backend(Metal, [MimiRvq])` at the `BackendUnavailable` layer
    /// (Metal is not compiled in), so the `UnsupportedOp` from the coverage
    /// gate on Metal builds and the `BackendUnavailable` on non-Metal builds
    /// are both explicit — never a silent CPU substitute (FR-EX-08).
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn metal_mimi_rvq_off_metal_is_backend_unavailable() {
        let err = match Compute::for_backend(BackendKind::Metal, &[HotOp::MimiRvq]) {
            Ok(_) => panic!(
                "Metal must fail explicitly when not compiled in — never a silent CPU substitute",
            ),
            Err(e) => e,
        };
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable, got {err:?}",
        );
    }

    /// On a CUDA build the `mimi_rvq_f32` CUDA arm is an explicit
    /// `UnsupportedOp` — no silent CPU fall back for the deferred M3-06 T15
    /// NVRTC kernel (FR-EX-08). Exercised on the vast.ai RTX 4090
    /// (M2-03-T25 style); here it skips if no CUDA loader is present.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    #[test]
    fn cuda_mimi_rvq_arm_is_unsupported_no_silent_fallback() {
        let compute = match Compute::for_backend(BackendKind::Cuda, &[]) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no CUDA loader; mimi_rvq_f32 CUDA arm test skipped");
                return;
            }
            Err(e) => panic!("unexpected CUDA for_backend error: {e}"),
        };
        assert_eq!(compute.backend_name(), "cuda");

        let attrs = MimiRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            d_model: 1,
        };
        let tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        let err = compute
            .mimi_rvq_f32(&[0u32], 1, &tables, &attrs)
            .expect_err("CUDA arm of mimi_rvq_f32 must be UnsupportedOp");
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp, got {err:?}",
        );
    }

    /// Off the CUDA build, `for_backend(Cuda, [MimiRvq])` is
    /// `BackendUnavailable` (CUDA is not compiled in) — never a silent CPU
    /// substitute (FR-EX-08).
    #[cfg(not(all(feature = "cuda", any(unix, windows))))]
    #[test]
    fn cuda_mimi_rvq_off_cuda_is_backend_unavailable() {
        let err = match Compute::for_backend(BackendKind::Cuda, &[HotOp::MimiRvq]) {
            Ok(_) => panic!(
                "CUDA must fail explicitly when not compiled in — never a silent CPU substitute",
            ),
            Err(e) => e,
        };
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable, got {err:?}",
        );
    }

    #[test]
    fn prenorm_stack_cpu_is_unsupported_no_silent_fallback() {
        // The CPU never fuses the encoder: `prenorm_stack_is_fused()` is false (so
        // `whisper::encoder::encode` keeps the per-op `encoder_block` loop), and
        // `encode_prenorm_encoder` on CPU is an explicit UnsupportedOp — never a
        // silent CPU substitute of the GPU-only device-resident path (FR-EX-08).
        let cpu = Compute::cpu();
        assert!(!cpu.prenorm_stack_is_fused());
        let layer = PrenormLayer {
            attn_ln_gamma: &[1.0, 1.0],
            attn_ln_beta: &[0.0, 0.0],
            q_w: &[1.0, 0.0, 0.0, 1.0],
            q_bias: None,
            k_w: &[1.0, 0.0, 0.0, 1.0],
            k_bias: None,
            v_w: &[1.0, 0.0, 0.0, 1.0],
            v_bias: None,
            out_w: &[1.0, 0.0, 0.0, 1.0],
            out_bias: None,
            mlp_ln_gamma: &[1.0, 1.0],
            mlp_ln_beta: &[0.0, 0.0],
            fc1_w: &[1.0, 0.0, 0.0, 1.0],
            fc1_bias: None,
            fc2_w: &[1.0, 0.0, 0.0, 1.0],
            fc2_bias: None,
        };
        let mut out = [0.0f32; 2];
        assert!(matches!(
            cpu.encode_prenorm_encoder(
                1,
                2,
                2,
                1,
                1e-5,
                &[0.0, 0.0],
                &[layer],
                &[1.0, 1.0],
                &[0.0, 0.0],
                &mut out,
            ),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn cpu_for_backend_covers_every_op() {
        // The CPU backend covers the full hot-op set unconditionally —
        // including MimiRvq (M3-06 T04 kernel via `vokra_ops::mimi_rvq_decode`).
        let all = [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
        ];
        let c = Compute::for_backend(BackendKind::Cpu, &all).expect("cpu covers all");
        assert_eq!(c.backend_name(), "cpu");
    }

    #[test]
    fn make_backend_cpu_is_the_cpu_backend() {
        let b = make_backend(BackendKind::Cpu).expect("cpu backend");
        assert_eq!(b.name(), "cpu");
    }

    /// Off the Metal build (feature off or non-Apple), selecting Metal is an
    /// explicit unavailability error — never a silent CPU substitute.
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn metal_without_the_feature_is_explicit_unavailable() {
        assert!(matches!(
            Compute::for_backend(BackendKind::Metal, &[HotOp::Gemm]),
            Err(VokraError::BackendUnavailable(_))
        ));
        assert!(matches!(
            make_backend(BackendKind::Metal),
            Err(VokraError::BackendUnavailable(_))
        ));
    }

    /// On a Metal build, coverage is enforced. As of Phase 4 the Metal backend
    /// covers the **whole** Whisper hot-op set, so `for_backend` never returns
    /// `UnsupportedOp` for it — it either builds (device present) or reports an
    /// explicit device unavailability (no silent CPU fall back). `HotOp::MimiRvq`
    /// is deliberately NOT covered (M3-06 T14 kernel deferred to M3-09
    /// mimi_bridge upgrade), so a request that lists it fails with a coverage
    /// `UnsupportedOp` — this is verified below as the FR-EX-08 gate.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_coverage_is_consistent() {
        // Every Whisper hot op is covered (this pins `covered_by_metal` to the
        // wired Metal method arms — all now dispatch to a `MetalContext`
        // kernel).
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ] {
            assert!(
                op.covered_by_metal(),
                "{op:?} unexpectedly NOT Metal-covered"
            );
        }

        // MimiRvq is NOT covered on Metal (M3-06 T14 MSL kernel deferred to
        // M3-09 mimi_bridge follow-up). This is the honest state — if the
        // kernel has just landed, flip `HotOp::covered_by_metal` for MimiRvq
        // and update the negative assertion below.
        assert!(
            !HotOp::MimiRvq.covered_by_metal(),
            "HotOp::MimiRvq unexpectedly Metal-covered — the M3-06 T14 MSL kernel is deferred; if \
             it has just landed, flip `HotOp::covered_by_metal` for MimiRvq and update this test.",
        );
        // Same deferred posture for the M4-04 RVQ siblings and the M4-16 FSQ
        // family (lock-step with the Metal arms of `dac_rvq_f32` /
        // `encodec_rvq_f32` / `wavtokenizer_vq_f32` / `xcodec2_fsq_f32`).
        for op in [
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
        ] {
            assert!(
                !op.covered_by_metal(),
                "{op:?} unexpectedly Metal-covered — the M4-04/M4-16 GPU kernels are deferred; \
                 if one has just landed, flip `HotOp::covered_by_metal` and update this test.",
            );
            assert!(matches!(
                Compute::for_backend(BackendKind::Metal, &[op]),
                Err(VokraError::UnsupportedOp(_) | VokraError::BackendUnavailable(_)),
            ));
        }
        // A request that lists MimiRvq therefore fails the Metal coverage
        // gate with an explicit `UnsupportedOp` — never a silent CPU fall
        // back (FR-EX-08).
        assert!(matches!(
            Compute::for_backend(BackendKind::Metal, &[HotOp::MimiRvq]),
            Err(VokraError::UnsupportedOp(_)),
        ));

        // Whisper's full set is therefore a covered request: it either builds
        // (device present) or fails with an explicit device error — never a
        // coverage `UnsupportedOp`, never a silent CPU fall back (FR-EX-08).
        let whisper = [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ];
        match Compute::for_backend(BackendKind::Metal, &whisper) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; full-coverage build path is device-gated");
            }
            Err(e) => panic!("unexpected error for a fully-covered request: {e}"),
        }
    }

    /// On a CUDA build, coverage is enforced. As of Phase 4 (M2-03 T10-T14) the
    /// CUDA backend covers the **whole** Whisper hot-op set, so `for_backend`
    /// never returns `UnsupportedOp` for it — it either builds (device present)
    /// or reports an explicit device unavailability (no silent CPU fall back,
    /// FR-EX-08 / NFR-RL-06). `HotOp::MimiRvq` is deliberately NOT covered
    /// (M3-06 T15 NVRTC kernel deferred to M3-09 mimi_bridge upgrade), so a
    /// request that lists it fails with a coverage `UnsupportedOp` — this is
    /// verified below as the FR-EX-08 gate. The device branch is exercised on
    /// the vast.ai RTX 4090 (M2-03-T25); here it skips.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    #[test]
    fn cuda_coverage_is_consistent() {
        // Every Whisper hot op is covered (this pins `covered_by_cuda` to the
        // wired CUDA method arms — all now dispatch to a `CudaContext` kernel).
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ] {
            assert!(op.covered_by_cuda(), "{op:?} unexpectedly NOT CUDA-covered");
        }

        // MimiRvq is NOT covered on CUDA (M3-06 T15 NVRTC kernel deferred to
        // M3-09 mimi_bridge follow-up). This is the honest state — if the
        // kernel has just landed, flip `HotOp::covered_by_cuda` for MimiRvq
        // and update the negative assertion below.
        assert!(
            !HotOp::MimiRvq.covered_by_cuda(),
            "HotOp::MimiRvq unexpectedly CUDA-covered — the M3-06 T15 NVRTC kernel is deferred; \
             if it has just landed, flip `HotOp::covered_by_cuda` for MimiRvq and update this \
             test.",
        );
        // Same deferred posture for the M4-04 RVQ siblings and the M4-16 FSQ
        // family (lock-step with the CUDA arms of `dac_rvq_f32` /
        // `encodec_rvq_f32` / `wavtokenizer_vq_f32` / `xcodec2_fsq_f32`).
        for op in [
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
        ] {
            assert!(
                !op.covered_by_cuda(),
                "{op:?} unexpectedly CUDA-covered — the M4-04/M4-16 GPU kernels are deferred; \
                 if one has just landed, flip `HotOp::covered_by_cuda` and update this test.",
            );
            assert!(matches!(
                Compute::for_backend(BackendKind::Cuda, &[op]),
                Err(VokraError::UnsupportedOp(_) | VokraError::BackendUnavailable(_)),
            ));
        }
        // A request that lists MimiRvq therefore fails the CUDA coverage
        // gate with an explicit `UnsupportedOp` — never a silent CPU fall
        // back (FR-EX-08).
        assert!(matches!(
            Compute::for_backend(BackendKind::Cuda, &[HotOp::MimiRvq]),
            Err(VokraError::UnsupportedOp(_)),
        ));

        // Whisper's full set is therefore a covered request: it either builds
        // (device present) or fails with an explicit device error — never a
        // coverage `UnsupportedOp`, never a silent CPU fall back (FR-EX-08).
        let whisper = [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ];
        match Compute::for_backend(BackendKind::Cuda, &whisper) {
            Ok(c) => assert_eq!(c.backend_name(), "cuda"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!(
                    "no CUDA device; full-coverage build path is device-gated (run on vast.ai)"
                );
            }
            Err(e) => panic!("unexpected error for a fully-covered CUDA request: {e}"),
        }
    }

    /// M3-02 Vulkan seam contract in the foundation slice: **no hot op is
    /// covered**, so any non-empty required set surfaces `UnsupportedOp` (never
    /// silent CPU). This pins the lock-step between `covered_by_vulkan` and
    /// `for_backend(Vulkan, …)` — as T14〜T22 land, this test tightens.
    /// `HotOp::MimiRvq` is in the iteration too, but note MimiRvq is *not* on
    /// the M3-02 T14〜T22 track — it needs the M3-06 GPU kernels' Vulkan
    /// sibling (M4+), so the negative assertion for MimiRvq holds even after
    /// T22 lands.
    #[cfg(all(
        feature = "vulkan",
        any(target_os = "linux", target_os = "android", target_os = "windows")
    ))]
    #[test]
    fn vulkan_coverage_is_consistent() {
        // Foundation slice: `covered_by_vulkan` is `false` for every variant.
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
        ] {
            assert!(
                !op.covered_by_vulkan(),
                "{op:?} unexpectedly covered by the M3-02 foundation-slice Vulkan backend \
                 (kernels/precompiled/ still ships no .spv). If T14+ has just landed a kernel, \
                 update `HotOp::covered_by_vulkan` to `true` for the covered variants and shrink \
                 this test's negative-assertion set accordingly.",
            );
        }
        // Every non-empty required set therefore fails coverage with an
        // explicit `UnsupportedOp` — no silent CPU fall back (FR-EX-08).
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
        ] {
            assert!(matches!(
                Compute::for_backend(BackendKind::Vulkan, &[op]),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
        // Empty required set is also explicit `UnsupportedOp` (no callable
        // kernel exists to build a `Be::Vulkan` around today).
        assert!(matches!(
            Compute::for_backend(BackendKind::Vulkan, &[]),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    /// `make_backend(Vulkan)` returns a real `VulkanBackend` on a Vulkan-
    /// capable Linux/Android/Windows build, or an explicit
    /// `BackendUnavailable` off Vulkan — never a silent CPU substitute.
    #[cfg(all(
        feature = "vulkan",
        any(target_os = "linux", target_os = "android", target_os = "windows")
    ))]
    #[test]
    fn vulkan_make_backend_is_honest_on_any_host() {
        match make_backend(BackendKind::Vulkan) {
            Ok(b) => assert_eq!(b.name(), "vulkan"),
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("no Vulkan loader/device; make_backend(Vulkan) errored: {msg}");
            }
            Err(other) => panic!(
                "expected BackendUnavailable off Vulkan, got {other} (never a silent CPU \
                 substitute, FR-EX-08)"
            ),
        }
    }

    /// Default-feature build (no `--features vulkan`): `BackendKind::Vulkan`
    /// falls to the target-agnostic error path — the compile-out is honest,
    /// never a silent CPU substitute.
    #[cfg(not(all(
        feature = "vulkan",
        any(target_os = "linux", target_os = "android", target_os = "windows")
    )))]
    #[test]
    fn vulkan_not_compiled_in_is_explicit_backend_unavailable() {
        // `for_backend` falls through to the catch-all `_ =>` arm — the
        // error mentions the `vulkan` feature, so the caller knows exactly
        // what to enable. `Compute` does not derive `Debug`, so unwrap the
        // error manually instead of `expect_err`.
        let err = match Compute::for_backend(BackendKind::Vulkan, &[HotOp::Gemm]) {
            Ok(_) => panic!("Vulkan must fail explicitly when not compiled in"),
            Err(e) => e,
        };
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable, got {err:?}"
        );
        assert!(matches!(
            make_backend(BackendKind::Vulkan),
            Err(VokraError::BackendUnavailable(_))
        ));
    }

    /// M4-01-T16 off-target contract: on every non-wasm32 build (including
    /// native `--features webgpu`), `BackendKind::WebGpu` falls to the
    /// target-agnostic error path — an explicit `BackendUnavailable` naming
    /// the `webgpu` feature. Never a silent CPU substitute (FR-EX-08); the
    /// WASM CPU path is only ever the caller's explicit `BackendKind::Cpu`
    /// choice.
    #[cfg(not(all(feature = "webgpu", target_arch = "wasm32")))]
    #[test]
    fn webgpu_off_target_is_explicit_backend_unavailable() {
        let err = match Compute::for_backend(BackendKind::WebGpu, &[HotOp::Gemm]) {
            Ok(_) => panic!("WebGpu must fail explicitly off wasm32 / without the feature"),
            Err(e) => e,
        };
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("webgpu"),
            "the error must name the `webgpu` feature so the caller knows what to enable: {msg}"
        );
        assert!(matches!(
            make_backend(BackendKind::WebGpu),
            Err(VokraError::BackendUnavailable(_))
        ));
    }

    /// M4-01-T16 on-target coverage lock-step (compiled for wasm32 + `webgpu`
    /// only; executed by the browser/Node harness runs, not native CI): the
    /// six Whisper hot ops are covered, the RVQ codec ops are not — listing
    /// one fails the coverage gate with an explicit `UnsupportedOp` (never a
    /// silent CPU fall back, FR-EX-08). This pins `covered_by_webgpu` to the
    /// `Be::WebGpu` method arms above.
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    #[test]
    fn webgpu_coverage_is_consistent() {
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ] {
            assert!(
                op.covered_by_webgpu(),
                "{op:?} unexpectedly NOT WebGPU-covered"
            );
        }
        for op in [HotOp::MimiRvq, HotOp::DacRvq, HotOp::EncodecRvq] {
            assert!(
                !op.covered_by_webgpu(),
                "{op:?} unexpectedly WebGPU-covered — the RVQ GPU kernels are deferred; if one \
                 has just landed, flip `HotOp::covered_by_webgpu` and update this test.",
            );
            assert!(matches!(
                Compute::for_backend(BackendKind::WebGpu, &[op]),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
    }
}
