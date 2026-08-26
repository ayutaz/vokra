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

use vokra_backend_cpu::IsaPath;
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
    Qwen3TtsCodecConfig, SnacConfig, SnacDecoder, SnacWeights, WavTokenizerVqAttrs,
    Xcodec2FsqAttrs, dac_rvq_decode, encodec_rvq_decode, mimi_rvq_decode, qwen3_tts_codec_decode,
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
    /// Gamma-only RMS normalisation (`rms_norm_f32`): no mean subtraction and
    /// no bias. NeuCodec's decoder Transformer uses this before attention and
    /// its MLP. CPU is the scalar reference; Metal dispatches the existing
    /// `vokra_rms_norm_f32` kernel. Other backends remain explicitly
    /// uncovered until they gain matching kernels.
    RmsNorm,
    /// Scalar-gain ScaleNorm (`scale_norm_f32`) used by MossFormer2 FLASH
    /// projections: divide each row by
    /// `max(||row||₂ * cols^-0.5, eps)`, then multiply by the learned gain.
    /// This is not an RMSNorm alias because epsilon clamps the completed norm.
    /// CPU and Metal have dedicated kernels; other backends stay explicitly
    /// uncovered rather than running the reduction on the host.
    ScaleNorm,
    /// One-group affine GroupNorm over channel-major audio features. SepFormer
    /// uses this for the full mask tensor and needs a stable large reduction.
    GroupNorm,
    /// Exact (erf) GELU (`gelu_f32`) — Whisper MLP / conv stem.
    Gelu,
    /// GPT-2 / Transformers `gelu_new` tanh approximation. Distinct from
    /// [`Self::Gelu`]: substituting the exact/erf form changes released MOSS-
    /// TTS Nano numerics. CPU uses the portable scalar kernel and Metal uses a
    /// dedicated MSL kernel; uncovered backends fail the whole-model coverage
    /// gate rather than falling back.
    GeluNew,
    /// Element-wise ReLU (`max(x, 0)`). T5-base uses this between its two
    /// feed-forward projections. CPU dispatches the existing SIMD kernel and
    /// Metal has a dedicated MSL kernel; other backends remain explicit
    /// unsupported operations.
    Relu,
    /// Element-wise hyperbolic tangent. SpeechT5's four activated postnet
    /// convolution blocks require this exact nonlinearity. CPU dispatches the
    /// existing portable kernel and Metal has a dedicated MSL kernel; other
    /// backends remain explicitly uncovered rather than executing on the
    /// host.
    Tanh,
    /// Element-wise SiLU / Swish (`x * sigmoid(x)`). WavTokenizer's
    /// positional ResNet applies this after every GroupNorm. The CPU arm is
    /// the scalar mathematical reference; Metal dispatches the existing
    /// `vokra_silu_f32` kernel. Other backends remain explicitly uncovered.
    Silu,
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
    /// **Metal-covered since M3-06 T14 (2026-08-13)** — do not report the
    /// Metal kernel as missing. [`Compute::mimi_rvq_f32`] dispatches the
    /// `vokra_mimi_rvq_gather_fold_f32` MSL kernel (shape-generic FP32 gather
    /// + fold, bit-identical to `vokra_ops::mimi_rvq::rvq_fold_core`, with the
    ///   per-index bound check host-side upstream of the dispatch — FR-EX-08),
    ///   so `covered_by_metal` returns `true` for this variant.
    ///
    /// Still uncovered on **CUDA** (the M3-06 T15 NVRTC sibling is on the
    /// vast.ai owner track), on **Vulkan** (needs the M3-06 kernels' Vulkan
    /// sibling — it is not on the M3-02 T14〜T22 track at all) and on
    /// **WebGPU**, so a model listing `MimiRvq` still fails
    /// `for_backend(Cuda|Vulkan|WebGpu, …)` with a coverage
    /// `UnsupportedOp` — never a silent CPU fall back (FR-EX-08). The
    /// `metal_coverage_is_consistent` / `vulkan_coverage_is_consistent` tests
    /// pin this table to the `Compute` method arms.
    MimiRvq,
    /// DAC (Descript) factorized residual VQ codec decode
    /// (`dac_rvq_decode`) — M4-04, FR-OP-30. Same heterogeneous-signature /
    /// heap-returning shape as [`HotOp::MimiRvq`] plus the per-quantizer
    /// projection operands ([`DacOutProj`]). Kept a **separate variant** from
    /// `MimiRvq` so the coverage table stays honest per op (ADR M4-04 §D-e).
    ///
    /// **Metal-covered since M4-04 WF2 (2026-08-13)** — do not report the
    /// Metal kernel as missing. [`Compute::dac_rvq_f32`] dispatches the
    /// `vokra_dac_rvq_gather_project_fold_f32` MSL kernel (factorized gather +
    /// per-quantizer projection + FP32 fold, equal to
    /// `vokra_ops::dac_rvq::dac_rvq_decode` within FP32 fast-math tolerance,
    /// with the per-index bound check host-side upstream — FR-EX-08), so
    /// `covered_by_metal` returns `true` for this variant.
    ///
    /// Still uncovered on **CUDA / Vulkan / WebGPU** (the CUDA sibling is on
    /// the vast.ai owner track; the naive gather + GEMV + fold layout note in
    /// `vokra_ops::mimi_rvq` L104-106 applies to all three RVQ ops), so the
    /// coverage gate rejects those listings (FR-EX-08).
    DacRvq,
    /// EnCodec residual VQ codec decode (`encodec_rvq_decode`) — M4-04,
    /// FR-OP-30 op / FR-OP-32 permanent weight exclusion. The op rides the
    /// shape-generic gather + FP32 fold; **pretrained EnCodec weights never
    /// ship** (the official zoo excludes them permanently; the M2-13 gate
    /// refuses them without a research flag). Separate variant for honest
    /// per-op coverage (ADR M4-04 §D-e).
    ///
    /// **Metal-covered since the AudioCraft waveform-decode wave
    /// (2026-08-26)** — EnCodec's unfactorized RVQ is mathematically the same
    /// shape-generic gather + FP32 fold as [`HotOp::MimiRvq`]. The Metal arm
    /// therefore dispatches the already-parity-pinned
    /// `vokra_mimi_rvq_gather_fold_f32` kernel after EnCodec-specific host
    /// shape/index validation. This does not change FR-OP-32: standalone
    /// pretrained EnCodec artifacts remain excluded from the official model
    /// zoo; the consumer is an authenticated MusicGen composite artifact.
    /// CUDA / Vulkan / WebGPU remain explicitly unsupported.
    EncodecRvq,
    /// WavTokenizer single-codebook VQ decode (`wavtokenizer_vq_decode`) —
    /// M4-16, FR-OP-31 **FSQ family** (single-stage, *separate subgraph from
    /// the RVQ family* — no cross-codebook residual sum, no paged variant;
    /// module docs in `vokra_ops::fsq_codec`). Heterogeneous-signature /
    /// heap-returning shape like the RVQ seam methods, but the table operand
    /// is a *singular* [`CodebookTable`].
    ///
    /// **M4-16 WF2 (2026-08-13):** the Metal arm is now wired via
    /// `vokra_wavtokenizer_vq_gather_f32` (pure single-codebook gather —
    /// bit-identical semantics to `vokra_ops::wavtokenizer_vq_decode`, host-
    /// side per-index bound check upstream of the dispatch — FR-EX-08). The
    /// CUDA arm remains an explicit [`VokraError::UnsupportedOp`] (deferred
    /// to the vast.ai owner track). `covered_by_metal` returns `true`;
    /// `covered_by_cuda` / `covered_by_vulkan` / `covered_by_webgpu` return
    /// `false`, so any model listing this variant against the un-wired
    /// backends fails the coverage gate (never a silent CPU fall back).
    WavTokenizerVq,
    /// X-Codec 2 FSQ dequant (`xcodec2_fsq_decode`) — M4-16, FR-OP-31 FSQ
    /// family sibling of [`HotOp::WavTokenizerVq`]. Implicit per-dimension
    /// grid (no codebook tensor) + one out-projection GEMV per timestep.
    /// Separate variant for honest per-op coverage.
    ///
    /// **M4-16 WF2 (2026-08-13):** the Metal arm is now wired via
    /// `vokra_xcodec2_fsq_decode_f32` (grid decompose + optional Linear
    /// projection, semantics equal to `vokra_ops::xcodec2_fsq_decode` within
    /// FP32 fast-math tolerance, host-side per-index bound check upstream —
    /// FR-EX-08). The CUDA arm remains an explicit
    /// [`VokraError::UnsupportedOp`] (deferred to the vast.ai owner track).
    /// `covered_by_metal` returns `true`; `covered_by_cuda` /
    /// `covered_by_vulkan` / `covered_by_webgpu` return `false`.
    Xcodec2Fsq,
    /// NVIDIA NanoCodec grouped finite scalar quantizer dequantization — the
    /// per-group mixed-radix code-to-latent transform proposed in #45. It is
    /// deliberately distinct from
    /// [`HotOp::Xcodec2Fsq`]: NanoCodec partitions the latent channels into
    /// independently configured groups and concatenates their dequantized
    /// values, rather than applying X-Codec 2's single grid and optional
    /// projection.
    ///
    /// **CPU-only initial slice (#51, 2026-08-21).** No Metal, CUDA, Vulkan,
    /// WebGPU, CoreML, or QNN kernel is claimed. A NanoCodec model must list
    /// this variant in its required-op registry, so pinning any non-CPU
    /// backend fails at [`Compute::for_backend`] with an explicit
    /// [`VokraError::UnsupportedOp`] instead of silently falling back.
    GroupFsq,
    /// Snake activation (`vokra_ops::snake_activation_f32`) — the per-channel
    /// closed-form periodic activation `y = x + (1/(α+ε))·sin(α·x)²` shared
    /// by the BigVGAN / HiFTNet / Kokoro-82M vocoder lineage. Consumed by
    /// the M2-07 Kokoro decoder (private `kokoro::nn::snake_activation`
    /// helper — unchanged) and every future vocoder that wants a GPU
    /// dispatch for the plain-Snake variant (the two-vector SnakeBeta stays
    /// on its own [`vokra_ops::bigvgan_generator::SnakeBeta`] path).
    ///
    /// **Vocoder Metal wave WF2 (2026-08-13):** the Metal arm is wired via
    /// `vokra_snake_activation_f32` (element-wise MSL kernel, semantics
    /// equal to `vokra_ops::snake_activation_f32` within the FP32
    /// transcendental gap, host-side shape validation upstream — FR-EX-08).
    /// The CUDA arm remains an explicit [`VokraError::UnsupportedOp`]
    /// (deferred to the vast.ai owner track). `covered_by_metal` returns
    /// `true`; `covered_by_cuda` / `covered_by_vulkan` / `covered_by_webgpu`
    /// / `covered_by_coreml` / `covered_by_qnn` return `false` — any model
    /// that lists it against those backends fails the coverage gate
    /// (never a silent CPU fall back).
    SnakeActivation,
    /// SNAC 3-stage hierarchical residual VQ codec decode
    /// (`vokra_ops::snac_decode::SnacDecoder::decode`) — the Vocoder wave
    /// WF5 op consumed by Orpheus / Maya1 (upstream `hubertsiuzdak/snac`,
    /// MIT / Apache-2.0). **Distinct from every RVQ / FSQ variant**: SNAC's
    /// multi-scale structure runs each stage at `base / vq_strides[s]`
    /// (SNAC 24 kHz canonical `[4, 2, 1]` → ~12 / 23 / 47 Hz per stage),
    /// which no other codec op does — the per-stage
    /// `t_stage = t_out / strides[s]` lookup is baked into the kernel. The
    /// per-quantizer projection reuses the DAC factorized shape
    /// ([`DacOutProj`]).
    ///
    /// **Vocoder Metal wave WF5 (2026-08-13):** the Metal arm is wired via
    /// `vokra_snac_decode_f32` (3-stage gather + factorized projection +
    /// temporal upsample + FP32 residual sum, semantics equal to
    /// `SnacDecoder::decode` within FP32 fast-math tolerance, host-side
    /// per-index bound check upstream — FR-EX-08). The CUDA arm remains an
    /// explicit [`VokraError::UnsupportedOp`] (deferred to the vast.ai
    /// owner track). `covered_by_metal` returns `true`; `covered_by_cuda` /
    /// `covered_by_vulkan` / `covered_by_webgpu` / `covered_by_coreml` /
    /// `covered_by_qnn` return `false` — any model that lists it against
    /// those backends fails the coverage gate (never a silent CPU fall
    /// back).
    SnacDecode,
    /// Denoise spectral-gate primitive
    /// (`vokra_ops::denoise::denoise_apply_mask_f32`) — element-wise complex
    /// × real gain multiply, the "spectral gate + phase preservation" step
    /// every mask-based denoiser ends in. Extracted from the
    /// [`vokra_ops::denoise::DenoiseModel::enhance_inner`] output-stage
    /// loop (denoise.rs L1852-1870) so a per-freq-per-time mask denoiser
    /// (GTCRN / RNNoise) can call it directly, and the mask apply can
    /// move to a GPU dispatch while the rest of the front-end runs on
    /// the host.
    ///
    /// # Not the whole DenoiseModel
    ///
    /// This variant is the mask-apply primitive alone, not the whole
    /// DFN3 network (which lives in `DenoiseModel::enhance`).
    /// [`DenoiseModel::enhance`] still uses its fused inline loop for
    /// the CPU-only path it has always taken; this primitive is what
    /// downstream consumers wire when they want a GPU dispatch for that
    /// inner loop.
    ///
    /// **Vocoder Metal wave WF5 (2026-08-13):** the Metal arm is wired
    /// via `vokra_denoise_apply_mask_f32` (element-wise MSL kernel,
    /// bit-for-bit identical to `vokra_ops::denoise_apply_mask_f32` —
    /// pure real × complex multiply has no reduction, no transcendental,
    /// no FMA opportunity). Host-side shape validation upstream of the
    /// dispatch (FR-EX-08). The CUDA arm remains an explicit
    /// [`VokraError::UnsupportedOp`] (deferred to the vast.ai owner
    /// track). `covered_by_metal` returns `true`; `covered_by_cuda` /
    /// `covered_by_vulkan` / `covered_by_webgpu` / `covered_by_coreml` /
    /// `covered_by_qnn` return `false` — any model that lists it against
    /// those backends fails the coverage gate (never a silent CPU fall
    /// back).
    DenoiseApplyMask,
    /// Qwen3-TTS-Codec RVQ decode
    /// (`vokra_ops::qwen3_tts_codec::qwen3_tts_codec_decode`) — the per-
    /// quantizer summed feature decode step consumed by every released
    /// Qwen3-TTS-12Hz voice (`Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,
    /// CustomVoice,VoiceDesign}`, Apache-2.0). **Distinct from every other
    /// RVQ variant**: Qwen3-TTS-Codec is a **hybrid semantic + acoustic RVQ**
    /// where the first `num_semantic_quantizers` quantizers use a larger
    /// `semantic_codebook_size` vocab (canonical 4096) than the remaining
    /// acoustic quantizers use `codebook_size` (canonical 2048). The
    /// [`Compute::qwen3_tts_codec_f32`] method mirrors the CPU op's
    /// heap-returning shape (per-quantizer `Vec<u32>` streams +
    /// `Vec<CodebookTable>` → `Vec<f32>`), which is why the op is a runtime
    /// function in `vokra-ops` rather than an [`vokra_core::OpKind`] variant
    /// (module docs in `vokra_ops::qwen3_tts_codec`).
    ///
    /// **Vocoder Metal wave WF5 (2026-08-13):** the Metal arm is wired via
    /// `vokra_qwen3_tts_codec_decode_f32` (semantic + acoustic gather + FP32
    /// fold, semantics equal to `qwen3_tts_codec_decode` within FP32
    /// fast-math tolerance, host-side per-index bound check upstream —
    /// FR-EX-08). The CUDA arm remains an explicit
    /// [`VokraError::UnsupportedOp`] (deferred to the vast.ai owner track).
    /// `covered_by_metal` returns `true`; `covered_by_cuda` /
    /// `covered_by_vulkan` / `covered_by_webgpu` / `covered_by_coreml` /
    /// `covered_by_qnn` return `false` — any model that lists it against
    /// those backends fails the coverage gate (never a silent CPU fall
    /// back).
    Qwen3TtsCodec,
    /// SnakeBeta activation (`vokra_ops::snake_beta_f32`) — the per-channel
    /// two-vector closed-form periodic activation
    /// `y = x + (1/(β+ε))·sin(α·x)²` consumed by the BigVGAN family
    /// (upstream `activations.py:62-114`, MIT / NVIDIA). Distinct from
    /// [`HotOp::SnakeActivation`] (single-vector `α`-only variant), the
    /// stateful [`vokra_ops::bigvgan_generator::SnakeBeta`] type is the CPU
    /// forward that BigVGAN's terminal `activation_post` and every AMP
    /// block that selects [`vokra_ops::bigvgan_generator::SnakeKind::SnakeBeta`]
    /// dispatch through; the [`vokra_ops::snake_beta_f32`] free function is
    /// the stateless out-of-place adapter this seam routes through so a GPU
    /// dispatch (Metal / CUDA / etc.) can go through the same shape as the
    /// sibling snake_activation seam.
    ///
    /// **Vocoder Metal wave (2026-08-14):** the Metal arm is wired via
    /// `vokra_snake_beta_f32` (element-wise MSL kernel, semantics equal to
    /// [`vokra_ops::snake_beta_f32`] within the FP32 transcendental gap,
    /// host-side shape validation upstream — FR-EX-08). The CUDA arm
    /// remains an explicit [`VokraError::UnsupportedOp`] (deferred to the
    /// vast.ai owner track). `covered_by_metal` returns `true`;
    /// `covered_by_cuda` / `covered_by_vulkan` / `covered_by_webgpu` /
    /// `covered_by_coreml` / `covered_by_qnn` return `false`.
    SnakeBeta,
    /// SineGen deterministic forward (`vokra_ops::sinegen_deterministic_f32`)
    /// — the F0-driven multi-harmonic sinusoid source of every HiFTNet-family
    /// vocoder (upstream CosyVoice `cosyvoice/hifigan/generator.py:200-214`,
    /// `SineGen.forward` under `NsfEntropy::Deterministic`). Consumed by
    /// CosyVoice2, CosyVoice3, the Chatterbox family, and every other
    /// vocoder that wires the M4-05 `SourceModuleHnNSF` chain (which
    /// currently only exposes the deterministic path from the GPU — the
    /// seeded variant carries per-harmonic phase + Gaussian noise host-side
    /// draws that no consumer needs on the GPU today; a follow-up if one
    /// materialises).
    ///
    /// **Vocoder Metal wave (2026-08-14):** the Metal arm is wired via
    /// `vokra_sinegen_deterministic_f32` (one thread per harmonic walking
    /// the full time axis sequentially — same per-harmonic reduction order
    /// as the CPU forward, semantics equal to
    /// [`vokra_ops::sinegen_deterministic_f32`] within the FP32
    /// transcendental gap, host-side shape validation upstream —
    /// FR-EX-08). The CUDA arm remains an explicit
    /// [`VokraError::UnsupportedOp`] (deferred to the vast.ai owner track).
    /// `covered_by_metal` returns `true`; `covered_by_cuda` /
    /// `covered_by_vulkan` / `covered_by_webgpu` / `covered_by_coreml` /
    /// `covered_by_qnn` return `false`.
    SinegenDeterministic,
    /// Polyphase anti-aliased upsample (`vokra_ops::anti_aliased_upsample_f32`)
    /// — the multiply-add core of BigVGAN's `UpSample1d` (upstream
    /// `alias_free_activation.torch.act`, MIT) and every HiFTNet-family
    /// alias-free activation chain. Consumes a caller-supplied Kaiser-window
    /// filter kernel (the design step lives on the host — see the vokra-ops
    /// module docs), so the runtime op signature is narrow (three tensor
    /// inputs + one scalar `ratio`), a good fit for a GPU dispatch.
    ///
    /// **Vocoder Metal wave (2026-08-14):** the Metal arm is wired via
    /// `vokra_anti_aliased_upsample_f32` (2-D dispatch, one thread per
    /// `(t_out, c)`, semantics equal to
    /// [`vokra_ops::anti_aliased_upsample_f32`] within the FMA-vs-non-FMA
    /// gap `atol ≤ 1e-4`, host-side shape validation upstream — FR-EX-08).
    /// The CUDA arm remains an explicit [`VokraError::UnsupportedOp`]
    /// (deferred to the vast.ai owner track). `covered_by_metal` returns
    /// `true`; `covered_by_cuda` / `covered_by_vulkan` /
    /// `covered_by_webgpu` / `covered_by_coreml` / `covered_by_qnn` return
    /// `false`.
    AntiAliasedUpsample,
    /// Grouped 1-D convolution with PyTorch weight layout
    /// `[out_ch, in_ch / groups, kernel]`. Vocos uses the depthwise case
    /// (`groups == in_ch == out_ch`) in every ConvNeXt block.
    ///
    /// The CPU arm composes the existing dense convolution kernel per group;
    /// the Metal arm uses group-local indexing in the direct Conv1d shader.
    /// CUDA/WebGPU/Vulkan/delegates remain explicitly uncovered.
    GroupedConv1d,
    /// Stateful causal HiFi-GAN generator forward (#46), including the
    /// generator's causal transposed-convolution stages and streaming state.
    /// This is one model-level hot op rather than a claim that the existing
    /// non-causal [`HotOp::Conv1d`] kernel covers transposed convolution.
    ///
    /// **CPU-only initial slice (#51, 2026-08-21).** No Metal, CUDA, Vulkan,
    /// WebGPU, CoreML, or QNN implementation exists for the stateful causal
    /// transposed-convolution path. Models using the generator must list this
    /// variant in their required-op registry; selecting a non-CPU backend is
    /// therefore rejected loudly by [`Compute::for_backend`] before inference.
    CausalHifiGan,
}

impl HotOp {
    /// Whether the Metal backend's imperative [`Compute`] seam covers this op.
    ///
    /// Kept in sync with the Metal arms of the [`Compute`] methods below; the
    /// `metal_coverage_is_consistent` test pins the two together. As of Phase 4
    /// (M2-01 T09-T13) the whole Whisper hot-op set (GEMM / GEMV / softmax /
    /// layer_norm / GELU / conv1d) has a `MetalContext` kernel, so the whole
    /// Whisper forward runs on the GPU through this seam. **M3-06 T14 (2026-
    /// 08-13)**: [`HotOp::MimiRvq`] is now covered on Metal too — the
    /// `vokra_mimi_rvq_gather_fold_f32` MSL kernel implements the shape-generic
    /// FP32 gather + fold behind [`Compute::mimi_rvq_f32`] (bit-identical
    /// semantics to `vokra_ops::mimi_rvq::rvq_fold_core`, host-side per-index
    /// bound check upstream of the dispatch — FR-EX-08). **M4-04 (WF2, 2026-
    /// 08-13)**: [`HotOp::DacRvq`] is now covered on Metal too — the
    /// `vokra_dac_rvq_gather_project_fold_f32` MSL kernel implements the
    /// factorized gather + per-quantizer projection + FP32 fold behind
    /// [`Compute::dac_rvq_f32`] (semantics equal to
    /// `vokra_ops::dac_rvq::dac_rvq_decode` within FP32 fast-math tolerance,
    /// host-side per-index bound check upstream — FR-EX-08). The AudioCraft
    /// waveform-decode wave (2026-08-26) wires [`HotOp::EncodecRvq`] through
    /// that same shape-generic Mimi kernel with EnCodec-specific validation.
    /// CUDA sibling (M3-06 T15 NVRTC kernel) is on the vast.ai owner track and
    /// remains uncovered here. (The *graph* backend
    /// `MetalBackend::supports` / `eval_op` is a separate path and still
    /// covers only `MatMul` — the two coverage surfaces are intentionally
    /// independent.)
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    fn covered_by_metal(self) -> bool {
        // Phase 4 wired the six Whisper hot ops; M3-06 T14 (2026-08-13) added
        // MimiRvq via `vokra_mimi_rvq_gather_fold_f32`; M4-04 WF2 (2026-08-13)
        // added DacRvq via `vokra_dac_rvq_gather_project_fold_f32`; M4-16 WF2
        // (2026-08-13) added the FSQ family (`WavTokenizerVq` via
        // `vokra_wavtokenizer_vq_gather_f32`, `Xcodec2Fsq` via
        // `vokra_xcodec2_fsq_decode_f32`). Vocoder Metal wave WF2 (2026-08-13)
        // added `SnakeActivation` via `vokra_snake_activation_f32`; Vocoder
        // Metal wave WF5 (2026-08-13) added `SnacDecode` via
        // `vokra_snac_decode_f32` and `DenoiseApplyMask` via
        // `vokra_denoise_apply_mask_f32`. AudioCraft waveform decoding
        // (2026-08-26) added EncodecRvq through
        // the same shape-generic gather + fold kernel as MimiRvq. The
        // EnCodec-specific seam still performs its own host validation before
        // dispatch; no CPU fall back is involved (FR-EX-08).
        matches!(
            self,
            HotOp::Gemm
                | HotOp::Gemv
                | HotOp::Softmax
                | HotOp::LayerNorm
                | HotOp::RmsNorm
                | HotOp::ScaleNorm
                | HotOp::GroupNorm
                | HotOp::Gelu
                | HotOp::GeluNew
                | HotOp::Relu
                | HotOp::Tanh
                | HotOp::Silu
                | HotOp::Conv1d
                | HotOp::GroupedConv1d
                | HotOp::MimiRvq
                | HotOp::DacRvq
                | HotOp::EncodecRvq
                | HotOp::WavTokenizerVq
                | HotOp::Xcodec2Fsq
                | HotOp::SnakeActivation
                | HotOp::SnacDecode
                | HotOp::DenoiseApplyMask
                | HotOp::Qwen3TtsCodec
                // Vocoder Metal wave common vocoder primitives (2026-08-14):
                // SnakeBeta / SineGen deterministic / anti-aliased upsample.
                // Every one is a stateless out-of-place free function with a
                // matching MSL kernel (`vokra_snake_beta_f32` /
                // `vokra_sinegen_deterministic_f32` /
                // `vokra_anti_aliased_upsample_f32`), so listing any of them
                // in a Metal `required` set builds against the wired kernel.
                | HotOp::SnakeBeta
                | HotOp::SinegenDeterministic
                | HotOp::AntiAliasedUpsample
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
    /// **`false` for every variant — but NOT because the shaders are
    /// missing.** Do not read this as "compile the SPIR-V first": that work is
    /// done. `crates/vokra-backend-vulkan/kernels/precompiled/` ships all 12
    /// `.spv` blobs (M4-13-T16, 2026-07-19, glslangValidator-pinned with
    /// `SHA256SUMS` + `PROVENANCE`), and `VulkanBackend` already exposes typed
    /// dispatch entry points over them — `gemm_f32` / `gemv_f32` /
    /// `softmax_f32` / `softmax_causal_f32` / `layer_norm_f32` / `gelu_f32` /
    /// `conv1d_f32` (M4-13-T03〜T08).
    ///
    /// What is missing is on **this** side of the seam: the private `Be` enum
    /// has no `Vulkan` variant, so the `Compute` methods below have no arm to
    /// delegate into. Flipping an op to `true` therefore means adding that arm
    /// and routing it at the already-landed `VulkanBackend` method — in
    /// lock-step, pinned by the `vulkan_coverage_is_consistent` test. Claiming
    /// a shader gap here would send the next reader off to recompile kernels
    /// that are already committed.
    ///
    /// [`HotOp::MimiRvq`] is the one variant whose `false` survives that
    /// wiring: it is not on the M3-02 T14〜T22 track at all and needs the
    /// M3-06 GPU kernels' Vulkan sibling (M4+).
    ///
    /// The consequence today is that `Compute::for_backend(BackendKind::Vulkan,
    /// &required)` returns an explicit [`VokraError::UnsupportedOp`] for every
    /// non-empty `required` — never a silent CPU fall back (FR-EX-08).
    #[cfg(all(
        feature = "vulkan",
        any(target_os = "linux", target_os = "android", target_os = "windows")
    ))]
    fn covered_by_vulkan(self) -> bool {
        // The `Compute` seam has no Vulkan arm — NOT "the shaders are
        // missing". The 12 `.spv` blobs and the typed `VulkanBackend::*_f32`
        // dispatch entry points both landed (see this method's docs); what is
        // absent is a `Be::Vulkan` variant here to delegate into. Flipping an
        // op to `true` means adding that arm, not compiling a kernel. Note
        // that MimiRvq is off the M3-02 T14〜T22 track (it needs the M3-06 GPU
        // kernels' Vulkan sibling, which is an M4+ item), so this method will
        // still return `false` for `HotOp::MimiRvq` after that wiring lands.
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

    /// Whether the CoreML delegate backend's [`Compute`] seam covers this op
    /// (M5-01-T10).
    ///
    /// **Scaffold slice:** the CoreML backend has NO wired execution path — it
    /// turns on the M5-01-T02 model-supply ADR (owner-ratified), so this is
    /// deliberately `false` for every op, the honest state (mirrors the Vulkan
    /// foundation slice). Consequently `for_backend(CoreMl, &required)` returns
    /// an explicit [`VokraError::UnsupportedOp`] for every non-empty `required`
    /// — never a silent CPU fall back (FR-EX-08).
    #[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
    fn covered_by_coreml(self) -> bool {
        let _ = self;
        false
    }

    /// Whether the QNN delegate backend's [`Compute`] seam covers this op
    /// (M5-02-T07).
    ///
    /// **Scaffold slice:** the QNN backend has NO wired execution path — QNN
    /// graph construction lands in the SDK-gated re-issue wave (owner T11 gates
    /// it), so this is deliberately `false` for every op, the honest state
    /// (mirrors the Vulkan foundation slice and the CoreML scaffold).
    /// Consequently `for_backend(Qnn, &required)` returns an explicit
    /// [`VokraError::UnsupportedOp`] for every op it reaches (after the probe) —
    /// never a silent CPU fall back (FR-EX-08).
    #[cfg(all(
        feature = "qnn",
        any(target_os = "android", target_os = "linux", target_os = "windows")
    ))]
    fn covered_by_qnn(self) -> bool {
        let _ = self;
        false
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
    cpu_isa: Option<IsaPath>,
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
    /// Metal GPU context. Covers every [`HotOp`] (Phase 4). `Box`ed for the
    /// same reason as the `Cuda` arm below — with the M3-06 T14 / M4-04 /
    /// M4-16 codec pipelines wired, `MetalContext` now embeds 25+ compiled
    /// pipelines by value (each an `Id`, +1-owned MTLComputePipelineState),
    /// so the inline size (216 B as of M4-16 WF2) trips
    /// `clippy::large_enum_variant`. The heap alloc is negligible — a
    /// `Compute` is built once per model entry, after a far costlier
    /// per-pipeline MSL compile inside `MetalContext::new`.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    Metal(Box<vokra_backend_metal::MetalContext>),
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
        Compute {
            be: Be::Cpu,
            cpu_isa: None,
        }
    }

    /// Builds the requested backend while forcing the portable scalar CPU
    /// kernels only when `kind` is [`BackendKind::Cpu`]. GPU selections remain
    /// unchanged and never fall back to the CPU.
    pub(crate) fn for_backend_with_scalar_cpu(
        kind: BackendKind,
        required: &[HotOp],
    ) -> Result<Self> {
        let mut compute = Self::for_backend(kind, required)?;
        if kind == BackendKind::Cpu {
            compute.cpu_isa = Some(IsaPath::Scalar);
        }
        Ok(compute)
    }

    /// Builds a dispatcher for `kind`, requiring it to cover every op in
    /// `required` (one model = one backend, FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] if `kind` is a real backend that does not
    ///   cover some op in `required` (e.g. Metal for a model that needs GroupFsq)
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
            all(feature = "webgpu", target_arch = "wasm32"),
            all(
                feature = "qnn",
                any(target_os = "android", target_os = "linux", target_os = "windows")
            )
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
                    be: Be::Metal(Box::new(vokra_backend_metal::MetalContext::new()?)),
                    cpu_isa: None,
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
                    cpu_isa: None,
                })
            }
            #[cfg(all(
                feature = "vulkan",
                any(target_os = "linux", target_os = "android", target_os = "windows")
            ))]
            BackendKind::Vulkan => {
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_vulkan()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "vulkan: the Compute seam has no arm for {op:?}; the model requires \
                         {required:?}. NOTE — the SPIR-V kernels are NOT the gap: \
                         `crates/vokra-backend-vulkan/kernels/precompiled/` ships all 12 .spv \
                         blobs (M4-13-T16) and `VulkanBackend` exposes gemm_f32 / gemv_f32 / \
                         softmax_f32 / softmax_causal_f32 / layer_norm_f32 / gelu_f32 / \
                         conv1d_f32 over them (M4-13-T03〜T08). What is missing is a \
                         `Be::Vulkan` variant in `vokra_models::compute` delegating to those \
                         methods. One model = one backend — Vokra does not silently run the \
                         uncovered ops on the CPU (FR-EX-08). Select BackendKind::Cpu until \
                         that seam arm lands."
                    )));
                }
                // `required` is empty AND every hot op is uncovered — there is
                // no `Be::Vulkan` variant to construct a dispatcher around, so
                // surface an explicit error rather than pretending a
                // coverage-empty dispatcher is usable. Wiring that variant
                // turns this branch into `Ok(Compute { be: Be::Vulkan(...) })`
                // — the same shape as the Metal / CUDA arms above.
                Err(VokraError::UnsupportedOp(
                    "vulkan: `vokra_models::compute` has no `Be::Vulkan` seam arm, so no \
                     covered required set exists. The SPIR-V kernels are NOT the gap — all 12 \
                     .spv blobs and the typed `VulkanBackend::*_f32` dispatch entry points have \
                     landed; the seam arm delegating to them has not."
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
                    cpu_isa: None,
                })
            }
            #[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
            BackendKind::CoreMl => {
                // Probe first: no reachable Apple Neural Engine is a
                // `BackendUnavailable` (an Intel Mac, or a runner that hides
                // the ANE), never a silent CPU fall back (FR-EX-08 / NFR-RL-06).
                let caps = vokra_backend_coreml::vokra_coreml_probe()?;
                // Coverage gate, kept in lock-step with `covered_by_coreml`
                // (the `coreml_coverage_is_empty_in_scaffold` test pins the
                // two together): in the scaffold slice every op is uncovered,
                // so a non-empty `required` reports the first uncovered op.
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_coreml()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "coreml delegate backend has no wired execution path for {op:?} in the \
                         M5-01 scaffold slice ({}); the model requires {required:?}. The op path \
                         lands after the M5-01-T02 model-supply ADR is ratified. One model = one \
                         backend — Vokra does not silently run the uncovered ops on the CPU \
                         (FR-EX-08). Select BackendKind::Cpu explicitly for now.",
                        caps.summary()
                    )));
                }
                // `required` empty AND every op uncovered — the scaffold cannot
                // construct a usable `Compute::CoreMl` dispatcher (no callable
                // execution path). Surface an explicit error rather than
                // pretending a coverage-empty dispatcher is usable (same shape
                // as the Vulkan foundation slice). Once the T02 ADR lands and
                // the execution path is wired, this becomes
                // `Ok(Compute { be: Be::CoreMl(...) })`.
                Err(VokraError::UnsupportedOp(format!(
                    "coreml delegate Compute path has no wired execution in the M5-01 scaffold \
                     slice ({}) — no covered required set exists. Wait for the M5-01-T02 ADR + \
                     the execution-path ticket.",
                    caps.summary()
                )))
            }
            #[cfg(all(
                feature = "qnn",
                any(target_os = "android", target_os = "linux", target_os = "windows")
            ))]
            BackendKind::Qnn => {
                // Probe first: no reachable QNN runtime is a `BackendUnavailable`
                // (no SDK installed, a runner without the Hexagon runtime), never
                // a silent CPU fall back (FR-EX-08 / NFR-RL-06). Mirrors the
                // CoreML delegate arm above (the sister M5-01 WP).
                let caps = vokra_backend_qnn::vokra_qnn_probe()?;
                // Coverage gate, kept in lock-step with `covered_by_qnn` (the
                // `qnn_coverage_is_empty_in_scaffold` test pins the two
                // together): in the scaffold slice every op is uncovered, so a
                // non-empty `required` reports the first uncovered op.
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_qnn()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "qnn delegate backend has no wired execution path for {op:?} in the M5-02 \
                         scaffold slice ({}); the model requires {required:?}. QNN graph \
                         construction lands in the SDK-gated re-issue wave. One model = one \
                         backend — Vokra does not silently run the uncovered ops on the CPU \
                         (FR-EX-08). Select BackendKind::Cpu explicitly for now.",
                        caps.summary()
                    )));
                }
                // `required` empty AND every op uncovered — the scaffold cannot
                // construct a usable `Compute::Qnn` dispatcher (no callable
                // execution path). Surface an explicit error rather than
                // pretending a coverage-empty dispatcher is usable (same shape
                // as the Vulkan foundation slice / CoreML scaffold). Once the
                // graph-construction re-issue wave lands (owner T11 gates it),
                // this becomes `Ok(Compute { be: Be::Qnn(...) })`.
                Err(VokraError::UnsupportedOp(format!(
                    "qnn delegate Compute path has no wired execution in the M5-02 scaffold slice \
                     ({}) — no covered required set exists. Wait for the SDK-gated \
                     graph-construction re-issue wave.",
                    caps.summary()
                )))
            }
            other => Err(VokraError::BackendUnavailable(format!(
                "{other:?} backend is not built into vokra-models (build with the `metal` feature \
                 on macOS / iOS for Metal, the `cuda` feature on Windows / Linux for CUDA, the \
                 `vulkan` feature on Linux / Android / Windows for Vulkan, the `webgpu` \
                 feature on wasm32 for browser WebGPU, or the `qnn` feature on Android / Linux / \
                 Windows for the Qualcomm Hexagon NPU delegate)"
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
            Be::Cpu => match self.cpu_isa {
                Some(isa) => kernels::gemm_f32_on(isa, m, n, k, a, b, bias, out),
                None => kernels::gemm_f32(m, n, k, a, b, bias, out),
            },
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
            Be::Cpu => match self.cpu_isa {
                Some(isa) => kernels::softmax_f32_on(isa, input, out, rows, cols),
                None => kernels::softmax_f32(input, out, rows, cols),
            },
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
            Be::Cpu => match self.cpu_isa {
                Some(isa) => {
                    kernels::layer_norm_f32_on(isa, input, out, rows, cols, gamma, beta, eps)
                }
                None => kernels::layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            },
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
        }
    }

    /// Gamma-only RMS normalisation over the innermost axis of a
    /// `rows × cols` buffer. Unlike [`Self::layer_norm_f32`], this does not
    /// subtract a mean and has no beta term.
    pub fn rms_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        eps: f32,
    ) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "rms_norm_f32: rows and cols must be non-zero, got {rows}x{cols}"
            )));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "rms_norm_f32: eps must be finite and positive, got {eps}"
            )));
        }
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            VokraError::InvalidArgument("rms_norm_f32: rows*cols overflow".to_owned())
        })?;
        if input.len() != expected || out.len() != expected || gamma.len() != cols {
            return Err(VokraError::InvalidArgument(format!(
                "rms_norm_f32: expected input/out {expected} and gamma {cols}, got input {}, out {}, gamma {}",
                input.len(),
                out.len(),
                gamma.len()
            )));
        }
        match &self.be {
            Be::Cpu => {
                for row in 0..rows {
                    let start = row * cols;
                    let src = &input[start..start + cols];
                    let sum_sq: f32 = src.iter().map(|value| value * value).sum();
                    let inverse_rms = 1.0 / (sum_sq / cols as f32 + eps).sqrt();
                    for col in 0..cols {
                        out[start + col] = src[col] * inverse_rms * gamma[col];
                    }
                }
                Ok(())
            }
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.rms_norm_f32(input, out, rows, cols, gamma, eps),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "rms_norm_f32 has no wired CUDA Compute-seam kernel; Vokra does not silently run it on the CPU"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "rms_norm_f32 has no wired WebGPU Compute-seam kernel; Vokra does not silently run it on the CPU"
                    .to_owned(),
            )),
        }
    }

    /// Scalar-gain ScaleNorm over the innermost axis of a `rows × cols`
    /// buffer. Unlike [`Self::rms_norm_f32`], epsilon clamps the completed
    /// scaled L2 norm instead of being added inside the square root.
    pub fn scale_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gain: f32,
        eps: f32,
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::scale_norm_f32(input, out, rows, cols, gain, eps),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.scale_norm_f32(input, out, rows, cols, gain, eps),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "scale_norm_f32 has no wired CUDA Compute-seam kernel; Vokra does not silently run it on the CPU"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "scale_norm_f32 has no wired WebGPU Compute-seam kernel; Vokra does not silently run it on the CPU"
                    .to_owned(),
            )),
        }
    }

    /// One-group affine GroupNorm over channel-major `[channels, positions]`.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        channels: usize,
        positions: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::group_norm_f32(input, out, channels, positions, gamma, beta, eps),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.group_norm_f32(input, out, channels, positions, gamma, beta, eps),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "group_norm_f32 has no wired CUDA kernel; Vokra does not silently run the op on \
                 the CPU (FR-EX-08)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "group_norm_f32 has no wired WebGPU kernel; Vokra does not silently run the op \
                 on the CPU (FR-EX-08)"
                    .to_owned(),
            )),
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

    /// Element-wise GPT-2 / Transformers `gelu_new` tanh approximation.
    pub fn gelu_new_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gelu_new_f32(x, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.gelu_new_f32(x, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "gelu_new_f32 has no wired CUDA Compute-seam kernel; Vokra does not silently run it on the CPU (FR-EX-08)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "gelu_new_f32 has no wired WebGPU Compute-seam kernel; Vokra does not silently run it on the CPU (FR-EX-08)"
                    .to_owned(),
            )),
        }
    }

    /// Element-wise ReLU (`out = max(x, 0)`).
    ///
    /// Metal uses its dedicated MSL kernel. CUDA and WebGPU are explicit
    /// unsupported-operation arms so a model listing [`HotOp::Relu`] cannot
    /// silently execute this activation on the host.
    pub fn relu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::relu_f32(x, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.relu_f32(x, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "relu_f32 has no wired CUDA Compute-seam kernel; Vokra does not silently run it on the CPU (FR-EX-08)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "relu_f32 has no wired WebGPU Compute-seam kernel; Vokra does not silently run it on the CPU (FR-EX-08)"
                    .to_owned(),
            )),
        }
    }

    /// Element-wise hyperbolic tangent (`out = tanh(x)`).
    ///
    /// Metal uses its dedicated MSL kernel. CUDA and WebGPU are explicit
    /// unsupported-operation arms so a model listing [`HotOp::Tanh`] cannot
    /// silently execute this activation on the host.
    pub fn tanh_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::tanh_f32(x, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.tanh_f32(x, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "tanh_f32 has no wired CUDA Compute-seam kernel; Vokra does not silently run it on the CPU (FR-EX-08)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "tanh_f32 has no wired WebGPU Compute-seam kernel; Vokra does not silently run it on the CPU (FR-EX-08)"
                    .to_owned(),
            )),
        }
    }

    /// Element-wise SiLU / Swish (`out = x * sigmoid(x)`).
    ///
    /// Metal uses its native MSL kernel. CUDA and WebGPU are explicit
    /// unsupported-operation arms because listing [`HotOp::Silu`] must never
    /// conceal a host fallback.
    pub fn silu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != out.len() {
            return Err(VokraError::InvalidArgument(format!(
                "silu_f32: input length {} != output length {}",
                x.len(),
                out.len()
            )));
        }
        match &self.be {
            Be::Cpu => {
                for (output, &value) in out.iter_mut().zip(x) {
                    *output = value / (1.0 + (-value).exp());
                }
                Ok(())
            }
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.silu_f32(x, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "silu_f32 has no wired CUDA Compute-seam kernel; Vokra does not silently run it on the CPU"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "silu_f32 has no wired WebGPU Compute-seam kernel; Vokra does not silently run it on the CPU"
                    .to_owned(),
            )),
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
            Be::Cpu => match self.cpu_isa {
                Some(isa) => kernels::conv1d_f32_on(
                    isa, input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
                ),
                None => kernels::conv1d_f32(
                    input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
                ),
            },
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

    /// Grouped 1-D convolution. The CPU and Metal arms execute real grouped
    /// kernels; other GPU arms refuse explicitly because their coverage gate
    /// does not list [`HotOp::GroupedConv1d`].
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_conv1d_f32(
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
        groups: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::grouped_conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, groups, out,
            ),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.grouped_conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, groups, out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "grouped_conv1d has no CUDA Compute-seam kernel; no CPU fallback is performed"
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "grouped_conv1d has no WebGPU Compute-seam kernel; no CPU fallback is performed"
                    .to_owned(),
            )),
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
            Be::Metal(ctx) => {
                // Explicit shape + index validation on the host. The MSL kernel
                // guards `t >= time` and `d >= d_model` but has no per-element
                // bound check on `codes[..]`; silent OOB reads inside the
                // gather + fold are the failure mode we prevent by mirroring
                // the shape checks that `vokra_ops::mimi_rvq::check_tables_shape`
                // / `check_codes_shape` / `CodebookTable::row` do on the CPU
                // arm (FR-EX-08 — never a silent GPU OOB or CPU fall back).
                if attrs.n_codebooks == 0 || attrs.codebook_size == 0 || attrs.d_model == 0 {
                    return Err(VokraError::InvalidArgument(format!(
                        "mimi_rvq_f32 metal: attrs must have every axis > 0, got n_codebooks={} \
                         codebook_size={} d_model={}",
                        attrs.n_codebooks, attrs.codebook_size, attrs.d_model,
                    )));
                }
                if codebook_tables.len() != attrs.n_codebooks {
                    return Err(VokraError::InvalidArgument(format!(
                        "mimi_rvq_f32 metal: codebook_tables.len() {} != attrs.n_codebooks {}",
                        codebook_tables.len(),
                        attrs.n_codebooks
                    )));
                }
                for (i, t) in codebook_tables.iter().enumerate() {
                    if t.codebook_size != attrs.codebook_size || t.d_model != attrs.d_model {
                        return Err(VokraError::InvalidArgument(format!(
                            "mimi_rvq_f32 metal: codebook_tables[{i}] shape [{},{}] != attrs [{},{}]",
                            t.codebook_size, t.d_model, attrs.codebook_size, attrs.d_model
                        )));
                    }
                }
                let expected_codes = time.checked_mul(attrs.n_codebooks).ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "mimi_rvq_f32 metal: time ({time}) * n_codebooks ({}) overflows usize",
                        attrs.n_codebooks
                    ))
                })?;
                if codes.len() != expected_codes {
                    return Err(VokraError::InvalidArgument(format!(
                        "mimi_rvq_f32 metal: codes.len() {} != time * n_codebooks {expected_codes}",
                        codes.len()
                    )));
                }
                // Per-index bound check — the MSL kernel does NOT range-check
                // `codes[..]`, so a stray index would be a silent OOB gather
                // (FR-EX-08). Cheap: O(time * n_codebooks) unpredictable
                // branches, dwarfed by the FP32 fold on the GPU.
                for &idx in codes {
                    if (idx as usize) >= attrs.codebook_size {
                        return Err(VokraError::InvalidArgument(format!(
                            "mimi_rvq_f32 metal: codes contains index {idx} >= codebook_size {}",
                            attrs.codebook_size
                        )));
                    }
                }
                if time == 0 {
                    return Ok(Vec::new());
                }
                // Flatten [n_codebooks][codebook_size, d_model] into one
                // row-major buffer of `n_codebooks * codebook_size * d_model`
                // FP32s. This is the layout the MSL kernel's stride math
                // (`cb * cb_stride + idx * d_model + delem`) expects. Chunk
                // granularity — allocating one Vec here is negligible next
                // to the GPU dispatch (matches the heap-returning shape).
                let mut tables_flat =
                    Vec::with_capacity(attrs.n_codebooks * attrs.codebook_size * attrs.d_model);
                for tbl in codebook_tables {
                    tables_flat.extend_from_slice(&tbl.data);
                }
                ctx.mimi_rvq_gather_fold_f32(
                    codes,
                    &tables_flat,
                    attrs.n_codebooks,
                    attrs.codebook_size,
                    attrs.d_model,
                    time,
                )
            }
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
            Be::Metal(ctx) => {
                // Explicit shape + index validation on the host. The MSL kernel
                // guards `t >= time` and `d >= d_model` but has no per-element
                // bound check on `codes[..]`; silent OOB reads inside the
                // factorized-projection fold are the failure mode we prevent
                // by mirroring the CPU-arm shape checks in `vokra_ops::dac_rvq`
                // (FR-EX-08 — never a silent GPU OOB or CPU fall back).
                if attrs.n_codebooks == 0
                    || attrs.codebook_size == 0
                    || attrs.codebook_dim == 0
                    || attrs.d_model == 0
                {
                    return Err(VokraError::InvalidArgument(format!(
                        "dac_rvq_f32 metal: attrs must have every axis > 0, got n_codebooks={} \
                         codebook_size={} codebook_dim={} d_model={}",
                        attrs.n_codebooks, attrs.codebook_size, attrs.codebook_dim, attrs.d_model,
                    )));
                }
                if codebook_tables.len() != attrs.n_codebooks {
                    return Err(VokraError::InvalidArgument(format!(
                        "dac_rvq_f32 metal: codebook_tables.len() {} != attrs.n_codebooks {}",
                        codebook_tables.len(),
                        attrs.n_codebooks
                    )));
                }
                for (i, t) in codebook_tables.iter().enumerate() {
                    // DAC's `CodebookTable::d_model` field holds the row
                    // width, which for the low-dim factorized table must be
                    // `attrs.codebook_dim` — NOT `attrs.d_model` (mirror of
                    // `vokra_ops::dac_rvq::check_shapes`).
                    if t.codebook_size != attrs.codebook_size || t.d_model != attrs.codebook_dim {
                        return Err(VokraError::InvalidArgument(format!(
                            "dac_rvq_f32 metal: codebook_tables[{i}] shape [{},{}] != attrs \
                             [{},{}] (row width must be the factorized codebook_dim)",
                            t.codebook_size, t.d_model, attrs.codebook_size, attrs.codebook_dim
                        )));
                    }
                }
                if out_projs.len() != attrs.n_codebooks {
                    return Err(VokraError::InvalidArgument(format!(
                        "dac_rvq_f32 metal: out_projs.len() {} != attrs.n_codebooks {}",
                        out_projs.len(),
                        attrs.n_codebooks
                    )));
                }
                for (i, p) in out_projs.iter().enumerate() {
                    if p.d_model != attrs.d_model || p.codebook_dim != attrs.codebook_dim {
                        return Err(VokraError::InvalidArgument(format!(
                            "dac_rvq_f32 metal: out_projs[{i}] shape [{},{}] != attrs [{},{}]",
                            p.d_model, p.codebook_dim, attrs.d_model, attrs.codebook_dim
                        )));
                    }
                }
                let expected_codes = time.checked_mul(attrs.n_codebooks).ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "dac_rvq_f32 metal: time ({time}) * n_codebooks ({}) overflows usize",
                        attrs.n_codebooks
                    ))
                })?;
                if codes.len() != expected_codes {
                    return Err(VokraError::InvalidArgument(format!(
                        "dac_rvq_f32 metal: codes.len() {} != time * n_codebooks {expected_codes}",
                        codes.len()
                    )));
                }
                // Per-index bound check — the MSL kernel does NOT range-check
                // `codes[..]`, so a stray index would be a silent OOB gather
                // (FR-EX-08). Cheap: O(time * n_codebooks) unpredictable
                // branches, dwarfed by the GPU dispatch.
                for &idx in codes {
                    if (idx as usize) >= attrs.codebook_size {
                        return Err(VokraError::InvalidArgument(format!(
                            "dac_rvq_f32 metal: codes contains index {idx} >= codebook_size {}",
                            attrs.codebook_size
                        )));
                    }
                }
                if time == 0 {
                    return Ok(Vec::new());
                }
                // Flatten the three per-quantizer arrays into contiguous
                // buffers matching the MSL kernel's stride math. Chunk
                // granularity — three Vec allocations here are negligible next
                // to the GPU dispatch.
                let mut low_tables_flat = Vec::with_capacity(
                    attrs.n_codebooks * attrs.codebook_size * attrs.codebook_dim,
                );
                for tbl in codebook_tables {
                    low_tables_flat.extend_from_slice(&tbl.data);
                }
                let mut proj_weights_flat =
                    Vec::with_capacity(attrs.n_codebooks * attrs.d_model * attrs.codebook_dim);
                for p in out_projs {
                    proj_weights_flat.extend_from_slice(&p.weight);
                }
                let mut proj_biases_flat = Vec::with_capacity(attrs.n_codebooks * attrs.d_model);
                for p in out_projs {
                    proj_biases_flat.extend_from_slice(&p.bias);
                }
                ctx.dac_rvq_gather_project_fold_f32(
                    codes,
                    &low_tables_flat,
                    &proj_weights_flat,
                    &proj_biases_flat,
                    attrs.n_codebooks,
                    attrs.codebook_size,
                    attrs.codebook_dim,
                    attrs.d_model,
                    time,
                )
            }
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
    /// Same shape-generic gather + FP32 fold as [`Compute::mimi_rvq_f32`].
    /// The CPU arm delegates verbatim to [`vokra_ops::encodec_rvq_decode`];
    /// Metal dispatches the already-pinned Mimi gather/fold kernel after
    /// EnCodec-specific host validation. CUDA / WebGPU remain explicit
    /// [`VokraError::UnsupportedOp`] (FR-EX-08 — no silent CPU fall back).
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates [`vokra_ops::encodec_rvq_decode`]'s
    ///   [`VokraError::InvalidArgument`].
    /// - Metal arm: EnCodec-specific validation plus backend failures.
    /// - CUDA / WebGPU arms: explicit [`VokraError::UnsupportedOp`].
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
            Be::Metal(ctx) => {
                if attrs.n_codebooks == 0 || attrs.codebook_size == 0 || attrs.d_model == 0 {
                    return Err(VokraError::InvalidArgument(format!(
                        "encodec_rvq_f32 metal: attrs must have every axis > 0, got \
                         n_codebooks={} codebook_size={} d_model={}",
                        attrs.n_codebooks, attrs.codebook_size, attrs.d_model,
                    )));
                }
                if codebook_tables.len() != attrs.n_codebooks {
                    return Err(VokraError::InvalidArgument(format!(
                        "encodec_rvq_f32 metal: codebook_tables.len() {} != \
                         attrs.n_codebooks {}",
                        codebook_tables.len(),
                        attrs.n_codebooks
                    )));
                }
                for (index, table) in codebook_tables.iter().enumerate() {
                    if table.codebook_size != attrs.codebook_size || table.d_model != attrs.d_model
                    {
                        return Err(VokraError::InvalidArgument(format!(
                            "encodec_rvq_f32 metal: codebook_tables[{index}] shape [{},{}] \
                             != attrs [{},{}]",
                            table.codebook_size, table.d_model, attrs.codebook_size, attrs.d_model
                        )));
                    }
                }
                let expected_codes = time.checked_mul(attrs.n_codebooks).ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "encodec_rvq_f32 metal: time ({time}) * n_codebooks ({}) overflows usize",
                        attrs.n_codebooks
                    ))
                })?;
                if codes.len() != expected_codes {
                    return Err(VokraError::InvalidArgument(format!(
                        "encodec_rvq_f32 metal: codes.len() {} != time * n_codebooks \
                         {expected_codes}",
                        codes.len()
                    )));
                }
                for (position, &code) in codes.iter().enumerate() {
                    if (code as usize) >= attrs.codebook_size {
                        return Err(VokraError::InvalidArgument(format!(
                            "encodec_rvq_f32 metal: codes[{position}] = {code} >= \
                             codebook_size {} (no silent clamp — FR-EX-08)",
                            attrs.codebook_size
                        )));
                    }
                }
                if time == 0 {
                    return Ok(Vec::new());
                }
                let table_values = attrs
                    .n_codebooks
                    .checked_mul(attrs.codebook_size)
                    .and_then(|value| value.checked_mul(attrs.d_model))
                    .ok_or_else(|| {
                        VokraError::InvalidArgument(
                            "encodec_rvq_f32 metal: flattened codebook size overflows usize"
                                .to_owned(),
                        )
                    })?;
                let mut tables_flat = Vec::with_capacity(table_values);
                for table in codebook_tables {
                    tables_flat.extend_from_slice(&table.data);
                }
                ctx.mimi_rvq_gather_fold_f32(
                    codes,
                    &tables_flat,
                    attrs.n_codebooks,
                    attrs.codebook_size,
                    attrs.d_model,
                    time,
                )
            }
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
    /// # Metal wired; CUDA deferred
    ///
    /// The CPU arm delegates verbatim to
    /// [`vokra_ops::wavtokenizer_vq_decode`] (bit-for-bit vs a direct call);
    /// the **Metal** arm dispatches the M4-16 WF2 kernel
    /// (`vokra_wavtokenizer_vq_gather_f32`, bit-identical to the CPU gather
    /// within the FP32 5e-4 codec-family bound); the **CUDA** arm returns an
    /// explicit [`VokraError::UnsupportedOp`] — the M4-16 GPU kernels are deferred
    /// (single-stage gather bound: the Metal path reuses the same gather
    /// primitive as mimi_rvq / dac_rvq), and Vokra never silently substitutes
    /// the CPU (FR-EX-08). The [`Compute::for_backend`] coverage gate accepts
    /// [`HotOp::WavTokenizerVq`] against Metal (post M4-16 WF2) but rejects
    /// it against CUDA / Vulkan / WebGPU (their arms remain deferred).
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates [`vokra_ops::wavtokenizer_vq_decode`]'s
    ///   [`VokraError::InvalidArgument`] (shape mismatch, out-of-range code).
    /// - Metal arm: propagates the same [`VokraError::InvalidArgument`]
    ///   variants (mirrored on the host before dispatch, FR-EX-08) plus
    ///   [`VokraError::BackendUnavailable`] on a Metal device / command
    ///   failure.
    /// - CUDA / WebGPU arms: explicit [`VokraError::UnsupportedOp`].
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
            Be::Metal(ctx) => {
                // Explicit shape + index validation on the host. The MSL
                // kernel guards `t >= time` and `d >= d_model` but has no
                // per-element bound check on `codes[..]`; silent OOB reads
                // inside the gather are the failure mode we prevent by
                // mirroring the CPU-arm shape checks in
                // `vokra_ops::wavtokenizer_vq_decode` (FR-EX-08 — never a
                // silent GPU OOB or CPU fall back).
                if attrs.vocab_size == 0 || attrs.d_model == 0 {
                    return Err(VokraError::InvalidArgument(format!(
                        "wavtokenizer_vq_f32 metal: attrs must have every axis > 0, got \
                         vocab_size={} d_model={}",
                        attrs.vocab_size, attrs.d_model
                    )));
                }
                if codebook_table.codebook_size != attrs.vocab_size
                    || codebook_table.d_model != attrs.d_model
                {
                    return Err(VokraError::InvalidArgument(format!(
                        "wavtokenizer_vq_f32 metal: codebook_table shape [{},{}] != attrs [{},{}]",
                        codebook_table.codebook_size,
                        codebook_table.d_model,
                        attrs.vocab_size,
                        attrs.d_model
                    )));
                }
                if codes.len() != time {
                    return Err(VokraError::InvalidArgument(format!(
                        "wavtokenizer_vq_f32 metal: codes.len() {} != time {time} (single \
                         codebook — one code per timestep; the [time, n_codebooks] layout is \
                         the RVQ family's)",
                        codes.len()
                    )));
                }
                // Per-index bound check — the MSL kernel does NOT range-check
                // `codes[..]`, so a stray index would be a silent OOB gather
                // (FR-EX-08). Cheap: O(time) unpredictable branches, dwarfed
                // by the GPU dispatch.
                for &idx in codes {
                    if (idx as usize) >= attrs.vocab_size {
                        return Err(VokraError::InvalidArgument(format!(
                            "wavtokenizer_vq_f32 metal: codes contains index {idx} >= \
                             vocab_size {} (no silent clamp — FR-EX-08)",
                            attrs.vocab_size
                        )));
                    }
                }
                if time == 0 {
                    return Ok(Vec::new());
                }
                // `CodebookTable::data` is already the flat row-major buffer
                // the MSL kernel expects — no re-layout needed. Chunk
                // granularity means the &[f32] pass-through is negligible next
                // to the GPU dispatch.
                ctx.wavtokenizer_vq_gather_f32(
                    codes,
                    &codebook_table.data,
                    attrs.vocab_size,
                    attrs.d_model,
                    time,
                )
            }
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
    /// # Metal wired; CUDA deferred
    ///
    /// The CPU arm delegates verbatim to [`vokra_ops::xcodec2_fsq_decode`];
    /// the **Metal** arm dispatches the M4-16 WF2 kernel
    /// (`vokra_xcodec2_fsq_decode_f32`, grid decompose + optional Linear
    /// GEMV, semantics equal to the CPU op within the FP32 5e-4 codec-
    /// family bound). The **CUDA** arm is an explicit
    /// [`VokraError::UnsupportedOp`] (FR-EX-08 — no silent CPU fall back),
    /// and the coverage gate accepts [`HotOp::Xcodec2Fsq`] against Metal but
    /// rejects it against CUDA / Vulkan / WebGPU.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates [`vokra_ops::xcodec2_fsq_decode`]'s
    ///   [`VokraError::InvalidArgument`].
    /// - Metal arm: propagates the same [`VokraError::InvalidArgument`]
    ///   variants (mirrored on the host before dispatch, FR-EX-08) plus
    ///   [`VokraError::BackendUnavailable`] on a Metal device / command
    ///   failure.
    /// - CUDA / WebGPU arms: explicit [`VokraError::UnsupportedOp`].
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
            Be::Metal(ctx) => {
                // Explicit shape + level + index validation on the host,
                // mirroring the CPU-arm shape checks in
                // `vokra_ops::xcodec2_fsq_decode` (FR-EX-08 — never a silent
                // GPU OOB, divide-by-zero, or CPU fall back). The MSL kernel's
                // `half_width = levels[k] / 2` needs `levels[k] ≥ 2` to be
                // `≥ 1` (else divide-by-zero in the grid formula) — validate
                // it upstream of the dispatch.
                if attrs.d_model == 0 {
                    return Err(VokraError::InvalidArgument(
                        "xcodec2_fsq_f32 metal: attrs.d_model must be > 0".to_owned(),
                    ));
                }
                let n_dims = attrs.n_dims();
                if n_dims == 0 {
                    return Err(VokraError::InvalidArgument(
                        "xcodec2_fsq_f32 metal: attrs.levels must be non-empty".to_owned(),
                    ));
                }
                // Effective vocab = Π levels — this both validates `levels`
                // (every entry ≥ 2, no overflow) and gives the per-code bound.
                let vocab = attrs.effective_vocab()?;
                match out_proj {
                    Some(proj) => {
                        if proj.n_dims != n_dims || proj.d_model != attrs.d_model {
                            return Err(VokraError::InvalidArgument(format!(
                                "xcodec2_fsq_f32 metal: out_proj shape [{},{}] != attrs \
                                 [d_model={}, n_dims={n_dims}]",
                                proj.d_model, proj.n_dims, attrs.d_model
                            )));
                        }
                    }
                    None => {
                        if attrs.d_model != n_dims {
                            return Err(VokraError::InvalidArgument(format!(
                                "xcodec2_fsq_f32 metal: out_proj = None (Identity) requires \
                                 d_model == len(levels), got d_model={} len(levels)={n_dims} \
                                 — the released X-Codec 2 projects 8 → 2048 and must pass \
                                 Some(&FsqOutProj)",
                                attrs.d_model
                            )));
                        }
                    }
                }
                if codes.len() != time {
                    return Err(VokraError::InvalidArgument(format!(
                        "xcodec2_fsq_f32 metal: codes.len() {} != time {time} (single-stage — \
                         one code per timestep; the [time, n_codebooks] layout is the RVQ \
                         family's)",
                        codes.len()
                    )));
                }
                for (t, &idx) in codes.iter().enumerate() {
                    if (idx as usize) >= vocab {
                        return Err(VokraError::InvalidArgument(format!(
                            "xcodec2_fsq_f32 metal: codes[{t}] = {idx} >= Π levels {vocab} \
                             (no silent clamp — FR-EX-08)"
                        )));
                    }
                }
                if time == 0 {
                    return Ok(Vec::new());
                }
                let (w_slice, b_slice) = match out_proj {
                    Some(p) => (Some(p.weight.as_slice()), Some(p.bias.as_slice())),
                    None => (None, None),
                };
                ctx.xcodec2_fsq_decode_f32(
                    codes,
                    &attrs.levels,
                    w_slice,
                    b_slice,
                    attrs.d_model,
                    n_dims,
                    time,
                )
            }
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

    /// Snake activation (Ziyin et al. 2020) — the per-channel closed-form
    /// periodic activation shared by the BigVGAN / HiFTNet / Kokoro-82M
    /// vocoder lineage, wired into the imperative `Compute` seam.
    ///
    /// Applies `out[c, t] = x[c, t] + (1 / (alpha[c] + ε)) · sin(alpha[c] ·
    /// x[c, t])²` for a `[channels, time]` row-major FP32 tensor (channel-
    /// outer). `alpha` is length-`channels`; `x` and `out` are both length
    /// `channels · time`. Delegates on the CPU arm to
    /// [`vokra_ops::snake_activation_f32`] (which is bit-identical to
    /// [`vokra_ops::hiftnet::Snake::forward_in_place`] under
    /// `alpha_logscale = false` and the private
    /// `vokra_models::kokoro::nn::snake_activation` helper — same eps, same
    /// primitives, trivial reduction).
    ///
    /// # `alpha_logscale` and `SnakeBeta` are NOT this op
    ///
    /// - `alpha_logscale = true` is an upstream-side transformation
    ///   (`alpha_eff = exp(alpha_raw)`) applied by the converter or the
    ///   stateful [`vokra_ops::hiftnet::Snake`]; the caller passes the
    ///   already-effective vector to this method.
    /// - `SnakeBeta` (`y = x + (1/(β+ε))·sin(α·x)²`, two per-channel
    ///   vectors) is a different closed form and is provided by
    ///   [`vokra_ops::bigvgan_generator::SnakeBeta`] — not through this seam.
    ///
    /// # CPU-only through this seam today (Metal wired; CUDA / Vulkan /
    /// WebGPU / CoreML / QNN return `UnsupportedOp`)
    ///
    /// **Vocoder Metal wave WF2 (2026-08-13):** the Metal arm dispatches to
    /// [`vokra_backend_metal::MetalContext::snake_activation_f32`]
    /// (`vokra_snake_activation_f32` MSL kernel — semantics equal to the CPU
    /// free function within the FP32 transcendental gap, `atol ≤ 5e-4`).
    /// Every other GPU arm returns an explicit
    /// [`VokraError::UnsupportedOp`] until the corresponding kernel lands —
    /// never a silent CPU fall back (FR-EX-08). The coverage gate on
    /// [`Compute::for_backend`] additionally rejects any model that lists
    /// [`HotOp::SnakeActivation`] against those backends.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`vokra_ops::snake_activation_f32`] raises (wrong `alpha.len()` /
    ///   `x.len()` / `out.len()`, or `channels * time` overflow — never a
    ///   silent shape clamp, FR-EX-08).
    /// - Metal arm: same host-side shape validation before dispatch, plus
    ///   any [`VokraError::BackendUnavailable`] from the underlying command
    ///   buffer.
    /// - CUDA / Vulkan / WebGPU / CoreML / QNN arms: explicit
    ///   [`VokraError::UnsupportedOp`] until the GPU kernel lands.
    pub fn snake_activation_f32(
        &self,
        x: &[f32],
        alpha: &[f32],
        channels: usize,
        time: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => vokra_ops::snake_activation_f32(x, alpha, channels, time, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.snake_activation_f32(x, alpha, channels, time, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "snake_activation_f32 has no wired CUDA NVRTC kernel; the Vocoder wave CUDA \
                 arm is deferred to the vast.ai owner track. Select BackendKind::Cpu (which \
                 delegates to vokra_ops::snake_activation_f32), or wait for the CUDA kernel — \
                 Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "snake_activation_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six \
                 Whisper hot ops only; the Vocoder wave GPU arms are deferred like \
                 Metal/CUDA/Vulkan). Select BackendKind::Cpu (which delegates to \
                 vokra_ops::snake_activation_f32) — Vokra does not silently run the op on the \
                 CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// SnakeBeta activation (Ziyin et al. 2020 + Lee et al. 2023 BigVGAN) —
    /// the per-channel **two-vector** closed-form periodic activation shared
    /// by the BigVGAN family, wired into the imperative `Compute` seam.
    ///
    /// Applies `out[c, t] = x[c, t] + (1 / (beta[c] + ε)) · sin(alpha[c] ·
    /// x[c, t])²` for a `[channels, time]` row-major FP32 tensor (channel-
    /// outer). `alpha` and `beta` are length-`channels`; `x` and `out` are
    /// both length `channels · time`. Delegates on the CPU arm to
    /// [`vokra_ops::snake_beta_f32`] (which is bit-identical to
    /// [`vokra_ops::bigvgan_generator::SnakeBeta::forward_in_place`] under
    /// `alpha_logscale = false` — same eps, same primitives, trivial per-
    /// element).
    ///
    /// # Distinct from [`Compute::snake_activation_f32`]
    ///
    /// The plain-Snake closed form ties frequency and magnitude to a single
    /// per-channel `alpha`. SnakeBeta separates the two: `alpha` scales the
    /// sinusoid argument (frequency), `beta` scales the reciprocal in front
    /// of the squared sine (magnitude). Two per-channel weight vectors →
    /// distinct op shape, distinct MSL kernel, distinct HotOp variant.
    ///
    /// # `alpha_logscale` and the stateful `SnakeBeta`
    ///
    /// `alpha_logscale = true` (upstream `bigvgan_generator::SnakeBeta` with
    /// `snake_logscale = true`) is an upstream-side transformation
    /// (`alpha_eff = exp(alpha_raw)`, `beta_eff = exp(beta_raw)`); the
    /// caller passes the already-effective vectors to this method (same
    /// contract as [`Compute::snake_activation_f32`]).
    ///
    /// # CPU + Metal wired; CUDA / Vulkan / WebGPU / CoreML / QNN return `UnsupportedOp`
    ///
    /// **Vocoder Metal wave (2026-08-14):** the Metal arm dispatches to
    /// [`vokra_backend_metal::MetalContext::snake_beta_f32`]
    /// (`vokra_snake_beta_f32` MSL kernel — semantics equal to the CPU free
    /// function within the FP32 transcendental gap, `atol ≤ 5e-4`). Every
    /// other GPU arm returns an explicit [`VokraError::UnsupportedOp`]
    /// until the corresponding kernel lands — never a silent CPU fall back
    /// (FR-EX-08). The coverage gate on [`Compute::for_backend`]
    /// additionally rejects any model that lists [`HotOp::SnakeBeta`]
    /// against those backends.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`vokra_ops::snake_beta_f32`] raises (wrong `alpha.len()` /
    ///   `beta.len()` / `x.len()` / `out.len()`, or `channels * time`
    ///   overflow — never a silent shape clamp, FR-EX-08).
    /// - Metal arm: same host-side shape validation before dispatch, plus
    ///   any [`VokraError::BackendUnavailable`] from the underlying command
    ///   buffer.
    /// - CUDA / Vulkan / WebGPU / CoreML / QNN arms: explicit
    ///   [`VokraError::UnsupportedOp`] until the GPU kernel lands.
    pub fn snake_beta_f32(
        &self,
        x: &[f32],
        alpha: &[f32],
        beta: &[f32],
        channels: usize,
        time: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => vokra_ops::snake_beta_f32(x, alpha, beta, channels, time, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.snake_beta_f32(x, alpha, beta, channels, time, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "snake_beta_f32 has no wired CUDA NVRTC kernel; the Vocoder wave CUDA arm is \
                 deferred to the vast.ai owner track. Select BackendKind::Cpu (which delegates \
                 to vokra_ops::snake_beta_f32), or wait for the CUDA kernel — Vokra does not \
                 silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "snake_beta_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six Whisper \
                 hot ops only; the Vocoder wave GPU arms are deferred like Metal/CUDA/Vulkan). \
                 Select BackendKind::Cpu (which delegates to vokra_ops::snake_beta_f32) — Vokra \
                 does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// SineGen deterministic forward — the F0-driven multi-harmonic
    /// sinusoid source of HiFTNet-family vocoders (upstream CosyVoice
    /// `cosyvoice/hifigan/generator.py:200-214`, `SineGen.forward` under
    /// `NsfEntropy::Deterministic`), wired into the imperative `Compute`
    /// seam.
    ///
    /// Writes `f0.len() * (harmonic_num + 1)` FP32 samples to `out`,
    /// matching the deterministic path of
    /// [`vokra_ops::nsf::SineGen::forward`] bit-for-bit modulo the FP32
    /// transcendental gap. Output layout is `[T, H+1]` row-major
    /// (time-outer / harmonic-inner — upstream
    /// `sine_wavs.transpose(1, 2)`). Delegates on the CPU arm to
    /// [`vokra_ops::sinegen_deterministic_f32`].
    ///
    /// # Deterministic-only (no seeded path on the GPU)
    ///
    /// The `SineGen::forward` seeded mode carries a per-harmonic phase draw
    /// and a Gaussian noise stream that live on the host RNG; the GPU seam
    /// only exposes the deterministic slice. A caller that needs the
    /// seeded mode on the GPU would have to push a SplitMix64 state
    /// through device memory (a separate follow-up if a consumer needs it
    /// — none does today).
    ///
    /// # CPU + Metal wired; CUDA / Vulkan / WebGPU / CoreML / QNN return `UnsupportedOp`
    ///
    /// **Vocoder Metal wave (2026-08-14):** the Metal arm dispatches to
    /// [`vokra_backend_metal::MetalContext::sinegen_deterministic_f32`]
    /// (`vokra_sinegen_deterministic_f32` MSL kernel — one thread per
    /// harmonic walking the full time axis sequentially, same per-harmonic
    /// reduction order as the CPU forward, `atol ≤ 5e-4`). Every other GPU
    /// arm returns an explicit [`VokraError::UnsupportedOp`] — never a
    /// silent CPU fall back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`vokra_ops::sinegen_deterministic_f32`] raises (empty `f0`,
    ///   `samp_rate == 0`, wrong `out.len()`).
    /// - Metal arm: same host-side shape validation before dispatch, plus
    ///   any [`VokraError::BackendUnavailable`] from the underlying command
    ///   buffer.
    /// - CUDA / Vulkan / WebGPU / CoreML / QNN arms: explicit
    ///   [`VokraError::UnsupportedOp`] until the GPU kernel lands.
    pub fn sinegen_deterministic_f32(
        &self,
        f0: &[f32],
        samp_rate: u32,
        harmonic_num: u32,
        sine_amp: f32,
        voiced_threshold: f32,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => vokra_ops::sinegen_deterministic_f32(
                f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                out,
            ),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.sinegen_deterministic_f32(
                f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "sinegen_deterministic_f32 has no wired CUDA NVRTC kernel; the Vocoder wave \
                 CUDA arm is deferred to the vast.ai owner track. Select BackendKind::Cpu \
                 (which delegates to vokra_ops::sinegen_deterministic_f32), or wait for the \
                 CUDA kernel — Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "sinegen_deterministic_f32 has no wired WebGPU WGSL kernel (M4-01 covers the \
                 six Whisper hot ops only; the Vocoder wave GPU arms are deferred like \
                 Metal/CUDA/Vulkan). Select BackendKind::Cpu (which delegates to \
                 vokra_ops::sinegen_deterministic_f32) — Vokra does not silently run the op on \
                 the CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// Polyphase anti-aliased upsample — the multiply-add core of BigVGAN's
    /// `UpSample1d` (upstream `alias_free_activation.torch.act`, MIT) and
    /// every HiFTNet-family alias-free activation chain, wired into the
    /// imperative `Compute` seam.
    ///
    /// Writes `channels * time_in * ratio` FP32 samples to `out`, matching
    /// [`vokra_ops::anti_aliased_upsample_f32`] within `atol ≤ 1e-4` (the
    /// FMA-vs-non-FMA gap between MSL fast-math and the CPU strict-left-fold
    /// FIR accumulator). Delegates on the CPU arm to
    /// [`vokra_ops::anti_aliased_upsample_f32`].
    ///
    /// # Kaiser design lives on the host
    ///
    /// The audit's attribute list — `cutoff`, `filter_kernel`, `periodicity`
    /// — is Kaiser-window design metadata: given a target low-pass
    /// `cutoff` (in units of the Nyquist rate) and a `periodicity` / kernel
    /// length, a Kaiser window sinc produces the `filter_kernel` taps. The
    /// design step is **host-side, once per model load** (matches upstream
    /// BigVGAN's `UpSample1d.__init__`); this method consumes the already-
    /// designed taps and does the per-timestep multiply-add.
    ///
    /// # CPU + Metal wired; CUDA / Vulkan / WebGPU / CoreML / QNN return `UnsupportedOp`
    ///
    /// **Vocoder Metal wave (2026-08-14):** the Metal arm dispatches to
    /// [`vokra_backend_metal::MetalContext::anti_aliased_upsample_f32`]
    /// (`vokra_anti_aliased_upsample_f32` MSL kernel — 2-D dispatch, one
    /// thread per `(t_out, c)`, `atol ≤ 1e-4`). Every other GPU arm returns
    /// an explicit [`VokraError::UnsupportedOp`] — never a silent CPU fall
    /// back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`vokra_ops::anti_aliased_upsample_f32`] raises (`ratio == 0`,
    ///   empty `kernel`, wrong `x.len()` / `out.len()`).
    /// - Metal arm: same host-side shape validation before dispatch, plus
    ///   any [`VokraError::BackendUnavailable`] from the underlying command
    ///   buffer.
    /// - CUDA / Vulkan / WebGPU / CoreML / QNN arms: explicit
    ///   [`VokraError::UnsupportedOp`] until the GPU kernel lands.
    #[allow(clippy::too_many_arguments)] // intrinsic to the polyphase upsample shape
    pub fn anti_aliased_upsample_f32(
        &self,
        x: &[f32],
        kernel: &[f32],
        ratio: usize,
        channels: usize,
        time_in: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => {
                vokra_ops::anti_aliased_upsample_f32(x, kernel, ratio, channels, time_in, out)
            }
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => {
                ctx.anti_aliased_upsample_f32(x, kernel, ratio, channels, time_in, out)
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "anti_aliased_upsample_f32 has no wired CUDA NVRTC kernel; the Vocoder wave \
                 CUDA arm is deferred to the vast.ai owner track. Select BackendKind::Cpu \
                 (which delegates to vokra_ops::anti_aliased_upsample_f32), or wait for the \
                 CUDA kernel — Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "anti_aliased_upsample_f32 has no wired WebGPU WGSL kernel (M4-01 covers the \
                 six Whisper hot ops only; the Vocoder wave GPU arms are deferred like \
                 Metal/CUDA/Vulkan). Select BackendKind::Cpu (which delegates to \
                 vokra_ops::anti_aliased_upsample_f32) — Vokra does not silently run the op on \
                 the CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// SNAC 3/4-stage hierarchical residual VQ codec decode — the Vocoder
    /// wave WF5 op wired into the imperative `Compute` seam (upstream
    /// `hubertsiuzdak/snac`, MIT / Apache-2.0).
    ///
    /// Given one per-stage `codes` vector of `u32` codebook indices, the
    /// SNAC [`SnacConfig`] (holds per-stage temporal strides) and the
    /// [`SnacWeights`] bundle (3 factorized [`CodebookTable`]s and 3
    /// [`DacOutProj`]s), returns a fresh `[t_expanded × d_model]` row-major
    /// `Vec<f32>`: `out[t, :] = Σ_s (W_s @ codebooks[s].row(codes[s][t /
    /// strides[s]]) + b_s)` in FP32. Heap-returning for the same
    /// heterogeneous-signature reason as [`Compute::mimi_rvq_f32`] (chunk
    /// granularity, not per-token hot path).
    ///
    /// # Multi-scale distinction (why this is not `dac_rvq_f32`)
    ///
    /// Unlike Mimi / DAC where every quantizer shares the same time axis,
    /// SNAC's `k`th stage runs at frame rate `base / vq_strides[k]`. The
    /// per-stage `t_stage = t_out / strides[s]` lookup upsamples via
    /// `repeat_interleave(stride)` semantics (upstream
    /// `ResidualVectorQuantize.from_codes`, `snac/vq.py` L61-71). The
    /// factorized-projection shape (per-stage `WNConv1d(codebook_dim →
    /// d_model)` + bias) is shared with DAC, so this method reuses
    /// [`CodebookTable`] + [`DacOutProj`] rather than introducing new
    /// weight types.
    ///
    /// # CPU-only through this seam today (Metal wired; CUDA / Vulkan /
    /// WebGPU / CoreML / QNN return `UnsupportedOp`)
    ///
    /// **Vocoder Metal wave WF5 (2026-08-13):** the Metal arm dispatches
    /// to [`vokra_backend_metal::MetalContext::snac_decode_f32`]
    /// (`vokra_snac_decode_f32` MSL kernel — semantics equal to
    /// `SnacDecoder::decode` within the FP32 GEMV-scale bound
    /// `atol ≤ 5e-4`). The CPU arm builds a [`SnacDecoder`] and calls
    /// `decode` — bit-for-bit reproduces a direct
    /// `SnacDecoder::new(config, weights).decode(codes)` call. Every other
    /// GPU arm returns an explicit [`VokraError::UnsupportedOp`] — never a
    /// silent CPU fall back (FR-EX-08). The coverage gate on
    /// [`Compute::for_backend`] additionally rejects any model that lists
    /// [`HotOp::SnacDecode`] against those backends.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`SnacDecoder::new`] / [`SnacDecoder::decode`] raise (stride 0,
    ///   codebook / projection shape mismatch, cross-stage `T` mis-
    ///   alignment, `codes[i][t] >= codebook_size`; never a silent
    ///   0-clamp — FR-EX-08).
    /// - Metal arm: same host-side shape validation before dispatch, plus
    ///   any [`VokraError::BackendUnavailable`] from the underlying command
    ///   buffer.
    /// - CUDA / Vulkan / WebGPU / CoreML / QNN arms: explicit
    ///   [`VokraError::UnsupportedOp`] until the GPU kernel lands.
    pub fn snac_decode_f32(
        &self,
        codes: &[Vec<u32>],
        config: SnacConfig,
        codebooks: &[CodebookTable],
        out_projs: &[DacOutProj],
    ) -> Result<Vec<f32>> {
        match &self.be {
            Be::Cpu => {
                // Build a decoder inline and call decode. Bit-for-bit
                // reproduces a direct `SnacDecoder::new(config, weights).
                // decode(codes)` call — the CPU arm is a thin adapter, no
                // algorithmic changes. Chunk-granularity so building the
                // decoder per call is negligible next to the FP32 fold.
                let weights = SnacWeights {
                    codebooks: codebooks.to_vec(),
                    out_projs: out_projs.to_vec(),
                };
                let decoder = SnacDecoder::new(config, weights)?;
                decoder.decode(codes)
            }
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => {
                // Explicit shape + index validation on the host. The MSL
                // kernel guards `t >= t_expanded` and `d >= d_model` but has
                // no per-element bound check on `codes[..]`; silent OOB reads
                // inside the gather + factorized projection are the failure
                // mode we prevent by mirroring the CPU-arm shape checks in
                // `vokra_ops::snac_decode::SnacDecoder::new` /
                // `SnacDecoder::decode` (FR-EX-08 — never a silent GPU OOB or
                // CPU fall back).
                if !(1..=vokra_ops::MAX_SNAC_STAGES).contains(&config.n_stages) {
                    return Err(VokraError::InvalidArgument(format!(
                        "snac_decode_f32 metal: config.n_stages {} is outside 1..={}",
                        config.n_stages,
                        vokra_ops::MAX_SNAC_STAGES
                    )));
                }
                if codes.len() != config.n_stages
                    || codebooks.len() != config.n_stages
                    || out_projs.len() != config.n_stages
                {
                    return Err(VokraError::InvalidArgument(format!(
                        "snac_decode_f32 metal: n_stages={} but codes.len()={}, \
                         codebooks.len()={}, out_projs.len()={} (one entry per stage required)",
                        config.n_stages,
                        codes.len(),
                        codebooks.len(),
                        out_projs.len()
                    )));
                }
                let strides = config.vq_strides;
                for (s, &stride) in strides[..config.n_stages].iter().enumerate() {
                    if stride == 0 {
                        return Err(VokraError::InvalidArgument(format!(
                            "snac_decode_f32 metal: config.vq_strides[{s}] = 0 (stride 0 would \
                             divide the base frame rate by zero — FR-EX-08)"
                        )));
                    }
                }
                // Every stage must share codebook_size / codebook_dim /
                // d_model (SnacDecoder::new invariant). Derive the common
                // shape from stage 0 and validate the other two.
                let codebook_size = codebooks[0].codebook_size;
                let codebook_dim = codebooks[0].d_model;
                let d_model = out_projs[0].d_model;
                if codebook_size == 0 || codebook_dim == 0 || d_model == 0 {
                    return Err(VokraError::InvalidArgument(format!(
                        "snac_decode_f32 metal: axes must be > 0, got codebook_size={codebook_size} \
                         codebook_dim={codebook_dim} d_model={d_model}"
                    )));
                }
                for (s, cb) in codebooks.iter().enumerate() {
                    if cb.codebook_size != codebook_size || cb.d_model != codebook_dim {
                        return Err(VokraError::InvalidArgument(format!(
                            "snac_decode_f32 metal: codebooks[{s}] shape [{},{}] != [{},{}] \
                             (all stages must share the same codebook architecture)",
                            cb.codebook_size, cb.d_model, codebook_size, codebook_dim
                        )));
                    }
                }
                for (s, p) in out_projs.iter().enumerate() {
                    if p.d_model != d_model || p.codebook_dim != codebook_dim {
                        return Err(VokraError::InvalidArgument(format!(
                            "snac_decode_f32 metal: out_projs[{s}] shape [{},{}] != [{},{}] \
                             (all stages must project into the same d_model)",
                            p.d_model, p.codebook_dim, d_model, codebook_dim
                        )));
                    }
                }
                // Cross-stage T alignment: `codes[s].len() * strides[s]` must
                // equal the same T for every stage (SNAC's co-aligned base
                // frames invariant). This mirrors
                // `SnacDecoder::check_and_measure`.
                let mut common: Option<usize> = None;
                for (s, stage_codes) in codes.iter().enumerate() {
                    let expanded = stage_codes
                        .len()
                        .checked_mul(strides[s] as usize)
                        .ok_or_else(|| {
                            VokraError::InvalidArgument(format!(
                                "snac_decode_f32 metal: codes[{s}].len() ({}) * strides[{s}] ({}) \
                                 overflows usize",
                                stage_codes.len(),
                                strides[s]
                            ))
                        })?;
                    match common {
                        Some(prev) if prev != expanded => {
                            return Err(VokraError::InvalidArgument(format!(
                                "snac_decode_f32 metal: stage {s} expands to T={expanded}, but \
                                 earlier stages expand to T={prev} (codes[i].len() * strides[i] \
                                 must be the same for every stage — SNAC's multi-scale RVQ \
                                 requires co-aligned base frames)"
                            )));
                        }
                        Some(_) => {}
                        None => common = Some(expanded),
                    }
                }
                let t_expanded = common.unwrap_or(0);
                if t_expanded == 0 {
                    return Ok(Vec::new());
                }
                // Per-index bound check — the MSL kernel does NOT range-check
                // `codes[..]`. Cheap: O(Σ codes[s].len()) unpredictable
                // branches, dwarfed by the GPU dispatch.
                for (s, stage_codes) in codes.iter().enumerate() {
                    for (t_stage, &idx) in stage_codes.iter().enumerate() {
                        if (idx as usize) >= codebook_size {
                            return Err(VokraError::InvalidArgument(format!(
                                "snac_decode_f32 metal: codes[{s}][{t_stage}] = {idx} >= \
                                 codebook_size {codebook_size} (no silent clamp — FR-EX-08)"
                            )));
                        }
                    }
                }
                // Flatten codes / codebooks / projection weights / biases into
                // the row-major buffers the MSL kernel's stride math expects.
                // Chunk granularity — allocating a few Vecs here is negligible
                // next to the GPU dispatch (matches the heap-returning shape).
                let total_codes: usize = codes.iter().map(std::vec::Vec::len).sum();
                let mut codes_flat: Vec<u32> = Vec::with_capacity(total_codes);
                let mut stage_offsets: [u32; 4] = [0, 0, 0, 0];
                let mut running: usize = 0;
                for (s, stage_codes) in codes.iter().enumerate() {
                    stage_offsets[s] = u32::try_from(running).map_err(|_| {
                        VokraError::InvalidArgument(format!(
                            "snac_decode_f32 metal: stage_offsets[{s}] {running} overflows u32"
                        ))
                    })?;
                    codes_flat.extend_from_slice(stage_codes);
                    running = running.checked_add(stage_codes.len()).ok_or_else(|| {
                        VokraError::InvalidArgument(format!(
                            "snac_decode_f32 metal: cumulative codes length overflow at stage {s}"
                        ))
                    })?;
                }
                let mut codebooks_flat: Vec<f32> =
                    Vec::with_capacity(config.n_stages * codebook_size * codebook_dim);
                for cb in codebooks {
                    codebooks_flat.extend_from_slice(&cb.data);
                }
                let mut proj_weights_flat: Vec<f32> =
                    Vec::with_capacity(config.n_stages * d_model * codebook_dim);
                let mut proj_biases_flat: Vec<f32> = Vec::with_capacity(config.n_stages * d_model);
                for p in out_projs {
                    proj_weights_flat.extend_from_slice(&p.weight);
                    proj_biases_flat.extend_from_slice(&p.bias);
                }
                ctx.snac_decode_f32(
                    &codes_flat,
                    stage_offsets,
                    strides,
                    config.n_stages,
                    &codebooks_flat,
                    &proj_weights_flat,
                    &proj_biases_flat,
                    codebook_size,
                    codebook_dim,
                    d_model,
                    t_expanded,
                )
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "snac_decode_f32 has no wired CUDA NVRTC kernel; the Vocoder wave WF5 CUDA arm \
                 is deferred to the vast.ai owner track. Select BackendKind::Cpu (which builds a \
                 SnacDecoder inline and calls decode), or wait for the CUDA kernel — Vokra does \
                 not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "snac_decode_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six Whisper \
                 hot ops only; the Vocoder wave WF5 GPU arms are deferred like Metal/CUDA/Vulkan). \
                 Select BackendKind::Cpu — Vokra does not silently run the op on the CPU \
                 (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// Denoise spectral-gate primitive — element-wise complex × real gain
    /// multiply for a `[n_frames, n_bins]` row-major FP32 complex spectrogram
    /// (Vocoder Metal wave WF5, 2026-08-13). The "spectral gate + phase
    /// preservation" step every mask-based denoiser (DFN3, GTCRN, RNNoise)
    /// ends in, wired into the imperative `Compute` seam so the mask apply
    /// can move to a GPU dispatch while the rest of the front-end runs on
    /// the host.
    ///
    /// Applies `out_re[t, f] = spec_re[t, f] · gain[t, f]` and
    /// `out_im[t, f] = spec_im[t, f] · gain[t, f]` for every `(t, f)`
    /// position. Delegates on the CPU arm to
    /// [`vokra_ops::denoise_apply_mask_f32`] (bit-identical to the
    /// [`vokra_ops::denoise::DenoiseModel::enhance_inner`] output-stage
    /// inline loop when the caller pre-expands the ERB mask through
    /// `erb_inv_fb`).
    ///
    /// # Phase preservation, no silent numeric drift
    ///
    /// `re · g` and `im · g` with the same real scalar `g` leaves phase
    /// `atan2(im, re)` bit-identically unchanged; only magnitude
    /// `sqrt(re² + im²)` is scaled. Multiplication is IEEE-754
    /// correctly-rounded; there is no reduction, no transcendental, no FMA
    /// opportunity in a single `re * g`, so CPU and GPU produce
    /// **bit-for-bit identical** outputs on every finite input. The parity
    /// harness (`tests/denoise_metal_bit_identical.rs`) still enforces the
    /// sibling `atol ≤ 5e-4` codec-family bound to keep a discriminating
    /// negative control, but the achieved max |Δ| = 0 is logged.
    ///
    /// # Not the whole DenoiseModel
    ///
    /// The full DFN3 network (STFT → ERB features → DfNet → mask +
    /// deep-filter → iSTFT) lives in
    /// [`vokra_ops::denoise::DenoiseModel::enhance`], which still runs the
    /// fused inline loop for the CPU-only path it has always taken. This
    /// seam is the **primitive** for the mask-apply step alone, so a
    /// per-freq-per-time mask denoiser (GTCRN / RNNoise) or a future GPU
    /// port of DFN3's output stage can dispatch through it.
    ///
    /// # CPU-only through this seam today (Metal wired; CUDA / Vulkan /
    /// WebGPU / CoreML / QNN return `UnsupportedOp`)
    ///
    /// **Vocoder Metal wave WF5 (2026-08-13):** the Metal arm dispatches to
    /// [`vokra_backend_metal::MetalContext::denoise_apply_mask_f32`]
    /// (`vokra_denoise_apply_mask_f32` MSL kernel). Every other GPU arm
    /// returns an explicit [`VokraError::UnsupportedOp`] until the
    /// corresponding kernel lands — never a silent CPU fall back
    /// (FR-EX-08). The coverage gate on [`Compute::for_backend`]
    /// additionally rejects any model that lists [`HotOp::DenoiseApplyMask`]
    /// against those backends.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`vokra_ops::denoise_apply_mask_f32`] raises (wrong `spec_re.len()`
    ///   / `spec_im.len()` / `gain.len()` / `out_re.len()` / `out_im.len()`,
    ///   or `n_frames * n_bins` overflow — never a silent shape clamp,
    ///   FR-EX-08).
    /// - Metal arm: same host-side shape validation before dispatch, plus
    ///   any [`VokraError::BackendUnavailable`] from the underlying command
    ///   buffer.
    /// - CUDA / Vulkan / WebGPU / CoreML / QNN arms: explicit
    ///   [`VokraError::UnsupportedOp`] until the GPU kernel lands.
    #[allow(clippy::too_many_arguments)] // intrinsic to the two-output complex-multiply shape
    pub fn denoise_apply_mask_f32(
        &self,
        spec_re: &[f32],
        spec_im: &[f32],
        gain: &[f32],
        n_frames: usize,
        n_bins: usize,
        out_re: &mut [f32],
        out_im: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => vokra_ops::denoise_apply_mask_f32(
                spec_re, spec_im, gain, n_frames, n_bins, out_re, out_im,
            ),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => {
                ctx.denoise_apply_mask_f32(spec_re, spec_im, gain, n_frames, n_bins, out_re, out_im)
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "denoise_apply_mask_f32 has no wired CUDA NVRTC kernel; the Vocoder wave WF5 CUDA \
                 arm is deferred to the vast.ai owner track. Select BackendKind::Cpu (which \
                 delegates to vokra_ops::denoise_apply_mask_f32), or wait for the CUDA kernel — \
                 Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "denoise_apply_mask_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six \
                 Whisper hot ops only; the Vocoder wave WF5 GPU arms are deferred like \
                 Metal/CUDA/Vulkan). Select BackendKind::Cpu (which delegates to \
                 vokra_ops::denoise_apply_mask_f32) — Vokra does not silently run the op on the \
                 CPU (FR-EX-08)."
                    .to_owned(),
            )),
        }
    }

    /// Qwen3-TTS-Codec RVQ decode — the Vocoder wave WF5 op consumed by every
    /// released Qwen3-TTS-12Hz voice (`Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,
    /// CustomVoice,VoiceDesign}`, Apache-2.0), wired into the imperative
    /// `Compute` seam (mirror of [`Compute::mimi_rvq_f32`], plus the semantic
    /// vs acoustic per-quantizer vocab split).
    ///
    /// Given per-quantizer `u32` code streams (one `Vec<u32>` per quantizer),
    /// one [`CodebookTable`] per quantizer (semantic first, then acoustic),
    /// and the codec [`Qwen3TtsCodecConfig`], returns a fresh
    /// `[time × codebook_dim]` row-major `Vec<f32>`:
    /// `out[t, :] = Σ_q tables[q].row(codes[q][t])` in FP32 (see
    /// [`vokra_ops::qwen3_tts_codec_decode`]). Heap-returning for the same
    /// heterogeneous-signature reason as `mimi_rvq_f32` (chunk granularity,
    /// not per-token hot path).
    ///
    /// # Semantic vs acoustic vocab split (why this is not `mimi_rvq_f32`)
    ///
    /// Qwen3-TTS-Codec is a **hybrid semantic + acoustic RVQ**: the first
    /// `config.num_semantic_quantizers` quantizers use a **larger**
    /// `config.semantic_codebook_size` vocab (canonical 4096) than the
    /// remaining acoustic quantizers use `config.codebook_size` (canonical
    /// 2048). Every codebook still emits the same `config.codebook_dim`-wide
    /// row (canonical 512). Mimi's uniform `MimiRvqAttrs` cannot express this
    /// asymmetry without silently clamping the semantic index (which would
    /// violate FR-EX-08 / the CPU op's "no silent clamp" rule); this method
    /// takes the CPU op's config verbatim.
    ///
    /// # CPU-only through this seam today (Metal wired; CUDA / Vulkan /
    /// WebGPU / CoreML / QNN return `UnsupportedOp`)
    ///
    /// **Vocoder Metal wave WF5 (2026-08-13):** the Metal arm dispatches to
    /// [`vokra_backend_metal::MetalContext::qwen3_tts_codec_decode_f32`]
    /// (`vokra_qwen3_tts_codec_decode_f32` MSL kernel — semantics equal to
    /// `qwen3_tts_codec_decode` within the FP32 GEMV-scale bound
    /// `atol ≤ 5e-4`). The CPU arm delegates verbatim to
    /// [`vokra_ops::qwen3_tts_codec_decode`] (bit-for-bit vs a direct kernel
    /// call). Every other GPU arm returns an explicit
    /// [`VokraError::UnsupportedOp`] — never a silent CPU fall back
    /// (FR-EX-08). The coverage gate on [`Compute::for_backend`] additionally
    /// rejects any model that lists [`HotOp::Qwen3TtsCodec`] against those
    /// backends.
    ///
    /// # Errors
    ///
    /// - CPU arm: propagates the [`VokraError::InvalidArgument`] variants
    ///   [`vokra_ops::qwen3_tts_codec_decode`] raises (config axis
    ///   validation, wrong number of codebook tables, per-quantizer shape
    ///   mismatch — semantic entries must use `semantic_codebook_size` and
    ///   acoustic entries must use `codebook_size` — wrong number of code
    ///   streams, inner-length mismatch, per-index out-of-range; never a
    ///   silent clamp — FR-EX-08).
    /// - Metal arm: same host-side shape validation before dispatch, plus
    ///   any [`VokraError::BackendUnavailable`] from the underlying command
    ///   buffer.
    /// - CUDA / Vulkan / WebGPU / CoreML / QNN arms: explicit
    ///   [`VokraError::UnsupportedOp`] until the GPU kernel lands.
    pub fn qwen3_tts_codec_f32(
        &self,
        codes: &[Vec<u32>],
        codebook_tables: &[CodebookTable],
        config: &Qwen3TtsCodecConfig,
    ) -> Result<Vec<f32>> {
        match &self.be {
            Be::Cpu => qwen3_tts_codec_decode(codes, codebook_tables, config),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => {
                // Explicit shape + index validation on the host. The MSL
                // kernel guards `t >= time` and `delem >= codebook_dim` but
                // has no per-element bound check on `codes[..]`; silent OOB
                // reads inside the semantic / acoustic gather are the failure
                // mode we prevent by mirroring the CPU-arm shape checks in
                // `vokra_ops::qwen3_tts_codec::check_weights_shape` /
                // `check_codes_shape` (FR-EX-08 — never a silent GPU OOB or
                // CPU fall back).
                if config.num_quantizers == 0
                    || config.semantic_codebook_size == 0
                    || config.codebook_size == 0
                    || config.codebook_dim == 0
                {
                    return Err(VokraError::InvalidArgument(format!(
                        "qwen3_tts_codec_f32 metal: config axes must be > 0, got \
                         num_quantizers={} semantic_codebook_size={} codebook_size={} \
                         codebook_dim={}",
                        config.num_quantizers,
                        config.semantic_codebook_size,
                        config.codebook_size,
                        config.codebook_dim,
                    )));
                }
                if config.num_semantic_quantizers > config.num_quantizers {
                    return Err(VokraError::InvalidArgument(format!(
                        "qwen3_tts_codec_f32 metal: config.num_semantic_quantizers {} > \
                         num_quantizers {}",
                        config.num_semantic_quantizers, config.num_quantizers,
                    )));
                }
                if codebook_tables.len() != config.num_quantizers {
                    return Err(VokraError::InvalidArgument(format!(
                        "qwen3_tts_codec_f32 metal: codebook_tables.len() {} != \
                         config.num_quantizers {}",
                        codebook_tables.len(),
                        config.num_quantizers
                    )));
                }
                // Per-quantizer shape check — semantic entries carry the
                // semantic vocab, acoustic entries carry the acoustic vocab,
                // every table emits `codebook_dim`-wide rows. Mirrors
                // `vokra_ops::qwen3_tts_codec::check_weights_shape`.
                for (q, tbl) in codebook_tables.iter().enumerate() {
                    let (expected_vocab, role) = if q < config.num_semantic_quantizers {
                        (config.semantic_codebook_size, "semantic")
                    } else {
                        (config.codebook_size, "acoustic")
                    };
                    if tbl.codebook_size != expected_vocab {
                        return Err(VokraError::InvalidArgument(format!(
                            "qwen3_tts_codec_f32 metal: codebook_tables[{q}] ({role}) \
                             codebook_size {} != expected {expected_vocab}",
                            tbl.codebook_size,
                        )));
                    }
                    if tbl.d_model != config.codebook_dim {
                        return Err(VokraError::InvalidArgument(format!(
                            "qwen3_tts_codec_f32 metal: codebook_tables[{q}] ({role}) d_model {} \
                             != config.codebook_dim {}",
                            tbl.d_model, config.codebook_dim,
                        )));
                    }
                }
                // Codes: `codes.len() == num_quantizers`; every inner stream
                // shares the same time axis. Mirrors
                // `vokra_ops::qwen3_tts_codec::check_codes_shape`.
                if codes.len() != config.num_quantizers {
                    return Err(VokraError::InvalidArgument(format!(
                        "qwen3_tts_codec_f32 metal: codes.len() {} != config.num_quantizers {}",
                        codes.len(),
                        config.num_quantizers
                    )));
                }
                let time = if codes.is_empty() { 0 } else { codes[0].len() };
                for (q, stream) in codes.iter().enumerate().skip(1) {
                    if stream.len() != time {
                        return Err(VokraError::InvalidArgument(format!(
                            "qwen3_tts_codec_f32 metal: codes[{q}].len() {} != codes[0].len() \
                             {time} (per-quantizer streams must share the same time axis)",
                            stream.len(),
                        )));
                    }
                }
                if time == 0 {
                    return Ok(Vec::new());
                }
                // Per-index bound check — semantic quantizers must obey the
                // larger `semantic_codebook_size`; acoustic quantizers must
                // obey `codebook_size`. The MSL kernel does NOT range-check
                // `codes[..]`, so a stray index would be a silent OOB gather
                // (FR-EX-08). Cheap: O(time * num_quantizers) unpredictable
                // branches, dwarfed by the FP32 fold on the GPU.
                for (q, stream) in codes.iter().enumerate() {
                    let vocab = if q < config.num_semantic_quantizers {
                        config.semantic_codebook_size
                    } else {
                        config.codebook_size
                    };
                    for (t, &idx) in stream.iter().enumerate() {
                        if (idx as usize) >= vocab {
                            return Err(VokraError::InvalidArgument(format!(
                                "qwen3_tts_codec_f32 metal: codes[{q}][{t}] = {idx} >= per-\
                                 quantizer vocab {vocab} (no silent clamp — FR-EX-08)"
                            )));
                        }
                    }
                }
                // Flatten [num_quantizers][time] → [time × num_quantizers]
                // row-major (the MSL kernel's `codes[t * num_quantizers + q]`
                // stride math). Chunk granularity — allocating one Vec here
                // is negligible next to the GPU dispatch.
                let mut codes_flat: Vec<u32> = Vec::with_capacity(time * config.num_quantizers);
                for t in 0..time {
                    for stream in codes {
                        codes_flat.push(stream[t]);
                    }
                }
                // Split codebook_tables into semantic + acoustic flat buffers
                // matching the kernel's two-buffer layout.
                let mut semantic_tables_flat: Vec<f32> = Vec::with_capacity(
                    config.num_semantic_quantizers
                        * config.semantic_codebook_size
                        * config.codebook_dim,
                );
                let num_acoustic = config.num_quantizers - config.num_semantic_quantizers;
                let mut acoustic_tables_flat: Vec<f32> =
                    Vec::with_capacity(num_acoustic * config.codebook_size * config.codebook_dim);
                for (q, tbl) in codebook_tables.iter().enumerate() {
                    if q < config.num_semantic_quantizers {
                        semantic_tables_flat.extend_from_slice(&tbl.data);
                    } else {
                        acoustic_tables_flat.extend_from_slice(&tbl.data);
                    }
                }
                ctx.qwen3_tts_codec_decode_f32(
                    &codes_flat,
                    &semantic_tables_flat,
                    &acoustic_tables_flat,
                    config.num_quantizers,
                    config.num_semantic_quantizers,
                    config.semantic_codebook_size,
                    config.codebook_size,
                    config.codebook_dim,
                    time,
                )
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "qwen3_tts_codec_f32 has no wired CUDA NVRTC kernel; the Vocoder wave WF5 CUDA \
                 arm is deferred to the vast.ai owner track. Select BackendKind::Cpu (which \
                 delegates to vokra_ops::qwen3_tts_codec_decode), or wait for the CUDA kernel — \
                 Vokra does not silently run the op on the CPU (FR-EX-08)."
                    .to_owned(),
            )),
            #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
            Be::WebGpu(_) => Err(VokraError::UnsupportedOp(
                "qwen3_tts_codec_f32 has no wired WebGPU WGSL kernel (M4-01 covers the six \
                 Whisper hot ops only; the Vocoder wave WF5 GPU arms are deferred like \
                 Metal/CUDA/Vulkan). Select BackendKind::Cpu (which delegates to \
                 vokra_ops::qwen3_tts_codec_decode) — Vokra does not silently run the op on the \
                 CPU (FR-EX-08)."
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

    /// On a Metal build both FSQ-family Metal arms are wired
    /// (`wavtokenizer_vq_f32` and `xcodec2_fsq_f32`, commit `a7a05e8` MSL
    /// kernels `vokra_wavtokenizer_vq_gather_f32` and
    /// `vokra_xcodec2_fsq_decode_f32`) — the trivial (1,1,1) shapes must
    /// therefore return `Ok`. This test confirms both arms stay in
    /// lock-step with their [`HotOp::covered_by_metal`] flags (FR-EX-08,
    /// mirror of the RVQ tests). Failing HERE (rather than at `Err`) means
    /// the sibling wave's Metal arm regressed to a stale `UnsupportedOp`.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_fsq_family_arms_track_coverage_no_silent_fallback() {
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
        let wt_out = compute
            .wavtokenizer_vq_f32(&[0u32], 1, &table, &wt_attrs)
            .expect(
                "WavTokenizerVq Metal arm is wired (Vocoder wave WF2 sibling, 2026-08-13) — \
                 must succeed",
            );
        assert_eq!(wt_out.len(), 1);

        let fsq_attrs = Xcodec2FsqAttrs {
            levels: vec![2],
            d_model: 1,
        };
        let fsq_out = compute
            .xcodec2_fsq_f32(&[0u32], 1, None, &fsq_attrs)
            .expect(
                "Xcodec2Fsq Metal arm is wired (Vocoder wave WF2 sibling, 2026-08-13) — \
                 must succeed",
            );
        assert_eq!(fsq_out.len(), 1);
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

    /// On a Metal build both DAC and EnCodec RVQ arms dispatch real kernels.
    /// EnCodec reuses Mimi's identical shape-generic gather/fold kernel; the
    /// standalone-weight exclusion is a publication rule, not a reason to
    /// leave an authenticated MusicGen composite's learned gather on CPU.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_dac_and_encodec_rvq_arms_track_coverage_no_silent_fallback() {
        let compute = match Compute::for_backend(BackendKind::Metal, &[]) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; dac/encodec rvq Metal arm test skipped");
                return;
            }
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        };
        // DAC is Metal-covered (Vocoder wave WF2 sibling, 2026-08-13). The
        // trivial (1,1,1) shape yields `Ok(vec![0.0])` on the CPU arm, so
        // the Metal arm must succeed (bit-identical on this shape — nothing
        // to re-associate). Failing HERE (rather than at `Err`) means the
        // sibling wave's Metal arm regressed to a stale `UnsupportedOp`.
        let dac_attrs = DacRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            codebook_dim: 1,
            d_model: 1,
        };
        let dac_tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        let dac_projs = vec![DacOutProj::new(1, 1, vec![0.0], vec![0.0]).unwrap()];
        let dac_out = compute
            .dac_rvq_f32(&[0u32], 1, &dac_tables, &dac_projs, &dac_attrs)
            .expect("DAC Metal arm is wired (Vocoder wave WF2, 2026-08-13) — must succeed");
        assert_eq!(dac_out.len(), 1);

        let enc_attrs = EncodecRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            d_model: 1,
        };
        let enc_tables = vec![CodebookTable::new(1, 1, vec![0.0]).unwrap()];
        let enc_out = compute
            .encodec_rvq_f32(&[0u32], 1, &enc_tables, &enc_attrs)
            .expect("EnCodec Metal arm reuses the wired Mimi gather/fold kernel");
        assert_eq!(enc_out, vec![0.0]);
        assert!(matches!(
            compute.encodec_rvq_f32(&[1u32], 1, &enc_tables, &enc_attrs),
            Err(VokraError::InvalidArgument(_)),
        ));

        let parity_attrs = EncodecRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            d_model: 2,
        };
        let parity_tables = vec![
            CodebookTable::new(3, 2, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]).unwrap(),
            CodebookTable::new(3, 2, vec![3.0, 3.5, 4.0, 4.5, 5.0, 5.5]).unwrap(),
        ];
        let parity_codes = [0, 2, 1, 1, 2, 0];
        let cpu_out = encodec_rvq_decode(&parity_codes, 3, &parity_tables, &parity_attrs).unwrap();
        let metal_out = compute
            .encodec_rvq_f32(&parity_codes, 3, &parity_tables, &parity_attrs)
            .expect("nontrivial EnCodec gather/fold must run on Metal");
        assert_eq!(metal_out, cpu_out);
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

    /// M3-06 T14 (2026-08-13): the Metal arm of `mimi_rvq_f32` now dispatches
    /// to `vokra_mimi_rvq_gather_fold_f32` (a real MSL kernel — no
    /// `UnsupportedOp`). This smoke verifies two things at the seam level:
    ///
    /// 1. **A trivial (1,1,1) shape returns the expected FP32 fold** (bit-
    ///    identical to the CPU arm on this shape — there is nothing to
    ///    re-associate over `n_codebooks = 1`).
    /// 2. **FR-EX-08 host-side validation still fires**: an out-of-range code
    ///    is rejected with `InvalidArgument` before dispatch, so the MSL
    ///    kernel never silently reads OOB from the `tables` buffer.
    ///
    /// The full CPU-vs-Metal parity test lives in
    /// `tests/mimi_rvq_metal_bit_identical.rs`.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_mimi_rvq_arm_runs_kernel_and_rejects_oob_index() {
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

        // (1) Trivial (1,1,1) shape: one codebook, one row [3.5], time = 1,
        // code = 0 → out = [3.5]. n_codebooks = 1 means no re-association is
        // possible, so the CPU and Metal folds are bit-identical.
        let attrs = MimiRvqAttrs {
            n_codebooks: 1,
            codebook_size: 1,
            d_model: 1,
        };
        let tables = vec![CodebookTable::new(1, 1, vec![3.5]).unwrap()];
        let out = compute
            .mimi_rvq_f32(&[0u32], 1, &tables, &attrs)
            .expect("Metal arm of mimi_rvq_f32 must run cleanly post M3-06 T14");
        assert_eq!(out, vec![3.5_f32]);

        // (2) FR-EX-08 host-side OOB index: codebook_size = 1, so idx = 1 is
        // OOB. The Metal arm's host-side per-index check must reject with
        // `InvalidArgument` — never a silent GPU OOB read.
        let oob_err = compute
            .mimi_rvq_f32(&[1u32], 1, &tables, &attrs)
            .expect_err("OOB code index must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(oob_err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for OOB code, got {oob_err:?}",
        );

        // (3) FR-EX-08 shape mismatch: codes.len() != time * n_codebooks.
        let shape_err = compute
            .mimi_rvq_f32(&[0u32, 0u32], 1, &tables, &attrs)
            .expect_err("codes.len() mismatch must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(shape_err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for shape mismatch, got {shape_err:?}",
        );
    }

    /// Off the Metal build (or off Apple), `for_backend(Metal, [MimiRvq])`
    /// is a `BackendUnavailable` at the coverage layer (Metal is not
    /// compiled in), so a request for the Metal-covered MimiRvq path on a
    /// non-Metal build still fails explicitly — never a silent CPU substitute
    /// (FR-EX-08). Post M3-06 T14 (2026-08-13) the Metal build side is a
    /// device probe (see `metal_mimi_rvq_arm_runs_kernel_and_rejects_oob_index`);
    /// off the Metal build this belt-and-braces `BackendUnavailable` holds.
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
    fn cpu_rms_norm_matches_gamma_only_reference() {
        let input = [1.0f32, -2.0, 3.0, 4.0, -1.0, 0.5];
        let gamma = [0.5f32, 1.5, -2.0];
        let mut actual = [0.0f32; 6];
        Compute::cpu()
            .rms_norm_f32(&input, &mut actual, 2, 3, &gamma, 1.0e-6)
            .expect("cpu RMSNorm");
        for row in 0..2 {
            let src = &input[row * 3..row * 3 + 3];
            let inverse_rms =
                1.0 / ((src.iter().map(|x| x * x).sum::<f32>() / 3.0) + 1.0e-6).sqrt();
            for col in 0..3 {
                let expected = src[col] * inverse_rms * gamma[col];
                assert!((actual[row * 3 + col] - expected).abs() <= f32::EPSILON);
            }
        }
    }

    #[test]
    fn cpu_rms_norm_rejects_degenerate_axes_and_epsilon() {
        let cpu = Compute::cpu();
        assert!(matches!(
            cpu.rms_norm_f32(&[], &mut [], 0, 3, &[1.0; 3], 1.0e-6),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            cpu.rms_norm_f32(&[1.0], &mut [0.0], 1, 1, &[1.0], 0.0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn cpu_scale_norm_matches_released_equation() {
        let input = [1.0f32, -2.0, 3.0, 4.0, -1.0, 0.5];
        let gain = 0.75f32;
        let eps = 1.0e-5f32;
        let mut actual = [0.0f32; 6];
        Compute::cpu()
            .scale_norm_f32(&input, &mut actual, 2, 3, gain, eps)
            .expect("cpu ScaleNorm");
        let dimension_scale = (3.0f64).sqrt().recip() as f32;
        for row in 0..2 {
            let source = &input[row * 3..row * 3 + 3];
            let squared_norm = source.iter().map(|value| value * value).sum::<f32>();
            let denominator = (squared_norm.sqrt() * dimension_scale).max(eps);
            for col in 0..3 {
                let expected = source[col] / denominator * gain;
                assert_eq!(actual[row * 3 + col], expected);
            }
        }
    }

    #[test]
    fn cpu_scale_norm_rejects_invalid_epsilon() {
        assert!(matches!(
            Compute::cpu().scale_norm_f32(&[1.0], &mut [0.0], 1, 1, 1.0, 0.0),
            Err(VokraError::InvalidArgument(_))
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
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::Gelu,
            HotOp::GeluNew,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::Conv1d,
            HotOp::GroupedConv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
            HotOp::GroupFsq,
            HotOp::CausalHifiGan,
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
        // kernel). MimiRvq is covered as of M3-06 T14 (2026-08-13, MSL kernel
        // `vokra_mimi_rvq_gather_fold_f32`); DacRvq is covered as of M4-04 WF2
        // (2026-08-13, MSL kernel `vokra_dac_rvq_gather_project_fold_f32`);
        // the FSQ family (WavTokenizerVq / Xcodec2Fsq) is covered as of M4-16
        // WF2 (2026-08-13, MSL kernels `vokra_wavtokenizer_vq_gather_f32` /
        // `vokra_xcodec2_fsq_decode_f32`); SnakeActivation is covered as of
        // the Vocoder Metal wave WF2 (2026-08-13, MSL kernel
        // `vokra_snake_activation_f32`); SnacDecode and DenoiseApplyMask are
        // covered as of the Vocoder Metal wave WF5 (2026-08-13, MSL kernels
        // `vokra_snac_decode_f32` / `vokra_denoise_apply_mask_f32`).
        // EncodecRvq reuses Mimi's identical shape-generic gather/fold kernel
        // as of the AudioCraft waveform-decode wave (2026-08-26).
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::Gelu,
            HotOp::GeluNew,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::Conv1d,
            HotOp::GroupedConv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            // Vocoder Metal wave common vocoder primitives (2026-08-14):
            // SnakeBeta / SineGen deterministic / anti-aliased upsample —
            // all three have wired MSL kernels via the sibling snake_activation
            // path (`vokra_snake_beta_f32` / `vokra_sinegen_deterministic_f32`
            // / `vokra_anti_aliased_upsample_f32`).
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
        ] {
            assert!(
                op.covered_by_metal(),
                "{op:?} unexpectedly NOT Metal-covered"
            );
        }

        // The two NanoCodec hot ops remain deliberately CPU-only. Keep these
        // flags in lock-step with the
        // corresponding model required-op registries and Compute method arms.
        for op in [HotOp::GroupFsq, HotOp::CausalHifiGan] {
            assert!(
                !op.covered_by_metal(),
                "{op:?} unexpectedly Metal-covered — its GPU kernel is deferred; if it has \
                 just landed, flip `HotOp::covered_by_metal` and update this test.",
            );
            let Err(VokraError::UnsupportedOp(msg)) =
                Compute::for_backend(BackendKind::Metal, &[op])
            else {
                panic!("Metal must reject CPU-only {op:?} with UnsupportedOp");
            };
            assert!(
                msg.contains(&format!("{op:?}")),
                "Metal coverage error must identify the required CPU-only op: {msg}",
            );
        }
        // A request that lists MimiRvq is now a covered request post M3-06
        // T14: it either builds (device present) or reports an explicit
        // device unavailability (no silent CPU fall back, FR-EX-08).
        match Compute::for_backend(BackendKind::Metal, &[HotOp::MimiRvq]) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; MimiRvq covered path is device-gated");
            }
            Err(e) => panic!("unexpected error for a Metal-covered MimiRvq request: {e}"),
        }
        // A request that lists DacRvq is now a covered request post M4-04
        // WF2: same posture as MimiRvq above.
        match Compute::for_backend(BackendKind::Metal, &[HotOp::DacRvq]) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; DacRvq covered path is device-gated");
            }
            Err(e) => panic!("unexpected error for a Metal-covered DacRvq request: {e}"),
        }
        // The FSQ family (M4-16 WF2, 2026-08-13) is now a covered request:
        // same device-gated posture as MimiRvq / DacRvq above.
        for op in [HotOp::WavTokenizerVq, HotOp::Xcodec2Fsq] {
            match Compute::for_backend(BackendKind::Metal, &[op]) {
                Ok(c) => assert_eq!(c.backend_name(), "metal"),
                Err(VokraError::BackendUnavailable(_)) => {
                    eprintln!("no Metal device; {op:?} covered path is device-gated");
                }
                Err(e) => panic!("unexpected error for a Metal-covered {op:?} request: {e}"),
            }
        }
        // SnakeActivation (Vocoder Metal wave WF2, 2026-08-13) is now a
        // covered request: same device-gated posture as MimiRvq / DacRvq /
        // the FSQ family above.
        match Compute::for_backend(BackendKind::Metal, &[HotOp::SnakeActivation]) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; SnakeActivation covered path is device-gated");
            }
            Err(e) => panic!("unexpected error for a Metal-covered SnakeActivation request: {e}"),
        }
        // SnacDecode (Vocoder Metal wave WF5, 2026-08-13) is now a covered
        // request: same device-gated posture as the sibling codec ops above.
        match Compute::for_backend(BackendKind::Metal, &[HotOp::SnacDecode]) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; SnacDecode covered path is device-gated");
            }
            Err(e) => panic!("unexpected error for a Metal-covered SnacDecode request: {e}"),
        }
        // DenoiseApplyMask (Vocoder Metal wave WF5, 2026-08-13) is now a
        // covered request: same device-gated posture as the sibling codec
        // ops above.
        match Compute::for_backend(BackendKind::Metal, &[HotOp::DenoiseApplyMask]) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; DenoiseApplyMask covered path is device-gated");
            }
            Err(e) => panic!("unexpected error for a Metal-covered DenoiseApplyMask request: {e}"),
        }
        // Qwen3TtsCodec (Vocoder Metal wave WF5, 2026-08-13) is now a covered
        // request: same device-gated posture as the sibling codec ops above.
        match Compute::for_backend(BackendKind::Metal, &[HotOp::Qwen3TtsCodec]) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; Qwen3TtsCodec covered path is device-gated");
            }
            Err(e) => panic!("unexpected error for a Metal-covered Qwen3TtsCodec request: {e}"),
        }
        // Vocoder Metal wave common vocoder primitives (2026-08-14): each
        // is a covered request with the same device-gated posture as the
        // sibling codec / snake_activation ops above.
        for op in [
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
        ] {
            match Compute::for_backend(BackendKind::Metal, &[op]) {
                Ok(c) => assert_eq!(c.backend_name(), "metal"),
                Err(VokraError::BackendUnavailable(_)) => {
                    eprintln!("no Metal device; {op:?} covered path is device-gated");
                }
                Err(e) => panic!("unexpected error for a Metal-covered {op:?} request: {e}"),
            }
        }

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
        // Same deferred posture for the M4-04 RVQ siblings, the M4-16 FSQ
        // family, the Vocoder wave WF2 SnakeActivation, and the Vocoder wave
        // WF5 SnacDecode / DenoiseApplyMask (lock-step with the CUDA arms of
        // `dac_rvq_f32` / `encodec_rvq_f32` / `wavtokenizer_vq_f32` /
        // `xcodec2_fsq_f32` / `snake_activation_f32` / `snac_decode_f32` /
        // `denoise_apply_mask_f32`).
        for op in [
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::GeluNew,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            // Vocoder Metal wave common vocoder primitives (2026-08-14) — CUDA
            // NVRTC kernels are deferred to the vast.ai owner track (same
            // posture as the sibling snake_activation / codec CUDA arms).
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
            HotOp::GroupedConv1d,
            HotOp::GroupFsq,
            HotOp::CausalHifiGan,
        ] {
            assert!(
                !op.covered_by_cuda(),
                "{op:?} unexpectedly CUDA-covered — the M4-04/M4-16/Vocoder GPU kernels are \
                 deferred; if one has just landed, flip `HotOp::covered_by_cuda` and update this \
                 test.",
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
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::Gelu,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::Conv1d,
            HotOp::GroupedConv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            // Vocoder Metal wave common vocoder primitives (2026-08-14).
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
            HotOp::GroupFsq,
            HotOp::CausalHifiGan,
        ] {
            assert!(
                !op.covered_by_vulkan(),
                "{op:?} unexpectedly covered: `HotOp::covered_by_vulkan` says `true` but the \
                 `Compute` methods have no `Be::Vulkan` arm to delegate into. Flip the coverage \
                 flag and the seam arm together, then shrink this test's negative-assertion set \
                 accordingly.",
            );
        }

        // --- Anti-rot guard: the diagnostic must not blame the shaders.
        //
        // All 12 `.spv` blobs (M4-13-T16) and the typed
        // `VulkanBackend::*_f32` dispatch entry points (M4-13-T03〜T08)
        // landed long ago; an earlier revision of this arm still told the
        // reader to "wait for the SPIR-V kernels", which sends them off to
        // recompile committed artefacts. Assert the stale phrasing is ABSENT
        // as well as asserting the live blocker is named — omission alone is
        // not enforceable (mirror of the `beat_this` guard).
        let Err(VokraError::UnsupportedOp(msg)) =
            Compute::for_backend(BackendKind::Vulkan, &[HotOp::Gemm])
        else {
            panic!("Vulkan must reject an uncovered required set with UnsupportedOp");
        };
        assert!(
            !msg.contains("ships no .spv"),
            "stale claim — kernels/precompiled/ ships all 12 blobs: {msg}"
        );
        assert!(
            !msg.contains("wait for the SPIR-V kernels"),
            "stale instruction — the SPIR-V kernels have landed: {msg}"
        );
        assert!(
            msg.contains("Be::Vulkan"),
            "must name the seam arm that is actually missing: {msg}"
        );
        // Every non-empty required set therefore fails coverage with an
        // explicit `UnsupportedOp` — no silent CPU fall back (FR-EX-08).
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::Gelu,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::Conv1d,
            HotOp::GroupedConv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            // Vocoder Metal wave common vocoder primitives (2026-08-14).
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
            HotOp::GroupFsq,
            HotOp::CausalHifiGan,
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

    /// CoreML scaffold slice: `covered_by_coreml` is `false` for every variant,
    /// so `for_backend(CoreMl, …)` never silently runs on the CPU. Because the
    /// arm probes the ANE first, the outcome is probe-gated and honest:
    /// `UnsupportedOp` on a host WITH an ANE (coverage empty), or
    /// `BackendUnavailable` on a host without one — never `Ok` in the scaffold,
    /// and never a fabricated pass. When the M5-01-T02 ADR + execution path
    /// land, flip the covered variants in `HotOp::covered_by_coreml` and tighten
    /// this test.
    #[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn coreml_coverage_is_empty_in_scaffold() {
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::Gelu,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::Conv1d,
            HotOp::GroupedConv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            // Vocoder Metal wave common vocoder primitives (2026-08-14).
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
            HotOp::GroupFsq,
            HotOp::CausalHifiGan,
        ] {
            assert!(
                !op.covered_by_coreml(),
                "{op:?} unexpectedly covered by the M5-01 scaffold CoreML backend (no execution \
                 path is wired). If the T02 ADR + execution path have landed, update \
                 `HotOp::covered_by_coreml` and shrink this test accordingly.",
            );
            // The seam must never return `Ok` in the scaffold, and must never
            // fall back to the CPU. On an ANE host it is `UnsupportedOp`; with
            // no ANE it is `BackendUnavailable` (a probe-gated skip, not a
            // fabricated pass). `Compute` has no `Debug`, so assert on the
            // matched shape rather than formatting the value.
            assert!(
                matches!(
                    Compute::for_backend(BackendKind::CoreMl, &[op]),
                    Err(VokraError::UnsupportedOp(_)) | Err(VokraError::BackendUnavailable(_))
                ),
                "for_backend(CoreMl, &[{op:?}]) must be UnsupportedOp (ANE present) or \
                 BackendUnavailable (no ANE) in the scaffold — never Ok, never a CPU fall back",
            );
        }
        // Empty required set is also explicit (no callable execution path).
        assert!(matches!(
            Compute::for_backend(BackendKind::CoreMl, &[]),
            Err(VokraError::UnsupportedOp(_)) | Err(VokraError::BackendUnavailable(_))
        ));
    }

    /// QNN scaffold slice: `covered_by_qnn` is `false` for every variant, so
    /// `for_backend(Qnn, …)` never silently runs on the CPU. Because the arm
    /// probes the QNN runtime first, the outcome is probe-gated and honest:
    /// `UnsupportedOp` on a host WITH a QNN runtime (coverage empty), or
    /// `BackendUnavailable` on a host without one (every CI runner today) —
    /// never `Ok` in the scaffold, and never a fabricated pass. When the
    /// SDK-gated graph-construction re-issue wave lands, flip the covered
    /// variants in `HotOp::covered_by_qnn` and tighten this test.
    #[cfg(all(
        feature = "qnn",
        any(target_os = "android", target_os = "linux", target_os = "windows")
    ))]
    #[test]
    fn qnn_coverage_is_empty_in_scaffold() {
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::Gelu,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::Conv1d,
            HotOp::GroupedConv1d,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            // Vocoder Metal wave common vocoder primitives (2026-08-14).
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
            HotOp::GroupFsq,
            HotOp::CausalHifiGan,
        ] {
            assert!(
                !op.covered_by_qnn(),
                "{op:?} unexpectedly covered by the M5-02 scaffold QNN backend (no execution path \
                 is wired). If the SDK-gated re-issue wave has landed, update \
                 `HotOp::covered_by_qnn` and shrink this test accordingly.",
            );
            // The seam must never return `Ok` in the scaffold, and must never
            // fall back to the CPU. On a QNN host it is `UnsupportedOp`; with no
            // QNN runtime it is `BackendUnavailable` (a probe-gated skip, not a
            // fabricated pass). `Compute` has no `Debug`, so assert on the
            // matched shape rather than formatting the value.
            assert!(
                matches!(
                    Compute::for_backend(BackendKind::Qnn, &[op]),
                    Err(VokraError::UnsupportedOp(_)) | Err(VokraError::BackendUnavailable(_))
                ),
                "for_backend(Qnn, &[{op:?}]) must be UnsupportedOp (QNN runtime present) or \
                 BackendUnavailable (no runtime) in the scaffold — never Ok, never a CPU fall back",
            );
        }
        // Empty required set is also explicit (no callable execution path).
        assert!(matches!(
            Compute::for_backend(BackendKind::Qnn, &[]),
            Err(VokraError::UnsupportedOp(_)) | Err(VokraError::BackendUnavailable(_))
        ));
    }

    /// Off-target / feature-off build: `BackendKind::Qnn` falls to the
    /// target-agnostic error path — an explicit `BackendUnavailable` naming the
    /// `qnn` feature. Never a silent CPU substitute (FR-EX-08). This is the path
    /// taken on the macOS authoring host (QNN is not an Apple backend) and in
    /// every default build.
    #[cfg(not(all(
        feature = "qnn",
        any(target_os = "android", target_os = "linux", target_os = "windows")
    )))]
    #[test]
    fn qnn_not_compiled_in_is_explicit_backend_unavailable() {
        // `Compute` does not derive `Debug`, so unwrap the error manually.
        let err = match Compute::for_backend(BackendKind::Qnn, &[HotOp::Gemm]) {
            Ok(_) => panic!("Qnn must fail explicitly when not compiled in"),
            Err(e) => e,
        };
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("qnn"),
            "the error must name the `qnn` feature so the caller knows what to enable: {msg}"
        );
        // make_backend has no Qnn arm (QNN is a delegate, not a per-op graph
        // backend — same as CoreML); it is BackendUnavailable regardless.
        assert!(matches!(
            make_backend(BackendKind::Qnn),
            Err(VokraError::BackendUnavailable(_))
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
        for op in [
            HotOp::RmsNorm,
            HotOp::ScaleNorm,
            HotOp::GroupNorm,
            HotOp::GeluNew,
            HotOp::Relu,
            HotOp::Tanh,
            HotOp::Silu,
            HotOp::MimiRvq,
            HotOp::DacRvq,
            HotOp::EncodecRvq,
            HotOp::WavTokenizerVq,
            HotOp::Xcodec2Fsq,
            HotOp::SnakeActivation,
            HotOp::SnacDecode,
            HotOp::DenoiseApplyMask,
            HotOp::Qwen3TtsCodec,
            // Vocoder Metal wave common vocoder primitives (2026-08-14).
            HotOp::SnakeBeta,
            HotOp::SinegenDeterministic,
            HotOp::AntiAliasedUpsample,
            HotOp::GroupedConv1d,
            HotOp::GroupFsq,
            HotOp::CausalHifiGan,
        ] {
            assert!(
                !op.covered_by_webgpu(),
                "{op:?} unexpectedly WebGPU-covered — the RVQ / Vocoder GPU kernels are \
                 deferred; if one has just landed, flip `HotOp::covered_by_webgpu` and update \
                 this test.",
            );
            assert!(matches!(
                Compute::for_backend(BackendKind::WebGpu, &[op]),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
    }
}
