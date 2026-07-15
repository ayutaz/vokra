# WGSL compute kernels (M4-01-T11〜T15)

This directory holds the **WGSL sources** for the Vokra WebGPU backend.
Unlike the Vulkan backend's `kernels/` there is **no precompiled blob
directory**: the Web standard has no binary shader format (there is no
SPIR-V path in WebGPU), so the WGSL **text itself is the shipped artifact**
— embedded into the crate with `include_str!` (`src/wgsl.rs` manifest) and
handed to `device.createShaderModule` at runtime. WGSL → GPU ISA compilation
is the browser / driver's responsibility, not host-side JIT (NFR-RL-05; the
same separation ADR M3-02 records for the driver-side SPIR-V translation).

## Layout

```
kernels/
├── README.md            # this file
└── wgsl/                # WGSL sources (git-committed = the artifact)
    ├── copy_f32.wgsl        # identity copy (dispatch-chain smoke op)
    ├── add_f32.wgsl         # element-wise sum (graph OpKind::Add arm)
    ├── elementwise.wgsl     # add / mul (op uniform flag; OpKind::Mul arm)
    ├── gemm_f32.wgsl        # 16x16 shared-tile GEMM, bias-seeded ascending-k
    ├── gemv_f32.wgsl        # per-row 64-thread tree reduction (tied logits head)
    ├── softmax.wgsl         # row softmax, max-shift stabilized
    ├── softmax_causal.wgsl  # causal mask (exp(-inf)=0 host-mask equivalence)
    ├── layer_norm.wgsl      # biased variance; eps via uniform (model config, never invented)
    ├── gelu.wgsl            # EXACT (erf) form — A&S 7.1.26, identical coefficients to the CPU kernel
    ├── conv1d.wgsl          # Whisper stem envelope (direct conv, ic-major-kk order = im2col+GEMM order)
    └── activation.wgsl      # relu / sigmoid / tanh (kind uniform flag)
```

## Drift gate

Every source is SHA-256-pinned in `src/wgsl.rs` (`expected_sha256_hex`,
zero-dep FIPS-180-4 — the Vulkan `spirv.rs` implementation duplicated).
`cargo test -p vokra-backend-webgpu` fails on any source/pin divergence and
prints the offender's actual hash; updating a pin is a reviewed change.
The plan/WGSL `@workgroup_size` lock-step and the glue bind contract
(storage buffers `0..n`, output last, uniform at `n`) are pinned by the
native tests in `src/plan.rs` / `src/wgsl.rs`.

## Graph-executor op coverage

The WebGPU graph arm covers `{Copy, Add, MatMul, Mul, Softmax}` — identical
to the M4-13 Vulkan target set (the CUDA arm's `{MatMul, Add, Mul, Softmax}`
plus `Copy`, the Vulkan/WebGPU runtime-verification extra). `Stft` and the
other front-end signal ops are covered by **no** backend graph arm (the
honest all-backend gap — they run in `vokra-ops`). The remaining kernels —
`gemv_f32`, `softmax_causal`, `layer_norm`, `gelu`, `conv1d`, `activation`
— have **no `OpKind` variant**: they are the imperative Whisper primitives
reached through the `vokra-models` `Compute` seam (M4-01-T16), surface 2 of
the M4-13-T01 two-surface distinction. The cross-backend table lives in
`crates/vokra-backend-vulkan/kernels/README.md` (WebGPU column added by
M4-01-T17).

## Numerical posture (NFR-QL-01)

FP32 storage and accumulators, fixed; no f16 anywhere (structural test).
Parity vs the CPU oracle is judged at atol = 0.01 in the browser harness
(`tools/wasm/parity.html`, gated on a live adapter — CC M1 iMac Chrome +
owner spot check T28); the `gemm` accumulation order matches the CPU scalar
chain (bias-seeded, ascending k) so the expected residual is driver
mul+add-contraction only. The gelu erf approximation is transcription-pinned
natively against the CPU kernel (`src/plan.rs`), which uses the identical
A&S 7.1.26 coefficients.
