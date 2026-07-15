# SPIR-V compute kernels (M3-02-T13〜T22 / M4-13)

This directory holds the GLSL compute-shader **sources** and their
**precompiled SPIR-V blobs** for the Vokra Vulkan backend.

## Layout

```
kernels/
├── README.md            # this file
├── glsl/                # GLSL sources (git-committed)
│   ├── gemm_subgroup.comp     # fallback path (Adreno 6xx+ / Mali G7x+); vulkan1.1
│   ├── gemm_coopmat.comp      # fast path (Ampere+/RDNA3+, VK_KHR_cooperative_matrix); vulkan1.3
│   ├── gemv.comp              # decoder-step hot path (shared-memory workgroup reduction)
│   ├── softmax.comp           # row softmax (shared-memory reduction)
│   ├── softmax_causal.comp    # causal mask, exp(-inf)=0 host-mask equivalence
│   ├── layer_norm.comp        # eps via push constant (model config, never invented)
│   ├── gelu.comp              # EXACT (erf) form — A&S 7.1.26 coefficients identical to the CPU kernel
│   ├── conv1d.comp            # Whisper front-end stride/padding envelope, batched
│   ├── elementwise.comp       # add / mul (OP specialization constant)
│   ├── activation.comp        # relu / sigmoid / tanh (KIND specialization constant)
│   ├── transpose.comp         # 2-D axis swap (reshape needs NO shader — buffer view)
│   └── gather.comp            # embedding lookup (OOB rejected host-side pre-dispatch)
├── handcrafted/         # 2 smoke-test kernels as Rust `const [u32]` (ADR §5 cap — no more)
│   ├── copy_f32.spv.rs
│   └── add_f32.spv.rs
└── precompiled/         # SPIR-V binaries (`glslc`-produced, git-committed; owner M4-13-T16)
    ├── <name>.spv       # none committed yet — see "Placeholder status" below
    └── SHA256SUMS       # written by scripts/compile-vulkan-shaders.sh --update
```

## Why precompile? (M3-02-T01(g) / T13, ADR M3-02-spirv-generation)

- **NFR-RL-05 forbids CPU-side JIT.** Android's SELinux profile treats
  W^X-flipped pages as a compromise, and iOS forbids JIT outright.
  SPIR-V → GPU ISA translation is the *driver's* job (happens on the GPU
  side, not in Vokra's address space), which is legal — but Vokra never
  generates SPIR-V at runtime.
- **NFR-DS-02 forbids external dependencies.** `spirv-tools` /
  `shaderc-rs` / `glslang-sys` etc. would violate the zero-dep invariant
  that keeps `Cargo.lock` `vokra-*`-only (permanently banned in
  `deny.toml`, ADR §4-(d)).
- **Build tools are developer-side, not runtime deps.** `glslc` (Vulkan
  SDK) is used *once* by the developer to produce a `.spv` blob per
  shader, which is then committed to the repo. `build.rs` verifies
  existence at build time but never invokes `glslc`.

## Compiling / drift-checking (developer instructions, M4-13-T11)

```bash
# Recompile every shader in place + refresh SHA256SUMS (owner M4-13-T16):
scripts/compile-vulkan-shaders.sh --update

# One kernel only:
scripts/compile-vulkan-shaders.sh --update gemm_subgroup

# CI drift gate (gpu-vulkan-parity.yml): recompile to a temp dir and
# SHA-256-diff against the committed blobs; exits 1 on drift, exits 0 with
# an honest "nothing committed yet" note in the placeholder slice:
scripts/compile-vulkan-shaders.sh --check
```

Each `.comp` header names its own `--target-env` (gemm_coopmat = vulkan1.3,
everything else vulkan1.1); the script parses it per file. `glslc` is
preferred; `glslangValidator` (Ubuntu `glslang-tools`) is the fallback —
the two emit different bytes, so `--check` must use the tool family that
produced the committed blobs.

After `--update`, paste each hash from `precompiled/SHA256SUMS` into
`src/spirv.rs`'s `SHADERS` manifest (`expected_sha256_hex`) and switch the
matching `load_spv` arm to `include_bytes!` — `verify_pinned_hashes` and
every blob-gated parity test light up automatically (placeholder-then-swap,
M4-13-T02).

## Graph-executor op coverage vs the CUDA arm (M4-13-T14)

Machine-checked by `tests/graph_arm_coverage.rs` (host-portable) and the
blob-driven lock-step tests; `supports()` additionally requires the backing
blob to be committed (conservative honesty — an advertised op must actually
dispatch).

| graph `OpKind` | CUDA arm `supports()` | Vulkan arm (principled) | Vulkan backing shader | status today (no `.spv` committed) |
|----------------|-----------------------|--------------------------|------------------------|-------------------------------------|
| `MatMul`       | ✅                    | ✅                       | `gemm_subgroup` / `gemm_coopmat` (probe-selected) | blob-gated → `UnsupportedOp` |
| `Add`          | ✅                    | ✅                       | `add_f32` (hand-crafted) | **live** |
| `Mul`          | ✅                    | ✅                       | `elementwise` (OP=mul)  | blob-gated → `UnsupportedOp` |
| `Softmax`      | ✅                    | ✅                       | `softmax`               | blob-gated → `UnsupportedOp` |
| `Copy`         | ❌ (Vulkan-only runtime-verification op) | ✅ | `copy_f32` (hand-crafted) | **live** |
| `Stft` (+ other front-end signal ops) | ❌ | ❌ | — | **honest gap on BOTH arms** — front-end ops run in `vokra-ops`; putting `Stft` on a GPU graph arm is a separate M4-01+ decision |

Vulkan minus `Copy` == the CUDA `supports()` set
(`crates/vokra-backend-cuda/src/backend.rs`, M3-01-T06) — the milestones §8
M4-13 "graph-executor の op coverage が CUDA arm と同等" condition.

The remaining kernels — `gemv`, `softmax_causal`, `layer_norm`, `gelu`,
`conv1d`, `activation`, `transpose`, `gather` — have **no `OpKind`
variant** (surface 2 of the M4-13-T01 two-surface distinction); they are
the imperative Whisper-base primitives exercised by
`tests/parity_vulkan.rs` (per-kernel, T12) and
`tests/parity_whisper_chain_vulkan.rs` (model-level chain, T13).

## Consuming a shader (at runtime — M4-13-T02〜T08)

The typed entry points live on `VulkanBackend` (`src/kernels.rs`):
`gemm_f32` / `gemv_f32` / `softmax_f32` / `softmax_causal_f32` /
`layer_norm_f32` / `gelu_f32` / `conv1d_f32` / `elementwise_f32` /
`activation_f32` / `transpose_f32` / `gather_f32`. Each pairs a
host-portable plan (`src/plan.rs`: shape validation, push-constant packing,
workgroup math — mirrors the `.comp` contracts field-for-field) with the
generic dispatch chain (`src/context.rs::dispatch_kernel`). A missing blob
is an explicit `UnsupportedOp` (FR-EX-08), never a silent CPU fall back.

## Placeholder status (M4-13, 2026-07-15)

Full kernel bodies are committed for all 12 GLSL sources — including the
M4-13 corrections: the reduction kernels (`gemv` / `softmax` /
`softmax_causal` / `layer_norm`) were rewritten from subgroup reduces
(wrong when a workgroup spans several subgroups: lavapipe sg=8, Mali
sg=16, RDNA sg=64) to barrier-based shared-memory tree reductions, and
`gelu` was corrected from the tanh approximation to the CPU kernel's
exact/erf form (identical A&S 7.1.26 coefficients). **No `.spv` is
committed yet** — the owner compiles + commits + SHA-256-pins them
(M4-13-T16), which lights up the blob-gated dispatch arms, `supports()`
coverage, and the T12/T13 parity suites with zero further code changes.
lavapipe CI proves the dispatch chain; it CANNOT catch driver-level
GLSL→SPIR-V bugs, so the Android real-device soak (M4-13-T17) is the WP's
exit hard gate.
