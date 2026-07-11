# SPIR-V compute kernels (M3-02-T13〜T22)

This directory holds the GLSL compute-shader **sources** and their
**precompiled SPIR-V blobs** for the Vokra Vulkan backend.

## Layout

```
kernels/
├── README.md            # this file
├── glsl/                # GLSL sources (git-committed)
│   ├── gemm_subgroup.comp     # T14 fallback path (Adreno 6xx+ / Mali G7x+)
│   ├── gemm_coopmat.comp      # T14 fast path (Ampere+/RDNA3+, VK_KHR_cooperative_matrix)
│   ├── gemv.comp              # T15
│   ├── softmax.comp           # T16
│   ├── softmax_causal.comp    # T16 (causal mask, IEEE-754 bit-identical vs host-mask + softmax)
│   ├── layer_norm.comp        # T17
│   ├── gelu.comp              # T18 (tanh approximation, CPU-parity)
│   ├── conv1d.comp            # T19
│   ├── elementwise.comp       # T20 (add / mul, specialization constant selects)
│   ├── activation.comp        # T21 (relu / sigmoid / tanh, spec constant)
│   ├── transpose.comp         # T22
│   └── gather.comp            # T22
└── precompiled/         # SPIR-V binaries (`glslc`-produced, git-committed)
    ├── gemm_subgroup.spv
    ├── gemm_coopmat.spv
    ├── ...
    └── .gitignore       # (empty — .spv are tracked, this file just documents)
```

## Why precompile? (M3-02-T01(g) / T13)

- **NFR-RL-05 forbids CPU-side JIT.** Android's SELinux profile treats
  W^X-flipped pages as a compromise, and iOS forbids JIT outright.
  SPIR-V → GPU ISA translation is the *driver's* job (happens on the GPU
  side, not in Vokra's address space), which is legal — but Vokra never
  generates SPIR-V at runtime.
- **NFR-DS-02 forbids external dependencies.** `spirv-tools` /
  `shaderc-rs` / `glslang-sys` etc. would violate the zero-dep invariant
  that keeps `Cargo.lock` `vokra-*`-only.
- **Build tools are developer-side, not runtime deps.** `glslc` (Vulkan
  SDK) is used *once* by the developer to produce a `.spv` blob per
  shader, which is then committed to the repo. `build.rs` verifies
  existence at build time but never invokes `glslc`.

## Compiling a shader (developer instructions)

```bash
# One-off, per shader:
glslc --target-env=vulkan1.1 -o precompiled/gemm_subgroup.spv glsl/gemm_subgroup.comp

# Or in a batch — recompile every shader whose source changed:
for f in glsl/*.comp; do
    base=$(basename "$f" .comp)
    glslc --target-env=vulkan1.1 -o "precompiled/${base}.spv" "$f"
done
```

The compiled `.spv` files are committed to the repo. CI verifies that they are
in sync with the sources by:
1. Running `scripts/verify-spirv-in-sync.sh` (M3-02-T36 follow-up), OR
2. Recompiling from source in CI (needs `glslc`; Vulkan SDK install ~30 s)
   and diffing the output against the committed `.spv`.

## Consuming a shader (at runtime, in `context.rs` — M3-02-T14)

```rust
// Once VulkanContext exists (T08):
static GEMM_SUBGROUP_SPV: &[u8] = include_bytes!("../kernels/precompiled/gemm_subgroup.spv");
static GEMM_COOPMAT_SPV: &[u8] = include_bytes!("../kernels/precompiled/gemm_coopmat.spv");

// The probe (T30/T31) tells us which one to bind:
let spv = if caps.coop_matrix_precondition_met && driver_supports_coop_matrix_ext {
    GEMM_COOPMAT_SPV
} else {
    GEMM_SUBGROUP_SPV
};
// Feed spv into vkCreateShaderModule (once T14 wires it up).
```

## Foundation-slice status (2026-07-09)

The GLSL sources have skeleton content committed today (all inputs / bindings
declared, kernel body kept minimal so `glslc --preprocess` succeeds and the
declared workgroup layout matches what T14〜T22 expects). Full kernel bodies
land in T14〜T22 once T08〜T12 (`VulkanDevice` / command / descriptor /
pipeline / memory / buffer) lands. No `.spv` is committed yet — those land as
each ticket completes and `glslc` is run against the finalized source.
