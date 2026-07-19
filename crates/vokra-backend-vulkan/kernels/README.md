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
└── precompiled/         # SPIR-V binaries (git-committed; M4-13-T16 done 2026-07-19)
    ├── <name>.spv       # all 12 committed — see "Blob status" below
    ├── SHA256SUMS       # written by scripts/compile-vulkan-shaders.sh --update
    └── PROVENANCE       # compiler family+version pin + per-source SHA-256 (drift gate)
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

# CI drift gate (gpu-vulkan-parity.yml): two stages —
#   (1) source-hash gate: every committed .spv's .comp source is
#       SHA-256-compared against precompiled/PROVENANCE (compiler-
#       independent — catches "edited source without recompiling" on any
#       host, including CI runners with a different glslang version);
#   (2) recompile byte-diff, run ONLY when the local compiler family AND
#       version match the PROVENANCE pin (cross-tool SPIR-V is not
#       byte-stable — a mismatch is an honest skip, not a fabricated
#       pass; blob-byte integrity is separately enforced by the SHA-256
#       pins in src/spirv.rs via cargo test on every host).
# Exits 1 on drift; exits 0 with an honest note when nothing is committed:
scripts/compile-vulkan-shaders.sh --check
```

Each `.comp` header names its own `--target-env` (gemm_coopmat = vulkan1.3,
everything else vulkan1.1); the script parses it per file. `glslc` is
preferred; `glslangValidator` (Ubuntu `glslang-tools`, Homebrew `glslang`)
is the fallback — the two emit different bytes, so the committed blobs'
producing tool is pinned in `precompiled/PROVENANCE` and `--update` refuses
single-kernel rebuilds with a non-matching tool (no mixed-toolchain blob
sets).

After `--update`, paste each hash from `precompiled/SHA256SUMS` into
`src/spirv.rs`'s `SHADERS` manifest (`expected_sha256_hex`) and switch the
matching `load_spv` arm to `include_bytes!` — `verify_pinned_hashes` and
every blob-gated parity test light up automatically (placeholder-then-swap,
M4-13-T02).

## Graph-executor op coverage vs the CUDA / WebGPU arms (M4-13-T14 + M4-01-T17)

Machine-checked by `tests/graph_arm_coverage.rs` (host-portable) and the
blob-driven lock-step tests; `supports()` additionally requires the backing
blob to be committed (conservative honesty — an advertised op must actually
dispatch). The WebGPU column (M4-01-T17) is machine-checked by
`vokra-backend-webgpu`'s `graph_arm_mapping_is_the_vulkan_target_set` test;
WGSL sources are embedded text, so the WebGPU arm has **no blob gate** — all
five arms are live from the M4-01 commit.

| graph `OpKind` | CUDA arm `supports()` | Vulkan arm (principled) | Vulkan backing shader | Vulkan status today (all 12 `.spv` committed, M4-13-T16) | WebGPU arm (M4-01) | WebGPU backing WGSL |
|----------------|-----------------------|--------------------------|------------------------|-------------------------------------|--------------------|----------------------|
| `MatMul`       | ✅                    | ✅                       | `gemm_subgroup` / `gemm_coopmat` (probe-selected) | **live** | ✅ | `gemm_f32` |
| `Add`          | ✅                    | ✅                       | `add_f32` (hand-crafted) | **live** | ✅ | `add_f32` |
| `Mul`          | ✅                    | ✅                       | `elementwise` (OP=mul)  | **live** | ✅ | `elementwise` (op=mul) |
| `Softmax`      | ✅                    | ✅                       | `softmax`               | **live** | ✅ | `softmax` |
| `Copy`         | ❌ (Vulkan/WebGPU-only runtime-verification op) | ✅ | `copy_f32` (hand-crafted) | **live** | ✅ | `copy_f32` |
| `Stft` (+ other front-end signal ops) | ❌ | ❌ | — | **honest gap on ALL backend graph arms** — front-end ops run in `vokra-ops`; putting `Stft` on a GPU graph arm is a separate M4+ decision | ❌ | — |

Vulkan minus `Copy` == WebGPU minus `Copy` == the CUDA `supports()` set
(`crates/vokra-backend-cuda/src/backend.rs`, M3-01-T06) — the milestones §8
M4-13 "graph-executor の op coverage が CUDA arm と同等" condition; the
WebGPU arm (M4-01-T17) targets the identical `{Copy, Add, MatMul, Mul,
Softmax}` set (`Copy` is the Vulkan/WebGPU extra — the CUDA arm does NOT
have it).

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

## Blob status (M4-13-T16 done, 2026-07-19)

Full kernel bodies are committed for all 12 GLSL sources — including the
M4-13 corrections: the reduction kernels (`gemv` / `softmax` /
`softmax_causal` / `layer_norm`) were rewritten from subgroup reduces
(wrong when a workgroup spans several subgroups: lavapipe sg=8, Mali
sg=16, RDNA sg=64) to barrier-based shared-memory tree reductions, and
`gelu` was corrected from the tanh approximation to the CPU kernel's
exact/erf form (identical A&S 7.1.26 coefficients).

**All 12 `.spv` blobs are committed** (glslangValidator 16.4.0 — "Glslang
Version: 11:16.4.0", the ADR §4 (a) `brew install glslang` path; all 12
`.comp` compiled clean with zero source changes, so the `glsl_mirror.rs`
transcription tests remain true transcriptions), SHA-256-pinned in
`src/spirv.rs` and embedded via `include_bytes!` — the blob-gated dispatch
arms, `supports()` coverage, and the T12/T13 parity suites are lit up with
zero further code changes (the placeholder-then-swap seam worked as
designed). The T12/T13 parity suites still need a live Vulkan ICD to
*execute* — lavapipe CI (`gpu-vulkan-parity.yml` workflow_dispatch,
M4-13-T18) proves the dispatch chain; it CANNOT catch driver-level
GLSL→SPIR-V bugs, so the Android real-device soak (M4-13-T17) is the WP's
exit hard gate. The Apple-Silicon authoring host compiles the manifest and
verifies every SHA-256 pin, but has no Vulkan target arm (macOS uses
Metal), so dispatch tests skip there with a logged reason.
