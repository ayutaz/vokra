# Pre-compiled SPIR-V blobs

Once each `../glsl/*.comp` is finalised (M3-02-T14〜T22), the developer runs
`glslc --target-env=vulkan1.1 -o <name>.spv ../glsl/<name>.comp` and commits
the resulting `.spv` blob to this directory. `build.rs` (via
`include_bytes!`) will then embed each blob at compile time.

The `.spv` blobs are binary and git-committed on purpose — they are the
frozen kernel artefacts the runtime dispatches. The developer's `glslc` (Vulkan
SDK) is **not** a runtime dependency — it is a one-off developer-side tool,
same as `cargo`, and it never appears in `Cargo.lock` (NFR-DS-02).

No `.spv` files are committed in the foundation slice (2026-07-09) — see
`../README.md` for the full ticket status.
