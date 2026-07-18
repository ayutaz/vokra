# Pre-compiled SPIR-V blobs

All 12 `.spv` blobs compiled from `../glsl/*.comp` are committed here
(M4-13-T16, 2026-07-19), together with:

- `SHA256SUMS` — the `sha256sum` of every blob, written by
  `scripts/compile-vulkan-shaders.sh --update`. Cross-checked against the
  manifest pins in `../../src/spirv.rs` by the
  `sha256sums_file_matches_manifest_pins` test (pure file check, no Vulkan
  driver needed).
- `PROVENANCE` — the compiler family + version that produced the blobs
  (glslangValidator, "Glslang Version: 11:16.4.0" — the sanctioned
  `brew install glslang` path, ADR M3-02-spirv-generation §4 (a)) plus the
  SHA-256 of each `.comp` source at compile time. `--check` uses it as a
  two-stage drift gate: the source hashes catch "edited `.comp` without
  recompiling" on any host, and the recompile byte-diff runs only when the
  local tool matches the pin (different compiler families/versions emit
  different — equally valid — SPIR-V bytes).

The runtime embeds each blob at compile time via `include_bytes!` in
`src/spirv.rs::load_spv`. The `.spv` blobs are binary and git-committed on
purpose — they are the frozen kernel artefacts the runtime dispatches. The
compiler is **not** a runtime dependency — it is a one-off developer-side
tool, same as `cargo`, and it never appears in `Cargo.lock` (NFR-DS-02).

Regenerating (deliberate kernel change only):

```bash
scripts/compile-vulkan-shaders.sh --update   # rewrites *.spv + SHA256SUMS + PROVENANCE
# then re-paste the new hashes into ../../src/spirv.rs (SHADERS manifest)
# and commit all of it together.
```
