# API reference

**English** | [日本語](api-reference.ja.md)

An index of Vokra's API surfaces and where each one's reference lives. Most of
it is **auto-generated** from the source; this page is a thin pointer, not a
hand-maintained copy (which would rot). What is generated versus written by
hand is stated in §4.

## 1. Rust — docs.rs

The Rust crates are documented with `rustdoc`. Once a future release publishes
the crates, each crate auto-links to its own page:

- `https://docs.rs/vokra-core` — the IR, `Backend` trait, GGUF loader, engine
- `https://docs.rs/vokra-capi` — the C ABI surface crate (`IF-01`)
- `https://docs.rs/vokra-models`, `.../vokra-ops`, and the backend crates

The feature-gated GPU/NPU backends carry `[package.metadata.docs.rs]` so
docs.rs builds their platform-specific API (Metal / CoreML on an Apple target,
WebGPU on wasm32, CUDA / Vulkan / QNN via their features). Build a focused,
memory-safe crate locally with:

```sh
cargo doc -p vokra-core --no-deps --open
```

Maintainers build workspace-wide rustdoc on VAST or CI, never on the 16 GB
development Mac.

## 2. C ABI — `include/vokra.h`

The canonical C reference is the generated header
[`include/vokra.h`](../include/vokra.h). It is produced by
`scripts/gen-c-abi.sh` from the `vokra-capi` crate and its doc comments are the
reference text; a CI drift check keeps it in sync with the Rust source. Every
Unity, Godot, Swift, Kotlin, Python and JS binding sits on this one header
(`IF-01`). Vokra is distributed as an ordinary Cargo crate / single library, so
this header plus the library is the whole integration surface (`NFR-DS-03`).

## 3. Language bindings

Each binding documents its own idiomatic surface on top of the C ABI:

- **Unity (C#)** — see the [Unity tutorial](tutorials/unity.md)
- **Python** — see [`bindings/python/README.md`](../bindings/python/README.md)
- **Godot (GDScript)** — see the [Godot tutorial](tutorials/godot.md)
- **Swift / iOS** — the [`Package.swift`](../Package.swift) SwiftPM manifest and
  the [iOS tutorial](tutorials/ios.md)

## 4. What is auto-generated, and what is not

- **Auto-generated**: the Rust docs (rustdoc → docs.rs) and the C header
  (`gen-c-abi.sh` → `include/vokra.h`). These regenerate from source and are
  the source of truth.
- **Manual, but thin**: this index and the binding tutorials. They point at the
  generated references and the working examples; they are not a second copy of
  the API.
- **Deferred (honest)**: HTML rendering of the C header (doxygen) and
  per-language HTML generators (C# / Python / Swift doc tools) are not wired —
  the header comments and the tutorials are the reference for now. The first
  docs.rs render is verified by the owner after a crates.io publish.

## 5. Current 0.3.0 release and Apple verification status

The current release line is workspace version `0.3.0`.

**2026-09-12 current snapshot:** the current `main` baseline is `43d127f1`.
The live, read-only public audit reports 194 repositories, 193 GGUF-bearing
repositories and 198 GGUF files. CPU status is `full=133`, `partial=45`,
`no-runtime-binder=15`, `not-artifact=1`; Metal status is `full=133`,
`blocked-by-cpu=60`, `not-artifact=1`, leaving 61 unresolved public rows. The
authorized Scaleway Apple CPU/reference, Metal/reference and no-fallback batch
passed only for its named scopes; four separately approved artifacts were then
published through the gated workflow. UTMOS numeric parity remains unclaimed,
and there are 0 release tags and 0 GitHub Releases. This is not a claim of
complete public model-catalog support or v1.0 release readiness.

**2026-09-09 audit-start historical snapshot:** PR #79 was at
`9efcd16eb63b857f48fc00d0b83d1113defd578b`; its remote checks then reported
110 successful checks, 13 expected skips, and 0 failures. The audit-start
public audit reported 194 repositories (193 GGUF repositories, 198 GGUF
files).
CPU coverage is
`full=131`, `partial=45`, `no-runtime-binder=17`, `not-artifact=1`; Metal is
`full=131`, `blocked-by-cpu=62`, `not-artifact=1`; source-level CPU-only
coverage is 0.
At that audit-start snapshot, there were 0 release tags and 0 GitHub Releases.

GigaAM v3 and Multilingual had complete conservative Metal code routes, but
their Apple-hardware verdicts were not available. The remaining 63 public rows
were not claimed complete. Scaleway had not started; CI/audit results did not
substitute for Apple hardware evidence. UTMOS numeric parity was also not
claimed because its legacy Lightning checkpoint is intentionally refused by
the restricted `weights_only=True` loader.

## Keeping this page current

**Last verified: 2026-09-12 — against current `main` baseline `43d127f1` and
`include/vokra.h`.** The
pre-alpha Python generator and checked-in `ctypes` table cover all 57 generated
C functions exactly; the header has 15 typedefs, four enums, two concrete
structures, and nine opaque handles. The high-level Python package remains a
smaller idiomatic surface rather than a wrapper class for every C handle.

- **Update responsibility**: a PR that adds a published crate, a new binding, or
  changes the C ABI generation updates this index and its Japanese twin in the
  same PR.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`).
- **Re-fetch the generated surfaces**:

```sh
scripts/gen-c-abi.sh
# Maintainers run workspace rustdoc on VAST/CI, not on the development Mac:
cargo doc --no-deps --workspace
```
