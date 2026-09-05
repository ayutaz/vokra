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

The current release line is workspace version `0.3.0`. The parity figures below
are a pre-documentation-refresh snapshot from PR #79 at `d8a93bc3`, reviewed
against `origin/main` `41ce9ffd`; that snapshot recorded 109 passes and 13
expected skips. The live public audit currently reports 194 repositories (193
GGUF repositories, 198 GGUF files). CPU coverage is
`full=131`, `partial=42`, `no-runtime-binder=20`, `not-artifact=1`; Metal is
`full=131`, `blocked-by-cpu=62`, `not-artifact=1`; source-level CPU-only
coverage is 0.
There are currently 0 release tags and 0 GitHub Releases.

GigaAM v3 and Multilingual have complete conservative Metal code routes, but
their Apple-hardware verdicts are not available. OmniASR also awaits the
authenticated Scaleway run. The CI Quality `hf-mac-coverage-unit` and live
advisory checks are green on the latest PR; these CI/audit results do not
substitute for Apple hardware evidence.

## Keeping this page current

**Last verified: 2026-08-31 — against GitHub `main`
`41ce9ffdd4b0959497f55afa5016822f77a8a7b6`, the pre-documentation code
baseline branch `feat/mac-cpu-metal-full-coverage-2026-08-28` at
`9f69277d8a0d5df574c1ee95563bd1f005de91d0`, and `include/vokra.h`.** The
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
