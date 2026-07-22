# API reference

**English** | [日本語](api-reference.ja.md)

An index of Vokra's API surfaces and where each one's reference lives. Most of
it is **auto-generated** from the source; this page is a thin pointer, not a
hand-maintained copy (which would rot). What is generated versus written by
hand is stated in §4.

## 1. Rust — docs.rs

The Rust crates are documented with `rustdoc`. Once the crates are published
(the release train, X-07), each crate auto-links to its own page:

- `https://docs.rs/vokra-core` — the IR, `Backend` trait, GGUF loader, engine
- `https://docs.rs/vokra-capi` — the C ABI surface crate (`IF-01`)
- `https://docs.rs/vokra-models`, `.../vokra-ops`, and the backend crates

The feature-gated GPU/NPU backends carry `[package.metadata.docs.rs]` so
docs.rs builds their platform-specific API (Metal / CoreML on an Apple target,
WebGPU on wasm32, CUDA / Vulkan / QNN via their features). Build the same
locally with:

```sh
cargo doc --no-deps --open
```

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
  docs.rs render is verified by the owner after the crates.io publish (X-07).

## Keeping this page current

**Last verified: 2026-07-21 — against the workspace publish set and
`include/vokra.h`.**

- **Update responsibility**: a PR that adds a published crate, a new binding, or
  changes the C ABI generation updates this index and its Japanese twin in the
  same PR.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`).
- **Re-fetch the generated surfaces**:

```sh
scripts/gen-c-abi.sh && cargo doc --no-deps --workspace
```
