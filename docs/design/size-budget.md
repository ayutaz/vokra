# Binary-size budget & levers (M1-11a)

Status: Draft (Claude Code, M1-11a). Enforced by
[`scripts/check-binary-size.sh`](../../scripts/check-binary-size.sh); levers live
in the root [`Cargo.toml`](../../Cargo.toml) `[profile.release]` /
`[profile.release-min]`.

## Budget

| Artifact | Target | Kind |
|---|---|---|
| `libvokra` cdylib (`libvokra.dylib` / `libvokra.so` / `vokra.dll`) | **< 10 MiB** | **hard gate** (NFR-DS-01 "core single binary") |
| same, on mobile | < 5 MiB | informational goal (NFR-DS-01) |
| `libvokra.a` staticlib | — | not gated (see below) |
| CLI / `vokra-convert` bins | — | not gated (design decision #4) |

The gate applies to the **cdylib** — the single loadable binary that Unity /
Godot / iOS / Android and the CLI embed. The **staticlib** (`libvokra.a`) is an
*archive* of every object file; it is naturally several times larger and the
consumer's linker dead-strips it into the final binary, so gating its raw size
would be misleading. `check-binary-size.sh` prints it for information only.

## Why this is a *soft* gate today

At M1-11a the spike `libvokra` is far under 10 MiB (it is hundreds of KiB): **no
model weights are linked into the binary** — Silero VAD / Whisper / piper-plus
weights are loaded from external GGUF files at runtime. The 10 MiB budget only
becomes a real constraint once full-model code paths and any embedded tables
exist. Therefore:

* The 10 MiB number is **provisional**; it is a regression tripwire on *code*
  size, not yet a validated full-model figure.
* Real verification against a full-model build is **M1-11b**, which is blocked on
  real GGUFs / hardware.
* The threshold is configurable (`VOKRA_SIZE_BUDGET_BYTES`) and the gate can run
  advisory-only (`VOKRA_SIZE_SOFT=1`) while it is being calibrated.

## Levers applied (root `Cargo.toml`)

`[profile.release]` (builds the shipped `libvokra` cdylib/staticlib + release
bins):

| Lever | Value | Why |
|---|---|---|
| `opt-level` | **`3` (kept)** | RTF is the first-priority NFR; the runtime hot path ships from here. Size does **not** come from lowering opt-level. |
| `lto` | `"fat"` | Cross-crate inlining across the `vokra-*` graph removes duplicated / monomorphised code — smaller *and* faster. |
| `codegen-units` | `1` | Maximises LTO / dead-code removal (modest compile-time cost). |
| `strip` | `true` | Strips symbols + debuginfo — the single biggest lever for a debug-heavy library. |
| `panic` | **left `"unwind"`** | **Must not be `"abort"`** — see below. |

## HARD CONSTRAINT: `panic="abort"` must not reach `libvokra`

`crates/vokra-capi/src/ffi_guard.rs` implements the FFI panic firewall with
`std::panic::catch_unwind`, converting a Rust panic into `VOKRA_ERROR_PANIC`
instead of letting it unwind into C (undefined behaviour). Under
`panic="abort"`, `catch_unwind` becomes a **no-op** and a panic would **abort the
entire host process** (the Unity / Godot / iOS / Android app), not return an
error. `panic` is a whole-compilation setting — it is **not** overridable
per-package the way `opt-level` / `codegen-units` are — so the `libvokra`
cdylib/staticlib **must** build with unwinding.

`[profile.release-min]` (`inherits = "release"`, `panic = "abort"`) exists only
for **standalone binaries that never cross the C ABI** (e.g. the offline
`vokra-convert` tool, a future `vokra` CLI), which have no firewall to preserve.
Invoke it **targeted at a bin package**, never workspace-wide or for
`vokra-capi`:

```sh
cargo build --profile release-min -p vokra-convert   # OK: non-FFI standalone bin
cargo build --profile release-min                    # WRONG: rebuilds libvokra with abort
cargo build --profile release-min -p vokra-capi      # WRONG: breaks the FFI firewall
```

Every script and CI job builds the runtime with `release`, never `release-min`.

## Feature-gating (off by default)

Optional-but-heavy pieces should sit behind cargo features that are **off by
default**, so the default and mobile builds stay lean:

* **GPU backends** (Metal / CUDA — GPU execution). Already implemented as the
  first-party, zero-external-dep crates `vokra-backend-metal` /
  `vokra-backend-cuda`, gated **off by default** behind `vokra-models`'s `metal`
  / `cuda` cargo features. The shipped/gated `libvokra` cdylib is built
  `-p vokra-capi` with **default features only** — `vokra-capi` adds no
  `metal`/`cuda` passthrough and `check-binary-size.sh` passes no `--features`,
  so the gated artifact is **CPU-backend-only** and its size is **unchanged** by
  the addition of the two GPU backend crates. Enabling a GPU feature grows *that*
  build, not the gated default artifact (and adds no external crate — NFR-DS-02
  still holds, both backends are hand-written FFI with no binding crate).
* **Watermarking** (AudioSeal / C2PA — M1-07). If a C2PA path ever depends on the
  `c2pa-rs` crate (Apache-2.0), it **must** be an opt-in feature so the default
  build + CI stay zero-dependency (NFR-DS-02). Coordinate the feature name with
  M1-07.
* **Neural MOS models** (UTMOS / DNSMOS — M1-09b).
* Extra-ISA code paths beyond the default runtime-dispatch set.

These gates touch crate `Cargo.toml`s and are owned by their respective WPs;
M1-11a only provides the profile + gate scaffolding. Zero-dependency (NFR-DS-02)
and no-GPL/LGPL remain invariant regardless of features.

## Zero-dependency note

`check-binary-size.sh` uses only POSIX tooling (`cargo`, `wc`, `tr`, `printf`).
`cargo bloat` (referenced for diagnosis) is a **build-time dev tool**, not a
`Cargo.lock` dependency — the same status as `cargo-deny` / `cargo-audit` — so it
does not affect NFR-DS-02.

## Usage

```sh
scripts/check-binary-size.sh              # build release cdylib, measure, gate
scripts/check-binary-size.sh --self-test  # unit-test the compare logic (no build)

# CI (after the build job already ran `cargo build --release`):
VOKRA_SIZE_SKIP_BUILD=1 scripts/check-binary-size.sh
```
