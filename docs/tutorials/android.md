# Android (native JNI) tutorial

**English** | [日本語](android.ja.md)

This is the **native** Android path: build `libvokra.so`, load it over JNI, and
call the C ABI directly. If you are integrating through Unity instead, use the
[Unity tutorial](unity.md) — this page does not repeat that. Android GPU
acceleration is **Vulkan** (`FR-BE-07`: NNAPI is permanently unsupported).

## 1. Build the CPU-only library

The repository ships `scripts/build-android.sh` <!-- anchor: scripts/build-android.sh -->
(from M2-11). It cross-compiles the JNI library with the Android NDK. It
requires `ANDROID_NDK_HOME`, targets API level 24 (Android 7.0) and produces
`target/aarch64-linux-android/release/libvokra.so`:

```sh
ANDROID_NDK_HOME=/path/to/ndk scripts/build-android.sh
```

**Honest scope note.** `build-android.sh` is a `--no-default-features`
**CPU-only** build (its header says "NO Metal, NO CUDA"). It does *not* build
the Vulkan path — that is §2. The CPU-only library runs everywhere and is the
baseline SKU; add Vulkan only when you want GPU acceleration.

## 2. Build with the Vulkan backend (optional)

There is no plain `cargo build --target aarch64-linux-android` that "just
works": without an NDK linker the link step fails. Set the NDK toolchain
explicitly (the same pattern the `godot-crossbuild` CI workflow uses — pick the
`<host-tag>` for your build machine, e.g. `darwin-x86_64` or `linux-x86_64`),
then build `vokra-capi` with the `vulkan` feature:

```
export NDK="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host-tag>/bin"
export CC_aarch64_linux_android="$NDK/aarch64-linux-android24-clang"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"
cargo build -p vokra-capi --no-default-features --features vulkan \
  --target aarch64-linux-android --release
```

The Vulkan FFI is `dlopen`-loaded at run time (nothing is linked or bundled, so
the zero-dependency invariant `NFR-DS-02` holds); `libvulkan.so` is loaded on
device.

## 3. Load a model and call from JNI

Models are **never bundled**. Ship `libvokra.so` under `jniLibs/arm64-v8a/`,
push a GGUF into the app's private storage, and pass its real filesystem path
to the C ABI. Load the library and declare the extern:

```
class VokraBridge {
    companion object { init { System.loadLibrary("vokra") } }
    external fun transcribe(modelPath: String, wavPath: String): String
}
```

Your JNI `.c`/`.cpp` shim includes `vokra.h` and calls the same session API the
[getting-started](../getting-started.md#5-min-call-from-c-abi) C example shows
(`vokra_session_create_from_file` → `vokra_asr_transcribe` →
`vokra_string_free` → `vokra_session_destroy`). Convert the model to GGUF
offline with `vokra-cli convert`, then copy it to a path under the app's
`filesDir` (a real filesystem path so the C ABI can `mmap` it — `NFR-RL-04`).

## 4. Backend selection: Vulkan, never NNAPI

Select the backend explicitly through the C ABI's session-builder surface. A
missing Vulkan loader / device is an explicit `BackendUnavailable`, and an op
the Vulkan backend does not cover is an explicit `UnsupportedOp` — Vokra never
silently drops to CPU (`FR-EX-08`). **NNAPI is not an option** and never will
be (`FR-BE-07`; Google deprecated it with Android 15). The CPU backend remains
a fully-supported *explicit* choice.

## 5. Performance and on-device verification

The Android target is Whisper base at RTF < 0.7 (`NFR-PF-06`). That number is a
**real-hardware** measurement: this tutorial documents the build and call path,
but running the RTF on a physical device (and the Android Vulkan soak) is an
owner task — see `docs/m3-18-android-rtf-handover.md`. Do not read a passing
build as a passing RTF; the two are separate.

## 6. Troubleshooting

| symptom | cause / fix |
|---|---|
| `error: ANDROID_NDK_HOME must be set` | `build-android.sh` needs the NDK root. Install NDK r25+ and export `ANDROID_NDK_HOME`. |
| `error: linking with 'cc' failed` on a Vulkan build | The NDK linker was not configured — §2's `CC_*` / `CARGO_TARGET_*_LINKER` env vars are required. |
| `UnsatisfiedLinkError: libvokra.so` at runtime | The `.so` is missing from `jniLibs/arm64-v8a/`, or you built a different ABI. |
| `BackendUnavailable: vulkan` on device | No usable Vulkan loader/device — fall back to the CPU build *explicitly* (`--backend cpu`), never silently. |

## Next steps

- [Adding a backend](../backend-guide.md) — how the Vulkan backend is wired
- [Desktop CLI](cli.md) — the `convert` step that produces the GGUF you ship
- [Web (WASM / WebGPU)](web.md) — the browser sibling of this native path

## Keeping this page current

**Last verified: 2026-07-21 — against `scripts/build-android.sh` (CPU-only,
API 24) and the `godot-crossbuild` NDK-linker precedent.**

- **Update responsibility**: a PR that changes the Android build script, the
  NDK floor, or the Vulkan feature wiring updates this page and its Japanese
  twin in the same PR.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`); the on-device
  RTF is re-measured by the owner, not invented here.
- **Re-fetch the build facts**:

```sh
sed -n '1,40p' scripts/build-android.sh
```
