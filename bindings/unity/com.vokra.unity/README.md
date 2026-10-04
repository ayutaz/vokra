# Vokra Unity Package (`com.vokra.unity`)

Unity Package Manager (UPM) source package for the Vokra audio inference
runtime.

Vokra is an ONNX Runtime alternative specialized for speech (TTS / ASR /
Speech-to-Speech / VC / Speaker-ID / VAD). This package contains a thin C#
binding layer around the C ABI declared in `include/vokra.h`. Native plugin
binaries are staged separately for local development or assembled by
authorized CD; they are not present in the tracked UPM tree.

## Status

The workspace and UPM metadata are released at `0.3.0` (`v0.3.0`). The UPM
registry publication remains owner-gated; the GitHub release workflow assembles
the signed package tarball and native slices. This source checkout intentionally
does not contain those binaries: its tracked `Plugins/` tree contains only
`.gitkeep`/`.meta` placeholders. The release job is the source of truth for the
assembled package (see `.github/workflows/release.yml`, job
`unity-package-release`).

## Supported Unity versions

- **Minimum**: Unity 2022.3 LTS
- **Forward-compat**: Unity 6 (verified via nightly IL2CPP smoke test)

## Supported platforms

| Platform | Native lib after local staging | ABI | Feature set |
|---|---|---|---|
| macOS (Editor + Standalone) | `Plugins/macOS/libvokra.dylib` | CI host architecture; universal2 is not claimed | CPU by default (Metal requires a feature-specific build) |
| Windows (Editor + Standalone) | `Plugins/Windows/x86_64/vokra.dll` | x86_64 | CPU (CUDA opt-in, system-installed) |
| Linux (Editor + Standalone) | `Plugins/Linux/x86_64/libvokra.so` | x86_64 | CPU (CUDA opt-in, system-installed) |
| iOS (Player) | `Plugins/iOS/libvokra.a` (`__Internal`) | arm64 (device) | CPU |
| Android (Player) | `Plugins/Android/libs/arm64-v8a/libvokra.so` | arm64-v8a | CPU |
| WebGL (Player, M4-02) | `Plugins/WebGL/libvokra.a` (`__Internal`) | wasm32 (Emscripten, simd128 off) | CPU (WASM). WebGPU is NOT wired (honest gap — ADR M4-02 §6) |

Simulator (`ios-arm64_x86_64-simulator`), 32-bit, and other targets are out
of scope for `0.1.x`.

### WebGL specifics (M4-02)

- **Load models from bytes**: `VokraAndroidAssets.ReadBytesAsync(relative)` +
  `VokraSession.CreateFromBytes(bytes)`. `CreateFromFile` fails loudly on
  WebGL (StreamingAssets are HTTP-served, and the Rust-side file loader is
  ABI-skewed under Unity-bundled Emscripten — ADR M4-02 §2). The synchronous
  `VokraAndroidAssets.EnsureLocalCopy` throws `NotSupportedException` on
  WebGL instead of deadlocking the main thread.
- **Prefer the streaming poll APIs** (`OpenVadStream` + `PushPcm`/`Poll`,
  one chunk per frame): one-shot calls (`Transcribe` / `Synthesize`) block
  the browser tab for their duration.
- **CPU (WASM) is the explicit default backend** on WebGL — not a silent
  fallback. GPU features (Metal/CUDA/Vulkan/WebGPU) are compiled out of the
  shipped `.a` (CI-audited); requesting one is an explicit error.
- **A Rust panic aborts the wasm module** (the WebGL `.a` is built with
  `panic=abort`; the `VOKRA_ERROR_PANIC` status never fires there — errors
  still report normally, but a genuine panic is a loud trap, not a hang).
- **Emscripten alignment**: the `.a` links against Unity-bundled Emscripten
  3.1.8 (Unity 2022.3) and 3.1.38 (Unity 6) — both verified via a node
  harness. A future Unity Emscripten bump surfaces as a link error at build
  time (fail-loud), not a silent break.

## Installation

### UPM Git URL (source inspection only)

```
https://github.com/ayutaz/vokra.git?path=/bindings/unity/com.vokra.unity
```

This URL can inspect the package source, but a clean Git URL import is not
runnable because the native libraries are not included yet.

### Local `file:` reference (development)

```json
{
  "dependencies": {
    "com.vokra.unity": "file:../bindings/unity/com.vokra.unity"
  }
}
```

Before opening the Unity project, clone the repository, check out the released
`v0.3.0` tag, and stage the native library
for the target platform:

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
git checkout --detach v0.3.0

# Host desktop (macOS, Linux, or Windows): stages the current host library.
scripts/build-unity-plugin.sh
# Android: requires ANDROID_NDK_HOME; stages arm64-v8a.
ANDROID_NDK_HOME=/path/to/ndk scripts/build-android.sh
# iOS: builds the XCFramework, then stages its device slice for Unity.
scripts/build-ios.sh
scripts/collect-ios-lib.sh
# WebGL: builds and stages the CPU-only wasm archive.
scripts/build-unity-webgl-lib.sh
```

Run only the helper(s) for the platform(s) you will test; each helper requires
its corresponding native SDK/toolchain. These local outputs are development
artifacts, not a release claim.

### GitHub Release tarball (production)

Download `com.vokra.unity-0.3.0.tgz` from the authorized `v0.3.0` GitHub
Release and drag it into Package Manager's *Add package from tarball…* dialog.
OpenUPM publication remains a separate owner-gated decision.

## Samples

Import the *VAD -> ASR -> TTS demo* from the Package Manager window.
Demo model weights (Silero VAD v5 MIT, Whisper base MIT, piper-plus
voice MIT) are NOT bundled, and the `v0.3.0` GitHub Release contains no GGUF
assets. Before running `Samples~/VadAsrTts/scripts/fetch-demo-models.sh`, set
all three URL environment variables to independently verified MIT sources;
release presence alone does not make the model fetches usable. See the sample
README for the required variables and license/provenance checks (NFR-DS-04).

## License and third-party notices

- Package source: Apache-2.0 (`LICENSE.md`).
- Third-party attributions and CUDA-runtime non-bundling policy:
  see `NOTICE`.
- Model licenses vary; the sample uses MIT-licensed weights only.
  CC-BY-NC / CC-BY-NC-SA / non-commercial weights (F5-TTS, Fish-Speech,
  Bark, EnCodec) are excluded from official distributions per
  M2-13 compliance gate.

## Not bundled: NVIDIA CUDA runtime

Per NVIDIA CUDA EULA ("installed only in a private (non-shared)
directory location"), this package does NOT ship `cudart` / `cudnn` /
`cublas`. When CUDA acceleration is enabled at runtime, Vokra loads
`libcuda.so` / `nvcuda.dll` from the system install via `dlopen`.

See `NOTICE` for the full statement and CI enforcement
(`scripts/check-unity-package-no-nvidia.sh`).
