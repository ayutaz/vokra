# Vokra Unity Package (`com.vokra.unity`)

Unity Package Manager (UPM) distribution of the Vokra audio inference runtime.

Vokra is an ONNX Runtime alternative specialized for speech (TTS / ASR /
Speech-to-Speech / VC / Speaker-ID / VAD). This package ships the Rust
`vokra-capi` cdylib as a Unity native plugin plus a thin C# binding layer
around the C ABI declared in `include/vokra.h`.

## Status

`0.1.0-preview.1` — M2-11 preview. Skeleton only; native binaries are
assembled by CD (see `.github/workflows/release.yml` job `unity-package`).

## Supported Unity versions

- **Minimum**: Unity 2022.3 LTS
- **Forward-compat**: Unity 6 (verified via nightly IL2CPP smoke test)

## Supported platforms

| Platform | Native lib | ABI | Feature set |
|---|---|---|---|
| macOS (Editor + Standalone) | `Plugins/macOS/libvokra.dylib` | universal2 (arm64+x86_64) | CPU (Metal opt-in via feature flag) |
| Windows (Editor + Standalone) | `Plugins/Windows/x86_64/vokra.dll` | x86_64 | CPU (CUDA opt-in, system-installed) |
| Linux (Editor + Standalone) | `Plugins/Linux/x86_64/libvokra.so` | x86_64 | CPU (CUDA opt-in, system-installed) |
| iOS (Player) | `Plugins/iOS/libvokra.a` (`__Internal`) | arm64 (device) | CPU |
| Android (Player) | `Plugins/Android/libs/arm64-v8a/libvokra.so` | arm64-v8a | CPU |

Simulator (`ios-arm64_x86_64-simulator`), 32-bit, WebGL, and other targets
are out of scope for `0.1.x`. WebGL is planned for M4-02.

## Installation

Via UPM Git URL (once repo is public):

```
https://github.com/ayutaz/vokra.git?path=/bindings/unity/com.vokra.unity
```

Via local `file:` reference (development):

```json
{
  "dependencies": {
    "com.vokra.unity": "file:../bindings/unity/com.vokra.unity"
  }
}
```

Via `.tgz` from GitHub Releases (production): download
`com.vokra.unity-<version>.tgz` and `npm`-install / drag into Package
Manager's *Add package from tarball…* dialog.

## Samples

Import the *VAD -> ASR -> TTS demo* from the Package Manager window.
Demo model weights (Silero VAD v5 MIT, Whisper base MIT, piper-plus
voice MIT) are NOT bundled; run
`Samples~/VadAsrTts/scripts/fetch-demo-models.sh` after import per
NFR-DS-04.

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
