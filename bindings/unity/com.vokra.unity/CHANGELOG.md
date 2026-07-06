# Changelog

All notable changes to `com.vokra.unity` are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [SemVer](https://semver.org/spec/v2.0.0.html) with UPM preview
suffix conventions (`-preview.N` for pre-1.0 iterations).

## [Unreleased]

### Added
- M2-11-T02: UPM package skeleton (`package.json`, `Runtime/` and
  `Editor/` assembly definitions, empty per-platform `Plugins/`
  subtree, empty `Samples~/VadAsrTts` slot).

## [0.1.0-preview.1] — TBD

Initial preview release. M2-11 workstream.

### Added
- Package layout at `bindings/unity/com.vokra.unity/`.
- `Vokra.Runtime` and `Vokra.Editor` assembly definitions.
- iOS PostProcessBuild hook wiring `libvokra.a`, `Metal.framework`,
  `Accelerate.framework` into the exported Xcode project.
- Apache-2.0 license and NVIDIA-runtime non-bundling `NOTICE`.

### Not included
- Native binaries (`libvokra.dylib` / `vokra.dll` / `libvokra.so` /
  `libvokra.a`) — populated by CD, not committed to git.
- Model weights — fetched per-sample via
  `Samples~/VadAsrTts/scripts/fetch-demo-models.sh`.
- CUDA runtime — resolved via `dlopen` at runtime per NVIDIA EULA.

[Unreleased]: https://github.com/ayutaz/vokra/compare/v0.1.0-preview.1...HEAD
[0.1.0-preview.1]: https://github.com/ayutaz/vokra/releases/tag/v0.1.0-preview.1
