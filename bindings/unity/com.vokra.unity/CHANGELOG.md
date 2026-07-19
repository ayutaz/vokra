# Changelog

All notable changes to `com.vokra.unity` are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [SemVer](https://semver.org/spec/v2.0.0.html) with UPM preview
suffix conventions (`-preview.N` for pre-1.0 iterations).

## [Unreleased]

### Added
- M4-02: **WebGL target** — `Plugins/WebGL/libvokra.a`
  (wasm32-unknown-emscripten staticlib, `DllImport("__Internal")`, CPU/WASM,
  simd128 off; assembled by CD like every other native slice).
- M4-02: `VokraSession.CreateFromBytes(byte[])` — in-memory GGUF session
  create (`vokra_session_create_from_bytes`); the model path on WebGL and a
  general-purpose alternative everywhere.
- M4-02: `VokraAndroidAssets.ReadBytesAsync` — StreamingAssets bytes fetch
  on every platform (WebGL/Android via UnityWebRequest); pairs with
  `CreateFromBytes`.
- M4-02: `Samples~/VadAsrTts` WebGL path — main-thread async pipeline
  (`PipelineRunner.RunWebGlAsync`), VAD streaming poll one chunk per frame,
  browser-console PASS/FAIL markers for the nightly headless smoke.
- M2-11-T02: UPM package skeleton (`package.json`, `Runtime/` and
  `Editor/` assembly definitions, empty per-platform `Plugins/`
  subtree, empty `Samples~/VadAsrTts` slot).

### Changed
- M4-02: `VokraAndroidAssets.EnsureLocalCopy` (sync) now throws
  `NotSupportedException` on WebGL — a synchronous busy-wait would deadlock
  the browser main thread (loud fail instead of hang, FR-EX-08).
  `EnsureLocalCopyAsync` gains a WebGL branch whose returned path is valid
  for managed (C#) file IO only — pass models to `CreateFromBytes` instead.

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
