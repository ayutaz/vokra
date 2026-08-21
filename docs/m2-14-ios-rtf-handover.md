# M2-14 iPhone RTF Measurement — Handover

**Owner**: 依頼者 (physical iPhone required; CC cannot execute this WP).
**Predecessor**: M2-02 (iOS build scaffold) produces `Vokra.xcframework` + `Package.swift`.
**Requirement under measurement**: NFR-PF-03 (Whisper base RTF < 0.5 on target device).

> **Explicit boundary**: Whether NFR-PF-03 RTF < 0.5 is met by CPU (NEON) or Metal (GPU) is determined by measurement here; this WP does not assert it. M2-02 only delivers the build artifact and Swift Package wiring.

## 1. Prerequisites checklist

- [ ] `Vokra.xcframework` built from tagged Vokra release (record `git rev-parse HEAD` used at build).
- [ ] `Package.swift` reachable — either local path `.binaryTarget(path: "build/ios/Vokra.xcframework")` (dev) or release URL + SHA256 (`.binaryTarget(url:, checksum:)`).
- [ ] Xcode 14 or newer installed on macOS host (matches CI floor from M2-02 ADR).
- [ ] Whisper base GGUF model file (`whisper-base.gguf`) converted via `vokra-cli convert` and bundled in the app target's Resources.
- [ ] Mimi GGUF model file (`mimi.gguf`) for the 30-minute streaming-codec run; build from a revision containing issue #48's `vokra_codec_decoder_*` ABI.
- [ ] Fixture WAV (16 kHz mono, 30 s) — recommend `tests/fixtures/audio/jfk-30s.wav` from the repo, bundled as an app resource.
- [ ] Physical iOS 15+ device (iPhone or iPad); Simulator RTF is NOT valid for NFR-PF-03.
- [ ] Apple Developer signing profile for on-device deployment.

## 2. Xcode project setup

1. Xcode → File → New → Project → **iOS App** (SwiftUI, Swift, iOS 15.0 min).
2. File → Add Package Dependencies → **Add Local** → point at repo root (or paste the release URL). Product `Vokra` → attach to app target.
3. Drag `whisper-base.gguf` and `jfk-30s.wav` into the target; verify "Copy items if needed" + "Add to target" both checked.
4. Signing & Capabilities → set Team; Bundle ID unique to依頼者.
5. Build Settings → **Enable Bitcode = No** (Bitcode is deprecated by Apple; the XCFramework is not bitcode-bundled).

## 3. Measurement app

Use the checked and iPhoneOS-typechecked harness at
[`tools/bench/ios-device/IOSDeviceBench.swift`](../tools/bench/ios-device/IOSDeviceBench.swift)
and follow its [README](../tools/bench/ios-device/README.md). It fixes two
defects in the historical snippet that used to live here:

- `vokra_asr_transcribe` accepts mono `f32` PCM plus its sample count/rate; it
  does **not** accept a WAV path. The harness decodes the WAV before timing and
  passes exactly 480,000 samples at 16 kHz.
- Backend selection happens on `vokra_session_options_t` before session
  construction; there is no `vokra_session_set_backend` mutation API.

The harness performs 3 warm iterations then 10 measured iterations, reports
P50 RTF, and samples Darwin `ru_maxrss` (process-lifetime peak RSS, so a peak
inside the blocking inference call cannot be missed). It hashes the model and
fixture before loading the session and writes a JSON report to the app's
Documents directory.

For the codec leg it opens a `vokra_codec_decoder_t`, derives the real
`frame_hop`, sample rate, and codebook count from the GGUF, then paces one valid
code frame at the model frame rate for 1,800 seconds. Export the JSONL and run:

```sh
uv run --project tools/parity --python 3.12 python tools/bench/ios_sustained_analyze.py \
  vokra-ios-codec-sustained-....jsonl \
  --json-output sustained-report.json \
  --markdown-output sustained-report.md
```

The analyzer fails closed on a short, sparse, non-contiguous, non-finite, or
conditions-incomplete log. Its report contains p50/p95/p99, peak RSS, deadline
misses, thermal-state transition, and the exact last-decile/first-decile p50
ratio. It does not invent a pass threshold for degradation.

## 4. Recording template

| Run | Backend | Device model | iOS version | Elapsed P50 (s) | Audio (s) | RTF P50 | Peak RSS | Build SHA | NFR-PF-03 (<0.5) |
|-----|---------|--------------|-------------|-----------------|-----------|---------|----------|-----------|------------------|
| 1   | CPU     |              |             |                 | 30.0      |         |          |           | pass / fail      |
| 2   | Metal   |              |             |                 | 30.0      |         |          |           | pass / fail      |

| Codec model | Backend | Duration | Frames | p50 (ms) | p95 (ms) | p99 (ms) | Deadline misses | Peak RSS | First→last decile p50 | Thermal start→end |
|-------------|---------|----------|--------|----------|----------|----------|-----------------|----------|-----------------------|-------------------|
| Mimi        | CPU     | 1800 s   |        |          |          |          |                 |          |                       |                   |

Also record: XCFramework SHA256, Xcode version, model SHA256, fixture SHA256,
ambient temperature, starting thermal state, screen on/off, charging or not,
and case installed/removed. iOS does not expose internal device temperature in
degrees through a public API, so record `ProcessInfo.processInfo.thermalState`
instead and do not fabricate a Celsius value.

## 5. Backend selection

Backend defaults to CPU. To try Metal, create `vokra_session_options_t`, call
`vokra_session_options_set_backend(opts, VOKRA_BACKEND_METAL)`, then construct
the model with `vokra_session_create_from_file_with_options`. Destroy the
options after construction. Per FR-EX-08, unavailable/unsupported Metal must
surface as an **explicit error** — never silent CPU fallback. Log the returned
status and `vokra_last_error()`.

## 6. R4 boundary — Metal probe failure on iPhone

If Metal init fails on iPhone (e.g., `MTLGPUFamily.Apple7` not recognized because M2-01's device probe was written against macOS `MTLGPUFamily.Mac*` families), do NOT patch here. Actions:

1. CPU-only RTF is still measurable — record it and mark Metal row as "blocked-by-M2-01".
2. File a defect against **M2-01** with: XCFramework SHA256, iPhone model + iOS version, exact `NSError` from Metal init, and the `MTLDevice.supportsFamily(_:)` results.
3. Rerun this measurement once M2-01 ships an iOS GPU-family fix.

CPU pass alone satisfies NFR-PF-03 for this WP if RTF < 0.5; Metal is a separate row.

## 7. Handover deliverable back to Vokra

Attach to the M2-14 completion ticket: the filled tables above, raw Whisper
JSON, raw codec JSONL, analyzer JSON/Markdown, and the device video (optional)
showing the run. If any RTF ≥ 0.5, open a perf ticket referencing this
handover; do NOT close M2-14 as pass.
