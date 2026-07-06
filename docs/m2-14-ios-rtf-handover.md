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
- [ ] Fixture WAV (16 kHz mono, 30 s) — recommend `tests/fixtures/audio/jfk-30s.wav` from the repo, bundled as an app resource.
- [ ] Physical iOS 15+ device (iPhone or iPad); Simulator RTF is NOT valid for NFR-PF-03.
- [ ] Apple Developer signing profile for on-device deployment.

## 2. Xcode project setup

1. Xcode → File → New → Project → **iOS App** (SwiftUI, Swift, iOS 15.0 min).
2. File → Add Package Dependencies → **Add Local** → point at repo root (or paste the release URL). Product `Vokra` → attach to app target.
3. Drag `whisper-base.gguf` and `jfk-30s.wav` into the target; verify "Copy items if needed" + "Add to target" both checked.
4. Signing & Capabilities → set Team; Bundle ID unique to依頼者.
5. Build Settings → **Enable Bitcode = No** (Bitcode is deprecated by Apple; the XCFramework is not bitcode-bundled).

## 3. Minimal measurement app (SwiftUI)

```swift
import SwiftUI
import Vokra
import Foundation

struct ContentView: View {
    @State private var result: String = "Idle"
    var body: some View {
        VStack(spacing: 20) {
            Text(result).font(.system(.body, design: .monospaced))
            Button("Run RTF Measurement") { measure() }
        }.padding()
    }
    func measure() {
        guard let gguf = Bundle.main.path(forResource: "whisper-base", ofType: "gguf"),
              let wav  = Bundle.main.path(forResource: "jfk-30s",     ofType: "wav")
        else { result = "resource missing"; return }
        var session: OpaquePointer?
        let rc = vokra_session_create_from_file(gguf, &session)
        guard rc == 0, let s = session else { result = "session_create err=\(rc)"; return }
        defer { vokra_session_destroy(s) }

        let audioSec: Double = 30.0
        let t0 = Date()
        var out: UnsafeMutablePointer<CChar>?
        let arc = vokra_asr_transcribe(s, wav, &out)
        let elapsed = Date().timeIntervalSince(t0)
        let rtf = elapsed / audioSec
        let text = out.map { String(cString: $0) } ?? "<null>"
        if let o = out { vokra_string_free(o) }
        result = "rc=\(arc) elapsed=\(String(format: "%.3f", elapsed))s RTF=\(String(format: "%.4f", rtf))\n\(text.prefix(60))"
    }
}
```

Notes:
- Exact C symbol names may differ; consult `include/vokra.h` in the XCFramework `Headers/` and adjust. The Clang module `Vokra` re-exports all `vokra_*` symbols.
- Run 3 warm iterations, then record the median of 5 timed runs (drop min/max).
- Fix CPU governor by keeping device plugged in + screen on; disable Low Power Mode.

## 4. Recording template

| Run | Backend | Device model | iOS version | Elapsed (s) | Audio (s) | RTF | NFR-PF-03 (<0.5) |
|-----|---------|--------------|-------------|-------------|-----------|-----|------------------|
| 1   | CPU     |              |             |             | 30.0      |     | pass / fail      |
| 2   | Metal   |              |             |             | 30.0      |     | pass / fail      |

Also record: Vokra `git rev-parse HEAD`, XCFramework SHA256, Xcode version, thermal state (`ProcessInfo.processInfo.thermalState`), whether device was plugged in.

## 5. Backend selection

Backend defaults to CPU. To try Metal, invoke `vokra_session_set_backend(s, VOKRA_BACKEND_METAL)` before `vokra_asr_transcribe` (see `vokra.h` for the exact enum name). Per FR-EX-08, unsupported ops on Metal must surface as **explicit errors** — never silent CPU fallback. Log the returned rc.

## 6. R4 boundary — Metal probe failure on iPhone

If Metal init fails on iPhone (e.g., `MTLGPUFamily.Apple7` not recognized because M2-01's device probe was written against macOS `MTLGPUFamily.Mac*` families), do NOT patch here. Actions:

1. CPU-only RTF is still measurable — record it and mark Metal row as "blocked-by-M2-01".
2. File a defect against **M2-01** with: XCFramework SHA256, iPhone model + iOS version, exact `NSError` from Metal init, and the `MTLDevice.supportsFamily(_:)` results.
3. Rerun this measurement once M2-01 ships an iOS GPU-family fix.

CPU pass alone satisfies NFR-PF-03 for this WP if RTF < 0.5; Metal is a separate row.

## 7. Handover deliverable back to Vokra

Attach to the M2-14 completion ticket: the filled table above, raw logs, and the device video (optional) showing the run. If any RTF ≥ 0.5, open a perf ticket referencing this handover; do NOT close M2-14 as pass.
