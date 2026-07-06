# iOS + Swift Package tutorial

**English** | [日本語](ios.ja.md)

Vokra ships an iOS-ready **XCFramework** wrapped in a
[`Package.swift`](../../Package.swift) Swift Package. This tutorial covers
getting from `git clone` to a running Vokra call on device.

## 1. Prerequisites

- **Xcode 14 or newer** on macOS.
- **iOS 15+** target device or Simulator (iOS 14 and older is out of scope).
- Apple Developer signing profile for device deployment.
- The Vokra repository (or a tagged release URL).

## 2. Build the XCFramework

Two supported paths:

### Path A — from source (local dev)

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
scripts/build-ios.sh
# Produces build/ios/Vokra.xcframework with two slices:
#   - iOS device (arm64)
#   - iOS Simulator (arm64 + x86_64)
```

Confirm slices with:

```sh
scripts/verify-ios-xcframework.sh build/ios/Vokra.xcframework
```

### Path B — release download

CD publishes `Vokra.xcframework.zip` + its SHA-256 as a GitHub Release
asset. Update `Package.swift` to the URL form (the file already has the
template commented out):

```swift
.binaryTarget(
    name: "Vokra",
    url: "https://github.com/ayutaz/vokra/releases/download/<tag>/Vokra.xcframework.zip",
    checksum: "<sha256>"
)
```

## 3. Add Vokra to an iOS app

In Xcode:

1. **File → Add Package Dependencies → Add Local** and select the repo
   root (Path A), or paste the release URL (Path B).
2. Attach the **`Vokra`** product to your app target.
3. Add a GGUF (`whisper-base.gguf`) and a WAV fixture
   (`tests/fixtures/audio/jfk-30s.wav`) to the app's bundled Resources
   with **Copy items if needed** + **Add to target** both checked.
4. **Signing & Capabilities**: set your Team and a unique Bundle ID.
5. **Build Settings** → set **Enable Bitcode = No** (Bitcode is
   deprecated by Apple; the XCFramework is not bitcode-bundled).

## 4. Minimal SwiftUI usage

```swift
import SwiftUI
import Vokra   // Clang module re-exports all `vokra_*` C symbols.
import Foundation

struct ContentView: View {
    @State private var status = "Idle"
    var body: some View {
        VStack(spacing: 20) {
            Text(status).font(.system(.body, design: .monospaced))
            Button("Transcribe") { transcribe() }
        }.padding()
    }

    func transcribe() {
        guard
            let gguf = Bundle.main.path(forResource: "whisper-base", ofType: "gguf"),
            let wav  = Bundle.main.path(forResource: "jfk-30s",     ofType: "wav")
        else {
            status = "resource missing"; return
        }

        var session: OpaquePointer?
        let rc = vokra_session_create_from_file(gguf, &session)
        guard rc == 0, let s = session else {
            status = "session_create err=\(rc)"; return
        }
        defer { vokra_session_destroy(s) }

        // See docs/m2-14-ios-rtf-handover.md for a full end-to-end RTF
        // measurement harness (fixture WAV -> transcribe -> record RTF).
        var out: UnsafeMutablePointer<CChar>?
        let arc = vokra_asr_transcribe(s, wav, &out)
        let text = out.map { String(cString: $0) } ?? "<null>"
        if let o = out { vokra_string_free(o) }
        status = "rc=\(arc): \(text.prefix(120))"
    }
}
```

The `vokra_*` C symbols are exact matches for `include/vokra.h` in the
XCFramework's `Headers/`. The Clang module `Vokra` re-exports all of them.

## 5. iOS-specific invariants

The Vokra iOS build enforces the platform's static-linking / no-JIT rules:

- **Static-only** — dynamic library loading is prohibited on iOS
  (NFR-RL-03). The XCFramework is `.a` slices; call sites use
  `DllImport("__Internal")` on the Unity side, and the Swift side gets the
  same symbols through the Clang module.
- **No JIT / runtime code generation** — iOS enforces W^X. Vokra uses
  runtime CPU dispatch only (NFR-RL-05); no JIT ships.
- **CUDA is a compile error on iOS** — a `compile_error!(...)` in
  `vokra-backend-cuda` makes this explicit rather than silently linking a
  CUDA-shaped surface that cannot run.
- **Metal is the accelerated path** — the CPU path (NEON on Apple
  Silicon) is always available; opt into Metal via
  `vokra_session_set_backend(s, VOKRA_BACKEND_METAL)` (see `vokra.h` for
  the exact enum name). Per FR-EX-08, unsupported ops on Metal surface as
  explicit errors, never silent CPU fallback.

## 6. On-device RTF measurement

For the NFR-PF-03 measurement (Whisper base **RTF < 0.5** on real
hardware — the v0.5 Exit criterion), follow the standalone handover
document:

- [`docs/m2-14-ios-rtf-handover.md`](../m2-14-ios-rtf-handover.md)

It contains: the SwiftUI measurement app, the recording template
(backend / device / iOS version / elapsed / audio / RTF), the R4
"Metal-probe-fails-on-iPhone" boundary case, and the deliverable
checklist.

## 7. Troubleshooting

- **`module 'Vokra' not found`**: SwiftPM cannot resolve the binary
  target. Confirm `build/ios/Vokra.xcframework` exists (Path A) or that
  the URL + checksum in `Package.swift` are current (Path B).
- **`Undefined symbol: _vokra_...`**: the XCFramework you have is
  probably too old — regenerate with `scripts/build-ios.sh` from a
  matching commit.
- **App size**: the release XCFramework is `.a`-based; App Store slicing
  should keep only the target-slice symbols in the final `.ipa`. If size
  is still too large, consider a K-quantized GGUF (`--quantize q4_k` in
  `vokra-cli convert`).

## Next steps

- **Simulator vs device RTF**: Simulator numbers are **not** valid for
  NFR-PF-03; use physical hardware.
- **Metal validation**: pair with the on-macOS Metal path; both paths
  share the same Rust runtime and the same GGUF, so a bug seen on iPhone
  will usually reproduce on macOS with lower turnaround.
- **Migration**: if you are coming from `onnxruntime-swift` or a Core ML
  pipeline, see [Migration Guide](../migration-guide.md).
