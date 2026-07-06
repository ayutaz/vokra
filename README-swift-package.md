# Vokra Swift Package

Consumer instructions for integrating Vokra into an iOS/macOS app via Swift Package Manager.

## License

Apache-2.0 (NFR-LC-01). See `LICENSE` at the repository root. The XCFramework is statically linked (NFR-RL-03) and JIT-free (NFR-RL-05).

## Add to an Xcode project

1. In Xcode, open your project and select **File → Add Package Dependencies…**.
2. Enter this repository URL (or the local path during development).
3. Select the `Vokra` library product and add it to your app target.

Alternatively, add to your own `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ayutaz/vokra.git", from: "0.5.0")
],
targets: [
    .target(name: "MyApp", dependencies: [.product(name: "Vokra", package: "vokra")])
]
```

## Usage

Vokra exposes a C ABI via the `Vokra` Clang module. From Swift:

```swift
import Vokra

var session: OpaquePointer?
let rc = vokra_session_create_from_file("whisper-base.gguf", &session)
guard rc == 0, let s = session else { fatalError("vokra init failed: \(rc)") }
defer { vokra_session_destroy(s) }
// ... call vokra_asr_transcribe / vokra_tts_synthesize etc.
```

Minimum platforms: iOS 15.0, macOS 12.0. Metal backend is enabled by default; CUDA is unavailable on iOS by design (see `docs/adr/` iOS build ADR).

## Development vs Release

- **Local dev / CI** — `Package.swift` uses `.binaryTarget(name: "Vokra", path: "build/ios/Vokra.xcframework")`. Build the XCFramework locally with `scripts/build-ios.sh`; the artifact lands at `build/ios/Vokra.xcframework`.
- **Release (post-CD, T12)** — `Package.swift` is patched to `.binaryTarget(name: "Vokra", url: "https://github.com/…/Vokra.xcframework.zip", checksum: "<sha256>")` referencing a GitHub Release asset. Consumers pin to a tagged version; no local build required.
