# Vokra Swift Package

The workspace is `0.2.0` development as of 2026-08-30. No Git tag or GitHub
Release is available yet; the package is consumed from repository source or a
locally built XCFramework until an authorized release is published.

Consumer instructions for integrating Vokra into an iOS/macOS app via Swift Package Manager.

## License

Apache-2.0 (NFR-LC-01). See `LICENSE` at the repository root. The XCFramework is statically linked (NFR-RL-03) and JIT-free (NFR-RL-05).

## Add to an Xcode project

There is no tagged release or release asset yet. The repository's
`Package.swift` currently points at the locally generated
`build/ios/Vokra.xcframework`, so a remote repository dependency cannot resolve
from a clean clone. Use the local flow below until an authorized CD release
publishes the XCFramework.

1. Clone the repository and check out the publicly fetchable GitHub `main`
   baseline verified on 2026-08-30:

   ```sh
   git clone https://github.com/ayutaz/vokra.git
   cd vokra
   git checkout --detach 41ce9ffdd4b0959497f55afa5016822f77a8a7b6
   scripts/build-ios.sh
   ```

2. In Xcode, select **File → Add Package Dependencies… → Add Local…** and
   choose that checkout, or add the generated
   `build/ios/Vokra.xcframework` directly to the app project.
3. Select the `Vokra` library product and add it to your app target.

For an app managed by its own `Package.swift`, use a local package dependency
that points at the checkout containing the generated XCFramework:

```swift
dependencies: [
    .package(path: "../vokra")
],
targets: [
    .target(name: "MyApp", dependencies: [.product(name: "Vokra", package: "vokra")])
]
```

The `../vokra` path is illustrative; adjust it to the checkout created above.
The exact revision is selected by the `git checkout` step, not by a currently
usable remote SwiftPM dependency. After CD publishes an authorized release
asset and patches the package's binary target to its URL and checksum, a
versioned remote dependency can be documented here.

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
- **Release (future, after authorized CD)** — `Package.swift` may be patched to
  a `.binaryTarget` URL and checksum for a GitHub Release asset. Consumers can
  pin the tag once that release exists; there is no current release asset.
