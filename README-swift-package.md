# Vokra Swift Package

The workspace release is `0.3.0` (`v0.3.0`), and the XCFramework is published
as the GitHub Release asset `Vokra.xcframework.zip`. This guide was reviewed
on 2026-10-08 against exact `main` checkout
`d100d93778191ccab77bd1c37fe3552e3d889758`; the released tag remains the
consumer pin. Current `main`'s `Package.swift` uses the pinned `v0.3.0` GitHub
binary target URL and checksum, while the released `v0.3.0` tag retains its
local XCFramework path. For local development, build the XCFramework with the
script below and use a consumer-side local package or `binaryTarget(path:)`.
Apple Silicon CPU/Metal hardware evidence is not implied by this package guide
and remains a
separate, row-scoped validation gate.

Consumer instructions for integrating Vokra into an iOS/macOS app via Swift Package Manager.

## License

Apache-2.0 (NFR-LC-01). See `LICENSE` at the repository root. The XCFramework is statically linked (NFR-RL-03) and JIT-free (NFR-RL-05).

## Add to an Xcode project

The `v0.3.0` GitHub Release publishes the XCFramework and checksum. The
released tag's `Package.swift` still points to the local XCFramework path;
current `main` contains the pinned URL/checksum target. Use the local flow
below when developing against a locally built XCFramework, or use the pinned
release target in a consumer-side manifest.

1. Clone the repository, check out the released tag, and build the local
   XCFramework:

   ```sh
   git clone https://github.com/ayutaz/vokra.git
   cd vokra
   git checkout --detach v0.3.0
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
For a released application that consumes the remote asset, use the explicit
`url`/`checksum` target below in the consumer's package manifest. Do not
replace the checksum with an unverified local or third-party artifact.

```swift
.binaryTarget(
    name: "Vokra",
    url: "https://github.com/ayutaz/vokra/releases/download/v0.3.0/Vokra.xcframework.zip",
    checksum: "fe74aeb45cc44c7fc2a1875bdd61af7d88c4ae87d2966851c88cfa4e9dcbc5a5"
)
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

Minimum platforms: iOS 15.0, macOS 12.0. The iOS build enables the Metal
feature and rejects CUDA at build time; backend availability is still checked
at runtime and unsupported work returns an explicit error rather than silently
falling back to CPU.

## Development vs Release

- **Local development / the `v0.3.0` tag** — build with
  `scripts/build-ios.sh`; the artifact lands at
  `build/ios/Vokra.xcframework`. The tag's local binary target can consume it
  from that checkout; a consumer using current `main` must use a separate
  local package reference or `.binaryTarget(path:)` because `main`'s target is
  URL-based.
- **Release asset (`v0.3.0`)** — use the explicit URL and checksum in the
  consumer-side manifest shown in the iOS tutorial. The current `main` manifest
  already contains that target; the released tag's local-path target does not
  automatically resolve the remote asset.
