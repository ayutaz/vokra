# Vokra Swift Package

The workspace release is `0.3.0` (`v0.3.0`), and the XCFramework is published
as the GitHub Release asset `Vokra.xcframework.zip`. This guide was reviewed
on 2026-10-04 against the local documentation head `3a3fd822`; that checkout
is a working branch and is not a replacement for the released tag. The tagged
source's `Package.swift` still uses the local XCFramework path, so a clean tag
checkout does not automatically resolve the remote asset. Apple Silicon
CPU/Metal hardware evidence is not implied by this package guide and remains a
separate, row-scoped validation gate.

Consumer instructions for integrating Vokra into an iOS/macOS app via Swift Package Manager.

## License

Apache-2.0 (NFR-LC-01). See `LICENSE` at the repository root. The XCFramework is statically linked (NFR-RL-03) and JIT-free (NFR-RL-05).

## Add to an Xcode project

The `v0.3.0` GitHub Release publishes the XCFramework and checksum. The tagged
source manifest uses a local binary target, so use the local flow below for a
clean tag checkout, or place the explicit URL/checksum target shown below in a
consumer-side manifest when consuming the release asset.

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

- **Local dev / the `v0.3.0` tag** — `Package.swift` uses
  `.binaryTarget(name: "Vokra", path: "build/ios/Vokra.xcframework")`.
  Build it with `scripts/build-ios.sh`; the artifact lands at
  `build/ios/Vokra.xcframework`.
- **Release asset (`v0.3.0`)** — use the explicit URL and checksum in the
  consumer-side manifest shown in the iOS tutorial. The GitHub asset is
  published, but the tag's source manifest remains local-path based.
