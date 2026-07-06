// swift-tools-version:5.9
// Vokra Swift Package — M2-02 iOS build scaffold.
// See docs/adr/000N-ios-build.md for the design decisions (static-only,
// XCFramework with device + simulator slices, Metal ON / CUDA OFF for iOS).
// License: Apache-2.0 (NFR-LC-01).
import PackageDescription

let package = Package(
    name: "Vokra",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        .library(name: "Vokra", targets: ["Vokra"]),
    ],
    targets: [
        // Local dev / CI: consumes the XCFramework produced by
        // scripts/build-ios.sh into build/ios/Vokra.xcframework.
        .binaryTarget(
            name: "Vokra",
            path: "build/ios/Vokra.xcframework"
        ),
        // Release path (T12 / CD switches Package.swift to the URL form):
        // .binaryTarget(
        //     name: "Vokra",
        //     url: "https://github.com/ayutaz/vokra/releases/download/<tag>/Vokra.xcframework.zip",
        //     checksum: "<sha256>"
        // ),
    ]
)
