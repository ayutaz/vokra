# Physical iPhone benchmark harness

This directory is the executable handoff for issue #52. It does not make a
Simulator result look like an iPhone result and it does not ship invented
numbers.

## Before opening Xcode

1. Build `build/ios/Vokra.xcframework` from the exact branch SHA and run
   `scripts/verify-ios-xcframework.sh`.
2. Create an iOS 15+ SwiftUI app, add this repository as a local package, and
   attach the `Vokra` product.
3. Add `IOSDeviceBench.swift`, `whisper-base.gguf`, `mimi.gguf`, and
   `tests/fixtures/audio/jfk-30s.wav` to the app target. The codec GGUF requires
   issue #48's `vokra_codec_decoder_*` ABI.
4. Set a signing Team and a unique bundle identifier. Connect, unlock, and
   trust a physical iPhone. Simulator output is not admissible.
5. Fill `IOSBenchConfiguration` with `git rev-parse HEAD`, the exact device
   model, measured ambient temperature, screen/charging/case conditions. iOS
   exposes only `ProcessInfo.thermalState`, not an internal temperature in °C;
   the harness records that categorical starting and ending state.

Call the blocking methods from a background task so the UI remains responsive:

```swift
let config = IOSBenchConfiguration(
    buildSHA: "<40-hex HEAD>",
    deviceModel: "<marketing model + hw identifier>",
    ambientTemperatureC: 23.0,
    screen: "on",
    charging: false,
    caseState: "removed"
)
let bench = try IOSDeviceBench(configuration: config)

Task.detached {
    let whisper = try bench.measureWhisper(
        modelURL: Bundle.main.url(forResource: "whisper-base", withExtension: "gguf")!,
        wavURL: Bundle.main.url(forResource: "jfk-30s", withExtension: "wav")!,
        backend: .cpu
    )
    print("Whisper report: \(whisper.1.path)")

    let codec = try bench.measureSustainedCodec(
        modelURL: Bundle.main.url(forResource: "mimi", withExtension: "gguf")!
    )
    print("Codec JSONL: \(codec.logURL.path)")
}
```

Run CPU and Metal Whisper measurements as separate cold launches. A Metal
construction/inference error is recorded as an error, not replaced with a CPU
number. The Whisper run fixes 3 warmups and 10 measured iterations over the
exact 30-second fixture and records median RTF plus process-lifetime peak RSS.

The codec run is paced at `frame_hop / sample_rate` for 1800 seconds. It stores
frame data in pre-reserved memory and writes JSONL only after timing, so file I/O
is outside per-frame latency. Export the report from the app's Documents folder
and validate it on the Mac:

```sh
uv run --project tools/parity --python 3.12 python tools/bench/ios_sustained_analyze.py \
  vokra-ios-codec-sustained-....jsonl \
  --json-output sustained-report.json \
  --markdown-output sustained-report.md
```

The analyzer refuses a shortened/sparse/non-contiguous run. Its Markdown gives
p50/p95/p99, frame-deadline misses, peak RSS, thermal transition, and the exact
last-decile/first-decile p50 ratio. That ratio is descriptive; no unapproved
performance threshold is invented.
