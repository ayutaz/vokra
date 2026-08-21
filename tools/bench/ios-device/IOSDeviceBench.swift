import AVFoundation
import CryptoKit
import Darwin
import Foundation
import UIKit
import Vokra

enum IOSBenchError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case resource(String)
    case vokra(String)
    case io(String)

    var description: String {
        switch self {
        case .invalidConfiguration(let message), .resource(let message),
             .vokra(let message), .io(let message):
            return message
        }
    }
}

enum IOSBenchBackend: String, Codable {
    case cpu
    case metal

    var abiValue: Int32 {
        switch self {
        case .cpu:
            return Int32(VOKRA_BACKEND_CPU.rawValue)
        case .metal:
            return Int32(VOKRA_BACKEND_METAL.rawValue)
        }
    }
}

/// Values the device cannot infer honestly. Fill these immediately before the
/// run; the validator rejects placeholders and missing conditions.
struct IOSBenchConfiguration {
    let buildSHA: String
    let deviceModel: String
    let ambientTemperatureC: Double
    let screen: String       // "on" or "off"
    let charging: Bool
    let caseState: String    // "installed" or "removed"

    func validate() throws {
        let hex = CharacterSet(charactersIn: "0123456789abcdefABCDEF")
        guard buildSHA.count == 40,
              buildSHA.unicodeScalars.allSatisfy(hex.contains)
        else {
            throw IOSBenchError.invalidConfiguration(
                "buildSHA must be the exact 40-hex git rev-parse HEAD"
            )
        }
        guard !deviceModel.isEmpty else {
            throw IOSBenchError.invalidConfiguration("deviceModel must be recorded")
        }
        guard ambientTemperatureC.isFinite else {
            throw IOSBenchError.invalidConfiguration("ambientTemperatureC must be finite")
        }
        guard screen == "on" || screen == "off" else {
            throw IOSBenchError.invalidConfiguration("screen must be on/off")
        }
        guard caseState == "installed" || caseState == "removed" else {
            throw IOSBenchError.invalidConfiguration("caseState must be installed/removed")
        }
    }
}

struct WhisperRun: Codable {
    let backend: String
    let elapsedS: Double
    let audioS: Double
    let rtf: Double
    let peakRSSBytes: UInt64
    let thermalState: String
}

struct WhisperReport: Codable {
    let schema: String
    let buildSHA: String
    let modelSHA256: String
    let fixtureSHA256: String
    let deviceModel: String
    let iosVersion: String
    let ambientTemperatureC: Double
    let startingThermalState: String
    let screen: String
    let charging: Bool
    let caseState: String
    let warmupRuns: Int
    let measuredRuns: Int
    let runs: [WhisperRun]
    let elapsedP50S: Double
    let rtfP50: Double
    let peakRSSBytes: UInt64
}

struct CodecSummary {
    let logURL: URL
    let frameCount: Int
    let durationS: Double
    let p50MS: Double
    let p95MS: Double
    let p99MS: Double
    let peakRSSBytes: UInt64
}

private struct CodecFrame {
    let index: Int
    let wallElapsedS: Double
    let decodeMS: Double
    let peakRSSBytes: UInt64
    let thermalState: String
}

@available(iOS 15.0, *)
final class IOSDeviceBench {
    static let whisperWarmups = 3
    static let whisperMeasuredRuns = 10
    static let sustainedDurationS = 1_800.0

    private let configuration: IOSBenchConfiguration

    init(configuration: IOSBenchConfiguration) throws {
        try configuration.validate()
        self.configuration = configuration
    }

    /// Measures one backend. WAV decode, model SHA, session creation, and
    /// warmups are deliberately outside the timed region. `ru_maxrss` is the
    /// Darwin process-lifetime peak and therefore cannot miss a transient peak
    /// inside the blocking C call.
    func measureWhisper(
        modelURL: URL,
        wavURL: URL,
        backend: IOSBenchBackend
    ) throws -> (WhisperReport, URL) {
        let modelSHA = try sha256(modelURL)
        let fixtureSHA = try sha256(wavURL)
        let pcm = try loadMonoFloat32WAV(wavURL, expectedRate: 16_000)
        guard pcm.count == 30 * 16_000 else {
            throw IOSBenchError.resource(
                "Whisper fixture must contain exactly 480000 mono samples; got \(pcm.count)"
            )
        }
        let session = try openSession(modelURL: modelURL, backend: backend)
        defer { vokra_session_destroy(session) }

        let startingThermal = thermalState()
        for _ in 0..<Self.whisperWarmups {
            _ = try transcribe(session: session, pcm: pcm)
        }

        var runs: [WhisperRun] = []
        runs.reserveCapacity(Self.whisperMeasuredRuns)
        for _ in 0..<Self.whisperMeasuredRuns {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try transcribe(session: session, pcm: pcm)
            let elapsed = secondsSince(start)
            runs.append(
                WhisperRun(
                    backend: backend.rawValue,
                    elapsedS: elapsed,
                    audioS: 30.0,
                    rtf: elapsed / 30.0,
                    peakRSSBytes: peakRSSBytes(),
                    thermalState: thermalState()
                )
            )
        }
        let elapsedP50 = percentile(runs.map(\.elapsedS), 0.50)
        let report = WhisperReport(
            schema: "vokra.ios-whisper-rtf.v1",
            buildSHA: configuration.buildSHA.lowercased(),
            modelSHA256: modelSHA,
            fixtureSHA256: fixtureSHA,
            deviceModel: configuration.deviceModel,
            iosVersion: UIDevice.current.systemVersion,
            ambientTemperatureC: configuration.ambientTemperatureC,
            startingThermalState: startingThermal,
            screen: configuration.screen,
            charging: configuration.charging,
            caseState: configuration.caseState,
            warmupRuns: Self.whisperWarmups,
            measuredRuns: Self.whisperMeasuredRuns,
            runs: runs,
            elapsedP50S: elapsedP50,
            rtfP50: elapsedP50 / 30.0,
            peakRSSBytes: runs.map(\.peakRSSBytes).max() ?? peakRSSBytes()
        )
        let url = try writeJSON(report, stem: "vokra-ios-whisper-\(backend.rawValue)")
        return (report, url)
    }

    /// Runs one valid all-zero code frame at the model's real frame rate for
    /// 30 minutes. File I/O happens only after timing completes; samples are
    /// pre-reserved, and push+pull timing excludes RSS/thermal sampling and
    /// pacing sleep. Analyze the JSONL with ios_sustained_analyze.py.
    ///
    /// This method requires the `vokra_codec_decoder_*` ABI from issue #48.
    func measureSustainedCodec(modelURL: URL) throws -> CodecSummary {
        let modelSHA = try sha256(modelURL)
        let session = try openSession(modelURL: modelURL, backend: .cpu)
        defer { vokra_session_destroy(session) }
        guard let decoder = vokra_codec_decoder_open(session) else {
            throw IOSBenchError.vokra("codec decoder open failed: \(lastVokraError())")
        }
        defer { vokra_codec_decoder_destroy(decoder) }

        let frameHop = Int(vokra_codec_decoder_frame_hop(decoder))
        let sampleRate = Int(vokra_codec_decoder_sample_rate(decoder))
        let nCodebooks = Int(vokra_codec_decoder_n_codebooks(decoder))
        guard frameHop > 0, sampleRate > 0, nCodebooks > 0 else {
            throw IOSBenchError.vokra("codec returned invalid model axes")
        }
        let periodS = Double(frameHop) / Double(sampleRate)
        let frameCount = Int(ceil(Self.sustainedDurationS / periodS))
        var frames: [CodecFrame] = []
        frames.reserveCapacity(frameCount)
        let codes = [UInt32](repeating: 0, count: nCodebooks)
        var pcm = [Float](repeating: 0, count: frameHop)
        let startingThermal = thermalState()
        let start = DispatchTime.now().uptimeNanoseconds

        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }

        for index in 0..<frameCount {
            let decodeStart = DispatchTime.now().uptimeNanoseconds
            var emitted: Int32 = 0
            let pushStatus = codes.withUnsafeBufferPointer { codeBuffer in
                vokra_codec_decoder_push_codes(
                    decoder,
                    codeBuffer.baseAddress,
                    codeBuffer.count,
                    &emitted
                )
            }
            try check(pushStatus, operation: "codec push frame \(index)")
            guard emitted == 1 else {
                throw IOSBenchError.vokra(
                    "codec push frame \(index) emitted \(emitted), expected 1"
                )
            }
            var written = 0
            let pullStatus = pcm.withUnsafeMutableBufferPointer { pcmBuffer in
                vokra_codec_decoder_pull_pcm(
                    decoder,
                    pcmBuffer.baseAddress,
                    pcmBuffer.count,
                    &written
                )
            }
            try check(pullStatus, operation: "codec pull frame \(index)")
            guard written == frameHop else {
                throw IOSBenchError.vokra(
                    "codec pull frame \(index) wrote \(written), expected \(frameHop)"
                )
            }
            let decodeMS = Double(
                DispatchTime.now().uptimeNanoseconds - decodeStart
            ) / 1_000_000.0

            let deadline = start + UInt64(Double(index + 1) * periodS * 1_000_000_000.0)
            let now = DispatchTime.now().uptimeNanoseconds
            if deadline > now {
                Thread.sleep(forTimeInterval: Double(deadline - now) / 1_000_000_000.0)
            }
            frames.append(
                CodecFrame(
                    index: index,
                    wallElapsedS: secondsSince(start),
                    decodeMS: decodeMS,
                    peakRSSBytes: peakRSSBytes(),
                    thermalState: thermalState()
                )
            )
        }

        let url = try writeCodecJSONL(
            modelSHA: modelSHA,
            sampleRate: sampleRate,
            frameHop: frameHop,
            nCodebooks: nCodebooks,
            startingThermal: startingThermal,
            frames: frames
        )
        let latencies = frames.map(\.decodeMS)
        return CodecSummary(
            logURL: url,
            frameCount: frames.count,
            durationS: frames.last?.wallElapsedS ?? 0.0,
            p50MS: percentile(latencies, 0.50),
            p95MS: percentile(latencies, 0.95),
            p99MS: percentile(latencies, 0.99),
            peakRSSBytes: frames.map(\.peakRSSBytes).max() ?? peakRSSBytes()
        )
    }

    private func openSession(
        modelURL: URL,
        backend: IOSBenchBackend
    ) throws -> OpaquePointer {
        guard let options = vokra_session_options_create() else {
            throw IOSBenchError.vokra("session options allocation failed")
        }
        defer { vokra_session_options_destroy(options) }
        try check(
            vokra_session_options_set_backend(options, backend.abiValue),
            operation: "select backend \(backend.rawValue)"
        )
        var session: OpaquePointer?
        let status = modelURL.path.withCString { path in
            vokra_session_create_from_file_with_options(path, options, &session)
        }
        try check(status, operation: "create \(backend.rawValue) session")
        guard let session else {
            throw IOSBenchError.vokra("session create returned OK with NULL handle")
        }
        return session
    }

    private func transcribe(session: OpaquePointer, pcm: [Float]) throws -> String {
        var output: UnsafeMutablePointer<CChar>?
        let status = pcm.withUnsafeBufferPointer { buffer in
            vokra_asr_transcribe(session, buffer.baseAddress, buffer.count, 16_000, &output)
        }
        try check(status, operation: "Whisper transcribe")
        guard let output else {
            throw IOSBenchError.vokra("transcribe returned OK with NULL text")
        }
        defer { vokra_string_free(output) }
        return String(cString: output)
    }

    private func check(_ status: vokra_status_t, operation: String) throws {
        guard status.rawValue == VOKRA_OK.rawValue else {
            throw IOSBenchError.vokra(
                "\(operation) failed rc=\(status.rawValue): \(lastVokraError())"
            )
        }
    }

    private func lastVokraError() -> String {
        guard let pointer = vokra_last_error() else { return "<no error detail>" }
        return String(cString: pointer)
    }

    private func loadMonoFloat32WAV(_ url: URL, expectedRate: Double) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard format.channelCount == 1, format.sampleRate == expectedRate else {
            throw IOSBenchError.resource(
                "WAV must be mono \(Int(expectedRate)) Hz; got " +
                "\(format.channelCount) channel(s) at \(format.sampleRate) Hz"
            )
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw IOSBenchError.resource("cannot allocate WAV buffer")
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else {
            throw IOSBenchError.resource("AVAudioFile did not expose Float32 PCM")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    private func sha256(_ url: URL) throws -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw IOSBenchError.io("cannot open for SHA-256: \(url.path)")
        }
        defer { try? handle.close() }
        var hash = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
            if chunk.isEmpty { break }
            hash.update(data: chunk)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func writeJSON<T: Encodable>(_ value: T, stem: String) throws -> URL {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        let url = documentsURL().appendingPathComponent("\(stem)-\(timestamp()).json")
        try data.write(to: url, options: .atomic)
        return url
    }

    private func writeCodecJSONL(
        modelSHA: String,
        sampleRate: Int,
        frameHop: Int,
        nCodebooks: Int,
        startingThermal: String,
        frames: [CodecFrame]
    ) throws -> URL {
        var data = Data()
        func append(_ object: [String: Any]) throws {
            data.append(try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
            data.append(0x0A)
        }
        try append([
            "kind": "metadata",
            "schema": "vokra.ios-codec-sustained.v1",
            "build_sha": configuration.buildSHA.lowercased(),
            "model_sha256": modelSHA,
            "device_model": configuration.deviceModel,
            "ios_version": UIDevice.current.systemVersion,
            "backend": IOSBenchBackend.cpu.rawValue,
            "sample_rate": sampleRate,
            "frame_hop": frameHop,
            "n_codebooks": nCodebooks,
            "target_duration_s": Self.sustainedDurationS,
            "conditions": [
                "ambient_temperature_c": configuration.ambientTemperatureC,
                "starting_thermal_state": startingThermal,
                "screen": configuration.screen,
                "charging": configuration.charging,
                "case": configuration.caseState,
            ],
        ])
        for frame in frames {
            try append([
                "kind": "frame",
                "index": frame.index,
                "wall_elapsed_s": frame.wallElapsedS,
                "decode_ms": frame.decodeMS,
                "peak_rss_bytes": frame.peakRSSBytes,
                "thermal_state": frame.thermalState,
            ])
        }
        let url = documentsURL().appendingPathComponent(
            "vokra-ios-codec-sustained-\(timestamp()).jsonl"
        )
        try data.write(to: url, options: .atomic)
        return url
    }

    private func documentsURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private func timestamp() -> String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: Date()).replacingOccurrences(of: ":", with: "-")
    }

    private func secondsSince(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
    }

    private func percentile(_ values: [Double], _ q: Double) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        let index = max(0, Int(ceil(q * Double(sorted.count))) - 1)
        return sorted[index]
    }

    private func peakRSSBytes() -> UInt64 {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
        // Darwin documents ru_maxrss in bytes (unlike Linux's KiB unit).
        return UInt64(max(0, usage.ru_maxrss))
    }

    private func thermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "critical"
        }
    }
}
