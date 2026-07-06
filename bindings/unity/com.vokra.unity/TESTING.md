# Vokra Unity — Testing procedure

This document records how the `com.vokra.unity` UPM package is smoke-tested
before merge and how the owner (依頼者) signs off on device-specific paths that
CI cannot cover. Referenced by M2-11 ticket spec T15 / T16 / T17 / T18.

The three verification tiers are:

1. **CI (automated, per-PR)** — `unity-package` job in `.github/workflows/ci.yml`
   asserts UPM structure, NVIDIA non-bundle, zero-dep package, native
   artifact assembly. Does NOT touch Unity Editor.
2. **Nightly IL2CPP (automated, best-effort)** — `.github/workflows/nightly-il2cpp.yml`
   compiles the `Samples~/VadAsrTts` demo under IL2CPP on Linux headless and
   runs `DemoUi.RunHeadless()`. Requires `secrets.UNITY_LICENSE`; if absent the
   job self-skips with a WARNING and this document's "Local IL2CPP invocation
   fallback" section is the substitute.
3. **Owner device tests (manual, pre-merge)** — iOS (T16) and Android (T17)
   physical-device runs the owner records below before the M2-11-T18 merge PR
   is approved.

---

## 1. CI (per-PR) — no owner action needed

Automatic. Green `unity-package` check on the PR is required for merge per
NFR-MT-07. Scope:

- `jq . package.json` and `jq . Runtime/Vokra.Runtime.asmdef` parse cleanly.
- `scripts/check-unity-package-no-nvidia.sh` passes (no `cudart*` / `cudnn*` /
  `cublas*` under `Plugins/`, no undefined `cudart` symbols in shipped libs).
- `scripts/check-unity-package-deps.sh` asserts `.dependencies == {}` in
  `package.json` and no Sentis / Barracuda references in the runtime `.asmdef`.
- `scripts/check-callback-pattern.sh` asserts every `[MonoPInvokeCallback]` is
  on a `static` method with a rooted `static readonly` delegate field and a
  `[Preserve]` annotation (R1 mitigation).
- Assembled `com.vokra.unity-<version>.tgz` is uploaded as a build artifact
  for smoke-import by the owner if desired.

## 2. Nightly IL2CPP (best-effort, license-gated)

Workflow: `.github/workflows/nightly-il2cpp.yml`, cron `17 4 * * *` UTC + on
`workflow_dispatch`.

### What it verifies

- `com.vokra.unity` package imports into a fresh 2022.3 LTS project via
  `Packages/manifest.json` `file:` reference (no OpenUPM round-trip required).
- The `Samples~/VadAsrTts` sample compiles under **IL2CPP** scripting backend
  on Linux Standalone (this is where R1 — AOT static callback stripping —
  would surface if the pattern regressed).
- `DemoUi.RunHeadless()` is invoked (auto-fires when `Application.isBatchMode`
  is true, see `Samples~/VadAsrTts/Scripts/DemoUi.cs:55-58`).
- Player process exits with code 0 (any pipeline stage error would return 1).
- `Application.consoleLogPath` contains the `Vokra Unity demo` marker.

### License handling

The job requires `secrets.UNITY_LICENSE` (base64-encoded Unity Personal or
Pro `.ulf` file). If the secret is absent the `preflight` step emits a
`::warning::` annotation and the `il2cpp-linux-headless` job is skipped via
`if: needs.preflight.outputs.has_license == 'true'`. This is intentional: the
2026-07-04 owner decisions did not commit to provisioning a Unity license,
so the nightly is opt-in without breaking main.

To enable:
1. In Unity Hub, export a Personal license (`Unity/Hub/UnityLicensingClient_V1/Licenses/*.ulf`).
2. `base64 -i Unity_*.ulf | pbcopy` (macOS) or `base64 -w0 Unity_*.ulf` (Linux).
3. Paste as `UNITY_LICENSE` under repo Settings → Secrets and variables → Actions.

### Failure triage

Player log is uploaded as `nightly-il2cpp-player-log` (retention 7 days). Grep
for `NullReferenceException` inside a `[MonoPInvokeCallback]` frame — this is
the R1 symptom (delegate collected by GC or stripped by IL2CPP AOT). Fix by
verifying (a) `link.xml` preserves `Vokra.Runtime`, (b) callback method has
`[Preserve]`, (c) delegate instance held in a `static readonly` field.

### Local IL2CPP invocation fallback (when the nightly is skipped)

Run this on any host with Unity 2022.3 LTS + IL2CPP module installed:

```bash
# 1. Build the native cdylib (CPU only, matches NFR-DS-02 / D7 policy).
cargo build --release -p vokra-capi --no-default-features --features cpu
mkdir -p bindings/unity/com.vokra.unity/Plugins/$(uname -s)/x86_64
cp target/release/libvokra.so bindings/unity/com.vokra.unity/Plugins/Linux/x86_64/    # Linux
# cp target/release/libvokra.dylib bindings/unity/com.vokra.unity/Plugins/macOS/      # macOS
# cp target/release/vokra.dll bindings/unity/com.vokra.unity/Plugins/Windows/x86_64/  # Windows

# 2. Stage a throwaway consumer project.
mkdir -p /tmp/vokra-consumer/{Assets,Packages,ProjectSettings}
cat > /tmp/vokra-consumer/Packages/manifest.json <<EOF
{
  "dependencies": {
    "com.vokra.unity": "file:$(pwd)/bindings/unity/com.vokra.unity"
  }
}
EOF

# 3. Open in Unity 2022.3 LTS, import the "VAD -> ASR -> TTS demo" sample from
#    the Package Manager UI, then build:
#      File -> Build Settings -> Player Settings -> Other Settings
#         -> Scripting Backend: IL2CPP
#         -> Api Compatibility Level: .NET Standard 2.1
#      File -> Build Settings -> Build (Standalone target for the current OS)

# 4. Run the built player headless:
./VokraDemo -batchmode -nographics \
  -logFile /tmp/vokra-player.log \
  -vokraInput bindings/unity/com.vokra.unity/Samples~/VadAsrTts/StreamingAssets/test_16k.wav \
  -vokraOutput /tmp/out.wav
echo "exit=$?"
grep "Vokra Unity demo" /tmp/vokra-player.log
```

Record the result in the "Owner sign-off log" section below.

## 3. Owner device tests (pre-merge, T16 / T17)

The M2-11-T18 merge PR is BLOCKED until both entries below have a
green-check line signed by the owner. CC cannot cover these — physical iOS /
Android hardware is required per M2-11 ticket spec.

### T16 · iOS device test

Environment: physical iOS device (iPhone / iPad, iOS 16+ recommended), Xcode
15+, Apple Developer signing certificate.

Procedure:

1. In Unity 2022.3 LTS, open `/tmp/vokra-consumer/` (see local fallback above),
   import the `VAD -> ASR -> TTS demo` sample from Package Manager.
2. Switch platform to `iOS` in `File -> Build Settings`.
3. `Player Settings -> Other Settings`: Scripting Backend = IL2CPP,
   Architecture = ARM64, Target minimum iOS Version = 13.0.
4. Build to an Xcode project directory, open in Xcode.
5. Confirm `Vokra.Editor`'s post-process build step attached
   `Metal.framework` and `Accelerate.framework` to the main target (T06).
6. Confirm `libvokra.a` (device slice from M2-02 XCFramework) is linked
   statically via `__Internal` (D4 / NFR-RL-03) — no `.dylib` in the app
   bundle.
7. Deploy to the device, run the sample; observe:
   - Bundled `test_16k.wav` transcribes without a `Failed to load GGUF` error.
   - Synthesized TTS clip plays without a native crash.
   - No `NullReferenceException` in the Xcode Console from a
     `[MonoPInvokeCallback]` frame (R1 negative check).
8. Record device model, iOS version, Xcode version, and pass/fail in the log
   below.

### T17 · Android device test

Environment: physical Android device (Android 10+, arm64-v8a), Android
Studio 2023.1+ or Unity built APK sideload.

Procedure:

1. In the same consumer project, switch platform to `Android`.
2. `Player Settings -> Other Settings`: Scripting Backend = IL2CPP,
   Target Architectures = `ARM64` only (deselect `ARMv7`), Minimum API Level
   = 24 (matches `scripts/build-android.sh`).
3. Ensure `bindings/unity/com.vokra.unity/Plugins/Android/libs/arm64-v8a/libvokra.so`
   exists (produced by `scripts/build-android.sh` with a valid
   `ANDROID_NDK_HOME`).
4. `Build and Run` to attached device.
5. Observe:
   - `VokraAndroidAssets.EnsureLocalCopy` correctly expands the model files
     from the APK `jar:file://...!/assets/...` URL to
     `Application.persistentDataPath/vokra/...` before the C ABI opens them
     (D6 / R2 — no `Failed to load GGUF` due to jar URL leakage).
   - Whisper / piper-plus complete on-device without an OOM (base model,
     RAM footprint under 1 GB).
   - No `SIGABRT` from `dlopen` failure of `libvokra.so`.
6. Record device model, Android version, NDK version, and pass/fail below.

---

## Owner sign-off log

Append one line per device-test run. Format:

```
YYYY-MM-DD | T16 iOS   | <device / iOS / Xcode> | PASS | notes
YYYY-MM-DD | T17 Android | <device / Android / NDK> | PASS | notes
YYYY-MM-DD | Nightly-fallback local IL2CPP | <host OS> | PASS | notes
```

_No entries yet — awaiting owner verification prior to M2-11-T18 merge._
