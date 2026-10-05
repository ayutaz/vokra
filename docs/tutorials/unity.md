# Unity + IL2CPP tutorial

**English** | [日本語](unity.ja.md)

Vokra provides a Unity Package (`com.vokra.unity`) and a C# API designed for
IL2CPP AOT compilation and iOS static linking constraints. The `v0.3.0` GitHub
Release assembles the package tarball and native slices; the repository tree
itself remains source-only for native plugins.
The tracked UPM tree contains only native-plugin `.gitkeep`/`.meta`
placeholders; a clean Git URL import is source-only and cannot run until the
target native library is built and staged. Prebuilt libraries are assembled by
the release workflow, not committed to the source tree.

## 1. Prerequisites

- **Unity 2022.3 LTS** or newer (Unity 6 verified via nightly IL2CPP
  smoke test in `.github/workflows/nightly-il2cpp.yml`).
- Target platforms: macOS, Windows, Linux, iOS, Android (Editor +
  Standalone / Player), and WebGL. WebGL uses the CPU-only staticlib path
  (M4-02, via `vokra_session_create_from_bytes`); the nightly Unity Editor
  smoke remains license-gated by `secrets.UNITY_LICENSE`.
- For iOS builds: Xcode 14+; for Android builds: Android SDK / NDK
  matching your Unity install.

## 2. Install the package

The package can be referenced in three ways: the Git URL is for source
inspection, the local flow is for development, and the `v0.3.0` tarball is for
release consumption:

### UPM Git URL (source inspection only)

```
Window → Package Manager → + → Add package from git URL…

https://github.com/ayutaz/vokra.git?path=/bindings/unity/com.vokra.unity
```

This Git URL points at the source-only package tree and does not provide
runnable native binaries; use the `v0.3.0` release tarball for staged slices.

### Local file reference (development)

```json
{
  "dependencies": {
    "com.vokra.unity": "file:../../vokra/bindings/unity/com.vokra.unity"
  }
}
```

Clone the released tag (or a newer explicitly reviewed commit), then build and
stage the native library for the platform you will test:

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
git checkout --detach v0.3.0

# Host desktop (macOS, Linux, or Windows).
scripts/build-unity-plugin.sh
# Android (requires ANDROID_NDK_HOME).
ANDROID_NDK_HOME=/path/to/ndk scripts/build-android.sh
# iOS (build the XCFramework, then stage its device slice for Unity).
scripts/build-ios.sh
scripts/collect-ios-lib.sh
# WebGL (CPU-only wasm archive).
scripts/build-unity-webgl-lib.sh
```

The helpers require their corresponding platform SDK/toolchain and produce
local development artifacts. The local `file:` reference becomes runnable
only after the relevant library is staged.

### Tarball (production)

Download `com.vokra.unity-0.3.0.tgz` from the authorized GitHub Release and use
**Add package from tarball…** in the Package Manager. OpenUPM publication is a
separate owner-gated decision.

## 3. Supported platform matrix

| Platform | Native lib after local staging           | Feature set                                 |
| -------- | --------------------------------------- | ------------------------------------------- |
| macOS    | `Plugins/macOS/libvokra.dylib`          | CPU (Metal opt-in)                          |
| Windows  | `Plugins/Windows/x86_64/vokra.dll`      | CPU (CUDA opt-in, system-installed)         |
| Linux    | `Plugins/Linux/x86_64/libvokra.so`      | CPU (CUDA opt-in, system-installed)         |
| iOS      | `Plugins/iOS/libvokra.a` (`__Internal`) | CPU                                         |
| Android  | `Plugins/Android/libs/arm64-v8a/libvokra.so` | CPU                                    |
| WebGL    | `Plugins/WebGL/libvokra.a` (`__Internal`) | CPU-only WASM (WebGPU not wired)          |

## 4. Minimal C# usage

```csharp
using Vokra;
using UnityEngine;

public class VokraDemo : MonoBehaviour
{
    void Start()
    {
        // Load a GGUF; the task (ASR / TTS / VAD) is selected automatically
        // from the model's vokra.model.arch metadata.
        using var session = VokraSession.CreateFromFile(
            System.IO.Path.Combine(Application.streamingAssetsPath, "whisper-base.gguf"));

        Debug.Log($"Vokra runtime version: {VokraSession.RuntimeVersion}");

        // ASR: pass mono float32 PCM at the model's native rate (Whisper: 16 kHz).
        float[] pcm = LoadMonoPcmFromAudioClip(myAudioClip, targetHz: 16000);
        string text = session.Transcribe(pcm, 16000);
        Debug.Log(text);
    }
}
```

For TTS:

```csharp
using var session = VokraSession.CreateFromFile(voicePath);
var (pcm, sampleRate) = session.Synthesize("Hello from Vokra.");
AudioClip clip = AudioClip.Create("vokra-tts", pcm.Length, 1, sampleRate, false);
clip.SetData(pcm, 0);
audioSource.PlayOneShot(clip);
```

For VAD (streaming):

```csharp
using var session = VokraSession.CreateFromFile(vadModelPath);
using var stream = session.OpenVadStream(16000);
// See VokraStream.Push / VokraStream.Poll for the streaming API.
```

## 5. IL2CPP-safe callback pattern

Unity's IL2CPP AOT compiler disallows C# closures on native callback
boundaries. Vokra's C# API sidesteps this in two ways:

- Every public method is a **synchronous** call over `NativeMethods.*`
  (no C# delegates crossing FFI).
- For streaming pump-based callbacks (planned), the package uses the
  `[MonoPInvokeCallback]` + `static readonly delegate root` + `GCHandle`
  pattern (see
  [`Runtime/Vokra/VokraCallbacks.cs`](../../bindings/unity/com.vokra.unity/Runtime/Vokra/VokraCallbacks.cs)).

If you extend the binding, keep any callback:

1. `static` and marked `[MonoPInvokeCallback(typeof(...))]`
2. Held by a `static readonly` field to prevent AOT stripping.
3. Attached to user state via a `GCHandle` payload.

## 6. iOS: `DllImport("__Internal")`

The Vokra binding declares its P/Invoke entries with a platform switch
so the same C# call site works on iOS (static link) and on Standalone
(dynamic link):

```csharp
#if UNITY_IOS && !UNITY_EDITOR
    const string Lib = "__Internal";
#else
    const string Lib = "vokra";
#endif
[DllImport(Lib)]
static extern int vokra_session_create_from_file(...);
```

See
[`Runtime/Vokra/NativeMethods.cs`](../../bindings/unity/com.vokra.unity/Runtime/Vokra/NativeMethods.cs)
for the full set.

Also included: an `iOSPostProcessBuild` (in the package's `Editor`
folder) that registers the static library with the Xcode project
generated by Unity.

## 7. Android: `persistentDataPath` helper

Android bundles `StreamingAssets` inside the APK / AAB as a jar URL,
which the native side cannot `fopen`. The package includes a helper
that extracts a bundled model into `persistentDataPath` on first use:

```csharp
using Vokra.Android;
string modelPath = await VokraAndroidAssets.EnsureExtracted("whisper-base.gguf");
using var session = VokraSession.CreateFromFile(modelPath);
```

Source:
[`Runtime/Vokra/VokraAndroidAssets.cs`](../../bindings/unity/com.vokra.unity/Runtime/Vokra/VokraAndroidAssets.cs).

## 8. NVIDIA runtime is **not** bundled

Per the NVIDIA CUDA EULA ("installed only in a private (non-shared)
directory location"), this package does NOT ship `cudart` / `cudnn` /
`cublas`. When CUDA acceleration is enabled at runtime, Vokra loads
`libcuda.so` / `nvcuda.dll` from the system install via `dlopen`. The
CI enforces this with
`scripts/check-unity-package-no-nvidia.sh`; do not add NVIDIA binaries
to the `Plugins/` folder.

## 9. Samples

Import the *VAD → ASR → TTS demo* from the Package Manager window's
**Samples** tab. Demo model weights (Silero VAD v5 MIT, Whisper base MIT,
piper-plus voice MIT) are **not** bundled, and the `v0.3.0` GitHub Release
contains no GGUF assets. Before running
`Samples~/VadAsrTts/scripts/fetch-demo-models.sh`, set all three environment
variables to independently verified MIT source URLs; release presence alone is
not evidence that a model URL exists (NFR-DS-04). See the sample README for
the required URL variables and license/provenance checks.

## 10. Troubleshooting

- **`DllNotFoundException: vokra`**: the Plugins folder is missing your
  platform's native library. A Git URL import cannot supply it from the
  source-only tree; for a local `file:` install, run the matching staging helper
  described in section 2, or install the `v0.3.0` release tarball.
- **`VokraException: Unsupported backend`**: FR-EX-08 forbids silent
  fallback. Either build with the matching backend feature or use a
  GGUF whose ops are covered by CPU.
- **iOS build fails on Bitcode**: turn off Bitcode in the generated
  Xcode project (Bitcode is deprecated by Apple).
- **IL2CPP smoke tests**: enable the nightly job by setting
  `secrets.UNITY_LICENSE` in the GitHub repository (see
  `docs/m2-owner-verification-checklist.md` §7).

## Next steps

- **Migration**: if you are coming from `sherpa-onnx-unity` /
  `onnxruntime-unity`, see [Migration Guide](../migration-guide.md).
- **iOS device RTF**: for the on-device NFR-PF-03 measurement (Whisper
  base **RTF < 0.5**), follow
  [`docs/m2-14-ios-rtf-handover.md`](../m2-14-ios-rtf-handover.md).
- **Zero third-party runtime deps**: the package's `link.xml` prevents
  IL2CPP stripping of the P/Invoke entries; it does NOT force any
  managed assembly beyond the Vokra binding itself.
