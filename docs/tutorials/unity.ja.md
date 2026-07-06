# Unity + IL2CPP チュートリアル

[English](unity.md) | **日本語**

Vokra は Unity パッケージ（`com.vokra.unity`）を、対応プラットフォーム
向けのネイティブライブラリ + IL2CPP AOT と iOS 静的リンク制約を満たす
C# API 付きで配布しています。

## 1. 前提

- **Unity 2022.3 LTS 以上**（Unity 6 も
  `.github/workflows/nightly-il2cpp.yml` の nightly IL2CPP smoke test で
  検証済み）
- 対応プラットフォーム: macOS / Windows / Linux / iOS / Android
  （Editor + Standalone / Player）。WebGL は `0.1.x` ではスコープ外
- iOS ビルド: Xcode 14 以上。Android ビルド: Unity インストールに合った
  Android SDK / NDK

## 2. パッケージのインストール

3 通り:

### UPM Git URL（推奨）

```
Window → Package Manager → + → Add package from git URL…

https://github.com/ayutaz/vokra.git?path=/bindings/unity/com.vokra.unity
```

### ローカル file: 参照（開発用）

```json
{
  "dependencies": {
    "com.vokra.unity": "file:../../vokra/bindings/unity/com.vokra.unity"
  }
}
```

### tarball（本番）

GitHub Releases から `com.vokra.unity-<version>.tgz` をダウンロードし、
Package Manager の **Add package from tarball…** で追加します。

## 3. 対応プラットフォーム

| プラットフォーム | ネイティブライブラリ                           | 機能セット                             |
| ---------------- | ---------------------------------------------- | -------------------------------------- |
| macOS            | `Plugins/macOS/libvokra.dylib`                 | CPU（Metal opt-in）                    |
| Windows          | `Plugins/Windows/x86_64/vokra.dll`             | CPU（CUDA opt-in、system install）     |
| Linux            | `Plugins/Linux/x86_64/libvokra.so`             | CPU（CUDA opt-in、system install）     |
| iOS              | `Plugins/iOS/libvokra.a`（`__Internal`）       | CPU                                    |
| Android          | `Plugins/Android/libs/arm64-v8a/libvokra.so`   | CPU                                    |

## 4. 最小 C# サンプル

```csharp
using Vokra;
using UnityEngine;

public class VokraDemo : MonoBehaviour
{
    void Start()
    {
        // GGUF をロード。タスク（ASR / TTS / VAD）はモデルの
        // vokra.model.arch メタデータから自動選択。
        using var session = VokraSession.CreateFromFile(
            System.IO.Path.Combine(Application.streamingAssetsPath, "whisper-base.gguf"));

        Debug.Log($"Vokra runtime version: {VokraSession.RuntimeVersion}");

        // ASR: モデルの frontend サンプルレートで mono float32 PCM を渡す
        // （Whisper は 16 kHz）
        float[] pcm = LoadMonoPcmFromAudioClip(myAudioClip, targetHz: 16000);
        string text = session.Transcribe(pcm, 16000);
        Debug.Log(text);
    }
}
```

TTS:

```csharp
using var session = VokraSession.CreateFromFile(voicePath);
var (pcm, sampleRate) = session.Synthesize("Hello from Vokra.");
AudioClip clip = AudioClip.Create("vokra-tts", pcm.Length, 1, sampleRate, false);
clip.SetData(pcm, 0);
audioSource.PlayOneShot(clip);
```

VAD（ストリーミング）:

```csharp
using var session = VokraSession.CreateFromFile(vadModelPath);
using var stream = session.OpenVadStream(16000);
// ストリーミング API は VokraStream.Push / VokraStream.Poll 参照
```

## 5. IL2CPP-safe callback パターン

Unity の IL2CPP AOT compiler は C# クロージャをネイティブ callback とし
て渡すことを禁止します。Vokra の C# API は以下 2 点でこれを回避します:

- 公開メソッドは `NativeMethods.*` を経由する**同期呼び出し**のみ
  （C# delegate を FFI 越しに渡さない）
- 将来のストリーミングポンプ callback は `[MonoPInvokeCallback]` +
  `static readonly delegate root` + `GCHandle` パターンで実装
  （[`Runtime/Vokra/VokraCallbacks.cs`](../../bindings/unity/com.vokra.unity/Runtime/Vokra/VokraCallbacks.cs)）

バインディングを拡張する場合、callback は必ず:

1. `static` かつ `[MonoPInvokeCallback(typeof(...))]` を付ける
2. `static readonly` フィールドで参照を保持し AOT strip を防ぐ
3. ユーザ状態は `GCHandle` payload 経由で渡す

## 6. iOS: `DllImport("__Internal")`

Vokra バインディングは P/Invoke エントリをプラットフォームスイッチで宣
言しており、同じ C# 呼び出し箇所が iOS（静的リンク）と Standalone（動的
リンク）の両方で動きます:

```csharp
#if UNITY_IOS && !UNITY_EDITOR
    const string Lib = "__Internal";
#else
    const string Lib = "vokra";
#endif
[DllImport(Lib)]
static extern int vokra_session_create_from_file(...);
```

全エントリは
[`Runtime/Vokra/NativeMethods.cs`](../../bindings/unity/com.vokra.unity/Runtime/Vokra/NativeMethods.cs)
参照。

パッケージ `Editor` フォルダには、Unity が生成する Xcode プロジェクト
に静的ライブラリを登録する `iOSPostProcessBuild` も同梱しています。

## 7. Android: `persistentDataPath` ヘルパー

Android では `StreamingAssets` が APK / AAB 内で jar URL として展開され
るため、ネイティブ側から `fopen` できません。パッケージには初回アクセ
ス時にモデルを `persistentDataPath` に展開するヘルパーを同梱しています:

```csharp
using Vokra.Android;
string modelPath = await VokraAndroidAssets.EnsureExtracted("whisper-base.gguf");
using var session = VokraSession.CreateFromFile(modelPath);
```

ソース:
[`Runtime/Vokra/VokraAndroidAssets.cs`](../../bindings/unity/com.vokra.unity/Runtime/Vokra/VokraAndroidAssets.cs)

## 8. NVIDIA ランタイムは同梱**しません**

NVIDIA CUDA EULA（"installed only in a private (non-shared) directory
location"）に基づき、本パッケージは `cudart` / `cudnn` / `cublas` を同
梱しません。CUDA アクセラレーションを有効化する際、Vokra は system
install の `libcuda.so` / `nvcuda.dll` を `dlopen` でロードします。CI は
`scripts/check-unity-package-no-nvidia.sh` でこれを強制します。
`Plugins/` に NVIDIA バイナリを追加しないでください。

## 9. サンプル

Package Manager の **Samples** タブから *VAD → ASR → TTS demo* をイン
ポートします。デモ用モデル weight（Silero VAD v5 MIT、Whisper base MIT、
piper-plus voice MIT）は**同梱していません** — インポート後に
`Samples~/VadAsrTts/scripts/fetch-demo-models.sh` を実行してください
（NFR-DS-04）。

## 10. トラブルシューティング

- **`DllNotFoundException: vokra`**: プラットフォーム向けネイティブラ
  イブラリが Plugins フォルダにありません。パッケージを再インポートす
  るか、ローカル `file:` インストールの場合は CD スクリプトをローカル
  実行してください。
- **`VokraException: Unsupported backend`**: FR-EX-08 により silent
  fallback は禁止されています。対応する backend feature でビルドする
  か、op が CPU でカバーされる GGUF を使ってください。
- **iOS ビルドが Bitcode で失敗**: Unity が生成する Xcode プロジェクト
  で Bitcode を OFF にしてください（Bitcode は Apple が deprecate 済み）。
- **IL2CPP smoke test**: `secrets.UNITY_LICENSE` を GitHub リポジトリに
  登録することで nightly job が有効化されます
  （`docs/m2-owner-verification-checklist.md` §7 参照）。

## 次のステップ

- **移行**: `sherpa-onnx-unity` / `onnxruntime-unity` から移行する場合
  は [Migration Guide](../migration-guide.ja.md) を参照。
- **iOS 実機 RTF**: 実機で NFR-PF-03（Whisper base **RTF < 0.5**）を計
  測する場合は [`docs/m2-14-ios-rtf-handover.md`](../m2-14-ios-rtf-handover.md)
  を参照。
- **サードパーティランタイム依存ゼロ**: パッケージの `link.xml` は
  P/Invoke エントリを IL2CPP の strip から守るためだけのもので、Vokra
  バインディング以外のマネージドアセンブリを強制しません。
