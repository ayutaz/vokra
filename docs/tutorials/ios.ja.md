# iOS + Swift Package チュートリアル

[English](ios.md) | **日本語**

Vokra は iOS 対応の **XCFramework** を [`Package.swift`](../../Package.swift)
にラップして配布しています。このチュートリアルは `git clone` から実機で
Vokra 呼び出しを動かすまでを扱います。

## 1. 前提

- **Xcode 14 以上**（macOS）
- **iOS 15 以上**の実機または Simulator（iOS 14 以下はスコープ外）
- 実機配布用の Apple Developer 署名プロファイル
- Vokra リポジトリ（またはタグ付きリリース URL）

## 2. XCFramework をビルド

2 通り:

### 経路 A — ソースからビルド（ローカル開発）

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
scripts/build-ios.sh
# build/ios/Vokra.xcframework に 2 slice を生成:
#   - iOS device (arm64)
#   - iOS Simulator (arm64 + x86_64)
```

Slice の検証:

```sh
scripts/verify-ios-xcframework.sh build/ios/Vokra.xcframework
```

### 経路 B — リリース DL

CD は `Vokra.xcframework.zip` と SHA-256 を GitHub Release asset として
公開します。`Package.swift` を URL 形式に切り替えます（テンプレートは
コメントアウトで既に用意済み）:

```swift
.binaryTarget(
    name: "Vokra",
    url: "https://github.com/ayutaz/vokra/releases/download/<tag>/Vokra.xcframework.zip",
    checksum: "<sha256>"
)
```

## 3. iOS アプリに Vokra を追加

Xcode で:

1. **File → Add Package Dependencies → Add Local** でリポジトリルートを
   選ぶ（経路 A）、または URL を貼る（経路 B）
2. **`Vokra`** product をアプリターゲットにアタッチ
3. GGUF（`whisper-base.gguf`）と WAV fixture
   （`tests/fixtures/audio/jfk-30s.wav`）をアプリのバンドル Resources に
   追加。**Copy items if needed** と **Add to target** を両方 ON
4. **Signing & Capabilities**: Team と一意な Bundle ID を設定
5. **Build Settings** → **Enable Bitcode = No** に設定（Apple が deprecate
   済み。XCFramework は Bitcode 同梱ではない）

## 4. 最小 SwiftUI サンプル

```swift
import SwiftUI
import Vokra   // Clang module がすべての `vokra_*` C シンボルを re-export
import Foundation

struct ContentView: View {
    @State private var status = "Idle"
    var body: some View {
        VStack(spacing: 20) {
            Text(status).font(.system(.body, design: .monospaced))
            Button("Transcribe") { transcribe() }
        }.padding()
    }

    func transcribe() {
        guard
            let gguf = Bundle.main.path(forResource: "whisper-base", ofType: "gguf"),
            let wav  = Bundle.main.path(forResource: "jfk-30s",     ofType: "wav")
        else {
            status = "resource missing"; return
        }

        var session: OpaquePointer?
        let rc = vokra_session_create_from_file(gguf, &session)
        guard rc == 0, let s = session else {
            status = "session_create err=\(rc)"; return
        }
        defer { vokra_session_destroy(s) }

        // 実機 RTF 計測の e2e harness は docs/m2-14-ios-rtf-handover.md 参照
        var out: UnsafeMutablePointer<CChar>?
        let arc = vokra_asr_transcribe(s, wav, &out)
        let text = out.map { String(cString: $0) } ?? "<null>"
        if let o = out { vokra_string_free(o) }
        status = "rc=\(arc): \(text.prefix(120))"
    }
}
```

`vokra_*` C シンボルは XCFramework の `Headers/vokra.h` と完全一致で、
Clang module `Vokra` がすべて re-export します。

## 5. iOS 固有の不変条件

Vokra iOS ビルドはプラットフォームの制約を強制します:

- **静的リンクのみ** — iOS の動的ライブラリロード禁止（NFR-RL-03）に対応。
  XCFramework は `.a` slice で、Unity 側は `DllImport("__Internal")`、
  Swift 側は Clang module 経由で同一シンボルにアクセス。
- **JIT 不使用** — iOS の W^X 制約。Vokra はランタイム CPU ディスパッチ
  のみ（NFR-RL-05）で JIT は同梱しません。
- **CUDA は iOS で compile error** — `vokra-backend-cuda` の
  `compile_error!(...)` により、動かせない CUDA 型シンボルを silent に
  リンクしてしまうのを防ぎます。
- **アクセラレーションは Metal** — CPU（Apple Silicon の NEON）は常に
  利用可能。Metal を有効にするには
  `vokra_session_set_backend(s, VOKRA_BACKEND_METAL)`（enum 名は
  `vokra.h` 参照）。FR-EX-08 により、Metal で未対応の op は明示エラー
  で silent CPU fallback は行いません。

## 6. 実機 RTF 計測

NFR-PF-03（実機で Whisper base **RTF < 0.5** — v0.5 Exit criteria）の計測
は、専用の引き渡し文書を参照してください:

- [`docs/m2-14-ios-rtf-handover.md`](../m2-14-ios-rtf-handover.md)

SwiftUI 計測アプリ、記録テンプレート（backend / device / iOS version /
elapsed / audio / RTF）、R4「iPhone で Metal probe が失敗した場合」の境
界ケース、そして deliverable checklist が含まれます。

## 7. トラブルシューティング

- **`module 'Vokra' not found`**: SwiftPM が binary target を解決できて
  いません。`build/ios/Vokra.xcframework` の存在（経路 A）、あるいは
  `Package.swift` の URL + checksum の一致（経路 B）を確認してください。
- **`Undefined symbol: _vokra_...`**: 古い XCFramework の可能性。該当
  コミットで `scripts/build-ios.sh` を再実行してください。
- **アプリサイズ**: リリース XCFramework は `.a` ベース。App Store
  slicing で不要 slice のシンボルを最終 `.ipa` から落とせます。それでも
  大きい場合は `vokra-cli convert --quantize q4_k` で K-quant 化した
  GGUF を検討してください。

## 次のステップ

- **Simulator vs 実機**: Simulator の RTF は NFR-PF-03 の判定には**使え
  ません**。実機で計測してください。
- **Metal 検証**: macOS の Metal 経路とペアで動かすと turnaround が速い
  です。両経路とも同じ Rust runtime + 同じ GGUF を使うため、iPhone で
  再現するバグは macOS でも通常再現します。
- **移行**: `onnxruntime-swift` や Core ML パイプラインから移行する場
  合は [Migration Guide](../migration-guide.ja.md) を参照。
