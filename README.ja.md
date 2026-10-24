# Vokra

[English](README.md) | **日本語**

[![CI](https://github.com/ayutaz/vokra/actions/workflows/ci.yml/badge.svg)](https://github.com/ayutaz/vokra/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Vokra は Rust で実装された、音声処理に特化した推論ランタイムです。一般的な
グラフランタイムではアプリケーション側へ追い出されがちなストリーミング状態、
STFT/iSTFT・mel frontend、vocoder、neural codec、CTC/RNN-T decode、VAD、
話者特徴、ピッチ抽出、音声強調をネイティブに扱います。

Vokra は provenance を含む GGUF を読み込み、ランタイムでは ONNX グラフを
ロードしません。デフォルトランタイムに外部 Cargo 依存はなく、root
`Cargo.lock` は first-party の `vokra-*` crate だけで構成されます。

> **CPU preview:** [v0.4.0-rc.1](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)
> を評価できます。公開済みLinux CLIと固定したWhisper base ASR・Silero VAD v5の
> 組合せで機能確認済みです。正確な取得先と制限は[導入ガイド](docs/release-0.4.0-rc1.ja.md)を
> 参照してください。数値parity、性能、他OSでの実モデル動作、全catalogのApple Metal対応を
> 示すものではありません。release・source・PR・残作業は[現況](docs/current-status.ja.md)で管理します。

## Vokra の特徴

- **音声ネイティブな実行:** 音声 frontend、streaming cache、decoder、
  vocoder、codec、VAD、音声強調を ONNX graph glue ではなく native operator
  として実装します。
- **小さな依存面:** runtime crate は first-party の `vokra-*` crate のみに
  依存し、オフライン変換は runtime load から分離されています。
- **明示的な失敗:** 未対応 op や利用できない device は error を返し、GPU から
  CPU へ暗黙に fallback しません。
- **再現可能なモデルファイル:** Vokra GGUF は frontend 設定、topology、量子化
  方針、source provenance、license 情報を保持します。
- **移植しやすい統合:** CPU が既定で、Metal・CUDA・Vulkan・WebGPU は opt-in
  です。生成済み C header から native / 言語 binding を構築できます。

## クイックスタート

目的に応じたガイドを選んでください。

| 目的 | ガイド |
|---|---|
| 検証済みprebuilt CPU previewを試す | [Preview導入](docs/release-0.4.0-rc1.ja.md) |
| ソースからビルドしcheckpointを変換する | [Getting started](docs/getting-started.ja.md) |
| モデル固有のCLIオプション・形式を使う | [CLI tutorial](docs/tutorials/cli.ja.md) |

## モデルとバックエンドの状態

converter、GGUF binder、native推論、独立数値parity、公開、実機検証は別々の段階です。
モデルの掲載やbuild成功だけでは、すべての完了を意味しません。日付付きinventoryと残作業は
[現況](docs/current-status.ja.md)、artifact別の利用条件は
[model card](https://huggingface.co/vokra)を参照してください。

CPUが既定です。accelerator対応はoptionalで、演算ごとに異なります。
[Backend guide](docs/backend-guide.ja.md)と[Architecture](docs/architecture.ja.md)を参照してください。
未対応演算や利用できないdeviceは、CPUへ暗黙にfallbackせずerrorを返します。

## ライブラリ統合

C library は次のコマンドでビルドします。

```sh
cargo build --release -p vokra-capi
```

[`include/vokra.h`](include/vokra.h) が生成済み C reference です。
[API index](docs/api-reference.ja.md) から Rust および Python、Swift/iOS、Unity、
Godot、Android、web、server の binding / example へ移動できます。C ABI は
pre-1.0 であり、まだ freeze されていません。

## ドキュメント

- [ドキュメントマップ](docs/README.md)
- [Getting Started](docs/getting-started.ja.md)
- [CLI tutorial](docs/tutorials/cli.ja.md)
- [Architecture](docs/architecture.ja.md)
- [Backend guide](docs/backend-guide.ja.md)
- [Migration guide](docs/migration-guide.ja.md)
- [License audit](docs/license-audit.md) と
  [法務・compliance note](docs/legal-compliance.md)

## コントリビューション

コントリビューションを歓迎します。大きな変更の前に
[CONTRIBUTING.md](CONTRIBUTING.md) を読み、範囲の明確な入口として
[good first tasks](docs/good-first-tasks.ja.md) を利用してください。不具合や提案は
[GitHub Issues](https://github.com/ayutaz/vokra/issues) で受け付けます。
参加時は[行動規範](CODE_OF_CONDUCT.ja.md)に従ってください。脆弱性は public
issue ではなく、[セキュリティポリシー](SECURITY.ja.md)に記載した非公開経路で
報告してください。

## ライセンス

Vokra のソースコードは [Apache-2.0](LICENSE) です。モデル weight や reference
asset には別の license が適用される場合があります。再配布や商用利用の前に、
各 model card、[`docs/license-audit.md`](docs/license-audit.md)、[NOTICE](NOTICE)
を確認してください。非商用 weight は明示的な research-only gate がない限り、
default の公開経路から除外されます。
