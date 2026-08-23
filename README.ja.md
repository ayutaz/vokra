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

> **リリース状況:** `0.1.0` を最初のタグ付きリリースとして準備しています。
> Rust API、C ABI、GGUF metadata、モデル対応範囲は引き続き pre-1.0 で、
> 変更される可能性があります。他プロジェクトで評価するときは正確な release
> を固定してください。

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

Git と Rust 1.89 以降が必要です。ソースから CLI をビルドします。

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
cargo build --release -p vokra-cli
```

公開済み Whisper base GGUF を取得し、同梱の Public Domain 音声 fixture を
実行します。

```sh
curl -L https://huggingface.co/vokra/whisper-base/resolve/main/whisper-base.gguf \
  -o whisper-base.gguf
target/release/vokra-cli run \
  --model whisper-base.gguf \
  --input tests/fixtures/audio/jfk-30s.wav
```

別の architecture を変換・実行する前に組み込み help を確認してください。

```sh
target/release/vokra-cli --help
target/release/vokra-cli convert --help
target/release/vokra-cli run --help
```

[Getting Started](docs/getting-started.ja.md) では変換、VAD、TTS、benchmark、
C ABI を説明しています。

## モデルとバックエンドの状態

Vokra は ASR、TTS、Speech-to-Speech、VAD / turn-taking、keyword spotting、
話者処理、pitch、codec / vocoder、音声強調、音源分離、音響理解を対象にします。
成熟度は architecture ごとに管理され、converter、GGUF loader、native forward、
numerical parity、公開 artefact はそれぞれ別の到達点です。どれか一つの存在を、
後続段階の完了とは扱いません。

コピーされたモデル一覧ではなく、次の情報源を参照してください。

- `vokra-cli convert --help` — converter が受理する identifier
- `vokra-cli run --help` — CLI の input / output / backend 契約
- [Vokra model hub](https://huggingface.co/vokra) — 公開 artefact とモデル別
  license card
- [`crates/vokra-cli/src/engine.rs`](crates/vokra-cli/src/engine.rs) — 開発者向けの
  明示的な runtime routing / deferred-operation registry

CPU が既定 backend です。Metal・CUDA・Vulkan・WebGPU は opt-in で、対応 op は
backend ごとに異なります。CoreML / QNN は experimental delegate です。
accelerator を選ぶ前に [backend guide](docs/backend-guide.ja.md) を確認してください。

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
