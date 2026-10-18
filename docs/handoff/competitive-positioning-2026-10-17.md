# Vokraの差別化と告知方針 — 2026-10-17

## 調査範囲

次版previewの対象と告知方針を決めるためのmanagement record。
競合の公式README／GitHub APIを参照し、モデルや実行ファイルは取得・実行していない。
競合の機能・速度に関する記述はupstreamの説明であり、Vokraとの独立比較実験ではない。
以下のcommitは調査時のdefault branch HEAD。実測のない速度・精度・メモリの優劣は付けない。

ユーザーのspeech.cppは、まず日本語圏の`nyosegawa/speech.cpp`と解釈する。
同名の[NairoDorian/speech.cpp](https://github.com/NairoDorian/speech.cpp)も存在し、
両者の機能・ベンチマーク・Rust bindingを混同しない。

## 一次資料による比較

| OSS／固定した資料 | 上流が説明している主な対象・方式 | Vokraで差別化を検討する軸 |
| --- | --- | --- |
| [nyosegawa/speech.cpp](https://github.com/nyosegawa/speech.cpp/blob/32cbe28e16bff82ca6abadeef3b6e7599ab8d0d3/README.md) | C++／ggml。ASR・TTS・VAD、CPU/Metal/Vulkan、単体CLI・C API、live transcription、HTTP APIとworker | 「音声全部入り」では差別化できない。Rust-nativeな実装と依存境界、ライブラリとしての音声処理部品・状態管理を中心にする |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/README.md) | Whisper中心のC/C++／ggml ASR。多様なhardware/backend最適化、C API、VADもある | Whisper専用の置き換えや速度勝負を主張しない。複数の音声処理をRust側の共通境界で扱う用途を説明する |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/blob/1a9d234ab95567eba8c2e89873a87d70b79633e2/README.md) | ONNX Runtimeベースの広い音声task、端末・組み込み対応、多言語binding。Rust APIもある | タスク数・binding数は独自性にならない。ONNX Runtimeを組み込まずnative Rust経路を選べる設計を説明する |
| [Candle](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/README.md) | Rustの汎用ML framework。音声以外のモデルもあり、Whisper等の音声exampleもある | Rustという言語だけでは差別化できない。音声frontend／codec／streaming stateを含む用途特化のruntime境界を説明する |
| [NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/caa3f47c7c75fa2e9e6340272ebad57c1647dd44/README.md) | NVIDIAの音声モデル群向けnative C++／ggml runtime。ASR以外の合成・話者・翻訳等も説明している | 「NVIDIAモデル以外も含むRust音声基盤」という設計方向は示せるが、対応数・速度・最新モデル対応で勝っているとは言わない |

全5資料のHEADはGitHub APIで再確認した。speech.cpp、sherpa-onnx、
NeMo-Speech.cppは固定commitのraw READMEも照合し、残る2件は固定commitの
GitHub本文を確認した。時系列の異なるbenchmark値を混ぜない。

## 告知で中心にするポジショニング

**Rustアプリへ組み込むための、依存境界を明示した音声推論・音声処理runtime。**

| 根拠を持って説明するVokraの選択 | ローカルの根拠 | 説明の限界 |
| --- | --- | --- |
| デフォルトRust workspaceの外部Cargo依存ゼロ | root `Cargo.lock`と`check-zero-deps.sh`。この調査でもgate成功 | offline変換・reference・独立integrationやOS/GPU SDKまで無依存とは言わない。無脆弱性・小メモリ・高速の証明でもない |
| 音声frontend、推論、codec、streamingを扱うRust側の共通engine境界 | `crates/vokra-core/src/engines.rs`、backend APIとCLI routing | 宣言済みの全モデルが動くことや、競合に同じAPIがないことを意味しない |
| 選択backendで未対応なら明示的に拒否する設計 | backend guide、モデル別preflightとunsupported-path error | 最新の全モデルについてMetal/no-fallback実測済みという意味ではない |
| provenance／licenseを実行時policyと公開gateに分離 | `crates/vokra-core/src/compliance/mod.rs`、license audit・公開runbook | 法令適合、全weightの商用可、競合に同様のgateがない、とは主張しない |
| 検証したモデル・HEAD・入力と未検証範囲を分けて公開 | exact-head handoffと次版smoke receipt | 品質検証は競合にもある。透明な運用方針として示し、独自機能とは言わない |

これは競合に対する網羅的な優位性の証明ではなく、利用者が選ぶ理由の整理である。
自前runtime/kernelを維持する負担、対応モデルごとの未検証部分、導入体験の未整備は
Vokra側の課題として明示する。すぐ使えるCLI／サーバー体験については
speech.cppの整備状況を参考にし、現時点でVokraが上回るとは扱わない。

## 独自性として使わない表現

- 「Python不要」「GGUF」「C API」「オフライン」「Metal」「ASR＋TTS＋VAD」だけで
  speech.cppとの差を説明しない。これらは競合にもある。
- 「Rustだから安全／速い／小さい」「C++製品はRustから呼べない」と言わない。
- `136 code/artifact-full`を「136モデル動作保証」と置き換えない。
- 最速・最軽量・最高精度・商用利用を全面保証・Metal完全対応・production-readyを
  裏付けなしに書かない。
- 競合のbenchmarkと異なるdevice/model/quantization/input条件を並べ、倍率を作らない。
- 同じGGUF拡張子でも、互換なtensor/metadata contractがあるとは限らない。

## 告知案と公開条件

同日の[実CPU検証](release-0.4.0-preview-cpu-results-2026-10-17.md)で、main
`9ee5834d`のWhisper base ASR／Silero VAD v5のspeech/silenceはLinux CPUで成功した。
これで具体的な開発状況は説明できるが、配布版の公開、Apple対応、精度・速度の優位性は
依然として証明していない。下のpreview公開用templateは実Releaseができてから使う。

### 現時点で使える開発状況の説明案

> Rust製の音声推論・音声処理runtime「Vokra」を開発しています。
> デフォルトruntimeの外部Cargo依存はゼロ。音声frontendやstreaming stateを含め、
> Rustアプリへ組み込むための基盤を目指しています。
> Whisper baseの音声認識とSilero VAD v5をLinux CPUで実音声検証しました。
> 開発版の対応範囲・制限と検証結果を公開しています。配布版は準備中です。

### 検証後だけ使うpreview告知テンプレート

> Vokraの次版previewを準備しました。
> Rust-nativeな音声runtimeとして、Whisper baseのASRとSilero VAD v5をLinux CPUで検証。
> 〔実際に確認した導入手順／デモ〕から試せます。
> 全モデル対応・Metal検証完了ではありません。対応範囲と制限：〔URL〕

括弧の項目を実結果で埋め、release URLと実際の配布物が存在するまでは後者を投稿しない。
競合を貶める比較ではなく、誰が何のためにVokraを選ぶのかを先に示す。
X投稿・タグ・GitHub Release作成・モデル公開はいずれもこの記録では実行していない。

## 次に測るなら

同一machine／同一audio／同等modelとquantizationで、cold/warm、compile flags、threads、
first-result latency、end-to-end時間、peak RSS、ASRのWER/CERを分ける。
VADはラベル付き入力でprecision/recall、streamingは境界・reset・最終flushも確認する。
今回の代表経路smokeはこの競合benchmarkを実施したことにはしない。
