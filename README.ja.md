# Vokra

[English](README.md) | **日本語**

**Vokra** は、音声 AI — TTS・ASR・Speech-to-Speech・ボイスコンバージョン・話者識別・VAD — に特化した推論ランタイムです。音声ワークロード向けの ONNX / ONNX Runtime 代替として Rust で構築しています。

- **発音**: "vo-krah"（英語）/「ヴォクラ」（日本語）
- **ライセンス**: [Apache-2.0](LICENSE)
- **ステータス**: **v0.1 spike、活発に開発中** — プレリリース段階。API・ファイルフォーマット・モデル対応はいずれも不安定かつ未完成です。

## Vokra とは

汎用ランタイムは音声モデルを慢性的に十分サポートできていません。STFT/iSTFT とストリーミング状態、vocoder の数値計算、ニューラルコーデック（RVQ/FSQ）のデコード、Flow Matching サンプラー、beam search / CTC / RNN-T デコード、VAD、話者埋め込み — これらはいずれも壊れやすいグラフエクスポートやホスト側の繋ぎコードに追いやられてしまいます。Vokra はこれらを第一級のネイティブオペレータにします。

主要な設計方針:

- **Rust コア + C ABI**（cbindgen で生成）により Unity / Godot その他のエンジン・言語バインディングへ対応。Apache-2.0 で GPL/LGPL 依存はありません。
- GGUF（`vokra.*` 音声メタデータ chunk 付き）と safetensors からの **weight 直接ロード**。**ランタイムは ONNX グラフを一切ロードしません** — ONNX モデルはオフライン変換ツールでのみ扱うため、ランタイムは onnx/protobuf 依存を持ちません。
- **音声ファーストなオペレータ集合**: STFT/iSTFT（window/hop/norm/RFFT を明示的な属性として持つ）、mel filterbank、リサンプリング、vocoder chain、Flow Matching サンプラー、コーデックデコード、beam search / CTC / RNN-T、ストリーミング KV cache、VAD、音声強調（AEC/denoise）、話者埋め込み、F0 抽出、音声電子透かし（EU AI Act Article 50 対応準備）。
- **CPU を第一級バックエンドに**（x86-64 は SSE2 ベースラインから AVX2/AVX-512/AMX まで、ARM64 は NEON から SVE/SME まで、ランタイムディスパッチ付き）。その後、GPU/NPU アクセラレーションを段階的に追加: Metal、CUDA、Vulkan、WebGPU、CoreML、QNN。
- **全プラットフォームがスコープ**: Windows / macOS / Linux / Android / iOS / Web、および x86-64・ARM64 サーバ。ロードマップが段階化しているのは各バックエンドが公式アクセラレーションを*いつ*得るかであって、あるプラットフォームを*対応するか否か*ではありません。

## ステータスと設計文書

本プロジェクトは **v0.1 spike** フェーズ（Rust scaffold、GGUF ローダー、STFT/iSTFT/mel op、Silero VAD、Whisper base、piper-plus native TTS、CPU バックエンド、C ABI、Unity デモ）にあります。まだ本番利用できる段階ではありません。

公開リファレンス文書（日本語）:

- [docs/license-audit.md](docs/license-audit.md) — モデル / 依存ライセンス監査
- [docs/legal-compliance.md](docs/legal-compliance.md) — EU AI Act, SB 942, ELVIS Act, C2PA 対応

要求定義・要件定義・成果物定義・マイルストーン計画の詳細は作者が非公開で管理しており、下記のロードマップ概要がその要約です。

## ロードマップ

期間は Claude Code 主導の実装モデルにおけるエンジニアリング見積りです。そこから導かれるカレンダー上の日付はいずれも**あくまで目安**であり、コミットメントではありません。

| フェーズ | 見積り期間 | 焦点 |
|---|---|---|
| **v0.1 spike**（現在） | 1.5〜2ヶ月 | Rust scaffold、GGUF ローダー + `vokra.*` メタデータ、STFT/iSTFT/mel op、Silero VAD、Whisper base、piper-plus native TTS、CPU バックエンド（AVX2/NEON）、C ABI、Unity デモ、リポジトリ public 化 + CI ゲート |
| v0.1 MVP | 1.5〜2.5ヶ月 | Silero VAD v5 + Whisper base の公式サポート。リリース直後にモデル parity チェックポイント |
| v0.5 | 2.5〜4ヶ月 | Metal バックエンド、CUDA バックエンド着手、Kokoro-82M、Whisper large-v3/turbo、OpenAI 互換サーバ API |
| v1.0 | 4〜5ヶ月 | CUDA 完成、Vulkan、CosyVoice2、Voxtral、RVV 1.0 ベースライン |
| v1.5 | 4〜5ヶ月 | WebGPU/WASM、Sesame CSM-1B、Moshi（full-duplex + AEC）、全プラットフォーム公式サポート完了 |
| v2.0 | 8ヶ月以上 | CoreML（ANE）/ QNN delegate、MCU tier 再評価 |

v2.0 までの累計見積り: **20〜25ヶ月**。v0.1 spike は piper-plus native TTS 実装をスコープに追加したことで 1〜1.5ヶ月から 1.5〜2ヶ月へ延長されました（2026-07-02 の決定）。

## piper-plus 統合（native TTS）

[piper-plus](https://github.com/ayutaz/piper-plus) は、プロジェクトオーナーによる MIT ライセンスの Piper フォークです（eSpeak-NG に依存しない 8 言語 G2P、MB-iSTFT-VITS2 decoder）。Vokra はこれを標準 TTS レイヤとして、かつ **Vokra 初の native 実装 TTS モデル** として統合します（2026-07-02 決定）:

- MB-iSTFT-VITS2 の推論スタック（text encoder / duration predictor / flow / MB-iSTFT decoder）を Rust でネイティブ再実装します。既存の ONNX ベース実装を wrap する当初計画は廃止しました。
- **e2e の推論経路に onnxruntime は一切含まれません。** voice model はオフラインで GGUF に変換し、ランタイムは GGUF のみをロードします。
- G2P（テキスト前処理、8 言語: JA/EN/ZH/ES/FR/PT/SV/KO）は当面 piper-plus のものを流用します。Rust 移植は将来再評価します。

## C ABI の利用

Vokra は単一の C ヘッダ [`include/vokra.h`](include/vokra.h) を公開します（cbindgen 生成。`scripts/gen-c-abi.sh` で再生成）。`vokra-capi` crate をビルドすると共有ライブラリと静的ライブラリが生成されます:

```sh
cargo build -p vokra-capi --release
# -> target/release/libvokra.dylib | libvokra.so | vokra.dll  (+ libvokra.a)
```

セッションは GGUF モデルから作成します。アーキテクチャはファイルの `vokra.model.arch` メタデータから検出され、対応するタスクが自動的に配線されます（Whisper → ASR、Silero VAD → VAD ストリーム、piper-plus → TTS）。すべての関数は `vokra_status_t` を返します（`VOKRA_OK` は 0）。エラー時はスレッドごとのメッセージを `vokra_last_error()` から取得できます。Vokra が確保した出力は、対応する `vokra_*_free` / `vokra_*_destroy` 関数で解放します。

```c
#include "vokra.h"

vokra_session_t *session = NULL;
if (vokra_session_create_from_file("whisper-base.gguf", &session) != VOKRA_OK) {
    fprintf(stderr, "load failed: %s\n", vokra_last_error());
    return 1;
}

char *text = NULL;
if (vokra_asr_transcribe(session, pcm, num_samples, 16000, &text) == VOKRA_OK) {
    printf("%s\n", text);
    vokra_string_free(text);
}
vokra_session_destroy(session);
```

ヘッダに対してコンパイルし、共有ライブラリをリンクします:

```sh
cc app.c -Iinclude -Ltarget/release -lvokra -Wl,-rpath,target/release -o app
```

実行可能な e2e サンプル（ASR / TTS / VAD）は [`tests/capi/`](tests/capi) にあります。`scripts/run-capi-smoke.sh` がビルドと実行を行います。M0（v0.1 spike）の ABI は **安定していません** — v1.0 の semver コミットまでは互換性を壊す変更があり得ます。

## 対応予定モデル

公式 model zoo は **Apache-2.0 / MIT の weight のみ** を配布します。完全な監査は [docs/license-audit.md](docs/license-audit.md) を参照してください。

| モデル | タスク | ライセンス（code / weight） | 商用利用 | 対応予定 |
|---|---|---|---|---|
| Silero VAD v5 | VAD | MIT / MIT | 可 | v0.1 MVP |
| Whisper base/small/medium/large-v3/turbo | ASR | MIT / MIT | 可 | v0.1 MVP（base）、v0.5（large-v3/turbo） |
| piper-plus | TTS | MIT / MIT | 可 | v0.1 spike（native 実装） |
| Kokoro-82M | TTS | Apache-2.0 / Apache-2.0 | 可 | v0.5 |
| CosyVoice2 | TTS / S2S | Apache-2.0 / Apache-2.0 | 可 | v1.0 |
| Voxtral (Mistral) | ASR / S2S | Apache-2.0 / Apache-2.0 | 可 | v1.0 |
| Sesame CSM-1B | S2S | Apache-2.0 / Apache-2.0 | 可 | v1.5 |
| Moshi (Helium + Mimi) | S2S | Apache-2.0 / CC-BY 4.0（attribution 要） | 可（credit 要） | v1.5 |
| F5-TTS | TTS | MIT / **CC-BY-NC 4.0** | **不可（非商用 weight）** | エンジン対応のみ。weight は公式 zoo から除外、research flag の背後 |
| Fish-Speech v1.4/v1.5 | TTS | Apache-2.0 / **CC-BY-NC-SA 4.0** | **不可（非商用 weight）** | エンジン対応のみ。weight 除外、research flag |
| RVC v2 / GPT-SoVITS | VC | MIT / 不明 | 制限あり（学習データ懸念） | 別リポジトリ `vokra-voiceclone-experimental` |
| Bark (Suno) | TTS | MIT / MIT（Suno ポリシーで voice-cloning 再学習は禁止） | 制限あり | v2.0+（検討中、research flag） |
| StyleTTS 2 | TTS | MIT / 不明（監査待ち） | 制限あり | v2.0+（監査後） |
| Matcha-TTS | TTS | MIT / MIT | 可 | v2.0+ |

補足:

- **F5-TTS と Fish-Speech の weight は CC-BY-NC(-SA) ライセンスであり、いかなる公式 Vokra 配布物にも含まれません。** エンジンは明示的な research flag により研究目的で実行できます。
- ボイスクローニング（RVC v2、GPT-SoVITS、話者クローン）は法的理由（ELVIS Act / NO FAKES Act）により `vokra-voiceclone-experimental` リポジトリへ完全に分離されています。zero-shot TTS 用の話者埋め込みは core に残します。
- Piper（OHF-Voice/piper1-gpl）は **非対応** です（GPL-3.0 + eSpeak-NG GPL-3.0）。Piper 系の統合は piper-plus のみです。

## コミュニティ

- **質問・議論**: [GitHub issue](https://github.com/ayutaz/vokra/issues) をご利用ください。
- **Issue / Pull Request**: [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。すべての変更は CI 品質ゲート付きの PR を経由します。

## ライセンス

[Apache License, Version 2.0](LICENSE) の下でライセンスされます。

追加のライセンス・配布告知 — BigVGAN のスクラッチ再実装ポリシーと NVIDIA ランタイムの非同梱ポリシー — は [NOTICE](NOTICE) に記録されています。
