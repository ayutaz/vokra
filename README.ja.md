# Vokra

[English](README.md) | **日本語**

**Vokra** は、音声 AI — TTS・ASR・Speech-to-Speech・ボイスコンバージョン・話者識別・VAD — に特化した推論ランタイムです。音声ワークロード向けの ONNX / ONNX Runtime 代替として Rust で構築しています。

- **発音**: "vo-krah"（英語）/「ヴォクラ」（日本語）
- **ライセンス**: [Apache-2.0](LICENSE)
- **ステータス**: **プレリリース、活発に開発中** — v0.5（M2）と v0.9（M3）は `main` にマージ済み。v1.0-rc（M4）の機能実装は開発ブランチ上で完了（依頼者検証待ち）。タグ付きリリースは `v0.1.0` のみで、API・ファイルフォーマット・モデル対応はいずれも不安定かつ未完成です。

## Vokra とは

汎用ランタイムは音声モデルを慢性的に十分サポートできていません。STFT/iSTFT とストリーミング状態、vocoder の数値計算、ニューラルコーデック（RVQ/FSQ）のデコード、Flow Matching サンプラー、beam search / CTC / RNN-T デコード、VAD、話者埋め込み — これらはいずれも壊れやすいグラフエクスポートやホスト側の繋ぎコードに追いやられてしまいます。Vokra はこれらを第一級のネイティブオペレータにします。

主要な設計方針:

- **Rust コア + C ABI**（cbindgen で生成）により Unity / Godot その他のエンジン・言語バインディングへ対応。Apache-2.0 で GPL/LGPL 依存はありません。
- GGUF（`vokra.*` 音声メタデータ chunk 付き）と safetensors からの **weight 直接ロード**。**ランタイムは ONNX グラフを一切ロードしません** — ONNX モデルはオフライン変換ツールでのみ扱うため、ランタイムは onnx/protobuf 依存を持ちません。
- **音声ファーストなオペレータ集合**: STFT/iSTFT（window/hop/norm/RFFT を明示的な属性として持つ）、mel filterbank、リサンプリング、vocoder chain、Flow Matching サンプラー、コーデックデコード、beam search / CTC / RNN-T、ストリーミング KV cache、VAD、音声強調（AEC/denoise）、話者埋め込み、F0 抽出。CC-BY-NC ライセンスの weight を research フラグなしではデフォルト経路から締め出す weight ライセンス compliance ゲートを備えます（音声電子透かしは設計済みですが未有効化）。
- **CPU を第一級バックエンドに**（x86-64 は SSE2 ベースラインから AVX2/AVX-512/AMX まで、ARM64 は NEON から SVE/SME まで、ランタイムディスパッチ付き）。その後、GPU/NPU アクセラレーションを段階的に追加: Metal、CUDA、Vulkan、WebGPU、CoreML、QNN。**Metal と CUDA バックエンドは実装済みで実機検証済み**（下記ステータス参照）。GPU 対応は zero-dependency の手書き FFI（`metal-rs` / `cudarc` 等の binding crate を使わない）で、未対応 op を CPU に暗黙フォールバックせず明示エラーにします。
- **全プラットフォームがスコープ**: Windows / macOS / Linux / Android / iOS / Web、および x86-64・ARM64 サーバ。ロードマップが段階化しているのは各バックエンドが公式アクセラレーションを*いつ*得るかであって、あるプラットフォームを*対応するか否か*ではありません。

## ステータスと設計文書

v0.1 spike と v0.1 MVP は完了。**v0.5**（Metal / CUDA GPU バックエンド）と **v0.9**（CUDA 完成、Vulkan、CosyVoice2、Voxtral、RVV 1.0）は `main` にマージ済みで、**v1.0-rc**（M4: WebGPU/WASM、Sesame CSM-1B、Moshi、全プラットフォームサポート）の機能実装は開発ブランチ上で完了しています。まだ本番利用できる段階ではありませんが、以下は実装・検証済みです:

- **CPU 音声スタック**: Silero VAD、Whisper（base〜large-v3、正しい転写のための detokenizer を埋め込み）、実 8 言語 G2P を配線した piper-plus native TTS、zero-shot 声クローン用の native CAM++ 話者エンコーダ。いずれも参照ランタイム（onnxruntime / PyTorch）に対し FP32 `atol = 0.01` で数値一致を検証済み。**実 checkpoint 検証**（Apple M1、同一 weight で onnxruntime 1.19.2 CPU と比較）: Whisper base/small/medium/turbo の転写は **ONNX Runtime と byte 一致**（WER 同値）、piper は near-bit-exact（mel-L1 ≈ 0.003）、Mimi/DAC/WavTokenizer の codec parity 全 PASS、DeepFilterNet3 は SI-SNR 差 2.0e-7 dB で upstream に一致（[`docs/bench-baselines/m1-real-weight-eval-2026-07-16/`](docs/bench-baselines/m1-real-weight-eval-2026-07-16/)）。
- **CPU 速度**（rig 限定: Apple M1・8 スレッド、同一マシン・同一 weight の onnxruntime 1.19.2 CPU 比。方法論と生ログは [`docs/bench-baselines/m5-14-final-2026-07-18/`](docs/bench-baselines/m5-14-final-2026-07-18/)）: packed-GEMM / ベクトル化 wave の後、**Whisper base は ONNX Runtime 比 約 2.5 倍高速、whisper-turbo 約 2.7 倍、Silero VAD 約 2.3 倍**。whisper-medium/small は ORT の 1.17〜1.24 倍以内、piper は約 2.2 倍以内。全最適化は構成上 bit-identical（parity 許容誤差の変更ゼロ）。
- **GPU バックエンド**（`vokra-backend-metal` / `vokra-backend-cuda`）: データ運搬グラフ評価器 + モデル単位のディスパッチ seam。**Whisper が Metal（Apple M1 で検証）と CUDA（RTX 4090 で検証）の両方で e2e 動作**し、greedy 出力が CPU 経路と完全一致。GPU 中間を **Whisper encoder 全体**（全 pre-norm block を 1 submission に融合）と**自己回帰 decoder の各ステップ**（causal attention 融合 + on-device KV cache）にわたって device 常駐にし、host↔device readback を小さな定数まで削減。これにより **Whisper large-v3 が RTX 4090 で RTF < 0.15**（30 秒音声で実測 0.081〜0.115、個体差・実行時条件で変動。CPU 経路の数倍）を達成。両バックエンドとも外部 crate ゼロの手書き FFI で、CI でも検証しています。
- **ツール**: `vokra-cli`（`run` / `convert` / `bench`、GPU RTF 計測用の `bench --backend cpu|metal|cuda`）、オフライン `vokra-convert`、`vokra-eval` メトリクス crate、true zero-copy `mmap` GGUF ローディング。
- **配布形態**: iOS **XCFramework + Swift Package**（arm64 実機 + Simulator slice、静的リンク、`DllImport("__Internal")` 対応）、**Unity UPM パッケージ**（`com.vokra.unity`、IL2CPP-safe callback + Android `persistentDataPath` ヘルパー）、**Python bindings**（pure `ctypes`、`pyo3` 不使用、`cibuildwheel` で PyPI wheel を発行）。詳細は [`bindings/`](bindings) と [`Package.swift`](Package.swift)。
- **サーバ**: [`integrations/vokra-server`](integrations/vokra-server) は隔離ワークスペース（独自 `Cargo.lock`）で、4 種の HTTP 互換 API を公開します — **OpenAI Whisper**（`/v1/audio/transcriptions`、faster-whisper ドロップイン）、**vLLM**（`/v1/completions`、`/v1/chat/completions`）、**piper-plus HTTP**（`/api/tts`）、**Wyoming Protocol**（Home Assistant Voice バックエンド）。ルートワークスペースから除外することで、コアの zero-dependency 不変条件を維持しています。
- **グラフ融合**: log-mel フロントエンド融合（STFT + magnitude + mel + log を 1 kernel に集約）を AVX2 / NEON 特化で実装。`mel-frontend` の `vokra-cli bench` タスクと CI 上の 5% regression ゲートで測定・保護されます。
- **量子化ポリシー**: config 駆動の層別量子化（`W4A16Q4K` / `W8A8Int8` / `FP16` / `FP32`）と最低精度 registry。Vocos / BigVGAN のように FP16 が必須の op には INT8 指定を拒否し、`vokra-convert` 実行時に `vokra.quant.*` chunk へ焼き込みます。
- **コンプライアンスゲート**: research フラグ enforcement 層。CC-BY-NC / CC-BY-NC-SA weight（F5-TTS / Fish-Speech / EnCodec）を明示的な opt-in なしにはロードさせません。判定は compliance API と同じ `vokra.provenance.*` chunk から行います。

上記はすべて Vokra の **zero-external-dependency** 不変条件を保っています（解決後の依存グラフは first-party の `vokra-*` crate のみ、CI で強制）。

公開リファレンス文書（日本語）:

- [docs/license-audit.md](docs/license-audit.md) — モデル / 依存ライセンス監査
- [docs/legal-compliance.md](docs/legal-compliance.md) — EU AI Act, SB 942, ELVIS Act, C2PA 対応

要求定義・要件定義・成果物定義・マイルストーン計画の詳細は作者が非公開で管理しており、下記のロードマップ概要がその要約です。

## ロードマップ

期間は Claude Code 主導の実装モデルにおけるエンジニアリング見積りです。そこから導かれるカレンダー上の日付はいずれも**あくまで目安**であり、コミットメントではありません。

| フェーズ | 見積り期間 | 焦点 |
|---|---|---|
| v0.1 spike | 1.5〜2ヶ月 | Rust scaffold、GGUF ローダー + `vokra.*` メタデータ、STFT/iSTFT/mel op、Silero VAD、Whisper base、piper-plus native TTS、CPU バックエンド（AVX2/NEON）、C ABI、Unity デモ、リポジトリ public 化 + CI ゲート — **完了** |
| v0.1 MVP | 1.5〜2.5ヶ月 | K-quant ローダー、engine、streaming、resample、`vokra-cli` / `vokra-eval`、実 8 言語 G2P 配線、native CAM++ zero-shot クローン、`vokra-mmap` — **完了** |
| v0.5 | 2.5〜4ヶ月 | Metal + CUDA バックエンド（グラフ評価器 + モデル単位の GPU ディスパッチ。Whisper が両方で e2e、M1 / RTX 4090 で検証）、Whisper large-v3 変換 + tokenizer、encoder 全体・decoder 各ステップの device 常駐（large-v3 が RTX 4090 で RTF < 0.15、実測 0.081〜0.115）、Kokoro-82M、`vokra-server`（4 種の HTTP 互換 API）、`bench --backend` — **完了**（`main` にマージ済み） |
| v0.9 | 4〜5ヶ月 | CUDA 完成、Vulkan、CosyVoice2、Voxtral、RVV 1.0 ベースライン — **完了**（`main` にマージ済み） |
| **v1.0-rc**（現在） | 4〜5ヶ月 | WebGPU/WASM（**実装済**: 生 WebGPU import shim + WASM SIMD128 2-artifact CPU パスでブラウザ Whisper base、npm パッケージ CD — [docs/tutorials/web.ja.md](docs/tutorials/web.ja.md) 参照）、Sesame CSM-1B、Moshi（full-duplex + AEC）、全プラットフォーム公式サポート — **開発ブランチ上で機能完成**（依頼者検証待ち） |
| v1.0 GA | 8ヶ月以上 | CoreML（ANE）/ QNN delegate、MCU tier 再評価、商用 GA、C ABI 凍結（v1.0 以降 semver 準拠） |

v1.0 GA までの累計見積り: **20〜25ヶ月**。バージョンラベルは 2026-07-14 に再割当されました: 従来 v2.0 までに計画していたスコープを v1.0 として出荷します（旧 v1.0 / v1.5 フェーズは v0.9 / v1.0-rc に）。v1.0-rc は semver プレリリースであり、C ABI は v1.0 GA タグで凍結されます。v0.1 spike は piper-plus native TTS 実装をスコープに追加したことで 1〜1.5ヶ月から 1.5〜2ヶ月へ延長されました（2026-07-02 の決定）。

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
| Whisper base/small/medium/large-v3/turbo | ASR | MIT / MIT | 可 | v0.1 MVP（base）、v0.5（large-v3）、v1.0-rc（small/medium/turbo） |
| piper-plus | TTS | MIT / MIT | 可 | v0.1 spike（native 実装） |
| Kokoro-82M | TTS | Apache-2.0 / Apache-2.0 | 可 | v0.5 |
| CosyVoice2 | TTS / S2S | Apache-2.0 / Apache-2.0 | 可 | v0.9 |
| Voxtral (Mistral) | ASR / S2S | Apache-2.0 / Apache-2.0 | 可 | v0.9 |
| Sesame CSM-1B | S2S | Apache-2.0 / Apache-2.0 | 可 | v1.0-rc |
| Moshi (Helium + Mimi) | S2S | Apache-2.0 / CC-BY 4.0（attribution 要） | 可（credit 要） | v1.0-rc |
| F5-TTS | TTS | MIT / **CC-BY-NC 4.0** | **不可（非商用 weight）** | エンジン対応のみ。weight は公式 zoo から除外、research flag の背後 |
| Fish-Speech v1.4/v1.5 | TTS | Apache-2.0 / **CC-BY-NC-SA 4.0** | **不可（非商用 weight）** | エンジン対応のみ。weight 除外、research flag |
| RVC v2 / GPT-SoVITS | VC | MIT / 不明 | 制限あり（学習データ懸念） | 別リポジトリ `vokra-voiceclone-experimental` |
| Bark (Suno) | TTS | MIT / MIT（Suno ポリシーで voice-cloning 再学習は禁止） | 制限あり | v1.0 GA 以降（検討中、research flag） |
| StyleTTS 2 | TTS | MIT / 不明（監査待ち） | 制限あり | v1.0 GA 以降（監査後） |
| Matcha-TTS | TTS | MIT / MIT | 可 | v1.0 GA 以降 |

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
