# Migration Guide（日本語）

[English](migration-guide.md) | **日本語**

**ONNX Runtime**・**whisper.cpp**・**sherpa-onnx** ですでに音声推論パイプ
ラインを組んでいる方向けに、Vokra への切替時に何が変わるか、API 対応
表、モデル変換、パフォーマンスの目安をまとめます。

## 1. 概念レベルで変わること

| 観点 | ONNX Runtime / sherpa-onnx | whisper.cpp | Vokra |
|---|---|---|---|
| ロードするファイル形式 | ONNX（Protobuf） | GGUF（ggml-audio） | GGUF（`vokra.*` 音声 chunk） |
| 音声 op（STFT / iSTFT / mel / VAD 状態 / Flow Matching sampler / KV cache） | ホストコード + graph glue の寄せ集め | Whisper 特化 inline | **第一級ネイティブオペレータ** |
| バックエンドの継目 | Execution Providers（op カバレッジが非対称） | CPU + オプションで CUDA/Metal | CPU + Metal + CUDA（Vulkan / WebGPU / CoreML / QNN は段階対応） |
| silent CPU fallback | 場合による | なし | **なし — 明示エラー（FR-EX-08）** |
| ランタイムに ONNX | あり | なし | **なし** — ONNX はオフライン変換のみ |
| weight ライセンス強制 | 外部 | 外部 | 内蔵の `vokra.provenance.*` ゲート（research flag なしで CC-BY-NC を拒否） |
| 配布形態 | ORT バイナリ + アプリ | 単一バイナリ | 単一バイナリ。**root `Cargo.lock` に `vokra-*` crate のみ** |

音声固有の関心事（フレーム精度の STFT、レイヤーごとの KV cache、スト
リーミング状態、bit-exact frontend）をアプリコードからランタイムへ移す
ことが設計目標です。

## 2. ONNX Runtime / sherpa-onnx から

### 2.1 API 対応

| ONNX Runtime | Vokra |
|---|---|
| `Ort::Session(env, "model.onnx", ...)` | `vokra_session_create_from_file("model.gguf", &s)` |
| `session.Run(inputs, outputs)` | `vokra_asr_transcribe(s, pcm, n, sr, &out)`（`vokra.model.arch` からタスク自動選択） |
| `SessionOptions::SetExecutionProviderCUDA(...)` | `vokra_session_set_backend(s, VOKRA_BACKEND_CUDA)`（`--features cuda` でビルド） |
| Custom op 登録 | すでに第一級化されているか、明示エラーで拒否 |
| `sherpa_onnx_offline_recognizer_*` | `vokra_session_create_from_file`（Whisper GGUF）+ `vokra_asr_transcribe` |
| `sherpa_onnx_online_recognizer_*` | `vokra_stream_open` + `vokra_stream_push_pcm` + `vokra_stream_poll` |

### 2.2 モデル変換

Vokra は GGUF のみをロードします。ONNX モデルは**オフライン**で変換し、
ランタイムは `onnxruntime` / `protobuf` / `abseil` にリンクしません
（FR-LD-05）:

```sh
# Silero VAD v5
vokra-cli convert --model silero-vad --input silero_vad.onnx --output silero_vad.gguf

# piper-plus voice
vokra-cli convert --model piper-plus \
  --input voice.onnx --config voice.config.json --output voice.gguf

# CAM++ speaker encoder
vokra-cli convert --model campplus --input campplus.onnx --output campplus.gguf
```

Whisper は safetensors 経路（upstream `openai/whisper-*`）を推奨:

```sh
vokra-cli convert --model whisper \
  --input model.safetensors --output whisper.gguf
```

### 2.3 ONNX を離れて得られるもの

- **STFT / iSTFT のセマンティクスが正確**: window / hop / normalization
  / RFFT はすべて明示的な attribute で、opset drift の影響を受けません。
- **KV cache をランタイム側で管理**: decoder の 3 分割や 6GB の静的
  cache は不要になります。
- **opset upgrade のリスクなし**: `torch.onnx.export` の dynamo /
  scriptmodule 分裂は無関係 — Vokra は safetensors / checkpoint を
  直接読み、ネイティブモデル実装（whisper.cpp 型）で動かします。

### 2.4 折り合いをつけるべき点

- **Custom ONNX op**（contrib operators、`com.microsoft.*` 等）は
  **移行しません**。必要な場合は op シグネチャを添えて issue を上げて
  ください（第一級化を検討します）、あるいはその手順だけホストコード
  に残してください。
- **Windows Bitcode** など ORT 特有のパッケージ話は消えます — Vokra
  は単一の `libvokra.dll` / `.dylib` / `.so` を配布します。

## 3. whisper.cpp から

### 3.1 API 対応

| whisper.cpp | Vokra |
|---|---|
| `whisper_init_from_file("model.bin")` | `vokra_session_create_from_file("whisper.gguf", &s)` |
| `whisper_full_default_params(WHISPER_SAMPLING_GREEDY)` + `whisper_full` | `vokra_asr_transcribe(s, pcm, n, 16000, &out)` |
| `whisper_full_get_segment_text` | 全テキストが 1 本の `char*` で返る（セグメント別 API は今後） |
| チャンク間で `whisper_state` を再利用 | `vokra_stream_open` + `vokra_stream_push_pcm`（streaming ASR は v0.5+） |
| `whisper.cpp` の GGUF | **互換なし** — Vokra の GGUF は `vokra.*` 音声メタデータ chunk を持ち、whisper.cpp は読めません（逆も然り）。safetensors から変換してください。 |

### 3.2 モデル変換

```sh
# Hugging Face safetensors から — base / small / medium / large-v3 / turbo 対応
vokra-cli convert --model whisper \
  --input openai_whisper-large-v3/model.safetensors \
  --output whisper-large-v3.gguf
```

量子化プリセット: `--quantize q4_k` / `q5_k` / `q6_k`（それぞれ
`--policy-preset whisper_q4_k` 等のエイリアス）。層別ポリシーの詳細は
`docs/design/quantization-policy.md` 参照。

### 3.3 パフォーマンスの目安

- **Whisper base CPU**: RTF < 0.3 が目標（whisper.cpp と同 CPU で同等。
  両者とも K-quant と手書きカーネルを使います）。
- **Whisper large-v3、RTX 4090**: e2e で **RTF < 0.15**（実測
  0.081〜0.115）。encoder + decoder-step とも device 常駐 + FA v2
  causal-attention 融合カーネル。whisper.cpp の CUDA 経路は同じ GPU で
  通常 2〜3 倍遅くなります（cuBLAS 経由で融合カーネルではないため）。

### 3.4 折り合いをつけるべき点

- **言語自動判定**は現状 Whisper prompt 経由。`detect_language()` の
  第一級ショートカットは今後。
- **単語レベルのタイムスタンプ**は C ABI から未提供。当面は Rust API
  の `beam_search` op 経路を使ってください。

## 4. `faster-whisper`（Python）から

Python バインディング（PyPI の `vokra`）は `faster-whisper` の主要な表
面をほぼ 1:1 で対応します:

```python
# faster-whisper
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda")
segments, _ = m.transcribe("speech.wav")
text = " ".join(s.text for s in segments)

# vokra
from vokra import Session
with Session.open("whisper-large-v3.gguf") as s:
    pcm, sr = read_wav_mono_f32(open("speech.wav", "rb"))
    text = s.transcribe(pcm, sr)
```

**HTTP クライアントのドロップイン置換**が欲しい場合は
[`integrations/vokra-server`](../integrations/vokra-server) を起動して
ください。OpenAI Whisper の `/v1/audio/transcriptions`（faster-whisper
互換）、vLLM の `/v1/completions` + `/v1/chat/completions`、piper-plus
HTTP `/api/tts`、および Home Assistant 用の Wyoming Protocol を公開しま
す。クライアントコードは URL だけ変えれば動きます。

## 5. `piper` / `piper-plus` から

Vokra の piper-plus 統合は `piper-plus` の推論スタック（MB-iSTFT-VITS2）
を**ネイティブ再実装**したものです。Voice model は上流の ONNX +
`config.json` から**オフライン**で変換され、ランタイムは
`onnxruntime` に依存しません:

```sh
vokra-cli convert --model piper-plus \
  --input voice.onnx --config voice.config.json --output voice.gguf
```

piper-plus の 8 言語 G2P（JA/EN/ZH/ES/FR/PT/SV/KO）は前処理レイヤとし
て流用します。Rust 移植は roadmap にありますが、Vokra ネイティブ TTS
経路の利用には必須ではありません。

## 6. コンプライアンス上の変化

パイプラインで **CC-BY-NC / CC-BY-NC-SA weight**（F5-TTS、Fish-Speech、
EnCodec）を直接使っていた場合、Vokra はデフォルトで拒否します。
`ComplianceLevel` に明示的な `research_flag: true`（または CLI の等価
スイッチ）で opt-in してください。ランタイムの商用利用者を保護する設
計判断です — 詳細は [`docs/legal-compliance.md`](legal-compliance.md)。

## 7. Vokra が（まだ）できないこと

- **Speaker diarization**（`pyannote` 相当）: 予定。v0.5 では非対応。
- **Bark / StyleTTS 2**: ライセンス audit 後、v2.0+ で対応予定。
- **Voice cloning（RVC v2 / GPT-SoVITS）**: 法務上の理由（ELVIS Act /
  NO FAKES Act）で別リポジトリ `vokra-voiceclone-experimental` に意図的
  に分離しています。
- **ランタイムでの ONNX**: 対応しません（設計判断 — 詳細は
  [`docs/onnx-alternative-research.md`](onnx-alternative-research.md)）。

## 次のステップ

- [Getting Started](getting-started.ja.md) — 5 分クイックスタート
- [Tutorials](tutorials/) — Unity / iOS / Python 統合
- [License audit](license-audit.md) — 全 weight ライセンス一覧
