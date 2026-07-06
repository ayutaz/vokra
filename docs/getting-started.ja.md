# Getting Started（日本語）

[English](getting-started.md) | **日本語**

**Vokra** を初めて触る人向けの 5 分クイックスタートです。ここでは VAD →
ASR → TTS の 3 デモを CPU で動かします。GPU（Metal / CUDA）や配布形態
（iOS / Unity / Python）は末尾の「次のステップ」を参照してください。

## 前提条件

- **Rust ツールチェイン**: 1.75 以上（`rustup default stable`）
- **git**: リポジトリの取得に使用
- **Python 3.10+**: モデル変換の PyTorch 依存を用意するだけで、Vokra ラン
  タイム自体は Python に依存しません（`FR-LD-05`）
- **ディスク**: モデル変換で 2〜4 GB（Whisper base + piper-plus voice）

Vokra ランタイムは **zero external dependency**（root `Cargo.lock` は
`vokra-*` crate のみ）なので、Rust ツールチェイン以外に追加の system
package は不要です。

## 1 分: ビルド

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
cargo build --release
```

これで CLI（`target/release/vokra-cli`）と C ABI 用 `libvokra` が生成され
ます。ビルド時間の目安: MacBook Air M2 で約 2 分（初回）。

## 2 分: モデルを GGUF に変換

Vokra ランタイムは **GGUF のみ**をロードします（ONNX グラフはランタイム
に一切載せません — オフライン変換ツール経由でのみ扱います）。以下はよく
使う 3 例です。

### Silero VAD v5

```sh
# Upstream ONNX → GGUF
wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
./target/release/vokra-cli convert \
  --model silero-vad \
  --input silero_vad.onnx \
  --output silero_vad.gguf
```

### Whisper base（ASR）

```sh
# Hugging Face safetensors → GGUF（サイズは checkpoint shape で自動検出）
pip install transformers safetensors
python3 -c "
from transformers import WhisperForConditionalGeneration
m = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')
m.save_pretrained('whisper-base', safe_serialization=True)
"
./target/release/vokra-cli convert \
  --model whisper \
  --input whisper-base/model.safetensors \
  --output whisper-base.gguf
```

**K-quant で軽量化**する場合:

```sh
./target/release/vokra-cli convert \
  --model whisper \
  --input whisper-base/model.safetensors \
  --output whisper-base.q4_k.gguf \
  --quantize q4_k
```

### piper-plus（TTS）

```sh
# piper-plus voice ONNX + config.json → GGUF
# ダウンロード例（en_US-lessac-medium）
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
  -O en_US-lessac-medium.config.json
./target/release/vokra-cli convert \
  --model piper-plus \
  --input en_US-lessac-medium.onnx \
  --config en_US-lessac-medium.config.json \
  --output en_US-lessac-medium.gguf
```

## 3 分: 実行

`vokra-cli run` は GGUF の `vokra.model.arch` メタデータからタスクを自動
判別します（Whisper→ASR / Silero VAD→VAD / piper-plus→TTS）。

### VAD

```sh
./target/release/vokra-cli run \
  --model silero_vad.gguf \
  --input speech.wav
# 出力: フレームごとの speech probability
```

### ASR

```sh
./target/release/vokra-cli run \
  --model whisper-base.gguf \
  --input speech.wav
# 出力: 転写テキスト
```

### TTS

```sh
./target/release/vokra-cli run \
  --model en_US-lessac-medium.gguf \
  --text "Hello from Vokra." \
  --output hello.wav
# 出力: hello.wav（22050 Hz mono PCM）
```

## 4 分: パフォーマンス計測

```sh
# CPU RTF
./target/release/vokra-cli bench --model whisper-base.gguf --input speech.wav

# GPU（Metal on macOS / CUDA on Linux 実機）
cargo build --release -p vokra-models --features metal   # macOS
cargo build --release -p vokra-models --features cuda    # Linux with system CUDA
./target/release/vokra-cli bench --model whisper-large-v3.gguf \
  --input speech30s.wav --backend cuda
```

RTF < 1.0 でリアルタイム、CPU で Whisper base は RTF < 0.3、CUDA で
Whisper large-v3 は RTF < 0.15（RTX 4090 実測 0.081〜0.115）が目安です。

## 5 分: C ABI からの呼び出し

`include/vokra.h` を include して `libvokra` にリンクするだけです（詳細は
[README.md](../README.md#using-the-c-abi) の例を参照）。

```c
#include "vokra.h"

vokra_session_t *s = NULL;
vokra_session_create_from_file("whisper-base.gguf", &s);

char *text = NULL;
vokra_asr_transcribe(s, pcm, num_samples, 16000, &text);
printf("%s\n", text);
vokra_string_free(text);
vokra_session_destroy(s);
```

## 次のステップ

- **プラットフォーム別チュートリアル**: [`docs/tutorials/`](tutorials/)
  - [Unity + IL2CPP](tutorials/unity.ja.md)
  - [iOS Swift Package](tutorials/ios.ja.md)
  - [Python bindings](tutorials/python.ja.md)
- **他ランタイムからの移行**: [Migration Guide](migration-guide.ja.md)
  （ONNX Runtime / whisper.cpp / sherpa-onnx から）
- **サーバ運用**: [`integrations/vokra-server`](../integrations/vokra-server)
  で OpenAI Whisper / vLLM / piper-plus HTTP / Wyoming Protocol の 4 互換
  API を単一バイナリで公開
- **License / Compliance**: [`docs/license-audit.md`](license-audit.md)、
  [`docs/legal-compliance.md`](legal-compliance.md)

## トラブルシューティング

- **`error: model file has no vokra.model.arch metadata`**: GGUF が Vokra
  以外のツール（`llama.cpp` など）で作られています。**Vokra runtime は
  Vokra converter が書いた GGUF のみを受け付けます**。上記の
  `vokra-cli convert` で再生成してください。
- **`error: backend does not implement op X`**: GPU バックエンドは silent
  CPU fallback を行いません（FR-EX-08）。未対応 op に当たった場合は
  `--backend cpu` に戻すか、issue を上げてください。
- **`error: research flag required for CC-BY-NC weight`**: F5-TTS /
  Fish-Speech / EnCodec など非商用 weight は compliance ゲートで拒否され
  ます。研究用途で使う場合は明示的な opt-in が必要です。
