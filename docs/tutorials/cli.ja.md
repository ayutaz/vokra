# デスクトップ CLI チュートリアル

[English](cli.md) | **日本語**

`vokra-cli` は umbrella コマンドラインツール（`FR-TL-01`, `FR-TL-02`）。
同じ native runtime 上の 3 つの subcommand — `run` / `convert` / `bench` —
を、手書きの引数パーサ・外部依存ゼロ（`NFR-DS-02`）で提供する。本ページは
deep dive。5 分コースは [getting-started.md](../getting-started.ja.md) を参照。

## 1. ビルド

```sh
cargo build --release
```

`target/release/vokra-cli` が生成される。各 subcommand のフルオプションは
`vokra-cli <subcommand> --help` で確認する。

## 2. `run` — task 自動選択の推論

`run` は GGUF をロードし、モデルの `vokra.model.arch` metadata から task を
選ぶ（Whisper → ASR、Silero VAD → VAD、piper-plus → TTS）。task を自分で
指定する必要はない:

```sh
# ASR — 音声入力、テキスト出力
./target/release/vokra-cli run --model whisper-base.gguf --input speech.wav

# TTS — テキスト入力、WAV 出力
./target/release/vokra-cli run --model voice.gguf \
  --text "Hello from Vokra." --output hello.wav
```

ASR には decode 制御がある: `--beam-size` / `--word-timestamps` /
`--length-penalty` / `--no-repeat-ngram` / `--language`。TTS には `--voice` /
`--style` / `--length-scale`:

```sh
./target/release/vokra-cli run --model whisper-base.gguf --input speech.wav \
  --beam-size 5 --word-timestamps
```

## 3. `convert` — checkpoint → GGUF（オフライン）

runtime は **GGUF のみ**をロードする。ONNX / safetensors はここでオフライン
処理する。`--model` はソース種別を指定し、`--quantize` は出力時に K-quant
する:

```sh
./target/release/vokra-cli convert --model whisper \
  --input whisper-base/model.safetensors --output whisper-base.gguf

# K-quant で小型化
./target/release/vokra-cli convert --model whisper \
  --input whisper-base/model.safetensors --output whisper-base.q4_k.gguf \
  --quantize q4_k
```

piper-plus voice には `config.json` も要る。モデルによっては `--tokenizer`
や `--adapter-config` の side-car を取る:

```sh
./target/release/vokra-cli convert --model piper-plus \
  --input voice.onnx --config voice.config.json --output voice.gguf
```

## 4. `bench` — RTF / TTFA / jitter と regression gate

`bench` は real-time factor・time-to-first-audio・jitter・p50/p95/p99
レイテンシを報告する。`--baseline` を付けると **regression gate** になり、
記録した baseline に対し 5% を超える相対劣化で非ゼロ終了する（`NFR-PF-13`）。

```sh
# 計測
./target/release/vokra-cli bench --model whisper-base.gguf --input speech.wav \
  --iters 20 --warmup 3 --format json

# 記録済み baseline に対して gate
./target/release/vokra-cli bench --model whisper-base.gguf --input speech.wav \
  --baseline baseline.json
```

## 5. バックエンド選択は明示的（`FR-EX-08`）

`--backend` で計算バックエンドを選ぶ。Vokra は silent fallback をしない:
GPU バックエンドが cover しない op、不在の device は明示エラーであり、CPU への
無言の切り替えはしない。

```sh
cargo build --release -p vokra-models --features metal   # macOS
./target/release/vokra-cli bench --model whisper-large-v3.gguf \
  --input speech30s.wav --backend metal
```

CPU を*意図的に*選ぶには `--backend cpu` を使う — それはあなたが下す決定で
あり、Vokra が裏で下す決定ではない。

## 6. トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| `error: model file has no vokra.model.arch metadata` | GGUF が非 Vokra ツール（例 `llama.cpp`）製。`vokra-cli convert` で再生成する。 |
| `error: backend does not implement op X` | GPU バックエンドがその op を cover していない（`FR-EX-08`）。`--backend cpu` で再試行するか model/op を報告する。 |
| `bench` が regression メッセージで非ゼロ終了 | `--baseline` gate が発火（5% 超の劣化）。変更を調査するか、意図的に baseline を更新する。 |
| `error: research flag required for CC-BY-NC weight` | 非商用 weight が compliance gate で拒否された。research 用途は明示 opt-in が必要。 |

## Next steps

- [Server（4 互換 API）](server.md) — CLI ではなく HTTP エンドポイントが欲しい
  場合の別バイナリ `vokra-server`
- [バックエンドの追加](../backend-guide.ja.md)
- [Migration Guide](../migration-guide.ja.md)（ONNX Runtime / whisper.cpp /
  sherpa-onnx から）

## Keeping this page current

**最終確認日: 2026-07-21 — `crates/vokra-cli/src/` の `run` / `convert` /
`bench` 引数パーサに対して確認。**

- **更新責任**: CLI フラグを追加・改名した PR が、同一 PR で本ページと英語版を
  更新する。本ページの全 `vokra-cli` 呼び出しは `doc-examples` CI job が実
  パーサに対して照合するため、古いフラグは CI を落とす。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）。
- **フラグ surface の再取得**:

```sh
grep -oE '"--[a-z0-9-]+"' crates/vokra-cli/src/run.rs crates/vokra-cli/src/convert.rs crates/vokra-cli/src/bench.rs
```
