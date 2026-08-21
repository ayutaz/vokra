# デスクトップ CLI チュートリアル

[English](cli.md) | **日本語**

`vokra-cli` は umbrella コマンドラインツール（`FR-TL-01`, `FR-TL-02`）。
同じ native runtime 上の 4 つの subcommand — `run` / `convert` / `bench` /
`f0` — を、手書きの引数パーサ・外部依存ゼロ（`NFR-DS-02`）で提供する。本ページは
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

### CT-Punc のtoken/idペア入力

CT-Puncはtokenizeを推測しない。token文字列とモデルに渡す正確なvocabulary
IDを、1つのversioned UTF-8 TSVに対で記述する:

```
vokra-ct-punc-tsv-v1
101	we
202	build
303	世界
```

各recordは`<u32 id><TAB><escaped token>`。tokenにはliteral Unicode、または
`\\` / `\t` / `\n` / `\r` / `\u{HEX}` escapeを使用できる。空record、余分な
TSV列、不正escape、vocabulary範囲外IDはエラーになる。単一record streamに
することでtoken列とID列の長さ不一致を構造的に排除する。

```sh
./target/release/vokra-cli run --model ct-punc.gguf \
  --tokens tokens.tsv --output restored.txt
```

`--output`なしでは`ct_punc: <text>`として表示する。指定時のfile内容は診断
prefixや暗黙newlineを含まない、復元後の正確なUTF-8 textになる。

### Mimi encode/decodeとportable code container

Mimiは方向を明示する。Rustのnested arrayのdebug表示を交換形式にはしない:

```sh
./target/release/vokra-cli run --model mimi.gguf --codec-mode encode \
  --input speech-24k.wav --output speech.vmc
./target/release/vokra-cli run --model mimi.gguf --codec-mode decode \
  --input speech.vmc --output reconstructed.wav
```

`speech.vmc`は`VKRMCODE` version 1。固定little-endian headerにmono channel、
sample rate、milli-Hz単位のframe rate、frame数、元PCM sample数、codebook数・
size、feature幅、GGUF effective codebook tableのSHA-256を保持する。payloadは
time-major `[frame, codebook]` 順のunsigned 32-bit code。decodeは異なるmodel
hash/topology、余分・不足byte、範囲外code、`frames * model_hop`と一致しない
PCM長を拒否する。encodeも正のframe-hop整数倍だけを受け付け、暗黙のresample・
padding・trimは行わない。

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

## 5. `f0` — checkpoint 不要のピッチ抽出

`f0` は WAV に対して YIN または PyIN を実行する。`run --task` ではなく独立
subcommand なのは、`run` が `--model` GGUF を必須とする一方、この 2 つの
extractor は重みを一切持たないため — checkpoint も license class も
`docs/license-audit.md` §3.1 の行も無い。呼び出し側が渡せる `--model` が存在
しない。

```sh
# YIN（デフォルト）
./target/release/vokra-cli f0 --input speech.wav

# PyIN、話声域に限定
./target/release/vokra-cli f0 --input speech.wav --algo pyin \
  --fmin 65 --fmax 400
```

出力行は tab 区切りで、neural F0 モデルに対して `run` が出す形と同一。
extractor を切り替えてもパース側は変更不要:

```
time_sec<TAB>hz<TAB>voiced<TAB>confidence
```

無声フレームは `hz=0.000`, `voiced=false`。どちらの op もフレーム単位の
confidence を持たないため、この列は捏造したスコアではなく `voiced` と同じ
`1.0` / `0.0` を報告する。

サンプルレートは**固定ではない**: 両 op とも WAV が持つレートから lag 探索
範囲を導出するので、暗黙のリサンプルは発生しない。同じ family の neural
メンバ — RMVPE / FCPE / CREPE — は checkpoint を要するため `run` 側に残る。

## 6. バックエンド選択は明示的（`FR-EX-08`）

`--backend` で計算バックエンドを選ぶ。Vokra は silent fallback をしない:
GPU バックエンドが cover しない op、不在の device は明示エラーであり、CPU への
無言の切り替えはしない。

```sh
cargo build --release -p vokra-cli --features metal   # macOS
./target/release/vokra-cli bench --model whisper-large-v3.gguf \
  --input speech30s.wav --backend metal
```

CPU を*意図的に*選ぶには `--backend cpu` を使う — それはあなたが下す決定で
あり、Vokra が裏で下す決定ではない。

## 7. トラブルシューティング

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

**最終確認日: 2026-08-21 — `crates/vokra-cli/src/` の `run` / `convert` /
`bench` / `f0` 引数パーサに対して確認。**

- **更新責任**: CLI フラグを追加・改名した PR が、同一 PR で本ページと英語版を
  更新する。本ページの全 `vokra-cli` 呼び出しは `doc-examples` CI job が実
  パーサに対して照合するため、古いフラグは CI を落とす。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）。
- **フラグ surface の再取得**:

```sh
grep -oE '"--[a-z0-9-]+"' crates/vokra-cli/src/run.rs crates/vokra-cli/src/convert.rs crates/vokra-cli/src/bench.rs crates/vokra-cli/src/f0.rs
```
