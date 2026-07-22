# サーバチュートリアル

[English](server.md) | **日本語**

`vokra-server`（`integrations/vokra-server` <!-- anchor: integrations/vokra-server -->）
は HTTP サーバ: **単一バイナリから 4 つの互換 API** を提供し、Vokra が
first-class として扱うサーバプラットフォーム（`NFR-PT-01` — x86-64 / ARM64
サーバは任意扱いではない）を対象とする。隔離 workspace なので root
`Cargo.lock` の zero-dependency 不変条件（`NFR-DS-02`）には非干渉。これは
`vokra-cli` とは別バイナリである — そちらは [CLI チュートリアル](cli.ja.md)。

## 1. ビルドと起動（単一バイナリ、Docker 不要）

Vokra は Docker を要求しない（`FR-SV-01`）: サーバは model GGUF を渡して起動
する 1 つの static-friendly なバイナリである。

```sh
cargo build --release -p vokra-server
./target/release/vokra-server \
  --http-bind 127.0.0.1:8080 \
  --whisper-base whisper-base.gguf --whisper-base-tokenizer tok.gguf \
  --piper-plus voice.gguf --piper-g2p
```

`--piper-g2p` は実 8 言語 G2P による plain-text TTS を有効化する。無い場合、
plain-text の `/api/tts` は明示的に 400 を返し、raw phoneme-id ペイロードのみ
機能する（`FR-EX-08` — 無言の挙動変化なし）。

## 2. 4 つの互換 API

1 プロセスで 4 つすべてを提供する（`IF-05` が piper-plus / Home Assistant
インターフェースを担う）:

| API | エンドポイント | 要件 |
|---|---|---|
| OpenAI Whisper | `POST /v1/audio/transcriptions` | `FR-SV-02` |
| vLLM 互換 | `POST /v1/completions`, `POST /v1/chat/completions` | vLLM HTTP 互換 |
| piper-plus HTTP | `POST /api/tts` | `FR-SV-04` |
| Wyoming Protocol | `--wyoming-bind`（Home Assistant） | `FR-SV-05` |

```sh
# OpenAI 互換の転写（faster-whisper drop-in）
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=whisper-base

# piper-plus TTS
curl -s http://127.0.0.1:8080/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Vokra."}' --output hello.wav
```

## 3. モデルとバックエンドのフラグ

必要なモデルをフラグでロードする。whisper family は base / small / medium /
turbo / large-v3 を cover し、それぞれ `-tokenizer` の相棒を持つ。さらに
`--piper-plus` / `--kokoro` / `--voxtral` / `--silero-vad` のスロットがある:

```sh
./target/release/vokra-server --http-bind 0.0.0.0:8080 \
  --whisper-large-v3 whisper-large-v3.gguf \
  --whisper-large-v3-tokenizer tok.gguf \
  --backend cuda
```

`--backend` は計算バックエンドを明示選択する。不在の device や未 cover の op
は明示エラーであり、silent CPU fallback は決してしない（`FR-EX-08`）。

## 4. multi-session と並行性

サーバは並行セッションを扱う（`FR-SV-06`）。`--max-concurrent-sessions` で上限
を設け、バーストがメモリを枯渇させないようにする:

```sh
./target/release/vokra-server --http-bind 0.0.0.0:8080 \
  --whisper-base whisper-base.gguf --whisper-base-tokenizer tok.gguf \
  --max-concurrent-sessions 8
```

## 5. bind アドレスとセキュリティ姿勢

`--http-bind` / `--wyoming-bind` は明示的な `host:port` を取る。ローカル専用は
`127.0.0.1`、`0.0.0.0` は自前の認証付き reverse proxy の背後でのみ使う — サーバ
自身は認証を足さない。全設定 surface は起動時に検証され、モデルファイル不在は
初回リクエスト時の遅延失敗ではなく hard error となる。

## 6. トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| plain-text `/api/tts` が 400 を返す | `--piper-g2p` 付きで起動する。無いと phoneme-id ペイロードのみ受理（§1）。 |
| 起動時にモデルパスを名指すエラー | `--whisper-*` / `--piper-plus` / … のパスが存在しない。サーバは後で 500 にせず fail fast する。 |
| `--backend cuda` で起動時 `BackendUnavailable` | 使える CUDA driver/GPU が無い。`--backend cpu` へ*明示的に*落とす（`FR-EX-08`）。 |
| Home Assistant がサーバを認識しない | `--wyoming-bind host:port` を渡し、そのアドレスを HA に登録する（`FR-SV-05`）。 |

## Next steps

- [デスクトップ CLI](cli.ja.md) — サーバがロードする GGUF を作る `convert` 手順
- [Migration Guide](../migration-guide.ja.md) — faster-whisper / OpenAI API
  drop-in の詳細
- [バックエンドの追加](../backend-guide.ja.md)

## Keeping this page current

**最終確認日: 2026-07-21 — `integrations/vokra-server/src/config.rs`
<!-- anchor: integrations/vokra-server/src/config.rs --> のフラグ surface に
対して確認。**

- **更新責任**: サーバのフラグやエンドポイントを追加・改名した PR が、同一 PR
  で本ページと英語版を更新する。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）。
- **フラグ surface の再取得**:

```sh
grep -oE '"--[a-z][a-z0-9-]+"' integrations/vokra-server/src/config.rs | sort -u
```
