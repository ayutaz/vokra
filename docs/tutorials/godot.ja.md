# Godot（GDExtension）チュートリアル

[English](godot.md) | **日本語**

Vokra は `integrations/vokra-godot` <!-- anchor: integrations/vokra-godot --> に
Godot 4.x **GDExtension** バインディングを同梱する。Vokra C ABI 上の隔離
workspace（生 FFI、binding crate 不使用）であり、root `Cargo.lock` の
zero-dependency 不変条件（`NFR-DS-02`）には非干渉である。2 つのクラス
`VokraSession` / `VokraStream` と 2 つの demo プロジェクトを提供する。

## 1. GDExtension のビルド

`scripts/build-godot-gdextension.sh` <!-- anchor: scripts/build-godot-gdextension.sh -->
は `TARGET_TRIPLE` で選んだ 5 target のいずれか（macOS Intel / Apple Silicon、
Linux x64、Windows MSVC、Android arm64）向けに native ライブラリをクロス
ビルドする。未知の triple は推測せず非ゼロ終了する（`FR-EX-08`）:

```sh
TARGET_TRIPLE=aarch64-apple-darwin scripts/build-godot-gdextension.sh
```

## 2. Godot プロジェクトへのインストール

`addons/vokra/` ツリーをプロジェクトにコピーする（Godot AssetLib layout:
`.gdextension` descriptor + プラットフォーム別 `bin/` ライブラリ）。Godot は
プロジェクトを開くと extension をロードし Vokra クラスを登録する。

## 3. `VokraSession` / `VokraStream` API

`VokraSession` は GGUF をロードし task を実行する。trampoline が Godot Variant
を unpack して実 runtime を呼ぶ:

```
var session := VokraSession.new()
session.load_model("res://models/whisper-base.gguf")

# ASR: PackedFloat32Array（16 kHz mono）+ sample rate -> String
var text: String = session.transcribe(pcm, 16000)

# TTS: String -> Dictionary { "pcm": PackedFloat32Array, "sample_rate": int }
var out: Dictionary = session.synthesize("Hello from Vokra.")
```

`VokraStream` は streaming プリミティブ — `push_pcm(pcm)` / `poll(n)` /
`interrupt()`（barge-in）を提供する。正直なギャップに注意: GDScript から
VAD stream を開く（Object を返す `session.vad_open_stream`）のは**まだ配線
されておらず**、偽の値ではなく明示エラーを返す — その return path には追加の
Variant plumbing が要る。

## 4. demo プロジェクト

すぐ開ける 2 プロジェクトが `demos/` 下にある:
`integrations/vokra-godot/demos/asr_demo` <!-- anchor: integrations/vokra-godot/demos/asr_demo -->
は 16 kHz mono PCM16 をロードして `transcribe` を呼ぶ。`demos/tts_demo` は
`synthesize` を呼び `AudioStreamGenerator` へストリームする。

## 5. 明示エラーと NVIDIA 非同梱

全 trampoline はバックエンドエラーを明示的な Godot `CallError` に流す
（`FR-EX-08`）。`vokra_last_error()` 文字列は GDScript introspection のため同一
スレッドで利用可能で、Rust panic は Godot に届く前に各境界で catch される
（`NFR-RL-07`）。パッケージ化された addon は
`scripts/compliance/check-godot-package-no-nvidia.sh`
<!-- anchor: scripts/compliance/check-godot-package-no-nvidia.sh --> により
**NVIDIA runtime を bundle していない**ことをスキャンされる（CUDA ビルドは
実行時に system CUDA を `dlopen` する。`libcudart` / `libcudnn` / `libcublas`
/ `libnvrtc` を同梱しない）。

## 6. 正直な状態（owner editor 検証）

trampoline の runtime dispatch は code-complete: `transcribe` / `synthesize` /
`push_pcm` / `poll` / `interrupt` が Variant を unpack/pack して runtime を呼ぶ
（`integrations/vokra-godot/src/trampoline.rs`
<!-- anchor: integrations/vokra-godot/src/trampoline.rs -->）。ここで**未完**
なのは、実 Godot 4.x editor 内で動かして demo を end-to-end で駆動すること —
その runtime 検証は owner タスク（M3-11 T19）である。本ページは API 契約で
あって、editor で実際に動作確認済みという証明ではない。

## 7. トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| extension がロードされない | `.gdextension` の `bin/` パスがプラットフォーム/arch と一致する必要がある。正しい `TARGET_TRIPLE` で再ビルドする。 |
| ビルドスクリプトが `unknown target triple` | サポートされる 5 triple のいずれかを渡す（`FR-EX-08`、無言の推測なし）。 |
| `session.transcribe` がエラーを返す | `vokra_last_error()` を読む。backend/op エラーは明示的 `CallError` として現れ、偽の結果にはならない。 |
| `vad_open_stream` がエラーを返す | 想定どおり — Object return path は未配線（§3）。 |

## Next steps

- [バックエンドの追加](../backend-guide.ja.md)
- [デスクトップ CLI](cli.ja.md) — ロードする GGUF を作る `convert` 手順
- [Unity + IL2CPP](unity.ja.md) — もう一つのゲームエンジンバインディング

## Keeping this page current

**最終確認日: 2026-07-21 — `integrations/vokra-godot/src/`（trampoline は
dispatch する。2026-07-10 付 README はそれより前で stale）に対して確認。**

- **更新責任**: GDExtension API・ビルド target・compliance scanner を変えた PR
  が、同一 PR で本ページと英語版を更新する。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）。実 editor の
  runtime 検証は owner（M3-11 T19）のもので、上に正直に記録している。
- **dispatch 状態の再検証**（README を信用しない）:

```sh
sed -n '1,45p' integrations/vokra-godot/src/trampoline.rs
```
