# Python バインディングチュートリアル

[English](python.md) | **日本語**

> **実装状態（2026-08-18 照合）: pre-alpha。** リポジトリには内部
> `ctypes` モジュールがありますが、package root が現在公開するのは
> `__version__` だけです。現時点で `from vokra import Session` が使える
> リリース済みパッケージとして案内してはいけません。正確な差分と完了条件は
> [binding README](../../bindings/python/README.md) を参照してください。

## 現在存在するもの

- package metadata: `0.1.0.dev0`、Python 3.9〜3.12。
- runtime dependency は空。NumPy は任意の interop のみ。
- 内部モジュール: native loader、handle/session/stream wrapper、WAV helper、
  9 個の error subclass。
- 生成済み FFI table: 旧来の ASR/TTS/streaming 14-function subset。
- 現行 C header: 41 functions。generator parser が認識するのは 39 で、さらに
  新しい `vokra_aec_config_t` の型 mapping がないため再生成時に停止する。

packaging/CI 上の wheel target は Linux x86_64/aarch64、macOS universal2、
Windows x86_64 です。target の宣言は、互換 wheel が公開済みである証拠では
ありません。

## 開発環境

source checkout では必ず uv を使います:

```sh
uv sync --project bindings/python --extra dev
uv run --project bindings/python --extra dev pytest bindings/python/tests
```

実際の `ctypes` load を試す前に platform 用 C library を build/stage します:

```sh
cargo build --release -p vokra-capi
cp target/release/libvokra.dylib bindings/python/src/vokra/_lib/  # macOS
# Linux は libvokra.so、Windows は vokra.dll をコピーする。
```

## 予定している public API

ABI generator と package export を現行化した後の予定 surface は
`Session`、`Stream`、WAV helper、`VokraError` subclass です。次は形を示す
例であり、現在の checkout で実行可能な quick start ではありません:

```python
from vokra import Session, read_wav_mono_f32

pcm, sample_rate = read_wav_mono_f32("speech.wav")
with Session.open("whisper-base.gguf") as session:
    text = session.transcribe(pcm, sample_rate)
```

この例を quick start に戻す前に、次をすべて満たす必要があります:

1. generator が `include/vokra.h` の全型を処理し、現行全関数を生成する。
2. generated drift check が通る。
3. `src/vokra/__init__.py` が文書化した名前を export し、test する。
4. 対応 native library 入り wheel が load、ASR/TTS、streaming、error、各
   platform smoke test を通る。
5. 正確な公開 version と配布先を確認する。

## Error / thread 契約

C enum は `VOKRA_OK` を含む 10 値で、9 個の error 値が
`src/vokra/errors.py` の 9 subclass に対応します。`vokra_last_error()` は
thread-local なので、失敗と同じ call frame で取得します。未対応処理を CPU
へ silent fallback してはいけません。

`Stream` は caller 側で直列化し、lock なしで thread 間共有しません。
parent `Session` より先に stream を close する nested context-manager 形が
意図した ownership です。

## 次のステップ

- 現在の binding status と source wheel 手順:
  [bindings/python/README.md](../../bindings/python/README.md)
- Python package と独立して使える native CLI:
  [Getting Started](../getting-started.ja.md)
- Python client から使える HTTP compatibility path:
  [`integrations/vokra-server`](../../integrations/vokra-server)
