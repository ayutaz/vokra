# Python バインディングチュートリアル

[English](python.md) | **日本語**

このチュートリアルでは、**Vokra の Python バインディング**
（`bindings/python`）のインストールと使い方を説明します。バインディングは
C ABI 上の薄い `ctypes` ラッパーで、**Python 側のサードパーティ依存はゼロ**
です。

## 1. インストール

```sh
pip install "vokra==0.1.0"
```

**バージョンは必ず固定してください**。Vokra の C ABI は v1.0 まで凍結され
ておらず（IF-01）、pre-1.0 のアップグレードは breaking change を含む可能
性があります。

**配布**: wheel は `cibuildwheel` 経由で PyPI に公開されます。

| OS      | Arch                       | Wheel tag                |
| ------- | -------------------------- | ------------------------ |
| Linux   | x86_64                     | `manylinux_2_28_x86_64`  |
| Linux   | aarch64                    | `manylinux_2_28_aarch64` |
| macOS   | universal2（x86_64 + arm64）| `macosx_11_0_universal2` |
| Windows | x86_64                     | `win_amd64`              |

対応 Python: 3.9〜3.12。iOS / Android / WASM は Python バインディングの
スコープ外です — Swift / Kotlin バインディング（予定）または WASM ター
ゲット（v1.5+）を使ってください。

## 2. モデルを変換

Vokra は GGUF のみをロードします。上流の Whisper checkpoint を CLI で変
換します（詳細は [Getting Started](../getting-started.ja.md)）:

```sh
cargo run --release --bin vokra-cli -- convert \
  --model whisper \
  --input whisper-base/model.safetensors \
  --output whisper-base.gguf
```

## 3. ASR — Python から転写

```python
from vokra import Session

session = Session.open("whisper-base.gguf")
try:
    # pcm は mono float32、範囲 [-1, 1]。任意のシーケンスを受け付ける
    # （list / tuple / array.array('f') / numpy.ndarray.tolist() 等）
    with open("speech.wav", "rb") as f:
        # bindings/python/src/vokra/audio.py に同梱の WAV リーダー
        pcm, sample_rate = read_wav_mono_f32(f)
    text = session.transcribe(pcm, sample_rate)
    print(text)
finally:
    session.close()  # あるいは `with Session.open(...) as session:` を使う
```

## 4. TTS — テキストから合成

```python
from vokra import Session

with Session.open("en_US-lessac-medium.gguf") as session:
    pcm, sample_rate = session.synthesize("Hello from Vokra.")
    write_wav_mono_f32("hello.wav", pcm, sample_rate)
```

`pcm` は素の Python `list[float]` です。必要に応じて呼び出し側で
`numpy.ndarray` に変換してください。バインディング側は意図的に `numpy`
を import しません — FFI seam を pure `ctypes` に保つためです（
`bindings/python/README.md` の "Zero third-party dependency" 節参照）。

## 5. エラー契約（silent fallback なし）

すべての C ABI 呼び出しは同じ Python フレーム内で status を検査し、非 OK
時は同フレームで `vokra_last_error()` を読みます。10 個の status コード
は 1:1 で例外クラスに対応します:

```python
from vokra import (
    Session,
    VokraError,            # 基底 — すべて捕捉
    VokraInvalidArgument,  # 例: sample rate 不一致
    VokraModelError,       # GGUF が壊れている / 見つからない
    VokraUnsupportedBackend,  # backend を指定したがリンクされていない
    # ... 全リストは bindings/python/src/vokra/errors.py 参照
)

try:
    with Session.open("some-model.gguf") as s:
        s.transcribe(pcm, 16000)
except VokraUnsupportedBackend as e:
    # モデルが要求する GPU バックエンドがこのビルドに含まれていない。
    # FR-EX-08 により silent fallback は禁止されており、これは明示的なシグナル。
    print(f"backend not available: {e}")
except VokraError as e:
    print(f"vokra error: {e}")
```

## 6. スレッド安全性

- `Session` は C 側で `Send + Sync` で、Python スレッド間で安全に共有で
  きます（GIL がさらに Python レベルの属性アクセスを直列化）。
- `vokra_last_error()` は thread-local です — 失敗した呼び出しと別の
  スレッドから読んではいけません。
- ラッパの `Session.__del__` は close 忘れ時に native ハンドルを解放しま
  す（RAII）。`with` ブロックを推奨します。

## 7. wheel の動作確認

Python wheel はプリビルドの `libvokra` を `src/vokra/_lib/` に同梱してい
ます。ロードパスの確認:

```python
import vokra
print(vokra.__version__)      # 例: "0.1.0"
print(vokra.__abi_version__)  # native lib と一致していなければならない
```

import 時に ABI バージョン不一致を検出すると即座に `VokraError` を投げま
す — 意図的な設計です（互換性のない C ABI を持つ古い wheel は「silent
に動かす」のではなく「はっきり失敗させる」）。

## 8. デバッグ

- **`OSError: cannot find libvokra.*`**: そのプラットフォーム向けの wheel
  ではないか、`src/vokra/_lib/` が空です。OS/arch に合った wheel を再イ
  ンストールするか、ソースからビルドしてください
  （[`bindings/python/README.md`](../../bindings/python/README.md#build-from-source)）。
- **`VokraInvalidArgument: sample rate 16000 != model rate 22050`**: モ
  デルの frontend sample rate と PCM が一致していません。M0 はリサンプル
  しないので、`transcribe` を呼ぶ前に `soxr` / `librosa` 等で変換してく
  ださい。

## 次のステップ

- **サーバ**: [`integrations/vokra-server`](../../integrations/vokra-server)
  を立てれば、`openai-python` や `faster-whisper` クライアントをそのまま
  使えます（Vokra は HTTP shape のドロップイン）。
- **移行**: `faster-whisper` / `piper-py` から移行する場合は
  [Migration Guide](../migration-guide.ja.md) を参照。
- **ライセンス**: Apache-2.0。wheel が同梱する native コードも同ライセン
  ス。詳細は wheel 内の `LICENSE` / `NOTICE`。
