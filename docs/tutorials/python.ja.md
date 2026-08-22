# Python バインディングチュートリアル

[English](python.md) | **日本語**

> **実装状態（2026-08-22 照合）: source 実装済み・未公開。** package root は
> `Session`、`Stream`、`Event`、typed error を export し、生成 table は現行
> 現行 C ABI の全 function を覆います。PyPI 公開は未検証・未承認です。正確な
> gate は [binding README](../../bindings/python/README.md) を参照してください。

## 現在存在するもの

- package metadata: `0.1.0.dev0`、Python 3.9〜3.12。
- runtime dependency は空。NumPy は任意の interop のみ。
- public source API: `Session`、`Stream`、`Event`、9 個の error subclass。
  音声 file の decode は caller 側の責務です。
- 生成済み FFI table: 現行 C header の全 function、4 enums、2 concrete
  structs、8 opaque handles。
- CI 契約: required `license` job がgenerator driftを検査し、各wheel smokeが
  public names、table件数、native symbolsのloadを検証します。

release workflowがbuildするwheelはLinux x86_64（`manylinux_2_28`）、macOS
arm64、macOS x86_64、Windows x86_64の4個です。macOS universal2はclaimせず、
source loaderが対応するLinux aarch64も現時点ではrelease wheel対象外です。
targetの宣言は、互換wheelが公開済みである証拠ではありません。

## 開発環境

source checkout では必ず uv を使います:

```sh
uv run --no-project --python 3.12 --with pytest \
  python -m pytest bindings/python/tests
```

実際の `ctypes` load を試す前に platform 用 C library を build/stage します:

```sh
cargo build --release -p vokra-capi
cp target/release/libvokra.dylib bindings/python/src/vokra/_lib/  # macOS
# Linux は libvokra.so、Windows は vokra.dll をコピーする。
```

## source public API

source surface は `Session`、`Stream`、`Event`、`VokraError` subclassです。
WAV読込はpackage APIに含めず、例では標準libraryだけのlocal helperを使います。
上記のmatching native libraryをbuild/stageした後に実行できます:

```python
import struct
import wave

from vokra import Session


def read_pcm16_wav_mono(path: str) -> tuple[list[float], int]:
    with wave.open(path, "rb") as source:
        wav_format = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getcomptype(),
        )
        if wav_format != (1, 2, "NONE"):
            raise ValueError("expected an uncompressed mono 16-bit PCM WAV")
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    pcm = [sample / 32768.0 for (sample,) in struct.iter_unpack("<h", frames)]
    return pcm, sample_rate


pcm, sample_rate = read_pcm16_wav_mono("speech.wav")
with Session.open("whisper-base.gguf") as session:
    text = session.transcribe(pcm, sample_rate)
```

source側の先頭3条件は実装済みです。release昇格には残り2条件の外部証跡が必要です:

1. generator が `include/vokra.h` の全型を処理し、現行全関数を生成する。
2. generated drift check が通る。
3. `src/vokra/__init__.py` が文書化した名前を export し、test する。
4. final-headの対応native library入りwheelがload、ASR/TTS、streaming、
   error、各platform smoke testを通る。
5. 正確なrelease versionと配布先が明示承認され、upload後に検証される。

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
