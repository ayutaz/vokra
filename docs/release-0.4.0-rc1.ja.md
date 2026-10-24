# v0.4.0-rc.1 preview guide

[English](release-0.4.0-rc1.md) | **日本語**

公開済みCPU-onlyの`v0.4.0-rc.1` previewを対象としたガイドです。[GitHubの正確な
releaseページ](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)でpre-releaseと検証済みassetを確認してから
downloadしてください。他のtagのassetで代用しないでください。

## 公開証跡

releaseはGitHub release `416018534`として`2026-10-17T19:37:51Z`に、exact source/tag
`d84d311e98d0abb2987b5c4e96157ae8f2068e1a`から公開されています（`draft=false`、
`prerelease=true`）。Latestは`v0.3.0`のままで、assetは22件です。最終hosted run
[`39062234813`](https://github.com/ayutaz/vokra/actions/runs/39062234813)は10 jobすべて成功し、
3 OS producer、転送後のnative CLI help/C ABI exact version確認、Linux assembled CLIのWhisper/Silero
機能smoke、公開byte round-tripを完了しました。回収receipt SHA-256は
`d4b4656fbfa67c8f10b7dd5f6642fc896780ab11da4bc4db80d7922bb2348697`です。distribution identityは
Linux CLIを13,344,272 bytes、SHA-256 `2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4`、
manifest SHA-256を`32b03f537f48ff6f2bb7b12fda1a9426090debbd59da361c0ffa59e8f7a60e92`として記録します。

## Scope

CPU-onlyのdesktop CLI・C ABIを配布するpreviewです。optional GPU backendの有効化、
モデルweightの更新、外部package registryへの公開、API/ABI安定の保証は含みません。
対応はモデル別です。binaryのbuildや`--help`成功は、そのOSでのモデル推論を証明しません。

Whisper base ASRとSilero VAD v5の音声・無音判定は、公開済みassembled Linux CLIを使い、
source/tag `d84d311e98d0abb2987b5c4e96157ae8f2068e1a`のrun
[`39062234813`](https://github.com/ayutaz/vokra/actions/runs/39062234813)で機能smokeに成功しました。
これはCPUの機能smokeだけで、独立数値parity、品質・速度benchmark、Apple/Metal検証ではありません。
[原本結果と制限](handoff/release-0.4.0-preview-cpu-results-2026-10-17.md)と
[候補版受入記録](handoff/release-0.4.0-rc1-packaging-2026-10-17.md)を参照してください。
日付付きrelease証跡では、macOS Apple SiliconとWindows x86_64の配布asset、native CLIの`--help`、
C ABIのexact version確認も成功しています。これはLinuxのmodel smokeとは別の確認です。Linuxの証跡から、
macOS/Windowsでのmodel推論、Apple Metal、数値parity、性能を判断することはできません。

> **2026-10-23 公開済みCLIとの組合せ確認。** 旧pinの公開CPU smokeとartifact identityは書き換えていない
> 過去の証跡です。修正済みファイルの出所とMITライセンスの証拠は別途検証済みで、公開済みRC1のLinux
> x86_64 glibc CLIも、修正済みWhisperとSileroの固定pinを使った最終CPU確認に成功しました。これは機能
> smokeだけで、数値parity、性能benchmark、Apple Metal、macOS/Windowsでのmodel推論を示すものではありません。
> [最終組合せの証跡](handoff/evidence/published-rc1-cli-pairing-final-2026-10-23.json)を参照してください。

## Whisper provenance修正記録（released CLIとの組合せ確認済み）

限定したWhisper provenance修正は、Hugging Faceのimmutable revision
[`bf440f6064ffc882d53d3694729dc4616dc2cfa7`](https://huggingface.co/vokra/whisper-base/tree/bf440f6064ffc882d53d3694729dc4616dc2cfa7)
として別途公開されています。修正済み`whisper-base.gguf`は290,859,488 bytes、SHA-256は
`5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014`です。
固定download URLは
`https://huggingface.co/vokra/whisper-base/resolve/bf440f6064ffc882d53d3694729dc4616dc2cfa7/whisper-base.gguf`です。
[修正記録](handoff/whisper-base-provenance-repair-2026-10-23.md)と
[成功receipt](handoff/evidence/whisper-base-provenance-repair-2026-10-23.json)に、固定したupstream/license
証跡とtensorを変更しないmetadata修正を記録しています。ただし、RC1 releaseのassetや過去のreceiptは変更していません。

上記の過去のhosted Linux機能smokeは、旧Whisper pin
（`dddd93ff6301a82aa3750d398cb2e0a599cd9e83`、SHA-256
`7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27`）で実行されたものです。
下記の最終receiptは修正済みpinを使ったもので、この過去のartifactやreceiptは書き換えていません。

## 修正済みmodelの組合せ確認

最終Linux CPU receiptでは、公開済みRC1 CLIのSHA-256
`2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4`、上記の修正済みWhisper
revision/hash、Silero revision `3670dfed054980307eeb1711c6329c29b80bdf14`とSHA-256
`23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6`を使いました。`--backend cpu`の
3つの確認はすべてexit code 0でした。最終runは
[`39953770618`](https://github.com/ayutaz/vokra/actions/runs/39953770618)、最終receiptは
[こちら](handoff/evidence/published-rc1-cli-pairing-final-2026-10-23.json)です。ASR stdoutは次のとおりです。

```text
asr:  And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

VADのspeech fixtureは343 frames中214 framesがspeech、meanは`0.5889`でした。silence fixtureは343 frames中
speech 0、meanは`0.0025`でした。

## Choose an asset

| 環境 | CLI asset |
| --- | --- |
| Linux x86_64、glibc | `vokra-cli-0.4.0-rc.1-x86_64-linux` |
| Linux x86_64、static musl | `vokra-cli-0.4.0-rc.1-x86_64-linux-musl` |
| macOS Apple Silicon | `vokra-cli-0.4.0-rc.1-aarch64-macos` |
| Windows x86_64 | `vokra-cli-0.4.0-rc.1-x86_64-windows.zip` |

Intel macOSなど他architectureはこのdesktop previewの配布対象ではありません。
各assetには`.sha256` sidecarがあります。配布物全体には`vokra-desktop-manifest.json`、
CLI/C ABI別のSPDX SBOM、native library、`vokra.h`も含めます。異なるtag/runのassetを
組み合わせないでください。

## Linux installation

モデル実行が許可された環境の空directoryで使用します。

```sh
mkdir vokra-rc1 && \
cd vokra-rc1 && \
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://github.com/ayutaz/vokra/releases/download/v0.4.0-rc.1/vokra-cli-0.4.0-rc.1-x86_64-linux \
  -o vokra-cli-0.4.0-rc.1-x86_64-linux && \
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://github.com/ayutaz/vokra/releases/download/v0.4.0-rc.1/vokra-cli-0.4.0-rc.1-x86_64-linux.sha256 \
  -o vokra-cli-0.4.0-rc.1-x86_64-linux.sha256 && \
sha256sum -c vokra-cli-0.4.0-rc.1-x86_64-linux.sha256 && \
chmod +x vokra-cli-0.4.0-rc.1-x86_64-linux && \
./vokra-cli-0.4.0-rc.1-x86_64-linux --help
```

static musl版は`-linux-musl` binaryと対応するsidecarを使用してください。

## macOS and Windows installation

Apple Siliconでは`-aarch64-macos` binaryとsidecarを同じ空directoryへ取得します。
`shasum -a 256 -c vokra-cli-0.4.0-rc.1-aarch64-macos.sha256`で検証し、`chmod +x`で
実行可能にしてから`--help`を実行します。Gatekeeperの無効化やquarantineの自動解除は
行わないでください。このpreviewはnotarizationを保証しません。

Windows x86_64では、展開前にPowerShellの`Get-FileHash -Algorithm SHA256`とsidecarで
ZIPのSHA-256を照合してください。空directoryに展開し、そのdirectoryから
`./vokra-cli.exe --help`を実行します。ZIP内には`vokra-cli.exe`一件だけを含めます。

## Reproduce the bounded Linux ASR example

このpreviewでは、修正済みWhisper revisionと固定した公開identityの音声fixtureを取得します。
モデルはmodel cardのlicenseで別途配布されるもので、runtime releaseによりlicenseや再配布条件が
変わるわけではありません。公開済みRC1 CLIとの組合せは、上記の最終Linux CPU receiptで確認済みです。

```sh
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://huggingface.co/vokra/whisper-base/resolve/bf440f6064ffc882d53d3694729dc4616dc2cfa7/whisper-base.gguf \
  -o whisper-base.gguf && \
test "$(wc -c < whisper-base.gguf)" -eq 290859488 && \
printf '%s\n' '5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014  whisper-base.gguf' \
  | sha256sum -c - && \
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/ayutaz/vokra/v0.4.0-rc.1/tests/fixtures/audio/jfk-30s.wav \
  -o jfk-30s.wav && \
test "$(wc -c < jfk-30s.wav)" -eq 352078 && \
printf '%s\n' '58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f  jfk-30s.wav' \
  | sha256sum -c - && \
./vokra-cli-0.4.0-rc.1-x86_64-linux run \
  --backend cpu --model whisper-base.gguf --input jfk-30s.wav
```

fixtureは過去のfile名と異なり11秒です。上記の最終Linux CPU receiptは、このfixtureに対して次のASR行を
そのまま出力しました。

```text
asr:  And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

この機能smokeで修正済みmodelと公開済みCLIの組合せを確認しています。正確な処理速度や認識精度をreleaseの
保証とは扱いません。

## Linux Silero VAD例

上記で検証した`jfk-30s.wav`を使い、review済みのSilero VAD v5 artifactを取得して、sizeとSHA-256を
確認してからCPU backendを実行します。上記の最終receiptにも同じ固定pinとLinux CPU経路を記録しています。

```sh
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://huggingface.co/vokra/silero-vad-v5/resolve/3670dfed054980307eeb1711c6329c29b80bdf14/silero-vad-v5.gguf \
  -o silero-vad-v5.gguf && \
test "$(wc -c < silero-vad-v5.gguf)" -eq 2183744 && \
printf '%s\n' '23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6  silero-vad-v5.gguf' \
  | sha256sum -c - && \
test -f jfk-30s.wav && \
printf '%s\n' '58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f  jfk-30s.wav' \
  | sha256sum -c - && \
./vokra-cli-0.4.0-rc.1-x86_64-linux run \
  --backend cpu --model silero-vad-v5.gguf --input jfk-30s.wav
```

## Known limitations and reporting

TTS、全catalog推論、独立実weight数値parity、モデル別Apple CPU/Metal/no-fallbackの
完了はpreviewの合格主張に含みません。modelのhosted機能証跡はLinux x86_64 glibcだけを対象とします。
別の日付付き配布asset/help/C ABI確認はmacOS Apple SiliconとWindows x86_64で成功していますが、
それらのOSでのmodel推論、Apple Metal、数値parity、性能を証明するものではありません。報告時は
正確なrelease/tag、OS/architecture、model revision/hash、command、error、期待した動作を添え、
keyやprivate音声は含めないでください。未対応backendは暗黙CPU fallbackではなく明示errorを返す必要があります。
