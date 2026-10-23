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

> **2026-10-23 ライセンス/provenance確認警告 — Whisper quickstartの推奨・共有は保留。**
> 下記の公開済みCPU smokeと固定artifact identityは、書き換えていない過去の証跡です。
> ただし、固定したHugging Face Whisper card/APIはweightを`apache-2.0`とし、upstreamの
> distribution sourceを具体的なupstream repository revisionに結び付けず一般的に記載しています。
> 一方、OpenAI upstreamの`LICENSE`とrepositoryのlicense auditはWhisperをMITとして記録しています。
> この不整合は未解決です。upstream source、revision、license、attribution、artifact provenanceを
> 照合して解消するまで、下記commandは正確な過去の再現手順としてのみ扱い、新たなlicense承認、
> 再配布の推奨、weightの再公開・共有許可とは扱わないでください。[2026-10-23 announcement-readiness確認記録](handoff/announcement-readiness-2026-10-23.md)を参照してください。

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
この手順はmaintainer Macでのモデル実行許可ではありません。

```sh
mkdir vokra-rc1
cd vokra-rc1
curl -fL --proto '=https' \
  https://github.com/ayutaz/vokra/releases/download/v0.4.0-rc.1/vokra-cli-0.4.0-rc.1-x86_64-linux \
  -o vokra-cli-0.4.0-rc.1-x86_64-linux
curl -fL --proto '=https' \
  https://github.com/ayutaz/vokra/releases/download/v0.4.0-rc.1/vokra-cli-0.4.0-rc.1-x86_64-linux.sha256 \
  -o vokra-cli-0.4.0-rc.1-x86_64-linux.sha256
sha256sum -c vokra-cli-0.4.0-rc.1-x86_64-linux.sha256
chmod +x vokra-cli-0.4.0-rc.1-x86_64-linux
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

公開済みreleaseについて、固定revisionのモデルと正確なrelease tagの音声fixtureを取得します。
モデルはmodel cardのlicenseで別途配布されるもので、runtime releaseによりlicenseや
再配布条件が変わるわけではありません。

```sh
curl -fL --proto '=https' \
  https://huggingface.co/vokra/whisper-base/resolve/dddd93ff6301a82aa3750d398cb2e0a599cd9e83/whisper-base.gguf \
  -o whisper-base.gguf
printf '%s\n' '7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27  whisper-base.gguf' \
  | sha256sum -c -
curl -fL --proto '=https' \
  https://raw.githubusercontent.com/ayutaz/vokra/v0.4.0-rc.1/tests/fixtures/audio/jfk-30s.wav \
  -o jfk-30s.wav
printf '%s\n' '58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f  jfk-30s.wav' \
  | sha256sum -c -
./vokra-cli-0.4.0-rc.1-x86_64-linux run \
  --backend cpu --model whisper-base.gguf --input jfk-30s.wav
```

fixtureは過去のfile名と異なり11秒です。公開済みassembled CLIのhosted Linux
CPU smokeでも、transcriptには “ask not what your country can do for you”
が含まれました。正確な処理速度や認識精度をreleaseの保証とは扱いません。

## Known limitations and reporting

TTS、全catalog推論、独立実weight数値parity、モデル別Apple CPU/Metal/no-fallbackの
完了はpreviewの合格主張に含みません。報告時は正確なrelease/tag、OS/architecture、
model revision/hash、command、error、期待した動作を添え、keyやprivate音声は含めないで
ください。未対応backendは暗黙CPU fallbackではなく明示errorを返す必要があります。
