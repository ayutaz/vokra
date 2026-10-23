# v0.4.0-rc.1 preview guide

**English** | [日本語](release-0.4.0-rc1.ja.md)

This guide targets the published CPU-only `v0.4.0-rc.1` preview. Confirm the
exact pre-release and its verified assets on the [GitHub release page](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)
before downloading; do not substitute another tag's assets.

## Publication evidence

The release is GitHub release `416018534`, published at
`2026-10-17T19:37:51Z` from exact source/tag
`d84d311e98d0abb2987b5c4e96157ae8f2068e1a`, with `draft=false` and
`prerelease=true`. Latest remains `v0.3.0`; the release contains 22 uploaded
assets. Final hosted run [`39062234813`](https://github.com/ayutaz/vokra/actions/runs/39062234813)
passed all 10 jobs, including three-OS producers, fresh transferred native CLI
help/C ABI exact-version checks, Linux assembled-CLI Whisper/Silero functional
smoke, and publication byte round-trip. The recovered receipt SHA-256 is
`d4b4656fbfa67c8f10b7dd5f6642fc896780ab11da4bc4db80d7922bb2348697`; its
distribution identity records the Linux CLI as 13,344,272 bytes with SHA-256
`2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4` and the
manifest SHA-256 as
`32b03f537f48ff6f2bb7b12fda1a9426090debbd59da361c0ffa59e8f7a60e92`.

## Scope

The preview distributes CPU-only desktop CLI and C ABI binaries. It does not
enable optional GPU backends, update model weights, publish to external
package registries, or promise API/ABI stability. Model coverage remains
per-model; a binary build or `--help` test does not prove model inference on
that platform.

Whisper base ASR and Silero VAD v5 speech/silence passed functional smoke on
the published assembled Linux CLI at source/tag
`d84d311e98d0abb2987b5c4e96157ae8f2068e1a` in run
[`39062234813`](https://github.com/ayutaz/vokra/actions/runs/39062234813).
This is functional CPU smoke only: it is not independent numerical parity, a
quality/performance benchmark, or Apple/Metal acceptance.
See the [original result and limits](handoff/release-0.4.0-preview-cpu-results-2026-10-17.md)
and [candidate acceptance record](handoff/release-0.4.0-rc1-packaging-2026-10-17.md).

> **2026-10-23 license/provenance review warning — hold Whisper quickstart
> recommendation and sharing.** The historical published CPU smoke and the
> exact pinned artifact identity below remain recorded evidence; they have not
> been rewritten. However, the pinned Hugging Face Whisper card/API still
> reports the weight as `apache-2.0` and describes its upstream distribution
> source generically, without a concrete upstream repository revision. The
> OpenAI upstream `LICENSE` and the repository's license audit identify Whisper
> as MIT. This discrepancy remains unresolved. Until the upstream source,
> revision, license, attribution, and artifact provenance are reconciled, the
> commands below are exact historical reproduction instructions only—not a new
> license approval, redistribution recommendation, or permission to republish
> or share the weight. See the [2026-10-23 announcement-readiness review](handoff/announcement-readiness-2026-10-23.md).

## Choose an asset

| Platform | CLI asset |
| --- | --- |
| Linux x86_64, glibc | `vokra-cli-0.4.0-rc.1-x86_64-linux` |
| Linux x86_64, static musl | `vokra-cli-0.4.0-rc.1-x86_64-linux-musl` |
| macOS Apple Silicon | `vokra-cli-0.4.0-rc.1-aarch64-macos` |
| Windows x86_64 | `vokra-cli-0.4.0-rc.1-x86_64-windows.zip` |

Intel macOS and other architectures are not included in this desktop preview.
Each asset has a `.sha256` sidecar; the complete payload also includes
`vokra-desktop-manifest.json`, separate CLI/C ABI SPDX SBOMs, the native
libraries and `vokra.h`. Do not substitute assets from another tag/run.

## Linux installation

Use an empty directory on a machine where model execution is permitted. These
instructions are not authorization to run models on the maintainer Mac.

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

For the static musl CLI, use the `-linux-musl` binary and its matching sidecar.

## macOS and Windows installation

On Apple Silicon, download the `-aarch64-macos` binary and its sidecar into
the same empty directory. Verify with
`shasum -a 256 -c vokra-cli-0.4.0-rc.1-aarch64-macos.sha256`, make it executable
with `chmod +x`, then run its `--help`. Do not disable Gatekeeper or remove
quarantine automatically; this preview does not claim notarization.

On Windows x86_64, verify the downloaded ZIP's SHA-256 using PowerShell
`Get-FileHash -Algorithm SHA256` against its sidecar before extracting it.
Extract into an empty directory and run `./vokra-cli.exe --help` from there.
The ZIP contains exactly one `vokra-cli.exe`.

## Reproduce the bounded Linux ASR example

For this published release, fetch the fixed model revision and the
audio fixture from that exact release tag. The model remains separately
distributed under its model-card licence; a runtime release does not change
its licence or redistribution conditions.

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

The fixture is 11 seconds despite its historical filename. The hosted Linux CPU
smoke for this published assembled CLI produced a transcript containing “ask
not what your country can do for you”. Exact performance and transcription
accuracy are not release guarantees.

## Known limitations and reporting

TTS, all-catalog inference, independent real-weight numerical parity and
per-model Apple CPU/Metal/no-fallback completion are not preview exit claims.
Report the exact release/tag, OS/architecture, model revision/hash, command,
error text and expected behavior, without keys or private audio. Unsupported
backends must return an explicit error rather than silently fall back to CPU.
