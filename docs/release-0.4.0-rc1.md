# v0.4.0-rc.1 preview guide

**English** | [日本語](release-0.4.0-rc1.ja.md)

This guide targets the CPU-only `v0.4.0-rc.1` preview. **A source version is
not proof of publication.** Use the commands only after locating the exact
`v0.4.0-rc.1` pre-release and its verified assets on the [GitHub releases page](https://github.com/ayutaz/vokra/releases).
Before publication, these URLs intentionally
do not provide a candidate; do not substitute another tag's assets.

## Scope

The preview distributes CPU-only desktop CLI and C ABI binaries. It does not
enable optional GPU backends, update model weights, publish to external
package registries, or promise API/ABI stability. Model coverage remains
per-model; a binary build or `--help` test does not prove model inference on
that platform.

Whisper base ASR and Silero VAD v5 speech/silence passed an earlier hosted
Linux CPU functional smoke at source `9ee5834d`. The candidate's packaged
binary is subject to separate validation before publication; consult the
candidate record for the actual accepted HEAD and run, if recorded.
Neither test is independent
numerical parity, a quality/performance benchmark, or Apple/Metal acceptance.
See the [original result and limits](handoff/release-0.4.0-preview-cpu-results-2026-10-17.md)
and [candidate acceptance record](handoff/release-0.4.0-rc1-packaging-2026-10-17.md).

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

After publication, use an empty directory on a machine where model execution
is permitted. These instructions are not authorization to run models on the
maintainer Mac.

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

After the candidate is published, fetch the fixed model revision and the
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

The fixture is 11 seconds despite its historical filename. The earlier
transcript contains “ask not what your country can do for you”. Exact
performance and transcription accuracy are not release guarantees.

## Known limitations and reporting

TTS, all-catalog inference, independent real-weight numerical parity and
per-model Apple CPU/Metal/no-fallback completion are not preview exit claims.
Report the exact release/tag, OS/architecture, model revision/hash, command,
error text and expected behavior, without keys or private audio. Unsupported
backends must return an explicit error rather than silently fall back to CPU.
