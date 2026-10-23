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
The dated release evidence also records passed macOS Apple Silicon and Windows
x86_64 packaging, native CLI `--help`, and C ABI exact-version checks. Those
packaging checks are distinct from the Linux model smoke; the Linux evidence
does not establish model inference on macOS or Windows, Apple Metal support,
numerical parity, or performance.

> **2026-10-23 released-CLI pairing verification.** The historical published
> CPU smoke and exact old-pinned artifact identity remain recorded evidence; they
> have not been rewritten. The repaired artifact's provenance and MIT license
> evidence were verified separately, and the published RC1 Linux x86_64 glibc
> CLI was then verified with the repaired Whisper and Silero pins in the final
> CPU receipt below. This is functional smoke only; it is not numerical parity,
> a performance benchmark, Apple Metal acceptance, or model inference evidence
> for macOS or Windows. See the [final pairing evidence](handoff/evidence/published-rc1-cli-pairing-final-2026-10-23.json).

## Whisper provenance repair record (released-CLI pairing verified)

The bounded Whisper provenance repair was published separately at immutable
Hugging Face revision
[`bf440f6064ffc882d53d3694729dc4616dc2cfa7`](https://huggingface.co/vokra/whisper-base/tree/bf440f6064ffc882d53d3694729dc4616dc2cfa7).
The repaired `whisper-base.gguf` is 290,859,488 bytes with SHA-256
`5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014`.
Its fixed download URL is
`https://huggingface.co/vokra/whisper-base/resolve/bf440f6064ffc882d53d3694729dc4616dc2cfa7/whisper-base.gguf`.
The [repair record](handoff/whisper-base-provenance-repair-2026-10-23.md)
and [success receipt](handoff/evidence/whisper-base-provenance-repair-2026-10-23.json)
document the fixed upstream/license evidence and tensor-preserving metadata
repair; they do not alter this RC1 release's assets or historical receipts.

The original hosted Linux functional smoke above used the old Whisper pin
(`dddd93ff6301a82aa3750d398cb2e0a599cd9e83`, SHA-256
`7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27`). The
final receipt below uses the repaired pin and does not rewrite that historical
artifact or receipt.

## Repaired-model pairing verification

The final Linux CPU receipt used the published RC1 CLI SHA-256
`2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4`, the
repaired Whisper revision/hash above, and Silero revision
`3670dfed054980307eeb1711c6329c29b80bdf14` with SHA-256
`23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6`. All
three `--backend cpu` checks returned exit code 0. The final run is
[`39953770618`](https://github.com/ayutaz/vokra/actions/runs/39953770618), and
the final receipt is [archived here](handoff/evidence/published-rc1-cli-pairing-final-2026-10-23.json).
ASR stdout was:

```text
asr:  And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

The VAD speech fixture reported 343 frames with 214 speech frames and mean
`0.5889`; the silence fixture reported 343 frames with 0 speech frames and
mean `0.0025`.

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

Use an empty directory on a machine where model execution is permitted.

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

For this preview, fetch the repaired Whisper revision and the audio fixture
from their fixed public identities. The model remains separately distributed
under its model-card licence; a runtime release does not change its licence or
redistribution conditions. The released-CLI pairing is verified in the final
Linux CPU receipt described above.

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

The fixture is 11 seconds despite its historical filename. The final Linux CPU
receipt above produced this exact ASR line for the fixture:

```text
asr:  And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

This confirms the repaired model/CLI pairing for this functional smoke. Exact
performance and transcription accuracy are not release guarantees.

## Reproduce the bounded Linux Silero VAD example

After the verified `jfk-30s.wav` download above, fetch the reviewed Silero VAD
v5 artifact, verify its exact size and SHA-256, and invoke the CPU backend. The
final receipt above records the same fixed pin and Linux CPU path.

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

TTS, all-catalog inference, independent real-weight numerical parity and
per-model Apple CPU/Metal/no-fallback completion are not preview exit claims.
The hosted model functional evidence described above is Linux x86_64 glibc only;
the separate dated packaging/help/C ABI checks passed for macOS Apple Silicon
and Windows x86_64, but do not establish model inference on those platforms,
Apple Metal support, numerical parity, or performance.
Report the exact release/tag, OS/architecture, model revision/hash, command,
error text and expected behavior, without keys or private audio. Unsupported
backends must return an explicit error rather than silently fall back to CPU.
