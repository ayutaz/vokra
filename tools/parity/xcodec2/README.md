# X-Codec2 official parity oracle

`dump_reference.py` imports the official `xcodec2==0.1.5` PyPI package,
verifies the installed decoder source and the audited public GGUF SHA-256,
restores the official `CodecDecoderVocos` modules, and calls their FSQ plus
decoder forward. It does not import Vokra and does not mirror the Rust
equations.

The public GGUF is 3,291,064,672 bytes, so dependency installation, reference
generation, and Rust consumer execution belong on VAST. The committed fixture
contains only codes and the official output:

```bash
cd tools/parity/xcodec2
uv sync --frozen --python 3.12
uv run --frozen python dump_reference.py \
  --gguf /path/to/vokra-xcodec2/model.gguf \
  --codes /path/to/codes.u32le \
  --output /path/to/reference
```

The project declares `xcodec2==0.1.5` and the audited patched CPU pair
`torch==2.13.0` / `torchaudio==2.11.0`, plus the audited
`transformers==5.10.4` floor, `vector-quantize-pytorch==1.17.8`,
`torchtune==0.3.1`, the official decoder source hashes, and public GGUF
SHA-256
`7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e`.

`transformers==5.10.4` is a non-yanked patched release for the path-traversal
advisory tracked by Dependabot. Keep this pin explicit and review it when a
newer patched release is published. The official `xcodec2==0.1.5`
distribution hard-pins `torch==2.5.0` and `torchaudio==2.5.0`; this oracle
uses an explicit `override-dependencies` contract and the official PyTorch CPU
index to replace those edges with the patched pair. The override is limited to
the decoder-only oracle and is not a claim that the full Transformers model
API is compatible. TorchAudio's official compatibility guide states that its
2.11 stable-ABI line supports PyTorch 2.11 and all later releases, including
2.13: [TorchAudio installation and compatibility](https://docs.pytorch.org/audio/main/installation.html).

## Dependency license/native audit

This is a model-free dependency candidate. No GGUF was downloaded or executed,
and no real-weight CPU/Metal parity was run. The lock resolves the patched CPU
pair from the explicit PyTorch index and removes the old CUDA wheel closure.
For Torch 2.13.0 / 2.13.0+cpu and TorchAudio 2.11.0 / 2.11.0+cpu, the
versioned official wheel metadata was checked for the license expression and
classifier shown below. The LICENSE text and bundled third-party notices were
not audited in this candidate; this tree has no owner/legal approval manifest,
so neither row is a commercial or publication sign-off.

The decoder-only reference path previously imported successfully with
`transformers` blocked from import under the original Torch 2.5.0 / TorchAudio
2.5.0 pair. That historical evidence has now been superseded for the patched
pair by the VAST model-free replay recorded below. The impossible-marker
override in `pyproject.toml` is recorded in `uv.lock`'s
`[manifest].overrides` and removes the unused
Transformers/Typer/tokenizers/shellingham branch from the resolved environment.
The guard is fail-closed: a review must require both `uv lock --check` and the
automated `dependency_guard.py` assertion that no `name = "transformers"`
package row returns before using this decoder-only oracle. The full official
`xcodec2.modeling_xcodec2` API is intentionally out of scope; under the
original pair it is blocked because Transformers 5.10.4 accesses
`torch.float8_e8m0fnu`, absent from hard-pinned Torch 2.5.0.

The model-free guard is:

```bash
cd tools/parity/xcodec2
uv lock --check
# Local maintainer check: no package sync/import and no model execution.
uv run --no-project --python 3.12 python dependency_guard.py --self-test --documents-only
# VAST only, after the frozen environment has been installed:
uv run --frozen python dependency_guard.py --self-test
```

The guard checks the exact Torch/TorchAudio CPU override and source index,
rejects Transformers, tokenizers, Typer, and shellingham lock or installed
rows, verifies that xcodec2's resolved dependency edges follow the patched
pair, verifies separate Linux CPU and macOS arm64 lock rows, runs tamper cases
for override/source/platform/dependency-edge reintroduction, and exercises
only the official decoder source/API contract. A successful documents-only
guard does not certify ABI compatibility on a target host, the full model API,
real-weight execution, or CPU/Metal parity; those require the recorded VAST
follow-up.

## VAST model-free replay

This is historical evidence from the VAST run at the earlier clean HEAD
`e3e90572a5125e585b0a6f9f681f5606fc38a7a4`; it is not a verification of the
current integrated candidate `696238a45f493a7a9cf4c9449810f5bc83218294`. The
current candidate has only passed the local static documents-only guard. A
fresh VAST replay is still pending.

On 2026-09-30, disposable VAST instance `53434693` verified the exact clean
HEAD `e3e90572a5125e585b0a6f9f681f5606fc38a7a4`. The frozen lock had SHA-256
`d59f4541f665d3517bec3498e8b5b48fdc24aa0299885cac7ae0574f9fc1d9d4`.
Python 3.12.14 completed `uv sync --frozen --python 3.12` with 62 packages,
and the environment resolved Torch `2.13.0+cpu`, TorchAudio `2.11.0+cpu`, and
XCodec2 `0.1.5`. Both `uv lock --check --offline` and
`uv run --frozen python dependency_guard.py --self-test` passed; the latter
validated the official decoder-only import/API contract without loading a
GGUF or executing weights. The instance was destroyed afterward; its exact-ID
API returned `instances: null` and the full instance list was empty.

This evidence covers only the frozen, model-free dependency/import boundary.
Real-weight execution, CPU parity, Metal parity, and CPU/GPU speed-versus-
quality selection remain unverified. LICENSE text and bundled third-party
notices remain unaudited, and owner/legal approval is still absent.

| Distribution | Version | License | Resolution | Native payload | Review |
| --- | ---: | --- | --- | --- | --- |
| `[torch][torch-license]` | 2.13.0 / 2.13.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT (official wheel metadata) | explicit PyTorch CPU index; macOS arm64 and Linux CPU rows | native binary wheel | owner required; no sign-off |
| `[torchaudio][torchaudio-license]` | 2.11.0 / 2.11.0+cpu | BSD License (official wheel metadata) | explicit PyTorch CPU index; macOS arm64 and Linux CPU rows | native extension wheel | owner required; no sign-off |
| `[click][click-license]` | 8.5.0 | BSD-3-Clause | resolved | none | owner required |
| `[huggingface-hub][hub-license]` | 1.33.0 | Apache-2.0 | resolved | none | owner required |
| `[versioned sdist LICENSE member][transformers-license]` | 5.10.4 | Apache-2.0 | declared, impossible-marker excluded | none | owner required |
| `[tokenizers][tokenizers-license]` | 0.22.2 | Apache-2.0 | transitive, excluded | not installed; would be native | owner required |
| `[typer][typer-license]` | 0.27.2 | MIT | transitive, excluded | none | owner required |
| `[shellingham][shellingham-license]` | 1.5.4 | ISC_BLOCKED_BY_POLICY | transitive, excluded | none | owner required |

[click-license]: https://raw.githubusercontent.com/pallets/click/8.5.0/LICENSE.txt
[torch-license]: https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
[torchaudio-license]: https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
[hub-license]: https://raw.githubusercontent.com/huggingface/huggingface_hub/v1.33.0/LICENSE
[shellingham-license]: https://raw.githubusercontent.com/sarugaku/shellingham/1.5.4/LICENSE
[tokenizers-license]: https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/tokenizers/LICENSE
[transformers-license]: https://files.pythonhosted.org/packages/f7/5d/1df789ca27a436ce09de67c8fee6acd2a528d34c28685991f8203e5418ae/transformers-5.10.4.tar.gz
[typer-license]: https://raw.githubusercontent.com/fastapi/typer/0.27.2/LICENSE

The Transformers link and the two SHA-256 values above are an independently
audited source-distribution/license-member record for the excluded
`transformers==5.10.4` row. They are not an installed frozen-lock package
record: the current `uv.lock` contains the impossible-marker/declared pin but
does not record this sdist URL or either payload digest.

The excluded `shellingham==1.5.4` remains recorded as
`ISC_BLOCKED_BY_POLICY` because ISC is outside the repository's Apache/MIT/BSD
allowlist. It is not present in the current lock, but any removal or weakening
of the impossible-marker override must stop at the license gate until
owner/legal decides the exception or an allowed-license dependency path is
available. A Typer downgrade is not assumed to remove it.

`httpx` is a newly added dependency edge of `huggingface-hub`, but its locked
distribution already existed at the previous version and is not a changed
distribution in this diff. No native payload from the excluded Transformers
branch is installed. The license classes above are evidence only, not
approval.
