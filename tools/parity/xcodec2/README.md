# X-Codec2 official parity oracle

> **Publication boundary (2026-10-04): `NO_UPLOAD`.** The primary-source
> license/build decisions and production-gate packet required for a replacement
> artifact are still pending in open draft PR #152. The archive-only audit
> verified candidate `a7bb24cbd7896d770d4da375febc3669b768ae9f`; it did not
> approve execution or publication of the dependency closure.
> That candidate is not merged and must not be described as released. The
> audited GGUF hash below identifies an input only; it does not authorize
> conversion, publication, or a fresh parity/completion claim.

## Latest archive-only evidence and disposition — 2026-10-04

The final A7 source candidate passed 35 stdlib-only tests and five helper
self-tests on disposable VAST worker `54137450`. All 13 source/contract
hashes, the clean HEAD, and main baseline
`4d753d81b084f6d2369e330656035042fd00fb9f` matched before and after.
The merged audit collected all 62 external archive rows; the earlier 61 rows
were retained unchanged and only the missing Torch row was supplemented.
This is **ARCHIVE_ONLY / BLOCKED_OWNER_REVIEW / NO_UPLOAD**, not approval of
all 62 licenses, installed RECORD/build identity, or runtime compatibility.
The worker and its stored data were destroyed after small-evidence recovery.

Primary bytes confirm LGPLv3 in `frozendict==2.4.7` and in setuptools 84's
vendored autocommand, and GPL/GCC-exception plus LGPL notices for NumPy 2.0.2's
bundled Fortran components. The exact Torch wheel also contains
`torch/lib/libgomp.so.1`, needed by `libtorch_cpu.so`, `libshm.so`, and
`libtorch_global_deps.so`; its exact source/license/exception binding remains
unresolved. Generic GPL notices in Torch's NVTX, kineto test, and llvm-openmp
license trees are not by themselves proof that those components are in the
selected native payload.

`frozendict` is required by the official decoder's
`xcodec2 -> vector-quantize-pytorch -> einx -> frozendict` path. Removing a
lock edge or substituting our own FSQ implementation would not preserve the
official independent oracle. Setuptools remains a Torch dependency even if
the direct declaration is removed. De-vendored setuptools and a source-built
NumPy wheel are unbuilt, unapproved candidates, not fixes. No downgrade to a
vulnerable historical Torch pair or change to upstream dependency metadata
is accepted as a shortcut.

The [dated audit and remediation record](../../../docs/handoff/xcodec2-dependency-license-disposition-2026-10-04.md)
binds the recovered receipt hashes, confirmed blockers, remaining evidence,
and acceptance criteria. Its current readback supersedes the pending-primary
collection and pending A7 remote-test statements in the older sections below;
historical runs retain their own exact heads. The execution gate manifest
still deliberately refuses the unapproved closure. This documentation update
does not relabel the A7 test receipt as a test of a later documentation commit.

## 2026-10-04 existing-PR refresh boundary

PR #152 is being refreshed in its existing branch, rather than replaced by
another PR. Its previous head was
`b0add994b4d0350c338e60cb29b8165d8145fe32`; the integration baseline is
`main` at `4d753d81b084f6d2369e330656035042fd00fb9f`. The refreshed candidate
is still draft and `NO_UPLOAD`. The earlier replay heads below are dated
evidence, not verification of this new integration.

The initial model-free, stdlib-only refresh tests pass (32 tests), as do the dependency audit
self-test and documents-only guard. No dependency installation or third-party
import was performed in this refresh. A fresh exact-HEAD remote verification
and CI are still required before a merge-readiness decision. In particular,
setuptools' bundled license payload, NumPy's native license closure, XCodec2's
primary license bytes, and the patched-pair compatibility/owner review remain
unresolved. Passing a fail-closed self-test does not clear those blockers or
authorize a model run, artifact replacement, or upload.

### Additional hardening candidate, still blocked

The follow-up candidate makes the already-locked `setuptools==84.0.0`
explicit in the project and override contract, without changing its official
wheel URL, digest, or size. It records a **`CANDIDATE_NOT_BUILT`** downstream
de-vendoring contract. Setuptools' [official v71.0.0 release
notes](https://setuptools.pypa.io/en/latest/history.html#v71-0-0) describe that
downstream packaging option and warn that compatible external dependencies
are required. That documentation is not evidence of a clean derived wheel,
an audited replacement dependency closure, or compatible XCodec2 execution.
This candidate does not remove vendored files, build or install a replacement,
or clear GPL/LGPL, native-payload, owner/legal, or publication blockers.

The dumper hardening preserves the audited 3,291,064,672-byte GGUF input
contract rather than imposing a local-memory-sized cap on a VAST input.
Code input must be a bounded, regular, uint32-aligned file. All four output
paths are checked for existing files or symlinks before imports/inference;
publication of each new file uses an exclusive temporary file and a
no-overwrite atomic link. This is file-I/O safety, not numerical parity. The
workflow still requires a trusted output parent: path ancestry checks are not
a sandbox against a concurrently hostile process replacing parent directories.
The reference equations, source pins, and numerical bounds are unchanged.

The manager-reviewed follow-up passed 35 stdlib-only tests, including a
partial-writer failure, a competing output-file creation, and AST ordering
checks for the four-path preflight. The five audit/collector/derived-source
self-tests also passed without installing dependencies or importing their
runtime packages. These results verify the tooling contract only: primary
package license bytes, native `NEEDED` review, compatibility, and exact-HEAD
remote verification remain pending. The PR remains draft and `NO_UPLOAD`.

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
uv run --frozen --python 3.12 python dump_reference.py \
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
For the target Linux x86_64 / CPython 3.12 environment, the marker- and
activated-extra-aware lock closure is 63 rows: one virtual project row and 62
external distributions. The earlier 57-row tally was incomplete: it included
the two macOS-only resolution rows `torch==2.13.0` and
`torchaudio==2.11.0`, while omitting the eight `fsspec[http]` / `aiohttp`
subgraph rows (`aiohttp`, `aiohappyeyeballs`, `aiosignal`, `attrs`,
`frozenlist`, `multidict`, `propcache`, and `yarl`). The Linux rows are
`torch==2.13.0+cpu` and `torchaudio==2.11.0+cpu`.
The committed `dependency_audit.json` and `license_gate_manifest.json` bind
those counts and remain `BLOCKED_PENDING_PRIMARY_BYTES` / `NO_UPLOAD`; they
are evidence contracts, not owner/legal approval.
For Torch 2.13.0 / 2.13.0+cpu and TorchAudio 2.11.0 / 2.11.0+cpu, the
versioned official wheel metadata was checked for the license expression and
classifier shown below. The LICENSE text and bundled third-party notices were
not audited in this candidate; the candidate gate manifest contains no
owner/legal approval or publication sign-off.

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
only the official decoder source/API contract. Before any Torch, NumPy, GGUF,
or XCodec2 import, the reference performs a metadata-only preflight: each
audited distribution must expose the exact version, a bounded, duplicate-free
RECORD whose installed file hashes and sizes match, a regular package entry
whose importlib origin is that same distribution, and the fixed XCodec2/TorchTune
source hashes. Symlink ancestry and same-version shadow modules are rejected.
This is installed-byte/RECORD evidence only, not archive authentication or a
license/owner approval. The direct CLI, decoder-import callable, and
dependency guard also read the existing audit/owner contract before any
third-party import; the current `BLOCKED_PENDING_PRIMARY_BYTES`, `NO_UPLOAD`,
and unresolved-owner state together therefore fails closed. `NO_UPLOAD` alone
is not treated as an execution prohibition for a future separately reviewed
private reference scope. No environment variable or proof object can override
the current blocked state. A
successful documents-only
guard does not certify ABI compatibility on a target host, the full model API,
real-weight execution, or CPU/Metal parity; those require the recorded VAST
follow-up.

## Candidate dependency evidence collector

`dependency_audit.py` performs only model-free lock/document checks. The
separate `collect_dependency_evidence.py` collector is intended for the
frozen Linux x86_64 / CPython 3.12 environment after an exact clean HEAD is
provided. It records installed versions, locked archive URLs and SHA-256
values, literal archive/installed LICENSE or NOTICE bytes, and native ELF
`readelf -d` NEEDED entries. It does not import Torch or XCodec2, acquire a
model, execute weights, or grant a license decision. Even a complete factual
collection is emitted as `BLOCKED_OWNER_REVIEW` with `NO_UPLOAD`; missing
primary bytes, xcodec2's primary license, and NumPy GPL/GCC/LGPL runtime
components remain explicit blockers. For a selected wheel, the collector
parses the publisher `RECORD` inventory, then streams and compares archive
bytes/SHA-256 with the installed LICENSE, METADATA, and native payload files.
It independently inventories archive LICENSE/native/METADATA members and
rejects any such member omitted from publisher `RECORD` or the installed
payload inventory.
The installed `RECORD` itself is not compared byte-for-byte because installers
rewrite it; `RECORD`, `WHEEL`, `INSTALLER`, `REQUESTED`, and generated `.pyc`
files are explicitly excluded from the payload proof. The result is scoped as
`PAYLOAD_SCOPED_VERIFIED`, while full installed-build identity remains
`UNVERIFIED`. A locked sdist is also marked `UNVERIFIED` and remains factually
blocked until an authenticated build/RECORD proof exists. Wheel `.data`
relocation is explicitly unsupported by this collector; any such publisher
path is bounded as a factual blocker rather than being treated as a general
wheel-install mapping.

Local documents-only checks (no environment sync) are:

```bash
cd tools/parity/xcodec2
uv run --no-project --python 3.12 python dependency_audit.py --self-test
uv run --no-project --python 3.12 python collect_dependency_evidence.py --self-test
uv run --no-project --python 3.12 python -m unittest discover -s . -p 'test_*.py'
```

The bounded collector itself is a VAST-only operation and must use a clean
checkout whose expected HEAD is supplied explicitly:

```bash
uv sync --frozen --python 3.12
uv run --frozen python collect_dependency_evidence.py \
  --project /abs/checkout/tools/parity/xcodec2 \
  --output /tmp/xcodec2-dependency-evidence.json \
  --expected-head <exact-clean-lowercase-40-hex>
```

The collector also applies one aggregate monotonic deadline (15 minutes),
8-GiB artifact input cap, 8-GiB aggregate streamed-unpacked-I/O cap, 8-MiB
retained-license cap, and 32-MiB serialized output cap in addition to the
per-artifact/member limits. The unpacked cap covers bounded streaming reads
without retaining a whole wheel in memory; archive inventory and installed
binding comparisons may perform separate bounded passes, and every pass is
counted against this aggregate I/O budget. The audit and gate manifest bind
SHA-256 digests
for every pre-import helper (`dependency_audit.py`, `dependency_guard.py`,
the collector, locked-sdist inspector, derived-sdist builder, and
`dump_reference.py`); changing one of these files invalidates the contract.

The output is an audit packet only. It must not be uploaded or used as a
publication/sign-off record without separate owner/legal review.

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

The Transformers link and these independently audited digests describe the
excluded `transformers==5.10.4` row: sdist SHA-256
`de37741509e64ccb88f7f5708beaf5b1914df447f5fe659f9c0fd95950413168` and
extracted `LICENSE` member SHA-256
`77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049`. They
are not an installed frozen-lock package record: the current `uv.lock`
contains the impossible-marker/declared pin but does not record this sdist URL
or either payload digest.

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
