# MOSS Audio Tokenizer Nano reference gate

This is a dedicated Python 3.12, Linux/x86_64 VAST oracle project for
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` at revision
`6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`. It is separate from the general
parity environment and contains a resolver-generated 35-lock-row closure for
Linux/x86_64 Python 3.12: Torch 2.13.0+cpu from the official PyTorch CPU
index and the isolated security pin Transformers 5.10.4 from PyPI. CUDA,
NVIDIA, and Triton distributions are explicitly rejected. The prior
2.7.1+cpu Torch pin and 5.5.0 Transformers pin are previous
isolated-reference provenance only; real-weight/API runtime compatibility is
not claimed. The four open Nano Dependabot alerts require Torch 2.8.0,
2.9.1, 2.10.0, and 2.13.0 respectively; the current 2.13.0+cpu pin meets
the highest patched minimum.
The 35 lock rows comprise 34 active installed distributions plus one virtual
project row; the virtual row is not an installed package.
Every non-virtual lock row carries resolver URL, SHA-256, and positive size
metadata. The official PyTorch CPU simple index omitted the Torch wheel's
size field; the exact 191817609-byte size was recorded from an HTTP HEAD of
the locked URL. No wheel or model was downloaded on the maintainer Mac.
No package sync is performed by the local gate.
NumPy 2.3.5 is deliberately excluded from the wheel closure: the project
forces a source build with `-Dallow-noblas=true`, `-Dblas=none`, and
`-Dlapack=none`. This removes the wheel's bundled BLAS/libgfortran/
libquadmath payload from the intended Linux route, but it is not yet an
audited result. A fresh VAST sync must prove that `numpy.libs` is absent and
that every installed NumPy native extension has no forbidden basename or ELF
`NEEDED` dependency. This change does not approve NumPy or any other package;
all 35 owner-review rows remain fail-closed.

The exact upstream payload contract at the fixed revision is eight files: the
seven non-weight files `.gitattributes`, `README.md`, `__init__.py`,
`config.json`, `configuration_moss_audio_tokenizer.py`,
`modeling_moss_audio_tokenizer.py`, and `model.safetensors.index.json`, plus
`model-00001-of-00001.safetensors`. There is no `LICENSE` file in that
complete tree. The authenticated HF model card reports `cardData.license` as
`apache-2.0`; this is recorded separately from the absent license file. The
2026-08-01 owner sign-off by yousan in `docs/license-audit.md:671` records
Apache-2.0 / Commercial for source and weights; it does not approve the Python
closure or parity. The manifest binds authenticated non-weight
byte/SHA-256 and canonical Git-blob SHA-1 identities, while the shard remains
server-identity-only with `content_not_downloaded=true`, server size 87922568,
LFS payload SHA-256, and LFS pointer Git-blob SHA-1. Transformers 5.10.4 is above the GHSA-xrqw-3rrv-vx5w
patched minimum of 5.10.0. Security remediation is complete for this pin;
the route remains `BLOCKED_UNVERIFIED_API_SMOKE` until authorized VAST API
and real-weight validation completes. The VAST evidence binds an authenticated
model-free API/meta-device route using `AutoConfig.from_pretrained` and
`AutoModel.from_config`; a successful route manifest records the exact
`torch==2.13.0+cpu` runtime before that probe. An unverified route records no
observed Torch version. No weight was loaded or executed. The meta-device
inspection authenticated quantizer shape `1x768x2` and nine decoder taps
through `decoder_8`; real-weight/API runtime compatibility and numerical
parity remain unresolved and blocked.
The fixed source `config.json` also binds the official Transformers mapping
`AutoConfig -> MossAudioTokenizerConfig` and `AutoModel ->
MossAudioTokenizerModel`. The dependency-free audit checks this mapping and
the shape-bearing decoder layout. The model construction probe now uses
PyTorch's native `torch.device("meta")` context and does not import a
third-party meta-device helper. It remains a model-free shape probe only; the
inspector's output is still `BLOCKED_UNVERIFIED_API_SMOKE` for runtime/API and
parity approval until the route is actually run on VAST and reviewed. The
official model methods remain part of the recorded contract.
Its ordered meta taps are `quantizer 1x768x2`, `decoder_0 1x192x8`,
`decoder_1 1x768x8`, `decoder_2 1x384x16`, `decoder_3 1x768x16`,
`decoder_4 1x384x32`, `decoder_5 1x768x32`, `decoder_6 1x384x64`,
`decoder_7 1x240x64`, and `decoder_8 1x1x15360`; after official channel
restoration the audio shape must be `1x2x7680`.
`license_gate.py` intentionally exits 2 before any uv
sync, source/model acquisition, conversion, Cargo, or accelerator work.

The first public `vokra/moss-audio-tokenizer-nano` GGUF is historically
mis-stamped with Full metadata and is never accepted by this gate. A corrected
replacement may only be converted on VAST and remains `MEASURED_NOT_GATED`
until an owner reviews the independent official reference and numerical bound.

Run only the dependency-free gate and its self-test locally:

```text
uv run --no-project --python 3.12 python license_gate.py --self-test
```

Before owner approval, the fixed-revision source contract can be collected on
a disposable VAST Linux host without running a model:

```text
scripts/publish/vast-ai/run-moss-audio-tokenizer-nano-inspection.sh \
  --expected-head <40-hex-commit>
```

The inspection performs its host, clean-checkout, pinned-environment, and
source-only acquisition guards before the model-free probe. It materializes
only the seven non-weight files; the safetensors shard remains server-identity
only. The `AutoConfig.from_pretrained` plus meta-device
`AutoModel.from_config` probe does not load or execute a safetensors tensor.
The dependency/native payload audit remains a separate no-model phase.
The output remains `BLOCKED` with source/weight `REVIEWED` (owner sign-off
`docs/license-audit.md:671`) but unresolved Python closure, real-weight/API
runtime approval, and `NOT_RUN` numerical parity; publication remains
`NO_UPLOAD`. The
evidence also binds the exact clean Vokra checkout `{expected_head, head,
clean}`. A complete inspection intentionally exits 2 so its evidence must be
recovered and reviewed before any conversion or parity worker is started.
An `INSPECTION_ERROR` manifest is never treated as complete.

Evidence output is no-clobber: the inspector refuses an existing output path,
including a prior evidence directory, and every inspection exits status 2 so
the evidence cannot be mistaken for runtime, parity, or publication approval.

The owner approval path is `MOSS_AUDIO_TOKENIZER_NANO_LICENSE_APPROVAL`; the
tracked manifest still cannot be self-approved because the 35 package-review
rows, real-weight runtime, and parity gates remain unresolved. The authenticated
model-free route is evidence only and does not grant execution approval.

The dependency/native-payload audit is a separate no-model VAST phase. The
previous exact Torch 2.13.0+cpu wheel closure was audited at exact Vokra HEAD
`94c7f63ddd3d5e773a5b8df23a3ac29c7f6caf09`; the canonical report is
`BLOCKED` with all 35 package-review rows unresolved and is bound by SHA-256
`c50199aa53c8659e6ac234ccf1e1af2da56e5be579457df1598ef45619d48eac`.
That historical report is superseded by the source-build project contract and
does not establish a result for NumPy 2.3.5. It records an audit attempt, not
owner approval. After the exact updated project has been synced on the
disposable Linux/x86_64 host, run:

```text
scripts/publish/vast-ai/audit-moss-audio-tokenizer-nano-dependencies.sh \
  --expected-head <40-hex-commit> \
  --output /dev/shm/moss-audio-tokenizer-nano-dependency-audit.json
```

It uses `--no-sync` and records every locked artifact URL/hash/size, installed
package license/EULA bytes, and hashes plus ELF `NEEDED` facts for native
payloads. CUDA/NVIDIA and Triton distributions are explicitly rejected. The
NumPy row additionally requires the source-build/no-BLAS configuration and
audits the installed package for the absence of `numpy.libs`, forbidden native
basenames, and forbidden ELF `NEEDED` links.
It never requests model files, imports model code, invokes Cargo, converts, or
publishes.  A blocked report is expected until the owner reviews the package
and native-payload rows; it is not a real-weight/API runtime/parity approval.  The
required `--expected-head` is checked against a clean Vokra checkout and is
bound into the report.  Report creation is atomic and no-replace; a concurrent
creator cannot be overwritten.
