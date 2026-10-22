# RMVPE parity sidecar (VAST-only, currently blocked)

This sidecar is an independent numerical-reference harness for the native
RMVPE CPU/Metal implementation. It imports the exact upstream implementation
only while producing VAST evidence; no upstream Python, PyTorch, checkpoint,
or generated model artifact enters the Vokra runtime or this checkout.

The route is intentionally blocked before artifact acquisition, environment
execution, or model/reference use. Metadata-only dependency resolution may be
performed to maintain the security candidate lock, but it is not an install or
runtime validation. Run the stdlib-only gate from the repository root to
inspect the current reason:

```text
uv run --no-project --offline --python 3.12 python tools/parity/rmvpe_inspect.py --dependency-gate
```

It must return exit 2 until all owner approvals below exist. Do not run
`uv sync`, download source/checkpoints/wheels, convert, or run Cargo locally.
The later validation path is VAST-only and is reached only after the gate is
explicitly changed to allow it:

```text
VOKRA_PUBLISH_ON_VAST=1 scripts/publish/vast-ai/run-rmvpe-validation.sh \
  --checkpoint-sha256 <independently-recorded-64-hex-sha256>
```

The worker itself requires Linux x86_64, at least 64 GiB RAM, 150 GB free disk,
a clean checkout, and the exact locks. It never uploads or pushes. Its
checkpoint SHA and the release-archive SHA remain unset until acquisition on
VAST; no model bytes have been acquired for this staging change.

## Immutable reference identity

- upstream: `https://github.com/yxlllc/RMVPE`
- source commit: `0aabafba18289ca938a73af0b0297686abf4922d`
- reference class: `src.inference.RMVPE` / `src.model.E2E0`
- frontend: 16 kHz, hop 160, `n_fft = win_length = 1024`, 128 HTK mels
- decoder: nine-bin local average, threshold `0.03`, no Viterbi
- release: tag `230917`, asset `rmvpe.zip`, 340638958 bytes, member `model.pt`

Source-role Git blob identities and the release metadata are emitted by the
stdlib inspector. Git blob IDs are not file SHA-256 values. The source has no
LICENSE file at the fixed revision, and the checkpoint license is unknown.
The historical `vokra/rmvpe` artifact is mis-stamped (`mit`/`permissive`) and
is rejected as a parity input; it must not be replaced without a separate
license grant and upload authorization.

## Dedicated Python lock

RMVPE is excluded from the parent `tools/parity` uv workspace. Its dedicated
`uv.lock` is Python `==3.12.*` with exact direct pins for `librosa==0.11.0`,
`numpy==2.3.5`, `soundfile==0.14.0`, `torch==2.13.0`, and
`torchaudio==2.11.0`, plus the exact `safetensors==0.8.0` checkpoint bridge.
Both PyTorch packages resolve only from the official CPU
index `https://download.pytorch.org/whl/cpu`; the lock records the official
`download-r2.pytorch.org` wheel mirror URLs returned during resolution.

The lock contains 40 package rows, including platform-qualified torch and
torchaudio rows. The inspector binds every row's name/version/source,
resolution marker, and dependency qualifier into canonical digests:

- lock SHA-256: `4b32df99126f739ef96a919f4c5404ea583c9c2911b4395661a21c357c79d635`
- package/dependency rows: `784e7468c3443858d9f16d46b11846d331df6de3e6846bb6b63437fc262883ff`
- resolution markers: `70a0c0d228b605430c8219bfc8e4ed66652a5f06d64cab841fee543266f3bffa`
- version-keyed license evidence: `e062eebf867187477b67d5b3c9001db9b02a14cbc1fb8df23fd6bb41656dcad6`

This security candidate changes the platform-qualified PyTorch rows from
torch `2.7.1`/`2.7.1+cpu` to `2.13.0`/`2.13.0+cpu`, and torchaudio from
`2.7.1`/`2.7.1+cpu` to `2.11.0`/`2.11.0+cpu`, using only the official CPU
index. The earlier attempt to require both packages at `2.13.0` was rejected
because the official index had no `torchaudio==2.13.0`; the published
torchaudio `2.11.0` release documents the torch 2.11 compatibility line and
future torch versions. This resolves metadata and lock identity only: it does
not establish RMVPE API/ABI parity, native wheel safety, or a model execution
pass.

The candidate is intended to remediate the vulnerable 2.7.1 Torch exposure
covered by the five reviewed advisory records: [GHSA-887c-mr87-cxwp](https://github.com/advisories/GHSA-887c-mr87-cxwp),
[GHSA-qfhq-4f3w-5fph](https://github.com/advisories/GHSA-qfhq-4f3w-5fph),
[GHSA-vgrw-7cvw-pwgx](https://github.com/advisories/GHSA-vgrw-7cvw-pwgx),
[GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), and
[GHSA-63cw-57p8-fm3p](https://github.com/advisories/GHSA-63cw-57p8-fm3p).
The [official torchaudio 2.11.0 release](https://github.com/pytorch/audio/releases/tag/v2.11.0)
documents the torch 2.11 compatibility line and future torch versions. The
last advisory is published under the global `pytorch` package alias and is
therefore missed by a torch-only metadata query; this lock records the
official `torch` package identity but does not claim that all advisories or
transitive native closure are cleared.

Resolution is not license approval. The gate remains blocked for the
`librosa -> soxr` LGPL/native route; `soundfile`'s bundled libsndfile LGPL
and cffi native route; numba/llvmlite native wheels; NumPy/SciPy bundled
notices; the changed official CPU torch/torchaudio versions' primary license
and bundled notices; and unresolved MPL-2.0, Unlicense, and PSF-2.0 policy
rows. The upstream source license, checkpoint license/SHA, publication gate,
and owner sign-off remain unchanged and blocked. Owner sign-off must be
recorded before any environment sync or source/model acquisition.

## Future VAST evidence

Once the gate is separately approved, the worker uses the fixed source and
release, loads the pickle only with `torch.load(weights_only=True)`, prepares
an ephemeral `unknown`-provenance GGUF, and emits raw little-endian PCM,
hidden `[frames, 384]`, probabilities `[frames, 360]`, argmax, and F0
fixtures. The existing `parity_rmvpe` CPU/Metal tests remain the numerical
gate. A successful conversion or parity run still does not authorize upload.
