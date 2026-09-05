# RMVPE parity sidecar (VAST-only, currently blocked)

This sidecar is an independent numerical-reference harness for the native
RMVPE CPU/Metal implementation. It imports the exact upstream implementation
only while producing VAST evidence; no upstream Python, PyTorch, checkpoint,
or generated model artifact enters the Vokra runtime or this checkout.

The route is intentionally blocked before dependency resolution or artifact
acquisition. Run the stdlib-only gate from the repository root to inspect the
current reason:

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
`numpy==2.3.5`, `soundfile==0.14.0`, `torch==2.7.1`, and
`torchaudio==2.7.1`, plus the exact `safetensors==0.8.0` checkpoint bridge.
Both PyTorch packages resolve only from the official CPU
index `https://download.pytorch.org/whl/cpu`.

The lock contains 40 package rows, including platform-qualified torch and
torchaudio rows. The inspector binds every row's name/version/source,
resolution marker, and dependency qualifier into canonical digests:

- lock SHA-256: `747057f4e8596d801d5d0450e6e10a33fc467ab9e9a6cf2063460d1ea019919d`
- package/dependency rows: `ecc622c63e8a487c4440cdc838f22af7b31fae783cca41f693b0f870dd9a1819`
- resolution markers: `70a0c0d228b605430c8219bfc8e4ed66652a5f06d64cab841fee543266f3bffa`
- version-keyed license evidence: `2afebac3c079863d28415885412c11fd2acf7e3f3b9a686e2c855455da8eedec`

Resolution is not license approval. The gate remains blocked for the
`librosa -> soxr` LGPL/native route; `soundfile`'s bundled libsndfile LGPL
and cffi native route; numba/llvmlite native wheels; NumPy/SciPy bundled
notices; official CPU torch/torchaudio bundled notices; and unresolved
MPL-2.0, Unlicense, and PSF-2.0 policy rows. Owner sign-off must be recorded
before any environment sync or source/model acquisition.

## Future VAST evidence

Once the gate is separately approved, the worker uses the fixed source and
release, loads the pickle only with `torch.load(weights_only=True)`, prepares
an ephemeral `unknown`-provenance GGUF, and emits raw little-endian PCM,
hidden `[frames, 384]`, probabilities `[frames, 360]`, argmax, and F0
fixtures. The existing `parity_rmvpe` CPU/Metal tests remain the numerical
gate. A successful conversion or parity run still does not authorize upload.
