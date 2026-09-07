# HT-Demucs Multi parity and audit sidecar

This is a dedicated Python 3.12 / Linux x86_64 project for an independent
reference run of the official Meta HT-Demucs release. It is not a Vokra
runtime dependency and is never executed on the maintainer Mac.

The VAST worker requires 32 GiB free in `/dev/shm`. This is a conservative
working-set guard for approximately 7 GiB of five checkpoint downloads, at
most about 1 GiB of selected raw taps after prefix truncation, and official
CPU reference overhead; it fits the observed 62 GiB VAST tmpfs while leaving
headroom for transient allocations.

The fixed source is `facebookresearch/demucs` at revision
`e976d93ecc3865e5757426930257e200846a520a`. The five official registry
members and their complete SHA-256 values are copied from the authenticated
inspection manifest. The manifest remains `BLOCKED_PENDING_AUTHENTICATED_MANIFEST`
with an `INSPECTION_ONLY` evidence stage and
`NO_UPLOAD`; a digest match does not grant weight redistribution rights.

The source-config contract is intentionally structural and exact: `htdemucs_ft`
uses the ordered members `f7e0c4bc`, `d12395a8`, `92cfc3b6`, `04573f0d` with
the declared 4x4 identity matrix; `htdemucs_6s` uses only `5c90dfd2` with the
derived 1x1 identity matrix and six-source output width. A single flattened
2,132-tensor artifact, an extra member, a reordered member, or an arbitrary
weight file is rejected. This contract does not authenticate tensor roles or
enable a Vokra runtime forward.

The exact upstream `requirements_minimal.txt` bytes are retained as
`upstream_requirements_minimal.snapshot` for authenticated provenance only;
it is not an active Python manifest and is preserved byte-for-byte. The
curated active `pyproject.toml` import closure is
`einops`, `julius`, `numpy`, `pyyaml`, `torch`, and `tqdm`. The upstream
`dora-search`, `lameenc`, `openunmix`, and `torchaudio` entries are explicitly
excluded from active imports: the report manually loads the authenticated
source and never uses Dora; `lameenc` is a GPL codec; `openunmix` is only
needed for the official top-level Wiener symbol import; and `torchaudio` is
replaced by the strict fixture reader below. The generated lock contains none
of those four packages. This checkout intentionally does not claim any
dependency or bundled-native notice approval.

The future worker uses the official `demucs.htdemucs.HTDemucs` class and the
official `BagOfModels`/`apply_model` aggregation from the pinned checkout. It
emits a JSON report plus selected raw little-endian F32 taps: source/config/
checkpoint identity, per-member STFT and dual-branch/cross-transformer/
terminal-stem taps, and the exact FT bag (or separate 6s member) terminal
output. Intermediate tensors retain their original shape/count but write only
the deterministic flat little-endian prefix of at most 1,048,576 F32 elements;
the manifest records `raw_count`, `raw_offset`, and `truncated`. Hooks are
selected on their first invocation only while call counts are recorded. It
never converts or uploads a model. A fixed public audio
fixture must be supplied by the VAST operator with an independently recorded
SHA-256; no fixture bytes are bundled or invented here.

The dependency audit row file is deliberately empty and blocked pending
primary artifact/license bytes. A Linux-x86_64 Python 3.12 `uv.lock` is now
resolved against the explicit PyTorch CPU index; it contains no CUDA, NVIDIA,
or Triton package/source. The lock is reproducibility evidence only: exact
package rows (`name`, `version`, artifact SHA-256, license) and license rows
remain owner-reviewed gates. Each future package row must bind a selected
artifact `{kind,url,sha256,bytes}`; its license evidence must repeat that
identity and bind separate license bytes. The virtual project uses
`{kind: virtual-local, url: pyproject.toml}` only for package identity, while
its license must bind the repository-root `LICENSE` bytes. Duplicate names,
malformed rows, GPL/LGPL/unknown licenses, or CUDA/NVIDIA/Triton indicators
fail closed. The upstream constraint remains unchanged and is not used as an
active torchaudio dependency.

Before importing official `demucs.htdemucs`, the dumper installs a
process-local `openunmix` / `openunmix.filtering` stub that exposes only a
`wiener` sentinel; any call raises. After each fixed checkpoint is
safe-loaded and instantiated, `cac=True`, `wiener_iters=0`, and
`end_iters=0` are asserted. The report records the sentinel identity and
zero calls, so the stub cannot contribute to output numerics. The official
`demucs.audio` module imports the GPL encoder even though this route only
needs its conversion helper, so the dumper then installs temporary
fail-closed stubs for both `lameenc` and `torchaudio` immediately before that
official import. Every non-metadata attribute access on either audio stub
raises; no MP3 or torchaudio functionality is exposed. The fixture reader
itself is stdlib-only and accepts exactly RIFF/WAVE, uncompressed PCM
integer, 16-bit, mono, 16000 Hz, little-endian bytes. It returns
`[channels, time]` and normalizes signed PCM16 values by dividing by 32768.
After that decode, the dumper calls the pinned upstream `convert_audio`,
preserving the official 16 kHz mono to 44.1 kHz stereo numerics without a
local resampler mirror.

The inspection worker is intentionally a separate, no-upload evidence path:
after its host and directory preflight it may acquire exactly the five fixed
checkpoint URLs and scan them while the dependency/reference gate remains
blocked. It does not import or execute the upstream model, convert a product,
or upload anything. The report-only reference worker is the path that invokes
the dependency/license and source gates before dependency resolution,
checkpoint use, or model execution; it exits 2 until those gates are approved.
Even for safe inspection, the inspector enumerates checkpoint unsafe globals
before deserialization and permits only the exact reviewed set bound to the
pinned `demucs/htdemucs.py` source blob; unknown globals and a missing/dirty
source checkout block before `torch.load`. The restricted load uses
`weights_only=True` and an inert class token, never `weights_only=False`.
The next VAST first pass must therefore provide a clean Git checkout at the
fixed revision, complete dependency/license evidence, and safe-loaded
per-member name/shape/dtype manifests before any converter or native forward
work can be considered.
