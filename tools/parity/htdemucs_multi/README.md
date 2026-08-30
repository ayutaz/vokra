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
inspection manifest. The manifest remains `INSPECTION_ONLY` and
`NO_UPLOAD`; a digest match does not grant weight redistribution rights.

The upstream `requirements_minimal.txt` is the dependency contract. The lock
must be generated and audited on VAST, then the exact lock and primary package
license/artifact records may be committed. This checkout intentionally does
not claim any dependency or bundled-native notice approval.

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

The dependency audit row file is deliberately empty and blocked. A VAST
Python 3.12.14 resolution run proved that the unchanged upstream
`torchaudio>=0.8,<2.1` constraint is unsatisfiable: available releases are
`<=2.0.2+cpu` and `>=2.1`, and `2.0.2+cpu` has only cp38/cp39/cp310/cp311
wheels. The evidence log SHA-256 is
`ed594c9014232b79e8bed1eceae767f0b339157b93c114aa8d9b3d418c6abeba`, recorded
at run commit `73307e99c83fdd59ca9693abdc343b929fb518de`. No
model/checkpoint/Torch import was performed. Do not loosen the upstream pin.
Before a future compatible source-specific decision, VAST must fill exact
package rows (`name`, `version`, artifact SHA-256, license) and license rows (`name`, license, status, source, SHA-256),
then pin each canonical row-array digest and the uv lock digest. Each active
package row binds a selected artifact `{kind,url,sha256,bytes}`; its license
evidence repeats that exact artifact identity and binds separate license
bytes. The virtual project uses `{kind: virtual-local, url: pyproject.toml}`
only for package identity, while its license must bind the repository-root
`LICENSE` bytes. Duplicate
names, malformed rows, GPL/LGPL/unknown licenses, or an unverified
Python-3.12 `torchaudio<2.1` wheel fail closed. The upstream pin may be
unsatisfiable on Python 3.12, and older Torch candidates may lack
`get_unsafe_globals_in_checkpoint`; both wheel compatibility and scanner
availability must be proven on VAST without changing the pin here. `numpy` is
an additional source-import requirement in the sidecar (not an upstream
direct requirement), while `lameenc` remains part of the upstream direct set.

Until the lock and license audit are complete, every normal worker path exits
2 before network, dependency resolution, checkpoint acquisition, or model
execution.
