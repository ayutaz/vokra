# OWSM primary dependency license readback — 2026-10-18

Management-only supplement at Vokra main
`8ae9566e25753b361be5703d46bcdad967f7cefb`. This preserves the
[existing OWSM admission and fixed-frontend findings](owsm-audit-admission-readback-2026-10-14.md)
and records additional primary-source inspection. It is not license advice,
an owner approval, an execution grant or a new numerical result.

## Exact source and inspection boundary

The existing reference closure includes `torch-complex` 0.4.4. Its PyPI
homepage metadata names `kamo-naoyuki/torch_complex`, whose GitHub API lookup
returns 404; the package README points to the actual
[official source repository](https://github.com/kamo-naoyuki/pytorch_complex).
Do not mistake the broken homepage spelling for disappearance of that source.
At inspection, repository license metadata is null and master remains
`8a2ad1e47f3df25a30eb426f6ad781b89103fab3`.

The [fixed recursive tree](https://api.github.com/repos/kamo-naoyuki/pytorch_complex/git/trees/8a2ad1e47f3df25a30eb426f6ad781b89103fab3?recursive=1)
has seventeen entries, thirteen text blobs and `truncated=false`, with no
LICENSE/COPYING/NOTICE filename. All thirteen text bodies were retrieved at
that fixed revision; each connector-reported Git blob SHA matches the fixed
tree. This is not an independently recomputed archive/received-byte SHA-256
audit. No wheel/sdist is downloaded, installed or imported by this follow-up.

| Fixed source path | Git blob SHA-1 |
| --- | --- |
| [Upstream publication workflow](https://github.com/kamo-naoyuki/pytorch_complex/blob/8a2ad1e47f3df25a30eb426f6ad781b89103fab3/.github/workflows/pythonpublish.yml) | `786a7c41485122a8503b59f87824d32b01727ed7` |
| `.gitignore` | `276aaa99f96529f895519d48e6c2eac75fa65f6a` |
| `.travis.yml` | `4e8450bd5db7b59924bb814546d37e6cfd94c8bf` |
| `README.md` | `1fea039d5c8a604107da3d5724c829b6dbb5e829` |
| `__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
| `setup.cfg` | `9c3b61d00065a6e7f5a3dee70c0a7a3f7d2b160d` |
| `setup.py` | `5d85fb40768f555b3dec3876e1d3e514711e78b2` |
| `tests/test_basic_math.py` | `eceab1901fa70752b76381482c26d962eaf57b7d` |
| `tests/test_functional.py` | `2556952fa5ddee850d9cd4a25f029a2b866b55b2` |
| `torch_complex/__init__.py` | `d19a54296ea1b29f7859432317fa5eda52577b32` |
| `torch_complex/functional.py` | `dddf7b6a0d82fec2cf9b10bad0436b44e5f8414d` |
| `torch_complex/tensor.py` | `be6fccb7f18f53f20d8eddce971286969600640a` |
| `torch_complex/utils.py` | `10546082b705891bdb2637acc9c459af7b678d05` |

## Why this does not close the gate

A case-insensitive license/copyright/permission/SPDX/redistribution keyword
scan finds only two candidate lines. Full contextual review identifies a
generic contribution-header lint comment in `setup.cfg` (the corresponding
checks are ignored), and an Apache license classifier in `setup.py`. Neither
supplies actual complete license terms for the existing package. The other
eleven text blobs have no keyword match. This finding is not a legal claim
that nobody holds rights; it means the required primary grant is still absent
from this inspected evidence. No upstream source bodies are republished here.

The [PyPI release JSON](https://pypi.org/pypi/torch-complex/0.4.4/json)
inspection likewise reports null license/expression/files fields and an
Apache classifier. That web result is marked crawled two days earlier, not
a fresh archive audit. Existing archive findings remain historical evidence;
the metadata and classifier are not substituted for authoritative terms.

The fixed ESPnet frontend actively constructs `ComplexTensor`; the prior
readback already rejects the guessed native-complex selection route. Next
obtain authoritative terms/permission for the exact dependency, or first
authenticate a genuinely compatible official alternative reference contract,
then close dependency/native/operator review. Do not repeat paid model compute
to resolve missing rights or replace the oracle with a local mirror.

The dependency gate remains blocked. No lock, source pin, protected manifest,
signer, numerical tolerance, owner decision, model status or publication state
changes. No model download/forward, package execution or paid worker is used.
