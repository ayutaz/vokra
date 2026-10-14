# OWSM dependency admission readback — 2026-10-14

> **2026-10-15 JST official-frontend follow-up:** a read-only investigation
> at Vokra main `627a641ffe96441a1c1e7040f151602a5ea7facf` rejected the proposed
> native-complex escape route in the existing fixed ESPnet source. The new
> source facts below do not approve dependencies, change the lock, retarget the
> source or establish numerical parity. Earlier dated findings are retained.

Inspected from accepted main `46d2c9dd9d49b7dfc7fc7a3e928d54b76594785a`
after ordinary acceptance of documentation PR #233. This is a pre-import gate
correction and source-only investigation, not an execution or model approval.
The [58-row inventory](current-58-prerequisite-inventory-2026-10-14.md) and
frozen owner packet retain their denominators and decisions.

The correction was committed as `c86f72ab`. After PR #232's normal acceptance
at `fa42f112`, that main was normally merged into the candidate at
`b92665f4`; all OWSM input bytes and the correction are unchanged. Manager
again passed the actual self-test, eight focused tests, the Conv-TasNet
binding self-test, actionlint and diff hygiene. Rust/Cargo/C-ABI inputs remain
identical to recorded green source-only VAST `4263f416`. This is not a new
remote result at the candidate, nor current-head PR CI acceptance. No local
workspace compilation or model execution was used.

## Declaration is not approval

The former `dependency_license_gate.py` returned `AUDITED_ALLOW` solely from
the same scalar in `pyproject.toml`, after checking direct pins and CPU-index
rows. It did not authenticate an owner-reviewed dependency/native audit or
its binding to the current inputs. The dedicated VAST worker independently
requires a blocked preflight and stops at the known factual license failure;
this investigation does **not** claim that its outer guard was bypassed or
that any model was executed.

The correction keeps valid pending inputs at
`BLOCKED_UNREVIEWED_TRANSITIVE`. A scalar-only allow declaration now yields
`BLOCKED_UNBOUND_AUDIT_APPROVAL` with exit 2. Structural failures keep
`BLOCKED_LOCK_CONTRACT`, including when an unbound allow declaration is also
present. No accepted approval format, signer or grant is invented: a future
execution path still needs genuinely reviewed, hash-bound evidence.

The checked-in project and lock are unchanged:

| Input | SHA-256 |
|---|---|
| `tools/parity/owsm_v4_medium_1b_reference/pyproject.toml` | `249d608efe024bd3c831549894f5616a4ce265edd8772e238cf1abfe81ce1ba7` |
| `tools/parity/owsm_v4_medium_1b_reference/uv.lock` | `a7358dac9d5e753dd9c296f8895bec08ee8a98f0eb73dfb9940b3e0a59c2fe52` |

Luna and manager independently pass the actual gate self-test and eight
focused stdlib tests using UV/Python 3.12. They exercise the real 43-row input,
scalar flip and CLI exit, transitive mutation remaining blocked, CUDA wheel
row, non-CPU index, direct pin, mixed structural/approval failure, and import
boundaries. The tests are explicitly wired into the existing
`workflow-security` job; action pins, permissions and dependencies are unchanged.
Local test success is not current-head CI acceptance or license/CPU parity.

## Why the license hold remains an active prerequisite

The official repository's current `master` and the fixed v0.4.4 revision both
resolve to `8a2ad1e47f3df25a30eb426f6ad781b89103fab3`. Their complete 17-entry
tree still has no LICENSE/COPYING/NOTICE file, and the repository license
field remains null. The [official release JSON](https://pypi.org/pypi/torch-complex/0.4.4/json)
has null license, license-expression and license-file fields; its Apache
classifier is not substituted for primary license bytes. The
[earlier exact archive finding](public-catalog-security-completion-2026-09-29.md#2026-09-29-owsm-torch-complex-license-readback-1256-utc)
is retained, not claimed as a new archive download.

The fixed [ESPnet DefaultFrontend](https://raw.githubusercontent.com/espnet/espnet/cccc29023d43a3f504e28df7d1324bb4eb6daedd/espnet2/asr/frontend/default.py)
constructs `ComplexTensor` in `_compute_stft`, before power-spectrum and
log-mel processing. It is an active dependency, not merely an import for an
unused enhancement path. Replacing it with a local numerical mirror or an
inactive-import stub would not preserve the audited independent reference.
The newer official frontend at `4ed1bd2dcf5f553702488b5104aa6bc3a6ab85a0`
was also inspected and still constructs it. Merely switching to that master
snapshot therefore does not establish a dependency-free route or authorize
retargeting the fixed source contract.

Next requirement remains authoritative permission/license bytes for the exact
dependency, or a separately authenticated compatible official reference route,
followed by dependency/native/operator review. Writer, complete decoder/search,
real-weight independent CPU parity and the final Apple packet are still open.
No local model download/forward, package install, paid worker, upload,
withholding/withdrawal, owner decision or model-row promotion occurred here.

## Fixed frontend has no native-complex selection flag — 2026-10-15 JST

Luna retrieved the official raw source at exact ESPnet commit
`cccc29023d43a3f504e28df7d1324bb4eb6daedd`, confirmed HTTP 200 and computed
received-byte hashes. The manager independently inspected the fixed
`espnet2/asr/frontend/default.py` through the GitHub connector: blob
`1cceef269d5d3baba80cc87942163fbb58690eae`, 4,413 bytes. Its constructor has
no `use_builtin_complex` selection flag. Both the STFT path and the
`apply_stft=False` path construct `ComplexTensor`; disabling enhancement does
not disable that active dependency.

The [fixed STFT source](https://raw.githubusercontent.com/espnet/espnet/cccc29023d43a3f504e28df7d1324bb4eb6daedd/espnet2/layers/stft.py)
can internally request native complex output, but converts it back to real/imag
storage before the frontend reconstructs `ComplexTensor`. The downstream
`complex_utils.py` helper's support for `torch.complex` does not change the
frontend caller contract. Fixed `setup.py` also declares `torch_complex`.
Changing window/FFT/mel or rewriting this active computation as a local mirror
would need a new source/reference contract and independent parity; it is not
an equivalent dependency-removal receipt.

Received source identities (all paths relative to that fixed ESPnet commit):

| Path | Bytes | SHA-256 |
|---|---:|---|
| `espnet2/asr/frontend/default.py` | 4,413 | `10be842dc150e8a5b13dbbd8bf3a0ce9662984a1c1633b4c7ceb6e2155e9c456` |
| `espnet2/layers/stft.py` | 7,832 | `232acacbb4fe0e9a2d40da62b46885343962de1cba8a010b6eea7ab2a725e43a` |
| `espnet2/layers/log_mel.py` | 2,578 | `6bb193bc09aaef22a91b1a2bde8e8f8859cf561792d7aa84623601270fe0ad63` |
| `espnet2/enh/layers/complex_utils.py` | 6,602 | `130f492b009e4621ffe7eef80c14378371a646711424f5f21e16555455753466` |
| `espnet/nets/pytorch_backend/frontends/frontend.py` | 4,584 | `9cbfc0d603b2393aa51f292599437a1d37d6c9b6b6b57b94df91841b579de4f2` |
| `espnet2/asr/espnet_model.py` | 24,419 | `b914fbbb7c2ba4a2b79779d24e3f83cc1b5fa2601e427fcbc9ed94b68dbcbb2c` |
| `espnet2/enh/layers/dnn_wpe.py` | 5,512 | `95272e89b32bfee96baaa164647591af93dcc80caab34aeddcd31f0918316161` |
| `espnet2/enh/layers/dnn_beamformer.py` | 23,363 | `72c71ae3fcb23183bd0175e0ea2bab9295349d14b7a5f57591538bf3fc4f1c56` |
| `setup.py` | 5,270 | `11cff9b0d8ea4e232077b547503b3e9305f90826901883db2776b74b9453224d` |

Do not add a guessed flag, call the active dependency an unused import, or
repeat a paid model audit to resolve its rights. An alternative official route
would first need exact revision/caller/constructor authentication, source and
dependency review, and separate numerical evidence. No model/config/tokenizer
acquisition, target-package import, inference, heavy Cargo, paid worker,
approval, manifest/lock change or publication was used in the source investigation.
The later ordinary documentation pre-commit hook synchronized 146 cached
software distributions into a reproducible task-only virtual environment;
this was not a model import, model download or reference-environment approval.
That disposable environment is removed after the management commit.
