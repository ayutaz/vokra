# OWSM dependency admission readback — 2026-10-14

Inspected from accepted main `46d2c9dd9d49b7dfc7fc7a3e928d54b76594785a`
after ordinary acceptance of documentation PR #233. This is a pre-import gate
correction and source-only investigation, not an execution or model approval.
The [58-row inventory](current-58-prerequisite-inventory-2026-10-14.md) and
frozen owner packet retain their denominators and decisions.

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
