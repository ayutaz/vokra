# Canary-1B official reference closure

This is the only Python environment permitted for the NeMo Canary-1B-Flash
and Canary-1B-v2 reference workers.  It is deliberately separate from the
generic `tools/parity` project, whose optional extras include unrelated model
families and native codecs.

The oracle remains the official NVIDIA implementation:

```python
from nemo.collections.asr.models import EncDecMultiTaskModel
model = EncDecMultiTaskModel.restore_from(...)
```

The dumpers contain no frontend, decoder, tokenizer, or search mirror.  This
project is Linux x86_64 / Python 3.12 only.  Torch is resolved exclusively
from the pinned PyTorch CPU index.  The impossible `cuda-bindings` and
`text-unidecode` overrides keep this CPU-only closure from silently installing
CUDA host bindings or the GPL text-unidecode package; their presence is
recorded by the audit and remains pending owner review.

## Commands

All commands run on VAST and use the committed lock; they do not acquire a
checkpoint in the closure-audit mode:

```bash
uv sync --project tools/parity/canary_1b_reference --frozen --python 3.12
uv run --project tools/parity/canary_1b_reference --frozen --python 3.12 \
  python tools/parity/canary_1b_flash_dump_reference.py --self-test
uv run --project tools/parity/canary_1b_reference --frozen --python 3.12 \
  python tools/parity/canary_1b_v2_dump_reference.py --self-test
```

The model-free audit must run after the frozen environment is synchronized:

```bash
uv run --project tools/parity/canary_1b_reference --frozen --python 3.12 \
  python tools/parity/canary_1b_reference/dependency_audit.py \
  --project tools/parity/canary_1b_reference/pyproject.toml \
  --lock tools/parity/canary_1b_reference/uv.lock \
  --repo-root /workspace/vokra --expected-head <40-hex> \
  --archive-dir /workspace/canary-audit/licenses \
  --output /workspace/canary-audit/dependency-audit.json \
  --project-sha256 <64-hex> --lock-sha256 <64-hex> --audit-sha256 <64-hex>
```

The report is factual evidence, not a publication or legal approval.  It is
always `BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD`; native payloads, ELF
`NEEDED` entries, publisher license metadata, and license-file bytes copied to
the no-clobber license archive remain owner-review rows.  `SHA256SUMS` binds
the project, lock, collector, executed wrapper, report, completed audit log,
and every archived publisher file.  The wrapper requires the requested
evidence directory to be an absent/new path (an existing empty directory is
rejected) and creates it exactly once below a regular, non-symlink parent; it
snapshots the bound source files before finalizing the no-clobber checksum
file.  The auditor never imports NeMo, accesses Hugging
Face, reads a checkpoint, invokes Cargo, or uploads anything.

## Closure review (lock/import evidence)

The two dumpers statically bind the independent oracle to
`nemo.collections.asr.models.EncDecMultiTaskModel.restore_from`; neither
dumper mirrors NeMo internals.  The committed lock's Linux x86_64 active
closure is 134 rows (133 installed distributions).  The audit report records
at least one witnessed root-to-package dependency path for every active lock
row, including selected extras.  It intentionally does not claim exhaustive
enumeration when a package has multiple parents.  The principal
native/license blockers are reached as follows:

```text
project -> nemo-toolkit[asr] -> librosa[selected-by=nemo-toolkit[asr];extra=asr] -> soxr
project -> nemo-toolkit[asr] -> librosa[selected-by=nemo-toolkit[asr];extra=asr] -> scipy
project -> nemo-toolkit[asr] -> scipy[selected-by=nemo-toolkit[asr];extra=asr]
project -> nemo-toolkit[asr] -> pandas[selected-by=nemo-toolkit[asr];extra=asr] -> numpy
project -> nemo-toolkit[asr] -> datasets[selected-by=nemo-toolkit[asr];extra=asr] -> pandas -> numpy
project -> nemo-toolkit[asr] -> librosa[selected-by=nemo-toolkit[asr];extra=asr] -> numba -> llvmlite
project -> nemo-toolkit[asr] -> datasets[selected-by=nemo-toolkit[asr];extra=asr] -> pyarrow
```

`numpy`, `scipy`, `pandas`, `pyarrow`, `numba`/`llvmlite`, and `soxr` are
therefore not generic-project leftovers: each is reached by the selected
official NeMo ASR extra (often through more than one path).  The native audit
records their ELF/archive bytes and `NEEDED` entries; publisher license bytes
are archived separately.  Static evidence does not authorize a reduction:
whether a smaller set can still import and execute the official
`EncDecMultiTaskModel.restore_from` path remains unproven until a VAST import
probe.  Replacing that oracle with a lighter implementation would change the
independent semantics under test.  The status consequently remains
fail-closed until owner/legal review resolves the observed
GPL/LGPL/unknown/native rows.

The direct `hydra-core==1.3.6` requirement is intentional. NeMo 3.0.0's
published extra metadata leaves `hydra-core` unconstrained; the old lock had
selected 1.3.2. The dedicated uv project now directly pins a current patched
release (1.3.6; the first patched version is 1.3.4) and
records the same resolution in
`tool.uv.override-dependencies`. This is a dependency decision for the
owner's review, not an advisory allowlist.

The same override mechanism handles the reviewed Lightning security advisory
`GHSA-qqmf-gpg7-g8gw` (CVE-2026-58659). NeMo 3.0.0 declares
`lightning<=2.4.0`, but that upstream cap would select a vulnerable release;
the lock therefore replaces only that transitive package with `lightning`
2.6.6, the first released version containing the upstream `_instantiator`
allowlist fix. The official NeMo oracle and its source contract remain
unchanged. This resolver override must still be covered by the authorized
VAST import probe before a real checkpoint is inspected.

### Security-compatible import closure (currently blocked)

The authorized VAST import probe found that the locked pair
`lightning==2.6.6` / `nv-one-logger-pytorch-lightning-integration==2.3.1`
cannot be imported: the released NVIDIA OneLogger trainer override has a
`save_checkpoint(weights_only: bool)` signature while Lightning 2.6.6
requires `Optional[bool]`, and the upstream `overrides` check rejects the
class at import time. NVIDIA has no newer released OneLogger PTL integration
to resolve this mismatch. The real-weight workers therefore run the
dependency gate's `--compatibility-check` mode and stop with
`BLOCKED_SECURITY_INCOMPATIBLE_CANARY_CLOSURE` before checkpoint inspection.
No older vulnerable Lightning release may be permitted by an advisory
allowlist, and no local monkeypatch is permitted. The CI dependency-review job
has one temporary,
exact exception for this database false positive: PyPA's primary advisory
record `PYSEC-2026-3624` marks `lightning` 2.6.6 as fixed and only enumerates
versions through 2.6.5, while GitHub currently applies the corresponding GHSA
to 2.6.6 as well. `scripts/check-canary-dependency-review-guard.sh` runs before
dependency-review and fails closed unless the tracked Canary lock contains
exactly one `lightning==2.6.6`, the exact pyproject override and compatibility
block are present, and the exception occurs exactly once. This exception must
never be copied to another lock or used to permit a vulnerable Lightning
release. A future upstream-compatible pair requires a new VAST import probe
before this gate can be relaxed.

## Dependency approval transition

`dependency_audit.py` always emits factual
`BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD` evidence. It never creates an
approval. Before either real-weight worker inspects a checkpoint, an owner
must supply all of the following externally to the checkout:

```text
--dependency-approval <owner-signed-record.json>
--dependency-approval-sha256 <sha256-of-record>
--dependency-signer-key <trusted-owner-ed25519-public-key.pub>
```

The record schema is `vokra-canary-1b-dependency-approval-v1`. Its exact keys
bind the variant, clean HEAD, audit report SHA-256,
`candidate_owner_scope_sha256`, `BLOCKED_UNREVIEWED_TRANSITIVE`,
`NO_UPLOAD`, `no_upload: true`, and `decision: APPROVED`. The record also
includes a non-placeholder signer, ISO date, `signature_algorithm: ssh-ed25519-v1`,
the trusted-key SHA-256, and a detached OpenSSH Ed25519 signature
over the canonical record without `signature_base64`. The worker verifies
the signature and all path boundaries before archive inspection, scratch
creation, model work, or Cargo. Existing model-license approval is a separate
gate; it cannot substitute for this dependency approval.
