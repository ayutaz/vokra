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
evidence directory to be absent and creates it exactly once below a regular,
non-symlink parent; it snapshots the bound source files before finalizing the
no-clobber checksum file.  The auditor never imports NeMo, accesses Hugging
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
