# FireRedASR-AED-L dependency audit

The dedicated Linux/x86_64 `uv.lock` is the immutable dependency authority.
The VAST worker collects the lock hash and each lock source identity, then
records installed distribution `METADATA` hashes (and the lock's virtual local
project `pyproject.toml` metadata under its actual project name), declared
publisher/project URLs, license candidates, and native payload hashes. Missing
or mismatched installed rows are collection failures, not empty evidence. The resulting
`dependency-audit.json` contains a 27-row `review_ledger`; its exact SHA-256
scope binds the lock, source identities, distribution evidence, license and
native payload aggregates, and the ledger itself.
The `kaldi-native-fbank` row additionally carries the checked-out source URL,
exact revision, `LICENSE` path/byte count/SHA-256, and independent verification
flags inside `license_candidates`; this source-license record is included in
the per-row and aggregate license digests. The auditor receives the lock
project directory separately from that native source checkout; if the native
source path is absent, the source-license verification remains blocked.

This evidence is not an approval. Without a separately supplied approval the
worker does not accept an owner decision and remains
`BLOCKED_UNREVIEWED_TRANSITIVE` before model acquisition. In approved mode the
worker only validates a separate owner artifact; it never creates or signs an
owner decision. That artifact must use
format `vokra-firered-asr-aed-l-owner-approval-v1`, owner handle exactly
`yousan`, a strict RFC3339 UTC `approved_at_utc` (not more than five minutes in
the future), decision `APPROVE`, the exact `exact_digest_gate.scope` plus its
`scope_sha256`, and one row for every active closure entry. Every row must independently set
`license_review`, `publisher_review`, and `native_payload_review` to
`APPROVE`, and must repeat the exact row/source and three evidence digests; a
common license or an absent native payload is not an approval.

## Model-free pre-approval collection

The owner-independent route collects only the frozen `uv.lock` graph and the
fixed FireRed source/model-card/config/training-provenance identity scope. It
does not inspect installed packages, fetch an upstream checkout, import a
runtime, acquire a checkpoint, run inference, or upload anything:

```bash
VOKRA_PUBLISH_ON_VAST=1 bash scripts/publish/vast-ai/run-firered-asr-aed-l-inspection.sh \
  --model-free --expected-head <40-lowercase-hex> \
  --work-dir /dev/shm/vokra-firered-asr-aed-l-model-free
```

The supplied exact HEAD is stored in both the manifest and its canonical
approval scope. The command intentionally exits with status 2 and writes
`evidence/model-free-audit.json` with `BLOCKED_OWNER_REVIEW`,
`PENDING_OWNER_REVIEW`, and `NO_UPLOAD` markers. A model-free scope is
evidence for later review, not a license, training-data, model, conversion, or
parity approval. The normal owner-approval route retains its existing
gate-first fail-closed boundary.

## VAST collection

The later owner-approved inspection route runs on a clean, provisioned
Linux/x86_64 VAST host:

```bash
VOKRA_PUBLISH_ON_VAST=1 bash scripts/publish/vast-ai/run-firered-asr-aed-l-inspection.sh \
  --expected-head <40-lowercase-hex> \
  --approval-sha256 <64-lowercase-hex> \
  --owner-approval /absolute/path/blocked-approval.json \
  --work-dir /dev/shm/vokra-firered-asr-aed-l-inspection
```

`--work-dir` must be an absolute, absent, non-symlink path; the worker creates
it only after canonical-parent and checkout-overlap checks. A fresh work
directory is required even when a previous directory is empty. An approval
file must be an existing regular JSON file outside both the checkout and this
fresh work directory.

The model snapshot remains after the dependency audit gate. An unapproved
first pass produces only `evidence/dependency-audit.json` and
`evidence/validation.log`; the worker exits with the intentional blocked
status before requesting the model snapshot. `manifest.json` and
`server_tree.json` belong to the later approved route and are not expected
artifacts of this blocked first pass. Do not edit the JSON to add an
approval: regenerate it from the same lock/source checkout and have the owner
review the exact scope and all 27 rows.

After a separately supplied approval file has been reviewed and placed
outside both the checkout and the work directory, the same worker may be
invoked with `--owner-approval /absolute/path/approval.json`. The auditor must
return `OWNER_APPROVED`, `owner_approval.status=VALIDATED`, no collection
failures, and matching artifact SHA-256 before the snapshot step is reachable.
An invalid, symlinked, stale, or partially reviewed file remains blocked and
is never rewritten by the worker.
