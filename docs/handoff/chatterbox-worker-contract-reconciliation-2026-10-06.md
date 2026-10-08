# Chatterbox stopped-worker contract reconciliation — 2026-10-06

This dated record covers the small follow-up to [PR #197](https://github.com/ayutaz/vokra/pull/197),
reviewed from parent `0a11d8d9ef1a7836c751c1c1ca239a5aafb56db7`.
It does not approve the reference environment, source compatibility, model
execution, numerical parity, Apple hardware results or publication.

## Cause and correction

The committed Chatterbox reference and dedicated lock had already moved to
the patched candidate, but both VAST workers still expected an earlier lock,
package-row identity and Torch/TorchAudio 2.6 distributions. A current blocked
audit therefore failed as an obsolete identity instead of reporting the actual
unresolved license disposition.

The two workers now require the existing candidate identities:

- Lock SHA-256: `3c1a295bd6d45e6b83f7a182a4421bbb7cc5904a554f4305b9f7d08d1e92029d`.
- Canonical package rows SHA-256: `a47f8a74ef9d990289002eaccd346d7d5cbbc9d1480d1213596bc01b0ed36c24`.
- Candidate Torch `2.13.0` / TorchAudio `2.11.0`, and their corresponding
  `+cpu` distributions at the official CPU index.

Blocked-audit JSON is parsed structurally, rejecting duplicate keys and
stale lock, package-row, core-version, CPU-distribution, index or audit-lock
identities. An unexpectedly successful audit command does not unlock execution.
The reference source, project and lock were not changed in this follow-up.
The upstream source-declared 2.6 requirements remain facts, not proof that the
patched candidate is compatible. `BLOCKED_UNRESOLVED`, the unconditional
`BLOCKED_APPROVAL/INSPECTION_ONLY` stop, `NO_UPLOAD`, and the dormant
`AUTHENTICATED_CLEAR` requirement remain intact.

## Focused verification

The manager reviewed both worker diffs and the complete new
[dependency-free test](../../tools/parity/test_chatterbox_worker_contract.py),
then independently ran its explicit entrypoint:

```sh
UV_NO_SYNC=1 UV_OFFLINE=1 uv run --offline --no-project --no-sync --python 3.12 \
  python -B -S tools/parity/test_chatterbox_worker_contract.py --self-test
```

Result: the UV deny probe and all **24 causal blocked-preflight cases passed**.
The cases cover current blocked output, stale identities, duplicate and malformed
JSON, an unexpected clear exit, and an unexpected failure exit for both workers.
Every worker case returned exit 2 without creating its work directory.

The test stubs the exact audit command; it does not execute the actual upstream
reference or its dependency-capable self-test. Real UV dispatch is restricted to
the reviewed offline stdlib stdin validators with exact arguments and source
hashes. Unexpected commands and Cargo/network/provider commands are rejected.
The temporary root is canonicalized and portable; subprocesses are bounded.
An earlier indirect test invocation is not used as acceptance evidence: the
accepted result above came from the manager's direct explicit self-test run,
without a hook override.

Both shell syntax checks, whitespace hygiene and the first-party Cargo-lock
gate passed. Normal commit/push checks and fresh exact-child CI must be recorded
separately; parent CI and parent VAST evidence do not certify a changed child.
No maintainer-Mac model execution, dependency sync, heavy Cargo, cloud allocation,
artifact acquisition or upload occurred in this focused acceptance.

## Remaining gates

PR #197 still requires fresh required CI and stopped-preparation merge review.
Its seven source PRs may be closed only after the accepted replacement is in
main and their responsibilities are accounted for. Chatterbox's dependency/native
license review, exact owner decisions, upstream compatibility, independent
real-weight CPU parity and later Apple CPU/Metal/no-fallback remain separate
unresolved gates. This fix does not change the public model coverage count.
