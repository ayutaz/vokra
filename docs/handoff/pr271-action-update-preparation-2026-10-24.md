# Production Actions update preparation — 2026-10-24

This is a **candidate-only** checkpoint for [PR #271](https://github.com/ayutaz/vokra/pull/271),
not accepted-main or hosted-integration evidence. Its published head remains
`d777dd718c48712c8d1c557a7dc7d49d2ce1fcbc` at this checkpoint.
The order remains: accept [#278](https://github.com/ayutaz/vokra/pull/278), reconcile
the real accepted main, publish the reviewed Actions/pin-contract change into
#271, then require fresh protected CI and genuine producer/consumer evidence.

## Frozen workflow candidate

The [workflow manifest](evidence/pr271-workflow-candidate-2026-10-24.json) has
SHA-256 `1ea72b089fda7f3234191187565a149c5b1a2a546a6f4505007d76ea3917f4a3`.
It binds 55 workflow files to main `23a6d31397331ed36ed109f2f4863c64858c57be`,
with five Candle definitions taken from #278 source
`a4a9c5b09baba4951b3b702066e47e966850abc3`.

All seven upstream repositories' exact release tags were resolved to commits;
the CodeQL init/analyze/upload-sarif uses share one commit, making nine updates.
The complete candidate equals those canonical source bytes after only the
approved action SHA replacements and version comments on their `uses` lines.
The manager independently checked the 50 main-derived workflows and the five
authenticated GitHub overlay blobs, reviewed the four remaining comment/blank-line
corrections, and checked every final file hash against the manifest. Actionlint
with the accepted `.github/actionlint.yaml` configuration passed independently.

The existing `.github/pins.yaml` stays byte-identical, SHA-256
`73a0c5683f0ea58147007deb4936b1a2cd42f495a95c46232ac0d01fa971381c`.
That catalog deliberately excludes third-party Actions; no catalog exemption,
permission, trigger, artifact identity check or historical receipt is changed.

## Separate Python contract and combined regression

Luna prepared the three required source-line updates separately. The manager
reviewed the complete diff and the official Install/Download Action tag identities.

| Candidate file | SHA-256 |
|---|---|
| `audit_lock.py` | `7cb99faf3fbe24fddf40432f76db205fa3b25bb5f85a7a4710f83b7a80f6e478` |
| `test_cpu_compatibility.py` | `02444b58f0a120f1b3146b8a09c4f756569701f28aa53c581e2dcb90a05c519f` |

The first file updates `INSTALL_ACTION_COMMIT` and `DOWNLOAD_ACTION_COMMIT`;
the second updates its existing exact Download Action assertion. The reviewed
#278 correction that binds extraction to the validated envelope total remains.
Old receipts keep their original installer SHA; new runs must report the updated
installer. No new guard or weaker assertion is introduced.

The isolated combined Candle projection contains the updated workflow inputs
and seven #278 Python modules, with these two source candidates overlaid.
The four existing UV/Python 3.12 stdlib suites (`test_audit_lock`,
`test_cpu_compatibility`, `test_current_artifact`, `test_applied_audit_lock`) ran
**53 tests: 37 executed passes, 16 fixture-dependent skips**, with no failures.
The manager independently repeated the same suites successfully. These prove
the scoped source/workflow contracts, not a genuine downloaded artifact or model.
This resolves the workflow manifest's separate local Python-review/test condition;
its accepted-main and hosted-integration conditions remain open.

## Remaining acceptance and limits

The full PyYAML-based pins checker was **not run locally**: the existing
no-project Python environment lacks PyYAML, and no dependency was installed.
Its hosted required check must pass on the eventual published candidate.
The current directory is a projection, not a complete checkout; successful
actionlint and the model-free suites are not whole-repository execution evidence.

The temporary workflow, Python and combined candidates are local-only. The
manifest records their checkpoint identities but does not make those directories
available in a public clone. If accepted main or an input changes, reconcile the
actual bytes and source identities before publication and fresh CI.

At 2026-10-24 04:37 UTC, server repair [#298](https://github.com/ayutaz/vokra/pull/298)
had succeeded on all three server-compat targets but still had a queued required
parity job. #278's new same-run producer was also queued. These are dated waits,
not merge or integration-success claims. The earlier #278 extracted-size failure
and #271 artifact-not-found failure stay distinct.

No local model download/inference, package installation, Cargo compile/test/audit,
VAST/Scaleway allocation, artifact upload or new license/owner signature occurred
in this preparation. The unrelated dirty local-model guard is preserved, and the
protected CosyVoice2 manifest is outside the inputs and changes. The
[CPU-preview announcement draft](cpu-preview-announcement-draft-2026-10-24.md)
remains a separate deliverable; this preparation does not complete all model or
Apple CPU/Metal/no-fallback obligations.
