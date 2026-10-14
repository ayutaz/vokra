# XCodec2 exact-byte guard binding — 2026-10-15 JST

This source-only follow-up starts from Draft [PR #152](https://github.com/ayutaz/vokra/pull/152)
at `dcb24a0e3d979d31fe838356da1944f8f2087fc0`. Accepted main at inspection is
`731592f97251f52f0e80ea81d211d9dde6b52d71`, after PR #246's normal merge at
18:24:02 UTC on 2026-10-14. That acceptance had 70 completed checks
(69 successful, one intentional full-history secret-scan skip), all 16 required
context/application pairs successful and all nine started workflows terminal
success. This is not acceptance of #152 or new model/Apple evidence.

Reviewed source integration is committed as
`4256a784a7a40dbdac7bb7d2445d9af5472aa551`, with parents `dcb24a0e` and
`731592f9`. All five normal pre-commit gates pass without creating a parity
environment. The accompanying management commit retains the accepted-main
plan/index and adds this dated evidence; neither commit is model execution.

## Concrete guard defect and bounded correction

The direct [dependency guard](../../tools/parity/xcodec2/dependency_guard.py)
validated TOML semantics and helper source digests, but did not compare the
current project/lock bytes with the hashes in both pending gate documents.
Appending a newline to either input was accepted by the previous validator.
The stronger collector/dumper contract already required exact hashes; the
direct guard was inconsistent with that contract.

The correction reads the actual files as bytes, preserving CRLF rather than
normalizing it through text-mode input. It binds both project and lock to the
manifest and dependency-audit hashes, and binds the actual dependency-audit
file to the manifest. Existing helper-source checks and semantic checks remain.
Only the changed guard's pending source digests and their enclosing audit-file
digest are updated; project/lock declarations and all approval holds are
unchanged.

The [model-free regression](../../tools/parity/xcodec2/test_dependency_audit.py)
uses the real validator with small temporary documents and actual on-disk
documents-only entry points. It checks unchanged inputs, newline/CRLF changes,
missing or wrong bindings, document disagreement and helper-source drift.
Sentinels verify that documents-only and rejected-byte cases do not reach the
installed-environment or official-decoder paths. Synthetic fixtures are not
retained as real dependency, license or approval evidence.

## Historical oracle and remaining exact-consumer gate

The committed [fixture manifest](../../crates/vokra-models/tests/fixtures/xcodec2/manifest.json)
records Torch `2.5.0+cu124`, the official `xcodec2==0.1.5` decoder and the audited
GGUF SHA-256 `7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e`.
Its immutable numerical bytes are not invalid merely because the pending
dumper declares Torch 2.13.0 / TorchAudio 2.11.0. They describe a historical
source/environment/artifact scope, not a new run under that patched pair.

The current [real Rust consumer](../../crates/vokra-models/tests/parity_xcodec2_real.rs)
does not authenticate this manifest, its fixture-file hashes or the exact GGUF
SHA before comparison. That exact-consumer binding remains a separate source
task. Fresh independent execution under the current reviewed environment also
remains unrun. This correction changes no Rust consumer, fixture bytes, bounds,
owner decisions or numerical results.

## Verification and retained holds

Manager independently runs all three actual stdlib discovery suites
(24 dependency-audit, seven source-inspector and six derived-sdist tests;
37 total, zero failures), dependency-audit self-test and dependency-guard
`--self-test --documents-only` through offline no-project UV/Python 3.12.
Bare direct test-file invocation is refused before process creation by the
maintainer guard; genuine unittest discovery is permitted, with no hook edit,
dummy self-test argument or guard override. Integrated-head CI and formal PR
acceptance remain separate.

The full lock contains 67 records; the declared Linux closure is 63 rows
(one virtual project and 62 external distributions), not 67 installed Linux
packages. Both documents remain `BLOCKED_PENDING_PRIMARY_BYTES` / `NO_UPLOAD`.
Primary package licenses, native payloads, NumPy's GPL/GCC/LGPL components,
derived-setuptools compatibility and exact owner/legal review remain open.
The historical Research-only weight sign-off in `docs/license-audit.md` is not
a new dependency/execution approval. The full Draft #152 responsibility is
not reduced to this guard correction.

No model/config/tokenizer/checkpoint acquisition or execution, dependency
installation, collector/runtime API run, heavy Cargo, paid worker, Scaleway
or publication occurs. Public unresolved inventory remains 58 rows and the
frozen owner packet remains 63 rows. Authorized real-weight CPU verification,
final reviewed clean HEAD and fresh Apple packets are still required by the
[prioritized plan](next-steps-2026-10-14.md).
