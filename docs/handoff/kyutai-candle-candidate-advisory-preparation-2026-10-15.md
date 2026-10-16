# Candle candidate advisory preparation — 2026-10-15

> **Actual-run supersession:** the [recovered result](kyutai-candle-candidate-advisory-result-2026-10-15.md)
> now records `FAIL_FINDINGS` / rc 1: zero vulnerabilities and the remaining
> paste unmaintained warning across all 154 records at the fixed database.
> The preparation-time pending statements below are historical. Do not
> repeat unchanged execution or infer a clean audit/model verdict.

The [corrected resolution](kyutai-candle-pin-remediation-2026-10-15.md#corrected-actual-resolution-and-independent-acceptance)
succeeds at producer `27a2841384220c078aa37ff93ea8f9fc8a18594f`, run
`38711006458` attempt 1, artifact `11961100426`. Its exact 38,252-byte
[candidate lock](kyutai-candle-candidate-2026-10-15.lock) has SHA-256
`d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8`.
This document prepares a new strict advisory run of all 154 candidate
records; it is not that run's outcome. The old 161-record/five-finding
report remains unchanged and cannot serve as the new lock's verdict.

## Two fixed contracts, one unchanged audit lifecycle

The reviewed wrapper registers only `baseline` and `candidate`. The default
baseline keeps its original seven members, 4 MiB extracted cap, producer,
161-record report count and scope. The old workflow, twelve-test module and
raw finding receipt are byte-unchanged. The shared bounded process/API,
auditor/version, database provenance, report classification and evidence
writing remain one pipeline, not two diverging implementations.

The candidate profile binds the exact producer/head/artifact above,
457,074-byte compressed artifact, both actual member prefixes and all
thirteen per-member lengths/hashes recorded in the resolution readback.
Its separate extracted-byte cap is **exactly 7,040,218**, not a raised
baseline limit or an arbitrary caller setting. Every member still has the
existing 2 MiB bound. Custom profile objects and caller-supplied scope
counts/identities are rejected; the CLI exposes two registered names only.

Artifact validation checks exact paths, ordinary single-link files, sizes
and hashes, then the 154 unique TOML identities and complete unfiltered
whole-metadata identity/node universe, Moshi root, typed edges and explicit
source-only authorization boundaries. CPU metadata is evidence, not an
alternative audit universe. The audit uses the authenticated `candidate.lock`,
never the old patched lock or a 105-package subset.

Tool and database pins remain cargo-audit **0.22.2**, release commit
`281452c35cf0870969042374110f099a411bc185`, installer action
`183e4297cca2404691e9380e1307288dced5c82a`, RustSec commit
`edb58f5974b83eb8302332821cb6d389777aa351` and tree
`9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63`. The same strict command retains
`--no-fetch --no-yanked --quiet --json --deny warnings`. There is no ignore,
target/severity filter, build fallback or warning waiver. Native JSON keeps
its explicit null database commit; external before/after Git provenance
authenticates the clean pinned database instead of rewriting that field.

## Independent preparation checks

Manager reviews the complete diff and independently passes all twelve legacy
tests in **4.107 seconds** and all nine candidate tests in **0.076 seconds**,
including the genuine thirteen-file artifact, with no skips in these local
replays. Portable cases cover the registered profile, report-count/filter,
whole identity/edge/root/authorization mutations and candidate pipeline
PASS/FAIL_FINDINGS/ERROR_RC0_FINDINGS behavior. Actual artifact mutation
tests are distinct from those portable negative fixtures.

Before publishing, manager catches an omitted old-workflow file required by
the legacy test. The corrected sparse checkout includes it. Early hosted
unit tests explicitly skip unavailable recovered-artifact cases; the workflow
then downloads the fixed artifact and reruns the candidate suite with its
actual path so those cases execute before the auditor. This does not borrow
the Mac's private evidence as a hosted pass. Scoped actionlint, workflow
hygiene and diff checks are separate preparation gates.

The direct local multi-command test invocation and a separate py_compile
attempt are refused by the model-safety hook before those processes start.
No hook is changed or disabled. The supported stdlib unittest discovery
interface and legacy `--self-test` execute the reviewed model-free tests;
their imports parse the changed code. No local production auditor or model
is run and py_compile is not reported as passing.

Reviewed helper/test/workflow SHA-256 values:

- `90358b546d402d2343190333c6187d54c2e35b46472da610b0730d1e57cb3052`
- `3d999b92bf70e4ebbe5ddb37cd4134da3e7d29d7c92d08b57f3d7f548e28ff7d`
- `5feb68edb2902826b5c05e0865075d79a240847a552c8fd9b0a15f90cbaf0646`

## Remaining boundary

Run the new exact-head hosted workflow, recover and independently validate
its native report/return code and unchanged database provenance. Findings
must keep a nonzero exit. `paste 1.0.15` is still present, so do not infer a
PASS from the three precise updates or any generic CI success. Its genuine
maintained replacement needs all affected parent manifests and independent
macro/API evidence; the [pastey source receipt](kyutai-candle-pastey-source-readback-2026-10-15.md)
alone is not compatibility approval.

PR #255 stays draft. License/native/operator, independent full-PCM reference,
real-weight CPU and final clean-HEAD/fresh Apple packets remain unfinished.
The full original step-4 objective is active, not narrowed to this audit.
No new paid worker, model/config/tokenizer acquisition or local execution,
owner signature, inventory promotion or public artifact update occurs here.
