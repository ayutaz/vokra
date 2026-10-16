# Six-stage execution readback — 2026-10-16

## Scope and current evidence

The active objective remains all six stages below, not merely the Candle
library build or PR acceptance. Inspection starts from published PR #255
`16a172e01ab540d42c20e7348ea70d7ca99d6821`, base main `f8d859dc`.
The separate maintainer checkout is clean on
`codex/fix-candle-readonly-guard-20261016` at `62e421e4`; it is not
represented as main. The reviewed guard repair is preserved.

The public metadata-only inventory command completes with rc 0:

```sh
env UV_CACHE_DIR=/private/tmp/vokra-uv-check UV_OFFLINE=1 \
  uv run --no-project --offline --python 3.12 python \
  tools/audit/hf_mac_coverage.py --format tsv --workers 4
```

Its recovered [complete 194-row TSV](public-model-metadata-2026-10-16.tsv)
has SHA-256
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`,
identical to the dated 10/14 inventory. All repository revisions and
classifications in that deterministic TSV are unchanged, not just its counts:
194 repositories, 193 GGUF-bearing, 198 files; 136 code/artifact-full,
43 partial, 14 no-runtime-binder and one not-artifact. The 58 non-full rows
remain in the [row-specific prerequisite inventory](current-58-prerequisite-inventory-2026-10-14.md).
The command fetches public API metadata and immutable README text only;
no model/config/tokenizer/weight is downloaded or executed.

These classifications are not current execution approval, real-weight CPU
parity or Apple evidence. Code-full rows also need exact current validation.
The protected CosyVoice2 LLM tree is not read, hashed or materialized. Its
execution readiness remains unknown/excluded; its public card metadata is
not a substitute. The frozen 63-row owner packet is not rewritten.

## Required stages and acceptance evidence

| Stage | Next work | Evidence required to close it |
|---|---|---|
| 1. Corrected model-free CPU/API check | Review the host proc-macro path fix and run one separately authorized confirmation job | Exact-head full library and uncalled API probe rc 0, all Linux tests, original bounded streams, unchanged source/lock/105-node graph |
| 2. CI and PR #255 | Review historical-vs-current audit routing and causal timeout regression; publish reviewed changes | Fresh required context/application pairs and all applicable started jobs pass; normal merge and synchronized main |
| 3. All non-Apple prerequisites | Reconcile every row's exact source, dependency/native/license, implementation and independent reference; obtain missing hash-bound owner/legal decisions | Each row's factual gates close, or an explicit exact owner-approved disposition accounts for it; no inferred signatures or silent omissions |
| 4. Real-weight VAST validation | Convert approved exact weights, run independent upstream reference and CPU parity | Exact clean HEAD, actual numerical evidence, small evidence recovery, owned worker and storage destruction |
| 5. Final packet and Apple hardware | Freeze reviewed clean HEAD, regenerate/hash packets, then use Scaleway last | Apple CPU/reference, Metal/reference and Metal/CPU parity with explicit no-fallback, per exact model/packet |
| 6. Results and completion audit | Record actual outcomes in follow-up PR/docs and verify cleanup | All rows accounted for, evidence linked, fresh CI accepted, unnecessary instance/storage absent; publication conditions separately recorded |

All six stages are incomplete at this readback. A technical gate for one
family cannot close the campaign. Scaleway cannot repair missing source,
licenses, runtime implementation or independent reference.

## Immediate Candle decision

The [first actual job](kyutai-candle-cpu-compatibility-result-2026-10-16.md)
remains failed: full library rc 0, API harness rc 1 because its rustc search
path omitted already-built host proc macros. Its original receipt is not
rewritten. The corrected checker SHA-256 remains
`e86ccaa0d6763cf21a0c8cb0dfe38e86450a5470f23812cc72fd6b321a591183`.
Manager's fresh fixture-backed offline run enumerates 17 tests in 0.476s:
16 pass and one Linux-only orphan test is skipped on macOS.

The owner authorizes one additional actual GitHub Linux confirmation job,
maximum 20 minutes, with a fresh
`candle-cpu-compatibility-confirm-once-20261016` label. The consumed
`candle-cpu-compatibility-once-20261016` label is not removed/reapplied.
No activation has occurred for the new scope at this preparation snapshot.
First-attempt, exact PR #255/same-repository/read-permission guards remain.
This does not authorize model execution, paid workers or publication.

Fresh GitHub API observations report only the original failed compatibility
run `38838522884`. Original source artifact `11985400172` remains
unexpired, expiring 2026-10-16 15:16:14 UTC with digest
`sha256:dca3ea05a0970802e718aef113dd240aa75ff3fdcb21bb35b2c8fd941a69005e`.
At published `16a172e0`, #255 is open/draft with 73 successful checks,
two failed historical-source/candidate checks and one intentional skip.
The strict current applied audit is successful. Failed historical reports
stay failed; candidate event-routing changes are under review, not accepted.

The audit test's initial failure is retained: descendant readiness was not
reached before a 0.4s deadline in
`test_return_codes_json_retention_caps_and_timeout_cleanup`. A subsequent
green run alone is not acceptance. A deterministic readiness-bound deadline
regression is required before publishing the CI repair.

### Reviewed CI repair and local evidence

Manager reviews the final Luna diff: historical source/candidate event
routing becomes manual-only, while current applied remains automatic and
also triggers on changes to either historical workflow. All producer IDs,
artifact bindings, fixed DB/auditor and strict command/returncode policy are
unchanged. No production auditor code is changed. Shared regression tests
assert this distinction and reject warning ignores or continue-on-error.

The descendant test now has a 30s bounded startup budget and injects one
clock cutoff only after actual fork/heartbeat/marker readiness. It asserts
that the injection was consumed, that timeout occurred and that descendant
heartbeat stops after cleanup. This is not merely a longer timeout.
An intermediate fixed-clock candidate hangs during cleanup and is interrupted
at session `94684`, terminal rc 130; that rejected candidate is not the
reviewed patch. Its cleanup was not independently proven by a process
inventory. The synthetic fixture's temporary directory is removed on normal
exception unwind, so subsequent heartbeat writes fail, but that mechanism
is not represented as a provider or process-inventory audit.

Manager's final offline auditor suite passes 12/12 with the causal assertion
in 5.267 seconds, zero failures/skips.
Actionlint for all four changed workflows passes. Workflow hygiene passes
for 56 workflows / 36 crons; an initial invocation failed to access the
default UV cache and the successful check uses the explicitly writable
offline cache. All eight compliance regressions and the full Codex hook
contract pass. Documentation examples, references, community docs,
owner-checklist, platform and ABI accounting gates pass. Runtime Rust/Cargo/
tests/toolchain/config inputs are still equal to green VAST `4263f416`;
that old run does not verify these new Python/workflow edits.

The artifact-backed `test_applied_audit_lock.py` local invocation is refused
by the safety hook before execution. It is not retried or delegated through
another local route; existing automatic applied CI must run it, including
the recovered-artifact leg. A quoted-pipe source search of `audit_lock.py`
is also refused; no alternate body read or hook change follows.

No VAST/Scaleway resource is created by this metadata/CI phase. This is not
a fresh provider-wide assertion that unrelated accounts have no resources.
Local model execution remains prohibited; credentials are not read.
