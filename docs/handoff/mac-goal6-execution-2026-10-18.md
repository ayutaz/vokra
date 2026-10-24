# Mac CPU / Metal six-stage execution readback — 2026-10-18

> **2026-10-23 supersession:** Use the [current status](../current-status.md)
> for present-day main, release and PR routing. This readback preserves the
> exact stage ordering, receipts, holds and no-promotion boundaries below.

> **Current PR acceptance and remaining plan:** [#268 is normally merged](#pr-268-accepted-after-complete-fresh-ci)
> at main `8ae9566e`. Complete fresh CI closes only the synthesized-session
> source batch. The [remaining order](#remaining-six-stage-order-after-pr-268)
> preserves five Draft PR responsibilities, formal source-read holds,
> independent reference, dependency/license/owner, real-weight CPU, final
> packet and Scaleway-last Apple requirements for all 194 rows. Earlier
> failed, pending and unrun statements below remain dated evidence.

> **Latest fixed-source acceptance:** [all four stages pass](#actual-session-fixed-source-four-stage-pass)
> at `59a0afda`, with the original receipt and six native streams verified.
> Owned `56503887` and data are destroyed and independently absent. This
> supersedes revised-source NOT_RUN/failed statements below only for this
> exact batch. Fresh PR CI, independent full PCM, real CPU and Apple remain.

> **Latest actual-session checkpoint:** [32 tests pass, strict lint remains](#actual-session-tests-pass-and-signature-lint-correction)
> at `155be525`. Original streams are retained; owned `56502593` and data
> are destroyed and independently absent. The further signature-only fix
> still needs an actual exact-source run before push. No model row advances.

> **Latest actual-session verification:** the [first runtime-observer run](#actual-session-compiler-failure-and-minimal-correction)
> fails at exact `7f2c0bc9` on two signature-probe lifetime errors, not a
> numerical assertion. Original native evidence is retained and owned worker
> `56501022` is independently absent. A test-only correction is reviewed,
> not yet remotely verified; no model or Apple row advances.

> **Latest PR acceptance:** [#267 complete fresh CI and normal merge](#pr-267-accepted-after-complete-fresh-ci)
> supersede source-PR-pending statements below at main `5b688666`. The next
> actual-session observer wave is unverified implementation, not independent
> full-PCM/real-weight/Apple evidence. All six stages and 194 rows remain.

> **Latest source checkpoint:** [corrected-source four-stage PASS](#corrected-kyutai-source-four-stage-pass)
> supersedes the pending compiler statements below, without rewriting their
> original failures. Main is `39dd2b00`; source verification is exact
> `c69395d5`, not a real-weight CPU or Apple result. Source PR acceptance and
> all unfinished parts of the six-stage objective remain open.

## Objective and baseline

The resumed user objective is to execute all six stages: existing PR
reconciliation; fresh model inventory; non-Apple source/reference closure;
approved real-weight VAST CPU validation; final clean-HEAD packet and
Scaleway Apple CPU/Metal/no-fallback; then result PRs, authorized public
reconciliation and cleanup. Completing an inventory or a maintenance PR
does not substitute for that full objective.

Execution starts from clean main
`7d910a9c0e2db2ffd544033e4ccb5792fd217ed6`. Work branch is
`codex/mac-goal6-execution-20261018`. The already published
[`v0.4.0-rc.1`](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)
remains the separate CPU-only preview at source `d84d311e`, with 22 assets.
Its representative Linux Whisper/Silero functional result does not complete
the all-model campaign or establish Apple model inference/Metal parity.

## Fresh metadata-only inventory

At this baseline, the existing `tools/audit/hf_mac_coverage.py` was inspected
and then executed through offline UV / Python 3.12 with four HTTP workers.
Its network path requests public model-list JSON and fixed-revision README
text only. It reads the CLI engine as text; it does not download model
weights/config/tokenizers, import a reference package or execute a model.

The new 37,881-byte TSV is byte-identical to the retained
[2026-10-16 inventory](public-model-metadata-2026-10-16.tsv), SHA-256
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`.
The complete retained bytes can therefore be reused without duplicating the
same snapshot payload. This is a new retrieval observation, not a rewrite
of that dated file's original provenance.

| Metadata/code classification | Current observation |
| --- | ---: |
| Public repositories | 194 |
| GGUF-bearing repositories | 193 |
| GGUF files | 198 |
| CPU code/artifact `full` | 136 |
| CPU `partial` | 43 |
| CPU `no-runtime-binder` | 14 |
| CPU `not-artifact` | 1 |
| Metal code/artifact `full` | 136 |
| Metal `blocked-by-cpu` | 57 |
| Metal `not-artifact` | 1 |

The 58 non-full rows retain their [row-specific prerequisites](current-58-prerequisite-inventory-2026-10-14.md).
The 136 full labels are code/artifact classifications, not 136 current
real-weight CPU or Apple passes. Every applicable public row still needs
exact source/artifact/reference/hardware evidence or an explicit exact
owner disposition. The frozen 63-row owner packet is a separate historical
decision scope and is not replaced by the live denominator.

## Existing PR queue: six open, not six merge-ready

Fresh GitHub API reads confirm these exact heads. The retired #259 is
closed without merge; its old pending compiler request is not automatically
reactivated or relabeled as accepted.

| PR | Exact head prefix | Current state | Next condition |
| --- | --- | --- | --- |
| [#258 Canary urllib3](https://github.com/ayutaz/vokra/pull/258) | `b2846466` | non-Draft, BEHIND, mergeable | Lock/approval/reference-hash impact review, latest-main reconciliation, fresh CI, normal merge judgment |
| [#198 legal records](https://github.com/ayutaz/vokra/pull/198) | `5614a92c` | Draft, BEHIND, mergeable | Authoritative source/version reconciliation and successful documentation-links; no legal sign-off inferred |
| [#191 Kyutai](https://github.com/ayutaz/vokra/pull/191) | `63cadf3c` | Draft, DIRTY | Resolve source conflicts without reverting newer main, close independent PCM/reference and dependency gates, then actual validation |
| [#182 VibeVoice](https://github.com/ayutaz/vokra/pull/182) | `a3afc1ec` | Draft, DIRTY | Current streaming/runtime/dependency/native/license/owner scope and actual real-weight evidence |
| [#169 VibeVoice Torch audit](https://github.com/ayutaz/vokra/pull/169) | `b3e9d5c1` | Draft, DIRTY | Reconcile its overlap with #182 and prove the stronger collector/security responsibilities are retained before retiring anything |
| [#152 XCodec2](https://github.com/ayutaz/vokra/pull/152) | `a9f54e1a` | Draft, DIRTY | Reconcile current main and exact source/dependency/license/API/owner gates before real CPU/Apple acceptance |

### Canary review: local refusal, no alternative local read

The public PR diff changes only one urllib3 package record in
`tools/parity/canary_1b_reference/uv.lock`: 2.7.0 to 2.8.0, including its
sdist and wheel identities. The new record agrees with the previously
documented [shared release review](../license-audit.md#shared-parity-urllib3-release-review-2026-10-05-pr-194).
Fresh primary [PyPI metadata](https://pypi.org/pypi/urllib3/2.8.0/json) and the
official release's [MIT license text](https://raw.githubusercontent.com/urllib3/urllib3/2.8.0/LICENSE.txt)
were read; this is not approval of the entire Canary closure.

The Luna worker's combined `git show` / `sed` / `rg` source-review command
was refused by the local model/checkpoint PreToolUse guard before execution.
The manager presented that same read-only command to normal escalation with
its exact scope and no Python/Cargo/model execution; it was refused again.
No alternate local source read, helper execution, guard modification, owner
signature, hash restamp or dependency edit follows from those refusals.
The two additional reference-directory names in the older planning note are
absent from current main; they are not invented as new paths.

The guard points real work toward the audited remote route. A new bounded
static VAST controller is being prepared and mock-tested, not yet executed:
exact main/PR source, eleven allowlisted Canary source/manifest/lock files,
stdlib JSON/TOML/AST/text/hash processing only, no project-helper import,
no model/dependency acquisition, no owner/signature/publication change.
The intended ceiling is USD 0.20/hour including 200 GB storage and thirty
minutes total, with exact offer/price validation, owned-ID cleanup and an
independent absence check. Existing credentials stay local; no HF key is
needed. No paid worker has been created at this checkpoint. Static audit
facts cannot promote a row to VAST_READY or authorize real-weight execution.

### Legal PR: unchanged authoritative endpoint hold

The actual failed [documentation-links job](https://github.com/ayutaz/vokra/actions/runs/37770169295/job/113287531384)
records fifteen California endpoint failures, not a Rust or model failure.
A fresh normal web read of the official AB 2713 chaptered body, SB 1000
chaptered body and BPC Chapter 25 endpoint again returns HTTP 403.
No challenge bypass, mirror substitution, link deletion, checker exclusion,
unchanged CI retry, current-law conclusion or owner/legal approval follows.
The original records and actual failed link gate remain attached to #198.

## Six-stage acceptance matrix

| Stage | State at this checkpoint | Required exit evidence |
| --- | --- | --- |
| 1. Existing PRs | IN_PROGRESS | Reviewed current-source deltas; actual required CI; normal merge, supersession or explicit hold without losing responsibilities |
| 2. Inventory and classification | METADATA_REFRESHED, execution state unproved | All 194 exact rows accounted; distinguish full code labels from source/approval/CPU/Apple/publication proof |
| 3. Non-Apple prerequisites | OPEN | Exact converter/binder/native/CLI/dependency/license and independent upstream reference, with explicit external holds |
| 4. Real-weight VAST CPU | OPEN | Approved exact scope, no-upload conversion/reference/numerical CPU result, small receipt recovery and worker/data destruction |
| 5. Final packet and Apple | OPEN | Reviewed final clean HEAD and fresh authenticated direct transfer; Apple CPU/reference, Metal/reference and Metal/CPU plus no-fallback |
| 6. Result/public audit | OPEN | Follow-up PR/CI acceptance, separately authorized repository-scoped publication, final live inventory/evidence/cleanup audit |

Scaleway stays last. Source, legal, implementation or independent-reference
gaps are not solved by renting Apple hardware. Protected CosyVoice2 LLM and
Silero fixture inputs, frozen producer helpers, existing safety hooks and
immutable approval records remain outside this change. No model runs on the
maintainer Mac; no HF model/Space, public repository withdrawal, external
package registry or X post is performed by this checkpoint.

## Initial lightweight checks

The unchanged coverage classifier's fourteen stdlib-only unit tests pass.
They construct metadata records and read the engine as text; they are not
model or Apple tests. Documentation references, runbook-path citations,
English/Japanese community checks and `git diff --check` pass. The default
documentation-example checker passes 144 blocks across 26 documents and
retains thirty explicit deferred blocks as unverified. No workspace or
`vokra-models` Cargo command is run locally. These checks support this
management snapshot only, not any unfinished execution stage above.

## Continuation review: exact VibeVoice responsibility comparison

A fresh source comparison uses the actual open heads, not their older PR-body
snapshots: #169 `b3e9d5c1cd2c75bcc0b2cbe81ce8b115f00e3690` and #182
`a3afc1ecc38db67434c564bbea424e0366e0cdd9`. Both remain Draft and conflict
with current main. The following three files have identical Git blob IDs
between those heads:

| Reference file | Shared blob |
| --- | --- |
| `pyproject.toml` | `05f5a1137d80ee2fc89c74b82d83fc0defba2714` |
| `uv.lock` | `a28686c5b48d74e6d0613efcd313a2211e54f411` |
| `test_lock_contract.py` | `ccc6483b6e58917a626eafaf11fafa394ace9094` |

All three are under `tools/parity/vibevoice_realtime_0_5b_reference/`.
The complete collector/test delta was inspected as text. Relative to #169,
#182 adds the explicit target-interpreter `-S` / `-B` check and recorded
runtime flags, plus duplicate-JSON-key rejection for both file and byte
inputs. The matching new regressions check rejection before artifact
verification/report creation, including preservation of pre-existing report
bytes. This comparison does not execute those regressions or establish
installed/native/license or real-weight acceptance.

The collector is absent from this branch's `HEAD` (management baseline
`b2fe14c6`). Therefore the stronger audit responsibilities have not entered
accepted main: retain #169 until #182, or a fully reviewed equivalent, is
actually accepted and the exact accepted source is rechecked. Do not merge
or close the older PR solely because the replacement contains a superset.
The added runtime/preset/streaming-reference code in #182 still needs its
own review and execution evidence; identical lock bytes are not proof that
the entire PR is ready.

### Reviews that remain unaccepted

The first bounded Kyutai observation patch was reviewed and returned to the
implementer for narrower visibility and stronger callback/finish/error
coverage. Neither this review nor formatting is a compiled/tested source
result. The existing MLX public and DSM private schedule separation must be
preserved; PR #191 is not accepted wholesale.

The first private Canary lifecycle draft was also rejected before provider
execution: query argument framing, SSH/SCP identity/settings, global deadline
and cleanup reservation, receipt/report hash binding, recovery bounds and
offline negative cases need correction. No paid instance is created by this
review. The unchanged repository safe-wrapper regression passed locally:
`test-vastai-safe.sh: OK (redaction + exit-status + destroy confirmation)`.
It uses mocks and proves neither provider cleanup nor the new controller.

The next actions remain corrected controller review and offline tests, then
the separately bounded static audit; corrected Kyutai source review, then
exact-source remote compilation/tests. No model, owner, CPU parity or Apple
state is promoted by these continuation observations.

## Kyutai observation groundwork: reviewed, remote compilation pending

The corrected two-file patch adds only test-configuration observation and
synthetic regression coverage. `layer_cache_view()` is `cfg(test)` and
crate-visible, not a new public runtime API. The PCM observer wraps the
existing `PcmSessionControl::push` / `finish` methods; it does not maintain a
second test-only padding, sampling or poison implementation. Public MLX
24,000/84,000-sample/floor behavior and private DSM 13-prefix/32-suffix/ceil
behavior remain separate and unchanged. No numerical tolerance, source pin,
dependency, owner decision, artifact identity or runtime backend is changed.

| Reviewed source | SHA-256 |
| --- | --- |
| `crates/vokra-models/src/kyutai_stt/pcm_session.rs` | `da47dd9f30398f1cd5d69a4058f8ac64117a2247a19f50cabf8bbbdc9597238b` |
| `crates/vokra-models/src/kyutai_stt/streaming_lm.rs` | `9b863f73e228a879c7ee7b26b04bb9e84ed50e173eba8c734196755d998d3b42` |

Review caught and corrected an unsigned-integer type mismatch and a synthetic
expectation accidentally calculated with the tiny configuration's vocabulary
instead of this control fixture's actual 4,000-entry vocabulary. The test now
asserts its actual 1,920-sample hop and 13/32-frame padding. An independent
integer-only review traces its two non-hop-aligned chunks to 48 sampled rows,
48 raw tokens and 47 emitted tokens; this is test arithmetic, not an upstream
reference or a measured model output. Normal and observed paths are compared
through finish/reset, with frame/LM/finish observer failures and read-only KV
geometry covered by the added test source.

Manager formatting, forbidden-symbol, zero-dependency and diff checks pass.
Compilation, the four added Rust tests and all-target Clippy are **NOT_RUN**
because `vokra-models` work must execute remotely, not on the maintainer Mac.
The reviewed source is a private checkpoint until exact-source remote evidence
is accepted. It does not restore PR #191's entire independent consumer, supply
full real-PCM/KV/logit reference comparison, approve a bound, or close PR #191.
Those remain required before real-weight CPU and the final Apple stage.

## Provider admission and credential-field redaction supplement

The source packet remains fixed at `ee2fb0d74be7846ee21366047fc25871490cd127`:
its incremental bundle is 12,994 bytes, SHA-256
`b6b41c63ee36ef9ddbf09235e314d2785015b7dde8e0cb5eb9e2a51aef17c9d9`,
and requires base `7d910a9c0e2db2ffd544033e4ccb5792fd217ed6`.
Bundle verification and its declared changed-object inventory contain only
the two reviewed Rust files, management documents, commits and their trees;
no model payload or protected-manifest change is introduced. Remote Rust
compilation remains **NOT_RUN**.

At 2026-10-18 02:47:37 UTC, a fresh safe-wrapper account instance read returns
the flat JSON list `[]`: zero instances at that observation. This does not
audit independent network volumes or guarantee future account state. The
existing local SSH public key matches an active provider registration; no key
is generated, replaced or uploaded. Existing credentials remain local.

Installed CLI source inspection confirms that the legacy raw instance-list
command returns a flat list, a live individual read returns a plain row, and
an absent individual read returns `{"instances": null}`. The private controller
must test these actual shapes, `cpu_arch` rather than `gpu_arch`, actual
direct-port mappings, decimal quote precision and readiness before rent is
accepted. An older mock-shaped success is not provider acceptance. Current
private lifecycle/bootstrap revisions are still under review, not live-run
evidence. Main remains `7d910a9c` and the six existing PR heads are unchanged
at this continuation check.

The provider SSH-key response schema includes a `private_key` field. No
private-key value is displayed; this observation is not a newly established
credential leak or rotation event. A bounded wrapper hardening change adds
exact redaction for `private_key`, `PRIVATE_KEY`, `ssh_private_key` and
`SSH_PRIVATE_KEY` using the existing JSON, Python-dict, query and key/value
redactors. Synthetic fixtures cover escaped multiline PEM-like JSON values,
near-match `public_key` / `tokenizer` preservation and wrapped exit status 23.
The diagnostic-option prohibition and mandatory destroy `--yes` gate are
unchanged. Manager diff review, shell syntax, the actual offline wrapper
regression and diff hygiene pass. This shell-only change does not modify the
frozen Rust packet, source/model approvals, reference bounds or any Apple
completion state; it is not a remote compiler or provider-cleanup result.

## First actual static-worker attempt: endpoint schema failure, destroyed

The manager independently ran the frozen lifecycle mock suite successfully
before provider execution. Fresh API reads still bind main `7d910a9c` and
PR #258 `b2846466`. The selected 200-GB offer was `54523925`, machine `17786`,
at USD 0.08888888888888889/hour; the controller retained the USD 0.20/hour
ceiling and thirty-minute total limit, including its cleanup reservation.

The actual attempt created instance `56487976`, then stopped at
`endpoint-readiness` before SSH or the remote static audit. The live plain
instance row contains both its authoritative top-level `id` / `label` and
an `instance` dictionary containing billing fields. The parser incorrectly
treated that billing dictionary as a wrapped instance response and reported
`FATAL wrong-id`. Its real port fields were also not ready yet (`ports=null`,
`direct_port_start=-1`); readiness should be retried on the same owned ID,
not misclassified as an identity mismatch or solved by blind rerenting.

The EXIT cleanup destroyed this exact instance and read back
`{"instances": null}`. An independent safe-wrapper individual read and
account listing, completed before 2026-10-18 03:01:43 UTC, confirm that ID
absent and the instance list empty. This is not an independent network-volume
audit. The instance's saved data was destroyed; it is not retained for reuse.
No model, HF key transfer, helper execution, static-audit report or CPU/Apple
result was produced. Correct the parser against this actual response shape
and add offline regressions before a changed attempt; the failed attempt is
not a successful audit or permission to restamp any approval.

## Corrected endpoint attempt: static input cap failure, destroyed

The corrected parser prioritizes the plain row's top-level identity over its
billing dictionary. The manager reviewed that change and independently reran
the full offline lifecycle suite successfully. A fresh quote remained the
same offer, machine and hourly rate. Instance `56488133` then passed endpoint
and SSH readiness, fetched the exact public revisions and entered the
stdlib-only audit. It failed because an allowlisted input exceeds the draft
256-KiB per-file cap. A bounded `FAIL` receipt was recovered, recording stage
`static-stdlib-audit` and both exact commits; no successful report exists.

Metadata-only `git ls-tree -rl` reads for both revisions identify the cause:
`license_closure_manifest.json` is 417,264 bytes. The eleven allowlisted
files total 677,792 bytes per revision; the lock is 118,835 bytes. Ten files
have identical blob identities across the revisions; only the lock changes.
No source body was recovered to the maintainer Mac through this metadata
inspection, and no refused local source-read route was substituted.

The controller destroyed `56488133` and read back absence. An independent
individual API read, completed before 2026-10-18 03:04:11 UTC, also returns
`{"instances": null}`. Its saved data is not retained. A further attempt must
first correct the file-specific cap, preserve the bounded small-report
recovery contract, and independently pass synthetic tests with representative
manifest size/shape and explicit over-limit rejection. These two failures
expose gaps in the original mocks, not model failures, license decisions,
CPU parity or Apple evidence. Do not replay the unchanged failed helper.

## Actual bounded Canary static facts recovered

The revised collector uses a 512-KiB cap for the two manifest files, 256 KiB
for the other allowlisted inputs, and a 4-MiB per-revision aggregate bound.
Its report remains capped at 64 KiB. Independent offline tests cover the
actual plain-row/billing response, readiness transitions, distinct nested
package/status/hash records in a manifest larger than 417 KiB, exact 512-KiB
acceptance, over-limit rejection, receipt tampering and failed cleanup. The
manager reviewed the revised collector and reran that frozen suite before
the changed live attempt.

Instance `56488878` completed the stdlib-only audit and bounded evidence
recovery. The controller then destroyed that instance and its saved data.
Independent individual and account reads completed before 2026-10-18
03:11:24 UTC return `{"instances": null}` and `[]`, respectively; independent
network volumes are not audited by these reads. No model, HF secret,
reference package, project-helper execution or upload is involved.

The original recovered bytes are retained as management evidence:

| Evidence | Bytes | SHA-256 |
| --- | ---: | --- |
| [Static report](canary-pr258-static-audit-2026-10-18.json) | 36,437 | `3b5c075791e4bee3b81191f66155cb77a9a92bbd28bae5a16ce3bfb804af581a` |
| [Remote receipt](canary-pr258-static-receipt-2026-10-18.json) | 218 | `747171b938c02004588de60ced9361bd5d0303951a115a5b0d1b0c4f2948ac97` |

Inputs are exact main `7d910a9c` and PR #258 `b2846466`, eleven files per
revision and 677,792 bytes per revision. Ten file hashes are unchanged. The
134-package full-record lock comparison finds exactly one changed package:
urllib3 2.7.0 to 2.8.0. Its new wheel and sdist identities match the fixed
[primary PyPI metadata](https://pypi.org/pypi/urllib3/2.8.0/json); the fixed
[upstream license body](https://raw.githubusercontent.com/urllib3/urllib3/2.8.0/LICENSE.txt)
is MIT. This verifies the bounded maintenance delta, not all 134 packages or
the reference environment's installed/native behavior.

Approval status remains `OWNER_SIGNOFF_REQUIRED`, signer/digest remain null,
and the recorded audit status remains `BLOCKED_UNREVIEWED_TRANSITIVE` in both
revisions. The closure snapshot records lock hash `004f0b4d...`; that differs
from both current-main `a931dfb0...` and candidate `67a7eab6...`. Its project
and audit-script snapshots also differ from current source. Preserve those
historical snapshots rather than restamping them as new approval. The
collector's sibling-path matching is heuristic: for example, its comparison
of `virtual_project_evidence.license_sha256` against the sibling project path
is not a valid license-byte verdict. Hashes without an allowlisted unique
target remain `UNKNOWN`. Grouped excerpts are explicitly incomplete; state
aggregates cover enumerated, shortened values, not a complete license proof.
No project gate was executed, so these observations are not an actual
semantic gate PASS/FAIL or a current owner/legal determination.

The maintenance PR may proceed to latest-main reconciliation and fresh CI
without inheriting the frozen reference approval. Real reference acquisition,
dependency/native closure and any execution decision require a new exact
scope and the applicable evidence/authorization. Static success does not
promote either Canary model to CPU/Apple PASS or authorize publication.

### PR #258 latest-main update and fresh CI

The normal GitHub update-branch request, guarded by expected head
`b284646627cc1e35effe3ef7eebe3bd3e6b359c1`, was accepted. The new head is
`41753aa0772f11dc67d8b4a1e2571a2f29745f2b` against main `7d910a9c`.
Directory-metadata API reads for both Canary directories confirm all eleven
blob IDs and byte counts unchanged from the audited candidate. This binds
the narrow static input facts to the updated branch, not a whole-new-HEAD
compiler/reference result. Fresh runs are actually created for this head,
including [CI](https://github.com/ayutaz/vokra/actions/runs/39092780284),
[Security](https://github.com/ayutaz/vokra/actions/runs/39092780265) and
[Quality](https://github.com/ayutaz/vokra/actions/runs/39092780357).
They are pending at this checkpoint; the earlier 69 successes / one skip
belong to the old head and cannot authorize the new merge. PR #258 is not
yet merged by this record.

### XCodec2 source integration map, not acceptance

A separate read-only investigation uses actual merge-base
`8bd30d9ab3857f77f3d0fa1edce34c450ea8c102`, current main `7d910a9c` and
PR #152 `a9f54e1a`. Its true three-way text conflicts are `docs/README.md`
and `docs/handoff/next-steps-2026-10-14.md`; both involve dated-record
insertions. The PR's Rust/reference/workflow source has no text conflict in
that comparison, which is not proof of compilation or semantic acceptance.
Keep its prior source-only nine-test evidence separate from real-weight
parity: the GGUF-unset tenth test returns early. LGPL/GPL/native provenance,
primary archive/license bindings, owner decisions, fresh reference and
CPU/Apple gates remain open. Its environment update is a new reference
scope; no fixture or bound is silently inherited. No PR #152 branch update,
merge or model execution is performed by this investigation.

## Actual Kyutai model-free compiler attempt: Clippy failure

The frozen source packet remains `ee2fb0d7`, not a new model or reference
scope. After manager review and independent offline entry/lifecycle tests,
instance `56490823` ran the fixed fmt, three PCM observer tests, one cache
view test and models-library/integration-test Clippy commands. The selected
200-GB offer remained `54523925` / machine `17786` at USD
0.08888888888888889/hour. The controller retained its USD 0.20/hour ceiling,
27-minute work deadline and cleanup reservation within thirty minutes.
No model, HF key, project reference environment or publication was involved.

The original [failure receipt](kyutai-observation-compiler-failure-2026-10-18.json),
SHA-256 `fedb299066410f114655d715866030d759c6cb26aaa3cfeaab8be1d8efd51891`,
records **FAIL**, stage `clippy`, exit 101, exact clean HEAD before/after,
and unchanged hashes of both reviewed Rust files. The known strict driver
reached Clippy only after fmt and the exact 3+1 nonzero/unignored test
summaries passed. Receipt metadata marks those earlier stages PASS, but their
native output streams were not recovered. This is preliminary stage evidence,
not a fully independently recovered four-test result or a successful overall
source-validation batch. No push/merge is justified by this failed batch.

An evidence-recovery defect is explicit: the failure path copied the receipt
and its matching hash, but not the 6,676-byte Clippy log body (or the other
stage/CPU/toolchain files listed in the receipt). Those files were lost when
the instance's data was destroyed. The failure's actual compiler/lint cause
therefore remains unknown; do not infer it from the exit code or source
inspection, call it flaky, weaken Clippy, or edit Rust based on a guess.
Correct and offline-test bounded failure-log recovery before any changed
diagnostic replay. A later replay must preserve this failed receipt and obtain
the original causal output before destruction, rather than rerent unchanged
and hope for green.

Controller destroy/readback and an independent individual read completed
before 2026-10-18 03:28:30 UTC confirm `56490823` absent with
`{"instances": null}`. Its saved data was destroyed and cannot serve as a
retained retry worker. No real-weight CPU, independent reference, Apple,
owner/legal or publication state advances.

### Monitoring transport boundary

The maintainer's `api.github.com` DNS lookup failed during CI monitoring;
unauthenticated `curl` confirmed a resolver failure. Existing runs were not
cancelled, restarted or treated as terminal. The already-connected GitHub
connector supplies fresh reads of those same run/check IDs; at 03:28:30 UTC
the updated #258 head has 43 successful checks, one intentional skip and
24 queued/in-progress checks with no failures. VAST's safe-wrapper API reads
still work. No machine DNS/network configuration or credential was changed.

## PR #258 accepted after complete fresh CI

The updated head `41753aa0772f11dc67d8b4a1e2571a2f29745f2b` completed
70 check runs: 69 successes and the scheduled-only `gitleaks full history`
skip. The normal PR gitleaks check succeeded. All sixteen required
context/application pairs matched that exact head and succeeded, and all
nine pull-request workflow runs completed successfully. The API reported
`mergeable_state=clean`, non-Draft and main `7d910a9c` before merge.

The expected-head-guarded, ordinary squash merge of
[#258](https://github.com/ayutaz/vokra/pull/258) succeeded at
2026-10-18 03:40:38 UTC as `39dd2b00ed6469db76b1cdcd3d1db39ed30673b4`.
Independent PR and main reads confirm merged/closed and that main SHA;
`origin/main` was fetched to the same commit. No admin override, forced
merge, check waiver, owner approval or historical snapshot restamp was used.
This closes the narrow urllib3 maintenance PR only. The other five PRs
remain separate Draft source/legal/reference responsibilities, and Canary's
owner/native/reference/real-weight/Apple gates remain open.

## Kyutai diagnostic replay: native cause recovered, source acceptance open

After independent full offline lifecycle success, the changed failure-log
recovery controller `e1b76462...` ran the unchanged reviewed Rust packet
`ee2fb0d7` with leaf `5f177423...` on disposable instance `56491836`.
The fresh quote remained offer `54523925`, machine `17786`, USD
0.08888888888888889/hour including 200 GB, within the USD 0.20/hour and
thirty-minute lifecycle limits. This was a model-free diagnostic replay,
not a model/reference run or an unchanged attempt to obtain a green result.

The original [FAIL receipt](kyutai-observation-diagnostic-2026-10-18/receipt.json)
is 2,152 bytes, SHA-256
`5c4d7381faa3cda64220e49bf071ca802f8e48dfbd00031b52a54bd1a0abcef1`.
It records exact clean HEAD before/after, unchanged reviewed Rust hashes,
stage `clippy` and exit 101. This time all six bounded native artifacts were
recovered and independently checked against the receipt's byte counts and
SHA-256 values. The [native artifact envelope](kyutai-observation-diagnostic-2026-10-18/native-artifacts.json)
retains the original bytes as lossless JSON-escaped UTF-8 text, including
original trailing whitespace and blank lines. Decode each `text` string and
encode it as UTF-8 to reproduce its recorded bytes and SHA, without relaxing
the repository's whitespace gate.

| Native artifact | Bytes | Result / scope |
| --- | ---: | --- |
| fmt | 0 | exit 0 |
| PCM observer | 1,935 | 3 passed, 0 failed, 0 ignored; synthetic control tests |
| KV cache view | 377 | 1 passed, 0 failed, 0 ignored; synthetic geometry test |
| Clippy | 6,668 | exit 101; existing BERT/CPU code lint failures |
| CPU/toolchain | 1,204 | x86_64 / AVX2; Rust and Cargo 1.99.1 |
| Rustup | 2,260 | actual bootstrap transcript |

The [recovery index](kyutai-observation-diagnostic-2026-10-18/recovery.tsv)
retains each original role, bytes and SHA. The previous failed receipt and
its unrecoverable-log gap remain historical evidence, not overwritten by
this new run. Controller destroy/readback and an independent individual API
read confirm `56491836` absent (`instances: null`), with its saved data
destroyed. This does not audit separate account network volumes.

The native error body identifies `manual_is_multiple_of` and `collapsible_if`
suggestions in existing `vokra-bert` and `vokra-backend-cpu` code, not failing
Kyutai observer assertions. The successful
[fresh CI Clippy job](https://github.com/ayutaz/vokra/actions/runs/39092780284/job/117572845660)
also records Rust 1.99.1, so a compiler-version difference is not the cause
of that comparison. The private sparse allowlist omitted the repository's
existing `clippy.toml`, although root `Cargo.toml` explicitly relies on it
to pin suggestions to the declared MSRV. The unchanged canonical config is
`msrv = "1.85"`, SHA-256
`a7f71233bb5b4dd63e3aa2c08c18a23100663b7c2f9cbf80992e7e35af984c9e`.

Correct the private checkout/config binding and its mocks, then rerun the
same strict commands with the actual canonical config present. Do not edit
unrelated runtime code, downgrade the compiler, add lint allowances or skip
Clippy to conceal this harness difference. A corrected whole-batch success
is still NOT_RUN at this checkpoint. Four recovered synthetic test successes
do not establish independent PCM reference, real-weight CPU or Apple parity;
the six-stage objective and all 194 public rows remain open.

### XCodec2 deeper read-only review

The fixed PR #152 helper review found no deletion of `Requires-Dist` or
license bytes to manufacture a green dependency result. Its builder emits
explicitly derived, unapproved, no-upload wheels rather than official build
provenance; it deliberately excludes build metadata/README payloads and
may append a missing metadata newline. The source inspector does not verify
PKG-INFO/dependency/license bytes, and the dependency auditor remains coupled
to the PR's exact blocked project/lock/manifest/rows. Synthetic tests lack
direct nonempty Requires-Dist and diverse attribution-preservation coverage.
Consequently no entire helper subset, replacement environment, official
reference, license/native closure or PR #152 acceptance follows from the
read-only review. Do not sever its blocked contracts merely to simplify
integration; all retained responsibilities still need actual closure.

## Config-bound Kyutai replay: source-test lint identified and corrected

The manager reviewed the private config-binding correction and independently
completed both full offline harnesses: leaf and controller. The leaf now
materializes the existing exact-HEAD `clippy.toml` / `rustfmt.toml`, checks
their hashes before Cargo and after the stages, and records both snapshots.
Negative tests reject missing/tampered inputs, actual mock-stage config drift,
missing receipt config fields and altered after-hashes. Syntax and shellcheck
pass. No compiler downgrade, lint allowance or repository config change is
used. Frozen controller/leaf SHA-256 values for this attempt are respectively
`cad913199ab3ad0eb8755dc346f8f804f3338f6828aecd3ed4ba1be3617e469a`
and `cd38f757ef7aff1c762bb46d5db848118a5b0e34316e677e2ab96aa35d2cb187`.

Disposable `56493358` then ran unchanged source `ee2fb0d7` at the same fresh
offer/price and USD 0.20/hour / thirty-minute limits. The original
[receipt](kyutai-observation-config-fixed-2026-10-18/receipt.json) is
2,548 bytes, SHA-256
`358f09d8b262af3d0402db74ef4dd89fadf191c71a552ca0edb9d4d3faa8aa01`.
All six recovered native byte streams in the lossless
[artifact envelope](kyutai-observation-config-fixed-2026-10-18/native-artifacts.json)
and [recovery index](kyutai-observation-config-fixed-2026-10-18/recovery.tsv)
match their recorded byte counts and hashes after independent verification.
Both config snapshots match the canonical hashes. The old BERT/CPU
newer-API lint errors are gone, confirming the prior harness diagnosis.

The overall batch still fails Clippy, exit 101: its actual 3,666-byte log
identifies three `drop_non_drop` errors in the new observer tests at original
PCM source lines 1084, 1135 and 1277. The fmt / three observer / one cache
test stages succeed; they do not override this overall FAIL. The controller
destroyed `56493358` and its saved data, and an independent individual read
confirms `instances: null`. No real model/reference/Apple result is produced.

The Luna correction changes only those three test borrow lifetimes to lexical
scopes, removing explicit `drop(observer_cell)`. Manager's whitespace-insensitive
diff review verifies unchanged callback order, control push/finish, every
assertion and reset boundary. Formatting and diff hygiene pass. Corrected PCM
source SHA-256 is
`804e5fd681503c9392498735070bfdaed6367348c403b1d7715a476f554e266f`;
streaming-LM source and all runtime/config/dependency/numerical policies are
unchanged. This reviewed source correction needs a new committed source packet
and fresh exact-source remote verification before code push/acceptance.

The deeper #191 comparison also confirms that its old 72,000-sample/ceil
public contract and removal of the private DSM padding plan must not replace
current main. The latest Candle library/API success is not the separate,
unmerged rich-cache observer candidate's compilation or a full PCM writer.
Native session logits/KV observation, a synchronized upstream full-PCM writer,
hash-bound independent consumer and approved real-weight CPU evidence remain
required before final Apple packets. A separate private runtime-observer patch
is under review, not applied or accepted; no public API or reference identity
is silently substituted. All 194 public rows and the full six-stage objective
remain in scope.

## Corrected Kyutai source: four-stage PASS

The corrected source was committed as
`c69395d53618004439b69717f44e01c7385c9eca` and transferred by verified bundle
from prerequisite `7d910a9c0e2db2ffd544033e4ccb5792fd217ed6`.
The 42,061-byte bundle SHA-256 is
`29c493921c2cda1e652f85815966035435f8b59542bd7e3086175202918d56b5`.
Frozen remote leaf/controller SHA-256 values are respectively
`c54f5a8da8b9559753cea0efe6b621a55e7d596d5deed8538fa72359c3960cb5`
and `489f44f4f62ae7cf23aae28b26c1eff8c8cdd017e4ab3f1ac829a5c4db52efd9`.
Their independently completed offline harnesses pass before activation;
only the five mechanical source/bundle identity pins changed from the
already reviewed config-bound helpers.

Disposable worker `56494571` completed the four fixed commands with
`CARGO_BUILD_JOBS=8`, Rust/Cargo 1.99.1 and the unchanged canonical
`clippy.toml` / `rustfmt.toml`:

```sh
cargo fmt --all -- --check
cargo test -p vokra-models --lib kyutai_stt::pcm_session::tests::observed_control -- --test-threads=1
cargo test -p vokra-models --lib kyutai_stt::streaming_lm::tests::layer_cache_view_tracks_geometry_and_reset_without_mutation -- --exact --test-threads=1
cargo clippy -p vokra-models --lib --tests -- -D warnings
```

The original [receipt](kyutai-observation-source-pass-2026-10-18/receipt.json)
is 2,628 bytes, SHA-256
`a2f801a88d453385b9e6b551058e33c0ecddc62aa182a1b5277d2c287884a32c`.
The lossless [native-artifact envelope](kyutai-observation-source-pass-2026-10-18/native-artifacts.json)
retains six original UTF-8 streams. Manager independently verifies each
decoded stream's exact bytes and SHA against the receipt, the unchanged
before/after source and config hashes, and clean exact HEAD before/after.

| Native artifact | Bytes | Actual result / scope |
| --- | ---: | --- |
| fmt | 0 | exit 0 |
| PCM control observer | 1,935 | 3 passed, 0 failed, 0 ignored |
| KV cache view | 377 | 1 passed, 0 failed, 0 ignored |
| strict Clippy | 1,587 | exit 0, actual dev build finished |
| CPU/toolchain | 1,204 | x86_64 / AVX2; Rust/Cargo 1.99.1 |
| Rustup | 2,260 | actual bootstrap transcript |

Canonical Clippy selects configured MSRV 1.85 and emits the existing
package-MSRV diagnostic; no lint allowance, compiler downgrade, assertion
removal or unrelated runtime correction is used. The three lexical-borrow
scope corrections resolve the actual preceding `drop_non_drop` failures.
Earlier failed receipts remain failures and are not replaced by this PASS.

The controller completes `PASS audit-and-destroy`. Independent readback at
04:08:26 UTC confirms the owned ID returns `instances: null` and the account
instance inventory is `[]`. Worker and attached saved data are destroyed;
this is not an independent audit of separate network volumes. The fresh
offer including 200 GB storage was USD 0.08888888888888889/hour, within the
USD 0.20/hour ceiling and thirty-minute lifecycle bound. No HF credential,
model acquisition/execution, publication or Apple work was involved.

The main `39dd2b00` post-merge CI readback separately completed 75 checks:
70 success and five event-gated skips (full-history gitleaks, PR-only semver
and dependency review, schedule/dispatch-only Miri and address sanitizer).
Those checks do not substitute for fresh CI on this source PR's new head.

Next retain this evidence in a management commit, push the reviewed source
after the independently passing compliance regression, normally reconcile
latest main and open the narrow source/safety/evidence PR. Require actual
fresh required contexts before normal merge. Keep the separate private rich
runtime-observer candidate out of that verified source change. Full native
PCM/upstream observation, independent real-weight CPU and final Apple gates
remain unfinished; all 194 rows and the complete six-stage goal remain.

## PR 267: immutable audit quote triggers spelling check

The verified source/evidence batch is pushed and normally reconciled with
main `39dd2b00` at work-branch head
`4396f97902e6334189897213e0bd584996359f78`; [PR #267](https://github.com/ayutaz/vokra/pull/267)
is non-Draft. The initial exact-head CI has a real
[typos failure](https://github.com/ayutaz/vokra/actions/runs/39096349344/job/117583506159),
not a Rust/compiler failure: typos v1.50.3 flags two identical clipped
source-quote fragments in the retained Canary static-audit JSON. Its raw
36,437-byte SHA-256 remains
`3b5c075791e4bee3b81191f66155cb77a9a92bbd28bae5a16ce3bfb804af581a`;
correcting the quoted bytes would invalidate original evidence.

Luna adds only that exact file to the existing immutable-source/receipt
exclusions in `_typos.toml`. No global word allowance, directory-wide
exclusion, original-evidence edit, runtime-policy change or hook
change is made. Focused spelling checks use the installed local typos 1.48.0;
the actual hosted 1.50.3 result on the new head is still required. A separate
stdin negative control still rejects the clipped token (exit 2), confirming
it is not allowed globally. The old
failed job is not retried unchanged or relabeled successful. Complete fresh
exact-head CI remains the normal-merge gate, and all real-weight/Apple
requirements remain open.

## PR 267 accepted after complete fresh CI

The narrow source/safety/evidence PR is normally squash-merged at
04:44:17 UTC as main `5b688666ee26bf66ae1e75dd011d49d45168dc70`.
Independent [PR readback](https://github.com/ayutaz/vokra/pull/267) confirms
closed/merged, exact reviewed head
`62ebb76b6ec95b194a859f55b9108b4652434dd6` and the matching protected-main
merge result. Maintainer main was normally fast-forwarded to that commit
and confirmed clean before starting a separate implementation branch.
No force/admin merge, waived check or safety-rule change is used.

The complete single-page exact-head check readback contains 86 results:
81 success, five conditionally skipped, zero failed or live. All sixteen
required context/application pairs succeed, including external CodeQL app
57789. All thirteen workflows complete successfully on attempt 1. This is
not a partial first-page count or reuse of the earlier failed head's checks.
The [bounded acceptance readback](pr267-ci-acceptance-2026-10-18.json)
retains all check IDs/application IDs and workflow identities for this exact
head, plus the independently confirmed normal merge identity and limits.

| Skipped check | Actual condition / non-claim |
| --- | --- |
| Miri / address sanitizer | Schedule or explicit dispatch only |
| Full-history gitleaks | Schedule or full-history dispatch only |
| Whisper WASM e2e | Explicit model-fetch opt-in or monthly schedule only |
| NeMo real-checkpoint matrix | Actual setup log reports checkpoint sourcing disabled; skipped, not real-weight parity |

The corrected [hosted spelling job](https://github.com/ayutaz/vokra/actions/runs/39097034789/job/117585745146)
actually downloads typos 1.50.3, checks the canonical `_typos.toml` and
succeeds on the PR merge ref binding that exact head to main `39dd2b00`.
The immutable Canary receipt remains byte-identical. The original failed
job remains failed; superseded still-running workflows are automatically
cancelled by their existing PR concurrency policy, not relabeled successful.
Ordinary backend/model-conditional CI does not establish the separate
all-row Apple CPU/reference, Metal/reference or no-fallback campaign.

Next branch `codex/kyutai-runtime-observation-20261018` starts from the
accepted main. The reviewed private native-session candidate adds a
borrowed synchronous observation seam to the real Mimi/LM/greedy route;
implementation and synthetic execution coverage still require actual remote
Rust verification. No public loader, provenance gate, numerical tolerance,
upstream reference identity or owner decision may be weakened to enable it.

### Separate XCodec2 private candidate: not accepted

A Luna private candidate adds metadata/attribution preservation tests for
the PR #152 helper scope. Its test invocation is refused by the maintainer's
local model/checkpoint guard. Follow-up source/hash/metadata inspection is
also refused; root presents the identical read-only Git inspection command
to normal escalation once, and it is refused again. No alternate read,
hash, execution, transfer or guard modification is used.

The worker reports additions to two private test files but cannot prove
the pre-edit private worktree was clean. Those changes remain unreviewed,
untested and unaccepted. Do not restore or discard that unknown pre-existing
private state, transfer it, or integrate it into main. The current repository
and frozen successful VAST helpers are unaffected. This hold does not close
PR #152, approve its reference environment or advance any model row.

## Actual-session observer implementation: remote verification pending

On the separate branch from accepted main `5b688666`, Luna extends only
`crates/vokra-models/src/kyutai_stt/pcm_session.rs`. Manager reviews the
complete diff: ordinary push/finish and the test-only observer use the same
Mimi encoding, streaming LM step and actual greedy selection. The production
callers supply no-op hooks; observer entry points, borrowed event and
collector tests remain `cfg(test)` and crate-private. No public ASR loader,
authenticated constructor, provenance/hash gate, MLX/DSM schedule,
dependency, license, numerical tolerance or unsupported-backend behavior
changes.

The callback borrows actual frame/codes, logits, sampled token, native step
position, next position and an existing read-only per-layer KV view. The
test collector synchronously copies observations before the next forward.
This is not an upstream oracle, output writer/consumer packet, new cache
owner or model-execution permission.

Synthetic execution tests construct the existing tiny first-party decoder
and synthesized Mimi with the private 24 kHz / 12.5 Hz / 1,920-sample
boundary and matching codebooks. The fixture's direct fields are test-only;
the empty tokenizer deliberately excludes transcript coverage. Tests compare
the actual normal/observed session route, different input partitions,
finish, chronological cache eviction and reset's complete captured values.
Separate frame/LM callback failures during push and finish must poison the
session; after reset, actual push/finish must succeed from position zero.

Manager catches and returns an eager optional-value expression that would
index the preceding token at index zero; Luna replaces it with lazy branch
selection before any remote run. No assertion or lint gate is removed.
The resulting PCM source SHA-256 is
`bd665ef70491e44930235c7818a88802a1e77df6e4290093b8bd4a6ee632c830`;
streaming-LM source stays
`9b863f73e228a879c7ee7b26b04bb9e84ed50e173eba8c734196755d998d3b42`.
Static source enumeration finds 27 PCM and five streaming-LM unit tests,
with no ignored test in either module. These are expected execution counts,
not successful test results.

Formatting and diff hygiene pass. Local model, compiler/test/Clippy,
provider and SSH execution remain unperformed for this wave. Freeze the
reviewed source commit and fresh bundle, independently check the separately
derived bounded lifecycle helpers and all offline mocks, then perform actual
remote verification before code push. Intended strict remote commands are:

```sh
cargo fmt --all -- --check
cargo test -p vokra-models --lib kyutai_stt::pcm_session:: -- --test-threads=1
cargo test -p vokra-models --lib kyutai_stt::streaming_lm:: -- --test-threads=1
cargo clippy -p vokra-models --lib --tests -- -D warnings
```

Even a successful synthetic session run would close only this source seam.
Independent full-PCM upstream capture/consumer, dependency/license/operator
closure, approved real-weight CPU, final clean HEAD/fresh transfer packets,
Apple CPU/Metal/no-fallback and final public reconciliation remain open.
The complete six-stage objective and all 194 public rows are unchanged.

## Actual-session compiler failure and minimal correction

The fixed `7f2c0bc9a98d326e8c02e7cf6f8ae541cc669728` source packet is
actually transferred to disposable VAST `56501022` on machine `17786`.
The fresh 200-GB-inclusive quote is USD 0.08888888888888889/hour, within
the unchanged USD 0.20/hour and thirty-minute lifecycle ceiling. Independent
manager review and full offline leaf/controller harnesses pass first;
SSH/SCP explicitly disable agent forwarding. Only source bundle and leaf
are transferred, with no model, HF credential or upstream framework job.

Formatting succeeds. The PCM library-test compilation exits 101 with two
E0308 diagnostics at lines 731 and 735: a method item tied to one engine
lifetime cannot be coerced to the signature test's independently
`for<'a>`-quantified engine-lifetime function pointer. No PCM or LM test
executes, and streaming-LM/Clippy stages are not reached. Do not count the
planned 27/5 tests as passes or retry this unchanged source.

The [lossless native evidence envelope](kyutai-runtime-observation-failure-2026-10-18.json)
retains the original 2,061-byte receipt (SHA-256
`5d85acda71a0690d05c217c7a2929e5218de48f6b123050d0e5ef5bc12199351`),
empty successful fmt stream, complete 3,844-byte compiler stream, CPU record,
Rust setup stream and recovery ledger. Manager independently verifies all
reported byte counts and SHA-256 values. Source and canonical config hashes
are unchanged and checkout dirty counts remain `[0,0]` before/after.

The controller exits nonzero after recovering the original evidence and
destroying its owned instance/data. Its individual readback is
`{"instances": null}`; independent safe-wrapper individual/account reads
also confirm target absence and zero account instances at this checkpoint.
This does not audit separate network volumes or guarantee future inventory.

Luna's correction changes only the signature test: a generic helper fixes
the engine lifetime while checking both method pointers and keeps the
observer/event type assertions. Production behavior, runtime tests, public
loaders, source pins, bounds and dependencies are unchanged. Reviewed PCM
source SHA-256 becomes
`69947534c4c02b916dc99176b28624bdb0aadee98f51386674fd6ff757fe7cf7`.
Formatting and diff hygiene pass; actual revised compilation/tests/Clippy
remain NOT_RUN until a new clean source packet is verified remotely.
Independent full-PCM reference, real-weight CPU, owner/legal, final packet,
Apple CPU/Metal/no-fallback and public reconciliation remain open.

## Actual-session tests pass and signature lint correction

The first revised-source attempt stops at offer preflight because machine
`17786` returns no current quote. It reaches no baseline/create/SSH stage
and creates no instance. A new read-only market observation identifies
machine `31844` at USD 0.09074074074074073/hour including 200 GB, with the
same fee, capacity, bandwidth, ownership and thirty-minute lifecycle gates.

Disposable `56502593` then actually verifies exact clean
`155be525c356430c360b3fb4c16dff2685fc96a0`. Fmt succeeds, all 27 PCM tests
pass in 6.17 seconds, and all five streaming-LM tests pass in 0.02 seconds;
both have zero failures and zero ignored tests. The native PCM observations,
chunking/reset equivalence and callback-error recovery therefore have actual
synthesized execution evidence, not independent full-PCM or real-weight parity.

Strict Clippy exits 101 on one `extra_unused_lifetimes` diagnostic: the
generic signature helper names its lifetime only inside body-local types,
not in its function signature. This is not another E0308 compiler failure
or a failed numerical assertion. The canonical dependency MSRV diagnostic
also remains visible; no config, warning policy or assertion is altered.

The [lossless native envelope](kyutai-runtime-observation-lint-failure-2026-10-18.json)
retains the original 2,541-byte failure receipt, SHA-256
`89bbd6477706244c8220673eba195b879770a3d28de5c0a6b89b09c19d7a769a`,
all four command streams, CPU and Rust setup records, and recovery ledger.
Manager independently checks every original length/hash, actual 27/5
summaries, unchanged source/config hashes and `[0,0]` checkout dirty counts.
Controller cleanup destroys this exact worker/data and reads individual
absence. Independent safe-wrapper individual/account reads confirm the ID
absent and zero account instances. Separate network volumes are not audited.

The next reviewed change adds a zero-sized `PhantomData` argument to the
same helper, binding the engine lifetime in its signature while preserving
both method-pointer checks and requiring no model construction. Only this
test changes; no lint allow, assertion removal or production change occurs.
New PCM SHA-256 is
`be8544416fc4c8ee19f32bdbee33d3e7439e014ef40ca1d7ede731bc127f8a57`.
Its remote Rust/test/Clippy verdict is still NOT_RUN. Preserve the original
failed heads and require the entire new fixed-source batch, not merely the
last lint command, before push. All six-stage and 194-row duties remain.

## Actual-session fixed-source four-stage PASS

The additional signature-only correction is committed and frozen at clean
`59a0afda1b6b4bbe6d158c54bccb1add841691b2`, base
`5b688666ee26bf66ae1e75dd011d49d45168dc70`. The fresh bundle is 27,328 bytes,
SHA-256 `7f270b1cc8aba1785a1f20e86bc6d774e9a55fff211acdb516a8973a7ef443a7`.
The prior `155be525` packet and original failures remain unchanged. Manager
reviews the complete new helper diff (pins/paths only), rechecks syntax,
ShellCheck and source/bundle hashes, and reviews Luna's terminal full leaf
and lifecycle mock evidence. Their existing price, deadline, ownership,
failure-recovery, no-forwarding and destroy/readback logic is unchanged.

Machine `31844` has no matching current quote at admission, so that attempt
stops before instance creation. A fresh market observation identifies offer
`53652712`, machine `39233`, at USD 0.1325925925925926/hour including 200 GB.
The controller independently revalidates the exact offer and creates only
owned `56503887` under the existing USD 0.20/hour / thirty-minute ceiling.
It transfers the reviewed source bundle and leaf only, not credentials or
model artifacts, and actually executes all four fixed commands:

| Stage | Actual verdict |
| --- | --- |
| `cargo fmt --all -- --check` | exit 0 |
| PCM module tests, serial | 27 passed / 0 failed / 0 ignored; 7.91 s |
| Streaming-LM module tests, serial | 5 passed / 0 failed / 0 ignored; 0.03 s |
| `cargo clippy -p vokra-models --lib --tests -- -D warnings` | exit 0; actual dev build finished |

The [lossless native evidence](kyutai-runtime-observation-pass-2026-10-18.json)
preserves the original 2,628-byte PASS receipt, SHA-256
`5d89bdf185317013bf2b6372cda71dd97b6d9fa9a198e6143fe8549cc3e25bef`,
and all six original native streams. Manager independently checks every
reported byte count and SHA-256, the actual 27/5 zero-ignored summaries,
all four zero exit codes, no log truncation, exact before/after HEAD,
`[0,0]` dirty counts and unchanged source/canonical-config hashes. PCM stays
`be8544416fc4c8ee19f32bdbee33d3e7439e014ef40ca1d7ede731bc127f8a57`;
streaming LM stays
`9b863f73e228a879c7ee7b26b04bb9e84ed50e173eba8c734196755d998d3b42`.
Rust setup and CPU/ISA metadata are retained. The existing dependency MSRV
diagnostic remains visible; no warning allow or config change is introduced.

The controller returns zero with `PASS audit-and-destroy` after evidence
verification and destruction of its exact owned instance and saved data.
Its individual readback is `{"instances": null}`. Independent safe-wrapper
individual/account reads confirm `56503887` absent and the instance list
empty. This is not a separate network-volume audit or a future inventory
guarantee. No source job or retained instance remains running.

This accepts the synthesized actual-session observation seam only. The
source push and new exact-head PR CI come next. Do not replay this green
Cargo batch merely to collect its already verified native bytes. Continue
the independent full-PCM writer/consumer and existing Draft responsibilities,
dependency/license/operator closure, approved real-weight CPU, final clean
HEAD/fresh authenticated packets, final Scaleway CPU/Metal/no-fallback and
result/public audit. Do not substitute additional synthetic seam tests for
the still-missing independent reference. All 194 rows remain accounted.

## PR 268 accepted after complete fresh CI

[PR #268](https://github.com/ayutaz/vokra/pull/268) has exact reviewed head
`32e3376a0d6143e18d43b6a14b93d018a840182a` and base
`5b688666ee26bf66ae1e75dd011d49d45168dc70`. Its last commit records the
already verified native result only: runtime/build/config/workflow inputs
are unchanged from native-verified `59a0afda`. The complete fresh run ends
with 86 terminal checks: 81 successful, five justified skips and zero
failures. Every one of the sixteen required context/application pairs is
uniquely successful, including external CodeQL app `57789`; all thirteen
workflows finish successfully on attempt one. The last Unity package job
`117600840007` finishes its actual package/platform work, not a cancellation.

The [bounded acceptance evidence](pr268-ci-acceptance-2026-10-18.json)
retains every check ID, required pair, workflow/head/attempt, merge identity
and explicit limits. NeMo's real-checkpoint matrix is skipped because its
actual admission gate logs disabled opt-in. The other skips are opt-in
Whisper WASM model fetching and schedule/dispatch-only sanitizer, Miri and
full-history secret scanning. None supplies real-model evidence. Hosted
typos uses the pinned version and canonical dictionary without a new
exclusion. No rerun, cancellation, waiver or required-check change is used.

Normal expected-head-guarded squash merge completes at
`2026-10-18T06:03:33Z` as main
`8ae9566e25753b361be5703d46bcdad967f7cefb`. Independent raw PR and main API
reads agree; local main is normally fetched and fast-forwarded cleanly.
Source/runtime/build/config/workflow inputs are unchanged between accepted
PR head and merge commit. These are PR-head CI results, not a claim that
post-merge CI has separately completed. This documentation branch starts
from that exact main and adds no implementation or execution authority.

Original native success and both earlier failure records remain unchanged.
The owned worker `56503887` and saved data were destroyed and independently
absent at the recorded checkpoint; no new paid worker is rented for this
acceptance record. This is not a fresh account/network-volume billing audit.
No independent full-PCM, real-weight CPU, Apple, owner or public-model row
is promoted by this source acceptance.

## Remaining six-stage order after PR 268

The fresh PR API read after #268's merge confirms five open Drafts:
#198 at `5614a92c`, #191 at `63cadf3c`, #182 at `a3afc1ec`, #169 at
`b3e9d5c1`, and #152 at `a9f54e1a`. Their responsibilities are retained;
neither an old mergeability value nor successful synthetic CI makes them
ready for automatic merge or closure.

The next Kyutai review of the existing decoder runner/reference was stopped
by the local protection guard. A combined read-only source command was
refused normally and the identical command was refused once through normal
escalation. It requested the first 245 runner lines and approval/source/
dtype/streaming-related reference matches, not model execution. No alternate
read, source transfer, agent/remote detour, hidden environment change or
guard weakening follows. Formal read permission or a legitimate permission
state change is required before that inspection resumes. The separate
private XCodec2 candidate remains unreviewed/unrun after its recorded normal
and escalated refusals; preserve it without integration or discard.

The [OWSM primary-source readback](owsm-primary-license-readback-2026-10-18.md)
adds exact-source evidence but does not resolve the active dependency's
missing license grant. The fixed ESPnet frontend still requires that
dependency; no guessed native-complex switch, local mirror or unused-import
stub is an independent official replacement. #198 retains the official
California endpoint's HTTP 403 hold, without treating that response as a
passed citation or a legal sign-off.

| Stage | Remaining acceptance condition |
| --- | --- |
| 1. Existing PR reconciliation | Review/reconcile the five Draft duties against current main; resolve actual source conflicts and overlap; use fresh exact-head CI before normal acceptance. |
| 2. Inventory and exact prerequisites | Retain all 194 rows and dated 136 code/artifact-full / 58 non-full classification; bind missing source/artifact/reference/hardware facts per row. These labels are not model passes. |
| 3. Non-Apple closure | Resolve formal source-read holds, independent upstream writer/consumer and exact dependency/license/native/operator/owner scopes. Do not invent or restamp frozen approvals. |
| 4. Real-weight VAST CPU | Only after prerequisites: authorized no-upload conversion, independent upstream reference and CPU parity at a reviewed clean HEAD; recover small original evidence and destroy owned worker/data. |
| 5. Final packet and Apple | Freeze final clean HEAD and authenticated fresh direct packets; only then provision Scaleway for Apple CPU/reference, Metal/reference and Metal/CPU with explicit no fallback. |
| 6. Results and final audit | Result PRs, fresh CI, normal merges and all-row/evidence reconciliation; audit unnecessary cloud resources. Any model/Space/registry publication or withdrawal needs its own exact authority. |

The frozen 63-row owner packet is unchanged and separate from the live
194-row denominator. Published `v0.4.0-rc.1` remains a CPU-only preview;
there is no retag/rebuild or new HF/model/Space/X publication here. The
withdrawn `ayousanz` destination instruction is not used. No additional
synthetic seam wave, documentation PR or CI success substitutes for the
remaining real-model and Apple requirements. The complete objective stays
open; these holds must be resolved, not silently removed from scope.
