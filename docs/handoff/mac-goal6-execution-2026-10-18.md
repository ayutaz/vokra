# Mac CPU / Metal six-stage execution readback — 2026-10-18

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
