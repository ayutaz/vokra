# v0.4.0-rc.1 packaging and publication plan — 2026-10-17

## Scope and starting point

The user's new request is to execute items 1–3 of the latest release plan:
version/distribution configuration, candidate payload validation, and a GitHub
pre-release after normal PR/CI acceptance. X posting is not included. This is
a new scope, not a repeat of the consumed representative CPU one-shot.

Baseline is clean main `fab4638a8d85f6d9ce7615a02bd20bbcc2253683`.
Work branch: `codex/release-040-rc1-packaging-20261017`.
The starting published Latest is `v0.3.0`; `v0.4.0-rc.1` is not yet published.
The [previous representative CPU result](release-0.4.0-preview-cpu-results-2026-10-17.md)
remains evidence for its exact source/model scope, not a candidate-artifact test.

## Required acceptance evidence

1. Synchronize the workspace, first-party path/lock records and relevant
   bindings to `0.4.0-rc.1`, keeping runtime external Cargo dependencies zero.
   Roll the existing Unreleased notes without rewriting historical releases.
2. Build same-run CPU-only desktop CLI/C ABI payloads for Linux x86_64,
   macOS arm64 and Windows x86_64, including the static Linux musl CLI.
   Verify binary identity, native model-free loading/help, exact version,
   checksums, separate CLI/C ABI SBOMs, and the complete payload manifest.
   Platform build/help evidence is not an Apple model/Metal result.
3. Revalidate Whisper base ASR and Silero VAD v5 speech/silence using the
   candidate's assembled Linux CLI, not a separate source rebuild. Use the
   previously reviewed exact revisions/hashes (293,043,104 bytes combined),
   a hosted worker, CPU-only execution, bounded receipt recovery and cleanup.
   A model job is bounded to 30 minutes; no local model acquisition/execution.
4. Review the complete diff and pass focused static/regression gates, the
   normal PR's exact-head CI and required context/application pairs. Merge
   normally, then bind the final release to the reviewed clean source HEAD.
5. Publish only a complete, verified GitHub pre-release `v0.4.0-rc.1`, explicitly
   not Latest. Verify its tag/HEAD, release flags and actual asset inventory.
   Keep an incomplete release draft rather than advertise it as released.

## Publication and safety boundaries

At the start, repository variables explicitly disable crates.io, npm, OpenUPM,
PyPI and the global desktop/AAR gate. Do not change global registry decisions
or use them to infer permission. RC desktop publication requires explicit,
candidate-scoped routing, separate from the existing stable/AAR path.

No HF upload/withdrawal, external registry publication, X post, owner/legal
sign-off, parity-bound change or model-catalog promotion is authorized here.
Do not read or change protected owner/fixture trees or frozen producer helpers.
Existing hooks and local memory/model guards remain unchanged. Heavy builds
run remotely; credentials and models are not recovered onto the maintainer Mac.

## Current execution state

Local implementation and management review are prepared. No candidate tag,
Release, distribution payload or new model-validation result exists yet.
This record is not proof of a remote build or publication gate.
Append actual PR/run/HEAD/asset evidence and failures when observed; never
replace the earlier CPU receipt with a new claim.

The first read-only VAST offer query through `vastai-safe.sh` returned
`401 Invalid user key`. No instance was created and no new provider charge
was incurred by this attempt. The maintainer's deep pre-push path requires
recorded remote VAST verification before its permitted skip path; do not
bypass it with an alternative upload/ref mechanism or weaken the hook.
Authentication restoration was requested while local preparation continues.

Local lightweight evidence collected during preparation (not a remote build):

- `cargo metadata --no-deps --locked --offline`: 22 workspace packages resolve
  to `0.4.0-rc.1`, without compilation.
- `test_version_contract.py`: 17 checks passed; `test_changelog.py`: 13 passed;
  `test_desktop_release.py`: 16 mocked payload tests passed.
- `test_release_artifact_handoff.py`: 24 static checks passed;
  `test_rc_release_workflow.py`: 34 static checks passed. The initially stale
  macOS runner assertion was corrected and the complete RC suite rerun green.
- `actionlint` passed for the four changed/new workflows; workflow hygiene
  passed across 59 workflows. These parsers do not execute hosted jobs.
- Zero-dependency, changelog integrity, documentation references/community,
  documentation-example self-tests/default scope and `git diff --check` pass.
  The default documentation-example scope does not include the new preview
  guides; their shell blocks were parsed with `bash -n`, not executed.
- [English](../release-0.4.0-rc1.md) and [Japanese](../release-0.4.0-rc1.ja.md)
  candidate guides require the exact published pre-release/assets before use;
  a candidate source version alone does not establish publication.

Manager review corrected the initial payload path, executable-mode restoration,
cross-platform shell/error handling, sparse-checkout boundaries, source/version
binding, and draft/tag handling. The RC path uses an explicit non-latest
pre-release after model-free native install checks, the assembled Linux model
smoke and a byte-for-byte remote asset round-trip. Registry decisions and the
global desktop/AAR gate remain unchanged. No implementation change claims a
new runtime/model capability or weakens an existing safety hook.

Next execution order after authentication is restored: commit/bundle the
reviewed exact source, perform model-free VAST verification and recover small
evidence, destroy this task's worker/storage, then push via the established
verified path, complete normal PR/CI/merge, and dispatch the RC workflow only
at its reviewed merged main SHA. Review actual logs/receipt/assets before
claiming items 2–3 complete. No fresh paid worker was started during preparation.

The all-model campaign remains separate: non-Apple source/dependency/license/
implementation/reference closure, approved real-weight CPU parity, final
clean HEAD/fresh packet, Scaleway Apple CPU/Metal/no-fallback, and final audit.

## 2026-10-18 continuation — source verification not yet accepted

The resumed request covers the four-step plan through the actual GitHub
non-Latest pre-release. Preparation is committed as
`3d4a88f573c95707a37df546ac40e691d005325f`; the public main baseline remains
`fab4638a`. No RC branch push, PR, tag or Release has occurred at this checkpoint.

The earlier authentication hold is superseded: the existing repository `.env`
credential works when supplied through UV's environment-file support to the
safe wrapper. No credential is printed, copied to a worker or replaced.

Pre-build lifecycle diagnostics exposed separate admission/readiness problems:

- Consecutive searches can return different offer IDs for the same machine
  and quote. The private controller now binds an explicitly reviewed machine
  and price to the current response's offer ID; it does not substitute a
  different machine or accept a changed quote.
- The installed CLI's default search does not include the SDK's explicit
  `rented=false` condition. Both the query and the local response validator
  must require `rentable=true` and `rented=false`.
- An instance may exist before its connection fields are ready. The current
  CLI's direct SSH helper reads `ports["22/tcp"][0]["HostPort"]` with
  `public_ipaddr`; a legacy `direct_port_start` value alone is not sufficient
  evidence of the SSH endpoint. Exact instance/label validation and bounded
  readiness are required before connecting. The earlier combined endpoint
  error does not identify which individual field was unready.

Three short-lived workers were created and destroyed with their stored data:
`56416236` (16:07:11–16:07:17 UTC), `56416558` (16:10:16–16:10:21 UTC), and
`56416871` (16:13:04–16:13:08 UTC), all on 2026-10-17 UTC / 2026-10-18 JST.
Each controller cleanup recorded `instances: null`, and the manager independently
confirmed the individual absence. These are failed startup attempts, not
compilation/model evidence. Other admission failures occurred before creation.

The controller's conservative hourly ceiling was revised from USD 0.20 to
USD 0.25 including 200 GB storage, with the thirty-minute work/cleanup bound
retained; metered network transfer is separate. Each actual run still requires
its exact current quote. The controller and mocked lifecycle regressions are
private operational helpers, not changes to repository safety hooks.

At this checkpoint the updated SSH mapping regression is being finalized.
Exact-source Clippy/CPU builds, normal PR CI/merge, candidate payload/model
validation and publication remain unaccepted. Append actual recovered evidence
below before advancing any of those gates. No local model acquisition/execution,
HF/Space publication, external registry publication, X post or Scaleway worker
is part of this continuation.

## 2026-10-18 successful model-free VAST source verification

This result supersedes the pending source-verification checkpoint above,
not the still-unrun candidate-distribution/model/publication gates.

One further startup attempt, `56418229`, reached connection-port assignment
but rejected a multiple-record mapping (16:22:44–16:23:33 UTC). It was
destroyed with its data and independently read back as absent. The corrected
controller validates all `22/tcp` records, deduplicates a shared HostPort,
prefers an explicit IPv4 binding and still refuses genuinely ambiguous ports.

Final worker `56418617`, machine `14503`, verified exact source
`3d4a88f573c95707a37df546ac40e691d005325f`. Live readback confirmed 32 effective
cores, 200 GB storage and USD `0.24814814814814815`/hour. It was created at
16:26:05 UTC and destroyed at 16:33:13 UTC on 2026-10-17 (2026-10-18 JST),
within the thirty-minute limit. Only small text/JSON evidence was recovered;
no binary or model was recovered onto the maintainer Mac.

All twelve stages returned zero:

- zero-dependency gate; version contract 17; changelog 13; desktop tests 16;
  artifact-handoff checks 24; RC workflow checks 34; workspace format check;
- `CARGO_BUILD_JOBS=32 cargo clippy --workspace --all-targets --locked -- -D warnings`
  (41.89 seconds). Cargo emitted warning summaries in the recovered tail;
  successful exit is not a claim that the log contains no warnings;
- CPU-only release CLI build (2 minutes 40 seconds) and C ABI build
  (1 minute 59 seconds), followed by CLI help and native C ABI version smoke
  against expected `0.4.0-rc.1`.

The original receipt is:

```json
{"status":"PASS","scope":"model-free-release-candidate-static-and-compile","head":"3d4a88f573c95707a37df546ac40e691d005325f","version":"0.4.0-rc.1","stage":"capi_native_smoke"}
```

Recovered evidence SHA-256 values:

| Evidence | SHA-256 |
| --- | --- |
| Original receipt | `92939fe832e7e1dfac978d6b937ab190553f0b5dd125cac098d76d773cfc8727` |
| Bounded stage logs | `4454e62d97c8ddf92334a4af5fac4d53305ab5b07bb4d4ea54077a6cb2aa3b90` |
| Controller status | `a90f9dfb9981a605334ab0411200173d160597041b01e8ce230696b8e5a3e590` |
| Reviewed source bundle | `4e1759b97b4738aaf8bf2df2a165762781b2e64a4df8fd0e4b647a6c44be721f` |
| Final private controller | `b360afd8e75517c9c685c8d2dfad5b23b7019e9b4c2ce040f521ebbab3206e4d` |

Cleanup returned `destroyed-and-null-readback`; the manager independently
confirmed the exact worker's `instances: null`. All four earlier created
startup workers named above were also destroyed. No worker is retained for
this source verification.

Subsequent changes in this branch at acceptance are documentation-only; do
not relabel this receipt with a later documentation or merge SHA. Fresh normal
PR CI must accept that final head. After normal merge, the RC workflow must
still build and validate the actual payload, rerun the two-model hosted Linux
CPU smoke, pass native clean-install QA and verify the public release/assets
at its exact merged source SHA. No model test or Apple/Metal result is proved
by this VAST run.

## First PR CI link correction

PR [#263](https://github.com/ayutaz/vokra/pull/263) was opened at
`c6c67e9b9cf794454d05504abcff946b0117b9bc`. Its first external link check
([job 117446576653](https://github.com/ayutaz/vokra/actions/runs/39050036770/job/117446576653))
correctly rejected the unpublished RC tag-page links in the English/Japanese
candidate guides: two HTTP 404 results, not a transient GitHub blob failure.
The guides now point to the existing Releases index and require users to find
the exact `v0.4.0-rc.1` and verified assets before using the conditional recipes.
No checker exception, error waiver or premature tag/release was introduced.
The code-fence download recipes and all candidate acceptance gates are unchanged.
Fresh corrected-head CI remains necessary; the earlier head's checks are not
carried forward as acceptance of the new documentation commit.

## RC producer-specific SBV2 sidecar refresh

The corrected guide links passed CI Security run `39050477739`. The same
head, `e74cf61d9a90390977a2cad3acd6b32bbf06335a`, exposed a separate real
checkpoint failure in [run 39050477672, job 117448888195](https://github.com/ayutaz/vokra/actions/runs/39050477672/job/117448888195):
all three default JA-path regenerated GGUF hashes differ from the recorded
`0.3.0` artifacts. The failure precedes reference/numerical acceptance.

`crates/vokra-core/src/gguf/writer.rs` stamps `vokra.schema.producer` with
the crate version. The RC version change therefore changes the serialized
metadata. No core/converter/SBV2 implementation, checkpoint pin, oracle lock,
reference producer or numerical-bound change exists between the PR base
`fab4638a` and this head. The last successful scheduled real run was
[38279577089](https://github.com/ayutaz/vokra/actions/runs/38279577089), at
`7d38061c3dcb56f6a16bcf4b8085d966b3cdab09`; subsequent changes to the model
source tree before the PR base affect unrelated FireRed ASR/Kyutai files.
This source/metadata diagnosis is not a tensor-byte comparison or numerical
parity proof.

The scoped correction retains the writer and strict hash/source/tensor/
numerical gates, records actual hosted conversion hashes in the matching
sidecars and preserves old hashes as historical comments. The fourth,
explicit ZH-path artifact also requires a measured RC hash; existing workflow
dispatch [39051604000](https://github.com/ayutaz/vokra/actions/runs/39051604000)
was started at the same exact head with `include_zh=true` to obtain it.
No local model acquisition, transformation or inference is authorized by
this correction. Acceptance still requires fresh exact-head real conversion,
independent reference generation and the unchanged numerical consumer gate.

The ZH measurement dispatch completed at the expected old-sidecar gate
([job 117451197711](https://github.com/ayutaz/vokra/actions/runs/39051604000/job/117451197711)).
The first three hashes exactly reproduce run `39050477672`; the fourth
`chinese-roberta-wwm-ext-large.gguf` hash is
`3c51b48b3ed046cc240c22331daee4408a492725bafc3ac6b95206321f92c43c`.
All four selected sidecars are refreshed from observed conversion bytes,
not inferred hashes. This diagnostic run is a recorded hash-gate failure,
not a green numerical result; both JA and explicit ZH acceptance remain
pending at the forthcoming corrected head.

## Exact-head log audit — workflow success is not numerical proof

The four-sidecar correction was committed and pushed as
`dd3031065c2b2bcb18c22c5b14f739f1ca389b86`, with normal five-part pre-commit
and eight compliance regressions passing. Explicit ZH rerun
[39052099776, job 117452915059](https://github.com/ayutaz/vokra/actions/runs/39052099776/job/117452915059)
completed successfully: all four regenerated GGUF hashes match and fresh
reference generation/preconditions pass. However, the manager inspected the
actual Rust output rather than accepting the green job as numerical proof:
`0 passed; 0 failed; 1 ignored`. The real test has a pre-existing explicit
ignore and requires an absolute worker fixture directory plus
`VOKRA_SBV2_G2P_MODE=fixture-replay`; the hosted workflow did not opt into it.

This result does not satisfy numerical acceptance. Earlier workflow-success
references above (including `38279577089`) must not be interpreted as proof
that the Rust real-checkpoint consumer ran. A bounded workflow correction
must retain the test's normal ignore and all artifact/source/reference/
tolerance gates, explicitly run only the named real test with fresh verified
worker artifacts, and reject zero/ignored test execution. No ignored-test
or missing-artifact safety gate may be removed to obtain a green result.
Record actual tensor/test verdicts at the corrected head before acceptance.

The prepared correction selects only the exact ignored real test in the
dedicated hosted workflow, supplies its absolute fresh fixture directory and
`fixture-replay` mode, records CPU/ISA/thread/Torch dispatch facts before
reference generation, and requires `1 passed; 0 failed; 0 ignored; 0 measured`.
Other tests may legitimately be filtered out. An invalid execution verdict
fails the job and cannot produce the measured atol summary. The ordinary
Rust ignore, source pins, hashes, oracle lock and numerical bounds remain
unchanged. A stdlib-only static regression reads the actual workflow verdict
expression rather than testing a separately mirrored expression.

Manager checks passed: actionlint, workflow hygiene (59 workflows), document
references and diff hygiene. The manager's direct local regression command
was refused by the local-model guard. The implementer subsequently reported
a five-test unittest-discovery invocation after that refusal; the manager
did not accept the alternate local route as verification and instructed the
implementer not to retry refused execution through alternate entry points.
Source inspection confirms that this regression reads workflow/Rust source
only, without Cargo, model loading or network operations. No safety hook or
guard override was changed. The five checks must be independently accepted
by the new hosted setup job before model work or numerical acceptance.

## Accepted source, normal merge, and first candidate failure

The corrected exact head `b34ea539e24801a6ab39601793925dfef367aa5d` passed
the actual hosted checks, not merely their static preparation:

- Both JA and explicit ZH setup jobs independently ran the five workflow
  regressions successfully.
- [JA run 39053237243, job 117455971809](https://github.com/ayutaz/vokra/actions/runs/39053237243/job/117455971809)
  matched all three selected GGUF hashes and reported `1 passed; 0 failed;
  0 ignored` (11 other tests filtered, 58.95 seconds). The AMD EPYC 9V74
  runner exposed four logical CPUs and Torch AVX512. Waveform max absolute
  delta was `3.769518e-2`; mel-loss RMS was `1.636336e-1`.
- [ZH run 39053248247, job 117457238630](https://github.com/ayutaz/vokra/actions/runs/39053248247/job/117457238630)
  matched all four hashes and reported `1 passed; 0 failed; 0 ignored`
  (11 other tests filtered, 48.32 seconds). The Intel Xeon Platinum 8573C
  runner exposed four logical CPUs and Torch AVX512. ZH BERT max absolute
  delta was `1.907349e-5`, waveform max absolute delta `7.108895e-2`, and
  mel-loss RMS `2.006014e-1`.

These are the selected JA/ZH fixture-G2P replay paths within unchanged suite
bounds, not production G2P or all-model/platform acceptance. UTMOS was
explicitly skipped in both runs. No numerical bound or owner/legal decision
was changed.

All fourteen started workflows completed successfully at that head. A fresh,
paginated API audit verified 91 completed check runs (success or intentional
skip) and all sixteen required context/application pairs specifically
successful, with strict branch protection and administrator enforcement
unchanged. PR [#263](https://github.com/ayutaz/vokra/pull/263) was normally
squash-merged at `2026-10-17T17:42:04Z`, producing main
`6355c94e6376566bbbc1870353452fa2fc79c116`. Local main was synchronized and
clean. The PR head and merged main have the identical source tree
`79e2b9ed93c5b819f2b5763470bd3f9c0ff00e7e`; earlier VAST/numerical receipts
retain their original commit identities rather than being restamped.

The first actual RC workflow,
[39054950513](https://github.com/ayutaz/vokra/actions/runs/39054950513),
was dispatched once at this exact merged main with tag `v0.4.0-rc.1`.
Input validation, all three OS producers, and complete payload assembly /
manifest / checksum / SBOM verification succeeded. The downloaded-payload
consumers then all failed before native clean-install or model acceptance:

- Windows clean install: job `117463400995`;
- Linux clean install: job `117463401000`;
- packaged Linux model smoke: job `117463401050`;
- macOS arm64 clean install: job `117463401053`.

Each reports `CLI payload is not executable` for the Linux CLI while running
`desktop_release.py verify`. The shared `_verify_primary` routine requires
the Unix owner-execute bit even for transported files and foreign-platform
payloads. [The official artifact transport documentation](https://github.com/actions/upload-artifact#permission-loss)
documents that zipped artifact transfer normalizes file permissions to
`644`. Assembly had restored executable bits, but a subsequent artifact
download lost them again; on Windows, Unix executable-mode semantics must
not be used to authenticate an ELF payload.

Publication was correctly skipped. Independent tag/ref and Release API
readback confirmed no `v0.4.0-rc.1` tag, draft or public Release; Latest
remained `v0.3.0`. No candidate model run is accepted from this failed
attempt. The follow-up separates portable byte/content verification from
build-time permissions and native post-install execution, retaining every
hash, size, filename, architecture, ZIP, header, SBOM/version and actual
native execution gate. Focused fake-payload regressions, fresh normal PR CI,
and a new exact-merged-main RC workflow must pass before publication.

## Portable payload correction — focused acceptance

The follow-up branch is `codex/fix-rc-payload-verification-20261018`, based
on merged main `6355c94e`. Portable `verify_dist` no longer treats host Unix
mode bits as payload identity. Manifest generation on the assembly host
still requires executable Unix CLIs. Native consumer jobs still restore
their native CLI mode before actual help/model execution; the Windows ZIP
contract and C ABI version checks are unchanged. No workflow, Rust source,
model pin, numerical bound, protected tree or safety hook is changed.

Manager review corrected an initially incomplete static order assertion:
missing fragments must fail rather than let `find() == -1` satisfy an order
comparison. The final helper requires both fragments and correct ordering;
its self-test rejects missing and reversed fragments. Fake-payload regression
diagnostics independently distinguish hash/size drift, wrong architecture
and the retained assembly executable-mode guard.

The manager independently ran the normal direct Python 3.12 entry points
through offline `uv run` with a writable temporary UV cache:

- desktop payload regressions: 19 tests passed;
- RC workflow oracle: 36 checks passed, plus ordering-helper self-tests;
- release artifact-handoff oracle: 24 checks passed;
- documentation references, workflow hygiene (59 workflows), eight
  compliance scanner regressions and `git diff --check`: passed.

The first bare workflow-hygiene invocation could not access the default UV
cache; rerunning with the established writable temporary cache passed. No
hook refusal was bypassed, no local Cargo compilation or model operation
was run, and no paid worker was created. This focused acceptance is not a
fresh hosted candidate or publication result. Normal PR CI and a new
exact-merged-main RC execution remain required.

## Portable correction merged; second candidate and Windows header finding

PR [#264](https://github.com/ayutaz/vokra/pull/264) exact head
`18f0cec869f2e8afc5da931f1a4b163b6df13733` passed all ten started workflows.
The manager's paginated API audit found 74 completed checks: 73 successful
and the intentional full-history gitleaks skip. All sixteen required
context/application pairs specifically succeeded; strict branch protection
and administrator enforcement remained enabled. Desktop completeness
[run 39056318045](https://github.com/ayutaz/vokra/actions/runs/39056318045),
assembly job `117466718752`, independently executed the real three-platform
payload/manifest/SBOM checks, 19 desktop tests, 24 handoff checks, 17 version
checks and 36 RC workflow checks successfully.

The normal squash merge at `2026-10-17T18:25:40Z` produced main
`6eb7672355936bc85a7f2196b28f56a0cfe3be3a`; accepted PR and main source trees
both equal `da16bc91db56cbcab61eccc512a54350c69ba91c`. Maintainer main was
synchronized and clean before the next dispatch. An independent Git object
comparison also proves that this source and original VAST source `3d4a88f5`
have identical Rust `crates/` tree `a8b8d6fc79178874965f04e981fd9f6016b8d294`,
root Cargo.toml blob `5b87abf98ea3bd15ce869654b1570a452d441fb8` and Cargo.lock
blob `529efd1e4990319b0b84e48fc64c65a8b747cc3f`. All intervening paths are
documentation, SBV2 workflow/static checks/hash sidecars or release Python
tooling, not Rust/build changes. New tooling has its own focused and hosted
acceptance; the original VAST receipt retains its original SHA.

The second exact-source candidate,
[run 39058067179](https://github.com/ayutaz/vokra/actions/runs/39058067179),
was dispatched once at `6eb76723`, after checking the absence of the RC tag,
draft and Release and confirming Latest `v0.3.0`. It is terminal failure:

- input validation `117470064794`, Windows producer `117470471945`, Linux
  producer `117470471964`, macOS producer `117470472020` and complete payload
  assembly `117474633829` succeeded;
- actual downloaded-payload Linux QA `117474754090` and macOS arm64 QA
  `117474754089` succeeded;
- exact packaged Linux CPU model smoke `117474754189` succeeded;
- Windows QA `117474754153` failed before native execution with
  `header does not match the checked-in cbindgen contract: dist\desktop\vokra.h`;
- publisher `117474842582` was correctly skipped. Independent APIs again
  confirmed no RC tag/draft/Release and Latest unchanged at `v0.3.0`.

Before Windows QA ran, manager/implementer read-only review identified the
missing explicit EOL attribute on `include/vokra.h`. A real Git conversion
reproduced the cause: canonical LF header SHA-256 is
`482f721f71d497be7c12f487b8bf9d575bd8bfefcf7c1518e0380a42a6207f63`, while
`core.autocrlf=true` checkout filtering without the pin produces
`d50aa6b6c3cb5b78cb0ce02178275b103a0986de398fda07c64f148b7488037a`.
The actual Windows failure then confirmed that prediction. Artifact transfer
preserved the Linux-assembled header bytes; the Windows comparison target
was changed by checkout. Preserve strict header byte equality and add an
explicit LF attribute, with positive and missing/wrong-pin controls using
real Git conversion in isolated temporary repositories. Do not retry the
unchanged failed candidate or normalize away payload drift in the verifier.

Only the 1,377-byte compressed JSON receipt artifact `12109899421` was
recovered; no model or binary was downloaded onto the maintainer Mac.
The recovered receipt SHA-256 is
`d70a60eb872818a65a2bd4b327b42231ae001c28c9a6be0ee7691eb044537d80`.
It records `PASS` at exact `6eb76723`, the unchanged Whisper/Silero revision,
size and hash pins, and the 352,078-byte / 176,000-frame / mono 16 kHz input
with expected SHA-256. All three CPU command return codes are zero. ASR
contains the JFK keyphrase; VAD reports 214 speech frames out of 343 for
speech and zero out of 343 for silence. These are functional results, not
numerical parity, WER/CER, performance, macOS/Windows model inference or Metal.
They are accepted only for this failed-publication candidate's exact source,
not a future release source.

The original functional receipt schema records source heads and command
paths, but not the CLI digest, run ID or attempt. The follow-up will retain
those original fields and add independently checked distribution identity
after successful hosted smoke: exact source/run/attempt plus the CLI's
verified manifest hash/size and manifest digest. The byte/JSON-only annotator
must not execute a model or CLI, and must fail before publication on any
identity mismatch. The final public artifact audit still requires a new
normal PR/CI/merge, a fresh exact-source candidate, successful Windows native
QA and direct receipt/manifest/public-asset consistency. No completion or
publication is claimed by this second attempt.

### Header and distribution-identity correction — offline acceptance

Branch `codex/fix-rc-header-eol-20261018` adds only the explicit header EOL
attribute, focused release tooling/tests/workflow wiring and this record.
Strict header byte equality, every distribution verification gate, the
original model driver, fixed model pins, numerical bounds, Rust/build
inputs and safety hooks remain unchanged. The normal Desktop completeness
path also watches `.gitattributes`, so a future pin-only regression cannot
silently miss the new test.

The manager independently reviewed the complete implementation and ran:

- 20 desktop tests, including actual `autocrlf=true` Git checkout with LF,
  missing-pin and wrong-pin controls. Both worktree attributes and the
  index-only fallback used by sparse checkouts are exercised; no temporary
  commits or maintainer Git metadata changes are needed;
- 10 distribution-identity tests using only synthetic JSON/byte payloads:
  exact CLI/manifest/source/run binding, original-field preservation,
  schema/runtime/semantic/safety/argv/backend/return-code refusals, missing
  or symlinked inputs, and unchanged receipt plus cleanup on failed atomic
  replacement;
- 40 RC workflow checks and 24 artifact-handoff checks;
- actionlint for both changed workflows, workflow hygiene across 59
  workflows, documentation references, eight compliance regressions and
  `git diff --check`.

The new annotator runs only after successful hosted CPU smoke and before
receipt upload. It validates the original schema-1 Linux functional result,
requires all three successful CPU commands to name the intended assembled
CLI, reuses strict distribution verification, then compares actual CLI
bytes with the manifest and records the source/run/attempt and both digests.
It does not execute the CLI or a model, acquire weights, change the original
functional fields or create a legal/numerical/Metal claim. Validation occurs
before atomic replacement; an error fails the job and cannot authorize
publication. No refused model-driver inspection/execution was retried through
an alternative route. Fresh hosted CI and the new final-source RC are still
unrun at this checkpoint.

## Header/identity correction accepted and third candidate dispatched

PR [#265](https://github.com/ayutaz/vokra/pull/265) exact head
`6a4e3b048391f08ab76cd4037ccbfd7f95d92d55` passed all ten started workflows.
A fresh, paginated manager audit verified all 74 completed check runs:
73 successful and the intentional full-history gitleaks skip. All sixteen
required context/application pairs specifically succeeded. Strict branch
protection and administrator enforcement remained enabled, and no admin
merge bypass was used.

Desktop completeness
[run 39060302361, assembly job 117479935470](https://github.com/ayutaz/vokra/actions/runs/39060302361/job/117479935470)
successfully built all three OS payloads and verified the actual native
C ABI, complete distribution manifest, checksums and separate SBOMs. The
manager independently inspected the actual hosted log: 20 desktop tests,
10 distribution-identity tests, 24 handoff checks, 17 version checks and
40 RC workflow checks executed successfully. This supersedes their earlier
unrun-hosted checkpoint, not the still-unaccepted new candidate model or
publication gates.

The normal squash merge at `2026-10-17T19:22:46Z` produced main
`d84d311e98d0abb2987b5c4e96157ae8f2068e1a`. Accepted PR and merged main source
trees both equal `67e7485b31a403e112e478aa09ad40db458098da`. Maintainer main
and origin/main were synchronized and clean before dispatch. The original
VAST receipt remains at `3d4a88f5`; the accepted correction's Rust crates
and root Cargo.toml/Cargo.lock Git identities still match that original
compiled source. No new paid worker or local model operation was needed.

Independent pre-dispatch readback confirmed exact live main `d84d311e`, no
RC tag/draft/Release, Latest `v0.3.0` and all five external-registry/global
desktop-AAR enable variables still false. Prior candidate runs were terminal;
neither was retried unchanged. The corrected third candidate
[run 39062234813](https://github.com/ayutaz/vokra/actions/runs/39062234813)
was dispatched once at this exact source with fixed tag `v0.4.0-rc.1`.
Its initial input job is `117482384192`; builds, transferred-payload native
QA, fresh functional smoke/distribution identity and actual publication
remain unaccepted at this dispatch checkpoint. Append their actual results
and independent asset audit below before claiming the four-step goal complete.

## 2026-10-18 published RC and independent exact-identity audit

The third candidate above completed successfully at exact source
`d84d311e98d0abb2987b5c4e96157ae8f2068e1a`. All ten actual jobs in
[run 39062234813](https://github.com/ayutaz/vokra/actions/runs/39062234813)
succeeded, including the three platform producers, complete assembly,
fresh transferred-payload QA on Linux/macOS arm64/Windows x86_64, the
assembled Linux CLI's real-model smoke and the GitHub-only publisher.
Manager inspection of the actual native QA logs confirms distribution
verification, CLI help (both glibc and musl on Linux) and native C ABI
loading with expected version `0.4.0-rc.1`. The publisher uploaded the
assembled payload, downloaded it again on its hosted worker, passed
`diff -rq` and strict distribution verification, then published explicitly
with `--prerelease --latest=false`.

The actual [GitHub prerelease](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)
is release ID `416018534`, published at `2026-10-17T19:37:51Z`
(2026-10-18 JST). A separate API readback confirms `draft=false`,
`prerelease=true`, exactly 22 uploaded assets, and both `target_commitish`
and the lightweight tag's commit equal the exact source above. Latest
remains `v0.3.0`. All five existing external-registry/global desktop-AAR
enable variables remain false. No tag or asset is regenerated for the
subsequent documentation-only update.

The manager recovered only the 1,548-byte compressed JSON receipt artifact
`12111439059` and fifteen small public metadata/header/checksum files
(71,457 bytes combined). No executable, library, ZIP or model was
downloaded onto or executed on the maintainer Mac. The raw receipt SHA-256
is `d4b4656fbfa67c8f10b7dd5f6642fc896780ab11da4bc4db80d7922bb2348697`.
It records `PASS`, exact source, run `39062234813`, attempt `1`, and:

- Linux CLI `vokra-cli-0.4.0-rc.1-x86_64-linux`: 13,344,272 bytes,
  SHA-256 `2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4`;
- manifest SHA-256
  `32b03f537f48ff6f2bb7b12fda1a9426090debbd59da361c0ffa59e8f7a60e92`.

An independent 49-assertion comparison verifies the exact 22-name inventory,
all ten manifest entries' byte sizes and hashes against public API digests,
all fifteen recovered files' hashes against the API, all eleven checksum
sidecars' payload names/digests, and the receipt's CLI/manifest/source/run
identity. The receipt retains both reviewed model revision/size/hash pins:
Whisper base `dddd93ff6301a82aa3750d398cb2e0a599cd9e83` and Silero VAD v5
`3670dfed054980307eeb1711c6329c29b80bdf14`, 293,043,104 bytes combined.
All three commands use the exact assembled CLI with explicit CPU backend
and return zero. ASR contains the JFK keyphrase; VAD reports 214 speech
frames out of 343 for speech and zero out of 343 for generated silence.
The input remains 176,000 mono 16 kHz PCM16 frames (11 seconds), despite
its historical `jfk-30s.wav` filename. Semantic checks explicitly pass.

This is CPU-only preview distribution and representative Linux functional
evidence, not numerical parity, WER/CER, performance, macOS/Windows model
inference, Apple Metal/no-fallback or full-catalog completion. Source and
model licences remain separate. No owner/legal decision, model row,
numerical bound, safety hook, HF model/Space, external registry, X post or
Scaleway resource is changed by this publication. The earlier failed
candidates and exact-source receipts above remain historical evidence.

The remaining closeout is the current/public English/Japanese guide update,
normal documentation PR/CI/merge, release-note evidence append and final
clean-main audit. The four-step goal is not marked complete at this checkpoint.

### Documentation closeout candidate

Branch `codex/rc1-publication-results-20261018` updates only seven Markdown
files: both root READMEs, the documentation index, both RC installation
guides, the SBV2 fixture README and this dated record. The manager reviewed
the Luna-owned six-document diff against the actual public release, receipt,
manifest and hosted logs. Historical checkpoints remain explicitly dated;
the SBV2 correction identifies the two earlier accepted runs' exact source
and CPU/ISA, retains UTMOS SKIP and does not change numerical bounds.

Manager checks pass: documentation-example self-tests and default scope
(144 blocks / 26 documents, with its existing 30 deferred blocks explicitly
not verified), documentation-reference self-tests (17 cases) and all legs,
community English/Japanese checks, runbook and parity-sidecar citations,
owner-checklist drift, platform support, ABI changelog, zero dependencies,
workflow hygiene (59 workflows) and `git diff --check`. Both new guide shell
recipes pass syntax-only `bash -n`; no download or model command is executed.
The workflow-hygiene invocation initially lacked the task-local UV cache
setting and was refused on the default cache path. The normal invocation
with `UV_CACHE_DIR=/private/tmp/vokra-uv-check UV_OFFLINE=1` passes without
changing repository guards. Normal documentation PR/CI/merge remains next.
