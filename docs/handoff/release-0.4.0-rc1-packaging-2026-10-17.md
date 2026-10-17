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
