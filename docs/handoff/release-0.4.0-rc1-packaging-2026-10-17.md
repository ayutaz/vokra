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
