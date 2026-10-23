# CPU preview announcement execution — 2026-10-23

This executes steps 1–4 of the
[announcement plan](cpu-preview-announcement-plan-2026-10-23.md).
X posting, a new Release, additional HF publication and the full Apple campaign
are outside these steps. This is a dated execution record; later results are
appended rather than attributed to earlier runs.

## Initial state and boundaries

Accepted main at execution start:
`92a336318a54f659fa991c129aaa991f46ea6231`.
Existing dirty local guard SHA-256:
`0e268b8ca047d16019ba4df60ef478d44b615d38e73313adc4c7bd392fe93351`.
The guard remains unrelated and is not staged. The protected manifest is not
read, hashed, materialized, changed or committed.

## PR dispositions and work

### Whisper repair — PR #291 opened

Normal pre-commit passed all five compile-free gates. Commit
`41c445f54ed3515b2fb2c240701d42c8ba12b7d7` contains only the converter regression,
license source-citation correction and the completed repair evidence.
[PR #291](https://github.com/ayutaz/vokra/pull/291) was opened against main.
Its fresh CI is in progress; it is not yet accepted or merged at this entry.

The exact Rust blob matches the recorded VAST overlay SHA-256
`91306b190af19313bc37ed6a660aa6488a18e97fa9c61b58aba3904fc51f1568`
on base `92a33631`, with two focused regressions and converter release build
passed. That is the bounded evidence for the corresponding code push's
documented remote-verification path; local workspace/model Cargo was not run.

### PR #290 — hold pending new dependency/compatibility evidence

At reviewed head `6d1f02dde190e013b531dc2b4458c26866975a70`, GitHub reported
mergeable/clean, 69 successful checks and one skip. No review thread was open.
The root independently compared the base and candidate lock metadata:
57 package records become 61, Torch 2.7.1 becomes 2.13.0, Triton 3.3.1 becomes
3.7.1, and the CUDA 12 package set is replaced by a CUDA 13 set including new
CUDA bindings/pathfinder/toolkit and NVSHMEM records. These are lock records,
not a measured installed closure.

The current pyproject still records the old authenticated API-smoke receipt
and a 56-package installed-closure audit. Those results do not certify this
new dependency set. #169/#182 instead use a specific official CPU-index
`2.13.0+cpu` contract; identical Torch version text is not equal closure proof.
Retain #290 open/unmerged until its new selected packages/native/license and
API evidence, or a reviewed explicit stale-evidence disposition, are supplied.
No owner or reference-execution gate is cleared by green general CI. This
reference-only dependency task does not block the bounded Whisper/Silero
published-CLI announcement path.

### PR #287 — metadata correction under review

Head `efaf7d027259bf0e282ab3393b36e4812f9cce9b` remains changes-requested and
behind main despite successful general CI. The initial worktree setup was
refused because the safety hook matched the protected exclusion path. That
checkout operation was not retried. No protected data was materialized.

The independent metadata-only path retrieves only this PR's pyproject and
lock into a temporary candidate directory; it does not create another checkout
or touch the main worktree. Luna prepared canonical lock/package/license-row
bindings and version-specific Pillow evidence. The root required a fresh
primary recheck rather than inheriting the previous version's reviewed label.
PyPI Pillow 13.0.0 reports MIT-CMU, and its tagged LICENSE contains that text.
Diffusers 0.32.2's official metadata and lock entry establish its Pillow
dependency. Root review requires the version-specific MIT-CMU conclusion,
while all transitive/API/owner blocked states remain intact. No candidate has
been published to this PR at this entry.

## Pending accepted-CLI pairing

The official RC1 release API and direct tag ref were freshly checked:
release `416018534`, prerelease/non-draft, source
`d84d311e98d0abb2987b5c4e96157ae8f2068e1a`.
Linux CLI asset `651405463` is 13,344,272 bytes / SHA-256
`2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4`;
its sidecar asset `651405477` is 100 bytes / SHA-256
`093cdf4bad82a2ebcd2d3ac7ed8da6cbba0651e9991e9c969c86cfd1caf72e98`.
Desktop manifest asset `651405529` is 2,453 bytes / SHA-256
`32b03f537f48ff6f2bb7b12fda1a9426090debbd59da361c0ffa59e8f7a60e92`.
No binary/model was acquired on the maintainer Mac.

The smoke wrapper is under Luna implementation and root review. It reuses the
accepted driver at main `92a33631`, blob
`4107f83b64fe59d6a121d051271c9b41e763649b`, with baseline pin blob
`869505eaf238db05074b27599359d3a3fd25f646`, and the new fixed Whisper artifact.
The original driver receipt in the
[dated CPU result](release-0.4.0-preview-cpu-results-2026-10-17.md) supplies the
ASR/VAD command and semantic-check contract. The execution environment must be
recorded honestly; no hosted-runtime marker is fabricated for a VAST worker.

The local guard allowed driver help but refused local pin-validation and a
combined read-only source request. Those operations were not retried through
another source-reading route. Remote execution remains a separate authorized
task, with positive file allowlists, no local weights, no HF token, bounded
cost/time and mandatory destruction. No paid worker has been created for
these steps at the time of this entry.

Guide/current-status updates, the new smoke receipt, fresh CI acceptance,
normal merges and returning to latest main remain required before step 4 is
complete. Old warning text and dated RC receipts have not been promoted to a
new pairing success.

## Progress at 2026-10-23 08:20 UTC

PR #287's reviewed two-file candidate is published as
`1ffe4d9c5c3bfa2332be5d26c01f3257549a8fb0`, a non-force update retaining the
previous PR head and accepted main as parents. The candidate tree is accepted
main plus only the reviewed lock and pyproject changes. Its Pillow 13.0.0
row records MIT-CMU, and root independently checked the lock bytes, 44-row
canonical license rendering/digest and unchanged blocked states. The original
changes-requested review was superseded by an approval on this exact head.
Fresh CI remains pending; no merge is claimed.

The root submitted a changes-requested review on PR #290's exact reviewed
head, documenting the changed CUDA closure and the need for current evidence.
That is a hold on accepting the update, not a model execution or owner/legal
approval. PR #291 also remains in fresh CI with no merge claimed.

### VAST pairing attempt — genuine runtime gate rejected

The first bounded search found no qualifying offer and allocated nothing.
A later read-only search returned a qualifying offer; one fresh worker
`57343389`, label `vokra-rc1-cpu-smoke-4a545131e911`, was created at
USD 0.1477037037037037/hour including the requested 200 GB storage.
The worker verified the public CLI, sidecar, manifest, tag/source, fixture and
the exact accepted driver/pin blobs, then invoked the unchanged driver.

The driver returned exit 1 with
`real release smoke requires GITHUB_ACTIONS=true`. Its failure receipt has
empty model/command lists and no audio processing result: no model smoke was
performed. The driver receipt SHA-256 is
`6a467955cadd3ab1f4a9ef1bfa5e952805ab118d8fb17d843799c936aed14588`.
This refusal is preserved as a failure, not weakened by setting a fabricated
hosted-runtime marker on VAST.

The lifecycle destroyed the worker and individually confirmed absence. Root
independently obtained `instances: null` for `57343389` and a complete empty
instance inventory (`success: true`, `total_instances: 0`, `next_token: null`).
No further paid VAST worker is retained for the hosted-only driver.

The same smoke is being moved to a new manual GitHub-hosted workflow, with
read-only permissions, explicit model consent, exact released-CLI identity,
unchanged driver, fixed two-model pins and JSON-only evidence retention.
The original successful RC1 job `117485177180` in run `39062234813` confirms
the genuine runtime consent flag `VOKRA_RELEASE_CPU_SMOKE=1`; the flag name
was read from primary execution metadata, not the refused source files.
New workflow source must pass review/CI and enter main before its explicit
manual dispatch. No successful new pairing is claimed yet.

## Source PRs at 2026-10-23 08:40 UTC

Fresh CUDA jobs on #287 and #291 passed their Rust tests and then failed the
real-GGUF identity step. The public download still used mutable HF `main`
while the expected Whisper hash/cache key described the pre-repair artifact.
This is a CI input-contract defect; rerunning the same input is not a fix.

[PR #292](https://github.com/ayutaz/vokra/pull/292), initially
`0c090f10d2963c21e9d9dd35e8c006237ead0f9a`, pins both downloads to immutable
revisions and binds the cache to their complete identities. CampPlus retains
its established SHA-256 at revision `7963b5f8e21a75d900e4fc4d6b342ba3f989f6f9`;
the repaired Whisper uses the recorded new revision/hash. Exact sizes and
SHA-256 are checked even on cache hits. YAML syntax and diff hygiene passed;
the refused local synthetic shell check was not retried.

The first fresh required pin-sync check failed because the four newly named
revision/hash environment pins were not registered in `.github/pins.yaml`.
The failure is retained, and the candidate is being corrected by registering
the real artifact identities rather than allowlisting or weakening the check.

[PR #293](https://github.com/ayutaz/vokra/pull/293), initially
`210c2981e81b16101774ab70e9b22c3d076f4b06`, adds five files for the genuine
hosted published-CLI smoke. It is based on #292's reviewed CI fix and must be
refreshed after that prerequisite is accepted. Manager review, seven offline
UV Python 3.12 tests, bash syntax and actionlint passed. The output preserves
the full original driver receipt (including ASR/VAD stdout, audio identities
and functional-only limits), bounded to 64 KiB of JSON. No hosted model
dispatch has occurred, and neither PR is claimed merged at this entry.

### Pin registry and CI queue follow-up — 08:56 UTC

The registered candidate passed hosted pin synchronization on both source
PRs. A further consumer review found that the drift probe does not support
the proposed `artifact` kind. Both entries now use the existing `checkpoint`
kind with their real HF repository IDs; revision/hash/size and mismatch
policies are unchanged. This corrects consumer compatibility rather than
turning a probe error into an exception.

The resulting reviewed heads are #292
`9e80f5f802b5ecbb477368865c60ef30a41df8b2` and #293
`d1e0b3f155544253a53949db62f5ad4db92dddda`. Their fresh CI is pending at
this entry. Local manager Git-blob hashes match the uploaded workflow and
pin registry blobs; no unrelated root-worktree changes were included.

Obsolete #287 Core/Quality runs were cancelled after the known identity
failure; both APIs confirmed completed/cancelled. Its branch still requires
a latest-main update and fresh CI before acceptance. #291's old head reached
80 successes, one CUDA identity failure and five skips; that failure is not
treated as green.

The initial #292 Core run `39926050913` remained queued for over five minutes
after ordinary cancellation and held up its newer concurrency group. A
single force-cancel request was sent for that obsolete run, following the
[GitHub workflow-run API](https://docs.github.com/en/rest/actions/workflow-runs#force-cancel-a-workflow-run).
The first readback showed the request processing; terminal completion is not
claimed at this entry. No latest-head check was waived, no source was
rerun solely because a polling timeout elapsed, and main protections remain
unchanged.

## Integrated acceptance route — 2026-10-23 09:06 UTC

The obsolete Core run `39926050913` reached completed/cancelled after the
single force-cancel request. The new group began real jobs rather than
remaining behind that obsolete run.

On #292's reviewed head `9e80f5f8`, CUDA job `120329573632` in run
`39928007071` completed successfully: immutable GGUF download, file identity
checks and the C ABI backend selector all passed. This is the actual
software-job outcome for that head, not a CUDA-hardware numerical-parity or
full-catalog claim.

The exact four reviewed #291 blobs and two #292 blobs were integrated into
#293, whose head is now `8f2f9433919daed9f70b1d6d7d4eb9220282bfd6`.
Its final 11-file change is the accepted-review unit: converter regression,
audit/publication evidence, fixed CI identities/registry and the five hosted
pairing files. This avoids serially invalidating the same source proof with
multiple prerequisite merges. All final-head CI gates still apply. #291 and
#292 remain unmerged until that integration is accepted; only then may they
be closed with explicit incorporation records. #287 remains independent and
must be refreshed and verified after the shared fix enters main.

Remaining old source-PR runs were individually identified by their exact
superseded heads and sent ordinary cancellation requests to free execution
slots for the integrated head. A request alone is not terminal proof. No
main or unrelated worker was stopped, no final-head check was bypassed, and
no local model/Cargo operation occurred.
