# Candle CPU compatibility preparation — 2026-10-16 JST

## Bounded technical authorization, not a result

The owner explicitly approves creation/review/execution of the proposed
model-free Candle CPU compatibility check: **GitHub-hosted Linux, one actual
job, at most 20 minutes**, including upstream build scripts and proc-macro
execution. The owner subsequently explicitly authorizes the identified
read-only source inspection. This supersedes the earlier audit-only execution
boundary in the [current CPU readback](kyutai-candle-current-cpu-evidence-2026-10-16.md),
not its immutable factual index or unperformed-test status.

No model/config/tokenizer/weight acquisition, model construction/forward,
local upstream compilation/model execution, paid VAST/Scaleway worker,
provider/HF credentials, publication or owner/legal signature is authorized
by this technical scope. It does not supply §3.1 sign-off, approve model/data
licenses or change the runtime dependency policy.

Preparation starts from private clean HEAD
`929ed7ea14b5b81c27ed67b092b5db7de356021a`; published PR #255 remains
`04574e71bf9f63c40ad62bf3c351bf65db8e6c51`, base main `f8d859dc`.
No compatibility job has been activated at this preparation snapshot.

## Required actual scope

- Authenticate the [original application-89 eight-file evidence](kyutai-candle-paste-application-result-2026-10-16.md),
  exact producer/run/artifact and all original hashes before using it.
- Reuse the frozen, reviewed `apply_paste_overlay.py` producer. The original
  artifact contains no materialized source tree; do not fabricate a new
  producer or compile a convenient unrelated pastey-only workspace.
- Match final lock SHA-256
  `21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6`,
  exact parent/source proofs and current CPU105 = registry90/path14/Moshi1.
- Preserve `x86_64-unknown-linux-gnu`, `--no-default-features`, the 105-node
  feature/edge fingerprint
  `99364d685eb1898f2ccf442c68fb51d70f0b5e42f90c04dd0171cb07933be51b`
  and actual library scope. A relocated metadata path is not compared to an
  unrelated raw-byte hash; original artifact bytes stay authenticated.
- Compile the full locked CPU library, including actual gemm-c32/c64 and
  exported macro expansion at both 0.17.1 and 0.18.2. Check real Moshi/Candle
  API signatures without creating a model or invoking a forward.
- Use a sanitized compiler environment, separate source-fetch and offline
  compilation phases, bounded process-group cleanup and bounded original
  stdout/stderr evidence. Offline Cargo alone does not prevent arbitrary
  upstream scripts from using the network; actual isolation must be checked.
- Recheck lock, source/vendor files and the CPU graph after compilation.
  Timeout, nonzero compiler status, source/feature drift, incomplete cleanup
  or truncated required evidence must remain non-success.

The implementation is delegated to Luna for three new files only:
`check_cpu_compatibility.py`, `test_cpu_compatibility.py` under the dedicated
Candle reference directory, and
`.github/workflows/kyutai-candle-cpu-compatibility.yml`.
Manager reviews the diff and causal synthetic tests before activation.
Frozen source/audit implementations, warnings and failed legacy findings
are not changed to make this new check pass.

## One-shot activation and current artifact availability

The workflow must not build automatically on every subsequent PR push.
The proposed activation is one unique
`candle-cpu-compatibility-once-20261016` label on PR #255, with
`pull_request: types: [labeled]`, same-repository/PR guards and first-attempt
only. Manager applies that label once, after exact-head review. No re-run or
second compatibility attempt follows from this approval.

[GitHub's official event contract](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request)
supports labeled pull-request events and evaluates the PR merge branch.
Do not substitute `pull_request_target` or inject a token into upstream
compiler subprocesses. The workflow is limited to read permissions; activation
is an external maintainer action, not a token-bearing build-script action.

Fresh read-only GitHub preflight confirms #255 remains open/draft and
mergeable at the head/base above, and its labels are empty. Artifact
**11985400172**, producer run **38758745134**, remains unexpired with its
original 259,361-byte compressed size/digest and expiry
**2026-10-16 15:16:14 UTC**. The artifact-specific generic fetch route rejects
that endpoint; the purpose-built run-artifact API confirms the actual record.
No replacement artifact is claimed or downloaded locally by this preflight.

## Independently authenticated API facts

Manager reads four official fixed-source files at Moshi
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` and independently reproduces
their byte length, SHA-256 and Git blob SHA-1 against the original source receipt:

| Source | Bytes | SHA-256 |
|---|---:|---|
| `moshi-core/src/asr.rs` | 9125 | `4bbe8dc8b12a2520ff0f228ca8c18b004927446d699b2b7712ea4fdec0e80e03` |
| `moshi-core/src/mimi.rs` | 8940 | `ddacdbadd26395cae281b7dd12f03c2b617af6e60be1f15f682bfc11e3049605` |
| `moshi-core/src/streaming.rs` | 8156 | `0068598c3526051ea31a3fdcb70b2ee52922bf775b568a52bc39b4ce48bdee79` |
| `moshi-core/src/lib.rs` | 718 | `58557baa228f3b4d8fb051d4ce591c05861c9e18803e895b28af050716c24279` |

The library publicly reexports Candle/Candle-NN, allowing a Moshi-only API
signature harness instead of adding a new direct Candle feature contract.
Mimi encode/decode-step take mutable Mimi, StreamTensor and StreamMask
references and return a Result of StreamTensor. ASR step-PCM accepts its
Tensor/condition/mask/callback boundary. These are source facts for
compile-only signature checks, not executed model behavior or PCM parity.

## Read-only safety boundary and remaining work

The explicitly authorized applier/two-workflow reads succeed. The listed
`capture_metadata.py` filename does not exist; the frozen helper is actually
`collect_metadata.py`. A further combined read naming that actual collector
is refused; it is not retried through another reader or delegated as a
workaround. Existing authenticated producer interfaces/hashes remain usable.
No hook is edited or disabled. The previously refused planner-file read is
not repeated.

Implementation/review, offline synthetic verification and the one actual
compatibility job remain unperformed at this snapshot. A green compile will
not close model/license/native/operator decisions, independently executed
full-PCM reference, real-weight CPU parity or the original full non-Apple and
final clean-HEAD/fresh Apple-packet obligations. Scaleway stays last and
publication separate. No inventory or frozen owner denominator changes.

## Manager preflight and review boundary

The manager independently confirms PR #255 still has head `04574e71` and
base `f8d859dc`, remains draft, and has no activation label. The exact
artifact API again reports artifact `11985400172` unexpired with digest
`sha256:dca3ea05a0970802e718aef113dd240aa75ff3fdcb21bb35b2c8fd941a69005e`.
This is an availability check, not a second source capture or a compiler run.

Implementation review identifies invalid root-level Moshi API imports and
incomplete subprocess/evidence checks; these are returned to the implementer
before activation. Earlier implementer-only test counts are not manager
acceptance. No compatibility job is consumed by this review.

The reviewed runtime Rust/Cargo/test inputs are byte-identical to recorded
green VAST HEAD `4263f41631bd15288f06d3cd5c6b83ea4a8fdf60`, verified with
`git diff --exit-code` over `crates`, `tests`, Cargo manifests/lock and Rust
toolchain/config inputs. That [remote result](execution-readback-2026-10-14.md#exact-source-replay-passed-and-owned-worker-destroyed)
contains 8,223 passed / zero failed / 105 ignored tests and confirmed
destruction of its owned worker/storage. It does not verify these new
Python/workflow files; their focused tests and review remain separate.

Fresh local read-only gates pass for documentation examples and their
self-tests, document references, community docs, owner-checklist drift,
platform-support anchors, ABI changelog accounting, parity-sidecar citations
and all eight compliance-scanner regression cases. The full runbook-path
gate reports nine existing references to the deliberately excluded
CosyVoice2 LLM tree in this private clone. Do not materialize that protected
tree or rewrite historical citations to turn this clone limitation green.

### Subsequent read-only review refusal

After the correction request, the manager's function/API inventory via
`rg` against `check_cpu_compatibility.py` is refused by the PreToolUse
local-model guard as model/checkpoint execution. It was a source read, not
a Python/Cargo/model invocation. No alternate reader or delegated substitute
is used to evade that refusal, and no safety hook is changed or disabled.
Already-authorized implementation and offline synthetic tests remain
separate, but manager acceptance and one-shot activation remain pending
until the exact source review is permitted. The compatibility job has not
been launched and its one-attempt allowance remains unused.

### Independent offline verification of the corrected candidate

Manager reruns the explicitly allowed fixture-backed `--self-test` through
offline UV/Python 3.12: **12 tests pass, zero failures, zero skips**, in
0.407 seconds. `actionlint` for the new workflow and the repository workflow
hygiene gate also pass. These are synthetic/contract tests, not execution of
the upstream library or a compatibility result. The candidate's three files
remain untracked/uncommitted pending manager source review.

Submitting the same read-only inventory through the normal escalated
permission route is also refused by the local-model PreToolUse guard. No
alternate reader, hook bypass, source push or actual Linux job follows.
Next resolve this exact read-only false positive without authorizing local
model execution, then review the candidate and only afterward activate the
one permitted remote job. No compatibility success is asserted here.

### Read-only blocker resolved; implementation review resumed

The owner's subsequent permission enables a [byte-exact source-read guard
correction](candle-source-read-guard-correction-2026-10-16.md), with all
original model/download rules preserved. Manager reviews the complete guard
diff, independently passes its self-test and successfully executes the
previously refused command. This supersedes the read-only blocker above.
The resumed checker review identifies Rust toolchain environment, mutable
Cargo-cache snapshot and child-cleanup defects. Correct and causally test
these before the one permitted remote job; no activation or compatibility
success is implied by the hook repair or the earlier twelve synthetic tests.

### Reviewed candidate admitted to its one actual job

Manager reviews the corrected checker, tests and one-shot workflow. The
candidate binds the installed Rust 1.99 toolchain separately from fresh
HOME/Cargo settings, snapshots selected registry source trees rather than
mutable Cargo operational caches, checks checkout HEAD, retains original
bounded compiler streams and compares CPU graph/source/lock after building.
The uncalled API harness uses the authenticated Moshi module paths and an
explicit compilation target. No model is constructed or executed.

Manager's independent offline fixture run reports **16 tests: 15 pass and
one Linux-only orphan-process test is skipped on macOS**, in 0.426 seconds.
The hosted job must execute that case; the local skip is not a Linux cleanup
result. Other tests include source-vs-operational-cache mutation and mocked
orchestration/build failure; mocks are not upstream compilation evidence.
Actionlint passes. The previously recorded Rust-input equality with green
VAST `4263f416` is rechecked, and all eight compliance regressions pass again.
The new hooks have their own passing full contract gate. Corresponding source
push may use the documented pre-push skip to avoid forbidden local workspace
Cargo; it does not waive these focused checks or authorize a new paid worker.

Reviewed checker SHA-256:
`d0f7d4f655e79619a7352ea8271d9387ebd70de6917eccc021fe63819f5bc267`.
Reviewed test SHA-256:
`574e01b26027809c5b4e4a4a035c5818279b816d3050024fe5331acf0713629a`.
Reviewed workflow SHA-256:
`3ba3acd6e33637f9560169029144fdc9e320d5ab17cd8dfd65748ae74aa1da0d`.
This is admission, not the actual job result; no retry or second build is
authorized by it.

### Actual one-shot activation

The clean reviewed candidate is committed/published as
`ece1d9b2041ad325f3a1416d168f9d174a6a7450`. Manager verifies that exact PR
head/base, the original unexpired artifact/digest and unused activation label,
then creates the unique label and applies it once at **02:04 UTC**.
[Run 38838522884](https://github.com/ayutaz/vokra/actions/runs/38838522884),
attempt 1 / job `116772969390`, is the only activated compatibility job.
Initial observation is queued with no steps executed; that is not a pass or
a compiler result. The PR title/body now point to the new head and this run,
while preserving older head-bound evidence as history. No reapplication,
retry, second job, model run or paid-worker allocation occurs.

### Actual outcome supersedes the queued snapshot

The [recovered run result](kyutai-candle-cpu-compatibility-result-2026-10-16.md)
records sixteen passing hosted tests and a successful 42.495-second full CPU
library build, but final API-harness failure due to a missing host proc-macro
search path. The overall job remains failed. The reviewed path correction
has seventeen offline tests (one explicit macOS skip), not a second actual
run. The single-job allowance is used; further activation needs a new bounded
scope. No model/owner/Apple/final-packet gate advances from this partial result.
