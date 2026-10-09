# Kyutai STT independent-reference responsibility (2026-10-04)

## Baseline and required result

This dated plan starts from reviewed main
`4d753d81b084f6d2369e330656035042fd00fb9f`. The existing native PCM session,
per-layer KV state, reset and eviction implementation remain in place.
The task is to close their independent-reference and real-weight evidence
gap, not to rewrite the native implementation or duplicate the same math
as an oracle. Implementation is delegated to Luna; the manager owns design
review, evidence acceptance, cloud lifecycle and the PR handoff.

The official source identities are Moshi
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` and delayed-streams-modeling
`4c4f65e147df056adf3346290d64c7b9649b18c9`. Accepted source-only receipts
authenticate bytes but do not prove upstream execution or numerical parity.

## Ordered implementation and execution

1. Add a distinct reference capture path that directly executes pinned
   official `LMGen` and records the main-transformer `RingKVCache.complete()`
   positions, K and V for each step. Preserve original dtype before producing
   comparison values; keep LMGen scheduling state separate from attention KV.
2. Capture initial state, feedback, capacity boundary, eviction and reset.
   Derive ordering and validity from the actual official state, not assumptions
   about native contiguous cache layout. Bind source, model, companion,
   configuration, input audio, environment, shapes and file hashes.
3. Add a Rust consumer that authenticates the exact packet and compares step
   outputs and per-layer KV state with existing native borrowed views.
   Synthetic packet/schema tests are not official reference parity. Reject
   missing or mismatched real evidence rather than reporting a skipped success.
4. Prepare a dedicated reproducible dependency closure. The pinned
   [official Moshi project](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/pyproject.toml)
   specifies Torch below 2.10 and NumPy below 2.3; the existing Torch 2.13
   environment is not accepted merely because an override resolves.
   Review primary package/native license facts and any unused dependency
   exclusion before upstream imports or model acquisition.
5. Bind the complete scope to `KYUTAI_STT_PYTORCH_PCM_ORACLE_CAPTURE`.
   Historical decoder-only, model-license or source-only permission does not
   supply this composite approval. Missing facts remain explicitly blocked.
6. After all execution gates pass, run the official capture and real-weight
   CPU comparison on VAST only. Record CPU/ISA/thread/toolchain fingerprints
   before numerical results. Keep existing registered bounds unchanged;
   diagnose failures on the same worker instead of fitting wider tolerances.
7. Recover bounded evidence and verify hashes, then destroy the owned VAST
   instance and storage. Open the responsibility-specific PR with exact-head
   evidence, coverage and honest unresolved gates. No upload or merge is implied.

## Maintainer and Apple boundaries

The maintainer Mac performs only lightweight inspection, stdlib contract
tests, formatting and shell gates. No model download, import/load/forward,
conversion, workspace or `vokra-models` Cargo execution is permitted locally.
Implementation or synthetic test success does not establish real-weight CPU
parity. CPU proof does not establish Apple CPU/Metal/no-fallback parity.

Scaleway remains the final compute wave after the campaign's non-Apple gates
and exact-head transfer packet are ready. This two-branch task does not finish
the entire public catalog; preserve the denominator and row-by-row status in
the [canonical ledger](mac-pre-scaleway-remaining-tasks-2026-09-05.md) and
[execution plan](mac-cpu-metal-execution-plan-2026-09-07.md).

## Current result boundary

The isolated branch and implementation design exist. Actual official
`LMGen`/KV capture, reset/eviction oracle, complete dependency/native review,
composite approval, real-weight CPU comparison and Apple verification remain
unproven. No fixtures, parity values or approval signatures are invented here.
Append candidate hashes and actual test receipts when available.

The parallel bounded candidate repair added ModuleList-name handling and
logical storage-window tests (11 stdlib mock tests passed). Manager review
does **not** accept this as completion: checking only a container's class name
does not prove it is the trusted official `nn.ModuleList`, and
`bytes(storage())` still allocates the whole backing storage before taking
the logical slice. The logical-byte cap is therefore not a backing-allocation
bound; CPU/contiguous copies also precede the budget check. The legacy typed
storage fallback is not accepted as raw byte evidence. These remaining
identity/resource defects require another bounded implementation correction.
No Torch/model import, real parity, commit or new PR is attributed to those
mock tests. All prior execution, dependency, approval and consumer gaps remain.

## Source-review readback and corrective work — 2026-10-04

This is an audit-start note, not a passing fixture or implementation receipt.
The main-LM reference worktree still starts at `4d753d81`; its code is under
review and no upstream package/model has been executed locally. Existing
native PCM/KV mathematics are preserved. The proposed native seam is a
borrowed, read-only per-layer chronological KV view, not a new cache engine.

The initial WIP could not be accepted for real capture: it used nonexistent
Path ancestry APIs, undefined checkpoint identifiers and a missing capture
constructor argument; it attempted NumPy export of raw BF16 tensors and
reused the narrower decoder-only approval. The flat source receipt does not
contain the assumed `source_path` fields. It must be authenticated through
commit/tree/blob identity and the actual retained source bytes, without
editing the historical packet to match our parser. These are production-path
defects to repair, not grounds to relabel source-only evidence as parity.

The authenticated configuration and pinned attention source establish a
main-transformer ring capacity of **375**, not 3000. The boundary input has
377 frames; captures must include initial state, first/feedback steps,
capacity fill, eviction, reset completion, and the first post-reset step.
Layer identity must come from the registered official main-transformer
caches, not incidental hook invocation order. Official reset changes the
end offset but retains physical backing slots; stale invalid slots are not
compared as valid history. Preserve raw dtype/shape/bytes before conversion,
and identify each artifact by phase, step, layer, range, and digest.

The delegated first implementation wave owns only the reference-tool
directory and its stdlib tests. It must cover real helper/hook/writer/gate
paths through mocks, including malformed receipts, dirty source checkouts,
raw-byte alias safety, reset snapshots, output races, resource bounds, and
wrong approval scope. Mock success proves orchestration/self-consistency,
not official numerical behavior. The current dependency approval remains
blocked; no fabricated approval signature, fallback implementation, or
tolerance is introduced. Any eligible real run needs a dedicated dependency
closure plus primary license/native evidence and the complete composite scope
before model acquisition or third-party import.

The existing decoder real-parity test has `FIXED_ATOL = None`, deliberately
pending a reviewed first measurement. That historical decoder seam is not a
pre-approved streaming-KV bound; measurements cannot be reported as PASS
merely because a CLI/environment supplies a tolerance. No bound is changed
by this plan.

Remaining deliverables are still the full original task: authenticate and
capture the official KV/reset/eviction oracle, add the native consumer without
rewriting the PCM/KV implementation, extend the independent evidence to the
real PCM composite, run approved exact-weight comparisons on bounded remote
capacity, recover verified evidence, and deliver the responsibility-specific
PR. A code-boundary capture alone does not finish PCM or ASR parity. Existing
PR #152 is updated in place; no extra PR is created for these draft repairs.

### Security range conflict discovered in the dependency review

The [official PyTorch advisory GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
lists versions through 2.9.1 as affected and 2.10.0 or later as patched for
the malicious-checkpoint `weights_only` unpickler flaw. The pinned Moshi
constraint `torch>=2.2,<2.10` excludes that stated patched floor. Do not
declare a compatible older wheel secure merely because it resolves.
Likewise, the existing 2.13 audit does not authenticate a different version.

The next decision requires primary evidence for an upstream compatibility
range correction or a separately reviewed, exact source/input security
boundary. Safetensors-only input paths, if established, describe exploit
reachability; they do not make an affected package patched or automatically
approve its license/native closure. No downgrade, security exception,
dependency installation, or real-weight execution is authorized by this note.

### Manager review of the first capture candidate

The reported seven stdlib tests and self-test cover mocks, not official
execution. The first candidate is **not accepted** for execution or a passing
parity claim. Read-only review identified these corrective requirements:

- The pinned transformer constructs `nn.ModuleList` for its layers; the
  candidate's list/tuple-only registration rejects the actual official model.
- Raw export must copy exactly the logical tensor's bytes, including a view's
  storage offset, rather than its entire shared backing storage.
- An approval JSON containing only scope/head/path labels does not bind the
  source, input, dependency/native-license and authorization evidence. The
  current real path has no enforced host/dependency gate before model access
  and third-party imports; its README's blocked/VAST-only claim is not proof.
- Capture must enforce per-step/layer coverage and resource limits before
  copying, record the environment before forward, and distinguish original
  dtype from any FP32 comparison artifact. The configured 512 MiB output cap
  is insufficient for the stated full-cache snapshots and deltas.
- Required source-receipt entries, portable negative tests and output-path
  ownership/race checks are incomplete. A test depending on an untracked local
  primary-source packet is not a reproducible clean-checkout test.

These are repairable orchestration defects, not a reason to rewrite the
native PCM/KV implementation, invent oracle values, widen bounds, or execute
an unapproved dependency closure. The candidate and existing dirty native
accessor remain preserved while existing-PR review and safe local disk cleanup
take priority. Corrective implementation must return to Luna and pass manager
review before any source/model capture or responsibility-specific PR.

## Corrective source preparation and independent local checks — 2026-10-04

The later candidate repairs the earlier orchestration defects; the historical
findings above remain evidence of why the first candidate was rejected.
Manager review and independent stdlib-only execution now cover **34 tests**
(12 source/readiness contract tests and 22 capture/writer/export tests).
The command used Python 3.12 through `uv run --no-project --no-sync` with
`python -S`; no project dependency sync or Torch/Moshi/model import is part
of that test invocation. This is source-contract and mock verification only.

- `contract.py` SHA-256:
  `c7c65d09eac4ce3df6b6d7fdb0562ecb65d3fa48120093848ec068a69a61afa4`.
  Consumed source-receipt files are bound to the actual manifest, exact
  commit/tree/blob identities and bounded bytes; synthetic fixtures are
  portable and do not require the private retained source packet.
- `dump.py` SHA-256:
  `e32df29a9832a0d3fccea4ca8e51e43444a3a92961522d76eb6a1f5f75fc96e7`.
  It checks the original logical byte budget before copying tensors, copies
  only the bounded untyped-storage window, freezes delta bytes before the
  official call, and identifies the trusted official `nn.ModuleList` type.
  Coverage requires 48 layers across 377 warmup steps plus one post-reset
  step, 96 initial/reset snapshots and all six checkpoint artifact sets.
  The derived raw KV output is 1,033,371,648 bytes; the aggregate ceiling is
  1.25 GiB, not the rejected 512 MiB budget. Actual export performance is
  still unmeasured. Output creation is exclusive and manifest-last; failures
  preserve an invalid partial tree instead of recursively deleting it.
- The producer requires the authenticated source packet and records original
  logit dtype separately from its float32 comparison export. Its execution
  gate runs before model authentication or third-party imports. Linux/x86_64
  is a necessary host condition, not proof of VAST ownership. External
  approval receipts are bounded to 64 KiB; the code-side reviewed dependency
  closure set remains **empty**, so even an `APPROVED` label cannot execute
  this candidate. It fails with `BLOCKED_DEPENDENCY_CLOSURE`.

The Rust consumer and read-only native cache accessor are additional source
candidates. Formatting and metadata inspection are not compile/test evidence.
Their checkpoint-free compile and focused structural tests must run remotely
before code push. Raw official BF16 K/V must be reordered by actual valid
absolute positions before comparison; stale reset slots are excluded. KV and
logit diagnostics retain maximum/mean differences and worst KV coordinates.
An unset reviewed numerical bound must fail explicitly, never report a green
parity test merely because measurements were collected.

Actual official execution, independent real-weight CPU comparison, complete
PCM input/Mimi/scheduling comparison, secure compatible dependencies, exact
native/primary-license review and composite approval remain incomplete.
No fixture values, tolerances or approval signatures are supplied by these
mock tests. The original full-PCM responsibility and dedicated PR deliverable
remain open; this safe preparation does not redefine their completion.

## PCM-source and logits-shape review — 2026-10-04

This supplements the historical candidate receipts above. The working branch
still starts at `4d753d81`; it is not committed or remotely Rust-tested yet.
Manager-run stdlib-only tests passed 45 cases (16 contract, 22 streaming-dump,
7 initial PCM-dump cases). These tests do not execute upstream models. The
source-only packet parsers independently accepted the retained packet with
manifest SHA-256 `85223a7ac8b947eaeafa7b2f337a1ac60dea84a75d44ee0c607df72ad25d6342`.
No fixture, numerical verdict, dependency approval or owner signature follows.

Source inspection found a production-path defect in the old LM capture: the
official `on_text_logits_hook` receives `[1,1,1,4000]`, but the candidate
accepted only a vector or a two-dimensional row. Luna corrected the source
shape validation and exports an explicitly reshaped `[1,4000]` comparison
row; both shapes and the source dtype are recorded and required by the Rust
consumer. The manager independently passed `cargo fmt --all -- --check`,
`git diff --check` and the stdlib suite. Cargo compilation/tests are not part
of that evidence. Reviewed candidate identities:

- `dump.py`: `e8870aee2d8a6df08d960ca2926a1774d3d3dc4549c1418fc871b8e0af2dab61`.
- `test_dump.py`: `36a52e04567d094e037b835091c6ae227c0e11aea45e138efbcba13582f864aa`.
- Rust streaming consumer: `a5701ce61135f76f5dee8187f46eace50880e84f5d697b475339ed9d0a81ab41`.
- Native borrowed PCM observer: `e471ffab69c6708a929ade6c4400bd06d8b9a08bdb072279a871b5999135b95f`.

The pinned [official evaluator](https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/stt_evaluate_on_dataset.py)
and [fixed model configuration](https://huggingface.co/kyutai/stt-2.6b-en/blob/a07aec56d22be5589cd0bc8709c75b6cf3e3039d/config.json)
establish a comparison-boundary difference, not a numerical pass:

| Boundary | Official Python evaluator | Existing native PCM session |
| --- | --- | --- |
| Left silence | 1.0 seconds | 1.0 seconds |
| Right silence | delay + 0.5 = 3.0 seconds | delay + 1.0 = 3.5 seconds |
| Final framing | Ceil to a complete 1,920-sample frame | Drain complete hops; discard a final partial hop |
| LM calls per encoded frame | One | Two for the first frame, then one |

The server transport client is a different oracle boundary and does not
justify substituting native padding for the evaluator. The new PCM producer
must retain the evaluator schedule explicitly; the native consumer must
report disagreements rather than silently shifting, dropping or fitting
calls. Existing PCM/KV computation is not rewritten by this preparation.

The initial new `pcm_dump.py` candidate is under corrective review, not
accepted as full-PCM proof. Required corrections include bounded input reads
under file-growth races, an external expected PCM hash, source/comparison
logit shape separation, exact geometry/token coverage, and model-free tests
that drive the production orchestration with official-shaped mocks through
Mimi encode, LM step and joint reset. Actual full-PCM capture, native consumer,
secure compatible dependency closure, composite approval, real-weight CPU
comparison and the dedicated PR remain incomplete. `FIXED_ATOL=None`, empty
reviewed-closure allowlists and `NO_UPLOAD` remain unchanged. Apple campaign
statuses, manifests and public artifact decisions are not updated here.

## Full-PCM consumer corrective review — 2026-10-04

The manager subsequently ran the stdlib-only discovery suite: 52 tests passed
(16 contract, 22 streaming-dump, 14 PCM-dump tests). This is a mutable-worktree
receipt, not an exact committed-HEAD or real-weight execution receipt. The
producer tests use fake tensors and patched imports; their "reference written"
console message is from mocked orchestration, not a generated official packet.
No upstream import, dependency synchronization or local model execution took
place. Producer/consumer schema alignment is still under review.

The first native PCM consumer candidate, SHA-256
`b22ef570725323238a9f829ab61521a285550b92c213a93ea374c8820d638886`,
was rejected by management review. It observed native structure but did not
authenticate or consume the official reference/approval files or perform
official/native numerical comparisons. It also incorrectly applied the
reference-artifact budget to multi-gigabyte model components, checked flattened
KV geometry incorrectly, and did not replay the same padded post-reset frame.
These are substantive gaps, not evidence of PCM parity or a completed consumer.
Luna owns the corrective implementation; existing production PCM/KV computation
is not being rewritten to force agreement with the reference schedule.

The required consumer must authenticate bounded reference, approval, source
and model identities before model access; compare common-frame Mimi codes;
compare logits, greedy tokens and valid chronological KV positions only at
explicitly aligned frame/call/causal-position boundaries; and report every
unmatched call/tail rather than discarding it. Initial/reset/first-post-reset
state and eviction diagnostics remain required. Borrowed KV must be consumed
immediately without retaining 48 layers of full history. The reviewed numerical
bound remains unset, so collected measurements cannot become a parity PASS.

Remote verification preparation also exposed an argparse boundary bug:
`--build-jobs 8` was rejected although integer helper tests passed. Luna fixed
the CLI parser and added subprocess checks for accepted 1/8/16 and rejected
0/17/non-integer arguments. The manager independently passed the offline
controller self-test with the actual `--build-jobs 8` argument. Toolkit test
coverage must be refreshed again after the consumer corrections. No VAST
instance has been allocated for this candidate, and no Cargo compilation,
real capture, model upload, or Kyutai PR has been performed by this receipt.

## Producer metadata and dependency readback — 2026-10-04

After the producer-only schema correction, manager-run stdlib discovery again
passed all 52 cases and the PCM producer self-test passed. The reviewed files
at that point were:

- `pcm_dump.py`: `387a4d12cbbf32f74cfe2b0d41eb0a85cd066e155325daa201dedc4817400a8c`.
- `test_pcm_dump.py`: `d651834720f911ca52e78d1ecac28d4ecc529ed72f1a995424fc4bab58dc0a14`.

The producer now records `lm_call_ordinal=0` for joint steps, logits, tokens
and ring events, and documents warmup checkpoints 0/1/374/375/376 and the
first `after_reset:0` checkpoint. Its official-reference packet is deliberately
distinct from the native four-component packet; neither substitutes for the
other. This source/test receipt does not prove an actual upstream capture.

The documentation reference gate, runbook path-citation gate (1,394 anchored
citations), documentation-example checks and their self-test, zero-dependency
gate, and forbidden-symbol gate passed. The documentation-example checker
reported its 30 deferred toolchain/prose cases; they are not verified examples.
No broad local Cargo or real model test was run. Native consumer implementation
and remote test counts remain under corrective review.

Read-only upstream investigation found that the Moshi main revision remains
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`, and DSM main remains
`4c4f65e147df056adf3346290d64c7b9649b18c9`. The pinned
[official Moshi dependency declaration](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/pyproject.toml)
and [publisher metadata for Moshi 0.2.13](https://pypi.org/pypi/moshi/0.2.13/json)
still require Torch below 2.10. The
[reviewed PyTorch advisory](https://github.com/advisories/GHSA-63cw-57p8-fm3p)
lists versions below 2.10.0 as affected and 2.10.0 as patched. It concerns the
checkpoint unpickler; this is not a finding that the authenticated safetensors
probe exploits that vulnerability. Restricting input reachability is also not
an approved security exception or proof of a patched dependency closure.
No official newer compatible path was established in this investigation.
Overriding package constraints, substituting an MLX oracle, or inferring a
policy waiver from a no-upload scope is not an accepted resolution.

Existing PR #152 was independently read back at
`6e61207a6810a78a84fbecc5c31affc82ebe85dc`: OPEN/Draft, 69 successful
checks, one skipped check, no pending checks. Its dependency/license and
real-execution gates remain open. The original goal remains active; neither
this model-free receipt nor PR #152's green CI completes it.

## Consumer v2 review disposition — 2026-10-04

Manager full-source review rejected the native PCM consumer v2, SHA-256
`8864c11ff99dd3d565b9c2d69be7302104234ee8c1fa41a57598e5312fb9a1b4`.
The candidate added authentication and comparison code but is not executable
acceptance evidence. Source inspection found a format-string argument mismatch,
the legacy `sources` key instead of the producer's `source` key, incorrect
rejection of absolute KV positions 375/376 at eviction, and a 1,250-MiB budget
instead of the producer's exact 1,280-MiB budget. It also conflated the final
warmup tail with the post-reset row and compared ordinal-zero calls without
checking the known causal-position shift. These findings are source review,
not a completed Rust compilation or measurement. A further corrective wave
and producer-compatible metadata/comparison regression tests are delegated.

Separately, the legacy decoder-boundary consumer was hardened to bound JSON
and artifact reads, authenticate the returned bytes rather than rehash a
different path read, recheck descriptor/path identity, and bound artifact-tree
entry/depth traversal. It uses the already required digest utilities through
stdin without adding a runtime crate. The unavailable-first-tool regression
also supports Linux hosts that have `sha256sum` but not `shasum`.
The manager reviewed the changes and verified the frozen source SHA-256
`084c96debdfb4f96daa97562de56a4a58e0eae4fc0e3a56e245545f139e162f8`.
It now defines ten normal tests plus the existing ignored real-weight test;
those Rust tests have not yet run. Oversize/content-substitution/tree-bound
tests are present, but a concurrent-write race was not experimentally replayed.
Manager `cargo fmt --all -- --check` passed; this is formatting only, not
Cargo compilation. No VAST worker was allocated for these source reviews.

## Consumer v3 full-source review — 2026-10-04

The manager read all 2,248 lines of candidate v3, SHA-256
`e505c598840e490703476acf748840ea4d185f8afb7901ad035894532d943772`.
This candidate is **not accepted**. It fixes the top-level `source` spelling,
the exact 1,342,177,280-byte artifact budget, and absolute eviction positions,
but the final diagnostic still has a missing format argument. The actual
producer's `source.pcm` receipt contains DSM/Moshi tree hashes, whereas the
consumer incorrectly validates it as an LM/transformer-hash receipt. A real
producer-compatible packet would therefore be rejected before comparison.

Source review also found that native warmup frame 377 still indexes the
official post-reset code row, event coordinates are not yet bound to binary
offsets/order, and `step_outputs.i64` is authenticated only as a generic
artifact rather than parsed as the exact joint-output stream. The eight normal
tests plus one ignored test defined by this candidate do not exercise a full
producer-compatible metadata parser. This count is source inspection, not a
test result. The corrective scope includes complete delta/checkpoint artifact
association, exact ordered coverage and byte/value bindings, finite bounded
tree traversal including empty directories, and exact native KV geometry.

Luna owns these corrections and substantive positive/negative model-free
regressions; another agent independently reviews the frozen v3 source. The
remote toolkit's expected counts must follow the final corrected candidate,
not this rejected snapshot. The network-free `test-vastai-safe.sh` contract
test passed, including redaction, exit status and destroy confirmation. No
VAST instance, Cargo compilation, numerical bound approval, real reference
capture, upload or new Kyutai PR is claimed by this review.

## Consumer schema reviews v4–v6 — 2026-10-04

Further manager and independent review rejected v4/v5 rather than inferring
producer compatibility from handwritten synthetic metadata. Candidate v4
had SHA-256 `ead0ca41e948e0f19a0297728e5b1a63839b433e775017257b91633d6c113b7e`;
v5 had `ea05e23ade2cbc1693b1649181e709ed3ed5d6a1961af9a405d7d9bedab96052`.
The latter added exact 378-row coverage, LM-output dtype, ordered logits
metadata and execution facts, but still rejected the actual nested source
contract. Those improvements are source findings, not executed Rust tests.

The two identity helpers have different contracts: the outer PCM producer's
`contract.git_identity` returns `repository/revision/origin`, whereas the
decoder helper used inside `source.contract.source_identity` returns
`repository/revision/roles`, with source-file size, SHA-256 and Git blob rows.
The official temperature-zero contract also contains `tie_resolution`.
Validating both identities as the same shape, or requiring only four keys
for every contract row, rejects the actual producer. The earlier independent
review conclusion that these schemas matched was explicitly withdrawn.

Candidate v6, SHA-256
`ce7d8683506bd1062d0c505f9186970fed8e2992c9967501bba63325afce811f`,
corrected that distinction, required the exact four model-file rows and fixed
the LM source-packet hash. Manager review found another definite failure:
it passes 40-character Git blob SHA-1 strings to the 64-character-only
SHA-256 validator, so both the real producer and its new positive synthetic
fixture are rejected. It is **not accepted**. Eleven normal tests and one
ignored real-weight test are defined; none has been run with Cargo.

The implementer overwrote the v5 review-copy path with v6. The old SHA remains
a historical receipt, but that path no longer retains v5 bytes; no continuous
immutable v5 audit is claimed. Subsequent snapshots must use distinct names
and remain unchanged throughout independent review. The next correction must
separate SHA-1/SHA-256 validation and add a small fixture from the actual
source-contract helper's returned schema, with any mock boundary explicitly
identified. Source-contract fixtures are not numerical reference values or
proof of an upstream model/checkpoint run.

The manager's latest stdlib-only discovery run passed 52 producer/mock tests
in 0.652 seconds without dependency sync or third-party imports. Their mocked
"reference written" message is not a real reference artifact. The remote
toolkit's native-module test count is still provisional; no live invocation
is accepted until the corrected source, exact counts and clean committed
HEAD are reviewed. The original independent PCM/KV/reset/eviction and
real-weight task remains incomplete, with unchanged empty closure allowlists,
unset numerical bounds and NO_UPLOAD.

## Actual-helper schema fixture and v9 correction — 2026-10-04

The source-contract identity gap was resolved using the actual existing
decoder helper's AST-derived return rather than a handwritten schema mirror.
The retained packet lacks the DSM TOML body; a separate read-only fixed-revision
[primary config read](https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/configs/config-stt-en-hf.toml)
authenticated 1,016 bytes, Git blob
`75382f8dabeb31832f28c8754afaaf63aaa7b158`, SHA-256
`81f77d642689e1acb276089f62064dab2e71a5532fcac2d3c12563cf4946552c`.
This is a source identity, not model/config execution or legal approval.

Independent source review confirmed v8's separate SHA-1/SHA-256 validators,
actual nested role schema and temperature-zero tie-resolution key. Root then
found a definite test compile defect: the new `include_str!` fixture was passed
as `&str` to the existing byte-slice JSON parser. V8, SHA-256
`9ade808c067ab7408aa11b38048252274ee6a1d0df8277c8fc24c6e549bede5d`,
was therefore not compile-ready. Luna's distinct v9 snapshot corrects that
one call to `.as_bytes()`; live and frozen SHA-256 both are
`b586dd8b30472432eaf9ff9a8aabe41be830bc7e7f04bdbb9a18a27d7ec3de92`.
It defines 13 normal tests and one ignored real-weight test; no Rust execution
is claimed by this source inventory.

The fixture generator now requires an explicit retained-packet path, invokes
the real PCM source-packet authenticator and checks consumed role sizes/hashes
before AST parsing. It creates no fake TOML file. Its only identity mock
injects separately reviewed Git metadata; it imports no Torch/Moshi runtime.
The committed positive JSON remains portable for ordinary Rust schema tests,
while regeneration requires the external authenticated source packet. Root's
stdlib-only regeneration produced the same canonical JSON; a nonexistent
packet was rejected without JSON output. This is source-schema evidence, not
independent numerical values or a real checkpoint run.

Root reran 52 producer/mock tests (0.456 seconds), formatting, zero-dependency
and forbidden-symbol checks successfully. The quoted mock reference-written
message is not a real packet. Production PCM/KV mathematics remain unchanged.
A clean committed candidate and exact-head VAST Rust verification are still
required. Secure compatible dependency/native closure, composite approval,
official capture, independent real-weight comparison and measured numerical
disposition remain the original incomplete responsibility.

## First committed candidate and VAST compile diagnostics — 2026-10-05 JST

The normal pre-commit gates passed for candidate
`c654719e56d3a7d06d02e111eba94cedced3b5c8`. Its complete-history transfer
bundle was verified locally and remotely, SHA-256
`689cbb2ee2df6ba000e7172e1bb0024b1ca979fd09e29335e28c37ab90210c8d`.
An initial SSH-readiness failure occurred before Cargo on owned worker
`54161884`; root independently confirmed its individual null readback and
inventory absence after the old controller's acknowledgement-parser error.
The repaired controller requires actual direct `22/tcp` mapping, bounded
same-endpoint readiness checks, and both owned-ID destroy readbacks; it does
not infer cleanup from a successful mutation response.

Second owned worker `54162678` authenticated the clean candidate and ancestry,
recorded environment fingerprints and passed formatting. The first clippy
step failed with Rust E0034 and E0599: `File::by_ref` was ambiguous with both
Read and Write in scope, and `KyutaiSttWeights.is_synthesized` was called as a
method despite being a public field. Raw diagnostic log SHA-256:
`9661583254ec8f4bd352106de8c9440e0267c910fc2ef03e9e8505e260e04480`;
FAIL_MODEL_FREE evidence SHA-256:
`e831b44a951ea6a98ee493350811d696fcc5df349149fce29bc33a9d37bbf04d`.
Root recovered the logs, and the controller completed destroy on its first
attempt with both individual null and independent inventory absence. No
tests or real-weight comparison passed in this failed attempt.

Luna corrected only these API boundaries and the analogous native bounded
reader, preserving all resource/identity/anti-synthetic gates and test counts.
Root reviewed the three-line diff against the actual first-party APIs.
Legacy source SHA-256 is
`7eb82a58b6e04c64a88e666164eab833a3d5ee7d1214c04bca1456bbe1a4a0c2`;
native source SHA-256 is
`d74c7b8519acb1cc6180b4994df25d79764ebb7fc345749d3e841331336049f3`.
They still require compilation and model-free test execution on a new clean
HEAD. No old-head failure is reused as a passing verification receipt.

An independent source-only investigation located an official Rust/Candle
ASR/cache route, but did not establish a drop-in oracle: all-layer bounded
capture, exact Rust source/model/tensor identities, different PCM schedule,
dtype behavior and a distinct dependency/native closure remain unproved.
The current PyTorch composite/approvals cannot be reused for it. This is an
alternative to evaluate, not a replacement of the original full PCM/KV,
reset/eviction and real-weight task, nor a reason to rewrite native math.

## Corrected candidate lint diagnostics — 2026-10-05 JST

Candidate `95f164f20d1e77a3624bad805a5927790a75312d` passed normal
pre-commit gates and was transferred in a complete-history bundle, SHA-256
`92ae5e8bb3ba3df93ced8664f2bdf0a7cf39977b8876a71520a3abca5dccb857`.
Owned VAST worker `54163567` authenticated the clean HEAD, ancestry, bundle
and toolchain, fingerprinted its environment and passed formatting. Clippy
then rejected the legacy consumer's index-based audio-code loop with
`needless_range_loop`; the earlier type/API errors were no longer reported.
Raw lint log SHA-256:
`109f6c3616cd2d45041240d65bfe234588d23a3e42e5f98a9a2e8abdc37f58f2`.
FAIL_MODEL_FREE evidence SHA-256:
`33b26e823690c6f7e530e7049896a26dfb7f8349de2470c9507c2ab91d74dbb5`.
Root independently matched all 14 recovered raw logs to their evidence
digests. No Rust tests or real-weight comparison passed in this attempt.
The controller destroyed the owned worker on its first attempt and recorded
both individual null readback and independent inventory absence. These
owned-ID receipts do not claim the provider account is otherwise empty.

Luna replaced only the index-based loop with iteration and enumeration.
Root confirmed the authenticated manifest already requires exactly FRAMES
audio rows and N_Q valid codes per row; no truncation, warning suppression,
numerical bound change or native PCM/KV math change was introduced. Legacy
source SHA-256 is
`f2cb3d63c6931b99dba3c545ccd409306e312982bf05a468a419d7101a15be9d`;
the native consumer remains at its preceding recorded source identity.

Disposable verification tooling now collects both required clippy steps
even if the first fails, then stops before tests if either is nonzero.
This is diagnostic aggregation, not a passing-gate exception. Root ran the
actual eight-job CLI offline self-test, leaf smoke test, bash syntax and
ShellCheck successfully; none executes Cargo or a model. The repaired
source still requires a new clean committed HEAD and exact-head remote
Rust verification. The original secure dependency/native closure,
composite approval, independent official capture, real-weight PCM/KV
comparison and numerical disposition remain incomplete.

## Actual library-test compilation and coverage correction — 2026-10-05 JST

Clean candidate `d8a8d7fe2ae4dccd38ace311f42a91682c632082` was tested
on owned VAST worker `54166305` using complete-history bundle SHA-256
`f5607ae5d1f96f04556ccb23d49538aba8d67bedf02a8d564bfd2f6449fd7751`.
The two old clippy commands passed, but the first actual library-test build
failed with five errors: missing `AsBytes` trait for two mmap calls, nonexistent
`JsonValue::as_f64`, `Vec::len` where the JSON API returns a slice, and a
borrowed test contract passed to a by-value `finish` method. Thus the old
`--lib --test <integration-target>` clippy command did not establish that the
native `cfg(test)` module had compiled. Earlier source-only descriptions of
that command as cfg(test) lint coverage are withdrawn; the failed actual
library-test build is the stronger evidence. No Rust test pass is claimed.

Root independently verified all 16 recovered raw-log digests. Actual failed
library-test log SHA-256:
`7e80475437739e8e44e14d9f2e27e9209cc5cb792082f65ccf96d1af44af89cd`;
FAIL_MODEL_FREE evidence SHA-256:
`ec74fd3badd0a9776455b1211de361aacff8e13e8cb3b52683048a0e75cafe30`.
Owned worker and storage were destroyed on the first attempt, with both
individual null readback and independent owned-ID inventory absence.
The current official Moshi dependency recheck still finds the fixed revision
and Torch upper bound unchanged; secure supported real-execution closure and
the complete composite approval remain unresolved.

Luna repaired the four API/type boundaries in the two owned Rust files,
without changing production math, loader/approval guards, test inventory,
or numerical bounds. Verification tooling is being corrected to lint actual
library test targets and bind that coverage to verbose compiler evidence.
A new clean HEAD must pass actual Rust lint and tests before code push or PR.
No passing result from either old lint command is reused as native library-
test coverage, real-weight evidence or independent numerical parity.

## Actual native library-test lint and corrective review — 2026-10-05 JST

Clean source `1924a8ca74ea761991ab0d188b3f66dc03f811d8` was transferred
with complete-history bundle SHA-256
`26ee62f5687e878a09909e043d65ae2a3b6d6bb2b19204225ae9b17d6db5f4e4`.
Run5 worker `54167641` failed bounded SSH readiness before Cargo and was
destroyed with both owned-ID readbacks complete; it has no compiler verdict.
Run6 used a different physical host and owned worker `54169200`, sixteen
build jobs, and the reviewed corrected verification tooling. SSH succeeded.
Legacy clippy passed. Native `--lib --tests --no-deps --verbose` clippy
reached the actual `vokra_models` library test target: one recovered Cargo
Running line binds clippy-driver, the library source and `--test` together.
This establishes the missing lint-target coverage, not a passing verdict.

The five prior API/type errors did not recur. Native lint failed on four
new-code warnings: manual integer ceil, two unnecessarily verbose optional
empty-array predicates, and explicit drop of a non-Drop observer wrapper
in a unit test. No Rust tests ran after the failed lint. Root independently
verified all fifteen recovered raw-log digests, including native lint SHA-256
`f6b250367201079606b8796e9b20cc8255ed6a6a40cc05e2411ad472faade6e2`.
FAIL_MODEL_FREE evidence SHA-256 is
`bb6dfa5277ad5bf17cc5fb2c14eb9f62917e45ae92a13e3c2e36ed18b82aa351`.
Run6 worker and storage were destroyed on the first attempt, with individual
null readback and independent owned-ID inventory absence.

Luna's bounded correction uses equivalent integer `div_ceil`, `is_none_or`
and lexical test-observer scope. Root reviewed the four diff hunks. The
workspace MSRV is 1.85 and these APIs are available within that boundary.
Packet size bounds keep the ceil inputs finite and well below overflow.
No production PCM/KV arithmetic, scheduling, numerical bounds, approval
allowlists or test inventory changed. Formatting and whitespace checks pass;
the correction is not yet compiled or tested. A newly committed clean HEAD
still requires its own remote model-free verification before code push/PR.
Secure compatible upstream closure, complete composite approval, official
real capture and independent real-weight comparison remain open.

## Linux model-free PASS and subsequent OS-CI corrections — 2026-10-05 JST

Clean candidate `a5892f4aa889240cfbde522de03b6390aa512315` passed
the reviewed model-free run7 on owned VAST worker `54169776`, using sixteen
build jobs and Rust 1.95.0. All twenty recorded steps returned zero; root
and an independent reviewer accepted all recovered raw-log SHA-256 bindings.
Actual native-library test-target compilation is present in the verbose
clippy log, not inferred from an integration-target lint command. The five
focused Rust invocations passed 26 tests in total, with two real-weight tests
ignored. Evidence SHA-256:
`2f1150b6607ac37878e97e4785c3975124cec63470374c6b8f176d68b5df5df6`.
The worker and storage were destroyed, with individual null readback and
independent owned-ID inventory absence. This is Linux model-free evidence
for a5892f4a only, not official reference, real-weight or Apple parity.

PR [#191](https://github.com/ayutaz/vokra/pull/191) was opened as a distinct
Draft responsibility on that verified HEAD. Its subsequent actual CI
contradicted cross-platform test portability: macOS default, Metal and
CoreML jobs rejected a synthetic temporary-file path with a symlink ancestor;
Windows rejected a Unix-only positive fixture path and detected a changing
elapsed-time-based fingerprint in the native consumer. Failed job identities
and raw diagnostic log SHA-256 values are:

| Job | Failure boundary | Raw log SHA-256 |
| --- | --- | --- |
| `111475123039` | macOS default library test | `05d443d90f372b055cf273005516896b63ca08b9b6d48b6a6305c70ced567d80` |
| `111475121754` | Metal library test | `f2fa7d4bd86efca639810a83996a7ceb25aeb0958a08f974f3674dfc0bd04c70` |
| `111475121690` | CoreML library test | `f1b42f5371b5d22c95905f189cba8bae76b8ad6648017e499213d53b16d3705b` |
| `111475123149` | Windows library tests | `3bed3d48ba89925bf327b6626afe15062c6ef2030b25f9a3a002ae9e8bded42b` |

These are diagnosed defects, not flakiness; no known-defective old-head job
was rerun. Linux run7 success does not override these OS failures. The
existing CI run `37215524411` still had one live Unity-package job at the
later API snapshot, while the four failures above were terminal.

Luna's bounded corrections, reviewed by the manager, preserve production
PCM/KV math, real-input symlink rejection, Linux/x86_64 real-execution gates,
empty reviewed-closure allowlists, unset numerical bounds and test inventory.
Trusted test temporary parents are canonicalized before joining synthetic
names. Positive packet paths are native absolute paths with JSON escaping.
The native non-Unix fingerprint now uses a checked immutable epoch timestamp
instead of elapsed time; missing timestamps, pre-epoch values and overflow
fail closed. Unix device/inode semantics remain unchanged. An independent
reviewer verified the native diff and guard preservation.

The legacy helper already used epoch time before this correction: the earlier
agent diagnosis that it also used elapsed time was incorrect and withdrawn.
Its actual hardening rejects previously defaulted/truncated timestamp errors,
propagates them through bounded readers, and repairs synthetic temporary
parents. The stability regression unwraps successful identities before
comparison. The non-Unix length/mtime fingerprint is not inode-equivalent
and cannot establish strong replacement detection for equal-size/equal-mtime
files. It is not promoted into a Windows real-weight execution permission;
the existing real consumer remains Linux/x86_64-only.

Current source SHA-256 values before the new correction commit are native
`f47ecab7a3013cec50ba8dcf3ec837480c3f29e7cb1e11ec2301fd36acd0e738`
and legacy
`daeb75daf5022183207cb7c49a2ddaf392d119a5385b66624c05910a84837645`.
Manager-run Python 3.12 stdlib producer/mock discovery passed 52 tests in
0.343 seconds with `UV_NO_SYNC=1`, `uv run --no-project` and `python -S`;
the mocked reference-written message is not an official capture. These Rust
corrections are not yet compiled or tested and require a new clean HEAD's
own remote verification and subsequent OS CI. The normal a5892f4a commit
hook previously synchronized 146 parity dependencies; that side effect was
disclosed in the PR and private review record. Future local hook invocations
carry `UV_NO_SYNC=1` to prevent repeating it. No model import or execution
was attributed to that sync.

PR #152 remains OPEN/Draft at
`1f75166bd0f35eff087394debf67ff729414c65e`, with 69 successful checks,
one skipped and no live checks in the separate API read. Its dependency/
license findings are organized, not approved for execution by green CI.
The original secure compatible upstream closure, native/legal evidence,
complete composite approval, official PCM/KV/reset/eviction capture and
independent real-weight comparison remain incomplete. Neither Draft PR is
merged, and no upload, Apple completion or numerical PASS is claimed.

## Actual OS readback and Windows rooted-ancestry correction — 2026-10-05 JST

Run8 for clean `cc08a7babe415ceb98ee330c758708d5c62bd0b0` passed
all twenty model-free steps and raw-log digest checks, with 26 Rust tests
passed and two real tests ignored. Root and an independent reviewer accepted
the actual native libtest compiler line and five result counts. Evidence
SHA-256 `51585c0ee841d78210ad7af3573f2cf7d028b3bdc678bad3c2e1b54b51845293`;
lifecycle SHA-256 `b1ef57add88e5dd2ee7957c129e22a97cfbefd1df630ccc3797498de01e8795e`.
Owned worker `54173247` and storage were destroyed with both owned-ID
readbacks. Source was pushed to the same Draft PR191 only after acceptance.

Actual fresh CI proved macOS default, Metal and CoreML target jobs successful
on cc08a7ba. Windows still failed one native file-binding test, not the earlier
Unix-only positive fixture: job `111480749364` reported inaccessible path
component with `Incorrect function (os error 1)`. Raw log SHA-256:
`16750e2c9de9c037d401b0da8aa564dad7d051bb95a370a38430da3f12e48425`.
The old component-by-component construction probes a Windows verbatim drive
prefix before its root separator exists. Such a prefix is not the complete
filesystem path. The manager and independent reviewer identified this
remaining platform-grammar defect; old Linux success does not override it.

Luna's next bounded correction checks the complete target with
`symlink_metadata`, rejects symlinks/non-regular files, then checks each
complete parent from `path.ancestors().skip(1)`. Absolute/dot-path rejection
and every metadata-error rejection remain. The [official Rust Path
documentation](https://doc.rust-lang.org/std/path/struct.Path.html#method.ancestors)
defines this traversal through successive parents and marks it stable since
1.28, within the project's 1.85 MSRV. No actual input is canonicalized to
permit a symlink, and no identity, host, approval or numerical gate changes.
The existing test now also rejects an owned parent-directory symlink leading
to an otherwise regular file. Only that owned link, owned test file and empty
owned directory are removed afterward. Test inventory remains 13 normal plus
one ignored. Reviewed native source SHA-256:
`7f795fa7e9e266b3a46a1b667f0509ce2736ca273e67dd94d346a549b5f862fb`.
Formatting and whitespace checks pass; this correction is not yet compiled,
remote-tested or Windows-validated. It needs another clean exact-head run.

Fresh CI also reported two external HF timeout failures. Coverage advisory
job `111480748720` timed out reading the public model-listing API before
classification (raw SHA-256
`cbe026d443387a317613b17933a7594c4a0dc21cb80f95e7baf1c590fcd2cb19`).
A bounded metadata-only probe reproduced TimeoutError at 30.11 seconds with
no weights or token transferred. Documentation-links job `111480749007`
reported zero link errors and two timeouts on existing HF organization links
(raw SHA-256
`3068a8378c434d3cecd6feddb0d66e41f5354336ffc5a60370e5628d52cd5105`).
No exclusion, timeout widening, source gate relaxation, catalog promotion or
workflow rerun is attributed to these diagnoses. Other existing CI handles
remain tracked. The original official/real-weight responsibility is still
open; no merge, upload or full goal completion follows from these repairs.

## Evaluator PCM schedule correction — 2026-10-06 JST

The selected PCM oracle is the authenticated DSM evaluator at revision
`4c4f65e147df056adf3346290d64c7b9649b18c9`, whose retained source identity is
11,674 bytes, Git blob
`684fe5cc5512c6d2e7802ecfd6152f9b7dcf6373`, and raw SHA-256
`2832c048c77aa8ac4baa5535d5b723d17acb3ac91a33f943a3538d4c563d6dcb`. The
evaluator pads 24 kHz PCM with 24,000 left samples and 72,000 right samples,
ceils the complete input to 1,920-sample frames, and performs one Mimi encode
and one LM step per encoded frame, including the first and post-reset frames.

The native crate-private PCM route was corrected to this explicit sample-level
boundary: right padding is 72,000 samples, frame counts use checked ceiling,
the final residual is zero-padded rather than discarded, and the first frame
performs one LM step. Raw text rows 0 through 3 remain retained in the raw
stream; only the evaluator emission view forwards ids greater than 3. The
existing broader `emits_text_token` behavior used by other code-boundary
routes is unchanged.

The pinned `stt_from_file_pytorch.py` receipt remains a distinct official
caller boundary (input ceiling plus 13 prefix and 32 suffix chunks), and the
server transport client is also distinct. This correction does not claim
equivalence between those boundaries, numerical parity, real-weight
execution, owner/legal approval, or publication readiness. The affected Rust
tests and exact-head remote verification remain required; no local model
execution or broad Cargo run was performed for this correction.

## Reset-probe boundary correction — 2026-10-06 JST

Read-only comparison of the authenticated `pcm_dump.py` producer and the
native consumer found a reset-boundary mismatch in the old consumer path. The
producer resets Mimi and `LMGen`, then captures exactly one 1,920-sample frame
from the already padded PCM tensor (`warmup=377`, `post_reset=1`, one LM call
for that frame). A normal native `push_pcm` starts with the session's 24,000
left-prefix samples and drains them before appending a 1,920-sample input;
that path therefore performs 12 prefix-frame calls plus one input-frame call.
Those extra calls were correctly marked unmatched, but they could not provide
the producer's one-frame reset oracle.

The test-only consumer probe now shares the production `drain_one` callback
path and requires a fresh reset state. It executes exactly one padded prefix
frame, invokes the same Mimi encode and LM step/feedback callbacks, preserves
the remaining 22,080 prefix samples, and fails closed on repeated, non-reset,
encoder-error, or LM-error use. The regular production `push_pcm`/`finish`/
`reset` behavior, evaluator padding, token boundary, KV math, alignment
rejection, `FIXED_ATOL=None`, and `NO_UPLOAD` gates are unchanged. Structural
tests distinguish normal 1,920-sample push (13 frames) from the reset probe
(one frame, ordinal zero, first feedback `None`).

This is an orchestration-boundary correction only. It is not a Rust execution,
official upstream capture, real-weight CPU parity result, Apple CPU/Metal
result, license approval, or publication approval.
