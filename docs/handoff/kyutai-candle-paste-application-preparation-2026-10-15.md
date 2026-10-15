# Candle source-only paste application review — 2026-10-15

## 2026-10-16 JST: independent offline acceptance of finalization regressions

This continuation supersedes the local-review and missing-regression holds
below, not the absence of actual hosted application, strict advisory,
macro/API compatibility, owner approval or real-weight CPU/Apple evidence.
The original full non-Apple / authorized real-weight CPU / final clean HEAD
and fresh Apple-packet objective remains incomplete.

The owner explicitly permits review of the two application/test files and the
presented offline stdlib command. Root replays the existing eight tests at
producer `63a010aa…` successfully (0.554 seconds). A subsequent supplemental
pattern-search command is refused; it is not retried or used as a reason to
change guards. During the resumed implementation goal, Luna's producer-search
command is also refused and stopped. No alternate read of that producer is
used to evade the refusal. Root had already read both complete files under
the owner's permission. Luna can therefore perform the separately delegated
test-file edits from already reviewed content, without another source search.
The ordinary approved offline discovery command remains the only test runner.

Accepted file SHA-256 values:

| File | SHA-256 |
|---|---|
| `apply_paste_overlay.py` | `63a010aabfaa2e0b974cce634befdf5be3cb6d4533dcc6404351292296319e76` |
| `test_apply_paste_overlay.py` | `137f1f489448618f37a61234a7773fee5c66f3393929762b91f85c0260178740` |
| `kyutai-candle-paste-application.yml` | `a5ab7969ae6114808790a6668d475ddd8c5132aaee1f3d4c5f5d0f5084c29232` |

Luna adds two actual orchestration cases: an `OSError` after the final
review-pending receipt is physically written, and a clock fault after that
same write crosses the observed Cargo operation cutoff. Both must return no
success result and recover a `SOURCE_APPLICATION_BLOCKED` receipt. The latter
uses the original outer deadline's reserved interval, not a reset timeout.
It also confirms that the final write was reached, rather than accidentally
passing on an earlier fixture failure.

The overlap case now covers equality and all three root pairs in both
ancestor/descendant directions. Snapshots include directories, file kinds and
contents. Fail-fast module/input boundaries prevent an accidentally broken
early check from reaching external source or Cargo work. Root rejects the
first test draft's file-only snapshot and possible external fall-through
before acceptance. The old 0.01-second wall-clock failure case is replaced
with a deterministic archive-I/O deadline error; the separate real signal
deadline-interruption test is retained.

Root independently runs all **10 tests / zero failures / zero skips** in
**1.173 seconds** at 2026-10-15 15:12 UTC (2026-10-16 JST), using the exact
offline command recorded below. `actionlint`, workflow hygiene (54 workflows,
36 crons) and diff hygiene also pass. This accepts the model-free local
orchestration/regression preparation, not actual upstream source application.
No validator is mocked green and no production fixture exception is added.
No model/package/Cargo execution, network request, provider operation or
safety-rule change occurs in this continuation. Hosted source application,
recovered evidence validation and strict advisory remain next; source-only
results cannot authorize package/macro or model execution.

This is an implementation-review record, **not an accepted application run**.
The [actual hosted plan](kyutai-candle-paste-overlay-preparation-2026-10-15.md#actual-hosted-plan-and-independent-recovered-data-acceptance)
remains accepted at `bd0384c2`; no source overlay, macro compatibility, clean
strict advisory, model CPU parity or Apple result follows from that plan.
PR #255 remains draft. The original all-row non-Apple, authorized real-weight
CPU and final clean-HEAD/fresh Apple-packet objective remains incomplete.

## Continuation review, before application acceptance

Root independently reads the complete production candidate at SHA-256
`685f46dbace6593919af707f3754b9ae93b0ba93b398ac83a6928018bb503098`.
Review requires three further corrections: deadline coverage for source proofs
and final processing, complete parsed-root-manifest preservation rather than
top-level key equality alone, and rejection of ancestor/descendant input/output
roots before any mutation. These are review findings, not accepted fixes.
The separate test owner must finish a positive real-entry orchestration test
and causal rejection cases without production fixture exceptions.

Independent lightweight checks pass: `actionlint` for the new workflow,
workflow hygiene (54 workflows / 36 crons), zero runtime dependencies,
forbidden symbols, documentation references, community documentation and
`git diff --check`. The initial workflow-hygiene invocation cannot initialize
the default UV cache; using the documented writable temporary cache succeeds
offline. No guard is disabled and no Cargo compilation or model runs.

The subsequent published-`bd0384c2` check readback contains 73 checks:
54 successful, one intentional skip, 16 queued and two failed source-advisory
jobs. Queued work is not terminal failure; the known advisory findings are
not dismissed or retried unchanged. PR #255 is still open/draft/behind.
This is not the check state of the uncommitted application candidate.

## Independent eight-test result and late-finalization hold

Root independently invokes the regular offline stdlib discovery command at
producer SHA-256
`b5d1e9aa51119408d839cdfb0a47e3a35c2533885d957435b040bd5973200e16`
and test SHA-256
`892dbb64ad76a6fa8c310b79e1d466689db1ee30c76405eec0f3679f1c73d82c`:

```sh
env UV_CACHE_DIR=/private/tmp/vokra-uv-check UV_OFFLINE=1 \
  uv run --no-project --offline --python 3.12 python -m unittest discover \
  -s tools/parity/kyutai_candle_reference \
  -p 'test_apply_paste_overlay.py' -v
```

All eight tests pass in 0.499 seconds, with no skipped cases. This includes
the actual orchestration entry point, explicitly synthetic source/network/Cargo
boundaries, real archive/manifest/lock/graph/source validators, the 23/21 source
proofs, 308/292 parent-member proofs and 105/154 package scopes. It does not
authenticate a new upstream capture or execute actual Cargo, package or model
code. The compliance scanner's eight independent regressions also pass.

Further review finds a late-finalization failure: once the operation deadline
expires, failure recovery using that same expired cutoff cannot replace an
already-written review-pending status. Luna prepares producer SHA-256
`63a010aabfaa2e0b974cce634befdf5be3cb6d4533dcc6404351292296319e76`,
reserving at most one second (or one quarter of a smaller budget) for bounded
failure recovery **inside** the unchanged outer deadline. Luna reports the
existing eight tests passing, but this is not root's independent acceptance
of the new candidate or a causal late-finalization regression.

The test owner stops when the safety hook refuses a compound `git status`,
`rg` and `sed` read of the test file as prohibited local model/checkpoint
execution. No test additions or retries follow that refusal; no renamed
runner, path, alternate agent or guard override is used. Root requests narrow
authorization for the two-file review and regular offline stdlib tests only,
without local model work, network activity or safety-rule changes. Until this
is resolved and the final fault cases pass, the implementation remains
uncommitted/unpublished and PR #255 stays draft.

## Review rejects fixture-driven production exceptions

The initial application helper confused the authenticated planner/receipt
schemas and lacked complete lock/source/graph proofs. Subsequent helper
tests did not exercise its actual orchestration entry point; a loop over
sixteen archive-extraction helpers was not an end-to-end application test.
Manager rejects those claims before any hosted application run.

Review also finds production branches recognizing synthetic fixture IDs or
relaxing package counts for them. Those changes are not accepted or published.
Implementation and test ownership are separated: production must keep the
real strict contract, while tests must construct valid Cargo-shaped inputs.
No marker, test flag or invented `path+` lock identity may bypass a gate.

Manager reopens the previously authenticated actual thirteen-file producer
artifact rather than inferring formats from fixtures. Candidate lock is
38,252 bytes, SHA-256
`d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8`.
Whole metadata is 1,148,255 bytes, SHA-256
`959357f105d4118301fc0be3bf94bc6f49944e4df5122cb6a8124bfc2b6c2793`.
Both are independently rehashed before JSON/TOML inspection.

Actual parent lock records use bare `paste` dependency tokens. The existing
local Moshi 0.6.4 record has neither `source` nor `checksum`; source-free
path lock records are not fabricated `path+file:///...` strings. Cargo target
metadata describes macro targets through `kind` and `crate_types`, not the
manifest's `lib.proc-macro` boolean. Registry gemm target paths include full
`src/lib.rs` and `benches/bench.rs` suffixes, not just their basenames.
These observed facts guide the corrected fixtures and production validators.

## Plan paths and actual isolated materialization

The accepted 187,212-byte plan, SHA-256
`677d8d840d43974c9f2ae1e567feb7df49169a3f06425e58f560e83b99b21874`,
contains hypothetical `vendor/name-version` patch paths. It is unchanged.
Actual application must preserve one Moshi workspace member and the full
105-CPU / 154-whole dependency scope, including the two Metal parent versions.

[Cargo's workspace rules](https://doc.rust-lang.org/cargo/reference/workspaces.html#the-members-and-exclude-fields)
state that path dependencies within a workspace are automatically members
and that Cargo searches ancestor directories for workspace definitions.
Manager therefore chooses a separate sibling `workspace/vendor` directory,
outside `workspace/projection`, rather than adding workspace exclusions or
allowing extra members. This is a preventive layout decision, not a claimed
observation from an actual patched Cargo run.

The actual root patch paths become `../vendor/name-version`. Record an exact
sixteen-entry plan-to-materialization mapping and the actual patch snippet.
The package keys, names, versions and proposed parent-manifest bytes are
unchanged. Complete parsed-root-manifest comparison must allow only this
deterministic patch-table append; every prior root field remains identical.
Path verification must use the exact owned sibling directory and preserve
each target's complete relative path, not accept an arbitrary `vendor`
directory with a matching suffix.

## Required acceptance evidence

Before committing/publishing the application implementation, manager must
review the complete source diff and independently exercise the real entry
point with explicit model-free synthetic I/O/process boundaries. No source,
lock, graph, alias, path or count validator under review may be mocked green.
Negative cases must fail at their intended gate, not an earlier fixture error.

The future hosted source-only run must independently authenticate all sixteen
parent archives plus pastey, retain all 308 parent members / 292 unchanged
members, prove the fixed 23-file Moshi projection before and after metadata,
and validate exactly sixteen registry-to-source-free parent transitions plus
the real registry paste-to-pastey transition. Unrelated full records, root
fields, relative ordering, declarations, features and typed edges remain
unchanged except explicitly recorded identity/path correspondences.

Keep raw CPU/whole metadata. Any view needed to validate patched-registry
declarations against their resolved owned paths must have an exact sixteen-
identity proof; it must not erase arbitrary sources or hide dependency drift.
Preserve the `paste` dependency alias while its target becomes `pastey`.
[Cargo metadata's declared-dependency and resolved-edge schema](https://doc.rust-lang.org/stable/cargo/commands/cargo-metadata.html#output-format)
is a source reference, not a replacement for the pinned actual run.

Require one shared at-most-480-second budget, including blocking source I/O,
fixed archive/evidence bounds, exact metadata-only argv, immutable locks after
the first permitted regeneration, and honest bounded partial-failure evidence.
No build script, macro, upstream package code, model, config/tokenizer or
weight executes. No new paid worker, provider secret access or publication
occurs during this review. Passing this source-only application would still
leave strict advisory, executable compatibility, owner/legal/native/operator,
independent full-PCM and real-weight CPU gates separate and unfinished.
