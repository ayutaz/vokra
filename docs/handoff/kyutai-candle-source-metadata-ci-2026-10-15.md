# Kyutai Candle source-metadata CI preparation — 2026-10-15 JST

## Actual hosted diagnostic — 04:38 UTC

At exact `8fd71d676f5a3f9263e83628c78a04cfe2b90621`,
[run 38681926040](https://github.com/ayutaz/vokra/actions/runs/38681926040),
job `116248167505` ran from 04:38:11 to 04:38:33 UTC. Checkout, interpreter,
all **23 stdlib tests**, pinned Rust and host verification passed. Actual
metadata capture failed at the unchanged retained-record gate with
`projected base lock mutated retained package: backtrace`. This is one
actual source attempt, separate from synthetic rejection messages in tests.
The missing-evidence check also failed as intended; bounded upload succeeded.
No patched-lock or final metadata receipt was produced, and no source/model
gate is accepted by this failed run.

Artifact `11948213954` is 47,201 ZIP bytes according to GitHub, with one-day
retention. Manager recovers exactly the source receipt, original Cargo.lock
and projected base Cargo.lock; there is no model payload. The original
150,274-byte lock independently matches primary Git blob
`70182edd0eb78002fac85d824c6c35b558328b0d`. The projected lock is 39,842 bytes.

| Recovered member | Independently computed SHA-256 |
|---|---|
| Source receipt | `157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec` |
| Original lock | `bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab` |
| Projected base lock | `eafbfd9c58c0b8aa60c4298f4df7e0187c78161a8aa898d0490a98c52faf28de` |

Manager parses all records: 592 original and 161 projected, with no added
package identities. All version/source/checksum and other non-dependency
fields of retained records match. Exactly 45 records differ in dependencies:
39 resolve to identical canonical package identities, while six actually
remove edges. Those six are cc 1.2.19 (jobserver/libc), hashbrown 0.15.2
(allocator-api2/equivalent/foldhash), toml_datetime 0.6.8 (serde), toml_edit
0.22.24 (serde/serde_spanned), tracing 0.1.41 (log) and tracing-core 0.1.33
(valuable). None is silently dismissed as display-only or approved pruning.

Official fixed-version registry JSON for all six matches the recovered
package checksums and supplies feature declarations. Declarations alone do
not establish the actual activated feature set. The initial Cargo metadata
was inspected in memory but not recovered before this failure. The next
bounded diagnostic preserves it before the same strict rejection, without
extra Cargo commands, an equality waiver or a model run. That candidate is
not yet actual hosted evidence. The preparation snapshot below remains
dated history; its not-yet-run statements are superseded only by this record.

## Evidence-preservation correction — reviewed candidate

The corrected collector has SHA-256
`dd1503f496a0e4c9cade24c4177577bed2013a4910974a043da95f799d760fe5`.
Before the unchanged base-lock equality gate, it saves the already obtained
Cargo metadata object and inspected active graph as `initial_metadata.json`.
The receipt is explicitly `INITIAL_METADATA_UNREVIEWED`, with null approval,
`NO_UPLOAD` and model execution `NOT_AUTHORIZED`. The existing JSON size cap,
Cargo command allowlist, fixed source/toolchain and shared deadline remain
unchanged. This adds no Cargo command or permission to accept pruned edges.

The workflow checks and always recovers exactly six bounded paths: the
original five plus this initial metadata receipt. One-day retention and
nonzero capture/missing-evidence failures are unchanged. Manager independently
passes all **25 stdlib tests**, including failure-time evidence recovery and
the exact six-path contract. Those fake-process tests do not establish a
real active feature graph. Actual hosted evidence for this correction is
still pending; do not relabel the original failed run as successful.
Actionlint, workflow hygiene (49 workflows), documentation references and
`git diff --check` also pass for the corrected candidate.

## Original preparation snapshot

Prepared against accepted main
`8bd30d9ab3857f77f3d0fa1edce34c450ea8c102`. This moves the previously
reviewed model-free diagnostic collector onto an ordinary GitHub Ubuntu
runner, not onto the maintainer Mac and not onto another paid VAST worker.
It is not a new reference implementation or an execution approval.

## Why a bounded source diagnostic is next

The [actual source-only VAST attempt](kyutai-candle-reference-readback-2026-10-14.md#actual-source-only-capture-and-retained-lock-rejection)
stopped at `projected base lock mutated retained package: backtrace`.
Its worker/data were destroyed. The actual changed fields were not
recovered; dependency-display relabeling remains a hypothesis, not a cause
established by that log. Repeating a paid lease or weakening record equality
is not the chosen remedy.

The promoted collector is byte-identical to the reviewed diagnostic candidate:
SHA-256 `8bc8fbf2be559264edc67bb49da44288d54c3c43d54cd44c4ccdbbc4cc94bc3d`.
It authenticates 23 fixed official Moshi Rust source files, retains original
lock bytes and rejects changed retained records. Cargo commands are limited
to metadata, dependency trees and the precise memmap2 candidate update.
No upstream compilation, build-script/model execution, model/config/tokenizer
acquisition, license/advisory approval or independent PCM parity is included.

## Workflow and independent verification

The path-restricted workflow uses Ubuntu, read-only contents permission,
pinned actions, Rust 1.99.0 and Python 3.12. Checkout is non-cone sparse with
exactly the workflow and four owned Python files, with credentials not
persisted. Manager read the pinned checkout implementation and confirmed
that sparse checkout selects its internal `blob:none` fetch when the filter
input is absent. Model paths and the protected owner manifest are excluded.

Capture has a shared 480-second deadline and the job a 15-minute timeout.
The collector, not the workflow, creates its fresh owned output directory.
Nonzero capture status remains a failure. Even on failure, upload selects
only the five bounded receipt/lock/metadata paths, never the Cargo cache or
whole source tree, with one-day artifact retention. Missing evidence is not
silently accepted. No provider, HF or project secret is passed to capture.

Manager independently passes all 23 stdlib tests (14 original collector,
five lock diagnostics, four workflow/entrypoint contracts), actionlint and
diff hygiene. The only original operational-test change is a canonical
portable temporary parent instead of Mac-only `/private/tmp`; diagnostic
tests and the collector are otherwise byte-identical. Synthetic main-entry
success is not an actual source graph or successful patched-lock receipt.

## Remaining acceptance boundary

Actual hosted capture is not yet observed. Inspect its bounded field
diagnostics and recovered locks before choosing a correction. Preserve
checksum, source, package/dependency identity and fail-closed review. Then
complete active CPU graph, native/license/advisory and full-PCM/API contracts,
exact owner decisions and authorized real-weight CPU parity. Draft #191's
scope does not approve this distinct Candle reference scope. The 58 public
unresolved rows, frozen 63-row owner scope and final clean-HEAD/Apple packets
remain incomplete; no model acquisition, publication or Scaleway follows.
