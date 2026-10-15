# Kyutai Candle source-metadata CI preparation — 2026-10-15 JST

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
