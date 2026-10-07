# Desktop release completeness follow-up — 2026-10-07

## Scope and baseline

This follow-up starts from `main` at
`d100d93778191ccab77bd1c37fe3552e3d889758`. It addresses the independent
desktop distribution gap in [deliverables §3.2](../deliverables.md) and
[the platform support matrix §5.1](../platform-support/v1.0-rc-support-matrix.md).
It does not retry or waive external model-package, legal, or upstream waits.

The bounded implementation scope is same-run CPU-only C ABI and CLI artifact
production for Linux x86_64, macOS arm64, and Windows x86_64, including the
Linux musl CLI; deterministic Windows CLI ZIP packaging; separate C ABI/CLI
SBOMs; and a fail-closed complete payload manifest with SHA256 sidecars.
The Windows ZIP member remains `vokra-cli.exe`, as required by the existing
winget template. The legacy macOS library asset filename must not be read as
an x86_64 or universal-binary claim.

The implementation was reviewed in
`codex/desktop-release-completeness-20261007`. At this checkpoint, native
builds and complete three-OS artifact assembly have **not** been verified.
Fake binary fixtures exercise rejection logic only; they are not native
build, installation, runtime, or numerical parity evidence.

## Compile-free verification checkpoint

The manager independently ran the following checks through offline UV/Python
3.12 where Python is required:

- Desktop payload verifier: 16 unit tests passed. Synthetic binary fixtures
  and a mocked C ABI loader test rejection logic, not native execution.
- Same-run release artifact handoff oracle: 24 checks passed.
- Publication configuration oracle: 11 checks passed; publication authority
  was not enabled.
- Version contract oracle: 17 checks passed, including dispatch candidates.
- Workflow hygiene: 48 workflows passed the static checker.
- Platform-support citations: all 53 existing anchors resolved.
- Documentation references, documentation example static checks,
  first-party-only `Cargo.lock`, forbidden-symbol gate, and diff hygiene
  passed. Documentation example Tier C cases remain explicitly unverified.

These results are not a VAST receipt or hosted native CI result. The binary
smoke check uses the compiled workspace version independently of the candidate
asset version, so an allowed workflow-dispatch dry-run does not require a
source version bump.

## Verification boundaries

Local work is limited to source review, standard-library Python regressions
through UV/Python 3.12, formatting, and compile-free repository gates. No
model execution, model download, conversion, or workspace/model Cargo build
is authorized locally.

Linux native release builds and model-free startup checks belong on VAST.
macOS and Windows native release checks belong in the non-publishing hosted
CI workflow. Record their exact source HEADs and actual outcomes before
promoting this scope from implementation to verified distribution tooling.

`DESKTOP_AAR_ENABLED`, tag validation, and dry-run/publication controls remain
unchanged. Neither a packaging change nor green model-free CI authorizes a
release upload or registry publication. No published asset is replaced by
this follow-up. This work does not establish Apple Metal/no-fallback model
parity, full public-catalog coverage, six-platform official support, or GA.

## Next independent work

The Android AAR helper/`classes.jar` and consumer-install gap in matrix §5.3
remains a separate implementation scope. Do not combine it with the desktop
release job while that job is still under review. Owner/device and legal
decisions remain pending where their original ledgers require them.

## Local maintenance

On 2026-10-07, 37 stale Git worktree registrations whose directories were
already absent were pruned after a read-only preview and exact target
validation. All Git refs were byte-for-byte unchanged afterward. No source,
model, evidence, branch, or commit was deleted; no model-storage saving is
claimed from this metadata-only cleanup.
