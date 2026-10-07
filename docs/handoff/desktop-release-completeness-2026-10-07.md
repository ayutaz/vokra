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

## Linux native evidence audit — 2026-10-08 JST

The executed source was frozen at
`3468a48c5e2e9ffb66651196991ea1ce28016b8d`; its sole-ref source bundle SHA256
was `f52b1aba23442ed83f5eba9b3007dea6c7bd692a68a0cf4c12914a99fb1c1635`.
The actual LIVE003 remote leaf SHA256 was
`0357b131dba5693ec1b9fc567b85525a6e2924e0f3971a98ccab4b78d0c17e24`.
All 14 ordered remote commands returned zero, including 16 desktop unit tests,
24 handoff checks, three locked release builds (Linux CPU C ABI, glibc CLI,
musl CLI), three binary-identity checks, two real CLI help invocations, and a
real C ABI version smoke check. The source HEAD was unchanged and clean before
and after execution. No model or independent model reference was acquired or
run, and no public artifact was uploaded.

The original remote receipt and controller receipt remain **FAIL**. The remote
receipt writer incorrectly counted the unit suite as zero because buffered
`desktop-release: assemble ok` stdout followed unittest's successful `OK`.
The controller consequently rejected the receipt at `validate-remote`; this
was not a failed Cargo build or native startup check. The original evidence
was preserved, not relabelled or overwritten:

- Remote receipt SHA256:
  `722dd9ca81694af28dbb02274c268f49549e0a3140126eff63eef3f5bf60407b`.
- Log-hash index SHA256:
  `b4ab20880d7483347e3d18631ab360459bf2747dc762a871c21f68eea288a7a4`.
- Original archive SHA256:
  `caafbddb2a9f4f06ab3221d61c78c282ef8ba7c3801f45153327a19e7f55c6c9`.
- Controller receipt SHA256:
  `0f2f06100c40d4a66fd51e93ed5f1609f79661187adf33e7fbd5fa1b85afdeeb`.

The manager independently ran a separate `NON_ORIGINAL_AUDIT`: hard-pinned
original evidence, duplicate-key rejection, all 28 unique regular archive
members and their sizes/hashes, exact source and step bindings, raw command
success markers, and unchanged originals before/after. Validation passed.
Only the known count and aggregate status were corrected **in memory** for
validation; no replacement receipt was written. This records accepted actual
Linux native-check evidence, not an original overall PASS receipt. The helper
writer was separately corrected and remains `NOT_READY`; it was not rerun on
another paid instance.

The earlier LIVE001 failed because `uv` was absent from PATH and performed no
native build. LIVE002 rejected a stale offer before creating an instance.
Both instances actually allocated for this scope, `54659993` and `54661102`,
were destroyed with explicit-null readback, including independent manager
readback; no storage is retained for them. Other projects' instances were not
changed.

For this source-only packaging push, accepted audited VAST native checks and
the independently passed eight-check compliance regression replace the unsafe
local deep pre-push Cargo path. macOS/Windows native builds and complete
same-run assembly still require hosted CI before merge. This checkpoint does
not waive required CI, model/legal gates, or Apple Metal completion.
