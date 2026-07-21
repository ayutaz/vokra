# Vokra ABI Changelog (pre-1.0 prerelease window: v0.9 + v1.0-rc)

This file tracks **binary-facing** surface changes between v0.1.0 (the M0/M1
baseline, tagged 2026-07-04) and v1.0 GA (the IF-01 freeze point, owned by
**M5-13** — 2026-07-14 v-label reassignment #2, see the note below; M4-12
before that date). It is **narrower and machine-checkable** vs. the
human-readable `CHANGELOG.md`: only symbols that cross the ABI boundary
belong here.

> **2026-07-14 v-label reassignment #2** (owner decision): M4 = **v1.0-rc**
> (was v1.0 GA), M5 = **v1.0 GA** (was v2.0 GA); the scope through the former
> v2.0 ships as v1.0. The IF-01 freeze executor moves **M4-12 → M5-13**; the
> v1.0 GA tag referenced throughout this file is now the **M5 close** tag.
> v1.0-rc is a semver prerelease (`1.0.0-rc.N`), so the "Pre-1.0 policy"
> below stays in force through the whole rc series — the freeze point moved,
> the policy text did not. At the v1.0-rc tag, M4-12 (re-scoped) snapshots an
> intermediate advisory anchor `docs/abi/vokra.h.v1.0-rc-baseline.symbols`.
> Details: `docs/handoff/m4-12.md` §(f).

- WP: M3-16 (docs/tickets/m3/M3-16-abi-changelog.md).
- Requirements: IF-01 (v1.0 semver freeze), FR-API-01 (single header
  `include/vokra.h`), NFR-MT-03 (changelog automation), NFR-DS-02 (zero-dep).
- Sibling: `CHANGELOG.md` (Keep-a-Changelog, human-facing prose).
- Sibling: `docs/adr/0003-c-abi-design.md` (ownership / error / M0 scope).

## Scope: what belongs in this file

**In-scope** (recorded here on every change):

- **C ABI** — exported symbols in `include/vokra.h` (functions, opaque
  handles, `enum`s, `struct`s with public layout, `typedef`s). This is the
  primary IF-01 freeze target.
- **Rust `pub` surface** of `vokra-core` / `vokra-ops` / `vokra-capi` when it
  is reflected into the C header via cbindgen.
- **GGUF metadata schema** under the `vokra.*` prefix — chunk names, key
  names, value types. Model files are content-addressed by these chunks, so
  a rename is a compatibility break for on-disk artefacts.

**Out-of-scope** (recorded in `CHANGELOG.md` only):

- `vokra-server` HTTP compat APIs (OpenAI-Whisper / vLLM / piper-plus /
  Wyoming). These live in the isolated `integrations/vokra-server` workspace
  and are versioned independently.
- CLI flags, internal Rust API changes not exposed via cbindgen.
- Documentation, tests, tooling.

## Pre-1.0 policy (prerelease semver)

Up to and including the v1.0 GA tag the ABI is **not frozen** (see the
STABILITY block at the top of `include/vokra.h`, ADR-0003, and IF-01):

- v0.9.x may add, remove, rename, or change signatures of any exported
  symbol.
- The single hard rule is that **every such change lands with an entry in
  this file, dated on the day the PR is opened**. `scripts/check-abi-changelog.sh`
  enforces this: if the current `include/vokra.h` differs from the active
  gate anchor (`docs/abi/vokra.h.v0.9-baseline.symbols` during the v0.9
  window, rotated to `docs/abi/vokra.h.v1.0-rc-baseline.symbols` at M4-12)
  and this file does not have an entry dated today, the script exits
  non-zero.
- At v1.0 GA (M5-13; M4-12 before the 2026-07-14 reassignment) the baseline
  is re-anchored to that release, the freeze commitment is written into
  `include/vokra.h`, and post-1.0 breaking changes require a major bump.

### CI posture of the three ABI gates (X-08, 2026-07-20) — ADVISORY until M5-13

`scripts/abi-diff.sh`, `scripts/check-abi-changelog.sh` and
`scripts/rust-public-api-list.sh` were unwired from CI until X-08. They now run
in the `abi-surface (advisory)` job of `.github/workflows/ci.yml`, which sets
`continue-on-error: true`.

**That job must stay advisory until M5-13.** Promoting these three from
advisory to a branch-protection required check *is* the content of M5-13
(`docs/milestones.md` §9), which executes together with the IF-01 freeze at the
v1.0 GA tag. X-08 deliberately wired them advisory-only so the progression is
one step at a time: unwired → advisory (X-08) → required (M5-13). Had X-08
promoted them, M5-13 would have had nothing left to execute. The cool-off
posture mirrors `gpu-vulkan-parity.yml` and the platform-support drift step in
the `license` job.

Known state at wiring time: `rust-public-api-list.sh` is **already red** on
`13a2a6e` (53 added / 13 removed lines vs.
`docs/abi/vokra-rust-public-api.v1.0-rc.list`), from surface added in `ff12104`
without a snapshot rotation. X-08 did not rotate it — that is M5-13/IF-01's
call — and the advisory posture keeps the red from blocking PRs. See
`docs/adr/X-08-ci-gate-completion.md` §2 and §7-(4).

## Entry schema

One `###` heading per **PR-day + version**. Under it, a table of the
individual symbol deltas. Fields are:

| Field       | Meaning                                                          |
| ----------- | ---------------------------------------------------------------- |
| Date        | ISO 8601 (YYYY-MM-DD), the day the PR that ships the change is opened. |
| Version     | Semver of the release the entry rolls into (e.g. `0.9.0-dev`, `0.9.1`). |
| Crate / area| `include/vokra.h`, `vokra-capi::session`, `gguf:vokra.frontend.*`, ... |
| Symbol      | Function name, struct name, `enum` variant, or GGUF key.         |
| Kind        | `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security` / `Breaking`. |
| Signature   | Full normalized declaration (or key + type for GGUF chunks).     |
| Rationale   | One sentence — link the WP/ticket ID.                            |
| Breaking?   | `yes` / `no`. Pre-1.0, `yes` is permitted; post-1.0 requires major bump. |
| PR          | `#NNN` — the merge PR.                                           |

Order within a day: `Removed` / `Breaking` first, then `Changed`, then
`Added`, then `Deprecated` / `Fixed` / `Security`. Sorted alphabetically
by symbol inside each kind.

## Baseline snapshot: v0.9.0-dev (2026-07-09)

This snapshot was the `scripts/check-abi-changelog.sh` diff anchor for the
entire v0.9 window, captured on the merge day of PR #3 (2026-07-08, M2
rollup). At the v1.0-rc tag (M4-12, 2026-07-15) the active gate anchor
rotated to the v1.0-rc baseline below; this file stays on disk as the
v0.9-window historical anchor (`scripts/abi-diff.sh --anchor v0.9`), so the
0.9 → 1.0 delta can still be rendered at the M5-13 freeze.

- Anchor file: `docs/abi/vokra.h.v0.9-baseline.symbols`
- Anchor version: `0.9.0-dev` (workspace `Cargo.toml` still reads
  `0.1.0-alpha.0`; the bump to `0.9.0-*` is scheduled for the M3
  tag-preparation WP, not this one)
- Header commit: HEAD of `feat/m3-plan-and-wave1` at anchor time
- Exported C function count: **14**
- Public typedefs (enums, opaque structs, value structs): **5**
- Exported functions (sorted):
  - `vokra_asr_transcribe`
  - `vokra_audio_free`
  - `vokra_last_error`
  - `vokra_session_create_from_file`
  - `vokra_session_destroy`
  - `vokra_session_retain`
  - `vokra_stream_destroy`
  - `vokra_stream_open`
  - `vokra_stream_poll`
  - `vokra_stream_poll_events`
  - `vokra_stream_push_pcm`
  - `vokra_string_free`
  - `vokra_tts_synthesize`
  - `vokra_version`
- Public typedefs (sorted):
  - `enum vokra_event_kind_t`  (variants: `VOKRA_EVENT_UNKNOWN=0`, `VOKRA_EVENT_SPEECH_PROB=1`, `VOKRA_EVENT_TOKEN=2`)
  - `enum vokra_status_t`      (10 variants, `VOKRA_OK=0` .. `VOKRA_ERROR_OTHER=9`)
  - `struct vokra_event_t`     (`{ vokra_event_kind_t kind; uint32_t a; float b; }`)
  - `struct vokra_session_t`   (opaque)
  - `struct vokra_stream_t`    (opaque)

## Baseline snapshot: v1.0-rc (2026-07-15)

The v1.0-rc-tag snapshot of the narrow C ABI, captured by **M4-12** (re-scoped
by the 2026-07-14 v-label reassignment #2 — this WP records the rc baseline
and keeps the gate **advisory**; the IF-01 freeze itself fires at v1.0 GA =
**M5-13**). This is now the anchor `scripts/check-abi-changelog.sh` diffs the
working-tree `include/vokra.h` against for the rc window.

**This is a recorded, diffable advisory baseline — NOT a frozen one.** The
"Pre-1.0 policy (prerelease semver)" section above stays in force through the
whole `1.0.0-rc.N` series: any add / rename / remove of an exported symbol is
still legal, and still requires a dated entry in `## Entries` below. The freeze
(and the advisory → required CI flip) is M5-13's action at the v1.0 GA tag
(`docs/handoff/m4-12.md` §(b)(d)(f)).

- Anchor file: `docs/abi/vokra.h.v1.0-rc-baseline.symbols`
- Anchor version: `1.0.0-rc.1-dev` (the workspace `Cargo.toml` version bump to
  `1.0.0-rc.*` is scheduled for the M4 tag-preparation step, not this WP)
- Header commit: HEAD of `feat/m4-plan-and-wave1` at rc-snapshot time
  (`41a5ad1`). M4-12 changes only the `include/vokra.h` STABILITY comment,
  never a FUNC/TYPEDEF symbol, so the extracted symbol set is stable across
  this WP's own header regeneration.
- Delta vs. the v0.9 baseline: **+18 functions, +6 typedefs, 0 removed,
  0 changed** — the M4-02 (`vokra_session_create_from_bytes`), M4-03
  (`vokra_aec_*`) and M4-06 (`vokra_s2s_*` + `vokra_model_attribution`)
  additive surfaces, each recorded in a dated `## Entries` section below
  (reconciled by `scripts/abi-diff.sh --anchor v0.9`: every delta maps to an
  entry, 0 unrecorded). This +18 is measured against the **15**-symbol anchor
  file (`docs/abi/vokra.h.v0.9-baseline.symbols`), not the 14-function prose
  list under the "Baseline snapshot: v0.9.0-dev" section above: that list is
  the 2026-07-08 PR #3 capture instant, whereas `vokra_stream_interrupt`
  (M3-14, the 2026-07-09 entry below) is the +1 that grew the anchor to 15 the
  next day — so 15 + 18 = **33** (the count below), not 14 + 18 = 32.
- Exported C function count: **33**
- Public typedefs (enums, opaque structs, value structs): **11**
- Exported functions (sorted):
  - `vokra_aec_create`
  - `vokra_aec_destroy`
  - `vokra_aec_process`
  - `vokra_aec_ref_push`
  - `vokra_aec_ref_writer_destroy`
  - `vokra_aec_reset`
  - `vokra_asr_transcribe`
  - `vokra_audio_free`
  - `vokra_last_error`
  - `vokra_model_attribution`
  - `vokra_s2s_duplex_destroy`
  - `vokra_s2s_duplex_open`
  - `vokra_s2s_frame_hop`
  - `vokra_s2s_interrupt`
  - `vokra_s2s_interrupt_destroy`
  - `vokra_s2s_interrupt_handle`
  - `vokra_s2s_pull_audio`
  - `vokra_s2s_push_mic`
  - `vokra_s2s_sample_rate`
  - `vokra_s2s_text`
  - `vokra_session_create_from_bytes`
  - `vokra_session_create_from_file`
  - `vokra_session_destroy`
  - `vokra_session_retain`
  - `vokra_stream_destroy`
  - `vokra_stream_interrupt`
  - `vokra_stream_open`
  - `vokra_stream_poll`
  - `vokra_stream_poll_events`
  - `vokra_stream_push_pcm`
  - `vokra_string_free`
  - `vokra_tts_synthesize`
  - `vokra_version`
- Public typedefs (sorted):
  - `enum vokra_aec_status_t`  (variants: `VOKRA_AEC_CANCELLED=0`, `VOKRA_AEC_PASS_THROUGH=1`, `VOKRA_AEC_PARTIAL_REFERENCE=2`, `VOKRA_AEC_RESET=3`)
  - `enum vokra_event_kind_t`  (variants: `VOKRA_EVENT_UNKNOWN=0`, `VOKRA_EVENT_SPEECH_PROB=1`, `VOKRA_EVENT_TOKEN=2`)
  - `enum vokra_status_t`      (10 variants, `VOKRA_OK=0` .. `VOKRA_ERROR_OTHER=9`)
  - `struct vokra_aec_config_t`     (`{ uint32_t sample_rate; size_t frame_size; size_t filter_length; size_t ref_queue_capacity_samples; }`)
  - `struct vokra_aec_ref_writer_t` (opaque)
  - `struct vokra_aec_t`            (opaque)
  - `struct vokra_event_t`          (`{ vokra_event_kind_t kind; uint32_t a; float b; }`)
  - `struct vokra_s2s_duplex_t`     (opaque)
  - `struct vokra_s2s_interrupt_t`  (opaque)
  - `struct vokra_session_t`        (opaque)
  - `struct vokra_stream_t`         (opaque)

## Entries

### 2026-07-21 — 1.0.0-rc.1-dev (M5-05: consent manifest schema + structural validator — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
**untouched** (`scripts/gen-c-abi.sh --check` = no diff; no `vokra_consent_*` /
`vokra_voiceclone_*` symbol exists). No GGUF metadata schema is added. M5-05
adds the signed-consent-manifest surface to `vokra-core::compliance`
(`docs/legal-compliance.md` §3.3 schema): a `ConsentManifest` struct + a
`ConsentScope` enum, a zero-dependency structural validator
(`ConsentManifest::parse`, via `vokra_core::json` — no `serde`, NFR-DS-02), a
`SignatureStatus` enum, and a consent seam on the existing
`SpeakerEmbeddingPolicy` (`authorize_embedding_for_tts`). Consumed by the
separate `vokra-voiceclone-experimental` binary (FR-CP-04); core keeps voice
cloning unrepresentable (`VoiceCloningPolicy::Disabled`-only).

**Honesty boundary recorded on purpose:** `SignatureStatus` has **no `Verified`
variant** — core performs *structural* observation of the `signature` field
(present / absent), never a cryptographic verification. Real signature
verification is an owner-chosen trust-root mechanism outside core (M5-05-T04);
and the watermark forced-embed completion leg stays UNMET because
`WatermarkConfig::backend_status()` remains `Deferred` (2026-07-04 drop) — this
WP does **not** flip it (see `docs/adr/M5-05-watermark-dependency.md`).

M5-13 relevance (why this is recorded here): these are additive **Rust** public
items with **no C surface**, so `scripts/check-abi-changelog.sh` does not gate
on this entry (no C symbol changed). `scripts/rust-public-api-list.sh` picks
them up (`vokra-core::compliance::consent::*` + the new
`SpeakerEmbeddingPolicy` method); as with the M5-01/02/03/06 entries above, the
`docs/abi/vokra-rust-public-api.v1.0-rc.list` snapshot is **not** rotated by
this WP — snapshot rotation is the M5-13/IF-01 freeze owner's action. All items
are additive (existing signatures unchanged; the two new enums are
`#[non_exhaustive]`), Breaking? = no.

| Crate / area                        | Symbol                                            | Kind  | Signature / note                                                                              | Rationale                                                                              | Breaking? | PR    |
| ----------------------------------- | ------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------- | ----- |
| `vokra-core::compliance` (`consent`)| `ConsentManifest`                                 | Added | `pub struct ConsentManifest { voice_owner_name, consent_scope, grant_date, signature, vokra_session_id }` | signed consent manifest schema (FR-CP-04, `docs/legal-compliance.md` §3.3), WP M5-05    | no        | (TBD) |
| `vokra-core::compliance` (`consent`)| `ConsentScope`                                    | Added | `pub enum ConsentScope { Commercial, Personal, Research }` (`#[non_exhaustive]`)               | consent scope token (§3.3), WP M5-05                                                     | no        | (TBD) |
| `vokra-core::compliance` (`consent`)| `SignatureStatus`                                 | Added | `pub enum SignatureStatus { Present, Absent }` (`#[non_exhaustive]`; **no `Verified`** — structural only) | honest signature boundary (core does not verify; owner trust-root), WP M5-05            | no        | (TBD) |
| `vokra-core::compliance` (`consent`)| `ConsentManifest::parse`                          | Added | `pub fn parse(bytes: &[u8]) -> Result<Self>`                                                   | fail-closed structural validation via `vokra_core::json` (NFR-DS-02, FR-EX-08), WP M5-05| no        | (TBD) |
| `vokra-core::compliance` (`consent`)| `ConsentManifest::signature_status`               | Added | `pub fn signature_status(&self) -> SignatureStatus`                                            | structural signature observation (not verification), WP M5-05                            | no        | (TBD) |
| `vokra-core::compliance` (`consent`)| `ConsentScope::{from_token, as_token}`            | Added | `pub fn from_token(&str) -> Option<Self>` / `pub fn as_token(self) -> &'static str`            | scope token round-trip, WP M5-05                                                         | no        | (TBD) |
| `vokra-core::compliance` (`level`)  | `SpeakerEmbeddingPolicy::authorize_embedding_for_tts` | Added | `pub fn authorize_embedding_for_tts(self, consent: Option<&ConsentManifest>) -> Result<()>` | wires the reserved `RequireConsent` policy to the consent type (§3.2), WP M5-05          | no        | (TBD) |

### 2026-07-21 — 1.0.0-rc.1-dev (M5-03: IoT Tier 3 no_std Silero VAD — new `vokra-vad-micro` crate, Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
**untouched** (`scripts/gen-c-abi.sh --check` = no diff; a grep for `micro` /
`nostd` / `silero` in the header matches **0** new symbols). No GGUF metadata
schema is added. M5-03 splits the Silero VAD v5 forward core out of
`vokra-models::silero_vad` into a new `#![no_std]`(+`alloc`) crate,
**`vokra-vad-micro`**, so it cross-compiles for bare-metal Cortex-M55
(thumbv8m, IoT Tier 3 / NFR-PT-03) without pulling in the std-heavy
`vokra-ops` / `vokra-backend-cpu` (ADR `docs/adr/M5-03-iot-tier3-nostd.md`
§(a), topology 案1). The std `vokra-models::silero_vad` is now a thin veneer
that depends on and re-exports it.

M5-13 relevance (why this is recorded here): the new crate adds a **Rust**
public surface but **no C surface**, and it introduces a **feature-cfg
dimension** (`std` default-ON; `--no-default-features` = `#![no_std]`) that
M5-13's freeze snapshot must account for. `scripts/rust-public-api-list.sh`
scans only `vokra-core` / `vokra-ops` / `vokra-capi`, so this crate does not
appear in that snapshot; the M5-13 owner decides whether to extend the
snapshot to `vokra-vad-micro` before the freeze. The `std`/no_std split does
**not** change the default (std) build's Rust surface of any existing crate —
`vokra_models::silero_vad::{SileroVadV5, SampleRate, wav::read_wav_f32}` and
`SileroVadV5::{from_gguf, open, supports, forward_chunk, open_stream}` are all
source-compatible (`SampleRate` is now a `pub use` re-export of
`vokra_vad_micro::SampleRate`, the identical type). No C ABI is added or
changed, so `scripts/check-abi-changelog.sh` does not gate on this entry.

The Wave-1 `std` gate on `vokra-core`'s public modules
(session/stream/safetensors/… behind `#[cfg(feature = "std")]`) was recorded
under the v1.0-rc baseline; Wave 2/3 add no further `vokra-core` gating.

| Crate / area                     | Symbol                                                      | Kind  | Signature / note                                                                                          | Rationale                                                                                     | Breaking? | PR    |
| -------------------------------- | ---------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------- | ----- |
| `vokra-vad-micro` (new crate)    | `SampleRate` / `SileroWeights` / `RateWeights` / `LstmState` | Added | `#![no_std]`(+alloc) crate; `SileroWeights::{from_gguf, rate, forward_chunk}`, `run_frame`, stage fns     | no_std Silero forward core for Cortex-M55 Tier 3 (SRS §6, NFR-PT-03); first-party, `vokra-core` dep only | no        | (TBD) |
| `vokra-vad-micro::scalar`        | `exp` / `tanh` / `sqrt`                                     | Added | `pub fn (f32) -> f32`, `core`-only (no `std`, no `libm`)                                                  | shared transcendentals so std ↔ no_std Silero are bit-identical (T08); Newton `sqrt` default (ADR §(d)) | no        | (TBD) |
| `vokra-models::silero_vad`       | `SampleRate`                                                | Moved | now `pub use vokra_vad_micro::SampleRate` (identical type; source-compatible re-export)                   | forward core relocation (ADR §(a)); existing consumers (`vokra-cli` / `vokra-capi` / example) unchanged | no        | (TBD) |

### 2026-07-21 — 1.0.0-rc.1-dev (M5-02: QNN delegate backend selector — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched. This is deliberate and load-bearing for M5-13: a **C-level** QNN
delegate selector is *not* exposed during the v1.0-rc window (same posture as
M5-01 CoreML). `include/vokra.h` records that a backend/delegate selector, if
ever exported, is "an M5 decision after the real-hardware NPU bakeoff", and
`docs/handoff/m4-12.md` says to land the delegate API as a *new* C symbol after
the ANE/Hexagon bakeoff. So the only way to select QNN in the rc window is the
Rust surface (`SessionBuilder::with_backend(BackendKind::Qnn)` / `vokra-cli
--backend qnn`). `scripts/check-abi-changelog.sh` does not gate on this entry
(no C symbol changed); it is recorded for the v1.0-rc baseline snapshot
(`scripts/rust-public-api-list.sh` audits that `BackendKind` still carries
`#[non_exhaustive]`, so the variant addition is backward-compatible) and for the
M5-13 freeze decision on whether to promote the selector to the C ABI.

Scaffold status: the backend covers no op yet (QNN graph construction — the
`QnnGraph_create` → `addNode` → `finalize` → `execute` path — lands in an
SDK-gated CC re-issue wave, gated by owner T11 = SDK download + Qualcomm EULA
acceptance + real-header layout verification), so selecting it is an explicit
`UnsupportedOp` (QNN runtime present) or `BackendUnavailable` (no runtime / off
target) — never a silent CPU fall back. No GGUF metadata schema is added by this
slice; if the model-supply scheme later adds a `vokra.qnn.*` chunk, that gets its
own dated entry. **QNN is not NNAPI** (FR-BE-07): NNAPI remains permanently
unsupported; QNN is the Qualcomm Hexagon NPU delegate.

| Crate / area              | Symbol                 | Kind  | Signature                            | Rationale                                                        | Breaking? | PR    |
| ------------------------- | ---------------------- | ----- | ------------------------------------ | ---------------------------------------------------------------- | --------- | ----- |
| `vokra-core::backend`     | `BackendKind::Qnn`     | Added | `enum BackendKind { …, Qnn }` (`#[non_exhaustive]`, additive) | QNN delegate selector (FR-BE-06), WP M5-02; raw QNN dlopen FFI, no binding crate, no bundled SDK. C-ABI exposure deferred to M5-13 post-bakeoff | no        | (TBD) |

### 2026-07-21 — 1.0.0-rc.1-dev (M5-06: `wfst_decode` — Rust surface only, opt-in feature)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched (`scripts/gen-c-abi.sh --check` = no diff; a grep for `wfst` / `fst`
in the header matches **0** symbols; `wfst_decode` is a host-side Rust runtime
search, like `beam_search`, never a C export — ADR M5-06 defers the C-surface
decision to the M5-13 freeze, so a C consumer cannot call it during the rc
window). The whole surface lives under the **opt-in `vokra-wfst` feature**
(default OFF, cfg-only — no crate dependency, root `Cargo.lock` unchanged,
NFR-DS-02), so it is invisible to the default build and to a default
`rust-public-api-list.sh` run; `scripts/check-abi-changelog.sh` does not gate on
it (no C symbol changed). Recorded here per the recording rules for the M5-13
freeze inventory.

**No GGUF metadata is added** (ADR M5-06 §3 chose the *independent `.fst`
file* input form over a `vokra.wfst.*` GGUF chunk; the developer-side OpenFST
toolchain composes HCLG offline and Vokra reads the finished binary). If a
future revision adopts the GGUF-chunk form, that is an in-scope GGUF-schema
addition and gets its own row in the "GGUF Metadata additions" section.

| Crate / area                | Symbol                                            | Kind  | Signature / shape                                                                 | Rationale                                                                     | Breaking? | PR    |
| --------------------------- | ------------------------------------------------- | ----- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------- | ----- |
| `vokra-core::decode::wfst`  | module (feature `vokra-wfst`)                     | Added | `pub mod wfst` gated `#[cfg(feature = "vokra-wfst")]`                              | FR-OP-43 `wfst_decode` — decode-only token-passing WFST search                | no        | (TBD) |
| `vokra-core::decode::wfst`  | `Semiring` / `TropicalWeight`                     | Added | trait `Semiring` (`plus`/`times`/`zero`/`one`/`approx_eq`) + tropical impl        | Viterbi min/plus semiring; `log` semiring is a documented future additive     | no        | (TBD) |
| `vokra-core::decode::wfst`  | `Fst` / `Arc` / `StateId` / `Label`               | Added | decode-only FST + `validate()`                                                    | in-memory graph the reader/decoder share (no `compose`/`determinize`)         | no        | (TBD) |
| `vokra-core::decode::wfst`  | `read_openfst_vector`                             | Added | `fn read_openfst_vector(&[u8]) -> Result<Fst<TropicalWeight>>`                    | from-scratch OpenFST `VectorFst<StdArc>` binary reader (no OpenFST link)      | no        | (TBD) |
| `vokra-core::decode::wfst`  | `WfstDecoder` / `WfstDecodeConfig`                | Added | `WfstDecoder::new(&fst).decode(&emission) -> Result<Option<WfstHypothesis>>`      | frame-synchronous token-passing decode + `decode_nbest` + `lattice`          | no        | (TBD) |
| `vokra-core::decode::wfst`  | `WfstLattice` / `WfstHypothesis` / `LatArc`       | Added | lattice + best-path + n-best output types                                         | decode output (best-first n-best mirrors `BeamHypothesis`)                    | no        | (TBD) |

### 2026-07-20 — 1.0.0-rc.1-dev (M5-14-BACKLOG: batched-beam scoring interface — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched (a grep for `beam` / `logits` / `scorer` in the header matches **0**
symbols; beam search is a host-side Rust runtime function, FR-OP-40, never a C
export). Two model↔decoder traits gain a **batched** sibling method, each with
a **default implementation that loops the existing single-item method in order**
— so every existing `LogitsSource` / `BeamScorer` keeps byte-for-byte identical
behaviour, and `scripts/check-abi-changelog.sh` does not gate on this entry (no
C symbol changed). It is recorded for the v1.0-rc baseline snapshot
(`scripts/rust-public-api-list.sh` picks the variants up) and the M5-13 freeze.

`beam_search` now expands every active beam through `logprobs_batch` in one
call, so a scorer with a batched decoder step can fold the `beam_width` per-beam
forwards into one forward; the default keeps the prior per-beam behaviour
bit-for-bit. An optimized override (Whisper folding the projections into an
m = `beam_width` GEMM) is deferred to a follow-up (measured to help only at
beam ≥ 5, ADR `M5-14-BACKLOG`); the interface + its bit-identity oracle land now.
Both new methods are **additive** (default-provided) so no `impl` breaks.

| Crate / area          | Symbol                       | Kind  | Signature                                                        | Rationale                                                              | Breaking? | PR    |
| --------------------- | ---------------------------- | ----- | --------------------------------------------------------------- | --------------------------------------------------------------------- | --------- | ----- |
| `vokra-core::decode`  | `LogitsSource::logits_batch` | Added | `fn logits_batch(&mut self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>>` (default = loop `logits`) | batched next-token logits for beam expansion (M5-14-BACKLOG-T07) | no        | (TBD) |
| `vokra-core::decode`  | `BeamScorer::logprobs_batch` | Added | `fn logprobs_batch(&mut self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>>` (default = loop `logprobs`) | batched log-probs; `beam_search` folds all active beams into one call | no        | (TBD) |

### 2026-07-20 — 1.0.0-rc.1-dev (M5-01: CoreML delegate backend selector — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched. This is deliberate and load-bearing for M5-13: a **C-level** CoreML
delegate selector is *not* exposed during the v1.0-rc window. `include/vokra.h`
records that a backend/delegate selector, if ever exported, is "an M5 decision
after the real-hardware NPU bakeoff", and `docs/handoff/m4-12.md` says to land
the delegate API as a *new* C symbol after the ANE/Hexagon bakeoff. So the only
way to select CoreML in the rc window is the Rust surface
(`SessionBuilder::with_backend(BackendKind::CoreMl)` / `vokra-cli --backend
coreml`). `scripts/check-abi-changelog.sh` does not gate on this entry (no C
symbol changed); it is recorded for the v1.0-rc baseline snapshot
(`scripts/rust-public-api-list.sh` picks the variant up) and for the M5-13
freeze decision on whether to promote the selector to the C ABI.

Scaffold status: the backend covers no op yet (the execution path lands after
the M5-01-T02 model-supply ADR), so selecting it is an explicit `UnsupportedOp`
(ANE present) or `BackendUnavailable` (no ANE) — never a silent CPU fall back.
No GGUF metadata schema is added by this slice; if the T02 ADR chooses a
`vokra.coreml.*` artifact-binding scheme, that schema addition gets its own
dated entry (per the "GGUF metadata schema" scope rule above).

| Crate / area              | Symbol                 | Kind  | Signature                            | Rationale                                                        | Breaking? | PR    |
| ------------------------- | ---------------------- | ----- | ------------------------------------ | ---------------------------------------------------------------- | --------- | ----- |
| `vokra-core::backend`     | `BackendKind::CoreMl`  | Added | `enum BackendKind { …, CoreMl }` (`#[non_exhaustive]`, additive) | CoreML delegate selector (FR-BE-06), WP M5-01; raw ObjC/CoreML FFI, no binding crate. C-ABI exposure deferred to M5-13 post-bakeoff | no        | (TBD) |

### 2026-07-15 — 1.0.0-rc.1-dev (M4-06: Moshi full-duplex S2S + FR-MD-09 attribution)

Additive C ABI surface (WP **M4-06**, FR-MD-09 / FR-OP-60 / FR-ST-03):
the full-duplex session handle (push mic / pull model / inner-monologue
text), a **dedicated cross-thread barge-in handle** (own atomic flag —
one step past the stream.rs follow-on note, ADR M4-06 §D6), and the
attribution query every deployer UI reads (`AttributionRequired` weights
— Moshi/Mimi CC-BY 4.0 — always yield a non-empty text; permissive
weights report `*out_needed == 0`). Prerelease policy applies (freeze
fires at M5-13 / v1.0 GA); `vokra_s2s_duplex_open` flattens the
`#[non_exhaustive]` Rust `DuplexSessionConfig` into scalars.

| Crate / area  | Symbol                       | Kind  | Signature                                                                                  | Rationale                                                              | Breaking? | PR    |
| ------------- | ---------------------------- | ----- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | --------- | ----- |
| `vokra-capi`  | `vokra_s2s_duplex_t`         | Added | opaque handle                                                                              | full-duplex session (M4-06-T20)                                        | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_interrupt_t`      | Added | opaque handle                                                                              | cross-thread barge-in flag (M3-14 semantics, duplex core feature)      | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_duplex_open`      | Added | `(const vokra_session_t*, int32 deterministic, uint64 seed, int32 aec_disabled_explicitly, uint64 playback_offset_samples, vokra_s2s_duplex_t**) -> vokra_status_t` | open a duplex session; AEC-required posture (FR-OP-60, explicit opt-out only) | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_frame_hop`        | Added | `(const vokra_s2s_duplex_t*, usize*) -> vokra_status_t`                                    | PCM buffer sizing (1920 @ 24 kHz / 12.5 Hz)                            | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_sample_rate`      | Added | `(const vokra_s2s_duplex_t*, uint32*) -> vokra_status_t`                                   | PCM rate of both directions                                            | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_push_mic`         | Added | `(vokra_s2s_duplex_t*, const float*, usize, int32* out_emitted) -> vokra_status_t`         | one mic frame through AEC + one model step                             | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_pull_audio`       | Added | `(vokra_s2s_duplex_t*, float*, usize cap, usize* out_len) -> vokra_status_t`               | pop the next model frame; stamps the echo reference (playback hand-off) | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_text`             | Added | `(const vokra_s2s_duplex_t*, char*, usize, usize* out_needed) -> vokra_status_t`           | inner-monologue transcript (two-call UTF-8 discipline)                 | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_interrupt_handle` | Added | `(const vokra_s2s_duplex_t*, vokra_s2s_interrupt_t**) -> vokra_status_t`                   | obtain the cross-thread barge-in handle                                | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_interrupt`        | Added | `(const vokra_s2s_interrupt_t*) -> vokra_status_t`                                         | fire barge-in (flush + reset at the next boundary; mic continues)      | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_interrupt_destroy`| Added | `(vokra_s2s_interrupt_t*)`                                                                 | free the barge-in handle                                               | no        | (TBD) |
| `vokra-capi`  | `vokra_s2s_duplex_destroy`   | Added | `(vokra_s2s_duplex_t*)`                                                                    | free the duplex session                                                | no        | (TBD) |
| `vokra-capi`  | `vokra_model_attribution`    | Added | `(const vokra_session_t*, char*, usize, usize* out_needed) -> vokra_status_t`              | FR-MD-09 attribution text (CC-BY 4.0 display obligation; 0 = permissive) | no        | (TBD) |
| `vokra-core::engines` | `S2sDuplexEngine` / `S2sDuplexHandle` / `DuplexSessionConfig` / `DuplexPushReport` / `DuplexInterruptHandle` | Added | Rust traits/types behind the C surface (facade `S2s::duplex`)             | model-agnostic duplex face (Moshi = first engine)                      | no        | (TBD) |
| `vokra-core::compliance` | `AttributionInfo` / `resolve_attribution` / `stamp_attribution` + `Session::{attribution,with_attribution}` + GGUF key `vokra.provenance.attribution` | Added | Rust API + chunk                                              | the FR-MD-09 attribution surface (registry fallback = never empty for AttributionRequired) | no        | (TBD) |
| `vokra-core::gguf` | `GgmlType::BF16` | Added | `enum GgmlType { …, BF16 = 30 }` (ggml.h tag, verified 2026-07-15) | read the all-BF16 `kyutai/moshiko-pytorch-bf16` checkpoint; converter writes F32 (exact) | no | (TBD) |

### 2026-07-15 — 1.0.0-rc.1-dev (M4-02: Unity WebGL — bytes-based session create)

Additive C ABI symbol (WP **M4-02**, FR-API-04 / FR-BE-05 / NFR-RL-04): the
bytes-based twin of `vokra_session_create_from_file`. Motivation is empirical
(ADR M4-02 §2/§3): Unity WebGL statically links `libvokra.a` built for
`wasm32-unknown-emscripten`, where (a) StreamingAssets are HTTP-served — no
`fopen` — and (b) prebuilt rust-std's fs syscalls are ABI-skewed against
Unity-bundled Emscripten (3.1.8 / 3.1.38 — measured: `metadata().is_file()`
misreads `st_mode` and fails loudly). The embedder (C# / IL2CPP, which is
ABI-consistent with Unity's own Emscripten) reads the model bytes and hands
them over; Rust never touches the filesystem on this path. General-purpose on
all platforms. `Session::from_gguf` is the matching Rust-core entry.
**rc-window prerelease ABI policy applies** (IF-01 freeze fires at M5-13).

| Crate / area      | Symbol                            | Kind  | Signature                                                                                              | Rationale                                                                     | Breaking? | PR    |
| ----------------- | --------------------------------- | ----- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------- | ----- |
| `include/vokra.h` | `vokra_session_create_from_bytes` | Added | `enum vokra_status_t vokra_session_create_from_bytes(const uint8_t *data, size_t len, struct vokra_session_t **out_session)` | In-memory GGUF session create (Unity WebGL primary model path), WP M4-02       | no        | (TBD) |
| `vokra-core`      | `Session::from_gguf`              | Added | `pub fn from_gguf(gguf: GgufFile) -> SessionBuilder`                                                    | Filesystem-free builder entry backing the C symbol (ADR M4-02 §3)              | no        | (TBD) |
| `vokra-core`      | `IN_MEMORY_MODEL_PATH`            | Added | `pub const IN_MEMORY_MODEL_PATH: &str = "<in-memory>"`                                                  | Documented `model_path()` sentinel for bytes-built sessions                    | no        | (TBD) |

### 2026-07-15 — 1.0.0-rc.1-dev (M4-01: WebGPU / WASM)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched by this WP: the header exposes no backend-selection surface today
(sessions are CPU-fixed at the C boundary), so `scripts/check-abi-changelog.sh`
does not gate on this entry; it is recorded for the M4-12 v1.0-rc baseline
snapshot (`scripts/rust-public-api-list.sh` picks the variant up). Whether a
WebGPU backend selector should be exposed through the C ABI is deferred to
M4-02 (Unity WebGL) — see the M4-01 spec T26 hand-over note (f).

The npm Web distribution (`web/pkg`, `@vokra/web` placeholder scope until the
owner registers the org — M4-01-T27) and its JS/TS API
(`createSession` / `session.transcribe` / `session.close`) are **outside the
C ABI**; they are versioned with the npm package itself (tag semver,
prerelease `1.0.0-rc.N` included) and recorded in `CHANGELOG.md`, not here —
same posture as the vokra-server HTTP APIs ("Out-of-scope" above).

| Crate / area              | Symbol                 | Kind  | Signature                            | Rationale                                                        | Breaking? | PR    |
| ------------------------- | ---------------------- | ----- | ------------------------------------ | ---------------------------------------------------------------- | --------- | ----- |
| `vokra-core::backend`     | `BackendKind::WebGpu`  | Added | `enum BackendKind { …, WebGpu }` (`#[non_exhaustive]`, additive) | WebGPU backend selector (FR-BE-05), WP M4-01; raw extern-import shim, no wgpu crate (ADR M4-01) | no        | (TBD) |

### 2026-07-15 — 1.0.0-rc.1-dev (M4-08: RVV 0.7.1 fallback tier)

Additive **Rust dispatch surface** change only (WP **M4-08**, FR-BE-01) —
the C ABI (`include/vokra.h`) is untouched: `IsaPath` is a within-CPU-backend
dispatch enum that has never been exposed through the C boundary (grep of
`docs/abi/vokra-rust-public-api.v0.9.list` at ticket time and again at land
time: 0 hits — no snapshot update needed), so `scripts/check-abi-changelog.sh`
does not gate on this entry; it is recorded under the rc-window prerelease
policy ("every change lands with an entry"). The env-var token space of
`VOKRA_CPU_ISA` grows by `rvv071` — env tokens are configuration, not ABI,
but recorded here for the same M4-12 baseline-snapshot completeness.

| Crate / area                  | Symbol                       | Kind  | Signature                                             | Rationale                                                                                                                                              | Breaking? | PR    |
| ----------------------------- | ---------------------------- | ----- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ----- |
| `vokra-backend-cpu::features` | `IsaPath::Rvv071`            | Added | `enum IsaPath { …, Rvv071, … }`                        | RVV draft-0.7.1 tier for T-Head C910/C906 (LicheePi 4A / Milk-V Duo), encoding-incompatible peer of `Rvv` (ADR M4-08)                                    | no        | (TBD) |
| `vokra-backend-cpu::features` | `CpuFeatures::rvv_071`       | Added | `pub rvv_071: bool`                                    | 0.7.1 probe (xtheadvector isa token / vendor `cpu-vector : 0.7.1` line) with the RVV 1.0 misdetection guard — `rvv_v` and `rvv_071` never both true      | no        | (TBD) |
| `vokra-backend-cpu::features` | `CpuFeatures::rvv_071_auto`  | Added | `pub rvv_071_auto: bool`                               | Auto-select eligibility (mainline xtheadvector signal only; vendor-kernel hosts are override-only — fabricated auto-detect forbidden, ADR M4-08 §c)      | no        | (TBD) |
| env (`VOKRA_CPU_ISA`)         | `rvv071` token               | Added | `VOKRA_CPU_ISA=rvv071`                                 | First-class enablement path on vendor-kernel boards; unsupported hosts get an explicit `BackendUnavailable` (FR-EX-08), never a silent switch            | no        | (TBD) |

### 2026-07-15 — 1.0.0-rc.1-dev (M4-17: CPU ISA server tier)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched: ISA-tier selection is an internal `vokra-backend-cpu` dispatch
surface, no ISA enum is exported through cbindgen, and the only header delta
is the comment-block "RESERVED — CPU ISA tiers" note in the STABILITY banner
(no symbol change, so `scripts/check-abi-changelog.sh`'s symbol gate is not
tripped; this entry is informational for the M4-12 rc baseline snapshot).
`IsaPath` gained `#[non_exhaustive]` in the same change — see
`## Reserved additions` below for the forward-compat contract this pins.

| Crate / area                 | Symbol                                    | Kind  | Signature                                                                                     | Rationale                                                                                                                | Breaking? | PR    |
| ---------------------------- | ----------------------------------------- | ----- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------- | ----- |
| `vokra-backend-cpu::features`| `IsaPath` (attribute)                     | Changed | `#[non_exhaustive] pub enum IsaPath` | Freeze preparation (`docs/handoff/m4-12.md` §(e)-2): future tiers become backward-compat variant additions, WP M4-17-T04. Technically source-breaking for out-of-tree `match` users (none exist in-tree); pre-1.0 policy applies | no*       | (TBD) |
| `vokra-backend-cpu::features`| `IsaPath::{Avx512, Avx512Vnni, Avx512Bf16, AvxVnni256}` | Added | x86-64 server tiers (AVX-512 F/DQ/BW/VL f32, VNNI INT8, BF16 matmul, AVX-VNNI-256 client INT8) | FR-BE-01 ISA ladder expansion, WP M4-17 (ADR M4-17 §(b))                                                                  | no        | (TBD) |
| `vokra-backend-cpu::features`| `IsaPath::{NeonFp16, NeonDotprod, NeonI8mm, NeonBf16}` | Added | ARM64 server tiers (fp16 GEMM, dotprod INT8, i8mm SMMLA, BFMMLA)                               | FR-BE-01 ISA ladder expansion, WP M4-17 (ADR M4-17 §(b))                                                                  | no        | (TBD) |
| `vokra-backend-cpu::features`| `CpuFeatures::{avx512f, avx512dq, avx512bw, avx512vl, avx512vnni, avx512bf16, avxvnni256, neon_fp16, neon_dotprod, neon_i8mm, neon_bf16}` | Added | `pub bool` probe fields (std `is_x86_feature_detected!` / `is_aarch64_feature_detected!` only — no getauxval FFI, NFR-DS-02) | Server-tier runtime probe, WP M4-17-T02/T03. Struct-literal construction outside the crate breaks (use `CpuFeatures::NONE` + update syntax); pre-1.0 | no*       | (TBD) |
| `vokra-backend-cpu::features`| `CpuFeatures::{NONE, best_int8_isa, best_bf16_isa, best_fp16_isa}` + `IsaPath::ALL_SIMD` | Added | op-kind tier selectors + all-SIMD iteration list                                               | Specialized (INT8/BF16/FP16) tiers are opt-in per op kind, not part of the f32 table ladder (ADR M4-17 §(b)-2)             | no        | (TBD) |
| `vokra-backend-cpu::kernels` | `KQuantDtype`, `kquant_dequant_on`, `kquant_gemv_i8{,_on}`, `kquant_gemv2_i8_on`, `gemm_bf16_on`, `gemm_fp16_on`, converters (`f32_to_f16_rne` 等) | Added | specialized kernel surface (bit-identical dequant fusion / INT8 / reduced-precision matmul)    | K-quants dequant fusion + INT8/BF16/FP16 kernels, WP M4-17-T10..T17                                                        | no        | (TBD) |
| `Cargo.toml` (vokra-backend-cpu) | `rust-version = "1.89"` (crate override) | Changed | workspace stays `1.85`; backend-cpu floor rises | AVX-512 intrinsics stabilized in Rust 1.89; cargo enforces per-crate. Effective workspace build floor is 1.89 (backend-cpu is in every build) — owner may want to lift the workspace declaration at M4-11/M4-12 | no*       | (TBD) |

`no*` = additive at the C ABI, source-affecting at the Rust API edge; the
pre-1.0 prerelease policy (rename/remove allowed with a dated entry) covers
it.

### 2026-07-15 — 1.0.0-rc.1-dev

Additive `vokra_aec_*` surface (WP **M4-03**, FR-OP-60): the SpeexDSP-MDF
echo canceller + the sample-clock far-end reference queue, the hard-gate
(G1) prerequisite of M4-05 (CSM) / M4-06 (Moshi full-duplex). Split-handle
design: the far-end writer is a separate opaque handle so the playback
callback thread and the inference thread run concurrently over the internal
SPSC queue (the M3-14 cross-thread lesson, ADR M4-03 §D-(j)). **rc-window
prerelease ABI policy applies** (IF-01 freeze fires at M5-13, not here):
these symbols may still be renamed/removed before the v1.0 GA tag, with a
dated entry per change.

| Crate / area      | Symbol                        | Kind  | Signature                                                                                                                                                          | Rationale                                                        | Breaking? | PR    |
| ----------------- | ----------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | --------- | ----- |
| `include/vokra.h` | `vokra_aec_create`            | Added | `enum vokra_status_t vokra_aec_create(const struct vokra_aec_config_t *config, struct vokra_aec_t **out_aec, struct vokra_aec_ref_writer_t **out_writer)`           | AEC construction (canceller + far-end writer pair), WP M4-03     | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_ref_push`          | Added | `enum vokra_status_t vokra_aec_ref_push(struct vokra_aec_ref_writer_t *writer, const float *pcm, size_t num_samples, uint64_t playback_pos, size_t *out_accepted)` | Far-end push, sample-clock tag + visible backpressure (FR-EX-08) | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_process`           | Added | `enum vokra_status_t vokra_aec_process(struct vokra_aec_t *aec, const float *mic, uint64_t mic_pos, float *out, size_t num_samples, enum vokra_aec_status_t *out_status, size_t *out_missing)` | Per-frame cancellation + status visibility, WP M4-03             | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_reset`             | Added | `enum vokra_status_t vokra_aec_reset(struct vokra_aec_t *aec)`                                                                                                      | As-new reset (pairs with `vokra_stream_interrupt` barge-in)      | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_destroy`           | Added | `void vokra_aec_destroy(struct vokra_aec_t *aec)`                                                                                                                   | Handle release (NULL no-op, ADR-0003 §3-a)                       | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_ref_writer_destroy`| Added | `void vokra_aec_ref_writer_destroy(struct vokra_aec_ref_writer_t *writer)`                                                                                          | Writer release (independent lifetime)                            | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_config_t`          | Added | `typedef struct vokra_aec_config_t { uint32_t sample_rate; size_t frame_size; size_t filter_length; size_t ref_queue_capacity_samples; } vokra_aec_config_t`        | Public-layout config (0 capacity = 8×filter_length default)      | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_status_t`          | Added | `typedef enum vokra_aec_status_t { VOKRA_AEC_CANCELLED = 0, VOKRA_AEC_PASS_THROUGH = 1, VOKRA_AEC_PARTIAL_REFERENCE = 2, VOKRA_AEC_RESET = 3, } vokra_aec_status_t` | Per-frame outcome (degraded modes visible, FR-EX-08)             | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_t`                 | Added | `typedef struct vokra_aec_t vokra_aec_t` (opaque)                                                                                                                   | Canceller + queue-reader handle (inference thread)               | no        | (TBD) |
| `include/vokra.h` | `vokra_aec_ref_writer_t`      | Added | `typedef struct vokra_aec_ref_writer_t vokra_aec_ref_writer_t` (opaque)                                                                                             | Far-end writer handle (playback thread)                          | no        | (TBD) |

### 2026-07-09 — 0.9.0-dev

| Crate / area                    | Symbol                                        | Kind  | Signature                                                                   | Rationale                                                                                                                 | Breaking? | PR    |
| ------------------------------- | --------------------------------------------- | ----- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------- | ----- |
| `include/vokra.h`               | `vokra_stream_interrupt`                      | Added | `enum vokra_status_t vokra_stream_interrupt(struct vokra_stream_t *stream)` | Barge-in / cancel (FR-ST-03), WP M3-14                                                                                    | no        | (TBD) |
| `gguf:vokra.voxtral.adapter.*`  | `vokra.voxtral.adapter.{kind,tensor_prefix,in_dim,out_dim,has_bias,has_layernorm,activation,time_stride,weight_name,bias_name,layernorm_gamma_name,layernorm_beta_name,mlp_hidden_dims,mlp_layer_names}` | Added | Kind = `string` \| dims = `u32` \| flags = `bool` \| names = `string` (see `crates/vokra-models/src/voxtral/adapter.rs` for the loader) | Voxtral audio-adapter (encoder → soft-prefix) framework — M3-10 Wave 8 (real ASR conditioning; absent = LM-continuation) | no        | (TBD) |

### 2026-07-15 — 1.0.0-rc.1-dev (M4-20: audio dialect op subset)

**Additive Rust public API only — `include/vokra.h` is untouched** by this WP
(word timestamps / speaker_verify / the speech-enhancement ops are Rust-surface
functions, not C symbols; the T14 anchors are `&'static str` constants that add
**no** C ABI symbol — the whole point of the trigger-backed subset rule, ADR
M4-20 §D-6). `scripts/check-abi-changelog.sh` does not gate on these; they are
recorded for the M4-12 v1.0-rc baseline snapshot (`rust-public-api-list.sh`).
One **behaviour change**: `beam_search` with `word_timestamps` now returns
`UnsupportedOp` when the scorer supplies no alignment (was `NotImplemented`
while unimplemented) — a Rust-surface semantic change, not an ABI break.

| Crate / area                | Symbol                                                                 | Kind    | Signature                                                                                     | Rationale                                                              | Breaking? | PR    |
| --------------------------- | --------------------------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | --------- | ----- |
| `vokra-core::decode`        | `WordTiming` / `CrossAttention` / `AlignmentParams`                    | Added   | Rust structs (host-side word-timestamp alignment)                                            | FR-OP-40 word timestamps, M4-20 (a)                                    | no        | (TBD) |
| `vokra-core::decode`        | `token_alignment` / `words_from_alignment`                            | Added   | `fn(&CrossAttention, &AlignmentParams) -> Result<Vec<f32>>` / grouping fn                     | cross-attention DTW core (openai-whisper timing.py), M4-20 (a)         | no        | (TBD) |
| `vokra-core::decode`        | `BeamScorer::align_words`                                              | Added   | `fn align_words(&mut self, &[u32]) -> Result<Option<Vec<WordTiming>>>` (default `Ok(None)`)   | model supplies word alignment; default keeps existing scorers valid    | no        | (TBD) |
| `vokra-core::decode`        | `BeamHypothesis.word_timestamps`                                       | Added   | `Option<Vec<WordTiming>>` field (additive)                                                    | word-timing result on the best hypothesis, M4-20 (a)                   | no        | (TBD) |
| `vokra-core::decode`        | `beam_search` (`word_timestamps` path)                                 | Changed | `NotImplemented` → `UnsupportedOp` when no alignment supplied (FR-EX-08)                       | word timestamps implemented; explicit error replaces "unimplemented"  | no        | (TBD) |
| `vokra-models::speaker`     | `cosine_similarity` / `speaker_verify` / `SpeakerVerifyResult`         | Added   | `fn(&[f32], &[f32]) -> Result<f32>` / `fn(&[f32], &[f32], Option<f32>) -> Result<…>`          | FR-OP-81 speaker verification (CAM++ trigger), M4-20 (b)               | no        | (TBD) |
| `vokra-models::whisper`     | `WhisperConfig.alignment_heads`                                        | Added   | `Vec<(usize, usize)>` field (from optional `vokra.whisper.alignment_heads`)                   | Whisper word-timestamp alignment heads, M4-20 (a)                     | no        | (TBD) |
| `vokra-ops`                 | `agc` / `AgcAttrs` / `hpf` / `HpfAttrs` / `loudness_norm` / `LoudnessNormAttrs` / `integrated_lufs` | Added | runtime functions (FR-OP-62 / FR-OP-63)                                                       | speech-enhancement subset (agc/hpf/loudness), M4-20 (c)               | no        | (TBD) |
| `vokra-ops`                 | `denoise` / `DenoiseModel` / `DenoiseWeights` / `DeepFilterNetConfig`  | Added   | DeepFilterNet-topology denoiser (FR-OP-61)                                                    | speech enhancement `denoise`, M4-20 (c)                                | no        | (TBD) |
| `vokra-convert`             | `convert_denoise_synthetic` / `convert_denoise_from_model`             | Added   | `vokra.denoise.*` GGUF writers                                                                | denoise offline path, M4-20 (c) T12                                    | no        | (TBD) |

#### Reserved additions — M5-residual op anchors (M4-20 T14)

Forward reservations recorded **before** the IF-01 freeze (M5-13; ADR M4-20
§D-6) so a post-freeze M5 op landing is a backward-compatible additive, never a
shape break. These are `vokra-core::m5_residual_ops` `&'static str` constants —
**declared, never registered** (the `KOKORO_ISTFT_HEAD_OP` pattern; guarded by
`m5_residual_ops::tests::new_anchors_are_reserved_but_unregistered`). They add
**no** C ABI symbol and are **not** inserted into `MinDtypeRegistry` / `OpKind`.

| Reserved op-kind id          | FR-OP    | M5 blocker (why deferred)                                     |
| ---------------------------- | -------- | ------------------------------------------------------------ |
| `bigvgan_generator` (op)     | FR-OP-11 | no trigger model; min-dtype anchor already registered (M2-08), only the generator **op landing** is M5 |
| `ctc_decode`                 | FR-OP-41 | NeMo-family trigger pending                                  |
| `rnnt_decode`                | FR-OP-42 | NeMo-family trigger pending                                  |
| `ecapa_tdnn_speaker_encode`  | FR-OP-80 | CAM++ already covers speaker embedding                       |
| `wespeaker_speaker_encode`   | FR-OP-80 | CAM++ already covers speaker embedding                       |
| `titanet_speaker_encode`     | FR-OP-80 | CAM++ covers it; TitaNet NVIDIA NC restriction unconfirmed   |
| `diarize`                    | FR-OP-82 | trigger + license (pyannote HF-gated) double blocker         |

### 2026-07-19 — 1.0.0-rc.1-dev (cc-06: Moshi full-7B streaming convert + mmap load)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched (baseline 33 fn / 11 typedefs unchanged; `scripts/check-abi-changelog.sh`
green), and **no `vokra.*` GGUF key was added or renamed**. Behavioral note
for on-disk artefacts: the Moshi converter now writes BF16 checkpoint tensors
**verbatim as GGUF `BF16` (ggml type 30)** instead of widening to F32 at
convert time (the Voxtral 12e574e posture; the runtime's single `tensor_f32`
path widens BF16 → f32 exactly at load, so values are bit-identical and any
M4-era runtime — where type 30 landed — reads both layouts). The C-ABI
`vokra_session_create_from_file` path for Moshi now loads through the
true-mmap + mapped-lazy-blocks route (same numerics, bounded memory);
`vokra_session_create_from_bytes` keeps the fully resident binding.

| Crate / area   | Symbol                                        | Kind  | Signature                                                                                              | Rationale                                                                                                     | Breaking? | PR    |
| -------------- | --------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | --------- | ----- |
| `vokra-core::gguf` | `GgufStreamWriter` / `GgufTensorDecl`     | Added | streaming GGUF writer: declarations first, payloads streamed in order, byte-identical to `GgufBuilder::to_bytes` | bounded-memory conversion primitive (Moshi full-7B ~97 GiB materialization fix), cc-06                        | no        | (TBD) |
| `vokra-core::gguf` | `GgufError::InvalidStreamUse`             | Added | new enum variant (stream-writer contract violations)                                                    | loud mis-sequencing errors (FR-EX-08)                                                                          | no        | (TBD) |
| `vokra-core::safetensors` | `SafetensorsFileReader`            | Added | header-only open + windowed `read_tensor_into` (same parser as `SafetensorsFile`)                       | one-tensor-at-a-time checkpoint reads, cc-06                                                                   | no        | (TBD) |
| `vokra-models::moshi` | `MoshiEngine::from_path_with_policy`   | Added | `from_path` under an explicit `CompliancePolicy`                                                        | mmap + mapped-lazy load with policy control                                                                    | no        | (TBD) |
| `vokra-models::moshi` | `MappedTemporalBlocks` / `MoshiBackbone::{new_mapped,is_mapped}` / `MoshiBackboneWeights::head_from_gguf` / `MoshiModel::from_parts` / `MoshiDepthTransformer::config` | Added | mapped-lazy temporal-block store + assembly surface (bit-identical to resident — pinned by tests)       | full-7B on 16 GB machines (cc-06); `MoshiEngine::from_path` semantics change from buffered-resident to mmap+mapped-lazy (identical numerics, explicit `Unsupported` on Emscripten instead of a silent buffered fallback) | no        | (TBD) |

## GGUF Metadata additions (non-C-ABI, informational)

The following GGUF metadata chunks were added during the M3 waves. **These
are model-file (`.gguf`) additions only, NOT part of the C ABI surface** —
`include/vokra.h` does not expose any GGUF key by name, so
`scripts/check-abi-changelog.sh` does not gate on them. This section is
informational and prepares the M3-16 changelog for the M5-13 v1.0 GA
freeze, at which point the GGUF schema is co-frozen with the C ABI
(baseline anchor `docs/abi/vokra.h.v0.9-baseline.symbols` covers C symbols
only; a paired GGUF metadata anchor is out of scope for M3-16).

Rationale for tracking this on-file (even though the gate does not care):

- **Content-addressed compat**: model files are the exchange format between
  the offline converter (`vokra-convert`) and the runtime (`vokra-models`).
  A GGUF key rename is a compatibility break for on-disk artefacts even if
  no C symbol moved. Recording it here lets a future consumer of a v0.9.x
  `.gguf` file (produced by an older converter) find out from a single
  document what keys they can expect.
- **Trace to WP / commit**: each row names the M3 work-package that
  introduced the chunk; a bisect against a `.gguf` regression can point at
  the WP without re-reading commit logs.

Recording rules for entries here:

- **Do NOT overlap** with C-ABI entries. If a WP added both C symbols and
  GGUF keys, the C symbols go in the `## Entries` sections above (gated by
  `scripts/check-abi-changelog.sh`); the GGUF keys go here.
- **Kind field** = the GGUF value type (`u32` / `f32` / `bool` / `string` /
  `u8-array` etc.), matching the writer call in the converter
  (`add_u32` / `add_string` / `add_bool` / `add_f32`).
- **Status field**: `persisted` = the converter writes the key today;
  `documented` = the runtime docstring references the key but the
  converter does not yet emit it (the runtime falls back to defaults or
  errors). `documented` rows become `persisted` when the corresponding
  converter WP lands the writer call.

### v0.9 window — GGUF metadata additions

| WP    | Chunk prefix                   | Keys                                                                                                                                                                                                             | Kind          | Status      | Rationale                                                                                                                                                                              | Introducing wave / commit |
| ----- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| M3-03 | `vokra.paged_kv.*`             | `vokra.paged_kv.block_size` (proposed; **Mimi 12.5 Hz RVQ state uses `block_size = 2` (primary)**, higher-rate RVQ codecs (DAC, 50–86 Hz released variants) and the 25–50 Hz Whisper / CosyVoice2 / Voxtral decode paths use `block_size = 4` — ADR M3-06 §D4 / ADR M4-04 §T02. The earlier "RVQ codec paths use `block_size = 4`, LLM decode paths use `block_size = 2`" phrasing here was an over-generalization with the roles inverted for Mimi; corrected by M4-04 T12.) | `u32`         | documented  | Paged KV cache `[time, stream, codebook]` 3D layout. Converter-side emission lands with the M3-06 mimi_rvq / M3-09 CosyVoice2 wiring (M3-03-native paths use the runtime default today). | Wave 2                     |
| M3-04 | `vokra.kv_quant.*`             | `vokra.kv_quant.format` (proposed; `"q4_0"` / `"q5_0"` / `"q8_0"` / absent = fp32/fp16 native), `vokra.kv_quant.block_size` (proposed; per-format tile size)                                                       | `string` + `u32` | documented | KV cache quantization discriminator. Persistence lands when the converter has weights whose scheme differs from `Q4_K/Q5_K/Q6_K` (which are model-weight quants, not KV-cache quants).   | Wave 2 / Wave 6            |
| M3-06 | `vokra.mimi.*`                 | `vokra.mimi.n_codebooks`, `vokra.mimi.codebook_size`, `vokra.mimi.d_model` — **checkpoint-driven** (the kyutai release physically carries 1 semantic + 31 acoustic = `32` codebooks × `2048` × d_model `512`; the M3-06 canonical `MimiRvqAttrs::mimi()` 8×2048×512 is the consumer *prefix view*, not what the converter writes)                                                                                    | `u32`         | **persisted** (M4-04 T10) | Static shape attributes for the Mimi RVQ decoder — read by `MimiCodecGguf::from_gguf` (`crates/vokra-models/src/codec.rs`) into `MimiRvqAttrs`. **documented → persisted transition**: M3-09 persisted only the namespaced `vokra.cosyvoice2.mimi.*` copy; the standalone `vokra.mimi.*` keys are first emitted by the M4-04 T10 standalone codec converter (`crates/vokra-convert/src/models/mimi.rs`), which also writes the derived tensor `vokra.mimi.codebook_tables` (f32 `[n_codebooks, codebook_size, d_model]`, effective pre-projected tables — ADR M4-04 §D-f). | Wave 3 (documented) → M4-04 wave 1 (persisted)                     |
| M3-07 | `vokra.hifigan.*`              | `vokra.hifigan.{initial_channel, n_upsample_stages, n_mrf_branches, conv_pre_kernel, conv_post_kernel, upsample_kernels[], upsample_strides[]}` + per-stage MRF descriptors                                        | `u32` + array | documented  | HiFi-GAN generator arch attributes — read by `HifiGanWeights` in `crates/vokra-ops/src/hifigan.rs` (see docstring L136–142). Converter-side emission lands when a dedicated HiFi-GAN converter or the M3-09 CosyVoice2 converter writes it. | Wave 3                     |
| M3-09 | `vokra.cosyvoice2.*`           | `vokra.cosyvoice2.sample_rate` (`24000`), `vokra.cosyvoice2.arch.{vocab_size,hidden_dim,n_layer,n_head,ffn_dim}`, `vokra.cosyvoice2.flow.{nfe,schedule}`, `vokra.cosyvoice2.mimi.{n_codebooks,codebook_size,d_model}`, `vokra.cosyvoice2.streaming.{chunk_size,chunk_hop}` | `u32` + `string` | persisted  | CosyVoice2 architecture / Flow Matching / Mimi codec / streaming attributes — written by `crates/vokra-convert/src/models/cosyvoice2.rs` and read by `crates/vokra-models/src/cosyvoice2/mod.rs`. `flow.schedule` values: `"linear"` / `"sway"` / `"epss"` (M3-05 flow_sampler). | Wave 5                     |
| M3-10 | `vokra.voxtral.audio_encoder.*` | `vokra.voxtral.audio_encoder.{n_layer,n_head,hidden_dim,n_mels}`                                                                                                                                                  | `u32`         | persisted   | Voxtral audio encoder (Whisper-family arch) attributes — written by `crates/vokra-convert/src/models/voxtral.rs`, read by `crates/vokra-models/src/voxtral/`.                            | Wave 5                     |
| M3-10 | `vokra.voxtral.text_decoder.*`  | `vokra.voxtral.text_decoder.{n_layer,hidden_dim,ffn_dim,vocab_size}`                                                                                                                                              | `u32`         | persisted   | Voxtral Mistral-family text decoder attributes.                                                                                                                                          | Wave 5                     |
| M3-10 | `vokra.voxtral.mode`           | `vokra.voxtral.mode`                                                                                                                                                                                             | `string`      | persisted   | Voxtral mode discriminator: `"asr"` (audio → text) or `"s2s"` (speech-to-speech scaffold). Read by `crates/vokra-convert/src/main.rs::convert_voxtral_file`.                             | Wave 5                     |
| M3-10 | `vokra.voxtral.adapter.*`      | (see the C-ABI-adjacent entry above under `## Entries` → 2026-07-09 → `gguf:vokra.voxtral.adapter.*`)                                                                                                             | mixed         | persisted   | Audio-adapter framework — the primary changelog entry lives in the `## Entries` section above so both C-ABI and GGUF views find it; the row here cross-references only.                  | Wave 8                     |
| M4-20 | `vokra.denoise.*`              | `vokra.denoise.{n_fft,hop,sample_rate,n_erb,hidden,df_bins,df_order}` (`u32`) + flat F32 tensors `vokra.denoise.{encoder,erb_decoder,df_decoder}.{weight,bias}` — read by `DenoiseModel::from_gguf` / written by `DenoiseModel::to_gguf_bytes` (`crates/vokra-ops/src/denoise.rs`) | `u32` + `f32` tensors | persisted (synthetic path) | DeepFilterNet `denoise` (FR-OP-61) config + neural-scaffold tensors. The synthetic converter (`convert_denoise_synthetic`) writes/reads this today; the **real** DeepFilterNet checkpoint → tensors mapping is owner (T17). | M4-20 (c)                  |
| M4-20 | `vokra.whisper.alignment_heads`| `vokra.whisper.alignment_heads` — OPTIONAL flat `[layer0,head0,layer1,head1,…]` `u32` pair array; read by `WhisperConfig::from_gguf` into `alignment_heads`. Absent → word timestamps fail explicitly (FR-EX-08). | `u32-array`   | documented  | Whisper cross-attention DTW alignment heads (FR-OP-40 word timestamps). Model-specific data (not fabricated); converter-side emission is owner (real `model.alignment_heads` blob).      | M4-20 (a)                  |

### v1.0-rc window (M4) — GGUF metadata additions

| WP    | Chunk prefix                   | Keys                                                                                                                                                                                                             | Kind          | Status      | Rationale                                                                                                                                                                              | Introducing wave / commit |
| ----- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| M4-04 | `vokra.dac.*`                  | `vokra.dac.{n_codebooks,codebook_size,codebook_dim,d_model,sample_rate,hop_length}` (config-side-car-driven; the zoo-primary 24 kHz / 8 kbps variant is `32 / 1024 / 8 / 1024 / 24000 / 320` — verified from the release checkpoint metadata, ADR M4-04 §T02). Companion **derived tensor names** in the same namespace: `vokra.dac.quantizer.{i}.{codebook,out_proj_weight,out_proj_bias}` (weight-norm folded offline). | `u32`         | persisted   | DAC factorized RVQ decode attributes — written by `crates/vokra-convert/src/models/dac.rs` (`convert_dac_file`), read by `DacCodecGguf::from_gguf` (`crates/vokra-models/src/codec.rs`) into `DacRvqAttrs` + `DacOutProj`s. Frame rate = `sample_rate / hop_length` (75 Hz for the primary variant → paged `BlockSize::Four`). | M4-04 wave 1               |
| M4-05 | `vokra.csm.*`                  | `vokra.csm.{sample_rate,frame_rate_mhz}`, `vokra.csm.arch.backbone.{n_layer,d_model,n_head_q,n_head_kv,ffn_dim}`, `vokra.csm.arch.depth.{n_layer,d_model,n_head_q,n_head_kv,ffn_dim}`, `vokra.csm.arch.{rms_norm_eps,rope_base,n_ctx}`, `vokra.csm.rope.{scale_factor,low_freq_factor,high_freq_factor,old_context_len}` (Llama-3 scaled RoPE — torchtune `Llama3ScaledRoPE`, ADR M4-05 §D3), `vokra.csm.audio.{n_codebooks,vocab_size}`, `vokra.csm.text.vocab_size`. Flavor dims / rates / RoPE params are primary-source transcriptions (`SesameAILabs/csm` `models.py`/`generator.py`); the two vocab axes are `0`-placeholders until the T29 gated checkpoint (runtime rejects `0` at load, FR-EX-08). `vokra.tokenizer.model` (u8-array) is **reused** (M2-06 Whisper / M3-10 Voxtral key, not a new key) for the Llama-3.2 tokenizer blob. `frame_rate_mhz` is milli-Hz integer anchoring (12.5 Hz → `12500`, no f32 drift). | `u32` + `f32` | persisted   | Sesame CSM-1B (S2S) architecture attributes — written by `crates/vokra-convert/src/models/csm.rs` (`convert_file` / `convert_csm_file`), read by `CsmConfig::from_gguf` (`crates/vokra-models/src/csm/config.rs`). No `vokra.frontend.*` chunk: CSM has no mel frontend (audio enters via the Mimi encoder) — ADR M4-05 §D9 records the omission decision. | M4-05 wave 1               |
| M4-05 | `vokra.mimi.seanet.*` / `vokra.mimi.quantizer.*` / `vokra.mimi.transformer.*` (+ `vokra.mimi.{sample_rate,frame_rate_mhz}`) | `vokra.mimi.seanet.{dimension,n_filters,n_residual_layers,kernel_size,residual_kernel_size,last_kernel_size,compress,dilation_base,n_ratios}` + indexed `vokra.mimi.seanet.ratio.{i}` (count + indexed keys — the `vokra.quant.rule.*` precedent, no GGUF-array plumbing), `vokra.mimi.quantizer.{dimension,n_q,bins,input_dimension,output_dimension}`, `vokra.mimi.transformer.{d_model,n_head,n_layer,ff_dim,context,max_period,layer_scale}`. Values are `kyutai-labs/moshi` `loaders.py` transcriptions (ADR M4-05 §D2). **Distinct from** the M3-06/M4-04 `vokra.mimi.{n_codebooks,codebook_size,d_model}` RVQ-table triple (same namespace, different sub-prefixes — no key collision). | `u32` + `f32` | persisted   | Mimi **neural chain** (encoder audio→RVQ + neural decoder features→PCM) shape attributes for the shared `crates/vokra-models/src/mimi/` module (M4-05 lands, M4-06 Moshi consumes) — written by the CSM converter, read by `MimiNeuralConfig::from_gguf`. The Mimi *weights* travel in the standalone M4-04 mimi GGUF (CC-BY 4.0, NOTICE). | M4-05 wave 1               |
| M4-16 | `vokra.wavtokenizer.*`         | `vokra.wavtokenizer.vocab_size`, `vokra.wavtokenizer.d_model` — 1:1 with `WavTokenizerVqAttrs` (`crates/vokra-ops/src/fsq_codec.rs`). Released WavTokenizer configs = `4096 / 512` (upstream `vq_bins: 4096` / `dimension=512`, `num_quantizers: 1` — ADR M4-16 §D-c, verified 2026-07-15). | `u32`         | documented  | FSQ-family (FR-OP-31) single-codebook VQ decode attributes — read contract fixed by the `WavTokenizerVqAttrs` rustdoc; converter-side emission (`documented` → `persisted`) lands with the real WavTokenizer model-integration WP. **M5-13 freeze: to be declared `EXPERIMENTAL`** (`docs/handoff/m4-12.md` §(e)-2) so schema evolution stays legal at minor bumps until the codec API stabilizes — this row is the M4-16 intent record; the marker itself is burned in at M5-13 (v1.0 GA tag). | M4-16 (2026-07-15)         |
| M4-16 | `vokra.xcodec2.*`              | `vokra.xcodec2.levels` (`u32` array), `vokra.xcodec2.d_model` — 1:1 with `Xcodec2FsqAttrs` (`crates/vokra-ops/src/fsq_codec.rs`). Released X-Codec 2 checkpoint = levels `[4; 8]` (effective vocab 4^8 = 65536) / `d_model 2048` (`vq_dim`; upstream `vq/codec_decoder_vocos.py` + `modeling_xcodec2.py`, pin `vector-quantize-pytorch==1.17.8` — ADR M4-16 §D-c, verified 2026-07-15). | `u32-array` + `u32` | documented  | FSQ-family (FR-OP-31) finite-scalar-quantization dequant attributes (implicit grid + out-projection GEMV; **separate subgraph from the RVQ `vokra.mimi.*` / `vokra.dac.*` chunks** — no cross-codebook axis). Converter-side emission lands with the real X-Codec 2 model-integration WP. **M5-13 freeze: to be declared `EXPERIMENTAL`** (handoff §(e)-2) — same intent record as the `vokra.wavtokenizer.*` row above. | M4-16 (2026-07-15)         |

| M4-06 | `vokra.moshi.*`                | `vokra.moshi.arch.temporal.{n_layer,d_model,n_head,ffn_hidden}`, `vokra.moshi.arch.depth.{n_layer,d_model,n_head,ffn_hidden}`, `vokra.moshi.arch.{rms_norm_eps,rope_max_period,context,max_ctx}`, `vokra.moshi.audio.{n_q_in,dep_q,card}`, `vokra.moshi.text.{card,pad_id,end_pad_id}`, `vokra.moshi.n_delays` + indexed `vokra.moshi.delay.{i}` (count + indexed keys — the `vokra.mimi.seanet.ratio.{i}` precedent). Shape-driven where derivable (layer counts / widths / gating hidden / stream tallies / vocabs from the T02 355-tensor manifest); head counts / ε (1e-8 `rms_norm_f32`) / max_period (10000) / context (3000) / pad ids (3, 0) / delays (`[0,0,1×7,0,1×7]` structural rule — 7B verbatim) are `_lm_kwargs` transcriptions (ADR M4-06 §D2/§D3). Audio rates deliberately absent — the shared `vokra.mimi.*` chunk is the single rate authority (§D3 no-duplication rule; `quantizer.n_q = max(dep_q, n_q−dep_q)`, `bins ≡ card` per loaders.py). `vokra.tokenizer.model` reused for the raw SentencePiece blob; `vokra.provenance.attribution` (new provenance key) carries the FR-MD-09 display text. | `u32` + `f32` + `string` | persisted | Moshi (Helium temporal + depformer, full-duplex S2S) architecture attributes — written by `crates/vokra-convert/src/models/moshi.rs` (`convert_moshi_file`), read by `MoshiConfig::from_gguf` (`crates/vokra-models/src/moshi/config.rs`). BF16 checkpoint tensors are decoded to F32 **exactly** at conversion (`GgmlType::BF16 = 30` read support). | M4-06 (2026-07-15)         |
| RW-fix | `vokra.voxtral.text_decoder.*` (extension) | Adds `vokra.voxtral.text_decoder.{head_dim,n_head_q,n_head_kv,rope_base,rms_norm_eps,n_ctx}` to the M3-10 base set ({n_layer,hidden_dim,ffn_dim,vocab_size}). `head_dim` decouples the attention width from `hidden/n_head_q` (real Voxtral-Mini: 32 q-heads x 128 = 4096 != hidden 3072); `head_dim = 0` (or absent) = legacy `hidden/n_head_q` derivation, so pre-fix GGUFs still load. Written by `crates/vokra-convert/src/models/voxtral.rs`, read by `VoxtralConfig::from_gguf` (`crates/vokra-models/src/voxtral/config.rs`). | `u32` + `f32` | persisted | Real-weight campaign fix `12e574e` (GQA loader + BF16 passthrough converter): the real checkpoint's GQA head split and untied `lm_head` are now representable; converter also accepts sharded `*.index.json` input and hard-errors on weightless output. | 2026-07-16 (campaign 1 P1 fix) |
| RW-fix | `vokra.cosyvoice2.arch.*` (extension + real values) | Adds **written** emission of `vokra.cosyvoice2.arch.{n_head_kv,rope_base,rms_norm_eps,n_ctx}` (key strings pre-existed as read-side constants only) and replaces the previously **0-placeholder** values of `arch.{vocab_size,hidden_dim,n_layer,n_head,ffn_dim}` with shape-derived reals (0.5B: vocab 151936 / hidden 896 / 24L / ffn 4864; head split 14q/2kv from `--config`, cross-checked vs shapes). Plus q/k/v bias tensors now travel in the GGUF. | `u32` + `f32` | persisted | Real-weight campaign fix `7336079`: pre-fix GGUFs bound `llm=None` (all-zero hparams); old files still load (back-compat verified). NOTE: `~/.cache` artifacts converted before this fix are stale — reconvert. | 2026-07-16 (campaign 1 P1 fix) |
| RW-fix | `vokra.denoise.*` (schema v2 — REPLACES the M4-20 scaffold row above) | Config keys now: `vokra.denoise.{n_fft,hop,sample_rate,n_erb,df_bins,df_order,min_nb_erb_freqs,conv_lookahead,df_lookahead,conv_ch,emb_hidden_dim,df_hidden_dim,enc_linear_groups,linear_groups,df_gru_linear_groups,emb_num_layers,df_num_layers}` (`u32`) + `vokra.denoise.{lsnr_min,lsnr_max,norm_alpha}` (`f32`). **REMOVED**: `vokra.denoise.hidden` and the 6 scaffold tensor names — tensors are now the 115 verbatim upstream-named DeepFilterNet3 tensors (exact-shape validated, unknown names hard-error). Written by the real-checkpoint converter (`convert --model denoise`), read by `DenoiseModel::from_gguf`. | `u32` + `f32` + tensors | persisted | Campaign-2 P1 fix `9b718d1` (DFN3 real topology, sample-level parity SI-SNR gap 2.0e-7 dB). Pre-1.0 removal is legal (prerelease ABI policy) and recorded here per the recording rules; scaffold-schema GGUFs no longer load (hard error, FR-EX-08). | 2026-07-17 (campaign 2 P1 fix) |
| RW-fix | `vokra.mimi.*` (standalone converter now emits the neural chain) | The **standalone** mimi converter (`convert --model mimi`) now also writes the `vokra.mimi.seanet.*`/`quantizer.*`/`transformer.*` config chunk group (previously CSM-converter-only, see the M4-05 row) **plus 284 structural `mimi.enc.*`/`mimi.dec.*` tensors** (linear transposes `w_t`, fused in_proj splits, channel-wise upsample dense expansion — mathematically exact re-layouts of the same bytes) alongside the raw passthrough, making standalone Mimi GGUFs PCM-encode/decode bindable. | `u32` + `f32` tensors | persisted | Campaign-2 fix `ebe1cc5` (first real-weight PCM roundtrip: encode codes 4384/4384 = 100% vs upstream, decode max delta 3.67e-6). Runtime binds the structural names when present; raw-only GGUFs keep the previous behavior. | 2026-07-17 (campaign 2) |
| RW-fix | silero both-rate tensor namespaces `sr16k.*` / `sr8k.*` | The silero-vad converter emits **both** sample-rate branches as namespaced tensors (`sr16k.stft.forward_basis_buffer`, `sr8k.*`, name-sorted; then-branch = 16 k per `If(sr==16000)`), replacing the previous 8 kHz-only de-duplicated output that hard-errored on 16 kHz input. Output is byte-identical to the committed fixture GGUF. | tensors (naming) | persisted | Campaign-1 P1 fix `7639dc0` (official ctx576/288 rolling context + both-rate converter). Tensor-name schema change on the model file, no metadata key change. | 2026-07-16 (campaign 1 P1 fix) |

| M3-10 residual (cc-05, 2026-07-19) | `vokra.voxtral.adapter.frame_stack` | `vokra.voxtral.adapter.frame_stack` — u32 ≥ 1, required when `vokra.voxtral.adapter.kind = "frame_stack_mlp"` (new kind string in the existing `vokra.voxtral.adapter.kind` value set). ×N **consecutive-frame concatenation** factor applied to the encoder hidden before the MLP stack (`in_dim = frame_stack × encoder hidden`); 4 on the shipping Voxtral-Mini-3B-2507 (`params.json multimodal.downsample_args.downsample_factor`, upstream `get_audio_features` `reshape(-1, intermediate_size)`). Runtime rejects a missing/0 value at load and a non-divisible `t` at apply (FR-EX-08 — upstream reshape semantics, no pad/truncate). | `u32` (+ new `kind` value) | persisted | Real Voxtral projector conditioning: the campaign-1 `mlp` side-car could not express the ×4 stacking ([1500,1280] → [375,5120]), so real audio conditioning was impossible. Written by `crates/vokra-convert/src/models/voxtral.rs` (`AdapterSpec.frame_stack`, side-car field `"frame_stack"`), read by `AudioAdapter::from_gguf` (`crates/vokra-models/src/voxtral/adapter.rs`, `AdapterKind::FrameStackMlp`). | M4-residual audit cc-05 (2026-07-19) |

**M5-13 freeze treatment note (T12)**: `docs/handoff/m4-12.md` §(e)-2 names
only `vokra.wavtokenizer.*` / `vokra.xcodec2.*` (M4-16 FSQ) as
EXPERIMENTAL-marked at the freeze; the RVQ-side `vokra.dac.*` /
`vokra.mimi.*` chunks are **not** so named. Whether they enter the frozen
(stable) GGUF schema or carry an EXPERIMENTAL marker is decided at the
M4-12 v1.0-rc baseline snapshot and executed at the **M5-13** freeze
(v1.0 GA tag — 2026-07-14 v-label reassignment #2). Decision inputs
recorded here: both chunks are consumed by the M4-05/M4-06 (CSM / Moshi)
model WPs, which argues for stable; the `vokra.mimi.codebook_tables`
derived-tensor layout is converter-versioned and could still change if the
M4-05/06 PCM chain wants raw-only GGUFs (argues for a one-rc soak before
freezing).

Notes:

- **Existing baseline keys** (already stable pre-M3, not repeated here): `vokra.frontend.*`, `vokra.whisper.*`, `vokra.piper.*`, `vokra.campplus.*`, `vokra.tokenizer.model`, `vokra.provenance.*`, `vokra.quant.default_scheme` / `vokra.quant.rule_count`, `vokra.model.name` / `vokra.model.arch`. See ADR-0001 §"vokra.* namespace" (planning doc) for the pre-M3 chunk set.
- **Namespace policy** (unchanged): every Vokra-specific chunk lives under the `vokra.*` prefix; llama.cpp-compatible chunks (e.g. `general.*`) are honored in read but the writer never emits them under the `vokra.*` namespace. This keeps a `.gguf` interoperable with llama.cpp inspection tools while giving Vokra its own reserved namespace (CLAUDE.md L146 / "vokra-audio dialect" clause).
- **Removal rule**: a v0.9.x chunk MAY be renamed / removed pre-1.0 without a major bump, but a `documented` → `removed` transition must land a row here even though the C-ABI gate is silent about it. This is the honest-report contract for the pre-freeze window (mirrors the C-ABI pre-1.0 policy above).

### v1.0-rc window (M4) — GGUF metadata additions

| WP    | Chunk prefix    | Keys                                                                                                                                                                                                                                                                                                                                     | Kind                              | Status     | Rationale                                                                                                                                                                                                                                                                                                                                                                                                             | Introducing wave / commit |
| ----- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| M4-18 | `vokra.utmos.*` | `vokra.utmos.arch.variant` (`"wav2vec2_regression.v0"` guard), `vokra.utmos.sample_rate`, `vokra.utmos.conv.{channels[],kernels[],strides[],activation}`, `vokra.utmos.transformer.{n_layer,n_head,hidden_dim,ffn_dim,norm,ln_eps}`, `vokra.utmos.head.{dims[],pool,scale,offset}` (`scale`/`offset` optional, identity defaults) | `string` + `u32` + `u32-array` + `f32` | persisted | UTMOS scorer config — read by `UtmosConfig::from_gguf` in `crates/vokra-eval/src/metrics/utmos.rs`; required keys have no silent defaults, an unknown `arch.variant` is rejected loudly (FR-EX-08). **Status moved `documented` → `persisted` on 2026-07-20**: the M5-15 T14 converter (`vokra-convert --model utmos`, `crates/vokra-convert/src/models/utmos.rs`) writes every key in this row, so the row's original "converter-side emission lands with the owner weight flip (v1.0.x patch)" note is superseded — the 2026-07-18 un-defer removed that gate. Precision, so the promotion is not read as more than it is: the converter always emits `arch.variant = "wav2vec2_regression.v1"` (the real UTMOS22-strong checkpoint is v1), so the **`…v0` variant string itself** is still only produced by the in-crate round-trip test — that test, plus `v0_forward_is_untouched_by_the_v1_addition`, is what keeps the v0 read path exercised. | M4 Wave 1 (status updated M5-15 wave 1) |

### v1.0 GA window (M5) — GGUF metadata additions

| WP    | Chunk prefix    | Keys                                                                                                                                                                                                                                                                                                          | Kind                              | Status        | Rationale                                                                                                                                                                                                                                                                                       | Introducing wave / commit |
| ----- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| M5-15 | `vokra.utmos.*` | **v1 additions** (required iff `arch.variant == "wav2vec2_regression.v1"`, forbidden for `…v0`): `vokra.utmos.conv.{group_norm_layers[],group_norm_groups[],group_norm_eps}`, `vokra.utmos.pos_conv.{kernel,groups}`, `vokra.utmos.cond.{domain_dim,domain_id,judge_dim,judge_id}`, `vokra.utmos.blstm.hidden`, `vokra.utmos.head.activation` (`"relu"` / `"none"`) | `u32` + `u32-array` + `f32` + `string` | **persisted** | The M4-18 UTMOS un-defer (依頼者承認 2026-07-18). The real UTMOS22-strong stack needs eight structures the v0 skeleton could not express, so `ARCH_VARIANT_V1 = "wav2vec2_regression.v1"` was added — **additively**, exactly as ADR `M4-18-utmos-arch.md`:41 pre-authorized: a v0 GGUF still loads and still produces the same score. `v0_forward_is_untouched_by_the_v1_addition` pins that on two axes: the GGUF and in-memory paths agree **bit-for-bit**, and the value itself is held to a golden literal (`V0_GOLDEN_SCORE`, ±1e-6 — a tolerance because the f32 forward moves by one ULP between this host's own scalar and NEON kernel paths, so bit-exactness across ISAs is measurably false; derivation in the constant's rustdoc). The M4-18 row above was moved `documented` → **persisted** to match, since `vokra-convert --model utmos` (M5-15 T14) now emits those keys as well — with the `…v0` variant *string* still test-only, as that row records. A v0-labelled GGUF carrying any v1 key is a loud `ModelLoad` error rather than a half-honoured stack (FR-EX-08). | M5-15 wave 1               |

Note: `vokra.dnsmos.*` is **reserved but deliberately not designed** — DNSMOS is license fail-closed until the owner's M4-18 T03 verification (no keys are invented ahead of it).

<!-- Template — copy into an `### YYYY-MM-DD — vX.Y.Z-dev` section per PR-day:

### 2026-07-XX — 1.0.0-rc.1-dev

| Crate / area          | Symbol                     | Kind    | Signature                                                                       | Rationale                                | Breaking? | PR   |
| --------------------- | -------------------------- | ------- | ------------------------------------------------------------------------------- | ---------------------------------------- | --------- | ---- |
| `include/vokra.h`     | `vokra_stream_interrupt`   | Added   | `enum vokra_status_t vokra_stream_interrupt(struct vokra_stream_t *stream)`     | Barge-in cancel, M3-14                    | no        | #NN  |
| `gguf:vokra.paged_kv` | `vokra.paged_kv.block_size`| Added   | `u32`                                                                           | Paged KV cache, M3-03                    | no        | #NN  |

-->

## Reserved additions

Forward reservations recorded **before** the IF-01 freeze so that
post-freeze landings are backward-compatible additions, never shape breaks
(`docs/handoff/m4-12.md` §(e)-2; recorded by WP M4-17-T06 on 2026-07-15).

- **`vokra_backend_cpu::IsaPath` is `#[non_exhaustive]`** (since M4-17-T04).
  Downstream `match` expressions must carry a `_` arm, so adding a variant is
  a **non-breaking variant addition** under semver. Within the defining crate
  the attribute is inert — `dispatch::build_table` deliberately stays an
  exhaustive match so a variant added without a kernel table is a compile
  error.
- **Reserved variant name families** (do NOT reuse for anything else):
  - `Amx*` — Intel AMX-TILE/INT8/BF16 tiles (**M5**; excluded from M4-17
    because stable-Rust intrinsic supply is unconfirmed and Sapphire-Rapids
    soak time is unavailable — `docs/m4-scope-expansion-2026-07-13.md`
    §BIG-6). AMX-FP16 / AVX10.x remain v1.5+ anchors on top of that.
  - `Sme*` — ARM SME tiles (**M5**; Apple M4+ is the only shipping
    implementation).
  - `RvvZvfh*` — RISC-V Zvfh-gated fp16 vector tiers (future; the `rvv_zvfh`
    probe bit exists since M3-13, the tier name is reserved here).
- **The C ABI carries no ISA enum** (see the `include/vokra.h` STABILITY
  block, "RESERVED — CPU ISA tiers"): the IF-01 freeze surface excludes
  ISA-tier naming entirely. A C-level backend/delegate selector, if ever
  exported, is an M5 decision after the NPU real-hardware bakeoff
  (`docs/handoff/m4-12.md` §(e)-3 / §(f)-4) and lands as a new symbol.
- **v1.0-rc window additions covered by this reservation policy**: the eight
  M4-17 variants (`Avx512`, `Avx512Vnni`, `Avx512Bf16`, `AvxVnni256`,
  `NeonFp16`, `NeonDotprod`, `NeonI8mm`, `NeonBf16`) — prerelease-semver
  additive, recorded in the dated entry above.

## Handoff to M4-12 (v1.0 GA freeze)

> **2026-07-14 note**: after v-label reassignment #2 the freeze executor is
> **M5-13** (v1.0 GA tag = M5 close); read "M4-12" in this section as the WP
> that executes at that tag. The section heading is kept verbatim because
> other documents link to it by name. M4-12 itself (v1.0-rc tag) only
> snapshots the intermediate rc baseline and stays advisory.

**Scope of this section.** M3-16 (this WP) ships the pre-freeze machinery:
the anchor files, the advisory changelog gate, and the recording rules
above. **M3-16 does NOT fire the IF-01 freeze — that action is M5-13's**
(see `docs/milestones.md` §7.2 / §8 / §9; the v-label relabel of 2026-07-08
moved the freeze from the old M3-16 to M4-12, and reassignment #2 of
2026-07-14 moved it again to M5-13). The four items below are a
forward checklist for the freeze-executing owner; landing any of them under
the M3 branch would prematurely commit the ABI while v0.9 features are still
being wired.

### Input artefacts M4-12 will consume

These are the pre-freeze anchor artefacts. **M4-12 executed at the v1.0-rc
tag on 2026-07-15** (`docs/handoff/m4-12.md` §(g)) and added the two v1.0-rc
anchors below; **M5-13** (the freeze WP) now diffs the v1.0 GA header + Rust
surface against all of these to build the "0.1 → 1.0" (cumulative) and
"0.9 → 1.0" / "rc → GA" (incremental) delta summaries:

- **`docs/abi/vokra.h.v0.9-baseline.symbols`** — the v0.9-window anchor
  used by `scripts/check-abi-changelog.sh` during the M3 window. Captured
  at PR #3 merge (2026-07-08) per the "Baseline snapshot" section above.
  Retired as the active gate anchor at the v1.0-rc rotation (M4-12); kept on
  disk as the v0.9 historical anchor (`scripts/abi-diff.sh --anchor v0.9`).
- **`docs/abi/vokra.h.m0-anchor.symbols`** *(from M3-16-T02)* — the
  historical M0 (v0.1.0, 2026-07-04) anchor, preserved so the M4-12 rollup
  can render the **full v0.1 → v1.0 delta** — not just the v0.9-window
  slice — into `CHANGELOG.md`. The M4-12 owner should diff v1.0's
  `include/vokra.h` against **both** anchors: the m0 anchor gives the
  "since GA-1 tag" cumulative surface story, and the v0.9-baseline anchor
  gives the "since last prerelease window" incremental one.
- **`docs/abi/vokra-rust-public-api.v0.9.list`** *(from M3-16-T03; forward
  reference if not yet landed)* — snapshot of the `vokra-core` /
  `vokra-ops` / `vokra-capi` `pub` surface that cbindgen reflects into
  `include/vokra.h`. The C header is the primary IF-01 target, but the
  Rust surface is the upstream source and is worth diffing separately
  because a Rust-only change (e.g. a hidden internal helper going public)
  can still leak into the C header on a later cbindgen run. Format is
  one line per public item, sorted, generated by `cargo public-api` or
  the equivalent hand-curated dump per T03's spec.
- **`docs/abi/vokra.h.v1.0-rc-baseline.symbols`** *(from M4-12-T02)* — the
  v1.0-rc-window C anchor (33 exported functions + 11 typedefs, header commit
  `41a5ad1`), now the active `scripts/check-abi-changelog.sh` diff target.
  `scripts/abi-diff.sh --anchor v1.0-rc` renders the rc → GA increment M5-13
  needs on top of the m0 (cumulative) and v0.9 (prerelease-window) views.
- **`docs/abi/vokra-rust-public-api.v1.0-rc.list`** *(from M4-12-T05)* — the
  paired v1.0-rc Rust `pub` surface snapshot (now the active
  `scripts/rust-public-api-list.sh` diff target; its `#[non_exhaustive]` audit
  additionally covers `IsaPath`). **GA-naming flag for M5-13**: the on-disk
  convention is `vokra-rust-public-api.*`, but the "M4-12 action checklist"
  first bullet below names the GA Rust list `docs/abi/rust-public-api.v1.0.list`
  (no `vokra-` prefix) — M5-13 reconciles the GA name to the on-disk convention
  when it snapshots the GA Rust surface.

### M4-12 action checklist (do NOT execute under M3-16)

- [ ] **Re-anchor the v1.0 baseline.** Copy the v1.0-tag `include/vokra.h`
      symbol list to `docs/abi/vokra.h.v1.0-baseline.symbols`, retire
      `vokra.h.v0.9-baseline.symbols` (keep the file, but stop diffing
      against it), and switch `scripts/check-abi-changelog.sh` to diff the
      working tree against the v1.0 anchor. The m0 and v0.9 anchors stay
      on disk as historical references — the diff target is what moves.
      Also snapshot the paired Rust surface as
      `docs/abi/rust-public-api.v1.0.list`.
- [ ] **Amend the STABILITY block in `include/vokra.h`** to declare the
      IF-01 freeze in force. The current block (see the header top) reads
      "the ABI is NOT frozen; the semver ABI-stability commitment starts
      at v1.0 GA (IF-01; …)"; replace it with the frozen-form text
      mandated by ADR-0003 §"安定性方針（IF-01 / 表注 3）" (post-1.0
      breaking changes require a major-version bump; see the rejection
      clause below).
- [ ] **Roll all v0.9 entries in this file into a "0.9 → 1.0 delta"
      summary** and append that summary to `CHANGELOG.md` under the
      `[1.0.0]` heading. Then clear the `## Entries` section of this
      file for the next (v1.x) prerelease window while keeping the schema
      / policy / baseline-snapshot sections intact. The GGUF metadata
      additions table (v0.9 window) is likewise rolled into
      `CHANGELOG.md` under a `### GGUF metadata` sub-heading.
- [ ] **Promote `scripts/check-abi-changelog.sh` from advisory (M3-16) to
      required CI check** (blocks merge on `main`). Update
      `.github/workflows/ci.yml` — or the successor ABI gate workflow —
      to add the script to the required checks list, and update GitHub
      branch protection accordingly. The advisory-vs-required flip is
      deliberate scope: M3-16 ships the tool + baseline advisory, and the
      CI required-check wiring is M4-12's call so that PRs are not
      blocked on a still-churning v0.9 header.

### Post-1.0 semver contract (rejection of the pre-1.0 free-change rule)

The "Pre-1.0 policy (prerelease semver)" section above **is explicitly
retracted at v1.0 GA**. Once M4-12 lands, the following clauses of that
section are dead:

- **REVOKED**: "v0.9.x may add, remove, rename, or change signatures of
  any exported symbol" — this **no longer applies** post-1.0. Any add /
  rename / signature change to an exported C symbol, cbindgen-reflected
  Rust `pub` item, or `vokra.*` GGUF chunk requires a semver major bump
  (v2.0.0), or a deprecation path that keeps the old symbol live through
  at least one minor release before removal.
- **AMENDED**: "the single hard rule is that every such change lands
  with an entry in this file" — the recording rule survives, but
  breaking-change entries under `[1.0.0]` require a linked ADR
  justifying the break (M4-12 amends this document at freeze time to
  add the ADR-link requirement to the entry schema).

Positively stated, the post-1.0 rule is:

- **Non-breaking changes** (`Added` / `Deprecated` / `Fixed` / `Security`)
  land under a minor / patch bump (v1.1.0 / v1.0.1).
- **Breaking changes** (`Removed` / `Breaking` / signature-changing
  `Changed`) require a major bump (v2.0.0) **and** an ADR link in the
  entry `Rationale` column.
- **GGUF metadata renames** on the `vokra.*` prefix count as breaking
  under this rule even though `scripts/check-abi-changelog.sh` does not
  gate on the informational GGUF-additions table today (M4-12 may
  optionally extend the gate to cover GGUF; that decision is out of
  scope for M3-16-T05 and is deferred to M4 planning — see
  `docs/milestones.md` §8 M4-12).

This section is the honest report contract: the pre-1.0 free-change
policy is time-boxed to the pre-1.0 prerelease window (v0.9 through
v1.0-rc), and M4-12 formally revokes it at freeze time. Nothing in
M3-16 fires the freeze.
