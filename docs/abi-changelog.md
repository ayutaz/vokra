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
  enforces the recorded-symbol part of this rule: if the current
  `include/vokra.h` differs from the active gate anchor
  (`docs/abi/vokra.h.v0.9-baseline.symbols` during the v0.9 window, rotated to
  `docs/abi/vokra.h.v1.0-rc-baseline.symbols` at M4-12) and any changed symbol
  has no changelog row, the script exits non-zero. The date remains historical
  metadata for the PR entry rather than a wall-clock CI condition.
- At v1.0 GA (M5-13; M4-12 before the 2026-07-14 reassignment) the baseline
  is re-anchored to that release, the freeze commitment is written into
  `include/vokra.h`, and post-1.0 breaking changes require a major bump.

### CI posture of the three ABI gates (X-08, 2026-07-20) — ADVISORY until M5-13

`scripts/abi-diff.sh`, `scripts/check-abi-changelog.sh` and
`scripts/rust-public-api-list.sh` were unwired from CI until X-08. They now run
in the `abi-surface (advisory)` job of `.github/workflows/ci-quality.yml`,
which sets `continue-on-error: true`. (This said `ci.yml` until 2026-08-16;
that file contains no `abi-surface` job, so the citation pointed at nothing.)

**That job must stay advisory until M5-13.** Promoting these three from
advisory to a branch-protection required check *is* the content of M5-13
(`docs/milestones.md` §9), which executes together with the IF-01 freeze at the
v1.0 GA tag. X-08 deliberately wired them advisory-only so the progression is
one step at a time: unwired → advisory (X-08) → required (M5-13). Had X-08
promoted them, M5-13 would have had nothing left to execute. The cool-off
posture mirrors `gpu-vulkan-parity.yml` and the platform-support drift step in
the `license` job.

The v0.9-rc Rust snapshot was rotated on 2026-08-17 to record the intentional
public surface accumulated by PR #29. Before rotation the gate measured **180
additions / 5 removals** against the snapshot; after rotation it records
**864 functions, 264 structs, 78 enums, 30 traits, 6 types, 104 constants,
131 re-exports and 127 modules**. Reproduce the clean check with
`bash scripts/rust-public-api-list.sh` (no flags).

**The recorded delta is not purely additive.** It includes a source-BREAKING
`vokra-core` change: `pub struct BeamSearchConfig` gained a `<'a>` lifetime
for the FR-OP-41/42 shallow-fusion LM, alongside a changed `beam_search`
signature and the new `LmScorer` / `LmFusionConfig` public items. Recording
that fact before snapshot rotation keeps the breaking change visible to the
M5-13 freeze review.

The additions are dominated by `vokra-ops`: the `vit`, `itn` (grammar +
token), `wpe`, `aec`, `moe_*` and `f0` (pyin / yin) modules landed across
this campaign.

### 2026-08-17 — 1.0.0-rc.1-dev (PR #29 Rust public-API snapshot rotation)

The `docs/abi/vokra-rust-public-api.v1.0-rc.list` snapshot was regenerated
after the intentional PR #29 public-surface wave. The generated snapshot is
the current reference for `vokra-core`, `vokra-ops`, and `vokra-capi`; the
`#[non_exhaustive]` audit still passes for all six protected enums. This is
an anchor update only: no additional Rust API is introduced by this entry,
and the C ABI remains covered by the separate v1.0-rc header anchor.

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

### 2026-08-27 — 1.0.0-rc.1-dev (Ultravox native multimodal companion decode)

The exact separately acquired Llama companion now executes a bounded-memory
native decoder. It widens one authenticated BF16 layer at a time, keeps the
tied embedding mmap-backed for both token gather and a chunked vocabulary
head, applies Llama-3 scaled half-split RoPE and replaces exactly the
processor-declared consecutive prompt span with projected audio embeddings.
The combined audio-tower route requires both artifacts to use the same CPU or
Metal backend and preflights the union of all learned operations before any
encoder work. Because neither GGUF embeds a tokenizer, chat template or
generation sidecar, callers provide exact prompt, audio-start and stop-token
IDs; malformed spans and missing stop IDs are explicit errors. The CLI accepts
one 16 kHz mono WAV and follows the official variable-length Whisper processor
contract (minimum two hops, hop-rounded valid frames); over-30-second audio is
rejected until multi-chunk prompt composition is represented explicitly. A
fixed official-reference dumper plus gated VAST CPU and remote Apple
CPU/Metal consumers are staged without fabricated fixture values; actual
real-checkpoint verdicts remain pending. No C symbol, allocation ABI or
publication permission changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::ultravox` | `UltravoxGenerationOptions`, `UltravoxGeneration`; `UltravoxLlamaCompanion::{generate_with_audio_embeddings,next_token_logits_with_audio_embeddings}` | Added | exact token IDs + one consecutive `[audio_frames,2048]` replacement span → bounded greedy token IDs or full first-step logits | Requires explicit in-vocabulary stop IDs and validates prompt/span/finite audio/context bounds. One BF16 layer and at most 512 tied-head rows are widened at a time; no tokenizer download, guessed sidecar or CPU fallback | no | `3095255e` |
| `vokra-models::ultravox::UltravoxAudioTower` | `generate_from_log_mel_with_companion`, `ULTRAVOX_HOT_OPS` | Added | exact `[128,n_frames]` log-mel + separately licensed strict companion + explicit prompt/start/stop IDs → generated IDs | Rejects mixed CPU/Metal artifacts and preflights the complete audio+Llama learned-op union before execution. Public MIT audio weights and gated conditional-commercial Llama weights remain separate mappings and licenses | no | `10bbcb5b` |
| `vokra-models::whisper::mel` | `log_mel_variable` | Added | mono PCM up to 30 s + mel count → channel-major variable frame tensor and valid frame count | Reproduces Ultravox's minimum-two-hop / hop-rounded single-clip contract; long-audio chunking is explicit unsupported rather than a silent truncate | no | `10bbcb5b` |
| `vokra-cli run` | `ultravox` task; `--ultravox-companion`, `--ultravox-audio-start`, `--ultravox-stop-token-ids`, `--ultravox-max-new-tokens` | Added | 16 kHz mono WAV + exact expanded prompt contract → generated token IDs on stdout or UTF-8 output file | Both strict GGUFs bind on one selected CPU/Metal backend; no tokenizer/template/download/fallback. `bench` rejects because prompt and generated duration are content-dependent | no | `10bbcb5b` |

### 2026-08-27 — 1.0.0-rc.1-dev (Ultravox gated Llama companion boundary)

The separately licensed Meta Llama-3.2-1B-Instruct companion now has a strict,
bounded-memory conversion and mmap binding boundary distinct from the public
MIT Ultravox audio tower. Both sides admit only upstream revision
`9213176726f574b556790deb65791e0c5aa438b6`, the exact official 146-BF16-tensor
tied-embedding manifest and the raw `llama3.2` license classified as
`ConditionalCommercial`. The runtime authenticates the full topology and
descriptor types before preflighting one complete CPU or Metal decoder hot-op
set. It does not download, publish or merge the gated weights into the MIT
artifact. Tokenizer/chat composition, audio-embedding interleave and generation
remain explicit unsupported boundaries in this slice. No C symbol, ownership
rule or allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `gguf:vokra.ultravox.companion.*` | 22 source/config/manifest/topology/activation/RoPE/tied-embedding/bias keys | Added | exact fixed-revision 146-tensor BF16 Llama-3.2-1B-Instruct companion | Converter rejects every other revision, config, tensor name/shape/dtype and license; the companion remains a user-acquired gated artifact separate from the public MIT audio GGUF | no | `f83ea25b` |
| `vokra-models::ultravox` | `UltravoxLlamaCompanion`, `UltravoxLlamaConfig`, companion identity constants, `ULTRAVOX_LLAMA_HOT_OPS` | Added | strict mmap descriptor bind for the exact tied-embedding checkpoint on an explicit CPU or Metal backend | Validates provenance, conditional-commercial policy, complete manifest, all topology fields and dense descriptors before backend preflight. Generation is intentionally not exposed by this binding-only slice | no | `f83ea25b` |

### 2026-08-27 — 1.0.0-rc.1-dev (Bark and Bark Small native CPU/Metal runtime)

The two public Suno Bark GGUFs now bind through their exact complete manifests
and execute the released three-stage semantic/coarse/fine token hierarchy plus
the embedded causal 24 kHz EnCodec decoder. The historical Full artifact's
incorrect Small width/head stamp is admitted only behind its immutable
758-tensor manifest and remains visibly marked for metadata repair; Small uses
its exact 518-tensor contract. Callers provide already-tokenized text ids
because neither public GGUF embeds Bark's tokenizer, so the runtime does not
guess or download a mutable sidecar. The model preflights one complete CPU or
Metal hot-op set before generation and never substitutes another execution
backend. Standalone Meta EnCodec weights remain excluded; this path consumes
only the codec embedded in the separately audited, signed Bark composite.
Independent official-reference VAST parity and Apple-device real-weight Metal
parity remain pending, so this records code reachability rather than a
numerical-pass claim. No C symbol or publication permission is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `gguf:vokra.bark.*` | `variant`, complete LM/codec topology and `tensor_manifest_sha256`; exact provenance revision/SHA | Changed / Added | exact 518-tensor Small or 758-tensor Full all-F32 mmap checkpoint | New conversions require the pinned Suno checkpoint and complete tensor manifest. The exact historical Full header may repair only `hidden_size=768` / `num_heads=12` to `1024` / `16` after its immutable manifest matches; any partial or unrelated file fails closed | no | `6c00beca` |
| `vokra-models::bark` | `BarkModel`, `BarkVariant`, `BarkConfig`, topology constants and `BARK_HOT_OPS` | Added | strict descriptor or mapping-owned CPU/Metal bind for both public variants | Authenticates manifest, metadata, provenance, F32 layout and permissive Bark license before tensor views; unsupported backends fail whole-model preflight | no | `2f265348` |
| `vokra-models::bark` | `BarkGenerationConfig`, `BarkGeneratedCodes`, `BarkGeneratedCodes::from_frame_major`, `BarkModel::generate_codes_from_tokens` | Added / Corrected | explicit text token ids + optional pad mask to validated frame-major `[frames,8]` codec indices | Implements semantic EOS sampling, alternating two-codebook coarse schedule and six fine passes with deterministic seed. The mask replaces text IDs with the official pad token while preserving all 257 causal prefill positions, matching Transformers 4.31.0; tokenizer/history-prompt composition remains an explicit caller boundary | no | `64de9fa9`, (TBD) |
| `vokra-models::bark` | `BarkSynthesis`, `BarkModel::decode_codes`, `BarkModel::synthesize_tokens` | Added | frame-major eight-codebook packet to mono `frames*320` FP32 PCM at 24 kHz | Uses only the authenticated embedded 32-table checkpoint's first eight released codebooks, causal reflect-padded SEANet, two-layer LSTM, learned residual shortcuts and exact `[8,5,4,2]` upsampling; no standalone EnCodec loading path | no | (TBD) |
| `vokra-models::compute`; `vokra-backend-{cpu,metal}` | `HotOp::Elu`, `Compute::elu_f32`, Metal `elu_f32` | Added | element-wise ELU with fixed `alpha=1` | CPU and Metal have explicit kernels and parity coverage; uncovered backends fail preflight rather than running the activation on CPU | no | `b216ad02` |
| `vokra-cli run` / `bench` | `bark` dispatch; `--token-ids`, `--bark-max-semantic-tokens`, `--bark-seed` | Added | exact caller-supplied text ids to 24 kHz WAV on CPU/Metal | Raw text is rejected because the public GGUF has no tokenizer. Bench refuses to fabricate a prompt or fixed duration denominator and points to the executable run route | no | (TBD) |
| `tools/parity`; `vokra-models/tests` | `bark/dump_reference.py`, `parity_bark_real`, `run-bark-validation.sh` | Added | fixed-revision official Transformers 4.31.0 greedy semantic/coarse/fine codes plus embedded-codec PCM for Small and Full; Rust exact-code and FP32 `atol=0.01` consumers | Locked Python 3.12, VAST-only, no-upload worker. References/results remain unclaimed until the worker actually runs; the same fixtures drive the separate remote Apple Silicon Metal gate | no | (TBD) |

### 2026-08-27 — 1.0.0-rc.1-dev (Qwen3-ASR native CPU/Metal runtime)

The public Qwen3-ASR 0.6B and 1.7B GGUF headers now bind through exact
release-specific contracts without eagerly decoding their multi-gigabyte BF16
payloads. New conversions accept the official single file or Hugging Face shard
index directly, validate the exact BF16 manifest and pinned Apache-2.0 contract,
stamp the exact 13-field Whisper frontend, then stream one tensor at a time.
Conversion also authenticates and embeds the five fixed-revision tokenizer,
chat-template and generation sidecars, so execution never downloads mutable
assets. The runtime validates those blobs again and exposes the Qwen2 BPE,
exact ChatML/audio-placeholder prompt and structural ASR output parser.
The runtime adds true-mmap constructors, structures all 612/708 descriptors
without payload decode and implements the variable-length log-mel, Conv2D,
18/24-layer audio Transformer, projector and 28-layer autoregressive Qwen3
decoder. Prompt prefill is chunked, K/V state is cached, one decoder layer is
widened at a time and the vocabulary head is evaluated in bounded row chunks.
All learned audio/text projections, normalizations, activations, attention
softmaxes and the output head use one preflighted CPU or Metal `Compute`
backend. Unsupported backends fail before inference; no operation silently
selects CPU. `vokra-cli run` and `bench` now route `qwen3_asr` through the
concrete mmap runtime instead of the bound-only diagnostic registry; `run`
accepts optional official language names or `auto`. The strict binder now also
requires both provenance revision fields to name the exact admitted upstream
commit. Transcriptions expose the exact emitted greedy ids, including the
first natural EOS, so the independent official-package parity gate can compare
token sequences rather than decoded text alone. The pinned reference dumper,
real-checkpoint Rust test and no-upload VAST worker are staged; real-checkpoint
CPU and Apple-device Metal numerical parity remain pending and are not inferred
from source reachability. No C symbol, ownership rule or allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-convert::qwen3_asr` | `convert_qwen3_asr_file_with_variant`; `vokra.qwen3_asr.source_revision`; `vokra.qwen3_asr.tensor_manifest_sha256`; `vokra.frontend.*`; audio execution keys; tokenizer/chat/generation U8 keys | Changed / Added | single `.safetensors` or local `.index.json` + shards plus five exact fixed-revision sidecars → exact 612/708-BF16-tensor self-contained GGUF with 13 frontend fields and exact GELU/LayerNorm/scale/tokenizer contract | Rejects extra/missing/renamed/reshaped/non-BF16 tensors, unsafe/non-local shard paths, index ownership drift, conflicting license and sidecar size/SHA drift; bounded by the largest tensor rather than the checkpoint total | no | (TBD) |
| `vokra-models::qwen3_asr` | `Qwen3Asr`, `Qwen3AsrAudioEmbeddings`, `Qwen3AsrTokenizer`, `Qwen3AsrTranscription::token_ids`, `Qwen3AsrGenerationOptions`, `Qwen3AsrVariant::source_revision`, Qwen3-ASR token/language constants, `QWEN3_ASR_AUDIO_HOT_OPS`, `QWEN3_ASR_TEXT_HOT_OPS`, `QWEN3_ASR_HOT_OPS`, mapped checkpoint constructors | Changed / Added | variable-length 16 kHz PCM → parsed multilingual transcription plus exact greedy ids through one explicit CPU/Metal backend; optional context, forced language and bounded greedy generation | Exact 612/708 descriptors stay mmap-backed; Conv2D and decoder projections lower to backend GEMM, the vocabulary head uses chunked GEMV, and softmax/LayerNorm/RMSNorm/GELU/SiLU are preflighted as a whole. Historical headers remain descriptor-bindable, while execution requires both exact upstream revision keys, all 29 topology keys, exact frontend metadata and byte-authenticated `vocab.json` / `merges.txt` / tokenizer / chat / generation sidecars. Unexpected generated control tokens, non-finite values, position overflow, malformed audio placeholders, missing assets and unsupported backends are explicit errors; no runtime download or silent CPU fallback | no | (TBD) |
| `vokra-cli run` / `bench` | `ModelTask::AsrQwen3`; `qwen3_asr` dispatch; concrete mmap runner | Changed | mono 16 kHz WAV + optional `--language <official-name\|auto>` + explicit CPU/Metal backend → parsed transcript/language; bench measures the same complete graph | Removes the stale `BOUND_ARCHES` diagnostic row. Exact tokenizer/frontend/tensor drift and unsupported backend operations remain explicit errors; beam-only flags and implicit resampling are rejected rather than ignored | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (SpeechT5 CPU/Metal TTS runtime)

The canonical strict SpeechT5 artifact now executes its complete
text-to-spectrogram graph through one selected CPU or Metal backend and can
attach the already strict 16 kHz SpeechT5 HiFi-GAN companion. The historical
tokenizer-less public GGUF still fails the checkpoint contract before tensor
decode. Independent VAST CPU and Apple-device real-weight parity remain
pending, so this records code reachability rather than a numerical-pass claim.
No C symbol, ownership rule or allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::speecht5` | `SpeechT5Tts`, `SpeechT5Mel`, `SpeechT5GenerationOptions`, `SPEECHT5_HOT_OPS` | Added | strict GGUF load; text/token ids + 512-value x-vector → frame-major 80-bin mel; optional mel → 16 kHz PCM | Complete relative-position encoder, cached decoder, seeded always-on prenet dropout, stop rule and postnet use one CPU/Metal selection. Missing x-vector/vocoder and unsupported request fields/backends fail explicitly | no | (TBD) |
| `vokra-models::speecht5::SpeechT5Tts` | `from_gguf_with_vocoder`, `with_vocoder`, `generate_text_mel`, `generate_tokens_mel`; `TtsEngine` impl | Added | canonical 393-tensor text-to-mel GGUF plus canonical 158-tensor SpeechT5 HiFi-GAN GGUF | Vocoder must be 80-bin/16 kHz; backend selection propagates to both components; legacy tokenizer-less artifact remains incompatible by design | no | (TBD) |
| `vokra-cli run` | `speecht5` arch; `--vocoder`, shared `--speaker-embedding` | Added | `--model <speecht5.gguf> --vocoder <speecht5-hifigan.gguf> --speaker-embedding <512-f32> --text <string> [--backend cpu\|metal]` | Both strict artifacts bind once; the x-vector is exact-width finite little-endian f32; missing/misrouted companions and unsupported backends fail explicitly. `bench` refuses to fabricate these sidecars and points to the complete run route | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (SpeechT5 postnet Tanh CPU/Metal seam)

SpeechT5's first four postnet convolution blocks use an element-wise
hyperbolic tangent. The imperative model execution surface now exposes that
operation on CPU and Metal. CUDA and WebGPU remain explicitly unsupported for
this hot op; selecting either through a model registry fails coverage before
inference rather than executing on the host. No C symbol, ownership rule or
allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::compute` | `HotOp::Tanh`, `Compute::tanh_f32` | Added | `(&[f32], &mut [f32]) -> Result<()>` | CPU dispatches the existing portable tanh kernel; Metal dispatches `vokra_tanh_f32`; uncovered backends fail the whole-model coverage gate | no | (TBD) |
| `vokra-backend-metal::MetalContext` | `tanh_f32` | Added | `(&[f32], &mut [f32]) -> Result<()>` | Dedicated MSL element-wise kernel; length mismatch is rejected before dispatch | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (SpeechT5 strict executable conversion contract)

The historical SpeechT5 pass-through conversion accepted arbitrary float
tensors and could produce a GGUF without the text tokenizer. It is replaced by
a pinned, total conversion contract. This change does not claim a runtime or
real-weight parity yet; the public legacy GGUF remains an explicit boundary.
No C symbol, ownership rule or allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-convert::models::speecht5` | `convert_speecht5_file_with_tokenizer`, `SpeechT5Report`; existing `convert_speecht5_file` | Added/tightened | `(input, output, license, spm_char_model) -> Result<SpeechT5Report, ConvertError>`; compatibility entry point now returns an explicit tokenizer-required error | Requires the fixed upstream revision, exact 393 F32 tensor names/shapes, exact 79-piece SentencePiece CHAR model plus fixed `<mask>`/`<ctc_blank>` IDs 79/80, and MIT provenance metadata. The preparation wrapper removes only five exact scalar integer BatchNorm training counters. Arbitrary/partial checkpoints and tokenizer-less conversion fail before output | yes for the formerly loose offline conversion input contract | (TBD) |
| `gguf:vokra.speecht5.*` | 50-key identity/topology/tokenizer group | Added/tightened | immutable hashes and source revision; typed topology/generation scalars; exact raw SentencePiece model, tokenizer strings/scores/IDs and normalizer contract | The original 13 topology keys keep their meanings. New artifacts carry 37 additive keys. A historical tokenizer-less public file is not promoted to executable compatibility | no for canonical new conversion; legacy output remains explicitly unsupported | (TBD) |
| `vokra-models::speecht5` | `SpeechT5Checkpoint`, `SpeechT5Config`, `SpeechT5Tokenizer` | Added | cheap strict 393-tensor checkpoint identity/topology bind plus exact expanded CHAR vocabulary validation and `encode_text` | Raw tokenizer SHA, expanded piece/score manifest, all 50 model-specific keys and MIT class must match before the handle is returned. Ordinary already-normalized English text is encoded with EOS; unsupported NMT-NFKC rewrites and training-only added tokens return explicit errors. The complete numerical runtime is recorded in the later SpeechT5 CPU/Metal entry above; this cheap handle remains available without decoding weights | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (Canary-1B-v2 CPU/Metal ASR and AST)

The immutable `nvidia/canary-1b-v2` main checkpoint now has a strict native
PCM-to-token/text route over its authenticated 1,478 F32 inference tensors,
16,384-piece aggregate tokenizer and exact Canary2 prompt. CPU and Metal share
one whole-model hot-op registry; an unavailable or uncovered backend returns
an explicit error. The auxiliary timestamp checkpoint is rejected rather than
silently selected. Independent VAST CPU and Apple-device real-weight Metal
parity remain pending. No C ABI symbol is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::canary` | `CanaryAsr`, `Canary1bV2Options`, `CanaryLanguage`, `CanaryEmotion`, `CanaryTokenizer`, `CANARY_HOT_OPS` | Changed | strict complete bind; CPU/Metal ASR/AST greedy token and text forward | Replaces the earlier metadata-only scaffold; incomplete or mismatched artifacts fail their complete-manifest gate | no | (TBD) |
| `gguf:vokra.canary.*` | complete topology, immutable provenance, and aggregate tokenizer vocabulary/hash metadata | Added | released encoder/decoder/head axes plus authenticated `u8[]` decode table | Strict converter selects only the official main checkpoint and requires the exact aggregate tokenizer | no | (TBD) |
| `vokra-cli convert`, `run`, `bench` | `canary` tokenizer sidecar, 25-language ASR/AST dispatch, and `--target-language` | Added | 16 kHz mono; equal language pair = ASR, different pair = AST | Unsupported search/backend/language combinations return explicit errors; no alternate model or CPU fallback | no | (TBD) |
| VAST parity worker | `run-canary-1b-v2-validation.sh` and official NeMo dumper | Added | exact English ASR and English-to-German AST greedy token equality | Large artifacts stay remote; worker performs no upload/push and evidence is pulled before instance destruction | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (Canary-1B-Flash CPU/Metal ASR and AST)

The complete immutable `nvidia/canary-1b-flash` release now has a strict
native PCM-to-token/text route over its 1,374 F32 inference tensors, aggregate
5,248-piece tokenizer and exact Canary2 prompt. CPU and Metal share one
whole-model hot-op registry; an unavailable or uncovered backend returns an
explicit error. The historical public `vokra/canary-1b-flash` GGUF remains an
encoder-only 1,292-tensor artifact and is rejected rather than completed with
fabricated state. The independent VAST NeMo ASR/AST token gate and Apple-device
real-weight Metal parity remain pending. No C ABI symbol is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::canary_1b_flash` | `Canary1bFlashAsr`, `Canary1bFlashOptions`, `CanaryLanguage`, `CanaryEmotion`, `CanaryTokenizer`, `CANARY_1B_FLASH_HOT_OPS` | Changed | strict complete bind; CPU/Metal ASR/AST token and text forward | Replaces the earlier loud partial; exact legacy encoder-only artifact now fails its complete-manifest gate | no | `f2f45297` |
| `gguf:vokra.canary_1b_flash.*` | complete topology and aggregate tokenizer vocabulary/hash metadata | Added | released encoder/decoder/head axes plus authenticated `u8[]` decode table | Canonical complete conversion is self-describing; missing/conflicting metadata or tokenizer fails closed | no | `48dda91a` |
| `vokra-cli run`, `vokra-cli bench` | `canary-1b-flash` ASR/AST dispatch; `--target-language` | Added | 16 kHz mono, `en/de/es/fr`; equal language pair = ASR, different pair = AST | Unsupported language/search/backend combinations return explicit errors; no alternate model or CPU fallback | no | `f2f45297` |
| VAST parity worker | `run-canary-1b-flash-validation.sh` and official NeMo dumper | Added | exact English ASR and English→German AST greedy token equality | Large artefacts stay remote; worker performs no upload/push and evidence is pulled before instance destruction | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (Parakeet-TDT-1.1B CPU/Metal runtime)

The immutable 1,667-F32-tensor `nvidia/parakeet-tdt-1.1b` release now has a
strict native PCM-to-text route. Its 42-layer FastConformer, two-layer LSTM
prediction network and 1,030-wide token-plus-duration joint reuse the shared
Parakeet CPU/Metal execution path. The official NeMo config declares no EOS;
the Rust config therefore represents EOS as optional rather than inventing a
token id. The exact public legacy GGUF is authenticated by its complete
name/shape manifest and accepts the pinned 11 KB plaintext SentencePiece
vocabulary as a sidecar. Real-weight VAST and Apple-device parity remain
pending and are not implied by source reachability. No C ABI symbol is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::parakeet` | `ParakeetJointConfig::eos_token_id` | Changed | `u32` → `Option<u32>` | Preserves `Some(3)` for 0.6B-v3 and models the original 1.1B release as `None`; decode skips/stops only when an EOS is actually declared | yes (Rust source) | (TBD) |
| `vokra-models::parakeet_tdt_1_1b` | `ParakeetTdt11b`, `ParakeetTdt11bWeights`, verified config/duration constants | Changed | strict `from_gguf*`, CPU/Metal backend, encoder/head/token/text transcription | Replaces the decoder-only loud partial with a complete native route over the exact 1,667-tensor manifest; no backend fallback | no | (TBD) |
| `gguf:vokra.parakeet_tdt_1_1b.*` | source, frontend, encoder, decoder, joint and tokenizer metadata group | Added | 25 `u32`, one `f32`, three identity/activation strings, five duration entries, optional `u8[]` tokenizer vocabulary plus SHA-256 | New artifacts are self-describing. The exact historical public manifest may omit the group; a partial or conflicting group fails closed | no | (TBD) |
| `vokra-cli run`, `vokra-cli bench` | `parakeet-tdt-1_1b` ASR dispatch | Added | 16 kHz mono; optional authenticated `--tokenizer tokenizer.vocab`; greedy TDT only | The 4.28 GB payload binds once. Unsupported search modes/backends and missing text tokenizer return explicit errors | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (Nemotron 3.5 ASR offline CPU/Metal runtime)

The exact 655-tensor Nemotron-3.5-ASR-Streaming-0.6B release now has a strict
native offline PCM-to-text route. The released causal/chunk-limited
FastConformer, prompt projector and RNN-T prediction/joint networks execute
through one selected CPU or Metal backend. Stateful cache streaming remains
an explicit unsupported boundary and is never replaced with whole-utterance
or CPU fallback. The public legacy GGUF remains token-decodable with an
authenticated tokenizer sidecar; new conversions may embed the official
`tokenizer.json`. Real-weight VAST and Apple-device numerical parity remain
pending and are not implied by this source-reachability entry. No C ABI symbol
is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::nemotron_asr_streaming` | `NemotronAsr`, `NemotronAsrConfig`, identity/config constants, `NEMOTRON_HOT_OPS`, `prompt_id_for_language` | Added | strict `from_gguf*`, backend/license/config accessors, token/text transcription with prompt selection, explicit stateful-streaming error | Authenticates the exact public manifest and executes causal FastConformer + prompt-conditioned greedy RNN-T through one CPU/Metal backend; unsupported backends fail before inference | no | `e751995b` |
| `gguf:vokra.nemotron_asr.*` | frontend, encoder, decoder, joint and prompt configuration group; optional `tokenizer.json` | Added | `u32`, `f32`, `string` values covering the audited release and byte-exact tokenizer JSON | New artifacts are self-describing. The exact historical public manifest may omit the group and use an authenticated tokenizer sidecar; a partial group fails closed | no | `0705e39c` |
| `vokra-cli run`, `vokra-cli bench` | `nemotron_asr_streaming` ASR dispatch | Added | 16 kHz mono input, optional `--language` prompt and `--tokenizer` sidecar; greedy decode only | Stateful streaming and unsupported decode modes return explicit errors; no alternate ASR model or CPU fallback is selected | no | `393ad24b` |

### 2026-08-26 — 1.0.0-rc.1-dev (MossFormer2-SS-16K CPU/Metal separator)

The exact public ClearerVoice-Studio checkpoint now has a strict native
two-speaker separation runtime. CPU and Metal share one complete learned-op
registry; the released ScaleNorm equation has dedicated kernels rather than a
host reduction, while unwired backends fail before execution. The independent
official full-model numerical comparison remains a recorded VAST gate and is
not implied by the landed code path. No C ABI symbol or new GGUF key is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-backend-cpu::kernels` | `scale_norm_f32` | Added | `(&[f32], &mut [f32], rows, cols, gain, eps) -> Result<()>` | Portable reference for `x / max(||x||₂·cols⁻¹ᐟ², eps)·gain`; distinct from RMSNorm | no | `35ae7689` |
| `vokra-backend-metal::MetalContext` | `scale_norm_f32` | Added | matching row-wise MSL dispatch | Apple-silicon differential test observed max absolute CPU/Metal error `4.768e-7` against the unchanged FP32 `0.01` bound | no | `35ae7689` |
| `vokra-models::compute` | `HotOp::ScaleNorm`, `Compute::scale_norm_f32` | Added | whole-model coverage plus typed imperative dispatch | CPU and Metal are covered; CUDA/WebGPU and other unwired backends remain explicit unsupported operations | no | `35ae7689` |
| `vokra-models::mossformer2_ss_16k` | `Mossformer2Ss16k`, identity/topology constants, `MOSSFORMER2_HOT_OPS` | Added | strict `from_gguf*`, backend/license/rate accessors, `separate(&[f32]) -> Result<Vec<Vec<f32>>>` and `SeparationEngine` | Authenticates the exact 1,076-tensor public artifact and executes encoder, FLASH, gated FSMN, mask and decoder on one CPU/Metal route; full-model VAST measurement remains pending | no | `64fc24d4` |
| `vokra-cli run`, `vokra-cli bench` | `mossformer2_ss_16k` separation dispatch | Added | 16 kHz mono input to two separated PCM streams | Foreign/partial manifests and unavailable backends fail explicitly; no alternate separator is substituted | no | `64fc24d4` |

### 2026-08-26 — 1.0.0-rc.1-dev (SpeechTokenizer CPU/Metal decoder)

The exact public Fudan/OpenMOSS SpeechTokenizer checkpoint now exposes native
frame-major residual-VQ token-to-waveform decode. CPU and Metal select the same
whole-model hot-op inventory; uncovered backends return explicit errors and
PCM encode remains unavailable rather than substituting another model. No C
ABI symbol or GGUF key is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::speechtokenizer` | `SpeechTokenizer`, identity/topology constants, `SPEECHTOKENIZER_DECODE_HOT_OPS` | Added | strict `from_gguf*` / `open`, backend/license/rate accessors, `decode_frame_major(&[u32], frames, num_quantizers)` | Authenticates the exact 166-tensor public artifact and runs 16 kHz RVQ + weight-normalized SEANet on CPU/Metal without fallback | no | (TBD) |
| `vokra-cli run` | `speechtokenizer` dispatch and decode mode | Added | `--codec-mode decode --num-quantizers <1..8> --input <codes.u32le> --output <out.wav>` | Encode and malformed matrices fail explicitly; bench points to the discrete-code run contract | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (FunCodec CPU/Metal decoder)

The exact public Alibaba DAMO FunCodec checkpoint now exposes native
frame-major residual-VQ token-to-waveform decode. CPU and Metal select the same
whole-model hot-op inventory; uncovered backends return explicit errors and
PCM encode remains unavailable rather than substituting another model. No C
ABI symbol or GGUF key is added.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::funcodec` | `FunCodec`, identity/topology constants, `FUNCODEC_DECODE_HOT_OPS` | Added | strict `from_gguf*` / `open`, backend/license/rate accessors, `decode_frame_major(&[u32], frames, num_quantizers)` | Authenticates the exact 230-tensor public artifact and runs 16 kHz RVQ + SEANet on CPU/Metal without fallback | no | (TBD) |
| `vokra-cli run` | `funcodec` dispatch and decode mode | Added | `--codec-mode decode --num-quantizers <1..32> --input <codes.u32le> --output <out.wav>` | Encode and malformed matrices fail explicitly; bench points to the discrete-code run contract | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (GPT-2 `gelu_new` CPU/Metal op)

The imperative backend seam now distinguishes Transformers/GPT-2
`gelu_new` from exact/erf GELU. CPU implements the official tanh
approximation as its portable reference; Metal dispatches a dedicated MSL
kernel and passes the fixed `2e-6` backend-parity bound. CUDA/WebGPU and other
unwired selections remain explicit unsupported operations. No C ABI or GGUF
schema changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-backend-cpu::kernels` | `gelu_new_f32` | Added | `(&[f32], &mut [f32]) -> Result<()>` | Portable scalar reference for the official GPT-2 tanh equation; distinct from existing exact `gelu_f32` | no | (TBD) |
| `vokra-backend-metal::MetalContext` | `gelu_new_f32` | Added | matching element-wise MSL dispatch | CPU/Metal max-absolute error must remain `<= 2e-6` on the committed wide-range test | no | (TBD) |
| `vokra-models::compute` | `HotOp::GeluNew`, `Compute::gelu_new_f32` | Added | whole-model backend coverage plus typed imperative dispatch | Metal is covered; unwired backends fail without CPU fallback | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (MOSS-VoiceGenerator topology correction)

The converter no longer infers 8B/32-codebook Delay axes from the shared
upstream `moss_tts_delay` class. MOSS-VoiceGenerator is a separately named
Qwen3-1.7B/16-codebook release. This source correction does not replace or
upload the already-public GGUF with its stale legacy header.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-convert::ModelKind` | `MossVoiceGenerator` | Added | dedicated CLI/converter selector | Existing VoiceGenerator aliases now preserve the official release instead of routing through `MossTts` | behavior correction | (TBD) |
| `vokra-convert::models::moss_tts` | `MossTtsVariant::VoiceGenerator` | Added | Qwen3 2048/6144/28/16Q/8KV, 16 codebooks, 24 kHz | Pinned to official revision `97521ec2…`; distinct from Delay despite shared upstream class | no | (TBD) |
| `gguf:vokra.model.*`, `gguf:vokra.provenance.*`, `gguf:vokra.moss_tts.*` | VoiceGenerator identity and axes | Corrected | name `moss-voice-generator`, upstream `OpenMOSS-Team/MOSS-VoiceGenerator`, variant `voice_generator` | Future conversions are faithful; the published legacy artifact remains explicitly stale | yes for code relying on the incorrect 8B header | (TBD) |
| `gguf:vokra.provenance.*`, `gguf:vokra.moss_tts.*` | VoiceGenerator fixed source and framing contract | Added | revision `97521ec2…`, config/modeling/processing SHA-256, max positions 40960, nine framing/code IDs and Full codec upstream | Corrected conversions pin the official 1.7B source rather than inheriting family defaults; no checkpoint digest is invented without the actual source file | no | (TBD) |
| `vokra-models::moss_tts` | `MossVoiceGeneratorCheckpoint` | Added | mmap strict bind for the exact 343-tensor manifest `9a874f01…`, dense F32/F16/BF16 only | Corrected metadata binds directly; the one historical public `moss-tts`/8B header is accepted only behind the complete VoiceGenerator manifest and surfaces `requires_metadata_repair=true` | no | (TBD) |
| `vokra-models::moss_tts` | `MossTtsDelayCheckpoint`, `MossTtsDelayRelease` | Added | mmap-backed strict bind for Base/v1.5, complete 463-tensor manifest `5a3578fb…`, dense F32/F16/BF16 only | Keeps the ~17 GB payload mapped and validates every descriptor without widening; the mapping is retained by the native generation runtime | no | (TBD) |
| `vokra-models::moss_tts` | `MossTtsDelay`, `MossTtsDelayLogits`, `MOSS_TTS_DELAY_HOT_OPS` | Added | explicit `[rows,33]` token prompt to last-position text `[155648]` plus audio `[32,1025]` logits | Base/v1.5 Qwen3 runs with chunked mmap embeddings/heads, one-layer materialization and KV cache; GEMM/GEMV/softmax/RMSNorm/SiLU preflight and execute on one CPU/Metal backend with no hidden learned-op fallback | no | (TBD) |
| `vokra-models::moss_tts` | `MossTtsDelayGenerationOptions`, `MossTtsDelayGeneration`, `MossTtsDelay::generate_delay_rows` | Added | seeded text/audio sampling over raw `[rows,33]` prompts; returns appended delayed rows | Ports the fixed-revision upstream audio-length/delay-length masks, 32-step drain and forced audio-end order; default temperature/top-k/top-p match upstream while Vokra owns deterministic RNG | no | (TBD) |
| `vokra-models::moss_tts` | `MossTtsDelayAudioSegment`, `MossTtsDelaySynthesis`, `MossTtsDelay::synthesize_prompt_rows` | Added | Delay generation plus official 32-codebook de-delay, all-pad segment split and Full codec decode | Requires the exact Full companion and one shared CPU/Metal backend; continuation audio is decoded causally before the official waveform-ratio prefix trim; zero/multiple segments remain explicit in the library API | no | (TBD) |
| `vokra-models::moss_tts` | `MossVoiceGenerator` | Added | explicit `[rows,17]` prompt to Qwen3-1.7B text `[155648]` plus audio `[16,1025]` logits, official 16-step delay generation and Full codec decode | Reuses the topology-parameterized Delay algorithm under the strict 343-tensor manifest; VoiceGenerator sampling defaults remain release-specific; CPU/Metal preflight rejects any missing learned op without fallback | no | (TBD) |
| `vokra-cli run` | MOSS-TTS Base/v1.5 route | Added | raw `[rows,33]` u32le prompt IDs plus `--audio-tokenizer <full.gguf>` and a generation cap to 24 kHz mono WAV | Refuses raw text, a non-Full sidecar, mixed backends, and zero/multiple-segment output for the single-WAV command; VAST real-weight parity remains pending | no | (TBD) |
| `vokra-cli run` | MOSS-VoiceGenerator route | Added | raw `[rows,17]` u32le prompt IDs plus `--audio-tokenizer <full.gguf>` and a generation cap to 24 kHz mono WAV | Routes corrected identity directly and the exact historical stale header by provenance ID before strict manifest bind; VAST real-weight parity remains pending | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (MOSS-TTS Nano contract correction)

The MOSS-TTS Nano converter now follows the pinned custom GPT-2 source rather
than the former generic-GPT-2 assumption. Nano uses rotary positions with base
10,000, not learned absolute positions. New conversions also stamp the exact
upstream revision/checkpoint/config identity, LayerNorm and local-transformer
axes, framing token IDs and required MOSS Audio Tokenizer Nano companion. The
already-public 194-tensor GGUF keeps its old `rope_base = 0` header and must be
treated as a manifest-scoped legacy artifact requiring metadata replacement;
this entry does not authorize an upload.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `gguf:vokra.moss_tts.llm.rope_base` | Nano value | Corrected | `f32`: `0` → `10000` | Fixes a pre-runtime converter bug against pinned upstream revision `44502f80…`; old public GGUF is never accepted by metadata alone | yes for code that depended on the incorrect Nano value | (TBD) |
| `gguf:vokra.moss_tts.*` | `config_sha256`, `llm.position_embedding_type`, `llm.layer_norm_eps`, `llm.max_position_embeddings`, `local_transformer_layers`, eight framing token IDs, `audio_tokenizer_upstream_hf` | Added | strings, `f32`, `u32` | Nano-only additive contract; other MOSS variants do not receive guessed values | no | (TBD) |
| `gguf:vokra.provenance.*` | `upstream_revision`, `checkpoint_sha256` | Added | pinned 40-hex revision and 64-hex checkpoint digest | Nano-only provenance identity; exact historical artifact may omit only behind its complete manifest | no | (TBD) |
| `vokra-models::moss_tts` | `MossTtsNano`, release/topology constants, `MOSS_TTS_NANO_HOT_OPS` | Added | strict `open` / `from_gguf*`, backend/license/repair accessors and owned 194-tensor weight binding | Accepts the old header only behind manifest SHA-256 `125c074b…`; partial corrected metadata fails closed and no other MOSS topology is inferred | no | (TBD) |
| `vokra-models::moss_tts` | `MossTtsGeneratedCodes`, `MossTtsNano::generate_codes` | Added | explicit frame-major `[rows,17]` prompt IDs plus a frame cap to greedy `[frames,16]` codes | Runs the official no-KV-cache global/local GPT-2 order on one CPU/Metal backend; tokenizer/template and codec remain explicit companions | no | (TBD) |
| `vokra-models::moss_tts` | `MossTtsSynthesis`, `MossTtsNano::synthesize_prompt_rows` | Added | explicit prompt matrix + exact `MossAudioTokenizer` companion to codes and 48 kHz stereo PCM | Requires Nano and identical selected backends; Full substitution and mixed Metal/CPU composition are explicit errors | no | (TBD) |
| `vokra-cli run` | `moss_tts` Nano task, `--audio-tokenizer`, `--max-new-frames` | Added | raw `[rows,17]` u32le prompt IDs plus exact codec sidecar to stereo WAV | Refuses raw text because `tokenizer.model` is not embedded; non-Nano family names and mixed backends fail explicitly | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (MOSS Audio Tokenizer strict runtime)

The two public OpenMOSS codec artifacts now have a fail-closed native runtime
identity. Full and Nano are selected from complete tensor manifests rather
than their shared upstream Python class. Nano exposes variable-quantizer
token-to-48-kHz-stereo decode. Full exposes a distinct 24-kHz-mono decoder that
retains the 7 GB GGUF mapping, materialises one Transformer layer at a time and
tiles its local causal attention window. Both execute every learned reduction
on one selected CPU/Metal backend. The historically mis-stamped public Nano
artifact is recognized only behind its exact manifest and remains visibly
marked for replacement. Both encode paths stay explicit unsupported
operations. The Full source still requires VAST real-weight parity before a
release claim; no model upload is authorized by this entry.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::moss_audio_tokenizer` | `MossAudioTokenizer`, `MossAudioTokenizerVariant`, `MossDecodedAudio`, identity/topology constants, `MOSS_AUDIO_TOKENIZER_NANO_HOT_OPS` | Added | strict `open` / `from_gguf*`, backend/license/variant accessors, frame-major `[frames, num_quantizers]` decode to interleaved stereo PCM | Owns decoded Nano weights and output PCM; Full has no substitute route; unsupported backends never fall back to CPU | no | `a22d0868`, `fea90206` |
| `vokra-models::moss_audio_tokenizer` | `MOSS_AUDIO_TOKENIZER_FULL_HOT_OPS`, `open_mapped_with_backend`, `from_gguf_mapped`, `from_gguf_mapped_with_backend` | Added | mapping-owning Full bind; `[frames,1..=32]` LFQ codes to 1,920 mono samples/frame at 24 kHz | Full LFQ + 68 Transformer layers use bounded one-layer materialisation and tiled local attention; borrowed Full binds remain an explicit non-decode surface rather than copying 7 GB | no | `e17177ac` |
| `vokra-models::moss_audio` | `MossAudioCheckpoint`, `MossAudio`, `MossAudioVariant`, fixed topology structs, `MossAudioEmbeddings`, `MossAudioGenerationOptions`, `MossAudioTokenOutput`, `MOSS_AUDIO_*_HOT_OPS` | Added | exact 4B/8B 901-tensor mmap bind; 16 kHz PCM plus authenticated prompt-token sequence to greedy Qwen3 token output | historical `moss_tts` stamp is exact-manifest-only; audio tower/adapters/DeepStack Qwen3 decode use one explicit CPU/Metal backend; absent tokenizer/chat-template string route and quantized artifacts fail loudly | no | working tree (2026-08-27) |
| `vokra-core::Session` | `gguf_arc` | Added | `&Session -> Arc<GgufFile>` | Cheap shared-handle clone for mapping-owned concrete models; tensor payloads are not copied | no | `e17177ac` |
| `vokra-cli run` | `moss_audio_tokenizer` task, `--num-quantizers`, multichannel float WAV writer | Added | raw u32le frame-major LFQ matrix to Full 24 kHz mono or Nano 48 kHz stereo WAV | Public Nano's legacy metadata warning is preserved; encode stays explicit unsupported | no | `0c44bf61`, `e17177ac` |
| `tools/parity`, `vokra-models/tests` | `moss_audio_tokenizer_dump_reference.py`, `parity_moss_audio_tokenizer_full_real` | Added | fixed-revision official custom-code decode with quantizer/stage/final taps; ignored real-GGUF consumer at FP32 `atol=0.01` | VAST-only independent oracle/consumer sources; Full upstream source hashes are pinned; no fixture or parity result is claimed before an actual run | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (MusicGen public composite LM execution)

The already-published Transformers-composite MusicGen Small/Melody artifacts
gain the same mapping-owned raw-code generation surface as the AudioCraft
Medium/Large files. Split Q/K/V tensors, the checkpointed sinusoidal position
table, Transformers mask semantics and four codebook heads are authenticated
without copying the full decoder into memory. No C ABI or GGUF schema changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::audiocraft_lm` | `AudioCraftLmDecoder` mapped layout support | Changed | Transformers split-Q/K/V LM step and CFG/delay sampling | Mapping-owned only; CPU/Metal selected once; unsupported backends fail explicitly | no | (TBD) |
| `vokra-models::musicgen::MusicGen` | `from_path*`, `prepare_lm_condition`, `new_lm_state`, `lm_step_into`, `generate_codes` | Changed | raw T5-hidden-to-four-codebook route for public Small/Melody composites | Text tokenization/composition remains a loud companion boundary | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (MusicGen embedded T5 token-id route)

The public Transformers-composite Small/Melody artifacts now bind their
embedded canonical T5-base encoder on the same selected CPU/Metal backend.
Callers provide conditional and unconditional token ids explicitly because
the GGUFs do not embed tokenizer assets; no tokenizer or null prompt is
guessed. No C ABI or GGUF schema changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::musicgen::MusicGen` | `prepare_text_condition`, `generate_codes_from_token_ids`, `generate_from_token_ids` | Added | explicit T5 token ids + optional masks to prepared condition, frame-major codes or mono 32 kHz PCM | Mapping-owned Small/Melody composites only; Medium/Large return explicit missing-companion errors | no | (TBD) |
| `vokra-cli run` | `musicgen` task; `--music-unconditional-token-ids`, `--music-frames`, `--music-seed` | Added | conditional/null T5 ids plus 50 Hz frame count to WAV | Small/Melody execute; legacy AudioGen and LM-only Medium/Large retain explicit companion errors | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (MusicGen embedded EnCodec waveform decode)

The already-published Transformers-composite MusicGen Small/Melody artifacts
gain a native 32 kHz token-to-waveform component. This is runtime consumption
of tensors already covered by the artifact's non-commercial compliance gate;
it adds no standalone EnCodec converter, zoo artifact, metadata key, C symbol,
or publication permission.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::audiocraft_encodec` | `AudioCraftEncodecDecoder`, `AUDIOCRAFT_ENCODEC_HOT_OPS`, `SAMPLE_RATE`, `FRAME_HOP`, `NUM_CODEBOOKS`, `CODEBOOK_SIZE`, `DIMENSION` | Added | authenticated frame-major `[frames, 4]` EnCodec codes to mono `frames*640` FP32 PCM | Public construction remains through a compliance-checked mapping-owned MusicGen composite; no standalone weight path | no | (TBD) |
| `vokra-models::musicgen` | `decode_codes`, `decode_frame_major` | Added | composite Small/Melody codec component API | LM-only Medium/Large return an explicit companion-required error; no fallback | no | (TBD) |
| `vokra-models::compute` | `HotOp::EncodecRvq` Metal coverage | Changed | same shape-generic gather/fold kernel as Mimi RVQ, with EnCodec-specific host validation | CPU unchanged; Metal is now real; CUDA/WebGPU remain explicit unsupported | no | `cb3f2f67` |

### 2026-08-26 — 1.0.0-rc.1-dev (AudioCraft delayed-code generation)

The mapping-owned AudioCraft LM route now composes its raw logits with the
authenticated four-codebook delay pattern, two-state classifier-free guidance
and deterministic host sampling. The output stops at frame-major EnCodec
indices; prompt tokenization and waveform decode remain explicit companion
boundaries. No C ABI or GGUF metadata changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::audiocraft_lm` | `AudioCraftGenerationConfig`, `AudioCraftGeneratedCodes`, `AUDIOCRAFT_DEFAULT_CFG_COEF`, `AUDIOCRAFT_DEFAULT_TEMPERATURE`, `AUDIOCRAFT_DEFAULT_TOP_K`, `AudioCraftLmDecoder::generate_codes` | Added | prepared conditional/null conditions + generation controls to `[frames, num_codebooks]` frame-major `u32` codes | Runs CFG 3.0 and top-k 250 defaults; strips delay/special tokens; extreme sequences shorter than the authenticated complete-delay geometry fail explicitly | no | (TBD) |
| `vokra-models::{musicgen,audiogen}` | `generate_codes` | Added | family wrappers over the shared mapped decoder | Available only on mapping-owned AudioCraft layouts and one selected CPU/Metal backend; prompt and codec companions are not implied | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (AudioCraft mapped autoregressive LM step)

The public AudioCraft-layout MusicGen Medium/Large and AudioGen Medium GGUFs
gain a mapping-owned, bounded-memory raw autoregressive LM route. This change
does not claim prompt tokenization, delay-pattern sampling or waveform decode,
and it does not alter the C ABI or GGUF metadata schema.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::audiocraft_lm` | `AudioCraftLmConfig`, `AudioCraftCondition`, `AudioCraftLmDecoder`, `AudioCraftLmState`, `AUDIOCRAFT_LM_HOT_OPS` | Added | mapped F32/F16/BF16 condition projection, state creation and one-position codebook-major logits step | Decoder owns the GGUF mapping; conditions and states are checkpoint-bound; learned reductions use one selected CPU/Metal backend with no fallback | no | (TBD) |
| `vokra-models::musicgen::MusicGen` | `from_path*`, `backend`, `prepare_lm_condition`, `new_lm_state`, `lm_step_into` | Added | raw T5-hidden-to-four-codebook-logits path for authenticated AudioCraft Medium/Large layouts | Strict policy refuses NC weights without research opt-in; borrowed handles return explicit unsupported errors | no | (TBD) |
| `vokra-models::audiogen::AudioGen` | `from_path*`, `backend`, `prepare_lm_condition`, `new_lm_state`, `lm_step_into` | Added | raw T5-large-hidden-to-four-codebook-logits path for the authenticated AudioCraft layout | Strict policy refuses NC weights without research opt-in; borrowed handles and unsupported backends fail explicitly | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (AudioGen 16 kHz topology correction)

New AudioGen conversions stamp the complete native transformer and codec
topology. The runtime's historical fallback is corrected from MusicGen's
32 kHz companion to the official AudioGen 16 kHz / stride-320 contract. No C
symbol, ownership rule or allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `gguf:vokra.audiogen.*` | `d_model`, `num_layers`, `n_heads`, `ffn_dim`, `vocab_size`, `num_codebooks`, `codec_frame_rate_hz`, `sample_rate_hz` | Added / corrected | eight `u32` values: `1536 / 48 / 24 / 6144 / 2048 / 4 / 50 / 16000` | New converters stamp the complete group. The exact historical 588-F16-tensor public AudioGen GGUF may omit it only because its identity, provenance and complete manifest are pinned; missing keys use the same primary-source defaults. | yes only for callers that relied on the incorrect 32 kHz fallback; no for exact artifact binding | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (exact pyannote diarization execution)

The public speaker-diarization 3.1 contract now executes the exact pinned
PyanNet + masked WeSpeaker + centroid-clustering pipeline on CPU or Metal and
is routed by `vokra-cli run` and `bench` using explicit dependency paths. The
former generic diarization scaffold is removed because its invented thresholds,
whole-region embeddings and average linkage did not describe the released
pyannote pipeline. Unsupported backends fail explicitly; no dependency model
falls back to CPU. No C symbol, allocation rule or ownership ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::pyannote::diarization` | `SpeakerEncoder`, `DiarizationPipeline` | Removed | former generic encoder trait and threshold-driven pipeline | Intentional prerelease correction: the removed scaffold used an algorithm and defaults that are not present in the exact released model; callers must use the strict three-GGUF handle | yes (Rust source) | (TBD) |
| `vokra-models::pyannote::diarization` | `PyannoteSpeakerDiarization31`, `ARCH`, `SAMPLE_RATE`, `SEGMENTATION_{WINDOW_SAMPLES,STEP_SAMPLES,FRAMES}`, `LOCAL_SPEAKERS` | Added | `open(pipeline, segmentation, embedding, BackendKind)`, `diarize(&[f32], u32)`, `diarize_to_rttm(&[f32], u32, &str)` plus exact geometry constants | Binds all three audited artifacts, executes the pinned 5 s / 0.5 s / 293-frame algorithm, and keeps CPU/Metal selection observable with fail-closed backend preflight | no | (TBD) |
| `vokra-models::wespeaker` | `WeSpeaker::from_path_with_backend` | Added | `(path, BackendKind) -> Result<WeSpeaker>` | Performs eager hot-op availability checks for a composed pipeline before audio execution; no fallback | no | (TBD) |
| `vokra-cli run` / `bench` | `arch=pyannote-speaker-diarization`, `--segmentation-model`, `--embedding-model` | Added (behavior only) | weightless pipeline GGUF plus two required dependency GGUF paths and 16 kHz mono WAV | The CLI never downloads or guesses dependency artifacts; missing sidecars, wrong sample rate and unsupported backends are explicit errors | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (pyannote masked WeSpeaker pooling)

The native WeSpeaker handle now accepts one PyanNet local-speaker activity
mask and applies the pinned pyannote.audio 3.1.1 weighted `StatsPool` contract.
The existing unmasked speaker API and C ABI remain unchanged.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::wespeaker` | `WeSpeaker::{embed_pcm_masked,embed_features_masked}` | Added | `(&self, pcm/features, rate/frames, mask: &[f32]) -> Result<Vec<f32>>` | Nearest-resizes a finite `[0,1]` local-speaker mask to the final ResNet time axis and applies upstream weighted unbiased statistics before the existing CPU/Metal projection; unmasked methods retain their prior behavior | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (pyannote centroid speaker clustering)

The host-side speaker clustering primitive now implements the exact centroid
geometry, inclusive SciPy distance cut and minimum-cluster-size post-processing
required by the public pyannote speaker-diarization 3.1 configuration. No C
symbol, allocation rule or ownership ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-ops::clustering` | `AgglomerativeClustering::cluster` | Fixed | `cluster(&self, &[Vec<f32>]) -> Vec<usize>` | Distance exactly equal to `threshold` now merges, matching SciPy `fcluster(..., criterion="distance")`; the former strict-less-than boundary was incompatible with the pinned upstream | yes (intentional pre-1.0 behavior correction) | (TBD) |
| `vokra-ops::clustering` | `AgglomerativeClustering::cluster_with_min_cluster_size` | Added | `(&self, &[Vec<f32>], usize) -> Vec<usize>` | Applies Python ties-to-even short-input heuristic, nearest-large-centroid reassignment and the explicit no-large-cluster fold used by pyannote.audio 3.1.1 | no | (TBD) |
| `vokra-ops::clustering` | `LinkageMethod::Centroid` | Added | public enum variant | Cosine rows are L2-normalized before Euclidean arithmetic-centroid linkage, matching the exact public pipeline contract; adding a variant can break an exhaustive downstream match | yes (intentional pre-1.0 enum extension) | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (pyannote diarization pipeline contract)

The public zero-tensor `pyannote-speaker-diarization-3.1` GGUF now has a
strict runtime metadata binder. This change does not yet claim end-to-end
diarization: it establishes the fail-closed orchestration contract that the
subsequent PyanNet + masked WeSpeaker + centroid-clustering execution route
must consume. No C symbol, allocation rule or ownership ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::pyannote::diarization` | `PyannoteSpeakerDiarization31Config`, `PIPELINE_{MODEL_TAG,NAME,CATEGORY,UPSTREAM_HF,SOURCE_REVISION,PUBLIC_HF,PUBLIC_REVISION,PUBLIC_FILE,PUBLIC_BYTES,PUBLIC_SHA256}` | Added | `from_gguf(&GgufFile) -> Result<Self>` plus immutable public/source identity constants | Accepts only the exact MIT/permissive zero-tensor pipeline identity and all 13 typed `vokra.pyannote_pipeline.*` values; a foreign tensor payload, missing key, wrong type, model reference or clustering method fails closed | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (deepfake Wav2Vec2 CPU/Metal runtime)

The exact public 215-F32-tensor deepfake-audio-detection-v2 artifact gains a
strict native waveform-to-binary-classification forward plus CLI run and bench
routes. The corrected implementation follows the pinned official
`Wav2Vec2ForSequenceClassification` contract and retains the audited legacy
GGUF through its immutable manifest and generic Apache-2.0 provenance. No C
symbol, ownership rule or allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::deepfake_detection` | `DeepfakeDetection`, `DeepfakeDetectionConfig`, `DeepfakeDetectionWeights`, `DeepfakeScore`, topology/identity constants | Tightened/completed | strict open/bind/policy/backend selection; `encode_features(&[f32], 16_000)` and `score_pcm(&[f32], 16_000)`; result order is `0=fake, 1=real` | Owns decoded F32 weights. Conv1D/grouped-Conv1D/GEMM/Softmax/LayerNorm/GELU honor explicit CPU/Metal selection; unsupported backends fail with no fallback. Existing score/threshold APIs remain source-compatible; invalid thresholds now fail explicitly | no for the exact audited public file; malformed, partial or foreign files fail closed | (TBD) |
| `vokra-cli run` / `bench` | `arch=deepfake_detection` | Added (behavior only) | 16 kHz mono WAV to ranked `[fake, real]` logits/scores; optional output is two little-endian f32 scores | Wrong sample rate, non-finite/short PCM, provenance/manifest drift and unsupported backend fail explicitly; the CLI does not select a deployment threshold | no | (TBD) |
| `vokra-models::wav2vec2_ctc` | shared positional-convolution binder | Internal extension | accepts exactly one complete weight-norm pair: legacy `weight_g/weight_v` or PyTorch `parametrizations.weight.original0/original1` | Existing CTC GGUFs keep the legacy route; partial or ambiguous pairs fail closed | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (emotion2vec CPU/Metal runtime)

The exact public 185-tensor emotion2vec+ Large artifact gains a strict native
waveform-to-nine-class forward plus CLI run and bench routes. No C symbol,
ownership rule or allocation ABI changes. The audited legacy GGUF remains
readable only through its immutable tensor manifest and generic MIT provenance.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::emotion2vec` | `Emotion2Vec`, `Emotion2VecConfig`, `Emotion2VecScores`, topology/identity constants | Added/tightened | strict open/bind/backend selection; `classify_pcm(&[f32], 16_000) -> Result<Vec<f32>>`; `classify_scores` returns official bilingual labels with probabilities | Owns decoded F32 weights. Conv1D/grouped-Conv1D/GEMM/Softmax/LayerNorm/GELU honor explicit CPU/Metal selection; unsupported backends fail with no fallback | no for exact audited public file; malformed or partial files fail closed | (TBD) |
| `vokra-cli run` / `bench` | `arch=emotion2vec` | Added (behavior only) | 16 kHz mono WAV to all nine scores in official label order; optional output is nine little-endian f32 values | Wrong sample rate, non-finite/short PCM, provenance/manifest drift and unsupported backend fail explicitly | no | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (emotion2vec strict conversion contract)

The historical pass-through converter now accepts only the immutable official
emotion2vec+ Large 185-F32-tensor checkpoint and stamps its complete topology.
No C symbol or allocation ABI changes. The exact historical public GGUF remains
identifiable by the same complete tensor manifest and generic provenance.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-convert::models::emotion2vec` | `convert_emotion2vec_file`, `Emotion2vecReport`, source/topology constants | Tightened/additive | same conversion signature; exact 185 F32 names/shapes and immutable source/checkpoint required | Canonical official input remains accepted. Partial/foreign/dtype-drifted inputs and conflicting license overrides now fail before output | no for canonical release; intentionally rejects previously loose inputs | `f2e10046` |
| `gguf:vokra.emotion2vec.*` | 17-key topology/label group | Added | sample/encoder/Conv axes, normalization flag, arrays and official bilingual labels | New converters stamp the group. The exact historical public GGUF may omit it while its immutable 185-tensor manifest and generic MIT provenance match; partial groups fail closed in the runtime wave | no for audited public file | `f2e10046` |

### 2026-08-26 — 1.0.0-rc.1-dev (YuE-upsampler CPU/Metal runtime)

The exact public 81-tensor YuE Vocos decoder gains a strict native
feature-to-waveform forward and CLI route. No C symbol, ownership rule or
allocation ABI changes. The public legacy GGUF remains readable through its
immutable manifest and generic provenance.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::yue_upsampler` | `YueUpsampler`, `YUE_UPSAMPLER_HOT_OPS`, identity/topology constants | Added | strict bind/open/backend selection; `decode(&[f32], frames) -> Result<Vec<f32>>` | Owns decoded F32 weights. Conv1d/grouped-Conv1d/LayerNorm/GELU use one CPU/Metal route; iSTFT is deterministic host DSP | no for exact audited public file; malformed/partial files fail closed | (TBD) |
| `vokra-cli run` | `arch=yue_upsampler` | Added (behavior only) | channel-major 1024-channel little-endian f32 frames to 44.1 kHz mono WAV | Wrong feature shape, non-finite input, unsupported backend and provenance/manifest mismatches fail explicitly | no | (TBD) |
| `gguf:vokra.yue_upsampler.*` | 18-key source/checkpoint/public/topology group | Added/enforced | revision/hash strings; byte counts and sample/channel/dim/layer/FFT/hop axes; padding string | New converters stamp the complete group. The exact historical public GGUF may omit it while its full manifest and generic Apache-2.0 provenance match; a partial group fails closed. | no for audited public file; yes for malformed/partial files | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (FRCRN-SE-16K CPU/Metal runtime)

The exact public 812-tensor FRCRN release gains a strict native enhancement
forward and CLI/bench route. No C symbol, ownership rule or allocation ABI
changes. The additive metadata group pins the official checkpoint/source and
full fixed-front-end/two-U-Net topology while retaining exact compatibility
with the separately audited historical public GGUF.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::frcrn` | `Frcrn`, `FRCRN_HOT_OPS`, identity/topology constants | Added | strict `from_gguf`, `from_gguf_with_backend`, `open`, backend selection, `enhance(&[f32]) -> Result<Vec<f32>>`; `DenoiseEngine` and one-stream `SeparationEngine` | Owns decoded F32 weights. Learned complex convolution/affine/FSMN reductions use one CPU/Metal GEMM/grouped-Conv1d route; fixed STFT/iSTFT and scalar complex/layout operations are deterministic host glue | no for exact audited public file; malformed/partial or foreign files now fail closed | (TBD) |
| `vokra-cli run` / `bench` | `arch=frcrn` | Added (behavior only) | fixed 16 kHz mono WAV to enhanced mono WAV | Wrong rate, short/non-finite PCM, unsupported backend and provenance/manifest mismatches fail explicitly | no | (TBD) |
| `gguf:vokra.frcrn.*` | 17-key source/checkpoint/frontend/topology group | Added/enforced | revision/hash strings; byte count, sample/window/hop/FFT/feature/depth/channel/FSMN/SE/U-Net values; window type and complex flag | New converters stamp the complete group. The exact historical public GGUF may omit it only while its immutable manifest and generic Apache-2.0 provenance match; a partial group fails closed. | no for audited public file; yes for malformed/partial files | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (NISQA v2 CPU/Metal runtime)

The exact public 94-tensor multidimensional NISQA release gains a strict
native forward and CLI/bench route. No C symbol, ownership rule or allocation
ABI changes. The rate-aware API is required because the official checkpoint
keeps the WAV's native sample rate.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::nisqa` | `Nisqa`, `NisqaWeights`, `NisqaScore`, `NISQA_HOT_OPS` | Changed/Added | strict bind/backend selection; `score_at_sample_rate(&[f32], u32) -> Result<NisqaScore>`; five fields in `mos,noi,dis,col,loud` order | Owns decoded F32 weights. Learned Conv/Linear/attention uses one CPU/Metal GEMM/softmax/LayerNorm route; front-end, BatchNorm and adaptive pooling are deterministic host glue | no for exact public bundle; generic/count-only files now fail closed | (TBD) |
| `vokra-cli run` / `bench` | `arch=nisqa_v2_weight` | Added (behavior only) | native-rate mono WAV to five quality dimensions | Missing rate, non-finite PCM, oversized segment count, unsupported backend and provenance mismatches fail explicitly | no | (TBD) |
| `gguf:vokra.nisqa.*` | source/public identity plus exact front-end/topology group | Added/enforced | revision/hash strings; native-rate sentinel, FFT/hop/window/mel/segment/pool/head values | New converters stamp the complete additive group. The exact historical public GGUF may omit it only because its full manifest and generic provenance are independently pinned; a partial group fails closed. | no for audited public file; yes for malformed/partial files | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (DNSMOS P.808/P.835 CPU/Metal runtime)

The exact public 38-tensor DNSMOS bundle gains strict native P.808 and P.835
forwards plus CLI/bench routing. No ONNX dependency enters the runtime and no
C symbol, ownership rule, or allocation ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::dnsmos_p808_p835` | `Dnsmos`, `DnsmosWeights`, `DNSMOS_HOT_OPS` | Changed/Added | strict `from_gguf` / `from_path`, backend selection, `score_p808`, `score_p835`, `score_all`, `MosScorerEngine` | Owns decoded F32 weights; all learned Conv2d/dense/STFT projections use one CPU/Metal GEMM backend with no fallback | no for exact public bundle; malformed/count-only files now fail closed | (TBD) |
| `vokra-cli run` / `bench` | `arch=dnsmos` | Added (behavior only) | 16 kHz mono WAV to P.808/SIG/BAK/OVRL scores | Wrong rate, non-finite PCM, incomplete bundle, unsupported backend and provenance mismatches fail explicitly | no | (TBD) |
| `gguf:vokra.dnsmos.*` | source/public identity and exact frontend/topology group | Added/enforced | revision/hash strings; `u32` input length, frame/window/hop/bin/mel values; existing bundle/checkpoint/sample-rate keys retained | New converters stamp the complete additive group. The exact historical public GGUF may omit only the new group because its complete manifest and generic provenance are independently pinned; a partial new group fails closed. | no for audited public file; yes for malformed/partial files | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (Facebook Denoiser DNS48 CPU/Metal runtime)

The exact historical public DNS48 GGUF gains a strict native utterance
forward and CLI/bench route. No C symbol, ownership rule or allocation ABI
changes. New conversion adds a self-describing topology/source metadata group;
the historical public file remains compatible through its complete immutable
manifest and generic provenance tuple.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::facebook_denoiser` | `FbDenoiser`, `FbDenoiserWeights`, `FACEBOOK_DENOISER_HOT_OPS` | Changed/Added | strict `from_gguf` / `open`, `with_backend`, `denoise(&[f32]) -> Result<Vec<f32>>`, utterance-buffered `DenoiseEngine` and one-stream `SeparationEngine` | Owns decoded F32 weights and returned PCM; Conv1d/LSTM/ConvTranspose projections use one CPU/Metal backend with no fallback | no for exact public DNS48; malformed/count-only files now fail closed | (TBD) |
| `vokra-cli run` / `bench` | `arch=facebook_denoiser` | Added (behavior only) | 16 kHz mono WAV to same-length enhanced WAV | Wrong rate, non-finite PCM, unsupported backend and non-commercial policy violations fail explicitly | no | (TBD) |
| `gguf:vokra.facebook_denoiser.*` | source/public identity and fixed DNS48 topology group | Added/enforced | strings for revisions/hashes/URLs; `u32` checkpoint bytes/topology/correction values; `f32` floor; `bool` causal/GLU/normalize | New converters stamp the complete group. The exact historical 48-tensor public file may omit it; a partial new group fails closed. | no for audited public file; yes for malformed/partial files | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (AudioSeal CPU/Metal runtime)

The previously published four-checkpoint AudioSeal GGUF gains a strict native
Rust binder and explicit CLI embed/detect route. Automatic application through
`WatermarkConfig` remains Deferred. No C symbol, ownership rule or allocation
ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::audioseal` | `Audioseal`, `AudiosealVariant`, `AudiosealDetection`, `AUDIOSEAL_HOT_OPS` | Added | strict `from_gguf` / `from_file`, `with_backend`, `watermark_pcm`, `embed_pcm`, `detect_pcm`, `detect_pcm_with_thresholds` | Owns decoded FP32 weights and returned vectors; complete-buffer base/streaming variants dispatch Conv1D, transposed-convolution projections, LSTM projections and softmax through one CPU/Metal backend with no fallback | no | (TBD) |
| `vokra-cli run` / `bench` | `arch=audioseal_real_weight`; `--watermark-{mode,variant,message,alpha}` | Added (behavior only) | 16 kHz mono WAV to detector report or watermarked WAV; message is exactly 16 ASCII bits | Non-AudioSeal flag use, wrong rate, non-finite gain, missing message and unsupported backend fail explicitly; streaming means causal weights over a complete buffer, not a stateful chunk API | no | (TBD) |
| `gguf:vokra.audioseal.*` | fixed topology/revision metadata group | Added/enforced | 24 fixed keys plus exactly 310 F32 tensors across four variants | Canonical conversions are self-describing. The exact historical public header is accepted only through full manifest and identity/provenance validation; partial groups fail closed. | no for the audited public file; yes for malformed/partial files | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (RMVPE exact E2E0 behavior correction)

No Rust or C signature is added. The existing default `RMVPE` loader/forward
changes from a name-discovered reduced graph to the fixed 623-tensor E2E0
contract. `CnnChainPolicy::Optional` preserves the old structural-fixture path
only when explicitly requested.

| Surface | Symbol / key | Change | Compatibility | Breaking? | Commit |
|---|---|---|---|---:|---|
| `vokra-models::f0::rmvpe` | `RMVPE::{from_gguf,open,extract,extract_real,forward_from_hidden}` | Behavior corrected; signatures unchanged | Default loading now requires the exact five-stage skip U-Net + BiGRU manifest, fixed provenance, and canonical/historical metadata pair. Partial/fork-named and mislicensed public files fail before forward; unsupported Metal ops never fall back. | yes for previously accepted incomplete artifacts; those outputs were not upstream RMVPE | (TBD) |
| `gguf:vokra.rmvpe.*` | `n_fft`, `base_hz`, `upstream_revision` | Two meanings corrected; one key added | New converter values are 1024 and 31.7 and stamp source commit `0aabafba18289ca938a73af0b0297686abf4922d`. The exact historical public 2048/32.703197 pair is normalized without modifying tensor payload; provenance-corrected legacy copies may omit the new key, while canonical new artifacts require it. | no for the audited historical pair | (TBD) |

### 2026-08-26 — 1.0.0-rc.1-dev (Audiobox Aesthetics CPU/Metal runtime)

The historical public Audiobox GGUF remains loadable only when its complete
324-tensor F32 manifest and canonical identity/provenance match. New
conversions add a self-describing topology/target-transform group. The native
Rust scorer and CLI route are additive; no C symbol, ownership rule or buffer
ABI changes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::audiobox_aesthetics` | `AudioboxAesthetics`, `AudioboxAestheticsConfig`, `AudioboxScores`, `AUDIOBOX_AESTHETICS_HOT_OPS` | Added | strict `from_gguf` / `from_file`, `with_backend`, `score_pcm(&[f32], u32) -> Result<AudioboxScores>` | Owns decoded FP32 weights and returned scores; Conv1D, grouped Conv1D, GEMM, softmax, LayerNorm and GELU use one selected CPU/Metal backend with no fallback | no | (TBD) |
| `vokra-cli run` / `bench` | `arch=audiobox-aesthetics` | Added (behavior only) | 16 kHz mono WAV to four CE/CU/PC/PQ values; optional raw little-endian f32 output in that order | Wrong sample rates and unsupported backends fail explicitly; no learned `BALANCED` value is synthesized | no | (TBD) |
| `gguf:vokra.audiobox_aesthetics.*` | canonical topology and target-transform metadata group | Added | revisions (`string`); 15 topology values (`u32`); `normalize_embed`/`use_weighted_layer_sum` (`bool`); `layer_norm_eps` (`f32`); axes (`string[]`); target means/stds (`f32[]`) | New conversions are self-describing. The exact historical public file is accepted only through full identity/provenance/manifest validation; partial metadata fails closed. | no for audited public file; yes for malformed/partial files | (TBD) |

### 2026-08-25 — 1.0.0-rc.1-dev (AST and DeepFilterNet backend-selectable Rust seams)

The existing scalar AST/ViT and DeepFilterNet3 graphs gain additive Rust
backend seams so model crates can dispatch every learned reduction through a
selected `Compute` backend while retaining the established CPU routines as
their arithmetic oracles. No C symbol, ownership rule, GGUF key or serialized
shape changes.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-ops::denoise` | `DenoiseBackendOps` | Added | public trait for grouped Conv2D, ConvTranspose2D, pointwise convolution, grouped/dense linear and GRU projections | Defines the complete learned-op boundary used by the DeepFilterNet3 Metal adapter; DSP, nonlinearities and layout glue remain host-side and backend errors propagate without scalar fallback. | no | (TBD) |
| `vokra-ops::denoise::DenoiseModel` | `enhance_with_backend_ops`, `enhance_with_backend_ops_and_taps` | Added | `(&self, &[f32], &dyn DenoiseBackendOps) -> Result<Vec<f32>>` and diagnostic-tap counterpart | Adds device-selectable forward and parity entry points without changing the bit-identical `enhance` CPU oracle. | no | (TBD) |
| `vokra-ops::kaldi_fbank::KaldiFbankOpts` | `ast_audioset` | Added | `pub fn ast_audioset() -> Self` | Pins the released AST AudioSet frontend geometry and normalization in one public constructor. | no | (TBD) |
| `vokra-ops::vit` | `ViTBackendOps` | Added | public trait for linear, matmul, softmax, LayerNorm and GELU | Defines AST's complete learned-op boundary while leaving positional/layout/residual glue on the host. | no | (TBD) |
| `vokra-ops::vit::ViTEncoder` | `forward_with_backend`, `encode_tokens_with_backend` | Added | backend-selectable waveform-patch and token-stack forwards accepting `&dyn ViTBackendOps` | Makes the released AST graph reachable through Metal without changing the existing scalar entry points or allowing silent fallback. | no | (TBD) |

### 2026-08-25 — 1.0.0-rc.1-dev (X-Codec2 token-to-waveform CPU/Metal runtime)

The public X-Codec2 GGUF now strict-binds its released pass-through manifest
and decodes discrete 50 Hz FSQ frames through Rust and CLI. No C function,
type, ownership rule or allocation contract changed. The non-commercial weight
gate remains mandatory, and waveform encoding remains explicitly unsupported.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::xcodec2` | `XCodec2`, `ARCH`, `SAMPLE_RATE`, `HOP_LENGTH`, `CODEBOOK_SIZE`, `XCODEC2_DECODE_HOT_OPS` | Added | strict `from_gguf` / `open`, `with_backend`, geometry/accessor methods and `decode_codes(&[u32]) -> Result<Vec<f32>>` | Owns decoded FP32 weights and returned PCM; exact released manifest dispatches FSQ, Conv1D, GroupNorm, RMSNorm, GEMM, Softmax, SiLU and LayerNorm through one selected backend | no | `96c5fb62` |
| `vokra-cli run` | `--codec-mode decode` for `arch=xcodec2` | Changed (behavior only) | one little-endian `u32` code per 50 Hz frame to 16 kHz mono WAV; `encode` is an explicit error | Adds a standalone decoder route without claiming an encoder or defining a serialized container ABI; the existing research-license policy is checked before input/model execution | no | `96c5fb62` |
| `gguf:xcodec2` | released manifest, provenance and licence contract | Enforced | exactly 1,153 F32 tensors; manifest SHA-256 `ee543e96b5150376101396197bb0add53daf913eb991deb42aad7be74eed33f5`; `HKUSTAudio/xcodec2`, `cc-by-nc-4.0`, `non-commercial` | The audited public file remains compatible. Missing, extra, wrong-shaped, differently identified or differently licensed artifacts are rejected before inference. | no for the exact released file; yes for malformed/partial files | `96c5fb62` |

### 2026-08-25 — 1.0.0-rc.1-dev (NeuCodec token-to-waveform CPU/Metal runtime)

The public NeuCodec base and distill GGUFs now strict-bind their two released
manifests and decode discrete 50 Hz FSQ frames through Rust and CLI. The
runtime accepts the base artifact's legacy normalized decoder namespace and
the distill artifact's pass-through namespace as separate fixed contracts.
No C function, type, ownership rule or allocation contract changed, and
waveform encoding remains explicitly unsupported.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::neucodec` | `NeuCodec`, `NeuCodecVariant`, `ARCH`, `SAMPLE_RATE`, `HOP_LENGTH`, `CODEBOOK_SIZE`, `QUANTIZED_DIM`, `HIDDEN_DIM`, `NEUCODEC_DECODE_HOT_OPS` | Added | strict `from_gguf` / `open`, `with_backend`, geometry/accessor methods and `decode_codes(&[u32]) -> Result<Vec<f32>>` | Owns decoded FP32 weights and returned PCM; exact base/distill manifests dispatch FSQ, Conv1D, GroupNorm, RMSNorm, GEMM, Softmax, SiLU and LayerNorm through one selected backend | no | `ddc7b4f2` |
| `vokra-cli run` | `--codec-mode decode` for `arch=neucodec` | Changed (behavior only) | one little-endian `u32` code per 50 Hz frame to 24 kHz mono WAV; `encode` is an explicit error | Adds a standalone decoder route without claiming an encoder or defining a serialized container ABI | no | `63a2a87b` |
| `gguf:neucodec` | released manifest and variant/provenance contract | Enforced | base: legacy 811-tensor manifest without `vokra.neucodec.variant`; distill: 294-tensor pass-through manifest with `variant=distill` | Both audited public files remain compatible. Variant recognition is still pinned by complete manifest, model identity and upstream provenance; missing, extra, wrong-shaped or unknown layouts are rejected. | no for exact released files; yes for malformed/partial files | `ddc7b4f2` |

### 2026-08-25 — 1.0.0-rc.1-dev (WavTokenizer token-to-waveform CPU/Metal runtime)

The two byte-identical public WavTokenizer GGUF repositories now strict-bind
the released decoder and decode discrete code frames through Rust and CLI.
The shared Vocos tail gains one additive Rust entry point so WavTokenizer can
insert its official positional network before reusing the ConvNeXt/iSTFT
implementation. No C function, type, ownership rule or allocation contract
changed, and waveform encoding remains explicitly unsupported.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-ops::vocos` | `vocos_decode_from_embedded_with_ops` | Added | `pub fn vocos_decode_from_embedded_with_ops<O: VocosBackendOps>(...) -> Result<Vec<f32>>` | Borrows the embedded feature matrix and weights, returns owned PCM, and reuses the existing backend-parametric ConvNeXt/iSTFT tail without a fallback | no | (TBD) |
| `vokra-models::wavtokenizer` | `WavTokenizer`, `ARCH`, `WAVTOKENIZER_HOT_OPS` | Added | strict `from_gguf` / `from_path`, `with_backend`, geometry/accessor methods, `decode_codes` and `decode_codes_with_condition` | Owns folded FP32 weights and returned PCM; exact released manifests dispatch VQ, Conv1D, grouped Conv1D, normalization, attention/GEMM and SiLU through one selected backend | no | (TBD) |
| `vokra-cli run` | `--codec-mode decode` for `arch=wavtokenizer` | Changed (behavior only) | one little-endian `u32` code per 75 Hz frame to 24 kHz mono WAV; `--bandwidth` selects one of four released conditioning rows; `encode` is an explicit error | Adds a standalone decoder route without claiming an encoder or defining a serialized container ABI | no | (TBD) |
| `gguf:vokra.wavtokenizer.*` | released topology and provenance contract | Enforced | exact F32 1,091-tensor public manifest, model identity, sample rate, hop, codebook and conditioning geometry | The two audited public files remain compatible; missing, extra, wrong-shaped or unknown layouts are rejected | no for exact released files; yes for malformed/partial files | (TBD) |

### 2026-08-25 — 1.0.0-rc.1-dev (MeloTTS released-config metadata correction)

The MeloTTS converter's English and Chinese language axes are corrected to the
released `myshell-ai/MeloTTS-*` configs and tensor tables. The key names and
types are unchanged, but newly converted English and Chinese artifacts carry
different values. The five public GGUFs predate this correction; runtime
compatibility for those exact audited manifests is tracked separately and must
not turn arbitrary metadata/tensor disagreement into a permissive fallback.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `gguf:vokra.melotts.*` | `vokra.melotts.{n_symbols,num_tones,num_languages}` | Fixed values | English `219 / 16 / 10` (was `178 / 0 / 1`); Chinese `112 / 11 / 4` (was `112 / 11 / 1`); Korean, Spanish and Japanese remain `219 / 16 / 10` | Values are pinned to the official per-language `config.json` and released embedding-table shapes. New conversions are self-consistent; consumers must reject unrecognized mismatches. | yes for code that depended on the incorrect values; no for key/type ABI | (TBD) |

### 2026-08-25 — 1.0.0-rc.1-dev (DAC token-to-waveform CPU/Metal runtime)

The three public Descript DAC GGUF families now bind exact released manifests
and decode discrete code frames to PCM through Rust and CLI. This is an
additive Rust model surface and a CLI behavior extension. No C function, type,
ownership rule or allocation contract changed; DAC is not added to the C ABI.
The encoder remains explicitly unsupported rather than being represented as a
completed bidirectional codec.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::dac` | `Dac`, `DacVariant`, `DacDecoder`, `ARCH`, `DAC_HOT_OPS` | Added | strict `from_gguf` / `from_path`, `with_backend`, backend/variant/geometry accessors, `decode_codes`, `decode_features` and `output_samples` | Owns folded FP32 weights and returned PCM; exact 16/24/44.1 kHz manifests dispatch RVQ, Conv1D and Snake through one selected backend with no fallback | no | (TBD) |
| `vokra-cli run` | `--codec-mode decode` for `arch=dac` | Changed (behavior only) | raw time-major `[frames,n_codebooks]` little-endian `u32` input to mono WAV; `encode` is an explicit error | Adds a standalone decoder route without claiming an encoder or defining a serialized container ABI | no | (TBD) |
| `gguf:vokra.dac.*` | released topology and provenance contract | Enforced | exact F32 name/shape manifests: 358 tensors / 12 codebooks at 16 kHz, 558 / 32 at 24 kHz, and 328 / 9 at 44.1 kHz | Existing canonical metadata is validated fail-closed; missing, extra, wrong-shaped or incompatible provenance is rejected | no for exact released files; yes for malformed/partial files | (TBD) |

### 2026-08-25 — 1.0.0-rc.1-dev (TitaNet-L CPU/Metal speaker runtime)

The two byte-identical public TitaNet-Large GGUFs now bind the exact
108-tensor NeMo inference topology and emit 192-dimensional embeddings through
Rust, CLI and the existing model-generic C speaker API. No C function, type,
ownership rule or embedding-buffer ABI changed. The existing converter surface
is hardened: canonical NVIDIA conversion now rejects partial/extra manifests
and an incompatible licence override rather than publishing ambiguous bytes.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::titanet` | `TitaNet`, `ARCH`, `NAME`, `CATEGORY`, `UPSTREAM_HF`, `UPSTREAM_REVISION`, `SOURCE_REVISION`, `EMBED_DIM` | Added | strict `from_gguf` / `from_path`, `with_backend`, backend/licence accessors, frontend/features/embed methods and `SpeakerEngine` implementation | Complete NeMo 1.10 frontend, depthwise-separable Conv1D/SE encoder, attentive statistics pool and 192-d projection with explicit CPU/Metal dispatch and no fallback | no | (TBD) |
| `gguf:vokra.titanet.*` | canonical TitaNet topology/frontend contract | Persisted and enforced | sample rate, mel/STFT axes, channel widths, BN/stat epsilons, source revision, frontend, block schedule, pooling and artifact layout | New conversions write the full contract. The exact historical public 108-tensor files remain compatible through pinned identity/provenance/manifest validation; a partial contract is rejected. | no for audited public files; yes for malformed/partial files | (TBD) |
| `include/vokra.h` / `vokra-capi::speaker` | `vokra_speaker_embed` model coverage | Changed (behavior only) | existing signature unchanged; TitaNet sessions report/write 192 floats | Extends the dynamic two-call speaker API without a new symbol or ownership change. | no | (TBD) |
| `vokra-core::m5_residual_ops` | `TITANET_SPEAKER_ENCODE_OP` description | Corrected (semantic only) | constant value remains `"titanet_speaker_encode"` | The reservation now names only the graph-side `OpKind` and dedicated op-level C export; it no longer falsely claims the model runtime is absent. | no | (TBD) |

### 2026-08-24 — 1.0.0-rc.1-dev (NPU whole-submodel delegate surface — Rust only, C ABI NO-GO)

M5-01 adds an explicit Rust execution boundary for delegates that own a whole
model subgraph rather than individual Vokra operators. The first implementation
executes Whisper's complete audio encoder through a hash-bound CoreML sidecar.
The real Apple M1 bakeoff recorded 99.63% estimated-cost ANE placement, but
failed both CPU-oracle parity and the 2x speed criterion; therefore no delegate
selector, enum value, or other symbol is added to `include/vokra.h`. QNN remains
SDK/Hexagon-gated and likewise has no C value. This NO-GO is recoverable through
an additive minor-version API after both acceptance gates are met.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-core::backend` | `DelegateSubmodel` | Added | `#[non_exhaustive] pub enum DelegateSubmodel { WhisperEncoder }` | Names the indivisible model-owned graph boundary without pretending the CoreML/QNN delegate has per-op coverage. | no | (TBD) |
| `vokra-core::backend` | `DelegateBackend` | Added | `pub trait DelegateBackend { fn delegate_name(&self) -> &str; fn supports_submodel(&self, DelegateSubmodel) -> bool; fn execute_submodel(&self, DelegateSubmodel, &[&Tensor]) -> Result<Vec<Tensor>>; }` | Carries real tensor data through a declared complete submodel and requires unsupported boundaries to fail instead of silently running CPU. | no | (TBD) |
| `vokra-core::engines` | `TtsEngine::backend` | Changed | `fn backend(&self) -> BackendKind` (new required method) | Makes the selected TTS execution backend observable, matching `AsrEngine` / `SpeakerEngine`; it is deliberately not defaulted so an out-of-tree engine cannot silently report CPU. Source-breaking for out-of-tree implementations during the prerelease window. | no\* | (TBD) |
| `vokra-core::tasks` | `Tts::backend` | Added | `pub fn backend(&self) -> BackendKind` | Lets CLI/C ABI wiring tests prove that a caller-selected backend reached an injected TTS engine even when CPU and GPU audio are numerically equal. | no | (TBD) |
| `vokra-core::engines` | `DenoiseStreamHandle::finalize` | Added | `fn finalize(&mut self) -> Result<Vec<f32>>` (empty default) | Lets STFT denoisers flush right-padded analysis frames and overlap-add tails; NSNet2 overrides it so streaming and one-shot output lengths agree. Frame-complete denoisers remain source-compatible through the default. | no | (TBD) |
| `vokra-core::ir::graph` | `Window::SqrtHann` | Added | new `#[non_exhaustive]` enum variant | Represents Microsoft's NSNet2 analysis/synthesis window without precomputing or mislabelling it as Hann. | no | (TBD) |
| `vokra-core::ir::graph` | `IstftAttrs::normalize_window` | Added | `pub bool`, constructor default `true` | Makes raw overlap-add explicit for upstream pipelines such as NSNet2 while preserving the historical normalized iSTFT default for every existing caller. Struct-literal callers must add the field during the prerelease window. | no\* | (TBD) |
| `vokra-models::nsnet2` | `Nsnet2V1::{with_backend,backend}` | Added | `pub fn with_backend(self, BackendKind) -> Self`; `pub fn backend(&self) -> BackendKind` | Makes Mac Metal selection explicit for the complete NSNet2 learned projection/mask path while preserving CPU as the constructor default. Unsupported backends fail through the whole-model coverage gate. | no | (TBD) |
| `vokra-models::nsnet2` / `gguf:vokra.nsnet2.*` | canonical Microsoft 20 ms topology | Changed / historical artifact compatibility | `n_bins=161`, `n_fft=320`, `win_length=320`; exact 14 F32 tensors renamed from the pinned ONNX manifest; separate ONNX GRU input/recurrent biases; exact `vokra/nsnet2@983e1cc` numeric-initializer header accepted as a closed legacy variant | Replaces the incorrect 257-bin/512-FFT runtime assumptions with the official graph and `linear_before_reset=1` arithmetic. The canonical converter remains strict and self-describing. The immutable 2026-08-03 public GGUF is repaired only after its ten metadata values plus all fourteen tensor names/order/dtypes/shapes/offsets match; partial metadata, mixed schemas and drift fail closed. Legacy acceptance establishes topology identity only: the live file's incorrect MIT/permissive provenance remains a publication blocker until an authorized CC-BY-4.0 replacement. | yes (artifact) | (TBD) |
| `vokra-backend-metal::MetalContext` | `grouped_conv1d_f32` | Added | checked grouped Conv1D with `[out_ch, in_ch / groups, kernel]` weights | Adds a real Metal depthwise/grouped convolution path by extending the existing direct Conv1D shader with group-local input indexing. Shape/group errors remain explicit. | no | (TBD) |
| `vokra-models::compute` | `HotOp::GroupedConv1d`, `Compute::grouped_conv1d_f32` | Added | whole-model coverage variant plus typed CPU/Metal dispatcher | Lets Vocos declare depthwise ConvNeXt coverage honestly. CUDA/WebGPU/delegates remain uncovered and fail before inference instead of falling back to CPU. | no | (TBD) |
| `vokra-backend-{cpu,metal}` / `vokra-models::compute` | `group_norm_f32`; `HotOp::GroupNorm`, `Compute::group_norm_f32` | Added | one-group channel-major GroupNorm with a fixed 256-partial pairwise reduction on CPU and Metal | Gives SepFormer's 130k–384k-value mask normalization a stable dedicated primitive. CUDA/WebGPU remain explicitly unsupported through the model dispatcher; no CPU fallback is introduced. | no | (TBD) |
| `vokra-backend-metal::MetalContext` | `gemm_f32` numerical contract | Changed | bias-first accumulation and disabled FP contraction in the existing signature | Matches the documented CPU `bias + sum(a*b)` order and reduces amplified SepFormer DNS4 drift without weakening the direct GEMM parity boundary. | no | (TBD) |
| `vokra-models::sepformer` | `SepFormer::{separate,encode_features}` DNS4 CPU numerical route | Changed | DNS4 alone selects the portable scalar CPU reduction order; the other six variants keep normal ISA dispatch and Metal remains GPU-only | Keeps the ill-conditioned DNS4 waveform inside the independent official-FP64 gate without imposing the scalar performance cost on the rest of the family. | no | (TBD) |
| `vokra-ops::vocos` | `VocosBackendOps`, `vocos_decode_with_ops` | Added | backend-supplied dense/depthwise Conv1D, LayerNorm, pointwise and GELU seam | Preserves the original scalar CPU decoder while allowing all learned Vocos operations to execute through one selected backend; magnitude/phase and iSTFT remain DSP glue. | no | (TBD) |
| `vokra-models::vocos` | `Vocos::{with_backend,backend}`, `VOCOS_HOT_OPS` | Added | explicit `BackendKind` selector and declared learned-op registry | Makes both Vocos variants reachable from the CLI on Mac Metal and keeps CPU as the default. The historical public Encodec file remains fail-closed because its metadata/provenance contradict its tensor topology. | no | (TBD) |
| `vokra-ops::fsmn_vad` | `FsmnBackendOps`, `fsmn_vad_forward_with_ops` | Added | backend-supplied row-wise Linear and causal depthwise-memory seam | Preserves the established scalar CPU oracle while allowing all released FSMN learned operations to execute on one selected backend; ReLU, residual addition, softmax and stream history remain host control/glue. | no | (TBD) |
| `vokra-models::fsmn_vad` | `FsmnVadV1::{with_backend,backend}` | Added | explicit `BackendKind` selector for GEMV plus grouped Conv1D | Makes the strict FSMN-VAD runtime reachable from the CLI on Mac Metal without silently accepting the stale public GGUF, which predates the required provenance, CMVN and topology keys. | no | (TBD) |
| `vokra-ops::firered_vad` | `FireredVadBackendOps`, `firered_vad_dfsmn_forward_with_ops` | Added | backend-supplied dense and causal-memory projection seam | Preserves the independent scalar CPU forward while routing every learned FireRedVAD DFSMN projection through one selected backend. | no | (TBD) |
| `vokra-models::firered_vad` | `FireredVad::{with_backend,backend}` | Added | explicit `BackendKind` selector and complete GEMV hot-op registry | Makes the strict FireRedVAD stream reachable on Metal while keeping frontend, activation, cache and softmax work as host glue. | no | (TBD) |
| `vokra-ops::ten_vad` | `TenVadBackendOps`, `network_forward_with_ops` | Added | backend-supplied GRU and output projection seam | Preserves the scalar TEN-VAD oracle while dispatching all six GRU matrix products and the output affine through the selected backend. | no | (TBD) |
| `vokra-models::ten_vad` | `TenVad::{with_backend,backend}` | Added | explicit `BackendKind` selector and complete GEMV hot-op registry | Makes the strict TEN-VAD stream reachable on Metal without moving frontend/state/softmax control flow off host. | no | (TBD) |
| `vokra-models::silero_vad` | `SileroVadV5::{with_backend,backend}` | Added | explicit `BackendKind` selector for the shared Conv1D/LSTM/output backend adapter | Routes every learned Silero operation through the selected backend while retaining the `no_std` implementation as the CPU arithmetic oracle. | no | (TBD) |
| `vokra-models::rnnoise` | `RnnoiseV02::{with_backend,backend}`, `RNNOISE_HOT_OPS` | Added | explicit `BackendKind` selector and GEMV hot-op registry | Expands the release's compressed matrices once for backend GEMV while keeping activation quantization, rational nonlinearities and recurrent state updates identical to the compressed CPU path. | no | (TBD) |
| `vokra-models::aec::nkf_aec` | `NkfAec::{with_backend,backend}`, `NKF_AEC_HOT_OPS` | Added | explicit `BackendKind` selector and GEMV hot-op registry | Routes every learned complex-dense and GRU projection through Metal; STFT, gates, complex Kalman arithmetic and overlap-add remain host DSP/control flow. | no | (TBD) |
| `vokra-ops::hifigan` / `vokra-ops::bigvgan_generator` | `HifiGanBackendOps`, `hifigan_generator_with_backend_ops`, `BigVganBackendOps`, `BigVGanGenerator::forward_with_backend_ops` | Added | backend-supplied Conv1D plus BigVGAN Snake/SnakeBeta seams | Preserves the scalar vocoder oracles while allowing every learned HiFi-GAN/BigVGAN operation to use one selected backend; resampling and waveform assembly remain host DSP. | no | (TBD) |
| `vokra-models::{hifigan,bigvgan}` | `HiFiGan::{with_backend,backend}`, `HIFIGAN_HOT_OPS`; `BigVGan::{with_backend,backend}`, `BIGVGAN_HOT_OPS` | Added | explicit backend selectors and complete learned-op registries | Makes both public HiFi-GAN and all four BigVGAN repositories reachable on Metal without a per-op CPU fallback. | no | (TBD) |
| `vokra-models::f0::fcpe` | `FCPE::{with_backend,backend}`, `FCPE_HOT_OPS` | Added | explicit backend selector and Conv2D/GroupedConv1D/GEMM/GEMV/LayerNorm/Softmax registry | Routes the complete learned FCPE forward through one backend while retaining frontend and pitch decoding as host DSP. | no | (TBD) |
| `vokra-models::smart_turn` | `SmartTurn::{with_backend,backend}`, `SMART_TURN_HOT_OPS` | Added | explicit backend selector for the complete Wav2Vec2 endpoint forward | Covers all learned convolution, attention, normalization, pooling and classifier operations; the stale 223-tensor public GGUF remains fail-closed. | no | (TBD) |
| `vokra-models::pyannote` | `PyanNet::{from_gguf,from_gguf_with_backend,segment,segment_powerset,with_backend,backend,model_name,weight_license,legacy_metadata_repaired}`, `PYANNOTE_HOT_OPS` | Changed | strict exact-release binder plus default-on CPU/Metal segmentation | Requires the exact 54-F32-tensor release identity, narrowly repairs the immutable historical two-layer metadata stamp to the tensor-proven four-layer topology, and routes every learned SincNet/BiLSTM/projection/classifier reduction through one preflighted backend with no CPU fallback. Loose illustrative GGUFs no longer load through the public model handle. | no (behavior tightened) | (TBD) |
| `vokra-models::f0::rmvpe` | `RMVPE::{with_backend,backend}`, `RMVPE_HOT_OPS` | Added | explicit backend selector for lowered Conv2D/ConvTranspose2D, BiGRU and head projections | Adds backend dispatch without claiming model completion: the existing CPU forward still omits upstream decoder skip-concat and remains routed-partial. | no | (TBD) |
| `vokra-models::parakeet` | `ParakeetAsr::{with_backend,backend}`, `PARAKEET_HOT_OPS` | Added | explicit backend selector and complete FastConformer/TDT learned-op registry | Preserves backend selection through subsampling, encoder attention/convolution and recurrent duration-aware joint decoding. | no | (TBD) |
| `vokra-models::parakeet_ctc` | `ParakeetCtcAsr::{with_backend,backend}` | Added | explicit backend selector reusing `PARAKEET_HOT_OPS` | Runs the shared FastConformer and complete CTC vocabulary head on the selected backend. | no | (TBD) |

**C ABI decision for this run:** CoreML = **NO-GO**; QNN = **INSUFFICIENT
DATA** because no approved SDK/runtime, executable graph, or Hexagon target is
available; combined v1.0 delegate-selector decision = **NO-GO**.
`include/vokra.h` and the C symbol snapshot are intentionally unchanged.

### 2026-08-24 — 1.0.0-rc.1-dev (SpeechBrain X-vector CPU/Metal speaker runtime)

The two existing public X-vector artifact layouts now bind and execute as a
native 512-dimensional speaker engine. This extends the behavior of the
existing C speaker entry without adding or changing a C symbol. New
conversions persist the complete SpeechBrain frontend/topology contract;
legacy public artifacts remain accepted only through their exact audited
32- or 46-tensor manifests and pinned identity/provenance.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `include/vokra.h` / `vokra-capi::speaker` | `vokra_speaker_embed` model coverage | Changed (behavior only) | existing signature unchanged; X-vector sessions return 512 floats instead of the CAM++-only 192-float model set | Makes the already model-sized two-call C contract work for both public X-vector artifacts while preserving explicit rate/backend errors. | no | (TBD) |
| `vokra-ops::speechbrain_fbank` | `SpeechbrainFbankAttrs`, `speechbrain_fbank` | Added | public exact frontend attributes and `fn(&[f32], &SpeechbrainFbankAttrs) -> Result<(Vec<f32>, usize)>` | Represents SpeechBrain's centered Hamming STFT, nonstandard triangular filters, dB floor and sentence CMN without misusing Kaldi/librosa semantics. | no | (TBD) |
| `vokra-models::xvector` | `XVector`, `XVectorArtifactLayout` and identity constants | Added | strict `from_gguf`/`from_path`, `with_backend`, frontend/network/embed accessors, `SpeakerEngine` impl; enum variants `EmbeddingOnlyBare`, `CombinedPrefixed` | Exposes exact 32- and 46-tensor public layouts, deterministic native inference and observable CPU/Metal selection with no fallback. | no | (TBD) |
| `gguf:vokra.xvector.*` | canonical X-vector metadata group | Added | `sample_rate`, `n_mels`, `n_fft`, `win_length`, `hop_length`, `embed_dim`, `tdnn_blocks` (`u32`); `bn_eps`, `stats_std_eps` (`f32`); `frontend`, `padding`, `artifact_layout` (`string`) | Makes future conversions self-describing and pins the official 16-kHz SpeechBrain fbank/TDNN contract. Historical public files are accepted only by exact manifest compatibility, never guessed from partial tensors. | no | (TBD) |

### 2026-08-24 — 1.0.0-rc.1-dev (SpeechBrain ECAPA-TDNN CPU/Metal speaker runtime)

The two canonical public SpeechBrain ECAPA artifacts now bind the exact
200-tensor speaker topology and emit 192-dimensional embeddings through Rust,
CLI and the existing model-generic C speaker entry. The separately implemented
202-tensor voice-gender artifact is rejected despite its historical
`ecapa_tdnn` tag; no topology is inferred from a partial name overlap.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-ops::speechbrain_fbank` | `SpeechbrainFbankAttrs::ecapa_voxceleb` | Added | `pub fn ecapa_voxceleb() -> SpeechbrainFbankAttrs` | Pins the official 80-bin SpeechBrain ECAPA frontend while sharing only the axes that are genuinely identical to X-vector. | no | (TBD) |
| `vokra-models::ecapa_tdnn` | `EcapaTdnn` and identity constants | Added | strict `from_gguf`/`from_path`, `with_backend`, frontend/network/embed accessors and `SpeakerEngine` impl | Implements the complete reflect-padded TDNN, three SE-Res2Net blocks, MFA, attentive statistics pool and 192-d projection with explicit CPU/Metal dispatch. | no | (TBD) |
| `gguf:vokra.ecapa.*` | canonical ECAPA metadata group | Added | ten `u32` topology/frontend axes, `bn_eps`/`stats_eps` (`f32`), and `frontend`/`padding`/`artifact_layout` (`string`) | Makes new conversions self-describing; historical public files remain accepted only by exact 200-tensor manifest and pinned identity/provenance. | no | (TBD) |
| `include/vokra.h` / `vokra-capi::speaker` | `vokra_speaker_embed` model coverage | Changed (behavior only) | existing signature unchanged; ECAPA sessions report/write 192 floats | Extends the already dynamic two-call speaker API without adding a C symbol or changing ownership. | no | (TBD) |

### 2026-08-24 — 1.0.0-rc.1-dev (WeSpeaker ResNet34-LM CPU/Metal speaker runtime)

The public pyannote WeSpeaker checkpoint now binds an exact 182-tensor
embedding layout and runs through the existing dynamic speaker API. The
219-tensor official combined layout is also structurally supported, but the
current `vokra/wespeaker` public file fails before tensor binding because its
Apache-2.0/permissive stamp contradicts the audited CC-BY-4.0 checkpoint.

| Surface | Symbol / key | Change | Shape / signature | Ownership / compatibility | Breaking? | Commit |
|---|---|---|---|---|---:|---|
| `vokra-models::wespeaker` | `WeSpeaker`, `WeSpeakerArtifactLayout`, `ARCH`, `NAME`, `CATEGORY`, `UPSTREAM_HF`, `UPSTREAM_REVISION`, `SOURCE_REVISION`, `EMBED_DIM` | Added | strict constructors, backend/layout/license accessors, frontend/network/embed methods and `SpeakerEngine`; layout variants `PyannotePrefixed`, `OfficialCombinedBare` | Exposes exact 182/219-tensor contracts and observable CPU/Metal selection; no fallback. | no | (TBD) |
| `gguf:vokra.wespeaker.*` | canonical WeSpeaker metadata group | Added | six `u32` values (`sample_rate`, `n_mels`, `frame_length`, `frame_shift`, `embed_dim`, `stage_count`), two `f32` values (`bn_eps`, `stats_eps`), and six strings (`source_revision`, `frontend`, `blocks`, `channels`, `pooling`, `artifact_layout`) | New conversions pin the checkpoint/source revisions, frontend and ResNet/TSTP topology. Historical files are accepted only through exact manifest and provenance gates. | no | (TBD) |
| `include/vokra.h` / `vokra-capi::speaker` | `vokra_speaker_embed` model coverage | Changed (behavior only) | existing signature unchanged; WeSpeaker sessions report/write 256 floats | Extends the existing two-call speaker API without adding a C symbol or changing ownership. | no | (TBD) |

### 2026-08-24 — 1.0.0-rc.1-dev (SepFormer separation CPU/Metal and C ABI)

The seven public SepFormer GGUF repositories now share one complete native
dual-path Transformer runtime. The additive C entry returns stream-major
waveforms and reuses `vokra_audio_free`; session construction selects CPU or
Metal through the existing options object, with no implicit resampling or CPU
fallback. No new GGUF key is introduced.

The first WHAMR 16 kHz and 8 kHz artifacts were published with
`category=enhancement` / `n_out=1`, while the official SpeechBrain
hyperparameters and `[512,256,1,1]` speaker projection require two-source
separation. The runtime recognizes only those exact variant/provenance
combinations as audited legacy metadata and validates the two-stream tensor
shape. New conversions stamp the corrected values.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `include/vokra.h` | `vokra_separate` | Added | `vokra_status_t(const vokra_session_t *, const float *, size_t, int32_t, float **, size_t *, size_t *, int32_t *)` | Exposes complete mono-waveform separation/enhancement as one stream-major Vokra-owned allocation. | no | (TBD) |
| `vokra-core::engines` | `SeparationEngine` | Added | `separate`, `sample_rate`, `output_streams`, and `backend` required methods | Provides a model-independent, backend-observable separation engine slot shared by retained sessions and the C ABI. | no | (TBD) |
| `vokra-core::Session` | separation engine methods | Added | `with_separation_engine`, `separate_audio`, `separation_sample_rate`, `separation_output_streams`, `separation_backend` | Attaches and calls a complete separator while a missing engine remains a loud `NotImplemented`. | no | (TBD) |
| `gguf:vokra.sepformer.*` | WHAMR category/output values | Changed / Breaking artifact semantics | `vokra.model.category: "separation"`; `vokra.sepformer.n_out: u32 = 2` for `whamr16k` / `whamr8k` | Corrects the public converter against official `num_spks: 2`; the strict runtime retains audited compatibility with the two already-published mis-stamped files. | yes (new artifact metadata) | (TBD) |

### 2026-08-21 — 1.0.0-rc.1-dev (runtime gap closure wave)

Additive GGUF wire schema and Rust/CLI surface; the C ABI is untouched.  The
new `vokra.charsiu.*` group binds one pinned
`charsiu/en_w2v2_fc_10ms` release and has no permissive defaults: revision,
checkpoint SHA-256, all topology axes, the official 42-label inventory, and
every tensor name/shape must agree before inference. The same branch completes
PyIN temporal decoding and adds an additive detailed-result API; the historical
`pyin(...) -> Vec<f32>` signature remains source-compatible. The vocoder wave
also adds the minimal public operator surfaces needed by strict BigVGAN,
SpeechT5/SpeechBrain HiFi-GAN, and Vocos real-weight forwards. The existing
`vokra.vocos.variant` wire key is unchanged; the runtime now enforces its two
previously documented values against exact tensor manifests. Qwen3-TTS adds a
strict 0.6B checkpoint handle and decoder-block API over its already-existing
GGUF schema. The remaining TTS loader slice adds strict checkpoint/real-tensor
consumer APIs for eight published GGUFs plus the operator-provisioned VITS-JA
path. No existing `vokra.qwen3_tts.*`, `vokra.chatterbox*.*`,
`vokra.cosyvoice3.*`, `vokra.dia.*`, `vokra.vibevoice.*`, `vokra.voxcpm2.*`,
`vokra.zonos.*`, or `vokra.vits_ja.*` key is added, removed, or changed.
FireRedVAD replaces its unstamped passthrough converter with a canonical native
Stream-VAD variant, persisting the previously optional frontend fields plus
the causal-DFSMN geometry and strict tensor manifest under the existing
`vokra.firered_vad.*` prefix. Historical unvariant artifacts retain their
explicit loud-partial behavior.
SmartTurn replaces its pass-through/loud-partial scaffold with a pinned
Wav2Vec2-base converter and native endpoint forward. Its new
`vokra.smart_turn.*` group is additive and required by the canonical artifact;
older unstamped SmartTurn files fail closed instead of being assigned guessed
processor or topology values.
TEN-VAD similarly replaces a permissive pass-through scaffold with a pinned
19-tensor native contract. Its new `vokra.ten_vad.*` group is additive and
required; the provenance stamp is deliberately corrected from permissive
Apache-2.0 to the non-publishable Agora LicenseRef after primary-source audit.
Moonshine replaces its metadata-only pass-through/binder with exact official
Tiny/Base manifests, a required tokenizer blob, and a native ASR forward. Its
new `vokra.moonshine.*` group pins every topology and artifact identity axis;
old weight-only scaffold artifacts remain identifiable but intentionally fail
the strict executable bind rather than inheriting guessed defaults.
RNNoise adds a native 48 kHz denoise stream and engine implementation over its
existing GGUF schema. This closes the former loud-partial runtime/CLI behavior;
it does not add a C symbol or change a GGUF key.
FSMN-VAD replaces its synthetic two-class/identity-CMVN artifact contract with
the pinned official 248-pdf topology and real `am.mvn` transform. This is an
intentional pre-1.0 breaking GGUF/Rust correction: historical scaffold files
fail closed rather than being guessed onto incompatible weights.
Parakeet-TDT-0.6B-v3 replaces its decoder-head-only loud partial with complete
raw-PCM ASR. Executable conversion embeds the official tokenizer JSON and
stamps the released EOS id. Historical 699-tensor artifacts remain loadable
for the existing decoder/head parity seam by taking the fixed official EOS
default, but text ASR requires an embedded tokenizer and fails loudly when it
is absent.
Whisper-Medusa-v1 replaces its permissive pass-through and plain-Whisper
fallback posture with a config-required canonical artifact and the official
module-0 residual-SiLU output transform. The accelerated future-token tree API
remains explicitly unsupported; ordinary `AsrEngine` now represents the real
released module-0 model rather than silently omitting it.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `gguf:vokra.charsiu.*` | canonical Charsiu metadata group | Added | 16 keys: `revision`/`checkpoint_sha256` (`string`), `hidden_size`/`ffn_dim`/`n_layer`/`n_head`/`vocab_size`/`silence_id`/`pad_id`/`sample_rate`/`pos_conv_kernel`/`pos_conv_groups`/`silence_threshold` (`u32`), `frame_shift_sec`/`layer_norm_eps` (`f32`), `vocab` (`string[]`) | Writer/reader handshake for the canonical 10 ms frame classifier; no topology or label default is inferred. | no | #44 |
| `gguf:vokra.fsmn_vad.*` | canonical FSMN-VAD metadata group | Changed / Breaking | removes scaffold `hidden_dim`, `n_class`, `cmvn_mean`, `cmvn_var`; adds `input_affine_dim`, `linear_dim`, `lstride`, `rstride`, `output_affine_dim`, `output_dim` (`u32`), `cmvn_add_shift`, `cmvn_rescale` (`f32[]`), pinned HF/ModelScope identities, revision and three source hashes (`string`) | Corrects invented topology/identity CMVN to the official 24-weight, 248-pdf release. Old scaffold GGUFs are deliberately rejected before inference. | yes (artifact schema) | #44 |
| `gguf:vokra.firered_vad.*` | canonical Stream-VAD metadata group | Added / persisted | `variant` (`string`), `sample_rate`/`n_mels`/`window_length`/`hop_length`/`n_blocks`/`hidden_dim`/`projection_dim`/`memory_order`/`memory_stride`/`n_class` (`u32`), `required_tensors` (`string[]`) | Selects the official causal DFSMN variant and makes every frontend/cache/tensor axis load-bearing; unknown variants and partial manifests fail closed. | no | #44 |
| `gguf:vokra.parakeet.*` | executable Parakeet-TDT metadata | Added / persisted | `vokra.parakeet.joint.eos_token_id` (`u32`) and `vokra.parakeet.tokenizer.json` (`u8[]`) | Pins official generation termination and embeds the exact BPE + Metaspace decoder needed to render text. Older head-parity artifacts use the fixed official EOS default but remain tokenizer-free. | no | #44 |
| `gguf:vokra.smart_turn.*` | canonical SmartTurn v2 metadata group | Added | `revision`/`checkpoint_sha256`/`config_sha256`/`preprocessor_config_sha256`/`reference_revision` (`string`), `sample_rate`/`max_input_samples`/`hidden_size`/`feature_dim`/`ffn_dim`/`n_layer`/`n_head`/`pos_conv_kernel`/`pos_conv_groups` (`u32`), `max_segment_seconds`/`layer_norm_eps`/`normalization_eps`/`completion_threshold` (`f32`) | Pins the official processor, Wav2Vec2-base topology, endpoint threshold, and independent reference revision; absent or mismatched canonical fields fail closed. | no | #44 |
| `gguf:vokra.ten_vad.*` | canonical TEN-VAD v1.0 metadata group | Added | `revision`/`onnx_sha256`/`frontend_license_spdx` (`string`), `sample_rate`/`hop_size`/`n_features`/`context_frames`/`hidden_dim`/`n_layers` (`u32`) | Pins the exact official graph and frontend contract. Missing/mismatched fields fail closed; canonical provenance also changes to `LicenseRef-Agora-TEN-VAD-Open-Source-License-2025` / `RedistributionForbidden`, preventing model-zoo republication under the formerly incorrect Apache label. | yes (artifact schema/provenance) | #44 |
| `gguf:vokra.moonshine.*` | canonical Moonshine Tiny/Base metadata group | Added | `revision`/`checkpoint_sha256`/`tokenizer_sha256`/`encoder_activation`/`decoder_activation` (`string`), `sample_rate`/`hidden_size`/`intermediate_size`/`encoder_layers`/`decoder_layers`/`encoder_heads`/`decoder_heads`/`vocab_size`/`max_positions`/`decoder_start_token_id`/`eos_token_id` (`u32`), `rope_theta`/`partial_rotary_factor` (`f32`); reuses `vokra.tokenizer.model` U8 array | Pins the two official revisions and corrects the old six-head/all-SwiGLU scaffold assumptions. Missing/mismatched metadata or tokenizer fails closed before weights execute. | yes (artifact schema/behavior) | #44 |
| `gguf:vokra.medusa.*` | canonical Whisper-Medusa-v1 metadata group | Changed / Breaking | `revision`, `source_revision`, `config_sha256`, `checkpoint_sha256`, `heads_type` (`string`); `num_heads`, `module_count`, `num_layers`, `hidden_size` (`u32`); `choices` (`u32[]`); `init_from_proj`, `output_whisper_original` (`bool`) | Replaces optional/guessed scaffold metadata with the exact official 10-future-head/11-module contract. Historical pass-through artifacts fail closed instead of silently running plain Whisper. | yes (artifact schema/behavior) | #44 |
| `vokra-ops::firered_vad` | `FireredVadDfsmnConfig`, `FireredVadDfsmnBlockWeights`, `FireredVadDfsmnWeights`, `FireredVadDfsmnState`, `firered_vad_dfsmn_forward` | Added | stateful causal DFSMN primitive over normalized fbank rows | Native transcription of the official eight-stage streaming graph without an ONNX runtime dependency. | no | #44 |
| `vokra-ops::fsmn_vad` | `FsmnEncoderConfig`, `FsmnBlockWeights`, `FsmnVadWeights` | Changed / Breaking | replaces scaffold hidden/two-class fields with exact input/output affines, 250-wide blocks, 128-wide causal memory and 248-pdf output geometry | Makes the public primitive represent the official checkpoint instead of a shape-incompatible synthetic network. | yes (Rust source) | #44 |
| `vokra-ops::kaldi_fbank` | `KaldiFbankWindow`, `kaldi_fbank_with_window` | Added | explicit `Povey` or symmetric `Hamming` analysis-window selection; existing `kaldi_fbank` remains Povey | Adds the released FunASR frontend contract without changing existing CAM++ callers. | no | #44 |
| `vokra-ops::ten_vad` | TEN-VAD constants, weight/state structs, `TenVadFrontend`, `network_forward` | Added | native 16 kHz LPCNet-derived streaming frontend plus separable-conv/two-LSTM graph | Replaces the explicit unsupported forward without ONNX Runtime, protobuf, or a silent fallback. | no | #44 |
| `vokra-models::firered_vad` | `FireredVadNativeConfig`, `FireredVad::{native_config,forward_features}` | Added | strict native GGUF binder plus feature-level parity entry point | Runs canonical Stream-VAD while retaining the old unstamped loud-partial contract for compatibility. | no | #44 |
| `vokra-models::smart_turn` | `SmartTurnWeights::from_gguf`, `SmartTurn::predict_endpoint` | Changed | strict 221-tensor binder plus native utterance-level Wav2Vec2 endpoint forward | Replaces the old topology-presence binder and explicit unsupported error with the pinned released checkpoint contract; no frame-level `VadEngine` is fabricated. | yes (Rust behavior) | #44 |
| `vokra-models::moonshine` | `Moonshine::from_gguf`, `Moonshine::transcribe`, `AsrEngine for Moonshine` | Changed | strict 160/210-tensor binder, raw-waveform encoder-decoder, tied greedy logits and HF BPE decode | Replaces the explicit unsupported forward and routes `moonshine` through the CLI ASR task. The later Mac backend wave routes both composed attention matrix products through the existing GEMM seam, so Metal covers the declared Moonshine hot-op set without a CPU fallback. | yes (Rust source/behavior) | #44 / post-0.1 Mac wave |
| `vokra-models::rnnoise` | `RnnoiseFrameOutput`, `RnnoiseStream`, `RnnoiseV02::{stream,denoise_pcm}`, `DenoiseEngine for RnnoiseV02`, `DenoiseStreamHandle for RnnoiseStream` | Added / Changed | native 48 kHz, 480-sample-frame denoise engine with arbitrary-chunk streaming and reset | Replaces the explicit unsupported waveform/CLI behavior while preserving the existing canonical GGUF metadata schema. No C ABI symbol is added. | yes (Rust source/behavior) | #44 |
| `vokra-models::fsmn_vad` | canonical binder and `VadEngine` behavior | Changed / Breaking | strict 24-tensor binder; Hamming fbank + online LFR + real AddShift/Rescale; score `1-p(pdf0)` | Replaces identity CMVN, wrong Povey/LFR assumptions and a fabricated two-class speech column with the official executable contract. | yes (Rust/artifact behavior) | #44 |
| `vokra-models::parakeet` | `ParakeetJointConfig::eos_token_id`, `ParakeetAsr::{encode_pcm,transcribe,has_tokenizer}`, `ParakeetTokenizer`, `AsrEngine for ParakeetAsr` | Added / Changed | strict 699-tensor binder; raw PCM to 24-layer FastConformer; recurrent duration-aware TDT ids; BPE + Metaspace text | Replaces decoder/head-only execution and explicit unsupported transcription with complete CPU ASR. Non-CPU requests remain explicit errors in the CLI. | yes (Rust source/behavior) | #44 |
| `vokra-models::whisper_medusa` | `MedusaConfig`; `WhisperMedusa::{from_gguf,config,num_heads,module_count,transcribe_tokens,prefix_logits,transcribe,transcribe_speculative}`; `AsrEngine for WhisperMedusa` | Changed / Breaking | strict pinned 1,281-tensor binder plus backend-dispatched module-0 `hidden + SiLU(linear(hidden))` output adaptation and tied logits | Removes the old optional `MedusaHeads`/`BaseTowerStatus` scaffold and plain-Whisper fallback. The module-0 dense projection follows CPU/Metal selection; Metal uses the per-op Whisper route until its resident session exposes a pre-projection adapter. Accelerated tree decoding remains a named error instead of fabricating a speedup. | yes (Rust source/behavior) | #44 / post-0.1 Mac wave |
| `vokra-convert` | `ModelKind::Charsiu` / `--model charsiu` | Added | canonical 213-tensor safetensors manifest → 211 F32 GGUF tensors | Consumes the eval-dead masking vector and folds positional-conv weight norm offline; missing, extra, retyped, or reshaped tensors are errors. | no | #44 |
| `vokra-convert` | `ModelKind::SmartTurn` / `--model smart-turn` | Changed | canonical 223-tensor F32 safetensors manifest → 221 F32 GGUF tensors | Replaces pass-through conversion with strict manifest binding, offline positional weight-norm folding, and omission of the eval-unused SpecAugment vector. | yes (artifact schema) | #44 |
| `vokra-convert` | `ModelKind::FsmnVad` / `--model fsmn-vad` | Changed | prepare-script-verified 26-tensor bundle → 24 F32 GGUF weights plus real CMVN metadata | Rejects ordinary weight-only files and the former identity-CMVN placeholder; stamps Apache-2.0 weight provenance and pinned source identity. | yes (input/artifact contract) | #44 |
| `vokra-convert` | `convert_parakeet_file_with_tokenizer`; Parakeet `--tokenizer` | Added / Changed | official 699-tensor safetensors plus required BPE + Metaspace `tokenizer.json` → executable GGUF | Makes tokenizer identity an explicit conversion input and validates its model/decoder contract before writing. | yes (CLI input contract) | #44 |
| `vokra-convert` | `convert_whisper_medusa_v1_with_config`; `convert_whisper_medusa_v1_file` | Added / Changed | merged safetensors + exact pinned `config.json` → canonical Whisper-large-v2/Medusa GGUF | The working public path requires the side-car; legacy `convert_file[_licensed]` rejects before reading the 6.25-GB input. The direct file converter gains a config parameter. | yes (Rust/CLI input contract) | #44 |
| `vokra-models::align::charsiu` | `Charsiu::from_file`, `Charsiu::logits`, canonical config/vocabulary fields | Added / Changed | strict GGUF binder + real frame-classification forward + upstream silence-mask/monotone-DTW alignment | Replaces the synthesized-only/unwired loader and the incorrect pre-norm/CTC description with the released post-norm topology and phone-alignment algorithm. This is pre-1.0 source churn; no C symbol changes. | yes (Rust source) | #44 |
| `vokra-backend-cpu::kernels` | `grouped_conv1d_f32` | Added | checked grouped Conv1D composition over the existing dispatched dense kernel | Shared by Charsiu positional convolution and `vokra-eval`; invalid groups/extents fail loudly and no backend fallback is introduced. | no | #44 |
| `vokra-ops::bigvgan_generator` | `AliasFreeActivationWeights` | Added | `pub struct AliasFreeActivationWeights { pub upsample_filter: Vec<f32>, pub downsample_filter: Vec<f32> }` | Binds the released checkpoint's per-activation Kaiser buffers instead of regenerating remembered constants. | no | #44 |
| `vokra-ops::hifigan` | `HifiGanConvPadding` / `hifigan_generator_with_conv_padding` | Added | explicit `Zero` or `Reflect` stride-1 convolution padding | Preserves canonical zero padding while allowing SpeechBrain's released reflect-padding forward without a hidden compatibility mode. | no | #44 |
| `vokra-ops::openwakeword` | `Openwakeword{Melspec,Embedding,DnnClassifier,Conv2d,Dense}Weights`, `openwakeword_{melspectrogram,embedding_forward,dnn_classifier_forward}` | Added | native learned-DFT/mel frontend, embedding CNN, and ordered DNN classifier primitives | Replaces the classifier-only scaffold with the complete official v0.5.1 streaming inference graph; all tensor axes are validated before arithmetic. | no | #44 |
| `vokra-ops::waveform_frontend` | `waveform_frontend_with_right_padding` | Added | `pub fn waveform_frontend_with_right_padding(&[f32], usize, &WaveformFrontendAttrs, &WaveformFrontendWeights) -> Result<Vec<f32>>` | Reproduces fixed-window Wav2Vec2 first-layer GroupNorm statistics without materializing or convolving the constant zero tail. | no | #44 |
| `vokra-ops::f0::pyin` | `PyinFrame` | Added | `pub struct PyinFrame { pub hz: f32, pub voiced: bool, pub confidence: f32 }` | Carries the decoded voiced state plus the real pre-decode voiced probability required by the CLI's shared F0 row contract. | no | #44 |
| `vokra-ops::f0::pyin` | `pyin_detailed` | Added | `pub fn pyin_detailed(&[f32], u32, f32, f32) -> Result<Vec<PyinFrame>>` | Exposes full PyIN output without changing the compatibility wrapper; re-exported through `f0` and the crate root. | no | #44 |
| `vokra-ops::f0::pyin` | `pyin` | Changed | `pub fn pyin(&[f32], u32, f32, f32) -> Result<Vec<f32>>` | Replaces per-frame first-dip argmax with all-trough Beta interval observations and voiced/unvoiced Viterbi smoothing; signature and `0.0` unvoiced convention are unchanged. | no | #44 |
| `vokra-ops::vocos` | `VocosAttrs`, `VocosIstftPadding`, `VocosNormWeights`, `VocosBlockWeights`, `VocosWeights`, `vocos_decode` | Added | native ConvNeXt-1D feature decoder with conditional/plain normalization and center/same iSTFT trimming | Exposes the exact two official Vocos numerical contracts; validation rejects inconsistent axes and condition ids before arithmetic. | no | #44 |
| `vokra-models::{chatterbox,chatterbox_nano,chatterbox_turbo}` | `ChatterboxCheckpoint`, `ChatterboxNanoCheckpoint`, `ChatterboxTurboCheckpoint` and corresponding `*SpeakerProjection` consumers | Added | exact release-manifest binders plus native speaker-conditioning affine projections | Replaces synthesized-only load posture with strict real-GGUF identity/shape checks while keeping AR token generation and terminal vocoders loud-partial. No C symbol or GGUF key changes. | no | #44 |
| `vokra-models::cosyvoice3` | `CosyVoice3Checkpoint`, `CosyVoice3QProjection` | Added | exact 293-tensor binder plus native layer-0 Q projection | Establishes a real checkpoint consumer without claiming the pending flow/HiFTNet PCM path. No C symbol or GGUF key changes. | no | #44 |
| `vokra-models::dia` | `DiaCheckpoint`, `DiaTextEmbedding` | Added | exact 343-tensor binder plus native text embedding lookup | Establishes a real checkpoint consumer while the encoder/decoder generation loop remains loud-partial. No C symbol or GGUF key changes. | no | #44 |
| `vokra-models::qwen3_tts` | `Qwen3TtsCheckpoint`, `Qwen3TtsBoundBlockWeights`, `qwen3_tts_talker_block_forward`, `qwen3_tts_code_predictor_block_forward` | Added | strict official 0.6B-Base 478-tensor binder plus native bias-free GQA/RMSNorm/mRoPE/SwiGLU decoder block | Moves the existing Qwen3-TTS wire schema from synthesized-only inspection to real-checkpoint binding while retaining a loud end-to-end PCM refusal. This is pre-1.0 additive Rust source surface; no C symbol or GGUF key changes. | no | #44 |
| `vokra-models::irodori` | `IrodoriCheckpoint`, `IrodoriTextBlockWeights`, `irodori_text_block_forward` | Added | strict official v3 637-tensor binder plus native gated non-causal MHA/RMSNorm/RoPE/SwiGLU text block | Moves the existing Irodori GGUF schema from synthesized-only inspection to real-checkpoint binding while retaining a loud refusal for the unbound RF-DiT/terminal-codec path. No C symbol or GGUF key changes. | no | #44 |
| `vokra-models::vibevoice` | `VibeVoiceCheckpoint`, `VibeVoiceAcousticProjection` | Added | exact 1,204-tensor binder plus native acoustic-connector affine projection | Establishes a real checkpoint consumer while the diffusion/codec path remains loud-partial. No C symbol or GGUF key changes. | no | #44 |
| `vokra-models::vits_ja` | `VitsJaCheckpoint`, `VitsJaTextEmbedding` | Added | exact 885-tensor canonical ESPnet generator binder plus native phoneme embedding lookup | Allows lawful operator-provisioned local artifacts without fetching or redistributing the JSUT-trained weight; the full VITS forward remains loud-partial. No C symbol or GGUF key changes. | no | #44 |
| `vokra-models::voxcpm2` | `VoxCpm2Checkpoint`, `VoxCpm2StopProjection` | Added | exact 377-tensor binder plus native stop projection | Establishes a real checkpoint consumer while the MiniCPM/audio-VAE generation path remains loud-partial. No C symbol or GGUF key changes. | no | #44 |
| `vokra-models::zonos` | `ZonosCheckpoint`, `ZonosSpeakerProjection` | Added | exact 246-tensor binder plus native speaker-conditioner projection | Establishes a real checkpoint consumer while conditioning, sampling, and DAC decode remain loud-partial. No C symbol or GGUF key changes. | no | #44 |

### 2026-08-22 — 1.0.0-rc.1-dev (NanoCodec grouped FSQ decode — Rust surface only, additive)

Issue #45 adds the allocating and caller-owned-buffer forms of NVIDIA
NanoCodec's grouped, non-residual FSQ dequantizer. The C ABI and GGUF schema
are unchanged. The Rust API is additive; `group_fsq_decode_into` is the
steady-state zero-allocation entry and `group_fsq_decode` is its convenience
wrapper. Both infer `n_groups` from the runtime `[time, n_groups]` code buffer,
so the G=4/8/13 released variants share one API.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-ops::fsq_codec` | `group_fsq_decode` | Added | `pub fn group_fsq_decode(codes: &[u32], time: usize, levels_per_group: &[u32]) -> Result<Vec<f32>>` | Allocating convenience wrapper for #45; concatenates runtime groups and never applies an RVQ residual sum. | no | (TBD) |
| `vokra-ops::fsq_codec` | `group_fsq_decode_into` | Added | `pub fn group_fsq_decode_into(codes: &[u32], time: usize, levels_per_group: &[u32], out: &mut [f32]) -> Result<()>` | Caller-owned per-frame path for #45, guarded by source scan and 1000-call counting allocator test. | no | (TBD) |

Numerical parity is pinned at `atol = 1e-6` against the upstream
`GroupFiniteScalarQuantizer.decode` restored from NVIDIA's real 0.6 kbps
checkpoint. The fixture records the HF repo commit, checkpoint SHA-256 and
NVIDIA-NeMo/Speech source commit; Python is offline-only under a dedicated
Python 3.12 `uv.lock`. Root `Cargo.lock` remains first-party-only.

### 2026-08-22 — 1.0.0-rc.1-dev (streaming continuous speech features — 7 new functions, 1 new typedef, additive)

The C surface grows from **41 fn + 13 typedef to 48 fn + 14 typedef**. All
changes are additive and the v1.0-rc baseline snapshot is intentionally not
rotated before the M5-13 GA freeze. The first implementation is Moshi's causal
Mimi input encoder: it exposes the continuous bottleneck-transformer grid at
25 Hz, before token-rate resampling and RVQ. No GGUF key or runtime dependency
is added.

`push_pcm` accepts arbitrary chunks and retains a partial token frame. `pull`
writes complete feature rows and reports the exact source-sample index of its
first row; later rows follow the queried native frame rate. Both operations use
bounded, preallocated storage after warmup. Queue overflow, undersized output,
NULL arguments, and a session without a feature engine fail loudly without
consuming queued data.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| ------------ | ------ | ---- | --------- | --------- | --------- | -- |
| `include/vokra.h` | `vokra_feat_t` | Added | `typedef struct vokra_feat_t vokra_feat_t` | Opaque streaming-feature state retaining its session | no | (TBD) |
| `include/vokra.h` | `vokra_feat_open` | Added | `vokra_feat_t *(const vokra_session_t *)` | Open the feature stream selected by the session's native model family | no | (TBD) |
| `include/vokra.h` | `vokra_feat_frame_rate_mhz` | Added | `int32_t(const vokra_feat_t *)` | Query the encoder's native rate without caller hard-coding | no | (TBD) |
| `include/vokra.h` | `vokra_feat_dim` | Added | `int32_t(const vokra_feat_t *)` | Query floats per row for caller-owned output sizing | no | (TBD) |
| `include/vokra.h` | `vokra_feat_push_pcm` | Added | `vokra_status_t(vokra_feat_t *, const float *, size_t)` | Stream arbitrary mono PCM chunks into bounded causal state | no | (TBD) |
| `include/vokra.h` | `vokra_feat_pull` | Added | `vokra_status_t(vokra_feat_t *, float *, size_t, size_t *, int64_t *)` | Non-blocking complete-row pull plus sample-accurate start timestamp | no | (TBD) |
| `include/vokra.h` | `vokra_feat_reset` | Added | `void(vokra_feat_t *)` | Return PCM tail, queue, clock, and recurrent caches to as-new state | no | (TBD) |
| `include/vokra.h` | `vokra_feat_destroy` | Added | `void(vokra_feat_t *)` | Release the opaque stream (`NULL` is a no-op) | no | (TBD) |
| `vokra-core` | `SpeechFeatureEngine` / `SpeechFeatureStream` | Added | two public traits | Model-independent engine injection and streaming metadata/push/pull/reset contract | no | (TBD) |
| `vokra-core` | `Session::with_speech_feature_engine` / `Session::open_speech_feature_stream` | Added | public methods | Attach or open the family-specific engine; absence is a loud `NotImplemented` | no | (TBD) |
| `vokra-models` | `MimiEncoder::{feature_dim, feature_frame_hop, encode_features_into, encode_features_all}` / `MimiEncoderState::reset` | Added | public methods | Reuse the native causal encoder and expose its pre-RVQ continuous grid without changing codec output arithmetic | no | (TBD) |

### 2026-08-25 — 1.0.0-rc.1-dev (SNAC hierarchy and conditioned HiFi-GAN backend surface)

The accumulated SNAC 24/44.1 kHz decoder work and the MeloTTS decoder add one
intentional Rust API wave in `vokra-ops`; the C ABI and GGUF schema are
unchanged. SNAC's former fixed three-stage input changes to a checkpoint-shaped
slice so the official four-stage 44.1 kHz release can use the same decoder.
This is a source-breaking prerelease signature change, recorded before the
M5-13 freeze. The HiFi-GAN addition is the backend-dispatched sibling of the
existing conditioned scalar entry and preserves the explicit
conditioning-layer mismatch errors.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-ops::snac_decode` | `SnacDecoder::decode` | Changed | `pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>>` | Accept the checkpoint-derived three- or four-stage hierarchy instead of freezing one array width. | yes | (TBD) |
| `vokra-ops::hifigan` | `hifigan_generator_conditioned_with_backend_ops` | Added | `pub fn hifigan_generator_conditioned_with_backend_ops<O: HifiGanBackendOps>(...) -> Result<Vec<f32>>` | Route speaker conditioning and every learned HiFi-GAN convolution through the selected backend for MeloTTS without a CPU fallback. | no | (TBD) |
| `vokra-ops::snac_decode` | `MAX_SNAC_STAGES` | Added | `pub const MAX_SNAC_STAGES: usize = 4` | Publish the validated upper bound shared by the two official SNAC releases. | no | (TBD) |
| `vokra-ops::snac_decode` | `SnacDecoder::active_vq_strides` | Added | `pub fn active_vq_strides(&self) -> &[u32]` | Expose the checkpoint-selected hierarchy for generic codec callers. | no | (TBD) |
| `vokra-ops::snac_decode` | `SnacConfig::snac_44khz` | Added | `pub const fn snac_44khz() -> Self` | Add the official four-stage 44.1 kHz topology alongside the existing 24 kHz constructor. | no | (TBD) |

### 2026-08-21 — 1.0.0-rc.1-dev (stateful streaming resampler — Rust surface only, additive)

Issue #50 adds a caller-owned-buffer streaming wrapper around Vokra's existing
Kaiser-windowed-sinc resampler. The C ABI and GGUF schema are unchanged. The
Rust API is additive, preserves the one-shot sample sequence across arbitrary
chunk boundaries, and allocates all history storage in `new`; a
1000-steady-state-call counting-allocator test and the zero-allocation source
gate cover `process_into`.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-ops::resample` | `StreamingResampler` | Added | `pub struct StreamingResampler` | Owns the fixed history ring and absolute phase for #50 without exposing implementation fields. | no | (TBD) |
| `vokra-ops::resample` | `StreamingResampler::delay_samples` | Added | `pub fn delay_samples(&self) -> usize` | Reports centered-kernel look-ahead in output-rate samples for device latency budgeting. | no | (TBD) |
| `vokra-ops::resample` | `StreamingResampler::new` | Added | `pub fn new(in_rate: u32, out_rate: u32, quality: u8) -> Result<Self>` | Allocates and validates one fixed-rate streaming resampler. | no | (TBD) |
| `vokra-ops::resample` | `StreamingResampler::process_into` | Added | `pub fn process_into(&mut self, input: &[f32], out: &mut [f32]) -> Result<usize>` | Processes chunks without allocation and uses empty input as an explicit final flush. | no | (TBD) |
| `vokra-ops::resample` | `StreamingResampler::reset` | Added | `pub fn reset(&mut self)` | Reuses the existing allocation for a new stream. | no | (TBD) |

### 2026-08-22 — 1.0.0-rc.1-dev (NanoCodec decoder converter — additive GGUF schema)

Additive on-disk GGUF metadata only. The C ABI, `vokra-core` public surface,
and `vokra-ops` public surface are unchanged. The new converter writes the
checkpoint-derived NanoCodec decoder contract under a dedicated prefix; it
does not reuse or change the existing `vokra.wavtokenizer.*` or
`vokra.xcodec2.*` FSQ schemas.

| Chunk prefix | Keys | Kind | Status | Rationale | Introducing issue |
| --- | --- | --- | --- | --- | --- |
| `vokra.nanocodec.*` | `n_codebooks`, `embed_dim`, `sample_rate`, `frame_hop`, `generator_hop`, `base_channels`, `input_kernel_size`, `output_kernel_size`; `levels_per_group`, `upsample_rates`, `resblock_kernel_sizes`, `resblock_dilations`; `activation`, `output_activation`, `pad_mode`, `nemo_speech_commit`, `nemo_source_url`; `grouped_upsample_expanded` | `u32`; `u32-array`; `string`; `bool` | persisted | Complete checkpoint-derived decoder topology plus the verified official NeMo source identity for the pinned NanoCodec transform. `frame_hop` and `generator_hop` remain separate provenance fields, and conversion rejects disagreement; the published 1.89 kbps checkpoint records 1024 for both. | #47 (2026-08-22) |

The converter also reuses the existing `vokra.provenance.*` namespace for the
immutable source revision and checkpoint SHA-256. Those are not new prefixes.

### 2026-08-22 — 1.0.0-rc.1-dev (NanoCodec causal HiFi-GAN streaming decoder — Rust surface only, advisory)

Additive **Rust public API** only; `include/vokra.h` is untouched. The
`vokra-models` crate now exports module `nanocodec` and the public
`CausalHifiGan`, `CausalHifiGanConfig`, `CausalHifiGanState`, and checkpoint
weight carrier types for causal Conv1d, dense-expanded grouped
ConvTranspose1d, HalfSnake, residual blocks, stages, and the complete decoder.

`CausalHifiGan` adds `new`, `from_gguf`, `config`, `expected_feature_dim`,
`frame_hop`, `state`, `decode_into`, `reset`, and `decode_all`. `from_gguf`
binds the complete decoder-only `vokra.nanocodec.*` schema emitted by the
offline converter and rejects foreign architecture tags, unsupported transform
markers, a foreign/missing NeMo source URL or commit, missing/non-F32 tensors,
and exact-shape mismatches. `decode_into` is
the allocation-free streaming surface and returns the exact sample count
written. The config reads `frame_hop` independently from the checkpoint and
requires it to equal both the stamped `generator_hop` and the checked generator
stride product. The published 21.5 fps archive reports `frame_hop=1024` and
`up_sample_rates=[8,8,4,2,2]` (product 1024), verified directly from the fixed
official checkpoint config. A mismatched future artifact is rejected rather
than silently dropping or inventing waveform samples. No known NanoCodec hop
is embedded as a runtime default.

The implementation is CPU-native. No GPU backend is selected implicitly and
there is no silent CPU fallback. Backend capability registration is tracked by
the sibling #51 change; the offline converter and its end-to-end conversion
roundtrip are tracked by #47.

The topology is transcribed from NVIDIA NeMo Speech (Apache-2.0) commit
`4fcff72febec9395fdbd4bfa0747bfda2ecd3cef`, cross-checked with
NeMo-Speech.cpp commit `4f9676226f667d14608487df744f375db87127f8`.
No model weights are bundled and this entry makes no publication-license
claim.

### 2026-08-22 — 1.0.0-rc.1-dev (generic streaming codec decoder C ABI, issue #48)

Additive prerelease C ABI and Rust engine contract. Codebook count is loaded
from the GGUF and repeated as a call-time argument; no codec topology is frozen
into the header. Each opaque decoder is single-owner-thread state retaining its
source session. Standalone Mimi and NVIDIA NanoCodec provide complete native
implementations; partial codec binders do not opt in.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `include/vokra.h` | `vokra_codec_decoder_t` | Added | opaque struct | Independently owned streaming token-to-PCM state (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_destroy` | Added | `void vokra_codec_decoder_destroy(struct vokra_codec_decoder_t *decoder)` | Release a decoder handle; NULL-tolerant (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_frame_hop` | Added | `int32_t vokra_codec_decoder_frame_hop(const struct vokra_codec_decoder_t *decoder)` | Query checkpoint-derived PCM frame size (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_n_codebooks` | Added | `int32_t vokra_codec_decoder_n_codebooks(const struct vokra_codec_decoder_t *decoder)` | Query model-specific codebook width without an ABI constant (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_open` | Added | `struct vokra_codec_decoder_t *vokra_codec_decoder_open(const struct vokra_session_t *session)` | Open independent causal codec state (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_pull_pcm` | Added | `enum vokra_status_t vokra_codec_decoder_pull_pcm(struct vokra_codec_decoder_t *decoder, float *out, size_t capacity, size_t *out_len)` | Copy pending mono PCM without allocation (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_push_codes` | Added | `enum vokra_status_t vokra_codec_decoder_push_codes(struct vokra_codec_decoder_t *decoder, const uint32_t *codes, size_t n_codebooks, int32_t *out_frames_emitted)` | Push one call-time-shaped code frame (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_reset` | Added | `void vokra_codec_decoder_reset(struct vokra_codec_decoder_t *decoder)` | Restore as-new causal state (#48) | no | #62 |
| `include/vokra.h` | `vokra_codec_decoder_sample_rate` | Added | `int32_t vokra_codec_decoder_sample_rate(const struct vokra_codec_decoder_t *decoder)` | Query checkpoint-derived PCM rate (#48) | no | #62 |
| `vokra-core::engines` | `CodecDecoderEngine` / `CodecDecoderHandle` | Added | public traits | Codec-family-neutral session injection and stateful push/pull contract (#48) | no | #62 |

### 2026-08-15 — 1.0.0-rc.1-dev (LLaMA-Omni2: the converter now stamps the full `vokra.llama_omni2.*` group its own binder reads, and refuses without `--config` — GGUF schema fill + Rust surface, advisory)

**Behaviour change** plus additive Rust surface. The C ABI
(`include/vokra.h`, 33 fn + 11 typedef, v1.0-rc baseline) is **untouched** —
LLaMA-Omni2 is not cbindgen-exported, so `scripts/gen-c-abi.sh --check` sees
no diff. The Rust public-API snapshot gate covers `vokra-core` / `vokra-ops`
/ `vokra-capi` only, so `vokra-convert` and `vokra-models` changes do not
move it. The GGUF chunk-prefix leg of `scripts/check-abi-changelog.sh` is
satisfied: no new `vokra.<group>` prefix appears — `vokra.llama_omni2` was
already stamped (by the one key that was) and is already named in this file.
Recorded here because §Scope puts the `vokra.*` GGUF schema in scope, and
because the pre-1.0 recording rule covers Rust `pub` items.

**Motivation**: `crates/vokra-convert/src/models/llama_omni2.rs` stamped
five strings — arch, name, category, `vokra.llama_omni2.variant`,
upstream_hf — plus provenance, and none of the **ten** numeric keys
`vokra_models::llama_omni2::LlamaOmni2Config::from_gguf` reads. Those ten go
through `read_u32_or_zero` / `read_f32_or`, so every one decayed to its `0`
placeholder and `validate_for_forward` refused the load with
`InvalidArgument("backbone ill-formed (n_layer=0, d_model=0, n_head=0)")`.
**Every GGUF `vokra-cli convert --model llama-omni2` produced failed to load
in the binder written for it.** This is the same defect the
`vokra.openwakeword.*` repair closed one round earlier; the sibling
`kyutai_stt`, which this module's docs name as its precedent, does stamp its
full group, so the precedent was real and simply was not carried over.

Nothing caught it because both halves were tested against a mock of the
other: the binder's unit tests hand-build their GGUF with `GgufBuilder`, and
the converter's tests asserted only the five strings it did stamp. The new
`crates/vokra-models/tests/llama_omni2_convert_bind.rs` runs the real
converter into the real binder, fixture-free, so neither can drift again.

**GGUF schema**: the `vokra.llama_omni2.*` group grows from 1 stamped key to
11 (see the chunk-prefix table). Four are derived from the tensors —
`arch.backbone.n_layer` from the contiguous layer run (a gap is a hard
error), `arch.backbone.d_model` / `arch.backbone.vocab` from the token
embedding axes, and `arch.backbone.intermediate_size` from the SwiGLU gate
projection, whose second axis cross-checks `d_model`. Existing artifacts are
unaffected in the only sense that matters: there are none that worked.

**Rust surface** (`vokra-convert`):

- **Breaking**: `convert_llama_omni2_file` and `convert_llama_omni2_bytes`
  now always return `ConvertError::Usage`. Six axes (`n_head`,
  `rope_max_period`, `rms_norm_eps`, `sample_rate`, `speech_encoder_dim`,
  `speech_decoder_dim`) cannot be read off any tensor shape, and the binder
  refuses a `0` on every one, so no `--config`-less conversion can produce a
  loadable artifact. Refusing is the `ModelKind::Crepe` /
  `openwakeword_op` precedent; the alternative is to keep writing files that
  cannot be opened. Both are kept as named refusals rather than deleted so an
  existing caller gets a routing message instead of a link error.
- **Breaking**: `ModelKind::LlamaOmni2` through `convert_file` /
  `convert_file_licensed` / `convert_file_with_slug` likewise returns
  `ConvertError::Usage` naming the side-car.
- **Added**: `convert_llama_omni2_file_with_config(input, config, output,
  variant, license)` — the working route.
- **Added**: ten `pub const KEY_*` metadata-key constants plus
  `DEFAULT_LAYER_PREFIX` / `DEFAULT_EMBEDDING_TENSOR` /
  `DEFAULT_GATE_PROJ_SUFFIX`, mirroring the binder's own list.
- **Added**: `LlamaOmni2Report` gains `n_layer` / `d_model` / `vocab` /
  `intermediate_size`, so a caller can see what was derived rather than
  re-deriving it.

**CLI**: `--model llama-omni2[-<release>]` gains a dedicated arm and now
requires `--config <config.json>`. It previously fell through the generic
dispatch, which hard-codes `config_side_car = None` and would have dropped
the flag silently.

**Honest limits**: the upstream tensor manifest has still not been
transcribed into this tree, so the names the derivation searches for
(`layer_prefix`, `embedding_tensor`, `gate_proj_suffix`) are side-car knobs
defaulting to the bare HuggingFace Qwen2 spelling. A default *search key* is
admissible where a default *model axis* is not, because a key that matches
nothing yields a hard error naming the knob rather than a plausible wrong
number. The binder's weight store also remains `synthesized` — a successful
load is not a claim that real ICTNLP weights are bound, and `converse` is
still a loud-partial.

### 2026-08-15 — 1.0.0-rc.1-dev (FCPE: the `vokra.f0.fcpe.*` chunk is now stamped by the converter and REQUIRED by the loader — GGUF schema break + Rust surface, advisory)

**Breaking, for GGUF artifacts.** The C ABI (`include/vokra.h`, 33 fn + 11
typedef, v1.0-rc baseline) is **untouched** — no F0 extractor is
cbindgen-exported, so `scripts/gen-c-abi.sh --check` sees no diff. The Rust
public-API snapshot gate covers `vokra-core` / `vokra-ops` / `vokra-capi`
only, so `vokra-models` changes do not move it. The GGUF chunk-prefix leg of
`scripts/check-abi-changelog.sh` is satisfied: no new `vokra.<group>` prefix
is introduced (`vokra.f0` was already stamped by the CREPE converter, and is
already named in this file). Recorded here because §Scope puts the GGUF
metadata schema under `vokra.*` in scope, and because the pre-1.0 recording
rule covers Rust `pub` items.

**Motivation**: `crates/vokra-convert/src/models/fcpe.rs` stamped **none** of
the `vokra.f0.fcpe.*` axes, while `vokra_models::f0::fcpe::FcpeConfig::
from_gguf` read all of them with `.unwrap_or(DEFAULT_*)` and documented
"Missing keys are honored silently". Every FCPE GGUF Vokra has ever produced
therefore described no topology at all, and the loader supplied the released
`fcpe_c_v001` shape for all fourteen axes. FCPE was alone in this among the F0
siblings: the CREPE converter stamps its four axes from a required side-car
config, and the RMVPE converter stamps all ten of its own.

Seven axes are pinned by a tensor length the binder already checks, so a wrong
value there was at least loud. The other seven — `hop`, `n_fft`,
`sample_rate`, `fmin`, `fmax`, `stem_groups`, `confidence_threshold` — are
cross-checked by nothing, and neither was `n_layers` in one direction: an
artifact carrying **more** encoder blocks than the config declared had the
surplus dropped by the `0..n_layers` bind loop and ran a truncated encoder to
completion. Wrong pitch, full frame count, finite values, no error — the same
failure shape round 7 found in RMVPE. `crates/vokra-convert/src/main.rs`
carried a comment asserting the chunk was "written by the model, not the
converter"; nothing wrote it.

| Item | Kind | Signature | Rationale |
| --- | --- | --- | --- |
| GGUF `vokra.f0.fcpe.*` | Changed (schema) | 14 keys, all REQUIRED on a weight-carrying artifact | Seven axes have no cross-check; substituting any of them yields a completed forward and wrong numbers |
| `f0::fcpe::GGUF_KEY_*` | Added | 14 × `pub const &str` | Lets producer, loader and tests name one key each instead of transcribing literals |
| `f0::fcpe::GGUF_REQUIRED_KEYS` | Added | `pub const [&str; 14]` | The enforced set, walked by the test that proves each key is individually required |
| `f0::fcpe::FcpeConfig::from_gguf` | Changed (behaviour) | signature unchanged | Every axis required; an absent key is a `LoadError::Gguf` naming it |
| `f0::fcpe::FcpeConfig::from_gguf_metadata_only` | Added | `(&GgufFile) -> Result<Self, LoadError>` | The lenient reader, scoped to weightless artifacts that cannot run a forward |
| `f0::fcpe::FcpeConfig` | Changed | now derives `PartialEq` | So a test can assert a whole config rather than field-by-field (matches `CrepeConfig`) |
| `f0::fcpe::FcpeWeights::try_from_gguf` | Changed (behaviour) | signature unchanged | Declared `n_layers` is cross-checked against the layer set the artifact carries |
| `vokra-convert models::fcpe::FcpeTopology` | Added | `pub struct` (crate-visible — `models::fcpe` is `pub(crate)`) | The seven axes derived from tensor shapes |
| `vokra-convert models::fcpe::FCPE_V001_TOPOLOGY` | Added | `pub const FcpeTopology` | The gate deciding when the front-end constants may be asserted |
| `vokra-convert models::fcpe::FcpeReport` | Changed | `+ topology`, `+ axes_stamped`, `+ front_end_withheld` | So the CLI note can say which axes were written and which were withheld |

**What the converter now does**: it derives `d_model`, `n_mels`,
`stem_kernel`, `ffn_dim`, `conv_kernel`, `n_layers` and `n_pitch_bins` from
the checkpoint's own tensor shapes — measured, not assumed — and enforces
every cross-check the shapes permit (head width against stem width, pointwise
input against `d_model`, depthwise channels against the post-GLU width, mutual
uniformity across layers, both kernels odd, `ffn_dim` even, at least one
encoder block). The remaining seven live in upstream's Python config and are
in no tensor: the prep script even drops `cent_table`, the one buffer that
would have pinned `fmin` / `fmax`. Those are asserted from the documented
`fcpe_c_v001` constants **only when the derived topology equals
`FCPE_V001_TOPOLOGY` exactly**, and withheld otherwise, because a 16 kHz
front-end asserted onto an unidentified variant is a fabricated axis.

**Who this breaks**: every FCPE GGUF converted before today that carries
weights. They stamp zero axes, so they no longer load; the error names the
first absent key and says to re-run the converter. Nothing trustworthy is
lost — such an artifact was being interpreted entirely by assumption, and if
it was not v001 the pitch it produced was wrong with no way to tell. Nothing
is published: FCPE has no `huggingface.co/vokra` repo and no model-zoo entry,
so the blast radius is locally converted files. Metadata-only artifacts (no
weights) still load, under the lenient reader, because they cannot run a
forward at all.

**Second break, narrower**: a *variant* checkpoint now converts to an artifact
carrying 7 of 14 axes, which the loader then refuses. That is deliberate — the
weights are preserved and correct, and the axes nobody can derive have to come
from whoever knows the variant. The conversion note says so in full. A
`--config` side-car for those seven, mirroring CREPE's, is the natural
follow-up and is not in this change.

**Correction to the 2026-07-30 entry below**: it records this chunk as "13
keys" including `n_heads` and `kernel_size`, and describes it as "Additive —
every key defaults if absent". The 2026-07-30 CFNaiveMelPEInfer rewrite
superseded that shape (no attention, so no `n_heads`; `kernel_size` split into
`conv_kernel` + `stem_kernel`; `stem_groups` added) without updating the
entry, and the "defaults if absent" clause is exactly what this change
removes. The historical entry is left as written; this note is the correction.

**Verify**: no `cargo` run on this pass (16 GB host; sequential verification
deferred to the integrating loop). Checked by hand: `rustfmt --edition 2024
--check` parses all four touched files clean; the derived axes were traced
against the tensor-name / shape contract in the prep script's docstring
(`tools/parity/fcpe_prepare_checkpoint.py`) and the runtime module header; no
`include/vokra.h` edit; no new third-party dependency (root `Cargo.lock`
untouched, NFR-DS-02 preserved); no `docs/license-audit.md` §3.1 change.

### 2026-08-15 — 1.0.0-rc.1-dev (RMVPE: a checkpoint whose U-Net is not discoverable is refused instead of running without it — Rust surface only, advisory)

**Behaviour change** plus additive Rust surface. The C ABI
(`include/vokra.h`, 33 fn + 11 typedef, v1.0-rc baseline) is **untouched** —
no F0 extractor is cbindgen-exported, so `scripts/gen-c-abi.sh --check` sees
no diff and `scripts/check-abi-changelog.sh` does not fire. The Rust
public-API snapshot gate covers `vokra-core` / `vokra-ops` / `vokra-capi`
only, so `vokra-models` changes do not move it either. Recorded here
because the pre-1.0 policy's recording rule covers Rust `pub` items.

**Motivation**: `RMVPE::extract_real` discovered its U-Net by walking the
literal upstream scheme `unet.encoder.block{i}.conv.weight` and breaking at
the first gap, with no check that the walk found anything. Meanwhile the
loader's acceptance filter, `REQUIRED_TENSOR_PREFIXES`, admits seven broad
prefixes — including `encoder.`, `decoder.` and `cnn.`, annotated in-file as
"fallback prefix used by some RMVPE forks". A fork-convention checkpoint
therefore loaded successfully, discovered **zero** blocks, and fed the raw
mel plane straight into the BiGRU. The result was a full-length, finite,
in-band pitch track produced by a model missing its entire CNN, and nothing
distinguished it from a real measurement: not the return type, not the frame
count, not any assertion in the tree. The module header compounded it by
claiming "Every tensor referenced by the CNN + GRU + head is required", which
`from_gguf` never implemented — it requires one tensor matching one of seven
prefixes.

| Item | Kind | Signature | Rationale |
| --- | --- | --- | --- |
| `f0::rmvpe::CnnChainPolicy` | Added | `pub enum { Required, Optional }` (`Default` = `Required`) | Makes the CNN-less path something a caller names, never something a failed lookup selects |
| `f0::rmvpe::RMVPE::from_gguf_with_cnn_policy` | Added | `(&Path, CnnChainPolicy) -> Result<Self, VokraError>` | The explicit constructor for structural fixtures |
| `f0::rmvpe::RMVPE::cnn_policy` | Added | `(&self) -> CnnChainPolicy` | Lets a caller (and a test) see which posture a handle carries |
| `f0::rmvpe::RMVPE::encoder_block_count` | Added | `(&self) -> usize` | Separates "the U-Net ran" from "the U-Net was skipped" — frame count and finiteness cannot |
| `f0::rmvpe::RMVPE::decoder_block_count` | Added | `(&self) -> usize` | Decoder-side counterpart |
| `f0::rmvpe::RMVPE::extract_real` | Changed (behaviour) | signature unchanged | Zero discoverable encoder *or* decoder blocks is now `ModelLoad` under `CnnChainPolicy::Required` |
| `f0::rmvpe::RMVPE::extract` | Changed (behaviour) | signature unchanged | Delegates to `extract_real`, so it inherits the refusal |

**Who this breaks**: a caller loading a checkpoint that does not name its
U-Net the upstream way now gets a `ModelLoad` where it previously got a
track. That track was wrong, so the break is the fix. The error names the
scheme searched, the prefixes the artifact actually carries and a bounded
sample of its tensor names, so a fork checkpoint is diagnosable from the
message alone — and it names `CnnChainPolicy::Optional` for the one case
where running without a CNN is intended.

**The scheme is itself unverified.** `unet.encoder.block{i}.conv.weight` is
this runtime's reading of the upstream layout; no real checkpoint has been
through it. `vokra-convert`'s RMVPE converter passes `state_dict` keys
through verbatim ("Tensor naming contract"), so whatever upstream emits is
what lands in the GGUF — and that crate's own synthetic fixture uses a third
shape again (`unet.encoder.layer0.weight`). If upstream turns out to name
its blocks differently, the first real checkpoint will hit the new error
rather than silently mis-running, and the error prints the artifact's actual
names, which is what settles it. The correct response then is to fix the
walker, not to relax the gate; both the enum docs and `extract_real` say so
in as many words.

**Honest scope — what did NOT change**: the decoder still omits upstream's
`concat(paired encoder feature)` skip branch, so even a fully discovered
U-Net does not reproduce upstream numerics. That divergence was previously
disclosed only in a parenthetical inside a numbered pipeline step; it is now
stated in the module header, in `extract_real`'s doc under "Where this
forward is not upstream RMVPE", and in the Path A parity harness output. No
real checkpoint has been through this path — the parity harness remains
env-gated (`VOKRA_RMVPE_REAL_GGUF` / `VOKRA_RMVPE_REAL_HIDDEN`) and its Path
A compares against no reference. Path A now additionally asserts non-zero
encoder and decoder block counts, and its docstring stops describing the
per-frame `hz` / `confidence` band checks as range validation: both columns
are clamped by `decode_class_to_hz` before return, so those assertions fail
only on `NaN`.

**Verify**: no `cargo` run on this pass (16 GB host; sequential verification
deferred to the integrating loop). Checked by hand: the new fixtures'
shapes were traced against `conv2d_pad_same` / `maxpool2d_2x2` /
`conv_transpose2d_stride2` / `collapse_nchw_to_frames`
(mel `[1,1,101,128]` → encoder `[2,50,64]` → decoder `[2,100,128]` →
BiGRU input 256 → 100 frames, matching the `frame_times` contract for 1 s at
16 kHz); the three in-module tests that relied on the old silent
fall-through now opt in via `from_gguf_with_cnn_policy`; no `include/vokra.h`
edit and no new third-party dependency (root `Cargo.lock` untouched,
NFR-DS-02 preserved).

### 2026-08-15 — 1.0.0-rc.1-dev (F0 family: `CREPE` / `FCPE` gain a fallible `extract` + `extract_real` + `frame_times`, matching RMVPE — Rust surface only, advisory)

**Breaking (Rust surface, pre-1.0 window)** plus additive. The C ABI
(`include/vokra.h`, 33 fn + 11 typedef, v1.0-rc baseline) is **untouched** —
no F0 extractor is cbindgen-exported, so `scripts/gen-c-abi.sh --check`
sees no diff and `scripts/check-abi-changelog.sh` does not fire. Recorded
here because the pre-1.0 policy's recording rule covers Rust `pub` items.

**Motivation**: `CREPE::extract` matched
`Some(w) if sample_rate == 16_000` and let its `_` arm answer **two
different failures with the same value** — "no weights bound" and "weights
bound but the caller passed 44.1 kHz" both produced a frame-count-correct
track of `hz = 0.0 / voiced = false / confidence = 0.0`. Downstream that is
indistinguishable from "this audio is entirely unvoiced", so silently wrong
pitch flowed into whatever consumed it (a vocoder, a VC pipeline). The
docstring directly above claimed the opposite — "a non-16 kHz caller is
honest-refused when weights are bound (no silent resample, FR-EX-08)" —
which no code implemented, and could not have: `extract` returned
`Vec<F0Frame>` with no error channel at all. `FCPE::extract` had the same
shape plus a third case: a `compute_mel` failure discarded by `Err(_) =>`,
under a comment claiming "no silent success on garbage weights".

| Item | Kind | Signature | Rationale |
| --- | --- | --- | --- |
| `f0::crepe::CREPE::extract` | Changed | `(&self, &[f32], u32) -> Result<Vec<F0Frame>, VokraError>` (was `-> Vec<F0Frame>`) | Delegates to `extract_real`; the obvious name is now the one that measures |
| `f0::crepe::CREPE::extract_real` | Added | `(&self, &[f32], u32) -> Result<Vec<F0Frame>, VokraError>` | The real forward, under the name the parity harnesses use |
| `f0::crepe::CREPE::frame_times` | Added | `(&self, usize, u32) -> Vec<f32>` | Analysis timebase alone; bare seconds, never `F0Frame` |
| `f0::crepe::CREPE::has_real_weights` | Added | `(&self) -> bool` | Lets a caller branch instead of handling the error |
| `f0::crepe::NATIVE_SAMPLE_RATE` | Added | `pub const u32 = 16_000` | The rate the CNN is defined at, so the refusal can name it |
| `f0::fcpe::FCPE::extract` | Changed | `(&self, &[f32], u32) -> Result<Vec<F0Frame>, VokraError>` (was `-> Vec<F0Frame>`) | Same delegation |
| `f0::fcpe::FCPE::extract_real` | Added | `(&self, &[f32], u32) -> Result<Vec<F0Frame>, VokraError>` | Real forward; propagates the STFT/mel error verbatim |
| `f0::fcpe::FCPE::frame_times` | Added | `(&self, usize, u32) -> Vec<f32>` | Analysis timebase alone |

**Errors are distinguished, not merged**: `VokraError::ModelLoad` for an
unbound weight set (and, for FCPE, an unusable `hop` / `sample_rate`),
`VokraError::InvalidArgument` for a rate the checkpoint is not defined at
(naming both what it received and what it needs), and for FCPE the
front-end's own error propagated verbatim. Neither extractor resamples on
the caller's behalf — refusing is the point (FR-EX-08).

**Shape**: deliberately identical to the `extract` / `extract_real` /
`frame_times` split RMVPE received the same day, so the family reads the
same way at every call site rather than carrying three different answers to
one problem. `CREPE::extract_full` also stopped `.expect()`-ing on
`forward_one` now that an error channel exists above it.

**Callers updated**: `crates/vokra-models/tests/parity_crepe.rs` (now
asserts `has_real_weights()` before comparing, so a weightless GGUF cannot
silently "pass" a parity run against zeros); `crates/vokra-cli/src/engine.rs`
(both `BOUND_ARCHES` rows move to `BoundReason::RealForwardNoCliTask`, and
`BoundReason::SkeletonFallback` — whose only two users these were — is
**removed** rather than left as a label no row could honestly carry; a
`dead_code` warning under `-D warnings` would have caught it either way);
`crates/vokra-cli/src/run.rs` and `crates/vokra-models/src/{lib.rs,f0/mod.rs}`
doc corrections.

**Verify**: no `cargo` run on this pass (16 GB host, sequential verification
deferred to the integrating loop). Checked by hand: every workspace call
site of the changed methods was re-grepped and updated; `SkeletonFallback`
has zero remaining referents; no `include/vokra.h` edit and no new
third-party dependency (root `Cargo.lock` untouched, NFR-DS-02 preserved).

### 2026-08-14 — 1.0.0-rc.1-dev (WP-23 piper-plus landing: `PiperPlusTts::synthesize_streaming` + `PiperPlusTtsStream` single-chunk fallback + `TtsStreamHandle` re-export — Rust surface only, advisory)

Additive **Rust public API** entry for the FR-ST-04 streaming-surface
unification on piper-plus (Vokra's first native TTS). The C ABI
(`include/vokra.h`, 33 fn + 11 typedef, v1.0-rc baseline) is **untouched** —
every change is Rust-surface, not cbindgen-exported (`scripts/gen-c-abi.sh
--check` = no diff).

**Motivation**: The `TtsEngine::synthesize_stream` trait method (WP-23,
landed in the 2026-08-10 SBV2 v2 wave) loudly refused with
`VokraError::UnsupportedOp` on every engine — including piper-plus — so
downstream callers had to special-case full-utterance engines against
future chunk-wise engines (piper-plus M4-03 chunked decode / SBV2
streaming). This wave lands the piper-plus override as a **single-chunk
fallback**: the full `TtsEngine::synthesize` path runs synchronously to
produce the PCM, which is then wrapped in a `PiperPlusTtsStream` that
yields the buffer once on `next_pcm_chunk()` and `None` afterwards. The
generation time is unchanged — MB-iSTFT-VITS2 is a full-utterance
synthesizer with no architectural chunk-boundary — and per FR-EX-08 this
is stated in the docstring rather than hidden behind a name that
promises incremental generation. A future chunk-boundary decoder (the
WP-M4-03 piper streaming path) overrides this method to emit true
incremental chunks; downstreams that already use `TtsStreamHandle` will
pick up the improvement without an API change.

**Backward compatibility**: Purely additive. The pre-existing
`PiperPlusTts::synthesize_pseudo_streaming` inherent method (the honest
name that FR-ST-04 mandates for the full-PCM route) is unchanged and
still returns `SynthesizedAudio`. The new inherent methods return
`Result<PiperPlusTtsStream>` (the stronger-typed variant). No existing
call sites break.

**Files touched**:
- `crates/vokra-core/src/lib.rs`
  — Added `TtsStreamHandle` to the `pub use engines::{...}` re-export
    list, so downstream consumers can name the trait as
    `vokra_core::TtsStreamHandle` (in addition to the pre-existing
    `vokra_core::engines::TtsStreamHandle` path). The trait itself was
    already `pub trait` and is present in
    `docs/abi/vokra-rust-public-api.v1.0-rc.list` (line 1313); this
    change adds a shorter re-export path, not a new symbol.
- `crates/vokra-models/src/piper_plus/mod.rs`
  — New top-level import `use vokra_core::{... TtsStreamHandle, ...};`.
  — New `pub fn PiperPlusTts::synthesize_streaming(&self, request:
    &SynthesisRequest) -> Result<PiperPlusTtsStream>` — mirrors
    `TtsEngine::synthesize` (uses the placeholder `tokenize()` path)
    and wraps the resulting `SynthesizedAudio` in the single-chunk
    stream.
  — New `pub fn PiperPlusTts::synthesize_streaming_with(&self, request:
    &SynthesisRequest, phonemizer: &dyn Phonemizer) ->
    Result<PiperPlusTtsStream>` — mirrors
    `synthesize_pseudo_streaming` (takes an injected G2P) and wraps its
    output.
  — New `pub struct PiperPlusTtsStream { pcm: Option<Vec<f32>>,
    sample_rate: u32 }` with a `pub(crate) fn new` constructor (only
    the two `synthesize_streaming*` methods build one at runtime; the
    crate-visible constructor exists for the wrapper-only unit tests).
  — New `impl TtsStreamHandle for PiperPlusTtsStream` — drains the
    buffer on the first `next_pcm_chunk()` call, returns `None`
    afterwards, and reports the voice's sample rate.
  — New `impl TtsEngine for PiperPlusTts { fn synthesize_stream(...)
    -> Result<Box<dyn TtsStreamHandle + Send>> { ... } }` — overrides
    the default (which loudly refuses per FR-EX-08) with a boxed
    single-chunk wrapper.
  — Six new tests in a new `stream_wrapper_tests` module:
    `stream_yields_single_chunk_then_none`,
    `stream_returns_first_chunk_bit_exact`,
    `stream_reports_configured_sample_rate`,
    `stream_handle_is_send`,
    `stream_handles_empty_pcm_as_single_empty_chunk`,
    `synthesize_streaming_symbol_shape_matches_wrapper`,
    `tts_engine_stream_override_returns_boxed_handle_shape`.

**Verification**: `scripts/gen-c-abi.sh --check` clean (no C ABI diff);
zero-dep unchanged (root `Cargo.lock` still `vokra-*` only, NFR-DS-02);
FR-ST-04 respected — the surface unification is documented as a
single-chunk fallback and the existing `synthesize_pseudo_streaming`
name is preserved so the honest full-PCM route stays reachable under
its FR-ST-04-mandated name.

### 2026-08-14 — 1.0.0-rc.1-dev (`AsrEngine` impl for `DistilWhisperAsr` — Rust surface only, advisory)

Additive **Rust public API** entry that wires
`vokra_models::distil_whisper::DistilWhisperAsr` into the
[`vokra_core::engines::AsrEngine`] trait, so a distil-whisper handle can be
injected via `vokra_core::Session::with_asr_engine` and drive
`session.asr().transcribe()` end-to-end. The C ABI (`include/vokra.h`, 33 fn
+ 11 typedef, v1.0-rc baseline) is **untouched** — every change is
Rust-surface, not cbindgen-exported (`scripts/gen-c-abi.sh --check` = no diff).

**Motivation**: `DistilWhisperAsr::from_gguf` had been landed as a Delegate-
kind handle over the shared `WhisperAsr` load path (real weights, real
greedy decode), but the type did not implement the `AsrEngine` trait, so
`Session::with_asr_engine(distil).asr().transcribe(pcm)` would trip a
`NotImplemented` from the facade instead of running the delegate. This
wave adds the trait impl (verbatim the `WhisperAsr` composition pattern:
inherent transcribe → `render_ids` → `Transcription::new`) so the same
`session.asr()` surface every other ASR consumes also drives distil-whisper.

**Backward compatibility**: Additive. The inherent
`DistilWhisperAsr::transcribe(pcm) -> Result<Vec<u32>>` is unchanged and
still wins method resolution when the receiver is a concrete
`DistilWhisperAsr` (Rust prefers inherent methods with matching receiver +
argument shape over trait methods, so callers of the inherent are
unaffected). The trait method is reached via `dyn AsrEngine` dispatch or
explicit UFCS (`<DistilWhisperAsr as AsrEngine>::transcribe(&asr, pcm)`).
The empty-PCM `InvalidArgument` early return is honored by both entry
points because the trait method delegates to the inherent one.

**Files touched**:
- `crates/vokra-models/src/distil_whisper/mod.rs`
  — New top-level `use vokra_core::engines::AsrEngine;` and
    `use vokra_core::tasks::Transcription;` imports.
  — New `impl AsrEngine for DistilWhisperAsr { fn transcribe(&self,
    pcm: &[f32]) -> Result<Transcription> { ... } }` (composition of the
    two existing inherent helpers).
  — New `#[cfg(test)] pub(crate) fn
    DistilWhisperAsr::from_whisper_asr_for_test(WhisperAsr) -> Self`
    (test-only constructor that wraps an already-loaded `WhisperAsr` into a
    Delegate-kind handle without enforcing the distil invariant — required
    for the trait-dispatch tests below; not exposed outside `#[cfg(test)]`
    so production callers still go through `from_gguf`).
  — Three new tests exercising the trait method end-to-end via the
    Delegate arm: `asr_engine_transcribe_delegate_returns_finite_transcription`,
    `asr_engine_transcribe_rejects_empty_pcm`,
    `asr_engine_transcribe_composes_with_inherent_transcribe`.
- `crates/vokra-models/src/whisper/decoder.rs`
  — New `pub(crate) fn test_support::tiny_model_distil(n_audio_layer,
    n_text_layer) -> Arc<WhisperModel>` and its private
    `tiny_encoder_layer` helper. Whisper-shape (`n_audio_ctx = 1500`) so
    the encoder path passes its post-conv2 length check; consumed by the
    distil-whisper trait tests. Encoder-layer weights are deterministic
    with the same `rect` / `tiny_attn` shape convention the decoder
    `tiny_layer` uses. Adds `EncoderLayer` to the existing
    `whisper::weights::{...}` import list.

**Verification**: `scripts/gen-c-abi.sh --check` clean (no C ABI diff);
zero-dep unchanged (root `Cargo.lock` still `vokra-*` only, NFR-DS-02);
`use vokra_core::engines::AsrEngine` is an internal-crate trait import,
not a new external dep.

### 2026-08-13 — 1.0.0-rc.1-dev (M3-06 T14: `mimi_rvq` Metal MSL kernel + `Compute::mimi_rvq_f32` Metal arm wired — Rust surface only, advisory)

Additive **Rust public API** entry for the M3-06 T14 landing that flips
`HotOp::MimiRvq.covered_by_metal()` from `false` to `true` and swaps the
Metal arm of `Compute::mimi_rvq_f32` from an explicit
`VokraError::UnsupportedOp` to a real MSL kernel dispatch. The C ABI
(`include/vokra.h`, 33 fn + 11 typedef, v1.0-rc baseline) is
**untouched** — every change is Rust-surface, not cbindgen-exported
(`scripts/gen-c-abi.sh --check` = no diff).

**Motivation**: `mimi_rvq_decode` — the shape-generic FP32 gather + fold
behind the Mimi (Kyutai) RVQ codec decode (and the M3-06 FR-OP-30
op family — the still-deferred DAC / EnCodec siblings will reuse the
same kernel shape once their per-quantizer projections are wired) — had
no GPU implementation. The M3-06 T14 MSL kernel
(`vokra_mimi_rvq_gather_fold_f32`) closes that gap for Metal; the CUDA
half (M3-06 T15 NVRTC kernel) stays owner-track on vast.ai and remains a
loud `UnsupportedOp` as before. FR-EX-08 no-silent-fallback posture is
preserved by validating shape + per-index bounds on the host **before**
dispatch (the MSL kernel itself has no per-element bound check; a stray
index would be a silent OOB gather without the host-side guard).

**Backward compatibility**: The public signature of
`Compute::mimi_rvq_f32` is unchanged. Every prior caller that did
`.expect_err("must be UnsupportedOp on Metal")` on a valid input must
update — the Metal arm now returns `Ok(Vec<f32>)` when Metal is
available and the input passes validation. Invalid inputs (shape
mismatch, out-of-range codebook index, wrong table count / shape) are
still explicit `VokraError::InvalidArgument`; the only removed error
surface is the deferred-kernel `UnsupportedOp`.

**Files touched**:
- `crates/vokra-backend-metal/src/context.rs`
  — New MSL kernel in `KERNELS_MSL`: `vokra_mimi_rvq_gather_fold_f32`
    (naive gather + FP32 fold; grid `(d_model, time)` 2-D dispatch with
    16×16 threadgroups; ragged-tail guard). Mirrors
    `vokra_ops::mimi_rvq::rvq_fold_core` semantics.
  — New `#[repr(C)] struct MimiRvqDims { n_codebooks, codebook_size,
    d_model, time: u32 }` block for `setBytes:` at index 3.
  — New `MetalContext` field: `mimi_rvq_gather_fold_pipeline: Id`,
    compiled at `build`, released in `Drop` (LIFO order).
  — New public method `MetalContext::mimi_rvq_gather_fold_f32(codes,
    tables_flat, n_codebooks, codebook_size, d_model, time) ->
    Result<Vec<f32>>` (heap-returning to match `Compute::mimi_rvq_f32`).
  — New private helper `MetalContext::run_mimi_rvq_gather_fold` (mirror
    of `run_dequant_gemv`).
- `crates/vokra-models/src/compute.rs`
  — `HotOp::covered_by_metal` flipped: MimiRvq is now Metal-covered;
    DAC / EnCodec RVQ and FSQ siblings stay deferred (docstring +
    lock-step test updated).
  — `Compute::mimi_rvq_f32` Metal arm: was
    `Err(VokraError::UnsupportedOp)`, now validates shape + per-index
    bounds on the host, flattens `[n_codebooks][codebook_size, d_model]`
    into one FP32 buffer, then delegates to
    `MetalContext::mimi_rvq_gather_fold_f32`.
  — `metal_coverage_is_consistent` test: MimiRvq moved into the
    positive-cover list; removed the "MimiRvq must be uncovered"
    assertion; kept the still-deferred DAC / EnCodec / FSQ negative
    assertions.
  — Renamed `metal_mimi_rvq_arm_is_unsupported_no_silent_fallback` →
    `metal_mimi_rvq_arm_runs_kernel_and_rejects_oob_index`. New body
    verifies (a) trivial (1,1,1) FP32 fold matches CPU, (b) OOB code
    index → `InvalidArgument` (FR-EX-08), (c) shape mismatch →
    `InvalidArgument`.
  — `metal_mimi_rvq_off_metal_is_backend_unavailable` docstring updated
    (behavior unchanged: off-feature Metal is still
    `BackendUnavailable`).
- `crates/vokra-ops/src/mimi_rvq.rs`
  — Module docs updated: the "GPU seam" section now records the Metal
    landing (kernel name, atol bound, FR-EX-08 host-side validation) and
    narrows the still-deferred backend list to CUDA / Vulkan / WebGPU.
- `crates/vokra-models/tests/mimi_rvq_metal_bit_identical.rs` (NEW)
  — Off-feature band: `Compute::for_backend(Metal, [MimiRvq])` is
    `BackendUnavailable`.
  — Metal band (Apple + `--features metal`): tiny-shape CPU vs Metal
    parity (atol ≤ 5e-4, plus `mimi_rvq_decode` bit-identical anchor on
    CPU); canonical-shape (n_codebooks = 8, codebook_size = 32,
    d_model = 64) parity with a **negative control** (a 0.1 codebook
    perturbation moves CPU output past 5e-4, proving the bound is not
    vacuous); OOB code → `InvalidArgument`; wrong table count / shape →
    `InvalidArgument`; empty `time = 0` → empty `Vec<f32>`. Each Metal
    test clean-skips on hosts without a Metal device (prints a reason —
    never a fabricated pass).

**GGUF metadata**: none added (this is a runtime kernel wire; the
existing `vokra.mimi.n_codebooks` / `vokra.mimi.codebook_size` /
`vokra.mimi.d_model` chunks that M3-09 will emit stay untouched).

**Semver impact (0.9.x window)**: minor, additive. No public function
signature changed. The `HotOp::MimiRvq` variant is `#[non_exhaustive]`
in intent; downstream `match` arms that already handle the
still-deferred DAC / EnCodec / FSQ variants continue to compile.

**Verify (2026-08-13)**: `cargo check -p vokra-backend-metal` clean;
`cargo check -p vokra-models --features metal` clean; `cargo test -p
vokra-models --features metal mimi_rvq` — see PR body / commit message
for the atol max on this M1 iMac. `scripts/check-zero-deps.sh`
unaffected (no new external crate). `scripts/gen-c-abi.sh --check` no
diff (Rust surface only).

**Landing checkpoint**: One CC-wave landing (Vocoder Metal 先鋒 = P2
sub-wave 1/11). The DAC / EnCodec RVQ kernels + M3-06 T15 CUDA NVRTC
kernel remain follow-up waves.

### 2026-08-13 — 1.0.0-rc.1-dev (RMVPE Wave 2: real U-Net + BiGRU forward + `forward_from_hidden` — Rust surface only, advisory)

Additive **Rust public API** entry for the RMVPE Wave 2 landing that
resolves the loud-partial `extract_real` stub on
`vokra_models::f0::rmvpe::RMVPE`. The C ABI (`include/vokra.h`, 33 fn
+ 11 typedef, v1.0-rc baseline) is **untouched** — every change is
Rust-surface, not cbindgen-exported (`scripts/gen-c-abi.sh --check` =
no diff).

**Motivation**: `RMVPE::extract_real` had returned a loud
`VokraError::UnsupportedOp` "kernel binding pending" stub since
2026-07-30, alongside a real weight loader + rank-shape gate + real
mel front-end + `decode_class_to_hz` primitive. The stub was honest
(FR-EX-08 loud-partial posture) but blocked downstream VC / TTS
consumers from wiring the API surface end-to-end against a real
checkpoint. This wave binds the missing CNN + BiGRU + head + sigmoid
+ decoder chain against the already-bound weights, so a real GGUF
converts straight through to a per-hop F0 track without the loud
stub. Bit-exact numeric parity against the upstream `yxlllc/RMVPE`
Python remains gated on the owner-side dumper wave (see below).

**Backward compatibility**: `extract_real` still returns
`Result<Vec<F0Frame>, VokraError>` — the signature is unchanged; only
the error surface is narrowed (never returns `UnsupportedOp` under
this landing; a mis-composed weight set is a
`VokraError::ModelLoad`). Every prior caller that did
`.expect_err("must be unsupported")` must update to
`.expect("must run cleanly on a real GGUF")` — search for
`extract_real_is_loud_pending_error` / `expect_err(_, ..., UnsupportedOp)`
against RMVPE. Existing `RMVPE::extract` placeholder is preserved
verbatim.

**Files touched**:
- `crates/vokra-models/src/f0/rmvpe.rs`
  — New public methods:
    - `RMVPE::forward_from_hidden(hidden, n_frames, feature_dim, sample_rate)`
      — env-gated alternate entry point for the parity harness to feed
      a dumped post-CNN hidden state directly (bypasses mel + CNN).
  — Rewritten (previously stub):
    - `RMVPE::extract_real(pcm, sample_rate)` — real forward through
      `mel_spectrogram` + discoverable U-Net encoder / decoder blocks
      + bidirectional PyTorch-native GRU + Linear (or Conv1d/Conv2d
      with kernel=1) head + sigmoid + `decode_class_to_hz` per frame.
  — New private forward primitives (file-scope, no `vokra-ops`
    dependency to preserve NFR-DS-02 at the RMVPE seam):
    `conv2d_pad_same`, `batchnorm2d_apply`, `maxpool2d_2x2`,
    `conv_transpose2d_stride2`, `leaky_relu_inplace`,
    `sigmoid_inplace`, `linear_forward`, `gru_cell_step`,
    `collapse_nchw_to_frames`, `head_shape`.
  — New private weight-discovery helpers on `RmvpeWeights`:
    `encoder_block(i)`, `decoder_block(i)`, `discover_gru_shape()`,
    `apply_bigru(...)`, `head_shape_and_slices()`. All return
    `RmvpeBlock<'_>` / `RmvpeHead<'_>` view structs.
  — `RMVPE::weights` field's `#[allow(dead_code)]` removed (the
    forward now consumes it).
- `crates/vokra-models/tests/parity_rmvpe.rs`
  — `parity_rmvpe_gguf_smoke` (env-gated) upgraded from "expect
    UnsupportedOp" to "assert real frames + shape / finite /
    sigmoid-range contract".
  — New env-gated `parity_rmvpe_from_hidden_argmax_match_rate` —
    Path B parity harness feeding `VOKRA_RMVPE_REAL_HIDDEN` +
    `VOKRA_RMVPE_REAL_ARGMAX` + `VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM`
    against the argmax-match-rate ≥ 99 % gate.
- Unit tests in `f0::rmvpe::tests`:
  — Removed `extract_real_is_loud_pending_error` (superseded).
  — Added `extract_real_refuses_gguf_missing_required_tensors`
    (FR-EX-08 loud-error contract).
  — Added `extract_real_returns_real_frames_with_synthetic_weights`
    (positive smoke on a self-consistent no-CNN fixture).
  — Added `forward_from_hidden_returns_real_frames_with_synthetic_weights`
    (positive smoke on the hidden-driven path).
  — Added `forward_from_hidden_refuses_wrong_length` (FR-EX-08 length
    check).

**Zero-dep** (NFR-DS-02): every forward primitive is inline
(`crates/vokra-models/src/f0/rmvpe.rs` file scope). No new
dependencies, no vokra-ops seam changes; root `Cargo.lock` unchanged.

**No new C ABI surface** — RMVPE is a Rust-only model surface (the
consumer is `vokra_models::VoiceClonePipeline` in
`vokra-voiceclone-experimental`, which is out-of-tree per the ELVIS
Act separation). `include/vokra.h` byte-for-byte unchanged;
`docs/abi/vokra.h.v1.0-rc-baseline.symbols` untouched.

**M5-13 relevance**: additive Rust surface only, so
`scripts/check-abi-changelog.sh` does not gate on this entry.
`abi-diff.sh --gate` is still non-firing (v1.0-rc pre-release policy;
IF-01 semver freeze is M5-13/v1.0 GA).

**Parity gate — env-gated, owner action**: bit-exact numeric parity
against the upstream `yxlllc/RMVPE` Python is gated on the owner-side
dumper `tools/parity/rmvpe/dump_reference.py`. That dumper **has since
landed** (this entry originally called it a future WP at path
`tools/parity/rmvpe_dump_reference.py`, which never existed); its real
invocation is `--pt-path` / `--upstream-src` / `--out-dir` plus exactly
one of `--pcm | --canned`, and it emits raw little-endian
`hidden.f32` + `argmax.u32` alongside a `meta.json` carrying
`feature_dim` — **not** `.npy` files. See
`tools/parity/rmvpe/README.md` for the owner walkthrough. Env vars
gate the harness:

- `VOKRA_RMVPE_REAL_GGUF` — Path A (full end-to-end shape / finite /
  sigmoid-range contract).
- `VOKRA_RMVPE_REAL_HIDDEN` + `VOKRA_RMVPE_REAL_ARGMAX` +
  `VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM` — Path B (argmax-match-rate
  ≥ 99 % against dumped reference indices, isolates numerical parity
  from CNN topology drift).

Absent either env var, the harness skips cleanly — never a fabricated
pass. See `crates/vokra-models/tests/parity_rmvpe.rs` for both fixture
recipes.
### 2026-08-14 — 1.0.0-rc.1-dev (C ABI: GPU backend selection + speaker embedding — 8 new functions, 2 new typedefs)

The first **real C ABI addition** since the v1.0-rc baseline snapshot: the
surface goes from **33 fn + 11 typedef to 41 fn + 13 typedef**. Every delta is
`Added`; nothing is removed, renamed or re-signed, and the two existing
constructors keep their exact behaviour. Design:
`docs/superpowers/specs/2026-08-14-c-abi-backend-speaker-design.md` (Approved
2026-08-14). The **rc-window prerelease ABI policy applies** — the IF-01 freeze
still fires at M5-13 / v1.0 GA, so the v1.0-rc anchor is deliberately **not**
rotated here.

**Motivation**: two capabilities existed in Rust but were unreachable from C,
i.e. unreachable from *every* binding (Unity / Godot / Python / Swift /
Kotlin), and both had to land before the freeze made them impossible to add
without a major bump.

1. **Backend selection.** `crates/vokra-capi/src/session.rs` hard-coded
   `BackendKind::Cpu`, and its own comment recorded that adding a selector
   argument would be a breaking change — so the selector lands as *new*
   symbols. Metal and CUDA are already hardware-verified (M2-01 / M2-03) but
   could not be chosen from C.
2. **Speaker embedding** (`speaker_encode` / `speaker_verify`, FR-OP-80 /
   FR-OP-81). `SpeakerEncoder::embed` and `speaker_verify` had no C entry.
   CLAUDE.md design note 8 keeps voice *cloning* in the separate
   `vokra-voiceclone-experimental` repo under the ELVIS Act split while
   speaker *embedding* stays in core; exposing it here is that decision's
   consequence, and it is what makes zero-shot TTS usable from a binding.

| Crate / area      | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| ----------------- | ------ | ---- | --------- | --------- | --------- | -- |
| `include/vokra.h` | `vokra_backend_available` | Added | `bool vokra_backend_available(int32_t backend)` | Ask whether a backend is usable before selecting it; a query has no failure mode, so it returns `bool` and leaves `vokra_last_error()` alone (design §3.4) | no | (TBD) |
| `include/vokra.h` | `vokra_backend_t` | Added | `typedef enum vokra_backend_t { VOKRA_BACKEND_CPU = 0, VOKRA_BACKEND_METAL = 1, VOKRA_BACKEND_CUDA = 2, VOKRA_BACKEND_VULKAN = 3, VOKRA_BACKEND_WEBGPU = 4, } vokra_backend_t` | The five compute backends. CoreML / QNN get **no value** — the delegate selector lands after the real-hardware NPU bakeoff (design D1, `docs/handoff/m4-12.md` §(e)-3); appending `= 5` / `= 6` later stays additive | no | (TBD) |
| `include/vokra.h` | `vokra_session_create_from_bytes_with_options` | Added | `enum vokra_status_t vokra_session_create_from_bytes_with_options(const uint8_t *data, size_t len, const struct vokra_session_options_t *opts, struct vokra_session_t **out_session)` | Backend-selectable twin of `vokra_session_create_from_bytes`; `opts = NULL` reproduces it exactly (design §3.4) | no | (TBD) |
| `include/vokra.h` | `vokra_session_create_from_file_with_options` | Added | `enum vokra_status_t vokra_session_create_from_file_with_options(const char *path_utf8, const struct vokra_session_options_t *opts, struct vokra_session_t **out_session)` | Backend-selectable twin of `vokra_session_create_from_file`; `opts = NULL` reproduces it exactly (design §3.4) | no | (TBD) |
| `include/vokra.h` | `vokra_session_options_create` | Added | `struct vokra_session_options_t *vokra_session_options_create(void)` | Handle-returning constructor: the single failure mode is allocation, so `NULL` is the whole error contract (design §3.4) | no | (TBD) |
| `include/vokra.h` | `vokra_session_options_destroy` | Added | `void vokra_session_options_destroy(struct vokra_session_options_t *opts)` | Destroy contract, `NULL` is a no-op (ADR-0003 §3-a) | no | (TBD) |
| `include/vokra.h` | `vokra_session_options_set_backend` | Added | `enum vokra_status_t vokra_session_options_set_backend(struct vokra_session_options_t *opts, int32_t backend)` | Records the backend; an unknown value is `VOKRA_ERROR_INVALID_ARGUMENT` and leaves the object unchanged (design §5) | no | (TBD) |
| `include/vokra.h` | `vokra_session_options_t` | Added | `typedef struct vokra_session_options_t vokra_session_options_t` | Opaque options object (design D2) — future knobs are one more setter, never an `_ex2` overload and never a struct layout pinned into the frozen surface | no | (TBD) |
| `include/vokra.h` | `vokra_speaker_embed` | Added | `enum vokra_status_t vokra_speaker_embed(const struct vokra_session_t *session, const float *pcm, size_t num_samples, int32_t sample_rate, float *out_embedding, size_t out_capacity, size_t *out_written)` | `speaker_encode` (FR-OP-80): PCM in, embedding into a caller-owned buffer. Takes the waveform, not a filterbank, because no host binding can compute Kaldi fbank (design §3.4) | no | (TBD) |
| `include/vokra.h` | `vokra_speaker_verify` | Added | `enum vokra_status_t vokra_speaker_verify(const float *a, size_t a_len, const float *b, size_t b_len, float threshold, float *out_similarity, bool *out_same_speaker)` | `speaker_verify` (FR-OP-81): takes no session, so stored embeddings can be matched later without a model loaded (design §3.4) | no | (TBD) |
| `vokra-core`      | `SpeakerEngine` | Added | `pub trait SpeakerEngine: Send + Sync { fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>>; fn backend(&self) -> BackendKind; }` | Injection point behind `vokra_speaker_embed`, same shape as `AsrEngine` / `VadEngine` (design §4) | no | (TBD) |
| `vokra-core`      | `AsrEngine::backend` | Changed | `fn backend(&self) -> BackendKind` (new **required** method) | Lets a caller verify the backend it selected actually reached the engine. Deliberately **not** defaulted: a defaulted `Cpu` would let a new engine report the CPU while dispatching elsewhere, which is the exact lie this accessor exists to prevent. Source-breaking for out-of-tree `impl AsrEngine` (none exist in-tree beyond the three updated here); pre-1.0 policy applies | no\* | (TBD) |
| `vokra-core`      | `Session::with_speaker_engine` | Added | `pub fn with_speaker_engine(self, engine: Arc<dyn SpeakerEngine>) -> Self` | Attaches the speaker engine; sibling of `with_asr_engine` etc. | no | (TBD) |
| `vokra-core`      | `Session::speaker` | Added | `pub fn speaker(&self) -> Speaker<'_>` | Task facade (`session.speaker().embed(...)`); the engine accessors are `pub(crate)`, so this is the public entry `vokra-capi` calls | no | (TBD) |
| `vokra-core`      | `Speaker` | Added | `pub struct Speaker<'a>` (with `pub fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>>` and `pub fn backend(&self) -> BackendKind`) | Facade type, re-exported at the crate root beside `Asr` / `Tts` / `S2s` | no | (TBD) |
| `vokra-core`      | `Asr::backend` | Added | `pub fn backend(&self) -> BackendKind` | Facade sibling of `Speaker::backend`; reports the session's own selection when no engine is injected | no | (TBD) |
| `vokra-models`    | `impl SpeakerEngine for SpeakerEncoder` | Added | trait impl | Owns the PCM → `kaldi_fbank(camplus)` → `embed` chain so the C entry, the CLI speaker arm and `PiperPlusTts::embed_reference` agree numerically | no | (TBD) |
| `vokra-models`    | `impl AsrEngine::backend for WhisperAsr` / `VoxtralAsr` | Added | trait method impl (returns the engine's own `backend_kind` / `backend` field) | Follows the new required method; no behaviour change | no | (TBD) |

**Why the selector is `int32_t` and not `enum vokra_backend_t` in the
prototypes**: C permits any `int` to travel through an `enum` parameter, but
materialising an out-of-range discriminant as a Rust enum is undefined
behaviour — and the contract requires exactly that case to return
`VOKRA_ERROR_INVALID_ARGUMENT`, which is only definable if the value arrives
as an integer. The named constants are still emitted (cbindgen `[export]
include`), so C callers write `vokra_session_options_set_backend(opts,
VOKRA_BACKEND_METAL)` unchanged and the implicit enum → int conversion does the
rest. This is the one place the shipped header differs from the design's §3.3
sketch, which wrote `enum vokra_backend_t backend`; §5's "unknown enum value →
`INVALID_ARGUMENT`" row is the requirement that decided it.

**No silent CPU fall back (FR-EX-08)**: a backend that is not compiled in, or
whose device is absent, is rejected at session creation with
`VOKRA_ERROR_BACKEND_UNAVAILABLE` (probed through
`vokra_models::make_backend`, the same oracle `vokra_backend_available`
answers with). A backend that *is* present but that the model's engine has no
path onto is `VOKRA_ERROR_UNSUPPORTED_OP` — the Whisper ASR and CAM++ speaker
arches bind `.with_backend(...)`; piper-plus TTS and Moshi full-duplex S2S now
do the same, while Silero VAD still refuses a non-CPU selection rather than
running on the CPU behind the caller's back. That arch split is the same one `vokra-cli`'s
`cpu_only_engine_label` guard enforces; keep the two in lock-step.

**Backward compatibility**: `vokra_session_create_from_file` and
`vokra_session_create_from_bytes` are unchanged in signature *and* in
behaviour — both now call the shared builder with the same
`BackendKind::Cpu` they always passed, and an integration test asserts the
options path with an explicit CPU backend, and with `opts = NULL`, reproduces
the legacy constructors' model output **bit-for-bit** over the committed
Silero fixture. No existing symbol was touched, so binaries built against the
previous header keep working.

**Files touched**: `crates/vokra-capi/src/options.rs` (new),
`crates/vokra-capi/src/speaker.rs` (new), `crates/vokra-capi/src/session.rs`,
`crates/vokra-capi/src/ffi_guard.rs` (`guard_ptr` / `guard_bool` — the panic
firewall for the handle-returning and `bool`-returning entries, which the
status-typed `guard` cannot wrap), `crates/vokra-capi/src/lib.rs`,
`crates/vokra-capi/Cargo.toml` (`metal` / `cuda` / `webgpu` pass-throughs, all
first-party), `crates/vokra-capi/cbindgen.toml` (`[export] include` + banner),
`crates/vokra-core/src/{engines,session,tasks,lib}.rs`,
`crates/vokra-models/src/speaker/camplus.rs`, `include/vokra.h` (regenerated).
Review follow-up additionally touched
`crates/vokra-models/src/{whisper/asr.rs,voxtral/asr.rs}` (the new required
`AsrEngine::backend`), `crates/vokra-capi/tests/c_abi_backend_options.rs`,
`.github/workflows/ci-platform.yml` and
`.github/workflows/nightly-full-parity.yml`.

**Zero-dep** (NFR-DS-02): unchanged — every new dependency edge is
`vokra-*`-internal (`vokra-capi` already depended on `vokra-models` and
`vokra-ops`), and `scripts/check-zero-deps.sh` passes. The new `metal` / `cuda`
/ `webgpu` features only forward to the existing `vokra-models` features and
are off by default, so the CPU-only audit-trail build and the CPU + Vulkan-only
build target keep excluding those backends from the dependency graph.

**Review follow-up (same day)**: an adversarial pass over this diff ran
mutations against the new tests and found three places where deleting
production code left every test green. All three are closed here.

1. **Nothing verified that a selected backend reached the engine.** Removing
   `.with_backend(backend)` from both backend-honoring arms of `inject_engine`
   passed the whole suite, including a Metal build with a real CAM++ GGUF.
   Backends are bit-identical by design, so no output comparison can catch
   this — hence the new required `AsrEngine::backend` / `SpeakerEngine::backend`
   accessors and `build_session_threads_the_selected_backend_into_the_engine`,
   which was re-run under the same mutation and now fails
   (`left: Cpu, right: Metal`).
2. **The GPU-feature build of `vokra-capi` had no CI job**, making the three
   `reject_cpu_only_backend` guards unreachable in every configuration. Added
   to `ci-platform.yml`'s `gpu-backends` job (metal / cuda arms only —
   `vokra-capi` has no coreml / qnn feature).
3. **The gated real-GGUF C ABI tests ran nowhere.** The `campplus` cell in
   `nightly-full-parity.yml` only exercised `vokra-models`, so "the C ABI
   returns the right embedding" was unverified. Added a `campplus-capi` leg
   over the same GGUF.

Two smaller fixes: `t3` accepted `VOKRA_OK` from an available GPU backend over
a CPU-only arch — the very silent fall back it exists to forbid — and is now
pinned to `UNSUPPORTED_OP` with a both-branches-were-taken assertion; and the
`copy_nonoverlapping` in `vokra_speaker_embed` no longer rests on an
unverified `embedding.len() > 0` premise (its `out_embedding` NULL check only
runs when `out_capacity > 0`).

`no*` in the table above = additive at the C ABI, source-affecting at the Rust
API edge; the pre-1.0 prerelease policy (rename / remove allowed with a dated
entry) covers it.

**Closing the GPU-feature × real-model gap**: the mutation above is only caught
where a GPU feature *and* a real model are both present, and no job had both.
`nightly-full-parity.yml` cannot supply one — it resolves models from a
runner-local path held in a repository variable, which is unset, and its
`ubuntu-latest` runners have no such file, so every real-GGUF leg there
(including the `campplus-capi` one added above) currently takes the honest-skip
path. `gpu-backends` had the features but no model at all.

So `gpu-backends` now fetches the models itself: the published, already
converted GGUFs from **huggingface.co/vokra** (public, no token) —
`campplus-speaker-encoder/campplus.gguf` (27.7 MB) and
`whisper-base/whisper-base.gguf` (290.9 MB) — cached across runs and pinned by
sha256 (the HF LFS oid), so an upstream change fails the job instead of quietly
testing something else. Both files are **byte-identical to the ones the local
mutation check ran against** (`c760971d…` / `7e774425…` verified on both
sides), so the hand verification and the CI leg exercise the same weights. The
test step also gained `--nocapture` so a skip states its reason in the log
rather than passing silently.

Two limits remain, both honest rather than hidden. Whether a hosted macOS
runner exposes a usable Metal device is **not yet established** — no
device-requiring test has run in CI, and the existing `camplus_metal_matches_cpu`
leg has always short-circuited on the missing GGUF before reaching its device
probe — so the first run of this job is what settles it; if the device is
absent the wiring test skips and says so. And the `cuda` arm is `ubuntu-latest`
with no GPU, so its GPU leg skips by construction; that arm's value is the
feature build and the CPU-only-arch refusal path.

**M5-13 relevance**: these ten C symbols are part of the surface IF-01 will
freeze at v1.0 GA. They were added *before* the freeze precisely so they need
not break it later. The v1.0-rc anchor
(`docs/abi/vokra.h.v1.0-rc-baseline.symbols`) is intentionally left at 33 fn +
11 typedef — rotating it is an owner action at the freeze, and this entry is
what lets the gate accept the delta in the meantime.

### 2026-08-10 — 1.0.0-rc.1-dev (WP-23: `TtsEngine` trait extension + `SynthesisRequest::style_vec` / `speaker_id` — Rust surface only, advisory)

Additive **Rust public API** entry for the WP-23 `TtsEngine` extension
(SBV2 `style_vec` + multi-speaker `speaker_id` threading; streaming
placeholder). The C ABI (`include/vokra.h`, 33 fn + 11 typedef) is
**untouched** — the trait, the new struct fields, and the new placeholder
trait are all Rust-surface-only, not cbindgen-exported
(`scripts/gen-c-abi.sh --check` = no diff).

**Motivation**: the pre-WP-23 SBV2 `TtsEngine` adapter (`vokra-models::sbv2`)
hard-coded `SbV2SynthRequest::speaker_id = 0` and `style_vec = vec![0.0;
d_style()]`, silently discarding any caller-supplied speaker choice or
style conditioning that came in through the cross-engine
[`SynthesisRequest`] shape. WP-23 lifts both into the unified request
and lets a caller advertise capability up-front so a mixed-engine
pipeline (piper-plus + SBV2 + Kokoro + ...) never has to know which
engine reads which optional field.

**Backward compatibility**: `SynthesisRequest` is already
`#[non_exhaustive]` (external crates cannot use struct-literal
construction), so adding two `Option<..>` fields with `None` defaults
in `SynthesisRequest::new` and a matching builder for each is a purely
additive change. The `TtsEngine` trait's three new methods all have
defaults (`false` / `false` / `Err(UnsupportedOp)`) so every existing
implementor (piper-plus, Kokoro, CosyVoice2, ...) keeps compiling
without a source edit.

**Files touched**:
- `crates/vokra-core/src/engines.rs` — `SynthesisRequest` gains
  `style_vec: Option<Vec<f32>>` + `speaker_id: Option<u32>`, matching
  `with_style_vec` / `with_speaker_id` builders, and `SynthesisRequest::new`
  updated to default both to `None`. `TtsEngine` gains three defaulted
  methods (`supports_style_vec` = `false`, `supports_multi_speaker` =
  `false`, `synthesize_stream` = loud `VokraError::UnsupportedOp`). New
  placeholder trait `TtsStreamHandle` (methods `next_pcm_chunk`,
  `sample_rate`) pins the incremental-streaming *shape* for a later WP.
- `crates/vokra-models/src/sbv2/mod.rs` — `<SbV2Model as TtsEngine>::synthesize`
  now threads `request.style_vec` / `request.speaker_id`. Both capability
  probes override to `true`. New `#[doc(hidden)] pub fn
  synthetic_for_test_with_nonzero_style() -> Self` for a WP-23 threading
  test that observes `None` vs `Some(nonzero)` PCM difference.
- `crates/vokra-models/tests/sbv2_tts_engine_extension.rs` (new) — 5 tests
  covering both PCM-difference paths + loud-error paths + capability
  advertisement.
- `crates/vokra-core/src/engines.rs` (unit tests) — 4 trait-level tests
  using a spy `TtsEngine`.

**Downstream `TtsEngine` implementors** (piper-plus / Kokoro /
CosyVoice2) compile untouched.

**Zero-dep** (NFR-DS-02): all edits inside `vokra-core` and
`vokra-models`; root `Cargo.lock` unchanged.

**M5-13 relevance**: additive Rust surface only, so
`scripts/check-abi-changelog.sh` does not gate on this entry. Snapshot
rotation is the M5-13/IF-01 freeze owner's action. All items are
additive (existing signatures unchanged; `SynthesisRequest` is already
`#[non_exhaustive]` for future-safe growth), Breaking? = no.

**Snapshot regenerated 2026-08-10** to close the advisory gate
(`abi-surface (advisory)` was red because the new `with_speaker_id` /
`with_style_vec` builders + `TtsStreamHandle` trait were missing from
the v1.0-rc snapshot). Regenerated via
`bash scripts/rust-public-api-list.sh --update-snapshot`; delta = 3
entries under `vokra-core::engines::` matching this WP verbatim.
`abi-diff.sh --gate` is still non-firing (v1.0-rc pre-release policy;
IF-01 semver freeze is M5-13/v1.0 GA).

### 2026-08-10 — 1.0.0-rc.1-dev (SBV2 v2 ZH branch: WP-07 `vokra-math` + WP-13a/14/16/18/19 — Rust surface only, advisory)

Grouped additive **Rust public API** entry covering the 2026-08-10 wave
that landed the SBV2 v2 Chinese (`language_id = 2`) branch and the
first-party scalar transcendental crate it — plus `vokra-ops`,
`vokra-bert` and `vokra-models::sbv2` — was extracted for. C ABI
(`include/vokra.h`, 33 fn + 11 typedef baseline) is **untouched**
across all six WPs (`scripts/gen-c-abi.sh --check` = no diff); no
`vokra.*` GGUF chunk was renamed or removed, though WP-14 adds new
optional `vokra.bert_base.*` hparam keys + a `vokra.bert.wordpiece.*`
tokenizer side-car chunk (both additive, existing SBV2 v2 3-file
loader path unaffected).

| WP / area                                | New export(s) / behaviour change                                                                                                                                                            | Kind    | Rationale                                                                                                                                                | Breaking? |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| WP-07 (new crate `vokra-math`)           | 7 new top-level `pub fn` for `f32`: `exp`, `tanh`, `sqrt`, `sin`, `cos`, `log`, `log1p`                                                                                                     | Added   | Extracts the scalar transcendental primitives that `vokra-backend-cpu`'s scalar path already used so `vokra-ops`, `vokra-bert` and `vokra-models::sbv2` can reach them without pulling the whole CPU-kernel tier upward (WP-05 owner decision, `docs/adr/sbv2-libm-strategy.md`). `core`-only, no `libm`. Follows the M5-03 `vokra-vad-micro` precedent for first-party leaf-crate additions. | no        |
| WP-14 (`vokra-convert`)                  | `convert_bert_base_file(input, output, license, tokenizer_bytes, do_lower_case) -> Result<BertBaseReport, ConvertError>`; `pub struct BertBaseReport`; `ModelKind::BertBase` variant + slug + `convert_file` dispatch | Added   | Plain-BERT (`BertForMaskedLM`) converter for `hfl/chinese-roberta-wwm-ext-large` (Apache-2.0). Emits `bert_base.*` tensor names + `vokra.bert_base.*` hparam chunk + optional `vokra.bert.wordpiece.*` tokenizer side-car. First consumer = SBV2 v2 ZH branch; the `--tokenizer` + `do_lower_case` axes also cover future English WordPiece checkpoints. | no        |
| WP-16 (`vokra-bert`)                     | `pub struct BertBaseEncoder` (+ `impl BertEncoder for BertBaseEncoder` in `lib.rs`, `from_gguf` constructor, `forward(ids, segments)`, `d_model()`)                                       | Added   | Clean-room plain-BERT encoder (Devlin 2018) sitting on WP-14's GGUF schema. The runtime side of the WP-14 converter; consumed by WP-19.                | no        |
| WP-18 (`vokra-models::sbv2::g2p`)        | `Language::ZH` enum variant (language_id = 2); `SbV2Phonemizer::with_zh_g2p(zh_g2p: Box<dyn Phonemizer>, zh_mapping: HashMap<i64, (u16, u8)>) -> Self` builder                              | Added   | SBV2 v2 gains ZH via piper-plus 8-language G2P reuse; this WP lands only the vokra-models-side trait boundary + delegation. `phonemize(_, Language::ZH)` fail-closes with `NotImplemented` when no ZH G2P is wired (FR-EX-08 — no synthetic char-map that could mask absence). | no        |
| WP-19 (`vokra-models::sbv2`)             | `SbV2Model::from_gguf_with_zh_bert(main, bert_ja, bert_en, bert_zh) -> Result<Self>` additive 4-file loader; `SbV2BertContainer` gains `zh: Option<BertBaseEncoder>` + `zh_tokenizer: Option<BertWordpieceTokenizer>` fields (both default `None`) | Added   | Wires WP-16 + WP-17 into the SBV2 v2 language-id-2 slot without changing the pre-WP-19 3-file `from_gguf(main, bert_ja, bert_en)` signature — every existing call site keeps compiling and behaving identically. `d_bert` consistency guard extended to the ZH branch. | no        |
| WP-13a (`vokra-models::sbv2`, behaviour) | `<SbV2Model as TtsEngine>::synthesize` — pre-Blocker-3 orphan rejection block that returned `VokraError::InvalidArgument` on `SynthesisRequest::speaker_embedding = Some(_)` is **removed** | Fixed   | The Blocker-3 refactor (commits `0351a3a` / `2a50088` / `70bd8a7`, speaker conditioning moved into the pipeline) landed the test + rustdoc contract but accidentally left the adapter's upstream rejection block intact. WP-13a removes the orphan; the loud-error contract is now correctly enforced by the inherent `SbV2Model::synthesize` (which raises `InvalidArgument` when `.with_external_speaker_projection` has not been wired). No signature change. | no        |

**Companion WPs recorded in prose only (no Rust surface change, out-of-scope
per L44)**: WP-08/10/11/12 (`sbv2/libm`) — replace direct `f32::exp` /
`f32::tanh` / etc. call sites in `vokra-ops::hifigan`, `vokra-bert::deberta_v2`,
`vokra-models::sbv2` with the newly-extracted `vokra_math::*` primitives, so
the same reproducible-across-hosts scalar path is used everywhere. The
`vokra-math` dependency edge is additive to each `Cargo.toml`; behaviour is
bit-identical to WP-07's own tests.

**Zero-dep** (NFR-DS-02): `vokra-math` is a first-party leaf crate with
`core`-only dependencies; root `Cargo.lock` gains no external entries.
The new `[dependencies]` edges from `vokra-ops`, `vokra-bert`, and
`vokra-models` into `vokra-math` are all `vokra-*` → `vokra-*`, so the
`vokra-*`-only invariant holds.

**M5-13 relevance**: additive Rust surface only, so
`scripts/check-abi-changelog.sh` does not gate on this entry. The
`docs/abi/vokra-rust-public-api.v1.0-rc.list` snapshot needs a
subsequent regenerate (`bash scripts/rust-public-api-list.sh
--update-snapshot`) to close the `abi-surface (advisory)` job — same
posture as the WP-23 companion above.

### 2026-08-10 — 1.0.0-rc.1-dev (SBV2 v2 Blocker 3 close-out + Blocker 2c Wave 1 spline primitive — Rust surface only, advisory)

Additive **Rust public API** entry for the two SBV2 v2 blocker
close-outs that landed on `feat/sbv2-voxtral-real-verify-2026-08-06`
on 2026-08-10, alongside the WP-23 and SBV2 v2 ZH branch entries above.
The C ABI (`include/vokra.h`, 33 fn + 11 typedef baseline) is
**untouched** (`scripts/gen-c-abi.sh --check` = no diff). Follows the
SBV2 v2 ZH branch and X-Codec-2 precedents for new pub modules /
accessors in `vokra-models`: **advisory Rust-surface entry**,
`scripts/check-abi-changelog.sh` does not gate on it (no C symbol
changed).

| Commit    | Area                                       | New export(s) / behaviour change                                                                                                                                                                                                                                                                                       | Kind  | Rationale                                                                                                                                                                                                                                                                    | Breaking? |
| --------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `1a90e0d` | `vokra-models::sbv2` (Blocker 3 close-out) | `pub fn SbV2Model::speaker_projection(&self) -> Option<&ExternalSpeakerProjection>` accessor                                                                                                                                                                                                                          | Added | Companion to the pre-existing (WP-13a era) `ExternalSpeakerProjection` type + `with_external_speaker_projection` builder + `speaker_projection: Option<..>` field. Lets callers / tests observe whether external-speaker wiring has been attached without reaching into private state. | no        |
| `f1b7815` | `vokra-models::sbv2::spline` (new mod)     | `pub mod spline` at `crates/vokra-models/src/sbv2/mod.rs:83`, exposing `pub struct SplineParams<'a>` (bin-edge / knot-height / knot-slope descriptor) + `pub fn rational_quadratic_spline_forward(x: f32, p: SplineParams<'_>) -> f32` + `pub fn rational_quadratic_spline_inverse(y: f32, p: SplineParams<'_>) -> f32` | Added | Rational-quadratic spline math primitive that the SBV2 SDP flow-body forward / inverse pair need. Kept as a small `pub mod` so downstream tests (and the `#[ignore]`d `sdp_body_matches_torch_ref` scaffold) can exercise the math primitive in isolation.                | no        |

**Blocker 2c residual behaviour anchors** (commits `5027b2b` /
`879ba8e` / `c8e2777`): no Rust surface change worth flagging
separately, but preserved here as behavioural anchors for the SDP
flow-body parity gate:

- `5027b2b` — `spline.rs`'s internal `.sqrt()` sites route through
  `vokra_math::sqrt` (the WP-07 first-party scalar transcendental
  crate) so the spline math is bit-exact across every host without
  a `libm` intrusion into `vokra-models`'s dependency graph.
- `879ba8e` — SBV2 `from_gguf` gains a defensive loud-fail check for
  `sbv2.sdp.flows.<even>.*` unread tensors (FR-EX-08 — a mis-shaped
  converter can no longer silently drop even-indexed flow layers).
- `c8e2777` — `#[ignore]`d `sdp_body_matches_torch_ref` scaffold
  parked on the owner fixture wait; the gate flips on when
  `tests/fixtures/sbv2/sdp-body-torch-ref.bin` lands.

**Zero-dep** (NFR-DS-02): all edits inside `vokra-models`; root
`Cargo.lock` unchanged.

**M5-13 relevance**: additive Rust surface only, so
`scripts/check-abi-changelog.sh` does not gate on this entry. Snapshot
rotation is the M5-13/IF-01 freeze owner's action; regenerating
`docs/abi/vokra-rust-public-api.v1.0-rc.list` via
`bash scripts/rust-public-api-list.sh --update-snapshot` will pick up
both the `speaker_projection` accessor and the new `spline::*` re-exports.

### 2026-08-09 — 1.0.0-rc.1-dev (WP-17: `BertWordpieceTokenizer` clean-room in `vokra-bert` — Rust surface only, advisory)

Additive **Rust public API** entry for the WP-17 clean-room WordPiece
tokenizer (Devlin 2018 + Wu 2016 primary sources). C ABI
(`include/vokra.h`, 33 fn + 11 typedef baseline) is **untouched**
(`scripts/gen-c-abi.sh --check` = no diff); the new
`vokra.bert.wordpiece.*` GGUF side-car chunk is additive and gated on
the caller passing `--tokenizer <vocab.txt>` to WP-14's
`convert_bert_base_file` (default = tokenizer chunk not emitted).

| Crate / area                    | Symbol                                                                                                                                                                                                                          | Kind  | Rationale                                                                                                                                                                                                                                 | Breaking? |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `vokra-bert::wordpiece`         | `pub struct BertWordpieceTokenizer { … }` (with `from_gguf`, `encode`, `decode`, `vocab_size`, `cls_id`, `sep_id`, `pad_id`, `unk_id` methods); `pub enum OovPolicy` (variants `Strict`, `MapToUnk`)                            | Added | Runtime tokenizer for `hfl/chinese-roberta-wwm-ext-large` (SBV2 v2 ZH branch) and for future English `google-bert/bert-base-*` checkpoints. Feeds `BertBaseEncoder::forward` (WP-16) with the same tokenization the upstream `BertTokenizer` produces. | no        |

**Zero-dep** (NFR-DS-02): entirely inside `vokra-bert`; no new external
crate. `OovPolicy::Strict` (the default) enforces FR-EX-08 by returning
a loud error on unknown pieces instead of silently substituting `[UNK]`.

**M5-13 relevance**: additive Rust surface only, so
`scripts/check-abi-changelog.sh` does not gate on this entry. Snapshot
rotation is the M5-13/IF-01 freeze owner's action.

### 2026-08-09 — 1.0.0-rc.1-dev (Wave 3 HGAN-05 speaker conditioning + Wave 6 packed-cache exports — Rust surface only)

Additive **Rust public API** changes only — C ABI (`include/vokra.h`) untouched
(baseline 33 fn / 11 typedefs unchanged; `scripts/check-abi-changelog.sh`
green), no `vokra.*` GGUF key added or renamed. Captures the SBV2 v2 waves
that landed 2026-08-09 (72-gap audit-plan execution, Waves 1-8).

The `docs/abi/vokra-rust-public-api.v1.0-rc.list` snapshot is regenerated
(net +38 lines) to match. The full-surface additions grouped by wave:

| Wave / area                    | New export(s)                                                                                                     | Kind  | Rationale                                                                        | Breaking? |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------------- | --------- |
| Wave 3 HGAN-05 (vokra-ops)     | `GinCondition`, `hifigan_generator_conditioned`                                                                   | Added | Real HiFi-GAN `dec.cond(g)` speaker conditioning path (SBV2/CosyVoice2/BigVGAN)  | no        |
| Wave 2 HGAN-01 (vokra-core)    | `ir::graph::ResBlockType` enum                                                                                    | Added | ResBlock1 vs ResBlock2 topology switch — SBV2 needs V1's convs2 chain            | no        |
| Wave 4 (vokra-core)            | `gguf::writer::has_tensor`                                                                                         | Added | Converter needs to check-then-emit optional-slot fabricated zeros                | no        |
| Wave 7B LOUD-PARTIAL (vokra-ops) | `rnnoise::MIN_LAG_SAMPLES`, `rnnoise::MAX_LAG_SAMPLES`                                                          | Added | Real autocorrelation-based pitch analysis exposes its search bounds              | no        |

Plus the `vokra_core::rng::*` re-exports the 2026-08-08 entry already
records; regenerating the snapshot captures them under `USE` rows too.
Snapshot generated via `scripts/rust-public-api-list.sh --update-snapshot`
per the tool's own README.

### 2026-08-08 — 1.0.0-rc.1-dev (Philox4x32-10 + MT19937 + torch.randn parity primitives added to `vokra_core::rng` — Rust surface only, advisory)

Additive **Rust public API** entry for the PR #27 RNG plumbing that gives
`SbV2SDP::sample` a byte-exact `torch.randn(..., device='cpu')` parity path.
The C ABI (`include/vokra.h`, 33 fn + 11 typedef baseline) is **untouched**
(`scripts/gen-c-abi.sh --check` = no diff). Follows the SBV2 / FSMN-VAD
precedent for internal-only Rust-surface additions: advisory changelog entry,
`scripts/check-abi-changelog.sh` does not gate on it (no C symbol changed).

**Module reshuffle** (structural, no behaviour change): `crates/vokra-core/src/rng.rs`
→ `crates/vokra-core/src/rng/mod.rs` with children `mt19937.rs`,
`normal_kernel.rs`, `philox_round.rs`, `philox_state.rs`, `seed_init.rs`.
Pre-existing exports (`SplitMix64`, `GaussianSplitMix64`, `Xorshift64Star`,
`NormalSource`) stay at the same public paths — the `pub use` re-exports at
the crate root are byte-exact against the flat-file layout.

**New public exports** (all re-exported through `vokra_core::rng::*`, so
consumers see them at the same crate path they'd use for the pre-existing
generators):

| Symbol | Path | Kind | Origin |
|---|---|---|---|
| `philox4x32_10` | `mt19937::` — via `philox_round` | Added | Steps 1-2 (Random123 KAT-audited) |
| `PHILOX_M0` / `PHILOX_M1` / `PHILOX_W0` / `PHILOX_W1` / `PHILOX_ROUNDS` | `philox_round::` | Added | Steps 1-2 (KAT constants) |
| `PhiloxState` | `philox_state::` | Added | Step 3 (128-bit counter, O(1) seek) |
| `TorchPhiloxState` | `seed_init::` | Added | Step 4 (PyTorch-compatible seed init) |
| `SCALE` / `u32_to_uniform_f32_pytorch` | `normal_kernel::` | Added | Step 5 (bit-exact `PhiloxRNGEngine.h` bridge) |
| `philox_randn_sample` | `normal_kernel::` | Added | Steps 6-7 (Box-Muller of one Philox block, internal primitive) |
| `TorchRandnStream` | `normal_kernel::` | Added | Steps 6-7 (streaming source for `torch.randn(K<16)`) |
| `torch_randn_f32` | `normal_kernel::` | Added | Steps 6-7 (top-level dispatcher, mirrors ATen `normal_kernel`) |
| `TorchMt19937Engine` | `mt19937::` | Added | MT19937 rewrite (bit-exact `at::mt19937_engine`) |

**Motivation**: real CPU `torch.randn` uses `at::mt19937_engine` +
`at::normal_distribution<double>` (BSD-3-Clause), NOT `PhiloxRNGEngine.h::randn`
(the earlier "Philox is torch.randn" claim was traced by bisect
`wf_20fa0933-53d` to be wrong; upstream's own header disclaims that path).
The MT19937 rewrite (commit `b28f35e`) makes `TorchRandnStream::next_f32`
byte-exact against `torch.manual_seed(N); torch.randn(K)` for `K < 16` and
for non-contiguous tensors; `torch_randn_f32` adds the `K >= 16`
`normal_fill` scalar fast-path port (contiguous slice, in-place uniform fill
+ 16-wide `normal_fill_16` blocks + tail-recompute). See
`crates/vokra-core/src/rng/normal_kernel.rs` §"Historical note" +
`crates/vokra-core/src/rng/mt19937.rs` for the derivation.

**Interop caveat**: the Philox primitives (`philox4x32_10`, `PhiloxState`,
`TorchPhiloxState`, `philox_randn_sample`, `u32_to_uniform_f32_pytorch`)
are kept as **internal, KAT-audited primitives** — they do NOT reproduce
`torch.randn` on any real torch backend. Documented in
`normal_kernel.rs` §"`SCALE` / `u32_to_uniform_f32_pytorch` — legacy
pipeline glue" and `philox_randn_sample`'s "Not a torch.randn parity path"
section. Kept because a future CUDA `curandStatePhilox4_32_10_t` parity
path can build on the same block function once subsequence/offset packing
is settled.

**Files touched**:
- `crates/vokra-core/src/rng/mod.rs` (was `rng.rs`) — `pub use` re-exports
  + `NormalSource` trait boundary.
- `crates/vokra-core/src/rng/{mt19937,normal_kernel,philox_round,philox_state,seed_init}.rs` — new.
- `crates/vokra-core/tests/rng_{philox_kat,philox_randn,philox_state,torch_randn_cpu_parity,torch_randn_e2e,torch_seed,uniform_transform,module_layout}.rs` — new integration tests + KAT anchors.
- `crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed*.f32.bin` +
  `torch_philox_seed*.u32.bin` — pinned byte anchors regenerated on M1
  aarch64 via `tools/parity/torch_philox_dump.py`.

**Zero-dep** (NFR-DS-02): all edits inside `vokra-core` (std + core only,
no new external crates); root `Cargo.lock` unchanged.

**Related**: PR27-RNG-CROSS-ARCH audit gap — the `torch_randn_seed_42_k_100`
and `torch_randn_seed_12345_k_1000` fixture tests now apply per-arch
1-ULP tolerance for the `K >= 16` fast path, since Rust's `f32::ln` /
`f32::cos` / `f32::sin` lower to target-dependent LLVM intrinsics. See
`docs/adr/sbv2-libm-strategy.md` for the "bit-exact within Vokra on all
platforms" contract vs the "match torch bit-exact on every host"
impossibility ADR.

### 2026-08-03 — 1.0.0-rc.1-dev (GGUF `MAX_TENSOR_DIMS` raised 4 → 8 for multimodal Conv3d weights — Rust surface only, advisory)

Additive **Rust public API** + **GGUF wire semantics** entry: the loader
constant `vokra_core::gguf::tensor::MAX_TENSOR_DIMS` is raised from `4`
to `8`. Both the reader (`gguf::reader`) and both writer paths
(`GgufBuilder::add_tensor` and `GgufStreamWriter::begin`) now accept
tensors of rank ≤ 8; ranks 9+ are still rejected as
`GgufError::TooManyDimensions`. The GGUF wire format itself is uncapped
(`n_dims: u32` then `dims[n_dims]: u64`) — this constant is a Vokra-side
sanity guard, and the bump reflects the largest rank any planned model
weight uses (multimodal vision Conv3d = 5-D). The C ABI
(`include/vokra.h`, 33 fn + 11 typedef) is **untouched**
(`MAX_TENSOR_DIMS` is a Rust-side `pub const`, not cbindgen-exported;
`scripts/gen-c-abi.sh --check` = no diff).

**Motivation**: Qwen2.5-Omni's thinker subsumes the Qwen2.5-VL vision
path, whose `visual.patch_embed.proj.weight` is an `nn.Conv3d`
`[embed_dim=1280, in_channels=3, temporal_patch=2, spatial_patch=14,
spatial_patch=14]` — a 5-D tensor the previous `MAX_TENSOR_DIMS = 4`
cap rejected on the vast.ai converter path. Raising the cap unblocks
*conversion + load* of any current-day multimodal weight; forward
inference on the 5-D tensor still requires a downstream `Conv3d` op WP
(none exists in vokra-ops today, so a rank-5 tensor loads as opaque
bytes reachable via `GgufFile::tensor_data(name)` — honest per FR-EX-08).

**Interop caveat**: GGUFs Vokra emits with rank > 4 will NOT round-trip
through stock llama.cpp (its `GGML_MAX_DIMS = 4` gate rejects them).
This is acceptable because Vokra's `vokra.*` metadata prefix already
isolates its GGUFs from the llama.cpp namespace (CLAUDE.md §3).

**Files touched**:
- `crates/vokra-core/src/gguf/tensor.rs` — const bump + rustdoc.
- `crates/vokra-core/src/gguf/writer.rs` — negative tests bumped from
  `TooManyDimensions(5)` to `(9)`; two new positive round-trip tests
  (`builder_accepts_5d_conv3d_shape`, `stream_writer_accepts_5d_conv3d_shape`)
  exercise the 5-D `[2, 3, 2, 2, 2]` F32 shape end-to-end.
- `crates/vokra-core/src/gguf/reader.rs` — negative test bumped
  (n_dims = 5 → 9).
- `crates/vokra-core/src/gguf/mod.rs` — `TooManyDimensions` variant
  rustdoc updated.
- `crates/vokra-models/src/f0/rmvpe.rs` — comments referring to
  "impossible on the load path because cap = 4" updated; the RMVPE
  rank-5 rejection arm is now a real code path (RMVPE has no 5D
  weight, so a rogue converter would be loudly refused).

**Zero-dep** (NFR-DS-02): all edits inside `vokra-core` and
`vokra-models`; root `Cargo.lock` unchanged.

### 2026-07-30 — 1.0.0-rc.1-dev (FSMN-VAD backend — Rust surface only, advisory)

Additive **Rust public API** entry for the historical FSMN-VAD scaffold (FunASR
`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`) first-class
audio-dialect op posture. The C ABI (`include/vokra.h`) is **untouched**
(33 fn + 11 typedef baseline unchanged; `scripts/gen-c-abi.sh --check` =
no diff). Follows the X-Codec-2 / SBV2 precedent for new `ModelKind`
variants: **advisory Rust-surface entry**, `scripts/check-abi-changelog.sh`
does not gate on it (no C symbol changed).

> Superseded by the PR #44 correction recorded in the 2026-08-21 entry:
> the scaffold topology/schema was incompatible with the official checkpoint,
> and the weight is Apache-2.0 while the FunASR reference code is MIT. This
> block remains as the historical description of the initial prerelease API.

- **vokra-ops::fsmn_vad (new module)**: `FsmnEncoderConfig` (fields
  `n_blocks` / `input_dim` / `proj_dim` / `hidden_dim` / `lorder` /
  `rorder` / `n_class` + `upstream_default()` + `validate()` +
  `memory_kernel()`), `FsmnBlockWeights`, `FsmnVadWeights`,
  `FsmnStreamState` (`zeros()` / `reset()` / `is_zero()` / `matches()`),
  `fsmn_vad_forward()`, `softmax_last_axis()`. Distinct from the Silero
  VAD subgraph posture (FR-LD-06) — FSMN's stateless FFN + memory blocks
  lower to graph-level ops.
- **vokra-models::fsmn_vad (new module)**: `FsmnVadConfig`,
  `FsmnVadV1` (`from_gguf` / `open` / `config` / `forward_features`),
  `FsmnVadStream` (`push_features`), plus `pub const`s
  (`ARCH="fsmn-vad"`, `DEFAULT_NAME`, `CATEGORY="vad"`, `UPSTREAM_HF`,
  `KEY_*` for every `vokra.fsmn_vad.*` metadata chunk, `TENSOR_*` names,
  `tensor_ffn1_weight(i)` / etc formatters). `VadEngine` trait impl for
  `FsmnVadV1` matches the Silero `VadEngine` surface — a caller sees no
  FSMN-vs-Silero asymmetry at the trait boundary. `VadStreamHandle::push_pcm`
  returns loud `VokraError::UnsupportedOp` (FR-EX-08 — the Kaldi fbank +
  LFR + CMVN front-end pipeline lands with the real-weight parity harness
  once the checkpoint is fetched; silently zero-padding would be a fake
  data path).
- **vokra-convert::ModelKind**: `FsmnVad` variant added (public enum),
  plus 5 aliases routed to it (`fsmn-vad` / `fsmn_vad` / `fsmnvad` /
  `fsmn-vad-zh-cn-16k-common` /
  `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`). New module
  `crates/vokra-convert/src/models/fsmn_vad.rs`
  (`convert_fsmn_vad_file(input, output, license) -> Result<FsmnVadReport, _>`
  with SPDX override per the `convert_file_licensed` standing pattern).
  BF16 / F16 / F32 pass-through mirror of emotion2vec / wespeaker; full
  `vokra.fsmn_vad.*` hparam chunk group stamped unconditionally with the
  released FunASR checkpoint's fixed axes.
- **vokra-cli**: `convert --model fsmn-vad --input <safetensors> --output
  <out.gguf>` — the upstream release is `.pt`, so callers pre-flatten with
  `tools/parity/nemo_pt_to_safetensors.py` (emotion2vec / funcodec /
  wespeaker precedent).

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none.
`docs/license-audit.md` retains the 2026-07-30 yousan commercial sign-off; its
weight SPDX was corrected to Apache-2.0 on 2026-08-22 without changing that
commercial classification.

### 2026-07-30 — 1.0.0-rc.1-dev (JA-ASR bundle: hybrid CTC/attention decode + LSTM LM shallow fusion — Rust surface only, advisory)

Additive **Rust public API** entry for the M5 gap JA-ASR-3 primitive
(hybrid CTC/attention decoder + LSTM LM shallow fusion). The C ABI
(`include/vokra.h`) is **untouched** (33 fn + 11 typedef baseline
unchanged; `scripts/gen-c-abi.sh --check` = no diff). Follows the
X-Codec-2 / VoxCPM2-2B / SBV2 v2 precedent: **advisory Rust-surface
entry**, `scripts/check-abi-changelog.sh` does not gate on it (no C
symbol changed).

- **vokra-ops::hybrid_ctc_attention** (new mod):
  `hybrid_ctc_attention_decode` (fn) / `HybridCtcAttentionAttrs` /
  `HybridHypothesis` / `LstmLmCell` / `LstmLmState` / `AttnNextStepFn`
  (type alias) / `LmScoreFn` (type alias).

Runtime function (NOT an OpKind variant, same posture as `ctc_decode` /
`beam_search` — FR-OP-40 / FR-EX-10). Combines the attention beam
(caller-supplied next-step callback), the CTC prefix score (Watanabe-Hori
DP over the encoder log-prob matrix), and an optional LSTM LM shallow
fusion into a joint rank:
`(1-α) · attn_lp + α · ctc_prefix_lp + lm_weight · lm_lp`. The
`LstmLmCell` helper exposes a single-layer LSTM (matching PyTorch
`LSTMCell` gate layout `[i;f;g;o]`) so callers can wire a stateful
shallow-fusion closure without reimplementing the gate arithmetic.

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none.

### 2026-07-30 — 1.0.0-rc.1-dev (JA-ASR bundle: E-Branchformer encoder — Rust surface only, advisory)

Additive **Rust public API** entry for the M5 gap JA-ASR-4 primitive
(E-Branchformer encoder). The C ABI (`include/vokra.h`) is **untouched**
(33 fn + 11 typedef baseline unchanged; `scripts/gen-c-abi.sh --check` = no
diff). Follows the X-Codec-2 / VoxCPM2-2B / SBV2 v2 precedent: **advisory
Rust-surface entry**, `scripts/check-abi-changelog.sh` does not gate on it
(no C symbol changed).

- **vokra-ops::ebranchformer** (new mod): `EBranchformerEncoder` /
  `EBranchformerConfig` / `EBranchformerWeights` / `EBranchformerStemWeights` /
  `EBranchformerLayerWeights` / `CgMlpWeights` / `MergeWeights` plus
  `pub fn new(cfg, weights) -> Result<Self>` +
  `pub fn forward(&self, mel, mel_frames) -> Result<(Vec<f32>, usize)>` +
  accessors (`config`, `head_dim`, `cgmlp_half_dim`).

Parallel two-branch encoder — attention branch + cgMLP branch merged via
a DepthwiseConv + Linear "Merge" module (Kim et al. 2023,
[arXiv:2210.00077](https://arxiv.org/abs/2210.00077)). Primary consumer =
ESPnet OWSM family (`espnet/owsm-ctc-v3.1-1B`, CC-BY-4.0). Reuses the
Conformer primitive's `FeedForwardWeights` / `MhaWeights` /
`ConvSubsampleKind` / `PositionEncoding` layouts.

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none.

### 2026-07-30 — 1.0.0-rc.1-dev (JA-ASR bundle: Zipformer encoder — Rust surface only, advisory)

Additive **Rust public API** entry for the M5 gap JA-ASR-5 primitive
(Zipformer encoder). The C ABI (`include/vokra.h`) is **untouched** (33 fn
+ 11 typedef baseline unchanged; `scripts/gen-c-abi.sh --check` = no diff).
Follows the X-Codec-2 / VoxCPM2-2B / SBV2 v2 precedent: **advisory
Rust-surface entry**, `scripts/check-abi-changelog.sh` does not gate on it
(no C symbol changed).

- **vokra-ops::zipformer** (new mod): `ZipformerEncoder` /
  `ZipformerConfig` / `ZipformerStackDesc` / `ZipformerWeights` /
  `ZipformerStemWeights` / `ZipformerStackWeights` /
  `ZipformerLayerWeights` / `SharedMhaQkWeights` plus
  `pub fn new(cfg, weights) -> Result<Self>` +
  `pub fn forward(&self, mel, mel_frames) -> Result<(Vec<f32>, usize)>` +
  accessors (`config`, `head_dim`).

Multi-resolution encoder with attention weight sharing (single `Q` / `K`
per stack, per-layer `V` + output projection + FF + Conv + LN). Direct
port of `k2-fsa/icefall/egs/librispeech/ASR/zipformer/zipformer.py`
(Apache-2.0). Primary consumer = the reazonspeech-k2 CTC family
(`reazon-research/reazonspeech-k2-v2`, Apache-2.0). Reuses the Conformer
primitive's `FeedForwardWeights` / `ConformerConvWeights` /
`ConvSubsampleKind` / `PositionEncoding` layouts so a caller can share
stem wiring across ASR encoders.

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none.

### 2026-07-30 — 1.0.0-rc.1-dev (TitaNet-L converter + §3.1 sign-off — Rust surface only, advisory)

Additive **Rust public API** entry for the NVIDIA TitaNet-Large converter
landing (mirror of the wespeaker / ecapa_tdnn / speaker_3d skeleton
pattern). The C ABI (`include/vokra.h`) is **untouched** (33 fn + 11
typedef baseline unchanged; `scripts/gen-c-abi.sh --check` = no diff).
Follows the X-Codec-2 precedent: **advisory Rust-surface entry**,
`scripts/check-abi-changelog.sh` does not gate on it (no C symbol
changed).

- **vokra-convert::ModelKind**: `TitaNet` variant added (public enum),
  plus 5 aliases routed to it (`titanet-large` / `titanet_large` /
  `titanet` / `speakerverification_en_titanet_large` /
  `nvidia/speakerverification_en_titanet_large`). New module
  `crates/vokra-convert/src/models/titanet.rs`
  (`convert_titanet_file(input, output, license: Option<&str>) ->
  Result<TitaNetReport, ConvertError>` + `TitaNetReport` struct + arch /
  category / attribution constants). Category = `speaker` (mirror of
  the sibling `wespeaker` / `ecapa_tdnn` / `speaker_3d` GGUF layout).
  Default provenance stamp = **cc-by-4.0** (`LicenseClass::AttributionRequired`
  — the converter additionally writes the FR-MD-09
  `vokra.provenance.attribution` chunk with the NVIDIA credit text;
  runtime `vokra_core::resolve_attribution` / `vokra_model_attribution`
  already surface it, no new C ABI needed).
- **vokra-cli**: `convert --model titanet-large --input <safetensors>
  --output <gguf> [--license <spdx>]` — the license override at the
  outer `convert_file --license` boundary flips the class back to
  Permissive and drops the attribution chunk (silent inheritance of
  NVIDIA credit into a permissive retrain would misattribute — pinned
  by `license_override_to_permissive_drops_attribution_chunk` test).
- **vokra-core::m5_residual_ops**: `TITANET_SPEAKER_ENCODE_OP` blocker
  text refreshed from "NVIDIA NC unconfirmed" to "already covers
  speaker embedding" (semantic-only edit; the const value +
  `M5ResidualAnchor` catalogue entry are unchanged, `MinDtypeRegistry`
  reservation still holds — `new_anchors_are_reserved_but_unregistered`
  passes unchanged).

Owner sign-off = ☑ Commercial 2026-07-30 yousan
(`docs/license-audit.md` §3.1 row 262; primary source = HF
`nvidia/speakerverification_en_titanet_large` cardData YAML frontmatter
`license: cc-by-4.0` + card body citation, 2026-07-30 fetch). NOTICE
§11 records the code-level NVIDIA credit.

Runtime port is **out-of-scope** for the converter landing — the
`TITANET_SPEAKER_ENCODE_OP` op is M5-residual (CAM++ already covers the
speaker-embedding surface under Apache-2.0 with no attribution
overhead); a future M5 landing would be a backward-compatible additive
per the M4-20 T14 mechanism-anchor discipline.

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none.

### 2026-07-30 — 1.0.0-rc.1-dev (M5-16 / FR-OP-83: FCPE real Conformer forward + converter — Rust surface only, advisory)

Additive **Rust public API** entry promoting `vokra-models::f0::fcpe` from the
2026-07-25 SoTA Wave-F skeleton (`Fcpe::{from_gguf, extract}` on a metadata-
only surface) to a real Conformer-based F0 forward: mel[T, n_mels] → Linear
stem → `vokra_ops::conformer::ConformerEncoder` (SoTA Phase 2 landed
primitive — no new op) → LayerNorm → Linear head → softmax → cent-grid
soft-argmax → Hz + V/UV. Follows the X-Codec-2 / SBV2 precedent: **advisory
Rust-surface entry**, `scripts/check-abi-changelog.sh` does not gate on it
(no C symbol changed).

- **vokra-models::f0::fcpe** — `FcpeConfig` + `FcpeWeights` types added
  (`pub struct` with public fields; the fields are the Conformer + head
  shape descriptors so a downstream that wants to introspect a bound FCPE
  can walk them without another Rust round-trip). `FCPE::from_gguf` now
  binds the canonical tensor set when present (loud on partial /
  mis-shaped sets — FR-EX-08 posture); metadata-only GGUFs continue to
  return the frame-count-contract skeleton (backward-compat with the
  Wave-F consumers). New associated fn `FCPE::has_real_weights() -> bool`
  and `FCPE::config() -> &FcpeConfig` expose the state to callers /
  tests.
- **vokra-convert::ModelKind** — `Fcpe` variant added (public enum) plus
  5 aliases (`fcpe`, `torchfcpe`, `fast-context-pitch-estimator`,
  `fast_context_pitch_estimator`, `cnchtu/fcpe`). New module
  `crates/vokra-convert/src/models/fcpe.rs` (`convert_fcpe_file` +
  internal `models::fcpe::convert(bytes)` helper shared with
  `convert_file_licensed`) — F32 / F16 / BF16 pass-through, `vokra.model.
  arch = "fcpe"` + `vokra.model.category = "f0"` + `vokra.provenance.
  upstream_hf = "CNChTu/FCPE"` + `mit` Permissive stamp.
- **GGUF metadata schema** — new `vokra.f0.fcpe.*` config chunk group
  (13 keys: `hop` u32 / `fmin` f32 / `fmax` f32 / `sample_rate` u32 /
  `n_mels` u32 / `n_fft` u32 / `n_pitch_bins` u32 /
  `confidence_threshold` f32 / `d_model` u32 / `n_heads` u32 /
  `ffn_dim` u32 / `n_layers` u32 / `kernel_size` u32) read by
  `FcpeConfig::from_gguf` with per-key defaults from the FCPE_v001
  primary source. Additive — every key defaults if absent, so a
  metadata-only GGUF still loads.
- **License** — `docs/license-audit.md` §3.1 sign-off row added
  (`CNChTu/FCPE`, MIT Permissive, 2026-07-30 yousan =
  ☑ Commercial; primary source `github.com/CNChTu/FCPE/LICENSE`).

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. `scripts/gen-c-abi.sh --check` =
no diff.

### 2026-07-30 — 1.0.0-rc.1-dev (M5 gap CC wave 2 — VoxCPM2-2B config + scaffolds — Rust surface only, advisory)

Additive **Rust public API** entry for the M5 owner-checklist CC-side wave 2
(A-4/A-5/A-6/B-1/B-2 ultracode workflow, PR #24 commit `b13a3c0`). The C ABI
(`include/vokra.h`) is **untouched** (33 fn + 11 typedef baseline unchanged;
`scripts/gen-c-abi.sh --check` = no diff). Follows the X-Codec-2 precedent:
**advisory Rust-surface entry**, `scripts/check-abi-changelog.sh` does not
gate on it (no C symbol changed).

- **vokra-ops::vae_continuous**: `voxcpm2_2b()` associated fn added on
  `ContinuousVaeConfig` (sibling of the existing `voxcpm_0_5b()` factory).
  Returns the primary-source-anchored 2B config for `openbmb/VoxCPM2`
  (Apache-2.0). Owner Q1 topology ADR is still Proposed — this only lands
  the config primitives and the accompanying `voxcpm2_2b_config_matches_primary_source`
  / `voxcpm2_2b_config_validates` pin tests; the 2B tensor-name mapping in
  `vokra-convert::models` is deferred until the ADR is Accepted.

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none. Commit:
`b13a3c0` (2026-07-30).

### 2026-07-28 — 1.0.0-rc.1-dev (X-Codec-2 converter + LicenseClass flip — Rust surface only, advisory)

Additive **Rust public API** change plus one observable behaviour change on
`LicenseClass::from_license_str` / `registry_lookup` (semantics only — no
signature change). The C ABI (`include/vokra.h`) is **untouched** (33 fn + 11
typedef baseline unchanged; `scripts/gen-c-abi.sh --check` = no diff). Follows
the SBV2 precedent above for new `ModelKind` variants: **advisory Rust-surface
entry**, `scripts/check-abi-changelog.sh` does not gate on it (no C symbol
changed).

- **vokra-convert::ModelKind**: `XCodec2` variant added (public enum), plus 6
  aliases routed to it (`xcodec2` / `x-codec-2` / `x_codec_2` / `xcodec-2` /
  `x-codec2` / `hkustaudio-xcodec2`). New module
  `crates/vokra-convert/src/models/xcodec2.rs` (`convert_xcodec2_file` +
  internal `models::xcodec2::convert(bytes)` helper shared with
  `convert_file_licensed`).
- **vokra-core::compliance::license_class**: `x-codec-2` / `xcodec2`
  registry entries move from `LicenseClass::Permissive` → `LicenseClass::NonCommercial`
  (grouped with `f5-tts` / `encodec`). This is an **observable behaviour
  change** on the public `registry_lookup` / `from_license_str` predicates
  — downstream code that matched on the returned `LicenseClass` will now
  see `NonCommercial` instead of `Permissive`, which flips
  `requires_research_flag`, `commercial_ok`, and `redistributable` for the
  two slugs. Rationale: HF `HKUSTAudio/xcodec2` cardData
  `license: cc-by-nc-4.0` (primary source 2026-07-15 / re-verify 2026-07-28)
  is the **weight** authoritative source, and the M2-13 runtime gate + M4-04
  publish gate both classify by weight (§3.1 sign-off row 254 records the
  Research-only sign-off). Pin tests updated: `dac` / `wavtokenizer` stay
  `Permissive`; a new `NonCommercial` pin covers `x-codec-2` / `xcodec2`
  with case-insensitive spellings and asserts every derived predicate.
- **vokra-cli**: `convert --model xcodec2 --license <spdx>` flag added on
  the generic fallthrough dispatch (mutually exclusive with `--quantize` /
  `--policy-preset`; loud rejection if combined — silently ignoring a user
  flag is FR-EX-08).

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc
baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none. Commit:
`53fa432` (2026-07-28).

### 2026-07-26: SBV2 v2 + BERT DeBERTa v2/v3 addition (Rust surface only, advisory)

- **vokra-bert (new crate)**: `SbertTokenizer`, `DebertaV2Encoder`, `DebertaV3Encoder`, `BertEncoder` trait
- **vokra-models::sbv2 (new module)**: `SbV2Model`, `Language`, `SbV2Phonemizer`, `SbV2SynthRequest`, all supporting types
- **vokra-convert::ModelKind**: `SbV2`, `DebertaV2`, `DebertaV3` variants added

All additions are **Rust surface only** — no new C ABI symbols. v1.0-rc baseline (33 fn + 11 typedef) unchanged. gen-c-abi drift = none.

### 2026-07-24 — 1.0.0-rc.1-dev (SoTA Phase 1: HiFTNet vocoder primitives + NSF module — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
**untouched** (`scripts/gen-c-abi.sh --check` = no diff; a grep for `hiftnet` /
`nsf` / `snake` in the header matches **0** new symbols). No GGUF metadata
schema is added at the Vokra prefix level for this Phase 1 slice; the models
that consume these primitives (CosyVoice2 HiFTGenerator chain, and the
scaffolds for Dia-1.6B / Zonos-v0.1) reuse the existing
`vokra.cosyvoice2.*` / model-specific chunks that were already recorded in
the v0.9 baseline. Phase 1 wires the HiFTNet (Neural Source Filter +
iSTFTNet) vocoder as the correct upstream chain for CosyVoice2
(`cosyvoice/hifigan/generator.py:378 class HiFTGenerator`, arXiv:2412.10117)
— superseding the previously-scaffolded `mimi_bridge.rs` which was based on
the 2026-07-22 corrected SSOT (SoTA plan §1(a): CosyVoice2 does **not** use
Mimi; Mimi is Moshi/CSM-only).

M5-13 relevance (why this is recorded here): all items are additive **Rust**
public items with **no C surface**, so `scripts/check-abi-changelog.sh` does
not gate on this entry (no C symbol changed). `scripts/rust-public-api-list.sh`
picks them up (`vokra-ops::nsf::*` and `vokra-ops::hiftnet::*`); as with the
M5-01/02/03/05/06 entries below, the
`docs/abi/vokra-rust-public-api.v1.0-rc.list` snapshot is **not** rotated by
this PR — snapshot rotation is the M5-13/IF-01 freeze owner's action. All
items are additive (existing signatures unchanged; `NsfEntropy` is
`#[non_exhaustive]` for future extension), Breaking? = no.

New model scaffolds landed under this PR (`vokra-models::dia`,
`vokra-models::zonos`, and the corresponding converters `vokra-convert::models::dia`
/ `vokra-convert::models::zonos`) are excluded from the
`rust-public-api-list.sh` scan surface (scan crates =
`vokra-core` / `vokra-ops` / `vokra-capi` only — same posture as
`vokra-models` scaffolds landed in prior WPs). CosyVoice2 wiring
(`vokra-models::cosyvoice2::hift_chain`) is likewise `vokra-models`
internal, not part of the snapshot. Real-weight parity harnesses (Dia /
Zonos / real CosyVoice2 checkpoint) land here as flip-the-switch skeletons;
they do not add public surface until owner provides the checkpoints
(`docs/m4-owner-verification-checklist.md`).

| Crate / area              | Symbol                                                                                                                       | Kind  | Signature / note                                                                                                                                    | Rationale                                                                                                | Breaking? | PR    |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------- | ----- |
| `vokra-ops::nsf` (new mod) | `SineGen` / `SineGenConfig` / `SineGenOutput`                                                                                | Added | `pub struct` + `pub fn new(cfg) -> Self` + `pub fn forward(&self, f0: &[f32], entropy: NsfEntropy) -> Result<SineGenOutput>`                          | Sine oscillator front-end for NSF (HiFTNet upstream; CosyVoice2 arXiv:2412.10117 §Vocoder), SoTA Phase 1-2 wave 1 | no        | (TBD) |
| `vokra-ops::nsf`          | `SourceModuleHnNSF` / `SourceModuleHnNSFConfig` / `SourceModuleHnNSFWeights` / `SourceModuleHnNSFOutput`                     | Added | `pub struct` + `pub fn new(cfg, weights) -> Result<Self>` + `pub fn forward(&self, f0, entropy) -> Result<SourceModuleHnNSFOutput>`                   | Harmonic-plus-noise Neural Source Filter (paired with HiFTGenerator), SoTA Phase 1-2 wave 1              | no        | (TBD) |
| `vokra-ops::nsf`          | `NsfEntropy`                                                                                                                 | Added | `pub enum NsfEntropy { Deterministic, /* future variants */ }` (`#[non_exhaustive]`)                                                                  | entropy source knob for NSF noise term (owner-selectable at higher layers), SoTA Phase 1-2               | no        | (TBD) |
| `vokra-ops::hiftnet` (new mod) | `F0Predictor` / `F0PredictorConfig` / `F0PredictorWeights`                                                              | Added | `pub struct` + `pub fn new(cfg, weights) -> Result<Self>` + `pub fn forward(&self, mel, t_mel) -> Result<Vec<f32>>`                                    | F0 predictor (Conv1d+ELU stack + Linear head) driving NSF from mel, SoTA Phase 1-2 wave 2                | no        | (TBD) |
| `vokra-ops::hiftnet`      | `Snake`                                                                                                                      | Added | `pub struct Snake { alpha: Vec<f32>, alpha_logscale: bool }` + `pub fn new(alpha, alpha_logscale) -> Result<Self>` + `pub fn forward_in_place(x, channels, time) -> Result<()>` | Snake activation (BigVGAN/HiFTNet), per-channel alpha, SoTA Phase 1-2 wave 3a                            | no        | (TBD) |
| `vokra-ops::hiftnet`      | `ResBlock` / `ResBlockConfig` / `ResBlockWeights`                                                                            | Added | `pub struct` + `pub fn new(cfg, weights) -> Result<Self>` + `pub fn forward_in_place(x, t) -> Result<()>`                                             | HiFTNet MRF residual block with dilated Conv1d, SoTA Phase 1-2 wave 3b                                   | no        | (TBD) |
| `vokra-ops::hiftnet`      | `HiFTGenerator` / `HiFTGeneratorConfig` / `HiFTGeneratorWeights`                                                             | Added | `pub struct` + `pub fn new(cfg, weights) -> Result<Self>` + `pub fn forward(&self, mel, t) -> Result<Vec<f32>>` + accessors (`config`, `num_kernels`, `num_upsamples`, `output_channels_at`, `total_upsample_factor`) | HiFTNet generator (NSF + iSTFTNet vocoder head, Snake) for CosyVoice2 mel → PCM, SoTA Phase 1-2 wave 3c  | no        | (TBD) |
| `vokra-ops::hiftnet`      | `f0_predictor_forward` (module fn)                                                                                            | Added | `pub fn f0_predictor_forward(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>>` (on `F0Predictor`)                                                | direct-use convenience for CosyVoice2 chain wiring, SoTA Phase 1-2 wave 2                                | no        | (TBD) |

### 2026-07-24 — 1.0.0-rc.1-dev (SoTA Phase 2/3/4 + JA: ASR/TTS primitives + models — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
**untouched** (`scripts/check-abi-changelog.sh` = clean; a grep for any of the
new module names in the header matches **0** new symbols). The GGUF metadata
schema is extended **only under model-specific chunks** (`vokra.parakeet.*`,
`vokra.canary.*`, `vokra.distil_whisper.*`, `vokra.kotoba_whisper.*`,
`vokra.chatterbox.*`, `vokra.qwen3_tts.*`, `vokra.cosyvoice3.*`,
`vokra.voxcpm.*`, `vokra.voxcpm2.*`, `vokra.vibevoice.*`, `vokra.irodori.*`,
`vokra.vits_ja.*`) — these are additive under §"Scope: what belongs in this
file" (recorded here in the GGUF metadata additions block at the bottom of
this file for M5-13 rollup) and never rename existing keys.

Consolidates four in-progress branches that landed additive vokra-ops
primitives + model scaffolds on the current SoTA plan implementation branch
`feat/sota-phase1-2026-07-23`:

- **SoTA Phase 2** (ASR trigger models: Parakeet-TDT/CTC, Canary-1B-v2,
  Kyutai-STT, OmniASR-CTC, Distil-Large-v3.5) — adds three vokra-ops
  primitives (`conformer`, `rnnt_decode`, `ctc_decode`) plus six model
  scaffolds under `vokra-models::{parakeet,parakeet_ctc,canary,kyutai_stt,
  omniasr_ctc,distil_whisper}` and matching converters. FR-OP-40 /
  FR-OP-41 / FR-OP-42.
- **SoTA Phase 3** (TTS trigger models: FunAudioLLM-CosyVoice3-0.5B,
  Chatterbox-Multilingual/Turbo/Nano, Qwen3-TTS-0.6B) — adds three
  vokra-ops primitives (`bigvgan_generator`, `snac_decode`,
  `qwen3_tts_codec`) plus five model scaffolds under
  `vokra-models::{cosyvoice3,chatterbox,chatterbox_turbo,chatterbox_nano,
  qwen3_tts}` and matching converters. FR-OP-11 (BigVGAN) + FR-OP-35 (SNAC
  as an FSQ-adjacent codec landing on the shared `codebook_lookup` seam)
  + FR-OP-36 (Qwen3-TTS 16-quantizer RVQ, semantic + acoustic split at
  12.5 Hz).
- **SoTA Phase 4** (long-form TTS: Microsoft VoxCPM2 + Microsoft
  VibeVoice-1.5B) — adds two vokra-ops primitives (`vae_continuous`
  shared between VoxCPM2 and VibeVoice per the vae_continuous rustdoc;
  `ddpm_sampler` new to VibeVoice, distinct axes from `flow_sampler` per
  ADR M3-05 §D4 = v-prediction + cosine β schedule) plus two model
  scaffolds under `vokra-models::{voxcpm2,vibevoice}` and matching
  converters. FR-OP-30 / FR-EX-10.
- **JA (Japanese-first ASR/TTS)** — adds one vokra-ops primitive
  (`waveform_frontend` = 7-layer strided Conv1d over raw PCM, used by
  Kotoba-Whisper distilled encoders that skip the mel step) plus three
  model scaffolds under `vokra-models::{kotoba_whisper,irodori,vits_ja}`
  and matching converters, plus the eval crate's language axis
  (`vokra-eval::lang`; CER as the JA primary metric). JA-ASR-0/1/2 +
  JA-TTS-1/2.

M5-13 relevance: this WP is additive **Rust** public items only with **no
C surface**, so `scripts/check-abi-changelog.sh` does not gate on this
entry (no C symbol changed). `scripts/rust-public-api-list.sh` picks all
128 new symbols up (9 new modules under `vokra-ops`; 7 new enums; 33 new
structs; 68 new fns; plus 9 re-exports at `vokra-ops::lib`; plus 1 type
alias); as with all other pre-1.0-rc entries in this file, the
`docs/abi/vokra-rust-public-api.v1.0-rc.list` snapshot **is** rotated by
this PR (following the Phase 1 pattern at commit `351dc42`) — but the
rotation is a mechanical restatement of the additive surface and does
not fire IF-01 (freeze is the M5-13/v1.0 GA owner action). All items are
additive (existing signatures unchanged; every new public enum is
`#[non_exhaustive]`), Breaking? = no.

Model scaffolds landed under this PR (`vokra-models::{parakeet,
parakeet_ctc, canary, kyutai_stt, omniasr_ctc, distil_whisper,
cosyvoice3, chatterbox, chatterbox_turbo, chatterbox_nano, qwen3_tts,
voxcpm2, vibevoice, kotoba_whisper, irodori, vits_ja}`) are **excluded**
from the `rust-public-api-list.sh` scan surface (scan crates =
`vokra-core` / `vokra-ops` / `vokra-capi` only — same posture as prior
`vokra-models` scaffolds landed in the SoTA Phase 1 entry above and in
prior WPs). Real-weight parity harnesses land as flip-the-switch
skeletons; they do not add public surface until owner provides the
checkpoints (`docs/m4-owner-verification-checklist.md`).

Adversarial audit gaps landed as tests only (`feat(sota-audit)` +
`test(sota-audit)`): `vokra-convert::lib` gained a
`modelkind_alias_and_roundtrip_tests` module (~100 alias→variant
assertions across all Phase 2-5 families + Whisper 2 aliases +
denoise/utmos canonical spellings), and
`vokra-core::compliance::license_class` gained coverage for the
`Copyleft` / `RedistributionForbidden` / `ConditionalCommercial`
variants added on 2026-07-23. These are internal test items and do not
show up in the Rust public-API snapshot.

| Crate / area                          | Symbol                                                                                                                                                                                                                                                                                                            | Kind  | Signature / note                                                                                                                                                                                                                                                                                | Rationale                                                                                                                              | Breaking? | PR    |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------- | ----- |
| `vokra-ops::conformer` (new mod)      | `ConformerEncoder` / `ConformerConfig` / `ConformerWeights` / `ConformerLayerWeights` / `ConformerConvWeights` / `ConformerSubsampleWeights` / `FeedForwardWeights` / `MhaWeights` / `ConvSubsampleKind` / `PositionEncoding`                                                                                     | Added | `pub struct` + `pub fn new(cfg, weights) -> Result<Self>` + `pub fn forward(&self, mel, mel_frames) -> Result<(Vec<f32>, usize)>` + accessors (`config`, `factor`, `has_norm`, `head_dim`, `projection_in_dim`); two `#[non_exhaustive]` enums for subsample kind + position encoding.           | NeMo Conformer/FastConformer encoder primitive for Parakeet + Canary + OmniASR-CTC, SoTA Phase 2 wave 1                                | no        | (TBD) |
| `vokra-ops::rnnt_decode` (new mod)    | `rnnt_decode` (fn) / `RnntAttrs` / `RnntHypothesis` / `RnntDecoderKind`                                                                                                                                                                                                                                           | Added | `pub fn rnnt_decode(logits, attrs) -> Result<Vec<RnntHypothesis>>` + attrs struct + hypothesis struct + `#[non_exhaustive]` enum `{Greedy, Beam, Tdt, LabelLooping}` (label-looping stubbed; TDT active)                                                                                        | RNN-T decoder primitive (greedy + beam + TDT) for Parakeet-TDT-0.6B-v3, SoTA Phase 2 wave 1                                            | no        | (TBD) |
| `vokra-ops::ctc_decode` (new mod)     | `ctc_decode_greedy` (fn) / `ctc_decode_beam` (fn) / `CtcBeamAttrs` / `CtcHypothesis` / `CtcBeamAttrs::plain` (ctor)                                                                                                                                                                                                | Added | `pub fn ctc_decode_greedy(logits, ...) -> Result<Vec<u32>>` + `pub fn ctc_decode_beam(logits, attrs) -> Result<Vec<CtcHypothesis>>` + attrs (blank_id + beam_width + optional n-gram LM fusion path + hotword boost) + hypothesis struct + `plain` ctor                                          | CTC decoder primitive (greedy blank-fold + prefix beam search with LM fusion + hotword boost) for OmniASR-CTC-1B + Parakeet-CTC-1.1B, SoTA Phase 2 wave 1 | no        | (TBD) |
| `vokra-ops::bigvgan_generator` (new mod) | `BigVGanGenerator` / `BigVGanConfig` / `BigVGanWeights` / `AmpBlock1` / `AmpBlock1Weights` / `SnakeBeta` / `SnakeKind`                                                                                                                                                                                            | Added | `pub struct` + `pub fn new(cfg, weights) -> Result<Self>` + `pub fn forward(&self, mel, t_mel) -> Result<Vec<f32>>` + accessors + `#[non_exhaustive]` enum for Snake vs SnakeBeta variant                                                                                                       | anti-aliased AMPBlock1 + Snake/SnakeBeta + tanh terminal for Chatterbox family + downstream vocoder-sharing models, SoTA Phase 3 wave 1 | no        | (TBD) |
| `vokra-ops::snac_decode` (new mod)    | `SnacDecoder` / `SnacConfig` / `SnacWeights`                                                                                                                                                                                                                                                                       | Added | `pub struct` + `pub fn new(cfg, weights) -> Result<Self>` + decode fn (multi-scale)                                                                                                                                                                                                              | Multi-Scale Neural Audio Codec 3-stage RVQ decode for Chatterbox family, SoTA Phase 3 wave 1                                            | no        | (TBD) |
| `vokra-ops::qwen3_tts_codec` (new mod) | `Qwen3TtsCodec` / `Qwen3TtsCodecConfig` / `qwen3_tts_codec_decode` (fn) + accessors (`config`, `decode`, `frame_rate_hz`, `new`, `quantizer_vocab_size`)                                                                                                                                                          | Added | `pub struct` + `pub fn new(cfg, weights: Vec<CodebookTable>) -> Result<Self>` + `pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>>` (16-quantizer @ 12.5 Hz, semantic + acoustic split)                                                                                                | Qwen3-TTS RVQ code→feature decode primitive for Qwen3-TTS-0.6B, SoTA Phase 3 wave 1                                                     | no        | (TBD) |
| `vokra-ops::vae_continuous` (new mod) | `ContinuousVaeEncoder` / `ContinuousVaeDecoder` / `ContinuousVaeConfig` / `ContinuousVaeEncoderWeights` / `ContinuousVaeDecoderWeights` / `continuous_vae_encode` (fn) / `continuous_vae_decode` (fn)                                                                                                              | Added | `pub struct` encoder/decoder + `pub fn new(cfg, weights) -> Result<Self>` + module-level fns for one-shot encode/decode                                                                                                                                                                            | continuous VAE (encoder/decoder) shared between VoxCPM2 and VibeVoice, SoTA Phase 4 wave 1                                              | no        | (TBD) |
| `vokra-ops::ddpm_sampler` (new mod)   | `ddpm_sample` (fn) / `DdpmSamplerConfig` / `BetaSchedule` / `PredictionType` / `build_alphas_cumprod` (fn) / `pick_inference_timesteps` (fn) / `DdpmSamplerConfig::vibevoice_defaults` (ctor)                                                                                                                     | Added | `pub fn ddpm_sample<F>(f, cfg) -> Result<Vec<f32>>` + config (`num_train_timesteps` + `num_inference_timesteps` + `prediction_type` + `beta_schedule`) + `#[non_exhaustive]` enums `PredictionType {Epsilon, VPrediction}` and `BetaSchedule {Linear, Cosine, ScaledLinear}` + helper fns + preset | DDPM sampler for VibeVoice-1.5B (v-prediction + cosine β schedule — the two axes flow_sampler cannot express per ADR M3-05 §D4), SoTA Phase 4 wave 1 | no        | (TBD) |
| `vokra-ops::waveform_frontend` (new mod) | `waveform_frontend` (fn) / `WaveformFrontendAttrs` / `WaveformFrontendWeights` / `ConvLayerAttrs` / `ConvLayerWeights` / `Norm`                                                                                                                                                                                    | Added | `pub fn waveform_frontend(pcm, attrs, weights) -> Result<Vec<f32>>` + attrs (7-layer strided Conv1d chain over raw 16 kHz PCM) + `#[non_exhaustive]` enum `Norm {None, LayerNorm, GroupNorm}` for per-layer normalisation                                                                        | 7-layer strided conv frontend replacing mel for Kotoba-Whisper distilled encoders, SoTA plan JA-ASR-1                                    | no        | (TBD) |
| `vokra-ops::lib` re-exports           | (9 new `pub use` re-exports for the modules above — one per module + `Norm` alias)                                                                                                                                                                                                                                | Added | flat re-export block per the parallel-wave rebase pattern used throughout `vokra-ops::lib`                                                                                                                                                                                                          | ergonomic top-level access mirroring existing ops (flow_sampler, mimi_rvq, etc.), SoTA Phase 2/3/4 + JA                                  | no        | (TBD) |

### 2026-07-25 — 1.0.0-rc.1-dev (Wave E BF16 pass-through fleet + Wave F audio primitives + CI fix wave — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
**untouched** (33 fn + 11 typedef, baseline unbroken; `scripts/gen-c-abi.sh --check`
= no diff). No new C symbol is added by any of the three waves this day. The
Rust public-API snapshot (`docs/abi/vokra-rust-public-api.v1.0-rc.list`) is
rotated by mechanical restatement per the Phase 1 pattern (commit `351dc42`
precedent) — **NOT an IF-01 firing**; the freeze commitment remains M5-13 /
v1.0 GA tag / owner action. Anchor files (`docs/abi/vokra.h.*.symbols` /
`docs/abi/vokra-rust-public-api.*.list`) are **not modified by this WP** —
snapshot rotation is the M5-13/IF-01 freeze owner's action.

**Wave E — BF16 pass-through converter fleet** (2026-07-23〜25): 15 new
converters emit BF16 verbatim via `GgufBuilder` (`GgmlType::BF16`, GGUF type
30 — previously landed for Moshi under the 2026-07-15 M4-06 entry) + 20
existing converters extended with a BF16 arm. All changes are inside
`crates/vokra-convert/src/models/*.rs` module-private paths + config chunks;
**no new public functions in the `vokra-convert` `pub` surface**, no new GGUF
metadata keys, no shape/layout change to any existing tensor. Existing F32 /
F16 pathways are byte-for-byte unchanged.

**Wave F — audio primitives + WhisperX-style long-form orchestrator**
(2026-07-24〜25): `vokra-models::f0::{rmvpe, fcpe, crepe}` is a new `pub mod f0`
with `F0Frame`, `LoadError`, and the three extractor types (`Rmvpe`, `Fcpe`,
`Crepe`), each exposing `from_gguf` / `extract` (Rustdoc marked **SKELETON** —
construction and I/O contract only; real inference lands in a follow-up WP
alongside the first F0-consuming model). `vokra-models::align::{ctc_segmentation, charsiu}`
lands the align-op skeleton plus a **full `ctc_segmentation` Viterbi
implementation** (non-skeleton; ready for the first alignment consumer). The
WhisperX-style native long-form orchestrator lives at
`integrations/vokra-server/src/longform.rs` (+928 lines) — this is
**server-side only** (isolated `integrations/vokra-server` workspace,
"Out-of-scope" per the top of this file) and is not part of the core ABI
surface; noted here only to explain why the core Rust-snapshot delta this
day is core-only despite the large diff.

**New crate — `vokra-kws-micro`** (Wave F sibling): `#![no_std]`(+`alloc`),
`publish = false`, identical posture to `vokra-vad-micro` (2026-07-21 M5-03
entry). Not part of the public API snapshot (`scripts/rust-public-api-list.sh`
scans only `vokra-core` / `vokra-ops` / `vokra-capi`); recorded here so the
M5-13 owner has a complete crate inventory before the freeze. Root
`Cargo.lock` remains `vokra-*` only (NFR-DS-02).

**CI fix wave** (2026-07-25, land commits from the ultracode fix workflow —
see PR #20): repo-hygiene YAML heredoc syntax fix (`SNAPSHOT_STDOUT="$(python -
<<'PY' ... PY )"` → `python -c "..."`) / license — FunCodec (Alibaba DAMO,
MIT) is a **separate codec from Meta EnCodec** (CC-BY-NC 4.0), but the
upstream slug embeds the literal substring "encodec"; owner decision
2026-07-25 was to add an **explicit SLUG_ALLOWLIST to `scripts/compliance/
check-encodec-exclusion.sh`** (transparency preserved for audit) rather than
`concat!`-split the source (subagent's alternative rejected as
"indistinguishable from defeating the audit control"); `funcodec.rs` stays
byte-exact so `vokra.provenance.upstream_hf` cross-checks byte-for-byte
against upstream / zonos parity `rotary_emb_interleaved` type alignment.
**No public API changes** — recorded here for the M4-12 rc baseline-snapshot
completeness (CI-tooling deltas the gate scripts depend on).

M5-13 relevance (why this is recorded here): all items are additive **Rust**
public items, module-private converter deltas, or CI tooling deltas with **no
C surface**, so `scripts/check-abi-changelog.sh` does not gate on this entry
(no C symbol changed). `scripts/rust-public-api-list.sh` picks up the new
`f0` and `align` modules; the new `vokra-kws-micro` crate is out of its scan
set. Rotation of the snapshot itself remains an owner action at M5-13.

| Crate / area                            | Symbol                                            | Kind  | Signature / note                                                                                                                                       | Rationale                                                                                | Breaking? | PR    |
| --------------------------------------- | ------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | --------- | ----- |
| `vokra-models::align` (new module)      | `pub mod align`                                   | Added | `pub mod align` (contains `ctc_segmentation`, `charsiu` submodules)                                                                                     | forced-alignment / segmentation op family, Wave F                                          | no        | (TBD) |
| `vokra-models::align::charsiu`          | Charsiu-style forced-alignment                    | Added | forced-alignment types **SKELETON** (Rustdoc-marked)                                                                                                    | forced alignment secondary path, Wave F                                                    | no        | (TBD) |
| `vokra-models::align::ctc_segmentation` | CTC-segmentation Viterbi decoder                  | Added | full CTC-segmentation Viterbi implementation (non-skeleton; ready for first consumer)                                                                   | forced alignment via CTC (FR-OP-40 sibling), Wave F                                        | no        | (TBD) |
| `vokra-models::f0` (new module)         | `pub mod f0`                                      | Added | `pub mod f0` (contains `rmvpe`, `fcpe`, `crepe` submodules)                                                                                             | F0 / pitch extraction op family (FR-OP-83), Wave F skeleton                                | no        | (TBD) |
| `vokra-models::f0`                      | `F0Frame`                                         | Added | `pub struct F0Frame { … }` (per-frame f0 + confidence)                                                                                                  | shared F0 output frame across RMVPE / FCPE / CREPE, Wave F                                 | no        | (TBD) |
| `vokra-models::f0`                      | `LoadError`                                       | Added | `pub enum LoadError` (GGUF load failure modes)                                                                                                          | shared error kind for the three extractor `from_gguf` entries, Wave F                      | no        | (TBD) |
| `vokra-models::f0::crepe`               | `Crepe::{from_gguf, extract}`                     | Added | `pub fn from_gguf(&GgufFile) -> Result<Self, LoadError>` / `pub fn extract(&self, pcm: &[f32]) -> Result<Vec<F0Frame>>`                                 | CREPE extractor **SKELETON**; real inference in follow-up                                  | no        | (TBD) |
| `vokra-models::f0::fcpe`                | `Fcpe::{from_gguf, extract}`                      | Added | `pub fn from_gguf(&GgufFile) -> Result<Self, LoadError>` / `pub fn extract(&self, pcm: &[f32]) -> Result<Vec<F0Frame>>`                                 | FCPE extractor **SKELETON**; real inference in follow-up                                   | no        | (TBD) |
| `vokra-models::f0::rmvpe`               | `Rmvpe::{from_gguf, extract}`                     | Added | `pub fn from_gguf(&GgufFile) -> Result<Self, LoadError>` / `pub fn extract(&self, pcm: &[f32]) -> Result<Vec<F0Frame>>`                                 | RMVPE extractor **SKELETON**; real inference in follow-up                                  | no        | (TBD) |
| `vokra-kws-micro` (new crate)           | crate scaffold                                    | Added | `#![no_std]`(+`alloc`), `publish = false`, `vokra-core` dep only; not in public API snapshot                                                            | no_std KWS scaffold for IoT tiers, sibling of `vokra-vad-micro`                            | no        | (TBD) |
| `vokra-convert` (module-private)        | Wave E BF16 pass-through — 35 converters          | Added | 15 new converters + 20 existing extended with a BF16 arm via `GgufBuilder` (`GgmlType::BF16`, GGUF type 30); no new `pub` fn, no new GGUF keys           | preserve BF16 upstream fidelity end-to-end (no F32 promotion at conversion), Wave E        | no        | (TBD) |

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

### 2026-07-23 — 1.0.0-rc.1-dev #3 (HF publication: restamp_provenance — Rust surface only)

Additive **Rust public API** only; C ABI untouched.

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-convert` | `restamp_provenance` | Added | `pub fn restamp_provenance(input, output, license: &str, model_id: &str, source: &str, attribution: Option<&str>) -> Result<ConvertSummary, ConvertError>` | Rewrite `vokra.provenance.*` on an existing GGUF **without re-materialising tensors** — mmap the input, copy each payload straight to a `GgufStreamWriter`. Peak memory is one tensor payload, not the whole file. Publishes 8.7 GiB Voxtral on a 16 GiB host (measured: peak footprint 6.4 MB) where full re-conversion needs >16 GiB RAM. Also rescues every pre-provenance cache GGUF without re-running its converter. | no | (TBD) |

CLI: `vokra-convert restamp --input <in.gguf> --output <out.gguf> --license <spdx> [--model-id <id>] [--source <text>] [--attribution <text>]` — subcommand routed before the converter arg parser (no `--model`).

### 2026-07-23 — 1.0.0-rc.1-dev #2 (HF publication: convert_file_licensed — Rust surface only)

Additive **Rust public API** only; C ABI untouched (conversion is an offline
Rust tool, never a C export).

| Crate / area | Symbol | Kind | Signature | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-convert` | `convert_file_licensed` | Added | `pub fn convert_file_licensed(model, input, output, license: Option<&str>) -> Result<ConvertSummary, ConvertError>` | Override the stamped weight licence when the distribution *source* declares a different one from the converter's built-in default (Whisper is MIT on OpenAI's GitHub but the HF weight repos tag base/small/medium as apache-2.0). Keeps the GGUF the single source of truth the model card is generated from. `convert_file` now delegates to it with `None` — its signature and behaviour are unchanged. | no | (TBD) |

CLI: `vokra-convert` gains `--license <spdx>`, routed to `convert_file_licensed`
on the plain single-input path (Whisper etc.). GGUF metadata effect: overrides
`vokra.provenance.{weight_license,license,source}` when set.

### 2026-07-23 — 1.0.0-rc.1-dev (HF publication: LicenseClass gains three variants — Rust surface only)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched; `LicenseClass` is a host-side compliance type and has never been a C
export. Recorded here because it is a public enum without `#[non_exhaustive]`,
so downstream exhaustive `match`es would need a new arm (permitted under the
Pre-1.0 / prerelease policy: rename/remove free, changelog entry required).

| Crate / area | Symbol | Kind | Signature / shape | Rationale | Breaking? | PR |
| --- | --- | --- | --- | --- | --- | --- |
| `vokra-core::compliance` | `LicenseClass::Copyleft` | Added | enum variant, wire name `"copyleft"` | Share-alike / strong copyleft (CC-BY-SA, AGPL, GPL, LGPL): redistribution permitted **with the licence preserved**. Previously CC-BY-SA mis-classified as `AttributionRequired` and AGPL fell to `Unknown`. | yes (exhaustive match) | (TBD) |
| `vokra-core::compliance` | `LicenseClass::RedistributionForbidden` | Added | enum variant, wire name `"redistribution-forbidden"` | Contractual bans that no licence string expresses (VOICEVOX reverse-engineering clause, CSJ agreement, JSUT/JVS corpus terms). **Never inferred from text** — set only from an explicit list. | yes | (TBD) |
| `vokra-core::compliance` | `LicenseClass::ConditionalCommercial` | Added | enum variant, wire name `"conditional-commercial"` | Threshold-conditioned commercial grants (LFM ≥$10M revenue, Boson >100k AAU, IndexTTS-2 >100M MAU). | yes | (TBD) |
| `vokra-core::compliance` | `LicenseClass::redistributable` | Added | `pub fn redistributable(self) -> bool` | The **publishing** gate, deliberately separate from the loading gate (`requires_research_flag`). The two answers differ for nearly every non-permissive class. | no | (TBD) |
| `vokra-core::compliance` | `LicenseClass::requires_license_preserved` | Added | `pub fn requires_license_preserved(self) -> bool` | Whether republishing must carry the original licence unchanged (relabelling a CC-BY-SA-derived GGUF as Apache-2.0 is a misrepresentation). | no | (TBD) |

**Behaviour changes to existing variants** (no signature change):

- `from_license_str` now tests share-alike / copyleft **before** the plain
  `cc-by` arm. `cc-by-sa-4.0` contains `cc-by`, so the previous ordering
  reported every share-alike weight as attribution-only. Load-bearing today:
  Style-Bert-VITS2's mandatory runtime BERT and the JVNV weights are both
  `cc-by-sa-4.0`.
- `agpl-3.0` / `gpl-*` / `lgpl-*` / `openrail` now classify as `Copyleft`
  instead of `Unknown`. These licences do not restrict *use*, so such weights
  **no longer require a research flag to load** — the old gating was an
  artifact of an unrecognised string, not a considered position.
- `commercial_ok` returns `true` for `Copyleft` (AGPL/GPL/CC-BY-SA all permit
  commercial use) and no longer doubles as the official-zoo admission test;
  that question moved to `redistributable`.
- `requires_attribution` now also covers `Copyleft` and
  `NonCommercialShareAlike`, whose licences carry BY / notice-retention terms.

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
**no** C ABI symbol (machine-gated by `scripts/check-m5-residual-no-abi.sh`)
and are **not** `OpKind` variants. They are also absent from `MinDtypeRegistry`
with one documented exception: `bigvgan_generator`'s min-dtype anchor *is*
registered (M2-08, fp16 minimum) — only its op landing is M5, which is exactly
what `m5_residual_ops::tests::bigvgan_min_dtype_anchor_is_registered_but_op_is_m5`
pins.

Read the blocker column as **what is still reserved**, not as "what does not
exist". Per ADR M4-20 §D-5 these decoders / generators are deliberately
*runtime functions* rather than `OpKind` variants, so a landed runtime
primitive and a still-reserved graph-side id coexist by design — the
reservation stays valid after the primitive ships, and only the graph-side
variant + C ABI export are actually deferred to the M5-13 freeze policy.

**Updated 2026-08-15** — the `bigvgan_generator` / `ctc_decode` / `rnnt_decode`
rows previously asserted a missing trigger model. That is no longer true (all
three runtime primitives landed, and `rnnt_decode` has a live consumer), so the
rows now name what is genuinely reserved. `titanet_speaker_encode` and
`diarize` are synced to the license decisions already recorded in
`crates/vokra-core/src/m5_residual_ops.rs`. The new gate
`scripts/check-m5-residual-blockers.sh` keeps this column from drifting back.

| Reserved op-kind id          | FR-OP    | M5 blocker (what is still reserved)                           |
| ---------------------------- | -------- | ------------------------------------------------------------ |
| `bigvgan_generator` (op)     | FR-OP-11 | graph-side `OpKind` variant + C ABI export. Runtime vocoder, strict real-weight binder, alias-free forward parity, and explicit mel-file CLI route landed; min-dtype anchor registered (M2-08) |
| `ctc_decode`                 | FR-OP-41 | graph-side `OpKind` variant + C ABI export. Runtime primitives landed (`ctc_decode_greedy` / `ctc_decode_beam`, LM shallow fusion + hotwords), with live greedy consumers in `parakeet_ctc`, `wav2vec2_ctc`, and `data2vec_audio`; Canary is Transformer-AED and does not consume CTC |
| `rnnt_decode`                | FR-OP-42 | graph-side `OpKind` variant + C ABI export. Runtime primitive and live `ParakeetTdt11b::decode_tdt` consumer landed; the same strict model now also has a complete native PCM-to-token/text TDT forward |
| `ecapa_tdnn_speaker_encode`  | FR-OP-80 | graph-side `OpKind` variant + dedicated op-level C ABI export. Native strict binder, CPU/Metal model forward, CLI and model-generic `vokra_speaker_embed` route landed |
| `wespeaker_speaker_encode`   | FR-OP-80 | native strict model forward and generic speaker C ABI landed; graph `OpKind` + dedicated op export remain reserved |
| `titanet_speaker_encode`     | FR-OP-80 | native strict TitaNet-L model forward and generic speaker C ABI landed; graph `OpKind` + dedicated op export remain reserved |
| `diarize`                    | FR-OP-82 | trigger only — the license half unblocked 2026-07-30: pyannote is MIT by primary source (`gated: auto` is access control, no extra terms), `docs/license-audit.md` §3.1 row 263 |

### 2026-08-09 — 1.0.0-rc.1-dev (Wave 7 Part C: Moshi head mapped-lazy, MOSHI-16GB-STRATEGY residual)

Additive **Rust public API** change only — the C ABI (`include/vokra.h`) is
untouched (baseline 33 fn / 11 typedefs unchanged; `scripts/check-abi-changelog.sh`
green), and **no `vokra.*` GGUF key was added or renamed**. Extends the
cc-06 (2026-07-19) mapped-lazy load path with the Voxtral `MappedHeads`
pattern (12e574e): `MoshiEngine::from_path` / `from_path_with_policy` now
serve head reads (`text_emb` / `audio_emb` / `text_linear`) straight out of
the mapping too, saving ~1.3 GiB additional resident footprint at the
full-7B shape (projected ~10 GB peak on 16 GB hosts, from cc-06's 11.43 GB
measurement — actual measurement is owner scope). Bit-identical to the
resident path — per-row / per-chunk widen preserves the byte formula and
each row's inner accumulation order (pinned by
`fully_mapped_backbone_matches_resident_bitwise` + the existing
`converted_gguf_loads_under_strict_policy_..._end_to_end` bit-identity
assertion on dialog turn text + PCM).

| Crate / area   | Symbol                                        | Kind  | Signature                                                                                              | Rationale                                                                                                     | Breaking? | PR    |
| -------------- | --------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | --------- | ----- |
| `vokra-models::moshi` | `MappedHeadWeights` (struct + `bind` / `embed_text_into` / `embed_audio_into` / `text_logits_into` / `out_norm_gamma` / `n_audio_tables` / `text_card`) | Added | bounded-memory head store: `text_emb`, `emb.{k}.weight`, `text_linear` descriptors kept mapped; `out_norm.alpha` widened once at bind and cached; per-row widen+accumulate for embeddings; chunked GEMV (`MOSHI_HEAD_CHUNK_ROWS = 128`) for the text head — all bit-identical to the resident path | full-7B on 16 GB machines, ~1.3 GiB additional headroom vs cc-06 | no | (TBD) |
| `vokra-models::moshi` | `MoshiBackbone::new_mapped_full` / `MoshiBackbone::is_head_mapped` | Added | fully bounded-memory backbone constructor (head + blocks both mapped); observability accessor | wires `MappedHeadWeights` into the backbone dispatch (embed_step / text_logits_into / forward_impl.out_norm) | no | (TBD) |
| `vokra-models::moshi` | `MoshiEngine::backbone_is_mapped` / `MoshiEngine::backbone_is_head_mapped` | Added | observability accessors — regression guards for `from_path`'s bounded-memory contract | test / operator visibility into the load posture | no | (TBD) |

Notes:
- `WeightResidency::MappedLazy` (private enum) was replaced by
  `WeightResidency::MappedLazyFull` — no public API change, and direct
  callers can still construct the intermediate "head resident + blocks
  mapped" posture via `MoshiBackbone::new_mapped` (which `parity_moshi.rs`
  Stage C exercises directly).
- Audit `MOSHI-16GB-STRATEGY` was substantially resolved by cc-06
  (measured peak 11.43 GB on M1 16 GB, 2026-07-19). This Wave 7 Part C
  entry lands the natural Voxtral-pattern extension so future audits find
  the head store mapped-lazy, not just the blocks.

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
| M4-16 | `vokra.wavtokenizer.*`         | `vokra.wavtokenizer.vocab_size`, `vokra.wavtokenizer.d_model` — 1:1 with `WavTokenizerVqAttrs` (`crates/vokra-ops/src/fsq_codec.rs`). Released WavTokenizer configs = `4096 / 512` (upstream `vq_bins: 4096` / `dimension=512`, `num_quantizers: 1` — ADR M4-16 §D-c, verified 2026-07-15). | `u32`         | documented  | FSQ-family (FR-OP-31) single-codebook VQ decode attributes. The strict native runtime landed on 2026-08-25 and accepts the exact audited public 1,091-tensor files, which predate these optional keys, through a fixed 4096×512 released-manifest contract. Converter-side emission remains owed before this row can become `persisted`; arbitrary missing metadata never enables a generic fallback. **M5-13 freeze: to be declared `EXPERIMENTAL`** (`docs/handoff/m4-12.md` §(e)-2) so schema evolution stays legal at minor bumps until the codec API stabilizes. | M4-16 (2026-07-15)         |
| M4-16 | `vokra.xcodec2.*`              | `vokra.xcodec2.levels` (`u32` array), `vokra.xcodec2.d_model` — 1:1 with `Xcodec2FsqAttrs` (`crates/vokra-ops/src/fsq_codec.rs`). Released X-Codec 2 checkpoint = levels `[4; 8]` (effective vocab 4^8 = 65536) / `d_model 2048` (`vq_dim`; upstream `vq/codec_decoder_vocos.py` + `modeling_xcodec2.py`, pin `vector-quantize-pytorch==1.17.8` — ADR M4-16 §D-c, verified 2026-07-15). | `u32-array` + `u32` | documented  | FSQ-family (FR-OP-31) finite-scalar-quantization dequant attributes (implicit grid + out-projection GEMV; **separate subgraph from the RVQ `vokra.mimi.*` / `vokra.dac.*` chunks** — no cross-codebook axis). Converter-side emission lands with the real X-Codec 2 model-integration WP. **M5-13 freeze: to be declared `EXPERIMENTAL`** (handoff §(e)-2) — same intent record as the `vokra.wavtokenizer.*` row above. | M4-16 (2026-07-15)         |
| M5 Mac coverage | `vokra.ast.*` | `vokra.ast.{hidden_size,num_hidden_layers,num_attention_heads,intermediate_size,patch_size,frequency_stride,time_stride,num_mel_bins,max_length,num_labels,num_prefix_tokens,sample_rate,layer_norm_eps_scaled_1e12,frame_length,frame_shift,low_freq_hz}` (`u32`), `qkv_bias` / `subtract_mean` (`bool`), `hidden_act` / `window_type` (`string`), and `normalize_mean` / `normalize_std` (`f64`) | mixed | persisted | Complete released AudioSet AST topology and official 16 kHz Kaldi-fbank frontend contract, written by `convert_ast_file` and strict-bound by `vokra-models::ast`. Unknown values or tensor manifests fail closed; no default silently substitutes a different classifier. | `187d1f02` |

| M4-06 | `vokra.moshi.*`                | `vokra.moshi.arch.temporal.{n_layer,d_model,n_head,ffn_hidden}`, `vokra.moshi.arch.depth.{n_layer,d_model,n_head,ffn_hidden}`, `vokra.moshi.arch.{rms_norm_eps,rope_max_period,context,max_ctx}`, `vokra.moshi.audio.{n_q_in,dep_q,card}`, `vokra.moshi.text.{card,pad_id,end_pad_id}`, `vokra.moshi.n_delays` + indexed `vokra.moshi.delay.{i}` (count + indexed keys — the `vokra.mimi.seanet.ratio.{i}` precedent). Shape-driven where derivable (layer counts / widths / gating hidden / stream tallies / vocabs from the T02 355-tensor manifest); head counts / ε (1e-8 `rms_norm_f32`) / max_period (10000) / context (3000) / pad ids (3, 0) / delays (`[0,0,1×7,0,1×7]` structural rule — 7B verbatim) are `_lm_kwargs` transcriptions (ADR M4-06 §D2/§D3). Audio rates deliberately absent — the shared `vokra.mimi.*` chunk is the single rate authority (§D3 no-duplication rule; `quantizer.n_q = max(dep_q, n_q−dep_q)`, `bins ≡ card` per loaders.py). `vokra.tokenizer.model` reused for the raw SentencePiece blob; `vokra.provenance.attribution` (new provenance key) carries the FR-MD-09 display text. | `u32` + `f32` + `string` | persisted | Moshi (Helium temporal + depformer, full-duplex S2S) architecture attributes — written by `crates/vokra-convert/src/models/moshi.rs` (`convert_moshi_file`), read by `MoshiConfig::from_gguf` (`crates/vokra-models/src/moshi/config.rs`). BF16 checkpoint tensors are decoded to F32 **exactly** at conversion (`GgmlType::BF16 = 30` read support). | M4-06 (2026-07-15)         |
| RW-fix | `vokra.voxtral.text_decoder.*` (extension) | Adds `vokra.voxtral.text_decoder.{head_dim,n_head_q,n_head_kv,rope_base,rms_norm_eps,n_ctx}` to the M3-10 base set ({n_layer,hidden_dim,ffn_dim,vocab_size}). `head_dim` decouples the attention width from `hidden/n_head_q` (real Voxtral-Mini: 32 q-heads x 128 = 4096 != hidden 3072); `head_dim = 0` (or absent) = legacy `hidden/n_head_q` derivation, so pre-fix GGUFs still load. Written by `crates/vokra-convert/src/models/voxtral.rs`, read by `VoxtralConfig::from_gguf` (`crates/vokra-models/src/voxtral/config.rs`). | `u32` + `f32` | persisted | Real-weight campaign fix `12e574e` (GQA loader + BF16 passthrough converter): the real checkpoint's GQA head split and untied `lm_head` are now representable; converter also accepts sharded `*.index.json` input and hard-errors on weightless output. | 2026-07-16 (campaign 1 P1 fix) |
| RW-fix | `vokra.cosyvoice2.arch.*` (extension + real values) | Adds **written** emission of `vokra.cosyvoice2.arch.{n_head_kv,rope_base,rms_norm_eps,n_ctx}` (key strings pre-existed as read-side constants only) and replaces the previously **0-placeholder** values of `arch.{vocab_size,hidden_dim,n_layer,n_head,ffn_dim}` with shape-derived reals (0.5B: vocab 151936 / hidden 896 / 24L / ffn 4864; head split 14q/2kv from `--config`, cross-checked vs shapes). Plus q/k/v bias tensors now travel in the GGUF. | `u32` + `f32` | persisted | Real-weight campaign fix `7336079`: pre-fix GGUFs bound `llm=None` (all-zero hparams); old files still load (back-compat verified). NOTE: `~/.cache` artifacts converted before this fix are stale — reconvert. | 2026-07-16 (campaign 1 P1 fix) |
| RW-fix | `vokra.denoise.*` (schema v2 — REPLACES the M4-20 scaffold row above) | Config keys now: `vokra.denoise.{n_fft,hop,sample_rate,n_erb,df_bins,df_order,min_nb_erb_freqs,conv_lookahead,df_lookahead,conv_ch,emb_hidden_dim,df_hidden_dim,enc_linear_groups,linear_groups,df_gru_linear_groups,emb_num_layers,df_num_layers}` (`u32`) + `vokra.denoise.{lsnr_min,lsnr_max,norm_alpha}` (`f32`). **REMOVED**: `vokra.denoise.hidden` and the 6 scaffold tensor names — tensors are now the 115 verbatim upstream-named DeepFilterNet3 tensors (exact-shape validated, unknown names hard-error). Written by the real-checkpoint converter (`convert --model denoise`), read by `DenoiseModel::from_gguf`. | `u32` + `f32` + tensors | persisted | Campaign-2 P1 fix `9b718d1` (DFN3 real topology, sample-level parity SI-SNR gap 2.0e-7 dB). Pre-1.0 removal is legal (prerelease ABI policy) and recorded here per the recording rules; scaffold-schema GGUFs no longer load (hard error, FR-EX-08). | 2026-07-17 (campaign 2 P1 fix) |
| M5-resid | `vokra.schema.*` | `vokra.schema.version` (`u32`) + `vokra.schema.producer` (`string`). Written **unconditionally by `GgufBuilder::effective_metadata`**, so every Vokra-written GGUF carries them — not per-converter (only 4 of 13 model converters share a provenance helper, so a per-converter stamp would silently miss the rest). Caller-supplied values are filtered out and replaced, so the stamp cannot be spoofed. Read via `vokra_core::gguf::schema::{schema_version,producer,describe,stale_group_hint}`. Absent = **pre-stamping artifact** (every GGUF converted before 2026-07-22), which is a first-class answer, not an error — old files keep loading. | `u32` + `string` | persisted | Makes a **stale** GGUF visible. Observed 2026-07-22: a cached `mimi.gguf` (319 tensors / 9 keys, no `vokra.mimi.seanet.*`) loaded clean in one consumer and silently fell back to a synthesized bridge, while a re-conversion produced 603 tensors / 36 keys with a bindable PCM chain. `SCHEMA_VERSION` is a hand-bumped generation integer, deliberately **not** `CARGO_PKG_VERSION` — that string has been `0.1.0-alpha.0` since M0 and was identical on both sides of the incident. | 2026-07-22 |
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
| M5-gap | `vokra.f0.crepe.*` | `vokra.f0.crepe.capacity` (`string`, one of `"tiny"`/`"small"`/`"medium"`/`"large"`/`"full"` — the upstream size knob, `crepe/core.py::build_and_load_model`), `vokra.f0.crepe.hop` (`u32`, default 160 = 10 ms @ 16 kHz), `vokra.f0.crepe.fmin` (`f32`, informational search-grid floor, default 50.0), `vokra.f0.crepe.fmax` (`f32`, informational search-grid ceiling, default 1100.0). Weight tensors travel under the upstream Keras layer names permuted to Vokra layout: `conv{1..6}.{weight,bias,bn.gamma,bn.beta,bn.moving_mean,bn.moving_variance}` (F32) + `classifier.{weight,bias}` (F32). Weight-tensor absence keeps the runtime in the honest UNIMPLEMENTED skeleton path (metadata-only GGUFs written by the earlier skeleton still load — the frame-count contract is preserved). | `string` + `u32` + `f32` + `f32` tensors | persisted | CREPE (Kim et al. 2018) F0 (fundamental-frequency) extractor — written by `crates/vokra-convert/src/models/crepe.rs` (`convert_crepe_file`), read by `CREPE::from_gguf` (`crates/vokra-models/src/f0/crepe.rs`). Weight license = **MIT** (`marl/crepe/main/LICENSE.txt`, "Copyright (c) 2018 Jong Wook Kim et al.", CC-verified 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」). The offline `tools/parity/keras_h5_to_safetensors.py` bridges the upstream Keras `.h5` release into the safetensors + config side-car this converter consumes (the DAC / Kokoro / UTMOS split — zero-dep, no TensorFlow / Keras / torch in the runtime, NFR-DS-02 / FR-LD-05). Real-weight parity is env-gated (`VOKRA_CREPE_GGUF` + `VOKRA_CREPE_REFERENCE_WAV` + `VOKRA_CREPE_REFERENCE_JSON`, `crates/vokra-models/tests/parity_crepe.rs`, atol_hz = 3.0). | 2026-07-30 (M5 gap follow-up) |
| runtime-gap Wave 3 | `vokra.charsiu.*` | `revision`, `checkpoint_sha256`, `hidden_size`, `ffn_dim`, `n_layer`, `n_head`, `vocab_size`, `silence_id`, `pad_id`, `sample_rate`, `frame_shift_sec`, `layer_norm_eps`, `pos_conv_kernel`, `pos_conv_groups`, `silence_threshold`, `vocab` | `string` + `u32` + `f32` + `string[]` | persisted | Canonical `charsiu/en_w2v2_fc_10ms` writer/reader contract. The converter verifies all 213 upstream tensors, emits 211 runtime tensors, folds positional-conv weight norm, and stamps the official 42-label inventory. The loader requires every key and exact pinned provenance before the post-norm Wav2Vec2 forward. | PR #44 / 2026-08-21 |
| runtime-gap Wave 4 ASR | `vokra.moonshine.*` | `revision`, `checkpoint_sha256`, `tokenizer_sha256`, `sample_rate`, `hidden_size`, `intermediate_size`, `encoder_layers`, `decoder_layers`, `encoder_heads`, `decoder_heads`, `vocab_size`, `max_positions`, `decoder_start_token_id`, `eos_token_id`, `rope_theta`, `partial_rotary_factor`, `encoder_activation`, `decoder_activation` | `string` + `u32` + `f32` | persisted | Canonical `moonshine-ai/moonshine-{tiny,base}` identity/topology contract. The strict binder additionally requires the existing `vokra.tokenizer.model` U8 blob and all 160/210 exact tensors. | PR #44 / 2026-08-22 |
| runtime-gap Wave 4 ASR | `vokra.medusa.*` | `revision`, `source_revision`, `config_sha256`, `checkpoint_sha256`, `num_heads`, `module_count`, `num_layers`, `hidden_size`, `heads_type`, `choices`, `init_from_proj`, `output_whisper_original` | `string` + `u32` + `u32[]` + `bool` | persisted | Canonical `aiola/whisper-medusa-v1` identity/topology contract. The strict binder additionally requires all 22 `medusa_heads.{0..10}.0.linear.{weight,bias}` tensors, the ordinary Whisper contract, and MIT provenance. | PR #44 / 2026-08-22 |
| runtime-gap Wave 4 prerequisite + 2026-08-24 native completion | `vokra.conv_tasnet.*` | `n_filters`, `n_kernel`, `stride`, `n_blocks`, `n_repeats`, `bn_chan`, `hid_chan`, `skip_chan`, `conv_kernel_size`, `sample_rate`, `n_src`, `causal` | `u32` | persisted | Asteroid `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k` topology contract — written by `crates/vokra-convert/src/models/conv_tasnet_libri1mix.rs` and required by `ConvTasnet::from_gguf`. **Additive**: the group is new; no existing key is renamed. The completed strict loader pins the official `n_kernel=32` / `stride=16`, 345-tensor, 24-block non-causal topology; the former 16/8 public artifact fails closed. Rust `SeparationEngine`, CLI and the pre-existing C `vokra_separate` symbol use this metadata without adding a new C ABI symbol. | PR #44 / 2026-08-21; native completion 2026-08-24 |
| runtime-gap Wave 4 prerequisite | `vokra.jasco.*` | `d_model`, `num_layers`, `n_heads`, `ffn_dim`, `num_codebooks`, `codec_frame_rate_hz`, `sample_rate_hz`, `text_prefix_len`, `chord_vocab_size`, `drum_vocab_size`, `num_flow_steps`, `cfg_scale` | `u32` + `f32` | persisted | Official AudioCraft JASCO 400M chords+drums configuration contract — written by `crates/vokra-convert/src/models/jasco_400m_chords_drums.rs` and read by `JascoConfig::from_gguf`. **Additive**: the group is new; no existing key is renamed or changes meaning. `chord_vocab_size = 195` includes the null condition; the legacy-named drum axis records the official 128-wide EnCodec latent input; `num_flow_steps = 100` is the official Euler fallback and `cfg_scale = 5.0` is the official default. | PR #44 / 2026-08-21 |
| Mac CPU/Metal coverage — Lang-ID | `vokra.lang_id.*` | `upstream_revision`, `sample_rate`, `n_mels`, `tdnn_channels`, `mfa_channels`, `attention_channels`, `res2net_scale`, `embedding_dim`, `classifier_kind`, `classifier_hidden_dim` (XVector only), `class_count`, `labels`, `bn_eps`, `stats_eps`, `leaky_relu_slope` (XVector only), `artifact_layout`; `block_kernels[]`, `block_dilations[]` | `string` + `u32` + `u32-array` + `string-array` + `f32` | persisted | Strict prepared-checkpoint contract for the two official SpeechBrain ECAPA Lang-ID variants. The converter accepts only the v2 sidecar layout, preserves the ordered official label encoder, canonicalizes the complete classifier head, and records each residual block's actual kernel/dilation instead of assuming VoxLingua107 and CommonLanguage share one topology. Historical embedding-only files remain explicit load errors. | 2026-08-26 / `e73a8c8c` |

### 2026-07-30 — VoxCPM2 2B variant support (Option C hybrid)

| Crate / area                    | Symbol                                                | Kind    | Signature                                                                                                                                                                                                          | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                | Breaking? | PR   |
| ------------------------------- | ----------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---- |
| `gguf:vokra.model.name`         | `"voxcpm2-0.5b"` / `"voxcpm2-2b"`                     | Added   | `string`                                                                                                                                                                                                            | Rename `voxcpm-0.5b` → `voxcpm2-0.5b`; new `voxcpm2-2b` for `openbmb/VoxCPM2`. Both variants share `vokra.model.arch = "voxcpm2"`; the parity harness dispatches on `vokra.model.name` (`crates/vokra-models/tests/parity_tts_continuous_vae.rs`). The legacy `voxcpm-0.5b` string stays registered in `vokra_core::compliance::license_class` for backward compat with any pre-rename GGUF on disk. Spec: `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`. | no        | —    |
| `gguf:vokra.voxcpm2.*`          | `vokra.voxcpm2.lm.kv_channels`                        | Added   | `u32` (0.5B: 64 derived, 2B: 128 explicit)                                                                                                                                                                          | LM per-head channel width. 2B primary source `openbmb/VoxCPM2/config.json.lm_config.kv_channels` is explicit; the 0.5B derived value was previously implicit and is now stamped so the runtime can cross-check without recomputing.                                                                                                                                                                                                                     | no        | —    |
| `gguf:vokra.voxcpm2.*`          | `vokra.voxcpm2.encoder.kv_channels`                   | Added   | `u32`                                                                                                                                                                                                              | Same rationale as LM. Encoder + DiT share the pattern.                                                                                                                                                                                                                                                                                                                                                                                                | no        | —    |
| `gguf:vokra.voxcpm2.*`          | `vokra.voxcpm2.dit.kv_channels`                       | Added   | `u32`                                                                                                                                                                                                              | Same.                                                                                                                                                                                                                                                                                                                                                                                                                                                 | no        | —    |
| `gguf:vokra.voxcpm2.*`          | `vokra.voxcpm2.dit.mean_mode`                         | Added   | `bool` (both variants: false)                                                                                                                                                                                       | 2B `config.json.dit_config.mean_mode` is explicit `false`; 0.5B non-explicit training-side default is `false`. Field is scaffold-only until the runtime forward branch consumes it (T29-equivalent follow-up); recording it now avoids a silent drift when a future variant flips it to `true`.                                                                                                                                                             | no        | —    |
| `gguf:vokra.voxcpm2.*`          | `vokra.voxcpm2.residual_lm.no_rope`                   | Added   | `bool` (0.5B: false, 2B: true)                                                                                                                                                                                     | 2B skips RoPE on the residual acoustic LM Q/K path. Runtime forward branch that consumes this axis lands in a follow-up wave (T29-equivalent); the flag exists so a converter that emits it and a runtime that ignores it are catchable at parity time.                                                                                                                                                                                                | no        | —    |
| `gguf:vokra.vae_continuous.*`   | `vokra.vae_continuous.sr_bin_boundaries`              | Added   | `u32-array` (only present when non-empty — 0.5B: absent, 2B: `[20_000, 30_000, 40_000]`)                                                                                                                            | Bandwidth-adaptive decoder-head boundaries. Absent on the 0.5B GGUF (single-head decoder) — a downstream consumer reading `Option<Vec<u32>>` sees `None`. Matched to `ContinuousVaeConfig::voxcpm2_2b().sr_bin_boundaries` element-wise by the parity harness.                                                                                                                                                                                              | no        | —    |
| `vokra-convert`                 | `models::voxcpm2::VoxCpm2Variant` (enum, crate-priv) | Added   | `enum VoxCpm2Variant { HalfB, TwoB }`                                                                                                                                                                              | Selected by `models::voxcpm2::detect_variant(&SafetensorsFile)` from `base_lm.embed_tokens.weight`'s hidden dim (1024 → HalfB, 2048 → TwoB). Any other value is a loud `ConvertError::Parse` — no silent default (FR-EX-08). Not a C ABI symbol; recorded here so the GGUF metadata delta above has a symmetrical Rust-surface entry.                                                                                                                    | no        | —    |
| `vokra-convert`                 | `models::voxcpm2::VoxCpm2Report::variant`             | Added   | `pub(crate) variant: Option<VoxCpm2Variant>`                                                                                                                                                                       | Report field so the CLI trailer surfaces the detected variant. `None` reserved for pre-detection shape (currently unused — every successful `convert` sets it, every failure returns early).                                                                                                                                                                                                                                                          | no        | —    |
| `vokra-core:license_class`      | `voxcpm2-0.5b` / `voxcpm2-2b` / `voxcpm2-` prefix     | Added   | `LicenseClass::Permissive` (apache-2.0 end-to-end)                                                                                                                                                                | The registry accepts every canonical + underscore + `-base` spelling of both variants, plus the `voxcpm2-` prefix guard for future 2B-lineage variants. The pre-existing `voxcpm-` prefix + `voxcpm-0.5b` explicit entries stay live for backward compat with legacy GGUFs.                                                                                                                                                                              | no        | —    |

### 2026-08-15 — model-converter chunk backfill (50 prefixes, retroactive)

Fifty `vokra.<model>.*` chunk groups were being stamped by converters in
`crates/vokra-convert/src/models/` with **no row anywhere in this file**.
Under §"Scope: what belongs in this file" the GGUF metadata schema is
in-scope precisely because model files are content-addressed by these
chunks: a consumer who converted last month and one who converts today got
different metadata with nothing on disk recording the difference. This
section closes that, and `scripts/check-abi-changelog.sh
--check-gguf-prefixes` (added the same day) keeps it closed mechanically.

Read this as a **record, not a change**: nothing here alters a shipped key.
Every group listed is new in its entirety, so the "no pre-existing key
changed meaning" claim in each row is a statement about the whole group,
not a per-key diff against an earlier schema.

Scope limits, stated rather than implied:

- **The `WP` column reads `backfill`** because the work-package that
  introduced each group is not recoverable from the converter source, and
  inventing one would be worse than omitting it. The `Introducing wave /
  commit` column carries the real traceability — each SHA is the
  `--diff-filter=A` commit that added the converter file.
- **`Keys` is truncated to the first four leaf names plus an exact count.**
  The converter file named in the row is the authority for the full set;
  transcribing 500-odd leaf names here would be a second copy to drift.
- **`Kind` is the observed writer call** (`add_u32` / `add_f32` /
  `add_bool` / `add_string`), and for the four groups that go through
  `GgufBuilder::add_metadata` the concrete `GgufMetadataValue` variant
  (`vokra.kokoro.*` / `vokra.ct_punc.*` string arrays, `vokra.maest.*`
  `F64`, `vokra.itn.*` `U8` array + `U64` + `I64`).
- **`vokra.voila.*` is deliberately NOT in this table.** An audit listed it
  as an unrecorded stamped prefix; it is not stamped at all.
  `models/voila.rs` mentions the string only in a refusal message and in a
  test (`no vokra.voila.* axis may be stamped without a primary source`)
  that asserts the emitted GGUF carries zero such keys. Adding a row for it
  would document a chunk group that does not exist.
- **Converter-crate scope.** These rows, and the gate, cover
  `crates/vokra-convert/src/`. Two writers live elsewhere and are already
  recorded: `vokra.denoise.*` (`DenoiseModel::to_gguf_bytes` in
  `vokra-ops`) and the unconditional `vokra.schema.*` / `vokra.provenance.*`
  / `vokra.model.*` stamp in `GgufBuilder::effective_metadata`
  (`vokra-core`).

| WP    | Chunk prefix | Keys | Kind | Status | Rationale | Introducing wave / commit |
| ----- | ------------ | ---- | ---- | ------ | --------- | -------------------------- |
| backfill | `vokra.atst.*` | **34** keys — `act_layer`, `amp_to_db_stype`, `amp_to_db_top_db`, `depth`, …  | `u32` + `f32` + `bool` + `string` | persisted | `atst-base`, `github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst` — written by `crates/vokra-convert/src/models/atst.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `a8867cf` (2026-08-13) |
| backfill | `vokra.audiosr.*` | **32** keys — `attention_resolution_`, `attention_resolutions_count`, `beta_schedule`, `channel_mult_`, …  | `u32` + `string` | persisted | `audiosr` — written by `crates/vokra-convert/src/models/audiosr.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `10e42e5` (2026-08-15) |
| backfill + correction | `vokra.bark.*` | **16** keys — `block_size`, `coarse.input_vocab_size`, `coarse.output_vocab_size`, `codec.sample_rate`, `tensor_manifest_sha256`, …  | `u32` + `string` | persisted | `bark` — written by `crates/vokra-convert/src/models/bark.rs`. **Correction (2026-08-27)**: full Bark is `hidden_size=1024` / `num_heads=16`; only Bark Small is `768` / `12`. `tensor_manifest_sha256` is additive and pins the complete 758-tensor Full or 518-tensor Small checkpoint, including the embedded `codec_model.*` EnCodec decoder. The exact historical public Full artifact may carry the old width/head metadata only when its immutable full manifest authenticates it; unrelated or partial checkpoints fail closed. | `02664f6` (2026-08-06); `6c00beca` (2026-08-27) |
| backfill | `vokra.beat_this.*` | **6** keys — `d_model`, `n_classes`, `n_frames`, `n_head`, …  | `u32` | persisted | `beat-this`, `github.com/CPJKU/beat_this` — written by `crates/vokra-convert/src/models/beat_this.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `173fea8` (2026-08-14) |
| backfill | `vokra.bigvgan.*` | **1** keys — `variant` | `string` | persisted | `bigvgan` — written by `crates/vokra-convert/src/models/bigvgan.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.canary_qwen.*` | **22** keys — `arch.cross_attn.hidden_dim`, `arch.decoder.ffn_dim`, `arch.decoder.head_dim`, `arch.decoder.hidden_dim`, …  | `u32` + `f32` | persisted | `canary-qwen-2.5b` — written by `crates/vokra-convert/src/models/canary_qwen.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.chatterbox_nano.*` | **25** keys — `arch.ffn_dim`, `arch.head_dim`, `arch.hidden_dim`, `arch.hop_size`, …  | `u32` + `f32` + `string` | persisted | `chatterbox_nano` — written by `crates/vokra-convert/src/models/chatterbox_nano.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |
| backfill | `vokra.chatterbox_turbo.*` | **21** keys — `arch.head_dim`, `arch.hidden_dim`, `arch.hop_size`, `arch.max_speech_tokens`, …  | `u32` + `string` | persisted | `chatterbox_turbo` — written by `crates/vokra-convert/src/models/chatterbox_turbo.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |
| backfill | `vokra.ct_punc.*` | **11** keys — `att_unit`, `attention_heads`, `embed_unit`, `kernel_size`, …  | `u32` + `f32` + `string[]` | persisted | `ct-punc`, `funasr/ct-punc` — written by `crates/vokra-convert/src/models/ct_punc.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `10e42e5` (2026-08-15) |
| runtime-gap | `vokra.data2vec_audio.*` | **14** keys — `hidden_size`, `n_layer`, `n_head`, `intermediate_size`, `vocab_size`, `layer_norm_eps`, `num_conv_pos_embeddings`, `conv_pos_kernel_size`, `num_conv_pos_embedding_groups`, `num_feat_extract_layers`, `conv_bias`, `conv_dim`, `conv_kernel`, `conv_stride` | `u32` + `u32[]` + `f32` + `bool` | persisted | Dedicated `facebook/data2vec-audio-base-960h` topology contract — written by `crates/vokra-convert/src/models/data2vec_audio.rs` and read by the strict native binder. **Additive**: it replaces the incorrect assumption that Data2Vec was byte-compatible with `vokra.wav2vec2_ctc.*`; the one exact legacy public artifact remains readable only through its audited provenance/tensor contract. | working tree (2026-08-24) |
| runtime-gap | `vokra.deepfake.*` | **22** keys — `architecture`, `model_type`, `sample_rate`, `normalize`, `return_attention_mask`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `classifier_proj_size`, `num_classes`, `layer_norm_eps`, `feat_extract_norm`, `do_stable_layer_norm`, `hidden_act`, `num_conv_pos_embeddings`, `num_conv_pos_embedding_groups`, `use_weighted_layer_sum`, `conv_dim`, `conv_kernel`, `conv_stride`, `id2label` | `string` + `u32` + `f32` + `bool` + `u32[]` + `string[]` | persisted | `deepfake-audio-detection-v2` — written by the strict converter from the exact 215-F32-tensor `Wav2Vec2ForSequenceClassification` checkpoint. The additive group corrects the historical WavLM misidentification and pins the 16 kHz normalized waveform frontend, base encoder, 256-wide projector and `0=fake / 1=real` class order. The audited historical public GGUF may omit the group only while its immutable manifest and generic Apache-2.0 provenance match; partial/conflicting groups fail closed. | working tree (2026-08-26) |
| backfill | `vokra.openwakeword.*` | **7** keys — `n_wakewords`, `embedding_dim`, `window_frames`, `mel_bins`, `sample_rate`, `hop_samples`, `wakeword_names` | `u32` + `string[]` | persisted | `openwakeword-op` — written by `crates/vokra-convert/src/models/openwakeword_op.rs`. **Additive**, but note the repair it belongs to: the binder (`vokra-models/src/kws/openwakeword/mod.rs`) had required all seven since it landed, while the converter stamped none, so every GGUF it produced failed to load. Nothing caught it — the unit tests hand-build their GGUF and the parity harness is env-gated. Stamping the group is what makes the documented convert-then-run recipe work for the first time. **Behaviour change**: `--model openwakeword-op` now REFUSES without `--config`, because `wakeword_names` is a user-facing label that cannot be derived from tensors and must not be synthesised (the `ModelKind::Crepe` refusal precedent). The config-less form previously "succeeded" and wrote an unloadable artifact. | `173a811` (2026-08-15) |
| additive | `vokra.openwakeword.*` | **4** keys — `classifier_format`, `classifier_input_frames`, `classifier_layer_counts`, `predict_chunk_samples` | `string` + `u32` + `u32[]` | persisted | `openwakeword-op` native v0.5.1 streaming contract. The keys select the execution-order DNN tensor layout and pin the 16-embedding / 1280-sample prediction windows. Legacy classifier-only GGUFs omit them and keep their old loud-partial behavior; no existing key changes meaning. | #44 (2026-08-22) |
| backfill | `vokra.dia.*` | **28** keys — `arch.decoder.cross_head_dim`, `arch.decoder.cross_query_heads`, `arch.decoder.gqa_head_dim`, `arch.decoder.gqa_query_heads`, …  | `u32` + `f32` | persisted | `dia-1.6b` — written by `crates/vokra-convert/src/models/dia.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |
| backfill | `vokra.diffsinger.*` | **24** keys — `backbone_type`, `diff_accelerator`, `enc_layers`, `f0_max`, …  | `u32` + `f32` + `string` | persisted | `diffsinger`, `github.com/openvpi/DiffSinger` — written by `crates/vokra-convert/src/models/diffsinger.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `10e42e5` (2026-08-15) |
| backfill | `vokra.dtln_aec.*` | **5** keys — `block_len`, `hop`, `lstm_units`, `n_fft`, …  | `u32` | persisted | `dtln-aec`, `github.com/breizhn/DTLN-aec` — written by `crates/vokra-convert/src/models/dtln_aec.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `c8320f0` (2026-08-14) |
| backfill | `vokra.eat.*` | **38** keys — `decoder_dim`, `decoder_groups`, `decoder_kernel`, `decoder_layers`, …  | `u32` + `f32` + `bool` + `string` | persisted | `eat-base`, `github.com/cwx-worst-one/EAT` — written by `crates/vokra-convert/src/models/eat.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `ca04c1b` (2026-08-13) |
| backfill | `vokra.facodec.*` | **6** keys — `hop_size`, `n_quantizers_content`, `n_quantizers_detail`, `n_quantizers_prosody`, …  | `u32` + `string` | persisted | `naturalspeech3-facodec-v2`, `amphion/naturalspeech3_facodec` — written by `crates/vokra-convert/src/models/naturalspeech3_facodec.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.focalcodec.*` | **1** keys — `variant` | `string` | persisted | `focalcodec-50hz`, `lucadellalib/focalcodec_50hz` — written by `crates/vokra-convert/src/models/focalcodec.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| runtime-gap | `vokra.frcrn.*` | **17** keys — `upstream_revision`, `source_revision`, `checkpoint_sha256`, `checkpoint_bytes`, `tensor_manifest_sha256`, `sample_rate`, `window_length`, `hop_length`, `fft_length`, `feature_dim`, `model_depth`, `channels`, `fsmn_order`, `se_hidden`, `unet_count`, `window_type`, `complex` | `string` + `u32` + `bool` | persisted | `frcrn` / `alibabasglab/FRCRN_SE_16K` — written by the strict converter and read by the native FRCRN binder. The additive group pins the exact official checkpoint/source revisions, 812-tensor manifest, fixed sqrt-periodic-Hann frontend and two-complex-U-Net/FSMN topology. The audited historical public GGUF may omit the group only while its exact identity, Apache-2.0 provenance, and full manifest match; a partial or conflicting group fails closed. | working tree (2026-08-26) |
| backfill | `vokra.granite_speech.*` | **36** keys — `arch.decoder.attention_multiplier`, `arch.decoder.embedding_multiplier`, `arch.decoder.ffn_dim`, `arch.decoder.hidden_dim`, …  | `u32` + `f32` | persisted | `granite-speech-4.1-2b`, `ibm-granite/granite-speech-4.1-2b` — written by `crates/vokra-convert/src/models/granite_speech.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.gtcrn.*` | **5** keys — `gru_hidden`, `hop`, `n_bands`, `n_fft`, …  | `u32` | persisted | `gtcrn`, `github.com/Xiaobin-Rong/gtcrn` — written by `crates/vokra-convert/src/models/gtcrn.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `c8320f0` (2026-08-14) |
| runtime-gap | `vokra.hubert.*` | **16** keys — `hidden_size`, `n_layer`, `n_head`, `intermediate_size`, `vocab_size`, `layer_norm_eps`, `feat_extract_norm`, `do_stable_layer_norm`, `hidden_act`, `num_conv_pos_embeddings`, `num_conv_pos_embedding_groups`, `has_ctc_head`, `num_feat_extract_layers`, `conv_dim`, `conv_kernel`, `conv_stride` | `u32` + `u32[]` + `f32` + `bool` + `string` | persisted | `facebook/hubert-large-ls960-ft` identity/topology contract — written by `crates/vokra-convert/src/models/hubert_large_ls960.rs` and read by the strict native HuBERT binder. **Additive**: HuBERT shares learned primitives with Wav2Vec2 but retains its distinct arch and tensor namespace so a sibling checkpoint cannot be silently misrouted. | working tree (2026-08-24) |
| backfill | `vokra.itn.*` | **12** keys — `direction`, `language`, `prefix`, `tagger_bytes`, …  | `u32` + `bool` + `string` + `u8[]` + `u64` + `i64` | persisted | `wetextprocessing`, `github.com/wenet-e2e/WeTextProcessing` — written by `crates/vokra-convert/src/models/wetextprocessing.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `10e42e5` (2026-08-15) |
| backfill | `vokra.kokoro.*` | **11** keys — `hidden_dim`, `istft.hop`, `istft.n_fft`, `istft.win_length`, …  | `u32` + `string[]` | persisted | `kokoro-82m` — written by `crates/vokra-convert/src/models/kokoro.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `e294034` (2026-07-06) |
| backfill | `vokra.kyutai_stt.*` | **24** keys — `arch.backbone.causal`, `arch.backbone.context`, `arch.backbone.d_model`, `arch.backbone.ffn_hidden`, …  | `u32` + `f32` | persisted | `kyutai-stt-2.6b-en` — written by `crates/vokra-convert/src/models/kyutai_stt.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |
| backfill | `vokra.llama_omni2.*` | **11** keys — `variant`, `sample_rate`, `arch.backbone.{n_layer, d_model, n_head, vocab, intermediate_size, rope_max_period, rms_norm_eps}`, `arch.speech_encoder.dim`, `arch.speech_decoder.dim` | `u32` + `f32` + `string` | persisted | `llama_omni2` — written by `crates/vokra-convert/src/models/llama_omni2.rs`. **Additive**, but note the repair it belongs to — the same defect class as the `vokra.openwakeword.*` row above, found one round later. The binder (`vokra-models/src/llama_omni2/mod.rs`) has declared all eleven since it landed; the converter stamped only `variant`, so the other ten decayed to `0` through `read_u32_or_zero` / `read_f32_or` and `validate_for_forward` refused every artifact with "backbone ill-formed (n_layer=0, d_model=0, n_head=0)". Nothing caught it: both halves were tested against a mock of the other, and no test ran the real converter into the real binder until `crates/vokra-models/tests/llama_omni2_convert_bind.rs`. Four of the ten are derived from the tensors (`n_layer` from the contiguous layer run, `d_model` / `vocab` from the embedding axes, `intermediate_size` from the SwiGLU gate projection, whose second axis cross-checks `d_model`). **Behaviour change**: `--model llama-omni2` now REFUSES without `--config`, because the other six (`n_head`, `rope_max_period`, `rms_norm_eps`, `sample_rate`, `speech_encoder_dim`, `speech_decoder_dim`) cannot be read off any tensor shape and must not be synthesised (the `ModelKind::Crepe` refusal precedent). The config-less form previously "succeeded" and wrote an unloadable artifact. | `9346982` (2026-08-14), repaired 2026-08-15 |
| backfill | `vokra.m2d.*` | **8** keys — `hidden_size`, `inference_branch`, `n_mels`, `num_attention_heads`, …  | `u32` + `string` | persisted | `m2d-base`, `github.com/nttcslab/m2d` — written by `crates/vokra-convert/src/models/m2d.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `bdce8c3` (2026-08-13) |
| backfill | `vokra.maest.*` | **33** keys — `attention_dropout_scaled_1e3`, `do_normalize`, `fmax_hz`, `fmin_hz`, …  | `u32` + `bool` + `string` + `f64` | persisted | `maest-30s-pw-129e`, `mtg-upf/discogs-maest-30s-pw-129e` — written by `crates/vokra-convert/src/models/maest.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `79c3691` (2026-08-13) |
| backfill | `vokra.melotts.*` | **18** keys — `filter_channels`, `gin_channels`, `hidden_channels`, `hop_length`, …  | `u32` + `string` | persisted | `melotts` — written by `crates/vokra-convert/src/models/melotts.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.miocodec.*` | **17** keys — `upstream_revision`, `source_revision`, `model_sha256`, `config_sha256`, `sample_rate`, `n_fft`, `hop_length`, `content_dim`, `global_dim`, `wave_dim`, `code_dim`, `vocab_size`, `fsq_levels`, `wave_upsample_factors`, `wave_upsample_kernels`, `istft_padding`, `decode_only` | `string` + `u32` + `bool` + `u32[]` | persisted | `miocodec-25hz-44khz-v2` — written by `crates/vokra-convert/src/models/miocodec.rs`, read by `crates/vokra-models/src/miocodec`. **Additive**: the whole group is new and pins the exact source/checkpoint/config revisions, decode topology and versioned FSQ input contract. The audited historical public GGUF may omit the group only while its strict immutable identity/provenance/350-tensor manifest matches. | `35d3f127`, `775e1757` (2026-08-26) |
| backfill | `vokra.mp_senet.*` | **38** keys — `upstream_revision`, `reference_source`, `reference_revision`, `publication_revision`, `official_source`, `official_source_revision`, `reference_model_sha256`, `publication_model_sha256`, `reference_transformer_sha256`, `source_license`, `reference_license_sha256`, `official_license_sha256`, `model_sha256`, `config_sha256`, `public_revision`, `public_model_sha256`, `manifest_sha256`, `sample_rate`, `n_fft`, `hop_length`, `win_length`, `compress_factor`, `mask_beta`, `dense_channels`, `ts_blocks`, `attention_heads`, `gru_hidden`, `segment_size`, `attention_batch_first`, `instance_norm_eps`, `layer_norm_eps`, `stft_center`, `stft_normalized`, `stft_onesided`, `hann_periodic`, `magnitude_eps`, `phase_imag_eps`, `phase_real_eps` | `string` + `u32` + `f32` + `bool` | persisted | `mp-senet-dns` — written by `crates/vokra-convert/src/models/mp_senet.rs`, strictly read by `crates/vokra-models/src/mp_senet`. The additive group separately pins the initial weight-publication code and the later exact package segment-wrapper oracle, plus immutable upstream/official/public revisions and hashes, complete F32 manifest, frontend, topology and the released package's `attention_batch_first=false` axis behaviour. The audited historical public GGUF may omit the group while its exact 247-tensor manifest and base identity/provenance match. | working tree (2026-08-26) |
| backfill | `vokra.moss_audio_tokenizer.*` | **1** keys — `variant` | `string` | persisted | `moss-audio-tokenizer`, `OpenMOSS-Team/MOSS-Audio-Tokenizer` — written by `crates/vokra-convert/src/models/moss_audio_tokenizer.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| Mac CPU/Metal coverage — MOSS Audio | `vokra.moss_audio.*` | **42** keys — fixed model/source revisions and four SHA-256 identities; exact 15-key audio encoder contract including DeepStack layer indexes; adapter/deepstack widths; 12-key Qwen3 decoder contract; five audio/text control-token IDs | `string` + `u32` + `f32` + `bool` + `u32[]` | persisted | Dedicated `MossAudioModel` contract for `OpenMOSS-Team/MOSS-Audio-{4B,8B}-Instruct`, written by `crates/vokra-convert/src/models/moss_tts.rs` alongside the pre-existing generic `vokra.frontend.*` group. **Additive**: no MOSS-TTS key changes meaning. Corrected conversions move these two releases to `vokra.model.arch = "moss_audio"` and require their exact 901-tensor manifests; historical public files retain a wrong `moss_tts` stamp and are eligible only for exact-manifest legacy binding. | working tree (2026-08-27) |
| backfill | `vokra.moss_tts.*` | **14** keys — `audio_vocab_size`, `llm.family`, `llm.ffn_dim`, `llm.head_dim`, …  | `u32` + `f32` + `string` | persisted | `moss_tts` — written by `crates/vokra-convert/src/models/moss_tts.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.mt3.*` | **9** keys — `d_ff`, `d_kv`, `d_model`, `music_vocab_size`, …  | `u32` | persisted | `mt3-multitrack`, `github.com/magenta/mt3` — written by `crates/vokra-convert/src/models/mt3.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `382addc` (2026-08-14) |
| backfill | `vokra.neucodec.*` | **1** keys — `variant` | `string` | persisted | `neucodec`, `neuphonic/neucodec` — written by `crates/vokra-convert/src/models/neucodec.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |
| runtime-gap | `vokra.nisqa.*` | **25** keys — `source_revision`, four source/checkpoint hashes, `public_hf`, `public_revision`, `public_gguf_sha256`, `manifest_sha256`, `sample_rate`, `n_fft`, `hop_length_sec`, `win_length_sec`, `n_mels`, `fmax`, `seg_length`, `seg_hop_length`, `max_segments`, six adaptive-pool extents, `td_sa_nhead` | `string` + `u32` + `f32` | persisted | `nisqa_v2_weight` — written by the strict converter and read by `vokra-models::nisqa`. Values come from the pinned official checkpoint args, not training-YAML guesses. The exact historical 94-tensor public GGUF may omit the group only while its immutable manifest and generic non-commercial-share-alike provenance match; any partial or conflicting group fails closed. | working tree (2026-08-26) |
| backfill | `vokra.nsnet2.*` | **8** keys — `fc1_dim`, `fc2_dim`, `hidden_dim`, `hop`, …  | `u32` | persisted | `nsnet2-20ms-baseline`, `github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline` — written by `crates/vokra-convert/src/models/nsnet2.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.omniasr_ctc.*` | **20** keys — `arch.encoder.feature_dim`, `arch.encoder.feature_extractor_bias`, `arch.encoder.feature_extractor_kernel.`, `arch.encoder.feature_extractor_layer_count`, …  | `u32` | persisted | `omniasr-ctc-1b` — written by `crates/vokra-convert/src/models/omniasr_ctc.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |
| backfill | `vokra.parakeet_ctc.*` | **18** keys — `arch.encoder.attention_bias`, `arch.encoder.conv_kernel_size`, `arch.encoder.convolution_bias`, `arch.encoder.d_model`, …  | `u32` | persisted | `parakeet-ctc-1.1b` — written by `crates/vokra-convert/src/models/parakeet_ctc.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |
| backfill | `vokra.parler.*` | **17** keys — `audio_encoder.codebook_size`, `audio_encoder.sampling_rate`, `decoder.ffn_dim`, `decoder.hidden_size`, …  | `u32` + `bool` + `string` | persisted | `parler_tts` — written by `crates/vokra-convert/src/models/parler.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill / strict release identity | `vokra.pyannote.*` | **18** keys — the original 9 topology keys plus `upstream_revision`, `pyannote_audio_version`, `pyannote_audio_revision`, `tensor_manifest_sha256`, `public_hf`, `public_revision`, `public_file`, `public_bytes`, `public_sha256` | `string` + `u32` + `bool` | persisted | `pyannote-segmentation-3.0`, `pyannote/segmentation-3.0` — written by `crates/vokra-convert/src/models/pyannote_segmentation.rs`. The nine identity keys are additive. The topology value `lstm.num_layers` is corrected from the PyanNet class default `2` to the exact release override `4`, proven independently by the preserved config and l0..l3 tensor manifest; this intentionally rejects newly converted two-layer lookalikes. | `02664f6` (2026-08-06); working tree (2026-08-26) |
| backfill | `vokra.pyannote_pipeline.*` | **13** keys — `clustering.algorithm`, `clustering.method`, `clustering.min_cluster_size`, `clustering.threshold`, …  | `u32` + `f32` + `bool` + `string` | persisted | `pyannote-speaker-diarization-3.1`, `pyannote/speaker-diarization-3.1` — written by `crates/vokra-convert/src/models/pyannote_speaker_diarization_3_1.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.qwen3_asr.*` | **31** keys — 26 original topology keys (`audio.conv_chunksize`, `audio.d_model`, `audio.downsample_hidden_size`, `audio.ffn_dim`, …), `source_revision`, `tensor_manifest_sha256`, plus `audio.layer_norm_eps`, `audio.activation_function`, `audio.scale_embedding` | `u32` + `f32` + `bool` + `string` | persisted | `qwen3_asr` — written by `crates/vokra-convert/src/models/qwen3_asr.rs`. The 26-key topology group was introduced by `02664f6`; the two immutable source-identity keys and three exact audio-execution keys are additive in the 2026-08-27 waves. Historical exact public headers may omit the final three only on descriptor-only binding; executable mmap loading requires them. No pre-existing key changed meaning. | `02664f6` (topology, 2026-08-06) + (TBD) (identity/audio execution) |
| backfill | `vokra.redimnet.*` | **12** keys — `c`, `do_preemph`, `embed_dim`, `f`, …  | `u32` | persisted | `wespeaker-voxceleb-redimnet2-b6-lm`, `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM` — written by `crates/vokra-convert/src/models/redimnet.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `56581d7` (2026-08-14) |
| backfill | `vokra.rmvpe.*` | **11** keys — `base_hz`, `cents_per_class`, `fmax`, `fmin`, `hop`, `n_class`, `n_fft`, `n_mels`, `sample_rate`, `win_length`, `upstream_revision` | `u32` + `f32` + `string` | persisted | `rmvpe`, fixed `yxlllc/RMVPE` commit `0aabafba18289ca938a73af0b0297686abf4922d` — written by `crates/vokra-convert/src/models/rmvpe.rs`. The first ten keys landed in `02664f6`; `upstream_revision` is additive on 2026-08-26. The strict loader permits that key to be absent only on the exact audited historical public metadata pair after provenance is corrected to Unknown. | `02664f6` (2026-08-06); working tree (2026-08-26) |
| runtime-gap | `vokra.reazonspeech_nemo_v2.*` | **26** keys — fixed source/archive/config/tensor/tokenizer hashes, `tokenizer.vocab`, sample rate, encoder/decoder/RNN-T joint topology and decode limits | `string` + `u32` + `u8[]` | persisted | `reazonspeech-nemo-v2`, fixed `reazon-research/reazonspeech-nemo-v2` revision — written by `crates/vokra-convert/src/models/reazonspeech_nemo_v2.rs` and strictly consumed by `crates/vokra-models/src/reazonspeech_nemo_v2`. The group makes the 965-F32-tensor checkpoint executable only with its exact 3,000-piece tokenizer and pins all runtime axes. The audited historical public GGUF lacks the tokenizer/new metadata and therefore remains an explicit non-executable boundary pending an authorized gated replacement. | `b115282a` (2026-08-26); documentation repaired in working tree |
| runtime-gap | `vokra.rnnoise.*` | **11** keys — `release_tarball_sha256`, `sample_rate`, `frame_size`, `window_size`, `n_bands`, `n_features`, `conv1_width`, `hidden_size`, `n_gru`, `quantization`, `gate_order` | `u32` + `string` | persisted | Xiph RNNoise v0.2 — written by `crates/vokra-convert/src/models/rnnoise.rs`, read and cross-checked by `crates/vokra-models/src/rnnoise.rs`. The group pins the 36-array canonical network manifest and its signed-int8-in-F32-container semantics; opaque-blob artifacts are rejected. **Additive**: this replaces no existing key, but old blob-only GGUFs do not satisfy the new strict binder. | `235dca6` (2026-08-21) |
| backfill | `vokra.sbv2.*` | **32** keys — `converter_zero_defaults`, `d_bert`, `d_ff`, `d_model`, …  | `u32` + `f32` + `bool` + `string` | persisted | `sbv2-v2-multilingual-base`, `litagin02/style_bert_vits2` — written by `crates/vokra-convert/src/models/sbv2.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `f7af1ba` (2026-07-28) |
| backfill | `vokra.sepformer.*` | **2** keys — `n_out`, `variant` | `u32` + `string` | persisted | `sepformer` — written by `crates/vokra-convert/src/models/sepformer.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.snac.*` | **1** keys — `variant` | `string` | persisted | `snac-24khz`, `hubertsiuzdak/snac_24khz` — written by `crates/vokra-convert/src/models/snac.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill / strict conversion contract | `vokra.speecht5.*` | **50** keys — the original 13 topology keys plus fixed source revision/weight/393-tensor-manifest/tokenizer hashes, omitted-counter count, remaining topology/generation scalars and `tokenizer.{scheme,kind,model,pieces,scores,base_vocab_size,vocab_manifest_sha256,normalizer.*,unk_id,bos_id,pad_id,eos_id,mask_id,ctc_blank_id,vocab_size}` | `string` + `u32` + `f32` + `bool` + `u8[]` + `string[]` + `f32[]` | persisted | `speecht5-tts`, `microsoft/speecht5_tts` — written by `crates/vokra-convert/src/models/speecht5.rs`. The original 13 keys retain their meanings; 37 keys are additive. New conversion requires the fixed 393-F32 inference manifest, exact 79-piece CHAR `spm_char.model`, and fixed Hugging Face added-token IDs 79/80. The historical public GGUF lacks the tokenizer/new identity group and remains explicitly non-executable pending a gated replacement. | `02664f6` (2026-08-06); working tree (2026-08-26) |
| backfill | `vokra.storm.*` | **6** keys — `d_model`, `hop`, `n_fft`, `n_stages`, …  | `u32` | persisted | `storm`, `github.com/sp-uhh/storm` — written by `crates/vokra-convert/src/models/storm.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `82cbd3c` (2026-08-14) |
| backfill | `vokra.styletts2.*` | **12** keys — `decoder.dim_in`, `decoder.gen_istft_hop_size`, `decoder.gen_istft_n_fft`, `diffusion.steps`, …  | `u32` + `bool` + `string` | persisted | `styletts2-ljspeech-24khz` — written by `crates/vokra-convert/src/models/styletts2.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.tiger.*` | **28** keys — `variant`, `upstream_revision`, `source_revision`, `source_file_sha256`, `source_license`, `source_license_sha256`, `model_sha256`, `config_sha256`, `public_revision`, `public_model_sha256`, `manifest_sha256`, `sample_rate`, `n_fft`, `hop_length`, `feature_channels`, `internal_channels`, `num_blocks`, `num_sources`, `upsampling_depth`, `attention_heads`, `attention_hidden_channels`, `attention_kernel_size`, `attention_stride`, `band_widths`, `stft_center`, `stft_normalized`, `stft_onesided`, `hann_periodic` | `string` + `u32` + `bool` + `u32[]` | persisted | `tiger_separator` DnR/speech — written by `crates/vokra-convert/src/models/tiger.rs`, strictly read by `crates/vokra-models/src/tiger`. The original `variant` key remains unchanged; the other 27 keys are additive and pin the exact official F32 manifests, upstream/config/source/public hashes, topology, band partition and STFT convention. The audited historical public GGUF pair may omit the additive keys only when its exact immutable revision/hash/provenance/manifest tuple matches. | `02664f6` (2026-08-06); working tree (2026-08-26) |
| backfill | `vokra.vieneu.*` | **35** keys — `audio_pad_token_id`, `audio_ref_slot_token_id`, `audio_sample_rate`, `audio_tokenizer_ref`, …  | `u32` + `f32` + `bool` + `string` | persisted | `vieneu-tts-v3-turbo`, `pnnbao-ump/VieNeu-TTS-v3-Turbo` — written by `crates/vokra-convert/src/models/vieneu.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.vocos.*` | **1** keys — `variant` | `string` | persisted | `vocos` — written by `crates/vokra-convert/src/models/vocos.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.wav2vec2_ctc.*` | **16** keys — `conv_dim`, `conv_kernel`, `conv_stride`, `do_stable_layer_norm`, …  | `u32` + `f32` + `bool` + `string` | persisted | `wav2vec2_ctc` — written by `crates/vokra-convert/src/models/wav2vec2_ctc.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| backfill | `vokra.wavlm.*` | **19** keys — `conv_dim`, `conv_kernel`, `conv_stride`, `feat_extract_norm_group`, …  | `u32` | persisted | `wavlm-base-plus-sv`, `microsoft/wavlm-base-plus-sv` — written by `crates/vokra-convert/src/models/wavlm_sv.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7a0f823` (2026-08-14) |
| backfill | `vokra.yue_bundle.*` | **1** keys — `variant` | `string` | persisted | YuE bundle (`yue-upsampler` + `yue-xcodec-mini`, `m-a-p/YuE-upsampler` / `m-a-p/xcodec_mini_infer`) — written by `crates/vokra-convert/src/models/yue_bundle.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `02664f6` (2026-08-06) |
| runtime-gap | `vokra.yue_upsampler.*` | **18** keys — `upstream_revision`, `checkpoint_file`, `checkpoint_sha256`, `checkpoint_bytes`, `source_package`, `source_package_sha256`, `tensor_manifest_sha256`, `public_revision`, `public_gguf_sha256`, `public_gguf_bytes`, `sample_rate`, `input_channels`, `dim`, `intermediate_dim`, `num_layers`, `n_fft`, `hop_length`, `padding` | `string` + `u32` | persisted | `yue_upsampler` — written by the strict converter and read by `vokra-models::yue_upsampler`. The group pins the exact official 151k checkpoint, independent `vocos==0.1.0` oracle, historical public artifact and complete 81-tensor/44.1 kHz topology. The audited historical public GGUF may omit the group while its immutable manifest and generic provenance match; a partial/conflicting group fails closed. | working tree (2026-08-26) |
| backfill | `vokra.zonos.*` | **22** keys — `arch.backbone.causal`, `arch.backbone.d_intermediate`, `arch.backbone.d_model`, `arch.backbone.n_layer`, …  | `u32` + `f32` + `bool` | persisted | `zonos-v0.1` — written by `crates/vokra-convert/src/models/zonos.rs`. **Additive**: the whole group is new; no pre-existing `vokra.*` key was renamed or changed meaning. | `7ed0548` (2026-07-26) |

| runtime-gap | `vokra.emotion2vec.*` | **17** keys — `sample_rate`, `embed_dim`, `depth`, `prenet_depth`, `num_heads`, `mlp_dim`, `num_extra_tokens`, `num_classes`, `conv_pos_depth`, `conv_pos_kernel`, `conv_pos_groups`, `layer_norm_eps`, `normalize`, `conv_dim`, `conv_kernel`, `conv_stride`, `class_labels` | `u32` + `f32` + `bool` + `u32[]` + `string[]` | persisted | `emotion2vec-plus-large` — written by the strict converter from the exact official 185-F32-tensor checkpoint. The group pins the 16 kHz waveform frontend, 8 global + 4 context post-norm ALiBi blocks, grouped positional convolutions and official bilingual 9-label order. The exact historical public GGUF may omit this additive group only while its immutable manifest and generic MIT provenance match. | working tree (2026-08-26) |

**Broken cross-reference repaired by this table**: the 2026-07-24 "SoTA
Phase 2/3/4 + JA" entry above says its model chunks are "recorded here in
the GGUF metadata additions block at the bottom of this file". They were
not, and no such rows were ever added. The twelve prefixes that entry names
in its own prose are findable by grep only because it names them; its
siblings from the same waves — `vokra.chatterbox_nano.*`,
`vokra.chatterbox_turbo.*`, `vokra.parakeet_ctc.*`, `vokra.omniasr_ctc.*`,
`vokra.kyutai_stt.*` — were named nowhere at all. The rows above supply
what that entry promised.

Three prefixes that entry names are the opposite error and get **no** row
here, because no converter stamps them: `vokra.distil_whisper.*`,
`vokra.kotoba_whisper.*` and `vokra.voxcpm.*` (the shipped chunk is
`vokra.voxcpm2.*`, recorded in the 2026-07-30 VoxCPM2 section above).
Verified with `scripts/check-abi-changelog.sh --list-gguf-prefixes`, which
reports zero stamped keys under each. A row for a chunk group nothing
writes would be the same kind of defect this section exists to fix.

<!-- Template — copy into an `### YYYY-MM-DD — vX.Y.Z-dev` section per PR-day:

### 2026-07-XX — 1.0.0-rc.1-dev

| Crate / area          | Symbol                     | Kind    | Signature                                                                       | Rationale                                | Breaking? | PR   |
| --------------------- | -------------------------- | ------- | ------------------------------------------------------------------------------- | ---------------------------------------- | --------- | ---- |
| `include/vokra.h`     | `vokra_stream_interrupt`   | Added   | `enum vokra_status_t vokra_stream_interrupt(struct vokra_stream_t *stream)`     | Barge-in cancel, M3-14                    | no        | #NN  |
| `gguf:vokra.paged_kv` | `vokra.paged_kv.block_size`| Added   | `u32`                                                                           | Paged KV cache, M3-03                    | no        | #NN  |

-->

### 2026-08-11 — 1.0.0-rc.1-dev (SBV2 v2 Blocker 2b/2c/3/5 verification: test helpers + integration tests — Rust surface only, advisory)

Advisory **Rust public API** entry for the final T21 closure of the SBV2 v2
blocker verification branch (`feat/sbv2-v2-blockers-2b-2c-3-2026-08-11`).
The C ABI (`include/vokra.h`, 33 fn + 11 typedef baseline) is **untouched**
(`scripts/gen-c-abi.sh --check` = no diff); four test-helper visibility
changes and ~15 new integration/unit tests are the sole surface additions.
This is purely a **verification and integrity-check entry** — the main
SBV2 v2 functionality was landed in earlier branches.

**Test visibility additions** (Rust surface only, no C ABI impact):

| Module / Type                                   | New visibility | Rationale                                                                                                                                 | Kind  |
| ----------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `crates/vokra-models/tests/parity_sbv2_real.rs` | `pub`          | `ENV_UTMOS_ENABLE`, `ENV_UTMOS_GGUF`, `UtmosGateSettings`, `UtmosGateSettings::resolve()` — lifted to `pub` for test-suite cross-crate access within the blocker verification harness (T19 snapshot pins) | Added |

**Tests added** (no C ABI impact):

- **SPM / WordPiece scheme dispatch** (T7–T11, `crates/vokra-bert/tests/`):
  3 roundtrip tests + 2 tokenizer metadata tests verifying Blocker 5 (BERT
  tokenizer scheme selection) end-to-end behavior across `SentencePiece` and
  `WordPiece` vocabulary formats.
- **ADR (c) speaker regression** (T14, `crates/vokra-models/tests/sbv2_speaker_external.rs`):
  structural pin test confirming that external speaker-embedding wiring does
  not regress after the speaker-projection no-op inference consolidation
  (SBV2-SPK-EMB-LINEAR-DECISION ADR resolution, T13 comment-only update).
- **Flow-layer readiness verification** (T17, `crates/vokra-models/tests/sbv2_parity_atol_calibration.rs`):
  `flow_layers_are_structurally_ready_but_functionally_inert` pins three
  structural properties of the dumper's per-layer flow output: the
  manifest's `flow_layers` sibling key is present with exactly 4 entries
  (one per `TransformerCouplingLayer`), those 4 entry names are absent
  from the manifest's `tensors[]` array (the only collection the harness's
  per-tensor lookup ever resolves against), and `atol_calibration_for` has
  no match arm for any of them (still unwired). This is a "present but
  unread" pin, not a gate on the SDP flow body's forward pass.
- **UTMOS gate environment resolution** (T19, `crates/vokra-models/tests/parity_sbv2_real.rs`):
  new `utmos_gate_settings` module containing one sequential test,
  `utmos_gate_settings_env_resolution_matrix`, pinning all three
  `UtmosGateSettings::resolve()` outcomes: `VOKRA_SBV2_UTMOS_ENABLE` unset
  → `Disabled` (silent skip); `VOKRA_SBV2_UTMOS_ENABLE=1` with
  `VOKRA_SBV2_UTMOS_GGUF` unset → loud panic (FR-EX-08); both set →
  `Enabled` with the exact `VOKRA_SBV2_UTMOS_GGUF` path. The three
  scenarios run inline in one test body (rather than as three separate
  `#[test]` fns) so the shared env-var mutations can't race under
  parallel test execution.

**Code edits** (comment/test-refactor only):

- `crates/vokra-models/tests/parity_sbv2_real.rs` — all-stage aggregation
  refactor (T6) consolidating per-stage result dumps into a single
  `Vec<StageResult>` structure for cleaner comparison loop.
- `crates/vokra-models/src/sbv2/mod.rs` — SBV2-SPK-EMB-LINEAR-DECISION ADR
  comment block updated to reflect the Blocker-3 refactor resolution (T13);
  dispatch table + discard code (`let _ = projected;`) remain **unchanged**.

**GGUF metadata** (pre-existing, verified working):

- `vokra.bert.tokenizer.scheme` — recorded as `"sentencepiece-unigram"` or
  `"bert-charsplit"` depending on the BERT checkpoint type; pre-existed in
  the codebase prior to this branch, but this branch added test coverage
  verifying end-to-end load and encode behavior.
- `vokra.bert.tokenizer.{pieces, scores, unk_id, bos_id, eos_id}` — optional
  side-car keys for BERT tokenizer configuration; all additive (existing
  GGUFs without these keys continue to work).

**CI and fixtures**:

- `.github/workflows/parity-sbv2-real.yml` (T20) — advisory workflow wiring
  the end-to-end SBV2 v2 parity CI; not required, weekly + manual dispatch.
- Manifest.json refreshed with `flow_layers` sibling key (T16).
- `tools/parity/sbv2_v2_bundle_prepare_checkpoint.py` (T4) — offline converter
  helper; not reflected in C or Rust public API.
- `tools/parity/sbv2_dump_reference.py` extended for per-layer flow (T15).

**Zero-dep** (NFR-DS-02): all edits inside `vokra-bert`, `vokra-models`,
`vokra-core`; root `Cargo.lock` **unchanged** (`scripts/check-zero-deps.sh`
clean per T18).

**M5-13 relevance**: advisory Rust-surface entry only, so
`scripts/check-abi-changelog.sh` does not gate on this entry (no C symbol
changed). Snapshot rotation is the M5-13/IF-01 freeze owner's action.

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
