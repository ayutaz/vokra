# Vokra ABI Changelog (v0.9 window)

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
  enforces this: if the current `include/vokra.h` differs from
  `docs/abi/vokra.h.v0.9-baseline.symbols` and this file does not have an
  entry dated today, the script exits non-zero.
- At v1.0 GA (M5-13; M4-12 before the 2026-07-14 reassignment) the baseline
  is re-anchored to that release, the freeze commitment is written into
  `include/vokra.h`, and post-1.0 breaking changes require a major bump.

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

This snapshot is what `scripts/check-abi-changelog.sh` diffs the working-tree
`include/vokra.h` against. It was captured on the merge day of PR #3
(2026-07-08, M2 rollup) and is the anchor for the entire v0.9 window.

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

## Entries

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

## GGUF Metadata additions (non-C-ABI, informational)

The following GGUF metadata chunks were added during the M3 waves. **These
are model-file (`.gguf`) additions only, NOT part of the C ABI surface** —
`include/vokra.h` does not expose any GGUF key by name, so
`scripts/check-abi-changelog.sh` does not gate on them. This section is
informational and prepares the M3-16 changelog for the M4-12 v1.0 GA
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

### v1.0-rc window (M4) — GGUF metadata additions

| WP    | Chunk prefix                   | Keys                                                                                                                                                                                                             | Kind          | Status      | Rationale                                                                                                                                                                              | Introducing wave / commit |
| ----- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| M4-04 | `vokra.dac.*`                  | `vokra.dac.{n_codebooks,codebook_size,codebook_dim,d_model,sample_rate,hop_length}` (config-side-car-driven; the zoo-primary 24 kHz / 8 kbps variant is `32 / 1024 / 8 / 1024 / 24000 / 320` — verified from the release checkpoint metadata, ADR M4-04 §T02). Companion **derived tensor names** in the same namespace: `vokra.dac.quantizer.{i}.{codebook,out_proj_weight,out_proj_bias}` (weight-norm folded offline). | `u32`         | persisted   | DAC factorized RVQ decode attributes — written by `crates/vokra-convert/src/models/dac.rs` (`convert_dac_file`), read by `DacCodecGguf::from_gguf` (`crates/vokra-models/src/codec.rs`) into `DacRvqAttrs` + `DacOutProj`s. Frame rate = `sample_rate / hop_length` (75 Hz for the primary variant → paged `BlockSize::Four`). | M4-04 wave 1               |
| M4-05 | `vokra.csm.*`                  | `vokra.csm.{sample_rate,frame_rate_mhz}`, `vokra.csm.arch.backbone.{n_layer,d_model,n_head_q,n_head_kv,ffn_dim}`, `vokra.csm.arch.depth.{n_layer,d_model,n_head_q,n_head_kv,ffn_dim}`, `vokra.csm.arch.{rms_norm_eps,rope_base,n_ctx}`, `vokra.csm.rope.{scale_factor,low_freq_factor,high_freq_factor,old_context_len}` (Llama-3 scaled RoPE — torchtune `Llama3ScaledRoPE`, ADR M4-05 §D3), `vokra.csm.audio.{n_codebooks,vocab_size}`, `vokra.csm.text.vocab_size`. Flavor dims / rates / RoPE params are primary-source transcriptions (`SesameAILabs/csm` `models.py`/`generator.py`); the two vocab axes are `0`-placeholders until the T29 gated checkpoint (runtime rejects `0` at load, FR-EX-08). `vokra.tokenizer.model` (u8-array) is **reused** (M2-06 Whisper / M3-10 Voxtral key, not a new key) for the Llama-3.2 tokenizer blob. `frame_rate_mhz` is milli-Hz integer anchoring (12.5 Hz → `12500`, no f32 drift). | `u32` + `f32` | persisted   | Sesame CSM-1B (S2S) architecture attributes — written by `crates/vokra-convert/src/models/csm.rs` (`convert_file` / `convert_csm_file`), read by `CsmConfig::from_gguf` (`crates/vokra-models/src/csm/config.rs`). No `vokra.frontend.*` chunk: CSM has no mel frontend (audio enters via the Mimi encoder) — ADR M4-05 §D9 records the omission decision. | M4-05 wave 1               |
| M4-05 | `vokra.mimi.seanet.*` / `vokra.mimi.quantizer.*` / `vokra.mimi.transformer.*` (+ `vokra.mimi.{sample_rate,frame_rate_mhz}`) | `vokra.mimi.seanet.{dimension,n_filters,n_residual_layers,kernel_size,residual_kernel_size,last_kernel_size,compress,dilation_base,n_ratios}` + indexed `vokra.mimi.seanet.ratio.{i}` (count + indexed keys — the `vokra.quant.rule.*` precedent, no GGUF-array plumbing), `vokra.mimi.quantizer.{dimension,n_q,bins,input_dimension,output_dimension}`, `vokra.mimi.transformer.{d_model,n_head,n_layer,ff_dim,context,max_period,layer_scale}`. Values are `kyutai-labs/moshi` `loaders.py` transcriptions (ADR M4-05 §D2). **Distinct from** the M3-06/M4-04 `vokra.mimi.{n_codebooks,codebook_size,d_model}` RVQ-table triple (same namespace, different sub-prefixes — no key collision). | `u32` + `f32` | persisted   | Mimi **neural chain** (encoder audio→RVQ + neural decoder features→PCM) shape attributes for the shared `crates/vokra-models/src/mimi/` module (M4-05 lands, M4-06 Moshi consumes) — written by the CSM converter, read by `MimiNeuralConfig::from_gguf`. The Mimi *weights* travel in the standalone M4-04 mimi GGUF (CC-BY 4.0, NOTICE). | M4-05 wave 1               |
| M4-16 | `vokra.wavtokenizer.*`         | `vokra.wavtokenizer.vocab_size`, `vokra.wavtokenizer.d_model` — 1:1 with `WavTokenizerVqAttrs` (`crates/vokra-ops/src/fsq_codec.rs`). Released WavTokenizer configs = `4096 / 512` (upstream `vq_bins: 4096` / `dimension=512`, `num_quantizers: 1` — ADR M4-16 §D-c, verified 2026-07-15). | `u32`         | documented  | FSQ-family (FR-OP-31) single-codebook VQ decode attributes — read contract fixed by the `WavTokenizerVqAttrs` rustdoc; converter-side emission (`documented` → `persisted`) lands with the real WavTokenizer model-integration WP. **M5-13 freeze: to be declared `EXPERIMENTAL`** (`docs/handoff/m4-12.md` §(e)-2) so schema evolution stays legal at minor bumps until the codec API stabilizes — this row is the M4-16 intent record; the marker itself is burned in at M5-13 (v1.0 GA tag). | M4-16 (2026-07-15)         |
| M4-16 | `vokra.xcodec2.*`              | `vokra.xcodec2.levels` (`u32` array), `vokra.xcodec2.d_model` — 1:1 with `Xcodec2FsqAttrs` (`crates/vokra-ops/src/fsq_codec.rs`). Released X-Codec 2 checkpoint = levels `[4; 8]` (effective vocab 4^8 = 65536) / `d_model 2048` (`vq_dim`; upstream `vq/codec_decoder_vocos.py` + `modeling_xcodec2.py`, pin `vector-quantize-pytorch==1.17.8` — ADR M4-16 §D-c, verified 2026-07-15). | `u32-array` + `u32` | documented  | FSQ-family (FR-OP-31) finite-scalar-quantization dequant attributes (implicit grid + out-projection GEMV; **separate subgraph from the RVQ `vokra.mimi.*` / `vokra.dac.*` chunks** — no cross-codebook axis). Converter-side emission lands with the real X-Codec 2 model-integration WP. **M5-13 freeze: to be declared `EXPERIMENTAL`** (handoff §(e)-2) — same intent record as the `vokra.wavtokenizer.*` row above. | M4-16 (2026-07-15)         |

| M4-06 | `vokra.moshi.*`                | `vokra.moshi.arch.temporal.{n_layer,d_model,n_head,ffn_hidden}`, `vokra.moshi.arch.depth.{n_layer,d_model,n_head,ffn_hidden}`, `vokra.moshi.arch.{rms_norm_eps,rope_max_period,context,max_ctx}`, `vokra.moshi.audio.{n_q_in,dep_q,card}`, `vokra.moshi.text.{card,pad_id,end_pad_id}`, `vokra.moshi.n_delays` + indexed `vokra.moshi.delay.{i}` (count + indexed keys — the `vokra.mimi.seanet.ratio.{i}` precedent). Shape-driven where derivable (layer counts / widths / gating hidden / stream tallies / vocabs from the T02 355-tensor manifest); head counts / ε (1e-8 `rms_norm_f32`) / max_period (10000) / context (3000) / pad ids (3, 0) / delays (`[0,0,1×7,0,1×7]` structural rule — 7B verbatim) are `_lm_kwargs` transcriptions (ADR M4-06 §D2/§D3). Audio rates deliberately absent — the shared `vokra.mimi.*` chunk is the single rate authority (§D3 no-duplication rule; `quantizer.n_q = max(dep_q, n_q−dep_q)`, `bins ≡ card` per loaders.py). `vokra.tokenizer.model` reused for the raw SentencePiece blob; `vokra.provenance.attribution` (new provenance key) carries the FR-MD-09 display text. | `u32` + `f32` + `string` | persisted | Moshi (Helium temporal + depformer, full-duplex S2S) architecture attributes — written by `crates/vokra-convert/src/models/moshi.rs` (`convert_moshi_file`), read by `MoshiConfig::from_gguf` (`crates/vokra-models/src/moshi/config.rs`). BF16 checkpoint tensors are decoded to F32 **exactly** at conversion (`GgmlType::BF16 = 30` read support). | M4-06 (2026-07-15)         |

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
| M4-18 | `vokra.utmos.*` | `vokra.utmos.arch.variant` (`"wav2vec2_regression.v0"` guard), `vokra.utmos.sample_rate`, `vokra.utmos.conv.{channels[],kernels[],strides[],activation}`, `vokra.utmos.transformer.{n_layer,n_head,hidden_dim,ffn_dim,norm,ln_eps}`, `vokra.utmos.head.{dims[],pool,scale,offset}` (`scale`/`offset` optional, identity defaults) | `string` + `u32` + `u32-array` + `f32` | documented | UTMOS scorer config (M4-18, weight-deferred skeleton) — read by `UtmosConfig::from_gguf` in `crates/vokra-eval/src/metrics/utmos.rs`; required keys have no silent defaults, an unknown `arch.variant` is rejected loudly (FR-EX-08). Converter-side emission (`vokra-convert --model utmos`, T05) lands with the owner weight flip (v1.0.x patch); until then only the in-crate round-trip test writes the schema. | M4 Wave 1                  |

Note: `vokra.dnsmos.*` is **reserved but deliberately not designed** — DNSMOS is license fail-closed until the owner's M4-18 T03 verification (no keys are invented ahead of it).

<!-- Template — copy into an `### YYYY-MM-DD — vX.Y.Z-dev` section per PR-day:

### 2026-07-XX — 0.9.0-dev

| Crate / area          | Symbol                     | Kind    | Signature                                                                       | Rationale                                | Breaking? | PR   |
| --------------------- | -------------------------- | ------- | ------------------------------------------------------------------------------- | ---------------------------------------- | --------- | ---- |
| `include/vokra.h`     | `vokra_stream_interrupt`   | Added   | `enum vokra_status_t vokra_stream_interrupt(struct vokra_stream_t *stream)`     | Barge-in cancel, M3-14                    | no        | #NN  |
| `gguf:vokra.paged_kv` | `vokra.paged_kv.block_size`| Added   | `u32`                                                                           | Paged KV cache, M3-03                    | no        | #NN  |

-->

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

These are the pre-freeze anchor artefacts M3-16 produces / references.
M4-12 diffs the v1.0 header + Rust surface against these to build the
"0.9 → 1.0 delta" summary:

- **`docs/abi/vokra.h.v0.9-baseline.symbols`** — the v0.9-window anchor
  used by `scripts/check-abi-changelog.sh` during the M3 window. Captured
  at PR #3 merge (2026-07-08) per the "Baseline snapshot" section above.
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
policy is time-boxed to the v0.9 window, and M4-12 formally revokes it
at freeze time. Nothing in M3-16 fires the freeze.
