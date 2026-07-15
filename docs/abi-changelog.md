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

### 2026-07-09 — 0.9.0-dev

| Crate / area                    | Symbol                                        | Kind  | Signature                                                                   | Rationale                                                                                                                 | Breaking? | PR    |
| ------------------------------- | --------------------------------------------- | ----- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------- | ----- |
| `include/vokra.h`               | `vokra_stream_interrupt`                      | Added | `enum vokra_status_t vokra_stream_interrupt(struct vokra_stream_t *stream)` | Barge-in / cancel (FR-ST-03), WP M3-14                                                                                    | no        | (TBD) |
| `gguf:vokra.voxtral.adapter.*`  | `vokra.voxtral.adapter.{kind,tensor_prefix,in_dim,out_dim,has_bias,has_layernorm,activation,time_stride,weight_name,bias_name,layernorm_gamma_name,layernorm_beta_name,mlp_hidden_dims,mlp_layer_names}` | Added | Kind = `string` \| dims = `u32` \| flags = `bool` \| names = `string` (see `crates/vokra-models/src/voxtral/adapter.rs` for the loader) | Voxtral audio-adapter (encoder → soft-prefix) framework — M3-10 Wave 8 (real ASR conditioning; absent = LM-continuation) | no        | (TBD) |

### 2026-07-15 — 0.9.0-dev (M4-20: audio dialect op subset)

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
| M3-03 | `vokra.paged_kv.*`             | `vokra.paged_kv.block_size` (proposed; RVQ codec paths use `block_size = 4`, LLM decode paths use `block_size = 2` per CLAUDE.md §"paged KV cache")                                                                | `u32`         | documented  | Paged KV cache `[time, stream, codebook]` 3D layout. Converter-side emission lands with the M3-06 mimi_rvq / M3-09 CosyVoice2 wiring (M3-03-native paths use the runtime default today). | Wave 2                     |
| M3-04 | `vokra.kv_quant.*`             | `vokra.kv_quant.format` (proposed; `"q4_0"` / `"q5_0"` / `"q8_0"` / absent = fp32/fp16 native), `vokra.kv_quant.block_size` (proposed; per-format tile size)                                                       | `string` + `u32` | documented | KV cache quantization discriminator. Persistence lands when the converter has weights whose scheme differs from `Q4_K/Q5_K/Q6_K` (which are model-weight quants, not KV-cache quants).   | Wave 2 / Wave 6            |
| M3-06 | `vokra.mimi.*`                 | `vokra.mimi.n_codebooks` (canonical `8`), `vokra.mimi.codebook_size` (canonical `2048`), `vokra.mimi.d_model` (canonical `512`)                                                                                    | `u32`         | documented  | Static shape attributes for the Mimi RVQ decoder — read by `MimiRvqAttrs` in `crates/vokra-ops/src/mimi_rvq.rs` (see docstring L116–117). Converter-side emission lands with M3-09.       | Wave 3                     |
| M3-07 | `vokra.hifigan.*`              | `vokra.hifigan.{initial_channel, n_upsample_stages, n_mrf_branches, conv_pre_kernel, conv_post_kernel, upsample_kernels[], upsample_strides[]}` + per-stage MRF descriptors                                        | `u32` + array | documented  | HiFi-GAN generator arch attributes — read by `HifiGanWeights` in `crates/vokra-ops/src/hifigan.rs` (see docstring L136–142). Converter-side emission lands when a dedicated HiFi-GAN converter or the M3-09 CosyVoice2 converter writes it. | Wave 3                     |
| M3-09 | `vokra.cosyvoice2.*`           | `vokra.cosyvoice2.sample_rate` (`24000`), `vokra.cosyvoice2.arch.{vocab_size,hidden_dim,n_layer,n_head,ffn_dim}`, `vokra.cosyvoice2.flow.{nfe,schedule}`, `vokra.cosyvoice2.mimi.{n_codebooks,codebook_size,d_model}`, `vokra.cosyvoice2.streaming.{chunk_size,chunk_hop}` | `u32` + `string` | persisted  | CosyVoice2 architecture / Flow Matching / Mimi codec / streaming attributes — written by `crates/vokra-convert/src/models/cosyvoice2.rs` and read by `crates/vokra-models/src/cosyvoice2/mod.rs`. `flow.schedule` values: `"linear"` / `"sway"` / `"epss"` (M3-05 flow_sampler). | Wave 5                     |
| M3-10 | `vokra.voxtral.audio_encoder.*` | `vokra.voxtral.audio_encoder.{n_layer,n_head,hidden_dim,n_mels}`                                                                                                                                                  | `u32`         | persisted   | Voxtral audio encoder (Whisper-family arch) attributes — written by `crates/vokra-convert/src/models/voxtral.rs`, read by `crates/vokra-models/src/voxtral/`.                            | Wave 5                     |
| M3-10 | `vokra.voxtral.text_decoder.*`  | `vokra.voxtral.text_decoder.{n_layer,hidden_dim,ffn_dim,vocab_size}`                                                                                                                                              | `u32`         | persisted   | Voxtral Mistral-family text decoder attributes.                                                                                                                                          | Wave 5                     |
| M3-10 | `vokra.voxtral.mode`           | `vokra.voxtral.mode`                                                                                                                                                                                             | `string`      | persisted   | Voxtral mode discriminator: `"asr"` (audio → text) or `"s2s"` (speech-to-speech scaffold). Read by `crates/vokra-convert/src/main.rs::convert_voxtral_file`.                             | Wave 5                     |
| M3-10 | `vokra.voxtral.adapter.*`      | (see the C-ABI-adjacent entry above under `## Entries` → 2026-07-09 → `gguf:vokra.voxtral.adapter.*`)                                                                                                             | mixed         | persisted   | Audio-adapter framework — the primary changelog entry lives in the `## Entries` section above so both C-ABI and GGUF views find it; the row here cross-references only.                  | Wave 8                     |
| M4-20 | `vokra.denoise.*`              | `vokra.denoise.{n_fft,hop,sample_rate,n_erb,hidden,df_bins,df_order}` (`u32`) + flat F32 tensors `vokra.denoise.{encoder,erb_decoder,df_decoder}.{weight,bias}` — read by `DenoiseModel::from_gguf` / written by `DenoiseModel::to_gguf_bytes` (`crates/vokra-ops/src/denoise.rs`) | `u32` + `f32` tensors | persisted (synthetic path) | DeepFilterNet `denoise` (FR-OP-61) config + neural-scaffold tensors. The synthetic converter (`convert_denoise_synthetic`) writes/reads this today; the **real** DeepFilterNet checkpoint → tensors mapping is owner (T17). | M4-20 (c)                  |
| M4-20 | `vokra.whisper.alignment_heads`| `vokra.whisper.alignment_heads` — OPTIONAL flat `[layer0,head0,layer1,head1,…]` `u32` pair array; read by `WhisperConfig::from_gguf` into `alignment_heads`. Absent → word timestamps fail explicitly (FR-EX-08). | `u32-array`   | documented  | Whisper cross-attention DTW alignment heads (FR-OP-40 word timestamps). Model-specific data (not fabricated); converter-side emission is owner (real `model.alignment_heads` blob).      | M4-20 (a)                  |

Notes:

- **Existing baseline keys** (already stable pre-M3, not repeated here): `vokra.frontend.*`, `vokra.whisper.*`, `vokra.piper.*`, `vokra.campplus.*`, `vokra.tokenizer.model`, `vokra.provenance.*`, `vokra.quant.default_scheme` / `vokra.quant.rule_count`, `vokra.model.name` / `vokra.model.arch`. See ADR-0001 §"vokra.* namespace" (planning doc) for the pre-M3 chunk set.
- **Namespace policy** (unchanged): every Vokra-specific chunk lives under the `vokra.*` prefix; llama.cpp-compatible chunks (e.g. `general.*`) are honored in read but the writer never emits them under the `vokra.*` namespace. This keeps a `.gguf` interoperable with llama.cpp inspection tools while giving Vokra its own reserved namespace (CLAUDE.md L146 / "vokra-audio dialect" clause).
- **Removal rule**: a v0.9.x chunk MAY be renamed / removed pre-1.0 without a major bump, but a `documented` → `removed` transition must land a row here even though the C-ABI gate is silent about it. This is the honest-report contract for the pre-freeze window (mirrors the C-ABI pre-1.0 policy above).

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
