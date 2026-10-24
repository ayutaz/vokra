# Documentation tooling coverage — 2026-10-23

This is an audit record for the documentation-only refresh of the Markdown
files under `tools/parity/`, `scripts/`, `.github/workflows/`, and the pull
request template. It is not a model, license, parity, Apple-device, or
publication approval record.

## Scope and evidence

- Repository HEAD at audit time: `151d94a8613d5bd4acb1671a461f59742c37b8fc`.
- Inventory: 62 tracked Markdown files (`git ls-files`), checked by path and
  heading/status scan before editing.
- Current facts used for the two current snapshots were the exact local HEAD,
  the 59 files in `.github/workflows/*.yml`, and the manager-provided
  metadata-only public inventory: 194 repositories / 193 GGUF-bearing
  repositories / 198 GGUF files; CPU `136/43/14/1`; Metal `136/57/1`.
- Historical dates, hashes, benchmark values, fixture identities, and pinned
  upstream/vendor evidence were not rewritten merely because they are old.
  They remain labeled as historical or pinned evidence.
- The protected CosyVoice2 manifest was not read, hashed, staged, reverted, or
  cleaned. Held source bodies were not reread to manufacture a new source,
  license, owner, or parity conclusion.

Static checks run after the refresh: `check-doc-references.sh`,
`check-runbook-path-citations.sh`, `check-parity-sidecar-citations.sh`,
`check-workflow-hygiene.sh` (59 workflows),
`uv run --no-project --offline --python 3.12 python
tools/docs/check_doc_examples.py` (self-test and normal run), and
`git diff --check`. All passed. The first uv attempt used the restricted
default cache and failed before execution; rerunning with a temporary
`UV_CACHE_DIR` passed.

Legend for the per-file review below:

- **Current** — current/public guidance or an operational index; compared with
  the checked-in companion workflow/script and the relevant static checker.
- **Historical** — dated evidence, receipts, or a frozen status snapshot;
  retained rather than rewritten. Link/path checkers still cover it where
  applicable.
- **Vendor/fixture** — pinned source, license, fixture, or hash contract;
  hashes were not regenerated and no upstream/model was acquired.
- **Blocked boundary** — fail-closed instructions; the refresh does not imply
  a new approval or execution result.

## Per-file audit

| File | Classification | Evidence read or checker coverage | Result / limit |
|---|---|---|---|
| `.github/PULL_REQUEST_TEMPLATE.md` | Current | Template body and repository verification policy; doc/community link checks | Unchanged; generic checklist remains current. No PR state inferred. |
| `.github/workflows/README.md` | Current | Workflow definitions, local workflow inventory, workflow-hygiene gate | Updated current count, main/release snapshot, and three RC workflow rows. Historical inventories retained. |
| `scripts/install-vulkan-toolchain.md` | Current | Install commands and referenced script paths; runbook-path checker | Unchanged; no stale repository fact found. No toolchain installation run. |
| `tools/parity/README-csm.md` | Blocked boundary | CSM runbook/status text and parity-sidecar citations | Unchanged; license/VAST/owner gates remain fail-closed. No model-free VAST run. |
| `tools/parity/README-cuda-rtf-variance.md` | Current + historical | Harness paths, VAST lifecycle, dated benchmark boundary; runbook/path checkers | Added 2026-10-23 owner-run status note. Dated RTF values remain historical. |
| `tools/parity/ast/README.md` | Historical | Pinned AST oracle identity and parity-sidecar citations | Unchanged; pinned evidence was not regenerated. |
| `tools/parity/bark/README.md` | Vendor/fixture | Bark oracle contract and sidecar citations | Unchanged; no weight or license judgment changed. |
| `tools/parity/bf16_gemm/README.md` | Historical | BF16 fixture/VAST evidence and cited paths | Unchanged; numeric evidence remains tied to its dated run. |
| `tools/parity/canary_1b/README.md` | Blocked boundary | Canary preflight/runbook and companion project paths | Unchanged; no dependency sync or checkpoint access. |
| `tools/parity/canary_1b_reference/README.md` | Blocked boundary | Dedicated lock/audit instructions and sidecar citations | Unchanged; 134-row closure is pinned evidence, not a new audit. |
| `tools/parity/charsiu/README.md` | Historical | Charsiu reference contract and cited paths | Unchanged; no checkpoint or reference execution. |
| `tools/parity/clap/README.md` | Vendor/fixture | CLAP source/hash contract and citation checker | Unchanged; no dependency/source refresh. |
| `tools/parity/conv_tasnet/README.md` | Blocked boundary | Conv-TasNet runbook and parity-sidecar citations | Unchanged; source/weight holds were not bypassed. |
| `tools/parity/cosyvoice2_hift_reference/README.md` | Blocked boundary | HiFT gate instructions and path citations | Unchanged; no source or weight acquisition. |
| `tools/parity/cosyvoice2_llm_reference/README.md` | Blocked boundary | README-only gate text and sidecar citation checker | Unchanged; protected manifest was explicitly excluded from review. |
| `tools/parity/dia_1_6b_reference/README.md` | Blocked boundary | Dated dependency candidates and fail-closed VAST instructions | Unchanged; candidate evidence remains pending and historical. |
| `tools/parity/facodec/README.md` | Historical | FACodec reference contract and citations | Unchanged; no real-weight execution. |
| `tools/parity/firered_asr_aed_l/DEPENDENCY_AUDIT.md` | Historical | Frozen dependency audit sections and citation checker | Unchanged; audit rows are dated evidence, not restamped. |
| `tools/parity/firered_asr_llm_l/README.md` | Blocked boundary | Checked-in pre-acquisition gate and runbook paths | Added 2026-10-23 gate-status note. It does not infer owner approval or new acquisition evidence. |
| `tools/parity/focalcodec/README.md` | Historical | FocalCodec oracle contract and citations | Unchanged; no package import or checkpoint. |
| `tools/parity/funcodec/README.md` | Historical | FunCodec decoder contract and citations | Unchanged; no model execution. |
| `tools/parity/higgs_audio_v3_tts_4b/README.md` | Blocked boundary | Checked-in source/license refusal gate and runbook text | Added 2026-10-23 gate-status note. It does not infer a new owner/license decision. |
| `tools/parity/htdemucs_multi/README.md` | Historical | Registry identities, sidecar and runbook citations | Unchanged; registry/hash evidence remains pinned. |
| `tools/parity/irodori_text_block_reference/README.md` | Historical | Irodori reference identity and path citations | Unchanged; no source/model refresh. |
| `tools/parity/kyutai_candle_reference/README.md` | Blocked boundary | README gate/runbook text; path and sidecar checkers | Unchanged; held Candle source bodies were not reread or executed. |
| `tools/parity/magnet_medium_30secs/README.md` | Blocked boundary | VAST-only workflow and vendored-license hold text | Unchanged; no license row or model status was changed. |
| `tools/parity/magnet_small_10secs/README.md` | Blocked boundary | VAST-only workflow and vendored-license hold text | Unchanged; no license row or model status was changed. |
| `tools/parity/melodyflow_t24_30secs/README.md` | Historical | Pinned upstream identity and VAST runbook paths | Unchanged; no model acquisition or execution. |
| `tools/parity/microwakeword-reference/README.md` | Blocked boundary | Dependency evidence contract and sidecar citations | Unchanged; no dependency or model execution. |
| `tools/parity/microwakeword/README.md` | Blocked boundary | Checked-in lock/status text and validation runbook | Added 2026-10-23 status note and repaired the blockquote boundary. LiteRT parity remains unclaimed. |
| `tools/parity/miocodec/README.md` | Historical | MioCodec pinned decoder instructions and citations | Unchanged; no source or weight refresh. |
| `tools/parity/mms_1b_all/README.md` | Blocked boundary | Server-metadata audit/runbook and sidecar citations | Unchanged; no weight execution. |
| `tools/parity/moss_audio/README.md` | Blocked boundary | No-weight identity audit and runbook citations | Unchanged; source/weight holds remain. |
| `tools/parity/moss_audio/api_smoke/README.md` | Blocked boundary | Model-free API smoke/security-floor text | Unchanged; no package sync or API import. |
| `tools/parity/moss_audio_tokenizer_nano/README.md` | Blocked boundary | Tokenizer gate and license/status references | Unchanged; no owner/legal conclusion added. |
| `tools/parity/moss_audio_tokenizer_v2/README.md` | Historical | Tokenizer identity/size evidence and citations | Unchanged; byte evidence remains dated. |
| `tools/parity/moss_tts_local/README.md` | Blocked boundary | Local composite validation boundary and citations | Unchanged; no local model execution. |
| `tools/parity/mossformer2_ss_16k/README.md` | Historical | Checkpoint/source identity and path citations | Unchanged; no model acquisition. |
| `tools/parity/nanocodec/README.md` | Blocked boundary | NeMo checkpoint preparation instructions and citations | Unchanged; no NeMo import or checkpoint access. |
| `tools/parity/neucodec/README.md` | Historical | NeuCodec decoder contract and sidecar citations | Unchanged; no source/weight refresh. |
| `tools/parity/neutts_air/README.md` | Blocked boundary | VAST API-smoke and real-checkpoint boundary | Unchanged; no model execution or owner approval inferred. |
| `tools/parity/omniasr_ctc/README.md` | Current + historical | Local HEAD, release facts, metadata inventory supplied by manager, and sidecar/path checkers | Updated current main/inventory/release boundary; historical PR #79 and Apple receipts retained. |
| `tools/parity/owsm_v4_medium_1b_reference/README.md` | Blocked boundary | Dependency audit/runbook and sidecar citations | Unchanged; approval declaration remains fail-closed. |
| `tools/parity/parler_tts/README.md` | Historical | Official-reference pins and citations | Unchanged; no package/model execution. |
| `tools/parity/qwen3_asr/README.md` | Historical | Reference identity, hashes, and runbook citations | Unchanged; no source/model refresh. |
| `tools/parity/qwen3_tts/README.md` | Blocked boundary | Upstream compatibility hold and parity boundary | Unchanged; no upstream PR or license conclusion inferred. |
| `tools/parity/reazonspeech_nemo_v2/README.md` | Historical | ReazonSpeech identity and citations | Unchanged; no checkpoint or package execution. |
| `tools/parity/rmvpe/README.md` | Blocked boundary | Current lock/inspector boundary and VAST instructions | Unchanged; dependency/license gate remains blocked. |
| `tools/parity/sbv2_jp_extra/README.md` | Vendor/fixture | Authenticated source/fixture contract and path checker | Unchanged; pinned hashes and clean-room boundary retained. |
| `tools/parity/speechbrain_lang_id/README.md` | Blocked boundary | Dependency candidate and audit boundary | Unchanged; no package sync or owner approval inferred. |
| `tools/parity/speecht5_tts/README.md` | Blocked boundary | Dependency/security floor and API-smoke instructions | Unchanged; no real-weight execution. |
| `tools/parity/speechtokenizer/README.md` | Historical | Official decoder contract and citations | Unchanged; no source/model refresh. |
| `tools/parity/ultravox/README.md` | Blocked boundary | Security/API hold and parity runbook | Unchanged; unresolved advisory boundary remains. |
| `tools/parity/vendor/vits/README.md` | Vendor/fixture | Vendored file list, upstream commit, hashes, and clean-room contract | Unchanged; hashes were not regenerated. |
| `tools/parity/vendor/vits2/README.md` | Vendor/fixture | Vendored file hashes and verification commands | Unchanged; no upstream fetch or hash rewrite. |
| `tools/parity/vibevoice_realtime_0_5b_reference/README.md` | Blocked boundary | Dated VAST receipts, source pins, and citation checker | Unchanged; held Realtime source bodies were not reread. |
| `tools/parity/wespeaker/README.md` | Blocked boundary | Source reconciliation, dependency hold, and citations | Unchanged; no source/API execution. |
| `tools/parity/whisper_extras/README.md` | Historical | Apple handoff and pinned reference paths | Unchanged; no Apple run. |
| `tools/parity/xcodec2/README.md` | Blocked boundary | Draft PR boundary, dependency lock, and citations | Unchanged; open draft status remains a boundary, not an approval. |
| `tools/parity/xy_tokenizer_reference/README.md` | Blocked boundary | Dedicated lock/dependency contract and citations | Unchanged; no package sync or tokenizer execution. |
| `tools/parity/yue_xcodec_mini/README.md` | Blocked boundary | Model-free dependency evidence and `NO_UPLOAD` boundary | Unchanged; no source/model acquisition or publication. |
| `tools/parity/zonos_v0_1_reference/README.md` | Blocked boundary | Dependency license closure and approval transition text | Unchanged; broad license/owner gate remains fail-closed. |

No commit, push, PR, model download/import/forward, package installation,
Cargo build/test, VAST/Scaleway allocation, or publication was performed for
this audit.
