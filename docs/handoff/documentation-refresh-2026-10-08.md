# Documentation refresh — 2026-10-08

## Scope and source boundary

Documentation branch: `docs/update-documentation-20261008`, starting at exact
GitHub `main` `d100d93778191ccab77bd1c37fe3552e3d889758`. This audit
supersedes current-status pointers in the [2026-10-04 review](documentation-refresh-2026-10-04.md),
not its dated measurements, evidence hashes, owner decisions or historical
per-file review ledger. [The documentation index](../README.md) is the current
public entry point.

The starting inventory is **322 tracked Markdown/MDX files**. Review is split
among three independent documentation agents and the manager: core public
guides and workflow/governance records; tutorials, bindings, integration and
reference-tool instructions; historical handoffs and benchmark records; and
the current index, agent policy and audit integration. The new report itself
is additional to that starting inventory. Fixture text and verbatim licenses
are evidence, not prose to replace with current measurements or new grants.

The review uses checked-in source, generated surfaces and static gates plus
read-only public metadata. A 26-guide example checker is useful but does not
by itself prove whole-repository documentation coverage. Source comparison,
per-area review and the tracked-file link inventory are separate checks.

## Authoritative read-only snapshot

| Source | 2026-10-08 observation | What it does not prove |
|---|---|---|
| GitHub `main` API | `d100d93778191ccab77bd1c37fe3552e3d889758` | Unmerged PR code is not in this baseline. |
| [GitHub release](https://github.com/ayutaz/vokra/releases/tag/v0.3.0) | `v0.3.0`, published 2026-09-20, 18 assets | Desktop PR #199 completeness or external registry publication. |
| All-pages public HF API plus revision-pinned model cards, classified by `tools/audit/hf_mac_coverage.py` | 194 repositories / 193 GGUF repositories / 198 GGUF files; CPU full 136, partial 43, no binder 14, non-artifact 1; Metal full 136, CPU-blocked 57, non-artifact 1; 58 unresolved rows | New real-weight parity, whole-catalog Apple completion or an upload approval. |
| GitHub branch-protection API | 16 required contexts, `strict=true`; linear history and conversation resolution required | A setting change or bypass approval. |
| All-pages Dependabot API | 283 open alerts: 254 with patched versions, 29 without; severity 3 critical / 81 high / 106 medium / 93 low | Applied fixes, exploitable runtime scope or dismissed alerts. |
| All-pages Code Scanning API | Four open Scorecard findings: `CIIBestPracticesID`, `CodeReviewID`, `SASTID`, `VulnerabilitiesID` | Remediation or waiver. |
| Open-PR API | 16 open PRs, 15 drafts; #199 non-draft and unmerged | Permission to merge, or CI evidence for a different head. |

PR #152's current head is `e22913d9aaf6e9fc26659e55637dce5e6458f978`;
the former `b0add994` head's CI snapshot is historical. Draft legal PR #198
and desktop PR #199 are not integrated by this documentation review.

Mac completion remains separated into conversion, binding, native forward,
independent parity, exact-head Apple CPU/Metal/no-fallback checks and
publication. The [accepted 2026-09-11 Apple batch](mac-cpu-metal-scaleway-results-2026-09-11.md)
is valid only for its named heads and contracts, including OmniASR. The
separate four-artifact publication approval does not expand that verification
scope. The frozen 63-row owner ledger and 53/29 M5 checklist remain historical
or requirement-specific denominators, not replacements for the live 58-row
code-classification count.

## Review and verification status

Implementation, model execution, cloud allocation and publication are outside
this documentation-only task. No model weights were downloaded or run, no
workspace or `vokra-models` Cargo command was run, and no provider, branch
protection, alert or PR state was changed. The protected
`tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json` is not read,
edited, staged, reverted or included in the diff review.

The repository-wide initial relative-link inspection covered 322 tracked
Markdown/MDX files and 734 relative Markdown links outside fenced examples.
It found the not-yet-created report pointer and six distinct targets that
exist only in ignored local ADR/ticket files. Local existence must not be
mistaken for availability in a public clone; public instructions use tracked
alternatives and historical local references are labeled as such.

After correction, the full link inspection includes this new report:
**323 Markdown/MDX documents / 734 relative links / zero missing or
local-only public targets**. The finished diff updates 53 existing Markdown
files and adds this report; there are no implementation-file changes.

All 322 starting Markdown/MDX paths are accounted for by the review lanes
below. The manager reviewed the delegated diffs and sent back two substantive
corrections: the released Swift manifest must not be conflated with current
`main`, and OmniASR's pre-Scaleway pending note must not erase its later
bounded Apple PASS. Both corrections were incorporated and rechecked.

| Review lane | Files | Source/contract checks |
|---|---:|---|
| Manager: policy and current index | 22 | AGENTS and skill twins, compliance `Strict` default, current `HotOp`/unsafe boundaries, explicit unsupported/no-fallback rules, metadata and API receipts. |
| Core public guides | 20 | Cargo/version/MSRV, header/Python prototype equality, architecture anchors, CLI flags and English/Japanese consistency. |
| Workflow/governance and M2/M3/M4 records | 20 | 47 actual workflows, 36 schedules, protected-branch contexts, one tag/release, tracked review-record inventory; preserve original device measurements. |
| Tutorial, integration, binding and reference instructions | 146 | Swift current/tag manifests; iOS scripts; server config/API limits; Godot trampolines/CI; Unity C# and packaging; web exports/COOP/COEP; Python metadata and UV/dumper contracts. |
| Legal/ABI/vendor policy boundaries | 5 | No new sign-off or EULA-compliance guarantee; real CUDA no-cuDNN implementation and SDK-gated QNN limitations; ABI ledger coverage. Current-law reauthentication is explicitly incomplete. |
| Dated handoff, benchmark and design records | 109 | Preserve exact heads, source/packet hashes, numerical values, owner decisions and named-hardware scopes; supersede misleading live pointers, not their historical evidence. |

The static checks below passed on the integrated documentation diff. They
are not runtime, compiler, hardware or deployment-law tests:

- `tools/docs/check_doc_examples.py` and `--self-test`: 140 blocks across
  26 guides; 30 Tier-C UI/Swift/C#/Godot examples remain explicitly unverified.
- `scripts/check-doc-references.sh` and `--self-test`: 110 requirement IDs,
  33 architecture anchors and public English/Japanese pairs.
- `scripts/check-runbook-path-citations.sh`: 1,713 citations / 694 distinct
  paths after adding this inventory; absent/ignored evidence is
  labeled, not manufactured.
- `scripts/check-community-docs.sh`, `scripts/check-parity-sidecar-citations.sh`,
  `scripts/check-owner-checklist-drift.sh`, `scripts/check-platform-support.sh`
  and `scripts/check-workflow-hygiene.sh`: PASS.
- `scripts/check-abi-changelog.sh`: all changed C symbols and 124
  converter-stamped GGUF prefixes covered, historical anchor unchanged.
- `scripts/check-zero-deps.sh`, `scripts/check-forbidden-symbols.sh`,
  `scripts/publish/signoff_match.py --self-test`,
  `scripts/publish/check-catalog-reality.sh` (22 rows / zero known gaps) and
  `scripts/check-codex-hooks.sh`: PASS.
- Bundled `skill-creator/scripts/quick_validate.py`: all 16 repository skill
  documents valid; the eight Codex/legacy twins remain byte-identical.
- `git diff --check` with the protected-manifest exclusion: PASS.

The external legal-source limitation below and the Tier-C examples are not
silently counted as freshly verified. Documentation correction does not
perform the remaining model, security remediation, PR or hardware tasks.

## External-source limitations

Re-access to the official California AB 2713 and SB 1000 pages returned HTTP
403 through the read-only web client. EUR-Lex required a browser verification
step; the in-app browser was unavailable. Therefore the historical legal
verification date is not advanced as though current law had been authenticated.
Draft [PR #198](https://github.com/ayutaz/vokra/pull/198) records a separate
legal-text correction and unresolved automated official-link gate; this review
does not waive that gate or claim the draft has merged. Deployment decisions
still require current official sources and owner/legal review, not the old
SB 942 threshold or marking summary alone.

## Second-pass omissions and consolidation

The additional review on the same `d100d937` code baseline follows the first
three documentation commits (`f3452023`, `446e86df`, `5df96912`). The initial
pass counts and receipts above remain historical; this section records the
additional findings rather than rewriting that evidence. This pass changes
38 existing Markdown files plus this audit record, with no implementation or
workflow-definition changes.

Source comparison identified the missing fifth CLI subcommand
`npu-bakeoff`, the authenticated CoreML sidecar contract, incomplete server
configuration rows, and server security claims that exceeded production
wiring. The corrected server documentation distinguishes Wyoming session
scheduling from HTTP concurrency, route-local body limits from uniform error
responses, and the reusable panic helper from its still-missing production
HTTP attachment. Built-in authentication, HTTP timeout/admission control and
global HTTP panic isolation are not claimed as delivered. Proxy recipes are
illustrative, authenticated deployment inputs, not deployment-tested results.

Other corrections cover soxr's LGPL classification, missing reference input
filenames, a JSON fragment mislabeled as a complete document, four relative
heading links, relocated source/test paths, and the distinct Voxtral runtime
mmap and converter streaming contracts. Historical hashes, measurements,
license grants and owner decisions remain unchanged.

The unnecessary duplicated content was consolidated as follows:

| Removed duplicate | Canonical source retained |
|---|---|
| Embedded collector implementation in the metrics runbook | `scripts/kill-switch-metrics.sh` |
| Repeated blank quarterly review template | `docs/governance/quarterly-reviews/README.md` and `vokra-go-nogo-v0.5.md` |
| Unauthenticated proxy recipe repeated in the server README | `integrations/vokra-server/docs/security-ops.md` |

No whole historical document was deleted. Exact-content comparison found
only the eight intentional Codex/legacy skill twin pairs. GPU/CUDA handoffs,
the two HF gap inventories, benchmark records and reusable bakeoff templates
contain distinct evidence or callers; age, size and a newer summary are not
sufficient grounds to discard them. Their historical/current boundary is
clarified where needed. Ignored local planning files are not force-added.

The expanded example-checker diagnostic covered 184 documents and 385 fenced
blocks. It is not a whole-repository green gate: unsupported languages,
generated outputs, historical recipes and CLI argument-position heuristics
need manual classification. Genuine filename/fragment errors were corrected;
unsupported examples were not relabeled to manufacture a pass. The collector
still has documented pagination/event-timestamp limitations; documentation
consolidation does not fix those implementation limitations or make an owner
Go/No-go decision.

Final lightweight checks before recording these changes:

- Whole tracked Markdown/MDX link probe: 323 documents, 765 relative links,
  183 heading-fragment links, zero missing tracked targets or fragment
  candidates. Heading normalization is a static approximation, not a rendered
  GitHub navigation test.
- Standard example checker and self-test: PASS; 144 blocks across 26 guides,
  with the same 30 Tier-C examples explicitly unverified.
- Documentation references and self-test, runbook paths, community docs,
  parity-sidecar citations, owner-checklist drift, platform support, ABI
  changelog, workflow hygiene, Codex hooks and agent contracts: PASS.
- Zero-dependency and forbidden-symbol checks, collector self-test,
  VAST skill twin comparison and protected-manifest-excluding diff check: PASS.

The workflow checker initially could not write its default UV cache inside
the sandbox; rerunning with the existing temporary UV cache passed, without
changing permissions or dependencies. No model execution, heavy Cargo build,
provider operation, live metrics collection, publication or PR mutation was
performed. The external-law and hardware limits recorded above still apply.

## Per-file coverage inventory

This is the exact starting tracked Markdown/MDX inventory assigned to the review
lanes above (322 files). Dated fixture/history files are reviewed for
scope, source/citation continuity and misleading current-state claims; their
original measurements and grants are retained, not rerun.

### Manager — policy/current index (22)

```text
.agents/skills/add-audio-operator/SKILL.md
.agents/skills/add-speech-model/SKILL.md
.agents/skills/complete-mac-cpu-metal/SKILL.md
.agents/skills/license-audit/SKILL.md
.agents/skills/numerical-parity/SKILL.md
.agents/skills/publish-model-to-hf/SKILL.md
.agents/skills/refresh-vokra-docs/SKILL.md
.agents/skills/vast-ai-workflow/SKILL.md
.claude/skills/add-audio-operator/SKILL.md
.claude/skills/add-speech-model/SKILL.md
.claude/skills/complete-mac-cpu-metal/SKILL.md
.claude/skills/license-audit/SKILL.md
.claude/skills/numerical-parity/SKILL.md
.claude/skills/publish-model-to-hf/SKILL.md
.claude/skills/refresh-vokra-docs/SKILL.md
.claude/skills/vast-ai-workflow/SKILL.md
AGENTS.md
CODE_OF_CONDUCT.ja.md
CODE_OF_CONDUCT.md
SECURITY.ja.md
SECURITY.md
docs/README.md
```

### Core-guide agent — workflows/governance/history (20)

```text
.github/PULL_REQUEST_TEMPLATE.md
.github/workflows/README.md
assets/branding/README.md
docs/governance/dod-judgment-template.md
docs/governance/exit-path-playbook.md
docs/governance/kill-switch-metrics-runbook.md
docs/governance/quarterly-review-runbook.md
docs/governance/quarterly-reviews/README.md
docs/governance/vokra-go-nogo-v0.5.md
docs/m2-14-ios-rtf-handover.md
docs/m2-cuda-rtf-variance-2026-07-08.md
docs/m2-cuda-rtf-variance-template.md
docs/m2-owner-verification-checklist.md
docs/m3-11-godot-demo-handover.md
docs/m3-15-server-latency-handover.md
docs/m3-18-android-rtf-handover.md
docs/m3-owner-verification-checklist.md
docs/m4-07-hopper-bench-handover.md
docs/m4-owner-verification-checklist.md
docs/m4-scope-expansion-2026-07-13.md
```

### Core-guide agent — public guides (20)

```text
CHANGELOG.md
CONTRIBUTING.md
README.ja.md
README.md
docs/api-reference.ja.md
docs/api-reference.md
docs/architecture.ja.md
docs/architecture.md
docs/backend-guide.ja.md
docs/backend-guide.md
docs/c-api-streaming-codec.md
docs/getting-started.ja.md
docs/getting-started.md
docs/good-first-tasks.ja.md
docs/good-first-tasks.md
docs/migration-guide.ja.md
docs/migration-guide.md
docs/nanocodec-conversion.md
docs/requirement-ids.ja.md
docs/requirement-ids.md
```

### Integration agent — tutorials/bindings/reference guides (146)

```text
README-swift-package.md
bindings/python/README.md
bindings/unity/com.vokra.unity/CHANGELOG.md
bindings/unity/com.vokra.unity/LICENSE.md
bindings/unity/com.vokra.unity/README.md
bindings/unity/com.vokra.unity/Samples~/VadAsrTts/README.md
bindings/unity/com.vokra.unity/TESTING.md
crates/vokra-backend-vulkan/kernels/README.md
crates/vokra-backend-vulkan/kernels/precompiled/README.md
crates/vokra-backend-webgpu/kernels/README.md
crates/vokra-core/tests/fixtures/rng_torch/README.md
crates/vokra-core/tests/parity/fixtures/m2-08/README.md
crates/vokra-core/tests/parity/fixtures/m5-06/README.md
crates/vokra-kws-micro/README.md
crates/vokra-models/src/fsmn_vad/SPEC.md
crates/vokra-models/src/kokoro/data/README.md
crates/vokra-models/src/silero_vad/SPEC.md
crates/vokra-models/tests/fixtures/focalcodec/README.md
crates/vokra-models/tests/fixtures/lang_id/README.md
crates/vokra-models/tests/fixtures/melotts_chinese/README.md
crates/vokra-models/tests/fixtures/melotts_english/README.md
crates/vokra-models/tests/fixtures/melotts_japanese/README.md
crates/vokra-models/tests/fixtures/melotts_korean/README.md
crates/vokra-models/tests/fixtures/melotts_spanish/README.md
crates/vokra-models/tests/fixtures/metricgan_plus/README.md
crates/vokra-models/tests/fixtures/neucodec/README.md
crates/vokra-models/tests/fixtures/parakeet_ctc/README.md
crates/vokra-models/tests/fixtures/sepformer/README.md
crates/vokra-models/tests/fixtures/wavtokenizer/README.md
crates/vokra-models/tests/fixtures/whisper_medusa/README.md
crates/vokra-models/tests/fixtures/xcodec2/README.md
crates/vokra-ops/tests/fixtures/flow_sampler/README.md
docs/platform-support/cdn-selection-material.md
docs/platform-support/v1.0-rc-support-matrix.md
docs/tutorials/android.ja.md
docs/tutorials/android.md
docs/tutorials/cli.ja.md
docs/tutorials/cli.md
docs/tutorials/godot.ja.md
docs/tutorials/godot.md
docs/tutorials/ios.ja.md
docs/tutorials/ios.md
docs/tutorials/python.ja.md
docs/tutorials/python.md
docs/tutorials/server.ja.md
docs/tutorials/server.md
docs/tutorials/unity.ja.md
docs/tutorials/unity.md
docs/tutorials/web.ja.md
docs/tutorials/web.md
examples/unity-demo/Assets/Plugins/README.md
examples/unity-demo/Assets/StreamingAssets/models/README.md
examples/unity-demo/ProjectSettings/README.md
examples/unity-demo/README.md
examples/unity-demo/TESTING.md
integrations/vokra-android/README.md
integrations/vokra-cli-bench-server/README.md
integrations/vokra-godot/README.md
integrations/vokra-godot/demos/README.md
integrations/vokra-misaki-g2p/README.md
integrations/vokra-piper-g2p/README.md
integrations/vokra-server-bench/README.md
integrations/vokra-server/README.md
integrations/vokra-server/docs/adr-http-stack.md
integrations/vokra-server/docs/scope.md
integrations/vokra-server/docs/security-ops.md
integrations/vokra-server/docs/wyoming-design.md
integrations/vokra-server/tests/wyoming-ha-smoke.md
scripts/install-vulkan-toolchain.md
tests/capi/README.md
tests/fixtures/audio/README.md
tests/fixtures/audiobox_aesthetics/README.md
tests/fixtures/audioseal/README.md
tests/fixtures/sbv2/README.md
tests/parity/README.md
tests/parity/bf16_gemm/README.md
tests/parity/canary_1b_flash/README.md
tests/parity/canary_1b_v2/README.md
tests/parity/conv2d/README.md
tests/parity/fcpe/README.md
tests/parity/fixtures/m1-02/kquant/README.md
tests/parity/fsq/nanocodec/README.md
tests/parity/piper_plus/README.md
tests/parity/silero_vad/README.md
tests/parity/utmos/README.md
tests/parity/vocoder_conv/README.md
tools/bench/ios-device/README.md
tools/parity/README-csm.md
tools/parity/README-cuda-rtf-variance.md
tools/parity/ast/README.md
tools/parity/bark/README.md
tools/parity/bf16_gemm/README.md
tools/parity/canary_1b/README.md
tools/parity/canary_1b_reference/README.md
tools/parity/charsiu/README.md
tools/parity/clap/README.md
tools/parity/conv_tasnet/README.md
tools/parity/cosyvoice2_hift_reference/README.md
tools/parity/cosyvoice2_llm_reference/README.md
tools/parity/dia_1_6b_reference/README.md
tools/parity/facodec/README.md
tools/parity/firered_asr_aed_l/DEPENDENCY_AUDIT.md
tools/parity/firered_asr_llm_l/README.md
tools/parity/focalcodec/README.md
tools/parity/funcodec/README.md
tools/parity/higgs_audio_v3_tts_4b/README.md
tools/parity/htdemucs_multi/README.md
tools/parity/irodori_text_block_reference/README.md
tools/parity/magnet_medium_30secs/README.md
tools/parity/magnet_small_10secs/README.md
tools/parity/melodyflow_t24_30secs/README.md
tools/parity/microwakeword-reference/README.md
tools/parity/microwakeword/README.md
tools/parity/miocodec/README.md
tools/parity/mms_1b_all/README.md
tools/parity/moss_audio/README.md
tools/parity/moss_audio/api_smoke/README.md
tools/parity/moss_audio_tokenizer_nano/README.md
tools/parity/moss_audio_tokenizer_v2/README.md
tools/parity/moss_tts_local/README.md
tools/parity/mossformer2_ss_16k/README.md
tools/parity/nanocodec/README.md
tools/parity/neucodec/README.md
tools/parity/neutts_air/README.md
tools/parity/omniasr_ctc/README.md
tools/parity/owsm_v4_medium_1b_reference/README.md
tools/parity/parler_tts/README.md
tools/parity/qwen3_asr/README.md
tools/parity/qwen3_tts/README.md
tools/parity/reazonspeech_nemo_v2/README.md
tools/parity/rmvpe/README.md
tools/parity/sbv2_jp_extra/README.md
tools/parity/speechbrain_lang_id/README.md
tools/parity/speecht5_tts/README.md
tools/parity/speechtokenizer/README.md
tools/parity/ultravox/README.md
tools/parity/vendor/vits/README.md
tools/parity/vendor/vits2/README.md
tools/parity/vibevoice_realtime_0_5b_reference/README.md
tools/parity/wespeaker/README.md
tools/parity/whisper_extras/README.md
tools/parity/xcodec2/README.md
tools/parity/xy_tokenizer_reference/README.md
tools/parity/yue_xcodec_mini/README.md
tools/parity/zonos_v0_1_reference/README.md
web/pkg/README.md
```

### Legal/ABI agent — source and license boundaries (5)

```text
docs/abi-changelog.md
docs/legal-compliance.md
docs/license-audit.md
third_party/NVIDIA-EULA.md
third_party/QUALCOMM-QNN-NOTES.md
```

### History agent — dated records (109)

```text
docs/bench-baselines/README.md
docs/bench-baselines/awaiting-real-weight-2026-07-22/report.md
docs/bench-baselines/eval-cache-artifacts-2026-07-19/README.md
docs/bench-baselines/m1-real-weight-eval-2026-07-16/report-campaign2.md
docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md
docs/bench-baselines/m4-05-csm-fixture-reference.md
docs/bench-baselines/m4-19-wyoming-real-gguf-2026-07-19/report.md
docs/bench-baselines/m5-01-coreml-bakeoff-2026-08-24/README.md
docs/bench-baselines/m5-02-qnn-bakeoff-2026-08-24/README.md
docs/bench-baselines/m5-14-final-2026-07-18/report.md
docs/bench-baselines/m5-14-wave0-2026-07-18/hotspot-tables.md
docs/bench-baselines/m5-14-wave0-2026-07-18/target-table.md
docs/bench-baselines/metal-transcript-parity-2026-07-19/report.md
docs/bench-baselines/rtf-decomposed-2026-07-08.report.md
docs/bench-baselines/rtf-fa-v2-2026-07-08.report.md
docs/bench-baselines/sbom-reproducibility-2026-07-19/report.md
docs/bench-baselines/server-real-gguf-slots-2026-07-21/report.md
docs/bench-baselines/silero-8k-ctx288-2026-07-19/report.md
docs/bench-baselines/vast-2026-07-10/rtf-decomposed.report.md
docs/bench-baselines/vast-2026-07-10/rtf-fa-v2.report.md
docs/bench-baselines/vast-2026-08-10-h100/README.md
docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-decomposed.report.md
docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-fa-v2.report.md
docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-fa-v3.report.md
docs/bench-baselines/web-2026-07-15/README.md
docs/bench-baselines/x-06-preverify-2026-07-20/README.md
docs/bench-baselines/x-06-preverify-2026-07-20/asr-wer-summary.m1.md
docs/benchmarks/v0.5-device-benchmarks.md
docs/benchmarks/v0.5-device-runbook.md
docs/benchmarks/v0.9-device-benchmarks.md
docs/design/firered-pcm-beam-composite.md
docs/design/kyutai-stt-pcm-composite.md
docs/design/kyutai-stt-streaming-lm.md
docs/design/m0-03-gguf-loader.md
docs/design/mimi-checkpoint-provenance.md
docs/design/mimi-rust-core-metadata-binding.md
docs/design/quantization-policy.md
docs/design/size-budget.md
docs/design/vokra-gguf-chunks.md
docs/handoff/audit-followup-2026-08-14.md
docs/handoff/codex-operations-2026-08-18.md
docs/handoff/codex-operations-2026-08-28.md
docs/handoff/cosyvoice3-soxr-route-2026-09-08.md
docs/handoff/coverage-audit-2026-08-03-wave-a.md
docs/handoff/documentation-refresh-2026-10-04.md
docs/handoff/hf-audio-gap-2026-07-30.md
docs/handoff/hf-audio-gap-comprehensive-2026-07-30.md
docs/handoff/m4-02.md
docs/handoff/m4-11.md
docs/handoff/m4-12.md
docs/handoff/m4-15.md
docs/handoff/m4-18.md
docs/handoff/m4-19.md
docs/handoff/m5-01-coreml-bakeoff-2026-08-24.md
docs/handoff/m5-01-coreml-bakeoff-template.md
docs/handoff/m5-02-qnn-bakeoff-2026-08-24.md
docs/handoff/m5-02-qnn-bakeoff-template.md
docs/handoff/m5-02.md
docs/handoff/m5-03.md
docs/handoff/m5-04.md
docs/handoff/m5-06.md
docs/handoff/m5-13.md
docs/handoff/mac-cpu-metal-completion-plan-2026-08-30.md
docs/handoff/mac-cpu-metal-coverage-2026-08-24.md
docs/handoff/mac-cpu-metal-execution-plan-2026-09-07.md
docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md
docs/handoff/mac-cpu-metal-owner-decision-record-2026-09-09.md
docs/handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md
docs/handoff/mac-cpu-metal-residual-execution-2026-09-12.md
docs/handoff/mac-cpu-metal-scaleway-results-2026-09-11.md
docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md
docs/handoff/model-publish-and-parity-2026-07-28.md
docs/handoff/parity-ci-flip-switch.md
docs/handoff/parity-deberta-v3-large-real.md
docs/handoff/parity-deepfilternet3-real.md
docs/handoff/parity-sbv2-real-vast-2026-08-18.md
docs/handoff/parity-speechbrain-lang-id-real.md
docs/handoff/post-audit-2026-08-13-summary.md
docs/handoff/public-catalog-security-completion-2026-09-29.md
docs/handoff/publish-unhandled-2026-07-28.md
docs/handoff/pyannote-implementation-plan-2026-07-30.md
docs/handoff/readback-branch-closeout-2026-10-04.md
docs/handoff/release-dry-run-2026-08-22.md
docs/handoff/release-preparation-0.3.0-2026-09-17.md
docs/handoff/remaining-work-plan-2026-08-20.md
docs/handoff/residual-wave3-2026-07-30.md
docs/handoff/runtime-gap-execution-plan-2026-08-21.md
docs/handoff/sbv2-bug4-resolved-2026-08-09.md
docs/handoff/sbv2-parity-owner-handoff-2026-08-11.md
docs/handoff/sbv2-sdp-debug-2026-08-08.md
docs/handoff/sbv2-sdp-vast-parity.md
docs/handoff/security-remediation-2026-09-21.md
docs/handoff/sota-candidates-2026-07-25.md
docs/handoff/tier1-tier2-audio-impl-2026-07-30.md
docs/handoff/vast-ai-execution-priority.md
docs/handoff/vast-ai-large-model-publish.md
docs/handoff/vast-ai-publish-firered-asr-llm-l.md
docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md
docs/handoff/vast-ai-publish-rmvpe.md
docs/handoff/vast-ai-publish-voxcpm2-2b.md
docs/handoff/vast-ai-vocoder-cuda-kernels.md
docs/handoff/vast-ai-vocoder-gpu-kernels.md
docs/handoff/workflow-python-uv-migration-2026-08-18.md
docs/handoff/x-06-breach-response.md
docs/handoff/x-06-requirement-revision.md
docs/handoff/x-06.md
docs/handoff/x-07.md
docs/handoff/x-10.md
docs/m5-owner-verification-checklist.md
```
