# Whole-document refresh audit (2026-10-04)

Status: **documentation refresh complete within the boundaries below**.
The user requested that all documentation be brought
up to date. This record covers all **315 tracked Markdown/MDX files** at
review-start HEAD `3a3fd82281ae3ef9e995cb212ec794531ed13493`, not only the
26 public files covered by the example checker. Ignored local planning files
are not forced into Git. A path assignment is not a completed content review.

**2026-10-07 California legal-source reconciliation boundary:** The legal
section's 2026-08-30 SB 942 facts and this 2026-10-04 audit remain historical
records. The [California Secretary of State's 2025 chapter index](https://admin.cdn.sos.ca.gov/bill-chapters/2025/Chapter-Number.pdf)
(p. 27, row 0674) verifies AB0853 / Chapter 674 enactment metadata; the
[Governor's 2026-09-30 release](https://www.gov.ca.gov/2026/09/30/californias-nation-leading-ai-framework-just-got-stronger-governor-newsom-signs-more-first-in-the-nation-worker-protections-and-more/)
lists AB 2713 and SB 1000 as signed California AI Transparency Act bills; and
the [2026-04-21 Senate Judiciary Committee analysis](https://sjud.senate.ca.gov/system/files/2026-04/sb-1050-ashby-sjud-analysis.pdf)
(p. 2) is only committee analysis, not enacted/current full text. The required
[official SB 942 bill-text endpoint](https://leginfo.legislature.ca.gov/faces/billVersionsCompareClient.xhtml?bill_id=202320240SB942)
returned 403 for the full body on 2026-10-07. These sources do not establish current
applicability, timetable, compliance, or owner/legal approval. The [official
2026-08-03 Senate press statement](https://sd13.senate.ca.gov/news/press-release/august-3-2026/californias-landmark-ai-transparency-law-took-effect-august-1)
says implementation began on August 1 and describes later January 1/2028 phases,
which differs from the April analysis; neither replaces the chaptered statutory
body or current AB 2713 / SB 1000 text. Mirrors are not substituted and the
pending authoritative reconciliation remains fail-closed. The 2026-10-07
normal-browser read is retained as a separate dated observation for the AB 2713
chaptered body, the SB 1000 chaptered body, and the current BPC Chapter 25
display; it is not a fresh current-law reauthentication. The historical SB 942
403 observation remains unchanged; applicability, compliance, CI gates, and
owner/legal decisions remain pending.

**2026-10-07 official chaptered-text browser read:** A normal Chrome rendering
also exposed the official chaptered text for [AB 2713](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260AB2713#93CHP)
(`09/30/26 - Chaptered`, Chapter 856, approved/filed 2026-09-30; BPC
§22757.3.1 subdivision (e) operative 2027-01-01), [SB 1000](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260SB1000#94CHP)
(`09/30/26 - Chaptered`, Chapter 861, approved/filed 2026-09-30; SEC. 8
urgency/immediate effect), and the [current BPC Chapter 25 display](https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?article=&chapter=25.&division=8.&lawCode=BPC&part=&title=).
The rendered text supplies current version and section boundaries, including
the SB 1000 changes to §§22757.1, 22757.2, 22757.3, 22757.4, 22757.4.1, and
22757.5 (including deletion of the old one-million threshold and old manifest
option obligation) and the current display's Stats 2026 Ch. 861 effective
2026-09-30 / §22757.6 operative 2026-08-02 markers. The same
code display shows §22757.3.1 as AB 0853 / Stats 2025 Ch. 674, operative
2027-01-01; that code view must not be conflated with the AB 2713 chaptered
bill text. This normal browser read does not erase the dated HTTP 403 record,
waive any CI gate, or establish Vokra applicability, compliance, or
owner/legal approval; the authoritative legal reconciliation remains pending.

## Verified current snapshot

- GitHub main: `97447185361a37af64c1b30fe87e8e2618d96e20`, committed
  2026-09-30, read through the API on 2026-10-04. This is distinct from the
  local review-start HEAD and unmerged candidates.
- Latest GitHub release: `v0.3.0`, published 2026-09-20T04:52:18Z,
  18 assets. Workspace Cargo metadata remains version 0.3.0, edition 2024,
  rust-version 1.85; this is not a GA/ABI-freeze declaration.
- Metadata-only Hugging Face audit on 2026-10-04:
  194 public repositories, 193 GGUF-bearing repositories, 198 GGUF files;
  CPU full 136 / partial 43 / no-runtime-binder 14 / non-artifact 1;
  Metal full 136 / CPU-blocked 57 / non-artifact 1. The 58 unresolved rows
  and code/artifact classifications are not actual Apple completion verdicts.
  The audit fetched only organization metadata and revision-pinned README
  text, not model configuration, tokenizers or weights.
- All-pages Dependabot read: 209 open, 182 with a patched version and 27
  without. Code Scanning: three open Scorecard findings. Readback does not
  remediate or dismiss an alert.
- PR #152 remains OPEN/DRAFT at
  `b0add994b4d0350c338e60cb29b8165d8145fe32`, 76 successful / three skipped
  checks. Model-free green CI does not resolve package/license or model gates.
- M5 literal Markdown boxes remain 53 checked / 29 unchecked. Prose-only
  GA conditions remain outside that count.

## Review policy and scope

Current public recipes must match source, generated ABI/bindings and scripts;
English/Japanese twins and repository links are checked together. Dated
measurements, signed scopes, release notes and legal primary texts retain
original facts. A current pointer or explicit supersession note may be added
without changing historical hardware, hashes, test totals or approvals.
Agent instructions are operational policy, not evidence of model completion.
The protected CosyVoice2 LLM owner manifest is never read, edited, staged or
included in diffs.

No local model execution, model downloads, broad Cargo verification,
cloud allocation, publication, push or PR mutation is part of this refresh.
Static example checks validate limited source/header surfaces; unavailable
Swift/C#/Godot/runtime/model tiers must remain explicitly unexecuted.

## Per-file coverage ledger

All 315 original paths have full-content review dispositions and applicable
source evidence. Root accepted the delegated diffs after corrections. The new
audit record itself is additionally reviewed, making 316 tracked Markdown/MDX
files after this refresh. An unchanged historical record is not a new runtime,
legal or hardware PASS.

| Tracked file | Review scope | Disposition |
|---|---|---|
| `.agents/skills/add-audio-operator/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.agents/skills/add-speech-model/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.agents/skills/complete-mac-cpu-metal/SKILL.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `.agents/skills/license-audit/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.agents/skills/numerical-parity/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.agents/skills/publish-model-to-hf/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.agents/skills/refresh-vokra-docs/SKILL.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `.agents/skills/vast-ai-workflow/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.claude/skills/add-audio-operator/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.claude/skills/add-speech-model/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.claude/skills/complete-mac-cpu-metal/SKILL.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `.claude/skills/license-audit/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.claude/skills/numerical-parity/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.claude/skills/publish-model-to-hf/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.claude/skills/refresh-vokra-docs/SKILL.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `.claude/skills/vast-ai-workflow/SKILL.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `.github/PULL_REQUEST_TEMPLATE.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `.github/workflows/README.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `AGENTS.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `CHANGELOG.md` | management/history/policy | content/source reviewed; retained unchanged |
| `CODE_OF_CONDUCT.ja.md` | public core | reviewed against local source; retained unchanged |
| `CODE_OF_CONDUCT.md` | public core | reviewed against local source; retained unchanged |
| `CONTRIBUTING.md` | public core | reviewed against local source; retained unchanged |
| `README-swift-package.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `README.ja.md` | public core | reviewed against local source; updated, root diff accepted |
| `README.md` | public core | reviewed against local source; updated, root diff accepted |
| `SECURITY.ja.md` | public core | reviewed against local source; retained unchanged |
| `SECURITY.md` | public core | reviewed against local source; retained unchanged |
| `assets/branding/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `bindings/python/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `bindings/unity/com.vokra.unity/CHANGELOG.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `bindings/unity/com.vokra.unity/LICENSE.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `bindings/unity/com.vokra.unity/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `bindings/unity/com.vokra.unity/Samples~/VadAsrTts/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `bindings/unity/com.vokra.unity/TESTING.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `crates/vokra-backend-vulkan/kernels/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-backend-vulkan/kernels/precompiled/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-backend-webgpu/kernels/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-core/tests/fixtures/rng_torch/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `crates/vokra-core/tests/parity/fixtures/m2-08/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-core/tests/parity/fixtures/m5-06/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-kws-micro/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `crates/vokra-models/src/fsmn_vad/SPEC.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/src/kokoro/data/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/src/silero_vad/SPEC.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/focalcodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/lang_id/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/melotts_chinese/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/melotts_english/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `crates/vokra-models/tests/fixtures/melotts_japanese/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/melotts_korean/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/melotts_spanish/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/metricgan_plus/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/neucodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/parakeet_ctc/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `crates/vokra-models/tests/fixtures/sepformer/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `crates/vokra-models/tests/fixtures/wavtokenizer/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `crates/vokra-models/tests/fixtures/whisper_medusa/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `crates/vokra-models/tests/fixtures/xcodec2/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `crates/vokra-ops/tests/fixtures/flow_sampler/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `docs/README.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `docs/abi-changelog.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/api-reference.ja.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/api-reference.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/architecture.ja.md` | public core | reviewed against local source; retained unchanged |
| `docs/architecture.md` | public core | reviewed against local source; retained unchanged |
| `docs/backend-guide.ja.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/backend-guide.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/bench-baselines/README.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/awaiting-real-weight-2026-07-22/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/eval-cache-artifacts-2026-07-19/README.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m1-real-weight-eval-2026-07-16/report-campaign2.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m4-05-csm-fixture-reference.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m4-19-wyoming-real-gguf-2026-07-19/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m5-01-coreml-bakeoff-2026-08-24/README.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m5-02-qnn-bakeoff-2026-08-24/README.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m5-14-final-2026-07-18/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m5-14-wave0-2026-07-18/hotspot-tables.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/m5-14-wave0-2026-07-18/target-table.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/metal-transcript-parity-2026-07-19/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/rtf-decomposed-2026-07-08.report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/rtf-fa-v2-2026-07-08.report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/sbom-reproducibility-2026-07-19/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/server-real-gguf-slots-2026-07-21/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/silero-8k-ctx288-2026-07-19/report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/vast-2026-07-10/rtf-decomposed.report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/vast-2026-07-10/rtf-fa-v2.report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/vast-2026-08-10-h100/README.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-decomposed.report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-fa-v2.report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-fa-v3.report.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/web-2026-07-15/README.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/x-06-preverify-2026-07-20/README.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/bench-baselines/x-06-preverify-2026-07-20/asr-wer-summary.m1.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/benchmarks/v0.5-device-benchmarks.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/benchmarks/v0.5-device-runbook.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/benchmarks/v0.9-device-benchmarks.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/c-api-streaming-codec.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/design/m0-03-gguf-loader.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/design/quantization-policy.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/design/size-budget.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/design/vokra-gguf-chunks.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/getting-started.ja.md` | public core | reviewed against local source; retained unchanged |
| `docs/getting-started.md` | public core | reviewed against local source; retained unchanged |
| `docs/good-first-tasks.ja.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/good-first-tasks.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/governance/dod-judgment-template.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/governance/exit-path-playbook.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/governance/kill-switch-metrics-runbook.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/governance/quarterly-review-runbook.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/governance/quarterly-reviews/README.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/governance/vokra-go-nogo-v0.5.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/handoff/audit-followup-2026-08-14.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/codex-operations-2026-08-18.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/codex-operations-2026-08-28.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/cosyvoice3-soxr-route-2026-09-08.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/coverage-audit-2026-08-03-wave-a.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/hf-audio-gap-2026-07-30.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/hf-audio-gap-comprehensive-2026-07-30.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m4-02.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m4-11.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m4-12.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m4-15.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m4-18.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m4-19.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-01-coreml-bakeoff-2026-08-24.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-01-coreml-bakeoff-template.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-02-qnn-bakeoff-2026-08-24.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-02-qnn-bakeoff-template.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-02.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-03.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/m5-04.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-06.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/m5-13.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/mac-cpu-metal-completion-plan-2026-08-30.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/mac-cpu-metal-coverage-2026-08-24.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/mac-cpu-metal-execution-plan-2026-09-07.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/mac-cpu-metal-owner-decision-record-2026-09-09.md` | management/history/policy | content/source reviewed; retained unchanged |
| `docs/handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/handoff/mac-cpu-metal-residual-execution-2026-09-12.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/mac-cpu-metal-scaleway-results-2026-09-11.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/handoff/model-publish-and-parity-2026-07-28.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/parity-ci-flip-switch.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/parity-deberta-v3-large-real.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/parity-deepfilternet3-real.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/parity-sbv2-real-vast-2026-08-18.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/parity-speechbrain-lang-id-real.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/post-audit-2026-08-13-summary.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/public-catalog-security-completion-2026-09-29.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/handoff/publish-unhandled-2026-07-28.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/pyannote-implementation-plan-2026-07-30.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/release-dry-run-2026-08-22.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/release-preparation-0.3.0-2026-09-17.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/remaining-work-plan-2026-08-20.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/residual-wave3-2026-07-30.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/runtime-gap-execution-plan-2026-08-21.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/sbv2-bug4-resolved-2026-08-09.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/sbv2-parity-owner-handoff-2026-08-11.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/sbv2-sdp-debug-2026-08-08.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/sbv2-sdp-vast-parity.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/security-remediation-2026-09-21.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `docs/handoff/sota-candidates-2026-07-25.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/tier1-tier2-audio-impl-2026-07-30.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/vast-ai-execution-priority.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/vast-ai-large-model-publish.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/vast-ai-publish-firered-asr-llm-l.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/vast-ai-publish-rmvpe.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/vast-ai-publish-voxcpm2-2b.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/vast-ai-vocoder-cuda-kernels.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/vast-ai-vocoder-gpu-kernels.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/workflow-python-uv-migration-2026-08-18.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/x-06-breach-response.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/x-06-requirement-revision.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/handoff/x-06.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/handoff/x-07.md` | management/history/policy | content/source reviewed by root; current status reconciled |
| `docs/handoff/x-10.md` | management/history/policy | content reviewed; dated pointer added, root accepted |
| `docs/legal-compliance.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/license-audit.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/m2-14-ios-rtf-handover.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m2-cuda-rtf-variance-2026-07-08.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m2-cuda-rtf-variance-template.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m2-owner-verification-checklist.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/m3-11-godot-demo-handover.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m3-15-server-latency-handover.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m3-18-android-rtf-handover.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m3-owner-verification-checklist.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/m4-07-hopper-bench-handover.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m4-owner-verification-checklist.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/m4-scope-expansion-2026-07-13.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/m5-owner-verification-checklist.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/migration-guide.ja.md` | public core | reviewed against local source; retained unchanged |
| `docs/migration-guide.md` | public core | reviewed against local source; retained unchanged |
| `docs/nanocodec-conversion.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/platform-support/cdn-selection-material.md` | management/history/policy | content reviewed; history/policy retained |
| `docs/platform-support/v1.0-rc-support-matrix.md` | management/history/policy | content/source reviewed; corrected and root diff accepted |
| `docs/requirement-ids.ja.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/requirement-ids.md` | public core | reviewed against local source; updated, root diff accepted |
| `docs/tutorials/android.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/android.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/cli.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/cli.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/godot.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/godot.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/ios.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/ios.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/python.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/python.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/server.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/server.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/unity.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/unity.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/web.ja.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `docs/tutorials/web.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `examples/unity-demo/Assets/Plugins/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `examples/unity-demo/Assets/StreamingAssets/models/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `examples/unity-demo/ProjectSettings/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `examples/unity-demo/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `examples/unity-demo/TESTING.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-android/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-cli-bench-server/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-godot/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-godot/demos/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-misaki-g2p/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-piper-g2p/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-server-bench/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-server/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-server/docs/adr-http-stack.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-server/docs/scope.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-server/docs/security-ops.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-server/docs/wyoming-design.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `integrations/vokra-server/tests/wyoming-ha-smoke.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `scripts/install-vulkan-toolchain.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/capi/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `tests/fixtures/audio/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/fixtures/audiobox_aesthetics/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/fixtures/audioseal/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/fixtures/sbv2/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tests/parity/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tests/parity/bf16_gemm/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/canary_1b_flash/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/canary_1b_v2/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/conv2d/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/fcpe/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/fixtures/m1-02/kquant/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/fsq/nanocodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/piper_plus/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tests/parity/silero_vad/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/utmos/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tests/parity/vocoder_conv/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `third_party/NVIDIA-EULA.md` | management/history/policy | content reviewed; history/policy retained |
| `third_party/QUALCOMM-QNN-NOTES.md` | management/history/policy | content reviewed; history/policy retained |
| `tools/bench/ios-device/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |
| `tools/parity/README-csm.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/README-cuda-rtf-variance.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/ast/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/bark/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/bf16_gemm/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/canary_1b/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/canary_1b_reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/charsiu/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/clap/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/conv_tasnet/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/cosyvoice2_hift_reference/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/cosyvoice2_llm_reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/dia_1_6b_reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/facodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/firered_asr_aed_l/DEPENDENCY_AUDIT.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/firered_asr_llm_l/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/focalcodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/funcodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/higgs_audio_v3_tts_4b/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/htdemucs_multi/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/irodori_text_block_reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/magnet_medium_30secs/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/magnet_small_10secs/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/melodyflow_t24_30secs/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/microwakeword-reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/microwakeword/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/miocodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/mms_1b_all/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/moss_audio/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/moss_audio/api_smoke/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/moss_audio_tokenizer_nano/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/moss_audio_tokenizer_v2/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/moss_tts_local/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/mossformer2_ss_16k/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/nanocodec/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/neucodec/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/neutts_air/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/omniasr_ctc/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/owsm_v4_medium_1b_reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/parler_tts/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/qwen3_asr/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/qwen3_tts/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/reazonspeech_nemo_v2/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/rmvpe/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/sbv2_jp_extra/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/speechbrain_lang_id/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/speecht5_tts/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/speechtokenizer/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/ultravox/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/vendor/vits/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/vendor/vits2/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/vibevoice_realtime_0_5b_reference/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/wespeaker/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/whisper_extras/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/xcodec2/README.md` | tools/fixtures | content/source reviewed; corrected and root diff accepted |
| `tools/parity/xy_tokenizer_reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/yue_xcodec_mini/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `tools/parity/zonos_v0_1_reference/README.md` | tools/fixtures | content/source reviewed; retained unchanged |
| `web/pkg/README.md` | platform/bindings | content/source reviewed; root accepted; historical evidence retained |

## Validation

The final integrated lightweight run passed examples (140 blocks across 26
docs, 30 explicitly deferred blocks), example self-tests, documentation
references and their 17 self-tests, runbook paths (1,430 citations across 123
records), community documents, parity-sidecar citations (468 citations across
1,236 source files), owner checklist drift (28 anchors), platform anchors (53),
ABI changelog (124 converter prefixes and changed C symbols), workflow hygiene
(47 workflows / 36 crons), Codex hook checks and self-tests, agent contracts,
and Apple worker contract self-tests (55 scripts / zero syntax or self-test
failures). The six updated source/legacy skill pairs are byte-identical.
Negative-case diagnostics in the self-tests are expected; all checkers exited 0.
`git diff --check` also passed, excluding the protected owner manifest.

These are bounded static/contract checks, not runtime/toolchain/device execution.
The default uv cache was sandbox-denied before execution; reruns with a
task-specific writable cache passed. Per-file content review, not these static
checks alone, establishes the complete documentation coverage below.

## Completed content review and remaining external boundaries

The 21 public-core paths have source-backed per-file review evidence and
root-accepted diffs. The 45 platform/binding paths also have source-backed review and root-accepted
diffs. Root corrections distinguish the tag's local Swift manifest from the
later URL patch, GitHub's four wheel assets from unpublished PyPI/TestPyPI,
and the release's absence of GGUF assets from Unity's required verified model
URLs. Documentation review dates remain distinct from device evidence dates.
The tools/fixtures scope contains 101 paths with Luna per-file content review.
Root reviewed the 20 changed-file diffs and accepted the final working-directory
and path corrections for flow sampler, WavTokenizer, Whisper Medusa and Parler
TTS. Frozen uv/Python 3.12, remote-model boundaries and pending publication
gates are explicit; the recipes were not executed against models.
Management/history/policy scope contains 148 original paths. The historical corpus review lists
107 exact paths, including 13 miscellaneous policy/handover files; nine
operational records received root-accepted current pointers. The remaining
41 paths comprise 10 root-reviewed policy/current records, 11 long-form
canonical records, eight checklist/governance records and 12 operational skill
documents. All were read through EOF, with source-backed review and accepted
corrections; header-only reads are not counted as content review. The 10,005-line
catalog and 4,372-line ABI history were included. Historical 131/63 statements
and immutable 63-row approval packets are explicitly distinguished from the
fresh 136/58 metadata inventory, not overwritten.

Coverage totals: **21 public core + 45 platform/binding + 101 tools/fixtures +
148 management/history/policy = 315 reviewed original files; zero unreviewed
coverage rows**. The additional audit document records the scope and limits.

The legal refresh attempted official-source reads. EU/California texts were
not revalidated because the official endpoints challenged/timed out. Keep
their dated legal-source verification and explicit pending current-law
boundary; do not infer a new legal sign-off from a document refresh.

## Logical commits

- `1599ce66`: public-core guides (10 changed files, 21 reviewed paths).
- `2236023d`: platform/binding guides (31 changed files, 45 reviewed paths).
- `4311ea3c`: current status, historical operational pointers and initial
  whole-document coverage ledger (14 changed files).
- `22cc9d88`: parity/fixture recipes and remote-execution boundaries
  (20 changed files, 101 reviewed paths).
- `85d0f017`: six operational skill pairs (12 changed files).
- `da340976`: canonical ledgers, owner checklists, governance and legal review
  boundaries (17 changed files).
- This final audit-only commit reconciles all 315 original coverage rows and
  records the final checks; its hash is available in Git history rather than
  self-embedded in this file.

The six preceding commits passed the normal five-step pre-commit checks; the
final audit commit uses the same checks. They remain local; this refresh does
not push or mutate a PR. Documentation changes do not add a model acceptance,
legal sign-off, release promotion or provider-inventory guarantee.
