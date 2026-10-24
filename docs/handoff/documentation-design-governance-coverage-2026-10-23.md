# Design, benchmark, and governance documentation coverage — 2026-10-23

This is a coverage record for the delegated documentation scope. It does not
replace dated measurements, owner checklists, or governance decisions.

## Review method and limits

- Inventory: the union of `rg --files docs/design docs/bench-baselines
  docs/benchmarks docs/governance -g '*.md'` and the delegated `docs/*`
  handover/checklist/scope/variance globs, de-duplicated and sorted: **59
  paths**.
- Every path below was inspected at its opening section to classify it as a
  current index/protocol/template, a dated measurement or handover, a design
  record, a fixture/provenance record, or an owner-controlled checklist.
- The opening sections and relevant current-routing sections were read for
  the ten active files edited in this refresh, together with their linked
  source pointers where needed to validate the routing. Other records retain
  their explicit dates, hashes, measured values, thresholds, verdicts, and
  source pointers; they were not rerun or numerically revalidated.
- Static coverage used the repository documentation gates listed below. No
  model was downloaded or executed, no device or cloud run was performed, and
  no Cargo command was run.
- The `docs/current-status.md` and `.ja.md` targets are maintained by the
  root agent; the new top notes below point to those canonical current-status
  pages without copying their live ledger.

## Per-file coverage

| Path | Classification | Evidence reviewed | Disposition / limits |
|---|---|---|---|
| `docs/bench-baselines/README.md` | current index/provenance | index rules, CI/rig distinction, linked baseline paths | **Edited**: added current-status pointer and release facts; no gate value or baseline rerun |
| `docs/bench-baselines/awaiting-real-weight-2026-07-22/report.md` | dated measurement | dated report header and measurement provenance | Unchanged; measured values and “awaiting real weight” boundary preserved |
| `docs/bench-baselines/eval-cache-artifacts-2026-07-19/README.md` | dated fixture/provenance | cache-staleness and artifact provenance header | Unchanged; no local cache inspection or model access |
| `docs/bench-baselines/m1-real-weight-eval-2026-07-16/report-campaign2.md` | dated measurement | campaign date, HEAD, model/eval scope in header | Unchanged; no real-weight rerun |
| `docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md` | dated measurement | campaign date, HEAD, rig and method in header | Unchanged; measured numbers and provenance preserved |
| `docs/bench-baselines/m4-05-csm-fixture-reference.md` | fixed fixture/reference measurement | fixture-floor scope and non-gate boundary | Unchanged; fixture results not remeasured |
| `docs/bench-baselines/m4-19-wyoming-real-gguf-2026-07-19/report.md` | dated measurement | dated real-GGUF report and gated-leg provenance | Unchanged; no model execution |
| `docs/bench-baselines/m5-01-coreml-bakeoff-2026-08-24/README.md` | dated measurement | dated Apple/CoreML evidence header | Unchanged; device result and hashes preserved |
| `docs/bench-baselines/m5-02-qnn-bakeoff-2026-08-24/README.md` | dated prerequisite record | dated no-SDK/no-target boundary | Unchanged; no QNN claim added |
| `docs/bench-baselines/m5-14-final-2026-07-18/report.md` | dated measurement | rig, tree, profile, and protocol in header | Unchanged; optimization numbers preserved |
| `docs/bench-baselines/m5-14-wave0-2026-07-18/hotspot-tables.md` | dated measurement | rig, detached worktree, and profiling protocol | Unchanged; hotspot values preserved |
| `docs/bench-baselines/m5-14-wave0-2026-07-18/target-table.md` | dated measurement/reference | rig, ORT comparison, and target-table header | Unchanged; no new performance claim |
| `docs/bench-baselines/metal-transcript-parity-2026-07-19/report.md` | dated measurement | complete Metal/CPU parity report header | Unchanged; parity values and run date preserved |
| `docs/bench-baselines/rtf-decomposed-2026-07-08.report.md` | dated measurement | analyzer-generated reference report header | Unchanged; formal-gate boundary preserved |
| `docs/bench-baselines/rtf-fa-v2-2026-07-08.report.md` | dated measurement | analyzer-generated reference report header | Unchanged; formal-gate boundary preserved |
| `docs/bench-baselines/sbom-reproducibility-2026-07-19/report.md` | dated reproducibility record | SBOM evidence/non-ratification boundary | Unchanged; no reproducibility rerun |
| `docs/bench-baselines/server-real-gguf-slots-2026-07-21/report.md` | dated measurement | real-GGUF slot report and env-gated boundary | Unchanged; no weight access |
| `docs/bench-baselines/silero-8k-ctx288-2026-07-19/report.md` | dated measurement | 8 kHz real-speech report header | Unchanged; measured values preserved |
| `docs/bench-baselines/vast-2026-07-10/rtf-decomposed.report.md` | dated VAST measurement | analyzer report and reference-gate boundary | Unchanged; no VAST rerun |
| `docs/bench-baselines/vast-2026-07-10/rtf-fa-v2.report.md` | dated VAST measurement | analyzer report and reference-gate boundary | Unchanged; no VAST rerun |
| `docs/bench-baselines/vast-2026-08-10-h100/README.md` | dated VAST measurement | H100 hardware/date and handover pointer | Unchanged; no GPU access |
| `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-decomposed.report.md` | dated VAST measurement | analyzer report header | Unchanged; values preserved |
| `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-fa-v2.report.md` | dated VAST measurement | analyzer report header | Unchanged; values preserved |
| `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-fa-v3.report.md` | dated VAST measurement | analyzer report header | Unchanged; FA-v3 evidence preserved |
| `docs/bench-baselines/web-2026-07-15/README.md` | dated measurement/protocol | WASM/WebGPU timing and no-threshold boundary | Unchanged; no browser rerun |
| `docs/bench-baselines/x-06-preverify-2026-07-20/README.md` | dated CI preverify record | explicit preverify/non-gate boundary | Unchanged; no workflow execution |
| `docs/bench-baselines/x-06-preverify-2026-07-20/asr-wer-summary.m1.md` | fixed measurement | dated WER summary and model/utterance facts | Unchanged; WER value preserved |
| `docs/benchmarks/v0.5-device-benchmarks.md` | current owner template | scaffold, blank result cells, NFR threshold references | **Edited**: added current-status pointer; no device result or verdict |
| `docs/benchmarks/v0.5-device-runbook.md` | current owner runbook | red-lines, source harness links, owner execution boundary | **Edited**: added current-status pointer; no command or threshold change |
| `docs/benchmarks/v0.9-device-benchmarks.md` | current owner template | blank Android/Godot/server result scaffold | **Edited**: added current-status pointer; no device result or verdict |
| `docs/design/firered-pcm-beam-composite.md` | current design, parity pending | status and VAST-only boundary in header | Unchanged; no parity or redistribution claim added |
| `docs/design/kyutai-stt-pcm-composite.md` | dated design/source reconciliation | 2026-10-14 reconciliation note and contract boundary | Unchanged; no source/model execution |
| `docs/design/kyutai-stt-streaming-lm.md` | design/spec, parity pending | streaming contract and synthesized-test boundary | Unchanged; no real-weight claim added |
| `docs/design/m0-03-gguf-loader.md` | dated design with current-source pointer | 2026-08-30 boundary and Rust authority references | Unchanged; source implementation remains authoritative |
| `docs/design/mimi-checkpoint-provenance.md` | provenance design contract | input-byte identity contract and converter pointer | Unchanged; no checkpoint inspection |
| `docs/design/mimi-rust-core-metadata-binding.md` | source-only design/audit protocol | OPEN/SOURCE-ONLY/VAST-ONLY status and no-build boundary | Unchanged; no cargo metadata or package fetch |
| `docs/design/quantization-policy.md` | dated design with supersession note | 2026-08-30 implementation/supersession note | Unchanged; policy and measured history preserved |
| `docs/design/size-budget.md` | dated design with current-source pointer | 2026-08-30 boundary and size-gate source reference | Unchanged; no size build/check |
| `docs/design/vokra-gguf-chunks.md` | dated design/spec | 2026-08-30 boundary and converter/core authority | Unchanged; no schema or source edit |
| `docs/governance/dod-judgment-template.md` | current owner template | blank verdicts, dated routing, no-owner-approval boundary | **Edited**: supersession/current-status pointer; no judgment or signature |
| `docs/governance/exit-path-playbook.md` | current governance protocol | exit-path principles and maintainer-decision boundary | Unchanged; no exit decision or status claim |
| `docs/governance/kill-switch-metrics-runbook.md` | current owner runbook | manual collection commands, thresholds, dated observations | **Edited**: supersession/current-status pointer; no metrics run or threshold change |
| `docs/governance/quarterly-review-runbook.md` | current owner runbook | cadence, kill-switch procedure, dated release observations | **Edited**: supersession/current-status pointer; no review or date assignment |
| `docs/governance/quarterly-reviews/README.md` | current governance index/template guide | naming rules, append-only snapshot policy, dated state section | **Edited**: supersession/current-status pointer; no record or snapshot created |
| `docs/governance/vokra-go-nogo-v0.5.md` | current owner-review template | blank A–L table, normative thresholds, dated state | **Edited**: supersession/current-status pointer; no verdict, metric, or signature |
| `docs/handoff/dia-dependency-scope-reconciliation-2026-10-14.md` | dated source reconciliation | 2026-10-14/15 source and refusal-boundary header | Unchanged; no dependency or source edit |
| `docs/handoff/parity-urllib3-three-scope-reconciliation-2026-10-15.md` | dated dependency reconciliation | isolated-scope and provenance boundary in header | Unchanged; no dependency or parity rerun |
| `docs/m2-14-ios-rtf-handover.md` | dated owner handover | 2026-09-09 audit-start snapshot and device/cloud boundary | Unchanged; no Apple-device run |
| `docs/m2-cuda-rtf-variance-2026-07-08.md` | fixed dated measurement | filled variance report and reference-gate position | Unchanged; measured numbers and owner judgment preserved |
| `docs/m2-cuda-rtf-variance-template.md` | current owner measurement template | blank fields, VAST command interface, formal-gate boundary | **Edited**: added current-status pointer; no VAST run or threshold change |
| `docs/m2-owner-verification-checklist.md` | dated owner checklist | 2026-10-04 reading boundary and historical counts | Unchanged; historical checklist is not current status |
| `docs/m3-11-godot-demo-handover.md` | dated owner handover | 2026-09-09 audit-start and manual-device boundary | Unchanged; no Godot/device execution |
| `docs/m3-15-server-latency-handover.md` | dated owner handover | 2026-09-09 benchmark and owner-procedure boundary | Unchanged; no server benchmark run |
| `docs/m3-18-android-rtf-handover.md` | dated owner handover | 2026-09-09 Android handover boundary | Unchanged; no Android-device run |
| `docs/m3-owner-verification-checklist.md` | dated owner checklist | 2026-10-04 reading boundary and historical counts | Unchanged; historical checklist is not current status |
| `docs/m4-07-hopper-bench-handover.md` | dated VAST owner handover | 2026-09-09 boundary and H100 lifecycle evidence | Unchanged; no GPU/VAST run |
| `docs/m4-owner-verification-checklist.md` | dated owner checklist | 2026-10-04 reading boundary and historical counts | Unchanged; historical checklist is not current status |
| `docs/m4-scope-expansion-2026-07-13.md` | fixed dated scope decision | 2026-07-14 decision and 2026-09-09 audit boundary | Unchanged; candidate history and verdict preserved |
| `docs/m5-owner-verification-checklist.md` | root-owned current checklist | current routing and owner-controlled acceptance fields | Not edited by this delegated scope; root added a current pointer and verified unchanged 53 checked / 29 unchecked states in the integrated audit |

## Static checks

Passed after the nine edits:

- `bash scripts/check-doc-references.sh`
- `bash scripts/check-runbook-path-citations.sh`
- `git diff --check` for the edited files and this coverage record

The broader documentation gates remain the root agent's integrated review
responsibility because other delegated documentation scopes are dirty in the
same worktree. No commit, push, publication, model operation, Cargo build, or
cloud/device verification was performed.
