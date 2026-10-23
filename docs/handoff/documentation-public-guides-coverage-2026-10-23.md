# Public documentation coverage audit — 2026-10-23

This is a dated coverage record for the 2026-10-23 public-documentation
refresh. It records which files were reviewed, what evidence was consulted, and
where the refresh deliberately stopped. It is not a model, license, release,
Apple-hardware, or publication approval record.

## Authoritative evidence used

- Public `main`: `151d94a8613d5bd4acb1671a461f59742c37b8fc` (2026-10-22 UTC).
- Workspace version and effective toolchain facts: `Cargo.toml`,
  `crates/vokra-backend-cpu/Cargo.toml`, and `Package.swift`.
- Stable/release boundary: GitHub `v0.3.0` remains the latest stable release;
  workspace `0.4.0-rc.1` is a prerelease CPU-only preview. The RC facts and
  exact source/tag are recorded in `README.md` and the public release record.
- Current model metadata snapshot: `docs/handoff/model-inventory-2026-10-23.tsv`
  (194 repositories / 193 GGUF-bearing repositories / 198 GGUF files;
  CPU 136 full, 43 partial, 14 without a runtime binder, 1 non-artifact;
  Metal 136 full, 57 blocked by CPU, 1 non-artifact). These are metadata
  classifications, not real-weight or Apple-device results.
- API/backend/source pointers: `include/vokra.h`, `crates/vokra-cli/src/`,
  `crates/vokra-backend-{cpu,metal,cuda,vulkan,webgpu,coreml,qnn}/`, the
  referenced integration sources, and the scripts named by each guide.
- Static gates: doc examples (144 blocks / 26 docs, with 30 Tier-C blocks
  deferred), document references (110 IDs / 33 architecture anchors), runbook
  path citations, community docs, parity sidecars, platform-support anchors,
  ABI changelog, and `git diff --check` all passed. No model, Cargo-heavy,
  package-network, cloud, or device execution was performed.

## Root policy and entry documents

| Path | Review/result | Limits or unchanged rationale |
|---|---|---|
| `README.md` | **Changed** — current checkpoint and stable/prerelease boundary updated; historical snapshots retained. | No catalog completion or hardware result inferred. |
| `README.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `README-swift-package.md` | **Changed** — workspace line, stable consumer pin, current `main`, and `Package.swift` relationship updated. | Swift compiler/device validation not run. |
| `CONTRIBUTING.md` | **Unchanged** — branch-protection, CI, VAST, license, and hook policy match `AGENTS.md` and workflow index. | The delegated guide review did not fetch protection; the root's integrated audit separately read the current API. |
| `SECURITY.md` | **Changed** — stable `v0.3.0` support and `0.4.0-rc.1` development line separated. | No security triage or advisory action. |
| `SECURITY.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `CODE_OF_CONDUCT.md` | **Unchanged** — reporting and enforcement text has no version/current-state claim. | No community-platform mutation. |
| `CODE_OF_CONDUCT.ja.md` | **Unchanged** — Japanese twin has no stale release or implementation claim. | Same limits as English page. |

## Current product documents directly under `docs/`

| Path | Review/result | Limits or unchanged rationale |
|---|---|---|
| `docs/api-reference.md` | **Changed** — current release/main and Apple-verification boundary updated; dated snapshots retained. | Header regeneration and API runtime checks were not run. |
| `docs/api-reference.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/architecture.md` | **Unchanged** — crate/backend topology and anchors resolve; no stale current release claim found. | No workspace Cargo build or runtime execution. |
| `docs/architecture.ja.md` | **Unchanged** — twin topology and anchors resolve. | Same limits as English page. |
| `docs/backend-guide.md` | **Changed** — current routing/release boundary retained; the 2026-09-11 Apple batch is labeled historical, and CPU comparison now requires prior independent-reference validation. | Physical GPU/NPU validation intentionally deferred; CPU/GPU agreement alone is not completion. |
| `docs/backend-guide.ja.md` | **Changed** — English twin aligned with the dated batch and independent-reference boundary. | Same limits as English page. |
| `docs/c-api-streaming-codec.md` | **Unchanged** — current codec contract matches the header/source pointers; no stale release assertion. | C ABI execution was not run. |
| `docs/deliverables.md` | **Changed** — old 131/63 current-state claim replaced with the dated 136/58 metadata snapshot and explicit non-completion boundary. | Future v1.0 deliverables remain normative goals. |
| `docs/getting-started.md` | **Unchanged** — Rust 1.89 effective MSRV, uv-only Python, GGUF-only runtime, and VAST safety guidance match source/policy. | Sample model conversion was not run. |
| `docs/getting-started.ja.md` | **Unchanged** — Japanese twin matches the same current contract. | Same limits as English page. |
| `docs/good-first-tasks.md` | **Changed** — review date and current count of 23 scripts without `--help` updated; six task entries retained. | Task completion was not implemented or claimed. |
| `docs/good-first-tasks.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/migration-guide.md` | **Changed** — Python distribution boundary now distinguishes stable GitHub wheel assets from unpublished PyPI/TestPyPI and wheel-less `0.4.0-rc.1`; RTX 4090 figures are labeled as dated exact-setup references. | No migration runtime test or benchmark rerun. |
| `docs/migration-guide.ja.md` | **Changed** — English twin aligned on Python distribution and dated benchmark limits. | Same limits as English page. |
| `docs/nanocodec-conversion.md` | **Unchanged** — explicitly dated conversion/audit boundary correctly defers to current source and license records. | No checkpoint acquisition or conversion. |
| `docs/onnx-alternative-research.md` | **Unchanged** — explicitly a 2026-07 research snapshot with supersession/legal boundary notes. | Historical research was not rewritten. |
| `docs/piper-plus-integration.md` | **Unchanged** — explicitly dated M0-07 analysis; historical upstream facts and unfilled owner cells are preserved. | No upstream/package investigation or owner decision. |
| `docs/requirement-ids.md` | **Changed** — 110 IDs / 33 anchors and verification date refreshed. | The checker, not manual counting, is authoritative. |
| `docs/requirement-ids.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/requirements.md` | **Changed** — current-state note now points to public main and current ledger instead of PR #79/Scaleway-start claims. | Requirements remain goals, not implementation proof. |
| `docs/system-requirements.md` | **Changed** — observed baseline and release boundary updated; normative requirements preserved. | No requirement was marked satisfied by this refresh. |

The excluded direct-doc files (`docs/README.md`, `docs/license-audit.md`,
`docs/legal-compliance.md`, `docs/abi-changelog.md`, RC installation guides,
`docs/milestones.md`, and dated `m*` / handover / scope records) remain under
the root manager's ownership or historical-record rules and were not edited by
this subtask.

## Tutorials

| Path | Review/result | Limits or unchanged rationale |
|---|---|---|
| `docs/tutorials/cli.md` | **Changed** — parser verification date refreshed; examples pass the doc-example gate. | No CLI build or model run. |
| `docs/tutorials/cli.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/tutorials/android.md` | **Changed** — build-script/NDK verification date refreshed. | No Android NDK or device run. |
| `docs/tutorials/android.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/tutorials/godot.md` | **Changed** — documentation review date refreshed; recorded runtime evidence remains dated and explicitly not rerun. | No Godot runtime or model execution. |
| `docs/tutorials/godot.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/tutorials/ios.md` | **Unchanged** — stable `v0.3.0` URL/checksum and local-tag distinction match `Package.swift`. | Swift/Xcode/device validation not run. |
| `docs/tutorials/ios.ja.md` | **Unchanged** — Japanese twin matches the stable package contract. | Same limits as English page. |
| `docs/tutorials/python.md` | **Changed** — current workspace line and stable-release/PyPI boundary refreshed. | No native wheel load or package publication. |
| `docs/tutorials/python.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/tutorials/server.md` | **Changed** — config flag-surface verification date refreshed. | No server process or endpoint test. |
| `docs/tutorials/server.ja.md` | **Changed** — English twin aligned. | Same limits as English page. |
| `docs/tutorials/unity.md` | **Unchanged** — stable `v0.3.0` tarball/source-only distinction and platform caveats match release policy. | No Unity/IL2CPP/toolchain run. |
| `docs/tutorials/unity.ja.md` | **Unchanged** — Japanese twin matches the same release contract. | Same limits as English page. |
| `docs/tutorials/web.md` | **Unchanged** — local `0.0.0-dev` package, explicit backend selection, and COOP/COEP guidance match source. | No npm publication or browser/WebGPU run. |
| `docs/tutorials/web.ja.md` | **Unchanged** — Japanese twin matches the same web contract. | Same limits as English page. |

## Platform-support documents

| Path | Review/result | Limits or unchanged rationale |
|---|---|---|
| `docs/platform-support/cdn-selection-material.md` | **Unchanged** — owner-entry scaffold correctly contains no invented provider, price, or contract decision. | CDN selection remains owner work. |
| `docs/platform-support/v1.0-rc-support-matrix.md` | **Changed** — current routing header updated to main `151d94a8`, eight open PRs, stable/RC boundary, and retained 136/58 metadata snapshot. | Platform cells, hardware results, and owner verdicts were not promoted. |

## Remaining gaps

The doc-example checker reports 30 Tier-C snippets requiring owner toolchains
(Swift, Unity, Godot, or prose-only blocks); they are deferred rather than
fabricated as passes. The refresh did not run model weights, package installs,
Cargo workspace builds/tests, VAST/Scaleway jobs, Apple hardware, or publication
actions. The manager must review the combined worktree diff and decide commit/PR
boundaries.
