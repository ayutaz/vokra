# Documentation integration coverage — 2026-10-23

This record covers every Markdown file under the bounded integration scope
(`bindings/`, `integrations/`, `examples/`, `crates/`, `tests/`, `web/`,
`assets/`, and `tools/bench/`). It is a coverage record for the documentation
refresh; it is not a model, device, release, or security-test result.

## Boundary and evidence

- Inventory: 69 Markdown files were enumerated with `rg --files` and each file's
  opening section was inspected on 2026-10-23.
- Detailed body/source review was applied to the release/package docs changed
  for factual drift. For status-date-only edits, the status block, relevant
  source-history delta, and static checker coverage were reviewed; this is not
  a runtime audit. The remaining records below are deliberately preserved as
  dated fixture, provenance, license, design, or owner-gated evidence; their historical
  numbers, hashes, grants, and verdicts were not reinterpreted.
- Source facts read for the release/package docs: `Cargo.toml` (workspace
  `0.4.0-rc.1`), `bindings/unity/com.vokra.unity/package.json`,
  `bindings/python/pyproject.toml`, `bindings/python/hatch_build.py`,
  `.github/workflows/python-wheels.yml`,
  `.github/workflows/release-rc1-gh-only.yml`, `include/vokra.h`,
  `bindings/python/scripts/gen-py-bindings.py`, and the current generated
  Python bindings. The Unity sample README was checked against its
  `PipelineRunner.cs`, `DemoUi.cs`, and `Scripts/fetch-demo-models.sh` without
  executing Unity or downloading models. The inspected integration history was checked with
  `git log`; the post-review changes visible there touched dependency/security
  metadata, not the documented API surfaces.
- Static gates run after the edits: `check-doc-references.sh` and its
  self-test; `check_doc_examples.py` and its self-test;
  `check-runbook-path-citations.sh`; `check-community-docs.sh`;
  `check-parity-sidecar-citations.sh`; `check-owner-checklist-drift.sh`;
  `check-platform-support.sh`; `check-abi-changelog.sh`;
  `check-workflow-hygiene.sh`; and `git diff --check`.
- The normal example checker covered 144 blocks in 26 docs and passed. Its 30
  Tier-C entries remain explicitly deferred because they need Swift, Unity,
  or Godot tooling. The self-test's intentional negative diagnostics are
  expected; the self-test exited successfully.
- No model was downloaded, loaded, converted, or executed. No device, Unity
  Editor, Godot Editor, WebGPU browser, VAST, or Scaleway run was performed.
  Therefore review dates below mean **documentation/source review only**, not
  fresh runtime, device, or security evidence.

## Per-file coverage

Legend: `current` means a current/public or current technical guide;
`history` means a dated design, test, or status record; `fixture` means a
fixture/provenance contract; `license` means license/attribution text.
`edited` means this refresh changed the file. `unchanged` means it was
reviewed and intentionally left as-is.

| File | Class | Result and evidence boundary |
| --- | --- | --- |
| `assets/branding/README.md` | current | unchanged; inventory/opening-section review; branding data has no version/API claim; no asset rendering run |
| `bindings/python/README.md` | current | **edited**; `Cargo.toml`, Python build metadata, wheel workflow, C header/generator checked; distinguishes workspace `0.4.0-rc.1` from latest published Python wheels `v0.3.0`; no upload claim |
| `bindings/unity/com.vokra.unity/Samples~/VadAsrTts/README.md` | current/sample | **edited**; checked against `PipelineRunner.cs`, `DemoUi.cs`, and `Scripts/fetch-demo-models.sh`; documents current-status routing, unavailable GGUF release assets, URL overrides, and WebGL async boundary; no Unity/model execution |
| `bindings/unity/com.vokra.unity/CHANGELOG.md` | history | unchanged; release history intentionally preserved; release-note ownership remains outside this integration record |
| `bindings/unity/com.vokra.unity/LICENSE.md` | license | unchanged; Apache-2.0 text preserved verbatim; no license audit or legal reclassification |
| `bindings/unity/com.vokra.unity/README.md` | current | **edited**; `package.json`, workspace version, RC1 workflow, and release guide checked; states package metadata `0.4.0-rc.1` while the GH-only desktop preview has no Unity tarball |
| `bindings/unity/com.vokra.unity/TESTING.md` | current/history | **edited** review date only; CI/nightly/device procedure and no-device-evidence boundary preserved; no Unity, IL2CPP, iOS, or Android test rerun |
| `crates/vokra-backend-vulkan/kernels/README.md` | current technical | unchanged; inventory/opening-section and runbook/path gates; shader source/blob claims not regenerated |
| `crates/vokra-backend-vulkan/kernels/precompiled/README.md` | fixture/provenance | unchanged; committed SPIR-V/hash procedure preserved; no shader compilation or hash refresh |
| `crates/vokra-backend-webgpu/kernels/README.md` | current technical | unchanged; inventory/opening-section and path gates; no WebGPU execution |
| `crates/vokra-core/tests/fixtures/rng_torch/README.md` | fixture/history | unchanged; dated Torch RNG evidence preserved; no Python/Torch or fixture regeneration |
| `crates/vokra-core/tests/parity/fixtures/m2-08/README.md` | fixture/history | unchanged; analytic fixture contract preserved; no local test or GGUF materialization |
| `crates/vokra-core/tests/parity/fixtures/m5-06/README.md` | fixture/history | unchanged; independent OpenFST provenance preserved; no OpenFST or parity execution |
| `crates/vokra-kws-micro/README.md` | current/status boundary | unchanged; explicit VAST-gated and non-green real-weight boundary preserved; no model work |
| `crates/vokra-models/src/fsmn_vad/SPEC.md` | history/spec | unchanged; dated identity/geometry contract preserved; no acquisition/conversion/parity claim added |
| `crates/vokra-models/src/kokoro/data/README.md` | fixture/provenance | unchanged; dated upstream tensor manifest boundary preserved; no Kokoro conversion or parity run |
| `crates/vokra-models/src/silero_vad/SPEC.md` | history/spec | unchanged; dated upstream/real-weight evidence preserved; no new device/model result |
| `crates/vokra-models/tests/fixtures/focalcodec/README.md` | fixture/provenance | unchanged; official upstream and VAST evidence preserved; no weight load or parity rerun |
| `crates/vokra-models/tests/fixtures/lang_id/README.md` | fixture/license | unchanged; official SpeechBrain revisions and Apple/VAST limitation preserved; no source or model execution |
| `crates/vokra-models/tests/fixtures/melotts_chinese/README.md` | fixture/provenance | unchanged; pinned upstream revision and shared oracle contract preserved; no conversion |
| `crates/vokra-models/tests/fixtures/melotts_english/README.md` | fixture/provenance | unchanged; pinned official source/checkpoint contract preserved; no conversion |
| `crates/vokra-models/tests/fixtures/melotts_japanese/README.md` | fixture/provenance | unchanged; pinned upstream revision and shared oracle contract preserved; no conversion |
| `crates/vokra-models/tests/fixtures/melotts_korean/README.md` | fixture/provenance | unchanged; pinned upstream revision and shared oracle contract preserved; no conversion |
| `crates/vokra-models/tests/fixtures/melotts_spanish/README.md` | fixture/provenance | unchanged; pinned upstream revision and shared oracle contract preserved; no conversion |
| `crates/vokra-models/tests/fixtures/metricgan_plus/README.md` | fixture/provenance | unchanged; official SpeechBrain package/checkpoint contract preserved; no reference execution |
| `crates/vokra-models/tests/fixtures/neucodec/README.md` | fixture/license | unchanged; pinned upstream, dependency, and GGUF evidence preserved; no weight load |
| `crates/vokra-models/tests/fixtures/parakeet_ctc/README.md` | fixture/provenance | unchanged; dated official Transformers reference and VAST evidence preserved; no model run |
| `crates/vokra-models/tests/fixtures/sepformer/README.md` | fixture/provenance | unchanged; official SpeechBrain oracle and measured historical bounds preserved; no parity rerun |
| `crates/vokra-models/tests/fixtures/wavtokenizer/README.md` | fixture/provenance | unchanged; official source and GGUF hash contract preserved; no weight load |
| `crates/vokra-models/tests/fixtures/whisper_medusa/README.md` | fixture/provenance | unchanged; dated VAST/Apple evidence preserved; no model run or claim promotion |
| `crates/vokra-models/tests/fixtures/xcodec2/README.md` | fixture/license/status | unchanged; `NO_UPLOAD` and pending primary-source gate preserved; no publication decision |
| `crates/vokra-ops/tests/fixtures/flow_sampler/README.md` | fixture/history | unchanged; offline reference contract preserved; no reference generation |
| `examples/unity-demo/Assets/Plugins/README.md` | current | **edited**; package metadata/source boundary checked; distinguishes current package metadata from the absent RC1 Unity tarball; no plugin build |
| `examples/unity-demo/Assets/StreamingAssets/models/README.md` | fixture/license | unchanged; model source/license and non-committed artifact boundary preserved; no model fetch |
| `examples/unity-demo/ProjectSettings/README.md` | current/owner | unchanged; Unity version remains intentionally environment-generated; no Unity Editor run |
| `examples/unity-demo/README.md` | current | **edited**; package metadata and RC1 asset boundary checked; demo remains a v0.1 spike; no runtime claim added |
| `examples/unity-demo/TESTING.md` | history/owner | unchanged; pending per-OS test template preserved; no macOS/Linux/Windows Unity run |
| `integrations/vokra-android/README.md` | current/proposed | unchanged; proposed ADR and owner-gated boundary preserved; no Android build/device test |
| `integrations/vokra-cli-bench-server/README.md` | history/current tool | unchanged; dated Wave 14 status and zero-dependency claim preserved; no benchmark run |
| `integrations/vokra-godot/README.md` | current/status | **edited** review date only; source-history and existing status text retained; no Godot headless/editor run, no new GGUF result |
| `integrations/vokra-godot/demos/README.md` | current/owner | **edited** review date only; manual Editor boundary retained; no Editor run |
| `integrations/vokra-misaki-g2p/README.md` | current integration | unchanged; opt-in bridge and dependency boundary preserved; no subprocess/provider run |
| `integrations/vokra-piper-g2p/README.md` | current integration | unchanged; opt-in bridge and license boundary preserved; no synthesis run |
| `integrations/vokra-server-bench/README.md` | history/current tool | unchanged; dated bench status and isolated dependency boundary preserved; no benchmark run |
| `integrations/vokra-server/README.md` | current | **edited** review date only; CLI/config/security claims were checked against current documentation and integration history; no server build or runtime/security test |
| `integrations/vokra-server/docs/adr-http-stack.md` | history/design | unchanged; accepted ADR and dependency rationale preserved; no dependency audit rerun |
| `integrations/vokra-server/docs/scope.md` | history/current pointer | **edited** current-status date only; kickoff language remains historical and points to the README/source; no server execution |
| `integrations/vokra-server/docs/security-ops.md` | history/current pointer | **edited** current-status date only; reverse-proxy/TLS/auth boundary preserved; no security test, deployment, TLS, or authentication run |
| `integrations/vokra-server/docs/wyoming-design.md` | history/design | unchanged; historical protocol decisions and implementation limits preserved; no Wyoming run |
| `integrations/vokra-server/tests/wyoming-ha-smoke.md` | history/test evidence | unchanged; dated Docker smoke capture preserved; no Docker/server/Home Assistant test |
| `tests/capi/README.md` | current API | unchanged; C ABI behavior/opening contract preserved; no native build or C smoke test |
| `tests/fixtures/audio/README.md` | fixture/history | unchanged; dated audio provenance boundary preserved; no audio/model parity run |
| `tests/fixtures/audiobox_aesthetics/README.md` | fixture/provenance | unchanged; immutable upstream source/snapshot contract preserved; no reference generation |
| `tests/fixtures/audioseal/README.md` | fixture/license | unchanged; VAST-only official source/checkpoint boundary preserved; no download or execution |
| `tests/fixtures/sbv2/README.md` | fixture/provenance/history | unchanged; sidecar hashes and dated VAST evidence preserved; no GGUF regeneration or parity run |
| `tests/parity/README.md` | current pointer/history | **edited** current-pointer date/link only; audit-start values and no-new-parity boundary preserved; no numeric parity run |
| `tests/parity/bf16_gemm/README.md` | fixture/history | unchanged; VAST packet provenance preserved; no reference or Cargo run |
| `tests/parity/canary_1b_flash/README.md` | fixture/provenance | unchanged; structural-reference/non-numeric boundary preserved; no checkpoint load |
| `tests/parity/canary_1b_v2/README.md` | fixture/provenance | unchanged; structural-reference/non-numeric boundary preserved; no checkpoint load |
| `tests/parity/conv2d/README.md` | fixture/history | unchanged; reserved/not-generated boundary preserved; no reference generation or model run |
| `tests/parity/fcpe/README.md` | fixture/provenance | unchanged; official wheel/checkpoint boundary preserved; no reference execution |
| `tests/parity/fixtures/m1-02/kquant/README.md` | fixture/history | unchanged; analytic quant fixture contract preserved; no test or quantizer run |
| `tests/parity/fsq/nanocodec/README.md` | fixture/provenance/license | unchanged; NVIDIA/NeMo checkpoint provenance preserved; no checkpoint load |
| `tests/parity/piper_plus/README.md` | fixture/license | unchanged; ONNX/reference boundary and runtime exclusion preserved; no ONNX or model run |
| `tests/parity/silero_vad/README.md` | fixture/provenance | unchanged; official ORT oracle contract preserved; no ORT/model execution |
| `tests/parity/utmos/README.md` | history/status | unchanged; unsafe-pickle refusal and historical fixture boundary preserved; no UTMOS run |
| `tests/parity/vocoder_conv/README.md` | fixture/provenance | unchanged; VAST reference command/hash evidence preserved; no reference generation |
| `tools/bench/ios-device/README.md` | current/owner handoff | unchanged; physical-device benchmark boundary preserved; no Xcode/device run |
| `web/pkg/README.md` | current | **edited** review date only; `0.0.0-dev`/unpublished status and WebGPU/WASM loader docs checked against package metadata/source; no browser/WebGPU run |

## Result and remaining limits

The 13 edited files are now aligned with the checked local version and
publication boundaries. The 56 unchanged files retain their fixed evidence,
license, owner, and historical semantics. This record does not close any
owner-gated device/model/security item, and it does not replace the root
current-status ledger or the later VAST/Scaleway verification stages.
