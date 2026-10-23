# Current project status

**English** | [日本語](current-status.ja.md)

Reviewed on **2026-10-23 UTC**, against accepted GitHub `main`
[`c1515500aabe7893e8176fd304c642e1b8ba2a03`](https://github.com/ayutaz/vokra/commit/c1515500aabe7893e8176fd304c642e1b8ba2a03).
This is the current routing page. Dated handoffs and benchmarks retain their
original commits, permissions, measurements and limitations.

## Releases and source

| Surface | Verified state |
|---|---|
| Workspace source version | `0.4.0-rc.1` in `Cargo.toml`; current main includes later maintenance changes |
| Latest stable GitHub release | [`v0.3.0`](https://github.com/ayutaz/vokra/releases/tag/v0.3.0), published 2026-09-20; 18 assets |
| Published GitHub preview | [`v0.4.0-rc.1`](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1), published 2026-10-17; 22 assets; prerelease, not a stable release |
| Exact preview source | `d84d311e98d0abb2987b5c4e96157ae8f2068e1a`, not current main |
| Preview scope | CPU-only desktop CLI/C ABI; hosted Linux Whisper/Silero functional smoke and native packaging checks, not all-model or Apple/Metal parity |
| External package registries | Separate gated channels; GitHub preview publication is not crates.io, PyPI, npm, OpenUPM or Godot AssetLib publication |

Use the [preview installation guide](release-0.4.0-rc1.md) for that exact
release. Python, Unity, Godot and web packages have separate source/asset
states; consult their binding READMEs rather than inferring distribution from
the workspace version. No API/ABI stability or full-catalog completion is
promised before 1.0.

Whisper's repaired provenance/MIT artifact is published at fixed revision
`bf440f6064ffc882d53d3694729dc4616dc2cfa7`; see the
[repair evidence](handoff/whisper-base-provenance-repair-2026-10-23.md).
The initial pairing attempts stopped before inference on the temporary audio
path and then a missing checksum sidecar; both failures remain in the
[execution record](handoff/cpu-preview-announcement-execution-2026-10-23.md).
The [reviewed-source preflight](handoff/evidence/published-rc1-cli-pairing-preflight-2026-10-23.json)
now passes all three explicit CPU commands: Whisper keyphrase, Silero speech
214/343 frames and silence 0/343. That run is bound to PR #295's source,
not accepted main. #295 acceptance, a final main run and the English/Japanese
guide acceptance remain required before announcing the repaired pairing.

## Main CI and open pull requests

Whisper integration [#293](https://github.com/ayutaz/vokra/pull/293) and the
reviewed Pillow metadata update [#287](https://github.com/ayutaz/vokra/pull/287)
and fixture-path correction [#294](https://github.com/ayutaz/vokra/pull/294)
are normally merged. Each passed all sixteen required context/application
pairs and its relevant audit checks at the exact reviewed source head.
#293's final source readback was 81 successes and five event-specific skips.
#287's optional iOS/musl jobs were still running at its merge readback; their
pending state is not a completed pass. Source PR acceptance is separate from
later main-push workflows and the pending manual functional run. These results
do not certify this document branch or the unrelated dirty guard.
Main retains sixteen strict required context/application pairs, administrator
enforcement, linear history and conversation resolution; no protection was changed.

Nine PRs remain open at this readback (six drafts):

| PR | Head prefix | Current responsibility |
|---|---|---|
| [#295](https://github.com/ayutaz/vokra/pull/295) | `e17f1446` | Include the committed audio checksum in sparse inputs; nine offline regressions and genuine preflight pass, fresh PR CI and normal acceptance pending |
| [#290](https://github.com/ayutaz/vokra/pull/290) | `6d1f02dd` | Torch/CUDA reference closure changes require fresh compatibility/native/license evidence and CPU-index responsibility reconciliation; changes requested |
| [#278](https://github.com/ayutaz/vokra/pull/278) | `11e33a10` | Draft: Candle same-run producer/receipt/consumer contract; preserve artifact size and authentication gates |
| [#271](https://github.com/ayutaz/vokra/pull/271) | `d777dd71` | Nine production-action updates; missing producer-artifact failure is distinct from #278's extracted-size failure |
| [#198](https://github.com/ayutaz/vokra/pull/198) | `5614a92c` | Draft: dated legal-source reconciliation; not a legal sign-off or merged policy |
| [#191](https://github.com/ayutaz/vokra/pull/191) | `63cadf3c` | Draft: Kyutai DSM PCM/reference reconciliation |
| [#182](https://github.com/ayutaz/vokra/pull/182) | `a3afc1ec` | Draft: distinct VibeVoice Realtime runtime/reference consolidation |
| [#169](https://github.com/ayutaz/vokra/pull/169) | `b3e9d5c1` | Draft: VibeVoice CPU-reference dependency closure |
| [#152](https://github.com/ayutaz/vokra/pull/152) | `a9f54e1a` | Draft: XCodec2 pre-import/dependency/evidence gates |

PRs #284, #286, #288, #293, #287 and #294 are merged in the accepted main.
#291/#292 were closed as incorporated through #293, not independently merged.
An open PR's green
ordinary CI does not clear its source, license, owner or real-weight gates.
The [announcement plan](handoff/cpu-preview-announcement-plan-2026-10-23.md)
separates its required path from the held model/CI/legal follow-ups.

## Model and Apple completion

A fresh public-API/model-card-only audit on 2026-10-23 reports **194 public
repositories, 193 GGUF-bearing repositories and 198 GGUF files**. The
[complete row snapshot](handoff/model-inventory-2026-10-23.tsv) has SHA-256
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`.
It records source/artifact reachability, not newly executed model parity:

| Classification | Rows |
|---|---:|
| CPU code/artifact full | 136 |
| CPU partial | 43 |
| No complete runtime binder | 14 |
| Non-artifact repository | 1 |
| Metal code/artifact full | 136 |
| Metal blocked by CPU | 57 |
| Metal non-artifact | 1 |

Thus **58 public rows remain non-full**; there are no CPU-full/Metal-unsupported
rows in this classification. The frozen **63-row owner-decision packet** is a
different historical, hash-bound scope and must not be rewritten to 58.
Historical named [Apple results](handoff/mac-cpu-metal-scaleway-results-2026-09-11.md)
and separately authorized artifact publications remain valid only for their
exact recorded scopes; the full catalog is **not finished**.

Remaining order: resolve PR/source/dependency/license/owner prerequisites →
authorized VAST real-weight conversion, pinned independent reference and CPU
parity → reviewed clean HEAD and fresh authenticated packets → **Scaleway
last**, for Apple CPU/reference, Metal/reference, Metal/CPU and explicit
no-fallback → result acceptance and cleanup → separately authorized publication.
Real BF16 checkpoint validation, HiFTNet and **all four BigVGAN variants**
remain explicit cross-cutting requirements. See the
[execution plan](handoff/mac-cpu-metal-execution-plan-2026-09-07.md),
[remaining-task ledger](handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md)
and [owner packet](handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md).

## Security, law and operational boundaries

The latest all-pages GitHub readback on 2026-10-23 has **638 open Dependabot alerts**:
94 critical, 259 high, 201 medium and 84 low; 614 name a patched version and
24 do not. All are pip alerts under `tools/parity/` or `integrations/`;
353 are critical/high. The [dated selected-metadata receipt](handoff/evidence/announcement-security-readback-2026-10-23.json)
records the counts and manifest distribution; the earlier 572-alert snapshot
remains historical. Eight Code Scanning alerts remain: four Scorecard findings and
CodeQL `rust/log-injection` #68–#71 in server test code. A test-file location
is not a false-positive disposition; no finding was dismissed. These
repository-wide/reference-environment alerts are not a new external dependency
in the first-party-only root runtime. They also prevent a blanket security-clean claim.

The [license audit](license-audit.md) preserves its dated findings and exact
sign-offs. The [legal-compliance guide](legal-compliance.md) separates source
retrieval from current-law confirmation: EUR-Lex bot verification and California
HTTP 403 prevent a blanket fresh-law attestation. Automatic synthesis-path
watermark integration remains `Deferred`; standalone AudioSeal does not
establish deployment compliance.

The maintainer Mac must not download or execute models or run workspace-wide /
`vokra-models` Cargo. Documentation checks are static/offline. The announcement
execution used a now-destroyed VAST worker and a genuine hosted Actions run;
the [execution record](handoff/cpu-preview-announcement-execution-2026-10-23.md)
preserves their distinct failures and cleanup evidence. No new Release,
registry, Hugging Face or X publication is part of steps 1–4. Old destroyed
IDs are not current workers or usable transfer sources.

The previous task's uncommitted local-model guard repair remains separate
from this documentation change and from accepted main. It has offline test
evidence, not a merged-code or model-completion claim.

## Documentation audit

The [2026-10-23 coverage and verification record](handoff/documentation-refresh-2026-10-23.md)
lists reviewed surfaces, source evidence, preserved history, static gate
results and explicit unverified runtime/device/legal axes. Use
[the documentation index](README.md) for guides; historical checkpoints are
not current execution instructions.
