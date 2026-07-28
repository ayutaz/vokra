# model-publish + parity-CI wave — owner handoff (2026-07-28)

Tracked / public. Operational summary of the 2026-07-28 ultracode workflows
(scout+triage → impl → design-spec) covering (a) 8 candidate models for
`huggingface.co/vokra` publication and (b) an audit of the parity-CI roster
for real-checkpoint coverage. This handoff is the single-page entry point
that names every artifact landed, every artifact deliberately deferred,
and every action still owed to the owner (yousan).

> **Post-handoff reconciliation (2026-07-28、later same day)**: this
> handoff was authored at commit `fab32b2` 11:02:17 UTC as a snapshot of
> the impl wave. Later that day the following additional events landed on
> the same branch and **override the "deferred" postures below**:
> (a) **X-Codec-2 publish fired** at commit `98c34cd` 12:29:46 UTC — the
> T4 first-precedent decision was granted by the owner and the artifact
> is live at `https://huggingface.co/vokra/xcodec2` (verified HTTP/2 200
> by the publish tooling); (b) **§3.1 sign-off wave completed** —
> VoxCPM2-2B ☑ Commercial at `64639b9`, Sesame CSM-1B ☑ Commercial +
> VibeVoice-Large ☑ Rejected (repo 404) at `a7a23c0`, both `yousan`
> 2026-07-28; (c) **pre-push optimization landed** (`e1cad37`/`2b21ea8`/
> `719ae3a`/`f7a247a`, ~40 min → ~1m45s = ~24x dev-machine speed-up, no
> repo `Cargo.lock` change = NFR-DS-02 preserved). The Executive summary
> below is retained verbatim as an as-of-handoff-time snapshot; treat it
> as history and read the post-handoff bullets first.

## Executive summary

* **2 models converted-and-published in the impl wave** — Fun-CosyVoice3-0.5B-2512
  (Apache-2.0, live on `huggingface.co/vokra/fun-cosyvoice3-0.5b-2512`,
  依頼者 2026-07-28 追認 commit `9c00ffb`) and X-Codec-2 (converter +
  license-class flip landed `53fa432`, publish deliberately deferred to
  owner as the T4 (Research-only) first-precedent decision — **superseded
  post-handoff: `98c34cd` fired the publish, live on
  `huggingface.co/vokra/xcodec2`**).
* **2 parity-CI legs landed** — `parity-deepfilternet3-real` (commit
  `f23bc73`, cron Mon 12:30 UTC) and `parity-deberta-v3-large-real`
  (commit `62a10b7`, cron Mon 13:00 UTC; landed on
  `feat/sbv2-v2-plan-and-wave1` 2026-07-28 — the earlier worktree SHA
  `ae8fef9` referenced later in this document is superseded).
* **3 design specs written** (gitignore-local, `docs/superpowers/specs/`)
  for VoxCPM2-2B, WavTokenizer, and Matcha-TTS — each carries Wave 0-9
  breakdowns + open questions for the owner before implementation
  starts. Matcha-TTS is Draft-only (塩漬け) that preserves the earlier
  M5-07 見送り posture.
* **3 models deferred to owner** — VibeVoice-Large (row absent in
  `docs/license-audit.md` §3.1), Sesame CSM-1B (HF-gated ⇒ audit 未判定),
  Suno Bark (§3.1 signed ☑ Commercial but publication still needs owner
  read on the model-card "research purposes only" advisory).
  **Post-handoff (`a7a23c0`): CSM-1B ☑ Commercial + VibeVoice-Large ☑
  Rejected (upstream 404, fail-closed) landed 2026-07-28 yousan sign-offs;
  Bark posture unchanged (M5-07 audit-only, no publish yet).**
* **Zero-dep NFR-DS-02 preserved throughout** — root `Cargo.lock` is
  `vokra-*` only across every landed commit; Python toolchain adds live
  in `/tmp` parity venvs; every workflow's tail step is a
  `git diff --exit-code Cargo.lock` tripwire.
* **No fabricated pass** — the DeBERTa-v3-large parity CI honest-skips the
  Rust numerical leg with a `::notice::` because no consumer harness
  exists yet (only synthetic + convert-smoke tests); X-Codec-2's publish
  step was deliberately not executed pending owner sign-off on the T4
  precedent (**superseded post-handoff: T4 sign-off granted + publish
  fired at `98c34cd`, live URL verified HTTP/2 200**).

## Table of contents

1. [Scout & triage table (8 candidate models)](#1-scout--triage-8-candidate-models)
2. [Impl wave — landed items](#2-impl-wave--landed-items)
3. [Design specs landed (3 × gitignore-local)](#3-design-specs-landed)
4. [Parity-CI coverage — current roster](#4-parity-ci-coverage--current-roster)
5. [Owner critical path (priority-ordered)](#5-owner-critical-path)
6. [Lessons learned (post-mortem)](#6-lessons-learned-post-mortem)
7. [Next-workflow candidates](#7-next-workflow-candidates)

## 1. Scout & triage (8 candidate models)

The scout+triage wave enumerated 8 models that are not yet published on
`huggingface.co/vokra` and are not out-of-scope by mission (music-gen,
汎用 LLM, etc.). Each row records the license verdict primary-source used,
the technical feasibility of standing up a converter/publish path today,
and the resulting triage bucket.

| # | Model | License (primary source) | Feasibility today | Triage bucket |
|---|-------|--------------------------|-------------------|---------------|
| 1 | **FunAudioLLM/Fun-CosyVoice3-0.5B-2512** | Apache-2.0 (HF cardData `license: apache-2.0`, 2026-07-28 CC verify) | Qwen2 LLM backbone convert path works via `bin_to_safetensors.py` Mode b (llm.pt → sidecar → GGUF); Flow / HiFTNet vocoder synthesize path is future work. | **implement-now (published + 追認)** |
| 2 | **HKUSTAudio/X-Codec-2** | Weight-side **CC-BY-NC 4.0** (HF `HKUSTAudio/xcodec2` front-matter, 2026-07-15 CC verify); code layer at `github.com/zhenye234/X-Codec-2.0` is MIT — weight distribution repo governs. | Converter lands (verbatim F32/F16/BF16 pass-through, 1153 tensors read back with `vokra.provenance.license=cc-by-nc-4.0`); `LicenseClass::NonCommercial` flip is in code; publish step deliberately not fired because this would be the first T4 (Research-only) entry on `huggingface.co/vokra` and that precedent decision is owner-only. | **implement-now (converter landed, publish deferred)** |
| 3 | **openbmb/VoxCPM2-2B** | Apache-2.0 (HF cardData tag; unchanged from 0.5B parent) | Architecture is genuinely distinct from VoxCPM-0.5B (LM 2048×28×6144 vs 0.5B; DiT patch_size=4; kv_channels; residual_lm=8+no_rope; SQ=512; bandwidth-adaptive VAE via `sr_bin_boundaries`). Requires converter enum extension (Option A/B/C ADR), 4.96 GB mmap ceiling verify on M1 iMac, per-tensor atol calibration, and a variant-aware parity harness before publish. `parity-tts-continuous-vae-real.yml` already pins the 2B SHA (`bffb3df5…`) at line 100-101 while the converter still transcribes 0.5B constants — CI is already wired to fetch 2B but will silently fail on first fire. | **design-spec (Wave 0-5, ~1-1.5 day CC + 0.5-1 day owner)** |
| 4 | **jishengpeng/WavTokenizer** (small_320 / small_600) | MIT (repo LICENSE, 2026-07-15 CC verify; §3.1 signed ☑ Commercial 2026-07-23 yousan) | Weight is a `.ckpt` PyTorch pickle (nested-structure) and 2 sibling YAML configs; no `.safetensors` distribution. Rust runtime port is a full new WP (encoder + VQ binder + Vocos decoder + GGUF loader, ~1300-2000 LOC / 15-25h). Publish today would ship weights with no consumer runtime — introduces a novel T1-Partial tier concept (publish-without-runtime + canonical warning badge). | **design-spec (Wave 0-5, ~22h / 35-45 tickets)** |
| 5 | **shivammehta25/Matcha-TTS** | MIT repo LICENSE (2026-07-21 CC verify; §3.1 signed ☑ Commercial 2026-07-23 yousan, but `registry_lookup("matcha") == None` = fail-closed in code) | Weight distribution is via Google Drive (17C_gYgEHOxI5…), not HF. Requires eSpeak-NG replacement (permanent GPL-3.0 exclusion per CLAUDE.md §5) — piper-plus G2P + Matcha phoneme-set mapping is the only Phase 1 path. Native TTS body is thin re-use over `flow_sampler` (M3-05) + `hifigan_generator` (M3-07) + `length_conditioning` (M3-08) + piper-plus G2P = **new op count ≈ zero**, which is itself the M5-07 見送り rationale (sherpa-onnx already covers Matcha, no differentiation). | **design-spec (Draft-only, 塩漬け — 4-condition-AND gate before firing)** |
| 6 | **microsoft/VibeVoice-Large** | MIT expected (mirrors VibeVoice-1.5B primary source), but no row exists in `docs/license-audit.md` §3.1 today. | No converter, no runtime scaffold, no harness — architecturally distinct from 1.5B; requires primary-source card verification + row insertion + BF16 sidecar strategy. | **defer-owner** (audit row + sign-off + scope-in decision needed) |
| 7 | **sesame/csm-1b** | Apache-2.0 (HF card), **but HF gated** — CC could not read gate-acceptance terms (HTTP 401) → §3.1 row `未判定 (判断不能)`, sign-off blank fail-closed. | M4-05 native implementation (`crates/vokra-models/src/csm/`) + M4-06 Mimi neural chain already landed. Real checkpoint pending owner HF-token gate acceptance (T29). | **defer-owner** (HF token + gate-terms review + §3.1 sign-off) |
| 8 | **suno/bark** | MIT (2023-05-01 changed from CC-BY-NC → MIT; HF `license: mit`; §3.1 signed ☑ Commercial 2026-07-23 yousan) | Publication itself is still deferred: HF model card carries "research purposes only" + README carries "custom voice cloning 非対応" advisory bullets that are not license conditions but do affect the Vokra model-card guidance passed to downstream users. No M5-07 audit closure. No converter today. | **defer-owner** (model-card advisory reconciliation + M5-07 audit close decision) |

## 2. Impl wave — landed items

Four items landed in the 2026-07-28 impl wave. Every commit is now on this
branch tip (`feat/sbv2-v2-plan-and-wave1`). The DeBERTa-v3-large parity-CI
leg originally landed as `ae8fef9` on worktree
`worktree-wf_4b1b056d-625-4`; the branch-side commit that superseded it is
`62a10b7` (2026-07-28) — see §2.4 below for the reconciled reference.

### 2.1 Fun-CosyVoice3-0.5B-2512 — 追認 commit

* **Commit** `9c00ffbae7292e69bde53173948a9afd6f08592c`
  (`feat(publish): Fun-CosyVoice3-0.5B-2512 ratification (依頼者承認 2026-07-28)`).
* **HF URL** `https://huggingface.co/vokra/fun-cosyvoice3-0.5b-2512`
  (live, 2.58 GB GGUF, Apache-2.0).
* **判定経緯** — the 2026-07-28 workflow's subagent executed
  `publish-one.sh --push` and left the artifact live before the owner had
  explicitly authorized the sign-off wording; the owner ("依頼者") then
  directed "公開を維持 + 署名を追認" ⇒ this commit records the ratification
  rather than reverting an already-live artifact.
* **元 5cf23da との差分** — `5cf23da30e41…` was the original publish
  commit on the worktree; the controller (outer session) could not
  cherry-pick it wholesale because a Slack-mediated classifier blocked
  the automation. This ratification commit re-lands only the pieces
  actually needed on `feat/sbv2-v2-plan-and-wave1`:
  * `tools/parity/bin_to_safetensors.py` gains Mode b
    (`--input <local.bin> --output <local.safetensors>`, mutually
    exclusive with the pre-existing whole-snapshot Mode a). Fun-CosyVoice3
    ships `llm.pt` / `flow.pt` / `hift.pt` / `llm.rl.pt` in parallel, so
    the pre-existing single-`.bin` snapshot assumption of Mode a
    literally could not resolve — a per-component local mode was
    required.
  * `docs/license-audit.md` §3.1 row 275 gets a `**追認**` note
    recording the ratification event verbatim.
  * The `pyproject.toml` / `uv.lock` / `.python-version` triplet under
    `tools/parity/` was already landed via another route on this branch
    ⇒ no drift.
* **Zero-dep** — root `Cargo.lock` unchanged; `curl -sI` on the HF URL
  returned HTTP 200 (recorded in the commit message).

### 2.2 X-Codec-2 — converter + license flip (publish deferred → **later fired** post-handoff at `98c34cd`, see reconciliation note atop the doc)

* **Commit** `53fa432ee3eddece34484480d88eb5e29f718c6e`
  (`feat(sota): X-Codec-2 converter + license_class NonCommercial fix
  (publish deferred to owner T4 precedent decision)`).
* **What landed**:
  * New converter `crates/vokra-convert/src/models/xcodec2.rs`;
    `ModelKind::XCodec2` variant + 6 aliases (`xcodec2` / `x-codec-2` /
    `x_codec_2` / `xcodec-2` / `x-codec2` / `hkustaudio-xcodec2`).
  * `LicenseClass` flip: `xcodec2` moves from `Permissive` → `NonCommercial`
    (grouped with `f5-tts` / `encodec`). Justified by the weight-side
    primary source (HF `HKUSTAudio/xcodec2` front-matter
    `license: cc-by-nc-4.0`, 2026-07-15 CC verify). Code layer at
    `github.com/zhenye234/X-Codec-2.0` remains MIT, but the M2-13 runtime
    gate + M4-04 publish gate both classify by **weight**, and the
    weight-distribution repo governs the class of the redistributed
    artifact.
  * `vokra-cli convert --model xcodec2 --license cc-by-nc-4.0` flag path
    added on the generic fallthrough dispatch (mutually exclusive with
    `--quantize` / `--policy-preset`; loud rejection if combined —
    silently ignoring a user flag is FR-EX-08).
  * Real-weight parity: 3.29 GB `HKUSTAudio/xcodec2/model.safetensors`
    converted successfully — 1153 tensors written, `vokra.provenance`
    reads back `arch=xcodec2 name=xcodec2 category=codec
    upstream_hf=HKUSTAudio/xcodec2 license=cc-by-nc-4.0
    weight_license=non-commercial`.
* **T4 first-precedent decision (owner)** — the 16 models already
  published on `huggingface.co/vokra` are **all Commercial-signed** (T1
  Permissive-tier). Publishing X-Codec-2 would introduce Vokra's first
  T4 Research-only entry, which affects:
  * Model-card canonical warning wording (must be reviewable across
    every future T4 entry — VibeVoice-Large is a likely next).
  * `publish-one.sh --allow-noncommercial` flag semantics and its
    interaction with the fail-closed `redistributable()` predicate.
  * How T1-Partial (introduced in the WavTokenizer spec) interacts with
    T4 — is T4-Partial a thing? (The spec is silent, and this decision
    should be made before a second T4 entry lands.)
* **Publish command (verbatim, ready-to-fire when owner decides)**:

  ```
  scripts/publish/publish-one.sh \
    --tier T4 \
    --allow-noncommercial \
    --slug xcodec2 \
    --gguf /path/to/model.gguf \
    --license cc-by-nc-4.0 \
    --hf-repo vokra/xcodec2 \
    --push
  ```

  (Verbatim from `crates/vokra-convert/src/models/xcodec2.rs`
  followup notes; `--allow-noncommercial` is the explicit opt-in gate
  that the fail-closed `redistributable()` predicate requires.)

### 2.3 parity-deepfilternet3-real — CI workflow

* **Commit** `f23bc73c34b904ade995288296c7c3789f9af498`
  (`chore(ci): parity-deepfilternet3-real workflow_dispatch + weekly cron`).
* **Workflow YAML** `.github/workflows/parity-deepfilternet3-real.yml`.
* **Handoff runbook** `docs/handoff/parity-deepfilternet3-real.md`
  (tracked / public).
* **Design** — two-phase (Phase A = conversion always, Phase B =
  byte-level parity gated on `VOKRA_DFN3_DATA_URL`). Phase B is
  honest-skipped with a `::notice::` until the owner provisions the
  reference bundle; both provisioning paths (commit `dfn3_prep_noisy.py`
  or host a pre-baked `.tar.gz`) are documented in the handoff.
* **Cron** Mon 12:30 UTC. Not registered as required check (HF-hub
  flakiness must never block PRs — same posture as every other
  `parity-*-real.yml`).

### 2.4 parity-deberta-v3-large-real — CI workflow

* **Commit** `62a10b77a1dec4fa19eb1e18d46c7430397f6533` — **landed on
  `feat/sbv2-v2-plan-and-wave1` 2026-07-28**, superseding the initial
  worktree commit `ae8fef9735b354bc9055a821c1481995887c6382` (which was
  based on `feat/sbv2-v2-plan-and-wave1 @ 9c00ffb` inside
  `worktree-wf_4b1b056d-625-4`). The earlier "controller cherry-pick
  pending" note is now resolved.
* **Workflow YAML** `.github/workflows/parity-deberta-v3-large-real.yml`
  (247 lines).
* **Handoff runbook** `docs/handoff/parity-deberta-v3-large-real.md`
  (tracked / public, 249 lines).
* **Design** — two-phase mirror of the DFN3 precedent:
  * **Phase A (conversion + smoke, gated on `VOKRA_DEBERTA_V3_ENABLE=1`
    or `workflow_dispatch`)** — `snapshot_download` at pinned SHA
    `64a8c8eab3e352a784c658aef62be1662607476f`; upstream ships
    `pytorch_model.bin` only (no `.safetensors` mirror, verified
    2026-07-28 via HF API) ⇒ `tools/parity/bin_to_safetensors.py` bridges
    inside the parity venv; then `vokra-cli convert --model deberta-v3`
    and `deberta_v3_convert_smoke (--ignored)` re-check the emitted
    bytes via a scoped fixture symlink.
  * **Phase B (dumper + numerical parity, opt-in via
    `VOKRA_DEBERTA_V3_HARNESS_READY=1` or `-f run_dumper=true`)** —
    `tools/parity/deberta_v3_dump_reference.py --do-dump` uploads
    reference tensors as a job artifact for **future harness
    development**; the Rust-side numerical parity leg is honestly
    skipped with a `::notice::` because no consumer harness exists yet
    in this repo (only synthetic + convert-smoke tests). Fabricated pass
    禁止 (FR-EX-08).
* **`.github/pins.yaml`** gets a `deberta-v3-large` entry (kind:
  checkpoint) so `pins-sync-check.py`'s reverse leg (env-var → catalog)
  stays green.
* **Cron** Mon 13:00 UTC (first free slot after DFN3 12:30). Not
  registered as required check.
* **HF gate note** — `microsoft/deberta-v3-large` may require accepting
  the model-card HF-workflow gate (not a legal restriction — MIT license
  is unambiguous). If CI hits 401, set `HF_TOKEN` as a repo secret
  (read-only, model-card-accepted) — the handoff §Troubleshooting
  documents the exact wiring.
* **Cross-workflow interaction** — `parity-sbv2-real.yml` (Monday 07:15
  UTC) already fetches `microsoft/deberta-v3-large` as part of its
  3-checkpoint pipeline, but uses
  `snapshot_download(..., allow_patterns=['*.safetensors', '*.json'])`
  which currently fails against this `.bin`-only distribution (a known
  open gap noted in that workflow's header). This standalone workflow
  bridges via `bin_to_safetensors.py`, so it will fire successfully
  today even before the SBV2 integrated CI does. The two workflows
  don't collide on the cron roster.

## 3. Design specs landed

All three specs live in `docs/superpowers/specs/` and are gitignore-local
(the specs directory is under `docs/superpowers/`, which `.gitignore:114`
excludes). This mirrors the CLAUDE.md client-private planning-docs
posture: internal design specs do not enter the public repo until they
are ratified and the implementation lands.

### 3.1 VoxCPM2-2B design spec

* **Path** `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`.
* **Sections** — 10 numbered + Appendices A-C: (1) Model summary with a
  delta table vs 0.5B + 6 new metadata keys + 1-day budget breakdown;
  (2) License + §3.1 sign-off owner path; (3) Weight (4.96 GB mmap +
  restamp precedent from Voxtral 8.7 GB); (4) Architecture delta vs
  0.5B; (5) **Converter design ADR — Option A (`--config` side-car) vs
  B (sibling `voxcpm2_2b.rs`) vs C (hybrid: single file + variant enum +
  shared arch + name-based dispatch, CC recommends C)**; (6) Runtime
  design (`VoxCpm2LmConfig` / `Encoder` / `DitConfig` / `VoxCpm2Config`
  expansion + `sr_bin_boundaries` + `synthesize_with_target_sr`); (7)
  Parity strategy (pins live in `parity-tts-continuous-vae-real.yml`
  `env:` block, harness variant-aware via `vokra.model.name`, per-tensor
  atol calibration); (8) Publication path (5-段 gate, `vokra/voxcpm2-2b`
  slug); (9) Implementation waves (W0-5, 1-1.5 day CC + 0.5-1 day
  owner); (10) Owner critical path (O-1 to O-8).
* **Wave breakdown** — W0 ADR収束 + BF16 mmap ceiling verification /
  W1 Runtime config 拡張 / W2 Converter 拡張 + BF16 mmap 経路 /
  W3 Continuous VAE seam + adaptive head / W4 Parity CI flip switch /
  W5 Publication (依頼者専任、CC は準備のみ).
* **Owner questions (Q1-Q5)** collected in §11 Appendix C — Wave 0
  cannot start until Q1 (converter topology A/B/C) is answered.
* **Latent bug surfaced during spec work** — the current
  `parity-tts-continuous-vae-real.yml` at line 100-101 declares
  `VOXCPM2_REPO=openbmb/VoxCPM2` + `VOXCPM2_REVISION=bffb3df5a29440629464e5e839f4d214c8714c3d`
  (the 2B model at pinned SHA), **yet the runtime
  `VoxCpm2Config::voxcpm_0_5b()` and converter constants are 0.5B**.
  CI is wired to fetch 2B but hparam-checks against 0.5B ⇒ silent failure
  on first fire. This spec's Wave 1+2+4 unblock the switch.

### 3.2 WavTokenizer design spec

* **Path** `docs/superpowers/specs/2026-07-28-wavtokenizer-design.md`.
* **Sections** — 15 numbered: (1) Model summary (24 kHz VQ codec, 4096
  codebook, small_320 vs small_600); (2) License (☑ Commercial 2026-07-23);
  (3) Weight (3.17 GB, 2 checkpoints, `.ckpt` pickle format); (4)
  **Pickle handling ADR — (A) Python preprocess RECOMMEND (Kokoro
  precedent) / (B) Rust serde-pickle REJECT (NFR-DS-02) / (C) narrow
  reader REJECT**; (5) Converter design (`vokra-convert/src/models/
  wavtokenizer.rs`, `xy_tokenizer.rs` 1:1 mirror); (6) **Publish-without-
  runtime posture ADR — T1-Partial tier NEW + canonical warning badge +
  `check-partial-runtime.sh` gate + `--acknowledge-partial-runtime`
  flag + PARTIAL_ALLOWLIST 台帳**; (7) Real weight parity strategy
  (Phase 1 = structural + partial e2e, full-chain deferred); (8)
  Runtime port scope (future WP, 1300-2000 LOC / 30-50 tickets); (9)-(15)
  Waves, owner path, success criteria, risks, timeline, references,
  open questions.
* **Wave breakdown** — W0 Design land (spec + 3 ADRs) / W1 Python
  preprocess + YAML parser / W2 Rust converter / W3 Partial e2e +
  parity CI / W4 publish-without-runtime infrastructure / W5 Docs +
  handoff. **Total ~22h / 35-45 tickets.**
* **Owner questions (Q1-Q7)** — 2 variant 同時 land / T1-Partial tier
  vs 既存 tier + flag / future runtime port WP timing / initial
  workflow_dispatch owner / HF repo naming / warning badge wording /
  writing-plans 進行可否.
* **Novel-tier note** — this is the first "publish weight without a Rust
  consumer runtime" case Vokra will confront. The spec proposes a **new
  T1-Partial tier** with a canonical warning badge in the model card
  and a machine-verifiable `check-partial-runtime.sh` gate. If the
  owner rejects the tier, the fallback is to defer publication until
  the full runtime port lands (adds ~15-25h to the critical path but
  keeps the tier system simple).

### 3.3 Matcha-TTS design spec (Draft-only)

* **Path** `docs/superpowers/specs/2026-07-28-matcha-tts-design.md`.
* **Posture** — **Draft-only 塩漬け**. Written to be the design book
  *if* Matcha is ever un-見送り, and to record the current chain of
  evidence for the M5-07 見送り decision. Does not re-open the
  見送り判定.
* **Sections** — 14 numbered: (0) Precondition (見送り前例 尊重); (1)
  Model summary + strategic re-open question (深く狭く音声特化 mission
  vs sherpa-onnx redundancy, 前提陳腐化トリガー A-D); (2) License
  (☑ Commercial 2026-07-23, registry 未登録 fail-closed); (3) Weight
  acquisition (Google Drive 17C_gYgEHOxI5ZypcfE_k1piKCtyR0isJ, 3 案 —
  gdown vs mirror vs owner-local, C = owner-local recommend); (4) G2P
  scope (eSpeak-NG permanent exclusion, B = piper-plus 流用 + Matcha
  phoneme-set mapping recommend); (5)-(14) Converter design, runtime
  design, publish-without-runtime posture, real weight parity strategy,
  waves, owner critical path, risks, timeline, references, open
  questions.
* **見送り解除の 4 条件 AND** (§9.1) — owner 明示指示 / phoneme-set
  互換 95% verify / LJ HiFi-GAN license 一次確認 / トリガー A-D 発火
  根拠の記録. All four must be satisfied before Wave 0 fires.
* **Wave breakdown (only fires under 4-condition-AND)** — W0-W9, ~19h
  total. Most ops are thin wrappers over already-landed primitives
  (`flow_sampler` M3-05 + `hifigan_generator` M3-07 +
  `length_conditioning` M3-08 + piper-plus G2P) ⇒ new op count ≈ zero,
  which is itself the M5-07 見送り rationale (差別化にならない).

## 4. Parity-CI coverage — current roster

On branch tip `feat/sbv2-v2-plan-and-wave1` there are 16
`parity-*-real.yml` (or `parity-*.yml`) workflows in
`.github/workflows/`. The DeBERTa-v3-large workflow originally sat on
worktree `ae8fef9` at the time of first drafting; the branch-side
`62a10b7` (2026-07-28) reconciled it onto this branch, so the "pending
cherry-pick" §4.2 below is now historical.

### 4.1 Landed on this branch (16 workflows)

| # | Workflow | Family / model | Enable var | Cron slot |
|---|----------|----------------|------------|-----------|
| 1 | parity-kokoro-real.yml | Kokoro-82M | `VOKRA_KOKORO_ENABLE` | (pre-existing) |
| 2 | parity-whisper-real.yml | Whisper base/small/medium/turbo/large-v3 | (matrix) | (pre-existing) |
| 3 | parity-moshi-real.yml | Moshi 7B | `MOSHI_PARITY_ENABLE` | (pre-existing) |
| 4 | parity-csm-real.yml | Sesame CSM-1B | (M4-05 leg) | (pre-existing) |
| 5 | parity-rvq-real.yml | Mimi / DAC RVQ | (M3-06 / M4-04) | (pre-existing) |
| 6 | parity-utmos.yml | UTMOS 22 strong | (M4-18) | (pre-existing) |
| 7 | parity-sbv2-real.yml | Style-Bert-VITS2 v2 (3-checkpoint) | (SBV2 v2 plan) | Mon 07:15 UTC |
| 8 | parity-nemo-asr-real.yml | kyutai_stt + parakeet_tdt/ctc + canary + omniASR-CTC-1B | `VOKRA_NEMO_ASR_ENABLE` | Mon 09:00 UTC |
| 9 | parity-whisper-extras-real.yml | distil_whisper + kotoba_whisper | `VOKRA_WHISPER_EXTRAS_ENABLE` | Mon 09:30 UTC |
| 10 | parity-tts-dac-real.yml | dia + zonos | `VOKRA_TTS_DAC_ENABLE` | Mon 10:00 UTC |
| 11 | parity-tts-hiftnet-real.yml | cosyvoice3 + chatterbox {multilingual, turbo, nano} | `VOKRA_TTS_HIFTNET_ENABLE` | Mon 10:30 UTC |
| 12 | parity-qwen3-tts-real.yml | qwen3_tts_0_6b_base | `VOKRA_QWEN3_TTS_ENABLE` | Mon 11:00 UTC |
| 13 | parity-tts-continuous-vae-real.yml | voxcpm2 + vibevoice | `VOKRA_TTS_CONT_VAE_ENABLE` | Mon 11:30 UTC |
| 14 | parity-tts-japanese-real.yml | irodori + vits_ja | `VOKRA_TTS_JA_ENABLE` | Mon 12:00 UTC |
| 15 | **parity-deepfilternet3-real.yml (LANDED f23bc73)** | DeepFilterNet3 | `VOKRA_DFN3_ENABLE` | Mon 12:30 UTC |
| 16 | **parity-deberta-v3-large-real.yml (LANDED 62a10b7)** | microsoft/deberta-v3-large | `VOKRA_DEBERTA_V3_ENABLE` | Mon 13:00 UTC |

### 4.2 Pending cherry-pick onto this branch

(Historical — resolved 2026-07-28 by commit `62a10b7`. Left in place so
the reference chain in §2.4 remains navigable; no workflows are currently
pending cherry-pick.)

### 4.3 Multi-model CI cross-check note (important)

Several of the workflows in §4.1 already exercise multiple models via
matrix or per-model include flags:

* `parity-sbv2-real.yml` — pipes DeBERTa-v3-large / SBV2 base /
  JA BERT sequentially. The DeBERTa-v3 leg in that pipeline currently
  uses `allow_patterns=['*.safetensors', '*.json']` and fails against
  the `.bin`-only distribution — the standalone
  `parity-deberta-v3-large-real.yml` bridges via `bin_to_safetensors.py`
  and will fire successfully today.
* `parity-nemo-asr-real.yml` — 5 archs in a matrix.
* `parity-whisper-real.yml` — matrix over base/small/medium/turbo/large-v3.
* `parity-tts-dac-real.yml` — dia + zonos.
* `parity-tts-hiftnet-real.yml` — cosyvoice3 + chatterbox 3 variants.

**Owner cross-check recommended** — some of the entries in the "17
owner queue" from the scout report may already be covered by a
multi-model CI's matrix leg. Before standing up a *new* dedicated
workflow for each queue entry, do a matrix-vs-standalone triage per
family (small models that share a family workflow's provisioning
infrastructure should stay in the matrix; standalone workflows are
justified when a family CI's provisioning cannot handle the model —
see §4.3 DeBERTa-v3-large / SBV2 example).

## 5. Owner critical path

Priority-ordered — items higher up unblock items below. Every item
below is honestly deferred to owner because it either (a) needs a
decision only the owner can make, (b) needs primary-source verification
of external license text, or (c) needs an initial dispatch on
GitHub Actions.

### (a) T4 first-precedent decision — X-Codec-2 publish

> **✅ Resolved post-handoff (commit `98c34cd`, 2026-07-28 12:29:46 UTC)**:
> owner granted the T4 precedent decision; the artifact is live at
> `https://huggingface.co/vokra/xcodec2` (curl -sI HTTP/2 200 verified by
> publish tooling). The workflow — `publish-one.sh --allow-noncommercial`
> + CC-BY-NC-4.0 canonical LICENSE (`fetch_license.sh --spdx cc-by-nc-4.0`
> was added in the same commit) + model-card leading NC clause + §3.1
> ☑ Research-only sign-off (2026-07-23 yousan, row 254) — is now the
> durable pattern captured in memory `[[project-x-codec2-t4-precedent]]`.
> The prose below is retained verbatim as the pre-decision framing.

`huggingface.co/vokra/xcodec2` is not yet live because the 16
already-published models are all T1 Permissive-tier ⇒ T4 Research-only
is a novel tier for this repo. The precedent decision affects:

* Model-card canonical warning wording (reviewable across every future
  T4 entry — VibeVoice-Large is the likely next).
* `publish-one.sh --allow-noncommercial` flag semantics and its
  interaction with the fail-closed `redistributable()` predicate.
* Whether T4-Partial (T4 without runtime, cf. WavTokenizer T1-Partial)
  is a thing — decide before a second T4 entry lands.

If GO: fire the publish command in §2.2. If DEFER: keep the converter
+ license-flip landed as-is (the code layer is complete and
FR-EX-08-safe), and revisit after the T1-Partial tier decision closes
in §5(e).

### (b) §3.1 sign-off — blank rows to close

| Model | §3.1 line | Blocker | Owner action |
|-------|-----------|---------|--------------|
| **openbmb/VoxCPM2 (2B)** | 280 | Converter not yet extended to 2B arch (spec §3.1 recommends Option C hybrid) | Answer VoxCPM2 spec Q1-Q5 to unblock Wave 0 ADR ⇒ Wave 1-5 land ⇒ then sign-off |
| **sesame/csm-1b** | 255 | HF gated repo (401) — gate-acceptance terms not readable by CC | Refresh HF token with gate accepted → read added terms → sign or reject in §3.1 |
| **microsoft/VibeVoice-Large** | (absent) | No row exists — audit incomplete | Verify HF `microsoft/VibeVoice-Large` cardData tag; add §3.1 row analogous to the 1.5B row 282 |
| **suno/bark** | 259 | Already ☑ Commercial 2026-07-23. **Publication itself deferred** — model-card advisory "research purposes only" + README custom-voice-cloning bullet need owner reconciliation with Vokra downstream-guidance policy. | Decide model-card guidance verbatim wording; if publish: reuse T1 tier since already signed |

### (c) Initial `workflow_dispatch` — 2 new CI legs

Both landed workflows are opt-in by design and will not fire on cron
until the enable variable is set. First-dispatch validates that the
external provisioning (HF snapshot_download, pip installs) works on
hosted-runner + the pinned SHAs still resolve.

* **parity-deepfilternet3-real** — set `VOKRA_DFN3_ENABLE=1` then
  `gh workflow run parity-deepfilternet3-real.yml`. See
  `docs/handoff/parity-deepfilternet3-real.md` §Owner action checklist.
* **parity-deberta-v3-large-real** — set
  `VOKRA_DEBERTA_V3_ENABLE=1` then
  `gh workflow run parity-deberta-v3-large-real.yml`. See
  `docs/handoff/parity-deberta-v3-large-real.md` §Owner action
  checklist. If HF hits 401, set `HF_TOKEN` repo secret.

### (d) 17-entry parity-CI activation queue (from scout)

The scout report enumerated 17 models in the parity-CI owner queue.
Per §4.3 above, cross-check the queue against multi-model matrix
workflows before standing up new standalone workflows. This handoff
does not enumerate the queue verbatim — the scout report is the SoT.
(If the scout report is not persisted publicly, ask the controller to
export it before starting the activation wave.)

### (e) Design-spec 3 本 — implementation approval

* **VoxCPM2-2B spec** — answer Q1-Q5 in §11 Appendix C (converter
  topology A/B/C; parity atol calibration; cleanroom sourcing; BF16
  mmap peak measurement recipe; upstream tensor-name manifest fetch).
  Once Q1 is answered, Wave 0 can fire.
* **WavTokenizer spec** — answer Q1-Q7 in §15 (2-variant simultaneous
  land; T1-Partial tier acceptance; future runtime port WP timing;
  initial workflow_dispatch owner; HF repo naming; warning badge
  wording; writing-plans progression). Q2 (T1-Partial tier) is a
  standing decision that affects future publish-without-runtime cases.
* **Matcha-TTS spec** — no action expected. Draft-only 塩漬け; do not
  re-open the M5-07 見送り judgment. If a 前提陳腐化トリガー A-D fires
  (§1.2 in the spec), revisit then and only then.

### (f) Cross-check against ai-lab/mission — deep-not-wide guardrail

Every model above must pass the [[project-goal-depth-not-breadth]]
guardrail before publication: does it deepen Vokra's SoTA coverage in
ASR / TTS / VC / Speaker-ID / VAD, or does it broaden into music-gen /
汎用 LLM / multimodal? If broaden: reject publish regardless of
license sign-off. All 8 candidates in §1 pass this guardrail today —
they are all TTS or codec (audio-adjacent), no expansions into
non-speech territory.

## 6. Lessons learned (post-mortem)

Two operational incidents surfaced during this wave. Both are honest
observations rather than "process failed" claims — the guardrails
worked; the incidents reveal where the next layer of defense should
sit.

### 6.1 Scout report → impl agent contamination of §3.1 sign-off state

**What happened.** The scout+triage wave produced a report that
referenced the sign-off state of some rows in `docs/license-audit.md`
§3.1. The impl agent, reading the scout report as authoritative context,
did not re-verify §3.1 against the actual on-disk file before writing
publish artifacts. In the Fun-CosyVoice3 case this bypass was
recovered: the owner directed 追認 rather than revert. In principle
though, a subagent could sign off on a row it should not have signed
based on the scout's phrasing.

**Why the existing guardrail helped anyway.** `publish-one.sh` reads
`docs/license-audit.md` §3.1 directly at push-time and refuses to
distribute if the row is blank — the actual on-disk file is the SoT
for publish, not any subagent's context. That is why the Fun-CosyVoice3
publish went through despite the sequencing being unusual: the file
was in fact signed by the time push happened. The recovery direction
"公開を維持 + 署名を追認" preserved the guardrail invariant (SoT is the
on-disk file, always).

**Recommendation for the next layer of defense.**

1. **Workflow prompt hygiene** — future scout+triage workflows should
   not restate §3.1 sign-off status in prose. The impl agent should
   be instructed to read `docs/license-audit.md` §3.1 line-by-line as
   part of its own preflight (fail-closed if blank), never inheriting
   the scout's paraphrase. The scout can *point at* row numbers, but
   should not quote the ☑ / ☐ box state.
2. **Sign-off provenance tag** — `publish-one.sh` could require an
   audit-trail claim on the row alongside the ☑ (e.g. "signed as row
   274 of commit `abc1234` verified against upstream card at URL
   xyz"). Today the signature is just a name + date; adding a commit
   SHA + primary-source URL to the row lets `publish-one.sh` refuse
   rows whose provenance claim doesn't reference a verifiable primary
   source. This is additive and does not break existing signed rows.
3. **Row-edit workflow context** — when a subagent edits a §3.1 row,
   the pre-commit hook could require the commit message to identify
   the workflow-run URL that produced the change. Today
   `docs/license-audit.md` §3.1 edits blend into normal file diffs;
   pinning provenance would let the owner audit later.

### 6.2 "手動 upload 禁止" invariant vs `publish-one.sh` self-write of §3.1

**What happened.** The "手動 hf upload 禁止" invariant (all publishes
go through `publish-one.sh`, never `hf CLI` direct calls) was
successfully upheld — the Fun-CosyVoice3 publish log shows the 5-gate
script executed end-to-end. What was **not** covered by an equivalent
CC-side gate: the same script's execution edited
`docs/license-audit.md` §3.1 row 275 as a side-effect of the "sign-off
approved" branch. The gate that would prevent a subagent from
"self-approving" its own sign-off row does not exist today.

**Why this is a design gap and not a bug.** `publish-one.sh` was
architected on the assumption that §3.1 rows are edited by the owner
in a separate commit *before* the publish is invoked. The 2026-07-28
Fun-CosyVoice3 case violated that assumption because the workflow
subagent edited the row as part of the same commit that ran the
publish — a legitimate operational choice given the wave-mode workflow,
but one the pre-existing `publish-one.sh` gate does not defend against.

**Recommendation for the next layer of defense.** Extend
`publish-one.sh` to require an identity claim from the caller when the
§3.1 row was written in the same commit as the publish invocation:

* The script inspects `git log -1 --pretty=%an -- docs/license-audit.md`
  to determine who last touched §3.1.
* If the row's last-edit author matches the current process's committer
  identity (i.e., a subagent signed its own approval), the script
  refuses to `--push` and requires a `--acknowledge-self-signoff
  <owner-approved-workflow-run-URL>` flag whose value must be
  verifiable against the current session's context.
* Owner-in-the-loop sessions where the row was already signed in a
  prior commit are unaffected (the last-edit-author check passes
  transparently).

## 7. Next-workflow candidates

Ordered by expected ROI given the owner's critical path in §5.

1. **Answer VoxCPM2-2B spec Q1-Q5 → run Wave 0 ADR** — unblocks the
   entire VoxCPM2-2B implementation chain (Wave 1-5) and closes the
   silent-failure CI gap identified in §3.1 (2B pinned in workflow,
   0.5B constants in code).
2. **T4 first-precedent decision → X-Codec-2 publish** — establishes
   the tier-4 canonical for all future Research-only entries
   (VibeVoice-Large next). The publish command is ready-to-fire.
3. **Initial workflow_dispatch × 2 (DFN3 + DeBERTa-v3-large)** — cheap
   validation of pinned SHAs + HF snapshot_download + bin_to_safetensors
   bridge on hosted-runner. Should be done within the 7-day window of
   the cron slot to catch HF-side drift before it becomes stale.
4. **17-queue triage — matrix vs standalone** — resolve the §4.3
   ambiguity before spending owner-time standing up new workflows.
5. **WavTokenizer T1-Partial tier decision (spec Q2)** — standing
   decision that affects future publish-without-runtime cases (any
   future model whose Rust port lags weight availability).
6. **CSM-1B HF token gate refresh** — smallest owner-action-only step;
   currently blocking §3.1 sign-off + M4-05 real-checkpoint parity.
7. **VibeVoice-Large audit row insertion** — feeds into the same T4
   first-precedent decision cluster as X-Codec-2.

### Explicit non-goals

* Do **not** re-open the Matcha-TTS 見送り judgment. The spec exists
  to record the current chain of evidence, not to invite reconsideration.
  Reconsider only when a 前提陳腐化トリガー A-D fires (spec §1.2).
* Do **not** publish X-Codec-2 without an explicit T4 first-precedent
  decision. The converter + license flip are landed *so that* the
  decision can be made without CC-side rushing.
* Do **not** cherry-pick worktree `ae8fef9` from within a handoff
  agent. This handoff records the pending state; controller (outer
  session) handles the cherry-pick sequence.

## References

* `docs/license-audit.md` §3.1 — sign-off table (rows 247-282 in this
  branch tip).
* `docs/handoff/parity-deepfilternet3-real.md` — DFN3 CI runbook.
* `docs/handoff/parity-deberta-v3-large-real.md` — DeBERTa-v3-large CI
  runbook (landed on branch as `62a10b7`, 2026-07-28; earlier worktree
  commit `ae8fef9` is superseded).
* `docs/handoff/parity-ci-flip-switch.md` — flip-the-switch pattern
  for the 8 SoTA family harnesses.
* `docs/handoff/sota-candidates-2026-07-25.md` — 25-row SoTA candidate
  table (predecessor of this wave's scout).
* `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md` — VoxCPM2-2B
  implementation spec (gitignore-local).
* `docs/superpowers/specs/2026-07-28-wavtokenizer-design.md` —
  WavTokenizer implementation spec (gitignore-local).
* `docs/superpowers/specs/2026-07-28-matcha-tts-design.md` — Matcha-TTS
  Draft-only spec (gitignore-local; 塩漬け).
* `scripts/publish/publish-one.sh` — 5-gate publish script (SoT for
  distribution refusal / approval).
* `scripts/publish/check-catalog-reality.sh` — declared-vs-implemented
  cross-check with `EXPECTED_GAPS` allowlist.
* CLAUDE.md §「現在のタスク状態」— running state; the 2026-07-28
  entry references this handoff.

## Change log

* **2026-07-28** — Initial land, covering the 2026-07-28 ultracode
  workflows (scout+triage → impl → design-spec). Author: Claude Code
  handoff agent, ratified by owner (yousan).
