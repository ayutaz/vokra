# Codex operations handoff (2026-08-28)

> **Superseded current-state note (2026-08-31):** This document remains a
> dated 2026-08-28 historical record. For the current Mac CPU/Metal baseline,
> branch state, VAST evidence, and remaining work, read
> [`docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md`](mac-cpu-metal-full-coverage-2026-08-28.md).
> The ordered current route is the
> [`Mac CPU/Metal completion plan`](mac-cpu-metal-completion-plan-2026-08-30.md),
> reconciled at code `9f69277d8a0d5df574c1ee95563bd1f005de91d0` and
> evidence/package checkpoint `5cd97d124bc9eb9d2bb7b0367541dcd1492e4d1e`.
> Those checkpoints are historical workspace `0.2.0` evidence. The active
> branch is workspace `0.3.0`; immediately before this documentation refresh
> its remote head was `d8a93bc3acdb8f9648ecb8dd37ef41657fbf425b` in open PR #79,
> with 109 passing checks, 13 expected skips, and no failures or pending checks.
> The historical baseline and session narrative below are intentionally not
> rewritten into present tense.

## Current baseline

- Repository: `ayutaz/vokra`
- Current branch: `main`
- Baseline commit: `e3b12c450318a884961a9fa430b5ec69fc67b545`
  (`feat(mac): complete CPU and Metal coverage for published models (#74)`)
- Workspace version: **`0.2.0`** (was `0.1.0`; see §4)
- PR #74 merged 2026-08-28 06:56 UTC by `ayutaz`, squash, 937 files / +246k
  lines, with **104 checks green and 0 failing** (`mergeStateStatus = CLEAN`).

This file records what the Claude Code session carried into Codex. Reusable
policy belongs in `AGENTS.md`, `.agents/skills/` and the hooks; this file keeps
the dated evidence, the open items, and the traps that cost time.

## 1. What this session was asked to do

The task inherited from the previous Codex session, in the owner's words:

> 1. 今回のPRに関係するCI失敗を修正 2. VASTで再検証 3. GitHub CI、特にMac Metalを成功させる

All three are done. A fourth item — the root-cause fix for `dependency-review`
— was added mid-session by owner decision (option A, "オラクル一式を最新版に
更新する") after the allow-list approach was shown to be unbounded.

## 2. Commits landed (12, in `26f13f3a..5f32fbcd`)

| Commit | Subject |
| --- | --- |
| `5ef0c249` | `fix(deps): update AST parity oracle to transformers 5.5.0` |
| `e1e01f1b` | `fix(ci): reconcile zoo marker and wheel prototype count` |
| `615abb82` | `test(capi): align t3 with backend-honoring silero vad` |
| `180f5883` | `docs(ci): record the silero backend shift in the capi leg` |
| `59891c87` | `fix(ci): allow-list the parity-oracle torch advisories` |
| `2fc6d31b` | `fix(parity): upgrade the pinned reference toolchains` |
| `e815dd8b` | `fix(ci): narrow the advisory allow-list to the blocked oracles` |
| `839d8bfe` | `fix(parity): raise the sidecar hydra-core floor` |
| `fd09fbd7` | `fix(parity): keep GPL text-unidecode out of the sidecar tree` |
| `32efad34` | `fix(parity): load parler torchaudio from the torch index` |
| `4e59e12b` | `chore(workspace): open the 0.2.0 line` |
| `5f32fbcd` | `fix(workspace): carry the 0.2.0 bump into locks and GGUF sidecars` |

`5ef0c249` is the previous Codex session's own VAST-verified AST work, which had
been left uncommitted; it was committed byte-identical to what that session had
sha256-verified on VAST, with the pyannote fix staged separately to keep it so.

## 3. The Metal failure was a stale test premise, not a bug

`gpu-backends (macos-latest, metal)` failed on
`t3_unavailable_backend_never_falls_back_to_cpu` with
`create_from_file_with_options(METAL) returned VOKRA_OK for an available backend
over a CPU-only arch (Silero VAD)`.

Commit `53c4823e`, inside this same branch, had made Silero VAD
backend-parameterised (`with_backend` plus `Compute::for_backend(SILERO_HOT_OPS)`),
and Metal covers both `HotOp::Conv1d` and `HotOp::Gemv`. `VOKRA_OK` is therefore
the correct result and the test's premise had gone stale. `reject_cpu_only_backend`
now guards only `ARCH_MIMI` and `ARCH_NANOCODEC`.

The fix accepts either outcome, and on `VOKRA_OK` **drives the session** and
validates the returned probability vector against the CPU reference length and
range, rather than accepting the status alone. That preserves the anti-rubber-stamp
property the 2026-08-14 review installed.

## 4. The 0.2.0 line

`cargo-semver-checks` failed because the branch changes the public Rust API while
both sides read `0.1.0`: `ModelKind` gained roughly a hundred variants and the
inserted ones shifted every later implicit discriminant, `RmvpeReport` gained a
field, and `FrcrnReport` / `Emotion2vecReport` became `Copy`. All intended.

`CHANGELOG.md` closed `[0.1.0]` on 2026-08-23 with the v0.5 (M2), v0.9 (M3) and
**v1.0-rc (M4)** milestone work already inside it, so the milestone labels are
process names, not crate versions. Under 0.x a minor bump is the breaking bump,
so `0.2.0` is the next line.

**Trap**: `docs/abi-changelog.md` still carries an older trajectory
(`0.1.0 → 0.9.0-* → 1.0.0-rc.* → 1.0.0`) that schedules the Cargo bump at the M3
and M4 tag-preparation steps. That document is stale relative to `CHANGELOG.md`.
Reading it first produces the wrong answer (`1.0.0-rc.1`), which is what happened
here before `CHANGELOG.md` was checked. Consider reconciling the two.

Nothing is released: **0 git tags, 0 GitHub releases, and no `vokra-*` crate on
crates.io** (`vokra-core`, `vokra-convert`, `vokra-capi` all return HTTP 404).

### 4.1 What the version bump dragged with it

Two consequences were not obvious up front and both broke CI:

- **Excluded-workspace lockfiles.** The seven workspaces under `integrations/`
  keep their own `Cargo.lock`, each recording the vokra path crates at `0.1.0`,
  so every `--locked` build there stopped resolving. Regenerated;
  `cargo metadata --locked` passes for all seven.
- **GGUF producer stamp.** `GgufWriter` writes
  `general.schema_producer = "vokra-core <CARGO_PKG_VERSION>"` so the stamp always
  describes the build that produced the bytes
  (`vokra-core::gguf::schema::tests::every_builder_written_gguf_is_stamped` pins
  this). The bump therefore changes the bytes of every regenerated GGUF and
  invalidates the committed SHA-256 sidecars.

`vokra-bert` and `vokra-kws-micro` do **not** inherit the workspace version but
are pinned by it through `[workspace.dependencies]`, so they must move with it or
the workspace stops resolving. The same applies to the `vokra-core` requirement
carried by `vokra-vad-micro` and `vokra-kws-micro`.

## 5. Parity-oracle dependency upgrade

Every lockfile in the repository — 31 files, 483 unique package/version pairs —
was checked against the GitHub Advisory Database and PyPI. 16 pins carried a
moderate-or-higher advisory. Seventeen trees were affected; fourteen moved and
now resolve clean. Transformers to `5.5.0`, torch to `2.13.0`, `sentencepiece` to
`0.2.2`, `setuptools` to `84.0.0`, `hydra-core` to `1.3.5`, `protobuf` to
`7.36.0`. Each dumper's fail-closed `TRANSFORMERS_VERSION` guard and Bark's
pinned Transformers source revision (`c1c34249…`, the v5.5.0 commit) moved with
them.

`dac` moved only `protobuf`, through a `tool.uv` override because
`descript-audiotools` caps it below 3.20. Diffing the lockfiles confirmed no
package on the numeric path changed, so its committed 16/24/44.1 kHz fixtures
stay valid.

### 5.1 Three trees are blocked upstream

| Tree | Upstream pin | Residual |
| --- | --- | --- |
| `qwen3_asr` | `qwen-asr==0.0.6` (latest) requires `transformers==4.57.6` | 3 advisories |
| `parler_tts` | `parler-tts==0.2.2` requires `transformers==4.46.1` | 16 advisories |
| `xcodec2` | `xcodec2==0.1.5` (latest) requires `torch==2.5.0` | 4 advisories |

`xcodec2` would accept Transformers 5.5.0, but its torch pin keeps the tree
flagged either way and the bump would void five committed fixtures, so it was
left alone. The union — 20 exact GHSA ids — is allow-listed in
`.github/workflows/ci-security.yml` with each blocking pin named. torch and
Transformers appear nowhere outside `tools/parity/**`: the Rust runtime carries
no dependencies (`scripts/check-zero-deps.sh`) and the published Python wheel
declares `dependencies = []`.

## 6. VAST evidence

| Run | Instance | Result |
| --- | --- | --- |
| Workspace verification | `48894896` (RTX 3090, 64 core) | fmt, clippy `-D warnings`, `cargo test --workspace` **7606 passed / 0 failed**, CUDA `c_abi_backend_options` 13 passed, zoo manifest gate OK. Log SHA-256 `7804787e3be084645d332577b70169b29e24d9457f8930b2232270236ff85c89` |
| Oracle verification | `48950897` (20 core, 125 GB) | 16 trees installed from their committed lockfiles, executed every `transformers`/`torch` import each dumper declares, ran its argument parser: **16 pass / 0 fail**. Log SHA-256 `f4f295abe4140bb6d87087608082657a0c4ac651fe170ad63af71548d830c1c3` |

Both instances were destroyed after log recovery and the account was verified at
zero running instances.

The oracle run found one real defect. `parler_tts` could not start: torch
resolved from the `pytorch-cpu` index while `torchaudio` came from PyPI, so
`_torchaudio.abi3.so` could not load. **The split predates this branch's
dependency work** — it was already there at `torch 2.5.1+cpu` — so that oracle
had never run on this branch. Fixed by routing `torchaudio` through the same
index.

**Trap**: version arithmetic sends this the wrong way. `torch 2.13.0` with
`torchaudio 2.11.0` looks mismatched but imports cleanly in six other trees where
both wheels come from one index. Only measuring each tree separated the two
cases.

## 7. Open items

- [ ] **`tests/fixtures/sbv2/chinese-roberta-wwm-ext-large.gguf.sha256`** still
      holds its `0.1.0` value. Its artefact is only rebuilt when
      `parity-sbv2-real` is dispatched with `RUN_ZH=true`, so no run has produced
      the `0.2.0` hash. The next ZH dispatch fails closed on it; re-pin from that
      run. The other three sidecars were re-pinned to what the
      `4e59e12b` run measured.
- [ ] **Numerical verification of the upgraded oracles.** §6 exercises
      installation and the import surface only. No reference tensor was
      re-derived, so a numerical change inside Transformers 5.5.0 would not be
      caught. Owner checklist §8 carries this.
- [ ] **PR #73 (`dependabot/uv/.../parity-python`) is now `CONFLICTING`.** It
      touches `tools/parity/pyproject.toml` and several `tools/parity/*/uv.lock`,
      which §5 rewrote. Dependabot may rebase it; if not, close it and let a
      fresh scan re-open one against the new floors.
- [ ] **PR #72 is `BEHIND`** — a plain update, no conflict.
- [ ] **Re-check the three blocked trees** when `qwen-asr`, `parler-tts` or
      `xcodec2` publish a release relaxing its pin, and drop the corresponding
      ids from `allow-ghsas`.
- [ ] **Published GGUFs on `huggingface.co/vokra` carry the `0.1.0` stamp.**
      Nothing republishes them automatically, but any re-upload will differ in
      `general.schema_producer` from the artefact recorded in
      `docs/license-audit.md`.
- [ ] **PR #74's body still cites HEAD `4a22d2f4`** and the older log SHA-256.
      Squash used commit messages, not the body, so `main`'s history is accurate;
      only the PR page is stale.
- [ ] **Stale release branches.** `chore/release-v0.1.0` (at `0.1.0`),
      `chore/release-publish-config`, `fix/release-artifact-handoff`,
      `fix/release-crate-count-18` and `fix/release-version-contract` (all at
      `0.1.0-alpha.0`) have no open PR. With `main` now at `0.2.0` they are
      candidates for retirement, matching the 2026-08-18 reduction to
      `origin/main`.
- [ ] **`feat/npu-delegate-execution-2026-08-24`** is merged and can be deleted.

## 8. Corrections made during the session

Recorded because each one was believed and stated before being checked.

1. **CodeQL was called a false positive on the wrong evidence.** The six
   `rust/access-invalid-pointer` alerts in `integrations/vokra-godot` are indeed
   identical on `main` and the PR ref, but they were not what failed the check.
   The two new alerts were `rust/cleartext-logging` in
   `parity_ecapa_tdnn_real.rs:65` and `parity_speechbrain_lang_id_real.rs:200`,
   both files added by the PR. They log float parity diagnostics computed from
   committed fixtures, so they are false positives in substance; both were
   dismissed as such with a written reason (alerts `#65`, `#66`).
2. **The first advisory scan globbed `tools/parity/*/uv.lock`** and so skipped
   the sidecar tree's own `tools/parity/uv.lock`, where `hydra-core 1.3.2`
   survived. CI caught it. The rescan covers all 31 lockfiles.
3. **A license scan over PyPI metadata flagged six GPL/LGPL packages**
   (`soxr`, `frozendict`, `soynlp`, `pycountry`, `num2words`, `phonemizer-fork`).
   GitHub enforces on its own license data, which reports those as undetected —
   a warning, not a gate failure. Only `text-unidecode@1.3` was actually denied.
4. **`1.0.0-rc.1` was recommended for the version bump** on the strength of
   `docs/abi-changelog.md`, before `CHANGELOG.md` showed that `0.1.0` already
   contains the M4 work. The owner's original instinct, `0.2.0`, was correct.

## 9. Standing execution rules that applied

Unchanged from the 2026-08-18 handoff and re-confirmed in practice:

- uv for every Python command; no bare `python`, pip, conda or hand-managed venv.
- No workspace-wide Cargo and no `-p vokra-models` Cargo on the maintainer Mac.
  `-p vokra-core` for the producer-stamp check was inside the permitted
  light-crate set. `cargo metadata` needed `RUSTC_WRAPPER=""` because sccache
  cannot run under the session sandbox.
- `VOKRA_SKIP_HOOKS=1` was used for every push, justified by the remote
  verification in §6, never by haste.
- Both VAST instances destroyed; account verified at zero.
- Nothing was published to Hugging Face and no model artefact was uploaded.
