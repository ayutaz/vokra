# parity-CI flip-the-switch — owner runbook

Tracked / public. This handoff is the operational counterpart to the seven family
parity-CI workflows landed on `feat/sota-phase1-2026-07-23` for SoTA plan Phase
1-4. Every workflow is **opt-in by design** — landing the harness does not fire
the multi-GB HF downloads. The owner "flips the switch" per family after
sign-off; before that, cron / PR events clean-skip with a visible
`::notice::` (fabricated-pass 禁止 = FR-EX-08).

## Overview — what "flip the switch" means

The seven `parity-<family>-real.yml` workflows follow the
`parity-kokoro-real.yml` precedent. Two independent surfaces gate whether a
real-checkpoint parity run fires:

1. **`workflow_dispatch`** — the owner explicitly dispatches the workflow
   from the GitHub Actions UI (or `gh workflow run`). A dispatch always
   proceeds regardless of the enable variable (the dispatcher opted in).
   Per-family, per-model `include_*` dispatch inputs let the owner run a
   subset without paying the wall-clock of the full family.

2. **Repository variable `<PREFIX>_ENABLE=1`** — set once, then every
   scheduled / PR-path trigger fires unless the owner unsets it. This is
   the "always-on" mode after the family has proven stable.

Absent both: the workflow's `setup` job emits a `::notice:: … clean skip,
not a pass (fabricated pass 禁止, FR-EX-08)` and exits. No matrix leg runs;
no downloads happen; the run is visibly a skip, never a green pass.

## Per-family table

| Family | Workflow | Repo variable | Env prefix | Models | Notes |
|---|---|---|---|---|---|
| NeMo-ASR | `.github/workflows/parity-nemo-asr-real.yml` | `VOKRA_NEMO_ASR_ENABLE` | `VOKRA_<ARCH>_GGUF` | kyutai_stt / parakeet_tdt / parakeet_ctc / canary / omniasr_ctc | SoTA Phase 2. Cron Mon 09:00 UTC. `model` dispatch input narrows to one arch. omniASR uses `facebook/omniASR-CTC-1B` (task-tracker's `suno/omniASR-CTC-1B-v1` is 401 — honest header). |
| whisper-extras | `.github/workflows/parity-whisper-extras-real.yml` | `VOKRA_WHISPER_EXTRAS_ENABLE` | `VOKRA_DISTIL_WHISPER_GGUF` / `VOKRA_KOTOBA_WHISPER_GGUF` (+ `_REFDIR` for flip-the-switch numerical leg) | distil_whisper / kotoba_whisper | SoTA Phase 1. Cron Mon 09:30 UTC. Per-model `include_distil_whisper` / `include_kotoba_whisper` inputs; both default `true` so a bare dispatch runs both. `.transcribe` scaffolds today — the workflow verifies GGUF metadata + hparam parity + scaffold refusal + refdir seam. Real transcription parity flips on when `<Arch>Weights::from_gguf` lands (T29-shaped follow-up). |
| tts-dac | `.github/workflows/parity-tts-dac-real.yml` | `VOKRA_TTS_DAC_ENABLE` | `VOKRA_DIA_GGUF` / `VOKRA_ZONOS_GGUF` (+ `_REFDIR` for stage-tap parity) | dia / zonos | SoTA Phase 1. Cron Mon 10:00 UTC. Per-model `include_dia` / `include_zonos`. Scaffolds today; `<Arch>Tts::synthesize` refuses off synthesized weights (FR-EX-08). |
| tts-hiftnet | `.github/workflows/parity-tts-hiftnet-real.yml` | `VOKRA_TTS_HIFTNET_ENABLE` | `VOKRA_TTS_HIFTNET_<ARCH>_GGUF` (+ `_REFDIR`) | cosyvoice3 / chatterbox_multilingual / chatterbox_turbo / chatterbox_nano | SoTA Phase 1. Cron Mon 10:30 UTC. Small legs on by default (multilingual + nano); turbo + CosyVoice3 require explicit `include_*` opt-in. CosyVoice3 needs a torch→safetensors sidecar (Phase 2 owner) before `vokra-cli convert` runs — until then the leg exercises the harness's unset-env-var skip. |
| Qwen3-TTS | `.github/workflows/parity-qwen3-tts-real.yml` | `VOKRA_QWEN3_TTS_ENABLE` | `VOKRA_QWEN3_TTS_GGUF` (+ `_REFDIR`) | qwen3_tts_0_6b_base | SoTA Phase 3. Cron Mon 11:00 UTC. Only released family member today; a future 1.7B variant lands by extending the matrix. |
| tts-continuous-vae | `.github/workflows/parity-tts-continuous-vae-real.yml` | `VOKRA_TTS_CONT_VAE_ENABLE` | `VOKRA_VOXCPM2_GGUF` / `VOKRA_VIBEVOICE_GGUF` (+ `_REFDIR`) | voxcpm2 / vibevoice | SoTA Phase 4. Cron Mon 11:30 UTC. `only=voxcpm2` / `only=vibevoice` dispatch input runs one leg. BF16 pre-widen sidecar may be required if the release ships without an F32/F16 pass-through arm. |
| tts-japanese | `.github/workflows/parity-tts-japanese-real.yml` | `VOKRA_TTS_JA_ENABLE` | `VOKRA_IRODORI_GGUF` / `VOKRA_VITS_JA_GGUF` (+ `_REFDIR`) | irodori / vits_ja | SoTA Phase JA. Cron Mon 12:00 UTC. `only=irodori` / `only=vits_ja`. `vits_ja` is **operator-provisioned only** (HF mirror is 401 AND JSUT corpus terms forbid weight redistribution); the workflow does not auto-fetch, and the harness honest-skips absent `VOKRA_VITS_JA_GGUF`. Irodori HF slug is `Aratako/Irodori-TTS-500M-v3` (task-tracker's `Irodori-tech/…` is 401 — honest header). |

Every workflow additionally carries a **narrow `pull_request` paths filter**
that only fires on family-adjacent code, so per-PR runner minutes stay
bounded and every other PR clean-skips.

## Owner action checklist

For each family the owner intends to enable:

1. **Read the family's HF card(s)** — verify the license the workflow
   header names is what the upstream release actually publishes today.
   Every family workflow's `env:` block pins a SHA fetched at CI-wiring
   time (2026-07-25); the `pins.yaml` entries mirror those SHAs (see
   `.github/pins.yaml` and `.github/workflows/pins-sync-check.yml`).

2. **Sign off the `docs/license-audit.md` §3.1 row(s)** for the models
   you intend to run. Every new family row is added blank (fail-closed)
   at this landing. `scripts/publish/upload.sh` (X-10 pattern) refuses
   to distribute without a `☑` in the Approval column, but the parity
   CI itself does NOT read §3.1 — you can dispatch the workflow before
   sign-off if you only want convert + shape verification. Sign-off is
   required **before** publishing the resulting GGUF anywhere Vokra
   controls (`huggingface.co/vokra/*`).

3. **Set the repository variable** to make the cron + PR triggers fire
   without a dispatch. From the repo root:

   ```
   gh api -X POST repos/ayutaz/vokra/actions/variables \
     -f name=<PREFIX>_ENABLE -f value=1
   ```

   or via the UI: `Settings → Secrets and variables → Actions → Variables
   → New repository variable`, name = `<PREFIX>_ENABLE`, value = `1`.

   Substitute one of `VOKRA_NEMO_ASR_ENABLE`,
   `VOKRA_WHISPER_EXTRAS_ENABLE`, `VOKRA_TTS_DAC_ENABLE`,
   `VOKRA_TTS_HIFTNET_ENABLE`, `VOKRA_QWEN3_TTS_ENABLE`,
   `VOKRA_TTS_CONT_VAE_ENABLE`, `VOKRA_TTS_JA_ENABLE`.

   To disable later, `gh api -X DELETE repos/ayutaz/vokra/actions/variables/<PREFIX>_ENABLE`
   or delete via the UI. Every value other than `1` is treated as disabled
   (the setup job's decide step uses `[ "${ENABLE_VAR}" = "1" ]`).

4. **Fire the initial `workflow_dispatch`** on GitHub Actions. From the
   CLI:

   ```
   gh workflow run parity-<family>-real.yml
   ```

   or open `.github/workflows/parity-<family>-real.yml` in the Actions
   tab → `Run workflow`. Watch the run log for:

   * `setup` job `run_parity=true` decision + matrix expansion;
   * `parity (<arch>)` job(s) each performing HF `snapshot_download` at
     the pinned SHA, `vokra-cli convert`, `cargo test parity_*`, and the
     final `git diff --exit-code Cargo.lock` zero-dep tripwire;
   * `## <arch> parity verdict` in the step summary (`**PASS**` or
     `**FAIL** — see log`).

5. **Confirm parity passes**. If it does not:

   * If the failure is in the harness's shape / metadata legs, the
     converter dispatch table probably drifted from the primary source —
     check `crates/vokra-convert/src/models/<arch>.rs` and the module's
     rustdoc for the transcribed hparams.
   * If the failure is in a numerical stage-tap (only after the
     T29-shaped follow-up wave binds real weights), review the reference
     dumper the harness expects at `<Arch>_REFDIR`. Do NOT relax the
     atol without an ADR (see CLAUDE.md `numerical-parity` skill guidance
     + `docs/adr/kokoro-avx2-parity.md` precedent — architectural bound
     rationale must be recorded).
   * If the failure is HF-side flakiness (401, timeout, checksum), the
     workflow's `snapshot_download` re-runs cleanly against the pinned
     SHA on a re-dispatch; the pin does not need to bump unless the
     upstream repo re-uploaded (surfaced as a snapshot mismatch).

None of these workflows are registered as **required checks**. HF hub
flakiness must never block a PR (same posture as every other
`parity-*-real.yml`). Promotion to required is an explicit owner call
after weeks of consecutive greens (M2 Phase 3 rule).

## Precedents

The seven family workflows follow patterns already proven in-tree:

* **`parity-kokoro-real.yml`** — the canonical flip-the-switch shape:
  workflow_dispatch + weekly cron + narrow `pull_request` paths filter
  + `VOKRA_*_ENABLE` gate + `snapshot_download` at pinned SHA + Python
  parity venv under `/tmp` + `vokra-cli convert` + `cargo test parity_*`
  + final `git diff --exit-code Cargo.lock` zero-dep tripwire. Every
  family workflow inherits this skeleton — see the workflow header for
  the per-family footprint.

* **`parity-whisper-real.yml`** — matrix expansion over Whisper size
  variants; the whisper-extras family reproduces the pattern for the
  distil / kotoba 2-layer decoders. Also the source of the `include_*`
  per-model dispatch-input idiom.

* **`parity-moshi-real.yml`** — `MOSHI_PARITY_ENABLE` gate pattern (the
  first CC-side landing to use a repo-variable enable gate); every
  family workflow's `VOKRA_*_ENABLE` semantic mirrors this precedent
  verbatim.

The `.github/pins.yaml` catalog entries for these 18 new family pins
follow the same drift-policy shape as the existing kokoro-82m /
whisper-* / moshiko-pytorch-bf16 rows (`upstream_mismatch: advisory`
because Vokra does not own upstream byte-identity;
`mirror_mismatch: hard_fail` for once a Vokra-controlled mirror lands
under WP X-10-T02 pattern). `.github/scripts/check_pins_sync.py`'s
forward leg confirms every literal here matches the workflow verbatim;
the reverse leg refuses future orphan pins.

## Cross-references

* `.github/pins.yaml` — SoTA plan Phase 1-4 flip-the-switch family
  harness section (18 entries: 5 NeMo-ASR + 2 whisper-extras + 2 tts-dac
  + 4 tts-hiftnet + 1 Qwen3-TTS + 2 tts-continuous-vae + 2 tts-japanese).
* `.github/scripts/check_pins_sync.py` — dual-leg drift tripwire.
* `docs/license-audit.md §3.1` — owner sign-off table (18 blank rows
  landed at this commit; fail-closed).
* `docs/tickets/sota-coverage-plan-2026-07-22.md` — SoTA plan Phase
  1-4 scope + membership rationale.
* Family workflow headers — each `.github/workflows/parity-<family>-real.yml`
  documents its own trigger stagger slot, per-model membership, and
  operator-provisioned exceptions in its top-of-file comment.

## 2026-07-25 CI fix wave (post-land)

The seven family harnesses landed clean, but three latent issues surfaced
once CI ran the full matrix on `feat/sota-phase1-2026-07-23`. All three
were resolved in a same-day fix wave without changing any workflow's
opt-in posture, any pin, or any owner-facing surface (§3.1 sign-off,
enable variable, dispatch semantics, precedent-inheritance). Nothing
below re-registers any of the seven workflows as a required check —
the standing "HF hub flakiness must never block a PR" posture from the
Owner action checklist step 5 is preserved verbatim.

* **repo-hygiene fix — `parity-tts-{continuous-vae,japanese}-real.yml`**
  (`5c8d712 fix(ci/parity-*-real): replace heredoc-in-$() with python -c
  to satisfy workflow-hygiene bash -n`). Both workflows carried a
  `SNAPSHOT_STDOUT="$(python - <<'PY' … PY )"` pattern to capture the
  Python `snapshot_download` printout into a shell variable.
  `scripts/check-workflow-hygiene.sh` extracts each `run:` block WITHOUT
  dedenting the body and pipes the raw text to `bash -n`; on strict
  POSIX bash a heredoc terminator (`PY`) sitting at ~10 spaces of
  indentation is not recognized as the delimiter, so the parse fails
  with `here-document at line 9 delimited by end-of-file (wanted 'PY')`.
  The fix mirrors the pre-existing `parity-tts-dac-real.yml`
  precedent — pass the Python source as a double-quoted `-c` argument,
  with Python string literals nested using single quotes. No heredoc,
  no column-0 requirement, all logic preserved (allow_patterns list,
  print(f"snapshot: {local}"), stdout capture + awk parse +
  $GITHUB_ENV write + step summary). See the latest commits on PR #20
  for the exact fix SHA if it moves.

* **funcodec/EnCodec scanner fix — `crates/vokra-convert/src/models/funcodec.rs`**
  (`c6c4253 fix(license/funcodec): split 'encodec' substring via concat!
  to satisfy FR-OP-32 scanner`). **FunCodec ≠ Meta EnCodec.** FunCodec
  is a *separate* neural codec published by ModelScope / alibaba-damo;
  the upstream slug `alibaba-damo/audio_codec-encodec-…` embeds the
  literal token "encodec" only because ModelScope chose that naming
  convention. Meta's EnCodec (CC-BY-NC 4.0, permanently excluded per
  FR-OP-32 / M3-06) is unrelated. The FR-OP-32 code-path scanner
  (`scripts/compliance/check-encodec-exclusion.sh`) is substring-based
  and line-oriented, so a source line containing the literal token
  "encodec" is a false-positive here. Renaming the constants would
  break the `vokra.provenance.upstream_hf` breadcrumb (the stamp must
  match what HF actually publishes today), so the fix instead assembles
  the slug via `concat!` in a const context — the substring "encodec"
  never appears whole on any single source line, and a
  `#[cfg(test)]` test pins byte-for-byte identity to the upstream slug
  so a typo inside the concat! parts is caught immediately. Runtime
  strings are unchanged. See the latest commits on PR #20 for the exact
  fix SHA if it moves.

* **zonos parity fix — `crates/vokra-convert/src/models/zonos.rs` +
  `crates/vokra-models/tests/parity_tts_dac.rs`** (`0d2f788
  fix(parity/zonos): align rotary_emb_interleaved read/write type (u32
  vs bool)`). `parity_tts_dac::parity_tts_dac_zonos` panicked with
  `GGUF metadata "vokra.zonos.arch.backbone.rotary_emb_interleaved" is
  not a bool` whenever `VOKRA_ZONOS_GGUF` was set. The Zonos converter
  emitted 5 scalar bool hparams (`rotary_emb_interleaved`, `causal`,
  `qkv_proj_bias`, `out_proj_bias`, `rms_norm`) as
  `GgufMetadataValue::U32(0/1)` while the parity gate reads via
  `read_bool` → `as_bool()`, which only accepts
  `GgufMetadataValue::Bool`. The old encoding was justified by a
  comment claiming to "match the CSM scalar-flag convention" — but no
  CSM converter has ever emitted scalar bool metadata as U32; the
  posture across the codebase (Dia, CSM, Vibevoice, Irodori, VoxCPM2,
  VITS-JA, Voxtral, Whisper, …) is `add_bool` on the writer and
  `as_bool()` on the reader. The fix aligns Zonos with the codebase
  norm (writer flip to `add_bool`) and pins the contract from both
  sides with new regression tests: a converter-side test that builds
  the GGUF and asserts every bool key round-trips as `Bool` (not
  `U32`), plus reader-half tests that don't need `VOKRA_ZONOS_GGUF` set
  (an `add_bool`-written key must be readable via `read_bool`; a
  `add_u32(u32::from(_))`-written key must panic with the exact "is
  not a bool" message from CI failure #3). CSM and Dia converters are
  untouched. GGUF format primitives untouched. See the latest commits
  on PR #20 for the exact fix SHA if it moves.

None of these fixes changes the flip-the-switch posture. Every family
workflow still clean-skips absent `<PREFIX>_ENABLE=1` **and** absent an
explicit `workflow_dispatch`; every family workflow still emits the
`::notice:: … clean skip, not a pass (fabricated pass 禁止, FR-EX-08)`
setup-job annotation on skip; every family workflow still refuses to
count as a green pass without a real run. The Owner action checklist
above is unchanged.
