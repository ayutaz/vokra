# parity-deberta-v3-large-real — owner runbook

Tracked / public. Operational counterpart to
`.github/workflows/parity-deberta-v3-large-real.yml`, landed 2026-07-28 as
the standalone CI leg for the DeBERTa v3 EN BERT encoder path
(`microsoft/deberta-v3-large`, MIT). Sibling of
`parity-sbv2-real.yml` (which exercises DeBERTa v3 as part of the full
SBV2 multi-checkpoint pipeline) — this workflow isolates the EN BERT
convert / round-trip / smoke leg so a DeBERTa v3 regression is caught
independently of the SBV2 base (AGPL-3.0) and JA BERT (CC-BY-SA-4.0)
checkpoint gates.

## Overview

The workflow is **two-phase**:

* **Phase A — Conversion + smoke (reproducible on any hosted-runner).**
  `snapshot_download`s `microsoft/deberta-v3-large` at the pinned
  revision (SHA `64a8c8eab3e352a784c658aef62be1662607476f`, verified live
  via the HF Hub API primary source 2026-07-28). The upstream release
  ships `pytorch_model.bin` only — no `model.safetensors` mirror —
  so `tools/parity/bin_to_safetensors.py` (owner Task 3 Wave 2b) bridges
  the pickle → safetensors conversion inside the parity venv, then
  `vokra-cli convert --model deberta-v3` produces the GGUF. The
  converter's own `DebertaV3Encoder::from_gguf` self-check re-parses the
  emitted bytes as a built-in round-trip assertion, and the
  `deberta_v3_convert_smoke` `#[ignore]`d test in
  `crates/vokra-convert/tests/deberta_convert.rs` (structural: `written >
  0`, `read == written + skipped_non_float`) is additionally run against
  the freshly-produced safetensors via a scoped fixture symlink.

* **Phase B — Numerical parity (`final_hidden`) — LANDED 2026-07-29.**
  Runs `tools/parity/deberta_v3_dump_reference.py --do-dump` when
  enabled, producing `reference_dump.manifest.json` +
  `reference_dump/<name>.bin` for every per-layer hidden_states +
  attention tensor. Then a Rust-side harness at
  `crates/vokra-bert/tests/deberta_v3_real.rs`
  (`deberta_v3_real_weight_final_hidden_parity`) consumes
  `input_ids.bin` + `final_hidden.bin`, runs
  `DebertaV3Encoder::from_gguf(...)?.forward(&ids)`, and asserts an
  architectural-bound atol (see the harness module doc for the
  1024 * f32::EPSILON × 24 layers × 2x hdrm ≈ 6.0e-3 derivation).
  Per-layer `layer_NN_output` + `layer_NN_attention` legs remain deferred
  because the Rust encoder does not expose per-layer taps today —
  enabling those requires an additive `forward_with_layer_taps()` on
  `DebertaV3Encoder`, tracked separately.

The workflow's "Phase B status" step emits an explicit per-run summary
(what ran vs why anything skipped), preserving the `NFR-QL-04 / FR-EX-08`
audit trail every run.

## Owner action checklist

### Phase A only (conversion + GGUF sanity + smoke)

1. Set the enable variable so cron + PR triggers exercise the conversion
   path:

   ```
   gh api -X POST repos/ayutaz/vokra/actions/variables \
     -f name=VOKRA_DEBERTA_V3_ENABLE -f value=1
   ```

   Or via the UI: `Settings → Secrets and variables → Actions →
   Variables → New repository variable`, name
   `VOKRA_DEBERTA_V3_ENABLE`, value `1`. Every value other than `1` is
   treated as disabled (the setup job's decide step uses
   `[ "${ENABLE_VAR}" = "1" ]`).

2. Fire the initial dispatch:

   ```
   gh workflow run parity-deberta-v3-large-real.yml
   ```

   or open the workflow in the Actions tab → `Run workflow`.

3. In the run log, verify:
   * `setup` job → `run_conversion=true`, `run_dumper=false`, and the
     Phase B `::notice::` is present in the summary.
   * `parity (deberta-v3-large)` job → the pinned SHA line
     (`64a8c8eab3e352a784c658aef62be1662607476f`), the safetensors size
     table, the conversion "OK" table, the "SKIPPED (harness not
     landed)" summary block, and the `deberta_v3_convert_smoke` test
     result.
   * Final `git diff --exit-code Cargo.lock` step exits clean — zero-dep
     NFR-DS-02 held.

### Phase A + reference dump (feeds the future Rust harness)

Once you want the reference tensors to be produced and uploaded as an
artifact (so a harness author can start work locally), either:

* **Manual dispatch with the input flag**:

  ```
  gh workflow run parity-deberta-v3-large-real.yml -f run_dumper=true
  ```

* **Or set the cron/PR flag** so it fires automatically on the weekly
  cron and on adjacent-path PRs:

  ```
  gh api -X POST repos/ayutaz/vokra/actions/variables \
    -f name=VOKRA_DEBERTA_V3_HARNESS_READY -f value=1
  ```

The dumper's output ships as the `deberta-v3-large-parity-artifacts`
artifact bundle (`reference_dump.manifest.json` +
`reference_dump/<name>.bin`).

### Phase B active (final_hidden — 2026-07-29)

**Status**: `final_hidden` parity leg LANDED 2026-07-29.

* **Harness path**: `crates/vokra-bert/tests/deberta_v3_real.rs`
  (`deberta_v3_real_weight_final_hidden_parity`). Reads
  `VOKRA_DEBERTA_V3_GGUF` + `VOKRA_DEBERTA_V3_REFDIR` (env-var gates,
  honest-skip if unset or not resolvable to a file/dir — mirrors
  `parity_denoise_dfn3.rs`'s `env_paths()` idiom, plus a second
  is_file / is_dir check so a broken workflow step surfaces loudly
  rather than silently degrading to a green skip).
* **Atol calibration record** (module doc): `6.0e-3`, derived from
  `1024 * f32::EPSILON × 24 layers × 2x cross-machine libm headroom`.
  Not a CI-green-seeking constant. Update this only alongside a fresh
  measured run — memory `feedback-honest-parity-atol`.
* **Workflow step**: "Run deberta_v3 numerical parity harness
  (final_hidden)", gated on `needs.setup.outputs.run_dumper == 'true'`.
  Exports `VOKRA_DEBERTA_V3_GGUF` + `VOKRA_DEBERTA_V3_REFDIR` from the
  already-emitted `env.DEBERTA_V3_GGUF` + `env.REF_DIR` — no additional
  plumbing needed.
* **Skipped legs (deferred, honest)**: `layer_NN_output` +
  `layer_NN_attention` per-layer tensors. `DebertaV3Encoder::forward`
  returns final hidden only; enabling per-layer comparison requires an
  additive `forward_with_layer_taps()` on the encoder (mirrors the
  `parity_denoise_dfn3.rs` `enhance_with_taps()` refactor). Not
  fabricated as a reachable assertion path — see the harness module
  doc's Coverage note.
* **Cron / PR gate posture unchanged**: the two enable variables
  (`VOKRA_DEBERTA_V3_ENABLE` for conversion; `VOKRA_DEBERTA_V3_HARNESS_READY`
  for cron/PR dumper trigger) continue to keep HF flakiness from
  blocking PRs. The harness itself uses env-var gates on its inputs as
  a second line of defence.

### Future — Phase B extension (per-layer taps)

Follow-up to close the deferred legs above. Not currently prioritized;
land only when a genuine drift-in-a-layer regression is suspected. The
work would:

1. Add `pub fn forward_with_layer_taps(&self, ids: &[u32]) ->
   (Vec<f32>, Vec<LayerTap>)` to `DebertaV3Encoder`, where each
   `LayerTap` holds the layer's post-residual hidden and its
   post-softmax attention tensor.
2. Extend the harness to iterate `manifest.tensors[]`, read every
   `layer_NN_output.bin` + `layer_NN_attention.bin`, and assert per-tap
   atols (each family gets its own architectural bound — attention
   noise is different from residual noise; do not reuse the
   `final_hidden` constant).
3. Update the workflow's "Phase B status" summary to record per-layer
   coverage on future runs.

No workflow-side changes needed for step 3 beyond the summary text
edit — the existing "Run deberta_v3 numerical parity harness" step will
run whatever test names the harness gains.

## Troubleshooting

* **`snapshot_download` HTTP 401 / 403.** DeBERTa v3 is publicly gated
  behind acceptance of Microsoft's model card terms (MIT license text
  is unambiguous — the "gate" is HuggingFace's own workflow, not a legal
  restriction). If CI hits 401, the runner's HF cache does not have an
  accepted token; set `HF_TOKEN` as a repo secret (read-only, model-
  card-accepted) and add it to the "Download microsoft/deberta-v3-large
  @ pinned SHA (HF hub)" step's `env:` block.

* **`bin_to_safetensors.py` fails with an "empty state_dict" error.**
  `pytorch_model.bin` should hold ~400 tensors for the large variant; an
  empty dict indicates a corrupt download. Re-run the workflow; the HF
  cache key includes the pinned revision so a fresh fetch will land.

* **`vokra-cli convert --model deberta-v3` fails with "no token-
  embedding-shaped tensor found".** The DeBERTa v3 converter refuses
  to invent a `vocab_size` when the checkpoint does not expose one via
  the safetensors metadata / adjacent `config.json`. Verify that the
  fetch step also downloaded `config.json` (the workflow's
  `allow_patterns` list includes it explicitly).

* **`deberta_v3_convert_smoke` fails.** The smoke asserts
  `report.written > 0` and `report.read == report.written +
  report.skipped_non_float`. A `written == 0` outcome means the
  converter walked every tensor as "skipped_non_float", which for a
  BF16 checkpoint indicates a converter regression in the F32/F16/BF16
  pass-through arm — check
  `crates/vokra-convert/src/models/deberta_v3.rs`'s type-dispatch
  match.

* **`git diff --exit-code Cargo.lock` fails.** The parity venv's `pip
  install torch transformers safetensors` must not touch the root
  Cargo.lock. If this ever fires, something in the workflow moved to
  `cargo install`; revert and investigate.

## Non-goals

* **Not** a required check. HF flakiness / GitHub raw CDN outages must
  not block PRs (same posture as every other parity-*-real workflow on
  this branch). Promotion to required is an explicit owner decision
  after weeks of consecutive greens
  (`docs/handoff/parity-ci-flip-switch.md` §Promotion criteria).

* **Not** a load bearer for the SBV2 v2 pipeline GA judgment. The
  `parity-sbv2-real.yml` end-to-end run is the authoritative record for
  the full SBV2 v2 3-checkpoint parity; this CI leg guards against
  future drift on the EN BERT encoder in isolation.

* **Not** overlapping with `parity-sbv2-real.yml`. Both workflows fetch
  `microsoft/deberta-v3-large` and convert it, but SBV2 uses the
  `--allow_patterns=['*.safetensors', '*.json']` shortcut (which
  currently fails against this repo's `.bin`-only distribution — a
  known open gap in that workflow) whereas this workflow bridges via
  `bin_to_safetensors.py`. The two workflows can run on the same cron
  day without contention because they target different cron minutes
  (SBV2 = Monday 07:15 UTC, this = Monday 13:00 UTC).

## Related

* Reference dumper: `tools/parity/deberta_v3_dump_reference.py`
  (Task 31, clean-room per HF transformers Apache-2.0 +
  arXiv:2111.09543 — see the module docstring's "NOT REFERENCED" list).
* Bin→safetensors bridge: `tools/parity/bin_to_safetensors.py` (owner
  Task 3 Wave 2b).
* Rust converter: `crates/vokra-convert/src/models/deberta_v3.rs`
  (Task 12).
* Rust encoder: `crates/vokra-bert/src/deberta_v3.rs`.
* Converter smoke: `crates/vokra-convert/tests/deberta_convert.rs`
  (Task 11 — `deberta_v3_convert_smoke` `#[ignore]`d).
* SBV2 integrated CI (contains DeBERTa v3 in the pipeline):
  `.github/workflows/parity-sbv2-real.yml`.
* License audit sign-off: `docs/license-audit.md` §3.1 row
  `deberta-v3-large` (2026-07-27 yousan, ☑ Commercial, MIT primary
  source `github.com/microsoft/DeBERTa/master/LICENSE`).
* Pin catalog: `.github/pins.yaml` entry `deberta-v3-large`.
* Flip-switch overview: `docs/handoff/parity-ci-flip-switch.md`.
