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

* **Phase B — Numerical parity (Rust-side harness not landed yet).**
  Runs `tools/parity/deberta_v3_dump_reference.py --do-dump` when
  enabled, producing `reference_dump.manifest.json` +
  `reference_dump/<name>.bin` for every per-layer hidden_states +
  attention tensor. Those tensors are uploaded as a job artifact so a
  future harness author can develop against them locally. **The Rust-side
  numerical parity harness that would consume these dumps does not exist
  yet in this repo**: `crates/vokra-bert/tests/deberta_v3_synthetic.rs`
  covers only synthetic weights (`DebertaV3Encoder::synthetic_for_test`),
  and `crates/vokra-convert/tests/deberta_convert.rs`'s smoke is
  structural, not byte-level. The workflow honestly skips the numerical
  leg today with an explicit `::notice::` (fabricated pass 禁止,
  FR-EX-08).

Absent Phase B provisioning, the workflow ONLY exercises Phase A + the
reference-dump artifact production (when force-enabled), plus the
explicit `::notice::` documenting the deferred numerical leg.

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

### Phase B (numerical parity leg — requires landing a harness)

Enabling the numerical parity leg is a code change, not a variable flip.
The steps:

1. **Write the Rust harness.** Recommended location:
   `crates/vokra-bert/tests/deberta_v3_real.rs` (a sibling of
   `deberta_v3_synthetic.rs`; naming matches
   `crates/vokra-models/tests/parity_sbv2_real.rs`). The harness should:

   * Read `VOKRA_DEBERTA_V3_GGUF` (path to the converted GGUF the
     workflow just produced — the workflow already exports this env var
     before running any test step, so no extra plumbing is needed).
   * Read `VOKRA_DEBERTA_V3_REFDIR` (path to `reference_dump/`,
     populated only when the dumper leg ran — gate the harness on this
     env var being both present AND a real directory, per the
     `parity_denoise_dfn3.rs` `env_paths()` idiom, and print a loud
     `skipping: …` if either is missing).
   * Parse `reference_dump.manifest.json`, tokenize the same `--text`
     the manifest records (default `"This is a test."` — see
     `deberta_v3_dump_reference.py`'s `DEFAULT_TEXT`), read
     `input_ids.bin` (`int64`), run
     `DebertaV3Encoder::from_gguf(gguf)?.forward(&ids)`, and compare the
     per-layer `layer_NN_output.bin` + `final_hidden.bin` tensors with
     an `atol` / `rtol` bound set by an architectural-floor calculation
     (see the parity-kokoro-real.yml handover doc for the "atol as
     architectural bound, not CI green" discipline; DeBERTa v3's 24
     layers × 16 heads × 1024 hidden makes a fresh calibration
     necessary — do not copy Kokoro's numbers).
   * Attention tensors (`layer_NN_attention.bin`) can be a follow-up
     if `DebertaV3Encoder::forward` does not currently return them
     (check the API before you write the assertions — do not fabricate
     an assertion path that cannot be reached).

2. **Register the harness in the workflow.** Add a step between "Emit
   Phase B skip notice" and "Run deberta_v3_convert_smoke" that runs the
   new test only when both `needs.setup.outputs.run_dumper == 'true'`
   and the harness landed:

   ```yaml
   - name: Run deberta_v3 numerical parity harness
     if: needs.setup.outputs.run_dumper == 'true'
     env:
       VOKRA_DEBERTA_V3_GGUF: ${{ env.DEBERTA_V3_GGUF }}
       VOKRA_DEBERTA_V3_REFDIR: ${{ env.REF_DIR }}/reference_dump
     run: |
       cargo test --release -p vokra-bert \
         --test deberta_v3_real -- --nocapture
   ```

   Remove or downgrade the "Emit Phase B skip notice" step to an
   informational summary (do not delete it wholesale — the
   `NFR-QL-04 / FR-EX-08` audit trail benefits from an explicit "this
   changed on <date>" boundary).

3. **Land docs updates.** Add a "Phase B active" section to this handoff
   doc noting the harness path, the atol calibration record, and the
   date the flip happened.

Do NOT downgrade the workflow's cron / PR gates when Phase B lands —
the same enable-variable gates continue to keep HF flakiness from
blocking PRs, and the harness itself uses env-var gates on its inputs.

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
