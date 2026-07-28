# parity-deepfilternet3-real — owner runbook

Tracked / public. Operational counterpart to
`.github/workflows/parity-deepfilternet3-real.yml`, landed 2026-07-28 as
the CI leg for the M4-20 T17 DeepFilterNet3 `denoise` parity harness
(`crates/vokra-ops/tests/parity_denoise_dfn3.rs`).

## Overview

The workflow is **two-phase**:

* **Phase A — Conversion (reproducible on any hosted-runner).** Fetches
  the pinned `Rikorose/DeepFilterNet` GitHub commit's
  `models/DeepFilterNet3.zip` (sha256 `49c52edc…`, the same prefix the
  harness docstring cites), flattens the `.ckpt.best` via
  `tools/parity/dfn3_prepare_checkpoint.py`, and runs
  `vokra-cli convert --model denoise`. The converter's own self-check re-
  parses the emitted GGUF via `DenoiseModel::from_gguf`, so a successful
  convert already asserts full round-trip loadability.

* **Phase B — Byte-level parity (owner-provisioned reference bundle).**
  Runs `parity_denoise_dfn3::dfn3_real_weight_stage_and_output_parity`
  against the reference dumps
  (`clean_48k.f32` / `noisy_48k.f32` / `enhanced_upstream.f32` + `taps/`)
  the harness demands. The bundle is fetched from
  `vars.VOKRA_DFN3_DATA_URL` — currently unset because the exact
  `prep_noisy.py` recipe (specific 11 s clean source + noise seed + mixing
  formula that landed the harness's tight `snr_noisy = 5.002 ± 0.01` and
  `snr_up = 14.768 ± 0.01` bounds) lives outside the repo (M4-20
  owner-local `~/.cache/vokra-eval/out/dfn3-real/`).

Absent Phase B provisioning, the workflow ONLY exercises Phase A and
emits a `::notice::` documenting the deferred parity leg — **fabricated
pass 禁止** (FR-EX-08).

## Owner action checklist

### Phase A only (conversion + GGUF sanity)

1. Set the enable variable so cron + PR triggers exercise the conversion
   path:

   ```
   gh api -X POST repos/ayutaz/vokra/actions/variables \
     -f name=VOKRA_DFN3_ENABLE -f value=1
   ```

   Or via the UI: `Settings → Secrets and variables → Actions →
   Variables → New repository variable`, name `VOKRA_DFN3_ENABLE`,
   value `1`. Every value other than `1` is treated as disabled
   (the setup job's decide step uses `[ "${ENABLE_VAR}" = "1" ]`).

2. Fire the initial dispatch:

   ```
   gh workflow run parity-deepfilternet3-real.yml
   ```

   or open the workflow in the Actions tab → `Run workflow`.

3. In the run log, verify:
   * `setup` job → `run_conversion=true`, `run_parity=false`, and the
     Phase B `::notice::` is present in the summary.
   * `parity (dfn3)` job → the sha256 verification line
     (`sha256 OK: 49c52edc…`), the safetensors size table, the
     conversion "OK" table, and the "SKIPPED (Phase B not enabled)"
     summary block.
   * Final `git diff --exit-code Cargo.lock` step exits clean — zero-dep
     NFR-DS-02 held.

### Phase B (byte-level parity — requires reference bundle)

Enabling Phase B needs a reproducible `prep_noisy.py` (or an equivalent
pre-baked `.tar.gz` bundle) whose output bit-exactly matches the
`snr_noisy = 5.002 ± 0.01` bound in the harness. Two paths:

**Path 1 — commit `prep_noisy.py` alongside the existing dumpers.**
Recommended: land a `tools/parity/dfn3_prep_noisy.py` next to
`dfn3_prepare_checkpoint.py` / `dfn3_dump_reference.py`, mirroring the
2026-07-17 M4-20 recipe (documented in
`docs/bench-baselines/m1-real-weight-eval-2026-07-16/agent-results-campaign2.json`
under the `dfn3-real` leg). The workflow then invokes:

```
. "${PARITY_VENV}/bin/activate"
python -m pip install 'soundfile>=0.12' 'deepfilternet==0.5.6'
python tools/parity/dfn3_prep_noisy.py \
  --clean-source tests/fixtures/audio/jfk-30s.wav \
  --out "${RUNNER_TEMP}/dfn3-refdata"
python tools/parity/dfn3_dump_reference.py \
  --model-dir "${DFN3_MODEL_DIR}" \
  --noisy "${RUNNER_TEMP}/dfn3-refdata/noisy_48k.wav" \
  --out "${RUNNER_TEMP}/dfn3-refdata/taps"
cp "${RUNNER_TEMP}/dfn3-refdata/taps/enhanced.f32" \
  "${RUNNER_TEMP}/dfn3-refdata/enhanced_upstream.f32"
```

The above snippet needs to land as an additional step in the workflow
between "Convert" and "Run parity_denoise_dfn3 harness". Once
`prep_noisy.py` is committed, edit `parity-deepfilternet3-real.yml` to
call it inline (removing the `VOKRA_DFN3_DATA_URL` gate) and the parity
leg becomes fully reproducible on every dispatch.

**Path 2 — host a pre-baked `.tar.gz` at a stable URL.** If committing
`prep_noisy.py` is deferred, the owner can bake the reference bundle
locally on the M4-20 machine and host it at any HTTPS URL Vokra
controls (`huggingface.co/vokra/dfn3-parity-refdata`, a Vokra release
asset, etc.). The bundle layout the workflow expects:

```
dfn3-refdata.tar.gz →
  clean_48k.f32
  noisy_48k.f32
  enhanced_upstream.f32
  taps/
    spec.f32
    feat_erb.f32
    feat_spec.f32
    e0.f32 e1.f32 e2.f32 e3.f32
    c0.f32
    cemb.f32
    emb_in.f32 emb.f32
    lsnr.f32
    m.f32
    df_gru_out.f32
    coefs.f32
    spec_e.f32
    enhanced.f32
```

Then set:

```
gh api -X POST repos/ayutaz/vokra/actions/variables \
  -f name=VOKRA_DFN3_DATA_URL -f value=https://example.org/.../dfn3-refdata.tar.gz
# Optional: pin the bundle sha256 for tamper detection:
gh api -X POST repos/ayutaz/vokra/actions/variables \
  -f name=VOKRA_DFN3_DATA_SHA256 -f value=<sha256>
```

Subsequent cron + PR + dispatch runs fetch the bundle, verify sha256
(if set), and run the full `parity_denoise_dfn3` per-stage bound
assertion.

## Troubleshooting

* **`sha256 mismatch` on the DFN3 zip.** The `82b0c7ad…` commit has been
  static since 2023-05-23; a mismatch indicates either a Rikorose
  upstream re-upload (extremely unlikely; would invalidate the harness
  bounds too) or transfer corruption. Re-run the workflow; if the
  mismatch persists, escalate — do not bump the pin without also
  re-baking the reference bundle.

* **`parity_denoise_dfn3 skipped despite both env vars being set`.** The
  workflow has an explicit guard for this: if the harness prints
  `skipping: set VOKRA_DFN3_GGUF …` while both env vars are set, the
  step flips PARITY_EXIT=1. Usually indicates the env var was set to an
  unresolved path.

* **`git diff --exit-code Cargo.lock` fails.** The parity venv's `pip
  install torch` must not touch the root Cargo.lock. If this ever fires,
  something in the workflow moved to `cargo install`; revert and
  investigate.

## Non-goals

* **Not** a required check. HF flakiness / GitHub raw CDN outages
  must not block PRs (same posture as every other parity-*-real
  workflow on this branch). Promotion to required is an explicit owner
  decision after weeks of consecutive greens (`docs/handoff/parity-ci-
  flip-switch.md` §Promotion criteria).

* **Not** a load bearer for the M4-20 T17 GA judgment. The M1
  owner-local run is the authoritative parity record; this CI leg
  guards against future drift once Phase B is provisioned.

## Related

* Harness: `crates/vokra-ops/tests/parity_denoise_dfn3.rs`
* Prep tool: `tools/parity/dfn3_prepare_checkpoint.py`
* Reference dumper: `tools/parity/dfn3_dump_reference.py`
* Primitives fixture: `tools/parity/dfn3_primitives_fixture.py` (already
  committed under `tests/parity/dfn3/`, exercised by the per-PR fixture
  parity in `ci.yml`)
* Local M4-20 recipe: `docs/bench-baselines/m1-real-weight-eval-2026-07-16/`
* Flip-switch overview: `docs/handoff/parity-ci-flip-switch.md`
