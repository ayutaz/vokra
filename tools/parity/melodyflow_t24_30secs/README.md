# tools/parity/melodyflow_t24_30secs

Offline sidecar for **facebook/melodyflow-t24-30secs** (Meta AudioCraft
MelodyFlow T24 30secs, CC-BY-NC-4.0, ~4.0 GB bundle = 1 B flow-matching
DiT transformer + 48 kHz RVQ codec + T5-base text encoder) — bridges the
upstream torch-pickle bundle to the flat safetensors the Rust converter
(`crates/vokra-convert/src/models/melodyflow_t24_30secs.rs`) consumes.

Sibling of:

- `../magnet_small_10secs/prepare_checkpoint.py` (~2 GB, 500 M / 10 sec
  — non-autoregressive masked-LM decoding, entirely different sampler)
- `../magnet_medium_30secs/prepare_checkpoint.py` (~5.7 GB, 1.5B / 30 sec
  — non-autoregressive masked-LM decoding, same-scale sibling in the
  Meta music-gen catalog)
- `../jasco_400m_chords_drums/` (~1.6 GB, joint audio-symbolic
  conditioning — same op family (flow-matching) but different
  conditioning stack from MelodyFlow's dual text + audio prefix for
  editing)
- `../musicgen_medium_prepare_checkpoint.py` (11.4 GB, vast.ai required
  — AR-over-EnCodec sibling family, entirely different decoder loop
  from MelodyFlow flow-matching)

## What this directory contains

- `prepare_checkpoint.py` — the actual torch-pickle → flat safetensors
  bridge. Loads `state_dict.bin` (or `*.th` / `*.bin`) under
  `--input-dir` via `torch.load(..., weights_only=True)` (safe path),
  dedupes tied tensors (data_ptr collision → clone + audit trail),
  strips non-float training scaffold (`.num_batches_tracked` /
  `.total_ops` / `.total_params`), rejects unexpected non-float
  dtypes loudly (FR-EX-08). Emits `<output>` + `<output>.sha256` +
  `<output>.shared_pairs.json`. See the script's module docstring
  for the honest write-up.
- `pyproject.toml` — uv project spec (Python 3.12 pinned per
  `[[feedback-python-3-12]]`; deps = `torch>=2.0` + `safetensors>=0.4`
  + `huggingface-hub>=0.26`).
- `.python-version` — `3.12`.

## Prerequisites

- **`uv`** (`[[feedback-python-uses-uv]]`) — install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh` or
  `brew install uv`.
- **Python 3.12** — pinned in `.python-version`. `uv sync` downloads
  it if absent.
- **vast.ai per phase task** — the ~4.0 GB scale is at the CC / owner
  cutoff per memory `[[feedback-large-models-on-vast-ai]]`. Local
  convert on M1 iMac 16 GB is technically feasible (~4 GB sits below
  the 8 GB local ceiling), but the phase task pins vast.ai as the
  conservative default for weights ≥ 2 GB — the Voxtral-Small-24B
  incident (mmap swap 40 GB → Mac forced-shutdown) is the precedent
  that raised the local safety margin. Peak resident memory is
  roughly the whole model plus a `safetensors.torch.save_file`
  serialisation buffer; sibling `magnet_medium_30secs` (~5.7 GB,
  ~12 GB working set) landed without vast.ai but the audiocraft
  pickle format is heavier than a native safetensors of the same
  bytesize, so ~4.0 GB MelodyFlow may push resident closer to
  ~10 GB on the pickle-decode step. See
  `docs/handoff/vast-ai-large-model-publish.md` for the vast.ai
  runbook — provision.sh Wave 12 handles the hf_config.pth shim +
  certifi + xet routing gotchas.

## Owner walkthrough (vast.ai preferred, M1 iMac secondary)

Per memory `[[feedback-large-models-on-vast-ai]]` the ~4.0 GB scale is
at the CC / owner cutoff; the phase task pins vast.ai as the conservative
default. If the owner elects to run local (~4 GB pickle-decode may push
peak resident closer to ~10 GB), the steps are the same — only the
provisioning surface changes.

1. **Rent + provision** (vast.ai path; skip if running local):
   ```bash
   # See docs/handoff/vast-ai-large-model-publish.md §2 for the full
   # runbook (image, budget, provision.sh Wave 12 handling).
   vastai create instance <id> --image nvidia/cuda:13.0.0-devel-ubuntu24.04
   vastai ssh <id>
   # Once on the box, run scripts/provision.sh — it installs uv, pins
   # Python 3.12, and patches hf_config.pth to remove the malicious
   # HF_ENDPOINT override.
   ```

2. **Download** the release:
   ```bash
   hf download facebook/melodyflow-t24-30secs \
     --local-dir ./checkpoints/melodyflow-t24-30secs
   ```

3. **Prepare (torch pickle → flat safetensors)** — from this
   directory:
   ```bash
   cd tools/parity/melodyflow_t24_30secs
   uv sync
   uv run python prepare_checkpoint.py \
     --input-dir ../../../checkpoints/melodyflow-t24-30secs \
     --output    ../../../checkpoints/melodyflow-t24-30secs/flat.safetensors
   ```

   Alternatively, if the release ships as native safetensors from a
   mirror publisher:
   ```bash
   uv run python prepare_checkpoint.py \
     --input-safetensors ../../../checkpoints/melodyflow-t24-30secs/model.safetensors \
     --output            ../../../checkpoints/melodyflow-t24-30secs/flat.safetensors
   ```

4. **Convert** to Vokra GGUF:
   ```bash
   cd ../../..
   ./target/release/vokra-cli convert \
     --model melodyflow-t24-30secs \
     --input ./checkpoints/melodyflow-t24-30secs/flat.safetensors \
     --output ./out/melodyflow-t24-30secs.gguf
   ```

5. **Publish** — T4 tier (Research-only), `--allow-noncommercial`
   **mandatory** per MusicGen family / X-Codec-2 /
   jasco_400m_chords_drums / sibling `magnet_small_10secs` /
   `magnet_medium_30secs` precedent:
   ```bash
   bash scripts/publish/publish-one.sh \
     --gguf ./out/melodyflow-t24-30secs.gguf \
     --repo vokra/melodyflow-t24-30secs \
     --license-spdx cc-by-nc-4.0 \
     --allow-noncommercial \
     --push
   ```

   **Publish will refuse** unless the `docs/license-audit.md` §3.1 row
   `Meta MelodyFlow T24 30secs (\`facebook/melodyflow-t24-30secs\`)`
   has an Approval cell filled in with ☑ Commercial or ☑ Research-only
   (owner fail-closed default per memory
   `[[feedback-license-signoff-primary-source]]`).

6. **Verify**:
   ```bash
   curl -sI https://huggingface.co/vokra/melodyflow-t24-30secs | head -1
   ```

## What the script does NOT do

- **Runtime forward**. This is a converter-side bridge — the
  `flow_editing_inversion` + `t24_transformer` runtime ops are a
  follow-up wave (FR-OP-86 anchor). Loud-partial per RMVPE /
  Charsiu / MOSS-Audio-Tokenizer / MioCodec / sibling MAGNeT
  precedent. The core DiT forward can reuse `vokra_ops::flow_sampler`
  from M3-05 for the ODE integrator, but the editing-specific
  inversion path and the 48 kHz RVQ codec bundle need explicit
  binder ADR judgement — the phase task explicitly punts DiT sampler
  forward to a future wave.
- **License override**. The default `cc-by-nc-4.0` SPDX resolves to
  `LicenseClass::NonCommercial` (T4 fail-closed). A caller who trained
  on a different corpus (or holds the weight under a distinct SPDX id)
  overrides at the outer `--license <spdx>` boundary in `vokra-cli
  convert`.
- **Real-weight parity**. This land is converter code only. A future
  wave (once §3.1 sign-off is granted) will add a
  `parity_melodyflow.rs` test that dumps upstream reference outputs
  and byte-compares the first ODE step / mel frame. Same loud-partial
  defer pattern as RMVPE / DeepFilterNet3 / Charsiu / sibling MAGNeT.

## Owner critical path (post-land)

- **§3.1 sign-off**: fill the `Meta MelodyFlow T24 30secs` row Approval
  cell in `docs/license-audit.md` §3.1. Primary source =
  `https://huggingface.co/facebook/melodyflow-t24-30secs` cardData
  `license: cc-by-nc-4.0` + audiocraft LICENSE file + arXiv:2407.03648.
  Consider **bundling** sign-off with sibling MelodyFlow / MAGNeT /
  MusicGen family rows — same license, same publisher, overlapping
  training-data audit posture; a single owner audit session can cover
  the family cluster.
- **training-data audit** (medium-high risk): Meta MusicGen family
  shares training corpus with Suno / Udio litigation cloud, and the
  MelodyFlow **editing** use-case (existing audio rewritten under a
  new text prompt) is a direct target of the copyright-infringement
  argument in those suits. Legal review before publish (higher
  scrutiny than text-to-music sibling releases).
- **runtime binder ADR** (FR-OP-86): decide whether MelodyFlow's
  editing-specific ODE inversion path and the 48 kHz RVQ codec bundle
  get first-class op paths or stay as a loud-partial defer. Owner
  judgement. The `vokra_ops::flow_sampler` from M3-05 is reusable for
  the core DiT forward; the incremental scope is the inversion path +
  the codec bundle.
- **vast.ai vs local decision** (operational): the phase task pins
  vast.ai as the conservative default for weights ≥ 2 GB, but the
  actual bytesize (~4 GB) is below the 8 GB local ceiling. Owner may
  elect to run local if vast.ai budget is tight; peak resident memory
  during pickle-decode may push closer to ~10 GB, so factor in a
  ~6 GB margin above `free -h` before starting.
