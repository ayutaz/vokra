# tools/parity/magnet_medium_30secs

Offline sidecar for **facebook/magnet-medium-30secs** (Meta AudioCraft
MAGNeT Medium 30secs, CC-BY-NC-4.0, ~5.7 GB = 1.5B params
non-autoregressive masked-LM transformer + bundled EnCodec 32 kHz codec
+ T5-base text encoder) — bridges the upstream torch-pickle bundle to
the flat safetensors the Rust converter
(`crates/vokra-convert/src/models/magnet_medium_30secs.rs`) consumes.

Sibling of:

- `../magnet_small_10secs/prepare_checkpoint.py` (~2 GB, 500 M / 10 sec
  — same non-autoregressive masked-LM decoding op path, narrower
  hidden width, shorter span)
- `../musicgen_medium_prepare_checkpoint.py` (11.4 GB, vast.ai
  required — AR-over-EnCodec sibling family, entirely different
  decoder loop from MAGNeT non-autoregressive masked-LM)
- `../firered_asr_llm_l/prepare_checkpoint.py` (16.6 GB, vast.ai
  required)

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
- **~12 GB free RAM** — the bridger loads the whole ~5.7 GB checkpoint
  into a single in-memory state dict before calling
  `safetensors.torch.save_file` on the merged output, plus a
  serialisation buffer. The CC laptop (M1 iMac 16 GB) is **sufficient**
  for this release per memory `[[feedback-large-models-on-vast-ai]]`
  (below the 8 GB owner cutoff that pushes work to vast.ai — sibling
  small was ~2 GB with ~4 GB working set, medium is ~5.7 GB with
  ~12 GB working set, still within the 16 GB local ceiling with modest
  margin). No vast.ai handoff needed.

## Local owner walkthrough (M1 iMac safe, no vast.ai)

Per memory `[[feedback-large-models-on-vast-ai]]` the ~5.7 GB scale is
below the 8 GB local-safe threshold, so the entire path runs on the CC
laptop — no rental cost.

1. **Download** the release:
   ```bash
   hf download facebook/magnet-medium-30secs \
     --local-dir ./checkpoints/magnet-medium-30secs
   ```

2. **Prepare (torch pickle → flat safetensors)** — from this
   directory:
   ```bash
   cd tools/parity/magnet_medium_30secs
   uv sync
   uv run python prepare_checkpoint.py \
     --input-dir ../../../checkpoints/magnet-medium-30secs \
     --output    ../../../checkpoints/magnet-medium-30secs/flat.safetensors
   ```

   Alternatively, if the release ships as native safetensors from a
   mirror publisher:
   ```bash
   uv run python prepare_checkpoint.py \
     --input-safetensors ../../../checkpoints/magnet-medium-30secs/model.safetensors \
     --output            ../../../checkpoints/magnet-medium-30secs/flat.safetensors
   ```

3. **Convert** to Vokra GGUF:
   ```bash
   cd ../../..
   ./target/release/vokra-cli convert \
     --model magnet-medium-30secs \
     --input ./checkpoints/magnet-medium-30secs/flat.safetensors \
     --output ./out/magnet-medium-30secs.gguf
   ```

4. **Publish** — T4 tier (Research-only), `--allow-noncommercial`
   **mandatory** per MusicGen family / X-Codec-2 /
   jasco_400m_chords_drums / sibling `magnet_small_10secs` precedent:
   ```bash
   bash scripts/publish/publish-one.sh \
     --gguf ./out/magnet-medium-30secs.gguf \
     --repo vokra/magnet-medium-30secs \
     --license-spdx cc-by-nc-4.0 \
     --allow-noncommercial \
     --push
   ```

   **Publish will refuse** unless the `docs/license-audit.md` §3.1 row
   `Meta MAGNeT Medium 30secs (\`facebook/magnet-medium-30secs\`)` has
   an Approval cell filled in with ☑ Commercial or ☑ Research-only
   (owner fail-closed default per memory
   `[[feedback-license-signoff-primary-source]]`).

5. **Verify**:
   ```bash
   curl -sI https://huggingface.co/vokra/magnet-medium-30secs | head -1
   ```

## What the script does NOT do

- **Runtime forward**. This is a converter-side bridge — the
  `magnet_masked_decode` + `span_masking_scheduler` runtime ops are
  a follow-up wave (FR-OP-85 anchor). Loud-partial per RMVPE /
  Charsiu / MOSS-Audio-Tokenizer / MioCodec / sibling
  `magnet_small_10secs` precedent.
- **License override**. The default `cc-by-nc-4.0` SPDX resolves to
  `LicenseClass::NonCommercial` (T4 fail-closed). A caller who trained
  on a different corpus (or holds the weight under a distinct SPDX id)
  overrides at the outer `--license <spdx>` boundary in `vokra-cli
  convert`.
- **Real-weight parity**. This land is converter code only. A future
  wave (once §3.1 sign-off is granted) will add a `parity_magnet.rs`
  test that dumps upstream reference outputs and byte-compares the
  first token / mel frame. Same loud-partial defer pattern as
  RMVPE / DeepFilterNet3 / Charsiu / sibling `magnet_small_10secs`.

## Owner critical path (post-land)

- **§3.1 sign-off**: fill the `Meta MAGNeT Medium 30secs` row Approval
  cell in `docs/license-audit.md` §3.1. Primary source =
  `https://huggingface.co/facebook/magnet-medium-30secs` cardData
  `license: cc-by-nc-4.0` + audiocraft LICENSE file + arXiv:2401.04577.
  Consider **bundling** sign-off with sibling `magnet_small_10secs`
  row — same license, same publisher, same training-data audit
  posture; a single owner audit session can cover both rows.
- **training-data audit** (medium-high risk): Meta MusicGen family
  shares training corpus with Suno / Udio litigation cloud. Legal
  review before publish (same posture as sibling small).
- **runtime binder ADR** (FR-OP-85): decide whether MAGNeT masked-LM
  parallel decoding gets a first-class op path or stays as a
  loud-partial defer. Owner judgement. The ADR outcome applies to
  BOTH small and medium variants — the op path is shared, only
  hparams differ.
