# tools/parity/higgs_audio_v3_tts_4b

Offline sidecar for **BosonAI Higgs-Audio v3 TTS 4B**
(`bosonai/higgs-audio-v3-tts-4b`, Apache-2.0, ~8 GB BF16) — bridges the
upstream **sharded safetensors** release to the flat safetensors the
Rust converter (`crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`)
consumes. Sibling of the wave-B fast-track scripts
`../magpietts_v2602_prepare_checkpoint.py` and
`../firered_asr_aed_l_prepare_checkpoint.py`.

## What this directory contains

- `prepare_checkpoint.py` — the actual sharded → flat merger.
  Discovers shards via `model.safetensors.index.json`, dedupes tied
  tensors (data_ptr collision → clone + audit trail), strips
  non-float training scaffold (`.num_batches_tracked` /
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
- **A ≥ 16 GB RAM machine** — the merger loads all ~8 GB BF16 shards
  into a single in-memory state dict before calling
  `safetensors.torch.save_file` on the merged output. On the CC
  laptop (M1 iMac 16 GB) this is at the edge of the safe zone; the
  actual convert typically runs on vast.ai per the size threshold
  memory `[[feedback-large-models-on-vast-ai]]` demands and the
  runbook `docs/handoff/vast-ai-large-model-publish.md` documents.

## vast.ai owner walkthrough — DL + prep + convert + publish

Per memory `[[feedback-large-models-on-vast-ai]]` the ~8 GB scale
puts this model above the 2 GB CC-workflow local-convert threshold.
The full path runs on a rented vast.ai GPU box.

1. **Rent** a vast.ai box (>= 24 GB RAM, e.g. RTX 4090 or A6000 —
   any CUDA image will do since the merger is CPU-only; a GPU box is
   only cheaper per-hour on vast.ai's market than a CPU box today).

2. **Provision** per `docs/handoff/vast-ai-large-model-publish.md`:

   ```bash
   # On the vast.ai host, from a fresh clone of the repo:
   cd /root/vokra
   bash scripts/vast-ai/provision.sh          # torch, huggingface_hub<0.30 pin, etc.
   ```

   The provision script pins `huggingface_hub<0.30` on vast.ai to
   sidestep the xet-token routing 404 documented in
   `[[reference-huggingface-hub-lt-030-vast-ai]]`. The local M1 iMac
   has no such constraint but this sidecar's `pyproject.toml`
   declares the range broadly (`>= 0.26`) so both environments work
   — the runbook enforces the vast.ai pin explicitly.

3. **Download** the sharded upstream release (~8 GB across ~2–3
   shards depending on the release's shard cadence):

   ```bash
   export HF_TOKEN=<your-token>
   uv run huggingface-cli download \
       bosonai/higgs-audio-v3-tts-4b \
       --local-dir /root/models/higgs-audio-v3-tts-4b \
       --exclude "*.bin" "*.pt"
   ```

   The `--exclude` drops any PyTorch pickle mirrors — the sidecar
   works from safetensors only (FR-LD-05 no pickle in the runtime,
   and we don't need pickles here since safetensors is authoritative).

4. **Sync** this sidecar's deps once:

   ```bash
   cd /root/vokra/tools/parity/higgs_audio_v3_tts_4b
   uv sync
   ```

5. **Merge** the shards into one flat safetensors:

   ```bash
   uv run python prepare_checkpoint.py \
       --input-dir /root/models/higgs-audio-v3-tts-4b \
       --output    /root/models/higgs-audio-v3-tts-4b/merged.safetensors
   ```

   The script emits:
   - `merged.safetensors` — the flat file the Rust converter reads.
   - `merged.safetensors.sha256` — sha256 provenance sidecar.
   - `merged.safetensors.shared_pairs.json` — the alias graph for
     any tied embedding (e.g. text embed ↔ lm_head — a Qwen /
     MiniCPM `tie_word_embeddings=true` posture and a plausible
     Higgs-Audio topology).

6. **Convert** to Vokra GGUF (~8 GB in → ~8 GB out; BF16 stays
   BF16 verbatim per the ADR the sibling `qwen3_tts.rs` +
   `moshi.rs` share):

   ```bash
   /root/vokra/target/release/vokra-cli convert \
       --model  higgs-audio-v3-tts-4b \
       --input  /root/models/higgs-audio-v3-tts-4b/merged.safetensors \
       --output /root/gguf/higgs-audio-v3-tts-4b.gguf
   ```

7. **Publish** through the 5-gate publish chain
   (`docs/license-audit.md` §3.1 sign-off must be marked ☑
   Commercial / ☑ Research-only by owner first — the row is added by
   this CC land, the sign-off is fail-closed until owner audits the
   HF card + BosonAI GitHub LICENSE + training-corpus commercial
   posture):

   ```bash
   bash scripts/publish/publish-one.sh \
       --gguf /root/gguf/higgs-audio-v3-tts-4b.gguf \
       --repo vokra/higgs-audio-v3-tts-4b \
       --license-spdx apache-2.0 \
       --allow-large \
       --push
   ```

8. **Destroy** the vast.ai instance to stop billing:

   ```bash
   vastai destroy <instance-id>
   ```

## Owner critical path (from the audit ticket §Owner critical path)

Even after this CC land, publish stays fail-closed until owner:

1. Verifies the HF card `license: apache-2.0` at primary source
   (`https://huggingface.co/bosonai/higgs-audio-v3-tts-4b`).
2. Cross-checks the BosonAI GitHub LICENSE file
   (`github.com/boson-ai/higgs-audio`).
3. Audits the training-corpus commercial-use posture — 100+
   languages implies possible Common Voice / VoxPopuli 混成 which
   may carry per-corpus attribution obligations even though the
   released weight itself is Apache-2.0.
4. Adds the ☑ Commercial or ☑ Research-only sign-off to
   `docs/license-audit.md` §3.1 row (this CC land inserts the row
   with blank sign-off + `______________` — the fail-closed default
   per memory `[[feedback-license-signoff-primary-source]]`).

Only step (4) matters for the publish gate. `publish-one.sh` reads
the signoff via `scripts/publish/signoff_match.py`; a blank row
refuses the publish (`upload.sh` refuses per gate 4).

## Honest boundaries

- **Real weight fetch + merge + convert is owner-triggered on
  vast.ai**, not run by CC (memory
  `[[feedback-large-models-on-vast-ai]]`). CC lands only the
  converter code + this sidecar + the tests.
- **Real audio inference accuracy verification is fixture-gated** —
  when the future runtime binder in
  `crates/vokra-models/src/higgs_audio_v3_tts_4b/` lands, it will
  reach for the same `VOKRA_HIGGS_AUDIO_V3_TTS_4B_REAL_GGUF` env
  pattern the sibling wave-B binders use, and the parity workflow
  runs on the runner owner triggers.
- **Emotion inline tags** (`[happy]` / `[sad]` / …) — the ticket
  notes these are baked into the LM tokenizer at training time; the
  Rust runtime binder will consume them through the tokenizer
  layer, not through any converter-side transformation. This
  sidecar preserves whatever tokenizer state the upstream shards
  ship verbatim.
- **SGLang sampler → Vokra Sampler primitive** — the upstream
  reference implementation uses SGLang's sampler. When the future
  runtime binder lands, it will consume Vokra's
  `crates/vokra-core/src/engine/sampler.rs` primitive (already
  wired through voxtral / cosyvoice2 / canary_qwen). SGLang is a
  _reference-side implementation detail_ that the Vokra runtime
  does not inherit. No converter change is needed for the swap.

## License / distribution note

The **`bosonai/higgs-audio-v3-tts-4b`** upstream release is
Apache-2.0 per the HF card at the time of writing; the wave-B ticket
records this. **Owner primary-source verification is pending** per
§Owner critical path above — the ☑ sign-off column in
`docs/license-audit.md` §3.1 stays blank until owner attests to it.

The Vokra runtime consumes the produced GGUF as an opaque numeric
artefact; no Python / SGLang / torch / safetensors code enters the
runtime tree (FR-LD-05 sidecar isolation, NFR-DS-02 zero-dep).

## Related

- Rust converter: `crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`
- Audit ticket: `docs/tickets/coverage-audit-2026-08-03/wave-b/higgs-audio-v3-tts-4b.md`
- vast.ai runbook: `docs/handoff/vast-ai-large-model-publish.md`
- Wave-B sibling scripts: `../magpietts_v2602_prepare_checkpoint.py`,
  `../firered_asr_aed_l_prepare_checkpoint.py`,
  `../voxtral_dump_reference.py`
- Upstream: <https://huggingface.co/bosonai/higgs-audio-v3-tts-4b>
