# tools/parity/higgs_audio_v3_tts_4b

# Current status (2026-08-29): BLOCKED

The upstream weight/code release is governed by the custom Boson Higgs TTS 3
Research-and-Non-Commercial license, and its exact SGLang/codec source closure
has not been authenticated. There is intentionally no `uv.lock`; do not sync,
download, or convert. The VAST worker exits at the stdlib-only gate:
`scripts/publish/vast-ai/run-higgs-audio-v3-tts-4b-validation.sh`.

> ## ⛔ DO NOT RENT A BOX FOR THIS MODEL — publish is refused
>
> **This model cannot be published, and the walkthrough below will not
> complete.** The upstream license is **not** Apache-2.0. Primary-source
> verification on 2026-08-13 found HF `cardData` carrying
> `license: other` +
> `license_name: boson-higgs-tts-3-research-and-non-commercial-license`,
> whose LICENSE §II-A(c) **explicitly forbids redistribution, hosting,
> and TTS-product embedding**.
>
> `docs/license-audit.md` §3.1 accordingly records
> **☑ Rejected 2026-08-14 yousan**, which maps to
> `LicenseClass::RedistributionForbidden` →
> `redistributable() == false` → **`publish-one.sh` refuses at gate 2**.
> Not even the T4 `--allow-noncommercial` route clears it: that flag
> admits non-commercial *redistribution*, and this license forbids
> redistribution outright.
>
> So every step below still runs — rent, download ~8 GB, merge, convert
> — and then the **final** step fails on a licensing check that was
> already knowable before the clock started.
> `docs/handoff/vast-ai-execution-priority.md` therefore recommends
> **skipping this handoff** rather than spending the rental.
>
> The steps are kept for the day the upstream license changes, or for
> local research use that never redistributes. They are **not** a
> publish path today. Primary sources, with the exact `curl` commands
> and the LICENSE quote, are preserved in
> `docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md` §0.1–0.2.

Offline sidecar for **BosonAI Higgs-Audio v3 TTS 4B**
(`bosonai/higgs-audio-v3-tts-4b`,
**LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial** — SPDX-unregistered
custom license, *not* Apache-2.0; see the banner above, ~8 GB BF16) —
bridges the upstream **sharded safetensors** release to the flat
safetensors the Rust converter
(`crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`)
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
   bash scripts/publish/vast-ai/provision.sh  # torch, huggingface_hub<0.30 pin, etc.
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

7. **Publish** — ⛔ **not available for this model.** There is no
   working publish command to give here. `docs/license-audit.md` §3.1
   records **☑ Rejected 2026-08-14 yousan**
   (`LicenseClass::RedistributionForbidden`), so `publish-one.sh`
   refuses at **gate 2 (redistributable)** regardless of flags. The
   command that a reader of the old revision of this file would have
   typed —

   ```bash
   # ⛔ REFUSED at gate 2 — shown so it is recognisable, not to be run.
   #    `--license-spdx apache-2.0` also misstates the upstream license.
   # bash scripts/publish/publish-one.sh \
   #     --gguf /root/gguf/higgs-audio-v3-tts-4b.gguf \
   #     --repo vokra/higgs-audio-v3-tts-4b \
   #     --license-spdx apache-2.0 --allow-large --push
   ```

   — cannot succeed, and `--allow-noncommercial` does not rescue it
   (that flag permits non-commercial *redistribution*; this license
   forbids redistribution outright). If the upstream license ever
   changes, re-verify at primary source, get a fresh §3.1 sign-off,
   and only then restore a publish step with the *correct* SPDX id.

8. **Destroy** the vast.ai instance to stop billing:

   ```bash
   vastai destroy <instance-id>
   ```

## Owner critical path — CLOSED (rejected 2026-08-14)

**This section is resolved; nothing here is outstanding.** It is kept
because the earlier revision of this file listed these as open owner
tasks premised on an Apache-2.0 reading, and a reader who remembers
that list needs to see how each item actually landed:

1. ~~Verify the HF card `license: apache-2.0` at primary source.~~
   **Done, and it refuted the premise** — the card carries
   `license: other` +
   `license_name: boson-higgs-tts-3-research-and-non-commercial-license`
   (CC `curl`, 2026-08-13; commands preserved in
   `docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md` §0.1).
2. ~~Cross-check the BosonAI GitHub LICENSE file.~~ **Done** — the
   LICENSE is titled *"BOSON HIGGS TTS 3 RESEARCH AND NON-COMMERCIAL
   LICENSE AGREEMENT"*; §II-A(c) forbids redistribution, hosting, and
   TTS-product embedding (§0.2 of the same handoff).
3. ~~Audit the training-corpus commercial-use posture.~~ **Moot** —
   the weight's own license already blocks redistribution, so the
   corpus question never becomes load-bearing.
4. ~~Add the ☑ sign-off to `docs/license-audit.md` §3.1.~~ **Done:
   ☑ Rejected 2026-08-14 yousan.**

`publish-one.sh` reads the sign-off via
`scripts/publish/signoff_match.py`. A **blank** row refuses at gate 4;
this row is not blank but **Rejected**, so it refuses earlier, at
gate 2 (`redistributable() == false`).

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
  `crates/vokra-core/src/decode/sampler.rs` primitive (already
  wired through voxtral / cosyvoice2 / canary_qwen). SGLang is a
  _reference-side implementation detail_ that the Vokra runtime
  does not inherit. No converter change is needed for the swap.

## License / distribution note

The **`bosonai/higgs-audio-v3-tts-4b`** upstream release is
**LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial** — a bespoke,
SPDX-unregistered research-and-non-commercial license that forbids
redistribution (§II-A(c)).

**The wave-B audit ticket says Apache-2.0. The ticket is wrong**, and
is left unedited as the historical record of what was assumed. That
assumption came from a default, not from the HF card; primary-source
`curl` verification on 2026-08-13 refuted it. Where this sidecar and
the ticket disagree about the license, **this file and
`docs/license-audit.md` §3.1 are authoritative** —
§3.1 records ☑ Rejected 2026-08-14 yousan.

Practical consequence: the GGUF may be produced locally for research,
but it **must not be uploaded to `huggingface.co/vokra`**, and the
publish chain enforces that rather than relying on the reader.

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
