# tools/parity/firered_asr_llm_l

> **Current status (2026-08-29): BLOCKED.** The pinned HF release is a
> `model.pth.tar` bundle rather than the sharded safetensors index assumed by
> this preparer, and the FireRed source revision needed to interpret it is not
> authenticated here. Do not run `uv sync`, fetch weights, or convert locally;
> use `scripts/publish/vast-ai/run-firered-asr-llm-l-validation.sh`, which exits
> before acquisition until the blocker is resolved.

Offline sidecar for **FireRedTeam/FireRedASR-LLM-L**
(`FireRedTeam/FireRedASR-LLM-L`, Apache-2.0 weight record, ~16.6 GB BF16
historical estimate) — the preparer is intended to bridge a future
authenticated safetensors export to the flat safetensors the Rust converter
(`crates/vokra-convert/src/models/firered_asr_llm_l.rs`) consumes.
Sibling of the wave-B fast-track scripts
`../higgs_audio_v3_tts_4b/prepare_checkpoint.py`,
`../magpietts_v2602_prepare_checkpoint.py`, and
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
- **A ≥ 32 GB RAM machine** — the merger loads all ~16.6 GB BF16
  shards into a single in-memory state dict before calling
  `safetensors.torch.save_file` on the merged output. The CC laptop
  (M1 iMac 16 GB) is **insufficient** for this release; the actual
  convert must run on vast.ai per memory
  `[[feedback-large-models-on-vast-ai]]` and the runbook
  `docs/handoff/vast-ai-large-model-publish.md`.

## Historical VAST walkthrough (disabled; do not execute)

> The following is retained only as historical provenance from the earlier
> fast-track. It is **not an execution recipe**. The current worker exits
> before acquisition because the upstream format/source closure is unresolved;
> no command in this section authorizes download, conversion, publication, or
> credential use. Use `run-firered-asr-llm-l-validation.sh` as the only current
> entry point.

The historical notes below reflect the former ≥2 GB VAST planning path; they
are retained for provenance only and are not an approved execution sequence.

1. **Historical planning note — rent** a vast.ai box (≥ 32 GB RAM, e.g. RTX
   4090 or A6000 —
   any CUDA image will do since the merger is CPU-only; a GPU box is
   only cheaper per-hour on vast.ai's market than a CPU box today).
   Reserve ≥ 100 GB SSD for the ~16.6 GB download + ~16.6 GB merged
   safetensors + ~16.6 GB GGUF output.

2. **Historical planning note — provision** per
   `docs/handoff/vast-ai-large-model-publish.md`:

```text
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

3. **Historical planning note — download** the upstream release (~16.6 GB across
   multiple shards):

```text
   export HF_TOKEN=<your-token>
   uv run huggingface-cli download \
       FireRedTeam/FireRedASR-LLM-L \
       --local-dir /root/models/firered-asr-llm-l \
       --exclude "*.bin" "*.pt"
   ```

   The `--exclude` drops any PyTorch pickle mirrors — the sidecar
   works from safetensors only (FR-LD-05 no pickle in the runtime,
   and we don't need pickles here since safetensors is authoritative).

4. **Historical planning note — sync** this sidecar's deps once:

```text
   cd /root/vokra/tools/parity/firered_asr_llm_l
   uv sync
   ```

5. **Historical planning note — merge** the shards into one flat safetensors:

```text
   uv run python prepare_checkpoint.py \
       --input-dir /root/models/firered-asr-llm-l \
       --output    /root/models/firered-asr-llm-l/merged.safetensors
   ```

   The script emits:
   - `merged.safetensors` — the flat file the Rust converter reads.
   - `merged.safetensors.sha256` — sha256 provenance sidecar.
   - `merged.safetensors.shared_pairs.json` — the alias graph for
     any tied embedding (e.g. Qwen2 LM decoder's text embed ↔
     lm_head — a Qwen family `tie_word_embeddings=true` posture
     and a plausible FireRedASR-LLM-L topology).

6. **Historical planning note — convert** to Vokra GGUF (~16.6 GB in → ~16.6 GB out; BF16
   stays BF16 verbatim per the ADR the sibling `qwen3_tts.rs` +
   `moshi.rs` share):

```text
   /root/vokra/target/release/vokra-cli convert \
       --model  firered-asr-llm-l \
       --input  /root/models/firered-asr-llm-l/merged.safetensors \
       --output /root/gguf/firered-asr-llm-l.gguf
   ```

7. **Publication (withheld)** — no publication command is retained here.
   (`docs/license-audit.md` §3.1 sign-off must be marked ☑
   Commercial / ☑ Research-only by owner first — the row is added
   by this CC land, the sign-off is fail-closed until owner audits
   the HF card + FireRedTeam GitHub LICENSE + training-corpus
   commercial posture):

   The historical publish command has been intentionally removed. Upload is
   not authorized by this task and remains `NO_UPLOAD`.

8. **Historical planning note — destroy** the vast.ai instance to stop billing:

```text
   vastai destroy <instance-id>
   ```

## Owner license sign-off (historical; does not clear execution)

**The weight-license sign-off is recorded, but the current source-format,
dependency, and runtime eligibility gates remain blocked.**
`docs/license-audit.md` §3.1 records **☑ Commercial 2026-08-14
yousan**, on primary-source `curl` verification of the HF `cardData`
and the README frontmatter (both `license: apache-2.0`; the commands
and their output are preserved in
`docs/handoff/vast-ai-publish-firered-asr-llm-l.md` §0.1).

What that sign-off did and did not cover:

1. ✅ HF card `license: apache-2.0` verified at primary source
   (`https://huggingface.co/FireRedTeam/FireRedASR-LLM-L`).
2. ✅ FireRedTeam GitHub LICENSE cross-checked
   (`github.com/FireRedTeam/FireRedASR`).
3. ⚠️ **Training-corpus posture was explicitly held OUT of scope** —
   Chinese ASR implies possible WenetSpeech / KeSpeech 混成, which may
   carry per-corpus attribution or NC obligations even though the
   released weight is Apache-2.0. The §3.1 entry records this as a
   risk-acceptance judgement made at publish time, not as a completed
   audit. Re-read it before publishing if that risk posture has moved.
4. ✅ §3.1 row signed ☑ Commercial.

The historical sign-off covers the declared weight license only. It does not
authenticate the current tarball extraction contract, source revision, Qwen
license inheritance, or authorize an upload.

## Honest boundaries

- **Real weight fetch + merge + convert remains blocked** until the tarball
  format and source closure are authenticated. If cleared later, the owner
  must trigger it on vast.ai (memory
  `[[feedback-large-models-on-vast-ai]]`); CC performs no acquisition here.
- **Real audio inference accuracy verification is fixture-gated** —
  when the future runtime binder in
  `crates/vokra-models/src/firered_asr_llm_l/` lands, it will reach
  for a `VOKRA_FIRERED_ASR_LLM_L_REAL_GGUF` env pattern similar to
  the sibling wave-B binders, and the parity workflow runs on the
  runner owner triggers.
- **Conformer + audio-text adapter + Qwen2 LM decoder** — this
  sidecar preserves whatever tokenizer / adapter / LM state the
  upstream shards ship verbatim. A future runtime binder will need a
  Qwen2-family LM decoder forward. A shared `vokra_ops::qwen2` op is a
  PROPOSED consolidation, not a landed module — no such module exists
  today; the only landed Qwen2-family forward is the inline one in
  `crates/vokra-models/src/voxtral/text_decoder.rs`, which `canary_qwen`
  reuses and `kyutai_stt` does not. The Conformer encoder half will
  reuse `vokra_ops::conformer` (canary / parakeet_ctc precedent, a
  module that does exist); the audio-text adapter is a plain
  Linear / MLP.

## License / distribution note

The **`FireRedTeam/FireRedASR-LLM-L`** weight-license record is Apache-2.0,
verified at primary source and signed off as **☑ Commercial 2026-08-14
yousan** in `docs/license-audit.md` §3.1. That record is not an approval to
execute the blocked source conversion or publish a derived artifact.

The Vokra runtime consumes the produced GGUF as an opaque numeric
artefact; no Python / torch / safetensors code enters the runtime
tree (FR-LD-05 sidecar isolation, NFR-DS-02 zero-dep).

## Related

- Rust converter: `crates/vokra-convert/src/models/firered_asr_llm_l.rs`
- Audit ticket: `docs/tickets/coverage-audit-2026-08-03/wave-b/firered-asr-llm-l.md`
- vast.ai runbook: `docs/handoff/vast-ai-large-model-publish.md`
- Wave-B sibling scripts:
  `../higgs_audio_v3_tts_4b/prepare_checkpoint.py`,
  `../magpietts_v2602_prepare_checkpoint.py`,
  `../firered_asr_aed_l_prepare_checkpoint.py`,
  `../voxtral_dump_reference.py`
- Upstream: <https://huggingface.co/FireRedTeam/FireRedASR-LLM-L>
