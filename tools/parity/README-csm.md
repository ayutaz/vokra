# CSM staged parity fixtures (M4-05 T23/T24)

Maintainers run every `cargo ... -p vokra-models` command below on VAST. Do
not compile or test `vokra-models` on the development Mac.

Offline recipe for the Sesame CSM-1B staged reference evidence. CI never runs
Python; the real dump is a **VAST owner step** and remains evidence-only until
the native CSM+Mimi composite binder and CPU parity are accepted.

## Committed today

- `tests/parity/csm/self-test/` — a synthetic fixture written by
  `csm_dump.py self-test` (stdlib-only, SplitMix64-deterministic). It
  carries **no reference semantics**; it pins the file/manifest format and
  the Rust reader (`parity_csm.rs::synthetic_fixture_manifest_roundtrip`).

## Official reference evidence (owner, VAST)

1. Run `scripts/publish/vast-ai/run-csm-1b-inspection.sh` on a clean Linux
   x86-64 VAST checkout. It pins and inspects the fixed `sesame/csm-1b`
   snapshot, the clean source checkout, and the pinned Transformers CSM
   implementation. It performs no conversion or upload. The worker requires
   the dedicated `tools/parity/csm_1b_reference/uv.lock`; it does not use the
   broad parity lock or `uv run --with`.
2. Supply the resulting inspection bundle and an authenticated packet to
   `scripts/publish/vast-ai/run-csm-1b-validation.sh`. The worker invokes
   `csm_1b_dump_reference.py` through the dedicated Transformers-4.52.1 lock:

   The packet boundary is an exact conversation: `messages` contains the
   source-shaped role/content entries; each typed `audio` content item embeds
   a relative path contained by the packet directory. The adapter calls the
   authenticated processor's `apply_chat_template(tokenize=True,
   return_dict=True)` before generation, so BOS/EOS handling is owned by the
   official route. It authenticates each audio input's size/SHA-256.

   ```sh
   uv run --frozen --project tools/parity/csm_1b_reference --python 3.12 python \
       tools/parity/csm_1b_dump_reference.py \
       --snapshot /path/to/csm-1b \
       --transformers /path/to/transformers \
       --inspection-manifest /path/to/inspection/evidence/manifest.json \
       --packet /path/to/reference-packet.json \
       --output /dev/shm/csm-reference
   ```

   The official Transformers path runs **greedy** (`do_sample=False` and
   `depth_decoder_do_sample=False`). The packet includes per-step backbone
   logits, generation-frame-aligned last hidden states, exact depth-decoder
   per-codebook logits in call order, generated code IDs, and codec-decoded
   PCM. The adapter requires generated codes `[B,frames,32]`, one depth call
   per frame/codebook (`frames * 31`), frame-aligned logits/hidden states, and
   PCM at exactly `decoded_frames * 1920` samples, where a final all-zero EOS
   frame is excluded from `decoded_frames`. The PCM is explicitly
   **pre-watermark**; it is not source final watermarked output.

   Reference evidence is independent of the native/composite gate. A complete
   GGUF and accepted CPU baseline are checked only by the later native stage;
   they are not prerequisites for collecting official reference evidence.

3. Point the staged Rust leg at the resulting `reference/` directory only to
   authenticate artifact presence. It deliberately fails closed with
   `INSPECTION_ONLY`; it does not claim CPU or Metal parity:

   ```sh
   VOKRA_CSM_PARITY_DIR=tests/parity/csm/reference \
       cargo test -p vokra-models --test parity_csm -- --nocapture
   ```

   Until the complete composite binder and accepted VAST CPU baseline land,
   the env-gated leg reports a loud blocker — never a pass.

## Judgement (ADR M4-05 §D7)

- `generated_frame_codes.u32le` — discrete: **bit-exact** primary judgement.
- `backbone_hidden_last` / `backbone_logits` / `depth_decoder_logits` /
  `official_pcm_pre_watermark` — FP32
  `atol = 0.01` (NFR-QL-01) starting point; any per-tensor relaxation must
  be architectural-bound-derived and recorded in rustdoc + ADR + CI
  (Kokoro `PROSODY_F0_ATOL` precedent).
