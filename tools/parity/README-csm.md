# CSM staged parity fixtures (M4-05 T23/T24)

Offline recipe for the Sesame CSM-1B staged reference dump the Rust parity
test (`crates/vokra-models/tests/parity_csm.rs`) consumes. CI never runs
Python; the real dump is an **owner step after T29** (both upstream repos
are gated downloads).

## Committed today

- `tests/parity/csm/self-test/` — a synthetic fixture written by
  `csm_dump.py self-test` (stdlib-only, SplitMix64-deterministic). It
  carries **no reference semantics**; it pins the file/manifest format and
  the Rust reader (`parity_csm.rs::synthetic_fixture_manifest_roundtrip`).

## Real dump (owner, post-T29)

1. Accept the gated licenses and download:
   - `sesame/csm-1b` (HF, Apache 2.0 weights — record the revision),
   - `meta-llama/Llama-3.2-1B` tokenizer file,
   - Mimi weights `kyutai/moshiko-pytorch-bf16`
     `tokenizer-e351c8d8-checkpoint125.safetensors` (CC-BY 4.0).
2. Create a venv and install the upstream stack **pinned** (record every
   version in the manifest): `torch`, the `SesameAILabs/csm` package at the
   commit SHA you re-pin in ADR M4-05 §D2, `moshi`, `transformers`.
   Re-use the version pins from `tools/parity/parity-requirements.txt`
   where they overlap.
3. Run:

   ```sh
   python3 tools/parity/csm_dump.py dump \
       --checkpoint /path/to/csm-1b \
       --tokenizer /path/to/llama-3.2-tokenizer \
       --text "Hello from Vokra." --speaker 0 --max-frames 25 \
       --out tests/parity/csm/reference
   ```

   The dump runs **temperature-0** so the code sequence is exactly
   reproducible (a stochastic dump must never become a parity reference —
   fabricated pass 禁止).

4. Point the Rust side at it:

   ```sh
   VOKRA_CSM_PARITY_DIR=tests/parity/csm/reference \
       cargo test -p vokra-models --test parity_csm -- --nocapture
   ```

   Until the T29 tensor manifest also lands the real-weight `from_gguf`
   binding, the env-gated legs report a **loud skip naming T29** — never a
   pass.

## Judgement (ADR M4-05 §D7)

- `frame_codes.u32` — discrete: **bit-exact** primary judgement.
- `backbone_hidden` / `c0_logits` / `depth_logits` / `decode_pcm` — FP32
  `atol = 0.01` (NFR-QL-01) starting point; any per-tensor relaxation must
  be architectural-bound-derived and recorded in rustdoc + ADR + CI
  (Kokoro `PROSODY_F0_ATOL` precedent).
