# Eval-cache GGUF artifacts — provenance and staleness (2026-07-19)

**cc-28** from the M4-residual audit. `~/.cache/vokra-eval/gguf/` is a local,
uncommitted measurement cache shared by every real-weight leg. Some of its GGUFs
predate converter fixes, and because a stale GGUF still *loads*, a leg can look
green while silently exercising far less than it claims. This note records which
artifacts are current, which are superseded, and what was hardened in the repo
as a result.

The cache lives outside the repository, so this note is the repo-side
deliverable. It is filed under `docs/bench-baselines/` (rather than the
gitignore-local `docs/tickets/m5/`) because it is provenance metadata for the
measurement artifacts the tracked bench reports consume — notably
`silero-8k-ctx288-2026-07-19/report.md` §4.2 — and because it needs to survive
for owner ratification.

## Status table

| artifact | state | evidence |
|---|---|---|
| `cosyvoice2-0.5b-llm.gguf` | **superseded** | all 5 `arch.*` dims are the 0 sentinel; loads with `llm = None` |
| `cosyvoice2-0.5b-llm-hparams.gguf` | **current** (regenerated 2026-07-19) | loads with `llm = Some`, real backbone bound |
| `voxtral-mini-3b-f16.gguf` | **superseded** | `text_decoder` `n_head_q`/`n_head_kv`/`n_ctx`/`rope_base`/`rms_norm_eps` all 0 |
| `voxtral-mini-3b-f16-full.gguf` | **superseded** | real hparams but **no `text_decoder.head_dim`** (pre-`12e574e`) |
| `voxtral-mini-3b-bf16-real.gguf` | current | `head_dim = 128` present |
| `voxtral-mini-3b-bf16-fs.gguf` | current | `head_dim = 128`, `adapter.kind = frame_stack_mlp`, `frame_stack = 4` |
| `silero-vad-v5-master.gguf` | current | byte-identical to the committed fixture (`9de80aca...`) |
| `silero-vad-v5-bothrate.gguf` | **mis-named, not stale** | built from the **v5.0 release** ONNX, not master (`f24a814c...`) |

## 1. CosyVoice2 — regenerated

`cosyvoice2-0.5b-llm.gguf` was produced before the hparam fix (`7336079`): it
carries 295 real weight tensors but every LLM dimension is 0.

The regeneration trap the audit recorded is real and confirmed: the checkpoint's
**top-level `config.json` is a 2-byte `{}` stub**. The usable config is
`CosyVoice-BlankEN/config.json` (Qwen2: hidden 896, 24 layers, 14 heads, 2 KV
heads, ffn 4864, vocab 151936).

```sh
vokra-cli convert --model cosyvoice2 \
  --input  ~/.cache/vokra-eval/gguf/cosyvoice2-llm-f32.safetensors \
  --config ~/.cache/vokra-eval/weights/cosyvoice2-0.5b/CosyVoice-BlankEN/config.json \
  --output ~/.cache/vokra-eval/gguf/cosyvoice2-0.5b-llm-hparams.gguf
```

Emits 295 tensors / 19 metadata keys in ~18 s, deriving `vocab=151936
hidden=896 n_layer=24 ffn=4864 n_head=14 n_head_kv=2 n_ctx=32768
attn_bias=true`.

Load verification (real `CosyVoice2Tts::from_path_with_policy`, strict policy):

| file | load | LLM handle | wall |
|---|---|---|---|
| `cosyvoice2-0.5b-llm.gguf` (old) | Ok | **None** | 2.5-2.7 s (metadata only) |
| `cosyvoice2-0.5b-llm-hparams.gguf` (new) | Ok | **Some** | 4.8-10.6 s (real tensor binding) |

The bound config, read back off the live handle, is the upstream Qwen2 geometry
exactly as `CosyVoice-BlankEN/config.json` declares it:

```
LlmBackboneConfig { vocab_size: 151936, hidden_dim: 896, n_layer: 24,
                    n_head_q: 14, n_head_kv: 2, ffn_dim: 4864,
                    rope_base: 1e6, rms_norm_eps: 1e-6, n_ctx: 32768 }
```

Both wall-clock ranges span repeat runs on a machine shared with other build
jobs; the spread is contention, and only the None-vs-Some distinction is load
bearing here.

The new file was written under a **new name** rather than overwriting the old
one: sibling agents may hold the original open, and an in-place 2.5 GB rewrite
is not safe to do underneath them. Deleting the superseded file is an owner
call.

## 2. Repo-side hardening — the silent `llm = None` bind was real (FR-EX-08)

`CosyVoice2Tts::from_gguf_with_policy` tolerated a missing LLM backbone by
matching on the **error variant**:

```rust
let llm = match llm::LlmBackbone::from_gguf(&file, &config) {
    Ok(b) => Some(b),
    Err(VokraError::InvalidArgument(_)) => None,   // <- too wide
    Err(e) => return Err(e),
};
```

The tolerance is deliberate and documented, but only for one case: a GGUF whose
LLM dims are all the converter's 0 sentinel must stay loadable so it can be
inspected and re-converted. `InvalidArgument` is however *also* raised for
wrong-typed metadata keys and for dims that are non-zero but not GQA-well-formed
— so a **genuinely malformed** container reported a successful load, was
indistinguishable from a merely old one, and only failed later with a
misattributed message.

Landed:

- `LlmBackboneConfig::is_placeholder_shape()` — true only when *every* LLM dim
  is 0, deliberately narrower than "cannot host a backbone".
- The bind site now decides from the **config**, not the error variant;
  everything except the sentinel propagates.
- `CosyVoice2Tts::synthesize` names the actual blocker when no backbone is
  bound (0-placeholder dims + the re-convert command + the `config.json` stub
  warning) instead of falling through to the generic scaffold message.
- Tests: `placeholder_shape_gguf_loads_with_absent_llm` (the tolerated case
  still loads) and `malformed_llm_dims_fail_loudly_instead_of_binding_none`
  (n_head 7 vs hidden 512 — non-zero but not GQA-well-formed; **failed before
  the fix, passes after**).

## 3. Voxtral — superseded, deliberately not regenerated

Measured `vokra.voxtral.text_decoder.*` across all four cache variants:

| file | head_dim | n_head_q / kv | n_ctx | rope_base | adapter |
|---|---|---|---|---|---|
| `f16.gguf` | absent | **0 / 0** | **0** | **0.0** | mlp |
| `f16-full.gguf` | **absent** | 32 / 8 | 32768 | 1e8 | mlp |
| `bf16-real.gguf` | 128 | 32 / 8 | 131072 | 1e8 | mlp |
| `bf16-fs.gguf` | 128 | 32 / 8 | 32768 | 1e8 | frame_stack_mlp (x4) |

The f16 pair is **not regenerated**, and that is a deliberate call rather than
an omission: post-fix `bf16` artifacts already exist and are the ones in use,
f16 is the lossy dtype `12e574e` added a BF16 converter to replace, and
regenerating two 8.7 GiB obsolete artifacts costs ~17 GiB and buys no
verification. They should be treated as superseded (deletion is an owner call).

Two things worth carrying forward:

1. **A missing `head_dim` is silently mis-derived for this exact geometry.**
   `TextDecoderConfig::head_dim()` falls back to `hidden_dim / n_head_q` when
   the key is absent. For Voxtral-Mini-3B that is `3072 / 32 = 96`, but the true
   head width is **128** (Mistral decouples the two; the Q projection is
   4096-wide). The divisibility guard does not fire because 32 *does* divide
   3072, so `f16-full.gguf` parses "successfully" with a wrong head width and
   can only fail later, as a tensor-shape mismatch with no re-convert guidance.
   The in-repo test `reads_real_mini_hparams_with_explicit_head_dim` already
   documents that the derived value "would be 96". **Recommended follow-up (not
   done here):** when `head_dim` is absent and the bound tensor shapes
   contradict the derived width, raise the existing "re-convert with a converter
   that writes `text_decoder.head_dim`" error instead of a bare shape mismatch.
   This was left alone because it cannot be honestly validated without loading
   the real 8.7 GiB container (see 2 below).

2. **The Voxtral GGUFs cannot be loaded on this 16 GB machine.** The loader
   reads the whole file and then `GgufFile::parse(bytes.to_vec())` copies it, so
   a 9.4 GiB container needs ~19 GiB resident. No Rust-side load of the bf16
   artifacts was attempted here; the table above is from direct GGUF header
   parsing. This is the same non-mmap constraint tracked for Moshi (audit cc-06)
   and is why the audit's cc-10 lists a "16 GB machine load smoke" as its own
   item.

## 4. Reproducing the metadata survey

The tables above come from parsing GGUF headers directly (no model load, no
large allocation) — key/value walk of the GGUF v3 header, stopping before the
tensor-data region. `silero-8k-ctx288-2026-07-19/measure.py` contains the same
style of dependency-free reader for WAV.
