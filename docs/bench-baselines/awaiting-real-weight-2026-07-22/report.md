# Real-weight verification of the "implemented, awaiting real weights" models (2026-07-22)

Four models were catalogued as **implemented but not verified against a real
checkpoint**: Voxtral, Moshi, CSM-1B and WavTokenizer. This report records what
was actually measured for each on this machine (Apple M1 iMac, 16 GiB, CPU).
The sibling records are `server-real-gguf-slots-2026-07-21` (cc-40) and
`m1-real-weight-eval-2026-07-16` (the two real-weight campaigns).

Every number below is a tool's own output — `cargo test` harness `eprintln`,
`vokra-convert` / `vokra-cli` stdout, or `/usr/bin/time -l`. Nothing is
estimated. Where a leg could not run, the reason is measured, not guessed.

**Result in one line: Voxtral's audio tower is verified against upstream
reference taps, Moshi's full 7.688B checkpoint converts and runs end-to-end for
the first time, and the two remaining gaps (Voxtral's text decoder, CSM) are
blocked for reasons that are now precisely identified rather than assumed.**

## 1. Voxtral — audio tower verified, text decoder memory-blocked

### 1.1 What passed

`cargo test --release -p vokra-models --test voxtral_tower_parity --test
parity_voxtral -- --test-threads=1 --skip decoder_step_logits`, with
`VOKRA_VOXTRAL_GGUF` = the real 8.71 GiB `voxtral-mini-3b-bf16-fs.gguf` and
`VOKRA_VOXTRAL_REF_DIR` = the upstream tap dumps. **5 passed, 0 failed.**

Against **upstream reference taps** (`voxtral_tower_parity`), audio hparams
`n_layer=32 d=1280 n_head=20 n_ctx=1500 n_mels=128 ffn=5120`:

| tap | max abs delta | atol | verdict |
|---|---|---|---|
| `own_log_mel` | 2.086e-5 | 5.0e-5 | pass |
| `conv_stem_pos` | 6.914e-6 | 1.5e-5 | pass |
| `after_layer_0` | 6.795e-6 | 1.5e-5 | pass |
| `after_layer_15` | 2.480e-5 | 5.0e-5 | pass |
| `encoder_final` | 6.454e-4 | 1.5e-3 | pass |
| `soft_prefix` | 2.617e-5 | 6.0e-5 | pass |

The soft prefix is 375 rows x 3072, projected from 1500 encoder positions.

Against **committed fixtures** (`parity_voxtral`): `own_log_mel` 2.086e-5
(atol 5.0e-5), `audio_encoder_out` 1.717e-5 (atol 1.5e-3), `soft_prefix`
2.682e-6 (atol 6.0e-5) — all pass.

So the full mel frontend → 32-layer tower → projector path matches upstream on
the real checkpoint.

### 1.2 What did not run, and exactly why

The `decoder_step_logits_and_greedy_match_committed_fixtures` leg was skipped.
A first attempt that included it was killed after ~6 minutes with swap growing
from 6.0 GB to 9.2 GB. Header-only accounting of the GGUF tensor directory
(payloads never read) gives the reason:

| prefix | stored (BF16) | as f32 |
|---|---|---|
| `language_model` | 7.48 GiB | **14.95 GiB** |
| `audio_tower` | 1.19 GiB | 2.37 GiB |
| `multi_modal_projector` | 0.05 GiB | 0.09 GiB |
| **total** | **8.71 GiB** | **17.42 GiB** |

`vokra_mmap::open_gguf` removes the file-read copy, but Voxtral's weight
binding materialises owned `Vec<f32>` (e.g. `AudioEncoder`'s `conv1_w`,
`pos_emb`, layer fields), so the text decoder alone needs ~15 GiB resident.
That does not fit in 16 GiB alongside the OS. **This is a binding-strategy
limit, not a hardware verdict** — see §5.1.

## 2. Moshi — full 7.688B converts in 305 MiB and runs end-to-end

### 2.1 Conversion (first run of the streaming path at full scale)

`vokra-convert --model moshi` on the real `kyutai/moshiko-pytorch-bf16`
`model.safetensors` (15,375,500,136 bytes) with the SentencePiece side-car:

```
converted moshi: 355 tensors, 71 metadata keys, 15,376,043,200 bytes
  355 BF16 passthrough, 0 non-float skipped, tokenizer embedded: true
verified load: version 3, alignment 32, 355 tensors, 71 metadata keys;
  arch=moshi temporal_layers=32 depth_layers=6 n_q_in=16 dep_q=8 attribution=present
       17.93 real
   320,667,648  maximum resident set size
   319,555,264  peak memory footprint
             0  swaps
```

Against the campaign-2 record (`m1-real-weight-eval-2026-07-16`, same machine),
which could only convert a **2+2-layer truncation**:

| | campaign-2 (truncated 1.111B) | this run (full 7.688B) |
|---|---|---|
| peak memory footprint | 15.12 GiB | **0.30 GiB** |
| output | 4.44 GB | **15.38 GB** |
| wall | 19.56 s (median of 3) | 17.93 s |

The bounded-memory `convert_streaming` path landed in `ff12104` (PR #8,
2026-07-19) — **after** the campaign-2 measurement, which is why the campaign
recorded the old materialise-everything behaviour. The "full-7B needs ~97 GiB /
mmap unwired" note is empirically obsolete.

### 2.2 Full-model run

`vokra-cli run --model <full.gguf> --input mic-24k-5f.wav --duplex
--echo-sim 0.3 --deterministic`, i.e. the same 5-frame input campaign-2 used:

```
s2s-duplex: 5 mic frames -> 5 model frames (9600 samples) @ 24000 Hz,
            echo-sim gain 0.3 (AEC active)
s2s-duplex monologue: " Hello"
      409.34 real
 7,151,501,312  maximum resident set size   (6.66 GiB)
11,425,688,704  peak memory footprint       (10.64 GiB)
             0  swaps
```

The inner monologue is the qualitative result that matters. Campaign-2's
truncated model emitted `"workscrack unitary "pine"`; the full model emits
`" Hello"`. A 2-layer truncation producing word salad is expected, so a
coherent token from the full stack is direct evidence that the conversion and
the model binding are correct at full scale.

Loading works because `MoshiEngine::from_path` uses
`WeightResidency::MappedLazy` — temporal blocks stay in the GGUF mapping and
widen one layer at a time per forward. The enum's own doc puts the `Resident`
alternative at "~30 GiB at the full-7B shape", which matches the independent
header accounting below:

| prefix | stored (BF16) | as f32 |
|---|---|---|
| `transformer` | 12.25 GiB | 24.50 GiB |
| `depformer` | 1.15 GiB | 2.30 GiB |
| others (8 prefixes) | 0.92 GiB | 1.84 GiB |
| **total** | **14.32 GiB** | **28.64 GiB** |

**Speed is not a result to celebrate**: 409.34 s for 5 frames is 81.9 s/frame,
and one frame is 80 ms of audio, so RTF is roughly 1024. Campaign-2's truncated
model ran 0.699 s/frame (RTF 8.7); the full model is ~117x slower for ~7x the
parameters, i.e. worse than linear. That is the `MappedLazy` trade-off — every
forward re-widens the layers it touches. **This run demonstrates that the model
runs, not that it runs usefully.**

### 2.3 With the real Mimi codec bound

Re-running with `--mimi` pointed at a freshly converted Mimi GGUF (§3):

```
vokra: real Mimi codec bound from mimi-reconv.gguf
s2s-duplex: 5 mic frames -> 5 model frames (9600 samples) @ 24000 Hz
s2s-duplex monologue: ""
      415.38 real
 6,935,920,640  maximum resident set size   (6.46 GiB)
12,603,885,696  peak memory footprint       (11.74 GiB)
             0  swaps
```

Exit 0, 9600 samples written. **The monologue is empty here, where the
synthesized bridge produced `" Hello"`.** This is recorded as an observation,
not a defect: with the bridge the audio tokens carry no real audio semantics
(the CLI says so), so the model was responding to essentially arbitrary input,
whereas with the real codec it hears 400 ms of actual audio and stays silent —
which may well be correct behaviour. **5 frames is too short to distinguish
"correct silence" from "no output".** Settling it needs a longer run (25 frames
≈ 2 s ≈ 35 min at the measured rate); that has not been done.

## 3. Mimi — the cached GGUF was a stale artifact, not a code defect

The first real-codec attempt failed with an explicit error rather than a silent
fallback (FR-EX-08 behaving correctly):

```
error: --mimi .../mimi.gguf: invalid argument: mimi config: seanet carries a
0-placeholder (MimiSeanetConfig { dimension: 0, n_filters: 0, ... })
```

Inspecting that cached file's header shows 319 tensors / 9 metadata keys, with
only `vokra.mimi.{n_codebooks,codebook_size,d_model}` — no seanet group.
Re-converting the same upstream checkpoint with the HEAD converter gives:

```
converted mimi: 603 tensors, 36 metadata keys, 906,350,656 bytes
  ... neural-chain adapter wrote 284 structural mimi.enc.*/mimi.dec.* tensors
  + the vokra.mimi.* config chunk group (PCM encode/decode bindable)
verified load: arch=mimi n_codebooks=32 codebook_size=2048 d_model=512
        6.46 real / 2,360,115,200 max RSS / 0 swaps
```

The converter has emitted `vokra.mimi.seanet.*` all along (29 references in
`crates/vokra-convert/src/models/mimi.rs`). The cached GGUF simply predates
that support. **No code change was needed; the artifact was stale.**

## 4. CSM-1B — blocked on owner action, precisely diagnosed

Two distinct blockers, measured:

1. **The Hugging Face token is dead.** `GET /api/whoami-v2` with
   `~/.cache/huggingface/token` (dated 2025-12-02) returns **HTTP 401**.
2. **The repos are gated.** `sesame/csm-1b` metadata is public (HTTP 200,
   `license:apache-2.0`, `gated: auto`) but file resolve returns **HTTP 401**:
   *"Access to model sesame/csm-1b is restricted."* Same for
   `meta-llama/Llama-3.2-1B`.

Owner action is therefore: issue a fresh token, then accept the terms on both
repos (`sesame/csm-1b` is auto-approve; the Llama repo needs the Meta licence
click). Everything downstream — download, convert, parity — is CC-side.

## 5. What this leaves open

### 5.1 Voxtral has no `MappedLazy` equivalent

Moshi carries `WeightResidency::MappedLazy`; Voxtral binds eagerly to
`Vec<f32>`. Porting the same strategy would (a) let the Voxtral decoder parity
leg run in bounded memory, and (b) close the `voxtral` slot skip in
`integrations/vokra-server/tests/real_gguf_slots.rs`, which skips for the same
underlying reason. This is CC-implementable and needs no owner input.

### 5.2 `MappedLazy` re-widens on every forward

The measured 81.9 s/frame is dominated by repeated BF16→f32 widening. A bounded
layer cache would keep the memory ceiling while removing most of that cost. The
current implementation sits at the extreme memory-first end of the trade-off.

### 5.3 Stale GGUFs degrade silently

The stale `mimi.gguf` did not fail loudly in the no-`--mimi` path — it caused a
quiet fallback to the synthesized bridge (with a NOTE). Stamping a converter
version into the GGUF and warning on load would make this class of drift
visible.

### 5.4 Not attempted here

- Voxtral text-decoder parity and multilingual WER (needs §5.1 or a bigger host).
- A Moshi run long enough to interpret §2.3's empty monologue.
- Moshi parity against a PyTorch reference at full scale (the reference side
  would need the 14 GB checkpoint resident in torch).
- WavTokenizer: **not a verification task.** `wavtokenizer_vq` implements the
  single-stage VQ lookup only; there is no `vokra-models` module and no
  converter for the model, so a real-weight roundtrip needs a model WP first.

## 6. Red lines

No fabricated number: every value is a tool's own output. No parity bound was
changed or relaxed — the Voxtral legs ran at their committed atols. No silent
fallback was accepted as a pass: the `--mimi` placeholder error and the
memory-blocked decoder leg are both recorded as failures/skips, not smoothed
over. Zero-dep is untouched (no dependency was added; the header accounting used
python3 stdlib only, outside the build).
