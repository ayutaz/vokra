# Mac CPU / Metal coverage ledger (2026-08-24)

This is the execution ledger for the maintainer request to make the public
`huggingface.co/vokra` GGUFs usable on Mac CPU and Metal. Qualcomm/QNN is out
of scope for this wave. Counts below are repository counts, not architecture
counts; one architecture can have many public checkpoints.

## Live public inventory

The read-only audit command is:

```text
uv run --no-project --python 3.12 python tools/audit/hf_mac_coverage.py
```

At 2026-08-25, after the SNAC, FocalCodec, MeloTTS, DAC-sibling, speaker,
Piper, FCPE, standalone BERT-family, WavTokenizer and NeuCodec CPU/Metal
waves, it reported:

| Inventory / code reachability | Public repos |
|---|---:|
| Public model repositories | 194 |
| Repositories carrying at least one GGUF | 193 |
| GGUF files | 198 |
| Complete CPU code route | 87 |
| Route/binder present, released-artifact CPU forward incomplete | 49 |
| No complete runtime binder | 57 |
| Empty non-artifact repository (`seamless-m4t-v2-large`) | 1 |
| Complete Metal code route among the CPU-complete set | 87 |
| CPU-complete but Metal-unsupported | 0 |
| Metal blocked by missing/partial CPU forward | 106 |

These are deliberately **code reachability** counts. They are not a claim that
the current Hub file loads, that its sidecars are complete, or that its
real-weight CPU/Metal parity has passed. The TSV form prints the per-repository
revision, GGUF count, architecture and classification:

```text
uv run --no-project --python 3.12 python tools/audit/hf_mac_coverage.py --format tsv
```

The 87 repositories with a complete Metal code route are the four BigVGAN
checkpoints, CAM++, CrisperWhisper, both Distil-Whisper checkpoints, FCPE,
the three DAC checkpoints (16, 24 and 44.1 kHz), the three FocalCodec
checkpoints (50, 25 and 12.5 Hz), FireRedVAD, FSMN-VAD,
HiFi-GAN LibriTTS, both Kokoro checkpoints, all five MeloTTS checkpoints,
Kotoba-Whisper, Mimi, Moshiko-7B, Moonshine Tiny/Base, NKF-AEC, Parakeet CTC
1.1B, Parakeet TDT 0.6B v3, both Piper Plus checkpoints, RNNoise, both Silero
VAD checkpoints, both SNAC checkpoints (24 and 44.1 kHz), SmartTurn v2,
SpeechT5 HiFi-GAN, TEN-VAD, both Vocos
checkpoints, the three Voxtral checkpoints, nine plain Whisper checkpoints,
Whisper-Medusa-v1, all seven Wav2Vec2 CTC checkpoints, Data2Vec Audio Base,
HuBERT Large LS960, all seven SepFormer checkpoints, and both SpeechBrain
X-vector repositories, the canonical SpeechBrain ECAPA-TDNN repository, the
public pyannote WeSpeaker ResNet34-LM repository, and both byte-identical
TitaNet-Large repositories (`vokra/titanet-l` and `vokra/titanet-large`), plus
the standalone Chinese RoBERTa, Japanese DeBERTa v2 and DeBERTa v3 text
encoders.
They also include `vokra/wavtokenizer-large` and
`vokra/wavtokenizer-large-speech-75token`, whose GGUF payloads are
byte-identical, plus `vokra/neucodec` and `vokra/distill-neucodec`.
Pyannote Segmentation 3.0 and RMVPE are deliberately
omitted from this list (see below). Each listed repository still needs its own
public-artifact load and real-weight parity verdict; sharing an architecture
does not turn one checkpoint's pass into a sibling pass.

There are no CPU-complete, Metal-unsupported repositories in this inventory.
The remaining 106 Metal-blocked repositories first need a complete released-
artifact CPU runtime; they are not counted as Metal implementations merely
because a converter or partial binder exists.

The routed-partial set deliberately includes `csm`, `nsnet2`,
`pyannote-segmentation`, `rmvpe` and `sbv2`. CSM still constructs synthesized
model weights in its public GGUF loader, and SBV2's public conversion does not
satisfy the strict runtime tensor-name contract. NSNet2's 2026-08-03 Hub
artifact predates its strict metadata and complete tensor contract. Pyannote's
real forward is disabled by default pending independent parity; RMVPE omits
the released U-Net decoder skip-concat and explicitly warns that real values
diverge. All five have substantial code, but none is a
released-artifact-complete CPU runtime; counting them as complete would hide
the actual blocker.

## 2026-08-24 implementation wave

### Wav2Vec2 CTC, Data2Vec Audio and HuBERT

The shared self-supervised speech-encoder surface now has strict native CPU
binders and explicit Metal dispatch for Conv1D, GEMM, Softmax and LayerNorm.
Seven public Wav2Vec2 CTC repositories, Data2Vec Audio Base and HuBERT Large
LS960 are classified as complete code routes. Data2Vec keeps its distinct
five-layer positional-convolution topology instead of being treated as a
byte-compatible Wav2Vec2 alias. The public `mms-1b-all-base` file remains
partial: its 8.9 MB payload is an adapter fragment, not the 1B backbone, and
is rejected with an explicit artifact error.

Independent reference parity on VAST produced:

| Model / surface | max abs | mean abs | Discrete output |
|---|---:|---:|---|
| Wav2Vec2 Base encoder | `5.701e-4` | `1.705e-5` | — |
| Wav2Vec2 Base logits | `6.142e-3` | `2.869e-4` | exact tokens/text |
| Data2Vec Audio encoder | `8.492e-4` | `2.217e-6` | — |
| Data2Vec Audio logits | `1.656e-3` | `3.805e-5` | exact tokens/text |
| HuBERT Large encoder | `2.693e-4` | `1.135e-5` | — |
| HuBERT Large logits | `1.964e-3` | `8.498e-5` | exact tokens/text |

The public Wav2Vec2 Large 960h LV60 Self artifact also loaded through the
strict binder and emitted `SO MY FELLOW AMERICANS` in the CLI CPU run. These
results establish native CPU/reference behavior. The Metal route is complete
in code, but this wave has not yet recorded an Apple-device execution for
these newly completed encoders; that distinction is kept explicit below.

### CAM++ public speaker encoder

The current `vokra/campplus-speaker-encoder` revision
`7963b5f8e21a75d900e4fc4d6b342ba3f989f6f9` carries one 27,684,640-byte,
619-tensor `campplus.gguf` with SHA-256
`c760971dc698fe7bfc5b9af9a4ba3b1ed1668b6c7a4e19086836b41e14285bea`.
The cached file used below matched the live fixed-revision LFS size and digest.

The pre-existing independent ONNX Runtime campaign used the official Alibaba
distribution of `iic/speech_campplus`, with a separately implemented
torchaudio frontend. Across three real WAVs, the native CPU embeddings had
component max-abs at most `1.87e-5`, cosine at least `0.99999999998`, and the
same speaker-similarity ranking to six decimals. The committed intermediate
fixture independently covers all seven recorded graph surfaces at the
registered `0.01` bound.

The follow-up at commit `bcff8c1f` used the exact public GGUF and one
93,680-sample LibriSpeech utterance. The CLI exported the complete native
embedding on CPU and real Apple M1 Metal:

| Dimension | Different values | max abs CPU/Metal | mean abs | relative L1 | cosine |
|---:|---:|---:|---:|---:|---:|
| 192 | 180 | `1.668930054e-6` | `4.522735253e-7` | `5.783065861e-7` | `1.000000000` |

The complete embedding passed the unchanged `0.01` FP32 bound. A Seatbelt
probe returned the explicit `no system default Metal device` error rather
than falling back; the real device run then completed outside that restriction
after macOS reported Apple M1 Metal support. The real-file CLI output gate
reported `1 passed; 0 failed`. The 12-file evidence package is at
`/private/tmp/vokra-campplus-mac-bcff8c1f`; its `SHA256SUMS` digest is
`dcb17a401418e4b20b313b75459678bb77bb74fd586e826c7539d4219c189523`.
No public upload or replacement was performed.

### SepFormer seven-checkpoint family

SepFormer implements the full learned forward: encoder, dedicated one-group
GroupNorm, segmentation/overlap-add, two dual-path blocks with 16 total
Transformer layers, multi-head attention, ReLU feed-forward networks, PReLU
mask heads and decoder. Its complete backend hot-op set is GEMM, Softmax,
LayerNorm, GroupNorm and Conv1D. CLI and Rust sessions expose source
separation, and the C ABI adds
`vokra_separate`; the returned allocation is stream-major and is released
once with the existing `vokra_audio_free` function.

All seven public repositories strict-bound their complete 417-tensor payloads:
WSJ0-2mix, Libri2Mix, Libri3Mix, WHAM 16 kHz enhancement, WHAMR 16 kHz,
WHAMR 8 kHz, and DNS-4 16 kHz enhancement. The initial WSJ artifact omitted
`vokra.sepformer.n_out`; the two WHAMR artifacts were incorrectly stamped as
one-output enhancement models. Compatibility repair is accepted only for the
exact known variant/provenance plus audited `[512,256,1,1]` mask-head shape.
WHAMR is treated as two-speaker separation (`category=separation`, `n_out=2`),
while WHAM 16 kHz and DNS-4 remain one-output enhancement models.

The independent oracle now covers all seven checkpoints with the official
`speechbrain==1.0.3` implementation at each fixed upstream revision. The model
and deterministic 4,096-sample input execute in float64 before little-endian
f32 serialization; the dumper never imports Vokra or reads a GGUF. Six variants
retain the established waveform boundary `max_abs <= 0.01`,
`mean_abs <= 0.001`. DNS4 is separately pre-registered at `0.1513 / 0.00515`:
the pinned official FP32 output itself differed across Apple ARM and VAST x86
by max `0.0972996`, mean `0.00243752`, and the bound is twice the largest
observed official FP32-to-FP64 floor. The full derivation and SHA-256 ledger are
in `crates/vokra-models/tests/fixtures/sepformer/README.md`.

The Apple M1 campaign ran the exact seven public GGUFs through CPU, real Metal
and the official FP64 fixtures. Largest waveform errors per variant were:

| Variant | CPU / official max | Metal / official max | CPU / Metal max |
|---|---:|---:|---:|
| WSJ02Mix | `9.1076e-5` | `1.6022e-4` | `7.1526e-5` |
| Libri2Mix | `5.2452e-5` | `8.2612e-5` | `3.8028e-5` |
| Libri3Mix | `5.7220e-6` | `5.7220e-6` | `4.7684e-6` |
| WHAM16k enhancement | `1.3560e-6` | `1.3635e-6` | `9.9838e-7` |
| WHAMR16k | `2.4587e-7` | `6.7800e-7` | `4.3213e-7` |
| WHAMR 8 kHz | `3.3248e-7` | `3.5577e-7` | `4.0419e-7` |
| DNS4 enhancement | `8.8398e-2` | `1.1426e-1` | `6.3866e-2` |

Every encoder comparison remained at or below `2.3842e-6`. Metal GroupNorm
matched CPU over SepFormer production shapes at max `4.768e-7`; the corrected
bias-first Metal GEMM matched CPU/naive references at max `6.676e-6`. Metal is
the selected backend for every learned op and never falls back to CPU.

Commit `f02a7556` was then checked on disposable VAST instance `48639251`.
`parity_sepformer_real` passed `4 / 0 / 0` in 274.38 seconds, including all
seven strict binds and all seven CPU/official-FP64 forwards. DNS4 x86 CPU
measured max `0.12589264`, mean `0.00357618`; all other x86 rows remained far
inside the shared boundary. The command
`cargo clippy -p vokra-models --all-targets -- -D warnings` exited zero. No
upload occurred, no VAST-generated artifact required
pullback, and the instance was destroyed; the live VAST inventory returned
`[]`.

### Standalone BERT / DeBERTa CPU and Metal front door

`BertRuntime` and `vokra-cli run --token-ids` now expose the three public
standalone text encoders as raw final-hidden features on CPU and Metal. Input
is an explicit comma-separated `u32` token-id sequence; raw text, audio-shaped
input and empty/out-of-vocabulary sequences fail before forward. Bench refuses
to invent an audio real-time factor for this non-audio task. CUDA, QNN and any
other unimplemented backend remain explicit unsupported-operation errors.

The scalar CPU forwards remain unchanged as the comparison path. A backend
seam sends every learned transformer hot operation through the selected
`Compute` implementation: GEMM, Softmax, LayerNorm and GELU for all three
architectures, plus Conv1D for DeBERTa v2/v3. Relative-position bucket lookup,
residual addition and layout transforms remain host control/pointwise glue;
they do not invoke a hidden CPU model forward.

The exact fixed-revision public files verified on disposable VAST instance
`48640495` were:

| Public repository | Revision | Bytes | GGUF SHA-256 |
|---|---|---:|---|
| `vokra/chinese-roberta-wwm-ext-large` | `42201d07523983914c683d431c2ecce7d88ecf6f` | 1,298,182,944 | `a1a1df298fedb585b5278a2c048c5a11515968e2fdf43b856354f964c3e89b59` |
| `vokra/deberta-v2-large-japanese-char-wwm` | `fb75652c6bbc2daf39fb1089079ea75a68d9597f` | 1,551,428,992 | `c74f2b6594e5837e5fe49318ec4a2e13bed76e32529e761c21260528be8aea1a` |
| `vokra/deberta-v3-large` | `a36bdb29209e214b81dee9a3c80484320aa4d66a` | 873,573,120 | `b3c07c6f91ec36bfd556dd511e6c9471ccc417b95a0d75bd605d26feb4666b0d` |

All three completed two-token CPU forwards with `hidden=1024`. The BERT
artifact took 1.65 seconds, DeBERTa v2 took 24.86 seconds, and DeBERTa v3 took
25.38 seconds on the rented x86 host. The DeBERTa v3 file exposed an early
public layout with verbatim Hugging Face `deberta.*` tensor names rather than
the current converter's canonical `bert.*` layout. The loader now detects the
two schemas without ambiguous precedence, performs the converter-equivalent
Q/K sharing and shared-relative-embedding LayerNorm for that exact legacy
layout, and rejects mixed or incomplete schemas. A differential fixture proves
the legacy and canonical paths agree within `1e-6`; the existing independent
real-weight final-hidden parity test retained its original `6e-3` bound and
passed unchanged.

The exact same public files then completed two-token final-hidden forwards on
the maintainer's Apple M1 CPU and GPU. All 2,048 output values were compared:

| Public model | CPU time | Metal time | max abs CPU/Metal | mean abs | relative L1 | cosine |
|---|---:|---:|---:|---:|---:|---:|
| Chinese RoBERTa | 0.88 s | 2.56 s | `3.814697266e-6` | `4.656343719e-7` | `1.231983967e-6` | `1.000000000` |
| Japanese DeBERTa v2 | 23.08 s | 3.58 s | `3.406405449e-5` | `7.116260662e-6` | `1.219052028e-5` | `0.999999999925` |
| DeBERTa v3 | 23.92 s | 3.84 s | `2.861022949e-6` | `4.208982887e-7` | `8.738921501e-7` | `1.000000000` |

Every row passed the pre-registered untuned final-hidden limits
`max_abs <= 5e-4` and `mean_abs <= 1e-4`; every negative-control CPU/Metal
comparison contained a non-zero difference. A sandboxed Metal probe first
returned `no system default Metal device`, while the same command on the real
Apple device succeeded, demonstrating that the route does not silently fall
back to CPU. The fixed-revision GGUFs were removed locally after their hashes
and outputs were recorded.

Commits `52eaccd9`, `b9b24ea3` and `7f766b26` were checked with the local
`vokra-bert` suite and warnings-as-errors clippy where locally safe. VAST then
passed the focused `vokra-models` runtime tests and
`cargo clippy -p vokra-models --all-targets -- -D warnings`. No upload occurred
and no unique remote artifact needed pullback: the source was already local,
while all three large files were reproducible public SHA-matched inputs. The
instance was destroyed rather than stopped, and the live VAST inventory
returned `[]`.

The Metal follow-up at commit `b17b3d90` passed the local 21-test
`vokra-bert` suite, the Metal-enabled CLI regression build, real Apple M1
execution for all three public artifacts, and Metal-enabled CLI Clippy with
warnings denied. Disposable VAST instance `48644299` then passed the focused
`vokra-models` BERT runtime tests (`2 passed / 0 failed`) and
`cargo clippy -p vokra-models --all-targets -- -D warnings`. The logs were
pulled to `/private/tmp/vokra-bert-metal-vast-48644299.DJoSRg`; their SHA-256
values are respectively
`831ef5b87e48ef4a6d8f966787fcabb02a12dad2fdcbec7f7d58c248cb764d5d`
and
`d79a0d317c988a7045eceec109da5e3565d76b8db840ac3ee2372e578aa9479d`.
The first candidate contract (`48644242`) never acquired compute resources and
was destroyed before any data transfer. Instance `48644299` was destroyed
after log verification; the paginated live inventory then reported zero
instances and zero labels.

### Conv-TasNet Libri1Mix enhancement

The source-tree route is now complete: the strict binder accepts exactly 345
official Asteroid tensors (5,000,881 parameters), validates the corrected
32-sample encoder/decoder kernel and 16-sample stride, and runs all 24 dilated
TCN blocks. Conv1D, depthwise grouped Conv1D and flattened Global LayerNorm use
the common CPU/Metal `Compute` seam; PReLU/ReLU, residual addition, masking and
layout work are host pointwise/control glue. Rust `SeparationEngine`, CLI
run/bench and the existing C `vokra_separate` surface all route this arch.

The independent Asteroid 0.7.0 oracle at revision
`bb8a876bc157b5cf3c405994accb798c49146016` produced the following VAST CPU
comparison for a deterministic 4,096-sample input:

| Conv-TasNet surface | max abs | mean abs | relative L1 |
|---|---:|---:|---:|
| Encoder | `4.768371582e-7` | `2.630132778e-10` | `1.379561243e-9` |
| Bottleneck | `5.512237549e-4` | `3.821378050e-5` | `4.431215712e-5` |
| Mask | `2.495117188e-1` | `1.139879413e-2` | `1.775571873e-4` |
| Final waveform | `3.155212402e-1` | `4.500496760e-2` | `3.302495461e-4` |

The larger absolute mask/waveform values reflect the checkpoint's large
intermediate/output scale; relative-L1 remains below `3.31e-4`. The corrected
GGUF also passed a real Session/C-ABI call and CLI emitted a 32,000-sample
16-kHz WAV on VAST.

The public `vokra/conv-tasnet-libri1mix` artifact is still classified partial,
so the headline counts above do not increase. It carries the obsolete
kernel=16/stride=8 topology and CC-BY-SA-3.0 provenance, while the pinned
checkpoint is kernel=32/stride=16 and its upstream card conflicts among
CC-BY-SA-4.0, CC-BY-SA-3.0 and WHAM-derived CC-BY-NC-4.0 Research-only terms.
The corrected converter defaults to `LicenseClass::Unknown`; replacement and
publication stay fail-closed. Code-level Metal routing is complete, but no
Apple-device execution has been recorded for Conv-TasNet in this wave.

### SpeechBrain X-vector two-artifact family

Both public repositories are now executable through one strict native
speaker-embedding runtime:

| Public repository / layout | Revision | GGUF SHA-256 | Tensors |
|---|---|---|---:|
| `vokra/xvector-voxceleb` / bare embedding | `00e4e360a5c1d7c6d0d6cbbc209fe7d49e25c4f3` | `1422a264dd8ee9367f1cbb0b59240e7a6048b24f2587eb68c79b228909318cdc` | 32 |
| `vokra/xvector` / combined classifier | `9b0530f8e71603ee303f9f7534b79ecfc8f8696f` | `a1aaad3efe781a45683cdec08bc0b4d5c7618f613ed3241be0d7230e6b28981a` | 46 |

The runtime checks the model identity, provenance, exact tensor count and
every inference tensor shape. The combined file's classifier and global
normalizer tensors are validated but intentionally not run by the embedding
surface. The learned path is five reflect-padded/dilated TDNN convolutions,
eval BatchNorm, LeakyReLU, Bessel-corrected statistics pooling and the final
512-dimensional projection. All learned operations declare `Conv1d`; CPU and
Metal therefore share one complete backend route, and an unavailable backend
is an explicit error.

The independent oracle uses SpeechBrain 1.0.3, upstream revision
`56895a2df401be4150a159f3a1c653f00051d477`, and official checkpoint SHA-256
`9d96cafa0ede1a84799b67dc9b5645f31f5b7d094e7e4775e5d5c12547883a93`.
It never reads a Vokra GGUF. VAST instance `48553051` measured the same result
for both public layouts and for a fresh GGUF produced by the strict converter:

| X-vector surface | max abs | mean abs | relative L1 | cosine |
|---|---:|---:|---:|---:|
| 24-bin SpeechBrain fbank | `9.517669678e-4` | `2.333762859e-5` | `5.480862455e-6` | `1.000000477` |
| 512-d embedding | `2.414703369e-3` | `5.878672237e-4` | `8.009463636e-5` | `0.9999998212` |

SpeechBrain's official statistics pool adds tiny random mean dither even in
evaluation mode. The fixture pins that upstream seed; Vokra omits the dither
to make repeated native inference deterministic, and the measured difference
is covered by the committed narrow bounds. The CLI emitted two 512-d vectors
and cosine `1.000000` for identical WAV inputs. The existing
`vokra_speaker_embed` C API reported the required dimension as 512 and matched
the Rust output bit-for-bit. A non-16-kHz input remains a loud error.

The raw official `.ckpt` has 37 state entries: 32 floating inference tensors
plus five integer BatchNorm training counters. The offline preparation tool
removes only those counters; the Rust converter then validates all 32 names
and shapes and stamps the upstream revision, exact frontend, TDNN, pooling,
padding and artifact-layout metadata. Linux VAST could not execute Metal, so
the initial CPU wave left the Apple-device comparison as a separate gate even
though the complete code route was counted.

That gate was closed at commit `20f56ab3`. The public CLI now optionally
writes a speaker encoder's native embedding as raw little-endian F32, allowing
the complete vector rather than only its norm or a self-cosine to be audited.
The two unchanged public GGUFs processed the same 16,000-sample input through
CPU and real Apple M1 Metal:

| Public X-vector artifact | Dimension | Different values | max abs CPU/Metal | relative L1 | cosine |
|---|---:|---:|---:|---:|---:|
| `vokra/xvector-voxceleb` | 512 | 388 | `4.768371582e-6` | `1.102813177e-7` | `1.000000000` |
| `vokra/xvector` | 512 | 388 | `4.768371582e-6` | `1.102813177e-7` | `1.000000000` |

The two public layouts were bit-identical to each other on CPU and again on
Metal. Both passed the unchanged `0.01` FP32 bound. A Seatbelt probe returned
the explicit `no system default Metal device` error rather than falling back;
the real device runs then completed outside that restriction after macOS
reported Apple M1 Metal support. The real-file CLI output gate passed once for
each public GGUF. The 17-file evidence ledger is at
`/private/tmp/vokra-xvector-mac-20f56ab3`; its `SHA256SUMS` digest is
`680d66f460eeb58c49df507b35857a7e6fe9c501c80888052031fedc8dc01ce9`.
No public upload or replacement was performed.

### SpeechBrain ECAPA-TDNN and related public artifacts

The canonical `vokra/ecapa-tdnn` repository now has a strict native speaker
embedding runtime. Its live revision is
`24be4349d49c23bb3b80b5afccf37538e8d616b4`; `model.gguf` is 83,239,808 bytes,
has SHA-256
`207cebb8ee3da5e306b05d782215411954a2d8ca76ecd9d32b7ec52ffaaa5fc3`, and
contains the exact 200-tensor SpeechBrain inference manifest. The learned
forward implements the three scale-8 SE-Res2Net blocks, multi-layer feature
aggregation, global-context attentive statistics pooling and the final
192-dimensional embedding projection. Conv1D and Softmax use the common
CPU/Metal compute seam; an unavailable backend is an explicit error.

The independent oracle uses SpeechBrain 1.0.3, pinned upstream revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, and the official checkpoint. The
preparation step removed only 31 integer BatchNorm training counters and
produced a 200-tensor safetensors file with SHA-256
`62f680eda09178c65f0b68793c5bfc04d057f637ae67df642966c534571c9774`.
The strict converter produced an 83,240,544-byte GGUF with 24 metadata keys.
Both that fresh file and the canonical public file produced the same VAST CPU
parity result:

| ECAPA surface | max abs | mean abs | relative L1 | cosine |
|---|---:|---:|---:|---:|
| 80-bin SpeechBrain fbank | `6.217956543e-4` | `2.224215677e-5` | `2.086387440e-6` | `0.9999997616` |
| 192-d embedding / end-to-end | `2.670288086e-4` | `7.719127461e-5` | `4.414713658e-6` | `0.9999998212` |

The CLI produced a 192-dimensional vector, and the C API reported dimension
192 and matched the Rust result bit-for-bit. The following related Hub files
remain fail-closed rather than being counted as runnable:

| Public repository | Live artifact verdict |
|---|---|
| `vokra/speechbrain-spkrec-ecapa-voxceleb` at `3dc7704b2dcb80b8ea8eb2d3db7280f682ac3657` | The 83,239,904-byte `spkrec-ecapa-voxceleb.restamped.gguf` (SHA-256 `75e74d4e41d16bf2af5a0176c189fc1c7f7597fe66aae47cacef17343cbb4c01`) fails GGUF parsing with tensor data out of bounds at `mfa.conv.conv.weight`; it requires a gated replacement. |
| `vokra/voice-gender-classifier` | Mis-stamped as canonical ECAPA, but carries a distinct 202-tensor `conv1/layer1-3/attention/fc6/fc7` classifier topology and incorrect provenance; the 200-tensor binder rejects it. |
| `vokra/lang-id-voxlingua107` | Carries the same 200 embedding tensors but no language classifier head or 107-entry label map; the shared ECAPA trunk is implemented, while classification remains an explicit artifact error pending a correctly licensed, gated replacement. |

The SHA-256 above is the value on the live fixed-revision model card and the
downloaded 83,239,808-byte file; it corrects a stale ledger transcription that
started with the same eight characters but did not match the artifact.

Linux VAST could not execute Metal, so the initial CPU wave left the Apple
device gate open. The follow-up at commit `13dc415c` ran the canonical public
GGUF against the pinned upstream example WAV on CPU and real Apple M1 Metal:

| Samples | Dimension | Different values | max abs CPU/Metal | mean abs | relative L1 | cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 52,173 | 192 | 190 | `8.010864258e-5` | `2.169384000e-5` | `1.240711873e-6` | `1.000000000` |

The complete embedding passed the unchanged `0.01` FP32 bound. A Seatbelt
probe returned the explicit `no system default Metal device` error rather
than falling back; the real device run then completed outside that restriction
after macOS reported Apple M1 Metal support. The real-file CLI output gate
reported `1 passed; 0 failed`. The 12-file evidence package is at
`/private/tmp/vokra-ecapa-mac-13dc415c`; its `SHA256SUMS` digest is
`1f2320cb86e45027852360605191723f8eb5c0d26311a671528e2eca162da6bb`.
This closes Metal for the canonical artifact only; the three related files
retain the explicit blockers in the table above. No public upload or
replacement was performed.

### WeSpeaker ResNet34-LM two-artifact family

The `wespeaker` arch now implements the complete official inference chain:
80-bin Hamming-window Kaldi fbank with global CMN, the `[3,4,6,3]` basic-block
ResNet, time-axis mean plus Bessel-corrected standard deviation pooling, and
the 256-dimensional `seg_1` projection. The runtime accepts exactly two tensor
contracts: 182 `resnet.*` embedding tensors or 219 bare tensors containing the
same backbone plus 36 training counters and the unused 17,982-way LM
classifier. All 36 learned Conv2D operations and the projection lower to the
common GEMM dispatcher, so CPU and Metal use the same graph and unavailable
backends fail explicitly.

| Public repository | Revision | GGUF SHA-256 | Tensors | Verdict |
|---|---|---|---:|---|
| `vokra/pyannote-wespeaker-voxceleb-resnet34-lm` | `8e27acd8a875088f1a7321f40610397bf964a446` | `6dccbc026e9c32a8f99f3441e64f1ff52e36afb055442595c86cda8021c78c39` | 182 | strict CPU bind and parity pass; complete CPU/Metal code route |
| `vokra/wespeaker` | `a20ec15a61be1b5c5cb0f4805dbf72bb341e946f` | `d2dd9114179e28d14bd7c6ec372807823f1064c4f6cdc2349a83aa652635553d` | 219 | topology supported, but public metadata mislabels CC-BY-4.0 weights as `apache-2.0` / permissive and omits attribution; strict provenance gate rejects it pending authorized replacement |

The independent oracle imports WeSpeaker source revision
`45941e7cba2c3ea99e232d02bedf617fc71b0dad`, loads the official checkpoint at
HF revision `f0c48c298fd835726c27956a5d617bad7115627e` (SHA-256
`9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449`),
and never reads a Vokra GGUF. VAST instance `48565792` measured:

| WeSpeaker surface | max abs | mean abs | relative L1 | cosine |
|---|---:|---:|---:|---:|
| 80-bin Kaldi fbank | `1.580715179e-4` | `3.359085895e-6` | `3.944553555e-6` | `1.000000119` |
| 256-d embedding / end-to-end | `1.337379217e-6` | `3.573832146e-7` | `8.788952982e-6` | `1.000000119` |

The strict converter now writes the exact checkpoint/source revisions,
frontend, block/channel schedule, TSTP and layout contracts and refuses a
license override incompatible with the audited CC-BY-4.0 checkpoint. Linux
VAST could not execute Metal, so the initial CPU wave left the Apple-device
comparison as a separate artifact gate.

The follow-up at commit `4244e31d` ran the unchanged canonical 182-tensor GGUF
on CPU and real Apple M1 Metal with one identical 16,000-sample input:

| Dimension | Different values | max abs CPU/Metal | mean abs | relative L1 | cosine |
|---:|---:|---:|---:|---:|---:|
| 256 | 193 | `2.980232239e-8` | `4.696630640e-9` | `1.204850094e-7` | `1.000000000` |

The complete embedding passed the unchanged `0.01` FP32 bound. A Seatbelt
probe returned the explicit `no system default Metal device` error rather
than falling back; the real device run then completed outside that restriction
after macOS reported Apple M1 Metal support. The real-file CLI output gate
reported `1 passed; 0 failed`. The 12-file evidence package is at
`/private/tmp/vokra-wespeaker-mac-4244e31d`; its `SHA256SUMS` digest is
`6a6f3ea000770d459fe73bd4e7ad9e970427057c21c594312274d05ebf54e391`.
This closes Metal only for the canonical pyannote artifact. The 219-tensor
`vokra/wespeaker` file remains rejected by its provenance gate and is not
claimed as a GPU pass. No public upload or replacement was performed.

### NVIDIA TitaNet-Large two-artifact family

Both public TitaNet-Large repositories carry the same 101,574,272-byte GGUF
(SHA-256
`17388234419f1208a39adc6c19faadcfde918848da663b6f008c8fc3e7f71f85`)
with the exact 108 floating inference tensors from NVIDIA's pinned NeMo
checkpoint. Their audited live revisions begin `f72b242` for
`vokra/titanet-l` and `a8cbe31` for `vokra/titanet-large`; byte identity means
one strict real-file result applies to both artifacts without inferring sibling
topology.

The native runtime implements the NeMo 1.10 frontend and complete TitaNet-L
forward: 512-point reflect-centred STFT, 80-bin mel log-power normalization,
five Jasper depthwise-separable Conv1D/SE blocks, attentive statistics pooling,
and the 192-dimensional decoder projection. Conv1D, grouped Conv1D, GEMV and
Softmax all pass through the common `Compute` seam. CPU and Metal therefore
run the complete learned graph; an uncovered backend is rejected explicitly.

The independent reference restores
`nvidia/speakerverification_en_titanet_large` revision
`0dc382f40121a5fbd34db10a2bb04d826c2be6a8` (checkpoint SHA-256
`e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3`)
and pins the checkpoint-era NeMo v1.10.0 frontend semantics instead of using
the changed NeMo 3 STFT padding/sequence-length defaults. VAST measured:

| TitaNet surface | max abs | relative L1 | cosine |
|---|---:|---:|---:|
| 80-bin frontend | `1.907348633e-5` | `3.124698196e-6` | `1.000000119` |
| 192-d embedding / end-to-end | `7.338821888e-7` | `8.251741747e-6` | `1.000000119` |

VAST instance `48569784` also passed the focused model-library, converter,
CLI, C-ABI and selected-backend tests, followed by
`cargo test --workspace --quiet` and
`cargo clippy --workspace --all-targets -- -D warnings` with zero code
failures. It was destroyed after verification.

The feature-gated real-file test was then run on the maintainer Apple host
against the public GGUF. The Codex sandbox correctly surfaced that it could
not see a system Metal device; the exact same test outside that sandbox ran
the complete graph on Metal and passed against CPU with max-abs
`2.803280950e-7`, relative L1 `3.596126135e-6` and cosine `1.000000000`.
The committed Apple gate requires max-abs and relative L1 at most `1e-4` and
cosine at least `0.99999`. Because the two public GGUFs are byte-identical,
that one real-file result applies to both audited revisions.

The strict converter rejects missing, extra or shape-incompatible tensors,
pins both checkpoint and NeMo source revisions, and refuses a licence override
that contradicts the audited CC-BY-4.0 weights. Rust, CLI and the existing
model-generic C speaker API all route the model. No public upload or
replacement was performed.

### Descript DAC token-to-waveform runtime

The three public DAC repositories now have a strict native released
token-to-waveform route: 16 kHz binds 358 F32 tensors and 12 codebooks, 24 kHz
binds 558 tensors and 32 codebooks, and 44.1 kHz binds 328 tensors and 9
codebooks. The loader folds upstream weight normalization and rejects missing,
extra or wrong-shaped tensors. Factorized RVQ, every SEANet Conv1D /
ConvTranspose1D and every Snake activation are selected together through
`Compute`; an unsupported backend is an error and cannot fall back to CPU.

The CLI accepts raw time-major `[frames,n_codebooks]` little-endian `u32`
codes and emits mono WAV. Output length follows the released graphs exactly:
16/24 kHz emit `frames * 320 - 8`, while 44.1 kHz emits `frames * 512`.
Encoding is not implemented and returns an explicit error, so this is not
presented as a completed bidirectional codec or as support for upstream's
pickle-backed `.dac` container.

The independent 44.1 kHz fixture was produced by
`descript-audio-codec==1.0.0` through its public
`ResidualVectorQuantize.from_codes` and `DAC.decode` APIs. It pins official
release tag `0.0.1` `weights.pth` at SHA-256
`a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa`.
The public `vokra/dac-44khz` GGUF at revision
`20073ecbdd15b0826ebbde3dc6f00f463592a6fc` had SHA-256
`cab82af37f4751006d017bd1c49660053a14c8e5c615f56f10fddd0ae95e1592`.

| 44.1 kHz public artifact surface | max abs | mean abs | relative L1 | cosine |
|---|---:|---:|---:|---:|
| VAST CPU, official features through decoder | `8.940696716e-7` | `2.130511945e-7` | `9.884971632e-7` | `0.9999999404` |
| VAST CPU, official codes through RVQ + decoder | `1.013278961e-6` | `2.336270768e-7` | `1.083963525e-6` | `0.9999999404` |
| M1 Metal, official codes through RVQ + decoder | `1.013278961e-6` | `2.209130372e-7` | `1.024974133e-6` | `1.000000000` |

The committed gates were tightened after measurement to max-abs `2e-6`,
relative L1 `2.5e-6` and cosine at least `0.9999995`. Those bounds were not
widened for the sibling runs.

The 16/24 kHz follow-up uses its own independent official fixtures rather
than imputing the 44.1 kHz result. A dedicated Python 3.12 environment pins
`descript-audio-codec==1.0.0`, `torch==2.13.0` and `numpy==2.5.2`; the dumper
calls the public `ResidualVectorQuantize.from_codes` and `DAC.decode` APIs.
The 16 kHz fixture pins release tag `0.0.5` `weights_16khz.pth` at SHA-256
`95ab7176b67137d4d4c6c54b8d6ef3cea797faec228cb03ad084badcad570b4d`;
the 24 kHz fixture pins tag `0.0.4` `weights_24khz.pth` at SHA-256
`44bad592fc393e03eb0be7a5120b7d487fe9612fa41269dc03fca3d4b87e20ad`.

The unchanged public artifacts used by both VAST and Apple were:

| Variant | Public revision | GGUF SHA-256 | Bytes |
|---|---|---|---:|
| 16 kHz | `10e37fc86b57320d9f7339f3d2ee831af9655ac0` | `7e631db79a05ad8b083d39231ac655efb1f7824eb277aacf44c59e9b2754192f` | 297,568,448 |
| 24 kHz | `d4e6bbff62e70c99dfe8a08d3779cf243e104cf5` | `c2a603c74acacbe359b0a8a7b9e9bb03ddc4e0b9e95123b8e02c6b16ddd8d4f2` | 301,108,032 |

One complete official-code RVQ-plus-decoder frame produced 312 PCM samples
for each sibling. The numerical results were:

| Public artifact / backend vs official PCM | max abs | mean abs | relative L1 | cosine |
|---|---:|---:|---:|---:|
| 16 kHz VAST x86 CPU | `2.831220627e-7` | `6.683871590e-8` | `1.039660447e-6` | `0.9999999404` |
| 16 kHz M1 CPU | `2.831220627e-7` | `6.708078451e-8` | `1.043425419e-6` | `1.0000000000` |
| 16 kHz M1 Metal | `3.650784492e-7` | `8.340243650e-8` | `1.297304778e-6` | `1.0000000000` |
| 24 kHz VAST x86 CPU | `9.238719940e-7` | `1.238024652e-7` | `8.231455695e-7` | `0.9999998212` |
| 24 kHz M1 CPU | `7.748603821e-7` | `1.206719436e-7` | `8.023315547e-7` | `1.0000000000` |
| 24 kHz M1 Metal | `8.344650269e-7` | `1.261479412e-7` | `8.387407279e-7` | `1.0000000000` |

Commit `a6d6e79e` records both fixtures and three env-gated public-artifact
tests. Disposable VAST instance `48625896` checked both fixed-revision GGUF
hashes, passed the focused test (`4 passed; 0 failed`), format and
`vokra-models` test-target Clippy with warnings denied. Its pulled log set is
under `/private/tmp/vokra-dac-vast-a6d6e79e/dac-evidence`; the locally
rechecked SHA-ledger digest is
`fc3a326c1500206bc7378bc91f28207d03827ff367238f4e8465ecf11da75656`.
The instance was then destroyed rather than stopped, and the Vast API returned
`instances: null` for that ID.

The same fixed GGUFs and committed official codes were exercised through the
public CLI on the maintainer Apple M1. CPU and Metal produced the same sample
geometry and the direct official comparisons above; CPU/Metal max-abs was
`1.415610313e-7` at 16 kHz and `3.576278687e-7` at 24 kHz. A sandboxed Metal
probe returned the explicit error `backend unavailable: no system default
Metal device`; it did not fall back. The Mac log/WAV/environment set is under
`/private/tmp/vokra-dac-siblings-mac-a6d6e79e`; its locally rechecked
SHA-ledger digest is
`dfee5811b0dfa07f82c9aa0accc5d28dec4e4148ec324eb9756ed08b42cc518e`.

Therefore all three public DAC GGUFs now have their own independent
official-reference CPU verdict and real Apple-device Metal verdict for the
complete token-to-waveform path. Encoding remains an explicit unsupported
operation. No Hub upload or artifact replacement was performed.

### SNAC bidirectional CPU and Metal decode

The two public SNAC repositories now bind their exact released graphs:
`vokra/snac-24khz` requires 269 F32 tensors and three hierarchical RVQ stages,
while `vokra/snac-44khz` requires 286 F32 tensors, four RVQ stages and both
32-frame local-attention blocks. CPU implements waveform encode, normalized
nearest-codebook search, hierarchical feature reconstruction and stochastic
waveform decode. Metal implements the complete token-to-waveform route,
including four-stage RVQ, grouped/dilated Conv1D, odd-stride ConvTranspose1D
`output_padding`, Snake, layer normalization, GEMM, softmax and local
attention. Metal encode returns an explicit unsupported-operation error at
the missing codebook-search kernel; it never performs a silent CPU search.

The independent oracle uses upstream `snac==1.2.1` source revision
`8f79a718f1ad71f94f79999f0071348227aff22e`. Its official 24 kHz checkpoint is
revision `d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec`, SHA-256
`4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40`;
the 44.1 kHz checkpoint is revision
`873ebef9718b89660340c6f55a2b515e98cfa1d9`, SHA-256
`b0a676cbdc8d1cc53186f6d777bc956fb7932ceacdc657a4c3741646e9e7ead0`.
The dumper calls the official encoder, quantizer and decoder. It captures the
four PyTorch NoiseBlock inputs so learned-graph parity is not confounded with
an invented cross-library RNG-seed equivalence.

The public Vokra GGUFs were fetched at revisions
`17fe131f825e82151ad87c511c49fef4b9209564` (24 kHz, SHA-256
`3c357841858fd24a05f4a7fb5b28daffa5931373aad3112f83a9d09d3575cc8a`)
and `3c7e4fad6b0b2ac34e65c674647d9914780cb90e` (44.1 kHz, SHA-256
`91ac3e7b6d1a3c332efe098b1b1773d32df98f14651cfc19d5f12ec147331e3a`).
Both strict binders accepted those unchanged public artifacts, and CPU encode
produced the exact official code sequence.

| Public SNAC surface | max abs | relative L1 | cosine |
|---|---:|---:|---:|
| 24 kHz CPU encoder | `7.820129395e-5` | `2.271287742e-6` | `1.000000000` |
| 24 kHz CPU RVQ decode | `9.536743164e-7` | `4.393151833e-8` | `1.000000000` |
| 24 kHz CPU decoder | `8.046627045e-7` | `2.054521656e-6` | `1.000000000` |
| 44.1 kHz CPU encoder + local attention | `3.101825714e-4` | `4.241102395e-6` | `1.000000000` |
| 44.1 kHz CPU RVQ decode | `3.814697266e-6` | `4.973113820e-8` | `1.000000000` |
| 44.1 kHz CPU decoder + local attention | `5.215406418e-7` | `8.614468708e-7` | `1.000000000` |

The measured CPU gates are max-abs `4e-4` / relative L1 `1e-5` for the
encoder, `5e-6` / `1e-7` for RVQ and `1.5e-6` / `4e-6` for the decoder, with
the corresponding cosine floors committed beside the fixtures. The initial
Linux VAST wave could not execute Metal, and host policy forbade a local
`vokra-models` real-weight Cargo run, so that wave left the Apple device gate
open rather than imputing it from CPU or primitive Metal tests.

The standalone route provided a policy-safe follow-up at commit `20408cc7`:
the two unchanged public GGUFs decoded their CPU-produced, GGUF-fingerprint-
pinned `VKRSNAC1` containers through both CPU and real Metal on an Apple M1.
Both routes produced equal geometry and finite IEEE-float mono WAVs:

| Variant | Samples | Different samples | max abs CPU/Metal | RMSE CPU/Metal |
|---|---:|---:|---:|---:|
| 24 kHz | 1,567 | 1,533 | `9.126961231e-7` | `1.728146126e-7` |
| 44.1 kHz | 5,003 | 4,690 | `3.129243851e-7` | `6.654532390e-8` |

The unchanged `0.01` FP32 bound passed for both public artifacts. A Seatbelt
probe returned the explicit `no system default Metal device` error instead of
falling back; the device runs then completed outside that restriction after
macOS reported Apple M1 Metal support. Focused Metal-feature CLI tests
reported `5 passed; 0 failed`. The 17-file evidence ledger is at
`/private/tmp/vokra-snac-mac-20408cc7`; its `SHA256SUMS` digest is
`1009a198d098417769bb26881a58e2037973d82892d34d109f2b850374708600`.
This closes public-artifact Metal **decode** for both checkpoints. Metal encode
remains explicitly unsupported at nearest-codebook search and is not claimed
as GPU-capable. No Hub upload or artifact replacement was performed.

### SNAC standalone CLI reachability

SNAC is now a routed `ModelTask::SnacCodec`, not a successful binder probe
that `vokra-cli run` still refuses. The CLI accepts CPU waveform encode and
CPU/Metal hierarchical decode through a versioned `VKRSNAC1` container. The
container records the original PCM length, padded base-frame extent, every
stage stride and length, the model topology, and a SHA-256 ledger over every
GGUF tensor name, dtype, shape and payload. A sibling variant or modified
checkpoint therefore fails before decode instead of producing plausible audio
with the wrong model. Metal encode reaches SNAC's explicit missing
nearest-codebook-search error and never performs a host fallback.

Both unchanged public GGUFs passed the real CLI route on VAST commit
`7df00dc4`: 24 kHz encoded 1,567 samples to four base frames / three stages
and decoded back to an IEEE-float mono WAV with exactly 1,567 frames; 44.1 kHz
encoded 5,003 samples to 32 base frames / four stages and decoded back to
exactly 5,003 frames. Decoding the 24 kHz container with the 44.1 kHz model
returned the explicit rate/stage/latent/hop/stride mismatch. The fixed public
GGUF SHA-256 values remained the two values recorded above; no upload or
replacement occurred.

The post-route live Hub audit reported 194 public repositories / 193 GGUF
repositories: CPU `full=72`, `partial=49`, `no-runtime-binder=72`,
`not-artifact=1`; Metal `full=72`, `blocked-by-cpu=121`, `not-artifact=1`.
Those historical counts describe code reachability plus the VAST CPU wave.
The Apple-device SNAC decode comparison was subsequently completed as
recorded above; it does not change the route-count classification.

### Piper Plus CLI/C ABI reachability

Piper already declared GEMM as its complete backend hot-op set and already had
a CPU/Metal real-weight parity harness, but `vokra-cli` and `vokra-capi`
classified the engine as CPU-only. Both loaders now pass the selected backend
to `PiperPlusTts`, and `TtsEngine::backend` / `Tts::backend` make that wiring
observable. A non-CPU selection is no longer rejected before it can reach the
existing Metal implementation.

The two unchanged public GGUFs exposed two independent compatibility gaps.
Both are legacy language-only voices and therefore do not carry the zero-shot
`spk_proj` group; the runtime now uses `emb_lang[lid]` directly and rejects a
caller-supplied speaker embedding explicitly. Their decoder heads also differ:
CSS10 is MB-iSTFT plus PQMF, while Mera is a standard three-stage HiFi-GAN
waveform decoder with a biasless post-convolution, final LeakyReLU alpha 0.01
and tanh. The loader requires exactly one complete head and rejects both/neither
instead of guessing or substituting a CPU path.

| Public repository | Revision | GGUF SHA-256 | Bytes |
|---|---|---|---:|
| `vokra/piper-plus-css10-ja-6lang` | `25e963b334ad77f89d77a07928e6dfdf7ff1fbf5` | `a3a17b9b020ef223efaa28d534271c3439774951c9486b95c5231b70cbe1340c` | 77,369,504 |
| `vokra/piper-plus-mera-multilingual` | `91837920abbe7bea8d81a2b843fdbdcf90aee16a` | `2b65e6b3f6a3052918edccae89e33591d76ef5239bb27cae767a6857ba3d5deb` | 76,904,480 |

The independent CPU oracles are the official upstream ONNX files, not a
Vokra-side mirror. CSS10 is pinned to
`ayousanz/piper-plus-css10-ja-6lang@bf70fae2e21f9670456ebb40e8df131f146f1821`
(`onnx_sha256=5ebc51dbf897238523f3df0d6e0f6c93033bc5cda3f8602a8379ebe2a4738c42`),
and Mera to
`kizuna-intelligence/piper-plus-mera-multilingual@d5ed57f04a1fd1cada7940ecd79a51d523b83462`
(`onnx_sha256=97952c17fd60d47c4db2a6bf3903184df2054b6fa97396c971102bc60a9f135f`).
Both fixtures were generated with onnxruntime 1.23.2 and record the exact config
hash in their manifest.

| Official ONNX CPU comparison | max abs |
|---|---:|
| CSS10 encoder `m_p` | `3.0e-6` |
| CSS10 duration / exact ceil | `3.0e-6` / exact |
| CSS10 reverse-flow latent | `1.1e-5` |
| CSS10 decoder and end-to-end PCM | `4.0e-6` |
| Mera encoder `m_p` | `1.311302e-6` |
| Mera reverse-flow latent | `5.245209e-5` |
| Mera waveform PCM | `2.003741e-6` |

Commit `41bfb570` passed both official fixtures, both public-file load/synthesis
tests, the legacy-speaker explicit-error test, decoder/conditioning unit tests,
CSS10 geometry and `vokra-models` test-target Clippy on disposable VAST
instance `48629407`. The logs are under
`/private/tmp/vokra-piper-vast-41bfb570/piper-evidence`; the instance was
destroyed, and the subsequent API inventory returned zero instances.

The same public files then synthesized the identical `aiueo` request on CPU
and real Apple M1 Metal through `vokra-cli`:

| Public artifact | Samples | Different samples | max abs CPU/Metal | RMSE | cosine |
|---|---:|---:|---:|---:|---:|
| CSS10 | 3,328 | 3,320 | `7.823109627e-8` | `1.204062987e-8` | `0.9999999999` |
| Mera | 9,728 | 1,061 | `3.576278687e-7` | `4.090683373e-8` | `1.0000000000` |

Both comparisons passed the unchanged `0.01` FP32 bound with equal sample
rate/geometry and finite PCM. A Seatbelt probe returned the explicit
`backend unavailable: no system default Metal device` error; it did not fall
back to CPU. The real device run used macOS 26.3, Apple M1 (8 GPU cores) and
Metal 4. The nine-file evidence package is at
`/private/tmp/vokra-piper-mac-41bfb570`; its verified `SHA256SUMS` digest is
`9ed247bf45acc7d8edf27d48771459bc8c0f713f6f5941385bffe43720d55641`.
No Hub upload or artifact replacement was performed.

### Moonshine composed attention

Moonshine already dispatched projections, softmax, normalization, GELU and
Conv1D, but Q/K and attention/value products were scalar host loops and a
front-door CPU-only check rejected Metal. Both matrix products now use the
shared GEMM dispatcher, causal masking is applied before the backend softmax,
and the declared hot-op set includes GEMM. CPU and Metal therefore execute the
same complete composed-attention route; an uncovered backend still fails at
`Compute::for_backend`.

The fixed independent CPU oracle remains the pinned Transformers fixture
recorded in `runtime-gap-execution-plan-2026-08-21.md`. A feature-gated
real-weight CPU/Metal test now compares encoder, decoder, all tied logits and
generated ids at the unchanged FP32 `atol = 0.01` GPU bound for both Tiny and
Base.

On this Mac, locally regenerated strict GGUFs produced identical CLI results
on CPU and real Metal for the committed 16 kHz fixture:

| Variant | Strict local GGUF SHA-256 | CPU | Metal |
|---|---|---|---|
| Tiny | `37d098b674f7583b4bbbc96f27a5481b55aa0d4c1f60fd5fd68eff2b13de4ab0` | `asr:` (empty) | `asr:` (empty) |
| Base | `e29adf5cac090771d971b8ac7a9a08c02105410fba2542ca7e8215ce947f388f` | `asr:` (empty) | `asr:` (empty) |

The empty transcription is expected for this VAD-oriented non-speech fixture;
the acceptance point here is exact CPU/Metal route agreement and successful
real-device execution. The intermediate-value test remains to be run in an
Apple runner that is allowed to execute the `vokra-models` package test scope.

### Public Moonshine artifact defect

The current Hub GGUFs are older compatibility artifacts, not the strict files
above. Their SHA-256 values are:

| Public repo | Current public GGUF SHA-256 | Runtime defect |
|---|---|---|
| `vokra/moonshine-tiny` | `77fc91fc5e22e46a1caa647e9a5ed6e3b9a5c7f2b665ad3a29bd1b61723dfe59` | Old minimal metadata; no embedded tokenizer/current Moonshine contract |
| `vokra/moonshine-base` | `b5ba78435e97cc6558cbaa5263ab20c0c929c3ff1da784a857214bab7ea97844` | Old minimal metadata; no embedded tokenizer/current Moonshine contract |

The Base artifact was exercised directly and failed on CPU at the first
missing strict key. Its older provenance also uses the exact redirected ids
`UsefulSensors/moonshine-{tiny,base}`; the loader now recognizes only those
two historical aliases, but does not pretend their missing topology/tokenizer
payload is complete. The correct repair is replacement with the regenerated
strict GGUFs through `scripts/publish/publish-one.sh`, followed by live hash,
CPU and Metal verification. Upload is a separate irreversible action and has
not been performed or authorized by this code wave.

### Whisper search paths and Whisper-Medusa module 0

The ordinary Whisper greedy path already preserved the selected backend, but
beam search and stochastic sampling constructed a CPU decoder after a Metal
encoder. That was an implicit backend substitution. Their logits sources and
scorers now receive the selected decoder backend, and the optional
cross-attention word-alignment pass uses the same per-op backend as the decoder
state. CPU remains the explicit decoder half of the declared CoreML hybrid
plan; Metal no longer enters that hybrid accidentally.

Whisper-Medusa-v1's required module-0 transform now dispatches its dense
projection through the same `Compute` backend and retains the official
`hidden + SiLU(linear(hidden))` semantics before tied logits. Metal currently
uses the correctness-first per-op Whisper path because the resident decoder
session fuses the vanilla projection and has no pre-projection adapter slot.
The feature-gated real-weight test compares full prefix logits and argmax
against CPU at the unchanged FP32 `atol = 0.01` bound. Its 6.25 GB artifact is
VAST-only under the repository safety rule, so that Apple-device test is not a
local verification claim. Accelerated modules 1–10 tree decoding remains a
separate explicit unsupported API and is not counted as ordinary module-0
runtime failure.

### Moshi CLI/C ABI reachability

Moshi's temporal transformer, depth transformer, Mimi encoder and Mimi neural
decoder already carried backend selectors and Metal parity tests. The CLI and
C ABI loaders nevertheless classified the complete duplex engine as CPU-only.
Both now apply `MoshiEngine::with_backend` after any real-Mimi side-car swap,
so the selected backend reaches all four neural components. The cheap paged
RVQ table gather remains documented host bookkeeping; it is not substituted
for a missing neural kernel. A feature-gated CLI test runs the full synthetic
duplex route on CPU and Metal and compares emitted PCM at `atol = 0.01` when a
Metal device is available. The public 7B artifact and real Mimi side-car still
require their separate large-artifact verification.

### NSNet2 learned forward and mask

The initial Metal seam exposed a deeper public-artifact defect. The pinned
official Microsoft graph is SHA-256
`88429b6253600be840ab816f46f466811d20078142fb12bff8cafe2b27bd4ca9`
(DNS-Challenge commit `8b87a33b2892f147b5c7ad39ea978453730db269`). It has 161 spectral bins,
`n_fft = win_length = 320`, numeric MatMul initializer names, and two ONNX GRUs
with `linear_before_reset = 1`. The old converter/runtime contract assumed 257
bins / `n_fft = 512`, passed initializer names through, and fused GRU input and
recurrent biases. Those are topology and arithmetic errors, not aliases.

The converter now accepts the exact official 14-F32-tensor manifest only,
renames numeric initializers, removes singleton direction axes, and transposes
MatMul weights into native row-major layout. The runtime now implements the
official frontend and synthesis: symmetric square-root Hann, 161-bin log10
power with a `1e-12` floor, right-hop padding, raw overlap-add without
window-sum normalization, and the `-80 dB` minimum gain. GRU input/recurrent
biases remain separate so the recurrent candidate bias stays inside the reset
gate. `DenoiseStreamHandle::finalize` also flushes the final padded frame and
overlap-add tail instead of shortening a streamed utterance.

NSNet2 stores an explicit backend selection. On Metal, all five dense layers
and both GRU cells dispatch their learned projections through GEMV, and the
real-valued gain mask is applied to the complex STFT through the Metal
denoise-mask kernel. ReLU, sigmoid, tanh and STFT/iSTFT remain host DSP/glue;
no missing learned projection runs on CPU. Whole-model coverage is checked
before the first frame.

`tools/parity/nsnet2_dump_reference.py` is an independent oracle: it executes
the pinned ONNX with `onnx.reference.ReferenceEvaluator` and separately
transcribes Microsoft's `featurelib.py`. On
`tests/parity/silero_vad/test_16k.wav` (24,576 input samples), all three routes
emitted 24,800 samples. The locally regenerated strict GGUF SHA-256 is
`6c6a1a45cffa2e9bf69515c1fcbc49aa53d1cb0841373cc7a2781efa8e66794a`.

| Route comparison | max abs | mean abs | RMSE |
|---|---:|---:|---:|
| CPU vs official ONNX | `3.61e-7` | `6.88e-9` | `2.76e-8` |
| Metal vs official ONNX | `3.61e-7` | `6.85e-9` | `2.77e-8` |
| CPU vs Metal | `3.73e-8` | — | `2.35e-9` |

The registered real-weight PCM bound is therefore tightened from the inherited
`0.01` placeholder to `5e-5` (>138× margin over the measured official-output
maximum). RNNoise remains a separate implementation: its signed-int8 sparse
matrix layout and rational activations are preserved exactly, with only the
expanded learned matrix-vector products dispatched through the backend.

The current public revision
`983e1cc1397810201f93a121a9daf60cf247813b` (GGUF SHA-256
`abeca882165909fb0897b39b97882d0ebd9f95cf176a4d2e58482e52a8b19e13`)
still fails at missing `vokra.nsnet2.n_bins` and carries the incompatible old
tensor/topology contract. NSNet2 therefore remains **routed partial in the
live Hub count** even though the corrected local artifact passes CPU, real
Metal, and official-ONNX parity. Moving the public repo to complete requires an
explicitly authorized replacement through `scripts/publish/publish-one.sh`.

### Vocos ConvNeXt / iSTFT vocoders

Vocos now stores an explicit backend selection and declares the complete
learned-op set: dense Conv1D, grouped/depthwise Conv1D, LayerNorm and GELU.
The new Metal grouped Conv1D path uses group-local shader indexing rather than
expanding depthwise kernels into dense diagonal weights. Host transposes,
residual/LayerScale arithmetic, magnitude/phase assembly and iSTFT remain
layout/DSP glue; no learned convolution or normalization silently executes on
CPU after Metal is selected. Unsupported backends fail the whole-model
coverage gate before inference.

The grouped-convolution kernel matched the CPU oracle over dense, grouped and
Vocos depthwise shapes with a measured maximum error of `1.192e-7`. Both
official variants were then checked using non-zero features emitted by
`tools/parity/vocos_dump_reference.py` and the pinned `vocos==0.1.0` runtime:

| Variant / route | samples | max abs vs official | RMSE vs official |
|---|---:|---:|---:|
| mel-24khz public GGUF / CPU | 1,024 | `1.68e-7` | `4.59e-8` |
| mel-24khz public GGUF / Metal | 1,024 | `2.30e-7` | `5.76e-8` |
| encodec-24khz corrected local GGUF / CPU | 1,600 | `5.57e-6` | `7.36e-7` |
| encodec-24khz corrected local GGUF / Metal | 1,600 | `9.10e-6` | `9.54e-7` |

All four are inside the pre-existing Vocos real-weight gate `atol = 1e-5`.
The public mel file at revision
`3fb388b9ed98406a1492a8f89aa12bf1927cc7d2` (GGUF SHA-256
`00586c971bb14d0b96aed1eebd2fc94637619fd718d52d78808d60cbc51116b9`)
loads and passes both backends directly.

The public Encodec filename at revision
`68cce2a1a7b624dd20db698ba3c63c6122205609` (GGUF SHA-256
`041504037c46eb24880a7d062a92a7a76e2dd40430aa27e75102a0b87b353c2f`)
is internally contradictory: its tensors have the Encodec axes
(`backbone.embed.weight = [384,128,7]`), while `vokra.model.name`,
`vokra.vocos.variant` and provenance all claim `vocos-mel-24khz`. The strict
loader correctly rejects it rather than guessing past false provenance. A
fresh conversion from the pinned official Encodec revision produced local
GGUF SHA-256
`23f55d6f0f2eca2e98d1ba103a947bc1e9a102389a4fc1f57740709d83f3447f`,
which is the artifact used for the passing CPU/Metal/reference rows above.
Replacing the public file requires a separately authorized gated publish.

### FSMN-VAD causal memory

FSMN-VAD now stores an explicit backend selection and declares GEMV plus
grouped Conv1D as its complete learned-op set. On Metal, the two input
affines, four block projections, four block expansions and two output affines
run as row-wise GEMV. Each learned `[128, 1, 20, 1]` causal-memory tensor runs
as a depthwise grouped Conv1D over the retained 19-frame history and current
chunk. Layout transposes, residual addition, ReLU, softmax, Kaldi fbank, CMVN
and history updates remain host preprocessing/control flow. A non-released
`lstride != 1` contract is rejected explicitly rather than executed on CPU.

The source-tree CPU route remains pinned to the independent FunASR fixture at
network max abs `8.344650269e-7` and PCM/stream max abs
`1.370906830e-6`. A new real-weight Apple test compares every posterior from
the CPU oracle and Metal at the unchanged FP32 GPU bound `0.01`; the local
package-test invocation was refused by the maintainer-machine heavy-scope
guard, so that exact all-posterior test remains an Apple-runner gate rather
than a claimed pass.

The strict locally regenerated GGUF was produced from official revision
`df20e6b30c653645fa4ff125cacfcabd1020a669` after all three registered source
hashes passed. Its SHA-256 is
`99df745c76316f6fe03d14545c7eb35cd45346237a4087ae3b5ada65790acada`.
The committed 16 kHz fixture ran end-to-end through the already-built CLI on
both CPU and real Metal with the same summary: 150 frames, 122 speech frames,
mean probability `0.6895`.

The current public revision
`ea6fe20a02b0b465023ad1ed080b9d92994f5d3d` (GGUF SHA-256
`a4aa69f08acaf39d6a97a4d1b7519ae873a1045a75d2dd8d1419617d87a8940d`)
is an older 1.7 MB artifact. It stores the ModelScope id in
`vokra.provenance.upstream_hf` and lacks the strict upstream revision/hash,
complete geometry and real CMVN keys. CPU correctly fails at the first
identity mismatch; accepting one alias would merely expose the next missing
contract. It requires gated replacement, not a permissive loader shim.

### Silero, FireRed and TEN VAD

Silero v5 now shares one recurrent arithmetic implementation between its
existing `no_std` CPU route and a backend adapter. All learned Conv1D, LSTM
input/recurrent projections and output affine operations reach the selected
backend; feature construction, sigmoid/tanh, state updates and stream framing
remain host-side control/DSP. FireRedVAD dispatches every released DFSMN dense
and memory projection through GEMV, and TEN-VAD does the same for its two-layer
GRU and output projection. Their frontend transforms, activations, cache/state
updates and final softmax remain host glue. Each architecture performs a
whole-model hot-op coverage check, so selecting an unsupported backend cannot
turn into an implicit CPU run.

The independent CPU references and their established tolerances remain the
numerical oracles. Feature-gated tests add real-weight CPU/Metal posterior
comparison at the repository-wide FP32 GPU gate `atol = 0.01`. Those exact
tests still require an Apple runner allowed to compile `vokra-models`; the
Linux VAST aggregate validates source behavior and all CPU tests but cannot
claim Apple GPU execution.

### RNNoise v0.2

RNNoise keeps the release's compressed float/int8/sparse matrices as the CPU
oracle. At load time it additionally expands each learned matrix into one
output-major FP32 backend view. A Metal frame quantizes activations on the host
with the same signed-int8 rule, dispatches the matrix-vector product through
GEMV, then applies the original scale, bias, diagonal term, rational
activation and recurrent state update on the host. One `Compute` context is
reused across all frames produced by a streaming push. This covers every
learned projection without redefining RNNoise's quantized arithmetic.

The public GGUF is not yet a valid real-artifact test: it predates the strict
`vokra.rnnoise.release_tarball_sha256` and canonical 36-array manifest. The
strict loader correctly refuses it. A regenerated GGUF plus independent Xiph
CPU and real Apple Metal parity is required before artifact completion; no
replacement upload is authorized by this branch.

### NKF-AEC

NKF-AEC now stores the selected backend and dispatches all learned complex
dense and GRU input/recurrent projections through GEMV. STFT, PReLU and gate
nonlinearities, complex Kalman filtering, overlap-add, and stream bookkeeping
remain host DSP/control flow. A single `Compute` and shared projection scratch
are reused for each available drain batch. This is a complete correctness
route, but its many small per-frequency-bin dispatches still need an Apple
performance measurement; no speedup is claimed.

No strict real NKF-AEC GGUF was available on the maintainer machine. The
feature-gated artifact test compares CPU and Metal output at `atol = 0.01`
when an explicit fixture path is supplied, so the remaining gate is artifact
acquisition plus execution on Apple hardware rather than a silent skip.

### HiFi-GAN and BigVGAN vocoders

The two HiFi-GAN repositories and all four BigVGAN repositories now carry an
explicit whole-model backend. HiFi-GAN lowers dilated and transposed
convolutions to the selected Conv1D backend without changing zero/reflect
padding semantics. BigVGAN routes every learned convolution and both periodic
Snake variants through the selected backend while alias-free resampling and
final waveform assembly remain host DSP. CPU keeps the existing scalar
generator as the oracle; unsupported backends fail the complete hot-op gate.

Focused synthetic CPU/Compute tests and the existing real-artifact binder
tests pass on VAST. Feature-gated Apple tests compare complete real-weight
waveforms at the registered FP32 bound; they are still Apple-runner gates, so
this ledger does not infer a Metal artifact pass from Linux source tests.

### FCPE, SmartTurn and Pyannote code routes

FCPE dispatches its complete learned Conv2D, grouped/depthwise Conv1D, dense,
normalization, attention and softmax path through one `Compute` context. The
frontend STFT/mel construction, tensor rearrangement and pitch decoding remain
host DSP/control. SmartTurn similarly routes the Wav2Vec2 feature convolutions,
positional convolution, all Transformer projections/attention, attention
pooling and classifier through the selected backend. Its padded first
GroupNorm statistics and query selection remain host glue.

Pyannote's SincNet, BiLSTM, linear stack and classifier now also have a
complete backend-dispatch implementation. It is **not** counted in the 63
CPU/Metal-complete repositories: the public API still requires
`VOKRA_PYANNET_ENABLE_FORWARD=1` because its real checkpoint has not passed the
independent upstream numeric gate. Adding a GPU route does not convert that
default loud-partial CPU posture into completion.

The current `vokra/smart-turn-v2` public GGUF is also not the canonical file
described by the strict runtime. It is 379,150,368 bytes, has 223 source-style
tensors including the unfused positional weight-norm pair and masked-spec
vector, and lacks `vokra.smart_turn.revision`. The canonical converter emits
221 tensors, folds weight norm, consumes the eval-unused vector and stamps all
pinned hashes. The public file therefore fails on CPU before compute and must
be regenerated and verified, not accepted through a permissive loader change.
No replacement upload is authorized by this branch.

### FCPE public artifact, official CPU parity and Apple Metal

The live `vokra/fcpe` revision
`93271c742e2f926e3ca84fb44b81606f1aa342cf` carries one 43,334,720-byte,
70-tensor `fcpe.gguf` with SHA-256
`6f5321ce0db16f9611ecd551204165817fdcc80efe85775508426f813535ce68`.
The fixed-revision download matched the model card's size and digest, but its
actual header contains zero of the fourteen required `vokra.f0.fcpe.*` axes.
The current strict loader therefore stops before inference with an actionable
error naming the first absent key (`vokra.f0.fcpe.d_model`) and instructing the
operator to reconvert artifacts produced before 2026-08-15. CPU and Metal have
the same loader verdict; no backend can silently guess the seven non-tensor
frontend/decode axes or fall back to CPU. The live card's claim that the full
hparam group is stamped is therefore stale relative to the bytes users
download.

The independent oracle is the official PyPI `torchfcpe==0.0.4` wheel
(SHA-256 `f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d`)
and its bundled `fcpe_c_v001.pt` (SHA-256
`b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015`).
`tools/parity/fcpe_dump_reference.py` imports that wheel and commits a
byte-reproducible 33-frame fixture for the official mel, sigmoid latent and
local-argmax F0 boundaries. It exposed four errors hidden by the former
finite-output smoke: the runtime used the wrong STFT padding, projected power
instead of magnitude, dropped the official final alignment frame, and stamped
the lower-level decoder's `0.05` default instead of the public waveform
wrapper's `0.006` threshold.

A current conversion stamps all fourteen axes and produces a 43,335,264-byte
strict GGUF with SHA-256
`3c17bd841e750d1cd0ab55bfe85eefd4ab1f158e91f4d9390f633c5c5fa12a9e`.
VAST instance `48630865` tested commit `4bb308ae` against that exact file:

| Official boundary | max abs | mean abs | RMSE |
|---|---:|---:|---:|
| log-mel | `5.890846252e-3` | `1.450321579e-4` | `4.932388547e-4` |
| sigmoid latent | `1.027882099e-4` | `3.072616437e-7` | `3.500768344e-6` |
| decoded F0 (Hz) | `8.697509766e-4` | `1.655347442e-4` | `2.678696765e-4` |

The largest log-mel difference is a floor-adjacent bin (reference `-11.3527`,
while `ln(1e-5) = -11.5129`), where log magnifies a tiny FFT reduction
difference. Signal-bearing bins above `-9` remain within `1.50681e-4`; the
test gates overall maximum, mean, active-band maximum and both downstream
surfaces independently instead of relaxing one global tolerance. The focused
suite reported `19 passed; 0 failed; 1 ignored`, and all-target package Clippy
with warnings denied passed. Logs are under
`/private/tmp/vokra-fcpe-vast-4bb308ae/fcpe-evidence`; the final combined log
SHA-256 is
`ee2db13c76a7c8446e2c84b10186cad530c1755cd0b476b3f802fbba1aeb5253`.

At commit `eba4ca07`, the exact same strict GGUF and committed input were run
through CPU and the real Apple M1 8-core GPU (macOS 26.3 build 25D125,
Metal 4). All 33 timestamps and voiced decisions were identical. Unrounded
CPU/Metal differences were F0 max/mean `1.525878906e-4` /
`3.560384115e-5` Hz and confidence max/mean `1.013278961e-6` /
`2.254430751e-7`. The Metal-feature CLI parity test and package Clippy with
warnings denied both passed. The Mac log is
`/private/tmp/vokra-fcpe-mac-eba4ca07/final.log`, SHA-256
`ceecee9efcca3047dc86784cac1bb9931349fe6d676e48a32d351f7640c1c82a`.

The VAST evidence was pulled before instance `48630865` was destroyed; the
post-delete inventory was empty. The strict replacement GGUF has **not** been
uploaded: the current public revision therefore retains an explicit artifact
error until the owner separately authorizes the gated Hugging Face publish
chain. Code-route completion and a verified local replacement do not make the
stale Hub bytes a pass.

### RMVPE code route (still CPU-partial)

RMVPE now routes Conv2D/ConvTranspose2D lowering, BiGRU projections and the
pitch head through GEMM/GEMV for non-CPU backends, with batch normalization,
pooling, activations, scatter/layout work and pitch decoding on the host. The
synthetic learned-primitive CPU/Compute checks pass.

RMVPE is nevertheless classified as routed-partial. Its own documented CPU
forward omits the upstream U-Net decoder skip-concat and discovers topology
from unverified tensor-name conventions; no real-checkpoint numeric parity has
closed that gap. Metal cannot be called complete while the CPU oracle is known
to diverge, even though the implemented primitives have a GPU path.

### Parakeet CTC and TDT

Parakeet CTC 1.1B and TDT 0.6B v3 now preserve the caller-selected backend
through their shared FastConformer encoder and task heads. Initial 2-D
subsampling convolutions are lowered to GEMM, Conformer depthwise time
convolution uses grouped Conv1D, norms and attention softmax use their native
dispatchers, and all projections/attention matrix products use GEMM/GEMV. TDT
also dispatches both LSTM projections and its duration-aware joint head; CTC
dispatches the complete vocabulary head. Frontend DSP, masks, layout and
token search remain host control.

The 2-D depthwise subsampling route currently uses one small GEMM per channel.
That is a correctness-first complete route, not a performance claim. VAST
all-target compilation and the focused 100-test Parakeet suite pass. The
public artifacts are 2.51 GB and 4.25 GB, so their real-file gates run only on
VAST and their final CPU/Metal comparison remains an Apple-hardware action.

The real public-file checks exposed two artifact blockers rather than a code
fallback:

| Public repo / revision | File verdict |
|---|---|
| `vokra/parakeet-tdt-0.6b-v3` / `e2448d380310b49b74a6776e9903929ae5a4467d` | 2,508,284,704 bytes, SHA-256 `df5e044b040fa27447de23912694b462c6e97b8d5510c24e8c1ed6090dcc0a18`; strict 699-tensor bind and real PCM encoder plus prediction-LSTM/joint-head CPU step passed, but the file has no embedded tokenizer, so CLI text fails closed pending gated replacement. |
| `vokra/parakeet-ctc-1.1b` / `ea89a33ba60b3a0be36e7c091086515078a58935` | 4,250,632,160 bytes, SHA-256 `6d7fe361d6440f5ae43d471836288329748e64ed0f0443475a678dccd6d04daa`; strict bind rejects its `convolution_bias=false` metadata because the pinned official 1.1B contract and independently validated canonical artifact require `true`. |

The already recorded canonical CTC artifact is 4,251,045,248 bytes with
SHA-256 `8cbe063dc66b5395c2c5b352f34b2864cbd933fc4637f2a6a10455a2e2313f6d`
and passed the official 138-frame encoder/logit/token/text fixture before this
Metal wave. The current Hub file is not silently treated as that artifact.
Replacing either Hub file remains a separately authorized publication action.

### VAST verification record

VAST instance `48520699` reproduced the uncommitted worktree without creating
or pushing a commit. `cargo fmt --all -- --check`, `cargo metadata`, and the
final `cargo test --workspace --quiet` all passed. The first full run caught an
NSNet2 identity-mask test whose reference omitted the official no-delay
right-padded frame; the reference was corrected to the pinned frame-count
formula, the single test passed, and the full workspace rerun returned zero.
`cargo clippy --workspace --all-targets -- -D warnings` then found one new
too-many-arguments lint in the NSNet2 GRU adapter; grouping its borrowed
weights fixed the lint, the full Clippy run passed, and all 16 NSNet2 unit
tests passed afterward. The instance was destroyed immediately after these
checks.

Disposable instance `48524614` then validated the VAD wave. The complete
workspace test suite and all-target Clippy passed after Clippy identified and
the branch corrected two TEN-VAD needless borrows plus one range-loop style
issue. The focused TEN tests passed and the instance was destroyed.

Disposable instance `48525871` validated the RNNoise/NKF-AEC wave and the
combined worktree. `cargo check -p vokra-models --all-targets`, focused
RNNoise/NKF-AEC tests, `cargo test --workspace --quiet`, and
`cargo clippy --workspace --all-targets -- -D warnings` all completed with
zero failures. Local `git diff --check` also passed. The instance was destroyed
and the Vast API subsequently returned no instance record.

Disposable instance `48527624` validated the vocoder, FCPE/SmartTurn,
Pyannote, RMVPE and Parakeet additions. Focused HiFi-GAN/BigVGAN, FCPE,
Pyannote, RMVPE and Parakeet tests passed; the final
`cargo test --workspace --quiet` completed with zero failures. The final
`cargo clippy --workspace --all-targets -- -D warnings` first identified two
SmartTurn needless borrows, which were removed; the full rerun passed. A
post-fix five-test SmartTurn run and the real public TDT PCM/head smoke also
passed. The CTC public-file test stopped at the intentional strict metadata
gate described above. Linux VAST cannot execute Metal; the feature-gated
real-weight CPU/Metal tests remain Apple-runner gates.

Instance `48527624` was then destroyed; the Vast API returned no instance
record (`instances: null`). Its downloaded multi-gigabyte GGUFs were temporary
verification inputs and were deleted with the instance.

Disposable instance `48533965` validated the Wav2Vec2/Data2Vec/Hubert,
SepFormer and Conv-TasNet wave together with the accumulated worktree. For
Conv-TasNet, the isolated Asteroid 0.7.0 oracle, strict checkpoint preparation,
three converter tests, two real-weight parity tests, the real Session/C-ABI
test and a real CLI WAV run all passed. The deliberately stale public
Conv-TasNet artifact failed at its missing pinned-revision metadata instead of
being accepted as the corrected topology. The live Hub audit still reported
194 public repositories, 193 GGUF repositories and 198 GGUF files, with CPU
coverage `full=61`, `partial=48`, `no-runtime-binder=84`, `not-artifact=1` and
Metal coverage `full=61`, `blocked-by-cpu=132`, `not-artifact=1`; the corrected
source route does not inflate those counts while its public artifact remains
blocked. `cargo clippy --workspace --all-targets -- -D warnings` and the final
`cargo test --workspace --quiet` both completed with zero failures. Instance
`48533965` was then destroyed, and the Vast API returned no instance record
(`instances: null`).

Disposable instance `48553051` validated the SpeechBrain X-vector wave and
the accumulated worktree. The official upstream checkpoint was reduced only
by removing five non-inference BatchNorm counters, converted to a strict
32-tensor GGUF, and tested alongside the public 46-tensor combined artifact.
Both layouts produced the same bounded CPU result against the independent
SpeechBrain fixture: frontend max-abs `9.517669678e-4`, embedding max-abs
`2.414703369e-3`, and embedding cosine `0.9999998212`. Real CLI and C-ABI
speaker embedding tests, strict metadata/manifest tests, and the converter
tests passed as well.

The final `cargo test --workspace --quiet` completed with zero failures, and
`cargo clippy --workspace --all-targets -- -D warnings` completed with zero
code warnings after the TDNN layer parameters were grouped into one internal
specification type. The Rust public-API snapshot, generated C header, ABI
changelog gate, and local `git diff --check` also passed. Instance `48553051`
was destroyed after verification; the Vast API returned no instance record
(`instances: null`).

Disposable instance `48558924` validated the SpeechBrain ECAPA-TDNN wave and
the accumulated worktree. The canonical public 200-tensor GGUF and a fresh
strict conversion both passed the independent frontend and end-to-end fixture
at the bounds recorded above. Real public-file CLI and C-ABI speaker embedding
tests passed. The second public ECAPA file failed at its malformed tensor data
boundary, which is recorded as an artifact blocker rather than hidden by the
working sibling checkpoint.

After correcting one stale Lang-ID error-message assertion and one Clippy
range-loop style finding, `cargo test --workspace --quiet` and
`cargo clippy --workspace --all-targets -- -D warnings` both completed with
zero code failures. The live Hub audit then reported CPU coverage `full=64`,
`partial=50`, `no-runtime-binder=79`, `not-artifact=1` and Metal coverage
`full=64`, `blocked-by-cpu=129`, `not-artifact=1`. Linux VAST cannot execute
Metal; the real Apple-device parity gate remains open. Instance `48558924`
was destroyed after verification; the Vast API returned no instance record
(`instances: null`).

Disposable instance `48565792` validated the WeSpeaker ResNet34-LM wave and
the accumulated worktree. The independent official-source fixture, strict
public 182-tensor GGUF bind, frontend/end-to-end parity, real CLI speaker run,
real C-ABI speaker embedding, and selected-backend threading test all passed.
The old public 219-tensor sibling reached its intentional fail-closed
CC-BY-4.0 provenance gate instead of being accepted under its incorrect
Apache-2.0 stamp. The full `cargo test --workspace --quiet` and
`cargo clippy --workspace --all-targets -- -D warnings` runs both completed
with zero code failures. The live Hub audit then reported CPU coverage
`full=65`, `partial=51`, `no-runtime-binder=77`, `not-artifact=1` and Metal
coverage `full=65`, `blocked-by-cpu=128`, `not-artifact=1`. Linux VAST cannot
execute Metal; the feature-gated Apple comparison remains open. Instance
`48565792` was destroyed after verification; the Vast API returned no instance
record (`instances: null`).

Disposable instance `48577185` validated the DAC token-to-waveform wave and
the accumulated worktree. The committed official 44.1 kHz fixture passed both
decoder-only and complete RVQ-plus-decoder CPU gates at the narrow bounds
recorded above. `cargo fmt --all -- --check`,
`cargo test --workspace --quiet`, and
`cargo clippy --workspace --all-targets -- -D warnings` all completed with
zero failures. The M1 Metal comparison was run separately against the same
public GGUF and official code/PCM fixture because Linux VAST cannot execute
Metal. Before teardown, the four raw logs and their SHA-256 ledger were pulled
to the maintainer host; the workspace-test and Clippy log digests are
`e1388eb5b05e528612563bc75843cbc07b67c9a3f0d2952501bd7b4a40fe0fad`
and `9afd3058b31e85932d66d0ac43cc01e52081f8e55103261406c389a9e39e5eb8`.
Instance `48577185` was then destroyed; the paginated Vast inventory returned
three unrelated labels and no `vokra-*` instance.

Disposable instance `48584151` validated the SNAC 24/44 kHz wave and the
accumulated worktree. Both public GGUFs passed strict binding, exact official
encode-code comparison and the encoder/RVQ/decoder CPU gates recorded above.
The workspace test run completed with zero failures. Two Clippy-only
range-loop findings were corrected; the final four-stage unit test and
`cargo clippy --workspace --all-targets -- -D warnings` then completed with
zero code failures on commit `2b9899a8`. Before teardown, all logs, the
environment/SHA ledger and generated fixture archive were pulled to
`/private/tmp/vokra-vast-48584151-results`; the complete archive SHA-256 is
`4d9d6b775ef17a95afe721cbfc2814b67077a1adf43499b66723495c4df8f020`,
the workspace-test log SHA-256 is
`c5ae60f43e52973f4e32d525aca3a25dadc9906e236b3747a36832514747b357`,
and the final Clippy log SHA-256 is
`66f667e1101401a045f84aedaf4d027a579bdbbb0b11cf9e8c221dbc28e7feb4`.
Instance `48584151` was destroyed; the paginated inventory then returned only
the unrelated `tiny-s2s-m2-{judge,generator}` instances and no volumes.

Disposable instance `48591549` validated the standalone SNAC CLI wave and the
accumulated branch at commit `7df00dc4`. The fixed-revision public 24/44.1 kHz
GGUFs passed encode/decode, exact output-rate/frame-count checks, cross-variant
rejection, and the explicit Metal-encode/no-fallback rejection. The full
workspace run recorded 269 test-result suites, 7,153 passed, zero failed and
34 ignored; `cargo clippy --workspace --all-targets -- -D warnings` returned
status zero. The logs and small code/WAV products were pulled to
`/private/tmp/vokra-vast-48591549-results`; the archive SHA-256 is
`c093325f1955c0b65fcac48e1df3bd21bc4da251c89cb9e4e4d79afc548d4ff5`,
the clean public-GGUF CLI log SHA-256 is
`8cba52271cf13536c6de42ddc3b968d0fb8e696c45a3d33d2070e3be21b678d6`,
and the workspace/Clippy log SHA-256 is
`9a106b6627c80141093b16052a1cfcc8c3fed874714e810936e2aa6d0b94ca97`.
Instance `48591549` was destroyed after local SHA verification; the Vast API
again returned only the unrelated `tiny-s2s-m2-{judge,generator}` instances
and no volumes.

Disposable instance `48595336` validated the first MeloTTS wave against the
fixed public English revision `41fc375b3677373e2141ba5b80cd072581ee4308`.
All five official releases received strict 1,051-tensor manifest binders, and
the English artifact passed strict bind, decoding of the 119 inference
`enc_p` tensors, and a one-token real-weight CPU text-encoder forward. The
MeloTTS and shared SBV2 text tests, package check, and package Clippy all
passed. Five raw logs were pulled to
`/private/tmp/vokra-vast-48595336-results`; the real-smoke and Clippy log
digests are
`1ccf1a23c53eb94c1ad092614fc47547b1c39afc666c953ccf4e61e49632b5d3`
and `9dd506647afd577829b6e415281e6803f3a6008d1d121ef1460a259120244e95`.
Instance `48595336` was destroyed after local verification.

Disposable instance `48597790` validated the MeloTTS duration and latent-flow
wave. The public English GGUF (SHA-256
`1196312e86d8e9ba553f505d8cbc151cf6a53c56d0c91dd1c1989c26e2567ee4`)
passed strict bind and a real-weight CPU run through the text encoder,
deterministic duration predictor, and all four VITS2 Transformer coupling-flow
blocks. Focused MeloTTS/shared-duration/shared-flow tests reported
`6 + 13 + 15` passes and zero failures; package check, format, package Clippy
with warnings denied, both architecture shell gates, and `git diff --check`
also passed. The ten-log SHA ledger was rechecked locally after extraction to
`/private/tmp/vokra-vast-48597790-results`; the complete archive SHA-256 is
`ab702cf3f7ba50b82baaa1b644cb33cc8079de76fe1b2f715e231f9a2ca1ecbb`.
Instance `48597790` was destroyed, and the paginated Vast inventory returned
only the unrelated `tiny-s2s-m2-{judge,generator}` instances with no Vokra
label.

Contract `48599976` was allocated for the next MeloTTS wave but Vast reported
its required resources unavailable and queued it with intended state
`stopped`. It contained no work or result data and was immediately destroyed
instead of being left stopped or billable.

Disposable instance `48600202` validated the MeloTTS HiFi-GAN and integrated
acoustic-core wave. The same pinned public English GGUF passed a real-weight
CPU path from supplied phoneme/tone/language/BERT features through text,
duration expansion, acoustic-prior sampling, all four latent-flow blocks,
speaker-conditioned HiFi-GAN and 44.1 kHz PCM. The one-token complete smoke
finished in 9.52 seconds and checked exact 512-sample-per-frame output,
finiteness and the terminal `[-1, 1]` range. Ten focused MeloTTS tests and the
conditioned shared-HiFi-GAN backend comparison passed; package check, format,
package Clippy with warnings denied and `git diff --check` also passed. All
eleven log/text entries passed the locally re-run SHA ledger after extraction
to `/private/tmp/vokra-vast-48600202-results`; the complete archive SHA-256 is
`a9f288e272d4ee3e02647d35b778b45c11081173cc51f77d6133a987dac1c38b`.
Instance `48600202` was destroyed, and the inventory again returned only the
unrelated `tiny-s2s-m2-{judge,generator}` instances with no Vokra label.

Disposable instance `48602407` generated an independent MeloTTS English
reference from MyShell source commit
`209145371cff8fc3bd60d7be902ea69cbdb7965a` and the fixed upstream checkpoint
revision `bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef`, then compared it with Vokra's
fixed public GGUF revision `41fc375b3677373e2141ba5b80cd072581ee4308`
(GGUF SHA-256
`1196312e86d8e9ba553f505d8cbc151cf6a53c56d0c91dd1c1989c26e2567ee4`).
The fixture covers speaker conditioning, both BERT feature planes, text
encoder taps, deterministic log-duration and exact integer durations, length
regulation, inverse flow, decoder PCM, and the integrated acoustic path. The
official checkpoint and config SHA-256 values are respectively
`acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c`
and `039116c927c70eaa4458d315ea83aaaa99e1fca1c621b50c8ca56b4a5700eb77`.

The first independent run exposed a real graph mismatch: the shared SBV2
coupling flow injected projected speaker conditioning before the transformer
stack, while released MeloTTS injects it immediately before block two through
`Encoder.cond_layer_idx = 2`. After making that position a model-private
override, maximum absolute errors were `7.23e-7` for encoder hidden state,
`1.31e-6` for the prior mean, `5.96e-7` for prior log-scale, `1.19e-6` for
deterministic log-duration, `1.91e-6` for inverse-flow latent, `4.27e-7` for
decoder PCM, and `6.65e-7` for integrated PCM. All remain far below the
unchanged FP32 gate `atol = 0.01`; exact integer durations and expanded tensors
also matched.

At commit `4e193622`, package Clippy with warnings denied, format, the real
official parity test, and the complete `vokra-models` package test all exited
zero. The library summary was `2,541 passed; 0 failed; 6 ignored`. Final logs
were pulled to `/private/tmp/vokra-melotts-evidence-4e193622.tar.gz`; local and
remote archive SHA-256 both equal
`2706576bf676f9b1ac6d5a98c345f745af3208875378fde9e1a96135443adcb8`.
The independently generated fixture archive was also pulled to
`/private/tmp/melotts-fixture-6565903c.tar.gz` and verified at SHA-256
`047b5ed7f34dfbd046a95d3b071b4ec36b3a30ad8b0e32b4a3390351369a51ad`.
Instance `48602407` was destroyed after those checks. The Vast inventory then
contained only the unrelated `tiny-s2s-m2-{judge,generator}` instances: no
Vokra-labelled instance remains stopped or running.

That first wave was not a full five-release MeloTTS completion claim: only the
English feature-to-PCM path had an independent official numerical gate, and
the Apple-hardware comparison was still open. The follow-up wave closed those
two gaps for every acoustic GGUF actually published under `vokra/melotts-*`.
It did not fabricate the five language-specific raw-text/G2P/BERT frontends,
which are not embedded in these acoustic artifacts.

The independent dumper now imports the same pinned MyShell source commit and
one of five fixed official checkpoints in a dedicated Python 3.12 environment
(`torch 2.13.0+cpu`, `numpy 2.5.2`, `huggingface-hub 0.29.3`, `numba 0.67.0`).
The public-artifact and upstream-revision ledger is:

| Variant | Upstream checkpoint revision | Public revision | GGUF SHA-256 | Bytes |
|---|---|---|---|---:|
| English | `bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef` | `41fc375b3677373e2141ba5b80cd072581ee4308` | `1196312e86d8e9ba553f505d8cbc151cf6a53c56d0c91dd1c1989c26e2567ee4` | 207,575,360 |
| Chinese | `af5d207a364ea4208c6f589c89f57f88414bdd16` | `2d02213da50af3d5384c2f972681014a2eb05ab5` | `11f87f890e95cf572ad207aae87f6a961b7c9ebe4eee81c69b0a6c2440376a1e` | 207,484,736 |
| Korean | `0207e5adfc90129a51b6b03d89be6d84360ed323` | `3737e27dba5f54e98ab3ae816bf610ae6edaeeb2` | `6e27bbc9c55dd5acc756317044be42fac4a85f5315aca38cfb881ac5984f24d9` | 207,575,360 |
| Spanish | `dbb5496df39d11a66c1d5f5a9ca357c3c9fb95fb` | `1ee8c1c2df484ea59bd7382f88b292b0da95df3e` | `3a293e474c3d51e271a4bcb7e980f5f3e6866cbf2ba9a7c3780cf36f9c10e184` | 207,575,360 |
| Japanese | `367f8795464b531b4e97c1515bddfc1243e60891` | `5c61fa7b6f723c039e7d4721f3d5ab77b99d867e` | `f12c079ae4df51e59895ac29a8bb0043ae3c78be3aa1ad22ab84de71d4ff81a8` | 207,575,360 |

VAST instance `48616858` generated all five fixtures from the official code,
then ran the complete speaker conditioning, text encoder, deterministic
duration, exact integer duration, length regulation, inverse flow, decoder and
integrated acoustic paths against all five public GGUFs. The largest observed
per-variant absolute error was:

| Variant | Largest official CPU max abs |
|---|---:|
| English | `1.907348633e-6` |
| Chinese | `2.145767212e-6` |
| Korean | `1.907348633e-6` |
| Spanish | `1.907348633e-6` |
| Japanese | `3.576278687e-6` |

All exact duration checks passed and every value remains far below the
unchanged FP32 `atol = 0.01`. The five-artifact gate finished in 163.90
seconds; the complete library run reported `2,548 passed; 0 failed; 6
ignored`. Format, package check and package Clippy with warnings denied also
exited zero. The fixture/log/environment archive was pulled to
`/private/tmp/vokra-melotts-five-vast-48616858-results`; remote and local
archive SHA-256 both equal
`c62903a7596309ca7f385f3089aed97aab7b8c8483cd27f88f366ea3cd6acbf0`,
and its internal SHA ledger was rechecked locally. The five fixtures were
committed at `766f8255`. The instance was then destroyed rather than stopped.

The exact same five GGUFs and committed `VKRMELO1` inputs were next exercised
through the public CLI on the maintainer's Apple M1 at commit `766f8255`.
CPU and Metal produced equal geometry and finite 44.1 kHz float32 WAVs:

| Variant | Samples | Different samples | max abs CPU/Metal | RMSE CPU/Metal |
|---|---:|---:|---:|---:|
| English | 3,072 | 439 | `4.768371582e-7` | `5.321865458e-8` |
| Chinese | 3,072 | 33 | `1.192092896e-7` | `1.235539016e-8` |
| Korean | 3,584 | 21 | `1.192092896e-7` | `9.125060375e-9` |
| Spanish | 4,096 | 35 | `1.192092896e-7` | `1.101955731e-8` |
| Japanese | 3,072 | 18 | `1.192092896e-7` | `9.125060375e-9` |

A sandboxed probe first returned the explicit error `backend unavailable: no
system default Metal device`; it did not fall back to CPU. The real Apple
device runs then completed outside Seatbelt, and focused Metal-feature CLI
tests reported `7 passed; 0 failed`; Metal-feature CLI Clippy with warnings
denied also exited zero. The 25-file Mac log/WAV/environment set is under
`/private/tmp/vokra-melotts-mac-766f8255`; its locally rechecked SHA-ledger
digest is
`4bd04b0340ff29dd042349cabff3f572cc63aa0a8e58e671e47fd30a1d3015ce`.

Therefore every currently published MeloTTS GGUF has an explicit real-file
CPU verdict and an Apple-hardware Metal verdict for its complete learned
acoustic feature-to-wave path. Raw language text still requires caller-side
normalization, G2P and BERT feature extraction; `--text` is an explicit error
rather than a silent approximation or CPU fallback.

### FocalCodec 50 / 25 / 12.5 Hz family

All three public FocalCodec releases now have a strict native binder, complete
waveform encode and BSQ-token decode, CLI routing, and a versioned `VKRFOC01`
container. The container records the exact GGUF tensor fingerprint, sample
rate, frame hop, token rate, codebook size and original PCM length; a sibling,
modified checkpoint or out-of-range token is rejected before decoding. Every
learned GEMM, Softmax, LayerNorm, GELU, Conv1D, grouped Conv1D and Snake
activation uses the selected `Compute` backend. There is no CPU fallback in
the Metal path.

The public artifacts used for the final gates were:

| Variant | Public revision | GGUF SHA-256 | Bytes |
|---|---|---|---:|
| 50 Hz | `f9b5504c2e4fd7c4545e4b1a1344968b54f81813` | `3d19613193fe8cd4f3725209fa83c278e33d8b1e96fde43594b6c4328cf18d93` | 568,542,752 |
| 25 Hz | `346b834d7399b5276419c57683cef235b2c84e0f` | `1b11f8deb5fb0447b3f3b6a8cbdacbdb43e2aeb02604aff93bbfe1c8c4c57be6` | 576,931,392 |
| 12.5 Hz | `213e11c0105a71d6ea3f0883ab7e1f7509cf4ce2` | `d17de845cd25ec434d05df56e6befca0c992cb3c072d1d76c97285371c39e4cb` | 581,125,728 |

The independent upstream oracle is pinned to FocalCodec source commit
`912b7f2c0cd43d54a8aed296bbcc925dec7d4ea3`. On VAST, all three official
fixtures produced exact token IDs. Final decoded-PCM errors against the
independent reference were:

| Variant | max abs | RMSE |
|---|---:|---:|
| 50 Hz | `1.169741154e-6` | `2.094736810e-7` |
| 25 Hz | `1.396238804e-5` | `2.263951977e-6` |
| 12.5 Hz | `6.938353181e-7` | `9.222271064e-8` |

VAST instance `48606474` then validated commit `51a0681c`: the complete CLI
run reported `219 + 4 + 2` tests passed with zero failures, package Clippy
with warnings denied passed, and every public GGUF completed real CPU CLI
encode/decode. The resulting 50/25/12.5 Hz containers held 9/5/3 tokens and
had SHA-256 values `57ebef3d83cb773b85453c5139b2e3a63c3dd9d363883c82455781b9ac12b34a`,
`bb5f854b13df36093e7160d1b957194fdd604c5e36721e34d444fadf6682190a`
and `d9a2d1d3fc8e5f39178872664a5353f5947e1c38f5b7c5c9c69bda72e821c438`.
The 38-file log/fixture/environment set was pulled to
`/private/tmp/vokra-vast-48606474-results` and passed its locally re-run SHA
ledger; the ledger SHA-256 is
`c3962ea9f83dd5b662d65999f8b86923af92686fab10aede94b36fdf27710073`.
The instance was destroyed rather than stopped. The post-delete inventory
contained only the unrelated `tiny-s2s-m2-{judge,generator}` instances and no
Vokra-labelled instance.

The same three fixed public GGUFs were then exercised on the maintainer's
Apple M1 with one identical 3,200-sample input. CPU and Metal produced
bitwise-identical containers for all three variants, including the same hashes
as the VAST CPU run. Decoding the same tokens through CPU and Metal produced:

| Variant | samples | max abs CPU/Metal | RMSE CPU/Metal |
|---|---:|---:|---:|
| 50 Hz | 3,200 | `1.627951860428e-6` | `2.565435226431e-7` |
| 25 Hz | 3,200 | `3.869179636240e-6` | `8.217389695893e-7` |
| 12.5 Hz | 3,200 | `1.234409864992e-6` | `1.822034192231e-7` |

Every sample was finite and all three comparisons remain far below the
unchanged FP32 `atol = 0.01` gate. A first sandboxed probe correctly returned
an explicit `no system default Metal device` error rather than falling back;
the Apple-device measurements above were run outside that Seatbelt restriction
after macOS independently reported the M1 GPU as Metal-capable. The 20-file
Mac log/output/environment set is under
`/private/tmp/vokra-focalcodec-mac-51a0681c`; its locally verified SHA-ledger
digest is
`887d3c047ff8dfbaa75f50ed06bcc5ad602dc82282940fed726fa0ed0b2cc3bb`.

### WavTokenizer large-speech 75-token decoder

The two public repositories `vokra/wavtokenizer-large` (revision
`fa3dc6c15581c76c86be61513003fd84fa161d54`) and
`vokra/wavtokenizer-large-speech-75token` (revision
`103a577083a7221728cc7c60354044acc664657c`) carry the same
846,393,344-byte, 1,091-tensor GGUF. Its SHA-256 is
`99b7dce0426266f7f2f6615091d832cea71387ce57edfae66666143a5c33a36b`.
The strict loader accepts that exact released MIT/provenance/model contract
and rejects extra, missing, wrong-shaped or differently identified manifests.

The native decoder covers the single 4,096-entry, 512-dimensional codebook,
four conditioning rows, four positional ResNet blocks and temporal attention,
all 12 conditioned ConvNeXt blocks, magnitude/phase head and same-padded
1,280-point iSTFT. VQ gather, Conv1D, grouped Conv1D, GroupNorm, LayerNorm,
GEMM, Softmax, GELU and SiLU honor the requested CPU or Metal backend. CLI
decode accepts one little-endian `u32` code per 75 Hz frame and writes 24 kHz
mono WAV. Encode remains an explicit unsupported-operation error.

The independent oracle imports the official `jishengpeng/WavTokenizer`
implementation at commit `5cf440d91ac420ca338f117b7003a77450d64730`, loads
the audited GGUF's verbatim F32 inference state into the upstream modules and
calls the official `codes_to_features` plus `decode` methods. For the
four-code/1,280-sample fixture, the public GGUF produced:

| Comparison | max abs | mean abs | relative L1 | cosine |
|---|---:|---:|---:|---:|
| Mac CPU / official | `1.628510654e-5` | `2.329980862e-6` | `5.741390002e-6` | `0.999999999978` |
| Apple M1 Metal / official | `1.708976924e-5` | `2.329881909e-6` | `5.741146168e-6` | `0.999999999978` |
| Mac CPU / Apple M1 Metal | `5.155801773e-6` | `1.240446591e-6` | `3.056629167e-6` | `0.999999999995` |

The official fixture is registered at `atol=2e-5` and cosine
`>=0.999999999`, both derived after the oracle was fixed and without relaxing
the measured result. Disposable VAST instance `48645355` first validated the
focused strict-loader/model tests (`4 passed / 0 failed`) and warnings-as-error
library Clippy at implementation commit `97f3db3e`; its logs were pulled to
`/private/tmp/vokra-wavtokenizer-vast-48645355` and the instance was destroyed.
After the official fixture landed, instance `48646449` validated exact commit
`a409b780`: the same four model tests passed, while the env-gated integration
binary compiled and exercised its documented no-artifact skip (`5 test
results / 0 failures` in total); the shared Vocos tests passed (`3 / 0`), and
`cargo clippy -p vokra-models --all-targets -- -D warnings` exited zero. Its
three logs were pulled to `/private/tmp/vokra-wavtokenizer-vast-48646449` with
SHA-256 values `826c7dc3ac57cf465842a15a34e6f1be88855e765bb970f2a85d118aa7b22b80`,
`764da45a554e243811e7aaac5e6bbce2f4d271885e291d44c753e65d2e32150f`
and `d2da208390b411708b601b3329fe47005cdaeae63cab10f0289754f63a50a331`.
The instance was destroyed and the live VAST inventory returned zero
instances. No Hugging Face upload or public artifact change was performed.

### NeuCodec base and distill 50 Hz decoders

The public base GGUF is 2,519,825,344 bytes with 811 F32 tensors and SHA-256
`b71d9d7867a4c244562caa2d735e93c9b744c70110c346f3f65e0862e41163fc`.
It predates the additive `vokra.neucodec.variant` key and uses the normalized
`acoustic_decoder.*` namespace with separate Q/K/V projections. The public
distill GGUF is 1,025,417,504 bytes with 294 tensors and SHA-256
`15e60e7e5f7242255b18e1386b26c2a8f872c77a56ca241ee82c8aa5d8b6327f`.
It uses the current pass-through `generator.*` namespace and explicitly stamps
`variant=distill`. The strict runtime pins the complete name/shape manifests
independently
(`1b76dc8f93c5c68f01329f9f05b6f34292b41bd39b4c46e08229327daa0102e0`
for base and
`8bf4f171559b9da0d1531867a7f2bfec5265cc5932b0df895a51913438744f1b`
for distill),
then maps both to one released decoder topology. Unknown variants, manifests,
shapes and provenance fail before inference.

The native path decodes one 65,536-way `[4; 8]` FSQ code per 50 Hz frame,
projects 2,048 to 1,024 channels, executes four ResNet blocks and twelve
non-causal 16-head Transformer blocks, and emits magnitude/phase bins through
the 1,920-point same-padded iSTFT head. FSQ, Conv1D, GroupNorm, RMSNorm, GEMM,
Softmax, SiLU and LayerNorm honor the selected CPU or Metal backend. The
official source's torchtune 0.3.1 RoPE call receives `[B,H,T,D]` despite the
documented `[B,T,H,D]` contract, so the released head-axis behavior is
preserved intentionally rather than silently corrected. CLI decode accepts
one little-endian `u32` per frame and writes 24 kHz mono WAV; encode is an
explicit unsupported operation.

The independent oracle imports Neuphonic's official source at commit
`ed3e6cd1bdc374ce14a21355e5eee66a777149ce`, pins
`vector-quantize-pytorch==1.17.8`, and directly loads the SHA-pinned official
`torchtune==0.3.1` RoPE source file without importing unrelated torchao
initializers. It restores each GGUF into the upstream `CodecDecoderVocos` and
calls the official FSQ plus decoder forward. VAST CPU real-weight results were:

| Public artifact | samples | max abs vs official | RMSE | cosine |
|---|---:|---:|---:|---:|
| base | 1,920 | `3.561377525e-6` | `1.159706864e-6` | `0.999999999968` |
| distill | 1,920 | `3.233551979e-6` | `9.182642293e-7` | `0.999999999981` |

The 2 GB artifact rule keeps the base public file off the maintainer Mac. The
distill artifact is below that threshold and completed the same fixture on the
real Apple M1 outside Seatbelt after the sandboxed probe correctly returned
`no system default Metal device`:

| Comparison | max abs | RMSE | cosine |
|---|---:|---:|---:|
| Mac CPU / official | `3.620982170e-6` | `1.103911115e-6` | `0.999999999971` |
| Apple M1 Metal / official | `7.655471563e-6` | `1.753893327e-6` | `0.999999999927` |
| Mac CPU / Apple M1 Metal | `5.222856998e-6` | `1.169858302e-6` | `0.999999999967` |

The shared base/distill decoder therefore has a complete Metal code route and
real Apple-device evidence for distill. A public-file Apple run for the
2.52 GB base artifact remains unrecorded under the local artifact safety rule;
its exact public loader and decoder weights have CPU/official parity, while
the same backend-parametric graph has the distill Metal verdict above.

Disposable VAST instance `48648187` validated exact commit `eb7b9f35`: four
NeuCodec model tests passed, both real-public CPU parity tests passed, the base
CLI decoded 4 codes to 1,920 samples, and
`cargo clippy -p vokra-models --all-targets -- -D warnings` exited zero. The
six-file evidence set was pulled to
`/private/tmp/vokra-neucodec-vast-48648187`; its model-test, real-parity and
Clippy log SHA-256 values are respectively
`234ae6b3be5d57e2a62abc42b1e610a93ca02889259d6db41d6463df524dbc18`,
`c3b4dca82457005fc109d8e048b48593547c655cb89a0a5d46b580b558e1cc03`
and `4ec4b714ceda93c417bed4ebc9845df83bf9a7ce370c7e2b415f47da611868f2`.
The instance was destroyed, and the live VAST inventory returned zero
instances. No Hugging Face upload or public artifact change was performed.

## Remaining execution order

1. Make all remaining no-binder repositories CPU-runnable, family by family, with a
   strict loader and independent upstream reference.
2. Finish the partial CPU forwards; do not mark a converter, synthesized
   bridge, or tensor probe as runtime completion.
3. The former CPU-complete/CPU-only set is closed at zero. Expand Metal only
   after each currently blocked CPU family closes its independent CPU gate.
4. Run every one of the 193 public GGUF repositories through a real-file CPU
   smoke and independent numerical gate. A missing file/variable is a skip,
   not a pass.
5. Run every Metal-capable public checkpoint on Apple hardware with CPU as the
   independently validated oracle. Preserve `atol = 0.01` for FP32 unless a
   separately justified, pre-registered model output bound is stricter.
6. Replace stale or incomplete public artifacts only through the gated publish
   chain and only after explicit upload authorization for the exact repos.

Completion means all 193 GGUF repositories have explicit real-file CPU
verdicts, every supported Metal checkpoint has an Apple-hardware parity
verdict, and the Hub revisions users download are the revisions that passed.
The empty Seamless repository must either receive a real gated artifact or be
withdrawn by an explicitly authorized publication action; it cannot count as
a model pass.
