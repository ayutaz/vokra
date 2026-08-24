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

At 2026-08-25, after the DAC 16/24/44.1 kHz wave, it reported:

| Inventory / code reachability | Public repos |
|---|---:|
| Public model repositories | 194 |
| Repositories carrying at least one GGUF | 193 |
| GGUF files | 198 |
| Complete CPU code route | 70 |
| Route/binder present, released-artifact CPU forward incomplete | 51 |
| No complete runtime binder | 72 |
| Empty non-artifact repository (`seamless-m4t-v2-large`) | 1 |
| Complete Metal code route among the CPU-complete set | 70 |
| CPU-complete but Metal-unsupported | 0 |
| Metal blocked by missing/partial CPU forward | 123 |

These are deliberately **code reachability** counts. They are not a claim that
the current Hub file loads, that its sidecars are complete, or that its
real-weight CPU/Metal parity has passed. The TSV form prints the per-repository
revision, GGUF count, architecture and classification:

```text
uv run --no-project --python 3.12 python tools/audit/hf_mac_coverage.py --format tsv
```

The 70 repositories with a complete Metal code route are the four BigVGAN
checkpoints, CAM++, CrisperWhisper, both Distil-Whisper checkpoints, FCPE,
the three DAC checkpoints (16, 24 and 44.1 kHz), FireRedVAD, FSMN-VAD,
HiFi-GAN LibriTTS, both Kokoro checkpoints,
Kotoba-Whisper, Mimi, Moshiko-7B, Moonshine Tiny/Base, NKF-AEC, Parakeet CTC
1.1B, Parakeet TDT 0.6B v3, both Piper Plus checkpoints, RNNoise, both Silero
VAD checkpoints, SmartTurn v2, SpeechT5 HiFi-GAN, TEN-VAD, both Vocos
checkpoints, the three Voxtral checkpoints, nine plain Whisper checkpoints,
Whisper-Medusa-v1, all seven Wav2Vec2 CTC checkpoints, Data2Vec Audio Base,
HuBERT Large LS960, all seven SepFormer checkpoints, and both SpeechBrain
X-vector repositories, the canonical SpeechBrain ECAPA-TDNN repository, the
public pyannote WeSpeaker ResNet34-LM repository, and both byte-identical
TitaNet-Large repositories (`vokra/titanet-l` and `vokra/titanet-large`).
Pyannote Segmentation 3.0 and RMVPE are deliberately
omitted from this list (see below). Each listed repository still needs its own
public-artifact load and real-weight parity verdict; sharing an architecture
does not turn one checkpoint's pass into a sibling pass.

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

### SepFormer seven-checkpoint family

SepFormer now implements the full learned forward: encoder, GroupNorm,
segmentation/overlap-add, two dual-path blocks with 16 total Transformer
layers, multi-head attention, ReLU feed-forward networks, PReLU mask heads and
decoder. Its complete backend hot-op set is GEMM, Softmax, LayerNorm and
Conv1D. CLI and Rust sessions expose source separation, and the C ABI adds
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

The independent SpeechBrain 1.0.3 fixture at pinned upstream revision
`90b3c5c3ffe3e04387b566715ab5fff36ec7b9d9` passed on VAST:

| SepFormer surface | max abs | mean abs |
|---|---:|---:|
| Encoder | `1.192092896e-7` | `1.695379948e-11` |
| Final 4,096-sample waveform | `1.866221428e-4` | `1.220094873e-5` |

The CLI produced a 4,096-sample output WAV from the public WHAM 16 kHz GGUF,
and the real public-model C ABI test passed in 90.08 seconds. These are CPU
measurements; SepFormer's declared Metal path still needs the same explicit
Apple-device verification as the encoder family above.

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
padding and artifact-layout metadata. No public upload or replacement was
performed. Linux VAST cannot execute Metal, so the real Apple-device
CPU/Metal comparison remains a separate gate even though the complete code
route is now counted.

### SpeechBrain ECAPA-TDNN and related public artifacts

The canonical `vokra/ecapa-tdnn` repository now has a strict native speaker
embedding runtime. Its live revision is
`24be4349d49c23bb3b80b5afccf37538e8d616b4`; `model.gguf` is 83,239,808 bytes,
has SHA-256
`207cebb84e53f6eab77d6da65dab4546489dbeb5d4cfc799f64b6e34588a118c`, and
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

No public upload or replacement was performed. Linux VAST cannot execute
Metal, so a real Apple-device comparison remains required before recording a
Metal artifact pass.

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
license override incompatible with the audited CC-BY-4.0 checkpoint. No public
upload or replacement was performed. Linux VAST cannot execute Metal, so the
Apple-device comparison remains a separate artifact gate.

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
relative L1 `2.5e-6` and cosine at least `0.9999995`. The shared strict
definitions cover the released 16/24 kHz manifest contracts, but those two
siblings still require their own official fixture and real public-artifact
runs; the 44.1 kHz result is not imputed to them. No Hub upload or artifact
replacement was performed.

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
the corresponding cosine floors committed beside the fixtures. Linux VAST
cannot execute Metal, and current host policy forbids a local
`vokra-models` real-weight Cargo run, so the feature-gated 24/44 kHz Apple
hardware comparison remains open. This is recorded as an unrun device gate,
not imputed from the passing CPU result or primitive Metal tests. No Hub
upload or artifact replacement was performed.

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
This is code reachability plus the SNAC real-file CPU evidence above, not a
claim that the still-open Apple-device SNAC Metal comparison has run.

### Piper Plus CLI/C ABI reachability

Piper already declared GEMM as its complete backend hot-op set and already had
a CPU/Metal real-weight parity harness, but `vokra-cli` and `vokra-capi`
classified the engine as CPU-only. Both loaders now pass the selected backend
to `PiperPlusTts`, and `TtsEngine::backend` / `Tts::backend` make that wiring
observable. A non-CPU selection is no longer rejected before it can reach the
existing Metal implementation.

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

This is not yet a MeloTTS synthesis completion claim. The HiFi-GAN decoder,
duration expansion/prior sampling integration, language-specific raw-text and
BERT sidecars, independent upstream numerical fixture, and Apple-hardware
Metal comparison remain open. The five releases must stay partial until those
gates close; Linux VAST cannot execute the Metal leg.

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
