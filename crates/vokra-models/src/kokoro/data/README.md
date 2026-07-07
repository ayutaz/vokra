# Kokoro-82M upstream tensor manifest

`upstream_tensors_v1_0.tsv` is the flat tensor manifest of the upstream
[hexgrad/Kokoro-82M `kokoro-v1_0.pth`](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/kokoro-v1_0.pth)
checkpoint. It was captured on **2026-07-07** by CC while closing out
M2-07 T13-T17 (Kokoro converter / native forward schema alignment).

## Format

Tab-separated, one tensor per line:

    <flattened.name>\t(shape,tuple)\t<torch.dtype>

Example:

    text_encoder.module.embedding.weight\t(178, 512)\ttorch.float32
    predictor.module.lstm.weight_ih_l0\t(512, 640)\ttorch.float32

The tensor tree is flattened depth-first: the upstream `.pth` is a
nested `Dict[str, Dict[...] | Tensor]` with top-level keys `bert`,
`bert_encoder`, `predictor`, `decoder`, `text_encoder`. Every leaf
`torch.Tensor` becomes one row; nested dicts contribute their key as
the flattened prefix (dot-separated).

## Regeneration

Idempotent — the checkpoint is public + versioned by SHA:

```
tools/parity/parity-venv/bin/python -c "
import torch
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename='kokoro-v1_0.pth')
state = torch.load(p, map_location='cpu', weights_only=True)
def enum(d, prefix=''):
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            yield from enum(v, key)
        elif isinstance(v, torch.Tensor):
            yield key, tuple(v.shape), str(v.dtype)
for k, s, dt in sorted(enum(state)):
    print(f'{k}\t{s}\t{dt}')
" > crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv
```

## Why this file exists

The `crates/vokra-models/src/kokoro/{text_encoder,prosody,decoder}.rs`
scaffolds were authored during the M2-07 T01–T08 design phase, before
T02 (upstream inspection) had been run against the real checkpoint.
The scaffolds encode a *placeholder* architecture (LayerNorm + Linear
text encoder, minimal prosody, minimal decoder) whose expected tensor
names (`text_encoder.embedding.weight`, `text_encoder.norm.weight`,
`text_encoder.proj.weight`, ...) do NOT exist in the real checkpoint.

The real architecture, discovered on 2026-07-07 by dumping the .pth,
is:

- **text_encoder**: `Embedding(178, 512)` → 3× (WeightNormedConv1d 512→512 + LayerNorm1d) → BiLSTM 512→256 (bi = 512)
- **predictor** (prosody): BiLSTM stacks + duration/F0/energy projections + WeightNormedConv1d refinement
- **bert / bert_encoder**: ALBERT-shaped encoder + a 768→512 projection into the predictor
- **decoder**: WeightNorm-heavy iSTFTNet with `decoder.module.decode.{0..3}.{conv1, conv2, conv1x1, norm1, norm2}` blocks, an `asr_res` residual bridge, and mag/phase output heads

The scaffolds must therefore be reimplemented against this manifest as
part of the M2-07 T13–T17 follow-up. This TSV is the ground truth that
the reimplementation binds to (tensor-by-tensor shape check at load
time, per FR-EX-08). Every scaffold `store.tensor_shaped(...)` call
should map 1:1 to a line in this file after the rewrite.

## Grep-friendly cross-references

Count per top-level prefix (as of 2026-07-07):

```
bert.module.*           : 214 tensors (ALBERT-4 shared-weight backbone)
bert_encoder.module.*   :   2 tensors (Linear 768→512)
predictor.module.*      :  92 tensors (BiLSTM 6-stack + duration/F0/energy heads)
decoder.module.*        : 217 tensors (iSTFTNet: asr_res + decode.0..3 + F0_conv/N_conv + spec heads)
text_encoder.module.*   :  23 tensors (Embedding + 3× WeightNormedConv1d+LN + BiLSTM)
```

Total: **548 tensors** for the whole model. Sample rate = 24 kHz;
phoneme vocab = 178; hidden dim = 512.
