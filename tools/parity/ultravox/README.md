# Ultravox v0.5 independent real-checkpoint parity

This directory stages the VAST-only numerical oracle for the exact public
`vokra/ultravox-v0-5-llama-3-2-1b` audio GGUF and a user-converted, separately
licensed `meta-llama/Llama-3.2-1B-Instruct` companion.

The oracle loads Fixie's official Hugging Face custom code from immutable
revision `b95bec8ab291eeb04b5cd600dd473377f6b79026`. It authenticates and imports
`ultravox_model.py`, `ultravox_processing.py`, and `ultravox_config.py`, then
calls the released `UltravoxModel`, `ModifiedWhisperEncoder`,
`UltravoxProjector`, tokenizer chat template, and `UltravoxProcessor` directly.
It does not reimplement the neural graph and never imports Vokra.

The exact gated Meta snapshot
`9213176726f574b556790deb65791e0c5aa438b6` is loaded locally by the official
model. The generated reference contains the deterministic PCM input, official
Whisper features, projected audio embeddings, complete expanded prompt IDs,
first-position logits, and a short greedy token sequence. The Rust real-GGUF
test compares every FP32 tensor at `atol=0.01` and greedy IDs exactly. Every
reference artifact and both downloaded source closures are authenticated by
SHA-256 manifests. The Rust test requires all three paths and the companion
GGUF hash; it never treats a missing input as a passing skip. The official
reference environment is pinned to `transformers==5.5.0` (the 4.x-to-5.x
upgrade requires a fresh VAST rerun before parity can be claimed).

Run only on a provisioned VAST host:

```sh
scripts/publish/vast-ai/run-ultravox-validation.sh
```

After the separately authorized named VAST job has frozen the exact uv
environment, the model-free dependency/license audit can be run independently:

```sh
VOKRA_PUBLISH_ON_VAST=1 \
scripts/publish/vast-ai/audit-ultravox-dependencies.sh \
  --output /external/evidence/ultravox-dependency-audit.json
```

The audit accounts for every lock row, derives the active Linux x86_64 closure
by marker-aware traversal from the virtual root (including inactive dependency
edges such as Windows-only colorama), compares the installed multiset, records package/native publisher evidence, and inspects
only bounded ELF metadata. Missing installed publisher files may be supported
only by the exact locked PyPI sdist, inspected in memory without extraction or
execution. Its dependency sdist requests are reported separately from the
fixed source/model/Meta companion LICENSE-only requests. It never acquires
weights, imports model code, or invokes Cargo; do not run it before an
authorized named VAST sync.

The worker has no upload or publish option. It downloads fixed snapshots,
converts the gated companion locally, runs CPU/reference and repository gates,
and leaves a small evidence/reference directory to pull. Do not copy model
payloads to the maintainer Mac. Destroy the VAST instance after evidence is
retrieved.

After the VAST CPU gate passes, transfer the two GGUFs and reference directly
from VAST to a disposable Apple Silicon host with at least 24 GB RAM:

```sh
VOKRA_REMOTE_APPLE_SILICON=1 \
scripts/verify/apple-silicon-ultravox.sh \
  --gguf /remote/stage/ultravox.gguf \
  --companion /remote/stage/ultravox-llama-companion.gguf \
  --companion-sha256 <value-from-VAST-input-hashes.txt> \
  --reference /remote/stage/reference \
  --reference-manifest-sha256 <value-from-VAST-input-hashes.txt> \
  --evidence-dir /remote/evidence/ultravox-metal
```

That script performs no download, upload, conversion, publication, or model
deletion. It runs the same real-weight gate on Apple CPU and Metal, recording
unsupported operations as failures rather than using a CPU fallback.

The tracked `dependency_audit_evidence.json` is a compact, model-free VAST
proof for the exact clean audit head. It binds all 37 active Linux rows and
three inactive/virtual lock rows, declared licenses/classifiers, publisher and
native counts/canonical hashes/unsafe lists, the locked-sdist
`tokenizers-0.22.2/tokenizers/LICENSE` fallback, and the exact closure. Fixie
metadata is authenticated at its exact HF revision; the gated Meta companion
records `LICENSE.txt` existence and
`401` for its raw license request, without claiming that its bytes were
reviewed. The proof also records the public model LICENSE bytes and the
Fixie/Meta `404`/`401` fallback facts, and explicitly records no model import,
no Cargo, and `NO_UPLOAD`. It is evidence, not owner sign-off: package/license
rows and signer/digest remain pending, and publication stays blocked until the
owner reviews the bound scope. The evidence proof is independent of numerical
model parity; the parity result and Apple CPU/Metal result must come from their
respective VAST/Apple runs.
