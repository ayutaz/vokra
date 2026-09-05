# NeuTTS Air independent real-checkpoint parity

This directory contains the VAST-only numerical oracle for the exact public
`vokra/neutts-air` GGUF. The oracle loads the exact gated
`neuphonic/neutts-air` snapshot at revision
`3b58b776406b62fdc137e31ea53d728f5c22a4ed` through Hugging Face's official
`Qwen2ForCausalLM`; it does not contain a second Qwen implementation.

Prompt construction executes Neuphonic's released
`NeuTTSAir._apply_chat_template` directly from source commit
`3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e`. The source file is fixed at
9,035 bytes and SHA-256
`e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1`.
Only the phonemizer call is replaced with identity over already-phonemized test
strings, so this gate isolates the language model without inventing an eSpeak
result. Official tokenizer control IDs and the complete prompt are recorded.

The reference emits the first-position 217,652-way FP32 logit vector and a
short deterministic greedy token sequence. The Rust test compares logits at
the repository FP32 bound `atol=0.01` and greedy IDs exactly. The separately
validated NeuCodec route is composition-smoked when the Distill companion is
provided. No fixture is committed before a real run, and a missing environment
variable is a visible skip rather than a pass.

Run only on a provisioned VAST host after the checkout containing this work is
available there:

```sh
scripts/publish/vast-ai/run-neutts-air-validation.sh
```

The worker has no upload or publish path. It downloads the exact public GGUF,
the exact upstream snapshot and the exact public Distill NeuCodec companion,
then runs the official CPU comparison and workspace/Metal cross-build gates.
Pull only its small evidence/reference directory before destroying the VAST
instance; do not pull model payloads to the maintainer Mac.

The separately authorized factual route first performs the exact frozen Linux
environment sync, then runs the model-free audit before any model or source
acquisition:

```sh
VOKRA_PUBLISH_ON_VAST=1 uv sync --project tools/parity/neutts_air --frozen --python 3.12
scripts/publish/vast-ai/audit-neutts-air-dependencies.sh \
  --output /root/scratchpad/neutts-air-dependency-audit.json
```

The audit records every locked row, including the inactive Torch variant and
virtual project row, installed metadata/license files, native ELF inventories,
and bounded exact-sdist fallback evidence. It requests only fixed LICENSE
paths plus exact locked PyPI sdists when publisher files are absent; it never
fetches weights, imports model code, or runs Cargo. The validation worker runs
the same audit immediately after its frozen sync and before model download or
Cargo. Missing publisher evidence or unresolved fixed LICENSE facts remains a
blocker; no license class is inferred from raw bytes.

The VAST worker is fail-closed behind the standard-library-only
`preflight_gate.py`. Its 39-row lock (36 active Linux x86_64 distributions,
the inactive Win32-only `colorama`, the Darwin Torch row, and the virtual root), public GGUF/companion identities,
official source identity and the gated upstream's seven-file contract are all
bound into the approval scope. The gated upstream license and payload hashes
remain explicit unresolved review blockers; no identity is guessed and the
production gate therefore exits 2 until authenticated owner evidence exists.
The reference dumper also has a dependency-free `--self-test` for fixed token,
source-identity, manifest-safety and source-inventory invariants; it never
imports Torch/Transformers or creates model data.

After the VAST CPU gate passes, transfer model/reference data directly to a
disposable Apple Silicon host and run:

```sh
VOKRA_REMOTE_APPLE_SILICON=1 \
scripts/verify/apple-silicon-neutts-air.sh \
  --gguf /remote/stage/neutts-air.gguf \
  --gguf-sha256 <PUBLIC_GGUF_SHA256> \
  --companion /remote/stage/distill-neucodec.gguf \
  --companion-sha256 <COMPANION_GGUF_SHA256> \
  --reference /remote/stage/reference \
  --reference-sha256 <REFERENCE_MANIFEST_SHA256> \
  --evidence-dir /remote/evidence/neutts-air-metal
```

The VAST evidence includes the same complete command in
`evidence/apple-verifier-command.txt`. The Apple worker refuses the
maintainer-machine class, performs no network or publication action, checks
the three supplied hashes before opening the reference, validates the exact
schema/repository/revision/runtime contract and every artifact hash (including
the source inventory), then records CPU/Metal logits, greedy IDs and
composition results against the same reference.
