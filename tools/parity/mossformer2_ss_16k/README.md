# MossFormer2-SS-16K validation

This directory holds the pinned 48-package Python 3.12 reference environment
and the offline, stdlib-only `preflight_gate.py`. The VAST worker runs that gate
before host checks, token checks, scratch/cache creation, synchronization,
source/model acquisition, checkpoint preparation, CUDA, or Cargo.

The validation identities are fixed to `vokra/mossformer2-ss-16k` revision
`0e9ba9258cead4252f8e5279598af296ada08bf7` (223,058,240 bytes), the upstream
checkpoint `alibabasglab/MossFormer2_SS_16K` revision
`407cb030cd66340918ebb6c8cc63b18f8592cdbe` (670,353,271 bytes), and
ClearerVoice-Studio revision `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`.
The audited public GGUF manifest identity is 1,076 tensors and 55,735,666
parameters. Numeric status remains `MEASURED_NOT_GATED` with bounds `UNSET`;
no tolerance or parity claim is added here.

After the exact source checkout, the worker invokes `preflight_gate.py
--verify-source`. That stdlib-only mode checks the pinned Git revision and the
six code/license files as regular non-symlink files, streaming their fixed
byte/SHA identities once those identities are resolved. The same no-cache,
stdlib-only `--validate-reference` path is run before native CPU measurement;
it checks the CUDA reference schema, locked NumPy/Torch versions, every fixed
shape/byte count, and every artifact hash.

The Apple verifier consumes only the VAST reference directory and requires
caller-supplied GGUF/reference-manifest hashes before parsing or Cargo. It
accepts no upload or download path. Production remains blocked until
authenticated primary evidence fills the upstream/source license and exact
source-file bytes/SHA rows in `license_gate_manifest.json`.
