# XCodec2 historical packet admission — 2026-10-15 JST

Private source commit `a0a9ec3053187e3d7fff4db8767f0118403247b3` extends
published Draft #152 head `eaf0e5fa9f7b6498782f36e4df9a60665caa5f2d`.
It changes only the [Rust consumer](../../crates/vokra-models/tests/parity_xcodec2_real.rs).
This candidate is not pushed or accepted, and its Rust tests have not run.
The earlier [direct-guard correction](xcodec2-exact-byte-guard-binding-2026-10-15.md)
has separate hosted evidence: job `116038029251` actually executes all three
stdlib suites, 24 + 7 + 6 = 37 passing tests. That proof does not cover this
new Rust diff, and recorded VAST `4263f416` is not reused for it.

## Required correction

The existing consumer reads code/PCM data but ignores the external manifest,
fixture digests and raw GGUF digest. The strict model binder authenticates
internal tensor names/shapes; it does not authenticate the complete external
reference packet or raw artifact bytes.

The prepared correction authenticates the exact historical manifest, all
three declared payload files, metadata types/shapes, finite float values,
regular-file paths and symlink ancestry. Before a real decoder run, it reads
the exact-size GGUF from one opened handle, verifies a streaming SHA-256 and
passes that same owned byte buffer to `GgufFile::parse`. It does not hash one
path and then load a separately reopened file. Portable pathname replacement
atomicity is not claimed; the parsed bytes themselves are authenticated.

Historical identities remain unchanged:

- Manifest SHA-256: `cfe60f214df9229b0f519d0548f0ce14cabe810df680d8eddb5d5e80420eb2ba`.
- GGUF: 3,291,064,672 bytes,
  `7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e`.
- Oracle: official `xcodec2==0.1.5`, Torch `2.5.0+cu124`, torchtune `0.3.1`,
  vector-quantize-pytorch `1.17.8`; 117 official state tensors.
- Existing bounds: max absolute `2e-4`, RMSE `2e-5`, cosine `0.999999`.

The numerical comparison and diagnostic messages are preserved. No fixture,
model, license, provenance, bound, dependency declaration or owner decision is
restamped. This historical oracle is not a new Torch 2.13/TorchAudio 2.11 run.

## Review and verification boundary

Luna implements; manager reviews the full diff and independently checks the
four existing fixture hashes, manifest size, 64-hex digest literals and
unchanged bounds. Static UV/Python 3.12 assertions, rustfmt, diff hygiene and
all five normal pre-commit checks pass. None compiles or executes Rust tests.

The prepared model-free tests include independent SHA vectors and chunk
boundaries, pristine per-case file mutations, malformed/duplicate metadata,
real NaN/infinity rejection, missing/extra/nonregular files, genuine symlink
and ancestor cases, and explicit invalid/non-UTF8 backend refusal. Review
corrects finite data mistakenly labeled NaN, self-consistency-only hash tests,
contaminated negative cases and pre-existing-directory deletion before the
candidate is committed. These are source-reviewed tests, not runtime passes.

Fresh remote source verification remains required. Its proposed narrow scope
clears the artifact environment and runs static checks plus this test target,
recording the explicit artifact-unset skip rather than calling it real parity.
A new source-only VAST lease was requested separately; the earlier one-off
Candle capture does not authorize a retry or a model run.

Real GGUF acquisition/decode, current dependency/native/API/owner gates,
independent real-weight CPU results, final clean HEAD and fresh Apple packets
remain unfinished. No model work, paid worker, Scaleway or publication occurred.
The [full plan](next-steps-2026-10-14.md), 58 unresolved public rows and frozen
63-row owner scope remain intact.
