# AST official parity oracle

This isolated Python 3.12 uv project generates a small, independent reference
fixture from Hugging Face Transformers `5.10.4` and the immutable upstream AST
revision `f826b80d28226b62986cc218e5cec390b1096902`. The dumper verifies the
installed official AST source files and upstream safetensors by SHA-256. It
does not import Vokra or reproduce the Rust equations.

The dependency project is deliberately exact: `torch==2.13.0`,
`torchaudio==2.11.0`, and `transformers==5.10.4`. The AST feature/modeling
source pins in `dump_reference.py` were calculated from the official
`transformers-5.10.4-py3-none-any.whl` named by `uv.lock` (wheel SHA-256
`8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e`), before
any model execution. The committed fixture still records the earlier
Transformers 5.5.0 oracle; it is not silently restamped by a dependency-only
change.

Run the dependency install, model download, oracle, and Rust real-weight
consumer through the VAST worker (the dumper-only portion is shown here for
inspection):

```sh
uv sync --python 3.12 --project tools/parity/ast --locked
uv run --python 3.12 --project tools/parity/ast \
  tools/parity/ast/dump_reference.py \
  --audio tests/fixtures/audio/jfk-30s.wav \
  --output /tmp/vokra-ast-candidate-reference

scripts/publish/vast-ai/run-ast-security-parity.sh
```

The command above is a VAST-only procedure. The checked-in worker
`scripts/publish/vast-ai/run-ast-security-parity.sh` additionally requires a
clean checkout, Linux identity, the exact public GGUF and upstream checkpoint
revisions/hashes, and records the locked dependency versions before running
the reference and native CPU parity. It has no upload/publish path; collect
the evidence and destroy the instance after review. Do not run it on the
maintainer Mac. A successful worker summary uses
`numeric_verdict=PASS_PREEXISTING_BOUNDS_CANDIDATE`: the candidate passed the
existing Rust frontend/logit bounds, but this does not replace the required
fixture review or Apple CPU/Metal validation. Use `--self-test` for its
model-free contract check.

The logit acceptance bounds were registered before observing Vokra output:

- logits max absolute error `1e-2`;
- logits RMSE `2e-3`;
- logits cosine similarity at least `0.99999`;
- exact top-5 class-index ordering.

The frontend initially used a pre-registered max-only bound of `2e-5`. The
first real run stopped at `3.23415e-4`. Investigation located the maximum in a
near-f32-floor high-frequency mel bin; more importantly, the Transformers
5.5.0 TorchAudio float32 frontend differs from the independent NumPy float64
Kaldi-equation cross-check by max `3.33128e-4` (RMSE `5.99753e-6`). The
evidence-backed frontend gate is therefore max `5e-4`, RMSE `1e-5`, and p99
`2e-5`. This preserves strict distribution checks instead of hiding broad
drift behind a max-only tolerance.

Do not widen a failed bound without locating and documenting the numerical
cause. The public GGUF and upstream checkpoint remain on VAST; only the
roughly 515 KiB official fixture is committed.
