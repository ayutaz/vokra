# AST official parity oracle

This isolated Python 3.12 uv project generates a small, independent reference
fixture from Hugging Face Transformers `5.5.0` and the immutable upstream AST
revision `f826b80d28226b62986cc218e5cec390b1096902`. The dumper verifies the
installed official AST source files and upstream safetensors by SHA-256. It
does not import Vokra or reproduce the Rust equations.

Run the dependency install, model download, oracle, and Rust real-weight
consumer on VAST:

```sh
uv sync --python 3.12 --project tools/parity/ast --locked
uv run --python 3.12 --project tools/parity/ast \
  tools/parity/ast/dump_reference.py \
  --audio tests/fixtures/audio/jfk-30s.wav \
  --output tests/fixtures/ast

VOKRA_AST_GGUF=/root/models/ast.gguf \
  CARGO_BUILD_JOBS=1 cargo test --release -p vokra-models \
  --test parity_ast_real -- --nocapture
```

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
