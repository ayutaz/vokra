# AST official parity oracle

This isolated Python 3.12 uv project generates a small, independent reference
fixture from Hugging Face Transformers `4.45.2` and the immutable upstream AST
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

The acceptance bounds were registered before observing Vokra output:

- normalized frontend max absolute error `2e-5`;
- logits max absolute error `1e-2`;
- logits RMSE `2e-3`;
- logits cosine similarity at least `0.99999`;
- exact top-5 class-index ordering.

Do not widen a failed bound without locating and documenting the numerical
cause. The public GGUF and upstream checkpoint remain on VAST; only the
roughly 515 KiB official fixture is committed.
