# Whisper-Medusa v1 official reference

These bounded fixtures were generated from the upstream
`whisper_medusa.models.WhisperMedusaModel`, not from a mirror of the Rust
equations.  They cover official module 0: the residual SiLU adapter feeding
the shared Whisper vocabulary projection.

Pinned inputs:

- checkpoint: `aiola/whisper-medusa-v1` at
  `6ea7c2f47658cfc7f9c8d1c158a9fbdb33458462`
- upstream source: `aiola-lab/whisper-medusa` at
  `19819c37ab15db6e68826e406614a2c86fbb946e`
- environment: Python 3.12.14, PyTorch 2.2.2+cu121,
  Transformers 4.49.0, NVIDIA GeForce GTX 1070 Ti
- input: one second of deterministic, low-amplitude 220/440/880 Hz tones at
  16 kHz; it has no dataset or recording licence dependency

The pinned upstream `utils/__init__.py` eagerly imports training, metrics, and
`wandb` code even though its `requirements.txt` does not declare `wandb`.
The dumper therefore exposes the exact upstream `utils/` directory as a
package without executing that initializer.  It still imports the exact
upstream `models/model.py` and `utils/config_and_args.py`; no model equation is
reimplemented in the dumper.

Regenerate on VAST because the checkpoint totals 6.25 GB:

```text
uv lock --directory tools/parity/whisper_medusa
uv sync --frozen --directory tools/parity/whisper_medusa
uv run --frozen --directory tools/parity/whisper_medusa python \
  tools/parity/whisper_medusa/dump_reference.py \
  --model-dir /path/to/pinned-hf-snapshot \
  --source-parent /path/to/pinned-upstream-parent \
  --output-dir /tmp/whisper-medusa-reference \
  --max-new-tokens 8 --device cuda
```

Run the Rust consumer with
`VOKRA_WHISPER_MEDUSA_GGUF=/path/to/model.gguf cargo test --release -p
vokra-models --test parity_whisper_medusa_real -- --nocapture`.  The FP32
logits gate is `max_abs <= 5e-4`; the measured VAST result was
`1.182556152e-4` at vocabulary index 14525, and the greedy token matched
exactly (`50257`, EOT).  The bound was selected before the measurement and was
not relaxed.

SHA-256:

```text
a1d3acb03d768e3e6a5defac18d83c2732a8afc11854828a24a697802e927573  greedy_tokens.u32
f96031445b3c848b15c5c13027b0b014055db0cdaa0d7283e770d0bc9e51191e  manifest.json
08e3b320f36969a972bb3b7edba3c53a2a64fb7f6f9579699c46be05d711c3e3  pcm.f32
d15ae4b67f4c7e0c166bca932145974c1d82ceeab7758e14468e65e24810f51d  prefix_logits.f32
```
