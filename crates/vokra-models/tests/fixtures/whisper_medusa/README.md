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
- environment: Python 3.12.14, PyTorch distribution 2.13.0
  (runtime 2.13.0+cu130), Transformers 5.10.4, x86_64 VAST CPU
- reference device: CPU; CUDA was not used for fixture generation
- input: one second of deterministic, low-amplitude 220/440/880 Hz tones at
  16 kHz; it has no dataset or recording licence dependency

VAST evidence for the current fixture:

- native CPU parity: `max_abs=5.722045898e-5` at vocabulary index `33684`,
  within the pre-existing `5e-4` bound
- greedy token: `50257` (EOT), exact
- Apple route: `aarch64-apple-darwin` with the Metal feature cross-compiled
  successfully
- converted GGUF: `6245932960` bytes,
  SHA-256 `1e7dc41c545853aba1b56c5375ce7a6fc88e8720c4eafd296a93d5eeec983fa3`
- publication: NO_UPLOAD; no upload was performed

The pinned upstream `utils/__init__.py` eagerly imports training, metrics, and
`wandb` code even though its `requirements.txt` does not declare `wandb`.
The dumper therefore exposes the exact upstream `utils/` directory as a
package without executing that initializer.  It still imports the exact
upstream `models/model.py` and `utils/config_and_args.py`; no model equation is
reimplemented in the dumper.

Run from the repository root on VAST because the checkpoint totals 6.25 GB.
Reuse the committed lock; do not regenerate it as part of this recipe:

```text
uv sync --frozen --python 3.12 --project tools/parity/whisper_medusa
uv run --frozen --python 3.12 --project tools/parity/whisper_medusa python \
  tools/parity/whisper_medusa/dump_reference.py \
  --model-dir /path/to/pinned-hf-snapshot \
  --source-parent /path/to/pinned-upstream-repository \
  --output-dir /tmp/whisper-medusa-reference \
  --max-new-tokens 8 --device cpu
```

Run the Rust consumer on VAST (the checkpoint is 6.25 GB) with
`VOKRA_WHISPER_MEDUSA_GGUF=/path/to/model.gguf cargo test --release -p
vokra-models --test parity_whisper_medusa_real -- --nocapture`.  The FP32
logits gate is `max_abs <= 5e-4`; the measured VAST result above is within the
pre-existing bound, and the greedy token matched exactly (`50257`, EOT).  The
bound was selected before the measurement and was not relaxed.

SHA-256:

```text
7dc06b4f5de6b5803df950f7aa79997992806875dc00b2028245d41e20398d19  manifest.json
36eb8143d598ded217fd9e235fa26292cfe40a570a52c6eda9e8edeadf11aced  pcm.f32
9a01033601202c8330c67d866505e5a44d190cd2b6497d2cf89aa1ac417ff4e8  prefix_logits.f32
a1d3acb03d768e3e6a5defac18d83c2732a8afc11854828a24a697802e927573  greedy_tokens.u32
```
