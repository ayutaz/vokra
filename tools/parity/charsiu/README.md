# Charsiu real-checkpoint parity

`charsiu_dump_reference.py` loads the pinned `charsiu/en_w2v2_fc_10ms`
checkpoint through the official `transformers.Wav2Vec2ForCTC` implementation.
The committed 400-sample PCM and 42-logit fixture is the independent oracle;
the real checkpoint, generated GGUF, and regeneration evidence are produced
by `scripts/publish/vast-ai/run-charsiu-validation.sh` on VAST.

The Apple worker consumes only a VAST-produced packet. It verifies the GGUF
SHA-256, reference manifest and evidence identities, binds the exact clean
checkout, and runs one explicitly selected ignored real-weight test on Darwin
arm64. The test selects CPU and Metal explicitly and reports all three required
comparisons:

- CPU/reference uses the registered Charsiu FP32 bound `0.0002`.
- Metal/reference and Metal/CPU use the repository GPU FP32 bound `0.01`
  documented in `.agents/skills/numerical-parity/SKILL.md` under “GPU backend
  parity” (the bound is not calibrated from this Apple run).

An unavailable Metal device or an uncovered hot operation is a failure. The
Charsiu forward dispatches convolution, grouped convolution, GroupNorm, GEMM,
LayerNorm, GELU, and softmax through one selected `Compute`; no operation may
silently execute on CPU for a Metal run. Apple evidence is stamped
`publication=NO_UPLOAD`. No numbers in this README are an Apple measurement;
the Apple result is established only by the remote worker log and its evidence
packet.
