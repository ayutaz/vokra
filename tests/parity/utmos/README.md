# UTMOS parity fixtures (M4-18 T09 — flip-the-switch, weights deferred)

**No fixture is committed yet, deliberately.** The M4-18 kickoff gate deferred
the UTMOS weights + license (owner sourcing still open), and committing an
`expected_score` without running the real upstream reference would fabricate
the very number the gate exists to verify (NFR-QL-04). This directory ships
only the format contract below; the harness
(`crates/vokra-eval/tests/parity_utmos.rs`) skips cleanly until the switch is
flipped.

## Fixture format

`score.json` (single object; every field required):

```json
{
  "clip": "ref-clip.wav",
  "sample_rate": 16000,
  "expected_score": 0.0,
  "atol": 0.0,
  "provenance": "sarulab-speech/UTMOS22 @ <commit>, generated YYYY-MM-DD, reproduction band <x> × <1.5-2>"
}
```

- `clip` — mono WAV filename in this directory (PCM16 or float32, readable by
  `vokra_eval::wav::read_wav`). Its header rate must equal `sample_rate`.
- `sample_rate` — must also equal the GGUF's `vokra.utmos.sample_rate`
  (the scorer rejects mismatches loudly; no silent resample, FR-EX-08).
- `expected_score` — the score the pinned upstream implementation produced
  for `clip`, recorded verbatim.
- `atol` — **honest tolerance**: measure the upstream reference's own
  reproduction error band (re-run variance / platform delta), then set
  `atol = band × 1.5–2`. Never a constant chosen to make CI green
  (memory `feedback-honest-parity-atol`; Kokoro `PROSODY_F0_ATOL` precedent —
  if an architectural bound forces a wider value, record the derivation here
  and in the rustdoc/ADR).
- `provenance` — upstream repo + commit, generation date, and the measured
  band the atol was derived from. The harness rejects an empty value.

## Owner flip recipe

1. Complete M4-18 T02 (weight URL + `docs/license-audit.md` sign-off — the
   fixture must not be generated from weights that failed sign-off).
2. Run the pinned upstream SaruLab UTMOS22 on a chosen clip offline; write
   `score.json` + the clip here (plain blobs, no git-lfs — see
   `.gitattributes` conventions for `tests/parity/`).
3. Convert the checkpoint: `vokra-convert --model utmos …` (T05, deferred —
   lands with the weights; the GGUF schema it must emit is pinned by ADR
   `M4-18-utmos-arch` §(c)/(d) and machine-verified by the round-trip test in
   `crates/vokra-eval/src/metrics/utmos.rs`).
4. `VOKRA_UTMOS_GGUF=path/to/utmos.gguf cargo test -p vokra-eval --test
   parity_utmos` — the gated test stops skipping automatically. CI: run the
   `parity-utmos` workflow via workflow_dispatch.

The harness refuses synthesized (seed-random) weights on the parity path;
only a GGUF converted from the real checkpoint is comparable.
