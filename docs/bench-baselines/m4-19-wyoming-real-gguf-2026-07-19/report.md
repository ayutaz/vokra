# M4-19 — Wyoming real-GGUF round trip, recorded (2026-07-19)

**cc-34** from the M4-residual audit. `m4_19_asr_real_gguf_round_trip_gated` is
the flip-the-switch leg that drives the Wyoming TCP server with a real Whisper
base + real piper-plus voice instead of the mock service. The audit found it had
passed twice but was never recorded, so no artifact backed the claim. This is
that record, re-run on this HEAD.

## Result

`cargo test --release --manifest-path integrations/vokra-server/Cargo.toml
--test wyoming_compat -- --nocapture m4_19` at `53d216c`, with:

```sh
VOKRA_WHISPER_BASE_GGUF=~/.cache/vokra-eval/gguf/whisper-base.gguf
VOKRA_PIPER_GGUF=~/.cache/vokra-eval/gguf/piper-plus-css10-ja-6lang-neutralspk.gguf
```

**5 passed, 0 failed, 0 SKIP** (`grep -c SKIP` = 0). Test time was 0.98-1.88 s
across three runs after a 4 m 58 s cold release build; the spread is sibling
load on the machine, not the leg:

| test | result |
|---|---|
| `m4_19_asr_real_gguf_round_trip_gated` | **ok — ran, real service** |
| `m4_19_asr_golden_transcript_round_trip_over_tcp` | ok |
| `m4_19_barge_in_stops_tts_stream_over_tcp` | ok |
| `m4_19_faster_whisper_behavioral_parity_shape` | ok |
| `m4_19_scheduler_overload_is_explicit_error_over_tcp` | ok |

The gated leg logged `vokra-server: wyoming serving full ASR+TTS (multi-session
scheduler wired)` — i.e. a real `InferenceService` was built and registered,
not the discovery-only path.

## The `-neutralspk` requirement (the point of this record)

**The multi-speaker voice does not work and must not be substituted.** Swapping
in `piper-plus-css10-ja-6lang.gguf` (same voice, multi-speaker export):

```
service build failed: model load failed for `piper-plus` at
".../piper-plus-css10-ja-6lang.gguf": invalid argument:
piper voice GGUF missing tensor `spk_proj.0.weight`
```

That is a correct FR-EX-08 hard error from the loader — the multi-speaker
loader path is simply not implemented (audit note 3, recommended to ride along
with the piper loader work in cc-21). The usable file is
`piper-plus-css10-ja-6lang-**neutralspk**.gguf`.

## Control legs — why the "ok" above is not vacuous

A gated test that returns early still reports `ok`, so passing alone proves
nothing. Both controls were run:

| leg | env | observed (after the §Finding fix) |
|---|---|---|
| A | both GGUFs set (`-neutralspk`) | **ran** — no SKIP line, service wired, transcript frame asserted, 1.70 s |
| B | no env vars | `SKIP — set VOKRA_WHISPER_BASE_GGUF + VOKRA_PIPER_GGUF ...`, 0.00 s |
| C | multi-speaker voice | **FAILED**, 0.38 s — `missing tensor 'spk_proj.0.weight'` (was a green skip before the fix) |

Leg B confirms the gate is live (so leg A's `ok` means work happened); leg C
pins the `-neutralspk` requirement with the exact error.

## Finding — a wrong GGUF degraded to a green skip (fixed)

Leg C mattered beyond the voice requirement. The test's own error arm was:

```rust
Err(e) => { eprintln!("...SKIP — service build failed: {e}"); return; }
```

so pointing the leg at a **wrong or broken** GGUF was absorbed into a skip and
the suite still reported `ok`. The loader behaved correctly (loud, specific
error); the harness then discarded it. An owner running this flip-the-switch
leg with the wrong voice saw green and would reasonably conclude the round trip
had been exercised — the exact failure mode this record exists to prevent.

That is a silent fallback in harness code (FR-EX-08), and the same class as the
CosyVoice2 `llm = None` bind recorded in
`docs/bench-baselines/eval-cache-artifacts-2026-07-19/` §2: a container problem
surviving as a successful-looking outcome. **Fixed here**: supplying both env
vars is the request to run the leg, so a build failure past that point now
panics, and the panic message names the `-neutralspk` requirement inline so the
failure is self-explanatory. The genuinely-unset case (leg B) is still a skip,
so CI — which has no GGUF — is unaffected.

The skip contract did change, which is normally an owner call; it is taken here
because the previous contract could only ever convert a real defect into a
pass, and leg C above is the measured proof that it did.

## Scope / honesty notes

- The gated leg feeds **1 second of silence** and asserts the *shape* of the
  reply (`type: "transcript"`, `data.text` a string). It is a real model over a
  real socket, but it is **not** a transcript-accuracy or WER measurement, and
  the text may legitimately be empty. Real-audio WER over Wyoming remains the
  owner task in `docs/handoff/m4-19.md` §"Owner tasks" item 3.
- The 38 s figure in the audit is not reproduced here because it bundled the
  build; the measured test time on a warm build is **1.88 s** for all five
  `m4_19_*` tests.
- One sibling carries the same pattern and was **not** touched here, to avoid
  colliding with the audit item that owns it:
  `integrations/vokra-server/tests/openai_compat.rs`
  (`verbose_json_word_timestamps_e2e_with_real_gguf`, labelled cc-19) skips when
  a **configured** GGUF path does not exist. It is milder — the message says
  "Not a pass." — but the harness still reports `ok`, so the same
  configured-yet-unexercised hole exists there.
- Environment: Apple M1 iMac, macOS arm64, release profile, loopback TCP.
