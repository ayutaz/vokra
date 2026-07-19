# M4-05 T19 — CSM streaming TTFA / RTF reference measurement (fixture floor)

**What this is (and is not).** The M4-05 exit criterion is "CSM streaming
動作" — the WP carries **no** native RTF completion number, so nothing here
is hard-asserted (M4-05 spec T19). These are the *synthesized tiny-fixture*
floor numbers from `csm::streaming::tests::ttfa_rtf_reference_measurement_
tiny_fixture` — the honest analog of the M3-15 in-process FakeSynth floor:
they pin the harness, not the real model. Real-model / real-device numbers
are the T29/T30 owner track.

- Host: Apple M1 iMac (CLAUDE.md dev environment), CPU backend.
- Model: `CsmConfig::tiny_for_tests` synthesized fixture
  (backbone 2×16d / depth 2×8d / 4 codebooks; 16 kHz, 8-sample frame hop),
  greedy sampling, 8 frames.
- Measured 2026-07-15, single run each (reference, not a variance study):

| build   | TTFA      | wall (8 frames) | RTF    |
|---------|-----------|-----------------|--------|
| debug   | 0.707 ms  | 4.860 ms        | 1.2149 |
| release | 0.095 ms  | 0.282 ms        | 0.0705 |

Interpretation notes:

- the tiny fixture's audio second is tiny (64 samples @ 16 kHz per 8-frame
  run), so RTF here measures per-frame fixed cost, not model FLOPs;
- the frame loop is allocation-free (pinned by
  `tests/csm_hot_path_alloc.rs`) — the release-mode per-frame cost is
  dominated by the small GEMMs;
- real CSM-1B (16×2048 backbone + 32 codebooks @ 12.5 Hz) numbers land
  with the T29 weights; the owner records them here alongside these floor
  values.
